from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from unidream.cli.ac_fire_dd_guard_probe import _pnl_bar
from unidream.cli.ac_fire_timing_probe import ProbeRun, _load_actor_for_run, _parse_run
from unidream.cli.fire_control_label_probe import (
    _format_float,
    _online_state,
    _rolling_past_sum,
    _rolling_past_vol,
    _summary_metrics,
)
from unidream.cli.fire_mdd_label_probe import MDDProbeConfig, _build_mdd_labels, _global_mdd_mask
from unidream.experiments.policy_fire import predict_with_policy_flags
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class FireTypeProbeConfig:
    horizon: int
    fire_eps: float
    max_z_dim: int
    rel_vol_window: int
    min_type_count: int
    mdd_rel_threshold: float
    post_dd_quantile: float


def _linear_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * (arr - float(np.mean(arr)))) / denom)


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i - int(window))
        out[i] = _linear_slope(arr[start:i])
    return out


def _bar_pnl_series(returns: np.ndarray, positions: np.ndarray, costs_cfg: dict) -> np.ndarray:
    t = min(len(returns), len(positions))
    out = np.zeros(t, dtype=np.float64)
    prev_pos = 0.0
    for i in range(t):
        out[i] = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        prev_pos = float(positions[i])
    return out


def _mdd_proximity(index: np.ndarray, interval: dict) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(index, dtype=np.int64)
    peak = int(interval.get("peak", 0))
    trough = int(interval.get("trough", peak))
    phase = np.full(len(idx), "after_mdd", dtype=object)
    phase[idx < peak] = "pre_mdd"
    phase[(idx >= peak) & (idx <= trough)] = "inside_mdd"
    dist = np.minimum(np.abs(idx - peak), np.abs(idx - trough)).astype(np.float64)
    if trough > peak:
        dist = dist / float(trough - peak + 1)
    return phase, dist


def _regime_labels(regime: np.ndarray | None, n: int) -> np.ndarray:
    if regime is None:
        return np.full(n, "regime_unknown", dtype=object)
    arr = np.asarray(regime)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.full(n, "regime_unknown", dtype=object)
    idx = np.argmax(arr[:n], axis=1)
    return np.asarray([f"regime_{int(x)}" for x in idx], dtype=object)


def _build_fire_frame(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    regime: np.ndarray | None,
    costs_cfg: dict,
    cfg: FireTypeProbeConfig,
) -> dict[str, np.ndarray | dict]:
    t = min(len(returns), len(positions), len(no_adapter))
    returns = np.asarray(returns[:t], dtype=np.float64)
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    delta = positions - no_adapter
    fire = np.abs(delta) > float(cfg.fire_eps)

    mdd_cfg = MDDProbeConfig(
        horizons=(int(cfg.horizon),),
        primary_horizon=int(cfg.horizon),
        fire_eps=float(cfg.fire_eps),
        train_frac=0.6,
        ridge_l2=1e-2,
        max_z_dim=int(cfg.max_z_dim),
        rel_vol_window=int(cfg.rel_vol_window),
        mdd_rel_threshold=float(cfg.mdd_rel_threshold),
        post_dd_quantile=float(cfg.post_dd_quantile),
    )
    labels_by_h, global_mdd = _build_mdd_labels(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        fire=fire,
        costs_cfg=costs_cfg,
        cfg=mdd_cfg,
    )
    labels = labels_by_h[int(cfg.horizon)]
    valid = np.asarray(labels["valid"], dtype=bool) & fire
    idx = np.flatnonzero(valid)

    current_dd, underwater, equity_before, peak_before = _online_state(returns, positions, costs_cfg)
    policy_pnl = _bar_pnl_series(returns, positions, costs_cfg)
    benchmark_pnl = _bar_pnl_series(returns, np.ones_like(positions), costs_cfg)
    rel_pnl = policy_pnl - benchmark_pnl
    equity_slope_32 = _rolling_slope(np.cumsum(policy_pnl), 32)
    rel_equity_slope_32 = _rolling_slope(np.cumsum(rel_pnl), 32)
    trailing_return_16 = _rolling_past_sum(returns, 16)
    trailing_return_32 = _rolling_past_sum(returns, 32)
    trailing_slope_16 = _rolling_slope(np.cumsum(returns), 16)
    trailing_slope_32 = _rolling_slope(np.cumsum(returns), 32)
    trailing_vol_64 = _rolling_past_vol(returns, 64)
    regime_label = _regime_labels(regime, t)
    mdd_phase, mdd_distance = _mdd_proximity(np.arange(t, dtype=np.int64), global_mdd["interval"])

    fire_frame: dict[str, np.ndarray | dict] = {
        "idx": idx,
        "position": positions[idx],
        "no_adapter": no_adapter[idx],
        "delta": delta[idx],
        "fire_pnl": policy_pnl[idx],
        "current_drawdown_depth": np.maximum(0.0, -current_dd[idx]),
        "underwater_duration": underwater[idx],
        "equity_before": equity_before[idx],
        "peak_before": peak_before[idx],
        "trailing_return_16": trailing_return_16[idx],
        "trailing_return_32": trailing_return_32[idx],
        "trailing_slope_16": trailing_slope_16[idx],
        "trailing_slope_32": trailing_slope_32[idx],
        "trailing_vol_64": trailing_vol_64[idx],
        "equity_slope_32": equity_slope_32[idx],
        "benchmark_relative_equity_slope_32": rel_equity_slope_32[idx],
        "fire_advantage_h": labels["fire_advantage"][idx],
        "post_fire_dd_contribution": labels["post_fire_dd_contribution"][idx],
        "mdd_contribution": labels["mdd_contribution"][idx],
        "future_mdd_overlap": labels["future_mdd_overlap"][idx],
        "pre_dd_state": labels["pre_dd_state"][idx],
        "global_mdd_overlap": labels["global_mdd_overlap"][idx],
        "mdd_rel": labels["mdd_rel"][idx],
        "mdd_phase": mdd_phase[idx],
        "mdd_distance": mdd_distance[idx],
        "regime": regime_label[idx],
        "global_mdd": global_mdd["interval"],
        "all_fire_count": np.asarray([int(np.sum(fire))], dtype=np.int64),
    }
    fire_frame["fire_type"] = _assign_fire_types(fire_frame)
    return fire_frame


def _quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return default
    return float(np.quantile(arr, q))


def _assign_fire_types(frame: dict[str, np.ndarray | dict]) -> np.ndarray:
    n = len(np.asarray(frame["idx"]))
    out = np.full(n, "noise_fire", dtype=object)
    if n == 0:
        return out

    adv = np.asarray(frame["fire_advantage_h"], dtype=np.float64)
    post_dd = np.asarray(frame["post_fire_dd_contribution"], dtype=np.float64)
    future_overlap = np.asarray(frame["future_mdd_overlap"], dtype=np.float64) > 0.5
    global_overlap = np.asarray(frame["global_mdd_overlap"], dtype=np.float64) > 0.5
    pre_dd = np.asarray(frame["pre_dd_state"], dtype=np.float64) > 0.5
    dd_depth = np.asarray(frame["current_drawdown_depth"], dtype=np.float64)
    underwater = np.asarray(frame["underwater_duration"], dtype=np.float64)
    trend32 = np.asarray(frame["trailing_slope_32"], dtype=np.float64)
    ret16 = np.asarray(frame["trailing_return_16"], dtype=np.float64)
    rel_equity = np.asarray(frame["benchmark_relative_equity_slope_32"], dtype=np.float64)

    adv_pos = adv > 0.0
    post_dd_low = post_dd <= _quantile(post_dd, 0.60)
    dd_deep = dd_depth >= _quantile(dd_depth, 0.60)
    long_underwater = underwater >= _quantile(underwater, 0.60)
    trend_pos = (trend32 > 0.0) & (ret16 > 0.0)
    rel_pos = rel_equity >= 0.0

    dangerous = future_overlap | pre_dd
    out[dangerous] = "pre_dd_danger_fire"
    mdd_profitable = global_overlap & adv_pos
    out[mdd_profitable] = "mdd_inside_profitable_fire"
    recovery = (~dangerous) & dd_deep & long_underwater & adv_pos & post_dd_low
    out[recovery] = "recovery_fire"
    trend = (~dangerous) & (~recovery) & trend_pos & rel_pos & adv_pos & post_dd_low
    out[trend] = "trend_continuation_fire"
    low_dd_profitable = (~dangerous) & (~recovery) & (~trend) & adv_pos & post_dd_low
    out[low_dd_profitable] = "profitable_low_dd_fire"
    return out


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def _group_stats(frame: dict[str, np.ndarray | dict], mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            "count": 0,
            "rate": 0.0,
            "fire_advantage_mean": float("nan"),
            "fire_advantage_median": float("nan"),
            "fire_advantage_positive_rate": float("nan"),
            "post_fire_dd_contribution": float("nan"),
            "mdd_contribution": float("nan"),
            "future_mdd_overlap_rate": float("nan"),
            "pre_dd_state_rate": float("nan"),
            "global_mdd_overlap_rate": float("nan"),
            "fire_pnl": float("nan"),
            "current_drawdown_depth": float("nan"),
            "underwater_duration": float("nan"),
            "trailing_return_32": float("nan"),
            "trailing_vol_64": float("nan"),
            "delta": float("nan"),
        }
    total = max(1, len(np.asarray(frame["idx"])))
    adv = np.asarray(frame["fire_advantage_h"], dtype=np.float64)[mask]
    return {
        "count": int(np.sum(mask)),
        "rate": float(np.sum(mask) / total),
        "fire_advantage_mean": _safe_mean(adv),
        "fire_advantage_median": _safe_median(adv),
        "fire_advantage_positive_rate": float(np.mean(adv > 0.0)),
        "post_fire_dd_contribution": _safe_mean(np.asarray(frame["post_fire_dd_contribution"], dtype=np.float64)[mask]),
        "mdd_contribution": _safe_mean(np.asarray(frame["mdd_contribution"], dtype=np.float64)[mask]),
        "future_mdd_overlap_rate": float(np.mean(np.asarray(frame["future_mdd_overlap"], dtype=np.float64)[mask] > 0.5)),
        "pre_dd_state_rate": float(np.mean(np.asarray(frame["pre_dd_state"], dtype=np.float64)[mask] > 0.5)),
        "global_mdd_overlap_rate": float(np.mean(np.asarray(frame["global_mdd_overlap"], dtype=np.float64)[mask] > 0.5)),
        "fire_pnl": float(np.sum(np.asarray(frame["fire_pnl"], dtype=np.float64)[mask])),
        "current_drawdown_depth": _safe_mean(np.asarray(frame["current_drawdown_depth"], dtype=np.float64)[mask]),
        "underwater_duration": _safe_mean(np.asarray(frame["underwater_duration"], dtype=np.float64)[mask]),
        "trailing_return_32": _safe_mean(np.asarray(frame["trailing_return_32"], dtype=np.float64)[mask]),
        "trailing_vol_64": _safe_mean(np.asarray(frame["trailing_vol_64"], dtype=np.float64)[mask]),
        "delta": _safe_mean(np.asarray(frame["delta"], dtype=np.float64)[mask]),
    }


def _summarize_frame(frame: dict[str, np.ndarray | dict]) -> dict:
    fire_type = np.asarray(frame["fire_type"], dtype=object)
    regime = np.asarray(frame["regime"], dtype=object)
    mdd_phase = np.asarray(frame["mdd_phase"], dtype=object)
    type_stats = {str(name): _group_stats(frame, fire_type == name) for name in sorted(set(fire_type.tolist()))}
    regime_stats = {str(name): _group_stats(frame, regime == name) for name in sorted(set(regime.tolist()))}
    phase_stats = {str(name): _group_stats(frame, mdd_phase == name) for name in sorted(set(mdd_phase.tolist()))}
    type_regime_stats: dict[str, dict] = {}
    for t_name in sorted(set(fire_type.tolist())):
        for r_name in sorted(set(regime.tolist())):
            key = f"{t_name}|{r_name}"
            type_regime_stats[key] = _group_stats(frame, (fire_type == t_name) & (regime == r_name))
    return {
        "type_stats": type_stats,
        "regime_stats": regime_stats,
        "mdd_phase_stats": phase_stats,
        "type_regime_stats": type_regime_stats,
    }


def _evaluate_run_fold(
    *,
    run: ProbeRun,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    probe_cfg: FireTypeProbeConfig,
    device: str,
) -> dict:
    payload = _load_actor_for_run(
        run=run,
        split=split,
        features_df=features_df,
        raw_returns=raw_returns,
        cfg=cfg,
        device=device,
    )
    actor = payload["actor"]
    enc = payload["enc_test"]
    returns = np.asarray(payload["test_returns"], dtype=np.float64)
    regime = payload["test_regime_probs"]
    advantage = payload["test_advantage_values"]
    costs_cfg = payload["costs_cfg"]
    benchmark_position = float(payload["benchmark_position"])
    positions = actor.predict_positions(
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
    )
    no_adapter = predict_with_policy_flags(
        actor,
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
        use_floor=bool(getattr(actor, "use_benchmark_exposure_floor", False)),
        use_adapter=False,
    )
    t = min(len(returns), len(positions), len(no_adapter))
    returns = returns[:t]
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    frame = _build_fire_frame(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        regime=regime,
        costs_cfg=costs_cfg,
        cfg=probe_cfg,
    )
    return {
        "label": run.label,
        "mode": "ac" if run.use_ac else "bc",
        "checkpoint_dir": run.checkpoint_dir,
        "fold": int(split.fold_idx),
        "summary": _summary_metrics(
            positions=positions,
            no_adapter=no_adapter,
            returns=returns,
            cfg=payload["cfg"],
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
            fire_eps=probe_cfg.fire_eps,
        ),
        "global_mdd": {
            "peak": int(frame["global_mdd"]["peak"]),
            "trough": int(frame["global_mdd"]["trough"]),
            "maxdd": float(frame["global_mdd"]["maxdd"]),
        },
        "fire_features": {
            "count": int(len(np.asarray(frame["idx"]))),
            "horizon": int(probe_cfg.horizon),
        },
        "clusters": _summarize_frame(frame),
    }


def _readiness(records: list[dict], min_type_count: int) -> dict:
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        for name, stats in rec["clusters"]["type_stats"].items():
            by_type.setdefault(name, []).append(stats)
    allowed_types = {"recovery_fire", "trend_continuation_fire", "profitable_low_dd_fire"}
    candidates = {}
    for name, stats_list in by_type.items():
        if len(stats_list) != len(records):
            continue
        enough = all(int(s["count"]) >= int(min_type_count) for s in stats_list)
        adv_pos = all(float(s["fire_advantage_mean"]) > 0.0 for s in stats_list)
        adv_positive_rate = all(float(s["fire_advantage_positive_rate"]) >= 0.60 for s in stats_list)
        low_postdd = all(float(s["post_fire_dd_contribution"]) <= 0.0005 for s in stats_list)
        low_future_mdd = all(float(s["future_mdd_overlap_rate"]) <= 0.35 for s in stats_list)
        low_global_mdd = all(float(s["global_mdd_overlap_rate"]) <= 0.35 for s in stats_list)
        allowed = name in allowed_types
        candidates[name] = {
            "enough_count_all_folds": enough,
            "positive_adv_all_folds": adv_pos,
            "adv_positive_rate_ge_60_all_folds": adv_positive_rate,
            "low_post_fire_dd_all_folds": low_postdd,
            "low_future_mdd_overlap_all_folds": low_future_mdd,
            "low_global_mdd_overlap_all_folds": low_global_mdd,
            "allowed_type": allowed,
            "plan16_candidate": bool(
                allowed and enough and adv_pos and adv_positive_rate and low_postdd and low_future_mdd and low_global_mdd
            ),
        }
    return candidates


def _write_markdown(path: str, *, records: list[dict], args: argparse.Namespace, cfg: FireTypeProbeConfig) -> None:
    lines = [
        "# Plan15-C Fire Type / Regime Split Probe",
        "",
        "## Setup",
        "",
        f"- config: `{args.config}`",
        f"- folds: `{args.folds}`",
        f"- horizon: `{cfg.horizon}`",
        "- scope: diagnostic fire type split only; no inference guard, WM head, AC unlock, or config adoption.",
        "",
        "## Policy Summary",
        "",
        "| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl | global MDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        s = rec["summary"]
        g = rec["global_mdd"]
        lines.append(
            f"| {rec['label']} | {rec['fold']} | {_format_float(s['alpha_excess_pt'], 2, True)} | "
            f"{_format_float(s['sharpe_delta'], 3, True)} | {_format_float(s['maxdd_delta_pt'], 2, True)} | "
            f"{_format_float(s['turnover'], 2)} | {s['long']:.1%} | {s['short']:.1%} | {s['flat']:.1%} | "
            f"{s['fire_rate']:.1%}/{s['fire_count']} | {_format_float(s['fire_pnl'], 4, True)} | "
            f"{_format_float(100.0 * g['maxdd'], 2, True)}pt |"
        )

    lines += [
        "",
        "## Fire Type Summary",
        "",
        "| run | fold | fire_type | count | rate | adv | adv+ | postDD | futureMDD | globalMDD | fire_pnl | dd_depth | trail_ret32 | vol64 | delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for name, s in rec["clusters"]["type_stats"].items():
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {name} | {s['count']} | {s['rate']:.1%} | "
                f"{_format_float(s['fire_advantage_mean'], 5, True)} | {_format_float(s['fire_advantage_positive_rate'])} | "
                f"{_format_float(s['post_fire_dd_contribution'], 5, True)} | {_format_float(s['future_mdd_overlap_rate'])} | "
                f"{_format_float(s['global_mdd_overlap_rate'])} | {_format_float(s['fire_pnl'], 4, True)} | "
                f"{_format_float(s['current_drawdown_depth'])} | {_format_float(s['trailing_return_32'], 5, True)} | "
                f"{_format_float(s['trailing_vol_64'], 5)} | {_format_float(s['delta'], 4, True)} |"
            )

    lines += [
        "",
        "## Regime Summary",
        "",
        "| run | fold | regime | count | adv | postDD | futureMDD | globalMDD | fire_pnl |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for name, s in rec["clusters"]["regime_stats"].items():
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {name} | {s['count']} | "
                f"{_format_float(s['fire_advantage_mean'], 5, True)} | {_format_float(s['post_fire_dd_contribution'], 5, True)} | "
                f"{_format_float(s['future_mdd_overlap_rate'])} | {_format_float(s['global_mdd_overlap_rate'])} | "
                f"{_format_float(s['fire_pnl'], 4, True)} |"
            )

    lines += [
        "",
        "## Type x Regime Summary",
        "",
        "| run | fold | type_regime | count | adv | postDD | futureMDD | globalMDD |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for name, s in rec["clusters"]["type_regime_stats"].items():
            if int(s["count"]) <= 0:
                continue
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {name} | {s['count']} | "
                f"{_format_float(s['fire_advantage_mean'], 5, True)} | {_format_float(s['post_fire_dd_contribution'], 5, True)} | "
                f"{_format_float(s['future_mdd_overlap_rate'])} | {_format_float(s['global_mdd_overlap_rate'])} |"
            )

    readiness = _readiness(records, cfg.min_type_count)
    lines += [
        "",
        "## Plan16 Readiness By Type",
        "",
        "| fire_type | allowed | enough count | adv > 0 all | adv+ >= 60% all | low postDD all | low futureMDD all | low globalMDD all | candidate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in readiness.items():
        lines.append(
            f"| {name} | {r['allowed_type']} | {r['enough_count_all_folds']} | {r['positive_adv_all_folds']} | "
            f"{r['adv_positive_rate_ge_60_all_folds']} | {r['low_post_fire_dd_all_folds']} | "
            f"{r['low_future_mdd_overlap_all_folds']} | {r['low_global_mdd_overlap_all_folds']} | "
            f"{r['plan16_candidate']} |"
        )

    selected = [name for name, r in readiness.items() if r["plan16_candidate"]]
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Plan16-ready types: `{', '.join(selected) if selected else 'none'}`",
        "- A type is Plan16-ready only if it has enough samples, positive fire_advantage, low post-fire DD contribution, low future-MDD overlap, and low global-MDD overlap in every evaluated fold.",
        "- If no type passes, do not add an inference guard yet; continue feature/type design.",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.fire_type_cluster_probe")
    parser.add_argument("--config", default="configs/trading_wm_control_headonly.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="checkpoints/data_cache")
    parser.add_argument("--run", action="append", required=True, help="label=checkpoint_dir[@ac_file][:ac|:bc]")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--fire-eps", type=float, default=1e-6)
    parser.add_argument("--max-z-dim", type=int, default=128)
    parser.add_argument("--rel-vol-window", type=int, default=64)
    parser.add_argument("--min-type-count", type=int, default=20)
    parser.add_argument("--mdd-rel-threshold", type=float, default=0.5)
    parser.add_argument("--post-dd-quantile", type=float, default=0.8)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=args.cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.folds))
    if not splits:
        raise RuntimeError(f"No folds matched: {args.folds}")
    probe_cfg = FireTypeProbeConfig(
        horizon=int(args.horizon),
        fire_eps=float(args.fire_eps),
        max_z_dim=int(args.max_z_dim),
        rel_vol_window=int(args.rel_vol_window),
        min_type_count=int(args.min_type_count),
        mdd_rel_threshold=float(args.mdd_rel_threshold),
        post_dd_quantile=float(args.post_dd_quantile),
    )
    runs = [_parse_run(spec) for spec in args.run]
    records: list[dict] = []
    for split in splits:
        for run in runs:
            try:
                print(f"[Plan15-C] run={run.label} fold={split.fold_idx} checkpoint={run.checkpoint_dir}")
                records.append(
                    _evaluate_run_fold(
                        run=run,
                        split=split,
                        features_df=features_df,
                        raw_returns=raw_returns,
                        cfg=cfg,
                        probe_cfg=probe_cfg,
                        device=args.device,
                    )
                )
            except FileNotFoundError:
                if not args.skip_missing:
                    raise
                print(f"[Plan15-C] skip missing run={run.label} fold={split.fold_idx}")

    serializable = {
        "config": args.config,
        "folds": args.folds,
        "runs": [run.__dict__ for run in runs],
        "probe": probe_cfg.__dict__,
        "records": records,
        "readiness": _readiness(records, probe_cfg.min_type_count),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, allow_nan=True)
    _write_markdown(args.output_md, records=records, args=args, cfg=probe_cfg)
    print(f"[Plan15-C] wrote {args.output_md}")


if __name__ == "__main__":
    main()
