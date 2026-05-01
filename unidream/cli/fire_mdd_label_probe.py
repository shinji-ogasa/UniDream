from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from unidream.cli.ac_fire_dd_guard_probe import _maxdd_interval, _pnl_bar
from unidream.cli.ac_fire_timing_probe import ProbeRun, _load_actor_for_run, _parse_run
from unidream.cli.fire_control_label_probe import (
    _average_precision,
    _build_feature_matrix,
    _chronological_split,
    _format_float,
    _ridge_predict,
    _roc_auc,
    _rolling_past_vol,
    _summary_metrics,
    _window_pnl,
)
from unidream.experiments.policy_fire import predict_with_policy_flags
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class MDDProbeConfig:
    horizons: tuple[int, ...]
    primary_horizon: int
    fire_eps: float
    train_frac: float
    ridge_l2: float
    max_z_dim: int
    rel_vol_window: int
    mdd_rel_threshold: float
    post_dd_quantile: float


def _local_drawdown_magnitude(pnl: np.ndarray) -> np.ndarray:
    if len(pnl) == 0:
        return np.zeros(0, dtype=np.float64)
    equity = np.exp(np.cumsum(np.asarray(pnl, dtype=np.float64)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return np.maximum(0.0, -drawdown)


def _fire_run_end_indices(fire: np.ndarray) -> np.ndarray:
    mask = np.asarray(fire, dtype=bool)
    out = np.arange(len(mask), dtype=np.int64)
    end = len(mask)
    for i in range(len(mask) - 1, -1, -1):
        if mask[i]:
            end = end if i + 1 < len(mask) and mask[i + 1] else i + 1
            out[i] = end
        else:
            out[i] = i
    return out


def _global_mdd_mask(returns: np.ndarray, positions: np.ndarray, costs_cfg: dict) -> tuple[np.ndarray, dict]:
    t = min(len(returns), len(positions))
    pnl = np.zeros(t, dtype=np.float64)
    prev_pos = 0.0
    for i in range(t):
        pnl[i] = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        prev_pos = float(positions[i])
    equity = np.exp(np.cumsum(pnl))
    interval = _maxdd_interval(equity)
    mask = np.zeros(t, dtype=bool)
    if interval["trough"] >= interval["peak"]:
        mask[interval["peak"] : interval["trough"] + 1] = True
    return mask, interval


def _build_mdd_labels(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    fire: np.ndarray,
    costs_cfg: dict,
    cfg: MDDProbeConfig,
) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    t = min(len(returns), len(positions), len(no_adapter), len(fire))
    run_end = _fire_run_end_indices(fire[:t])
    roll_vol = _rolling_past_vol(returns[:t], cfg.rel_vol_window)
    global_mask, global_interval = _global_mdd_mask(returns[:t], positions[:t], costs_cfg)
    out: dict[int, dict[str, np.ndarray]] = {}

    for horizon in cfg.horizons:
        valid = np.zeros(t, dtype=bool)
        fire_advantage = np.full(t, np.nan, dtype=np.float64)
        local_mdd = np.full(t, np.nan, dtype=np.float64)
        no_adapter_mdd = np.full(t, np.nan, dtype=np.float64)
        mdd_contribution = np.full(t, np.nan, dtype=np.float64)
        post_fire_dd_contribution = np.full(t, np.nan, dtype=np.float64)
        future_mdd_overlap = np.full(t, np.nan, dtype=np.float64)
        pre_dd_state = np.full(t, np.nan, dtype=np.float64)
        global_mdd_overlap = np.full(t, np.nan, dtype=np.float64)
        local_peak = np.full(t, np.nan, dtype=np.float64)
        local_trough = np.full(t, np.nan, dtype=np.float64)
        mdd_rel = np.full(t, np.nan, dtype=np.float64)

        for i in range(0, max(0, t - int(horizon) + 1)):
            pnl_on = _window_pnl(returns, positions, i, horizon, costs_cfg)
            pnl_off = _window_pnl(returns, no_adapter, i, horizon, costs_cfg)
            if len(pnl_on) < int(horizon) or len(pnl_off) < int(horizon):
                continue
            valid[i] = True
            equity_on = np.exp(np.cumsum(pnl_on))
            interval = _maxdd_interval(equity_on)
            dd_on = _local_drawdown_magnitude(pnl_on)
            dd_off = _local_drawdown_magnitude(pnl_off)
            mdd_on = float(np.max(dd_on)) if len(dd_on) else 0.0
            mdd_off = float(np.max(dd_off)) if len(dd_off) else 0.0
            denom = max(float(roll_vol[i]) * math.sqrt(float(horizon)), 1e-8)
            threshold = float(cfg.mdd_rel_threshold) * denom
            peak = int(interval["peak"])
            trough = int(interval["trough"])
            run_stop = max(0, min(int(horizon) - 1, int(run_end[i] - i - 1)))
            overlap = mdd_on > threshold and peak <= run_stop and trough >= 0
            pre_dd = mdd_on > threshold and peak > run_stop and peak <= max(run_stop + 1, int(horizon) // 2)
            dd_diff = dd_on[: min(len(dd_on), len(dd_off))] - dd_off[: min(len(dd_on), len(dd_off))]

            fire_advantage[i] = float(np.sum(pnl_on) - np.sum(pnl_off))
            local_mdd[i] = mdd_on
            no_adapter_mdd[i] = mdd_off
            mdd_contribution[i] = mdd_on - mdd_off
            post_fire_dd_contribution[i] = float(max(0.0, np.max(dd_diff))) if len(dd_diff) else 0.0
            future_mdd_overlap[i] = 1.0 if overlap else 0.0
            pre_dd_state[i] = 1.0 if pre_dd else 0.0
            global_mdd_overlap[i] = 1.0 if bool(global_mask[i]) else 0.0
            local_peak[i] = float(peak)
            local_trough[i] = float(trough)
            mdd_rel[i] = mdd_on / denom

        out[int(horizon)] = {
            "valid": valid,
            "fire_advantage": fire_advantage,
            "local_mdd": local_mdd,
            "no_adapter_mdd": no_adapter_mdd,
            "mdd_contribution": mdd_contribution,
            "post_fire_dd_contribution": post_fire_dd_contribution,
            "future_mdd_overlap": future_mdd_overlap,
            "pre_dd_state": pre_dd_state,
            "global_mdd_overlap": global_mdd_overlap,
            "local_peak": local_peak,
            "local_trough": local_trough,
            "mdd_rel": mdd_rel,
        }
    return out, {"mask": global_mask, "interval": global_interval}


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def _take_fraction(eval_idx: np.ndarray, score: np.ndarray, frac: float, *, high: bool) -> np.ndarray:
    if len(eval_idx) == 0:
        return eval_idx
    n = max(1, int(math.ceil(len(eval_idx) * float(frac))))
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    if high:
        order = order[::-1]
    return np.asarray(eval_idx, dtype=int)[order[:n]]


def _zscore(train_score: np.ndarray, eval_score: np.ndarray) -> np.ndarray:
    mean = float(np.mean(train_score))
    std = float(np.std(train_score))
    if std < 1e-8:
        std = 1.0
    return (np.asarray(eval_score, dtype=np.float64) - mean) / std


def _binary_eval(y_eval: np.ndarray, score_eval: np.ndarray) -> dict:
    y = np.asarray(y_eval, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
            "auc": float("nan"),
            "pr_auc": float("nan"),
        }
    top10 = _take_fraction(np.arange(len(y)), score_eval, 0.10, high=True)
    return {
        "positive_rate": float(np.mean(y)),
        "auc": _roc_auc(y, score_eval),
        "pr_auc": _average_precision(y, score_eval),
        "top10_positive_rate": float(np.mean(y[top10])) if len(top10) else float("nan"),
    }


def _selection_stats(idx: np.ndarray, labels: dict[str, np.ndarray]) -> dict:
    selected = np.asarray(idx, dtype=int)
    if len(selected) == 0:
        return {"count": 0}
    return {
        "count": int(len(selected)),
        "fire_advantage": _safe_mean(labels["fire_advantage"][selected]),
        "post_fire_dd_contribution": _safe_mean(labels["post_fire_dd_contribution"][selected]),
        "mdd_contribution": _safe_mean(labels["mdd_contribution"][selected]),
        "local_mdd": _safe_mean(labels["local_mdd"][selected]),
        "mdd_rel": _safe_mean(labels["mdd_rel"][selected]),
        "future_mdd_overlap_rate": float(np.mean(labels["future_mdd_overlap"][selected] > 0.5)),
        "pre_dd_state_rate": float(np.mean(labels["pre_dd_state"][selected] > 0.5)),
        "global_mdd_overlap_rate": float(np.mean(labels["global_mdd_overlap"][selected] > 0.5)),
        "nonpositive_mdd_contribution_rate": float(np.mean(labels["mdd_contribution"][selected] <= 0.0)),
    }


def _evaluate_horizon(
    *,
    x: np.ndarray,
    labels: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    cfg: MDDProbeConfig,
) -> dict:
    valid = sample_mask.copy()
    for key in (
        "fire_advantage",
        "post_fire_dd_contribution",
        "mdd_contribution",
        "future_mdd_overlap",
        "pre_dd_state",
        "global_mdd_overlap",
    ):
        valid &= np.isfinite(labels[key])
    train_idx, eval_idx = _chronological_split(valid, cfg.train_frac)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return {"samples": int(np.sum(valid)), "eval_samples": 0}

    train_scores: dict[str, np.ndarray] = {}
    eval_scores: dict[str, np.ndarray] = {}
    targets = {
        "adv": labels["fire_advantage"],
        "post_dd": labels["post_fire_dd_contribution"],
        "mdd_contrib": labels["mdd_contribution"],
        "future_overlap": labels["future_mdd_overlap"],
        "pre_dd": labels["pre_dd_state"],
        "global_overlap": labels["global_mdd_overlap"],
    }
    for name, target in targets.items():
        train_score, eval_score = _ridge_predict(x[train_idx], target[train_idx], x[eval_idx], l2=cfg.ridge_l2)
        train_scores[name] = train_score
        eval_scores[name] = eval_score

    post_threshold = float(np.quantile(labels["post_fire_dd_contribution"][train_idx], cfg.post_dd_quantile))
    binary = {
        "future_mdd_overlap": _binary_eval(labels["future_mdd_overlap"][eval_idx], eval_scores["future_overlap"]),
        "pre_dd_state": _binary_eval(labels["pre_dd_state"][eval_idx], eval_scores["pre_dd"]),
        "global_mdd_overlap": _binary_eval(labels["global_mdd_overlap"][eval_idx], eval_scores["global_overlap"]),
        "post_dd_q": {
            "threshold": post_threshold,
            **_binary_eval(labels["post_fire_dd_contribution"][eval_idx] > post_threshold, eval_scores["post_dd"]),
        },
    }

    adv_top10 = _take_fraction(eval_idx, eval_scores["adv"], 0.10, high=True)
    adv_top20 = _take_fraction(eval_idx, eval_scores["adv"], 0.20, high=True)
    low_postdd10 = _take_fraction(eval_idx, eval_scores["post_dd"], 0.10, high=False)
    high_postdd10 = _take_fraction(eval_idx, eval_scores["post_dd"], 0.10, high=True)
    low_overlap10 = _take_fraction(eval_idx, eval_scores["future_overlap"], 0.10, high=False)

    z_adv = _zscore(train_scores["adv"], eval_scores["adv"])
    z_post = _zscore(train_scores["post_dd"], eval_scores["post_dd"])
    z_overlap = _zscore(train_scores["future_overlap"], eval_scores["future_overlap"])
    z_pre = _zscore(train_scores["pre_dd"], eval_scores["pre_dd"])
    z_global = _zscore(train_scores["global_overlap"], eval_scores["global_overlap"])
    combined_scores = {
        "A_adv_only": z_adv,
        "B_adv_minus_postdd": z_adv - z_post,
        "C_adv_minus_overlap_predd": z_adv - z_overlap - z_pre,
        "D_adv_minus_all_mdd": z_adv - z_post - z_overlap - z_pre - z_global,
    }
    combined = {}
    for name, score in combined_scores.items():
        top10 = _take_fraction(eval_idx, score, 0.10, high=True)
        top20 = _take_fraction(eval_idx, score, 0.20, high=True)
        combined[name] = {
            "top10": _selection_stats(top10, labels),
            "top20": _selection_stats(top20, labels),
        }

    corr = float("nan")
    adv_eval = labels["fire_advantage"][eval_idx]
    if np.std(eval_scores["adv"]) > 1e-12 and np.std(adv_eval) > 1e-12:
        corr = float(np.corrcoef(eval_scores["adv"], adv_eval)[0, 1])

    return {
        "samples": int(np.sum(valid)),
        "train_samples": int(len(train_idx)),
        "eval_samples": int(len(eval_idx)),
        "base_rates": {
            "future_mdd_overlap": float(np.mean(labels["future_mdd_overlap"][eval_idx] > 0.5)),
            "pre_dd_state": float(np.mean(labels["pre_dd_state"][eval_idx] > 0.5)),
            "global_mdd_overlap": float(np.mean(labels["global_mdd_overlap"][eval_idx] > 0.5)),
            "post_dd_q": float(np.mean(labels["post_fire_dd_contribution"][eval_idx] > post_threshold)),
        },
        "binary": binary,
        "advantage": {
            "corr": corr,
            "top10": _selection_stats(adv_top10, labels),
            "top20": _selection_stats(adv_top20, labels),
        },
        "risk_rankings": {
            "low_postdd_top10": _selection_stats(low_postdd10, labels),
            "high_postdd_top10": _selection_stats(high_postdd10, labels),
            "low_future_overlap_top10": _selection_stats(low_overlap10, labels),
        },
        "combined_scores": combined,
    }


def _evaluate_run_fold(
    *,
    run: ProbeRun,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    probe_cfg: MDDProbeConfig,
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
    fire = np.abs(positions - no_adapter) > float(probe_cfg.fire_eps)
    x = _build_feature_matrix(
        enc=enc,
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        fire=fire,
        regime=regime,
        advantage=advantage,
        costs_cfg=costs_cfg,
        max_z_dim=probe_cfg.max_z_dim,
    )
    labels, global_mdd = _build_mdd_labels(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        fire=fire,
        costs_cfg=costs_cfg,
        cfg=probe_cfg,
    )
    horizons = {}
    for horizon, rec in labels.items():
        horizons[str(horizon)] = _evaluate_horizon(
            x=x,
            labels=rec,
            sample_mask=rec["valid"] & fire,
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
            "peak": int(global_mdd["interval"]["peak"]),
            "trough": int(global_mdd["interval"]["trough"]),
            "maxdd": float(global_mdd["interval"]["maxdd"]),
            "fire_in_global_mdd_rate": float(np.mean(fire & global_mdd["mask"])),
        },
        "horizons": horizons,
    }


def _readiness(records: list[dict], primary_horizon: int) -> dict:
    adv_ok = []
    postdd_low_ok = []
    combined_ok = []
    overlap_auc_ok = []
    for rec in records:
        metrics = rec["horizons"].get(str(primary_horizon))
        if not metrics or metrics.get("eval_samples", 0) == 0:
            continue
        adv_top = metrics["advantage"]["top10"]
        low_post = metrics["risk_rankings"]["low_postdd_top10"]
        combined = metrics["combined_scores"]["D_adv_minus_all_mdd"]["top10"]
        adv_ok.append(float(adv_top["fire_advantage"]) > 0.0)
        postdd_low_ok.append(
            float(low_post["fire_advantage"]) > 0.0 and float(low_post["post_fire_dd_contribution"]) <= float(adv_top["post_fire_dd_contribution"])
        )
        combined_ok.append(
            float(combined["fire_advantage"]) > 0.0
            and float(combined["post_fire_dd_contribution"]) <= float(adv_top["post_fire_dd_contribution"])
            and float(combined["global_mdd_overlap_rate"]) <= float(adv_top["global_mdd_overlap_rate"])
        )
        overlap_auc_ok.append(float(metrics["binary"]["future_mdd_overlap"].get("auc", float("nan"))) >= 0.55)
    return {
        "adv_top10_positive_all": bool(adv_ok) and all(adv_ok),
        "low_postdd_keeps_positive_adv_all": bool(postdd_low_ok) and all(postdd_low_ok),
        "combined_D_improves_mdd_risk_all": bool(combined_ok) and all(combined_ok),
        "future_mdd_overlap_auc_ge_0_55_all": bool(overlap_auc_ok) and all(overlap_auc_ok),
    }


def _write_markdown(path: str, *, records: list[dict], args: argparse.Namespace, cfg: MDDProbeConfig) -> None:
    lines = [
        "# Plan15-B MDD Fire Label Probe",
        "",
        "## Setup",
        "",
        f"- config: `{args.config}`",
        f"- folds: `{args.folds}`",
        f"- horizons: `{','.join(str(h) for h in cfg.horizons)}`",
        f"- primary horizon: `{cfg.primary_horizon}`",
        "- scope: MDD-window label quality only; no guard, WM head, or AC unlock.",
        "",
        "## Policy Summary",
        "",
        "| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl | global MDD | fire in MDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        s = rec["summary"]
        g = rec["global_mdd"]
        lines.append(
            f"| {rec['label']} | {rec['fold']} | {_format_float(s['alpha_excess_pt'], 2, True)} | "
            f"{_format_float(s['sharpe_delta'], 3, True)} | {_format_float(s['maxdd_delta_pt'], 2, True)} | "
            f"{_format_float(s['turnover'], 2)} | {s['long']:.1%} | {s['short']:.1%} | {s['flat']:.1%} | "
            f"{s['fire_rate']:.1%}/{s['fire_count']} | {_format_float(s['fire_pnl'], 4, True)} | "
            f"{_format_float(100.0 * g['maxdd'], 2, True)}pt | {g['fire_in_global_mdd_rate']:.1%} |"
        )

    lines += [
        "",
        "## MDD Label Predictability",
        "",
        "| run | fold | h | samples | eval | future MDD AUC | future MDD top10 | pre-DD AUC | global MDD AUC | postDD q AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            b = metrics["binary"]
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {horizon} | {metrics['samples']} | {metrics['eval_samples']} | "
                f"{_format_float(b['future_mdd_overlap']['auc'])} | {_format_float(b['future_mdd_overlap'].get('top10_positive_rate', float('nan')))} | "
                f"{_format_float(b['pre_dd_state']['auc'])} | {_format_float(b['global_mdd_overlap']['auc'])} | "
                f"{_format_float(b['post_dd_q']['auc'])} |"
            )

    lines += [
        "",
        "## Advantage vs MDD Risk Ranking",
        "",
        "| run | fold | h | adv top10 | adv postDD | adv futureMDD | adv globalMDD | low postDD adv | low postDD | low overlap adv | low overlap futureMDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            adv = metrics["advantage"]["top10"]
            low_post = metrics["risk_rankings"]["low_postdd_top10"]
            low_overlap = metrics["risk_rankings"]["low_future_overlap_top10"]
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {horizon} | {_format_float(adv['fire_advantage'], 5, True)} | "
                f"{_format_float(adv['post_fire_dd_contribution'], 5, True)} | {_format_float(adv['future_mdd_overlap_rate'])} | "
                f"{_format_float(adv['global_mdd_overlap_rate'])} | {_format_float(low_post['fire_advantage'], 5, True)} | "
                f"{_format_float(low_post['post_fire_dd_contribution'], 5, True)} | {_format_float(low_overlap['fire_advantage'], 5, True)} | "
                f"{_format_float(low_overlap['future_mdd_overlap_rate'])} |"
            )

    lines += [
        "",
        "## Combined Score Top-Decile",
        "",
        "| run | fold | h | score | top10 adv | postDD | mdd contrib | futureMDD | preDD | globalMDD | top20 adv | top20 postDD |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            for name, score_metrics in metrics["combined_scores"].items():
                top10 = score_metrics["top10"]
                top20 = score_metrics["top20"]
                lines.append(
                    f"| {rec['label']} | {rec['fold']} | {horizon} | {name} | {_format_float(top10['fire_advantage'], 5, True)} | "
                    f"{_format_float(top10['post_fire_dd_contribution'], 5, True)} | {_format_float(top10['mdd_contribution'], 5, True)} | "
                    f"{_format_float(top10['future_mdd_overlap_rate'])} | {_format_float(top10['pre_dd_state_rate'])} | "
                    f"{_format_float(top10['global_mdd_overlap_rate'])} | {_format_float(top20['fire_advantage'], 5, True)} | "
                    f"{_format_float(top20['post_fire_dd_contribution'], 5, True)} |"
                )

    readiness = _readiness(records, cfg.primary_horizon)
    lines += [
        "",
        "## Readiness",
        "",
        "| criterion | pass |",
        "|---|---:|",
    ]
    for key, value in readiness.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- If MDD-window labels are not separable, Plan16 guard should not use them.",
        "- A useful risk score must lower post-fire DD contribution or MDD overlap without killing fire_advantage.",
        "- This probe is diagnostic only and does not alter `configs/trading.yaml`.",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.fire_mdd_label_probe")
    parser.add_argument("--config", default="configs/trading_wm_control_headonly.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="checkpoints/data_cache")
    parser.add_argument("--run", action="append", required=True, help="label=checkpoint_dir[@ac_file][:ac|:bc]")
    parser.add_argument("--horizons", default="16,32")
    parser.add_argument("--primary-horizon", type=int, default=32)
    parser.add_argument("--fire-eps", type=float, default=1e-6)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--ridge-l2", type=float, default=1e-2)
    parser.add_argument("--max-z-dim", type=int, default=128)
    parser.add_argument("--rel-vol-window", type=int, default=64)
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
    probe_cfg = MDDProbeConfig(
        horizons=tuple(int(x) for x in args.horizons.split(",") if x.strip()),
        primary_horizon=int(args.primary_horizon),
        fire_eps=float(args.fire_eps),
        train_frac=float(args.train_frac),
        ridge_l2=float(args.ridge_l2),
        max_z_dim=int(args.max_z_dim),
        rel_vol_window=int(args.rel_vol_window),
        mdd_rel_threshold=float(args.mdd_rel_threshold),
        post_dd_quantile=float(args.post_dd_quantile),
    )
    runs = [_parse_run(spec) for spec in args.run]
    records: list[dict] = []
    for split in splits:
        for run in runs:
            try:
                print(f"[Plan15-B] run={run.label} fold={split.fold_idx} checkpoint={run.checkpoint_dir}")
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
                print(f"[Plan15-B] skip missing run={run.label} fold={split.fold_idx}")

    serializable = {
        "config": args.config,
        "folds": args.folds,
        "runs": [run.__dict__ for run in runs],
        "probe": probe_cfg.__dict__,
        "records": records,
        "readiness": _readiness(records, probe_cfg.primary_horizon),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, allow_nan=True)
    _write_markdown(args.output_md, records=records, args=args, cfg=probe_cfg)
    print(f"[Plan15-B] wrote {args.output_md}")


if __name__ == "__main__":
    main()
