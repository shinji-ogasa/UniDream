from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.cli.round1_meta_label_probe import _fmt, _nanmean, _nanmin
from unidream.cli.round2_selector_audit_probe import _nanmedian
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy


EXPERIMENT_NAME = "plan005_meta_guard_bc_ac_probe"


@dataclass(frozen=True)
class GuardRule:
    name: str
    signal_name: str
    low_position: float
    high_position: float
    hold_bars: int
    cooldown_bars: int


CORE_RULE_A = GuardRule(
    name="core_mom768_dd12288",
    signal_name="mom768_lt_-0p03_or_dd12288_lt_-0p18",
    low_position=0.50,
    high_position=1.00,
    hold_bars=256,
    cooldown_bars=64,
)
CORE_RULE_B = GuardRule(
    name="core_mom768_dd1536",
    signal_name="mom768_lt_-0p05_or_dd1536_lt_-0p18",
    low_position=0.25,
    high_position=1.00,
    hold_bars=64,
    cooldown_bars=64,
)
PRE_HALVING_REBOUND_RULE = GuardRule(
    name="pre_halving_rebound_dd3072",
    signal_name="dd3072_lt_-0p03",
    low_position=0.25,
    high_position=1.00,
    hold_bars=256,
    cooldown_bars=64,
)
DEEP_BEAR_RECOVERY_RULE = GuardRule(
    name="deep_bear_recovery_mom1536_dd6144",
    signal_name="mom1536_lt_0_or_dd6144_lt_-0p18",
    low_position=0.00,
    high_position=1.00,
    hold_bars=64,
    cooldown_bars=64,
)


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _shifted_log_price(returns: np.ndarray) -> np.ndarray:
    log_price = np.cumsum(np.asarray(returns, dtype=np.float64))
    if len(log_price) == 0:
        return log_price
    return np.concatenate([[log_price[0]], log_price[:-1]])


def _rolling_past_momentum(prev_log_price: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(prev_log_price, dtype=np.float64)
    w = int(max(window, 1))
    lag = np.empty_like(x)
    lag[:w] = x[0] if len(x) else 0.0
    lag[w:] = x[:-w]
    return x - lag


def _rolling_past_drawdown(prev_log_price: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(prev_log_price, dtype=np.float64)
    w = int(max(window, 1))
    out = np.zeros(len(x), dtype=np.float64)
    dq: deque[int] = deque()
    for i, value in enumerate(x):
        while dq and dq[0] < i - w + 1:
            dq.popleft()
        while dq and x[dq[-1]] <= value:
            dq.pop()
        dq.append(i)
        out[i] = value - x[dq[0]]
    return out


def _build_past_features(returns: np.ndarray) -> dict[str, np.ndarray]:
    prev_log_price = _shifted_log_price(returns)
    out: dict[str, np.ndarray] = {}
    for window in (768, 1536, 3072, 6144, 12288, 24576):
        out[f"mom{window}"] = _rolling_past_momentum(prev_log_price, window)
    for window in (1536, 3072, 6144, 12288):
        out[f"dd{window}"] = _rolling_past_drawdown(prev_log_price, window)
    return out


def _signal_positions(signal: np.ndarray, rule: GuardRule) -> np.ndarray:
    sig = np.asarray(signal, dtype=bool)
    out_signal = np.zeros(len(sig), dtype=bool)
    hold_until = -1
    cooldown_until = -1
    hold = max(int(rule.hold_bars), 1)
    cooldown = max(int(rule.cooldown_bars), 0)
    for i, active in enumerate(sig):
        if i < hold_until:
            out_signal[i] = True
            continue
        if bool(active) and i >= cooldown_until:
            hold_until = i + hold
            cooldown_until = hold_until + cooldown
            out_signal[i] = True
    return np.where(out_signal, float(rule.low_position), float(rule.high_position)).astype(np.float64)


def _rule_signal(features: dict[str, np.ndarray], rule: GuardRule) -> np.ndarray:
    if rule.signal_name == CORE_RULE_A.signal_name:
        return (features["mom768"] < -0.03) | (features["dd12288"] < -0.18)
    if rule.signal_name == CORE_RULE_B.signal_name:
        return (features["mom768"] < -0.05) | (features["dd1536"] < -0.18)
    if rule.signal_name == PRE_HALVING_REBOUND_RULE.signal_name:
        return features["dd3072"] < -0.03
    if rule.signal_name == DEEP_BEAR_RECOVERY_RULE.signal_name:
        return (features["mom1536"] < 0.0) | (features["dd6144"] < -0.18)
    raise ValueError(f"unknown guard signal: {rule.signal_name}")


def _start_value(features: dict[str, np.ndarray], name: str, start_idx: int) -> float:
    if len(features.get(name, [])) == 0:
        return 0.0
    idx = int(min(max(start_idx, 0), len(features[name]) - 1))
    return float(features[name][idx])


def select_meta_guard_mode(features: dict[str, np.ndarray], start_idx: int) -> tuple[str, dict[str, float]]:
    """Select a guard family from information available before the evaluation segment.

    The feature arrays are shifted by one bar, so values at ``start_idx`` use
    returns up to ``start_idx - 1`` only.
    """
    diag = {
        "start_mom3072": _start_value(features, "mom3072", start_idx),
        "start_mom6144": _start_value(features, "mom6144", start_idx),
        "start_dd12288": _start_value(features, "dd12288", start_idx),
        "start_mom24576": _start_value(features, "mom24576", start_idx),
    }
    if diag["start_mom3072"] > 0.20 and diag["start_mom6144"] < -0.30:
        return "pre_halving_rebound", diag
    if diag["start_dd12288"] < -0.65 and diag["start_mom24576"] > 0.0:
        return "deep_bear_recovery", diag
    return "core_pair", diag


def guard_positions_from_features(
    features: dict[str, np.ndarray],
    *,
    mode: str,
) -> np.ndarray:
    if mode == "core_pair":
        a = _signal_positions(_rule_signal(features, CORE_RULE_A), CORE_RULE_A)
        b = _signal_positions(_rule_signal(features, CORE_RULE_B), CORE_RULE_B)
        return 0.5 * (a + b)
    if mode == "pre_halving_rebound":
        return _signal_positions(_rule_signal(features, PRE_HALVING_REBOUND_RULE), PRE_HALVING_REBOUND_RULE)
    if mode == "deep_bear_recovery":
        return _signal_positions(_rule_signal(features, DEEP_BEAR_RECOVERY_RULE), DEEP_BEAR_RECOVERY_RULE)
    raise ValueError(f"unknown meta guard mode: {mode}")


def apply_meta_guard(
    *,
    full_returns: np.ndarray,
    base_positions: np.ndarray,
    segment_mask: np.ndarray,
    min_position: float,
    max_position: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = _build_past_features(full_returns)
    mask = np.asarray(segment_mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("segment_mask is empty")
    start_idx = int(indices[0])
    mode, mode_diag = select_meta_guard_mode(features, start_idx)
    guard_full = guard_positions_from_features(features, mode=mode)
    guard = guard_full[mask][: len(base_positions)]
    base = np.asarray(base_positions, dtype=np.float64)[: len(guard)]
    positions = np.clip(np.minimum(base, guard), float(min_position), float(max_position))
    diag = {
        "mode": mode,
        "guard_mean": float(np.mean(guard)) if len(guard) else 0.0,
        "guard_underweight_rate": float(np.mean(guard < 0.999)) if len(guard) else 0.0,
        **mode_diag,
    }
    return positions, diag


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        group_rows = [r for r in rows if r["group"] == group]
        metrics = [r["stress"]["cost_x1"] for r in group_rows]
        alphas = [float(m["alpha_excess_pt"]) for m in metrics]
        dds = [float(m["maxdd_delta_pt"]) for m in metrics]
        turns = [float(m["turnover"]) for m in metrics]
        out[group] = {
            "folds": len(group_rows),
            "pass_alpha_ge3_dd_le_neg3": int(sum(a >= 3.0 and d <= -3.0 for a, d in zip(alphas, dds))),
            "pass_alpha_ge10_dd_le_neg5": int(sum(a >= 10.0 and d <= -5.0 for a, d in zip(alphas, dds))),
            "alpha_mean": _nanmean(alphas),
            "alpha_median": _nanmedian(alphas),
            "alpha_worst": _nanmin(alphas),
            "maxdd_worst": float(np.nanmax(dds)) if dds else float("nan"),
            "turnover_mean": _nanmean(turns),
            "turnover_max": float(np.nanmax(turns)) if turns else float("nan"),
        }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan005 Meta-Guard BC/AC Probe",
        "",
        "Plan005 keeps Plan004 fold-local base policies, then applies a deterministic past-only guard.",
        "Guard mode is selected from pre-test regime features shifted by one bar.",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        "",
        "## Aggregate",
        "",
        "| group | folds | pass +3/-3 | pass +10/-5 | Alpha mean | Alpha median | Alpha worst | MaxDD worst | turnover mean | turnover max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row["folds"]),
                    str(row["pass_alpha_ge3_dd_le_neg3"]),
                    str(row["pass_alpha_ge10_dd_le_neg5"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                    _fmt(row["turnover_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Fold Detail: cost_x1", ""])
    lines.append("| fold | group | mode | AlphaEx | MaxDDDelta | turnover | guard_mean | start regime |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        d = row.get("diag", {})
        start_regime = (
            f"mom3072={_fmt(d.get('start_mom3072'))}, "
            f"mom6144={_fmt(d.get('start_mom6144'))}, "
            f"dd12288={_fmt(d.get('start_dd12288'))}, "
            f"mom24576={_fmt(d.get('start_mom24576'))}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    str(d.get("mode", "")),
                    _fmt(m["alpha_excess_pt"], 6),
                    _fmt(m["maxdd_delta_pt"], 6),
                    _fmt(m["turnover"]),
                    _fmt(d.get("guard_mean")),
                    start_regime,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leak Discipline",
            "",
            "- Plan004 base policy is produced fold-locally: train fit, validation extraction, test reporting.",
            "- Plan005 guard features are shifted by one bar and use only returns observed before each bar.",
            "- The meta mode for a test segment uses only shifted features at the test start.",
            "- The 0-12 rule constants are development-set tuned; this report does not claim fold13/out-of-sample validity.",
        ]
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _load_base_cache(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan005_meta_guard_bc_ac_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--selection-stress-mode", choices=("primary", "include_costx3"), default="primary")
    parser.add_argument("--base-positions-cache", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-positions", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    date = _date_prefix()
    if not args.output_json:
        args.output_json = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.json")
    if not args.output_md:
        args.output_md = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.md")
    if not args.output_positions:
        args.output_positions = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}_positions.npz")

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
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
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, selected_folds = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    min_position = float(cfg.get("ac", {}).get("abs_min_position", 0.0))
    max_position = float(cfg.get("ac", {}).get("abs_max_position", 1.25))
    full_returns = np.asarray(raw_returns, dtype=np.float64)

    base_cache = _load_base_cache(args.base_positions_cache)
    rows: list[dict[str, Any]] = []
    final_positions: list[np.ndarray] = []
    base_positions_all: list[np.ndarray] = []
    test_masks: list[np.ndarray] = []
    selected_base: list[str] = []
    for cache_idx, split in enumerate(splits):
        fid = int(split.fold_idx)
        print(f"[Plan005] fold={fid} Plan004 base + meta guard")
        ds = WFODataset(features_df, raw_returns, split, seq_len=data_cfg.get("seq_len", 64))
        if base_cache is not None:
            base_pos = np.asarray(base_cache["positions"][cache_idx], dtype=np.float64)[: len(ds.test_returns)]
            base_selected = str(base_cache["selected"][cache_idx]) if "selected" in base_cache else "cache"
            segment_mask = np.asarray(base_cache["test_masks"][cache_idx], dtype=bool)
        else:
            base_rec = run_plan004_fold_policy(
                ds=ds,
                cfg=cfg,
                costs_cfg=costs_cfg,
                fold_idx=fid,
                seed=args.seed,
                ridge_l2=args.ridge_l2,
                max_train_samples=args.max_train_samples,
                selection_stress_mode=args.selection_stress_mode,
            )
            base_pos = np.asarray(base_rec["positions"], dtype=np.float64)[: len(ds.test_returns)]
            base_selected = f"{base_rec['selected_row'].get('source')}:{base_rec['selected_row'].get('spec')}"
            segment_mask = (features_df.index >= split.test_start) & (features_df.index <= split.test_end)
            segment_mask = np.asarray(segment_mask, dtype=bool)
        positions, diag = apply_meta_guard(
            full_returns=full_returns,
            base_positions=base_pos,
            segment_mask=segment_mask,
            min_position=min_position,
            max_position=max_position,
        )
        base_stress = _stress_metrics(
            returns=ds.test_returns,
            positions=base_pos,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        stress = _stress_metrics(
            returns=ds.test_returns,
            positions=positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan004_base",
                "source": base_selected,
                "stress": base_stress,
                "diag": {"mode": "plan004_base"},
            }
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan005_meta_guard",
                "source": base_selected,
                "stress": stress,
                "diag": diag,
            }
        )
        final_positions.append(positions.astype(np.float32))
        base_positions_all.append(base_pos.astype(np.float32))
        test_masks.append(segment_mask)
        selected_base.append(base_selected)

    payload = {
        "experiment": EXPERIMENT_NAME,
        "seed": args.seed,
        "folds": selected_folds,
        "config": args.config,
        "selection_stress_mode": args.selection_stress_mode,
        "base_positions_cache": args.base_positions_cache,
        "rows": rows,
        "aggregate": _aggregate(rows),
        "target": {
            "alpha_excess_pt_min": 3.0,
            "maxdd_delta_pt_max": -3.0,
            "scope": "folds 0-12 development set",
        },
        "leak_discipline": {
            "base": "Plan004 fold-local train/validation selection; test is report-only",
            "guard": "shifted rolling return features; mode selected at segment start from pre-segment information",
            "development_note": "0-12 constants are tuned on the development folds and should be re-tested before claiming new out-of-sample performance",
        },
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    os.makedirs(os.path.dirname(args.output_positions) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_positions,
        positions=np.asarray(final_positions, dtype=object),
        base_positions=np.asarray(base_positions_all, dtype=object),
        test_masks=np.asarray(test_masks, dtype=bool),
        selected=np.asarray(selected_base, dtype=object),
        folds=np.asarray(selected_folds, dtype=np.int64),
        index=np.asarray([str(x) for x in features_df.index], dtype=object),
    )
    print(f"[Plan005] wrote {args.output_json}")
    print(f"[Plan005] wrote {args.output_md}")
    print(f"[Plan005] wrote {args.output_positions}")


if __name__ == "__main__":
    main()
