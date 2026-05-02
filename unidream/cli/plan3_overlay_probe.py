"""
Plan 3: Standalone Overlay Verification Probe.

Refines the Plan 2 best candidate (D + triple-barrier guard + pullback eval-only)
through standalone reproducibility, ablation, and fold3-dependency analysis.

No AC, no route unlock, no configs/trading.yaml changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    SELECTOR_SPECS,
    SelectorSpec,
    _aggregate_selectors,
    _apply_event_throttle,
    _backtest_positions,
    _candidate_utilities,
    _fit_binary_model,
    _fit_ridge_multi,
    _json_sanitize,
    _path_max_drawdown,
    _positions_from_prediction,
    _pullback_no_fire_mask,
    _rolling_past_sum,
    _rolling_past_vol,
    _score_binary,
    _select_threshold,
    _selected_event_context_stats,
    _selected_utility_stats,
    _shift_for_execution,
    _state_features,
    _triple_barrier_labels,
    _unit_cost,
)
from unidream.data.dataset import WFODataset
from unidream.eval.pbo import compute_pbo
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


# ---------------------------------------------------------------------------
# Ablation selector specs
# ---------------------------------------------------------------------------
ABLATION_SPECS = {
    "D_baseline": SelectorSpec(
        name="D_baseline",
        lane="ablation",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="plain",
    ),
    "D_guard": SelectorSpec(
        name="D_guard",
        lane="ablation",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard",
        cooldown_grid=(0, 32),
    ),
    "D_cooldown": SelectorSpec(
        name="D_cooldown",
        lane="ablation",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="plain",
        cooldown_grid=(0, 32),
    ),
    "D_pullback": SelectorSpec(
        name="D_pullback",
        lane="ablation",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback_evalonly",
        cooldown_grid=(0, 32),
    ),
    "D_guard_pullback": SelectorSpec(
        name="D_guard_pullback",
        lane="ablation",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback_evalonly",
        cooldown_grid=(0, 32),
    ),
    "FULL": SelectorSpec(
        name="FULL",
        lane="full",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback_evalonly",
        cooldown_grid=(0, 32),
    ),
}

# Also include the original Plan2 best candidate for verification
REPRODUCE_SPEC = {
    "D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly": (
        SELECTOR_SPECS[14]  # index 14 in the tuple
        if len(SELECTOR_SPECS) > 14
        else SelectorSpec(
            name="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly",
            lane="D_risk_sensitive_plus_A_guard",
            candidates=(0.75, 1.0, 1.05, 1.10),
            horizon=32,
            dd_penalty=1.50,
            vol_penalty=0.15,
            active_cap=0.25,
            maxdd_cap_pt=0.00,
            turnover_cap=3.50,
            min_threshold=0.001,
            mode="tb_guard_pullback_evalonly",
            cooldown_grid=(0, 32),
        )
    ),
}


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _min(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.min(vals)) if vals else float("nan")


def _max(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.max(vals)) if vals else float("nan")


def compute_fold3_dependency_stats(results: dict[str, dict], variant_name: str) -> dict:
    """Compute stats with and without fold3 to check dependency."""
    rows_all = []
    rows_no_fold3 = []
    for fold_str, fold_rows in results.items():
        row = fold_rows.get("selectors", {}).get(variant_name)
        if row and row.get("status") == "ok":
            test = row["test"]
            rows_all.append(test)
            if fold_str != "3":
                rows_no_fold3.append(test)

    if not rows_all:
        return {"status": "no_data"}

    def _stats(rows):
        alpha = [r["alpha_excess_pt"] for r in rows]
        maxdd = [r["maxdd_delta_pt"] for r in rows]
        sharpe = [r["sharpe_delta"] for r in rows]
        turnover = [r["turnover"] for r in rows]
        flat_rate = [r["flat_rate"] for r in rows]
        pass_count = sum(
            1
            for a, d, to, fl in zip(alpha, maxdd, turnover, flat_rate)
            if a > 0.0 and d <= 0.0 and to <= 3.5
        )
        active_count = sum(1 for fl in flat_rate if fl < 0.995)
        positive_count = sum(1 for a in alpha if a > 0.0)
        return {
            "n_folds": len(rows),
            "alpha_mean": _mean(alpha),
            "alpha_median": _median(alpha),
            "alpha_worst": _min(alpha),
            "maxdd_mean": _mean(maxdd),
            "maxdd_worst": _max(maxdd),
            "sharpe_mean": _mean(sharpe),
            "turnover_max": _max(turnover),
            "pass_rate": pass_count / len(rows) if rows else 0.0,
            "active_folds": active_count,
            "positive_folds": positive_count,
            "fold_win_rate": positive_count / len(rows) if rows else 0.0,
        }

    return {
        "all_folds": _stats(rows_all),
        "without_fold3": _stats(rows_no_fold3),
        "fold3_contribution": {
            "alpha_delta": _stats(rows_all)["alpha_mean"] - _stats(rows_no_fold3)["alpha_mean"],
            "folds_included": [int(f) for f in results.keys()],
        },
    }


def _evaluate_single_selector(
    *,
    spec: SelectorSpec,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    l2: float,
    seed: int,
) -> dict:
    """Evaluate a single selector spec. Re-implements the core logic from exploration_board_probe
    so the overlay is self-contained and verifiable."""
    unit_cost = _unit_cost(costs_cfg)

    # Candidate utilities
    y_train, train_valid = _candidate_utilities(
        train_returns, candidates=spec.candidates, horizon=spec.horizon,
        benchmark_position=benchmark_position, unit_cost=unit_cost,
        dd_penalty=spec.dd_penalty, vol_penalty=spec.vol_penalty,
    )
    y_val, val_valid = _candidate_utilities(
        val_returns, candidates=spec.candidates, horizon=spec.horizon,
        benchmark_position=benchmark_position, unit_cost=unit_cost,
        dd_penalty=spec.dd_penalty, vol_penalty=spec.vol_penalty,
    )
    y_test, test_valid = _candidate_utilities(
        test_returns, candidates=spec.candidates, horizon=spec.horizon,
        benchmark_position=benchmark_position, unit_cost=unit_cost,
        dd_penalty=spec.dd_penalty, vol_penalty=spec.vol_penalty,
    )

    # Fit ridge model
    uses_tb_guard = spec.mode in {"tb_guard", "tb_guard_pullback", "tb_guard_pullback_evalonly"}
    if spec.mode == "uncertainty":
        from unidream.cli.exploration_board_probe import _fit_bootstrap_models, _predict_ensemble
        models = _fit_bootstrap_models(x_train[train_valid], y_train[train_valid], l2=l2, seed=seed, n_models=5)
        if not models:
            return {"status": "no_model"}
        pred_val, unc_val_all = _predict_ensemble(models, x_val)
        pred_test, unc_test_all = _predict_ensemble(models, x_test)
    else:
        model = _fit_ridge_multi(x_train[train_valid], y_train[train_valid], l2=l2)
        if model is None:
            return {"status": "no_model"}
        pred_val = model.predict(x_val)
        pred_test = model.predict(x_test)
        unc_val_all = np.zeros_like(pred_val)
        unc_test_all = np.zeros_like(pred_test)

    # Triple-barrier guard
    danger_val = None
    danger_test = None
    danger_caps: list[float | None] = [None]
    if uses_tb_guard:
        train_tb = _triple_barrier_labels(
            train_returns, horizon=spec.guard_horizon,
            vol_window=spec.guard_vol_window, barrier_k=spec.guard_barrier_k,
        )
        val_tb = _triple_barrier_labels(
            val_returns, horizon=spec.guard_horizon,
            vol_window=spec.guard_vol_window, barrier_k=spec.guard_barrier_k,
        )
        test_tb = _triple_barrier_labels(
            test_returns, horizon=spec.guard_horizon,
            vol_window=spec.guard_vol_window, barrier_k=spec.guard_barrier_k,
        )
        tb_train_valid = np.asarray(train_tb["valid"], dtype=bool)
        tb_model = _fit_binary_model(
            x_train[tb_train_valid],
            np.asarray(train_tb["tb_down"][tb_train_valid], dtype=np.int64),
            max_train_samples=50000,
            seed=seed + 917,
        )
        danger_val = _score_binary(tb_model, x_val)
        danger_test = _score_binary(tb_model, x_test)
        finite_danger = danger_val[np.asarray(val_tb["valid"], dtype=bool) & np.isfinite(danger_val)]
        if len(finite_danger):
            danger_caps = [float(np.quantile(finite_danger, q)) for q in (0.55,)]

    bench_idx = int(np.argmin(np.abs(np.asarray(spec.candidates) - benchmark_position)))
    best_idx_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_idx_val] - pred_val[:, bench_idx]
    best_idx_test = np.argmax(pred_test, axis=1)
    improve_test = pred_test[np.arange(len(pred_test)), best_idx_test] - pred_test[:, bench_idx]

    # Threshold grid
    from unidream.cli.exploration_board_probe import _threshold_grid
    thresholds = _threshold_grid(improve_val[val_valid], active_cap=spec.active_cap)
    thresholds = sorted({
        float("inf") if not math.isfinite(float(t)) else max(float(t), float(spec.min_threshold))
        for t in thresholds
    })

    # Cooldown choices
    cooldown_choices = tuple(spec.cooldown_grid) if spec.cooldown_grid else (int(spec.cooldown_bars),)

    # Pullback masks
    test_pullback_block = (
        _pullback_no_fire_mask(test_returns)
        if spec.mode in {"tb_guard_pullback", "tb_guard_pullback_evalonly"}
        else np.zeros(len(test_returns), dtype=bool)
    )
    val_pullback_block = (
        _pullback_no_fire_mask(val_returns)
        if spec.mode == "tb_guard_pullback"
        else np.zeros(len(val_returns), dtype=bool)
    )

    # Validation selection loop
    best: dict[str, Any] | None = None
    from unidream.cli.exploration_board_probe import _metric_score
    for danger_cap in danger_caps:
        danger_mask = np.ones(len(val_returns), dtype=bool)
        if danger_val is not None and danger_cap is not None:
            danger_mask = np.isfinite(danger_val) & (danger_val <= float(danger_cap))
        for cooldown_choice in cooldown_choices:
            for threshold in thresholds:
                selected_val, diag = _positions_from_prediction(
                    pred_val, candidates=spec.candidates, threshold=threshold,
                    benchmark_position=benchmark_position,
                    active_mask=val_valid & danger_mask & (~val_pullback_block),
                    uncertainty=None, uncertainty_cap=None,
                )
                selected_val = _apply_event_throttle(
                    selected_val, benchmark_position=benchmark_position,
                    cooldown_bars=int(cooldown_choice), hold_bars=spec.hold_bars,
                )
                val_positions = _shift_for_execution(selected_val, benchmark_position)
                val_metrics, _val_pnl = _backtest_positions(
                    val_returns, val_positions, cfg=cfg, costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                )
                score = _metric_score(val_metrics, spec)
                candidate = {
                    "threshold": float(threshold) if math.isfinite(float(threshold)) else "inf",
                    "cooldown_bars": int(cooldown_choice),
                    "danger_cap": float(danger_cap) if danger_cap is not None and math.isfinite(float(danger_cap)) else None,
                    "val_score": float(score),
                }
                if best is None or score > float(best["val_score"]):
                    best = candidate

    if best is None:
        return {"status": "no_selection"}

    # Test evaluation
    threshold = float("inf") if best["threshold"] == "inf" else float(best["threshold"])
    danger_test_mask = np.ones(len(test_returns), dtype=bool)
    if danger_test is not None and best.get("danger_cap") is not None:
        danger_cap = float(best["danger_cap"])
        danger_test_mask = np.isfinite(danger_test) & (danger_test <= danger_cap)
    selected_test, diag_test = _positions_from_prediction(
        pred_test, candidates=spec.candidates, threshold=threshold,
        benchmark_position=benchmark_position,
        active_mask=test_valid & danger_test_mask & (~test_pullback_block),
        uncertainty=None, uncertainty_cap=None,
    )
    selected_test = _apply_event_throttle(
        selected_test, benchmark_position=benchmark_position,
        cooldown_bars=int(best.get("cooldown_bars", spec.cooldown_bars)),
        hold_bars=spec.hold_bars,
    )
    test_positions = _shift_for_execution(selected_test, benchmark_position)
    test_metrics, test_pnl = _backtest_positions(
        test_returns, test_positions, cfg=cfg, costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )
    util_stats = _selected_utility_stats(
        selected_test, y_test, candidates=spec.candidates, benchmark_position=benchmark_position,
    )
    context_stats = _selected_event_context_stats(
        selected_test, test_returns, benchmark_position=benchmark_position,
    )
    from unidream.cli.exploration_board_probe import _ranking_stats
    ranking = _ranking_stats(pred_test, y_test, test_valid, bench_idx)

    # Per-component diagnostics
    diag_components = _component_diagnostics(
        spec=spec, pred_test=pred_test, test_valid=test_valid,
        benchmark_position=benchmark_position,
        danger_test=danger_test, danger_cap=best.get("danger_cap"),
        test_pullback_block=test_pullback_block,
    )

    return {
        "status": "ok",
        "lane": spec.lane,
        "variant": spec.name,
        "candidates": list(spec.candidates),
        "horizon": int(spec.horizon),
        "dd_penalty": float(spec.dd_penalty),
        "vol_penalty": float(spec.vol_penalty),
        "selection": best,
        "test": {**test_metrics, **diag_test, **util_stats, **context_stats, **ranking, **diag_components},
        "test_pnl": test_pnl,
    }


def _component_diagnostics(
    *,
    spec: SelectorSpec,
    pred_test: np.ndarray,
    test_valid: np.ndarray,
    benchmark_position: float,
    danger_test: np.ndarray | None,
    danger_cap: float | str | None,
    test_pullback_block: np.ndarray,
) -> dict:
    """Compute per-component diagnostic metrics."""
    candidates_arr = np.asarray(spec.candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(candidates_arr - float(benchmark_position))))
    best_idx = np.argmax(pred_test, axis=1)
    improve = pred_test[np.arange(len(pred_test)), best_idx] - pred_test[:, bench_idx]

    n_total = len(pred_test)
    n_valid = int(np.sum(test_valid)) if len(test_valid) > 0 else n_total

    # Danger guard effect
    danger_blocked = 0
    if danger_test is not None and danger_cap is not None:
        dc = float("inf") if danger_cap == "inf" else float(danger_cap)
        danger_blocked = int(np.sum(
            np.asarray(test_valid, dtype=bool)
            & np.isfinite(danger_test)
            & (danger_test > dc)
        ))

    # Pullback blocker effect
    pullback_blocked = int(np.sum(np.asarray(test_pullback_block, dtype=bool)))

    # Threshold effect: how many cross min_threshold
    above_floor = int(np.sum(improve > float(spec.min_threshold)))

    has_tb_guard = spec.mode in {"tb_guard", "tb_guard_pullback", "tb_guard_pullback_evalonly"}
    has_pullback = spec.mode in {"tb_guard_pullback", "tb_guard_pullback_evalonly"}
    has_cooldown = bool(spec.cooldown_grid) or spec.cooldown_bars > 0

    return {
        "n_total_bars": n_total,
        "n_valid_bars": n_valid,
        "above_threshold_floor": above_floor,
        "danger_guard_blocked": danger_blocked if has_tb_guard else 0,
        "pullback_blocked": pullback_blocked if has_pullback else 0,
        "has_tb_guard": has_tb_guard,
        "has_pullback": has_pullback,
        "has_cooldown": has_cooldown,
        "cooldown_bars": spec.cooldown_bars if has_cooldown else 0,
        "cooldown_grid": list(spec.cooldown_grid) if has_cooldown else [],
        "min_threshold": float(spec.min_threshold),
    }


def _write_md(path: str, payload: dict) -> None:
    lines = [
        "# Plan 3 Overlay Verification Report",
        "",
        f"Config: `{payload['config']}`",
        f"Period: `{payload['start']}` — `{payload['end']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        "",
        "---",
        "",
    ]

    # Ablation table
    if payload.get("ablation_aggregate"):
        lines.append("## Round A: Standalone Overlay Reproduction")
        lines.append("")
        for group, rows in payload["ablation_aggregate"].items():
            lines.extend([
                f"### {group}",
                "",
                "| variant | folds | AlphaEx mean | AlphaEx median | AlphaEx worst | MaxDDΔ mean | MaxDDΔ worst | SharpeΔ mean | turnover max | flat mean | pass rate | PBO |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for variant, row in rows.items():
                if variant.startswith("_"):
                    continue
                lines.append(
                    "| " + " | ".join([
                        variant, str(row["folds"]),
                        _fmt(row["alpha_mean"]), _fmt(row.get("alpha_median", float("nan"))),
                        _fmt(row["alpha_worst"]), _fmt(row["maxdd_mean"]),
                        _fmt(row["maxdd_worst"]), _fmt(row["sharpe_mean"]),
                        _fmt(row["turnover_max"]), _fmt(row.get("flat_mean", float("nan"))),
                        _fmt(row.get("pass_rate", row.get("pass_rate_alpha_pos_maxdd_nonpos", float("nan")))),
                        _fmt(row.get("pbo", float("nan"))),
                    ]) + " |"
                )
            lines.append("")

    # Fold3 dependency
    if payload.get("fold3_dependency"):
        lines.append("## Round C: Fold3 Dependency Check")
        lines.append("")
        fd = payload["fold3_dependency"]
        for variant_name, stats in fd.items():
            lines.append(f"### {variant_name}")
            lines.append("")
            for key, s in stats.items():
                if not isinstance(s, dict):
                    continue
                lines.append(f"**{key}**:")
                lines.append("")
                lines.append("| metric | value |")
                lines.append("|---:|---:|")
                for k, v in s.items():
                    if isinstance(v, float):
                        lines.append(f"| {k} | {_fmt(v)} |")
                    else:
                        lines.append(f"| {k} | {v} |")
                lines.append("")

    # Per-fold detail
    if payload.get("results"):
        lines.append("## Fold Detail")
        lines.append("")
        for fold, fold_rows in payload["results"].items():
            lines.append(f"### Fold {fold}")
            lines.append("")
            lines.append("| variant | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat | above_floor | danger_blocked | pullback_blocked |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for variant, row in fold_rows.get("selectors", {}).items():
                if row.get("status") != "ok":
                    continue
                t = row["test"]
                lines.append(
                    "| " + " | ".join([
                        variant,
                        _fmt(t["alpha_excess_pt"]),
                        _fmt(t["maxdd_delta_pt"]),
                        _fmt(t["sharpe_delta"]),
                        _fmt(t["turnover"]),
                        _fmt(t["flat_rate"]),
                        _fmt(t.get("above_threshold_floor", 0)),
                        _fmt(t.get("danger_guard_blocked", 0)),
                        _fmt(t.get("pullback_blocked", 0)),
                    ]) + " |"
                )
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan3_overlay_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--mode", choices=["reproduce", "ablation", "full"], default="full",
                        help="reproduce: verify Plan2 result; ablation: component ablation; full: both")
    parser.add_argument("--output-json", default="documents/20260502_plan3_overlay.json")
    parser.add_argument("--output-md", default="documents/20260502_plan3_overlay.md")
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol, interval=interval, start=args.start, end=args.end,
        zscore_window=zscore_window, cache_dir="checkpoints/data_cache", cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))

    # Collect selectors to run
    selector_specs: list[SelectorSpec] = []
    if args.mode in ("reproduce", "full"):
        selector_specs.extend(REPRODUCE_SPEC.values())
    if args.mode in ("ablation", "full"):
        selector_specs.extend(ABLATION_SPECS.values())

    if not selector_specs:
        raise ValueError("No selectors to run")

    results: dict[str, dict] = {}
    for split in splits:
        print(f"[Plan3Overlay] fold={split.fold_idx}")
        dataset = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train = _state_features(dataset.train_features, dataset.train_returns)
        x_val = _state_features(dataset.val_features, dataset.val_returns)
        x_test = _state_features(dataset.test_features, dataset.test_returns)
        fold_rows: dict[str, Any] = {"selectors": {}}
        for spec in selector_specs:
            print(f"  selector={spec.name}")
            fold_rows["selectors"][spec.name] = _evaluate_single_selector(
                spec=spec, x_train=x_train, x_val=x_val, x_test=x_test,
                train_returns=dataset.train_returns, val_returns=dataset.val_returns,
                test_returns=dataset.test_returns, cfg=cfg, costs_cfg=costs_cfg,
                benchmark_position=benchmark_position, l2=args.ridge_l2,
                seed=args.seed + int(split.fold_idx) * 100,
            )
        results[str(split.fold_idx)] = fold_rows

    fold_ids = [int(split.fold_idx) for split in splits]
    groups = {"all": fold_ids}
    if all(f in fold_ids for f in (4, 5, 6)):
        groups["f456"] = [4, 5, 6]
    if all(f in fold_ids for f in (0, 4, 5)):
        groups["f045"] = [0, 4, 5]

    # Aggregate
    ablation_aggregate = _aggregate_selectors(results, groups)

    # Fold3 dependency
    fold3_dependency = {}
    for variant_name in [s.name for s in selector_specs]:
        fold3_dependency[variant_name] = compute_fold3_dependency_stats(results, variant_name)

    payload = {
        "config": args.config,
        "start": args.start,
        "end": args.end,
        "folds": fold_ids,
        "seed": args.seed,
        "mode": args.mode,
        "results": _json_sanitize(results),
        "ablation_aggregate": _json_sanitize(ablation_aggregate),
        "fold3_dependency": _json_sanitize(fold3_dependency),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Plan3Overlay] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
