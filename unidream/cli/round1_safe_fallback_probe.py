from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle,
    _backtest_positions,
    _candidate_utilities,
    _fit_ridge_multi,
    _positions_from_prediction,
    _shift_for_execution,
    _state_features,
    _threshold_grid,
    _triple_barrier_labels,
    _unit_cost,
)
from unidream.cli.route_separability_probe import (
    _fit_binary_model,
    _score_binary,
)
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "round1_safe_fallback_probe"

# ── Safe-fallback configuration presets ──────────────────────────────────────
SAFE_FALLBACK_SPECS: list[dict[str, Any]] = [
    {
        "name": "SF_base",
        "candidates": (0.75, 1.0, 1.05),
        "horizon": 32,
        "dd_penalty": 1.00,
        "vol_penalty": 0.10,
        "use_tb_guard": True,
        "guard_horizon": 32,
        "guard_vol_window": 64,
        "guard_barrier_k": 1.25,
        "cooldown_grid": (0, 32, 64),
        "active_cap": 0.25,
        "maxdd_cap": 0.0,
        "turnover_cap": 3.5,
        "ridge_l2": 1.0,
    },
    {
        "name": "SF_narrow",
        "candidates": (0.85, 1.0, 1.05),
        "horizon": 32,
        "dd_penalty": 1.00,
        "vol_penalty": 0.10,
        "use_tb_guard": True,
        "guard_horizon": 32,
        "guard_vol_window": 64,
        "guard_barrier_k": 1.25,
        "cooldown_grid": (0, 32, 64),
        "active_cap": 0.20,
        "maxdd_cap": 0.0,
        "turnover_cap": 3.0,
        "ridge_l2": 1.0,
    },
    {
        "name": "SF_noguard",
        "candidates": (0.75, 1.0, 1.05),
        "horizon": 32,
        "dd_penalty": 1.00,
        "vol_penalty": 0.10,
        "use_tb_guard": False,
        "cooldown_grid": (0, 32, 64),
        "active_cap": 0.25,
        "maxdd_cap": 0.0,
        "turnover_cap": 3.5,
        "ridge_l2": 1.0,
    },
    {
        "name": "SF_risk_averse",
        "candidates": (0.75, 1.0, 1.05),
        "horizon": 32,
        "dd_penalty": 2.00,
        "vol_penalty": 0.20,
        "use_tb_guard": True,
        "guard_horizon": 32,
        "guard_vol_window": 64,
        "guard_barrier_k": 1.25,
        "cooldown_grid": (0, 32, 64, 96),
        "active_cap": 0.15,
        "maxdd_cap": 0.0,
        "turnover_cap": 2.5,
        "ridge_l2": 1.0,
    },
    {
        "name": "SF_long_cd",
        "candidates": (0.75, 1.0, 1.05),
        "horizon": 32,
        "dd_penalty": 1.00,
        "vol_penalty": 0.10,
        "use_tb_guard": True,
        "guard_horizon": 32,
        "guard_vol_window": 64,
        "guard_barrier_k": 1.25,
        "cooldown_grid": (32, 64, 96, 128),
        "active_cap": 0.15,
        "maxdd_cap": 0.0,
        "turnover_cap": 3.0,
        "ridge_l2": 1.0,
    },
]


# ── Utility helpers ──────────────────────────────────────────────────────────


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanmin(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def _nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


# ── Safe-fallback validation scoring ─────────────────────────────────────────


def _safe_fallback_val_score(
    *,
    alpha_excess_pt: float,
    maxdd_delta_pt: float,
    turnover: float,
    active_rate: float,
    maxdd_cap: float = 0.0,
    turnover_cap: float = 3.5,
    active_cap: float = 0.25,
    alpha_weight: float = 1.0,
    maxdd_penalty: float = 10.0,
    turnover_penalty: float = 0.05,
) -> float:
    """Validation objective that rewards AlphaEx > 0, MaxDDDelta <= 0, low turnover.

    Hard veto on constraint violations; within bounds, a linear trade-off.
    """
    alpha = float(alpha_excess_pt)
    maxdd = float(maxdd_delta_pt)
    to = float(turnover)
    active = float(active_rate)

    # Hard veto for constraint violations
    if maxdd > float(maxdd_cap):
        return -10_000.0 + alpha - 20.0 * maxdd - turnover_penalty * to
    if to > float(turnover_cap):
        return -10_000.0 + alpha - 20.0 * maxdd - turnover_penalty * to
    if active > float(active_cap):
        return -10_000.0 + alpha - 20.0 * maxdd - turnover_penalty * to

    # Within bounds: trade alpha for maxdd and turnover
    maxdd_cost = float(maxdd_penalty) * max(0.0, maxdd)
    turnover_cost = float(turnover_penalty) * to
    return float(alpha_weight) * alpha - maxdd_cost - turnover_cost


# ── Core evaluation ──────────────────────────────────────────────────────────


def _evaluate_safe_fallback(
    *,
    spec: dict[str, Any],
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    unit_cost: float,
    seed: int,
    max_train_samples: int = 50000,
) -> dict[str, Any]:
    """Run one safe-fallback spec on one fold.

    1. Train ridge model on train set utility targets.
    2. Grid-search threshold / danger-cap / cooldown on validation
       using the safe-fallback scoring function.
    3. Apply the best config once to the test set.
    """
    bench = float(benchmark_position)
    candidates = tuple(spec["candidates"])
    horizon = int(spec["horizon"])
    dd_penalty = float(spec["dd_penalty"])
    vol_penalty = float(spec["vol_penalty"])
    ridge_l2 = float(spec["ridge_l2"])
    use_tb_guard = bool(spec.get("use_tb_guard", True))
    guard_horizon = int(spec.get("guard_horizon", 32))
    guard_vol_window = int(spec.get("guard_vol_window", 64))
    guard_barrier_k = float(spec.get("guard_barrier_k", 1.25))
    cooldown_grid = tuple(spec.get("cooldown_grid", (0, 32, 64)))
    active_cap = float(spec.get("active_cap", 0.25))
    maxdd_cap = float(spec.get("maxdd_cap", 0.0))
    turnover_cap = float(spec.get("turnover_cap", 3.5))
    maxdd_penalty = float(spec.get("maxdd_penalty", 10.0))
    turnover_penalty_val = float(spec.get("turnover_penalty", 0.05))

    # ── Compute candidate utilities ──
    y_train, train_valid = _candidate_utilities(
        train_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=bench,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )
    y_val, val_valid = _candidate_utilities(
        val_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=bench,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )
    y_test, test_valid = _candidate_utilities(
        test_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=bench,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )

    # ── Train ridge model ──
    model = _fit_ridge_multi(x_train[train_valid], y_train[train_valid], l2=ridge_l2)
    if model is None:
        return {"status": "no_model"}

    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)

    bench_idx = int(np.argmin(np.abs(np.asarray(candidates) - bench)))
    best_idx_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_idx_val] - pred_val[:, bench_idx]

    # ── Triple-barrier danger guard ──
    danger_val_scores: np.ndarray | None = None
    danger_test_scores: np.ndarray | None = None
    danger_caps: list[float | None] = [None]
    if use_tb_guard:
        train_tb = _triple_barrier_labels(
            train_returns,
            horizon=guard_horizon,
            vol_window=guard_vol_window,
            barrier_k=guard_barrier_k,
        )
        val_tb = _triple_barrier_labels(
            val_returns,
            horizon=guard_horizon,
            vol_window=guard_vol_window,
            barrier_k=guard_barrier_k,
        )
        test_tb = _triple_barrier_labels(
            test_returns,
            horizon=guard_horizon,
            vol_window=guard_vol_window,
            barrier_k=guard_barrier_k,
        )
        tb_train_valid = np.asarray(train_tb["valid"], dtype=bool)
        tb_model = _fit_binary_model(
            x_train[tb_train_valid],
            np.asarray(train_tb["tb_down"][tb_train_valid], dtype=np.int64),
            max_train_samples=max_train_samples,
            seed=seed + 917,
        )
        if tb_model is not None:
            danger_val_scores = _score_binary(tb_model, x_val)
            danger_test_scores = _score_binary(tb_model, x_test)
            finite_danger = danger_val_scores[
                np.asarray(val_tb["valid"], dtype=bool) & np.isfinite(danger_val_scores)
            ]
            if len(finite_danger) > 0:
                danger_caps = [
                    float(np.quantile(finite_danger, q)) for q in (0.15, 0.25, 0.40, 0.55, 0.70)
                ]
                danger_caps.append(float("inf"))
                danger_caps.append(None)

    # ── Threshold grid from validation improvements ──
    thresholds = _threshold_grid(improve_val[val_valid], active_cap=active_cap)

    # ── Grid search on validation ──
    best_val_score = float("-inf")
    best_config: dict[str, Any] = {}
    val_search_count = 0

    for threshold in thresholds:
        for danger_cap in danger_caps:
            danger_mask = np.ones(len(val_returns), dtype=bool)
            if (
                danger_val_scores is not None
                and danger_cap is not None
                and math.isfinite(float(danger_cap))
            ):
                danger_mask = np.isfinite(danger_val_scores) & (
                    danger_val_scores <= float(danger_cap)
                )

            for cooldown in cooldown_grid:
                val_search_count += 1
                selected_val, diag = _positions_from_prediction(
                    pred_val,
                    candidates=candidates,
                    threshold=float(threshold) if math.isfinite(float(threshold)) else float("inf"),
                    benchmark_position=bench,
                    active_mask=val_valid & danger_mask,
                )
                selected_val = _apply_event_throttle(
                    selected_val,
                    benchmark_position=bench,
                    cooldown_bars=int(cooldown),
                    hold_bars=1,
                )
                val_positions = _shift_for_execution(selected_val, bench)
                val_metrics, _val_pnl = _backtest_positions(
                    val_returns,
                    val_positions,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=bench,
                )

                active_rate = 1.0 - float(val_metrics.get("flat_rate", 1.0))
                score = _safe_fallback_val_score(
                    alpha_excess_pt=float(val_metrics.get("alpha_excess_pt", 0.0)),
                    maxdd_delta_pt=float(val_metrics.get("maxdd_delta_pt", 0.0)),
                    turnover=float(val_metrics.get("turnover", 0.0)),
                    active_rate=active_rate,
                    maxdd_cap=maxdd_cap,
                    turnover_cap=turnover_cap,
                    active_cap=active_cap,
                    maxdd_penalty=maxdd_penalty,
                    turnover_penalty=turnover_penalty_val,
                )

                if score > best_val_score:
                    best_val_score = score
                    best_config = {
                        "threshold": (
                            float(threshold)
                            if math.isfinite(float(threshold))
                            else "inf"
                        ),
                        "danger_cap": (
                            float(danger_cap)
                            if danger_cap is not None and math.isfinite(float(danger_cap))
                            else ("inf" if danger_cap is not None else None)
                        ),
                        "cooldown_bars": int(cooldown),
                        "val_score": float(score),
                        "val_alpha": float(val_metrics.get("alpha_excess_pt", 0.0)),
                        "val_maxdd": float(val_metrics.get("maxdd_delta_pt", 0.0)),
                        "val_turnover": float(val_metrics.get("turnover", 0.0)),
                        "val_active": active_rate,
                        "val_sharpe": float(val_metrics.get("sharpe_delta", 0.0)),
                    }

    if not best_config:
        return {"status": "no_selection", "val_search_size": val_search_count}

    # ── Apply best config to test ──
    threshold = (
        float("inf")
        if best_config.get("threshold") == "inf"
        else float(best_config["threshold"])
    )
    danger_cap = best_config.get("danger_cap")

    danger_test_mask = np.ones(len(test_returns), dtype=bool)
    if (
        danger_test_scores is not None
        and danger_cap is not None
        and danger_cap != "inf"
        and math.isfinite(float(danger_cap))
    ):
        danger_test_mask = np.isfinite(danger_test_scores) & (
            danger_test_scores <= float(danger_cap)
        )

    cooldown = int(best_config.get("cooldown_bars", 0))

    selected_test, diag_test = _positions_from_prediction(
        pred_test,
        candidates=candidates,
        threshold=threshold,
        benchmark_position=bench,
        active_mask=test_valid & danger_test_mask,
    )
    selected_test = _apply_event_throttle(
        selected_test,
        benchmark_position=bench,
        cooldown_bars=cooldown,
        hold_bars=1,
    )
    test_positions = _shift_for_execution(selected_test, bench)
    test_metrics, _test_pnl = _backtest_positions(
        test_returns,
        test_positions,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=bench,
    )

    active_rate_test = 1.0 - float(test_metrics.get("flat_rate", 1.0))

    return {
        "status": "ok",
        "spec_name": spec["name"],
        "candidates": list(candidates),
        "horizon": horizon,
        "dd_penalty": dd_penalty,
        "vol_penalty": vol_penalty,
        "use_tb_guard": use_tb_guard,
        "val_selection": best_config,
        "val_search_size": val_search_count,
        "test": {
            "alpha_excess_pt": float(test_metrics.get("alpha_excess_pt", 0.0)),
            "sharpe_delta": float(test_metrics.get("sharpe_delta", 0.0)),
            "maxdd_delta_pt": float(test_metrics.get("maxdd_delta_pt", 0.0)),
            "turnover": float(test_metrics.get("turnover", 0.0)),
            "long_rate": float(test_metrics.get("long_rate", 0.0)),
            "flat_rate": float(test_metrics.get("flat_rate", 1.0)),
            "period_win_rate": float(test_metrics.get("period_win_rate", 0.0)),
            "n_trades": int(test_metrics.get("n_trades", 0)),
            "active_rate": active_rate_test,
        },
        "test_pnl": _test_pnl,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate(
    results: dict[str, dict[str, Any]],
    fold_ids: list[int],
) -> dict[str, dict[str, Any]]:
    """Aggregate across folds for each spec variant."""
    spec_names = sorted(
        {
            row["spec_name"]
            for fold_rows in results.values()
            for row in fold_rows.get("specs", [])
            if row.get("status") == "ok"
        }
    )
    out: dict[str, dict[str, Any]] = {}
    for name in spec_names:
        rows = []
        pnls = []
        for fid in fold_ids:
            for row in results.get(str(fid), {}).get("specs", []):
                if row.get("spec_name") == name and row.get("status") == "ok":
                    rows.append(row["test"])
                    if "test_pnl" in row:
                        pnls.append(row["test_pnl"])
                    break
        if not rows:
            continue
        alphas = [r["alpha_excess_pt"] for r in rows]
        maxdds = [r["maxdd_delta_pt"] for r in rows]
        turnovers = [r["turnover"] for r in rows]
        out[name] = {
            "folds": int(len(rows)),
            "alpha_mean": _nanmean(alphas),
            "alpha_worst": _nanmin(alphas),
            "maxdd_mean": _nanmean(maxdds),
            "maxdd_worst": _nanmax(maxdds),
            "turnover_mean": _nanmean(turnovers),
            "turnover_max": _nanmax(turnovers),
            "pass_alpha_gt0": float(
                np.mean([a > 0.0 for a in alphas])
            ),
            "pass_maxdd_le0": float(
                np.mean([d <= 0.0 for d in maxdds])
            ),
            "pass_both": float(
                np.mean([a > 0.0 and d <= 0.0 for a, d in zip(alphas, maxdds)])
            ),
            "test_pnls": [p.tolist() if isinstance(p, np.ndarray) else p for p in pnls],
        }
    return out


# ── Report writing ───────────────────────────────────────────────────────────


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 1 Safe-Fallback Probe Report",
        "",
        f"**Date**: {payload.get('date', 'unknown')}",
        f"**Experiment**: `{payload.get('experiment', EXPERIMENT_NAME)}`",
        f"**Config**: `{payload.get('config', 'N/A')}`",
        f"**Date range**: {payload.get('start', '?')} → {payload.get('end', '?')}",
        f"**Folds evaluated**: `{', '.join(map(str, payload.get('folds', [])))}`",
        f"**Seed**: {payload.get('seed', '?')}",
        "",
        "---",
        "",
        "## Goal",
        "",
        "Improve toward **AlphaEx > 0** and **MaxDDDelta <= 0** on as many folds as possible",
        "using conservative safe-fallback thresholding selected on validation.",
        "",
        "## Method",
        "",
        "1. Train ridge model to predict utility of candidate positions (0.75 / 1.0 / 1.05).",
        "2. Optionally train triple-barrier down model as danger guard.",
        "3. Grid-search threshold, danger cap, and cooldown on validation.",
        "4. Score each config with: `alpha - 10*max(0,maxdd) - 0.05*turnover`",
        "   (hard veto if maxdd > 0, turnover > cap, or active > cap).",
        "5. Apply best config once to test.",
        "",
        "---",
        "",
        "## Commands",
        "",
        "```bash",
        f"python -m unidream.cli.{EXPERIMENT_NAME} \\",
        "    --config configs/trading.yaml \\",
        f"    --folds {','.join(map(str, payload.get('folds', [])))} \\",
        f"    --seed {payload.get('seed', '?')} \\",
        f"    --output-json docs_local/{payload.get('date', 'yyyymmdd')}_{EXPERIMENT_NAME}.json \\",
        f"    --output-md docs_local/{payload.get('date', 'yyyymmdd')}_{EXPERIMENT_NAME}.md",
        "```",
        "",
        "---",
        "",
        "## Aggregate Results",
        "",
        "| spec | folds | α mean | α worst | maxdd mean | maxdd worst | turnover mean | turnover max | pass α>0 | pass dd≤0 | pass both |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload.get("aggregate", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["folds"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_mean"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                    _fmt(row["turnover_max"]),
                    _fmt(row["pass_alpha_gt0"]),
                    _fmt(row["pass_maxdd_le0"]),
                    _fmt(row["pass_both"]),
                ]
            )
            + " |"
        )
    lines.append("")

    # Per-fold detail
    lines.extend(["---", "", "## Per-Fold Detail", ""])
    for fold_id in payload.get("folds", []):
        lines.extend([f"### Fold {fold_id}", ""])
        lines.append(
            "| spec | α | maxdd | sharpe | turnover | long% | flat% | active% | trades | "
            "val α | val maxdd | val threshold | val danger | val cd |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        fold_rows = payload.get("results", {}).get(str(fold_id), {}).get("specs", [])
        for row in fold_rows:
            if row.get("status") != "ok":
                continue
            test = row.get("test", {})
            sel = row.get("val_selection", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("spec_name", "?"),
                        _fmt(test.get("alpha_excess_pt")),
                        _fmt(test.get("maxdd_delta_pt")),
                        _fmt(test.get("sharpe_delta")),
                        _fmt(test.get("turnover")),
                        _fmt(test.get("long_rate")),
                        _fmt(test.get("flat_rate")),
                        _fmt(test.get("active_rate")),
                        str(test.get("n_trades", 0)),
                        _fmt(sel.get("val_alpha")),
                        _fmt(sel.get("val_maxdd")),
                        str(sel.get("threshold", "?")),
                        str(sel.get("danger_cap", "?")),
                        str(sel.get("cooldown_bars", "?")),
                    ]
                )
                + " |"
            )
        lines.append("")

    # Best configs summary
    lines.extend(["---", "", "## Best Configs by Spec", ""])
    for name in sorted(payload.get("aggregate", {}).keys()):
        agg = payload["aggregate"][name]
        lines.append(f"- **{name}**: α mean={_fmt(agg['alpha_mean'])}, "
                     f"α worst={_fmt(agg['alpha_worst'])}, "
                     f"maxdd worst={_fmt(agg['maxdd_worst'])}, "
                     f"pass both={_fmt(agg['pass_both'])}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Goal Assessment",
            "",
        ]
    )
    # Per-fold goal check
    for fold_id in payload.get("folds", []):
        fold_rows = payload.get("results", {}).get(str(fold_id), {}).get("specs", [])
        best_row = None
        best_score = float("-inf")
        for row in fold_rows:
            if row.get("status") != "ok":
                continue
            test = row.get("test", {})
            a = float(test.get("alpha_excess_pt", 0.0))
            d = float(test.get("maxdd_delta_pt", 0.0))
            score = a - 10.0 * max(0.0, d)
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None:
            lines.append(f"- **Fold {fold_id}**: no valid result")
            continue
        test = best_row["test"]
        a = float(test.get("alpha_excess_pt", 0.0))
        d = float(test.get("maxdd_delta_pt", 0.0))
        a_ok = "PASS" if a > 0.0 else "FAIL"
        d_ok = "PASS" if d <= 0.0 else "FAIL"
        both_ok = "**MET**" if (a > 0.0 and d <= 0.0) else "NOT MET"
        lines.append(
            f"- **Fold {fold_id}** [{best_row['spec_name']}]: "
            f"α={_fmt(a)} ({a_ok}), "
            f"maxddΔ={_fmt(d)} ({d_ok}), "
            f"turnover={_fmt(test.get('turnover'))}, "
            f"goal → {both_ok}"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Notes",
            "",
            "- Scope: experimental CLI only; no config or production behavior changes.",
            "- Thresholds are selected on validation using the safe-fallback objective, then applied once to test.",
            "- The validation objective hard-vetoes any config where MaxDDDelta > 0 bp, then trades alpha for turnover.",
            "- See `docs_local/` for the full JSON output.",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=f"python -m unidream.cli.{EXPERIMENT_NAME}"
    )
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,4,5")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spec-names", default="")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    date = _date_prefix()
    if not args.output_json:
        args.output_json = os.path.join(
            "docs_local", f"{date}_{EXPERIMENT_NAME}.json"
        )
    if not args.output_md:
        args.output_md = os.path.join(
            "docs_local", f"{date}_{EXPERIMENT_NAME}.md"
        )

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
    splits, _selected = select_wfo_splits(
        build_wfo_splits(features_df, data_cfg), args.folds
    )
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(
        cfg.get("reward", {}).get("benchmark_position", 1.0)
    )
    unit_cost = _unit_cost(costs_cfg)

    spec_filter = {
        x.strip()
        for x in str(args.spec_names or "").split(",")
        if x.strip()
    }
    specs = [
        s for s in SAFE_FALLBACK_SPECS if not spec_filter or s["name"] in spec_filter
    ]
    if not specs:
        raise ValueError(
            f"No spec matched --spec-names={args.spec_names!r}"
        )

    results: dict[str, dict[str, Any]] = {}
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[SafeFallback] fold={fid} start")
        dataset = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=cfg.get("data", {}).get("seq_len", 64),
        )
        x_train = _state_features(dataset.train_features, dataset.train_returns)
        x_val = _state_features(dataset.val_features, dataset.val_returns)
        x_test = _state_features(dataset.test_features, dataset.test_returns)

        spec_rows: list[dict[str, Any]] = []
        for spec in specs:
            print(f"[SafeFallback] fold={fid} spec={spec['name']}")
            row = _evaluate_safe_fallback(
                spec=spec,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                train_returns=dataset.train_returns,
                val_returns=dataset.val_returns,
                test_returns=dataset.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                seed=args.seed + fid * 100,
                max_train_samples=args.max_train_samples,
            )
            spec_rows.append(row)
        results[str(fid)] = {"specs": spec_rows}

    fold_ids = [int(split.fold_idx) for split in splits]
    payload = {
        "experiment": EXPERIMENT_NAME,
        "date": date,
        "config": args.config,
        "start": args.start,
        "end": args.end,
        "folds": fold_ids,
        "seed": int(args.seed),
        "spec_names": [s["name"] for s in specs],
        "results": _json_sanitize(results),
        "aggregate": _json_sanitize(_aggregate(results, fold_ids)),
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[SafeFallback] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
