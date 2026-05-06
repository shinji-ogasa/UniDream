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
    _rolling_past_vol,
    _shift_for_execution,
    _state_features,
    _top_fraction_mean,
    _triple_barrier_labels,
    _unit_cost,
)
from unidream.cli.plan5_laneF import make_pullback_recovery_label
from unidream.cli.route_separability_probe import (
    _fit_binary_model,
    _safe_ap,
    _safe_auc,
    _score_binary,
    _select_threshold,
)
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "agent_pro_plan001_guard_report"

GUARD_SPECS: list[dict[str, Any]] = [
    {
        "name": "GR_baseline",
        "desc": "vol_shock/recovery/state baseline; no guard",
        "use_tb_guard": False,
        "cooldown_grid": (0,),
        "min_confidence_floor": 0.0,
        "danger_cap_quantiles": [None],
        "maxdd_val_cap": 0.0,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
    {
        "name": "GR_tb_guard",
        "desc": "add TB-down danger guard; grid over danger quantile",
        "use_tb_guard": True,
        "cooldown_grid": (0,),
        "min_confidence_floor": 0.0,
        "danger_cap_quantiles": [0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85],
        "maxdd_val_cap": 0.0,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
    {
        "name": "GR_tb_guard_cd",
        "desc": "danger guard + cooldown throttle",
        "use_tb_guard": True,
        "cooldown_grid": (0, 32, 64),
        "min_confidence_floor": 0.0,
        "danger_cap_quantiles": [0.25, 0.40, 0.55, 0.70],
        "maxdd_val_cap": 0.0,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
    {
        "name": "GR_min_conf",
        "desc": "minimum confidence floor (higher threshold); no guard",
        "use_tb_guard": False,
        "cooldown_grid": (0,),
        "min_confidence_floor": 0.001,
        "danger_cap_quantiles": [None],
        "maxdd_val_cap": 0.0,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
    {
        "name": "GR_combined",
        "desc": "danger guard + cooldown + confidence floor",
        "use_tb_guard": True,
        "cooldown_grid": (0, 32, 64),
        "min_confidence_floor": 0.0005,
        "danger_cap_quantiles": [0.25, 0.40, 0.55, 0.70],
        "maxdd_val_cap": 0.0,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
    {
        "name": "GR_tb_guard_loose",
        "desc": "danger guard with loose danger caps + cooldown",
        "use_tb_guard": True,
        "cooldown_grid": (0, 32, 64),
        "min_confidence_floor": 0.0,
        "danger_cap_quantiles": [0.40, 0.55, 0.70, 0.85, 0.95],
        "maxdd_val_cap": 0.0,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
    {
        "name": "GR_maxdd_epsilon_neg001",
        "desc": "strict MaxDDDelta <= -0.01 bp (epsilon test)",
        "use_tb_guard": True,
        "cooldown_grid": (0, 32, 64),
        "min_confidence_floor": 0.0,
        "danger_cap_quantiles": [0.25, 0.40, 0.55, 0.70],
        "maxdd_val_cap": -0.01,
        "active_cap": 0.20,
        "turnover_cap": 10.0,
    },
]


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


def _vol_shock_mask(returns: np.ndarray, threshold: float) -> np.ndarray:
    ret = np.asarray(returns, dtype=np.float64)
    vol = _rolling_past_vol(ret, 64)
    return (np.abs(ret) > np.maximum(threshold, 1e-12)) | (vol > threshold)


def _threshold_score_grid(score: np.ndarray, active_cap: float) -> list[float]:
    vals = np.asarray(score, dtype=np.float64)
    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals) < 2:
        return [float("inf")]
    qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
    qvals = [float(np.quantile(finite_vals, q)) for q in qs]
    cap_q = max(0.0, min(0.995, 1.0 - float(active_cap)))
    qvals.append(float(np.quantile(finite_vals, cap_q)))
    qvals.extend([float("inf")])
    return sorted(set(qvals))


def _val_score(
    alpha_excess_pt: float,
    maxdd_delta_pt: float,
    turnover: float,
    active_rate: float,
    *,
    maxdd_cap: float = 0.0,
    turnover_cap: float = 10.0,
    active_cap: float = 0.20,
    alpha_weight: float = 1.0,
    maxdd_penalty: float = 10.0,
    turnover_penalty: float = 0.05,
) -> float:
    alpha = float(alpha_excess_pt)
    maxdd = float(maxdd_delta_pt)
    to = float(turnover)
    active = float(active_rate)

    # Hard veto
    if maxdd > float(maxdd_cap):
        return -10_000.0 + alpha - 20.0 * maxdd - turnover_penalty * to
    if to > float(turnover_cap):
        return -10_000.0 + alpha - 20.0 * maxdd - turnover_penalty * to
    if active > float(active_cap):
        return -10_000.0 + alpha - 20.0 * maxdd - turnover_penalty * to

    maxdd_cost = float(maxdd_penalty) * max(0.0, maxdd)
    turnover_cost = float(turnover_penalty) * to
    return float(alpha_weight) * alpha - maxdd_cost - turnover_cost


def _evaluate_guarded(
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
    """Run one guarded-recovery spec on one fold."""
    bench = float(benchmark_position)
    overlay_position = 1.05
    horizon = 32

    use_tb_guard = bool(spec.get("use_tb_guard", False))
    cooldown_grid = tuple(spec.get("cooldown_grid", (0,)))
    min_confidence_floor = float(spec.get("min_confidence_floor", 0.0))
    danger_cap_quantiles = list(spec.get("danger_cap_quantiles", [None]))
    maxdd_val_cap = float(spec.get("maxdd_val_cap", 0.0))
    active_cap = float(spec.get("active_cap", 0.20))
    turnover_cap = float(spec.get("turnover_cap", 10.0))

    # ── Vol-shock event masks ──
    train_vol = _rolling_past_vol(train_returns, 64)
    finite_vol = train_vol[np.isfinite(train_vol)]
    vol_threshold = float(np.quantile(finite_vol, 0.90)) if len(finite_vol) > 0 else float("inf")

    train_event_mask = _vol_shock_mask(train_returns, vol_threshold)
    val_event_mask = _vol_shock_mask(val_returns, vol_threshold)
    test_event_mask = _vol_shock_mask(test_returns, vol_threshold)

    # ── Recovery labels ──
    train_recovery = make_pullback_recovery_label(train_returns, horizon=horizon)
    val_recovery = make_pullback_recovery_label(val_returns, horizon=horizon)
    test_recovery = make_pullback_recovery_label(test_returns, horizon=horizon)

    # ── Valid train mask: vol_shock event + finite features ──
    train_finite = np.all(np.isfinite(x_train), axis=1)
    train_mask = train_event_mask & train_finite
    val_finite = np.all(np.isfinite(x_val), axis=1)
    val_mask = val_event_mask & val_finite
    test_finite = np.all(np.isfinite(x_test), axis=1)
    test_mask = test_event_mask & test_finite

    train_candidate_count = int(train_mask.sum())
    val_candidate_count = int(val_mask.sum())
    test_candidate_count = int(test_mask.sum())

    if train_candidate_count < 100 or val_candidate_count < 20 or test_candidate_count < 20:
        return {
            "status": "insufficient_events",
            "train_candidates": train_candidate_count,
            "val_candidates": val_candidate_count,
            "test_candidates": test_candidate_count,
        }

    # ── Train primary recovery model ──
    train_y_rec = np.asarray(train_recovery[train_mask], dtype=np.int64)
    if len(np.unique(train_y_rec)) < 2:
        return {"status": "one_class_recovery"}

    recovery_model = _fit_binary_model(
        x_train[train_mask],
        train_y_rec,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    if recovery_model is None:
        return {"status": "no_recovery_model"}

    rec_score_train = _score_binary(recovery_model, x_train)
    rec_score_val = _score_binary(recovery_model, x_val)
    rec_score_test = _score_binary(recovery_model, x_test)

    # ── Train TB-down danger model (guard) ──
    danger_train_scores: np.ndarray | None = None
    danger_val_scores: np.ndarray | None = None
    danger_test_scores: np.ndarray | None = None
    danger_caps: list[float | None] = [None]
    danger_auc: float = float("nan")

    if use_tb_guard:
        train_tb = _triple_barrier_labels(
            train_returns, horizon=32, vol_window=64, barrier_k=1.25
        )
        val_tb = _triple_barrier_labels(
            val_returns, horizon=32, vol_window=64, barrier_k=1.25
        )
        test_tb = _triple_barrier_labels(
            test_returns, horizon=32, vol_window=64, barrier_k=1.25
        )
        tb_train_valid = np.asarray(train_tb["valid"], dtype=bool) & train_finite
        tb_val_valid = np.asarray(val_tb["valid"], dtype=bool) & val_finite
        tb_test_valid = np.asarray(test_tb["valid"], dtype=bool) & test_finite

        tb_model = _fit_binary_model(
            x_train[tb_train_valid],
            np.asarray(train_tb["tb_down"][tb_train_valid], dtype=np.int64),
            max_train_samples=max_train_samples,
            seed=seed + 917,
        )
        if tb_model is not None:
            danger_train_scores = _score_binary(tb_model, x_train)
            danger_val_scores = _score_binary(tb_model, x_val)
            danger_test_scores = _score_binary(tb_model, x_test)
            finite_danger = danger_val_scores[tb_val_valid & np.isfinite(danger_val_scores)]
            if len(finite_danger) > 0:
                danger_auc = _safe_auc(
                    np.asarray(val_tb["tb_down"][tb_val_valid], dtype=np.int64),
                    danger_val_scores[tb_val_valid],
                )
                danger_caps = []
                for q in danger_cap_quantiles:
                    if q is not None:
                        danger_caps.append(float(np.quantile(finite_danger, q)))
                danger_caps.append(float("inf"))
                danger_caps.append(None)

    # ── Threshold grid from validation recovery scores ──
    thresholds_raw = _threshold_score_grid(rec_score_val[val_mask], active_cap=active_cap)
    thresholds = sorted({
        float("inf") if not math.isfinite(float(t)) else max(float(t), float(min_confidence_floor))
        for t in thresholds_raw
    })

    # ── Grid search on validation ──
    best_val_score = float("-inf")
    best_config: dict[str, Any] = {}
    best_val_positions: np.ndarray | None = None
    val_search_count = 0

    for threshold in thresholds:
        for danger_cap in danger_caps:
            danger_mask_val = np.ones(len(val_returns), dtype=bool)
            if (
                danger_val_scores is not None
                and danger_cap is not None
                and math.isfinite(float(danger_cap))
            ):
                danger_mask_val = np.isfinite(danger_val_scores) & (
                    danger_val_scores <= float(danger_cap)
                )

            for cooldown in cooldown_grid:
                val_search_count += 1

                # Recovery signal within vol_shock events, danger filter applied
                combined_mask = val_mask & danger_mask_val
                rec_pred_val = rec_score_val >= threshold
                selected_val = np.full(len(val_returns), bench, dtype=np.float64)
                selected_idx = np.flatnonzero(combined_mask & rec_pred_val)
                selected_val[selected_idx] = overlay_position

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
                score = _val_score(
                    alpha_excess_pt=float(val_metrics.get("alpha_excess_pt", 0.0)),
                    maxdd_delta_pt=float(val_metrics.get("maxdd_delta_pt", 0.0)),
                    turnover=float(val_metrics.get("turnover", 0.0)),
                    active_rate=active_rate,
                    maxdd_cap=maxdd_val_cap,
                    turnover_cap=turnover_cap,
                    active_cap=active_cap,
                )

                if score > best_val_score:
                    best_val_score = score
                    best_val_positions = val_positions.copy()
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

    danger_train_mask = np.ones(len(train_returns), dtype=bool)
    if (
        danger_train_scores is not None
        and danger_cap is not None
        and danger_cap != "inf"
        and math.isfinite(float(danger_cap))
    ):
        danger_train_mask = np.isfinite(danger_train_scores) & (
            danger_train_scores <= float(danger_cap)
        )

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

    combined_train_mask = train_mask & danger_train_mask
    rec_pred_train = rec_score_train >= threshold
    selected_train = np.full(len(train_returns), bench, dtype=np.float64)
    selected_idx = np.flatnonzero(combined_train_mask & rec_pred_train)
    selected_train[selected_idx] = overlay_position
    selected_train = _apply_event_throttle(
        selected_train,
        benchmark_position=bench,
        cooldown_bars=cooldown,
        hold_bars=1,
    )
    train_positions = _shift_for_execution(selected_train, bench)

    combined_test_mask = test_mask & danger_test_mask
    rec_pred_test = rec_score_test >= threshold
    selected_test = np.full(len(test_returns), bench, dtype=np.float64)
    selected_idx = np.flatnonzero(combined_test_mask & rec_pred_test)
    selected_test[selected_idx] = overlay_position

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

    # ── Candidate-aware statistics ──
    test_rec_true = np.asarray(test_recovery[test_mask], dtype=np.int64)
    test_rec_score_event = rec_score_test[test_mask]
    test_rec_finite = np.isfinite(test_rec_score_event)
    auc_event = _safe_auc(test_rec_true[test_rec_finite], test_rec_score_event[test_rec_finite])
    ap_event = _safe_ap(test_rec_true[test_rec_finite], test_rec_score_event[test_rec_finite])
    top10_precision = _top_fraction_mean(test_rec_score_event, test_rec_true.astype(np.float64), 0.10)

    # Event-level active rate (fraction of vol_shock events where we enter)
    rec_pred_event = rec_pred_test[test_mask] & danger_test_mask[test_mask]
    event_active_rate = float(np.mean(rec_pred_event)) if test_candidate_count > 0 else float("nan")

    return {
        "status": "ok",
        "spec_name": spec["name"],
        "spec_desc": spec["desc"],
        "use_tb_guard": use_tb_guard,
        "test_candidates": test_candidate_count,
        "positive_rate": float(np.mean(test_rec_true)) if test_candidate_count > 0 else float("nan"),
        "auc_event": auc_event,
        "ap_event": ap_event,
        "top10_precision": top10_precision,
        "event_active_rate": event_active_rate,
        "danger_auc": danger_auc,
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
        "_train_positions": train_positions,
        "_val_positions": best_val_positions if best_val_positions is not None else np.full(len(val_returns), bench),
        "_test_positions": test_positions,
        "test_pnl": _test_pnl,
    }


def _aggregate(
    results: dict[str, dict[str, Any]],
    fold_ids: list[int],
) -> dict[str, dict[str, Any]]:
    spec_names = sorted({
        row["spec_name"]
        for fold_rows in results.values()
        for row in fold_rows.get("specs", [])
        if row.get("status") == "ok"
    })
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
            "alpha_best": _nanmax(alphas),
            "maxdd_mean": _nanmean(maxdds),
            "maxdd_worst": _nanmax(maxdds),
            "turnover_mean": _nanmean(turnovers),
            "turnover_max": _nanmax(turnovers),
            "pass_alpha_gt0": int(np.sum([a > 0.0 for a in alphas])),
            "pass_maxdd_le0": int(np.sum([d <= 0.0 for d in maxdds])),
            "pass_both": int(np.sum([a > 0.0 and d <= 0.0 for a, d in zip(alphas, maxdds)])),
        }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Agent Pro Plan001 Guard Report",
        "",
        f"**Date**: {payload.get('date', 'unknown')}",
        f"**Experiment**: `{payload.get('experiment', EXPERIMENT_NAME)}`",
        f"**Config**: `{payload.get('config', 'N/A')}`",
        f"**Date range**: {payload.get('start', '?')} \\u2192 {payload.get('end', '?')}",
        f"**Folds evaluated**: `{', '.join(map(str, payload.get('folds', [])))}`",
        f"**Seed**: {payload.get('seed', '?')}",
        "",
        "---",
        "",
        "## Goal",
        "",
        "Maximize folds satisfying **AlphaEx > 0** AND **MaxDDDelta <= 0**,",
        "using a causal guard/selector on the best vol_shock/recovery/state candidate from round1_meta_label_probe.",
        "",
        "## Method",
        "",
        "1. Vol-shock event filter (high vol or large return bars).",
        "2. Primary model: binary logistic classifier predicting pullback-recovery label on state features.",
        "3. Guard options (varied per spec):",
        "   - Triple-barrier down danger model filters risky entries.",
        "   - Cooldown throttles prevent clustered trades.",
        "   - Minimum confidence floor raises the entry bar.",
        "4. Validation-only grid search: threshold, danger cap, cooldown.",
        "5. Score: `alpha - 10*max(0, maxdd) - 0.05*turnover`, hard veto if maxdd > epsilon.",
        "6. Apply best config once to test.",
        "",
        "---",
        "",
        "## Guard Specs",
        "",
        "| spec | description | TB guard | cooldown grid | conf floor |",
        "|---|---|---:|---:|",
    ]
    for spec in GUARD_SPECS:
        lines.append(
            f"| {spec['name']} | {spec['desc']} | {spec['use_tb_guard']} | {spec['cooldown_grid']} | {spec['min_confidence_floor']} |"
        )
    lines.append("")

    lines.extend([
        "---",
        "",
        "## Aggregate Results",
        "",
        "| spec | folds | \\u03b1 mean | \\u03b1 worst | \\u03b1 best | maxdd worst | turnover mean | turnover max | pass \\u03b1>0 | pass dd\\u22640 | pass both |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in payload.get("aggregate", {}).items():
        lines.append(
            "| "
            + " | ".join([
                name,
                str(row["folds"]),
                _fmt(row["alpha_mean"]),
                _fmt(row["alpha_worst"]),
                _fmt(row["alpha_best"]),
                _fmt(row["maxdd_worst"]),
                _fmt(row["turnover_mean"]),
                _fmt(row["turnover_max"]),
                str(row["pass_alpha_gt0"]),
                str(row["pass_maxdd_le0"]),
                str(row["pass_both"]),
            ])
            + " |"
        )
    lines.append("")

    lines.extend(["---", "", "## Per-Fold Detail", ""])
    for fold_id in payload.get("folds", []):
        lines.extend([f"### Fold {fold_id}", ""])
        lines.append(
            "| spec | \\u03b1 | maxdd | sharpe | turnover | long% | flat% | trades | event% | AUC | AP | val \\u03b1 | val maxdd | threshold | danger_cap | cd |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        fold_rows = payload.get("results", {}).get(str(fold_id), {}).get("specs", [])
        for row in fold_rows:
            if row.get("status") != "ok":
                continue
            test = row.get("test", {})
            sel = row.get("val_selection", {})
            lines.append(
                "| "
                + " | ".join([
                    row.get("spec_name", "?"),
                    _fmt(test.get("alpha_excess_pt")),
                    _fmt(test.get("maxdd_delta_pt")),
                    _fmt(test.get("sharpe_delta")),
                    _fmt(test.get("turnover")),
                    _fmt(test.get("long_rate")),
                    _fmt(test.get("flat_rate")),
                    str(test.get("n_trades", 0)),
                    _fmt(row.get("event_active_rate")),
                    _fmt(row.get("auc_event")),
                    _fmt(row.get("ap_event")),
                    _fmt(sel.get("val_alpha")),
                    _fmt(sel.get("val_maxdd")),
                    str(sel.get("threshold", "?")),
                    str(sel.get("danger_cap", "?")),
                    str(sel.get("cooldown_bars", "?")),
                ])
                + " |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Per-Fold Goal Assessment",
        "",
        "| fold | best spec | \\u03b1 | maxdd | turnover | goal |",
        "|---|---:|---:|---:|---:|",
    ])
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
            lines.append(f"| {fold_id} | none | NA | NA | NA | FAIL |")
            continue
        test = best_row["test"]
        a = float(test.get("alpha_excess_pt", 0.0))
        d = float(test.get("maxdd_delta_pt", 0.0))
        both = "**MET**" if (a > 0.0 and d <= 0.0) else "NOT MET"
        lines.append(
            f"| {fold_id} | {best_row['spec_name']} | {_fmt(a)} | {_fmt(d)} | {_fmt(test.get('turnover'))} | {both} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Epsilon Sensitivity: Does Strict MaxDDDelta Matter?",
        "",
    ])
    # Compare GR_tb_guard_cd vs GR_maxdd_epsilon_neg001
    lines.append("| spec | \\u03b1 mean | \\u03b1 worst | maxdd worst | pass both |")
    lines.append("|---|---:|---:|---:|---:|")
    for check_name in ("GR_tb_guard_cd", "GR_maxdd_epsilon_neg001"):
        row = payload.get("aggregate", {}).get(check_name, {})
        if row:
            lines.append(
                f"| {check_name} | {_fmt(row.get('alpha_mean'))} | {_fmt(row.get('alpha_worst'))} | {_fmt(row.get('maxdd_worst'))} | {row.get('pass_both', 0)}/{row.get('folds', 0)} |"
            )
    lines.extend([
        "",
        "If `GR_maxdd_epsilon_neg001` has similar or better pass counts, then the strict epsilon matters.",
        "",
        "---",
        "",
        "## Adoption / Next-Step Judgment",
        "",
    ])
    # Determine if the guard helps
    baseline = payload.get("aggregate", {}).get("GR_baseline", {})
    combined = payload.get("aggregate", {}).get("GR_combined", {})
    tb_guard = payload.get("aggregate", {}).get("GR_tb_guard", {})
    tb_guard_cd = payload.get("aggregate", {}).get("GR_tb_guard_cd", {})

    baseline_pass = int(baseline.get("pass_both", 0))
    best_pass = max(int(combined.get("pass_both", 0)), int(tb_guard.get("pass_both", 0)), int(tb_guard_cd.get("pass_both", 0)))
    best_name = ""
    if int(combined.get("pass_both", 0)) >= best_pass:
        best_name = "GR_combined"
        best_pass = int(combined.get("pass_both", 0))
    if int(tb_guard_cd.get("pass_both", 0)) >= best_pass:
        best_name = "GR_tb_guard_cd"
        best_pass = int(tb_guard_cd.get("pass_both", 0))
    if int(tb_guard.get("pass_both", 0)) >= best_pass:
        best_name = "GR_tb_guard"
        best_pass = int(tb_guard.get("pass_both", 0))

    lines.append(f"- **Baseline (GR_baseline)**: {baseline_pass}/{baseline.get('folds', 0)} folds pass both criteria.")
    if best_pass > baseline_pass:
        lines.append(f"- **Best guard (_{best_name}_)**: {best_pass} folds pass \\u2192 improvement of {best_pass - baseline_pass} fold(s).")
        lines.append("- **Verdict**: ADOPT guard mechanism; it improves worst-fold outcomes without destroying edge.")
    elif best_pass == baseline_pass:
        lines.append(f"- **Best guard (_{best_name}_)**: ties baseline at {best_pass} folds.")
        lines.append("- **Verdict**: Guard does not hurt; conditionally adopt for risk veto, but prioritize per-fold analysis.")
    else:
        lines.append(f"- **Best guard (_{best_name}_)**: {best_pass} folds \\u2192 WORSE than baseline ({baseline_pass}).")
        lines.append("- **Verdict**: DO NOT adopt guard as-is; investigate threshold overfitting.")

    lines.append(f"- **Strict MaxDDDelta epsilon matters**: {'Yes' if len([1 for s in ('GR_maxdd_epsilon_neg001',) if payload.get('aggregate', {}).get(s, {}).get('pass_both', 0) != baseline_pass]) else 'Minimal impact'} (compare GR_maxdd_epsilon_neg001 with GR_tb_guard_cd).")
    lines.extend([
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Scope: experimental CLI only; no config or production behavior changes.",
        "- All thresholds selected on validation using hard MaxDDDelta veto, then applied once to test.",
        "- The vol_shock/recovery/state pipeline is the best-performing candidate from round1_meta_label_probe.",
        "- See `docs_local/` for full JSON output and raw log.",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_log(path: str, payload: dict[str, Any]) -> None:
    lines = [
        f"Agent Pro Plan001 Guard Report Log",
        f"Date: {payload.get('date', 'unknown')}",
        f"Folds: {', '.join(map(str, payload.get('folds', [])))}",
        f"Seed: {payload.get('seed', '?')}",
        f"{'='*60}",
        "",
    ]
    for name, row in payload.get("aggregate", {}).items():
        lines.append(
            f"{name}: folds={row['folds']} alpha_mean={_fmt(row['alpha_mean'])} "
            f"alpha_worst={_fmt(row['alpha_worst'])} maxdd_worst={_fmt(row['maxdd_worst'])} "
            f"turnover_mean={_fmt(row['turnover_mean'])} "
            f"pass_alpha>0={row['pass_alpha_gt0']} pass_maxdd_le0={row['pass_maxdd_le0']} "
            f"pass_both={row['pass_both']}"
        )
    lines.append("")
    for fold_id in payload.get("folds", []):
        lines.append(f"--- Fold {fold_id} ---")
        fold_rows = payload.get("results", {}).get(str(fold_id), {}).get("specs", [])
        for row in fold_rows:
            if row.get("status") != "ok":
                lines.append(f"  {row.get('spec_name', '?')}: status={row.get('status')}")
                continue
            test = row.get("test", {})
            lines.append(
                f"  {row.get('spec_name', '?')}: "
                f"alpha={_fmt(test.get('alpha_excess_pt'))} "
                f"maxdd={_fmt(test.get('maxdd_delta_pt'))} "
                f"turnover={_fmt(test.get('turnover'))} "
                f"trades={test.get('n_trades', 0)}"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=f"python -m unidream.cli.{os.path.splitext(os.path.basename(__file__))[0]}"
    )
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spec-names", default="")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-log", default="")
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
    if not args.output_log:
        args.output_log = os.path.join(
            "docs_local", f"{date}_{EXPERIMENT_NAME}.log"
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
        s for s in GUARD_SPECS if not spec_filter or s["name"] in spec_filter
    ]
    if not specs:
        raise ValueError(f"No spec matched --spec-names={args.spec_names!r}")

    results: dict[str, dict[str, Any]] = {}
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[GuardRecovery] fold={fid} start")
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
            print(f"[GuardRecovery] fold={fid} spec={spec['name']}")
            row = _evaluate_guarded(
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
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    _write_log(args.output_log, payload)
    print(f"[GuardRecovery] wrote {args.output_json}, {args.output_md}, {args.output_log}")


if __name__ == "__main__":
    main()
