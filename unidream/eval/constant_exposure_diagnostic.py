"""Development-only audit of low-frequency constant exposure effects.

This screen is deliberately not a forecast experiment.  It selects one value
from the existing fixed exposure grid on each validation interval and applies
that value unchanged to the corresponding test interval.  The test interval
is report-only.  All portfolio calculations go through the shared
``backtest_metrics``/``Backtest`` contract so that the result is comparable to
the other UniDream reports.

The hard scope is folds 0--11 (twelve even WFO subperiods).  The loader reads
only the returns cache needed for this diagnostic; feature caches and model
artifacts are intentionally not used.  Rows at or after fold 11's exclusive
test end are discarded before any path or metric is constructed, so fold 12
and later development/holdout periods cannot enter the screen.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from unidream.data.dataset import WFOSplit, get_wfo_splits
from unidream.eval.alpha_attribution import (
    backtest_metrics,
    select_fixed_exposure,
    sha256_file,
)
from unidream.eval.backtest import (
    align_execution_path,
    compute_pnl,
    validate_execution_delay,
)
from unidream.eval.forecast_tournament import (
    CONSTANT_EXPOSURE_GRID,
    _assert_wfo_split_contract,
    _normalize_index,
)
from unidream.eval.statistical_gate import (
    CandidatePath,
    DevelopmentFold,
    StatisticalGateConfig,
    StressCase,
    bootstrap_confidence_intervals,
    compute_cscv_pbo,
    compute_deflated_sharpe,
    evaluate_stress,
    fold_sign_test,
)
from unidream.experiments.run_config import config_fingerprint, source_fingerprint


SCHEMA_VERSION = 1
WAVE_NAME = "wave3d_constant_exposure_diagnostic"
WAVE_FOLDS = tuple(range(12))
WAVE_SEED = 7
BENCHMARK_POSITION = 1.0
FIXED_EXECUTION_DELAY_BARS = 1
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_BLOCK_LENGTH = 16
BOOTSTRAP_BLOCK_SENSITIVITY = (8, 16, 32)
STATISTICAL_TRIALS = 7  # fixed grid six + one validation-selected adaptive path
MIN_POSITIVE_FOLDS = 8
MIN_SELECTION_MODE_FRACTION = 0.5
MAX_SELECTION_DISTINCT = 3


@dataclass(frozen=True)
class ConstantExposureData:
    """Returns-only cache truncated at the exclusive end of fold 11."""

    returns: pd.Series
    splits: tuple[WFOSplit, ...]
    returns_path: Path
    train_years: int
    val_months: int
    test_months: int
    evaluation_cutoff_exclusive: pd.Timestamp
    source_rows: int
    excluded_future_rows: int


def _strict_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def validate_wave3d_folds(folds: Iterable[int] | None = None) -> tuple[int, ...]:
    """Require every development subperiod exactly once, including fold 0/11."""
    values = WAVE_FOLDS if folds is None else tuple(folds)
    normalized = tuple(_strict_int("fold", value, minimum=0) for value in values)
    if len(normalized) != len(WAVE_FOLDS) or len(set(normalized)) != len(normalized):
        raise ValueError(
            f"Wave3D requires exact folds {list(WAVE_FOLDS)} once each; got {list(normalized)}"
        )
    if set(normalized) != set(WAVE_FOLDS):
        raise ValueError(
            f"Wave3D requires exact folds {list(WAVE_FOLDS)}; got {list(normalized)}"
        )
    return tuple(sorted(normalized))


def _normalize_returns(frame: pd.DataFrame | pd.Series, label: str) -> pd.Series:
    normalized = _normalize_index(frame, label)
    if isinstance(normalized, pd.DataFrame):
        if "returns" in normalized.columns:
            series = normalized["returns"]
        elif normalized.shape[1] == 1:
            series = normalized.iloc[:, 0]
        else:
            raise ValueError("returns cache must contain one column or a `returns` column")
    else:
        series = normalized
    series = series.astype(np.float64)
    if not np.isfinite(series.to_numpy(dtype=np.float64)).all():
        raise ValueError("returns cache contains non-finite values")
    series.name = "returns"
    return series


def _validate_periods(train_years: int, val_months: int, test_months: int) -> tuple[int, int, int]:
    periods = (
        _strict_int("train_years", train_years, minimum=1),
        _strict_int("val_months", val_months, minimum=1),
        _strict_int("test_months", test_months, minimum=1),
    )
    return periods


def load_constant_exposure_data(
    returns_path: str | Path,
    *,
    folds: Iterable[int] | None = None,
    train_years: int = 2,
    val_months: int = 3,
    test_months: int = 3,
) -> ConstantExposureData:
    """Load returns and construct the exact twelve right-exclusive WFO folds.

    ``get_wfo_splits`` is called on the complete index only to establish the
    boundaries.  The returned series is then truncated at fold 11's exclusive
    end before any evaluation.  No fold 12+ row can be passed to a metric.
    """
    requested = validate_wave3d_folds(folds)
    train_years, val_months, test_months = _validate_periods(
        train_years, val_months, test_months
    )
    path = Path(returns_path)
    returns = _normalize_returns(pd.read_parquet(path), "returns cache")
    generated = get_wfo_splits(
        returns.to_frame(),
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
    )
    for split in generated:
        _assert_wfo_split_contract(
            split,
            train_years=train_years,
            val_months=val_months,
            test_months=test_months,
        )
    by_fold = {split.fold_idx: split for split in generated}
    missing = sorted(set(requested) - set(by_fold))
    if missing:
        raise ValueError(f"returns cache is missing required Wave3D folds: {missing}")
    splits = tuple(by_fold[fold] for fold in requested)
    cutoff = pd.Timestamp(splits[-1].test_end)
    if any(pd.Timestamp(split.test_end) > cutoff for split in splits):
        raise ValueError("Wave3D fold ordering produced a future split beyond the cutoff")
    development_mask = returns.index < cutoff
    development_returns = returns.loc[np.asarray(development_mask, dtype=bool)].copy()
    if len(development_returns) == 0 or development_returns.index.max() < splits[-1].test_start:
        raise ValueError("returns cache ends before the required fold 11 test interval")
    return ConstantExposureData(
        returns=development_returns,
        splits=splits,
        returns_path=path,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
        evaluation_cutoff_exclusive=cutoff,
        source_rows=int(len(returns)),
        excluded_future_rows=int(np.sum(~np.asarray(development_mask, dtype=bool))),
    )


def _slice_right_exclusive(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, label: str) -> pd.Series:
    mask = (series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))
    result = series.loc[np.asarray(mask, dtype=bool)]
    if len(result) == 0:
        raise ValueError(f"empty right-exclusive {label} slice [{start}, {end})")
    return result


def configured_execution_delay(cfg: Mapping[str, Any]) -> int:
    """Read and enforce the fixed operational delay; no validation tuning."""
    evaluation = cfg.get("eval", {})
    configured = evaluation.get(
        "forecast_execution_delay_bars",
        cfg.get("forecast_execution_delay_bars", FIXED_EXECUTION_DELAY_BARS),
    )
    delay = validate_execution_delay(configured)
    if delay != FIXED_EXECUTION_DELAY_BARS:
        raise ValueError(
            "Wave3D fixes the operational execution delay at "
            f"{FIXED_EXECUTION_DELAY_BARS}; configured {delay} is not allowed"
        )
    return delay


def _cost_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    costs = cfg.get("costs", {})
    return {
        "spread_bps": float(costs.get("spread_bps", 5.0)),
        "fee_rate": float(costs.get("fee_rate", 0.0004)),
        "slippage_bps": float(costs.get("slippage_bps", 2.0)),
    }


def _net_paths(
    returns: np.ndarray,
    positions: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    execution_delay: int,
) -> dict[str, np.ndarray]:
    """Build aligned additive net paths with the canonical delay contract."""
    returns_arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    positions_arr = np.asarray(positions, dtype=np.float64).reshape(-1)
    benchmark = np.full(len(returns_arr), BENCHMARK_POSITION, dtype=np.float64)
    aligned_returns, aligned_positions, aligned_benchmark = align_execution_path(
        returns_arr,
        positions_arr,
        benchmark,
        execution_delay,
    )
    assert aligned_benchmark is not None
    costs = _cost_kwargs(cfg)
    strategy_pnl = compute_pnl(aligned_returns, aligned_positions, **costs)
    benchmark_pnl = compute_pnl(aligned_returns, aligned_benchmark, **costs)
    return {
        "returns": aligned_returns,
        "effective_positions": aligned_positions,
        "benchmark_positions": aligned_benchmark,
        "strategy_pnl": strategy_pnl,
        "benchmark_pnl": benchmark_pnl,
        "alpha_excess": strategy_pnl - benchmark_pnl,
    }


def _exposure_method(exposure: float) -> str:
    return f"fixed_{float(exposure):.3f}".replace(".", "p")


def _metric_summary(metrics: Mapping[str, Any], path: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result = {str(key): value for key, value in metrics.items()}
    result.update(
        {
            "path_alpha_excess_pt": 100.0 * float(np.sum(path["alpha_excess"])),
            "path_strategy_sum_pt": 100.0 * float(np.sum(path["strategy_pnl"])),
            "path_benchmark_sum_pt": 100.0 * float(np.sum(path["benchmark_pnl"])),
            "path_rows": int(len(path["returns"])),
        }
    )
    return result


def _run_fold(
    data: ConstantExposureData,
    split: WFOSplit,
    cfg: Mapping[str, Any],
    *,
    previous_exposure: float | None,
    execution_delay: int,
) -> dict[str, Any]:
    validation = _slice_right_exclusive(
        data.returns,
        split.val_start,
        split.val_end,
        f"fold {split.fold_idx} validation",
    )
    test = _slice_right_exclusive(
        data.returns,
        split.test_start,
        split.test_end,
        f"fold {split.fold_idx} test",
    )
    selected, selection_trials = select_fixed_exposure(
        validation.to_numpy(dtype=np.float64),
        CONSTANT_EXPOSURE_GRID,
        cfg,
        benchmark=BENCHMARK_POSITION,
        execution_delay_bars=execution_delay,
    )
    exposure_by_method: dict[str, float] = {"bnh": BENCHMARK_POSITION}
    exposure_by_method.update(
        {_exposure_method(exposure): float(exposure) for exposure in CONSTANT_EXPOSURE_GRID}
    )
    exposure_by_method["selected_constant"] = float(selected)
    if previous_exposure is not None:
        exposure_by_method["previous_fold_selected_constant"] = float(previous_exposure)

    test_values = test.to_numpy(dtype=np.float64)
    timestamps_ns = test.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)[execution_delay:]
    methods: dict[str, dict[str, Any]] = {}
    for method, exposure in exposure_by_method.items():
        positions = np.full(len(test_values), exposure, dtype=np.float64)
        metrics = backtest_metrics(
            test_values,
            positions,
            cfg,
            benchmark=BENCHMARK_POSITION,
            execution_delay_bars=execution_delay,
        )
        path = _net_paths(test_values, positions, cfg, execution_delay=execution_delay)
        if len(path["returns"]) != len(timestamps_ns):
            raise AssertionError("delay-aligned path and timestamp lengths differ")
        methods[method] = {
            "exposure": exposure,
            "metrics": _metric_summary(metrics, path),
            "path": path,
            "timestamps_ns": timestamps_ns,
        }
    return {
        "fold": int(split.fold_idx),
        "bounds": {
            "train_start": str(split.train_start),
            "train_end": str(split.train_end),
            "val_start": str(split.val_start),
            "val_end": str(split.val_end),
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
        },
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "selection": {
            "selected_exposure": float(selected),
            "candidate_grid": [float(value) for value in CONSTANT_EXPOSURE_GRID],
            "trials": selection_trials,
        },
        "methods": methods,
    }


def _aggregate_method(fold_results: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    rows = [item["methods"][method] for item in fold_results if method in item["methods"]]
    if not rows:
        return {"status": "N/A", "reason": f"no fold rows for {method}"}
    fields = (
        "alpha_excess_pt",
        "maxdd_delta_pt",
        "sharpe_delta",
        "turnover",
        "cost_turnover",
        "max_drawdown_pt",
        "n_trades",
        "path_alpha_excess_pt",
    )
    result: dict[str, Any] = {
        "status": "ok",
        "method": method,
        "folds": [int(item["fold"]) for item in fold_results if method in item["methods"]],
        "n_folds": len(rows),
    }
    for field in fields:
        values = np.asarray([float(row["metrics"][field]) for row in rows], dtype=np.float64)
        result[f"median_{field}"] = float(np.median(values))
        result[f"mean_{field}"] = float(np.mean(values))
        result[f"fold_{field}"] = [float(value) for value in values]
    alpha_values = np.asarray(
        [float(row["metrics"]["alpha_excess_pt"]) for row in rows], dtype=np.float64
    )
    result["positive_alpha_folds"] = int(np.sum(alpha_values > 0.0))
    result["nonpositive_alpha_folds"] = int(np.sum(alpha_values <= 0.0))
    return result


def _selection_stability(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [float(item["selection"]["selected_exposure"]) for item in fold_results]
    counts: dict[str, int] = {}
    for value in selected:
        key = f"{value:.6f}"
        counts[key] = counts.get(key, 0) + 1
    mode_count = max(counts.values()) if counts else 0
    return {
        "selected_exposures": selected,
        "counts": counts,
        "mode_exposure": float(max(selected, key=lambda value: (counts[f"{value:.6f}"], -value))) if selected else None,
        "mode_count": int(mode_count),
        "mode_fraction": float(mode_count / len(selected)) if selected else 0.0,
        "distinct_exposures": int(len(counts)),
        "switches": int(sum(left != right for left, right in zip(selected, selected[1:]))),
    }


def _selected_vs_previous(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare adaptive selection with the previous-fold selector on folds 1--11."""
    selected_rows: list[float] = []
    previous_rows: list[float] = []
    for item in fold_results:
        if int(item["fold"]) == WAVE_FOLDS[0]:
            continue
        selected_rows.append(float(item["methods"]["selected_constant"]["metrics"]["alpha_excess_pt"]))
        previous = item["methods"].get("previous_fold_selected_constant")
        if previous is not None:
            previous_rows.append(float(previous["metrics"]["alpha_excess_pt"]))
    if len(selected_rows) != len(previous_rows) or not selected_rows:
        return {
            "status": "N/A",
            "reason": "selected/prior-fold paths do not overlap on folds 1--11",
        }
    selected = np.asarray(selected_rows, dtype=np.float64)
    previous = np.asarray(previous_rows, dtype=np.float64)
    return {
        "status": "ok",
        "folds": list(WAVE_FOLDS[1:]),
        "selected_median_alpha_excess_pt": float(np.median(selected)),
        "previous_median_alpha_excess_pt": float(np.median(previous)),
        "selected_mean_alpha_excess_pt": float(np.mean(selected)),
        "previous_mean_alpha_excess_pt": float(np.mean(previous)),
        "selected_positive_folds": int(np.sum(selected > 0.0)),
        "previous_positive_folds": int(np.sum(previous > 0.0)),
        "selected_fold_alpha_excess_pt": selected.tolist(),
        "previous_fold_alpha_excess_pt": previous.tolist(),
    }


def _stat_config(seed: int) -> StatisticalGateConfig:
    return StatisticalGateConfig(
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        block_length_sensitivity=BOOTSTRAP_BLOCK_SENSITIVITY,
        seed=seed,
        min_folds=len(WAVE_FOLDS),
        min_observations=32,
        n_trials=STATISTICAL_TRIALS,
        cscv_subperiods=len(WAVE_FOLDS),
        require_cost_stress=True,
        require_regime_stress=True,
    )


def _development_folds_for_method(
    fold_results: Sequence[Mapping[str, Any]],
    method: str,
) -> tuple[DevelopmentFold, ...]:
    result: list[DevelopmentFold] = []
    for item in fold_results:
        method_row = item["methods"].get(method)
        if method_row is None:
            continue
        alpha = method_row["path"]["alpha_excess"]
        result.append(
            DevelopmentFold(
                fold=int(item["fold"]),
                alpha_excess_returns=alpha,
                timing_increment_returns=np.zeros_like(alpha),
                strategy_returns=method_row["path"]["strategy_pnl"],
            )
        )
    return tuple(result)


def _bootstrap_selected(fold_results: Sequence[Mapping[str, Any]], cfg: StatisticalGateConfig) -> dict[str, Any]:
    folds = _development_folds_for_method(fold_results, "selected_constant")
    if len(folds) != len(WAVE_FOLDS):
        return {"status": "N/A", "reason": "selected constant paths are incomplete"}
    raw = bootstrap_confidence_intervals(folds, cfg)
    return {
        "status": raw["status"],
        "primary": raw["primary"]["alpha_excess_pt"],
        "sensitivity": [
            {
                "block_length": int(item["block_length"]),
                "seed": int(item["seed"]),
                "alpha_excess_pt": item["alpha_excess_pt"],
            }
            for item in raw["sensitivity"]
        ],
        "block_length_sensitivity": raw["block_length_sensitivity"],
        "definition": "100 * sum of aligned additive strategy-minus-B&H net PnL; selected constant only",
        "timing_increment": {
            "status": "N/A",
            "reason": "a constant exposure has no temporal timing path; no timing alpha is claimed",
        },
    }


def _stress_cases(
    fold_results: Sequence[Mapping[str, Any]],
    data: ConstantExposureData,
    cfg: Mapping[str, Any],
) -> tuple[list[StressCase], dict[str, Any]]:
    """Build pre-registered cost and outcome-conditioned regime stress cases.

    Cost cases recompute the same aligned selected paths at 1x/1.5x/2x costs.
    Regime labels are determined after the report path is observed, but their
    volatility threshold is fit only on each fold's train+validation returns;
    they are diagnostic conditioning, never a selection input.
    """
    cases: list[StressCase] = []
    records: dict[str, Any] = {"cost": [], "regime": [], "definition": "report-only stress diagnostics"}
    base_costs = _cost_kwargs(cfg)
    for factor in (1.0, 1.5, 2.0):
        alpha_total = 0.0
        rows = 0
        for item in fold_results:
            selected = item["methods"]["selected_constant"]["path"]
            costs = {key: value * factor for key, value in base_costs.items()}
            strategy = compute_pnl(selected["returns"], selected["effective_positions"], **costs)
            benchmark = compute_pnl(selected["returns"], selected["benchmark_positions"], **costs)
            alpha_total += float(np.sum(strategy - benchmark))
            rows += len(strategy)
        case = StressCase(
            name=f"cost_{factor:g}x",
            kind="cost",
            alpha_excess_pt=100.0 * alpha_total,
            timing_increment_pt=0.0,
        )
        cases.append(case)
        records["cost"].append({"name": case.name, "rows": rows, "alpha_excess_pt": case.alpha_excess_pt})

    regime_names = ("positive_return", "negative_return", "high_volatility")
    regime_values: dict[str, list[float]] = {name: [] for name in regime_names}
    regime_counts: dict[str, int] = {name: 0 for name in regime_names}
    for item in fold_results:
        split = next(split for split in data.splits if split.fold_idx == int(item["fold"]))
        reference = _slice_right_exclusive(
            data.returns,
            split.train_start,
            split.val_end,
            f"fold {item['fold']} train+validation stress reference",
        ).to_numpy(dtype=np.float64)
        threshold = float(np.quantile(np.abs(reference), 0.75))
        selected = item["methods"]["selected_constant"]["path"]
        returns = selected["returns"]
        alpha = selected["alpha_excess"]
        masks = {
            "positive_return": returns > 0.0,
            "negative_return": returns <= 0.0,
            "high_volatility": np.abs(returns) >= threshold,
        }
        for name, mask in masks.items():
            if np.any(mask):
                regime_values[name].append(float(np.sum(alpha[mask])))
                regime_counts[name] += int(np.sum(mask))
    for name in regime_names:
        values = regime_values[name]
        if not values:
            records["regime"].append({"name": name, "status": "N/A", "reason": "no eligible test bars"})
            continue
        case = StressCase(
            name=f"regime_{name}",
            kind="regime",
            alpha_excess_pt=100.0 * float(np.sum(values)),
            timing_increment_pt=0.0,
        )
        cases.append(case)
        records["regime"].append(
            {
                "name": case.name,
                "rows": regime_counts[name],
                "alpha_excess_pt": case.alpha_excess_pt,
                "threshold_source": "fold train+validation abs-return 75th percentile",
            }
        )
    return cases, records


def _statistical_diagnostics(
    fold_results: Sequence[Mapping[str, Any]],
    data: ConstantExposureData,
    cfg: StatisticalGateConfig,
    runtime_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_methods = [_exposure_method(value) for value in CONSTANT_EXPOSURE_GRID]
    candidates: list[CandidatePath] = []
    for method in fixed_methods:
        folds = _development_folds_for_method(fold_results, method)
        if len(folds) != len(WAVE_FOLDS):
            continue
        candidates.append(CandidatePath(method, folds))
    selected_folds = _development_folds_for_method(fold_results, "selected_constant")
    if len(selected_folds) != len(WAVE_FOLDS):
        dsr: dict[str, Any] = {"status": "N/A", "passed": False, "reason": "selected paths incomplete"}
        sign: dict[str, Any] = {"status": "N/A", "passed": False, "reason": "selected paths incomplete"}
    else:
        selected_totals = np.asarray(
            [100.0 * float(np.sum(item.alpha_excess_returns)) for item in selected_folds],
            dtype=np.float64,
        )
        sign = fold_sign_test(
            selected_totals,
            alpha=cfg.alpha,
            min_folds=len(WAVE_FOLDS),
            label="selected constant alpha fold totals",
        )
        dsr_candidates = [*candidates, CandidatePath("selected_constant", selected_folds)]
        if dsr_candidates:
            dsr = compute_deflated_sharpe(dsr_candidates, "selected_constant", cfg)
        else:
            dsr = {"status": "N/A", "passed": False, "reason": "fixed candidate paths incomplete"}
    pbo = compute_cscv_pbo(candidates, cfg) if candidates else {
        "status": "N/A",
        "passed": False,
        "reason": "fixed candidate paths incomplete",
    }
    stress_cases, stress_records = _stress_cases(fold_results, data, runtime_cfg)
    stress = evaluate_stress(stress_cases, cfg)
    return {
        "selected_alpha_sign_test": sign,
        "deflated_sharpe": {
            **dsr,
            "selected_candidate": "selected_constant",
            "note": "DSR is a diagnostic over fixed grid plus the adaptive selector; it is not used to select exposure",
        },
        "cscv_pbo": {
            **pbo,
            "note": "CSCV/PBO is report-only over twelve test subperiods and never feeds validation selection",
        },
        "stress": {
            **stress,
            "records": stress_records,
            "note": "cost paths recompute the same selected decisions; regime thresholds use train+validation only",
        },
    }


def _evaluate_gate(
    fold_results: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    stability: Mapping[str, Any],
    selected_vs_previous: Mapping[str, Any],
) -> dict[str, Any]:
    selected = aggregates.get("selected_constant", {})
    primary_ci = bootstrap.get("primary", {}).get("lower_pt") if isinstance(bootstrap.get("primary"), Mapping) else None
    sensitivity_cis = []
    for item in bootstrap.get("sensitivity", []):
        if isinstance(item, Mapping):
            ci = item.get("alpha_excess_pt")
            if isinstance(ci, Mapping):
                sensitivity_cis.append(ci.get("lower_pt"))
    criteria: dict[str, bool] = {
        "exact_development_folds": [int(item["fold"]) for item in fold_results] == list(WAVE_FOLDS),
        "unique_development_folds": len({int(item["fold"]) for item in fold_results}) == len(WAVE_FOLDS),
        "validation_selection_complete": len(fold_results) == len(WAVE_FOLDS)
        and all("selected_exposure" in item["selection"] for item in fold_results),
        "selected_median_alpha_positive": float(selected.get("median_alpha_excess_pt", 0.0)) > 0.0,
        "selected_positive_folds_at_least_8": int(selected.get("positive_alpha_folds", 0)) >= MIN_POSITIVE_FOLDS,
        "selected_bootstrap_ci_lower_positive": primary_ci is not None and float(primary_ci) > 0.0,
        "selected_bootstrap_sensitivity_lower_positive": bool(sensitivity_cis)
        and all(value is not None and float(value) > 0.0 for value in sensitivity_cis),
        "selected_median_maxdd_delta_nonpositive": float(selected.get("median_maxdd_delta_pt", 0.0)) <= 0.0,
        "selection_mode_fraction_at_least_0.5": float(stability.get("mode_fraction", 0.0)) >= MIN_SELECTION_MODE_FRACTION,
        "selection_distinct_at_most_3": int(stability.get("distinct_exposures", 0)) <= MAX_SELECTION_DISTINCT,
        "selected_median_superior_to_previous": selected_vs_previous.get("status") == "ok"
        and float(selected_vs_previous.get("selected_median_alpha_excess_pt", 0.0))
        > float(selected_vs_previous.get("previous_median_alpha_excess_pt", 0.0)),
        "selected_mean_superior_to_previous": selected_vs_previous.get("status") == "ok"
        and float(selected_vs_previous.get("selected_mean_alpha_excess_pt", 0.0))
        > float(selected_vs_previous.get("previous_mean_alpha_excess_pt", 0.0)),
        "selected_positive_folds_not_lower_than_previous": selected_vs_previous.get("status") == "ok"
        and int(selected_vs_previous.get("selected_positive_folds", 0))
        >= int(selected_vs_previous.get("previous_positive_folds", 0)),
        "dsr_contract": bool(diagnostics.get("deflated_sharpe", {}).get("passed", False)),
        "cscv_contract": bool(diagnostics.get("cscv_pbo", {}).get("passed", False)),
        "stress_contract": bool(diagnostics.get("stress", {}).get("passed", False)),
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "passed": not failed,
        "promotion_eligible": False,
        "criteria": criteria,
        "failed_criteria": failed,
        "thresholds": {
            "positive_folds": MIN_POSITIVE_FOLDS,
            "bootstrap_lower_pt": 0.0,
            "median_alpha_excess_pt": 0.0,
            "median_maxdd_delta_pt": 0.0,
            "selection_mode_fraction": MIN_SELECTION_MODE_FRACTION,
            "selection_distinct_exposures": MAX_SELECTION_DISTINCT,
            "selected_vs_previous": "median and mean AlphaEx strictly higher; positive-fold count no lower on folds 1..11",
        },
        "interpretation": (
            "Even if every baseline criterion passes, this is a low-frequency constant-exposure "
            "diagnostic and cannot promote a forecast/model candidate."
        ),
    }


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        normalized = float(value)
        return normalized if np.isfinite(normalized) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_path_artifact(
    output_dir: Path,
    fold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "constant_exposure_paths.npz"
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for item in fold_results:
        fold = int(item["fold"])
        for method, method_row in item["methods"].items():
            path = method_row["path"]
            prefix = f"fold_{fold:02d}_{method}"
            keys = {
                "timestamps_ns": f"{prefix}__timestamps_ns",
                "returns": f"{prefix}__returns",
                "effective_positions": f"{prefix}__effective_positions",
                "strategy_pnl": f"{prefix}__strategy_pnl",
                "benchmark_pnl": f"{prefix}__benchmark_pnl",
                "alpha_excess": f"{prefix}__alpha_excess",
            }
            arrays[keys["timestamps_ns"]] = np.asarray(method_row["timestamps_ns"], dtype=np.int64)
            for field in keys:
                if field == "timestamps_ns":
                    continue
                arrays[keys[field]] = np.asarray(path[field], dtype=np.float64)
            timestamps = arrays[keys["timestamps_ns"]]
            entries.append(
                {
                    "fold": fold,
                    "method": method,
                    "exposure": float(method_row["exposure"]),
                    "rows": int(len(timestamps)),
                    "keys": keys,
                    "first_timestamp": str(pd.Timestamp(timestamps[0], unit="ns")),
                    "last_timestamp": str(pd.Timestamp(timestamps[-1], unit="ns")),
                }
            )
    np.savez_compressed(npz_path, **arrays)
    index_path = output_dir / "constant_exposure_paths.json"
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "wave": WAVE_NAME,
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "entries": entries,
        "definition": "effective positions are decision positions[:-1] scored on returns[1:] under fixed delay=1",
    }
    index_path.write_text(
        json.dumps(_json_safe(index_payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": index_payload["npz_sha256"],
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "entries": len(entries),
    }


def _base_provenance(
    data: ConstantExposureData,
    cfg: Mapping[str, Any],
    config_path: str,
    *,
    seed: int,
    execution_delay: int,
) -> dict[str, Any]:
    data_contract = {
        "returns_path": str(data.returns_path.resolve()),
        "returns_sha256": sha256_file(data.returns_path),
        "source_rows": data.source_rows,
        "evaluated_rows": int(len(data.returns)),
        "excluded_future_rows": data.excluded_future_rows,
        "evaluation_cutoff_exclusive": str(data.evaluation_cutoff_exclusive),
        "wfo": {
            "train_years": data.train_years,
            "val_months": data.val_months,
            "test_months": data.test_months,
            "folds": list(WAVE_FOLDS),
        },
        "features_used": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "wave": WAVE_NAME,
        "diagnostic_type": "low_frequency_constant_exposure_baseline",
        "git_commit": _git_commit(),
        "source_sha256": source_fingerprint(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_fingerprint(dict(cfg)),
        "data_contract": data_contract,
        "data_contract_sha256": _canonical_hash(data_contract),
        "seed": int(seed),
        "folds": list(WAVE_FOLDS),
        "selection_split": "validation",
        "report_split": "development_test_report_only",
        "fit_split": "none_returns_only",
        "candidate_grid": [float(value) for value in CONSTANT_EXPOSURE_GRID],
        "benchmark_position": BENCHMARK_POSITION,
        "execution_delay_bars": execution_delay,
        "cost_contract": _cost_kwargs(cfg),
        "holdout_policy": "fold 12+ and folds 15+ are excluded; no holdout data informs selection",
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _ledger_rows(
    provenance: Mapping[str, Any],
    fold_results: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Mapping[str, Any]],
    stability: Mapping[str, Any],
    selected_vs_previous: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    gate: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    result_path: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "wave": WAVE_NAME,
        "git_commit": provenance.get("git_commit"),
        "config_sha256": provenance.get("config_sha256"),
        "data_contract_sha256": provenance.get("data_contract_sha256"),
        "seed": provenance.get("seed"),
        "folds": list(WAVE_FOLDS),
        "selection_split": "validation",
        "report_split": "development_test_report_only",
        "artifact_paths": {
            **{str(key): value for key, value in artifacts.items() if str(key).endswith("path")},
            "result_path": str(result_path.resolve()),
            "report_path": str(report_path.resolve()),
        },
    }
    rows: list[dict[str, Any]] = [{
        **common,
        "row_type": "run",
        "status": "complete",
        "diagnostic_type": "low_frequency_constant_exposure_baseline",
        "note": "not a forecast accuracy or model-promotion result",
    }]
    for item in fold_results:
        rows.append(
            {
                **common,
                "row_type": "validation_selection",
                "status": "ok",
                "fold": int(item["fold"]),
                "fold_bounds": item["bounds"],
                "selected_exposure": item["selection"]["selected_exposure"],
                "candidate_grid": item["selection"]["candidate_grid"],
                "validation_rows": item["validation_rows"],
                "selection_trials": item["selection"]["trials"],
            }
        )
        for method, method_row in item["methods"].items():
            rows.append(
                {
                    **common,
                    "row_type": "test_method",
                    "status": "ok",
                    "fold": int(item["fold"]),
                    "method": method,
                    "exposure": method_row["exposure"],
                    "metrics": method_row["metrics"],
                    "test_rows": item["test_rows"],
                }
            )
        if "previous_fold_selected_constant" not in item["methods"]:
            rows.append(
                {
                    **common,
                    "row_type": "test_method",
                    "status": "N/A",
                    "fold": int(item["fold"]),
                    "method": "previous_fold_selected_constant",
                    "reason": "fold 0 has no previous validation-selected exposure",
                }
            )
    for method, aggregate in aggregates.items():
        rows.append({**common, "row_type": "aggregate", "status": aggregate.get("status"), "method": method, "metrics": aggregate})
    rows.extend(
        [
            {**common, "row_type": "selection_stability", "status": "ok", "metrics": stability},
            {**common, "row_type": "selected_vs_previous", "status": selected_vs_previous.get("status"), "metrics": selected_vs_previous},
            {**common, "row_type": "bootstrap", "status": bootstrap.get("status"), "metrics": bootstrap},
            {**common, "row_type": "statistical_diagnostics", "status": "ok", "metrics": diagnostics},
            {**common, "row_type": "gate", "status": gate.get("status"), "metrics": gate},
        ]
    )
    return [_json_safe(row) for row in rows]


def _write_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return "N/A"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _render_report(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    aggregates = result["aggregates"]
    lines = [
        "# Wave3D constant-exposure development diagnostic",
        "",
        "- Status: **{}** (promotion eligible: `{}`)".format(gate["status"], gate["promotion_eligible"]),
        "- Diagnostic type: `low_frequency_constant_exposure_baseline` (low-frequency constant-exposure baseline)",
        "- This is not a forecast-accuracy result and cannot promote a model; Wave3C forecast candidates remain a separate failed screen.",
        "",
        "## Scope and fixed contract",
        "",
        f"- Exact WFO folds: `{result['folds']}`; each appears once. Fold 12 and folds 15+ are excluded.",
        f"- Selection: fixed grid `{result['candidate_grid']}` on validation only; test is report-only.",
        f"- Benchmark: constant B&H position `{result['benchmark_position']}`.",
        f"- Execution delay: fixed `{result['execution_delay_bars']}` bar; no delay tuning.",
        f"- Costs: `{result['cost_contract']}`.",
        f"- Returns cache rows `{result['data_contract']['source_rows']}`; evaluated rows `{result['data_contract']['evaluated_rows']}`; excluded at cutoff `{result['data_contract']['excluded_future_rows']}`.",
        f"- Exclusive evaluation cutoff: `{result['data_contract']['evaluation_cutoff_exclusive']}`.",
        "- Features and model artifacts were not used; this screen makes no external-feature quality claim.",
        "",
        "## Fold and exposure results",
        "",
        "| method | folds | median AlphaEx (pt) | positive folds | median MaxDDDelta (pt) | median turnover |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in aggregates.items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                method,
                row.get("n_folds", 0),
                _fmt(row.get("median_alpha_excess_pt")),
                row.get("positive_alpha_folds", "N/A"),
                _fmt(row.get("median_maxdd_delta_pt")),
                _fmt(row.get("median_turnover")),
            )
        )
    stability = result["selection_stability"]
    diagnostics = result["statistical_diagnostics"]
    bootstrap = result["bootstrap"]
    lines.extend(
        [
            "",
            "Selected validation exposures: `{}`; mode `{}` (fraction `{}`); distinct `{}`; switches `{}`.".format(
                stability["selected_exposures"],
                stability["mode_exposure"],
                _fmt(stability["mode_fraction"]),
                stability["distinct_exposures"],
                stability["switches"],
            ),
            "",
            "## Statistical diagnostics",
            "",
            "- Selected-constant block-bootstrap additive AlphaEx CI: `{}`; sensitivity: `{}`.".format(
                bootstrap.get("primary"), bootstrap.get("sensitivity")
            ),
            "- Selected-constant fold sign test: `{}`.".format(diagnostics["selected_alpha_sign_test"]),
            "- DSR: `{}`; fixed `n_trials={}`.".format(diagnostics["deflated_sharpe"], STATISTICAL_TRIALS),
            "- CSCV/PBO: `{}` over twelve even test subperiods; report-only and not used for selection.".format(diagnostics["cscv_pbo"]),
            "- Stress diagnostics: `{}`; cost cases recompute 1x/1.5x/2x costs and regime thresholds use train+validation only.".format(
                diagnostics["stress"]
            ),
            "",
            "## Gate decision",
            "",
            f"- Passed criteria: `{[name for name, passed in gate['criteria'].items() if passed]}`.",
            f"- Failed criteria: `{gate['failed_criteria']}`.",
            "- A passing constant baseline would remain a low-frequency allocation finding, not evidence of predictive precision or a reason to advance DLinear/Transformer/RL.",
            "",
            "## Artifacts and provenance",
            "",
            f"- Result: `{result['result_path']}`",
            f"- Ledger: `{result['ledger_path']}`",
            f"- Per-bar NPZ: `{result['path_artifacts']['npz_path']}`",
            f"- Per-bar index: `{result['path_artifacts']['index_path']}`",
            f"- Config SHA256: `{result['config_sha256']}`",
            f"- Data-contract SHA256: `{result['data_contract_sha256']}`",
            f"- Git commit at run: `{result['git_commit']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_constant_exposure_diagnostic(
    *,
    data: ConstantExposureData,
    cfg: Mapping[str, Any],
    config_path: str,
    seed: int = WAVE_SEED,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one deterministic validation-select/test-report diagnostic."""
    seed = _strict_int("seed", seed, minimum=0)
    requested_delay = configured_execution_delay(cfg)
    if tuple(split.fold_idx for split in data.splits) != WAVE_FOLDS:
        raise ValueError("ConstantExposureData must contain exact Wave3D folds 0..11")
    started = time.perf_counter()
    fold_results: list[dict[str, Any]] = []
    previous_exposure: float | None = None
    for split in data.splits:
        fold_result = _run_fold(
            data,
            split,
            cfg,
            previous_exposure=previous_exposure,
            execution_delay=requested_delay,
        )
        fold_results.append(fold_result)
        previous_exposure = float(fold_result["selection"]["selected_exposure"])
    aggregates: dict[str, dict[str, Any]] = {}
    methods = ["bnh", *[_exposure_method(value) for value in CONSTANT_EXPOSURE_GRID], "selected_constant", "previous_fold_selected_constant"]
    for method in methods:
        aggregates[method] = _aggregate_method(fold_results, method)
    stability = _selection_stability(fold_results)
    stat_cfg = _stat_config(seed)
    bootstrap = _bootstrap_selected(fold_results, stat_cfg)
    diagnostics = _statistical_diagnostics(fold_results, data, stat_cfg, cfg)
    path_artifacts = _write_path_artifact(Path(output_dir), fold_results)
    provenance = _base_provenance(
        data,
        cfg,
        config_path,
        seed=seed,
        execution_delay=requested_delay,
    )
    selected_vs_previous = _selected_vs_previous(fold_results)
    gate = _evaluate_gate(
        fold_results,
        aggregates,
        bootstrap,
        diagnostics,
        stability,
        selected_vs_previous,
    )
    output_root = Path(output_dir)
    result_path = output_root / "result.json"
    ledger_path = output_root / "ledger.jsonl"
    report_path = output_root / "report.md"
    result: dict[str, Any] = {
        **provenance,
        "status": "complete",
        "runtime_seconds": float(time.perf_counter() - started),
        "splits": [item["bounds"] for item in fold_results],
        "validation_selection": [
            {
                "fold": int(item["fold"]),
                "selected_exposure": item["selection"]["selected_exposure"],
                "validation_rows": item["validation_rows"],
            }
            for item in fold_results
        ],
        "aggregates": aggregates,
        "selection_stability": stability,
        "selected_vs_previous": selected_vs_previous,
        "bootstrap": bootstrap,
        "statistical_diagnostics": diagnostics,
        "gate": gate,
        "path_artifacts": path_artifacts,
        "result_path": str(result_path.resolve()),
        "ledger_path": str(ledger_path.resolve()),
        "report_path": str(report_path.resolve()),
        "next_wave_candidates": [],
        "fold12_or_later_evaluated": False,
    }
    ledger_rows = _ledger_rows(
        provenance,
        fold_results,
        aggregates,
        stability,
        selected_vs_previous,
        bootstrap,
        diagnostics,
        gate,
        path_artifacts,
        result_path=result_path,
        report_path=report_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_ledger(ledger_path, ledger_rows)
    result["ledger_sha256"] = sha256_file(ledger_path)
    result_path.write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result


__all__ = [
    "BENCHMARK_POSITION",
    "BOOTSTRAP_BLOCK_LENGTH",
    "BOOTSTRAP_BLOCK_SENSITIVITY",
    "BOOTSTRAP_REPLICATES",
    "ConstantExposureData",
    "FIXED_EXECUTION_DELAY_BARS",
    "SCHEMA_VERSION",
    "STATISTICAL_TRIALS",
    "WAVE_FOLDS",
    "WAVE_NAME",
    "configured_execution_delay",
    "load_constant_exposure_data",
    "run_constant_exposure_diagnostic",
    "validate_wave3d_folds",
]
