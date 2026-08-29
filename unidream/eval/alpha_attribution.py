"""Leak-aware alpha attribution and saved-artifact diagnostics.

This module evaluates already materialized position paths.  It deliberately
delegates portfolio metrics to :class:`unidream.eval.backtest.Backtest` and
position summaries to :func:`unidream.eval.policy_stats.action_stats` so that
the attribution report uses the same cost, benchmark, and execution-delay
definitions as the training reports.

The evaluator never tunes on a test path.  Optional validation paths are used
only to select a fixed exposure or an actor-mean constant; when the historical
artifact has no validation positions, the actor-mean row is marked
``diagnostic_only_test_mean`` and is not a selected model.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
)

from unidream.eval.backtest import Backtest
from unidream.eval.policy_stats import action_stats
from unidream.experiments.run_config import config_fingerprint, source_fingerprint


SCHEMA_VERSION = 1
DEFAULT_FIXED_EXPOSURES = (1.0, 1.005, 1.01, 1.015)
DEFAULT_LAGS = (1, 4, 16)
DEFAULT_NULL_SHIFTS = (1, 16, 64)
DEFAULT_EXTERNAL_FEATURES = ("funding_rate", "basis", "basis_mom", "basis_abs")


@dataclass(frozen=True)
class FoldSeries:
    """One saved test position path and its aligned return path."""

    fold: int
    timestamps: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    test_start: str | None = None
    test_end: str | None = None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: str | Path) -> str:
    """Return a content hash for an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def right_exclusive_mask(values: Iterable[Any], start: Any, end: Any) -> np.ndarray:
    """Build the WFO mask ``start <= t < end`` used by the research pipeline."""
    values_arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if np.issubdtype(values_arr.dtype, np.number):
        if isinstance(start, (str, pd.Timestamp)) or isinstance(end, (str, pd.Timestamp)):
            values_arr = pd.to_datetime(values_arr.astype(np.int64), unit="ns")
            return (values_arr >= pd.Timestamp(start)) & (values_arr < pd.Timestamp(end))
        return (values_arr >= start) & (values_arr < end)
    parsed = pd.to_datetime(values_arr)
    return (parsed >= pd.Timestamp(start)) & (parsed < pd.Timestamp(end))


def _timestamp_index(values: np.ndarray) -> pd.DatetimeIndex:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return pd.DatetimeIndex(pd.to_datetime(arr.astype(np.int64), unit="ns"))
    return pd.DatetimeIndex(pd.to_datetime(arr))


def _summary_fold_bounds(path: Path) -> dict[int, tuple[str | None, str | None]]:
    summary_path = path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    bounds: dict[int, tuple[str | None, str | None]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, Mapping) or "fold" not in row:
            continue
        bounds[int(row["fold"])] = (
            str(row["test_start"]) if row.get("test_start") is not None else None,
            str(row["test_end"]) if row.get("test_end") is not None else None,
        )
    return bounds


def _finite_vector(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def load_timeseries_artifact(
    path: str | Path,
    folds: Iterable[int] | None = None,
) -> list[FoldSeries]:
    """Load the persisted ``fold_XX_*`` time-series artifact schema."""
    artifact_path = Path(path)
    requested = None if folds is None else {int(value) for value in folds}
    bounds = _summary_fold_bounds(artifact_path)
    loaded: list[FoldSeries] = []
    with np.load(artifact_path, allow_pickle=False) as data:
        fold_ids = sorted(
            int(key.split("_")[1])
            for key in data.files
            if key.startswith("fold_") and key.endswith("_returns")
        )
        if requested is not None:
            missing = sorted(requested - set(fold_ids))
            if missing:
                raise ValueError(f"artifact {artifact_path} is missing requested folds: {missing}")
            fold_ids = [fold for fold in fold_ids if fold in requested]
        if not fold_ids:
            raise ValueError(f"artifact {artifact_path} contains no requested fold paths")

        for fold in fold_ids:
            prefix = f"fold_{fold:02d}"
            required = [
                f"{prefix}_time_ns",
                f"{prefix}_returns",
                f"{prefix}_positions",
            ]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"artifact fold {fold} is missing keys: {missing}")
            timestamps = np.asarray(data[f"{prefix}_time_ns"])
            returns = _finite_vector(f"fold {fold} returns", data[f"{prefix}_returns"])
            positions = _finite_vector(f"fold {fold} positions", data[f"{prefix}_positions"])
            if len(timestamps) != len(returns) or len(returns) != len(positions):
                raise ValueError(f"artifact fold {fold} has unaligned time/return/position lengths")
            index = _timestamp_index(timestamps)
            if not index.is_monotonic_increasing or not index.is_unique:
                raise ValueError(f"artifact fold {fold} timestamps are not sorted and unique")
            test_start, test_end = bounds.get(fold, (None, None))
            if test_start is not None and test_end is not None:
                mask = right_exclusive_mask(index, test_start, test_end)
                # Historical plot artifacts contain the first boundary bar
                # of the following split (t == test_end).  It is not part of
                # the test fold under the research contract, so drop exactly
                # that one endpoint before any metric is computed.
                if not bool(np.all(mask)):
                    end_timestamp = pd.Timestamp(test_end)
                    if bool(index[-1] == end_timestamp) and bool(np.all(mask[:-1])):
                        timestamps = timestamps[:-1]
                        returns = returns[:-1]
                        positions = positions[:-1]
                        index = index[:-1]
                        mask = mask[:-1]
                if not bool(np.all(mask)):
                    raise ValueError(
                        f"artifact fold {fold} violates right-exclusive test bounds "
                        f"[{test_start}, {test_end})"
                    )
            loaded.append(
                FoldSeries(
                    fold=fold,
                    timestamps=timestamps,
                    returns=returns,
                    positions=positions,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
    return loaded


def data_fingerprint_from_series(series: Iterable[FoldSeries]) -> str:
    """Hash aligned fold data and metadata deterministically."""
    digest = hashlib.sha256()
    for item in sorted(series, key=lambda value: value.fold):
        digest.update(str(int(item.fold)).encode("ascii"))
        digest.update(b"\0")
        for value in (item.timestamps, item.returns, item.positions):
            arr = np.ascontiguousarray(value)
            digest.update(str(arr.dtype).encode("ascii"))
            digest.update(repr(arr.shape).encode("ascii"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible contract without relying on dict insertion order."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_column_coverage(values: Any, *, rows: int) -> dict[str, Any]:
    """Summarize a numeric feature while preserving missing/zero ambiguity."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(numeric)
    missing = ~finite
    nonzero = finite & (np.abs(numeric) > 1e-12)
    zero = finite & ~nonzero
    if rows == 0:
        quality_flag = "N/A_no_fold_rows"
        status = "N/A"
    elif not np.any(finite):
        quality_flag = "N/A_no_finite_values"
        status = "N/A"
    else:
        # The cache schema does not carry an availability mask.  A zero can be
        # a real observation or an imputed/unavailable value, so expose this
        # as a quality flag rather than silently treating it as valid signal.
        # Even an all-nonzero slice cannot prove that the cache would have
        # represented an unavailable value differently: the schema has no
        # availability mask.  Keep the limitation explicit for every present
        # column; observed zero/missing counts remain separately measurable.
        quality_flag = "N/A_zero_vs_missing_indistinguishable"
        status = "ok_with_quality_flag"
    denominator = max(int(rows), 1)
    return {
        "rows": int(rows),
        "finite_count": int(np.sum(finite)),
        "missing_count": int(np.sum(missing)),
        "zero_count": int(np.sum(zero)),
        "nonzero_count": int(np.sum(nonzero)),
        "finite_rate": float(np.sum(finite) / denominator) if rows else None,
        "missing_rate": float(np.sum(missing) / denominator) if rows else None,
        "zero_rate": float(np.sum(zero) / denominator) if rows else None,
        "nonzero_rate": float(np.sum(nonzero) / denominator) if rows else None,
        "status": status,
        "quality_flag": quality_flag,
    }


def _feature_coverage_for_frame(
    frame: pd.DataFrame,
    *,
    rows: int,
    required_external: Iterable[str],
) -> dict[str, Any]:
    columns: dict[str, dict[str, Any]] = {}
    for name in map(str, frame.columns):
        columns[name] = _feature_column_coverage(frame[name].to_numpy(), rows=rows)
    external: dict[str, dict[str, Any]] = {}
    missing_external: list[str] = []
    for name in map(str, required_external):
        if name not in frame.columns:
            missing_external.append(name)
            external[name] = {
                "rows": int(rows),
                "finite_count": 0,
                "missing_count": int(rows),
                "zero_count": 0,
                "nonzero_count": 0,
                "finite_rate": 0.0 if rows else None,
                "missing_rate": 1.0 if rows else None,
                "zero_rate": 0.0 if rows else None,
                "nonzero_rate": 0.0 if rows else None,
                "status": "N/A",
                "quality_flag": "N/A_missing_column",
            }
        else:
            external[name] = columns[name]
    quality_flags = sorted({str(value["quality_flag"]) for value in external.values()})
    if missing_external:
        status = "N/A_missing_external_column"
    elif rows == 0:
        status = "N/A_no_fold_rows"
    elif any(flag.startswith("N/A_") for flag in quality_flags):
        status = "ok_with_quality_flag"
    else:
        status = "ok"
    return {
        "rows": int(rows),
        "columns": columns,
        "external": external,
        "missing_external": missing_external,
        "quality_flags": quality_flags,
        "status": status,
        "contract_note": (
            "Current cache has no availability mask; zero values cannot be distinguished "
            "from missing/imputed values."
        ),
    }


def feature_coverage_for_fold(
    features: pd.DataFrame,
    item: FoldSeries,
    *,
    required_external: Iterable[str] = DEFAULT_EXTERNAL_FEATURES,
) -> dict[str, Any]:
    """Measure fold-local feature coverage using right-exclusive timestamps."""
    if not isinstance(features.index, pd.DatetimeIndex):
        normalized = features.copy()
        normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
        features = normalized
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("feature cache index must be sorted and unique")
    if item.test_start is not None and item.test_end is not None:
        mask = right_exclusive_mask(features.index.to_numpy(), item.test_start, item.test_end)
    else:
        timestamps = _timestamp_index(item.timestamps)
        mask = features.index.isin(timestamps)
    selected = features.loc[np.asarray(mask, dtype=bool)]
    result = _feature_coverage_for_frame(
        selected,
        rows=len(selected),
        required_external=required_external,
    )
    result.update(
        {
            "fold": int(item.fold),
            "test_start": item.test_start,
            "test_end_exclusive": item.test_end,
        }
    )
    return result


def summarize_feature_cache(
    features: pd.DataFrame,
    *,
    required_external: Iterable[str] = DEFAULT_EXTERNAL_FEATURES,
) -> dict[str, Any]:
    """Return all-cache and calendar-year coverage for provenance reporting."""
    if not isinstance(features.index, pd.DatetimeIndex):
        normalized = features.copy()
        normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
        features = normalized
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("feature cache index must be sorted and unique")
    years: dict[str, Any] = {}
    for year in sorted(set(int(value) for value in features.index.year)):
        selected = features.loc[features.index.year == year]
        years[str(year)] = _feature_coverage_for_frame(
            selected,
            rows=len(selected),
            required_external=required_external,
        )
    return {
        "rows": int(len(features)),
        "start": str(features.index.min()) if len(features) else None,
        "end_inclusive": str(features.index.max()) if len(features) else None,
        "all": _feature_coverage_for_frame(
            features,
            rows=len(features),
            required_external=required_external,
        ),
        "years": years,
    }


def _cost_rate(cfg: Mapping[str, Any]) -> float:
    costs = cfg.get("costs", {})
    return (
        (float(costs.get("spread_bps", 5.0)) / 10000.0) / 2.0
        + float(costs.get("fee_rate", 0.0004))
        + float(costs.get("slippage_bps", 2.0)) / 10000.0
    )


def backtest_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    benchmark: float = 1.0,
    execution_delay_bars: int = 0,
) -> dict[str, float | int]:
    """Compute the shared research metrics for one strategy path."""
    returns_arr = _finite_vector("returns", returns)
    positions_arr = _finite_vector("positions", positions)
    if len(returns_arr) != len(positions_arr):
        raise ValueError("returns and positions must have equal lengths")
    benchmark_arr = np.full(len(returns_arr), float(benchmark), dtype=np.float64)
    costs = cfg.get("costs", {})
    result = Backtest(
        returns_arr,
        positions_arr,
        spread_bps=float(costs.get("spread_bps", 5.0)),
        fee_rate=float(costs.get("fee_rate", 0.0004)),
        slippage_bps=float(costs.get("slippage_bps", 2.0)),
        interval=str(cfg.get("data", {}).get("interval", "15m")),
        benchmark_positions=benchmark_arr,
        execution_delay_bars=int(execution_delay_bars),
    ).run()
    stats = action_stats(positions_arr, benchmark_position=float(benchmark))
    return {
        "alpha_excess_pt": 100.0 * float(result.alpha_excess or 0.0),
        "maxdd_delta_pt": 100.0 * float(result.maxdd_delta or 0.0),
        "sharpe_delta": float(result.sharpe_delta or 0.0),
        "turnover": float(stats["turnover"]),
        "total_return_pt": 100.0 * float(result.total_return),
        "benchmark_total_return_pt": 100.0 * float(result.benchmark_total_return or 0.0),
        "max_drawdown_pt": 100.0 * abs(float(result.max_drawdown)),
        "benchmark_max_drawdown_pt": 100.0 * abs(float(result.benchmark_max_drawdown or 0.0)),
        "n_trades": int(result.n_trades),
        "execution_delay_bars": int(execution_delay_bars),
    }


def selection_score(metrics: Mapping[str, float]) -> float:
    """Validation-only score shared by fixed-constant selection."""
    return (
        float(metrics["alpha_excess_pt"])
        - max(0.0, float(metrics["maxdd_delta_pt"]))
        - 0.05 * float(metrics["turnover"])
    )


def select_fixed_exposure(
    validation_returns: np.ndarray,
    candidates: Iterable[float],
    cfg: Mapping[str, Any],
    *,
    benchmark: float = 1.0,
    execution_delay_bars: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """Select a constant exposure from validation returns only."""
    returns_arr = _finite_vector("validation returns", validation_returns)
    records = []
    for index, value in enumerate(float(candidate) for candidate in candidates):
        metrics = backtest_metrics(
            returns_arr,
            np.full(len(returns_arr), value, dtype=np.float64),
            cfg,
            benchmark=benchmark,
            execution_delay_bars=execution_delay_bars,
        )
        records.append(
            {
                "candidate": value,
                "score": selection_score(metrics),
                "metrics": metrics,
                "order": index,
            }
        )
    if not records:
        raise ValueError("fixed exposure candidate grid is empty")
    selected = max(records, key=lambda row: (float(row["score"]), -int(row["order"])))
    return float(selected["candidate"]), records


def actor_mean_exposure(validation_positions: np.ndarray) -> float:
    """Return the constant actor mean selected from a validation path."""
    values = _finite_vector("validation actor positions", validation_positions)
    return float(values.mean())


def circular_shift_positions(positions: np.ndarray, shift: int) -> np.ndarray:
    """Deterministic circular-shift null preserving the full position path."""
    values = _finite_vector("positions", positions)
    if len(values) == 0:
        raise ValueError("positions is empty")
    return np.roll(values, int(shift)).astype(np.float64, copy=False)


def _trial_row(
    *,
    fold: int,
    method: str,
    variant: str,
    positions: np.ndarray,
    returns: np.ndarray,
    cfg: Mapping[str, Any],
    benchmark: float,
    execution_delay_bars: int,
    selection: Mapping[str, Any],
    provenance: Mapping[str, Any],
    test_start: str | None,
    test_end: str | None,
    notes: str | None = None,
) -> dict[str, Any]:
    metrics = backtest_metrics(
        returns,
        positions,
        cfg,
        benchmark=benchmark,
        execution_delay_bars=execution_delay_bars,
    )
    row = {
        **dict(provenance),
        "schema_version": SCHEMA_VERSION,
        "record_type": "alpha_attribution_trial",
        "status": "ok",
        "fold": int(fold),
        "split": "test",
        "selection_split": selection.get("split"),
        "selection_status": selection.get("status"),
        "method": method,
        "variant": variant,
        "test_start": test_start,
        "test_end_exclusive": test_end,
        "execution_delay_bars": int(execution_delay_bars),
        "selection": dict(selection),
        "metrics": metrics,
    }
    if notes:
        row["notes"] = notes
    return row


def evaluate_fold_attribution(
    item: FoldSeries,
    cfg: Mapping[str, Any],
    *,
    benchmark: float = 1.0,
    fixed_exposures: Iterable[float] = DEFAULT_FIXED_EXPOSURES,
    lags: Iterable[int] = DEFAULT_LAGS,
    null_shifts: Iterable[int] = DEFAULT_NULL_SHIFTS,
    execution_delay_bars: int = 0,
    validation_returns: np.ndarray | None = None,
    validation_positions: np.ndarray | None = None,
    provenance: Mapping[str, Any] | None = None,
    feature_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all attribution variants for one fold.

    ``validation_returns`` and ``validation_positions`` are optional because
    the retained historical snapshot only contains test paths.  Their absence
    never causes test data to be used for selection: the actor-mean fallback is
    explicitly labelled retrospective/diagnostic-only.
    """
    provenance = dict(provenance or {})
    if feature_coverage is not None:
        # Attach the fold-local quality contract to each JSONL trial row so
        # consumers do not need a second join just to audit feature coverage.
        provenance["feature_coverage"] = dict(feature_coverage)
    fixed_grid = tuple(float(value) for value in fixed_exposures)
    lag_values = tuple(int(value) for value in lags)
    null_values = tuple(int(value) for value in null_shifts)
    if validation_returns is not None and len(validation_returns) > 0:
        selected_fixed, fixed_candidates = select_fixed_exposure(
            validation_returns,
            fixed_grid,
            cfg,
            benchmark=benchmark,
            execution_delay_bars=execution_delay_bars,
        )
        fixed_selection = {
            "split": "validation",
            "status": "selected_on_validation",
            "selected_candidate": selected_fixed,
            "candidates": fixed_candidates,
        }
    else:
        selected_fixed = None
        fixed_selection = {
            "split": "validation",
            "status": "unavailable_no_validation_returns",
            "selected_candidate": None,
            "candidates": [],
        }

    if validation_positions is not None and len(validation_positions) > 0:
        mean_exposure = actor_mean_exposure(validation_positions)
        mean_selection = {
            "split": "validation",
            "status": "selected_on_validation",
            "selected_candidate": mean_exposure,
        }
    else:
        mean_exposure = float(_finite_vector("test actor positions", item.positions).mean())
        mean_selection = {
            "split": "validation",
            "status": "diagnostic_only_test_mean",
            "selected_candidate": mean_exposure,
            "reason": "retained artifact has no validation actor-position path",
        }

    rows: list[dict[str, Any]] = []
    rows.append(
        _trial_row(
            fold=item.fold,
            method="bnh",
            variant="exposure_1.0",
            positions=np.full(len(item.returns), benchmark, dtype=np.float64),
            returns=item.returns,
            cfg=cfg,
            benchmark=benchmark,
            execution_delay_bars=execution_delay_bars,
            selection={"split": "none", "status": "reference_benchmark"},
            provenance=provenance,
            test_start=item.test_start,
            test_end=item.test_end,
        )
    )
    for exposure in fixed_grid:
        selected_note = "selected_on_validation" if selected_fixed == exposure else "grid_comparator"
        rows.append(
            _trial_row(
                fold=item.fold,
                method="fixed_exposure",
                variant=f"exposure_{exposure:.3f}",
                positions=np.full(len(item.returns), exposure, dtype=np.float64),
                returns=item.returns,
                cfg=cfg,
                benchmark=benchmark,
                execution_delay_bars=execution_delay_bars,
                selection={
                    **fixed_selection,
                    "row_role": selected_note,
                },
                provenance=provenance,
                test_start=item.test_start,
                test_end=item.test_end,
            )
        )
    mean_row = _trial_row(
        fold=item.fold,
        method="actor_mean",
        variant="constant_actor_mean",
        positions=np.full(len(item.returns), mean_exposure, dtype=np.float64),
        returns=item.returns,
        cfg=cfg,
        benchmark=benchmark,
        execution_delay_bars=execution_delay_bars,
        selection=mean_selection,
        provenance=provenance,
        test_start=item.test_start,
        test_end=item.test_end,
    )
    rows.append(mean_row)
    actor_row = _trial_row(
        fold=item.fold,
        method="actor_sequence",
        variant="raw_actor_path",
        positions=item.positions,
        returns=item.returns,
        cfg=cfg,
        benchmark=benchmark,
        execution_delay_bars=execution_delay_bars,
        selection={"split": "none", "status": "saved_actor_path"},
        provenance=provenance,
        test_start=item.test_start,
        test_end=item.test_end,
    )
    rows.append(actor_row)
    for lag in lag_values:
        if lag < 0:
            raise ValueError("actor lag must be non-negative")
        rows.append(
            _trial_row(
                fold=item.fold,
                method="actor_lag",
                variant=f"lag_{lag}",
                positions=item.positions,
                returns=item.returns,
                cfg=cfg,
                benchmark=benchmark,
                execution_delay_bars=execution_delay_bars + lag,
                selection={"split": "none", "status": "fixed_lag_null_diagnostic"},
                provenance=provenance,
                test_start=item.test_start,
                test_end=item.test_end,
            )
        )
    for shift in null_values:
        rows.append(
            _trial_row(
                fold=item.fold,
                method="null_circular_shift",
                variant=f"shift_{shift}",
                positions=circular_shift_positions(item.positions, shift),
                returns=item.returns,
                cfg=cfg,
                benchmark=benchmark,
                execution_delay_bars=execution_delay_bars,
                selection={"split": "none", "status": "deterministic_null"},
                provenance=provenance,
                test_start=item.test_start,
                test_end=item.test_end,
            )
        )

    constant_alpha = float(mean_row["metrics"]["alpha_excess_pt"])
    actor_alpha = float(actor_row["metrics"]["alpha_excess_pt"])
    decomposition = {
        "constant_exposure_component_alpha_excess_pt": constant_alpha,
        "timing_component_incremental_alpha_excess_pt": actor_alpha - constant_alpha,
        "actor_sequence_alpha_excess_pt": actor_alpha,
        "constant_exposure_selection_status": mean_selection["status"],
    }
    return {
        "fold": int(item.fold),
        "test_start": item.test_start,
        "test_end_exclusive": item.test_end,
        "rows": rows,
        "decomposition": decomposition,
        "selection": {
            "fixed_exposure": fixed_selection,
            "actor_mean": mean_selection,
        },
    }


def aggregate_trials(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate successful test rows by method/variant."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = f"{row['method']}::{row['variant']}"
        grouped.setdefault(key, []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for key, values in sorted(grouped.items()):
        alpha = np.asarray([float(row["metrics"]["alpha_excess_pt"]) for row in values])
        dd = np.asarray([float(row["metrics"]["maxdd_delta_pt"]) for row in values])
        sharpe = np.asarray([float(row["metrics"]["sharpe_delta"]) for row in values])
        turnover = np.asarray([float(row["metrics"]["turnover"]) for row in values])
        output[key] = {
            "method": values[0]["method"],
            "variant": values[0]["variant"],
            "folds": int(len(values)),
            "alpha_excess_mean_pt": float(alpha.mean()),
            "alpha_excess_median_pt": float(np.median(alpha)),
            "alpha_excess_worst_pt": float(alpha.min()),
            "maxdd_delta_mean_pt": float(dd.mean()),
            "maxdd_delta_median_pt": float(np.median(dd)),
            "sharpe_delta_mean": float(sharpe.mean()),
            "turnover_mean": float(turnover.mean()),
            "alpha_positive_folds": int(np.sum(alpha > 0.0)),
            "dd_improved_folds": int(np.sum(dd < 0.0)),
        }
    return output


def _future_windows(
    returns: np.ndarray,
    horizon: int,
    *,
    include_current: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_vector("diagnostic returns", returns)
    h = max(int(horizon), 1)
    targets = np.zeros(len(values), dtype=np.float64)
    mask = np.zeros(len(values), dtype=bool)
    offset = 0 if include_current else 1
    for index in range(len(values)):
        start = index + offset
        end = start + h
        if start >= 0 and end <= len(values):
            targets[index] = float(values[start:end].sum())
            mask[index] = True
    return targets, mask


def _future_risk_targets(
    returns: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = _finite_vector("diagnostic returns", returns)
    h = max(int(horizon), 1)
    vol = np.zeros(len(values), dtype=np.float64)
    drawdown = np.zeros(len(values), dtype=np.float64)
    crash = np.zeros(len(values), dtype=np.float64)
    excess = np.zeros(len(values), dtype=np.float64)
    mask = np.zeros(len(values), dtype=bool)
    threshold = 0.012
    for index in range(len(values)):
        end = index + 1 + h
        if end > len(values):
            continue
        window = values[index + 1 : end]
        cumulative = np.cumsum(window)
        dd = max(0.0, float(-min(0.0, float(cumulative.min(initial=0.0)))))
        vol[index] = float(np.sqrt(np.mean(np.square(window))))
        drawdown[index] = dd
        crash[index] = float(dd >= threshold)
        excess[index] = max(0.0, dd - threshold)
        mask[index] = True
    return vol, drawdown, crash, excess, mask


def _future_control_targets(
    returns: np.ndarray,
    horizon: int,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _finite_vector("diagnostic returns", returns)
    h = max(int(horizon), 1)
    ow = np.zeros(len(values), dtype=np.float64)
    recovery = np.zeros(len(values), dtype=np.float64)
    mask = np.zeros(len(values), dtype=bool)
    wm_cfg = cfg.get("world_model", {})
    delta = float(wm_cfg.get("overweight_delta", 0.25))
    dd_penalty = float(wm_cfg.get("overweight_drawdown_penalty", 0.35))
    recovery_penalty = float(wm_cfg.get("recovery_drawdown_penalty", 0.50))
    target_scale = float(wm_cfg.get("control_target_scale", wm_cfg.get("risk_target_scale", 1.0)))
    for index in range(len(values)):
        end = index + 1 + h
        if end > len(values):
            continue
        window = values[index + 1 : end]
        cumulative = np.cumsum(window)
        downside = max(0.0, float(-min(0.0, float(cumulative.min(initial=0.0)))))
        vol = float(np.sqrt(np.mean(np.square(window))))
        ow[index] = (
            delta * float(cumulative[-1])
            - delta * dd_penalty * downside
            - _cost_rate(cfg) * abs(delta)
        ) * target_scale
        recovery[index] = (
            (float(cumulative[-1]) - recovery_penalty * downside)
            / (vol * h**0.5 + 1e-6)
        )
        recovery[index] = float(np.clip(recovery[index], -5.0, 5.0) / 5.0)
        mask[index] = True
    return ow, recovery, mask


def _position_utility_targets(
    returns: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    values = _finite_vector("diagnostic returns", returns)
    wm_cfg = cfg.get("world_model", {})
    positions = [float(value) for value in wm_cfg.get("position_utility_positions", [])]
    if not positions:
        return np.zeros((len(values), 0), dtype=np.float64), np.zeros(len(values), dtype=bool), positions
    benchmark = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    h = max(int(wm_cfg.get("position_utility_horizon", 32)), 1)
    dd_penalty = float(wm_cfg.get("position_utility_dd_penalty", 1.0))
    dd_improve_reward = float(wm_cfg.get("position_utility_dd_improve_reward", 0.0))
    vol_penalty = float(wm_cfg.get("position_utility_vol_penalty", 0.25))
    target_scale = float(wm_cfg.get("position_utility_target_scale", 1.0))
    targets = np.zeros((len(values), len(positions)), dtype=np.float64)
    mask = np.zeros(len(values), dtype=bool)
    for index in range(len(values)):
        end = index + 1 + h
        if end > len(values):
            continue
        window = values[index + 1 : end]
        cumulative = np.cumsum(window)
        future_vol = float(np.sqrt(np.mean(np.square(window))))
        benchmark_path = benchmark * cumulative
        bench_peak = np.maximum.accumulate(np.r_[0.0, benchmark_path])
        bench_dd = float(np.max(bench_peak - np.r_[0.0, benchmark_path]))
        for col, position in enumerate(positions):
            path = position * cumulative
            peak = np.maximum.accumulate(np.r_[0.0, path])
            dd = float(np.max(peak - np.r_[0.0, path]))
            overlay = position - benchmark
            utility = (
                overlay * float(cumulative[-1])
                - abs(overlay) * _cost_rate(cfg)
                - dd_penalty * max(0.0, dd - bench_dd)
                + dd_improve_reward * max(0.0, bench_dd - dd)
                - vol_penalty * abs(overlay) * future_vol
            )
            targets[index, col] = utility * target_scale
        mask[index] = True
    return targets, mask, positions


def _metric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    signed_target: bool = True,
) -> tuple[dict[str, float | None], list[str]]:
    reasons: list[str] = []
    metrics: dict[str, float | None] = {}
    if len(prediction) == 0:
        return metrics, ["no finite target/prediction rows"]
    metrics["mae"] = _metric_or_none(mean_absolute_error(target, prediction))
    metrics["rmse"] = _metric_or_none(np.sqrt(mean_squared_error(target, prediction)))
    if signed_target:
        metrics["sign_accuracy"] = _metric_or_none(np.mean(np.sign(target) == np.sign(prediction)))
    else:
        metrics["sign_accuracy"] = None
        reasons.append("sign accuracy is N/A for a one-sided non-negative target")
    if len(np.unique(target)) < 2 or len(np.unique(prediction)) < 2:
        metrics["spearman_ic"] = None
        reasons.append("Spearman IC undefined for a constant target or prediction")
    else:
        metrics["spearman_ic"] = _metric_or_none(spearmanr(target, prediction).statistic)
    return metrics, reasons


def _ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float | None:
    if len(probability) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return float(total)


def _classification_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float | None], list[str]]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = np.isfinite(probability) & np.isfinite(labels)
    probability = np.clip(probability[valid], 0.0, 1.0)
    labels = labels[valid]
    metrics: dict[str, float | None] = {}
    reasons: list[str] = []
    if len(labels) == 0:
        return metrics, ["no finite classification rows"]
    predicted = (probability >= 0.5).astype(np.int64)
    if len(np.unique(labels)) < 2:
        reasons.append("balanced accuracy and MCC require both classes")
        metrics["balanced_accuracy"] = None
        metrics["mcc"] = None
    else:
        metrics["balanced_accuracy"] = _metric_or_none(balanced_accuracy_score(labels, predicted))
        metrics["mcc"] = _metric_or_none(matthews_corrcoef(labels, predicted))
    metrics["brier"] = _metric_or_none(brier_score_loss(labels, probability))
    metrics["ece"] = _metric_or_none(_ece(probability, labels))
    return metrics, reasons


def _head_kind(name: str) -> tuple[str, int | None]:
    for prefix, kind in (
        ("wm_pred_return_h", "return"),
        ("wm_pred_vol_h", "vol"),
        ("wm_pred_drawdown_h", "drawdown"),
        ("wm_pred_crash_h", "crash"),
        ("wm_pred_drawdown_excess_h", "drawdown_excess"),
        ("wm_pred_overweight_advantage_h", "overweight_advantage"),
        ("wm_pred_recovery_h", "recovery"),
    ):
        if name.startswith(prefix):
            try:
                return kind, int(name[len(prefix) :])
            except ValueError:
                return kind, None
    if name.startswith("wm_pred_position_utility_p"):
        return "position_utility", None
    return "unknown", None


def _inverse_predictive_state(
    advantage: np.ndarray,
    names: list[str],
    predictive_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Undo the saved actor-input transform where clipping is invertible.

    ``sample_input.npz:advantage`` is not a raw WM-head dump.  It is the
    output of ``build_wm_predictive_state_bundle``: train-stat standardize,
    clip, then scale.  Inverting the affine part is exact; rows at the clip
    boundary are censored and excluded from raw-head metrics.
    """
    values = np.asarray(advantage, dtype=np.float64)
    mean = np.asarray(predictive_state.get("mean", []), dtype=np.float64)
    std = np.asarray(predictive_state.get("std", []), dtype=np.float64)
    state_names = [str(value) for value in np.asarray(predictive_state.get("names", []), dtype=object)]
    if mean.ndim != 2 or std.ndim != 2 or mean.shape[1] != values.shape[1] or std.shape[1] != values.shape[1]:
        raise ValueError("predictive_state mean/std dimensions do not match saved advantage")
    if state_names and state_names != names:
        index = {name: idx for idx, name in enumerate(state_names)}
        if any(name not in index for name in names):
            raise ValueError("predictive_state names do not cover saved advantage columns")
        mean = mean[:, [index[name] for name in names]]
        std = std[:, [index[name] for name in names]]
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    ac_cfg = cfg.get("ac", {})
    scale = float(ac_cfg.get("wm_predictive_state_scale", 1.0))
    if abs(scale) < 1e-12:
        raise ValueError("wm_predictive_state_scale must be non-zero")
    standardized = values / scale
    clip = float(ac_cfg.get("wm_predictive_state_clip", 5.0))
    if clip > 0.0:
        clipped = np.abs(standardized) >= clip - 1e-6
        standardized = np.clip(standardized, -clip, clip)
    else:
        clipped = np.zeros_like(standardized, dtype=bool)
    if bool(ac_cfg.get("wm_predictive_state_standardize", True)):
        raw = standardized * std[0] + mean[0]
    else:
        raw = standardized
    return raw, ~clipped, {
        "prediction_space": "raw_head_output",
        "source_space": "standardized_clipped_scaled_actor_input",
        "standardize": bool(ac_cfg.get("wm_predictive_state_standardize", True)),
        "clip": clip,
        "scale": scale,
        "clipped_rows_excluded": int(np.sum(clipped)),
    }


def diagnose_saved_artifact(
    sample_input_path: str | Path,
    *,
    predictive_state_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    fold: int = 23,
    provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Diagnose saved predictive/regime inputs without inventing unavailable labels."""
    cfg = cfg or {}
    provenance = dict(provenance or {})
    sample_path = Path(sample_input_path)
    with np.load(sample_path, allow_pickle=True) as data:
        required = {"returns", "advantage"}
        missing = sorted(required - set(data.files))
        if missing:
            return [
                {
                    **provenance,
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "predictive_head_diagnostic",
                    "status": "N/A",
                    "fold": int(fold),
                    "head": "all",
                    "reason": f"sample artifact missing required keys: {missing}",
                }
            ]
        returns = _finite_vector("sample returns", data["returns"])
        advantage = np.asarray(data["advantage"], dtype=np.float64)
        names: list[str]
        predictive_state = None
        if predictive_state_path is not None and Path(predictive_state_path).exists():
            with np.load(predictive_state_path, allow_pickle=True) as state:
                predictive_state = {key: np.asarray(state[key]) for key in state.files}
                names = [str(value) for value in np.asarray(predictive_state.get("names", []), dtype=object)]
        else:
            names = []
        if advantage.ndim != 2 or not names or advantage.shape[1] != len(names):
            return [
                {
                    **provenance,
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "predictive_head_diagnostic",
                    "status": "N/A",
                    "fold": int(fold),
                    "head": "advantage",
                    "reason": "advantage columns cannot be aligned to predictive_state names",
                }
            ]
        if advantage.shape[0] != len(returns):
            return [
                {
                    **provenance,
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "predictive_head_diagnostic",
                    "status": "N/A",
                    "fold": int(fold),
                    "head": "advantage",
                    "reason": "saved advantage rows are not aligned to saved returns",
                }
            ]
        try:
            raw, transform_valid, transform_meta = _inverse_predictive_state(
                advantage,
                names,
                predictive_state,
                cfg,
            )
        except ValueError as exc:
            return [
                {
                    **provenance,
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "predictive_head_diagnostic",
                    "status": "N/A",
                    "fold": int(fold),
                    "head": "advantage",
                    "reason": str(exc),
                }
            ]

        records: list[dict[str, Any]] = []
        handled: set[str] = set()
        for name in names:
            if name in handled:
                continue
            kind, horizon = _head_kind(name)
            common = {
                **provenance,
                "schema_version": SCHEMA_VERSION,
                "record_type": "predictive_head_diagnostic",
                "fold": int(fold),
                "head": name,
                "artifact_path": str(sample_path),
                "prediction_space": transform_meta["prediction_space"],
                "source_space": transform_meta["source_space"],
            }
            if kind == "unknown":
                records.append({**common, "status": "N/A", "reason": "unknown predictive head name"})
                handled.add(name)
                continue
            if kind == "position_utility":
                utility_names = [value for value in names if _head_kind(value)[0] == "position_utility"]
                utility_indices = [names.index(value) for value in utility_names]
                targets, mask, positions = _position_utility_targets(returns, cfg)
                if targets.shape[1] != len(utility_indices):
                    records.append(
                        {
                            **common,
                            "head": "position_utility",
                            "status": "N/A",
                            "reason": "position_utility target positions are absent or mismatched",
                        }
                    )
                else:
                    pred_values = raw[:, utility_indices]
                    state_valid = transform_valid[:, utility_indices].all(axis=1)
                    valid = mask & state_valid & np.isfinite(pred_values).all(axis=1)
                    clipped_count = int(np.sum(mask & ~state_valid))
                    if not np.any(valid):
                        records.append(
                            {
                                **common,
                                "head": "position_utility",
                                "status": "N/A",
                                "reason": (
                                    "no valid future utility rows after horizon mask or "
                                    "predictive-state clipping"
                                ),
                            }
                        )
                    else:
                        # The WM position-utility head is a vector of
                        # smooth-L1/ranking regression scores, not logits.
                        # Compare each action column to its utility target;
                        # do not invent a softmax/Brier classification task.
                        for utility_name, utility_index, position in zip(
                            utility_names,
                            utility_indices,
                            positions,
                        ):
                            utility_valid = valid & transform_valid[:, utility_index]
                            utility_metrics, utility_reasons = _regression_metrics(
                                raw[utility_valid, utility_index],
                                targets[utility_valid, utility_names.index(utility_name)],
                            )
                            records.append(
                                {
                                    **common,
                                    "head": utility_name,
                                    "status": "ok" if utility_metrics else "N/A",
                                    "target_type": "position_utility_regression",
                                    "action_position": float(position),
                                    "n_valid": int(np.sum(utility_valid)),
                                    "metrics": utility_metrics,
                                    **({"reason": "; ".join(utility_reasons)} if utility_reasons else {}),
                                }
                            )
                        labels = np.argmax(targets[valid], axis=1)
                        predicted = np.argmax(pred_values[valid], axis=1)
                        records.append(
                            {
                                **common,
                                "head": "position_utility_argmax",
                                "status": "ok",
                                "target_type": "best_utility_action_from_regression_scores",
                                "n_valid": int(np.sum(valid)),
                                "positions": positions,
                                "metrics": {
                                    "accuracy": _metric_or_none(np.mean(labels == predicted)),
                                },
                                "reason": "; ".join(
                                    [
                                        "argmax is a decision diagnostic over regression scores, not a classification/logit metric",
                                        *([f"excluded {clipped_count} rows at predictive-state clip boundary"] if clipped_count else []),
                                    ]
                                ),
                            }
                        )
                handled.update(utility_names)
                continue
            if horizon is None:
                records.append({**common, "status": "N/A", "reason": "head horizon is not parseable"})
                handled.add(name)
                continue
            column_index = names.index(name)
            prediction = raw[:, column_index]
            if kind == "return":
                target, mask = _future_windows(
                    returns,
                    horizon,
                    include_current=bool(cfg.get("world_model", {}).get("return_include_current", True)),
                )
                target *= float(cfg.get("world_model", {}).get("return_target_scale", 1.0))
                target_type = "future_return"
                classification = False
            elif kind == "vol":
                target, _, _, _, mask = _future_risk_targets(returns, horizon)
                target *= float(cfg.get("world_model", {}).get("risk_target_scale", 1.0))
                target_type = "future_realized_volatility"
                classification = False
            elif kind == "drawdown":
                _, target, _, _, mask = _future_risk_targets(returns, horizon)
                target *= float(cfg.get("world_model", {}).get("risk_target_scale", 1.0))
                target_type = "future_drawdown"
                classification = False
            elif kind == "drawdown_excess":
                _, _, _, target, mask = _future_risk_targets(returns, horizon)
                target *= float(cfg.get("world_model", {}).get("risk_target_scale", 1.0))
                target_type = "future_drawdown_excess"
                classification = False
            elif kind == "crash":
                _, _, target, _, mask = _future_risk_targets(returns, horizon)
                target_type = "future_crash_label"
                classification = True
            elif kind in {"overweight_advantage", "recovery"}:
                ow, recovery, mask = _future_control_targets(returns, horizon, cfg)
                target = ow if kind == "overweight_advantage" else recovery
                target_type = kind
                classification = False
            else:
                records.append({**common, "status": "N/A", "reason": f"no target implementation for {kind}"})
                handled.add(name)
                continue
            valid = mask & transform_valid[:, column_index] & np.isfinite(target) & np.isfinite(prediction)
            clipped_count = int(np.sum(mask & ~transform_valid[:, column_index]))
            if not np.any(valid):
                records.append(
                    {
                        **common,
                        "status": "N/A",
                        "target_type": target_type,
                        "reason": (
                            "no valid future rows after horizon mask or predictive-state clipping"
                        ),
                    }
                )
            elif classification:
                probability = 1.0 / (1.0 + np.exp(-np.clip(prediction[valid], -60.0, 60.0)))
                metrics, reasons = _classification_metrics(probability, target[valid].astype(np.int64))
                records.append(
                    {
                        **common,
                        "status": "ok",
                        "target_type": target_type,
                        "n_valid": int(np.sum(valid)),
                        "metrics": metrics,
                        **({
                            "reason": "; ".join(
                                [
                                    *reasons,
                                    *([f"excluded {clipped_count} rows at predictive-state clip boundary"] if clipped_count else []),
                                ]
                            )
                        } if reasons or clipped_count else {}),
                    }
                )
            else:
                metrics, reasons = _regression_metrics(
                    prediction[valid],
                    target[valid],
                    signed_target=kind not in {"vol", "drawdown", "drawdown_excess"},
                )
                records.append(
                    {
                        **common,
                        "status": "ok" if metrics else "N/A",
                        "target_type": target_type,
                        "n_valid": int(np.sum(valid)),
                        "metrics": metrics,
                        **({
                            "reason": "; ".join(
                                [
                                    *reasons,
                                    *([f"excluded {clipped_count} rows at predictive-state clip boundary"] if clipped_count else []),
                                ]
                            )
                        } if reasons or clipped_count else {}),
                    }
                )
            handled.add(name)

        regime = np.asarray(data["regime"], dtype=np.float64) if "regime" in data.files else None
        regime_target_key = next(
            (key for key in ("regime_target", "regime_labels", "regime_label") if key in data.files),
            None,
        )
        common = {
            **provenance,
            "schema_version": SCHEMA_VERSION,
            "record_type": "predictive_head_diagnostic",
            "fold": int(fold),
            "head": "regime",
            "artifact_path": str(sample_path),
        }
        if regime is None or regime_target_key is None:
            records.append(
                {
                    **common,
                    "status": "N/A",
                    "reason": "saved artifact has regime probabilities but no regime target labels",
                }
            )
        else:
            labels_raw = np.asarray(data[regime_target_key])
            labels = np.argmax(labels_raw, axis=1) if labels_raw.ndim > 1 else labels_raw.reshape(-1)
            probabilities = regime
            valid = len(labels) == len(probabilities) and np.isfinite(probabilities).all()
            if not valid:
                records.append({**common, "status": "N/A", "reason": "regime labels/probabilities are unaligned"})
            else:
                predicted = np.argmax(probabilities, axis=1)
                records.append(
                    {
                        **common,
                        "status": "ok",
                        "target_type": "regime_class",
                        "n_valid": int(len(labels)),
                        "metrics": {
                            "balanced_accuracy": _metric_or_none(
                                balanced_accuracy_score(labels, predicted)
                                if len(np.unique(labels)) > 1
                                else None
                            ),
                            "mcc": _metric_or_none(
                                matthews_corrcoef(labels, predicted)
                                if len(np.unique(labels)) > 1
                                else None
                            ),
                            "brier": None,
                            "ece": None,
                        },
                    }
                )
        return records


def build_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render the compact human-readable report from the JSON payload."""
    lines = [
        "# Plan011 v31 Alpha Attribution and Forecast Diagnostics",
        "",
        "## Scope and provenance",
        "",
        f"- folds: `{', '.join(str(value) for value in payload['folds'])}`",
        f"- selection split: `{payload['selection_contract']}`",
        f"- config SHA-256: `{payload['config_sha256']}`",
        f"- data SHA-256: `{payload['data_sha256']}`",
        f"- data-contract SHA-256: `{payload['data_contract_sha256']}`",
        f"- commit: `{payload['commit_hash']}`",
        f"- costs: `{json.dumps(payload['costs'], sort_keys=True)}`",
        "",
        "Attribution/model selection uses development folds only; holdout folds 15-23 are never used for selection. "
        "When present, the saved fold23 bundle diagnostics below are reference-only and do not alter a candidate, threshold, or test result.",
        "Validation-only selection is recorded when validation paths are supplied. The retained historical artifact has no validation actor path, so its actor-mean row is diagnostic-only.",
        "",
        "## Attribution summary",
        "",
        "Constant exposure component is the actor-mean constant path; timing component is actor-sequence AlphaEx minus that constant path under the same cost contract.",
        "",
        "| method | variant | folds | mean AlphaEx | mean MaxDDDelta | mean SharpeDelta | mean turnover |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"].values():
        lines.append(
            f"| {row['method']} | {row['variant']} | {row['folds']} | "
            f"{row['alpha_excess_mean_pt']:+.3f}pt | {row['maxdd_delta_mean_pt']:+.3f}pt | "
            f"{row['sharpe_delta_mean']:+.4f} | {row['turnover_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Constant versus timing",
        "",
        "| fold | constant AlphaEx | timing increment | actor sequence AlphaEx | mean selection status |",
        "|---:|---:|---:|---:|---|",
    ])
    for row in payload["decomposition"]:
        lines.append(
            f"| {row['fold']} | {row['constant_exposure_component_alpha_excess_pt']:+.3f}pt | "
            f"{row['timing_component_incremental_alpha_excess_pt']:+.3f}pt | "
            f"{row['actor_sequence_alpha_excess_pt']:+.3f}pt | {row['constant_exposure_selection_status']} |"
        )
    lines.extend([
        "",
        "## Feature coverage",
        "",
        "Coverage is measured on each fold with `[test_start, test_end)` and is report-only. "
        "The current cache has no availability mask, so a zero cannot be distinguished from a missing/imputed value; this is retained as an explicit quality flag.",
        "",
        "| fold | rows | funding nonzero | basis nonzero | basis_mom nonzero | basis_abs nonzero | quality/status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ])
    for coverage in payload.get("feature_coverage", {}).values():
        external = coverage.get("external", {})

        def _rate(name: str) -> str:
            value = external.get(name, {}).get("nonzero_rate")
            return "N/A" if value is None else f"{100.0 * float(value):.2f}%"

        lines.append(
            f"| {coverage.get('fold', '')} | {coverage.get('rows', 0)} | "
            f"{_rate('funding_rate')} | {_rate('basis')} | {_rate('basis_mom')} | "
            f"{_rate('basis_abs')} | {coverage.get('status', '')}; "
            f"{', '.join(coverage.get('quality_flags', []))} |"
        )
    cache_summary = payload.get("feature_cache_summary") or {}
    if cache_summary:
        lines.extend([
            "",
            "### Cache diagnostic (not a model-selection input)",
            "",
            f"- rows: `{cache_summary.get('rows')}`; observed range: `[{cache_summary.get('start')}, {cache_summary.get('end_inclusive')}]`",
            "- Year-level derived-series rates are retained below. These diagnostics do not select, tune, or re-adjust any holdout result.",
            "",
            "| year | rows | funding nonzero | basis nonzero | basis_mom nonzero | basis_abs nonzero |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for year, coverage in cache_summary.get("years", {}).items():
            external = coverage.get("external", {})
            values = []
            for name in DEFAULT_EXTERNAL_FEATURES:
                rate = external.get(name, {}).get("nonzero_rate")
                values.append("N/A" if rate is None else f"{100.0 * float(rate):.2f}%")
            lines.append(f"| {year} | {coverage.get('rows', 0)} | " + " | ".join(values) + " |")
    lines.extend(["", "## Predictive-head diagnostics", "", "| fold | head | status | n | metrics / reason |", "|---:|---|---|---:|---|"])
    for row in payload.get("diagnostics", []):
        detail = json.dumps(row.get("metrics"), sort_keys=True) if row.get("metrics") is not None else str(row.get("reason", ""))
        lines.append(
            f"| {row.get('fold', '')} | {row.get('head', '')} | {row.get('status', '')} | "
            f"{row.get('n_valid', '')} | `{detail}` |"
        )
    lines.extend([
        "",
        "## Metric and leakage contract",
        "",
        "- AlphaEx = strategy final total return minus B&H final total return; MaxDDDelta = strategy absolute MaxDD minus B&H absolute MaxDD; SharpeDelta and turnover come from the shared Backtest/action_stats implementations.",
        "- All strategy paths use the configured costs and execution delay. B&H is the benchmark path and is not delayed by Backtest's strategy-only delay.",
        "- Actor lags use deterministic strategy execution delay; nulls are deterministic circular shifts and are not candidates for selection.",
        "- Fold bounds are validated as right-exclusive `[test_start, test_end)`; predictive targets mask the final unavailable future horizon.",
        "- Saved `sample_input.npz` advantage values are standardized/clipped/scaled actor inputs; diagnostics invert the affine transform and exclude clip-boundary rows from raw-head metrics. Volatility/drawdown sign accuracy is N/A because those targets are one-sided.",
        "- Position-utility outputs are smooth-L1/ranking regression scores per action, not logits; diagnostics use per-action regression plus an argmax decision accuracy and do not report softmax/Brier classification for that head.",
        "- A diagnostic marked `N/A` has no computable target/label; unavailable metrics are never replaced with zero.",
        "",
        f"Machine-readable ledger: `{payload['ledger_path']}`",
    ])
    return "\n".join(lines) + "\n"


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")


def run_attribution(
    *,
    series: list[FoldSeries],
    cfg: Mapping[str, Any],
    config_path: str,
    artifact_path: str | Path,
    seed: int,
    benchmark: float = 1.0,
    fixed_exposures: Iterable[float] = DEFAULT_FIXED_EXPOSURES,
    lags: Iterable[int] = DEFAULT_LAGS,
    null_shifts: Iterable[int] = DEFAULT_NULL_SHIFTS,
    execution_delay_bars: int = 0,
    sample_input_path: str | Path | None = None,
    predictive_state_path: str | Path | None = None,
    output_dir: str | Path = "docs/alpha_attribution_plan011_v31_dev",
    holdout_reference: bool = False,
    feature_frame: pd.DataFrame | None = None,
    feature_artifact_path: str | Path | None = None,
    validation_series: list[FoldSeries] | None = None,
    validation_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run attribution and optionally saved-bundle diagnostics."""
    if not series:
        raise ValueError("attribution requires at least one fold")
    if any(item.fold >= 15 for item in series) and not holdout_reference:
        raise ValueError("holdout folds 15-23 are reference-only; pass holdout_reference=True explicitly")
    if validation_series and any(item.fold >= 15 for item in validation_series):
        raise ValueError("holdout folds 15-23 cannot be used as a validation selection split")
    if feature_frame is None and feature_artifact_path is not None:
        feature_frame = pd.read_parquet(feature_artifact_path)
    validation_by_fold = {
        item.fold: item for item in (validation_series or [])
    }
    if any(item.fold >= 15 for item in series):
        # Reference-only holdout runs must not select a constant, even if a
        # caller accidentally supplies a same-fold validation path.
        validation_by_fold = {}
    feature_coverage: dict[str, dict[str, Any]] = {}
    feature_cache_summary: dict[str, Any] | None = None
    if feature_frame is not None:
        feature_coverage = {
            str(item.fold): feature_coverage_for_fold(feature_frame, item)
            for item in sorted(series, key=lambda value: value.fold)
        }
        feature_cache_summary = summarize_feature_cache(feature_frame)
    data_contract = {
        "symbol": cfg.get("data", {}).get("symbol", "BTCUSDT"),
        "interval": cfg.get("data", {}).get("interval", "15m"),
        "benchmark_position": float(benchmark),
        "right_exclusive_folds": True,
        "feature_coverage_indexing": "[test_start, test_end)",
        "feature_zero_missing_distinction": False,
    }
    artifact_paths = {
        "timeseries": str(Path(artifact_path)),
        "timeseries_sha256": sha256_file(artifact_path),
        "sample_input": str(sample_input_path) if sample_input_path is not None else None,
        "sample_input_sha256": (
            sha256_file(sample_input_path)
            if sample_input_path is not None and Path(sample_input_path).exists()
            else None
        ),
        "predictive_state": str(predictive_state_path) if predictive_state_path is not None else None,
        "predictive_state_sha256": (
            sha256_file(predictive_state_path)
            if predictive_state_path is not None and Path(predictive_state_path).exists()
            else None
        ),
        "features": str(Path(feature_artifact_path)) if feature_artifact_path is not None else None,
        "features_sha256": (
            sha256_file(feature_artifact_path)
            if feature_artifact_path is not None and Path(feature_artifact_path).exists()
            else None
        ),
        "validation_timeseries": (
            str(Path(validation_artifact_path)) if validation_artifact_path is not None else None
        ),
        "validation_timeseries_sha256": (
            sha256_file(validation_artifact_path)
            if validation_artifact_path is not None and Path(validation_artifact_path).exists()
            else None
        ),
    }
    provenance = {
        "commit_hash": _git_commit(),
        "config_path": str(config_path),
        "config_sha256": config_fingerprint(dict(cfg)),
        "data_contract": data_contract,
        "data_contract_sha256": _canonical_sha256(data_contract),
        "data_sha256": data_fingerprint_from_series(series),
        "source_sha256": source_fingerprint(),
        "artifact_paths": artifact_paths,
        "seed": int(seed),
    }
    fold_payloads = []
    rows: list[dict[str, Any]] = []
    for item in sorted(series, key=lambda value: value.fold):
        validation_item = validation_by_fold.get(item.fold)
        result = evaluate_fold_attribution(
            item,
            cfg,
            benchmark=benchmark,
            fixed_exposures=fixed_exposures,
            lags=lags,
            null_shifts=null_shifts,
            execution_delay_bars=execution_delay_bars,
            provenance=provenance,
            feature_coverage=feature_coverage.get(str(item.fold)),
            validation_returns=(validation_item.returns if validation_item is not None else None),
            validation_positions=(validation_item.positions if validation_item is not None else None),
        )
        fold_payloads.append(result)
        rows.extend(result["rows"])
    diagnostics = []
    if sample_input_path is not None:
        diagnostics = diagnose_saved_artifact(
            sample_input_path,
            predictive_state_path=predictive_state_path,
            cfg=cfg,
            fold=23,
            provenance={
                **provenance,
                "selection_split": "none",
                "selection_status": "holdout_reference_only",
            },
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ledger_path = output_path / "alpha_attribution_ledger.jsonl"
    payload: dict[str, Any] = {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "record_type": "alpha_attribution_report",
        "status": "complete",
        "config": str(config_path),
        "folds": [int(item.fold) for item in sorted(series, key=lambda value: value.fold)],
        "seed": int(seed),
        "benchmark_position": float(benchmark),
        "costs": dict(cfg.get("costs", {})),
        "selection_contract": (
            "validation-only selection; test report-only; holdout reference-only"
            if not any(item.fold >= 15 for item in series)
            else "holdout reference-only; no candidate selection"
        ),
        "fixed_exposures": [float(value) for value in fixed_exposures],
        "lags": [int(value) for value in lags],
        "null_shifts": [int(value) for value in null_shifts],
        "execution_delay_bars": int(execution_delay_bars),
        "rows": rows,
        "summary": aggregate_trials(rows),
        "decomposition": [
            {"fold": item["fold"], **item["decomposition"]} for item in fold_payloads
        ],
        "feature_coverage": feature_coverage,
        "feature_cache_summary": feature_cache_summary,
        "diagnostics": diagnostics,
        "ledger_path": str(ledger_path),
    }
    records: list[dict[str, Any]] = [
        {
            **provenance,
            "schema_version": SCHEMA_VERSION,
            "record_type": "alpha_attribution_run",
            "status": "complete",
            "fold": None,
            "selection_split": "validation",
            "metrics": None,
            "artifact_paths": provenance["artifact_paths"],
        },
        *rows,
        *diagnostics,
    ]
    write_jsonl(ledger_path, records)
    payload["report_path"] = str(output_path / "report.md")
    (output_path / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "report.md").write_text(build_report_markdown(payload), encoding="utf-8")
    return payload
