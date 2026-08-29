"""Leak-safe low-cost forecast/timing tournament for Plan011.

The tournament is intentionally narrower than the main Plan011 training
pipeline.  It fits direct future-return/risk forecasts on development folds
only, chooses model/policy settings on each fold's validation interval, and
reports the corresponding test interval once.  No Oracle labels or position
paths are consumed.

The implementation keeps portfolio definitions in the shared
``alpha_attribution``/``Backtest`` helpers.  This is important here: a useful
forecast must add timing alpha after the constant exposure component and
under the same costs, execution delay, and benchmark contract.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.data.dataset import WFOSplit, get_wfo_splits
from unidream.eval.alpha_attribution import (
    backtest_metrics,
    circular_shift_positions,
    select_fixed_exposure,
    sha256_file,
)
from unidream.experiments.run_config import config_fingerprint, source_fingerprint


SCHEMA_VERSION = 1
DEV_FOLDS = (0, 2, 8)
DEV_CUTOFF = pd.Timestamp("2024-01-01")
BENCHMARK_POSITION = 1.0
DEFAULT_HORIZONS = (4, 16, 64)
DEFAULT_FEATURE_SETS = ("ohlcv13", "full17")
EXTERNAL_FEATURES = ("funding_rate", "basis", "basis_mom", "basis_abs")
OHLCV_DERIVED_FEATURES = (
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "vol_ret",
    "RSI_14",
    "macd",
    "macd_signal",
    "atr_norm_ret",
    "atr",
    "rv_4",
    "rv_16",
    "rv_96",
)
EXPECTED_FEATURES = (*OHLCV_DERIVED_FEATURES, *EXTERNAL_FEATURES)
CONSTANT_EXPOSURE_GRID = (0.5, 0.75, 1.0, 1.05, 1.10, 1.12)
# Features at bar t only contain information available before bar t; raw
# returns[t] is earned by the bar that starts at t. Targets are intentionally
# ``t+1..t+h`` per the tournament contract, so a decision at t is evaluated
# against return[t+1] with one fixed execution delay.
DEFAULT_EXECUTION_DELAY_BARS = 1
EXECUTION_DELAY_SENSITIVITY_LAGS = (1, 16)
DEFAULT_NULL_SHIFTS = (1, 16, 64)
TIMING_SUPERIORITY_MARGIN_PT = 0.02
TIMING_MIN_WIN_RATE = 2.0 / 3.0
TIMING_REQUIRED_COMPARATORS = ("constant", "lag16", "null_shift64")
TIMING_DIAGNOSTIC_COMPARATORS = ("lag1", "null_shift1", "null_shift16")
DEFAULT_OVERLAY_GRID = (0.04, 0.12)
DEFAULT_HYSTERESIS_GRID = (0.0, 0.25)
DEFAULT_MIN_HOLD_GRID = (0, 32)
DEFAULT_MAX_POSITION_STEP = 0.08
DEFAULT_POLICY_HORIZON = 16
DEFAULT_MAX_FIT_ROWS = 40_000
DEFAULT_HIST_MAX_ITER = 60


@dataclass(frozen=True)
class DevelopmentData:
    """Development-only feature/return frame and the requested WFO splits."""

    features: pd.DataFrame
    returns: pd.Series
    splits: tuple[WFOSplit, ...]
    feature_path: Path
    returns_path: Path
    train_years: int = 2
    val_months: int = 3
    test_months: int = 3


@dataclass(frozen=True)
class PolicyParams:
    """Validation-selected causal overlay policy parameters."""

    threshold: float
    overlay_magnitude: float
    hysteresis: float
    min_hold: int
    execution_delay: int


@dataclass
class FittedCandidate:
    """Direct multi-horizon return/risk models for one hyperparameter choice."""

    name: str
    hyperparameters: dict[str, Any]
    horizons: tuple[int, ...]
    return_models: dict[int, Any]
    risk_models: dict[int, Any]
    trend_return_coefs: dict[int, tuple[float, float]] | None = None
    trend_risk_coefs: dict[int, tuple[float, float]] | None = None
    fit_rows: int = 0
    dropped_nonfinite_rows: int = 0

    def predict(
        self,
        features: pd.DataFrame | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict return and realized-volatility targets without future data."""
        if isinstance(features, pd.DataFrame):
            frame = features
            values = frame.to_numpy(dtype=np.float64)
            names = list(frame.columns)
        else:
            values = np.asarray(features, dtype=np.float64)
            names = list(feature_names or [])
        if values.ndim != 2:
            raise ValueError("candidate features must be a two-dimensional matrix")
        finite_rows = np.isfinite(values).all(axis=1)
        result_return: dict[int, np.ndarray] = {}
        result_risk: dict[int, np.ndarray] = {}
        if self.name == "causal_trend_vol_rule":
            trend = _trend_score(values, names)
            risk_proxy = _risk_proxy(values, names)
            for horizon in self.horizons:
                ret_intercept, ret_slope = (self.trend_return_coefs or {}).get(horizon, (0.0, 0.0))
                risk_intercept, risk_slope = (self.trend_risk_coefs or {}).get(horizon, (0.0, 0.0))
                ret = ret_intercept + ret_slope * trend
                risk = np.maximum(0.0, risk_intercept + risk_slope * risk_proxy)
                ret[~finite_rows] = np.nan
                risk[~finite_rows] = np.nan
                result_return[horizon] = ret
                result_risk[horizon] = risk
        else:
            for horizon in self.horizons:
                ret = np.full(len(values), np.nan, dtype=np.float64)
                risk = np.full(len(values), np.nan, dtype=np.float64)
                if np.any(finite_rows):
                    # Some sklearn/numpy combinations emit benign matmul
                    # RuntimeWarnings in Ridge.predict even when the inputs,
                    # coefficients, and returned values are finite. Keep the
                    # diagnostic output clean while the finite check below
                    # remains the actual safety contract.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning,
                            module=r"sklearn\.linear_model\._base",
                        )
                        ret[finite_rows] = self.return_models[horizon].predict(values[finite_rows])
                        risk[finite_rows] = self.risk_models[horizon].predict(values[finite_rows])
                result_return[horizon] = ret
                result_risk[horizon] = np.maximum(0.0, risk)
        return {
            "return": np.column_stack([result_return[h] for h in self.horizons]),
            "risk": np.column_stack([result_risk[h] for h in self.horizons]),
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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_requested_folds(folds: Iterable[int] | None) -> tuple[int, ...]:
    """Validate the hard development-tournament fold allow-list."""
    values = DEV_FOLDS if folds is None else tuple(int(value) for value in folds)
    requested = tuple(sorted(set(values)))
    if not requested:
        raise ValueError("forecast tournament requires at least one development fold")
    forbidden = sorted(set(requested) - set(DEV_FOLDS))
    if forbidden:
        raise ValueError(
            f"forecast tournament is development-only; forbidden folds requested: {forbidden}"
        )
    return requested


def _normalize_index(frame: pd.DataFrame | pd.Series, label: str) -> pd.DataFrame | pd.Series:
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if not result.index.is_monotonic_increasing or not result.index.is_unique:
        raise ValueError(f"{label} index must be sorted and unique")
    if len(result) == 0:
        raise ValueError(f"{label} is empty")
    return result


def _validate_wfo_periods(train_years: int, val_months: int, test_months: int) -> tuple[int, int, int]:
    periods = (int(train_years), int(val_months), int(test_months))
    if any(value <= 0 for value in periods):
        raise ValueError(
            "WFO train_years, val_months, and test_months must all be positive; "
            f"got {periods}"
        )
    return periods


def _assert_wfo_split_contract(
    split: WFOSplit,
    *,
    train_years: int,
    val_months: int,
    test_months: int,
) -> None:
    """Assert that a split was generated by the configured right-exclusive periods."""
    train_years, val_months, test_months = _validate_wfo_periods(
        train_years, val_months, test_months
    )
    expected = {
        "train_end": split.train_start + pd.DateOffset(years=train_years),
        "val_start": split.train_end,
        "val_end": split.val_start + pd.DateOffset(months=val_months),
        "test_start": split.val_end,
        "test_end": split.test_start + pd.DateOffset(months=test_months),
    }
    actual = {
        "train_end": split.train_end,
        "val_start": split.val_start,
        "val_end": split.val_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
    }
    mismatches = {
        name: {"expected": str(expected[name]), "actual": str(actual[name])}
        for name in expected
        if pd.Timestamp(actual[name]) != pd.Timestamp(expected[name])
    }
    if mismatches:
        raise ValueError(
            f"fold {split.fold_idx} does not match configured WFO periods "
            f"(train_years={train_years}, val_months={val_months}, test_months={test_months}): "
            f"{mismatches}"
        )
    if not (split.train_start < split.train_end <= split.val_start < split.val_end <= split.test_start < split.test_end):
        raise ValueError(f"fold {split.fold_idx} has invalid ordered right-exclusive boundaries")


def _validate_feature_schema(features: pd.DataFrame) -> None:
    actual = tuple(str(name) for name in features.columns)
    expected = tuple(EXPECTED_FEATURES)
    if actual != expected:
        raise ValueError(
            "full17 feature schema must exactly match EXPECTED_FEATURES in order; "
            f"expected={list(expected)}, actual={list(actual)}"
        )


def load_development_data(
    feature_path: str | Path,
    returns_path: str | Path,
    *,
    folds: Iterable[int] | None = None,
    train_years: int = 2,
    val_months: int = 3,
    test_months: int = 3,
) -> DevelopmentData:
    """Load only the frozen 2018-2023 development cache.

    The cutoff check is intentionally strict.  A caller cannot accidentally
    point this screen at a later cache and receive a plausible-looking
    result, even if the requested fold list happens to contain only 0/2/8.
    """
    requested = validate_requested_folds(folds)
    train_years, val_months, test_months = _validate_wfo_periods(
        train_years, val_months, test_months
    )
    feature_file = Path(feature_path)
    returns_file = Path(returns_path)
    with pd.option_context("mode.copy_on_write", True):
        features = _normalize_index(pd.read_parquet(feature_file), "feature cache")
        returns_frame = _normalize_index(pd.read_parquet(returns_file), "returns cache")
    if features.index.max() >= DEV_CUTOFF or returns_frame.index.max() >= DEV_CUTOFF:
        raise ValueError(
            f"forecast tournament refuses data at/after development cutoff {DEV_CUTOFF}: "
            f"features={features.index.max()}, returns={returns_frame.index.max()}"
        )
    if not features.index.equals(returns_frame.index):
        raise ValueError("feature and return cache indexes are not exactly aligned")
    if isinstance(returns_frame, pd.DataFrame):
        if "returns" not in returns_frame.columns:
            if returns_frame.shape[1] != 1:
                raise ValueError("returns cache must contain a single `returns` column")
            returns = returns_frame.iloc[:, 0]
        else:
            returns = returns_frame["returns"]
    else:
        returns = returns_frame
    returns = returns.astype(np.float64)
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        # Missing values are not replaced by zero.  The current cache is
        # finite; future caches must choose an explicit imputation contract.
        raise ValueError("feature cache contains non-finite values; no implicit zero fill is allowed")
    if not np.isfinite(returns.to_numpy(dtype=np.float64)).all():
        raise ValueError("returns cache contains non-finite values")
    _validate_feature_schema(features)
    generated_splits = get_wfo_splits(
        features,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
    )
    for split in generated_splits:
        _assert_wfo_split_contract(
            split,
            train_years=train_years,
            val_months=val_months,
            test_months=test_months,
        )
    splits_by_fold = {split.fold_idx: split for split in generated_splits}
    missing = sorted(set(requested) - set(splits_by_fold))
    if missing:
        raise ValueError(f"requested development folds are absent from cache: {missing}")
    splits = tuple(splits_by_fold[fold] for fold in requested)
    for split in splits:
        if split.test_end >= DEV_CUTOFF:
            raise ValueError(f"fold {split.fold_idx} reaches beyond development cutoff")
    return DevelopmentData(
        features=features,
        returns=returns,
        splits=splits,
        feature_path=feature_file,
        returns_path=returns_file,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
    )


def _slice_frame(frame: pd.DataFrame | pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    mask = (frame.index >= start) & (frame.index < end)
    selected = frame.loc[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        raise ValueError(f"empty right-exclusive slice [{start}, {end})")
    return selected


def _feature_quality(frame: pd.DataFrame, *, feature_set: str, fold: int, split_name: str) -> dict[str, Any]:
    """Report zero and missing counts separately without calling zeros valid."""
    result: dict[str, Any] = {
        "fold": int(fold),
        "feature_set": feature_set,
        "split": split_name,
        "rows": int(len(frame)),
        "external": {},
        "quality_flags": [],
    }
    for name in EXTERNAL_FEATURES:
        if name not in frame.columns:
            result["external"][name] = {
                "rows": int(len(frame)),
                "finite_count": 0,
                "missing_count": int(len(frame)),
                "zero_count": 0,
                "nonzero_count": 0,
                "finite_rate": 0.0,
                "missing_rate": 1.0,
                "zero_rate": 0.0,
                "nonzero_rate": 0.0,
                "status": "N/A_excluded_from_ablation",
                "quality_flag": "N/A_external_columns_not_in_ohlcv13",
            }
            result["quality_flags"].append("N/A_external_columns_not_in_ohlcv13")
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        zero = finite & (np.abs(values) <= 1e-12)
        nonzero = finite & ~zero
        missing = ~finite
        denominator = max(len(values), 1)
        quality_flag = "N/A_zero_vs_missing_indistinguishable"
        result["external"][name] = {
            "rows": int(len(values)),
            "finite_count": int(finite.sum()),
            "missing_count": int(missing.sum()),
            "zero_count": int(zero.sum()),
            "nonzero_count": int(nonzero.sum()),
            "finite_rate": float(finite.sum() / denominator),
            "missing_rate": float(missing.sum() / denominator),
            "zero_rate": float(zero.sum() / denominator),
            "nonzero_rate": float(nonzero.sum() / denominator),
            "status": "ok_with_quality_flag",
            "quality_flag": quality_flag,
        }
        result["quality_flags"].append(quality_flag)
    result["quality_flags"] = sorted(set(result["quality_flags"]))
    result["status"] = "ok_with_quality_flag" if result["quality_flags"] else "ok"
    result["contract_note"] = (
        "The cache has no availability mask; observed zero and missing/imputed values "
        "cannot be distinguished. Counts remain separate and no zero fill is applied."
    )
    return result


def feature_ablation_columns(features: pd.DataFrame) -> dict[str, list[str]]:
    """Return the fixed 13-column and full17 ablation schemas."""
    _validate_feature_schema(features)
    return {
        "ohlcv13": list(OHLCV_DERIVED_FEATURES),
        "full17": list(EXPECTED_FEATURES),
    }


def future_targets(
    returns: Sequence[float] | np.ndarray,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    *,
    target_kind: str = "return",
) -> tuple[np.ndarray, np.ndarray]:
    """Build targets for exactly ``t+1 .. t+h`` and right-exclusive masks."""
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("future-target returns must be non-empty and finite")
    if target_kind not in {"return", "risk"}:
        raise ValueError(f"unknown future target kind: {target_kind}")
    horizon_values = tuple(sorted(set(max(1, int(value)) for value in horizons)))
    if not horizon_values:
        raise ValueError("at least one forecast horizon is required")
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    squared_cumulative = np.concatenate(([0.0], np.cumsum(np.square(values))))
    targets = np.zeros((len(values), len(horizon_values)), dtype=np.float64)
    masks = np.zeros_like(targets, dtype=bool)
    for column, horizon in enumerate(horizon_values):
        valid_length = len(values) - horizon
        if valid_length <= 0:
            continue
        # For row t, [t+1, t+h] is represented by cumulative[h+t+1] - cumulative[t+1].
        total = cumulative[horizon + 1 : horizon + 1 + valid_length] - cumulative[1 : 1 + valid_length]
        if target_kind == "return":
            targets[:valid_length, column] = total
        else:
            squared = (
                squared_cumulative[horizon + 1 : horizon + 1 + valid_length]
                - squared_cumulative[1 : 1 + valid_length]
            )
            targets[:valid_length, column] = np.sqrt(np.maximum(squared / float(horizon), 0.0))
        masks[:valid_length, column] = True
    return targets, masks


def _trend_score(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
    def col(name: str) -> np.ndarray:
        if name not in names:
            return np.zeros(len(values), dtype=np.float64)
        return values[:, names.index(name)]

    # Every source column is a pre-shifted/causal cache feature.  No target or
    # future bar is consulted by this rule.
    return 0.50 * col("close_ret") + 0.30 * col("macd") + 0.20 * col("macd_signal")


def _risk_proxy(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
    if "rv_16" in names:
        return np.abs(values[:, names.index("rv_16")])
    if "rv_4" in names:
        return np.abs(values[:, names.index("rv_4")])
    return np.zeros(len(values), dtype=np.float64)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2 or np.unique(x[finite]).size < 2:
        return float(np.nanmean(y[finite])) if np.any(finite) else 0.0, 0.0
    design = np.column_stack((np.ones(int(finite.sum())), x[finite]))
    intercept, slope = np.linalg.lstsq(design, y[finite], rcond=None)[0]
    return float(intercept), float(slope)


def _fit_indices(count: int, max_rows: int) -> np.ndarray:
    if count <= max_rows:
        return np.arange(count, dtype=np.int64)
    # Deterministic temporal subsampling stays inside the training split.
    return np.linspace(0, count - 1, max_rows, dtype=np.int64)


def _model_hyperparameters(name: str, hist_max_iter: int) -> tuple[dict[str, Any], ...]:
    if name == "causal_trend_vol_rule":
        return ({"rule": "causal_trend_vol"},)
    if name == "ridge_direct_forecast":
        return tuple({"alpha": float(alpha)} for alpha in (0.1, 1.0, 10.0))
    if name == "histgb_direct_forecast":
        return (
            {"learning_rate": 0.05, "max_iter": int(hist_max_iter)},
            {"learning_rate": 0.10, "max_iter": max(20, int(hist_max_iter * 2 // 3))},
        )
    raise ValueError(f"unknown forecast candidate: {name}")


def fit_candidate(
    name: str,
    features: pd.DataFrame,
    returns: Sequence[float] | np.ndarray,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    hyperparameters: Mapping[str, Any] | None = None,
    seed: int = 7,
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS,
) -> FittedCandidate:
    """Fit a causal direct-return and direct-risk candidate on train only."""
    if len(features) != len(returns):
        raise ValueError("fit features and returns must be aligned")
    horizon_values = tuple(sorted(set(max(1, int(value)) for value in horizons)))
    params = dict(hyperparameters or _model_hyperparameters(name, DEFAULT_HIST_MAX_ITER)[0])
    values = features.to_numpy(dtype=np.float64)
    names = list(features.columns)
    finite_x = np.isfinite(values).all(axis=1)
    selected_indices = _fit_indices(len(features), max_fit_rows)
    fit_rows = int(len(selected_indices))
    dropped_nonfinite = int(np.sum(~finite_x[selected_indices]))
    return_targets, return_masks = future_targets(returns, horizon_values, target_kind="return")
    risk_targets, risk_masks = future_targets(returns, horizon_values, target_kind="risk")

    if name == "causal_trend_vol_rule":
        trend = _trend_score(values, names)
        risk_proxy = _risk_proxy(values, names)
        ret_coefs: dict[int, tuple[float, float]] = {}
        risk_coefs: dict[int, tuple[float, float]] = {}
        for column, horizon in enumerate(horizon_values):
            selected = selected_indices[return_masks[selected_indices, column] & finite_x[selected_indices]]
            ret_coefs[horizon] = _fit_affine(trend[selected], return_targets[selected, column])
            selected_risk = selected_indices[risk_masks[selected_indices, column] & finite_x[selected_indices]]
            risk_coefs[horizon] = _fit_affine(risk_proxy[selected_risk], risk_targets[selected_risk, column])
        return FittedCandidate(
            name=name,
            hyperparameters=params,
            horizons=horizon_values,
            return_models={},
            risk_models={},
            trend_return_coefs=ret_coefs,
            trend_risk_coefs=risk_coefs,
            fit_rows=fit_rows,
            dropped_nonfinite_rows=dropped_nonfinite,
        )

    return_models: dict[int, Any] = {}
    risk_models: dict[int, Any] = {}
    for column, horizon in enumerate(horizon_values):
        valid_return = selected_indices[return_masks[selected_indices, column] & finite_x[selected_indices]]
        valid_risk = selected_indices[risk_masks[selected_indices, column] & finite_x[selected_indices]]
        if len(valid_return) < 2 or len(valid_risk) < 2:
            raise ValueError(f"candidate {name} has insufficient finite rows at horizon {horizon}")
        if name == "ridge_direct_forecast":
            return_model = make_pipeline(
                StandardScaler(),
                # ``lsqr`` is deterministic and avoids the unstable dense
                # normal-equation path on the wide, highly correlated
                # rolling-feature cache.
                Ridge(alpha=float(params["alpha"]), solver="lsqr", max_iter=1000, tol=1e-8),
            )
            risk_model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(params["alpha"]), solver="lsqr", max_iter=1000, tol=1e-8),
            )
        elif name == "histgb_direct_forecast":
            common = {
                "learning_rate": float(params["learning_rate"]),
                "max_iter": int(params["max_iter"]),
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": int(seed),
            }
            return_model = HistGradientBoostingRegressor(**common)
            risk_model = HistGradientBoostingRegressor(**common)
        else:
            raise ValueError(f"unknown forecast candidate: {name}")
        return_model.fit(values[valid_return], return_targets[valid_return, column])
        risk_model.fit(values[valid_risk], risk_targets[valid_risk, column])
        return_models[horizon] = return_model
        risk_models[horizon] = risk_model
    return FittedCandidate(
        name=name,
        hyperparameters=params,
        horizons=horizon_values,
        return_models=return_models,
        risk_models=risk_models,
        fit_rows=fit_rows,
        dropped_nonfinite_rows=dropped_nonfinite,
    )


def forecast_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    horizons: Iterable[int],
    *,
    split: str,
    target_kind: str,
) -> list[dict[str, Any]]:
    """Calculate direct forecast metrics with explicit N/A reasons."""
    predictions = np.asarray(prediction, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    masks = np.asarray(mask, dtype=bool)
    if predictions.shape != targets.shape or targets.shape != masks.shape:
        raise ValueError("prediction, target, and mask shapes must match")
    records = []
    for column, horizon in enumerate(tuple(horizons)):
        valid = masks[:, column] & np.isfinite(predictions[:, column]) & np.isfinite(targets[:, column])
        reason: list[str] = []
        metrics: dict[str, float | None] = {}
        if not np.any(valid):
            records.append(
                {
                    "split": split,
                    "target_kind": target_kind,
                    "horizon": int(horizon),
                    "status": "N/A",
                    "n_valid": 0,
                    "metrics": metrics,
                    "reason": "no finite forecast/target rows",
                }
            )
            continue
        actual = targets[valid, column]
        predicted = predictions[valid, column]
        if target_kind == "risk":
            metrics["sign_accuracy"] = None
            reason.append("sign accuracy is N/A for a one-sided non-negative realized-risk target")
        else:
            metrics["sign_accuracy"] = float(np.mean(np.sign(actual) == np.sign(predicted)))
        if np.unique(actual).size < 2 or np.unique(predicted).size < 2:
            metrics["spearman_ic"] = None
            reason.append("Spearman IC is N/A for a constant target or prediction")
        else:
            metrics["spearman_ic"] = float(spearmanr(actual, predicted).statistic)
        metrics["mae"] = float(np.mean(np.abs(predicted - actual)))
        metrics["rmse"] = float(np.sqrt(np.mean(np.square(predicted - actual))))
        records.append(
            {
                "split": split,
                "target_kind": target_kind,
                "horizon": int(horizon),
                "status": "ok",
                "n_valid": int(valid.sum()),
                "metrics": metrics,
                **({"reason": "; ".join(reason)} if reason else {}),
            }
        )
    return records


def policy_positions(
    forecast: Sequence[float] | np.ndarray,
    params: PolicyParams,
    *,
    benchmark: float = BENCHMARK_POSITION,
    min_position: float = 0.50,
    max_position: float = 1.12,
    max_position_step: float = DEFAULT_MAX_POSITION_STEP,
) -> np.ndarray:
    """Convert a causal forecast to a bounded hysteretic overlay path."""
    score = np.asarray(forecast, dtype=np.float64).reshape(-1)
    if not np.isfinite(score).all():
        raise ValueError("policy forecast contains non-finite values")
    if params.threshold < 0 or params.overlay_magnitude < 0 or not 0 <= params.hysteresis < 1:
        raise ValueError("invalid policy threshold, magnitude, or hysteresis")
    positions = np.full(len(score), float(benchmark), dtype=np.float64)
    current = float(benchmark)
    hold_remaining = 0
    active = False
    for index, value in enumerate(score):
        absolute = abs(float(value))
        if hold_remaining > 0:
            hold_remaining -= 1
        else:
            enter = absolute > float(params.threshold)
            stay = absolute > float(params.threshold) * (1.0 - float(params.hysteresis))
            active = stay if active else enter
            raw_overlay = np.sign(value) * float(params.overlay_magnitude) if active else 0.0
            target = float(np.clip(benchmark + raw_overlay, min_position, max_position))
            delta = float(np.clip(target - current, -max_position_step, max_position_step))
            next_position = float(np.clip(current + delta, min_position, max_position))
            if abs(next_position - current) > 1e-12 and params.min_hold > 0:
                hold_remaining = int(params.min_hold)
            current = next_position
        positions[index] = current
    return positions


def _threshold_grid(validation_forecast: np.ndarray) -> tuple[float, ...]:
    absolute = np.abs(np.asarray(validation_forecast, dtype=np.float64))
    finite = absolute[np.isfinite(absolute)]
    if len(finite) == 0:
        return (0.0,)
    values = [0.0, float(np.quantile(finite, 0.50)), float(np.quantile(finite, 0.75))]
    return tuple(sorted(set(values)))


def _validation_policy_selection(
    validation_forecast: np.ndarray,
    validation_returns: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    benchmark: float,
    execution_delay: int = DEFAULT_EXECUTION_DELAY_BARS,
    overlay_grid: Iterable[float],
    hysteresis_grid: Iterable[float],
    min_hold_grid: Iterable[int],
    constant_exposure: float,
    execution_grid: Iterable[int] | None = None,
) -> tuple[PolicyParams, dict[str, Any]]:
    """Choose policy controls on validation under one fixed execution contract.

    ``execution_grid`` is retained as a compatibility-only argument for old
    callers. It is deliberately ignored: execution delay is an operational
    contract, not a validation hyperparameter. Lagged executions are emitted
    later as fixed sensitivity diagnostics.
    """
    execution_delay = int(execution_delay)
    if execution_delay < 0:
        raise ValueError("fixed execution delay must be non-negative")
    if execution_grid is not None:
        legacy_delays = tuple(int(value) for value in execution_grid)
        if any(value < 0 for value in legacy_delays):
            raise ValueError("execution delays must be non-negative")
    constant_metrics = backtest_metrics(
        validation_returns,
        np.full(len(validation_returns), constant_exposure, dtype=np.float64),
        cfg,
        benchmark=benchmark,
        execution_delay_bars=execution_delay,
    )
    trials: list[dict[str, Any]] = []
    for threshold in _threshold_grid(validation_forecast):
        for magnitude in tuple(float(value) for value in overlay_grid):
            for hysteresis in tuple(float(value) for value in hysteresis_grid):
                for min_hold in tuple(int(value) for value in min_hold_grid):
                    params = PolicyParams(
                        threshold=threshold,
                        overlay_magnitude=magnitude,
                        hysteresis=hysteresis,
                        min_hold=min_hold,
                        execution_delay=execution_delay,
                    )
                    positions = policy_positions(validation_forecast, params, benchmark=benchmark)
                    metrics = backtest_metrics(
                        validation_returns,
                        positions,
                        cfg,
                        benchmark=benchmark,
                        execution_delay_bars=execution_delay,
                    )
                    score = (
                        float(metrics["alpha_excess_pt"])
                        - max(0.0, float(metrics["maxdd_delta_pt"]))
                        - 0.05 * float(metrics["turnover"])
                    )
                    trials.append(
                        {
                            "params": params,
                            "metrics": metrics,
                            "score": float(score),
                        }
                    )
    if not trials:
        raise ValueError("validation policy grid is empty")
    selected = max(
        enumerate(trials),
        key=lambda pair: (float(pair[1]["score"]), -int(pair[0])),
    )[1]
    params = selected["params"]
    return params, {
        "split": "validation",
        "status": "selected_on_validation",
        "constant_exposure": float(constant_exposure),
        "constant_metrics": constant_metrics,
        "selected_params": {
            "threshold": float(params.threshold),
            "overlay_magnitude": float(params.overlay_magnitude),
            "hysteresis": float(params.hysteresis),
            "min_hold": int(params.min_hold),
            "execution_delay": int(params.execution_delay),
        },
        "execution_delay_selection": "fixed_operational_contract",
        "selected_metrics": selected["metrics"],
        "selection_score": float(selected["score"]),
        "candidate_trial_count": int(len(trials)),
    }


def timing_economics(
    returns: np.ndarray,
    dynamic_positions: np.ndarray,
    constant_exposure: float,
    cfg: Mapping[str, Any],
    *,
    benchmark: float,
    execution_delay: int,
    null_shifts: Iterable[int] = DEFAULT_NULL_SHIFTS,
) -> dict[str, Any]:
    """Evaluate dynamic, constant, fixed-lag, and deterministic null paths.

    ``execution_delay`` is the fixed operational delay for the main result.
    Lagged paths add the pre-registered sensitivity lag; circular shifts keep
    the operational delay unchanged and are deterministic nulls.
    """
    execution_delay = int(execution_delay)
    if execution_delay < 0:
        raise ValueError("fixed execution delay must be non-negative")
    lag_values = tuple(int(lag) for lag in EXECUTION_DELAY_SENSITIVITY_LAGS)
    null_values = tuple(int(shift) for shift in null_shifts)
    if any(value < 0 for value in null_values):
        raise ValueError("null shifts must be non-negative")
    dynamic = backtest_metrics(
        returns,
        dynamic_positions,
        cfg,
        benchmark=benchmark,
        execution_delay_bars=execution_delay,
    )
    constant = backtest_metrics(
        returns,
        np.full(len(returns), constant_exposure, dtype=np.float64),
        cfg,
        benchmark=benchmark,
        execution_delay_bars=execution_delay,
    )
    lags = {
        str(lag): backtest_metrics(
            returns,
            dynamic_positions,
            cfg,
            benchmark=benchmark,
            execution_delay_bars=execution_delay + int(lag),
        )
        for lag in lag_values
    }
    nulls = {
        str(shift): backtest_metrics(
            returns,
            circular_shift_positions(dynamic_positions, int(shift)),
            cfg,
            benchmark=benchmark,
            execution_delay_bars=execution_delay,
        )
        for shift in null_values
    }
    return {
        "operational_execution_delay_bars": execution_delay,
        "execution_delay_sensitivity_lags": list(lag_values),
        "null_shifts": list(null_values),
        "dynamic": dynamic,
        "constant": constant,
        "lags": lags,
        "nulls": nulls,
        "timing_increment_alpha_excess_pt": float(
            dynamic["alpha_excess_pt"] - constant["alpha_excess_pt"]
        ),
        "constant_exposure_component_alpha_excess_pt": float(constant["alpha_excess_pt"]),
        "dynamic_alpha_excess_pt": float(dynamic["alpha_excess_pt"]),
    }


def _fold_bounds(split: WFOSplit) -> dict[str, str]:
    return {
        "train_start": str(split.train_start),
        "train_end_exclusive": str(split.train_end),
        "validation_start": str(split.val_start),
        "validation_end_exclusive": str(split.val_end),
        "test_start": str(split.test_start),
        "test_end_exclusive": str(split.test_end),
    }


def _metric_at_horizon(records: Sequence[Mapping[str, Any]], horizon: int, metric: str) -> float | None:
    for row in records:
        if int(row.get("horizon", -1)) == int(horizon):
            value = (row.get("metrics") or {}).get(metric)
            return None if value is None else float(value)
    return None


def _safe_median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def aggregate_candidate_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_horizon: int = DEFAULT_POLICY_HORIZON,
    max_turnover: float = 6.5,
    max_dd_worsening_over_constant_pt: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply the pre-registered successive-halving gate to selected test rows.

    A dynamic path is not credited for merely beating a constant exposure. It
    must also beat each fixed-lag and deterministic circular-shift null by the
    same fixed AlphaEx margin on the aggregate median and on at least the
    pre-registered fold win rate. This keeps the development tournament from
    promoting a simple exposure or a lucky timing alignment.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["feature_set"]), str(row["candidate"])), []).append(row)

    def alpha(row: Mapping[str, Any], path: str, key: str | None = None) -> float | None:
        try:
            value: Any = row["test_economics"][path]
            if key is not None:
                value = value[key]
            value = value["alpha_excess_pt"]
            value = float(value)
        except (KeyError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def superiority(
        dynamic: Sequence[float | None], comparator: Sequence[float | None]
    ) -> dict[str, float | int | None]:
        differences = [
            float(dynamic_value) - float(comparator_value)
            for dynamic_value, comparator_value in zip(dynamic, comparator)
            if dynamic_value is not None
            and comparator_value is not None
            and np.isfinite(float(dynamic_value))
            and np.isfinite(float(comparator_value))
        ]
        wins = sum(value > TIMING_SUPERIORITY_MARGIN_PT for value in differences)
        return {
            "median_alpha_difference_pt": _safe_median(differences),
            "win_folds": int(wins),
            "win_rate": float(wins / len(differences)) if differences else None,
            "required_win_rate": float(TIMING_MIN_WIN_RATE),
            "margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
        }

    output = []
    for (feature_set, candidate), values in sorted(grouped.items()):
        ic_values = [
            _metric_at_horizon(row["test_return_metrics"], policy_horizon, "spearman_ic")
            for row in values
        ]
        dynamic_alpha = [alpha(row, "dynamic") for row in values]
        dynamic_dd = [
            float(row["test_economics"]["dynamic"]["maxdd_delta_pt"])
            for row in values
            if "test_economics" in row
            and "dynamic" in row["test_economics"]
            and row["test_economics"]["dynamic"].get("maxdd_delta_pt") is not None
        ]
        dynamic_turnover = [
            float(row["test_economics"]["dynamic"]["turnover"])
            for row in values
            if "test_economics" in row
            and "dynamic" in row["test_economics"]
            and row["test_economics"]["dynamic"].get("turnover") is not None
        ]
        constant_dd = [
            float(row["test_economics"]["constant"]["maxdd_delta_pt"])
            for row in values
            if "test_economics" in row
            and "constant" in row["test_economics"]
            and row["test_economics"]["constant"].get("maxdd_delta_pt") is not None
        ]
        constant_alpha = [alpha(row, "constant") for row in values]
        lag1_alpha = [alpha(row, "lags", "1") for row in values]
        lag16_alpha = [alpha(row, "lags", "16") for row in values]
        null1_alpha = [alpha(row, "nulls", "1") for row in values]
        null16_alpha = [alpha(row, "nulls", "16") for row in values]
        null64_alpha = [alpha(row, "nulls", "64") for row in values]
        timing_increment = [
            float(dynamic_value) - float(constant_value)
            for dynamic_value, constant_value in zip(dynamic_alpha, constant_alpha)
            if dynamic_value is not None
            and constant_value is not None
            and np.isfinite(float(dynamic_value))
            and np.isfinite(float(constant_value))
        ]
        comparisons = {
            "constant": superiority(dynamic_alpha, constant_alpha),
            "lag1": superiority(dynamic_alpha, lag1_alpha),
            "lag16": superiority(dynamic_alpha, lag16_alpha),
            "null_shift1": superiority(dynamic_alpha, null1_alpha),
            "null_shift16": superiority(dynamic_alpha, null16_alpha),
            "null_shift64": superiority(dynamic_alpha, null64_alpha),
        }

        def beats(comparison: Mapping[str, Any]) -> bool:
            median = comparison.get("median_alpha_difference_pt")
            win_rate = comparison.get("win_rate")
            return bool(
                median is not None
                and float(median) > TIMING_SUPERIORITY_MARGIN_PT
                and win_rate is not None
                and float(win_rate) >= TIMING_MIN_WIN_RATE
            )

        positive_ic = sum(value is not None and value > 0.0 for value in ic_values)
        criteria = {
            "ic_sign_stable": bool(positive_ic >= max(2, (len(values) + 1) // 2)),
            "median_timing_increment_positive": bool(
                _safe_median(timing_increment) is not None
                and float(_safe_median(timing_increment)) > 0.0
            ),
            "median_alpha_excess_positive": bool(
                _safe_median(dynamic_alpha) is not None
                and float(_safe_median(dynamic_alpha)) > 0.0
            ),
            # These are the pre-registered temporal-destruction gate. Lag1,
            # shift1, and shift16 remain visible robustness diagnostics but
            # are not hard failures because they can preserve real short-term
            # autocorrelation in an otherwise valid signal.
            "timing_beats_constant": beats(comparisons["constant"]),
            "timing_beats_lag16": beats(comparisons["lag16"]),
            "timing_beats_null_shift64": beats(comparisons["null_shift64"]),
            "dd_turnover_tradeoff": bool(
                _safe_median(dynamic_dd) is not None
                and _safe_median(constant_dd) is not None
                and _safe_median(dynamic_turnover) is not None
                and float(_safe_median(dynamic_dd))
                <= float(_safe_median(constant_dd)) + float(max_dd_worsening_over_constant_pt)
                and float(_safe_median(dynamic_turnover)) <= float(max_turnover)
            ),
        }
        output.append(
            {
                "feature_set": feature_set,
                "candidate": candidate,
                "folds": int(len(values)),
                "test_median_spearman_ic": _safe_median(ic_values),
                "test_positive_ic_folds": int(positive_ic),
                "test_median_alpha_excess_pt": _safe_median(dynamic_alpha),
                "test_median_timing_increment_alpha_excess_pt": _safe_median(timing_increment),
                "timing_superiority": comparisons,
                "test_median_maxdd_delta_pt": _safe_median(dynamic_dd),
                "test_median_constant_maxdd_delta_pt": _safe_median(constant_dd),
                "test_median_turnover": _safe_median(dynamic_turnover),
                "timing_superiority_margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
                "timing_min_win_rate": float(TIMING_MIN_WIN_RATE),
                "criteria": criteria,
                "status": "pass" if all(criteria.values()) else "fail",
                "selection_contract": "validation-only; development test report-only tournament screen",
                "failure_reasons": [name for name, passed in criteria.items() if not passed],
            }
        )
    return output


def _provenance(
    *,
    data: DevelopmentData,
    cfg: Mapping[str, Any],
    config_path: str,
    seed: int,
    horizons: tuple[int, ...],
    folds: tuple[int, ...],
    execution_delay: int,
) -> dict[str, Any]:
    contract = {
        "allowed_folds": list(folds),
        "cutoff_exclusive": str(DEV_CUTOFF),
        "right_exclusive_split": True,
        "train_years": int(data.train_years),
        "val_months": int(data.val_months),
        "test_months": int(data.test_months),
        "benchmark_position": BENCHMARK_POSITION,
        "horizons": list(horizons),
        "target_windows": "t+1..t+h",
        "feature_ablation": {
            "ohlcv13": list(OHLCV_DERIVED_FEATURES),
            "full17": list(EXPECTED_FEATURES),
        },
        "feature_schema_exact": True,
        "external_zero_missing_distinction": False,
        "execution_delay": {
            "operational_bars": int(execution_delay),
            "sensitivity_lags": list(EXECUTION_DELAY_SENSITIVITY_LAGS),
        },
        "timing_superiority_margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
        "timing_min_win_rate": float(TIMING_MIN_WIN_RATE),
    }
    return {
        "commit_hash": _git_commit(),
        "config_path": str(config_path),
        "config_sha256": config_fingerprint(dict(cfg)),
        "source_sha256": source_fingerprint(),
        "data_contract": contract,
        "data_contract_sha256": _canonical_sha256(contract),
        "data_artifacts": {
            "features": str(data.feature_path),
            "features_sha256": sha256_file(data.feature_path),
            "returns": str(data.returns_path),
            "returns_sha256": sha256_file(data.returns_path),
        },
        "data_sha256": _canonical_sha256(
            {
                "features_sha256": sha256_file(data.feature_path),
                "returns_sha256": sha256_file(data.returns_path),
                "folds": list(folds),
                "horizons": list(horizons),
            }
        ),
        "seed": int(seed),
    }


def run_tournament(
    *,
    data: DevelopmentData,
    cfg: Mapping[str, Any],
    config_path: str,
    seed: int = 7,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    policy_horizon: int = DEFAULT_POLICY_HORIZON,
    feature_sets: Iterable[str] = DEFAULT_FEATURE_SETS,
    candidates: Iterable[str] = (
        "causal_trend_vol_rule",
        "ridge_direct_forecast",
        "histgb_direct_forecast",
    ),
    execution_delay_bars: int | None = None,
    execution_grid: Iterable[int] | None = None,
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS,
    hist_max_iter: int = DEFAULT_HIST_MAX_ITER,
    output_dir: str | Path = "docs/forecast_tournament_plan011_dev",
) -> dict[str, Any]:
    """Run the development-only forecast/timing tournament and write outputs."""
    horizons_tuple = tuple(sorted(set(max(1, int(value)) for value in horizons)))
    if not horizons_tuple:
        raise ValueError("tournament requires at least one forecast horizon")
    if policy_horizon not in horizons_tuple:
        raise ValueError("policy horizon must be present in the direct forecast horizons")
    feature_schema = feature_ablation_columns(data.features)
    feature_set_tuple = tuple(str(value) for value in feature_sets)
    unknown_feature_sets = sorted(set(feature_set_tuple) - set(feature_schema))
    if unknown_feature_sets:
        raise ValueError(f"unknown feature ablations: {unknown_feature_sets}")
    candidate_tuple = tuple(str(value) for value in candidates)
    if not candidate_tuple:
        raise ValueError("tournament candidate list is empty")
    eval_cfg = cfg.get("eval", {})
    configured_delay = eval_cfg.get(
        "forecast_execution_delay_bars", DEFAULT_EXECUTION_DELAY_BARS
    )
    execution_delay = int(configured_delay if execution_delay_bars is None else execution_delay_bars)
    if execution_delay < 0:
        raise ValueError("fixed execution delay must be non-negative")
    if execution_grid is not None:
        # Keep the pre-refactor keyword import-compatible, but never let it
        # become a validation search dimension.
        legacy_delays = tuple(int(value) for value in execution_grid)
        if any(value < 0 for value in legacy_delays):
            raise ValueError("execution delays must be non-negative")
    if any(split.fold_idx not in DEV_FOLDS for split in data.splits):
        raise ValueError("DevelopmentData contains a fold outside the hard development allow-list")
    folds = tuple(split.fold_idx for split in data.splits)
    provenance = _provenance(
        data=data,
        cfg=cfg,
        config_path=config_path,
        seed=seed,
        horizons=horizons_tuple,
        folds=folds,
        execution_delay=execution_delay,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ledger_path = output_path / "forecast_tournament_ledger.jsonl"
    ledger_records: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    feature_quality_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for split in data.splits:
        train_features = _slice_frame(data.features, split.train_start, split.train_end)
        val_features = _slice_frame(data.features, split.val_start, split.val_end)
        test_features = _slice_frame(data.features, split.test_start, split.test_end)
        train_returns = _slice_frame(data.returns, split.train_start, split.train_end).to_numpy(dtype=np.float64)
        val_returns = _slice_frame(data.returns, split.val_start, split.val_end).to_numpy(dtype=np.float64)
        test_returns = _slice_frame(data.returns, split.test_start, split.test_end).to_numpy(dtype=np.float64)
        bounds = _fold_bounds(split)
        for feature_set in feature_set_tuple:
            columns = feature_schema[feature_set]
            train_x = train_features.loc[:, columns]
            val_x = val_features.loc[:, columns]
            test_x = test_features.loc[:, columns]
            for split_name, frame in (
                ("train", train_x),
                ("validation", val_x),
                ("test", test_x),
            ):
                quality = _feature_quality(frame, feature_set=feature_set, fold=split.fold_idx, split_name=split_name)
                quality["fold_bounds"] = bounds
                feature_quality_rows.append(quality)
                ledger_records.append(
                    {
                        **provenance,
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "feature_coverage",
                        "status": quality["status"],
                        "fold": int(split.fold_idx),
                        "feature_set": feature_set,
                        "fit_split": "none",
                        "selection_split": "none",
                        "report_split": split_name,
                        "fold_bounds": bounds,
                        "metrics": quality,
                    }
                )

            constant_exposure, constant_trials = select_fixed_exposure(
                val_returns,
                CONSTANT_EXPOSURE_GRID,
                cfg,
                benchmark=BENCHMARK_POSITION,
                execution_delay_bars=execution_delay,
            )
            constant_selection = {
                "split": "validation",
                "status": "selected_on_validation",
                "candidate_grid": list(CONSTANT_EXPOSURE_GRID),
                "selected_candidate": float(constant_exposure),
                "selection_score": float(
                    max(constant_trials, key=lambda row: float(row["score"]))["score"]
                ),
            }
            for candidate in candidate_tuple:
                candidate_started = time.perf_counter()
                hyper_trials: list[dict[str, Any]] = []
                for hyperparameters in _model_hyperparameters(candidate, hist_max_iter):
                    trial_started = time.perf_counter()
                    try:
                        fitted = fit_candidate(
                            candidate,
                            train_x,
                            train_returns,
                            horizons=horizons_tuple,
                            hyperparameters=hyperparameters,
                            seed=seed,
                            max_fit_rows=max_fit_rows,
                        )
                        val_prediction = fitted.predict(val_x, columns)
                        val_return_targets, val_return_mask = future_targets(
                            val_returns, horizons_tuple, target_kind="return"
                        )
                        val_risk_targets, val_risk_mask = future_targets(
                            val_returns, horizons_tuple, target_kind="risk"
                        )
                        val_return_metrics = forecast_metrics(
                            val_prediction["return"],
                            val_return_targets,
                            val_return_mask,
                            horizons_tuple,
                            split="validation",
                            target_kind="return",
                        )
                        val_risk_metrics = forecast_metrics(
                            val_prediction["risk"],
                            val_risk_targets,
                            val_risk_mask,
                            horizons_tuple,
                            split="validation",
                            target_kind="risk",
                        )
                        policy_column = horizons_tuple.index(policy_horizon)
                        policy_forecast = val_prediction["return"][:, policy_column]
                        policy, policy_selection = _validation_policy_selection(
                            policy_forecast,
                            val_returns,
                            cfg,
                            benchmark=BENCHMARK_POSITION,
                            execution_delay=execution_delay,
                            overlay_grid=DEFAULT_OVERLAY_GRID,
                            hysteresis_grid=DEFAULT_HYSTERESIS_GRID,
                            min_hold_grid=DEFAULT_MIN_HOLD_GRID,
                            constant_exposure=constant_exposure,
                        )
                        hyper_trials.append(
                            {
                                "fitted": fitted,
                                "validation_prediction": val_prediction,
                                "validation_return_metrics": val_return_metrics,
                                "validation_risk_metrics": val_risk_metrics,
                                "policy": policy,
                                "policy_selection": policy_selection,
                                "runtime_seconds": float(time.perf_counter() - trial_started),
                            }
                        )
                    except (ValueError, RuntimeError) as exc:
                        ledger_records.append(
                            {
                                **provenance,
                                "schema_version": SCHEMA_VERSION,
                                "record_type": "forecast_candidate_fit",
                                "status": "N/A",
                                "candidate": candidate,
                                "feature_set": feature_set,
                                "fold": int(split.fold_idx),
                                "fold_bounds": bounds,
                                "fit_split": "train",
                                "selection_split": "validation",
                                "report_split": "development_test",
                                "hyperparameters": dict(hyperparameters),
                                "runtime_seconds": float(time.perf_counter() - trial_started),
                                "reason": str(exc),
                            }
                        )
                if not hyper_trials:
                    ledger_records.append(
                        {
                            **provenance,
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "forecast_tournament_candidate",
                            "status": "N/A",
                            "candidate": candidate,
                            "feature_set": feature_set,
                            "fold": int(split.fold_idx),
                            "fold_bounds": bounds,
                            "fit_split": "train",
                            "selection_split": "validation",
                            "report_split": "development_test",
                            "reason": "all hyperparameter fits failed",
                        }
                    )
                    continue
                selected_trial = max(
                    enumerate(hyper_trials),
                    key=lambda pair: (
                        float(pair[1]["policy_selection"]["selection_score"]),
                        -int(pair[0]),
                    ),
                )[1]
                fitted = selected_trial["fitted"]
                policy: PolicyParams = selected_trial["policy"]
                test_prediction = fitted.predict(test_x, columns)
                test_return_targets, test_return_mask = future_targets(
                    test_returns, horizons_tuple, target_kind="return"
                )
                test_risk_targets, test_risk_mask = future_targets(
                    test_returns, horizons_tuple, target_kind="risk"
                )
                test_return_metrics = forecast_metrics(
                    test_prediction["return"],
                    test_return_targets,
                    test_return_mask,
                    horizons_tuple,
                    split="development_test",
                    target_kind="return",
                )
                test_risk_metrics = forecast_metrics(
                    test_prediction["risk"],
                    test_risk_targets,
                    test_risk_mask,
                    horizons_tuple,
                    split="development_test",
                    target_kind="risk",
                )
                policy_column = horizons_tuple.index(policy_horizon)
                test_positions = policy_positions(
                    test_prediction["return"][:, policy_column],
                    policy,
                    benchmark=BENCHMARK_POSITION,
                )
                economics = timing_economics(
                    test_returns,
                    test_positions,
                    constant_exposure,
                    cfg,
                    benchmark=BENCHMARK_POSITION,
                    execution_delay=execution_delay,
                )
                row = {
                    **provenance,
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "forecast_tournament_candidate",
                    "status": "ok",
                    "candidate": candidate,
                    "feature_set": feature_set,
                    "fold": int(split.fold_idx),
                    "fold_bounds": bounds,
                    "fit_split": "train",
                    "selection_split": "validation",
                    "report_split": "development_test",
                    "test_is_report_only": True,
                    "hyperparameters": fitted.hyperparameters,
                    "fit_rows": int(fitted.fit_rows),
                    "fit_rows_dropped_nonfinite": int(fitted.dropped_nonfinite_rows),
                    "policy_horizon": int(policy_horizon),
                    "policy": {
                        "threshold": float(policy.threshold),
                        "overlay_magnitude": float(policy.overlay_magnitude),
                        "hysteresis": float(policy.hysteresis),
                        "min_hold": int(policy.min_hold),
                        "execution_delay": int(policy.execution_delay),
                    },
                    "constant_selection": constant_selection,
                    "validation_policy_selection": selected_trial["policy_selection"],
                    "validation_return_metrics": selected_trial["validation_return_metrics"],
                    "validation_risk_metrics": selected_trial["validation_risk_metrics"],
                    "test_return_metrics": test_return_metrics,
                    "test_risk_metrics": test_risk_metrics,
                    "test_economics": economics,
                    "runtime_seconds": float(time.perf_counter() - candidate_started),
                }
                selected_rows.append(row)
                ledger_records.append(row)
                for target_kind, metric_rows in (
                    ("return", test_return_metrics),
                    ("risk", test_risk_metrics),
                ):
                    for metric_row in metric_rows:
                        ledger_records.append(
                            {
                                **provenance,
                                "schema_version": SCHEMA_VERSION,
                                "record_type": "forecast_metric",
                                "status": metric_row["status"],
                                "candidate": candidate,
                                "feature_set": feature_set,
                                "fold": int(split.fold_idx),
                                "fold_bounds": bounds,
                                "fit_split": "train",
                                "selection_split": "validation",
                                "report_split": "development_test",
                                "test_is_report_only": True,
                                **metric_row,
                            }
                        )
                economic_rows = [
                    ("dynamic", economics["dynamic"]),
                    ("constant_validation_selected", economics["constant"]),
                    *[(f"lag_{lag}", metrics) for lag, metrics in economics["lags"].items()],
                    *[(f"null_shift_{shift}", metrics) for shift, metrics in economics["nulls"].items()],
                ]
                for method, metrics in economic_rows:
                    ledger_records.append(
                        {
                            **provenance,
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "economic_metric",
                            "status": "ok",
                            "candidate": candidate,
                            "feature_set": feature_set,
                            "fold": int(split.fold_idx),
                            "method": method,
                            "fold_bounds": bounds,
                            "fit_split": "train",
                            "selection_split": "validation",
                            "report_split": "development_test",
                            "test_is_report_only": True,
                            "metrics": metrics,
                        }
                    )
                ledger_records.append(
                    {
                        **provenance,
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "timing_attribution",
                        "status": "ok",
                        "candidate": candidate,
                        "feature_set": feature_set,
                        "fold": int(split.fold_idx),
                        "fold_bounds": bounds,
                        "fit_split": "train",
                        "selection_split": "validation",
                        "report_split": "development_test",
                        "test_is_report_only": True,
                        "metrics": {
                            "constant_exposure_component_alpha_excess_pt": economics[
                                "constant_exposure_component_alpha_excess_pt"
                            ],
                            "timing_increment_alpha_excess_pt": economics[
                                "timing_increment_alpha_excess_pt"
                            ],
                            "dynamic_alpha_excess_pt": economics["dynamic_alpha_excess_pt"],
                            "lag1": economics["lags"]["1"],
                            "lag16": economics["lags"]["16"],
                            "null_shift1": economics["nulls"]["1"],
                            "null_shift16": economics["nulls"]["16"],
                            "null_shift64": economics["nulls"]["64"],
                        },
                    }
                )

    gate_rows = aggregate_candidate_gate(selected_rows, policy_horizon=policy_horizon)
    for gate in gate_rows:
        ledger_records.append(
            {
                **provenance,
                "schema_version": SCHEMA_VERSION,
                "record_type": "successive_halving_gate",
                "status": gate["status"],
                "fold": None,
                "fit_split": "train",
                "selection_split": "validation",
                "report_split": "development_test",
                "test_is_report_only": True,
                **gate,
            }
        )
    next_wave = [
        {"feature_set": row["feature_set"], "candidate": row["candidate"]}
        for row in sorted(
            gate_rows,
            key=lambda row: float(row["test_median_alpha_excess_pt"]),
            reverse=True,
        )
        if row["status"] == "pass"
    ]
    runtime = float(time.perf_counter() - started)
    result = {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "record_type": "forecast_tournament_report",
        "status": "complete",
        "folds": list(folds),
        "fold_count": len(folds),
        "horizons": list(horizons_tuple),
        "policy_horizon": int(policy_horizon),
        "feature_sets": list(feature_set_tuple),
        "candidates": list(candidate_tuple),
        "execution_delay_bars": int(execution_delay),
        "execution_delay_sensitivity_lags": list(EXECUTION_DELAY_SENSITIVITY_LAGS),
        "constant_exposure_grid": list(CONSTANT_EXPOSURE_GRID),
        "selection_contract": (
            "fit train only; select hyperparameters and policy controls on validation "
            "under the fixed operational execution delay; "
            "development test is report-only and may shortlist the next wave"
        ),
        "rows": selected_rows,
        "feature_quality": feature_quality_rows,
        "gate": gate_rows,
        "next_wave_candidates": next_wave,
        "all_candidates_failed_gate": not bool(next_wave),
        "runtime_seconds": runtime,
        "ledger_path": str(ledger_path),
    }
    ledger_records.insert(
        0,
        {
            **provenance,
            "schema_version": SCHEMA_VERSION,
            "record_type": "forecast_tournament_run",
            "status": "complete",
            "fold": None,
            "fit_split": "train",
            "selection_split": "validation",
            "report_split": "development_test",
            "test_is_report_only": True,
            "runtime_seconds": runtime,
            "config": {
                "folds": list(folds),
                "horizons": list(horizons_tuple),
                "policy_horizon": int(policy_horizon),
                "feature_sets": list(feature_set_tuple),
                "candidates": list(candidate_tuple),
                "execution_delay_bars": int(execution_delay),
                "execution_delay_sensitivity_lags": list(EXECUTION_DELAY_SENSITIVITY_LAGS),
                "max_fit_rows": int(max_fit_rows),
                "hist_max_iter": int(hist_max_iter),
            },
        },
    )
    with ledger_path.open("w", encoding="utf-8") as handle:
        for record in ledger_records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result_path = output_path / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path = output_path / "report.md"
    report_path.write_text(build_report_markdown(result), encoding="utf-8")
    result["result_path"] = str(result_path)
    result["report_path"] = str(report_path)
    # Persist the final paths in the payload after writing the report; the
    # ledger already contains its canonical path and remains machine-readable.
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def build_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact report with the tournament gate and its caveats."""
    lines = [
        "# Plan011 Development Forecast/Timing Tournament",
        "",
        "## Scope and selection contract",
        "",
        f"- development folds only: `{', '.join(str(value) for value in payload['folds'])}`",
        f"- horizons: `{payload['horizons']}`; policy horizon: `{payload['policy_horizon']}`",
        f"- commit: `{payload['commit_hash']}`",
        f"- config SHA-256: `{payload['config_sha256']}`",
        f"- data SHA-256: `{payload['data_sha256']}`",
        f"- source SHA-256: `{payload['source_sha256']}`",
        f"- fixed operational execution delay: `{payload.get('execution_delay_bars', DEFAULT_EXECUTION_DELAY_BARS)}` bar; sensitivity lags: `{payload.get('execution_delay_sensitivity_lags', list(EXECUTION_DELAY_SENSITIVITY_LAGS))}`",
        "",
        "Candidates are causal trend+volatility, Ridge direct multi-horizon, and HistGradientBoosting direct multi-horizon forecasts. "
        "Targets use only `t+1..t+h`; Oracle positions are not an input. Hyperparameters, no-trade threshold, overlay magnitude, hysteresis, and minimum hold are selected on validation only under the fixed operational execution delay. "
        "The development test interval is report-only; its aggregate is an explicitly labeled tournament screen for the next wave.",
        "",
        "## Successive-halving gate",
        "",
        f"Pass requires positive IC on at least half of the three folds (minimum two), positive median timing increment and median AlphaEx, and dynamic timing must beat the validation-selected constant, fixed execution-lag 16 sensitivity, and temporal-destruction circular-shift null 64 by more than the pre-registered `{TIMING_SUPERIORITY_MARGIN_PT:.2f}pt` margin on the aggregate median and at least `{TIMING_MIN_WIN_RATE:.0%}` of comparable folds. Lag1, shift1, and shift16 are reported robustness diagnostics rather than hard gate criteria. It must also satisfy the median MaxDDDelta/turnover tradeoff (no more than 0.05pt worse than the constant; turnover at most 6.5).",
        "",
        "| feature set | candidate | folds | median IC | IC+ folds | median AlphaEx | Δconstant (median; wins/rate) | Δlag1 | Δlag16 | Δnull1 | Δnull16 | Δnull64 | status |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for row in payload.get("gate", []):
        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):+.4f}"

        def comparison_fmt(name: str) -> str:
            comparison = row.get("timing_superiority", {}).get(name, {})
            median = comparison.get("median_alpha_difference_pt")
            wins = comparison.get("win_folds")
            rate = comparison.get("win_rate")
            if median is None or wins is None or rate is None:
                return "N/A"
            return f"{float(median):+.3f}; {int(wins)}/{float(rate):.0%}"

        lines.append(
            f"| {row['feature_set']} | {row['candidate']} | {row['folds']} | {fmt(row['test_median_spearman_ic'])} | "
            f"{row['test_positive_ic_folds']} | {fmt(row['test_median_alpha_excess_pt'])}pt | "
            f"{comparison_fmt('constant')} | {comparison_fmt('lag1')} | {comparison_fmt('lag16')} | "
            f"{comparison_fmt('null_shift1')} | {comparison_fmt('null_shift16')} | {comparison_fmt('null_shift64')} | {row['status']} |"
        )
    lines.extend([
        "",
        "Next-wave candidates (development tournament screen only): `"
        + ", ".join(
            f"{row['feature_set']}/{row['candidate']}" for row in payload.get("next_wave_candidates", [])
        )
        + "`.",
        "",
        "## Selected per-fold paths",
        "",
        "| fold | feature set | candidate | selected policy | val IC(h=16) | test IC(h=16) | constant AlphaEx | timing increment | dynamic AlphaEx | MaxDDDelta | turnover |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload.get("rows", []):
        val_ic = _metric_at_horizon(row["validation_return_metrics"], payload["policy_horizon"], "spearman_ic")
        test_ic = _metric_at_horizon(row["test_return_metrics"], payload["policy_horizon"], "spearman_ic")
        policy = row["policy"]
        dynamic = row["test_economics"]["dynamic"]
        constant = row["test_economics"]["constant"]
        fmt = lambda value: "N/A" if value is None else f"{float(value):+.4f}"
        lines.append(
            f"| {row['fold']} | {row['feature_set']} | {row['candidate']} | "
            f"thr={policy['threshold']:.5g}, mag={policy['overlay_magnitude']:.3g}, hyst={policy['hysteresis']:.2g}, hold={policy['min_hold']}, delay={policy['execution_delay']} | "
            f"{fmt(val_ic)} | {fmt(test_ic)} | {constant['alpha_excess_pt']:+.3f}pt | "
            f"{row['test_economics']['timing_increment_alpha_excess_pt']:+.3f}pt | "
            f"{dynamic['alpha_excess_pt']:+.3f}pt | {dynamic['maxdd_delta_pt']:+.3f}pt | {dynamic['turnover']:.4f} |"
        )
    lines.extend([
        "",
        "## Forecast diagnostics",
        "",
        "The ledger contains every selected validation-fit and report-only development-test metric for return/risk targets at each requested horizon. This compact table shows the principal horizons and risk diagnostics; risk sign accuracy is intentionally N/A because realized risk is one-sided.",
        "",
        "| fold | feature set | candidate | split | return IC h4 | return IC h16 | return IC h64 | return sign h16 | risk MAE h16 | risk RMSE h16 | risk IC h16 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload.get("rows", []):
        for split_name, return_records, risk_records in (
            (
                "validation",
                row.get("validation_return_metrics", []),
                row.get("validation_risk_metrics", []),
            ),
            (
                "development_test",
                row.get("test_return_metrics", []),
                row.get("test_risk_metrics", []),
            ),
        ):
            fmt = lambda value: "N/A" if value is None else f"{float(value):+.4f}"
            lines.append(
                f"| {row['fold']} | {row['feature_set']} | {row['candidate']} | {split_name} | "
                f"{fmt(_metric_at_horizon(return_records, 4, 'spearman_ic'))} | "
                f"{fmt(_metric_at_horizon(return_records, 16, 'spearman_ic'))} | "
                f"{fmt(_metric_at_horizon(return_records, 64, 'spearman_ic'))} | "
                f"{fmt(_metric_at_horizon(return_records, 16, 'sign_accuracy'))} | "
                f"{fmt(_metric_at_horizon(risk_records, 16, 'mae'))} | "
                f"{fmt(_metric_at_horizon(risk_records, 16, 'rmse'))} | "
                f"{fmt(_metric_at_horizon(risk_records, 16, 'spearman_ic'))} |"
            )
    lines.extend([
        "",
        "## Feature coverage",
        "",
        "Full17 rows carry separate finite/missing/zero/nonzero counts for funding_rate, basis, basis_mom, and basis_abs. The cache has no availability mask, so zero-versus-missing remains an explicit quality flag and is not silently treated as observed signal.",
        "",
        "| fold | feature set | split | rows | external nonzero/missing counts (funding, basis, basis_mom, basis_abs) | quality/status |",
        "|---:|---|---|---:|---|---|",
    ])
    for row in payload.get("feature_quality", []):
        external_counts = ", ".join(
            f"{name}={row['external'][name]['nonzero_count']}/{row['external'][name]['missing_count']}"
            for name in EXTERNAL_FEATURES
        )
        lines.append(
            f"| {row['fold']} | {row['feature_set']} | {row['split']} | {row['rows']} | "
            f"{external_counts} | "
            f"{row['status']}; {', '.join(row['quality_flags'])} |"
        )
    lines.extend([
        "",
        "## Timing/null contract",
        "",
        "Each selected path reports the validation-selected constant exposure, dynamic path, execution lag 1 and 16, and deterministic circular-shift nulls 1, 16, and 64. The fixed operational delay is the main result; lagged executions are sensitivity diagnostics. AlphaEx, MaxDDDelta, SharpeDelta, and turnover come from the shared Backtest/action_stats contract with B&H position 1.0 and the configured costs.",
        "",
        "Unavailable or undefined forecast metrics are recorded as `N/A` with a reason; no test result selects a hyperparameter or policy. A passing row is a candidate for a later wave, not evidence that the full Plan011 model should be retrained.",
        "",
        f"Machine-readable ledger: `{payload['ledger_path']}`",
    ])
    return "\n".join(lines) + "\n"
