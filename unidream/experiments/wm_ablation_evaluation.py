"""Targets, calendar splits and paired evaluation for the WM input ablation.

All timestamps are UTC *bar opens*.  A decision at row t is taken after bar
t-1 closes: its input ends at t-1 and its label uses bars t through t+h-1.
The downside target is the fractional decline from close[t-1] to the lowest
future CLOSE, floored at zero.  It is neither a high/low target nor a
peak-to-trough drawdown.  QLIKE is applied to the sum of squared future
15-minute log returns (realized variance), never its square root.

Bootstrap observations are paired losses averaged over seeds at each date,
not independent seed replications.  Moving blocks comprise UTC days and
cannot cross a fold boundary or a discontinuity in the 15-minute grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score


BAR_INTERVAL = pd.Timedelta(minutes=15)
DEFAULT_HORIZONS = (4, 16, 64)
VARIANCE_FLOOR = 1.0e-12
CRASH_THRESHOLD = 0.012


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must contain unique increasing UTC timestamps")
    if len(index) > 1 and not np.all(np.diff(index.asi8) == BAR_INTERVAL.value):
        raise ValueError("keep the complete 15-minute grid; do not compress missing bars")
    if len(index) and np.any(index.asi8 % BAR_INTERVAL.value):
        raise ValueError("bar opens must align to UTC 15-minute boundaries")
    return index


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def fold_idx(self) -> int:
        return self.fold

    def bounds(self, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        if split == "validation":
            split = "val"
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val/validation or test")
        return getattr(self, f"{split}_start"), getattr(self, f"{split}_end")


def make_walk_forward_folds(
    first_test: str | pd.Timestamp = "2023-01-01",
    last_test: str | pd.Timestamp = "2026-07-01",
) -> list[WalkForwardFold]:
    """Two calendar years train, three months val and three months test.

    Both arguments are inclusive quarter starts.  Every interval is half-open.
    Earlier feature context is permitted; outcome labels are purged separately.
    """
    first, last = _utc(first_test), _utc(last_test)
    for value in (first, last):
        if value.month not in (1, 4, 7, 10) or value != value.normalize() or value.day != 1:
            raise ValueError("test dates must be UTC calendar-quarter starts")
    if first > last:
        raise ValueError("first_test must not follow last_test")
    result: list[WalkForwardFold] = []
    cursor = first
    while cursor <= last:
        val_start = cursor - pd.DateOffset(months=3)
        result.append(WalkForwardFold(
            fold=len(result),
            train_start=val_start - pd.DateOffset(years=2),
            train_end=val_start,
            val_start=val_start,
            val_end=cursor,
            test_start=cursor,
            test_end=cursor + pd.DateOffset(months=3),
        ))
        cursor += pd.DateOffset(months=3)
    return result


@dataclass(frozen=True)
class TargetBundle:
    index: pd.DatetimeIndex
    horizons: tuple[int, ...]
    returns: np.ndarray
    variance: np.ndarray
    downside: np.ndarray
    crash: np.ndarray
    mask: np.ndarray
    label_end: np.ndarray
    baseline_variance: np.ndarray

    def horizon_index(self, horizon: int) -> int:
        return self.horizons.index(horizon)


def build_future_targets(
    index: pd.DatetimeIndex,
    close: Sequence[float] | np.ndarray,
    observed: Sequence[bool] | np.ndarray | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    crash_threshold: float = CRASH_THRESHOLD,
) -> TargetBundle:
    """Build labels on an uncompressed grid, with NaN labels for missing paths.

    ``label_end[t,h]`` is the close time of the final outcome bar, t+h*15min.
    The simple variance baseline is the preceding 16 log returns' squared sum,
    scaled by h/16 for each horizon; its own missing history remains NaN.
    """
    index = _grid(index)
    close = np.asarray(close, dtype=np.float64)
    if close.shape != (len(index),):
        raise ValueError("close must be one value per full-grid row")
    supplied_horizons = tuple(horizons)
    if any(int(h) != h for h in supplied_horizons):
        raise ValueError("horizons must be whole bar counts")
    horizons = tuple(int(h) for h in supplied_horizons)
    if not horizons or min(horizons) <= 0 or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique positive bar counts")
    if not 0 < crash_threshold < 1:
        raise ValueError("crash_threshold must be a fixed fractional decline")
    valid = np.isfinite(close) & (close > 0)
    if observed is not None:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != close.shape:
            raise ValueError("observed must be one boolean per full-grid row")
        valid &= observed
    safe = np.where(valid, close, np.nan)
    log_return = np.full(len(index), np.nan)
    log_return[1:] = np.diff(np.log(safe))
    squares = np.square(log_return)
    squared_sum = np.r_[0.0, np.cumsum(np.nan_to_num(squares))]
    valid_sum = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    shape = (len(index), len(horizons))
    ret, variance, downside, crash, baseline = [np.full(shape, np.nan) for _ in range(5)]
    mask = np.zeros(shape, dtype=bool)
    label_end = np.empty(shape, dtype="datetime64[ns]")
    close_series = pd.Series(safe)
    trailing16 = pd.Series(squares).rolling(16, min_periods=16).sum().shift(1).to_numpy()
    rows = np.arange(len(index))
    for column, horizon in enumerate(horizons):
        label_end[:, column] = (index.asi8 + horizon * BAR_INTERVAL.value).astype("datetime64[ns]")
        start = rows[(rows > 0) & (rows + horizon <= len(index))]
        if len(start):
            # Include both the reference close and every future close.
            complete = valid_sum[start + horizon] - valid_sum[start - 1] == horizon + 1
            selected = start[complete]
            future_min = close_series.rolling(horizon, min_periods=horizon).min().shift(1 - horizon).to_numpy()
            mask[selected, column] = True
            ret[selected, column] = np.log(safe[selected + horizon - 1] / safe[selected - 1])
            variance[selected, column] = squared_sum[selected + horizon] - squared_sum[selected]
            downside[selected, column] = np.maximum(0.0, 1.0 - future_min[selected] / safe[selected - 1])
            crash[selected, column] = downside[selected, column] >= crash_threshold
        baseline[:, column] = trailing16 * horizon / 16.0
    return TargetBundle(index, horizons, ret, variance, downside, crash, mask, label_end, baseline)


def split_mask(
    index: pd.DatetimeIndex,
    targets: TargetBundle,
    fold: WalkForwardFold,
    split: str,
    horizon: int | None = None,
    context_valid: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Keep decisions in the split and labels ending STRICTLY before its end.

    Without a horizon, all configured labels must be valid: this common mask
    purges 16-hour labels at train/validation/test boundaries consistently.
    Feature windows may begin in an earlier split, but cannot use later bars.
    """
    index = _grid(index)
    if not index.equals(targets.index):
        raise ValueError("target and split grids must match")
    start, end = fold.bounds(split)
    eligible = np.asarray((index >= start) & (index < end))
    if horizon is None:
        eligible &= np.all(targets.mask & (targets.label_end < end.tz_localize(None).to_datetime64()), axis=1)
    else:
        column = targets.horizon_index(horizon)
        eligible &= targets.mask[:, column] & (targets.label_end[:, column] < end.tz_localize(None).to_datetime64())
    if context_valid is not None:
        context_valid = np.asarray(context_valid, dtype=bool)
        if context_valid.shape != eligible.shape:
            raise ValueError("context_valid must be one boolean per grid row")
        eligible &= context_valid
    return eligible


def variance_qlike(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Per-row QLIKE with a predeclared 1e-12 floor on both variances."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if np.any(actual < 0):
        raise ValueError("realized variance cannot be negative")
    ratio = np.maximum(actual, VARIANCE_FLOOR) / np.maximum(predicted, VARIANCE_FLOOR)
    return ratio - np.log(ratio) - 1.0


_PREDICTION_COLUMNS = (
    "actual_return", "pred_return", "actual_variance", "pred_variance",
    "actual_downside", "pred_downside", "actual_crash", "pred_crash",
)


def _balanced(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    return float(balanced_accuracy_score(actual, predicted)) if len(np.unique(actual)) == 2 else None


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    crash_probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score one group/fold/seed/horizon on the same finite rows for all heads.

    OOS R2 uses the zero-return denominator, sum(actual_return**2).  A zero
    denominator, constant Spearman input, or missing class produces None.
    Thresholds are fixed before test evaluation and never tuned here.
    """
    if not 0 <= crash_probability_threshold <= 1:
        raise ValueError("crash threshold must be a probability")
    values = frame.loc[:, list(_PREDICTION_COLUMNS)].to_numpy(dtype=float)
    keep = np.all(np.isfinite(values), axis=1)
    values = values[keep]
    result: dict[str, Any] = {"n": int(len(values)), "n_nonfinite_excluded": int((~keep).sum())}
    if not len(values):
        return result
    actual, predicted, actual_var, pred_var, actual_dd, pred_dd, actual_crash, pred_crash = values.T
    if not np.all(np.isin(actual_crash, (0, 1))) or np.any((pred_crash < 0) | (pred_crash > 1)):
        raise ValueError("crash labels must be binary and predictions probabilities")
    error, dd_error = predicted - actual, pred_dd - actual_dd
    denominator = float(np.sum(actual ** 2))
    sse = float(np.sum(error ** 2))
    result.update({
        "return_rmse": float(np.sqrt(np.mean(error ** 2))),
        "return_mae": float(np.mean(np.abs(error))),
        "return_oos_r2": 1.0 - sse / denominator if denominator > 0 else None,
        "return_spearman_ic": float(spearmanr(actual, predicted).statistic)
        if np.ptp(actual) > 0 and np.ptp(predicted) > 0 else None,
        "return_direction_accuracy": float(np.mean((actual >= 0) == (predicted >= 0))),
        "return_direction_balanced_accuracy": _balanced(actual >= 0, predicted >= 0),
        "variance_qlike": float(np.mean(variance_qlike(actual_var, pred_var))),
        "variance_rmse": float(np.sqrt(np.mean((pred_var - actual_var) ** 2))),
        "variance_floor_count": int(np.sum(pred_var < VARIANCE_FLOOR)),
        "volatility_rmse": float(np.sqrt(np.mean((np.sqrt(np.maximum(pred_var, VARIANCE_FLOOR)) - np.sqrt(actual_var)) ** 2))),
        "downside_mae": float(np.mean(np.abs(dd_error))),
        "downside_rmse": float(np.sqrt(np.mean(dd_error ** 2))),
        "crash_brier": float(np.mean((pred_crash - actual_crash) ** 2)),
        "crash_pr_auc": float(average_precision_score(actual_crash, pred_crash)) if np.any(actual_crash == 1) else None,
        "crash_balanced_accuracy": _balanced(actual_crash, pred_crash >= crash_probability_threshold),
        "crash_probability_threshold": float(crash_probability_threshold),
        "crash_prevalence": float(np.mean(actual_crash)),
    })
    return result


def _paired_daily_losses(
    frame: pd.DataFrame, group_a: str, group_b: str, horizon: int,
) -> tuple[pd.DataFrame, int, int]:
    keys = ["timestamp", "fold", "seed"]
    cols = [*keys, "actual_return", "pred_return", "actual_variance", "pred_variance"]
    selected = frame.loc[frame["horizon"] == horizon]
    left = selected.loc[selected["group"] == group_a, cols].copy()
    right = selected.loc[selected["group"] == group_b, cols].copy()
    for part in (left, right):
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        if part.duplicated(keys).any():
            raise ValueError("duplicate prediction keys cannot be paired")
    paired = left.merge(right, on=keys, suffixes=("_a", "_b"), how="outer", indicator=True, validate="one_to_one")
    if not len(paired) or not (paired["_merge"] == "both").all():
        raise ValueError("both conditions must have identical timestamp/fold/seed coverage")
    numeric = [f"{name}_{side}" for name in ("actual_return", "pred_return", "actual_variance", "pred_variance") for side in ("a", "b")]
    if not np.isfinite(paired[numeric].to_numpy(dtype=float)).all():
        raise ValueError("bootstrap requires finite common predictions")
    for name in ("actual_return", "actual_variance"):
        if not np.array_equal(paired[f"{name}_a"], paired[f"{name}_b"]):
            raise ValueError("paired conditions must share identical outcomes")
        # Every seed must forecast the same outcome at a given timestamp.
        if (paired.groupby(["fold", "timestamp"])[f"{name}_a"].nunique() != 1).any():
            raise ValueError("seeds must share identical outcomes")
    n_seeds = int(paired["seed"].nunique())
    counts = paired.groupby(["fold", "timestamp"])["seed"].nunique()
    if not (counts == n_seeds).all():
        raise ValueError("each timestamp must contain the same complete seed set")
    paired["mse_a"] = (paired["actual_return_a"] - paired["pred_return_a"]) ** 2
    paired["mse_b"] = (paired["actual_return_b"] - paired["pred_return_b"]) ** 2
    paired["zero_mse"] = paired["actual_return_a"] ** 2
    paired["qlike_a"] = variance_qlike(paired["actual_variance_a"].to_numpy(), paired["pred_variance_a"].to_numpy())
    paired["qlike_b"] = variance_qlike(paired["actual_variance_b"].to_numpy(), paired["pred_variance_b"].to_numpy())
    losses = ["mse_a", "mse_b", "zero_mse", "qlike_a", "qlike_b"]
    averaged = paired.groupby(["fold", "timestamp"], as_index=False)[losses].mean().sort_values(["timestamp", "fold"])
    if averaged["timestamp"].duplicated().any():
        raise ValueError("test folds must not overlap")
    delta = averaged["timestamp"].diff()
    breaks = (delta != BAR_INTERVAL) | (averaged["fold"] != averaged["fold"].shift())
    averaged["segment"] = breaks.cumsum().to_numpy()
    averaged["date"] = averaged["timestamp"].dt.floor("D")
    averaged["count"] = 1.0
    daily = averaged.groupby(["segment", "date"], as_index=False)[[*losses, "count"]].sum()
    return daily, len(averaged), n_seeds


def _statistics(sums: np.ndarray) -> dict[str, np.ndarray]:
    mse_a, mse_b, zero_sse, qlike_a, qlike_b, count = np.moveaxis(sums, -1, 0)
    rmse_a, rmse_b = np.sqrt(mse_a / count), np.sqrt(mse_b / count)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "return_rmse_improvement": rmse_a - rmse_b,
            "return_rmse_improvement_pct": 100.0 * (rmse_a - rmse_b) / rmse_a,
            "return_oos_r2_difference": (mse_a - mse_b) / zero_sse,
            "return_mse_improvement": (mse_a - mse_b) / count,
            "variance_qlike_improvement": (qlike_a - qlike_b) / count,
        }


def paired_block_bootstrap(
    frame: pd.DataFrame,
    group_a: str = "A",
    group_b: str = "B",
    horizon: int = 16,
    block_days: Sequence[int] = (1, 7, 28),
    replicates: int = 2000,
    random_seed: int = 20261109,
) -> list[dict[str, Any]]:
    """Pooled paired loss inference using non-circular moving UTC-day blocks.

    Resampling is stratified by each contiguous fold segment.  Segments shorter
    than the requested block retain their length (and therefore report limited
    resampling diversity).  Seed losses are averaged *before* day aggregation.
    RMSE is derived from pooled resampled squared errors, not averaged daily
    RMSE.  P values are two-sided, centered-bootstrap approximate tests; the
    seven-day result is the predeclared primary inference and 1/28 are sensitivity.
    """
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    if not block_days or any(int(days) != days or days < 1 for days in block_days):
        raise ValueError("block_days must contain positive whole days")
    daily, n_timestamps, n_seeds = _paired_daily_losses(frame, group_a, group_b, horizon)
    loss_cols = ["mse_a", "mse_b", "zero_mse", "qlike_a", "qlike_b", "count"]
    segments = [part[loss_cols].to_numpy(dtype=float) for _, part in daily.groupby("segment", sort=True)]
    total = daily[loss_cols].sum().to_numpy(dtype=float)
    estimate = _statistics(total)
    output: list[dict[str, Any]] = []
    for days in block_days:
        days = int(days)
        rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), days]))
        resampled = np.zeros((replicates, len(loss_cols)), dtype=float)
        n_short = 0
        n_resampleable = 0
        resampleable_timestamps = 0.0
        for segment in segments:
            length = len(segment)
            block = min(days, length)
            n_short += length < days
            n_resampleable += length > block
            if length > block:
                resampleable_timestamps += float(segment[:, -1].sum())
            n_blocks = (length + block - 1) // block
            starts = rng.integers(0, length - block + 1, size=(replicates, n_blocks))
            indices = (starts[:, :, None] + np.arange(block)).reshape(replicates, -1)[:, :length]
            resampled += segment[indices].sum(axis=1)
        bootstrap = _statistics(resampled)
        metrics: dict[str, dict[str, float | None]] = {}
        for name, value in estimate.items():
            value = float(value)
            distribution = bootstrap[name]
            finite = distribution[np.isfinite(distribution)]
            if not np.isfinite(value) or not len(finite) or not n_resampleable:
                # Entire-segment replay has no sampling diversity, so its
                # point estimate does not support a confidence interval.
                metrics[name] = {"estimate": None, "ci_low": None, "ci_high": None}
                if np.isfinite(value):
                    metrics[name]["estimate"] = value
                if name in ("return_mse_improvement", "variance_qlike_improvement"):
                    metrics[name]["p_value"] = None
                continue
            low, high = np.quantile(finite, [0.025, 0.975])
            metrics[name] = {"estimate": value, "ci_low": float(low), "ci_high": float(high)}
            if name in ("return_mse_improvement", "variance_qlike_improvement"):
                centered = finite - value
                p = (1 + np.sum(np.abs(centered) >= abs(value))) / (len(finite) + 1)
                metrics[name]["p_value"] = float(p)
        output.append({
            "comparison": f"{group_b}-{group_a}", "horizon": int(horizon),
            "block_days": days, "replicates": int(replicates),
            "n_paired_timestamps": int(n_timestamps), "n_seeds": n_seeds,
            "n_segments": len(segments), "n_segments_shorter_than_block": int(n_short),
            "n_resampleable_segments": int(n_resampleable),
            "resampleable_timestamp_fraction": resampleable_timestamps / float(total[-1]),
            "inference_available": bool(n_resampleable),
            "primary_inference": days == 7, "metrics": metrics,
        })
    return output


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm adjusted P values; missing tests retain their place in the family."""
    valid: list[tuple[str, float]] = []
    result: dict[str, float | None] = {name: None for name in p_values}
    for name, value in p_values.items():
        if value is None or not np.isfinite(value):
            continue
        if not 0 <= value <= 1:
            raise ValueError("p values must lie in [0,1]")
        valid.append((name, float(value)))
    valid.sort(key=lambda pair: pair[1])
    running = 0.0
    for rank, (name, value) in enumerate(valid):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        result[name] = running
    return result
