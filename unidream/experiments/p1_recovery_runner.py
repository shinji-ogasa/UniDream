"""Execution-free core for the pre-registered P1 recovery run.

The P1 experiment is intentionally staged.  This module implements only the
deterministic synthetic data contract, chronological fit masks, and the four
forecast primitives needed by a later runner.  It does not replay actions,
bootstrap metrics, load S3 data, write result artifacts, or execute the outer
report operation.

The preregistration validator remains in :mod:`p1_recovery_prereg`.  The
runner calls that validator when loading a manifest, but does not duplicate or
weaken its immutable field checks.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .p1_recovery_prereg import (
    DEFAULT_MANIFEST_PATH,
    P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
    REGISTERED_MANIFEST_SHA256,
    load_fixed_manifest,
)


class P1RunnerError(ValueError):
    """Raised when runner input cannot satisfy the fixed P1 contract."""


class P1OuterReportBlocked(RuntimeError):
    """Raised if code attempts to execute the report-only outer operation."""


SYNTHETIC_RAW_ROWS = 120_512
SYNTHETIC_BURN_IN = 512
SYNTHETIC_ROWS = 120_000
FEATURE_DIMENSION = 17
CONTEXT_BARS = 64
PURGE_BARS = 16
MIN_HISTORY_ROWS = 16_384
BAR_NS = 15 * 60 * 1_000_000_000
SYNTHETIC_START = np.datetime64("2018-01-01T00:00:00", "ns")
FORECAST_HORIZONS = (1, 4, 8, 16)
OOF_ORIGINS = (20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000)
OOF_BATCH_SPAN = 10_000
MODEL_IDS = (
    "zero_return",
    "persistence_last_observed",
    "ridge",
    "logistic",
)
MODEL_TASKS: Mapping[str, Literal["continuous", "binary"]] = MappingProxyType(
    {
        "zero_return": "continuous",
        "persistence_last_observed": "continuous",
        "ridge": "continuous",
        "logistic": "binary",
    }
)
_MODEL_ALLOWED_TASKS: Mapping[str, tuple[Literal["continuous", "binary"], ...]] = MappingProxyType(
    {
        "zero_return": ("continuous", "binary"),
        "persistence_last_observed": ("continuous", "binary"),
        "ridge": ("continuous",),
        "logistic": ("binary",),
    }
)
PROBABILITY_CLIP_EPS = 1e-6
S3_SIGNAL_FEATURE = "close_ret"
S3_INJECTION_BETA = 0.0005
S3_PREFIX_ROWS_MIN = 256
V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT = (
    "unidream.experiments.runtime.validate_v4_runtime_inputs"
)


def validate_15m_timestamps(
    timestamps: Any,
    *,
    n_rows: int | None = None,
    label: str = "timestamps",
) -> np.ndarray:
    """Validate a strictly ordered datetime64 index for the 15-minute body.

    The full-grid body is allowed to contain a timestamp gap.  Window
    builders use :func:`timestamp_edge_mask` to invalidate every context or
    target window which crosses that gap while retaining the original rows.
    This keeps timestamp evidence and sidecar masks aligned instead of
    silently compressing a missing bar.
    """

    try:
        array = np.asarray(timestamps)
    except (TypeError, ValueError) as exc:
        raise P1RunnerError(f"{label} must be a one-dimensional datetime64 array") from exc
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.datetime64):
        raise P1RunnerError(f"{label} must be a one-dimensional datetime64 array")
    if n_rows is not None and len(array) != n_rows:
        raise P1RunnerError(f"{label} is not row-aligned")
    try:
        normalized = np.asarray(array, dtype=np.dtype("datetime64[ns]") ).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1RunnerError(f"{label} cannot be represented as datetime64[ns]") from exc
    if np.isnat(normalized).any():
        raise P1RunnerError(f"{label} must not contain NaT")
    if len(normalized) > 1:
        ticks = normalized.astype(np.int64)
        if np.any(np.diff(ticks) <= 0):
            raise P1RunnerError(f"{label} must be strictly increasing and unique")
    return _read_only(normalized, dtype=np.dtype("datetime64[ns]"))


def timestamp_edge_mask(timestamps: Any) -> np.ndarray:
    """Return exact 15-minute adjacency for each ``t -> t+1`` edge."""

    normalized = validate_15m_timestamps(timestamps)
    if len(normalized) < 2:
        return _read_only(np.zeros(0, dtype=np.bool_), dtype=np.bool_)
    ticks = normalized.astype(np.int64)
    return _read_only(np.diff(ticks) == BAR_NS, dtype=np.bool_)


def _strict_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise P1RunnerError("seed must be an integer; bool and string coercion are forbidden")
    return int(seed)


def _strict_horizon(horizon: Any) -> int:
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)):
        raise P1RunnerError("horizon must be an integer")
    horizon_int = int(horizon)
    if horizon_int not in FORECAST_HORIZONS:
        raise P1RunnerError(
            f"horizon must be one of {FORECAST_HORIZONS}, got {horizon_int}"
        )
    return horizon_int


def _strict_origin(origin: Any) -> int:
    if isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)):
        raise P1RunnerError("origin must be an integer")
    origin_int = int(origin)
    if origin_int < 0:
        raise P1RunnerError("origin must be non-negative")
    return origin_int


def _read_only(array: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _mapping_proxy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _is_deeply_frozen(value: Any) -> bool:
    if isinstance(value, MappingProxyType):
        return all(_is_deeply_frozen(item) for item in value.values())
    if isinstance(value, tuple):
        return all(_is_deeply_frozen(item) for item in value)
    if isinstance(value, (dict, list, set, bytearray)):
        return False
    return True


def _require_authenticated_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require the exact fixed, deeply frozen manifest at the runner boundary."""
    if not isinstance(manifest, MappingProxyType) or not _is_deeply_frozen(manifest):
        raise P1RunnerError(
            "P1 runner requires the deeply frozen manifest from load_fixed_manifest"
        )
    if manifest.get("manifest_sha256") != REGISTERED_MANIFEST_SHA256:
        raise P1RunnerError("P1 runner manifest digest is not the registered digest")
    if manifest.get("results_observed") is not False:
        raise P1RunnerError("P1 runner requires results_observed=false")
    common = manifest.get("common")
    if not isinstance(common, Mapping):
        raise P1RunnerError("P1 runner manifest common contract is missing")
    v4_load = common.get("v4_load_contract")
    if not isinstance(v4_load, Mapping):
        raise P1RunnerError("P1 runner v4 load contract is missing")
    if v4_load.get("runtime_validation_entrypoint") != P1_V4_RUNTIME_VALIDATION_ENTRYPOINT:
        raise P1RunnerError("P1 runner authenticated v4 entrypoint is not pinned")
    if v4_load.get("runtime_body_validator_entrypoint") != V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT:
        raise P1RunnerError("P1 runner v4 body validator entrypoint is not pinned")
    runner_contract = common.get("runner_contract")
    if not isinstance(runner_contract, Mapping):
        raise P1RunnerError("P1 runner contract is missing")
    if runner_contract.get("v4_runtime_validation_entrypoint") != P1_V4_RUNTIME_VALIDATION_ENTRYPOINT:
        raise P1RunnerError("P1 runner contract bypasses authenticated v4 validation")
    if runner_contract.get("v4_runtime_body_validator_entrypoint") != V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT:
        raise P1RunnerError("P1 runner contract body validator is not pinned")
    return manifest


@dataclass(frozen=True)
class ManifestEcho:
    """Immutable provenance fields every future result must echo."""

    manifest_id: str
    manifest_sha256: str
    base_revision: str

    def as_dict(self) -> dict[str, str]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "base_revision": self.base_revision,
        }


@dataclass(frozen=True)
class OuterReportSpec:
    """The outer operation's fixed metadata, without an execution method."""

    origin: int
    fit_prefix_range: tuple[int, int]
    prediction_range: tuple[int, int]
    refit_origins: tuple[int, ...]
    report_only: bool
    selection_allowed: bool
    threshold_revision_allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "fit_prefix_range": list(self.fit_prefix_range),
            "prediction_range": list(self.prediction_range),
            "refit_origins": list(self.refit_origins),
            "role": "report_only" if self.report_only else "invalid",
            "selection_allowed": self.selection_allowed,
            "threshold_revision_allowed": self.threshold_revision_allowed,
        }


@dataclass(frozen=True)
class RunnerPlan:
    """Fixed synthetic OOF schedule loaded from the preregistered manifest."""

    manifest: Mapping[str, Any]
    manifest_echo: ManifestEcho
    horizons: tuple[int, ...]
    origins: tuple[int, ...]
    purge_bars: int
    context_bars: int
    outer_report: OuterReportSpec
    outer_report_only: bool = True


def _manifest_echo(manifest: Mapping[str, Any]) -> ManifestEcho:
    _require_authenticated_manifest(manifest)
    values: dict[str, str] = {}
    for field in ("manifest_id", "manifest_sha256", "base_revision"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            raise P1RunnerError(f"manifest.{field} must be a non-empty string")
        values[field] = value
    return ManifestEcho(**values)


def load_runner_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> Mapping[str, Any]:
    """Load and validate the immutable preregistration for the runner."""

    try:
        loaded = load_fixed_manifest(path)
    except (OSError, TypeError, ValueError) as exc:
        raise P1RunnerError("P1 runner could not load the fixed manifest") from exc
    return _require_authenticated_manifest(loaded)


def manifest_echo(manifest: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the fixed manifest provenance fields for a result envelope."""

    if manifest is not None:
        raise P1RunnerError(
            "arbitrary manifest injection is forbidden; load the fixed manifest by path"
        )
    loaded = load_runner_manifest()
    return _manifest_echo(loaded).as_dict()


def build_runner_plan(manifest: Mapping[str, Any] | None = None) -> RunnerPlan:
    """Build the fixed OOF plan; no data, model, or result operation runs."""

    if manifest is not None:
        raise P1RunnerError(
            "arbitrary manifest injection is forbidden; build the plan from the fixed manifest"
        )
    loaded = load_runner_manifest()
    echo = _manifest_echo(loaded)
    try:
        common = loaded["common"]
        oof = common["oof"]
        synthetic = loaded["synthetic_contract"]
        outer = synthetic["outer_report_operation"]
    except (KeyError, TypeError) as exc:
        raise P1RunnerError("validated P1 manifest is missing runner fields") from exc
    if tuple(common.get("forecast_horizons", ())) != FORECAST_HORIZONS:
        raise P1RunnerError("manifest forecast horizons do not match runner constants")
    if common.get("sequence_context_bars") != CONTEXT_BARS:
        raise P1RunnerError("manifest context length does not match runner constants")
    schedule = oof.get("origin_schedule")
    if not isinstance(schedule, Mapping) or tuple(schedule.get("origins", ())) != OOF_ORIGINS:
        raise P1RunnerError("manifest OOF origin schedule does not match runner constants")
    if oof.get("purge_bars") != PURGE_BARS:
        raise P1RunnerError("manifest purge length does not match runner constants")
    if not isinstance(outer, Mapping):
        raise P1RunnerError("manifest outer report operation must be an object")
    outer_spec = OuterReportSpec(
        origin=int(outer["origin"]),
        fit_prefix_range=tuple(int(v) for v in outer["fit_prefix_range"]),
        prediction_range=tuple(int(v) for v in outer["prediction_range"]),
        refit_origins=tuple(int(v) for v in outer["refit_origins"]),
        report_only=outer.get("role") == "report_only",
        selection_allowed=bool(outer.get("selection_allowed")),
        threshold_revision_allowed=bool(outer.get("threshold_revision_allowed")),
    )
    if (
        not outer_spec.report_only
        or outer_spec.selection_allowed
        or outer_spec.threshold_revision_allowed
        or outer_spec.refit_origins
    ):
        raise P1RunnerError("outer operation must remain report-only with no refits")
    return RunnerPlan(
        manifest=loaded,
        manifest_echo=echo,
        horizons=FORECAST_HORIZONS,
        origins=OOF_ORIGINS,
        purge_bars=PURGE_BARS,
        context_bars=CONTEXT_BARS,
        outer_report=outer_spec,
    )


def outer_report_spec(manifest: Mapping[str, Any] | None = None) -> OuterReportSpec:
    """Return the outer metadata; intentionally never fit or score it."""

    return build_runner_plan(manifest).outer_report


def execute_outer_report(*_: Any, **__: Any) -> None:
    """Keep accidental outer execution impossible in this implementation unit."""

    raise P1OuterReportBlocked(
        "P1 outer report is metadata-only in the runner core; execution is staged later"
    )


@dataclass(frozen=True)
class SyntheticBase:
    """Per-seed random arrays and paired availability masks shared by every beta."""

    seed: int
    z_raw: np.ndarray
    xi: np.ndarray
    noise_features: np.ndarray
    epsilon: np.ndarray
    gap_starts: Mapping[str, tuple[int, ...]]
    availability: Mapping[str, np.ndarray]

    @property
    def spot_bar_observed(self) -> np.ndarray:
        return self.availability["spot_bar_observed"]

    @property
    def funding_rate_available(self) -> np.ndarray:
        return self.availability["funding_rate_available"]

    @property
    def mark_close_available(self) -> np.ndarray:
        return self.availability["mark_close_available"]


def _make_availability(seed: int) -> tuple[Mapping[str, tuple[int, ...]], Mapping[str, np.ndarray]]:
    starts_by_source: dict[str, tuple[int, ...]] = {}
    masks: dict[str, np.ndarray] = {}
    source_offsets = {
        "spot_bar_observed": 11,
        "funding_rate_available": 23,
        "mark_close_available": 37,
    }
    # The registered range is deliberately expressed in output coordinates.
    # Its lower bound leaves the first post-burn-in context region untouched.
    possible_start_count = (SYNTHETIC_ROWS - 2) - SYNTHETIC_BURN_IN
    for source, offset in source_offsets.items():
        gap_rng = np.random.default_rng(seed + 50_000 + offset)
        relative_starts = gap_rng.choice(
            possible_start_count,
            size=40,
            replace=False,
            shuffle=True,
        )
        starts = np.asarray(relative_starts, dtype=np.int64) + SYNTHETIC_BURN_IN
        starts_by_source[source] = tuple(int(v) for v in starts)
        mask = np.ones(SYNTHETIC_ROWS, dtype=np.bool_)
        for start in starts:
            mask[start : start + 2] = False
        masks[source] = _read_only(mask, dtype=np.bool_)
    return MappingProxyType(starts_by_source), MappingProxyType(masks)


@lru_cache(maxsize=32)
def _generate_synthetic_base_cached(seed: int) -> SyntheticBase:
    # Keep this draw order in lockstep with synthetic_contract.draw_order.
    rng = np.random.default_rng(seed + 100)
    z0 = float(rng.standard_normal())
    xi = np.asarray(rng.standard_normal(SYNTHETIC_RAW_ROWS - 1), dtype=np.float64)
    z_raw = np.empty(SYNTHETIC_RAW_ROWS, dtype=np.float64)
    z_raw[0] = z0
    innovation_std = float(np.sqrt(1.0 - 0.95**2))
    # A recurrence is used rather than a second random draw, preserving the
    # exact one-scalar-plus-xi stream specified by the manifest.
    for index in range(1, SYNTHETIC_RAW_ROWS):
        z_raw[index] = 0.95 * z_raw[index - 1] + innovation_std * xi[index - 1]
    noise_features = np.asarray(
        rng.standard_normal((SYNTHETIC_RAW_ROWS, FEATURE_DIMENSION - 1)),
        dtype=np.float64,
        order="C",
    )
    epsilon = np.asarray(rng.standard_normal(SYNTHETIC_RAW_ROWS), dtype=np.float64)
    gap_starts, availability = _make_availability(seed)
    return SyntheticBase(
        seed=seed,
        z_raw=_read_only(z_raw, dtype=np.float64),
        xi=_read_only(xi, dtype=np.float64),
        noise_features=_read_only(noise_features, dtype=np.float64),
        epsilon=_read_only(epsilon, dtype=np.float64),
        gap_starts=gap_starts,
        availability=availability,
    )


def generate_synthetic_base(seed: int) -> SyntheticBase:
    """Generate (and cache) one exact shared random base for a seed."""

    return _generate_synthetic_base_cached(_strict_seed(seed))


@dataclass(frozen=True)
class SyntheticDataset:
    """Full-grid synthetic features, returns, targets, and distinct masks."""

    seed: int
    beta: float
    timestamps: np.ndarray
    base: SyntheticBase
    features: np.ndarray
    returns: np.ndarray
    targets: np.ndarray
    target_end: np.ndarray
    target_mask: np.ndarray
    binary_labels: np.ndarray
    context_mask: np.ndarray

    @property
    def context_eligible(self) -> np.ndarray:
        return self.context_mask

    @property
    def target_complete(self) -> np.ndarray:
        return self.target_mask

    @property
    def availability(self) -> Mapping[str, np.ndarray]:
        return self.base.availability

    @property
    def spot_bar_observed(self) -> np.ndarray:
        return self.base.spot_bar_observed

    @property
    def funding_rate_available(self) -> np.ndarray:
        return self.base.funding_rate_available

    @property
    def mark_close_available(self) -> np.ndarray:
        return self.base.mark_close_available

    def with_returns(self, returns: Any) -> "SyntheticDataset":
        """Rebuild only targets/masks for a deliberate causal perturbation test."""

        target_values, target_mask, target_end = build_target_arrays(
            returns,
            self.spot_bar_observed,
            horizons=FORECAST_HORIZONS,
            timestamps=self.timestamps,
        )
        labels = binary_labels_from_targets(target_values)
        return SyntheticDataset(
            seed=self.seed,
            beta=self.beta,
            timestamps=self.timestamps,
            base=self.base,
            features=self.features,
            returns=_read_only(np.asarray(returns, dtype=np.float64)),
            targets=target_values,
            target_end=target_end,
            target_mask=target_mask,
            binary_labels=labels,
            context_mask=self.context_mask,
        )


def build_target_arrays(
    returns: Any,
    spot_bar_observed: Any,
    *,
    horizons: Sequence[int] = FORECAST_HORIZONS,
    timestamps: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full-grid ``t+h+1``-exclusive targets with a Spot-only mask.

    A finite zero return is a valid target value.  A false Spot mask or a
    non-finite return in any target bar invalidates the complete horizon, while
    funding/mark availability is intentionally not consulted here.  When
    timestamps are supplied, every edge ``t -> t+1`` through
    ``t+h-1 -> t+h`` must be exactly 15 minutes; the following edge
    ``t+h -> t+h+1`` is deliberately not inspected.
    """

    returns_array = np.asarray(returns)
    if returns_array.ndim != 1 or not np.issubdtype(returns_array.dtype, np.number):
        raise P1RunnerError("returns must be a one-dimensional numeric array")
    returns_array = np.asarray(returns_array, dtype=np.float64)
    spot = np.asarray(spot_bar_observed)
    if spot.dtype != np.dtype(np.bool_) or spot.ndim != 1 or len(spot) != len(returns_array):
        raise P1RunnerError("spot_bar_observed must be a strict bool array aligned to returns")
    horizon_values = tuple(_strict_horizon(h) for h in horizons)
    n_rows = len(returns_array)
    timestamp_array = (
        None
        if timestamps is None
        else validate_15m_timestamps(timestamps, n_rows=n_rows)
    )
    target_values = np.full((n_rows, len(horizon_values)), np.nan, dtype=np.float64)
    target_mask = np.zeros((n_rows, len(horizon_values)), dtype=np.bool_)
    target_end = np.empty((n_rows, len(horizon_values)), dtype=np.int64)
    finite_returns = np.isfinite(returns_array)
    safe_returns = np.where(finite_returns, returns_array, 0.0)
    return_prefix = np.concatenate(([0.0], np.cumsum(safe_returns, dtype=np.float64)))
    bad_returns_prefix = np.concatenate(
        ([0], np.cumsum((~finite_returns).astype(np.int64), dtype=np.int64))
    )
    bad_spot_prefix = np.concatenate(
        ([0], np.cumsum((~spot).astype(np.int64), dtype=np.int64))
    )
    bad_edge_prefix = None
    if timestamp_array is not None:
        edges = timestamp_edge_mask(timestamp_array)
        bad_edge_prefix = np.concatenate(
            ([0], np.cumsum((~edges).astype(np.int64), dtype=np.int64))
        )
    rows = np.arange(n_rows, dtype=np.int64)
    for column, horizon in enumerate(horizon_values):
        ends = rows + horizon + 1
        target_end[:, column] = ends
        inside = ends <= n_rows
        if not np.any(inside):
            continue
        starts = rows[inside] + 1
        end_values = ends[inside]
        valid = (
            bad_returns_prefix[end_values] - bad_returns_prefix[starts] == 0
        ) & (bad_spot_prefix[end_values] - bad_spot_prefix[starts] == 0)
        if bad_edge_prefix is not None:
            # ``end_values - 1`` is the exclusive prefix endpoint for the
            # h edges beginning at decision row t.  Using ``end_values``
            # would incorrectly require t+h -> t+h+1 as well.
            valid &= (
                bad_edge_prefix[end_values - 1]
                - bad_edge_prefix[rows[inside]]
                == 0
            )
        values = return_prefix[end_values] - return_prefix[starts]
        valid_rows = rows[inside][valid]
        target_values[valid_rows, column] = values[valid]
        target_mask[valid_rows, column] = True
    return (
        _read_only(target_values, dtype=np.float64),
        _read_only(target_mask, dtype=np.bool_),
        _read_only(target_end, dtype=np.int64),
    )


def binary_labels_from_targets(targets: Any) -> np.ndarray:
    """Map positive finite targets to class 1; zero maps exactly to class 0.

    Incomplete/non-finite targets receive ``-1`` so they cannot be silently
    treated as negative labels by a caller that forgot the target mask.
    """

    array = np.asarray(targets)
    if not np.issubdtype(array.dtype, np.number):
        raise P1RunnerError("targets must be numeric for binary labels")
    finite = np.isfinite(array)
    labels = np.full(array.shape, -1, dtype=np.int8)
    labels[finite] = (np.asarray(array, dtype=np.float64)[finite] > 0.0).astype(np.int8)
    return _read_only(labels, dtype=np.int8)


def build_context_mask(
    features: Any,
    availability: Mapping[str, Any],
    *,
    context_bars: int = CONTEXT_BARS,
    timestamps: Any | None = None,
) -> np.ndarray:
    """Return the all-three-source, finite, timestamp-complete context mask.

    For decision row ``t`` the current-inclusive window is ``[t-63, t]``.
    Every one of its 63 inter-row edges must be exactly 15 minutes when a
    timestamp array is supplied.  A later edge is not part of this context
    contract.
    """

    feature_array = validate_current_row_features(features)
    if isinstance(context_bars, (bool, np.bool_)) or not isinstance(
        context_bars, (int, np.integer)
    ):
        raise P1RunnerError("context_bars must be an integer")
    context_bars_int = int(context_bars)
    if context_bars_int <= 0:
        raise P1RunnerError("context_bars must be positive")
    required_sources = (
        "spot_bar_observed",
        "funding_rate_available",
        "mark_close_available",
    )
    try:
        source_arrays = [np.asarray(availability[name]) for name in required_sources]
    except (KeyError, TypeError) as exc:
        raise P1RunnerError("all three availability masks are required") from exc
    n_rows = len(feature_array)
    timestamp_array = (
        None
        if timestamps is None
        else validate_15m_timestamps(timestamps, n_rows=n_rows)
    )
    for name, source in zip(required_sources, source_arrays):
        if source.dtype != np.dtype(np.bool_) or source.ndim != 1 or len(source) != n_rows:
            raise P1RunnerError(f"{name} must be a strict bool array aligned to features")
    finite_rows = np.isfinite(feature_array).all(axis=1)
    all_three = source_arrays[0] & source_arrays[1] & source_arrays[2] & finite_rows
    bad_prefix = np.concatenate(([0], np.cumsum((~all_three).astype(np.int64))))
    bad_edge_prefix = None
    if timestamp_array is not None:
        edges = timestamp_edge_mask(timestamp_array)
        bad_edge_prefix = np.concatenate(
            ([0], np.cumsum((~edges).astype(np.int64), dtype=np.int64))
        )
    rows = np.arange(n_rows, dtype=np.int64)
    eligible = np.zeros(n_rows, dtype=np.bool_)
    enough_history = rows >= context_bars_int - 1
    current = rows[enough_history]
    starts = current - context_bars_int + 1
    eligible[current] = bad_prefix[current + 1] - bad_prefix[starts] == 0
    if bad_edge_prefix is not None:
        eligible[current] &= bad_edge_prefix[current] - bad_edge_prefix[starts] == 0
    return _read_only(eligible, dtype=np.bool_)


def validate_current_row_features(features: Any) -> np.ndarray:
    """Validate the canonical current-row 17-feature model input.

    A 64x17 sequence or its 1088-column flattening is rejected explicitly;
    context belongs only in eligibility masks under this contract.
    """

    array = np.asarray(features)
    if array.ndim == 3 and array.shape[1:] == (CONTEXT_BARS, FEATURE_DIMENSION):
        raise P1RunnerError(
            "64-bar context must not be passed to the model; use the current row's 17 features"
        )
    if array.ndim != 2 or array.shape[1] != FEATURE_DIMENSION:
        raise P1RunnerError(
            f"model input must have shape (n_rows, {FEATURE_DIMENSION}); "
            "64x17 flattening/augmentation is forbidden"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise P1RunnerError("model input must contain numeric features")
    return np.asarray(array, dtype=np.float64)


# Short aliases used by callers that name the boundary in different ways.
validate_model_input = validate_current_row_features
reject_flattened_context = validate_current_row_features


def build_synthetic_dataset(seed: int, beta: float = 0.0) -> SyntheticDataset:
    """Materialize one beta from a shared seed base using the fixed DGP."""

    seed_int = _strict_seed(seed)
    try:
        beta_float = float(beta)
    except (TypeError, ValueError) as exc:
        raise P1RunnerError("beta must be a finite numeric value") from exc
    if not np.isfinite(beta_float):
        raise P1RunnerError("beta must be finite")
    base = generate_synthetic_base(seed_int)
    features_raw = np.empty((SYNTHETIC_RAW_ROWS, FEATURE_DIMENSION), dtype=np.float64)
    features_raw[:, 0] = base.z_raw
    features_raw[:, 1:] = base.noise_features
    returns_raw = np.empty(SYNTHETIC_RAW_ROWS, dtype=np.float64)
    returns_raw[0] = 0.001 * base.epsilon[0]
    returns_raw[1:] = beta_float * base.z_raw[:-1] + 0.001 * base.epsilon[1:]
    timestamps = _read_only(
        SYNTHETIC_START
        + np.arange(SYNTHETIC_ROWS, dtype=np.int64) * np.timedelta64(15, "m"),
        dtype=np.dtype("datetime64[ns]"),
    )
    features = _read_only(features_raw[SYNTHETIC_BURN_IN:], dtype=np.float64)
    returns = _read_only(returns_raw[SYNTHETIC_BURN_IN:], dtype=np.float64)
    targets, target_mask, target_end = build_target_arrays(
        returns,
        base.spot_bar_observed,
        horizons=FORECAST_HORIZONS,
        timestamps=timestamps,
    )
    labels = binary_labels_from_targets(targets)
    context_mask = build_context_mask(
        features,
        base.availability,
        timestamps=timestamps,
    )
    return SyntheticDataset(
        seed=seed_int,
        beta=beta_float,
        timestamps=timestamps,
        base=base,
        features=features,
        returns=returns,
        targets=targets,
        target_end=target_end,
        target_mask=target_mask,
        binary_labels=labels,
        context_mask=context_mask,
    )


# Friendly aliases for future callers and tests.
generate_synthetic_dataset = build_synthetic_dataset
make_synthetic_dataset = build_synthetic_dataset


def _horizon_column(dataset: SyntheticDataset, horizon: int) -> int:
    _strict_horizon(horizon)
    try:
        return FORECAST_HORIZONS.index(int(horizon))
    except ValueError as exc:  # pragma: no cover - guarded by _strict_horizon
        raise P1RunnerError(f"unsupported horizon: {horizon}") from exc


def _ensure_dataset(dataset: SyntheticDataset) -> SyntheticDataset:
    if not isinstance(dataset, SyntheticDataset):
        raise P1RunnerError("runner model APIs require a SyntheticDataset")
    features = validate_current_row_features(dataset.features)
    n_rows = len(features)
    timestamps = validate_15m_timestamps(dataset.timestamps, n_rows=n_rows)
    returns = np.asarray(dataset.returns)
    if returns.shape != (n_rows,) or not np.issubdtype(returns.dtype, np.number):
        raise P1RunnerError("dataset returns are not row-aligned")
    targets = np.asarray(dataset.targets)
    if (
        targets.shape != (n_rows, len(FORECAST_HORIZONS))
        or not np.issubdtype(targets.dtype, np.number)
    ):
        raise P1RunnerError("dataset targets do not cover all fixed horizons")
    target_end = np.asarray(dataset.target_end)
    target_mask = np.asarray(dataset.target_mask)
    if (
        target_end.shape != targets.shape
        or target_mask.shape != targets.shape
        or target_end.dtype != np.dtype(np.int64)
    ):
        raise P1RunnerError("dataset target arrays are not shape-aligned")
    context_mask = np.asarray(dataset.context_mask)
    if context_mask.shape != (n_rows,):
        raise P1RunnerError("dataset context mask is not row-aligned")
    for name, mask in (
        ("target_mask", dataset.target_mask),
        ("context_mask", dataset.context_mask),
    ):
        if np.asarray(mask).dtype != np.dtype(np.bool_):
            raise P1RunnerError(f"dataset {name} must have strict bool dtype")
    if targets.dtype != np.dtype(np.float64):
        raise P1RunnerError("dataset targets must use float64 values")
    expected_end = (
        np.arange(n_rows, dtype=np.int64)[:, None]
        + np.asarray(FORECAST_HORIZONS, dtype=np.int64)[None, :]
        + 1
    )
    if not np.array_equal(target_end, expected_end):
        raise P1RunnerError("dataset target_end must equal row + horizon + 1 for every row")

    try:
        availability = dataset.availability
        required_sources = (
            "spot_bar_observed",
            "funding_rate_available",
            "mark_close_available",
        )
        for name in required_sources:
            source = np.asarray(availability[name])
            if source.dtype != np.dtype(np.bool_) or source.shape != (n_rows,):
                raise P1RunnerError(
                    f"dataset availability {name} must be a strict bool row vector"
                )
    except (KeyError, TypeError, AttributeError) as exc:
        raise P1RunnerError("dataset availability masks are incomplete") from exc

    expected_targets, expected_target_mask, expected_target_end = build_target_arrays(
        returns,
        availability["spot_bar_observed"],
        horizons=FORECAST_HORIZONS,
        timestamps=timestamps,
    )
    if not np.array_equal(target_end, expected_target_end):
        raise P1RunnerError("dataset target_end does not match the fixed horizon contract")
    if not np.array_equal(target_mask, expected_target_mask):
        raise P1RunnerError("dataset target_mask does not match returns, Spot, and timestamps")
    if not np.array_equal(targets, expected_targets, equal_nan=True):
        raise P1RunnerError("dataset targets do not equal the canonical return sums")

    binary_labels = np.asarray(dataset.binary_labels)
    if binary_labels.shape != targets.shape or binary_labels.dtype != np.dtype(np.int8):
        raise P1RunnerError("dataset binary labels are not shape/dtype aligned")
    expected_labels = binary_labels_from_targets(expected_targets)
    if not np.array_equal(binary_labels, expected_labels):
        raise P1RunnerError("dataset binary labels do not match canonical targets")

    expected_context_mask = build_context_mask(
        features,
        availability,
        context_bars=CONTEXT_BARS,
        timestamps=timestamps,
    )
    if not np.array_equal(context_mask, expected_context_mask):
        raise P1RunnerError("dataset context_mask does not match availability, features, and timestamps")
    return dataset


def train_mask_for_origin(
    dataset: SyntheticDataset,
    origin: int,
    horizon: int,
    *,
    purge_bars: int = PURGE_BARS,
) -> np.ndarray:
    """Return the exact context/label/purge/row admissibility mask."""

    data = _ensure_dataset(dataset)
    origin_int = _strict_origin(origin)
    horizon_int = _strict_horizon(horizon)
    if isinstance(purge_bars, (bool, np.bool_)) or not isinstance(
        purge_bars, (int, np.integer)
    ):
        raise P1RunnerError("purge_bars must be an integer")
    purge_int = int(purge_bars)
    if purge_int < 0:
        raise P1RunnerError("purge_bars must be non-negative")
    if origin_int > len(data.features):
        raise P1RunnerError("origin exceeds dataset rows")
    column = _horizon_column(data, horizon_int)
    rows = np.arange(len(data.features), dtype=np.int64)
    mask = (
        data.context_mask
        & data.target_mask[:, column]
        & (data.target_end[:, column] <= origin_int - purge_int)
        & (rows < origin_int)
    )
    return _read_only(mask, dtype=np.bool_)


def build_train_mask(*args: Any, **kwargs: Any) -> np.ndarray:
    """Compatibility alias for :func:`train_mask_for_origin`."""

    return train_mask_for_origin(*args, **kwargs)


def prediction_mask_for_range(
    dataset: SyntheticDataset,
    horizon: int,
    *,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Return full-grid context-and-target eligibility inside a support range."""

    data = _ensure_dataset(dataset)
    horizon_int = _strict_horizon(horizon)
    start_int = _strict_origin(start)
    end_int = len(data.features) if end is None else _strict_origin(end)
    if end_int < start_int or end_int > len(data.features):
        raise P1RunnerError("prediction support range is outside the dataset")
    column = _horizon_column(data, horizon_int)
    rows = np.arange(len(data.features), dtype=np.int64)
    mask = (
        data.context_mask
        & data.target_mask[:, column]
        & (data.target_end[:, column] <= end_int)
        & (rows >= start_int)
        & (rows < end_int)
    )
    return _read_only(mask, dtype=np.bool_)


@dataclass(frozen=True)
class ModelFit:
    """One origin x horizon fit and its full-grid forecast view."""

    model_id: str
    task: Literal["continuous", "binary"]
    horizon: int
    origin: int
    train_mask: np.ndarray
    eligible_mask: np.ndarray
    prediction_mask: np.ndarray
    predictions: np.ndarray
    status: str
    reason: str | None
    scaler: StandardScaler | None
    estimator: Ridge | LogisticRegression | None

    @property
    def is_na(self) -> bool:
        return self.status == "N/A"


def clip_probabilities(probabilities: Any, *, eps: float = PROBABILITY_CLIP_EPS) -> np.ndarray:
    """Clip finite probability values to the preregistered open interval."""

    if not np.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise P1RunnerError("probability clip eps must be finite and in (0, 0.5)")
    array = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(array).all():
        raise P1RunnerError("probabilities must be finite before clipping")
    return _read_only(np.clip(array, eps, 1.0 - eps), dtype=np.float64)


def _prediction_mask(
    dataset: SyntheticDataset,
    horizon: int,
    origin: int,
    prediction_range: tuple[int, int] | None,
) -> np.ndarray:
    if prediction_range is None:
        start, end = origin, len(dataset.features)
    else:
        if len(prediction_range) != 2:
            raise P1RunnerError("prediction_range must be (start, end)")
        start, end = prediction_range
    return prediction_mask_for_range(dataset, horizon, start=start, end=end)


def _na_model_fit(
    *,
    model_id: str,
    task: Literal["continuous", "binary"],
    horizon: int,
    origin: int,
    train_mask: np.ndarray,
    eligible_mask: np.ndarray,
    predictions: np.ndarray,
    reason: str,
) -> ModelFit:
    """Construct a read-only N/A fit without fitting or scoring a model."""

    return ModelFit(
        model_id=model_id,
        task=task,
        horizon=horizon,
        origin=origin,
        train_mask=train_mask,
        eligible_mask=eligible_mask,
        prediction_mask=_read_only(np.zeros(len(predictions), dtype=np.bool_)),
        predictions=_read_only(predictions, dtype=np.float64),
        status="N/A",
        reason=reason,
        scaler=None,
        estimator=None,
    )


def fit_model_at_origin(
    dataset: SyntheticDataset,
    model_id: str,
    origin: int,
    horizon: int,
    *,
    task: Literal["continuous", "binary"] | None = None,
    prediction_range: tuple[int, int] | None = None,
) -> ModelFit:
    """Fit one fixed model using only its origin/horizon admissible prefix.

    Ridge and Logistic each get a fresh ``StandardScaler`` fit only on their
    own origin x horizon training rows.  Baselines do not scale.  A Logistic
    prefix with one observed class is returned as N/A without repair.
    """

    data = _ensure_dataset(dataset)
    if model_id not in MODEL_IDS:
        raise P1RunnerError(f"unknown model_id: {model_id}")
    expected_task = MODEL_TASKS[model_id]
    resolved_task = expected_task if task is None else task
    if resolved_task not in {"continuous", "binary"}:
        raise P1RunnerError("task must be 'continuous' or 'binary'")
    if resolved_task not in _MODEL_ALLOWED_TASKS[model_id]:
        allowed = ", ".join(_MODEL_ALLOWED_TASKS[model_id])
        raise P1RunnerError(f"model {model_id} only supports task(s): {allowed}")
    horizon_int = _strict_horizon(horizon)
    origin_int = _strict_origin(origin)
    if origin_int > len(data.features):
        raise P1RunnerError("origin exceeds dataset rows")
    train_mask = train_mask_for_origin(data, origin_int, horizon_int)
    eligible_mask = prediction_mask_for_range(
        data,
        horizon_int,
        start=origin_int,
        end=len(data.features),
    )
    if prediction_range is not None:
        eligible_mask = _prediction_mask(data, horizon_int, origin_int, prediction_range)
    prediction_mask = np.zeros(len(data.features), dtype=np.bool_)
    predictions = np.full(len(data.features), np.nan, dtype=np.float64)
    column = _horizon_column(data, horizon_int)
    scaler: StandardScaler | None = None
    estimator: Ridge | LogisticRegression | None = None
    reason: str | None = None

    if np.count_nonzero(train_mask) < MIN_HISTORY_ROWS:
        return _na_model_fit(
            model_id=model_id,
            task=resolved_task,
            horizon=horizon_int,
            origin=origin_int,
            train_mask=train_mask,
            eligible_mask=eligible_mask,
            predictions=predictions,
            reason=f"fewer than {MIN_HISTORY_ROWS} admissible training rows",
        )

    if model_id == "zero_return":
        predictions[eligible_mask] = 0.0 if resolved_task == "continuous" else 0.5
        prediction_mask = eligible_mask.copy()
    elif model_id == "persistence_last_observed":
        if resolved_task == "continuous":
            predictions[eligible_mask] = horizon_int * data.returns[eligible_mask]
        else:
            positive = data.returns[eligible_mask] > 0.0
            predictions[eligible_mask] = np.where(
                positive,
                1.0 - PROBABILITY_CLIP_EPS,
                PROBABILITY_CLIP_EPS,
            )
        prediction_mask = eligible_mask.copy()
    elif model_id == "ridge":
        X_train = data.features[train_mask]
        y_train = data.targets[train_mask, column]
        if len(X_train) == 0:
            return _na_model_fit(
                model_id=model_id,
                task=resolved_task,
                horizon=horizon_int,
                origin=origin_int,
                train_mask=train_mask,
                eligible_mask=eligible_mask,
                predictions=predictions,
                reason="no admissible training rows",
            )
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_train)
        estimator = Ridge(
            alpha=1.0,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-12,
            max_iter=10_000,
            random_state=None,
        )
        estimator.fit(X_scaled, y_train)
        if np.any(eligible_mask):
            predictions[eligible_mask] = estimator.predict(
                scaler.transform(data.features[eligible_mask])
            )
            prediction_mask = eligible_mask.copy()
    elif model_id == "logistic":
        labels = data.binary_labels[train_mask, column]
        # The target mask guarantees finite labels; keep this explicit so a
        # malformed custom dataset cannot turn an invalid sentinel into class 0.
        if np.any(labels < 0):
            raise P1RunnerError("logistic training labels include an invalid target")
        classes = np.unique(labels)
        if len(classes) < 2:
            reason = "one observed class in admissible prefix"
            return _na_model_fit(
                model_id=model_id,
                task=resolved_task,
                horizon=horizon_int,
                origin=origin_int,
                train_mask=train_mask,
                eligible_mask=eligible_mask,
                predictions=predictions,
                reason=reason,
            )
        X_train = data.features[train_mask]
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_train)
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            tol=1e-10,
            max_iter=1000,
            class_weight=None,
            random_state=0,
        )
        estimator.fit(X_scaled, labels)
        if np.any(eligible_mask):
            class_one_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
            raw_probability = estimator.predict_proba(
                scaler.transform(data.features[eligible_mask])
            )[:, class_one_index]
            predictions[eligible_mask] = clip_probabilities(raw_probability)
            prediction_mask = eligible_mask.copy()

    return ModelFit(
        model_id=model_id,
        task=resolved_task,
        horizon=horizon_int,
        origin=origin_int,
        train_mask=train_mask,
        eligible_mask=eligible_mask,
        prediction_mask=_read_only(prediction_mask, dtype=np.bool_),
        predictions=_read_only(predictions, dtype=np.float64),
        status="ok",
        reason=reason,
        scaler=scaler,
        estimator=estimator,
    )


def fit_origin_model(*args: Any, **kwargs: Any) -> ModelFit:
    """Alias for :func:`fit_model_at_origin`."""

    return fit_model_at_origin(*args, **kwargs)


@dataclass(frozen=True)
class OOFRun:
    """Chronological OOF forecasts with no action or result-artifact layer."""

    plan: RunnerPlan
    dataset: SyntheticDataset
    fits: Mapping[tuple[int, int, str], ModelFit]
    outer_report_only: bool = True
    outer_test_executed: bool = False

    def get(self, origin: int, horizon: int, model_id: str) -> ModelFit:
        return self.fits[(int(origin), int(horizon), model_id)]


def run_synthetic_oof(
    dataset: SyntheticDataset | int,
    beta: float = 0.0,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    outer_report_only: bool = True,
) -> OOFRun:
    """Run only fixed synthetic OOF development/validation fits.

    ``outer_report_only`` is an API guard, not a request to execute the outer
    operation.  Passing ``False`` is rejected so a caller cannot accidentally
    convert this staged unit into outer selection/tuning.
    """

    if outer_report_only is not True:
        raise P1OuterReportBlocked("outer operation must remain report-only")
    data = (
        build_synthetic_dataset(dataset, beta=beta)
        if isinstance(dataset, (int, np.integer)) and not isinstance(dataset, (bool, np.bool_))
        else dataset
    )
    data = _ensure_dataset(data)
    plan = build_runner_plan()
    requested_models = tuple(model_ids)
    if not requested_models:
        raise P1RunnerError("at least one model is required")
    if any(model not in MODEL_IDS for model in requested_models):
        raise P1RunnerError("model_ids contain an unknown fixed model")
    fits: dict[tuple[int, int, str], ModelFit] = {}
    for origin in plan.origins:
        batch_end = min(origin + OOF_BATCH_SPAN, len(data.features))
        for horizon in plan.horizons:
            for model_id in requested_models:
                fits[(origin, horizon, model_id)] = fit_model_at_origin(
                    data,
                    model_id,
                    origin,
                    horizon,
                    prediction_range=(origin, batch_end),
                )
    return OOFRun(
        plan=plan,
        dataset=data,
        fits=MappingProxyType(fits),
        outer_report_only=True,
        outer_test_executed=False,
    )


run_oof = run_synthetic_oof


__all__ = [
    "CONTEXT_BARS",
    "FEATURE_DIMENSION",
    "FORECAST_HORIZONS",
    "MODEL_IDS",
    "MODEL_TASKS",
    "OOF_BATCH_SPAN",
    "OOF_ORIGINS",
    "PROBABILITY_CLIP_EPS",
    "PURGE_BARS",
    "P1OuterReportBlocked",
    "P1RunnerError",
    "ManifestEcho",
    "ModelFit",
    "OOFRun",
    "OuterReportSpec",
    "RunnerPlan",
    "SyntheticBase",
    "SyntheticDataset",
    "binary_labels_from_targets",
    "build_context_mask",
    "build_runner_plan",
    "build_synthetic_dataset",
    "build_target_arrays",
    "build_train_mask",
    "clip_probabilities",
    "execute_outer_report",
    "fit_model_at_origin",
    "fit_origin_model",
    "generate_synthetic_base",
    "generate_synthetic_dataset",
    "load_runner_manifest",
    "make_synthetic_dataset",
    "manifest_echo",
    "outer_report_spec",
    "reject_flattened_context",
    "run_oof",
    "run_synthetic_oof",
    "train_mask_for_origin",
    "validate_current_row_features",
    "validate_model_input",
]
