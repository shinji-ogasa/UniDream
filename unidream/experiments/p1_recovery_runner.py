"""Execution-free core for the pre-registered P1 recovery run.

The P1 experiment is intentionally staged.  This module implements only the
deterministic synthetic data contract, chronological fit masks, and the four
forecast primitives needed by a later runner.  Its S3 entrypoint delegates
body access to the authenticated v4 runtime wrapper and only constructs the
preregistered injection/control arrays.  It does not replay actions,
bootstrap metrics, write result artifacts, or execute the outer report
operation.

The preregistration validator remains in :mod:`p1_recovery_prereg`.  The
runner calls that validator when loading a manifest, but does not duplicate or
weaken its immutable field checks.

The public ``fit_model_at_origin`` path is the production fit boundary and is
fail-closed on the fixed purge, timestamp, range, finite-output, and coverage
contracts.  ``run_synthetic_oof`` is deliberately fixture/diagnostic-only: it
accepts one synthetic seed/beta and optional model subsets for deterministic
contract tests.  A future registered scenario runner must bind the complete
manifest model/task grid, seed set, beta/level, and per-key coverage before
any result or gate operation is added.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..data.cache_v4 import MODEL_FEATURE_COLUMNS, REQUIRED_AVAILABILITY_COLUMNS

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
SYNTHETIC_OOF_SCOPE = "fixture_diagnostic_only"
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
S3_BODY_ROWS = 173_111
S3_TRAIN_START = 52_492
S3_VALIDATION_ORIGIN = 104_528
S3_VALIDATION_END = 139_568
S3_OUTER_END = 173_111
V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT = (
    "unidream.experiments.runtime.validate_v4_runtime_inputs"
)
_S3_BODY_SEAL = object()


def _s3_hash_plain(value: Any) -> Any:
    """Normalize immutable provenance into canonical JSON hash input."""

    if isinstance(value, Mapping):
        return {str(key): _s3_hash_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_s3_hash_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _s3_hash_plain(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise P1RunnerError("S3 provenance hash input contains a non-finite scalar")
        return value
    raise P1RunnerError("S3 provenance hash input contains an unsupported value")


def _s3_array_sha256(value: Any, *, label: str) -> str:
    """Hash dtype, shape, and exact C-order bytes for one S3 source array."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise P1RunnerError(f"S3 {label} cannot use object dtype")
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


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


def _build_target_arrays(
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


def build_target_arrays(
    returns: Any,
    spot_bar_observed: Any,
    *,
    horizons: Sequence[int] = FORECAST_HORIZONS,
    timestamps: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build production targets; an explicit timestamp grid is mandatory."""

    if timestamps is None:
        raise P1RunnerError(
            "production target arrays require timestamps; use "
            "build_target_arrays_fixture for timestamp-free contract fixtures"
        )
    return _build_target_arrays(
        returns,
        spot_bar_observed,
        horizons=horizons,
        timestamps=timestamps,
    )


def build_target_arrays_fixture(
    returns: Any,
    spot_bar_observed: Any,
    *,
    horizons: Sequence[int] = FORECAST_HORIZONS,
    timestamps: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a timestamp-free target fixture for isolated unit tests only."""

    return _build_target_arrays(
        returns,
        spot_bar_observed,
        horizons=horizons,
        timestamps=timestamps,
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


def _build_context_mask(
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


def build_context_mask(
    features: Any,
    availability: Mapping[str, Any],
    *,
    context_bars: int = CONTEXT_BARS,
    timestamps: Any | None = None,
) -> np.ndarray:
    """Build a production context mask from an explicit timestamp grid."""

    if timestamps is None:
        raise P1RunnerError(
            "production context masks require timestamps; use "
            "build_context_mask_fixture for timestamp-free contract fixtures"
        )
    return _build_context_mask(
        features,
        availability,
        context_bars=context_bars,
        timestamps=timestamps,
    )


def build_context_mask_fixture(
    features: Any,
    availability: Mapping[str, Any],
    *,
    context_bars: int = CONTEXT_BARS,
    timestamps: Any | None = None,
) -> np.ndarray:
    """Build a timestamp-free context fixture for isolated unit tests only."""

    return _build_context_mask(
        features,
        availability,
        context_bars=context_bars,
        timestamps=timestamps,
    )


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


@dataclass(frozen=True)
class S3InjectionControl:
    """Authenticated v4 S3 body plus the preregistered S3 arm pair.

    The body is materialized only from the result of
    ``validate_p1_v4_runtime_inputs``.  ``injection_mask[t]`` identifies a
    decision row whose next observed return was modified; the control and
    injected arrays retain the original full-grid row order and timestamps.
    ``runtime`` contains only immutable provenance echoes, paths, disposition,
    and digest scalars; it never retains the wrapper's mutable pandas body.
    """

    runtime: Mapping[str, Any]
    timestamps: np.ndarray
    features: np.ndarray
    returns_v4: np.ndarray
    control_returns: np.ndarray
    injected_returns: np.ndarray
    injection_mask: np.ndarray
    z_scores: np.ndarray
    context_mask: np.ndarray
    availability: Mapping[str, np.ndarray]
    body_sha256: str = ""
    _production_seal: object | None = field(default=None, repr=False, compare=False)

    @property
    def returns(self) -> np.ndarray:
        """The unmodified v4 return body."""

        return self.returns_v4

    @property
    def returns_control(self) -> np.ndarray:
        """Alias for the zero-injection control arm."""

        return self.control_returns

    @property
    def returns_injected(self) -> np.ndarray:
        """Alias for the observable-prefix injection arm."""

        return self.injected_returns

    @property
    def context_eligible(self) -> np.ndarray:
        """The timestamp- and availability-complete context mask."""

        return self.context_mask


@dataclass(frozen=True)
class S3ArmDataset:
    """One full-grid injected/control target body under the fixed v4 features."""

    seed: int
    arm: Literal["injected", "zero_injection_control"]
    beta: float
    timestamps: np.ndarray
    source: S3InjectionControl
    features: np.ndarray
    returns: np.ndarray
    targets: np.ndarray
    target_end: np.ndarray
    target_mask: np.ndarray
    binary_labels: np.ndarray
    context_mask: np.ndarray
    availability: Mapping[str, np.ndarray]
    source_body_sha256: str = ""
    _production_seal: object | None = field(default=None, repr=False, compare=False)

    @property
    def context_eligible(self) -> np.ndarray:
        return self.context_mask

    @property
    def target_complete(self) -> np.ndarray:
        return self.target_mask


def _s3_body_sha256(body: S3InjectionControl) -> str:
    """Bind the authenticated wrapper echoes to every materialized S3 array."""

    if not isinstance(body, S3InjectionControl):
        raise P1RunnerError("S3 source hash requires an S3InjectionControl")
    arrays = {
        "timestamps": _s3_array_sha256(body.timestamps, label="timestamps"),
        "features": _s3_array_sha256(body.features, label="features"),
        "returns_v4": _s3_array_sha256(body.returns_v4, label="returns_v4"),
        "control_returns": _s3_array_sha256(
            body.control_returns, label="control_returns"
        ),
        "injected_returns": _s3_array_sha256(
            body.injected_returns, label="injected_returns"
        ),
        "injection_mask": _s3_array_sha256(
            body.injection_mask, label="injection_mask"
        ),
        "z_scores": _s3_array_sha256(body.z_scores, label="z_scores"),
        "context_mask": _s3_array_sha256(body.context_mask, label="context_mask"),
    }
    for name in REQUIRED_AVAILABILITY_COLUMNS:
        try:
            arrays[f"availability.{name}"] = _s3_array_sha256(
                body.availability[name], label=f"availability.{name}"
            )
        except (KeyError, TypeError) as exc:
            raise P1RunnerError("S3 source hash is missing availability masks") from exc
    payload = {
        "schema": "unidream.p1.s3_authenticated_body",
        "version": 1,
        "runtime": _s3_hash_plain(body.runtime),
        "arrays": arrays,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_production_s3_body(body: S3InjectionControl) -> S3InjectionControl:
    """Reject fixture/direct-constructor bodies at every production fit boundary."""

    if not isinstance(body, S3InjectionControl):
        raise P1RunnerError("S3 production requires an S3InjectionControl")
    if body._production_seal is not _S3_BODY_SEAL:
        raise P1RunnerError(
            "S3 body was not materialized by the authenticated public runtime loader"
        )
    if (
        not isinstance(body.body_sha256, str)
        or len(body.body_sha256) != 64
        or any(character not in "0123456789abcdef" for character in body.body_sha256)
    ):
        raise P1RunnerError("S3 authenticated body digest is malformed")
    if _s3_body_sha256(body) != body.body_sha256:
        raise P1RunnerError("S3 authenticated body digest mismatch")
    return body


def _require_production_s3_arm(dataset: S3ArmDataset) -> S3ArmDataset:
    """Bind an S3 arm to the still-authenticated immutable source body."""

    if dataset._production_seal is not _S3_BODY_SEAL:
        raise P1RunnerError("S3 arm was not built by the authenticated arm builder")
    source = _require_production_s3_body(dataset.source)
    if dataset.source_body_sha256 != source.body_sha256:
        raise P1RunnerError("S3 arm source digest mismatch")
    if dataset.seed != 20260830 or dataset.arm not in {
        "injected",
        "zero_injection_control",
    }:
        raise P1RunnerError("S3 arm identity differs from the preregistered arm")
    expected_beta = S3_INJECTION_BETA if dataset.arm == "injected" else 0.0
    if dataset.beta != expected_beta:
        raise P1RunnerError("S3 arm beta differs from the preregistered value")
    expected_returns = (
        source.injected_returns
        if dataset.arm == "injected"
        else source.control_returns
    )
    exact_arrays = (
        ("timestamps", dataset.timestamps, source.timestamps),
        ("features", dataset.features, source.features),
        ("returns", dataset.returns, expected_returns),
        ("context_mask", dataset.context_mask, source.context_mask),
    )
    for label, actual, expected in exact_arrays:
        if not np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True):
            raise P1RunnerError(f"S3 arm {label} differs from its authenticated source")
    for name in REQUIRED_AVAILABILITY_COLUMNS:
        try:
            matches = np.array_equal(
                np.asarray(dataset.availability[name]),
                np.asarray(source.availability[name]),
            )
        except (KeyError, TypeError) as exc:
            raise P1RunnerError("S3 arm availability is incomplete") from exc
        if not matches:
            raise P1RunnerError(
                f"S3 arm availability {name} differs from its authenticated source"
            )
    return dataset


def _require_authenticated_v4_result(
    runtime_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reject generic/forged body mappings at the S3 construction boundary."""

    if not isinstance(runtime_result, Mapping):
        raise P1RunnerError(
            "S3 body must be the mapping returned by validate_p1_v4_runtime_inputs"
        )
    expected_identity = {
        "v4_runtime_validation_status": "passed",
        "p1_runtime_validation_entrypoint": P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
        "p1_runtime_body_validator_entrypoint": V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT,
        "results_observed": False,
        "manifest_sha256": REGISTERED_MANIFEST_SHA256,
        "p1_manifest_sha256": REGISTERED_MANIFEST_SHA256,
        "v4_runtime_loaded_body_match": True,
    }
    for field, expected in expected_identity.items():
        if runtime_result.get(field) != expected:
            raise P1RunnerError(
                "S3 body must carry the authenticated v4 wrapper identity: "
                f"{field} mismatch"
            )
    metadata = runtime_result.get("metadata")
    if not isinstance(metadata, Mapping):
        raise P1RunnerError("authenticated S3 body is missing frozen v4 metadata")
    disposition = runtime_result.get("v4_runtime_provenance_disposition")
    if not isinstance(disposition, Mapping):
        raise P1RunnerError("authenticated S3 body is missing v4 provenance disposition")
    return runtime_result


def _immutable_provenance_value(value: Any) -> Any:
    """Deep-freeze the scalar provenance subset retained by an S3 result."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable_provenance_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_provenance_value(item) for item in value)
    if isinstance(value, np.generic):
        return _immutable_provenance_value(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise P1RunnerError("S3 provenance echo contains a non-scalar mutable value")


def _s3_provenance_echo(runtime_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Retain only immutable authenticated identity/provenance evidence."""

    scalar_fields = (
        "status",
        "manifest_id",
        "manifest_sha256",
        "base_revision",
        "results_observed",
        "p1_manifest_id",
        "p1_manifest_sha256",
        "p1_base_revision",
        "p1_results_observed",
        "p1_runtime_validation_entrypoint",
        "p1_runtime_body_validator_entrypoint",
        "v4_runtime_validation_status",
        "v4_runtime_body_match",
        "v4_runtime_loaded_body_match",
        "v4_runtime_source_provenance_match",
        "v4_runtime_frozen_metadata_sha256",
        "v4_runtime_cache_local_metadata_sha256",
        "v4_runtime_cache_local_source_provenance_digest",
        "v4_runtime_cache_local_schema_digest",
        "v4_frozen_metadata_sha256",
        "v4_frozen_source_provenance_digest",
        "v4_cache_local_metadata_sha256",
        "v4_cache_local_source_provenance_digest",
    )
    echo: dict[str, Any] = {
        field: _immutable_provenance_value(runtime_result[field])
        for field in scalar_fields
        if field in runtime_result
    }
    paths = runtime_result.get("paths")
    if isinstance(paths, Mapping):
        echo["paths"] = _immutable_provenance_value(paths)
    disposition = runtime_result.get("v4_runtime_provenance_disposition")
    if isinstance(disposition, Mapping):
        echo["v4_runtime_provenance_disposition"] = _immutable_provenance_value(
            disposition
        )
    metadata = runtime_result.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_fields = (
            "cache_tag",
            "schema_version",
            "schema_digest",
            "content_digests",
            "rows",
            "sidecar_rows",
            "feature_columns",
            "availability_columns",
            "returns_columns",
        )
        echo["frozen_metadata"] = _immutable_provenance_value(
            {field: metadata[field] for field in metadata_fields if field in metadata}
        )
    return MappingProxyType(echo)


def _s3_body_timestamps(
    frame: pd.DataFrame,
    *,
    label: str,
) -> np.ndarray:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise P1RunnerError(f"S3 {label} index must be a DatetimeIndex")
    try:
        return validate_15m_timestamps(np.asarray(frame.index), label=f"S3 {label} timestamps")
    except P1RunnerError:
        raise
    except (TypeError, ValueError) as exc:
        raise P1RunnerError(f"S3 {label} timestamps are invalid") from exc


def _s3_return_values(
    returns_body: Any,
    feature_index: pd.DatetimeIndex,
) -> np.ndarray:
    if isinstance(returns_body, pd.DataFrame):
        if returns_body.shape[1] != 1:
            raise P1RunnerError("authenticated S3 returns must contain one column")
        series = returns_body.iloc[:, 0]
    elif isinstance(returns_body, pd.Series):
        series = returns_body
    else:
        raise P1RunnerError("authenticated S3 returns must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.equals(feature_index):
        raise P1RunnerError("authenticated S3 features and returns indices differ")
    try:
        values = series.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise P1RunnerError("authenticated S3 returns must be numeric") from exc
    if values.ndim != 1 or len(values) != len(feature_index):
        raise P1RunnerError("authenticated S3 returns are not row-aligned")
    return _read_only(values, dtype=np.float64)


def _s3_availability_values(
    availability_body: Any,
    feature_index: pd.DatetimeIndex,
) -> Mapping[str, np.ndarray]:
    required = tuple(REQUIRED_AVAILABILITY_COLUMNS)
    if isinstance(availability_body, pd.DataFrame):
        if not isinstance(availability_body.index, pd.DatetimeIndex):
            raise P1RunnerError("authenticated S3 availability index must be a DatetimeIndex")
        if not availability_body.index.is_unique or not availability_body.index.is_monotonic_increasing:
            raise P1RunnerError("authenticated S3 availability index must be ordered and unique")
        if not feature_index.isin(availability_body.index).all():
            raise P1RunnerError("authenticated S3 availability does not cover feature timestamps")
        missing = [name for name in required if name not in availability_body.columns]
        if missing:
            raise P1RunnerError("authenticated S3 availability is missing: " + ", ".join(missing))
        aligned = availability_body.loc[feature_index, list(required)]
        if aligned.isna().any().any():
            raise P1RunnerError("authenticated S3 availability contains missing values")
        for name in required:
            if aligned[name].dtype != np.dtype(np.bool_):
                raise P1RunnerError(f"authenticated S3 availability {name} must be bool")
        return MappingProxyType(
            {
                name: _read_only(aligned[name].to_numpy(copy=True), dtype=np.bool_)
                for name in required
            }
        )
    if not isinstance(availability_body, Mapping):
        raise P1RunnerError("authenticated S3 availability must be a DataFrame")
    values: dict[str, np.ndarray] = {}
    for name in required:
        if name not in availability_body:
            raise P1RunnerError(f"authenticated S3 availability is missing: {name}")
        array = np.asarray(availability_body[name])
        if array.dtype != np.dtype(np.bool_) or array.ndim != 1 or len(array) != len(feature_index):
            raise P1RunnerError(f"authenticated S3 availability {name} must be a strict bool row vector")
        values[name] = _read_only(array, dtype=np.bool_)
    return MappingProxyType(values)


def _prepare_s3_injection_control(
    runtime_result: Mapping[str, Any],
    *,
    _production_seal: object | None = None,
) -> S3InjectionControl:
    """Test-only materializer for an authenticated v4 wrapper result.

    The named ``close_ret`` column is standardized using only prior
    context-eligible rows (minimum prefix 256), then applied to the next
    contiguous, Spot-observed return.  No hidden/generated feature is added
    to the model input and the original row grid is never sorted or compacted.
    """

    result = _require_authenticated_v4_result(runtime_result)
    features_body = result.get("features")
    if not isinstance(features_body, pd.DataFrame):
        raise P1RunnerError("authenticated S3 features must be a pandas DataFrame")
    if list(features_body.columns) != list(MODEL_FEATURE_COLUMNS):
        raise P1RunnerError("authenticated S3 features must equal canonical 17 columns")
    timestamps = _s3_body_timestamps(features_body, label="feature")
    feature_index = features_body.index
    try:
        feature_values = features_body.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise P1RunnerError("authenticated S3 features must be numeric") from exc
    if feature_values.shape != (len(timestamps), len(MODEL_FEATURE_COLUMNS)):
        raise P1RunnerError("authenticated S3 features are not row-aligned")
    feature_values = _read_only(feature_values, dtype=np.float64)
    returns = _s3_return_values(result.get("returns"), feature_index)
    availability = _s3_availability_values(result.get("availability"), feature_index)
    context_mask = build_context_mask(
        feature_values,
        availability,
        timestamps=timestamps,
    )
    edges = timestamp_edge_mask(timestamps)
    close_index = list(MODEL_FEATURE_COLUMNS).index(S3_SIGNAL_FEATURE)
    close_ret = feature_values[:, close_index]
    n_rows = len(timestamps)
    prefix_context_count = np.cumsum(context_mask.astype(np.int64), dtype=np.int64)
    prefix_context_sum = np.cumsum(
        np.where(context_mask, close_ret, 0.0), dtype=np.float64
    )
    prefix_context_sumsq = np.cumsum(
        np.where(context_mask, close_ret * close_ret, 0.0), dtype=np.float64
    )
    z_scores = np.full(n_rows, np.nan, dtype=np.float64)
    injection_mask = np.zeros(n_rows, dtype=np.bool_)
    injected_returns = np.array(returns, dtype=np.float64, copy=True)
    spot = availability["spot_bar_observed"]
    for decision in range(n_rows):
        if not context_mask[decision] or decision == 0:
            continue
        next_row = decision + 1
        if next_row >= n_rows or not edges[decision] or not spot[next_row]:
            continue
        if not np.isfinite(returns[next_row]):
            continue
        prior_count = int(prefix_context_count[decision - 1])
        if prior_count < S3_PREFIX_ROWS_MIN:
            continue
        prior_sum = float(prefix_context_sum[decision - 1])
        prior_sumsq = float(prefix_context_sumsq[decision - 1])
        prior_mean = prior_sum / prior_count
        variance = max(prior_sumsq / prior_count - prior_mean * prior_mean, 0.0)
        prior_std = float(np.sqrt(variance))
        z_value = (float(close_ret[decision]) - prior_mean) / max(prior_std, 1e-12)
        if not np.isfinite(z_value):
            continue
        z_scores[decision] = z_value
        injected_returns[next_row] += S3_INJECTION_BETA * z_value
        injection_mask[decision] = True
    if _production_seal not in (None, _S3_BODY_SEAL):
        raise P1RunnerError("invalid S3 production seal")
    body = S3InjectionControl(
        runtime=_s3_provenance_echo(result),
        timestamps=timestamps,
        features=feature_values,
        returns_v4=returns,
        control_returns=_read_only(returns, dtype=np.float64),
        injected_returns=_read_only(injected_returns, dtype=np.float64),
        injection_mask=_read_only(injection_mask, dtype=np.bool_),
        z_scores=_read_only(z_scores, dtype=np.float64),
        context_mask=context_mask,
        availability=availability,
        _production_seal=_production_seal,
    )
    return replace(body, body_sha256=_s3_body_sha256(body))


def load_s3_validation_body(
    manifest: Mapping[str, Any] | str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    root: str | Path | None = None,
    path_overrides: Mapping[str, str | Path] | None = None,
    paths: Mapping[str, str | Path] | None = None,
    feature_path: str | Path | None = None,
    returns_path: str | Path | None = None,
    availability_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    cache_local_metadata_path: str | Path | None = None,
    provenance_disposition: Mapping[str, Any] | None = None,
) -> S3InjectionControl:
    """Authenticate v4 first, then construct the preregistered S3 arm pair."""

    from .runtime import validate_p1_v4_runtime_inputs

    validated = validate_p1_v4_runtime_inputs(
        manifest,
        manifest_path=manifest_path,
        root=root,
        path_overrides=path_overrides,
        paths=paths,
        feature_path=feature_path,
        returns_path=returns_path,
        availability_path=availability_path,
        metadata_path=metadata_path,
        cache_local_metadata_path=cache_local_metadata_path,
        provenance_disposition=provenance_disposition,
    )
    return _prepare_s3_injection_control(
        validated,
        _production_seal=_S3_BODY_SEAL,
    )


def build_s3_arm_dataset(
    body: S3InjectionControl,
    arm: Literal["injected", "zero_injection_control"],
) -> S3ArmDataset:
    """Build canonical targets for one authenticated S3 arm without refitting."""

    body = _require_production_s3_body(body)
    if arm not in {"injected", "zero_injection_control"}:
        raise P1RunnerError("S3 arm must be 'injected' or 'zero_injection_control'")
    runtime = body.runtime
    required_runtime = {
        "v4_runtime_validation_status": "passed",
        "p1_manifest_sha256": REGISTERED_MANIFEST_SHA256,
        "p1_results_observed": False,
        "v4_runtime_loaded_body_match": True,
    }
    for field, expected in required_runtime.items():
        if runtime.get(field) != expected:
            raise P1RunnerError(f"S3 authenticated provenance echo mismatch: {field}")
    if len(body.features) != S3_BODY_ROWS or len(body.timestamps) != S3_BODY_ROWS:
        raise P1RunnerError("S3 body does not match the preregistered 173111-row v4 body")
    boundary_timestamps = {
        52_491: np.datetime64("2020-01-01T00:00:00", "ns"),
        S3_VALIDATION_ORIGIN: np.datetime64("2022-01-01T00:00:00", "ns"),
        S3_VALIDATION_END: np.datetime64("2023-01-01T00:00:00", "ns"),
    }
    for index, expected in boundary_timestamps.items():
        if body.timestamps[index] != expected:
            raise P1RunnerError(f"S3 raw boundary timestamp mismatch at index {index}")
    selected_returns = (
        body.injected_returns if arm == "injected" else body.control_returns
    )
    targets, target_mask, target_end = build_target_arrays(
        selected_returns,
        body.availability["spot_bar_observed"],
        horizons=FORECAST_HORIZONS,
        timestamps=body.timestamps,
    )
    labels = binary_labels_from_targets(targets)
    expected_context = build_context_mask(
        body.features,
        body.availability,
        timestamps=body.timestamps,
    )
    if not np.array_equal(expected_context, body.context_mask):
        raise P1RunnerError("S3 context mask does not rederive from its v4 source body")
    return S3ArmDataset(
        seed=20260830,
        arm=arm,
        beta=S3_INJECTION_BETA if arm == "injected" else 0.0,
        timestamps=body.timestamps,
        source=body,
        features=body.features,
        returns=_read_only(selected_returns, dtype=np.float64),
        targets=targets,
        target_end=target_end,
        target_mask=target_mask,
        binary_labels=labels,
        context_mask=expected_context,
        availability=body.availability,
        source_body_sha256=body.body_sha256,
        _production_seal=_S3_BODY_SEAL,
    )


RunnerDataset = SyntheticDataset | S3ArmDataset


def _horizon_column(dataset: RunnerDataset, horizon: int) -> int:
    _strict_horizon(horizon)
    try:
        return FORECAST_HORIZONS.index(int(horizon))
    except ValueError as exc:  # pragma: no cover - guarded by _strict_horizon
        raise P1RunnerError(f"unsupported horizon: {horizon}") from exc


def _ensure_dataset(dataset: RunnerDataset) -> RunnerDataset:
    if not isinstance(dataset, (SyntheticDataset, S3ArmDataset)):
        raise P1RunnerError("runner model APIs require a canonical P1 dataset")
    if isinstance(dataset, S3ArmDataset):
        _require_production_s3_arm(dataset)
    features = validate_current_row_features(dataset.features)
    n_rows = len(features)
    timestamps = validate_15m_timestamps(dataset.timestamps, n_rows=n_rows)
    returns = np.asarray(dataset.returns)
    if returns.shape != (n_rows,) or not np.issubdtype(returns.dtype, np.number):
        raise P1RunnerError("dataset returns are not row-aligned")
    returns = np.asarray(returns, dtype=np.float64)
    if np.isinf(returns).any():
        raise P1RunnerError("dataset returns must not contain infinity")
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
    if not np.array_equal(np.isfinite(targets), target_mask):
        raise P1RunnerError(
            "dataset targets must be finite exactly where target_mask is true"
        )
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
    # A return gap is represented by a non-finite value together with an
    # unobserved Spot bar.  It is valid source evidence for evaluation rows,
    # but a non-finite return on an observed Spot bar is malformed.  Do not
    # replace missing values with zero; target construction keeps them
    # masked instead.
    spot_observed = np.asarray(availability["spot_bar_observed"])
    if np.any(spot_observed & ~np.isfinite(returns)):
        raise P1RunnerError(
            "dataset returns may be non-finite only when spot_bar_observed is false"
        )

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
    dataset: RunnerDataset,
    origin: int,
    horizon: int,
    *,
    purge_bars: int = PURGE_BARS,
    train_start: int = 0,
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
    if purge_int != PURGE_BARS:
        raise P1RunnerError(
            f"production train masks require the fixed purge_bars={PURGE_BARS}"
        )
    if origin_int > len(data.features):
        raise P1RunnerError("origin exceeds dataset rows")
    train_start_int = _strict_origin(train_start)
    if train_start_int >= origin_int:
        raise P1RunnerError("train_start must be strictly before origin")
    column = _horizon_column(data, horizon_int)
    rows = np.arange(len(data.features), dtype=np.int64)
    mask = (
        data.context_mask
        & data.target_mask[:, column]
        & (data.target_end[:, column] <= origin_int - purge_int)
        & (rows >= train_start_int)
        & (rows < origin_int)
    )
    return _read_only(mask, dtype=np.bool_)


def build_train_mask(*args: Any, **kwargs: Any) -> np.ndarray:
    """Compatibility alias for :func:`train_mask_for_origin`."""

    return train_mask_for_origin(*args, **kwargs)


def prediction_mask_for_range(
    dataset: RunnerDataset,
    horizon: int,
    *,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Return the causal inference mask inside a registered support range.

    This mask is deliberately independent of future target availability.  A
    decision row is available for inference when its current-inclusive
    context is complete and its deterministic ``target_end`` remains inside
    the requested right-exclusive support boundary.  The latter is a known
    split geometry rule, not an inspection of future returns or a target
    availability sidecar.  Use :func:`score_eligible_mask_for_range` when a
    complete future target is required for evaluation.
    """

    data = _ensure_dataset(dataset)
    horizon_int = _strict_horizon(horizon)
    start_int = _strict_origin(start)
    end_int = len(data.features) if end is None else _strict_origin(end)
    if end_int <= start_int:
        raise P1RunnerError("prediction support range must satisfy end > start")
    if start_int > len(data.features) or end_int > len(data.features):
        raise P1RunnerError("prediction support range is outside the dataset")
    column = _horizon_column(data, horizon_int)
    rows = np.arange(len(data.features), dtype=np.int64)
    mask = (
        data.context_mask
        & (data.target_end[:, column] <= end_int)
        & (rows >= start_int)
        & (rows < end_int)
    )
    return _read_only(mask, dtype=np.bool_)


def inference_mask_for_range(
    dataset: RunnerDataset,
    horizon: int,
    *,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Return the causal inference mask for one horizon/support range.

    ``prediction_mask_for_range`` is retained as the historical public name
    for this causal mask.  This explicit alias makes the distinction visible
    at call sites and prevents a future implementation from accidentally
    coupling model inference to a future label/outcome gap.
    """

    return prediction_mask_for_range(dataset, horizon, start=start, end=end)


def score_eligible_mask_for_range(
    dataset: RunnerDataset,
    horizon: int,
    *,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Return context-and-target score eligibility inside a support range.

    The score mask is intentionally a strict subset of the causal inference
    mask whenever a future target/outcome gap is present.  It is the only
    range mask that consults ``target_mask``; model inference and action
    eligibility must use :func:`inference_mask_for_range` instead.
    """

    data = _ensure_dataset(dataset)
    horizon_int = _strict_horizon(horizon)
    start_int = _strict_origin(start)
    end_int = len(data.features) if end is None else _strict_origin(end)
    if end_int <= start_int:
        raise P1RunnerError("score support range must satisfy end > start")
    if start_int > len(data.features) or end_int > len(data.features):
        raise P1RunnerError("score support range is outside the dataset")
    column = _horizon_column(data, horizon_int)
    inference = prediction_mask_for_range(
        data,
        horizon_int,
        start=start_int,
        end=end_int,
    )
    score = inference & data.target_mask[:, column]
    return _read_only(score, dtype=np.bool_)


@dataclass(frozen=True)
class ModelFit:
    """One origin x horizon fit and its full-grid forecast view."""

    model_id: str
    task: Literal["continuous", "binary"]
    horizon: int
    origin: int
    train_start: int
    train_mask: np.ndarray
    eligible_mask: np.ndarray
    prediction_mask: np.ndarray
    predictions: np.ndarray
    status: str
    reason: str | None
    scaler: StandardScaler | None
    estimator: Ridge | LogisticRegression | None

    def __post_init__(self) -> None:
        """Reject forged or internally inconsistent fit masks and outputs."""

        if not isinstance(self.model_id, str) or self.model_id not in MODEL_IDS:
            raise P1RunnerError("model fit has an unknown model_id")
        if not isinstance(self.task, str) or self.task not in {"continuous", "binary"}:
            raise P1RunnerError("model fit task must be continuous or binary")
        if self.task not in _MODEL_ALLOWED_TASKS[self.model_id]:
            raise P1RunnerError("model fit task is not allowed for this model")
        horizon_int = _strict_horizon(self.horizon)
        origin_int = _strict_origin(self.origin)
        object.__setattr__(self, "horizon", horizon_int)
        object.__setattr__(self, "origin", origin_int)
        if self.status not in {"ok", "N/A"}:
            raise P1RunnerError("model fit status must be ok or N/A")
        if self.reason is not None and not isinstance(self.reason, str):
            raise P1RunnerError("model fit reason must be a string or None")

        train_mask = np.asarray(self.train_mask)
        eligible_mask = np.asarray(self.eligible_mask)
        prediction_mask = np.asarray(self.prediction_mask)
        predictions = np.asarray(self.predictions)
        if (
            train_mask.ndim != 1
            or eligible_mask.ndim != 1
            or prediction_mask.ndim != 1
            or predictions.ndim != 1
        ):
            raise P1RunnerError("model fit masks and predictions must be one-dimensional")
        n_rows = len(train_mask)
        if (
            len(eligible_mask) != n_rows
            or len(prediction_mask) != n_rows
            or len(predictions) != n_rows
        ):
            raise P1RunnerError("model fit masks and predictions must be row-aligned")
        for name, mask in (
            ("train_mask", train_mask),
            ("eligible_mask", eligible_mask),
            ("prediction_mask", prediction_mask),
        ):
            if mask.dtype != np.dtype(np.bool_):
                raise P1RunnerError(f"model fit {name} must have strict bool dtype")
        if predictions.dtype != np.dtype(np.float64):
            raise P1RunnerError("model fit predictions must use float64 values")
        # ``prediction_mask`` is the finite, causal inference mask.  It must
        # not be narrowed by a future target/outcome gap.  ``eligible_mask``
        # is the evaluation score mask and may therefore be a strict subset
        # of the inference rows.  An N/A fit may retain its score mask as
        # evidence while carrying no predictions, so the subset invariant is
        # enforced only for successful fits.
        finite_predictions = np.isfinite(predictions)
        if not np.array_equal(finite_predictions, prediction_mask):
            raise P1RunnerError(
                "model fit predictions must be finite exactly where prediction_mask is true"
            )
        if self.status == "ok":
            if np.any(eligible_mask & ~prediction_mask):
                raise P1RunnerError("model fit eligible_mask exceeds prediction_mask")
            if not np.any(prediction_mask):
                raise P1RunnerError("successful model fits require non-empty inference coverage")
        elif np.any(prediction_mask):
            raise P1RunnerError("N/A model fits cannot contain predictions")
        object.__setattr__(self, "train_mask", _read_only(train_mask, dtype=np.bool_))
        object.__setattr__(self, "eligible_mask", _read_only(eligible_mask, dtype=np.bool_))
        object.__setattr__(self, "prediction_mask", _read_only(prediction_mask, dtype=np.bool_))
        object.__setattr__(self, "predictions", _read_only(predictions, dtype=np.float64))

    @property
    def is_na(self) -> bool:
        return self.status == "N/A"

    @property
    def inference_mask(self) -> np.ndarray:
        """Causal rows receiving a forecast; alias of ``prediction_mask``."""

        return self.prediction_mask

    @property
    def score_eligible_mask(self) -> np.ndarray:
        """Rows with a complete target available for evaluation."""

        return self.eligible_mask


def clip_probabilities(probabilities: Any, *, eps: float = PROBABILITY_CLIP_EPS) -> np.ndarray:
    """Clip finite probability values to the preregistered open interval."""

    if not np.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise P1RunnerError("probability clip eps must be finite and in (0, 0.5)")
    array = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(array).all():
        raise P1RunnerError("probabilities must be finite before clipping")
    return _read_only(np.clip(array, eps, 1.0 - eps), dtype=np.float64)


def _prediction_mask(
    dataset: RunnerDataset,
    horizon: int,
    origin: int,
    prediction_range: tuple[int, int] | None,
) -> np.ndarray:
    if prediction_range is None:
        start, end = origin, len(dataset.features)
    else:
        if not isinstance(prediction_range, tuple) or len(prediction_range) != 2:
            raise P1RunnerError("prediction_range must be (start, end)")
        start, end = prediction_range
    start_int = _strict_origin(start)
    end_int = _strict_origin(end)
    origin_int = _strict_origin(origin)
    if start_int < origin_int:
        raise P1RunnerError("prediction_range start must be at or after origin")
    if end_int <= start_int:
        raise P1RunnerError("prediction_range must satisfy end > start")
    return prediction_mask_for_range(dataset, horizon, start=start_int, end=end_int)


def _validate_model_fit_arrays(
    *,
    eligible_mask: Any,
    prediction_mask: Any,
    predictions: Any,
    require_coverage: bool,
) -> None:
    """Validate the causal inference/score finite-prediction contract."""

    eligible = np.asarray(eligible_mask)
    predicted = np.asarray(prediction_mask)
    values = np.asarray(predictions)
    if eligible.dtype != np.dtype(np.bool_) or predicted.dtype != np.dtype(np.bool_):
        raise P1RunnerError("model fit masks must have strict bool dtype")
    if values.dtype != np.dtype(np.float64):
        raise P1RunnerError("model fit predictions must use float64 values")
    if eligible.ndim != 1 or predicted.ndim != 1 or values.ndim != 1:
        raise P1RunnerError("model fit masks and predictions must be one-dimensional")
    if len(eligible) != len(predicted) or len(eligible) != len(values):
        raise P1RunnerError("model fit masks and predictions must be row-aligned")
    if np.any(eligible & ~predicted):
        raise P1RunnerError("model fit eligible_mask exceeds prediction_mask")
    if not np.array_equal(np.isfinite(values), predicted):
        raise P1RunnerError(
            "model fit predictions must be finite exactly where prediction_mask is true"
        )
    if require_coverage:
        if not np.any(predicted):
            raise P1RunnerError("production fit requires non-empty inference coverage")
        if np.any(eligible & ~predicted):
            raise P1RunnerError("production fit score eligibility exceeds inference coverage")


def _na_model_fit(
    *,
    model_id: str,
    task: Literal["continuous", "binary"],
    horizon: int,
    origin: int,
    train_start: int,
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
        train_start=train_start,
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
    dataset: RunnerDataset,
    model_id: str,
    origin: int,
    horizon: int,
    *,
    task: Literal["continuous", "binary"] | None = None,
    prediction_range: tuple[int, int] | None = None,
    train_start: int = 0,
) -> ModelFit:
    """Fit one fixed model using only its origin/horizon admissible prefix.

    Ridge and Logistic each get a fresh ``StandardScaler`` fit only on their
    own origin x horizon training rows.  Baselines do not scale.  A Logistic
    prefix with one observed class is returned as N/A without repair.
    """

    data = _ensure_dataset(dataset)
    if not isinstance(model_id, str) or model_id not in MODEL_IDS:
        raise P1RunnerError(f"unknown model_id: {model_id}")
    expected_task = MODEL_TASKS[model_id]
    resolved_task = expected_task if task is None else task
    if not isinstance(resolved_task, str) or resolved_task not in {"continuous", "binary"}:
        raise P1RunnerError("task must be 'continuous' or 'binary'")
    if resolved_task not in _MODEL_ALLOWED_TASKS[model_id]:
        allowed = ", ".join(_MODEL_ALLOWED_TASKS[model_id])
        raise P1RunnerError(f"model {model_id} only supports task(s): {allowed}")
    horizon_int = _strict_horizon(horizon)
    origin_int = _strict_origin(origin)
    if origin_int > len(data.features):
        raise P1RunnerError("origin exceeds dataset rows")
    train_start_int = _strict_origin(train_start)
    train_mask = train_mask_for_origin(
        data,
        origin_int,
        horizon_int,
        train_start=train_start_int,
    )
    inference_mask = inference_mask_for_range(
        data,
        horizon_int,
        start=origin_int,
        end=len(data.features),
    )
    score_eligible_mask = score_eligible_mask_for_range(
        data,
        horizon_int,
        start=origin_int,
        end=len(data.features),
    )
    if prediction_range is not None:
        inference_mask = _prediction_mask(data, horizon_int, origin_int, prediction_range)
        score_eligible_mask = score_eligible_mask_for_range(
            data,
            horizon_int,
            start=prediction_range[0],
            end=prediction_range[1],
        )
    if not np.any(inference_mask):
        raise P1RunnerError(
            "production fit requires at least one causal inference row"
        )
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
            train_start=train_start_int,
            train_mask=train_mask,
            eligible_mask=score_eligible_mask,
            predictions=predictions,
            reason=f"fewer than {MIN_HISTORY_ROWS} admissible training rows",
        )

    if model_id == "zero_return":
        predictions[inference_mask] = 0.0 if resolved_task == "continuous" else 0.5
        prediction_mask = inference_mask.copy()
    elif model_id == "persistence_last_observed":
        if not np.isfinite(data.returns[inference_mask]).all():
            raise P1RunnerError(
                "persistence inference rows must have finite current observed returns"
            )
        if resolved_task == "continuous":
            predictions[inference_mask] = horizon_int * data.returns[inference_mask]
        else:
            positive = data.returns[inference_mask] > 0.0
            predictions[inference_mask] = np.where(
                positive,
                1.0 - PROBABILITY_CLIP_EPS,
                PROBABILITY_CLIP_EPS,
            )
        prediction_mask = inference_mask.copy()
    elif model_id == "ridge":
        X_train = data.features[train_mask]
        y_train = data.targets[train_mask, column]
        if len(X_train) == 0:
            return _na_model_fit(
                model_id=model_id,
                task=resolved_task,
                horizon=horizon_int,
                origin=origin_int,
                train_start=train_start_int,
                train_mask=train_mask,
                eligible_mask=score_eligible_mask,
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
        if np.any(inference_mask):
            predictions[inference_mask] = estimator.predict(
                scaler.transform(data.features[inference_mask])
            )
            prediction_mask = inference_mask.copy()
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
                train_start=train_start_int,
                train_mask=train_mask,
                eligible_mask=score_eligible_mask,
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
        if np.any(inference_mask):
            class_one_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
            raw_probability = estimator.predict_proba(
                scaler.transform(data.features[inference_mask])
            )[:, class_one_index]
            predictions[inference_mask] = clip_probabilities(raw_probability)
            prediction_mask = inference_mask.copy()

    _validate_model_fit_arrays(
        eligible_mask=score_eligible_mask,
        prediction_mask=prediction_mask,
        predictions=predictions,
        require_coverage=True,
    )
    return ModelFit(
        model_id=model_id,
        task=resolved_task,
        horizon=horizon_int,
        origin=origin_int,
        train_start=train_start_int,
        train_mask=train_mask,
        eligible_mask=score_eligible_mask,
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


def assert_future_perturbation_invariance(
    dataset: RunnerDataset,
    model_id: str,
    origin: int,
    horizon: int,
    *,
    prediction_range: tuple[int, int],
    perturb_start: int,
    delta: float = 0.123456789,
    task: Literal["continuous", "binary"] | None = None,
    train_start: int = 0,
) -> Mapping[str, Any]:
    """Fit twice and prove that a strictly later return mutation is invisible."""

    data = _ensure_dataset(dataset)
    origin_int = _strict_origin(origin)
    perturb_int = _strict_origin(perturb_start)
    if perturb_int <= origin_int or perturb_int >= len(data.returns):
        raise P1RunnerError("perturb_start must be strictly after origin and inside the body")
    try:
        delta_float = float(delta)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1RunnerError("future perturbation delta must be finite") from exc
    if not np.isfinite(delta_float) or delta_float == 0.0:
        raise P1RunnerError("future perturbation delta must be finite and non-zero")
    changed_returns = np.array(data.returns, dtype=np.float64, copy=True)
    future_view = changed_returns[perturb_int:]
    finite_future = np.isfinite(future_view)
    future_view[finite_future] = future_view[finite_future] + delta_float
    targets, target_mask, target_end = build_target_arrays(
        changed_returns,
        data.availability["spot_bar_observed"],
        horizons=FORECAST_HORIZONS,
        timestamps=data.timestamps,
    )
    changed = replace(
        data,
        returns=_read_only(changed_returns, dtype=np.float64),
        targets=targets,
        target_mask=target_mask,
        target_end=target_end,
        binary_labels=binary_labels_from_targets(targets),
    )
    original_fit = fit_model_at_origin(
        data,
        model_id,
        origin_int,
        horizon,
        task=task,
        prediction_range=prediction_range,
        train_start=train_start,
    )
    changed_fit = fit_model_at_origin(
        changed,
        model_id,
        origin_int,
        horizon,
        task=task,
        prediction_range=prediction_range,
        train_start=train_start,
    )
    horizon_column = _horizon_column(data, horizon)
    prefix = (
        (np.arange(len(data.features), dtype=np.int64) < perturb_int)
        & (data.target_end[:, horizon_column] <= perturb_int)
    )
    if not np.array_equal(original_fit.train_mask, changed_fit.train_mask):
        raise P1RunnerError("future perturbation changed the fitted-prefix mask")
    if not np.array_equal(
        original_fit.prediction_mask[prefix],
        changed_fit.prediction_mask[prefix],
    ):
        raise P1RunnerError("future perturbation changed an earlier prediction mask")
    earlier_predictions = original_fit.prediction_mask & prefix
    if not np.array_equal(
        original_fit.predictions[earlier_predictions],
        changed_fit.predictions[earlier_predictions],
    ):
        raise P1RunnerError("future perturbation changed an earlier prediction")
    return MappingProxyType(
        {
            "status": "passed",
            "origin": origin_int,
            "horizon": _strict_horizon(horizon),
            "perturb_start": perturb_int,
            "earlier_prediction_count": int(np.count_nonzero(earlier_predictions)),
        }
    )


@dataclass(frozen=True)
class OOFRun:
    """Chronological OOF forecasts with no action or result-artifact layer."""

    plan: RunnerPlan
    dataset: SyntheticDataset
    fits: Mapping[tuple[int, int, str, Literal["continuous", "binary"]], ModelFit]
    outer_report_only: bool = True
    outer_test_executed: bool = False

    def get(
        self,
        origin: int,
        horizon: int,
        model_id: str,
        task: Literal["continuous", "binary"] | None = None,
    ) -> ModelFit:
        """Get one explicitly task-qualified forecast using strict key types."""

        origin_int = _strict_origin(origin)
        horizon_int = _strict_horizon(horizon)
        if not isinstance(model_id, str) or model_id not in MODEL_IDS:
            raise P1RunnerError(f"unknown model_id: {model_id}")
        if not isinstance(task, str) or task not in {"continuous", "binary"}:
            raise P1RunnerError("OOF get requires an explicit continuous/binary task")
        if task not in _MODEL_ALLOWED_TASKS[model_id]:
            raise P1RunnerError(
                f"model {model_id} does not support task {task!r}"
            )
        return self.fits[(origin_int, horizon_int, model_id, task)]


@dataclass(frozen=True)
class S3ValidationRun:
    """The one fixed 2022 S3 validation fit operation; outer remains sealed."""

    manifest_echo: ManifestEcho
    arm: Literal["injected", "zero_injection_control"]
    dataset: S3ArmDataset
    fits: Mapping[tuple[int, str, Literal["continuous", "binary"]], ModelFit]
    fit_range: tuple[int, int] = (S3_TRAIN_START, S3_VALIDATION_ORIGIN)
    prediction_range: tuple[int, int] = (S3_VALIDATION_ORIGIN, S3_VALIDATION_END)
    outer_report_only: bool = True
    outer_test_executed: bool = False

    def get(
        self,
        horizon: int,
        model_id: str,
        task: Literal["continuous", "binary"] | None = None,
    ) -> ModelFit:
        if model_id not in MODEL_IDS:
            raise P1RunnerError(f"unknown model_id: {model_id}")
        resolved_task = MODEL_TASKS[model_id] if task is None else task
        if resolved_task not in _MODEL_ALLOWED_TASKS[model_id]:
            raise P1RunnerError(
                f"model {model_id} does not support task {resolved_task!r}"
            )
        return self.fits[(_strict_horizon(horizon), model_id, resolved_task)]


def run_s3_validation_fits(
    body_or_dataset: S3InjectionControl | S3ArmDataset,
    arm: Literal["injected", "zero_injection_control"] | None = None,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    outer_report_only: bool = True,
) -> S3ValidationRun:
    """Execute only the preregistered S3 validation-boundary fits.

    This function never reads 2023 outer outcomes and has no outer execution
    option.  Injected/control calls receive separate fitted objects and later
    separate action inventory paths.
    """

    if outer_report_only is not True:
        raise P1OuterReportBlocked("S3 outer operation must remain report-only")
    if isinstance(body_or_dataset, S3InjectionControl):
        if arm is None:
            raise P1RunnerError("arm is required when building S3 from its source body")
        dataset = build_s3_arm_dataset(body_or_dataset, arm)
    elif isinstance(body_or_dataset, S3ArmDataset):
        dataset = body_or_dataset
        if arm is not None and arm != dataset.arm:
            raise P1RunnerError("requested S3 arm disagrees with the dataset arm")
    else:
        raise P1RunnerError("S3 validation requires an authenticated S3 dataset/body")
    data = _ensure_dataset(dataset)
    if not isinstance(data, S3ArmDataset):  # pragma: no cover - guarded above
        raise P1RunnerError("S3 validation dataset type mismatch")
    requested_models = tuple(model_ids)
    if not requested_models or any(model not in MODEL_IDS for model in requested_models):
        raise P1RunnerError("model_ids must be a non-empty subset of the fixed models")
    echo = build_runner_plan().manifest_echo
    fits: dict[tuple[int, str, Literal["continuous", "binary"]], ModelFit] = {}
    for horizon in FORECAST_HORIZONS:
        for model_id in requested_models:
            for task in _MODEL_ALLOWED_TASKS[model_id]:
                fits[(horizon, model_id, task)] = fit_model_at_origin(
                    data,
                    model_id,
                    S3_VALIDATION_ORIGIN,
                    horizon,
                    task=task,
                    prediction_range=(S3_VALIDATION_ORIGIN, S3_VALIDATION_END),
                    train_start=S3_TRAIN_START,
                )
    return S3ValidationRun(
        manifest_echo=echo,
        arm=data.arm,
        dataset=data,
        fits=MappingProxyType(fits),
    )


def run_synthetic_oof(
    dataset: SyntheticDataset | int,
    beta: float = 0.0,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    outer_report_only: bool = True,
) -> OOFRun:
    """Run fixture/diagnostic synthetic OOF fits; never a registered result run.

    ``outer_report_only`` is an API guard, not a request to execute the outer
    operation.  Passing ``False`` is rejected so a caller cannot accidentally
    convert this staged unit into outer selection/tuning.  This helper accepts
    one seed/beta and optional model subsets solely for deterministic unit
    tests.  A registered scenario runner must bind the complete manifest
    model/task grid, seed set, beta/level, and every required coverage key
    before fitting a report or gate result.
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
    fits: dict[tuple[int, int, str, Literal["continuous", "binary"]], ModelFit] = {}
    for origin in plan.origins:
        batch_end = min(origin + OOF_BATCH_SPAN, len(data.features))
        for horizon in plan.horizons:
            for model_id in requested_models:
                for task in _MODEL_ALLOWED_TASKS[model_id]:
                    fits[(origin, horizon, model_id, task)] = fit_model_at_origin(
                        data,
                        model_id,
                        origin,
                        horizon,
                        task=task,
                        prediction_range=(origin, batch_end),
                    )
    return OOFRun(
        plan=plan,
        dataset=data,
        fits=MappingProxyType(fits),
        outer_report_only=True,
        outer_test_executed=False,
    )


run_synthetic_oof_fixture = run_synthetic_oof
run_oof = run_synthetic_oof


__all__ = [
    "BAR_NS",
    "CONTEXT_BARS",
    "FEATURE_DIMENSION",
    "FORECAST_HORIZONS",
    "MIN_HISTORY_ROWS",
    "MODEL_IDS",
    "MODEL_TASKS",
    "OOF_BATCH_SPAN",
    "OOF_ORIGINS",
    "PROBABILITY_CLIP_EPS",
    "PURGE_BARS",
    "SYNTHETIC_OOF_SCOPE",
    "P1OuterReportBlocked",
    "P1RunnerError",
    "ManifestEcho",
    "ModelFit",
    "OOFRun",
    "OuterReportSpec",
    "RunnerPlan",
    "S3InjectionControl",
    "S3ArmDataset",
    "S3ValidationRun",
    "S3_BODY_ROWS",
    "S3_OUTER_END",
    "S3_TRAIN_START",
    "S3_VALIDATION_END",
    "S3_VALIDATION_ORIGIN",
    "SyntheticBase",
    "SyntheticDataset",
    "binary_labels_from_targets",
    "assert_future_perturbation_invariance",
    "build_context_mask",
    "build_runner_plan",
    "build_synthetic_dataset",
    "build_s3_arm_dataset",
    "build_target_arrays",
    "build_target_arrays_fixture",
    "build_train_mask",
    "build_context_mask_fixture",
    "clip_probabilities",
    "execute_outer_report",
    "fit_model_at_origin",
    "fit_origin_model",
    "generate_synthetic_base",
    "generate_synthetic_dataset",
    "load_runner_manifest",
    "load_s3_validation_body",
    "make_synthetic_dataset",
    "manifest_echo",
    "outer_report_spec",
    "reject_flattened_context",
    "run_oof",
    "run_s3_validation_fits",
    "run_synthetic_oof",
    "run_synthetic_oof_fixture",
    "inference_mask_for_range",
    "score_eligible_mask_for_range",
    "train_mask_for_origin",
    "timestamp_edge_mask",
    "validate_15m_timestamps",
    "validate_current_row_features",
    "validate_model_input",
]
