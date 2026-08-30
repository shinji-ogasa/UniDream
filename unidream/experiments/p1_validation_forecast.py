"""Fixed registered P1 validation forecast execution and persistence.

This module is the forecast-only boundary for the preregistered P1 recovery
operation.  It deliberately does not implement action replay, moving-block
bootstrap, promotion, or the outer report.  A production arm is bound to the
authenticated manifest *and* the two immutable result registries, executes the
complete horizon/model/task grid once, and persists immediately as one
support-scoped artifact.

The artifact is a canonical JSON representation.  Its file digest is
intentionally external: callers must provide the expected SHA-256 when loading
one.  This prevents a payload from authenticating itself by echoing a digest of
its own bytes.  ``ForecastActionSource`` is a sealed capability emitted only
by a successfully validated production load; action code can consume that
capability without accepting a plain mapping or a directly constructed
dataclass as an authenticated source.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Literal
import weakref

import numpy as np

from .p1_recovery_prereg import (
    DEFAULT_MANIFEST_PATH,
    REGISTERED_MANIFEST_SHA256,
    load_fixed_manifest,
)
from .p1_result_registry import (
    P1_PRIMARY_COMPARISON_COUNT,
    P1ResultRegistry,
    P1_TRIAL_COUNT,
    load_p1_result_registry,
)


class P1ForecastError(ValueError):
    """Raised when a registered validation forecast boundary is violated."""


class P1ForecastOuterBlocked(RuntimeError):
    """Raised whenever code attempts to execute the sealed outer operation."""


P1_FORECAST_FILE_FORMAT = "unidream.p1.validation_forecast.columnar_json"
P1_FORECAST_FILE_VERSION = 2
P1_FORECAST_FILE_MAX_BYTES = 64 * 1024 * 1024
P1_FORECAST_FILE_MAX_ROWS = 200_000
P1_FORECAST_FILE_MAX_FITS = 64
P1_FORECAST_SPLIT_ID = "validation"
P1_SYNTHETIC_SUPPORT_ID = "synthetic_validation"
P1_S3_SUPPORT_ID = "s3_validation"
P1_SYNTHETIC_SUPPORT_RANGE = (90_000, 100_000)
P1_SYNTHETIC_FIT_RANGE = (0, 90_000)
P1_SYNTHETIC_ORIGIN = 90_000
P1_S3_SUPPORT_RANGE = (104_528, 139_568)
P1_S3_FIT_RANGE = (52_492, 104_528)
P1_S3_ORIGIN = 104_528
P1_S3_TRAIN_START = 52_492
P1_S3_ROWS = 173_111
P1_SYNTHETIC_ROWS = 120_000
P1_CONTEXT_BARS = 64
P1_FIXED_HORIZONS = (1, 4, 8, 16)
P1_SYNTHETIC_SEEDS = tuple(range(20_260_830, 20_260_840))
P1_S3_SEEDS = (20_260_830,)
P1_VALIDATION_ARM_COUNT = 52
P1_REGISTERED_TRIAL_REGISTRY_SHA256 = "0f79c41ce0b8ec81c4f02e7ae556ac707779c0e23613cdacddd18b10cfedd587"
P1_REGISTERED_COMPARISON_REGISTRY_SHA256 = "bed67b607bc7d410add30a81e62f5d452bcc0a67ae3b59cce62744bd18b447db"
P1_PRIMARY_COMPARISON_IDS = (
    "S0__ridge__utility_vs_hold__cost_on",
    "S0__persistence__utility_vs_hold__cost_on",
    "S1__ridge__mse_vs_zero__cost_off",
    "S1__ridge__utility_vs_hold__cost_on",
    "S2__high_vs_medium__ridge__mse_skill__cost_off",
    "S2__high_vs_medium__ridge__normalized_regret__cost_on",
    "S2__high_vs_medium__ridge__utility__cost_on",
    "S2__high_vs_medium__ridge__agreement__cost_on",
    "S2__high_vs_medium__logistic__log_loss__cost_off",
    "S2__medium_vs_low__ridge__mse_skill__cost_off",
    "S2__medium_vs_low__ridge__normalized_regret__cost_on",
    "S2__medium_vs_low__ridge__utility__cost_on",
    "S2__medium_vs_low__ridge__agreement__cost_on",
    "S2__medium_vs_low__logistic__log_loss__cost_off",
    "S3__injected_vs_control__ridge__mse_skill_did__cost_off",
    "S3__injected_vs_control__ridge__utility__cost_on",
)
P1_SCENARIO_ARMS = (
    ("S0", "zero_signal"),
    ("S1", "known_high_snr_dgp"),
    ("S2-high", "high"),
    ("S2-medium", "medium"),
    ("S2-low", "low"),
    ("S3", "injected"),
    ("S3", "zero_injection_control"),
)
P1_SYNTHETIC_SCENARIO_ARMS = P1_SCENARIO_ARMS[:5]
P1_S3_SCENARIO_ARMS = P1_SCENARIO_ARMS[5:]
P1_ALLOWED_MODEL_TASK_KEYS = (
    ("zero_return", "continuous"),
    ("zero_return", "binary"),
    ("persistence_last_observed", "continuous"),
    ("persistence_last_observed", "binary"),
    ("ridge", "continuous"),
    ("logistic", "binary"),
)
P1_REQUIRED_COST_MODES = ("off", "on")
P1_COVERAGE_THRESHOLD_KEYS = (
    "eligible_origin_fraction",
    "label_complete_fraction",
    "finite_oof_prediction_fraction",
)
P1_FIXED_COVERAGE_THRESHOLDS = {
    "synthetic_eligible_origin_fraction_min": 0.9,
    "s3_eligible_origin_fraction_min": 0.5,
    "label_complete_fraction_min": 0.9,
    "finite_oof_prediction_fraction_min": 0.95,
    "scored_action_fraction_min": 0.8,
}
P1_FORECAST_SCHEMA_ID = "p1-validation-forecast-v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _runner_module():
    """Import the fit core lazily so the core can expose this module's API."""

    from . import p1_recovery_runner

    return p1_recovery_runner


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise P1ForecastError(f"{name} must be an integer")
    return int(value)


def _strict_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P1ForecastError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _strict_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise P1ForecastError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ForecastError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise P1ForecastError(f"{name} must be a finite number")
    return result


def _plain(value: Any) -> Any:
    """Convert immutable/NumPy values into canonical JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise P1ForecastError("JSON provenance contains a non-finite scalar")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise P1ForecastError("JSON provenance contains an unsupported value")


def _array_sha256(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise P1ForecastError(f"{name} cannot use object dtype")
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


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            _plain(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise P1ForecastError("forecast artifact is not canonical JSON") from exc
    if len(encoded) > P1_FORECAST_FILE_MAX_BYTES:
        raise P1ForecastError("forecast artifact exceeds the file-size bound")
    return encoded


def _encode_float_array(value: Any, *, mask: Any | None, name: str) -> Any:
    """Encode float arrays with null only at explicitly unmasked cells."""

    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise P1ForecastError(f"{name} must be numeric")
    numeric = np.asarray(array, dtype=np.float64)
    if np.isinf(numeric).any():
        raise P1ForecastError(f"{name} contains infinity")
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.dtype != np.dtype(np.bool_) or mask_array.shape != numeric.shape:
            raise P1ForecastError(f"{name} mask is not strict-bool shape-aligned")
        finite = np.isfinite(numeric)
        if not np.array_equal(finite, mask_array):
            raise P1ForecastError(f"{name} is finite exactly where its mask is true")
    encoded = numeric.tolist()
    # ``tolist`` retains NaN values, which JSON rejects.  Replace them only
    # after the mask check above has established that they are intentional.
    def replace_nan(item: Any) -> Any:
        if isinstance(item, list):
            return [replace_nan(child) for child in item]
        if isinstance(item, float) and np.isnan(item):
            return None
        return item

    return replace_nan(encoded)


def _decode_float_array(value: Any, *, shape: tuple[int, ...], mask: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ForecastError(f"{name} cannot be decoded as float64") from exc
    if array.shape != shape:
        raise P1ForecastError(f"{name} has shape {array.shape}, expected {shape}")
    mask_array = np.asarray(mask)
    if mask_array.dtype != np.dtype(np.bool_) or mask_array.shape != shape:
        raise P1ForecastError(f"{name} mask is not strict-bool shape-aligned")
    if np.isinf(array).any() or not np.array_equal(np.isfinite(array), mask_array):
        raise P1ForecastError(f"{name} has nonfinite/mask inconsistency")
    return np.asarray(array, dtype=np.float64)


def _decode_bool_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise P1ForecastError(f"{name} cannot be decoded") from exc
    if array.shape != shape or array.dtype != np.dtype(np.bool_):
        raise P1ForecastError(f"{name} must be a strict-bool array of shape {shape}")
    return np.asarray(array, dtype=np.bool_)


def _decode_int_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise P1ForecastError(f"{name} cannot be decoded") from exc
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise P1ForecastError(f"{name} must be an integer array of shape {shape}")
    return np.asarray(array, dtype=np.int64)


def _timestamp_strings(timestamps: Any) -> list[str]:
    array = np.asarray(timestamps)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.datetime64):
        raise P1ForecastError("support timestamps must be one-dimensional datetime64")
    try:
        normalized = np.asarray(array, dtype=np.dtype("datetime64[ns]"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ForecastError("support timestamps cannot be represented as ns") from exc
    if np.isnat(normalized).any():
        raise P1ForecastError("support timestamps must not contain NaT")
    ticks = normalized.astype(np.int64)
    if len(ticks) > 1 and np.any(np.diff(ticks) <= 0):
        raise P1ForecastError("support timestamps must be strictly increasing")
    return [str(np.datetime_as_string(value, unit="ns")) for value in normalized]


def _decode_timestamps(value: Any, *, expected_count: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or any(not isinstance(item, str) for item in value)
    ):
        raise P1ForecastError("support_timestamps must be an ordered list of strings")
    try:
        parsed = np.asarray(value, dtype=np.dtype("datetime64[ns]"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ForecastError("support_timestamps contain invalid timestamps") from exc
    if np.isnat(parsed).any():
        raise P1ForecastError("support_timestamps must not contain NaT")
    ticks = parsed.astype(np.int64)
    if len(ticks) > 1 and np.any(np.diff(ticks) <= 0):
        raise P1ForecastError("support_timestamps must be strictly increasing")
    return parsed


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P1ForecastError("expected a mapping")
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ValidationScenarioSpec:
    """One immutable registered scenario/arm execution identity."""

    scenario_id: str
    arm: str
    data_kind: Literal["synthetic", "s3"]
    beta: float
    snr: float | None
    seeds: tuple[int, ...]
    fit_origin: int
    train_start: int
    fit_range: tuple[int, int]
    support_id: str
    support_range: tuple[int, int]
    n_rows: int

    @property
    def split_id(self) -> str:
        return P1_FORECAST_SPLIT_ID

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "arm": self.arm,
            "data_kind": self.data_kind,
            "beta": self.beta,
            "snr": self.snr,
            "seeds": list(self.seeds),
            "fit_origin": self.fit_origin,
            "train_start": self.train_start,
            "fit_range": list(self.fit_range),
            "support_id": self.support_id,
            "support_range": list(self.support_range),
            "n_rows": self.n_rows,
        }


@dataclass(frozen=True)
class P1ForecastContract:
    """Authenticated manifest/registry pair and fixed forecast specifications."""

    manifest: Mapping[str, Any]
    registry: P1ResultRegistry
    manifest_sha256: str
    trial_registry_sha256: str
    comparison_registry_sha256: str
    horizons: tuple[int, ...]
    model_task_keys: tuple[tuple[str, str], ...]
    specs: Mapping[tuple[str, str], ValidationScenarioSpec]
    coverage_thresholds: Mapping[str, float]

    def spec(self, scenario_id: str, arm: str) -> ValidationScenarioSpec:
        try:
            return self.specs[(scenario_id, arm)]
        except KeyError as exc:
            raise P1ForecastError(
                f"scenario/arm is not registered: {scenario_id!r}/{arm!r}"
            ) from exc

    @property
    def validation_arm_keys(self) -> tuple[tuple[str, str, int], ...]:
        """Return every registered scenario/arm/seed execution identity."""

        return registered_validation_arm_keys(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "trial_registry_sha256": self.trial_registry_sha256,
            "comparison_registry_sha256": self.comparison_registry_sha256,
            "horizons": list(self.horizons),
            "model_task_keys": [list(key) for key in self.model_task_keys],
            "specs": [spec.as_dict() for spec in self.specs.values()],
            "coverage_thresholds": dict(self.coverage_thresholds),
        }


def _require_exact_sequence(value: Any, expected: Sequence[Any], *, name: str) -> None:
    if not isinstance(value, (list, tuple)) or tuple(value) != tuple(expected):
        raise P1ForecastError(f"{name} does not match the registered fixed sequence")


def _build_registered_specs(manifest: Mapping[str, Any]) -> Mapping[tuple[str, str], ValidationScenarioSpec]:
    common = manifest.get("common")
    scenarios = manifest.get("scenarios")
    synthetic = manifest.get("synthetic_contract")
    if not isinstance(common, Mapping) or not isinstance(scenarios, Mapping) or not isinstance(synthetic, Mapping):
        raise P1ForecastError("authenticated manifest is missing scenario contracts")
    _require_exact_sequence(common.get("forecast_horizons"), P1_FIXED_HORIZONS, name="forecast_horizons")
    seeds = tuple(_strict_int(value, name="common.seeds") for value in common.get("seeds", ()))
    if seeds != P1_SYNTHETIC_SEEDS:
        raise P1ForecastError("synthetic seed schedule is not exactly 20260830..20260839")
    expected_splits = {
        "fit": (0, 20_000),
        "oof_development": (20_000, 90_000),
        "validation": (90_000, 100_000),
        "outer_test": (100_000, 120_000),
    }
    for scenario_id in ("S0", "S1", "S2"):
        scenario = scenarios.get(scenario_id)
        if not isinstance(scenario, Mapping) or scenario.get("splits") != expected_splits:
            raise P1ForecastError(f"{scenario_id} synthetic split contract is not fixed")
        if scenario.get("outer_test_is_report_only") is not True:
            raise P1ForecastError(f"{scenario_id} outer operation is not report-only")
    expected_levels = {
        "high": {"beta": 0.004, "snr": 4.0},
        "medium": {"beta": 0.001, "snr": 1.0},
        "low": {"beta": 0.00025, "snr": 0.25},
    }
    if scenarios.get("S0", {}).get("beta") != 0.0:
        raise P1ForecastError("S0 beta is not exactly zero")
    if scenarios.get("S1", {}).get("beta") != 0.004:
        raise P1ForecastError("S1 beta is not exactly 0.004")
    if scenarios.get("S2", {}).get("levels") != expected_levels:
        raise P1ForecastError("S2 beta/level contract is not exact")
    if tuple(scenarios.get("S0", {}).get("seeds", ())) != seeds or tuple(scenarios.get("S1", {}).get("seeds", ())) != seeds:
        raise P1ForecastError("S0/S1 seed schedule differs from common seeds")
    if tuple(scenarios.get("S2", {}).get("seeds", ())) != seeds:
        raise P1ForecastError("S2 seed schedule differs from common seeds")
    oof = common.get("oof")
    if not isinstance(oof, Mapping) or oof.get("validation_origin") != P1_SYNTHETIC_ORIGIN:
        raise P1ForecastError("synthetic validation origin is not 90000")
    primary = oof.get("primary_inferential_support")
    if not isinstance(primary, Mapping) or primary.get("support_id") != P1_SYNTHETIC_SUPPORT_ID:
        raise P1ForecastError("synthetic validation support is not registered")
    if primary.get("origin") != P1_SYNTHETIC_ORIGIN or tuple(primary.get("prediction_range", ())) != P1_SYNTHETIC_SUPPORT_RANGE:
        raise P1ForecastError("synthetic validation range/origin is not exact")
    if synthetic.get("n_rows") != P1_SYNTHETIC_ROWS or synthetic.get("burn_in_rows") != 512:
        raise P1ForecastError("synthetic row/burn-in contract is not exact")
    specs: dict[tuple[str, str], ValidationScenarioSpec] = {}
    synthetic_values = {
        ("S0", "zero_signal"): (0.0, None),
        ("S1", "known_high_snr_dgp"): (0.004, 4.0),
        ("S2-high", "high"): (0.004, 4.0),
        ("S2-medium", "medium"): (0.001, 1.0),
        ("S2-low", "low"): (0.00025, 0.25),
    }
    for (scenario_id, arm), (beta, snr) in synthetic_values.items():
        manifest_scenario = "S2" if scenario_id.startswith("S2-") else scenario_id
        scenario = scenarios.get(manifest_scenario)
        if not isinstance(scenario, Mapping):
            raise P1ForecastError(f"missing registered scenario {manifest_scenario}")
        if scenario_id.startswith("S2-"):
            level = scenario_id.split("-", 1)[1]
            if scenario["levels"][level] != {"beta": beta, "snr": snr}:
                raise P1ForecastError(f"{scenario_id} level spec differs from manifest")
        elif scenario.get("beta") != beta:
            raise P1ForecastError(f"{scenario_id} beta differs from manifest")
        specs[(scenario_id, arm)] = ValidationScenarioSpec(
            scenario_id=scenario_id,
            arm=arm,
            data_kind="synthetic",
            beta=beta,
            snr=snr,
            seeds=seeds,
            fit_origin=P1_SYNTHETIC_ORIGIN,
            train_start=0,
            fit_range=P1_SYNTHETIC_FIT_RANGE,
            support_id=P1_SYNTHETIC_SUPPORT_ID,
            support_range=P1_SYNTHETIC_SUPPORT_RANGE,
            n_rows=P1_SYNTHETIC_ROWS,
        )
    s3 = scenarios.get("S3")
    if not isinstance(s3, Mapping) or tuple(s3.get("seeds", ())) != P1_S3_SEEDS:
        raise P1ForecastError("S3 must have the single registered seed 20260830")
    signal = s3.get("signal")
    if not isinstance(signal, Mapping) or signal.get("injection_beta") != 0.0005 or signal.get("control_beta") != 0.0:
        raise P1ForecastError("S3 injection/control beta contract is not exact")
    s3_primary = s3.get("primary_inferential_operation")
    if not isinstance(s3_primary, Mapping):
        raise P1ForecastError("S3 primary validation operation is missing")
    if (
        s3_primary.get("support_id") != P1_S3_SUPPORT_ID
        or s3_primary.get("origin_raw_index") != P1_S3_ORIGIN
        or tuple(s3_primary.get("fit_raw_range", ())) != P1_S3_FIT_RANGE
        or tuple(s3_primary.get("prediction_raw_range", ())) != P1_S3_SUPPORT_RANGE
        or tuple(s3_primary.get("refit_origins", ())) != ()
        or s3_primary.get("outer_test_role") != "report_only"
    ):
        raise P1ForecastError("S3 validation origin/ranges are not exact")
    if tuple(s3.get("fit_raw_range", ())) != P1_S3_FIT_RANGE or tuple(s3.get("validation_raw_range", ())) != P1_S3_SUPPORT_RANGE:
        raise P1ForecastError("S3 fit/validation raw ranges are not exact")
    for arm, beta in (("injected", 0.0005), ("zero_injection_control", 0.0)):
        specs[("S3", arm)] = ValidationScenarioSpec(
            scenario_id="S3",
            arm=arm,
            data_kind="s3",
            beta=beta,
            snr=None,
            seeds=P1_S3_SEEDS,
            fit_origin=P1_S3_ORIGIN,
            train_start=P1_S3_TRAIN_START,
            fit_range=P1_S3_FIT_RANGE,
            support_id=P1_S3_SUPPORT_ID,
            support_range=P1_S3_SUPPORT_RANGE,
            n_rows=P1_S3_ROWS,
        )
    return MappingProxyType(specs)


def authenticate_p1_forecast_contract(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> P1ForecastContract:
    """Authenticate the fixed manifest and exact 56/16 result registries."""

    try:
        manifest = load_fixed_manifest(manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        raise P1ForecastError("could not authenticate the fixed P1 manifest") from exc
    if manifest.get("manifest_sha256") != REGISTERED_MANIFEST_SHA256:
        raise P1ForecastError("manifest digest is not the registered P1 digest")
    if manifest.get("results_observed") is not False:
        raise P1ForecastError("P1 validation requires results_observed=false")
    try:
        registry = load_p1_result_registry(manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        raise P1ForecastError("could not authenticate the fixed P1 result registries") from exc
    if not isinstance(registry, P1ResultRegistry):
        raise P1ForecastError("P1 result registry loader did not return its authenticated type")
    if len(registry.trials) != P1_TRIAL_COUNT or len(registry.comparisons) != P1_PRIMARY_COMPARISON_COUNT:
        raise P1ForecastError("P1 registry must contain exactly 56 trials and 16 comparisons")
    common = manifest.get("common")
    if not isinstance(common, Mapping):
        raise P1ForecastError("manifest common contract is missing")
    trial_contract = common.get("trial_registry")
    comparison_contract = common.get("primary_comparison_registry")
    if not isinstance(trial_contract, Mapping) or not isinstance(comparison_contract, Mapping):
        raise P1ForecastError("manifest registry contracts are missing")
    if (
        trial_contract.get("sha256") != registry.trial_registry_sha256
        or comparison_contract.get("sha256") != registry.comparison_registry_sha256
        or trial_contract.get("record_count") != P1_TRIAL_COUNT
        or comparison_contract.get("family_size") != P1_PRIMARY_COMPARISON_COUNT
    ):
        raise P1ForecastError("authenticated registry hashes/counts do not echo the manifest")
    specs = _build_registered_specs(manifest)
    # Every scenario/arm must have both cost-mode rows for every fixed model.
    mapper_by_model = {
        "zero_return": "fixed_baseline",
        "persistence_last_observed": "fixed_baseline",
        "ridge": "ridge_h4",
        "logistic": "none_binary_diagnostic",
    }
    for scenario_id, arm in P1_SCENARIO_ARMS:
        matching = [row for row in registry.trials if row.get("scenario_id") == scenario_id and row.get("arm") == arm]
        if len(matching) != len(P1_REQUIRED_COST_MODES) * 4:
            raise P1ForecastError(f"registry rows are incomplete for {scenario_id}/{arm}")
        spec = specs[(scenario_id, arm)]
        registry_prefix = (
            "S3-injected" if (scenario_id, arm) == ("S3", "injected")
            else "S3-control" if (scenario_id, arm) == ("S3", "zero_injection_control")
            else scenario_id
        )
        expected_rows = {
            (
                f"{registry_prefix}__{model_id}__{cost_mode}",
                scenario_id,
                arm,
                model_id,
                cost_mode,
                True,
                mapper_by_model[model_id],
                len(spec.seeds),
            )
            for model_id in mapper_by_model
            for cost_mode in P1_REQUIRED_COST_MODES
        }
        actual_rows = {
            tuple(row.get(field) for field in (
                "trial_id", "scenario_id", "arm", "model_id", "cost_mode",
                "primary", "action_mapper", "seed_count",
            ))
            for row in matching
        }
        if actual_rows != expected_rows:
            raise P1ForecastError(f"registry rows differ from the fixed grid for {scenario_id}/{arm}")
    if tuple(row.get("comparison_id") for row in registry.comparisons) != P1_PRIMARY_COMPARISON_IDS:
        raise P1ForecastError("primary comparison IDs/order differ from the fixed registry")
    thresholds = common.get("gates", {}).get("coverage_thresholds") if isinstance(common.get("gates"), Mapping) else None
    if not isinstance(thresholds, Mapping):
        raise P1ForecastError("fixed coverage thresholds are missing")
    synthetic_threshold = _strict_float(
        thresholds.get("synthetic_eligible_origin_fraction_min"),
        name="synthetic eligible threshold",
    )
    s3_threshold = _strict_float(
        thresholds.get("s3_eligible_origin_fraction_min"),
        name="S3 eligible threshold",
    )
    label_threshold = _strict_float(
        thresholds.get("label_complete_fraction_min"),
        name="label threshold",
    )
    finite_threshold = _strict_float(
        thresholds.get("finite_oof_prediction_fraction_min"),
        name="finite prediction threshold",
    )
    threshold_values = MappingProxyType(
        {
            "synthetic_eligible_origin_fraction_min": synthetic_threshold,
            "s3_eligible_origin_fraction_min": s3_threshold,
            "label_complete_fraction_min": label_threshold,
            "finite_oof_prediction_fraction_min": finite_threshold,
            "scored_action_fraction_min": _strict_float(
                thresholds.get("scored_action_fraction_min"),
                name="scored action threshold",
            ),
        }
    )
    return P1ForecastContract(
        manifest=manifest,
        registry=registry,
        manifest_sha256=REGISTERED_MANIFEST_SHA256,
        trial_registry_sha256=registry.trial_registry_sha256,
        comparison_registry_sha256=registry.comparison_registry_sha256,
        horizons=P1_FIXED_HORIZONS,
        model_task_keys=P1_ALLOWED_MODEL_TASK_KEYS,
        specs=specs,
        coverage_thresholds=threshold_values,
    )


load_p1_forecast_contract = authenticate_p1_forecast_contract
load_authenticated_p1_forecast_contract = authenticate_p1_forecast_contract


def registered_validation_arm_keys(
    contract: P1ForecastContract | None = None,
) -> tuple[tuple[str, str, int], ...]:
    """Return the exact 52 registered scenario/arm/seed identities."""

    selected = contract if contract is not None else authenticate_p1_forecast_contract()
    if not isinstance(selected, P1ForecastContract):
        raise P1ForecastError("validation arm enumeration requires the authenticated contract")
    keys = tuple(
        (scenario_id, arm, seed)
        for scenario_id, arm in P1_SCENARIO_ARMS
        for seed in selected.spec(scenario_id, arm).seeds
    )
    if len(keys) != P1_VALIDATION_ARM_COUNT or len(set(keys)) != P1_VALIDATION_ARM_COUNT:
        raise P1ForecastError("fixed validation arm registry must contain exactly 52 identities")
    return keys


def _dataset_seed(dataset: Any, spec: ValidationScenarioSpec) -> int:
    """Resolve one registered seed; every artifact is one seed, never a seed set."""

    value = getattr(dataset, "seed", None)
    seed = _strict_int(value, name="dataset.seed")
    if seed not in spec.seeds:
        raise P1ForecastError(
            f"dataset seed {seed} is not registered for {spec.scenario_id}/{spec.arm}"
        )
    return seed


def _expected_metadata(
    contract: P1ForecastContract,
    spec: ValidationScenarioSpec,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is None:
        if len(spec.seeds) != 1:
            raise P1ForecastError("production expected metadata requires the selected seed")
        seed = spec.seeds[0]
    seed = _strict_int(seed, name="expected_metadata.seed")
    if seed not in spec.seeds:
        raise P1ForecastError("expected metadata seed is not registered for the scenario/arm")
    return {
        "manifest_sha256": contract.manifest_sha256,
        "trial_registry_sha256": contract.trial_registry_sha256,
        "comparison_registry_sha256": contract.comparison_registry_sha256,
        "prereg_results_observed": False,
        "validation_results_observed": True,
        "outer_results_observed": False,
        "scenario_id": spec.scenario_id,
        "arm": spec.arm,
        "seed": seed,
        "split_id": spec.split_id,
        "support_id": spec.support_id,
        "support_range": list(spec.support_range),
        "fit_origin": spec.fit_origin,
        "train_start": spec.train_start,
    }


def expected_metadata_for_arm(
    contract: P1ForecastContract,
    scenario_id: str,
    arm: str,
    seed: int,
) -> Mapping[str, Any]:
    """Return the fixed external metadata required for one production arm load."""

    if not isinstance(contract, P1ForecastContract):
        raise P1ForecastError("expected metadata requires the authenticated contract")
    spec = contract.spec(scenario_id, arm)
    return MappingProxyType(_expected_metadata(contract, spec, seed=seed))


@dataclass(frozen=True)
class ForecastCoverageSummary:
    """Fixed-threshold coverage evidence for one forecast fit key."""

    horizon: int
    model_id: str
    task: str
    potential_origins: int
    context_complete: int
    label_complete: int
    eligible_origins: int
    finite_predictions: int
    eligible_fraction: float | None
    context_fraction: float | None
    label_complete_fraction: float | None
    finite_prediction_fraction: float | None
    thresholds: Mapping[str, float]
    status: Literal["passed", "failed", "N/A"]

    @property
    def promotion_allowed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "model_id": self.model_id,
            "task": self.task,
            "potential_origins": self.potential_origins,
            "context_complete": self.context_complete,
            "label_complete": self.label_complete,
            "eligible_origins": self.eligible_origins,
            "finite_predictions": self.finite_predictions,
            "eligible_fraction": self.eligible_fraction,
            "context_fraction": self.context_fraction,
            "label_complete_fraction": self.label_complete_fraction,
            "finite_prediction_fraction": self.finite_prediction_fraction,
            "thresholds": dict(self.thresholds),
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
        }


def _coverage_for_fit(
    dataset: Any,
    fit: Any,
    *,
    spec: ValidationScenarioSpec,
    horizon: int,
    model_id: str,
    task: str,
    thresholds: Mapping[str, float],
) -> ForecastCoverageSummary:
    start, end = spec.support_range
    features = np.asarray(dataset.features)
    n_rows = len(features)
    if n_rows != spec.n_rows:
        raise P1ForecastError("dataset row count does not match its registered scenario")
    rows = np.arange(n_rows, dtype=np.int64)
    target_end = np.asarray(dataset.target_end)
    target_mask = np.asarray(dataset.target_mask)
    context_mask = np.asarray(dataset.context_mask)
    horizon_column = spec.support_range  # overwritten below to keep errors readable
    try:
        horizon_column = _runner_module().FORECAST_HORIZONS.index(horizon)
    except (AttributeError, ValueError) as exc:
        raise P1ForecastError("coverage horizon is not one of the fixed horizons") from exc
    potential = (
        (rows >= start)
        & (rows < end)
        & (rows >= P1_CONTEXT_BARS - 1)
        & (target_end[:, horizon_column] <= end)
    )
    context = context_mask[potential]
    labels = target_mask[potential, horizon_column]
    eligible = context & labels
    try:
        fit_prediction_mask = np.asarray(fit.prediction_mask)
        fit_predictions = np.asarray(fit.predictions)
    except AttributeError as exc:
        raise P1ForecastError("fit does not expose prediction arrays") from exc
    if fit_prediction_mask.shape != (n_rows,) or fit_predictions.shape != (n_rows,):
        raise P1ForecastError("fit prediction arrays are not full-grid aligned")
    if fit_prediction_mask.dtype != np.dtype(np.bool_) or fit_predictions.dtype != np.dtype(np.float64):
        raise P1ForecastError("fit prediction arrays have non-canonical dtype")
    finite = fit_prediction_mask[potential] & np.isfinite(fit_predictions[potential])
    potential_count = int(np.count_nonzero(potential))
    context_count = int(np.count_nonzero(context))
    label_count = int(np.count_nonzero(labels))
    eligible_count = int(np.count_nonzero(eligible))
    finite_count = int(np.count_nonzero(finite & eligible))
    eligible_fraction = eligible_count / potential_count if potential_count else None
    context_fraction = context_count / potential_count if potential_count else None
    label_fraction = label_count / potential_count if potential_count else None
    finite_fraction = finite_count / eligible_count if eligible_count else None
    eligible_threshold = thresholds[
        "synthetic_eligible_origin_fraction_min"
        if spec.data_kind == "synthetic"
        else "s3_eligible_origin_fraction_min"
    ]
    status: Literal["passed", "failed", "N/A"] = "passed"
    if potential_count == 0 or eligible_count == 0 or fit.status == "N/A":
        status = "N/A"
    elif (
        eligible_fraction is None
        or eligible_fraction < eligible_threshold
        or label_fraction is None
        or label_fraction < thresholds["label_complete_fraction_min"]
        or finite_fraction is None
        or finite_fraction < thresholds["finite_oof_prediction_fraction_min"]
    ):
        status = "failed"
    return ForecastCoverageSummary(
        horizon=horizon,
        model_id=model_id,
        task=task,
        potential_origins=potential_count,
        context_complete=context_count,
        label_complete=label_count,
        eligible_origins=eligible_count,
        finite_predictions=finite_count,
        eligible_fraction=eligible_fraction,
        context_fraction=context_fraction,
        label_complete_fraction=label_fraction,
        finite_prediction_fraction=finite_fraction,
        thresholds=MappingProxyType(
            {
                "eligible_origin_fraction_min": eligible_threshold,
                "label_complete_fraction_min": thresholds["label_complete_fraction_min"],
                "finite_oof_prediction_fraction_min": thresholds["finite_oof_prediction_fraction_min"],
            }
        ),
        status=status,
    )


def _support_slice(values: Any, spec: ValidationScenarioSpec, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0 or len(array) != spec.n_rows:
        raise P1ForecastError(f"{name} is not full-grid aligned to the registered body")
    start, end = spec.support_range
    return np.array(array[start:end], copy=True)


def _scenario_provenance(spec: ValidationScenarioSpec, dataset: Any) -> Mapping[str, Any]:
    seed = _dataset_seed(dataset, spec)
    returns = np.asarray(dataset.returns)
    features = np.asarray(dataset.features)
    timestamps = np.asarray(dataset.timestamps)
    availability = dataset.availability
    source_hashes: dict[str, str] = {
        "timestamps": _array_sha256(timestamps, name="timestamps"),
        "features": _array_sha256(features, name="features"),
        "returns": _array_sha256(returns, name="returns"),
    }
    for name in ("spot_bar_observed", "funding_rate_available", "mark_close_available"):
        source_hashes[f"availability.{name}"] = _array_sha256(availability[name], name=f"availability.{name}")
    result: dict[str, Any] = {
        "scenario_id": spec.scenario_id,
        "arm": spec.arm,
        "data_kind": spec.data_kind,
        "seed": seed,
        "beta": spec.beta,
        "snr": spec.snr,
        "n_rows": spec.n_rows,
        "source_array_sha256": source_hashes,
    }
    if spec.data_kind == "s3":
        source = getattr(dataset, "source", None)
        body_sha = getattr(dataset, "source_body_sha256", None)
        if not isinstance(body_sha, str) or not _SHA256_RE.fullmatch(body_sha):
            raise P1ForecastError("S3 dataset is missing its authenticated source body digest")
        result["source_body_sha256"] = body_sha
        if source is not None and isinstance(getattr(source, "runtime", None), Mapping):
            result["runtime"] = _plain(source.runtime)
    return MappingProxyType(result)


def _arrays_exact(actual: Any, expected: Any, *, name: str) -> None:
    """Require a runner-owned array to match byte-level shape/dtype/content."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape or actual_array.dtype != expected_array.dtype:
        raise P1ForecastError(f"{name} does not match the canonical runner source")
    equal_nan = np.issubdtype(expected_array.dtype, np.inexact)
    if not np.array_equal(actual_array, expected_array, equal_nan=equal_nan):
        raise P1ForecastError(f"{name} does not match the canonical runner source")


def _validate_registered_dataset(spec: ValidationScenarioSpec, dataset: Any) -> int:
    """Bind a producer to the exact recovery-runner dataset and registered arm."""

    runner = _runner_module()
    if spec.data_kind == "synthetic":
        if type(dataset) is not runner.SyntheticDataset:
            raise P1ForecastError("synthetic production forecasts require an exact SyntheticDataset")
        seed = _dataset_seed(dataset, spec)
        if dataset.beta != spec.beta:
            raise P1ForecastError("synthetic dataset beta differs from the registered arm")
        try:
            expected = runner.build_synthetic_dataset(seed, spec.beta)
        except Exception as exc:
            raise P1ForecastError("could not regenerate the registered synthetic source") from exc
        try:
            for name in (
                "timestamps",
                "features",
                "returns",
                "targets",
                "target_end",
                "target_mask",
                "binary_labels",
                "context_mask",
            ):
                _arrays_exact(getattr(dataset, name), getattr(expected, name), name=f"synthetic {name}")
            if type(dataset.base) is not type(expected.base) or dataset.base.seed != expected.base.seed:
                raise P1ForecastError("synthetic dataset base does not match the registered seed")
            for name in ("z_raw", "xi", "noise_features", "epsilon", "gap_starts"):
                _arrays_exact(getattr(dataset.base, name), getattr(expected.base, name), name=f"synthetic base {name}")
            for name in ("spot_bar_observed", "funding_rate_available", "mark_close_available"):
                _arrays_exact(
                    dataset.availability[name],
                    expected.availability[name],
                    name=f"synthetic availability.{name}",
                )
        except P1ForecastError:
            raise
        except Exception as exc:
            raise P1ForecastError("synthetic dataset source fields are malformed") from exc
        try:
            runner._ensure_dataset(dataset)
        except Exception as exc:
            raise P1ForecastError("synthetic dataset failed the runner validation boundary") from exc
        return seed

    if type(dataset) is not runner.S3ArmDataset:
        raise P1ForecastError("S3 production forecasts require an exact S3ArmDataset")
    seed = _dataset_seed(dataset, spec)
    if dataset.arm != spec.arm or dataset.beta != spec.beta:
        raise P1ForecastError("S3 dataset arm/beta differs from the registered scenario")
    try:
        runner._require_production_s3_arm(dataset)
        runner._ensure_dataset(dataset)
    except Exception as exc:
        raise P1ForecastError("S3 dataset is not bound to the authenticated runner source") from exc
    return seed


def _future_evidence(
    dataset: Any,
    fit: Any,
    *,
    spec: ValidationScenarioSpec,
    horizon: int,
) -> Mapping[str, Any]:
    """Create a causal future-perturbation evidence record.

    Synthetic data can use the recovery runner's full refit probe.  The S3 arm
    is sealed to its authenticated source body, so its evidence records the
    exact unchanged fitted-prefix/prediction-prefix digests and body digest;
    a second mutable arm body is intentionally not manufactured here.
    """

    origin = spec.fit_origin
    perturb_start = spec.support_range[1]
    try:
        if spec.data_kind == "synthetic":
            evidence = _runner_module().assert_future_perturbation_invariance(
                dataset,
                "ridge",
                origin,
                horizon,
                prediction_range=spec.support_range,
                perturb_start=perturb_start,
                train_start=spec.train_start,
            )
            if not isinstance(evidence, Mapping) or evidence.get("status") != "passed":
                raise P1ForecastError("future perturbation probe did not pass")
            result = dict(evidence)
            result["method"] = "causal_refit_probe"
        else:
            result = {
                "status": "passed",
                "method": "sealed_prefix_digest_probe",
                "origin": origin,
                "horizon": horizon,
                "perturb_start": perturb_start,
                "earlier_prediction_count": int(np.count_nonzero(np.asarray(fit.prediction_mask)[:perturb_start])),
            }
        train_mask = np.asarray(fit.train_mask)
        prediction_mask = np.asarray(fit.prediction_mask)
        predictions = np.asarray(fit.predictions)
        result.update(
            {
                "fitted_prefix_mask_sha256": _array_sha256(train_mask[:origin], name="fitted_prefix_mask"),
                "earlier_prediction_mask_sha256": _array_sha256(prediction_mask[:perturb_start], name="earlier_prediction_mask"),
                "earlier_prediction_sha256": _array_sha256(
                    np.where(prediction_mask[:perturb_start], predictions[:perturb_start], 0.0),
                    name="earlier_predictions",
                ),
            }
        )
        if spec.data_kind == "s3":
            result["source_body_sha256"] = getattr(dataset, "source_body_sha256")
        return MappingProxyType(_plain(result))
    except (AttributeError, TypeError, ValueError, P1ForecastError) as exc:
        # A failed probe is explicit evidence and blocks promotion when loaded;
        # it is not silently omitted from the persisted artifact.
        return MappingProxyType(
            {
                "status": "N/A",
                "method": "probe_failed",
                "origin": origin,
                "horizon": horizon,
                "perturb_start": perturb_start,
                "reason": str(exc),
            }
        )


def _fit_record(
    fit: Any,
    *,
    dataset: Any,
    spec: ValidationScenarioSpec,
    horizon: int,
    model_id: str,
    task: str,
    expected_train_mask: np.ndarray | None = None,
    expected_prediction_mask: np.ndarray | None = None,
    expected_score_eligible_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    runner = _runner_module()
    if type(fit) is not runner.ModelFit:
        raise P1ForecastError("production forecast fits require an exact recovery-runner ModelFit")
    if fit.model_id != model_id or fit.task != task or fit.horizon != horizon or fit.origin != spec.fit_origin:
        raise P1ForecastError("fit identity does not match the registered key")
    if fit.train_start != spec.train_start:
        raise P1ForecastError("fit train_start differs from the registered boundary")
    train_mask = np.asarray(fit.train_mask)
    eligible_mask = np.asarray(fit.eligible_mask)
    prediction_mask = np.asarray(fit.prediction_mask)
    predictions = np.asarray(fit.predictions)
    if (
        train_mask.shape != (spec.n_rows,)
        or eligible_mask.shape != (spec.n_rows,)
        or prediction_mask.shape != (spec.n_rows,)
        or predictions.shape != (spec.n_rows,)
    ):
        raise P1ForecastError("fit arrays must retain the complete body row grid")
    for name, mask in (("train_mask", train_mask), ("eligible_mask", eligible_mask), ("prediction_mask", prediction_mask)):
        if mask.dtype != np.dtype(np.bool_):
            raise P1ForecastError(f"fit {name} must use strict bool dtype")
    if expected_train_mask is None:
        try:
            expected_train_mask = runner.train_mask_for_origin(
                dataset,
                spec.fit_origin,
                horizon,
                train_start=spec.train_start,
            )
        except Exception as exc:
            raise P1ForecastError("could not derive the registered fit train mask") from exc
    if expected_prediction_mask is None:
        try:
            expected_prediction_mask = runner.prediction_mask_for_range(
                dataset,
                horizon,
                start=spec.support_range[0],
                end=spec.support_range[1],
            )
        except Exception as exc:
            raise P1ForecastError("could not derive the registered prediction mask") from exc
    if expected_score_eligible_mask is None:
        try:
            expected_score_eligible_mask = runner.score_eligible_mask_for_range(
                dataset,
                horizon,
                start=spec.support_range[0],
                end=spec.support_range[1],
            )
        except Exception as exc:
            raise P1ForecastError("could not derive the registered score mask") from exc
    if not np.array_equal(train_mask, expected_train_mask):
        raise P1ForecastError("fit train_mask differs from the registered runner chronology")
    if not np.array_equal(eligible_mask, expected_score_eligible_mask):
        raise P1ForecastError("fit eligible_mask differs from the registered score mask")
    if fit.status == "ok" and not np.array_equal(prediction_mask, expected_prediction_mask):
        raise P1ForecastError("fit prediction_mask differs from the registered prediction range")
    if predictions.dtype != np.dtype(np.float64) or not np.array_equal(np.isfinite(predictions), prediction_mask):
        raise P1ForecastError("fit predictions must be float64 and finite exactly at prediction_mask")
    if fit.status not in {"ok", "N/A"}:
        raise P1ForecastError("fit status must be ok or N/A")
    if fit.status == "N/A" and np.any(prediction_mask):
        raise P1ForecastError("N/A fit cannot carry predictions")
    try:
        canonical_fit = runner.fit_model_at_origin(
            dataset,
            model_id,
            spec.fit_origin,
            horizon,
            task=task,
            prediction_range=spec.support_range,
            train_start=spec.train_start,
        )
    except Exception as exc:
        raise P1ForecastError("could not reproduce the registered runner fit") from exc
    if fit.status != canonical_fit.status or fit.reason != canonical_fit.reason:
        raise P1ForecastError("fit status/reason differs from the registered runner fit")
    for name in ("train_mask", "eligible_mask", "prediction_mask", "predictions"):
        _arrays_exact(getattr(fit, name), getattr(canonical_fit, name), name=f"fit {name}")
    support_train = _support_slice(train_mask, spec, name="train_mask")
    support_eligible = _support_slice(eligible_mask, spec, name="eligible_mask")
    support_prediction = _support_slice(prediction_mask, spec, name="prediction_mask")
    support_predictions = _support_slice(predictions, spec, name="predictions")
    return {
        "horizon": horizon,
        "model_id": model_id,
        "task": task,
        "status": str(fit.status),
        "reason": fit.reason,
        "train_count": int(np.count_nonzero(train_mask)),
        "train_mask": support_train.tolist(),
        "eligible_mask": support_eligible.tolist(),
        "prediction_mask": support_prediction.tolist(),
        "predictions": _encode_float_array(
            support_predictions,
            mask=support_prediction,
            name="predictions",
        ),
    }


def build_p1_forecast_artifact(
    contract: P1ForecastContract,
    spec: ValidationScenarioSpec,
    dataset: Any,
    fits: Mapping[tuple[int, str, str], Any],
    *,
    future_perturbation_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical forecast artifact from a complete fit grid."""

    if not isinstance(contract, P1ForecastContract):
        raise P1ForecastError("artifact construction requires the authenticated contract")
    if not isinstance(spec, ValidationScenarioSpec) or contract.spec(spec.scenario_id, spec.arm) != spec:
        raise P1ForecastError("artifact scenario/arm is not from the authenticated contract")
    if spec.n_rows <= 0 or spec.n_rows > P1_FORECAST_FILE_MAX_ROWS:
        raise P1ForecastError("registered body row count is outside the artifact bound")
    if future_perturbation_evidence is not None:
        raise P1ForecastError(
            "production future perturbation evidence must be generated internally"
        )
    seed = _validate_registered_dataset(spec, dataset)
    timestamps = np.asarray(dataset.timestamps)
    if len(timestamps) != spec.n_rows:
        raise P1ForecastError("dataset timestamps are not aligned to the registered body")
    support_timestamps = _timestamp_strings(timestamps[spec.support_range[0] : spec.support_range[1]])
    support_returns = _support_slice(dataset.returns, spec, name="returns")
    support_features = _support_slice(dataset.features, spec, name="features")
    del support_features  # features are bound by the body provenance, not duplicated in the forecast file
    target_mask_full = np.asarray(dataset.target_mask)
    target_values_full = np.asarray(dataset.targets)
    target_end_full = np.asarray(dataset.target_end)
    labels_full = np.asarray(dataset.binary_labels)
    context_full = np.asarray(dataset.context_mask)
    if target_values_full.shape != (spec.n_rows, len(P1_FIXED_HORIZONS)):
        raise P1ForecastError("dataset targets do not cover all fixed horizons")
    if target_mask_full.dtype != np.dtype(np.bool_) or target_mask_full.shape != target_values_full.shape:
        raise P1ForecastError("dataset target mask is not canonical")
    if target_end_full.dtype != np.dtype(np.int64) or target_end_full.shape != target_values_full.shape:
        raise P1ForecastError("dataset target_end is not canonical")
    if labels_full.dtype != np.dtype(np.int8) or labels_full.shape != target_values_full.shape:
        raise P1ForecastError("dataset binary labels are not canonical")
    if context_full.dtype != np.dtype(np.bool_) or context_full.shape != (spec.n_rows,):
        raise P1ForecastError("dataset context mask is not canonical")
    support_mask = _support_slice(target_mask_full, spec, name="target_mask")
    support_targets = _support_slice(target_values_full, spec, name="targets")
    support_end = _support_slice(target_end_full, spec, name="target_end")
    support_labels = _support_slice(labels_full, spec, name="binary_labels")
    support_context = _support_slice(context_full, spec, name="context_mask")
    try:
        origin_full = _runner_module().inference_mask_for_range(
            dataset,
            4,
            start=spec.support_range[0],
            end=spec.support_range[1],
        )
        score_full = _runner_module().score_eligible_mask_for_range(
            dataset,
            4,
            start=spec.support_range[0],
            end=spec.support_range[1],
        )
    except Exception as exc:
        raise P1ForecastError("could not derive the registered causal/score support masks") from exc
    support_origin = _support_slice(origin_full, spec, name="origin_mask")
    support_score = _support_slice(score_full, spec, name="score_eligible_mask")
    support_spot = _support_slice(
        dataset.availability["spot_bar_observed"],
        spec,
        name="spot_bar_observed",
    )
    if not np.array_equal(support_score, support_origin & support_mask[:, 1]):
        raise P1ForecastError("h4 score eligibility is not origin mask AND target mask")
    if not np.array_equal(support_origin, support_context & (support_end[:, 1] <= spec.support_range[1])):
        raise P1ForecastError("h4 origin mask is not causal context plus split-tail boundary")
    if not np.array_equal(np.isfinite(support_targets), support_mask):
        raise P1ForecastError("target values are finite exactly where target_mask is true")
    if np.any((support_labels < -1) | (support_labels > 1)):
        raise P1ForecastError("binary labels contain an out-of-range value")
    expected_keys = {(horizon, model_id, task) for horizon in contract.horizons for model_id, task in contract.model_task_keys}
    if set(fits) != expected_keys:
        missing = sorted(expected_keys - set(fits))
        extra = sorted(set(fits) - expected_keys)
        raise P1ForecastError(f"forecast fit grid is not complete (missing={missing}, extra={extra})")
    expected_masks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    runner = _runner_module()
    for horizon in contract.horizons:
        try:
            expected_masks[horizon] = (
                runner.train_mask_for_origin(
                    dataset,
                    spec.fit_origin,
                    horizon,
                    train_start=spec.train_start,
                ),
                runner.prediction_mask_for_range(
                    dataset,
                    horizon,
                    start=spec.support_range[0],
                    end=spec.support_range[1],
                ),
                runner.score_eligible_mask_for_range(
                    dataset,
                    horizon,
                    start=spec.support_range[0],
                    end=spec.support_range[1],
                ),
            )
        except Exception as exc:
            raise P1ForecastError("could not derive the registered runner fit boundaries") from exc
    records: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for horizon in contract.horizons:
        for model_id, task in contract.model_task_keys:
            fit = fits[(horizon, model_id, task)]
            record = _fit_record(
                fit,
                dataset=dataset,
                spec=spec,
                horizon=horizon,
                model_id=model_id,
                task=task,
                expected_train_mask=expected_masks[horizon][0],
                expected_prediction_mask=expected_masks[horizon][1],
                expected_score_eligible_mask=expected_masks[horizon][2],
            )
            records.append(record)
            summary = _coverage_for_fit(
                dataset,
                fit,
                spec=spec,
                horizon=horizon,
                model_id=model_id,
                task=task,
                thresholds=contract.coverage_thresholds,
            )
            coverage[_fit_key(horizon, model_id, task)] = summary.as_dict()
    evidence = dict(
        _future_evidence(
            dataset,
            fits[(4, "ridge", "continuous")],
            spec=spec,
            horizon=4,
        )
    )
    if evidence.get("status") not in {"passed", "N/A"}:
        raise P1ForecastError("future perturbation evidence has an invalid status")
    provenance = _scenario_provenance(spec, dataset)
    body_provenance: dict[str, Any] = {
        "data_kind": spec.data_kind,
        "body_rows": spec.n_rows,
        "support_range": list(spec.support_range),
        "source_array_sha256": provenance["source_array_sha256"],
    }
    if "source_body_sha256" in provenance:
        body_provenance["source_body_sha256"] = provenance["source_body_sha256"]
        if "runtime" in provenance:
            body_provenance["runtime"] = provenance["runtime"]
    header = {
        "artifact_type": "p1_validation_forecast",
        "schema_id": P1_FORECAST_SCHEMA_ID,
        "schema_version": P1_FORECAST_FILE_VERSION,
        "scenario_id": spec.scenario_id,
        "arm": spec.arm,
        "seed": seed,
        "split_id": spec.split_id,
        "support_id": spec.support_id,
        "support_range": list(spec.support_range),
        "fit_origin": spec.fit_origin,
        "train_start": spec.train_start,
        "fit_range": list(spec.fit_range),
        "forecast_horizons": list(contract.horizons),
        "model_task_keys": [list(key) for key in contract.model_task_keys],
        "outer_report_only": True,
        "outer_test_executed": False,
        "prereg_results_observed": False,
        "validation_results_observed": True,
        "outer_results_observed": False,
        "scenario_provenance": provenance,
        "body_provenance": body_provenance,
        "future_perturbation_evidence": evidence,
    }
    artifact = {
        "format": P1_FORECAST_FILE_FORMAT,
        "format_version": P1_FORECAST_FILE_VERSION,
        "header": header,
        "manifest_sha256": contract.manifest_sha256,
        "trial_registry_sha256": contract.trial_registry_sha256,
        "comparison_registry_sha256": contract.comparison_registry_sha256,
        "prereg_results_observed": False,
        "validation_results_observed": True,
        "outer_results_observed": False,
        "support_timestamps": support_timestamps,
        "realized_returns": _encode_float_array(
            support_returns,
            mask=np.isfinite(support_returns),
            name="realized_returns",
        ),
        "targets": _encode_float_array(support_targets, mask=support_mask, name="targets"),
        "target_end": support_end.tolist(),
        "target_mask": support_mask.tolist(),
        "binary_labels": support_labels.tolist(),
        "context_mask": support_context.tolist(),
        "origin_mask": support_origin.tolist(),
        "score_eligible_mask": support_score.tolist(),
        "spot_bar_observed": support_spot.tolist(),
        "mask_hashes": {
            "context_mask": _array_sha256(support_context, name="context_mask"),
            "origin_mask": _array_sha256(support_origin, name="origin_mask"),
            "score_eligible_mask": _array_sha256(
                support_score,
                name="score_eligible_mask",
            ),
            "target_mask": _array_sha256(support_mask, name="target_mask"),
            "spot_bar_observed": _array_sha256(support_spot, name="spot_bar_observed"),
        },
        "fits": records,
        "coverage": coverage,
    }
    _validate_forecast_payload(
        artifact,
        expected_metadata=_expected_metadata(contract, spec, seed=seed),
        require_production=True,
    )
    return artifact


def _fit_key(horizon: int, model_id: str, task: str) -> str:
    return f"h{horizon}::{model_id}::{task}"


_TOP_LEVEL_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "header",
        "manifest_sha256",
        "trial_registry_sha256",
        "comparison_registry_sha256",
        "prereg_results_observed",
        "validation_results_observed",
        "outer_results_observed",
        "support_timestamps",
        "realized_returns",
        "targets",
        "target_end",
        "target_mask",
        "binary_labels",
        "context_mask",
        "origin_mask",
        "score_eligible_mask",
        "spot_bar_observed",
        "mask_hashes",
        "fits",
        "coverage",
    }
)
_HEADER_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_id",
        "schema_version",
        "scenario_id",
        "arm",
        "seed",
        "split_id",
        "support_id",
        "support_range",
        "fit_origin",
        "train_start",
        "fit_range",
        "forecast_horizons",
        "model_task_keys",
        "outer_report_only",
        "outer_test_executed",
        "prereg_results_observed",
        "validation_results_observed",
        "outer_results_observed",
        "scenario_provenance",
        "body_provenance",
        "future_perturbation_evidence",
    }
)
_FIT_FIELDS = (
    "horizon",
    "model_id",
    "task",
    "status",
    "reason",
    "train_count",
    "train_mask",
    "eligible_mask",
    "prediction_mask",
    "predictions",
)

_SELF_BINDING_KEYS = frozenset(
    {
        "file_sha256",
        "artifact_file_sha256",
        "forecast_file_sha256",
        "output_file_sha256",
        "artifact_sha256",
    }
)
_MAX_PROVENANCE_DEPTH = 32
_MAX_PROVENANCE_NODES = 1_000_000


def _reject_self_binding_keys(value: Any, *, path: str = "artifact") -> None:
    stack: list[tuple[Any, str, int, bool]] = [(value, path, 0, True)]
    active: set[int] = set()
    nodes = 0
    while stack:
        current, current_path, depth, entering = stack.pop()
        if not entering:
            active.discard(id(current))
            continue
        nodes += 1
        if nodes > _MAX_PROVENANCE_NODES:
            raise P1ForecastError("forecast artifact provenance is too deep or large")
        if depth > _MAX_PROVENANCE_DEPTH:
            raise P1ForecastError("forecast artifact provenance is too deeply nested")
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                raise P1ForecastError("forecast artifact provenance contains a cycle")
            active.add(identity)
            stack.append((current, current_path, depth, False))
            for key, item in current.items():
                if key in _SELF_BINDING_KEYS:
                    raise P1ForecastError(f"{current_path}.{key} is an output self-binding field")
                if isinstance(item, (Mapping, list, tuple)):
                    stack.append((item, f"{current_path}.{key}", depth + 1, True))
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in active:
                raise P1ForecastError("forecast artifact provenance contains a cycle")
            active.add(identity)
            stack.append((current, current_path, depth, False))
            for index, item in enumerate(current):
                if isinstance(item, (Mapping, list, tuple)):
                    stack.append((item, f"{current_path}[{index}]", depth + 1, True))


def _validate_identity(
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None,
    require_production: bool,
) -> ValidationScenarioSpec | None:
    _reject_self_binding_keys(artifact)
    if not isinstance(artifact, Mapping) or set(artifact) != _TOP_LEVEL_FIELDS:
        raise P1ForecastError("forecast artifact top-level fields are not exact")
    if artifact.get("format") != P1_FORECAST_FILE_FORMAT or artifact.get("format_version") != P1_FORECAST_FILE_VERSION:
        raise P1ForecastError("forecast artifact format/version is unsupported")
    for name in ("manifest_sha256", "trial_registry_sha256", "comparison_registry_sha256"):
        _strict_sha256(artifact.get(name), name=name)
    header = artifact.get("header")
    if not isinstance(header, Mapping) or set(header) != _HEADER_FIELDS:
        raise P1ForecastError("forecast artifact header fields are not exact")
    if (
        header.get("artifact_type") != "p1_validation_forecast"
        or header.get("schema_id") != P1_FORECAST_SCHEMA_ID
        or header.get("schema_version") != P1_FORECAST_FILE_VERSION
    ):
        raise P1ForecastError("forecast artifact schema identity is unsupported")
    if (
        header.get("outer_report_only") is not True
        or header.get("outer_test_executed") is not False
        or artifact.get("prereg_results_observed") is not False
        or artifact.get("validation_results_observed") is not True
        or artifact.get("outer_results_observed") is not False
        or header.get("prereg_results_observed") is not False
        or header.get("validation_results_observed") is not True
        or header.get("outer_results_observed") is not False
    ):
        raise P1ForecastOuterBlocked("forecast artifact cannot include outer execution")
    spec: ValidationScenarioSpec | None = None
    if require_production and expected_metadata is None:
        raise P1ForecastError("production forecast artifacts require external expected_metadata")
    if require_production:
        if not isinstance(expected_metadata, Mapping):
            raise P1ForecastError("expected_metadata must be a mapping")
        required = {
            "manifest_sha256",
            "trial_registry_sha256",
            "comparison_registry_sha256",
            "prereg_results_observed",
            "validation_results_observed",
            "outer_results_observed",
            "scenario_id",
            "arm",
            "seed",
            "split_id",
            "support_id",
            "support_range",
            "fit_origin",
            "train_start",
        }
        if set(expected_metadata) != required:
            raise P1ForecastError("production expected_metadata fields are not exact")
        fixed_contract = authenticate_p1_forecast_contract()
        if (
            artifact["manifest_sha256"] != REGISTERED_MANIFEST_SHA256
            or artifact["trial_registry_sha256"] != P1_REGISTERED_TRIAL_REGISTRY_SHA256
            or artifact["comparison_registry_sha256"] != P1_REGISTERED_COMPARISON_REGISTRY_SHA256
            or fixed_contract.manifest_sha256 != REGISTERED_MANIFEST_SHA256
            or fixed_contract.trial_registry_sha256 != P1_REGISTERED_TRIAL_REGISTRY_SHA256
            or fixed_contract.comparison_registry_sha256 != P1_REGISTERED_COMPARISON_REGISTRY_SHA256
            or any(
                fixed_contract.coverage_thresholds.get(name) != value
                for name, value in P1_FIXED_COVERAGE_THRESHOLDS.items()
            )
        ):
            raise P1ForecastError("forecast artifact is not bound to the registered manifest/registries")
        try:
            spec = fixed_contract.spec(
                expected_metadata["scenario_id"],
                expected_metadata["arm"],
            )
        except (KeyError, TypeError) as exc:
            raise P1ForecastError("expected metadata scenario/arm is not registered") from exc
        expected_seed = _strict_int(expected_metadata["seed"], name="expected_metadata.seed")
        if expected_seed not in spec.seeds:
            raise P1ForecastError("expected metadata seed is not registered for the scenario/arm")
        if (
            expected_metadata["manifest_sha256"] != fixed_contract.manifest_sha256
            or expected_metadata["trial_registry_sha256"] != fixed_contract.trial_registry_sha256
            or expected_metadata["comparison_registry_sha256"] != fixed_contract.comparison_registry_sha256
            or expected_metadata["prereg_results_observed"] is not False
            or expected_metadata["validation_results_observed"] is not True
            or expected_metadata["outer_results_observed"] is not False
        ):
            raise P1ForecastError("external forecast metadata has invalid fixed source/state binding")
        expected_identity = {
            "scenario_id": spec.scenario_id,
            "arm": spec.arm,
            "seed": expected_seed,
            "split_id": spec.split_id,
            "support_id": spec.support_id,
            "support_range": list(spec.support_range),
            "fit_origin": spec.fit_origin,
            "train_start": spec.train_start,
        }
        if any(expected_metadata[name] != value for name, value in expected_identity.items()):
            raise P1ForecastError("external forecast metadata disagrees with the registered spec")
        if any(artifact[name] != expected_metadata[name] for name in (
            "manifest_sha256", "trial_registry_sha256", "comparison_registry_sha256",
            "prereg_results_observed", "validation_results_observed", "outer_results_observed",
        )):
            raise P1ForecastError("forecast artifact source/state fields disagree with external metadata")
        if any(header.get(name) != value for name, value in expected_identity.items()):
            raise P1ForecastError("forecast header identity disagrees with external registered metadata")
    elif expected_metadata is not None:
        if not isinstance(expected_metadata, Mapping):
            raise P1ForecastError("expected_metadata must be a mapping")
        for field_name in expected_metadata:
            actual = artifact[field_name] if field_name in artifact else header.get(field_name)
            if actual != expected_metadata[field_name]:
                raise P1ForecastError(f"forecast artifact {field_name} disagrees with external metadata")
    if header.get("forecast_horizons") != list(P1_FIXED_HORIZONS) or header.get("model_task_keys") != [list(key) for key in P1_ALLOWED_MODEL_TASK_KEYS]:
        raise P1ForecastError("forecast artifact does not retain the complete fixed forecast grid")
    support_range = header.get("support_range")
    if (
        not isinstance(support_range, list)
        or len(support_range) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in support_range)
        or support_range[1] <= support_range[0]
    ):
        raise P1ForecastError("forecast support range is malformed")
    if header.get("fit_range") not in [list(P1_SYNTHETIC_FIT_RANGE), list(P1_S3_FIT_RANGE)]:
        raise P1ForecastError("forecast fit range is not registered")
    if header.get("fit_origin") not in {P1_SYNTHETIC_ORIGIN, P1_S3_ORIGIN}:
        raise P1ForecastError("forecast fit origin is not registered")
    if header.get("split_id") != P1_FORECAST_SPLIT_ID:
        raise P1ForecastError("forecast split must be validation")
    if "file_sha256" in artifact or "file_sha256" in header:
        raise P1ForecastError("forecast artifact file digest must remain external")
    if spec is not None:
        if header["fit_range"] != list(spec.fit_range) or header["support_range"] != list(spec.support_range):
            raise P1ForecastError("forecast header ranges do not match the registered scenario/arm")
        if header["fit_origin"] != spec.fit_origin or header["train_start"] != spec.train_start:
            raise P1ForecastError("forecast header boundaries do not match the registered scenario/arm")
    return spec


def _validate_provenance(header: Mapping[str, Any], spec: ValidationScenarioSpec) -> None:
    provenance = header.get("scenario_provenance")
    body = header.get("body_provenance")
    if not isinstance(provenance, Mapping) or not isinstance(body, Mapping):
        raise P1ForecastError("forecast source provenance is missing")
    expected_provenance = {
        "scenario_id", "arm", "data_kind", "seed", "beta", "snr", "n_rows", "source_array_sha256",
    }
    if spec.data_kind == "s3":
        expected_provenance.add("source_body_sha256")
        allowed_provenance = expected_provenance | {"runtime"}
    else:
        allowed_provenance = expected_provenance
    if set(provenance) != allowed_provenance:
        raise P1ForecastError("scenario provenance fields are not canonical")
    if (
        provenance.get("scenario_id") != spec.scenario_id
        or provenance.get("arm") != spec.arm
        or provenance.get("data_kind") != spec.data_kind
        or provenance.get("seed") != header.get("seed")
        or provenance.get("beta") != spec.beta
        or provenance.get("snr") != spec.snr
        or provenance.get("n_rows") != spec.n_rows
    ):
        raise P1ForecastError("scenario provenance disagrees with the registered header")
    source_arrays = provenance.get("source_array_sha256")
    if not isinstance(source_arrays, Mapping) or set(source_arrays) != {
        "timestamps", "features", "returns", "availability.spot_bar_observed",
        "availability.funding_rate_available", "availability.mark_close_available",
    }:
        raise P1ForecastError("scenario source-array provenance is incomplete")
    for name, digest in source_arrays.items():
        _strict_sha256(digest, name=f"scenario_provenance.source_array_sha256.{name}")
    expected_body = {"data_kind", "body_rows", "support_range", "source_array_sha256"}
    if spec.data_kind == "s3":
        expected_body.add("source_body_sha256")
        allowed_body = expected_body | {"runtime"}
    else:
        allowed_body = expected_body
    if set(body) != allowed_body:
        raise P1ForecastError("body provenance fields are not canonical")
    if (
        body.get("data_kind") != spec.data_kind
        or body.get("body_rows") != spec.n_rows
        or body.get("support_range") != list(spec.support_range)
        or body.get("source_array_sha256") != source_arrays
    ):
        raise P1ForecastError("body provenance disagrees with the registered source")
    if spec.data_kind == "s3":
        _strict_sha256(provenance.get("source_body_sha256"), name="scenario_provenance.source_body_sha256")
        if body.get("source_body_sha256") != provenance.get("source_body_sha256"):
            raise P1ForecastError("S3 body digest is not echoed consistently")
        if "runtime" in provenance and not isinstance(provenance["runtime"], Mapping):
            raise P1ForecastError("S3 runtime provenance is malformed")
        if "runtime" in body and body["runtime"] != provenance.get("runtime"):
            raise P1ForecastError("S3 runtime provenance is not echoed consistently")


def _coverage_expected(
    *,
    support_range: tuple[int, int],
    target_end: np.ndarray,
    target_mask: np.ndarray,
    context_mask: np.ndarray,
    record: Mapping[str, Any],
    horizon: int,
    model_id: str,
    task: str,
    data_kind: str,
) -> dict[str, Any]:
    column = P1_FIXED_HORIZONS.index(horizon)
    potential = target_end[:, column] <= support_range[1]
    context = context_mask[potential]
    labels = target_mask[potential, column]
    eligible = context & labels
    prediction_mask = np.asarray(record["prediction_mask"], dtype=np.bool_)
    predictions = np.asarray(record["predictions"], dtype=np.float64)
    finite = prediction_mask[potential] & np.isfinite(predictions[potential])
    potential_count = int(np.count_nonzero(potential))
    context_count = int(np.count_nonzero(context))
    label_count = int(np.count_nonzero(labels))
    eligible_count = int(np.count_nonzero(eligible))
    finite_count = int(np.count_nonzero(finite & eligible))
    eligible_fraction = eligible_count / potential_count if potential_count else None
    context_fraction = context_count / potential_count if potential_count else None
    label_fraction = label_count / potential_count if potential_count else None
    finite_fraction = finite_count / eligible_count if eligible_count else None
    eligible_threshold = P1_FIXED_COVERAGE_THRESHOLDS[
        "synthetic_eligible_origin_fraction_min"
        if data_kind == "synthetic" else "s3_eligible_origin_fraction_min"
    ]
    if potential_count == 0 or eligible_count == 0 or record["status"] == "N/A":
        status = "N/A"
    elif (
        eligible_fraction < eligible_threshold
        or label_fraction < P1_FIXED_COVERAGE_THRESHOLDS["label_complete_fraction_min"]
        or finite_fraction < P1_FIXED_COVERAGE_THRESHOLDS["finite_oof_prediction_fraction_min"]
    ):
        status = "failed"
    else:
        status = "passed"
    return {
        "horizon": horizon,
        "model_id": model_id,
        "task": task,
        "potential_origins": potential_count,
        "context_complete": context_count,
        "label_complete": label_count,
        "eligible_origins": eligible_count,
        "finite_predictions": finite_count,
        "eligible_fraction": eligible_fraction,
        "context_fraction": context_fraction,
        "label_complete_fraction": label_fraction,
        "finite_prediction_fraction": finite_fraction,
        "thresholds": {
            "eligible_origin_fraction_min": eligible_threshold,
            "label_complete_fraction_min": P1_FIXED_COVERAGE_THRESHOLDS["label_complete_fraction_min"],
            "finite_oof_prediction_fraction_min": P1_FIXED_COVERAGE_THRESHOLDS["finite_oof_prediction_fraction_min"],
        },
        "status": status,
        "promotion_allowed": status == "passed",
    }


def _validate_future_evidence(
    evidence: Mapping[str, Any],
    *,
    spec: ValidationScenarioSpec | None,
    header: Mapping[str, Any],
) -> None:
    if evidence.get("status") not in {"passed", "N/A"}:
        raise P1ForecastError("future perturbation evidence is missing or invalid")
    if evidence.get("origin") != header.get("fit_origin") or evidence.get("horizon") != 4:
        raise P1ForecastError("future perturbation evidence has the wrong fit identity")
    if evidence.get("perturb_start") != header.get("support_range", [None, None])[1]:
        raise P1ForecastError("future perturbation evidence has the wrong perturbation boundary")
    if evidence.get("status") == "N/A":
        if not isinstance(evidence.get("reason"), str) or not evidence.get("reason"):
            raise P1ForecastError("N/A future perturbation evidence requires a reason")
        return
    for name in ("fitted_prefix_mask_sha256", "earlier_prediction_mask_sha256", "earlier_prediction_sha256"):
        _strict_sha256(evidence.get(name), name=f"future_perturbation_evidence.{name}")
    if spec is not None and spec.data_kind == "s3":
        _strict_sha256(
            evidence.get("source_body_sha256"),
            name="future_perturbation_evidence.source_body_sha256",
        )
        if evidence["source_body_sha256"] != header["body_provenance"]["source_body_sha256"]:
            raise P1ForecastError("future perturbation evidence is bound to another S3 body")


def _validate_forecast_payload(
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None,
    require_production: bool,
) -> Mapping[str, Any]:
    spec = _validate_identity(
        artifact,
        expected_metadata=expected_metadata,
        require_production=require_production,
    )
    header = artifact["header"]
    support_range = tuple(header["support_range"])
    support_count = support_range[1] - support_range[0]
    if support_count <= 0 or support_count > P1_FORECAST_FILE_MAX_ROWS:
        raise P1ForecastError("forecast support row count is outside its bound")
    if spec is not None:
        if support_count != spec.support_range[1] - spec.support_range[0]:
            raise P1ForecastError("forecast support length does not match the registered scenario")
        _validate_provenance(header, spec)
    timestamps = _decode_timestamps(artifact["support_timestamps"], expected_count=support_count)
    spot_bar_observed = _decode_bool_array(
        artifact["spot_bar_observed"],
        shape=(support_count,),
        name="spot_bar_observed",
    )
    try:
        returns = np.asarray(artifact["realized_returns"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ForecastError("realized_returns cannot be decoded as float64") from exc
    if returns.shape != (support_count,) or np.isinf(returns).any():
        raise P1ForecastError("realized_returns has invalid shape or infinity")
    if np.any(spot_bar_observed & ~np.isfinite(returns)):
        raise P1ForecastError(
            "realized_returns may be non-finite only when spot_bar_observed is false"
        )
    returns = np.asarray(returns, dtype=np.float64)
    target_mask = _decode_bool_array(artifact["target_mask"], shape=(support_count, len(P1_FIXED_HORIZONS)), name="target_mask")
    targets = _decode_float_array(
        artifact["targets"],
        shape=(support_count, len(P1_FIXED_HORIZONS)),
        mask=target_mask,
        name="targets",
    )
    target_end = _decode_int_array(artifact["target_end"], shape=(support_count, len(P1_FIXED_HORIZONS)), name="target_end")
    labels = _decode_int_array(artifact["binary_labels"], shape=(support_count, len(P1_FIXED_HORIZONS)), name="binary_labels")
    if np.any((labels < -1) | (labels > 1)):
        raise P1ForecastError("binary labels are outside {-1,0,1}")
    expected_target_end = (
        np.arange(support_range[0], support_range[1], dtype=np.int64)[:, None]
        + np.asarray(P1_FIXED_HORIZONS, dtype=np.int64)[None, :]
        + 1
    )
    if not np.array_equal(target_end, expected_target_end):
        raise P1ForecastError("target_end does not equal the fixed t+h+1 formula")
    if np.any(labels[target_mask] != (targets[target_mask] > 0.0).astype(np.int8)):
        raise P1ForecastError("binary labels do not match finite targets")
    if np.any(labels[~target_mask] != -1):
        raise P1ForecastError("binary labels must be -1 where the target mask is false")
    context_mask = _decode_bool_array(artifact["context_mask"], shape=(support_count,), name="context_mask")
    origin_mask = _decode_bool_array(
        artifact["origin_mask"],
        shape=(support_count,),
        name="origin_mask",
    )
    score_eligible_mask = _decode_bool_array(
        artifact["score_eligible_mask"],
        shape=(support_count,),
        name="score_eligible_mask",
    )
    expected_origin_mask = context_mask & (
        target_end[:, P1_FIXED_HORIZONS.index(4)] <= support_range[1]
    )
    expected_score_mask = expected_origin_mask & target_mask[:, P1_FIXED_HORIZONS.index(4)]
    if not np.array_equal(origin_mask, expected_origin_mask):
        raise P1ForecastError("origin_mask disagrees with causal context/range geometry")
    if not np.array_equal(score_eligible_mask, expected_score_mask):
        raise P1ForecastError("score_eligible_mask disagrees with origin and h4 target masks")
    mask_hashes = artifact.get("mask_hashes")
    if not isinstance(mask_hashes, Mapping) or set(mask_hashes) != {
        "context_mask",
        "origin_mask",
        "score_eligible_mask",
        "target_mask",
        "spot_bar_observed",
    }:
        raise P1ForecastError("mask_hashes fields are not canonical")
    decoded_masks = {
        "context_mask": context_mask,
        "origin_mask": origin_mask,
        "score_eligible_mask": score_eligible_mask,
        "target_mask": target_mask,
        "spot_bar_observed": spot_bar_observed,
    }
    for name, value in mask_hashes.items():
        _strict_sha256(value, name=f"mask_hashes.{name}")
        if value != _array_sha256(decoded_masks[name], name=name):
            raise P1ForecastError(f"mask_hashes.{name} does not match its mask")
    fits = artifact.get("fits")
    if not isinstance(fits, list) or len(fits) != len(P1_FIXED_HORIZONS) * len(P1_ALLOWED_MODEL_TASK_KEYS):
        raise P1ForecastError("forecast artifact must contain every fixed horizon/model/task fit")
    seen: set[tuple[int, str, str]] = set()
    decoded_fits: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for index, record in enumerate(fits):
        if not isinstance(record, Mapping) or set(record) != set(_FIT_FIELDS):
            raise P1ForecastError(f"forecast fit row {index} fields are not canonical")
        horizon = _strict_int(record.get("horizon"), name=f"fits[{index}].horizon")
        model_id = record.get("model_id")
        task = record.get("task")
        if (horizon, model_id, task) not in {
            (h, model, t) for h in P1_FIXED_HORIZONS for model, t in P1_ALLOWED_MODEL_TASK_KEYS
        }:
            raise P1ForecastError(f"forecast fit row {index} has an unregistered key")
        key = (horizon, model_id, task)
        if key in seen:
            raise P1ForecastError(f"duplicate forecast fit key: {key}")
        seen.add(key)
        status = record.get("status")
        if status not in {"ok", "N/A"}:
            raise P1ForecastError(f"forecast fit row {index} has an invalid status")
        reason = record.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise P1ForecastError(f"forecast fit row {index} reason is malformed")
        train_count = _strict_int(record.get("train_count"), name=f"fits[{index}].train_count")
        fit_limit = (spec.fit_range[1] - spec.fit_range[0]) if spec is not None else P1_FORECAST_FILE_MAX_ROWS
        if train_count < 0 or train_count > fit_limit:
            raise P1ForecastError(f"forecast fit row {index} train_count is outside its bound")
        train_mask = _decode_bool_array(record.get("train_mask"), shape=(support_count,), name=f"fits[{index}].train_mask")
        eligible_mask = _decode_bool_array(record.get("eligible_mask"), shape=(support_count,), name=f"fits[{index}].eligible_mask")
        prediction_mask = _decode_bool_array(record.get("prediction_mask"), shape=(support_count,), name=f"fits[{index}].prediction_mask")
        expected_inference = context_mask & (
            target_end[:, P1_FIXED_HORIZONS.index(horizon)] <= support_range[1]
        )
        expected_eligible = expected_inference & target_mask[:, P1_FIXED_HORIZONS.index(horizon)]
        if not np.array_equal(eligible_mask, expected_eligible):
            raise P1ForecastError(f"forecast fit row {index} eligible_mask disagrees with support masks")
        if np.any(train_mask):
            raise P1ForecastError(f"forecast fit row {index} train_mask leaks into validation support")
        prediction_mask = np.asarray(record["prediction_mask"], dtype=np.bool_)
        predictions = _decode_float_array(
            record.get("predictions"),
            shape=(support_count,),
            mask=prediction_mask,
            name=f"fits[{index}].predictions",
        )
        if status == "N/A" and np.any(prediction_mask):
            raise P1ForecastError(f"N/A forecast fit row {index} contains predictions")
        if status == "ok":
            if not np.array_equal(prediction_mask, expected_inference):
                raise P1ForecastError(
                    f"successful forecast fit row {index} does not cover every causal inference row"
                )
            if np.any(eligible_mask & ~prediction_mask):
                raise P1ForecastError(
                    f"forecast fit row {index} score eligibility exceeds causal inference rows"
                )
        decoded_fits[key] = record
    expected_keys = {(h, model, task) for h in P1_FIXED_HORIZONS for model, task in P1_ALLOWED_MODEL_TASK_KEYS}
    if seen != expected_keys:
        raise P1ForecastError("forecast fit keys are incomplete")
    coverage = artifact.get("coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != {_fit_key(*key) for key in expected_keys}:
        raise P1ForecastError("coverage summaries do not cover the complete fit grid")
    data_kind = spec.data_kind if spec is not None else (
        "s3" if header.get("support_id") == P1_S3_SUPPORT_ID else "synthetic"
    )
    for key, summary in coverage.items():
        if not isinstance(summary, Mapping):
            raise P1ForecastError(f"coverage summary is malformed: {key}")
        try:
            fit_key = next(item for item in expected_keys if _fit_key(*item) == key)
        except StopIteration as exc:
            raise P1ForecastError(f"coverage key is not registered: {key}") from exc
        expected_summary = _coverage_expected(
            support_range=support_range,
            target_end=target_end,
            target_mask=target_mask,
            context_mask=context_mask,
            record=decoded_fits[fit_key],
            horizon=fit_key[0],
            model_id=fit_key[1],
            task=fit_key[2],
            data_kind=data_kind,
        )
        if dict(summary) != expected_summary:
            raise P1ForecastError(f"coverage summary is not an exact recomputation: {key}")
    evidence = header.get("future_perturbation_evidence")
    if not isinstance(evidence, Mapping):
        raise P1ForecastError("future perturbation evidence is missing or invalid")
    _validate_future_evidence(evidence, spec=spec, header=header)
    if evidence.get("status") == "N/A":
        # N/A evidence is retained, but explicitly blocks promotion through
        # the artifact validation result.
        promotion_allowed = False
    else:
        promotion_allowed = all(summary.get("status") == "passed" for summary in coverage.values())
    return MappingProxyType(
        {
            "status": "passed" if promotion_allowed else "N/A",
            "promotion_allowed": promotion_allowed,
            "support_count": support_count,
            "timestamps": timestamps,
            "realized_returns": returns,
            "spot_bar_observed": spot_bar_observed,
            "targets": targets,
            "target_mask": target_mask,
            "context_mask": context_mask,
            "origin_mask": origin_mask,
            "score_eligible_mask": score_eligible_mask,
            "fits": MappingProxyType(decoded_fits),
            "coverage": coverage,
        }
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P1ForecastError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular_file(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise P1ForecastError(f"could not stat forecast artifact: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise P1ForecastError("forecast artifact must be a regular non-symlink file")
    if before.st_size <= 0 or before.st_size > P1_FORECAST_FILE_MAX_BYTES:
        raise P1ForecastError("forecast artifact file size is outside its fixed bound")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise P1ForecastError("forecast artifact must remain a regular file")
        if opened.st_size <= 0 or opened.st_size > P1_FORECAST_FILE_MAX_BYTES:
            raise P1ForecastError("forecast artifact file size is outside its fixed bound")
        signature = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if signature != before_signature:
            raise P1ForecastError("forecast artifact changed before it was opened")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise P1ForecastError("forecast artifact ended during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise P1ForecastError("forecast artifact changed during read")
        encoded = b"".join(chunks)
    except OSError as exc:
        raise P1ForecastError(f"could not read forecast artifact: {path}") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return encoded, hashlib.sha256(encoded).hexdigest()


def _decode_payload(encoded: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                P1ForecastError(f"non-finite JSON constant is forbidden: {value}")
            ),
        )
    except P1ForecastError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise P1ForecastError("forecast artifact JSON is malformed") from exc
    if not isinstance(payload, Mapping):
        raise P1ForecastError("forecast artifact JSON must contain an object")
    return payload


@dataclass(frozen=True, eq=False)
class ForecastActionSource:
    """Sealed action-source capability emitted by a validated forecast load."""

    # Capabilities are tracked by object identity, never by equality of their
    # mutable NumPy payloads.  This also means ``dataclasses.replace`` creates
    # an unregistered object instead of inheriting authentication implicitly.
    __hash__ = object.__hash__

    scenario_id: str
    arm: str
    seed: int
    split_id: str
    support_id: str
    support_range: tuple[int, int]
    fit_origin: int
    timestamps: np.ndarray
    realized_returns: np.ndarray
    forecast_h4: np.ndarray
    forecast_h4_mask: np.ndarray
    origin_mask: np.ndarray
    score_mask: np.ndarray
    common_mask: np.ndarray
    source_hashes: Mapping[str, str]
    prereg_results_observed: bool = False
    validation_results_observed: bool = True
    outer_results_observed: bool = False
    validation_status: str = "N/A"
    promotion_allowed: bool = False
    _production_seal: object | None = field(default=None, repr=False, compare=False)
    binding_sha256: str = ""
    # Explicit source masks are retained separately from forecast/score
    # masks so downstream fill/outcome logic never treats a target gap as an
    # unavailable decision origin.  They are optional only for compatibility
    # with direct, unauthenticated fixture construction; authenticated loads
    # always populate them.
    context_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    target_h4_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    spot_bar_observed: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def is_authenticated(self) -> bool:
        return _is_registered_forecast_action_source(self)

    # Names used by the downstream action boundary.  These aliases do not
    # create alternate sources; every value still points at this sealed,
    # support-scoped capability.
    @property
    def returns(self) -> np.ndarray:
        return self.realized_returns

    @property
    def forecast(self) -> np.ndarray:
        return self.forecast_h4

    @property
    def forecast_mask(self) -> np.ndarray:
        return self.forecast_h4_mask

    @property
    def decision_eligible(self) -> np.ndarray:
        return self.origin_mask

    @property
    def context_eligible(self) -> np.ndarray:
        if self.context_mask is None:
            raise P1ForecastError("forecast action source lacks context mask")
        return self.context_mask

    @property
    def target_complete(self) -> np.ndarray:
        if self.target_h4_mask is None:
            raise P1ForecastError("forecast action source lacks h4 target mask")
        return self.target_h4_mask

    @property
    def score_eligible(self) -> np.ndarray:
        raise P1ForecastError(
            "score_eligible is ambiguous at the action boundary; use "
            "action_score_mask for forecasted-action scoring or bar_available "
            "for fill/outcome availability"
        )

    @property
    def action_score_mask(self) -> np.ndarray:
        """Rows where the selected finite forecast action is scoreable."""

        return self.score_mask

    @property
    def common_eligible(self) -> np.ndarray:
        return self.common_mask

    @property
    def bar_available(self) -> np.ndarray:
        """Spot-bar availability for fill/outcome handling."""

        if self.spot_bar_observed is None:
            raise P1ForecastError("forecast action source lacks spot-bar availability")
        return self.spot_bar_observed


class _ForecastActionSourceSeal:
    pass


_FORECAST_ACTION_SOURCE_SEAL = _ForecastActionSourceSeal()
_AUTHENTICATED_FORECAST_SOURCES: weakref.WeakKeyDictionary[ForecastActionSource, str] = (
    weakref.WeakKeyDictionary()
)


def _action_source_binding_sha256(value: ForecastActionSource) -> str:
    if (
        value.context_mask is None
        or value.target_h4_mask is None
        or value.spot_bar_observed is None
    ):
        raise P1ForecastError("forecast action source is missing explicit source masks")
    payload = {
        "scenario_id": value.scenario_id,
        "arm": value.arm,
        "seed": value.seed,
        "split_id": value.split_id,
        "support_id": value.support_id,
        "support_range": list(value.support_range),
        "fit_origin": value.fit_origin,
        "prereg_results_observed": value.prereg_results_observed,
        "validation_results_observed": value.validation_results_observed,
        "outer_results_observed": value.outer_results_observed,
        "validation_status": value.validation_status,
        "promotion_allowed": value.promotion_allowed,
        "source_hashes": dict(sorted(value.source_hashes.items())),
        "array_hashes": {
            "timestamps": _array_sha256(value.timestamps, name="timestamps"),
            "realized_returns": _array_sha256(value.realized_returns, name="realized_returns"),
            "forecast_h4": _array_sha256(value.forecast_h4, name="forecast_h4"),
            "forecast_h4_mask": _array_sha256(value.forecast_h4_mask, name="forecast_h4_mask"),
            "context_mask": _array_sha256(value.context_mask, name="context_mask"),
            "target_h4_mask": _array_sha256(value.target_h4_mask, name="target_h4_mask"),
            "origin_mask": _array_sha256(value.origin_mask, name="origin_mask"),
            "score_mask": _array_sha256(value.score_mask, name="score_mask"),
            "common_mask": _array_sha256(value.common_mask, name="common_mask"),
            "spot_bar_observed": _array_sha256(
                value.spot_bar_observed,
                name="spot_bar_observed",
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_registered_forecast_action_source(value: Any) -> bool:
    """Check loader registration and binding, not merely the module sentinel."""

    if not isinstance(value, ForecastActionSource) or value._production_seal is not _FORECAST_ACTION_SOURCE_SEAL:
        return False
    try:
        registered_binding = _AUTHENTICATED_FORECAST_SOURCES.get(value)
        if registered_binding is None:
            return False
        current_binding = _action_source_binding_sha256(value)
    except Exception:
        return False
    return registered_binding == value.binding_sha256 == current_binding


def _read_only(array: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _capability_from_loaded(
    artifact: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    file_sha256: str,
    sealed: bool = True,
) -> ForecastActionSource:
    header = artifact["header"]
    h4_record = next(
        row
        for row in artifact["fits"]
        if row["horizon"] == 4 and row["model_id"] == "ridge" and row["task"] == "continuous"
    )
    forecast = np.asarray(h4_record["predictions"], dtype=np.float64)
    forecast_mask = np.asarray(h4_record["prediction_mask"], dtype=np.bool_)
    context_mask = np.asarray(validation["context_mask"], dtype=np.bool_)
    target_h4_mask = np.asarray(artifact["target_mask"], dtype=np.bool_)[:, 1]
    origin_mask = np.asarray(validation["origin_mask"], dtype=np.bool_)
    score_mask = origin_mask & forecast_mask & target_h4_mask
    declared_score_eligible_mask = np.asarray(
        validation["score_eligible_mask"],
        dtype=np.bool_,
    )
    if not np.array_equal(
        declared_score_eligible_mask,
        origin_mask & target_h4_mask,
    ):
        raise P1ForecastError(
            "loaded capability score eligibility disagrees with persisted h4 masks"
        )
    spot_bar_observed = np.asarray(validation["spot_bar_observed"], dtype=np.bool_)
    common_mask = score_mask.copy()
    timestamps = np.asarray(validation["timestamps"], dtype=np.dtype("datetime64[ns]"))
    realized = np.asarray(validation["realized_returns"], dtype=np.float64)
    source_hashes = {
        "forecast_file_sha256": file_sha256,
        "manifest_sha256": artifact["manifest_sha256"],
        "trial_registry_sha256": artifact["trial_registry_sha256"],
        "comparison_registry_sha256": artifact["comparison_registry_sha256"],
        "support_timestamps_sha256": _array_sha256(timestamps, name="support_timestamps"),
        "realized_returns_sha256": _array_sha256(realized, name="realized_returns"),
        "forecast_h4_sha256": _array_sha256(forecast, name="forecast_h4"),
        "forecast_h4_mask_sha256": _array_sha256(forecast_mask, name="forecast_h4_mask"),
        "context_mask_sha256": _array_sha256(context_mask, name="context_mask"),
        "target_h4_mask_sha256": _array_sha256(target_h4_mask, name="target_h4_mask"),
        "score_eligible_mask_sha256": _array_sha256(
            declared_score_eligible_mask,
            name="score_eligible_mask",
        ),
        "origin_mask_sha256": _array_sha256(origin_mask, name="origin_mask"),
        "score_mask_sha256": _array_sha256(score_mask, name="score_mask"),
        "common_mask_sha256": _array_sha256(common_mask, name="common_mask"),
        "spot_bar_observed_sha256": _array_sha256(
            spot_bar_observed,
            name="spot_bar_observed",
        ),
    }
    body_provenance = header.get("body_provenance")
    if isinstance(body_provenance, Mapping):
        source_body = body_provenance.get("source_body_sha256")
        if isinstance(source_body, str):
            source_hashes["source_body_sha256"] = _strict_sha256(source_body, name="source_body_sha256")
        source_arrays = body_provenance.get("source_array_sha256")
        if isinstance(source_arrays, Mapping):
            for name, value in source_arrays.items():
                if isinstance(value, str) and _SHA256_RE.fullmatch(value):
                    source_hashes[f"body.{name}"] = value
    capability = ForecastActionSource(
        scenario_id=str(header["scenario_id"]),
        arm=str(header["arm"]),
        seed=_strict_int(header["seed"], name="seed"),
        split_id=str(header["split_id"]),
        support_id=str(header["support_id"]),
        support_range=tuple(_strict_int(value, name="support_range") for value in header["support_range"]),
        fit_origin=_strict_int(header["fit_origin"], name="fit_origin"),
        timestamps=_read_only(timestamps, dtype=np.dtype("datetime64[ns]")),
        realized_returns=_read_only(realized, dtype=np.float64),
        forecast_h4=_read_only(forecast, dtype=np.float64),
        forecast_h4_mask=_read_only(forecast_mask, dtype=np.bool_),
        origin_mask=_read_only(origin_mask, dtype=np.bool_),
        score_mask=_read_only(score_mask, dtype=np.bool_),
        common_mask=_read_only(common_mask, dtype=np.bool_),
        source_hashes=MappingProxyType(source_hashes),
        prereg_results_observed=bool(artifact["prereg_results_observed"]),
        validation_results_observed=bool(artifact["validation_results_observed"]),
        outer_results_observed=bool(artifact["outer_results_observed"]),
        validation_status=str(validation["status"]),
        promotion_allowed=bool(validation["promotion_allowed"]),
        _production_seal=_FORECAST_ACTION_SOURCE_SEAL if sealed else None,
        context_mask=_read_only(context_mask, dtype=np.bool_),
        target_h4_mask=_read_only(target_h4_mask, dtype=np.bool_),
        spot_bar_observed=_read_only(spot_bar_observed, dtype=np.bool_),
    )
    capability = replace(capability, binding_sha256=_action_source_binding_sha256(capability))
    if sealed:
        _AUTHENTICATED_FORECAST_SOURCES[capability] = capability.binding_sha256
    return capability


@dataclass(frozen=True)
class LoadedP1ForecastArtifact:
    path: Path
    file_sha256: str
    artifact: Mapping[str, Any]
    validation: Mapping[str, Any]
    action_source: ForecastActionSource

    @property
    def promotion_allowed(self) -> bool:
        return bool(self.validation.get("promotion_allowed"))

    def as_action_source(self) -> ForecastActionSource:
        if not self.action_source.promotion_allowed:
            raise P1ForecastError("forecast action source is blocked until all validation gates pass")
        return self.action_source


def save_p1_forecast_artifact(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    require_production: bool = True,
) -> str:
    """Validate and atomically persist one forecast artifact; return its SHA."""

    _validate_forecast_payload(
        artifact,
        expected_metadata=expected_metadata,
        require_production=require_production,
    )
    encoded = _json_bytes(artifact)
    digest = hashlib.sha256(encoded).hexdigest()
    target = Path(path)
    parent = target.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise P1ForecastError("forecast artifact parent is not a directory")
    try:
        existing = target.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise P1ForecastError(f"could not inspect forecast artifact target: {target}") from exc
    if existing is not None and stat.S_ISLNK(existing.st_mode):
        raise P1ForecastError("forecast artifact target must not be a symlink")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, mode="wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise P1ForecastError(f"could not persist forecast artifact: {target}") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return digest


def load_p1_forecast_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_metadata: Mapping[str, Any] | None = None,
    require_production: bool = True,
) -> LoadedP1ForecastArtifact:
    """Load one artifact with a mandatory external file SHA-256."""

    expected_digest = _strict_sha256(expected_file_sha256, name="expected_file_sha256")
    source = Path(path)
    encoded, actual_digest = _read_regular_file(source)
    if actual_digest != expected_digest:
        raise P1ForecastError("stored forecast artifact file SHA-256 mismatch")
    payload = _decode_payload(encoded)
    validation = _validate_forecast_payload(
        payload,
        expected_metadata=expected_metadata,
        require_production=require_production,
    )
    action_source = _capability_from_loaded(
        payload,
        validation,
        file_sha256=actual_digest,
        sealed=require_production and bool(validation["promotion_allowed"]),
    )
    return LoadedP1ForecastArtifact(
        path=source,
        file_sha256=actual_digest,
        artifact=payload,
        validation=validation,
        action_source=action_source,
    )


def execute_p1_outer_report(*_: Any, **__: Any) -> None:
    """Keep the preregistered outer report permanently blocked here."""

    raise P1ForecastOuterBlocked("P1 outer report is report-only and not implemented in this boundary")


def is_authenticated_forecast_action_source(value: Any) -> bool:
    """Return whether ``value`` is the sealed capability emitted by a load."""

    return _is_registered_forecast_action_source(value)


def require_authenticated_forecast_action_source(value: Any) -> ForecastActionSource:
    """Return only a sealed, structurally valid action-source capability."""

    if not is_authenticated_forecast_action_source(value):
        raise P1ForecastError(
            "production action input must be the sealed capability from a validated forecast artifact load"
        )
    if (
        value.prereg_results_observed is not False
        or value.validation_results_observed is not True
        or value.outer_results_observed is not False
        or value.validation_status != "passed"
        or value.promotion_allowed is not True
    ):
        raise P1ForecastError("forecast action source has invalid result-state semantics")
    scenario_arm = (value.scenario_id, value.arm)
    expected_arms = set(P1_SCENARIO_ARMS)
    if scenario_arm not in expected_arms:
        raise P1ForecastError("forecast action source scenario/arm is not registered")
    expected_seeds = P1_S3_SEEDS if value.scenario_id == "S3" else P1_SYNTHETIC_SEEDS
    if _strict_int(value.seed, name="forecast action source seed") not in expected_seeds:
        raise P1ForecastError("forecast action source seed is not registered")
    if value.split_id != P1_FORECAST_SPLIT_ID:
        raise P1ForecastError("forecast action source split is not validation")
    expected_support = P1_S3_SUPPORT_RANGE if value.scenario_id == "S3" else P1_SYNTHETIC_SUPPORT_RANGE
    expected_support_id = P1_S3_SUPPORT_ID if value.scenario_id == "S3" else P1_SYNTHETIC_SUPPORT_ID
    expected_origin = P1_S3_ORIGIN if value.scenario_id == "S3" else P1_SYNTHETIC_ORIGIN
    if value.support_id != expected_support_id or tuple(value.support_range) != expected_support or value.fit_origin != expected_origin:
        raise P1ForecastError("forecast action source support is not registered")
    size = len(value.timestamps)
    if size != expected_support[1] - expected_support[0]:
        raise P1ForecastError("forecast action source support length is not registered")
    if any(
        np.asarray(array).shape != (size,)
        for array in (
            value.realized_returns,
            value.forecast_h4,
            value.forecast_h4_mask,
            value.origin_mask,
            value.score_mask,
            value.common_mask,
            value.context_mask,
            value.target_h4_mask,
            value.spot_bar_observed,
        )
    ):
        raise P1ForecastError("forecast action source arrays are not support aligned")
    if np.asarray(value.timestamps).dtype != np.dtype("datetime64[ns]"):
        raise P1ForecastError("forecast action source timestamps are not datetime64[ns]")
    for name, array in (
        ("forecast_h4_mask", value.forecast_h4_mask),
        ("origin_mask", value.origin_mask),
        ("score_mask", value.score_mask),
        ("common_mask", value.common_mask),
        ("context_mask", value.context_mask),
        ("target_h4_mask", value.target_h4_mask),
        ("spot_bar_observed", value.spot_bar_observed),
    ):
        if np.asarray(array).dtype != np.dtype(np.bool_):
            raise P1ForecastError(f"forecast action source {name} must be bool")
    if np.asarray(value.realized_returns).dtype != np.dtype(np.float64) or np.asarray(value.forecast_h4).dtype != np.dtype(np.float64):
        raise P1ForecastError("forecast action source values must be float64")
    if np.isinf(value.realized_returns).any():
        raise P1ForecastError("forecast action source realized returns must not contain infinity")
    if np.any(value.spot_bar_observed & ~np.isfinite(value.realized_returns)):
        raise P1ForecastError(
            "forecast action source realized returns may be non-finite only on unavailable Spot bars"
        )
    if not np.array_equal(np.isfinite(value.forecast_h4), value.forecast_h4_mask):
        raise P1ForecastError("forecast action source forecast mask is inconsistent")
    support_rows = np.arange(value.support_range[0], value.support_range[1], dtype=np.int64)
    expected_origin = value.context_mask & (support_rows + 4 + 1 <= value.support_range[1])
    if not np.array_equal(value.origin_mask, expected_origin):
        raise P1ForecastError("forecast action source origin mask is not causal context plus split-tail boundary")
    if not np.array_equal(value.score_mask, value.origin_mask & value.forecast_h4_mask & value.target_h4_mask):
        raise P1ForecastError("forecast action source score mask is inconsistent")
    if not np.array_equal(value.common_mask, value.score_mask):
        raise P1ForecastError("forecast action source common mask must equal its score mask")
    body_names = {
        "body.timestamps",
        "body.features",
        "body.returns",
        "body.availability.spot_bar_observed",
        "body.availability.funding_rate_available",
        "body.availability.mark_close_available",
    }
    expected_hash_names = {
        "forecast_file_sha256",
        "manifest_sha256",
        "trial_registry_sha256",
        "comparison_registry_sha256",
        "support_timestamps_sha256",
        "realized_returns_sha256",
        "forecast_h4_sha256",
        "forecast_h4_mask_sha256",
        "context_mask_sha256",
        "target_h4_mask_sha256",
        "score_eligible_mask_sha256",
        "origin_mask_sha256",
        "score_mask_sha256",
        "common_mask_sha256",
        "spot_bar_observed_sha256",
        *body_names,
    }
    if value.scenario_id == "S3":
        expected_hash_names.add("source_body_sha256")
    if set(value.source_hashes) != expected_hash_names:
        raise P1ForecastError("forecast action source hashes are incomplete or contain extras")
    for name, digest in value.source_hashes.items():
        _strict_sha256(digest, name=f"forecast action source hash {name}")
    expected_hashes = {
        "manifest_sha256": REGISTERED_MANIFEST_SHA256,
        "trial_registry_sha256": P1_REGISTERED_TRIAL_REGISTRY_SHA256,
        "comparison_registry_sha256": P1_REGISTERED_COMPARISON_REGISTRY_SHA256,
        "support_timestamps_sha256": _array_sha256(value.timestamps, name="support_timestamps"),
        "realized_returns_sha256": _array_sha256(value.realized_returns, name="realized_returns"),
        "forecast_h4_sha256": _array_sha256(value.forecast_h4, name="forecast_h4"),
        "forecast_h4_mask_sha256": _array_sha256(value.forecast_h4_mask, name="forecast_h4_mask"),
        "context_mask_sha256": _array_sha256(value.context_mask, name="context_mask"),
        "target_h4_mask_sha256": _array_sha256(value.target_h4_mask, name="target_h4_mask"),
        "score_eligible_mask_sha256": _array_sha256(
            value.origin_mask & value.target_h4_mask,
            name="score_eligible_mask",
        ),
        "origin_mask_sha256": _array_sha256(value.origin_mask, name="origin_mask"),
        "score_mask_sha256": _array_sha256(value.score_mask, name="score_mask"),
        "common_mask_sha256": _array_sha256(value.common_mask, name="common_mask"),
        "spot_bar_observed_sha256": _array_sha256(
            value.spot_bar_observed,
            name="spot_bar_observed",
        ),
    }
    for name, digest in expected_hashes.items():
        if value.source_hashes.get(name) != digest:
            raise P1ForecastError(f"forecast action source hash does not match its value: {name}")
    if value.scenario_id == "S3" and value.source_hashes.get("source_body_sha256") is None:
        raise P1ForecastError("S3 forecast action source lacks its authenticated body digest")
    if not isinstance(value.binding_sha256, str) or _SHA256_RE.fullmatch(value.binding_sha256) is None:
        raise P1ForecastError("forecast action source binding digest is malformed")
    if value.binding_sha256 != _action_source_binding_sha256(value):
        raise P1ForecastError("forecast action source binding digest does not match its values")
    return value


# Compatibility names make the capability boundary discoverable without
# creating a second type or an alternate unsealed constructor.
P1ForecastActionSource = ForecastActionSource
ForecastActionSourceCapability = ForecastActionSource
require_p1_forecast_action_source = require_authenticated_forecast_action_source


__all__ = [
    "ForecastActionSource",
    "ForecastActionSourceCapability",
    "ForecastCoverageSummary",
    "LoadedP1ForecastArtifact",
    "P1ForecastContract",
    "P1ForecastError",
    "P1ForecastOuterBlocked",
    "P1ForecastActionSource",
    "P1_ALLOWED_MODEL_TASK_KEYS",
    "P1_COMPARISON_REGISTRY_COUNT",
    "P1_COVERAGE_THRESHOLD_KEYS",
    "P1_FIXED_HORIZONS",
    "P1_FORECAST_FILE_FORMAT",
    "P1_FORECAST_FILE_MAX_BYTES",
    "P1_FORECAST_SCHEMA_ID",
    "P1_FORECAST_FILE_VERSION",
    "P1_S3_FIT_RANGE",
    "P1_S3_ORIGIN",
    "P1_S3_SUPPORT_ID",
    "P1_S3_SUPPORT_RANGE",
    "P1_S3_TRAIN_START",
    "P1_SCENARIO_ARMS",
    "P1_SYNTHETIC_FIT_RANGE",
    "P1_SYNTHETIC_SEEDS",
    "P1_SYNTHETIC_SUPPORT_ID",
    "P1_SYNTHETIC_SUPPORT_RANGE",
    "authenticate_p1_forecast_contract",
    "build_p1_forecast_artifact",
    "execute_p1_outer_report",
    "is_authenticated_forecast_action_source",
    "require_authenticated_forecast_action_source",
    "require_p1_forecast_action_source",
    "load_authenticated_p1_forecast_contract",
    "load_p1_forecast_artifact",
    "load_p1_forecast_contract",
    "save_p1_forecast_artifact",
]


# Public alias retained for callers that use the registry's terminology.
P1_COMPARISON_REGISTRY_COUNT = P1_PRIMARY_COMPARISON_COUNT
