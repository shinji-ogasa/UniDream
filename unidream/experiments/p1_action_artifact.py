"""Atomic, externally bound persistence for P1 action primitive artifacts.

The canonical action hashes intentionally cover the external schema and the
full record grid, not every provenance field in the in-memory header.  This
module adds the storage boundary used by the experiment runner: exact file
bytes are hashed outside the payload, production loads require that digest,
and all source arrays/arm metadata are revalidated instead of trusting the
stored header.  A production load additionally requires the sealed forecast
source and the exact registered source binding, then returns an identity-
registered capability whose MBB input contains immutable metric values,
field-specific effective masks, and provenance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any
import weakref

import numpy as np

from .action_primitives import (
    ACTION_PRIMITIVE_ARM_FIELDS,
    ACTION_PRIMITIVE_HASH_FIELDS,
    ACTION_PRIMITIVE_METRIC_MASK_REGISTRY,
    ACTION_PRIMITIVE_METRIC_FIELDS,
    ACTION_PRIMITIVE_RECORD_FIELDS,
    ActionPrimitiveContractError,
    validate_action_primitive_semantics,
)


class P1ActionArtifactError(ValueError):
    """Raised when a stored P1 action artifact is ambiguous or altered."""


P1_ACTION_FILE_FORMAT = "unidream.p1.action_primitive.columnar_json"
P1_ACTION_FILE_VERSION = 1
P1_ACTION_FILE_MAX_BYTES = 64 * 1024 * 1024
P1_ACTION_FILE_MAX_RECORDS = 100_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "header",
        "record_fields",
        "record_count",
        "columns",
        *ACTION_PRIMITIVE_HASH_FIELDS,
    }
)
_HEADER_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_id",
        "schema_version",
        "record_fields",
        "record_count",
        "bar_count",
        "support_start",
        "support_range",
        "contract_hash",
        "contract_path",
        "contract",
        "action_grid",
        "cooldown",
        "execution",
        "cost",
        "schedule",
        "mask_logic",
        "metric_mask_registry",
        "paired_common_mask_supplied",
        "paired_common_mask",
        "paired_common_mask_sha256",
        "arm_metadata",
        *ACTION_PRIMITIVE_ARM_FIELDS,
        "source_role",
        "teacher_oracle_execution",
        "action_primitive_producer_status",
        "metric_source",
        "moving_block_bootstrap_status",
        "contract_json_sha256",
        "trial_id",
        "source_binding",
        "source_binding_sha256",
        *ACTION_PRIMITIVE_HASH_FIELDS,
    }
)
_LEGACY_FIXTURE_HEADER_FIELDS = frozenset(_HEADER_FIELDS - {
    "trial_id",
    "source_binding",
    "source_binding_sha256",
    "paired_common_mask_sha256",
})
_PRODUCTION_EXPECTED_FIELDS = frozenset(
    {
        *ACTION_PRIMITIVE_ARM_FIELDS,
        "support_start",
        "support_range",
        "trial_id",
        "source_binding_sha256",
        "paired_common_mask_sha256",
    }
)
P1_ACTION_PRODUCTION_METADATA_FIELDS: tuple[str, ...] = (
    *ACTION_PRIMITIVE_ARM_FIELDS,
    "support_start",
    "support_range",
    "trial_id",
    "source_binding_sha256",
    "paired_common_mask_sha256",
)

# ``source_binding`` is the exact, canonical projection of the sealed
# ForecastActionSource consumed by the action producer.  Keep this contract
# here as a public constant so the producer and persistence boundary cannot
# silently drift.  The nested source_hashes set is checked separately because
# S3 carries one additional registered body digest.
P1_ACTION_SOURCE_BINDING_SCHEMA_ID = "p1-forecast-action-source-binding-v1"
P1_ACTION_SOURCE_BINDING_ROLE = "authenticated_p1_forecast_action_source"
P1_ACTION_SOURCE_BINDING_FIELDS: tuple[str, ...] = (
    "schema_id",
    "source_role",
    "scenario_id",
    "arm",
    "seed",
    "model_id",
    "split_id",
    "support_id",
    "support_range",
    "fit_origin",
    "prereg_results_observed",
    "validation_results_observed",
    "outer_results_observed",
    "validation_status",
    "promotion_allowed",
    "capability_binding_sha256",
    "source_hashes",
)
P1_ACTION_SOURCE_ARRAY_HASH_FIELDS: tuple[str, ...] = (
    "support_timestamps_sha256",
    "realized_returns_sha256",
    "forecast_h4_sha256",
    "forecast_h4_mask_sha256",
    "forecast_fit_record_sha256",
    "context_mask_sha256",
    "target_h4_mask_sha256",
    "score_eligible_mask_sha256",
    "origin_mask_sha256",
    "score_mask_sha256",
    "common_mask_sha256",
    "spot_bar_observed_sha256",
    "body.timestamps",
    "body.features",
    "body.returns",
    "body.availability.spot_bar_observed",
    "body.availability.funding_rate_available",
    "body.availability.mark_close_available",
)
P1_ACTION_SOURCE_SCALAR_HASH_FIELDS: tuple[str, ...] = (
    "forecast_file_sha256",
    "manifest_sha256",
    "trial_registry_sha256",
    "comparison_registry_sha256",
)
P1_ACTION_SOURCE_BINDING_HASH_FIELDS: tuple[str, ...] = (
    *P1_ACTION_SOURCE_SCALAR_HASH_FIELDS,
    *P1_ACTION_SOURCE_ARRAY_HASH_FIELDS,
    "source_body_sha256",
)
_ACTION_MBB_UTILITY_FIELDS = frozenset(
    {
        "candidate_utility",
        "benchmark_hold_utility",
        "same_state_local_hold_utility",
    }
)
_ACTION_MBB_ACTION_FIELDS = frozenset(
    {"clairvoyant_utility", "regret", "opportunity", "agreement"}
)
_ACTION_MBB_METRIC_FIELDS = frozenset(
    {*_ACTION_MBB_UTILITY_FIELDS, *_ACTION_MBB_ACTION_FIELDS}
)
_ACTION_MBB_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "policy_utility_delta": ("candidate_utility", "benchmark_hold_utility"),
    "normalized_regret": ("regret", "opportunity"),
    "agreement": ("agreement",),
}


def _freeze_json_value(value: Any, *, name: str) -> Any:
    """Copy a JSON-shaped value into an immutable, deterministic projection."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise P1ActionArtifactError(f"{name} mapping keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json_value(item, name=f"{name}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (bool, np.bool_, str)) or value is None:
        return bool(value) if isinstance(value, (bool, np.bool_)) else value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if not np.isfinite(converted):
            raise P1ActionArtifactError(f"{name} must contain finite numbers")
        return converted
    raise P1ActionArtifactError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _freeze_artifact_value(value: Any, *, name: str) -> Any:
    """Freeze decoded artifact data while preserving canonical metric NaNs."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise P1ActionArtifactError(f"{name} mapping keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_artifact_value(item, name=f"{name}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_artifact_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if np.isinf(converted):
            raise P1ActionArtifactError(f"{name} must not contain infinity")
        return converted
    if isinstance(value, str) or value is None:
        return value
    raise P1ActionArtifactError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _array_sha256(value: np.ndarray, *, name: str) -> str:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise P1ActionArtifactError(f"{name} must be a one-dimensional array")
    descriptor = f"{value.dtype.str}:{value.shape[0]}".encode("ascii")
    return hashlib.sha256(descriptor + b"\x00" + value.tobytes(order="C")).hexdigest()


def _bool_mask_sha256(value: Sequence[Any], *, name: str) -> str:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ActionArtifactError(f"{name} is not a valid bool mask") from exc
    if raw.ndim != 1 or raw.dtype != np.dtype(np.bool_):
        raise P1ActionArtifactError(f"{name} must be a one-dimensional bool mask")
    return hashlib.sha256(np.ascontiguousarray(raw, dtype=np.bool_).tobytes(order="C")).hexdigest()


def _canonical_json_value(value: Any, *, name: str) -> Any:
    """Return a finite, JSON-compatible value without stringifying types."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise P1ActionArtifactError(f"{name} mapping keys must be strings")
        return {
            key: _canonical_json_value(item, name=f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if not np.isfinite(converted):
            raise P1ActionArtifactError(f"{name} must contain finite numbers")
        return converted
    if isinstance(value, str) or value is None:
        return value
    raise P1ActionArtifactError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any, *, name: str) -> bytes:
    try:
        return json.dumps(
            _canonical_json_value(value, name=name),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise P1ActionArtifactError(f"{name} is not canonical JSON") from exc


def _source_hash_fields_for_scenario(scenario_id: str) -> frozenset[str]:
    fields = set(P1_ACTION_SOURCE_BINDING_HASH_FIELDS)
    if scenario_id != "S3":
        fields.remove("source_body_sha256")
    return frozenset(fields)


def _strict_source_binding(
    value: Mapping[str, Any] | None,
    *,
    name: str = "source_binding",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P1ActionArtifactError(f"{name} must be a mapping")
    if set(value) != set(P1_ACTION_SOURCE_BINDING_FIELDS):
        missing = sorted(set(P1_ACTION_SOURCE_BINDING_FIELDS) - set(value))
        extra = sorted(set(value) - set(P1_ACTION_SOURCE_BINDING_FIELDS))
        raise P1ActionArtifactError(
            f"{name} fields are not exact (missing={missing}, extra={extra})"
        )
    schema_id = value["schema_id"]
    if not isinstance(schema_id, str) or not schema_id:
        raise P1ActionArtifactError(f"{name}.schema_id must be non-empty text")
    source_role = value["source_role"]
    if not isinstance(source_role, str) or not source_role:
        raise P1ActionArtifactError(f"{name}.source_role must be non-empty text")
    for field_name in (
        "scenario_id",
        "arm",
        "model_id",
        "split_id",
        "support_id",
    ):
        field_value = value[field_name]
        if not isinstance(field_value, str) or not field_value:
            raise P1ActionArtifactError(
                f"{name}.{field_name} must be non-empty text"
            )
    seed = value["seed"]
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise P1ActionArtifactError(f"{name}.seed must be an integer")
    support_range = value["support_range"]
    if (
        not isinstance(support_range, (list, tuple))
        or len(support_range) != 2
        or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            for item in support_range
        )
        or int(support_range[1]) <= int(support_range[0])
    ):
        raise P1ActionArtifactError(
            f"{name}.support_range must be an increasing two-integer range"
        )
    fit_origin = value["fit_origin"]
    if isinstance(fit_origin, (bool, np.bool_)) or not isinstance(
        fit_origin, (int, np.integer)
    ):
        raise P1ActionArtifactError(f"{name}.fit_origin must be an integer")
    for field_name in (
        "prereg_results_observed",
        "validation_results_observed",
        "outer_results_observed",
        "promotion_allowed",
    ):
        if not isinstance(value[field_name], (bool, np.bool_)):
            raise P1ActionArtifactError(f"{name}.{field_name} must be bool")
    validation_status = value["validation_status"]
    if not isinstance(validation_status, str) or not validation_status:
        raise P1ActionArtifactError(
            f"{name}.validation_status must be non-empty text"
        )
    _strict_sha256(
        value["capability_binding_sha256"],
        name=f"{name}.capability_binding_sha256",
    )
    source_hashes = value["source_hashes"]
    if not isinstance(source_hashes, Mapping):
        raise P1ActionArtifactError(f"{name}.source_hashes must be a mapping")
    expected_hash_fields = _source_hash_fields_for_scenario(value["scenario_id"])
    if set(source_hashes) != expected_hash_fields:
        missing = sorted(expected_hash_fields - set(source_hashes))
        extra = sorted(set(source_hashes) - expected_hash_fields)
        raise P1ActionArtifactError(
            f"{name}.source_hashes fields are not exact "
            f"(missing={missing}, extra={extra})"
        )
    for field_name, digest in source_hashes.items():
        _strict_sha256(digest, name=f"{name}.source_hashes.{field_name}")
    # Keep JSON list types (notably support_range) in the canonical mapping:
    # the upstream validator compares this mapping directly with its derived
    # source binding.  Immutable copies are made only for capability
    # provenance below.
    return {
        "schema_id": schema_id,
        "source_role": source_role,
        "scenario_id": value["scenario_id"],
        "arm": value["arm"],
        "seed": int(seed),
        "model_id": value["model_id"],
        "split_id": value["split_id"],
        "support_id": value["support_id"],
        "support_range": [int(item) for item in support_range],
        "fit_origin": int(fit_origin),
        "prereg_results_observed": bool(value["prereg_results_observed"]),
        "validation_results_observed": bool(value["validation_results_observed"]),
        "outer_results_observed": bool(value["outer_results_observed"]),
        "validation_status": validation_status,
        "promotion_allowed": bool(value["promotion_allowed"]),
        "capability_binding_sha256": value["capability_binding_sha256"],
        "source_hashes": {
            str(field_name): digest for field_name, digest in source_hashes.items()
        },
    }


def _source_binding_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(value, name="source_binding")
    ).hexdigest()


def _strict_expected_hashes(
    expected_hashes: Mapping[str, Any] | None,
    *,
    expected_action_primitive_schema_sha256: str | None,
    expected_action_primitive_content_sha256: str | None,
    expected_action_primitive_payload_sha256: str | None,
    require_production: bool,
) -> Mapping[str, str] | None:
    names = tuple(ACTION_PRIMITIVE_HASH_FIELDS)
    individual = {
        names[0]: expected_action_primitive_payload_sha256,
        names[1]: expected_action_primitive_schema_sha256,
        names[2]: expected_action_primitive_content_sha256,
    }
    if expected_hashes is not None:
        if not isinstance(expected_hashes, Mapping):
            raise P1ActionArtifactError("expected_hashes must be a mapping")
        if set(expected_hashes) != set(names):
            missing = sorted(set(names) - set(expected_hashes))
            extra = sorted(set(expected_hashes) - set(names))
            raise P1ActionArtifactError(
                "expected_hashes fields are not exactly the three action hashes "
                f"(missing={missing}, extra={extra})"
            )
        for field_name, supplied in individual.items():
            if supplied is not None and supplied != expected_hashes[field_name]:
                raise P1ActionArtifactError(
                    f"conflicting expected action hash for {field_name}"
                )
        individual = {
            field_name: expected_hashes[field_name] for field_name in names
        }
    if not any(value is not None for value in individual.values()):
        if require_production:
            raise P1ActionArtifactError(
                "production action artifacts require all three externally pinned action hashes"
            )
        return None
    if not all(value is not None for value in individual.values()):
        missing = [field_name for field_name, value in individual.items() if value is None]
        raise P1ActionArtifactError(
            "external action hashes are incomplete: " + ", ".join(missing)
        )
    return {
        field_name: _strict_sha256(value, name=f"expected {field_name}")
        for field_name, value in individual.items()
    }


@dataclass(frozen=True, eq=False)
class P1MBBMetricInput:
    """Immutable, field-aware action values for the P1 MBB boundary.

    ``effective_mask`` is derived from the persisted metric-mask registry and
    is never interchangeable with the raw ``common_mask``.  ``values`` only
    contains the fields selected by ``metric`` (for example, the two utility
    fields for ``policy_utility_delta``).
    """

    metric: str
    values: Mapping[str, np.ndarray]
    effective_mask: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def metric_values(self) -> Mapping[str, np.ndarray]:
        return self.values

    @property
    def mask(self) -> np.ndarray:
        return self.effective_mask


@dataclass(frozen=True, eq=False)
class P1MBBActionInput:
    """Immutable action metrics, effective masks, and source provenance."""

    metric_values: Mapping[str, np.ndarray]
    effective_masks: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]

    @property
    def values(self) -> Mapping[str, np.ndarray]:
        return self.metric_values

    def select_metric(self, metric: str) -> P1MBBMetricInput:
        if not isinstance(metric, str):
            raise P1ActionArtifactError("MBB metric name must be text")
        fields = _ACTION_MBB_METRIC_ALIASES.get(metric, (metric,))
        if any(field not in _ACTION_MBB_METRIC_FIELDS for field in fields):
            raise P1ActionArtifactError(f"MBB metric is not registered: {metric}")
        values = MappingProxyType(
            {field: self.metric_values[field] for field in fields}
        )
        effective_mask = self.effective_masks[fields[0]]
        if any(
            not np.array_equal(self.effective_masks[field], effective_mask)
            for field in fields[1:]
        ):
            raise P1ActionArtifactError(f"MBB metric {metric} has conflicting field-specific masks")
        return P1MBBMetricInput(
            metric=metric,
            values=values,
            effective_mask=effective_mask,
            provenance=self.provenance,
        )

    def for_metric(self, metric: str) -> P1MBBMetricInput:
        return self.select_metric(metric)

    def mask_for(self, field: str) -> np.ndarray:
        try:
            return self.effective_masks[field]
        except KeyError as exc:
            raise P1ActionArtifactError(
                f"MBB metric field is not registered: {field}"
            ) from exc


@dataclass(frozen=True, eq=False)
class LoadedP1ActionArtifact:
    # Weak-key authentication is identity based.  A direct constructor or
    # ``dataclasses.replace`` therefore cannot inherit a loader's authority.
    __hash__ = object.__hash__

    path: Path
    file_sha256: str
    artifact: Mapping[str, Any]
    validation: Mapping[str, Any]
    _production_seal: object | None = field(default=None, repr=False, compare=False)
    _binding_sha256: str | None = field(default=None, repr=False, compare=False)
    _mbb_input: P1MBBActionInput | None = field(default=None, repr=False, compare=False)

    @property
    def is_authenticated(self) -> bool:
        return _is_registered_loaded_action_artifact(self)

    @property
    def mbb_input(self) -> P1MBBActionInput:
        if not self.is_authenticated or self._mbb_input is None:
            raise P1ActionArtifactError(
                "MBB input is available only from an authenticated production action load"
            )
        return self._mbb_input

    def as_mbb_input(self) -> P1MBBActionInput:
        return self.mbb_input


class _P1ActionArtifactSeal:
    pass


_P1_ACTION_ARTIFACT_SEAL = _P1ActionArtifactSeal()
_AUTHENTICATED_P1_ACTION_ARTIFACTS: weakref.WeakKeyDictionary[
    LoadedP1ActionArtifact, str
] = weakref.WeakKeyDictionary()


def _strict_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P1ActionArtifactError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _strict_expected_metadata(
    value: Mapping[str, Any] | None,
    *,
    require_production: bool,
) -> Mapping[str, Any] | None:
    if value is None:
        if require_production:
            raise P1ActionArtifactError(
                "production action artifacts require externally supplied expected_metadata"
            )
        return None
    if not isinstance(value, Mapping):
        raise P1ActionArtifactError("expected_metadata must be a mapping")
    if require_production and set(value) != _PRODUCTION_EXPECTED_FIELDS:
        missing = sorted(_PRODUCTION_EXPECTED_FIELDS - set(value))
        extra = sorted(set(value) - _PRODUCTION_EXPECTED_FIELDS)
        raise P1ActionArtifactError(
            "production expected_metadata fields are not exact "
            f"(missing={missing}, extra={extra})"
        )
    if require_production:
        trial_id = value.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise P1ActionArtifactError(
                "production expected_metadata.trial_id must be non-empty text"
            )
        _strict_sha256(
            value.get("source_binding_sha256"),
            name="expected_metadata.source_binding_sha256",
        )
        _strict_sha256(
            value.get("paired_common_mask_sha256"),
            name="expected_metadata.paired_common_mask_sha256",
        )
    return value


def _require_production_sources(
    *,
    require_production: bool,
    realized_returns: Sequence[Any] | None,
    decision_block_scores: Sequence[Any] | None,
    decision_deltas: Sequence[Any] | None,
    decision_eligible: Sequence[Any] | None,
    score_eligible: Sequence[Any] | None,
    expected_common_mask: Sequence[Any] | None,
) -> None:
    if not require_production:
        return
    missing: list[str] = []
    if realized_returns is None:
        missing.append("realized_returns")
    if decision_block_scores is None and decision_deltas is None:
        missing.append("decision_block_scores_or_decision_deltas")
    if decision_eligible is None:
        missing.append("decision_eligible")
    if score_eligible is None:
        missing.append("score_eligible")
    if expected_common_mask is None:
        missing.append("expected_common_mask")
    if missing:
        raise P1ActionArtifactError(
            "production action artifact validation is missing external sources: "
            + ", ".join(missing)
        )


def _require_authenticated_forecast_source(value: Any) -> Any:
    """Require the identity-sealed source capability owned by the forecast loader."""
    try:
        from .p1_validation_forecast import require_authenticated_forecast_action_source

        return require_authenticated_forecast_action_source(value)
    except Exception as exc:
        raise P1ActionArtifactError(
            "production action artifacts require the sealed ForecastActionSource capability"
        ) from exc


def _validate_source_binding_against_capability(
    source_binding: Mapping[str, Any],
    source: Any,
) -> None:
    """Check every source binding field against the authenticated forecast source."""
    try:
        from .action_primitives import ACTION_PRIMITIVE_SOURCE_BINDING_SCHEMA_ID
    except ImportError:  # pragma: no cover - compatibility with pre-chain branch
        ACTION_PRIMITIVE_SOURCE_BINDING_SCHEMA_ID = P1_ACTION_SOURCE_BINDING_SCHEMA_ID
    if source_binding["schema_id"] != ACTION_PRIMITIVE_SOURCE_BINDING_SCHEMA_ID:
        raise P1ActionArtifactError(
            "source_binding.schema_id is not the registered source-binding schema"
        )
    if source_binding["source_role"] != P1_ACTION_SOURCE_BINDING_ROLE:
        raise P1ActionArtifactError(
            "source_binding.source_role is not the registered action-source role"
        )
    scalar_fields = (
        "scenario_id",
        "arm",
        "seed",
        "split_id",
        "support_id",
        "fit_origin",
        "prereg_results_observed",
        "validation_results_observed",
        "outer_results_observed",
        "validation_status",
        "promotion_allowed",
    )
    for field_name in scalar_fields:
        if source_binding[field_name] != getattr(source, field_name):
            raise P1ActionArtifactError(
                f"source_binding.{field_name} does not match the authenticated forecast source"
            )
    if tuple(source_binding["support_range"]) != tuple(source.support_range):
        raise P1ActionArtifactError(
            "source_binding.support_range does not match the authenticated forecast source"
        )
    # The current ForecastActionSource capability is selected from the fixed
    # h4/ridge continuous forecast.  A future producer may expose this as a
    # property; until then this fixed source identity is the only registered
    # model identity accepted here.
    source_model_id = getattr(source, "model_id", "ridge")
    if source_binding["model_id"] != source_model_id:
        raise P1ActionArtifactError(
            "source_binding.model_id does not match the authenticated forecast source"
        )
    if source_binding["capability_binding_sha256"] != source.binding_sha256:
        raise P1ActionArtifactError(
            "source_binding.capability_binding_sha256 does not match the authenticated forecast source"
        )
    if dict(source_binding["source_hashes"]) != dict(source.source_hashes):
        raise P1ActionArtifactError(
            "source_binding.source_hashes do not match the authenticated forecast source"
        )


def _validate_source_binding(
    artifact: Mapping[str, Any],
    *,
    expected_source_binding: Mapping[str, Any] | None,
    expected_metadata: Mapping[str, Any] | None,
    authenticated_action_source: Any,
    require_production: bool,
) -> Mapping[str, Any] | None:
    header = artifact.get("header")
    if not isinstance(header, Mapping):
        raise P1ActionArtifactError("action artifact header is required")
    actual = header.get("source_binding")
    if actual is None:
        if require_production:
            raise P1ActionArtifactError(
                "production action artifacts require the canonical source_binding header"
            )
        if expected_source_binding is not None:
            raise P1ActionArtifactError(
                "expected_source_binding cannot bind an artifact without source_binding"
            )
        return None
    actual_binding = _strict_source_binding(actual)
    actual_digest = _source_binding_sha256(actual_binding)
    declared_digest = _strict_sha256(
        header.get("source_binding_sha256"),
        name="header.source_binding_sha256",
    )
    if declared_digest != actual_digest:
        raise P1ActionArtifactError(
            "header.source_binding_sha256 does not match source_binding"
        )
    if expected_source_binding is None:
        if require_production:
            raise P1ActionArtifactError(
                "production action artifacts require externally pinned expected_source_binding"
            )
        return actual_binding
    expected_binding = _strict_source_binding(
        expected_source_binding,
        name="expected_source_binding",
    )
    expected_digest = _source_binding_sha256(expected_binding)
    if actual_digest != expected_digest or dict(actual_binding) != dict(expected_binding):
        raise P1ActionArtifactError(
            "action artifact source_binding does not match the externally pinned binding"
        )
    if expected_metadata is not None and expected_metadata.get("source_binding_sha256") != expected_digest:
        raise P1ActionArtifactError(
            "expected_metadata.source_binding_sha256 does not match expected_source_binding"
        )
    if require_production:
        source = _require_authenticated_forecast_source(authenticated_action_source)
        _validate_source_binding_against_capability(actual_binding, source)
    return actual_binding


def _validate_raw_sources_against_capability(
    source: Any,
    *,
    realized_returns: Sequence[Any] | None,
    decision_block_scores: Sequence[Any] | None,
    decision_deltas: Sequence[Any] | None,
    decision_eligible: Sequence[Any] | None,
    score_eligible: Sequence[Any] | None,
) -> None:
    """Prevent caller arrays from replacing the authenticated source arrays."""
    if decision_deltas is not None:
        raise P1ActionArtifactError(
            "production action validation cannot accept caller-selected decision deltas"
        )
    pairs = (
        ("realized_returns", realized_returns, source.realized_returns),
        ("decision_block_scores", decision_block_scores, source.forecast_h4),
        ("decision_eligible", decision_eligible, source.origin_mask),
        ("score_eligible", score_eligible, source.bar_available),
    )
    for name, supplied, authenticated in pairs:
        if supplied is None:
            raise P1ActionArtifactError(
                f"production action validation is missing authenticated {name}"
            )
        supplied_array = np.asarray(supplied)
        authenticated_array = np.asarray(authenticated)
        if (
            supplied_array.shape != authenticated_array.shape
            or supplied_array.dtype != authenticated_array.dtype
            or not np.array_equal(supplied_array, authenticated_array, equal_nan=True)
        ):
            raise P1ActionArtifactError(
                f"production {name} differs from the sealed forecast capability"
            )


def _semantic_validate(
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None,
    expected_hashes: Mapping[str, Any] | None,
    expected_output_hashes: Mapping[str, Any] | None,
    expected_action_primitive_schema_sha256: str | None,
    expected_action_primitive_content_sha256: str | None,
    expected_action_primitive_payload_sha256: str | None,
    expected_source_binding: Mapping[str, Any] | None,
    authenticated_action_source: Any,
    realized_returns: Sequence[Any] | None,
    decision_block_scores: Sequence[Any] | None,
    decision_deltas: Sequence[Any] | None,
    decision_eligible: Sequence[Any] | None,
    score_eligible: Sequence[Any] | None,
    expected_common_mask: Sequence[Any] | None,
    require_production: bool,
    require_external_hashes: bool,
) -> Mapping[str, Any]:
    if expected_hashes is not None and expected_output_hashes is not None:
        raise P1ActionArtifactError(
            "expected_hashes and expected_output_hashes are aliases and cannot both be supplied"
        )
    if expected_hashes is None:
        expected_hashes = expected_output_hashes
    expected = _strict_expected_metadata(
        expected_metadata,
        require_production=require_production,
    )
    normalized_hashes = _strict_expected_hashes(
        expected_hashes,
        expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
        expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
        expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
        require_production=require_external_hashes,
    )
    if normalized_hashes is not None:
        for field_name, expected_digest in normalized_hashes.items():
            actual_digest = artifact.get(field_name)
            if actual_digest != expected_digest:
                raise P1ActionArtifactError(
                    f"action artifact {field_name} does not match its external expected digest"
                )
    source_binding = _validate_source_binding(
        artifact,
        expected_source_binding=expected_source_binding,
        expected_metadata=expected,
        authenticated_action_source=authenticated_action_source,
        require_production=require_production,
    )
    _require_production_sources(
        require_production=require_production,
        realized_returns=realized_returns,
        decision_block_scores=decision_block_scores,
        decision_deltas=decision_deltas,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        expected_common_mask=expected_common_mask,
    )
    if require_production:
        header = artifact["header"]
        if header.get("trial_id") != expected["trial_id"]:
            raise P1ActionArtifactError(
                "action artifact trial_id does not match expected_metadata"
            )
        declared_mask_digest = _strict_sha256(
            header.get("paired_common_mask_sha256"),
            name="header.paired_common_mask_sha256",
        )
        if declared_mask_digest != expected["paired_common_mask_sha256"]:
            raise P1ActionArtifactError(
                "header.paired_common_mask_sha256 does not match expected_metadata"
            )
        expected_mask_digest = expected["paired_common_mask_sha256"]
        actual_mask_digest = _bool_mask_sha256(
            expected_common_mask,
            name="expected_common_mask",
        )
        if actual_mask_digest != expected_mask_digest:
            raise P1ActionArtifactError(
                "expected_common_mask does not match expected_metadata.paired_common_mask_sha256"
            )
        source = _require_authenticated_forecast_source(authenticated_action_source)
        _validate_raw_sources_against_capability(
            source,
            realized_returns=realized_returns,
            decision_block_scores=decision_block_scores,
            decision_deltas=decision_deltas,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
    if require_production and authenticated_action_source is None:
        raise P1ActionArtifactError(
            "production action artifacts require authenticated_action_source"
        )
    validator_kwargs: dict[str, Any] = {
        "expected_metadata": expected,
        "realized_returns": realized_returns,
        "decision_block_scores": decision_block_scores,
        "decision_deltas": decision_deltas,
        "decision_eligible": decision_eligible,
        "score_eligible": score_eligible,
        "expected_common_mask": expected_common_mask,
        "require_production": require_production,
    }
    if normalized_hashes is not None:
        validator_kwargs["expected_output_hashes"] = normalized_hashes
    # The action primitive validator gains the sealed forecast source on the
    # production-chain branch.  Keep fixture compatibility with this commit's
    # earlier validator signature, but never use that fallback without the
    # independent source-binding checks above.
    try:
        validator_parameters = inspect.signature(
            validate_action_primitive_semantics
        ).parameters
    except (TypeError, ValueError):  # pragma: no cover - normal Python function
        validator_parameters = {}
    if "authenticated_action_source" in validator_parameters:
        validator_kwargs["authenticated_action_source"] = authenticated_action_source
    if "expected_source_binding" in validator_parameters:
        validator_kwargs["expected_source_binding"] = source_binding
    if require_production and "authenticated_action_source" not in validator_parameters:
        validator_kwargs["require_production"] = False
    try:
        return validate_action_primitive_semantics(artifact, **validator_kwargs)
    except (ActionPrimitiveContractError, TypeError, ValueError, OverflowError) as exc:
        raise P1ActionArtifactError("action primitive semantic validation failed") from exc


def _validate_header_shape(
    header: Mapping[str, Any],
    *,
    require_production: bool,
) -> None:
    if not isinstance(header, Mapping):
        raise P1ActionArtifactError("action artifact header must be a mapping")
    fields = set(header)
    if fields == _HEADER_FIELDS:
        return
    if not require_production and fields == _LEGACY_FIXTURE_HEADER_FIELDS:
        return
    missing = sorted(_HEADER_FIELDS - fields)
    extra = sorted(fields - _HEADER_FIELDS)
    raise P1ActionArtifactError(
        "action artifact header fields are not exact "
        f"(missing={missing}, extra={extra})"
    )


def _columnar_payload(
    artifact: Mapping[str, Any],
    *,
    require_production: bool,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise P1ActionArtifactError("action artifact must be a mapping")
    header = artifact.get("header")
    records = artifact.get("records")
    if not isinstance(header, Mapping) or not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        raise P1ActionArtifactError("action artifact header/records are malformed")
    _validate_header_shape(header, require_production=require_production)
    if not 0 < len(records) <= P1_ACTION_FILE_MAX_RECORDS:
        raise P1ActionArtifactError("action artifact record count is outside its bound")
    columns: dict[str, list[Any]] = {field: [] for field in ACTION_PRIMITIVE_RECORD_FIELDS}
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping) or tuple(record) != ACTION_PRIMITIVE_RECORD_FIELDS:
            raise P1ActionArtifactError(
                f"action artifact row {row_index} does not retain canonical field order"
            )
        for field in ACTION_PRIMITIVE_RECORD_FIELDS:
            value = record[field]
            if field in ACTION_PRIMITIVE_METRIC_FIELDS and isinstance(value, float):
                # JSON has no NaN.  Null is reserved solely for canonical
                # unscored metric cells and is restored before semantic checks.
                if value != value:
                    value = None
            columns[field].append(value)
    result: dict[str, Any] = {
        "format": P1_ACTION_FILE_FORMAT,
        "format_version": P1_ACTION_FILE_VERSION,
        "header": dict(header),
        "record_fields": list(ACTION_PRIMITIVE_RECORD_FIELDS),
        "record_count": len(records),
        "columns": columns,
    }
    for field in ACTION_PRIMITIVE_HASH_FIELDS:
        result[field] = artifact.get(field)
    return result


def _canonical_file_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise P1ActionArtifactError("action artifact is not canonical JSON") from exc
    if len(encoded) > P1_ACTION_FILE_MAX_BYTES:
        raise P1ActionArtifactError("action artifact exceeds the file-size bound")
    return encoded


def save_p1_action_artifact(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    expected_hashes: Mapping[str, Any] | None = None,
    expected_output_hashes: Mapping[str, Any] | None = None,
    expected_action_primitive_schema_sha256: str | None = None,
    expected_action_primitive_content_sha256: str | None = None,
    expected_action_primitive_payload_sha256: str | None = None,
    expected_source_binding: Mapping[str, Any] | None = None,
    authenticated_action_source: Any = None,
    realized_returns: Sequence[Any] | None = None,
    decision_block_scores: Sequence[Any] | None = None,
    decision_deltas: Sequence[Any] | None = None,
    decision_eligible: Sequence[Any] | None = None,
    score_eligible: Sequence[Any] | None = None,
    expected_common_mask: Sequence[Any] | None = None,
    require_production: bool = True,
) -> str:
    """Validate and atomically persist one action artifact; return file SHA-256."""
    _semantic_validate(
        artifact,
        expected_metadata=expected_metadata,
        expected_hashes=expected_hashes,
        expected_output_hashes=expected_output_hashes,
        expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
        expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
        expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
        expected_source_binding=expected_source_binding,
        authenticated_action_source=authenticated_action_source,
        realized_returns=realized_returns,
        decision_block_scores=decision_block_scores,
        decision_deltas=decision_deltas,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        expected_common_mask=expected_common_mask,
        require_production=require_production,
        require_external_hashes=False,
    )
    encoded = _canonical_file_bytes(
        _columnar_payload(artifact, require_production=require_production)
    )
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise P1ActionArtifactError("action artifact parent directory does not exist")
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
        raise P1ActionArtifactError(f"could not persist action artifact {target}") from exc
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
    try:
        _, post_write_digest = _read_regular_file(target)
    except P1ActionArtifactError:
        raise
    return post_write_digest


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P1ActionArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular_file(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise P1ActionArtifactError(f"could not stat action artifact {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise P1ActionArtifactError("action artifact must be a regular non-symlink file")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise P1ActionArtifactError("action artifact must remain a regular file")
        if opened.st_size > P1_ACTION_FILE_MAX_BYTES:
            raise P1ActionArtifactError("action artifact exceeds the file-size bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise P1ActionArtifactError("action artifact ended during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        signature_before = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if signature_before != signature_after:
            raise P1ActionArtifactError("action artifact changed during read")
        encoded = b"".join(chunks)
    except OSError as exc:
        raise P1ActionArtifactError(f"could not read action artifact {path}") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return encoded, hashlib.sha256(encoded).hexdigest()


def _decode_payload(
    payload: Any,
    *,
    require_production: bool,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise P1ActionArtifactError("stored action artifact top-level fields are not exact")
    if payload.get("format") != P1_ACTION_FILE_FORMAT:
        raise P1ActionArtifactError("stored action artifact format is unsupported")
    if payload.get("format_version") != P1_ACTION_FILE_VERSION:
        raise P1ActionArtifactError("stored action artifact version is unsupported")
    record_fields = payload.get("record_fields")
    if record_fields != list(ACTION_PRIMITIVE_RECORD_FIELDS):
        raise P1ActionArtifactError("stored action artifact record fields are not canonical")
    count = payload.get("record_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= P1_ACTION_FILE_MAX_RECORDS:
        raise P1ActionArtifactError("stored action artifact record count is invalid")
    columns = payload.get("columns")
    if (
        not isinstance(columns, Mapping)
        or any(not isinstance(field, str) for field in columns)
        or set(columns) != set(ACTION_PRIMITIVE_RECORD_FIELDS)
    ):
        raise P1ActionArtifactError("stored action artifact columns are not canonical")
    for field in ACTION_PRIMITIVE_RECORD_FIELDS:
        values = columns[field]
        if not isinstance(values, list) or len(values) != count:
            raise P1ActionArtifactError(f"stored action column {field} is not row-aligned")
    records: list[dict[str, Any]] = []
    for row_index in range(count):
        record: dict[str, Any] = {}
        for field in ACTION_PRIMITIVE_RECORD_FIELDS:
            value = columns[field][row_index]
            if value is None:
                if field not in ACTION_PRIMITIVE_METRIC_FIELDS:
                    raise P1ActionArtifactError(
                        f"null is forbidden for {field} at row {row_index}"
                    )
                value = float("nan")
            record[field] = value
        records.append(record)
    header = payload.get("header")
    if not isinstance(header, Mapping) or any(
        not isinstance(field, str) for field in header
    ):
        raise P1ActionArtifactError("stored action artifact header fields are not exact")
    _validate_header_shape(header, require_production=require_production)
    artifact: dict[str, Any] = {"header": dict(header), "records": records}
    for field in ACTION_PRIMITIVE_HASH_FIELDS:
        artifact[field] = _strict_sha256(payload.get(field), name=field)
    return artifact


def _read_only_vector(
    values: Sequence[Any],
    *,
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=dtype).copy(order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1ActionArtifactError(f"{name} cannot be represented as {dtype}") from exc
    if result.ndim != 1:
        raise P1ActionArtifactError(f"{name} must be a one-dimensional vector")
    if np.issubdtype(result.dtype, np.floating) and np.isinf(result).any():
        raise P1ActionArtifactError(f"{name} must not contain infinity")
    result.setflags(write=False)
    return result


def _build_mbb_input(
    artifact: Mapping[str, Any],
    *,
    file_sha256: str,
) -> P1MBBActionInput:
    header = artifact.get("header")
    records = artifact.get("records")
    if not isinstance(header, Mapping) or not isinstance(records, Sequence):
        raise P1ActionArtifactError("action artifact is missing MBB input fields")
    registry = header.get("metric_mask_registry")
    if not isinstance(registry, Mapping) or dict(registry) != dict(
        ACTION_PRIMITIVE_METRIC_MASK_REGISTRY
    ):
        raise P1ActionArtifactError(
            "action artifact metric_mask_registry does not match the fixed registry"
        )
    common_mask = _read_only_vector(
        [record["common_mask"] for record in records],
        dtype=np.dtype(np.bool_),
        name="common_mask",
    )
    outcome_mask = _read_only_vector(
        [record["outcome_complete_mask"] for record in records],
        dtype=np.dtype(np.bool_),
        name="outcome_complete_mask",
    )
    scored_mask = _read_only_vector(
        [record["scored_action_mask"] for record in records],
        dtype=np.dtype(np.bool_),
        name="scored_action_mask",
    )
    utility_mask = np.logical_and(outcome_mask, common_mask)
    utility_mask.setflags(write=False)
    action_mask = np.logical_and(scored_mask, common_mask)
    action_mask.setflags(write=False)
    metric_values: dict[str, np.ndarray] = {}
    effective_masks: dict[str, np.ndarray] = {}
    for field_name in _ACTION_MBB_METRIC_FIELDS:
        metric_values[field_name] = _read_only_vector(
            [record[field_name] for record in records],
            dtype=np.dtype("<f8"),
            name=field_name,
        )
        effective_masks[field_name] = (
            utility_mask if field_name in _ACTION_MBB_UTILITY_FIELDS else action_mask
        )
    source_binding = header.get("source_binding")
    provenance: dict[str, Any] = {
        "file_sha256": _strict_sha256(file_sha256, name="file_sha256"),
        "trial_id": header.get("trial_id"),
        "support_range": tuple(header.get("support_range", ())),
        "metric_mask_registry": dict(registry),
        "effective_mask_fields": {
            field_name: (
                "utility_metrics"
                if field_name in _ACTION_MBB_UTILITY_FIELDS
                else "action_metrics"
            )
            for field_name in _ACTION_MBB_METRIC_FIELDS
        },
    }
    for field_name in ACTION_PRIMITIVE_HASH_FIELDS:
        provenance[field_name] = _strict_sha256(
            artifact.get(field_name),
            name=field_name,
        )
    if source_binding is not None:
        provenance["source_binding"] = _strict_source_binding(source_binding)
        provenance["source_binding_sha256"] = _strict_sha256(
            header.get("source_binding_sha256"),
            name="source_binding_sha256",
        )
    frozen_provenance = _freeze_json_value(provenance, name="MBB provenance")
    return P1MBBActionInput(
        metric_values=MappingProxyType(metric_values),
        effective_masks=MappingProxyType(effective_masks),
        provenance=frozen_provenance,
    )


def _loaded_binding_sha256(value: LoadedP1ActionArtifact) -> str:
    input_value = value._mbb_input
    if input_value is None:
        raise P1ActionArtifactError("loaded action artifact lacks its MBB input")
    payload = {
        "file_sha256": value.file_sha256,
        "artifact_hashes": {
            field_name: value.artifact[field_name]
            for field_name in ACTION_PRIMITIVE_HASH_FIELDS
        },
        "header": value.artifact["header"],
        "metric_values": {
            field_name: _array_sha256(array, name=field_name)
            for field_name, array in input_value.metric_values.items()
        },
        "effective_masks": {
            field_name: _array_sha256(array, name=f"{field_name}.mask")
            for field_name, array in input_value.effective_masks.items()
        },
        "provenance": input_value.provenance,
    }
    return hashlib.sha256(
        _canonical_json_bytes(payload, name="authenticated action binding")
    ).hexdigest()


def _is_registered_loaded_action_artifact(value: Any) -> bool:
    """Check identity registration and rederived binding, not just the type."""
    if not isinstance(value, LoadedP1ActionArtifact):
        return False
    if value._production_seal is not _P1_ACTION_ARTIFACT_SEAL:
        return False
    try:
        registered = _AUTHENTICATED_P1_ACTION_ARTIFACTS.get(value)
        current = _loaded_binding_sha256(value)
    except Exception:
        return False
    return (
        registered is not None
        and registered == value._binding_sha256
        and registered == current
    )


def is_authenticated_loaded_action_artifact(value: Any) -> bool:
    """Return whether ``value`` is the registered production-load capability."""
    return _is_registered_loaded_action_artifact(value)


def require_authenticated_loaded_action_artifact(
    value: Any,
) -> LoadedP1ActionArtifact:
    """Require the identity-sealed action artifact capability emitted by load."""
    if not _is_registered_loaded_action_artifact(value):
        raise P1ActionArtifactError(
            "production action input must be the identity-sealed capability from a validated action artifact load"
        )
    return value


def load_p1_action_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_metadata: Mapping[str, Any] | None = None,
    expected_hashes: Mapping[str, Any] | None = None,
    expected_output_hashes: Mapping[str, Any] | None = None,
    expected_action_primitive_schema_sha256: str | None = None,
    expected_action_primitive_content_sha256: str | None = None,
    expected_action_primitive_payload_sha256: str | None = None,
    expected_source_binding: Mapping[str, Any] | None = None,
    authenticated_action_source: Any = None,
    realized_returns: Sequence[Any] | None = None,
    decision_block_scores: Sequence[Any] | None = None,
    decision_deltas: Sequence[Any] | None = None,
    decision_eligible: Sequence[Any] | None = None,
    score_eligible: Sequence[Any] | None = None,
    expected_common_mask: Sequence[Any] | None = None,
    require_production: bool = True,
) -> LoadedP1ActionArtifact:
    """Load by externally pinned file digest and rederive every semantic field."""
    expected_digest = _strict_sha256(expected_file_sha256, name="expected_file_sha256")
    source = Path(path)
    encoded, actual_digest = _read_regular_file(source)
    if actual_digest != expected_digest:
        raise P1ActionArtifactError("stored action artifact file SHA-256 mismatch")
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                P1ActionArtifactError(f"non-finite JSON constant is forbidden: {value}")
            ),
        )
    except P1ActionArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise P1ActionArtifactError("stored action artifact JSON is malformed") from exc
    artifact = _decode_payload(payload, require_production=require_production)
    validation = _semantic_validate(
        artifact,
        expected_metadata=expected_metadata,
        expected_hashes=expected_hashes,
        expected_output_hashes=expected_output_hashes,
        expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
        expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
        expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
        expected_source_binding=expected_source_binding,
        authenticated_action_source=authenticated_action_source,
        realized_returns=realized_returns,
        decision_block_scores=decision_block_scores,
        decision_deltas=decision_deltas,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        expected_common_mask=expected_common_mask,
        require_production=require_production,
        require_external_hashes=require_production,
    )
    frozen_artifact = _freeze_artifact_value(artifact, name="loaded action artifact")
    frozen_validation = _freeze_json_value(validation, name="action validation")
    mbb_input = _build_mbb_input(
        frozen_artifact,
        file_sha256=actual_digest,
    )
    loaded = LoadedP1ActionArtifact(
        path=source,
        file_sha256=actual_digest,
        artifact=frozen_artifact,
        validation=frozen_validation,
        _mbb_input=mbb_input,
    )
    if require_production:
        binding = _loaded_binding_sha256(loaded)
        loaded = LoadedP1ActionArtifact(
            path=source,
            file_sha256=actual_digest,
            artifact=frozen_artifact,
            validation=frozen_validation,
            _production_seal=_P1_ACTION_ARTIFACT_SEAL,
            _binding_sha256=binding,
            _mbb_input=mbb_input,
        )
        _AUTHENTICATED_P1_ACTION_ARTIFACTS[loaded] = binding
    return loaded


# Discoverable compatibility aliases retain one authenticated type and one
# strict requirement function; they do not create alternate constructors.
AuthenticatedP1ActionArtifact = LoadedP1ActionArtifact
P1ActionArtifactCapability = LoadedP1ActionArtifact
require_authenticated_p1_action_artifact = require_authenticated_loaded_action_artifact
load_p1_action_artifact_production = load_p1_action_artifact


__all__ = [
    "AuthenticatedP1ActionArtifact",
    "LoadedP1ActionArtifact",
    "P1ActionArtifactCapability",
    "P1ActionArtifactError",
    "P1MBBActionInput",
    "P1MBBMetricInput",
    "P1_ACTION_PRODUCTION_METADATA_FIELDS",
    "P1_ACTION_SOURCE_ARRAY_HASH_FIELDS",
    "P1_ACTION_SOURCE_BINDING_FIELDS",
    "P1_ACTION_SOURCE_BINDING_HASH_FIELDS",
    "P1_ACTION_SOURCE_BINDING_ROLE",
    "P1_ACTION_SOURCE_BINDING_SCHEMA_ID",
    "P1_ACTION_SOURCE_SCALAR_HASH_FIELDS",
    "P1_ACTION_FILE_FORMAT",
    "P1_ACTION_FILE_MAX_BYTES",
    "P1_ACTION_FILE_MAX_RECORDS",
    "P1_ACTION_FILE_VERSION",
    "is_authenticated_loaded_action_artifact",
    "load_p1_action_artifact",
    "load_p1_action_artifact_production",
    "require_authenticated_loaded_action_artifact",
    "require_authenticated_p1_action_artifact",
    "save_p1_action_artifact",
]
