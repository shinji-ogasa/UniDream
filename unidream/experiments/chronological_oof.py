"""Leak-safe chronological OOF contracts for the conditional experiment path.

The legacy pipeline intentionally remains available for historical replay.  It
must not, however, be mistaken for a conditional teacher: its future-derived
states are fit on the complete training window and then read in-sample.  This
module provides the small, model-agnostic contract needed by a new path while
the expensive full WM re-training integration is still being staged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


class ChronologicalOOFError(ValueError):
    """Raised when an OOF request cannot satisfy the causal contract."""


class ConditionalPathBlocked(RuntimeError):
    """Raised when legacy hindsight state is requested by the new path."""


class ConditionalOOFArtifactError(ChronologicalOOFError):
    """Raised when a persisted conditional OOF artifact is not promotable.

    The raw chronological helper remains usable for legacy diagnostics.  This
    stricter error is reserved for the artifact boundary used by the new
    conditional path, where missing provenance, hash mismatches, and zero
    coverage must stop execution rather than degrade to an in-sample fallback.
    """


OOF_ARTIFACT_SCHEMA = "unidream.conditional_oof"
OOF_ARTIFACT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_DIGEST_FIELDS = {"artifact_sha256", "artifact_hash"}
_OOF_ARTIFACT_ARRAY_FIELDS = {
    "predictions",
    "prediction_mask",
    "oof_mask",
    "prediction_eligibility_mask",
    "training_label_eligibility_mask",
    "target_end_exclusive",
    "train_count",
}


@dataclass(frozen=True)
class OOFOrigin:
    """One chronological prediction origin and its admissible training prefix."""

    prediction_index: int
    train_start: int
    train_end_exclusive: int
    label_cutoff_exclusive: int
    n_train: int


def strict_bool_array(value: Any, *, name: str) -> np.ndarray:
    """Return a copy of a boolean mask without coercing other dtypes.

    Availability masks are part of the causal contract. ``np.asarray(...,
    dtype=bool)`` would silently turn integers, strings, and NaN values into
    booleans, so every mask boundary uses this helper instead.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ChronologicalOOFError(f"{name} must be a strict bool array") from exc
    if array.dtype != np.dtype(np.bool_):
        raise ChronologicalOOFError(
            f"{name} must have dtype bool; implicit coercion from {array.dtype} is forbidden"
        )
    return np.array(array, dtype=np.bool_, copy=True)


def strict_bool_value(value: Any, *, name: str) -> bool:
    """Validate a configuration boolean without accepting truthy strings."""
    if type(value) is not bool:
        raise ChronologicalOOFError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def strict_integer_value(value: Any, *, name: str) -> int:
    """Validate an integer option without accepting bool/fraction/string casts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ChronologicalOOFError(
            f"{name} must be an integer (bool, fraction, and string coercion are forbidden)"
        )
    return int(value)


def strict_integer_array(value: Any, *, name: str) -> np.ndarray:
    """Validate an integer index/cutoff array without truncating other dtypes."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ChronologicalOOFError(f"{name} must be an integer array") from exc
    if array.dtype.kind not in "iu":
        raise ChronologicalOOFError(
            f"{name} must have an integer dtype; implicit coercion from {array.dtype} is forbidden"
        )
    return np.array(array, dtype=np.int64, copy=True)


def _sha256_text(value: Any, *, name: str) -> str:
    """Validate a content/provenance SHA-256 field without coercion."""
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ConditionalOOFArtifactError(
            f"{name} must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _array_digest(value: Any, *, name: str) -> str:
    """Hash ndarray dtype, shape, and exact C-order bytes.

    NaN bytes are intentionally retained.  An unavailable NaN and a finite
    zero therefore cannot become the same artifact through serialization or
    hashing.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ConditionalOOFArtifactError(f"{name} must be an ndarray-compatible value") from exc
    if array.dtype.kind not in "biufc":
        raise ConditionalOOFArtifactError(f"{name} has unsupported dtype {array.dtype}")
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _artifact_json_value(value: Any, *, for_hash: bool = False) -> Any:
    """Convert metadata to deterministic JSON-safe values.

    Arrays in the artifact's top level are represented by a content digest
    when hashing; nested metadata is expected to be scalar/list/mapping data.
    Rejecting implicit stringification here keeps provenance hashes auditable.
    """
    if isinstance(value, np.ndarray):
        if for_hash:
            return {
                "__ndarray__": True,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": _array_digest(value, name="artifact array"),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return _artifact_json_value(value.item(), for_hash=for_hash)
    if isinstance(value, Mapping):
        return {
            str(key): _artifact_json_value(item, for_hash=for_hash)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_artifact_json_value(item, for_hash=for_hash) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            # Non-finite values are not valid provenance scalars.  Numeric
            # NaNs belong in explicitly typed ndarray payloads instead.
            raise ConditionalOOFArtifactError(
                "non-finite scalar provenance values are not supported"
            )
        return value
    raise ConditionalOOFArtifactError(
        f"unsupported artifact metadata value type: {type(value).__name__}"
    )


def hash_conditional_oof_artifact(artifact: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 over a conditional OOF artifact.

    The self-referential ``artifact_sha256``/``artifact_hash`` fields are
    excluded.  All arrays contribute dtype, shape, and exact bytes, while
    metadata is canonicalized with sorted keys.  This makes a same-shaped
    tamper (including a NaN-to-zero substitution) observable.
    """
    if not isinstance(artifact, Mapping):
        raise ConditionalOOFArtifactError("conditional OOF artifact must be a mapping")
    payload = {
        str(key): value
        for key, value in artifact.items()
        if str(key) not in _ARTIFACT_DIGEST_FIELDS
    }
    canonical = _artifact_json_value(payload, for_hash=True)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _origin_digest(origins: Any) -> str:
    """Hash the exact chronological origin records used by the artifact."""
    return hashlib.sha256(
        json.dumps(
            _artifact_json_value(origins),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _contract_digest(contract: Any, *, explicit_hash: str | None = None) -> str:
    """Resolve an ActionExecutionContract object/mapping to its hash."""
    resolved_from_contract: str | None = None
    if contract is not None and not isinstance(contract, str):
        if hasattr(contract, "contract_hash"):
            resolved_from_contract = _sha256_text(
                getattr(contract, "contract_hash"),
                name="action_execution_contract_sha256",
            )
        elif isinstance(contract, Mapping):
            supplied = contract.get(
                "contract_hash",
                contract.get("action_execution_contract_hash"),
            )
            if supplied is not None:
                resolved_from_contract = _sha256_text(
                    supplied,
                    name="action_execution_contract_sha256",
                )
            else:
                try:
                    from unidream.eval.action_execution import ActionExecutionContract

                    resolved_from_contract = ActionExecutionContract.from_config(contract).contract_hash
                except (ImportError, TypeError, ValueError) as exc:
                    raise ConditionalOOFArtifactError(
                        "action_execution_contract must expose a canonical contract hash"
                    ) from exc
    if explicit_hash is not None:
        resolved = _sha256_text(
            explicit_hash,
            name="action_execution_contract_sha256",
        )
        if resolved_from_contract is not None and resolved != resolved_from_contract:
            raise ConditionalOOFArtifactError(
                "action_execution_contract hash does not match the explicit hash"
            )
        return resolved
    elif isinstance(contract, str):
        resolved = _sha256_text(contract, name="action_execution_contract_sha256")
    elif resolved_from_contract is not None:
        resolved = resolved_from_contract
    else:
        raise ConditionalOOFArtifactError(
            "action_execution_contract or action_execution_contract_hash is required"
        )
    return resolved


def _coverage_rows(value: Any) -> list[dict[str, Any]]:
    """Normalize head-by-horizon coverage rows without hiding zero coverage."""
    if isinstance(value, Mapping):
        # A mapping keyed by ``(head, horizon)`` is convenient for callers but
        # is converted to the same stable list representation as JSONL rows.
        rows: list[Any] = []
        for key, row in value.items():
            if not isinstance(row, Mapping):
                raise ConditionalOOFArtifactError("coverage mapping values must be mappings")
            item = dict(row)
            if isinstance(key, tuple) and len(key) == 2:
                item.setdefault("head", key[0])
                item.setdefault("horizon", key[1])
            rows.append(item)
        value = rows
    if not isinstance(value, (list, tuple)) or not value:
        raise ConditionalOOFArtifactError(
            "conditional OOF artifact requires non-empty head-by-horizon coverage"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ConditionalOOFArtifactError("coverage rows must be mappings")
        row = dict(raw)
        head = row.get("head")
        if not isinstance(head, str) or not head.strip():
            raise ConditionalOOFArtifactError("coverage.head must be a non-empty string")
        horizon = strict_integer_value(row.get("horizon"), name="coverage.horizon")
        if horizon < 1:
            raise ConditionalOOFArtifactError("coverage.horizon must be >= 1")
        pair = (head, horizon)
        if pair in seen:
            raise ConditionalOOFArtifactError(
                f"duplicate head-by-horizon coverage row: {head}:{horizon}"
            )
        seen.add(pair)
        for field in ("target_count", "gradient_steps", "nonzero_gradient_steps"):
            if field in row:
                count = strict_integer_value(row[field], name=f"coverage.{field}")
                if count < 0:
                    raise ConditionalOOFArtifactError(f"coverage.{field} must be >= 0")
                row[field] = count
        if "target_count" not in row:
            raise ConditionalOOFArtifactError("coverage.target_count is required")
        for field in ("target_coverage", "gradient_coverage"):
            if field in row:
                raw_rate = row[field]
                if isinstance(raw_rate, (bool, np.bool_)) or not isinstance(raw_rate, (int, float, np.integer, np.floating)):
                    raise ConditionalOOFArtifactError(f"coverage.{field} must be numeric")
                rate = float(raw_rate)
                if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
                    raise ConditionalOOFArtifactError(f"coverage.{field} must lie in [0, 1]")
                row[field] = rate
        row["head"] = head
        row["horizon"] = horizon
        # Never drop a row because it is zero-covered: h64 is a visible
        # contract failure, not an absent diagnostic.
        normalized.append(row)
    return normalized


def _copy_oof_arrays(result: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the raw OOF fields while retaining NaNs and strict mask dtypes."""
    copied = dict(result)
    for name in _OOF_ARTIFACT_ARRAY_FIELDS:
        if name in result:
            copied[name] = np.array(result[name], copy=True)
    if "origins" in result:
        copied["origins"] = [dict(item) if isinstance(item, Mapping) else item for item in result["origins"]]
    if isinstance(result.get("metadata_by_row"), (list, tuple)):
        copied["metadata_by_row"] = [
            dict(item) if isinstance(item, Mapping) else item
            for item in result["metadata_by_row"]
        ]
    if isinstance(result.get("provenance"), Mapping):
        copied["provenance"] = dict(result["provenance"])
    return copied


def build_conditional_oof_artifact(
    oof_result: Mapping[str, Any],
    *,
    horizon: int,
    action_execution_contract: Any | None = None,
    action_execution_contract_hash: str | None = None,
    checkpoint_sha256: str | None = None,
    normalizer_sha256: str | None = None,
    calibrator_sha256: str | None = None,
    teacher_weight_sha256: str | None = None,
    teacher_sha256: str | None = None,
    coverage: Iterable[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned, hashable artifact for the conditional OOF path.

    This is intentionally separate from :func:`chronological_oof_predict`:
    the historical helper keeps its model-agnostic ``t + horizon`` default so
    old diagnostics remain reproducible, while a conditional artifact must
    use the explicit future-only label rule ``target_end_exclusive = t + h +
    1``.  All four model/provenance hashes are mandatory here even though no
    WM/BC/AC training is performed by this implementation unit.

    ``coverage`` is copied verbatim apart from strict scalar normalization.
    In particular a zero-covered h64 row is retained with ``target_count=0``
    and is rejected by the strict consumer gate rather than being filtered.
    """
    if not isinstance(oof_result, Mapping):
        raise ConditionalOOFArtifactError("oof_result must be a mapping")
    horizon = strict_integer_value(horizon, name="horizon")
    if horizon < 1:
        raise ConditionalOOFArtifactError("horizon must be >= 1")
    if not isinstance(metadata, Mapping) and metadata is not None:
        raise ConditionalOOFArtifactError("metadata must be a mapping")
    if coverage is None:
        coverage = oof_result.get("coverage")
    coverage_rows = _coverage_rows(coverage)

    # Require a complete provenance tuple before an artifact can be emitted.
    # These are content hashes, not free-form labels, so a placeholder cannot
    # accidentally pass the later promotion gate.
    if teacher_weight_sha256 is None:
        teacher_weight_sha256 = teacher_sha256
    hash_values = {
        "checkpoint_sha256": checkpoint_sha256,
        "normalizer_sha256": normalizer_sha256,
        "calibrator_sha256": calibrator_sha256,
        "teacher_weight_sha256": teacher_weight_sha256,
    }
    for name, value in hash_values.items():
        _sha256_text(value, name=name)
    contract_hash = _contract_digest(
        action_execution_contract,
        explicit_hash=action_execution_contract_hash,
    )

    predictions = np.asarray(oof_result.get("predictions"))
    if predictions.ndim != 2:
        raise ConditionalOOFArtifactError("oof_result.predictions must be a 2-D array")
    n_rows = len(predictions)
    expected_target_end = np.arange(n_rows, dtype=np.int64) + horizon + 1
    stored_target_end = oof_result.get("target_end_exclusive")
    if stored_target_end is None:
        raise ConditionalOOFArtifactError(
            "conditional OOF producer must persist target_end_exclusive"
        )
    target_end = strict_integer_array(
        stored_target_end,
        name="target_end_exclusive",
    )
    if target_end.shape != (n_rows,) or not np.array_equal(target_end, expected_target_end):
        raise ConditionalOOFArtifactError(
            "conditional OOF target_end_exclusive must equal t+h+1 for every row"
        )
    provenance_value = oof_result.get("provenance")
    if not isinstance(provenance_value, Mapping):
        raise ConditionalOOFArtifactError("oof_result.provenance must be a mapping")
    existing_horizon = provenance_value.get("horizon")
    if existing_horizon is not None and strict_integer_value(
        existing_horizon,
        name="oof_result.provenance.horizon",
    ) != horizon:
        raise ConditionalOOFArtifactError(
            "artifact horizon does not match oof_result.provenance.horizon"
        )
    if "in_sample" in provenance_value and strict_bool_value(
        provenance_value["in_sample"],
        name="oof_result.provenance.in_sample",
    ):
        raise ConditionalOOFArtifactError("conditional OOF artifact cannot be in-sample")
    origins = oof_result.get("origins")
    if not isinstance(origins, (list, tuple)):
        raise ConditionalOOFArtifactError("conditional OOF artifact requires origin records")

    artifact = _copy_oof_arrays(oof_result)
    artifact["schema"] = OOF_ARTIFACT_SCHEMA
    artifact["schema_version"] = OOF_ARTIFACT_SCHEMA_VERSION
    artifact["artifact_kind"] = "conditional_oof"
    artifact["target_end_exclusive"] = target_end
    artifact["coverage"] = coverage_rows
    provenance = dict(provenance_value)
    provenance.update(dict(metadata or {}))
    provenance.update(
        {
            "fit_scheme": "chronological_oof",
            "horizon": horizon,
            "target_end_rule": "t+h+1_exclusive",
            "execution_delay_bars": 1,
            "in_sample": False,
            "origin_sha256": _origin_digest(origins),
            "checkpoint_sha256": checkpoint_sha256,
            "normalizer_sha256": normalizer_sha256,
            "calibrator_sha256": calibrator_sha256,
            "teacher_weight_sha256": teacher_weight_sha256,
            "teacher_sha256": teacher_weight_sha256,
            "action_execution_contract_sha256": contract_hash,
            "action_execution_contract_hash": contract_hash,
        }
    )
    artifact["provenance"] = provenance
    # Duplicate the hash tuple at the artifact root to make lightweight
    # consumers able to inspect provenance without traversing nested metadata.
    artifact.update(
        {
            "origin_sha256": provenance["origin_sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "normalizer_sha256": normalizer_sha256,
            "calibrator_sha256": calibrator_sha256,
            "teacher_weight_sha256": teacher_weight_sha256,
            "teacher_sha256": teacher_weight_sha256,
            "action_execution_contract_sha256": contract_hash,
            "action_execution_contract_hash": contract_hash,
        }
    )
    artifact_digest = hash_conditional_oof_artifact(artifact)
    artifact["artifact_sha256"] = artifact_digest
    artifact["artifact_hash"] = artifact_digest
    # Structural validation is allowed to report zero coverage at production
    # time so the h64 defect is persisted.  The strict connection function
    # below performs the nonzero coverage promotion gate.
    validate_conditional_oof_artifact(
        artifact,
        require_nonzero_coverage=False,
    )
    return artifact


def _artifact_as_raw_oof(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the raw result subset consumed by ``validate_oof_result``."""
    return {
        key: artifact[key]
        for key in (
            "predictions",
            "prediction_mask",
            "oof_mask",
            "target_end_exclusive",
            "train_count",
            "origins",
            "metadata_by_row",
            "prediction_eligibility_mask",
            "training_label_eligibility_mask",
            "prediction_eligibility",
            "training_label_eligibility",
            "provenance",
        )
        if key in artifact
    }


def validate_conditional_oof_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_action_execution_contract: Any | None = None,
    expected_action_execution_contract_hash: str | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    expected_heads_horizons: Iterable[tuple[str, int]] | None = None,
    require_nonzero_coverage: bool = True,
    require_artifact_hash: bool = True,
) -> None:
    """Fail closed on the complete conditional OOF artifact contract.

    ``validate_oof_result`` checks the row-level chronological prefix.  This
    validator adds the producer-facing artifact schema, explicit ``t+h+1``
    target rule, delayed action alignment, immutable provenance hashes, and
    head-by-horizon target/gradient coverage.  The optional relaxed coverage
    mode exists only to persist a diagnostic (for example h64 with zero
    labels); a conditional consumer must use the strict default.
    """
    if not isinstance(artifact, Mapping):
        raise ConditionalOOFArtifactError("conditional OOF artifact must be a mapping")
    if artifact.get("schema") != OOF_ARTIFACT_SCHEMA:
        raise ConditionalOOFArtifactError(
            f"conditional OOF artifact schema must be {OOF_ARTIFACT_SCHEMA!r}"
        )
    version = artifact.get("schema_version")
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)):
        raise ConditionalOOFArtifactError("conditional OOF artifact schema_version must be an integer")
    if int(version) != OOF_ARTIFACT_SCHEMA_VERSION:
        raise ConditionalOOFArtifactError(
            f"unsupported conditional OOF artifact schema_version={version!r}"
        )
    if artifact.get("artifact_kind") != "conditional_oof":
        raise ConditionalOOFArtifactError("artifact_kind must be conditional_oof")
    if require_artifact_hash:
        supplied_digest = artifact.get("artifact_sha256")
        supplied_alias = artifact.get("artifact_hash")
        if supplied_digest is None or supplied_alias is None:
            raise ConditionalOOFArtifactError(
                "conditional OOF artifact requires artifact_sha256 and artifact_hash"
            )
        _sha256_text(supplied_digest, name="artifact_sha256")
        _sha256_text(supplied_alias, name="artifact_hash")
        if supplied_digest != supplied_alias:
            raise ConditionalOOFArtifactError("artifact_sha256 and artifact_hash differ")
        expected_digest = hash_conditional_oof_artifact(artifact)
        if supplied_digest != expected_digest:
            raise ConditionalOOFArtifactError(
                "conditional OOF artifact_sha256 does not match artifact content"
            )

    raw = _artifact_as_raw_oof(artifact)
    try:
        validate_oof_result(raw)
    except ChronologicalOOFError as exc:
        raise ConditionalOOFArtifactError(f"raw OOF contract failed: {exc}") from exc
    predictions = np.asarray(artifact["predictions"])
    n_rows = len(predictions)
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ConditionalOOFArtifactError("conditional OOF artifact provenance is missing")
    if strict_bool_value(provenance.get("in_sample"), name="provenance.in_sample"):
        raise ConditionalOOFArtifactError("conditional OOF artifact is marked in_sample")
    horizon = strict_integer_value(provenance.get("horizon"), name="provenance.horizon")
    if horizon < 1:
        raise ConditionalOOFArtifactError("provenance.horizon must be >= 1")
    target_end = strict_integer_array(
        artifact.get("target_end_exclusive"),
        name="target_end_exclusive",
    )
    expected_target_end = np.arange(n_rows, dtype=np.int64) + horizon + 1
    if target_end.shape != (n_rows,) or not np.array_equal(target_end, expected_target_end):
        raise ConditionalOOFArtifactError(
            "target_end_exclusive must equal t+h+1 for every row"
        )
    if provenance.get("target_end_rule") != "t+h+1_exclusive":
        raise ConditionalOOFArtifactError(
            "provenance.target_end_rule must be t+h+1_exclusive"
        )
    delay = strict_integer_value(
        provenance.get("execution_delay_bars"),
        name="provenance.execution_delay_bars",
    )
    if delay != 1:
        raise ConditionalOOFArtifactError(
            "conditional OOF artifact requires decision-to-fill delay of one bar"
        )

    origins = artifact.get("origins")
    if not isinstance(origins, (list, tuple)):
        raise ConditionalOOFArtifactError("conditional OOF artifact requires origins")
    previous_t = -1
    for origin in origins:
        if not isinstance(origin, Mapping):
            raise ConditionalOOFArtifactError("origin records must be mappings")
        t = strict_integer_value(origin.get("prediction_index"), name="origin.prediction_index")
        if t <= previous_t:
            raise ConditionalOOFArtifactError(
                "origin prediction_index values must be strictly increasing"
            )
        previous_t = t
        indices = strict_integer_array(origin.get("train_indices", []), name="origin.train_indices")
        purge = strict_integer_value(provenance.get("purge"), name="provenance.purge")
        if np.any(indices >= t):
            raise ConditionalOOFArtifactError(
                f"origin {t} contains a training index at/after the origin"
            )
        if len(indices) and np.any(target_end[indices] > t - purge):
            raise ConditionalOOFArtifactError(
                f"origin {t} contains an overlapping or incomplete training target"
            )
        cutoff = strict_integer_value(
            origin.get("label_cutoff_exclusive"),
            name="origin.label_cutoff_exclusive",
        )
        if cutoff != t - purge:
            raise ConditionalOOFArtifactError(
                f"origin {t} label cutoff does not match purge"
            )
    origin_digest = _sha256_text(
        provenance.get("origin_sha256"),
        name="provenance.origin_sha256",
    )
    if origin_digest != _origin_digest(origins):
        raise ConditionalOOFArtifactError("origin_sha256 does not match origin records")

    hash_fields = (
        "checkpoint_sha256",
        "normalizer_sha256",
        "calibrator_sha256",
        "teacher_weight_sha256",
        "action_execution_contract_sha256",
    )
    for field_name in hash_fields:
        nested = _sha256_text(provenance.get(field_name), name=f"provenance.{field_name}")
        root = _sha256_text(artifact.get(field_name), name=field_name)
        if nested != root:
            raise ConditionalOOFArtifactError(
                f"{field_name} differs between artifact root and provenance"
            )
    teacher_alias = _sha256_text(
        provenance.get("teacher_sha256"),
        name="provenance.teacher_sha256",
    )
    if teacher_alias != provenance["teacher_weight_sha256"] or teacher_alias != artifact.get("teacher_sha256"):
        raise ConditionalOOFArtifactError("teacher hash aliases do not match")
    contract_hash = provenance["action_execution_contract_sha256"]
    contract_alias = _sha256_text(
        provenance.get("action_execution_contract_hash"),
        name="provenance.action_execution_contract_hash",
    )
    if contract_alias != contract_hash or artifact.get("action_execution_contract_hash") != contract_hash:
        raise ConditionalOOFArtifactError("ActionExecutionContract hash aliases do not match")
    if expected_action_execution_contract is not None or expected_action_execution_contract_hash is not None:
        expected_contract_hash = _contract_digest(
            expected_action_execution_contract,
            explicit_hash=expected_action_execution_contract_hash,
        )
        if expected_contract_hash != contract_hash:
            raise ConditionalOOFArtifactError(
                "ActionExecutionContract hash mismatch"
            )
    if expected_hashes is not None:
        if not isinstance(expected_hashes, Mapping):
            raise ConditionalOOFArtifactError("expected_hashes must be a mapping")
        for name, expected in expected_hashes.items():
            if name not in hash_fields and name != "teacher_sha256":
                raise ConditionalOOFArtifactError(f"unsupported expected hash field: {name}")
            if _sha256_text(expected, name=f"expected.{name}") != artifact.get(name):
                raise ConditionalOOFArtifactError(f"{name} mismatch")

    coverage = _coverage_rows(artifact.get("coverage"))
    if expected_heads_horizons is not None:
        expected_pairs = {
            (str(head), strict_integer_value(h, name="expected coverage horizon"))
            for head, h in expected_heads_horizons
        }
        actual_pairs = {(row["head"], row["horizon"]) for row in coverage}
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        if missing or extra:
            raise ConditionalOOFArtifactError(
                f"coverage head-by-horizon set mismatch: missing={missing}, extra={extra}"
            )
    for row in coverage:
        target_count = int(row["target_count"])
        nonzero_steps = int(row.get("nonzero_gradient_steps", row.get("gradient_steps", 0)))
        blocked = target_count == 0 or nonzero_steps == 0
        if blocked and require_nonzero_coverage:
            reason = row.get("block_reason", "zero_valid_targets_or_gradients")
            raise ConditionalOOFArtifactError(
                f"coverage blocked for head={row['head']} horizon={row['horizon']}: {reason}"
            )
        if "status" in row and not isinstance(row["status"], str):
            raise ConditionalOOFArtifactError("coverage.status must be a string when supplied")


def _encode_artifact_json(value: Any) -> Any:
    """Encode an artifact for JSON while preserving typed NaN arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise ConditionalOOFArtifactError(
                f"cannot persist artifact array with dtype {value.dtype}"
            )
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data_b64": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _encode_artifact_json(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _encode_artifact_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_encode_artifact_json(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ConditionalOOFArtifactError(
                "non-finite scalar metadata cannot be persisted; use a typed array"
            )
        return value
    raise ConditionalOOFArtifactError(
        f"unsupported artifact JSON value type: {type(value).__name__}"
    )


def _decode_artifact_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__ndarray__") is True:
            dtype = np.dtype(value.get("dtype"))
            shape_value = value.get("shape")
            if not isinstance(shape_value, list) or any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, int) or item < 0
                for item in shape_value
            ):
                raise ConditionalOOFArtifactError("persisted ndarray shape is invalid")
            encoded = value.get("data_b64")
            if not isinstance(encoded, str):
                raise ConditionalOOFArtifactError("persisted ndarray data is missing")
            try:
                raw = base64.b64decode(encoded.encode("ascii"), validate=True)
            except (ValueError, UnicodeError) as exc:
                raise ConditionalOOFArtifactError("persisted ndarray data is not valid base64") from exc
            expected_nbytes = int(np.prod(shape_value, dtype=np.int64)) * dtype.itemsize
            if len(raw) != expected_nbytes:
                raise ConditionalOOFArtifactError("persisted ndarray byte length does not match shape")
            return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape_value)).copy()
        return {str(key): _decode_artifact_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_artifact_json(item) for item in value]
    return value


def write_conditional_oof_artifact(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    require_nonzero_coverage: bool = True,
) -> str:
    """Write a validated conditional OOF artifact and return its SHA-256.

    JSON stores arrays as typed base64 payloads, so NaN masks and dtypes are
    round-trippable without relying on non-standard JSON ``NaN`` literals.
    The temporary file is replaced atomically after validation.
    """
    validate_conditional_oof_artifact(
        artifact,
        require_nonzero_coverage=require_nonzero_coverage,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_artifact_json(artifact)
    text = json.dumps(
        encoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(output)
    return str(artifact["artifact_sha256"])


def load_conditional_oof_artifact(
    path: str | Path,
    *,
    expected_action_execution_contract: Any | None = None,
    expected_action_execution_contract_hash: str | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    expected_heads_horizons: Iterable[tuple[str, int]] | None = None,
    require_nonzero_coverage: bool = True,
) -> dict[str, Any]:
    """Load and fail closed on a persisted conditional OOF artifact."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConditionalOOFArtifactError(
            f"could not load conditional OOF artifact {source}: {exc}"
        ) from exc
    artifact = _decode_artifact_json(payload)
    if not isinstance(artifact, Mapping):
        raise ConditionalOOFArtifactError("persisted conditional OOF artifact must be a mapping")
    validate_conditional_oof_artifact(
        artifact,
        expected_action_execution_contract=expected_action_execution_contract,
        expected_action_execution_contract_hash=expected_action_execution_contract_hash,
        expected_hashes=expected_hashes,
        expected_heads_horizons=expected_heads_horizons,
        require_nonzero_coverage=require_nonzero_coverage,
    )
    return dict(artifact)


def _conditional_artifact_required(config: Mapping[str, Any] | None) -> bool:
    """Read the explicit strict-artifact opt-in without truthy coercion."""
    if not isinstance(config, Mapping):
        return False
    values: list[Any] = []
    for section in (
        config,
        config.get("conditional_oracle"),
        config.get("oracle"),
    ):
        if isinstance(section, Mapping):
            for key in (
                "require_conditional_oof_artifact",
                "conditional_oof_artifact_required",
            ):
                if key in section:
                    values.append(section[key])
    if not values:
        return False
    if any(type(value) is not bool for value in values):
        raise ChronologicalOOFError(
            "conditional OOF artifact requirement flags must be booleans"
        )
    return any(values)


def conditional_oof_artifact_required(config: Mapping[str, Any] | None) -> bool:
    """Return whether the new conditional config explicitly requires an artifact."""
    return _conditional_artifact_required(config)


def require_conditional_oof_artifact(
    *,
    config: Mapping[str, Any] | None,
    artifact: Mapping[str, Any] | None,
    caller: str,
    expected_action_execution_contract: Any | None = None,
    expected_action_execution_contract_hash: str | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    expected_heads_horizons: Iterable[tuple[str, int]] | None = None,
) -> None:
    """Validate the strict artifact boundary for a conditional consumer.

    The legacy raw bundle gate remains available for historical diagnostics.
    A new conditional caller opts into this function with
    ``require_conditional_oof_artifact: true``; once opted in, absent,
    malformed, stale, zero-covered, or hash-mismatched artifacts are all
    blocked before model/teacher code can run.
    """
    if not conditional_path_enabled(config):
        return
    if not isinstance(artifact, Mapping):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: complete conditional "
            "OOF artifact is missing"
        )
    try:
        validate_conditional_oof_artifact(
            artifact,
            expected_action_execution_contract=expected_action_execution_contract,
            expected_action_execution_contract_hash=expected_action_execution_contract_hash,
            expected_hashes=expected_hashes,
            expected_heads_horizons=expected_heads_horizons,
            require_nonzero_coverage=True,
        )
    except ConditionalOOFArtifactError as exc:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: OOF artifact contract "
            f"is invalid ({exc})"
        ) from exc


# Short aliases make the contract easy to discover from experiment code while
# retaining the descriptive names used in the reports.
conditional_oof_artifact_hash = hash_conditional_oof_artifact
OOFArtifactError = ConditionalOOFArtifactError


def _finite_rows(array: np.ndarray, *, name: str) -> np.ndarray:
    try:
        return np.isfinite(array).all(axis=tuple(range(1, array.ndim)))
    except (TypeError, ValueError) as exc:
        raise ChronologicalOOFError(f"{name} must contain numeric finite values") from exc


def conditional_path_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether a config opts into the new conditional/OOF path.

    Several names are accepted so an experiment manifest can choose a clear
    spelling without weakening the guard. A ``conditional_oracle`` mapping may
    use an explicit boolean ``enabled`` field. Flag values are deliberately
    strict: strings such as ``"false"`` are rejected rather than interpreted
    as truthy.
    """
    if not isinstance(config, Mapping):
        return False
    flag_names = (
        "conditional_oracle",
        "conditional_oracle_path",
        "predictable_conditional_path",
        "p0_b_conditional_path",
    )
    sections: list[Mapping[str, Any]] = [config]
    for section_name in ("oracle", "world_model", "ac", "bc"):
        section = config.get(section_name)
        if isinstance(section, Mapping):
            sections.append(section)
    for section in sections:
        for name in flag_names:
            if name not in section:
                continue
            value = section[name]
            if isinstance(value, Mapping):
                if "enabled" not in value:
                    raise ChronologicalOOFError(
                        f"{name}.enabled must be a bool when {name} is a mapping"
                    )
                value = value["enabled"]
            if strict_bool_value(value, name=name):
                return True
        mode = str(section.get("oracle_mode", section.get("mode", ""))).strip().lower()
        teacher_mode = str(section.get("teacher_mode", "")).strip().lower()
        if mode in {"conditional", "conditional_oof", "predictable_conditional"}:
            return True
        if teacher_mode in {"conditional", "conditional_oof", "predictable_conditional"}:
            return True
    return False


def require_conditional_oof_inputs(
    *,
    config: Mapping[str, Any] | None,
    oof_bundle: Mapping[str, Any] | None,
    caller: str,
) -> None:
    """Fail closed unless a caller supplies a complete raw OOF result bundle."""
    if not conditional_path_enabled(config):
        return
    if not isinstance(oof_bundle, Mapping):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: complete chronological "
            "OOF WM retraining/state provenance is not supplied; legacy in-sample "
            "future-target state cannot cross this boundary"
        )
    # The strict artifact contract is an explicit migration boundary.  Raw
    # bundles continue to serve the existing integration fixture, while any
    # caller that opts into ``require_conditional_oof_artifact`` (or supplies
    # an artifact envelope) must pass the content/hash/coverage validator
    # before the ordinary raw bundle checks below.
    strict_artifact = _conditional_artifact_required(config)
    selected_artifact: Mapping[str, Any] | None = None
    if "schema" in oof_bundle and oof_bundle.get("schema") == OOF_ARTIFACT_SCHEMA:
        selected_artifact = oof_bundle
    else:
        for key in ("conditional_oof_artifact", "oof_artifact", "artifact"):
            candidate = oof_bundle.get(key)
            if isinstance(candidate, Mapping):
                selected_artifact = candidate
                break
    if strict_artifact or selected_artifact is not None:
        if selected_artifact is None:
            raise ConditionalPathBlocked(
                f"{caller} is blocked for conditional Oracle: strict conditional "
                "OOF artifact is missing"
            )
        expected_contract: Any | None = None
        expected_contract_hash: str | None = None
        if isinstance(config, Mapping):
            expected_contract = config.get("action_execution_contract")
            conditional = config.get("conditional_oracle")
            if expected_contract is None and isinstance(conditional, Mapping):
                expected_contract = conditional.get("action_execution_contract")
            if isinstance(expected_contract, str):
                expected_contract_hash = expected_contract
                expected_contract = None
        require_conditional_oof_artifact(
            config=config,
            artifact=selected_artifact,
            caller=caller,
            expected_action_execution_contract=expected_contract,
            expected_action_execution_contract_hash=expected_contract_hash,
        )
        # An artifact-only envelope is valid for this boundary; callers that
        # need split state views still validate those views in predictive_state.
        if selected_artifact is not oof_bundle and "predictions" not in oof_bundle:
            oof_bundle = selected_artifact
    if "predictions" not in oof_bundle:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: split-only/raw state "
            "views must carry the complete chronological OOF result, including "
            "predictions, eligibility masks, and provenance"
        )
    try:
        validate_oof_result(oof_bundle)
    except ChronologicalOOFError as exc:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: complete OOF "
            f"eligibility contract is invalid ({exc})"
        ) from exc
    provenance = oof_bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: OOF bundle provenance is missing"
        )
    if str(provenance.get("fit_scheme", "")).strip().lower() not in {
        "chronological_oof",
        "expanding_origin",
        "rolling_origin",
    }:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: fit_scheme must be chronological OOF"
        )
    if strict_bool_value(provenance["in_sample"], name="oof_bundle.provenance.in_sample"):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: in-sample state is forbidden"
        )


def _as_2d_targets(targets: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(targets)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        raise ChronologicalOOFError(
            f"targets must have shape (n_rows, n_outputs), got {arr.shape} for n_rows={n_rows}"
        )
    return arr


def _as_row_mask(mask: np.ndarray | None, targets: np.ndarray) -> np.ndarray:
    if mask is None:
        result = np.ones(targets.shape[0], dtype=bool)
    else:
        raw = strict_bool_array(mask, name="valid_target_mask")
        if raw.ndim == 2:
            if raw.shape != targets.shape:
                raise ChronologicalOOFError(
                    f"valid_target_mask shape {raw.shape} does not match targets {targets.shape}"
                )
            result = raw.all(axis=1)
        elif raw.ndim == 1 and raw.shape[0] == targets.shape[0]:
            result = raw.copy()
        else:
            raise ChronologicalOOFError(
                "valid_target_mask must have one value per row or one value per target"
            )
    return result & _finite_rows(targets, name="targets")


def _as_row_eligibility_mask(mask: np.ndarray | None, n_rows: int) -> tuple[np.ndarray, bool]:
    if mask is None:
        return np.ones(n_rows, dtype=bool), False
    raw = strict_bool_array(mask, name="row_eligibility_mask")
    if raw.ndim != 1 or len(raw) != n_rows:
        raise ChronologicalOOFError(
            f"row_eligibility_mask must have shape ({n_rows},), got {raw.shape}"
        )
    return raw, True


def _coerce_prediction(value: Any, n_outputs: int) -> tuple[np.ndarray, Mapping[str, Any] | None]:
    metadata: Mapping[str, Any] | None = None
    if isinstance(value, Mapping):
        if "prediction" not in value:
            raise ChronologicalOOFError("fit_predict mapping result must contain 'prediction'")
        metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else None
        value = value["prediction"]
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        value, metadata = value
    pred = np.asarray(value, dtype=np.float64)
    if pred.ndim == 0:
        pred = pred.reshape(1, 1)
    elif pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.shape != (1, n_outputs):
        raise ChronologicalOOFError(
            f"fit_predict must return one row with {n_outputs} outputs, got {pred.shape}"
        )
    return pred[0], metadata


def chronological_oof_predict(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], Any],
    horizon: int = 1,
    purge: int | None = None,
    min_train_size: int = 1,
    train_window: int | None = None,
    step: int = 1,
    target_end: np.ndarray | None = None,
    valid_target_mask: np.ndarray | None = None,
    row_eligibility_mask: np.ndarray | None = None,
    row_eligibility_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate predictions using only label-complete chronological prefixes.

    ``fit_predict`` is called as ``fit_predict(x_train, y_train, x_test)`` for
    one row at a time.  If it returns ``{"prediction": row, "metadata": ...}``
    the metadata is retained per origin, which lets a model attach normalizer
    or calibrator hashes without hiding them in a global, future-fitted state.

    A target at row ``i`` is assumed complete at ``i + horizon`` unless an
    explicit ``target_end`` (exclusive row index) is supplied.  The training
    prefix must end at or before ``prediction_index - purge`` and never
    includes the origin row itself.  No early-row prediction is imputed:
    unavailable rows remain NaN and false in ``prediction_mask``.
    ``row_eligibility_mask`` is an optional strict bool vector supplied by the
    caller (for example, a P0-A availability/window mask).  Prediction-origin
    eligibility is only ``row_eligibility_mask & finite_features``; a future
    target's value or validity mask cannot decide whether the decision-time
    state is generated.  Training-label eligibility additionally requires the
    strict ``valid_target_mask`` and finite targets.  Consequently, an
    incomplete target tail can still receive a decision-time prediction when
    its features/window are eligible; ``prediction_mask`` records only finite
    callback output, not score/evaluation label completeness, so downstream
    scoring/evaluation must apply its own label-completeness mask.  A false
    origin never calls ``fit_predict`` and stays NaN/false; unavailable values
    are never sidecar-zero-filled. For a sequence/window representation, the
    caller must provide one eligibility value per window (the first axis); this
    function does not infer window eligibility from a sidecar or repair
    invalid windows.
    """
    x = np.asarray(features)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim < 2:
        raise ChronologicalOOFError(f"features must have row axis, got {x.shape}")
    n_rows = x.shape[0]
    y = _as_2d_targets(np.asarray(targets), n_rows)
    n_outputs = y.shape[1]
    horizon = strict_integer_value(horizon, name="horizon")
    if horizon < 1:
        raise ChronologicalOOFError("horizon must be >= 1")
    if purge is None:
        # A target ending at the prediction origin is already non-overlapping
        # with a future target beginning after that origin.  Extra embargo for
        # serial dependence must be supplied explicitly and is recorded below.
        purge = 0
    purge = strict_integer_value(purge, name="purge")
    if purge < 0:
        raise ChronologicalOOFError("purge must be >= 0")
    min_train_size = strict_integer_value(min_train_size, name="min_train_size")
    train_window = (
        None
        if train_window is None
        else strict_integer_value(train_window, name="train_window")
    )
    step = strict_integer_value(step, name="step")
    if min_train_size < 1:
        raise ChronologicalOOFError("min_train_size must be >= 1")
    if train_window is not None and train_window < min_train_size:
        raise ChronologicalOOFError("train_window must be >= min_train_size")
    if step < 1:
        raise ChronologicalOOFError("step must be >= 1")

    target_valid = _as_row_mask(valid_target_mask, y)
    caller_row_mask, row_mask_supplied = _as_row_eligibility_mask(
        row_eligibility_mask,
        n_rows,
    )
    feature_valid = _finite_rows(x, name="features")
    prediction_origin_valid = caller_row_mask & feature_valid
    training_label_valid = prediction_origin_valid & target_valid
    if target_end is None:
        label_end = np.arange(n_rows, dtype=np.int64) + horizon
    else:
        label_end = strict_integer_array(target_end, name="target_end")
        if label_end.ndim != 1 or len(label_end) != n_rows:
            raise ChronologicalOOFError("target_end must have one exclusive index per row")

    if row_eligibility_provenance is not None and not isinstance(
        row_eligibility_provenance,
        Mapping,
    ):
        raise ChronologicalOOFError("row_eligibility_provenance must be a mapping")
    eligibility_provenance = dict(row_eligibility_provenance or {})
    eligibility_source = eligibility_provenance.get(
        "source",
        "caller" if row_mask_supplied else "finite_features",
    )

    prediction_eligibility = {
        "count": int(prediction_origin_valid.sum()),
        "eligible_rows": int(prediction_origin_valid.sum()),
        "n_rows": n_rows,
        "source": eligibility_source,
        "row_eligibility_mask_supplied": row_mask_supplied,
        "feature_finite_guard": True,
        "target_mask_applied": False,
        "provenance": dict(eligibility_provenance),
    }
    training_label_eligibility = {
        "count": int(training_label_valid.sum()),
        "eligible_rows": int(training_label_valid.sum()),
        "n_rows": n_rows,
        "source": "prediction_eligibility_and_valid_target_mask",
        "prediction_eligibility_source": eligibility_source,
        "valid_target_mask_supplied": valid_target_mask is not None,
        "valid_target_mask_applied": True,
        "finite_target_guard": True,
        "provenance": dict(eligibility_provenance),
    }

    predictions = np.full((n_rows, n_outputs), np.nan, dtype=np.float64)
    prediction_mask = np.zeros(n_rows, dtype=bool)
    train_count = np.zeros(n_rows, dtype=np.int64)
    row_indices = np.arange(n_rows, dtype=np.int64)
    origin_records: list[dict[str, Any]] = []
    metadata_by_row: list[Mapping[str, Any] | None] = [None] * n_rows

    for prediction_index in range(0, n_rows, step):
        if not prediction_origin_valid[prediction_index]:
            continue
        label_cutoff_exclusive = prediction_index - purge
        eligible = np.flatnonzero(
            training_label_valid
            & (row_indices < prediction_index)
            & (label_end <= label_cutoff_exclusive)
        )
        if train_window is not None and len(eligible) > train_window:
            eligible = eligible[-train_window:]
        if len(eligible) < min_train_size:
            continue
        # ``eligible`` is already sorted and right-exclusive by construction.
        train_start = int(eligible[0])
        train_end_exclusive = int(eligible[-1]) + 1
        result = fit_predict(
            np.array(x[eligible], copy=True),
            np.array(y[eligible], copy=True),
            np.array(x[prediction_index : prediction_index + 1], copy=True),
        )
        pred_row, metadata = _coerce_prediction(result, n_outputs)
        predictions[prediction_index] = pred_row
        if np.isfinite(pred_row).all():
            prediction_mask[prediction_index] = True
        train_count[prediction_index] = len(eligible)
        metadata_by_row[prediction_index] = metadata
        origin = OOFOrigin(
            prediction_index=prediction_index,
            train_start=train_start,
            train_end_exclusive=train_end_exclusive,
            label_cutoff_exclusive=label_cutoff_exclusive,
            n_train=len(eligible),
        )
        origin_records.append(
            {
                "prediction_index": origin.prediction_index,
                "train_start": origin.train_start,
                "train_end_exclusive": origin.train_end_exclusive,
                "train_indices": eligible.astype(int).tolist(),
                "label_cutoff_exclusive": origin.label_cutoff_exclusive,
                "n_train": origin.n_train,
            }
        )

    result = {
        "predictions": predictions,
        "prediction_mask": prediction_mask,
        "oof_mask": prediction_mask.copy(),
        "target_end_exclusive": label_end.copy(),
        "train_count": train_count,
        "origins": origin_records,
        "metadata_by_row": metadata_by_row,
        "prediction_eligibility_mask": prediction_origin_valid.copy(),
        "training_label_eligibility_mask": training_label_valid.copy(),
        "prediction_eligibility": prediction_eligibility,
        "training_label_eligibility": training_label_eligibility,
        "provenance": {
            "fit_scheme": "chronological_oof",
            "horizon": horizon,
            "purge": purge,
            "min_train_size": min_train_size,
            "train_window": train_window,
            "step": step,
            "n_rows": n_rows,
            "n_predictions": int(prediction_mask.sum()),
            "n_origins_called": len(origin_records),
            "in_sample": False,
            "row_eligibility_mask_supplied": row_mask_supplied,
            "row_eligibility_source": eligibility_source,
            "row_eligibility_mask_source": eligibility_source,
            "row_eligibility_provenance": eligibility_provenance,
            "row_eligibility_mask_provenance": eligibility_provenance,
            "row_eligibility_applied_with_target_mask": False,
            "row_eligibility_eligible_rows": int(prediction_origin_valid.sum()),
            "prediction_eligibility": prediction_eligibility,
            "training_label_eligibility": training_label_eligibility,
            "prediction_eligibility_count": int(prediction_origin_valid.sum()),
            "training_label_eligibility_count": int(training_label_valid.sum()),
            "training_label_eligibility_applied_with_target_mask": True,
        },
    }
    validate_oof_result(result, target_end=label_end)
    return result


def validate_oof_result(
    result: Mapping[str, Any],
    *,
    target_end: np.ndarray | None = None,
) -> None:
    """Validate OOF values, masks, fit provenance, and label-complete prefixes.

    ``target_end_exclusive`` is persisted by the producer and is mandatory at
    this consumer boundary.  Supplying ``target_end`` is only an optional
    cross-check; it cannot replace the persisted vector.  Thus a consumer that
    calls this validator without labels or an external cutoff still checks
    every recorded training index against the producer's label-completeness
    contract.
    """
    predictions = np.asarray(result.get("predictions"))
    prediction_mask_present = "prediction_mask" in result
    oof_mask_present = "oof_mask" in result
    if not prediction_mask_present and not oof_mask_present:
        raise ChronologicalOOFError(
            "OOF result requires prediction_mask or oof_mask"
        )
    mask = strict_bool_array(
        result["prediction_mask"]
        if prediction_mask_present
        else result["oof_mask"],
        name="prediction_mask",
    )
    if prediction_mask_present and oof_mask_present:
        oof_mask = strict_bool_array(result["oof_mask"], name="oof_mask")
        if oof_mask.shape != mask.shape or not np.array_equal(oof_mask, mask):
            raise ChronologicalOOFError(
                "prediction_mask and oof_mask aliases must be strict-bool and equal"
            )
    if predictions.ndim != 2 or mask.ndim != 1 or predictions.shape[0] != mask.shape[0]:
        raise ChronologicalOOFError("OOF predictions/mask have incompatible shapes")
    try:
        finite_predictions = np.isfinite(predictions)
    except (TypeError, ValueError) as exc:
        raise ChronologicalOOFError("OOF predictions must contain numeric values") from exc
    if np.any(mask & ~finite_predictions.all(axis=1)):
        raise ChronologicalOOFError("prediction_mask marks a non-finite OOF row")
    if np.any(~mask & finite_predictions.any(axis=1)):
        raise ChronologicalOOFError(
            "finite OOF state exists outside the prediction mask; refusing a partial fill"
        )
    n_rows = predictions.shape[0]
    eligibility_masks: dict[str, np.ndarray] = {}
    for name in (
        "prediction_eligibility_mask",
        "training_label_eligibility_mask",
    ):
        if name not in result:
            raise ChronologicalOOFError(f"OOF result is missing required {name}")
        eligibility = strict_bool_array(result[name], name=name)
        if eligibility.ndim != 1 or eligibility.shape != (n_rows,):
            raise ChronologicalOOFError(
                f"{name} must be a 1-D full-row mask with shape ({n_rows},), "
                f"got {eligibility.shape}"
            )
        eligibility_masks[name] = eligibility
    prediction_eligibility = eligibility_masks["prediction_eligibility_mask"]
    training_eligibility = eligibility_masks["training_label_eligibility_mask"]
    if np.any(mask & ~prediction_eligibility):
        raise ChronologicalOOFError(
            "prediction_mask contains a row outside prediction_eligibility_mask"
        )
    if np.any(training_eligibility & ~prediction_eligibility):
        raise ChronologicalOOFError(
            "training_label_eligibility_mask contains a row outside prediction_eligibility_mask"
        )
    origins = result.get("origins", [])
    provenance = result.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ChronologicalOOFError("OOF provenance must be a mapping")
    fit_scheme = provenance.get("fit_scheme")
    if not isinstance(fit_scheme, str) or fit_scheme.strip().lower() not in {
        "chronological_oof",
        "expanding_origin",
        "rolling_origin",
    }:
        raise ChronologicalOOFError(
            "OOF provenance.fit_scheme must identify chronological OOF"
        )
    if "in_sample" not in provenance:
        raise ChronologicalOOFError(
            "OOF provenance.in_sample must be explicitly false"
        )
    if strict_bool_value(provenance["in_sample"], name="provenance.in_sample"):
        raise ChronologicalOOFError("OOF result is marked in_sample")

    persisted_target_end = result.get("target_end_exclusive")
    if target_end is None:
        if persisted_target_end is None:
            raise ChronologicalOOFError(
                "OOF result is missing required target_end_exclusive"
            )
        ends = strict_integer_array(
            persisted_target_end,
            name="target_end_exclusive",
        )
    else:
        ends = strict_integer_array(target_end, name="target_end")
        if persisted_target_end is None:
            raise ChronologicalOOFError(
                "OOF result is missing required target_end_exclusive"
            )
        persisted_ends = strict_integer_array(
            persisted_target_end,
            name="target_end_exclusive",
        )
        if persisted_ends.shape != ends.shape or not np.array_equal(
            persisted_ends,
            ends,
        ):
            raise ChronologicalOOFError(
                "target_end_exclusive does not match the supplied target_end"
            )
    if ends.ndim != 1 or len(ends) != n_rows:
        raise ChronologicalOOFError(
            "target_end_exclusive must have one exclusive index per row"
        )
    if np.any(ends < 0):
        raise ChronologicalOOFError("target_end_exclusive cannot contain negative indices")

    def validate_eligibility_detail(
        name: str,
        detail: Any,
        mask_value: np.ndarray,
    ) -> None:
        if not isinstance(detail, Mapping):
            raise ChronologicalOOFError(
                f"OOF {name} count/provenance detail is missing or not a mapping"
            )
        expected_count = int(mask_value.sum())
        for field in ("count", "eligible_rows", "n_rows"):
            if field not in detail:
                raise ChronologicalOOFError(
                    f"OOF {name} provenance is missing {field}"
                )
            actual = strict_integer_value(detail[field], name=f"{name}.{field}")
            expected = n_rows if field == "n_rows" else expected_count
            if actual != expected:
                raise ChronologicalOOFError(
                    f"OOF {name}.{field}={actual} does not match expected {expected}"
                )
        if not isinstance(detail.get("provenance"), Mapping):
            raise ChronologicalOOFError(
                f"OOF {name}.provenance must be a mapping"
            )

    detail_masks = {
        "prediction_eligibility": prediction_eligibility,
        "training_label_eligibility": training_eligibility,
    }
    for name, detail_mask in detail_masks.items():
        validate_eligibility_detail(name, result.get(name), detail_mask)
        validate_eligibility_detail(
            f"provenance.{name}",
            provenance.get(name),
            detail_mask,
        )
    for field, expected in (
        ("n_rows", n_rows),
        ("n_predictions", int(mask.sum())),
    ):
        if field not in provenance:
            raise ChronologicalOOFError(f"OOF provenance is missing {field}")
        actual = strict_integer_value(
            provenance[field],
            name=f"provenance.{field}",
        )
        if actual != expected:
            raise ChronologicalOOFError(
                f"provenance.{field}={actual} does not match expected {expected}"
            )
    for field in ("horizon", "purge", "min_train_size", "step"):
        if field not in provenance:
            raise ChronologicalOOFError(f"OOF provenance is missing {field}")
        strict_integer_value(provenance[field], name=f"provenance.{field}")
    horizon = int(provenance["horizon"])
    purge = int(provenance["purge"])
    min_train_size = int(provenance["min_train_size"])
    step = int(provenance["step"])
    if horizon < 1:
        raise ChronologicalOOFError("provenance.horizon must be >= 1")
    if purge < 0:
        raise ChronologicalOOFError("provenance.purge must be >= 0")
    if min_train_size < 1:
        raise ChronologicalOOFError("provenance.min_train_size must be >= 1")
    if step < 1:
        raise ChronologicalOOFError("provenance.step must be >= 1")
    if "train_window" not in provenance:
        raise ChronologicalOOFError("OOF provenance is missing train_window")
    if provenance["train_window"] is not None:
        strict_integer_value(provenance["train_window"], name="provenance.train_window")
        if int(provenance["train_window"]) < min_train_size:
            raise ChronologicalOOFError(
                "provenance.train_window must be >= min_train_size"
            )
    if "n_origins_called" not in provenance:
        raise ChronologicalOOFError("OOF provenance is missing n_origins_called")
    n_origins_called = strict_integer_value(
        provenance["n_origins_called"],
        name="provenance.n_origins_called",
    )
    if "origins" not in result:
        raise ChronologicalOOFError("OOF result is missing origin records")
    if not isinstance(origins, (list, tuple)):
        raise ChronologicalOOFError("OOF origins must be a list or tuple")
    if n_origins_called != len(origins):
        raise ChronologicalOOFError(
            "provenance.n_origins_called does not match origin records"
        )
    origin_indices: list[int] = []
    for origin in origins:
        if not isinstance(origin, Mapping):
            raise ChronologicalOOFError("OOF origin must be a mapping")
        required_origin_fields = (
            "prediction_index",
            "train_start",
            "train_end_exclusive",
            "label_cutoff_exclusive",
            "n_train",
        )
        for field in required_origin_fields:
            if field not in origin:
                raise ChronologicalOOFError(f"OOF origin is missing {field}")
        t = strict_integer_value(
            origin["prediction_index"],
            name="origin.prediction_index",
        )
        if t < 0 or t >= n_rows:
            raise ChronologicalOOFError("OOF origin prediction_index is out of range")
        if not prediction_eligibility[t]:
            raise ChronologicalOOFError(
                f"OOF origin {t} is outside prediction_eligibility_mask"
            )
        origin_indices.append(t)
        label_cutoff = strict_integer_value(
            origin["label_cutoff_exclusive"],
            name="origin.label_cutoff_exclusive",
        )
        expected_cutoff = t - purge
        if label_cutoff != expected_cutoff:
            raise ChronologicalOOFError(
                f"OOF origin {t} label_cutoff_exclusive={label_cutoff} "
                f"does not match purge cutoff {expected_cutoff}"
            )
        n_train = strict_integer_value(origin["n_train"], name="origin.n_train")
        if n_train < min_train_size:
            raise ChronologicalOOFError(
                f"OOF origin {t} n_train={n_train} is below min_train_size={min_train_size}"
            )
        start = strict_integer_value(origin["train_start"], name="origin.train_start")
        end = strict_integer_value(
            origin["train_end_exclusive"],
            name="origin.train_end_exclusive",
        )
        indices_value = origin.get("train_indices")
        if indices_value is None:
            if end < start:
                raise ChronologicalOOFError(
                    f"OOF origin {t} train range is not right-exclusive"
                )
            indices = np.arange(start, end, dtype=np.int64)
        else:
            indices = strict_integer_array(
                indices_value,
                name="origin.train_indices",
            )
        if indices.ndim != 1:
            raise ChronologicalOOFError("origin.train_indices must be 1-D")
        if np.any(indices < 0) or np.any(indices >= len(ends)):
            raise ChronologicalOOFError("OOF origin.train_indices are out of range")
        if len(indices) and np.any(np.diff(indices) <= 0):
            raise ChronologicalOOFError(
                f"OOF origin {t} train_indices must be strictly increasing and unique"
            )
        if len(indices):
            if start != int(indices[0]) or end != int(indices[-1]) + 1:
                raise ChronologicalOOFError(
                    f"OOF origin {t} train range does not bound train_indices"
                )
        elif start != end:
            raise ChronologicalOOFError(
                f"OOF origin {t} empty train_indices require equal range bounds"
            )
        if n_train != len(indices):
            raise ChronologicalOOFError(
                f"OOF origin {t} n_train does not match train_indices"
            )
        if np.any(indices >= t):
            raise ChronologicalOOFError(
                f"OOF origin {t} includes its own/future row in the training prefix"
            )
        if len(indices) and np.any(~training_eligibility[indices]):
            raise ChronologicalOOFError(
                f"OOF origin {t} includes a row outside training_label_eligibility_mask"
            )
        cutoff = label_cutoff
        train_end = ends[indices]
        if len(train_end) and int(np.max(train_end)) > cutoff:
            raise ChronologicalOOFError(
                f"OOF origin {t} includes a future/incomplete label: max_end={int(np.max(train_end))} cutoff={cutoff}"
            )
    if len(origin_indices) != len(set(origin_indices)):
        raise ChronologicalOOFError("OOF origins contain duplicate prediction_index records")
    missing_origin_indices = np.flatnonzero(
        mask & ~np.isin(np.arange(n_rows), origin_indices)
    )
    if len(missing_origin_indices):
        raise ChronologicalOOFError(
            "OOF origins are missing records for prediction_mask rows: "
            f"{missing_origin_indices.astype(int).tolist()}"
        )


def chronological_oof_standardize(
    predictions: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    min_history: int = 1,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Standardize OOF states with an expanding prefix only.

    The row being standardized is excluded from its own mean/std.  Early rows
    without enough OOF history remain NaN/false; callers must not replace them
    with in-sample values or zeros.
    """
    values = np.asarray(predictions, dtype=np.float64)
    mask = strict_bool_array(prediction_mask, name="prediction_mask")
    if values.ndim != 2 or mask.ndim != 1 or len(values) != len(mask):
        raise ChronologicalOOFError("predictions/mask have incompatible shapes")
    if np.any(mask & ~np.isfinite(values).all(axis=1)):
        raise ChronologicalOOFError("usable OOF state contains a non-finite value")
    if np.any(~mask & np.isfinite(values).any(axis=1)):
        raise ChronologicalOOFError(
            "finite state exists outside the OOF mask; refusing a partial or implicit fill"
        )
    min_history = strict_integer_value(min_history, name="min_history")
    if min_history < 1:
        raise ChronologicalOOFError("min_history must be >= 1")
    output = np.full_like(values, np.nan, dtype=np.float64)
    output_mask = np.zeros_like(mask)
    means = np.full_like(values, np.nan, dtype=np.float64)
    scales = np.full_like(values, np.nan, dtype=np.float64)
    for t in range(len(values)):
        if not mask[t]:
            continue
        history = values[:t][mask[:t]]
        history = history[np.isfinite(history).all(axis=1)]
        if len(history) < min_history:
            continue
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        std = np.where(std < float(epsilon), 1.0, std)
        output[t] = (values[t] - mean) / std
        means[t] = mean
        scales[t] = std
        if np.isfinite(output[t]).all():
            output_mask[t] = True
    return {
        "values": output,
        "mask": output_mask,
        "mean_by_row": means,
        "std_by_row": scales,
        "provenance": {
            "fit_scheme": "chronological_oof",
            "normalizer": "expanding_prefix",
            "in_sample": False,
            "min_history": min_history,
        },
    }


# Explicit aliases make the contract discoverable to experiment code without
# introducing a second implementation under a different name.
build_chronological_oof = chronological_oof_predict
build_chronological_oof_predictions = chronological_oof_predict


__all__ = [
    "ChronologicalOOFError",
    "ConditionalPathBlocked",
    "ConditionalOOFArtifactError",
    "OOFArtifactError",
    "OOF_ARTIFACT_SCHEMA",
    "OOF_ARTIFACT_SCHEMA_VERSION",
    "OOFOrigin",
    "build_conditional_oof_artifact",
    "build_chronological_oof",
    "build_chronological_oof_predictions",
    "chronological_oof_predict",
    "chronological_oof_standardize",
    "conditional_path_enabled",
    "conditional_oof_artifact_hash",
    "conditional_oof_artifact_required",
    "hash_conditional_oof_artifact",
    "load_conditional_oof_artifact",
    "require_conditional_oof_artifact",
    "require_conditional_oof_inputs",
    "strict_bool_array",
    "strict_bool_value",
    "strict_integer_array",
    "strict_integer_value",
    "validate_oof_result",
    "validate_conditional_oof_artifact",
    "write_conditional_oof_artifact",
]
