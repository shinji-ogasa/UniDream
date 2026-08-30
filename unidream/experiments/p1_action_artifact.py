"""Atomic, externally bound persistence for P1 action primitive artifacts.

The canonical action hashes intentionally cover the external schema and the
full record grid, not every provenance field in the in-memory header.  This
module adds the storage boundary used by the experiment runner: exact file
bytes are hashed outside the payload, production loads require that digest,
and all source arrays/arm metadata are revalidated instead of trusting the
stored header.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from .action_primitives import (
    ACTION_PRIMITIVE_ARM_FIELDS,
    ACTION_PRIMITIVE_HASH_FIELDS,
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
        "paired_common_mask_supplied",
        "paired_common_mask",
        "arm_metadata",
        *ACTION_PRIMITIVE_ARM_FIELDS,
        "source_role",
        "teacher_oracle_execution",
        "action_primitive_producer_status",
        "metric_source",
        "moving_block_bootstrap_status",
        "contract_json_sha256",
        *ACTION_PRIMITIVE_HASH_FIELDS,
    }
)
_PRODUCTION_EXPECTED_FIELDS = frozenset(
    {*ACTION_PRIMITIVE_ARM_FIELDS, "support_start", "support_range"}
)


@dataclass(frozen=True)
class LoadedP1ActionArtifact:
    path: Path
    file_sha256: str
    artifact: Mapping[str, Any]
    validation: Mapping[str, Any]


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


def _semantic_validate(
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None,
    realized_returns: Sequence[Any] | None,
    decision_block_scores: Sequence[Any] | None,
    decision_deltas: Sequence[Any] | None,
    decision_eligible: Sequence[Any] | None,
    score_eligible: Sequence[Any] | None,
    expected_common_mask: Sequence[Any] | None,
    require_production: bool,
) -> Mapping[str, Any]:
    expected = _strict_expected_metadata(
        expected_metadata,
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
    try:
        return validate_action_primitive_semantics(
            artifact,
            expected_metadata=expected,
            realized_returns=realized_returns,
            decision_block_scores=decision_block_scores,
            decision_deltas=decision_deltas,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            expected_common_mask=expected_common_mask,
            require_production=require_production,
        )
    except (ActionPrimitiveContractError, TypeError, ValueError, OverflowError) as exc:
        raise P1ActionArtifactError("action primitive semantic validation failed") from exc


def _columnar_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise P1ActionArtifactError("action artifact must be a mapping")
    header = artifact.get("header")
    records = artifact.get("records")
    if not isinstance(header, Mapping) or not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        raise P1ActionArtifactError("action artifact header/records are malformed")
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
        realized_returns=realized_returns,
        decision_block_scores=decision_block_scores,
        decision_deltas=decision_deltas,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        expected_common_mask=expected_common_mask,
        require_production=require_production,
    )
    encoded = _canonical_file_bytes(_columnar_payload(artifact))
    digest = hashlib.sha256(encoded).hexdigest()
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
    return digest


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


def _decode_payload(payload: Any) -> Mapping[str, Any]:
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
    if (
        not isinstance(header, Mapping)
        or any(not isinstance(field, str) for field in header)
        or set(header) != _HEADER_FIELDS
    ):
        raise P1ActionArtifactError("stored action artifact header fields are not exact")
    artifact: dict[str, Any] = {"header": dict(header), "records": records}
    for field in ACTION_PRIMITIVE_HASH_FIELDS:
        artifact[field] = _strict_sha256(payload.get(field), name=field)
    return artifact


def load_p1_action_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_metadata: Mapping[str, Any] | None = None,
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
    artifact = _decode_payload(payload)
    validation = _semantic_validate(
        artifact,
        expected_metadata=expected_metadata,
        realized_returns=realized_returns,
        decision_block_scores=decision_block_scores,
        decision_deltas=decision_deltas,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        expected_common_mask=expected_common_mask,
        require_production=require_production,
    )
    return LoadedP1ActionArtifact(
        path=source,
        file_sha256=actual_digest,
        artifact=artifact,
        validation=validation,
    )


__all__ = [
    "LoadedP1ActionArtifact",
    "P1ActionArtifactError",
    "P1_ACTION_FILE_FORMAT",
    "P1_ACTION_FILE_MAX_BYTES",
    "P1_ACTION_FILE_MAX_RECORDS",
    "P1_ACTION_FILE_VERSION",
    "load_p1_action_artifact",
    "save_p1_action_artifact",
]
