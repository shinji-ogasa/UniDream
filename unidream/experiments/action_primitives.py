"""Contract-only helpers for the P1 action primitive artifact.

The producer and moving-block bootstrap are intentionally not implemented on
the preregistration branch.  This module fixes the byte-level boundary that a
future producer must satisfy and provides a small fail-closed validator for
contract tests.  In particular, invalid/gapped rows are represented by rows
with false masks; they must not be dropped before hashing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import struct
from typing import Any

import numpy as np


class ActionPrimitiveContractError(ValueError):
    """Raised when action primitive records or hashes violate the contract."""


class ActionPrimitiveImplementationBlocked(RuntimeError):
    """Raised when the staged producer/bootstrap is requested too early."""


ACTION_PRIMITIVE_INDEX_FIELDS: tuple[str, ...] = (
    "primitive_index",
    "decision_index",
    "fill_index",
    "end_index",
)
ACTION_PRIMITIVE_VALUE_FIELDS: tuple[str, ...] = (
    "previous_position",
    "selected_delta",
    "selected_position",
    "candidate_utility",
    "benchmark_hold_utility",
    "same_state_local_hold_utility",
    "clairvoyant_utility",
    "regret",
    "opportunity",
    "agreement",
    "turnover",
    "active_indicator",
)
ACTION_PRIMITIVE_MASK_FIELDS: tuple[str, ...] = (
    "origin_eligible_mask",
    "forecast_finite_mask",
    "fill_complete_mask",
    "outcome_complete_mask",
    "scored_action_mask",
    "common_mask",
)
ACTION_PRIMITIVE_INTEGER_ARM_FIELDS: tuple[str, ...] = ("seed",)
ACTION_PRIMITIVE_STRING_ARM_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "split_id",
    "support_id",
    "model_id",
    "cost_mode",
    "cost_contract_hash",
)
ACTION_PRIMITIVE_RECORD_FIELDS: tuple[str, ...] = (
    *ACTION_PRIMITIVE_INDEX_FIELDS,
    *ACTION_PRIMITIVE_VALUE_FIELDS,
    *ACTION_PRIMITIVE_MASK_FIELDS,
    *ACTION_PRIMITIVE_STRING_ARM_FIELDS[:1],
    "seed",
    *ACTION_PRIMITIVE_STRING_ARM_FIELDS[1:],
)
ACTION_PRIMITIVE_HASH_FIELDS: tuple[str, ...] = (
    "action_primitive_payload_sha256",
    "action_primitive_schema_sha256",
    "action_primitive_content_sha256",
)
ACTION_PRIMITIVE_EXTERNAL_SCHEMA_PATH = (
    "docs/experiments/action_primitive_schema.json"
)
ACTION_PRIMITIVE_EXTERNAL_SCHEMA_SHA256 = (
    "d0520b3dbc3c444e2efe5a55e175e96b662f97fb404d901ea51e1c32e5bb9955"
)
ACTION_PRIMITIVE_EXECUTION_STATUS = "blocked_not_implemented"

_MAGIC_CONTENT = b"UNIDREAM-P1-ACTION-PRIMITIVE-CONTENT\x00"
_MAGIC_PAYLOAD = b"UNIDREAM-P1-ACTION-PRIMITIVE-PAYLOAD\x00"
_U64 = struct.Struct("<Q")


def _frame(payload: bytes) -> bytes:
    return _U64.pack(len(payload)) + payload


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ActionPrimitiveContractError(f"{field} must have int64 dtype")
    if isinstance(value, np.integer) and np.dtype(value.dtype) != np.dtype("<i8"):
        raise ActionPrimitiveContractError(f"{field} must have int64 dtype")
    return int(value)


def _strict_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise ActionPrimitiveContractError(f"{field} must have float64 dtype")
    if isinstance(value, np.floating) and np.dtype(value.dtype) != np.dtype("<f8"):
        raise ActionPrimitiveContractError(f"{field} must have float64 dtype")
    result = float(value)
    if np.isinf(result):
        raise ActionPrimitiveContractError(f"{field} contains infinity")
    return result


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ActionPrimitiveContractError(f"{field} must have bool dtype")
    return bool(value)


def _strict_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ActionPrimitiveContractError(f"{field} must have UTF-8 string dtype")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ActionPrimitiveContractError(
            f"{field} must contain valid UTF-8 text"
        ) from exc
    return value


def _canonical_float64_bytes(values: Sequence[Any], *, field: str) -> bytes:
    try:
        array = np.asarray(
            [_strict_float(value, field=field) for value in values],
            dtype="<f8",
            order="C",
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ActionPrimitiveContractError(
            f"{field} cannot be represented as little-endian float64"
        ) from exc
    # Canonicalise every NaN payload while preserving all finite values (and
    # rejecting infinities above).  This is explicit rather than relying on
    # the platform's NaN representation.
    bits = array.view("<u8")
    bits[np.isnan(array)] = np.uint64(0x7FF8000000000000)
    return np.ascontiguousarray(bits, dtype="<u8").tobytes(order="C")


def _canonical_int64_bytes(values: Sequence[Any], *, field: str) -> bytes:
    try:
        array = np.asarray(
            [_strict_int(value, field=field) for value in values],
            dtype="<i8",
            order="C",
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ActionPrimitiveContractError(
            f"{field} cannot be represented as little-endian int64"
        ) from exc
    return np.ascontiguousarray(array, dtype="<i8").tobytes(order="C")


def _canonical_bool_bytes(values: Sequence[Any], *, field: str) -> bytes:
    # A bool is framed as one byte, 0x00 or 0x01.  Native NumPy bool layout is
    # not used as an implicit wire format.
    array = np.asarray(
        [1 if _strict_bool(value, field=field) else 0 for value in values],
        dtype=np.uint8,
        order="C",
    )
    return np.ascontiguousarray(array, dtype=np.uint8).tobytes(order="C")


def _canonical_string_bytes(values: Sequence[Any], *, field: str) -> bytes:
    return b"".join(
        _frame(_strict_string(value, field=field).encode("utf-8")) for value in values
    )


def _records_to_columns(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Any]], int]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise ActionPrimitiveContractError("action primitive records must be a sequence")
    if len(records) == 0:
        raise ActionPrimitiveContractError(
            "action primitive records must contain at least one full-grid row"
        )
    columns = {field: [] for field in ACTION_PRIMITIVE_RECORD_FIELDS}
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ActionPrimitiveContractError(f"action primitive row {row_index} must be an object")
        if any(not isinstance(key, str) for key in record):
            raise ActionPrimitiveContractError(
                f"action primitive row {row_index} keys must be strings"
            )
        if set(record) != set(ACTION_PRIMITIVE_RECORD_FIELDS):
            missing = sorted(set(ACTION_PRIMITIVE_RECORD_FIELDS) - set(record))
            unknown = sorted(set(record) - set(ACTION_PRIMITIVE_RECORD_FIELDS))
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if unknown:
                detail.append("unknown=" + ",".join(unknown))
            raise ActionPrimitiveContractError(
                f"action primitive row {row_index} fields are not canonical ({'; '.join(detail)})"
            )
        for field in ACTION_PRIMITIVE_RECORD_FIELDS:
            columns[field].append(record[field])
    return columns, len(records)


def _validate_grid_order(columns: Mapping[str, Sequence[Any]], row_count: int) -> None:
    primitive = [_strict_int(value, field="primitive_index") for value in columns["primitive_index"]]
    if primitive != list(range(row_count)):
        raise ActionPrimitiveContractError(
            "primitive_index must retain every full-grid row in ascending order"
        )
    decision = [_strict_int(value, field="decision_index") for value in columns["decision_index"]]
    fill = [_strict_int(value, field="fill_index") for value in columns["fill_index"]]
    end = [_strict_int(value, field="end_index") for value in columns["end_index"]]
    if any(next_value - value != 4 for value, next_value in zip(decision, decision[1:])):
        raise ActionPrimitiveContractError(
            "decision_index must retain scheduled four-bar starts in chronological order"
        )
    if any(
        fill_value != decision_value + 1 or end_value != decision_value + 4
        for decision_value, fill_value, end_value in zip(decision, fill, end)
    ):
        raise ActionPrimitiveContractError(
            "fill_index=decision_index+1 and end_index=decision_index+4 are required"
        )


def canonical_action_primitive_content_bytes(
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize canonical records for the content SHA-256.

    Field order is ``ACTION_PRIMITIVE_RECORD_FIELDS`` and row order is the
    original chronological primitive order.  Numeric columns are one-
    dimensional C-order arrays with explicit little-endian int64/float64
    dtypes; masks use one explicit byte per bool; strings are UTF-8 and each
    value is length-prefixed.  Every field carries a framed name, dtype,
    shape, and byte length, so same-shaped but differently typed payloads do
    not collide.  Float64 NaNs use one canonical quiet-NaN bit pattern.
    """
    columns, row_count = _records_to_columns(records)
    _validate_grid_order(columns, row_count)
    encoded = bytearray(_MAGIC_CONTENT)
    encoded.extend(_U64.pack(row_count))
    for field in ACTION_PRIMITIVE_RECORD_FIELDS:
        if field in ACTION_PRIMITIVE_INDEX_FIELDS or field in ACTION_PRIMITIVE_INTEGER_ARM_FIELDS:
            dtype = "<i8"
            data = _canonical_int64_bytes(columns[field], field=field)
        elif field in ACTION_PRIMITIVE_VALUE_FIELDS:
            dtype = "<f8"
            data = _canonical_float64_bytes(columns[field], field=field)
        elif field in ACTION_PRIMITIVE_MASK_FIELDS:
            dtype = "bool-u8"
            data = _canonical_bool_bytes(columns[field], field=field)
        else:
            dtype = "utf-8"
            data = _canonical_string_bytes(columns[field], field=field)
        name = field.encode("utf-8")
        dtype_bytes = dtype.encode("ascii")
        shape = _U64.pack(row_count)
        encoded.extend(_frame(name))
        encoded.extend(_frame(dtype_bytes))
        encoded.extend(_U64.pack(1))  # ndim
        encoded.extend(shape)
        encoded.extend(_frame(data))
    return bytes(encoded)


def action_primitive_content_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_action_primitive_content_bytes(records)).hexdigest()


def _strict_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ActionPrimitiveContractError(
            f"{field} must be a 64-character lowercase hex digest"
        )
    if any(character not in "0123456789abcdef" for character in value):
        raise ActionPrimitiveContractError(f"{field} must be lowercase hexadecimal")
    return value


def canonical_action_primitive_payload_bytes(
    records: Sequence[Mapping[str, Any]],
    *,
    schema_sha256: str,
    content_sha256: str | None = None,
) -> bytes:
    """Serialize the payload envelope whose hash includes schema/content IDs.

    The envelope uses deterministic JSON for its scalar header and framed
    canonical content bytes for the records.  The payload hash is therefore
    distinct from the content hash and cannot be self-bound by changing only
    a declared digest field.
    """
    schema_sha256 = _strict_sha256(schema_sha256, field="schema_sha256")
    content = canonical_action_primitive_content_bytes(records)
    actual_content = hashlib.sha256(content).hexdigest()
    if content_sha256 is None:
        content_sha256 = actual_content
    else:
        content_sha256 = _strict_sha256(content_sha256, field="content_sha256")
    if content_sha256 != actual_content:
        raise ActionPrimitiveContractError("content_sha256 does not match canonical records")
    header = json.dumps(
        {
            "schema_sha256": schema_sha256,
            "content_sha256": content_sha256,
            "record_fields": list(ACTION_PRIMITIVE_RECORD_FIELDS),
            "record_count": len(records),
            "serialization": "canonical action primitive content bytes v1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _MAGIC_PAYLOAD + _frame(header) + _frame(content)


def action_primitive_payload_sha256(
    records: Sequence[Mapping[str, Any]],
    *,
    schema_sha256: str,
    content_sha256: str | None = None,
) -> str:
    return hashlib.sha256(
        canonical_action_primitive_payload_bytes(
            records,
            schema_sha256=schema_sha256,
            content_sha256=content_sha256,
        )
    ).hexdigest()


def canonical_action_primitive_schema_sha256(schema: Mapping[str, Any]) -> str:
    """Hash an external schema mapping using canonical UTF-8 JSON bytes."""
    if not isinstance(schema, Mapping):
        raise ActionPrimitiveContractError("action primitive schema must be an object")
    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ActionPrimitiveContractError("action primitive schema is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_action_primitive_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_schema_sha256: str | None = None,
    schema: Mapping[str, Any] | None = None,
    expected_content_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a non-empty artifact against the pinned external schema.

    All three expected digests and the external schema mapping are mandatory.
    The schema digest is never accepted as a self-declared payload value: the
    canonical external schema must hash to the independently pinned digest.
    """
    if schema is None:
        raise ActionPrimitiveContractError(
            "external schema mapping is required for action primitive validation"
        )
    if expected_schema_sha256 is None:
        raise ActionPrimitiveContractError(
            "expected external schema SHA-256 is required"
        )
    if expected_content_sha256 is None:
        raise ActionPrimitiveContractError(
            "expected action primitive content SHA-256 is required"
        )
    if expected_payload_sha256 is None:
        raise ActionPrimitiveContractError(
            "expected action primitive payload SHA-256 is required"
        )
    expected_schema_sha256 = _strict_sha256(
        expected_schema_sha256,
        field="expected external schema SHA-256",
    )
    expected_content_sha256 = _strict_sha256(
        expected_content_sha256,
        field="expected action primitive content SHA-256",
    )
    expected_payload_sha256 = _strict_sha256(
        expected_payload_sha256,
        field="expected action primitive payload SHA-256",
    )
    if expected_schema_sha256 != ACTION_PRIMITIVE_EXTERNAL_SCHEMA_SHA256:
        raise ActionPrimitiveContractError(
            "expected external schema SHA-256 is not the independently pinned digest"
        )
    content_sha256 = action_primitive_content_sha256(records)
    schema_sha256 = canonical_action_primitive_schema_sha256(schema)
    if schema_sha256 != expected_schema_sha256:
        raise ActionPrimitiveContractError("external schema SHA-256 mismatch")
    if content_sha256 != expected_content_sha256:
        raise ActionPrimitiveContractError("action primitive content SHA-256 mismatch")
    payload_sha256 = action_primitive_payload_sha256(
        records,
        schema_sha256=schema_sha256,
        content_sha256=content_sha256,
    )
    if payload_sha256 != expected_payload_sha256:
        raise ActionPrimitiveContractError("action primitive payload SHA-256 mismatch")
    return {
        "action_primitive_schema_sha256": schema_sha256,
        "action_primitive_content_sha256": content_sha256,
        "action_primitive_payload_sha256": payload_sha256,
        "record_count": len(records),
        "record_fields": list(ACTION_PRIMITIVE_RECORD_FIELDS),
    }


def require_action_primitive_implementation(*args: Any, **kwargs: Any) -> None:
    """Fail closed until a separately audited producer/bootstrap is landed."""
    raise ActionPrimitiveImplementationBlocked(
        "P1 action primitive producer and moving-block bootstrap are not implemented; "
        "the generic MBB path is forbidden"
    )


# Explicit names make an accidental implementation call easy to identify in
# a future runner and provide a stable blocked boundary for tests.
def build_action_primitive_grid(*args: Any, **kwargs: Any) -> None:
    require_action_primitive_implementation(*args, **kwargs)


def run_action_primitive_mbb(*args: Any, **kwargs: Any) -> None:
    require_action_primitive_implementation(*args, **kwargs)


__all__ = [
    "ACTION_PRIMITIVE_EXECUTION_STATUS",
    "ACTION_PRIMITIVE_EXTERNAL_SCHEMA_PATH",
    "ACTION_PRIMITIVE_EXTERNAL_SCHEMA_SHA256",
    "ACTION_PRIMITIVE_HASH_FIELDS",
    "ACTION_PRIMITIVE_INTEGER_ARM_FIELDS",
    "ACTION_PRIMITIVE_INDEX_FIELDS",
    "ACTION_PRIMITIVE_MASK_FIELDS",
    "ACTION_PRIMITIVE_RECORD_FIELDS",
    "ACTION_PRIMITIVE_STRING_ARM_FIELDS",
    "ACTION_PRIMITIVE_VALUE_FIELDS",
    "ActionPrimitiveContractError",
    "ActionPrimitiveImplementationBlocked",
    "action_primitive_content_sha256",
    "action_primitive_payload_sha256",
    "build_action_primitive_grid",
    "canonical_action_primitive_content_bytes",
    "canonical_action_primitive_payload_bytes",
    "canonical_action_primitive_schema_sha256",
    "require_action_primitive_implementation",
    "run_action_primitive_mbb",
    "validate_action_primitive_records",
]
