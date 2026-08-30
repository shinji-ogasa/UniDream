"""Canonical deterministic P1 action primitive producer and validator.

The producer materialises stored fixture or externally authenticated runner
inputs under the registered h4 action/execution contract.  It does not fit a
model, create a teacher, run a hindsight policy, or bootstrap records.  The
moving-block runner remains a separate boundary.  Invalid or gapped scheduled
rows are retained with false masks and are never compressed before hashing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from unidream.eval.action_execution import (
    ActionExecutionContract,
    complete_decision_starts,
)


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

ACTION_PRIMITIVE_COST_CONTRACT_PATHS: dict[str, str] = {
    "on": "docs/experiments/action_execution_contract.json",
    "off": "docs/experiments/action_execution_contract_cost_off.json",
}
ACTION_PRIMITIVE_COST_CONTRACT_SHA256: dict[str, str] = {
    "on": "6f5beb7865fceac5ecbcfbb31dd11e8fdada02e1841fecac1c17e22377bb624f",
    "off": "0d0508fa38b4d98bc7736add7916ed1afd7bedcddbf0c47bdadf8ff1183ccdcc",
}

# The two inferential action supports are fixed by the P1 preregistration.
# Stored record indices are raw/global body coordinates, while input vectors
# are split-local arrays.  A production artifact must bind both views exactly.
ACTION_PRIMITIVE_PRIMARY_SUPPORT_RANGES: dict[str, tuple[int, int]] = {
    "synthetic_validation": (90_000, 100_000),
    "s3_validation": (104_528, 139_568),
}

ACTION_PRIMITIVE_ARTIFACT_TYPE = "p1_action_primitive"
ACTION_PRIMITIVE_SCHEMA_ID = "p1-action-primitive-v1"
ACTION_PRIMITIVE_SCHEMA_VERSION = 1
ACTION_PRIMITIVE_ARM_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "seed",
    "split_id",
    "support_id",
    "model_id",
    "cost_mode",
    "cost_contract_hash",
)
ACTION_PRIMITIVE_METRIC_FIELDS: tuple[str, ...] = (
    "candidate_utility",
    "benchmark_hold_utility",
    "same_state_local_hold_utility",
    "clairvoyant_utility",
    "regret",
    "opportunity",
    "agreement",
)
ACTION_PRIMITIVE_STATE_FIELDS: tuple[str, ...] = (
    "previous_position",
    "selected_delta",
    "selected_position",
    "turnover",
    "active_indicator",
)
_ACTION_PRIMITIVE_MASK_LOGIC = {
    "origin_eligible_mask": "decision_eligible at the scheduled decision index t",
    "forecast_finite_mask": "the scalar forecast at t is finite",
    "fill_complete_mask": "the delayed fill bar t+1 is available",
    "outcome_complete_mask": "all realized return bars t+1..t+4 are finite and available",
    "scored_action_mask": "origin_eligible AND forecast_finite AND fill_complete AND outcome_complete",
    "common_mask": "scored_action AND optional paired common mask",
}

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
    if decision[0] < 0 or decision != [decision[0] + 4 * index for index in range(row_count)]:
        raise ActionPrimitiveContractError(
            "decision_index must retain one non-negative global support start and exact four-bar spacing"
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


def _as_float_vector(
    value: Sequence[Any],
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    """Read a one-dimensional numeric vector without repairing it."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ActionPrimitiveContractError(f"{name} must be a one-dimensional numeric vector") from exc
    if raw.ndim != 1:
        raise ActionPrimitiveContractError(f"{name} must be a one-dimensional numeric vector")
    if length is not None and len(raw) != length:
        raise ActionPrimitiveContractError(
            f"{name} must contain exactly {length} values; got {len(raw)}"
        )
    if raw.dtype.kind == "b":
        raise ActionPrimitiveContractError(f"{name} must be numeric, not boolean")
    if raw.dtype.kind in "OUS":
        raise ActionPrimitiveContractError(f"{name} must contain numeric values, not text/object values")
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ActionPrimitiveContractError(f"{name} must be numeric") from exc
    if np.any(np.isinf(result)):
        raise ActionPrimitiveContractError(f"{name} contains infinity")
    return result


def _as_bool_vector(
    value: Sequence[Any],
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    """Read a strict one-dimensional bool vector; integers are not masks."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ActionPrimitiveContractError(f"{name} must be a one-dimensional boolean mask") from exc
    if raw.ndim != 1:
        raise ActionPrimitiveContractError(f"{name} must be a one-dimensional boolean mask")
    if length is not None and len(raw) != length:
        raise ActionPrimitiveContractError(
            f"{name} must contain exactly {length} values; got {len(raw)}"
        )
    if not all(isinstance(item, (bool, np.bool_)) for item in raw.tolist()):
        raise ActionPrimitiveContractError(f"{name} must contain only boolean values")
    return raw.astype(bool, copy=True)


def _load_fixed_cost_contract(cost_mode: str) -> tuple[ActionExecutionContract, str, str]:
    """Load the independently registered cost-on/off contract.

    ``ActionExecutionContract.from_config(require_canonical=True)`` quite
    intentionally accepts only the cost-on P0-C contract.  The P1 comparison
    family also has a diagnostic cost-off arm, so this producer resolves both
    modes against their tracked JSON artifacts and their independently pinned
    contract hashes.  A caller-provided mapping is never allowed to redefine
    either mode.
    """
    if cost_mode not in ACTION_PRIMITIVE_COST_CONTRACT_PATHS:
        raise ActionPrimitiveContractError(
            "cost_mode must be exactly 'on' or 'off' for an action primitive"
        )
    path = Path(__file__).resolve().parents[2] / ACTION_PRIMITIVE_COST_CONTRACT_PATHS[cost_mode]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionPrimitiveContractError(
            f"could not load fixed {cost_mode} action execution contract: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ActionPrimitiveContractError("fixed action execution contract must be an object")
    try:
        candidate = ActionExecutionContract.from_config(payload, require_canonical=False)
    except (TypeError, ValueError) as exc:
        raise ActionPrimitiveContractError(
            f"fixed {cost_mode} action execution contract is invalid"
        ) from exc
    # The JSON contains a few derived fields which from_config deliberately
    # recomputes.  Equality here therefore detects both semantic drift and a
    # forged derived cost value in the tracked file.
    if candidate.to_dict() != dict(payload):
        raise ActionPrimitiveContractError(
            f"fixed {cost_mode} action execution contract is not canonical"
        )
    expected_hash = ACTION_PRIMITIVE_COST_CONTRACT_SHA256[cost_mode]
    if candidate.contract_hash != expected_hash:
        raise ActionPrimitiveContractError(
            f"fixed {cost_mode} action execution contract hash mismatch"
        )
    return candidate, ACTION_PRIMITIVE_COST_CONTRACT_PATHS[cost_mode], expected_hash


def _canonical_contract(
    contract: ActionExecutionContract | Mapping[str, Any] | None,
    *,
    cost_mode: str = "on",
) -> ActionExecutionContract:
    """Resolve a registered h4 cost-on/off action execution contract."""
    expected, _, _ = _load_fixed_cost_contract(cost_mode)
    if contract is None:
        return expected
    if isinstance(contract, ActionExecutionContract):
        candidate = contract
    elif isinstance(contract, Mapping):
        try:
            candidate = ActionExecutionContract.from_config(contract, require_canonical=False)
        except (TypeError, ValueError) as exc:
            raise ActionPrimitiveContractError(
                "action primitive producer requires the registered action execution contract"
            ) from exc
    else:
        raise ActionPrimitiveContractError("contract must be an ActionExecutionContract or mapping")
    if candidate.to_dict() != expected.to_dict() or candidate.contract_hash != expected.contract_hash:
        raise ActionPrimitiveContractError(
            f"provided action execution contract does not match registered cost-{cost_mode} contract"
        )
    return candidate


def _normalise_block_mask(
    value: Sequence[Any] | None,
    *,
    name: str,
    n_bars: int,
    starts: tuple[int, ...],
    role: str,
    fill_delay: int,
    commitment_bars: int,
) -> np.ndarray | None:
    """Accept a full bar mask or an already materialised block mask."""
    if value is None:
        return None
    raw = _as_bool_vector(value, name=name)
    if len(raw) == len(starts):
        return raw
    if len(raw) != n_bars:
        raise ActionPrimitiveContractError(
            f"{name} must have full bar length {n_bars} or block length {len(starts)}"
        )
    result = np.zeros(len(starts), dtype=bool)
    for index, start in enumerate(starts):
        fill = start + fill_delay
        end = fill + commitment_bars
        if role == "origin":
            result[index] = raw[start]
        elif role == "fill":
            result[index] = raw[fill]
        elif role == "outcome":
            result[index] = bool(raw[fill:end].all())
        else:
            raise ActionPrimitiveContractError(f"unknown block-mask role: {role}")
    return result


def _normalise_block_metric(
    value: Sequence[Any] | None,
    *,
    name: str,
    n_bars: int,
    starts: tuple[int, ...],
) -> np.ndarray | None:
    if value is None:
        return None
    raw = _as_float_vector(value, name=name)
    if len(raw) == len(starts):
        return raw
    if len(raw) == n_bars:
        return raw[np.asarray(starts, dtype=np.int64)].copy()
    raise ActionPrimitiveContractError(
        f"{name} must have full bar length {n_bars} or block length {len(starts)}"
    )


def _normalise_block_action(
    value: Sequence[Any] | None,
    *,
    name: str,
    n_bars: int,
    starts: tuple[int, ...],
) -> np.ndarray | None:
    """Materialise stored actions without silently replaying every-bar paths."""
    if value is None:
        return None
    raw = _as_float_vector(value, name=name)
    if len(raw) == len(starts):
        return raw
    if len(raw) != n_bars:
        raise ActionPrimitiveContractError(
            f"{name} must have full bar length {n_bars} or block length {len(starts)}"
        )
    scheduled = np.zeros(n_bars, dtype=bool)
    scheduled[np.asarray(starts, dtype=np.int64)] = True
    off_schedule = raw[~scheduled]
    if np.any(np.isfinite(off_schedule) & (np.abs(off_schedule) > 1e-9)):
        raise ActionPrimitiveContractError(
            f"{name} contains a non-zero action outside the scheduled block grid"
        )
    return raw[np.asarray(starts, dtype=np.int64)].copy()


def _normalise_arm_metadata(
    arm_metadata: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    direct: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for label, value in (("arm_metadata", arm_metadata), ("metadata", metadata)):
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ActionPrimitiveContractError(f"{label} must be a mapping")
        for key, item in value.items():
            if key in merged and merged[key] != item:
                raise ActionPrimitiveContractError(f"conflicting arm metadata for {key!r}")
            merged[str(key)] = item
    for key, value in direct.items():
        if value is not None:
            if key in merged and merged[key] != value:
                raise ActionPrimitiveContractError(f"conflicting arm metadata for {key!r}")
            merged[key] = value
    missing = [field for field in ACTION_PRIMITIVE_ARM_FIELDS if field not in merged]
    if missing:
        raise ActionPrimitiveContractError(
            "arm metadata is missing required fields: " + ", ".join(missing)
        )
    for field in (
        "scenario_id",
        "split_id",
        "support_id",
        "model_id",
        "cost_mode",
        "cost_contract_hash",
    ):
        _strict_string(merged[field], field=field)
        if not merged[field]:
            raise ActionPrimitiveContractError(f"{field} must be non-empty")
    if isinstance(merged["seed"], (bool, np.bool_)) or not isinstance(
        merged["seed"], (int, np.integer)
    ):
        raise ActionPrimitiveContractError("seed must have int64 dtype")
    merged["seed"] = int(merged["seed"])
    _strict_sha256(merged["cost_contract_hash"], field="cost_contract_hash")
    return {field: merged[field] for field in ACTION_PRIMITIVE_ARM_FIELDS}


def _choose_forecast_delta(
    current_position: float,
    forecast: float,
    contract: ActionExecutionContract,
) -> tuple[float, float]:
    if not np.isfinite(forecast):
        raise ActionPrimitiveContractError("finite forecast is required for an eligible action block")
    candidates: list[tuple[float, float, float]] = []
    for delta in contract.candidate_deltas:
        next_position = float(
            np.clip(
                current_position + float(delta),
                contract.position_min,
                contract.position_max,
            )
        )
        value = next_position * float(forecast) - float(
            abs(next_position - current_position) * contract.transition_cost_rate
        )
        candidates.append((value, float(delta), next_position))
    _, selected_delta, selected_position = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return selected_delta, selected_position


def _expected_block_metrics(
    *,
    previous_position: float,
    selected_position: float,
    returns: np.ndarray,
    fill: int,
    end: int,
    contract: ActionExecutionContract,
) -> dict[str, float]:
    block_returns = returns[fill:end]
    if len(block_returns) != contract.commitment_bars or not np.all(np.isfinite(block_returns)):
        raise ActionPrimitiveContractError("realized returns must be finite on a complete action block")
    block_sum = float(block_returns.sum())
    candidate_utility = selected_position * block_sum - abs(
        selected_position - previous_position
    ) * contract.transition_cost_rate
    benchmark_hold_utility = block_sum
    same_state_local_hold_utility = previous_position * block_sum
    feasible: list[tuple[float, float, float]] = []
    for delta in contract.candidate_deltas:
        candidate_position = float(
            np.clip(
                previous_position + float(delta),
                contract.position_min,
                contract.position_max,
            )
        )
        utility = candidate_position * block_sum - abs(
            candidate_position - previous_position
        ) * contract.transition_cost_rate
        feasible.append((utility, float(delta), candidate_position))
    clairvoyant_utility, _, clairvoyant_position = max(
        feasible,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return {
        "candidate_utility": float(candidate_utility),
        "benchmark_hold_utility": float(benchmark_hold_utility),
        "same_state_local_hold_utility": float(same_state_local_hold_utility),
        "clairvoyant_utility": float(clairvoyant_utility),
        "regret": float(clairvoyant_utility - candidate_utility),
        "opportunity": float(clairvoyant_utility - same_state_local_hold_utility),
        "agreement": float(
            np.isclose(selected_position, clairvoyant_position, atol=1e-9, rtol=0.0)
        ),
    }


def _compare_metric_values(
    actual: float,
    expected: float,
    *,
    field: str,
    row_index: int,
) -> None:
    if np.isnan(expected):
        if not np.isnan(actual):
            raise ActionPrimitiveContractError(
                f"row {row_index} {field} must be NaN on an unscored block"
            )
        return
    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-9, rtol=0.0):
        raise ActionPrimitiveContractError(
            f"row {row_index} {field} does not match the deterministic contract"
        )


def _load_external_action_schema() -> Mapping[str, Any]:
    path = Path(__file__).resolve().parents[2] / ACTION_PRIMITIVE_EXTERNAL_SCHEMA_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionPrimitiveContractError(
            f"could not load external action primitive schema: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ActionPrimitiveContractError("external action primitive schema must be an object")
    return payload


def _same_sequence(left: Any, right: Any) -> bool:
    """Compare two optional vector inputs without NumPy truth-value traps."""
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
    except (TypeError, ValueError):
        return False
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind == "f" or right_array.dtype.kind == "f":
        try:
            return bool(np.array_equal(left_array, right_array, equal_nan=True))
        except TypeError:
            return bool(
                np.array_equal(left_array, right_array)
                or np.all((np.isnan(left_array) & np.isnan(right_array)))
            )
    return bool(np.array_equal(left_array, right_array))


def _resolve_alias(
    first: Sequence[Any] | None,
    second: Sequence[Any] | None,
    *,
    first_name: str,
    second_name: str,
) -> Sequence[Any] | None:
    if first is not None and second is not None and not _same_sequence(first, second):
        raise ActionPrimitiveContractError(
            f"{first_name} and {second_name} disagree"
        )
    return first if first is not None else second


def _infer_bar_count(
    n_bars: int | None,
    *,
    returns: Sequence[Any] | None,
    score_eligible: Sequence[Any] | None,
) -> int:
    candidates: list[tuple[str, int]] = []
    if n_bars is not None:
        if isinstance(n_bars, (bool, np.bool_)) or not isinstance(n_bars, (int, np.integer)):
            raise ActionPrimitiveContractError("n_bars must be an integer")
        if int(n_bars) < 1:
            raise ActionPrimitiveContractError("n_bars must be positive")
        candidates.append(("n_bars", int(n_bars)))
    for name, value in (("returns", returns), ("score_eligible", score_eligible)):
        if value is None:
            continue
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ActionPrimitiveContractError(f"{name} must be one-dimensional") from exc
        if raw.ndim != 1:
            raise ActionPrimitiveContractError(f"{name} must be one-dimensional")
        if len(raw) < 1:
            raise ActionPrimitiveContractError(f"{name} must not be empty")
        candidates.append((name, len(raw)))
    if not candidates:
        raise ActionPrimitiveContractError(
            "n_bars or a full-length returns/score_eligible vector is required"
        )
    values = {value for _, value in candidates}
    if len(values) != 1:
        detail = ", ".join(f"{name}={value}" for name, value in candidates)
        raise ActionPrimitiveContractError(f"bar-length inputs disagree ({detail})")
    return candidates[0][1]


def _resolve_support_start(
    value: Any,
    *,
    arm: Mapping[str, Any],
    bar_count: int,
    require_registered_support: bool,
) -> int:
    if value is None:
        if require_registered_support:
            raise ActionPrimitiveContractError(
                "production action primitives require an explicit support_start"
            )
        support_start = 0
    else:
        support_start = _strict_int(value, field="support_start")
    if support_start < 0:
        raise ActionPrimitiveContractError("support_start must be non-negative")
    if require_registered_support:
        if arm.get("split_id") != "validation":
            raise ActionPrimitiveContractError(
                "production action primitives are restricted to the registered validation split"
            )
        support_id = arm.get("support_id")
        expected = ACTION_PRIMITIVE_PRIMARY_SUPPORT_RANGES.get(str(support_id))
        if expected is None:
            raise ActionPrimitiveContractError(
                "production action primitive support_id is not preregistered"
            )
        if (support_start, support_start + bar_count) != expected:
            raise ActionPrimitiveContractError(
                "production support_start/bar_count do not match the preregistered support range"
            )
    return support_start


def _header_schedule(
    contract: ActionExecutionContract,
    *,
    support_start: int,
) -> dict[str, Any]:
    return {
        "local_decision_start": int(contract.initial_countdown),
        "global_decision_start": int(support_start + contract.initial_countdown),
        "support_start": int(support_start),
        "decision_step": int(contract.commitment_bars),
        "fill_delay_bars": int(contract.execution_delay_bars),
        "commitment_bars": int(contract.commitment_bars),
        "tail_policy": contract.tail_policy,
        "record_rule": "one record per complete scheduled four-bar block",
        "index_rule": "global decision index = support_start + split-local decision index",
        "target_rule": "decision t -> fill t+1 -> returns[t+1:t+5]",
    }


def _header_action_grid(contract: ActionExecutionContract) -> dict[str, Any]:
    return {
        "position_min": float(contract.position_min),
        "position_max": float(contract.position_max),
        "candidate_deltas": [float(value) for value in contract.candidate_deltas],
        "clip_policy": "clip_then_deduplicate",
    }


def _header_cooldown(contract: ActionExecutionContract) -> dict[str, Any]:
    return {
        "commitment_bars": int(contract.commitment_bars),
        "countdown_reset": int(contract.commitment_bars),
        "countdown_decrement": int(contract.countdown_decrement),
        "initial_countdown": int(contract.initial_countdown),
        "decision_step": int(contract.commitment_bars),
    }


def _header_execution(contract: ActionExecutionContract) -> dict[str, Any]:
    return {
        "decision_to_fill_delay_bars": int(contract.execution_delay_bars),
        "fill_policy": contract.fill_policy,
        "partial_fill_policy": contract.partial_fill_policy,
        "tail_policy": contract.tail_policy,
        "feature_unavailable_policy": contract.feature_unavailable_policy,
        "outcome_unavailable_policy": contract.outcome_unavailable_policy,
        "execution_skip_policy": contract.execution_skip_policy,
    }


def _header_cost(contract: ActionExecutionContract) -> dict[str, Any]:
    return {
        "spread_bps": float(contract.spread_bps),
        "spread_convention": contract.spread_convention,
        "spread_side": "half_transition",
        "slippage_bps": float(contract.slippage_bps),
        "fee_rate": float(contract.fee_rate),
        "transition_cost_rate": float(contract.transition_cost_rate),
        "return_unit": contract.return_unit,
        "funding_included": bool(contract.funding_included),
        "boundary_cost_policy": contract.boundary_cost_policy,
    }


def _canonical_metric_dict(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in ACTION_PRIMITIVE_METRIC_FIELDS:
        value = values[field]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
            raise ActionPrimitiveContractError(f"{field} must have float64 dtype")
        result[field] = float(value)
    return result


def produce_action_primitive_grid(
    *,
    returns: Sequence[Any] | None = None,
    n_bars: int | None = None,
    support_start: int | None = None,
    decision_block_scores: Sequence[Any] | None = None,
    decision_deltas: Sequence[Any] | None = None,
    selected_deltas: Sequence[Any] | None = None,
    decision_eligible: Sequence[Any] | None = None,
    score_eligible: Sequence[Any] | None = None,
    origin_eligible_mask: Sequence[Any] | None = None,
    forecast_finite_mask: Sequence[Any] | None = None,
    fill_complete_mask: Sequence[Any] | None = None,
    outcome_complete_mask: Sequence[Any] | None = None,
    common_mask: Sequence[Any] | None = None,
    paired_common_mask: Sequence[Any] | None = None,
    metrics: Mapping[str, Sequence[Any]] | None = None,
    arm_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    scenario_id: str | None = None,
    seed: int | None = None,
    split_id: str | None = None,
    support_id: str | None = None,
    model_id: str | None = None,
    cost_mode: str | None = None,
    cost_contract_hash: str | None = None,
    contract: ActionExecutionContract | Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    require_production: bool = False,
) -> dict[str, Any]:
    """Produce one deterministic, full-grid h4 action primitive artifact.

    This is deliberately a materialiser, not a training or teacher path.  It
    consumes stored forecasts/actions, masks, and (optionally) realized
    returns.  It never fits a model, derives an action from hindsight, or
    replays inventory over a resampled sequence.  The separate P1 moving
    block bootstrap/runner remains blocked.

    ``returns`` and ``score_eligible`` are split-local full bar-length vectors.
    Persisted decision/fill/end indices are global coordinates obtained by
    adding ``support_start`` to the scheduled local grid.  Scores and
    selected deltas may be full bar-length vectors (only scheduled starts are
    read) or one value per scheduled block.  Explicit masks may use either
    representation; all output rows are always the complete scheduled grid.
    """
    origin_input = _resolve_alias(
        origin_eligible_mask,
        decision_eligible,
        first_name="origin_eligible_mask",
        second_name="decision_eligible",
    )
    if origin_input is None:
        raise ActionPrimitiveContractError(
            "origin_eligible_mask/decision_eligible is required"
        )
    direct_delta_input = _resolve_alias(
        decision_deltas,
        selected_deltas,
        first_name="decision_deltas",
        second_name="selected_deltas",
    )
    if score_eligible is None:
        raise ActionPrimitiveContractError("score_eligible is required")
    arm = _normalise_arm_metadata(
        arm_metadata,
        metadata,
        {
            "scenario_id": scenario_id,
            "seed": seed,
            "split_id": split_id,
            "support_id": support_id,
            "model_id": model_id,
            "cost_mode": cost_mode,
            "cost_contract_hash": cost_contract_hash,
        },
    )
    if arm["cost_mode"] not in ACTION_PRIMITIVE_COST_CONTRACT_PATHS:
        raise ActionPrimitiveContractError(
            "cost_mode must be exactly 'on' or 'off' for an action primitive"
        )
    contract_obj = _canonical_contract(contract, cost_mode=arm["cost_mode"])
    if arm["cost_contract_hash"] != contract_obj.contract_hash:
        raise ActionPrimitiveContractError(
            "cost_contract_hash does not match the selected fixed cost contract"
        )
    bar_count = _infer_bar_count(
        n_bars,
        returns=returns,
        score_eligible=score_eligible,
    )
    support_start_int = _resolve_support_start(
        support_start,
        arm=arm,
        bar_count=bar_count,
        require_registered_support=require_production,
    )
    local_starts = complete_decision_starts(bar_count, contract_obj)
    if not local_starts:
        raise ActionPrimitiveContractError(
            "inputs must contain at least one complete scheduled four-bar block"
        )
    if schema is None:
        schema = _load_external_action_schema()
    if not isinstance(schema, Mapping):
        raise ActionPrimitiveContractError("action primitive schema must be a mapping")
    schema_sha256 = canonical_action_primitive_schema_sha256(schema)
    if schema_sha256 != ACTION_PRIMITIVE_EXTERNAL_SCHEMA_SHA256:
        raise ActionPrimitiveContractError("external action primitive schema SHA-256 mismatch")
    if schema.get("schema_id") != ACTION_PRIMITIVE_SCHEMA_ID:
        raise ActionPrimitiveContractError("external action primitive schema_id mismatch")
    if schema.get("schema_version") != ACTION_PRIMITIVE_SCHEMA_VERSION:
        raise ActionPrimitiveContractError("external action primitive schema_version mismatch")
    if schema.get("record_fields") != list(ACTION_PRIMITIVE_RECORD_FIELDS):
        raise ActionPrimitiveContractError("external action primitive record_fields mismatch")

    returns_arr = None
    if returns is not None:
        returns_arr = _as_float_vector(returns, name="returns", length=bar_count)
    score_eligible_arr = _as_bool_vector(
        score_eligible,
        name="score_eligible",
        length=bar_count,
    )
    origin_block = _normalise_block_mask(
        origin_input,
        name="origin_eligible_mask",
        n_bars=bar_count,
        starts=local_starts,
        role="origin",
        fill_delay=contract_obj.execution_delay_bars,
        commitment_bars=contract_obj.commitment_bars,
    )
    assert origin_block is not None

    score_block = _normalise_block_metric(
        decision_block_scores,
        name="decision_block_scores",
        n_bars=bar_count,
        starts=local_starts,
    )
    delta_block = _normalise_block_action(
        direct_delta_input,
        name="decision_deltas",
        n_bars=bar_count,
        starts=local_starts,
    )
    if score_block is None and delta_block is None:
        raise ActionPrimitiveContractError(
            "decision_block_scores or decision_deltas is required"
        )
    explicit_forecast_block = _normalise_block_mask(
        forecast_finite_mask,
        name="forecast_finite_mask",
        n_bars=bar_count,
        starts=local_starts,
        role="origin",
        fill_delay=contract_obj.execution_delay_bars,
        commitment_bars=contract_obj.commitment_bars,
    )
    if score_block is not None:
        derived_forecast_block = np.isfinite(score_block)
        if explicit_forecast_block is not None and not np.array_equal(
            explicit_forecast_block,
            derived_forecast_block,
        ):
            raise ActionPrimitiveContractError(
                "forecast_finite_mask must equal finite decision_block_scores"
            )
        forecast_block = derived_forecast_block
    else:
        if explicit_forecast_block is None:
            raise ActionPrimitiveContractError(
                "forecast_finite_mask is required when decision_block_scores are absent"
            )
        forecast_block = explicit_forecast_block

    expected_fill_block = np.asarray(
        [score_eligible_arr[start + contract_obj.execution_delay_bars] for start in local_starts],
        dtype=bool,
    )
    expected_outcome_block = np.asarray(
        [
            bool(
                score_eligible_arr[
                    start + contract_obj.execution_delay_bars :
                    start + contract_obj.execution_delay_bars + contract_obj.commitment_bars
                ].all()
                and (
                    returns_arr is None
                    or np.all(
                        np.isfinite(
                            returns_arr[
                                start + contract_obj.execution_delay_bars :
                                start
                                + contract_obj.execution_delay_bars
                                + contract_obj.commitment_bars
                            ]
                        )
                    )
                )
            )
            for start in local_starts
        ],
        dtype=bool,
    )
    fill_block = _normalise_block_mask(
        fill_complete_mask,
        name="fill_complete_mask",
        n_bars=bar_count,
        starts=local_starts,
        role="fill",
        fill_delay=contract_obj.execution_delay_bars,
        commitment_bars=contract_obj.commitment_bars,
    )
    if fill_block is None:
        fill_block = expected_fill_block
    elif not np.array_equal(fill_block, expected_fill_block):
        raise ActionPrimitiveContractError(
            "fill_complete_mask must equal score_eligible at fill=t+1"
        )
    outcome_block = _normalise_block_mask(
        outcome_complete_mask,
        name="outcome_complete_mask",
        n_bars=bar_count,
        starts=local_starts,
        role="outcome",
        fill_delay=contract_obj.execution_delay_bars,
        commitment_bars=contract_obj.commitment_bars,
    )
    if outcome_block is None:
        outcome_block = expected_outcome_block
    elif not np.array_equal(outcome_block, expected_outcome_block):
        raise ActionPrimitiveContractError(
            "outcome_complete_mask must equal the complete delayed score/return window"
        )
    scored_block = origin_block & forecast_block & fill_block & outcome_block

    paired_input = _resolve_alias(
        paired_common_mask,
        common_mask,
        first_name="paired_common_mask",
        second_name="common_mask",
    )
    paired_supplied = paired_input is not None
    if paired_input is None:
        paired_block = np.ones(len(local_starts), dtype=bool)
    else:
        paired_block = _normalise_block_mask(
            paired_input,
            name="paired_common_mask",
            n_bars=bar_count,
            starts=local_starts,
            role="origin",
            fill_delay=contract_obj.execution_delay_bars,
            commitment_bars=contract_obj.commitment_bars,
        )
        assert paired_block is not None
    common_block = scored_block & paired_block

    metric_overrides: dict[str, np.ndarray] = {}
    if metrics is not None:
        if not isinstance(metrics, Mapping):
            raise ActionPrimitiveContractError("metrics must be a mapping")
        unknown_metrics = set(metrics) - set(ACTION_PRIMITIVE_METRIC_FIELDS)
        if unknown_metrics:
            raise ActionPrimitiveContractError(
                "unknown action primitive metrics: " + ", ".join(sorted(unknown_metrics))
            )
        for field, values in metrics.items():
            normalised = _normalise_block_metric(
                values,
                name=field,
                n_bars=bar_count,
                starts=local_starts,
            )
            assert normalised is not None
            metric_overrides[field] = normalised

    records: list[dict[str, Any]] = []
    current = float(contract_obj.p_start)
    for row_index, local_start in enumerate(local_starts):
        local_fill = local_start + contract_obj.execution_delay_bars
        local_end = local_fill + contract_obj.commitment_bars
        decision_index = support_start_int + local_start
        fill_index = support_start_int + local_fill
        end_index = support_start_int + local_end - 1
        previous_position = current
        scored = bool(scored_block[row_index])
        if scored:
            if delta_block is not None:
                chosen_delta = float(delta_block[row_index])
                if not np.isfinite(chosen_delta):
                    raise ActionPrimitiveContractError(
                        f"decision_deltas[{row_index}] must be finite on a scored block"
                    )
                if not any(
                    np.isclose(chosen_delta, allowed, atol=1e-9, rtol=0.0)
                    for allowed in contract_obj.candidate_deltas
                ):
                    raise ActionPrimitiveContractError(
                        f"decision_deltas[{row_index}] is outside the canonical action grid"
                    )
                if score_block is not None:
                    expected_delta, _ = _choose_forecast_delta(
                        previous_position,
                        float(score_block[row_index]),
                        contract_obj,
                    )
                    if not np.isclose(chosen_delta, expected_delta, atol=1e-9, rtol=0.0):
                        raise ActionPrimitiveContractError(
                            f"decision_deltas[{row_index}] disagrees with the deterministic score mapper"
                        )
            else:
                chosen_delta, _ = _choose_forecast_delta(
                    previous_position,
                    float(score_block[row_index]),
                    contract_obj,
                )
            selected_position = float(
                np.clip(
                    previous_position + chosen_delta,
                    contract_obj.position_min,
                    contract_obj.position_max,
                )
            )
            current = selected_position
        else:
            if delta_block is not None and np.isfinite(delta_block[row_index]):
                if not np.isclose(delta_block[row_index], 0.0, atol=1e-9, rtol=0.0):
                    raise ActionPrimitiveContractError(
                        f"decision_deltas[{row_index}] must be zero on an ineligible block"
                    )
            chosen_delta = 0.0
            selected_position = previous_position
        turnover = float(abs(selected_position - previous_position))
        active_indicator = float(turnover > 1e-9)

        if common_block[row_index]:
            if returns_arr is not None:
                expected_metrics = _expected_block_metrics(
                    previous_position=previous_position,
                    selected_position=selected_position,
                    returns=returns_arr,
                    fill=local_fill,
                    end=local_end,
                    contract=contract_obj,
                )
            else:
                expected_metrics = {}
                for field in ACTION_PRIMITIVE_METRIC_FIELDS:
                    if field not in metric_overrides:
                        raise ActionPrimitiveContractError(
                            f"{field} metrics are required when returns are absent"
                        )
                    value = float(metric_overrides[field][row_index])
                    if not np.isfinite(value):
                        raise ActionPrimitiveContractError(
                            f"{field} must be finite on a common block"
                        )
                    expected_metrics[field] = value
                if not np.isclose(
                    expected_metrics["regret"],
                    expected_metrics["clairvoyant_utility"]
                    - expected_metrics["candidate_utility"],
                    atol=1e-9,
                    rtol=0.0,
                ) or not np.isclose(
                    expected_metrics["opportunity"],
                    expected_metrics["clairvoyant_utility"]
                    - expected_metrics["same_state_local_hold_utility"],
                    atol=1e-9,
                    rtol=0.0,
                ):
                    raise ActionPrimitiveContractError(
                        f"metrics at row {row_index} violate regret/opportunity identities"
                    )
            for field, expected in expected_metrics.items():
                override = metric_overrides.get(field)
                if override is not None:
                    _compare_metric_values(
                        float(override[row_index]),
                        expected,
                        field=field,
                        row_index=row_index,
                    )
            metric_values = {field: float(expected_metrics[field]) for field in ACTION_PRIMITIVE_METRIC_FIELDS}
        else:
            metric_values = {field: float("nan") for field in ACTION_PRIMITIVE_METRIC_FIELDS}
            for field, override in metric_overrides.items():
                if not np.isnan(override[row_index]):
                    raise ActionPrimitiveContractError(
                        f"{field} must be NaN outside the common mask at row {row_index}"
                    )

        record: dict[str, Any] = {
            "primitive_index": int(row_index),
            "decision_index": int(decision_index),
            "fill_index": int(fill_index),
            "end_index": int(end_index),
            "previous_position": float(previous_position),
            "selected_delta": float(chosen_delta),
            "selected_position": float(selected_position),
            **metric_values,
            "turnover": turnover,
            "active_indicator": active_indicator,
            "origin_eligible_mask": bool(origin_block[row_index]),
            "forecast_finite_mask": bool(forecast_block[row_index]),
            "fill_complete_mask": bool(fill_block[row_index]),
            "outcome_complete_mask": bool(outcome_block[row_index]),
            "scored_action_mask": scored,
            "common_mask": bool(common_block[row_index]),
            **arm,
        }
        if tuple(record) != ACTION_PRIMITIVE_RECORD_FIELDS:
            raise ActionPrimitiveContractError("producer emitted non-canonical record field order")
        records.append(record)

    paired_common_header = [bool(value) for value in paired_block]
    contract_path = ACTION_PRIMITIVE_COST_CONTRACT_PATHS[arm["cost_mode"]]
    header: dict[str, Any] = {
        "artifact_type": ACTION_PRIMITIVE_ARTIFACT_TYPE,
        "schema_id": ACTION_PRIMITIVE_SCHEMA_ID,
        "schema_version": ACTION_PRIMITIVE_SCHEMA_VERSION,
        "record_fields": list(ACTION_PRIMITIVE_RECORD_FIELDS),
        "record_count": len(records),
        "bar_count": int(bar_count),
        "support_start": int(support_start_int),
        "support_range": [int(support_start_int), int(support_start_int + bar_count)],
        "contract_hash": contract_obj.contract_hash,
        "contract_path": contract_path,
        "contract": contract_obj.to_dict(),
        "action_grid": _header_action_grid(contract_obj),
        "cooldown": _header_cooldown(contract_obj),
        "execution": _header_execution(contract_obj),
        "cost": _header_cost(contract_obj),
        "schedule": _header_schedule(contract_obj, support_start=support_start_int),
        "mask_logic": dict(_ACTION_PRIMITIVE_MASK_LOGIC),
        "paired_common_mask_supplied": bool(paired_supplied),
        "paired_common_mask": paired_common_header,
        "arm_metadata": dict(arm),
        **arm,
        "source_role": (
            "validated_stored_action_inputs"
            if require_production
            else (
                "deterministic_fixture_realized_return_inputs"
                if returns_arr is not None
                else "deterministic_fixture_stored_action_inputs"
            )
        ),
        "teacher_oracle_execution": "not_run",
        "action_primitive_producer_status": (
            "validated_production_input" if require_production else "deterministic_fixture_only"
        ),
        "metric_source": (
            "recomputed_from_realized_returns"
            if require_production
            else (
                "recomputed_from_fixture_realized_returns"
                if returns_arr is not None
                else "caller_supplied_fixture_metrics"
            )
        ),
        "moving_block_bootstrap_status": ACTION_PRIMITIVE_EXECUTION_STATUS,
        "contract_json_sha256": contract_obj.contract_hash,
        "action_primitive_schema_sha256": schema_sha256,
        "action_primitive_content_sha256": action_primitive_content_sha256(records),
    }
    header["action_primitive_payload_sha256"] = action_primitive_payload_sha256(
        records,
        schema_sha256=schema_sha256,
        content_sha256=header["action_primitive_content_sha256"],
    )
    artifact = {
        "header": header,
        "records": records,
        "action_primitive_schema_sha256": header["action_primitive_schema_sha256"],
        "action_primitive_content_sha256": header["action_primitive_content_sha256"],
        "action_primitive_payload_sha256": header["action_primitive_payload_sha256"],
    }
    validate_action_primitive_semantics(
        artifact,
        contract=contract_obj,
        schema=schema,
        realized_returns=returns_arr,
        decision_block_scores=score_block,
        decision_deltas=delta_block,
        decision_eligible=origin_block,
        score_eligible=score_eligible_arr,
        expected_common_mask=common_block,
        require_production=require_production,
    )
    return artifact


def validate_action_primitive_semantics(
    artifact_or_records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    header: Mapping[str, Any] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    contract: ActionExecutionContract | Mapping[str, Any] | None = None,
    realized_returns: Sequence[Any] | None = None,
    returns: Sequence[Any] | None = None,
    decision_block_scores: Sequence[Any] | None = None,
    decision_deltas: Sequence[Any] | None = None,
    selected_deltas: Sequence[Any] | None = None,
    decision_eligible: Sequence[Any] | None = None,
    score_eligible: Sequence[Any] | None = None,
    expected_common_mask: Sequence[Any] | None = None,
    require_production: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation of the action primitive semantics and hashes.

    Hash validation alone is intentionally insufficient.  This validator
    authenticates the external schema and fixed cost contract, then recomputes
    the h4 schedule, all component masks, inventory recurrence, clipped grid
    action, fill geometry, cost-aware deterministic metrics, and arm binding.
    Optional source arrays allow a caller to rederive finite/availability and
    forecast/action semantics; without them, the persisted row-level
    invariants are still checked.
    """
    artifact_hashes: Mapping[str, Any] = {}
    if isinstance(artifact_or_records, Mapping):
        records = artifact_or_records.get("records")
        if records is None:
            raise ActionPrimitiveContractError("action primitive artifact records are required")
        if header is None:
            header = artifact_or_records.get("header")
        artifact_hashes = artifact_or_records
    else:
        records = artifact_or_records
    if not isinstance(header, Mapping):
        raise ActionPrimitiveContractError("action primitive artifact header is required")
    if schema is None:
        schema = _load_external_action_schema()
    if not isinstance(schema, Mapping):
        raise ActionPrimitiveContractError("action primitive schema must be a mapping")
    schema_sha256 = canonical_action_primitive_schema_sha256(schema)
    if schema_sha256 != ACTION_PRIMITIVE_EXTERNAL_SCHEMA_SHA256:
        raise ActionPrimitiveContractError("external action primitive schema SHA-256 mismatch")
    if schema.get("schema_id") != ACTION_PRIMITIVE_SCHEMA_ID:
        raise ActionPrimitiveContractError("external action primitive schema_id mismatch")
    if schema.get("schema_version") != ACTION_PRIMITIVE_SCHEMA_VERSION:
        raise ActionPrimitiveContractError("external action primitive schema_version mismatch")
    if schema.get("record_fields") != list(ACTION_PRIMITIVE_RECORD_FIELDS):
        raise ActionPrimitiveContractError("external action primitive record_fields mismatch")

    if header.get("artifact_type") != ACTION_PRIMITIVE_ARTIFACT_TYPE:
        raise ActionPrimitiveContractError("action primitive artifact_type mismatch")
    if header.get("schema_id") != ACTION_PRIMITIVE_SCHEMA_ID:
        raise ActionPrimitiveContractError("action primitive header schema_id mismatch")
    if header.get("schema_version") != ACTION_PRIMITIVE_SCHEMA_VERSION:
        raise ActionPrimitiveContractError("action primitive header schema_version mismatch")
    if header.get("record_fields") != list(ACTION_PRIMITIVE_RECORD_FIELDS):
        raise ActionPrimitiveContractError("action primitive header record_fields mismatch")
    record_count = _strict_int(header.get("record_count"), field="header.record_count")
    if record_count != len(records):
        raise ActionPrimitiveContractError("header.record_count does not match records")
    bar_count = _strict_int(header.get("bar_count"), field="header.bar_count")
    if bar_count < 1:
        raise ActionPrimitiveContractError("header.bar_count must be positive")

    header_hashes: dict[str, str] = {}
    for field in ACTION_PRIMITIVE_HASH_FIELDS:
        value = header.get(field)
        header_hashes[field] = _strict_sha256(value, field=f"header.{field}")
        if field in artifact_hashes and artifact_hashes[field] != value:
            raise ActionPrimitiveContractError(
                f"artifact {field} does not match the header declaration"
            )
    if header_hashes["action_primitive_schema_sha256"] != schema_sha256:
        raise ActionPrimitiveContractError("header external schema SHA-256 mismatch")

    arm_mapping = header.get("arm_metadata")
    if not isinstance(arm_mapping, Mapping):
        raise ActionPrimitiveContractError("header.arm_metadata is required")
    direct_arm = {}
    for field in ACTION_PRIMITIVE_ARM_FIELDS:
        if field not in header:
            raise ActionPrimitiveContractError(f"header is missing arm field {field}")
        direct_arm[field] = header[field]
    arm = _normalise_arm_metadata(arm_mapping, None, direct_arm)
    if dict(arm_mapping) != arm:
        raise ActionPrimitiveContractError("header.arm_metadata contains unknown or conflicting fields")
    for field in ACTION_PRIMITIVE_ARM_FIELDS:
        if header[field] != arm[field]:
            raise ActionPrimitiveContractError(f"header arm field {field} disagrees with arm_metadata")
    if expected_metadata is not None:
        if not isinstance(expected_metadata, Mapping):
            raise ActionPrimitiveContractError("expected_metadata must be a mapping")
        for field, expected_value in expected_metadata.items():
            actual_value = arm.get(field, header.get(field))
            if actual_value != expected_value:
                raise ActionPrimitiveContractError(
                    f"arm metadata does not match expected {field}"
                )

    contract_obj = _canonical_contract(contract, cost_mode=arm["cost_mode"])
    header_contract = header.get("contract")
    if not isinstance(header_contract, Mapping):
        raise ActionPrimitiveContractError("header.contract is required")
    if dict(header_contract) != contract_obj.to_dict():
        raise ActionPrimitiveContractError("header.contract does not match the fixed contract")
    if header.get("contract_hash") != contract_obj.contract_hash:
        raise ActionPrimitiveContractError("header.contract_hash mismatch")
    if header.get("contract_json_sha256") != contract_obj.contract_hash:
        raise ActionPrimitiveContractError("header.contract_json_sha256 mismatch")
    expected_path = ACTION_PRIMITIVE_COST_CONTRACT_PATHS[arm["cost_mode"]]
    if header.get("contract_path") != expected_path:
        raise ActionPrimitiveContractError("header.contract_path does not match cost_mode")
    if arm["cost_contract_hash"] != contract_obj.contract_hash:
        raise ActionPrimitiveContractError("cost_contract_hash does not match cost_mode contract")

    source_role = header.get("source_role")
    production_source = source_role == "validated_stored_action_inputs"
    support_start_int = _resolve_support_start(
        header.get("support_start"),
        arm=arm,
        bar_count=bar_count,
        require_registered_support=production_source,
    )
    if header.get("support_range") != [support_start_int, support_start_int + bar_count]:
        raise ActionPrimitiveContractError(
            "header.support_range must equal [support_start, support_start+bar_count)"
        )

    for field, expected_value in (
        ("action_grid", _header_action_grid(contract_obj)),
        ("cooldown", _header_cooldown(contract_obj)),
        ("execution", _header_execution(contract_obj)),
        ("cost", _header_cost(contract_obj)),
        ("schedule", _header_schedule(contract_obj, support_start=support_start_int)),
        ("mask_logic", _ACTION_PRIMITIVE_MASK_LOGIC),
    ):
        if header.get(field) != expected_value:
            raise ActionPrimitiveContractError(f"header.{field} does not match the contract")
    producer_status = header.get("action_primitive_producer_status")
    metric_source = header.get("metric_source")
    allowed_status = {
        "deterministic_fixture_stored_action_inputs": (
            "deterministic_fixture_only",
            "caller_supplied_fixture_metrics",
        ),
        "deterministic_fixture_realized_return_inputs": (
            "deterministic_fixture_only",
            "recomputed_from_fixture_realized_returns",
        ),
        "validated_stored_action_inputs": (
            "validated_production_input",
            "recomputed_from_realized_returns",
        ),
    }
    if source_role not in allowed_status:
        raise ActionPrimitiveContractError("action primitive source_role is not registered")
    expected_status, expected_metric_source = allowed_status[source_role]
    if producer_status != expected_status or metric_source != expected_metric_source:
        raise ActionPrimitiveContractError("action primitive producer/metric status is inconsistent")
    if header.get("teacher_oracle_execution") != "not_run":
        raise ActionPrimitiveContractError("teacher/oracle execution must remain not_run")
    if require_production and source_role != "validated_stored_action_inputs":
        raise ActionPrimitiveContractError("production validation requires realized stored returns")
    if header.get("moving_block_bootstrap_status") != ACTION_PRIMITIVE_EXECUTION_STATUS:
        raise ActionPrimitiveContractError("moving-block bootstrap must remain blocked")
    paired_supplied = header.get("paired_common_mask_supplied")
    if not isinstance(paired_supplied, (bool, np.bool_)):
        raise ActionPrimitiveContractError("header.paired_common_mask_supplied must be bool")
    paired_block = _as_bool_vector(
        header.get("paired_common_mask"),
        name="header.paired_common_mask",
        length=record_count,
    )
    if not bool(paired_supplied) and not np.all(paired_block):
        raise ActionPrimitiveContractError(
            "unspecified paired common mask must be all true"
        )

    local_starts = complete_decision_starts(bar_count, contract_obj)
    if len(local_starts) != record_count:
        raise ActionPrimitiveContractError(
            "header.bar_count does not produce exactly the persisted complete block grid"
        )
    hash_result = validate_action_primitive_records(
        records,
        schema=schema,
        expected_schema_sha256=header_hashes["action_primitive_schema_sha256"],
        expected_content_sha256=header_hashes["action_primitive_content_sha256"],
        expected_payload_sha256=header_hashes["action_primitive_payload_sha256"],
    )
    expected_global_starts = tuple(support_start_int + value for value in local_starts)
    if expected_global_starts != tuple(
        _strict_int(record["decision_index"], field="decision_index") for record in records
    ):
        raise ActionPrimitiveContractError("records do not cover the complete scheduled action grid")

    returns_input = _resolve_alias(
        realized_returns,
        returns,
        first_name="realized_returns",
        second_name="returns",
    )
    returns_arr = None
    if returns_input is not None:
        returns_arr = _as_float_vector(
            returns_input,
            name="realized_returns",
            length=bar_count,
        )
    if source_role == "validated_stored_action_inputs" and returns_arr is None:
        raise ActionPrimitiveContractError(
            "validated production action primitives require realized returns"
        )
    if require_production and returns_arr is None:
        raise ActionPrimitiveContractError(
            "production validation requires realized stored returns"
        )
    score_eligible_arr = None
    if score_eligible is not None:
        score_eligible_arr = _as_bool_vector(
            score_eligible,
            name="score_eligible",
            length=bar_count,
        )
    origin_expected = None
    if decision_eligible is not None:
        origin_expected = _normalise_block_mask(
            decision_eligible,
            name="decision_eligible",
            n_bars=bar_count,
            starts=local_starts,
            role="origin",
            fill_delay=contract_obj.execution_delay_bars,
            commitment_bars=contract_obj.commitment_bars,
        )
    score_block = None
    forecast_expected = None
    if decision_block_scores is not None:
        score_block = _normalise_block_metric(
            decision_block_scores,
            name="decision_block_scores",
            n_bars=bar_count,
            starts=local_starts,
        )
        assert score_block is not None
        forecast_expected = np.isfinite(score_block)
    delta_block = _normalise_block_action(
        _resolve_alias(
            decision_deltas,
            selected_deltas,
            first_name="decision_deltas",
            second_name="selected_deltas",
        ),
        name="decision_deltas",
        n_bars=bar_count,
        starts=local_starts,
    )
    if expected_common_mask is not None:
        expected_common_block = _normalise_block_mask(
            expected_common_mask,
            name="expected_common_mask",
            n_bars=bar_count,
            starts=local_starts,
            role="origin",
            fill_delay=contract_obj.execution_delay_bars,
            commitment_bars=contract_obj.commitment_bars,
        )
        assert expected_common_block is not None
    else:
        expected_common_block = None

    current = float(contract_obj.p_start)
    for row_index, (local_start, record) in enumerate(zip(local_starts, records)):
        local_fill = local_start + contract_obj.execution_delay_bars
        local_end = local_fill + contract_obj.commitment_bars
        decision_index = support_start_int + local_start
        fill_index = support_start_int + local_fill
        end_index = support_start_int + local_end - 1
        if _strict_int(record["primitive_index"], field="primitive_index") != row_index:
            raise ActionPrimitiveContractError("primitive_index does not match row order")
        if _strict_int(record["decision_index"], field="decision_index") != decision_index:
            raise ActionPrimitiveContractError("decision_index does not match schedule")
        if _strict_int(record["fill_index"], field="fill_index") != fill_index:
            raise ActionPrimitiveContractError("fill_index does not equal decision t+1")
        if _strict_int(record["end_index"], field="end_index") != end_index:
            raise ActionPrimitiveContractError("end_index does not equal inclusive t+4")
        for field in ACTION_PRIMITIVE_ARM_FIELDS:
            if record[field] != arm[field]:
                raise ActionPrimitiveContractError(
                    f"row {row_index} {field} disagrees with artifact arm_metadata"
                )
        previous = _strict_float(record["previous_position"], field="previous_position")
        chosen_delta = _strict_float(record["selected_delta"], field="selected_delta")
        selected = _strict_float(record["selected_position"], field="selected_position")
        turnover = _strict_float(record["turnover"], field="turnover")
        active = _strict_float(record["active_indicator"], field="active_indicator")
        if not np.isclose(previous, current, atol=1e-9, rtol=0.0):
            raise ActionPrimitiveContractError(
                f"row {row_index} previous_position breaks chronological replay state"
            )
        for allowed in contract_obj.candidate_deltas:
            if np.isclose(chosen_delta, allowed, atol=1e-9, rtol=0.0):
                break
        else:
            raise ActionPrimitiveContractError(
                f"row {row_index} selected_delta is outside the canonical action grid"
            )
        if not contract_obj.position_min - 1e-9 <= selected <= contract_obj.position_max + 1e-9:
            raise ActionPrimitiveContractError(f"row {row_index} selected_position is outside bounds")
        expected_selected = float(
            np.clip(
                previous + chosen_delta,
                contract_obj.position_min,
                contract_obj.position_max,
            )
        )
        if not np.isclose(selected, expected_selected, atol=1e-9, rtol=0.0):
            raise ActionPrimitiveContractError(
                f"row {row_index} selected_position is not clipped from previous+selected_delta"
            )
        expected_turnover = abs(selected - previous)
        if not np.isclose(turnover, expected_turnover, atol=1e-9, rtol=0.0):
            raise ActionPrimitiveContractError(f"row {row_index} turnover is not abs(position delta)")
        expected_active = float(expected_turnover > 1e-9)
        if not np.isclose(active, expected_active, atol=1e-9, rtol=0.0) or active not in (0.0, 1.0):
            raise ActionPrimitiveContractError(f"row {row_index} active_indicator is inconsistent")

        origin = _strict_bool(record["origin_eligible_mask"], field="origin_eligible_mask")
        forecast_finite = _strict_bool(record["forecast_finite_mask"], field="forecast_finite_mask")
        fill_complete = _strict_bool(record["fill_complete_mask"], field="fill_complete_mask")
        outcome_complete = _strict_bool(record["outcome_complete_mask"], field="outcome_complete_mask")
        scored = _strict_bool(record["scored_action_mask"], field="scored_action_mask")
        common = _strict_bool(record["common_mask"], field="common_mask")
        expected_scored = origin and forecast_finite and fill_complete and outcome_complete
        if scored != expected_scored:
            raise ActionPrimitiveContractError(
                f"row {row_index} scored_action_mask is not the component-mask intersection"
            )
        if common != (scored and bool(paired_block[row_index])):
            raise ActionPrimitiveContractError(
                f"row {row_index} common_mask is not scored AND paired common"
            )
        if origin_expected is not None and origin != bool(origin_expected[row_index]):
            raise ActionPrimitiveContractError(f"row {row_index} origin mask disagrees with source mask")
        if forecast_expected is not None and forecast_finite != bool(forecast_expected[row_index]):
            raise ActionPrimitiveContractError(f"row {row_index} forecast finite mask disagrees with source score")
        if score_eligible_arr is not None:
            expected_fill = bool(score_eligible_arr[local_fill])
            expected_outcome = bool(
                score_eligible_arr[local_fill:local_end].all()
                and (returns_arr is None or np.all(np.isfinite(returns_arr[local_fill:local_end])))
            )
            if fill_complete != expected_fill:
                raise ActionPrimitiveContractError(f"row {row_index} fill mask disagrees with score availability")
            if outcome_complete != expected_outcome:
                raise ActionPrimitiveContractError(f"row {row_index} outcome mask disagrees with score/return availability")
        elif returns_arr is not None and outcome_complete and not np.all(np.isfinite(returns_arr[local_fill:local_end])):
            raise ActionPrimitiveContractError(f"row {row_index} outcome mask accepts a non-finite return")
        if expected_common_block is not None and common != bool(expected_common_block[row_index]):
            raise ActionPrimitiveContractError(f"row {row_index} common mask disagrees with expected mask")

        if delta_block is not None:
            supplied_delta = float(delta_block[row_index])
            if scored:
                if not np.isfinite(supplied_delta) or not np.isclose(
                    chosen_delta,
                    supplied_delta,
                    atol=1e-9,
                    rtol=0.0,
                ):
                    raise ActionPrimitiveContractError(f"row {row_index} selected action disagrees with stored action input")
            elif np.isfinite(supplied_delta) and not np.isclose(supplied_delta, 0.0, atol=1e-9, rtol=0.0):
                raise ActionPrimitiveContractError(f"row {row_index} has a non-zero action on an ineligible block")
        if score_block is not None and scored:
            expected_delta, _ = _choose_forecast_delta(
                previous,
                float(score_block[row_index]),
                contract_obj,
            )
            if not np.isclose(chosen_delta, expected_delta, atol=1e-9, rtol=0.0):
                raise ActionPrimitiveContractError(f"row {row_index} action does not match deterministic score mapping")
        if scored:
            current = selected
        elif not np.isclose(chosen_delta, 0.0, atol=1e-9, rtol=0.0):
            raise ActionPrimitiveContractError(f"row {row_index} ineligible action must be zero")
        elif not np.isclose(selected, previous, atol=1e-9, rtol=0.0):
            raise ActionPrimitiveContractError(f"row {row_index} ineligible block must hold inventory")

        metric_values: dict[str, float] = {}
        for field in ACTION_PRIMITIVE_METRIC_FIELDS:
            metric_values[field] = _strict_float(record[field], field=field)
        if common:
            if any(not np.isfinite(value) for value in metric_values.values()):
                raise ActionPrimitiveContractError(f"row {row_index} common metrics must be finite")
            if not np.isclose(
                metric_values["regret"],
                metric_values["clairvoyant_utility"] - metric_values["candidate_utility"],
                atol=1e-9,
                rtol=0.0,
            ):
                raise ActionPrimitiveContractError(f"row {row_index} regret identity mismatch")
            if not np.isclose(
                metric_values["opportunity"],
                metric_values["clairvoyant_utility"]
                - metric_values["same_state_local_hold_utility"],
                atol=1e-9,
                rtol=0.0,
            ):
                raise ActionPrimitiveContractError(f"row {row_index} opportunity identity mismatch")
            if metric_values["agreement"] not in (0.0, 1.0):
                raise ActionPrimitiveContractError(f"row {row_index} agreement must be 0 or 1")
            if returns_arr is not None:
                expected_metrics = _expected_block_metrics(
                    previous_position=previous,
                    selected_position=selected,
                    returns=returns_arr,
                    fill=local_fill,
                    end=local_end,
                    contract=contract_obj,
                )
                for field in ACTION_PRIMITIVE_METRIC_FIELDS:
                    _compare_metric_values(
                        metric_values[field],
                        expected_metrics[field],
                        field=field,
                        row_index=row_index,
                    )
        elif any(not np.isnan(value) for value in metric_values.values()):
            raise ActionPrimitiveContractError(
                f"row {row_index} metrics must be NaN outside common_mask"
            )

    return {
        **hash_result,
        "semantic_validation_status": "passed",
        "record_count": record_count,
        "bar_count": bar_count,
        "arm_metadata": dict(arm),
        "contract_hash": contract_obj.contract_hash,
        "cost_mode": arm["cost_mode"],
    }


validate_action_primitive_artifact = validate_action_primitive_semantics
produce_action_primitive_artifact = produce_action_primitive_grid


def require_action_primitive_implementation(*args: Any, **kwargs: Any) -> None:
    """Fail closed until the separately audited MBB/result integration is landed."""
    raise ActionPrimitiveImplementationBlocked(
        "P1 action primitive moving-block/result integration is not implemented; "
        "the generic MBB path remains forbidden"
    )


# Explicit names make an accidental implementation call easy to identify in
# a future runner.  ``build_action_primitive_grid`` is the deterministic
# producer; the MBB path remains a separate fail-closed boundary.
def build_action_primitive_grid(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return produce_action_primitive_grid(*args, **kwargs)


def run_action_primitive_mbb(*args: Any, **kwargs: Any) -> None:
    require_action_primitive_implementation(*args, **kwargs)


__all__ = [
    "ACTION_PRIMITIVE_ARTIFACT_TYPE",
    "ACTION_PRIMITIVE_ARM_FIELDS",
    "ACTION_PRIMITIVE_COST_CONTRACT_PATHS",
    "ACTION_PRIMITIVE_COST_CONTRACT_SHA256",
    "ACTION_PRIMITIVE_PRIMARY_SUPPORT_RANGES",
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
    "produce_action_primitive_artifact",
    "produce_action_primitive_grid",
    "require_action_primitive_implementation",
    "run_action_primitive_mbb",
    "validate_action_primitive_artifact",
    "validate_action_primitive_records",
    "validate_action_primitive_semantics",
]
