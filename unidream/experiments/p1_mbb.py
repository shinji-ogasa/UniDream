"""Exact, paired, non-circular moving-block bootstrap for the P1 protocol.

This module is deliberately narrower than :mod:`unidream.eval.statistical_gate`.
The latter is a generic development diagnostic and is not a P1 implementation.
P1 uses one immutable, full-grid index draw for a paired comparison, keeps
false-mask/N/A rows in their original positions, and persists every sampled
block start so a result can be replayed without drawing a new random stream.

No model fitting, action-policy replay, or outer/test operation belongs here.
The caller supplies already materialized per-primitive metric arrays.  Only
the paired metric forms named by the preregistration are accepted.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
import zipfile

import numpy as np


class P1MBBError(ValueError):
    """Raised when the fixed P1 MBB contract cannot be satisfied."""


class P1MBBImplementationBlocked(P1MBBError):
    """Raised when an unpaired or generic bootstrap is requested."""


P1_MBB_SCHEMA = "unidream.p1.moving_block_indices"
P1_MBB_SCHEMA_VERSION = 1
P1_MBB_REPLICATES = 2000
P1_MBB_BLOCK_LENGTHS = (8, 16, 32)
P1_MBB_PRIMARY_BLOCK_LENGTH = 16
P1_MBB_BASE_SEED = 20260830
P1_MBB_UNIT_CODES: Mapping[str, int] = {
    "synthetic_forecast": 1,
    "synthetic_action": 2,
    "s3_forecast": 3,
    "s3_action": 4,
}
P1_MBB_UNIT_SUPPORTS: Mapping[str, str] = {
    "synthetic_forecast": "synthetic_validation",
    "synthetic_action": "synthetic_validation",
    "s3_forecast": "s3_validation",
    "s3_action": "s3_validation",
}
P1_PAIRED_MEAN_METRICS = frozenset(
    {
        "mse_delta",
        "logloss",
        "agreement",
        "policy_utility_delta",
    }
)
P1_RECOMPUTE_METRICS = frozenset(
    {
        "mse_delta",
        "skill",
        "logloss",
        "agreement",
        "policy_utility_delta",
        "s2_contrast",
        "normalized_regret",
        "s3_skill_did",
        "s3_utility_did",
    }
)
_P1_METRIC_ARRAY_KEYS: Mapping[str, frozenset[str]] = {
    "mse_delta": frozenset({"candidate_se", "baseline_se"}),
    "skill": frozenset({"model_se", "zero_se"}),
    "logloss": frozenset({"candidate_logloss", "baseline_logloss"}),
    "agreement": frozenset({"candidate_agreement", "baseline_agreement"}),
    "policy_utility_delta": frozenset(
        {"candidate_utility", "benchmark_hold_utility"}
    ),
    "s2_contrast": frozenset({"level_a_values", "level_b_values"}),
    "normalized_regret": frozenset({"regret", "opportunity"}),
    "s3_skill_did": frozenset(
        {
            "injected_model_se",
            "injected_zero_se",
            "control_model_se",
            "control_zero_se",
        }
    ),
    "s3_utility_did": frozenset(
        {
            "injected_candidate_utility",
            "injected_benchmark_hold_utility",
            "control_candidate_utility",
            "control_benchmark_hold_utility",
        }
    ),
}
_P1_S2_LEVEL_METRICS = frozenset(
    {"mean", "skill", "logloss", "agreement", "policy_utility_delta", "normalized_regret"}
)
_P1_S2_LEVEL_ARRAY_KEYS: Mapping[str, frozenset[str]] = {
    "mean": frozenset({"level_a_values", "level_b_values"}),
    "skill": frozenset(
        {
            "level_a_model_se",
            "level_a_zero_se",
            "level_b_model_se",
            "level_b_zero_se",
        }
    ),
    "logloss": frozenset({"level_a_values", "level_b_values"}),
    "agreement": frozenset({"level_a_values", "level_b_values"}),
    "policy_utility_delta": frozenset({"level_a_values", "level_b_values"}),
    "normalized_regret": frozenset(
        {
            "level_a_regret",
            "level_a_opportunity",
            "level_b_regret",
            "level_b_opportunity",
        }
    ),
}
_P1_METRIC_DEFAULT_DIRECTIONS: Mapping[str, str] = {
    "mse_delta": "negative",
    "skill": "positive",
    "logloss": "negative",
    "agreement": "positive",
    "policy_utility_delta": "positive",
    "normalized_regret": "negative",
    "s3_skill_did": "positive",
    "s3_utility_did": "positive",
}
_P1_ACTION_PROVENANCE_METRICS = frozenset(
    {"agreement", "policy_utility_delta", "normalized_regret", "s3_utility_did"}
)
_P1_S2_DIRECTIONS = frozenset(
    {"high_ge_medium", "high_le_medium", "medium_ge_low", "medium_le_low"}
)
_P1_S2_DIRECTION_SIGN: Mapping[str, str] = {
    "high_ge_medium": "positive",
    "high_le_medium": "negative",
    "medium_ge_low": "positive",
    "medium_le_low": "negative",
}
_P1_S2_LEVEL_METRIC_ALLOWED_DIRECTIONS: Mapping[str, frozenset[str]] = {
    "skill": frozenset({"high_ge_medium", "medium_ge_low"}),
    "agreement": frozenset({"high_ge_medium", "medium_ge_low"}),
    "policy_utility_delta": frozenset({"high_ge_medium", "medium_ge_low"}),
    "logloss": frozenset({"high_le_medium", "medium_le_low"}),
    "normalized_regret": frozenset({"high_le_medium", "medium_le_low"}),
}
_P1_INDEX_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024
_P1_INDEX_ARTIFACT_MAX_STARTS = 100_000_000
_P1_INDEX_METADATA_MAX_BYTES = 8 * 1024 * 1024
P1_REGRET_DOMAIN_TOL = 1e-12
P1_MBB_RESULT_SCHEMA = "unidream.p1.moving_block_result"
P1_MBB_RESULT_SCHEMA_VERSION = 1
_P1_RESULT_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024
_P1_RESULT_METADATA_MAX_BYTES = 8 * 1024 * 1024
_P1_PRODUCTION_RESULT_STATUS: Mapping[str, bool] = {
    "prereg_results_observed": False,
    "validation_results_observed": True,
    "outer_results_observed": False,
}


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise P1MBBError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise P1MBBError(f"{name} must be >= {minimum}")
    return normalized


def _strict_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P1MBBError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise P1MBBError(f"{name} must be valid UTF-8") from exc
    return value


def _strict_block_length(block_length: Any) -> int:
    length = _strict_int(block_length, name="block_length", minimum=1)
    if length not in P1_MBB_BLOCK_LENGTHS:
        raise P1MBBError(
            f"P1 block_length must be one of {P1_MBB_BLOCK_LENGTHS}, got {length}"
        )
    return length


def _normalize_unit(unit: Any = None, *, unit_code: Any = None) -> tuple[str, int]:
    if unit is not None and unit_code is not None:
        raise P1MBBError("unit and unit_code are aliases; supply exactly one")
    if unit is None:
        unit = unit_code
    if isinstance(unit, (bool, np.bool_)):
        raise P1MBBError("unit code must not be bool")
    if isinstance(unit, (int, np.integer)):
        code = int(unit)
        matches = [name for name, value in P1_MBB_UNIT_CODES.items() if value == code]
        if len(matches) != 1:
            raise P1MBBError(f"unknown P1 MBB unit code: {code}")
        return matches[0], code
    if not isinstance(unit, str):
        raise P1MBBError("unit must be one of the fixed P1 MBB unit names")
    name = unit.strip()
    if name not in P1_MBB_UNIT_CODES:
        raise P1MBBError(f"unknown P1 MBB unit: {unit!r}")
    return name, P1_MBB_UNIT_CODES[name]


def derive_p1_seed(
    unit: str | int | None = None,
    block_length: Any = None,
    seed_ordinal: Any = None,
    *,
    unit_code: Any = None,
    base_seed: Any = P1_MBB_BASE_SEED,
    seed: Any = None,
) -> int:
    """Derive the immutable P1 RNG seed from the preregistered formula.

    ``seed`` is intentionally an explicit rejected alias.  The P1 protocol
    uses the synthetic/S3 seed *ordinal*, not a caller-selected arbitrary
    random seed.  Likewise, callers may provide either a fixed unit name or
    its code, never both.
    """
    if seed is not None:
        raise P1MBBError("seed is not accepted; use the preregistered seed_ordinal")
    base = _strict_int(base_seed, name="base_seed", minimum=0)
    if base != P1_MBB_BASE_SEED:
        raise P1MBBError("P1 MBB base_seed is fixed at 20260830")
    name, code = _normalize_unit(unit, unit_code=unit_code)
    length = _strict_block_length(block_length)
    ordinal = _strict_int(seed_ordinal, name="seed_ordinal", minimum=0)
    if name.startswith("s3_") and ordinal != 0:
        raise P1MBBError("S3 P1 MBB has only seed_ordinal=0")
    if name.startswith("synthetic_") and ordinal > 9:
        raise P1MBBError("synthetic P1 MBB seed_ordinal must be in 0..9")
    return base + 100_000 * code + 1_000 * length + ordinal


# A short name is useful at call sites, but it preserves the same strict
# argument contract and does not introduce a second seed convention.
derive_seed = derive_p1_seed


def _validate_rng(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise P1MBBError("P1 MBB requires a numpy.random.Generator")
    return rng


def _n_blocks(n: int, block_length: int) -> int:
    return (n + block_length - 1) // block_length


def draw_non_circular_mbb_starts(
    n: Any,
    block_length: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one exact non-circular P1 MBB start vector.

    The call to ``Generator.integers`` intentionally mirrors the preregistered
    API exactly, including keyword arguments and the little-endian int64
    result.  No wraparound or valid-row compression is possible here.
    """
    n_int = _strict_int(n, name="n", minimum=1)
    length = _strict_block_length(block_length)
    if n_int < length:
        raise P1MBBError(f"P1 MBB requires n >= L; got n={n_int}, L={length}")
    generator = _validate_rng(rng)
    count = _n_blocks(n_int, length)
    try:
        starts = generator.integers(
            low=0,
            high=n_int - length + 1,
            size=count,
            endpoint=False,
            dtype=np.int64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1MBBError("P1 MBB start draw failed") from exc
    starts = np.asarray(starts)
    if starts.dtype != np.dtype("<i8") or starts.shape != (count,):
        raise P1MBBError("P1 MBB start draw did not return the required int64 shape")
    if np.any(starts < 0) or np.any(starts > n_int - length):
        raise P1MBBError("P1 MBB start draw escaped its non-circular range")
    result = np.array(starts, dtype="<i8", copy=True, order="C")
    result.setflags(write=False)
    return result


def materialize_non_circular_mbb_indices(
    starts: Any,
    block_length: Any,
    n: Any,
) -> np.ndarray:
    """Materialize ``starts[:,None] + arange(L)`` in C order and truncate."""
    n_int = _strict_int(n, name="n", minimum=1)
    length = _strict_block_length(block_length)
    if n_int < length:
        raise P1MBBError(f"P1 MBB requires n >= L; got n={n_int}, L={length}")
    try:
        values = np.asarray(starts)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError("MBB starts are not a valid array") from exc
    count = _n_blocks(n_int, length)
    if values.dtype != np.dtype("<i8") or values.shape != (count,):
        raise P1MBBError("MBB starts must be a one-dimensional little-endian int64 vector")
    if np.any(values < 0) or np.any(values > n_int - length):
        raise P1MBBError("MBB starts are outside the non-circular range")
    # C-order flattening is explicit: the first L output rows are one complete
    # block, then the next start begins.  There is no modulo operation.
    blocks = values[:, None] + np.arange(length, dtype=np.int64)
    indices = np.ascontiguousarray(blocks, dtype="<i8").reshape(-1, order="C")[:n_int]
    if indices.shape != (n_int,) or np.any(indices < 0) or np.any(indices >= n_int):
        raise P1MBBError("materialized MBB indices are outside the full primitive grid")
    result = np.array(indices, dtype="<i8", copy=True, order="C")
    result.setflags(write=False)
    return result


def draw_non_circular_mbb_indices(
    n: Any,
    block_length: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw and materialize one exact P1 MBB index vector."""
    starts = draw_non_circular_mbb_starts(n, block_length, rng)
    return materialize_non_circular_mbb_indices(starts, block_length, n)


def _starts_digest(starts: np.ndarray) -> str:
    return hashlib.sha256(starts.tobytes(order="C")).hexdigest()


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _strict_sha256(value: Any, *, name: str) -> str:
    """Validate a caller-supplied digest without accepting a self-described value."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P1MBBError(f"{name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1MBBError("P1 MBB artifact metadata is not canonical JSON") from exc


def _artifact_digest(metadata: Mapping[str, Any], starts: np.ndarray) -> str:
    payload = _metadata_bytes(metadata) + b"\0" + starts.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _declared_index_layout(
    payload: Mapping[str, Any],
    *,
    require_shape_fields: bool = False,
) -> tuple[int, int, tuple[int, int], int]:
    """Validate bounded layout metadata before touching the starts array."""
    if not isinstance(payload, Mapping):
        raise P1MBBError("P1 MBB artifact metadata must be an object")
    try:
        replicate_count = _strict_int(payload["replicates"], name="replicates", minimum=1)
        n_int = _strict_int(payload["n"], name="n", minimum=1)
        length = _strict_block_length(payload["block_length"])
    except KeyError as exc:
        raise P1MBBError(f"P1 MBB metadata is missing {exc.args[0]}") from exc
    if replicate_count != P1_MBB_REPLICATES:
        raise P1MBBError("P1 MBB replicates are fixed at 2000")
    if n_int < length:
        raise P1MBBError(f"P1 MBB requires n >= L; got n={n_int}, L={length}")
    count = _n_blocks(n_int, length)
    total_elements = replicate_count * count
    if total_elements > _P1_INDEX_ARTIFACT_MAX_STARTS:
        raise P1MBBError("P1 MBB start artifact exceeds its bounded element count")
    declared_bytes = total_elements * np.dtype("<i8").itemsize
    if declared_bytes > _P1_INDEX_ARTIFACT_MAX_BYTES:
        raise P1MBBError("P1 MBB start artifact exceeds its bounded byte count")
    expected_shape = (replicate_count, count)
    if require_shape_fields and (
        "starts_shape" not in payload or "starts_dtype" not in payload
    ):
        raise P1MBBError("P1 MBB metadata must declare starts_shape and starts_dtype")
    if "starts_dtype" in payload and payload["starts_dtype"] != "<i8":
        raise P1MBBError("P1 MBB metadata starts_dtype must be exactly '<i8'")
    if "starts_shape" in payload:
        shape = payload["starts_shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise P1MBBError("P1 MBB metadata starts_shape must contain two dimensions")
        try:
            declared_shape = (
                _strict_int(shape[0], name="starts_shape[0]", minimum=0),
                _strict_int(shape[1], name="starts_shape[1]", minimum=0),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise P1MBBError("P1 MBB metadata starts_shape is malformed") from exc
        if declared_shape != expected_shape:
            raise P1MBBError(
                f"P1 MBB metadata starts_shape must be {expected_shape}, got {declared_shape}"
            )
    return n_int, length, expected_shape, declared_bytes


@dataclass(frozen=True)
class P1MBBIndexArtifact:
    """Reproducible P1 MBB draw artifact containing every start vector."""

    unit: str
    support_id: str
    seed_ordinal: int
    block_length: int
    n: int
    starts: np.ndarray
    derived_seed: int
    replicates: int = P1_MBB_REPLICATES
    schema: str = P1_MBB_SCHEMA
    schema_version: int = P1_MBB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        unit, code = _normalize_unit(self.unit)
        del code
        support = _strict_text(self.support_id, name="support_id")
        expected_support = P1_MBB_UNIT_SUPPORTS[unit]
        if support != expected_support:
            raise P1MBBError(
                f"support_id {support!r} is not the fixed support for {unit}: {expected_support!r}"
            )
        ordinal = _strict_int(self.seed_ordinal, name="seed_ordinal", minimum=0)
        if unit.startswith("s3_") and ordinal != 0:
            raise P1MBBError("S3 P1 MBB has only seed_ordinal=0")
        if unit.startswith("synthetic_") and ordinal > 9:
            raise P1MBBError("synthetic P1 MBB seed_ordinal must be in 0..9")
        length = _strict_block_length(self.block_length)
        n_int = _strict_int(self.n, name="n", minimum=1)
        replicate_count = _strict_int(self.replicates, name="replicates", minimum=1)
        if self.schema != P1_MBB_SCHEMA or self.schema_version != P1_MBB_SCHEMA_VERSION:
            raise P1MBBError("unsupported P1 MBB index artifact schema")
        _, _, expected_shape, declared_bytes = _declared_index_layout(
            {
                "replicates": replicate_count,
                "n": n_int,
                "block_length": length,
            }
        )
        expected_seed = derive_p1_seed(
            unit,
            length,
            ordinal,
        )
        supplied_seed = _strict_int(self.derived_seed, name="derived_seed", minimum=0)
        if supplied_seed != expected_seed:
            raise P1MBBError("derived_seed does not match the fixed P1 formula")
        try:
            values = np.asarray(self.starts)
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise P1MBBError("P1 MBB starts are not a valid array") from exc
        if (
            values.dtype != np.dtype("<i8")
            or values.ndim != 2
            or not values.flags.c_contiguous
            or values.shape != expected_shape
            or values.size != expected_shape[0] * expected_shape[1]
            or values.nbytes != declared_bytes
        ):
            raise P1MBBError(
                f"starts must have little-endian int64 shape {expected_shape}, got {values.dtype}/{values.shape}"
            )
        if np.any(values < 0) or np.any(values > n_int - length):
            raise P1MBBError("P1 MBB artifact contains an out-of-range non-circular start")
        canonical = np.array(values, dtype="<i8", copy=True, order="C")
        canonical.setflags(write=False)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "support_id", support)
        object.__setattr__(self, "seed_ordinal", ordinal)
        object.__setattr__(self, "block_length", length)
        object.__setattr__(self, "n", n_int)
        object.__setattr__(self, "starts", canonical)
        object.__setattr__(self, "derived_seed", supplied_seed)
        object.__setattr__(self, "replicates", replicate_count)

    @property
    def starts_sha256(self) -> str:
        return _starts_digest(self.starts)

    @property
    def artifact_sha256(self) -> str:
        return _artifact_digest(self.metadata(), self.starts)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "unit": self.unit,
            "support_id": self.support_id,
            "seed_ordinal": self.seed_ordinal,
            "block_length": self.block_length,
            "n": self.n,
            "replicates": self.replicates,
            "derived_seed": self.derived_seed,
            "starts_dtype": "<i8",
            "starts_shape": list(self.starts.shape),
            "starts_sha256": self.starts_sha256,
        }

    def to_dict(self, *, include_starts: bool = True) -> dict[str, Any]:
        payload = self.metadata()
        payload["artifact_sha256"] = self.artifact_sha256
        if include_starts:
            payload["starts"] = self.starts.tolist()
        return payload

    def indices_for(self, replicate: Any) -> np.ndarray:
        index = _strict_int(replicate, name="replicate", minimum=0)
        if index >= self.replicates:
            raise P1MBBError(f"replicate must be < {self.replicates}")
        return materialize_non_circular_mbb_indices(
            self.starts[index],
            self.block_length,
            self.n,
        )

    def materialize_indices(self) -> np.ndarray:
        """Materialize all indices in ``(replicate, n)`` C order on demand."""
        materialized_bytes = self.replicates * self.n * np.dtype("<i8").itemsize
        if materialized_bytes > _P1_INDEX_ARTIFACT_MAX_BYTES:
            raise P1MBBError(
                "all P1 MBB indices exceed the bounded materialization size"
            )
        try:
            result = np.empty((self.replicates, self.n), dtype="<i8", order="C")
        except (MemoryError, OverflowError) as exc:
            raise P1MBBError("all P1 MBB indices cannot be materialized in memory") from exc
        for replicate in range(self.replicates):
            result[replicate] = self.indices_for(replicate)
        result.setflags(write=False)
        return result

    @classmethod
    def _from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_artifact_sha256: str | None,
        require_external_digest: bool,
        verify_deterministic_starts: bool,
    ) -> "P1MBBIndexArtifact":
        if not isinstance(payload, Mapping):
            raise P1MBBError("P1 MBB artifact must be an object")
        required = (
            "schema",
            "schema_version",
            "unit",
            "support_id",
            "seed_ordinal",
            "block_length",
            "n",
            "replicates",
            "derived_seed",
            "starts",
        )
        missing = [field for field in required if field not in payload]
        if missing:
            raise P1MBBError("P1 MBB artifact is missing: " + ", ".join(missing))
        _, _, expected_shape, declared_bytes = _declared_index_layout(payload)
        try:
            starts = np.asarray(payload["starts"])
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise P1MBBError("P1 MBB artifact starts are not a valid array") from exc
        if (
            starts.dtype != np.dtype("<i8")
            or starts.ndim != 2
            or not starts.flags.c_contiguous
            or starts.shape != expected_shape
            or starts.size != expected_shape[0] * expected_shape[1]
            or starts.nbytes != declared_bytes
        ):
            raise P1MBBError(
                "P1 MBB artifact starts must be a C-contiguous little-endian "
                f"int64 array of shape {expected_shape}"
            )
        starts = np.array(starts, dtype="<i8", copy=True, order="C")
        artifact = cls(
            unit=payload["unit"],
            support_id=payload["support_id"],
            seed_ordinal=payload["seed_ordinal"],
            block_length=payload["block_length"],
            n=payload["n"],
            starts=starts,
            derived_seed=payload["derived_seed"],
            replicates=payload["replicates"],
            schema=payload["schema"],
            schema_version=payload["schema_version"],
        )
        if require_external_digest and expected_artifact_sha256 is None:
            raise P1MBBError(
                "production P1 MBB artifact loading requires an external expected_artifact_sha256"
            )
        if "starts_sha256" not in payload:
            if require_external_digest:
                raise P1MBBError("production P1 MBB artifact requires starts_sha256")
        elif _strict_sha256(payload["starts_sha256"], name="starts_sha256") != artifact.starts_sha256:
            raise P1MBBError("P1 MBB starts hash mismatch")
        if "artifact_sha256" not in payload:
            if require_external_digest:
                raise P1MBBError("production P1 MBB artifact requires artifact_sha256")
        elif _strict_sha256(payload["artifact_sha256"], name="artifact_sha256") != artifact.artifact_sha256:
            raise P1MBBError("P1 MBB artifact hash mismatch")
        if require_external_digest:
            external_digest = _strict_sha256(
                expected_artifact_sha256,
                name="expected_artifact_sha256",
            )
            if external_digest != artifact.artifact_sha256:
                raise P1MBBError(
                    "P1 MBB artifact does not match the independent expected_artifact_sha256"
                )
        if verify_deterministic_starts:
            # The fixed seed formula is an independent binding for the entire
            # starts matrix.  A caller cannot make a forged matrix acceptable by
            # merely recomputing and echoing its own hashes.
            try:
                rng = np.random.default_rng(artifact.derived_seed)
                for replicate in range(artifact.replicates):
                    expected = draw_non_circular_mbb_starts(
                        artifact.n,
                        artifact.block_length,
                        rng,
                    )
                    if not np.array_equal(expected, artifact.starts[replicate]):
                        raise P1MBBError(
                            f"P1 MBB starts do not match the fixed RNG stream at replicate {replicate}"
                        )
            except (MemoryError, OverflowError) as exc:
                raise P1MBBError("P1 MBB deterministic starts verification failed") from exc
        return artifact

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_artifact_sha256: str | None = None,
    ) -> "P1MBBIndexArtifact":
        """Load a production artifact with external and deterministic binding."""
        return cls._from_dict(
            payload,
            expected_artifact_sha256=expected_artifact_sha256,
            require_external_digest=True,
            verify_deterministic_starts=True,
        )

    @classmethod
    def from_dict_fixture(cls, payload: Mapping[str, Any]) -> "P1MBBIndexArtifact":
        """Load a relaxed in-memory fixture; never use this for promotion."""
        return cls._from_dict(
            payload,
            expected_artifact_sha256=None,
            require_external_digest=False,
            verify_deterministic_starts=False,
        )


def build_p1_mbb_index_artifact(
    n: Any,
    *,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    seed_ordinal: Any,
    block_length: Any,
    replicates: Any = P1_MBB_REPLICATES,
    base_seed: Any = P1_MBB_BASE_SEED,
    seed: Any = None,
) -> P1MBBIndexArtifact:
    """Draw all 2,000 start vectors with one RNG lifecycle and persist them."""
    if seed is not None:
        raise P1MBBError("seed is not accepted; use the preregistered seed_ordinal")
    n_int = _strict_int(n, name="n", minimum=1)
    length = _strict_block_length(block_length)
    if n_int < length:
        raise P1MBBError(f"P1 MBB requires n >= L; got n={n_int}, L={length}")
    replicate_count = _strict_int(replicates, name="replicates", minimum=1)
    if replicate_count != P1_MBB_REPLICATES:
        raise P1MBBError("P1 MBB replicates are fixed at 2000")
    name, code = _normalize_unit(unit, unit_code=unit_code)
    support = _strict_text(support_id, name="support_id")
    if support != P1_MBB_UNIT_SUPPORTS[name]:
        raise P1MBBError(
            f"support_id {support!r} is not the fixed support for {name}: "
            f"{P1_MBB_UNIT_SUPPORTS[name]!r}"
        )
    derived_seed = derive_p1_seed(
        name,
        length,
        seed_ordinal,
        base_seed=base_seed,
    )
    ordinal = _strict_int(seed_ordinal, name="seed_ordinal", minimum=0)
    count = _n_blocks(n_int, length)
    total_starts = P1_MBB_REPLICATES * count
    if total_starts > _P1_INDEX_ARTIFACT_MAX_STARTS:
        raise P1MBBError("P1 MBB start artifact exceeds its bounded size")
    if total_starts * np.dtype("<i8").itemsize > _P1_INDEX_ARTIFACT_MAX_BYTES:
        raise P1MBBError("P1 MBB start artifact exceeds its bounded byte count")
    try:
        rng = np.random.default_rng(derived_seed)
        starts = np.empty((P1_MBB_REPLICATES, count), dtype="<i8", order="C")
        # Do not vectorize this draw: the registered lifecycle is one exact
        # integers(...) call per replicate, in b=0..1999 order.
        for replicate in range(P1_MBB_REPLICATES):
            starts[replicate] = draw_non_circular_mbb_starts(n_int, length, rng)
    except (MemoryError, OverflowError) as exc:
        raise P1MBBError("P1 MBB start artifact cannot be allocated") from exc
    return P1MBBIndexArtifact(
        unit=name,
        support_id=support,
        seed_ordinal=ordinal,
        block_length=length,
        n=n_int,
        starts=starts,
        derived_seed=derived_seed,
        replicates=P1_MBB_REPLICATES,
    )


def save_p1_mbb_index_artifact(path: str | Path, artifact: P1MBBIndexArtifact) -> str:
    """Persist all start vectors losslessly and atomically as a NumPy archive."""
    if not isinstance(artifact, P1MBBIndexArtifact):
        raise P1MBBError("save requires a P1MBBIndexArtifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = artifact.metadata()
    metadata["artifact_sha256"] = artifact.artifact_sha256
    encoded_metadata = _metadata_bytes(metadata)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                starts=artifact.starts,
                metadata=np.frombuffer(encoded_metadata, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > _P1_INDEX_ARTIFACT_MAX_BYTES:
            raise P1MBBError("P1 MBB index artifact exceeds the file-size limit")
        temporary.replace(output)
        temporary = None
    except P1MBBError:
        raise
    except (OSError, TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError(f"could not persist P1 MBB index artifact {output}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return artifact.artifact_sha256


def _inspect_index_archive(source: Any) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    """Inspect NPZ member declarations without extracting a potentially huge array."""
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) != 2 or {info.filename for info in infos} != {
                "starts.npy",
                "metadata.npy",
            }:
                raise P1MBBError("P1 MBB index artifact has unexpected archive members")
            starts_info = next(info for info in infos if info.filename == "starts.npy")
            metadata_info = next(info for info in infos if info.filename == "metadata.npy")
            for info in (starts_info, metadata_info):
                if info.is_dir():
                    raise P1MBBError("P1 MBB index archive members must be regular files")
                # NumPy's NPZ writer emits ordinary Unix permission bits.  If
                # an archive records an explicit non-regular Unix type (for
                # example a symlink), reject it before opening the member.
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type not in (0, stat.S_IFREG):
                    raise P1MBBError("P1 MBB index archive contains a non-regular member")
            if starts_info.file_size > _P1_INDEX_ARTIFACT_MAX_BYTES:
                raise P1MBBError("P1 MBB starts member exceeds the file-size limit")
            if metadata_info.file_size > _P1_INDEX_METADATA_MAX_BYTES:
                raise P1MBBError("P1 MBB metadata member exceeds the file-size limit")
            if starts_info.compress_size > _P1_INDEX_ARTIFACT_MAX_BYTES:
                raise P1MBBError("P1 MBB compressed starts member exceeds the file-size limit")
            if metadata_info.compress_size > _P1_INDEX_METADATA_MAX_BYTES:
                raise P1MBBError("P1 MBB compressed metadata member exceeds the file-size limit")
            return starts_info, metadata_info
    except P1MBBError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise P1MBBError("P1 MBB index artifact archive is malformed") from exc


def _inspect_npy_member_header(
    source: Any,
    member_name: str,
    expected_member: zipfile.ZipInfo,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...] | None,
    max_payload_bytes: int,
) -> tuple[tuple[int, ...], int]:
    """Validate one NPY header and declared payload before NumPy materializes it."""
    header_cap = 64 * 1024
    try:
        expected_itemsize = np.dtype(expected_dtype).itemsize
        with zipfile.ZipFile(source, mode="r") as archive:
            member = archive.getinfo(member_name)
            if (
                member.file_size != expected_member.file_size
                or member.compress_size != expected_member.compress_size
                or member.CRC != expected_member.CRC
            ):
                raise P1MBBError("P1 MBB index archive member changed during parsing")
            with archive.open(member, mode="r") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
                    raise P1MBBError(f"P1 MBB {member_name} has an invalid NPY header")
                major, minor = prefix[6], prefix[7]
                if (major, minor) == (1, 0):
                    length_bytes = stream.read(2)
                    header_prefix = 10
                    encoding = "latin1"
                elif (major, minor) in ((2, 0), (3, 0)):
                    length_bytes = stream.read(4)
                    header_prefix = 12
                    encoding = "utf-8" if major == 3 else "latin1"
                else:
                    raise P1MBBError(f"P1 MBB {member_name} uses an unsupported NPY version")
                if len(length_bytes) != header_prefix - 8:
                    raise P1MBBError(f"P1 MBB {member_name} has a truncated NPY header")
                header_length = int.from_bytes(length_bytes, byteorder="little", signed=False)
                if header_length <= 0 or header_length > header_cap:
                    raise P1MBBError(f"P1 MBB {member_name} NPY header is too large")
                header_bytes = stream.read(header_length)
                if len(header_bytes) != header_length:
                    raise P1MBBError(f"P1 MBB {member_name} has a truncated NPY header")
                try:
                    header = ast.literal_eval(header_bytes.decode(encoding))
                except (UnicodeError, SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
                    raise P1MBBError(f"P1 MBB {member_name} NPY header is malformed") from exc
                if not isinstance(header, dict) or set(header) != {
                    "descr",
                    "fortran_order",
                    "shape",
                }:
                    raise P1MBBError(f"P1 MBB {member_name} NPY header fields are malformed")
                if header["descr"] != expected_dtype:
                    raise P1MBBError(f"P1 MBB {member_name} NPY dtype is not {expected_dtype!r}")
                if type(header["fortran_order"]) is not bool or header["fortran_order"]:
                    raise P1MBBError(f"P1 MBB {member_name} NPY array must be C-order")
                shape_value = header["shape"]
                if not isinstance(shape_value, tuple):
                    raise P1MBBError(f"P1 MBB {member_name} NPY shape is malformed")
                shape: list[int] = []
                for dimension in shape_value:
                    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                        raise P1MBBError(f"P1 MBB {member_name} NPY shape is malformed")
                    shape.append(dimension)
                normalized_shape = tuple(shape)
                if expected_shape is not None and normalized_shape != expected_shape:
                    raise P1MBBError(
                        f"P1 MBB {member_name} NPY shape must be {expected_shape}, got {normalized_shape}"
                    )
                elements = 1
                for dimension in normalized_shape:
                    if dimension and elements > max_payload_bytes // expected_itemsize // dimension:
                        raise P1MBBError(f"P1 MBB {member_name} NPY payload exceeds its byte limit")
                    elements *= dimension
                payload_bytes = elements * expected_itemsize
                if payload_bytes > max_payload_bytes:
                    raise P1MBBError(f"P1 MBB {member_name} NPY payload exceeds its byte limit")
                declared_file_size = header_prefix + header_length + payload_bytes
                if member.file_size != declared_file_size:
                    raise P1MBBError(
                        f"P1 MBB {member_name} NPY member size contradicts its header"
                    )
                return normalized_shape, payload_bytes
    except P1MBBError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, OverflowError, MemoryError, EOFError, KeyError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise P1MBBError(f"P1 MBB {member_name} NPY header is unreadable") from exc


def _open_regular_index_artifact(source: Path) -> tuple[Any, int, tuple[Any, ...]]:
    """Open one regular inode, rejecting links and path races before parsing."""
    try:
        link_stat = source.lstat()
    except (OSError, ValueError) as exc:
        raise P1MBBError(f"could not stat P1 MBB index artifact {source}") from exc
    if not stat.S_ISREG(link_stat.st_mode):
        raise P1MBBError("P1 MBB index artifact must be a regular file, not a link/device/pipe")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    non_blocking = getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    handle: Any = None
    try:
        descriptor = os.open(source, os.O_RDONLY | no_follow | non_blocking)
        handle = os.fdopen(descriptor, mode="rb", closefd=True)
        descriptor = -1
        opened_stat = os.fstat(handle.fileno())
    except (OSError, ValueError) as exc:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        elif descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise P1MBBError(f"could not open P1 MBB index artifact {source}") from exc
    if not stat.S_ISREG(opened_stat.st_mode):
        handle.close()
        raise P1MBBError("P1 MBB index artifact must remain a regular file")
    signature = (
        opened_stat.st_dev,
        opened_stat.st_ino,
        opened_stat.st_size,
        opened_stat.st_mtime_ns,
        opened_stat.st_ctime_ns,
    )
    return handle, int(opened_stat.st_size), signature


def _assert_index_artifact_unchanged(handle: Any, signature: tuple[Any, ...]) -> None:
    try:
        current = os.fstat(handle.fileno())
    except (OSError, ValueError) as exc:
        raise P1MBBError("P1 MBB index artifact disappeared during parsing") from exc
    current_signature = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if current_signature != signature:
        raise P1MBBError("P1 MBB index artifact changed during parsing")


def _load_p1_mbb_index_artifact(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None,
    production: bool,
) -> P1MBBIndexArtifact:
    """Load a lossless P1 MBB start archive after bounded structural checks."""
    source = Path(path)
    handle: Any = None
    try:
        handle, source_size, source_signature = _open_regular_index_artifact(source)
        if source_size > _P1_INDEX_ARTIFACT_MAX_BYTES:
            raise P1MBBError("P1 MBB index artifact exceeds the file-size limit")
        starts_info, metadata_info = _inspect_index_archive(handle)
        _assert_index_artifact_unchanged(handle, source_signature)

        # Inspect both NPY headers before asking NumPy to decompress either
        # payload.  The metadata header is the only safe source of the starts
        # shape, so it is parsed first and independently bounded.
        metadata_header_shape, metadata_payload_bytes = _inspect_npy_member_header(
            handle,
            "metadata.npy",
            metadata_info,
            expected_dtype="|u1",
            expected_shape=None,
            max_payload_bytes=_P1_INDEX_METADATA_MAX_BYTES,
        )
        if len(metadata_header_shape) != 1:
            raise P1MBBError("P1 MBB metadata NPY array must be one-dimensional")
        _assert_index_artifact_unchanged(handle, source_signature)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            if set(archive.files) != {"starts", "metadata"}:
                raise P1MBBError("P1 MBB index artifact has unexpected archive fields")
            metadata_bytes = np.asarray(archive["metadata"])
            if (
                metadata_bytes.dtype != np.dtype("uint8")
                or metadata_bytes.ndim != 1
                or metadata_bytes.nbytes > _P1_INDEX_METADATA_MAX_BYTES
                or metadata_bytes.shape != metadata_header_shape
                or metadata_bytes.nbytes != metadata_payload_bytes
            ):
                raise P1MBBError("P1 MBB metadata bytes are malformed")
            metadata_bytes = np.array(metadata_bytes, dtype=np.uint8, copy=True, order="C")
        _assert_index_artifact_unchanged(handle, source_signature)
        try:
            metadata = json.loads(bytes(metadata_bytes).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise P1MBBError("P1 MBB metadata JSON is malformed") from exc
        if not isinstance(metadata, Mapping):
            raise P1MBBError("P1 MBB metadata must be an object")
        _, _, expected_shape, declared_bytes = _declared_index_layout(
            metadata,
            require_shape_fields=True,
        )
        starts_header_shape, starts_payload_bytes = _inspect_npy_member_header(
            handle,
            "starts.npy",
            starts_info,
            expected_dtype="<i8",
            expected_shape=expected_shape,
            max_payload_bytes=_P1_INDEX_ARTIFACT_MAX_BYTES,
        )
        if starts_payload_bytes != declared_bytes or starts_header_shape != expected_shape:
            raise P1MBBError("P1 MBB starts payload contradicts its declared shape")
        _assert_index_artifact_unchanged(handle, source_signature)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            if set(archive.files) != {"starts", "metadata"}:
                raise P1MBBError("P1 MBB index artifact has unexpected archive fields")
            starts = np.asarray(archive["starts"])
        _assert_index_artifact_unchanged(handle, source_signature)
        payload = dict(metadata)
        payload["starts"] = starts
        if production:
            artifact = P1MBBIndexArtifact.from_dict(
                payload,
                expected_artifact_sha256=expected_artifact_sha256,
            )
        else:
            artifact = P1MBBIndexArtifact.from_dict_fixture(payload)
        if artifact.starts.shape != expected_shape or artifact.starts.nbytes != declared_bytes:
            raise P1MBBError("P1 MBB starts materialization contradicts its declared shape")
        if metadata_info.file_size > _P1_INDEX_METADATA_MAX_BYTES:
            raise P1MBBError("P1 MBB metadata member exceeds the file-size limit")
        return artifact
    except P1MBBError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, MemoryError, EOFError, KeyError, json.JSONDecodeError, UnicodeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise P1MBBError(f"could not load P1 MBB index artifact {source}") from exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


def load_p1_mbb_index_artifact(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> P1MBBIndexArtifact:
    """Load a production index artifact with an independent digest binding."""
    return _load_p1_mbb_index_artifact(
        path,
        expected_artifact_sha256=expected_artifact_sha256,
        production=True,
    )


def load_p1_mbb_index_artifact_fixture(path: str | Path) -> P1MBBIndexArtifact:
    """Load a relaxed fixture archive; this boundary cannot promote results."""
    return _load_p1_mbb_index_artifact(
        path,
        expected_artifact_sha256=None,
        production=False,
    )


def _strict_float64_vector(value: Any, *, name: str, n: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError(f"{name} is not a valid numeric vector") from exc
    if array.dtype != np.dtype("<f8") or array.ndim != 1 or array.shape != (n,):
        raise P1MBBError(f"{name} must be a little-endian float64 vector of shape ({n},)")
    if np.isinf(array).any():
        raise P1MBBError(f"{name} contains infinity")
    return np.array(array, dtype="<f8", copy=True, order="C")


def _strict_bool_mask(value: Any, *, name: str, n: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError(f"{name} is not a valid mask") from exc
    if array.dtype != np.dtype(np.bool_) or array.ndim != 1 or array.shape != (n,):
        raise P1MBBError(f"{name} must be a strict bool vector of shape ({n},)")
    return np.array(array, dtype=np.bool_, copy=True, order="C")


def p1_mask_sha256(mask: Any) -> str:
    """Return the canonical digest of a full-grid boolean mask.

    The production wrapper compares this value with an independently supplied
    digest.  It is intentionally just the C-order bool payload because the
    vector length is already bound to the metric artifact ``n``.
    """
    try:
        values = np.asarray(mask)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError("mask is not a valid array for hashing") from exc
    if values.dtype != np.dtype(np.bool_) or values.ndim != 1:
        raise P1MBBError("mask hash requires a one-dimensional strict bool vector")
    canonical = np.ascontiguousarray(values, dtype=np.bool_)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_paired_inputs(
    candidate_values: Any,
    baseline_values: Any,
    candidate_mask: Any,
    baseline_mask: Any,
    artifact: P1MBBIndexArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(artifact, P1MBBIndexArtifact):
        raise P1MBBError("paired P1 MBB requires a P1MBBIndexArtifact")
    candidate = _strict_float64_vector(
        candidate_values,
        name="candidate_values",
        n=artifact.n,
    )
    baseline = _strict_float64_vector(
        baseline_values,
        name="baseline_values",
        n=artifact.n,
    )
    candidate_ok = _strict_bool_mask(
        candidate_mask,
        name="candidate_mask",
        n=artifact.n,
    )
    baseline_ok = _strict_bool_mask(
        baseline_mask,
        name="baseline_mask",
        n=artifact.n,
    )
    if not np.array_equal(candidate_ok, baseline_ok):
        raise P1MBBError(
            "paired P1 MBB requires identical candidate/baseline masks; "
            "pass the fixed common mask to both arms"
        )
    if not candidate_ok.any():
        raise P1MBBError("paired P1 MBB has zero valid primitive records")
    if np.isnan(candidate[candidate_ok]).any() or np.isnan(baseline[candidate_ok]).any():
        raise P1MBBError("paired P1 MBB valid primitive values must be finite")
    # N/A rows may retain NaN values, but an infinity is never a canonical
    # masked value and was rejected above for every row.
    return candidate, baseline, candidate_ok


def _validate_metric(metric: Any) -> str:
    if not isinstance(metric, str) or metric not in P1_PAIRED_MEAN_METRICS:
        raise P1MBBError(
            "metric must be one of the preregistered paired metrics: "
            + ", ".join(sorted(P1_PAIRED_MEAN_METRICS))
        )
    return metric


def _validate_direction(direction: Any) -> str:
    if direction not in {"positive", "negative"}:
        raise P1MBBError("direction must be exactly 'positive' or 'negative'")
    return direction


def _p_value(samples: np.ndarray, *, direction: str) -> float:
    if direction == "positive":
        count = int(np.count_nonzero(samples <= 0.0))
    else:
        count = int(np.count_nonzero(samples >= 0.0))
    return float((1 + count) / (len(samples) + 1))


def _validate_recompute_metric(metric: Any) -> str:
    if not isinstance(metric, str) or metric not in P1_RECOMPUTE_METRICS:
        raise P1MBBError(
            "metric must be one of the preregistered recomputation metrics: "
            + ", ".join(sorted(P1_RECOMPUTE_METRICS))
        )
    return metric


def _validate_s2_direction(level_direction: Any) -> str:
    if not isinstance(level_direction, str) or level_direction not in _P1_S2_DIRECTIONS:
        raise P1MBBError(
            "level_direction must be one of the fixed S2 directions: "
            + ", ".join(sorted(_P1_S2_DIRECTIONS))
        )
    return level_direction


def _validate_s2_level_metric(level_metric: Any) -> str:
    if not isinstance(level_metric, str) or level_metric not in _P1_S2_LEVEL_METRICS:
        raise P1MBBError(
            "level_metric must be one of the fixed S2 metrics: "
            + ", ".join(sorted(_P1_S2_LEVEL_METRICS))
        )
    return level_metric


def _validate_s2_metric_direction(level_metric: str, level_direction: str) -> None:
    allowed = _P1_S2_LEVEL_METRIC_ALLOWED_DIRECTIONS.get(level_metric)
    if allowed is not None and level_direction not in allowed:
        raise P1MBBError(
            f"S2 {level_metric} does not permit level_direction={level_direction!r}"
        )


def _resolve_metric_direction(
    metric: str,
    direction: Any,
    *,
    level_direction: str | None,
) -> str:
    if metric == "s2_contrast":
        if level_direction is None:
            raise P1MBBError("s2_contrast requires a fixed level_direction")
        expected = _P1_S2_DIRECTION_SIGN[level_direction]
    else:
        expected = _P1_METRIC_DEFAULT_DIRECTIONS[metric]
    if direction is None:
        return expected
    resolved = _validate_direction(direction)
    # The preregistered metric definitions own their favorable sign.  Letting
    # a caller flip it would invert the one-sided p-value and could turn an
    # unfavorable result into a false pass.
    if resolved != expected:
        raise P1MBBError(
            f"{metric} requires direction={expected!r}, got {resolved!r}"
        )
    return resolved


def _prepare_recompute_arrays(
    arrays: Mapping[str, Any],
    mask: Any,
    *,
    expected_keys: frozenset[str],
    n: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    try:
        actual_keys = frozenset(arrays)
    except (TypeError, ValueError) as exc:
        raise P1MBBError("metric arrays must be a mapping with fixed field names") from exc
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise P1MBBError("metric array fields do not match the fixed contract (" + "; ".join(details) + ")")
    if n is None:
        first_name = sorted(expected_keys)[0]
        try:
            first = np.asarray(arrays[first_name])
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise P1MBBError(f"{first_name} is not a valid metric vector") from exc
        if first.ndim != 1:
            raise P1MBBError(f"{first_name} must be one-dimensional")
        n = int(first.shape[0])
    if n < 1:
        raise P1MBBError("metric grid must contain at least one row")
    common_mask = _strict_bool_mask(mask, name="mask", n=n)
    if not common_mask.any():
        raise P1MBBError("metric comparison has zero valid primitive records")
    validated: dict[str, np.ndarray] = {}
    for name in sorted(expected_keys):
        values = _strict_float64_vector(arrays[name], name=name, n=n)
        if np.isnan(values[common_mask]).any():
            raise P1MBBError(f"{name} has NaN on a valid common-mask row")
        validated[name] = values
    return validated, common_mask, n


def _validate_optional_arm_masks(
    common_mask: np.ndarray,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
) -> None:
    if candidate_mask is None and baseline_mask is None:
        return
    if candidate_mask is None or baseline_mask is None:
        raise P1MBBError("candidate_mask and baseline_mask must be supplied together")
    candidate = _strict_bool_mask(
        candidate_mask,
        name="candidate_mask",
        n=len(common_mask),
    )
    baseline = _strict_bool_mask(
        baseline_mask,
        name="baseline_mask",
        n=len(common_mask),
    )
    if not np.array_equal(candidate, baseline) or not np.array_equal(candidate, common_mask):
        raise P1MBBError(
            "paired P1 metric arms must use the identical fixed common mask"
        )


def _validate_required_arm_masks(
    common_mask: np.ndarray,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Require both externally produced arm masks at the production boundary."""
    if candidate_mask is None or baseline_mask is None:
        raise P1MBBError(
            "production P1 bootstrap requires candidate_mask and baseline_mask; "
            "arm masks may not be omitted"
        )
    candidate = _strict_bool_mask(
        candidate_mask,
        name="candidate_mask",
        n=len(common_mask),
    )
    baseline = _strict_bool_mask(
        baseline_mask,
        name="baseline_mask",
        n=len(common_mask),
    )
    if not np.array_equal(candidate, baseline) or not np.array_equal(candidate, common_mask):
        raise P1MBBError(
            "production P1 metric arms must use the identical externally bound common mask"
        )
    return candidate, baseline


def _validate_production_provenance(
    metric: str,
    common_mask: np.ndarray,
    *,
    level_metric: str | None,
    provenance: Mapping[str, Any] | None,
    expected_common_mask_sha256: Any,
    expected_common_mask_field: Any,
    expected_source_result_sha256: Any,
    expected_action_primitive_payload_sha256: Any,
    expected_action_primitive_schema_sha256: Any,
    expected_action_primitive_content_sha256: Any,
    expected_forecast_artifact_sha256: Any,
    expected_forecast_result_sha256: Any,
) -> dict[str, str]:
    """Authenticate action/forecast provenance before a production bootstrap.

    All expected values are supplied by the authenticated upstream artifact
    loader.  Values copied only from the candidate result are never used as the
    source of truth; the computed mask digest is checked independently here.
    """
    if not isinstance(provenance, Mapping):
        raise P1MBBError("production P1 bootstrap requires external provenance metadata")
    kind_value = provenance.get("kind", provenance.get("primitive_kind"))
    if kind_value not in {"action", "forecast"}:
        raise P1MBBError("production provenance kind must be exactly 'action' or 'forecast'")
    kind = str(kind_value)
    expected_kind = _expected_provenance_kind(metric, level_metric)
    if kind != expected_kind:
        raise P1MBBError(
            f"production provenance kind {kind!r} does not match {metric}/{level_metric} ({expected_kind!r})"
        )
    common_digest = _strict_sha256(
        expected_common_mask_sha256,
        name="expected_common_mask_sha256",
    )
    if p1_mask_sha256(common_mask) != common_digest:
        raise P1MBBError("production common mask does not match its external digest")
    field = _strict_text(
        expected_common_mask_field,
        name="expected_common_mask_field",
    )
    if field != "common_mask":
        raise P1MBBError(
            "production provenance must bind the registered common_mask field"
        )
    if provenance.get("common_mask_sha256") != common_digest:
        raise P1MBBError("production provenance common mask digest mismatch")
    if provenance.get("common_mask_field") != field:
        raise P1MBBError("production provenance common mask field mismatch")

    validated: dict[str, str] = {
        "kind": kind,
        "common_mask_sha256": common_digest,
        "common_mask_field": field,
    }
    if kind == "action":
        action_values = {
            "action_primitive_payload_sha256": expected_action_primitive_payload_sha256,
            "action_primitive_schema_sha256": expected_action_primitive_schema_sha256,
            "action_primitive_content_sha256": expected_action_primitive_content_sha256,
            "source_result_sha256": expected_source_result_sha256,
        }
        for name, expected in action_values.items():
            digest = _strict_sha256(expected, name=f"expected_{name}")
            if provenance.get(name) != digest:
                raise P1MBBError(f"production provenance {name} mismatch")
            validated[name] = digest
    else:
        artifact_digest = _strict_sha256(
            expected_forecast_artifact_sha256,
            name="expected_forecast_artifact_sha256",
        )
        result_digest = _strict_sha256(
            expected_forecast_result_sha256,
            name="expected_forecast_result_sha256",
        )
        if provenance.get("forecast_artifact_sha256") != artifact_digest:
            raise P1MBBError("production provenance forecast artifact digest mismatch")
        if provenance.get("forecast_result_sha256") != result_digest:
            raise P1MBBError("production provenance forecast result digest mismatch")
        validated["forecast_artifact_sha256"] = artifact_digest
        validated["forecast_result_sha256"] = result_digest
    # Echo only authenticated fields.  Additional caller-provided provenance is
    # deliberately ignored so an unregistered field cannot become evidence.
    return validated


def _validate_metric_indices(indices: Any, *, n: int) -> np.ndarray:
    try:
        values = np.asarray(indices)
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError("metric bootstrap indices are malformed") from exc
    if values.dtype != np.dtype("<i8") or values.ndim != 1 or values.size == 0:
        raise P1MBBError("metric bootstrap indices must be a non-empty little-endian int64 vector")
    if np.any(values < 0) or np.any(values >= n):
        raise P1MBBError("metric bootstrap indices escape the full primitive grid")
    return values


def _metric_views(
    arrays: Mapping[str, np.ndarray],
    common_mask: np.ndarray,
    *,
    indices: Any = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if indices is None:
        selected_mask = common_mask
        selected = {name: values for name, values in arrays.items()}
    else:
        index_values = _validate_metric_indices(indices, n=len(common_mask))
        selected_mask = common_mask[index_values]
        selected = {name: values[index_values] for name, values in arrays.items()}
    if not selected_mask.any():
        raise P1MBBError("metric bootstrap replicate has zero valid primitive records")
    for name, values in selected.items():
        if np.isnan(values[selected_mask]).any() or not np.isfinite(values[selected_mask]).all():
            raise P1MBBError(f"{name} is non-finite on a valid sampled row")
    return selected, selected_mask


def _valid_values(values: np.ndarray, mask: np.ndarray, *, name: str) -> np.ndarray:
    selected = values[mask]
    if selected.size == 0:
        raise P1MBBError(f"{name} has zero valid rows")
    if not np.isfinite(selected).all():
        raise P1MBBError(f"{name} is non-finite on a valid row")
    return selected


def _safe_mean(values: np.ndarray, *, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.mean(values, dtype=np.float64))
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise P1MBBError(f"{name} mean is non-finite") from exc
    if not np.isfinite(result):
        raise P1MBBError(f"{name} mean is non-finite")
    return result


def _safe_sum(values: np.ndarray, *, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.sum(values, dtype=np.float64))
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise P1MBBError(f"{name} sum is non-finite") from exc
    if not np.isfinite(result):
        raise P1MBBError(f"{name} sum is non-finite")
    return result


def _require_nonnegative(values: np.ndarray, *, name: str) -> None:
    if np.any(values < 0.0):
        raise P1MBBError(f"{name} must contain non-negative squared errors")


def _require_nonnegative_domain(values: np.ndarray, *, name: str) -> None:
    if np.any(values < 0.0):
        raise P1MBBError(f"{name} must contain non-negative values")


def _require_binary_agreement(values: np.ndarray, *, name: str) -> None:
    if not np.isin(values, (0.0, 1.0)).all():
        raise P1MBBError(f"{name} must contain only 0 or 1 agreement indicators")


def _require_regret_domain(
    regret: np.ndarray,
    opportunity: np.ndarray,
    *,
    regret_name: str,
    opportunity_name: str,
) -> None:
    if np.any(regret < -P1_REGRET_DOMAIN_TOL):
        raise P1MBBError(
            f"{regret_name} must be >= {-P1_REGRET_DOMAIN_TOL:g}"
        )
    _require_nonnegative_domain(opportunity, name=opportunity_name)


def _prepare_single_metric(
    arrays: Mapping[str, Any],
    mask: Any,
    *,
    metric: str,
    n: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    return _prepare_recompute_arrays(
        arrays,
        mask,
        expected_keys=_P1_METRIC_ARRAY_KEYS[metric],
        n=n,
    )


def _metric_value(
    metric: str,
    arrays: Mapping[str, np.ndarray],
    common_mask: np.ndarray,
    *,
    indices: Any = None,
    level_direction: str | None = None,
    level_metric: str | None = None,
) -> float:
    selected, selected_mask = _metric_views(arrays, common_mask, indices=indices)

    if metric == "mse_delta":
        _require_nonnegative(_valid_values(selected["candidate_se"], selected_mask, name="candidate_se"), name="candidate_se")
        _require_nonnegative(_valid_values(selected["baseline_se"], selected_mask, name="baseline_se"), name="baseline_se")
        value = _safe_mean(selected["candidate_se"][selected_mask], name="candidate_se") - _safe_mean(selected["baseline_se"][selected_mask], name="baseline_se")
    elif metric == "skill":
        model = _valid_values(selected["model_se"], selected_mask, name="model_se")
        zero = _valid_values(selected["zero_se"], selected_mask, name="zero_se")
        _require_nonnegative(model, name="model_se")
        _require_nonnegative(zero, name="zero_se")
        denominator = _safe_sum(zero, name="zero_se")
        if denominator <= 0.0:
            raise P1MBBError("skill denominator sum(SE_zero) must be positive")
        numerator = _safe_sum(model, name="model_se")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = float(1.0 - numerator / denominator)
        except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
            raise P1MBBError("skill is non-finite") from exc
    elif metric == "logloss":
        candidate_logloss = _valid_values(
            selected["candidate_logloss"],
            selected_mask,
            name="candidate_logloss",
        )
        baseline_logloss = _valid_values(
            selected["baseline_logloss"],
            selected_mask,
            name="baseline_logloss",
        )
        _require_nonnegative_domain(candidate_logloss, name="candidate_logloss")
        _require_nonnegative_domain(baseline_logloss, name="baseline_logloss")
        value = _safe_mean(candidate_logloss, name="candidate_logloss") - _safe_mean(baseline_logloss, name="baseline_logloss")
    elif metric == "agreement":
        candidate_agreement = _valid_values(
            selected["candidate_agreement"],
            selected_mask,
            name="candidate_agreement",
        )
        baseline_agreement = _valid_values(
            selected["baseline_agreement"],
            selected_mask,
            name="baseline_agreement",
        )
        _require_binary_agreement(candidate_agreement, name="candidate_agreement")
        _require_binary_agreement(baseline_agreement, name="baseline_agreement")
        value = _safe_mean(candidate_agreement, name="candidate_agreement") - _safe_mean(baseline_agreement, name="baseline_agreement")
    elif metric == "policy_utility_delta":
        value = _safe_mean(selected["candidate_utility"][selected_mask], name="candidate_utility") - _safe_mean(selected["benchmark_hold_utility"][selected_mask], name="benchmark_hold_utility")
    elif metric == "s2_contrast":
        if level_direction is None:
            raise P1MBBError("s2_contrast requires a fixed level_direction")
        level_name = _validate_s2_level_metric(level_metric or "mean")
        # The arrays are supplied in the registry's first-level/second-level
        # order (high,medium) or (medium,low); direction controls the gate and
        # p-value, while the returned contrast remains first-level minus
        # second-level.  Ratio/skill forms are recomputed independently for
        # both levels inside every replicate before this contrast is formed.
        _validate_s2_direction(level_direction)
        _validate_s2_metric_direction(level_name, level_direction)
        if level_name in {"mean", "logloss", "agreement", "policy_utility_delta"}:
            level_a_values = _valid_values(
                selected["level_a_values"],
                selected_mask,
                name="level_a_values",
            )
            level_b_values = _valid_values(
                selected["level_b_values"],
                selected_mask,
                name="level_b_values",
            )
            if level_name == "logloss":
                _require_nonnegative_domain(level_a_values, name="level_a_values")
                _require_nonnegative_domain(level_b_values, name="level_b_values")
            elif level_name == "agreement":
                _require_binary_agreement(level_a_values, name="level_a_values")
                _require_binary_agreement(level_b_values, name="level_b_values")
            value = _safe_mean(level_a_values, name="level_a_values") - _safe_mean(level_b_values, name="level_b_values")
        elif level_name == "skill":
            level_a_model = _valid_values(selected["level_a_model_se"], selected_mask, name="level_a_model_se")
            level_a_zero = _valid_values(selected["level_a_zero_se"], selected_mask, name="level_a_zero_se")
            level_b_model = _valid_values(selected["level_b_model_se"], selected_mask, name="level_b_model_se")
            level_b_zero = _valid_values(selected["level_b_zero_se"], selected_mask, name="level_b_zero_se")
            for values, name in (
                (level_a_model, "level_a_model_se"),
                (level_a_zero, "level_a_zero_se"),
                (level_b_model, "level_b_model_se"),
                (level_b_zero, "level_b_zero_se"),
            ):
                _require_nonnegative(values, name=name)
            level_a_denominator = _safe_sum(level_a_zero, name="level_a_zero_se")
            level_b_denominator = _safe_sum(level_b_zero, name="level_b_zero_se")
            if level_a_denominator <= 0.0 or level_b_denominator <= 0.0:
                raise P1MBBError("S2 skill contrast requires positive level zero-error denominators")
            level_a_skill = 1.0 - _safe_sum(level_a_model, name="level_a_model_se") / level_a_denominator
            level_b_skill = 1.0 - _safe_sum(level_b_model, name="level_b_model_se") / level_b_denominator
            value = float(level_a_skill - level_b_skill)
        else:  # normalized_regret
            level_a_regret = _valid_values(selected["level_a_regret"], selected_mask, name="level_a_regret")
            level_a_opportunity = _valid_values(selected["level_a_opportunity"], selected_mask, name="level_a_opportunity")
            level_b_regret = _valid_values(selected["level_b_regret"], selected_mask, name="level_b_regret")
            level_b_opportunity = _valid_values(selected["level_b_opportunity"], selected_mask, name="level_b_opportunity")
            _require_regret_domain(
                level_a_regret,
                level_a_opportunity,
                regret_name="level_a_regret",
                opportunity_name="level_a_opportunity",
            )
            _require_regret_domain(
                level_b_regret,
                level_b_opportunity,
                regret_name="level_b_regret",
                opportunity_name="level_b_opportunity",
            )
            level_a_denominator = _safe_sum(level_a_opportunity, name="level_a_opportunity")
            level_b_denominator = _safe_sum(level_b_opportunity, name="level_b_opportunity")
            if level_a_denominator <= 0.0 or level_b_denominator <= 0.0:
                raise P1MBBError("S2 normalized regret contrast requires positive level opportunity denominators")
            level_a_ratio = _safe_sum(level_a_regret, name="level_a_regret") / level_a_denominator
            level_b_ratio = _safe_sum(level_b_regret, name="level_b_regret") / level_b_denominator
            value = float(level_a_ratio - level_b_ratio)
    elif metric == "normalized_regret":
        regret = _valid_values(selected["regret"], selected_mask, name="regret")
        opportunity = _valid_values(selected["opportunity"], selected_mask, name="opportunity")
        _require_regret_domain(
            regret,
            opportunity,
            regret_name="regret",
            opportunity_name="opportunity",
        )
        denominator = _safe_sum(opportunity, name="opportunity")
        if denominator <= 0.0:
            raise P1MBBError(
                "normalized regret denominator sum(opportunity) must be positive"
            )
        numerator = _safe_sum(regret, name="regret")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = float(numerator / denominator)
        except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
            raise P1MBBError("normalized regret is non-finite") from exc
    elif metric == "s3_skill_did":
        injected_model = _valid_values(selected["injected_model_se"], selected_mask, name="injected_model_se")
        injected_zero = _valid_values(selected["injected_zero_se"], selected_mask, name="injected_zero_se")
        control_model = _valid_values(selected["control_model_se"], selected_mask, name="control_model_se")
        control_zero = _valid_values(selected["control_zero_se"], selected_mask, name="control_zero_se")
        for values, name in (
            (injected_model, "injected_model_se"),
            (injected_zero, "injected_zero_se"),
            (control_model, "control_model_se"),
            (control_zero, "control_zero_se"),
        ):
            _require_nonnegative(values, name=name)
        injected_denominator = _safe_sum(injected_zero, name="injected_zero_se")
        control_denominator = _safe_sum(control_zero, name="control_zero_se")
        if injected_denominator <= 0.0 or control_denominator <= 0.0:
            raise P1MBBError("S3 skill DID requires positive injected/control zero-error denominators")
        injected_numerator = _safe_sum(injected_model, name="injected_model_se")
        control_numerator = _safe_sum(control_model, name="control_model_se")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                injected_skill = 1.0 - injected_numerator / injected_denominator
                control_skill = 1.0 - control_numerator / control_denominator
                value = float(injected_skill - control_skill)
        except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
            raise P1MBBError("S3 skill DID is non-finite") from exc
    elif metric == "s3_utility_did":
        injected_delta = selected["injected_candidate_utility"][selected_mask] - selected["injected_benchmark_hold_utility"][selected_mask]
        control_delta = selected["control_candidate_utility"][selected_mask] - selected["control_benchmark_hold_utility"][selected_mask]
        if not np.isfinite(injected_delta).all() or not np.isfinite(control_delta).all():
            raise P1MBBError("S3 utility DID is non-finite")
        value = _safe_mean(injected_delta, name="injected utility delta") - _safe_mean(control_delta, name="control utility delta")
    else:  # pragma: no cover - guarded by _validate_recompute_metric
        raise P1MBBError(f"unsupported P1 recomputation metric: {metric}")
    if not np.isfinite(value):
        raise P1MBBError(f"{metric} recomputation is non-finite")
    return float(value)


def recompute_mse_delta(
    candidate_se: Any,
    baseline_se: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute mean squared-error delta on a full-grid (possibly sampled) mask."""
    arrays, common_mask, _ = _prepare_single_metric(
        {"candidate_se": candidate_se, "baseline_se": baseline_se},
        mask,
        metric="mse_delta",
    )
    return _metric_value("mse_delta", arrays, common_mask, indices=indices)


def recompute_skill(
    model_se: Any,
    zero_se: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute ``1-sum(SE_model)/sum(SE_zero)`` with a strict denominator."""
    arrays, common_mask, _ = _prepare_single_metric(
        {"model_se": model_se, "zero_se": zero_se},
        mask,
        metric="skill",
    )
    return _metric_value("skill", arrays, common_mask, indices=indices)


def recompute_logloss_mean(
    values: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute the mean log loss for one arm."""
    arrays, common_mask, _ = _prepare_recompute_arrays(
        {"values": values}, mask, expected_keys=frozenset({"values"})
    )
    selected, selected_mask = _metric_views(arrays, common_mask, indices=indices)
    valid = _valid_values(selected["values"], selected_mask, name="values")
    _require_nonnegative_domain(valid, name="values")
    return _safe_mean(valid, name="values")


def recompute_logloss_delta(
    candidate_logloss: Any,
    baseline_logloss: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute the paired candidate-minus-baseline log-loss contrast."""
    arrays, common_mask, _ = _prepare_single_metric(
        {"candidate_logloss": candidate_logloss, "baseline_logloss": baseline_logloss},
        mask,
        metric="logloss",
    )
    return _metric_value("logloss", arrays, common_mask, indices=indices)


def recompute_agreement_mean(
    values: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute feasible-action agreement for one arm."""
    arrays, common_mask, _ = _prepare_recompute_arrays(
        {"values": values}, mask, expected_keys=frozenset({"values"})
    )
    selected, selected_mask = _metric_views(arrays, common_mask, indices=indices)
    valid = _valid_values(selected["values"], selected_mask, name="values")
    _require_binary_agreement(valid, name="values")
    return _safe_mean(valid, name="values")


def recompute_agreement_delta(
    candidate_agreement: Any,
    baseline_agreement: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute the paired candidate-minus-baseline agreement contrast."""
    arrays, common_mask, _ = _prepare_single_metric(
        {"candidate_agreement": candidate_agreement, "baseline_agreement": baseline_agreement},
        mask,
        metric="agreement",
    )
    return _metric_value("agreement", arrays, common_mask, indices=indices)


def recompute_policy_utility_delta(
    candidate_utility: Any,
    benchmark_hold_utility: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute utility against the independent benchmark hold path."""
    arrays, common_mask, _ = _prepare_single_metric(
        {
            "candidate_utility": candidate_utility,
            "benchmark_hold_utility": benchmark_hold_utility,
        },
        mask,
        metric="policy_utility_delta",
    )
    return _metric_value("policy_utility_delta", arrays, common_mask, indices=indices)


def recompute_s2_level_contrast(
    level_a_values: Any,
    level_b_values: Any,
    mask: Any,
    *,
    level_direction: str,
    indices: Any = None,
) -> float:
    """Recompute a registry-directed adjacent S2 level contrast.

    ``level_a_values`` and ``level_b_values`` are supplied in registry order:
    high/medium for ``high_*_medium`` or medium/low for ``medium_*_low``.
    The result is always the raw first-level minus second-level contrast; the
    fixed direction controls its one-sided interpretation.
    """
    direction = _validate_s2_direction(level_direction)
    arrays, common_mask, _ = _prepare_single_metric(
        {"level_a_values": level_a_values, "level_b_values": level_b_values},
        mask,
        metric="s2_contrast",
    )
    return _metric_value(
        "s2_contrast",
        arrays,
        common_mask,
        indices=indices,
        level_direction=direction,
    )


def recompute_s2_skill_contrast(
    level_a_model_se: Any,
    level_a_zero_se: Any,
    level_b_model_se: Any,
    level_b_zero_se: Any,
    mask: Any,
    *,
    level_direction: str,
    indices: Any = None,
) -> float:
    """Recompute a directed adjacent contrast of normalized MSE skills."""
    direction = _validate_s2_direction(level_direction)
    arrays, common_mask, _ = _prepare_recompute_arrays(
        {
            "level_a_model_se": level_a_model_se,
            "level_a_zero_se": level_a_zero_se,
            "level_b_model_se": level_b_model_se,
            "level_b_zero_se": level_b_zero_se,
        },
        mask,
        expected_keys=_P1_S2_LEVEL_ARRAY_KEYS["skill"],
    )
    return _metric_value(
        "s2_contrast",
        arrays,
        common_mask,
        indices=indices,
        level_direction=direction,
        level_metric="skill",
    )


def recompute_s2_normalized_regret_contrast(
    level_a_regret: Any,
    level_a_opportunity: Any,
    level_b_regret: Any,
    level_b_opportunity: Any,
    mask: Any,
    *,
    level_direction: str,
    indices: Any = None,
) -> float:
    """Recompute adjacent S2 normalized-regret ratios before contrasting them."""
    direction = _validate_s2_direction(level_direction)
    arrays, common_mask, _ = _prepare_recompute_arrays(
        {
            "level_a_regret": level_a_regret,
            "level_a_opportunity": level_a_opportunity,
            "level_b_regret": level_b_regret,
            "level_b_opportunity": level_b_opportunity,
        },
        mask,
        expected_keys=_P1_S2_LEVEL_ARRAY_KEYS["normalized_regret"],
    )
    return _metric_value(
        "s2_contrast",
        arrays,
        common_mask,
        indices=indices,
        level_direction=direction,
        level_metric="normalized_regret",
    )


# The shorter name mirrors the preregistered registry label; it remains the
# same strict adjacent-level operation rather than a new generic reducer.
recompute_s2_contrast = recompute_s2_level_contrast


def recompute_normalized_regret(
    regret: Any,
    opportunity: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute ``sum(regret)/sum(opportunity)`` with a positive denominator."""
    arrays, common_mask, _ = _prepare_single_metric(
        {"regret": regret, "opportunity": opportunity},
        mask,
        metric="normalized_regret",
    )
    return _metric_value("normalized_regret", arrays, common_mask, indices=indices)


def recompute_s3_skill_did(
    injected_model_se: Any,
    injected_zero_se: Any,
    control_model_se: Any,
    control_zero_se: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute injected-minus-control MSE skill difference-in-differences."""
    arrays, common_mask, _ = _prepare_single_metric(
        {
            "injected_model_se": injected_model_se,
            "injected_zero_se": injected_zero_se,
            "control_model_se": control_model_se,
            "control_zero_se": control_zero_se,
        },
        mask,
        metric="s3_skill_did",
    )
    return _metric_value("s3_skill_did", arrays, common_mask, indices=indices)


def recompute_s3_utility_did(
    injected_candidate_utility: Any,
    injected_benchmark_hold_utility: Any,
    control_candidate_utility: Any,
    control_benchmark_hold_utility: Any,
    mask: Any,
    *,
    indices: Any = None,
) -> float:
    """Recompute injected-minus-control benchmark-relative utility DID."""
    arrays, common_mask, _ = _prepare_single_metric(
        {
            "injected_candidate_utility": injected_candidate_utility,
            "injected_benchmark_hold_utility": injected_benchmark_hold_utility,
            "control_candidate_utility": control_candidate_utility,
            "control_benchmark_hold_utility": control_benchmark_hold_utility,
        },
        mask,
        metric="s3_utility_did",
    )
    return _metric_value("s3_utility_did", arrays, common_mask, indices=indices)


def _metric_result(
    metric: str,
    artifact: P1MBBIndexArtifact,
    *,
    point_estimate: float,
    samples: np.ndarray,
    direction: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not np.isfinite(point_estimate) or not np.isfinite(samples).all():
        raise P1MBBError(f"{metric} result is non-finite")
    try:
        lower = float(np.quantile(samples, 0.025, method="linear"))
        upper = float(np.quantile(samples, 0.975, method="linear"))
    except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise P1MBBError(f"{metric} percentile interval cannot be computed") from exc
    result: dict[str, Any] = {
        "status": "ok",
        "metric": metric,
        "direction": direction,
        "unit": artifact.unit,
        "support_id": artifact.support_id,
        "seed_ordinal": artifact.seed_ordinal,
        "block_length": artifact.block_length,
        "replicates": artifact.replicates,
        "point_estimate": float(point_estimate),
        "favorable_point_estimate": float(
            point_estimate if direction == "positive" else -point_estimate
        ),
        "ci": {
            "lower": lower,
            "upper": upper,
            "method": "np.quantile(values, q, method='linear')",
            "confidence_level": 0.95,
        },
        "p_value": _p_value(samples, direction=direction),
        "p_value_formula": (
            "(1 + count(samples <= 0))/(B+1)"
            if direction == "positive"
            else "(1 + count(samples >= 0))/(B+1)"
        ),
        "index_artifact_sha256": artifact.artifact_sha256,
        "bootstrap_values": np.array(samples, dtype="<f8", copy=True),
    }
    result["bootstrap_values"].setflags(write=False)
    if extra:
        result.update(dict(extra))
    return result


def _production_result_status_fields() -> dict[str, bool]:
    """Return the fixed pre-execution validation-result state markers."""
    return dict(_P1_PRODUCTION_RESULT_STATUS)


def _result_values_digest(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _result_json_value(value: Any, *, field: str) -> Any:
    """Convert result metadata to a typed, finite JSON representation."""
    if isinstance(value, P1MBBIndexArtifact):
        return value.artifact_sha256
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise P1MBBError(f"result metadata {field} is non-finite")
        return result
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _result_json_value(value.item(), field=field)
        raise P1MBBError(
            f"result metadata {field} contains an untyped array; persist typed values separately"
        )
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if raw_key in {"bootstrap_values", "result_sha256"}:
                continue
            if not isinstance(raw_key, (str, int, np.integer)):
                raise P1MBBError(f"result metadata {field} has a non-scalar key")
            key = str(int(raw_key)) if isinstance(raw_key, (int, np.integer)) else raw_key
            if key == "index_artifacts":
                if not isinstance(raw_value, Mapping):
                    raise P1MBBError(f"result metadata {field}.index_artifacts must be a mapping")
                converted[key] = {
                    str(ordinal): _result_json_value(
                        artifact,
                        field=f"{field}.{key}.{ordinal}",
                    )
                    for ordinal, artifact in raw_value.items()
                }
                continue
            converted[key] = _result_json_value(raw_value, field=f"{field}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _result_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise P1MBBError(f"result metadata {field} has unsupported type {type(value).__name__}")


def _result_metadata_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise P1MBBError("P1 result must be a mapping")
    if result.get("status") != "ok":
        raise P1MBBError("only an ok P1 result can be persisted")
    metadata: dict[str, Any] = {}
    for raw_key, raw_value in result.items():
        key = str(raw_key)
        if key in {"bootstrap_values", "result_sha256"}:
            continue
        if key == "index_artifacts":
            if not isinstance(raw_value, Mapping):
                raise P1MBBError("index_artifacts must be a mapping")
            metadata[key] = {
                str(ordinal): _result_json_value(artifact, field=f"{key}.{ordinal}")
                for ordinal, artifact in raw_value.items()
            }
            continue
        # Nested per-seed/per-block results retain scalar statistics and hashes;
        # their replicate arrays remain in the top-level typed field only.
        metadata[key] = _result_json_value(raw_value, field=key)
    return metadata


def _result_digest(metadata: Mapping[str, Any], values: np.ndarray) -> str:
    return hashlib.sha256(
        _metadata_bytes(metadata) + b"\0" + values.tobytes(order="C")
    ).hexdigest()


def _expected_provenance_kind(metric: str, level_metric: str | None) -> str:
    if metric == "s2_contrast":
        return (
            "action"
            if level_metric in {"agreement", "policy_utility_delta", "normalized_regret"}
            else "forecast"
        )
    return "action" if metric in _P1_ACTION_PROVENANCE_METRICS else "forecast"


def _validate_result_index_binding_group(
    actual_digests: Any,
    expected_digests: Any,
    bindings: Any,
    *,
    expected_keys: set[str],
    name: str,
) -> None:
    """Verify persisted index hashes still carry an independent binding.

    A result that stores only the artifact's self-reported hash is not enough
    for promotion: the expected digest must be present and equal, and every
    binding must retain the starts digest that was authenticated before the
    bootstrap ran.
    """
    if not isinstance(actual_digests, Mapping) or set(actual_digests) != expected_keys:
        raise P1MBBError(
            f"production result {name} artifact digests must cover {sorted(expected_keys)}"
        )
    if not isinstance(expected_digests, Mapping) or set(expected_digests) != expected_keys:
        raise P1MBBError(
            f"production result {name} expected artifact digests must cover {sorted(expected_keys)}"
        )
    if not isinstance(bindings, Mapping) or set(bindings) != expected_keys:
        raise P1MBBError(
            f"production result {name} bindings must cover {sorted(expected_keys)}"
        )
    for key in sorted(expected_keys):
        actual = _strict_sha256(
            actual_digests[key],
            name=f"result {name} artifact_sha256[{key}]",
        )
        expected = _strict_sha256(
            expected_digests[key],
            name=f"result {name} expected_artifact_sha256[{key}]",
        )
        if actual != expected:
            raise P1MBBError(
                f"production result {name} artifact digest is not externally bound at {key}"
            )
        binding = bindings[key]
        if not isinstance(binding, Mapping):
            raise P1MBBError(f"production result {name} binding {key} is not an object")
        if binding.get("artifact_sha256") != actual:
            raise P1MBBError(f"production result {name} binding artifact mismatch at {key}")
        if binding.get("expected_artifact_sha256") != expected:
            raise P1MBBError(f"production result {name} binding expected digest mismatch at {key}")
        _strict_sha256(
            binding.get("starts_sha256"),
            name=f"result {name} starts_sha256[{key}]",
        )
        if "source_path" in binding:
            _strict_text(binding["source_path"], name=f"result {name} source_path[{key}]")


def _validate_result_index_bindings(metadata: Mapping[str, Any]) -> None:
    """Require external index bindings for persisted aggregate/sensitivity results."""
    if "index_artifact_expected_sha256_by_seed" in metadata:
        _validate_result_index_binding_group(
            metadata.get("index_artifact_sha256_by_seed"),
            metadata.get("index_artifact_expected_sha256_by_seed"),
            metadata.get("index_artifact_bindings"),
            expected_keys={str(index) for index in range(10)},
            name="by_seed",
        )
    if "index_artifact_expected_sha256_by_block_length" in metadata:
        actual = metadata.get("index_artifacts")
        _validate_result_index_binding_group(
            actual,
            metadata.get("index_artifact_expected_sha256_by_block_length"),
            metadata.get("index_artifact_bindings"),
            expected_keys={str(length) for length in P1_MBB_BLOCK_LENGTHS},
            name="by_block_length",
        )
    if "index_artifact_bindings" in metadata and not (
        "index_artifact_expected_sha256_by_seed" in metadata
        or "index_artifact_expected_sha256_by_block_length" in metadata
    ):
        raise P1MBBError("production result has unclassified index artifact bindings")
    nested = metadata.get("per_block_length")
    if isinstance(nested, Mapping):
        for child in nested.values():
            if isinstance(child, Mapping):
                _validate_result_index_bindings(child)


def _validate_result_provenance_metadata(metadata: Mapping[str, Any], *, production: bool) -> None:
    if not production:
        return
    for field, expected in _P1_PRODUCTION_RESULT_STATUS.items():
        if type(metadata.get(field)) is not bool or metadata.get(field) is not expected:
            raise P1MBBError(
                f"production result {field} must be exactly {expected!r}"
            )
    metric = metadata.get("metric")
    if not isinstance(metric, str) or metric not in P1_RECOMPUTE_METRICS:
        raise P1MBBError("production result metric is not registered")
    level_metric: str | None = None
    if metric == "s2_contrast":
        level_metric_value = metadata.get("level_metric")
        level_metric = _validate_s2_level_metric(level_metric_value)
        _validate_s2_direction(metadata.get("level_direction"))
    elif "level_metric" in metadata or "level_direction" in metadata:
        raise P1MBBError("non-S2 production result cannot declare a level metric/direction")
    if "provenance" not in metadata and "provenance_by_seed" not in metadata:
        raise P1MBBError(
            "production P1 result persistence requires authenticated provenance"
        )
    provenance_items: list[Mapping[str, Any]] = []
    if "provenance" in metadata:
        provenance = metadata["provenance"]
        if not isinstance(provenance, Mapping):
            raise P1MBBError("production result provenance must be a mapping")
        provenance_items.append(provenance)
    if "provenance_by_seed" in metadata:
        seed_provenance = metadata["provenance_by_seed"]
        if not isinstance(seed_provenance, Mapping) or set(seed_provenance) != set(
            str(index) for index in range(10)
        ):
            raise P1MBBError(
                "production ten-seed result requires provenance for every seed 0..9"
            )
        for ordinal in range(10):
            value = seed_provenance[str(ordinal)]
            if not isinstance(value, Mapping):
                raise P1MBBError(f"production result provenance for seed {ordinal} must be a mapping")
            provenance_items.append(value)
    for provenance in provenance_items:
        kind = provenance.get("kind")
        if kind not in {"action", "forecast"}:
            raise P1MBBError("production result provenance kind is invalid")
        if kind != _expected_provenance_kind(metric, level_metric):
            raise P1MBBError("production result provenance kind does not match its metric")
        _strict_sha256(
            provenance.get("common_mask_sha256"),
            name="result common_mask_sha256",
        )
        if provenance.get("common_mask_field") != "common_mask":
            raise P1MBBError("production result provenance must bind common_mask")
        if kind == "action":
            for field in (
                "action_primitive_payload_sha256",
                "action_primitive_schema_sha256",
                "action_primitive_content_sha256",
                "source_result_sha256",
            ):
                _strict_sha256(provenance.get(field), name=f"result {field}")
        else:
            _strict_sha256(
                provenance.get("forecast_artifact_sha256"),
                name="result forecast_artifact_sha256",
            )
            _strict_sha256(
                provenance.get("forecast_result_sha256"),
                name="result forecast_result_sha256",
            )
    _validate_result_index_bindings(metadata)


@dataclass(frozen=True)
class P1MBBResultArtifact:
    """Typed, hash-bound result payload used by the production promotion gate."""

    metadata: Mapping[str, Any]
    bootstrap_values: np.ndarray
    production: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise P1MBBError("P1 result metadata must be a mapping")
        metadata = dict(self.metadata)
        if metadata.get("schema") != P1_MBB_RESULT_SCHEMA:
            raise P1MBBError("unsupported P1 result artifact schema")
        if metadata.get("schema_version") != P1_MBB_RESULT_SCHEMA_VERSION:
            raise P1MBBError("unsupported P1 result artifact schema version")
        try:
            values = np.asarray(self.bootstrap_values)
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise P1MBBError("P1 result bootstrap_values are malformed") from exc
        expected_shape = (P1_MBB_REPLICATES,)
        if (
            values.dtype != np.dtype("<f8")
            or values.ndim != 1
            or values.shape != expected_shape
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
        ):
            raise P1MBBError(
                "P1 result bootstrap_values must be a finite C-order little-endian float64 vector of shape (2000,)"
            )
        values = np.array(values, dtype="<f8", copy=True, order="C")
        values.setflags(write=False)
        if metadata.get("bootstrap_values_dtype") != "<f8":
            raise P1MBBError("P1 result metadata bootstrap_values_dtype must be '<f8'")
        if metadata.get("bootstrap_values_shape") != [P1_MBB_REPLICATES]:
            raise P1MBBError("P1 result metadata bootstrap_values_shape must be [2000]")
        declared_values_digest = _strict_sha256(
            metadata.get("bootstrap_values_sha256"),
            name="bootstrap_values_sha256",
        )
        if declared_values_digest != _result_values_digest(values):
            raise P1MBBError("P1 result bootstrap_values hash mismatch")
        _validate_result_provenance_metadata(metadata, production=self.production)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "bootstrap_values", values)

    @property
    def result_sha256(self) -> str:
        return _result_digest(self.metadata, self.bootstrap_values)

    def to_dict(self, *, include_bootstrap_values: bool = False) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["result_sha256"] = self.result_sha256
        if include_bootstrap_values:
            payload["bootstrap_values"] = self.bootstrap_values.tolist()
        return payload

    @classmethod
    def from_result(
        cls,
        result: Mapping[str, Any],
        *,
        production: bool = False,
    ) -> "P1MBBResultArtifact":
        try:
            values = np.asarray(result["bootstrap_values"])
        except (KeyError, TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise P1MBBError("P1 result is missing typed bootstrap_values") from exc
        metadata = _result_metadata_from_result(result)
        metadata.update(
            {
                "schema": P1_MBB_RESULT_SCHEMA,
                "schema_version": P1_MBB_RESULT_SCHEMA_VERSION,
                "bootstrap_values_dtype": "<f8",
                "bootstrap_values_shape": [P1_MBB_REPLICATES],
            }
        )
        if values.dtype != np.dtype("<f8") or values.shape != (P1_MBB_REPLICATES,):
            raise P1MBBError(
                "P1 result bootstrap_values must be a little-endian float64 vector of shape (2000,)"
            )
        values = np.array(values, dtype="<f8", copy=True, order="C")
        if not np.isfinite(values).all():
            raise P1MBBError("P1 result bootstrap_values must be finite")
        metadata["bootstrap_values_sha256"] = _result_values_digest(values)
        return cls(metadata, values, production=production)

    @classmethod
    def from_result_fixture(cls, result: Mapping[str, Any]) -> "P1MBBResultArtifact":
        """Build a relaxed typed fixture artifact; it is not promotion eligible."""
        return cls.from_result(result, production=False)

    @classmethod
    def from_result_production(cls, result: Mapping[str, Any]) -> "P1MBBResultArtifact":
        """Build a production artifact only after provenance is present."""
        return cls.from_result(result, production=True)

    @classmethod
    def _from_dict(
        cls,
        payload: Mapping[str, Any],
        bootstrap_values: Any,
        *,
        expected_result_sha256: str | None,
        production: bool,
    ) -> "P1MBBResultArtifact":
        if not isinstance(payload, Mapping):
            raise P1MBBError("P1 result metadata must be a mapping")
        metadata = dict(payload)
        declared_result = metadata.pop("result_sha256", None)
        if production and expected_result_sha256 is None:
            raise P1MBBError(
                "production P1 result loading requires an external expected_result_sha256"
            )
        if production and declared_result is None:
            raise P1MBBError("production P1 result requires result_sha256")
        artifact = cls(metadata, bootstrap_values, production=production)
        actual = artifact.result_sha256
        if declared_result is not None and _strict_sha256(
            declared_result,
            name="result_sha256",
        ) != actual:
            raise P1MBBError("P1 result artifact hash mismatch")
        if production and _strict_sha256(
            expected_result_sha256,
            name="expected_result_sha256",
        ) != actual:
            raise P1MBBError(
                "P1 result artifact does not match the independent expected_result_sha256"
            )
        return artifact

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        bootstrap_values: Any,
        *,
        expected_result_sha256: str | None = None,
    ) -> "P1MBBResultArtifact":
        return cls._from_dict(
            payload,
            bootstrap_values,
            expected_result_sha256=expected_result_sha256,
            production=True,
        )

    @classmethod
    def from_dict_fixture(
        cls,
        payload: Mapping[str, Any],
        bootstrap_values: Any,
    ) -> "P1MBBResultArtifact":
        return cls._from_dict(
            payload,
            bootstrap_values,
            expected_result_sha256=None,
            production=False,
        )


def save_p1_mbb_result_artifact(
    path: str | Path,
    artifact: P1MBBResultArtifact,
) -> str:
    """Atomically persist one typed result as a bounded NPZ archive."""
    if not isinstance(artifact, P1MBBResultArtifact):
        raise P1MBBError("save requires a P1MBBResultArtifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = artifact.to_dict(include_bootstrap_values=False)
    encoded_metadata = _metadata_bytes(metadata)
    if len(encoded_metadata) > _P1_RESULT_METADATA_MAX_BYTES:
        raise P1MBBError("P1 result metadata exceeds the byte limit")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                bootstrap_values=artifact.bootstrap_values,
                metadata=np.frombuffer(encoded_metadata, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > _P1_RESULT_ARTIFACT_MAX_BYTES:
            raise P1MBBError("P1 result artifact exceeds the file-size limit")
        temporary.replace(output)
        temporary = None
    except P1MBBError:
        raise
    except (OSError, TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError(f"could not persist P1 result artifact {output}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return artifact.result_sha256


def save_p1_mbb_result(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    production: bool = False,
) -> str:
    """Typed result persistence; set production only for authenticated results."""
    artifact = P1MBBResultArtifact.from_result(result, production=production)
    return save_p1_mbb_result_artifact(path, artifact)


def save_p1_mbb_result_production(path: str | Path, result: Mapping[str, Any]) -> str:
    return save_p1_mbb_result(path, result, production=True)


def save_p1_mbb_result_fixture(path: str | Path, result: Mapping[str, Any]) -> str:
    return save_p1_mbb_result(path, result, production=False)


def _inspect_result_archive(source: Any) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) != 2 or {info.filename for info in infos} != {
                "bootstrap_values.npy",
                "metadata.npy",
            }:
                raise P1MBBError("P1 result archive has unexpected members")
            values_info = archive.getinfo("bootstrap_values.npy")
            metadata_info = archive.getinfo("metadata.npy")
            for info in (values_info, metadata_info):
                if info.is_dir():
                    raise P1MBBError("P1 result archive members must be regular files")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise P1MBBError("P1 result archive contains a non-regular member")
            if values_info.file_size > _P1_RESULT_ARTIFACT_MAX_BYTES:
                raise P1MBBError("P1 result values member exceeds the file-size limit")
            if metadata_info.file_size > _P1_RESULT_METADATA_MAX_BYTES:
                raise P1MBBError("P1 result metadata member exceeds the file-size limit")
            if values_info.compress_size > _P1_RESULT_ARTIFACT_MAX_BYTES:
                raise P1MBBError("P1 result compressed values member exceeds the file-size limit")
            if metadata_info.compress_size > _P1_RESULT_METADATA_MAX_BYTES:
                raise P1MBBError("P1 result compressed metadata member exceeds the file-size limit")
            return values_info, metadata_info
    except P1MBBError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise P1MBBError("P1 result archive is malformed") from exc


def _load_p1_mbb_result_artifact(
    path: str | Path,
    *,
    expected_result_sha256: str | None,
    production: bool,
) -> P1MBBResultArtifact:
    source = Path(path)
    handle: Any = None
    try:
        handle, source_size, source_signature = _open_regular_index_artifact(source)
        if source_size > _P1_RESULT_ARTIFACT_MAX_BYTES:
            raise P1MBBError("P1 result artifact exceeds the file-size limit")
        values_info, metadata_info = _inspect_result_archive(handle)
        _assert_index_artifact_unchanged(handle, source_signature)
        metadata_header_shape, metadata_payload_bytes = _inspect_npy_member_header(
            handle,
            "metadata.npy",
            metadata_info,
            expected_dtype="|u1",
            expected_shape=None,
            max_payload_bytes=_P1_RESULT_METADATA_MAX_BYTES,
        )
        if metadata_header_shape != (metadata_payload_bytes,):
            raise P1MBBError("P1 result metadata header is inconsistent")
        _assert_index_artifact_unchanged(handle, source_signature)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            if set(archive.files) != {"bootstrap_values", "metadata"}:
                raise P1MBBError("P1 result archive has unexpected fields")
            metadata_bytes = np.asarray(archive["metadata"])
            if (
                metadata_bytes.dtype != np.dtype("uint8")
                or metadata_bytes.ndim != 1
                or metadata_bytes.nbytes > _P1_RESULT_METADATA_MAX_BYTES
                or metadata_bytes.shape != metadata_header_shape
                or metadata_bytes.nbytes != metadata_payload_bytes
            ):
                raise P1MBBError("P1 result metadata bytes are malformed")
            metadata_bytes = np.array(metadata_bytes, dtype=np.uint8, copy=True, order="C")
        _assert_index_artifact_unchanged(handle, source_signature)
        try:
            metadata = json.loads(bytes(metadata_bytes).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise P1MBBError("P1 result metadata JSON is malformed") from exc
        if not isinstance(metadata, Mapping):
            raise P1MBBError("P1 result metadata must be an object")
        values_header_shape, values_payload_bytes = _inspect_npy_member_header(
            handle,
            "bootstrap_values.npy",
            values_info,
            expected_dtype="<f8",
            expected_shape=(P1_MBB_REPLICATES,),
            max_payload_bytes=P1_MBB_REPLICATES * np.dtype("<f8").itemsize,
        )
        if values_header_shape != (P1_MBB_REPLICATES,) or values_payload_bytes != P1_MBB_REPLICATES * 8:
            raise P1MBBError("P1 result bootstrap values payload is malformed")
        _assert_index_artifact_unchanged(handle, source_signature)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            values = np.asarray(archive["bootstrap_values"])
        _assert_index_artifact_unchanged(handle, source_signature)
        if production:
            artifact = P1MBBResultArtifact.from_dict(
                metadata,
                values,
                expected_result_sha256=expected_result_sha256,
            )
        else:
            artifact = P1MBBResultArtifact.from_dict_fixture(metadata, values)
        return artifact
    except P1MBBError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, MemoryError, EOFError, KeyError, json.JSONDecodeError, UnicodeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise P1MBBError(f"could not load P1 result artifact {source}") from exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


def load_p1_mbb_result(
    path: str | Path,
    *,
    expected_result_sha256: str | None = None,
) -> P1MBBResultArtifact:
    """Load a stored production result; promotion must use this boundary."""
    return _load_p1_mbb_result_artifact(
        path,
        expected_result_sha256=expected_result_sha256,
        production=True,
    )


def load_p1_mbb_result_fixture(path: str | Path) -> P1MBBResultArtifact:
    """Load a relaxed fixture result, explicitly outside promotion."""
    return _load_p1_mbb_result_artifact(
        path,
        expected_result_sha256=None,
        production=False,
    )


load_p1_mbb_result_production = load_p1_mbb_result


def _bootstrap_p1_metric(
    metric: Any,
    *,
    artifact: P1MBBIndexArtifact,
    mask: Any,
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
    candidate_mask: Any = None,
    baseline_mask: Any = None,
    production: bool,
    provenance: Mapping[str, Any] | None = None,
    expected_common_mask_sha256: Any = None,
    expected_common_mask_field: Any = None,
    expected_source_result_sha256: Any = None,
    expected_action_primitive_payload_sha256: Any = None,
    expected_action_primitive_schema_sha256: Any = None,
    expected_action_primitive_content_sha256: Any = None,
    expected_forecast_artifact_sha256: Any = None,
    expected_forecast_result_sha256: Any = None,
    **arrays: Any,
) -> dict[str, Any]:
    """Run one preregistered metric over one exact stored MBB artifact.

    This is the only generic-looking entrypoint in the module, but it is a
    closed registry: each metric has an exact required array set and no
    callback, arbitrary reducer, unpaired arm, or row-compression option.
    Every replicate indexes the original full grid, applies the same common
    mask to every arm, and recomputes the metric from the sampled arrays.
    """
    metric_name = _validate_recompute_metric(metric)
    if not isinstance(artifact, P1MBBIndexArtifact):
        raise P1MBBError("P1 metric bootstrap requires a P1MBBIndexArtifact")
    if metric_name == "s2_contrast":
        level_name = _validate_s2_direction(level_direction)
        level_metric_name = _validate_s2_level_metric(level_metric or "mean")
    elif level_direction is not None:
        raise P1MBBError("level_direction is only valid for s2_contrast")
        level_name = None
        level_metric_name = None
    elif level_metric is not None:
        raise P1MBBError("level_metric is only valid for s2_contrast")
        level_metric_name = None
    else:
        level_name = None
        level_metric_name = None
    direction_name = _resolve_metric_direction(
        metric_name,
        direction,
        level_direction=level_name,
    )
    expected_keys = (
        _P1_S2_LEVEL_ARRAY_KEYS[level_metric_name]
        if metric_name == "s2_contrast"
        else _P1_METRIC_ARRAY_KEYS[metric_name]
    )
    validated, common_mask, _ = _prepare_recompute_arrays(
        arrays,
        mask,
        expected_keys=expected_keys,
        n=artifact.n,
    )
    validated_provenance: dict[str, str] | None = None
    if production:
        _validate_required_arm_masks(
            common_mask,
            candidate_mask=candidate_mask,
            baseline_mask=baseline_mask,
        )
        validated_provenance = _validate_production_provenance(
            metric_name,
            common_mask,
            level_metric=level_metric_name,
            provenance=provenance,
            expected_common_mask_sha256=expected_common_mask_sha256,
            expected_common_mask_field=expected_common_mask_field,
            expected_source_result_sha256=expected_source_result_sha256,
            expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
            expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
            expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
            expected_forecast_artifact_sha256=expected_forecast_artifact_sha256,
            expected_forecast_result_sha256=expected_forecast_result_sha256,
        )
    elif metric_name in P1_PAIRED_MEAN_METRICS:
        _validate_optional_arm_masks(
            common_mask,
            candidate_mask=candidate_mask,
            baseline_mask=baseline_mask,
        )
    elif candidate_mask is not None or baseline_mask is not None:
        raise P1MBBError(
            "candidate_mask/baseline_mask are only valid for paired arm metrics"
        )
    try:
        point_estimate = _metric_value(
            metric_name,
            validated,
            common_mask,
            level_direction=level_name,
            level_metric=level_metric_name,
        )
        samples = np.empty(artifact.replicates, dtype="<f8")
    except (MemoryError, OverflowError) as exc:
        raise P1MBBError(f"{metric_name} bootstrap result cannot be allocated") from exc
    for replicate in range(artifact.replicates):
        try:
            samples[replicate] = _metric_value(
                metric_name,
                validated,
                common_mask,
                indices=artifact.indices_for(replicate),
                level_direction=level_name,
                level_metric=level_metric_name,
            )
        except P1MBBError as exc:
            raise P1MBBError(
                f"{metric_name} comparison blocked at replicate {replicate}: {exc}"
            ) from exc
    result_extra: dict[str, Any] = {}
    if level_name is not None:
        result_extra.update(
            {"level_direction": level_name, "level_metric": level_metric_name}
        )
    if production:
        result_extra.update(_production_result_status_fields())
    if validated_provenance is not None:
        result_extra["provenance"] = validated_provenance
    return _metric_result(
        metric_name,
        artifact,
        point_estimate=point_estimate,
        samples=samples,
        direction=direction_name,
        extra=result_extra or None,
    )


def bootstrap_p1_metric(
    metric: Any,
    *,
    artifact: P1MBBIndexArtifact,
    mask: Any,
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
    candidate_mask: Any = None,
    baseline_mask: Any = None,
    provenance: Mapping[str, Any] | None = None,
    expected_common_mask_sha256: Any = None,
    expected_common_mask_field: Any = None,
    expected_source_result_sha256: Any = None,
    expected_action_primitive_payload_sha256: Any = None,
    expected_action_primitive_schema_sha256: Any = None,
    expected_action_primitive_content_sha256: Any = None,
    expected_forecast_artifact_sha256: Any = None,
    expected_forecast_result_sha256: Any = None,
    **arrays: Any,
) -> dict[str, Any]:
    """Run a production P1 metric bootstrap with authenticated provenance."""
    return _bootstrap_p1_metric(
        metric,
        artifact=artifact,
        mask=mask,
        direction=direction,
        level_direction=level_direction,
        level_metric=level_metric,
        candidate_mask=candidate_mask,
        baseline_mask=baseline_mask,
        production=True,
        provenance=provenance,
        expected_common_mask_sha256=expected_common_mask_sha256,
        expected_common_mask_field=expected_common_mask_field,
        expected_source_result_sha256=expected_source_result_sha256,
        expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
        expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
        expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
        expected_forecast_artifact_sha256=expected_forecast_artifact_sha256,
        expected_forecast_result_sha256=expected_forecast_result_sha256,
        **arrays,
    )


def bootstrap_p1_metric_fixture(
    metric: Any,
    *,
    artifact: P1MBBIndexArtifact,
    mask: Any,
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
    candidate_mask: Any = None,
    baseline_mask: Any = None,
    **arrays: Any,
) -> dict[str, Any]:
    """Run the relaxed deterministic fixture API; never use for promotion."""
    return _bootstrap_p1_metric(
        metric,
        artifact=artifact,
        mask=mask,
        direction=direction,
        level_direction=level_direction,
        level_metric=level_metric,
        candidate_mask=candidate_mask,
        baseline_mask=baseline_mask,
        production=False,
        **arrays,
    )


bootstrap_p1_metric_production = bootstrap_p1_metric


def _payload_grid_length(payload: Mapping[str, Any]) -> int:
    try:
        mask = np.asarray(payload["mask"])
    except (KeyError, TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError("each synthetic seed payload requires a one-dimensional mask") from exc
    if mask.ndim != 1:
        raise P1MBBError("each synthetic seed mask must be one-dimensional")
    return int(mask.shape[0])


def _validate_external_index_artifacts(
    artifacts: Mapping[Any, Any] | None,
    expected_digests: Mapping[Any, Any] | None,
    *,
    keys: set[int],
    unit: str,
    support_id: str,
    block_length: int,
    grid_length: int,
    name: str,
    paths: Mapping[Any, Any] | None = None,
) -> tuple[dict[int, P1MBBIndexArtifact], dict[int, str], dict[int, str] | None]:
    """Authenticate a preloaded index artifact set at a production boundary.

    ``expected_digests`` is deliberately a separate input from the artifact
    objects.  It represents the digest recorded by the upstream artifact
    ledger; accepting ``artifact.artifact_sha256`` as the only source would
    make a forged starts matrix self-binding.  Paths are optional provenance
    labels, but when supplied they must cover the same exact key set.
    """
    if not isinstance(artifacts, Mapping):
        raise P1MBBError(
            f"production P1 MBB requires an external {name} mapping"
        )
    if not isinstance(expected_digests, Mapping):
        raise P1MBBError(
            f"production P1 MBB requires external expected {name} digests"
        )
    if set(artifacts) != keys:
        raise P1MBBError(
            f"production {name} artifacts must contain exactly {sorted(keys)}"
        )
    if set(expected_digests) != keys:
        raise P1MBBError(
            f"production expected {name} digests must contain exactly {sorted(keys)}"
        )
    if paths is not None:
        if not isinstance(paths, Mapping) or set(paths) != keys:
            raise P1MBBError(
                f"production {name} paths must contain exactly {sorted(keys)}"
            )
    authenticated: dict[int, P1MBBIndexArtifact] = {}
    digests: dict[int, str] = {}
    normalized_paths: dict[int, str] | None = {} if paths is not None else None
    for ordinal in sorted(keys):
        artifact = artifacts[ordinal]
        if not isinstance(artifact, P1MBBIndexArtifact):
            raise P1MBBError(
                f"production {name} artifact {ordinal} is not a P1MBBIndexArtifact"
            )
        if (
            artifact.unit != unit
            or artifact.support_id != support_id
            or artifact.seed_ordinal != ordinal
            or artifact.block_length != block_length
            or artifact.n != grid_length
        ):
            raise P1MBBError(
                f"production {name} artifact {ordinal} metadata does not match the registered run"
            )
        expected = _strict_sha256(
            expected_digests[ordinal],
            name=f"expected_{name}_sha256[{ordinal}]",
        )
        if artifact.artifact_sha256 != expected:
            raise P1MBBError(
                f"production {name} artifact {ordinal} does not match its independent digest"
            )
        authenticated[ordinal] = artifact
        digests[ordinal] = expected
        if normalized_paths is not None:
            normalized_paths[ordinal] = _strict_text(
                paths[ordinal],
                name=f"{name}_path[{ordinal}]",
            )
    return authenticated, digests, normalized_paths


def _index_binding_metadata(
    artifacts: Mapping[int, P1MBBIndexArtifact],
    expected_digests: Mapping[int, str],
    paths: Mapping[int, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Return explicit external index-digest bindings for result metadata."""
    return {
        str(ordinal): {
            "artifact_sha256": artifacts[ordinal].artifact_sha256,
            "starts_sha256": artifacts[ordinal].starts_sha256,
            "expected_artifact_sha256": expected_digests[ordinal],
            **({"source_path": paths[ordinal]} if paths is not None else {}),
        }
        for ordinal in sorted(artifacts)
    }


def _bootstrap_p1_metric_seed_aggregate(
    metric: Any,
    *,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    block_length: Any,
    seed_inputs: Mapping[Any, Mapping[str, Any]],
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
    production: bool,
    provenance_by_seed: Mapping[Any, Mapping[str, Any]] | None = None,
    index_artifacts: Mapping[Any, P1MBBIndexArtifact] | None = None,
    expected_index_artifact_sha256_by_seed: Mapping[Any, Any] | None = None,
    index_artifact_paths_by_seed: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Bootstrap ten synthetic seeds independently and equal-weight them.

    ``seed_inputs`` contains one full-grid payload per ordinal ``0..9``.  A
    payload has the exact metric array fields required by ``metric`` plus
    ``mask`` and, for paired metrics, optional ``candidate_mask`` and
    ``baseline_mask``.  Each seed gets its own fixed derived RNG/artifact;
    aggregation averages the ten per-seed bootstrap statistics at each
    replicate (1/10 each), while the reported point estimate is their
    preregistered median.
    """
    metric_name = _validate_recompute_metric(metric)
    name, _ = _normalize_unit(unit, unit_code=unit_code)
    if not name.startswith("synthetic_"):
        raise P1MBBError("synthetic seed aggregation only accepts synthetic P1 units")
    support = _strict_text(support_id, name="support_id")
    if support != P1_MBB_UNIT_SUPPORTS[name]:
        raise P1MBBError(
            f"support_id {support!r} is not the fixed support for {name}: "
            f"{P1_MBB_UNIT_SUPPORTS[name]!r}"
        )
    length = _strict_block_length(block_length)
    if not isinstance(seed_inputs, Mapping):
        raise P1MBBError("seed_inputs must be a mapping of seed ordinal to payload")
    if production:
        if not isinstance(provenance_by_seed, Mapping):
            raise P1MBBError(
                "production synthetic aggregation requires external provenance_by_seed"
            )
        if set(provenance_by_seed) != set(range(10)):
            raise P1MBBError(
                "production synthetic aggregation requires provenance for every seed 0..9"
            )
        # Production must consume immutable, externally loaded draw artifacts.
        # The fixture API below is the only place where this function may
        # construct starts internally.
        if not isinstance(index_artifacts, Mapping):
            raise P1MBBError(
                "production synthetic aggregation requires externally loaded index_artifacts"
            )
        if not isinstance(expected_index_artifact_sha256_by_seed, Mapping):
            raise P1MBBError(
                "production synthetic aggregation requires external expected index artifact digests"
            )
    try:
        raw_ordinals = list(seed_inputs.keys())
    except (TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise P1MBBError("seed_inputs keys are malformed") from exc
    if len(raw_ordinals) != 10:
        raise P1MBBError("synthetic P1 aggregation requires exactly seed ordinals 0..9")
    ordinals: list[int] = []
    for raw_ordinal in raw_ordinals:
        ordinal = _strict_int(raw_ordinal, name="seed_ordinal", minimum=0)
        if ordinal in ordinals:
            raise P1MBBError("seed_inputs contains duplicate seed ordinals")
        ordinals.append(ordinal)
    if set(ordinals) != set(range(10)):
        raise P1MBBError("synthetic P1 aggregation requires every seed ordinal 0..9")

    if metric_name == "s2_contrast":
        level_metric_name = _validate_s2_level_metric(level_metric or "mean")
        expected_arrays = _P1_S2_LEVEL_ARRAY_KEYS[level_metric_name]
    else:
        if level_metric is not None:
            raise P1MBBError("level_metric is only valid for s2_contrast")
        level_metric_name = None
        expected_arrays = _P1_METRIC_ARRAY_KEYS[metric_name]
    required_payload = expected_arrays | frozenset({"mask"})
    allowed_payload = required_payload | frozenset(
        {"candidate_mask", "baseline_mask"}
    )
    per_seed: dict[int, dict[str, Any]] = {}
    artifacts: dict[int, P1MBBIndexArtifact] = {}
    expected_index_digests: dict[int, str] | None = None
    normalized_index_paths: dict[int, str] | None = None
    grid_length: int | None = None
    for ordinal in range(10):
        try:
            payload = seed_inputs[ordinal]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise P1MBBError(f"seed_inputs is missing seed ordinal {ordinal}") from exc
        if not isinstance(payload, Mapping):
            raise P1MBBError(f"seed {ordinal} payload must be a mapping")
        try:
            actual_payload = frozenset(payload)
        except (TypeError, ValueError) as exc:
            raise P1MBBError(f"seed {ordinal} payload fields are malformed") from exc
        if not required_payload <= actual_payload or not actual_payload <= allowed_payload:
            missing = sorted(required_payload - actual_payload)
            extra = sorted(actual_payload - allowed_payload)
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("unexpected=" + ",".join(extra))
            raise P1MBBError(
                f"seed {ordinal} payload fields do not match the fixed contract "
                f"({'; '.join(details)})"
            )
        n = _payload_grid_length(payload)
        if grid_length is None:
            grid_length = n
        elif n != grid_length:
            raise P1MBBError(
                "synthetic seed aggregation requires the same full-grid length for every seed"
            )
        if production and ordinal == 0:
            (
                external_artifacts,
                expected_index_digests,
                normalized_index_paths,
            ) = _validate_external_index_artifacts(
                index_artifacts,
                expected_index_artifact_sha256_by_seed,
                keys=set(range(10)),
                unit=name,
                support_id=support,
                block_length=length,
                grid_length=n,
                name="index_artifacts_by_seed",
                paths=index_artifact_paths_by_seed,
            )
            artifacts.update(external_artifacts)
        if production:
            # The mapping was authenticated on seed 0 and covers every seed.
            artifact = artifacts[ordinal]
        else:
            artifact = build_p1_mbb_index_artifact(
                n,
                unit=name,
                support_id=support,
                seed_ordinal=ordinal,
                block_length=length,
            )
        if artifact.n != n:
            raise P1MBBError(
                f"index artifact for seed {ordinal} does not match the seed grid length"
            )
        artifacts[ordinal] = artifact
        array_payload = {field: payload[field] for field in expected_arrays}
        candidate_mask = payload.get("candidate_mask")
        baseline_mask = payload.get("baseline_mask")
        provenance_args: dict[str, Any] = {}
        if production:
            seed_provenance = provenance_by_seed[ordinal]
            if not isinstance(seed_provenance, Mapping):
                raise P1MBBError(f"seed {ordinal} production provenance must be a mapping")
            if "provenance" not in seed_provenance:
                raise P1MBBError(f"seed {ordinal} production provenance is missing provenance")
            provenance_args = {
                key: seed_provenance.get(key)
                for key in (
                    "provenance",
                    "expected_common_mask_sha256",
                    "expected_common_mask_field",
                    "expected_source_result_sha256",
                    "expected_action_primitive_payload_sha256",
                    "expected_action_primitive_schema_sha256",
                    "expected_action_primitive_content_sha256",
                    "expected_forecast_artifact_sha256",
                    "expected_forecast_result_sha256",
                )
            }
        per_seed[ordinal] = _bootstrap_p1_metric(
            metric_name,
            artifact=artifact,
            mask=payload["mask"],
            direction=direction,
            level_direction=level_direction,
            level_metric=level_metric_name,
            candidate_mask=candidate_mask,
            baseline_mask=baseline_mask,
            production=production,
            **provenance_args,
            **array_payload,
        )

    if grid_length is None:  # pragma: no cover - exact ordinal loop always runs
        raise P1MBBError("synthetic seed aggregation has no seed payloads")
    try:
        per_seed_samples = np.stack(
            [per_seed[ordinal]["bootstrap_values"] for ordinal in range(10)],
            axis=0,
        )
        with np.errstate(over="raise", invalid="raise"):
            aggregate_samples = np.mean(per_seed_samples, axis=0, dtype=np.float64)
        point_values = np.asarray(
            [per_seed[ordinal]["point_estimate"] for ordinal in range(10)],
            dtype="<f8",
        )
        point_estimate = float(np.median(point_values))
    except (MemoryError, OverflowError, ValueError, FloatingPointError) as exc:
        raise P1MBBError("synthetic seed bootstrap aggregation failed") from exc
    if not np.isfinite(aggregate_samples).all() or not np.isfinite(point_estimate):
        raise P1MBBError("synthetic seed bootstrap aggregation is non-finite")
    direction_name = per_seed[0]["direction"]
    aggregate_extra: dict[str, Any] = {}
    if metric_name == "s2_contrast":
        aggregate_extra.update(
            {
                "level_direction": per_seed[0].get("level_direction"),
                "level_metric": per_seed[0].get("level_metric"),
            }
        )
    if production:
        aggregate_extra.update(_production_result_status_fields())
        aggregate_extra["provenance_by_seed"] = {
            ordinal: dict(per_seed[ordinal].get("provenance", {}))
            for ordinal in range(10)
        }
    result = _metric_result(
        metric_name,
        artifacts[0],
        point_estimate=point_estimate,
        samples=np.asarray(aggregate_samples, dtype="<f8"),
        direction=direction_name,
        extra=aggregate_extra or None,
    )
    result.pop("index_artifact_sha256", None)
    result.update(
        {
            "seed_count": 10,
            "seed_ordinals": list(range(10)),
            "point_estimate_rule": "median of the ten per-seed metric values",
            "bootstrap_aggregation": "mean of the ten independently resampled seed statistics with equal weight 1/10 at each replicate",
            "index_artifact_sha256_by_seed": {
                ordinal: artifacts[ordinal].artifact_sha256 for ordinal in range(10)
            },
            "index_artifacts": artifacts,
            "per_seed": per_seed,
            "per_seed_point_estimates": {
                ordinal: float(per_seed[ordinal]["point_estimate"])
                for ordinal in range(10)
            },
        }
    )
    if production:
        # Persist the independent digest binding alongside the semantic result;
        # the artifact object itself is deliberately reduced to its hashes by
        # the typed JSON serializer.
        result["index_artifact_expected_sha256_by_seed"] = dict(
            expected_index_digests or {}
        )
        result["index_artifact_bindings"] = _index_binding_metadata(
            artifacts,
            expected_index_digests or {},
            normalized_index_paths,
        )
    return result


def bootstrap_p1_metric_seed_aggregate(
    metric: Any,
    *,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    block_length: Any,
    seed_inputs: Mapping[Any, Mapping[str, Any]],
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
    provenance_by_seed: Mapping[Any, Mapping[str, Any]] | None = None,
    index_artifacts: Mapping[Any, P1MBBIndexArtifact] | None = None,
    expected_index_artifact_sha256_by_seed: Mapping[Any, Any] | None = None,
    index_artifact_paths_by_seed: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Run the production ten-seed aggregate with external provenance."""
    return _bootstrap_p1_metric_seed_aggregate(
        metric,
        unit=unit,
        unit_code=unit_code,
        support_id=support_id,
        block_length=block_length,
        seed_inputs=seed_inputs,
        direction=direction,
        level_direction=level_direction,
        level_metric=level_metric,
        production=True,
        provenance_by_seed=provenance_by_seed,
        index_artifacts=index_artifacts,
        expected_index_artifact_sha256_by_seed=expected_index_artifact_sha256_by_seed,
        index_artifact_paths_by_seed=index_artifact_paths_by_seed,
    )


def bootstrap_p1_metric_seed_aggregate_fixture(
    metric: Any,
    *,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    block_length: Any,
    seed_inputs: Mapping[Any, Mapping[str, Any]],
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
) -> dict[str, Any]:
    """Run the relaxed ten-seed fixture aggregate; never promote its result."""
    return _bootstrap_p1_metric_seed_aggregate(
        metric,
        unit=unit,
        unit_code=unit_code,
        support_id=support_id,
        block_length=block_length,
        seed_inputs=seed_inputs,
        direction=direction,
        level_direction=level_direction,
        level_metric=level_metric,
        production=False,
    )


bootstrap_p1_metric_seed_aggregate_production = bootstrap_p1_metric_seed_aggregate

# Both names describe the same closed production operation; the fixture alias
# is separate so development fixtures cannot silently cross the promotion gate.
bootstrap_p1_metric_by_seed = bootstrap_p1_metric_seed_aggregate


def bootstrap_p1_metric_seed_sensitivity(
    metric: Any,
    *,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    seed_inputs: Mapping[Any, Mapping[str, Any]],
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
    provenance_by_seed: Mapping[Any, Mapping[str, Any]] | None = None,
    index_artifacts_by_block_length: Mapping[
        Any, Mapping[Any, P1MBBIndexArtifact]
    ] | None = None,
    expected_index_artifact_sha256_by_block_length: Mapping[
        Any, Mapping[Any, Any]
    ] | None = None,
    index_artifact_paths_by_block_length: Mapping[
        Any, Mapping[Any, Any]
    ] | None = None,
) -> dict[str, Any]:
    """Run the production synthetic sensitivity set with external provenance."""
    if not isinstance(index_artifacts_by_block_length, Mapping):
        raise P1MBBError(
            "production synthetic sensitivity requires externally loaded index artifacts for every block length"
        )
    if not isinstance(expected_index_artifact_sha256_by_block_length, Mapping):
        raise P1MBBError(
            "production synthetic sensitivity requires external index artifact digests for every block length"
        )
    required_lengths = set(P1_MBB_BLOCK_LENGTHS)
    if set(index_artifacts_by_block_length) != required_lengths:
        raise P1MBBError(
            "production synthetic sensitivity index artifacts must cover L=8,16,32"
        )
    if set(expected_index_artifact_sha256_by_block_length) != required_lengths:
        raise P1MBBError(
            "production synthetic sensitivity index digests must cover L=8,16,32"
        )
    if index_artifact_paths_by_block_length is not None and (
        not isinstance(index_artifact_paths_by_block_length, Mapping)
        or set(index_artifact_paths_by_block_length) != required_lengths
    ):
        raise P1MBBError(
            "production synthetic sensitivity index paths must cover L=8,16,32"
        )
    results: dict[int, dict[str, Any]] = {}
    for length in P1_MBB_BLOCK_LENGTHS:
        results[length] = bootstrap_p1_metric_seed_aggregate(
            metric,
            unit=unit,
            unit_code=unit_code,
            support_id=support_id,
            block_length=length,
            seed_inputs=seed_inputs,
            direction=direction,
            level_direction=level_direction,
            level_metric=level_metric,
            provenance_by_seed=provenance_by_seed,
            index_artifacts=index_artifacts_by_block_length[length],
            expected_index_artifact_sha256_by_seed=(
                expected_index_artifact_sha256_by_block_length[length]
            ),
            index_artifact_paths_by_seed=(
                index_artifact_paths_by_block_length[length]
                if index_artifact_paths_by_block_length is not None
                else None
            ),
        )
    result = {
        "status": "ok",
        "metric": _validate_recompute_metric(metric),
        "direction": results[P1_MBB_BLOCK_LENGTHS[0]]["direction"],
        "block_lengths": list(P1_MBB_BLOCK_LENGTHS),
        "per_block_length": results,
        "raw_p": max(float(result["p_value"]) for result in results.values()),
        "raw_p_rule": "max(p_block_length_8, p_block_length_16, p_block_length_32)",
    }
    if production:
        result.update(_production_result_status_fields())
    return result


bootstrap_p1_metric_sensitivity_by_seed = bootstrap_p1_metric_seed_sensitivity
bootstrap_p1_metric_seed_sensitivity_production = bootstrap_p1_metric_seed_sensitivity


def bootstrap_p1_metric_seed_sensitivity_fixture(
    metric: Any,
    *,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    seed_inputs: Mapping[Any, Mapping[str, Any]],
    direction: Any = None,
    level_direction: Any = None,
    level_metric: Any = None,
) -> dict[str, Any]:
    """Run synthetic sensitivity diagnostics explicitly as a non-production fixture."""
    results: dict[int, dict[str, Any]] = {}
    for length in P1_MBB_BLOCK_LENGTHS:
        results[length] = bootstrap_p1_metric_seed_aggregate_fixture(
            metric,
            unit=unit,
            unit_code=unit_code,
            support_id=support_id,
            block_length=length,
            seed_inputs=seed_inputs,
            direction=direction,
            level_direction=level_direction,
            level_metric=level_metric,
        )
    return {
        "status": "ok",
        "metric": _validate_recompute_metric(metric),
        "direction": results[P1_MBB_BLOCK_LENGTHS[0]]["direction"],
        "block_lengths": list(P1_MBB_BLOCK_LENGTHS),
        "per_block_length": results,
        "raw_p": max(float(result["p_value"]) for result in results.values()),
        "raw_p_rule": "max(p_block_length_8, p_block_length_16, p_block_length_32)",
    }


bootstrap_p1_metric_sensitivity_by_seed_fixture = bootstrap_p1_metric_seed_sensitivity_fixture


def _paired_bootstrap_mean_delta(
    candidate_values: Any,
    baseline_values: Any,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
    artifact: P1MBBIndexArtifact,
    metric: str,
    direction: str,
    production: bool,
    provenance: Mapping[str, Any] | None = None,
    expected_common_mask_sha256: Any = None,
    expected_common_mask_field: Any = None,
    expected_source_result_sha256: Any = None,
    expected_action_primitive_payload_sha256: Any = None,
    expected_action_primitive_schema_sha256: Any = None,
    expected_action_primitive_content_sha256: Any = None,
    expected_forecast_artifact_sha256: Any = None,
    expected_forecast_result_sha256: Any = None,
) -> dict[str, Any]:
    """Recompute one registered paired mean contrast over exact MBB draws.

    The inputs are per-primitive metric arrays, not policy paths.  The fixed
    common mask is applied after sampling, so duplicate sampled rows and
    masked full-grid rows retain their intended meaning.  CI is the mandated
    two-sided diagnostic percentile interval; p-value is the preregistered
    one-sided ``(1 + count)/ (B + 1)`` value.
    """
    metric_name = _validate_metric(metric)
    direction_name = _resolve_metric_direction(
        metric_name,
        direction,
        level_direction=None,
    )
    candidate, baseline, common_mask = _validate_paired_inputs(
        candidate_values,
        baseline_values,
        candidate_mask,
        baseline_mask,
        artifact,
    )
    array_names: Mapping[str, tuple[str, str]] = {
        "mse_delta": ("candidate_se", "baseline_se"),
        "logloss": ("candidate_logloss", "baseline_logloss"),
        "agreement": ("candidate_agreement", "baseline_agreement"),
        "policy_utility_delta": (
            "candidate_utility",
            "benchmark_hold_utility",
        ),
    }
    candidate_name, baseline_name = array_names[metric_name]
    result = _bootstrap_p1_metric(
        metric_name,
        artifact=artifact,
        mask=common_mask,
        direction=direction_name,
        candidate_mask=candidate_mask,
        baseline_mask=baseline_mask,
        production=production,
        provenance=provenance,
        expected_common_mask_sha256=expected_common_mask_sha256,
        expected_common_mask_field=expected_common_mask_field,
        expected_source_result_sha256=expected_source_result_sha256,
        expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
        expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
        expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
        expected_forecast_artifact_sha256=expected_forecast_artifact_sha256,
        expected_forecast_result_sha256=expected_forecast_result_sha256,
        **{candidate_name: candidate, baseline_name: baseline},
    )
    result["point_delta"] = result["point_estimate"]
    result["favorable_point_delta"] = result["favorable_point_estimate"]
    return result


def paired_bootstrap_mean_delta(
    candidate_values: Any,
    baseline_values: Any,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
    artifact: P1MBBIndexArtifact,
    metric: str,
    direction: str,
    provenance: Mapping[str, Any] | None = None,
    expected_common_mask_sha256: Any = None,
    expected_common_mask_field: Any = None,
    expected_source_result_sha256: Any = None,
    expected_action_primitive_payload_sha256: Any = None,
    expected_action_primitive_schema_sha256: Any = None,
    expected_action_primitive_content_sha256: Any = None,
    expected_forecast_artifact_sha256: Any = None,
    expected_forecast_result_sha256: Any = None,
) -> dict[str, Any]:
    """Production paired bootstrap with mandatory external provenance."""
    return _paired_bootstrap_mean_delta(
        candidate_values,
        baseline_values,
        candidate_mask=candidate_mask,
        baseline_mask=baseline_mask,
        artifact=artifact,
        metric=metric,
        direction=direction,
        production=True,
        provenance=provenance,
        expected_common_mask_sha256=expected_common_mask_sha256,
        expected_common_mask_field=expected_common_mask_field,
        expected_source_result_sha256=expected_source_result_sha256,
        expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
        expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
        expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
        expected_forecast_artifact_sha256=expected_forecast_artifact_sha256,
        expected_forecast_result_sha256=expected_forecast_result_sha256,
    )


def paired_bootstrap_mean_delta_fixture(
    candidate_values: Any,
    baseline_values: Any,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
    artifact: P1MBBIndexArtifact,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    """Relaxed fixture paired bootstrap; never use for production promotion."""
    return _paired_bootstrap_mean_delta(
        candidate_values,
        baseline_values,
        candidate_mask=candidate_mask,
        baseline_mask=baseline_mask,
        artifact=artifact,
        metric=metric,
        direction=direction,
        production=False,
    )


paired_bootstrap_mean_delta_production = paired_bootstrap_mean_delta


def paired_bootstrap_mean_delta_sensitivity(
    candidate_values: Any,
    baseline_values: Any,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    seed_ordinal: Any,
    metric: str,
    direction: str,
    provenance: Mapping[str, Any] | None = None,
    expected_common_mask_sha256: Any = None,
    expected_common_mask_field: Any = None,
    expected_source_result_sha256: Any = None,
    expected_action_primitive_payload_sha256: Any = None,
    expected_action_primitive_schema_sha256: Any = None,
    expected_action_primitive_content_sha256: Any = None,
    expected_forecast_artifact_sha256: Any = None,
    expected_forecast_result_sha256: Any = None,
    index_artifacts_by_block_length: Mapping[
        Any, P1MBBIndexArtifact
    ] | None = None,
    expected_index_artifact_sha256_by_block_length: Mapping[Any, Any] | None = None,
    index_artifact_paths_by_block_length: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate production L={8,16,32} with the same external provenance."""
    if not isinstance(index_artifacts_by_block_length, Mapping):
        raise P1MBBError(
            "production sensitivity requires externally loaded index artifacts for every block length"
        )
    if not isinstance(expected_index_artifact_sha256_by_block_length, Mapping):
        raise P1MBBError(
            "production sensitivity requires external index artifact digests for every block length"
        )
    required_lengths = set(P1_MBB_BLOCK_LENGTHS)
    if set(index_artifacts_by_block_length) != required_lengths:
        raise P1MBBError(
            "production sensitivity index artifacts must cover L=8,16,32"
        )
    if set(expected_index_artifact_sha256_by_block_length) != required_lengths:
        raise P1MBBError(
            "production sensitivity index digests must cover L=8,16,32"
        )
    if index_artifact_paths_by_block_length is not None and (
        not isinstance(index_artifact_paths_by_block_length, Mapping)
        or set(index_artifact_paths_by_block_length) != required_lengths
    ):
        raise P1MBBError(
            "production sensitivity index paths must cover L=8,16,32"
        )
    normalized_unit, _ = _normalize_unit(unit, unit_code=unit_code)
    normalized_support = _strict_text(support_id, name="support_id")
    normalized_seed = _strict_int(seed_ordinal, name="seed_ordinal", minimum=0)
    grid_length = len(np.asarray(candidate_values))
    results: dict[int, dict[str, Any]] = {}
    artifacts: dict[int, P1MBBIndexArtifact] = {}
    expected_digests: dict[int, str] = {}
    paths: dict[int, str] | None = {} if index_artifact_paths_by_block_length is not None else None
    for length in P1_MBB_BLOCK_LENGTHS:
        if index_artifact_paths_by_block_length is not None:
            current_paths = {normalized_seed: index_artifact_paths_by_block_length[length]}
        else:
            current_paths = None
        if isinstance(index_artifacts_by_block_length[length], Mapping):
            # A mapping here is never a valid single-seed artifact; reject it
            # explicitly instead of accidentally treating a seed map as an
            # artifact object.
            raise P1MBBError(
                "production paired sensitivity requires one artifact per block length"
            )
        loaded, loaded_expected, loaded_paths = _validate_external_index_artifacts(
            {normalized_seed: index_artifacts_by_block_length[length]},
            {normalized_seed: expected_index_artifact_sha256_by_block_length[length]},
            keys={normalized_seed},
            unit=normalized_unit,
            support_id=normalized_support,
            block_length=length,
            grid_length=grid_length,
            name=f"index_artifact_L{length}",
            paths=current_paths,
        )
        artifact = loaded[normalized_seed]
        artifacts[length] = artifact
        expected_digests[length] = loaded_expected[normalized_seed]
        if paths is not None and loaded_paths is not None:
            paths[length] = loaded_paths[normalized_seed]
        results[length] = _paired_bootstrap_mean_delta(
            candidate_values,
            baseline_values,
            candidate_mask=candidate_mask,
            baseline_mask=baseline_mask,
            artifact=artifact,
            metric=metric,
            direction=direction,
            production=True,
            provenance=provenance,
            expected_common_mask_sha256=expected_common_mask_sha256,
            expected_common_mask_field=expected_common_mask_field,
            expected_source_result_sha256=expected_source_result_sha256,
            expected_action_primitive_payload_sha256=expected_action_primitive_payload_sha256,
            expected_action_primitive_schema_sha256=expected_action_primitive_schema_sha256,
            expected_action_primitive_content_sha256=expected_action_primitive_content_sha256,
            expected_forecast_artifact_sha256=expected_forecast_artifact_sha256,
            expected_forecast_result_sha256=expected_forecast_result_sha256,
        )
    result = {
        "status": "ok",
        "metric": _validate_metric(metric),
        "direction": _validate_direction(direction),
        "block_lengths": list(P1_MBB_BLOCK_LENGTHS),
        "per_block_length": results,
        "raw_p": max(float(result["p_value"]) for result in results.values()),
        "index_artifacts": artifacts,
        "raw_p_rule": "max(p_block_length_8, p_block_length_16, p_block_length_32)",
        "index_artifact_expected_sha256_by_block_length": expected_digests,
        "index_artifact_bindings": {
            str(length): {
                "artifact_sha256": artifacts[length].artifact_sha256,
                "starts_sha256": artifacts[length].starts_sha256,
                "expected_artifact_sha256": expected_digests[length],
                **({"source_path": paths[length]} if paths is not None else {}),
            }
            for length in P1_MBB_BLOCK_LENGTHS
        },
    }
    result.update(_production_result_status_fields())
    return result


def paired_bootstrap_mean_delta_sensitivity_fixture(
    candidate_values: Any,
    baseline_values: Any,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
    unit: str | int | None = None,
    unit_code: Any = None,
    support_id: Any,
    seed_ordinal: Any,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    """Evaluate sensitivity diagnostics explicitly outside production."""
    results: dict[int, dict[str, Any]] = {}
    artifacts: dict[int, P1MBBIndexArtifact] = {}
    for length in P1_MBB_BLOCK_LENGTHS:
        artifact = build_p1_mbb_index_artifact(
            len(np.asarray(candidate_values)),
            unit=unit,
            unit_code=unit_code,
            support_id=support_id,
            seed_ordinal=seed_ordinal,
            block_length=length,
        )
        artifacts[length] = artifact
        results[length] = _paired_bootstrap_mean_delta(
            candidate_values,
            baseline_values,
            candidate_mask=candidate_mask,
            baseline_mask=baseline_mask,
            artifact=artifact,
            metric=metric,
            direction=direction,
            production=False,
        )
    return {
        "status": "ok",
        "metric": _validate_metric(metric),
        "direction": _validate_direction(direction),
        "block_lengths": list(P1_MBB_BLOCK_LENGTHS),
        "per_block_length": results,
        "raw_p": max(float(result["p_value"]) for result in results.values()),
        "index_artifacts": artifacts,
        "raw_p_rule": "max(p_block_length_8, p_block_length_16, p_block_length_32)",
    }


paired_bootstrap_mean_delta_sensitivity_production = paired_bootstrap_mean_delta_sensitivity


def reject_unpaired_or_generic_mbb(*_: Any, **__: Any) -> None:
    """Keep generic/unpaired resampling outside the P1 inference boundary."""
    raise P1MBBImplementationBlocked(
        "P1 MBB requires a fixed paired candidate/baseline comparison; "
        "unpaired and generic bootstrap paths are forbidden"
    )


def run_p1_mbb(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Named P1 entrypoint; no generic callback or unpaired mode is exposed."""
    if args:
        raise P1MBBImplementationBlocked(
            "P1 MBB entrypoint requires named paired inputs; positional/generic modes are forbidden"
        )
    allowed = {
        "candidate_values",
        "baseline_values",
        "candidate_mask",
        "baseline_mask",
        "artifact",
        "metric",
        "direction",
        "provenance",
        "expected_common_mask_sha256",
        "expected_common_mask_field",
        "expected_source_result_sha256",
        "expected_action_primitive_payload_sha256",
        "expected_action_primitive_schema_sha256",
        "expected_action_primitive_content_sha256",
        "expected_forecast_artifact_sha256",
        "expected_forecast_result_sha256",
    }
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise P1MBBImplementationBlocked(
            "P1 MBB does not accept generic callbacks or extra reducer fields: "
            + ", ".join(sorted(unexpected))
        )
    if not {"candidate_values", "baseline_values"} <= set(kwargs):
        raise P1MBBImplementationBlocked(
            "P1 MBB requires both candidate_values and baseline_values"
        )
    if kwargs.get("baseline_values") is None:
        raise P1MBBImplementationBlocked("P1 MBB does not support unpaired resampling")
    return paired_bootstrap_mean_delta(*args, **kwargs)


__all__ = [
    "P1_MBB_BASE_SEED",
    "P1_MBB_BLOCK_LENGTHS",
    "P1_MBB_PRIMARY_BLOCK_LENGTH",
    "P1_MBB_REPLICATES",
    "P1_MBB_SCHEMA",
    "P1_MBB_SCHEMA_VERSION",
    "P1_MBB_UNIT_CODES",
    "P1_MBB_UNIT_SUPPORTS",
    "P1_REGRET_DOMAIN_TOL",
    "P1_PAIRED_MEAN_METRICS",
    "P1_RECOMPUTE_METRICS",
    "P1MBBError",
    "P1MBBImplementationBlocked",
    "P1MBBIndexArtifact",
    "P1MBBResultArtifact",
    "build_p1_mbb_index_artifact",
    "bootstrap_p1_metric",
    "bootstrap_p1_metric_fixture",
    "bootstrap_p1_metric_production",
    "bootstrap_p1_metric_by_seed",
    "bootstrap_p1_metric_seed_aggregate",
    "bootstrap_p1_metric_seed_aggregate_fixture",
    "bootstrap_p1_metric_seed_aggregate_production",
    "bootstrap_p1_metric_seed_sensitivity",
    "bootstrap_p1_metric_seed_sensitivity_fixture",
    "bootstrap_p1_metric_seed_sensitivity_production",
    "bootstrap_p1_metric_sensitivity_by_seed",
    "bootstrap_p1_metric_sensitivity_by_seed_fixture",
    "derive_p1_seed",
    "derive_seed",
    "draw_non_circular_mbb_indices",
    "draw_non_circular_mbb_starts",
    "load_p1_mbb_index_artifact",
    "load_p1_mbb_index_artifact_fixture",
    "load_p1_mbb_result",
    "load_p1_mbb_result_fixture",
    "load_p1_mbb_result_production",
    "materialize_non_circular_mbb_indices",
    "paired_bootstrap_mean_delta",
    "paired_bootstrap_mean_delta_fixture",
    "paired_bootstrap_mean_delta_production",
    "paired_bootstrap_mean_delta_sensitivity",
    "paired_bootstrap_mean_delta_sensitivity_fixture",
    "paired_bootstrap_mean_delta_sensitivity_production",
    "recompute_agreement_delta",
    "recompute_agreement_mean",
    "recompute_logloss_delta",
    "recompute_logloss_mean",
    "recompute_mse_delta",
    "recompute_normalized_regret",
    "recompute_policy_utility_delta",
    "recompute_s2_contrast",
    "recompute_s2_level_contrast",
    "recompute_s2_normalized_regret_contrast",
    "recompute_s2_skill_contrast",
    "recompute_s3_skill_did",
    "recompute_s3_utility_did",
    "recompute_skill",
    "p1_mask_sha256",
    "reject_unpaired_or_generic_mbb",
    "run_p1_mbb",
    "save_p1_mbb_index_artifact",
    "save_p1_mbb_result",
    "save_p1_mbb_result_artifact",
    "save_p1_mbb_result_fixture",
    "save_p1_mbb_result_production",
]
