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

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

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
_P1_INDEX_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024
_P1_INDEX_ARTIFACT_MAX_STARTS = 100_000_000


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
    values = np.asarray(starts)
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
        if replicate_count != P1_MBB_REPLICATES:
            raise P1MBBError("P1 MBB replicates are fixed at 2000")
        if self.schema != P1_MBB_SCHEMA or self.schema_version != P1_MBB_SCHEMA_VERSION:
            raise P1MBBError("unsupported P1 MBB index artifact schema")
        expected_seed = derive_p1_seed(
            unit,
            length,
            ordinal,
        )
        supplied_seed = _strict_int(self.derived_seed, name="derived_seed", minimum=0)
        if supplied_seed != expected_seed:
            raise P1MBBError("derived_seed does not match the fixed P1 formula")
        values = np.asarray(self.starts)
        expected_shape = (replicate_count, _n_blocks(n_int, length))
        if values.dtype != np.dtype("<i8") or values.shape != expected_shape:
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
        try:
            result = np.empty((self.replicates, self.n), dtype="<i8", order="C")
        except (MemoryError, OverflowError) as exc:
            raise P1MBBError("all P1 MBB indices cannot be materialized in memory") from exc
        for replicate in range(self.replicates):
            result[replicate] = self.indices_for(replicate)
        result.setflags(write=False)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "P1MBBIndexArtifact":
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
        starts = np.asarray(payload["starts"])
        if starts.dtype.kind not in "iu":
            raise P1MBBError("P1 MBB artifact starts must be integer data")
        starts = np.asarray(starts, dtype="<i8", order="C")
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
        if "starts_sha256" in payload and payload["starts_sha256"] != artifact.starts_sha256:
            raise P1MBBError("P1 MBB starts hash mismatch")
        if "artifact_sha256" in payload and payload["artifact_sha256"] != artifact.artifact_sha256:
            raise P1MBBError("P1 MBB artifact hash mismatch")
        return artifact


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


def load_p1_mbb_index_artifact(path: str | Path) -> P1MBBIndexArtifact:
    """Load and verify a lossless P1 MBB start artifact without pickle."""
    source = Path(path)
    try:
        if source.stat().st_size > _P1_INDEX_ARTIFACT_MAX_BYTES:
            raise P1MBBError("P1 MBB index artifact exceeds the file-size limit")
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"starts", "metadata"}:
                raise P1MBBError("P1 MBB index artifact has unexpected archive fields")
            starts = np.asarray(archive["starts"])
            metadata_bytes = np.asarray(archive["metadata"])
        if metadata_bytes.dtype != np.dtype("uint8") or metadata_bytes.ndim != 1:
            raise P1MBBError("P1 MBB metadata bytes are malformed")
        metadata = json.loads(bytes(metadata_bytes).decode("utf-8"))
        if not isinstance(metadata, Mapping):
            raise P1MBBError("P1 MBB metadata must be an object")
        payload = dict(metadata)
        payload["starts"] = starts
        return P1MBBIndexArtifact.from_dict(payload)
    except P1MBBError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, MemoryError, json.JSONDecodeError, UnicodeError) as exc:
        raise P1MBBError(f"could not load P1 MBB index artifact {source}") from exc


def _strict_float64_vector(value: Any, *, name: str, n: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("<f8") or array.ndim != 1 or array.shape != (n,):
        raise P1MBBError(f"{name} must be a little-endian float64 vector of shape ({n},)")
    if np.isinf(array).any():
        raise P1MBBError(f"{name} contains infinity")
    return np.array(array, dtype="<f8", copy=True, order="C")


def _strict_bool_mask(value: Any, *, name: str, n: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_) or array.ndim != 1 or array.shape != (n,):
        raise P1MBBError(f"{name} must be a strict bool vector of shape ({n},)")
    return np.array(array, dtype=np.bool_, copy=True, order="C")


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


def paired_bootstrap_mean_delta(
    candidate_values: Any,
    baseline_values: Any,
    *,
    candidate_mask: Any,
    baseline_mask: Any,
    artifact: P1MBBIndexArtifact,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    """Recompute one registered paired mean contrast over exact MBB draws.

    The inputs are per-primitive metric arrays, not policy paths.  The fixed
    common mask is applied after sampling, so duplicate sampled rows and
    masked full-grid rows retain their intended meaning.  CI is the mandated
    two-sided diagnostic percentile interval; p-value is the preregistered
    one-sided ``(1 + count)/ (B + 1)`` value.
    """
    metric_name = _validate_metric(metric)
    direction_name = _validate_direction(direction)
    candidate, baseline, common_mask = _validate_paired_inputs(
        candidate_values,
        baseline_values,
        candidate_mask,
        baseline_mask,
        artifact,
    )
    observed_values = candidate[common_mask] - baseline[common_mask]
    if not np.isfinite(observed_values).all():
        raise P1MBBError("paired P1 MBB observed contrast is non-finite")
    point_delta = float(np.mean(observed_values))
    try:
        samples = np.empty(artifact.replicates, dtype="<f8")
    except (MemoryError, OverflowError) as exc:
        raise P1MBBError("paired P1 MBB result cannot be allocated") from exc
    for replicate in range(artifact.replicates):
        indices = artifact.indices_for(replicate)
        sampled_mask = common_mask[indices]
        if not sampled_mask.any():
            raise P1MBBError(
                f"paired P1 MBB replicate {replicate} is N/A: zero valid primitive records"
            )
        values = candidate[indices][sampled_mask] - baseline[indices][sampled_mask]
        if not np.isfinite(values).all():
            raise P1MBBError(
                f"paired P1 MBB replicate {replicate} is N/A: non-finite metric"
            )
        samples[replicate] = float(np.mean(values))
    lower = float(np.quantile(samples, 0.025, method="linear"))
    upper = float(np.quantile(samples, 0.975, method="linear"))
    return {
        "status": "ok",
        "metric": metric_name,
        "direction": direction_name,
        "unit": artifact.unit,
        "support_id": artifact.support_id,
        "seed_ordinal": artifact.seed_ordinal,
        "block_length": artifact.block_length,
        "replicates": artifact.replicates,
        "point_delta": point_delta,
        "favorable_point_delta": point_delta if direction_name == "positive" else -point_delta,
        "ci": {
            "lower": lower,
            "upper": upper,
            "method": "np.quantile(values, q, method='linear')",
            "confidence_level": 0.95,
        },
        "p_value": _p_value(samples, direction=direction_name),
        "p_value_formula": (
            "(1 + count(samples <= 0))/(B+1)" if direction_name == "positive"
            else "(1 + count(samples >= 0))/(B+1)"
        ),
        "index_artifact_sha256": artifact.artifact_sha256,
        "bootstrap_values": np.array(samples, dtype="<f8", copy=True),
    }


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
) -> dict[str, Any]:
    """Evaluate the fixed L={8,16,32} diagnostic set and conservative raw p."""
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
        results[length] = paired_bootstrap_mean_delta(
            candidate_values,
            baseline_values,
            candidate_mask=candidate_mask,
            baseline_mask=baseline_mask,
            artifact=artifact,
            metric=metric,
            direction=direction,
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


def reject_unpaired_or_generic_mbb(*_: Any, **__: Any) -> None:
    """Keep generic/unpaired resampling outside the P1 inference boundary."""
    raise P1MBBImplementationBlocked(
        "P1 MBB requires a fixed paired candidate/baseline comparison; "
        "unpaired and generic bootstrap paths are forbidden"
    )


def run_p1_mbb(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Named P1 entrypoint; no generic callback or unpaired mode is exposed."""
    if len(args) < 2 and not {"candidate_values", "baseline_values"} <= set(kwargs):
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
    "P1_PAIRED_MEAN_METRICS",
    "P1MBBError",
    "P1MBBImplementationBlocked",
    "P1MBBIndexArtifact",
    "build_p1_mbb_index_artifact",
    "derive_p1_seed",
    "derive_seed",
    "draw_non_circular_mbb_indices",
    "draw_non_circular_mbb_starts",
    "load_p1_mbb_index_artifact",
    "materialize_non_circular_mbb_indices",
    "paired_bootstrap_mean_delta",
    "paired_bootstrap_mean_delta_sensitivity",
    "reject_unpaired_or_generic_mbb",
    "run_p1_mbb",
    "save_p1_mbb_index_artifact",
]
