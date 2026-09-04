"""Authenticated causal teacher context for the conditional WM/BC/AC path.

The legacy fold code passes ``oracle_positions`` as an untyped ndarray.  That
is intentionally insufficient for a strict conditional run: a caller could
replace the positions after validating an OOF artifact and silently train on
another (possibly hindsight) teacher.  This module creates a small sealed
context whose position arrays are bound to the validated OOF bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .chronological_oof import (
    ConditionalPathBlocked,
    require_conditional_oof_inputs,
)


_CONTEXT_SEAL = object()
_CONTEXT_REGISTRY: set[int] = set()


def _freeze_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or len(array) <= 0:
        raise ConditionalPathBlocked(f"conditional teacher {name} must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ConditionalPathBlocked(f"conditional teacher {name} must be finite")
    result = np.array(array, dtype=np.float32, copy=True)
    result.flags.writeable = False
    return result


def _array_digest(array: np.ndarray) -> str:
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _artifact_digest(oof_bundle: Mapping[str, Any]) -> str:
    candidate = oof_bundle.get("conditional_oof_artifact")
    if not isinstance(candidate, Mapping):
        candidate = oof_bundle.get("oof_artifact")
    if not isinstance(candidate, Mapping) and oof_bundle.get("schema") == "unidream.conditional_oof":
        candidate = oof_bundle
    if not isinstance(candidate, Mapping):
        raise ConditionalPathBlocked("conditional teacher OOF bundle has no artifact envelope")
    digest = candidate.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ConditionalPathBlocked("conditional teacher OOF artifact hash is missing")
    return digest


@dataclass(frozen=True, eq=False)
class ConditionalTeacherContext:
    """Immutable teacher positions bound to one strict chronological OOF artifact."""

    oof_bundle: Mapping[str, Any]
    train_positions: np.ndarray = field(repr=False)
    val_positions: np.ndarray = field(repr=False)
    test_positions: np.ndarray | None = field(default=None, repr=False)
    binding_sha256: str = ""
    _seal: object = field(default=None, repr=False, compare=False)


def build_conditional_teacher_context(
    *,
    config: Mapping[str, Any],
    oof_bundle: Mapping[str, Any],
    train_positions: Any,
    val_positions: Any,
    test_positions: Any | None = None,
) -> ConditionalTeacherContext:
    """Validate and bind causal teacher paths before any stage consumes them."""

    require_conditional_oof_inputs(
        config=config,
        oof_bundle=oof_bundle,
        caller="build_conditional_teacher_context",
    )
    train = _freeze_vector(train_positions, name="train_positions")
    val = _freeze_vector(val_positions, name="val_positions")
    test = None if test_positions is None else _freeze_vector(test_positions, name="test_positions")
    binding_payload = {
        "artifact_sha256": _artifact_digest(oof_bundle),
        "train_positions": _array_digest(train),
        "val_positions": _array_digest(val),
        "test_positions": None if test is None else _array_digest(test),
    }
    binding = hashlib.sha256(
        json.dumps(binding_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    context = ConditionalTeacherContext(
        oof_bundle=oof_bundle,
        train_positions=train,
        val_positions=val,
        test_positions=test,
        binding_sha256=binding,
        _seal=_CONTEXT_SEAL,
    )
    _CONTEXT_REGISTRY.add(id(context))
    return context


def require_authenticated_conditional_teacher_context(
    context: ConditionalTeacherContext | None,
    *,
    config: Mapping[str, Any],
    caller: str,
) -> ConditionalTeacherContext:
    """Reject arbitrary mappings/arrays at the WM, BC, and AC stage boundary."""

    if not isinstance(context, ConditionalTeacherContext):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: authenticated teacher context is required"
        )
    if context._seal is not _CONTEXT_SEAL or id(context) not in _CONTEXT_REGISTRY:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: teacher context seal is invalid"
        )
    require_conditional_oof_inputs(
        config=config,
        oof_bundle=context.oof_bundle,
        caller=caller,
    )
    payload = {
        "artifact_sha256": _artifact_digest(context.oof_bundle),
        "train_positions": _array_digest(context.train_positions),
        "val_positions": _array_digest(context.val_positions),
        "test_positions": None if context.test_positions is None else _array_digest(context.test_positions),
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if context.binding_sha256 != expected:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: teacher binding hash mismatch"
        )
    return context


__all__ = [
    "ConditionalTeacherContext",
    "build_conditional_teacher_context",
    "require_authenticated_conditional_teacher_context",
]
