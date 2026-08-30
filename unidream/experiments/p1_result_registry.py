"""Authenticated execution registry for the frozen P1 validation operation.

The preregistration module proves that the manifest and its two JSONL files
match their pinned digests.  This module is the narrower runtime boundary: it
re-reads those exact regular files, rejects duplicate JSON keys and blank
records, verifies their bytes against the authenticated manifest, and exposes
immutable ordered rows.  It never accepts caller-created trial/comparison
rows and it does not execute a fit, action replay, bootstrap, or outer test.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

from .p1_recovery_prereg import (
    DEFAULT_MANIFEST_PATH,
    REGISTERED_MANIFEST_SHA256,
    load_fixed_manifest,
)


class P1ResultRegistryError(ValueError):
    """Raised when the frozen validation registry cannot be authenticated."""


P1_TRIAL_COUNT = 56
P1_PRIMARY_COMPARISON_COUNT = 16
P1_REGISTRY_MAX_BYTES = 4 * 1024 * 1024
P1_TRIAL_FIELDS = (
    "trial_id",
    "scenario_id",
    "arm",
    "model_id",
    "cost_mode",
    "primary",
    "action_mapper",
    "seed_count",
)


@dataclass(frozen=True)
class P1ResultRegistry:
    """One immutable, ordered view of the pre-results P1 registries."""

    manifest_sha256: str
    trial_registry_sha256: str
    comparison_registry_sha256: str
    trials: tuple[Mapping[str, Any], ...]
    comparisons: tuple[Mapping[str, Any], ...]
    trials_by_id: Mapping[str, Mapping[str, Any]]
    comparisons_by_id: Mapping[str, Mapping[str, Any]]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise P1ResultRegistryError("registry contains an unsupported value")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P1ResultRegistryError(f"duplicate JSON key in P1 registry: {key}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise P1ResultRegistryError(f"could not stat P1 registry: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise P1ResultRegistryError("P1 registry must be a regular non-symlink file")
    if not 0 < before.st_size <= P1_REGISTRY_MAX_BYTES:
        raise P1ResultRegistryError("P1 registry file size is outside its fixed bound")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise P1ResultRegistryError("P1 registry must remain a regular file")
        signature = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise P1ResultRegistryError("P1 registry ended during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise P1ResultRegistryError("P1 registry changed during read")
        encoded = b"".join(chunks)
    except OSError as exc:
        raise P1ResultRegistryError(f"could not read P1 registry: {path}") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return encoded, hashlib.sha256(encoded).hexdigest()


def _parse_jsonl(encoded: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise P1ResultRegistryError(f"{label} registry is not valid UTF-8") from exc
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise P1ResultRegistryError(f"{label} registry contains a blank record")
    rows: list[Mapping[str, Any]] = []
    for ordinal, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    P1ResultRegistryError(
                        f"non-finite JSON constant in {label} registry: {value}"
                    )
                ),
            )
        except P1ResultRegistryError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
            raise P1ResultRegistryError(
                f"{label} registry row {ordinal} is malformed"
            ) from exc
        if not isinstance(row, Mapping) or any(not isinstance(key, str) for key in row):
            raise P1ResultRegistryError(f"{label} registry row {ordinal} is not an object")
        rows.append(_freeze(row))
    return tuple(rows)


def _registry_path(root: Path, declared: Any, *, label: str) -> Path:
    if not isinstance(declared, str) or not declared:
        raise P1ResultRegistryError(f"{label} registry path is missing")
    candidate = (root / declared).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise P1ResultRegistryError(f"{label} registry escapes the repository root") from exc
    return candidate


def load_p1_result_registry(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> P1ResultRegistry:
    """Load only the two authenticated pre-results registries in fixed order."""

    selected_manifest = Path(manifest_path)
    manifest = load_fixed_manifest(selected_manifest)
    if manifest.get("manifest_sha256") != REGISTERED_MANIFEST_SHA256:
        raise P1ResultRegistryError("P1 result registry manifest digest mismatch")
    if manifest.get("results_observed") is not False:
        raise P1ResultRegistryError("P1 registry must be loaded before results are observed")
    common = manifest.get("common")
    if not isinstance(common, Mapping):
        raise P1ResultRegistryError("P1 manifest common contract is missing")
    trial_contract = common.get("trial_registry")
    comparison_contract = common.get("primary_comparison_registry")
    if not isinstance(trial_contract, Mapping) or not isinstance(
        comparison_contract, Mapping
    ):
        raise P1ResultRegistryError("P1 manifest registry contracts are missing")
    root = selected_manifest.resolve().parents[2]
    trial_path = _registry_path(root, trial_contract.get("path"), label="trial")
    comparison_path = _registry_path(
        root,
        comparison_contract.get("path"),
        label="comparison",
    )
    trial_bytes, trial_sha256 = _read_regular_bytes(trial_path)
    comparison_bytes, comparison_sha256 = _read_regular_bytes(comparison_path)
    if trial_sha256 != trial_contract.get("sha256"):
        raise P1ResultRegistryError("trial registry SHA-256 mismatch")
    if comparison_sha256 != comparison_contract.get("sha256"):
        raise P1ResultRegistryError("comparison registry SHA-256 mismatch")
    trials = _parse_jsonl(trial_bytes, label="trial")
    comparisons = _parse_jsonl(comparison_bytes, label="comparison")
    if len(trials) != P1_TRIAL_COUNT or trial_contract.get("record_count") != len(trials):
        raise P1ResultRegistryError("trial registry does not contain exactly 56 rows")
    if (
        len(comparisons) != P1_PRIMARY_COMPARISON_COUNT
        or comparison_contract.get("family_size") != len(comparisons)
    ):
        raise P1ResultRegistryError("comparison registry does not contain exactly 16 rows")
    if any(tuple(row) != P1_TRIAL_FIELDS for row in trials):
        raise P1ResultRegistryError("trial registry fields/order are not exact")
    if any(row.get("primary") is not True for row in (*trials, *comparisons)):
        raise P1ResultRegistryError("all frozen P1 registry rows must remain primary")
    trial_ids = [row.get("trial_id") for row in trials]
    comparison_ids = [row.get("comparison_id") for row in comparisons]
    if (
        any(not isinstance(value, str) or not value for value in trial_ids)
        or len(set(trial_ids)) != len(trial_ids)
    ):
        raise P1ResultRegistryError("trial registry IDs are invalid or duplicated")
    if (
        any(not isinstance(value, str) or not value for value in comparison_ids)
        or len(set(comparison_ids)) != len(comparison_ids)
    ):
        raise P1ResultRegistryError("comparison registry IDs are invalid or duplicated")
    return P1ResultRegistry(
        manifest_sha256=REGISTERED_MANIFEST_SHA256,
        trial_registry_sha256=trial_sha256,
        comparison_registry_sha256=comparison_sha256,
        trials=trials,
        comparisons=comparisons,
        trials_by_id=MappingProxyType(dict(zip(trial_ids, trials, strict=True))),
        comparisons_by_id=MappingProxyType(
            dict(zip(comparison_ids, comparisons, strict=True))
        ),
    )


__all__ = [
    "P1_PRIMARY_COMPARISON_COUNT",
    "P1_REGISTRY_MAX_BYTES",
    "P1ResultRegistry",
    "P1ResultRegistryError",
    "P1_TRIAL_COUNT",
    "P1_TRIAL_FIELDS",
    "load_p1_result_registry",
]
