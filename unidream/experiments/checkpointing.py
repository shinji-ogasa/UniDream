"""Checkpoint persistence and provenance helpers for reproducible runs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 2
_UNSUPPORTED = object()


def atomic_text_write(text: str, path: str | os.PathLike[str]) -> None:
    """Write a UTF-8 text file atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Write a torch payload atomically next to its final path.

    A killed process can otherwise leave a truncated ``.pt`` file that looks
    like a valid checkpoint to the next run.  ``os.replace`` is atomic when the
    temporary file and destination are on the same filesystem.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_safe_value(value: Any) -> Any:
    """Convert public actor runtime values into a portable checkpoint value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            safe_value = _checkpoint_safe_value(item)
            if safe_value is _UNSUPPORTED:
                return _UNSUPPORTED
            converted[str(key)] = safe_value
        return converted
    if isinstance(value, (list, tuple)):
        converted = [_checkpoint_safe_value(item) for item in value]
        if any(item is _UNSUPPORTED for item in converted):
            return _UNSUPPORTED
        return converted
    return _UNSUPPORTED


def snapshot_actor_inference_settings(actor: Any) -> dict[str, Any]:
    """Capture the non-parameter Actor state used by greedy inference.

    ``state_dict`` contains learned tensors, but the inventory controller also
    reads a large set of runtime attributes assigned by ``prepare_bc_setup``.
    Persist every public scalar/container attribute so adding a new inference
    knob cannot silently create a train-vs-replay mismatch.
    """
    settings: dict[str, Any] = {}
    for name, value in vars(actor).items():
        if name.startswith("_") or name == "training":
            continue
        safe_value = _checkpoint_safe_value(value)
        if safe_value is not _UNSUPPORTED:
            settings[name] = safe_value
    return settings


def apply_actor_inference_settings(actor: Any, settings: Any) -> None:
    """Restore the public Actor runtime state captured in a checkpoint."""
    if settings is None:
        return
    if not isinstance(settings, dict):
        raise RuntimeError("AC checkpoint inference_settings must be a mapping")
    for name, value in settings.items():
        if not isinstance(name, str) or name.startswith("_") or name == "training":
            raise RuntimeError(f"AC checkpoint has invalid inference setting: {name!r}")
        if not hasattr(actor, name):
            raise RuntimeError(f"AC checkpoint has unknown inference setting: {name}")
        setattr(actor, name, value)


def checkpoint_metadata_for_fold(
    manifest: dict[str, Any] | None,
    *,
    fold_idx: int,
    stage: str,
) -> dict[str, Any]:
    """Build the immutable provenance block embedded in every stage artifact."""
    manifest = manifest or {}
    data = dict(manifest.get("data") or {})
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": str(stage),
        "fold": int(fold_idx),
        "run_id": manifest.get("run_id"),
        "config_path": manifest.get("config_path"),
        "config_sha256": manifest.get("config_sha256"),
        "resolved_config": manifest.get("resolved_config"),
        "source_sha256": manifest.get("source_sha256"),
        "data_fingerprint_sha256": data.get("fingerprint_sha256"),
        "data_cache_tag": data.get("cache_tag"),
        "data_cache_contract_version": data.get("cache_contract_version"),
        "data_requested_start": data.get("requested_start"),
        "data_requested_end": data.get("requested_end"),
        "data_first_timestamp": data.get("first_timestamp"),
        "data_last_timestamp": data.get("last_timestamp"),
        "data_columns": list(data.get("columns") or []),
        "data_rows": data.get("rows"),
        "seed": manifest.get("seed"),
        "device": manifest.get("device"),
        "deterministic_algorithms": manifest.get("deterministic_algorithms"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "environment": dict(manifest.get("environment") or {}),
    }


def validate_checkpoint_metadata(
    metadata: Any,
    *,
    manifest: dict[str, Any],
    fold_idx: int,
    stage: str,
    path: Path,
    require_inference_settings: bool = False,
) -> None:
    """Reject artifacts that cannot be tied to the requested run manifest."""
    if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"run manifest uses unsupported checkpoint schema: "
            f"{manifest.get('checkpoint_schema_version')!r}"
        )
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Checkpoint {path} has no provenance metadata; it was not produced "
            "by the current reproducible training pipeline"
        )
    expected = checkpoint_metadata_for_fold(manifest, fold_idx=fold_idx, stage=stage)
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise RuntimeError(f"Checkpoint provenance mismatch for {path}: {details}")
    if require_inference_settings:
        selection = metadata.get("inference_selection")
        settings = metadata.get("inference_settings")
        if (
            not isinstance(selection, dict)
            or selection.get("adjust_rate_scale") is None
            or selection.get("advantage_level") is None
            or not isinstance(settings, dict)
        ):
            raise RuntimeError(
                f"Checkpoint {path} has no complete final inference settings"
            )
