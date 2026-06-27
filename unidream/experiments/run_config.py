from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


_RUN_KEYS = {
    "start",
    "end",
    "folds",
    "clean_checkpoint_dir",
    "deterministic_algorithms",
}
_REMOVED_CONFIG_PATHS = (
    ("world_model", "init_checkpoint"),
    ("bc", "init_checkpoint"),
    ("ac", "ignore_ac_checkpoint"),
)


@dataclass(frozen=True)
class TrainingRunConfig:
    start: str
    end: str
    folds: tuple[int, ...] | None
    clean_checkpoint_dir: bool
    deterministic_algorithms: bool
    checkpoint_dir: Path
    cache_dir: Path


def _required_mapping(cfg: dict, key: str) -> dict:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config requires a '{key}' mapping")
    return value


def _required_text(mapping: dict, key: str, scope: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config requires non-empty '{scope}.{key}'")
    return value.strip()


def _parse_folds(value: Any) -> tuple[int, ...] | None:
    if value == "all":
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("config 'run.folds' must be 'all' or a non-empty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("config 'run.folds' must contain only non-negative integers")
    if len(value) != len(set(value)):
        raise ValueError("config 'run.folds' must not contain duplicates")
    return tuple(sorted(value))


def load_training_run_config(cfg: dict) -> TrainingRunConfig:
    run_cfg = _required_mapping(cfg, "run")
    unknown = sorted(set(run_cfg) - _RUN_KEYS)
    missing = sorted(_RUN_KEYS - set(run_cfg))
    if unknown:
        raise ValueError(f"unknown run config keys: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing run config keys: {', '.join(missing)}")

    if "plan004_residual_bc_ac" in cfg:
        raise ValueError(
            "'plan004_residual_bc_ac' was removed from the strict training pipeline; "
            "only the current WM->BC->AC mainline is supported"
        )
    for section, key in _REMOVED_CONFIG_PATHS:
        section_cfg = cfg.get(section)
        if isinstance(section_cfg, dict) and key in section_cfg:
            raise ValueError(
                f"'{section}.{key}' was removed from strict training because warm-start/resume "
                "breaks standalone reproducibility"
            )

    start = _required_text(run_cfg, "start", "run")
    end = _required_text(run_cfg, "end", "run")
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except ValueError as exc:
        raise ValueError("run.start and run.end must be valid timestamps") from exc
    if start_ts >= end_ts:
        raise ValueError("run.start must be earlier than run.end")

    clean = run_cfg["clean_checkpoint_dir"]
    deterministic = run_cfg["deterministic_algorithms"]
    if not isinstance(clean, bool):
        raise ValueError("run.clean_checkpoint_dir must be boolean")
    if not isinstance(deterministic, bool):
        raise ValueError("run.deterministic_algorithms must be boolean")
    if not deterministic:
        raise ValueError("strict training requires run.deterministic_algorithms: true")

    logging_cfg = _required_mapping(cfg, "logging")
    checkpoint_dir = Path(_required_text(logging_cfg, "checkpoint_dir", "logging")).expanduser()
    cache_dir = Path(_required_text(logging_cfg, "cache_dir", "logging")).expanduser()
    checkpoint_resolved = checkpoint_dir.resolve()
    cache_resolved = cache_dir.resolve()
    if checkpoint_resolved == cache_resolved or checkpoint_resolved in cache_resolved.parents:
        raise ValueError("logging.cache_dir must be outside logging.checkpoint_dir")

    data_cfg = _required_mapping(cfg, "data")
    for key in ("include_funding", "include_oi", "include_mark"):
        if not isinstance(data_cfg.get(key), bool):
            raise ValueError(f"config requires explicit boolean 'data.{key}'")

    return TrainingRunConfig(
        start=start,
        end=end,
        folds=_parse_folds(run_cfg["folds"]),
        clean_checkpoint_dir=clean,
        deterministic_algorithms=deterministic,
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
    )


def configure_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_run_directory(run_cfg: TrainingRunConfig, resolved_cfg: dict) -> None:
    checkpoint_dir = run_cfg.checkpoint_dir
    if checkpoint_dir.exists():
        if run_cfg.clean_checkpoint_dir:
            shutil.rmtree(checkpoint_dir)
        elif any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"checkpoint directory is not empty: {checkpoint_dir}; "
                "set run.clean_checkpoint_dir: true for a clean deterministic run"
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "resolved_config.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=True, allow_unicode=True)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def data_fingerprint(features_df: pd.DataFrame, raw_returns: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(features_df.index.view("int64"), dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(features_df.to_numpy(dtype=np.float32)).tobytes())
    digest.update(np.ascontiguousarray(raw_returns.to_numpy(dtype=np.float32)).tobytes())
    digest.update("\n".join(str(column) for column in features_df.columns).encode("utf-8"))
    return digest.hexdigest()


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "unidream").rglob("*.py"))
    paths.extend(path for path in (root / "pyproject.toml", root / "uv.lock") if path.exists())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_run_manifest(
    *,
    run_cfg: TrainingRunConfig,
    cfg: dict,
    config_path: str,
    seed: int,
    device: str,
    active_cost_profile: str,
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    selected_folds: list[int],
) -> dict[str, Any]:
    config_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    fingerprint = data_fingerprint(features_df, raw_returns)
    source_sha256 = source_fingerprint()
    git_status = _git_value(
        "status",
        "--short",
        "--untracked-files=no",
        "--",
        ".",
        ":(exclude).DS_Store",
    ) or ""
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": hashlib.sha256(
            f"{config_sha256}:{seed}:{fingerprint}:{source_sha256}".encode()
        ).hexdigest()[:20],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "seed": int(seed),
        "device": device,
        "deterministic_algorithms": True,
        "cost_profile": active_cost_profile,
        "data": {
            "requested_start": run_cfg.start,
            "requested_end": run_cfg.end,
            "first_timestamp": str(features_df.index[0]),
            "last_timestamp": str(features_df.index[-1]),
            "rows": int(len(features_df)),
            "columns": list(features_df.columns),
            "fingerprint_sha256": fingerprint,
        },
        "folds": selected_folds,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(run_cfg.checkpoint_dir / "run_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def _semantic_hash_update(digest, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _semantic_hash_update(digest, key)
            _semantic_hash_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            _semantic_hash_update(digest, item)
        return
    digest.update(type(value).__name__.encode())
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))


def checkpoint_semantic_fingerprint(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    _semantic_hash_update(digest, payload)
    return digest.hexdigest()


def finalize_run_manifest(run_cfg: TrainingRunConfig, fold_results: dict[int, dict]) -> None:
    manifest_path = run_cfg.checkpoint_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, str]] = {}
    for fold_idx in sorted(fold_results):
        fold_dir = run_cfg.checkpoint_dir / f"fold_{fold_idx}"
        fold_artifacts: dict[str, str] = {}
        for filename in ("world_model.pt", "bc_actor.pt", "ac.pt"):
            path = fold_dir / filename
            if path.exists():
                fold_artifacts[filename] = checkpoint_semantic_fingerprint(path)
        artifacts[str(fold_idx)] = fold_artifacts
    manifest["checkpoint_semantic_sha256"] = artifacts
    manifest["completed"] = True
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
