"""Independent, resumable inference jobs over registered immutable caches.

This module fits no model and never changes a frozen prediction. Classical
joint conditions are expanded by the coordinator from their exact reused
head controls, rather than running duplicate bootstrap simulations.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_significance_statistics import NATIVE_PRIMARY, infer_panel


TASKS = ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery")
MODELS = ("linear", "lightgbm", "direct", "wm")
FORMULATIONS = ("regression", "classification", "joint")
FILTERS = ("past_variance", "ewma", "garch")
SCALAR_BASELINES = {"return": "zero_return", "variance": "ewma", "downside": "train_median",
    "crash": "causal_lowdim_logistic", "dd_excess": "train_median", "advantage": "causal_lowdim_ridge",
    "recovery": "causal_lowdim_ridge"}


def make_jobs(cfg=None):
    """Return the fixed 640 physical jobs; alias expansion restores 336 cells."""
    jobs = []
    def add(group, model, formulation, task, kind, **extra):
        job = dict(group=group, model=model, formulation=formulation, task=task, kind=kind, **extra)
        suffix = "_".join(str(extra[key]) for key in ("filter", "head") if key in extra)
        job["job_id"] = "_".join((group, model, formulation, task, kind)) + ("_" + suffix if suffix else "")
        jobs.append(job)
    for group in "ABCD":
        for model in MODELS:
            for formulation in FORMULATIONS:
                if model in ("linear", "lightgbm") and formulation == "joint":
                    continue
                for task in TASKS:
                    add(group, model, formulation, task, "main")
                    if formulation != "classification" and task != "crash":
                        add(group, model, formulation, task, "ic_aux", head="regression")
                for filter_name in FILTERS:
                    add(group, model, formulation, "variance", "variance_ba", filter=filter_name, head="classification")
                if formulation != "classification":
                    for filter_name in ("past_variance", "garch"):
                        add(group, model, formulation, "variance", "variance_qlike", filter=filter_name, head="regression")
                    add(group, model, formulation, "crash", "crash_brier", head="regression")
                if formulation != "regression":
                    add(group, model, formulation, "crash", "crash_brier", head="classification")
    if len(jobs) != 640 or len({job["job_id"] for job in jobs}) != len(jobs):
        raise AssertionError("registered physical inference grid must contain640 unique jobs")
    return jobs


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _claim(path):
    # Keep the inode in place: unlinking a lock file can let two processes
    # lock different inodes at the same pathname. Kernel ownership, rather
    # than diagnostic PID metadata, decides whether this job is claimed.
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("inference job is already claimed by a live process") from exc
        stream.seek(0)
        stream.truncate()
        json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                   "lock": "flock; PID is diagnostic only"}, stream)
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _array(data, key, shape):
    if key not in data or np.asarray(data[key]).shape != shape:
        raise ValueError(f"cache array {key} must have registered shape {shape}")
    return np.asarray(data[key])


def _panel_frame(baselines, panel, task, *, point=False):
    index = TASKS.index(task)
    n = len(baselines["fold"])
    data = {"fold": _array(baselines, "fold", (n,)),
        "timestamp": pd.to_datetime(_array(baselines, "index_ns", (n,)), utc=True),
        "actual": _array(baselines, "actual", (n, len(TASKS)))[:, index],
        "label": _array(baselines, "labels", (n, len(TASKS)))[:, index],
        "predicted_class": _array(panel, "classes", (n, len(TASKS)))[:, index],
        "baseline_class": _array(baselines, "classes", (n, len(TASKS)))[:, index]}
    if point:
        data.update(prediction=_array(panel, "scalar", (n, len(TASKS)))[:, index],
                    baseline_prediction=_array(baselines, "scalar", (n, len(TASKS)))[:, index])
    return pd.DataFrame(data)


def _decorate(row, task, *, head, role, family, baseline):
    result = dict(row)
    for name in ("group", "model", "formulation"):
        result[name] = task[name]
    result.update(head=head, role=role, family=family, baseline=baseline, split="test",
        estimate=row["model_estimate"], ci_low=row["model_ci_low"], ci_high=row["model_ci_high"],
        difference=(row["model_estimate"] - row["baseline_estimate"])
            if row["model_estimate"] is not None and row["baseline_estimate"] is not None else None,
        class_decision_is_scalar_proxy=task["formulation"] == "regression" and head == "classification")
    result["hypothesis_id"] = "/".join(str(result[name]) for name in
        ("group", "model", "task", "formulation", "head", "family", "baseline", "metric"))
    return result


def calculate_job_records(cfg, task, baselines, panel):
    """Pure cache-to-record operation, also usable for synthetic validation."""
    kind, target = task["kind"], task["task"]
    settings = cfg.get("statistics", {})
    main_reps = int(settings.get("bootstrap_replicates", 100000))
    sensitivity_reps = int(settings.get("sensitivity_replicates", 10000))
    ic_reps = int(settings.get("ic_replicates", 10000))
    blocks = tuple(settings.get("block_days", (1, 7, 28)))
    if 7 not in blocks:
        raise ValueError("registered primary seven-day block must be present")
    options = dict(task=target, blocks=blocks, replicates=main_reps,
        replicates_by_block={int(block): main_reps if int(block) == 7 else sensitivity_reps for block in blocks},
        random_seed=int(settings.get("seed", 20261112)) + TASKS.index(target) * 1009,
        rank_backend_path=cfg.get("rank_backend_path"),
        rank_cache_dir=str(Path(cfg["output_dir"]) / "prepared" / "rank_cache"))
    is_point = task["formulation"] != "classification"
    frame = _panel_frame(baselines, panel, target, point=is_point and kind in ("main", "ic_aux", "variance_qlike"))
    if kind == "main":
        metrics = ["balanced_accuracy"] + ([NATIVE_PRIMARY[target]] if is_point else [])
        rows = infer_panel(frame, metrics=metrics, **options)
        records = []
        for row in rows:
            if row["metric"] != "balanced_accuracy":
                records.append(_decorate(row, task, head="regression", role="primary", family="primary",
                                         baseline=SCALAR_BASELINES[target]))
                continue
            if task["formulation"] != "regression":
                records.append(_decorate(row, task, head="classification", role="primary", family="primary",
                                         baseline="causal_lowdim_logistic"))
            chance = dict(row)
            chance.update(baseline_estimate=row["chance_baseline"], baseline_ci_low=row["chance_baseline"],
                baseline_ci_high=row["chance_baseline"], improvement=row["chance_improvement"],
                improvement_ci_low=row["chance_improvement_ci_low"], improvement_ci_high=row["chance_improvement_ci_high"],
                p_value=row["chance_p_value"])
            if chance["p_value"] is not None:
                count = row["model_valid_replicates"]
                chance.update(n_valid_replicates=count, n_undefined_replicates=row["replicates"] - count,
                    p_monte_carlo_floor=1 / (count + 1),
                    bootstrap_extreme_count=int(round(chance["p_value"] * (count + 1) - 1)),
                    p_monte_carlo_standard_error=float(np.sqrt(chance["p_value"] * (1 - chance["p_value"]) / (count + 1))))
            chance["inference_available"] = chance["p_value"] is not None
            records.append(_decorate(chance, task, head="classification", role="auxiliary", family="ba_chance", baseline="fixed_chance"))
        return records
    if kind == "ic_aux":
        metrics = ["pearson_ic"] + ([] if NATIVE_PRIMARY[target] == "spearman_ic" else ["spearman_ic"])
        options.update(replicates=ic_reps, replicates_by_block={int(block): ic_reps for block in blocks})
        rows = infer_panel(frame, metrics=metrics, **options)
        records = []
        for row in rows:
            row = dict(row)
            row.update(baseline_estimate=None, baseline_ci_low=None, baseline_ci_high=None,
                improvement=None, improvement_ci_low=None, improvement_ci_high=None, p_value=None,
                p_monte_carlo_floor=None, p_monte_carlo_standard_error=None, bootstrap_extreme_count=None)
            row["inference_available"] = row["model_estimate"] is not None and row["model_ci_low"] is not None
            row["inference_unavailable_reason"] = None if row["inference_available"] else "absolute IC undefined or no varying supported segment"
            records.append(_decorate(row, task, head="regression", role="auxiliary", family="ic_ci", baseline="none"))
        return records
    if kind in ("variance_ba", "variance_qlike"):
        index = FILTERS.index(task["filter"])
        n = len(frame)
        frame["baseline_class"] = _array(baselines, "filter_classes", (n, len(FILTERS)))[:, index]
        if kind == "variance_qlike":
            frame["baseline_prediction"] = _array(baselines, "filter_variance", (n, len(FILTERS)))[:, index]
        metric = "balanced_accuracy" if kind == "variance_ba" else "qlike"
        return [_decorate(row, task, head=task["head"], role="auxiliary",
                family="variance_BA" if kind == "variance_ba" else "variance_QLIKE", baseline=task["filter"])
            for row in infer_panel(frame, metrics=[metric], **options)]
    if kind == "crash_brier":
        n = len(frame)
        frame["prediction"] = (_array(panel, "scalar", (n, len(TASKS)))[:, TASKS.index("crash")]
                                if task["head"] == "regression" else _array(panel, "crash_p", (n,)))
        frame["baseline_prediction"] = _array(baselines, "constant_crash_p", (n,))
        return [_decorate(row, task, head=task["head"], role="auxiliary", family="crash_Brier", baseline="train_crash_prior")
            for row in infer_panel(frame, metrics=["brier"], **options)]
    raise ValueError("unknown registered inference job kind")


def run_job(config_path, task):
    """Validate immutable provenance, claim a job, then atomically write it."""
    from . import wm_significance_benchmark as runner
    cfg = runner.load_config(config_path)
    registration = runner.verify(cfg)
    if task not in make_jobs(cfg):
        raise ValueError("task differs from the registered deterministic grid")
    root = Path(cfg["output_dir"])
    registration_sha = _sha(root / "preregistration.json")
    backend = registration.get("rank_backend", {})
    cfg["rank_backend_path"] = cfg.get("rank_backend_path") or backend.get("path")
    manifest_path = root / "prepared" / "cache_manifest.json"
    manifest_sha = _sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("registration_sha256") != registration_sha:
        raise ValueError("prediction cache differs from the registered significance study")
    if tuple(manifest.get("task_order", ())) != TASKS or tuple(manifest.get("filter_order", ())) != FILTERS:
        raise ValueError("cache target/filter order differs from the fixed protocol")
    panel_id = "_".join(task[key] for key in ("group", "model", "formulation"))
    panel_meta = manifest["panels"][panel_id]
    baseline_path, panel_path = Path(manifest["baseline_path"]), Path(panel_meta["path"])
    if _sha(baseline_path) != manifest["baseline_sha256"] or _sha(panel_path) != panel_meta["sha256"]:
        raise ValueError("registered paired cache file integrity failed")
    directory = root / "jobs"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (task["job_id"] + ".json")
    def metadata():
        return {"job_id": task["job_id"], "path": str(output), "sha256": _sha(output),
                "n_records": len(json.loads(output.read_text())["records"]), "complete": True,
                "registration_sha256": registration_sha}
    if output.exists():
        stored = json.loads(output.read_text())
        if (stored.get("complete") is not True or stored.get("task") != task
                or stored.get("registration_sha256") != registration_sha
                or stored.get("cache_manifest_sha256") != manifest_sha or not stored.get("records")):
            raise ValueError("existing inference job has incompatible completion provenance")
        return metadata()
    with _claim(directory / (task["job_id"] + ".lock")):
        with np.load(baseline_path, allow_pickle=False) as baseline_file, np.load(panel_path, allow_pickle=False) as panel_file:
            baselines = {key: baseline_file[key] for key in baseline_file.files}
            panel = {key: panel_file[key] for key in panel_file.files}
        n = int(manifest["n_rows"])
        if len(baselines["fold"]) != n or len(np.unique(baselines["decision_id"])) != n:
            raise ValueError("paired cache does not contain all unique registered decisions")
        for key in ("decision_id", "fold", "index_ns"):
            if key in panel and not np.array_equal(panel[key], baselines[key]):
                raise ValueError("model and baseline cached decision cohorts differ")
        records = calculate_job_records(cfg, task, baselines, panel)
        _atomic_json(output, {"complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": task["job_id"], "task": task, "registration_sha256": registration_sha,
            "cache_manifest_sha256": manifest_sha, "records": records})
    return metadata()
