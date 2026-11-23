"""Frozen, retrospective baseline inference for the completed336-cell study.

Registration precedes any new reading of Test prediction panels.  This is a
post-hoc analysis of previously inspected Test data, never a new Holdout.
Only the small causal baseline is newly fitted, using Train decisions only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from .wm_significance_report import GROUPS, MODELS, TASKS, FORMULATIONS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_significance_20261112.yaml"
NATIVE_PRIMARY = {"return": "zero_oos_r2", "variance": "qlike", "downside": "mae", "crash": "brier",
    "dd_excess": "mae", "advantage": "spearman_ic", "recovery": "spearman_ic"}
FAMILIES = {"primary": 448, "ba_chance": 336, "variance_BA": 144, "variance_QLIKE": 64, "crash_Brier": 64}


def now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    stats = cfg.get("statistics", {})
    if (cfg.get("groups") != list(GROUPS) or cfg.get("models") != list(MODELS)
            or cfg.get("tasks") != list(TASKS) or cfg.get("formulations") != list(FORMULATIONS)
            or cfg.get("folds") != list(range(15)) or cfg.get("seeds") != [7]
            or cfg.get("horizons") != [16] or cfg.get("test_selection_allowed") is not False
            or cfg.get("historical_seen_test") is not True or cfg.get("independently_unused_holdout") is not False
            or stats.get("primary_block_days") != 7 or stats.get("block_days") != [1, 7, 28]
            or stats.get("bootstrap_replicates") != 100000 or stats.get("sensitivity_replicates") != 10000
            or stats.get("ic_replicates") != 10000 or stats.get("primary_family_size") != 448
            or stats.get("global_family_size") != 1056):
        raise ValueError("fixed336 cells, historical Test,100000/10000 block inference and448/1056 families required")
    if (cfg.get("runtime", {}).get("workers", 0) < 1
            or cfg.get("runtime", {}).get("cpu_threads", 0) < 1
            or not 1 <= cfg.get("runtime", {}).get("status_seconds", 0) <= 60):
        raise ValueError("positive workers/threads and1-60 second status polling required")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "joint_config", "full_config", "point_table_path"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if "rank_backend_path" in cfg:
        cfg["rank_backend_path"] = str((ROOT / cfg["rank_backend_path"]).resolve())
    return cfg


def source_hashes():
    directory = Path(__file__).parent
    paths = sorted(directory.glob("wm_significance*.py")) + sorted(directory.glob("wm_significance*.cpp"))
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _old_contract(cfg, full=False):
    """Validate frozen sources/Train/Val artifacts, without Test panel reads."""
    from . import wm_joint_grid_benchmark as joint
    from . import wm_full_grid_benchmark as control
    from . import wm_seven_task_benchmark as seven
    from .wm_full_grid_report import _gate

    jcfg = joint.load_config(cfg["joint_config"])
    jreg, fcfg, pcfg = joint.verify(jcfg, full=full)
    configured = control.load_config(cfg["full_config"])
    if configured["config_path"] != fcfg["config_path"]:
        raise ValueError("configured full224 study differs from frozen joint parent")
    jroot, froot, proot = map(Path, (jcfg["output_dir"], fcfg["output_dir"], pcfg["output_dir"]))
    freg, _ = control.verify(fcfg)
    preg = seven.verify(pcfg)
    for root, count in ((jroot, 336), (froot, 224)):
        complete = json.loads((root / "completion.json").read_text())
        if not complete.get("complete") or complete.get("status") != "complete":
            raise ValueError("both frozen prerequisite studies must be complete")
        if count == 336 and complete.get("cells") != 336:
            raise ValueError("frozen joint table does not contain336 cells")
        report = json.loads((root / "final/report_completion.json").read_text())
        if not report.get("report_complete"):
            raise ValueError("frozen prerequisite report is incomplete")
    if Path(cfg["point_table_path"]) != jroot / "final/results_336.csv":
        raise ValueError("point table must be the frozen complete336 report")
    manifest = {}
    for root, reg in ((froot, freg), (jroot, jreg)):
        if full:
            # This checks all registered fits/validation hashes and the Test
            # barrier, without opening a Test prediction frame.
            _gate(root, reg, True)
        for name in ("preregistration.json", "completion.json", "validation_choices.json", "test_barrier.json",
                     "statistics_completion.json", "final/report_contract.json", "final/report_completion.json"):
            path = root / name
            manifest[str(path)] = _sha(path)
        for kind, name in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
            for task in reg[name]:
                path = root / kind / task["job"] / "fit.json"
                manifest[str(path)] = _sha(path)
    for path in (froot / "final/results_224.csv", jroot / "final/results_336.csv",
                 froot / "final/formulation_inference.csv", jroot / "final/joint_inference.csv",
                 proot / "preregistration.json", proot / "test_barrier.json", proot / "completion.json"):
        manifest[str(path)] = _sha(path)
    # The original D classical fit contains the already selected EWMA/GARCH.
    for fold in cfg["folds"]:
        for name in ("fit.json", "models.pkl", "evaluation.json"):
            path = proot / "classical" / f"fold_{fold:02d}_D" / name
            manifest[str(path)] = _sha(path)
    return {"manifest": manifest, "folds": freg["folds"],
        "parent_registration_sha256": _sha(proot / "preregistration.json"),
        "full_registration_sha256": _sha(froot / "preregistration.json"),
        "joint_registration_sha256": _sha(jroot / "preregistration.json"),
        "full_output_dir": str(froot), "joint_output_dir": str(jroot), "seven_output_dir": str(proot)}


def prepare(cfg):
    """Freeze the entire post-hoc protocol before reading any Test panel."""
    from .wm_significance_statistics import compile_rank_backend
    from .wm_significance_jobs import make_jobs

    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "preregistration.json").exists():
        raise ValueError("significance registration is immutable; verify or resume it")
    old = _old_contract(cfg, full=True)
    binary = compile_rank_backend(root / "prepared/rank_backend")
    compiler = subprocess.run(["clang++", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    tasks = make_jobs(cfg)
    if len(tasks) != 640 or len({task["job_id"] for task in tasks}) != 640:
        raise ValueError("exactly640 distinct frozen inference jobs required")
    registration = {"experiment_id": cfg["experiment_id"], "registered_utc": now(), "config": cfg,
        "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(), "old_contract": old,
        "rank_backend": {"path": str(binary), "binary_sha256": _sha(binary), "compiler": compiler,
            "source_sha256": _sha(Path(__file__).with_name("wm_significance_rank.cpp"))},
        "jobs": tasks, "grid": {"cells": 336, "distinct_empirical_conditions": 280, "reused_classical_joint_cells": 56},
        "analysis_type": "predeclared post-hoc inference protocol on historically inspected Test data",
        "test_selection_allowed": False, "historical_seen_test": True, "independently_unused_holdout": False,
        "seed_policy": "one frozen training seed7; uncertainty does not include refitting or seed variation",
        "primary_family_size": 448, "global_family_size": 1056, "family_counts": FAMILIES,
        "primary_comparators": {"classification": "fixed three-feature Train-only Logistic for all seven tasks",
            "return": "zero_return", "variance": "frozen_ewma", "downside": "Train_median",
            "dd_excess": "Train_median", "crash": "three-feature Train Logistic event probability",
            "advantage": "three-feature Train Ridge", "recovery": "clipped three-feature Train Ridge"},
        "baseline_features": ["log_past16_integrated_variance", "log_ewma_integrated_variance", "past16_log_return"],
        "labels_and_decisions": "unchanged fold-specific Train label contracts and argmax(p/Train_prior); scalar proxies use frozen label map",
        "resampling": "paired non-circular contiguous UTC-day moving blocks separately within fold and observed15min segment",
        "unsupported_segments": "retain fixed in all replicates, no shorter-block fallback; report resampleable fraction",
        "primary_block_days": 7, "primary_replicates": 100000, "sensitivity_days": [1, 28], "sensitivity_replicates": 10000,
        "IC_policy": "absolute Pearson and exactly re-ranked Spearman intervals10000rep; no zero-IC superiority hypothesis",
        "multiplicity": "Holm448 primary and Holm1056 all newly registered hypotheses; duplicate control aliases counted conservatively; unavailable P treated1",
        "old_comparisons": "keep frozen full112 and joint112 neural comparisons as their original separate historical Holm112 families",
        "confidence_intervals": "individual95% intervals, not family-wise simultaneous intervals",
        "no_new_training_of_existing_models": True, "cache_build_after_registration_only": True}
    atomic_json(root / "preregistration.json", registration)
    status(cfg, "registered")
    return registration


def verify(cfg, full=False):
    """Return registration after immutable config/code/old-selector checks."""
    root = Path(cfg["output_dir"])
    registration = json.loads((root / "preregistration.json").read_text())
    if registration["config_sha256"] != _sha(cfg["config_path"]) or registration["source_sha256"] != source_hashes():
        raise ValueError("significance registered source/config changed")
    rank = registration["rank_backend"]
    if rank["binary_sha256"] != _sha(rank["path"]):
        raise ValueError("registered exact-rank binary changed")
    for path, expected in registration["old_contract"]["manifest"].items():
        if _sha(path) != expected:
            raise ValueError("frozen prerequisite artifact changed: " + path)
    if full:
        old = _old_contract(cfg, full=True)
        if old != registration["old_contract"]:
            raise ValueError("frozen prerequisite scientific contract changed")
    return registration


@contextmanager
def campaign_lock(root):
    """OS-released flock cannot be confused by a stale or reused PID."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another significance coordinator holds the campaign lock") from exc
        atomic_json(root / "coordinator.json", {"pid": os.getpid(), "started_utc": now(), "lock": "flock"})
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _cache_manifest(cfg, registration, full=False):
    root = Path(cfg["output_dir"])
    path = root / "prepared/cache_manifest.json"
    data = json.loads(path.read_text())
    if data.get("registration_sha256") != _sha(root / "preregistration.json"):
        raise ValueError("cached predictions belong to a different registration")
    if len(data.get("panels", {})) != 48:
        raise ValueError("all48 condition panels are required, including explicit control aliases")
    if full:
        if _sha(data["baseline_path"]) != data["baseline_sha256"]:
            raise ValueError("frozen baseline cache changed")
        for panel in data["panels"].values():
            if _sha(panel["path"]) != panel["sha256"]:
                raise ValueError("frozen model prediction cache changed")
    return data


def _job_payload(cfg, task, *, required=False):
    root = Path(cfg["output_dir"])
    path = root / "jobs" / (task["job_id"] + ".json")
    if not path.exists():
        if required:
            raise ValueError("missing registered inference job: " + task["job_id"])
        return None
    payload = json.loads(path.read_text())
    if (payload.get("complete") is not True or payload.get("task") != task
            or payload.get("registration_sha256") != _sha(root / "preregistration.json")
            or payload.get("cache_manifest_sha256") != _sha(root / "prepared/cache_manifest.json")
            or not isinstance(payload.get("records"), list) or not payload["records"]):
        raise ValueError("inference job is stale, incomplete, or from different immutable data: " + task["job_id"])
    return payload


def status(cfg, stage=None):
    root = Path(cfg["output_dir"])
    registration_path = root / "preregistration.json"
    registration = json.loads(registration_path.read_text()) if registration_path.exists() else {"jobs": []}
    previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    done = sum((root / "jobs" / (task["job_id"] + ".json")).exists() for task in registration["jobs"])
    data = {"utc": now(), "stage": stage or previous.get("stage", "unregistered"), "pid": os.getpid(),
        "jobs": {"complete": done, "total": len(registration["jobs"])}, "cells": 336,
        "primary_hypotheses": 448, "global_hypotheses": 1056, "historical_test_exploratory": True}
    atomic_json(root / "status.json", data)
    (root / "progress.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="20">'
        '<title>336条件の統計検証</title><style>body{font-family:system-ui;margin:30px;line-height:1.7}</style>'
        '<h1>336条件の統計検証</h1>' + f'<p>段階: {data["stage"]} / 推論ジョブ: {done}/{len(registration["jobs"])}</p>'
        '<p>7日ブロック10万回、1日・28日の感度確認、448主比較・全1056比較のHolm補正。</p>'
        '<p>過去に確認したTestの探索的解析です。学習モデルや閾値を変更しません。</p>'
        '<p><a href="final/index.html">完成した統計レポート</a> / <a href="status.json">状態JSON</a></p></html>', encoding="utf-8")
    return data


def _worker(config_path, task):
    from threadpoolctl import threadpool_limits
    from .wm_significance_jobs import run_job
    cfg = load_config(config_path)
    threads = int(cfg["runtime"]["cpu_threads"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    with threadpool_limits(threads):
        return run_job(config_path, task)


def run_jobs(cfg, registration):
    pending = [task for task in registration["jobs"] if _job_payload(cfg, task) is None]
    if not pending:
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(int(cfg["runtime"]["workers"]), mp_context=context, max_tasks_per_child=1) as pool:
        futures = {pool.submit(_worker, cfg["config_path"], task): task for task in pending}
        while futures:
            ready, _ = wait(futures, timeout=cfg["runtime"]["status_seconds"], return_when=FIRST_COMPLETED)
            for future in ready:
                task = futures.pop(future)
                try:
                    future.result()
                    _job_payload(cfg, task, required=True)
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
            status(cfg, "block_bootstrap")


def _hypothesis_id(row):
    return "/".join(str(row[name]) for name in ("group", "model", "task", "formulation", "head", "family", "baseline", "metric"))


def expand_control_aliases(records):
    """Keep every predeclared classical joint head without pretending new fits."""
    result = [dict(row) for row in records]
    for original in records:
        if original["model"] not in ("linear", "lightgbm") or original["formulation"] == "joint":
            continue
        family, form, head = original["family"], original["formulation"], original["head"]
        copy = ((family == "primary" and form == head)
            or (family in ("ba_chance", "variance_BA") and form == "classification")
            or (family in ("variance_QLIKE", "ic_ci") and form == "regression")
            or (family == "crash_Brier" and form == head))
        if copy:
            alias = dict(original)
            alias.update(formulation="joint", control_alias=True,
                alias_source_formulation=form, shared_parameters=False,
                alias_reason="exact frozen classical independent-head control; no additional shared training")
            alias["hypothesis_id"] = _hypothesis_id(alias)
            result.append(alias)
    return result


def expected_slots():
    slots = set()
    for group in GROUPS:
        for model in MODELS:
            for task in TASKS:
                for form in FORMULATIONS:
                    heads = ("regression", "classification") if form == "joint" else (form,)
                    for head in heads:
                        metric = NATIVE_PRIMARY[task] if head == "regression" else "balanced_accuracy"
                        slots.add((group, model, task, form, head, "primary", metric, ""))
                        if task == "crash":
                            slots.add((group, model, task, form, head, "crash_Brier", "brier", ""))
                        if task == "variance" and head == "regression":
                            for baseline in ("past_variance", "garch"):
                                slots.add((group, model, task, form, head, "variance_QLIKE", "qlike", baseline))
                    slots.add((group, model, task, form, "classification", "ba_chance", "balanced_accuracy", ""))
                    if task == "variance":
                        for baseline in ("past_variance", "ewma", "garch"):
                            slots.add((group, model, task, form, "classification", "variance_BA", "balanced_accuracy", baseline))
    return slots


def adjust_families(records):
    """Require the full1056 family before any P value can be adjusted."""
    from .wm_significance_statistics import holm_fixed_family
    primary_rows = [row for row in records if int(row["block_days"]) == 7 and row["family"] in FAMILIES]
    def slot(row):
        return tuple(row[name] for name in ("group", "model", "task", "formulation", "head", "family", "metric")) + (
            row["baseline"] if row["family"] in ("variance_BA", "variance_QLIKE") else "",)
    if len(primary_rows) != 1056 or len({row["hypothesis_id"] for row in primary_rows}) != 1056:
        raise ValueError("exactly1056 distinct primary/secondary seven-day hypotheses required")
    if {slot(row) for row in primary_rows} != expected_slots():
        raise ValueError("the complete registered1056 hypothesis slots are required")
    def pvalues(rows):
        output = {}
        for row in rows:
            p = row.get("p_value")
            if p is not None and (not np.isfinite(p) or not 0 <= p <= 1):
                raise ValueError("raw P values must be None or probabilities")
            output[row["hypothesis_id"]] = 1. if p is None else float(p)
        return output
    primary = [row for row in primary_rows if row["family"] == "primary"]
    if len(primary) != 448:
        raise ValueError("the registered primary family contains448 comparisons")
    adjusted_primary = holm_fixed_family(pvalues(primary), family_size=448)
    adjusted_global = holm_fixed_family(pvalues(primary_rows), family_size=1056)
    output = []
    for old in records:
        row = dict(old)
        eligible = int(row["block_days"]) == 7 and row["family"] in FAMILIES
        row["p_holm_primary448"] = adjusted_primary.get(row["hypothesis_id"]) if eligible and row["family"] == "primary" else None
        row["p_holm_global"] = adjusted_global.get(row["hypothesis_id"]) if eligible else None
        improvement = row.get("improvement")
        row["significant_improvement_holm448"] = bool(row["p_holm_primary448"] is not None and row["p_holm_primary448"] < .05
            and improvement is not None and np.isfinite(improvement) and improvement > 0)
        row["significant_improvement_global"] = bool(row["p_holm_global"] is not None and row["p_holm_global"] < .05
            and improvement is not None and np.isfinite(improvement) and improvement > 0)
        row["split"] = "test"
        output.append(row)
    return output


def _existing_comparisons(root, registration):
    old = registration["old_contract"]
    full = pd.read_csv(Path(old["full_output_dir"]) / "final/formulation_inference.csv")
    joint = pd.read_csv(Path(old["joint_output_dir"]) / "final/joint_inference.csv")
    full = full.loc[full.block_days.eq(7)].copy()
    joint = joint.loc[joint.block_days.eq(7) & joint.model.isin(("direct", "wm"))].copy()
    if len(full) != 112 or len(joint) != 112:
        raise ValueError("existing separate historical families must contain112 comparisons each")
    full["historical_family"] = "frozen_classification_vs_regression_Holm112"
    joint["historical_family"] = "frozen_neural_joint_vs_single_head_Holm112"
    pd.concat([full, joint], ignore_index=True).to_csv(root / "final/existing_comparisons.csv", index=False)


def finalize(cfg, registration):
    from .wm_significance_report import write_significance_report
    root = Path(cfg["output_dir"])
    records, job_hashes = [], {}
    for task in registration["jobs"]:
        payload = _job_payload(cfg, task, required=True)
        records.extend(payload["records"])
        job_hashes[task["job_id"]] = _sha(root / "jobs" / (task["job_id"] + ".json"))
    records = adjust_families(expand_control_aliases(records))
    final = root / "final"
    final.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(final / "all_inference.csv", index=False)
    _existing_comparisons(root, registration)
    report = write_significance_report(root, cfg)
    statistics = {"complete": True, "utc": now(), "registration_sha256": _sha(root / "preregistration.json"),
        "jobs": len(registration["jobs"]), "job_sha256": job_hashes, "family_counts": FAMILIES,
        "n_primary_tests": 448, "n_global_tests": 1056, "n_inference_rows": len(records),
        "source_sha256": source_hashes(), "cache_manifest_sha256": _sha(root / "prepared/cache_manifest.json"),
        "all_inference_sha256": _sha(final / "all_inference.csv"), "historical_test_exploratory": True}
    atomic_json(root / "statistics_completion.json", statistics)
    files = ("final/all_inference.csv", "final/results_336_significance.csv", "final/index.html", "final/summary.json",
        "final/report_contract.json", "final/report_completion.json", "final/existing_comparisons.csv", "statistics_completion.json")
    complete = {"complete": True, "status": "complete", "utc": now(), "cells": 336,
        "n_primary_tests": 448, "n_global_tests": 1056, "registration_sha256": _sha(root / "preregistration.json"),
        "report": report, "artifact_sha256": {name: _sha(root / name) for name in files},
        "historical_seen_test": True, "independently_unused_holdout": False}
    atomic_json(root / "completion.json", complete)
    status(cfg, "complete")
    return complete


def campaign(cfg):
    root = Path(cfg["output_dir"])
    with campaign_lock(root):
        try:
            registration = verify(cfg, full=True)
            if (root / "completion.json").exists():
                complete = json.loads((root / "completion.json").read_text())
                if complete.get("complete") and complete.get("registration_sha256") == _sha(root / "preregistration.json"):
                    for path, expected in complete["artifact_sha256"].items():
                        if _sha(root / path) != expected:
                            raise ValueError("completed significance artifact changed")
                    status(cfg, "complete")
                    return complete
                raise ValueError("stale or incomplete completion marker")
            if not (root / "prepared/cache_manifest.json").exists():
                from .wm_significance_cache import build_cache
                status(cfg, "preparing_frozen_predictions")
                build_cache(cfg, registration)
            _cache_manifest(cfg, registration, full=True)
            status(cfg, "block_bootstrap")
            run_jobs(cfg, registration)
            status(cfg, "adjusting_multiplicity")
            result = finalize(cfg, registration)
            if (root / "failure.json").exists():
                (root / "failure.json").rename(root / ("resolved_failure_" + str(time.time_ns()) + ".json"))
            return result
        except Exception as exc:
            atomic_json(root / "failure.json", {"utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            status(cfg, "failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "campaign", "status"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = {"prepare": prepare, "verify": verify, "campaign": campaign, "status": status}[args.command](cfg)
    print(json.dumps(_clean(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
