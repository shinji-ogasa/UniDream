"""Frozen 10-seed extension, causal baselines, and split/gap audit.

This independent research runner never changes the mainline or old studies.
Seven new seeds start WM, BC and initial PPO from scratch. Three existing
seeds are immutable references. Every new Val choice freezes before new Test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

from . import wm_policy_ablation_benchmark as parent
from . import wm_rl_release_benchmark as release
from .wm_policy_ablation_policy import sha, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_seed_robustness_20261112.yaml"
SEEDS = [7, 19, 42, 3, 11, 23, 31, 53, 67, 89]
CONDITIONS = ["C_floor_gradual", "A_floor_gradual", "C_control"]


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    for key in ("output_dir", "source_config", "release_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if (cfg["seeds"] != SEEDS or cfg["reused_seeds"] != SEEDS[:3] or cfg["new_seeds"] != SEEDS[3:]
            or cfg["folds"] != list(range(6)) or cfg["conditions"] != CONDITIONS
            or cfg.get("test_selection_allowed") is not False or cfg.get("historical_seen_test") is not True
            or cfg.get("independently_unused_holdout") is not False
            or cfg.get("seed_scope") != "fresh_WM_BC_initial_RL_and_continuation_for_each_new_seed"):
        raise ValueError("fixed independent-seed known-Test research contract required")
    definitions = {"C_floor_gradual": {"arm": "C", "condition": "floor_gradual"},
                   "A_floor_gradual": {"arm": "A", "condition": "floor_gradual"},
                   "C_control": {"arm": "C", "condition": "control"}}
    if cfg["condition_definitions"] != definitions:
        raise ValueError("all registered condition definitions must remain fixed")
    old = release.load_config(cfg["release_config"])
    if any(cfg["continuation"][k] != old["continuation"][k] for k in (
            "max_updates", "selection_updates", "selection_rows", "early_stopping", "wm_update_during_policy_training", "rl_rollout_source")):
        raise ValueError("new seeds must use the original continuation budget and Val selector")
    if cfg["schedules"] != {name: old["schedules"][name] for name in ("control", "floor_gradual")}:
        raise ValueError("same registered release curriculum is required for all seeds")
    original = parent.load_config(cfg["source_config"])
    for key, value in cfg["accounting"].items():
        if key != "position_min" and original["accounting"].get(key) != value:
            raise ValueError("execution costs/risk/rate cap cannot change in this extension")
    if cfg["accounting"]["position_min"] != 0.:
        raise ValueError("reference baselines require explicitly disclosed floor0 accounting")
    if cfg["baselines"] != {"constant_exposure": .08, "ma_window_bars": 1920,
                             "volatility_window_bars": 672, "volatility_target_annual": .10}:
        raise ValueError("causal baselines must be fixed before additional Test evaluation")
    if (cfg["statistics"]["primary_family_size"] != 12 or cfg["statistics"]["policy_family_size"] != 2
            or cfg["statistics"]["combined_family_size"] != 14
            or cfg["statistics"]["expected_market_rows"] != 50038
            or not 1 <= cfg["runtime"]["workers"] <= 12 or cfg["runtime"]["cpu_threads"] != 1):
        raise ValueError("fixed comparison families and bounded workers required")
    kernel = copy.deepcopy(original)
    kernel["output_dir"] = cfg["output_dir"]
    kernel["device"] = cfg["device"]
    kernel["runtime"]["cpu_threads"] = cfg["runtime"]["cpu_threads"]
    cfg["kernel_config"] = kernel
    cache = json.loads((Path(original["output_dir"]) / "prepared/cache_manifest.json").read_text())
    cfg["data"] = {"raw_close_path": cache["arrays"]["raw_close"]["path"],
                   "raw_index_ns_path": cache["arrays"]["raw_index_ns"]["path"]}
    cfg["source_output_dir"] = original["output_dir"]
    cfg["release_output_dir"] = old["output_dir"]
    return cfg


def jobs(cfg):
    result = []
    for fold in cfg["folds"]:
        for seed in cfg["new_seeds"]:
            wm_job = f"wm_fold_{fold:02d}_s{seed}"
            result.append({"job": wm_job, "kind": "wm", "fold": fold, "seed": seed,
                           "fit_path": f"world_models/fold_{fold:02d}_s{seed}/fit.json", "dependencies": []})
            for arm in ("A", "C"):
                result.append({"job": f"policy_fold_{fold:02d}_{arm}_s{seed}", "kind": "policy", "fold": fold,
                               "seed": seed, "arm": arm, "fit_path": f"neural/fold_{fold:02d}_{arm}_s{seed}/fit.json",
                               "dependencies": [wm_job] if arm == "C" else []})
            for condition_id in cfg["conditions"]:
                definition = cfg["condition_definitions"][condition_id]
                arm, condition = definition["arm"], definition["condition"]
                job = f"fold_{fold:02d}_{arm}_s{seed}_{condition}"
                result.append({"job": job, "kind": "continuation", "fold": fold, "seed": seed,
                               "arm": arm, "condition": condition, "condition_id": condition_id,
                               "fit_path": f"neural/{job}/fit.json",
                               "dependencies": [f"policy_fold_{fold:02d}_{arm}_s{seed}"]})
    return result


def reused_jobs(cfg):
    root = Path(cfg["release_output_dir"])
    result = []
    for fold in cfg["folds"]:
        for seed in cfg["reused_seeds"]:
            for condition_id in cfg["conditions"]:
                definition = cfg["condition_definitions"][condition_id]
                job = f"fold_{fold:02d}_{definition['arm']}_s{seed}_{definition['condition']}"
                result.append({"job": job, "fold": fold, "seed": seed, "condition": condition_id,
                               "arm": definition["arm"], "source_condition": definition["condition"],
                               "fit_path": str(root / "neural" / job / "fit.json"),
                               "evaluation_path": str(root / "evaluations" / job / "executed_ledger.parquet"),
                               "evaluation_marker": str(root / "evaluations" / job / "evaluation.json"), "reused": True})
    return result


def source_hashes(cfg):
    files = sorted(Path(__file__).parent.glob("wm_seed_robustness_*.py"))
    files += sorted((ROOT / "tests").glob("test_wm_seed_robustness_*.py")) + [Path(cfg["config_path"])]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def source_bindings(cfg):
    old_cfg = release.load_config(cfg["release_config"])
    old_reg = release.verify(old_cfg, check_sources=True)
    release.verify_test_barrier(old_cfg)
    source_cfg = parent.load_config(cfg["source_config"])
    source_reg = parent.verify(source_cfg, cache=True)
    artifacts = {}
    for name in ("preregistration.json", "completion.json", "selection_barrier.json"):
        path = Path(cfg["release_output_dir"]) / name
        artifacts[str(path)] = sha(path)
    manifest_path = Path(cfg["source_output_dir"]) / "prepared/cache_manifest.json"
    artifacts[str(manifest_path)] = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for entry in [*manifest["arrays"].values(), *manifest["fold_arrays"].values()]:
        path = Path(entry["path"])
        if sha(path) != entry["sha256"]:
            raise ValueError("original prepared array/normalizer changed")
        artifacts[str(path)] = entry["sha256"]
    for task in reused_jobs(cfg):
        fit_path = Path(task["fit_path"])
        fit = json.loads(fit_path.read_text())
        if not fit.get("complete") or fit.get("test_used") is not False or fit.get("updates") != 200:
            raise ValueError("reused selected continuation fit must satisfy its original contract")
        artifacts[str(fit_path)] = sha(fit_path)
        for name, digest in fit["artifact_sha256"].items():
            path = fit_path.parent / name
            if sha(path) != digest:
                raise ValueError("reused selected continuation artifact changed")
            artifacts[str(path)] = digest
        for key in ("evaluation_path", "evaluation_marker"):
            path = Path(task[key]); artifacts[str(path)] = sha(path)
    return {"artifact_sha256": artifacts, "old_release_source_sha256": old_reg["source_sha256"],
            "old_policy_source_sha256": source_reg["source_sha256"],
            "old_release_registration_sha256": sha(Path(cfg["release_output_dir"]) / "preregistration.json")}


def register(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "preregistration.json"
    if path.exists():
        registration = verify(cfg, check_sources=True)
        prepare_cache(cfg, sha(path))
        return registration
    bindings = source_bindings(cfg)
    registration = {"registered_at": now(), "config": cfg, "jobs": jobs(cfg), "reused_jobs": reused_jobs(cfg),
                    "source_sha256": source_hashes(cfg), "sources": bindings,
                    "package_versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "pandas", "scipy", "PyYAML")},
                    "seed_scope": "7 fresh WM/BC/initial-RL/continuation seeds plus 3 immutable prior seeds",
                    "new_training_fits": {"wm": 42, "policy_bc_initial_rl": 84, "continuation": 126},
                    "total_evaluated_continuations": 180, "reused_continuations": 54,
                    "mainline_and_previous_studies_changed": False, "historical_seen_test": True,
                    "independently_unused_holdout": False, "test_selection_allowed": False,
                    "baseline_selection": "constant8 user requested from observed exposure; MA20d and vol10pct/7d fixed before new results",
                    "comparison_scope": "same saved50038bar cohort; not continuous full-calendar trading",
                    "barrier": "all252 new training fits and54 old selections immutable before new126 Test inferences",
                    "multiple_testing": "baseline12 Holm, policy2 Holm, all14 sensitivity; prior search not automatically corrected"}
    write_json(path, registration)
    prepare_cache(cfg, sha(path))
    return registration


def prepare_cache(cfg, registration_sha):
    root = Path(cfg["output_dir"]) / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    source = Path(cfg["source_output_dir"]) / "prepared/cache_manifest.json"
    manifest = copy.deepcopy(json.loads(source.read_text()))
    manifest["registration_sha256"] = registration_sha
    manifest["shared_readonly_source_manifest"] = str(source)
    manifest["shared_readonly_source_manifest_sha256"] = sha(source)
    for fold in cfg["folds"]:
        entry = manifest["fold_arrays"][str(fold)]
        destination = root / f"fold_{fold:02d}.npz"
        if destination.exists():
            if sha(destination) != entry["sha256"]:
                raise ValueError("new study fold normalizer changed")
        else:
            shutil.copy2(entry["path"], destination)
        entry["path"] = str(destination)
    path = root / "cache_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("new study cache binding changed")
    write_json(path, manifest)


def verify(cfg, *, check_sources=False):
    root = Path(cfg["output_dir"])
    registration = json.loads((root / "preregistration.json").read_text())
    if (registration["config"] != cfg or registration["jobs"] != jobs(cfg)
            or registration["reused_jobs"] != reused_jobs(cfg) or registration["source_sha256"] != source_hashes(cfg)):
        raise ValueError("registered experiment/config/source changed")
    if registration.get("package_versions") != {n: importlib.metadata.version(n) for n in ("torch", "numpy", "pandas", "scipy", "PyYAML")}:
        raise ValueError("registered numerical package versions changed")
    if check_sources:
        if source_bindings(cfg) != registration["sources"]:
            raise ValueError("immutable old-study or source-cache binding changed")
    return registration


def checked_fit(cfg, task):
    root = Path(cfg["output_dir"])
    fit_path = root / task["fit_path"]
    fit = json.loads(fit_path.read_text())
    if (not fit.get("complete") or fit.get("registration_sha256") != sha(root / "preregistration.json")
            or fit.get("fold") != task["fold"] or fit.get("seed") != task["seed"] or fit.get("test_used") is not False):
        raise ValueError("complete Train/Val-only fit identity required")
    if task["kind"] != "wm" and fit.get("arm") != task["arm"]:
        raise ValueError("actor input arm changed")
    fold_sha = sha(root / "prepared" / f"fold_{task['fold']:02d}.npz")
    if task["kind"] == "wm":
        if (fit.get("test_read") is not False or fit.get("normalizer_train_fold_sha256") != fold_sha
                or fit.get("latent_normalizer_fit_split") != "policy Train only"):
            raise ValueError("fresh WM must retain Train-only normalization and unread Test")
    elif fit.get("normalizer_fit_split") != "train":
        raise ValueError("policy normalization must be fitted inside this fold's Train")
    if task["kind"] == "policy" and (fit.get("warm_start") is not False or fit.get("fold_arrays_sha256") != fold_sha):
        raise ValueError("new seed initial BC/RL must start fresh with its own Train fold")
    if task["kind"] == "continuation":
        if (fit.get("condition") != task["condition"] or fit.get("condition_id") != task["condition_id"]
                or fit.get("updates") != 200 or fit.get("selected_update") not in cfg["continuation"]["selection_updates"]
                or fit.get("warm_start") is not True or fit.get("wm_frozen") is not True
                or fit.get("bc_prior_immutable") is not True or fit.get("optimizer_moments_inherited") is not False
                or fit.get("optimizer_reset_at_stage_boundary") is not False
                or fit.get("historical_transitions") != 204800):
            raise ValueError("same fixed continuation budget and Val selection window required")
    for name, digest in fit["artifact_sha256"].items():
        if sha(fit_path.parent / name) != digest:
            raise ValueError("selected fit artifact changed")
    return fit


def freeze_test_barrier(cfg):
    registration = verify(cfg, check_sources=True)
    root = Path(cfg["output_dir"])
    frozen, choices, fits = {}, {}, {}
    for task in registration["jobs"]:
        fit = checked_fit(cfg, task)
        fits[task["job"]] = fit
        frozen[task["fit_path"]] = sha(root / task["fit_path"])
        choices[task["job"]] = {"fit_sha256": frozen[task["fit_path"]], "test_used": False,
            "selected_update": fit.get("selected_update"), "wm_best_step": fit.get("best_step"),
            "bc_selection": fit.get("bc"), "initial_rl_selection": fit.get("rl")}
    for fold in cfg["folds"]:
        for seed in cfg["new_seeds"]:
            paired = [fits[f"fold_{fold:02d}_C_s{seed}_{name}"] for name in ("control", "floor_gradual")]
            for key in ("initial_state_sha256", "immutable_bc_prior_state_sha256", "market_start_sequence_sha256", "continuation_random_seed"):
                if paired[0].get(key) is None or paired[0][key] != paired[1].get(key):
                    raise ValueError("control and release must have common initialization, prior and market-start stream")
    old_fits = {t["fit_path"]: sha(Path(t["fit_path"])) for t in reused_jobs(cfg)}
    write_json(root / "validation_choices.json", {"complete": True, "test_used": False, "choices": choices})
    barrier = {"complete": True, "registration_sha256": sha(root / "preregistration.json"),
               "new_fits": 252, "reused_selections": 54, "frozen_fit_sha256": frozen,
               "reused_fit_sha256": old_fits, "validation_choices_sha256": sha(root / "validation_choices.json"),
               "cache_manifest_sha256": sha(root / "prepared/cache_manifest.json"), "test_selection_allowed": False}
    path = root / "test_barrier.json"
    if path.exists() and json.loads(path.read_text()) != barrier:
        raise ValueError("new Test barrier already frozen differently")
    write_json(path, barrier)
    return barrier


def verify_test_barrier(cfg):
    registration = verify(cfg)
    root = Path(cfg["output_dir"])
    barrier = json.loads((root / "test_barrier.json").read_text())
    expected = {task["fit_path"] for task in registration["jobs"]}
    if (not barrier.get("complete") or barrier.get("registration_sha256") != sha(root / "preregistration.json")
            or barrier.get("test_selection_allowed") is not False
            or set(barrier.get("frozen_fit_sha256", {})) != expected
            or barrier.get("validation_choices_sha256") != sha(root / "validation_choices.json")
            or barrier.get("cache_manifest_sha256") != sha(root / "prepared/cache_manifest.json")):
        raise ValueError("all new fits and immutable Val choices must precede Test")
    for name, digest in barrier["frozen_fit_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Train/Val fit changed after Test barrier")
    if set(barrier["reused_fit_sha256"]) != {task["fit_path"] for task in reused_jobs(cfg)}:
        raise ValueError("all reused prior-seed selections must remain bound")
    for name, digest in barrier["reused_fit_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("reused fit changed after Test barrier")
    for name in cfg["data"].values():
        if sha(Path(name)) != registration["sources"]["artifact_sha256"][name]:
            raise ValueError("registered raw price or time array changed")
    return barrier


def done(cfg, task, evaluation=False):
    root = Path(cfg["output_dir"])
    path = root / (f"evaluations/{task['job']}/evaluation.json" if evaluation else task["fit_path"])
    if not path.exists():
        return False
    obj = json.loads(path.read_text())
    return obj.get("complete") is True and obj.get("registration_sha256") == sha(root / "preregistration.json")


def progress(cfg, phase, tasks, completed, active, started):
    root = Path(cfg["output_dir"])
    value = {"phase": phase, "at": now(), "completed": len(completed), "total": len(tasks),
             "elapsed_seconds": time.monotonic() - started, "running": list(active),
             "test_selection_allowed": False, "independently_unused_holdout": False}
    write_json(root / "status.json", value)
    (root / "progress.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>10seed再現性実験</title><style>body{font:18px system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style><h1>10seed・単純売買対照・分割監査</h1><p>新規WM42＋BC/初期RL84＋追加RL126。元54条件を再利用し、計180のfold別成績を比較します。</p><pre>' + json.dumps(value, ensure_ascii=False, indent=2) + '</pre></html>')
    print(json.dumps(value), flush=True)


def run_pool(cfg, evaluation=False):
    root = Path(cfg["output_dir"])
    tasks = [t for t in jobs(cfg) if not evaluation or t["kind"] == "continuation"]
    completed = {t["job"] for t in tasks if done(cfg, t, evaluation)}
    active, started = {}, time.monotonic()
    (root / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    last_status = -float("inf")
    try:
        while len(completed) < len(tasks):
            for task in tasks:
                if len(active) >= cfg["runtime"]["workers"]:
                    break
                if task["job"] in completed or task["job"] in active:
                    continue
                if not evaluation and not set(task["dependencies"]).issubset(completed):
                    continue
                suffix = "_test" if evaluation else ""
                log = (root / "logs" / f"{task['job']}{suffix}.log").open("a")
                cmd = [sys.executable, "-m", "unidream.experiments.wm_seed_robustness_benchmark", "--config", cfg["config_path"],
                       "--action", "evaluate-worker" if evaluation else "train-worker", "--job", task["job"]]
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[task["job"]] = (process, task, log)
            for name, (process, task, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del active[name]
                if code != 0 or not done(cfg, task, evaluation):
                    write_json(root / "failure.json", {"job": name, "phase": "evaluation" if evaluation else "training", "returncode": code, "at": now()})
                    raise RuntimeError(f"worker failed: {name}; see logs")
                completed.add(name)
            if time.monotonic() - last_status >= cfg["runtime"]["status_seconds"] or len(completed) == len(tasks):
                progress(cfg, "evaluation" if evaluation else "training", tasks, completed, active, started)
                last_status = time.monotonic()
            if len(completed) < len(tasks):
                if not active:
                    raise RuntimeError("unsatisfied dependency graph without running workers")
                time.sleep(1)
    finally:
        for process, _, log in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()


def evaluation_manifest(cfg):
    root = Path(cfg["output_dir"])
    verify_test_barrier(cfg)
    entries = []
    for task in reused_jobs(cfg):
        path = Path(task["evaluation_path"])
        entries.append({"job": task["job"], "fold": task["fold"], "seed": task["seed"], "condition": task["condition"],
                        "path": str(path), "sha256": sha(path), "reused": True})
    for task in jobs(cfg):
        if task["kind"] != "continuation":
            continue
        directory = root / "evaluations" / task["job"]
        marker = json.loads((directory / "evaluation.json").read_text())
        path = directory / "executed_ledger.parquet"
        if marker.get("prediction_sha256") != sha(path) or marker.get("barrier_sha256") != sha(root / "test_barrier.json"):
            raise ValueError("completed new-seed Test ledger changed")
        entries.append({"job": task["job"], "fold": task["fold"], "seed": task["seed"], "condition": task["condition_id"],
                        "path": str(path), "sha256": sha(path), "reused": False})
    manifest = {"complete": True, "registration_sha256": sha(root / "preregistration.json"),
                "barrier_sha256": sha(root / "test_barrier.json"), "jobs": entries}
    if len(entries) != 180:
        raise ValueError("all ten seeds and six folds for all three policies required")
    write_json(root / "evaluation_manifest.json", manifest)
    return manifest


def run(cfg):
    root = Path(cfg["output_dir"])
    registration = register(cfg)
    completed_path = root / "completion.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        for name, digest in completed["artifact_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("completed report artifact changed")
        return completed
    from .wm_seed_robustness_audit import audit_source
    audit = audit_source(parent.load_config(cfg["source_config"]), root / "split_audit")
    write_json(root / "source_audit_result.json", audit)
    if not audit.get("passed"):
        raise ValueError("split/label/normalizer audit failed; inspect source_audit_result.json before training")
    # History may include older splits because only past inputs, not future
    # labels, enter. Actual fold-local leakage fails closed above.
    run_pool(cfg)
    freeze_test_barrier(cfg)
    run_pool(cfg, evaluation=True)
    evaluation_manifest(cfg)
    from .wm_seed_robustness_evaluation import aggregate
    report = aggregate(cfg, sha(root / "preregistration.json"))
    verify(cfg, check_sources=True)
    artifacts = {str(p.relative_to(root)): sha(p) for p in (root / "final").iterdir() if p.is_file()}
    completed = {"complete": True, "at": now(), "registration_sha256": sha(root / "preregistration.json"),
                 "new_training_fits": len(registration["jobs"]), "evaluated_continuations": 180,
                 "source_artifacts_unchanged": True, "test_selection_allowed": False,
                 "independently_unused_holdout": False, "report": report, "artifact_sha256": artifacts}
    write_json(completed_path, completed)
    progress(cfg, "complete", jobs(cfg), {t["job"] for t in jobs(cfg)}, {}, time.monotonic())
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--action", choices=("run", "register", "train-worker", "evaluate-worker"), default="run")
    parser.add_argument("--job")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action in ("train-worker", "evaluate-worker"):
        import torch
        from . import wm_seed_robustness_training as training
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        verify(cfg)
        task = next((t for t in jobs(cfg) if t["job"] == args.job), None)
        if task is None:
            raise ValueError("worker task must belong to the fixed YAML grid")
        suffix = "_test" if args.action == "evaluate-worker" else ""
        with parent.lock(Path(cfg["output_dir"]) / "locks" / f"{task['job']}{suffix}.lock"):
            (training.evaluate_policy if suffix else training.worker)(cfg, task, sha(Path(cfg["output_dir"]) / "preregistration.json"))
    elif args.action == "register":
        register(cfg)
    else:
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        with parent.lock(Path(cfg["output_dir"]) / "runner.lock"):
            run(cfg)


if __name__ == "__main__":
    main()
