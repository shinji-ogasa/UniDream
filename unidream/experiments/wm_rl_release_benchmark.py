"""Isolated continuation study of RL position-floor and BC-prior curricula.

The mainline and the frozen A/B/C/D study are read-only. This entry starts
from selected research RL weights with a fresh optimizer, and opens Test only
after the complete, registered continuation grid has frozen its Val choices.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

from . import wm_policy_ablation_benchmark as parent
from .wm_policy_ablation_policy import sha, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_rl_release_20261112.yaml"
CONDITIONS = ["control", "floor_abrupt", "floor_gradual", "prior_abrupt", "prior_gradual"]


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    for key in ("output_dir", "source_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    c = cfg["continuation"]
    if (cfg["arms"] != ["A", "C"] or cfg["folds"] != list(range(6))
            or cfg["seeds"] != [7, 19, 42] or cfg["conditions"] != CONDITIONS
            or cfg.get("test_selection_allowed") is not False
            or cfg.get("independently_unused_holdout") is not False
            or cfg.get("historical_seen_test") is not True
            or c["max_updates"] != 200 or c["selection_updates"] != [125, 150, 175, 200]
            or c.get("early_stopping") is not False
            or c.get("wm_update_during_policy_training") is not False
            or c.get("rl_rollout_source") != "actual_historical_Train_segments"):
        raise ValueError("fixed paired continuation grid and report-only known Test required")
    expected = {
        "control": [(200, .5, .01)],
        "floor_abrupt": [(200, 0., .01)],
        "floor_gradual": [(50, .5, .01), (100, .25, .01), (200, 0., .01)],
        "prior_abrupt": [(200, .5, 0.)],
        "prior_gradual": [(50, .5, .01), (100, .5, .003), (200, .5, 0.)],
    }
    for name, stages in expected.items():
        observed = [(s["through_update"], s["position_min"], s["bc_prior_coefficient"])
                    for s in cfg["schedules"][name]]
        if observed != stages:
            raise ValueError("registered release schedules cannot change")
    if not 1 <= cfg["runtime"]["workers"] <= 12 or cfg["runtime"]["cpu_threads"] != 1:
        raise ValueError("bounded worker pool and single CPU thread per worker required")
    p = parent.load_config(cfg["source_config"])
    for key, value in cfg["accounting"].items():
        if key != "inherited_from_source" and p["accounting"].get(key) != value:
            raise ValueError("source costs, risk objective and execution mechanics must remain unchanged")
    return cfg


def jobs(cfg):
    result = []
    for fold in cfg["folds"]:
        for seed in cfg["seeds"]:
            for arm in cfg["arms"]:
                for condition in cfg["conditions"]:
                    name = f"fold_{fold:02d}_{arm}_s{seed}_{condition}"
                    result.append({"job": name, "fold": fold, "seed": seed, "arm": arm,
                                   "condition": condition, "fit_path": f"neural/{name}/fit.json"})
    return result


def source_hashes(cfg):
    files = sorted(Path(__file__).parent.glob("wm_rl_release_*.py"))
    files += sorted((ROOT / "tests").glob("test_wm_rl_release_*.py"))
    files += [Path(cfg["config_path"])]
    # Reused kernels are bound without changing their previous registrations.
    files += [Path(__file__).with_name(name) for name in (
        "wm_inference_robust_bootstrap.py", "wm_inference_robust_hac.py",
        "wm_significance_statistics.py")]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def source_bindings(cfg):
    pcfg = parent.load_config(cfg["source_config"])
    prior = parent.verify(pcfg, cache=True)
    barrier = parent.verify_test_barrier(pcfg)
    root = Path(pcfg["output_dir"])
    manifest = json.loads((root / "prepared/cache_manifest.json").read_text())
    # Check actual immutable bytes once at registration and global barriers,
    # rather than hashing the common mmap arrays in every training process.
    for entry in [*manifest["arrays"].values(), *manifest["fold_arrays"].values()]:
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("original prepared market array bytes changed")
    artifacts = {}
    for name in ("preregistration.json", "test_barrier.json", "validation_choices.json",
                 "prepared/cache_manifest.json", "normalizer_provenance.json",
                 "final/executed_policy_ledger.parquet"):
        artifacts[name] = sha(root / name)
    for fold in cfg["folds"]:
        artifacts[f"prepared/fold_{fold:02d}.npz"] = sha(root / "prepared" / f"fold_{fold:02d}.npz")
        for seed in cfg["seeds"]:
            for arm in cfg["arms"]:
                sub = root / "neural" / f"fold_{fold:02d}_{arm}_s{seed}"
                fit = json.loads((sub / "fit.json").read_text())
                if not fit.get("complete") or fit["registration_sha256"] != artifacts["preregistration.json"]:
                    raise ValueError("complete source fits required for every continuation branch")
                for filename in ("fit.json", "rl.pt", "bc.pt"):
                    relative = str((sub / filename).relative_to(root))
                    digest = sha(sub / filename)
                    if filename != "fit.json" and fit["artifact_sha256"][filename] != digest:
                        raise ValueError("source policy checkpoint changed")
                    artifacts[relative] = digest
            wm_root = root / "world_models" / f"fold_{fold:02d}_s{seed}"
            wm_fit = json.loads((wm_root / "fit.json").read_text())
            for name in ("fit.json", "model.pt", "latent_train_val.npy", "latent_norm.npz",
                         "latent_test.npy", "test_latent.json"):
                digest = sha(wm_root / name)
                if name in wm_fit["artifact_sha256"] and digest != wm_fit["artifact_sha256"][name]:
                    raise ValueError("original frozen WM artifact changed")
                artifacts[str((wm_root / name).relative_to(root))] = digest
            marker = json.loads((wm_root / "test_latent.json").read_text())
            if (marker.get("registration_sha256") != artifacts["preregistration.json"]
                    or marker.get("barrier_sha256") != artifacts["test_barrier.json"]
                    or marker.get("latent_sha256") != artifacts[str((wm_root / "latent_test.npy").relative_to(root))]):
                raise ValueError("original report-only latent cache changed")
    return {"source_root": str(root), "source_config": pcfg,
            "source_sha256": prior["source_sha256"], "artifact_sha256": artifacts,
            "source_test_barrier_sha256": sha(root / "test_barrier.json"),
            "source_fits": len(barrier["frozen_fit_sha256"])}


def register(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "preregistration.json"
    if path.exists():
        return verify(cfg, check_sources=True)
    sources = source_bindings(cfg)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "scipy", "PyYAML")}
    registration = {"registered_at": now(), "config": cfg, "jobs": jobs(cfg),
        "source_sha256": source_hashes(cfg), "sources": sources, "package_versions": versions,
        "initialization": "same Val-selected source RL weights and fresh optimizer within each fold/seed/arm",
        "optimizer_state": "not present in source checkpoint; reset identically across all conditions, never reset at curriculum stages",
        "BC_reference": "original immutable BC checkpoint, not the RL initialization",
        "WM": "original frozen X-only WM and Train normalizers, shared by C continuation conditions",
        "comparison": "A X-only vs C X+WM latent; source A/C not chosen using new outcomes",
        "training": "fixed200 new Train PPO updates, same rollout-start random stream across conditions; fresh actions every update",
        "selection": "Val-only, common125/150/175/200 checkpoints after gradual branches reach final environment",
        "objective": "unchanged net log excess minus .25 incremental maxDD, same costs, lag and maximum exposure",
        "primary_family": "12: four releases vs continued control and gradual vs abrupt within each of A/C",
        "secondary_family": "WM C-minus-A for five conditions; separate Holm5; combined17 sensitivity",
        "risk_matched_family": "10: each arm/condition vs constant exposure selected from its full Val; Holm10 and combined27 sensitivity",
        "barrier": "all180 selected fits frozen before any new Test policy inference",
        "historical_seen_test": True, "independently_unused_holdout": False,
        "test_selection_allowed": False,
        "scope": "post-hoc development on saved contiguous forecast cohort; not a fresh holdout or full-calendar/live performance",
        "source_models_costs_or_mainline_changed": False}
    write_json(path, registration)
    return registration


def verify(cfg, *, check_sources=False):
    path = Path(cfg["output_dir"]) / "preregistration.json"
    registration = json.loads(path.read_text())
    if (registration["config"] != cfg or registration["jobs"] != jobs(cfg)
            or registration["source_sha256"] != source_hashes(cfg)):
        raise ValueError("registered continuation methods, sources or task grid changed")
    for name, version in registration["package_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("registered package version changed")
    if check_sources and registration["sources"] != source_bindings(cfg):
        raise ValueError("frozen original study changed")
    return registration


def _fit(cfg, task):
    path = Path(cfg["output_dir"]) / task["fit_path"]
    obj = json.loads(path.read_text())
    reg_sha = sha(Path(cfg["output_dir"]) / "preregistration.json")
    if (not obj.get("complete") or obj.get("registration_sha256") != reg_sha
            or obj.get("test_used") is not False
            or any(obj.get(key) != task[key] for key in ("job", "fold", "seed", "arm", "condition"))):
        raise ValueError("complete registered continuation fit required")
    for name, digest in obj["artifact_sha256"].items():
        if sha(path.parent / name) != digest:
            raise ValueError("selected fit artifact changed")
    return obj


def freeze_test_barrier(cfg):
    registration = verify(cfg, check_sources=True)
    root = Path(cfg["output_dir"])
    frozen, choices, initial, streams, references = {}, {}, {}, {}, {}
    for task in registration["jobs"]:
        obj = _fit(cfg, task)
        frozen[task["fit_path"]] = sha(root / task["fit_path"])
        if obj.get("selected_update") not in cfg["continuation"]["selection_updates"]:
            raise ValueError("checkpoint selected before common final-environment Val window")
        if obj.get("updates") != cfg["continuation"]["max_updates"]:
            raise ValueError("every continuation branch must use the same complete training budget")
        key = (task["fold"], task["seed"], task["arm"])
        initial.setdefault(key, set()).add(obj["initial_state_sha256"])
        streams.setdefault(key, set()).add(obj["market_start_sequence_sha256"])
        references.setdefault(key, set()).add(obj["immutable_bc_prior_state_sha256"])
        choices[task["job"]] = {"selected_update": obj["selected_update"],
            "fit_sha256": frozen[task["fit_path"]], "test_used": False}
    if any(len(value) != 1 for collection in (initial, streams, references) for value in collection.values()):
        raise ValueError("paired branches require identical initial weights, market starts and frozen BC reference")
    choice_path = root / "validation_choices.json"
    choice_obj = {"choices": choices, "test_used": False}
    if choice_path.exists() and json.loads(choice_path.read_text()) != choice_obj:
        raise ValueError("Val choices already frozen differently")
    write_json(choice_path, choice_obj)
    obj = {"complete": True, "registration_sha256": sha(root / "preregistration.json"),
        "validation_choices_sha256": sha(choice_path), "frozen_fit_sha256": frozen,
        "test_selection_allowed": False, "independently_unused_holdout": False,
        "source_bindings_sha256": hashlib.sha256(json.dumps(registration["sources"], sort_keys=True).encode()).hexdigest()}
    path = root / "selection_barrier.json"
    if path.exists() and json.loads(path.read_text()) != obj:
        raise ValueError("global Test barrier already frozen differently")
    write_json(path, obj)
    return obj


def verify_test_barrier(cfg):
    registration = verify(cfg)
    root = Path(cfg["output_dir"])
    obj = json.loads((root / "selection_barrier.json").read_text())
    choices = json.loads((root / "validation_choices.json").read_text())
    expected = {task["fit_path"] for task in registration["jobs"]}
    if (obj.get("complete") is not True or obj.get("test_selection_allowed") is not False
            or obj.get("registration_sha256") != sha(root / "preregistration.json")
            or set(obj.get("frozen_fit_sha256", {})) != expected
            or obj.get("validation_choices_sha256") != sha(root / "validation_choices.json")
            or choices.get("test_used") is not False
            or set(choices.get("choices", {})) != {task["job"] for task in registration["jobs"]}
            or obj.get("source_bindings_sha256") != hashlib.sha256(json.dumps(registration["sources"], sort_keys=True).encode()).hexdigest()):
        raise ValueError("complete immutable all-fit barrier must precede Test")
    for name, digest in obj["frozen_fit_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("a selected continuation fit changed after the barrier")
    for task in registration["jobs"]:
        fit = _fit(cfg, task)
        choice = choices["choices"][task["job"]]
        if (choice.get("test_used") is not False
                or choice.get("selected_update") != fit.get("selected_update")
                or choice.get("fit_sha256") != obj["frozen_fit_sha256"][task["fit_path"]]):
            raise ValueError("selected checkpoint and frozen Val choice disagree")
    return obj


def _done(cfg, task, evaluation_phase=False):
    root = Path(cfg["output_dir"])
    path = root / (f"evaluations/{task['job']}/evaluation.json" if evaluation_phase else task["fit_path"])
    return path.exists() and json.loads(path.read_text()).get("complete") is True


def worker(cfg, task, evaluation_phase=False):
    import torch
    from . import wm_rl_release_policy as policy
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    verify(cfg)
    reg_sha = sha(Path(cfg["output_dir"]) / "preregistration.json")
    suffix = "_test" if evaluation_phase else ""
    with parent.lock(Path(cfg["output_dir"]) / "locks" / f"{task['job']}{suffix}.lock"):
        return (policy.evaluate_policy if evaluation_phase else policy.train_policy)(cfg, task, reg_sha)


def progress(cfg, phase, tasks, done, active, started):
    root = Path(cfg["output_dir"])
    snapshot = {"phase": phase, "at": now(), "completed": len(done), "total": len(tasks),
        "elapsed_seconds": time.monotonic() - started, "running": list(active),
        "test_selection_allowed": False, "independently_unused_holdout": False}
    write_json(root / "status.json", snapshot)
    import html
    (root / "progress.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        '<title>RL 制限緩和実験</title><style>body{font:18px system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style>'
        '<h1>RL：保有下限とBC制約の緩和</h1><p>A/C × 5条件 × 6fold × 3seed = 180学習。既知Testによる開発評価。</p><pre>'
        + html.escape(json.dumps(snapshot, ensure_ascii=False, indent=2)) + '</pre></html>')
    print(json.dumps(snapshot), flush=True)


def run_pool(cfg, evaluation_phase=False):
    root = Path(cfg["output_dir"])
    tasks = jobs(cfg)
    done = {t["job"] for t in tasks if _done(cfg, t, evaluation_phase)}
    active, started = {}, time.monotonic()
    (root / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    last_status = -float("inf")
    try:
        while len(done) < len(tasks):
            for task in tasks:
                if len(active) >= cfg["runtime"]["workers"]:
                    break
                if task["job"] in done or task["job"] in active:
                    continue
                suffix = "_test" if evaluation_phase else ""
                log = (root / "logs" / f"{task['job']}{suffix}.log").open("a")
                cmd = [sys.executable, "-m", "unidream.experiments.wm_rl_release_benchmark", "--config", cfg["config_path"],
                       "--action", "evaluate-worker" if evaluation_phase else "train-worker", "--job", task["job"]]
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[task["job"]] = (process, task, log)
            for name, (process, task, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[name]
                if code != 0 or not _done(cfg, task, evaluation_phase):
                    write_json(root / "failure.json", {"job": name, "phase": "evaluation" if evaluation_phase else "training", "returncode": code, "at": now()})
                    raise RuntimeError(f"worker failed: {name}; see {root / 'logs' / (name + ('_test' if evaluation_phase else '') + '.log')}")
                done.add(name)
            if time.monotonic() - last_status >= cfg["runtime"]["status_seconds"] or len(done) == len(tasks):
                progress(cfg, "evaluation" if evaluation_phase else "training", tasks, done, active, started)
                last_status = time.monotonic()
            if len(done) < len(tasks):
                time.sleep(1)
    finally:
        for process, _, log in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()


def run(cfg):
    registration = register(cfg)
    root = Path(cfg["output_dir"])
    path = root / "completion.json"
    if path.exists():
        completed = json.loads(path.read_text())
        if completed["registration_sha256"] != sha(root / "preregistration.json"):
            raise ValueError("completion belongs to another protocol")
        for name, digest in completed["artifact_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("completed study result changed")
        return completed
    run_pool(cfg)
    freeze_test_barrier(cfg)
    run_pool(cfg, evaluation_phase=True)
    verify_test_barrier(cfg)
    from .wm_rl_release_evaluation import aggregate
    report = aggregate(cfg, sha(root / "preregistration.json"))
    verify(cfg, check_sources=True)
    artifacts = {str(p.relative_to(root)): sha(p) for p in (root / "final").iterdir() if p.is_file()}
    completed = {"complete": True, "at": now(), "registration_sha256": sha(root / "preregistration.json"),
        "fits": len(registration["jobs"]), "evaluations": len(registration["jobs"]), "artifact_sha256": artifacts,
        "source_artifacts_unchanged": True, "test_selection_allowed": False,
        "independently_unused_holdout": False, "report": report}
    write_json(path, completed)
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
        task = next((t for t in jobs(cfg) if t["job"] == args.job), None)
        if task is None:
            raise ValueError("worker task must belong to the YAML-registered grid")
        worker(cfg, task, args.action == "evaluate-worker")
    elif args.action == "register":
        register(cfg)
    else:
        with parent.lock(Path(cfg["output_dir"]) / "runner.lock"):
            run(cfg)


if __name__ == "__main__":
    main()
