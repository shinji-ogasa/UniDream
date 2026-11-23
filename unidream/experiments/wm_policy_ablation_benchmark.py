"""Registered, four-worker X/F/Z policy study; separate from mainline v31.

Run with --action run. All dates, folds, costs, budgets and seed axes come
from YAML. --job identifies an already registered worker, never overrides it.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import wm_policy_ablation_data as data
from . import wm_policy_ablation_evaluation as evaluation
from .wm_policy_ablation_policy import sha, write_json, train_source_id

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_policy_ablation_20261112.yaml"


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    cfg["output_dir"] = str((ROOT / cfg["output_dir"]).resolve())
    if cfg.get("arms") != ["A", "B", "C", "D"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("four registered arms and distinct paired seeds required")
    if cfg.get("test_selection_allowed") is not False or cfg.get("independently_unused_holdout") is not False:
        raise ValueError("this retrospective development study cannot claim unused Test")
    if cfg["policy"]["rl_rollout_source"] != "actual_historical_Train_segments":
        raise ValueError("A-D main comparison requires identical real-history RL")
    return cfg


def jobs(cfg):
    result = []
    for fold in cfg["folds"]:
        for seed in cfg["seeds"]:
            wm_id = f"wm_fold_{fold:02d}_s{seed}"
            result.append({"job": wm_id, "kind": "wm", "fold": fold, "seed": seed,
                "fit_path": f"world_models/fold_{fold:02d}_s{seed}/fit.json", "dependencies": []})
            for arm in cfg["arms"]:
                result.append({"job": f"policy_fold_{fold:02d}_{arm}_s{seed}", "kind": "policy",
                    "fold": fold, "seed": seed, "arm": arm,
                    "fit_path": f"neural/fold_{fold:02d}_{arm}_s{seed}/fit.json",
                    "dependencies": [wm_id] if arm in ("C", "D") else []})
    return result


def source_hashes(cfg):
    files = sorted(Path(__file__).parent.glob("wm_policy_ablation_*.py"))
    files += sorted((ROOT / "tests").glob("test_wm_policy_ablation_*.py"))
    files.append(Path(cfg["config_path"]))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def now():
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a live worker already owns {path}") from exc
        yield


def verify(cfg, *, cache=False):
    root = Path(cfg["output_dir"])
    path = root / "preregistration.json"
    registration = json.loads(path.read_text())
    if registration["config"] != cfg or registration["source_sha256"] != source_hashes(cfg):
        raise ValueError("registered policy configuration or source changed")
    if registration["jobs"] != jobs(cfg):
        raise ValueError("registered task grid changed")
    if cache:
        manifest = json.loads((root / "prepared/cache_manifest.json").read_text())
        if manifest.get("registration_sha256") != sha(path) or not manifest.get("complete"):
            raise ValueError("complete registered data preparation required")
    return registration


def verify_test_barrier(cfg):
    """Fail closed even when an evaluation worker is invoked directly."""
    root = Path(cfg["output_dir"])
    registration = verify(cfg, cache=True)
    barrier = json.loads((root / "test_barrier.json").read_text())
    expected = {task["fit_path"] for task in registration["jobs"]}
    if (barrier.get("registration_sha256") != sha(root / "preregistration.json")
            or barrier.get("validation_choices_sha256") != sha(root / "validation_choices.json")
            or set(barrier.get("frozen_fit_sha256", {})) != expected
            or barrier.get("test_selection_allowed") is not False):
        raise ValueError("every registered WM and policy fit must freeze before Test")
    choices = json.loads((root / "validation_choices.json").read_text())
    if choices.get("test_used") is not False:
        raise ValueError("Test was used in validation choices")
    if (barrier.get("cache_manifest_sha256") != sha(root / "prepared/cache_manifest.json")
            or barrier.get("normalizer_provenance_sha256") != sha(root / "normalizer_provenance.json")):
        raise ValueError("frozen policy cohort or normalizer provenance changed")
    for name, expected_sha in barrier["frozen_fit_sha256"].items():
        if sha(root / name) != expected_sha:
            raise ValueError("a fit changed after the complete global Test barrier")
    return barrier


def prepare(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "preregistration.json"
    if registration_path.exists():
        registration = verify(cfg)
    else:
        # Validate the frozen producer source/config chains before binding
        # their metadata. These verifiers do not open individual Test rows.
        for name, module in (("full", data.full), ("joint", data.joint), ("lstm", data.lstm)):
            module.verify(module.load_config(ROOT / cfg[name + "_config"]), full=False)
        sources = data.inspect_sources(cfg)
        if [row["fold"] for row in sources["folds"]] != cfg["folds"]:
            raise ValueError("YAML folds must exactly cover the six registered chronological windows")
        registration = {"registered_at": now(), "experiment_id": cfg["experiment_id"], "config": cfg,
            "source_sha256": source_hashes(cfg), "data_sources": sources, "jobs": jobs(cfg),
            "arm_definitions": {"A": "X+own_inventory", "B": "X+F+own_inventory",
                "C": "X+frozen_X_WM_Z+own_inventory", "D": "X+F+frozen_X_WM_Z+own_inventory"},
            "feature_selection": "all positively significant native heads; selected on previously seen forecast Test; exploratory",
            "wm_not_completely_removed_in_B": "F includes WM forecast heads; latent off differs from all WM information off",
            "teacher": "Train-only hindsight constant-position16bar utility argmax; projected current independently sampled inventory; not attainable Oracle upper bound",
            "rl_contract": "same real-history on-policy PPO in every arm; WM frozen; no imagination; no teacher loss in RL",
            "validation_selection": "fixed earlystop-month prefixes for checkpoint selection; final-month controls; no Test selection",
            "rl_checkpoint_selection": "genuinely updated checkpoint only, update>=1; no silent BC fallback",
            "global_barrier": "all 72 policy fits and18 shared upstream WM fits frozen before policy Test inference",
            "scope": "conditional contiguous saved-forecast cohort; reset accounts at gaps/folds; not full-calendar/live profit",
            "effective_raw_history_bars": 511,
            "statistics": "paired market-day block resampling shared across seeds; paired seed resampling; Holm12; 7day primary,1/28day sensitivity",
            "cost": "fixed fractional exposure approximation; 5.5bps turnover; annual10% excess leverage borrow assumption; initial endowed1x; onebar execution lag",
            "test_selection_allowed": False, "independently_unused_holdout": False}
        write_json(registration_path, registration)
    # This is the first individual-forecast/raw-price read for the new study.
    manifest = data.build_cache(cfg, registration)
    write_json(root / "prepared/preparation_complete.json", {"complete": True,
        "registration_sha256": sha(registration_path), "cache_manifest_sha256": manifest["cache_manifest_sha"],
        "n_heads": manifest["n_heads"], "forecast_dim": manifest["forecast_dim"],
        "n_context_valid": manifest["n_context_valid"], "at": now()})
    return registration


def _job_complete(root, task, *, evaluation_phase=False):
    path = root / task["fit_path"]
    if evaluation_phase:
        path = root / "evaluations" / path.parent.name / "evaluation.json"
    if not path.exists():
        return False
    obj = json.loads(path.read_text())
    return obj.get("complete") is True


def _worker(cfg, task, evaluation_phase):
    import torch
    torch.set_num_threads(int(cfg["runtime"]["cpu_threads"]))
    registration = verify(cfg, cache=True)
    reg_sha = sha(Path(cfg["output_dir"]) / "preregistration.json")
    task_id = task["job"] + ("_test" if evaluation_phase else "")
    root = Path(cfg["output_dir"])
    with lock(root / "locks" / f"{task_id}.lock"):
        if evaluation_phase:
            from .wm_policy_ablation_policy import evaluate_policy
            return evaluate_policy(cfg, task["fold"], task["seed"], task["arm"], reg_sha)
        if task["kind"] == "wm":
            from .wm_policy_ablation_world import train_wm
            return train_wm(cfg, task["fold"], task["seed"], reg_sha)
        for dependency in task["dependencies"]:
            upstream = next(j for j in registration["jobs"] if j["job"] == dependency)
            if not _job_complete(root, upstream):
                raise ValueError("policy worker upstream WM is not complete")
        from .wm_policy_ablation_policy import train_policy
        return train_policy(cfg, task["fold"], task["seed"], task["arm"], reg_sha)


def _progress(root, phase, tasks, done, active, started):
    snapshot = {"phase": phase, "at": now(), "elapsed_seconds": time.monotonic() - started,
        "completed": len(done), "total": len(tasks), "running": [t["job"] for _, t, _ in active.values()],
        "wm_complete": sum(t["job"] in done and t["kind"] == "wm" for t in tasks),
        "policy_complete": sum(t["job"] in done and t["kind"] == "policy" for t in tasks)}
    write_json(root / "status.json", snapshot)
    import html
    (root / "progress.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        '<title>WM policy A/B/C/D</title><style>body{font:18px system-ui;max-width:900px;margin:40px auto}pre{white-space:pre-wrap}</style>'
        '<h1>A/B/C/D: BC → RL</h1><p>開発評価。共通の保存予測区間上で比較。Testでの選択なし。</p><pre>'
        + html.escape(json.dumps(snapshot, indent=2, ensure_ascii=False)) + '</pre></html>')
    print(json.dumps(snapshot), flush=True)


def run_pool(cfg, registration, *, evaluation_phase=False):
    root = Path(cfg["output_dir"])
    tasks = [t for t in registration["jobs"] if not evaluation_phase or t["kind"] == "policy"]
    done = {t["job"] for t in tasks if _job_complete(root, t, evaluation_phase=evaluation_phase)}
    active, started = {}, time.monotonic()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    workers = int(cfg["runtime"]["workers"])
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(cfg["runtime"]["cpu_threads"]), MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    last_status = -np.inf
    try:
        while len(done) < len(tasks):
            for task in tasks:
                if len(active) >= workers:
                    break
                if task["job"] in done or task["job"] in active:
                    continue
                if not evaluation_phase and not all(dep in done for dep in task["dependencies"]):
                    continue
                suffix = "_test" if evaluation_phase else ""
                log = (logs / (task["job"] + suffix + ".log")).open("a")
                cmd = [sys.executable, "-m", "unidream.experiments.wm_policy_ablation_benchmark", "--config", cfg["config_path"],
                    "--action", "evaluate-worker" if evaluation_phase else "worker", "--job", task["job"]]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[task["job"]] = (proc, task, log)
            if not active and len(done) < len(tasks):
                raise RuntimeError("registered dependency graph cannot make progress")
            for key, (proc, task, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[key]
                if code != 0 or not _job_complete(root, task, evaluation_phase=evaluation_phase):
                    raise RuntimeError(f"worker {key} failed with exit {code}; see logs/{key}.log")
                done.add(key)
            if time.monotonic() - last_status >= cfg["runtime"]["status_seconds"]:
                _progress(root, "test_report_only" if evaluation_phase else "train_validation_only", tasks, done, active, started)
                last_status = time.monotonic()
            time.sleep(1)
    finally:
        for proc, task, log in active.values():
            if proc.poll() is None:
                proc.terminate()
            log.close()
    _progress(root, "test_predictions_complete" if evaluation_phase else "all_fits_frozen", tasks, done, {}, started)


def _verify_artifacts(root, fit):
    for filename, expected in fit.get("artifact_sha256", {}).items():
        if sha(root / filename) != expected:
            raise ValueError("a completed fit artifact changed before the global barrier")


def _control_choices(cfg):
    arrays = data.load_arrays(cfg)
    result = {}
    for fold in cfg["folds"]:
        fd = data.fold_arrays(cfg, fold)
        ids = fd["selection"]
        times = pd.to_datetime(np.asarray(arrays["index_ns"])[ids], utc=True)
        returns = np.asarray(arrays["bar_log_return"])[ids]
        constant = evaluation.select_constant_exposure(returns, times, cfg=cfg,
            candidates=cfg["baselines"]["constant_position_grid"])
        train_var = np.asarray(arrays["ewma_variance"])[fd["train"]]
        target = float(np.sqrt(np.median(train_var)))
        candidates = []
        for multiplier in cfg["baselines"]["volatility_target_multipliers"]:
            sigma = target * multiplier
            request = evaluation.volatility_target_requests(np.asarray(arrays["ewma_variance"])[ids], sigma, times, cfg=cfg)
            path = evaluation.execute_positions(returns, request, times, cfg=cfg)
            candidates.append({"target_sigma": sigma, "multiplier": multiplier,
                "score": evaluation.score_path(path, cfg)["reward_total"]})
        best = max(candidates, key=lambda c: c["score"])
        result[str(fold)] = {"constant": constant, "ewma": {**best, "candidates": candidates,
            "target_base_fit_split": "Train_only", "multiplier_selection_split": "Val_selection_only", "test_used": False}}
    return result


def freeze_selection(cfg, registration):
    root = Path(cfg["output_dir"])
    if (root / "test_barrier.json").exists():
        return verify_test_barrier(cfg)
    # Rehash the complete prepared cohort and immutable producer provenance
    # once at the barrier, rather than scanning GBs in every GPU worker.
    data._prior_cache(root / "prepared/cache_manifest.json", sha(root / "preregistration.json"))
    fits, normalizers, metadata, initial_states = {}, [], [], {}
    for task in registration["jobs"]:
        path = root / task["fit_path"]
        fit = json.loads(path.read_text())
        if not fit.get("complete") or fit.get("test_used") is not False or fit.get("registration_sha256") != sha(root / "preregistration.json"):
            raise ValueError("all registered fresh Train/Val fits must complete before Test")
        _verify_artifacts(path.parent, fit)
        fits[task["fit_path"]] = sha(path)
        if task["kind"] == "policy":
            fd = data.fold_arrays(cfg, task["fold"])
            expected = train_source_id(fd)
            normalizers.append({"fold": task["fold"], "seed": task["seed"], "arm": task["arm"], "fit_split": "train",
                "source_id": fit["normalizer_source_id"], "expected_source_id": expected})
            metadata.append(fit["metadata"])
            key = (task["fold"], task["seed"])
            initial_states.setdefault(key, set()).add(fit["initial_actor_state_sha256"])
    if len({m["total_parameters"] for m in metadata}) != 1 or any(len(values) != 1 for values in initial_states.values()):
        raise ValueError("actor capacity and paired initial weights must match across A/B/C/D")
    provenance = {"normalizers": normalizers, "cache_manifest_sha256": sha(root / "prepared/cache_manifest.json")}
    from .wm_policy_ablation_report import verify_normalizers
    verify_normalizers(provenance, cfg)
    write_json(root / "normalizer_provenance.json", provenance)
    choices = {"complete": True, "test_used": False, "registration_sha256": sha(root / "preregistration.json"),
        "fit_sha256": fits, "controls": _control_choices(cfg), "selection_split": "Val_only",
        "policy_checkpoint_selection_metric": "sum_relative_log_reward_minus_incremental_episode_maxDD_penalty",
        "at": now()}
    write_json(root / "validation_choices.json", choices)
    barrier = {"complete": True, "registration_sha256": sha(root / "preregistration.json"),
        "validation_choices_sha256": sha(root / "validation_choices.json"), "frozen_fit_sha256": fits,
        "normalizer_provenance_sha256": sha(root / "normalizer_provenance.json"),
        "cache_manifest_sha256": sha(root / "prepared/cache_manifest.json"),
        "test_selection_allowed": False, "test_used": False, "at": now()}
    write_json(root / "test_barrier.json", barrier)
    return barrier


def baseline_ledger(cfg, choices):
    arrays, rows = data.load_arrays(cfg), []
    for fold in cfg["folds"]:
        ids = data.fold_arrays(cfg, fold)["test"]
        times = pd.to_datetime(np.asarray(arrays["index_ns"])[ids], utc=True)
        control = choices["controls"][str(fold)]
        for name in ("val_constant", "ewma_voltarget"):
            if name == "val_constant":
                request = evaluation.bounded_requests(np.full(len(ids), control["constant"]["position"]), times, cfg=cfg)
            else:
                request = evaluation.volatility_target_requests(np.asarray(arrays["ewma_variance"])[ids], control["ewma"]["target_sigma"], times, cfg=cfg)
            rows.append(pd.DataFrame({"baseline": name, "fold": fold, "decision_id": np.asarray(arrays["decision_id"])[ids],
                "timestamp": times, "bar_log_return": np.asarray(arrays["bar_log_return"])[ids], "requested_position": request}))
    return pd.concat(rows, ignore_index=True)


def finish(cfg, registration):
    root = Path(cfg["output_dir"])
    verify(cfg, cache=True)
    freeze_selection(cfg, registration)
    frames = [pd.read_parquet(root / "evaluations" / Path(t["fit_path"]).parent.name / "predictions.parquet")
        for t in registration["jobs"] if t["kind"] == "policy"]
    final = root / "final"
    final.mkdir(exist_ok=True)
    ledger = pd.concat(frames, ignore_index=True)
    ledger.to_parquet(final / "policy_ledger.parquet", index=False)
    choices = json.loads((root / "validation_choices.json").read_text())
    controls = baseline_ledger(cfg, choices)
    controls.to_parquet(final / "baseline_ledger.parquet", index=False)
    from .wm_policy_ablation_report import write_policy_report
    report = write_policy_report(root, cfg, ledger, include_test=True, baseline_ledger=controls)
    write_json(root / "completion.json", {"complete": True, "at": now(), "report": report,
        "registration_sha256": sha(root / "preregistration.json"), "test_barrier_sha256": sha(root / "test_barrier.json"),
        "source_sha256": source_hashes(cfg), "test_selection_allowed": False,
        "policy_pipeline_fits": len(frames), "policy_fold_stage_results": len(frames) * 2,
        "independently_unused_holdout": False, "conditional_cohort_only": True})
    write_json(root / "status.json", {"phase": "complete", "at": now(), "report": str(final / "index.html")})
    return report


def run(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "locks/coordinator.lock"):
        registration = prepare(cfg)
        run_pool(cfg, registration)
        freeze_selection(cfg, registration)
        run_pool(cfg, registration, evaluation_phase=True)
        return finish(cfg, registration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--action", choices=("prepare", "run", "worker", "evaluate-worker", "status"), default="run")
    parser.add_argument("--job")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == "prepare":
        prepare(cfg)
    elif args.action == "run":
        run(cfg)
    elif args.action == "status":
        print((Path(cfg["output_dir"]) / "status.json").read_text())
    else:
        registration = verify(cfg, cache=True)
        matches = [t for t in registration["jobs"] if t["job"] == args.job]
        if len(matches) != 1 or (args.action == "evaluate-worker" and matches[0]["kind"] != "policy"):
            raise ValueError("--job must identify exactly one registered compatible task")
        _worker(cfg, matches[0], args.action == "evaluate-worker")


if __name__ == "__main__":
    main()
