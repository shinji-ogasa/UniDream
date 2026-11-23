"""Separately registered LSTM inference extension, preserving both studies.

LSTM adds264 hypotheses to the original1056 and112 primary heads to448.
New420-cell artifacts contain Holm1320/560; original P-value files stay fixed.
Registration must precede the LSTM campaign's global Test barrier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from . import wm_lstm_grid_benchmark as lstm
from . import wm_significance_benchmark as original
from .wm_significance_benchmark import ROOT, atomic_json, now, _sha
from .wm_significance_jobs import calculate_job_records
from .wm_significance_report import TASKS, FORMULATIONS, _match_frozen_point
from .wm_significance_statistics import holm_fixed_family


DEFAULT_CONFIG = ROOT / "configs/experiments/wm_lstm_significance_20261112.yaml"
FAMILY_COUNTS = {"primary": 112, "ba_chance": 84, "variance_BA": 36, "variance_QLIKE": 16, "crash_Brier": 16}
ALL_MODELS = ("linear", "lightgbm", "direct", "wm", "lstm")
NATIVE_PRIMARY = original.NATIVE_PRIMARY
KEYS = ("group", "model", "task", "formulation")


class DependencyFailure(RuntimeError):
    """An unfinished prerequisite has an explicit failure marker."""


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    settings = cfg.get("statistics", {})
    if (cfg.get("groups") != list("ABCD") or cfg.get("models") != ["lstm"]
            or cfg.get("tasks") != list(TASKS) or cfg.get("formulations") != list(FORMULATIONS)
            or cfg.get("folds") != list(range(15)) or cfg.get("seeds") != [7] or cfg.get("horizons") != [16]
            or cfg.get("test_selection_allowed") is not False or cfg.get("historical_seen_test") is not True
            or cfg.get("independently_unused_holdout") is not False
            or settings.get("bootstrap_replicates") != 100000 or settings.get("sensitivity_replicates") != 10000
            or settings.get("ic_replicates") != 10000 or settings.get("block_days") != [1, 7, 28]
            or settings.get("primary_block_days") != 7 or settings.get("combined_primary_family_size") != 560
            or settings.get("combined_global_family_size") != 1320 or settings.get("new_primary_family_size") != 112
            or settings.get("new_global_family_size") != 264):
        raise ValueError("fixed84-cell LSTM extension,100000-replicate inference and560/1320 families required")
    if (cfg.get("runtime", {}).get("workers", 0) < 1 or cfg.get("runtime", {}).get("cpu_threads", 0) < 1
            or not 1 <= cfg.get("runtime", {}).get("poll_seconds", 0) <= 60):
        raise ValueError("positive worker count and1-60 second polling required")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "significance_config", "lstm_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in sorted(Path(__file__).parent.glob("wm_lstm_significance*.py"))}


def make_jobs(cfg=None):
    from .wm_significance_jobs import make_jobs as parent_jobs
    jobs = []
    for source in parent_jobs():
        if source["model"] != "direct":
            continue
        task = {**source, "model": "lstm"}
        task["job_id"] = source["job_id"].replace("_direct_", "_lstm_", 1)
        jobs.append(task)
    if len(jobs) != 200 or len({task["job_id"] for task in jobs}) != 200:
        raise ValueError("exactly200 unique LSTM inference jobs required")
    return jobs


def _parents(cfg):
    scfg = original.load_config(cfg["significance_config"])
    sreg = original.verify(scfg)
    lcfg = lstm.load_config(cfg["lstm_config"])
    lreg, _, _ = lstm.verify(lcfg)
    return scfg, sreg, lcfg, lreg


def prepare(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "preregistration.json").exists():
        raise ValueError("LSTM inference extension registration is immutable")
    scfg, sreg, lcfg, lreg = _parents(cfg)
    if (Path(lcfg["output_dir"]) / "test_barrier.json").exists():
        raise ValueError("register the extension before the LSTM Test barrier opens")
    if (sreg.get("global_family_size") != 1056 or sreg.get("primary_family_size") != 448
            or lreg.get("grid", {}).get("cells") != 84):
        raise ValueError("the frozen1056-hypothesis study and84-cell LSTM campaign are required")
    registration = {"experiment_id": cfg["experiment_id"], "registered_utc": now(), "config": cfg,
        "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(),
        "significance_registration_sha256": _sha(Path(scfg["output_dir"]) / "preregistration.json"),
        "lstm_registration_sha256": _sha(Path(lcfg["output_dir"]) / "preregistration.json"),
        "rank_backend": sreg["rank_backend"], "jobs": make_jobs(cfg),
        "registered_before_lstm_test": True, "lstm_test_barrier_present_at_registration": False,
        "new_cells": 84, "combined_cells": 420, "new_family_counts": FAMILY_COUNTS,
        "new_primary_hypotheses": 112, "new_global_hypotheses": 264,
        "combined_primary_hypotheses": 560, "combined_global_hypotheses": 1320,
        "protocol": "same frozen causal baseline cache, Train labels, all124488 decisions and paired7-day100000-replicate inference;1/28day10000, absoluteIC10000 exact reranking",
        "baseline_policy": "reuse the original registered Train-fitted baseline without any refitting",
        "multiplicity": "new artifacts re-adjust all original1056 plus264 LSTM hypotheses with Holm1320; primary448+112 with Holm560; controls retained conservatively",
        "old_artifacts_modified": False, "test_selection_allowed": False, "historical_seen_test": True,
        "independently_unused_holdout": False, "single_training_seed": True,
        "interpretation": "exploratory expanded family; baseline comparisons do not test LSTM-versus-other-model superiority",
        "old_historical_comparisons": "original separate112+112 formulation families remain separate and are not added to1320"}
    atomic_json(root / "preregistration.json", registration)
    status(cfg, "registered_waiting_for_dependencies")
    return registration


def verify(cfg):
    root = Path(cfg["output_dir"])
    reg = json.loads((root / "preregistration.json").read_text())
    if reg["config_sha256"] != _sha(cfg["config_path"]) or reg["source_sha256"] != source_hashes():
        raise ValueError("registered LSTM extension source/config changed")
    if reg.get("jobs") != make_jobs(cfg):
        raise ValueError("the complete200-job extension grid must remain frozen")
    scfg, _, lcfg, _ = _parents(cfg)
    if (reg["significance_registration_sha256"] != _sha(Path(scfg["output_dir"]) / "preregistration.json")
            or reg["lstm_registration_sha256"] != _sha(Path(lcfg["output_dir"]) / "preregistration.json")):
        raise ValueError("frozen parent registration changed")
    barrier = Path(lcfg["output_dir"]) / "test_barrier.json"
    if barrier.exists():
        opened = json.loads(barrier.read_text())["opened_utc"]
        if pd.Timestamp(opened) < pd.Timestamp(reg["registered_utc"]):
            raise ValueError("extension was not registered before LSTM Test access")
    return reg


def dependencies_ready(cfg, reg, verify_artifacts=False):
    scfg = original.load_config(cfg["significance_config"])
    lcfg = lstm.load_config(cfg["lstm_config"])
    roots = [Path(scfg["output_dir"]), Path(lcfg["output_dir"])]
    expected = [reg["significance_registration_sha256"], reg["lstm_registration_sha256"]]
    ready = True
    for root, registration_sha in zip(roots, expected):
        path = root / "completion.json"
        complete = json.loads(path.read_text()) if path.exists() else {}
        if not complete.get("complete") or complete.get("status") != "complete":
            failure = root / "failure.json"
            if failure.exists():
                detail = json.loads(failure.read_text()).get("error", "failure marker exists")
                raise DependencyFailure("unfinished prerequisite failed: " + str(failure) + ": " + str(detail))
            ready = False
            continue
        if complete.get("registration_sha256") != registration_sha:
            raise ValueError("completed dependency belongs to a different registration")
        if verify_artifacts:
            for relative, checksum in complete["artifact_sha256"].items():
                if _sha(root / relative) != checksum:
                    raise ValueError("completed prerequisite artifact changed")
    return ready


def status(cfg, stage=None):
    root = Path(cfg["output_dir"])
    old = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    done = sum((root / "jobs" / (task["job_id"] + ".json")).exists() for task in make_jobs(cfg))
    data = {"utc": now(), "stage": stage or old.get("stage", "unregistered"), "jobs": {"complete": done, "total": 200},
        "new_cells": 84, "combined_cells": 420, "primary_hypotheses": 560, "global_hypotheses": 1320}
    atomic_json(root / "status.json", data)
    (root / "progress.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="20">'
        '<title>LSTM追加・420条件の統計検証</title><style>body{font-family:system-ui;margin:30px;line-height:1.7}</style>'
        '<h1>LSTM追加・420条件の統計検証</h1>' + f'<p>{data["stage"]}: {done}/200 推論ジョブ</p>'
        '<p>LSTM84条件の完成と元の336条件統計検証を待って、同じ基準と時点で追加検証します。'
        '旧結果を書き換えず、新しい420条件の表で主560比較・全1320比較の補正を出します。</p>'
        '<p><a href="final/index.html">完成レポート</a></p></html>', encoding="utf-8")
    return data


def _cached_manifest(cfg, reg):
    root = Path(cfg["output_dir"])
    path = root / "prepared/cache_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if (manifest.get("complete") is not True or manifest.get("registration_sha256") != _sha(root / "preregistration.json")
            or len(manifest.get("panels", {})) != 12):
        raise ValueError("incomplete or stale LSTM inference cache")
    expected = {manifest["baseline_path"]: manifest["baseline_sha256"], **manifest["source_artifact_sha256"]}
    expected.update({panel["path"]: panel["sha256"] for panel in manifest["panels"].values()})
    for filename, checksum in expected.items():
        if _sha(filename) != checksum:
            raise ValueError("immutable LSTM inference cache/source changed")
    return manifest


def build_cache(cfg, reg):
    from .wm_significance_cache import (_verified_prediction, _ordered, _assert_outcomes,
        panel_arrays, verify_balanced_accuracies, atomic_npz)
    root = Path(cfg["output_dir"])
    if not (root / "preregistration.json").exists() or json.loads((root / "preregistration.json").read_text()) != reg:
        raise ValueError("the addon registration must precede all new Test panel reads")
    if not dependencies_ready(cfg, reg, verify_artifacts=True):
        raise ValueError("both frozen prerequisite studies must be complete before Test panel reads")
    previous = _cached_manifest(cfg, reg)
    if previous is not None:
        return previous
    scfg = original.load_config(cfg["significance_config"])
    lcfg = lstm.load_config(cfg["lstm_config"])
    sroot, lroot = Path(scfg["output_dir"]), Path(lcfg["output_dir"])
    parent_manifest_path = sroot / "prepared/cache_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_statistics = json.loads((sroot / "statistics_completion.json").read_text())
    if (parent_manifest.get("registration_sha256") != reg["significance_registration_sha256"]
            or parent_manifest["baseline_sha256"] != _sha(parent_manifest["baseline_path"])
            or parent_statistics["cache_manifest_sha256"] != _sha(parent_manifest_path)):
        raise ValueError("the original frozen causal baseline cache changed")
    with np.load(parent_manifest["baseline_path"], allow_pickle=False) as source:
        baselines = {key: source[key] for key in source.files}
    n = len(baselines["fold"])
    if n != 124488 or set(baselines["fold"]) != set(range(15)):
        raise ValueError("the identical124488 decisions and15 folds are required")
    lreg = json.loads((lroot / "preregistration.json").read_text())
    folds = {int(row["fold"]): row for row in lreg["folds"]}
    points_path = lroot / "final/results_84.csv"
    points = pd.read_csv(points_path)
    if len(points) != 84 or points.duplicated(list(KEYS)).any():
        raise ValueError("the complete frozen84-cell LSTM table is required")
    provenance = {str(parent_manifest_path): _sha(parent_manifest_path), str(points_path): _sha(points_path),
        str(lroot / "final/results_420.csv"): _sha(lroot / "final/results_420.csv"),
        str(sroot / "final/all_inference.csv"): _sha(sroot / "final/all_inference.csv"),
        str(sroot / "completion.json"): _sha(sroot / "completion.json"),
        str(lroot / "completion.json"): _sha(lroot / "completion.json")}
    panels = {}
    for group in "ABCD":
        for form in FORMULATIONS:
            parts = []
            for fold in range(15):
                chosen = np.flatnonzero(baselines["fold"] == fold)
                task = next(task for task in lreg["neural_tasks"] if task["fold"] == fold and task["group"] == group and task["formulation"] == form)
                path, _ = _verified_prediction(lroot, "neural", task["job"], provenance)
                frame = _ordered(pd.read_parquet(path), {"fold": fold, "group": group, "model": "lstm", "formulation": form, "seed": 7, "split": "test"},
                    baselines["decision_id"][chosen], baselines["index_ns"][chosen])
                _assert_outcomes(frame, baselines["actual"][chosen], baselines["labels"][chosen])
                parts.append(panel_arrays(frame, folds[fold]["label_contract"], form))
            panel = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            verify_balanced_accuracies(panel, baselines["labels"], points, group, "lstm", form)
            panel_id = f"{group}_lstm_{form}"
            path = root / "prepared/panels" / (panel_id + ".npz")
            atomic_npz(path, **panel)
            panels[panel_id] = {"path": str(path.resolve()), "sha256": _sha(path), "group": group, "model": "lstm", "formulation": form}
    manifest = {"complete": True, "registration_sha256": _sha(root / "preregistration.json"),
        "baseline_path": parent_manifest["baseline_path"], "baseline_sha256": parent_manifest["baseline_sha256"],
        "panels": panels, "n_rows": n, "n_folds": 15, "source_artifact_sha256": provenance,
        "baseline_refitted": False, "new_model_training": False}
    atomic_json(root / "prepared/cache_manifest.json", manifest)
    return manifest


def _job_payload(cfg, task, required=False):
    root = Path(cfg["output_dir"])
    path = root / "jobs" / (task["job_id"] + ".json")
    if not path.exists():
        if required:
            raise ValueError("missing LSTM inference job")
        return None
    payload = json.loads(path.read_text())
    if (payload.get("complete") is not True or payload.get("task") != task
            or payload.get("registration_sha256") != _sha(root / "preregistration.json")
            or payload.get("cache_manifest_sha256") != _sha(root / "prepared/cache_manifest.json")
            or not payload.get("records")):
        raise ValueError("stale or incomplete LSTM inference job")
    return payload


def run_job(config_path, task):
    from threadpoolctl import threadpool_limits
    cfg = load_config(config_path)
    reg = verify(cfg)
    if task not in reg["jobs"]:
        raise ValueError("inference task was not registered in the fixed200-job grid")
    root = Path(cfg["output_dir"])
    dest = root / "jobs" / (task["job_id"] + ".json")
    with lstm.claim(dest.with_suffix(".lock")):
        manifest = json.loads((root / "prepared/cache_manifest.json").read_text())
        if (manifest.get("complete") is not True
                or manifest.get("registration_sha256") != _sha(root / "preregistration.json")):
            raise ValueError("prediction cache belongs to a different addon registration")
        panel = manifest["panels"][f"{task['group']}_lstm_{task['formulation']}"]
        for path, checksum in ((manifest["baseline_path"], manifest["baseline_sha256"]), (panel["path"], panel["sha256"])):
            if _sha(path) != checksum:
                raise ValueError("immutable baseline or LSTM prediction cache changed")
        previous = _job_payload(cfg, task)
        if previous is not None:
            return task
        with np.load(manifest["baseline_path"], allow_pickle=False) as source:
            baselines = {key: source[key] for key in source.files}
        with np.load(panel["path"], allow_pickle=False) as source:
            predictions = {key: source[key] for key in source.files}
        cfg["rank_backend_path"] = reg["rank_backend"]["path"]
        with threadpool_limits(cfg["runtime"]["cpu_threads"]):
            records = calculate_job_records(cfg, task, baselines, predictions)
        atomic_json(dest, {"complete": True, "task": task, "records": records,
            "registration_sha256": _sha(root / "preregistration.json"),
            "cache_manifest_sha256": _sha(root / "prepared/cache_manifest.json")})
    return task


def expected_slots(models=("lstm",)):
    slots = set()
    for group in "ABCD":
        for model in models:
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


def adjust_expanded(records):
    selected = [row for row in records if int(row["block_days"]) == 7 and row["family"] in FAMILY_COUNTS]
    def slot(row):
        return tuple(row[key] for key in (*KEYS, "head", "family", "metric")) + (
            row["baseline"] if row["family"] in ("variance_BA", "variance_QLIKE") else "",)
    if len(selected) != 1320 or len({row["hypothesis_id"] for row in selected}) != 1320 or {slot(row) for row in selected} != expected_slots(ALL_MODELS):
        raise ValueError("exactly1320 unique registered expanded hypotheses required")
    primary = [row for row in selected if row["family"] == "primary"]
    if len(primary) != 560:
        raise ValueError("exactly560 expanded primary hypotheses required")
    def pvalues(rows):
        result = {}
        for row in rows:
            p = row.get("p_value")
            # CSV-origin unavailable P values may arrive as NaN.
            if p is None or (isinstance(p, float) and np.isnan(p)):
                p = 1.
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("raw P values must be unavailable or probabilities")
            result[row["hypothesis_id"]] = float(p)
        return result
    primary_p = holm_fixed_family(pvalues(primary), family_size=560)
    global_p = holm_fixed_family(pvalues(selected), family_size=1320)
    output = []
    for source in records:
        row = dict(source)
        if source["model"] != "lstm":
            row["historical_p_holm_primary448"] = source.get("p_holm_primary448")
            row["historical_p_holm_global1056"] = source.get("p_holm_global")
        eligible = int(row["block_days"]) == 7 and row["family"] in FAMILY_COUNTS
        row["p_holm_primary560"] = primary_p.get(row["hypothesis_id"]) if eligible and row["family"] == "primary" else None
        row["p_holm_global1320"] = global_p.get(row["hypothesis_id"]) if eligible else None
        improvement = row.get("improvement")
        row["significant_improvement_global1320"] = bool(row["p_holm_global1320"] is not None and row["p_holm_global1320"] < .05
            and improvement is not None and np.isfinite(improvement) and improvement > 0)
        output.append(row)
    return output


def write_report(root, points_path, frame):
    root, points_path = Path(root), Path(points_path)
    points = pd.read_csv(points_path)
    expected = {(g, m, task, form) for g in "ABCD" for m in ALL_MODELS for task in TASKS for form in FORMULATIONS}
    if len(points) != 420 or points.duplicated(list(KEYS)).any() or set(points[list(KEYS)].itertuples(index=False, name=None)) != expected:
        raise ValueError("exactly420 frozen result cells required")
    selected = frame.loc[frame.block_days.eq(7)]
    rows = []
    for source in points.to_dict("records"):
        row = dict(source)
        panel = selected
        for key in KEYS:
            panel = panel.loc[panel[key].eq(source[key])]
        heads = ("regression", "classification") if source["formulation"] == "joint" else (source["formulation"],)
        for head in heads:
            current = panel.loc[panel.family.eq("primary") & panel["head"].eq(head)]
            if len(current) != 1:
                raise ValueError("every420 cell must retain its complete native primary heads")
            result = current.iloc[0]
            _match_frozen_point(source, result, head)
            prefix = "scalar_primary" if head == "regression" else "class_BA"
            _copy_report_fields(row, prefix, result)
        if source["formulation"] == "regression":
            proxy = panel.loc[panel.family.eq("ba_chance")]
            if len(proxy) != 1 or not np.isclose(proxy.estimate.iloc[0], source["balanced_accuracy"], rtol=1e-8, atol=1e-12):
                raise ValueError("scalar BA proxy must reproduce the frozen point table")
            _copy_report_fields(row, "class_BA", proxy.iloc[0])
            row["class_BA_is_auxiliary_scalar_proxy"] = True
        for metric in ("pearson_ic", "spearman_ic"):
            current = panel.loc[panel["head"].eq("regression") & panel.metric.eq(metric)]
            if len(current) > 1:
                current = current.loc[current.family.eq("primary")]
            if len(current) == 1:
                for field in ("estimate", "ci_low", "ci_high"):
                    row["native_" + metric + "_" + field] = current.iloc[0][field]
        row.update(historical_test_exploratory=True, independently_unused_holdout=False,
            baseline_comparison_is_not_pairwise_architecture_test=True)
        rows.append(row)
    table = pd.DataFrame(rows)
    final = root / "final"
    table.to_csv(final / "results_420_significance.csv", index=False)
    lstm_only = table.loc[table.model.eq("lstm")]
    lstm_only.to_csv(final / "results_84_significance.csv", index=False)
    columns = [*KEYS, "scalar_primary_metric", "scalar_primary_estimate", "scalar_primary_ci_low", "scalar_primary_ci_high",
        "scalar_primary_baseline", "scalar_primary_improvement", "scalar_primary_improvement_ci_low", "scalar_primary_improvement_ci_high",
        "scalar_primary_p_holm_primary560", "scalar_primary_p_holm_global1320", "class_BA_estimate", "class_BA_ci_low", "class_BA_ci_high",
        "class_BA_baseline", "class_BA_improvement", "class_BA_improvement_ci_low", "class_BA_improvement_ci_high", "class_BA_p_holm_primary560", "class_BA_p_holm_global1320"]
    (final / "index.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>420条件・LSTM追加統計検証</title>'
        '<style>body{font-family:system-ui;margin:24px;line-height:1.7}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:5px}.table{overflow:auto;max-height:75vh}</style>'
        '<h1>420条件・LSTM追加の統計検証</h1><p>4入力×5モデル×7対象×3方式。旧336条件の結果を保持し、LSTM84条件を追加しました。'
        '同じ124,488時点とTrainだけで学習済みの基準モデルを再利用し、主560比較・全1320比較を新しい表でHolm補正しています。旧検定ファイルは変更していません。</p>'
        '<p>7日ブロック10万回、1日・28日の感度確認。95%区間は各比較の区間で、同時信頼区間ではありません。'
        '優位性は正の改善かつ補正後P&lt;0.05で判定します。これは単純ベースラインとの差であり、LSTMとWM・LightGBMの直接の優位性検定ではありません。</p>'
        '<p>過去に確認したTestの探索的解析です。新しい未使用Holdoutではありません。学習seedは1つ。'
        'LSTMの追加解析契約はLSTMのTestを開く前に固定しました。線形・LightGBMのjoint56条件は既存独立headの対照です。</p>'
        '<p><a href="results_420_significance.csv">全420条件CSV</a> / <a href="results_84_significance.csv">LSTM84条件CSV</a> / '
        '<a href="all_inference.csv">全CI・検定CSV</a> / <a href="report_contract.json">入力hashと集計契約</a></p>'
        '<div class="table">' + table.reindex(columns=columns).to_html(index=False, escape=True) + '</div>', encoding="utf-8")
    contract = {"complete": True, "cells": 420, "lstm_cells": 84, "primary_tests": 560, "global_tests": 1320,
        "source_points_path": str(points_path.resolve()), "source_points_sha256": _sha(points_path),
        "all_inference_sha256": _sha(final / "all_inference.csv"), "report_source_sha256": _sha(Path(__file__)),
        "historical_seen_test": True, "independently_unused_holdout": False, "old_results_modified": False,
        "baseline_comparisons_do_not_test_pairwise_architecture_superiority": True,
        "baseline_refitted": False, "individual_not_simultaneous_confidence_intervals": True}
    atomic_json(final / "report_contract.json", contract)
    atomic_json(final / "report_completion.json", {"complete": True, "report_complete": True, **contract})
    return contract


def _copy_report_fields(row, prefix, result):
    for name in ("metric", "estimate", "ci_low", "ci_high", "baseline", "baseline_estimate", "improvement",
                 "improvement_ci_low", "improvement_ci_high", "p_value", "p_holm_primary560", "p_holm_global1320",
                 "historical_p_holm_primary448", "historical_p_holm_global1056", "inference_available", "inference_unavailable_reason"):
        row[prefix + "_" + name] = result.get(name)
    p = result.get("p_holm_global1320")
    delta = result.get("improvement")
    row[prefix + "_significant_improvement_global1320"] = bool(p is not None and np.isfinite(p) and p < .05
        and delta is not None and np.isfinite(delta) and delta > 0)


def finalize(cfg, reg):
    root = Path(cfg["output_dir"])
    _cached_manifest(cfg, reg)
    scfg = original.load_config(cfg["significance_config"])
    lcfg = lstm.load_config(cfg["lstm_config"])
    original_path = Path(scfg["output_dir"]) / "final/all_inference.csv"
    records = original._clean(pd.read_csv(original_path, float_precision="round_trip").to_dict("records"))
    addon = []
    hashes = {}
    for task in reg["jobs"]:
        addon.extend(_job_payload(cfg, task, required=True)["records"])
        hashes[task["job_id"]] = _sha(root / "jobs" / (task["job_id"] + ".json"))
    selected = [row for row in addon if int(row["block_days"]) == 7 and row["family"] in FAMILY_COUNTS]
    if len(selected) != 264 or {row["model"] for row in selected} != {"lstm"}:
        raise ValueError("all264 new LSTM hypotheses are required")
    records = adjust_expanded([*records, *addon])
    final = root / "final"
    final.mkdir(exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(final / "all_inference.csv", index=False)
    pd.DataFrame(addon).to_csv(final / "lstm_inference.csv", index=False)
    report = write_report(root, Path(lcfg["output_dir"]) / "final/results_420.csv", frame)
    paths = ("final/all_inference.csv", "final/lstm_inference.csv", "final/results_420_significance.csv", "final/results_84_significance.csv",
        "final/index.html", "final/report_contract.json", "final/report_completion.json")
    complete = {"complete": True, "status": "complete", "utc": now(), "cells": 420, "new_cells": 84,
        "registration_sha256": _sha(root / "preregistration.json"), "primary_hypotheses": 560, "global_hypotheses": 1320,
        "original_inference_sha256": _sha(original_path), "cache_manifest_sha256": _sha(root / "prepared/cache_manifest.json"),
        "job_sha256": hashes, "artifact_sha256": {path: _sha(root / path) for path in paths}, "report": report}
    atomic_json(root / "completion.json", complete)
    status(cfg, "complete")
    return complete


def campaign(cfg):
    root = Path(cfg["output_dir"])
    with lstm.claim(root / "campaign.lock"):
        try:
            reg = verify(cfg)
            while not dependencies_ready(cfg, reg):
                status(cfg, "waiting_for_lstm_and_original_significance")
                time.sleep(cfg["runtime"]["poll_seconds"])
                verify(cfg)
            dependencies_ready(cfg, reg, verify_artifacts=True)
            if (root / "completion.json").exists():
                complete = json.loads((root / "completion.json").read_text())
                if complete.get("registration_sha256") != _sha(root / "preregistration.json"):
                    raise ValueError("stale LSTM inference completion")
                for relative, checksum in complete["artifact_sha256"].items():
                    if _sha(root / relative) != checksum:
                        raise ValueError("completed LSTM inference output changed")
                status(cfg, "complete")
                return complete
            status(cfg, "preparing_matched_predictions")
            build_cache(cfg, reg)
            pending = [task for task in reg["jobs"] if _job_payload(cfg, task) is None]
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(cfg["runtime"]["workers"], mp_context=context, max_tasks_per_child=1) as pool:
                futures = {pool.submit(run_job, cfg["config_path"], task): task for task in pending}
                while futures:
                    completed, _ = wait(futures, timeout=cfg["runtime"]["poll_seconds"], return_when=FIRST_COMPLETED)
                    for future in completed:
                        task = futures.pop(future)
                        future.result()
                        _job_payload(cfg, task, required=True)
                    status(cfg, "block_bootstrap")
            status(cfg, "expanded_holm_adjustment")
            return finalize(cfg, reg)
        except Exception as error:
            atomic_json(root / "failure.json", {"utc": now(), "error": repr(error), "traceback": traceback.format_exc()})
            status(cfg, "blocked_dependency_or_failed" if isinstance(error, DependencyFailure) else "failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "campaign", "status"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = {"prepare": prepare, "verify": verify, "campaign": campaign, "status": status}[args.command](cfg)
    print(json.dumps(original._clean(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
