"""Hash-checked, common-cohort inputs for the extra significance analysis.

This module never alters registered forecasts.  Its entrypoint requires the
new analysis registration to exist before reading historical Test outcomes.
All original fit barriers, evaluation manifests, row identities, labels, and
the 336 reported balanced accuracies are checked before the cache completes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from . import wm_full_grid_benchmark as full_grid
from . import wm_joint_grid_benchmark as joint_grid
from . import wm_seven_task_benchmark as seven_task
from .wm_full_grid_evaluation import make_task_labels
from .wm_full_grid_report import MODELS
from .wm_seven_task_evaluation import TASKS
from .wm_significance_baselines import fit_baselines, predict_baselines


FORMULATIONS = ("regression", "classification", "joint")
FILTERS = ("past_variance", "ewma", "garch")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    """Write an ordinary uncompressed NPZ atomically, with no pickle arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("cache arrays must not require pickle loading")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def require_registration(output_dir, registration):
    path = Path(output_dir) / "preregistration.json"
    if not path.exists():
        raise ValueError("new significance registration must exist before any historical Test reads")
    if json.loads(path.read_text()) != registration:
        raise ValueError("supplied significance registration differs from the immutable disk registration")
    return _sha(path)


def _verified_prediction(root, kind, job, provenance, *, load_model=False):
    """Verify a source fit and its evaluated predictions against its barrier."""
    root = Path(root)
    directory = root / kind / job
    barrier_path, reg_path = root / "test_barrier.json", root / "preregistration.json"
    barrier, fit = json.loads(barrier_path.read_text()), json.loads((directory / "fit.json").read_text())
    reg_sha, fit_sha = _sha(reg_path), _sha(directory / "fit.json")
    if barrier.get("registration_sha256") != reg_sha or fit.get("registration_sha256") != reg_sha:
        raise ValueError("source registration changed after its Test barrier")
    relative = str((directory / "fit.json").relative_to(root))
    if barrier.get("frozen_fit_sha256", {}).get(relative) != fit_sha:
        raise ValueError("source fit differs from its frozen Test barrier")
    evaluation_path, prediction_path = directory / "evaluation.json", directory / "test_predictions.parquet"
    evaluation = json.loads(evaluation_path.read_text())
    barrier_sha, prediction_sha = _sha(barrier_path), _sha(prediction_path)
    if evaluation.get("barrier_sha256") != barrier_sha or evaluation.get("prediction_sha256") != prediction_sha:
        raise ValueError("source Test predictions or their evaluation barrier changed")
    provenance.update({str(reg_path.resolve()): reg_sha, str(barrier_path.resolve()): barrier_sha,
        str((directory / "fit.json").resolve()): fit_sha, str(evaluation_path.resolve()): _sha(evaluation_path),
        str(prediction_path.resolve()): prediction_sha})
    if load_model:
        model_path = directory / "models.pkl"
        model_sha = _sha(model_path)
        if fit.get("model_sha256") != model_sha or evaluation.get("selected_model_sha256") != model_sha:
            raise ValueError("frozen variance-filter model bundle changed")
        provenance[str(model_path.resolve())] = model_sha
    return prediction_path, directory


def _ordered(frame, expected, ids, index_ns):
    selected = frame
    for key, value in expected.items():
        if key not in selected:
            raise ValueError(f"source prediction metadata lacks {key}")
        selected = selected.loc[selected[key] == value]
    selected = selected.sort_values("decision_id").reset_index(drop=True)
    if selected.empty or selected.decision_id.duplicated().any():
        raise ValueError("source prediction cohort is missing or duplicated")
    if not np.array_equal(selected.decision_id.to_numpy(), ids):
        raise ValueError("source prediction decision ids differ from the complete common cohort")
    timestamps = pd.to_datetime(selected.timestamp, utc=True).astype("int64").to_numpy()
    if not np.array_equal(timestamps, index_ns):
        raise ValueError("source prediction timestamps differ from the complete common cohort")
    return selected


def panel_arrays(frame, contract, formulation):
    """Derive scalar heads, fixed-rule classes, and raw crash probabilities."""
    if formulation not in FORMULATIONS:
        raise ValueError("unknown formulation")
    n = len(frame)
    scalar = np.full((n, len(TASKS)), np.nan, dtype=np.float64)
    classes = np.empty((n, len(TASKS)), dtype=np.int64)
    crash_p = None
    for column, task in enumerate(TASKS):
        entry = contract[task]
        if formulation != "classification":
            point = frame["scalar_" + task].to_numpy(dtype=np.float64)
            if point.shape != (n,) or not np.isfinite(point).all():
                raise ValueError("native scalar forecasts must be finite on the complete cohort")
            scalar[:, column] = point
        if formulation == "regression" and task != "crash":
            classes[:, column] = make_task_labels(scalar[:, column], task, contract)
            continue
        if formulation == "regression":
            positive = scalar[:, column]
            probabilities = np.column_stack((1 - positive, positive))
        else:
            k = int(entry["n_classes"])
            probabilities = frame[[f"p_{task}_{i}" for i in range(k)]].to_numpy(dtype=np.float64)
        priors = np.asarray(entry["priors"], dtype=np.float64)
        if (probabilities.shape != (n, len(priors)) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
                or not np.isfinite(priors).all() or np.any(priors <= 0)):
            raise ValueError("source probabilities and Train priors must be finite, valid, and aligned")
        if task == "crash":
            crash_p = probabilities[:, 1].copy()
        normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
        classes[:, column] = np.argmax(normalized / priors, axis=1)
    if crash_p is None:
        raise ValueError("complete source panel must include crash probabilities")
    return {"scalar": scalar, "classes": classes, "crash_p": crash_p}


def _assert_outcomes(frame, actual, labels):
    for column, task in enumerate(TASKS):
        if not np.array_equal(frame["actual_" + task].to_numpy(dtype=np.float64), actual[:, column]):
            raise ValueError("source outcome arrays differ from the complete common cohort")
        if not np.array_equal(frame["label_" + task].to_numpy(dtype=np.int64), labels[:, column]):
            raise ValueError("source frozen class labels differ from the Train-only contract")


def verify_balanced_accuracies(panel, actual_labels, points, group, model, formulation):
    for column, task in enumerate(TASKS):
        k = 2 if task == "crash" else 3
        observed = actual_labels[:, column]
        recalls = [np.mean(panel["classes"][observed == c, column] == c) for c in range(k)]
        ba = float(np.mean(recalls))
        row = points.loc[(points.group == group) & (points.model == model)
            & (points.formulation == formulation) & (points.task == task)]
        if len(row) != 1 or not np.isfinite(ba) or not np.isclose(ba, float(row.balanced_accuracy.iloc[0]), rtol=1e-10, atol=1e-12):
            raise ValueError(f"cached BA differs from the frozen336 table: {group}/{model}/{formulation}/{task}")


def _prior_cache(path, reg_sha):
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if not manifest.get("complete") or manifest.get("registration_sha256") != reg_sha:
        raise ValueError("existing cache belongs to an incomplete or different registration")
    expected = {manifest["baseline_path"]: manifest["baseline_sha256"]}
    expected.update({entry["path"]: entry["sha256"] for entry in manifest["panels"].values()})
    for audit in manifest.get("baseline_fits", {}).values():
        expected[audit["bundle_path"]] = audit["bundle_sha256"]
        expected[audit["audit_path"]] = audit["audit_sha256"]
    expected.update(manifest["source_artifact_sha256"])
    for filename, checksum in expected.items():
        if _sha(filename) != checksum:
            raise ValueError("existing significance cache or its frozen source changed")
    return {**manifest, "cache_manifest_path": str(path.resolve()), "cache_manifest_sha": _sha(path)}


def build_cache(cfg, registration):
    """Build all48 matched model panels and fixed, Train-only comparators."""
    root = Path(cfg["output_dir"]).resolve()
    reg_sha = require_registration(root, registration)
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared / "cache_manifest.json"
    previous = _prior_cache(manifest_path, reg_sha)
    if previous is not None:
        return previous
    jcfg = joint_grid.load_config(cfg["joint_config"])
    jreg, fcfg, pcfg = joint_grid.verify(jcfg, full=True)
    requested_full = full_grid.load_config(cfg["full_config"])
    if Path(requested_full["config_path"]) != Path(fcfg["config_path"]):
        raise ValueError("requested full-grid controls differ from the joint campaign parent")
    freg, _ = full_grid.verify(fcfg)
    jroot, froot, oldroot = (Path(x["output_dir"]) for x in (jcfg, fcfg, pcfg))
    point_path = Path(cfg["point_table_path"]).resolve()
    points = pd.read_csv(point_path)
    keys = ["group", "model", "formulation", "task"]
    expected = {(g, m, f, t) for g in "ABCD" for m in MODELS for f in FORMULATIONS for t in TASKS}
    if (len(points) != 336 or points.duplicated(keys).any()
            or set(points[keys].itertuples(index=False, name=None)) != expected
            or not points.split.eq("test").all()):
        raise ValueError("the fixed complete336 Test point table is required")
    provenance = {str(point_path): _sha(point_path)}
    fold_meta = {int(row["fold"]): row for row in freg["folds"]}
    if set(fold_meta) != set(range(15)):
        raise ValueError("all15 frozen folds are required")
    arrays = seven_task.arrays_for(pcfg)
    baseline_parts = []
    audits = {}
    fold_data = {}
    for fold, meta in sorted(fold_meta.items()):
        data = seven_task.fold_for(pcfg, fold)
        ids = np.asarray(data["test"], dtype=np.int64)
        times = np.asarray(arrays["index_ns"][ids], dtype=np.int64)
        actual = np.column_stack([arrays[t][ids, 0] for t in TASKS]).astype(np.float64)
        labels = np.column_stack([make_task_labels(actual[:, c], t, meta["label_contract"]) for c, t in enumerate(TASKS)])
        source, directory = _verified_prediction(oldroot, "classical", f"fold_{fold:02d}_D", provenance, load_model=True)
        with (directory / "models.pkl").open("rb") as stream:
            original = pickle.load(stream)
        fit = fit_baselines(arrays, data["train"], meta["label_contract"], original["variance_filters"],
            context=fcfg["model"]["context"], horizon=16, fold=fold, target_constants=pcfg.get("targets"))
        predictions = predict_baselines(fit, arrays, ids, fold=fold)
        filters = []
        old_predictions = pd.read_parquet(source)
        for name in FILTERS:
            frame = _ordered(old_predictions, {"split": "test", "fold": fold, "group": "D", "model": name}, ids, times)
            if not np.array_equal(frame.actual_variance.to_numpy(dtype=np.float64), actual[:, 1]):
                raise ValueError("frozen variance-filter outcomes differ from the common4h cohort")
            variance = frame.pred_variance.to_numpy(dtype=np.float64)
            if not np.isfinite(variance).all() or np.any(variance < 0):
                raise ValueError("frozen variance baseline has unavailable or invalid forecasts; no fallback")
            filters.append(variance)
        filter_variance = np.column_stack(filters)
        generated_ewma = predictions["causal_simple"]["variance"]["scalar"]
        if not np.allclose(generated_ewma, filter_variance[:, 1], rtol=1e-10, atol=1e-14):
            raise ValueError("new EWMA comparator does not reproduce the frozen original baseline")
        scalar = np.column_stack([predictions["causal_simple"][t]["scalar"] for t in TASKS])
        scalar[:, 1] = filter_variance[:, 1]
        bundle_path = prepared / "baseline_fits" / f"fold_{fold:02d}.pkl"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bundle_path.with_name(bundle_path.name + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(fit, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(bundle_path)
        audit_path = bundle_path.with_suffix(".json")
        _json(audit_path, fit["metadata"])
        audits[str(fold)] = {"bundle_path": str(bundle_path), "bundle_sha256": _sha(bundle_path),
            "audit_path": str(audit_path), "audit_sha256": _sha(audit_path)}
        baseline_parts.append({"fold": np.full(len(ids), fold, dtype=np.int64), "decision_id": ids,
            "index_ns": times, "actual": actual, "labels": labels.astype(np.int64), "scalar": scalar,
            "classes": np.column_stack([predictions["causal_simple"][t]["predicted_class"] for t in TASKS]).astype(np.int64),
            "constant_crash_p": predictions["train_constant"]["crash"]["scalar"],
            "constant_scalar": np.column_stack([predictions["train_constant"][t]["scalar"] for t in TASKS]),
            "constant_classes": np.column_stack([predictions["train_constant"][t]["predicted_class"] for t in TASKS]).astype(np.int64),
            "ridge_scalar": np.column_stack([predictions["lowdim_ridge"][t]["scalar"] for t in TASKS]),
            "filter_variance": filter_variance,
            "filter_classes": np.column_stack([make_task_labels(filter_variance[:, c], "variance", meta["label_contract"]) for c in range(3)]).astype(np.int64)})
        fold_data[fold] = {"ids": ids, "times": times, "actual": actual, "labels": labels,
            "contract": meta["label_contract"]}
        print(json.dumps({"cache_baseline_fold": fold, "n_rows": len(ids)}), flush=True)
    baseline = {name: np.concatenate([part[name] for part in baseline_parts]) for name in baseline_parts[0]}
    if (len(baseline["fold"]) != 124488 or np.any(np.diff(baseline["index_ns"]) <= 0)
            or len(np.unique(baseline["decision_id"])) != len(baseline["decision_id"])):
        raise ValueError("the complete nonoverlapping124488-decision Test cohort is required")
    baseline_path = prepared / "baseline.npz"
    atomic_npz(baseline_path, **baseline)
    full_neural = {(int(t["fold"]), t["group"], t["model"], t["formulation"]): t["job"] for t in freg["neural_tasks"]}
    full_classical = {(int(t["fold"]), t["group"]): t["job"] for t in freg["classical_tasks"]}
    joint_neural = {(int(t["fold"]), t["group"], t["model"]): t["job"] for t in jreg["neural_tasks"]}
    joint_classical = {(int(t["fold"]), t["group"]): t["job"] for t in jreg["classical_tasks"]}
    panels, prediction_hashes = {}, {}
    panel_dir = prepared / "panels"
    for group in "ABCD":
        for model in MODELS:
            kind = "neural" if model in ("direct", "wm") else "classical"
            for formulation in FORMULATIONS:
                parts = []
                source_hashes = {}
                for fold in sorted(fold_meta):
                    reference = fold_data[fold]
                    source_root = jroot if formulation == "joint" else froot
                    job = (joint_neural[(fold, group, model)] if kind == "neural" else joint_classical[(fold, group)]) if formulation == "joint" else (
                        full_neural[(fold, group, model, formulation)] if kind == "neural" else full_classical[(fold, group)])
                    source, _ = _verified_prediction(source_root, kind, job, provenance)
                    source_hashes[str(source.resolve())] = provenance[str(source.resolve())]
                    frame = _ordered(pd.read_parquet(source), {"split": "test", "fold": fold,
                        "group": group, "model": model, "formulation": formulation, "seed": 7}, reference["ids"], reference["times"])
                    _assert_outcomes(frame, reference["actual"], reference["labels"])
                    parts.append(panel_arrays(frame, reference["contract"], formulation))
                panel = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
                verify_balanced_accuracies(panel, baseline["labels"], points, group, model, formulation)
                panel_id = f"{group}_{model}_{formulation}"
                panel_path = panel_dir / f"{panel_id}.npz"
                atomic_npz(panel_path, **panel, group=np.asarray(group), model=np.asarray(model), formulation=np.asarray(formulation))
                panels[panel_id] = {"path": str(panel_path), "sha256": _sha(panel_path),
                    "group": group, "model": model, "formulation": formulation, "source_prediction_sha256": source_hashes}
                prediction_hashes.update(source_hashes)
                print(json.dumps({"cache_panel": panel_id, "n_rows": len(panel["classes"])}), flush=True)
    prediction_hashes.update({path: checksum for path, checksum in provenance.items() if path.endswith("test_predictions.parquet")})
    manifest = {"complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": reg_sha, "baseline_path": str(baseline_path), "baseline_sha256": _sha(baseline_path),
        "panel_dir": str(panel_dir), "panels": panels, "n_rows": len(baseline["fold"]), "n_folds": 15,
        "task_order": list(TASKS), "filter_order": list(FILTERS), "baseline_fits": audits,
        "source_prediction_sha256": prediction_hashes, "source_artifact_sha256": provenance,
        "historical_seen_test": True, "test_selection_allowed": False,
        "no_missing_row_removal": True, "all336_balanced_accuracies_reproduced": True}
    _json(manifest_path, manifest)
    return {**manifest, "cache_manifest_path": str(manifest_path.resolve()), "cache_manifest_sha": _sha(manifest_path)}
