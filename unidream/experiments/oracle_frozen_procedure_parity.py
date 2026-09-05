"""Registered full-procedure replay on the already observed eight quarters.

This harness cannot select candidates or access additional evaluation periods.
It binds existing artifacts read-only, then refits the fixed procedure and
compares forecasts, scores, targets, traces and both accounting cost paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar, segment_masks
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import _immutable_json, _save_targets
from .oracle_derivative_delay import GROUPS, prepare as prepare_delay
from .oracle_derivative_features import make_derivative_groups
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_mean_controls import return_scores
from .robust_overlay import build_targets


FOLDS = tuple(range(5, 13))
MEANS = ("scale_mean", "technical_scaled", "perp_delay0_scaled", "technical_half", "perp_delay0_half")
RULES = ("utility_risk1", "utility_risk1_fallback_bh")
POLICIES = ("bh", "common_robust") + tuple(m + "_" + r for m in MEANS for r in RULES)
ROOT = "codex_outputs/oracle_mean_shrinkage_decisions_v1"
MEAN_ROOT = "codex_outputs/oracle_mean_control_decisions_v1"
DELAY_ROOT = "codex_outputs/oracle_derivative_delay_v1"
SOURCES = ("oracle_frozen_procedure_parity.py", "oracle_frozen_forecasts.py",
           "oracle_confirmation_contract.py", "oracle_mean_controls.py",
           "oracle_mean_shrinkage.py", "oracle_fallback_planner.py", "oracle_fallback_decisions.py",
           "oracle_conditional_planner.py", "oracle_derivative_delay.py",
           "oracle_derivative_delay_features.py", "oracle_derivative_features.py",
           "oracle_derivative_ablation.py", "oracle_derivative_crossed_decisions.py",
           "oracle_frontier.py", "oracle_frontier_features.py", "oracle_risk_calibration.py",
           "alpha_dd_search.py", "alpha_dd_features.py", "robust_overlay.py")
METADATA = ("configs/oracle_mean_shrinkage_decisions_20260905.yaml",
            "configs/oracle_mean_control_decisions_20260905.yaml",
            "configs/oracle_fallback_decisions_20260905.yaml",
            "configs/oracle_derivative_delay_20260905.yaml", "configs/oracle_frontier_20260905.yaml",
            "configs/oracle_confirmation_contract_20260906.yaml",
            "docs/experiments/oracle_mean_shrinkage_evidence_20260905/candidate_family_freeze.json",
            DELAY_ROOT + "/registration.json", DELAY_ROOT + "/preflight.json", DELAY_ROOT + "/results.json",
            MEAN_ROOT + "/registration.json", MEAN_ROOT + "/results.json",
            "codex_outputs/oracle_fallback_decisions_v1/registration.json")
FIXED = {"schema": "oracle-frozen-procedure-parity-v1", "development_folds": list(FOLDS),
         "calendar_evaluation_folds": list(range(4, 12)), "source_root": ROOT,
         "mean_source_root": MEAN_ROOT, "delay_source_root": DELAY_ROOT,
         "delay_config": "configs/oracle_derivative_delay_20260905.yaml",
         "output_dir": "codex_outputs/oracle_frozen_procedure_parity_v1",
         "mean_ids": list(MEANS), "policy_ids": list(POLICIES),
         "forecast_rtol": 1e-12, "forecast_atol": 1e-14,
         "metric_rtol": 1e-12, "metric_atol": 1e-12,
         "targets_must_match_exactly": True, "selection_permitted": False,
         "additional_periods_permitted": False, "expected_source_artifacts": 1064}


def validate_config(cfg):
    allowed = set(FIXED) | {"reference_registration_sha256", "reference_results_sha256",
                           "reference_preflight_sha256", "reference_fold_sha256",
                           "metadata_bindings", "preflight_sha256"}
    if set(cfg) != allowed or any(cfg.get(k) != v for k, v in FIXED.items()):
        raise ValueError("unregistered procedure parity contract")
    if set(cfg["reference_fold_sha256"]) != {f"fold_{f}.json" for f in FOLDS}:
        raise ValueError("incomplete registered reference folds")
    if set(cfg["metadata_bindings"]) != set(METADATA):
        raise ValueError("incomplete registered metadata set")


def compare_array(got, expected, *, name, exact=False, rtol=1e-12, atol=1e-14):
    """Require identical shapes/NaNs before any numerical tolerance check."""
    a, b = np.asarray(got), np.asarray(expected)
    if a.shape != b.shape or np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError(f"parity shape/type mismatch: {name}")
    if exact and a.dtype != b.dtype:
        raise ValueError(f"parity exact dtype mismatch: {name}")
    if a.dtype.kind not in "bifu" or b.dtype.kind not in "bifu":
        raise ValueError(f"parity nonnumeric array: {name}")
    if not np.array_equal(np.isnan(a), np.isnan(b)) or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError(f"parity nonfinite support mismatch: {name}")
    equal = np.array_equal(a, b, equal_nan=True) if exact else np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if not equal:
        raise ValueError(f"parity values differ: {name}")
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))) if finite.any() else 0.


def compare_tree(got, expected, *, name):
    if isinstance(expected, dict):
        if not isinstance(got, dict) or set(got) != set(expected):
            raise ValueError(f"parity object fields differ: {name}")
        return max((compare_tree(got[k], v, name=f"{name}.{k}") for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            raise ValueError(f"parity sequence differs: {name}")
        return max((compare_tree(a, b, name=f"{name}[{i}]") for i, (a, b) in enumerate(zip(got, expected))), default=0.)
    if expected is None or isinstance(expected, (str, bool, int)):
        if type(got) is not type(expected) or got != expected:
            raise ValueError(f"parity discrete value differs: {name}")
        return 0.
    return compare_array(got, expected, name=name, rtol=1e-12, atol=1e-12)


def _bound(path, sha):
    if file_digest(path) != sha:
        raise ValueError(f"reference binding changed: {path}")
    return json.loads(Path(path).read_text())


def load_references(cfg):
    """Verify the completed sources directly, never alter their output config."""
    root = Path(cfg["source_root"])
    reg = _bound(root / "registration.json", cfg["reference_registration_sha256"])
    result = _bound(root / "results.json", cfg["reference_results_sha256"])
    pre = _bound(root / "preflight.json", cfg["reference_preflight_sha256"])
    if (result["registration_sha256"] != digest(reg)
            or reg["preflight_sha256"] != cfg["reference_preflight_sha256"]
            or pre["source_sha256"] != reg["source_sha256"]
            or file_digest(Path(METADATA[0])) != reg["config_sha256"]
            or yaml.safe_load(Path(METADATA[0]).read_text()) != reg["config"]
            or digest({k: v for k, v in reg["config"].items() if k != "preflight_sha256"}) != pre["config_contract_sha256"]):
        raise ValueError("reference registration/preflight mismatch")
    artifacts = dict(reg["source_artifact_sha256"])
    for f in FOLDS:
        fold = _bound(root / f"fold_{f}.json", cfg["reference_fold_sha256"][f"fold_{f}.json"])
        if (fold["registration_sha256"] != digest(reg)
                or fold["rows"] != [r for r in result["rows"] if r["fold"] == f]
                or fold["scores"] != [r for r in result["scores"] if r["fold"] == f]):
            raise ValueError("reference fold payload mismatch")
        for path, sha in fold["artifact_sha256"].items():
            if path in artifacts and artifacts[path] != sha:
                raise ValueError("conflicting reference artifact binding")
            artifacts[path] = sha
    if len(artifacts) != cfg["expected_source_artifacts"]:
        raise ValueError("incomplete source artifact inventory")
    for path, sha in artifacts.items():
        if file_digest(Path(path)) != sha:
            raise ValueError(f"source artifact changed: {path}")
    for name, sha in reg["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name)) != sha:
            raise ValueError("reference source module changed")
    rows = {(r["fold"], r["candidate_id"]): r for r in result["rows"]}
    scores = {(r["fold"], r["mean_id"]): r for r in result["scores"]}
    if (len(rows) != len(result["rows"]) or set(rows) != {(f, c) for f in FOLDS for c in POLICIES}
            or len(scores) != len(result["scores"]) or set(scores) != {(f, m) for f in FOLDS for m in MEANS}):
        raise ValueError("reference family incomplete")
    return reg, pre, rows, scores, artifacts


def prepare(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text()); validate_config(cfg)
    for path, sha in cfg["metadata_bindings"].items():
        if file_digest(Path(path)) != sha:
            raise ValueError(f"frozen metadata changed: {path}")
    reg, reference_pre, rows, scores, artifacts = load_references(cfg)
    delay = prepare_delay(Path(cfg["delay_config"]))
    dc, fc, bars, groups, original, tv, y, old_masks, old_pre = delay
    if (reg["source_config"] != fc or reg["spot_data_proof"] != old_pre["spot_data_proof"]
            or old_pre != json.loads((Path(DELAY_ROOT) / "preflight.json").read_text())):
        raise ValueError("reference data/execution contract mismatch")
    common = np.isfinite(tv.to_numpy()).all(axis=1) & np.isfinite(original["flow"].to_numpy()).all(axis=1)
    for frame in make_derivative_groups(bars, delay_um(cfg, fc)).values():
        common &= np.isfinite(frame.to_numpy()).all(axis=1)
    for group in GROUPS:
        common &= np.isfinite(groups[group].to_numpy()).all(axis=1)
    masks, support = {}, []
    for f in FOLDS:
        dates = calendar(f - 1)
        m = segment_masks(bars.index, common, np.isfinite(y).all(axis=1), f - 1)
        for name in old_masks[f]:
            compare_array(m[name], old_masks[f][name], name=f"fold{f}.{name}", exact=True)
        masks[f] = m
        old_support = next(s for s in old_pre["folds"] if s["fold"] == f)
        if any(rows[f, cid]["regime"] != old_support["regime"] for cid in POLICIES):
            raise ValueError("reference regime no longer matches reconstructed inputs")
        ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        ref_path = Path(MEAN_ROOT) / "forecasts" / f"fold{f}_scale_mean.npz"
        with np.load(ref_path, allow_pickle=False) as saved:
            for name, value in {"timestamps": bars.index[ix].asi8, "inference_mask": m["inference"][ix],
                                "score_support": m["score"][ix]}.items():
                compare_array(value, saved[name], name=f"fold{f}.{name}", exact=True)
        support.append({"reference_validation_fold": f, "adapter_calendar_test_fold": f - 1,
                        "same_already_observed_interval": True,
                        **{k: v.isoformat() for k, v in dates.items() if isinstance(v, pd.Timestamp)},
                        "mask_sha256": {k: mask_digest(bars.index, v) for k, v in m.items()},
                        "counts": {k: int(v.sum()) for k, v in m.items()}})
    preflight = {"config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
                 "source_sha256": {n: file_digest(Path(__file__).with_name(n)) for n in SOURCES},
                 "source_artifact_sha256": artifacts, "support": support,
                 "spot_data_proof": old_pre["spot_data_proof"], "um_data_proof": old_pre["um_data_proof"],
                 "reference_registration_sha256": cfg["reference_registration_sha256"],
                 "reference_results_sha256": cfg["reference_results_sha256"],
                 "scope": "data-only full-procedure parity preflight on reused development intervals",
                 "new_forecast_or_policy_computed": False, "additional_periods_accessed": False}
    return cfg, dc, fc, bars, groups, original, y, masks, preflight, rows, scores, artifacts


def delay_um(cfg, fc):
    from .oracle_derivative_ablation import validate_um
    dc = yaml.safe_load(Path(cfg["delay_config"]).read_text())
    um, _ = validate_um(Path(dc["um_path"]), fc["data_cutoff"], fc["symbol"])
    return um


def validate_completed_fold(saved, f, output, registration_sha, reference_rows, reference_scores, sources):
    """A partial/resumed manifest cannot manufacture a successful empty fold."""
    if set(saved) != {"registration_sha256", "rows", "scores", "max_absolute_differences", "artifact_sha256"}:
        raise ValueError("incomplete saved fold schema")
    if saved["registration_sha256"] != registration_sha:
        raise ValueError("partial parity registration changed")
    rows = {r["candidate_id"]: r for r in saved["rows"]}
    scores = {r["mean_id"]: r for r in saved["scores"]}
    if (len(rows) != len(saved["rows"]) or set(rows) != set(POLICIES)
            or len(scores) != len(saved["scores"]) or set(scores) != set(MEANS)
            or any(r["fold"] != f for r in saved["rows"] + saved["scores"])):
        raise ValueError("incomplete saved fold family")
    names = {"models": [(n, ".joblib") for n in ("technical_mean", "perp_delay0_mean", "technical_variance")],
             "forecasts": [(n, ".npz") for n in MEANS], "targets": [(n, ".npz") for n in POLICIES],
             "traces": [(n, ".json") for n in POLICIES if n not in ("bh", "common_robust")],
             "calibration": [("technical", ".npz"), ("perp_delay0", ".npz"), ("provenance", ".json")]}
    expected_paths = {str(output / kind / f"fold{f}_{n}{ext}") for kind, entries in names.items() for n, ext in entries}
    bindings = saved["artifact_sha256"]
    if set(bindings) != expected_paths:
        raise ValueError("incomplete saved model/forecast/target/trace/calibration inventory")
    for path, sha in bindings.items():
        if file_digest(Path(path)) != sha: raise ValueError("saved fold artifact changed")
    for mean, score in scores.items():
        path = output / "forecasts" / f"fold{f}_{mean}.npz"
        ref = Path(ROOT if mean.endswith("half") else MEAN_ROOT) / "forecasts" / path.name
        if bindings[str(path)] != score["forecast_sha256"]:
            raise ValueError("saved forecast-to-score binding changed")
        with np.load(path, allow_pickle=False) as got, np.load(ref, allow_pickle=False) as expected:
            if set(got.files) != set(expected.files): raise ValueError("saved forecast schema changed")
            for k in expected.files:
                compare_array(got[k], expected[k], name="saved forecast " + k,
                              exact=k in ("timestamps", "inference_mask", "score_support", "actual"))
            measured = return_scores(got["actual"], got["mu"], got["score_support"], float(got["fit_return_mean"]))
        compare_tree(measured, {k: score[k] for k in measured}, name="saved scores")
        compare_tree({k: score[k] for k in measured}, {k: reference_scores[f, mean][k] for k in measured}, name="source scores")
        if score["regime"] != reference_scores[f, mean]["regime"]: raise ValueError("saved score regime changed")
    for cid, row in rows.items():
        path = output / "targets" / f"fold{f}_{cid}.npz"; ref = Path(ROOT) / "targets" / path.name
        if bindings[str(path)] != row["targets_sha256"] or sources[str(ref)] != reference_rows[f, cid]["targets_sha256"]:
            raise ValueError("target-to-row binding changed")
        with np.load(path, allow_pickle=False) as got, np.load(ref, allow_pickle=False) as expected:
            if set(got.files) != {"timestamps", "targets"}: raise ValueError("saved target schema changed")
            for k in got.files: compare_array(got[k], expected[k], name="saved " + k, exact=True)
        if row["regime"] != reference_rows[f, cid]["regime"]: raise ValueError("saved economic regime changed")
        for cost in ("base", "stress_2x"):
            compare_tree(row[cost], reference_rows[f, cid][cost], name="saved " + cost)
        if cid not in ("bh", "common_robust"):
            trace = output / "traces" / f"fold{f}_{cid}.json"
            sha = reference_rows[f, cid]["diagnostic"]["trace_sha256"]
            ref_paths = [p for p, value in sources.items() if value == sha and "/traces/" in p]
            if not ref_paths: raise ValueError("saved trace source missing")
            compare_tree(json.loads(trace.read_text()), json.loads(Path(ref_paths[0]).read_text()), name="saved trace")
    for group in ("technical", "perp_delay0"):
        path = output / "calibration" / f"fold{f}_{group}.npz"
        ref = Path(DELAY_ROOT) / "calibration" / path.name
        if str(ref) not in sources: raise ValueError("calibration outside reference inventory")
        expected_keys = {"timestamps", "actual", "scale_mask", "interval_mask", "mu"}
        if group == "technical": expected_keys |= {"log_variance", "variance"}
        with np.load(path, allow_pickle=False) as got, np.load(ref, allow_pickle=False) as expected:
            if set(got.files) != expected_keys: raise ValueError("saved calibration schema changed")
            for k in got.files:
                compare_array(got[k], expected[k], name="saved calibration " + k,
                              exact=k in ("timestamps", "actual", "scale_mask", "interval_mask"))
    doc = json.loads((output / "calibration" / f"fold{f}_provenance.json").read_text())
    if set(doc) != {"calibration", "provenance"}: raise ValueError("saved fitted provenance schema changed")
    cal = doc["calibration"]
    delay_scores = json.loads((Path(DELAY_ROOT) / "results.json").read_text())["scores"]
    original_scores = {(s["fold"], s["model_id"]): s for s in delay_scores}
    for group in ("technical", "perp_delay0"):
        compare_array(cal["return_bias"][group], original_scores[f, group + "_scaled"]["calibration"]["return_bias"], name="saved bias")
    compare_array(cal["variance_multiplier"], original_scores[f, "technical_scaled"]["calibration"]["variance_scale"], name="saved scale")
    for version in ("raw", "scaled"):
        reference = original_scores[f, "technical_" + version]["calibration"]
        compare_tree(cal["technical_quantiles"][version], {k: reference[k] for k in ("return_quantile", "volatility_quantile")}, name="saved quantile")
    with np.load(Path(MEAN_ROOT) / "forecasts" / f"fold{f}_scale_mean.npz", allow_pickle=False) as reference:
        compare_array(cal["fit_mean"], reference["fit_return_mean"], name="saved fit mean")
        compare_array(cal["scale_mean"], reference["mu"][reference["inference_mask"]][0], name="saved anchor")
    old_pre = json.loads((Path(DELAY_ROOT) / "preflight.json").read_text())
    count = next(s["counts"] for s in old_pre["folds"] if s["fold"] == f)
    compare_tree(cal["counts"], {k: count[k] for k in ("fit", "scale", "interval")}, name="saved calibration counts")
    compare_tree(doc["provenance"]["mask_counts"], {k: count[k] for k in ("fit", "scale", "interval", "predict", "inference")}, name="saved fitted counts")
    diff = saved["max_absolute_differences"]
    if (set(diff) != {"forecast", "score", "target", "trace", "account", "calibration"}
            or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in diff.values())
            or diff["target"] != 0):
        raise ValueError("invalid saved parity differences")


def run(config_path):
    from .oracle_frozen_forecasts import fit_frozen_forecasts
    cfg, dc, fc, bars, groups, original, y, masks, pre, reference_rows, reference_scores, sources = prepare(config_path)
    output = Path(cfg["output_dir"])
    if (output / "results.json").exists():
        raise ValueError("immutable parity already completed")
    if (file_digest(output / "preflight.json") != cfg["preflight_sha256"]
            or json.loads((output / "preflight.json").read_text()) != pre):
        raise ValueError("registered parity preflight changed")
    registration = {"config": cfg, "config_sha256": file_digest(config_path),
                    "source_sha256": pre["source_sha256"], "preflight_sha256": cfg["preflight_sha256"],
                    "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__},
                    "scope": "fixed procedure reproducibility on the same development8quarters; no new candidate or period"}
    _immutable_json(output / "registration.json", registration)
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": 2 * execution["one_way_cost"], "borrow_annual": 2 * execution["borrow_annual"]}
    all_rows, all_scores, differences = [], [], {}
    for f in FOLDS:
        fold_path = output / f"fold_{f}.json"
        if fold_path.exists():
            saved = json.loads(fold_path.read_text())
            validate_completed_fold(saved, f, output, digest(registration), reference_rows, reference_scores, sources)
            all_rows.extend(saved["rows"]); all_scores.extend(saved["scores"])
            differences[str(f)] = saved["max_absolute_differences"]
            continue
        m = masks[f]; dates = calendar(f - 1)
        ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        window = bars.loc[ix]
        fresh = fit_frozen_forecasts({k: groups[k] for k in ("technical", "perp_delay0")}, y,
            **{k + "_mask": m[k] for k in ("fit", "scale", "interval", "predict", "inference")})
        bindings, rows, scores = {}, [], []
        diff = {"forecast": 0., "score": 0., "target": 0., "trace": 0., "account": 0., "calibration": 0.}
        def save_arrays(kind, name, arrays):
            path = output / kind / f"fold{f}_{name}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays): raise ValueError("partial array schema changed")
                    for k, v in arrays.items(): compare_array(v, saved[k], name=str(path) + k, exact=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
            bindings[str(path)] = file_digest(path)
            return path
        for name, model in fresh["models"].items():
            path = output / "models" / f"fold{f}_{name}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Completed fold manifests are resumed above. A partial model can
            # only be reused if the exact deterministic serialization agrees.
            if path.exists():
                temp = path.with_suffix(".verify.joblib"); joblib.dump(model, temp)
                if file_digest(temp) != file_digest(path): raise ValueError("partial fitted model changed")
                temp.unlink()
            else: joblib.dump(model, path)
            bindings[str(path)] = file_digest(path)
        actual = y[ix].copy(); actual[~m["score"][ix]] = np.nan
        for mean in MEANS:
            mu, variance = fresh["means"][mean][ix], fresh["variance"][ix]
            arrays = {"timestamps": window.index.asi8, "mu": mu, "variance": variance,
                      "inference_mask": m["inference"][ix], "score_support": m["score"][ix],
                      "actual": actual, "fit_return_mean": np.asarray(fresh["calibration"]["fit_mean"])}
            reference = Path(ROOT if mean.endswith("half") else MEAN_ROOT) / "forecasts" / f"fold{f}_{mean}.npz"
            if file_digest(reference) != reference_scores[f, mean]["forecast_sha256"]:
                raise ValueError("reference forecast-to-score changed")
            with np.load(reference, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays): raise ValueError("reference forecast schema differs")
                for k, v in arrays.items():
                    diff["forecast"] = max(diff["forecast"], compare_array(v, saved[k], name=f"fold{f}.{mean}.{k}",
                        exact=k in ("timestamps", "inference_mask", "score_support", "actual"),
                        rtol=cfg["forecast_rtol"], atol=cfg["forecast_atol"]))
            forecast_path = save_arrays("forecasts", mean, arrays)
            score = return_scores(actual, mu, arrays["score_support"], fresh["calibration"]["fit_mean"])
            diff["score"] = max(diff["score"], compare_tree(score, {k: reference_scores[f, mean][k] for k in score}, name=f"fold{f}.{mean}.score"))
            scores.append({"fold": f, "mean_id": mean, "regime": reference_scores[f, mean]["regime"],
                           **score, "forecast_sha256": bindings[str(forecast_path)]})
        cal_ix = np.asarray((bars.index >= dates["scale_start"]) & (bars.index < dates["evaluation_start"]))
        cal_common = {"timestamps": bars.index[cal_ix].asi8,
                      "actual": fresh["calibration_arrays"]["actual"][cal_ix],
                      "scale_mask": m["scale"][cal_ix], "interval_mask": m["interval"][cal_ix]}
        for group in ("technical", "perp_delay0"):
            raw = fresh["raw_predictions"][group]
            arrays = {**cal_common, **{k: v[cal_ix] for k, v in raw.items()}}
            reference_cal = Path(DELAY_ROOT) / "calibration" / f"fold{f}_{group}.npz"
            if str(reference_cal) not in sources: raise ValueError("reference calibration inventory missing")
            with np.load(reference_cal, allow_pickle=False) as saved:
                for name, value in arrays.items():
                    diff["calibration"] = max(diff["calibration"], compare_array(value, saved[name],
                        name=f"fold{f}.{group}.calibration.{name}",
                        exact=name in ("timestamps", "actual", "scale_mask", "interval_mask"),
                        rtol=cfg["forecast_rtol"], atol=cfg["forecast_atol"]))
            save_arrays("calibration", group, arrays)
        # Refitting must also preserve the scale bias and technical variance
        # multiplier/quantiles, not only the final inference arrays.
        delay_result = json.loads((Path(DELAY_ROOT) / "results.json").read_text())
        source_scores = {(s["fold"], s["model_id"]): s for s in delay_result["scores"]}
        for group in ("technical", "perp_delay0"):
            expected = source_scores[f, group + "_scaled"]["calibration"]
            diff["calibration"] = max(diff["calibration"], compare_array(
                fresh["calibration"]["return_bias"][group], expected["return_bias"], name="return bias"))
        expected = source_scores[f, "technical_scaled"]["calibration"]
        diff["calibration"] = max(diff["calibration"], compare_array(
            fresh["calibration"]["variance_multiplier"], expected["variance_scale"], name="variance scale"))
        for version in ("raw", "scaled"):
            expected = source_scores[f, "technical_" + version]["calibration"]
            diff["calibration"] = max(diff["calibration"], compare_tree(
                fresh["calibration"]["technical_quantiles"][version],
                {k: expected[k] for k in ("return_quantile", "volatility_quantile")}, name="technical quantiles"))
        for cid in POLICIES:
            trace = None
            if cid == "bh": targets = np.full(len(window), np.nan)
            elif cid == "common_robust":
                targets = build_targets(original["base16"])[ix].copy(); targets[~m["inference"][ix]] = np.nan
            else:
                rule = RULES[1] if cid.endswith(RULES[1]) else RULES[0]
                mean = cid[:-(len(rule) + 1)]
                mu, variance, inference = fresh["means"][mean][ix], fresh["variance"][ix], m["inference"][ix]
                if rule == RULES[1]:
                    targets, trace = fallback_targets(window, mu, variance, execution,
                        inference_mask=inference, risk_aversion=1, cost_multiplier=2)
                    action = action_masks(window.index, window.open.to_numpy(), inference)
                    check_action_support(targets, action); check_trace_support(targets, action, trace)
                else:
                    targets, trace = conditional_targets(window, mu, variance, execution, risk_aversion=1, cost_multiplier=2)
                reference_trace_sha = reference_rows[f, cid]["diagnostic"]["trace_sha256"]
                candidates = [p for p, sha in sources.items() if sha == reference_trace_sha and "/traces/" in p]
                if not candidates: raise ValueError("reference trace binding missing")
                expected_trace = json.loads(Path(candidates[0]).read_text())
                diff["trace"] = max(diff["trace"], compare_tree(trace, expected_trace, name=f"fold{f}.{cid}.trace"))
                path = output / "traces" / f"fold{f}_{cid}.json"; _immutable_json(path, trace)
                bindings[str(path)] = file_digest(path)
            reference_target = Path(ROOT) / "targets" / f"fold{f}_{cid}.npz"
            if sources[str(reference_target)] != reference_rows[f, cid]["targets_sha256"]:
                raise ValueError("reference target-to-row binding changed")
            with np.load(reference_target, allow_pickle=False) as saved:
                compare_array(window.index.asi8, saved["timestamps"], name="target calendar", exact=True)
                diff["target"] = max(diff["target"], compare_array(targets, saved["targets"], name=f"fold{f}.{cid}.target", exact=True))
            target_path = output / "targets" / f"fold{f}_{cid}.npz"
            target_sha = _save_targets(target_path, window.index, targets); bindings[str(target_path)] = target_sha
            costs = {cost: metrics(window, targets, c) for cost, c in (("base", execution), ("stress_2x", stress))}
            for cost in costs:
                diff["account"] = max(diff["account"], compare_tree(costs[cost], reference_rows[f, cid][cost], name=f"fold{f}.{cid}.{cost}"))
            rows.append({"fold": f, "candidate_id": cid, "regime": reference_rows[f, cid]["regime"],
                         **costs, "targets_sha256": target_sha})
        # Full fitted/calibration evidence is saved before this fold can finish.
        provenance_path = output / "calibration" / f"fold{f}_provenance.json"
        _immutable_json(provenance_path, {"calibration": fresh["calibration"], "provenance": fresh["provenance"]})
        bindings[str(provenance_path)] = file_digest(provenance_path)
        saved = {"registration_sha256": digest(registration), "rows": rows, "scores": scores,
                 "max_absolute_differences": diff, "artifact_sha256": bindings}
        validate_completed_fold(saved, f, output, digest(registration), reference_rows, reference_scores, sources)
        _immutable_json(fold_path, saved); all_rows.extend(rows); all_scores.extend(scores); differences[str(f)] = diff
        print(json.dumps({"fold": f, "policies": len(rows), "forecasts": len(scores), "max_differences": diff}), flush=True)
    result = {"registration_sha256": digest(registration), "rows": all_rows, "scores": all_scores,
              "max_absolute_differences": differences, "parity_pass": True,
              "new_candidate_count": 0, "new_evaluation_period_count": 0,
              "selection_performed": False, "high_probability_generalization_established": False,
              "prospective_receipt_parity_established": False, "scope": registration["scope"]}
    _immutable_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        values = prepare(args.config); cfg, pre = values[0], values[8]
        path = Path(cfg["output_dir"]) / "preflight.json"; _immutable_json(path, pre)
        print(json.dumps({"path": str(path), "sha256": file_digest(path), "new_forecast_or_policy_computed": False}))
    else:
        run(args.config)
