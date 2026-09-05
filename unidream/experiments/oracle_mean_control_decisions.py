"""Decision value of frozen learned means against causal constant-mean controls."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import yaml

from .alpha_dd_search import digest, file_digest, fold_spec, load_bars, metrics, validate_data_artifact
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import (
    _copy_verified, _immutable_json, _save_targets, crossed_targets,
    decision_summaries, load_paired_forecasts,
)
from .oracle_mean_controls import constant_means, return_scores

FOLDS = tuple(range(5, 13))
LEARNED = ("technical_raw", "technical_scaled", "perp_delay0_raw", "perp_delay0_scaled")
MEANS = ("zero", "fit_mean", "scale_mean") + LEARNED
POLICIES = ("point", "utility_risk1")
VARIANCE_SOURCE = "technical_scaled"
PAIRS = tuple((candidate, reference) for reference, candidate in itertools.combinations(MEANS, 2))


def validate_config(cfg):
    fixed = {"schema": "oracle-mean-control-decisions-v1", "development_folds": list(FOLDS),
             "mean_sources": list(MEANS), "variance_source": VARIANCE_SOURCE,
             "policies": list(POLICIES), "utility_cost_multiplier": 2,
             "minimum_quarters_per_regime": 3, "inference_rows": 2586, "score_rows": 2574}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("unregistered mean control family")


def paired_summaries(scores, rows):
    si = {(s["fold"], s["mean_id"]): s for s in scores}
    ri = {(r["fold"], r["candidate_id"]): r for r in rows}
    policies = {m+"_"+p for m in MEANS for p in POLICIES} | {"bh", "common_robust"}
    if (len(si) != len(scores) or len(ri) != len(rows)
            or set(si) != {(f, m) for f in FOLDS for m in MEANS}
            or set(ri) != {(f, p) for f in FOLDS for p in policies}):
        raise ValueError("missing or duplicate registered results")
    for fold in FOLDS:
        anchor = si[fold, "zero"]
        if (any(si[fold, m]["regime"] != anchor["regime"] or si[fold, m]["rows"] != anchor["rows"] for m in MEANS)
                or any(ri[fold, p]["regime"] != anchor["regime"] for p in policies)):
            raise ValueError("unpaired mean control support")
    result = {}
    for candidate, reference in PAIRS:
        entry = {"candidate": candidate, "reference": reference,
                 "difference_convention": "candidate minus reference", "regimes": {}}
        for regime in ("all", "bull", "bear", "sideways"):
            folds = [f for f in FOLDS if regime == "all" or si[f, candidate]["regime"]["trend"] == regime]
            prediction = {}
            for metric in ("return_mse", "return_mae"):
                delta = [si[f, candidate][metric]-si[f, reference][metric] for f in folds]
                refmean = float(np.mean([si[f, reference][metric] for f in folds]))
                prediction[metric] = {"mean_difference": float(np.mean(delta)),
                    "relative_loss_reduction": -float(np.mean(delta))/refmean if refmean > 0 else None,
                    "improved_quarters": int(np.sum(np.asarray(delta) < 0))}
            economics = {p: {cost: {metric: float(np.mean([
                ri[f, candidate+"_"+p][cost][metric]-ri[f, reference+"_"+p][cost][metric] for f in folds]))
                for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades", "fees_initial_equity_units", "borrow_initial_equity_units")}
                for cost in ("base", "stress_2x")} for p in POLICIES}
            entry["regimes"][regime] = {"quarters": len(folds), "prediction": prediction, "policies": economics}
        result[candidate+"_vs_"+reference] = entry
    return result


def prediction_summaries(scores):
    indexed = {(s["fold"], s["mean_id"]): s for s in scores}
    if len(indexed) != len(scores) or set(indexed) != {(f, m) for f in FOLDS for m in MEANS}:
        raise ValueError("missing or duplicate registered forecasts")
    for fold in FOLDS:
        anchor = indexed[fold, "zero"]
        if any(indexed[fold, m]["regime"] != anchor["regime"] or indexed[fold, m]["rows"] != anchor["rows"] for m in MEANS):
            raise ValueError("unpaired mean control support")
    result = {}
    for mean in MEANS:
        result[mean] = {}
        for regime in ("all", "bull", "bear", "sideways"):
            selected = [s for s in scores if s["mean_id"] == mean and (regime == "all" or s["regime"]["trend"] == regime)]
            n = sum(s["rows"] for s in selected)
            result[mean][regime] = {"quarters": len(selected), "rows": n,
                "metrics": {metric: {"equal_quarter_mean": float(np.mean([s[metric] for s in selected])),
                    "pooled_row_mean": float(sum(s[metric]*s["rows"] for s in selected)/n)}
                    for metric in ("return_mse", "return_mae", "return_sign_accuracy", "zero_return_mse", "fit_mean_return_mse")},
                "mean_return_rank_ic": float(np.mean([s["return_rank_ic"] for s in selected]))
                    if all(s["return_rank_ic"] is not None for s in selected) else None}
    return result


def load_sources(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    validate_config(cfg)
    root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    if root.resolve() == output.resolve() or (output/"results.json").exists():
        raise ValueError("immutable source or completed output")
    for name in ("registration", "results", "preflight"):
        if file_digest(root/(name+".json")) != cfg["source_"+name+"_sha256"]:
            raise ValueError(f"parent {name} changed")
    reg, source, pre = [json.loads((root/(n+".json")).read_text()) for n in ("registration", "results", "preflight")]
    if source["registration_sha256"] != digest(reg) or reg["preflight_sha256"] != cfg["source_preflight_sha256"]:
        raise ValueError("parent result/registration/preflight binding mismatch")
    config = Path(cfg["source_config"])
    if file_digest(config) != reg["config_sha256"] or yaml.safe_load(config.read_text()) != reg["config"]:
        raise ValueError("parent config changed")
    if digest({k: v for k, v in reg["config"].items() if k != "preflight_sha256"}) != pre["config_contract_sha256"]:
        raise ValueError("parent preflight/config binding changed")
    fc = reg["source_config"]
    fc_path = Path(reg["config"]["source_config"])
    if file_digest(fc_path) != pre["source_config_sha256"] or yaml.safe_load(fc_path.read_text()) != fc:
        raise ValueError("source execution config changed")
    for mapping in (reg["source_sha256"], pre["source_sha256"]):
        for name, expected in mapping.items():
            if file_digest(Path(__file__).with_name(name)) != expected:
                raise ValueError(f"parent source changed: {name}")
    proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    if proof != pre["spot_data_proof"]:
        raise ValueError("Spot data proof changed")
    # The immutable input forecasts remain bound to their complete source manifest.
    artifacts = {}
    for fold in FOLDS:
        saved = json.loads((root/f"fold_{fold}.json").read_text())
        if saved["registration_sha256"] != digest(reg):
            raise ValueError("parent fold registration changed")
        for path, expected in saved["artifact_sha256"].items():
            if file_digest(Path(path)) != expected:
                raise ValueError("parent artifact changed")
            artifacts[path] = expected
        if (saved["rows"] != [r for r in source["rows"] if r["fold"] == fold]
                or saved["scores"] != [s for s in source["scores"] if s["fold"] == fold]):
            raise ValueError("parent fold/result mismatch")
    si = {(s["fold"], s["model_id"]): s for s in source["scores"]}
    ri = {(r["fold"], r["candidate_id"]): r for r in source["rows"]}
    if len(si) != len(source["scores"]) or len(ri) != len(source["rows"]):
        raise ValueError("duplicate source results")
    forecast_bindings, calibration_bindings = {}, {}
    for fold in FOLDS:
        for mid in LEARNED:
            name = f"fold{fold}_{mid}.npz"
            path = root/"forecasts"/name
            expected = si[fold, mid]["forecast_sha256"]
            if file_digest(path) != expected or artifacts[str(path)] != expected:
                raise ValueError("source forecast binding mismatch")
            forecast_bindings[name] = expected
        for group in ("technical", "perp_delay0"):
            cal = si[fold, group+"_raw"]["provenance"]["calibration"]
            path = root/"calibration"/f"fold{fold}_{group}.npz"
            if Path(cal["path"]).resolve() != path.resolve() or file_digest(path) != cal["sha256"]:
                raise ValueError("source calibration binding mismatch")
            calibration_bindings[path.name] = cal["sha256"]
    if forecast_bindings != cfg["forecast_sha256"] or calibration_bindings != cfg["calibration_sha256"]:
        raise ValueError("registered source forecast/calibration bindings changed")
    return cfg, fc, reg, source, pre, si, ri, proof, forecast_bindings, calibration_bindings


def run(config_path):
    cfg, fc, parent_reg, source, pre, si, ri, proof, forecasts, calibrations = load_sources(config_path)
    root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    sources = ("oracle_mean_control_decisions.py", "oracle_mean_controls.py", "oracle_derivative_crossed_decisions.py",
               "oracle_derivative_ablation.py", "oracle_frontier.py", "oracle_conditional_planner.py", "alpha_dd_search.py")
    registration = {"config": cfg, "config_sha256": file_digest(config_path), "source_config": fc,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {n: file_digest(Path(__file__).with_name(n)) for n in sources},
        "source_registration_sha256": cfg["source_registration_sha256"], "source_results_sha256": cfg["source_results_sha256"],
        "source_preflight_sha256": cfg["source_preflight_sha256"],
        "forecast_bindings": forecasts, "calibration_bindings": calibrations, "spot_data_proof": proof,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "scope": "adaptive mean information and decision diagnostic on reused validation; no selection or new significance tests"}
    _immutable_json(output/"registration.json", registration)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": 2*execution["one_way_cost"], "borrow_annual": 2*execution["borrow_annual"]}
    rows, scores, support = [], [], []
    max_control_diff = 0.
    pre_folds = {f["fold"]: f for f in pre["folds"]}
    for fold in FOLDS:
        spec = fold_spec(fold, fc["fold_anchor"])
        index = pd.date_range(spec["val_start"], spec["val_end"], freq="15min", inclusive="left")
        if spec["val_end"] > pd.Timestamp(fc["data_cutoff"]) or not index.isin(bars.index).all():
            raise ValueError("invalid fixed validation calendar")
        window = bars.loc[index]
        if window.bar_available.mean() < fc["minimum_bar_coverage"]:
            raise ValueError("incomplete source price support")
        risk_path = root/"forecasts"/f"fold{fold}_{VARIANCE_SOURCE}.npz"
        means = {}
        for mid in LEARNED:
            means[mid], variance, inference, scoring = load_paired_forecasts(root/"forecasts"/f"fold{fold}_{mid}.npz", risk_path, expected_index=index)
        with np.load(risk_path, allow_pickle=False) as saved:
            actual, fit_mean = saved["actual"].copy(), float(saved["fit_return_mean"])
        support_row = pre_folds[fold]
        for key, mask in (("inference", inference), ("score", scoring)):
            full = np.zeros(len(bars), bool); full[bars.index.get_indexer(index)] = mask
            if mask_digest(bars.index, full) != support_row["mask_sha256"][key]:
                raise ValueError("frozen forecast mask differs from data preflight")
        cal_path = root/"calibration"/f"fold{fold}_technical.npz"
        with np.load(cal_path, allow_pickle=False) as saved:
            cal_actual, scale_mask = saved["actual"].copy(), saved["scale_mask"].copy()
            cal_index = pd.to_datetime(saved["timestamps"], utc=True)
            interval_mask = saved["interval_mask"].copy()
        expected_cal = pd.date_range(support_row["train_end"], spec["val_start"], freq="15min", inclusive="left")
        if (not cal_index.equals(expected_cal) or scale_mask.dtype != bool or interval_mask.dtype != bool
                or scale_mask.shape != (len(cal_index),) or interval_mask.shape != (len(cal_index),)
                or cal_actual.shape != (len(cal_index), 3) or np.any(scale_mask & interval_mask)):
            raise ValueError("invalid source calibration calendar/masks")
        for key, mask in (("scale", scale_mask), ("interval", interval_mask)):
            full = np.zeros(len(bars), bool); full[bars.index.get_indexer(cal_index)] = mask
            if mask_digest(bars.index, full) != support_row["mask_sha256"][key]:
                raise ValueError("source calibration mask differs from preflight")
        # Confirm the two groups share outcomes/scale support, not only row counts.
        with np.load(root/"calibration"/f"fold{fold}_perp_delay0.npz", allow_pickle=False) as saved:
            if (not np.array_equal(saved["timestamps"], cal_index.asi8)
                    or not np.array_equal(saved["scale_mask"], scale_mask)
                    or not np.array_equal(saved["interval_mask"], interval_mask)
                    or not np.array_equal(saved["actual"], cal_actual, equal_nan=True)):
                raise ValueError("unpaired calibration outcomes")
        if np.any(cal_index[scale_mask]+pd.Timedelta(minutes=375) >= pd.Timestamp(support_row["scale_end"])):
            raise ValueError("scale mean includes unavailable labels")
        for mid in LEARNED:
            with np.load(root/"forecasts"/f"fold{fold}_{mid}.npz", allow_pickle=False) as saved:
                if float(saved["fit_return_mean"]) != fit_mean or si[fold, mid]["provenance"]["fit_return_mean"] != fit_mean:
                    raise ValueError("unpaired fitted return mean")
        means.update(constant_means(inference_mask=inference, fit_mean=fit_mean, calibration_actual=cal_actual, scale_mask=scale_mask))
        source_values = {"fit_mean": fit_mean, "scale_mean": float(means["scale_mean"][inference][0]),
                         "fit_mean_source": "parent purged fit row mean", "scale_mean_rows": int(scale_mask.sum()),
                         "scale_mean_label_end_exclusive": support_row["scale_end"]}
        support.append({"fold": fold, "inference_rows": int(inference.sum()), "score_rows": int(scoring.sum()),
            "regime": support_row["regime"], "validation_start": str(spec["val_start"]),
            "validation_end_exclusive": str(spec["val_end"]), **source_values})
        fold_rows, fold_scores, bindings = [], [], {}
        for cid in ("bh", "common_robust"):
            row = copy.deepcopy(ri[fold, cid])
            path = root/"targets"/f"fold{fold}_{cid}.npz"
            with np.load(path, allow_pickle=False) as saved:
                target = saved["targets"].copy()
                if not np.array_equal(saved["timestamps"], index.asi8) or np.any(np.isfinite(target) & ~inference):
                    raise ValueError("source control calendar/mask differs")
            for cost, contract in (("base", execution), ("stress_2x", stress)):
                replay = metrics(window, target, contract)
                for key, value in replay.items():
                    diff = abs(value-row[cost][key]); max_control_diff = max(max_control_diff, diff)
                    if not np.isclose(value, row[cost][key], rtol=1e-12, atol=1e-12):
                        raise ValueError("source control replay mismatch")
            dest = output/"targets"/path.name
            _copy_verified(path, dest, row["targets_sha256"]); bindings[str(dest)] = row["targets_sha256"]
            row["control_source_results_sha256"] = cfg["source_results_sha256"]
            fold_rows.append(row)
        for mid in MEANS:
            mu = means[mid]
            if np.any(np.isfinite(mu) & ~inference) or not np.isfinite(mu[inference]).all():
                raise ValueError("mean escaped frozen inference mask")
            pred_path = output/"forecasts"/f"fold{fold}_{mid}.npz"
            arrays = {"timestamps": index.asi8, "mu": mu, "variance": variance, "actual": actual,
                      "inference_mask": inference, "score_support": scoring, "fit_return_mean": np.asarray(fit_mean)}
            if pred_path.exists():
                with np.load(pred_path, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v, equal_nan=True) for k, v in arrays.items()):
                        raise ValueError("immutable mean forecast changed")
            else:
                pred_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(pred_path, **arrays)
            bindings[str(pred_path)] = file_digest(pred_path)
            fold_scores.append({"fold": fold, "mean_id": mid, "regime": support_row["regime"],
                **return_scores(actual=actual, mu=mu, score_mask=scoring, fit_mean=fit_mean),
                "forecast_sha256": bindings[str(pred_path)], "provenance": {**source_values,
                    "variance_source": VARIANCE_SOURCE, "variance_forecast_sha256": forecasts[risk_path.name],
                    "mean_forecast_sha256": forecasts.get(f"fold{fold}_{mid}.npz"),
                    "scale_calibration_sha256": calibrations[cal_path.name]}})
            for policy in POLICIES:
                cid = mid+"_"+policy
                targets, diagnostic = crossed_targets(window, mu, variance, execution,
                    policy=policy, cost_multiplier=cfg["utility_cost_multiplier"])
                if np.any(np.isfinite(targets) & ~inference):
                    raise ValueError("targets escaped inference mask")
                target_path = output/"targets"/f"fold{fold}_{cid}.npz"
                target_sha = _save_targets(target_path, index, targets); bindings[str(target_path)] = target_sha
                if diagnostic is not None:
                    trace_path = output/"traces"/f"fold{fold}_{cid}.json"
                    _immutable_json(trace_path, diagnostic); bindings[str(trace_path)] = file_digest(trace_path)
                    diagnostic = {k: v for k, v in diagnostic.items() if k != "decision_trace"}
                    diagnostic["trace_sha256"] = bindings[str(trace_path)]
                row = {"fold": fold, "candidate_id": cid, "regime": support_row["regime"],
                    "base": metrics(window, targets, execution), "stress_2x": metrics(window, targets, stress),
                    "targets_sha256": target_sha, "diagnostic": diagnostic,
                    "metadata": {**support[-1], "mean_source": mid, "variance_source": VARIANCE_SOURCE,
                        "mean_variance_forecast_sha256": bindings[str(pred_path)],
                        "variance_source_sha256": forecasts[risk_path.name]}}
                if mid == "zero" and policy == "point":
                    for cost in ("base", "stress_2x"):
                        for key, value in row[cost].items():
                            # target=1 and no-target B&H have different submitted-intent coverage.
                            if key == "intent_coverage":
                                continue
                            if not np.isclose(value, ri[fold, "bh"][cost][key], rtol=1e-12, atol=1e-12):
                                raise ValueError("zero-mean point does not reproduce B&H")
                if mid == "technical_scaled":
                    reference = ri[fold, cid]
                    with np.load(root/"targets"/target_path.name, allow_pickle=False) as saved:
                        if not np.array_equal(saved["targets"], targets, equal_nan=True):
                            raise ValueError("unchanged technical mean/risk orders differ")
                    for cost in ("base", "stress_2x"):
                        for key, value in row[cost].items():
                            diff = abs(value-reference[cost][key]); max_control_diff = max(max_control_diff, diff)
                            if not np.isclose(value, reference[cost][key], rtol=1e-12, atol=1e-12):
                                raise ValueError("unchanged technical control accounting differs")
                fold_rows.append(row)
        _immutable_json(output/f"fold_{fold}.json", {"registration_sha256": digest(registration),
            "rows": fold_rows, "scores": fold_scores, "artifact_sha256": bindings})
        rows.extend(fold_rows); scores.extend(fold_scores)
        print(json.dumps({"fold": fold, "policies": len(fold_rows), **source_values}), flush=True)
    if (sum(s["inference_rows"] for s in support) != cfg["inference_rows"]
            or sum(s["score_rows"] for s in support) != cfg["score_rows"]):
        raise ValueError("registered total support changed")
    result = {"registration_sha256": digest(registration), "rows": rows, "scores": scores,
        "summary": decision_summaries(rows, cfg["minimum_quarters_per_regime"]),
        "prediction_summary": prediction_summaries(scores), "paired": paired_summaries(scores, rows),
        "control_replay_max_metric_difference": max_control_diff, "support": support,
        "scope": registration["scope"], "selection_performed": False, "high_probability_generalization_established": False}
    _immutable_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
