"""Fixed half-weight mean forecasts with frozen hold and fallback controllers."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import yaml

from .alpha_dd_search import digest, file_digest, load_bars, metrics, validate_data_artifact
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_crossed_decisions import _copy_verified, _immutable_json, _save_targets, decision_summaries
from .oracle_fallback_decisions import (
    action_masks, check_action_support, check_trace_support, load_fold,
    validate_config as validate_fallback_config,
)
from .oracle_fallback_planner import fallback_targets
from .oracle_mean_controls import return_scores

FOLDS = tuple(range(5, 13))
ENDPOINTS = ("scale_mean", "technical_scaled", "perp_delay0_scaled")
HALVES = {"technical_half": "technical_scaled", "perp_delay0_half": "perp_delay0_scaled"}
RULES = ("utility_risk1", "utility_risk1_fallback_bh")
CONTROL_IDS = ("bh", "common_robust") + tuple(m + "_" + rule for m in ENDPOINTS for rule in RULES)
NEW_IDS = tuple(m + "_" + rule for m in HALVES for rule in RULES)
PREDICTION_PAIRS = tuple((m, ref) for m, own in HALVES.items() for ref in ("scale_mean", own)) + (("perp_delay0_half", "technical_half"),)
SOURCES = ("oracle_mean_shrinkage_decisions.py", "oracle_mean_shrinkage.py", "oracle_mean_controls.py",
           "oracle_fallback_decisions.py", "oracle_fallback_planner.py", "oracle_conditional_planner.py",
           "oracle_derivative_crossed_decisions.py", "oracle_derivative_ablation.py", "oracle_frontier.py", "alpha_dd_search.py")


def validate_config(cfg):
    fixed = {"schema": "oracle-mean-shrinkage-decisions-v1", "development_folds": list(FOLDS),
             "half_sources": HALVES, "anchor_source": "scale_mean", "weight": .5,
             "variance_source": "technical_scaled", "rules": list(RULES), "control_ids": list(CONTROL_IDS),
             "utility_cost_multiplier": 2, "minimum_quarters_per_regime": 3,
             "inference_rows": 2586, "score_rows": 2574, "fallback_eligible_rows": 332,
             "missing_current_open_rows": 2, "adaptive_prior_policy_names_observed": 154}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("unregistered half-mean family")


def _verify_sources(mapping):
    for name, expected in mapping.items():
        if file_digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"frozen source changed: {name}")


def _bound_json(root, name, sha):
    path = root / (name + ".json")
    if file_digest(path) != sha:
        raise ValueError(f"frozen {name} changed")
    return json.loads(path.read_text())


def load_sources(config_path):
    cfg = yaml.safe_load(config_path.read_text()); validate_config(cfg)
    root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    if root.resolve() == output.resolve() or (output / "results.json").exists():
        raise ValueError("immutable source or completed output")
    reg, parent = [_bound_json(root, n, cfg["source_" + n + "_sha256"]) for n in ("registration", "results")]
    if parent["registration_sha256"] != digest(reg):
        raise ValueError("parent result registration changed")
    parent_cfg = Path(cfg["source_config"])
    if file_digest(parent_cfg) != reg["config_sha256"] or yaml.safe_load(parent_cfg.read_text()) != reg["config"]:
        raise ValueError("parent configuration changed")
    validate_fallback_config(reg["config"]); _verify_sources(reg["source_sha256"])
    pre = _bound_json(root, "preflight", reg["preflight_sha256"])
    if (reg["config"]["preflight_sha256"] != reg["preflight_sha256"]
            or pre["source_sha256"] != reg["source_sha256"]
            or digest({k: v for k, v in reg["config"].items() if k != "preflight_sha256"}) != pre["config_contract_sha256"]):
        raise ValueError("parent preflight changed")
    fc = reg["source_config"]
    proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    if proof != reg["spot_data_proof"] or pre["spot_data_proof"] != proof:
        raise ValueError("Spot data proof changed")
    artifacts = dict(reg["source_artifact_sha256"])
    if len(artifacts) != 736:
        raise ValueError("incomplete ancestral artifact inventory")
    for path, sha in artifacts.items():
        if file_digest(Path(path)) != sha:
            raise ValueError("ancestral artifact changed")
    current_artifacts = {}
    for fold in FOLDS:
        path = root / f"fold_{fold}.json"
        if file_digest(path) != cfg["source_fold_sha256"][path.name]:
            raise ValueError("registered parent manifest changed")
        saved = json.loads(path.read_text())
        if (saved["registration_sha256"] != digest(reg)
                or saved["rows"] != [r for r in parent["rows"] if r["fold"] == fold]):
            raise ValueError("parent fold result mismatch")
        for name, sha in saved["artifact_sha256"].items():
            if file_digest(Path(name)) != sha:
                raise ValueError("parent artifact changed")
            current_artifacts[name] = sha
    if len(current_artifacts) != 184:
        raise ValueError("incomplete fallback artifact inventory")
    artifacts.update(current_artifacts)
    # Recheck configuration and provenance of the frozen mean and delay lineage.
    mean_root = Path(reg["config"]["source_root"])
    mean_reg, mean_results = [_bound_json(mean_root, n, reg["config"]["source_" + n + "_sha256"])
                              for n in ("registration", "results")]
    mean_cfg = Path(reg["config"]["source_config"])
    if (mean_results["registration_sha256"] != digest(mean_reg) or mean_reg["source_config"] != fc
            or file_digest(mean_cfg) != mean_reg["config_sha256"]
            or yaml.safe_load(mean_cfg.read_text()) != mean_reg["config"]):
        raise ValueError("mean source configuration changed")
    _verify_sources(mean_reg["source_sha256"])
    for fold in FOLDS:
        if file_digest(mean_root / f"fold_{fold}.json") != reg["config"]["source_fold_sha256"][f"fold_{fold}.json"]:
            raise ValueError("mean manifest changed")
    delay_root = Path(mean_reg["config"]["source_root"])
    dr, dp = [_bound_json(delay_root, n, mean_reg["config"]["source_" + n + "_sha256"])
              for n in ("registration", "preflight")]
    _bound_json(delay_root, "results", mean_reg["config"]["source_results_sha256"])
    if (dr["source_config"] != fc or dp["spot_data_proof"] != proof
            or dr["preflight_sha256"] != mean_reg["config"]["source_preflight_sha256"]
            or digest({k: v for k, v in dr["config"].items() if k != "preflight_sha256"}) != dp["config_contract_sha256"]):
        raise ValueError("delay source provenance changed")
    for path, sha, expected in ((Path(mean_reg["config"]["source_config"]), dr["config_sha256"], dr["config"]),
            (Path(dr["config"]["source_config"]), dp["source_config_sha256"], fc)):
        if file_digest(path) != sha or yaml.safe_load(path.read_text()) != expected:
            raise ValueError("frozen execution configuration changed")
    _verify_sources(dr["source_sha256"]); _verify_sources(dp["source_sha256"])
    si = {(s["fold"], s["mean_id"]): s for s in mean_results["scores"]}
    ri = {(r["fold"], r["candidate_id"]): r for r in parent["rows"]}
    if len(si) != 56 or len(si) != len(mean_results["scores"]) or len(ri) != 128 or len(ri) != len(parent["rows"]):
        raise ValueError("duplicate or incomplete parent results")
    for fold in FOLDS:
        for mean in ENDPOINTS:
            p = mean_root / "forecasts" / f"fold{fold}_{mean}.npz"
            if artifacts[str(p)] != si[fold, mean]["forecast_sha256"]:
                raise ValueError("endpoint forecast score binding changed")
        for cid in CONTROL_IDS:
            p = root / "targets" / f"fold{fold}_{cid}.npz"
            if artifacts[str(p)] != ri[fold, cid]["targets_sha256"]:
                raise ValueError("control targets changed")
    return cfg, fc, reg, parent, mean_root, si, ri, proof, artifacts, pre


def preflight(config_path, loaded=None):
    cfg, fc, reg, parent, mean_root, si, ri, proof, artifacts, parent_pre = loaded or load_sources(config_path)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    anchors = []
    for fold in FOLDS:
        window, means, variance, inference, scoring = load_fold(mean_root, bars, fc, fold)
        selected = means["scale_mean"][inference]
        if not len(selected) or not np.all(selected == selected[0]):
            raise ValueError("saved anchor is not a constant")
        support = next(x for x in parent_pre["support"] if x["fold"] == fold)
        masks = action_masks(window.index, window.open.to_numpy(), inference)
        from .oracle_derivative_ablation import mask_digest
        for key, mask in {**masks, "inference": inference, "score": scoring}.items():
            if mask_digest(window.index, mask) != support["mask_sha256"][key]:
                raise ValueError("inherited support mask changed")
        anchors.append({"fold": fold, "scale_mean": float(selected[0]),
                        "scale_mean_forecast_sha256": si[fold, "scale_mean"]["forecast_sha256"]})
    support = parent_pre["support"]
    if any(sum(s["action_support_counts"][key] for s in support) != cfg[target]
           for key, target in (("fallback_eligible", "fallback_eligible_rows"), ("missing_current_open", "missing_current_open_rows"))):
        raise ValueError("action support totals changed")
    if any(sum(s[key] for s in support) != cfg[key] for key in ("inference_rows", "score_rows")):
        raise ValueError("forecast totals changed")
    return {"config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
        "source_registration_sha256": cfg["source_registration_sha256"], "source_results_sha256": cfg["source_results_sha256"],
        "source_sha256": {n: file_digest(Path(__file__).with_name(n)) for n in SOURCES},
        "spot_data_proof": proof, "support": support, "anchor_provenance": anchors,
        "verified_source_artifacts": len(artifacts), "new_forecast_or_policy_computed": False,
        "scope": "data-only frozen endpoint geometry and provenance; no half predictions or losses"}


def compare(scores, rows):
    si = {(s["fold"], s["mean_id"]): s for s in scores}
    ri = {(r["fold"], r["candidate_id"]): r for r in rows}
    if (len(si) != len(scores) or set(si) != {(f, m) for f in FOLDS for m in ENDPOINTS + tuple(HALVES)}
            or len(ri) != len(rows) or set(ri) != {(f, cid) for f in FOLDS for cid in CONTROL_IDS + NEW_IDS}):
        raise ValueError("incomplete registered shrinkage inventory")
    for fold in FOLDS:
        anchor = si[fold, "scale_mean"]
        if (any(si[fold, m]["regime"] != anchor["regime"] or si[fold, m]["rows"] != anchor["rows"] for m in ENDPOINTS + tuple(HALVES))
                or any(ri[fold, cid]["regime"] != anchor["regime"] for cid in CONTROL_IDS + NEW_IDS)):
            raise ValueError("unpaired shrinkage support")
    pairs = {}
    for candidate, reference in PREDICTION_PAIRS:
        entry = {"candidate": candidate, "reference": reference,
                 "difference_convention": "candidate minus reference", "regimes": {}}
        for regime in ("all", "bull", "bear", "sideways"):
            folds = [f for f in FOLDS if regime == "all" or si[f, candidate]["regime"]["trend"] == regime]
            pred = {}
            for metric in ("return_mse", "return_mae"):
                delta = [si[f, candidate][metric] - si[f, reference][metric] for f in folds]
                reference_loss = float(np.mean([si[f, reference][metric] for f in folds]))
                pred[metric] = {"mean_difference": float(np.mean(delta)),
                    "relative_loss_reduction": -float(np.mean(delta)) / reference_loss if reference_loss > 0 else None,
                    "improved_quarters": sum(d < 0 for d in delta)}
            entry["regimes"][regime] = {"quarters": len(folds), "prediction": pred,
                "policies": {rule: {cost: {metric: float(np.mean([
                    ri[f, candidate + "_" + rule][cost][metric] - ri[f, reference + "_" + rule][cost][metric] for f in folds]))
                    for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades", "fees_initial_equity_units", "borrow_initial_equity_units")}
                    for cost in ("base", "stress_2x")} for rule in RULES}}
        pairs[candidate + "_vs_" + reference] = entry
    predictions = {}
    for mean in ENDPOINTS + tuple(HALVES):
        predictions[mean] = {}
        for regime in ("all", "bull", "bear", "sideways"):
            selected = [s for s in scores if s["mean_id"] == mean and (regime == "all" or s["regime"]["trend"] == regime)]
            n = sum(s["rows"] for s in selected)
            predictions[mean][regime] = {"quarters": len(selected), "rows": n,
                "metrics": {metric: {"equal_quarter_mean": float(np.mean([s[metric] for s in selected])),
                    "pooled_row_mean": float(sum(s[metric] * s["rows"] for s in selected) / n)}
                    for metric in ("return_mse", "return_mae", "return_sign_accuracy", "zero_return_mse", "fit_mean_return_mse")},
                "mean_return_rank_ic": float(np.mean([s["return_rank_ic"] for s in selected]))
                    if all(s["return_rank_ic"] is not None for s in selected) else None}
    rules = {}
    for mean in HALVES:
        entry = {"difference_convention": "fallback minus hold", "regimes": {}}
        for regime in ("all", "bull", "bear", "sideways"):
            folds = [f for f in FOLDS if regime == "all" or si[f, mean]["regime"]["trend"] == regime]
            entry["regimes"][regime] = {"quarters": len(folds), **{cost: {metric: float(np.mean([
                ri[f, mean + "_" + RULES[1]][cost][metric] - ri[f, mean + "_" + RULES[0]][cost][metric] for f in folds]))
                for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades", "fees_initial_equity_units", "borrow_initial_equity_units")}
                for cost in ("base", "stress_2x")}}
        rules[mean] = entry
    return pairs, predictions, rules


def run(config_path):
    from .oracle_mean_shrinkage import half_mean
    loaded = load_sources(config_path)
    cfg, fc, parent_reg, parent, mean_root, si, ri, proof, artifacts, parent_pre = loaded
    root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    pre = preflight(config_path, loaded)
    if (file_digest(output / "preflight.json") != cfg["preflight_sha256"]
            or json.loads((output / "preflight.json").read_text()) != pre):
        raise ValueError("registered preflight changed")
    registration = {"config": cfg, "config_sha256": file_digest(config_path), "source_config": fc,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "preflight_sha256": cfg["preflight_sha256"], "source_sha256": pre["source_sha256"],
        "source_artifact_sha256": artifacts, "spot_data_proof": proof,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "scope": "adaptive fixed half-mean diagnostic on reused validation; no fitting or selection"}
    _immutable_json(output / "registration.json", registration)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": 2 * execution["one_way_cost"], "borrow_annual": 2 * execution["borrow_annual"]}
    rows, scores, max_control_diff = [], [], 0.
    for fold in FOLDS:
        window, means, variance, inference, scoring = load_fold(mean_root, bars, fc, fold)
        masks = action_masks(window.index, window.open.to_numpy(), inference)
        support = next(x for x in pre["support"] if x["fold"] == fold)
        with np.load(mean_root / "forecasts" / f"fold{fold}_scale_mean.npz", allow_pickle=False) as saved:
            actual, fitted = saved["actual"].copy(), float(saved["fit_return_mean"])
        fold_rows, fold_scores, bindings = [], [], {}
        for mean in ENDPOINTS:
            score = copy.deepcopy(si[fold, mean])
            if any(not np.isclose(v, score[k], rtol=1e-12, atol=1e-12) if v is not None else score[k] is not None
                   for k, v in return_scores(actual, means[mean], scoring, fitted).items()):
                raise ValueError("endpoint forecast score changed")
            fold_scores.append(score)
        for cid in CONTROL_IDS:
            row = copy.deepcopy(ri[fold, cid]); path = root / "targets" / f"fold{fold}_{cid}.npz"
            with np.load(path, allow_pickle=False) as saved:
                target = saved["targets"].copy()
                if not np.array_equal(saved["timestamps"], window.index.asi8):
                    raise ValueError("endpoint control calendar changed")
            for cost, contract in (("base", execution), ("stress_2x", stress)):
                for key, value in metrics(window, target, contract).items():
                    difference = abs(value - row[cost][key]); max_control_diff = max(max_control_diff, difference)
                    if not np.isclose(value, row[cost][key], rtol=1e-12, atol=1e-12):
                        raise ValueError("endpoint control economics changed")
            dst = output / "targets" / path.name
            _copy_verified(path, dst, row["targets_sha256"]); bindings[str(dst)] = row["targets_sha256"]
            row["immediate_control_source_results_sha256"] = cfg["source_results_sha256"]
            fold_rows.append(row)
        for mean, learned in HALVES.items():
            mu = half_mean(means[learned], means["scale_mean"], inference_mask=inference)
            path = output / "forecasts" / f"fold{fold}_{mean}.npz"
            arrays = {"timestamps": window.index.asi8, "mu": mu, "variance": variance,
                      "inference_mask": inference, "score_support": scoring, "actual": actual,
                      "fit_return_mean": np.asarray(fitted)}
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v, equal_nan=True) for k, v in arrays.items()):
                        raise ValueError("immutable half forecast changed")
            else:
                path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
            bindings[str(path)] = file_digest(path)
            provenance = {"anchor_source": "scale_mean", "anchor_forecast_sha256": si[fold, "scale_mean"]["forecast_sha256"],
                "learned_source": learned, "learned_forecast_sha256": si[fold, learned]["forecast_sha256"],
                "anchor_weight": .5, "learned_weight": .5, "variance_source": "technical_scaled",
                "variance_forecast_sha256": si[fold, "technical_scaled"]["forecast_sha256"]}
            fold_scores.append({"fold": fold, "mean_id": mean, "regime": support["regime"],
                **return_scores(actual, mu, scoring, fitted), "forecast_sha256": bindings[str(path)], "provenance": provenance})
            generated = {}
            for rule in RULES:
                if rule.endswith("fallback_bh"):
                    targets, diagnostic = fallback_targets(window, mu, variance, execution, inference_mask=inference,
                        risk_aversion=1, cost_multiplier=cfg["utility_cost_multiplier"])
                    check_action_support(targets, masks); check_trace_support(targets, masks, diagnostic)
                else:
                    targets, diagnostic = conditional_targets(window, mu, variance, execution,
                        risk_aversion=1, cost_multiplier=cfg["utility_cost_multiplier"])
                    if np.isinf(targets).any() or np.any(np.isfinite(targets) & ~masks["learned_eligible"]):
                        raise ValueError("hold orders escaped inference support")
                cid = mean + "_" + rule
                target_path = output / "targets" / f"fold{fold}_{cid}.npz"
                sha = _save_targets(target_path, window.index, targets); bindings[str(target_path)] = sha
                trace_path = output / "traces" / f"fold{fold}_{cid}.json"
                _immutable_json(trace_path, diagnostic); bindings[str(trace_path)] = file_digest(trace_path)
                compact = {k: v for k, v in diagnostic.items() if k not in ("decision_trace", "decision_masks")}
                compact["trace_sha256"] = bindings[str(trace_path)]
                row = {"fold": fold, "candidate_id": cid, "regime": support["regime"],
                    "base": metrics(window, targets, execution), "stress_2x": metrics(window, targets, stress),
                    "targets_sha256": sha, "diagnostic": compact,
                    "metadata": {**provenance, "derived_forecast_sha256": bindings[str(path)], "action_support": support}}
                generated[rule] = (targets, row)
                fold_rows.append(row)
            if not masks["fallback_eligible"].any():
                a, b = [generated[rule] for rule in RULES]
                if not np.array_equal(a[0], b[0], equal_nan=True) or any(a[1][c] != b[1][c] for c in ("base", "stress_2x")):
                    raise ValueError("zero-fallback half controllers disagree")
        _immutable_json(output / f"fold_{fold}.json", {"registration_sha256": digest(registration),
            "rows": fold_rows, "scores": fold_scores, "artifact_sha256": bindings})
        rows.extend(fold_rows); scores.extend(fold_scores)
        print(json.dumps({"fold": fold, "policies": len(fold_rows), "forecast_scores": len(fold_scores)}), flush=True)
    paired, predictions, rules = compare(scores, rows)
    summary = decision_summaries(rows, cfg["minimum_quarters_per_regime"])
    for value in summary.values():
        value["observed_regime_mean_signs_pass"] = bool(all(g["quarters"] > 0 and all(
            g[c]["alpha_ex_mean"] > 0 and g[c]["maxdd_delta_mean"] < 0 for c in ("base", "stress_2x")) for g in value["regimes"].values()))
    result = {"registration_sha256": digest(registration), "rows": rows, "scores": scores,
        "summary": summary, "prediction_summary": predictions, "paired": paired, "availability_paired": rules,
        "support": pre["support"],
        "control_replay_max_metric_difference": max_control_diff, "selection_performed": False,
        "high_probability_generalization_established": False, "scope": registration["scope"]}
    _immutable_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.config); cfg = yaml.safe_load(args.config.read_text())
        _immutable_json(Path(cfg["output_dir"]) / "preflight.json", result)
        print(json.dumps(result, indent=2))
    else:
        run(args.config)
