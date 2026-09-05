"""Frozen-forecast comparison of hold versus B&H exposure when forecasts fail."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_search import digest, file_digest, fold_spec, load_bars, metrics, validate_data_artifact
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import (
    _copy_verified, _immutable_json, _save_targets, decision_summaries, load_paired_forecasts,
)
from .oracle_mean_control_decisions import FOLDS, MEANS, validate_config as validate_mean_config

CONTROL_IDS = ("bh", "common_robust") + tuple(m + "_utility_risk1" for m in MEANS)
NEW_IDS = tuple(m + "_utility_risk1_fallback_bh" for m in MEANS)
SOURCES = ("oracle_fallback_decisions.py", "oracle_fallback_planner.py",
           "oracle_mean_control_decisions.py", "oracle_conditional_planner.py",
           "oracle_derivative_crossed_decisions.py", "oracle_derivative_ablation.py",
           "oracle_frontier.py", "alpha_dd_search.py")


def validate_config(cfg):
    fixed = {"schema": "oracle-fallback-decisions-v1", "development_folds": list(FOLDS),
             "mean_sources": list(MEANS), "variance_source": "technical_scaled",
             "policy": "utility_risk1", "fallback_target": 1.0,
             "control_ids": list(CONTROL_IDS), "utility_cost_multiplier": 2,
             "minimum_quarters_per_regime": 3, "inference_rows": 2586, "score_rows": 2574,
             "fallback_eligible_rows": 332, "missing_current_open_rows": 2,
             "adaptive_prior_policy_names_observed": 147}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("unregistered forecast fallback family")


def action_masks(index, opens, inference):
    inference, opens = np.asarray(inference), np.asarray(opens, float)
    if (inference.dtype != bool or inference.shape != (len(index),)
            or opens.shape != (len(index),)):
        raise ValueError("invalid action support shapes")
    clock = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
    if np.any(inference & ~clock):
        raise ValueError("inference outside registered clock")
    known = np.isfinite(opens)
    return {"scheduled": clock, "learned_eligible": clock & known & inference,
            "fallback_eligible": clock & known & ~inference,
            "missing_current_open": clock & ~known}


def check_action_support(targets, masks):
    targets = np.asarray(targets, float)
    finite = np.isfinite(targets)
    allowed = masks["learned_eligible"] | masks["fallback_eligible"]
    if targets.shape != allowed.shape or np.isinf(targets).any() or np.any(finite & ~allowed):
        raise ValueError("order escaped causal action support")
    if np.any(targets[masks["fallback_eligible"]] != 1.0):
        raise ValueError("unregistered missing-forecast action")


def check_trace_support(targets, masks, diagnostic):
    expected_masks = {"learned": masks["learned_eligible"], "fallback": masks["fallback_eligible"],
                      "missing_open": masks["missing_current_open"],
                      "hold": masks["learned_eligible"] & np.isnan(targets)}
    for key, expected in expected_masks.items():
        count_key = "missing_open_decision_count" if key == "missing_open" else key + "_decision_count"
        if (not np.array_equal(np.asarray(diagnostic["decision_masks"][key]), expected)
                or diagnostic[count_key] != int(expected.sum())):
            raise ValueError("decision trace differs from causal action support")
    trace = diagnostic["decision_trace"]
    indices = np.flatnonzero(expected_masks["learned"] | expected_masks["fallback"])
    if trace["bar_indices"] != indices.tolist() or any(len(v) != len(indices) for v in trace.values()):
        raise ValueError("decision trace calendar changed")
    for j, i in enumerate(indices):
        fallback = expected_masks["fallback"][i]
        intent = None if np.isnan(targets[i]) else float(targets[i])
        if (trace["reasons"][j] != ("forecast_unavailable" if fallback else "learned")
                or trace["targets"][j] != intent
                or (fallback and (trace["estimated_utility_gain_over_hold"][j] is not None
                                  or trace["estimated_trade_turnover"][j] is not None))):
            raise ValueError("decision trace branch or intent changed")


def paired_summaries(rows):
    indexed = {(r["fold"], r["candidate_id"]): r for r in rows}
    expected = {(f, cid) for f in FOLDS for cid in CONTROL_IDS + NEW_IDS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("missing or duplicate registered fallback results")
    for f in FOLDS:
        if any(indexed[f, cid]["regime"] != indexed[f, "bh"]["regime"] for cid in CONTROL_IDS + NEW_IDS):
            raise ValueError("unpaired fallback regimes")
    result = {}
    for mean in MEANS:
        reference, candidate = mean + "_utility_risk1", mean + "_utility_risk1_fallback_bh"
        entry = {"candidate": candidate, "reference": reference,
                 "difference_convention": "fallback minus hold",
                 "aggregation": "equal quarter mean", "regimes": {}}
        for regime in ("all", "bull", "bear", "sideways"):
            folds = [f for f in FOLDS if regime == "all" or indexed[f, candidate]["regime"]["trend"] == regime]
            entry["regimes"][regime] = {"quarters": len(folds), **{
                cost: {metric: float(np.mean([
                    indexed[f, candidate][cost][metric] - indexed[f, reference][cost][metric] for f in folds]))
                    for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades",
                                   "fees_initial_equity_units", "borrow_initial_equity_units")}
                for cost in ("base", "stress_2x")}}
        result[candidate + "_vs_" + reference] = entry
    return result


def _verify_module_hashes(mapping):
    for name, expected in mapping.items():
        if file_digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"frozen source helper changed: {name}")


def load_sources(config_path):
    cfg = yaml.safe_load(config_path.read_text()); validate_config(cfg)
    root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    if root.resolve() == output.resolve() or (output / "results.json").exists():
        raise ValueError("immutable source or completed output")
    for name in ("registration", "results"):
        if file_digest(root / (name + ".json")) != cfg["source_" + name + "_sha256"]:
            raise ValueError(f"parent {name} changed")
    reg, parent = [json.loads((root / (n + ".json")).read_text()) for n in ("registration", "results")]
    if parent["registration_sha256"] != digest(reg):
        raise ValueError("parent result/registration mismatch")
    parent_cfg_path = Path(cfg["source_config"])
    if file_digest(parent_cfg_path) != reg["config_sha256"] or yaml.safe_load(parent_cfg_path.read_text()) != reg["config"]:
        raise ValueError("parent configuration changed")
    validate_mean_config(reg["config"]); _verify_module_hashes(reg["source_sha256"])
    fc = reg["source_config"]
    proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    if proof != reg["spot_data_proof"]:
        raise ValueError("Spot execution data proof changed")
    # Revalidate the pinned delay parent and its full 496-artifact lineage.
    delay_root = Path(reg["config"]["source_root"])
    for name in ("registration", "results", "preflight"):
        if file_digest(delay_root / (name + ".json")) != reg["config"]["source_" + name + "_sha256"]:
            raise ValueError("delay parent binding changed")
    delay_reg, delay_results, delay_pre = [json.loads((delay_root / (n + ".json")).read_text())
                                          for n in ("registration", "results", "preflight")]
    if (delay_results["registration_sha256"] != digest(delay_reg)
            or delay_reg["source_config"] != fc or delay_pre["spot_data_proof"] != proof
            or delay_reg["preflight_sha256"] != reg["config"]["source_preflight_sha256"]
            or digest({k: v for k, v in delay_reg["config"].items() if k != "preflight_sha256"})
               != delay_pre["config_contract_sha256"]):
        raise ValueError("delay source execution binding changed")
    delay_cfg_path = Path(reg["config"]["source_config"])
    if (file_digest(delay_cfg_path) != delay_reg["config_sha256"]
            or yaml.safe_load(delay_cfg_path.read_text()) != delay_reg["config"]):
        raise ValueError("delay configuration changed")
    _verify_module_hashes(delay_reg["source_sha256"]); _verify_module_hashes(delay_pre["source_sha256"])
    execution_path = Path(delay_reg["config"]["source_config"])
    if (file_digest(execution_path) != delay_pre["source_config_sha256"]
            or yaml.safe_load(execution_path.read_text()) != fc):
        raise ValueError("original execution configuration changed")
    all_artifacts = {}
    for directory, registration, results in ((delay_root, delay_reg, delay_results), (root, reg, parent)):
        directory_artifacts = {}
        for fold in FOLDS:
            path = directory / f"fold_{fold}.json"
            if directory == root and file_digest(path) != cfg["source_fold_sha256"][path.name]:
                raise ValueError("registered mean-control manifest changed")
            saved = json.loads(path.read_text())
            if (saved["registration_sha256"] != digest(registration)
                    or saved["rows"] != [r for r in results["rows"] if r["fold"] == fold]
                    or saved["scores"] != [s for s in results["scores"] if s["fold"] == fold]):
                raise ValueError("parent fold/result mismatch")
            for name, expected in saved["artifact_sha256"].items():
                if file_digest(Path(name)) != expected:
                    raise ValueError("parent manifest artifact changed")
                all_artifacts[name] = expected
                directory_artifacts[name] = expected
        if len(directory_artifacts) != (496 if directory == delay_root else 240):
            raise ValueError("incomplete source artifact inventory")
    si = {(s["fold"], s["mean_id"]): s for s in parent["scores"]}
    ri = {(r["fold"], r["candidate_id"]): r for r in parent["rows"]}
    if (len(si) != len(parent["scores"]) or set(si) != {(f, m) for f in FOLDS for m in MEANS}
            or len(ri) != len(parent["rows"]) or len(ri) != 128):
        raise ValueError("incomplete parent results")
    for fold in FOLDS:
        for mean in MEANS:
            p = root / "forecasts" / f"fold{fold}_{mean}.npz"
            if all_artifacts[str(p)] != si[fold, mean]["forecast_sha256"]:
                raise ValueError("forecast/score binding mismatch")
        for cid in CONTROL_IDS:
            p = root / "targets" / f"fold{fold}_{cid}.npz"
            if all_artifacts[str(p)] != ri[fold, cid]["targets_sha256"]:
                raise ValueError("control target binding mismatch")
    return cfg, fc, reg, parent, si, ri, proof, all_artifacts


def load_fold(root, bars, fc, fold):
    spec = fold_spec(fold, fc["fold_anchor"])
    index = pd.date_range(spec["val_start"], spec["val_end"], freq="15min", inclusive="left")
    if spec["val_end"] > pd.Timestamp(fc["data_cutoff"]) or not index.isin(bars.index).all():
        raise ValueError("invalid fixed validation calendar")
    window = bars.loc[index]
    if window.bar_available.mean() < fc["minimum_bar_coverage"]:
        raise ValueError("incomplete execution price support")
    risk_path = root / "forecasts" / f"fold{fold}_technical_scaled.npz"
    means = {}
    for mean in MEANS:
        means[mean], variance, inference, scoring = load_paired_forecasts(
            root / "forecasts" / f"fold{fold}_{mean}.npz", risk_path, expected_index=index)
        with np.load(root / "forecasts" / f"fold{fold}_{mean}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["variance"], variance, equal_nan=True):
                raise ValueError("mean-control risk stream differs from fixed risk")
    return window, means, variance, inference, scoring


def preflight(config_path, loaded=None):
    cfg, fc, reg, parent, si, ri, proof, artifacts = loaded or load_sources(config_path)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    support, parent_support = [], {s["fold"]: s for s in parent["support"]}
    for fold in FOLDS:
        window, means, variance, inference, scoring = load_fold(Path(cfg["source_root"]), bars, fc, fold)
        masks = action_masks(window.index, window.open.to_numpy(), inference)
        next_missing = ~np.isfinite(np.r_[window.open.to_numpy()[1:], np.nan])
        item = {"fold": fold, "regime": parent_support[fold]["regime"],
                "validation_start": str(window.index[0]),
                "validation_end_exclusive": str(window.index[-1] + pd.Timedelta(minutes=15)),
                "inference_rows": int(inference.sum()), "score_rows": int(scoring.sum()),
                "action_support_counts": {k: int(v.sum()) for k, v in masks.items()},
                "mask_sha256": {k: mask_digest(window.index, v) for k, v in {**masks, "inference": inference, "score": scoring}.items()},
                "fallback_slots_missing_next_open_descriptive_only": int(np.sum(masks["fallback_eligible"] & next_missing))}
        if any(item[k] != parent_support[fold][k] for k in ("inference_rows", "score_rows")):
            raise ValueError("inherited forecast support changed")
        support.append(item)
    if (sum(s["inference_rows"] for s in support) != cfg["inference_rows"]
            or sum(s["score_rows"] for s in support) != cfg["score_rows"]
            or sum(s["action_support_counts"]["fallback_eligible"] for s in support) != cfg["fallback_eligible_rows"]
            or sum(s["action_support_counts"]["missing_current_open"] for s in support) != cfg["missing_current_open_rows"]):
        raise ValueError("registered forecast totals changed")
    return {"config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
            "source_registration_sha256": cfg["source_registration_sha256"],
            "source_results_sha256": cfg["source_results_sha256"], "spot_data_proof": proof,
            "source_sha256": {n: file_digest(Path(__file__).with_name(n)) for n in SOURCES},
            "verified_source_artifacts": len(artifacts), "support": support,
            "future_fill_availability_used_for_actions": False,
            "new_policies_computed": False, "selection_performed": False,
            "scope": "data-only forecast versus action availability; retrospective inherited comparison mask"}


def run(config_path):
    from .oracle_fallback_planner import fallback_targets
    loaded = load_sources(config_path)
    cfg, fc, parent_reg, parent, si, ri, proof, artifacts = loaded
    root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    pre = preflight(config_path, loaded)
    if (file_digest(output / "preflight.json") != cfg["preflight_sha256"]
            or json.loads((output / "preflight.json").read_text()) != pre):
        raise ValueError("registered data-only preflight changed")
    registration = {"config": cfg, "config_sha256": file_digest(config_path), "source_config": fc,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": pre["source_sha256"], "spot_data_proof": proof,
        "preflight_sha256": cfg["preflight_sha256"],
        "source_artifact_sha256": artifacts,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "scope": "adaptive missing-forecast inventory rule on reused validation; no fitting or selection"}
    _immutable_json(output / "registration.json", registration)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": 2 * execution["one_way_cost"], "borrow_annual": 2 * execution["borrow_annual"]}
    support_by_fold = {s["fold"]: s for s in pre["support"]}
    rows, max_control_diff = [], 0.
    for fold in FOLDS:
        window, means, variance, inference, scoring = load_fold(root, bars, fc, fold)
        masks = action_masks(window.index, window.open.to_numpy(), inference)
        fold_rows, bindings = [], {}
        for cid in CONTROL_IDS:
            row = copy.deepcopy(ri[fold, cid]); path = root / "targets" / f"fold{fold}_{cid}.npz"
            with np.load(path, allow_pickle=False) as saved:
                target = saved["targets"].copy()
                if not np.array_equal(saved["timestamps"], window.index.asi8) or np.any(np.isfinite(target) & ~inference):
                    raise ValueError("control target calendar/support changed")
            for cost, contract in (("base", execution), ("stress_2x", stress)):
                for key, value in metrics(window, target, contract).items():
                    diff = abs(value - row[cost][key]); max_control_diff = max(max_control_diff, diff)
                    if not np.isclose(value, row[cost][key], rtol=1e-12, atol=1e-12):
                        raise ValueError("frozen control replay changed")
            dst = output / "targets" / path.name
            _copy_verified(path, dst, row["targets_sha256"]); bindings[str(dst)] = row["targets_sha256"]
            row["control_source_results_sha256"] = cfg["source_results_sha256"]
            if row.get("diagnostic") and "trace_sha256" in row["diagnostic"]:
                trace = root / "traces" / f"fold{fold}_{cid}.json"
                row["control_trace_source"] = {"path": str(trace), "sha256": file_digest(trace)}
                if row["control_trace_source"]["sha256"] != row["diagnostic"]["trace_sha256"]:
                    raise ValueError("control trace binding changed")
            fold_rows.append(row)
        for mean in MEANS:
            cid = mean + "_utility_risk1_fallback_bh"
            targets, diagnostic = fallback_targets(window, means[mean], variance, execution,
                inference_mask=inference, risk_aversion=1, cost_multiplier=cfg["utility_cost_multiplier"])
            check_action_support(targets, masks)
            check_trace_support(targets, masks, diagnostic)
            target_path = output / "targets" / f"fold{fold}_{cid}.npz"
            target_sha = _save_targets(target_path, window.index, targets); bindings[str(target_path)] = target_sha
            trace_path = output / "traces" / f"fold{fold}_{cid}.json"
            _immutable_json(trace_path, diagnostic); bindings[str(trace_path)] = file_digest(trace_path)
            compact = {k: v for k, v in diagnostic.items() if k not in ("decision_trace", "decision_masks")}
            compact["trace_sha256"] = bindings[str(trace_path)]
            row = {"fold": fold, "candidate_id": cid, "regime": support_by_fold[fold]["regime"],
                "base": metrics(window, targets, execution), "stress_2x": metrics(window, targets, stress),
                "targets_sha256": target_sha, "diagnostic": compact,
                "metadata": {"mean_source": mean, "variance_source": "technical_scaled",
                    "source_forecast_sha256": si[fold, mean]["forecast_sha256"],
                    "source_forecast_path": str(root / "forecasts" / f"fold{fold}_{mean}.npz"),
                    "fallback_target": 1.0, "forecast_support_unchanged": True,
                    "action_support": support_by_fold[fold]}}
            if not masks["fallback_eligible"].any():
                reference = ri[fold, mean + "_utility_risk1"]
                with np.load(root / "targets" / f"fold{fold}_{mean}_utility_risk1.npz", allow_pickle=False) as saved:
                    if not np.array_equal(saved["targets"], targets, equal_nan=True):
                        raise ValueError("no-fallback fold changed parent targets")
                if any(row[c] != reference[c] for c in ("base", "stress_2x")):
                    raise ValueError("no-fallback fold changed parent economics")
                row["unchanged_parent_on_zero_fallback_verified"] = True
            fold_rows.append(row)
        _immutable_json(output / f"fold_{fold}.json", {"registration_sha256": digest(registration),
            "rows": fold_rows, "artifact_sha256": bindings})
        rows.extend(fold_rows)
        print(json.dumps({"fold": fold, "policies": len(fold_rows),
                          "action_support_counts": support_by_fold[fold]["action_support_counts"]}), flush=True)
    paired = paired_summaries(rows)
    summary = decision_summaries(rows, cfg["minimum_quarters_per_regime"])
    for values in summary.values():
        values["observed_regime_mean_signs_pass"] = bool(all(
            entry["quarters"] > 0 and all(entry[c]["alpha_ex_mean"] > 0 and entry[c]["maxdd_delta_mean"] < 0
            for c in ("base", "stress_2x")) for entry in values["regimes"].values()))
    result = {"registration_sha256": digest(registration), "rows": rows, "summary": summary,
        "paired": paired, "support": pre["support"], "control_replay_max_metric_difference": max_control_diff,
        "scope": registration["scope"], "selection_performed": False,
        "forecast_metrics_unchanged_source_results_sha256": cfg["source_results_sha256"],
        "high_probability_generalization_established": False}
    _immutable_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="verify inputs and save data-only support")
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.config)
        cfg = yaml.safe_load(args.config.read_text())
        _immutable_json(Path(cfg["output_dir"]) / "preflight.json", result)
        print(json.dumps(result, indent=2))
    else:
        run(args.config)
