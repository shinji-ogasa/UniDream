"""Fixed report-only replay on the reused original test folds 15 through 24.

No ranking, tuning, selection, inferential p-values or prospective receipt
claims. A missing fold fails the family rather than changing its inventory.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from .alpha_dd_search import digest, file_digest, load_bars, metrics, validate_data_artifact
from .oracle_confirmation_contract import calendar, segment_masks
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import mask_digest, validate_um
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_derivative_delay_features import make_delayed_perp_groups
from .oracle_derivative_features import make_derivative_groups
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frontier import outcome_frame, quarter_regime
from .oracle_frontier_features import make_feature_groups
from .oracle_frozen_forecasts import fit_frozen_forecasts
from .oracle_frozen_procedure_parity import compare_array, compare_tree
from .oracle_mean_controls import return_scores
from .oracle_risk_calibration import trailing_variances
from .robust_overlay import build_targets


FOLDS = tuple(range(15, 25))
MEANS = ("scale_mean", "technical_scaled", "perp_delay0_scaled", "technical_half", "perp_delay0_half")
RULES = ("utility_risk1", "utility_risk1_fallback_bh")
POLICIES = ("bh", "common_robust") + tuple(m + "_" + r for m in MEANS for r in RULES)
CANDIDATES = tuple(m + "_" + r for m in MEANS[3:] for r in RULES)
CONTROLS = tuple(cid for cid in POLICIES if cid not in CANDIDATES)
STRATA = ("all", "bull", "bear", "sideways")
COSTS = ("base", "stress_2x")
PAIRS = (("technical_half", "scale_mean"), ("technical_half", "technical_scaled"),
         ("perp_delay0_half", "scale_mean"), ("perp_delay0_half", "perp_delay0_scaled"),
         ("perp_delay0_half", "technical_half"))
ECON_KEYS = ("alpha_ex", "maxdd_delta", "turnover", "trades", "fees_initial_equity_units", "borrow_initial_equity_units")
SOURCES = tuple("unidream/experiments/" + name for name in (
    "oracle_additional_window_replay.py", "oracle_frozen_forecasts.py", "oracle_frozen_procedure_parity.py",
    "oracle_confirmation_contract.py", "oracle_mean_controls.py", "oracle_mean_shrinkage.py",
    "oracle_fallback_planner.py", "oracle_fallback_decisions.py", "oracle_conditional_planner.py",
    "oracle_derivative_delay.py", "oracle_derivative_delay_features.py", "oracle_derivative_features.py",
    "oracle_derivative_ablation.py", "oracle_derivative_crossed_decisions.py", "oracle_frontier.py",
    "oracle_frontier_features.py", "oracle_risk_calibration.py", "alpha_dd_search.py",
    "alpha_dd_features.py", "robust_overlay.py"))
EXECUTION = {"one_way_cost": .00055, "borrow_annual": .10, "max_step": .08, "deadband": .01}
FIXED = {"schema": "oracle-additional-window-replay-v1", "evaluation_folds": list(FOLDS),
    "evaluation_split": "original_test_report_only", "fit_months": 18, "scale_months": 3,
    "interval_months": 3, "evaluation_months": 3, "horizon_bars": 24,
    "minimum_fit_rows": 512, "minimum_scale_rows": 64, "minimum_interval_rows": 64,
    "minimum_score_rows": 16, "minimum_bar_coverage": .995, "minimum_quarters_per_regime": 3,
    "mean_ids": list(MEANS), "policy_ids": list(POLICIES), "candidate_ids": list(CANDIDATES),
    "control_ids": list(CONTROLS), "utility_risk_aversion": 1, "utility_cost_multiplier": 2,
    "mean_weight": .5, "fallback_target": 1., "selection_permitted": False,
    "performance_early_stopping": False, "inferential_mode": "descriptive_only",
    "independent_confirmation": False, "receipt_provenance_established": False,
    "support_contract": "inherited_retrospective_common_mask_including_all_derivatives_and_delays",
    "source_config": "configs/oracle_frontier_20260905.yaml",
    "data_cutoff": "2026-07-16T13:45:00Z", "output_dir": "codex_outputs/oracle_additional_window_replay_v1"}


def _bindings(records):
    resolved = set()
    for path, sha in records:
        if not isinstance(path, str) or not path or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("invalid path/SHA256 binding")
        try:
            int(sha, 16)
        except ValueError as exc:
            raise ValueError("invalid SHA256 binding") from exc
        canonical = Path(path).resolve()
        if canonical in resolved:
            raise ValueError("duplicate or aliased binding")
        resolved.add(canonical)


def validate_config(cfg):
    variable = {"source_config_sha256", "spot_path", "spot_sha256", "um_path", "um_sha256",
                "family_path", "family_sha256", "source_bindings", "data_manifest_path",
                "data_manifest_sha256", "preflight_sha256"}
    if (not isinstance(cfg, dict) or set(cfg) != set(FIXED) | variable
            or any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in FIXED.items())):
        raise ValueError("unregistered additional-window contract")
    if not isinstance(cfg["source_bindings"], dict) or set(cfg["source_bindings"]) != set(SOURCES):
        raise ValueError("unregistered source dependency set")
    _bindings([(cfg[k + "_path"], cfg[k + "_sha256"]) for k in ("spot", "um", "family", "data_manifest")]
              + [(cfg["source_config"], cfg["source_config_sha256"]), *cfg["source_bindings"].items()])
    _bindings([("preflight.json", cfg["preflight_sha256"])])


def _bound_json(path, expected):
    if file_digest(Path(path)) != expected:
        raise ValueError(f"bound artifact changed: {path}")
    return json.loads(Path(path).read_text())


def validate_inputs(cfg):
    """Verify all declared source/data/family bindings before loading features."""
    validate_config(cfg)
    for path, sha in [(cfg["source_config"], cfg["source_config_sha256"]),
                      *cfg["source_bindings"].items()]:
        if file_digest(Path(path)) != sha:
            raise ValueError(f"source/config changed: {path}")
    family = _bound_json(cfg["family_path"], cfg["family_sha256"])
    if (family["candidate_ids"] != list(CANDIDATES) or family["control_ids"] != list(CONTROLS)
            or family["single_model_selected"] is not False or family["existing_selection_locks_modified"] is not False
            or family["weight_family_closed"] != [0., .5, 1.] or family["half_weight"] != .5):
        raise ValueError("frozen candidate family changed")
    for name, sha in family["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name)) != sha:
            raise ValueError("frozen family source changed")
    manifest = _bound_json(cfg["data_manifest_path"], cfg["data_manifest_sha256"])
    if (set(manifest) not in ({"schema", "data_cutoff", "spot_path", "spot_sha256", "um_path", "um_sha256", "bindings"},
                             {"schema", "data_cutoff", "spot_path", "spot_sha256", "um_path", "um_sha256", "bindings", "quality"})
            or manifest["schema"] != "oracle-additional-window-data-manifest-v1"
            or any(manifest[k] != cfg[k] for k in ("data_cutoff", "spot_path", "spot_sha256", "um_path", "um_sha256"))
            or not isinstance(manifest["bindings"], dict)):
        raise ValueError("data manifest identity mismatch")
    _bindings(list(manifest["bindings"].items()))
    for path, sha in manifest["bindings"].items():
        if file_digest(Path(path)) != sha:
            raise ValueError(f"data manifest binding changed: {path}")
    bound = {Path(path).resolve(): sha for path, sha in manifest["bindings"].items()}
    required = []
    for prefix in ("spot", "um"):
        raw = Path(cfg[prefix + "_path"])
        sidecar = raw.with_suffix(".sha256.json")
        if file_digest(raw) != cfg[prefix + "_sha256"]:
            raise ValueError("raw data digest changed")
        info = json.loads(sidecar.read_text())
        required.extend([raw, sidecar, Path(info["availability_path"]), Path(info["source_ledger_path"])])
        if prefix == "um": required.append(Path(info["registration_path"]))
    if any(path.resolve() not in bound for path in required):
        raise ValueError("data manifest omits required raw/provenance artifact")
    fc = yaml.safe_load(Path(cfg["source_config"]).read_text())
    if (fc["symbol"] != "BTCUSDT" or fc["execution"] != EXECUTION
            or fc["regime"]["normalized_momentum_90_threshold"] != .5
            or fc["minimum_bar_coverage"] != .995):
        raise ValueError("frozen execution/regime contract changed")
    return fc, manifest


def prepare(config_path):
    """Data-only preflight: no forecast fit, scoring, or policy rollout."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    fc, manifest = validate_inputs(cfg)
    spot_proof = validate_data_artifact(Path(cfg["spot_path"]), expected_symbol=fc["symbol"])
    um, um_proof = validate_um(Path(cfg["um_path"]), cfg["data_cutoff"], fc["symbol"])
    bars = load_bars(Path(cfg["spot_path"]), cutoff=cfg["data_cutoff"])
    original, derivative = make_feature_groups(bars), make_derivative_groups(bars, um)
    groups = make_delayed_perp_groups(bars, um, delays=(0, 1, 4))
    tv = trailing_variances(bars, 24)
    common = np.isfinite(tv.to_numpy()).all(axis=1) & np.isfinite(original["flow"].to_numpy()).all(axis=1)
    for frame in [*derivative.values(), *groups.values()]:
        if not frame.index.equals(bars.index): raise ValueError("unaligned feature calendar")
        common &= np.isfinite(frame.to_numpy()).all(axis=1)
    y = outcome_frame(bars, 24).to_numpy()
    masks, support = {}, []
    for fold in FOLDS:
        dates = calendar(fold)
        if dates["evaluation_end"] > pd.Timestamp(cfg["data_cutoff"]):
            raise ValueError("evaluation exceeds frozen cutoff")
        m = segment_masks(bars.index, common, np.isfinite(y).all(axis=1), fold)
        masks[fold] = m
        ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        window = bars.loc[ix]
        if (not window.index.equals(pd.date_range(dates["evaluation_start"], dates["evaluation_end"], freq="15min", inclusive="left"))
                or window.bar_available.mean() < cfg["minimum_bar_coverage"]
                or not window.bar_available.iloc[0] or not window.bar_available.iloc[-1]):
            raise ValueError(f"fold{fold}: incomplete evaluation grid/bar coverage")
        for name in ("fit", "scale", "interval", "score"):
            if int(m[name].sum()) < cfg["minimum_" + name + "_rows"]:
                raise ValueError(f"fold{fold}: insufficient {name} rows; entire family stopped")
        actions = action_masks(window.index, window.open.to_numpy(), m["inference"][ix])
        support.append({"fold": fold, **{k: v.isoformat() for k, v in dates.items() if isinstance(v, pd.Timestamp)},
            "regime": quarter_regime(original["base16"], ix, .5),
            "counts": {name: int(mask.sum()) for name, mask in m.items()},
            "action_counts": {name: int(mask.sum()) for name, mask in actions.items()},
            "mask_sha256": {name: mask_digest(bars.index, mask) for name, mask in m.items()},
            "bar_coverage": float(window.bar_available.mean()),
            "last_label_end": {name: (bars.index[m[name]][-1] + pd.Timedelta(minutes=375)).isoformat()
                               for name in ("fit", "scale", "interval", "score")}})
    preflight = {"schema": "oracle-additional-window-preflight-v1",
        "config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
        "source_bindings": cfg["source_bindings"], "source_config_sha256": cfg["source_config_sha256"],
        "family_sha256": cfg["family_sha256"], "data_manifest_sha256": cfg["data_manifest_sha256"],
        "spot_data_proof": spot_proof, "um_data_proof": um_proof, "support": support,
        "feature_columns": {g: list(groups[g].columns) for g in ("technical", "perp_delay0")},
        "full_common_mask_sha256": mask_digest(bars.index, common),
        "data_cutoff": cfg["data_cutoff"], "data_quality": manifest.get("quality"),
        "scope": "data-only on reused original test15-24; retrospective common availability",
        "new_forecast_or_policy_computed": False, "selection_performed": False,
        "independent_confirmation": False, "receipt_provenance_established": False}
    return {"config": cfg, "source_config": fc, "bars": bars, "groups": groups,
            "original": original, "outcomes": y, "masks": masks, "preflight": preflight}


def _finite(value, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("invalid finite summary endpoint")
    return float(value)


def _inventory(rows, scores, folds):
    def collect(values, key, ids):
        out = {}
        for row in values:
            if type(row.get("fold")) is not int: raise ValueError("integer original test fold required")
            pair = row["fold"], row[key]
            if pair in out or pair[0] not in folds or pair[1] not in ids:
                raise ValueError("duplicate or unregistered family row")
            out[pair] = row
        if set(out) != {(f, k) for f in folds for k in ids}:
            raise ValueError("incomplete family; missing quarter cannot be dropped")
        return out
    economic, predictive = collect(rows, "candidate_id", POLICIES), collect(scores, "mean_id", MEANS)
    regimes = {}
    for fold in folds:
        paired = [economic[fold, cid] for cid in POLICIES] + [predictive[fold, mid] for mid in MEANS]
        regime = paired[0].get("regime")
        if any(row.get("regime") != regime for row in paired): raise ValueError("unpaired regime metadata")
        trend = regime.get("trend") if isinstance(regime, dict) else regime
        if trend not in (*STRATA[1:], "unavailable"): raise ValueError("unknown regime")
        regimes[fold] = trend
        counts = [predictive[fold, mid]["rows"] for mid in MEANS]
        if any(type(n) is not int or n <= 0 for n in counts) or len(set(counts)) != 1:
            raise ValueError("unpaired scoring denominators")
    for row in economic.values():
        for cost in COSTS:
            for key in ECON_KEYS:
                _finite(row[cost][key])
    for row in predictive.values():
        for key in ("return_mse", "return_mae", "zero_return_mse", "fit_mean_return_mse"):
            _finite(row[key], True)
        if not 0 <= _finite(row["return_sign_accuracy"]) <= 1: raise ValueError("invalid sign accuracy")
        ic = row["return_rank_ic"]
        if ic is not None and not -1 <= _finite(ic) <= 1: raise ValueError("invalid rank IC")
        if row["mean_id"] == "scale_mean" and ic is not None: raise ValueError("constant mean rank IC must be undefined")
    return economic, predictive, regimes


def describe_additional_family(rows, scores):
    """All literal original test15-24 rows; equal-quarter and pooled losses."""
    economic, predictive, regimes = _inventory(rows, scores, FOLDS)
    counts = {s: sum(value == s for value in regimes.values()) for s in (*STRATA[1:], "unavailable")}
    coverage = all(counts[s] >= 3 for s in STRATA[1:]) and counts["unavailable"] == 0
    policy_summary, mean_summary, pairs, policy_pairs, rule_pairs = {}, {}, {}, {}, {}
    def average(values):
        result = math.fsum(float(v) / len(values) for v in values) if values else None
        if result is not None: _finite(result)
        return result
    for stratum in STRATA:
        folds = [f for f in FOLDS if stratum == "all" or regimes[f] == stratum]
        for cid in POLICIES:
            policy_summary.setdefault(cid, {})[stratum] = {"quarters": len(folds), **{
                cost: {"alpha_ex_mean": average([economic[f, cid][cost]["alpha_ex"] for f in folds]),
                       "maxdd_delta_mean": average([economic[f, cid][cost]["maxdd_delta"] for f in folds]),
                       **{key+"_mean": average([economic[f,cid][cost][key] for f in folds]) for key in ECON_KEYS[2:]},
                       "joint_positive_quarters": sum(economic[f, cid][cost]["alpha_ex"] > 0 and economic[f, cid][cost]["maxdd_delta"] < 0 for f in folds)}
                for cost in COSTS}}
        for mid in MEANS:
            selected = [predictive[f, mid] for f in folds]
            total = sum(row["rows"] for row in selected)
            losses = ("return_mse", "return_mae", "zero_return_mse", "fit_mean_return_mse", "return_sign_accuracy")
            rank = [row["return_rank_ic"] for row in selected if row["return_rank_ic"] is not None]
            mean_summary.setdefault(mid, {})[stratum] = {"quarters": len(folds), "scored_rows": total,
                "equal_quarter": {key: average([row[key] for row in selected]) for key in losses},
                "row_pooled": {key: math.fsum(row[key] * (row["rows"] / total) for row in selected) if total else None for key in losses},
                "rank_ic_mean_defined_quarters": average(rank), "rank_ic_defined_quarters": len(rank)}
        for candidate, reference in PAIRS:
            delta = [predictive[f, candidate]["return_mse"] - predictive[f, reference]["return_mse"] for f in folds]
            total = sum(predictive[f, candidate]["rows"] for f in folds)
            pairs.setdefault(candidate + "_vs_" + reference, {})[stratum] = {
                "quarters": len(folds), "scored_rows": total, "candidate_minus_reference_mse": average(delta),
                "row_pooled_candidate_minus_reference_mse": math.fsum(d * (predictive[f, candidate]["rows"] / total) for d, f in zip(delta, folds)) if total else None,
                "improved_quarters": sum(d < 0 for d in delta)}
            entry=pairs[candidate + "_vs_" + reference][stratum]
            for loss in ("return_mse", "return_mae"):
                cmean=average([predictive[f,candidate][loss] for f in folds])
                rmean=average([predictive[f,reference][loss] for f in folds])
                entry[loss+"_relative_loss_reduction"] = 1-cmean/rmean if rmean is not None and rmean>0 else None
            for rule in RULES:
                cid,rid=candidate+"_"+rule,reference+"_"+rule
                policy_pairs.setdefault(cid+"_vs_"+rid,{})[stratum]={"quarters":len(folds), **{
                    cost:{key:average([economic[f,cid][cost][key]-economic[f,rid][cost][key] for f in folds]) for key in ECON_KEYS}
                    for cost in COSTS}}
        for mid in MEANS[3:]:
            cid,rid=mid+"_"+RULES[1],mid+"_"+RULES[0]
            rule_pairs.setdefault(mid,{})[stratum]={"quarters":len(folds), **{
                cost:{key:average([economic[f,cid][cost][key]-economic[f,rid][cost][key] for f in folds]) for key in ECON_KEYS}
                for cost in COSTS}}
    candidate_summary,components = {},[]
    for cid in CANDIDATES:
        for s in STRATA:
            for cost in COSTS:
                for metric,key,sign in (("alpha_ex","alpha_ex_mean",1),("negative_maxdd_delta","maxdd_delta_mean",-1)):
                    value=policy_summary[cid][s][cost][key]
                    components.append({"id":"/".join(("economic",cid,s,cost,metric)),"value":sign*value if value is not None else None,"p_value":None})
    for c,r in PAIRS[:4]:
        for s in STRATA:
            value=pairs[c+"_vs_"+r][s]["candidate_minus_reference_mse"]
            components.append({"id":"/".join(("predictive",c,r,s,"negative_mse_difference")),"value":-value if value is not None else None,"p_value":None})
    for cid in CANDIDATES:
        mean = next(m for m in MEANS[3:] if cid.startswith(m + "_"))
        economic_pass = all(policy_summary[cid][s][cost]["alpha_ex_mean"] is not None
            and policy_summary[cid][s][cost]["alpha_ex_mean"] > 0 and policy_summary[cid][s][cost]["maxdd_delta_mean"] < 0
            for s in STRATA for cost in COSTS)
        predictive_pass = all(pairs[c + "_vs_" + r][s]["candidate_minus_reference_mse"] is not None
            and pairs[c + "_vs_" + r][s]["candidate_minus_reference_mse"] < 0
            for c, r in PAIRS if c == mean and r != "technical_half" for s in STRATA)
        candidate_summary[cid] = {"observed_economic_signs": economic_pass,
            "observed_predictive_signs": predictive_pass,
            "observed_metric_and_coverage_conditions_met": coverage and economic_pass and predictive_pass,
            "high_probability_generalization_established": False}
    return {"scope": "descriptive reused original test15-24; no independent confirmation",
        "original_test_folds": list(FOLDS), "complete_family": True, "policy_rows": len(rows), "forecast_rows": len(scores),
        "regime_counts": counts, "regime_coverage": coverage, "policies": policy_summary,
        "means": mean_summary, "paired_mse": pairs, "paired_policies":policy_pairs,
        "fallback_minus_hold":rule_pairs, "candidates": candidate_summary,
        "descriptive_components":components, "component_count":len(components),
        "p_values":None, "confidence_intervals":None, "adjusted_p_values":None,
        "selection_performed": False, "high_probability_generalization_established": False,
        "receipt_provenance_established": False, "independent_confirmation": False}


def artifact_inventory(output, fold):
    groups={"models":[(n,".joblib") for n in ("technical_mean","perp_delay0_mean","technical_variance")],
        "forecasts":[(n,".npz") for n in MEANS], "targets":[(n,".npz") for n in POLICIES],
        "traces":[(n,".json") for n in POLICIES[2:]],
        "calibration":[("technical",".npz"),("perp_delay0",".npz"),("provenance",".json")]}
    return {str(output/kind/f"fold{fold}_{name}{ext}") for kind,entries in groups.items() for name,ext in entries}


def plan(window, cid, predictions, execution, robust):
    if cid == "bh": return np.full(len(window),np.nan),None
    inference=predictions["scale_mean"]["inference_mask"]
    if cid == "common_robust":
        target=robust.copy();target[~inference]=np.nan
        return target,None
    rule=RULES[1] if cid.endswith(RULES[1]) else RULES[0]
    mean=cid[:-(len(rule)+1)];pred=predictions[mean]
    if rule == RULES[1]:
        targets,trace=fallback_targets(window,pred["mu"],pred["variance"],execution,
            inference_mask=inference,risk_aversion=1,cost_multiplier=2)
        actions=action_masks(window.index,window.open.to_numpy(),inference)
        check_action_support(targets,actions);check_trace_support(targets,actions,trace)
        return targets,trace
    return conditional_targets(window,pred["mu"],pred["variance"],execution,risk_aversion=1,cost_multiplier=2)


def validate_completed_fold(saved, fold, output, registration_sha, window, masks, support, execution, robust,
                            expected_actual, calibration_expected):
    """Resume only a complete bound family, recomputing saved score/decision paths."""
    if (set(saved)!={"registration_sha256","rows","scores","artifact_sha256"}
            or saved["registration_sha256"]!=registration_sha):
        raise ValueError("incomplete saved fold schema or changed registration")
    rows,scores,_=_inventory(saved["rows"],saved["scores"],(fold,))
    if any(r["regime"]!=support["regime"] for r in saved["rows"]+saved["scores"]):
        raise ValueError("saved regime differs from data-only preflight")
    if set(saved["artifact_sha256"])!=artifact_inventory(output,fold):
        raise ValueError("incomplete artifact inventory")
    for path,sha in saved["artifact_sha256"].items():
        if file_digest(Path(path))!=sha:raise ValueError("saved artifact changed")
    predictions={}
    for mean in MEANS:
        path=output/"forecasts"/f"fold{fold}_{mean}.npz"
        if file_digest(path)!=scores[fold,mean]["forecast_sha256"]:raise ValueError("forecast score binding changed")
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!={"timestamps","mu","variance","inference_mask","score_support","actual","fit_return_mean"}:
                raise ValueError("forecast schema changed")
            pred={k:arrays[k] for k in arrays.files}
        for key,value in {"timestamps":window.index.asi8,"inference_mask":masks["inference"],"score_support":masks["score"]}.items():
            compare_array(pred[key],value,name=key,exact=True)
        compare_array(pred["actual"],expected_actual,name="source evaluation labels",exact=True)
        if (pred["actual"].shape!=(len(window),3) or pred["mu"].shape!=(len(window),)
                or pred["variance"].shape!=(len(window),) or pred["fit_return_mean"].shape!=()
                or not np.isnan(pred["mu"][~pred["inference_mask"]]).all()
                or not np.isnan(pred["variance"][~pred["inference_mask"]]).all()
                or not np.isnan(pred["actual"][~pred["score_support"]]).all()
                or not np.isfinite(pred["mu"][pred["inference_mask"]]).all()
                or not np.isfinite(pred["variance"][pred["inference_mask"]]).all()
                or (pred["variance"][pred["inference_mask"]]<=0).any()):
            raise ValueError("forecast shape or finite support changed")
        predictions[mean]=pred
        for key in ("timestamps","inference_mask","score_support","variance","actual","fit_return_mean"):
            compare_array(pred[key],predictions["scale_mean"][key],name="common "+key,exact=True)
        score=return_scores(pred["actual"],pred["mu"],pred["score_support"],pred["fit_return_mean"])
        compare_tree(score,{k:scores[fold,mean][k] for k in score},name="saved score")
    for half,full in (("technical_half","technical_scaled"),("perp_delay0_half","perp_delay0_scaled")):
        compare_array(predictions[half]["mu"],.5*predictions["scale_mean"]["mu"]+.5*predictions[full]["mu"],name="half formula",exact=True)
    stress={**execution,"one_way_cost":2*execution["one_way_cost"],"borrow_annual":2*execution["borrow_annual"]}
    for cid in POLICIES:
        path=output/"targets"/f"fold{fold}_{cid}.npz"
        if file_digest(path)!=rows[fold,cid]["targets_sha256"]:raise ValueError("target row binding changed")
        expected,trace=plan(window,cid,predictions,execution,robust)
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!={"timestamps","targets"}:raise ValueError("target schema changed")
            compare_array(arrays["timestamps"],window.index.asi8,name="target calendar",exact=True)
            compare_array(arrays["targets"],expected,name="saved targets",exact=True)
        if trace is not None:
            compare_tree(trace,json.loads((output/"traces"/f"fold{fold}_{cid}.json").read_text()),name="saved trace")
        for cost,contract in (("base",execution),("stress_2x",stress)):
            compare_tree(metrics(window,expected,contract),rows[fold,cid][cost],name="saved accounting")
    provenance=json.loads((output/"calibration"/f"fold{fold}_provenance.json").read_text())
    expected_counts={k:support["counts"][k] for k in ("fit","scale","interval")}
    compare_tree(provenance["calibration"]["counts"],expected_counts,name="calibration counts")
    compare_tree(provenance["provenance"]["mask_counts"],{k:support["counts"][k] for k in ("fit","scale","interval","predict","inference")},name="fit provenance")
    for group in ("technical","perp_delay0"):
        with np.load(output/"calibration"/f"fold{fold}_{group}.npz",allow_pickle=False) as arrays:
            raw_keys={"mu","log_variance","variance"} if group=="technical" else {"mu"}
            if set(arrays.files)!={"timestamps","actual","scale_mask","interval_mask"}|raw_keys:
                raise ValueError("calibration schema changed")
            for key in ("timestamps","actual","scale_mask","interval_mask"):
                compare_array(arrays[key],calibration_expected[key],name="source calibration "+key,exact=True)
            for key in raw_keys:
                value=arrays[key];predict=calibration_expected["predict"]
                if (value.shape!=predict.shape or not np.isfinite(value[predict]).all()
                        or not np.isnan(value[~predict]).all()):
                    raise ValueError("calibration raw prediction support changed")
            scale=arrays["scale_mask"]
            compare_array(float(np.mean(arrays["actual"][scale,0]-arrays["mu"][scale])),
                provenance["calibration"]["return_bias"][group],name="scale return bias")
            compare_array(math.fsum(float(v)/int(scale.sum()) for v in arrays["actual"][scale,0]),
                provenance["calibration"]["scale_mean"],name="source scale anchor")
    anchor=predictions["scale_mean"]
    compare_array(anchor["mu"][anchor["inference_mask"]],np.full(int(anchor["inference_mask"].sum()),provenance["calibration"]["scale_mean"]),name="scale mean",exact=True)


def run(config_path):
    data=prepare(config_path)
    cfg,fc,bars,groups,original,y,masks,pre=(data[k] for k in ("config","source_config","bars","groups","original","outcomes","masks","preflight"))
    output=Path(cfg["output_dir"])
    if (output/"results.json").exists():raise ValueError("immutable additional-window replay already completed")
    if (file_digest(output/"preflight.json")!=cfg["preflight_sha256"]
            or json.loads((output/"preflight.json").read_text())!=pre):
        raise ValueError("registered preflight changed")
    registration={"config":cfg,"config_sha256":file_digest(config_path),"preflight_sha256":cfg["preflight_sha256"],
        "source_bindings":cfg["source_bindings"],"data_manifest_sha256":cfg["data_manifest_sha256"],
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__},
        "scope":"Frozen family report-only on reused original test15-24; no selection or independent confirmation"}
    _immutable_json(output/"registration.json",registration)
    execution=fc["execution"]
    stress={**execution,"one_way_cost":2*execution["one_way_cost"],"borrow_annual":2*execution["borrow_annual"]}
    robust_all=build_targets(original["base16"])
    all_rows,all_scores=[],[]
    for fold in FOLDS:
        dates=calendar(fold);m=masks[fold]
        ix=np.asarray((bars.index>=dates["evaluation_start"])&(bars.index<dates["evaluation_end"]))
        window=bars.loc[ix];support=next(s for s in pre["support"] if s["fold"]==fold)
        actual=y[ix].copy();actual[~m["score"][ix]]=np.nan
        cal_ix=np.asarray((bars.index>=dates["scale_start"])&(bars.index<dates["evaluation_start"]))
        cal_actual=y[cal_ix].copy();cal_actual[~(m["scale"]|m["interval"])[cal_ix]]=np.nan
        kwargs=dict(fold=fold,output=output,registration_sha=digest(registration),window=window,
            masks={k:v[ix] for k,v in m.items()},support=support,execution=execution,robust=robust_all[ix],
            expected_actual=actual,calibration_expected={"timestamps":bars.index[cal_ix].asi8,"actual":cal_actual,
                "scale_mask":m["scale"][cal_ix],"interval_mask":m["interval"][cal_ix],"predict":m["predict"][cal_ix]})
        fold_path=output/f"fold_{fold}.json"
        if fold_path.exists():
            saved=json.loads(fold_path.read_text());validate_completed_fold(saved,**kwargs)
            all_rows.extend(saved["rows"]);all_scores.extend(saved["scores"]);continue
        fresh=fit_frozen_forecasts({k:groups[k] for k in ("technical","perp_delay0")},y,
            **{k+"_mask":m[k] for k in ("fit","scale","interval","predict","inference")})
        bindings,rows,scores,predictions={},[],[],{}
        def save_arrays(kind,name,arrays):
            path=output/kind/f"fold{fold}_{name}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    if set(saved.files)!=set(arrays):raise ValueError("partial array schema changed")
                    for key,value in arrays.items():compare_array(value,saved[key],name=str(path)+key,exact=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
            bindings[str(path)]=file_digest(path)
            return path
        for name,model in fresh["models"].items():
            path=output/"models"/f"fold{fold}_{name}.joblib";path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                temp=path.with_suffix(".verify.joblib");joblib.dump(model,temp)
                if file_digest(temp)!=file_digest(path):raise ValueError("partial model differs")
                temp.unlink()
            else:joblib.dump(model,path)
            bindings[str(path)]=file_digest(path)
        actual=y[ix].copy();actual[~m["score"][ix]]=np.nan
        for mean in MEANS:
            pred={"timestamps":window.index.asi8,"mu":fresh["means"][mean][ix],"variance":fresh["variance"][ix],
                "inference_mask":m["inference"][ix],"score_support":m["score"][ix],"actual":actual,
                "fit_return_mean":np.asarray(fresh["calibration"]["fit_mean"])}
            predictions[mean]=pred;path=save_arrays("forecasts",mean,pred)
            score=return_scores(actual,pred["mu"],pred["score_support"],pred["fit_return_mean"])
            scores.append({"fold":fold,"mean_id":mean,"regime":support["regime"],**score,"forecast_sha256":bindings[str(path)]})
        cal_ix=np.asarray((bars.index>=dates["scale_start"])&(bars.index<dates["evaluation_start"]))
        for group in ("technical","perp_delay0"):
            arrays={"timestamps":bars.index[cal_ix].asi8,"actual":fresh["calibration_arrays"]["actual"][cal_ix],
                "scale_mask":m["scale"][cal_ix],"interval_mask":m["interval"][cal_ix],
                **{k:v[cal_ix] for k,v in fresh["raw_predictions"][group].items()}}
            save_arrays("calibration",group,arrays)
        provenance=output/"calibration"/f"fold{fold}_provenance.json"
        _immutable_json(provenance,{"calibration":fresh["calibration"],"provenance":fresh["provenance"]})
        bindings[str(provenance)]=file_digest(provenance)
        for cid in POLICIES:
            targets,trace=plan(window,cid,predictions,execution,robust_all[ix])
            path=save_arrays("targets",cid,{"timestamps":window.index.asi8,"targets":targets})
            if trace is not None:
                tracepath=output/"traces"/f"fold{fold}_{cid}.json";_immutable_json(tracepath,trace)
                bindings[str(tracepath)]=file_digest(tracepath)
            rows.append({"fold":fold,"candidate_id":cid,"regime":support["regime"],
                **{cost:metrics(window,targets,contract) for cost,contract in (("base",execution),("stress_2x",stress))},
                "targets_sha256":bindings[str(path)]})
        saved={"registration_sha256":digest(registration),"rows":rows,"scores":scores,"artifact_sha256":bindings}
        validate_completed_fold(saved,**kwargs);_immutable_json(fold_path,saved)
        all_rows.extend(rows);all_scores.extend(scores)
        print(json.dumps({"event":"fold_complete","fold":fold,"policy_rows":len(rows),"forecast_rows":len(scores)}),flush=True)
    result={"registration_sha256":digest(registration),"rows":all_rows,"scores":all_scores,
        "summary":describe_additional_family(all_rows,all_scores),"new_candidate_count":0,
        "additional_reused_evaluation_periods":10,"selection_performed":False,
        "high_probability_generalization_established":False,"independent_confirmation":False,
        "receipt_provenance_established":False,"scope":registration["scope"]}
    _immutable_json(output/"results.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        data=prepare(args.config);path=Path(data["config"]["output_dir"])/"preflight.json"
        _immutable_json(path,data["preflight"])
        print(json.dumps({"path":str(path),"sha256":file_digest(path),"new_forecast_or_policy_computed":False}))
    else:run(args.config)
