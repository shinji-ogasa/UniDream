"""Fixed 2x2 short-price/flow representation ablation, original development only."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import (
    FOLDS, RULES, compare_array, compare_tree, prepare as prepare_parity, delay_um,
)
from .oracle_mean_controls import return_scores
from .oracle_mean_reliability_decisions import arrays
from .oracle_rolling_centering_decisions import (
    prepare as prepare_parent, MEANS as OLD_MEANS, POLICIES as OLD_CONTROLS, SOURCES as PARENT_SOURCES,
)
from .oracle_short_features import make_short_feature_groups, PRICE_FEATURE_NAMES, FLOW_FEATURE_NAMES
from .oracle_short_mean_fit import fit_raw_mean_family


PARENT_ROOT = Path("codex_outputs/oracle_rolling_centering_decisions_v1")
FALLBACK_ROOT = Path("codex_outputs/oracle_fallback_decisions_v1")
GROUPS = ("technical", "technical_short_price", "technical_short_flow", "technical_short_both")
NEW_GROUPS = GROUPS[1:]
NEW_MEANS = tuple(g + "_raw" for g in NEW_GROUPS)
SIMPLE = ("zero", "fit_mean", "technical_raw")
EXTRA_CONTROLS = tuple(m + "_" + rule for m in SIMPLE for rule in RULES)
CONTROLS = OLD_CONTROLS + EXTRA_CONTROLS
NEW_IDS = tuple(m + "_" + rule for m in NEW_MEANS for rule in RULES)
POLICIES = CONTROLS + NEW_IDS
SCORE_MEANS = {"interval": SIMPLE + NEW_MEANS, "evaluation": OLD_MEANS + ("zero", "fit_mean") + NEW_MEANS}
REFERENCES = {NEW_MEANS[0]: ("technical_raw",), NEW_MEANS[1]: ("technical_raw",),
              NEW_MEANS[2]: ("technical_raw", NEW_MEANS[0], NEW_MEANS[1])}
SOURCES = PARENT_SOURCES + tuple("unidream/experiments/" + n for n in
    ("oracle_short_features.py", "oracle_short_mean_fit.py", "oracle_short_feature_decisions.py"))
FIXED = {"schema": "oracle-short-feature-decisions-v1", "development_folds": list(FOLDS),
    "source_prepare_config": "configs/oracle_rolling_centering_decisions_20260906.yaml",
    "parity_prepare_config": "configs/oracle_frozen_procedure_parity_20260906.yaml",
    "parent_root": str(PARENT_ROOT), "fallback_root": str(FALLBACK_ROOT),
    "parent_source_revision": "845a9fd167533599c7189019b38dc9ca2edf0f41",
    "output_dir": "codex_outputs/oracle_short_feature_decisions_v1",
    "data_cutoff": "2023-04-16T13:45:00Z", "groups": list(GROUPS),
    "group_dimensions": [29, 34, 32, 37], "price_columns": list(PRICE_FEATURE_NAMES),
    "flow_columns": list(FLOW_FEATURE_NAMES), "feature_shift_bars": 1,
    "new_mean_ids": list(NEW_MEANS), "control_ids": list(CONTROLS), "new_policy_ids": list(NEW_IDS),
    "score_means": {k: list(v) for k, v in SCORE_MEANS.items()}, "rules": list(RULES),
    "return_model": "fit_only_StandardScaler_Ridge100", "return_calibration": "none",
    "risk_source": "unchanged_technical_scaled", "ridge_alpha": 100.0, "threadpool_limit": 2,
    "sklearn_version": "1.8.0", "numpy_version": "2.2.6", "pandas_version": "2.3.3",
    "minimum_fit_rows": 512, "minimum_score_rows": 16,
    "utility_risk_aversion": 1, "utility_cost_multiplier": 2,
    "inference_rows": 2586, "evaluation_score_rows": 2574, "fallback_rows": 332,
    "missing_current_open_rows": 2, "adaptive_prior_causal_names": 168,
    "new_causal_policy_names": 6, "new_return_model_fits": 24, "baseline_parity_fits": 8,
    "partial_retry_policy": "replay_all_folds_and_verify_existing_artifacts_no_live_restart",
    "risk_fitting_permitted": False, "weight_fitting_permitted": False,
    "selection_permitted": False, "additional_test_access_permitted": False,
    "interval_width_claims_permitted": False}
EXTRA = {"source_bindings", "source_prepare_config_sha256", "parity_prepare_config_sha256",
         "parent_results_sha256", "parent_registration_sha256", "parent_preflight_sha256",
         "fallback_results_sha256", "fallback_registration_sha256", "preflight_sha256"}


def validate_config(cfg):
    if (set(cfg) != set(FIXED) | EXTRA
            or any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in FIXED.items())
            or set(cfg["source_bindings"]) != set(SOURCES)):
        raise ValueError("unregistered short feature family")


def prepare(config_path):
    """Data-only identity, feature support and source checks; no new fits."""
    cfg = yaml.safe_load(Path(config_path).read_text()); validate_config(cfg)
    if (sklearn.__version__ != cfg["sklearn_version"] or np.__version__ != cfg["numpy_version"]
            or pd.__version__ != cfg["pandas_version"]): raise ValueError("registered runtime version changed")
    direct = {**cfg["source_bindings"], cfg["source_prepare_config"]: cfg["source_prepare_config_sha256"],
        cfg["parity_prepare_config"]: cfg["parity_prepare_config_sha256"],
        **{str(PARENT_ROOT / (n + ".json")): cfg["parent_" + n + "_sha256"] for n in ("results", "registration", "preflight")},
        **{str(FALLBACK_ROOT / (n + ".json")): cfg["fallback_" + n + "_sha256"] for n in ("results", "registration")}}
    for p, h in direct.items():
        if file_digest(Path(p)) != h: raise ValueError("registered source changed: " + p)
    pc, fc, bars, forecasts, _, _, _, _, parent_pre = prepare_parent(Path(cfg["source_prepare_config"]))
    parent = json.loads((PARENT_ROOT / "results.json").read_text()); reg = json.loads((PARENT_ROOT / "registration.json").read_text())
    if (reg["config"] != pc or reg["config_sha256"] != cfg["source_prepare_config_sha256"]
            or reg["preflight_sha256"] != cfg["parent_preflight_sha256"]
            or reg["source_revision"] != cfg["parent_source_revision"]
            or parent_pre != json.loads((PARENT_ROOT / "preflight.json").read_text())
            or parent["registration_sha256"] != digest(reg)):
        raise ValueError("Stage14 parent chain changed")
    bindings = dict(parent_pre["source_artifact_bindings"])
    controls = {(r["fold"], r["candidate_id"]): r for r in parent["rows"]}
    if len(controls) != 176 or set(controls) != {(f, p) for f in FOLDS for p in OLD_CONTROLS}:
        raise ValueError("incomplete Stage14 controls")
    fallback = json.loads((FALLBACK_ROOT / "results.json").read_text())
    freg = json.loads((FALLBACK_ROOT / "registration.json").read_text())
    if fallback["registration_sha256"] != digest(freg): raise ValueError("fallback source chain changed")
    fallback_rows = {(r["fold"], r["candidate_id"]): r for r in fallback["rows"]}
    for f in FOLDS:
        fp = PARENT_ROOT / f"fold_{f}.json"; fold = json.loads(fp.read_text()); direct[str(fp)] = file_digest(fp)
        if (fold["registration_sha256"] != digest(reg) or len(fold["artifact_sha256"]) != 32
                or any(fold[k] != [r for r in parent[k] if r["fold"] == f] for k in ("rows", "scores", "fixed_weights"))):
            raise ValueError("Stage14 fold changed")
        for p, h in fold["artifact_sha256"].items():
            if file_digest(Path(p)) != h or (p in bindings and bindings[p] != h): raise ValueError("parent artifact changed")
            bindings[p] = h
        for m in ("rolling_anchor", "technical_rolling", "perp_delay0_rolling"):
            a = arrays(PARENT_ROOT / "forecasts" / f"fold{f}_{m}.npz")
            for k in a:
                if k != "mu": compare_array(a[k], forecasts[f, "scale_mean"][k], name="Stage14 forecast support", exact=True)
            forecasts[f, m] = a
        for cid in EXTRA_CONTROLS:
            row = fallback_rows[f, cid]; p = FALLBACK_ROOT / "targets" / f"fold{f}_{cid}.npz"
            if str(p) not in bindings or row["targets_sha256"] != bindings[str(p)]: raise ValueError("unbound extra control")
            if row["regime"] != controls[f, "bh"]["regime"]: raise ValueError("extra control regime differs")
            controls[f, cid] = row
    if len(bindings) != 1792 or len({str(Path(p).resolve()) for p in bindings}) != len(bindings):
        raise ValueError("ancestor artifact inventory or alias mismatch")
    par = prepare_parity(Path(cfg["parity_prepare_config"]))
    ppc, _, pfc, pbars, old_groups, _, y, masks, pp, _, _, _ = par
    if pfc != fc or not pbars.equals(bars) or fc["data_cutoff"] != cfg["data_cutoff"]:
        raise ValueError("original fit/evaluation data changed")
    groups = make_short_feature_groups(bars, delay_um(ppc, fc))
    if tuple(groups) != GROUPS or [len(groups[g].columns) for g in GROUPS] != cfg["group_dimensions"]:
        raise ValueError("feature group schema changed")
    if not groups["technical"].equals(old_groups["technical"]): raise ValueError("technical29 baseline features changed")
    support = []
    for f in FOLDS:
        dates = calendar(f - 1); m = masks[f]
        for name in ("fit", "scale", "interval", "inference", "score", "predict"):
            if any(not np.isfinite(groups[g].to_numpy()[m[name]]).all() for g in GROUPS):
                raise ValueError("new features remove original " + name + " support")
        if m["fit"].sum() < 512: raise ValueError("insufficient inherited fit rows")
        ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        ref = forecasts[f, "technical_raw"]
        for k, value in (("inference_mask", m["inference"][ix]), ("score_support", m["score"][ix])):
            compare_array(value, ref[k], name="original evaluation " + k, exact=True)
        support.append({"fold": f, "counts": {k: int(v.sum()) for k, v in m.items()},
            "mask_sha256": {k: mask_digest(bars.index, v) for k, v in m.items()},
            "feature_columns": {g: list(groups[g].columns) for g in GROUPS},
            "fit_feature_sha256": {g: digest(groups[g].to_numpy()[m["fit"]].tolist()) for g in GROUPS},
            "fit_return_sha256": digest(y[m["fit"], 0].tolist()),
            "regime": controls[f, "bh"]["regime"], "all_new_features_finite_on_all_original_masks": True})
    pre = {"schema": "oracle-short-feature-preflight-v1",
        "config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
        "source_bindings": cfg["source_bindings"], "direct_source_bindings": direct,
        "source_artifact_bindings": bindings, "parent_preflight_sha256": digest(parent_pre),
        "parity_preflight_sha256": digest(pp), "support": support,
        "fallback_rows": 332, "missing_current_open_rows": 2,
        "new_model_fitted": False, "new_forecast_or_policy_computed": False,
        "additional_test_used_for_modeling_or_scoring": False,
        "loader_scope": "Inherited Spot parquet decoding precedes cutoff; semantic data strictly below cutoff"}
    return cfg, fc, bars, groups, y, masks, forecasts, controls, parent, pre


def compare_models(got, expected):
    """Compare fitted numerical state, not historical pickle serialization."""
    if list(got.named_steps) != ["standardscaler", "ridge"] or list(expected.named_steps) != ["standardscaler", "ridge"]:
        raise ValueError("baseline model structure changed")
    errors = {}
    for step, attrs in (("standardscaler", ("mean_", "var_", "scale_", "n_features_in_", "n_samples_seen_")),
                        ("ridge", ("coef_", "intercept_", "n_features_in_"))):
        a, b = got.named_steps[step], expected.named_steps[step]
        if a.get_params() != b.get_params(): raise ValueError("baseline model parameters changed")
        for k in attrs: errors[step + "." + k] = compare_array(getattr(a, k), getattr(b, k), name="baseline " + k)
    return errors


def summarize(rows, scores):
    ri = {(r["fold"], r["candidate_id"]): r for r in rows}
    si = {(r["fold"], r["segment"], r["mean_id"]): r for r in scores}
    if (len(ri) != len(rows) or set(ri) != {(f, p) for f in FOLDS for p in POLICIES}
            or len(si) != len(scores) or set(si) != {(f, seg, m) for f in FOLDS for seg in SCORE_MEANS for m in SCORE_MEANS[seg]}):
        raise ValueError("incomplete short feature family")
    regime_counts = {g: sum(ri[f, "bh"]["regime"]["trend"] == g for f in FOLDS) for g in ("bull", "bear", "sideways")}
    if regime_counts != {"bull": 2, "bear": 4, "sideways": 2}: raise ValueError("regime coverage changed")
    for f in FOLDS:
        regime = ri[f, "bh"]["regime"]
        for p in POLICIES:
            if ri[f, p]["regime"] != regime or any(not math.isfinite(ri[f, p][c][k]) for c in ("base", "stress_2x")
                for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")): raise ValueError("invalid economic inputs")
        for seg, means in SCORE_MEANS.items():
            n = si[f, seg, "technical_raw"]["rows"]
            if type(n) is not int or n < 16: raise ValueError("invalid scored count")
            for m in means:
                s = si[f, seg, m]
                if s["rows"] != n or s["regime"] != regime or any(not math.isfinite(s[k]) or s[k] < 0 for k in
                    ("return_mse", "return_mae", "zero_return_mse", "fit_mean_return_mse")): raise ValueError("invalid paired scores")
    def mean(values): return math.fsum(v / len(values) for v in values)
    econ, pred, paired = {}, {}, {}
    for regime in ("all", "bull", "bear", "sideways"):
        fs = [f for f in FOLDS if regime == "all" or ri[f, "bh"]["regime"]["trend"] == regime]
        econ[regime] = {p: {"quarters": len(fs), "joint_positive_quarters_both_costs": sum(all(
            ri[f, p][c]["alpha_ex"] > 0 and ri[f, p][c]["maxdd_delta"] < 0 for c in ("base", "stress_2x")) for f in fs),
            **{c: {k: mean([ri[f, p][c][k] for f in fs]) for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")}
               for c in ("base", "stress_2x")}} for p in POLICIES}
        pred[regime] = {}
        for seg, means in SCORE_MEANS.items():
            pred[regime][seg] = {}
            for m in means:
                ss = [si[f, seg, m] for f in fs]; n = sum(a["rows"] for a in ss)
                pred[regime][seg][m] = {"quarters": len(fs), "rows": n,
                    "equal_quarter_mse": mean([a["return_mse"] for a in ss]),
                    "pooled_row_mse": math.fsum(a["rows"] * a["return_mse"] / n for a in ss),
                    "equal_quarter_mae": mean([a["return_mae"] for a in ss]),
                    "zero_return_mse": mean([a["zero_return_mse"] for a in ss]),
                    "fit_mean_return_mse": mean([a["fit_mean_return_mse"] for a in ss]),
                    "mse_minus_zero": mean([a["return_mse"] - a["zero_return_mse"] for a in ss]),
                    "mse_minus_fit_mean": mean([a["return_mse"] - a["fit_mean_return_mse"] for a in ss]),
                    "mean_rank_ic": mean([a["return_rank_ic"] for a in ss]) if all(a["return_rank_ic"] is not None for a in ss) else None}
        paired[regime] = {}
        for m in NEW_MEANS:
            paired[regime][m] = {}
            for ref in REFERENCES[m]:
                predictions = {}
                for seg in SCORE_MEANS:
                    delta = [si[f, seg, m]["return_mse"] - si[f, seg, ref]["return_mse"] for f in fs]
                    base = mean([si[f, seg, ref]["return_mse"] for f in fs])
                    predictions[seg] = {"mse_difference": mean(delta), "relative_mse_reduction": -mean(delta)/base if base else None,
                        "improved_quarters": sum(d < 0 for d in delta), "equal_quarters": sum(d == 0 for d in delta)}
                paired[regime][m][ref] = {"prediction": predictions,
                    "economics": {rule: {c: {k: mean([ri[f, m+"_"+rule][c][k] - ri[f, ref+"_"+rule][c][k] for f in fs])
                        for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")} for c in ("base", "stress_2x")} for rule in RULES}}
    direction = {}
    for m in NEW_MEANS:
        predictive = {seg: all(pred[g][seg][m]["mse_minus_zero"] < 0 and pred[g][seg][m]["mse_minus_fit_mean"] < 0
            and all(paired[g][m][ref]["prediction"][seg]["mse_difference"] < 0 for ref in REFERENCES[m]) for g in econ) for seg in SCORE_MEANS}
        for rule in RULES:
            cid = m+"_"+rule
            direction[cid] = {"economic_means_all_strata_both_costs": all(econ[g][cid][c]["alpha_ex"] > 0
                and econ[g][cid][c]["maxdd_delta"] < 0 for g in econ for c in ("base", "stress_2x")),
                "predictive_mse_vs_zero_fitmean_and_all_references_all_strata": predictive,
                "regime_count_gate_pass": False, "high_probability_generalization_established": False}
    return {"economics": econ, "prediction": pred, "paired": paired, "direction": direction,
        "regime_counts": regime_counts, "interval_regime_strata_are_retrospective_evaluation_groupings": True,
        "individual_feature_effects_identified": False, "selection_performed": False,
        "high_probability_generalization_established": False, "regime_count_gate_pass": False}


def run(config_path):
    cfg, fc, bars, groups, y, masks, forecasts, controls, parent, pre = prepare(config_path)
    out = Path(cfg["output_dir"])
    if (out/"results.json").exists(): raise ValueError("immutable short feature run already completed")
    if file_digest(out/"preflight.json") != cfg["preflight_sha256"] or json.loads((out/"preflight.json").read_text()) != pre:
        raise ValueError("registered short feature preflight changed")
    reg = {"config": cfg, "config_sha256": file_digest(config_path), "preflight_sha256": cfg["preflight_sha256"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_bindings": cfg["source_bindings"], "sklearn_version": sklearn.__version__,
        "scope": "24 new Ridge fits and 8 baseline parity fits; no risk or calibration fit"}
    _immutable_json(out/"registration.json", reg)
    parent_scores = {(a["fold"], a["mean_id"]): a for a in parent["scores"]}
    execution = fc["execution"]; stress = {**execution, "one_way_cost": 2*execution["one_way_cost"], "borrow_annual": 2*execution["borrow_annual"]}
    all_rows, all_scores, all_parity = [], [], []
    for f in FOLDS:
        dates = calendar(f-1); mask = masks[f]
        cal_ix = np.asarray((bars.index >= dates["scale_start"]) & (bars.index < dates["evaluation_start"]))
        eval_ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        window = bars.loc[eval_ix]; ref = forecasts[f, "scale_mean"]; inference = ref["inference_mask"]
        regime = controls[f, "bh"]["regime"]
        fitted = fit_raw_mean_family(groups, y, fit_mask=mask["fit"], predict_mask=mask["predict"])
        old_model = joblib.load(Path("codex_outputs/oracle_derivative_delay_v1/models") / f"fold{f}_technical_return.joblib")
        parity = {"fold": f, "model_state": compare_models(fitted["models"]["technical"], old_model)}
        saved_cal = arrays(Path("codex_outputs/oracle_frozen_procedure_parity_v1/calibration")/f"fold{f}_technical.npz")
        parity["calibration_raw_maxdiff"] = compare_array(fitted["raw"]["technical"][cal_ix], saved_cal["mu"], name="baseline S/I raw")
        parity["evaluation_raw_maxdiff"] = compare_array(fitted["raw"]["technical"][eval_ix], forecasts[f, "technical_raw"]["mu"], name="baseline E raw")
        if fitted["fit_return_mean"] != float(ref["fit_return_mean"]): raise ValueError("baseline fit mean changed")
        rows, scores, bindings = [], [], {}
        def save(kind, name, value, extension="npz"):
            p = out/kind/f"fold{f}_{name}.{extension}"; p.parent.mkdir(parents=True, exist_ok=True)
            if extension == "json": _immutable_json(p, value)
            elif extension == "joblib":
                buf = io.BytesIO(); joblib.dump(value, buf, compress=3); data = buf.getvalue()
                if p.exists():
                    if p.read_bytes() != data: raise ValueError("partial fitted model differs")
                else: p.write_bytes(data)
            elif p.exists():
                prior = arrays(p)
                if set(prior) != set(value): raise ValueError("partial array schema differs")
                for k in prior: compare_array(prior[k], value[k], name=str(p)+k, exact=True)
            else: np.savez_compressed(p, **value)
            bindings[str(p)] = file_digest(p); return p
        eval_means = {m: forecasts[f, m]["mu"] for m in OLD_MEANS}
        eval_means.update({"zero": np.where(inference, 0., np.nan), "fit_mean": np.where(inference, float(ref["fit_return_mean"]), np.nan)})
        cal_mask = mask["scale"][cal_ix] | mask["interval"][cal_ix]
        cal_actual = y[cal_ix].copy(); cal_actual[~cal_mask] = np.nan
        compare_array(cal_actual, saved_cal["actual"], name="original calibration labels", exact=True)
        for seg in ("scale", "interval"):
            compare_array(mask[seg][cal_ix], saved_cal[seg+"_mask"], name="original calibration mask", exact=True)
        interval_means = {"zero": np.where(cal_mask, 0., np.nan), "fit_mean": np.where(cal_mask, float(ref["fit_return_mean"]), np.nan)}
        for g in GROUPS:
            mean_id = g+"_raw"; mu = fitted["raw"][g]
            save("models", g, fitted["models"][g], "joblib")
            save("calibration", g, {"timestamps": bars.index[cal_ix].asi8, "mu": mu[cal_ix], "actual": cal_actual,
                "scale_mask": mask["scale"][cal_ix], "interval_mask": mask["interval"][cal_ix]})
            eval_means[mean_id] = mu[eval_ix]; interval_means[mean_id] = mu[cal_ix]
            save("forecasts", mean_id, {**ref, "mu": mu[eval_ix]})
        save("provenance", "fit", {"fold": f, "fit_provenance": fitted["provenance"], "baseline_parity": parity,
            "fit_source_binding": pre["support"][FOLDS.index(f)], "risk_source": "unchanged technical_scaled"}, "json")
        for seg, means in SCORE_MEANS.items():
            actual, support, mm = (ref["actual"], ref["score_support"], eval_means) if seg == "evaluation" else (cal_actual, mask["interval"][cal_ix], interval_means)
            for m in means:
                score = return_scores(actual, mm[m], support, float(ref["fit_return_mean"]))
                if seg == "evaluation" and m in OLD_MEANS:
                    compare_tree(score, {k: parent_scores[f, m][k] for k in score}, name="unchanged old score")
                scores.append({"fold": f, "segment": seg, "mean_id": m, "regime": regime,
                    "regime_known_at_scored_decisions": seg == "evaluation",
                    "regime_reference": "evaluation_quarter_start", **score})
        for cid in POLICIES:
            trace = None; rule = RULES[1] if cid.endswith(RULES[1]) else RULES[0]
            mean_id = cid[:-(len(rule)+1)]
            if cid in CONTROLS:
                source = FALLBACK_ROOT if cid in EXTRA_CONTROLS else PARENT_ROOT
                saved = arrays(source/"targets"/f"fold{f}_{cid}.npz")
                compare_array(saved["timestamps"], window.index.asi8, name="old target calendar", exact=True); targets = saved["targets"]
            if cid in NEW_IDS or cid.startswith("technical_raw_"):
                if rule == RULES[1]:
                    targets_new, trace = fallback_targets(window, eval_means[mean_id], ref["variance"], execution,
                        inference_mask=inference, risk_aversion=1, cost_multiplier=2)
                    am = action_masks(window.index, window.open.to_numpy(), inference)
                    check_action_support(targets_new, am); check_trace_support(targets_new, am, trace)
                else: targets_new, trace = conditional_targets(window, eval_means[mean_id], ref["variance"], execution, risk_aversion=1, cost_multiplier=2)
                if cid in CONTROLS: compare_array(targets_new, targets, name="refitted baseline own-state targets", exact=True)
                else: targets = targets_new
            p = save("targets", cid, {"timestamps": window.index.asi8, "targets": targets})
            row = {"fold": f, "candidate_id": cid, "regime": regime, "targets_sha256": bindings[str(p)],
                **{c: metrics(window, targets, ex) for c, ex in (("base", execution), ("stress_2x", stress))}}
            if cid in CONTROLS:
                for c in ("base", "stress_2x"): compare_tree(row[c], controls[f, cid][c], name="unchanged control account")
            else:
                p = save("traces", cid, trace, "json"); row["trace_sha256"] = bindings[str(p)]
            rows.append(row)
        if len(bindings) != 53: raise ValueError("incomplete short feature artifact inventory")
        _immutable_json(out/f"fold_{f}.json", {"registration_sha256": digest(reg), "rows": rows, "scores": scores,
            "baseline_parity": parity, "artifact_sha256": bindings})
        all_rows.extend(rows); all_scores.extend(scores); all_parity.append(parity)
        print(json.dumps({"event": "fold_complete", "fold": f, "model_fits": 4, "policies": len(rows), "scores": len(scores), "artifacts": len(bindings)}), flush=True)
    result = {"registration_sha256": digest(reg), "rows": all_rows, "scores": all_scores, "baseline_parity": all_parity,
        "summary": summarize(all_rows, all_scores), "return_model_fits": 32, "new_return_model_fits": 24, "baseline_parity_fits": 8,
        "risk_model_fits": 0, "calibration_weight_fits": 0, "new_causal_policy_names": 6,
        "total_adaptively_explored_causal_names": 174, "additional_test_used_for_modeling_or_scoring": False,
        "selection_performed": False, "teacher_use_allowed": False, "high_probability_generalization_established": False}
    _immutable_json(out/"results.json", result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if args.preflight:
        cfg, *_, pre = prepare(args.config); p = Path(cfg["output_dir"])/"preflight.json"; _immutable_json(p, pre)
        print(json.dumps({"path": str(p), "sha256": file_digest(p), "new_model_fitted": False}))
    else: run(args.config)
