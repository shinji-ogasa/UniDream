"""Development-only scale-fitted reliability of the frozen return forecasts.

No base model is refitted. A continuous convex weight is estimated only on
the original first calibration segment, then held fixed on interval/evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar, MATURITY
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frontier import outcome_frame
from .oracle_frozen_procedure_parity import FOLDS, MEANS as PARENT_MEANS, POLICIES as CONTROLS, RULES, compare_array, compare_tree
from .oracle_information_decomposition import prepare as prepare_parent, SOURCES as PARENT_SOURCES, SOURCE_ROOT
from .oracle_mean_controls import return_scores
from .oracle_mean_shrinkage import half_mean
from .oracle_mean_reliability import fit_reliability, apply_reliability, score_decomposition


GROUPS = ("technical", "perp_delay0")
NEW_MEANS = tuple(g + "_reliability" for g in GROUPS)
RAW_MEANS = tuple(g + "_raw" for g in GROUPS)
MEANS = PARENT_MEANS + RAW_MEANS + NEW_MEANS
NEW_IDS = tuple(m + "_" + r for m in NEW_MEANS for r in RULES)
POLICIES = CONTROLS + NEW_IDS
SEGMENTS = ("scale", "interval", "evaluation")
SOURCES = PARENT_SOURCES + tuple("unidream/experiments/" + n for n in
    ("oracle_mean_reliability.py", "oracle_mean_reliability_decisions.py"))
FIXED = {"schema": "oracle-mean-reliability-decisions-v1", "development_folds": list(FOLDS),
    "source_prepare_config": "configs/oracle_information_decomposition_20260906.yaml",
    "source_root": str(SOURCE_ROOT), "raw_source_root": "codex_outputs/oracle_derivative_delay_v1",
    "output_dir": "codex_outputs/oracle_mean_reliability_decisions_v1",
    "data_cutoff": "2023-04-16T13:45:00Z", "groups": list(GROUPS),
    "mean_ids": list(MEANS), "new_mean_ids": list(NEW_MEANS), "control_ids": list(CONTROLS),
    "new_policy_ids": list(NEW_IDS), "segments": list(SEGMENTS), "rules": list(RULES),
    "weight_bounds": [0., 1.], "fit_segment": "scale_only",
    "weight_objective": "MSE_on_saved_scaled_endpoint_to_exact_scale_anchor_segment",
    "minimum_scale_rows": 64, "minimum_score_rows": 16,
    "utility_risk_aversion": 1, "utility_cost_multiplier": 2,
    "inference_rows": 2586, "score_rows": 2574, "fallback_rows": 332,
    "missing_current_open_rows": 2, "adaptive_prior_causal_names": 158,
    "new_causal_policy_names": 4, "base_model_fitting_permitted": False,
    "selection_permitted": False, "additional_test_access_permitted": False,
    "interval_width_claims_permitted": False}


def validate_config(cfg):
    if (set(cfg) != set(FIXED) | {"source_bindings", "source_prepare_config_sha256", "preflight_sha256"}
            or any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in FIXED.items())
            or set(cfg["source_bindings"]) != set(SOURCES)):
        raise ValueError("unregistered return reliability family")


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def prepare(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text()); validate_config(cfg)
    for p, h in {**cfg["source_bindings"], cfg["source_prepare_config"]: cfg["source_prepare_config_sha256"]}.items():
        if file_digest(Path(p)) != h: raise ValueError("registered input source changed: " + p)
    _, fc, bars, forecasts, controls, parent_pre = prepare_parent(Path(cfg["source_prepare_config"]))
    if fc["data_cutoff"] != cfg["data_cutoff"]: raise ValueError("development cutoff changed")
    bindings = parent_pre["source_artifact_bindings"]
    actual = outcome_frame(bars, 24).to_numpy()
    calibration, raw, support = {}, {}, []
    fallback_count = missing_open_count = 0
    for f in FOLDS:
        dates = calendar(f - 1)
        cal_ix = np.asarray((bars.index >= dates["scale_start"]) & (bars.index < dates["evaluation_start"]))
        eval_ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        window = bars.loc[eval_ix]; ref = forecasts[f, "scale_mean"]
        am = action_masks(window.index, window.open.to_numpy(), ref["inference_mask"])
        fallback_count += int(am["fallback_eligible"].sum()); missing_open_count += int(am["missing_current_open"].sum())
        prov = json.loads((SOURCE_ROOT / "calibration" / f"fold{f}_provenance.json").read_text())
        anchor = float(ref["mu"][ref["inference_mask"]][0])
        counts = {}
        for g in GROUPS:
            cp = SOURCE_ROOT / "calibration" / f"fold{f}_{g}.npz"
            rp = Path(cfg["raw_source_root"]) / "forecasts" / f"fold{f}_{g}_raw.npz"
            if str(cp) not in bindings or str(rp) not in bindings: raise ValueError("calibration/raw source unbound")
            ca, ra = arrays(cp), arrays(rp)
            compare_array(ca["timestamps"], bars.index[cal_ix].asi8, name="calibration calendar", exact=True)
            cm = ca["scale_mask"] | ca["interval_mask"]
            expected = actual[cal_ix].copy(); expected[~cm] = np.nan
            compare_array(ca["actual"], expected, name="calibration actual reconstruction", exact=True)
            if g == GROUPS[1]:
                for k in ("timestamps", "actual", "scale_mask", "interval_mask"):
                    compare_array(ca[k], calibration[f, GROUPS[0]][k], name="paired calibration " + k, exact=True)
            times = pd.DatetimeIndex(pd.to_datetime(ca["timestamps"], utc=True))
            for seg in ("scale", "interval"):
                m = ca[seg + "_mask"]
                if m.dtype != np.dtype(bool) or m.shape != (len(times),) or m.sum() < 64:
                    raise ValueError("invalid calibration support")
                t = times[m]
                if not ((t >= dates[seg + "_start"]) & (t + MATURITY < dates[seg + "_end"])
                        & (t.hour % 6 == 0) & (t.minute == 0)).all(): raise ValueError("calibration maturity/clock changed")
                counts[seg] = int(m.sum())
            if np.any(ca["scale_mask"] & ca["interval_mask"]): raise ValueError("calibration overlap")
            a = math.fsum(float(v) / int(ca["scale_mask"].sum()) for v in ca["actual"][ca["scale_mask"], 0])
            if anchor != a or prov["calibration"]["scale_mean"] != a: raise ValueError("exact scale anchor changed")
            bias = float(prov["calibration"]["return_bias"][g])
            ca["scaled"] = ca["mu"] + bias
            if not np.isfinite(ca["scaled"][cm]).all(): raise ValueError("nonfinite selected calibration predictions")
            for k in ("timestamps", "actual", "score_support", "inference_mask", "fit_return_mean"):
                compare_array(ra[k], ref[k], name="raw source support " + k, exact=True)
            compare_array(ra["mu"] + bias, forecasts[f, g + "_scaled"]["mu"], name="saved bias endpoint", exact=True)
            ca["anchor"] = np.full(len(times), anchor)
            ca["bias"] = np.asarray(bias)
            calibration[f, g], raw[f, g] = ca, ra
        support.append({"fold": f, **{k: dates[k].isoformat() for k in
            ("scale_start", "scale_end", "interval_start", "interval_end", "evaluation_start", "evaluation_end")},
            "scale_rows": counts["scale"], "interval_rows": counts["interval"],
            "inference_rows": int(ref["inference_mask"].sum()), "score_rows": int(ref["score_support"].sum()),
            "regime": controls[f, CONTROLS[0]]["regime"]})
    if (sum(s["inference_rows"] for s in support) != 2586 or sum(s["score_rows"] for s in support) != 2574
            or fallback_count != 332 or missing_open_count != 2): raise ValueError("registered support totals changed")
    pre = {"schema": "oracle-mean-reliability-preflight-v1",
        "config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
        "source_bindings": cfg["source_bindings"], "source_artifact_bindings": bindings,
        "parent_prepare_preflight": parent_pre, "support": support,
        "fallback_rows": fallback_count, "missing_current_open_rows": missing_open_count,
        "new_weight_fitted": False, "new_forecast_or_policy_computed": False,
        "additional_test_accessed": False, "scope": "Data-only original development input/maturity/endpoint verification"}
    return cfg, fc, bars, forecasts, controls, calibration, raw, pre


def summarize(rows, scores, fits):
    ri = {(r["fold"], r["candidate_id"]): r for r in rows}
    si = {(r["fold"], r["segment"], r["mean_id"]): r for r in scores}
    fi = {(r["fold"], r["group"]): r for r in fits}
    if (len(ri) != len(rows) or set(ri) != {(f, p) for f in FOLDS for p in POLICIES}
            or len(si) != len(scores) or set(si) != {(f, s, m) for f in FOLDS for s in SEGMENTS for m in MEANS}
            or len(fi) != len(fits) or set(fi) != {(f, g) for f in FOLDS for g in GROUPS}):
        raise ValueError("incomplete reliability family")
    regime_counts = {g: sum(ri[f, CONTROLS[0]]["regime"]["trend"] == g for f in FOLDS)
                     for g in ("bull", "bear", "sideways")}
    if regime_counts != {"bull": 2, "bear": 4, "sideways": 2}:
        raise ValueError("registered regime coverage changed")
    for f in FOLDS:
        regime = ri[f, CONTROLS[0]]["regime"]
        for p in POLICIES:
            if ri[f, p]["regime"] != regime: raise ValueError("unpaired economic regimes")
            if any(not math.isfinite(ri[f, p][cost][k]) for cost in ("base", "stress_2x")
                   for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")):
                raise ValueError("nonfinite economic summary input")
        for seg in SEGMENTS:
            n = si[f, seg, "scale_mean"]["rows"]
            if type(n) is not int or n < 16: raise ValueError("invalid scored support count")
            for m in MEANS:
                row = si[f, seg, m]
                if row["regime"] != regime or row["rows"] != n: raise ValueError("unpaired predictive inputs")
                if any(not math.isfinite(row[k]) or row[k] < 0 for k in
                       ("return_mse", "return_mae", "zero_return_mse", "fit_mean_return_mse")):
                    raise ValueError("invalid predictive loss")
        for g in GROUPS:
            w = fi[f, g]["fit"]["weight"]
            if isinstance(w, bool) or not math.isfinite(w) or not 0 <= w <= 1:
                raise ValueError("invalid fitted weight")
    def mean(values): return math.fsum(v / len(values) for v in values)
    econ, pred, contrasts = {}, {}, {}
    for regime in ("all", "bull", "bear", "sideways"):
        fs = [f for f in FOLDS if regime == "all" or ri[f, CONTROLS[0]]["regime"]["trend"] == regime]
        econ[regime] = {p: {"quarters": len(fs), **{cost: {k: mean([ri[f, p][cost][k] for f in fs])
            for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")} for cost in ("base", "stress_2x")}} for p in POLICIES}
        pred[regime] = {}
        for seg in SEGMENTS:
            pred[regime][seg] = {}
            for m in MEANS:
                ss = [si[f, seg, m] for f in fs]; n = sum(s["rows"] for s in ss)
                pred[regime][seg][m] = {"quarters": len(fs), "rows": n,
                    "equal_quarter_mse": mean([s["return_mse"] for s in ss]),
                    "pooled_row_mse": math.fsum(s["rows"] * s["return_mse"] / n for s in ss),
                    "equal_quarter_mae": mean([s["return_mae"] for s in ss]),
                    "zero_return_mse": mean([s["zero_return_mse"] for s in ss]),
                    "fit_mean_return_mse": mean([s["fit_mean_return_mse"] for s in ss]),
                    "mse_minus_zero": mean([s["return_mse"] - s["zero_return_mse"] for s in ss]),
                    "mse_minus_fit_mean": mean([s["return_mse"] - s["fit_mean_return_mse"] for s in ss]),
                    "mean_rank_ic": mean([s["return_rank_ic"] for s in ss]) if all(s["return_rank_ic"] is not None for s in ss) else None,
                    "decomposition": {k: mean([s["decomposition"][k] for s in ss]) for k in
                        ("lossdiff", "innovation_secondmoment", "crossmoment", "centered_component", "drift_component", "identityresidual")}}
        contrasts[regime] = {}
        for g in GROUPS:
            m = g + "_reliability"; half = "technical_half" if g == "technical" else "perp_delay0_half"
            contrasts[regime][m] = {}
            for ref in ("scale_mean", g + "_scaled", half):
                per_seg = {}
                for seg in SEGMENTS:
                    delta = [si[f, seg, m]["return_mse"] - si[f, seg, ref]["return_mse"] for f in fs]
                    ref_mse = mean([si[f, seg, ref]["return_mse"] for f in fs])
                    per_seg[seg] = {"mse_difference": mean(delta), "relative_mse_reduction": -mean(delta) / ref_mse if ref_mse else None,
                        "improved_quarters": sum(d < 0 for d in delta), "equal_quarters": sum(d == 0 for d in delta)}
                contrasts[regime][m][ref] = {"prediction": per_seg, "economics": {rule: {cost: {k: mean([
                    ri[f, m + "_" + rule][cost][k] - ri[f, ref + "_" + rule][cost][k] for f in fs])
                    for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")} for cost in ("base", "stress_2x")} for rule in RULES}}
    direction = {}
    for g in GROUPS:
        m = g + "_reliability"
        predictive = {seg: all(pred[regime][seg][m]["mse_minus_zero"] < 0 and
            all(contrasts[regime][m][ref]["prediction"][seg]["mse_difference"] < 0 for ref in contrasts[regime][m])
            for regime in econ) for seg in ("interval", "evaluation")}
        for rule in RULES:
            cid = m + "_" + rule
            direction[cid] = {"economic_means_all_strata_both_costs": all(
                econ[regime][cid][cost]["alpha_ex"] > 0 and econ[regime][cid][cost]["maxdd_delta"] < 0
                for regime in econ for cost in ("base", "stress_2x")),
                "predictive_mse_vs_zero_scale_full_half_all_strata": predictive,
                "regime_count_gate_pass": False, "high_probability_generalization_established": False}
    return {"economics": econ, "prediction": pred, "paired": contrasts, "direction": direction,
        "fitted_weights": [{"fold": f, "group": g, **fi[f, g]["fit"]} for f in FOLDS for g in GROUPS],
        "calibration_regime_strata_are_retrospective_evaluation_quarter_groupings": True,
        "selection_performed": False, "new_information_established": False,
        "high_probability_generalization_established": False, "regime_count_gate_pass": False}


def run(config_path):
    cfg, fc, bars, forecasts, controls, calibration, raw, pre = prepare(config_path)
    out = Path(cfg["output_dir"])
    if (out / "results.json").exists(): raise ValueError("immutable reliability run already completed")
    if file_digest(out / "preflight.json") != cfg["preflight_sha256"] or json.loads((out / "preflight.json").read_text()) != pre:
        raise ValueError("registered reliability preflight changed")
    reg = {"config": cfg, "config_sha256": file_digest(config_path), "preflight_sha256": cfg["preflight_sha256"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_bindings": cfg["source_bindings"], "scope": "16 scale-only weights; no base fits; original development only"}
    _immutable_json(out / "registration.json", reg)
    parent_scores = {(r["fold"], r["mean_id"]): r for r in json.loads((SOURCE_ROOT / "results.json").read_text())["scores"]}
    execution = fc["execution"]; stress = {**execution, "one_way_cost": 2 * execution["one_way_cost"], "borrow_annual": 2 * execution["borrow_annual"]}
    all_rows, all_scores, all_fits = [], [], []
    for f in FOLDS:
        dates = calendar(f - 1); ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        window = bars.loc[ix]; regime = controls[f, CONTROLS[0]]["regime"]
        ref = forecasts[f, "scale_mean"]; inference = ref["inference_mask"]
        rows, scores, fits, bindings = [], [], [], {}
        def save(kind, name, value, extension="npz"):
            p = out / kind / f"fold{f}_{name}.{extension}"; p.parent.mkdir(parents=True, exist_ok=True)
            if extension == "json": _immutable_json(p, value)
            elif p.exists():
                prior = arrays(p)
                if set(prior) != set(value): raise ValueError("partial schema changed")
                for k in prior: compare_array(prior[k], value[k], name=str(p) + k, exact=True)
            else: np.savez_compressed(p, **value)
            bindings[str(p)] = file_digest(p); return p
        eval_means = {m: forecasts[f, m]["mu"] for m in PARENT_MEANS}
        cal_means = {}; c0 = calibration[f, GROUPS[0]]; cm = c0["scale_mask"] | c0["interval_mask"]
        cal_means["scale_mean"] = np.where(cm, c0["anchor"], np.nan)
        for g in GROUPS:
            ca = calibration[f, g]; m = g + "_reliability"; full = g + "_scaled"
            fit = fit_reliability(ca["scaled"], ca["actual"], scale_mask=ca["scale_mask"], anchor=float(ca["anchor"][0]))
            fitrow = {"fold": f, "group": g, "fit": fit, "fit_period": "scale", "interval_or_evaluation_used": False}
            fits.append(fitrow); save("weights", g, fitrow, "json")
            eval_means[g + "_raw"] = raw[f, g]["mu"]
            eval_means[m] = apply_reliability(eval_means[full], ref["mu"], inference_mask=inference, weight=fit["weight"])
            save("forecasts", m, {**ref, "mu": eval_means[m]})
            half = "technical_half" if g == "technical" else "perp_delay0_half"
            cal_means[g + "_raw"] = np.where(cm, ca["mu"], np.nan)
            cal_means[full] = np.where(cm, ca["scaled"], np.nan)
            cal_means[half] = half_mean(cal_means[full], cal_means["scale_mean"], inference_mask=cm)
            cal_means[m] = apply_reliability(cal_means[full], cal_means["scale_mean"], inference_mask=cm, weight=fit["weight"])
            save("calibration", g, {"timestamps": ca["timestamps"], "actual": ca["actual"],
                "scale_mask": ca["scale_mask"], "interval_mask": ca["interval_mask"],
                "raw": cal_means[g + "_raw"], "scaled": cal_means[full], "half": cal_means[half],
                "reliability": cal_means[m], "anchor": cal_means["scale_mean"]})
        for seg in SEGMENTS:
            actual, mask, mm, anchor = (ref["actual"], ref["score_support"], eval_means, ref["mu"]) if seg == "evaluation" else (
                c0["actual"], c0[seg + "_mask"], cal_means, cal_means["scale_mean"])
            for m in MEANS:
                measured = return_scores(actual, mm[m], mask, float(ref["fit_return_mean"]))
                decomp = score_decomposition(actual, mm[m], anchor, mask)
                compare_array(decomp["candidate_mse"], measured["return_mse"], name="MSE decomposition", rtol=1e-12, atol=1e-14)
                if seg == "evaluation" and m in PARENT_MEANS:
                    compare_tree(measured, {k: parent_scores[f, m][k] for k in measured}, name="unchanged parent score")
                scores.append({"fold": f, "segment": seg, "mean_id": m, "regime": regime,
                    "regime_known_at_scored_decisions": seg == "evaluation",
                    "regime_reference": "evaluation_quarter_start", "scale_fit_in_sample": seg == "scale",
                    **measured, "decomposition": decomp})
        for cid in POLICIES:
            trace = None
            if cid in CONTROLS:
                saved = arrays(SOURCE_ROOT / "targets" / f"fold{f}_{cid}.npz")
                compare_array(saved["timestamps"], window.index.asi8, name="control calendar", exact=True); target = saved["targets"]
            else:
                rule = RULES[1] if cid.endswith(RULES[1]) else RULES[0]; m = cid[:-(len(rule) + 1)]
                if rule == RULES[1]:
                    target, trace = fallback_targets(window, eval_means[m], ref["variance"], execution,
                        inference_mask=inference, risk_aversion=1, cost_multiplier=2)
                    am = action_masks(window.index, window.open.to_numpy(), inference)
                    check_action_support(target, am); check_trace_support(target, am, trace)
                else: target, trace = conditional_targets(window, eval_means[m], ref["variance"], execution, risk_aversion=1, cost_multiplier=2)
            p = save("targets", cid, {"timestamps": window.index.asi8, "targets": target})
            row = {"fold": f, "candidate_id": cid, "regime": regime, "targets_sha256": bindings[str(p)],
                **{cost: metrics(window, target, c) for cost, c in (("base", execution), ("stress_2x", stress))}}
            if cid in CONTROLS:
                for cost in ("base", "stress_2x"): compare_tree(row[cost], controls[f, cid][cost], name="original control")
            else:
                p = save("traces", cid, trace, "json"); row["trace_sha256"] = bindings[str(p)]
                g = m.removesuffix("_reliability"); w = next(z["fit"]["weight"] for z in fits if z["group"] == g)
                if w in (0., 1.):
                    endpoint = "scale_mean" if w == 0 else g + "_scaled"
                    compare_array(eval_means[m], eval_means[endpoint], name="exact weight endpoint", exact=True)
                    expected = arrays(SOURCE_ROOT / "targets" / f"fold{f}_{endpoint}_{rule}.npz")
                    compare_array(target, expected["targets"], name="exact endpoint targets", exact=True)
                    for cost in ("base", "stress_2x"):
                        compare_tree(row[cost], controls[f, endpoint + "_" + rule][cost], name="endpoint account")
            rows.append(row)
        payload = {"registration_sha256": digest(reg), "rows": rows, "scores": scores, "fits": fits, "artifact_sha256": bindings}
        if len(bindings) != 26: raise ValueError("incomplete fold artifact inventory")
        _immutable_json(out / f"fold_{f}.json", payload)
        all_rows.extend(rows); all_scores.extend(scores); all_fits.extend(fits)
        print(json.dumps({"event": "fold_complete", "fold": f, "policies": len(rows), "scores": len(scores), "artifacts": len(bindings)}), flush=True)
    result = {"registration_sha256": digest(reg), "rows": all_rows, "scores": all_scores, "fits": all_fits,
        "summary": summarize(all_rows, all_scores, all_fits), "base_models_fitted": 0, "calibration_weights_fitted": 16,
        "new_causal_policy_names": 4, "total_adaptively_explored_causal_names": 162,
        "additional_test_accessed": False, "selection_performed": False, "teacher_use_allowed": False}
    _immutable_json(out / "results.json", result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if args.preflight:
        cfg, *_, pre = prepare(args.config); p = Path(cfg["output_dir"]) / "preflight.json"; _immutable_json(p, pre)
        print(json.dumps({"path": str(p), "sha256": file_digest(p), "new_weight_fitted": False}))
    else: run(args.config)
