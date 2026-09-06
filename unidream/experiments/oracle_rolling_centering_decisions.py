"""Registered, development-only causal rolling intercept update; no new fits."""
from __future__ import annotations

import argparse
import hashlib
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
from .oracle_frozen_procedure_parity import FOLDS, RULES, compare_array, compare_tree
from .oracle_mean_controls import return_scores
from .oracle_mean_reliability_decisions import (
    prepare as prepare_parent, arrays, GROUPS, MEANS as OLD_MEANS,
    POLICIES as CONTROLS, SOURCES as PARENT_SOURCES,
)
from .oracle_rolling_centering import rolling_centered_forecasts, score_decomposition


PARENT_ROOT = Path("codex_outputs/oracle_mean_reliability_decisions_v1")
NEW_MEANS = ("rolling_anchor", "technical_rolling", "perp_delay0_rolling")
MEANS = OLD_MEANS + NEW_MEANS
NEW_IDS = tuple(m + "_" + r for m in NEW_MEANS for r in RULES)
POLICIES = CONTROLS + NEW_IDS
REFERENCES = {"rolling_anchor": ("scale_mean",), **{g + "_rolling":
    ("rolling_anchor", "scale_mean", g + "_reliability", g + "_scaled", g + "_half") for g in GROUPS}}
SOURCES = PARENT_SOURCES + tuple("unidream/experiments/" + n for n in
    ("oracle_rolling_centering.py", "oracle_rolling_centering_decisions.py"))
FIXED = {"schema": "oracle-rolling-centering-decisions-v1", "development_folds": list(FOLDS),
    "source_prepare_config": "configs/oracle_mean_reliability_decisions_20260906.yaml",
    "parent_root": str(PARENT_ROOT), "parent_source_revision": "ff9bbb92d588b4615ae8352b353a1301751202a7",
    "output_dir": "codex_outputs/oracle_rolling_centering_decisions_v1",
    "data_cutoff": "2023-04-16T13:45:00Z", "groups": list(GROUPS),
    "mean_ids": list(MEANS), "new_mean_ids": list(NEW_MEANS), "control_ids": list(CONTROLS),
    "new_policy_ids": list(NEW_IDS), "rules": list(RULES),
    "segments": ["evaluation"], "history_calendar_months": 3,
    "history_lower_bound": "inclusive_origin_time", "maturity_minutes": 375,
    "maturity_boundary": "less_than_or_equal_to_current_origin",
    "history_support": "same_fold_paired_raw_origin_availability_and_mature_canonical_label",
    "minimum_history_pairs": 64, "minimum_score_rows": 16,
    "insufficient_history_rule": "fail_closed_preserving_every_original_inference_row",
    "weights": "immutable_stage13_scale_fitted_no_updates", "new_weight_fitting_permitted": False,
    "utility_risk_aversion": 1, "utility_cost_multiplier": 2,
    "inference_rows": 2586, "score_rows": 2574, "fallback_rows": 332,
    "missing_current_open_rows": 2, "adaptive_prior_causal_names": 162,
    "new_causal_policy_names": 6, "base_model_fitting_permitted": False,
    "selection_permitted": False, "additional_test_access_permitted": False,
    "interval_width_claims_permitted": False}
EXTRA = {"source_bindings", "source_prepare_config_sha256", "parent_results_sha256",
         "parent_registration_sha256", "parent_preflight_sha256", "preflight_sha256"}


def validate_config(cfg):
    if (set(cfg) != set(FIXED) | EXTRA
            or any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in FIXED.items())
            or set(cfg["source_bindings"]) != set(SOURCES)):
        raise ValueError("unregistered rolling centering family")


def prepare(config_path):
    """Verify old artifacts and as-of support without new means or policies.

    The inherited loader decodes the original parquet before slicing it. Only
    original development bars below cutoff enter labels/history/scoring here.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()); validate_config(cfg)
    direct = {**cfg["source_bindings"], cfg["source_prepare_config"]: cfg["source_prepare_config_sha256"],
        **{str(PARENT_ROOT / (n + ".json")): cfg["parent_" + n + "_sha256"]
           for n in ("results", "registration", "preflight")}}
    for p, h in direct.items():
        if file_digest(Path(p)) != h: raise ValueError("registered source changed: " + p)
    pc, fc, bars, forecasts, _, calibration, raw, parent_pre = prepare_parent(Path(cfg["source_prepare_config"]))
    parent = json.loads((PARENT_ROOT / "results.json").read_text())
    reg = json.loads((PARENT_ROOT / "registration.json").read_text())
    if (reg["config"] != pc or reg["config_sha256"] != cfg["source_prepare_config_sha256"]
            or reg["source_revision"] != cfg["parent_source_revision"]
            or reg["preflight_sha256"] != cfg["parent_preflight_sha256"]
            or parent_pre != json.loads((PARENT_ROOT / "preflight.json").read_text())
            or parent["registration_sha256"] != digest(reg)
            or parent["base_models_fitted"] != 0 or parent["calibration_weights_fitted"] != 16):
        raise ValueError("parent registration/result chain changed")
    bindings = dict(parent_pre["source_artifact_bindings"])
    controls = {(r["fold"], r["candidate_id"]): r for r in parent["rows"]}
    if len(controls) != 128 or set(controls) != {(f, p) for f in FOLDS for p in CONTROLS}:
        raise ValueError("incomplete parent control family")
    canonical = outcome_frame(bars, 24).to_numpy()
    histories, weights, support = {}, {}, []
    for f in FOLDS:
        fp = PARENT_ROOT / f"fold_{f}.json"; fold = json.loads(fp.read_text()); direct[str(fp)] = file_digest(fp)
        if (fold["registration_sha256"] != digest(reg) or len(fold["artifact_sha256"]) != 26
                or any(fold[k] != [r for r in parent[k] if r["fold"] == f] for k in ("rows", "scores", "fits"))):
            raise ValueError("parent fold family changed")
        for p, h in fold["artifact_sha256"].items():
            if file_digest(Path(p)) != h or (p in bindings and bindings[p] != h):
                raise ValueError("parent artifact changed: " + p)
            bindings[p] = h
        dates = calendar(f - 1)
        hi = np.asarray((bars.index >= dates["scale_start"]) & (bars.index < dates["evaluation_end"]))
        index = bars.index[hi]; y = canonical[hi]
        old_scored, raw_pair = None, {}
        for g in GROUPS:
            ca, ra = calibration[f, g], raw[f, g]
            times = np.concatenate((ca["timestamps"], ra["timestamps"]))
            compare_array(times, index.asi8, name="same fold history calendar", exact=True)
            raw_pair[g] = np.concatenate((ca["mu"], ra["mu"]))
            old_scored = np.concatenate((ca["scale_mask"] | ca["interval_mask"], ra["score_support"]))
            m = g + "_reliability"; saved = arrays(PARENT_ROOT / "forecasts" / f"fold{f}_{m}.npz")
            for k in saved:
                if k != "mu": compare_array(saved[k], forecasts[f, "scale_mean"][k], name="parent stream " + k, exact=True)
            forecasts[f, m] = saved; forecasts[f, g + "_raw"] = ra
            wf = json.loads((PARENT_ROOT / "weights" / f"fold{f}_{g}.json").read_text())
            if wf != next(r for r in parent["fits"] if r["fold"] == f and r["group"] == g):
                raise ValueError("weight file/result disagreement")
            w = wf["fit"]["weight"]
            if isinstance(w, bool) or not math.isfinite(w) or not 0 <= w <= 1: raise ValueError("invalid fixed weight")
            weights[f, g] = wf
        paired = np.isfinite(raw_pair[GROUPS[0]])
        compare_array(paired, np.isfinite(raw_pair[GROUPS[1]]), name="paired raw availability", exact=True)
        scheduled = (index.hour % 6 == 0) & (index.minute == 0) & (index.second == 0)
        if np.any(paired & ~scheduled): raise ValueError("history forecast off original clock")
        label = np.isfinite(y[:, 0])
        ref = forecasts[f, "scale_mean"]; current = pd.DatetimeIndex(pd.to_datetime(ref["timestamps"], utc=True))
        counts, restored, membership = [], [], []
        for t in current[ref["inference_mask"]]:
            h = np.asarray((index >= t - pd.DateOffset(months=3)) & (index < t) & (index + MATURITY <= t)) & paired & label
            counts.append(int(h.sum())); restored.append(int((h & ~old_scored).sum()))
            membership.append({"decision_at": t.isoformat(),
                "history_timestamp_sha256": hashlib.sha256(index[h].asi8.tobytes()).hexdigest()})
        if min(counts) < 64: raise ValueError("insufficient historical pairs on original inference support")
        histories[f] = {"index": index, "raw": raw_pair, "actual": y,
                        "history_forecast_mask": paired, "label_available_mask": label}
        support.append({"fold": f, "inference_rows": len(counts), "score_rows": int(ref["score_support"].sum()),
            "minimum_history_pairs": min(counts), "maximum_history_pairs": max(counts),
            "histories_with_restored_boundary_rows": sum(n > 0 for n in restored),
            "restored_pairs_across_histories": sum(restored), "history_counts_sha256": digest(counts),
            "history_membership": membership, "history_membership_sha256": digest(membership),
            "paired_raw_masks_equal": True, "regime": controls[f, CONTROLS[0]]["regime"]})
    pre = {"schema": "oracle-rolling-centering-preflight-v1",
        "config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
        "source_bindings": cfg["source_bindings"], "source_artifact_bindings": bindings,
        "direct_source_bindings": direct, "parent_prepare_preflight_sha256": digest(parent_pre),
        "support": support, "fallback_rows": parent_pre["fallback_rows"],
        "missing_current_open_rows": parent_pre["missing_current_open_rows"],
        "new_weight_fitted": False, "new_forecast_or_policy_computed": False,
        "additional_test_used_for_modeling_or_scoring": False,
        "loader_scope": "Inherited parquet decoding precedes cutoff; semantic research data are strictly below cutoff"}
    return cfg, fc, bars, forecasts, controls, histories, weights, parent, pre


def summarize(rows, scores, fixed_weights):
    ri = {(r["fold"], r["candidate_id"]): r for r in rows}
    si = {(r["fold"], r["mean_id"]): r for r in scores}
    wi = {(r["fold"], r["group"]): r for r in fixed_weights}
    if (len(ri) != len(rows) or set(ri) != {(f, p) for f in FOLDS for p in POLICIES}
            or len(si) != len(scores) or set(si) != {(f, m) for f in FOLDS for m in MEANS}
            or len(wi) != len(fixed_weights) or set(wi) != {(f, g) for f in FOLDS for g in GROUPS}):
        raise ValueError("incomplete rolling family")
    regime_counts = {g: sum(ri[f, CONTROLS[0]]["regime"]["trend"] == g for f in FOLDS)
                     for g in ("bull", "bear", "sideways")}
    if regime_counts != {"bull": 2, "bear": 4, "sideways": 2}: raise ValueError("regime coverage changed")
    for f in FOLDS:
        regime = ri[f, CONTROLS[0]]["regime"]; n = si[f, "scale_mean"]["rows"]
        if type(n) is not int or n < 16: raise ValueError("invalid scored count")
        for p in POLICIES:
            if ri[f, p]["regime"] != regime: raise ValueError("unpaired economic regimes")
            if any(not math.isfinite(ri[f, p][cost][k]) for cost in ("base", "stress_2x")
                   for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")): raise ValueError("nonfinite economics")
        for m in MEANS:
            s = si[f, m]
            if s["regime"] != regime or s["rows"] != n or s["segment"] != "evaluation": raise ValueError("unpaired scores")
            if any(not math.isfinite(s[k]) or s[k] < 0 for k in
                   ("return_mse", "return_mae", "zero_return_mse", "fit_mean_return_mse")): raise ValueError("invalid loss")
        for g in GROUPS:
            w = wi[f, g]["fit"]["weight"]
            if isinstance(w, bool) or not math.isfinite(w) or not 0 <= w <= 1: raise ValueError("invalid fixed weight")
    def mean(values): return math.fsum(v / len(values) for v in values)
    econ, pred, paired = {}, {}, {}
    for regime in ("all", "bull", "bear", "sideways"):
        fs = [f for f in FOLDS if regime == "all" or ri[f, CONTROLS[0]]["regime"]["trend"] == regime]
        econ[regime] = {p: {"quarters": len(fs), "joint_positive_quarters_both_costs": sum(all(
            ri[f, p][c]["alpha_ex"] > 0 and ri[f, p][c]["maxdd_delta"] < 0 for c in ("base", "stress_2x")) for f in fs),
            **{c: {k: mean([ri[f, p][c][k] for f in fs]) for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")}
               for c in ("base", "stress_2x")}} for p in POLICIES}
        pred[regime] = {}
        for m in MEANS:
            ss = [si[f, m] for f in fs]; n = sum(s["rows"] for s in ss)
            pred[regime][m] = {"quarters": len(fs), "rows": n,
                "equal_quarter_mse": mean([s["return_mse"] for s in ss]),
                "pooled_row_mse": math.fsum(s["rows"] * s["return_mse"] / n for s in ss),
                "equal_quarter_mae": mean([s["return_mae"] for s in ss]),
                "zero_return_mse": mean([s["zero_return_mse"] for s in ss]),
                "fit_mean_return_mse": mean([s["fit_mean_return_mse"] for s in ss]),
                "mse_minus_zero": mean([s["return_mse"] - s["zero_return_mse"] for s in ss]),
                "mse_minus_fit_mean": mean([s["return_mse"] - s["fit_mean_return_mse"] for s in ss]),
                "mean_rank_ic": mean([s["return_rank_ic"] for s in ss]) if all(s["return_rank_ic"] is not None for s in ss) else None,
                "relative_to_rolling_anchor_decomposition": {k: mean([s["decomposition"][k] for s in ss]) for k in
                    ("lossdiff", "innovation_secondmoment", "crossmoment", "centered_component", "drift_component", "identityresidual")}}
        paired[regime] = {}
        for m in NEW_MEANS:
            paired[regime][m] = {}
            for ref in REFERENCES[m]:
                delta = [si[f, m]["return_mse"] - si[f, ref]["return_mse"] for f in fs]
                base = mean([si[f, ref]["return_mse"] for f in fs])
                paired[regime][m][ref] = {"prediction": {"mse_difference": mean(delta),
                    "relative_mse_reduction": -mean(delta) / base if base else None,
                    "improved_quarters": sum(d < 0 for d in delta), "equal_quarters": sum(d == 0 for d in delta)},
                    "economics": {rule: {cost: {k: mean([ri[f, m + "_" + rule][cost][k] -
                        ri[f, ref + "_" + rule][cost][k] for f in fs]) for k in ("alpha_ex", "maxdd_delta", "turnover", "trades")}
                        for cost in ("base", "stress_2x")} for rule in RULES}}
    direction = {}
    for m in NEW_MEANS:
        predictive = all(pred[g][m]["mse_minus_zero"] < 0 and pred[g][m]["mse_minus_fit_mean"] < 0
            and all(paired[g][m][r]["prediction"]["mse_difference"] < 0 for r in REFERENCES[m]) for g in econ)
        for rule in RULES:
            cid = m + "_" + rule
            direction[cid] = {"economic_means_all_strata_both_costs": all(econ[g][cid][c]["alpha_ex"] > 0
                and econ[g][cid][c]["maxdd_delta"] < 0 for g in econ for c in ("base", "stress_2x")),
                "predictive_mse_vs_zero_fitmean_and_all_registered_references_all_strata": predictive,
                "regime_count_gate_pass": False, "high_probability_generalization_established": False}
    return {"economics": econ, "prediction": pred, "paired": paired, "direction": direction,
        "fixed_weights": fixed_weights, "regime_counts": regime_counts,
        "selection_performed": False, "high_probability_generalization_established": False,
        "regime_count_gate_pass": False, "intercept_components_separately_identified": False}


def run(config_path):
    cfg, fc, bars, forecasts, controls, histories, weights, parent, pre = prepare(config_path)
    out = Path(cfg["output_dir"])
    if (out / "results.json").exists(): raise ValueError("immutable rolling run already completed")
    if file_digest(out / "preflight.json") != cfg["preflight_sha256"] or json.loads((out / "preflight.json").read_text()) != pre:
        raise ValueError("registered preflight changed")
    reg = {"config": cfg, "config_sha256": file_digest(config_path), "preflight_sha256": cfg["preflight_sha256"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_bindings": cfg["source_bindings"], "scope": "Fixed rolling intercept update; no new base or weight fits"}
    _immutable_json(out / "registration.json", reg)
    parent_scores = {(s["fold"], s["mean_id"]): s for s in parent["scores"] if s["segment"] == "evaluation"}
    execution = fc["execution"]; stress = {**execution, "one_way_cost": 2 * execution["one_way_cost"], "borrow_annual": 2 * execution["borrow_annual"]}
    all_rows, all_scores, all_weights = [], [], []
    for f in FOLDS:
        dates = calendar(f - 1); ix = np.asarray((bars.index >= dates["evaluation_start"]) & (bars.index < dates["evaluation_end"]))
        window = bars.loc[ix]; regime = controls[f, CONTROLS[0]]["regime"]
        ref = forecasts[f, "scale_mean"]; inference = ref["inference_mask"]; h = histories[f]
        rolling = rolling_centered_forecasts(h["index"], h["raw"], h["actual"], window.index,
            history_forecast_mask=h["history_forecast_mask"], label_available_mask=h["label_available_mask"],
            inference_mask=inference, weights={g: weights[f, g]["fit"]["weight"] for g in GROUPS}, minimum_pairs=64)
        compare_array(rolling["available"], inference, name="unchanged inference support", exact=True)
        if digest([int(n) for n in rolling["paired_count"][inference]]) != pre["support"][FOLDS.index(f)]["history_counts_sha256"]:
            raise ValueError("runtime history support differs from data-only preflight")
        membership = [{k: r[k] for k in ("decision_at", "history_timestamp_sha256")} for r in rolling["trace"]]
        if membership != pre["support"][FOLDS.index(f)]["history_membership"]:
            raise ValueError("runtime history membership differs from data-only preflight")
        means = {m: forecasts[f, m]["mu"] for m in OLD_MEANS}; means.update(rolling["means"])
        rows, scores, bindings = [], [], {}
        def save(kind, name, value, extension="npz"):
            p = out / kind / f"fold{f}_{name}.{extension}"; p.parent.mkdir(parents=True, exist_ok=True)
            if extension == "json": _immutable_json(p, value)
            elif p.exists():
                prior = arrays(p)
                if set(prior) != set(value): raise ValueError("partial schema changed")
                for k in prior: compare_array(prior[k], value[k], name=str(p) + k, exact=True)
            else: np.savez_compressed(p, **value)
            bindings[str(p)] = file_digest(p); return p
        for m in NEW_MEANS: save("forecasts", m, {**ref, "mu": means[m]})
        fixed = [weights[f, g] for g in GROUPS]
        save("rolling_traces", "shared_history", {"fold": f, "fixed_weights": fixed,
            "forecast_origin_window_calendar_months": 3, "maturity_minutes_inclusive": 375,
            "source_artifact_bindings": {p: v for p, v in pre["source_artifact_bindings"].items()
                if any(f"fold{f}_{g}" in p for g in GROUPS) and ("calibration/" in p or "forecasts/" in p or "weights/" in p)},
            "decisions": rolling["trace"]}, "json")
        for m in MEANS:
            measured = return_scores(ref["actual"], means[m], ref["score_support"], float(ref["fit_return_mean"]))
            decomposition = score_decomposition(ref["actual"], means[m], means["rolling_anchor"], ref["score_support"])
            compare_array(decomposition["candidate_mse"], measured["return_mse"], name="MSE decomposition", rtol=1e-12, atol=1e-14)
            if m in OLD_MEANS: compare_tree(measured, {k: parent_scores[f, m][k] for k in measured}, name="unchanged old forecast scores")
            scores.append({"fold": f, "segment": "evaluation", "mean_id": m, "regime": regime,
                "regime_known_at_scored_decisions": True, "decomposition_anchor": "rolling_anchor",
                **measured, "decomposition": decomposition})
        for cid in POLICIES:
            trace = None
            if cid in CONTROLS:
                saved = arrays(PARENT_ROOT / "targets" / f"fold{f}_{cid}.npz")
                compare_array(saved["timestamps"], window.index.asi8, name="control calendar", exact=True); target = saved["targets"]
            else:
                rule = RULES[1] if cid.endswith(RULES[1]) else RULES[0]; m = cid[:-(len(rule) + 1)]
                if rule == RULES[1]:
                    target, trace = fallback_targets(window, means[m], ref["variance"], execution,
                        inference_mask=inference, risk_aversion=1, cost_multiplier=2)
                    am = action_masks(window.index, window.open.to_numpy(), inference)
                    check_action_support(target, am); check_trace_support(target, am, trace)
                else: target, trace = conditional_targets(window, means[m], ref["variance"], execution, risk_aversion=1, cost_multiplier=2)
            p = save("targets", cid, {"timestamps": window.index.asi8, "targets": target})
            row = {"fold": f, "candidate_id": cid, "regime": regime, "targets_sha256": bindings[str(p)],
                **{c: metrics(window, target, e) for c, e in (("base", execution), ("stress_2x", stress))}}
            if cid in CONTROLS:
                for c in ("base", "stress_2x"): compare_tree(row[c], controls[f, cid][c], name="unchanged old control")
            else:
                p = save("traces", cid, trace, "json"); row["trace_sha256"] = bindings[str(p)]
                if m != "rolling_anchor" and weights[f, m.removesuffix("_rolling")]["fit"]["weight"] == 0:
                    compare_array(means[m], means["rolling_anchor"], name="exact rolling weight zero", exact=True)
                    old = next(r for r in rows if r["candidate_id"] == "rolling_anchor_" + rule)
                    anchor_targets = arrays(out / "targets" / f"fold{f}_rolling_anchor_{rule}.npz")
                    compare_array(target, anchor_targets["targets"], name="exact rolling endpoint targets", exact=True)
                    for c in ("base", "stress_2x"): compare_tree(row[c], old[c], name="exact rolling endpoint accounts")
            rows.append(row)
        payload = {"registration_sha256": digest(reg), "rows": rows, "scores": scores,
                   "fixed_weights": fixed, "artifact_sha256": bindings}
        if len(bindings) != 32: raise ValueError("incomplete rolling artifact inventory")
        _immutable_json(out / f"fold_{f}.json", payload)
        all_rows.extend(rows); all_scores.extend(scores); all_weights.extend(fixed)
        print(json.dumps({"event": "fold_complete", "fold": f, "policies": len(rows), "scores": len(scores), "artifacts": len(bindings)}), flush=True)
    result = {"registration_sha256": digest(reg), "rows": all_rows, "scores": all_scores, "fixed_weights": all_weights,
        "summary": summarize(all_rows, all_scores, all_weights), "base_models_fitted": 0, "calibration_weights_fitted": 0,
        "fixed_weights_copied": 16, "new_causal_policy_names": 6, "total_adaptively_explored_causal_names": 168,
        "additional_test_used_for_modeling_or_scoring": False, "selection_performed": False,
        "teacher_use_allowed": False, "high_probability_generalization_established": False}
    _immutable_json(out / "results.json", result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if args.preflight:
        cfg, *_, pre = prepare(args.config); p = Path(cfg["output_dir"]) / "preflight.json"; _immutable_json(p, pre)
        print(json.dumps({"path": str(p), "sha256": file_digest(p), "new_forecast_or_policy_computed": False}))
    else: run(args.config)
