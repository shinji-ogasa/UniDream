"""Metadata-only calendar and complete-family reporting for frozen confirmation.

No fitting, market loading, selection, p-value estimation, or order submission.
The proposed prospective protocol remains incomplete until a separately audited
receipt-aware runner and inference justification exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ANCHOR = pd.Timestamp("2020-04-16T13:45:00Z")
STEP = pd.Timedelta(minutes=15)
MATURITY = 25 * STEP
FOLDS = tuple(range(26, 38))
MEANS = ("technical_half", "perp_delay0_half")
RULES = ("utility_risk1", "utility_risk1_fallback_bh")
CANDIDATES = tuple(m + "_" + r for m in MEANS for r in RULES)
CONTROLS = ("bh", "common_robust") + tuple(
    m + "_" + r for m in ("scale_mean", "technical_scaled", "perp_delay0_scaled") for r in RULES)
STRATA = ("all", "bull", "bear", "sideways")
COSTS = ("base", "stress_2x")
REFERENCES = {"technical_half": ("scale_mean", "technical_scaled"),
              "perp_delay0_half": ("scale_mean", "perp_delay0_scaled")}
BINDING_PATHS = tuple("unidream/experiments/" + name for name in (
    "oracle_confirmation_contract.py", "oracle_receipt_support.py", "oracle_derivative_delay.py",
    "oracle_derivative_delay_features.py", "oracle_derivative_features.py", "oracle_derivative_ablation.py",
    "oracle_derivative_crossed_decisions.py", "oracle_frontier.py", "oracle_frontier_features.py",
    "oracle_conditional_planner.py", "oracle_risk_calibration.py", "alpha_dd_search.py",
    "alpha_dd_features.py", "robust_overlay.py", "oracle_mean_shrinkage.py", "oracle_fallback_planner.py")) + (
    "docs/experiments/oracle_mean_shrinkage_evidence_20260905/confirmation_access_audit.md",
    "docs/experiments/oracle_mean_shrinkage_evidence_20260905/confirmation_design_draft.md",
    "configs/oracle_derivative_delay_20260905.yaml", "configs/oracle_frontier_20260905.yaml",
    "configs/oracle_mean_shrinkage_decisions_20260905.yaml")
FIXED = {
    "schema": "oracle-confirmation-contract-v1",
    "status": "design_and_metadata_preflight_only",
    "evaluation_folds": list(FOLDS), "evaluation_split": "test",
    "fit_months": 18, "scale_months": 3, "interval_months": 3,
    "evaluation_months": 3, "horizon_bars": 24,
    "minimum_fit_rows": 512, "minimum_scale_rows": 64, "minimum_interval_rows": 64,
    "minimum_quarters_per_regime": 3, "mean_weight": .5,
    "candidate_ids": list(CANDIDATES), "control_ids": list(CONTROLS),
    "inferential_mode": "descriptive_only", "marginal_engine_id": None,
    "family_alpha_target": .05, "selection_permitted": False,
    "performance_early_stopping": False,
    "support_contract": "same_inherited_dependency_set_with_separate_receipt_gate",
    "decision_deadline_seconds": 60, "fill_delay_bars": 1,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(cfg):
    allowed = set(FIXED) | {"family_path", "family_sha256", "bindings", "output_path"}
    if set(cfg) != allowed or any(cfg.get(k) != v for k, v in FIXED.items()):
        raise ValueError("unregistered confirmation contract or override")
    if not isinstance(cfg["bindings"], dict) or not cfg["bindings"]:
        raise ValueError("source bindings required")
    records = [(cfg["family_path"], cfg["family_sha256"]), *cfg["bindings"].items()]
    resolved = set()
    for name, sha in records:
        if not isinstance(name, str) or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("invalid binding")
        try:
            int(sha, 16)
        except ValueError as exc:
            raise ValueError("invalid SHA256") from exc
        actual = Path(name).resolve()
        if actual in resolved:
            raise ValueError("duplicate or aliased source/family binding")
        resolved.add(actual)
    if set(cfg["bindings"]) != set(BINDING_PATHS):
        raise ValueError("unregistered source dependency set")


def calendar(fold):
    """Anchor a NEW 18/3/3 procedure at test_start, never old val_start."""
    if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) or fold < 0:
        raise ValueError("nonnegative integer fold required")
    start = ANCHOR + pd.DateOffset(months=3 * int(fold))
    return {"fold": int(fold), "fit_start": start - pd.DateOffset(months=24),
            "fit_end": start - pd.DateOffset(months=6),
            "scale_start": start - pd.DateOffset(months=6),
            "scale_end": start - pd.DateOffset(months=3),
            "interval_start": start - pd.DateOffset(months=3), "interval_end": start,
            "evaluation_start": start, "evaluation_end": start + pd.DateOffset(months=3)}


def _index(index):
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.tz is None:
        raise ValueError("nonempty timezone-aware DatetimeIndex required")
    ix = index.tz_convert("UTC").as_unit("ns")
    if ix.hasnans or not ix.is_unique or not ix.is_monotonic_increasing:
        raise ValueError("finite unique increasing calendar required")
    if np.any(ix.asi8 % STEP.value) or (len(ix) > 1 and np.any(np.diff(ix.asi8) != STEP.value)):
        raise ValueError("complete UTC 15-minute calendar required; missing rows must be explicit")
    return ix


def segment_masks(index, feature_available, label_available, fold):
    """Future labels affect training/calibration/scoring only, never inference."""
    ix = _index(index)
    arrays = [np.asarray(v) for v in (feature_available, label_available)]
    if any(v.shape != (len(ix),) or v.dtype != np.dtype(bool) for v in arrays):
        raise ValueError("aligned boolean availability arrays required")
    features, labels = arrays
    dates = calendar(fold)
    clock = np.asarray((ix.hour % 6 == 0) & (ix.minute == 0))
    result = {}
    for segment in ("fit", "scale", "interval"):
        result[segment] = features & labels & clock & np.asarray(
            (ix >= dates[segment + "_start"]) & (ix + MATURITY < dates[segment + "_end"]))
    window = np.asarray((ix >= dates["evaluation_start"]) & (ix < dates["evaluation_end"]))
    result["scheduled"] = window & clock
    result["inference"] = result["scheduled"] & features
    result["score"] = result["inference"] & labels & np.asarray(ix + MATURITY <= dates["evaluation_end"])
    result["predict"] = features & clock & np.asarray(
        (ix >= dates["scale_start"]) & (ix < dates["evaluation_end"]))
    return result


def endpoint_inventory():
    endpoints, candidate_components = [], {c: [] for c in CANDIDATES}
    for cid in CANDIDATES:
        for stratum in STRATA:
            for cost in COSTS:
                for metric in ("alpha_ex", "negative_maxdd_delta"):
                    eid = "/".join(("economic", cid, stratum, cost, metric))
                    endpoints.append({"id": eid, "kind": "economic", "candidate_id": cid,
                                      "stratum": stratum, "cost": cost, "metric": metric})
                    candidate_components[cid].append(eid)
    for mean in MEANS:
        for ref in REFERENCES[mean]:
            for stratum in STRATA:
                eid = "/".join(("predictive", mean, ref, stratum, "mse_reduction"))
                endpoints.append({"id": eid, "kind": "predictive", "mean_id": mean,
                                  "reference": ref, "stratum": stratum, "metric": "mse_reduction"})
                for rule in RULES:
                    candidate_components[mean + "_" + rule].append(eid)
    return endpoints, candidate_components


def describe_complete_family(economic_rows, predictive_rows):
    """Report all fixed cohort rows; a missing row fails instead of dropping it.

    Economic inputs contain all 12 policy IDs for all 12 quarters, base/stress
    alpha_ex and maxdd_delta. Predictive inputs contain the five fixed means'
    MSE and common positive integer scored_rows per quarter. Values are ratios,
    not percentage points. Inputs must already have separately verified source,
    mask, receipt and accounting provenance; this function does not prove that.
    """
    def collect(rows, id_key, ids):
        found = {}
        for row in rows:
            if type(row["fold"]) is not int:
                raise ValueError("integer cohort fold required")
            key = (row["fold"], row[id_key])
            if key in found or key[0] not in FOLDS or key[1] not in ids:
                raise ValueError("duplicate or unregistered cohort row")
            if row.get("regime") not in STRATA[1:]:
                raise ValueError("missing or undefined regime")
            found[key] = row
        if set(found) != {(f, i) for f in FOLDS for i in ids}:
            raise ValueError("incomplete cohort; preserve failure, do not drop quarter")
        return found
    econ = collect(economic_rows, "candidate_id", CANDIDATES + CONTROLS)
    mean_ids = ("scale_mean", "technical_scaled", "perp_delay0_scaled") + MEANS
    pred = collect(predictive_rows, "mean_id", mean_ids)
    regimes = {}
    for fold in FOLDS:
        rows = [econ[fold, cid] for cid in CANDIDATES + CONTROLS] + [pred[fold, mid] for mid in mean_ids]
        if len({r["regime"] for r in rows}) != 1:
            raise ValueError("unpaired regime labels")
        regimes[fold] = rows[0]["regime"]
        scored = [pred[fold, mid]["scored_rows"] for mid in mean_ids]
        if any(type(n) is not int or n <= 0 for n in scored) or len(set(scored)) != 1:
            raise ValueError("unpaired or invalid scoring denominators")
    for row in econ.values():
        for cost in COSTS:
            for metric in ("alpha_ex", "maxdd_delta"):
                value = row[cost][metric]
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError("nonfinite economic endpoint")
    for row in pred.values():
        value = row["mse"]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError("invalid forecast loss")
    endpoints, components = endpoint_inventory()
    counts = {g: sum(r == g for r in regimes.values()) for g in STRATA[1:]}
    coverage = all(n >= 3 for n in counts.values())
    summaries = {}
    for item in endpoints:
        folds = [f for f in FOLDS if item["stratum"] == "all" or regimes[f] == item["stratum"]]
        if item["kind"] == "economic":
            key = "alpha_ex" if item["metric"] == "alpha_ex" else "maxdd_delta"
            sign = 1 if key == "alpha_ex" else -1
            values = [sign * econ[f, item["candidate_id"]][item["cost"]][key] for f in folds]
        else:
            values = [pred[f, item["reference"]]["mse"] - pred[f, item["mean_id"]]["mse"] for f in folds]
        value = math.fsum(float(v) / len(values) for v in values) if values else None
        if value is not None and not math.isfinite(value):
            raise ValueError("nonfinite aggregate endpoint")
        summaries[item["id"]] = {"quarters": len(folds), "favorable_mean": value,
                                  "observed_sign_pass": value is not None and value > 0,
                                  "one_sided_p": None}
    candidates = {}
    for cid, ids in components.items():
        economic = all(summaries[e]["observed_sign_pass"] for e in ids if e.startswith("economic/"))
        predictive = all(summaries[e]["observed_sign_pass"] for e in ids if e.startswith("predictive/"))
        candidates[cid] = {"observed_economic_signs": economic, "observed_predictive_signs": predictive,
                           "observed_metric_and_coverage_conditions_met": coverage and economic and predictive,
                           "candidate_primary_p": None, "candidate_holm_p": None,
                           "high_probability_generalization_established": False}
    return {"scope": "descriptive fixed-cohort paired report; provenance must be verified separately",
            "complete_quarter_inventory": True, "complete_bar_calendar_verified": False,
            "protocol_provenance_integrity_verified": False,
            "regime_counts": counts, "regime_coverage": coverage, "endpoints": summaries,
            "candidates": candidates, "inferential_mode": "descriptive_only",
            "family_alpha_validity_established": False, "simultaneous_confidence_level": None,
            "selection_performed": False, "high_probability_generalization_established": False}


def metadata_preflight(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    validate_config(cfg)
    for path, sha in [(cfg["family_path"], cfg["family_sha256"]), *cfg["bindings"].items()]:
        if _sha(path) != sha:
            raise ValueError(f"frozen metadata/source changed: {path}")
    family = json.loads(Path(cfg["family_path"]).read_text())
    if (family["candidate_ids"] != list(CANDIDATES) or set(family["control_ids"]) != set(CONTROLS)
            or family["single_model_selected"] or family["existing_selection_locks_modified"]
            or family["weight_family_closed"] != [0, .5, 1]):
        raise ValueError("frozen family changed")
    for name, sha in family["source_sha256"].items():
        if _sha(Path(__file__).with_name(name)) != sha:
            raise ValueError("frozen development source changed")
    quarters = []
    for fold in FOLDS:
        dates = calendar(fold)
        ix = pd.date_range(dates["fit_start"], dates["evaluation_end"], freq="15min", inclusive="left")
        masks = segment_masks(ix, np.ones(len(ix), bool), np.ones(len(ix), bool), fold)
        quarters.append({k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k, v in dates.items()} | {
            "nominal_counts_assuming_complete_inputs": {k: int(v.sum()) for k, v in masks.items()},
            "first_decision": ix[masks["scheduled"]][0].isoformat(),
            "last_decision": ix[masks["scheduled"]][-1].isoformat()})
    endpoints, components = endpoint_inventory()
    return {"schema": FIXED["schema"], "scope": "metadata/calendar only; no market files or outcomes read",
            "config_sha256": _sha(config_path), "family_sha256": cfg["family_sha256"],
            "bindings": cfg["bindings"], "source_sha256": _sha(__file__), "quarters": quarters,
            "endpoints": endpoints, "candidate_components": components,
            "economic_endpoints": 64, "predictive_endpoints": 16,
            "candidate_component_references": 96, "receipt_deadline_seconds": 60,
            "nominal_receipt_counts_are_not_observed_coverage": True,
            "ready_for_outcome_scoring": False, "prospective_collection_started": False,
            "selection_performed": False, "high_probability_generalization_established": False,
            "remaining_requirements": ["receipt-authenticated complete dependency reconstruction",
                "audited fit/calibration/inference/account adapter and append-only decision log",
                "separately justified and preregistered marginal inference before confirmatory claims"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = metadata_preflight(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    path = Path(cfg["output_path"])
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("immutable preflight changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(json.dumps({"path": str(path), "sha256": _sha(path), "ready_for_outcome_scoring": False}))
