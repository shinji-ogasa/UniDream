"""Fixed crossed mean/risk decisions from immutable derivative forecasts.

This adaptive diagnostic changes neither fitted models nor calibration. It
crosses two frozen scaled return/variance sources, using the same causal
support and execution contract as the completed parent experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_search import (digest, file_digest, fold_spec, load_bars, metrics,
                              validate_data_artifact, write_json)
from .oracle_conditional_planner import conditional_targets
from .oracle_frontier import map_outcomes


MODELS = ("technical_scaled", "perp_flow_scaled")
POLICIES = ("point", "utility_risk1")
FOLDS = tuple(range(5, 13))
CONTROL_IDS = ("bh", "common_robust") + tuple(
    model + "_" + policy for model in MODELS
    for policy in ("point", "utility_risk0", "utility_risk1"))
CROSSES = (
    {"name": "mu_perp_flow_scaled__var_technical_scaled",
     "mu": "perp_flow_scaled", "variance": "technical_scaled"},
    {"name": "mu_technical_scaled__var_perp_flow_scaled",
     "mu": "technical_scaled", "variance": "perp_flow_scaled"},
)


def load_paired_forecasts(mean_path, risk_path, *, expected_index):
    """Reject unequal support, then mask forecasts using only causal support."""
    with np.load(mean_path, allow_pickle=False) as mean_saved:
        mean = {k: mean_saved[k].copy() for k in mean_saved.files}
    with np.load(risk_path, allow_pickle=False) as risk_saved:
        risk = {k: risk_saved[k].copy() for k in risk_saved.files}
    n = len(expected_index)
    for key in ("timestamps", "score_support", "inference_mask", "actual"):
        if not np.array_equal(mean[key], risk[key], equal_nan=True):
            raise ValueError(f"unpaired frozen forecast {key}")
    if (mean["timestamps"].shape != (n,) or
            not np.array_equal(mean["timestamps"], expected_index.asi8)):
        raise ValueError("forecast calendar differs from fixed validation window")
    for saved in (mean, risk):
        for key in ("mu", "variance", "inference_mask", "score_support"):
            if saved[key].shape != (n,):
                raise ValueError(f"invalid forecast shape: {key}")
        if (saved["inference_mask"].dtype != np.dtype(bool) or
                saved["score_support"].dtype != np.dtype(bool)):
            raise ValueError("forecast masks must be boolean")
        if saved["actual"].shape != (n, 3):
            raise ValueError("invalid actual outcome shape")
    inference, scoring = mean["inference_mask"], mean["score_support"]
    clock = np.asarray((expected_index.hour % 6 == 0) & (expected_index.minute == 0))
    if np.any(inference & ~clock) or np.any(scoring & ~inference):
        raise ValueError("invalid inference/scoring support")
    if (not np.isfinite(mean["actual"][scoring]).all() or
            not np.isnan(mean["actual"][~scoring]).all()):
        raise ValueError("actual outcomes do not match frozen scoring support")
    mu, variance = mean["mu"].copy(), risk["variance"].copy()
    if (not np.isfinite(mu[inference]).all() or
            not np.isfinite(variance[inference]).all() or
            not (variance[inference] > 0).all()):
        raise ValueError("invalid causal mean/variance forecast")
    # Future label availability must never suppress a causal submitted order.
    mu[~inference], variance[~inference] = np.nan, np.nan
    return mu, variance, inference.copy(), scoring.copy()


def crossed_targets(window, mu, variance, contract, *, policy, cost_multiplier):
    if policy == "point":
        prediction = np.column_stack((mu, np.zeros(len(mu)), np.sqrt(variance)))
        return map_outcomes(prediction, "return"), None
    if policy == "utility_risk1":
        return conditional_targets(window, mu, variance, contract,
            risk_aversion=1, cost_multiplier=cost_multiplier)
    raise ValueError("unregistered crossed policy")


def paired_decision_summaries(rows):
    """Report every registered cross against both same-policy source controls."""
    indexed = {(r["fold"], r["candidate_id"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate decision row")
    expected_ids = set(CONTROL_IDS) | {
        cross["name"] + "_" + policy for cross in CROSSES for policy in POLICIES}
    if set(indexed) != {(f, cid) for f in FOLDS for cid in expected_ids}:
        raise ValueError("unpaired or unexpected decision rows")
    result = {}
    for cross in CROSSES:
        for policy in POLICIES:
            candidate = cross["name"] + "_" + policy
            for reference_model in MODELS:
                reference = reference_model + "_" + policy
                entry = {"candidate": candidate, "reference": reference,
                         "difference_convention": "candidate minus reference",
                         "aggregation": "equal quarter mean", "regimes": {}}
                for regime in ("all", "bull", "bear", "sideways"):
                    folds = [f for f in FOLDS if regime == "all" or
                             indexed[f, candidate]["regime"]["trend"] == regime]
                    for f in folds:
                        if indexed[f, candidate]["regime"] != indexed[f, reference]["regime"]:
                            raise ValueError("unpaired decision regimes")
                    entry["regimes"][regime] = {"quarters": len(folds), **{
                        cost: {metric: float(np.mean([
                            indexed[f, candidate][cost][metric] -
                            indexed[f, reference][cost][metric] for f in folds]))
                            for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades",
                                           "fees_initial_equity_units", "borrow_initial_equity_units")}
                        for cost in ("base", "stress_2x")}}
                result[candidate + "_vs_" + reference] = entry
    return result


def decision_summaries(rows, minimum_quarters):
    """Economic point summaries only; no new interval estimation or selection."""
    def average(selected, cost):
        return {"folds": len(selected),
            "alpha_ex_mean": float(np.mean([r[cost]["alpha_ex"] for r in selected])),
            "maxdd_delta_mean": float(np.mean([r[cost]["maxdd_delta"] for r in selected])),
            "alpha_positive_folds": sum(r[cost]["alpha_ex"] > 0 for r in selected),
            "dd_improved_folds": sum(r[cost]["maxdd_delta"] < 0 for r in selected)}

    result = {}
    for cid in sorted({r["candidate_id"] for r in rows}):
        selected = [r for r in rows if r["candidate_id"] == cid]
        regimes, signs = {}, []
        for regime in ("bull", "bear", "sideways"):
            subset = [r for r in selected if r["regime"]["trend"] == regime]
            regimes[regime] = {"quarters": len(subset)}
            if subset:
                for cost in ("base", "stress_2x"):
                    values = average(subset, cost)
                    regimes[regime][cost] = values
                    signs.extend((values["alpha_ex_mean"], -values["maxdd_delta_mean"]))
        base, stress = average(selected, "base"), average(selected, "stress_2x")
        covered = all(r["quarters"] >= minimum_quarters for r in regimes.values())
        result[cid] = {"base": base, "stress_2x": stress, "regimes": regimes,
            "all_regime_sample_coverage": covered,
            "direction_pass": bool(base["alpha_ex_mean"] > 0 and base["maxdd_delta_mean"] < 0
                                   and stress["alpha_ex_mean"] > 0 and stress["maxdd_delta_mean"] < 0),
            "exploratory_regime_direction_pass": bool(covered and signs and min(signs) > 0),
            "high_probability_generalization_established": False}
    return result


def _immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable JSON changed: {path}")
    else:
        write_json(path, value)


def _copy_verified(source, destination, expected_sha):
    if file_digest(source) != expected_sha:
        raise ValueError(f"source artifact changed: {source}")
    if destination.exists() and file_digest(destination) != expected_sha:
        raise ValueError(f"immutable copied artifact changed: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if file_digest(destination) != expected_sha:
        raise ValueError("copy digest mismatch")


def _save_targets(path, index, targets):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["timestamps"], index.asi8) or
                    not np.array_equal(saved["targets"], targets, equal_nan=True)):
                raise ValueError("immutable target path changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, targets=targets, timestamps=index.asi8)
    return file_digest(path)


def run(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    if (cfg["schema"] != "oracle-derivative-crossed-decisions-v1" or
            cfg["development_folds"] != list(FOLDS) or
            cfg["crosses"] != list(CROSSES) or cfg["policies"] != list(POLICIES) or
            cfg["control_ids"] != list(CONTROL_IDS) or
            cfg["utility_cost_multiplier"] != 2 or cfg["minimum_quarters_per_regime"] != 3):
        raise ValueError("unregistered crossed decision family")
    source_root, output = Path(cfg["source_root"]), Path(cfg["output_dir"])
    if source_root.resolve() == output.resolve() or (output / "results.json").exists():
        raise ValueError("immutable source/results output")
    for name in ("registration", "results"):
        if file_digest(source_root / (name + ".json")) != cfg["source_" + name + "_sha256"]:
            raise ValueError(f"parent {name} changed")
    source_reg = json.loads((source_root / "registration.json").read_text())
    source = json.loads((source_root / "results.json").read_text())
    if source["registration_sha256"] != digest(source_reg):
        raise ValueError("parent result/registration binding mismatch")
    fc = source_reg["source_config"]
    source_config = Path(source_reg["config"]["source_config"])
    if (file_digest(source_config) != source_reg["source_config_sha256"] or
            yaml.safe_load(source_config.read_text()) != fc):
        raise ValueError("parent source config changed")
    for name, expected in source_reg["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"parent helper source changed: {name}")
    data_proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    if data_proof != source_reg["spot_data_proof"]:
        raise ValueError("parent Spot data proof changed")
    score_map = {(s["fold"], s["model_id"]): s for s in source["scores"]}
    if len(score_map) != len(source["scores"]):
        raise ValueError("duplicate parent score rows")
    source_rows = {(r["fold"], r["candidate_id"]): r for r in source["rows"]}
    if len(source_rows) != len(source["rows"]):
        raise ValueError("duplicate parent policy rows")
    forecast_bindings = {}
    for fold in FOLDS:
        for model in MODELS:
            name = f"fold{fold}_{model}.npz"
            expected = score_map[fold, model]["forecast_sha256"]
            if file_digest(source_root / "forecasts" / name) != expected:
                raise ValueError("frozen forecast changed")
            forecast_bindings[name] = expected
    if forecast_bindings != cfg["forecast_sha256"] or len(forecast_bindings) != 16:
        raise ValueError("registered sixteen forecast bindings changed")
    control_bindings, trace_bindings = {}, {}
    for fold in FOLDS:
        for cid in CONTROL_IDS:
            row = source_rows[fold, cid]
            name = f"fold{fold}_{cid}.npz"
            if file_digest(source_root / "targets" / name) != row["targets_sha256"]:
                raise ValueError("source control targets changed")
            control_bindings[name] = row["targets_sha256"]
            if row["diagnostic"] is not None:
                name = f"fold{fold}_{cid.replace('_utility_risk', '_risk')}.json"
                expected = row["diagnostic"]["trace_sha256"]
                if file_digest(source_root / "traces" / name) != expected:
                    raise ValueError("source control trace changed")
                trace_bindings[name] = expected
    helpers = {Path(__file__), Path(conditional_targets.__code__.co_filename),
               Path(metrics.__code__.co_filename), Path(map_outcomes.__code__.co_filename)}
    registration = {"config": cfg, "config_sha256": file_digest(config_path),
        "source_config": fc, "source_config_sha256": file_digest(source_config),
        "source_sha256": {p.name: file_digest(p) for p in sorted(helpers)},
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_registration_sha256": cfg["source_registration_sha256"],
        "source_results_sha256": cfg["source_results_sha256"],
        "forecast_bindings": forecast_bindings, "control_target_bindings": control_bindings,
        "control_trace_bindings": trace_bindings, "data_proof": data_proof,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "scope": "adaptive frozen forecast crossed decisions on reused validation; no selection or formal pass"}
    _immutable_json(output / "registration.json", registration)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    contract = fc["execution"]
    stress = {**contract, "one_way_cost": 2 * contract["one_way_cost"],
              "borrow_annual": 2 * contract["borrow_annual"]}
    rows, support = [], []
    replay_max_difference = 0.
    for fold in FOLDS:
        spec = fold_spec(fold, fc["fold_anchor"])
        index = pd.date_range(spec["val_start"], spec["val_end"], freq="15min", inclusive="left")
        if (spec["val_end"] > pd.Timestamp(fc["data_cutoff"]) or
                not index.isin(bars.index).all()):
            raise ValueError("invalid fixed validation support")
        window = bars.loc[index]
        if window.bar_available.mean() < fc["minimum_bar_coverage"]:
            raise ValueError("insufficient unchanged Spot coverage")
        forecasts = {}
        for cross in CROSSES:
            forecasts[cross["name"]] = load_paired_forecasts(
                source_root / "forecasts" / f"fold{fold}_{cross['mu']}.npz",
                source_root / "forecasts" / f"fold{fold}_{cross['variance']}.npz",
                expected_index=index)
        inference, scoring = forecasts[CROSSES[0]["name"]][2:]
        score = score_map[fold, MODELS[0]]
        if (inference.sum() != score["provenance"]["inference_rows"] or
                scoring.sum() != score["rows"] or
                score["regime"] != score_map[fold, MODELS[1]]["regime"]):
            raise ValueError("parent score/support metadata mismatch")
        support.append({"fold": fold, "validation_start": str(index[0]),
            "validation_end_exclusive": str(spec["val_end"]),
            "inference_rows": int(inference.sum()), "score_rows": int(scoring.sum()),
            "regime": score["regime"]})
        fold_rows = []
        for cid in CONTROL_IDS:
            row = copy.deepcopy(source_rows[fold, cid])
            source_path = source_root / "targets" / f"fold{fold}_{cid}.npz"
            with np.load(source_path, allow_pickle=False) as saved:
                targets = saved["targets"].copy()
                if (not np.array_equal(saved["timestamps"], index.asi8) or
                        targets.shape != (len(window),) or
                        np.any(np.isfinite(targets) & ~inference)):
                    raise ValueError("source control calendar or causal mask mismatch")
            if row["regime"] != score["regime"]:
                raise ValueError("source control regime mismatch")
            for cost, execution in (("base", contract), ("stress_2x", stress)):
                replay = metrics(window, targets, execution)
                if replay.keys() != row[cost].keys():
                    raise ValueError("source control accounting schema mismatch")
                for key in replay:
                    difference = abs(replay[key] - row[cost][key])
                    replay_max_difference = max(replay_max_difference, difference)
                    if not np.isclose(replay[key], row[cost][key], rtol=1e-12, atol=1e-12):
                        raise ValueError(f"source control accounting mismatch: {cid}/{cost}/{key}")
            _copy_verified(source_path, output / "targets" / source_path.name, row["targets_sha256"])
            if row["diagnostic"] is not None:
                name = f"fold{fold}_{cid.replace('_utility_risk', '_risk')}.json"
                _copy_verified(source_root / "traces" / name, output / "traces" / name,
                               row["diagnostic"]["trace_sha256"])
            row["control_source"] = {"results_sha256": cfg["source_results_sha256"],
                                     "targets_path": str(source_path)}
            fold_rows.append(row)
        for cross in CROSSES:
            mu, variance, _, _ = forecasts[cross["name"]]
            for policy in POLICIES:
                cid = cross["name"] + "_" + policy
                targets, diagnostic = crossed_targets(window, mu, variance, contract,
                    policy=policy, cost_multiplier=cfg["utility_cost_multiplier"])
                if np.any(np.isfinite(targets) & ~inference):
                    raise ValueError("crossed targets escaped causal inference mask")
                if diagnostic is not None:
                    trace_path = output / "traces" / f"fold{fold}_{cid}.json"
                    _immutable_json(trace_path, diagnostic)
                    diagnostic = {k: v for k, v in diagnostic.items() if k != "decision_trace"}
                    diagnostic["trace_sha256"] = file_digest(trace_path)
                target_path = output / "targets" / f"fold{fold}_{cid}.npz"
                target_sha = _save_targets(target_path, index, targets)
                fold_rows.append({"fold": fold, "candidate_id": cid, "regime": score["regime"],
                    "base": metrics(window, targets, contract),
                    "stress_2x": metrics(window, targets, stress),
                    "targets_sha256": target_sha, "diagnostic": diagnostic,
                    "metadata": {**support[-1], "mu_source": cross["mu"],
                        "variance_source": cross["variance"], "policy": policy,
                        "mu_forecast_sha256": forecast_bindings[f"fold{fold}_{cross['mu']}.npz"],
                        "variance_forecast_sha256": forecast_bindings[f"fold{fold}_{cross['variance']}.npz"]}})
        _immutable_json(output / f"fold_{fold}.json", {
            "registration_sha256": digest(registration), "rows": fold_rows})
        rows.extend(fold_rows)
        print(json.dumps({"fold": fold, "policies": len(fold_rows),
                          "inference_rows": int(inference.sum())}), flush=True)
    result = {"registration_sha256": digest(registration), "rows": rows,
        "summary": decision_summaries(rows, cfg["minimum_quarters_per_regime"]),
        "paired": paired_decision_summaries(rows), "support": support,
        "control_replay_max_metric_difference": float(replay_max_difference),
        "scope": registration["scope"], "selection_performed": False,
        "high_probability_generalization_established": False}
    _immutable_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
