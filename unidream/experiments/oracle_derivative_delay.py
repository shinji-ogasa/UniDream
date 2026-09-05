"""Registered UM input-age sensitivity with refits and frozen inference stress."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alpha_dd_search import (digest, file_digest, fold_spec, load_bars, metrics,
                              validate_data_artifact, write_json)
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import (forecast_summaries, mask_digest,
                                        score_forecast, validate_um)
from .oracle_derivative_crossed_decisions import decision_summaries
from .oracle_derivative_delay_features import make_delayed_perp_groups
from .oracle_derivative_features import make_derivative_groups
from .oracle_frontier import fit_mask, map_outcomes, outcome_frame, quarter_regime
from .oracle_frontier_features import make_feature_groups
from .oracle_risk_calibration import corrected_quantile, scale_and_bias, trailing_variances
from .robust_overlay import build_targets

GROUPS = ("technical", "perp_delay0", "perp_delay1", "perp_delay4")
FROZEN = ("frozen_delay1", "frozen_delay4")
VERSIONS = ("raw", "scaled")
POLICIES = ("point", "utility_risk1")
FOLDS = tuple(range(5, 13))
PAIRS = (("perp_delay0", "technical"), ("perp_delay1", "technical"),
         ("perp_delay4", "technical"), ("perp_delay1", "perp_delay0"),
         ("perp_delay4", "perp_delay0"), ("frozen_delay1", "perp_delay0"),
         ("frozen_delay4", "perp_delay0"), ("perp_delay1", "frozen_delay1"),
         ("perp_delay4", "frozen_delay4"))


def validate_config(cfg):
    fixed = {"schema": "oracle-derivative-delay-v1", "development_folds": list(FOLDS),
             "extra_delay_bars": [0, 1, 4], "horizon_bars": 24,
             "fit_months": 18, "scale_months": 3, "interval_months": 3,
             "versions": list(VERSIONS), "utility_risk_aversions": [1],
             "utility_cost_multiplier": 2, "return_ridge_alpha": 100.0}
    if any(cfg.get(k) != value for k, value in fixed.items()):
        raise ValueError("unregistered delay family")


def source_hashes():
    names = ("oracle_derivative_delay.py", "oracle_derivative_delay_features.py",
             "oracle_derivative_features.py", "oracle_derivative_ablation.py",
             "oracle_derivative_crossed_decisions.py", "oracle_frontier.py",
             "oracle_frontier_features.py", "oracle_conditional_planner.py",
             "oracle_risk_calibration.py", "alpha_dd_search.py", "alpha_dd_features.py",
             "robust_overlay.py")
    return {name: file_digest(Path(__file__).with_name(name)) for name in names}


def segment_masks(index, common, valid_labels, cfg, val_start, val_end):
    """Label support enters only fit/calibration/scoring, never inference."""
    scale_start = val_start-pd.DateOffset(months=6)
    interval_start = val_start-pd.DateOffset(months=3)
    train_start = scale_start-pd.DateOffset(months=18)
    clock = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
    masks = {name: fit_mask(index, common & valid_labels, start=start, end=end,
                           horizon=cfg["horizon_bars"], cadence_hours=6)
             for name, start, end in (("fit", train_start, scale_start),
                                     ("scale", scale_start, interval_start),
                                     ("interval", interval_start, val_start))}
    window = np.asarray((index >= val_start) & (index < val_end))
    masks["inference"] = window & common & clock
    masks["score"] = masks["inference"] & valid_labels & np.asarray(
        index+pd.Timedelta(minutes=15*(cfg["horizon_bars"]+1)) <= val_end)
    masks["predict"] = common & clock & np.asarray((index >= scale_start) & (index < val_end))
    return masks, {"train_start": str(train_start), "train_end": str(scale_start),
                   "scale_end": str(interval_start), "interval_end": str(val_start)}


def prepare(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    validate_config(cfg)
    fc_path = Path(cfg["source_config"])
    fc = yaml.safe_load(fc_path.read_text())
    if fc["data_cutoff"] != "2023-04-16T13:45:00Z":
        raise ValueError("unregistered data cutoff")
    data_proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    um, um_proof = validate_um(Path(cfg["um_path"]), fc["data_cutoff"], fc["symbol"])
    parent_preflight = Path(cfg["parent_preflight_path"])
    if file_digest(parent_preflight) != cfg["parent_preflight_sha256"]:
        raise ValueError("parent preflight changed")
    parent = json.loads(parent_preflight.read_text())
    for path, expected in parent["source_sha256"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"parent input/source changed: {path}")
    if (data_proof["artifact_sha256"] != parent["source_sha256"][fc["data_path"]]
            or um_proof["data_sha256"] != parent["source_sha256"][str(Path(cfg["um_path"]).resolve())]):
        raise ValueError("parent preflight/data binding mismatch")
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    groups = make_delayed_perp_groups(bars, um, delays=tuple(cfg["extra_delay_bars"]))
    original, derivative = make_feature_groups(bars), make_derivative_groups(bars, um)
    tv = trailing_variances(bars, cfg["horizon_bars"])
    clock = np.asarray((bars.index.hour % 6 == 0) & (bars.index.minute == 0))
    previous = np.isfinite(tv.to_numpy()).all(axis=1)
    previous &= np.isfinite(original["flow"].to_numpy()).all(axis=1)
    for frame in derivative.values():
        previous &= np.isfinite(frame.to_numpy()).all(axis=1)
    if mask_digest(bars.index, previous & clock) != parent["shared_mask"]["shared_6h_mask_sha256"]:
        raise ValueError("original support no longer matches parent preflight")
    common = previous.copy()
    for group in GROUPS:
        common &= np.isfinite(groups[group].to_numpy()).all(axis=1)
    y = outcome_frame(bars, cfg["horizon_bars"]).to_numpy()
    valid = np.isfinite(y).all(axis=1)
    folds, fold_masks = [], {}
    for fold in FOLDS:
        spec = fold_spec(fold, fc["fold_anchor"])
        vs, ve = spec["val_start"], spec["val_end"]
        if ve > pd.Timestamp(fc["data_cutoff"]):
            raise ValueError("validation beyond cutoff")
        masks, dates = segment_masks(bars.index, common, valid, cfg, vs, ve)
        fold_masks[fold] = masks
        ix = np.asarray((bars.index >= vs) & (bars.index < ve))
        window = bars.loc[ix]
        if (not window.index.equals(pd.date_range(vs, ve, freq="15min", inclusive="left"))
                or window.bar_available.mean() < fc["minimum_bar_coverage"]):
            raise ValueError("incomplete validation window")
        for name in ("fit", "scale", "interval"):
            if masks[name].sum() < cfg["minimum_"+name+"_rows"]:
                raise ValueError(f"insufficient {name} rows in fold {fold}")
        if masks["score"].sum() < 16:
            raise ValueError("insufficient forecast scoring rows")
        folds.append({"fold": fold, **dates, "validation_start": str(vs),
            "validation_end_exclusive": str(ve),
            "regime": quarter_regime(original["base16"], ix, fc["regime"]["normalized_momentum_90_threshold"]),
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "mask_sha256": {name: mask_digest(bars.index, mask) for name, mask in masks.items()},
            "last_label_end": {name: str(bars.index[masks[name]][-1]+pd.Timedelta(minutes=375))
                               for name in ("fit", "scale", "interval")},
            "parent_inference_rows": int((previous & clock & ix).sum()),
            "scheduled_rows": int((clock & ix).sum())})
    preflight = {"scope": "data-only support; no model fit or policy performance",
        "config_contract_sha256": digest({k: v for k, v in cfg.items() if k != "preflight_sha256"}),
        "source_config_sha256": file_digest(fc_path), "source_sha256": source_hashes(),
        "spot_data_proof": data_proof, "um_data_proof": um_proof,
        "parent_preflight_sha256": file_digest(parent_preflight),
        "features": {g: list(groups[g].columns) for g in GROUPS},
        "extra_delay_bars": cfg["extra_delay_bars"],
        "full_common_6h_mask_sha256": mask_digest(bars.index, common & clock),
        "folds": folds, "live_receipt_provenance_established": False,
        "mask_scope": "retrospective intersection includes undelayed availability; not an operational delayed-feed mask"}
    return cfg, fc, bars, groups, original, tv, y, fold_masks, preflight


def immutable_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"immutable JSON changed: {path}")
    if not path.exists():
        write_json(path, value)


def paired_results(scores, rows):
    score_index = {(r["fold"], r["model_id"]): r for r in scores}
    row_index = {(r["fold"], r["candidate_id"]): r for r in rows}
    expected_models = {g+"_"+v for g in GROUPS+FROZEN for v in VERSIONS}
    expected_policies = {m+"_"+p for m in expected_models for p in POLICIES} | {"bh", "common_robust"}
    if (len(score_index) != len(scores) or len(row_index) != len(rows)
            or set(score_index) != {(f, m) for f in FOLDS for m in expected_models}
            or set(row_index) != {(f, p) for f in FOLDS for p in expected_policies}):
        raise ValueError("missing, duplicate or unexpected delay results")
    for fold in FOLDS:
        regime = score_index[fold, "technical_raw"]["regime"]
        if (any(score_index[fold, mid]["regime"] != regime for mid in expected_models)
                or any(row_index[fold, cid]["regime"] != regime for cid in expected_policies)):
            raise ValueError("unpaired result regimes")
    out = {}
    for candidate, reference in PAIRS:
        for version in VERSIONS:
            cmid, rmid = candidate+"_"+version, reference+"_"+version
            record = {"candidate": cmid, "reference": rmid, "regimes": {}}
            for regime in ("all", "bull", "bear", "sideways"):
                folds = [f for f in FOLDS if regime == "all" or score_index[f, cmid]["regime"]["trend"] == regime]
                cs, rs = [score_index[f, cmid] for f in folds], [score_index[f, rmid] for f in folds]
                if any(c["regime"] != r["regime"] or c["rows"] != r["rows"] for c, r in zip(cs, rs)):
                    raise ValueError("unpaired forecast support")
                forecast = {}
                for metric in ("return_mse", "qlike", "variance_mse", "rms_mse"):
                    delta = np.asarray([c[metric]-r[metric] for c, r in zip(cs, rs)])
                    mean_reference = float(np.mean([r[metric] for r in rs]))
                    forecast[metric] = {"mean_difference": float(delta.mean()),
                        "relative_loss_reduction": -float(delta.mean())/mean_reference if mean_reference > 0 else None,
                        "improved_quarters": int((delta < 0).sum())}
                policies = {policy: {cost: {metric: float(np.mean([
                    row_index[f, cmid+"_"+policy][cost][metric]-row_index[f, rmid+"_"+policy][cost][metric]
                    for f in folds])) for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades")}
                    for cost in ("base", "stress_2x")} for policy in POLICIES}
                record["regimes"][regime] = {"quarters": len(folds), "forecast": forecast, "policies": policies}
            out[cmid+"_vs_"+rmid] = record
    return out


def run(config_path):
    cfg, fc, bars, groups, original, tv, y, fold_masks, preflight = prepare(config_path)
    preflight_path = Path(cfg["preflight_path"])
    if (file_digest(preflight_path) != cfg["preflight_sha256"]
            or json.loads(preflight_path.read_text()) != preflight):
        raise ValueError("data preflight differs from registered source/support")
    output = Path(cfg["output_dir"])
    if (output/"results.json").exists():
        raise ValueError("immutable experiment already completed")
    registration = {"config": cfg, "config_sha256": file_digest(config_path),
        "source_config": fc, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "preflight_sha256": file_digest(preflight_path), "source_sha256": source_hashes(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "scope": "adaptive reused validation input-age diagnostic; no selection or new inference"}
    immutable_json(output/"registration.json", registration)
    reg_sha = digest(registration)
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": 2*execution["one_way_cost"], "borrow_annual": 2*execution["borrow_annual"]}
    actual_var = np.maximum(y[:, 2]**2, cfg["variance_floor"])
    rows, scores = [], []
    for fold, support in zip(FOLDS, preflight["folds"]):
        fold_path = output/f"fold_{fold}.json"
        if fold_path.exists():
            saved = json.loads(fold_path.read_text())
            if saved["registration_sha256"] != reg_sha:
                raise ValueError("saved fold source changed")
            for path, expected in saved["artifact_sha256"].items():
                if file_digest(Path(path)) != expected:
                    raise ValueError("saved fold artifact changed")
            rows.extend(saved["rows"]); scores.extend(saved["scores"])
            continue
        m = fold_masks[fold]
        train, scale, interval, inference, scoring = [m[name] for name in ("fit", "scale", "interval", "inference", "score")]
        vs, ve = pd.Timestamp(support["validation_start"]), pd.Timestamp(support["validation_end_exclusive"])
        ix = np.asarray((bars.index >= vs) & (bars.index < ve))
        cal_ix = np.asarray((bars.index >= pd.Timestamp(support["train_end"])) & (bars.index < vs))
        window = bars.loc[ix]
        fit_mean = float(y[train, 0].mean())
        fold_rows, fold_scores, artifacts, cache = [], [], {}, {}
        def save_npz(kind, name, **arrays):
            path = output/kind/f"fold{fold}_{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            artifacts[str(path)] = file_digest(path)
            return {"path": str(path), "sha256": artifacts[str(path)]}
        def evaluate(cid, targets, diagnostic=None):
            targets = np.asarray(targets, float).copy()
            targets[~inference[ix]] = np.nan
            proof = save_npz("targets", cid, targets=targets, timestamps=window.index.asi8)
            fold_rows.append({"fold": fold, "candidate_id": cid, "regime": support["regime"],
                "base": metrics(window, targets, execution), "stress_2x": metrics(window, targets, stress),
                "targets_sha256": proof["sha256"], "diagnostic": diagnostic, "metadata": support})
        evaluate("bh", np.full(len(window), np.nan))
        evaluate("common_robust", build_targets(original["base16"])[ix])
        for group in GROUPS+FROZEN:
            frozen = group in FROZEN
            x_group = group.replace("frozen_", "perp_") if frozen else group
            x = groups[x_group].to_numpy(float)
            predict = inference if frozen else m["predict"]
            mu, logvar = np.full(len(bars), np.nan), np.full(len(bars), np.nan)
            if frozen:
                base = cache["perp_delay0"]
                mean_model, risk_model = base["models"]
                model_proof = {**base["proof"], "input_group": x_group,
                    "inference_delay_bars": int(group[-1]), "intervention": "frozen_validation_input_only"}
            else:
                mean_model = make_pipeline(StandardScaler(), Ridge(alpha=cfg["return_ridge_alpha"]))
                risk_model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=64,
                    learning_rate=.04, l2_regularization=10., early_stopping=False, random_state=cfg["seed"])
                with threadpool_limits(limits=2):
                    mean_model.fit(x[train], y[train, 0])
                    risk_model.fit(x[train], np.log(actual_var[train]))
                model_proof = {"training_group": group, "input_group": x_group,
                    "features": list(groups[x_group].columns), "fit_return_mean": fit_mean,
                    "training_delay_bars": None if group == "technical" else int(group[-1]),
                    "inference_delay_bars": None if group == "technical" else int(group[-1]),
                    "intervention": "refit_with_available_inputs", "models": {}}
                for label, model in (("return", mean_model), ("variance", risk_model)):
                    path = output/"models"/f"fold{fold}_{group}_{label}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, path); artifacts[str(path)] = file_digest(path)
                    model_proof["models"][label] = {"path": str(path), "sha256": artifacts[str(path)]}
            with threadpool_limits(limits=2):
                mu[predict] = mean_model.predict(x[predict])
                logvar[predict] = risk_model.predict(x[predict])
            if not np.isfinite(mu[predict]).all() or not np.isfinite(logvar[predict]).all():
                raise ValueError("nonfinite forecast")
            variance = np.exp(np.clip(logvar, np.log(cfg["variance_floor"]), 0))
            if frozen:
                bias, multiplier = base["bias"], base["multiplier"]
            else:
                bias, multiplier = scale_and_bias(y[scale, 0], actual_var[scale], mu[scale], variance[scale])
                cal_actual = y[cal_ix].copy(); cal_actual[~(scale | interval)[cal_ix]] = np.nan
                model_proof["calibration"] = save_npz("calibration", group, timestamps=bars.index[cal_ix].asi8,
                    mu=mu[cal_ix], log_variance=logvar[cal_ix], variance=variance[cal_ix], actual=cal_actual,
                    scale_mask=scale[cal_ix], interval_mask=interval[cal_ix])
                cache[group] = {"models": (mean_model, risk_model), "proof": model_proof,
                                "bias": bias, "multiplier": multiplier, "quantiles": {}}
            for version in VERSIONS:
                vm, vv = mu.copy(), variance.copy()
                if version == "scaled":
                    vm += bias; vv *= multiplier
                vv = np.maximum(vv, cfg["variance_floor"])
                if frozen:
                    qr, qv = base["quantiles"][version]
                else:
                    qr = corrected_quantile(np.abs(y[interval, 0]-vm[interval])/np.sqrt(vv[interval]), cfg["nominal_coverage"])
                    qv = corrected_quantile(np.abs(.5*np.log(actual_var[interval]/vv[interval])), cfg["nominal_coverage"])
                    cache[group]["quantiles"][version] = (qr, qv)
                mid = group+"_"+version
                record = {"fold": fold, "model_id": mid, "regime": support["regime"],
                    **score_forecast(y[scoring], vm[scoring], vv[scoring], qr, qv, fit_mean, tv["96"].to_numpy()[scoring]),
                    "calibration": {"return_bias": bias if version == "scaled" else 0.,
                        "variance_scale": multiplier if version == "scaled" else 1.,
                        "return_quantile": qr, "volatility_quantile": qv}, "provenance": model_proof}
                # Persist only validation forecasts; no future-label mask enters orders.
                actual = y[ix].copy(); actual[~scoring[ix]] = np.nan
                cmu, cvar = vm[ix].copy(), vv[ix].copy()
                cmu[~inference[ix]], cvar[~inference[ix]] = np.nan, np.nan
                pred_proof = save_npz("forecasts", mid, timestamps=window.index.asi8,
                    actual=actual, mu=cmu, variance=cvar, raw_log_variance=logvar[ix],
                    persistence96_variance=tv["96"].to_numpy()[ix], fit_return_mean=fit_mean,
                    score_support=scoring[ix], inference_mask=inference[ix])
                record["forecast_sha256"] = pred_proof["sha256"]
                fold_scores.append(record)
                evaluate(mid+"_point", map_outcomes(np.column_stack([cmu, np.zeros(len(window)), np.sqrt(cvar)]), "return"))
                targets, diagnostic = conditional_targets(window, cmu, cvar, execution,
                    risk_aversion=1, cost_multiplier=cfg["utility_cost_multiplier"])
                trace_path = output/"traces"/f"fold{fold}_{mid}_risk1.json"
                write_json(trace_path, diagnostic); artifacts[str(trace_path)] = file_digest(trace_path)
                compact = {k: v for k, v in diagnostic.items() if k != "decision_trace"}
                compact["trace_sha256"] = artifacts[str(trace_path)]
                evaluate(mid+"_utility_risk1", targets, compact)
        saved = {"registration_sha256": reg_sha, "rows": fold_rows, "scores": fold_scores,
                 "artifact_sha256": artifacts}
        immutable_json(fold_path, saved)
        rows.extend(fold_rows); scores.extend(fold_scores)
        print(json.dumps({"fold": fold, "forecasts": len(fold_scores), "policies": len(fold_rows),
                          "counts": support["counts"], "regime": support["regime"]["trend"]}), flush=True)
    result = {"registration_sha256": reg_sha, "rows": rows, "scores": scores,
              "forecast_summary": forecast_summaries(scores), "summary": decision_summaries(rows, 3),
              "paired": paired_results(scores, rows), "selection_performed": False,
              "high_probability_generalization_established": False, "scope": registration["scope"]}
    immutable_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        prepared = prepare(args.config)
        immutable_json(Path(prepared[0]["preflight_path"]), prepared[-1])
        print(json.dumps({"path": prepared[0]["preflight_path"], "folds": prepared[-1]["folds"]}), flush=True)
    else:
        run(args.config)
