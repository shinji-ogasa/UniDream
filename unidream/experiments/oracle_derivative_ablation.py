"""Fixed matched-support Spot/perpetual predictive and allocation ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alpha_dd_search import (digest, file_digest, fold_spec, load_bars, metrics,
                              validate_data_artifact, write_json)
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_features import make_derivative_groups
from .oracle_frontier import fit_mask, map_outcomes, outcome_frame, quarter_regime, summarize
from .oracle_frontier_features import make_feature_groups
from .oracle_paired_loss_bootstrap import paired_quarter_block_bootstrap
from .oracle_risk_calibration import (corrected_quantile, forecast_metrics,
                                      scale_and_bias, trailing_variances)
from .robust_overlay import build_targets


def mask_digest(index, mask):
    return hashlib.sha256(index.asi8.astype("<i8").tobytes()
                          + np.asarray(mask, "u1").tobytes()).hexdigest()


def validate_um(path, cutoff, symbol):
    sidecar_path = path.with_suffix(".sha256.json")
    sidecar = json.loads(sidecar_path.read_text())
    if (sidecar["schema"] != "oracle-derivative-raw-v1" or sidecar["symbol"] != symbol
            or pd.Timestamp(sidecar["feature_decision_cutoff_exclusive"]) != pd.Timestamp(cutoff)):
        raise ValueError("UM artifact contract mismatch")
    for p, key in [(path, "data_sha256"), (Path(sidecar["availability_path"]), "availability_sha256"),
                   (Path(sidecar["source_ledger_path"]), "source_ledger_sha256"),
                   (Path(sidecar["registration_path"]), "registration_sha256")]:
        if file_digest(p) != sidecar[key]:
            raise ValueError(f"UM artifact binding changed: {key}")
    raw = pd.read_parquet(path, filters=[("bar_open_ts", "<", pd.Timestamp(cutoff)-pd.Timedelta(minutes=15))])
    availability = pd.read_parquet(sidecar["availability_path"]).reindex(raw.index)
    observed = np.isfinite(raw[sidecar["raw_fields"]].to_numpy(float)).all(axis=1)
    if not np.array_equal(observed, availability["um_bar_observed"].to_numpy()):
        raise ValueError("UM observed mask mismatch")
    if (len(raw) and not (raw.decision_ts < pd.Timestamp(cutoff)).all()):
        raise ValueError("post-cutoff UM feature rows")
    return raw, {"sidecar_sha256": file_digest(sidecar_path), **sidecar}


def score_forecast(y, mu, variance, qr, qv, fit_mean, persistence):
    score = forecast_metrics(y, mu, variance, qr, qv)
    zero_mse = float(np.mean(y[:, 0]**2))
    mean_mse = float(np.mean((y[:, 0]-fit_mean)**2))
    ref = forecast_metrics(y, np.zeros(len(y)), persistence, qr, qv)
    score.update({"zero_return_mse": zero_mse, "fit_mean_return_mse": mean_mse,
                  "return_skill_vs_zero": 1-score["return_mse"]/zero_mse if zero_mse > 0 else None,
                  "return_skill_vs_fit_mean": 1-score["return_mse"]/mean_mse if mean_mse > 0 else None,
                  "return_sign_accuracy": float(np.mean((mu > 0) == (y[:, 0] > 0))),
                  "return_rank_ic": float(spearmanr(mu, y[:, 0]).statistic)
                  if np.ptp(mu) > 0 and np.ptp(y[:, 0]) > 0 else None,
                  "persistence96": {k: ref[k] for k in ("qlike", "variance_mse", "rms_mse")}})
    return score


def paired_summaries(scores, rows):
    """Equal-quarter paired differences; no model selection."""
    out = {}
    for candidate, reference in [("perp_flow", "technical"), ("derivative", "technical"),
                                 ("derivative", "perp_flow")]:
        for version in ("raw", "scaled"):
            cmid, rmid = f"{candidate}_{version}", f"{reference}_{version}"
            cs = {r["fold"]: r for r in scores if r["model_id"] == cmid}
            rs = {r["fold"]: r for r in scores if r["model_id"] == rmid}
            if cs.keys() != rs.keys():
                raise ValueError("unpaired forecast folds")
            entry = {"candidate": cmid, "reference": rmid, "folds": len(cs),
                     "difference_convention": "candidate minus reference"}
            entry["forecast"] = {}
            for metric in ("return_mse", "qlike", "variance_mse", "rms_mse"):
                differences = [cs[f][metric]-rs[f][metric] for f in cs]
                ref_mean = float(np.mean([rs[f][metric] for f in cs]))
                entry["forecast"][metric] = {"mean_difference": float(np.mean(differences)),
                    "relative_loss_reduction": -float(np.mean(differences))/ref_mean if ref_mean > 0 else None,
                    "improved_quarters": int(np.sum(np.asarray(differences) < 0))}
            entry["policies"] = {}
            for policy in ("point", "utility_risk0", "utility_risk1"):
                cr = {r["fold"]: r for r in rows if r["candidate_id"] == cmid+"_"+policy}
                rr = {r["fold"]: r for r in rows if r["candidate_id"] == rmid+"_"+policy}
                if cr.keys() != cs.keys() or rr.keys() != cs.keys():
                    raise ValueError("unpaired policy folds")
                entry["policies"][policy] = {cost: {metric: float(np.mean([
                    cr[f][cost][metric]-rr[f][cost][metric] for f in cr]))
                    for metric in ("alpha_ex", "maxdd_delta", "turnover", "trades")}
                    for cost in ("base", "stress_2x")}
            out[cmid+"_vs_"+rmid] = entry
    return out


def forecast_summaries(scores):
    result = {}
    keys = ("return_mse", "qlike", "variance_mse", "rms_mse", "zero_return_mse",
            "fit_mean_return_mse", "return_coverage", "volatility_coverage",
            "mean_return_interval_width", "return_sign_accuracy")
    for mid in sorted({s["model_id"] for s in scores}):
        result[mid] = {}
        for regime in ("all", "bull", "bear", "sideways"):
            selected = [s for s in scores if s["model_id"] == mid and
                        (regime == "all" or s["regime"]["trend"] == regime)]
            count = sum(s["rows"] for s in selected)
            means = {k: {"equal_quarter_mean": float(np.mean([s[k] for s in selected])),
                         "pooled_row_mean": float(sum(s[k]*s["rows"] for s in selected)/count)} for k in keys}
            means["persistence96"] = {k: {"equal_quarter_mean": float(np.mean([s["persistence96"][k] for s in selected])),
                "pooled_row_mean": float(sum(s["persistence96"][k]*s["rows"] for s in selected)/count)}
                for k in ("qlike", "variance_mse", "rms_mse")}
            result[mid][regime] = {"quarters": len(selected), "rows": count, "metrics": means,
                "mean_return_rank_ic": float(np.mean([s["return_rank_ic"] for s in selected]))
                if all(s["return_rank_ic"] is not None for s in selected) else None}
    return result


def loss_uncertainty(output, scores, cfg):
    """Construct full six-hour loss grids without compressing missing slots."""
    fold_losses, timestamps = {}, {}
    for score in scores:
        fold, mid = str(score["fold"]), score["model_id"]
        path = output/"forecasts"/f"fold{fold}_{mid}.npz"
        if file_digest(path) != score["forecast_sha256"]:
            raise ValueError("forecast changed before paired loss comparison")
        saved = np.load(path)
        index = pd.to_datetime(saved["timestamps"], utc=True)
        clock = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
        scheduled = index[clock].asi8
        if len(scheduled) < 2 or not np.all(np.diff(scheduled) == pd.Timedelta(hours=6).value):
            raise ValueError("bootstrap requires the complete scheduled six-hour grid")
        if fold in timestamps and not np.array_equal(timestamps[fold], scheduled):
            raise ValueError("unpaired bootstrap calendar")
        timestamps[fold] = scheduled
        actual = saved["actual"][clock]
        mu, variance = saved["mu"][clock], saved["variance"][clock]
        variance = np.maximum(variance, cfg["variance_floor"])
        actual_variance = np.maximum(actual[:, 2]**2, cfg["variance_floor"])
        ratio = actual_variance/variance
        losses = {"return_mse": (actual[:, 0]-mu)**2, "qlike": ratio-np.log(ratio)-1}
        for metric, values in losses.items():
            values[~saved["score_support"][clock]] = np.nan
            fold_losses.setdefault(fold, {})[mid+"_"+metric] = values
    comparisons = {}
    for candidate, reference in [("perp_flow", "technical"), ("derivative", "technical"),
                                 ("derivative", "perp_flow")]:
        for version in cfg["versions"]:
            for metric in ("return_mse", "qlike"):
                c, r = f"{candidate}_{version}_{metric}", f"{reference}_{version}_{metric}"
                comparisons[c+"_vs_"+r] = (c, r)
    bc = cfg["bootstrap"]
    return paired_quarter_block_bootstrap(fold_losses, comparisons=comparisons,
        block_lengths=bc["block_slots"], primary_block_length=bc["primary_block_slots"],
        n_bootstrap=bc["replicates"], seed=bc["seed"], confidence=.95)


def run(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    if cfg["horizon_bars"] != 24:
        raise ValueError("this registered decision family requires horizon 24")
    source_config = Path(cfg["source_config"])
    fc = yaml.safe_load(source_config.read_text())
    output = Path(cfg["output_dir"])
    if (output/"results.json").exists():
        raise ValueError("immutable experiment already completed")
    preflight_path = Path(cfg["preflight_path"])
    if file_digest(preflight_path) != cfg["preflight_sha256"]:
        raise ValueError("preflight changed")
    preflight = json.loads(preflight_path.read_text())
    data_proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    um, um_proof = validate_um(Path(cfg["um_path"]), fc["data_cutoff"], fc["symbol"])
    if (data_proof["artifact_sha256"] != preflight["source_sha256"][fc["data_path"]]
            or um_proof["data_sha256"] != preflight["source_sha256"][str(Path(cfg["um_path"]).resolve())]):
        raise ValueError("preflight/data binding mismatch")
    sources = [Path(__file__), Path(make_derivative_groups.__code__.co_filename),
               Path(outcome_frame.__code__.co_filename), Path(make_feature_groups.__code__.co_filename),
               Path(metrics.__code__.co_filename), Path(conditional_targets.__code__.co_filename),
               Path(trailing_variances.__code__.co_filename), Path(build_targets.__code__.co_filename),
               Path(paired_quarter_block_bootstrap.__code__.co_filename),
               Path(__file__).with_name("alpha_dd_features.py")]
    registration = {"config": cfg, "config_sha256": file_digest(config_path),
        "source_config": fc, "source_config_sha256": file_digest(source_config),
        "source_sha256": {p.name: file_digest(p) for p in sources},
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "spot_data_proof": data_proof, "um_data_proof": um_proof,
        "preflight_sha256": file_digest(preflight_path),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "scope": "adaptive matched-support reused validation; not formal P1 or prospective evidence"}
    reg_path = output/"registration.json"
    if reg_path.exists() and json.loads(reg_path.read_text()) != registration:
        raise ValueError("immutable registration changed")
    write_json(reg_path, registration)
    reg_sha = digest(registration)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    groups = make_derivative_groups(bars, um)
    original = make_feature_groups(bars)
    tv = trailing_variances(bars, cfg["horizon_bars"])
    common = np.isfinite(tv.to_numpy()).all(axis=1)
    common &= np.isfinite(original["flow"].to_numpy()).all(axis=1)
    for name in cfg["feature_groups"]:
        common &= np.isfinite(groups[name].to_numpy()).all(axis=1)
    schedule = np.asarray((bars.index.hour % 6 == 0) & (bars.index.minute == 0))
    if mask_digest(bars.index, common & schedule) != preflight["shared_mask"]["shared_6h_mask_sha256"]:
        raise ValueError("shared inference mask differs from data-only preflight")
    h = cfg["horizon_bars"]
    y = outcome_frame(bars, h).to_numpy()
    y_valid = np.isfinite(y).all(axis=1)
    actual_var = np.maximum(y[:, 2]**2, cfg["variance_floor"])
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": 2*execution["one_way_cost"],
              "borrow_annual": 2*execution["borrow_annual"]}
    rows, scores = [], []
    preflight_folds = {f["fold"]: f for f in preflight["folds"]}
    for fold_id in cfg["development_folds"]:
        fold_path = output/f"fold_{fold_id}.json"
        if fold_path.exists():
            saved = json.loads(fold_path.read_text())
            if saved["registration_sha256"] != reg_sha:
                raise ValueError("fold source binding changed")
            rows.extend(saved["rows"]); scores.extend(saved["scores"])
            continue
        spec = fold_spec(fold_id, fc["fold_anchor"])
        vs, ve = spec["val_start"], spec["val_end"]
        if ve > pd.Timestamp(fc["data_cutoff"]):
            raise ValueError("validation exceeds cutoff")
        scale_start = vs-pd.DateOffset(months=cfg["scale_months"]+cfg["interval_months"])
        interval_start = vs-pd.DateOffset(months=cfg["interval_months"])
        train_start = scale_start-pd.DateOffset(months=cfg["fit_months"])
        def segment(start, end):
            return fit_mask(bars.index, common & y_valid, start=start, end=end,
                            horizon=h, cadence_hours=6)
        train, scale, interval = segment(train_start, scale_start), segment(scale_start, interval_start), segment(interval_start, vs)
        counts = [int(m.sum()) for m in (train, scale, interval)]
        if counts != preflight_folds[fold_id]["split_18_3_3"][str(h)]["fit_scale_interval_rows"]:
            raise ValueError("preflight purged segment counts changed")
        if any(c < cfg["minimum_"+key+"_rows"] for c, key in zip(counts, ("fit", "scale", "interval"))):
            raise ValueError("insufficient fit/calibration support")
        ix = np.asarray((bars.index >= vs) & (bars.index < ve))
        window = bars.loc[ix]
        inference = ix & common & schedule
        if mask_digest(bars.index, inference) != preflight_folds[fold_id]["shared_inference_mask_sha256"]:
            raise ValueError("preflight fold inference mask changed")
        if (not window.index.equals(pd.date_range(vs, ve, freq="15min", inclusive="left"))
                or window.bar_available.mean() < fc["minimum_bar_coverage"]):
            raise ValueError("incomplete validation window")
        score_mask = inference & y_valid & np.asarray(bars.index+pd.Timedelta(minutes=15*(h+1)) <= ve)
        if score_mask.sum() < 16:
            raise ValueError("insufficient validation forecast scoring support")
        predict = common & schedule & np.asarray((bars.index >= scale_start) & (bars.index < ve))
        cal_ix = np.asarray((bars.index >= scale_start) & (bars.index < vs))
        regime = quarter_regime(groups["base16"], ix, fc["regime"]["normalized_momentum_90_threshold"])
        fit_mean = float(y[train, 0].mean())
        provenance = {"train_start": str(train_start), "train_end": str(scale_start),
            "scale_end": str(interval_start), "interval_end": str(vs),
            "fit_rows": counts[0], "scale_rows": counts[1], "interval_rows": counts[2],
            "inference_rows": int(inference.sum()), "score_rows": int(score_mask.sum()),
            "fit_return_mean": fit_mean,
            "last_label_end": {name: str(bars.index[m][-1]+pd.Timedelta(minutes=15*(h+1)))
                               for name, m in [("fit", train), ("scale", scale), ("interval", interval)]}}
        fold_rows, fold_scores = [], []
        def evaluate(cid, targets, diagnostic=None):
            targets = np.asarray(targets, float).copy()
            targets[~inference[ix]] = np.nan
            path = output/"targets"/f"fold{fold_id}_{cid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, targets=targets, timestamps=window.index.asi8)
            fold_rows.append({"fold": fold_id, "candidate_id": cid, "regime": regime,
                "base": metrics(window, targets, execution), "stress_2x": metrics(window, targets, stress),
                "targets_sha256": file_digest(path), "diagnostic": diagnostic,
                "metadata": {**provenance, "validation_start": str(vs), "validation_end_exclusive": str(ve)}})
        evaluate("bh", np.full(len(window), np.nan))
        evaluate("common_robust", build_targets(groups["base16"])[ix])
        for group in cfg["feature_groups"]:
            x = groups[group].to_numpy(float)
            return_model = make_pipeline(StandardScaler(), Ridge(alpha=cfg["return_ridge_alpha"]))
            risk_model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=64,
                learning_rate=.04, l2_regularization=10., early_stopping=False, random_state=cfg["seed"])
            mu, logvar = np.full(len(bars), np.nan), np.full(len(bars), np.nan)
            with threadpool_limits(limits=2):
                return_model.fit(x[train], y[train, 0])
                mu[predict] = return_model.predict(x[predict])
                risk_model.fit(x[train], np.log(actual_var[train]))
                logvar[predict] = risk_model.predict(x[predict])
            if not np.isfinite(mu[predict]).all() or not np.isfinite(logvar[predict]).all():
                raise ValueError("nonfinite fitted forecast")
            model_hashes = {}
            for label, model in [("return", return_model), ("variance", risk_model)]:
                p = output/"models"/f"fold{fold_id}_{group}_{label}.joblib"
                p.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, p); model_hashes[label+"_model_sha256"] = file_digest(p)
            variance = np.exp(np.clip(logvar, np.log(cfg["variance_floor"]), 0))
            bias, multiplier = scale_and_bias(y[scale, 0], actual_var[scale], mu[scale], variance[scale])
            cal_path = output/"calibration"/f"fold{fold_id}_{group}.npz"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            cal_actual = y[cal_ix].copy()
            cal_actual[~(scale | interval)[cal_ix]] = np.nan
            np.savez_compressed(cal_path, timestamps=bars.index[cal_ix].asi8, mu=mu[cal_ix],
                log_variance=logvar[cal_ix], variance=variance[cal_ix], actual=cal_actual,
                scale_mask=scale[cal_ix], interval_mask=interval[cal_ix])
            model_proof = {**provenance, **model_hashes, "group": group, "features": list(groups[group].columns),
                "calibration_sha256": file_digest(cal_path),
                "raw_log_variance_clip_count": int(np.sum((logvar[predict] < np.log(cfg["variance_floor"])) | (logvar[predict] > 0)))}
            for version in cfg["versions"]:
                vm, vv = mu.copy(), variance.copy()
                if version == "scaled":
                    vm += bias; vv *= multiplier
                vv = np.maximum(vv, cfg["variance_floor"])
                qr = corrected_quantile(np.abs(y[interval, 0]-vm[interval])/np.sqrt(vv[interval]), cfg["nominal_coverage"])
                qv = corrected_quantile(np.abs(.5*np.log(actual_var[interval]/vv[interval])), cfg["nominal_coverage"])
                mid = group+"_"+version
                record = {"fold": fold_id, "model_id": mid, "regime": regime,
                    **score_forecast(y[score_mask], vm[score_mask], vv[score_mask], qr, qv,
                                     fit_mean, tv["96"].to_numpy()[score_mask]),
                    "calibration": {"return_bias": bias if version == "scaled" else 0.,
                        "variance_scale": multiplier if version == "scaled" else 1.,
                        "return_quantile": qr, "volatility_quantile": qv}, "provenance": model_proof}
                pred_path = output/"forecasts"/f"fold{fold_id}_{mid}.npz"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                actual = y[ix].copy(); actual[~score_mask[ix]] = np.nan
                np.savez_compressed(pred_path, timestamps=window.index.asi8, actual=actual,
                    mu=vm[ix], variance=vv[ix], raw_log_variance=logvar[ix],
                    persistence96_variance=tv["96"].to_numpy()[ix], fit_return_mean=fit_mean,
                    score_support=score_mask[ix], inference_mask=inference[ix])
                record["forecast_sha256"] = file_digest(pred_path)
                fold_scores.append(record)
                point = map_outcomes(np.column_stack([vm[ix], np.zeros(len(window)), np.sqrt(vv[ix])]), "return")
                evaluate(mid+"_point", point)
                cmu, cvar = vm[ix].copy(), vv[ix].copy()
                cmu[~inference[ix]], cvar[~inference[ix]] = np.nan, np.nan
                for risk in cfg["utility_risk_aversions"]:
                    targets, diagnostic = conditional_targets(window, cmu, cvar, execution,
                        risk_aversion=risk, cost_multiplier=cfg["utility_cost_multiplier"])
                    # Full decision traces remain in separate immutable artifacts.
                    trace_path = output/"traces"/f"fold{fold_id}_{mid}_risk{risk}.json"
                    write_json(trace_path, diagnostic)
                    compact = {k: v for k, v in diagnostic.items() if k != "decision_trace"}
                    compact["trace_sha256"] = file_digest(trace_path)
                    evaluate(mid+f"_utility_risk{risk}", targets, compact)
        saved = {"registration_sha256": reg_sha, "rows": fold_rows, "scores": fold_scores}
        write_json(fold_path, saved)
        rows.extend(fold_rows); scores.extend(fold_scores)
        print(json.dumps({"fold": fold_id, "forecasts": len(fold_scores), "policies": len(fold_rows),
                          "counts": counts, "regime": regime["trend"]}), flush=True)
    result = {"registration_sha256": reg_sha, "rows": rows, "scores": scores,
              "forecast_summary": forecast_summaries(scores),
              "summary": summarize(rows, 3), "paired": paired_summaries(scores, rows),
              "paired_loss_uncertainty": loss_uncertainty(output, scores, cfg),
              "scope": registration["scope"], "selection_performed": False,
              "high_probability_generalization_established": False}
    write_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
