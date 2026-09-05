"""Disjoint fit/scale/interval/validation calibration and decision diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alpha_dd_search import digest, file_digest, fold_spec, load_bars, metrics, validate_data_artifact, write_json
from .oracle_frontier import fit_mask, map_outcomes, outcome_frame, quarter_regime, summarize
from .oracle_frontier_features import make_feature_groups
from .robust_overlay import build_targets


def trailing_variances(bars, horizon=24):
    square = np.log(bars.close / bars.close.shift(1)).pow(2)
    frame = pd.DataFrame(index=bars.index)
    for window in (24, 96, 672, 2880):
        series = square.rolling(window, min_periods=math.ceil(.995*window)).mean() * horizon
        series = series.where(np.arange(len(bars)) >= window)
        frame[str(window)] = series.shift(1)
    return frame


def corrected_quantile(scores, coverage):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite nonempty scores required")
    rank = math.ceil((len(scores)+1)*coverage)
    if rank > len(scores):
        raise ValueError("insufficient scores for finite corrected quantile")
    return float(np.sort(scores)[rank-1])


def scale_and_bias(y_return, actual_variance, mu, variance):
    arrays = [np.asarray(x, float) for x in (y_return, actual_variance, mu, variance)]
    if len({x.shape for x in arrays}) != 1 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("aligned finite calibration arrays required")
    if not len(arrays[0]) or (arrays[1] < 0).any() or (arrays[3] <= 0).any():
        raise ValueError("invalid variance calibration")
    return float(np.mean(arrays[0]-arrays[2])), float(np.mean(arrays[1]/arrays[3]))


def interval_targets(mu, variance, return_quantile, contract):
    sigma = np.sqrt(variance)
    lower, upper = mu-return_quantile*sigma, mu+return_quantile*sigma
    threshold = 2*contract["one_way_cost"] + contract["borrow_annual"]*24/35040
    target = np.where(lower > threshold, 1.12, np.where(upper < -2*contract["one_way_cost"], .5, 1.))
    target[~(np.isfinite(mu) & np.isfinite(variance))] = np.nan
    return target


def forecast_metrics(y, mu, variance, qr, qv):
    actual_variance = np.maximum(y[:, 2]**2, 1e-12)
    variance = np.maximum(variance, 1e-12)
    sigma = np.sqrt(variance)
    ratio = actual_variance/variance
    ret_score = np.abs(y[:, 0]-mu)/sigma
    vol_score = np.abs(np.log(np.sqrt(actual_variance)/sigma))
    return {"rows": len(y), "variance_mse": float(np.mean((actual_variance-variance)**2)),
            "rms_mse": float(np.mean((np.sqrt(actual_variance)-sigma)**2)),
            "qlike": float(np.mean(ratio-np.log(ratio)-1)),
            "return_mse": float(np.mean((y[:, 0]-mu)**2)),
            "return_coverage": float(np.mean(ret_score <= qr)),
            "mean_return_interval_width": float(np.mean(2*qr*sigma)),
            "return_lower_tail_miss": float(np.mean(y[:, 0] < mu-qr*sigma)),
            "volatility_coverage": float(np.mean(vol_score <= qv)),
            "volatility_interval_log_width": float(2*qv)}


def run(config_path):
    config = yaml.safe_load(config_path.read_text())
    source_config = Path(config["source_config"])
    fc = yaml.safe_load(source_config.read_text())
    output = Path(config["output_dir"])
    if (output/"results.json").exists():
        raise ValueError("immutable results already exist")
    data = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    source_files = [Path(__file__), Path(outcome_frame.__code__.co_filename),
                    Path(make_feature_groups.__code__.co_filename), Path(metrics.__code__.co_filename),
                    Path(build_targets.__code__.co_filename), Path(__file__).with_name("alpha_dd_features.py")]
    registration = {"config": config, "source_config": fc, "data_proof": data,
                    "source_sha256": {x.name:file_digest(x) for x in source_files},
                    "config_sha256":file_digest(config_path), "source_config_sha256":file_digest(source_config),
                    "scope":"adaptive reused validation; no later test selection; no exact conditional guarantee"}
    reg_path=output/"registration.json"
    if reg_path.exists() and json.loads(reg_path.read_text()) != registration:
        raise ValueError("registration changed")
    write_json(reg_path, registration)
    reg_hash=digest(registration)
    bars=load_bars(Path(fc["data_path"]),cutoff=fc["data_cutoff"])
    groups=make_feature_groups(bars)
    tv=trailing_variances(bars)
    common=np.isfinite(tv.to_numpy()).all(axis=1)
    for g in ("base16","technical"):
        common &= np.isfinite(groups[g].to_numpy()).all(axis=1)
    schedule=(bars.index.hour%6==0)&(bars.index.minute==0)
    horizon=config["horizon_bars"]
    y=outcome_frame(bars,horizon).to_numpy()
    y_valid=np.isfinite(y).all(axis=1)
    var=np.maximum(y[:,2]**2,config["variance_floor"])
    execution=fc["execution"]
    stress={**execution,"one_way_cost":execution["one_way_cost"]*2,"borrow_annual":execution["borrow_annual"]*2}
    all_rows,all_scores=[],[]
    for fold_id in fc["development_folds"]:
        fold_path=output/f"fold_{fold_id}.json"
        if fold_path.exists():
            saved=json.loads(fold_path.read_text())
            if saved["registration_sha256"]!=reg_hash:raise ValueError("fold binding changed")
            all_rows.extend(saved["rows"]);all_scores.extend(saved["scores"]);continue
        fold=fold_spec(fold_id,fc["fold_anchor"])
        vs,ve=fold["val_start"],fold["val_end"]
        scale_start=vs-pd.DateOffset(months=config["interval_months"]+config["scale_months"])
        interval_start=vs-pd.DateOffset(months=config["interval_months"])
        train_start=scale_start-pd.DateOffset(months=config["fit_months"])
        def segment(start,end):
            return fit_mask(bars.index,common&y_valid,start=start,end=end,horizon=horizon,cadence_hours=6)
        train=segment(train_start,scale_start);scale=segment(scale_start,interval_start);interval=segment(interval_start,vs)
        for name,mask,minimum in [("fit",train,config["minimum_fit_rows"]),("scale",scale,config["minimum_scale_rows"]),("interval",interval,config["minimum_interval_rows"])]:
            if mask.sum()<minimum:raise ValueError(f"fold{fold_id} {name} insufficient:{mask.sum()}")
        ix=np.asarray((bars.index>=vs)&(bars.index<ve));window=bars.loc[ix]
        if not window.index.equals(pd.date_range(vs,ve,freq="15min",inclusive="left")) or window.bar_available.mean()<fc["minimum_bar_coverage"]:
            raise ValueError("incomplete validation")
        inference=np.asarray(ix&common&schedule)
        scores_mask=inference&y_valid&np.asarray(bars.index+pd.Timedelta(minutes=15*(horizon+1))<=ve)
        prediction_mask=np.asarray(common&schedule&(bars.index>=scale_start)&(bars.index<ve))
        regime=quarter_regime(groups["base16"],ix,fc["regime"]["normalized_momentum_90_threshold"])
        provenance={"train_start":str(train_start),"train_end":str(scale_start),"scale_end":str(interval_start),"interval_end":str(vs),
                    "fit_rows":int(train.sum()),"scale_rows":int(scale.sum()),"interval_rows":int(interval.sum()),
                    "last_label_end":{n:str(bars.index[m][-1]+pd.Timedelta(minutes=15*(horizon+1))) for n,m in [("fit",train),("scale",scale),("interval",interval)]}}
        x=groups["technical"].to_numpy(float)
        ret_model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
        mu=np.full(len(bars),np.nan)
        with threadpool_limits(limits=2):
            ret_model.fit(x[train],y[train,0]);mu[prediction_mask]=ret_model.predict(x[prediction_mask])
        if not np.isfinite(mu[prediction_mask]).all():
            raise ValueError("nonfinite shared return forecast")
        model_root=output/"models";model_root.mkdir(parents=True,exist_ok=True)
        return_model_path=model_root/f"fold{fold_id}_return.joblib"
        joblib.dump(ret_model,return_model_path)
        provenance["return_model_sha256"]=file_digest(return_model_path)
        rows,scores=[],[]
        def evaluate(cid,targets):
            targets=np.asarray(targets,float).copy();targets[~inference]=np.nan
            path=output/"targets"/f"fold{fold_id}_{cid}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,targets=targets[ix],timestamps=window.index.asi8)
            rows.append({"fold":fold_id,"candidate_id":cid,"regime":regime,"base":metrics(window,targets[ix],execution),
                         "stress_2x":metrics(window,targets[ix],stress),"targets_sha256":file_digest(path),"metadata":provenance})
        evaluate("bh",np.full(len(bars),np.nan));evaluate("common_robust",build_targets(groups["base16"]))
        for family in config["models"]:
            prediction_var=np.full(len(bars),np.nan)
            variance_model_sha=None
            if family.startswith("persistence"):
                prediction_var[prediction_mask]=tv[family.removeprefix("persistence")].to_numpy()[prediction_mask]
            else:
                if family=="har_ridge":
                    fx=np.log(np.maximum(tv.to_numpy(),config["variance_floor"]))
                    model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                else:
                    fx=groups[family.removesuffix("_hgb")].to_numpy(float)
                    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=64,l2_regularization=10.,
                        learning_rate=.04,early_stopping=False,random_state=config["seed"])
                with threadpool_limits(limits=2):
                    model.fit(fx[train],np.log(var[train]));log_prediction=model.predict(fx[prediction_mask])
                prediction_var[prediction_mask]=np.exp(np.clip(log_prediction,np.log(config["variance_floor"]),0))
                variance_model_path=model_root/f"fold{fold_id}_{family}.joblib"
                joblib.dump(model,variance_model_path)
                variance_model_sha=file_digest(variance_model_path)
            prediction_var[prediction_mask]=np.clip(prediction_var[prediction_mask],config["variance_floor"],1.)
            if not np.isfinite(prediction_var[prediction_mask]).all():
                raise ValueError("nonfinite risk forecast")
            bias,multiplier=scale_and_bias(y[scale,0],var[scale],mu[scale],prediction_var[scale])
            for version in config["versions"]:
                vm=mu.copy();vv=prediction_var.copy()
                if version=="scaled":vm+=bias;vv*=multiplier
                vv=np.maximum(vv,config["variance_floor"])
                qr=corrected_quantile(np.abs(y[interval,0]-vm[interval])/np.sqrt(vv[interval]),config["nominal_coverage"])
                qv=corrected_quantile(np.abs(.5*np.log(var[interval]/vv[interval])),config["nominal_coverage"])
                mid=f"{family}_{version}"
                scores.append({"fold":fold_id,"model_id":mid,"regime":regime,
                    **forecast_metrics(y[scores_mask],vm[scores_mask],vv[scores_mask],qr,qv),
                    "calibration":{"return_bias":bias if version=="scaled" else 0.,"variance_scale":multiplier if version=="scaled" else 1.,
                                   "return_quantile":qr,"volatility_quantile":qv},
                    "provenance":{**provenance,"variance_model_sha256":variance_model_sha,
                                  "variance_model_fitted":variance_model_sha is not None}})
                path=output/"forecasts"/f"fold{fold_id}_{mid}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                actual=y[ix].copy();actual[~scores_mask[ix]]=np.nan
                np.savez_compressed(path,mu=vm[ix],variance=vv[ix],actual=actual,score_support=scores_mask[ix],inference_mask=inference[ix],timestamps=window.index.asi8)
                scores[-1]["forecast_sha256"]=file_digest(path)
                evaluate(mid+"_point",map_outcomes(np.column_stack([vm,np.zeros(len(bars)),np.sqrt(vv)]),"return"))
                evaluate(mid+"_interval_gate",interval_targets(vm,vv,qr,execution))
        saved={"registration_sha256":reg_hash,"rows":rows,"scores":scores};write_json(fold_path,saved)
        all_rows.extend(rows);all_scores.extend(scores)
        print(json.dumps({"fold":fold_id,"models":len(scores),"policies":len(rows),"provenance":provenance}),flush=True)
    result={"registration_sha256":reg_hash,"rows":all_rows,"scores":all_scores,
            "summary":summarize(all_rows,3),"scope":registration["scope"],"high_probability_generalization_established":False}
    write_json(output/"results.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    run(parser.parse_args().config)
