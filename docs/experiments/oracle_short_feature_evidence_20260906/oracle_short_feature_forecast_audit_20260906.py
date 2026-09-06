"""Independent Stage15 audit; real-data mode requires root's postfreeze go.

Reuses only frozen OLD feature helpers for technical29 and ancestor support.
The eight new columns, six masks, T-only standardization/normal equations and
scalar application are computed here, without short-feature/short-fit imports.
No evaluation losses, weight calibration, risk fits or policy rollouts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import yaml

GROUPS = ("technical", "technical_short_price", "technical_short_flow", "technical_short_both")
PRICE = ("spot_log_return4", "spot_log_return16", "spot_log_return48", "spot_body_sign1", "spot_close_location1")
FLOW = ("spot_weighted_flow4", "perp_weighted_flow4", "spot_quote_activity24_672")
SPOT = Path("/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/checkpoints/alpha_dd_data/spot_15m.parquet")
UM = Path("checkpoints/oracle_derivative_data/um_15m.parquet")
CUTOFF = pd.Timestamp("2023-04-16T13:45:00Z")
TOLERANCES = {"scaler_atol": 1e-10, "scaler_rtol": 1e-12,
    "coefficient_atol": 1e-12, "coefficient_rtol": 1e-10,
    "normal_equation_prediction_atol": 1e-12, "saved_scalar_prediction_atol": 1e-14,
    "registered_baseline_parity_atol": 1e-14, "registered_baseline_parity_rtol": 1e-12}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
def arrays(path):
    with np.load(path, allow_pickle=False) as stream: return {k: stream[k].copy() for k in stream.files}
def mask_digest(index, mask): return hashlib.sha256(index.asi8.astype("<i8").tobytes()+np.asarray(mask,"u1").tobytes()).hexdigest()
def position_digest(mask): return hashlib.sha256(np.asarray([len(mask)],dtype="<i8").tobytes()+np.asarray(mask,"u1").tobytes()).hexdigest()
def matrix_digest(array):
    a = np.asarray(array,dtype="<f8",order="C")
    return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype="<i8").tobytes()+a.tobytes(order="C")).hexdigest()
def index_digest(index):
    header=json.dumps({"type":type(index).__name__,"dtype":str(index.dtype),"length":len(index)},sort_keys=True).encode()
    return hashlib.sha256(header+b"\n"+pd.util.hash_pandas_object(index,index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()


def equal(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    same = np.array_equal(a,b,equal_nan=True) if a.dtype.kind in "fc" or b.dtype.kind in "fc" else np.array_equal(a,b)
    assert same, "exact mismatch: "+label


def near(a,b,label,maxima,atol=1e-12,rtol=0.):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape,label+" shape"
    equal(np.isfinite(a),np.isfinite(b),label+" finite mask")
    selected=np.isfinite(a)
    delta=float(np.max(np.abs(a[selected]-b[selected]))) if selected.any() else 0.
    maxima[label]=max(maxima.get(label,0.),delta)
    assert np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),(label,delta,atol,rtol)


def independent_short(spot,um):
    """Independent formulas; pandas rolling arithmetic matches registered sums."""
    close=spot.close.where(np.isfinite(spot.close)&spot.close.gt(0))
    opening=spot.open.where(np.isfinite(spot.open)&spot.open.gt(0))
    high=spot.high.where(np.isfinite(spot.high)&spot.high.gt(0))
    low=spot.low.where(np.isfinite(spot.low)&spot.low.gt(0))
    new=pd.DataFrame(index=spot.index)
    for k in (4,16,48):
        full=close.notna().rolling(k+1,min_periods=k+1).sum().eq(k+1)
        new[f"spot_log_return{k}"]=(np.log(close)-np.log(close.shift(k))).where(full)
    candle=opening.notna()&close.notna()&high.notna()&low.notna()
    candle &= high.ge(low)&high.ge(opening)&high.ge(close)&low.le(opening)&low.le(close)
    new["spot_body_sign1"]=np.sign(np.log(close/opening)).where(candle)
    span=(high-low).where(candle)
    clv=((2*close-high-low)/span.where(span>0)).where(candle)
    new["spot_close_location1"]=clv.mask(candle&span.eq(0),0.)
    for market,data in (("spot",spot),("perp",um)):
        q,b=data.quote_volume,data.taker_buy_quote
        okay=np.isfinite(q)&q.gt(0)&np.isfinite(b)&b.ge(0)&b.le(q)
        new[f"{market}_weighted_flow4"]=((2*b-q).where(okay).rolling(4,min_periods=4).sum()
                                          /q.where(okay).rolling(4,min_periods=4).sum())
    q=spot.quote_volume.where(np.isfinite(spot.quote_volume)&spot.quote_volume.gt(0))
    activity=np.log(q.rolling(24,min_periods=24).mean()/q.rolling(672,min_periods=669).mean())
    new["spot_quote_activity24_672"]=activity.where(np.arange(len(spot))>=671)
    return new.loc[:,PRICE+FLOW].replace([np.inf,-np.inf],np.nan).shift(1)


def normal_equations(x,y):
    """T-only independent moments, centered Ridge100 system; no sklearn fit."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    n,p=x.shape
    assert y.shape==(n,) and np.isfinite(x).all() and np.isfinite(y).all()
    mean=np.array([math.fsum(float(v)/n for v in x[:,j]) for j in range(p)])
    variance=np.array([math.fsum((float(v)-mean[j])**2/n for v in x[:,j]) for j in range(p)])
    eps=np.finfo(float).eps
    constant=variance <= n*eps*variance+(n*mean*eps)**2
    scale=np.sqrt(variance);scale[constant]=1.
    z=(x-mean)/scale
    offset=np.array([math.fsum(float(v)/n for v in z[:,j]) for j in range(p)])
    ymean=math.fsum(float(v)/n for v in y)
    zc=z-offset;yc=y-ymean
    gram=np.einsum("ij,ik->jk",zc,zc,optimize=False)
    rhs=np.einsum("ij,i->j",zc,yc,optimize=False)
    system=gram+100*np.eye(p)
    with threadpool_limits(limits=2): coef=solve(system,rhs,assume_a="pos",check_finite=True)
    intercept=ymean-math.fsum(float(a)*float(b) for a,b in zip(offset,coef))
    return {"mean":mean,"variance":variance,"scale":scale,"coefficient":coef,"intercept":intercept,
            "normal_residual_maxabs":float(np.max(np.abs(np.einsum("ij,j->i",system,coef,optimize=False)-rhs)))}


def scalar_predict(x,mean,scale,coef,intercept):
    return np.asarray([float(intercept)+math.fsum((float(v)-float(m))/float(s)*float(c)
                       for v,m,s,c in zip(row,mean,scale,coef)) for row in x])


def self_test():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng=np.random.default_rng(20260906)
    x=rng.normal(size=(640,6));x[:,2]=1.;x[:,4]=x[:,0]+.1*x[:,1]
    y=.02*x[:,0]-.01*x[:,1]+rng.normal(scale=.001,size=640)
    got=normal_equations(x,y)
    with threadpool_limits(limits=2): model=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(x,y)
    maxima={}
    near(got["coefficient"],model[1].coef_,"synthetic_coef",maxima,1e-12,1e-10)
    near(got["intercept"],model[1].intercept_,"synthetic_intercept",maxima)
    predicted=scalar_predict(x,got["mean"],got["scale"],got["coefficient"],got["intercept"])
    with threadpool_limits(limits=2): reference=model.predict(x)
    near(predicted,reference,"synthetic_prediction",maxima)
    print(json.dumps({"synthetic_only":True,"maximum_absolute_difference":maxima,"tolerances":TOLERANCES}))


def audit(config_path,output):
    cfg=yaml.safe_load(Path(config_path).read_text());out=Path(cfg["output_dir"])
    reg,pre,results=(read(out/(n+".json")) for n in ("registration","preflight","results"))
    assert reg["config"]==cfg and reg["config_sha256"]==sha(config_path)
    assert reg["preflight_sha256"]==cfg["preflight_sha256"]==sha(out/"preflight.json")
    assert results["registration_sha256"]==digest(reg)
    assert cfg["groups"]==list(GROUPS) and cfg["ridge_alpha"]==100.
    assert cfg["data_cutoff"]=="2023-04-16T13:45:00Z" and cfg["development_folds"]==list(range(5,13))
    bound={}
    for mapping in (cfg["source_bindings"],pre["direct_source_bindings"],pre["source_artifact_bindings"]):
        for p,h in mapping.items():
            assert p not in bound or bound[p]==h
            bound[p]=h
    for p,h in bound.items(): assert sha(p)==h,p
    assert sha(SPOT)=="5e20e81e86f76b95d1301be7a8a366aa9ad78134f954ec8c9dbf83c0db1acf69"
    um_sidecar=read(UM.with_suffix(".sha256.json"));assert sha(UM)==um_sidecar["data_sha256"]
    # These are unchanged OLD feature helpers, each covered by source bindings.
    from unidream.experiments.oracle_frontier_features import make_feature_groups
    from unidream.experiments.oracle_derivative_features import make_derivative_groups
    from unidream.experiments.oracle_derivative_delay_features import make_delayed_perp_groups
    from unidream.experiments.oracle_risk_calibration import trailing_variances
    bars=pd.read_parquet(SPOT,filters=[("bar_open_ts","<",CUTOFF)])
    bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq="15min"))
    um=pd.read_parquet(UM,filters=[("bar_open_ts","<",CUTOFF)]).reindex(bars.index)
    index=bars.index;assert index[-1]<CUTOFF
    bars["bar_available"]=bars[["open","high","low","close"]].notna().all(axis=1)
    old,derivative,delayed=make_feature_groups(bars),make_derivative_groups(bars,um),make_delayed_perp_groups(bars,um)
    common=np.isfinite(trailing_variances(bars).to_numpy()).all(axis=1)
    for frame in [old["flow"],*derivative.values(),*delayed.values()]: common &= np.isfinite(frame.to_numpy()).all(axis=1)
    short=independent_short(bars,um);base=old["technical"]
    groups={"technical":base,"technical_short_price":pd.concat([base,short.loc[:,PRICE]],axis=1),
        "technical_short_flow":pd.concat([base,short.loc[:,FLOW]],axis=1),"technical_short_both":pd.concat([base,short],axis=1)}
    # Metadata availability only; target magnitudes are constructed inside T below.
    valid=(bars.bar_available.astype(int).rolling(24).sum().shift(-24).eq(24)&bars.open.shift(-1).gt(0)).to_numpy()
    clock=np.asarray((index.hour%6==0)&(index.minute==0))
    maturity=index+pd.Timedelta(minutes=375)
    opening,closing=bars.open.to_numpy(float),bars.close.to_numpy(float)
    delay_pre=read("codex_outputs/oracle_derivative_delay_v1/preflight.json")
    maxima,folds={},[]
    counts={"models":0,"evaluation_forecasts":0,"calibration_forecasts":0,"masks":0,
        "runtime_artifact_hashes":0,"scalar_calibration_rows":0,"scalar_evaluation_rows":0,
        "baseline_model_state_parities":0,"normal_equation_models":0}
    model_hashes={}
    for f in range(5,13):
        manifest=read(out/f"fold_{f}.json")
        assert manifest["registration_sha256"]==digest(reg) and len(manifest["artifact_sha256"])==53
        for p,h in manifest["artifact_sha256"].items(): assert sha(p)==h,p
        counts["runtime_artifact_hashes"]+=53
        start=pd.Timestamp("2020-04-16T13:45:00Z")+pd.DateOffset(months=3*(f-1))
        end=start+pd.DateOffset(months=3);scale_start=start-pd.DateOffset(months=6)
        interval_start=start-pd.DateOffset(months=3);train_start=start-pd.DateOffset(months=24)
        masks={name:np.asarray((index>=a)&(maturity<b))&common&valid&clock for name,a,b in
            (("fit",train_start,scale_start),("scale",scale_start,interval_start),("interval",interval_start,start))}
        e=np.asarray((index>=start)&(index<end));c=np.asarray((index>=scale_start)&(index<start))
        masks["inference"]=e&common&clock
        masks["score"]=masks["inference"]&valid&np.asarray(maturity<=end)
        masks["predict"]=np.asarray((index>=scale_start)&(index<end))&common&clock
        support=next(s for s in pre["support"] if s["fold"]==f)
        ancestor=next(s for s in delay_pre["folds"] if s["fold"]==f)
        for name,mask in masks.items():
            assert mask_digest(index,mask)==support["mask_sha256"][name]==ancestor["mask_sha256"][name]
            assert int(mask.sum())==support["counts"][name]
            assert all(np.isfinite(frame.to_numpy()[mask]).all() for frame in groups.values())
            counts["masks"]+=1
        proof=read(out/"provenance"/f"fold{f}_fit.json")
        assert proof["fit_source_binding"]==support
        provenance=proof["fit_provenance"]
        assert provenance["index_sha256"]==index_digest(index)
        for name in ("fit","predict"):
            assert provenance["mask_position_sha256"][name]==position_digest(masks[name])
            assert provenance["mask_counts"][name]==int(masks[name].sum())
            pos=np.flatnonzero(masks[name]);assert provenance["mask_ranges"][name]==[int(pos[0]),int(pos[-1])]
        positions=np.flatnonzero(masks["fit"])
        assert len(positions)>=512 and maturity[positions[-1]]<scale_start
        yfit=np.log(closing[positions+24]/opening[positions+1])
        assert np.isfinite(yfit).all() and digest(yfit.tolist())==support["fit_return_sha256"]
        assert matrix_digest(yfit)==provenance["fit_return_sha256"]
        reference=arrays(Path("codex_outputs/oracle_mean_reliability_decisions_v1/forecasts")/f"fold{f}_technical_reliability.npz")
        original_cal=arrays(Path("codex_outputs/oracle_frozen_procedure_parity_v1/calibration")/f"fold{f}_technical.npz")
        residuals=[]
        for g in GROUPS:
            frame=groups[g];xfit=frame.to_numpy()[masks["fit"]];xp=frame.to_numpy()[masks["predict"]]
            assert list(frame.columns)==support["feature_columns"][g]==provenance["feature_columns"][g]
            assert digest(xfit.tolist())==support["fit_feature_sha256"][g]
            assert matrix_digest(xfit)==provenance["fit_features_sha256"][g]
            assert matrix_digest(xp)==provenance["predict_features_sha256"][g]
            assert matrix_digest(np.column_stack((xfit,yfit)))==provenance["fit_features_and_return_sha256"][g]
            model_path=out/"models"/f"fold{f}_{g}.joblib";model=joblib.load(model_path)
            model_hashes[str(model_path)]=sha(model_path)
            assert list(model.named_steps)==["standardscaler","ridge"]
            scaler,ridge=model[0],model[1]
            assert ridge.alpha==100. and ridge.fit_intercept and scaler.with_mean and scaler.with_std
            assert scaler.n_samples_seen_==len(yfit) and scaler.n_features_in_==len(frame.columns)
            for a in (scaler.mean_,scaler.var_,scaler.scale_,ridge.coef_,np.asarray(ridge.intercept_)): assert np.isfinite(a).all()
            if g=="technical":
                old_path=Path("codex_outputs/oracle_derivative_delay_v1/models")/f"fold{f}_technical_return.joblib"
                assert str(old_path) in bound and sha(old_path)==bound[str(old_path)]
                old_model=joblib.load(old_path)
                for a,b,attrs in ((scaler,old_model[0],("mean_","var_","scale_","n_features_in_","n_samples_seen_")),
                                  (ridge,old_model[1],("coef_","intercept_","n_features_in_"))):
                    assert a.get_params()==b.get_params()
                    for attr in attrs:
                        near(getattr(a,attr),getattr(b,attr),"baseline_state_"+attr,maxima,1e-14,1e-12)
                counts["baseline_model_state_parities"]+=1
            independent=normal_equations(xfit,yfit)
            for key,saved_attr in (("mean",scaler.mean_),("variance",scaler.var_),("scale",scaler.scale_)):
                near(independent[key],saved_attr,"normal_scaler_"+key,maxima,1e-10,1e-12)
            near(independent["coefficient"],ridge.coef_,"normal_coefficient",maxima,1e-12,1e-10)
            near(independent["intercept"],ridge.intercept_,"normal_intercept",maxima,1e-12)
            scalar=scalar_predict(xp,scaler.mean_,scaler.scale_,ridge.coef_,ridge.intercept_)
            normal=scalar_predict(xp,independent["mean"],independent["scale"],independent["coefficient"],independent["intercept"])
            full_scalar=np.full(len(index),np.nan);full_scalar[masks["predict"]]=scalar
            full_normal=np.full(len(index),np.nan);full_normal[masks["predict"]]=normal
            ca=arrays(out/"calibration"/f"fold{f}_{g}.npz")
            ev=arrays(out/"forecasts"/f"fold{f}_{g}_raw.npz")
            equal(ca["timestamps"],index[c].asi8,"cal calendar")
            equal(ca["scale_mask"],masks["scale"][c],"scale mask")
            equal(ca["interval_mask"],masks["interval"][c],"interval mask")
            equal(ca["actual"],original_cal["actual"],"old cal labels")
            for key in reference:
                if key!="mu": equal(ev[key],reference[key],"E reference "+key)
            equal(np.isfinite(ca["mu"]),masks["predict"][c],"cal predict")
            equal(np.isfinite(ev["mu"]),masks["inference"][e],"eval inference")
            assert float(ev["fit_return_mean"])==float(np.mean(yfit))
            for seg,selected,saved in (("calibration",c,ca),("evaluation",e,ev)):
                near(full_scalar[selected],saved["mu"],"saved_scalar_"+seg,maxima,1e-14)
                near(full_normal[selected],saved["mu"],"normal_prediction_"+seg,maxima,1e-12)
            if g=="technical":
                near(ca["mu"],original_cal["mu"],"baseline_calibration_raw",maxima,1e-14,1e-12)
                old_eval=arrays(Path("codex_outputs/oracle_derivative_delay_v1/forecasts")/f"fold{f}_technical_raw.npz")
                near(ev["mu"],old_eval["mu"],"baseline_evaluation_raw",maxima,1e-14,1e-12)
            counts["models"]+=1;counts["normal_equation_models"]+=1
            counts["calibration_forecasts"]+=1;counts["evaluation_forecasts"]+=1
            counts["scalar_calibration_rows"]+=int(masks["predict"][c].sum())
            counts["scalar_evaluation_rows"]+=int(masks["inference"][e].sum())
            residuals.append(independent["normal_residual_maxabs"])
        folds.append({"fold":f,"fit_rows":len(positions),"counts":support["counts"],
            "fit_only_labels_sha256":matrix_digest(yfit),"normal_equation_residual_maxabs":max(residuals)})
        print(json.dumps({"fold_audited":f,"models":4,"maximum_absolute_difference":maxima}),flush=True)
    assert counts["models"]==32 and counts["runtime_artifact_hashes"]==424 and counts["masks"]==48
    report={"status":"passed","schema":"independent-short-feature-forecast-audit-v1",
        "audit_script_sha256":sha(__file__),"config_sha256":sha(config_path),
        "registration_sha256":sha(out/"registration.json"),"results_sha256":sha(out/"results.json"),
        "source_revision":reg["source_revision"],"preflight_sha256":sha(out/"preflight.json"),
        "source_bindings_verified":len(bound),"source_bindings_digest":digest(bound),
        "spot_sha256":sha(SPOT),"um_sha256":sha(UM),"decoded_data_cutoff_exclusive":CUTOFF.isoformat(),
        "model_bindings":model_hashes,"counts":counts,"tolerances":TOLERANCES,
        "maximum_absolute_difference":maxima,"folds":folds,
        "independent_feature_scope":"All eight new columns; exact fit/predict matrix hashes. Frozen old helpers reused for technical29/common mask.",
        "normal_equation_scope":"Per-fold T returns only; no S/I/E label enters scaler moments, coefficients or predictions.",
        "no_weight_risk_or_policy_fit":True,"no_new_loss_or_economic_comparisons":True,
        "warning_scope":"Numerical agreement does not establish the cause of prior matmul warnings or out-of-sample generalization."}
    output=Path(output);assert not output.exists(),"audit output already exists"
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":"passed","path":str(output),"sha256":sha(output),"counts":counts,"maximum_absolute_difference":maxima}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-run",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--worktree",default="/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905")
    parser.add_argument("--config",default="configs/oracle_short_feature_decisions_20260906.yaml")
    parser.add_argument("--output",default="/tmp/oracle_short_feature_forecast_audit_20260906.json")
    args=parser.parse_args()
    if args.self_test: self_test()
    elif not args.authorized_run: parser.error("Real-data audit requires root's explicit postfreeze go")
    else:
        os.chdir(args.worktree);sys.path.insert(0,args.worktree)
        audit(Path(args.config),Path(args.output))
