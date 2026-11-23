"""Report-only continuous cash/units control-release evaluation.

All market bars remain in each account, including unavailable decision bars and
fold boundaries. Ruin remains -100% ROI/-inf log growth; affected hypotheses
stay in their registered multiplicity families with unavailable inference,p=1.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_control_release_accounting import initial_state, step
from .wm_inference_robust_bootstrap import infer_daily
from .wm_inference_robust_hac import hac_mean
from .wm_significance_statistics import holm_fixed_family


ARMS = ("A", "C")
CONDITIONS = ("control", "cap15", "cap20", "speed", "free")
BASELINES = ("constant_cap", "ma20d", "voltarget10")
BAR_NS = 900 * 10**9
DIAG = ("equity", "net_simple_return", "net_log_return", "executed_exposure", "turnover",
        "fee", "borrow", "bankrupt", "bankrupt_event", "alive", "decision_available")


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _checked_path(ledger):
    required = {"timestamp", "decision_id", "fold", *DIAG}
    if ledger.empty or not required.issubset(ledger):
        raise ValueError("complete continuous cash/units execution ledger required")
    frame = ledger.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    for name in ("decision_id", "fold"):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError("finite integer calendar/fold identities required")
        frame[name] = values.astype(np.int64)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if (frame.timestamp.isna().any() or frame.timestamp.duplicated().any()
            or np.any(np.diff(frame.timestamp.astype("int64")) != BAR_NS)
            or np.any(np.diff(frame.decision_id) != 1)):
        raise ValueError("every consecutive15-minute calendar bar must remain, including gaps in decisions")
    for name in ("bankrupt", "bankrupt_event", "alive", "decision_available"):
        if frame[name].isna().any() or not frame[name].isin([True, False]).all():
            raise ValueError("complete boolean account flags required")
        frame[name] = frame[name].astype(bool)
    return frame


def metrics(ledger, cfg):
    """Describe one continuous path; period callers supply its starting NAV.

    Fee/borrow are quote amounts with original initial NAV1. Period columns
    separately expose each period's starting NAV and fee fraction denominator.
    Maximum DD includes the period's starting NAV. After an earlier bankruptcy,
    a zero-start period has unavailable percentage metrics rather than0% ROI.
    """
    frame = _checked_path(ledger)
    initial = float(cfg.get("_period_start_equity", 1.))
    if not np.isfinite(initial) or initial < 0:
        raise ValueError("finite nonnegative period starting NAV required")
    r = frame.net_simple_return.to_numpy(float)
    equity = frame.equity.to_numpy(float)
    logs = frame.net_log_return.to_numpy(float)
    numeric = frame[["equity", "net_simple_return", "executed_exposure", "turnover", "fee", "borrow"]].to_numpy(float)
    if (not np.isfinite(numeric).all() or np.any(r < -1) or np.any(equity < 0)
            or frame[["executed_exposure", "turnover", "fee", "borrow"]].lt(0).any().any()):
        raise ValueError("finite nonnegative account assets/costs and returns>=-1 required")
    expected = initial * np.cumprod(1+r)
    if not np.allclose(equity, expected, rtol=1e-9, atol=1e-11):
        raise ValueError("equity differs from continuous self-financing return product; resets/injections forbidden")
    bankrupt = frame.bankrupt.to_numpy(bool)
    events = frame.bankrupt_event.to_numpy(bool)
    if (not np.array_equal(bankrupt, equity == 0) or not np.array_equal(frame.alive.to_numpy(), ~bankrupt)
            or events.sum() > 1 or np.any(np.diff(bankrupt.astype(int)) < 0)
            or np.any(events & ~bankrupt)):
        raise ValueError("bankruptcy must be terminal and absorbing")
    death = r == -1
    expected_logs = np.zeros(len(frame))
    expected_logs[death] = -np.inf
    expected_logs[~death] = np.log1p(r[~death])
    if not np.array_equal(events, death) or not np.allclose(logs, expected_logs, rtol=0, atol=1e-10):
        raise ValueError("performance log return must retain -inf at the one bankruptcy event")
    first_dead = np.flatnonzero(bankrupt)
    if len(first_dead):
        after = slice(first_dead[0]+1, None)
        if np.any(r[after] != 0) or frame.loc[first_dead[0]+1:, ["fee", "borrow", "turnover"]].ne(0).any().any():
            raise ValueError("dead accounts must retain zero assets and no later trading/returns")
    if initial == 0 and (not bankrupt.all() or events.any() or np.any(r != 0)):
        raise ValueError("zero-start period must belong to an already absorbed account")
    annual = float(cfg.get("accounting", {}).get("annual_bars", 35040))
    if not np.isfinite(annual) or annual <= 0:
        raise ValueError("positive annualization required")
    wealth = np.r_[initial, equity]
    peak = np.maximum.accumulate(wealth)
    dd = np.divide(peak-wealth, peak, out=np.zeros_like(wealth), where=peak>0)
    exposure = frame.executed_exposure.to_numpy(float)
    fee, borrow = float(frame.fee.sum()), float(frame.borrow.sum())
    available = initial > 0
    return {"n_rows": len(frame), "n_calendar_days": frame.timestamp.dt.floor("D").nunique(),
        "period_start_equity": initial, "final_equity": float(equity[-1]),
        "net_total_return": float(equity[-1]/initial-1) if available else None,
        "annualized_net_log_growth": float(logs.sum()*annual/len(frame)) if available else None,
        "max_drawdown": float(dd.max()) if available else None,
        "annualized_volatility": float(np.std(r,ddof=1)*np.sqrt(annual)) if available and len(r)>1 else None,
        "mean_exposure": float(exposure.mean()), "median_exposure": float(np.median(exposure)),
        "exposure_max": float(exposure.max()), "near_cash_fraction": float(np.mean(exposure<=.01)),
        "above1_fraction": float(np.mean(exposure>1.+1e-9)), "above112_fraction": float(np.mean(exposure>1.12+1e-9)),
        "above15_fraction": float(np.mean(exposure>1.5+1e-9)), "above2_fraction": float(np.mean(exposure>2.+1e-9)),
        "turnover": float(frame.turnover.sum()), "fee_quote": fee, "borrow_quote": borrow,
        "fees_in_initial_wealth": fee, "borrow_in_initial_wealth": borrow,
        "fees_in_period_start_wealth": fee/initial if available else None,
        "borrow_in_period_start_wealth": borrow/initial if available else None,
        "insolvent": bool(bankrupt.any()), "bankruptcy_events": int(events.sum()),
        "bankruptcy_timestamp": str(frame.loc[events,"timestamp"].iloc[0]) if events.any() else None,
        "n_effective_decisions": int(frame.decision_available.sum()), "n_resets": 0,
        "percentage_metrics_available": available, "log_metric_available": bool(available and not bankrupt.any())}


def _condition_cfg(cfg, condition):
    result = copy.deepcopy(cfg)
    definition = cfg["condition_definitions"][condition]
    result["accounting"].update({k: float(definition[k]) for k in ("position_max", "max_position_step")})
    return result


def _past_sum(values, ids, window):
    finite = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(finite, values, 0.))]
    missing = np.r_[0, np.cumsum(~finite)]
    if np.any(ids<window) or np.any(missing[ids]-missing[ids-window]):
        raise ValueError("complete causal raw baseline history required; no fills or omitted bars")
    return sums[ids]-sums[ids-window]


def baseline_targets(raw_ohlc, raw_index_ns, decision_ids, cfg):
    """Decision d sees raw closes<=d-1; history remains the full raw grid."""
    ohlc, times, ids = np.asarray(raw_ohlc,float), np.asarray(raw_index_ns), np.asarray(decision_ids)
    if (ohlc.ndim!=2 or ohlc.shape[1]!=4 or times.shape!=(len(ohlc),)
            or not np.issubdtype(times.dtype,np.integer) or np.any(np.diff(times)!=BAR_NS)
            or ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids)
            or np.any(ids<1) or np.any(ids>=len(ohlc))):
        raise ValueError("regular complete raw OHLC calendar and integer decision IDs required")
    close = ohlc[:,3]
    close = np.where(np.isfinite(close)&(close>0),close,np.nan)
    settings = cfg.get("baselines", {})
    ma = int(settings.get("ma_window_bars",1920)); vol = int(settings.get("volatility_window_bars",672))
    target_sigma = float(settings.get("volatility_target_annual",.10))
    if ma<1 or vol<1 or not np.isfinite(target_sigma) or target_sigma<=0:
        raise ValueError("positive fixed baseline windows and annual sigma required")
    sma = _past_sum(close,ids,ma)/ma
    log_r = np.r_[np.nan,np.log(close[1:]/close[:-1])]
    sigma = np.sqrt(_past_sum(log_r**2,ids,vol)/vol*float(cfg["accounting"]["annual_bars"]))
    scale = np.divide(target_sigma,sigma,out=np.full_like(sigma,np.inf),where=sigma>0)
    return {condition: {"constant_cap": np.full(len(ids),float(cfg["condition_definitions"][condition]["position_max"])),
        "ma20d": np.where(close[ids-1]>sma,float(cfg["condition_definitions"][condition]["position_max"]),0.),
        "voltarget10": np.minimum(float(cfg["condition_definitions"][condition]["position_max"]),scale)}
        for condition in CONDITIONS}


def build_baselines(raw_ohlc, raw_index_ns, decision_available, reference, cfg):
    """Continuous same-availability controls using the shared cash/units engine."""
    import torch
    torch.set_num_threads(int(cfg.get("runtime",{}).get("cpu_threads",1)))
    ids=reference.decision_id.to_numpy(np.int64)
    if len(ids)==0 or np.any(np.diff(ids)!=1):
        raise ValueError("continuous reference calendar required for every baseline")
    available=np.asarray(decision_available)
    if available.dtype!=bool or available.shape!=(len(raw_ohlc),):
        raise ValueError("common raw-grid decision-availability bool mask required")
    if not np.array_equal(pd.to_datetime(reference.timestamp,utc=True).astype("int64"),np.asarray(raw_index_ns)[ids]):
        raise ValueError("baseline/reference raw timestamps differ")
    targets=baseline_targets(raw_ohlc,raw_index_ns,ids,cfg)
    groups=defaultdict(list)
    for condition in CONDITIONS:
        active=_condition_cfg(cfg,condition)
        key=(active["accounting"]["position_max"],active["accounting"]["max_position_step"])
        for baseline in BASELINES:
            groups[key].append((condition,baseline,targets[condition][baseline],False))
    control=_condition_cfg(cfg,"control")
    key=(control["accounting"]["position_max"],control["accounting"]["max_position_step"])
    groups[key].extend([("descriptive","bh",np.ones(len(ids)),True),
                        ("descriptive","constant08",np.full(len(ids),.08),False)])
    result={}
    for (cap,speed),members in groups.items():
        active=copy.deepcopy(cfg);active["accounting"].update(position_max=float(cap),max_position_step=float(speed))
        state=initial_state(len(members),"cpu",float(raw_ohlc[ids[0]-1,3]),active)
        values={name:[] for name in DIAG}
        desired=np.stack([item[2] for item in members],axis=1)
        never=np.asarray([item[3] for item in members],bool)
        for row,d in enumerate(ids):
            current=state["requested"]
            request=(current+torch.as_tensor(desired[row],dtype=current.dtype).sub(current).clamp(-speed,speed)).clamp(0,cap)
            mask=torch.as_tensor(available[d]&~never,dtype=torch.bool)
            state,diag=step(state,request,float(raw_ohlc[d,0]),float(raw_ohlc[d,2]),float(raw_ohlc[d,3]),active,decision_available=mask)
            for name in DIAG:values[name].append(diag[name].numpy().copy())
        arrays={name:np.stack(rows) for name,rows in values.items()}
        for j,(condition,baseline,_,_) in enumerate(members):
            frame=reference[["timestamp","decision_id","fold"]].copy().reset_index(drop=True)
            for name,array in arrays.items():frame[name]=array[:,j]
            frame["condition"],frame["baseline"]=condition,baseline
            result[(condition,baseline)]=frame
    return result


def definitions():
    """All53 hypotheses:13 primary and40 secondary; descriptive BH excluded."""
    rows=[]
    for arm in ARMS:
        for previous,current in zip(CONDITIONS[:-1],CONDITIONS[1:]):
            rows.append({"condition":f"{arm}_{current}_minus_{previous}","family":"primary",
                "left":("trained",arm,current),"right":("trained",arm,previous)})
    for condition in CONDITIONS:
        rows.append({"condition":f"C_{condition}_minus_A_{condition}","family":"primary",
            "left":("trained","C",condition),"right":("trained","A",condition)})
    for arm in ARMS:
        for condition in CONDITIONS:
            rows.append({"condition":f"{arm}_{condition}_trained_minus_frozen","family":"secondary",
                "left":("trained",arm,condition),"right":("frozen",arm,condition)})
            for baseline in BASELINES:
                rows.append({"condition":f"{arm}_{condition}_minus_{baseline}","family":"secondary",
                    "left":("trained",arm,condition),"right":("baseline",condition,baseline)})
    return rows


def _correct(frame, hac=False):
    result=frame.copy();result["p_holm_family"]=np.nan;result["p_holm_all53"]=np.nan
    keys=["lag_days"] if hac else ["method","block_days","estimand"]
    for _,part in result.groupby(keys,sort=False):
        if len(part)!=53:raise ValueError("all53 registered contrasts, including unavailable ones, must remain")
        p={str(i):float(row.p_value) if pd.notna(row.p_value) else 1. for i,row in part.iterrows()}
        global_q=holm_fixed_family(p,family_size=53)
        for family,group in part.groupby("family"):
            size={"primary":13,"secondary":40}[family]
            if len(group)!=size:raise ValueError("fixed13/40 families must not change")
            q=holm_fixed_family({str(i):p[str(i)] for i in group.index},family_size=size)
            for i in group.index:
                result.loc[i,"p_holm_family"]=q[str(i)];result.loc[i,"p_holm_all53"]=global_q[str(i)]
    result["significant_difference"]=result.primary_inference & result.inference_available & result.p_holm_family.lt(.05)
    result["significant_improvement"]=result.significant_difference & result.estimate.gt(0)
    return result


def _summary(frame,keys):
    rows=[]
    metrics_list=["net_total_return","max_drawdown","annualized_volatility","mean_exposure","near_cash_fraction",
        "above1_fraction","above112_fraction","above15_fraction","above2_fraction","turnover",
        "fee_quote","borrow_quote","annualized_net_log_growth"]
    for group,part in frame.groupby(keys,sort=False):
        group=group if isinstance(group,tuple) else (group,)
        row=dict(zip(keys,group));row.update(n_seeds=len(part),n_unique_market_rows=int(part.n_rows.iloc[0]),
            n_insolvent_seeds=int(part.insolvent.sum()))
        for name in metrics_list:
            values=part[name].to_numpy(float)
            for suffix,operation in [("mean",np.mean),("median",np.median),("min",np.min),("max",np.max)]:
                row[name+"_"+suffix]=float(operation(values)) if len(values) and not np.isnan(values).any() else None
            row[name+"_std"]=float(np.std(values,ddof=1)) if len(values)>1 and np.isfinite(values).all() else None
        for name,prefix,maximum in [("max_drawdown","worst_drawdown",True),("net_total_return","worst_return",False)]:
            values=pd.to_numeric(part[name],errors="coerce")
            if values.notna().any():
                worst=float(values.max() if maximum else values.min());ids=part.loc[values.eq(worst),"seed"].tolist()
                row.update({prefix:worst,prefix+"_seed":int(ids[0]),prefix+"_seed_ids":"|".join(map(str,ids))})
            else:row.update({prefix:None,prefix+"_seed":None,prefix+"_seed_ids":""})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_paths(paths,baselines,cfg):
    """Pure in-memory report evaluation; aggregate() supplies the guarded IO."""
    seeds=list(cfg["seeds"])
    expected={(kind,arm,condition,seed) for kind in ("trained","frozen") for arm in ARMS for condition in CONDITIONS for seed in seeds}
    if set(paths)!=expected or len(seeds)!=len(set(seeds)):
        raise ValueError("every registered trained/frozen arm/condition/seed path required")
    expected_baselines={(c,b) for c in CONDITIONS for b in BASELINES}|{("descriptive","bh"),("descriptive","constant08")}
    if set(baselines)!=expected_baselines:raise ValueError("all15 registered plus2 descriptive baseline paths required")
    reference=None;pooled=[];periods={"fold":[],"monthly":[]};checked={}
    for key,path in paths.items():
        frame=_checked_path(path)
        cohort=frame[["timestamp","decision_id","fold"]]
        if reference is not None and not reference.equals(cohort):raise ValueError("all paths must share the full continuous calendar")
        reference=cohort;checked[key]=frame
        identity=dict(zip(("kind","arm","condition","seed"),key))
        pooled.append({**identity,**metrics(frame,cfg)})
        labels=frame.timestamp.dt.tz_localize(None).dt.to_period("M").astype(str)
        for group_key,groups in [("fold",frame.groupby("fold",sort=False)),("monthly",frame.groupby(labels,sort=False))]:
            for period,part in groups:
                start=int(part.index[0]);initial=float(frame.equity.iloc[start-1]) if start else 1.
                periods[group_key].append({**identity,"period":period,**metrics(part,{**cfg,"_period_start_equity":initial})})
    n_expected=cfg.get("statistics",{}).get("expected_market_rows",52608)
    if len(reference)!=n_expected:raise ValueError("registered complete Test calendar row count changed")
    base_metrics=[]
    for key,path in baselines.items():
        frame=_checked_path(path)
        if not reference.equals(frame[["timestamp","decision_id","fold"]]):raise ValueError("baseline calendar/cohort differs")
        baselines[key]=frame;base_metrics.append({"condition":key[0],"baseline":key[1],**metrics(frame,cfg)})
    day=reference.timestamp.dt.floor("D")
    day_codes,days=pd.factorize(day,sort=True)
    counts=np.bincount(day_codes);segments=np.zeros(len(days),np.int64)
    day_ids=pd.DatetimeIndex(days).asi8//(86400*10**9)
    specs=definitions();valid=[];sums=[];unavailable={}
    for spec in specs:
        values=[];reason=None
        for seed in seeds:
            left=checked[(*spec["left"],seed)]
            right=baselines[spec["right"][1:]] if spec["right"][0]=="baseline" else checked[(*spec["right"],seed)]
            x,y=left.net_log_return.to_numpy(float),right.net_log_return.to_numpy(float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                reason="bankruptcy in at least one compared seed; mean log growth difference is unavailable"
                break
            values.append(np.bincount(day_codes,weights=x-y,minlength=len(days)))
        if reason:unavailable[spec["condition"]]=reason
        else:valid.append(spec);sums.append(np.stack(values,axis=1))
    panels=np.stack(sums,axis=1) if sums else np.empty((len(days),0,len(seeds)))
    settings=cfg.get("statistics",{});blocks=settings.get("block_days",[7,14,30]);methods=settings.get("methods",["circular"])
    if valid:
        answer=infer_daily(panels,counts,segments,{**cfg,"condition_names":[s["condition"] for s in valid]},day_ids=day_ids)
        inference=answer["inference"].to_dict("records");design=answer["design"]
    else:inference=[];design={"primary":"circular7day actual H0-centered paired market fixed-seed inference"}
    metadata={s["condition"]:s for s in specs}
    for method in methods:
        for block in blocks:
            for estimand in ("market_fixed_seeds","hierarchical_seeds"):
                for name,reason in unavailable.items():
                    inference.append({"condition":name,"method":method,"block_days":int(block),"estimand":estimand,
                        "primary_inference":method=="circular" and block==7 and estimand=="market_fixed_seeds",
                        "replicates":settings.get("bootstrap_replicates",10000),"estimate":None,"ci_low":None,"ci_high":None,
                        "p_value":1.,"inference_available":False,"unavailable_reason":reason,
                        "n_unique_market_rows":len(reference),"n_training_seeds":len(seeds),"n_daily_units":len(days),"n_segments":1})
    frame=pd.DataFrame(inference)
    frame["family"]=frame.condition.map(lambda name:metadata[name]["family"])
    frame["unavailable_reason"]=frame.get("unavailable_reason",pd.Series(index=frame.index,dtype=str)).fillna("")
    frame=_correct(frame)
    hac=[]
    positions={s["condition"]:j for j,s in enumerate(valid)}
    for spec in specs:
        name=spec["condition"]
        for lag in settings.get("hac_lags",[6,13,29]):
            if name in unavailable:
                row={"estimate":None,"standard_error":None,"ci_low":None,"ci_high":None,"p_value":1.,"inference_available":False,
                    "unavailable_reason":unavailable[name]}
            else:
                result=hac_mean(panels[:,positions[name]],counts,segments,day_ids,lag=lag,annual_bars=cfg["accounting"]["annual_bars"])
                row={k:result[k] for k in ("estimate","standard_error","ci_low","ci_high","p_value","inference_available")}
                if row["p_value"] is None:row["p_value"]=1.
            hac.append({"condition":name,"family":spec["family"],"lag_days":lag,"primary_inference":lag==6,**row})
    pooled=pd.DataFrame(pooled);summary_records=[]
    groups={"all_seeds":seeds,"reused_seeds":cfg.get("reused_seeds",[]),"new_seeds":cfg.get("new_seeds",[])}
    for name,selected in groups.items():
        if selected:summary_records.extend(_summary(pooled.loc[pooled.seed.isin(selected)],["kind","arm","condition"]).assign(seed_subset=name).to_dict("records"))
    return {"pooled_metrics":pooled,"seed_summary":pd.DataFrame(summary_records),
        "fold_metrics":pd.DataFrame(periods["fold"]),"monthly_by_seed":pd.DataFrame(periods["monthly"]),
        "baseline_metrics":pd.DataFrame(base_metrics),"inference":frame,"primary_inference":frame.loc[frame.primary_inference].copy(),
        "hac_inference":_correct(pd.DataFrame(hac),hac=True),
        "daily_panel":{"sums":panels,"counts":counts,"segment_ids":segments,"day_ids":day_ids,
                       "condition_names":np.asarray([s["condition"] for s in valid])},
        "audit":{"n_market_rows":len(reference),"n_calendar_days":len(days),"n_segments":1,"n_resets":0,
            "n_seeds":len(seeds),"families":{"primary":13,"secondary":40},"combined_family":53,
            "unavailable_comparisons":unavailable,"historical_seen_test":True,"independently_unused_holdout":False,
            "seed_count_multiplies_market_rows":False,"design":design,
            "ruin":"terminal equity0; rawROI-100%,loggrowth-inf; affected contrasts N/A,p1 remain in fixed families",
            "exposure":"close actual units*price/NAV; may drift beyond target cap; below-target bound is not a margin limit",
            "period_fees":"quote amounts relative original initialNAV1; separate period-start denominator is explicit",
            "model":"fixed-APR spot cash borrowing research account; conservative low-NAV ruin, not a futures liquidation model"}}


def aggregate(cfg,registration_sha):
    """Read immutable trained/frozen paths only after all600 Val fits freeze."""
    from . import wm_control_release_benchmark as benchmark
    benchmark.verify_test_barrier(cfg)
    root=Path(cfg["output_dir"]);registration_path=root/"preregistration.json";barrier_path=root/"test_barrier.json"
    if _sha(registration_path)!=registration_sha:raise ValueError("registration changed before report access")
    manifest_path=root/"path_manifest.json";manifest=json.loads(manifest_path.read_text())
    if (manifest.get("complete") is not True or manifest.get("registration_sha256")!=registration_sha
            or manifest.get("barrier_sha256")!=_sha(barrier_path)):
        raise ValueError("complete immutable continuous-path manifest required")
    entries=manifest["paths"]
    expected={(kind,arm,c,seed) for kind in ("trained","frozen") for arm in ARMS for c in CONDITIONS for seed in cfg["seeds"]}
    identities=[(e["kind"],e["arm"],e["condition"],e["seed"]) for e in entries]
    if len(entries)!=len(expected) or set(identities)!=expected:raise ValueError("all trained/frozen paths must be present exactly once")
    inputs={str(registration_path):registration_sha,str(barrier_path):_sha(barrier_path),str(manifest_path):_sha(manifest_path)}
    paths={}
    for entry,key in zip(entries,identities):
        path=Path(entry["path"])
        if entry.get("initial_equity")!=1. or _sha(path)!=entry["sha256"]:raise ValueError("path identity/initialwealth/SHA changed")
        inputs[str(path)]=entry["sha256"]
        frame=pd.read_parquet(path)
        for name,value in zip(("kind","arm","condition","seed"),key):
            if name in frame and not frame[name].eq(value).all():raise ValueError("path metadata contradicts immutable identity")
        paths[key]=frame
    raw={}
    names={"ohlc":"raw_ohlc_path","index_ns":"raw_index_ns_path","decision_available":"decision_available_path"}
    for name,cfg_key in names.items():
        item=manifest["raw_arrays"][name];path=Path(item["path"])
        if path.resolve()!=Path(cfg["data"][cfg_key]).resolve() or _sha(path)!=item["sha256"]:
            raise ValueError("registered raw arrays differ from continuous-account report inputs")
        inputs[str(path)]=item["sha256"];raw[name]=np.load(path,mmap_mode="r")
    reference=_checked_path(next(iter(paths.values())))[["timestamp","decision_id","fold"]]
    ids=reference.decision_id.to_numpy(np.int64)
    if not np.array_equal(reference.timestamp.astype("int64"),raw["index_ns"][ids]):raise ValueError("raw and executed calendar mismatch")
    market_log=np.log(raw["ohlc"][ids,3]/raw["ohlc"][ids-1,3])
    for path in paths.values():
        ordered=path.sort_values("timestamp")
        if "bar_log_return" in ordered and not np.allclose(ordered.bar_log_return,market_log,rtol=0,atol=1e-12):
            raise ValueError("path market returns differ from registered raw prices")
        if "context_available" in ordered and not np.array_equal(ordered.context_available.to_numpy(bool),raw["decision_available"][ids]):
            raise ValueError("path context availability differs from the common immutable support")
        if {"cash","units"}.issubset(ordered):
            nav=ordered.cash.to_numpy(float)+ordered.units.to_numpy(float)*raw["ohlc"][ids,3]
            if not np.allclose(nav,ordered.equity,rtol=1e-9,atol=1e-11):
                raise ValueError("recorded cash/units and raw prices disagree with account equity")
    baselines=build_baselines(raw["ohlc"],raw["index_ns"],raw["decision_available"],reference,cfg)
    report=evaluate_paths(paths,baselines,cfg)
    final=root/"final";final.mkdir(exist_ok=True,parents=True)
    for name,value in report.items():
        if isinstance(value,pd.DataFrame):value.to_csv(final/(name+".csv"),index=False)
    np.savez(final/"daily_panel.npz",**report["daily_panel"])
    pd.concat([p.assign(condition=key[0],baseline=key[1]) for key,p in baselines.items()],ignore_index=True).to_parquet(final/"executed_baselines.parquet",index=False)
    report["audit"].update(registration_sha256=registration_sha,input_sha256=inputs,
        decision_available_bars=int(raw["decision_available"][ids].sum()),n_registered_continuous_paths=len(paths))
    if any(_sha(path)!=value for path,value in inputs.items()):raise ValueError("immutable inputs changed during evaluation")
    benchmark.verify_test_barrier(cfg)
    (final/"audit.json").write_text(json.dumps(report["audit"],ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    _report(final,report)
    return {"complete":True,"registration_sha256":registration_sha,"n_paths":len(paths),
        "n_market_rows":len(reference),"n_seeds":len(cfg["seeds"]),"independently_unused_holdout":False,
        "artifact_sha256":{str(p.relative_to(root)):_sha(p) for p in final.iterdir() if p.is_file()},"report_path":str(final/"index.html")}


def _report(final,result):
    summary=result["seed_summary"]
    main=summary.loc[summary.kind.eq("trained")&summary.seed_subset.eq("all_seeds")]
    cols=["arm","condition","n_seeds","net_total_return_mean","net_total_return_std","net_total_return_median",
        "net_total_return_min","net_total_return_max","max_drawdown_mean","worst_drawdown","worst_drawdown_seed",
        "annualized_volatility_mean","mean_exposure_mean","above1_fraction_mean","above112_fraction_mean",
        "above15_fraction_mean","fee_quote_mean","borrow_quote_mean","n_insolvent_seeds"]
    def table(frame):return "<div class='scroll'>"+frame.to_html(index=False,escape=True,float_format=lambda x:f"{x:.6g}")+"</div>"
    body=["<!doctype html><html lang='ja'><meta charset='utf-8'><title>行動制限解放：連続cash/units評価</title>",
        "<style>body{font:15px system-ui;line-height:1.7;margin:30px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}</style>",
        "<h1>WM有無・レバ上限・速度・BC priorの比較</h1><p>A＝WMなし、C＝WMあり。control→cap15→cap20→speed→freeの順で上限1.12→1.5→2、速度.08→.24、BC prior.01→0を一要因ずつ変更。DD罰則.25は固定です。</p>",
        "<p>全名目Test足を連続計上し、特徴量がない足も保有・価格損益・借入・既存注文を継続。fold境界に無料リセットはありません。初期資産1は既存BTC1倍、entryfeeなし、終端強制清算なしです。</p>",
        "<p>既知の歴史的Testを使う探索研究です。未使用Holdoutではなく、以前の特徴選択・条件探索すべてを今回の53件補正が訂正するわけではありません。seed追加は同じ市場を使う学習乱数感度で、市場標本数は増えません。</p>",
        "<p>収益・DD・保有率は割合（0.10＝10%）、SDは学習seed間の標本標準偏差で95%市場CIではありません。DDは正の損失率。最悪seedを平均に隠しません。fee/borrowは元初期資産1に対する実quote額です。</p>",
        table(main[cols]),"<h2>固定した単純売買対照</h2><p>常時cap、MA20日1920本でcap/0、過去7日672本の年率ボラ10%目標。全rawの過去closeだけを参照し、全方策と同じavailability・lag・費用・借入・速度制限を使用。B&Hと8%目標は記述のみで推論族に追加しません。</p>",
        table(result["baseline_metrics"]),"<h2>登録した対応あり推論</h2><p>主判定は7日循環Block Bootstrap・固定seed平均。隣接差8＋WM有無5＝主族13、学習前後10＋3対照×10＝副族40、全53補正も保存。14/30日・階層seed・HACは感度分析です。</p>",
        "<p>破産はROI−100%・log成長−infを保持します。該当比較はN/A・p=1で族から除かず、非破産seedだけで有意差を作りません。実レバは価格変動で注文上限を超え得ます。固定APR借入とlowでの保守的破産は実際の先物清算シミュレーターではありません。</p>",
        table(result["primary_inference"]),"<h2>元3seed・追加7seed</h2>",table(summary.loc[summary.kind.eq("trained"),["arm","condition","seed_subset","n_seeds","net_total_return_mean","net_total_return_std","max_drawdown_mean","worst_drawdown","worst_drawdown_seed","n_insolvent_seeds"]]),
        "<p><a href='pooled_metrics.csv'>seed別</a> / <a href='seed_summary.csv'>全seed集計</a> / <a href='fold_metrics.csv'>fold別</a> / <a href='monthly_by_seed.csv'>月別</a> / <a href='baseline_metrics.csv'>対照</a> / <a href='inference.csv'>全bootstrap</a> / <a href='hac_inference.csv'>HAC</a> / <a href='audit.json'>監査</a></p></html>"]
    (final/"index.html").write_text("\n".join(body))
