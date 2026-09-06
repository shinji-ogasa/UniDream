"""Independent audit of completed registered Stage13; never execute before authorization.
No canonical metrics/planner/reliability/scoring helper import or base model fitting.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, re, subprocess, sys
import numpy as np
import pandas as pd
import yaml

def scalar_account(frame, targets, contract, forecast=None, inference=None, fallback_enabled=True, inventory_trace=None):
    fee_rate = contract['one_way_cost']
    annual = contract['borrow_annual']
    step = contract['max_step']
    deadband = contract['deadband']
    op = frame.open.to_numpy()
    cl = frame.close.to_numpy()
    schedule = (frame.index.hour % 6 == 0) & (frame.index.minute == 0)
    cash = 0.0
    units = 1.0 / float(op[0])
    equity = []
    exposure = []
    turnover = fees = borrow = 0.0
    trades = 0
    planned = np.full(len(frame), np.nan)
    trace = []
    submitted = planned if forecast is not None else targets
    for t in range(len(frame)):
        price = float(op[t])
        mark = float(cl[t])
        known = math.isfinite(price)
        nav = cash + units * price if known else math.nan
        if known:
            assert nav > 0
        if known and t and schedule[t - 1] and math.isfinite(submitted[t - 1]):
            current = units * price / nav
            intent = min(max(float(submitted[t - 1]), 0.0), 1.12)
            change = max(-step, min(step, intent - current))
            if abs(change) >= deadband:
                desired = current + change
                trade = (desired * nav - units * price) / (1 + fee_rate * desired * (1 if change > 0 else -1))
                fee = fee_rate * abs(trade)
                cash -= trade + fee
                units += trade / price
                turnover += abs(trade) / nav
                fees += fee
                trades += 1
        if inventory_trace is not None and schedule[t] and known:
            nav_at_decision = cash + units * price
            inventory_trace.append((t, nav_at_decision, units * price / nav_at_decision))
        if forecast is not None and schedule[t] and known and (inference[t] or fallback_enabled):
            nav = cash + units * price
            asset = units * price
            current = asset / nav
            if inference[t]:
                mu = float(forecast['mu'][t])
                variance = float(forecast['variance'][t])
                assert math.isfinite(mu) and math.isfinite(variance) and (variance >= 0)
                best = 0.0
                chosen = math.nan
                estimated_turnover = 0.0
                reason = 'learned'
                for delta in (-0.08, -0.04, 0.04, 0.08):
                    intent = min(max(current + delta, 0.5), 1.12)
                    change = max(-step, min(step, intent - current))
                    desired = current + change
                    if change == 0 or abs(change) < deadband:
                        continue
                    trade = (desired * nav - asset) / (1 + fee_rate * desired * (1 if change > 0 else -1))
                    tv = abs(trade) / nav
                    score = (desired - current) * mu - 0.5 * (desired * desired - current * current) * variance - 2 * fee_rate * tv - (max(desired - 1, 0) - max(current - 1, 0)) * annual * 24 / 35040
                    if score > best:
                        best = score
                        chosen = intent
                        estimated_turnover = tv
            else:
                chosen = 1.0
                best = math.nan
                estimated_turnover = math.nan
                reason = 'forecast_unavailable'
            planned[t] = chosen
            trace.append((t, nav, current, best, estimated_turnover, chosen, reason))
        if cash < 0:
            charge = -cash * (math.exp(annual / 35040) - 1)
            cash -= charge
            borrow += charge
        if math.isfinite(mark):
            value = cash + units * mark
            assert value > 0
            equity.append(value)
            exposure.append(units * mark / value)

    def dd(values):
        peak = values[0]
        worst = 0.0
        for value in values:
            peak = max(peak, value)
            worst = max(worst, 1 - value / peak)
        return worst
    benchmark = [1.0] + [float(v) / float(op[0]) for v in cl if math.isfinite(v)]
    mdd = dd([1.0] + equity)
    bhdd = dd(benchmark)
    result = {'alpha_ex': equity[-1] - benchmark[-1], 'maxdd_delta': mdd - bhdd, 'maxdd': mdd, 'bh_maxdd': bhdd, 'total_return': equity[-1] - 1, 'bh_total_return': benchmark[-1] - 1, 'turnover': turnover, 'trades': trades, 'fees_initial_equity_units': fees, 'borrow_initial_equity_units': borrow, 'mean_exposure': math.fsum(exposure) / len(exposure), 'rows': len(frame), 'close_coverage': sum((math.isfinite(v) for v in cl)) / len(frame), 'bar_coverage': float(frame.bar_available.mean()), 'intent_coverage': sum((math.isfinite(v) for v in submitted)) / len(frame)}
    return (result, planned, trace)

def mean(x):
    return math.fsum(x) / len(x)

def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    j = 0
    while j < len(v):
        k = j + 1
        while k < len(v) and v[order[k]] == v[order[j]]:
            k += 1
        for t in range(j, k):
            r[order[t]] = (j + 1 + k) / 2
        j = k
    return r

def spearman(x, y):
    a, b = (rank(x), rank(y))
    am, bm = (mean(a), mean(b))
    aa = [v - am for v in a]
    bb = [v - bm for v in b]
    den = math.sqrt(math.fsum((v * v for v in aa)) * math.fsum((v * v for v in bb)))
    return math.fsum((x * y for x, y in zip(aa, bb))) / den if den else None

def average(values):
    values=list(values)
    return math.fsum(float(v)/len(values) for v in values)

def independent_fit(prediction, actual, mask, anchor):
    ids=np.flatnonzero(mask);ys=[float(actual[t,0]) for t in ids];ps=[float(prediction[t]) for t in ids]
    assert len(ids)>=64 and all(math.isfinite(v) for v in ys+ps) and anchor==average(ys)
    d=[v-anchor for v in ps];r=[v-anchor for v in ys];B=average(v*v for v in d);C=average(x*y for x,y in zip(d,r))
    if B==0:w,case=0.,'zero_dispersion'
    elif C<=0:w,case=0.,'nonpositive_crossmoment'
    elif C>=B:w,case=1.,'upper_endpoint'
    else:w,case=C/B,'interior'
    return {'weight':w,'n':len(ids),'anchor':anchor,'mean_d':average(d),'mean_r':average(r),'innovation_secondmoment':B,'crossmoment':C,'identifiable':B>0,'weight_case':case}

def convex(full, anchor, mask, weight):
    out=np.full(len(full),np.nan)
    for t in np.flatnonzero(mask):
        p,a=float(full[t]),float(anchor[t])
        out[t]=a if weight==0 else p if weight==1 else weight*p+(1-weight)*a
    return out

def independent_scores(actual, mu, anchor, mask, fit_mean):
    ids=np.flatnonzero(mask);y=[float(actual[t,0]) for t in ids];p=[float(mu[t]) for t in ids];a=[float(anchor[t]) for t in ids]
    assert len(ids)>=16 and len(set(a))==1 and all(math.isfinite(v) for v in y+p+a)
    err=[v-q for v,q in zip(y,p)];d=[v-c for v,c in zip(p,a)];r=[v-c for v,c in zip(y,a)]
    dm,rm=average(d),average(r);dc=[v-dm for v in d];rc=[v-rm for v in r]
    mse,am=average(v*v for v in err),average(v*v for v in r);B,C=average(v*v for v in d),average(x*z for x,z in zip(d,r));V,Cov=average(v*v for v in dc),average(x*z for x,z in zip(dc,rc))
    centered=V-2*Cov;drift=dm*dm-2*dm*rm;loss=mse-am
    score={'rows':len(ids),'return_mse':mse,'return_mae':average(abs(v) for v in err),'return_sign_accuracy':sum((v>0)==(q>0) for v,q in zip(y,p))/len(ids),'zero_return_mse':average(v*v for v in y),'fit_mean_return_mse':average((v-fit_mean)**2 for v in y),'return_rank_ic':spearman(y,p)}
    decomposition={'n':len(ids),'candidate_mse':mse,'anchor_mse':am,'lossdiff':loss,'mean_d':dm,'mean_r':rm,'innovation_secondmoment':B,'crossmoment':C,'centered_variance_d':V,'centered_covariance':Cov,'centered_component':centered,'drift_component':drift,'identityresidual':loss-(centered+drift)}
    return score,decomposition

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-source-revision',required=True);parser.add_argument('--authorized-completed-run',action='store_true');args=parser.parse_args()
    if not args.authorized_completed_run or not re.fullmatch('[0-9a-f]{40}',args.expected_source_revision):raise SystemExit('Full registered commit and completed-run authorization required; no audit started.')
    root=Path.cwd();out=root/'codex_outputs/oracle_mean_reliability_decisions_v1';src=root/'codex_outputs/oracle_frozen_procedure_parity_v1';delay=root/'codex_outputs/oracle_derivative_delay_v1'
    assert (out/'results.json').is_file(),'No completed registered results; no coefficients or scores may be audited.'
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    source_script=Path('/tmp/oracle_mean_reliability_source_audit_20260906.py')
    assert sha(source_script)=='af8ae9ad8e3fc9ab50f5750727e31fbfc581cbf4807b9530e0f1af64ab37fabc'
    # Reverify all inherited artifacts and source chronology using the separately reviewed data-only scalar script.
    subprocess.run([sys.executable,str(source_script),'--root',str(root)],check=True,stdout=subprocess.DEVNULL)
    source_report=read('/tmp/oracle_mean_reliability_source_audit_20260906.json');assert source_report['status']=='pass' and source_report['verified_source_artifacts']==1328
    counts=Counter();maxima={};verified={}
    def verify(p,h):
        p=Path(p).resolve()
        if p not in verified:verified[p]=sha(p)
        assert verified[p]==h,('hash',str(p));counts['direct_hash_binding_checks']+=1
    def arr(p,keys=None):
        with np.load(p,allow_pickle=False) as z:
            if keys is not None:assert set(z.files)==set(keys),('schema',str(p),z.files)
            v={k:z[k] for k in z.files}
        assert all(a.dtype.kind in 'bifu' and not np.isinf(a).any() for a in v.values()),('invalid array',str(p))
        return v
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b);assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True),('exact',name)
    def close(name,a,b,tol=1e-10):
        if a is None or b is None:assert a is None and b is None,('undefined',name,a,b);return
        a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('shape/finite',name)
        ok=np.isfinite(a);diff=float(np.max(np.abs(a[ok]-b[ok]))) if ok.any() else 0.;maxima[name]=max(maxima.get(name,0.),diff);assert diff<=tol,('difference',name,diff,tol)
    def tree(name,a,b):
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),('keys',name)
            for k in a:tree(name,a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),('length',name)
            for x,y in zip(a,b):tree(name,x,y)
        elif isinstance(a,(bool,str)) or a is None:assert type(a)==type(b) and a==b,('atom',name,a,b)
        else:close(name,a,b)
    reg,pre,res=(read(out/(p+'.json')) for p in ('registration','preflight','results'));cfg=reg['config']
    assert reg['source_revision']==args.expected_source_revision and digest(reg)==res['registration_sha256']
    cfgpath=root/'configs/oracle_mean_reliability_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg
    verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert reg['source_bindings']==pre['source_bindings']==cfg['source_bindings'] and len(cfg['source_bindings'])==25
    for path,h in cfg['source_bindings'].items():
        verify(root/path,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+path])).hexdigest()==h
    assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
    verify(root/cfg['source_prepare_config'],cfg['source_prepare_config_sha256']);pc=yaml.safe_load((root/cfg['source_prepare_config']).read_text())
    ppre=pre['parent_prepare_preflight'];assert ppre['config_contract_sha256']==digest({k:v for k,v in pc.items() if k!='preflight_sha256'})
    assert ppre['source_bindings']==pc['source_bindings'] and ppre['source_manifest_bindings']==pc['source_manifest_bindings']
    for path,h in pc['source_manifest_bindings'].items():verify(root/path,h)
    assert pre['source_artifact_bindings']==ppre['source_artifact_bindings'] and len(pre['source_artifact_bindings'])==1328
    assert digest(pre['source_artifact_bindings'])==source_report['source_artifact_inventory_sha256']
    assert ppre['spot_data_proof']==source_report['spot_data_proof'] and ppre['um_data_proof']==source_report['um_data_proof']
    for k in ('new_weight_fitted','new_forecast_or_policy_computed','additional_test_accessed'):assert pre[k] is False
    folds=tuple(range(5,13));groups=('technical','perp_delay0');parentmeans=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half');newmeans=tuple(g+'_reliability' for g in groups);rawmeans=tuple(g+'_raw' for g in groups);means=parentmeans+rawmeans+newmeans;rules=('utility_risk1','utility_risk1_fallback_bh');segments=('scale','interval','evaluation')
    controls=('bh','common_robust')+tuple(m+'_'+r for m in parentmeans for r in rules);newids=tuple(m+'_'+r for m in newmeans for r in rules);policies=controls+newids
    assert cfg['development_folds']==list(folds) and cfg['groups']==list(groups) and cfg['mean_ids']==list(means) and cfg['new_mean_ids']==list(newmeans) and cfg['control_ids']==list(controls) and cfg['new_policy_ids']==list(newids) and cfg['segments']==list(segments) and cfg['rules']==list(rules)
    assert cfg['weight_bounds']==[0.,1.] and cfg['fit_segment']=='scale_only' and cfg['weight_objective']=='MSE_on_saved_scaled_endpoint_to_exact_scale_anchor_segment'
    assert cfg['minimum_scale_rows']==64 and cfg['minimum_score_rows']==16 and cfg['utility_risk_aversion']==1 and cfg['utility_cost_multiplier']==2
    for k in ('base_model_fitting_permitted','selection_permitted','additional_test_access_permitted','interval_width_claims_permitted'):assert cfg[k] is False
    assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['new_causal_policy_names']==4 and cfg['adaptive_prior_causal_names']==158
    rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(s['fold'],s['segment'],s['mean_id']):s for s in res['scores']};fits={(x['fold'],x['group']):x for x in res['fits']}
    assert len(rows)==len(res['rows'])==128 and set(rows)=={(f,c) for f in folds for c in policies}
    assert len(scores)==len(res['scores'])==216 and set(scores)=={(f,s,m) for f in folds for s in segments for m in means}
    assert len(fits)==len(res['fits'])==16 and set(fits)=={(f,g) for f in folds for g in groups}
    own={}
    for f in folds:
        doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
        for k in ('rows','scores','fits'):assert doc[k]==[r for r in res[k] if r['fold']==f]
        expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('weights',groups,'json'),('forecasts',newmeans,'npz'),('calibration',groups,'npz'),('targets',policies,'npz'),('traces',newids,'json')] for v in vs}
        assert set(doc['artifact_sha256'])==expected and len(expected)==26
        for p,h in doc['artifact_sha256'].items():verify(root/p,h);own[p]=h
    assert len(own)==208
    for directory,n in [('weights',16),('forecasts',16),('calibration',16),('targets',128),('traces',32)]:assert len(list((out/directory).iterdir()))==n
    assert not (out/'models').exists()
    pres=read(src/'results.json');prows={(r['fold'],r['candidate_id']):r for r in pres['rows']};pscores={(s['fold'],s['mean_id']):s for s in pres['scores']}
    fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());execution=fc['execution'];cut=pd.Timestamp(cfg['data_cutoff'])
    assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
    bars=pd.read_parquet(fc['data_path'],filters=[('bar_open_ts','<',cut)]);index=pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC');bars=bars.reindex(index);bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
    print(json.dumps({'phase':'all_sources_and_208_own_artifacts_verified'}),flush=True)
    forecast_keys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};calkeys={'timestamps','actual','scale_mask','interval_mask','raw','scaled','half','reliability','anchor'};observed_weights=[];totals=Counter()
    for f in folds:
        E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3);ix=pd.date_range(E,end,freq='15min',inclusive='left');ci=pd.date_range(S,E,freq='15min',inclusive='left');frame=bars.loc[ix]
        support=next(s for s in pre['support'] if s['fold']==f)
        for k,t in [('scale_start',S),('scale_end',I),('interval_start',I),('interval_end',E),('evaluation_start',E),('evaluation_end',end)]:assert pd.Timestamp(support[k])==t
        parent={m:arr(src/'forecasts'/f'fold{f}_{m}.npz',forecast_keys) for m in parentmeans};ref=parent['scale_mean'];inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
        assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any();exact('reference calendar',ref['timestamps'],ix.asi8);assert (ix[score]+pd.Timedelta(minutes=375)<=end).all()
        assert support['inference_rows']==int(inf.sum()) and support['score_rows']==int(score.sum()) and support['regime']==prows[f,'bh']['regime']
        for k,mask in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(mask.sum())
        caldoc=read(src/'calibration'/f'fold{f}_provenance.json');anchor=float(ref['mu'][inf][0]);assert caldoc['calibration']['scale_mean']==anchor and np.all(ref['mu'][inf]==anchor)
        evalpred={m:parent[m]['mu'] for m in parentmeans};calpred={};localfit={};calactual=cal_scale=cal_interval=None
        for g in groups:
            sourcecal=arr(src/'calibration'/f'fold{f}_{g}.npz');raw=arr(delay/'forecasts'/f'fold{f}_{g}_raw.npz');savedcal=arr(out/'calibration'/f'fold{f}_{g}.npz',calkeys)
            scale,interval=sourcecal['scale_mask'],sourcecal['interval_mask'];cm=scale|interval
            for k in ('timestamps','actual','scale_mask','interval_mask'):exact('unchanged calibration '+k,savedcal[k],sourcecal[k])
            exact('calibration calendar',savedcal['timestamps'],ci.asi8)
            for label,mask,start,stop in [('scale',scale,S,I),('interval',interval,I,E)]:assert mask.dtype==bool and mask.sum()>=64 and mask.sum()==support[label+'_rows'] and (ci[mask]>=start).all() and (ci[mask]+pd.Timedelta(minutes=375)<stop).all()
            assert not (scale&interval).any()
            if calactual is None:calactual,cal_scale,cal_interval=sourcecal['actual'],scale,interval
            else:
                exact('paired calibration actual',sourcecal['actual'],calactual);exact('paired scale',scale,cal_scale);exact('paired interval',interval,cal_interval)
            a=np.where(cm,anchor,np.nan);p=np.where(cm,sourcecal['mu']+float(caldoc['calibration']['return_bias'][g]),np.nan)
            exact('calibration raw',savedcal['raw'],np.where(cm,sourcecal['mu'],np.nan));exact('calibration scaled',savedcal['scaled'],p);exact('calibration anchor',savedcal['anchor'],a);exact('calibration half',savedcal['half'],.5*p+.5*a)
            independently=independent_fit(p,sourcecal['actual'],scale,anchor);fitrow=fits[f,g]
            assert fitrow==read(out/'weights'/f'fold{f}_{g}.json') and fitrow['fit_period']=='scale' and fitrow['interval_or_evaluation_used'] is False
            tree('scale_only_weight_record',independently,fitrow['fit']);w=independently['weight'];localfit[g]=independently
            poisoned=sourcecal['actual'].astype(object);poisoned[~scale]='IGNORED NONSCALE';poisoned[:,1:]='IGNORED OTHER OUTCOMES'
            tree('independent_weight_nonscale_invariance',independently,independent_fit(p,poisoned,scale,anchor))
            exact('calibration convex formula',savedcal['reliability'],convex(p,a,cm,w));counts['independent_scale_weights']+=1;counts['calibration_convex_forecasts']+=1
            name=g+'_reliability';full=g+'_scaled';new=arr(out/'forecasts'/f'fold{f}_{name}.npz',forecast_keys)
            for k in forecast_keys-{'mu'}:exact('new inference unchanged '+k,new[k],ref[k])
            for k in ('timestamps','actual','score_support','inference_mask','fit_return_mean'):exact('raw evaluation shared '+k,raw[k],ref[k])
            exact('evaluation raw plus saved bias',raw['mu']+float(caldoc['calibration']['return_bias'][g]),parent[full]['mu'])
            expected=convex(parent[full]['mu'],ref['mu'],inf,w);exact('evaluation convex formula',new['mu'],expected);assert np.array_equal(np.isfinite(new['mu']),inf)
            if w==0:exact('zero endpoint forecast',new['mu'],ref['mu'])
            if w==1:exact('one endpoint forecast',new['mu'],parent[full]['mu'])
            evalpred[name]=new['mu'];evalpred[g+'_raw']=raw['mu'];calpred['scale_mean']=a
            half='technical_half' if g=='technical' else 'perp_delay0_half';calpred[g+'_raw']=savedcal['raw'];calpred[full]=savedcal['scaled'];calpred[half]=savedcal['half'];calpred[name]=savedcal['reliability']
            observed_weights.append({'fold':f,'group':g,'weight':w,'case':independently['weight_case']});counts['evaluation_convex_forecasts']+=1
        for seg in segments:
            actual,mask,mu,anchors=(ref['actual'],score,evalpred,ref['mu']) if seg=='evaluation' else (calactual,cal_scale if seg=='scale' else cal_interval,calpred,calpred['scale_mean'])
            for m in means:
                sr=scores[f,seg,m];sc,dc=independent_scores(actual,mu[m],anchors,mask,float(ref['fit_return_mean']))
                assert sr['regime']==support['regime'] and sr['scale_fit_in_sample']==(seg=='scale') and sr['regime_known_at_scored_decisions']==(seg=='evaluation') and sr['regime_reference']=='evaluation_quarter_start'
                for k,v in sc.items():close('score_'+k,v,sr[k],1e-12)
                for k,v in dc.items():close('decomposition_'+k,v,sr['decomposition'][k],1e-12)
                close('decomposition_B_minus_2C_identity',dc['lossdiff'],dc['innovation_secondmoment']-2*dc['crossmoment'],1e-12)
                close('decomposition_centered_plus_drift',dc['lossdiff'],dc['centered_component']+dc['drift_component'],1e-12)
                close('decomposition_matches_MSE',dc['candidate_mse'],sr['return_mse'],1e-14)
                if seg=='evaluation' and m in parentmeans:
                    for k in sc:assert sr[k]==pscores[f,m][k]
                    counts['exact_parent_evaluation_scores']+=1
                counts['independent_return_scores']+=1;counts['independent_drift_decompositions']+=1
            for g in groups:
                m=g+'_reliability';w=localfit[g]['weight'];full=g+'_scaled';n=scores[f,seg,m]
                if w==0:assert n['return_mse']==scores[f,seg,'scale_mean']['return_mse'] and n['return_rank_ic'] is None
                elif w==1:assert n['return_mse']==scores[f,seg,full]['return_mse']
                else:close('positive_weight_fold_rank_parity',n['return_rank_ic'],scores[f,seg,full]['return_rank_ic'],1e-12)
                if seg=='scale':
                    half='technical_half' if g=='technical' else 'perp_delay0_half'
                    assert n['return_mse']<=min(scores[f,seg,x]['return_mse'] for x in ('scale_mean',full,half))+1e-14
                    counts['scale_constrained_minimum_checks']+=1
        for cid in policies:
            row=rows[f,cid];targetfile=out/'targets'/f'fold{f}_{cid}.npz';verify(targetfile,row['targets_sha256']);ta=arr(targetfile,{'timestamps','targets'});target=ta['targets'];exact('target calendar',ta['timestamps'],ix.asi8)
            assert target.shape==(len(ix),) and not (np.isfinite(target)&~clock).any()
            if cid in controls:
                original=arr(src/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('copied control targets',target,original['targets'])
                for cost in ('base','stress_2x'):assert row[cost]==prows[f,cid][cost]
                counts['exact_parent_controls']+=1
            for cost,factor in [('base',1),('stress_2x',2)]:
                contract=dict(execution);contract['one_way_cost']*=factor;contract['borrow_annual']*=factor;computed,_,_=scalar_account(frame,target,contract)
                for k,v in computed.items():close('account_'+k,v,row[cost][k])
                counts['independent_scalar_account_paths']+=1
            if cid in controls:continue
            verify(out/'traces'/f'fold{f}_{cid}.json',row['trace_sha256']);trace=read(out/'traces'/f'fold{f}_{cid}.json');assert trace['metrics']==row['base'] and trace['future_information_used_for_decisions'] is False and trace['hindsight_only'] is False and trace['teacher_actions_used'] is False and trace['canonical_replay_verified'] is True
            assert trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24
            rule=rules[1] if cid.endswith(rules[1]) else rules[0];m=cid[:-(len(rule)+1)];g=m.removesuffix('_reliability');isfallback=rule==rules[1];pred={'mu':evalpred[m],'variance':ref['variance']}
            _,expected,tr=scalar_account(frame,target,execution,pred,inf,fallback_enabled=isfallback);exact('independent own-state targets',target,expected);st=trace['decision_trace'];assert st['bar_indices']==[v[0] for v in tr]
            for col,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('trace_'+key,[v[col] for v in tr],st[key])
            allowed=learned|fallback if isfallback else learned;assert not (np.isfinite(target)&~allowed).any()
            assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum())
            if isfallback:
                assert np.all(target[fallback]==1.) and st['reasons']==[v[6] for v in tr];close('trace_targets',[v[5] for v in tr],st['targets'])
                for k,mask in [('learned',learned),('fallback',fallback),('missing_open',missing),('hold',learned&np.isnan(target))]:exact('decision mask '+k,np.asarray(trace['decision_masks'][k]),mask)
                for j,v in enumerate(tr):
                    if v[6]=='forecast_unavailable':assert st['targets'][j]==1. and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None
                counts['fallback_decisions']+=int(fallback.sum())
            w=localfit[g]['weight']
            if w in (0.,1.):
                endpoint='scale_mean' if w==0 else g+'_scaled';ep=endpoint+'_'+rule;original=arr(src/'targets'/f'fold{f}_{ep}.npz',{'timestamps','targets'});exact('observed endpoint target identity',target,original['targets'])
                for cost in ('base','stress_2x'):assert row[cost]==prows[f,ep][cost]
                counts['endpoint_policy_paths_exact']+=1
            counts['new_own_state_paths']+=1;counts['new_own_state_decisions']+=len(tr);counts['unscored_inference_decisions_retained']+=int((learned&~score).sum())
        print(json.dumps({'phase':'all_fold_coefficients_scores_accounts_verified','fold':f}),flush=True)
    assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2}
    assert counts['independent_scale_weights']==16 and counts['evaluation_convex_forecasts']==counts['calibration_convex_forecasts']==16 and counts['exact_parent_evaluation_scores']==40
    assert counts['independent_return_scores']==counts['independent_drift_decompositions']==216 and counts['scale_constrained_minimum_checks']==16
    assert counts['exact_parent_controls']==96 and counts['independent_scalar_account_paths']==256 and counts['new_own_state_paths']==32 and counts['new_own_state_decisions']==11008 and counts['fallback_decisions']==664 and counts['unscored_inference_decisions_retained']==48
    econ,pred,contrasts,direction={},{},{},{}
    for regime in ('all','bull','bear','sideways'):
        fs=[f for f in folds if regime=='all' or rows[f,'bh']['regime']['trend']==regime]
        assert len(fs)=={'all':8,'bull':2,'bear':4,'sideways':2}[regime]
        econ[regime]={cid:{'quarters':len(fs),**{cost:{k:average(rows[f,cid][cost][k] for f in fs) for k in ('alpha_ex','maxdd_delta','turnover','trades')} for cost in ('base','stress_2x')}} for cid in policies}
        pred[regime]={};contrasts[regime]={}
        for seg in segments:
            pred[regime][seg]={}
            for m in means:
                ss=[scores[f,seg,m] for f in fs];n=sum(s['rows'] for s in ss)
                pred[regime][seg][m]={'quarters':len(fs),'rows':n,'equal_quarter_mse':average(s['return_mse'] for s in ss),'pooled_row_mse':math.fsum(s['rows']*s['return_mse']/n for s in ss),'equal_quarter_mae':average(s['return_mae'] for s in ss),'zero_return_mse':average(s['zero_return_mse'] for s in ss),'fit_mean_return_mse':average(s['fit_mean_return_mse'] for s in ss),'mse_minus_zero':average(s['return_mse']-s['zero_return_mse'] for s in ss),'mse_minus_fit_mean':average(s['return_mse']-s['fit_mean_return_mse'] for s in ss),'mean_rank_ic':average(s['return_rank_ic'] for s in ss) if all(s['return_rank_ic'] is not None for s in ss) else None,'decomposition':{k:average(s['decomposition'][k] for s in ss) for k in ('lossdiff','innovation_secondmoment','crossmoment','centered_component','drift_component','identityresidual')}}
        for g in groups:
            m=g+'_reliability';half='technical_half' if g=='technical' else 'perp_delay0_half';contrasts[regime][m]={}
            for refid in ('scale_mean',g+'_scaled',half):
                per_seg={}
                for seg in segments:
                    delta=[scores[f,seg,m]['return_mse']-scores[f,seg,refid]['return_mse'] for f in fs];baseline=average(scores[f,seg,refid]['return_mse'] for f in fs)
                    per_seg[seg]={'mse_difference':average(delta),'relative_mse_reduction':-average(delta)/baseline if baseline else None,'improved_quarters':sum(v<0 for v in delta),'equal_quarters':sum(v==0 for v in delta)}
                contrasts[regime][m][refid]={'prediction':per_seg,'economics':{rule:{cost:{k:average(rows[f,m+'_'+rule][cost][k]-rows[f,refid+'_'+rule][cost][k] for f in fs) for k in ('alpha_ex','maxdd_delta','turnover','trades')} for cost in ('base','stress_2x')} for rule in rules}}
    for g in groups:
        m=g+'_reliability';predictive={seg:all(pred[r][seg][m]['mse_minus_zero']<0 and all(contrasts[r][m][refid]['prediction'][seg]['mse_difference']<0 for refid in contrasts[r][m]) for r in econ) for seg in ('interval','evaluation')}
        for rule in rules:
            cid=m+'_'+rule;direction[cid]={'economic_means_all_strata_both_costs':all(econ[r][cid][cost]['alpha_ex']>0 and econ[r][cid][cost]['maxdd_delta']<0 for r in econ for cost in ('base','stress_2x')),'predictive_mse_vs_zero_scale_full_half_all_strata':predictive,'regime_count_gate_pass':False,'high_probability_generalization_established':False}
    expected_summary={'economics':econ,'prediction':pred,'paired':contrasts,'direction':direction,'fitted_weights':[{'fold':f,'group':g,**fits[f,g]['fit']} for f in folds for g in groups],'calibration_regime_strata_are_retrospective_evaluation_quarter_groupings':True,'selection_performed':False,'new_information_established':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    tree('all_summary_scalars',expected_summary,res['summary'])
    assert res['base_models_fitted']==0 and res['calibration_weights_fitted']==16 and res['new_causal_policy_names']==4 and res['total_adaptively_explored_causal_names']==162
    for k in ('additional_test_accessed','selection_performed','teacher_use_allowed'):assert res[k] is False
    report={'status':'pass','source_revision':reg['source_revision'],'audit_script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_data_audit_sha256':sha('/tmp/oracle_mean_reliability_source_audit_20260906.json'),'ancestor_artifacts_verified':1328,'own_artifacts_verified':208,'inventory':{'weights':16,'forecasts':16,'calibration':16,'targets':128,'traces':32,'economic_rows':128,'return_and_decomposition_records':216},'counts':dict(counts),'direct_distinct_hashed_files':len(verified),'inherited_distinct_hashed_files':source_report['distinct_hashed_files'],'max_absolute_differences':maxima,'observed_weights':observed_weights,'support':dict(totals),'scope':'All registered existing source artifacts; all new output bindings; 16 weights reconstructed only from selected S labels; 16 evaluation and 16 calibration convex forecasts; 216 independent return scores and drift decompositions; 96 exact parent controls; 32 own-state decision paths and 256 base/stress scalar accounts; observed endpoint parities; all paired and aggregate summaries. No canonical planner, metrics, simulator, reliability, scoring or fitting helper imported.','limitations':['Original adaptive reused development only; interval diagnostics are not independent new holdout evidence.','Fixed-anchor endpoint shrinkage is not exactly raw-centered OLS because inherited mean and bias arithmetic differ by roundoff.','Weight zero is an exact constant-mean plateau; positive affine shrinkage does not add information or establish predictive ranking improvements.','Original retrospective common availability, absence of historical receipt proof and 2/4/2 regime coverage persist.','Audit semantic raw Spot access is filtered before original cutoff; source bytes are fully hashed. The inherited runner decodes full parquet before truncating for feature/model/scoring calculations.']}
    reportpath=Path('/tmp/oracle_mean_reliability_audit_20260906.json');reportpath.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'pass','report':str(reportpath),'counts':dict(counts),'maxima':maxima}),flush=True)

if __name__=='__main__':main()
