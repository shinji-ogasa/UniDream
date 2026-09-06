"""Independent Stage15 immutable-source, order, own-state and cash/units audit.
Registered completion authorization required. No canonical helper is imported;
new fit/forecast derivation and score/summary math are separate auditors' scope.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,re,subprocess
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

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--expected-source-revision',required=True);ap.add_argument('--authorized-completed-run',action='store_true');args=ap.parse_args()
    if not args.authorized_completed_run or args.expected_source_revision!='d11cfee15b77f773e353c7ecb1ba4729c0b4abe7':raise SystemExit('Exact registered revision and completed-run authorization required.')
    root=Path.cwd();out=root/'codex_outputs/oracle_short_feature_decisions_v1';parent=root/'codex_outputs/oracle_rolling_centering_decisions_v1';fallback_root=root/'codex_outputs/oracle_fallback_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1';delay=root/'codex_outputs/oracle_derivative_delay_v1'
    assert (out/'results.json').is_file(),'Completed result required before any policy arithmetic.'
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    counts=Counter();maxima={};verified={}
    def verify(p,h):
        p=Path(p).resolve()
        if p not in verified:verified[p]=sha(p)
        assert verified[p]==h,('hash',str(p));counts['hash_binding_checks']+=1
    def arr(p,keys=None):
        with np.load(p,allow_pickle=False) as z:
            if keys is not None:assert set(z.files)==set(keys),('schema',str(p),z.files)
            a={k:z[k] for k in z.files}
        assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid array',str(p))
        return a
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),('exact',name)
    def close(name,a,b,tol=1e-10):
        a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('invalid',name)
        ok=np.isfinite(a);e=float(np.max(np.abs(a[ok]-b[ok]))) if ok.any() else 0.;maxima[name]=max(maxima.get(name,0.),e);assert e<=tol,('numerical',name,e,tol)
    sourcepath=Path('/tmp/oracle_short_feature_source_audit_20260906.json');verify(sourcepath,'7cf090b74d13df1e74ba66dbd0d6fc8188d2da917693d53ee1d72ab903ea91ff');source=read(sourcepath)
    verify('/tmp/oracle_short_feature_source_audit_20260906.py','d59a62b6a46e37fd2b3f78bf12aa6fabe8d21796386d2cab16a869414b9877a4');assert source['passed'] and source['checks']['ancestor_artifacts']==1792
    for p,h in {**source['source_artifact_bindings'],**source['direct_stage14_fold_bindings']}.items():verify(root/p,h)
    reg,pre,res=(read(out/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config']
    verify(out/'results.json','2f65382455e1266e035d38007a8dc939efb74d84df676c44da8ed3de9cd77218');verify(out/'registration.json','43ca3c37e272168a73a26b370c04b98ef394fada8f91d819d8765f453d629682');assert digest(reg)=='b363840afa4a6b3b6c263369c316fa8596141669e8fa82bbf84c0df4970275ee'
    assert reg['source_revision']==args.expected_source_revision and res['registration_sha256']==digest(reg)
    cfgpath=root/'configs/oracle_short_feature_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg
    assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
    verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert reg['source_bindings']==cfg['source_bindings']==pre['source_bindings']
    for p,h in cfg['source_bindings'].items():
        verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+p])).hexdigest()==h
    assert pre['source_artifact_bindings']==source['source_artifact_bindings']
    for p,h in pre['direct_source_bindings'].items():verify(root/p,h)
    for k in ('new_model_fitted','new_forecast_or_policy_computed','additional_test_used_for_modeling_or_scoring'):assert pre[k] is False
    pr,pres,pp=(read(parent/(k+'.json')) for k in ('registration','results','preflight'))
    assert digest(pr)==pres['registration_sha256'] and cfg['parent_source_revision']==pr['source_revision']==source['source_revision']
    for key in ('registration','preflight','results'):verify(parent/(key+'.json'),cfg['parent_'+key+'_sha256'])
    assert pre['parent_preflight_sha256']==digest(pp)
    fr,fg=(read(fallback_root/(k+'.json')) for k in ('results','registration'));assert fr['registration_sha256']==digest(fg)
    folds=tuple(range(5,13));rules=('utility_risk1','utility_risk1_fallback_bh');groups=('technical','technical_short_price','technical_short_flow','technical_short_both')
    fixed=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half');reliability=('technical_reliability','perp_delay0_reliability');rolling=('rolling_anchor','technical_rolling','perp_delay0_rolling')
    oldmeans=fixed+('technical_raw','perp_delay0_raw')+reliability+rolling
    oldcontrols=('bh','common_robust')+tuple(m+'_'+r for m in fixed+reliability+rolling for r in rules)
    simple=('zero','fit_mean','technical_raw');extras=tuple(m+'_'+r for m in simple for r in rules);controls=oldcontrols+extras;newmeans=tuple(g+'_raw' for g in groups[1:]);newids=tuple(m+'_'+r for m in newmeans for r in rules);policies=controls+newids
    scoremeans={'interval':simple+newmeans,'evaluation':oldmeans+('zero','fit_mean')+newmeans}
    assert len(controls)==28 and len(policies)==34 and cfg['control_ids']==list(controls) and cfg['new_policy_ids']==list(newids) and cfg['groups']==list(groups)
    assert cfg['score_means']=={k:list(v) for k,v in scoremeans.items()} and cfg['return_calibration']=='none' and cfg['risk_source']=='unchanged_technical_scaled'
    assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['ridge_alpha']==100 and cfg['threadpool_limit']==2 and cfg['adaptive_prior_causal_names']==168
    for k in ('risk_fitting_permitted','weight_fitting_permitted','selection_permitted','additional_test_access_permitted','interval_width_claims_permitted'):assert cfg[k] is False
    rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(s['fold'],s['segment'],s['mean_id']):s for s in res['scores']};baseline={v['fold']:v for v in res['baseline_parity']}
    prows={(r['fold'],r['candidate_id']):r for r in pres['rows']};frows={(r['fold'],r['candidate_id']):r for r in fr['rows']};pscores={(s['fold'],s['mean_id']):s for s in pres['scores']}
    assert len(rows)==len(res['rows'])==272 and set(rows)=={(f,c) for f in folds for c in policies}
    assert len(scores)==len(res['scores'])==184 and set(scores)=={(f,s,m) for f in folds for s in scoremeans for m in scoremeans[s]}
    assert len(baseline)==len(res['baseline_parity'])==8 and set(baseline)==set(folds)
    own={}
    for f in folds:
        doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
        for k in ('rows','scores'):assert doc[k]==[v for v in res[k] if v['fold']==f]
        assert doc['baseline_parity']==baseline[f]
        expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('models',groups,'joblib'),('calibration',groups,'npz'),('forecasts',tuple(g+'_raw' for g in groups),'npz'),('provenance',('fit',),'json'),('targets',policies,'npz'),('traces',newids,'json')] for v in vs}
        assert set(doc['artifact_sha256'])==expected and len(expected)==53
        for p,h in doc['artifact_sha256'].items():verify(root/p,h);own[p]=h
    assert len(own)==424
    for k,n in [('models',32),('calibration',32),('forecasts',32),('provenance',8),('targets',272),('traces',48)]:assert len(list((out/k).iterdir()))==n
    fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());execution=fc['execution'];cut=pd.Timestamp(cfg['data_cutoff']);assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
    sp=Path(fc['data_path']);side=read(sp.with_suffix('.sha256.json'));proof=read(delay/'preflight.json')
    for p,k in [(sp,'artifact_sha256'),(sp.with_suffix('.sha256.json'),'sidecar_sha256'),(side['availability_path'],'availability_sha256'),(side['source_ledger_path'],'ledger_sha256')]:verify(p,proof['spot_data_proof'][k])
    assert proof['spot_data_proof']['artifact_sha256']=='5e20e81e86f76b95d1301be7a8a366aa9ad78134f954ec8c9dbf83c0db1acf69'
    um=proof['um_data_proof'];up=root/um['data_path']
    for p,k in [(up,'data_sha256'),(up.with_suffix('.sha256.json'),'sidecar_sha256'),(root/um['availability_path'],'availability_sha256'),(root/um['source_ledger_path'],'source_ledger_sha256'),(root/um['registration_path'],'registration_sha256')]:verify(p,um[k])
    bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC'));bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
    fkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};ckeys={'timestamps','mu','actual','scale_mask','interval_mask'};scorekeys=('rows','return_mse','return_mae','return_sign_accuracy','zero_return_mse','fit_mean_return_mse','return_rank_ic');totals=Counter()
    print(json.dumps({'phase':'immutable_bindings_passed','ancestor_artifacts':1792,'own_artifacts':424}),flush=True)
    for f in folds:
        E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);ix=pd.date_range(E,end,freq='15min',inclusive='left');frame=bars.loc[ix];ref=arr(parity/'forecasts'/f'fold{f}_scale_mean.npz',fkeys);exact('forecast calendar',ref['timestamps'],ix.asi8)
        inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
        assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any() and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
        for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
        supp=next(v for v in pre['support'] if v['fold']==f);ind=next(v for v in source['support'] if v['fold']==f)
        assert supp['counts']==ind['counts'] and supp['mask_sha256']==ind['mask_sha256'] and supp['counts']['inference']==int(inf.sum()) and supp['counts']['score']==int(score.sum())
        saved={g:arr(out/'forecasts'/f'fold{f}_{g}_raw.npz',fkeys) for g in groups}
        oldcal=arr(parity/'calibration'/f'fold{f}_technical.npz');oldraw=arr(delay/'forecasts'/f'fold{f}_technical_raw.npz')
        for g,p in saved.items():
            for k in fkeys-{'mu'}:exact('shared forecast '+k,p[k],ref[k])
            exact('forecast finite support',np.isfinite(p['mu']),inf);counts['forecast_support_and_shared_variance']+=1
            ca=arr(out/'calibration'/f'fold{f}_{g}.npz',ckeys)
            for k in ckeys-{'mu'}:exact('same calibration '+k,ca[k],oldcal[k])
            exact('same calibration raw availability',np.isfinite(ca['mu']),np.isfinite(oldcal['mu']));counts['calibration_support_and_actual']+=1
            if g=='technical':
                close('baseline_raw_calibration',ca['mu'],oldcal['mu'],1e-14);close('baseline_raw_evaluation',p['mu'],oldraw['mu'],1e-14)
        bp=baseline[f];assert all(v==0 for v in bp['model_state'].values()) and bp['calibration_raw_maxdiff']==0 and bp['evaluation_raw_maxdiff']==0
        prov=read(out/'provenance'/f'fold{f}_fit.json');assert prov['fold']==f and prov['baseline_parity']==bp and prov['fit_source_binding']==supp
        fitp=prov['fit_provenance'];assert fitp['risk_or_calibration_fitted'] is False and fitp['evaluation_labels_used'] is False and fitp['model_selection_performed'] is False
        assert fitp['feature_columns']==supp['feature_columns'] and fitp['mask_counts']=={k:supp['counts'][k] for k in ('fit','predict')}
        for seg,means in scoremeans.items():
            for m in means:
                s=scores[f,seg,m];n=int(score.sum()) if seg=='evaluation' else int(oldcal['interval_mask'].sum())
                assert s['rows']==n and s['regime']==prows[f,'bh']['regime'] and s['regime_known_at_scored_decisions'] is (seg=='evaluation')
                if seg=='evaluation' and m in oldmeans:
                    for k in scorekeys:assert s[k]==pscores[f,m][k]
                    counts['exact_old_evaluation_scores']+=1
        for cid in policies:
            row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'];p=out/'targets'/f'fold{f}_{cid}.npz';verify(p,row['targets_sha256']);a=arr(p,{'timestamps','targets'});target=a['targets'];exact('target calendar',a['timestamps'],ix.asi8)
            finite=np.isfinite(target);assert target.shape==(len(ix),) and not (finite&~clock).any() and ((target[finite]>=0)&(target[finite]<=1.12)).all()
            if cid in controls:
                sr=fallback_root if cid in extras else parent;old=arr(sr/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('parent targets',target,old['targets']);parentrow=frows[f,cid] if cid in extras else prows[f,cid]
                for c in ('base','stress_2x'):assert row[c]==parentrow[c]
                assert 'trace_sha256' not in row;counts['exact_old_controls']+=1
            for cost,mul in [('base',1),('stress_2x',2)]:
                c=dict(execution);c['one_way_cost']*=mul;c['borrow_annual']*=mul;got,_,_=scalar_account(frame,target,c);assert set(got)==set(row[cost])
                for k,v in got.items():close('account_'+k,v,row[cost][k])
                counts['independent_scalar_accounts']+=1
            if cid not in newids and not cid.startswith('technical_raw_'):continue
            rule=rules[1] if cid.endswith(rules[1]) else rules[0];isfallback=rule==rules[1];g=cid[:-(len(rule)+1)].removesuffix('_raw');pred={'mu':saved[g]['mu'],'variance':ref['variance']}
            _,expected,tr=scalar_account(frame,target,execution,pred,inf,fallback_enabled=isfallback);exact('independent own-state targets',target,expected)
            allowed=learned|fallback if isfallback else learned;assert not (finite&~allowed).any()
            if isfallback:assert np.all(target[fallback]==1.)
            label='new' if cid in newids else 'baseline';counts[label+'_own_state_paths']+=1;counts[label+'_own_state_decisions']+=len(tr)
            if cid not in newids:continue
            tp=out/'traces'/f'fold{f}_{cid}.json';verify(tp,row['trace_sha256']);trace=read(tp);assert trace['metrics']==row['base']
            for k in ('future_information_used_for_decisions','hindsight_only','teacher_actions_used','global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed'):assert trace[k] is False
            assert trace['canonical_replay_verified'] is True and trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24 and trace['decision_cadence_hours']==6 and trace['execution_delay_bars']==1
            st=trace['decision_trace'];assert st['bar_indices']==[q[0] for q in tr]
            for col,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('trace_'+key,[q[col] for q in tr],st[key])
            assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum())
            if isfallback:
                assert trace['fallback_decision_count']==int(fallback.sum()) and st['reasons']==[q[6] for q in tr];close('trace_targets',[q[5] for q in tr],st['targets'])
                for k,m in [('learned',learned),('fallback',fallback),('missing_open',missing),('hold',learned&np.isnan(target))]:exact('decision mask '+k,np.asarray(trace['decision_masks'][k]),m)
                for j,q in enumerate(tr):
                    if q[6]=='forecast_unavailable':assert st['targets'][j]==1. and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None
                counts['new_fallback_decisions']+=int(fallback.sum())
            counts['new_unscored_inference_decisions_retained']+=int((learned&~score).sum())
        print(json.dumps({'phase':'independent_paths_verified','fold':f,'accounts':68,'new_paths':6,'baseline_paths':2}),flush=True)
    assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2}
    expectedcounts={'forecast_support_and_shared_variance':32,'calibration_support_and_actual':32,'exact_old_evaluation_scores':96,'exact_old_controls':224,'independent_scalar_accounts':544,'new_own_state_paths':48,'new_own_state_decisions':16512,'baseline_own_state_paths':16,'baseline_own_state_decisions':5504,'new_fallback_decisions':996,'new_unscored_inference_decisions_retained':72}
    for k,n in expectedcounts.items():assert counts[k]==n,(k,counts[k],n)
    for k,n in {'return_model_fits':32,'new_return_model_fits':24,'baseline_parity_fits':8,'risk_model_fits':0,'calibration_weight_fits':0,'new_causal_policy_names':6,'total_adaptively_explored_causal_names':174}.items():assert res[k]==n
    for k in ('additional_test_used_for_modeling_or_scoring','selection_performed','teacher_use_allowed','high_probability_generalization_established'):assert res[k] is False
    assert res['summary']['regime_counts']=={'bull':2,'bear':4,'sideways':2} and res['summary']['regime_count_gate_pass'] is False and res['summary']['selection_performed'] is False
    report={'status':'pass','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'data_source_audit_sha256':sha(sourcepath),'ancestor_artifacts_verified':1792,'own_artifacts_verified':424,'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maxima,'support':dict(totals),'inventory':{'models':32,'calibration_arrays':32,'forecasts':32,'provenance':8,'targets':272,'utility_traces':48,'economic_rows':272,'score_records':184},'scope':'All source and own artifact bindings; shared forecast/calibration masks, raw technical prediction parity; 96 exact old E scores; 224 exact controls; all544 independent scalar accounts; all48 new and16 baseline own-state paths. No canonical simulator/planner/metrics/fitter/features helper imported.','separate_audits':['Independent raw model fitting/coefficients/prediction formula verification is the technical audit scope.','New forecast loss and summary arithmetic is the statistical audit scope.'],'limitations':['Repeated development folds and retrospective common masks; regime gate remains unmet at 2/4/2.','Historical event times do not establish historical receipts.','Spot semantic reads filtered below cutoff; full source files are hashed.','Base-selected intentions are replayed unchanged under stress costs.']}
    path=Path('/tmp/oracle_short_feature_audit_20260906.json');path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':'pass','report':str(path),'counts':dict(counts),'maxima':maxima}),flush=True)
if __name__=='__main__':main()
