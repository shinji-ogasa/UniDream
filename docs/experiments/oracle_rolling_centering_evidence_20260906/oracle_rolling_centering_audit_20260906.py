"""Independent Stage14 saved-order/account audit; completion authorization mandatory.
Rolling forecast derivation and prediction-score math are separately audited.
No canonical simulator, planner, metrics or rolling helper imported.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,re,subprocess,sys
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-source-revision',required=True);parser.add_argument('--authorized-completed-run',action='store_true');args=parser.parse_args()
    if not args.authorized_completed_run or not re.fullmatch('[0-9a-f]{40}',args.expected_source_revision):raise SystemExit('Registered full commit and completed-run authorization required; nothing audited.')
    root=Path.cwd();out=root/'codex_outputs/oracle_rolling_centering_decisions_v1';parent=root/'codex_outputs/oracle_mean_reliability_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1'
    assert (out/'results.json').is_file(),'Completed result required; no forecasts or orders may be audited beforehand.'
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    source_script=Path('/tmp/oracle_rolling_centering_source_audit_20260906.py')
    assert sha(source_script)=='ef4d5a0619c13076db751adad3cc0b77eb924133e71ab597651ceb9ff803d1b9'
    source_path=Path('/tmp/oracle_rolling_centering_source_audit_20260906.json');source=read(source_path);assert source['status']=='pass' and source['total_artifacts_verified']==1536
    assert sha(source_path)=='c590d0511e910bbf1e0ecf26803f439473735ff4553ecb19a3abb172e08b607c'
    counts=Counter();maxima={};verified={}
    def verify(p,h):
        p=Path(p).resolve()
        if p not in verified:verified[p]=sha(p)
        assert verified[p]==h,('hash',str(p));counts['direct_hash_binding_checks']+=1
    def arr(p,keys=None):
        with np.load(p,allow_pickle=False) as z:
            if keys is not None:assert set(z.files)==set(keys),('schema',str(p),z.files)
            a={k:z[k] for k in z.files}
        assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid arrays',str(p))
        return a
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),('exact',name)
    def close(name,a,b,tol=1e-10):
        a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('shape/nonfinite',name)
        ok=np.isfinite(a);d=float(np.max(np.abs(a[ok]-b[ok]))) if ok.any() else 0.;maxima[name]=max(maxima.get(name,0.),d);assert d<=tol,('numeric',name,d,tol)
    for p,h in {**source['source_artifact_bindings'],**source['source_manifest_bindings']}.items():verify(root/p,h)
    reg,pre,res=(read(out/(n+'.json')) for n in ('registration','preflight','results'));cfg=reg['config'];pr,pres,pp=(read(parent/(n+'.json')) for n in ('registration','results','preflight'))
    assert reg['source_revision']==args.expected_source_revision and digest(reg)==res['registration_sha256'] and digest(pr)==pres['registration_sha256']
    cfgpath=root/'configs/oracle_rolling_centering_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg
    assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
    verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert reg['source_bindings']==cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==27
    for path,h in cfg['source_bindings'].items():
        verify(root/path,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+path])).hexdigest()==h
    verify(root/cfg['source_prepare_config'],cfg['source_prepare_config_sha256']);assert yaml.safe_load((root/cfg['source_prepare_config']).read_text())==pr['config']
    assert cfg['source_prepare_config_sha256']==pr['config_sha256']==source['stage13_config_sha256'] and cfg['parent_source_revision']==pr['source_revision']==source['stage13_source_revision']
    for key in ('registration','preflight','results'):verify(parent/(key+'.json'),cfg['parent_'+key+'_sha256'])
    assert cfg['parent_results_sha256']==source['stage13_results_sha256'] and cfg['parent_preflight_sha256']==source['stage13_preflight_sha256']
    assert pre['parent_prepare_preflight_sha256']==digest(pp) and pre['source_artifact_bindings']==source['source_artifact_bindings'] and len(pre['source_artifact_bindings'])==1536
    expected_direct={**cfg['source_bindings'],cfg['source_prepare_config']:cfg['source_prepare_config_sha256'],**{str((parent/(k+'.json')).relative_to(root)):cfg['parent_'+k+'_sha256'] for k in ('registration','preflight','results')},**{str((parent/f'fold_{f}.json').relative_to(root)):source['source_manifest_bindings'][str((parent/f'fold_{f}.json').relative_to(root))] for f in range(5,13)}}
    assert pre['direct_source_bindings']==expected_direct
    for p,h in pre['direct_source_bindings'].items():verify(root/p,h)
    for k in ('new_weight_fitted','new_forecast_or_policy_computed','additional_test_used_for_modeling_or_scoring'):assert pre[k] is False
    folds=tuple(range(5,13));groups=('technical','perp_delay0');fixedmeans=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half');oldnew=tuple(g+'_reliability' for g in groups);oldmeans=fixedmeans+tuple(g+'_raw' for g in groups)+oldnew;newmeans=('rolling_anchor','technical_rolling','perp_delay0_rolling');means=oldmeans+newmeans;rules=('utility_risk1','utility_risk1_fallback_bh')
    controls=('bh','common_robust')+tuple(m+'_'+r for m in fixedmeans+oldnew for r in rules);newids=tuple(m+'_'+r for m in newmeans for r in rules);policies=controls+newids
    assert cfg['development_folds']==list(folds) and cfg['groups']==list(groups) and cfg['mean_ids']==list(means) and cfg['new_mean_ids']==list(newmeans) and cfg['control_ids']==list(controls) and cfg['new_policy_ids']==list(newids) and cfg['rules']==list(rules)
    assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['segments']==['evaluation'] and cfg['history_calendar_months']==3 and cfg['maturity_minutes']==375
    assert cfg['minimum_history_pairs']==64 and cfg['insufficient_history_rule']=='fail_closed_preserving_every_original_inference_row' and cfg['weights']=='immutable_stage13_scale_fitted_no_updates'
    assert cfg['utility_risk_aversion']==1 and cfg['utility_cost_multiplier']==2 and cfg['adaptive_prior_causal_names']==162 and cfg['new_causal_policy_names']==6
    for k in ('new_weight_fitting_permitted','base_model_fitting_permitted','selection_permitted','additional_test_access_permitted','interval_width_claims_permitted'):assert cfg[k] is False
    rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(s['fold'],s['mean_id']):s for s in res['scores']};weights={(v['fold'],v['group']):v for v in res['fixed_weights']}
    prows={(r['fold'],r['candidate_id']):r for r in pres['rows']};pscores={(s['fold'],s['mean_id']):s for s in pres['scores'] if s['segment']=='evaluation'};pweights={(w['fold'],w['group']):w for w in pres['fits']}
    assert len(rows)==len(res['rows'])==176 and set(rows)=={(f,c) for f in folds for c in policies};assert len(scores)==len(res['scores'])==96 and set(scores)=={(f,m) for f in folds for m in means};assert len(weights)==len(res['fixed_weights'])==16 and set(weights)=={(f,g) for f in folds for g in groups} and weights==pweights
    own={}
    for f in folds:
        doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
        for k in ('rows','scores','fixed_weights'):assert doc[k]==[v for v in res[k] if v['fold']==f]
        expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('forecasts',newmeans,'npz'),('rolling_traces',('shared_history',),'json'),('targets',policies,'npz'),('traces',newids,'json')] for v in vs}
        assert set(doc['artifact_sha256'])==expected and len(expected)==32
        for path,h in doc['artifact_sha256'].items():verify(root/path,h);own[path]=h
    assert len(own)==256
    for k,n in [('forecasts',24),('rolling_traces',8),('targets',176),('traces',48)]:assert len(list((out/k).iterdir()))==n
    assert not (out/'models').exists() and not (out/'weights').exists() and not (out/'calibration').exists()
    fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());execution=fc['execution'];cut=pd.Timestamp(cfg['data_cutoff']);assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
    sp=Path(fc['data_path']);side=read(sp.with_suffix('.sha256.json'));proof=source['spot_data_proof']
    for path,key in [(sp,'artifact_sha256'),(sp.with_suffix('.sha256.json'),'sidecar_sha256'),(side['availability_path'],'availability_sha256'),(side['source_ledger_path'],'ledger_sha256')]:verify(path,proof[key])
    um=source['um_data_proof'];up=root/um['data_path']
    for path,key in [(up,'data_sha256'),(up.with_suffix('.sha256.json'),'sidecar_sha256'),(root/um['availability_path'],'availability_sha256'),(root/um['source_ledger_path'],'source_ledger_sha256'),(root/um['registration_path'],'registration_sha256')]:verify(path,um[key])
    bars=pd.read_parquet(fc['data_path'],filters=[('bar_open_ts','<',cut)]);index=pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC');bars=bars.reindex(index);bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
    forecastkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};scorekeys=('rows','return_mse','return_mae','return_sign_accuracy','zero_return_mse','fit_mean_return_mse','return_rank_ic');totals=Counter();endpoint_paths=[]
    print(json.dumps({'phase':'all_sources_and_output_bindings_verified','ancestor_artifacts':1536,'new_artifacts':256}),flush=True)
    for f in folds:
        E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);ix=pd.date_range(E,end,freq='15min',inclusive='left');frame=bars.loc[ix];ref=arr(parity/'forecasts'/f'fold{f}_scale_mean.npz',forecastkeys);exact('reference calendar',ref['timestamps'],ix.asi8)
        inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
        assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any() and frame.bar_available.iloc[0] and frame.bar_available.iloc[-1]
        assert (ix[score]+pd.Timedelta(minutes=375)<=end).all()
        for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
        supp=next(s for s in pre['support'] if s['fold']==f);ind=next(s for s in source['folds'] if s['fold']==f)
        assert supp['inference_rows']==int(inf.sum())==ind['current_inference_rows'] and supp['score_rows']==int(score.sum())==ind['current_score_rows']
        assert supp['minimum_history_pairs']==ind['minimum_history_rows'] and supp['maximum_history_pairs']==ind['maximum_history_rows'] and supp['history_counts_sha256']==digest(ind['history_row_counts'])
        assert supp['histories_with_restored_boundary_rows']==ind['decisions_with_restored_boundary_history'] and supp['restored_pairs_across_histories']==sum(ind['restored_counts'])
        assert supp['history_membership_sha256']==digest(supp['history_membership']) and supp['regime']==prows[f,'bh']['regime']
        savedpred={m:arr(out/'forecasts'/f'fold{f}_{m}.npz',forecastkeys) for m in newmeans}
        for m,p in savedpred.items():
            for k in forecastkeys-{'mu'}:exact('forecast unchanged '+k,p[k],ref[k])
            assert np.array_equal(np.isfinite(p['mu']),inf);counts['new_forecast_support_variance_verified']+=1
        history=read(out/'rolling_traces'/f'fold{f}_shared_history.json');assert history['fold']==f and history['fixed_weights']==[weights[f,g] for g in groups] and history['forecast_origin_window_calendar_months']==3 and history['maturity_minutes_inclusive']==375
        subset={p:h for p,h in pre['source_artifact_bindings'].items() if any(f'fold{f}_{g}' in p for g in groups) and ('calibration/' in p or 'forecasts/' in p or 'weights/' in p)};assert history['source_artifact_bindings']==subset
        membership=[{k:v[k] for k in ('decision_at','history_timestamp_sha256')} for v in history['decisions']];assert membership==supp['history_membership']
        assert len(history['decisions'])==int(inf.sum()) and digest([d['history_count'] for d in history['decisions']])==supp['history_counts_sha256']
        for q,t,lower,first,last,n in zip(history['decisions'],ix[inf],ind['calendar_lower_bounds_ns'],ind['history_first_tau_ns'],ind['history_last_tau_ns'],ind['history_row_counts']):
            assert pd.Timestamp(q['decision_at'])==t and q['reason']=='available' and q['minimum_pairs']==64 and q['history_count']==n
            assert pd.Timestamp(q['window_start']).value==lower and pd.Timestamp(q['window_end_exclusive'])==t and pd.Timestamp(q['maturity_limit_origin'])==t-pd.Timedelta(minutes=375)
            assert pd.Timestamp(q['oldest_origin']).value==first and pd.Timestamp(q['latest_origin']).value==last and pd.Timestamp(q['latest_maturity'])==pd.Timestamp(last,tz='UTC')+pd.Timedelta(minutes=375) and pd.Timestamp(q['latest_maturity'])<=t
            assert q['weights']=={g:weights[f,g]['fit']['weight'] for g in groups}
            pos=ix.get_loc(t)
            for m in newmeans:assert q['forecasts'][m]==float(savedpred[m]['mu'][pos])
            assert q['rolling_anchor']==q['forecasts']['rolling_anchor'];counts['rolling_trace_metadata_rows_verified']+=1
        # Deep rolling mean/label arithmetic is the separate technical auditor's scope.
        for m in means:
            sr=scores[f,m];assert sr['rows']==int(score.sum()) and sr['segment']=='evaluation' and sr['regime']==prows[f,'bh']['regime'] and sr['regime_known_at_scored_decisions'] is True and sr['decomposition_anchor']=='rolling_anchor'
            if m in oldmeans:
                for k in scorekeys:assert sr[k]==pscores[f,m][k]
                counts['exact_parent_prediction_scores']+=1
        for cid in policies:
            row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'];targetpath=out/'targets'/f'fold{f}_{cid}.npz';verify(targetpath,row['targets_sha256']);a=arr(targetpath,{'timestamps','targets'});target=a['targets'];exact('target calendar',a['timestamps'],ix.asi8)
            finite=np.isfinite(target);assert target.shape==(len(ix),) and not (finite&~clock).any() and ((target[finite]>=0)&(target[finite]<=1.12)).all()
            if cid in controls:
                old=arr(parent/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('parent control targets',target,old['targets'])
                for c in ('base','stress_2x'):assert row[c]==prows[f,cid][c]
                assert 'trace_sha256' not in row;counts['exact_parent_controls']+=1
            for cost,multiplier in [('base',1),('stress_2x',2)]:
                contract=dict(execution);contract['one_way_cost']*=multiplier;contract['borrow_annual']*=multiplier;computed,_,_=scalar_account(frame,target,contract)
                assert set(computed)==set(row[cost])
                for k,v in computed.items():close('account_'+k,v,row[cost][k])
                counts['independent_scalar_account_paths']+=1
            if cid in controls:continue
            verify(out/'traces'/f'fold{f}_{cid}.json',row['trace_sha256']);trace=read(out/'traces'/f'fold{f}_{cid}.json');assert trace['metrics']==row['base']
            assert trace['future_information_used_for_decisions'] is False and trace['hindsight_only'] is False and trace['teacher_actions_used'] is False and trace['canonical_replay_verified'] is True
            for k in ('global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed'):assert trace[k] is False
            assert trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24 and trace['decision_cadence_hours']==6 and trace['execution_delay_bars']==1
            rule=rules[1] if cid.endswith(rules[1]) else rules[0];m=cid[:-(len(rule)+1)];isfallback=rule==rules[1]
            pred={'mu':savedpred[m]['mu'],'variance':ref['variance']};_,expected,tr=scalar_account(frame,target,execution,pred,inf,fallback_enabled=isfallback);exact('independent own-state targets',target,expected)
            st=trace['decision_trace'];assert st['bar_indices']==[q[0] for q in tr]
            for col,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('trace_'+key,[q[col] for q in tr],st[key])
            allowed=learned|fallback if isfallback else learned;assert not (np.isfinite(target)&~allowed).any()
            assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum())
            if isfallback:
                assert np.all(target[fallback]==1.) and trace['fallback_decision_count']==int(fallback.sum()) and st['reasons']==[q[6] for q in tr];close('trace_targets',[q[5] for q in tr],st['targets'])
                for k,mask in [('learned',learned),('fallback',fallback),('missing_open',missing),('hold',learned&np.isnan(target))]:exact('decision mask '+k,np.asarray(trace['decision_masks'][k]),mask)
                for j,q in enumerate(tr):
                    if q[6]=='forecast_unavailable':assert st['targets'][j]==1. and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None
                counts['fallback_decisions']+=int(fallback.sum())
            if m!='rolling_anchor' and weights[f,m.removesuffix('_rolling')]['fit']['weight']==0:
                exact('zero rolling anchor forecast',savedpred[m]['mu'],savedpred['rolling_anchor']['mu']);ac='rolling_anchor_'+rule;at=arr(out/'targets'/f'fold{f}_{ac}.npz',{'timestamps','targets'});exact('zero rolling anchor targets',target,at['targets'])
                for c in ('base','stress_2x'):assert row[c]==rows[f,ac][c]
                endpoint_paths.append({'fold':f,'candidate_id':cid});counts['zero_weight_rolling_anchor_paths_exact']+=1
            counts['new_own_state_paths']+=1;counts['new_own_state_decisions']+=len(tr);counts['unscored_inference_decisions_retained']+=int((learned&~score).sum())
        print(json.dumps({'phase':'all_order_account_paths_verified','fold':f,'paths':44,'new_own_state_paths':6}),flush=True)
    assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2}
    assert counts['new_forecast_support_variance_verified']==24 and counts['rolling_trace_metadata_rows_verified']==2586 and counts['exact_parent_prediction_scores']==72
    assert counts['exact_parent_controls']==128 and counts['independent_scalar_account_paths']==352 and counts['new_own_state_paths']==48 and counts['new_own_state_decisions']==16512 and counts['fallback_decisions']==996 and counts['unscored_inference_decisions_retained']==72
    assert endpoint_paths==[{'fold':12,'candidate_id':'perp_delay0_rolling_'+r} for r in rules]
    assert res['base_models_fitted']==0 and res['calibration_weights_fitted']==0 and res['fixed_weights_copied']==16 and res['new_causal_policy_names']==6 and res['total_adaptively_explored_causal_names']==168
    for k in ('additional_test_used_for_modeling_or_scoring','selection_performed','teacher_use_allowed','high_probability_generalization_established'):assert res[k] is False
    report={'status':'pass','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'independent_data_source_audit_sha256':sha(source_path),'ancestor_artifacts_verified':1536,'own_artifacts_verified':256,'inventory':{'economic_rows':176,'prediction_score_records':96,'new_forecasts':24,'shared_history_traces':8,'targets':176,'utility_traces':48},'counts':dict(counts),'direct_distinct_hashed_files':len(verified),'source_distinct_hashed_files':source['distinct_hashed_files'],'max_absolute_differences':maxima,'support':dict(totals),'zero_weight_endpoint_paths':endpoint_paths,'scope':'All immutable source and output bindings; 24 new forecast calendars/support/shared variance; 2586 rolling trace metadata records against preflight and independent data-only history bounds/counts; 72 exact old prediction scores; 128 exact parent controls; all48 own-state policy paths and352 base/stress scalar accounts; both observed weight-zero endpoint paths. No canonical metrics, simulator, planner or rolling helper imported.','separate_audit_scope':['Deep rolling history/label/forecast arithmetic is independently audited by the technical agent.','New prediction-score/decomposition and summary arithmetic is independently audited by the primary statistics agent.'],'limitations':['All strategy conclusions remain adaptive reused development diagnostics, with unchanged retrospective availability and 2/4/2 regime coverage.','Future labels are used only after event-time maturity; this does not establish historical receipt-time availability.','Audit raw semantic reads are filtered before original cutoff, while source bytes are fully hashed; inherited runner decodes full parquet before truncating research calculations.','Same base intents are reused for stress costs; no separate stress optimization or model fitting.']}
    reportpath=Path('/tmp/oracle_rolling_centering_audit_20260906.json');reportpath.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'pass','report':str(reportpath),'counts':dict(counts),'maxima':maxima}),flush=True)

if __name__=='__main__':main()
