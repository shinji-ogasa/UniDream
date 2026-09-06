"""Independent audit of a completed registered information diagnostic; no canonical imports.
No model fitting, search, source mutation, or additional-period semantic data access.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, re, subprocess
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-source-revision',required=True)
    parser.add_argument('--authorized-completed-run',action='store_true')
    args=parser.parse_args()
    if not args.authorized_completed_run or not re.fullmatch('[0-9a-f]{40}',args.expected_source_revision):
        raise SystemExit('Explicit completed-run authorization and full registered commit required')
    root=Path.cwd();out=root/'codex_outputs/oracle_information_decomposition_v1'
    assert (out/'results.json').is_file(),'No completed result; no replay permitted'
    counts=Counter();maxima={};verified={}
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def verify(p,h):
        p=Path(p).resolve()
        if p not in verified:verified[p]=sha(p)
        assert verified[p]==h,('hash',str(p),h,verified[p]);counts['hash_binding_checks']+=1
    def arr(p,keys):
        with np.load(p,allow_pickle=False) as z:
            assert set(z.files)==set(keys),('schema',str(p),z.files)
            a={k:z[k] for k in z.files}
        assert all(v.dtype.kind in 'bifu' for v in a.values()),('numeric arrays',str(p))
        assert all(not np.isinf(v).any() for v in a.values()),('infinite array',str(p))
        return a
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),('exact',name)
    def close(name,a,b,tol=1e-10):
        a,b=np.asarray(a,float),np.asarray(b,float)
        assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('shape/finite',name)
        finite=np.isfinite(a);d=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        maxima[name]=max(maxima.get(name,0.),d)
        assert d<=tol,('numeric',name,d,tol)
    def tree(name,a,b):
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),('tree keys',name)
            for k in a:tree(name+'.'+k,a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),('tree length',name)
            for i,(x,y) in enumerate(zip(a,b)):tree(name+'.'+str(i),x,y)
        elif isinstance(a,(str,bool)) or a is None:assert type(a)==type(b) and a==b,('tree atom',name)
        else:close(name,a,b)
    folds=tuple(range(5,13));means=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')
    halves=('technical_half','perp_delay0_half');swaps=('return','realized_risk','both');rules=('utility_risk1','utility_risk1_fallback_bh')
    controls=('bh','common_robust')+tuple(m+'_'+r for m in means for r in rules)
    hybrids=tuple(m+'_oracle_'+s for m in halves for s in swaps)
    hybridids=tuple(m+'_'+r for m in hybrids for r in rules)
    rlids=tuple(f'matched_rl_beam32_{r}_risk{p}' for r in ('hold','fallback_bh') for p in (0,1))
    policies=controls+hybridids+rlids
    reg,res,pre=(read(out/(p+'.json')) for p in ('registration','results','preflight'));cfg=reg['config'];src=root/cfg['source_root']
    pr,pres,pp=(read(src/(p+'.json')) for p in ('registration','results','preflight'))
    assert reg['source_revision']==args.expected_source_revision and digest(reg)==res['registration_sha256']
    assert digest(pr)==pres['registration_sha256'] and pres['parity_pass'] is True
    config_path=root/'configs/oracle_information_decomposition_20260906.yaml'
    verify(config_path,reg['config_sha256']);assert yaml.safe_load(config_path.read_text())==cfg
    verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256']
    assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert cfg['control_ids']==list(controls) and cfg['diagnostic_ids']==list(hybridids+rlids)
    assert cfg['half_means']==list(halves) and cfg['swaps']==list(swaps) and cfg['rules']==list(rules)
    assert cfg['utility_risk_aversion']==1 and cfg['utility_cost_multiplier']==2 and cfg['horizon_bars']==24 and cfg['rl_beam_width']==32 and cfg['rl_risk_penalties']==[0.,1.] and cfg['rl_missing_input_rules']==['hold','fallback_bh']
    assert cfg['replacement_support']=='existing_saved_score_support_only_keep_learned_elsewhere'
    assert cfg['development_validation_folds']==list(folds) and cfg['data_cutoff']=='2023-04-16T13:45:00Z'
    for k in ('new_model_fitting_permitted','selection_permitted','teacher_use_allowed','test_or_additional_periods_permitted'):assert cfg[k] is False
    assert cfg['new_causal_candidate_count']==0
    assert reg['source_bindings']==pre['source_bindings']==cfg['source_bindings']
    assert pre['source_manifest_bindings']==cfg['source_manifest_bindings']
    assert len(cfg['source_bindings'])==23 and len(cfg['source_manifest_bindings'])==11
    for path,h in cfg['source_bindings'].items():
        verify(root/path,h)
        blob=subprocess.check_output(['git','show',reg['source_revision']+':'+path]);assert hashlib.sha256(blob).hexdigest()==h,('registered git source',path)
    assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(config_path.relative_to(root))])).hexdigest()==reg['config_sha256']
    for path,h in cfg['source_manifest_bindings'].items():verify(root/path,h)
    verify(root/cfg['parity_config'],cfg['parity_config_sha256']);assert cfg['parity_config_sha256']==pr['config_sha256']
    assert yaml.safe_load((root/cfg['parity_config']).read_text())==pr['config']
    verify(src/'preflight.json',pr['preflight_sha256']);assert pp['config_contract_sha256']==digest({k:v for k,v in pr['config'].items() if k!='preflight_sha256'})
    for path,h in pr['config']['metadata_bindings'].items():verify(root/path,h)
    for name,h in pr['source_sha256'].items():verify(root/'unidream/experiments'/name,h)
    ancestors=dict(pp['source_artifact_sha256']);assert len(ancestors)==1064
    for f in folds:
        pf=read(src/f'fold_{f}.json');assert pf['registration_sha256']==digest(pr) and len(pf['artifact_sha256'])==33
        assert pf['rows']==[r for r in pres['rows'] if r['fold']==f]
        for path,h in pf['artifact_sha256'].items():
            if path in ancestors:assert ancestors[path]==h
            ancestors[path]=h
    assert len(ancestors)==1328 and ancestors==pre['source_artifact_bindings']
    for path,h in ancestors.items():verify(root/path,h)
    fcpath=root/'configs/oracle_frontier_20260905.yaml';fc=yaml.safe_load(fcpath.read_text());assert fc['data_cutoff']==cfg['data_cutoff']
    execution=fc['execution'];assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
    sp=Path(fc['data_path']);side=read(sp.with_suffix('.sha256.json'));proof=pre['spot_data_proof'];assert proof==pp['spot_data_proof']
    for path,key in [(sp,'artifact_sha256'),(sp.with_suffix('.sha256.json'),'sidecar_sha256'),(side['availability_path'],'availability_sha256'),(side['source_ledger_path'],'ledger_sha256')]:verify(path,proof[key])
    um=pre['um_data_proof'];assert um==pp['um_data_proof'];up=root/um['data_path']
    for path,key in [(up,'data_sha256'),(up.with_suffix('.sha256.json'),'sidecar_sha256'),(root/um['availability_path'],'availability_sha256'),(root/um['source_ledger_path'],'source_ledger_sha256'),(root/um['registration_path'],'registration_sha256')]:verify(path,um[key])
    rows={(r['fold'],r['candidate_id']):r for r in res['rows']};prows={(r['fold'],r['candidate_id']):r for r in pres['rows']}
    assert len(rows)==len(res['rows'])==224 and set(rows)=={(f,c) for f in folds for c in policies}
    assert len(prows)==len(pres['rows'])==96 and set(prows)=={(f,c) for f in folds for c in controls}
    own={}
    for f in folds:
        fold=read(out/f'fold_{f}.json');assert fold['registration_sha256']==digest(reg) and fold['rows']==[r for r in res['rows'] if r['fold']==f]
        expected={str((out/k/f'fold{f}_{c}.{ext}').relative_to(root)) for k,ids,ext in [('targets',policies,'npz'),('forecasts',hybrids,'npz'),('traces',hybridids+rlids,'json')] for c in ids}
        assert set(fold['artifact_sha256'])==expected and len(expected)==50
        for path,h in fold['artifact_sha256'].items():verify(root/path,h);own[path]=h
    assert len(own)==400
    for k,n in [('targets',224),('forecasts',48),('traces',128)]:assert len(list((out/k).iterdir()))==n
    assert not (out/'models').exists() and not (out/'calibration').exists()
    for (f,c),r in rows.items():
        assert r['hindsight_only']==(c in hybridids+rlids);assert r['regime']==prows[f,'bh']['regime']
        verify(out/'targets'/f'fold{f}_{c}.npz',r['targets_sha256'])
        if c in hybridids+rlids:verify(out/'traces'/f'fold{f}_{c}.json',r['diagnostic_sha256'])
        else:
            assert 'diagnostic_sha256' not in r
            for cost in ('base','stress_2x'):assert r[cost]==prows[f,c][cost]
            counts['exact_parent_control_rows']+=1
    print(json.dumps({'phase':'source_and_output_bindings_verified','distinct_files':len(verified),'ancestors':len(ancestors),'own_artifacts':len(own)}),flush=True)
    cut=pd.Timestamp(cfg['data_cutoff']);bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)])
    assert bars.index.is_monotonic_increasing and bars.index.is_unique and bars.index.max()<cut
    index=pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC');bars=bars.reindex(index);bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1)
    npzkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'}
    totals=Counter();counts_by_fold=[]
    for support in pre['support']:
        f=support['fold'];start=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=start+pd.DateOffset(months=3)
        assert pd.Timestamp(support['evaluation_start'])==start and pd.Timestamp(support['evaluation_end'])==end and end<=cut
        ix=pd.date_range(start,end,freq='15min',inclusive='left');frame=bars.loc[ix];assert frame.bar_available.iloc[0] and frame.bar_available.iloc[-1]
        parent={m:arr(src/'forecasts'/f'fold{f}_{m}.npz',npzkeys) for m in means};ref=parent['technical_half'];inf,score=ref['inference_mask'],ref['score_support']
        clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
        assert inf.dtype==bool and score.dtype==bool and not (inf&~clock).any() and not (score&~inf).any()
        assert support['inference_rows']==int(inf.sum()) and support['replacement_rows']==int(score.sum()) and support['learned_remainder_rows']==int((inf&~score).sum()) and support['current_open_missing_inference_rows']==int((inf&~known).sum())
        assert support['regime']==prows[f,'bh']['regime'];assert (ix[score]+pd.Timedelta(minutes=375)<=end).all()
        expected_actual=np.full((len(ix),3),np.nan)
        # Independently construct only registered scored h24 labels; no tail or gap labels are consumed.
        for t in np.flatnonzero(score):
            entry=float(frame.open.iloc[t+1]);cl=[float(x) for x in frame.close.iloc[t+1:t+25]]
            assert len(cl)==24 and math.isfinite(entry) and all(math.isfinite(x) for x in cl) and frame.bar_available.iloc[t+1:t+25].all()
            terms=[math.log(cl[0]/entry)**2]+[math.log(cl[j]/cl[j-1])**2 for j in range(1,24)]
            expected_actual[t]=[math.log(cl[-1]/entry),max(-math.log(min(cl)/entry),0),math.sqrt(math.fsum(terms))]
        for m,p in parent.items():
            exact('parent timestamps',p['timestamps'],ix.asi8)
            for k in ('inference_mask','score_support','actual','variance','fit_return_mean'):exact('shared parent '+k,p[k],ref[k])
            assert np.array_equal(np.isfinite(p['mu']),inf) and np.array_equal(np.isfinite(p['variance']),inf) and (p['variance'][inf]>=0).all() and np.isnan(p['actual'][~score]).all()
            close('parent_actual_scalar_h24',p['actual'],expected_actual,1e-12);counts['parent_forecasts_verified']+=1
        localpred={}
        for half in halves:
            for swap in swaps:
                h=half+'_oracle_'+swap;p=arr(out/'forecasts'/f'fold{f}_{h}.npz',npzkeys);original=parent[half]
                for k in npzkeys-{'mu','variance'}:exact('unchanged hybrid '+k,p[k],original[k])
                for k in ('mu','variance'):exact('unchanged nonreplacement '+k,p[k][~score],original[k][~score])
                expected_mu=original['actual'][score,0] if swap in ('return','both') else original['mu'][score]
                expected_variance=original['actual'][score,2]**2 if swap in ('realized_risk','both') else original['variance'][score]
                exact('score mu substitution',p['mu'][score],expected_mu);exact('score risk substitution',p['variance'][score],expected_variance)
                assert np.array_equal(np.isfinite(p['mu']),inf) and np.array_equal(np.isfinite(p['variance']),inf)
                localpred[h]=p;counts['score_only_hybrid_forecasts_verified']+=1
        fold_targets={};base_inventory={}
        for cid in policies:
            row=rows[f,cid];saved=arr(out/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});target=saved['targets'];exact('target timestamps',saved['timestamps'],ix.asi8)
            assert target.shape==(len(ix),) and not (np.isfinite(target)&~clock).any() and ((target[np.isfinite(target)]>=0)&(target[np.isfinite(target)]<=1.12)).all()
            if cid in controls:
                old=arr(src/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('control target parity',target,old['targets']);counts['exact_parent_control_targets']+=1
            for cost,factor in [('base',1),('stress_2x',2)]:
                contract=dict(execution);contract['one_way_cost']*=factor;contract['borrow_annual']*=factor
                inv=[] if cid in rlids and cost=='base' else None
                actual,_,_=scalar_account(frame,target,contract,inventory_trace=inv)
                assert set(actual)==set(row[cost])
                for key,v in actual.items():close('account_'+key,v,row[cost][key])
                if inv is not None:base_inventory[cid]=inv
                counts['independent_scalar_account_paths']+=1
            fold_targets[cid]=target
        for half in halves:
            for swap in swaps:
                h=half+'_oracle_'+swap;p=localpred[h]
                for rule in rules:
                    cid=h+'_'+rule;target=fold_targets[cid];isfallback=rule.endswith('fallback_bh');row=rows[f,cid]
                    trace=read(out/'traces'/f'fold{f}_{cid}.json');assert trace['metrics']==row['base']
                    assert trace['diagnostic_kind']=='hindsight_hybrid_conditional_utility_planner' and trace['information_swap']==swap
                    assert trace['future_information_used_for_decisions'] is True and trace['hindsight_only'] is True and trace['deployable'] is False and trace['teacher_use_allowed'] is False
                    for k in ('teacher_actions_used','global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed','realized_risk_is_conditional_variance'):assert trace[k] is False
                    assert trace['canonical_replay_verified'] is True and trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24
                    meta=trace['information_intervention'];assert meta['swap']==swap and meta['hindsight_only'] is True and meta['future_information_used_for_decisions'] is True and meta['deployable'] is False and meta['teacher_use_allowed'] is False and meta['global_optimum_claimed'] is False
                    assert meta['inference_rows']==int(inf.sum()) and meta['replacement_rows']==int(score.sum()) and meta['learned_remainder_rows']==int((inf&~score).sum()) and meta['inference_and_missing_action_support_unchanged'] is True
                    _,planned,tr=scalar_account(frame,target,execution,p,inf,fallback_enabled=isfallback);close('hybrid_own_state_targets',target,planned)
                    st=trace['decision_trace'];assert st['bar_indices']==[v[0] for v in tr];assert st['hindsight_information_replaced']==[bool(score[v[0]]) for v in tr]
                    for column,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('hybrid_trace_'+key,[v[column] for v in tr],st[key])
                    allowed=learned|fallback if isfallback else learned;assert not (np.isfinite(target)&~allowed).any()
                    assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum())
                    if isfallback:
                        assert np.all(target[fallback]==1) and trace['fallback_decision_count']==int(fallback.sum())
                        assert st['reasons']==['hybrid_hindsight' if score[v[0]] else v[6] for v in tr]
                        close('hybrid_trace_targets',[v[5] for v in tr],st['targets'])
                        for k,mask in [('learned',learned),('fallback',fallback),('hold',learned&np.isnan(target)),('missing_open',missing)]:exact('hybrid action mask '+k,np.asarray(trace['decision_masks'][k]),mask)
                        for j,v in enumerate(tr):
                            if v[6]=='forecast_unavailable':assert st['targets'][j]==1 and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None and st['hindsight_information_replaced'][j] is False
                        counts['hybrid_fallback_decisions']+=int(fallback.sum())
                    counts['hybrid_own_state_paths']+=1;counts['hybrid_own_state_decisions']+=len(tr);counts['hybrid_replaced_decisions']+=int(score.sum());counts['hybrid_retained_learned_decisions']+=int((learned&~score).sum())
        for cid in rlids:
            fallback_rule='_fallback_bh_' in cid;penalty=int(cid[-1]);target=fold_targets[cid];row=rows[f,cid];trace=read(out/'traces'/f'fold{f}_{cid}.json')
            assert trace['metrics']==row['base'] and trace['diagnostic_kind']=='matched_dynamic_action_finite_beam_hindsight'
            assert trace['future_information_used'] is True and trace['deployable'] is False and trace['teacher_use_allowed'] is False and trace['global_optimum_claimed'] is False and trace['support_causality_verified'] is False
            assert trace['bound_direction']=='lower_bound_on_maximum_attainable_hindsight_objective' and trace['objective_definition']=='log_terminal_nav_minus_risk_penalty_times_maxdd'
            assert trace['risk_penalty']==penalty and trace['beam_width']==32 and trace['missing_input_rule']==('fallback_bh' if fallback_rule else 'hold')
            assert trace['candidate_exposure_deltas']==[-.08,-.04,.04,.08] and trace['target_floor']==.5 and trace['target_ceiling']==1.12 and trace['decision_cadence_hours']==6 and trace['execution_delay_bars']==1
            forced=fallback if fallback_rule else np.zeros(len(ix),bool);eligible=learned|forced
            exact('RL support',np.asarray(trace['decision_support']),inf);exact('RL eligibility',np.asarray(trace['eligible_decisions']),eligible)
            assert not (np.isfinite(target)&~eligible).any() and np.all(target[forced]==1)
            expected_counts={'scheduled_decision_count':clock.sum(),'supported_decision_count':inf.sum(),'decision_count':eligible.sum(),'search_event_count':clock.sum() if fallback_rule else learned.sum(),'free_branching_decision_count':learned.sum(),'forced_fallback_decision_count':forced.sum(),'unsupported_decision_count':(clock&~inf).sum(),'missing_open_decision_count':missing.sum(),'supported_missing_open_decision_count':(inf&~known).sum(),'intent_count':np.isfinite(target).sum(),'hold_decision_count':np.isnan(target[eligible]).sum()}
            for k,n in expected_counts.items():assert trace[k]==int(n),(cid,k,trace[k],n)
            inv=[v for v in base_inventory[cid] if eligible[v[0]]];st=trace['decision_trace'];assert st['bar_indices']==[v[0] for v in inv]
            close('rl_trace_known_open_nav',[v[1] for v in inv],st['known_open_nav']);close('rl_trace_known_open_exposure',[v[2] for v in inv],st['known_open_exposure']);close('rl_trace_targets',target[st['bar_indices']],st['targets'])
            assert st['reasons']==['forced_fallback_bh' if forced[v[0]] else 'free_dynamic_action' for v in inv]
            for t,nav,current in inv:
                if forced[t]:assert target[t]==1.;continue
                if not math.isfinite(target[t]):continue
                candidates=[]
                for delta in (-.08,-.04,.04,.08):
                    intent=min(max(current+delta,.5),1.12);change=max(-execution['max_step'],min(execution['max_step'],intent-current))
                    if change!=0 and abs(change)>=execution['deadband'] and intent not in candidates:candidates.append(intent)
                error=min(abs(float(target[t])-v) for v in candidates) if candidates else math.inf
                maxima['rl_dynamic_intent_feasibility']=max(maxima.get('rl_dynamic_intent_feasibility',0.),error);assert error<=1e-12,(cid,t,'infeasible',target[t],candidates)
                counts['rl_finite_free_intents_verified']+=1
            objective=math.log1p(row['base']['total_return'])-penalty*row['base']['maxdd'];close('rl_objective',objective,trace['objective'])
            incumbent=np.full(len(ix),np.nan);incumbent[forced]=1.
            baseline,_,_=scalar_account(frame,incumbent,execution);inc_obj=math.log1p(baseline['total_return'])-penalty*baseline['maxdd']
            close('rl_incumbent_objective',inc_obj,trace['incumbent_objective']);assert trace['incumbent_rule']==('hold_on_supported_fallback_bh_on_unsupported' if fallback_rule else 'all_hold')
            beam=trace['beam_objective_before_incumbent'];assert isinstance(beam,(int,float)) and math.isfinite(beam)
            close('rl_incumbent_envelope_objective',max(beam,inc_obj),objective)
            assert trace['incumbent_selected']==(inc_obj>=beam) and trace['all_hold_envelope_selected']==(trace['incumbent_selected'] and not fallback_rule)
            if trace['incumbent_selected']:exact('rl selected incumbent target',target,incumbent)
            assert trace['canonical_replay_verified'] is True and trace['exhaustive_for_matched_dynamic_action_set']==(trace['pruned_distinct_branches']==0)
            for k in ('expanded_branches','pruned_distinct_branches','duplicate_states_collapsed','insolvent_branches_rejected'):assert isinstance(trace[k],int) and not isinstance(trace[k],bool) and trace[k]>=0
            counts['rl_saved_path_feasibility_verified']+=1;counts['rl_decisions_verified']+=len(inv);counts['rl_forced_fallback_decisions']+=int(forced.sum());counts['rl_incumbent_accounts']+=1
        item={'fold':f,'bars':len(ix),'inference':int(inf.sum()),'replacement':int(score.sum()),'retained_learned':int((inf&~score).sum()),'fallback':int(fallback.sum()),'missing_current_open':int(missing.sum())};counts_by_fold.append(item)
        for k,v in item.items():
            if k!='fold':totals[k]+=v
        print(json.dumps({'phase':'all_saved_paths_scalar_verified','fold':f,'account_paths':56,'hybrid_paths':12,'rl_paths':4}),flush=True)
    assert totals=={'bars':70080,'inference':2586,'replacement':2574,'retained_learned':12,'fallback':332,'missing_current_open':2}
    assert counts['parent_forecasts_verified']==40 and counts['score_only_hybrid_forecasts_verified']==48 and counts['exact_parent_control_rows']==counts['exact_parent_control_targets']==96
    assert counts['hybrid_own_state_paths']==96 and counts['hybrid_own_state_decisions']==33024 and counts['hybrid_replaced_decisions']==30888 and counts['hybrid_retained_learned_decisions']==144 and counts['hybrid_fallback_decisions']==1992
    assert counts['independent_scalar_account_paths']==448 and counts['rl_saved_path_feasibility_verified']==32 and counts['rl_decisions_verified']==11008 and counts['rl_forced_fallback_decisions']==664 and counts['rl_incumbent_accounts']==32
    keys=('alpha_ex','maxdd_delta','turnover','trades');strata=('all','bull','bear','sideways');summary={'policies':{},'oracle_minus_own_learned':{}}
    def average(values):return math.fsum(v/len(values) for v in values) if values else None
    def fs_for(cid,stratum):return [f for f in folds if stratum=='all' or rows[f,cid]['regime']['trend']==stratum]
    for cid in policies:
        summary['policies'][cid]={}
        for stratum in strata:
            fs=fs_for(cid,stratum);summary['policies'][cid][stratum]={'quarters':len(fs),**{cost:{k:average([rows[f,cid][cost][k] for f in fs]) for k in keys} for cost in ('base','stress_2x')}}
    for half in halves:
        for swap in swaps:
            for rule in rules:
                cid=half+'_oracle_'+swap+'_'+rule;refid=half+'_'+rule;entry={'reference_id':refid,'strata':{}}
                for stratum in strata:
                    fs=fs_for(cid,stratum);entry['strata'][stratum]={'quarters':len(fs),**{cost:{k:average([rows[f,cid][cost][k]-rows[f,refid][cost][k] for f in fs]) for k in keys} for cost in ('base','stress_2x')}}
                summary['oracle_minus_own_learned'][cid]=entry
    for k,v in summary.items():tree('summary.'+k,v,res['summary'][k])
    assert res['summary']['new_causal_candidate_count']==0
    for k in ('selection_performed','teacher_use_allowed','high_probability_generalization_established'):assert res['summary'][k] is False
    assert res['new_models_fitted']==0 and res['new_causal_candidates']==0 and res['hindsight_only_diagnostics']==16
    for k in ('selection_performed','test_periods_used','teacher_use_allowed'):assert res[k] is False
    report={'status':'pass','source_revision':reg['source_revision'],'audit_script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_artifacts':1328,'own_artifacts':400,'inventory':{'economic_rows':224,'forecasts':48,'targets':224,'traces':128},'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maxima,'data_support':dict(totals),'counts_by_fold':counts_by_fold,'regime_quarters':{'bull':2,'bear':4,'sideways':2},'scope':'All source and output bindings; all 40 parent and 48 score-only hybrid forecast arrays; 96 exact parent controls; all 96 hybrid own-state paths and 448 base/stress scalar cash/unit accounts; 32 independently replayed RL chosen paths plus 32 rule-matched feasible incumbent accounts; all 28 policy summaries and 12 paired contrasts. No canonical metrics, simulator, planner, chooser, intervention, hindsight search, or fitting helper imported.','limitations':['Diagnostic future labels/prices are intentionally noncausal; no teacher, deployability, causal superiority, or new predictive accuracy claim.','The finite beam output is a feasible lower bound on the maximum attainable objective; search expansion/pruning completeness is not independently rerun and global optimality is not established.','Base-selected intents are reused under stress costs; stress is an accounting sensitivity, not a separately optimized Oracle.','Original adaptive reused development only; retrospective common availability and insufficient 2/4/2 regime coverage remain.','Semantic raw Spot reads are filtered strictly before 2023-04-16T13:45Z; entire source files are hashed as bytes. No later price, forecast, or economic values are interpreted.']}
    path=Path('/tmp/oracle_information_decomposition_audit_20260906.json');path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'pass','report':str(path),'counts':dict(counts),'maxima':{k:v for k,v in maxima.items() if not k.startswith('summary.')}}),flush=True)

if __name__=='__main__':main()
