"""Stage16 independent immutable-source, own-state and accounting audit.
Requires explicit registered completion authorization. No canonical simulator,
planner, substitution, scoring, quantile or model helper is imported.
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
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--expected-source-revision',required=True);ap.add_argument('--expected-results-sha256',required=True);ap.add_argument('--authorized-completed-run',action='store_true');args=ap.parse_args()
 if not args.authorized_completed_run or not re.fullmatch('[0-9a-f]{40}',args.expected_source_revision) or not re.fullmatch('[0-9a-f]{64}',args.expected_results_sha256):raise SystemExit('Full registered revision, final result SHA and completed-run authorization required.')
 root=Path.cwd();out=root/'codex_outputs/oracle_sign_magnitude_decisions_v1';parent=root/'codex_outputs/oracle_information_decomposition_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1'
 assert (out/'results.json').is_file(),'Completed result required before real substitutions or orders are audited.'
 def read(p):return json.loads(Path(p).read_text())
 def sha(p):
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  return h.hexdigest()
 def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 counts=Counter();maximum={};verified={}
 def verify(p,h):
  p=Path(p).resolve()
  if p not in verified:verified[p]=sha(p)
  assert verified[p]==h,('hash',str(p));counts['hash_binding_checks']+=1
 def arr(p,keys=None):
  with np.load(p,allow_pickle=False) as z:
   if keys is not None:assert set(z.files)==set(keys),('schema',str(p))
   a={k:z[k] for k in z.files}
  assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid',str(p))
  return a
 def exact(name,a,b):
  a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),('exact',name)
 def close(name,a,b,tol=1e-10):
  a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('invalid',name)
  finite=np.isfinite(a);e=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.;maximum[name]=max(maximum.get(name,0.),e);assert e<=tol,('numerical',name,e,tol)
 sourcepath=Path('/tmp/oracle_sign_magnitude_source_audit_20260906.json');verify(sourcepath,'7e4658010cdd1c201effd851eff7e2457c6ca5ba5e5bd1153054a7ac2f1ba52f');source=read(sourcepath)
 verify('/tmp/oracle_sign_magnitude_source_audit_20260906.py','6811a52c07f86dfa5591185d8755f0a47ef10ca412e87600fe6cf302c5e6532e');assert source['passed'] and source['counts']['total_source_artifacts']==1728
 for p,h in {**source['source_artifact_bindings'],**source['direct_source_bindings']}.items():verify(root/p,h)
 reg,pre,res=(read(out/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config'];verify(out/'results.json',args.expected_results_sha256)
 assert reg['source_revision']==args.expected_source_revision and res['registration_sha256']==digest(reg)
 cfgpath=root/'configs/oracle_sign_magnitude_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
 verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==25
 for p,h in cfg['source_bindings'].items():
  verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+p])).hexdigest()==h
 assert pre['source_artifact_bindings']==source['source_artifact_bindings']
 for p,h in pre['direct_source_bindings'].items():verify(root/p,h)
 assert pre['new_forecasts_orders_scores_or_quantiles_computed'] is False and pre['new_model_fits']==0
 pr,pres,pp=(read(parent/(k+'.json')) for k in ('registration','results','preflight'))
 assert digest(pr)==pres['registration_sha256'] and cfg['parent_source_revision']==pr['source_revision']==source['parent_source_revision']
 assert pre['spot_data_proof']==pp['spot_data_proof'] and pre['um_data_proof']==pp['um_data_proof']
 for p,h in cfg['parent_manifest_bindings'].items():verify(root/p,h)
 halves=('technical_half','perp_delay0_half');rules=('utility_risk1','utility_risk1_fallback_bh');components=('sign','magnitude');folds=tuple(range(5,13));subsets=('all','fit_q90_large','fit_q90_other')
 base_means=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half');causal=('bh','common_robust')+tuple(m+'_'+r for m in base_means for r in rules)
 oldhybrids=tuple(h+'_oracle_'+c for h in halves for c in ('return','realized_risk','both'));rl=tuple(f'matched_rl_beam32_{r}_risk{k}' for r in ('hold','fallback_bh') for k in (0,1));oldhindsight=tuple(m+'_'+r for m in oldhybrids for r in rules)+rl;controls=causal+oldhindsight
 newmeans=tuple(h+'_oracle_'+c for h in halves for c in components);newids=tuple(m+'_'+r for m in newmeans for r in rules);policies=controls+newids;scoremeans=tuple(m for h in halves for m in (h,h+'_oracle_sign',h+'_oracle_magnitude',h+'_oracle_return'));endpointmeans=tuple(m for h in halves for m in (h,h+'_oracle_return'));endpoints=tuple(m+'_'+r for m in endpointmeans for r in rules)
 assert cfg['halves']==list(halves) and cfg['components']==list(components) and cfg['control_ids']==list(controls) and cfg['new_diagnostic_ids']==list(newids) and cfg['score_means']==list(scoremeans) and cfg['score_subsets']==list(subsets)
 assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['tail_quantile']==.9 and cfg['tail_quantile_method']=='linear' and cfg['tail_fit_only'] is True
 assert cfg['adaptive_causal_names_unchanged']==174 and cfg['new_causal_names']==0 and cfg['new_model_fits']==0
 for k in ('selection_permitted','teacher_use_allowed','additional_test_permitted'):assert cfg[k] is False
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(s['fold'],s['mean_id'],s['subset']):s for s in res['scores']};thresholds={t['fold']:t for t in res['thresholds']};prows={(r['fold'],r['candidate_id']):r for r in pres['rows']}
 endpoint_records={(r['fold'],r['candidate_id']):r for r in res['endpoint_parity']}
 assert len(endpoint_records)==len(res['endpoint_parity'])==64 and set(endpoint_records)=={(f,c) for f in folds for c in endpoints}
 for (f,c),r in endpoint_records.items():assert r['targets_exact'] is True and r['full_decision_trace_matches'] is ('_oracle_return_' in c)
 assert len(rows)==len(res['rows'])==288 and set(rows)=={(f,c) for f in folds for c in policies};assert len(scores)==len(res['scores'])==192 and set(scores)=={(f,m,s) for f in folds for m in scoremeans for s in subsets};assert len(thresholds)==8 and set(thresholds)==set(folds)
 own={}
 for f in folds:
  doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
  for k in ('rows','scores','direction_diagnostics','endpoint_parity'):assert doc[k]==[r for r in res[k] if r['fold']==f]
  assert doc['threshold']==thresholds[f]
  expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('forecasts',newmeans,'npz'),('targets',policies,'npz'),('traces',newids,'json'),('thresholds',('fit_q90',),'json')] for v in vs}
  assert len(expected)==49 and set(doc['artifact_sha256'])==expected
  for p,h in doc['artifact_sha256'].items():verify(root/p,h);own[p]=h
 assert len(own)==392
 for k,n in [('forecasts',32),('targets',288),('traces',64),('thresholds',8)]:assert len(list((out/k).iterdir()))==n
 assert not (out/'models').exists() and not (out/'calibration').exists()
 fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());execution=fc['execution'];assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
 sp=Path(fc['data_path']);verify(sp,pre['spot_data_proof']['artifact_sha256']);cut=pd.Timestamp(cfg['data_cutoff']);bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC'));bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
 fkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};totals=Counter();regimes=Counter()
 print(json.dumps({'phase':'all_immutable_bindings_verified','ancestor_artifacts':1728,'own_artifacts':392}),flush=True)
 for f in folds:
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);ix=pd.date_range(E,end,freq='15min',inclusive='left');frame=bars.loc[ix];saved={h:arr(parity/'forecasts'/f'fold{f}_{h}.npz',fkeys) for h in halves};ref=saved[halves[0]];exact('calendar',ref['timestamps'],ix.asi8)
  inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
  assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any() and not (inf&~known).any() and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
  p=next(v for v in pre['support'] if v['fold']==f);s=next(v for v in source['support'] if v['fold']==f);fit=s['past_fit_return_identity'];assert p['fit_rows']==fit['fit_rows'] and p['fit_mask_sha256']==fit['fit_mask_sha256'] and p['fit_return_sha256']==fit['selected_fit_returns_sha256']
  assert p['inference_rows']==int(inf.sum()) and p['score_rows']==int(score.sum());t=thresholds[f];assert read(out/'thresholds'/f'fold{f}_fit_q90.json')==t and t['fit_rows']==fit['fit_rows'] and t['fit_return_sha256']==fit['selected_fit_returns_sha256'] and t['used_for_orders'] is False and t['hindsight_tail_grouping'] is True
  assert t['subset_rows']['all']==int(score.sum()) and t['subset_rows']['fit_q90_large']+t['subset_rows']['fit_q90_other']==int(score.sum())
  for h in halves:
   base=saved[h]
   for k in fkeys-{'mu'}:exact('paired base '+k,base[k],ref[k])
   full=arr(parent/'forecasts'/f'fold{f}_{h}_oracle_return.npz',fkeys);saved[h+'_oracle_return']=full
   for k in fkeys-{'mu'}:exact('full unchanged '+k,full[k],base[k])
   exact('full return label',full['mu'][score],base['actual'][score,0]);exact('full unchanged remainder',full['mu'][~score],base['mu'][~score])
   for component in components:
    name=h+'_oracle_'+component;a=arr(out/'forecasts'/f'fold{f}_{name}.npz',fkeys);saved[name]=a
    for k in fkeys-{'mu'}:exact('new nonmean '+k,a[k],base[k])
    expected=base['mu'].copy()
    for j in np.flatnonzero(score):
     b,y=float(base['mu'][j]),float(base['actual'][j,0]);sgn=lambda x:1. if x>0 else -1. if x<0 else 0.
     expected[j]=sgn(y)*abs(b) if component=='sign' else sgn(b)*abs(y)
    exact('independent score-only component substitution',a['mu'],expected);exact('new availability',np.isfinite(a['mu']),inf);counts['new_component_forecasts']+=1
  for cid in policies:
   row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'] and row['hindsight_only'] is (cid in oldhindsight+newids)
   pth=out/'targets'/f'fold{f}_{cid}.npz';verify(pth,row['targets_sha256']);a=arr(pth,{'timestamps','targets'});exact('target calendar',a['timestamps'],ix.asi8);target=a['targets'];finite=np.isfinite(target)
   assert not (finite&~clock).any() and ((target[finite]>=0)&(target[finite]<=1.12)).all()
   if cid in controls:
    old=arr(parent/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('old targets',target,old['targets'])
    for cost in ('base','stress_2x'):assert row[cost]==prows[f,cid][cost]
    counts['exact_old_controls']+=1
   for cost,mul in [('base',1),('stress_2x',2)]:
    contract=dict(execution);contract['one_way_cost']*=mul;contract['borrow_annual']*=mul;computed,_,_=scalar_account(frame,target,contract);assert set(computed)==set(row[cost])
    for k,v in computed.items():close('account_'+k,v,row[cost][k])
    counts['independent_scalar_accounts']+=1
   if cid not in newids+endpoints:continue
   rule=rules[1] if cid.endswith(rules[1]) else rules[0];mean=cid[:-(len(rule)+1)];isfallback=rule==rules[1];hindsight=mean not in halves
   _,expected,tr=scalar_account(frame,target,execution,saved[mean],inf,fallback_enabled=isfallback);exact('independent own-state targets',target,expected)
   allowed=learned|fallback if isfallback else learned;assert not (finite&~allowed).any()
   if isfallback:assert np.all(target[fallback]==1.)
   group='new' if cid in newids else 'endpoint';counts[group+'_own_state_paths']+=1;counts[group+'_own_state_decisions']+=len(tr)
   # New traces and frozen full-return endpoint traces are checked completely.
   if cid in newids:tp=out/'traces'/f'fold{f}_{cid}.json';verify(tp,row['diagnostic_sha256'])
   elif hindsight:tp=parent/'traces'/f'fold{f}_{cid}.json';verify(tp,prows[f,cid]['diagnostic_sha256'])
   else:continue
   trace=read(tp);assert trace['metrics']==row['base'] and trace['future_information_used_for_decisions'] is True and trace['hindsight_only'] is True and trace['deployable'] is False and trace['teacher_use_allowed'] is False
   for k in ('teacher_actions_used','global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed'):assert trace[k] is False
   assert trace['canonical_replay_verified'] is True and trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24 and trace['decision_cadence_hours']==6 and trace['execution_delay_bars']==1
   st=trace['decision_trace'];assert st['bar_indices']==[q[0] for q in tr] and st['hindsight_information_replaced']==[bool(score[q[0]]) for q in tr]
   for col,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('trace_'+key,[q[col] for q in tr],st[key])
   assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum())
   if isfallback:
    assert trace['fallback_decision_count']==int(fallback.sum());close('trace_targets',[q[5] for q in tr],st['targets']);assert st['reasons']==['hybrid_hindsight' if score[q[0]] else q[6] for q in tr]
    for k,m in [('learned',learned),('fallback',fallback),('missing_open',missing),('hold',learned&np.isnan(target))]:exact('decision mask '+k,np.asarray(trace['decision_masks'][k]),m)
    for j,q in enumerate(tr):
     if q[6]=='forecast_unavailable':assert st['targets'][j]==1. and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None
   if cid in newids:
    info=trace['information_intervention'];component=next(c for c in components if mean.endswith('_'+c));assert trace['information_swap']==component and info['component']==component and info['inference_rows']==int(inf.sum()) and info['replacement_rows']==int(score.sum()) and info['learned_remainder_rows']==int((inf&~score).sum())
    for k in ('hindsight_only','future_information_used_for_decisions','inference_and_missing_action_support_unchanged','variance_unchanged'):assert info[k] is True
    for k in ('deployable','teacher_use_allowed','global_optimum_claimed','other_outcome_columns_used'):assert info[k] is False
    counts['new_unscored_decisions_retained']+=int((inf&~score).sum());counts['new_replaced_decision_entries']+=int(sum(score[q[0]] for q in tr))
    if isfallback:counts['new_fallback_decisions']+=int(fallback.sum())
    counts['new_full_trace_replays']+=1
   else:counts['old_full_return_trace_replays']+=1
  regimes[prows[f,'bh']['regime']['trend']]+=1
  print(json.dumps({'phase':'fold_scalar_audit_complete','fold':f,'accounts':72,'new_paths':8,'endpoint_paths':8}),flush=True)
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2} and regimes=={'bull':2,'bear':4,'sideways':2}
 expectedcounts={'new_component_forecasts':32,'exact_old_controls':224,'independent_scalar_accounts':576,'new_own_state_paths':64,'new_own_state_decisions':22016,'endpoint_own_state_paths':64,'endpoint_own_state_decisions':22016,'new_full_trace_replays':64,'old_full_return_trace_replays':32,'new_unscored_decisions_retained':96,'new_fallback_decisions':1328,'new_replaced_decision_entries':20592}
 for k,n in expectedcounts.items():assert counts[k]==n,(k,counts[k],n)
 for k,n in {'new_model_fits':0,'fit_distribution_thresholds':8,'new_causal_names':0,'adaptive_causal_names_unchanged':174,'new_hindsight_policy_names':8}.items():assert res[k]==n
 for k in ('selection_performed','teacher_use_allowed','additional_test_used','high_probability_generalization_established'):assert res[k] is False
 report={'status':'pass','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_audit_sha256':sha(sourcepath),'ancestor_artifacts_verified':1728,'own_artifacts_verified':392,'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maximum,'support':dict(totals),'regime_counts':dict(regimes),'inventory':{'forecasts':32,'targets':288,'traces':64,'thresholds':8,'rows':288,'scores':192},'scope':'All immutable sources/artifacts; all32 score-only sign/magnitude substitutions; unchanged risk, masks and off-score values;224 exact source controls;576 scalar accounts;64 new and64 endpoint own-state paths;64 new and32 original full-return decision traces. No canonical helpers imported or model fits performed.','separate_audit_scope':['The 8 q90 values, 192 loss records, direction diagnostics and summary arithmetic are independently audited by the statistics auditor.'],'limitations':['All new policies use future information and are nondeployable diagnostics, never causal teachers or global optima.','Original repeatedly reused development, retrospective masks and absent historical receipt evidence remain.','Fit threshold groups are descriptive only and do not gate any orders.','Filtered raw Spot reads remain below the original cutoff; source hashes cover full files.']}
 path=Path('/tmp/oracle_sign_magnitude_audit_20260906.json');path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':'pass','path':str(path),'counts':dict(counts),'maximum':maximum}),flush=True)
if __name__=='__main__':main()
