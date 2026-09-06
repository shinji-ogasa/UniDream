"""Stage19 independent immutable-source, mapping, own-state and accounting audit.
Requires explicit registered completion authorization. No canonical simulator,
planner, mapping, scoring or model helper is imported.
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
 root=Path.cwd();out=root/'codex_outputs/oracle_soft_direction_decisions_v1';parent=root/'codex_outputs/oracle_regularized_direction_decisions_v1';stage17=root/'codex_outputs/oracle_direction_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1'
 assert (out/'results.json').is_file(),'Completed result required before new saved means or orders are audited.'
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
   if keys is not None:assert set(z.files)==set(keys),('schema',str(p),z.files)
   a={k:z[k] for k in z.files}
  assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid',str(p))
  return a
 def exact(name,a,b):
  a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),('exact',name)
 def close(name,a,b,tol=1e-10):
  a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('invalid',name)
  finite=np.isfinite(a);e=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.;maximum[name]=max(maximum.get(name,0.),e);assert e<=tol,('numerical',name,e,tol)
 sourcepath=Path('/tmp/oracle_soft_direction_source_audit_20260906.json');verify(sourcepath,'4965de3c0ba40741fb9bff7d588a2e62ec19fbd6b547d013a032a99bd3c2b96b');source=read(sourcepath)
 verify('/tmp/oracle_soft_direction_source_audit_20260906.py','ade635ec900560bce57f6be917599194efd6bb3f0e240d680832072d63ea665b');assert source['passed'] and source['counts']['total_source_artifacts']==3488
 for p,h in {**source['source_artifact_bindings'],**source['direct_source_bindings'],**source['saved_input_bindings']}.items():verify(root/p,h)
 bindingauditpath=Path('/tmp/oracle_soft_direction_preflight_binding_audit_20260906.json');verify(bindingauditpath,'36919a9a9bb725ee58006925bc45846c935e1d1514ac5b0170622f49ca3caa25');ba=read(bindingauditpath);assert ba['status']=='pass_pre_freeze_input_bindings'
 verify('/tmp/oracle_soft_direction_preflight_binding_audit_20260906.py',ba['script_sha256'])
 reg,pre,res=(read(out/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config'];verify(out/'results.json',args.expected_results_sha256)
 assert reg['source_revision']==args.expected_source_revision and res['registration_sha256']==digest(reg)
 cfgpath=root/'configs/oracle_soft_direction_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg and reg['config_sha256']==ba['config_sha256']
 assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
 verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256']==ba['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings']==ba['source_bindings'] and len(cfg['source_bindings'])==33
 for p,h in cfg['source_bindings'].items():
  verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+p])).hexdigest()==h
 assert pre['source_artifact_bindings']==source['source_artifact_bindings']
 for p,h in pre['direct_source_bindings'].items():verify(root/p,h)
 assert pre['new_statistics_mapped_predictions_losses_or_orders_computed'] is False and pre['no_estimator_fit_or_predict_called'] is True
 pr,pres,pp=(read(parent/(k+'.json')) for k in ('registration','results','preflight'))
 assert digest(pr)==pres['registration_sha256'] and pr['source_revision']==source['parent_source_revision']=='5a82c270c64a342ab7e9df8105b7d23d1336d876'
 assert pre['spot_data_proof']==source['spot_data_proof'] and pre['um_data_proof']==source['um_data_proof']
 groups=('technical','perp_delay0');halves=tuple(g+'_half' for g in groups);rules=('utility_risk1','utility_risk1_fallback_bh');folds=tuple(range(5,13));segments=('interval','evaluation')
 probabilityids=tuple(g+'_magnitude'+suffix for g in groups for suffix in ('','_l2unit'));softmeans=tuple(m+'_soft' for m in probabilityids);constantkinds=('mapped_prior','fit_mean','zero');constantmeans=tuple(g+'_soft_'+k for g in groups for k in constantkinds);newmeans=softmeans+constantmeans
 controls=tuple(pr['config']['control_ids'])+tuple(pr['config']['new_policy_ids']);assert len(controls)==60
 mapping={m+'_soft':{'group':g,'kind':'soft','role':'learned_mapping','source_classifier':m,'source_mean':m+'_direction','parent_mean':g+'_half'} for g in groups for m in probabilityids if m.startswith(g+'_')}
 mapping.update({g+'_soft_'+k:{'group':g,'kind':k,'role':'constant_control','source_classifier':g+'_magnitude','source_mean':g+'_magnitude_direction','parent_mean':g+'_half'} for g in groups for k in constantkinds})
 newids=tuple(m+'_'+rule for m in newmeans for rule in rules);learnedids=tuple(m+'_'+rule for m in softmeans for rule in rules);policies=controls+newids
 assert cfg['probability_ids']==list(probabilityids) and cfg['new_mean_ids']==list(newmeans) and cfg['new_policy_ids']==list(newids) and cfg['learned_policy_ids']==list(learnedids) and cfg['control_ids']==list(controls) and cfg['mapping']==mapping
 assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['surrogate_mean']=='saved_fit_abs_return_mean*(2.0*saved_probability-1.0)'
 for k in ('selection_permitted','teacher_use_allowed','additional_test_permitted','new_mean_risk_probability_or_weight_calibration_permitted'):assert cfg[k] is False
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};prows={(r['fold'],r['candidate_id']):r for r in pres['rows']};records={r['fold']:r for r in res['mapping_records']}
 scores={(r['fold'],r['segment'],r['mean_id']):r for r in res['scores']};classes={(r['fold'],r['segment'],r['classifier_id']):r for r in res['classification_scores']};diag={(r['fold'],r['segment'],r['mean_id']):r for r in res['mapping_diagnostics']}
 assert len(rows)==len(res['rows'])==640 and set(rows)=={(f,c) for f in folds for c in policies}
 assert len(records)==len(res['mapping_records'])==8 and set(records)==set(folds)
 assert len(scores)==len(res['scores'])==384 and set(scores)=={(f,seg,m) for f in folds for seg in segments for m in cfg['return_score_means']}
 assert len(classes)==len(res['classification_scores'])==160 and set(classes)=={(f,seg,m) for f in folds for seg in segments for m in cfg['classifiers']}
 assert len(diag)==len(res['mapping_diagnostics'])==64 and set(diag)=={(f,seg,m) for f in folds for seg in segments for m in softmeans}
 assert set(res['summary']['soft'])==set(learnedids)
 oldhindsight={c for c in controls if prows[5,c]['hindsight_only']};assert len(oldhindsight)==24
 own={}
 for f in folds:
  doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
  for k in ('rows','scores','classification_scores','mapping_diagnostics'):assert doc[k]==[r for r in res[k] if r['fold']==f]
  expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('provenance',('mapping',),'json'),('forecasts',newmeans,'npz'),('calibration',newmeans,'npz'),('targets',policies,'npz'),('traces',newids,'json')] for v in vs}
  assert len(expected)==121 and set(doc['artifact_sha256'])==expected
  for p,h in doc['artifact_sha256'].items():verify(root/p,h);own[p]=h
 assert len(own)==968
 inventory={'provenance':8,'forecasts':80,'calibration':80,'targets':640,'traces':160}
 for k,n in inventory.items():assert len(list((out/k).iterdir()))==n
 for k in ('models','fit_data'):assert not (out/k).exists()
 for key,idkey in [('scores','mean_id'),('classification_scores','classifier_id')]:
  current={(r['fold'],r['segment'],r[idkey]):r for r in res[key]}
  for r in pres[key]:assert current[r['fold'],r['segment'],r[idkey]]==r;counts['exact_old_'+key]+=1
 execution=cfg['execution'];assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
 sp=Path(cfg['data_path']);verify(sp,pre['spot_data_proof']['artifact_sha256']);cut=pd.Timestamp(cfg['data_cutoff']);bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC'));bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
 fkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};newfkeys=fkeys|{'parent_mu','logit','probability'};calk={'timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','mu','parent_mu','logit','probability','fit_return_mean'};totals=Counter();regimes=Counter()
 print(json.dumps({'phase':'all_immutable_bindings_verified','ancestor_artifacts':3488,'own_artifacts':968}),flush=True)
 for f in folds:
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3)
  ix=pd.date_range(E,end,freq='15min',inclusive='left');cix=pd.date_range(S,E,freq='15min',inclusive='left');frame=bars.loc[ix];saved={h:arr(parity/'forecasts'/f'fold{f}_{h}.npz',fkeys) for h in halves};ref=saved[halves[0]];exact('calendar',ref['timestamps'],ix.asi8)
  inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
  assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any() and not (inf&~known).any() and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
  sourcefold=next(r for r in source['support'] if r['fold']==f);presupport=next(r for r in pre['support'] if r['fold']==f);scalars=presupport['saved_mapping_scalars'];record=read(out/'provenance'/f'fold{f}_mapping.json')
  assert record==records[f] and record['saved_T_scalars']==scalars and record['new_fits']==0 and record['new_unique_priors']==0 and record['frozen_input_bindings']==presupport['saved_probability_inputs']
  assert record['mapping_formula']==cfg['surrogate_mean'] and record['probability_arrays_unchanged'] is True and record['caller_saved_input_provenance_and_calendar_verified'] is True and set(record['mean_records'])==set(newmeans)
  oldfit=read(stage17/'provenance'/f'fold{f}_fit.json');assert scalars['fit_abs_return_mean']==oldfit['fit_abs_return_mean'] and scalars['fit_return_mean']==oldfit['fit_return_mean'] and scalars['fit_statistical_magnitude_prior']==oldfit['fit_priors']['magnitude']
  for g in groups:assert scalars['prior_probability'][g]==sourcefold['saved_prior_probability_sources'][g]['saved_prior_probability']
  counts['frozen_mapping_provenance_records_bound']+=1
  for m in newmeans:
   mp=mapping[m];g=mp['group'];oldname=mp['source_mean'];oldroot=parent if oldname in pr['config']['new_mean_ids'] else stage17
   old=arr(oldroot/'forecasts'/f'fold{f}_{oldname}.npz',newfkeys);oldca=arr(oldroot/'calibration'/f'fold{f}_{oldname}.npz',calk)
   a=arr(out/'forecasts'/f'fold{f}_{m}.npz',newfkeys);ca=arr(out/'calibration'/f'fold{f}_{m}.npz',calk);saved[m]=a
   for k in newfkeys-{'mu'}:exact('unchanged E source '+k,a[k],old[k])
   for k in calk-{'mu'}:exact('unchanged cal source '+k,ca[k],oldca[k])
   exact('unchanged technical risk',a['variance'],ref['variance']);exact('E mapping support',np.isfinite(a['mu']),inf);exact('cal mapping support',np.isfinite(ca['mu']),ca['mapped_inference_mask'])
   mapped=ca['mapped_inference_mask'];assert mapped.dtype==bool and not (mapped&np.asarray(cix<I)).any();exact('I original support',mapped,ca['classifier_predict_mask']&np.asarray(cix>=I))
   assert not (ca['interval_mask']&~mapped).any() and (cix[ca['interval_mask']]+pd.Timedelta(minutes=375)<E).all()
   meta=record['mean_records'][m]
   for k,v in mp.items():assert meta[k]==v
   assert meta['source_probability_and_logit_fields_are_preserved_source_evidence'] is True
   for segment,support in [('interval',mapped),('evaluation',inf)]:
    sd=meta['segment_diagnostics'][segment];assert sd['inference_rows']==int(support.sum()) and sd['fit_abs_return_mean']==scalars['fit_abs_return_mean'] and sd['fit_return_mean']==scalars['fit_return_mean'] and sd['saved_weighted_prior_probability']==scalars['prior_probability'][g]
    assert sd['model_fits']==0 and sd['calibration_fits']==0 and sd['probabilities_recomputed'] is False and sd['future_outcomes_or_score_support_used'] is False and sd['prior_identity_passed'] is True
   for segment,support in [('interval',ca['interval_mask']),('evaluation',score)]:assert scores[f,segment,m]['rows']==int(support.sum())
   if mp['kind'] in ('fit_mean','zero'):
    value=scalars['fit_return_mean'] if mp['kind']=='fit_mean' else 0.
    exact('stored numeric constant E',a['mu'][inf],np.full(int(inf.sum()),value));exact('stored numeric constant I',ca['mu'][mapped],np.full(int(mapped.sum()),value))
   if m in softmeans:
    for segment,support in [('interval',mapped),('evaluation',inf)]:assert diag[f,segment,m]['rows']==int(support.sum()) and diag[f,segment,m]['uses_all_inference_not_score_support'] is True
   counts['new_E_source_and_support_arrays']+=1;counts['new_calibration_source_and_support_arrays']+=1
  for g in groups:
   for kind in constantkinds:
    a=saved[g+'_soft_'+kind];assert np.all(a['mu'][inf]==a['mu'][inf][0])
   for kind in ('zero','fit_mean'):
    exact('cross-group fixed constants',saved['technical_soft_'+kind]['mu'],saved['perp_delay0_soft_'+kind]['mu'])
  # I support counted once per fold, independently of the 10 means sharing it.
  totals['interval_inference']+=int(mapped.sum());totals['interval_score']+=int(ca['interval_mask'].sum())
  for cid in policies:
   row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'] and row['hindsight_only'] is (cid in oldhindsight)
   pth=out/'targets'/f'fold{f}_{cid}.npz';verify(pth,row['targets_sha256']);a=arr(pth,{'timestamps','targets'});exact('target calendar',a['timestamps'],ix.asi8);target=a['targets'];finite=np.isfinite(target)
   assert not (finite&~clock).any() and ((target[finite]>=0)&(target[finite]<=1.12)).all()
   if cid in controls:
    old=arr(parent/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('old targets',target,old['targets']);assert row==prows[f,cid]
    counts['exact_complete_old_control_rows']+=1
   for cost,mul in [('base',1),('stress_2x',2)]:
    contract=dict(execution);contract['one_way_cost']*=mul;contract['borrow_annual']*=mul;computed,_,_=scalar_account(frame,target,contract);assert set(computed)==set(row[cost])
    for k,v in computed.items():close('account_'+k,v,row[cost][k])
    counts['independent_scalar_accounts']+=1
   if cid not in newids:continue
   rule=rules[1] if cid.endswith(rules[1]) else rules[0];mean=cid[:-(len(rule)+1)];isfallback=rule==rules[1]
   _,expected,tr=scalar_account(frame,target,execution,saved[mean],inf,fallback_enabled=isfallback);exact('independent own-state targets',target,expected)
   allowed=learned|fallback if isfallback else learned;assert not (finite&~allowed).any()
   if isfallback:assert np.all(target[fallback]==1.)
   counts['new_own_state_paths']+=1;counts['new_own_state_decisions']+=len(tr)
   counts['new_learned_paths' if cid in learnedids else 'new_constant_control_paths']+=1
   tp=out/'traces'/f'fold{f}_{cid}.json';verify(tp,row['trace_sha256']);trace=read(tp);assert trace['metrics']==row['base']
   for k in ('future_information_used_for_decisions','hindsight_only','teacher_actions_used','global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed'):assert trace[k] is False
   assert trace['soft_direction_mapping']=={**mapping[mean],'mapping_formula':cfg['surrogate_mean'],'saved_T_scalars':scalars,'future_labels_used_for_orders':False,'new_fit_or_calibration':False}
   assert trace['canonical_replay_verified'] is True and trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24 and trace['decision_cadence_hours']==6 and trace['execution_delay_bars']==1
   st=trace['decision_trace'];assert st['bar_indices']==[q[0] for q in tr] and 'hindsight_information_replaced' not in st
   for col,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('trace_'+key,[q[col] for q in tr],st[key])
   assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum()) and trace['scheduled_decision_count']==int(clock.sum())
   if isfallback:
    assert trace['fallback_decision_count']==int(fallback.sum());close('trace_targets',[q[5] for q in tr],st['targets']);assert st['reasons']==[q[6] for q in tr]
    for k,mask in [('learned',learned),('fallback',fallback),('missing_open',missing),('hold',learned&np.isnan(target))]:exact('decision mask '+k,np.asarray(trace['decision_masks'][k]),mask)
    for j,q in enumerate(tr):
     if q[6]=='forecast_unavailable':assert st['targets'][j]==1. and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None
    counts['new_fallback_decisions']+=int(fallback.sum())
   counts['new_full_trace_replays']+=1;counts['new_unscored_E_decisions_retained']+=int((inf&~score).sum())
  regimes[prows[f,'bh']['regime']['trend']]+=1
  print(json.dumps({'phase':'fold_scalar_audit_complete','fold':f,'accounts':160,'new_paths':20}),flush=True)
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2,'interval_inference':2537,'interval_score':2523} and regimes=={'bull':2,'bear':4,'sideways':2}
 expectedcounts={'exact_old_scores':224,'exact_old_classification_scores':160,'frozen_mapping_provenance_records_bound':8,'new_E_source_and_support_arrays':80,'new_calibration_source_and_support_arrays':80,'exact_complete_old_control_rows':480,'independent_scalar_accounts':1280,'new_own_state_paths':160,'new_own_state_decisions':55040,'new_full_trace_replays':160,'new_unscored_E_decisions_retained':240,'new_fallback_decisions':3320,'new_learned_paths':64,'new_constant_control_paths':96}
 for k,n in expectedcounts.items():assert counts[k]==n,(k,counts[k],n)
 for k in ('selection_performed','teacher_use_allowed','additional_test_used_for_modeling_or_scoring','high_probability_generalization_established','new_probability_accuracy_improvement'):assert res[k] is False
 assert res['probability_predictions_and_scores_unchanged'] is True
 for k,n in {'new_model_fits':0,'new_unique_priors':0,'new_causal_policy_names':20,'new_learned_policy_names':8,'new_constant_control_policy_names':12,'total_adaptively_explored_causal_names':218,'risk_model_or_calibration_fits':0}.items():assert res[k]==n
 log=out/'run.log';warnings=Counter(line.split('RuntimeWarning: ',1)[1] for line in log.read_text().splitlines() if 'RuntimeWarning: ' in line)
 report={'status':'pass','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_audit_sha256':sha(sourcepath),'preflight_binding_audit_sha256':sha(bindingauditpath),'ancestor_artifacts_verified':3488,'own_artifacts_verified':968,'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maximum,'support':dict(totals),'regime_counts':dict(regimes),'inventory':{**inventory,'rows':640,'return_scores':384,'classification_scores':160,'mapping_diagnostics':64},'runtime_warnings':{'run_log_sha256':sha(log),'counts':dict(warnings),'total':sum(warnings.values())},'scope':'All immutable sources/artifacts,80 E+80 I source/support/variance bindings and frozen T/prior provenance;480 exact complete old rows including inherited trace hashes;224/160 exact old score records;1280 scalar accounts;160 new own-state paths and full decision traces, split64 learned/96 constant paths. No canonical helpers imported or estimator calls performed.','separate_audit_scope':['Numerical soft/mapped-prior arithmetic, constant residuals and64 mapping diagnostics are independently audited by the technical auditor. This audit independently uses saved mapped means for own-state/account recomputation and verifies zero/stored-fitmean constants directly.','The384 return losses and report summaries are independently audited by the statistics auditor.'],'limitations':['Constant NPZ q/logit fields remain frozen classifier source evidence, not new constant-classifier forecasts.','New probability accuracy is unchanged by construction; A_T is a constant-amplitude surrogate, not verified conditional E[abs(Y)|X].','Repeatedly reused development, retrospective masks and absent historical receipt evidence remain.','Filtered raw Spot reads remain below original cutoff; source hashes cover full files.']}
 path=Path('/tmp/oracle_soft_direction_audit_20260906.json');path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':'pass','path':str(path),'counts':dict(counts),'maximum':maximum,'warnings':dict(warnings)}),flush=True)
if __name__=='__main__':main()
