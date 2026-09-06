"""Stage20 independent immutable-source, mapping, own-state and accounting audit.
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
 if not args.authorized_completed_run or not re.fullmatch('[0-9a-f]{40}',args.expected_source_revision) or not re.fullmatch('[0-9a-f]{64}',args.expected_results_sha256):raise SystemExit('Registered completion authorization required')
 root=Path.cwd();out=root/'codex_outputs/oracle_short_direction_decisions_v1';parent=root/'codex_outputs/oracle_soft_direction_decisions_v1';s17=root/'codex_outputs/oracle_direction_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1'
 def read(p):return json.loads(Path(p).read_text())
 def sha(p):
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  return h.hexdigest()
 def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 def matrixsha(a):
  a=np.asarray(a,dtype='<f8',order='C');return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes()).hexdigest()
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
 sourcepath=Path('/tmp/oracle_short_direction_source_audit_20260906.json');verify(sourcepath,'88f6c1f6031029a404310b2ce1bc741d85b5f6fca94cc11e8a09f2b68c8f5c5b');source=read(sourcepath);assert source['passed']
 verify('/tmp/oracle_short_direction_prepare.py',source['script_sha256']);assert (root/'unidream/experiments/oracle_short_direction_inputs.py').read_text().rstrip()==Path('/tmp/oracle_short_direction_prepare.py').read_text().split("if __name__=='__main__':")[0].rstrip()
 reg,pre,res=(read(out/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config'];verify(out/'results.json',args.expected_results_sha256)
 assert reg['source_revision']==args.expected_source_revision and res['registration_sha256']==digest(reg)
 cp=root/'configs/oracle_short_direction_decisions_20260906.yaml';verify(cp,reg['config_sha256']);assert yaml.safe_load(cp.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cp.relative_to(root))])).hexdigest()==reg['config_sha256']
 verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==42
 for p,h in cfg['source_bindings'].items():verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+p])).hexdigest()==h
 for key in ('source_artifact_bindings','support','spot_data_proof','um_data_proof','inventory'):assert pre[key]==source[key],key
 assert len(pre['source_artifact_bindings'])==5344 and pre['no_new_fits_statistics_predictions_losses_or_orders'] is True
 for p,h in {**pre['source_artifact_bindings'],**pre['direct_source_bindings']}.items():verify(root/p,h)
 pr,pres=(read(parent/(k+'.json')) for k in ('registration','results'));assert pr['source_revision']=='9b4f6a0e5606831a26a8f2a7c401e05c52d41f6f' and digest(pr)==pres['registration_sha256']
 folds=tuple(range(5,13));rules=('utility_risk1','utility_risk1_fallback_bh');segments=('interval','evaluation');group='technical_short_both';newmean=group+'_magnitude_soft';models=tuple(group+'_'+w for w in ('ordinary','magnitude'));newids=tuple(newmean+'_'+rule for rule in rules);controls=tuple(pr['config']['control_ids'])+tuple(pr['config']['new_policy_ids']);policies=controls+newids
 assert len(controls)==80 and cfg['control_ids']==list(controls) and cfg['new_mean_ids']==[newmean] and cfg['new_policy_ids']==list(newids) and cfg['model_ids']==list(models)
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};prows={(r['fold'],r['candidate_id']):r for r in pres['rows']}
 assert len(rows)==len(res['rows'])==656 and set(rows)=={(f,c) for f in folds for c in policies}
 for field,idkey,n in [('scores','mean_id',400),('classification_scores','classifier_id',192)]:
  current={(r['fold'],r['segment'],r[idkey]):r for r in res[field]};assert len(current)==len(res[field])==n
  for r in pres[field]:assert current[r['fold'],r['segment'],r[idkey]]==r;counts['exact_old_'+field]+=1
 assert len(res['fit_records'])==8 and sum(r['new_model_fits'] for r in res['fit_records'])==16
 assert len(res['direction_diagnostics'])==32 and len(res['mapping_diagnostics'])==16
 own={}
 for f in folds:
  doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
  for k in ('rows','scores','classification_scores','direction_diagnostics','mapping_diagnostics'):assert doc[k]==[r for r in res[k] if r['fold']==f]
  specs=[('models',models,'joblib'),('fit_data',('training',),'npz'),('provenance',('fit','mapping'),'json'),('probabilities_interval',models,'npz'),('probabilities_evaluation',models,'npz'),('calibration',(newmean,),'npz'),('forecasts',(newmean,),'npz'),('targets',policies,'npz'),('traces',newids,'json')]
  expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in specs for v in vs};assert len(expected)==95 and set(doc['artifact_sha256'])==expected
  for p,h in doc['artifact_sha256'].items():assert p not in own;verify(root/p,h);own[p]=h
 assert len(own)==760
 inventory={'models':16,'fit_data':8,'provenance':16,'probabilities_interval':16,'probabilities_evaluation':16,'calibration':8,'forecasts':8,'targets':656,'traces':16}
 for k,n in inventory.items():assert len(list((out/k).iterdir()))==n
 execution=cfg['execution'];assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
 sp=Path('/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/checkpoints/alpha_dd_data/spot_15m.parquet');verify(sp,pre['spot_data_proof']['artifact_sha256']);cut=pd.Timestamp(cfg['data_cutoff']);assert cut==pd.Timestamp('2023-04-16T13:45:00Z')
 bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC'));bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
 fkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};nfkeys=fkeys|{'parent_mu','logit','probability'};ckeys={'timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','mu','parent_mu','logit','probability','fit_return_mean'};pkeys={'timestamps','logit','probability','predict_mask','mapped_inference_mask','score_support'}
 totals=Counter();regimes=Counter();hindsight={c for c in controls if prows[5,c]['hindsight_only']};assert len(hindsight)==24
 print(json.dumps({'phase':'source_inventory_verified','ancestor_artifacts':5344,'own_artifacts':760}),flush=True)
 for f in folds:
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3)
  ix=pd.date_range(E,end,freq='15min',inclusive='left');ci=pd.date_range(S,E,freq='15min',inclusive='left');frame=bars.loc[ix];ref=arr(parity/'forecasts'/f'fold{f}_technical_half.npz',fkeys);exact('E calendar',ref['timestamps'],ix.asi8)
  inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
  assert inf.dtype==bool and score.dtype==bool and not(score&~inf).any() and not(inf&~clock).any() and not(inf&~known).any() and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
  support=next(r for r in pre['support'] if r['fold']==f);scalars=support['saved_mapping_scalars'];oldfit=read(s17/'provenance'/f'fold{f}_fit.json');fit=read(out/'provenance'/f'fold{f}_fit.json');assert fit==next(r for r in res['fit_records'] if r['fold']==f)
  assert fit['fit_source_binding']==support and fit['fit_labels_weights_priors_and_scalars_match_parent_exactly'] is True
  for key in ('fit_priors','fit_return_mean','fit_abs_return_mean'):assert fit[key]==oldfit[key]
  pack=arr(out/'fit_data'/f'fold{f}_training.npz');oldpack=arr(s17/'fit_data'/f'fold{f}_training.npz');assert set(pack)==set(oldpack)|{'fit_features_'+group,'predict_features_'+group}
  for k in oldpack:exact('retained training snapshot '+k,pack[k],oldpack[k])
  for name in ('fit','predict'):
   x=pack[name+'_features_'+group];assert x.shape==(len(oldpack[name+'_positions']),37) and np.isfinite(x).all();exact('selected original29 '+name,x[:,:29],oldpack[name+'_features_technical']);assert matrixsha(x)==support['short_'+name+'_features_sha256']==fit['fit_provenance'][name+'_features_sha256'][group]
  counts['saved_selected_fit_predict37_and_old_pack_verified']+=1
  saved=None
  for segment,folder,times,maskkey in [('evaluation','forecasts',ix,'inference_mask'),('interval','calibration',ci,'mapped_inference_mask')]:
   original=arr(s17/folder/f'fold{f}_technical_magnitude_direction.npz',nfkeys if segment=='evaluation' else ckeys);a=arr(out/folder/f'fold{f}_{newmean}.npz',set(original));mask=a[maskkey]
   exact('prediction calendar '+segment,a['timestamps'],times.asi8)
   for k in a:
    if k not in ('mu','logit','probability'):exact('unchanged paired '+segment+' '+k,a[k],original[k])
   assert mask.dtype==bool;exact('mapped finite support '+segment,np.isfinite(a['mu']),mask)
   expected=scalars['fit_abs_return_mean']*(2.0*a['probability'][mask]-1.0);exact('independent soft mapping '+segment,a['mu'][mask],expected)
   predmask=None
   for mid in models:
    ps=arr(out/('probabilities_'+segment)/f'fold{f}_{mid}.npz',pkeys);exact('probability calendar',ps['timestamps'],times.asi8);exact('mapped support',ps['mapped_inference_mask'],mask)
    pm=ps['predict_mask'];assert pm.dtype==bool and not(mask&~pm).any();assert np.isfinite(ps['logit'][pm]).all() and np.isfinite(ps['probability'][pm]).all() and np.isnan(ps['logit'][~pm]).all() and np.isnan(ps['probability'][~pm]).all() and ((ps['probability'][pm]>=0)&(ps['probability'][pm]<=1)).all()
    exact('probability score support',ps['score_support'],a['score_support'] if segment=='evaluation' else a['interval_mask'])
    if mid.endswith('magnitude'):
     exact('mapped classifier logits',a['logit'],ps['logit']);exact('mapped classifier probabilities',a['probability'],ps['probability'])
    if predmask is not None:exact('same classifier predict mask',pm,predmask)
    predmask=pm;counts['new_probability_support_bindings']+=1
   if segment=='evaluation':saved=a;exact('unchanged risk',a['variance'],ref['variance'])
   else:
    assert not(mask&np.asarray(ci<I)).any();exact('all I prediction support',mask,a['classifier_predict_mask']&np.asarray(ci>=I));assert not(a['interval_mask']&~mask).any() and (ci[a['interval_mask']]+pd.Timedelta(minutes=375)<E).all()
    totals['interval_inference']+=int(mask.sum());totals['interval_score']+=int(a['interval_mask'].sum())
   counts['new_mapped_forecast_source_and_support_bindings']+=1
  for cid in policies:
   row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'] and row['hindsight_only'] is(cid in hindsight)
   p=out/'targets'/f'fold{f}_{cid}.npz';verify(p,row['targets_sha256']);a=arr(p,{'timestamps','targets'});exact('target calendar',a['timestamps'],ix.asi8);target=a['targets'];finite=np.isfinite(target);assert not(finite&~clock).any() and ((target[finite]>=0)&(target[finite]<=1.12)).all()
   if cid in controls:
    old=arr(parent/'targets'/f'fold{f}_{cid}.npz',{'timestamps','targets'});exact('old targets',target,old['targets']);assert row==prows[f,cid];counts['exact_complete_old_control_rows']+=1
   for cost,mul in [('base',1),('stress_2x',2)]:
    contract=dict(execution);contract['one_way_cost']*=mul;contract['borrow_annual']*=mul;got,_,_=scalar_account(frame,target,contract);assert set(got)==set(row[cost])
    for k,v in got.items():close('account_'+k,v,row[cost][k])
    counts['independent_scalar_accounts']+=1
   if cid not in newids:continue
   isfallback=cid.endswith(rules[1]);_,expected,tr=scalar_account(frame,target,execution,saved,inf,fallback_enabled=isfallback);exact('independent ownstate targets',target,expected)
   allowed=learned|fallback if isfallback else learned;assert not(finite&~allowed).any()
   if isfallback:assert np.all(target[fallback]==1.)
   counts['new_own_state_paths']+=1;counts['new_own_state_decisions']+=len(tr)
   tp=out/'traces'/f'fold{f}_{cid}.json';verify(tp,row['trace_sha256']);trace=read(tp);assert trace['metrics']==row['base']
   for k in ('future_information_used_for_decisions','hindsight_only','teacher_actions_used','global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed'):assert trace[k] is False
   assert trace['short_direction_mapping']=={'mean_id':newmean,'formula':cfg['surrogate_mean'],'saved_T_scalars':scalars,'future_labels_used_for_orders':False,'probability_calibration':False,'risk_fits':0}
   assert trace['canonical_replay_verified'] is True and trace['risk_aversion']==1 and trace['cost_multiplier']==2 and trace['horizon_bars']==24 and trace['decision_cadence_hours']==6 and trace['execution_delay_bars']==1
   st=trace['decision_trace'];assert st['bar_indices']==[x[0] for x in tr] and 'hindsight_information_replaced' not in st
   for col,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:close('trace_'+key,[x[col] for x in tr],st[key])
   assert trace['valid_decision_count']==int(learned.sum()) and trace['missing_open_decision_count']==int(missing.sum()) and trace['scheduled_decision_count']==int(clock.sum())
   if isfallback:
    assert trace['fallback_decision_count']==int(fallback.sum());close('trace_targets',[x[5] for x in tr],st['targets']);assert st['reasons']==[x[6] for x in tr]
    for k,mask in [('learned',learned),('fallback',fallback),('missing_open',missing),('hold',learned&np.isnan(target))]:exact('decision mask '+k,np.asarray(trace['decision_masks'][k]),mask)
    for j,x in enumerate(tr):
     if x[6]=='forecast_unavailable':assert st['targets'][j]==1. and st['estimated_utility_gain_over_hold'][j] is None and st['estimated_trade_turnover'][j] is None
    counts['new_fallback_decisions']+=int(fallback.sum())
   counts['new_full_trace_replays']+=1;counts['new_unscored_E_decisions_retained']+=int((inf&~score).sum())
  regimes[prows[f,'bh']['regime']['trend']]+=1
  print(json.dumps({'phase':'fold_scalar_audit_complete','fold':f,'accounts':164,'new_paths':2}),flush=True)
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2,'interval_inference':2537,'interval_score':2523} and regimes=={'bull':2,'bear':4,'sideways':2}
 expectedcounts={'exact_old_scores':384,'exact_old_classification_scores':160,'exact_complete_old_control_rows':640,'independent_scalar_accounts':1312,'new_own_state_paths':16,'new_own_state_decisions':5504,'new_full_trace_replays':16,'new_unscored_E_decisions_retained':24,'new_fallback_decisions':332,'saved_selected_fit_predict37_and_old_pack_verified':8,'new_probability_support_bindings':32,'new_mapped_forecast_source_and_support_bindings':16}
 for k,n in expectedcounts.items():assert counts[k]==n,(k,counts[k],n)
 for k in ('selection_performed','teacher_use_allowed','additional_test_used_for_modeling_or_scoring','high_probability_generalization_established'):assert res[k] is False
 for k,n in {'new_model_fits':16,'new_unique_priors':0,'new_causal_policy_names':2,'total_adaptively_explored_causal_names':220,'risk_model_or_calibration_fits':0}.items():assert res[k]==n
 log=Path('/tmp/oracle-short-direction-run.log');warnings=Counter(line.split('RuntimeWarning: ',1)[1] for line in log.read_text().splitlines() if 'RuntimeWarning: ' in line)
 report={'status':'pass','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_audit_sha256':sha(sourcepath),'ancestor_artifacts_verified':5344,'own_artifacts_verified':760,'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maximum,'support':dict(totals),'regime_counts':dict(regimes),'inventory':{**inventory,'rows':656,'return_scores':400,'classification_scores':192,'direction_diagnostics':32,'mapping_diagnostics':16},'runtime_warnings':{'run_log_sha256':sha(log),'counts':dict(warnings),'total':sum(warnings.values())},'scope':'All 5344 ancestral+760 new immutable artifacts and42 frozen sources. All1312 scalar accounts,16 new own-state paths/full traces,16 mapped E/I arrays,32 probability/support bindings;640 exact old full rows including trace hashes;384/160 exact old return/classifier scores;8 selected37 fit/predict matrices and complete original selected pack equality. No canonical model/planner/simulator/scorer helper imported; no fit performed.','separate_audit_scope':['Fitted coefficient/objective/gradient/logit numerical accuracy belongs to the independent model auditor.','New classification/return losses and summary decisions belong to the statistics auditor.'],'limitations':['Source input module differs from the independent /tmp bridge only by trailing whitespace; executable body equality is checked.','Repeatedly reused development and retrospective availability masks are not independent confirmation.','Filtered scalar price reads are below original cutoff; full file bytes are hashed.','Archived sources do not establish historical receipt timestamps.']}
 path=Path('/tmp/oracle_short_direction_audit_20260906.json');path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'pass','path':str(path),'sha256':sha(path),'script_sha256':sha(__file__),'counts':dict(counts),'maximum':maximum,'warnings':dict(warnings)}),flush=True)
if __name__=='__main__':main()
