"""Stage18 independent immutable-source, mapping, own-state and accounting audit.
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
 root=Path.cwd();out=root/'codex_outputs/oracle_regularized_direction_decisions_v1';parent=root/'codex_outputs/oracle_direction_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1'
 assert (out/'results.json').is_file(),'Completed result required before new mapped forecasts or orders are audited.'
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
   if keys is not None:assert set(z.files)==set(keys),('schema',str(p),z.files)
   a={k:z[k] for k in z.files}
  assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid',str(p))
  return a
 def exact(name,a,b):
  a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),('exact',name)
 def close(name,a,b,tol=1e-10):
  a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('invalid',name)
  finite=np.isfinite(a);e=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.;maximum[name]=max(maximum.get(name,0.),e);assert e<=tol,('numerical',name,e,tol)
 sourcepath=Path('/tmp/oracle_regularized_direction_source_audit_20260906.json');verify(sourcepath,'43d255bbc8e361d8ff9e78bbf308b9c1a476e90ee1442584ff0f678084ed69e7');source=read(sourcepath)
 verify('/tmp/oracle_regularized_direction_source_audit_20260906.py','6dc052ebba18c5f9d9cf81ae3a44911394359baad731ea9368f880f542ca06f4');assert source['passed'] and source['counts']['total_source_artifacts']==2840
 for p,h in {**source['source_artifact_bindings'],**source['direct_source_bindings'],**source['saved_input_bindings']}.items():verify(root/p,h)
 reg,pre,res=(read(out/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config'];verify(out/'results.json',args.expected_results_sha256)
 assert reg['source_revision']==args.expected_source_revision and res['registration_sha256']==digest(reg)
 cfgpath=root/'configs/oracle_regularized_direction_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
 verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==31
 for p,h in cfg['source_bindings'].items():
  verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+p])).hexdigest()==h
 assert pre['source_artifact_bindings']==source['source_artifact_bindings']
 assert pre['new_class_statistics_fits_logits_mapped_predictions_losses_or_orders_computed'] is False and pre['inputs_from_hash_bound_original_selected_snapshots'] is True
 for p,h in pre['direct_source_bindings'].items():verify(root/p,h)
 pr,pres,pp=(read(parent/(k+'.json')) for k in ('registration','results','preflight'))
 assert digest(pr)==pres['registration_sha256'] and pr['source_revision']==source['parent_source_revision']=='6ae673fcdfeed29280256450c05eb8905af77ee3'
 assert pre['spot_data_proof']==source['spot_data_proof'] and pre['um_data_proof']==source['um_data_proof']
 groups=('technical','perp_delay0');halves=tuple(g+'_half' for g in groups);weightings=('ordinary','magnitude');rules=('utility_risk1','utility_risk1_fallback_bh');folds=tuple(range(5,13));segments=('interval','evaluation')
 oldmodelids=tuple(g+'_'+w for g in groups for w in weightings);oldmeans=tuple(pr['config']['new_mean_ids']);controls=tuple(pr['config']['control_ids'])+tuple(pr['config']['new_policy_ids']);assert len(controls)==52
 modelids=tuple(cfg['model_ids']);newmeans=tuple(cfg['new_mean_ids']);newids=tuple(cfg['new_policy_ids']);mapping=cfg['mapping'];policies=controls+newids
 assert modelids==tuple(m+'_l2unit' for m in oldmodelids) and newmeans==tuple(m+'_direction' for m in modelids)
 assert len(modelids)==len(set(modelids))==4 and len(newmeans)==len(set(newmeans))==4 and len(newids)==len(set(newids))==8 and cfg['control_ids']==list(controls) and newids==tuple(m+'_'+r for m in newmeans for r in rules)
 assert {(mapping[m]['group'],mapping[m]['weighting']) for m in newmeans}=={(g,w) for g in groups for w in weightings}
 for m in newmeans:assert mapping[m]['parent_mean']==mapping[m]['group']+'_half'
 assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['regularization_C']=='1.0/float(np.sum(frozen_fit_weights))'
 for k,n in {'new_model_fits':32,'new_unique_fit_priors':0,'new_causal_names':8,'adaptive_prior_causal_names':190,'adaptive_total_causal_names':198,'score_classification_records':160,'score_return_records':224,'direction_diagnostic_records':64,'fold_artifacts':81,'economic_rows':480,'economic_accounts':960}.items():assert cfg[k]==n
 for k in ('selection_permitted','teacher_use_allowed','additional_test_permitted'):assert cfg[k] is False
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};prows={(r['fold'],r['candidate_id']):r for r in pres['rows']}
 fits={r['fold']:r for r in res['fit_records']};assert len(fits)==len(res['fit_records'])==8 and set(fits)==set(folds)
 score_records={(r['fold'],r['segment'],r['mean_id']):r for r in res['scores']};classifier_records={(r['fold'],r['segment'],r['classifier_id']):r for r in res['classification_scores']};diag={(r['fold'],r['segment'],r['classifier_id']):r for r in res['direction_diagnostics']}
 assert len(score_records)==len(res['scores'])==224 and set(score_records)=={(f,seg,m) for f in folds for seg in segments for m in cfg['return_score_means']}
 assert len(classifier_records)==len(res['classification_scores'])==160 and set(classifier_records)=={(f,seg,m) for f in folds for seg in segments for m in cfg['classifiers']}
 assert len(diag)==len(res['direction_diagnostics'])==64 and set(diag)=={(f,seg,m) for f in folds for seg in segments for m in modelids}
 assert len(rows)==len(res['rows'])==480 and set(rows)=={(f,c) for f in folds for c in policies}
 oldhindsight={c for c in controls if prows[5,c]['hindsight_only']};assert len(oldhindsight)==24
 own={}
 for f in folds:
  doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
  for k in ('rows','scores','classification_scores','direction_diagnostics'):assert doc[k]==[r for r in res[k] if r['fold']==f]
  expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('models',modelids,'joblib'),('provenance',('fit',),'json'),('forecasts',newmeans,'npz'),('calibration',newmeans,'npz'),('targets',policies,'npz'),('traces',newids,'json')] for v in vs}
  assert len(expected)==81 and set(doc['artifact_sha256'])==expected
  for p,h in doc['artifact_sha256'].items():verify(root/p,h);own[p]=h
 assert len(own)==648
 inventory={'models':32,'provenance':8,'forecasts':32,'calibration':32,'targets':480,'traces':64}
 for k,n in inventory.items():assert len(list((out/k).iterdir()))==n
 assert not (out/'fit_data').exists()
 # All old predictive records must remain exact, independently of new fit quality.
 for key,idkey in [('scores','mean_id'),('classification_scores','classifier_id')]:
  current={(r['fold'],r['segment'],r[idkey]):r for r in res[key]};assert len(current)==len(res[key])
  for r in pres[key]:assert current[r['fold'],r['segment'],r[idkey]]==r;counts['exact_old_'+key]+=1
 fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());execution=fc['execution'];assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
 sp=Path(fc['data_path']);verify(sp,pre['spot_data_proof']['artifact_sha256']);cut=pd.Timestamp(cfg['data_cutoff']);bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC'));bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
 fkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};newfkeys=fkeys|{'parent_mu','logit','probability'};calk={'timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','mu','parent_mu','logit','probability','fit_return_mean'};totals=Counter();regimes=Counter()
 def sigmoid(z):return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
 def sgn(z):return 1. if z>0 else -1. if z<0 else 0.
 print(json.dumps({'phase':'all_immutable_bindings_verified','ancestor_artifacts':2840,'own_artifacts':648}),flush=True)
 for f in folds:
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3)
  ix=pd.date_range(E,end,freq='15min',inclusive='left');cix=pd.date_range(S,E,freq='15min',inclusive='left');frame=bars.loc[ix];saved={h:arr(parity/'forecasts'/f'fold{f}_{h}.npz',fkeys) for h in halves};ref=saved[halves[0]];exact('calendar',ref['timestamps'],ix.asi8)
  inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
  assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any() and not (inf&~known).any() and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
  sourcefold=next(r for r in source['support'] if r['fold']==f);fd=arr(parent/'fit_data'/f'fold{f}_training.npz')
  assert digest(fd['returns'].tolist())==sourcefold['continuous_fit_return_sha256'] and matrixsha(fd['binary_labels'])==sourcefold['binary_fit_labels_sha256']
  for w in weightings:assert matrixsha(fd['weights_'+w])==sourcefold['saved_weight_sha256'][w]
  counts['immutable_original_fit_inputs_rebound']+=1
  fit=read(out/'provenance'/f'fold{f}_fit.json');oldfit=read(parent/'provenance'/f'fold{f}_fit.json');assert fit==fits[f]
  assert fit['fit_source_binding']==next(q for q in pre['support'] if q['fold']==f) and fit['frozen_fit_data_reused'] is True and fit['fit_labels_weights_and_priors_match_parent_exactly'] is True
  assert fit['new_unique_prior_estimates']==0 and fit['new_model_fits']==4 and fit['fit_priors']==oldfit['fit_priors'] and fit['fit_prior_logits']==oldfit['fit_prior_logits']
  prov=fit['fit_provenance'];oldprov=oldfit['fit_provenance']
  for key in ('feature_columns','feature_counts','index_sha256','mask_counts','mask_ranges','mask_position_sha256','fit_return_sha256','fit_binary_labels_sha256','fit_class_counts','fit_features_sha256','predict_features_sha256','fit_features_and_return_sha256','sample_weights'):assert prov[key]==oldprov[key],('unchanged provenance',f,key)
  assert prov['model_selection_performed'] is False and prov['evaluation_labels_used'] is False and prov['risk_or_calibration_fitted'] is False and prov['regularization']['search_performed'] is False
  assert set(fit['model_state'])==set(modelids) and set(prov['fitted_state'])==set(oldmodelids)
  for oldmid,newmid in zip(oldmodelids,modelids):
   state=prov['fitted_state'][oldmid];modelstate=fit['model_state'][newmid];oldstate=oldprov['fitted_state'][oldmid];regularization=state['regularization']
   assert modelstate['unchanged_scaler_exact'] is True and modelstate['only_C_setting_changed'] is True and modelstate['old_model_path']==str((parent/'models'/f'fold{f}_{oldmid}.joblib').relative_to(root))
   for key in ('C','solver_weight_sum','actual_l2_strength'):assert modelstate[key]==regularization[key]
   for key in ('group','weighting','scaler_mean','scaler_variance','scaler_scale','scaler_rows','classes','fit_features_labels_weights_sha256'):assert state[key]==oldstate[key]
   assert state['logistic_parameters']=={**oldprov['parameters']['logistic'],'C':modelstate['C']}
   v=state['scalar_verification'];assert v['checked'] is True and v['normalized_gradient_infinity']<=1e-6 and v['max_abs_logit_difference']<=1e-12 and v['max_abs_probability_difference']<=1e-14
   assert v['l2_gradient_strength']==regularization['actual_l2_strength'] and regularization['weight_sha256']==sourcefold['saved_weight_sha256'][state['weighting']]
   assert all(math.isfinite(float(x)) for x in v['normalized_gradient']) and math.isfinite(v['normalized_objective']);counts['bound_model_verification_records']+=1
  for m in newmeans:
   mp=mapping[m];g,w=mp['group'],mp['weighting'];base=saved[g+'_half'];oldname=g+'_'+w+'_direction';old=arr(parent/'forecasts'/f'fold{f}_{oldname}.npz',newfkeys);oldca=arr(parent/'calibration'/f'fold{f}_{oldname}.npz',calk)
   a=arr(out/'forecasts'/f'fold{f}_{m}.npz',newfkeys);ca=arr(out/'calibration'/f'fold{f}_{m}.npz',calk);saved[m]=a
   for k in newfkeys-{'mu','logit','probability'}:exact('unchanged E '+k,a[k],old[k])
   for k in calk-{'mu','logit','probability'}:exact('unchanged I '+k,ca[k],oldca[k])
   exact('frozen E magnitude source',a['parent_mu'],base['mu']);exact('E inference',np.isfinite(a['logit']),inf);exact('E probability availability',np.isfinite(a['probability']),inf)
   expected=np.full(len(ix),np.nan)
   for j in np.flatnonzero(inf):expected[j]=sgn(float(a['logit'][j]))*abs(float(base['mu'][j]))
   exact('independent E direction mapping',a['mu'],expected);exact('new availability',np.isfinite(a['mu']),inf);close('saved E probability sigmoid',a['probability'][inf],[sigmoid(float(z)) for z in a['logit'][inf]],1e-14)
   exact('calibration times',ca['timestamps'],cix.asi8);pred=ca['classifier_predict_mask'];mapped=ca['mapped_inference_mask'];assert pred.dtype==mapped.dtype==bool;exact('I mapping mask',mapped,pred&np.asarray(cix>=I));assert not (ca['interval_mask']&~mapped).any()
   exact('cal logit mask',np.isfinite(ca['logit']),pred);exact('cal probability mask',np.isfinite(ca['probability']),pred);exact('cal mapped availability',np.isfinite(ca['mu']),mapped)
   expectedcal=np.full(len(cix),np.nan)
   for j in np.flatnonzero(mapped):expectedcal[j]=sgn(float(ca['logit'][j]))*abs(float(oldca['parent_mu'][j]))
   exact('independent I direction mapping',ca['mu'],expectedcal);close('saved cal probability sigmoid',ca['probability'][pred],[sigmoid(float(z)) for z in ca['logit'][pred]],1e-14)
   for segment,support in [('interval',ca['interval_mask']),('evaluation',score)]:
    assert score_records[f,segment,m]['rows']==int(support.sum()) and classifier_records[f,segment,mp['classifier_id']]['rows']==int(support.sum())
   for segment,support in [('interval',mapped),('evaluation',inf)]:
    d=diag[f,segment,mp['classifier_id']];assert d['rows']==int(support.sum()) and d['uses_all_inference_not_score_support'] is True
    for key,value in fit['model_state'][mp['classifier_id']].items():assert d[key]==value
   counts['new_E_direction_forecasts']+=1;counts['new_calibration_direction_arrays']+=1
  for cid in policies:
   row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'] and row['hindsight_only'] is (cid in oldhindsight)
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
   if cid not in newids:continue
   rule=rules[1] if cid.endswith(rules[1]) else rules[0];mean=cid[:-(len(rule)+1)];isfallback=rule==rules[1]
   _,expected,tr=scalar_account(frame,target,execution,saved[mean],inf,fallback_enabled=isfallback);exact('independent own-state targets',target,expected)
   allowed=learned|fallback if isfallback else learned;assert not (finite&~allowed).any()
   if isfallback:assert np.all(target[fallback]==1.)
   counts['new_own_state_paths']+=1;counts['new_own_state_decisions']+=len(tr)
   tp=out/'traces'/f'fold{f}_{cid}.json';verify(tp,row['trace_sha256']);trace=read(tp);assert trace['metrics']==row['base']
   for k in ('future_information_used_for_decisions','hindsight_only','teacher_actions_used','global_optimum_claimed','bayes_optimum_claimed','drawdown_optimum_claimed'):assert trace[k] is False
   assert trace['direction_mapping']=={**mapping[mean],'C_schedule':cfg['regularization_C'],'surrogate_mean':cfg['surrogate_mean'],'future_labels_used_for_orders':False}
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
   counts['new_full_trace_replays']+=1;counts['new_unscored_decisions_retained']+=int((inf&~score).sum())
  regimes[prows[f,'bh']['regime']['trend']]+=1
  print(json.dumps({'phase':'fold_scalar_audit_complete','fold':f,'accounts':120,'new_paths':8}),flush=True)
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2} and regimes=={'bull':2,'bear':4,'sideways':2}
 expectedcounts={'bound_model_verification_records':32,'exact_old_scores':160,'exact_old_classification_scores':96,'immutable_original_fit_inputs_rebound':8,'new_E_direction_forecasts':32,'new_calibration_direction_arrays':32,'exact_old_controls':416,'independent_scalar_accounts':960,'new_own_state_paths':64,'new_own_state_decisions':22016,'new_full_trace_replays':64,'new_unscored_decisions_retained':96,'new_fallback_decisions':1328}
 for k,n in expectedcounts.items():assert counts[k]==n,(k,counts[k],n)
 for k in ('selection_performed','teacher_use_allowed','additional_test_used_for_modeling_or_scoring','high_probability_generalization_established'):assert res[k] is False
 for k,n in {'new_model_fits':32,'new_unique_priors':0,'new_causal_policy_names':8,'total_adaptively_explored_causal_names':198,'risk_model_or_calibration_fits':0}.items():assert res[k]==n
 report={'status':'pass','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_audit_sha256':sha(sourcepath),'ancestor_artifacts_verified':2840,'own_artifacts_verified':648,'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maximum,'support':dict(totals),'regime_counts':dict(regimes),'inventory':{**inventory,'rows':480,'return_scores':224,'classification_scores':160,'direction_diagnostics':64},'scope':'All immutable sources/artifacts;32 E and32 I mappings from saved logits and fixed parent magnitudes; unchanged risk/support;416 exact source controls;960 scalar accounts;64 new own-state paths and full decision traces. No canonical helpers imported or fits performed.','separate_audit_scope':['The 32 coefficients, fixed C=1/sum(frozen weights) values, normalized scalar gradients and predictor numerical verification are independently audited by the technical auditor.','New return/classification losses and summary arithmetic are independently audited by the statistics auditor.'],'limitations':['Original repeatedly reused development, retrospective masks and absent historical receipt evidence remain.','Mapped sign times fixed magnitude is a return surrogate; magnitude-weighted probability targets a tilted distribution.','Filtered raw Spot reads remain below original cutoff; source hashes cover full files.']}
 path=Path('/tmp/oracle_regularized_direction_audit_20260906.json');path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':'pass','path':str(path),'counts':dict(counts),'maximum':maximum}),flush=True)
if __name__=='__main__':main()
