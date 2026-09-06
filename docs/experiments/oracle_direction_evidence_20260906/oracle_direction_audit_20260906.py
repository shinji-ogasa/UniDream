"""Stage17 independent immutable-source, mapping, own-state and accounting audit.
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
 root=Path.cwd();out=root/'codex_outputs/oracle_direction_decisions_v1';parent=root/'codex_outputs/oracle_sign_magnitude_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1'
 assert (out/'results.json').is_file(),'Completed result required before new fitted statistics or orders are audited.'
 def read(p):return json.loads(Path(p).read_text())
 def sha(p):
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  return h.hexdigest()
 def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 def matrix_digest(a):
  a=np.asarray(a,dtype='<f8',order='C');return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes(order='C')).hexdigest()
 def mask_digest(index,mask):return hashlib.sha256(index.asi8.astype('<i8').tobytes()+np.asarray(mask,'u1').tobytes()).hexdigest()
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
 sourcepath=Path('/tmp/oracle_direction_source_audit_20260906.json');verify(sourcepath,'a64268eecfa8330440792aad5cec40c0a62ef450d9ac738fc0f41729edcfdf32');source=read(sourcepath)
 verify('/tmp/oracle_direction_source_audit_20260906.py','87d3bf94d2b1aeffc6118683746de32f3847c91ca31987e4d657f4ee514e6b45');assert source['passed'] and source['counts']['total_source_artifacts']==2120
 for p,h in {**source['source_artifact_bindings'],**source['direct_source_bindings'],**source['original_forecast_bindings']}.items():verify(root/p,h)
 reg,pre,res=(read(out/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config'];verify(out/'results.json',args.expected_results_sha256)
 assert reg['source_revision']==args.expected_source_revision and res['registration_sha256']==digest(reg)
 cfgpath=root/'configs/oracle_direction_decisions_20260906.yaml';verify(cfgpath,reg['config_sha256']);assert yaml.safe_load(cfgpath.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+str(cfgpath.relative_to(root))])).hexdigest()==reg['config_sha256']
 verify(out/'preflight.json',cfg['preflight_sha256']);assert reg['preflight_sha256']==cfg['preflight_sha256'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==29
 for p,h in cfg['source_bindings'].items():
  verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',reg['source_revision']+':'+p])).hexdigest()==h
 assert pre['source_artifact_bindings']==source['source_artifact_bindings']
 for p,h in pre['direct_source_bindings'].items():verify(root/p,h)
 assert pre['new_class_labels_priors_weights_fits_logits_mapped_predictions_or_orders_computed'] is False and pre['original_I_half_magnitudes_reconstructed_from_frozen_S_calibration'] is True
 pr,pres,pp=(read(parent/(k+'.json')) for k in ('registration','results','preflight'))
 assert digest(pr)==pres['registration_sha256'] and cfg['parent_source_revision']==pr['source_revision']==source['parent_source_revision']
 assert pre['spot_data_proof']==source['spot_data_proof'] and pre['um_data_proof']==source['um_data_proof']
 for p,h in cfg['parent_manifest_bindings'].items():verify(root/p,h)
 groups=('technical','perp_delay0');halves=tuple(g+'_half' for g in groups);weightings=('ordinary','magnitude');rules=('utility_risk1','utility_risk1_fallback_bh');folds=tuple(range(5,13));segments=('interval','evaluation')
 modelids=tuple(g+'_'+w for g in groups for w in weightings);priorids=tuple('prior_'+w for w in weightings);classifiers=modelids+priorids
 learnedmeans=tuple(m+'_direction' for m in modelids);priormeans=tuple(g+'_'+w+'_prior_direction' for g in groups for w in weightings);newmeans=learnedmeans+priormeans;means=halves+newmeans
 mapping={g+'_'+w+suffix:{'group':g,'weighting':w,'classifier_id':('prior_'+w if suffix=='_prior_direction' else g+'_'+w),'parent_mean':g+'_half','prior_mean':g+'_'+w+'_prior_direction'} for suffix in ('_direction','_prior_direction') for g in groups for w in weightings}
 basemeans=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half');causal=('bh','common_robust')+tuple(m+'_'+r for m in basemeans for r in rules)
 oldhybrids=tuple(h+'_oracle_'+c for h in halves for c in ('return','realized_risk','both'));rl=tuple(f'matched_rl_beam32_{r}_risk{k}' for r in ('hold','fallback_bh') for k in (0,1));hindsight=tuple(m+'_'+r for m in oldhybrids for r in rules)+rl+tuple(h+'_oracle_'+c+'_'+r for h in halves for c in ('sign','magnitude') for r in rules)
 controls=causal+hindsight;newids=tuple(m+'_'+r for m in newmeans for r in rules);policies=controls+newids
 for k,v in {'groups':groups,'weightings':weightings,'model_ids':modelids,'classifiers':classifiers,'new_mean_ids':newmeans,'return_score_means':means,'control_ids':controls,'new_policy_ids':newids,'rules':rules,'segments':segments}.items():assert cfg[k]==list(v),(k,cfg[k],v)
 assert cfg['mapping']==mapping and cfg['data_cutoff']=='2023-04-16T13:45:00Z'
 for k in ('selection_permitted','teacher_use_allowed','additional_test_permitted','probability_calibration_permitted','mean_risk_or_weight_calibration_permitted'):assert cfg[k] is False
 for k,v in {'new_model_fits':32,'fit_prior_probabilities':16,'new_causal_names':16,'adaptive_prior_causal_names':174,'adaptive_total_causal_names':190,'normalized_gradient_infinity_bound':1e-6}.items():assert cfg[k]==v
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(s['fold'],s['mean_id'],s['segment']):s for s in res['scores']};cs={(s['fold'],s['classifier_id'],s['segment']):s for s in res['classification_scores']};fits={r['fold']:r for r in res['fit_records']};prows={(r['fold'],r['candidate_id']):r for r in pres['rows']}
 assert len(rows)==len(res['rows'])==416 and set(rows)=={(f,c) for f in folds for c in policies}
 assert len(scores)==len(res['scores'])==160 and set(scores)=={(f,m,s) for f in folds for m in means for s in segments}
 assert len(cs)==len(res['classification_scores'])==96 and set(cs)=={(f,m,s) for f in folds for m in classifiers for s in segments}
 assert len(fits)==len(res['fit_records'])==8 and set(fits)==set(folds)
 own={}
 for f in folds:
  doc=read(out/f'fold_{f}.json');assert doc['registration_sha256']==digest(reg)
  for k in ('rows','scores','classification_scores'):assert doc[k]==[r for r in res[k] if r['fold']==f]
  expected={str((out/k/f'fold{f}_{v}.{ext}').relative_to(root)) for k,vs,ext in [('models',modelids,'joblib'),('fit_data',('training',),'npz'),('provenance',('fit',),'json'),('forecasts',newmeans,'npz'),('calibration',newmeans,'npz'),('targets',policies,'npz'),('traces',newids,'json')] for v in vs}
  assert len(expected)==90 and set(doc['artifact_sha256'])==expected
  for p,h in doc['artifact_sha256'].items():verify(root/p,h);own[p]=h
 assert len(own)==720
 inventory={'models':32,'fit_data':8,'provenance':8,'forecasts':64,'calibration':64,'targets':416,'traces':128}
 for k,n in inventory.items():assert len(list((out/k).iterdir()))==n
 fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());execution=fc['execution'];assert execution=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
 sp=Path(fc['data_path']);verify(sp,pre['spot_data_proof']['artifact_sha256']);cut=pd.Timestamp(cfg['data_cutoff']);bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);bars=bars.reindex(pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC'));bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1);assert bars.index[-1]<cut
 fkeys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};newfkeys=fkeys|{'parent_mu','logit','probability'};calk={'timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','mu','parent_mu','logit','probability','fit_return_mean'};totals=Counter();regimes=Counter()
 def sigmoid(z):return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
 def sgn(z):return 1. if z>0 else -1. if z<0 else 0.
 print(json.dumps({'phase':'all_immutable_bindings_verified','ancestor_artifacts':2120,'own_artifacts':720}),flush=True)
 for f in folds:
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3);T=E-pd.DateOffset(months=24)
  ix=pd.date_range(E,end,freq='15min',inclusive='left');cix=pd.date_range(S,E,freq='15min',inclusive='left');frame=bars.loc[ix];saved={h:arr(parity/'forecasts'/f'fold{f}_{h}.npz',fkeys) for h in halves};ref=saved[halves[0]];exact('calendar',ref['timestamps'],ix.asi8)
  inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));known=np.isfinite(frame.open.to_numpy());learned=clock&known&inf;fallback=clock&known&~inf;missing=clock&~known
  assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any() and not (inf&~clock).any() and not (inf&~known).any() and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  for k,m in [('inference',inf),('score',score),('fallback',fallback),('missing_current_open',missing)]:totals[k]+=int(m.sum())
  p=next(v for v in pre['support'] if v['fold']==f);s=next(v for v in source['support'] if v['fold']==f)
  for k in p['counts']:assert p['counts'][k]==s['counts'][k] and p['mask_sha256'][k]==s['mask_sha256'][k]
  assert p['fit_return_sha256']==s['continuous_return_selected_sha256']['fit'] and p['feature_columns']==source['features']
  for g in groups:
   assert p['fit_features_sha256'][g]==s['feature_selected_sha256']['fit'][g] and p['predict_features_sha256'][g]==s['feature_selected_sha256']['predict'][g]
  fit=read(out/'provenance'/f'fold{f}_fit.json');assert fit==fits[f] and fit['fit_source_binding']==p and fit['new_model_fits']==4 and fit['shared_prior_estimates']==2
  fd=arr(out/'fit_data'/f'fold{f}_training.npz',{'fit_positions','timestamps','returns','binary_labels','predict_positions','predict_timestamps','fit_features_technical','fit_features_perp_delay0','predict_features_technical','predict_features_perp_delay0','weights_ordinary','weights_magnitude'});ft=pd.DatetimeIndex(pd.to_datetime(fd['timestamps'],utc=True));fp=fd['fit_positions'];assert fp.dtype.kind in 'iu' and len(fp)==p['counts']['fit'] and (np.diff(fp)>0).all() and ft.is_unique and ft.is_monotonic_increasing
  exact('fit positions timestamp',bars.index[fp].asi8,fd['timestamps']);assert (ft>=T).all() and (ft<S).all() and (ft+pd.Timedelta(minutes=375)<S).all() and ((ft.hour%6==0)&(ft.minute==0)).all()
  fullfit=np.zeros(len(bars),dtype=bool);fullfit[fp]=True;assert mask_digest(bars.index,fullfit)==s['mask_sha256']['fit'] and digest(fd['returns'].tolist())==s['continuous_return_selected_sha256']['fit']
  predictpos=fd['predict_positions'];predict_times=pd.DatetimeIndex(pd.to_datetime(fd['predict_timestamps'],utc=True));assert predictpos.dtype.kind in 'iu' and len(predictpos)==p['counts']['predict'] and (np.diff(predictpos)>0).all() and predictpos[0]>fp[-1]
  exact('predict positions timestamp',bars.index[predictpos].asi8,fd['predict_timestamps']);assert (predict_times>=S).all() and (predict_times<end).all() and ((predict_times.hour%6==0)&(predict_times.minute==0)).all()
  saved_fullpredict=np.zeros(len(bars),dtype=bool);saved_fullpredict[predictpos]=True;assert mask_digest(bars.index,saved_fullpredict)==s['mask_sha256']['predict']
  for segment in ('fit','predict'):
   for g in groups:
    x=fd[segment+'_features_'+g];assert x.shape==(p['counts'][segment],len(source['features'][g])) and np.isfinite(x).all()
    assert matrix_digest(x)==s['feature_selected_float64le_sha256'][segment][g] and digest(x.tolist())==s['feature_selected_sha256'][segment][g]
    counts['bound_saved_feature_matrices']+=1
  labels=np.asarray([int(float(v)>0) for v in fd['returns']],dtype=np.int64);exact('fit-only binary labels',fd['binary_labels'],labels)
  absmean=math.fsum(abs(float(v))/len(fp) for v in fd['returns']);close('fit absolute mean',fit['fit_abs_return_mean'],absmean,0)
  provenance=fit['fit_provenance'];assert provenance['evaluation_labels_used'] is False and provenance['risk_or_calibration_fitted'] is False and provenance['feature_columns']==source['features'] and provenance['mask_counts']=={'fit':p['counts']['fit'],'predict':p['counts']['predict']}
  assert provenance['fit_return_sha256']==matrix_digest(fd['returns']) and provenance['fit_binary_labels_sha256']==matrix_digest(labels)
  for segment in ('fit','predict'):
   assert provenance[segment+'_features_sha256']==s['feature_selected_float64le_sha256'][segment]
  for w in weightings:
   weights=np.ones(len(fp)) if w=='ordinary' else np.asarray([abs(float(v))/absmean for v in fd['returns']]);exact('fit-only '+w+' weights',fd['weights_'+w],weights)
   total=math.fsum(float(v) for v in weights);masses=[math.fsum(float(v) for v in weights[labels==c]) for c in (0,1)];pi=masses[1]/total;z=math.log(pi)-math.log1p(-pi)
   close('fit prior',fit['fit_priors'][w],pi,0);close('fit prior logit',fit['fit_prior_logits'][w],z,0);assert provenance['sample_weights'][w]['weight_sha256']==matrix_digest(weights);counts['independent_fit_priors']+=1
  for mid in modelids:
   state=provenance['fitted_state'][mid];v=state['scalar_verification'];assert v['checked'] is True and v['gradient_bound']==1e-6 and v['normalized_gradient_infinity']<=1e-6 and v['max_abs_logit_difference']<=1e-12 and v['max_abs_probability_difference']<=1e-14
   assert all(math.isfinite(float(x)) for x in v['normalized_gradient']) and math.isfinite(v['normalized_objective']);counts['bound_model_verification_records']+=1
  oldcal={g:arr(parity/'calibration'/f'fold{f}_{g}.npz') for g in groups};prov=read(parity/'calibration'/f'fold{f}_provenance.json')['calibration'];classifier_arrays={};cal_classifier_arrays={}
  for h in halves:
   for k in fkeys-{'mu'}:exact('paired base '+k,saved[h][k],ref[k])
  for m in newmeans:
   mp=mapping[m];g=mp['group'];mid=mp['classifier_id'];base=saved[mp['parent_mean']];a=arr(out/'forecasts'/f'fold{f}_{m}.npz',newfkeys);saved[m]=a;ca=arr(out/'calibration'/f'fold{f}_{m}.npz',calk);old=oldcal[g]
   for k in fkeys-{'mu'}:exact('new nonmean '+k,a[k],base[k])
   exact('frozen E magnitude source',a['parent_mu'],base['mu']);exact('E inference',np.isfinite(a['logit']),inf);exact('E probability availability',np.isfinite(a['probability']),inf)
   expected=np.full(len(ix),np.nan)
   for j in np.flatnonzero(inf):expected[j]=sgn(float(a['logit'][j]))*abs(float(base['mu'][j]))
   exact('independent E direction mapping',a['mu'],expected);exact('new availability',np.isfinite(a['mu']),inf)
   close('saved E probability sigmoid',a['probability'][inf],[sigmoid(float(z)) for z in a['logit'][inf]],1e-14)
   if mid in classifier_arrays:
    for k in ('logit','probability'):exact('shared E classifier '+k,a[k],classifier_arrays[mid][k])
   else:classifier_arrays[mid]=a
   exact('calibration times',ca['timestamps'],cix.asi8)
   for k in ('actual','scale_mask','interval_mask'):exact('old calibration '+k,ca[k],old[k])
   pred=ca['classifier_predict_mask'];mapped=ca['mapped_inference_mask'];assert pred.dtype==bool and mapped.dtype==bool;exact('I mapping mask',mapped,pred&np.asarray(cix>=I));assert not (ca['interval_mask']&~mapped).any() and not (mapped&np.asarray(cix<I)).any()
   exact('cal logit mask',np.isfinite(ca['logit']),pred);exact('cal probability mask',np.isfinite(ca['probability']),pred);exact('cal mapped availability',np.isfinite(ca['mu']),mapped)
   fullpred=np.zeros(len(bars),dtype=bool);fullpred[bars.index.get_indexer(cix)]=pred;fullpred[bars.index.get_indexer(ix)]=inf;assert mask_digest(bars.index,fullpred)==s['mask_sha256']['predict']
   expectedparent=np.full(len(cix),np.nan)
   for j in np.flatnonzero(mapped):expectedparent[j]=.5*float(prov['scale_mean'])+.5*(float(old['mu'][j])+float(prov['return_bias'][g]))
   exact('frozen I magnitude reconstruction',ca['parent_mu'],expectedparent)
   expectedcal=np.full(len(cix),np.nan)
   for j in np.flatnonzero(mapped):expectedcal[j]=sgn(float(ca['logit'][j]))*abs(float(expectedparent[j]))
   exact('independent I direction mapping',ca['mu'],expectedcal);close('saved cal probability sigmoid',ca['probability'][pred],[sigmoid(float(z)) for z in ca['logit'][pred]],1e-14)
   if mid in cal_classifier_arrays:
    for k in ('logit','probability'):exact('shared cal classifier '+k,ca[k],cal_classifier_arrays[mid][k])
   else:cal_classifier_arrays[mid]=ca
   if mid in priorids:
    z=fit['fit_prior_logits'][mp['weighting']];exact('E prior constant',a['logit'][inf],np.full(int(inf.sum()),z));exact('cal prior constant',ca['logit'][pred],np.full(int(pred.sum()),z))
   for seg,mask in [('interval',ca['interval_mask']),('evaluation',score)]:assert scores[f,m,seg]['rows']==int(mask.sum()) and cs[f,mid,seg]['rows']==int(mask.sum())
   counts['new_E_direction_forecasts']+=1;counts['new_calibration_direction_arrays']+=1
  for cid in policies:
   row=rows[f,cid];assert row['regime']==prows[f,'bh']['regime'] and row['hindsight_only'] is (cid in hindsight)
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
   assert trace['direction_mapping']=={**mapping[mean],'surrogate_mean':cfg['surrogate_mean'],'future_labels_used_for_orders':False}
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
  print(json.dumps({'phase':'fold_scalar_audit_complete','fold':f,'accounts':104,'new_paths':16}),flush=True)
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2} and regimes=={'bull':2,'bear':4,'sideways':2}
 expectedcounts={'bound_saved_feature_matrices':32,'independent_fit_priors':16,'bound_model_verification_records':32,'new_E_direction_forecasts':64,'new_calibration_direction_arrays':64,'exact_old_controls':288,'independent_scalar_accounts':832,'new_own_state_paths':128,'new_own_state_decisions':44032,'new_full_trace_replays':128,'new_unscored_decisions_retained':192,'new_fallback_decisions':2656}
 for k,n in expectedcounts.items():assert counts[k]==n,(k,counts[k],n)
 for k,n in {'new_model_fits':32,'shared_prior_estimates':16,'new_causal_policy_names':16,'total_adaptively_explored_causal_names':190,'risk_model_or_calibration_fits':0}.items():assert res[k]==n
 for k in ('selection_performed','teacher_use_allowed','additional_test_used_for_modeling_or_scoring','high_probability_generalization_established'):assert res[k] is False
 warning_counts=Counter(line.split('RuntimeWarning: ',1)[1] for line in (out/'run.log').read_text().splitlines() if 'RuntimeWarning: ' in line)
 runtime_warning_report={'run_log_sha256':sha(out/'run.log'),'counts':dict(warning_counts),'total':sum(warning_counts.values()),'saved_gradient_infinity_max':max(v['scalar_verification']['normalized_gradient_infinity'] for f in fits.values() for v in f['fit_provenance']['fitted_state'].values()),'interpretation':'RuntimeWarning matmul messages occurred during the single canonical run. This audit does not diagnose their cause or claim warning-free numerical execution; stored finite predictions and scalar production verification records are bound, with model numerical replay audited separately.'}
 report={'status':'pass','runtime_warnings':runtime_warning_report,'auditor_revision_note':'Prepared schema was extended after first audit stop to require the frozen runner additional predict-position/timestamp and selected feature arrays. No research source, output or fit was changed.','source_revision':reg['source_revision'],'script_sha256':sha(__file__),'config_sha256':reg['config_sha256'],'preflight_sha256':sha(out/'preflight.json'),'results_sha256':sha(out/'results.json'),'source_audit_sha256':sha(sourcepath),'ancestor_artifacts_verified':2120,'own_artifacts_verified':720,'counts':dict(counts),'distinct_hashed_files':len(verified),'max_absolute_differences':maximum,'support':dict(totals),'regime_counts':dict(regimes),'inventory':{**inventory,'rows':416,'return_scores':160,'classification_scores':96},'scope':'All immutable sources/artifacts; fit-return, class label, sample weight and 16 prior identity;64 E and64 I mappings from saved logits and fixed causal parent magnitudes; unchanged risk/support;288 exact source controls;832 scalar accounts;128 new own-state paths and complete decision traces. No canonical helpers imported or fits performed.','separate_audit_scope':['The 32 model coefficients, scalar gradient/feature reconstruction and predictor numerical verification are independently audited by the technical auditor. This audit binds their saved verification records without re-solving models.','The 160 return losses,96 classification records and summary arithmetic are independently audited by the statistics auditor.'],'limitations':['Original repeatedly reused development, retrospective masks and absent historical receipt evidence remain.','Mapped sign times fixed magnitude is a return surrogate, not a calibrated conditional mean. Magnitude-weighted classifier probability targets a tilted distribution.','Interval parent magnitudes use frozen S calibration only from I onward; no S outcomes are used to score these causal mapped means.','Filtered raw Spot reads remain below the original cutoff; source hashes cover full files.']}
 path=Path('/tmp/oracle_direction_audit_20260906.json');path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':'pass','path':str(path),'counts':dict(counts),'maximum':maximum}),flush=True)
if __name__=='__main__':main()
