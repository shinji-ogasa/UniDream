"""Stage19 DATA-ONLY frozen probability and training-statistic source audit.
Only existing artifacts are read/hashed and saved values compared. No new soft
mapping, model, fitted statistic, loss or order is computed. No market parser or
canonical fitting/planning/scoring helper is imported.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess
import joblib
import numpy as np
import pandas as pd
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('/tmp/oracle_soft_direction_source_audit_20260906.json')
SOURCE='5a82c270c64a342ab7e9df8105b7d23d1336d876'
HASHES={'registration':'289968f71450300c65f515dfe3ed2f2eae50c7f4fc5442f8669d1867faf65888','preflight':'33b2ffc8d74a57b0827e35ea9c08c2e37eefdb421d9f6e726b566f0955e11285','results':'ab2b62a017a70fd65756614f3940c3194d60b8c0ebdab02aa058c988ccbdb678'}
FOLDS=tuple(range(5,13));GROUPS=('technical','perp_delay0');RULES=('utility_risk1','utility_risk1_fallback_bh')
STREAMS=tuple((g,reg) for g in GROUPS for reg in ('C1','L2unit'))
FKEYS={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean','parent_mu','logit','probability'}
CKEYS={'timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','mu','parent_mu','logit','probability','fit_return_mean'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def matrixsha(a):
 a=np.asarray(a,dtype='<f8',order='C');return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes()).hexdigest()
def masksha(index,mask):return hashlib.sha256(index.asi8.astype('<i8').tobytes()+np.asarray(mask,'u1').tobytes()).hexdigest()
def positionmasksha(mask):return hashlib.sha256(np.asarray([len(mask)],dtype='<i8').tobytes()+mask.astype('u1').tobytes()).hexdigest()
def indexsha(index):
 header=json.dumps({'type':type(index).__name__,'dtype':str(index.dtype),'length':len(index)},sort_keys=True).encode();values=pd.util.hash_pandas_object(index,index=False).to_numpy(dtype='<u8');return hashlib.sha256(header+b'\n'+values.tobytes()).hexdigest()
def arr(p,keys=None):
 with np.load(p,allow_pickle=False) as z:
  if keys is not None:assert set(z.files)==set(keys),('schema',str(p))
  a={k:z[k] for k in z.files}
 assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid',str(p))
 return a
def exact(a,b):
 a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)

def main():
 os.chdir(ROOT);counts=Counter();verified={};direct={}
 def verify(p,h):
  p=Path(p).resolve()
  if p not in verified:verified[p]=sha(p)
  assert verified[p]==h,('hash',str(p));counts['hash_binding_checks']+=1
 src=Path('codex_outputs/oracle_regularized_direction_decisions_v1');stage17=Path('codex_outputs/oracle_direction_decisions_v1');parity=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
 for k,h in HASHES.items():p=src/(k+'.json');verify(p,h);direct[str(p)]=h
 reg,pre,res=(read(src/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config'];assert reg['source_revision']==SOURCE and res['registration_sha256']==digest(reg)
 cp=Path('configs/oracle_regularized_direction_decisions_20260906.yaml');verify(cp,reg['config_sha256']);direct[str(cp)]=reg['config_sha256'];assert yaml.safe_load(cp.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+str(cp)])).hexdigest()==reg['config_sha256']
 assert reg['preflight_sha256']==cfg['preflight_sha256']==HASHES['preflight'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==31
 for p,h in cfg['source_bindings'].items():verify(p,h);direct[p]=h;assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+p])).hexdigest()==h
 for p,h in pre['direct_source_bindings'].items():verify(p,h);assert p not in direct or direct[p]==h;direct[p]=h
 oldsourcepath=Path('/tmp/oracle_regularized_direction_source_audit_20260906.json');oldsourcehash='43d255bbc8e361d8ff9e78bbf308b9c1a476e90ee1442584ff0f678084ed69e7';verify(oldsourcepath,oldsourcehash);direct[str(oldsourcepath)]=oldsourcehash;oldsource=read(oldsourcepath)
 oldauditpath=Path('/tmp/oracle_regularized_direction_audit_20260906.json');oldauditsha='a2d41ce9c4409883e17d4b85bb6440d6f799759df9ee5a0a0ac9fe7c697db8ec';verify(oldauditpath,oldauditsha);direct[str(oldauditpath)]=oldauditsha;oldaudit=read(oldauditpath)
 assert oldsource['passed'] and oldaudit['status']=='pass' and oldaudit['source_revision']==SOURCE and oldaudit['results_sha256']==HASHES['results']
 for p,h in {'/tmp/oracle_regularized_direction_source_audit_20260906.py':oldsource['script_sha256'],'/tmp/oracle_regularized_direction_audit_20260906.py':oldaudit['script_sha256']}.items():verify(p,h);direct[p]=h
 assert oldaudit['counts']['independent_scalar_accounts']==960 and oldaudit['counts']['new_own_state_paths']==64
 assert pre['source_artifact_bindings']==oldsource['source_artifact_bindings'] and len(pre['source_artifact_bindings'])==2840
 assert pre['spot_data_proof']==oldsource['spot_data_proof'] and pre['um_data_proof']==oldsource['um_data_proof']
 for k,v in {'new_model_fits':32,'new_unique_priors':0,'new_causal_policy_names':8,'total_adaptively_explored_causal_names':198,'risk_model_or_calibration_fits':0}.items():assert res[k]==v
 for k in ('selection_performed','teacher_use_allowed','additional_test_used_for_modeling_or_scoring','high_probability_generalization_established'):assert res[k] is False
 controls=tuple(cfg['control_ids'])+tuple(cfg['new_policy_ids']);newmeans=tuple(cfg['new_mean_ids']);modelids=tuple(cfg['model_ids']);assert len(controls)==len(set(controls))==60
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(r['fold'],r['segment'],r['mean_id']):r for r in res['scores']};cs={(r['fold'],r['segment'],r['classifier_id']):r for r in res['classification_scores']};fitrecords={r['fold']:r for r in res['fit_records']}
 assert len(rows)==len(res['rows'])==480 and set(rows)=={(f,c) for f in FOLDS for c in controls}
 assert len(scores)==len(res['scores'])==224 and set(scores)=={(f,s,m) for f in FOLDS for s in ('interval','evaluation') for m in cfg['return_score_means']}
 assert len(cs)==len(res['classification_scores'])==160 and set(cs)=={(f,s,c) for f in FOLDS for s in ('interval','evaluation') for c in cfg['classifiers']}
 assert len(fitrecords)==len(res['fit_records'])==8 and set(fitrecords)==set(FOLDS);assert sum(r['hindsight_only'] for r in rows.values())==192
 bindings=dict(pre['source_artifact_bindings']);own={}
 for f in FOLDS:
  p=src/f'fold_{f}.json';fold=read(p);direct[str(p)]=sha(p);assert fold['registration_sha256']==digest(reg)
  for k in ('rows','scores','classification_scores','direction_diagnostics'):assert fold[k]==[v for v in res[k] if v['fold']==f]
  expected={str(src/k/f'fold{f}_{name}.{ext}') for k,names,ext in [('models',modelids,'joblib'),('provenance',('fit',),'json'),('forecasts',newmeans,'npz'),('calibration',newmeans,'npz'),('targets',controls,'npz'),('traces',cfg['new_policy_ids'],'json')] for name in names}
  assert len(expected)==81 and set(fold['artifact_sha256'])==expected
  for p,h in fold['artifact_sha256'].items():assert p not in bindings;bindings[p]=h;own[p]=h
 assert len(own)==648 and len(bindings)==3488 and len({str(Path(p).resolve()) for p in bindings})==3488
 for p,h in bindings.items():verify(p,h)
 counts.update(parent_sources=31,parent_ancestor_artifacts=2840,parent_own_artifacts=648,total_source_artifacts=3488,parent_control_rows=480,parent_return_score_rows=224,parent_classification_score_rows=160,parent_causal_rows=288,parent_hindsight_rows=192)
 r17,p17,q17=(read(stage17/(k+'.json')) for k in ('registration','preflight','results'))
 assert r17['source_revision']=='6ae673fcdfeed29280256450c05eb8905af77ee3' and q17['registration_sha256']==digest(r17) and cfg['parent_source_revision']==r17['source_revision']
 assert p17['spot_data_proof']==pre['spot_data_proof'] and p17['um_data_proof']==pre['um_data_proof']
 fit17={r['fold']:r for r in q17['fit_records']};records=[];totals=Counter();regimes=Counter();inputbindings={};cut=pd.Timestamp(cfg['data_cutoff']);assert cut==pd.Timestamp('2023-04-16T13:45:00Z');step=pd.Timedelta(minutes=15);grid=None
 for f in FOLDS:
  s=next(v for v in oldsource['support'] if v['fold']==f);sp=next(v for v in pre['support'] if v['fold']==f)
  assert sp['counts']=={k:s['counts'][k] for k in sp['counts']}
  for k in sp['mask_sha256']:assert sp['mask_sha256'][k]==s['mask_sha256'][k]
  packpath=stage17/'fit_data'/f'fold{f}_training.npz';pack=arr(packpath);inputbindings[str(packpath)]=bindings[str(packpath)]
  p17path=stage17/'provenance'/f'fold{f}_fit.json';p18path=src/'provenance'/f'fold{f}_fit.json';a17,a18=read(p17path),read(p18path);inputbindings[str(p17path)]=bindings[str(p17path)];inputbindings[str(p18path)]=bindings[str(p18path)]
  assert a17==fit17[f] and a18==fitrecords[f] and a18['fit_priors']==a17['fit_priors'] and a18['fit_prior_logits']==a17['fit_prior_logits']
  pr17,pr18=a17['fit_provenance'],a18['fit_provenance'];assert a18['fit_labels_weights_and_priors_match_parent_exactly'] is True and a18['frozen_fit_data_reused'] is True
  for k in ('feature_columns','feature_counts','index_sha256','mask_counts','mask_ranges','mask_position_sha256','fit_return_sha256','fit_binary_labels_sha256','fit_class_counts','fit_features_sha256','predict_features_sha256','fit_features_and_return_sha256','sample_weights'):assert pr18[k]==pr17[k]
  fp,pp=pack['fit_positions'],pack['predict_positions'];ft=pd.DatetimeIndex(pd.to_datetime(pack['timestamps'],utc=True));pt=pd.DatetimeIndex(pd.to_datetime(pack['predict_timestamps'],utc=True));start=ft[0]-int(fp[0])*step
  if grid is None:grid=pd.date_range(start,cut,freq='15min',inclusive='left')
  assert grid[0]==start and indexsha(grid)==pr17['index_sha256'];exact(grid[fp].asi8,pack['timestamps']);exact(grid[pp].asi8,pack['predict_timestamps'])
  dates=s['calendar'];T=pd.Timestamp(dates['fit']['start_inclusive']);S=pd.Timestamp(dates['scale']['start_inclusive']);I=pd.Timestamp(dates['interval']['start_inclusive']);E=pd.Timestamp(dates['evaluation']['start_inclusive']);end=pd.Timestamp(dates['evaluation']['end_exclusive'])
  assert (ft>=T).all() and (ft<S).all() and (ft+pd.Timedelta(minutes=375)<S).all() and (pt>=S).all() and (pt<end).all() and pp[0]>fp[-1]
  masks={}
  for segment,pos in [('fit',fp),('predict',pp)]:
   mask=np.zeros(len(grid),dtype=bool);mask[pos]=True;masks[segment]=mask;assert masksha(grid,mask)==s['mask_sha256'][segment] and positionmasksha(mask)==pr17['mask_position_sha256'][segment]
   for g in GROUPS:
    x=pack[segment+'_features_'+g];assert x.shape==(len(pos),len(oldsource['features'][g])) and np.isfinite(x).all();assert matrixsha(x)==s['feature_selected_float64le_sha256'][segment][g]==pr17[segment+'_features_sha256'][g] and digest(x.tolist())==s['feature_selected_sha256'][segment][g];counts['saved_feature_matrices_bound']+=1
  assert matrixsha(pack['returns'])==pr17['fit_return_sha256'] and digest(pack['returns'].tolist())==s['continuous_fit_return_sha256'] and matrixsha(pack['binary_labels'])==pr17['fit_binary_labels_sha256']
  for w in ('ordinary','magnitude'):assert matrixsha(pack['weights_'+w])==pr17['sample_weights'][w]['weight_sha256'];counts['saved_weight_vectors_bound']+=1
  scalar_values={'fit_abs_return_mean':a17['fit_abs_return_mean'],'fit_return_mean':a17['fit_return_mean'],'magnitude_prior':a17['fit_priors']['magnitude'],'magnitude_prior_logit':a17['fit_prior_logits']['magnitude']}
  assert all(type(v) is float and np.isfinite(v) for v in scalar_values.values()) and scalar_values['fit_abs_return_mean']>0 and 0<scalar_values['magnitude_prior']<1
  assert scalar_values['magnitude_prior']==pr17['sample_weights']['magnitude']['positive_prior'];counts['saved_training_scalar_sets_bound']+=1
  ix=pd.date_range(E,end,freq='15min',inclusive='left');cix=pd.date_range(S,E,freq='15min',inclusive='left');epos,cpos=grid.get_indexer(ix),grid.get_indexer(cix);ref=None;calref=None;streamrecords={}
  for g,strength in STREAMS:
   root=stage17 if strength=='C1' else src;mid=g+'_magnitude'+('' if strength=='C1' else '_l2unit');name=mid+'_direction';oldmid=g+'_magnitude';ep=root/'forecasts'/f'fold{f}_{name}.npz';cp=root/'calibration'/f'fold{f}_{name}.npz';mp=root/'models'/f'fold{f}_{mid}.joblib';inputbindings[str(ep)]=bindings[str(ep)];inputbindings[str(cp)]=bindings[str(cp)];inputbindings[str(mp)]=bindings[str(mp)]
   a,ca=arr(ep,FKEYS),arr(cp,CKEYS);exact(a['timestamps'],ix.asi8);exact(ca['timestamps'],cix.asi8)
   if ref is None:ref=a;calref=ca
   else:
    for k in ('timestamps','variance','actual','score_support','inference_mask','fit_return_mean'):exact(a[k],ref[k])
    for k in ('timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','fit_return_mean'):exact(ca[k],calref[k])
   assert float(a['fit_return_mean'])==float(ca['fit_return_mean'])==scalar_values['fit_return_mean']
   inf,score=a['inference_mask'],a['score_support'];pred,mapped=ca['classifier_predict_mask'],ca['mapped_inference_mask'];assert inf.dtype==score.dtype==pred.dtype==mapped.dtype==bool
   exact(np.isfinite(a['probability']),inf);exact(np.isfinite(a['logit']),inf);exact(np.isfinite(ca['probability']),pred);exact(np.isfinite(ca['logit']),pred)
   assert ((a['probability'][inf]>=0)&(a['probability'][inf]<=1)).all() and ((ca['probability'][pred]>=0)&(ca['probability'][pred]<=1)).all()
   exact(inf,masks['predict'][epos]);exact(pred,masks['predict'][cpos]);exact(mapped,pred&np.asarray(cix>=I));assert not (score&~inf).any() and not (ca['interval_mask']&~mapped).any()
   assert (ix[score]+pd.Timedelta(minutes=375)<=end).all() and (cix[ca['interval_mask']]+pd.Timedelta(minutes=375)<E).all()
   for key,mask,pos in [('scale',ca['scale_mask'],cpos),('interval',ca['interval_mask'],cpos),('inference',inf,epos),('score',score,epos)]:
    full=np.zeros(len(grid),dtype=bool);full[pos]=mask;assert masksha(grid,full)==s['mask_sha256'][key];masks[key]=full
   state=(pr17 if strength=='C1' else pr18)['fitted_state'][oldmid];z=np.concatenate((ca['logit'][pred],a['logit'][inf]));prob=np.concatenate((ca['probability'][pred],a['probability'][inf]));assert len(z)==len(pp) and matrixsha(z)==state['predict_logits_sha256'] and matrixsha(prob)==state['predict_probability_sha256']
   model=joblib.load(mp);scaler,lr=model.steps[0][1],model.steps[1][1];params=pr17['parameters']['logistic'] if strength=='C1' else state['logistic_parameters'];assert lr.get_params()==params
   for attr,key in [('mean_','scaler_mean'),('var_','scaler_variance'),('scale_','scaler_scale')]:exact(getattr(scaler,attr),np.asarray(state[key],float))
   for attr,key in [('coef_','coefficient'),('intercept_','intercept')]:exact(getattr(lr,attr),np.asarray(state[key],float));assert matrixsha(getattr(lr,attr))==state[key+'_sha256']
   assert matrixsha(np.column_stack((pack['fit_features_'+g],pack['binary_labels'],pack['weights_magnitude'])))==state['fit_features_labels_weights_sha256']
   for segment,n in [('interval',int(ca['interval_mask'].sum())),('evaluation',int(score.sum()))]:
    targetcs={(v['fold'],v['segment'],v['classifier_id']):v for v in (q17 if strength=='C1' else res)['classification_scores']};assert targetcs[f,segment,mid]['rows']==n
   streamrecords[g+'_'+strength]={'group':g,'parent_regularization':strength,'classifier_id':mid,'forecast_path':str(ep),'forecast_sha256':bindings[str(ep)],'calibration_path':str(cp),'calibration_sha256':bindings[str(cp)],'model_path':str(mp),'model_sha256':bindings[str(mp)],'predict_probability_sha256':state['predict_probability_sha256'],'predict_logit_sha256':state['predict_logits_sha256'],'E_inference_rows':int(inf.sum()),'E_score_rows':int(score.sum()),'I_inference_rows':int(mapped.sum()),'I_score_rows':int(ca['interval_mask'].sum())}
   counts['selected_probability_streams_bound']+=1;counts['selected_model_states_bound']+=1;counts['selected_E_probability_arrays_bound']+=1;counts['selected_cal_probability_arrays_bound']+=1
  prior_sources={}
  for g in GROUPS:
   name=g+'_magnitude_prior_direction';ep=stage17/'forecasts'/f'fold{f}_{name}.npz';cp=stage17/'calibration'/f'fold{f}_{name}.npz';a,ca=arr(ep,FKEYS),arr(cp,CKEYS)
   inputbindings[str(ep)]=bindings[str(ep)];inputbindings[str(cp)]=bindings[str(cp)]
   for key in ('timestamps','actual','variance','fit_return_mean','inference_mask','score_support'):exact(a[key],ref[key])
   for key in ('timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','fit_return_mean'):exact(ca[key],calref[key])
   inf=a['inference_mask'];pred=ca['classifier_predict_mask'];mapped=ca['mapped_inference_mask'];q=float(a['probability'][np.flatnonzero(inf)[0]])
   assert np.isfinite(q) and 0<=q<=1 and np.all(a['probability'][inf]==q) and np.all(ca['probability'][pred]==q)
   exact(np.isfinite(a['probability']),inf);exact(np.isfinite(ca['probability']),pred)
   assert np.all(a['logit'][inf]==a17['fit_prior_logits']['magnitude']) and np.all(ca['logit'][pred]==a17['fit_prior_logits']['magnitude'])
   prior_sources[g]={'forecast_path':str(ep),'forecast_sha256':bindings[str(ep)],'calibration_path':str(cp),'calibration_sha256':bindings[str(cp)],'saved_prior_probability':q,'raw_fit_prior_probability':a17['fit_priors']['magnitude'],'saved_prior_logit':a17['fit_prior_logits']['magnitude'],'E_probability_selected_sha256':matrixsha(a['probability'][inf]),'I_probability_selected_sha256':matrixsha(ca['probability'][mapped]),'E_constant_rows':int(inf.sum()),'I_constant_rows':int(mapped.sum()),'all_S_I_predict_rows_constant':True,'sigmoid_recomputed':False,'raw_fit_prior_substituted':False}
   counts['saved_prior_probability_sources_bound']+=1
  assert set(masks)=={'fit','predict','scale','interval','inference','score'}
  for k,v in masks.items():assert int(v.sum())==s['counts'][k]
  for c in controls:
   p=src/'targets'/f'fold{f}_{c}.npz';assert rows[f,c]['targets_sha256']==bindings[str(p)];exact(arr(p,{'timestamps','targets'})['timestamps'],ix.asi8);counts['parent_target_calendars_bound']+=1
  record={'fold':f,'calendar':s['calendar'],'counts':s['counts'],'mask_sha256':s['mask_sha256'],'feature_columns':oldsource['features'],'feature_selected_sha256':s['feature_selected_sha256'],'feature_selected_float64le_sha256':s['feature_selected_float64le_sha256'],'continuous_fit_return_sha256':s['continuous_fit_return_sha256'],'saved_binary_labels_sha256':s['binary_fit_labels_sha256'],'saved_weights_sha256':s['saved_weight_sha256'],'fit_data_path':str(packpath),'fit_data_sha256':bindings[str(packpath)],'training_scalar_source_path':str(p17path),'training_scalar_source_sha256':bindings[str(p17path)],'training_scalar_fields':{'fit_abs_return_mean':'fit_abs_return_mean','fit_return_mean':'fit_return_mean','magnitude_prior':'fit_priors.magnitude','magnitude_prior_logit':'fit_prior_logits.magnitude'},'saved_training_scalar_values_sha256':digest(scalar_values),'probability_streams':streamrecords,'saved_prior_probability_sources':prior_sources,'E_inference_rows':int(ref['inference_mask'].sum()),'E_score_rows':int(ref['score_support'].sum()),'I_inference_rows':int(calref['mapped_inference_mask'].sum()),'I_score_rows':int(calref['interval_mask'].sum()),'fallback_rows':s['fallback_rows'],'missing_current_open_rows':s['missing_current_open_rows'],'regime':s['regime'],'new_mapping_statistics_or_outputs_computed':False};records.append(record)
  for k in ('E_inference_rows','E_score_rows','I_inference_rows','I_score_rows','fallback_rows','missing_current_open_rows'):totals[k]+=record[k]
  regimes[s['regime']]+=1
 assert totals=={'E_inference_rows':2586,'E_score_rows':2574,'I_inference_rows':2537,'I_score_rows':2523,'fallback_rows':332,'missing_current_open_rows':2} and regimes=={'bull':2,'bear':4,'sideways':2}
 for k,n in {'saved_prior_probability_sources_bound':16,'saved_feature_matrices_bound':32,'saved_weight_vectors_bound':16,'saved_training_scalar_sets_bound':8,'selected_probability_streams_bound':32,'selected_model_states_bound':32,'selected_E_probability_arrays_bound':32,'selected_cal_probability_arrays_bound':32,'parent_target_calendars_bound':480}.items():assert counts[k]==n
 proposed={'status':'proposed_not_registered_or_executed','probability_streams':['technical_C1','technical_L2unit','perp_delay0_C1','perp_delay0_L2unit'],'learned_mean_streams':4,'group_specific_constant_means':6,'constant_types':['prior','fitmean','zero'],'all_new_mean_streams':10,'missing_rules':list(RULES),'new_causal_policy_names':20,'learned_policy_names':8,'new_control_policy_names':12,'old_control_policy_names':60,'total_policy_names':80,'economic_rows':640,'base_stress_accounts':1280,'adaptive_causal_names_before':198,'adaptive_causal_names_after':218,'candidate_gates_apply_only_to_learned_policies':True,'new_model_fits':0,'new_training_statistics':0,'mapping_formula_not_evaluated':True,'mapped_prior_probability_source':'bound Stage17 NPZ probability, constant on original I/E support; no recomputed sigmoid or raw fit-prior substitution','prior_identity_guard_only':{'absolute_tolerance':1e-14,'relative_tolerance':1e-12,'residual_not_forced_to_zero':True},'return_mean_ids':24,'return_score_records':384,'unchanged_classifiers':10,'unchanged_classification_records':160,'learned_mapping_diagnostic_records':64,'artifacts_per_fold':121,'total_new_artifacts':968,'economic_references_per_learned':5}
 report={'passed':True,'schema':'oracle-soft-direction-source-audit-v1','scope':'DATA-ONLY immutable Stage18/17 inputs; selected magnitude probability arrays, model states, saved T statistic bindings, original six masks and old record inventories. No new mapping, statistics, model, loss or order; no canonical research helper or market parser.','script_sha256':sha(__file__),'parent_source_revision':SOURCE,'parent_file_sha256':HASHES,'parent_registration_canonical_sha256':digest(reg),'counts':dict(counts),'distinct_hashed_files':len(verified),'source_artifact_inventory_sha256':digest(bindings),'source_artifact_bindings':bindings,'direct_source_bindings':direct,'saved_input_bindings':inputbindings,'old_source_audit_sha256':oldsourcehash,'old_completed_audit_sha256':oldauditsha,'spot_data_proof':pre['spot_data_proof'],'um_data_proof':pre['um_data_proof'],'features':oldsource['features'],'support':records,'totals':dict(totals),'regime_counts':dict(regimes),'regime_gate_pass':False,'proposed_counts':proposed,'full_grid':{'start_inclusive':grid[0].isoformat(),'end_exclusive':cut.isoformat(),'rows':len(grid),'source':'Reconstructed from bound saved positions/timestamps, no market values.'},'limitations':['The new mean mapping formula, control construction, score inventories and contrasts must be fixed before freeze; this audit does not evaluate any proposed new mean.','Saved T statistics and labels/weights are bound to completed independent audits, not re-estimated. Saved NPZ prior q and raw fit prior are preserved separately; no sigmoid, soft mapping or equality forcing is performed.','Magnitude-weighted probability is a tilted-distribution target; soft mapping cannot be described as calibrated ordinary event probability without separate evidence.','The 24 hindsight old controls remain descriptive and are not deployable teachers.','Repeatedly reused development, retrospective common masks, missing historical receipts and failing 2/4/2 regime coverage remain.','Raw source files are byte-hashed only; no additional-test price values are decoded.']}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'path':str(OUT),'sha256':sha(OUT),'script_sha256':report['script_sha256'],'counts':dict(counts),'totals':dict(totals),'proposed_counts':proposed}))
if __name__=='__main__':main()
