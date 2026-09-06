"""Stage18 DATA-ONLY source audit using immutable, already-audited Stage17 arrays.
No canonical research helper or market parser is imported. No new label, prior,
weight, coefficient, prediction, loss, order or economic outcome is computed.
Saved prior-stage model states are loaded for exact provenance identity only.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess
import joblib
import numpy as np
import pandas as pd
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('/tmp/oracle_regularized_direction_source_audit_20260906.json')
SOURCE='6ae673fcdfeed29280256450c05eb8905af77ee3'
HASHES={'registration':'06ef781a2835b25ccfc01db9c758ecf79c7fabefe23346af37db1ec19cefbdab','preflight':'a8f20d76fa6ed17592c53c044cdaada7f7a08bd9a2c244fe9fa4a56c3d8eebcd','results':'c659163526547d5aecc75ccd8a9f987a4000eee3152cd6552f78c1428b158657'}
FOLDS=tuple(range(5,13));GROUPS=('technical','perp_delay0');WEIGHTINGS=('ordinary','magnitude');HALVES=tuple(g+'_half' for g in GROUPS);RULES=('utility_risk1','utility_risk1_fallback_bh')
MODEL_IDS=tuple(g+'_'+w for g in GROUPS for w in WEIGHTINGS)
FIT_KEYS={'fit_positions','timestamps','returns','binary_labels','predict_positions','predict_timestamps','weights_ordinary','weights_magnitude'}|{segment+'_features_'+g for segment in ('fit','predict') for g in GROUPS}
FORECAST_KEYS={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean','parent_mu','logit','probability'}
CAL_KEYS={'timestamps','actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask','mu','parent_mu','logit','probability','fit_return_mean'}

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
 src=Path('codex_outputs/oracle_direction_decisions_v1');parity=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
 for k,h in HASHES.items():p=src/(k+'.json');verify(p,h);direct[str(p)]=h
 reg,pre,res=(read(src/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config']
 assert reg['source_revision']==SOURCE and res['registration_sha256']==digest(reg)
 cp=Path('configs/oracle_direction_decisions_20260906.yaml');verify(cp,reg['config_sha256']);direct[str(cp)]=reg['config_sha256'];assert yaml.safe_load(cp.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+str(cp)])).hexdigest()==reg['config_sha256']
 assert reg['preflight_sha256']==cfg['preflight_sha256']==HASHES['preflight'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==29
 for p,h in cfg['source_bindings'].items():verify(p,h);direct[p]=h;assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+p])).hexdigest()==h
 for p,h in pre['direct_source_bindings'].items():verify(p,h);assert p not in direct or direct[p]==h;direct[p]=h
 for p,h in cfg['parent_manifest_bindings'].items():verify(p,h);direct[p]=h
 oldsourcepath=Path('/tmp/oracle_direction_source_audit_20260906.json');oldsourcehash='a64268eecfa8330440792aad5cec40c0a62ef450d9ac738fc0f41729edcfdf32';verify(oldsourcepath,oldsourcehash);direct[str(oldsourcepath)]=oldsourcehash;oldsource=read(oldsourcepath)
 oldauditpath=Path('/tmp/oracle_direction_audit_20260906.json');oldauditsha='bf7b4556ef80ecbd395cee1d00fbe914f08b30d10fa13d524c34b75806530fb0';verify(oldauditpath,oldauditsha);direct[str(oldauditpath)]=oldauditsha;oldaudit=read(oldauditpath)
 assert oldsource['passed'] and oldaudit['status']=='pass' and oldaudit['source_revision']==SOURCE and oldaudit['results_sha256']==HASHES['results']
 for p,h in {'/tmp/oracle_direction_source_audit_20260906.py':oldsource['script_sha256'],'/tmp/oracle_direction_audit_20260906.py':oldaudit['script_sha256']}.items():verify(p,h);direct[p]=h
 assert oldaudit['counts']['independent_scalar_accounts']==832 and oldaudit['counts']['new_own_state_paths']==128 and oldaudit['counts']['independent_fit_priors']==16 and oldaudit['counts']['bound_saved_feature_matrices']==32
 assert pre['source_artifact_bindings']==oldsource['source_artifact_bindings'] and len(pre['source_artifact_bindings'])==2120
 assert pre['spot_data_proof']==oldsource['spot_data_proof'] and pre['um_data_proof']==oldsource['um_data_proof']
 for k,v in {'new_model_fits':32,'shared_prior_estimates':16,'new_causal_policy_names':16,'total_adaptively_explored_causal_names':190,'risk_model_or_calibration_fits':0}.items():assert res[k]==v
 for k in ('selection_performed','teacher_use_allowed','additional_test_used_for_modeling_or_scoring','high_probability_generalization_established'):assert res[k] is False
 controls=tuple(cfg['control_ids'])+tuple(cfg['new_policy_ids']);newmeans=tuple(cfg['new_mean_ids']);means=tuple(cfg['return_score_means']);classifiers=tuple(cfg['classifiers']);assert len(controls)==len(set(controls))==52 and cfg['model_ids']==list(MODEL_IDS)
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(r['fold'],r['segment'],r['mean_id']):r for r in res['scores']};cs={(r['fold'],r['segment'],r['classifier_id']):r for r in res['classification_scores']};fitrecords={r['fold']:r for r in res['fit_records']}
 assert len(rows)==len(res['rows'])==416 and set(rows)=={(f,c) for f in FOLDS for c in controls}
 assert len(scores)==len(res['scores'])==160 and set(scores)=={(f,s,m) for f in FOLDS for s in ('interval','evaluation') for m in means}
 assert len(cs)==len(res['classification_scores'])==96 and set(cs)=={(f,s,c) for f in FOLDS for s in ('interval','evaluation') for c in classifiers}
 assert len(fitrecords)==len(res['fit_records'])==8 and set(fitrecords)==set(FOLDS)
 assert sum(r['hindsight_only'] for r in rows.values())==192
 bindings=dict(pre['source_artifact_bindings']);own={}
 for f in FOLDS:
  p=src/f'fold_{f}.json';fold=read(p);direct[str(p)]=sha(p);assert fold['registration_sha256']==digest(reg)
  for k in ('rows','scores','classification_scores'):assert fold[k]==[v for v in res[k] if v['fold']==f]
  expected={str(src/k/f'fold{f}_{name}.{ext}') for k,names,ext in [('models',MODEL_IDS,'joblib'),('fit_data',('training',),'npz'),('provenance',('fit',),'json'),('forecasts',newmeans,'npz'),('calibration',newmeans,'npz'),('targets',controls,'npz'),('traces',cfg['new_policy_ids'],'json')] for name in names}
  assert len(expected)==90 and set(fold['artifact_sha256'])==expected
  for p,h in fold['artifact_sha256'].items():assert p not in bindings;bindings[p]=h;own[p]=h
 assert len(own)==720 and len(bindings)==2840 and len({str(Path(p).resolve()) for p in bindings})==2840
 for p,h in bindings.items():verify(p,h)
 counts.update(parent_sources=29,parent_ancestor_artifacts=2120,parent_own_artifacts=720,total_source_artifacts=2840,parent_control_rows=416,parent_return_score_rows=160,parent_classification_score_rows=96,parent_causal_rows=224,parent_hindsight_rows=192)
 records=[];totals=Counter();regimes=Counter();inputbindings={};cut=pd.Timestamp(cfg['data_cutoff']);assert cut==pd.Timestamp('2023-04-16T13:45:00Z');step=pd.Timedelta(minutes=15);grid=None
 for f in FOLDS:
  s=next(v for v in oldsource['support'] if v['fold']==f);p=next(v for v in pre['support'] if v['fold']==f)
  for k in p['counts']:assert p['counts'][k]==s['counts'][k] and p['mask_sha256'][k]==s['mask_sha256'][k]
  assert p['fit_return_sha256']==s['continuous_return_selected_sha256']['fit'] and p['feature_columns']==oldsource['features']
  fitpath=src/'fit_data'/f'fold{f}_training.npz';fd=arr(fitpath,FIT_KEYS);inputbindings[str(fitpath)]=bindings[str(fitpath)]
  provenancepath=src/'provenance'/f'fold{f}_fit.json';record=read(provenancepath);inputbindings[str(provenancepath)]=bindings[str(provenancepath)];assert record==fitrecords[f] and record['fit_source_binding']==p;prov=record['fit_provenance']
  assert prov['model_selection_performed'] is False and prov['evaluation_labels_used'] is False and prov['risk_or_calibration_fitted'] is False and prov['feature_columns']==oldsource['features'] and prov['feature_counts']=={'technical':29,'perp_delay0':31}
  n=len(fd['fit_positions']);assert n==s['counts']['fit']==(800,1034,1313,1500,1503,1634,1672,1794)[f-5] and n>=512
  fp=fd['fit_positions'];pp=fd['predict_positions'];ft=pd.DatetimeIndex(pd.to_datetime(fd['timestamps'],utc=True));pt=pd.DatetimeIndex(pd.to_datetime(fd['predict_timestamps'],utc=True))
  assert fp.dtype.kind in 'iu' and pp.dtype.kind in 'iu' and len(pp)==s['counts']['predict'] and (np.diff(fp)>0).all() and (np.diff(pp)>0).all() and pp[0]>fp[-1]
  start=ft[0]-int(fp[0])*step
  if grid is None:grid=pd.date_range(start,cut,freq='15min',inclusive='left')
  assert grid[0]==start and grid[-1]+step==cut and indexsha(grid)==prov['index_sha256']
  exact(grid[fp].asi8,fd['timestamps']);exact(grid[pp].asi8,fd['predict_timestamps'])
  dates=s['calendar'];T=pd.Timestamp(dates['fit']['start_inclusive']);S=pd.Timestamp(dates['scale']['start_inclusive']);I=pd.Timestamp(dates['interval']['start_inclusive']);E=pd.Timestamp(dates['evaluation']['start_inclusive']);end=pd.Timestamp(dates['evaluation']['end_exclusive'])
  assert T==E-pd.DateOffset(months=24) and S==E-pd.DateOffset(months=6) and I==E-pd.DateOffset(months=3) and end==E+pd.DateOffset(months=3)
  assert (ft>=T).all() and (ft<S).all() and (ft+pd.Timedelta(minutes=375)<S).all() and (pt>=S).all() and (pt<end).all()
  assert ((ft.hour%6==0)&(ft.minute==0)).all() and ((pt.hour%6==0)&(pt.minute==0)).all()
  masks={}
  for segment,pos in [('fit',fp),('predict',pp)]:
   mask=np.zeros(len(grid),dtype=bool);mask[pos]=True;masks[segment]=mask
   assert masksha(grid,mask)==s['mask_sha256'][segment] and positionmasksha(mask)==prov['mask_position_sha256'][segment] and prov['mask_counts'][segment]==s['counts'][segment] and prov['mask_ranges'][segment]==[int(pos[0]),int(pos[-1])]
   for g in GROUPS:
    x=fd[segment+'_features_'+g];assert x.dtype==np.float64 and x.shape==(len(pos),len(oldsource['features'][g])) and np.isfinite(x).all()
    assert matrixsha(x)==s['feature_selected_float64le_sha256'][segment][g]==prov[segment+'_features_sha256'][g] and digest(x.tolist())==s['feature_selected_sha256'][segment][g]==p[segment+'_features_sha256'][g]
    counts['saved_feature_matrices_bound']+=1
   exact(fd[segment+'_features_perp_delay0'][:,:29],fd[segment+'_features_technical'])
  assert fd['returns'].shape==fd['binary_labels'].shape==(n,) and np.isfinite(fd['returns']).all()
  assert matrixsha(fd['returns'])==prov['fit_return_sha256'] and digest(fd['returns'].tolist())==s['continuous_return_selected_sha256']['fit'] and matrixsha(fd['binary_labels'])==prov['fit_binary_labels_sha256']
  assert fd['binary_labels'].dtype==np.int64 and np.isin(fd['binary_labels'],[0,1]).all()
  counts['saved_continuous_return_vectors_bound']+=1;counts['saved_binary_label_vectors_bound']+=1
  for w in WEIGHTINGS:
   weights=fd['weights_'+w];assert weights.shape==(n,) and np.isfinite(weights).all() and (weights>=0).all() and matrixsha(weights)==prov['sample_weights'][w]['weight_sha256']
   counts['saved_weight_vectors_bound']+=1
  for g in GROUPS:
   assert matrixsha(np.column_stack((fd['fit_features_'+g],fd['returns'])))==prov['fit_features_and_return_sha256'][g]
  for mid in MODEL_IDS:
   state=prov['fitted_state'][mid];g,w=state['group'],state['weighting'];assert mid==g+'_'+w and state['scaler_rows']==n
   mp=src/'models'/f'fold{f}_{mid}.joblib';inputbindings[str(mp)]=bindings[str(mp)];model=joblib.load(mp);scaler,logistic=model.steps[0][1],model.steps[1][1]
   assert [name for name,_ in model.steps]==['standardscaler','logisticregression'] and logistic.get_params()==prov['parameters']['logistic'] and logistic.C==1.0 and not logistic.warm_start
   for attribute,key in [('mean_','scaler_mean'),('var_','scaler_variance'),('scale_','scaler_scale')]:exact(getattr(scaler,attribute),np.asarray(state[key],dtype=float))
   for attribute,key in [('coef_','coefficient'),('intercept_','intercept')]:exact(getattr(logistic,attribute),np.asarray(state[key],dtype=float));assert matrixsha(getattr(logistic,attribute))==state[key+'_sha256']
   exact(logistic.classes_,np.asarray(state['classes'],dtype=np.int64));exact(logistic.n_iter_,np.asarray(state['n_iter'],dtype=logistic.n_iter_.dtype));assert int(scaler.n_samples_seen_)==n
   assert matrixsha(np.column_stack((fd['fit_features_'+g],fd['binary_labels'],fd['weights_'+w])))==state['fit_features_labels_weights_sha256']
   counts['saved_models_state_verified']+=1
  cix=pd.date_range(S,E,freq='15min',inclusive='left');ix=pd.date_range(E,end,freq='15min',inclusive='left');calpositions=grid.get_indexer(cix);evalpositions=grid.get_indexer(ix)
  base=arr(parity/'forecasts'/f'fold{f}_technical_half.npz');exact(base['timestamps'],ix.asi8)
  inf,score=base['inference_mask'],base['score_support'];assert inf.dtype==score.dtype==bool and not (score&~inf).any()
  exact(inf,masks['predict'][evalpositions]);assert (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  for key,v in [('inference',inf),('score',score)]:
   mask=np.zeros(len(grid),dtype=bool);mask[evalpositions]=v;assert masksha(grid,mask)==s['mask_sha256'][key];masks[key]=mask
  classifierseen={}
  for name in newmeans:
   mapping=cfg['mapping'][name];g=mapping['group'];mid=mapping['classifier_id'];ep=src/'forecasts'/f'fold{f}_{name}.npz';ca_path=src/'calibration'/f'fold{f}_{name}.npz';a=arr(ep,FORECAST_KEYS);ca=arr(ca_path,CAL_KEYS);inputbindings[str(ep)]=bindings[str(ep)];inputbindings[str(ca_path)]=bindings[str(ca_path)]
   exact(a['timestamps'],ix.asi8);exact(ca['timestamps'],cix.asi8);exact(a['inference_mask'],inf);exact(a['score_support'],score)
   exact(np.isfinite(a['mu']),inf);exact(np.isfinite(a['variance']),inf);exact(np.isfinite(a['logit']),inf);exact(np.isfinite(a['probability']),inf)
   exact(ca['classifier_predict_mask'],masks['predict'][calpositions]);exact(ca['mapped_inference_mask'],ca['classifier_predict_mask']&np.asarray(cix>=I));exact(np.isfinite(ca['mu']),ca['mapped_inference_mask'])
   oldcal=arr(parity/'calibration'/f'fold{f}_{g}.npz');original=arr(parity/'forecasts'/f"fold{f}_{mapping['parent_mean']}.npz")
   for key in ('timestamps','actual','scale_mask','interval_mask'):exact(ca[key],oldcal[key])
   for key in ('timestamps','actual','variance','fit_return_mean','inference_mask','score_support'):exact(a[key],original[key])
   exact(a['parent_mu'],original['mu'])
   for key in ('scale','interval'):
    mask=np.zeros(len(grid),dtype=bool);mask[calpositions]=ca[key+'_mask'];assert masksha(grid,mask)==s['mask_sha256'][key];masks[key]=mask
    assert scores[f,'interval',name]['rows']==s['counts']['interval'] and scores[f,'evaluation',name]['rows']==s['counts']['score']
   if mid in MODEL_IDS:
    z=np.concatenate((ca['logit'][ca['classifier_predict_mask']],a['logit'][inf]));prob=np.concatenate((ca['probability'][ca['classifier_predict_mask']],a['probability'][inf]));state=prov['fitted_state'][mid]
    assert len(z)==len(pp) and matrixsha(z)==state['predict_logits_sha256'] and matrixsha(prob)==state['predict_probability_sha256']
    classifierseen[mid]=True
   counts['saved_E_forecasts_bound']+=1;counts['saved_calibration_arrays_bound']+=1
  assert set(classifierseen)==set(MODEL_IDS)
  for name in masks:assert int(masks[name].sum())==s['counts'][name]
  for c in controls:
   r=rows[f,c];tp=src/'targets'/f'fold{f}_{c}.npz';assert r['targets_sha256']==bindings[str(tp)];ta=arr(tp,{'timestamps','targets'});exact(ta['timestamps'],ix.asi8);counts['parent_target_calendars_bound']+=1
   if c in cfg['new_policy_ids']:assert r['trace_sha256']==bindings[str(src/'traces'/f'fold{f}_{c}.json')]
  assert rows[f,'bh']['regime']['trend']==s['regime']
  for segment in ('interval','evaluation'):
   count=s['counts']['interval' if segment=='interval' else 'score']
   for name in means:assert scores[f,segment,name]['rows']==count and scores[f,segment,name]['regime']==rows[f,'bh']['regime']
   for name in classifiers:assert cs[f,segment,name]['rows']==count and cs[f,segment,name]['regime']==rows[f,'bh']['regime']
  records.append({'fold':f,'calendar':s['calendar'],'counts':s['counts'],'mask_sha256':s['mask_sha256'],'feature_columns':p['feature_columns'],'feature_selected_sha256':s['feature_selected_sha256'],'feature_selected_float64le_sha256':s['feature_selected_float64le_sha256'],'continuous_fit_return_sha256':s['continuous_return_selected_sha256']['fit'],'binary_fit_labels_sha256':prov['fit_binary_labels_sha256'],'saved_weight_sha256':{w:prov['sample_weights'][w]['weight_sha256'] for w in WEIGHTINGS},'fit_positions_sha256':hashlib.sha256(fp.astype('<i8').tobytes()).hexdigest(),'predict_positions_sha256':hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest(),'fit_data_path':str(fitpath),'fit_data_sha256':bindings[str(fitpath)],'fit_provenance_path':str(provenancepath),'fit_provenance_sha256':bindings[str(provenancepath)],'label_maturity_last':s['label_maturity_last'],'fallback_rows':s['fallback_rows'],'missing_current_open_rows':s['missing_current_open_rows'],'regime':s['regime'],'new_statistics_computed':False})
  for k in ('inference','score'):totals[k]+=s['counts'][k]
  totals['fallback']+=s['fallback_rows'];totals['missing_current_open']+=s['missing_current_open_rows'];regimes[s['regime']]+=1
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2} and regimes=={'bull':2,'bear':4,'sideways':2}
 for k,n in {'saved_feature_matrices_bound':32,'saved_continuous_return_vectors_bound':8,'saved_binary_label_vectors_bound':8,'saved_weight_vectors_bound':16,'saved_models_state_verified':32,'saved_E_forecasts_bound':64,'saved_calibration_arrays_bound':64,'parent_target_calendars_bound':416}.items():assert counts[k]==n
 proposed={'status':'proposed_not_registered_or_executed','new_C_rule':'1 / float(np.sum(frozen_T_weights)); same frozen weighting-specific T vector; no grid; C values not computed in this preflight','groups':list(GROUPS),'loss_weightings':list(WEIGHTINGS),'missing_rules':list(RULES),'new_classifier_fits':32,'new_learned_mean_streams':4,'new_causal_policy_names':8,'old_control_policies':52,'total_policy_names':60,'economic_rows':480,'base_stress_accounts':960,'adaptive_causal_names_before':190,'adaptive_causal_names_after':198,'new_fit_prior_estimates':0,'source_original_fit_counts':[r['counts']['fit'] for r in records]}
 report={'passed':True,'schema':'oracle-regularized-direction-source-audit-v1','scope':'DATA-ONLY reuse of completed Stage17 saved selected arrays, original labels/weights, exact old model states, support and immutable provenance. No canonical helper or market parser imported; no fit, new label/prior/weight, new coefficient/logit/forecast/loss/order computation.','script_sha256':sha(__file__),'parent_source_revision':SOURCE,'parent_file_sha256':HASHES,'parent_registration_canonical_sha256':digest(reg),'counts':dict(counts),'distinct_hashed_files':len(verified),'source_artifact_inventory_sha256':digest(bindings),'source_artifact_bindings':bindings,'direct_source_bindings':direct,'saved_input_bindings':inputbindings,'old_source_audit_sha256':oldsourcehash,'old_completed_audit_sha256':oldauditsha,'spot_data_proof':pre['spot_data_proof'],'um_data_proof':pre['um_data_proof'],'features':oldsource['features'],'support':records,'totals':dict(totals),'regime_counts':dict(regimes),'regime_gate_pass':False,'full_grid':{'start_inclusive':grid[0].isoformat(),'end_exclusive':cut.isoformat(),'rows':len(grid),'source':'Reconstructed from immutable saved positions/timestamps and registered cutoff; no price parsing.'},'proposed_counts':proposed,'limitations':['This verifies existing Stage17 artifacts and labels/weights by immutable hash plus the completed independent audit, without recalculating their statistical definitions.','The fixed C=1/float(np.sum(frozen_T_weights)) rule and all contrasts must be registered before new model fitting; no smaller grid or outcome-driven retry is authorized by this audit.','Raw market files are byte-hashed for provenance but no parquet or archive price values are decoded.','Original future-dependent common availability, repeated development reuse, absent historical receipt evidence and the failing 2/4/2 regime coverage remain.','The 24 old hindsight control policies remain nondeployable diagnostics and cannot be causal teachers.','Stage17 matmul warnings remain recorded in its independent audit; this source audit does not establish a cause or permit ignoring future numerical guard failures.']}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'path':str(OUT),'sha256':sha(OUT),'counts':dict(counts),'totals':dict(totals),'proposed_counts':proposed}))
if __name__=='__main__':main()
