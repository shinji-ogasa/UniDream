"""Stage19 final input-only preflight binding audit. No mapping computation."""
from pathlib import Path
import hashlib,json,os,math
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 os.chdir(ROOT);checks=0;verified={}
 def verify(p,h):
  nonlocal checks
  p=Path(p).resolve()
  if p not in verified:verified[p]=sha(p)
  assert verified[p]==h,(str(p),h,verified[p]);checks+=1
 sourcepath=Path('/tmp/oracle_soft_direction_source_audit_20260906.json');sourcehash='4965de3c0ba40741fb9bff7d588a2e62ec19fbd6b547d013a032a99bd3c2b96b';verify(sourcepath,sourcehash);source=read(sourcepath);assert source['passed']
 verify('/tmp/oracle_soft_direction_source_audit_20260906.py',source['script_sha256'])
 cp=Path('configs/oracle_soft_direction_decisions_20260906.yaml');ch='4c0a5a24cf5ece17bc0ded40aa96fefef86b4f5bcc0b6fa282bf5d2c4b2582b2';verify(cp,ch);cfg=yaml.safe_load(cp.read_text())
 out=Path(cfg['output_dir']);pp=out/'preflight.json';ph='be95da2abe1581b516cc8a0fd168ae919ad89527280818b283f46b92c95f2154';verify(pp,ph);pre=read(pp)
 assert not (out/'results.json').exists() and cfg['preflight_sha256']==ph and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 parent=Path(cfg['parent_root']);reg=read(parent/'registration.json');oldpre=read(parent/'preflight.json');oldcfg=reg['config']
 assert cfg['parent_source_revision']==source['parent_source_revision']==reg['source_revision']=='5a82c270c64a342ab7e9df8105b7d23d1336d876'
 assert pre['parent_registration_canonical_sha256']==source['parent_registration_canonical_sha256']==digest(reg)
 verify(cfg['parent_config'],cfg['parent_config_sha256']);assert yaml.safe_load(Path(cfg['parent_config']).read_text())==oldcfg
 assert set(cfg['source_bindings'])==set(oldcfg['source_bindings'])|{'unidream/experiments/oracle_soft_direction_mapping.py','unidream/experiments/oracle_soft_direction_decisions.py'} and len(cfg['source_bindings'])==33
 assert pre['source_bindings']==cfg['source_bindings']
 for p,h in oldcfg['source_bindings'].items():assert cfg['source_bindings'][p]==h
 assert len({str(Path(p).resolve()) for p in cfg['source_bindings']})==33
 for p,h in cfg['source_bindings'].items():verify(p,h)
 assert set(cfg['parent_manifest_bindings'])=={str(parent/n) for n in ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in range(5,13)]}
 for p,h in cfg['parent_manifest_bindings'].items():verify(p,h)
 assert pre['source_artifact_bindings']==source['source_artifact_bindings'] and len(pre['source_artifact_bindings'])==3488
 assert len({str(Path(p).resolve()) for p in pre['source_artifact_bindings']})==3488
 for p,h in {**source['source_artifact_bindings'],**source['direct_source_bindings'],**source['saved_input_bindings'],**pre['direct_source_bindings']}.items():verify(p,h)
 assert pre['spot_data_proof']==source['spot_data_proof'] and pre['um_data_proof']==source['um_data_proof']
 for p,h in oldpre['direct_source_bindings'].items():assert pre['direct_source_bindings'][p]==h
 for p,h in cfg['source_bindings'].items():assert pre['direct_source_bindings'][p]==h
 assert pre['new_statistics_mapped_predictions_losses_or_orders_computed'] is False and pre['no_estimator_fit_or_predict_called'] is True
 old_byfold={s['fold']:s for s in oldpre['support']};ind_byfold={s['fold']:s for s in source['support']};assert len(pre['support'])==8
 checked_inputs=0;checked_prior_pairs=0;checked_stats=0
 for p in pre['support']:
  f=p['fold'];s=ind_byfold[f]
  for k,v in old_byfold[f].items():assert p[k]==v,('old support identity',f,k)
  assert p['new_fits_or_feature_construction'] is False
  for k in p['counts']:assert p['counts'][k]==s['counts'][k] and p['mask_sha256'][k]==s['mask_sha256'][k]
  stat=p['saved_mapping_scalars'];saved=read(s['training_scalar_source_path']);assert stat['fit_abs_return_mean']==saved['fit_abs_return_mean'] and stat['fit_return_mean']==saved['fit_return_mean'] and stat['fit_statistical_magnitude_prior']==saved['fit_priors']['magnitude']
  assert all(type(stat[k]) is float and math.isfinite(stat[k]) for k in ('fit_abs_return_mean','fit_return_mean','fit_statistical_magnitude_prior'))
  assert stat['fit_abs_return_mean']>0 and 0<stat['fit_statistical_magnitude_prior']<1;checked_stats+=1
  expected_inputs={}
  for stream in s['probability_streams'].values():
   mid=stream['classifier_id']
   for seg,kind in [('interval','calibration'),('evaluation','forecast')]:expected_inputs[mid+'/'+seg]={'path':stream[kind+'_path'],'sha256':stream[kind+'_sha256']};checked_inputs+=1
  for g,a in s['saved_prior_probability_sources'].items():
   assert stat['prior_probability'][g]==a['saved_prior_probability'] and stat['fit_statistical_magnitude_prior']==a['raw_fit_prior_probability']
   for seg,kind in [('interval','calibration'),('evaluation','forecast')]:
    binding={'path':a[kind+'_path'],'sha256':a[kind+'_sha256']};expected_inputs[g+'/'+seg+'/prior']=binding;assert stat['prior_paths'][g][seg]==binding;checked_prior_pairs+=1
  assert len(set(stat['prior_probability'].values()))==1 and p['saved_probability_inputs']==expected_inputs
 assert (checked_inputs,checked_prior_pairs,checked_stats)==(64,32,8)
 for k,n in {'new_model_fits':0,'new_unique_fit_priors':0,'new_causal_names':20,'new_learned_policy_names':8,'new_constant_policy_names':12,'adaptive_prior_causal_names':198,'adaptive_total_causal_names':218,'score_classification_records':160,'score_return_records':384,'mapping_diagnostic_records':64,'fold_artifacts':121,'economic_rows':640,'economic_accounts':1280,'inference_rows':2586,'score_rows':2574,'interval_inference_rows':2537,'interval_score_rows':2523,'fallback_rows':332,'missing_current_open_rows':2}.items():assert cfg[k]==n
 assert cfg['prior_identity_absolute_tolerance']==1e-14 and cfg['prior_identity_relative_tolerance']==1e-12
 reviewpath=Path('/tmp/oracle_soft_direction_prerun_review_20260906.json');review=read(reviewpath)
 for p,h in review['reviewed_files_sha256'].items():verify(p,h)
 assert review['status']=='pass_source_review' and review['data_only_source_audit_sha256']==sourcehash
 report={'status':'pass_pre_freeze_input_bindings','script_sha256':sha(__file__),'config_path':str(cp),'config_sha256':ch,'preflight_path':str(pp),'preflight_sha256':ph,'data_only_input_audit_sha256':sourcehash,'source_review_sha256':sha(reviewpath),'source_bindings':cfg['source_bindings'],'source_count':33,'artifact_count':3488,'source_artifact_inventory_sha256':digest(pre['source_artifact_bindings']),'hash_binding_checks':checks,'distinct_hashed_files':len(verified),'support_folds_verified':8,'probability_input_bindings_verified':checked_inputs,'prior_probability_bindings_verified':checked_prior_pairs,'saved_T_scalar_sets_verified':checked_stats,'support_totals':source['totals'],'new_statistics_mapping_loss_or_order_computed':False,'findings':[],'execution_authorized_by_this_audit':False,'pending':['Registered source/config/protocol/tests/preflight must be committed and pushed and the full suite pass before the root authorizes the real mapping run.']}
 p=Path('/tmp/oracle_soft_direction_preflight_binding_audit_20260906.json');p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'path':str(p),'json_sha256':sha(p),'script_sha256':report['script_sha256'],'hash_checks':checks,'distinct_files':len(verified),'runner_sha256':cfg['source_bindings']['unidream/experiments/oracle_soft_direction_decisions.py']}))
if __name__=='__main__':main()
