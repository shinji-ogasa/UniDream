"""Post-run descriptive aggregation of registered mechanism records; no fitting."""
from pathlib import Path
import hashlib,json,math
import numpy as np

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/oracle_regularized_direction_decisions_v1'
PARENT=ROOT/'codex_outputs/oracle_direction_decisions_v1'
AUDIT=Path('/tmp/oracle_regularized_direction_model_audit_20260906.json')
REPORT=Path('/tmp/oracle_regularized_direction_mechanism_summary_20260906.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mean(v):return math.fsum(float(a)/len(v) for a in v)
assert sha(OUT/'results.json')=='ab2b62a017a70fd65756614f3940c3194d60b8c0ebdab02aa058c988ccbdb678'
assert sha(AUDIT)=='f8f533d70b2978ce53e1fd0951dc0b17d423589de32777ecc025c9b39f8e5159'
r=read(OUT/'results.json');a=read(AUDIT);pre=read(OUT/'preflight.json');sources={}
assert a['passed']
models={};directions={};enriched=[]
for mid in sorted({v['model_id'] for v in a['models']}):
 rows=[v for v in a['models'] if v['model_id']==mid];assert len(rows)==8
 ratios=[v['coefficient_l2_norm']/v['old_coefficient_l2_norm'] for v in rows]
 models[mid]={'folds':8,'new_norm_equal_fold_mean':mean([v['coefficient_l2_norm'] for v in rows]),
  'old_norm_equal_fold_mean':mean([v['old_coefficient_l2_norm'] for v in rows]),
  'new_to_old_norm_ratio_equal_fold_mean':mean(ratios),'norm_ratio_min':min(ratios),'norm_ratio_max':max(ratios),
  'strictly_smaller_norm_folds':sum(v['coefficient_l2_norm']<v['old_coefficient_l2_norm'] for v in rows),
  'C_min':min(v['C'] for v in rows),'C_max':max(v['C'] for v in rows),
  'actual_lambda_values':sorted({v['actual_l2_strength'] for v in rows})}
 for segment in ('interval','evaluation'):
  ds=[v for v in a['direction_diagnostics'] if v['classifier_id']==mid and v['segment']==segment]
  assert len(ds)==8
  for d in ds:
   f=d['fold'];kind='forecasts' if segment=='evaluation' else 'calibration'
   old_mid=mid.removesuffix('_l2unit');weighting=old_mid.rsplit('_',1)[1]
   path=PARENT/kind/f'fold{f}_{old_mid}_direction.npz';key=str(path.relative_to(ROOT))
   assert sha(path)==pre['source_artifact_bindings'][key];sources[key]=sha(path)
   with np.load(path,allow_pickle=False) as z:
    support=z['inference_mask'] if segment=='evaluation' else z['mapped_inference_mask']
    old_z=z['logit'][support];assert len(old_z)==d['rows'] and np.isfinite(old_z).all()
   prior=next(v for v in r['fit_records'] if v['fold']==f)['fit_prior_logits'][weighting]
   d['old_sign_matches_matched_prior']=int((np.sign(old_z)==np.sign(prior)).sum())
   enriched.append(d)
  n=sum(v['rows'] for v in ds)
  item={'folds':8,'inference_rows':n,'unscored_inference_rows':sum(v['unscored_inference_rows'] for v in ds),
   'sign_disagreements_vs_C1':sum(v['sign_disagreements_vs_C1'] for v in ds),
   'new_prior_agreement_count':sum(v['sign_matches_matched_prior'] for v in ds),
   'old_prior_agreement_count':sum(v['old_sign_matches_matched_prior'] for v in ds),
   'new_zero_logits':sum(v['zero_logit_rows'] for v in ds),'old_zero_logits':sum(v['old_zero_logit_rows'] for v in ds)}
  for label,key in [('sign_disagreement_rate','sign_disagreements_vs_C1'),('new_prior_agreement_rate','sign_matches_matched_prior'),('old_prior_agreement_rate','old_sign_matches_matched_prior')]:
   item[label+'_row_pooled']=sum(v[key] for v in ds)/n
   item[label+'_equal_fold_mean']=mean([v[key]/v['rows'] for v in ds])
  for label,key in [('new_mean_abs_logit','mean_abs_logit'),('old_mean_abs_logit','old_mean_abs_logit')]:
   item[label+'_equal_fold_mean']=mean([v[key] for v in ds])
   item[label+'_row_pooled']=math.fsum(v[key]*v['rows']/n for v in ds)
  directions[mid,segment]=item
output={'scope':'Post-run descriptive aggregation of the fixed mechanism records plus old-model matched-prior sign agreement; no fit, selection, outcome scoring or policy replay',
 'source_results_sha256':sha(OUT/'results.json'),'independent_audit_sha256':sha(AUDIT),'script_sha256':sha(__file__),
 'model_norms':models,'direction_summary':{mid:{seg:directions[mid,seg] for seg in ('interval','evaluation')} for mid in models},
 'enriched_fold_records':enriched,'source_old_prediction_sha256':sources,
 'row_pooled_does_not_replace_registered_equal_quarter_scores':True,
 'interpretation_limits':['Coefficient shrinkage and sign agreement are descriptive mechanism changes, not forecast skill or economic success.',
   'Mean absolute logits and signs use all inference rows; future score availability is not a filter.',
   'I regimes, when used elsewhere, refer retrospectively to E-start regimes. No new regime split is introduced here.']}
REPORT.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'path':str(REPORT),'sha256':sha(REPORT),'models':models,'directions':output['direction_summary'],
 'audit_runtime_log':a['runtime_log'],'distinct_hashed_files':a['distinct_hashed_files'],
 'minimum_hessian_eigenvalue':a['minimum_hessian_eigenvalue']},indent=2))
