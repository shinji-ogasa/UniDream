"""Post-run descriptive mapping summaries and saved-target identity; no replay."""
from collections import defaultdict
import hashlib,json,math
from pathlib import Path
import numpy as np
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/oracle_soft_direction_decisions_v1'
AUDIT=Path('/tmp/oracle_soft_direction_mapping_audit_20260906.json')
REPORT=Path('/tmp/oracle_soft_direction_mechanism_summary_20260906.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mean(a):return math.fsum(float(v)/len(a) for v in a)
assert sha(AUDIT)=='d5bc7358bb75f4a8f278b590355bff46955320d0b51a5b3738c37e33bbb40eef'
assert sha(OUT/'results.json')=='d85e14e1a8249601a1f28c0d4fa29b1fbb23a3571232dcedc5edee0644b82cc5'
a=read(AUDIT);r=read(OUT/'results.json');cfg=yaml.safe_load((ROOT/'configs/oracle_soft_direction_decisions_20260906.yaml').read_text())
assert a['passed']
mechanisms={};by_kind=defaultdict(int)
for mean_id in sorted({v['mean_id'] for v in a['mapping_diagnostics']}):
 mechanisms[mean_id]={}
 for seg in ('interval','evaluation'):
  rows=[v for v in a['mapping_diagnostics'] if v['mean_id']==mean_id and v['segment']==seg]
  assert len(rows)==8;n=sum(v['rows'] for v in rows)
  d={'quarters':8,'inference_rows':n,'unscored_inference_rows':sum(v['unscored_inference_rows'] for v in rows)}
  countkeys=('probability_half_rows','probability_zero_rows','probability_one_rows','source_zero_logit_rows',
   'mapped_zero_mean_rows','probability_direction_vs_logit_disagreements','mapped_direction_vs_logit_disagreements',
   'new_abs_mu_greater_than_hard_rows','new_abs_mu_equal_to_hard_rows')
  for key in countkeys:
   d[key]=sum(v[key] for v in rows);by_kind[key]+=d[key]
  d['new_abs_mu_less_than_hard_rows']=n-d['new_abs_mu_greater_than_hard_rows']-d['new_abs_mu_equal_to_hard_rows']
  d['new_abs_mu_greater_than_hard_fraction_row_pooled']=d['new_abs_mu_greater_than_hard_rows']/n
  for key in ('mean_abs_new_mu','mean_abs_hard_mu','mean_abs_parent_mu'):
   d[key+'_equal_quarter']=mean([v[key] for v in rows])
   d[key+'_row_pooled']=math.fsum(v[key]*v['rows']/n for v in rows)
  mechanisms[mean_id][seg]=d
rows_index={(v['fold'],v['candidate_id']):v for v in r['rows']}
clusters=[];pair_records=[];target_sources={}
for fold in range(5,13):
 fm=read(OUT/f'fold_{fold}.json');targets={};grouped=defaultdict(list)
 for cid in cfg['new_policy_ids']:
  path=OUT/'targets'/f'fold{fold}_{cid}.npz';key=str(path.relative_to(ROOT));h=sha(path)
  assert fm['artifact_sha256'][key]==h;target_sources[key]=h
  with np.load(path,allow_pickle=False) as z:times=z['timestamps'].copy();target=z['targets'].copy()
  targets[cid]=(times,target)
  canonical=target.astype('<f8',copy=True);canonical[np.isnan(canonical)]=np.nan
  digest=hashlib.sha256(times.astype('<i8').tobytes()+canonical.tobytes()).hexdigest()
  grouped[digest].append(cid)
 for ids in grouped.values():
  if len(ids)>1:
   base=targets[ids[0]]
   for cid in ids[1:]:
    assert np.array_equal(base[0],targets[cid][0]) and np.array_equal(base[1],targets[cid][1],equal_nan=True)
   clusters.append({'fold':fold,'candidate_ids':ids,'same_base_stress_metrics':all(
    rows_index[fold,c][cost]==rows_index[fold,ids[0]][cost] for c in ids for cost in ('base','stress_2x'))})
 for rule in cfg['rules']:
  pairs=[('cross_group_'+kind,'technical_soft_'+kind+'_'+rule,'perp_delay0_soft_'+kind+'_'+rule)
         for kind in ('mapped_prior','fit_mean','zero')]
  pairs.extend((g+'_mapped_prior_vs_fit_mean',g+'_soft_mapped_prior_'+rule,g+'_soft_fit_mean_'+rule) for g in ('technical','perp_delay0'))
  for label,left,right in pairs:
   lt,la=targets[left];rt,ra=targets[right]
   equal=bool(np.array_equal(lt,rt) and np.array_equal(la,ra,equal_nan=True))
   pair_records.append({'fold':fold,'rule':rule,'comparison':label,'left':left,'right':right,
    'targets_exact':equal,'same_base_stress_metrics':all(rows_index[fold,left][cost]==rows_index[fold,right][cost] for cost in ('base','stress_2x')),
    'different_target_rows':int((~((la==ra)|(np.isnan(la)&np.isnan(ra)))).sum())})
assert len(target_sources)==160 and len(pair_records)==80
constant=a['mapping_scalars']
output={'scope':'Descriptive aggregation of audited mapping quantities plus exact saved-target/saved-metric identity; no fit, scoring or policy replay',
 'script_sha256':sha(__file__),'source_results_sha256':sha(OUT/'results.json'),'source_mapping_audit_sha256':sha(AUDIT),
 'constant_identity_by_fold':constant,
 'max_abs_mapped_prior_fit_mean_residual':max(abs(v['mapped_prior_minus_fit_mean']) for v in constant),
 'max_abs_saved_vs_statistical_prior_difference':max(abs(v['saved_prior_minus_statistical_prior']) for v in constant),
 'nonzero_mapped_prior_fit_mean_residual_folds':sum(v['mapped_prior_minus_fit_mean']!=0 for v in constant),
 'nonzero_saved_vs_statistical_prior_difference_folds':sum(v['saved_prior_minus_statistical_prior']!=0 for v in constant),
 'mechanism_summary':mechanisms,'mechanism_aggregate_counts':dict(by_kind),'new_policy_duplicate_target_clusters':clusters,
 'fixed_constant_target_comparisons':pair_records,'fixed_constant_target_comparisons_exact':sum(v['targets_exact'] for v in pair_records),
 'fixed_constant_saved_metrics_comparisons_exact':sum(v['same_base_stress_metrics'] for v in pair_records),
 'source_target_sha256':target_sources,'limitations':['Duplicate target paths do not make floating mean arrays identical.',
  'Probabilities/logits retained inside constant mean NPZs are source evidence, not forecasts of those constant controls.',
  'Same saved metrics are checked for identity here; independent economic recomputation belongs to the separate accounting audit.',
  'Mechanism counts use all inference rows, not future-score support; equal-quarter and pooled-row means are distinct.']}
REPORT.write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(REPORT),'sha256':sha(REPORT),'mechanisms':mechanisms,
 'aggregate_counts':dict(by_kind),'constant_residuals':[{k:v for k,v in s.items() if k!='prior_bindings'} for s in constant],
 'constant_target_pairs_exact':output['fixed_constant_target_comparisons_exact'],
 'constant_saved_metric_pairs_exact':output['fixed_constant_saved_metrics_comparisons_exact'],
 'duplicate_clusters':clusters},indent=2))
