"""Stage18 formatted report audit, prepared before the final-report notification.

Only an explicitly supplied final report SHA can enable this read-only audit.
No model fitting, policy replay, canonical summary call or report/source edit.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import math
import re
import numpy as np

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
AUDIT=Path('/tmp/oracle_regularized_direction_score_audit_20260906.json')
MECHANISM=Path('/tmp/oracle_regularized_direction_mechanism_summary_20260906.json')
MODEL=Path('/tmp/oracle_regularized_direction_model_audit_20260906.json')
OWN=Path('/tmp/oracle_regularized_direction_audit_20260906.json')
SOURCE=Path('/tmp/oracle_regularized_direction_source_audit_20260906.json')
OLD_OBJECTIVE=Path('/tmp/oracle_regularized_direction_old_objective_audit_20260906.json')
WARNING=Path('/tmp/oracle_regularized_direction_warning_audit_20260906.json')
REPORT=ROOT/'docs/experiments/oracle_regularized_direction_results_20260906.md'
OUTPUT=Path('/tmp/oracle_regularized_direction_report_audit_20260906.json')
OUT=ROOT/'codex_outputs/oracle_regularized_direction_decisions_v1'
EXPECTED_HASHES={
 AUDIT:'41d25ec5708b2f72cadd94949a24e86bd65660de53a7d712547d37b9ecb28ada',
 MECHANISM:'e90d71d02b5dc4c4eb72ad6bd5b29d32a588faca5f4ae62db5364d6e0598b850',
 MODEL:'f8f533d70b2978ce53e1fd0951dc0b17d423589de32777ecc025c9b39f8e5159',
 OWN:'a2d41ce9c4409883e17d4b85bb6440d6f799759df9ee5a0a0ac9fe7c697db8ec',
 SOURCE:'43d255bbc8e361d8ff9e78bbf308b9c1a476e90ee1442584ff0f678084ed69e7',
 OLD_OBJECTIVE:'afc45c88413b29aa4882c7ab25b01ce8961a0a8dfbb5f7454d6088c3c52acd98',
 WARNING:'22628e8df5eda4147795b045eecc714c88193e65857eee3bfe41cd2e2cf55157'}
GROUPS={'Technical29':'technical','Perp31':'perp_delay0'}
RULES={'hold':'utility_risk1','fallback':'utility_risk1_fallback_bh'}
SEGMENTS=('interval','evaluation')
COSTS=('base','stress_2x')
FS=tuple(range(5,13))
MODELS=tuple(g+'_'+w+'_l2unit' for g in GROUPS.values() for w in ('ordinary','magnitude'))
CHECKS=[]
ERRORS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dec(x):return Decimal(str(x))
def avg(values):return sum(map(dec,values),Decimal(0))/Decimal(len(values))
def parse_tables(text):
 blocks=[]
 for section in text.split('\n\n'):
  lines=[line for line in section.splitlines() if line.startswith('|')]
  if len(lines)>2:blocks.append([[v.strip() for v in line.strip('|').split('|')] for line in lines])
 return blocks

def check(observed,expected,location,decimals=6,scale=1):
 target=float(expected)*scale
 value=float(str(observed).replace('−','-').replace('%',''))
 good=math.isfinite(value) and abs(value-target)<=.500001*10**(-decimals)
 row={'location':location,'shown':observed,'expected':target,'passed':good}
 CHECKS.append(row)
 if not good:ERRORS.append(row)

def exact(observed,expected,location):
 good=observed==expected
 row={'location':location,'shown':observed,'expected':expected,'passed':good}
 CHECKS.append(row)
 if not good:ERRORS.append(row)

def classifier_id(label, default_new=True):
 if label.startswith('prior '):return 'prior_'+label.split()[1]
 group,weighting=label.split()[:2]
 return GROUPS[group]+'_'+weighting+('_l2unit' if default_new and 'C1' not in label else '')

def mean_id(label,default_new=True):return classifier_id(label,default_new)+'_direction'

def verify_mechanism(j,m):
 """Independently recheck the saved mechanism aggregation, including old signs.

 The only old prediction fields inspected here are timestamps, logit and masks;
 no actual returns are used for this mechanism report verification.
 """
 assert m['source_results_sha256']==j['source_sha256'][str((OUT/'results.json').relative_to(ROOT))]
 assert m['independent_audit_sha256']==EXPECTED_HASHES[MODEL]
 assert m['row_pooled_does_not_replace_registered_equal_quarter_scores'] is True
 keyed={(r['fold'],r['segment'],r['classifier_id']):r for r in j['direction_diagnostics']}
 enriched={(r['fold'],r['segment'],r['classifier_id']):r for r in m['enriched_fold_records']}
 assert len(keyed)==len(enriched)==64 and set(keyed)==set(enriched)
 for key,r in keyed.items():
  e=enriched[key]
  for k,v in r.items():
   if isinstance(v,float):assert math.isclose(v,e[k],rel_tol=1e-12,abs_tol=1e-12),(key,k)
   else:assert v==e[k],(key,k)
  f,seg,mid=key
  oldmid=mid.removesuffix('_l2unit');kind='calibration' if seg=='interval' else 'forecasts'
  path=ROOT/'codex_outputs/oracle_direction_decisions_v1'/kind/f'fold{f}_{oldmid}_direction.npz'
  assert sha(path)==m['source_old_prediction_sha256'][str(path.relative_to(ROOT))]
  with np.load(path,allow_pickle=False) as source:
   support=source['mapped_inference_mask'] if seg=='interval' else source['inference_mask']
   z=source['logit'][support];times=source['timestamps'][support]
   sm=source['interval_mask'] if seg=='interval' else source['score_support']
  w='ordinary' if '_ordinary' in mid else 'magnitude'
  prior=next(x for x in j['fit_checks'] if x['fold']==f)['weights'][w]['prior_logit']
  matches=sum((int(v>0)-int(v<0))==(int(prior>0)-int(prior<0)) for v in z)
  assert matches==e['old_sign_matches_matched_prior']
  assert int((support&~sm).sum())==e['unscored_inference_rows']
  assert hashlib.sha256(np.asarray(times,dtype='<i8').tobytes()).hexdigest()==e['inference_timestamp_sha256']
 for mid in MODELS:
  norms=[next(r for r in j['model_checks'] if r['fold']==f and r['model_id']==mid) for f in FS]
  v=m['model_norms'][mid]
  expected={'folds':8,'old_norm_equal_fold_mean':avg([r['old_coefficient_l2_norm'] for r in norms]),
   'new_norm_equal_fold_mean':avg([r['coefficient_l2_norm'] for r in norms]),
   'new_to_old_norm_ratio_equal_fold_mean':avg([r['coefficient_l2_norm']/r['old_coefficient_l2_norm'] for r in norms]),
   'norm_ratio_min':min(r['coefficient_l2_norm']/r['old_coefficient_l2_norm'] for r in norms),
   'norm_ratio_max':max(r['coefficient_l2_norm']/r['old_coefficient_l2_norm'] for r in norms),
   'strictly_smaller_norm_folds':sum(r['coefficient_l2_norm']<r['old_coefficient_l2_norm'] for r in norms),
   'C_min':min(r['C'] for r in norms),'C_max':max(r['C'] for r in norms),
   'actual_lambda_values':sorted({r['actual_l2_strength'] for r in norms})}
  assert set(v)==set(expected)
  for k,target in expected.items():
   if isinstance(target,list):assert v[k]==target
   else:assert math.isclose(v[k],float(target),rel_tol=1e-12,abs_tol=1e-12),(mid,k)
  for seg in SEGMENTS:
   rr=[enriched[f,seg,mid] for f in FS];v=m['direction_summary'][mid][seg];n=sum(r['rows'] for r in rr)
   expected={'folds':8,'inference_rows':n,'unscored_inference_rows':sum(r['unscored_inference_rows'] for r in rr),
    'new_zero_logits':sum(r['zero_logit_rows'] for r in rr),'old_zero_logits':sum(r['old_zero_logit_rows'] for r in rr),
    'sign_disagreements_vs_C1':sum(r['sign_disagreements_vs_C1'] for r in rr),
    'new_prior_agreement_count':sum(r['sign_matches_matched_prior'] for r in rr),
    'old_prior_agreement_count':sum(r['old_sign_matches_matched_prior'] for r in rr)}
   for key,field in [('sign_disagreement_rate','sign_disagreements_vs_C1'),
                     ('new_prior_agreement_rate','sign_matches_matched_prior'),
                     ('old_prior_agreement_rate','old_sign_matches_matched_prior')]:
    expected[key+'_row_pooled']=sum(r[field] for r in rr)/n
    expected[key+'_equal_fold_mean']=avg([r[field]/r['rows'] for r in rr])
   for key,field in [('new_mean_abs_logit','mean_abs_logit'),('old_mean_abs_logit','old_mean_abs_logit')]:
    expected[key+'_equal_fold_mean']=avg([r[field] for r in rr])
    expected[key+'_row_pooled']=sum(dec(r[field])*r['rows'] for r in rr)/Decimal(n)
   assert set(v)==set(expected),(mid,seg,set(v)^set(expected))
   for k,target in expected.items():assert math.isclose(v[k],float(target),rel_tol=1e-12,abs_tol=1e-12),(mid,seg,k)
 return {'enriched_fold_records':64,'model_norm_rows':4,'direction_summary_rows':8,'old_prediction_bindings':len(m['source_old_prediction_sha256'])}

# Final report table/narrative handlers will be added only after root notification.
def audit_report(expected_report_sha):
 assert sha(REPORT)==expected_report_sha
 assert not OUTPUT.exists(), 'Preserve any earlier saved report audit.'
 for path,expected in EXPECTED_HASHES.items():assert sha(path)==expected,(path,'audit binding changed')
 j=read(AUDIT);s=j['summary'];m=read(MECHANISM);mj=read(MODEL);oj=read(OWN);src=read(SOURCE)
 oldobj=read(OLD_OBJECTIVE);warn=read(WARNING)
 assert j['passed'] and mj['passed'] and src['passed'] and oldobj['passed'] and oj['status']=='pass'
 text=REPORT.read_text();blocks=parse_tables(text)
 with localcontext() as ctx:
  ctx.prec=60
  mechanism_verification=verify_mechanism(j,m)
 counts={};seen={}
 def table(name,rows):
  assert name not in counts,(name,'duplicate table')
  counts[name]=len(rows);seen[name]=[]
 for block in blocks:
  h,rows=block[0],block[2:]
  if h[:2]==['Classifier','Rule']:
   table('economics',rows)
   for i,x in enumerate(rows):
    cid=mean_id(x[0])+'_'+RULES[x[1]];v=s['economics']['all'][cid];seen['economics'].append(cid)
    for k,(cost,key) in enumerate([('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta')]):check(x[k+2],v[cost][key],f'economics/{i}/{cost}/{key}',scale=100)
    exact(x[6],str(v['joint_positive_quarters_both_costs'])+'/8',f'economics/{i}/joint')
  elif h[:3]==['Classifier','Regime','Rule']:
   table('regime',rows)
   for i,x in enumerate(rows):
    cid=mean_id(x[0])+'_'+RULES[x[2]];v=s['economics'][x[1]][cid];seen['regime'].append((cid,x[1]))
    for k,(cost,key) in enumerate([('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta')]):check(x[k+3],v[cost][key],f'regime/{i}/{cost}/{key}',scale=100)
    exact(x[7],f"{v['joint_positive_quarters_both_costs']}/{v['quarters']}",f'regime/{i}/joint')
  elif h[:3]==['Classifier','Reference','Rule']:
   table('paired_economics',rows)
   for i,x in enumerate(rows):
    mid=classifier_id(x[0]);mu=mid+'_direction';g=GROUPS[x[0].split()[0]]
    ref=mid.removesuffix('_l2unit')+'_direction' if x[1]=='same-loss C1' else g+'_half'
    assert x[1] in ('same-loss C1','own original half')
    v=s['paired']['all'][mu][ref]['economics'][RULES[x[2]]];seen['paired_economics'].append((mu,ref,RULES[x[2]]))
    for k,(cost,key) in enumerate([('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta')]):check(x[k+3],v[cost][key],f'paired/{i}/{cost}/{key}',scale=100)
  elif h[:2]==['Segment','Classifier']:
   table('classification',rows)
   for i,x in enumerate(rows):
    mid=classifier_id(x[1]);v=s['classification']['all'][x[0]][mid];seen['classification'].append((x[0],mid))
    for k,key in enumerate(['brier','log_loss','binary_accuracy','weighted_brier','weighted_log_loss','weighted_binary_accuracy']):check(x[k+2],v[key],f'classification/{i}/{key}',scale=100 if 'accuracy' in key else 1)
  elif h[:3]==['Classifier','Segment','MSE ×10^6']:
   table('mapped_mse',rows)
   for i,x in enumerate(rows):
    mid=classifier_id(x[0]);mu=mid+'_direction';seg=x[1];g=GROUPS[x[0].split()[0]];w=x[0].split()[1]
    refs=(mid.removesuffix('_l2unit')+'_direction',g+'_half',g+'_'+w+'_prior_direction')
    v=s['prediction']['all'][seg][mu]
    values=[v['return_mse'],*[s['paired']['all'][mu][ref]['prediction'][seg]['mse_difference'] for ref in refs],v['zero_return_mse'],v['fit_mean_return_mse']]
    seen['mapped_mse'].append((mu,seg))
    for k,value in enumerate(values):check(x[k+2],value,f'mapped_mse/{i}/{k}',scale=1e6)
  elif h[:2]==['Classifier','C1 norm mean']:
   table('norms',rows)
   for i,x in enumerate(rows):
    mid=classifier_id(x[0]);v=m['model_norms'][mid];seen['norms'].append(mid)
    for k,key in enumerate(('old_norm_equal_fold_mean','new_norm_equal_fold_mean','norm_ratio_min','norm_ratio_max')):check(x[k+1],v[key],f'norms/{i}/{key}',scale=100 if 'ratio' in key else 1)
    exact(x[5],f"{v['strictly_smaller_norm_folds']}/{v['folds']}",f'norms/{i}/folds')
  elif h[:3]==['Classifier','Segment','Inference rows']:
   table('mechanism',rows)
   for i,x in enumerate(rows):
    mid=classifier_id(x[0]);seg=x[1];v=m['direction_summary'][mid][seg];seen['mechanism'].append((mid,seg))
    for k,key in enumerate(('inference_rows','sign_disagreements_vs_C1','sign_disagreement_rate_row_pooled','old_prior_agreement_count','old_prior_agreement_rate_row_pooled','new_prior_agreement_count','new_prior_agreement_rate_row_pooled')):
     rate='rate' in key;check(x[k+2],v[key],f'mechanism/{i}/{key}',decimals=6 if rate else 0,scale=100 if rate else 1)
  else:raise AssertionError(('unhandled table',h))
 expected_counts={'economics':8,'regime':24,'paired_economics':16,'classification':20,'mapped_mse':8,'norms':4,'mechanism':8}
 assert counts==expected_counts,counts
 newids={mid+'_direction_'+r for mid in MODELS for r in RULES.values()}
 expected_rows={'economics':newids,'regime':{(cid,g) for cid in newids for g in ('bull','bear','sideways')},
  'paired_economics':{(mid+'_direction',ref,r) for mid in MODELS for r in RULES.values() for ref in (mid.removesuffix('_l2unit')+'_direction',mid.rsplit('_',2)[0]+'_half')},
  'classification':{(seg,mid) for seg in SEGMENTS for mid in (*MODELS,*(x.removesuffix('_l2unit') for x in MODELS),'prior_ordinary','prior_magnitude')},
  'mapped_mse':{(mid+'_direction',seg) for mid in MODELS for seg in SEGMENTS},
  'norms':set(MODELS),'mechanism':{(mid,seg) for mid in MODELS for seg in SEGMENTS}}
 for key,values in seen.items():assert len(values)==len(set(values)) and set(values)==expected_rows[key],(key,'omitted/duplicated row')
 performance_cells=len(CHECKS)
 assert performance_cells==468,performance_cells
 loss_keys=('brier','log_loss','weighted_brier','weighted_log_loss')
 claims={
  'all_four_new_overall_four_losses_improve_same_loss_C1_in_I_and_E':all(s['classification_paired']['all'][seg][mid][mid.removesuffix('_l2unit')][key]<0 for mid in MODELS for seg in SEGMENTS for key in loss_keys),
  'all_four_new_E_overall_four_losses_worse_both_priors':all(s['classification_paired']['all']['evaluation'][mid][ref][key]>0 for mid in MODELS for ref in ('prior_ordinary','prior_magnitude') for key in loss_keys),
  'both_ordinary_I_overall_matched_losses_improve_matched_prior':all(s['classification_paired']['all']['interval'][mid]['prior_ordinary'][key]<0 for mid in MODELS if '_ordinary_' in mid for key in ('brier','log_loss')),
  'both_magnitude_I_overall_matched_losses_worse_matched_prior':all(s['classification_paired']['all']['interval'][mid]['prior_magnitude'][key]>0 for mid in MODELS if '_magnitude_' in mid for key in ('weighted_brier','weighted_log_loss')),
  'all_eight_new_overall_Alpha_negative_DD_positive_both_costs':all(s['economics']['all'][cid][cost]['alpha_ex']<0 and s['economics']['all'][cid][cost]['maxdd_delta']>0 for cid in newids for cost in COSTS),
  'all_32_overall_paired_cost_comparisons_worsen_Alpha_and_DD':all(s['paired']['all'][mid+'_direction'][ref]['economics'][rule][cost]['alpha_ex']<0 and s['paired']['all'][mid+'_direction'][ref]['economics'][rule][cost]['maxdd_delta']>0 for mid in MODELS for ref in (mid.removesuffix('_l2unit')+'_direction',mid.rsplit('_',2)[0]+'_half') for rule in RULES.values() for cost in COSTS),
  'new_mapped_E_overall_MSE_worse_half_I_better_half':all(s['paired']['all'][mid+'_direction'][mid.rsplit('_',2)[0]+'_half']['prediction']['evaluation']['mse_difference']>0 and s['paired']['all'][mid+'_direction'][mid.rsplit('_',2)[0]+'_half']['prediction']['interval']['mse_difference']<0 for mid in MODELS),
  'all_32_norms_smaller':all(x['coefficient_l2_norm']<x['old_coefficient_l2_norm'] for x in j['model_checks']),
  'all_E_prior_direction_agreements_increase':all(m['direction_summary'][mid]['evaluation']['new_prior_agreement_count']>m['direction_summary'][mid]['evaluation']['old_prior_agreement_count'] for mid in MODELS),
  'all_160_weighted_score_denominators_positive':len(j['classification_score_checks'])==160 and all(not r['weighted_null'] for r in j['classification_score_checks']),
  'no_old_or_new_zero_logit_on_64_I_E_inference_records':all(r['zero_logit_rows']==r['old_zero_logit_rows']==0 for r in j['direction_diagnostics']),
  'no_new_policy_tiny_DD':not any('_l2unit_' in r['candidate_id'] for r in j['tiny_nonzero_dd_values']),
  'ten_old_tiny_DD':len(j['tiny_nonzero_dd_values'])==10 and {abs(r['maxdd_delta']) for r in j['tiny_nonzero_dd_values']}=={1.1102230246251565e-16},
  'all_old_scores_and_accounts_exact':all(j['numeric_max_absolute_differences'][key]==0 for key in ('unchanged_old_return_score','unchanged_old_classification','unchanged_control_accounts')),
 }
 def flatten_bools(value):
  if isinstance(value,dict):return [b for v in value.values() for b in flatten_bools(v)]
  assert type(value) is bool;return [value]
 flags=flatten_bools(s['direction'])
 claims['all_64_nested_registered_direction_flags_false']=len(flags)==64 and not any(flags)
 for key,value in claims.items():exact(value,True,'claim/'+key)
 assert 'reduces overconfident probability errors' not in text
 assert 'suppressing logistic coefficient magnitude alone' not in text
 assert 'reduces overall probability losses relative to C1' in text
 assert 'this fixed stronger regularization did not recover usable directional forecasts' in text
 assert 'not equal-quarter means' in text and 'A match to the fitted prior' in text
 assert 'repeatedly reused' in text and 'No selection, promotion or deployment occurred' in text
 assert 'No Stage19 procedure is registered by this report' in text
 assert 'Their cause is unknown' in text
 assert 'not annualized' in text and 'not a correct-label count' in text
 assert 'No new strongest model is established' in text
 numeric_claims={
  'C minimum':('0.0005574136008918618',min(v['C'] for v in j['model_checks'])),
  'C maximum':('0.00125',max(v['C'] for v in j['model_checks'])),
  'new gradient infinity':('2.9278447350594705e−8',mj['maximum_gradient_infinity']),
  'new minimum Hessian eigenvalue':('0.23027817223723637',mj['minimum_hessian_eigenvalue']),
  'new scalar logit':('1.1102230246251565e−16',mj['max_absolute_differences']['scalar_logits']),
  'new scalar probability':('1.6653345369377348e−16',mj['max_absolute_differences']['scalar_probabilities']),
  'mean exposure':('2.220446049250313e−16',oj['max_absolute_differences']['account_mean_exposure']),
  'return score':('4e−18',j['numeric_max_absolute_differences']['scalar_return_score']),
  'classification score':('1e−16',j['numeric_max_absolute_differences']['scalar_classification_score']),
  'economic summary':('3.5e−15',j['numeric_max_absolute_differences']['summary_economics']),
  'paired summary':('1.0775e−15',j['numeric_max_absolute_differences']['summary_paired'])}
 for key,(shown,value) in numeric_claims.items():
  assert shown in text,(key,'not in report')
  assert math.isclose(float(shown.replace('−','-')),value,rel_tol=5e-5,abs_tol=0),(key,shown,value)
 assert sorted({v['actual_l2_strength'] for v in j['model_checks']})==[1.0,1.0000000000000002]
 assert 'actual lambda values were 1.0 and 1.0000000000000002' in text
 assert mj['counts']['models']==32 and mj['counts']['fit_model_rows']==45000 and mj['counts']['predict_model_rows']==30112
 assert mj['counts']['old_models_scalar_checked']==32 and mj['counts']['calibration_npz']==mj['counts']['forecasts_npz']==32
 assert oldobj['models']==32 and oldobj['fit_model_rows']==45000 and oldobj['new_coefficients_fitted'] is False
 assert mj['max_absolute_differences']['runtime_scalar_objective']==mj['max_absolute_differences']['runtime_scalar_gradient']==0
 assert oj['counts']['independent_scalar_accounts']==960 and oj['counts']['new_own_state_paths']==oj['counts']['new_full_trace_replays']==64
 assert oj['counts']['new_own_state_decisions']==22016 and oj['counts']['exact_old_controls']==416
 assert all(v==0 for key,v in oj['max_absolute_differences'].items() if key.startswith('trace_') or (key.startswith('account_') and key!='account_mean_exposure'))
 assert src['counts']['total_source_artifacts']==2840
 for segment,n,unscored in [('interval',2537,14),('evaluation',2586,12)]:
  assert all(m['direction_summary'][mid][segment]['inference_rows']==n and m['direction_summary'][mid][segment]['unscored_inference_rows']==unscored for mid in MODELS)
 for marker in ('733 tests OK','57.350 seconds','32 new classifiers','45,000 fit-model rows','30,112 predict-model rows',
  '960 independent scalar cost accounts','64 new own-state paths','22,016 decisions','64 full traces',
  '224 return','160 classification','64 mechanism records','2,840 inherited artifacts','648 new artifacts',
  'All 416 old economic rows','160 old return records','96 old classification records',
  '2,586 inference and 2,574 scored origins','2,537 inference and 2,523 scored origins',
  '2 bull / 4 bear / 2 sideways','190 to 198'):
  assert marker in text,('missing quantitative marker',marker)
 files={'Results file':OUT/'results.json','Registration file':OUT/'registration.json',
  'Preflight file':OUT/'preflight.json','Runtime log':OUT/'run.log',
  'Full test log':Path('/tmp/oracle-regularized-direction-full-tests.log')}
 for label,path in files.items():
  match=re.search(r'(?m)^- '+re.escape(label)+r' SHA256: `([0-9a-f]{64})`',text)
  assert match,label;exact(match.group(1),sha(path),'hash/'+label)
 reg=read(OUT/'registration.json')
 canonical=hashlib.sha256(json.dumps(reg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 match=re.search(r'Registration canonical digest \(distinct from file SHA256\): `([0-9a-f]{64})`',text)
 assert match;exact(match.group(1),canonical,'hash/registration canonical')
 assert reg['source_revision']=='5a82c270c64a342ab7e9df8105b7d23d1336d876' and reg['source_revision'] in text
 assert sha(OUT/'results.json')=='ab2b62a017a70fd65756614f3940c3194d60b8c0ebdab02aa058c988ccbdb678'
 log=(OUT/'run.log').read_text();testlog=files['Full test log'].read_text()
 warning_counts={category:log.count('RuntimeWarning: '+category) for category in ('divide by zero encountered in matmul','overflow encountered in matmul','invalid value encountered in matmul')}
 assert warning_counts==warn['runtime_warning_counts']==mj['runtime_log']['runtime_warning_categories_and_counts']
 assert sum(warning_counts.values())==warn['runtime_warning_total']==384
 assert log.count('ConvergenceWarning:')==0 and log.count('"event": "fold_complete"')==8
 assert re.search(r'Ran 733 tests in 57\.350s',testlog) and re.search(r'(?m)^OK$',testlog)
 local_links=[]
 for target in re.findall(r'\]\(([^)]+)\)',text):
  if target.startswith(('https://','http://')):continue
  p=(REPORT.parent/target.split('#',1)[0]).resolve();assert p.exists(),('dead local link',target)
  local_links.append({'target':target,'resolved':str(p),'sha256':sha(p) if p.is_file() else None})
 pending=[marker for marker in ('AUDIT_SECTION_PENDING','@@AUDIT@@','once complete','placeholder') if marker in text]
 assert not pending,pending
 assert sha(REPORT)==expected_report_sha,'report changed during audit'
 report={'schema':'independent-regularized-direction-formatted-report-audit-v1','passed':not ERRORS,
  'report_sha256':expected_report_sha,'script_sha256':sha(Path(__file__)),
  'bound_input_sha256':{str(p):h for p,h in EXPECTED_HASHES.items()},
  'reviewed_prior_draft_sha256':'dadbc8cdbc574249803b1fd3a589230cbc05db69745053f0bd73c4f9e4c64669',
  'earlier_draft_was_not_final':True,
  'two_narrative_precision_corrections_verified':True,
  'table_rows':counts,'total_performance_rows':sum(counts.values()),'performance_cells':performance_cells,
  'hash_claims':6,'checks':CHECKS,'errors':ERRORS,'quantitative_claims':claims,
  'numerical_audit_claims':{k:{'shown':v[0],'bound_value':v[1]} for k,v in numeric_claims.items()},
  'mechanism_verification':mechanism_verification,'registered_false_flags':len(flags),
  'warning_counts':warning_counts,'runtime_fold_completions':8,'convergence_warnings':0,
  'full_suite_before_fit':{'tests':733,'seconds':57.350},'local_links':local_links,
  'pending_markers':pending,
  'scope':'All 88 formatted performance rows / 468 cells, six hash claims, mechanism denominators and enumerated narrative/audit claims. No new fits, policy rollouts, source/report edits, model selection or independent-confirmation claim. Prior immutable score audit remains unchanged.'}
 OUTPUT.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'passed':report['passed'],'path':str(OUTPUT),'output_sha256':sha(OUTPUT),
  'report_sha256':expected_report_sha,'script_sha256':report['script_sha256'],'table_rows':counts,
  'performance_cells':performance_cells,'hash_claims':6,'errors':ERRORS},sort_keys=True))
 assert not ERRORS,ERRORS


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--execute-report-audit',action='store_true')
 parser.add_argument('--expected-report-sha')
 args=parser.parse_args()
 if not args.execute_report_audit or not args.expected_report_sha or len(args.expected_report_sha)!=64:
  parser.error('Await root final-report notification and explicit report SHA.')
 audit_report(args.expected_report_sha)

if __name__=='__main__':main()
