"""Read-only Stage17 formatted report audit against the completed independent score audit."""
from pathlib import Path
import hashlib,json,math,re
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
AUDIT=Path('/tmp/oracle_direction_score_audit_20260906.json')
REPORT=ROOT/'docs/experiments/oracle_direction_results_20260906.md'
OUTPUT=Path('/tmp/oracle_direction_report_audit_20260906.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(AUDIT)=='2d3aa83d18e3dc3596d1d52f38267c42997908b0c1faf0eb976f19eae2e8f3ce'
j=json.loads(AUDIT.read_text());s=j['summary'];text=REPORT.read_text()
MODEL=Path('/tmp/oracle_direction_model_audit_20260906.json')
OWN=Path('/tmp/oracle_direction_audit_20260906.json')
assert sha(MODEL)=='f23d8f52577369c08f92f845a448a437fc5b57641bddfaf8e61dfdf37f8afad2'
assert sha(OWN)=='bf7b4556ef80ecbd395cee1d00fbe914f08b30d10fa13d524c34b75806530fb0'
mj=json.loads(MODEL.read_text());oj=json.loads(OWN.read_text())
assert mj['passed'] and oj['status']=='pass'

blocks=[]
for section in text.split('\n\n'):
 lines=[l for l in section.splitlines() if l.startswith('|')]
 if len(lines)>2:blocks.append([[v.strip() for v in l.strip('|').split('|')] for l in lines])
groups={'Technical29':'technical','Perp31':'perp_delay0'}
rules={'hold':'utility_risk1','fallback':'utility_risk1_fallback_bh'}
def mid(label):
 if label.startswith('prior '):return 'prior_'+label.split()[1]
 group,weighting=label.split()[:2]
 return groups[group]+'_'+weighting

def meanid(label):return mid(label)+('_prior_direction' if label.endswith(' prior') else '_direction')
checks=[];errors=[];table_counts={}
def check(observed,expected,location,decimals=6,scale=1):
 value=float(observed);target=float(expected)*scale
 tolerance=0.500001*10**(-decimals)
 good=math.isfinite(value) and abs(value-target)<=tolerance
 checks.append({'location':location,'shown':observed,'expected':target,'passed':good})
 if not good:errors.append(checks[-1])
def exact(observed,expected,location):
 good=observed==expected;checks.append({'location':location,'shown':observed,'expected':expected,'passed':good})
 if not good:errors.append(checks[-1])
for b in blocks:
 h,rows=b[0],b[2:]
 if h[:2]==['Mean','Rule']:
  table_counts['economics']=len(rows)
  for i,x in enumerate(rows):
   cid=meanid(x[0])+'_'+rules[x[1]];v=s['economics']['all'][cid]
   for k,(c,m) in enumerate([('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta')]):check(x[k+2],v[c][m],f'econ/{i}/{m}/{c}',scale=100)
   exact(x[6],str(v['joint_positive_quarters_both_costs'])+'/8',f'econ/{i}/joint')
   exact(x[7],str(s['direction'][cid]['economic_means_all_strata_both_costs']).lower(),f'econ/{i}/allstrata')
 elif h[:3]==['Classifier','Regime','Rule']:
  table_counts['regime']=len(rows)
  for i,x in enumerate(rows):
   v=s['economics'][x[1]][meanid(x[0])+'_'+rules[x[2]]]
   for k,(c,m) in enumerate([('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta')]):check(x[k+3],v[c][m],f'regime/{i}/{m}/{c}',scale=100)
   exact(x[7],f"{v['joint_positive_quarters_both_costs']}/{v['quarters']}",f'regime/{i}/joint')
 elif h[:2]==['Classifier','Rule']:
  table_counts['paired_economic']=len(rows)
  for i,x in enumerate(rows):
   m=meanid(x[0]);parent=groups[x[0].split()[0]]+'_half';v=s['paired']['all'][m][parent]['economics'][rules[x[1]]]
   for k,(c,key) in enumerate([('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta')]):check(x[k+2],v[c][key],f'paired/{i}/{key}/{c}',scale=100)
 elif h[:2]==['Segment','Classifier']:
  table_counts['classification']=len(rows)
  for i,x in enumerate(rows):
   v=s['classification']['all'][x[0]][mid(x[1])]
   for k,key in enumerate(['brier','log_loss','binary_accuracy','weighted_brier','weighted_log_loss','weighted_binary_accuracy']):check(x[k+2],v[key],f'classification/{i}/{key}',scale=100 if 'accuracy' in key else 1)
 elif h[:2]==['Classifier','Segment']:
  table_counts['mapped_mse']=len(rows)
  for i,x in enumerate(rows):
   m=meanid(x[0]);g=groups[x[0].split()[0]];w=x[0].split()[1];v=s['prediction']['all'][x[1]][m]
   own=s['prediction']['all'][x[1]][g+'_half']['return_mse'];prior=s['prediction']['all'][x[1]][g+'_'+w+'_prior_direction']['return_mse']
   expected=[v['return_mse']*1e6,v['return_mae']*1e3,(v['return_mse']/own-1)*100,(v['return_mse']/prior-1)*100,v['zero_return_mse']*1e6,v['fit_mean_return_mse']*1e6]
   for k,val in enumerate(expected):check(x[k+2],val,f'mapped_mse/{i}/{k}')
 elif h[:2]==['Fold','Max magnitude weight']:
  table_counts['weight_concentration']=len(rows)
  for i,x in enumerate(rows):
   f=int(x[0]);v=next(y for y in j['fit_checks'] if y['fold']==f)['weights']['magnitude']
   check(x[1],v['maximum_weight'],f'weights/{f}/maximum')
   check(x[2],v['zero_weight_rows'],f'weights/{f}/zeros',decimals=0)
   check(x[3],v['weight_concentration_effective_rows'],f'weights/{f}/concentration')
   modelv=next(v for v in mj['fit_weight_diagnostics'] if v['fold']==f and v['weighting']=='magnitude')
   assert modelv['maximum']==v['maximum_weight'] and modelv['zero_rows']==v['zero_weight_rows']
   assert abs(modelv['squared_weight_effective_count']-v['weight_concentration_effective_rows'])<1e-10
 elif h[:2]==['Fold','Fit rows']:
  table_counts['priors']=len(rows)
  for i,x in enumerate(rows):
   f=int(x[0]);v=next(y for y in j['fit_checks'] if y['fold']==f)
   for k,val in enumerate([v['fit_rows'],*v['class_counts']]):check(x[k+1],val,f'priors/{f}/{k}',decimals=0)
   check(x[4],v['weights']['ordinary']['prior'],f'priors/{f}/ordinary',decimals=9)
   check(x[5],v['weights']['magnitude']['prior'],f'priors/{f}/magnitude',decimals=9)
   for k,model in enumerate(['technical_ordinary','technical_magnitude','perp_delay0_ordinary','perp_delay0_magnitude']):
    niter=next(v['n_iter'] for v in j['model_checks'] if v['fold']==f and v['model_id']==model)
    check(x[6].split('/')[k],niter,f'priors/{f}/iterations/{model}',decimals=0)
expected_counts={'economics':16,'regime':24,'paired_economic':8,'classification':12,'mapped_mse':8,'priors':8,'weight_concentration':8}
assert table_counts==expected_counts,table_counts
models=['technical_ordinary','technical_magnitude','perp_delay0_ordinary','perp_delay0_magnitude']
claims={
 'all_four_probability_losses_worse_both_priors_both_segments':all(s['classification_paired']['all'][seg][m][ref][k]>0 for seg in ['interval','evaluation'] for m in models for ref in ['prior_ordinary','prior_magnitude'] for k in ['brier','log_loss','weighted_brier','weighted_log_loss']),
 'ordinary_probability_losses_lower_than_same_group_magnitude_both_segments':all(s['classification_paired']['all'][seg][g+'_ordinary'][g+'_magnitude'][k]<0 for seg in ['interval','evaluation'] for g in groups.values() for k in ['brier','log_loss','weighted_brier','weighted_log_loss']),
 'all_eight_learned_economic_policies_worse_own_parents_both_costs':all(s['paired']['all'][m+'_direction'][m.rsplit('_',1)[0]+'_half']['economics'][r][c]['alpha_ex']<0 and s['paired']['all'][m+'_direction'][m.rsplit('_',1)[0]+'_half']['economics'][r][c]['maxdd_delta']>0 for m in models for r in rules.values() for c in ['base','stress_2x']),
 'perp_four_observed_allstrata_sign_gates_and_five_joint_quarters':all(s['direction'][m+'_direction_'+r]['economic_means_all_strata_both_costs'] and s['economics']['all'][m+'_direction_'+r]['joint_positive_quarters_both_costs']==5 for m in models if m.startswith('perp') for r in rules.values()),
 'all_probability_and_mapped_strict_predictive_gates_fail_both_segments':all(not s['direction'][m+'_direction_utility_risk1'][key][seg] for m in models for seg in ['interval','evaluation'] for key in ['matched_probability_losses_improved_all_strata','mapped_mse_vs_zero_fitmean_parent_and_matched_prior_all_strata']),
 'all_classifier_score_denominators_positive':all(not r['weighted_null'] for r in j['classification_score_checks']),
 'no_new_policy_tiny_dd':not any('_direction_' in v['candidate_id'] for v in j['tiny_nonzero_dd_values']),
 'all_priors_same_direction_per_fold':all((v['weights']['ordinary']['prior']>.5)==(v['weights']['magnitude']['prior']>.5)==(v['fold']<12) for v in j['fit_checks']),
 'original_score_and_account_identity':j['numeric_max_absolute_differences']['unchanged_parent_score']==0 and j['numeric_max_absolute_differences']['unchanged_control_accounts']==0,
}
for k,v in claims.items():
 if not v:errors.append({'claim':k,'passed':False})
# Direct hash table/log checks have a separate scope from substantive model output.
paths={'preflight.json':ROOT/'codex_outputs/oracle_direction_decisions_v1/preflight.json',
 'registration.json':ROOT/'codex_outputs/oracle_direction_decisions_v1/registration.json',
 'results.json':ROOT/'codex_outputs/oracle_direction_decisions_v1/results.json',
 'run.log':ROOT/'codex_outputs/oracle_direction_decisions_v1/run.log',
 'config YAML':ROOT/'configs/oracle_direction_decisions_20260906.yaml'}
for b in blocks:
 if b[0][:1]==['Item']:
  for x in b[2:]:
   if x[0] in paths:exact(x[1].strip('`'),sha(paths[x[0]]),'hash/'+x[0])
log=paths['run.log'].read_text()
log_checks={'runtime_warning_records':log.count('RuntimeWarning:'),'convergence_warning_records':log.count('ConvergenceWarning:'),'fold_complete_records':log.count('"event": "fold_complete"')}
assert log_checks=={'runtime_warning_records':384,'convergence_warning_records':0,'fold_complete_records':8},log_checks
# Added audit-evidence and tie claims are checked against the auditors' bound values.
assert 'learned perp31 direction replacements' in text.lower()
assert 'the learned replacements have higher controller turnover in this comparison' in text
assert all(v['zero_logit_rows']==0 and v['zero_parent_mean_rows']==0 for v in j['mapping_checks'])
for segment in ['interval','evaluation']:
 rows=[r for r in j['classification_score_checks'] if r['segment']==segment and r['classifier_id']=='technical_ordinary']
 assert sum(r['zero_actual_rows'] for r in rows)==1
assert len(j['tiny_nonzero_dd_values'])==10
assert {abs(r['maxdd_delta']) for r in j['tiny_nonzero_dd_values']}=={1.1102230246251565e-16}
assert not any('_direction_' in r['candidate_id'] for r in j['tiny_nonzero_dd_values'])
assert mj['counts']['models']==32 and mj['counts']['fit_model_rows']==45000
assert mj['counts']['predict_model_rows']==30112 and mj['counts']['fit_prior_estimates']==16
assert oj['counts']['independent_scalar_accounts']==832 and oj['counts']['new_own_state_paths']==128
assert oj['counts']['new_own_state_decisions']==44032 and oj['counts']['exact_old_controls']==288
for marker in ['32 fitted models, 45,000 fit-model rows, 30,112 predict-model rows and 16 priors',
               'all 832 cost accounts and 128 new paths / 44,032 decisions',
               'All 288 old controls / 576 accounts were unchanged',
               'all 160 return scores, 96 classifier scores',
               '720 new artifact bindings and 2,120 ancestral bindings',
               'zero zero-logit and zero parent-magnitude observations',
               'Each of I and E has one scored zero-return observation',
               'Ten old B&H/scale-mean cost rows',
               'None belongs to the new policies']:
 assert marker in text,marker
numbers={
 'runtime gradient':('1.4416003547403967e−7',mj['maximum_gradient_infinity']),
 'objective difference':('0',mj['max_absolute_differences']['runtime_scalar_objective']),
 'gradient difference':('1.0842e−19',mj['max_absolute_differences']['runtime_scalar_gradient']),
 'minimum Hessian eigenvalue':('0.0005574136008918624',mj['minimum_hessian_eigenvalue']),
 'saved logits':('1.3323e−15',mj['max_absolute_differences']['scalar_logits']),
 'saved probability':('2.2204e−16',mj['max_absolute_differences']['scalar_probabilities']),
 'mean exposure':('2.220446049250313e−16',oj['max_absolute_differences']['account_mean_exposure']),
 'summary return scores':('4e−18',j['numeric_max_absolute_differences']['scalar_return_score']),
 'summary classifier scores':('1e−16',j['numeric_max_absolute_differences']['scalar_classification_score']),
 'summary economics':('3.5e−15',j['numeric_max_absolute_differences']['summary_economics']),
 'summary paired contrasts':('1.2775e−15',j['numeric_max_absolute_differences']['summary_paired']),
}
for label,(shown,value) in numbers.items():
 assert shown in text,(label,shown)
 assert math.isclose(float(shown.replace('−','-')),value,rel_tol=5e-5,abs_tol=0),(label,shown,value)
assert all(v==0 for k,v in oj['max_absolute_differences'].items() if
 k.startswith('trace_') or k in ['account_alpha_ex','account_maxdd_delta'])
for v in mj['fit_weight_diagnostics']:
 if v['weighting']=='ordinary':assert v['mean']==1 and v['maximum']==1
 else:assert v['mean']==(1.0000000000000002 if v['fold']==5 else 1.)
assert 'Magnitude mean weight is exactly 1 except fold 5 at 1.0000000000000002' in text
report={'schema':'independent-direction-formatted-report-audit-v1','passed':not errors,
 'report_sha256':sha(REPORT),'score_audit_sha256':sha(AUDIT),'script_sha256':sha(Path(__file__)),
 'model_audit_sha256':sha(MODEL),'own_account_audit_sha256':sha(OWN),
 'earlier_draft_audit_sha256':'4b433fe413a49f120c406583ef6475848d62318e9fa974062d01bf63059f5db0',
 'earlier_draft_was_not_final':True,
 'audit_numerical_claims':{k:{'shown':v[0],'bound_value':v[1]} for k,v in numbers.items()},
 'zero_logit_and_parent_mean_claims_verified':True,'one_zero_return_per_segment_verified':True,
 'ten_old_tiny_dd_and_no_new_tiny_dd_verified':True,
 'table_rows':table_counts,'performance_table_cells':len(checks)-len(paths),'hash_table_cells':len(paths),
 'checks':checks,'quantitative_claims':claims,'run_log_checks':log_checks,'errors':errors,
 'pending_report_markers':[w for w in ['AUDIT_SECTION_PENDING','once complete','@@AUDIT@@'] if w in text],
 'scope':'Formatted table cells and enumerated narrative claims only; existing immutable summary audit retained unchanged. No new fits, policies or source edits.'}
OUTPUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','scope','quantitative_claims']},sort_keys=True))
print('output_sha256',sha(OUTPUT))
assert not errors,errors
