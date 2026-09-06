from pathlib import Path
import json,hashlib,shutil
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=['codex_outputs/oracle_short_feature_decisions_v1/results.json','codex_outputs/oracle_short_direction_decisions_v1/results.json']
records={};excluded={}
for path in paths:
 d=json.loads(Path(path).read_text());hindsight={r['candidate_id'] for r in d['rows'] if r.get('hindsight_only',False)}
 for cid,row in d['summary']['economics']['all'].items():
  reason=None
  if cid in hindsight:reason='hindsight_diagnostic'
  elif not cid.startswith(('technical_','perp_delay0_')):reason='nonlearned_baseline_or_constant'
  elif any(token in cid for token in ('_prior_','_soft_mapped_prior_','_soft_fit_mean_','_soft_zero_')):reason='constant_probability_or_mean_control'
  if reason:excluded[cid]=reason;continue
  v={'candidate_id':cid,'source':path,'base':row['base'],'stress_2x':row['stress_2x'],
   'quarters':row['quarters'],'joint_quarters_both_costs':row['joint_positive_quarters_both_costs'],
   'regimes':{g:d['summary']['economics'][g][cid] for g in ('bull','bear','sideways')}}
  v['minimum_pass']=all(v[c]['alpha_ex']>0 and v[c]['maxdd_delta']<0 for c in ('base','stress_2x'))
  if cid in records:
   for c in ('base','stress_2x'):
    for k in ('alpha_ex','maxdd_delta','turnover','trades'):assert records[cid][c][k]==v[c][k],(cid,c,k)
  records[cid]=v
rank=lambda v:(-min(v[c]['alpha_ex'] for c in ('base','stress_2x')),max(v[c]['maxdd_delta'] for c in ('base','stress_2x')),max(v[c]['turnover'] for c in ('base','stress_2x')),v['candidate_id'])
eligible=sorted([v for v in records.values() if v['minimum_pass']],key=rank)
assert eligible[0]['candidate_id']=='perp_delay0_reliability_utility_risk1'
out=Path('docs/experiments/btc_accuracy_selection_20260906.json')
result={'schema_version':1,'status':'engineering_compromise_recipe_selected','selected_candidate_id':eligible[0]['candidate_id'],
 'bundle_id':'btc-perp-reliability-20260906','run_id':'btc-perp-reliability-20260906',
 'scope':'Complete causal learned-feature policy union retained by Stage15 and Stage20 on exact development validation5-12; not all historical adaptive trials or a global optimum.',
 'selection_is_post_development_and_not_independent_confirmation':True,'additional_test_used_for_selection':False,
 'screen':'Equal-quarter mean AlphaEX>0 and MaxDDdelta<0 under base and fixed-target doubled-cost replay',
 'ranking':['maximize minimum AlphaEX across costs','minimize maximum DDdelta across costs','minimize maximum turnover across costs','stable candidate ID'],
 'compared_causal_ml_count':len(records),'qualifying_causal_ml_count':len(eligible),'selected':eligible[0],
 'ranked_qualifiers':[v['candidate_id'] for v in eligible], 'candidates':[records[k] for k in sorted(records)],'excluded':excluded,
 'validation_period':{'start':'2021-04-16T13:45:00Z','end_exclusive':'2023-04-16T13:45:00Z','folds':list(range(5,13)),'regime_counts':{'bull':2,'bear':4,'sideways':2}},
 'historical_predictive_and_generalization_gates_passed':False,'high_probability_generalization_established':False,
 'rl':{'qualified':False,'reason':'No retained causal learned actor establishes both minimum signs and deployable compatible provenance; Oracle is not a candidate.','new_algorithm_search_permitted':False},
 'production_refit':{'permitted_fits':3,'reliability_calibrations':1,'calendar_fold':25,'production_cutoff':'2026-07-16T13:45:00Z','evaluation_end':'2026-10-16T13:45:00Z','same_frozen_18_3_3_recipe':True,'new_evaluation_scores_permitted':False,'parameter_or_candidate_changes_permitted':False,'historical_scores_are_not_scores_of_new_weights':True},
 'source_results':{p:sha(p) for p in paths},'candidate_audit_sha256':sha('/tmp/oracle_deployment_candidate_audit_20260906.json'),'build_script_sha256':sha(__file__)}
out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print({'path':str(out),'sha256':sha(out),'compared':len(records),'qualified':len(eligible),'selected':result['selected_candidate_id']})
