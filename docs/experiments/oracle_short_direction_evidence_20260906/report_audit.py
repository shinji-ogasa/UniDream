"""Stage20 report cells and claims from sealed summary audit; no score recalculation."""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
REPORT=ROOT/'docs/experiments/oracle_short_direction_results_20260906.md'
OUT=ROOT/'codex_outputs/oracle_short_direction_decisions_v1'
AUDIT=Path('/tmp/oracle_short_direction_score_audit_20260906.json')
OUTPUT=Path('/tmp/oracle_short_direction_report_audit_20260906.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert sha(AUDIT)=='db05e1aa0d9e8b0dbebe3f8b895975c341d7d685422452e4726ebf83cee84f41'
a=read(AUDIT);s=a['summary'];result=read(OUT/'results.json');reg=read(OUT/'registration.json');pre=read(OUT/'preflight.json')
assert a['passed'] and sha(OUT/'results.json')=='e7c116ea5aae663ec42276d96c627813aad922b9f8796bdded8c9bb045ff9cd9'
text=REPORT.read_text();initial=sha(REPORT)
assert not re.search(r'PENDING|PLACEHOLDER|TODO|@@AUDIT@@',text)
M='technical_short_both_magnitude_soft';RULES=('utility_risk1','utility_risk1_fallback_bh');COSTS=('base','stress_2x');STRATA=('all','bull','bear','sideways');SEGMENTS=('interval','evaluation')
REFS=('technical_magnitude_soft','technical_magnitude_direction','technical_half','technical_soft_mapped_prior','technical_soft_fit_mean','technical_soft_zero')
ri={(r['fold'],r['candidate_id']):r for r in result['rows']}
checks=[];counts={};seen={};blocks=[];current=[]
for line in text.splitlines()+['']:
 if line.startswith('|'):current.append([x.strip() for x in line.strip().strip('|').split('|')])
 elif current:blocks.append(current);current=[]
def check(got,want,where,kind='performance'):
 assert got==want,(where,got,want)
 checks.append({'location':where,'value':got,'kind':kind})
def num(got,want,where,scale=1):check(got,f'{float(want)*scale:.6f}',where)
def econ(cells,v,where):
 for x,(co,k) in zip(cells,[(c,k) for c in COSTS for k in ('alpha_ex','maxdd_delta')]):num(x,v[co][k],where+'/'+co+'/'+k,100)
for block in blocks:
 h,rows=block[0],block[2:]
 assert all(set(c)<=set('-: ') for c in block[1])
 keys=[]
 if h[:3]==['Stratum','Rule','Quarters']:
  name='economic_strata'
  for i,x in enumerate(rows):
   g,rule=x[:2];v=s['economics'][g][M+'_'+rule];keys.append((g,rule))
   check(x[2],str(v['quarters']),f'{name}/{i}/quarters');econ(x[3:7],v,f'{name}/{i}');check(x[7],str(v['joint_positive_quarters_both_costs']),f'{name}/{i}/joint')
 elif h[:3]==['Fold','Trend','Rule']:
  name='economic_folds'
  for i,x in enumerate(rows):
   f=int(x[0]);rule=x[2];v=ri[f,M+'_'+rule];keys.append((f,rule))
   check(x[0],str(f),f'{name}/{i}/fold','identifier');check(x[1],v['regime']['trend'],f'{name}/{i}/trend','identifier');econ(x[3:7],v,f'{name}/{i}')
 elif h[:3]==['Stratum','Segment','Task']:
  name='classification_paired'
  for i,x in enumerate(rows):
   g,seg,w=x[:3];mid='technical_short_both_'+w;keys.append((g,seg,w));kk=('brier','log_loss') if w=='ordinary' else ('weighted_brier','weighted_log_loss')
   vals=[s['classification_paired'][g][seg][mid][r][k] for r in ('technical_'+w,'prior_'+w) for k in kk]
   for j,(xx,v) in enumerate(zip(x[3:],vals)):num(xx,v,f'{name}/{i}/{j}')
 elif h[:3]==['Stratum','Segment',M]:
  name='mapped_mse';assert tuple(h[2:])==(M,)+REFS
  for i,x in enumerate(rows):
   g,seg=x[:2];keys.append((g,seg))
   for j,mu in enumerate((M,)+REFS):num(x[j+2],s['prediction'][g][seg][mu]['return_mse'],f'{name}/{i}/{mu}',1e6)
 elif h[0]=='Task':
  name='proper_gates'
  for i,x in enumerate(rows):
   mid=x[0];keys.append(mid)
   for j,seg in enumerate(SEGMENTS):check(x[j+1],str(s['probability_gates'][mid][seg]),f'{name}/{i}/{seg}')
 elif h[:2]==['Rule','Absolute economics']:
  name='policy_gates'
  for i,x in enumerate(rows):
   rule=x[0];keys.append(rule);v=s['short_direction'][M+'_'+rule]
   vals=[v['economic_means_all_strata_both_costs'],v['economic_improvement_vs_all_six_references_all_strata_both_costs'],
    *[v['mapped_mse_vs_all_six_references_improved_all_strata'][seg] for seg in SEGMENTS],
    *[v['magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata'][seg] for seg in SEGMENTS],v['high_probability_generalization_established']]
   for j,v in enumerate(vals):check(x[j+1],str(v),f'{name}/{i}/{j}')
 elif h==['Binding','SHA256']:
  name='hashes'
  files={'config':ROOT/'configs/oracle_short_direction_decisions_20260906.yaml','protocol':ROOT/'docs/experiments/oracle_short_direction_registration_20260906.md',
   'research':ROOT/'docs/experiments/oracle_short_direction_research_20260906.md',**{k:OUT/(k+'.json') for k in ('registration','preflight','results')}}
  for x in rows:keys.append(x[0]);check(x[1],sha(files[x[0]]),'hash/'+x[0],'hash')
 else:raise AssertionError(h)
 assert name not in counts and len(keys)==len(set(keys));counts[name]=len(rows);seen[name]=set(keys)
expected={'economic_strata':8,'economic_folds':16,'classification_paired':16,'mapped_mse':8,'proper_gates':2,'policy_gates':2,'hashes':6}
assert counts==expected
assert seen['economic_strata']=={(g,r) for g in STRATA for r in RULES}
assert seen['economic_folds']=={(f,r) for f in range(5,13) for r in RULES}
assert seen['classification_paired']=={(g,seg,w) for g in STRATA for seg in SEGMENTS for w in ('ordinary','magnitude')}
assert seen['mapped_mse']=={(g,seg) for g in STRATA for seg in SEGMENTS}
assert seen['proper_gates']==set(s['probability_gates']) and seen['policy_gates']==set(RULES)
def flat(v):
 if isinstance(v,dict):return [x for v0 in v.values() for x in flat(v0)]
 assert type(v) is bool
 return [v]
hold=s['economics']['all'][M+'_'+RULES[0]];fallback=s['economics']['all'][M+'_'+RULES[1]]
claims={
 'overall_hold_joint_signs_both_costs':all(hold[c]['alpha_ex']>0 and hold[c]['maxdd_delta']<0 for c in COSTS),
 'overall_fallback_fails_both_costs':all(not(fallback[c]['alpha_ex']>0 and fallback[c]['maxdd_delta']<0) for c in COSTS),
 'both_new_policies_joint_2_of_8':all(s['economics']['all'][M+'_'+r]['joint_positive_quarters_both_costs']==2 for r in RULES),
 'all_four_proper_and_family_gates_false':not any(flat(s['probability_gates'])) and s['both_classifier_families_improve_matched_losses_all_strata_both_segments'] is False,
 'all16_nested_policy_flags_false':len(flat(s['short_direction']))==16 and not any(flat(s['short_direction'])),
 '16_new_fits_zero_uniquepriors_zero_riskfits':result['new_model_fits']==16 and result['new_unique_priors']==result['risk_model_or_calibration_fits']==0,
 'old80policies24means10classifiers_preserved':len(reg['config']['control_ids'])==80 and len({x['mean_id'] for x in result['scores']})==25 and len({x['classifier_id'] for x in result['classification_scores']})==12,
 '656rows1312accounts400returns192classes32direction16mapping95perfold':a['inventory']['economic_rows']==656 and a['inventory']['accounts']==1312 and a['inventory']['return_scores']==400 and a['inventory']['classification_scores']==192 and len(result['direction_diagnostics'])==32 and len(result['mapping_diagnostics'])==16 and a['inventory']['new_artifacts']==95*8,
 'old640rows384returns160class_exact':a['inventory']['unchanged_old_economic_rows']==640 and a['inventory']['unchanged_old_return_scores']==384 and a['inventory']['unchanged_old_classification_scores']==160 and all(a['numeric_max_absolute_differences'][k]==0 for k in ('exact_old_row','exact_old_return','exact_old_classification')),
 'complete_stage19_summary_preserved':result['summary']['inherited_Stage19_summary']==read(ROOT/'codex_outputs/oracle_soft_direction_decisions_v1/results.json')['summary'],
 'ledger220':result['total_adaptively_explored_causal_names']==220,
 '2bull4bear2side_and_retrospectiveI':s['regime_counts']=={'bull':2,'bear':4,'sideways':2} and s['interval_regime_strata_are_retrospective_evaluation_groupings'] is True,
 'inference_score_supports':all(sum(x[k] for x in a['new_support_checks'] if x['segment']==seg)==v for seg,k,v in [('evaluation','infer',2586),('evaluation','score',2574),('interval','infer',2537),('interval','score',2523)]),
 'registered332fallback2missingopens':reg['config']['fallback_rows']==332 and reg['config']['missing_current_open_rows']==2,
 'freeze_revision':reg['source_revision']=='69d2cd6bae732d9598135c20fe216a4fa9b48fa1' and reg['source_revision'] in text,
 'new_fit_feature37and2tasks':reg['config']['group_dimension']==37 and reg['config']['weightings']==['ordinary','magnitude'],
 'ordinary_notmapped_additionals_notused_no_selection':reg['config']['ordinary_probability_mapped_to_returns'] is False and not result['additional_test_used_for_modeling_or_scoring'] and not result['selection_performed'],
 'no_generalization_receiptclaim':not result['high_probability_generalization_established'] and not pre['historical_receipt_provenance_established'],
 'registered_execution_same_target_coststress':reg['config']['execution']=={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01},
 'no_pending_placeholder':not re.search(r'PENDING|PLACEHOLDER|TODO|@@AUDIT@@',text),
}
assert all(claims.values()),[k for k,v in claims.items() if not v]
# Explicit scalar highlights are present, including Unicode-minus rendering.
for phrase in ('+0.620736pt','−0.264463pt','+0.025604pt / −0.144629pt','only2/8','results_observed=false'):assert phrase in text
assert sha(REPORT)==initial
assert not OUTPUT.exists()
result={'schema':'independent-short-direction-report-audit-v1','passed':True,'report_sha256':initial,
 'audit_script_sha256':sha(__file__),'score_audit_sha256':sha(AUDIT),'results_sha256':sha(OUT/'results.json'),
 'table_row_counts':counts,'performance_rows':52,'performance_cells':sum(x['kind']=='performance' for x in checks),
 'identifier_cells':sum(x['kind']=='identifier' for x in checks),'hash_cells':6,'checks':checks,'narrative_claims':claims,
 'limitations':'Report-only audit; no aggregate rerun, fit, score function, predictor or policy/account replay. Independent scalar/Decimal audit is sealed. Source feature parity/own-state execution proofs remain delegated evidence, not newly replayed here.',
 'material_corrections_required':[]}
OUTPUT.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks','narrative_claims')},sort_keys=True));print('AUDIT_JSON_SHA',sha(OUTPUT))
