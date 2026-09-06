"""Independent Stage19 final formatted-report audit; saved sealed evidence only.

Does not refit, predict, score forecasts, or replay policies. The separately
sealed scalar score/Decimal60 summary and mapping/account audits supply evidence.
An earlier draft was reviewed for schema only; only the requested final SHA is
eligible for this report-verification output.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
REPORT=ROOT/'docs/experiments/oracle_soft_direction_results_20260906.md'
OUT=ROOT/'codex_outputs/oracle_soft_direction_decisions_v1'
OUTPUT=Path('/tmp/oracle_soft_direction_report_audit_20260906.json')
P=Path('/tmp')
SCORE=P/'oracle_soft_direction_score_audit_20260906.json'
MAPPING=P/'oracle_soft_direction_mapping_audit_20260906.json'
MECHANISM=P/'oracle_soft_direction_mechanism_summary_20260906.json'
OWN=P/'oracle_soft_direction_audit_20260906.json'
CONSTANT=P/'oracle_soft_direction_constant_behavior_audit_20260906.json'
SOURCE=P/'oracle_soft_direction_source_audit_20260906.json'
TESTLOG=ROOT/'docs/experiments/oracle_soft_direction_evidence_20260906/full-tests.log'
EXPECTED={
 TESTLOG:'9301f9e587a0fbcf1b61759c23dd9656dfc31b59d1ba4ebaa1f80c9effde766a',
 SCORE:'7988dd7eb6bb6cfe40b9d5cc4fad8a3119737817dc16560ac2d646fb61db16f3',
 MAPPING:'d5bc7358bb75f4a8f278b590355bff46955320d0b51a5b3738c37e33bbb40eef',
 MECHANISM:'c55511918bafb9cdc95a9e1595594d101372fcd7811f00e628b640a2367ee3d1',
 OWN:'e9305ef2460f8f67e23497bb195b0945b51260e47f88b5d2dacfc453ad512912',
 CONSTANT:'a30acd2a922dbfda98745cb16561111a02dfcfb790fe93ee66a269ae2454ccbe',
 SOURCE:'4965de3c0ba40741fb9bff7d588a2e62ec19fbd6b547d013a032a99bd3c2b96b',
 OUT/'results.json':'d85e14e1a8249601a1f28c0d4fa29b1fbb23a3571232dcedc5edee0644b82cc5',
 OUT/'registration.json':'dda43125f1f670fd84ae35809148561863018e8de5ea4c18b99a14fe454e2f22',
 OUT/'preflight.json':'be95da2abe1581b516cc8a0fd168ae919ad89527280818b283f46b92c95f2154',
 OUT/'run.log':'e9a42d0cff32a5f7de65aec59fbcbb9764d9730cd50fc15e8ab8c615d5733644',
}
LABELS={'Technical29 C1':'technical_magnitude_soft','Technical29 L2unit':'technical_magnitude_l2unit_soft',
 'Perp31 C1':'perp_delay0_magnitude_soft','Perp31 L2unit':'perp_delay0_magnitude_l2unit_soft'}
RULES={'hold':'utility_risk1','fallback':'utility_risk1_fallback_bh'}
SEGMENTS=('interval','evaluation'); STRATA=('all','bull','bear','sideways'); COSTS=('base','stress_2x')
KINDS=('mapped_prior','fit_mean','zero'); GROUPS=('technical','perp_delay0'); FOLDS=tuple(range(5,13))
CELL_CHECKS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dec(x):return Decimal(str(x))
def avg(xs):return sum(map(dec,xs))/Decimal(len(xs))
def group(mu):return 'technical' if mu.startswith('technical_') else 'perp_delay0'
def refs(mu):return (mu.removesuffix('_soft')+'_direction',group(mu)+'_half',*(group(mu)+'_soft_'+k for k in KINDS))
def tables(text):
 blocks=[];current=[]
 for line in text.splitlines()+['']:
  if line.startswith('|'):current.append([x.strip() for x in line.strip().strip('|').split('|')])
  elif current:blocks.append(current);current=[]
 return blocks

def cell(actual,value,loc,*,scale=1,precision=6,scientific=False):
 expected=format(float(value)*scale, f'.{precision}e' if scientific else f'.{precision}f')
 assert actual==expected,(loc,actual,expected)
 CELL_CHECKS.append({'location':loc,'reported':actual,'independent_expected':expected})

def exact(actual,expected,loc):
 assert actual==expected,(loc,actual,expected)
 CELL_CHECKS.append({'location':loc,'reported':actual,'independent_expected':expected})

def check_econ(cells,v,prefix):
 for i,(c,k) in enumerate((('base','alpha_ex'),('base','maxdd_delta'),('stress_2x','alpha_ex'),('stress_2x','maxdd_delta'))):
  tiny=0<abs(v[c][k]*100)<1e-6 and 'e' in cells[i]
  cell(cells[i],v[c][k],prefix+'/'+c+'/'+k,scale=100,precision=3 if tiny else 6,scientific=tiny)

def verify_mechanism(m,ma):
 exact(m['source_mapping_audit_sha256'],EXPECTED[MAPPING],'mechanism/source_mapping')
 exact(m['source_results_sha256'],EXPECTED[OUT/'results.json'],'mechanism/source_result')
 rr=ma['mapping_diagnostics'];assert len(rr)==64
 assert m['constant_identity_by_fold']==ma['mapping_scalars']
 for mu in LABELS.values():
  for seg in SEGMENTS:
   rows=[x for x in rr if x['mean_id']==mu and x['segment']==seg]
   assert {x['fold'] for x in rows}==set(FOLDS) and len(rows)==8
   a=m['mechanism_summary'][mu][seg]
   assert a['inference_rows']==sum(x['rows'] for x in rows)
   for k in ('mapped_direction_vs_logit_disagreements','mapped_zero_mean_rows','new_abs_mu_equal_to_hard_rows','new_abs_mu_greater_than_hard_rows','probability_direction_vs_logit_disagreements','probability_half_rows','probability_one_rows','probability_zero_rows','source_zero_logit_rows','unscored_inference_rows'):
    assert a[k]==sum(x[k] for x in rows),(mu,seg,k)
   for k in ('mean_abs_new_mu','mean_abs_hard_mu','mean_abs_parent_mu'):
    assert math.isclose(float(avg([x[k] for x in rows])),a[k+'_equal_quarter'],rel_tol=1e-13,abs_tol=1e-15),(mu,seg,k)
 for c in m['fixed_constant_target_comparisons']:
  assert c['targets_exact'] and c['same_base_stress_metrics'] and c['different_target_rows']==0
 assert len(m['fixed_constant_target_comparisons'])==80
 assert m['fixed_constant_saved_metrics_comparisons_exact']==m['fixed_constant_target_comparisons_exact']==80
 return {'mean_segment_aggregates':8,'underlying_audited_mechanism_records':64,'constant_scalars':8,'saved_target_pairs':80}

def main(expected_report_sha):
 assert sha(REPORT)==expected_report_sha
 assert not OUTPUT.exists(),'Do not overwrite a prior report audit.'
 for p,h in EXPECTED.items():assert sha(p)==h,(p,'sealed evidence changed')
 a=read(SCORE);s=a['summary'];m=read(MECHANISM);ma=read(MAPPING);o=read(OWN);co=read(CONSTANT);src=read(SOURCE)
 assert a['passed'] and ma['passed'] and src['passed'] and o['status']=='pass'
 text=REPORT.read_text(); assert not re.search('PENDING|PLACEHOLDER|TODO|@@AUDIT@@',text)
 with localcontext() as ctx:
  ctx.prec=60; mechanism_check=verify_mechanism(m,ma)
 seen={};counts={}
 for block in tables(text):
  h,rows=block[0],block[2:]; assert all(set(x)<=set('-: ') for x in block[1])
  if h[:2]==['Source','Rule']:
   name='learned_economics';keys=[]
   for i,x in enumerate(rows):
    mu=LABELS[x[0]];cid=mu+'_'+RULES[x[1]];v=s['economics']['all'][cid];keys.append(cid)
    check_econ(x[2:6],v,f'{name}/{i}');exact(x[6],f"{v['joint_positive_quarters_both_costs']}/8",f'{name}/{i}/joint')
  elif h[:3]==['Group','Control','Rule']:
   name='constant_economics';keys=[]
   for i,x in enumerate(rows):
    assert x[0] in GROUPS and x[1] in KINDS
    cid=x[0]+'_soft_'+x[1]+'_'+RULES[x[2]];v=s['economics']['all'][cid];keys.append(cid)
    check_econ(x[3:7],v,f'{name}/{i}');exact(x[7],f"{v['joint_positive_quarters_both_costs']}/8",f'{name}/{i}/joint')
  elif h[:3]==['Source','Regime','Rule']:
   name='regime_economics';keys=[]
   for i,x in enumerate(rows):
    cid=LABELS[x[0]]+'_'+RULES[x[2]];v=s['economics'][x[1]][cid];keys.append((cid,x[1]))
    check_econ(x[3:7],v,f'{name}/{i}');exact(x[7],f"{v['joint_positive_quarters_both_costs']}/{v['quarters']}",f'{name}/{i}/joint')
  elif h[:3]==['Source','Reference','Rule']:
   name='paired_economics';keys=[]
   for i,x in enumerate(rows):
    mu=LABELS[x[0]];assert x[1] in ('same-classifier hard','own original half')
    ref=refs(mu)[0 if x[1]=='same-classifier hard' else 1];rule=RULES[x[2]];keys.append((mu,ref,rule))
    check_econ(x[3:7],s['paired']['all'][mu][ref]['economics'][rule],f'{name}/{i}')
  elif h[:3]==['Source','Segment','MSE ×10^6']:
   name='mapped_mse';keys=[]
   for i,x in enumerate(rows):
    mu=LABELS[x[0]];seg=x[1];keys.append((mu,seg));v=s['prediction']['all'][seg][mu]
    values=[v['return_mse'],*[s['paired']['all'][mu][ref]['prediction'][seg]['mse_difference'] for ref in refs(mu)]]
    for k,v in enumerate(values):cell(x[k+2],v,f'{name}/{i}/{k}',scale=1e6)
  elif h[:3]==['Source','Segment','Inference rows']:
   name='mechanism';keys=[]
   for i,x in enumerate(rows):
    mu=LABELS[x[0]];seg=x[1];v=m['mechanism_summary'][mu][seg];keys.append((mu,seg))
    cell(x[2],v['inference_rows'],f'{name}/{i}/n',precision=0)
    for k,f in enumerate(('mean_abs_new_mu_equal_quarter','mean_abs_hard_mu_equal_quarter','mean_abs_parent_mu_equal_quarter')):cell(x[k+3],v[f],f'{name}/{i}/{f}',scale=1000)
    cell(x[6],v['new_abs_mu_greater_than_hard_rows'],f'{name}/{i}/greater',precision=0)
  elif h[:2]==['Fold','T mean abs ×10^3']:
   name='T_scalars';keys=[]
   for i,x in enumerate(rows):
    f=int(x[0]);v=next(r for r in m['constant_identity_by_fold'] if r['fold']==f);keys.append(f)
    cell(x[0],f,f'{name}/{i}/fold',precision=0)
    for k,field in enumerate(('fit_abs_return_mean','fit_return_mean')):cell(x[k+1],v[field],f'{name}/{i}/{field}',scale=1000)
    for k,field in enumerate(('saved_prior_minus_statistical_prior','mapped_prior_minus_fit_mean')):cell(x[k+3],v[field],f'{name}/{i}/{field}',scientific=True,precision=3)
  else:raise AssertionError(('Unexpected table',h))
  assert name not in counts;counts[name]=len(rows);seen[name]=keys
 expected_counts={'learned_economics':8,'constant_economics':12,'regime_economics':24,'paired_economics':16,'mapped_mse':8,'mechanism':8,'T_scalars':8}
 assert counts==expected_counts,counts
 ids={mu+'_'+rule for mu in LABELS.values() for rule in RULES.values()}
 const={g+'_soft_'+k+'_'+rule for g in GROUPS for k in KINDS for rule in RULES.values()}
 expected_sets={'learned_economics':ids,'constant_economics':const,'regime_economics':{(cid,st) for cid in ids for st in STRATA[1:]},'paired_economics':{(mu,ref,rule) for mu in LABELS.values() for ref in refs(mu)[:2] for rule in RULES.values()},'mapped_mse':{(mu,seg) for mu in LABELS.values() for seg in SEGMENTS},'mechanism':{(mu,seg) for mu in LABELS.values() for seg in SEGMENTS},'T_scalars':set(FOLDS)}
 for name,keys in seen.items():assert len(keys)==len(set(keys)) and set(keys)==expected_sets[name],(name,'missing or duplicate')
 table_cells=len(CELL_CHECKS)-2
 def mse(mu,ref,seg,st='all'):return s['paired'][st][mu][ref]['prediction'][seg]['mse_difference']
 def flat(v):
  if isinstance(v,dict):return [x for y in v.values() for x in flat(y)]
  assert type(v) is bool;return [v]
 flags=flat(s['soft']);assert len(flags)==72 and not any(flags)
 claims={
  'all_8_learned_policies_fail_all_registered_gates':not any(flags),
  'L2unit_E_MSE_better_hard_all_4_strata_7_of_8_quarters':all(mse(mu,refs(mu)[0],'evaluation',st)<0 for mu in LABELS.values() if '_l2unit_' in mu for st in STRATA) and all(s['paired']['all'][mu][refs(mu)[0]]['prediction']['evaluation']['improved_quarters']==7 for mu in LABELS.values() if '_l2unit_' in mu),
  'L2unit_E_sideways_MSE_worse_half_and_all_strata_worse_fit_prior':all(mse(mu,refs(mu)[1],'evaluation','sideways')>0 and all(mse(mu,ref,'evaluation',st)>0 for st in STRATA for ref in refs(mu)[2:4]) for mu in LABELS.values() if '_l2unit_' in mu),
  'scalar_mapping_max_difference_exact_0':ma['maximum_absolute_mapping_difference']==0,
  'mapping_audit_160_NPZ_160_helper_64_learned_8_provenance_48_masks':ma['counts']['new_prediction_npz']==160 and ma['counts']['helper_diagnostic_records']==160 and ma['counts']['learned_mapping_diagnostics']==64 and ma['counts']['mapping_records']==8 and ma['counts']['common_mask_bindings']==48,
  'mapping_audit_20492_learned_rows_10246_each_constant':ma['counts']['mapped_soft_rows']==20492 and all(ma['counts']['mapped_'+k+'_rows']==10246 for k in ('fit_mean','mapped_prior','zero')),
  'mapping_bound_33_source_3488_ancestor_968_new':a['verified_binding_counts']['registered_source']==33 and a['verified_binding_counts']['ancestor_artifact']==3488 and a['verified_binding_counts']['new_artifact']==968,
  'own_paths_64_learned_96_constant_160_traces':o['counts']['new_learned_paths']==64 and o['counts']['new_constant_control_paths']==96 and o['counts']['new_full_trace_replays']==160,
  'own_all_errors_zero_except_mean_exposure_2_220446049250313e_16':all(v==0 for k,v in o['max_absolute_differences'].items() if k!='account_mean_exposure') and o['max_absolute_differences']['account_mean_exposure']==2.220446049250313e-16,
  'scalar_errors_and_Decimal_summary_reported_exact':all(a['numeric_max_absolute_differences'][k]==v for k,v in {'scalar_return_score':5e-18,'scalar_classification_score':1e-16,'summary_economics':3.5e-15,'summary_paired':9e-16}.items()),
  '48_cross_group_plus_32_mapped_prior_fitmean_pairs':sum(x['comparison'].startswith('cross_group_') for x in m['fixed_constant_target_comparisons'])==48 and sum('mapped_prior_vs_fit_mean' in x['comparison'] for x in m['fixed_constant_target_comparisons'])==32,
  '32_zero_paths_24_trade_both_costs_base_range_0_to_8':co['zero_policy_paths']==32 and co['zero_paths_with_base_trades']==co['zero_paths_with_stress_trades']==24 and co['zero_base_trade_count_range']==[0,8],
  'prior_residual_max_1_301e_18_savedq_diff_fold12_plus5_551e_17':m['max_abs_mapped_prior_fit_mean_residual']==1.3010426069826053e-18 and [(x['fold'],x['saved_prior_minus_statistical_prior']) for x in m['constant_identity_by_fold'] if x['saved_prior_minus_statistical_prior']!=0]==[(12,5.551115123125783e-17)],
  'C1_overall_MSE_worse_all_5_refs_I_E':all(mse(mu,r,seg)>0 for mu in LABELS.values() if '_l2unit_' not in mu for r in refs(mu) for seg in SEGMENTS),
  'L2unit_overall_MSE_better_hard_half_I_E':all(mse(mu,r,seg)<0 for mu in LABELS.values() if '_l2unit_' in mu for r in refs(mu)[:2] for seg in SEGMENTS),
  'L2unit_overall_MSE_worse_all_3_constants_I_E':all(mse(mu,r,seg)>0 for mu in LABELS.values() if '_l2unit_' in mu for r in refs(mu)[2:] for seg in SEGMENTS),
  'Perp_C1_overall_Alpha_positive_DD_positive_both_costs':all(s['economics']['all']['perp_delay0_magnitude_soft_'+rule][c]['alpha_ex']>0 and s['economics']['all']['perp_delay0_magnitude_soft_'+rule][c]['maxdd_delta']>0 for rule in RULES.values() for c in COSTS),
  'all_remaining_learned_overall_joint_signs_fail':all(not(s['economics']['all'][cid][c]['alpha_ex']>0 and s['economics']['all'][cid][c]['maxdd_delta']<0) for cid in ids if not cid.startswith('perp_delay0_magnitude_soft_') for c in COSTS),
  'all_constant_overall_Alpha_negative_both_costs':all(s['economics']['all'][cid][c]['alpha_ex']<0 for cid in const for c in COSTS),
  'all_8_learned_economics_worse_half_overall_both_costs':all(s['paired']['all'][mu][group(mu)+'_half']['economics'][rule][c]['alpha_ex']<0 and s['paired']['all'][mu][group(mu)+'_half']['economics'][rule][c]['maxdd_delta']>0 for mu in LABELS.values() for rule in RULES.values() for c in COSTS),
  'probability_160_records_unchanged':a['inventory']['unchanged_classification_scores']==160 and a['numeric_max_absolute_differences']['unchanged_classification_score']==0,
  'old_480_rows_and_224_return_records_exact':a['inventory']['unchanged_control_rows']==480 and a['inventory']['unchanged_return_scores']==224 and a['numeric_max_absolute_differences']['unchanged_old_economic_row']==a['numeric_max_absolute_differences']['unchanged_old_return_score']==0,
  'inherited_source_weighted_gates_false_I_E':all(not v for x in s['soft'].values() for v in x['inherited_source_weighted_losses_below_prior_all_strata'].values()),
  'C1_amplitude_larger_L2unit_smaller_overall_both_segments':all((m['mechanism_summary'][mu][seg]['mean_abs_new_mu_equal_quarter']<m['mechanism_summary'][mu][seg]['mean_abs_hard_mu_equal_quarter'])==('_l2unit_' in mu) for mu in LABELS.values() for seg in SEGMENTS),
  'all_q_logit_mu_ties_and_direction_disagreements_zero':all(v==0 for k,v in m['mechanism_aggregate_counts'].items() if k!='new_abs_mu_greater_than_hard_rows'),
  'all_80_saved_constant_target_and_metric_pairs_exact':m['fixed_constant_target_comparisons_exact']==m['fixed_constant_saved_metrics_comparisons_exact']==80,
  'all_8_mapped_prior_fitmean_residuals_nonzero':m['nonzero_mapped_prior_fit_mean_residual_folds']==8,
  'one_raw_vs_saved_prior_difference_fold':m['nonzero_saved_vs_statistical_prior_difference_folds']==1,
  'no_fit_or_probability_prediction_call':a['inventory']['new_models']==a['inventory']['new_prediction_calls']==0,
  'score_support_2574_E_2523_I':all(s['prediction']['all'][seg][mu]['rows']==(2574 if seg=='evaluation' else 2523) for mu in LABELS.values() for seg in SEGMENTS),
  'inference_support_2586_E_2537_I':all(m['mechanism_summary'][mu][seg]['inference_rows']==(2586 if seg=='evaluation' else 2537) for mu in LABELS.values() for seg in SEGMENTS),
  'all_96_constant_ICs_null':a['inventory']['constant_score_records']==96 and all(s['prediction'][st][seg][g+'_soft_'+k]['return_rank_ic'] is None for st in STRATA for seg in SEGMENTS for g in GROUPS for k in KINDS),
  'tiny_DD_66_total_56_new_constant_10_old_none_new_learned':len(a['tiny_nonzero_dd_values'])==66 and sum('_soft_' in r['candidate_id'] for r in a['tiny_nonzero_dd_values'])==56 and not any(r['candidate_id'] in ids for r in a['tiny_nonzero_dd_values']) and {abs(r['maxdd_delta']) for r in a['tiny_nonzero_dd_values']}=={1.1102230246251565e-16},
  'numerical_own_state_audit_1280_accounts_160_paths_55040_decisions':o['counts']['independent_scalar_accounts']==1280 and o['counts']['new_own_state_paths']==160 and o['counts']['new_own_state_decisions']==55040,
  'independent_scalar_score_384_returns_160_classifications':a['inventory']['return_scores']==384 and a['inventory']['classification_scores']==160,
  'all_80_policies_640_rows_1280_cost_accounts':a['inventory']['policies']==80 and a['inventory']['economic_rows']==640 and a['inventory']['accounts']==1280,
  'selection_generalization_and_regime_gate_false':s['selection_performed']==s['high_probability_generalization_established']==s['regime_count_gate_pass']==False,
 }
 assert all(claims.values()),[k for k,v in claims.items() if not v]
 for phrase in ('not annualized','equal-quarter','Pooled-row MSE','both I and E','Probabilities and all 160 classification records are unchanged','not independent confirmation','2 bull / 4 bear / 2 sideways','retrospective E-start regimes','not proof of production receipt-time availability','No Stage20 procedure is registered'):
  assert phrase in text,('missing material qualifier',phrase)
 direct_report_hashes=[EXPECTED[OUT/x] for x in ('results.json','registration.json','preflight.json','run.log')]
 direct_report_hashes+=['9301f9e587a0fbcf1b61759c23dd9656dfc31b59d1ba4ebaa1f80c9effde766a','92f7382c4165adca5480d5df4560d84e2e4559ea09a090fb7e77092470a9345d']
 assert all(h in text for h in direct_report_hashes)
 log=(OUT/'run.log').read_text();assert 'Warning' not in log
 events=[json.loads(line) for line in log.splitlines()];assert len(events)==8 and [e['fold'] for e in events]==list(FOLDS) and all(e['event']=='fold_complete' for e in events)
 testtext=TESTLOG.read_text();assert re.search(r'Ran 752 tests in 58\.614s\s+OK(?:\s|$)',testtext) is not None
 assert sha(REPORT)==expected_report_sha,'Report changed during audit'
 result={'schema':'oracle_soft_direction_final_report_audit_v1','passed':True,'report_path':str(REPORT.relative_to(ROOT)),
 'report_sha256':expected_report_sha,'previous_draft_report_sha256':'4bf7326444140555c59891ee33b366aca95c310161292e67a1f5e9cd53129b4d',
 'previous_draft_was_not_final_verification':True,'initial_final_report_table_audit_sha256':'f922b313df3580c313e9746c5412e9be693b3e58c6cfbdbc3f5b5a811f76a269','initial_table_audit_superseded_by_extended_narrative_checks':True,'audit_script_sha256':sha(__file__),
 'scope':'Sealed independent score/Decimal60 summary and mapping/account evidence to all final report tables and material claims; no model, probability predictor, scorer or policy replay.',
 'source_sha256':{str(p):h for p,h in EXPECTED.items()},'table_rows':counts,'total_table_rows':sum(counts.values()),
 'numeric_and_joint_table_cells':table_cells,'report_direct_hash_count':len(direct_report_hashes),
 'table_checks':CELL_CHECKS,'mechanism_aggregation_check':mechanism_check,'narrative_claim_checks':claims,
 'registered_nested_flags_all_false':len(flags),'no_pending_report_markers':True,
 'limitations':['Evidence remains reused development data, not independent confirmation.','Full source/calendar, mapping arithmetic and economic account audits are separately sealed; not duplicated here.','Formatting parity verifies reported precision, not extra statistical significance.','The prior draft was not the final publication audit.']}
 OUTPUT.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'passed':True,'path':str(OUTPUT),'sha256':sha(OUTPUT),'report_sha256':expected_report_sha,'table_rows':counts,'table_cells':table_cells,'narrative_checks':len(claims)}))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--expected-report-sha',required=True);args=parser.parse_args()
 assert re.fullmatch('[0-9a-f]{64}',args.expected_report_sha)
 main(args.expected_report_sha)
