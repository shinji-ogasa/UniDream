"""Recompute fixed reused-cohort summaries using only immutable rows and scores.
No repository imports, models, policy rollout, or candidate selection.
"""
from decimal import Decimal, localcontext
from collections import Counter
import hashlib,json,math
from pathlib import Path

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/oracle_additional_window_replay_v1'
DEST=Path('/tmp/oracle_additional_window_summary_audit_20260906.json')
FOLDS=tuple(range(15,25))
MEANS=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')
RULES=('utility_risk1','utility_risk1_fallback_bh')
POLICIES=('bh','common_robust')+tuple(m+'_'+r for m in MEANS for r in RULES)
CANDIDATES=tuple(m+'_'+r for m in MEANS[3:] for r in RULES)
STRATA=('all','bull','bear','sideways')
COSTS=('base','stress_2x')
PAIRS=(('technical_half','scale_mean'),('technical_half','technical_scaled'),('perp_delay0_half','scale_mean'),('perp_delay0_half','perp_delay0_scaled'),('perp_delay0_half','technical_half'))
ECON=('alpha_ex','maxdd_delta','turnover','trades','fees_initial_equity_units','borrow_initial_equity_units')
LOSSES=('return_mse','return_mae','zero_return_mse','fit_mean_return_mse','return_sign_accuracy')
BINDINGS={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p).resolve();BINDINGS[str(p)]=sha(p);return p

def obj(p):return json.loads(bind(p).read_text())
def dec(v):return Decimal.from_float(float(v))
def average(values):
 values=list(values)
 if not values:return None
 with localcontext() as ctx:
  ctx.prec=60
  return float(sum((dec(v) for v in values),Decimal(0))/len(values))
def difference(a,b):
 with localcontext() as ctx:
  ctx.prec=60;return float(dec(a)-dec(b))
def weighted(values,weights):
 values=list(values);weights=list(weights)
 if not values:return None
 with localcontext() as ctx:
  ctx.prec=60;return float(sum((dec(v)*Decimal(w) for v,w in zip(values,weights)),Decimal(0))/sum(weights))
def relative(a,b):
 if b is None or b==0:return None
 with localcontext() as ctx:
  ctx.prec=60;return float(Decimal(1)-dec(a)/dec(b))

data=obj(OUT/'results.json');reg=obj(OUT/'registration.json');pre=obj(OUT/'preflight.json')
assert reg['source_revision']=='73cb806bffb7dc39d7454886e9bcfce9119dc435'
for path,expected in reg['source_bindings'].items():assert sha(bind(ROOT/path))==expected
cfg=reg['config'];assert sha(bind(ROOT/'configs/oracle_additional_window_replay_20260906.yaml'))==reg['config_sha256']
assert sha(bind(ROOT/cfg['family_path']))==cfg['family_sha256']
assert sha(bind(ROOT/cfg['data_manifest_path']))==cfg['data_manifest_sha256']
rows={(r['fold'],r['candidate_id']):r for r in data['rows']};scores={(s['fold'],s['mean_id']):s for s in data['scores']}
assert len(rows)==120 and len(scores)==50
assert set(rows)=={(f,c) for f in FOLDS for c in POLICIES}
assert set(scores)=={(f,m) for f in FOLDS for m in MEANS}
for f in FOLDS:
 fold=obj(OUT/f'fold_{f}.json')
 assert fold['rows']==[r for r in data['rows'] if r['fold']==f]
 assert fold['scores']==[s for s in data['scores'] if s['fold']==f]
 assert len({scores[f,m]['rows'] for m in MEANS})==1
 assert all(rows[f,c]['regime']==scores[f,'scale_mean']['regime'] for c in POLICIES)
 assert all(scores[f,m]['regime']==scores[f,'scale_mean']['regime'] for m in MEANS)
regimes={f:scores[f,'scale_mean']['regime']['trend'] for f in FOLDS}
counts={g:sum(v==g for v in regimes.values()) for g in ('bull','bear','sideways','unavailable')}
coverage=all(counts[g]>=3 for g in STRATA[1:]) and counts['unavailable']==0
policy_summary={c:{} for c in POLICIES};mean_summary={m:{} for m in MEANS};pairs={};policy_pairs={};rules={}
for g in STRATA:
 fs=[f for f in FOLDS if g=='all' or regimes[f]==g]
 for c in POLICIES:
  policy_summary[c][g]={'quarters':len(fs)}
  for cost in COSTS:
   policy_summary[c][g][cost]={k+'_mean':average(rows[f,c][cost][k] for f in fs) for k in ECON}
   policy_summary[c][g][cost]['joint_positive_quarters']=sum(rows[f,c][cost]['alpha_ex']>0 and rows[f,c][cost]['maxdd_delta']<0 for f in fs)
 for m in MEANS:
  ss=[scores[f,m] for f in fs];weights=[s['rows'] for s in ss];ics=[s['return_rank_ic'] for s in ss if s['return_rank_ic'] is not None]
  mean_summary[m][g]={'quarters':len(fs),'scored_rows':sum(weights),'equal_quarter':{k:average(s[k] for s in ss) for k in LOSSES},'row_pooled':{k:weighted((s[k] for s in ss),weights) for k in LOSSES},'rank_ic_mean_defined_quarters':average(ics),'rank_ic_defined_quarters':len(ics)}
 for c,r in PAIRS:
  key=c+'_vs_'+r;weights=[scores[f,c]['rows'] for f in fs];d=[difference(scores[f,c]['return_mse'],scores[f,r]['return_mse']) for f in fs]
  entry={'quarters':len(fs),'scored_rows':sum(weights),'candidate_minus_reference_mse':average(d),'row_pooled_candidate_minus_reference_mse':weighted(d,weights),'improved_quarters':sum(x<0 for x in d)}
  for k in ('return_mse','return_mae'):entry[k+'_relative_loss_reduction']=relative(average(scores[f,c][k] for f in fs),average(scores[f,r][k] for f in fs))
  pairs.setdefault(key,{})[g]=entry
  for rule in RULES:
   cid,rid=c+'_'+rule,r+'_'+rule
   policy_pairs.setdefault(cid+'_vs_'+rid,{})[g]={'quarters':len(fs),**{cost:{k:average(difference(rows[f,cid][cost][k],rows[f,rid][cost][k]) for f in fs) for k in ECON} for cost in COSTS}}
 for m in MEANS[3:]:
  c,r=m+'_'+RULES[1],m+'_'+RULES[0]
  rules.setdefault(m,{})[g]={'quarters':len(fs),**{cost:{k:average(difference(rows[f,c][cost][k],rows[f,r][cost][k]) for f in fs) for k in ECON} for cost in COSTS}}
components=[]
for c in CANDIDATES:
 for g in STRATA:
  for cost in COSTS:
   for k,metric,sign in [('alpha_ex_mean','alpha_ex',1),('maxdd_delta_mean','negative_maxdd_delta',-1)]:
    v=policy_summary[c][g][cost][k];components.append({'id':'/'.join(('economic',c,g,cost,metric)),'value':None if v is None else sign*v,'p_value':None})
for c,r in PAIRS[:4]:
 for g in STRATA:
  v=pairs[c+'_vs_'+r][g]['candidate_minus_reference_mse'];components.append({'id':'/'.join(('predictive',c,r,g,'negative_mse_difference')),'value':None if v is None else -v,'p_value':None})
assert len(components)==80 and len({c['id'] for c in components})==80
candidate_summary={};failed={}
for c in CANDIDATES:
 m=next(m for m in MEANS[3:] if c.startswith(m+'_'))
 econ=[x for x in components if x['id'].startswith('economic/'+c+'/')]
 pred=[x for x in components if x['id'].startswith('predictive/'+m+'/')]
 ep=all(x['value'] is not None and x['value']>0 for x in econ);pp=all(x['value'] is not None and x['value']>0 for x in pred)
 candidate_summary[c]={'high_probability_generalization_established':False,'observed_economic_signs':ep,'observed_predictive_signs':pp,'observed_metric_and_coverage_conditions_met':coverage and ep and pp}
 failed[c]={'economic':[x for x in econ if x['value'] is None or x['value']<=0],'predictive':[x for x in pred if x['value'] is None or x['value']<=0]}

stats={};checks=Counter()
def compare(a,b,path,group):
 checks['recursive_comparisons']+=1
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
  for k in a:compare(a[k],b[k],path+'.'+k,group)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]',group)
 elif a is None or type(a) in (str,bool,int):assert type(a) is type(b) and a==b,(path,a,b)
 else:
  assert isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b),(path,'nonfinite')
  d=abs(a-b);v=stats.setdefault(group,{'numeric_values':0,'max_absolute_difference':0.,'at':None});v['numeric_values']+=1
  if d>v['max_absolute_difference']:v.update(max_absolute_difference=d,at=path)
  assert d<=1e-14+1e-12*abs(b),(path,a,b,d)

expected={'policies':policy_summary,'means':mean_summary,'paired_mse':pairs,'paired_policies':policy_pairs,'fallback_minus_hold':rules,'descriptive_components':components,'candidates':candidate_summary,'regime_counts':counts,'regime_coverage':coverage}
for k,v in expected.items():compare(v,data['summary'][k],k,k)
for key in ('p_values','adjusted_p_values','confidence_intervals'):assert data['summary'][key] is None
for key in ('selection_performed','high_probability_generalization_established','independent_confirmation','receipt_provenance_established'):assert data[key] is False and data['summary'][key] is False
assert data['summary']['component_count']==80 and data['summary']['policy_rows']==120 and data['summary']['forecast_rows']==50

half_findings={}
for m,full in [('technical_half','technical_scaled'),('perp_delay0_half','perp_delay0_scaled')]:
 changed=[f for f in FOLDS if any(rows[f,m+'_'+RULES[1]][cost]!=rows[f,m+'_'+RULES[0]][cost] for cost in COSTS)]
 half_findings[m]={'economic_means_percentage_points':{rule:{g:{cost:{k:100*policy_summary[m+'_'+rule][g][cost][k+'_mean'] for k in ('alpha_ex','maxdd_delta')} for cost in COSTS} for g in STRATA} for rule in RULES},'mse_improved_quarters_vs_own_full':{g:pairs[m+'_vs_'+full][g]['improved_quarters'] for g in STRATA},'mse_improved_quarters_vs_scale':{g:pairs[m+'_vs_scale_mean'][g]['improved_quarters'] for g in STRATA},'forecast_loss_reduction_percent':{ref:{g:{loss:100*pairs[m+'_vs_'+ref][g][loss+'_relative_loss_reduction'] for loss in ('return_mse','return_mae')} for g in STRATA} for ref in (full,'scale_mean')},'fallback_changes_only_these_folds':changed,'fallback_minus_hold_all':rules[m]['all']}
bh_alpha=max(abs(rows[f,'bh'][c]['alpha_ex']) for f in FOLDS for c in COSTS);bh_dd=max(abs(rows[f,'bh'][c]['maxdd_delta']) for f in FOLDS for c in COSTS)
assert max(bh_alpha,bh_dd)<1e-14
report={'schema':'independent-additional-window-summary-audit-v1','passed':True,'source_revision':reg['source_revision'],'scope':'Reused original test15..24 only; independent Decimal60 sum of saved rows/scores, no new models, policies, selection, intervals or p-values. Forecast scalar/accounting path audits are separate.','old_test_folds':list(FOLDS),'regime_counts':counts,'regime_coverage':coverage,'inventory':{'economic_rows':120,'forecast_scores':50,'cost_metric_objects':240,'components':80,'primary_economic_components':64,'primary_predictive_components':16,'forecast_pairs':5,'policy_pairs':10,'fallback_rule_pairs':2},'comparison':stats,'checks':dict(checks),'candidate_conditions':candidate_summary,'failed_components':failed,'findings':half_findings,'all_policy_aggregate':{cid:policy_summary[cid]['all'] for cid in POLICIES},'paired_policy_all':{k:v['all'] for k,v in policy_pairs.items()},'bh_roundoff':{'maximum_abs_alpha_ratio':bh_alpha,'maximum_abs_dd_ratio':bh_dd,'stored_joint_positive_quarters':data['summary']['policies']['bh']['all']['base']['joint_positive_quarters'],'interpretation':'B&H against itself is the zero reference. Stored strict-sign counts from floating-point roundoff are not economic improvement; leave immutable result untouched and label reference zero in prose/tables.'},'source_file_sha256':BINDINGS,'script_sha256':sha(__file__)}
DEST.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'artifact':str(DEST),'sha256':sha(DEST),'bindings':len(BINDINGS),'comparison':stats,'fallback_changed_folds':{m:half_findings[m]['fallback_changes_only_these_folds'] for m in half_findings}},sort_keys=True))
