"""Independent saved-artifact audit: Decimal aggregation and scalar score/rank checks.
No helper imports, model fits, policies, or additional market windows.
"""
import argparse,hashlib,json,math
from pathlib import Path
from decimal import Decimal, localcontext
from collections import Counter
import numpy as np
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/oracle_rolling_centering_decisions_v1'
PARENT=ROOT/'codex_outputs/oracle_mean_reliability_decisions_v1'
PAR=ROOT/'codex_outputs/oracle_frozen_procedure_parity_v1'
RAW=ROOT/'codex_outputs/oracle_derivative_delay_v1'
REPORT=Path('/tmp/oracle_rolling_centering_summary_audit_20260906.json')
FS=tuple(range(5,13));GS=('technical','perp_delay0');SS=('evaluation',)
CS=('base','stress_2x');RS=('utility_risk1','utility_risk1_fallback_bh');STR=('all','bull','bear','sideways')
OLD=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')
RELIABILITY=tuple(g+'_reliability' for g in GS)
OLD_MEANS=OLD+tuple(g+'_raw' for g in GS)+RELIABILITY
NEW=('rolling_anchor','technical_rolling','perp_delay0_rolling')
MEANS=OLD_MEANS+NEW
CONTROLS=('bh','common_robust')+tuple(m+'_'+q for m in OLD for q in RS)+tuple(m+'_'+q for m in RELIABILITY for q in RS)
IDS=CONTROLS+tuple(m+'_'+q for m in NEW for q in RS)
REFERENCES={'rolling_anchor':('scale_mean',),**{g+'_rolling':('rolling_anchor','scale_mean',g+'_reliability',g+'_scaled',g+'_half') for g in GS}}
EK=('alpha_ex','maxdd_delta','turnover','trades')
DK=('lossdiff','innovation_secondmoment','crossmoment','centered_component','drift_component','identityresidual')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def D(x):return Decimal(str(x))
def avg(v):return sum((D(x) for x in v),Decimal(0))/Decimal(len(v))
def plain(x):
 if isinstance(x,Decimal):return float(x)
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
bindings={};categories=Counter()
def bind(p,h=None,cat='direct'):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;s=sha(p)
 assert h is None or s==h,(str(p),'hash changed')
 key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
 if key not in bindings:bindings[key]=s;categories[cat]+=1
 else:assert bindings[key]==s
 return p
maxdiff={};locations={}
def compare(a,b,path,cat):
 if isinstance(a,dict):
  assert set(a)==set(b),(path,'schema')
  for k in a:compare(a[k],b[k],path+'/'+str(k),cat)
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),cat)
 elif isinstance(a,(int,float,Decimal)) and not isinstance(a,bool):
  assert math.isfinite(float(a)) and math.isfinite(float(b)),path
  d=abs(D(a)-D(b))
  if cat not in maxdiff or d>maxdiff[cat]:maxdiff[cat]=d;locations[cat]=path
 else:assert a==b,(path,a,b)

def ranks(a):
 _,inv,c=np.unique(a,return_inverse=True,return_counts=True)
 return (np.cumsum(c)-(c-1)/2)[inv]
def scalar_scores(y,p,fit):
 n=len(y);e=[float(a)-float(b) for a,b in zip(y,p)]
 ic=None
 if len(np.unique(y))>1 and len(np.unique(p))>1:
  ry,rp=ranks(y),ranks(p);mid=(n+1)/2
  cov=math.fsum((float(a)-mid)*(float(b)-mid) for a,b in zip(ry,rp))
  den=math.sqrt(math.fsum((float(a)-mid)**2 for a in ry)*math.fsum((float(b)-mid)**2 for b in rp))
  ic=cov/den
 return {'rows':n,'return_mse':math.fsum(z*z for z in e)/n,'return_mae':math.fsum(abs(z) for z in e)/n,
  'zero_return_mse':math.fsum(float(z)**2 for z in y)/n,
  'fit_mean_return_mse':math.fsum((float(z)-fit)**2 for z in y)/n,
  'return_sign_accuracy':sum((float(a)>0)==(float(b)>0) for a,b in zip(y,p))/n,'return_rank_ic':ic}
def scalar_decomp(y,p,a):
 n=len(y);d=[float(z)-float(v) for z,v in zip(p,a)];r=[float(z)-float(v) for z,v in zip(y,a)];e=[float(z)-float(v) for z,v in zip(y,p)]
 md=math.fsum(d)/n;mr=math.fsum(r)/n
 b=math.fsum(z*z for z in d)/n;c=math.fsum(z*v for z,v in zip(d,r))/n
 vd=math.fsum((z-md)**2 for z in d)/n;cv=math.fsum((z-md)*(v-mr) for z,v in zip(d,r))/n
 mse=math.fsum(z*z for z in e)/n;am=math.fsum(z*z for z in r)/n;loss=mse-am;center=vd-2*cv;drift=md*md-2*md*mr
 return {'n':n,'candidate_mse':mse,'anchor_mse':am,'lossdiff':loss,'mean_d':md,'mean_r':mr,
 'innovation_secondmoment':b,'crossmoment':c,'centered_variance_d':vd,'centered_covariance':cv,
 'centered_component':center,'drift_component':drift,'identityresidual':loss-center-drift}

def canonical(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def equal_arrays(left,right,name):
 assert set(left)==set(right),(name,'schema')
 for k in left:
  assert left[k].dtype==right[k].dtype,(name,k,'dtype')
  assert np.array_equal(left[k],right[k],equal_nan=True),(name,k,'array changed')

def audit():
 with localcontext() as ctx:
  ctx.prec=60
  result=read(bind(OUT/'results.json'));reg=read(bind(OUT/'registration.json'));cfg=reg['config']
  pre=read(bind(OUT/'preflight.json',cfg['preflight_sha256']))
  configs=[p for p in (ROOT/'configs').glob('oracle_rolling_centering*.yaml') if sha(p)==reg['config_sha256']]
  assert len(configs)==1,'exact registered config not found'
  bind(configs[0],reg['config_sha256'],'config')
  assert result['registration_sha256']==canonical(reg)
  assert reg['source_bindings']==cfg['source_bindings']==pre['source_bindings']
  assert pre['config_contract_sha256']==canonical({k:v for k,v in cfg.items() if k!='preflight_sha256'})
  assert len(cfg['source_bindings'])==27
  for p,h in cfg['source_bindings'].items():bind(p,h,'registered_source')
  for p,h in pre['direct_source_bindings'].items():bind(p,h,'preflight_direct')
  bind(cfg['source_prepare_config'],cfg['source_prepare_config_sha256'],'parent_config')
  ancestor=pre['source_artifact_bindings']
  assert len(ancestor)==1536,'ancestral inventory changed'
  for p,h in ancestor.items():bind(p,h,'ancestral_artifact')
  parent=read(bind(PARENT/'results.json',cfg['parent_results_sha256']))
  parent_reg=read(bind(PARENT/'registration.json',cfg['parent_registration_sha256']))
  parent_pre=read(bind(PARENT/'preflight.json',cfg['parent_preflight_sha256']))
  assert parent_reg['source_revision']==cfg['parent_source_revision']
  assert parent['registration_sha256']==canonical(parent_reg)
  assert pre['parent_prepare_preflight_sha256']==canonical(parent_pre)
  rows,scores,weights=result['rows'],result['scores'],result['fixed_weights']
  ri={(x['fold'],x['candidate_id']):x for x in rows}
  si={(x['fold'],x['mean_id']):x for x in scores}
  wi={(x['fold'],x['group']):x for x in weights}
  assert len(rows)==len(ri)==176 and set(ri)=={(f,p) for f in FS for p in IDS}
  assert len(scores)==len(si)==96 and set(si)=={(f,m) for f in FS for m in MEANS}
  assert len(weights)==len(wi)==16 and set(wi)=={(f,g) for f in FS for g in GS}
  parent_ri={(x['fold'],x['candidate_id']):x for x in parent['rows']}
  parent_si={(x['fold'],x['mean_id']):x for x in parent['scores'] if x['segment']=='evaluation'}
  parent_wi={(x['fold'],x['group']):x for x in parent['fits']}
  assert wi==parent_wi,'copied fixed weights changed'
  assert result['base_models_fitted']==0 and result['calibration_weights_fitted']==0
  assert result['fixed_weights_copied']==16 and result['new_causal_policy_names']==6
  assert result['total_adaptively_explored_causal_names']==168
  assert not any(result[k] for k in ('selection_performed','additional_test_used_for_modeling_or_scoring','teacher_use_allowed','high_probability_generalization_established'))
  regimes={f:ri[f,'bh']['regime'] for f in FS}
  counts=dict(Counter(x['trend'] for x in regimes.values()))
  assert counts=={'bull':2,'bear':4,'sideways':2}
  artifacts={}
  for f in FS:
   q=read(bind(OUT/f'fold_{f}.json',cat='fold_manifest'))
   assert q['registration_sha256']==result['registration_sha256'] and len(q['artifact_sha256'])==32
   for key in ('rows','scores','fixed_weights'):assert q[key]==[x for x in result[key] if x['fold']==f]
   expected={str((OUT/'forecasts'/f'fold{f}_{m}.npz').relative_to(ROOT)) for m in NEW}
   expected|={str((OUT/'targets'/f'fold{f}_{cid}.npz').relative_to(ROOT)) for cid in IDS}
   expected|={str((OUT/'traces'/f'fold{f}_{cid}.json').relative_to(ROOT)) for cid in IDS if cid not in CONTROLS}
   expected.add(str((OUT/'rolling_traces'/f'fold{f}_shared_history.json').relative_to(ROOT)))
   assert set(q['artifact_sha256'])==expected,'new artifact identity mismatch'
   for p,h in q['artifact_sha256'].items():
    assert p not in artifacts;artifacts[p]=h;bind(p,h,'new_artifact')
  assert len(artifacts)==256
  for r in rows:
   assert r['regime']==regimes[r['fold']]
   p=str((OUT/'targets'/f"fold{r['fold']}_{r['candidate_id']}.npz").relative_to(ROOT))
   assert r['targets_sha256']==artifacts[p]
   if r['candidate_id'] not in CONTROLS:
    p=str((OUT/'traces'/f"fold{r['fold']}_{r['candidate_id']}.json").relative_to(ROOT));assert r['trace_sha256']==artifacts[p]
  for s in scores:
   assert s['regime']==regimes[s['fold']] and s['regime_known_at_scored_decisions'] is True
   assert s['segment']=='evaluation' and s['decomposition_anchor']=='rolling_anchor'
   assert s['rows']==si[s['fold'],'scale_mean']['rows'] and s['rows']>=16
  assert sum(si[f,'scale_mean']['rows'] for f in FS)==2574
  def folds(g):return [f for f in FS if g=='all' or regimes[f]['trend']==g]
  econ={};pred={};paired={}
  for g in STR:
   ff=folds(g)
   econ[g]={p:{'quarters':len(ff),'joint_positive_quarters_both_costs':sum(all(ri[f,p][c]['alpha_ex']>0 and ri[f,p][c]['maxdd_delta']<0 for c in CS) for f in ff),
    **{c:{k:avg([ri[f,p][c][k] for f in ff]) for k in EK} for c in CS}} for p in IDS}
   pred[g]={}
   for m in MEANS:
    xx=[si[f,m] for f in ff];n=sum(x['rows'] for x in xx)
    pred[g][m]={'quarters':len(ff),'rows':n,'equal_quarter_mse':avg([x['return_mse'] for x in xx]),
     'pooled_row_mse':sum(D(x['return_mse'])*Decimal(x['rows']) for x in xx)/Decimal(n),
     'equal_quarter_mae':avg([x['return_mae'] for x in xx]),'zero_return_mse':avg([x['zero_return_mse'] for x in xx]),
     'fit_mean_return_mse':avg([x['fit_mean_return_mse'] for x in xx]),'mse_minus_zero':avg([D(x['return_mse'])-D(x['zero_return_mse']) for x in xx]),
     'mse_minus_fit_mean':avg([D(x['return_mse'])-D(x['fit_mean_return_mse']) for x in xx]),
     'mean_rank_ic':avg([x['return_rank_ic'] for x in xx]) if all(x['return_rank_ic'] is not None for x in xx) else None,
     'relative_to_rolling_anchor_decomposition':{k:avg([x['decomposition'][k] for x in xx]) for k in DK}}
   paired[g]={}
   for m in NEW:
    paired[g][m]={}
    for ref in REFERENCES[m]:
     ds=[D(si[f,m]['return_mse'])-D(si[f,ref]['return_mse']) for f in ff]
     ref_loss=avg([si[f,ref]['return_mse'] for f in ff])
     pp={'mse_difference':avg(ds),'relative_mse_reduction':-avg(ds)/ref_loss if ref_loss else None,
      'improved_quarters':sum(x<0 for x in ds),'equal_quarters':sum(x==0 for x in ds)}
     paired[g][m][ref]={'prediction':pp,'economics':{q:{c:{k:avg([D(ri[f,m+'_'+q][c][k])-D(ri[f,ref+'_'+q][c][k]) for f in ff]) for k in EK} for c in CS} for q in RS}}
  direction={}
  for m in NEW:
   predictive=all(pred[g][m]['mse_minus_zero']<0 and pred[g][m]['mse_minus_fit_mean']<0 and all(paired[g][m][ref]['prediction']['mse_difference']<0 for ref in REFERENCES[m]) for g in STR)
   for q in RS:
    p=m+'_'+q;direction[p]={'economic_means_all_strata_both_costs':all(econ[g][p][c]['alpha_ex']>0 and econ[g][p][c]['maxdd_delta']<0 for g in STR for c in CS),
     'predictive_mse_vs_zero_fitmean_and_all_registered_references_all_strata':predictive,'regime_count_gate_pass':False,'high_probability_generalization_established':False}
  summary={'economics':econ,'prediction':pred,'paired':paired,'direction':direction,'fixed_weights':weights,'regime_counts':counts,
   'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False,'intercept_components_separately_identified':False}
  assert set(summary)==set(result['summary'])
  for name,x in summary.items():compare(x,result['summary'][name],name,'summary_'+name)
  cache={}
  def npz(p):
   p=Path(p);key=str(p.relative_to(ROOT))
   assert key in bindings,'consumed file not independently bound'
   if key not in cache:
    with np.load(p,allow_pickle=False) as z:cache[key]={k:z[k] for k in z.files}
   return cache[key]
  rank_checks=[];moment_rows=[];endpoint_count=0;copied_controls=0;copied_scores=0;history_records=0
  for f in FS:
   ref=npz(PAR/'forecasts'/f'fold{f}_scale_mean.npz');fit_mean=float(ref['fit_return_mean'])
   stream={m:npz(PAR/'forecasts'/f'fold{f}_{m}.npz') for m in OLD}
   stream.update({g+'_raw':npz(RAW/'forecasts'/f'fold{f}_{g}_raw.npz') for g in GS})
   stream.update({m:npz(PARENT/'forecasts'/f'fold{f}_{m}.npz') for m in RELIABILITY})
   stream.update({m:npz(OUT/'forecasts'/f'fold{f}_{m}.npz') for m in NEW})
   for m,z in stream.items():
    expected_keys=set(ref)|({'raw_log_variance','persistence96_variance'} if m.endswith('_raw') else set())
    assert set(z)==expected_keys,('forecast schema changed',f,m)
    shared=[k for k in ref if k!='mu' and (k!='variance' or not m.endswith('_raw'))]
    equal_arrays({k:z[k] for k in shared},{k:ref[k] for k in shared},f'{f}/{m}/shared')
    assert z['mu'].shape==ref['mu'].shape
    assert np.array_equal(np.isfinite(z['mu']),ref['inference_mask'])
   mask=ref['score_support'];yy=ref['actual'][mask,0];aa=stream['rolling_anchor']['mu'][mask]
   for m in MEANS:
    measured=scalar_scores(yy,stream[m]['mu'][mask],fit_mean);original=si[f,m]
    compare(measured,{k:original[k] for k in measured},f'{f}/{m}/scores','scalar_rescore')
    comp=scalar_decomp(yy,stream[m]['mu'][mask],aa)
    compare(comp,original['decomposition'],f'{f}/{m}/decomp','scalar_decomposition')
    if m in OLD_MEANS:
     compare({k:original[k] for k in measured},{k:parent_si[f,m][k] for k in measured},f'{f}/{m}/old','unchanged_old_scores');copied_scores+=1
   for cid in CONTROLS:
    target=npz(OUT/'targets'/f'fold{f}_{cid}.npz');old=npz(PARENT/'targets'/f'fold{f}_{cid}.npz')
    equal_arrays(target,old,f'{f}/{cid}/old_target')
    for cost in CS:compare(ri[f,cid][cost],parent_ri[f,cid][cost],f'{f}/{cid}/{cost}','unchanged_old_accounts')
    copied_controls+=1
   hp=OUT/'rolling_traces'/f'fold{f}_shared_history.json';trace=read(hp)
   assert trace['fold']==f and trace['fixed_weights']==[wi[f,g] for g in GS]
   assert trace['forecast_origin_window_calendar_months']==3 and trace['maturity_minutes_inclusive']==375
   for p,h in trace['source_artifact_bindings'].items():assert ancestor[p]==h
   pf=next(x for x in pre['support'] if x['fold']==f)
   membership=[{k:x[k] for k in ('decision_at','history_timestamp_sha256')} for x in trace['decisions']]
   assert membership==pf['history_membership'] and canonical(membership)==pf['history_membership_sha256']
   assert canonical([x['history_count'] for x in trace['decisions']])==pf['history_counts_sha256']
   assert len(trace['decisions'])==int(ref['inference_mask'].sum())
   history_records+=len(trace['decisions'])
   for j,x in zip(np.flatnonzero(ref['inference_mask']),trace['decisions']):
    assert x['reason']=='available' and x['history_count']>=64 and x['minimum_pairs']==64
    assert x['weights']=={g:wi[f,g]['fit']['weight'] for g in GS}
    for m in NEW:assert x['forecasts'][m]==float(stream[m]['mu'][j])
   for g in GS:
    m=g+'_rolling';old=g+'_reliability';w=wi[f,g]['fit']['weight']
    p=stream[m]['mu'][mask];oldp=stream[old]['mu'][mask]
    rank_checks.append({'fold':f,'group':g,'weight':w,'rolling_unique':len(np.unique(p)),
     'rolling_anchor_unique':len(np.unique(aa)),'ranks_equal_to_old_reliability':bool(np.array_equal(ranks(p),ranks(oldp))),
     'ranks_equal_to_rolling_anchor':bool(np.array_equal(ranks(p),ranks(aa))),'rolling_rank_ic':si[f,m]['return_rank_ic']})
    moment_rows.append({'fold':f,'group':g,'fixed_weight':w,'relative_to_rolling_anchor':si[f,m]['decomposition'],
     'mse_minus_old_reliability':D(si[f,m]['return_mse'])-D(si[f,old]['return_mse']),
     'new_weights_or_optimal_slopes_computed':False})
    if w==0:
     assert np.array_equal(stream[m]['mu'],stream['rolling_anchor']['mu'],equal_nan=True)
     for q in RS:
      equal_arrays(npz(OUT/'targets'/f'fold{f}_{m}_{q}.npz'),npz(OUT/'targets'/f'fold{f}_rolling_anchor_{q}.npz'),f'{f}/{g}/zero_endpoint')
      for c in CS:assert ri[f,m+'_'+q][c]==ri[f,'rolling_anchor_'+q][c]
      endpoint_count+=1
  assert copied_controls==128 and copied_scores==72 and history_records==2586
  assert all(v<Decimal('1e-12') for v in maxdiff.values()),plain(maxdiff)
  interpretation={m:{'mse_reduction_vs_zero':1-pred['all'][m]['equal_quarter_mse']/pred['all'][m]['zero_return_mse'],
   'mse_reduction_vs_fit_mean':1-pred['all'][m]['equal_quarter_mse']/pred['all'][m]['fit_mean_return_mse'],
   'relative_to_rolling_anchor_decomposition':pred['all'][m]['relative_to_rolling_anchor_decomposition'],
   'paired_registered_controls':paired['all'][m],
   'joint_quarters_both_costs':{q:econ['all'][m+'_'+q]['joint_positive_quarters_both_costs'] for q in RS}} for m in NEW}
  report={'schema':'independent-rolling-centering-summary-audit-v1','passed':True,
   'scope':'Saved original development artifacts only. Decimal60 aggregation, independent scalar96scores/decompositions and rank checks. No canonical helper, summary, planner or scorer imported. No new forecasts, slopes, policies, fits or later market periods.',
   'source_revision':reg['source_revision'],'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
   'inventory':{'economic_rows':176,'cost_accounts':352,'scores':96,'fixed_weights':16,'new_artifacts':256,
    'ancestral_artifacts':len(ancestor),'regime_counts':counts,'unchanged_old_controls':copied_controls,'unchanged_old_scores':copied_scores,
    'history_trace_records':history_records,'exact_zero_endpoint_policy_matches':endpoint_count},
   'verified_binding_counts':dict(categories),'source_sha256':bindings,
   'binding_scope':'Every file enumerated in Stage14 registered source, direct preflight, ancestral artifact and new artifact manifests was independently rehashed. Unenumerated underlying archives inherit prior source proofs.',
   'numeric_max_absolute_differences':maxdiff,'maximum_difference_locations':locations,
   'economics':econ,'prediction':pred,'paired':paired,'direction':direction,'interpretation':interpretation,
   'rank_checks':rank_checks,'relative_to_rolling_anchor_moments_per_fold':moment_rows,
   'limitations':['Reused8 development quarters and2/4/2 start-regime counts cannot establish high-probability trend invariance.',
    'Updated intercepts use past matured evaluation labels; this is sequential evaluation, not a quarter without label updates.',
    'The moving-anchor decomposition changes the reference relative to Stage13; component changes are not directly comparable across those stages.',
    'Joint return-mean and forecast-centering adaptation does not identify their separate causal effects.',
    'A time-varying intercept may change ranks; a zero slope copies a varying rolling anchor and need not have undefined rank.',
    'Forecast-loss improvements do not imply simultaneous AlphaEx and MaxDDDelta improvement.',
    'History membership hashes are checked against frozen preflight here; independent membership/mean recomputation and general account reconstruction are separate audits.',
    'No new weight fit, window choice, policy, significance test, confidence interval, confirmation data or promotion is introduced.']}
  REPORT.write_text(json.dumps(plain(report),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
  print(json.dumps({'output':str(REPORT),'sha256':sha(REPORT),'passed':True,'inventory':report['inventory'],
   'bindings':dict(categories),'maxdiff':plain(maxdiff),'direction':direction},ensure_ascii=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description='Independent saved Stage14 audit. Run only after root authorizes completed results.')
 parser.add_argument('--execute-saved-audit',action='store_true',required=True)
 args=parser.parse_args()
 audit()
