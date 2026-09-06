"""Independent saved-artifact audit: Decimal aggregation and scalar score/rank checks.
No helper imports, model fits, policies, or additional market windows.
"""
import hashlib,json,math
from pathlib import Path
from decimal import Decimal, localcontext
from collections import Counter
import numpy as np
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/oracle_mean_reliability_decisions_v1'
PAR=ROOT/'codex_outputs/oracle_frozen_procedure_parity_v1'
RAW=ROOT/'codex_outputs/oracle_derivative_delay_v1'
REPORT=Path('/tmp/oracle_mean_reliability_summary_audit_20260906.json')
FS=tuple(range(5,13));GS=('technical','perp_delay0');SS=('scale','interval','evaluation')
CS=('base','stress_2x');RS=('utility_risk1','utility_risk1_fallback_bh');STR=('all','bull','bear','sideways')
OLD=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')
NEW=tuple(g+'_reliability' for g in GS);MEANS=OLD+tuple(g+'_raw' for g in GS)+NEW
CONTROLS=('bh','common_robust')+tuple(m+'_'+q for m in OLD for q in RS)
IDS=CONTROLS+tuple(m+'_'+q for m in NEW for q in RS)
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
 n=len(y);d=[float(z)-a for z in p];r=[float(z)-a for z in y];e=[float(z)-float(v) for z,v in zip(y,p)]
 md=math.fsum(d)/n;mr=math.fsum(r)/n
 b=math.fsum(z*z for z in d)/n;c=math.fsum(z*v for z,v in zip(d,r))/n
 vd=math.fsum((z-md)**2 for z in d)/n;cv=math.fsum((z-md)*(v-mr) for z,v in zip(d,r))/n
 mse=math.fsum(z*z for z in e)/n;am=math.fsum(z*z for z in r)/n;loss=mse-am;center=vd-2*cv;drift=md*md-2*md*mr
 return {'n':n,'candidate_mse':mse,'anchor_mse':am,'lossdiff':loss,'mean_d':md,'mean_r':mr,
 'innovation_secondmoment':b,'crossmoment':c,'centered_variance_d':vd,'centered_covariance':cv,
 'centered_component':center,'drift_component':drift,'identityresidual':loss-center-drift}

with localcontext() as ctx:
 ctx.prec=60
 result=read(bind(OUT/'results.json'));reg=read(bind(OUT/'registration.json'));cfg=reg['config']
 pre=read(bind(OUT/'preflight.json',cfg['preflight_sha256']))
 bind(ROOT/'configs/oracle_mean_reliability_decisions_20260906.yaml',reg['config_sha256'],'config')
 for p,h in cfg['source_bindings'].items():bind(p,h,'registered_source')
 bind(cfg['source_prepare_config'],cfg['source_prepare_config_sha256'],'parent_config')
 ancestor=pre['source_artifact_bindings']
 rows,scores,fits=result['rows'],result['scores'],result['fits']
 ri={(x['fold'],x['candidate_id']):x for x in rows};si={(x['fold'],x['segment'],x['mean_id']):x for x in scores};fi={(x['fold'],x['group']):x for x in fits}
 assert len(rows)==len(ri)==128 and set(ri)=={(f,p) for f in FS for p in IDS}
 assert len(scores)==len(si)==216 and set(si)=={(f,s,m) for f in FS for s in SS for m in MEANS}
 assert len(fits)==len(fi)==16 and set(fi)=={(f,g) for f in FS for g in GS}
 regimes={f:ri[f,'bh']['regime'] for f in FS};counts=dict(Counter(x['trend'] for x in regimes.values()))
 assert counts=={'bull':2,'bear':4,'sideways':2}
 assert result['base_models_fitted']==0 and result['calibration_weights_fitted']==16 and result['new_causal_policy_names']==4
 assert not result['selection_performed'] and not result['additional_test_accessed'] and not result['teacher_use_allowed']
 artifacts={}
 for f in FS:
  q=read(bind(OUT/f'fold_{f}.json',cat='fold_manifest'))
  assert q['registration_sha256']==result['registration_sha256'] and len(q['artifact_sha256'])==26
  for key in ('rows','scores','fits'):assert q[key]==[x for x in result[key] if x['fold']==f]
  for p,h in q['artifact_sha256'].items():assert p not in artifacts;artifacts[p]=h;bind(p,h,'new_artifact')
 assert len(artifacts)==208
 for r in rows:assert r['regime']==regimes[r['fold']]
 for r in scores:
  assert r['regime']==regimes[r['fold']] and r['regime_known_at_scored_decisions']==(r['segment']=='evaluation')
  assert r['regime_reference']=='evaluation_quarter_start' and r['scale_fit_in_sample']==(r['segment']=='scale')
  assert r['rows']==si[r['fold'],r['segment'],'scale_mean']['rows']
 def folds(g):return [f for f in FS if g=='all' or regimes[f]['trend']==g]
 econ={};pred={};paired={}
 for g in STR:
  ff=folds(g);econ[g]={p:{'quarters':len(ff),**{c:{k:avg([ri[f,p][c][k] for f in ff]) for k in EK} for c in CS}} for p in IDS};pred[g]={}
  for s in SS:
   pred[g][s]={}
   for m in MEANS:
    xx=[si[f,s,m] for f in ff];n=sum(x['rows'] for x in xx)
    pred[g][s][m]={'quarters':len(ff),'rows':n,'equal_quarter_mse':avg([x['return_mse'] for x in xx]),
     'pooled_row_mse':sum(D(x['return_mse'])*Decimal(x['rows']) for x in xx)/Decimal(n),
     'equal_quarter_mae':avg([x['return_mae'] for x in xx]),'zero_return_mse':avg([x['zero_return_mse'] for x in xx]),
     'fit_mean_return_mse':avg([x['fit_mean_return_mse'] for x in xx]),'mse_minus_zero':avg([D(x['return_mse'])-D(x['zero_return_mse']) for x in xx]),
     'mse_minus_fit_mean':avg([D(x['return_mse'])-D(x['fit_mean_return_mse']) for x in xx]),
     'mean_rank_ic':avg([x['return_rank_ic'] for x in xx]) if all(x['return_rank_ic'] is not None for x in xx) else None,
     'decomposition':{k:avg([x['decomposition'][k] for x in xx]) for k in DK}}
  paired[g]={}
  for family in GS:
   m=family+'_reliability';paired[g][m]={}
   for ref in ('scale_mean',family+'_scaled',family+'_half'):
    pp={}
    for s in SS:
     ds=[D(si[f,s,m]['return_mse'])-D(si[f,s,ref]['return_mse']) for f in ff];ref_loss=avg([si[f,s,ref]['return_mse'] for f in ff])
     pp[s]={'mse_difference':avg(ds),'relative_mse_reduction':-avg(ds)/ref_loss if ref_loss else None,
      'improved_quarters':sum(x<0 for x in ds),'equal_quarters':sum(x==0 for x in ds)}
    paired[g][m][ref]={'prediction':pp,'economics':{q:{c:{k:avg([D(ri[f,m+'_'+q][c][k])-D(ri[f,ref+'_'+q][c][k]) for f in ff]) for k in EK} for c in CS} for q in RS}}
 direction={}
 for family in GS:
  m=family+'_reliability';pp={s:all(pred[g][s][m]['mse_minus_zero']<0 and all(paired[g][m][ref]['prediction'][s]['mse_difference']<0 for ref in paired[g][m]) for g in STR) for s in ('interval','evaluation')}
  for q in RS:
   p=m+'_'+q;direction[p]={'economic_means_all_strata_both_costs':all(econ[g][p][c]['alpha_ex']>0 and econ[g][p][c]['maxdd_delta']<0 for g in STR for c in CS),
    'predictive_mse_vs_zero_scale_full_half_all_strata':pp,'regime_count_gate_pass':False,'high_probability_generalization_established':False}
 for name,x in [('economics',econ),('prediction',pred),('paired',paired),('direction',direction)]:compare(x,result['summary'][name],name,'summary_'+name)
 # Independently score and rank existing arrays, never generate a new forecast/policy.
 cache={}
 def npz(p):
  p=Path(p);key=str(p.relative_to(ROOT))
  if key not in cache:
   if key not in artifacts:bind(p,ancestor[key],'consumed_ancestor')
   with np.load(p,allow_pickle=False) as z:cache[key]={k:z[k] for k in z.files}
  return cache[key]
 rank_checks=[];moment_rows=[];endpoint_count=0
 for f in FS:
  ref=npz(PAR/'forecasts'/f'fold{f}_scale_mean.npz');fit_mean=float(ref['fit_return_mean'])
  ca={g:npz(OUT/'calibration'/f'fold{f}_{g}.npz') for g in GS}
  evals={m:npz(PAR/'forecasts'/f'fold{f}_{m}.npz')['mu'] for m in OLD}
  evals.update({g+'_raw':npz(RAW/'forecasts'/f'fold{f}_{g}_raw.npz')['mu'] for g in GS})
  evals.update({m:npz(OUT/'forecasts'/f'fold{f}_{m}.npz')['mu'] for m in NEW})
  for s in SS:
   yy=ref['actual'] if s=='evaluation' else ca['technical']['actual'];mask=ref['score_support'] if s=='evaluation' else ca['technical'][s+'_mask']
   a=ref['mu'] if s=='evaluation' else ca['technical']['anchor'];av=float(a[mask][0]);assert np.all(a[mask]==av)
   mm=evals.copy() if s=='evaluation' else {'scale_mean':ca['technical']['anchor'],**{g+'_'+suffix:ca[g][suffix] for g in GS for suffix in ('raw','scaled','half','reliability')}}
   for m in MEANS:
    measured=scalar_scores(yy[mask,0],mm[m][mask],fit_mean);original=si[f,s,m]
    compare(measured,{k:original[k] for k in measured},f'{f}/{s}/{m}/scores','scalar_rescore')
    comp=scalar_decomp(yy[mask,0],mm[m][mask],av);compare(comp,original['decomposition'],f'{f}/{s}/{m}/decomp','scalar_decomposition')
   for g in GS:
    full=mm[g+'_scaled'][mask];new=mm[g+'_reliability'][mask];w=fi[f,g]['fit']['weight']
    check={'fold':f,'segment':s,'group':g,'weight':w,'full_unique':len(np.unique(full)),'reliability_unique':len(np.unique(new)),
     'same_ranks':bool(np.array_equal(ranks(full),ranks(new))),'reliability_rank_ic':si[f,s,g+'_reliability']['return_rank_ic']}
    if w>0:assert check['same_ranks']
    else:assert check['reliability_unique']==1 and check['reliability_rank_ic'] is None
    rank_checks.append(check)
    full_d=si[f,s,g+'_scaled']['decomposition'];new_d=si[f,s,g+'_reliability']['decomposition']
    moment_rows.append({'fold':f,'segment':s,'group':g,'fixed_weight':w,'full_crossmoment':full_d['crossmoment'],
      'full_centered_covariance':full_d['centered_covariance'],'centered_component':new_d['centered_component'],
      'drift_component':new_d['drift_component'],'lossdiff_vs_anchor':new_d['lossdiff'],
      'future_optimal_slope_computed':False})
  for g in GS:
   fit=fi[f,g]['fit']; sd=si[f,'scale',g+'_scaled']['decomposition']
   for k in ('innovation_secondmoment','crossmoment','mean_d','mean_r','n'):compare(fit[k],sd[k],f'{f}/{g}/scale_moment/'+k,'weight_scale_moment_binding')
   b,c=fit['innovation_secondmoment'],fit['crossmoment'];expected=0. if b==0 or c<=0 else 1. if c>=b else c/b
   compare(expected,fit['weight'],f'{f}/{g}/registered_weight','saved_weight_algebra')
   if fit['weight'] in (0.,1.):
    m=g+'_reliability';ep='scale_mean' if fit['weight']==0 else g+'_scaled'
    assert np.array_equal(evals[m],evals[ep],equal_nan=True)
    for q in RS:
     with np.load(OUT/'targets'/f'fold{f}_{m}_{q}.npz',allow_pickle=False) as x,np.load(OUT/'targets'/f'fold{f}_{ep}_{q}.npz',allow_pickle=False) as y:assert np.array_equal(x['targets'],y['targets'],equal_nan=True)
     for cst in CS:assert ri[f,m+'_'+q][cst]==ri[f,ep+'_'+q][cst]
     endpoint_count+=1
 assert all(v<Decimal('1e-12') for v in maxdiff.values()),maxdiff
 cases={g:dict(Counter(fi[f,g]['fit']['weight_case'] for f in FS)) for g in GS}
 interpretation={}
 for g in GS:
  m=g+'_reliability';stages={}
  for s in SS:
   pp=pred['all'][s][m];a_loss=pred['all'][s]['scale_mean']['equal_quarter_mse']
   stages[s]={'mse':pp['equal_quarter_mse'],'mse_reduction_vs_scale_mean':1-pp['equal_quarter_mse']/a_loss,
    'mse_reduction_vs_zero':1-pp['equal_quarter_mse']/pp['zero_return_mse'],
    'mse_reduction_vs_fit_mean':1-pp['equal_quarter_mse']/pp['fit_mean_return_mse'],
    'centered_component':pp['decomposition']['centered_component'],'drift_component':pp['decomposition']['drift_component'],
    'anchor_improved_quarters':paired['all'][m]['scale_mean']['prediction'][s]['improved_quarters'],
    'anchor_equal_quarters':paired['all'][m]['scale_mean']['prediction'][s]['equal_quarters'],
    'full_crossmoment_nonpositive_folds':[f for f in FS if si[f,s,g+'_scaled']['decomposition']['crossmoment']<=0]}
  interpretation[g]={'case_counts':cases[g],'weights':[{'fold':f,**fi[f,g]['fit']} for f in FS],'segments':stages,
   'economic_joint_quarter_counts':{q:{c:sum(ri[f,m+'_'+q][c]['alpha_ex']>0 and ri[f,m+'_'+q][c]['maxdd_delta']<0 for f in FS) for c in CS} for q in RS},
   'raw_scaled_evaluation_mse_difference':avg([D(si[f,'evaluation',g+'_scaled']['return_mse'])-D(si[f,'evaluation',g+'_raw']['return_mse']) for f in FS])}
 report={'schema':'independent-mean-reliability-summary-audit-v1','passed':True,'scope':'Saved original development artifacts only; independent Decimal60 aggregation and scalar 216-score/decomposition/rank verification; no helper imports, new fits, policies, or additional market periods.',
  'source_revision':reg['source_revision'],'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
  'inventory':{'economic_rows':128,'cost_accounts':256,'scores':216,'scale_weights':16,'new_artifacts':208,'regime_counts':counts,'exact_endpoint_policy_matches':endpoint_count},
  'verified_binding_counts':dict(categories),'source_sha256':bindings,'ancestor_scope':'Only consumed ancestor NPZs were rehashed; all1328 ancestral bindings are transitively bound through the preflight, not independently reverified.',
  'numeric_max_absolute_differences':maxdiff,'maximum_difference_locations':locations,'economics':econ,'prediction':pred,'paired':paired,'direction':direction,
  'calibration_interpretation':interpretation,'rank_checks':rank_checks,'centered_drift_per_fold':moment_rows,
  'limitations':['S-fit loss improvement is mechanical and is not future skill.','S/I stratification uses subsequent E-start regimes and is retrospective.',
   'The algebraic drift component is relative to the frozen scale anchor, not a causal estimate of stale-bias damage.',
   'The drift term also depends on forecast-centroid and outcome-mean movement; no feature or model causal attribution is identified.',
   'Negative centered component can coexist with positive total loss difference; averaged components are not an independent skill test.',
   'All4 economic direction flags are descriptive means on repeatedly reused2/4/2 quarters. Predictive and regime-count requirements remain separate.',
   'No I/E slope, additional model, new policy, inference interval, p-value or promotion was calculated.']}
 REPORT.write_text(json.dumps(plain(report),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps({'output':str(REPORT),'sha256':sha(REPORT),'passed':True,'bindings':dict(categories),'maxdiff':plain(maxdiff),
  'weight_cases':cases,'rank_comparisons':len(rank_checks),'positive_weight_same_rank':sum(z['weight']>0 and z['same_ranks'] for z in rank_checks),
  'zero_weight_constant_comparisons':sum(z['weight']==0 for z in rank_checks),'endpoint_policy_matches':endpoint_count,'interpretation':plain(interpretation),'direction':direction},ensure_ascii=False))
