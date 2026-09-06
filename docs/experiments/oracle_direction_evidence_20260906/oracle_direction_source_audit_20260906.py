"""Stage17 DATA-ONLY immutable source, feature and original-mask audit.
No binary labels, class priors, class weights, classifier coefficients, new
predictions, losses or orders are computed. Existing parity data preparation
reconstructs original continuous labels/features for identity hashes only.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('/tmp/oracle_direction_source_audit_20260906.json')
SOURCE='b44c211dccc38f719b6f893a95c0d1a2d4cbf638'
HASHES={'registration':'cebc21ae289f0e203381032227d171190cc9841fc9fab03082aad874734b4dbd','preflight':'0a0535c5020628ca337b2cacb66a692e8d5bd45b7ebe52855fe1f8285a0a2ffd','results':'9bba65ce12300fd16a4b617e4723b950453554b58070777c3eadd9ee458f3673'}
FOLDS=tuple(range(5,13));GROUPS=('technical','perp_delay0');HALVES=('technical_half','perp_delay0_half');REQUIRED_MASKS=('fit','scale','interval','predict','inference','score')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def masksha(index,mask):return hashlib.sha256(index.asi8.tobytes()+np.asarray(mask,bool).tobytes()).hexdigest()
def matrixsha(a):
 a=np.asarray(a,dtype='<f8',order='C');return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes()).hexdigest()
def arr(p,keys=None):
 with np.load(p,allow_pickle=False) as z:
  if keys is not None:assert set(z.files)==set(keys),('schema',str(p))
  a={k:z[k] for k in z.files}
 assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values())
 return a
def exact(a,b):assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
def main():
 os.chdir(ROOT);sys.path.insert(0,str(ROOT));counts=Counter();verified={};direct={}
 def verify(p,h):
  p=Path(p).resolve()
  if p not in verified:verified[p]=sha(p)
  assert verified[p]==h,('hash',str(p));counts['hash_binding_checks']+=1
 src=Path('codex_outputs/oracle_sign_magnitude_decisions_v1');parity=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
 for k,h in HASHES.items():p=src/(k+'.json');verify(p,h);direct[str(p)]=h
 reg,pre,res=(read(src/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config']
 assert reg['source_revision']==SOURCE and res['registration_sha256']==digest(reg)
 cp=Path('configs/oracle_sign_magnitude_decisions_20260906.yaml');verify(cp,reg['config_sha256']);direct[str(cp)]=reg['config_sha256'];assert yaml.safe_load(cp.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+str(cp)])).hexdigest()==reg['config_sha256']
 assert reg['preflight_sha256']==cfg['preflight_sha256']==HASHES['preflight'] and pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==25
 for p,h in cfg['source_bindings'].items():verify(p,h);assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+p])).hexdigest()==h
 for p,h in pre['direct_source_bindings'].items():verify(p,h);direct[p]=h
 assert res['new_model_fits']==0 and res['new_causal_names']==0 and res['adaptive_causal_names_unchanged']==174 and res['new_hindsight_policy_names']==8
 for k in ('selection_performed','teacher_use_allowed','additional_test_used','high_probability_generalization_established'):assert res[k] is False
 controls=tuple(cfg['control_ids'])+tuple(cfg['new_diagnostic_ids']);assert len(controls)==len(set(controls))==36
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']};scores={(s['fold'],s['mean_id'],s['subset']):s for s in res['scores']}
 assert len(rows)==len(res['rows'])==288 and set(rows)=={(f,c) for f in FOLDS for c in controls}
 assert len(scores)==len(res['scores'])==192 and set(scores)=={(f,m,s) for f in FOLDS for m in cfg['score_means'] for s in cfg['score_subsets']}
 assert sum(r['hindsight_only'] for r in rows.values())==192
 bindings=dict(pre['source_artifact_bindings']);assert len(bindings)==1728;own={}
 newmeans=tuple(cfg['cells'][h][c] for h in HALVES for c in cfg['components'])
 for f in FOLDS:
  p=src/f'fold_{f}.json';fold=read(p);direct[str(p)]=sha(p);assert fold['registration_sha256']==digest(reg)
  for k in ('rows','scores','direction_diagnostics','endpoint_parity'):assert fold[k]==[v for v in res[k] if v['fold']==f]
  assert fold['threshold']==next(t for t in res['thresholds'] if t['fold']==f)
  expected={str(src/k/f'fold{f}_{name}.{ext}') for k,names,ext in [('forecasts',newmeans,'npz'),('targets',controls,'npz'),('traces',cfg['new_diagnostic_ids'],'json'),('thresholds',('fit_q90',),'json')] for name in names}
  assert len(expected)==49 and set(fold['artifact_sha256'])==expected
  for p,h in fold['artifact_sha256'].items():assert p not in bindings or bindings[p]==h;bindings[p]=h;own[p]=h
 assert len(own)==392 and len(bindings)==2120 and len({str(Path(p).resolve()) for p in bindings})==2120
 for p,h in bindings.items():verify(p,h)
 counts.update(parent_sources=25,parent_ancestor_artifacts=1728,parent_own_artifacts=392,total_source_artifacts=2120,parent_control_rows=288,parent_score_rows=192,parent_causal_rows=96,parent_hindsight_rows=192)
 prior=Path('/tmp/oracle_sign_magnitude_audit_20260906.json');verify(prior,'57d0af1a805c623f7cf6edc1256ca1140474dcdee31d896725342bd33cc5d107');direct[str(prior)]=sha(prior)
 from unidream.experiments.oracle_frozen_procedure_parity import prepare as prepare_parity
 pc,dc,fc,bars,allgroups,original,y,masks,pp,*_=prepare_parity(Path(cfg['parity_config']))
 assert fc['data_cutoff']==cfg['data_cutoff']=='2023-04-16T13:45:00Z' and bars.index[-1]<pd.Timestamp(cfg['data_cutoff'])
 assert pp['spot_data_proof']==pre['spot_data_proof'] and pp['um_data_proof']==pre['um_data_proof']
 groups={g:allgroups[g] for g in GROUPS};assert [len(x.columns) for x in groups.values()]==[29,31]
 assert groups['technical'].equals(original['technical']) and groups['perp_delay0'].iloc[:,:29].equals(groups['technical'])
 assert list(groups['perp_delay0'].columns[-2:])==['perp_weighted_flow24','perp_weighted_flow96']
 for g,x in groups.items():assert x.index.equals(bars.index) and x.columns.is_unique
 keys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};records=[];totals=Counter();regimes=Counter();basebindings={}
 for f in FOLDS:
  old=next(v for v in pp['support'] if v['reference_validation_fold']==f);m=masks[f];E=pd.Timestamp(old['evaluation_start']);end=pd.Timestamp(old['evaluation_end']);fitstart=pd.Timestamp(old['fit_start']);S=pd.Timestamp(old['scale_start']);I=pd.Timestamp(old['interval_start'])
  dates={'fit':(fitstart,S),'scale':(S,I),'interval':(I,E),'evaluation':(E,end)}
  for name in m:assert m[name].dtype==bool and masksha(bars.index,m[name])==old['mask_sha256'][name]
  assert int(m['fit'].sum())==(800,1034,1313,1500,1503,1634,1672,1794)[f-5]
  source16=next(v for v in pre['support'] if v['fold']==f)
  assert source16['fit_rows']==int(m['fit'].sum()) and source16['fit_mask_sha256']==old['mask_sha256']['fit'] and source16['fit_return_sha256']==digest(y[m['fit'],0].tolist())
  for name in ('fit','scale','interval'):
   start,boundary=dates[name];assert ((bars.index[m[name]]>=start)&(bars.index[m[name]]<boundary)).all() and (bars.index[m[name]]+pd.Timedelta(minutes=375)<boundary).all()
   assert np.isfinite(y[m[name],0]).all()
  finite={name:{g:int(np.isfinite(x.to_numpy()[m[name]]).all(axis=1).sum()) for g,x in groups.items()} for name in REQUIRED_MASKS}
  assert all(n==int(m[name].sum()) for name,gs in finite.items() for n in gs.values())
  ix=np.asarray((bars.index>=E)&(bars.index<end));index=bars.index[ix];reference=None
  for mean in ('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half'):
   p=parity/'forecasts'/f'fold{f}_{mean}.npz';a=arr(p,keys);assert str(p) in bindings;basebindings[str(p)]=bindings[str(p)]
   exact(a['timestamps'],index.asi8);exact(a['inference_mask'],m['inference'][ix]);exact(a['score_support'],m['score'][ix]);exact(np.isfinite(a['mu']),a['inference_mask']);exact(np.isfinite(a['variance']),a['inference_mask'])
   if reference is None:reference=a
   else:
    for k in keys-{'mu'}:exact(a[k],reference[k])
   if mean in HALVES:counts['original_half_forecasts']+=1
   counts['original_parent_forecast_arrays']+=1
  expected_y=y[ix].copy();expected_y[~m['score'][ix]]=np.nan;exact(reference['actual'],expected_y)
  clock=np.asarray((index.hour%6==0)&(index.minute==0));known=np.isfinite(bars.open.to_numpy()[ix]);inf=reference['inference_mask'];score=reference['score_support'];fallback=clock&known&~inf;missing=clock&~known
  assert not (inf&~known).any() and not (inf&~clock).any() and not (score&~inf).any() and (index[score]+pd.Timedelta(minutes=375)<=end).all()
  for cid in controls:
   r=rows[f,cid];p=src/'targets'/f'fold{f}_{cid}.npz';assert r['targets_sha256']==bindings[str(p)];a=arr(p,{'timestamps','targets'});exact(a['timestamps'],index.asi8);counts['parent_target_calendar_bindings']+=1
   assert r['regime']==source16['regime']
   if cid in cfg['new_diagnostic_ids']:
    p=src/'traces'/f'fold{f}_{cid}.json';assert r['diagnostic_sha256']==bindings[str(p)];counts['parent_new_trace_bindings']+=1
  rawcalbindings={}
  for g in GROUPS:
   p=parity/'calibration'/f'fold{f}_{g}.npz';assert str(p) in bindings;ca=arr(p);cx=np.asarray((bars.index>=S)&(bars.index<E));exact(ca['timestamps'],bars.index[cx].asi8)
   for name in ('scale','interval'):exact(ca[name+'_mask'],m[name][cx])
   rawcalbindings[g]={'path':str(p),'sha256':bindings[str(p)]};counts['old_calibration_arrays']+=1
  record={'fold':f,'calendar':{k:{'start_inclusive':a.isoformat(),'end_exclusive':b.isoformat()} for k,(a,b) in dates.items()},'counts':{k:int(v.sum()) for k,v in m.items()},'mask_sha256':old['mask_sha256'],'finite_feature_rows':finite,
   'feature_selected_sha256':{name:{g:digest(x.to_numpy()[m[name]].tolist()) for g,x in groups.items()} for name in ('fit','predict')},
   'feature_selected_float64le_sha256':{name:{g:matrixsha(x.to_numpy()[m[name]]) for g,x in groups.items()} for name in ('fit','predict')},
   'continuous_return_selected_sha256':{name:digest(y[m[name],0].tolist()) for name in ('fit','scale','interval','score')},
   'label_maturity_last':{name:(bars.index[m[name]][-1]+pd.Timedelta(minutes=375)).isoformat() for name in ('fit','scale','interval','score')},
   'raw_calibration_bindings':rawcalbindings,'fallback_rows':int(fallback.sum()),'missing_current_open_rows':int(missing.sum()),'regime':source16['regime']['trend'],'binary_class_labels_or_priors_computed':False}
  records.append(record);regimes[record['regime']]+=1
  for k in ('inference','score'):totals[k]+=int(m[k].sum())
  totals['fallback']+=int(fallback.sum());totals['missing_current_open']+=int(missing.sum())
 assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2} and regimes=={'bull':2,'bear':4,'sideways':2}
 proposal={'classifier_models_per_fold':4,'classifier_models_across_8folds':32,'learned_mean_families':4,'class_prior_mean_families':4,'missing_rules_per_family':2,'new_causal_names':16,'adaptive_causal_names_before':174,'adaptive_causal_names_after':190,'old_control_policies':36,'old_causal_policies':12,'old_hindsight_policies':24,'total_proposed_policy_names':52,'proposed_economic_rows':416,'proposed_base_stress_accounts':832}
 report={'passed':True,'scope':'Stage17 source/data preflight only; no binary label transforms, class priors/weights, new coefficients, predictions, scores or orders','script_sha256':sha(__file__),'parent_source_revision':SOURCE,'parent_file_sha256':HASHES,'parent_registration_canonical_sha256':digest(reg),'counts':dict(counts),'distinct_hashed_files':len(verified),'source_artifact_inventory_sha256':digest(bindings),'source_artifact_bindings':bindings,'direct_source_bindings':direct,'original_forecast_bindings':basebindings,'spot_data_proof':pre['spot_data_proof'],'um_data_proof':pre['um_data_proof'],'features':{g:list(x.columns) for g,x in groups.items()},'required_finite_masks':list(REQUIRED_MASKS),'scheduled_mask_does_not_require_finite_features':True,'support':records,'totals':dict(totals),'regime_counts':dict(regimes),'regime_gate_pass':False,'proposed_counts':proposal,'limitations':['The classifier objective and probability-to-mean mapping must be fixed separately before new class statistics/fits.','The 24 hindsight control policies remain explicitly labeled and cannot be causal baselines or teachers.','Original full dependency availability remains retrospective; 8 reused quarters and 2/4/2 regimes are not independent confirmation.','Legacy data preparation decodes original raw parquet then restricts semantic features/labels before cutoff; full raw files are hashed.']}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'path':str(OUT),'sha256':sha(OUT),'counts':dict(counts),'totals':dict(totals),'proposed':proposal}))
if __name__=='__main__':main()
