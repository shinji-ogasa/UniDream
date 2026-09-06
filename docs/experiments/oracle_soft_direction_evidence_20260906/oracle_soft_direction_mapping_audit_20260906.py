"""Independent Stage19 scalar mapping audit; real mode requires explicit root GO.

No project helper, estimator fit/predict, loss scoring or policy replay is called.
Default mode reads no market or experiment artifacts and uses synthetic inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('codex_outputs/oracle_soft_direction_decisions_v1')
DIRECTION=Path('codex_outputs/oracle_direction_decisions_v1')
REGULARIZED=Path('codex_outputs/oracle_regularized_direction_decisions_v1')
CONFIG=Path('configs/oracle_soft_direction_decisions_20260906.yaml')
REPORT=Path('/tmp/oracle_soft_direction_mapping_audit_20260906.json')
FOLDS=tuple(range(5,13));GROUPS=('technical','perp_delay0');SEGMENTS=('interval','evaluation')
KINDS=('soft','mapped_prior','fit_mean','zero')
PRIOR_ATOL=1e-14;PRIOR_RTOL=1e-12
TOLERANCES={'mapped_mean_absolute_difference':0.,'preserved_source_fields':'exact including signed zero',
 'mapping_diagnostic_scalar_absolute_difference':0.,'prior_identity_absolute':PRIOR_ATOL,
 'prior_identity_relative':PRIOR_RTOL,'new_tolerances_selected_from_outcomes':False}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def arrays(path):
 with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def exact(a,b,name):
 a,b=np.asarray(a),np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),(name,'not exact')
 if a.dtype.kind=='f':assert np.array_equal(np.signbit(a[a==0]),np.signbit(b[b==0])),(name,'signed zero')
def matrixsha(a):
 a=np.asarray(a,dtype='<f8',order='C')
 return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes()).hexdigest()
def masksha(index,mask):return hashlib.sha256(index.asi8.tobytes()+np.asarray(mask,bool).tobytes()).hexdigest()
def scalar_sign(v):return 1. if v>0 else -1. if v<0 else 0.


def scalar_maps(q,mask,amplitude,prior,stored_mean):
 assert mask.dtype==bool and mask.ndim==1 and len(mask) and mask.any() and q.shape==mask.shape
 assert math.isfinite(amplitude) and amplitude>0 and math.isfinite(prior) and 0<=prior<=1 and math.isfinite(stored_mean)
 mapped_prior=float(amplitude)*(2.0*float(prior)-1.0)
 residual=mapped_prior-stored_mean;limit=PRIOR_ATOL+PRIOR_RTOL*abs(stored_mean)
 assert math.isfinite(residual) and abs(residual)<=limit
 means={k:np.full(len(mask),np.nan) for k in KINDS}
 for i in np.flatnonzero(mask):
  v=float(q[i]);assert math.isfinite(v) and 0<=v<=1
  soft=float(amplitude)*(2.0*v-1.0)
  assert math.isfinite(soft) and abs(soft)<=amplitude
  means['soft'][i]=soft;means['mapped_prior'][i]=mapped_prior
  means['fit_mean'][i]=stored_mean;means['zero'][i]=0.
 d={'schema':'oracle-soft-direction-mapping-v1',
  'formula':'fit_abs_return_mean * (2.0 * saved_probability - 1.0)',
  'prior_formula':'fit_abs_return_mean * (2.0 * saved_weighted_prior_probability - 1.0)',
  'fit_abs_return_mean':amplitude,'saved_weighted_prior_probability':prior,
  'fit_return_mean':stored_mean,'mapped_prior':mapped_prior,
  'prior_identity_signed_difference':residual,'prior_identity_absolute_difference':abs(residual),
  'prior_identity_tolerance':limit,'prior_identity_atol':PRIOR_ATOL,'prior_identity_rtol':PRIOR_RTOL,
  'prior_identity_passed':True,'total_rows':len(mask),'inference_rows':int(mask.sum()),
  'noninference_rows':int((~mask).sum()),'probability_half_rows':sum(float(v)==.5 for v in q[mask]),
  'model_fits':0,'calibration_fits':0,'probabilities_recomputed':False,
  'future_outcomes_or_score_support_used':False,'calendar_or_receipt_causality_verified':False,
  'saved_statistics_provenance_verified':False}
 return means,d


def synthetic():
 q=np.asarray([0.,.25,.5,.75,1.,object()],dtype=object);mask=np.asarray([1,1,1,1,1,0],dtype=bool)
 means,d=scalar_maps(q,mask,.04,.625,.01)
 exact(means['soft'][:5],np.asarray([-.04,-.02,0.,.02,.04]),'hand mapping')
 assert np.isnan(means['soft'][-1]) and d['probability_half_rows']==1
 a=.5;raw=.6;prior=float(np.nextafter(raw,1.));stored=a*(2*raw-1)
 means,d=scalar_maps(np.array([prior]),np.ones(1,bool),a,prior,stored)
 assert means['mapped_prior'][0]!=means['fit_mean'][0] and d['prior_identity_passed']
 for amplitude in (np.finfo(float).max,np.nextafter(0.,1.)):
  m,_=scalar_maps(np.array([0.,.5,1.]),np.ones(3,bool),amplitude,.5,-0.)
  assert m['soft'][0]==-amplitude and m['soft'][2]==amplitude and m['soft'][1]==0
  assert np.signbit(m['fit_mean']).all()
 print(json.dumps({'synthetic_passed':True,'real_artifacts_or_mappings_read_or_computed':False,
                   'predeclared_tolerances':TOLERANCES}))


def real(revision,result_sha):
 os.chdir(ROOT);verified={};counts=Counter();maximum=0.;mapping_summaries=[];diagnostic_rows=[];runtime={}
 def verify(path,expected):
  key=str(Path(path).resolve())
  if key not in verified:verified[key]=sha(path)
  assert verified[key]==expected,('SHA',str(path));counts['hash_binding_checks']+=1
 verify(OUT/'results.json',result_sha)
 reg,pre,result=[read(OUT/(name+'.json')) for name in ('registration','preflight','results')]
 cfg=yaml.safe_load(CONFIG.read_text())
 assert reg['source_revision']==revision and reg['config']==cfg and result['registration_sha256']==digest(reg)
 verify(CONFIG,reg['config_sha256']);verify(OUT/'preflight.json',cfg['preflight_sha256'])
 assert reg['preflight_sha256']==cfg['preflight_sha256']
 assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert hashlib.sha256(subprocess.check_output(['git','show',revision+':'+str(CONFIG)])).hexdigest()==reg['config_sha256']
 assert (np.__version__,pd.__version__,version('scikit-learn'))==('2.2.6','2.3.3','1.8.0')
 assert cfg['prior_identity_absolute_tolerance']==PRIOR_ATOL and cfg['prior_identity_relative_tolerance']==PRIOR_RTOL
 assert cfg['development_folds']==list(FOLDS) and cfg['groups']==list(GROUPS)
 assert len(cfg['source_bindings'])==33 and len(pre['source_artifact_bindings'])==3488
 assert len({str(Path(p).resolve()) for p in pre['source_artifact_bindings']})==3488
 for p,h in cfg['source_bindings'].items():
  verify(p,h);assert hashlib.sha256(subprocess.check_output(['git','show',revision+':'+p])).hexdigest()==h
 for section in ('direct_source_bindings','source_artifact_bindings'):
  for p,h in pre[section].items():verify(p,h)
 ancestors=pre['source_artifact_bindings']
 def source_arrays(path):
  assert str(path) in ancestors
  return arrays(path)
 def source_json(path):
  assert str(path) in ancestors
  return read(path)
 assert cfg['data_cutoff']=='2023-04-16T13:45:00Z'
 # Reconstruct the inherited calendar only, without market-price decoding.
 index=pd.date_range('2018-01-01T00:00:00Z','2023-04-16T13:30:00Z',freq='15min')
 records={v['fold']:v for v in result['mapping_records']}
 assert len(records)==len(result['mapping_records'])==8 and len(result['mapping_diagnostics'])==64
 for f in FOLDS:
  fold=read(OUT/f'fold_{f}.json');assert fold['registration_sha256']==digest(reg)
  assert len(fold['artifact_sha256'])==121
  for p,h in fold['artifact_sha256'].items():
   assert p not in runtime;verify(p,h);runtime[p]=h
  record=read(OUT/'provenance'/f'fold{f}_mapping.json');assert record==records[f]
  support=next(v for v in pre['support'] if v['fold']==f)
  scalars=record['saved_T_scalars'];assert scalars==support['saved_mapping_scalars']
  assert record['frozen_input_bindings']==support['saved_probability_inputs']
  assert record['new_fits']==record['new_unique_priors']==0
  assert record['probability_arrays_unchanged'] and record['caller_saved_input_provenance_and_calendar_verified']
  assert record['mapping_formula']==cfg['surrogate_mean']
  old=source_json(DIRECTION/'provenance'/f'fold{f}_fit.json')
  pack=source_arrays(DIRECTION/'fit_data'/f'fold{f}_training.npz')
  y=pack['returns'];n=len(y);assert n>=512 and np.isfinite(y).all()
  amplitude=math.fsum(abs(float(v))/n for v in y);stored_mean=float(np.mean(y))
  exact(pack['binary_labels'],(y>0).astype(np.int64),'frozen T labels')
  weight=np.asarray([abs(float(v))/amplitude for v in y])
  exact(weight,pack['weights_magnitude'],'frozen T magnitude weights')
  raw_prior=math.fsum(float(v) for v in weight[y>0])/math.fsum(float(v) for v in weight)
  assert amplitude==old['fit_abs_return_mean']==scalars['fit_abs_return_mean']
  assert stored_mean==old['fit_return_mean']==scalars['fit_return_mean']
  assert raw_prior==old['fit_priors']['magnitude']==scalars['fit_statistical_magnitude_prior']
  assert matrixsha(y)==old['fit_provenance']['fit_return_sha256']
  assert matrixsha(weight)==old['fit_provenance']['sample_weights']['magnitude']['weight_sha256']
  masks={k:np.zeros(len(index),bool) for k in ('fit','predict','scale','interval','inference','score')}
  for name,tkey in [('fit','timestamps'),('predict','predict_timestamps')]:
   positions=pack[name+'_positions'];assert len(positions) and np.all(np.diff(positions)>0)
   exact(index.asi8[positions],pack[tkey],name+' timestamps');masks[name][positions]=True
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5))
  assert (index[masks['fit']]+pd.Timedelta(minutes=375)<E-pd.DateOffset(months=6)).all()
  prior_values={};prior_bindings={}
  for group in GROUPS:
   values=[]
   for segment,kind,mkey in [('interval','calibration','mapped_inference_mask'),('evaluation','forecasts','inference_mask')]:
    path=DIRECTION/kind/f'fold{f}_{group}_magnitude_prior_direction.npz';pa=source_arrays(path)
    selected=pa[mkey];assert selected.dtype==bool and selected.any()
    qq=pa['probability'][selected];assert np.isfinite(qq).all() and np.all(qq==qq[0]) and 0<float(qq[0])<1
    values.append(float(qq[0]));binding={'path':str(path),'sha256':ancestors[str(path)]}
    assert scalars['prior_paths'][group][segment]==binding
    assert record['frozen_input_bindings'][group+'/'+segment+'/prior']==binding
    prior_bindings[group+'/'+segment]=binding
   assert values[0]==values[1]==scalars['prior_probability'][group];prior_values[group]=values[0]
  assert len(set(prior_values.values()))==1
  expected_means=set(cfg['new_mean_ids']);assert len(expected_means)==10 and set(record['mean_records'])==expected_means
  for mean,mp in cfg['mapping'].items():
   mr=record['mean_records'][mean]
   for k,v in mp.items():assert mr[k]==v
   assert mr['source_probability_and_logit_fields_are_preserved_source_evidence']
   group=mp['group'];mid=mp['source_classifier'];prior=prior_values[group]
   parentroot=REGULARIZED if mid.endswith('_l2unit') else DIRECTION
   for segment,kind,mkey in [('interval','calibration','mapped_inference_mask'),('evaluation','forecasts','inference_mask')]:
    path=parentroot/kind/f"fold{f}_{mp['source_mean']}.npz";source=source_arrays(path)
    assert record['frozen_input_bindings'][mid+'/'+segment]=={'path':str(path),'sha256':ancestors[str(path)]}
    mapped=arrays(OUT/kind/f'fold{f}_{mean}.npz')
    assert set(mapped)==set(source)
    for key in source:
     if key!='mu':exact(mapped[key],source[key],kind+' unchanged '+key)
    times=pd.DatetimeIndex(pd.to_datetime(source['timestamps'],utc=True));local=index.get_indexer(times)
    assert (local>=0).all() and times.is_unique and times.is_monotonic_increasing
    if segment=='evaluation':
     assert times[0]>=E and times[-1]<E+pd.DateOffset(months=3)
     maskfields={'inference':'inference_mask','score':'score_support'}
    else:
     assert times[0]>=E-pd.DateOffset(months=6) and times[-1]<E
     maskfields={'scale':'scale_mask','interval':'interval_mask'}
    for name,key in maskfields.items():
     v=source[key];assert v.dtype==bool and v.shape==(len(times),)
     candidate=np.zeros(len(index),bool);candidate[local]=v
     if masks[name].any():exact(candidate,masks[name],name+' common mask')
     masks[name]=candidate
    inference=source[mkey];assert inference.dtype==bool and inference.shape==(len(times),) and inference.any()
    q=source['probability'];assert q.dtype==np.float64 and q.shape==inference.shape
    calculated,d=scalar_maps(q,inference,amplitude,prior,stored_mean)
    exact(mapped['mu'],calculated[mp['kind']],'scalar '+mean+'/'+segment)
    delta=float(np.max(np.abs(mapped['mu'][inference]-calculated[mp['kind']][inference])));maximum=max(maximum,delta)
    assert mr['segment_diagnostics'][segment]==d
    assert float(source['fit_return_mean'])==stored_mean
    if segment=='interval':assert np.isnan(mapped['mu'][times<E-pd.DateOffset(months=3)]).all()
    counts['new_prediction_npz']+=1;counts['helper_diagnostic_records']+=1
    counts['mapped_'+mp['kind']+'_rows']+=int(inference.sum())
    if mp['kind']=='soft':
     ii=np.flatnonzero(inference);nrows=len(ii);z=source['logit'];mu=calculated['soft']
     assert np.isfinite(z[inference]).all()
     expected={'fold':f,'segment':segment,'mean_id':mean,'source_classifier':mid,'regime':fold['rows'][0]['regime'],'rows':nrows,
      'uses_all_inference_not_score_support':True,'probability_half_rows':sum(float(q[i])==.5 for i in ii),
      'probability_zero_rows':sum(float(q[i])==0 for i in ii),'probability_one_rows':sum(float(q[i])==1 for i in ii),
      'source_zero_logit_rows':sum(float(z[i])==0 for i in ii),'mapped_zero_mean_rows':sum(float(mu[i])==0 for i in ii),
      'probability_direction_vs_logit_disagreements':sum(scalar_sign(2.*float(q[i])-1.)!=scalar_sign(float(z[i])) for i in ii),
      'mapped_direction_vs_logit_disagreements':sum(scalar_sign(float(mu[i]))!=scalar_sign(float(z[i])) for i in ii),
      'mean_abs_new_mu':math.fsum(abs(float(mu[i]))/nrows for i in ii),
      'mean_abs_hard_mu':math.fsum(abs(float(source['mu'][i]))/nrows for i in ii),
      'mean_abs_parent_mu':math.fsum(abs(float(source['parent_mu'][i]))/nrows for i in ii),
      'new_abs_mu_greater_than_hard_rows':sum(abs(float(mu[i]))>abs(float(source['mu'][i])) for i in ii),
      'new_abs_mu_equal_to_hard_rows':sum(abs(float(mu[i]))==abs(float(source['mu'][i])) for i in ii)}
     selection=lambda row:(row['fold'],row['segment'],row['mean_id'])==(f,segment,mean)
     assert [v for v in result['mapping_diagnostics'] if selection(v)]==[expected]
     assert [v for v in fold['mapping_diagnostics'] if selection(v)]==[expected]
     score=source['score_support'] if segment=='evaluation' else source['interval_mask']
     assert not np.any(score&~inference)
     diagnostic_rows.append({**expected,'unscored_inference_rows':int((inference&~score).sum()),
      'inference_timestamp_sha256':hashlib.sha256(times.asi8[inference].tobytes()).hexdigest()})
     counts['learned_mapping_diagnostics']+=1
  for name,mask in masks.items():
   assert masksha(index,mask)==support['mask_sha256'][name]
   assert int(mask.sum())==support['counts'][name]
   counts['common_mask_bindings']+=1
  prior=next(iter(prior_values.values()));mapped_prior=amplitude*(2.*prior-1.)
  mapping_summaries.append({'fold':f,'T_rows':len(y),'fit_abs_return_mean':amplitude,'fit_return_mean':stored_mean,
   'fit_statistical_magnitude_prior':raw_prior,'saved_prior_probability':prior,'saved_prior_minus_statistical_prior':prior-raw_prior,
   'mapped_prior':mapped_prior,'mapped_prior_minus_fit_mean':mapped_prior-stored_mean,
   'identity_tolerance':PRIOR_ATOL+PRIOR_RTOL*abs(stored_mean),'prior_bindings':prior_bindings})
  counts['mapping_records']+=1
 assert len(runtime)==968 and counts['new_prediction_npz']==160 and counts['learned_mapping_diagnostics']==64
 assert counts['mapping_records']==8 and counts['common_mask_bindings']==48
 assert result['new_model_fits']==result['new_unique_priors']==result['risk_model_or_calibration_fits']==0
 assert result['probability_predictions_and_scores_unchanged'] and not result['new_probability_accuracy_improvement']
 assert not result['selection_performed'] and not result['additional_test_used_for_modeling_or_scoring']
 log=OUT/'run.log';text=log.read_text();warnings=Counter()
 for line in text.splitlines():
  if 'Warning:' in line:warnings[line.split('Warning:',1)[1].strip()]+=1
 events=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
 assert [v['fold'] for v in events if v.get('event')=='fold_complete']==list(FOLDS)
 report={'passed':True,'scope':'Independent scalar mapping and all fixed constants, exact source fields/support, T-stat/prior identity, source bindings and registered mechanism records; no estimator or policy execution',
  'script_sha256':sha(__file__),'registered_revision':revision,'tolerances':TOLERANCES,'counts':dict(counts),
  'maximum_absolute_mapping_difference':maximum,'mapping_scalars':mapping_summaries,'mapping_diagnostics':diagnostic_rows,
  'source_bindings':cfg['source_bindings'],'ancestor_artifacts':3488,'runtime_artifacts':968,'distinct_hashed_files':len(verified),
  'ancestor_inventory_sha256':digest(ancestors),'runtime_inventory_sha256':digest(runtime),
  'direct_file_bindings':{str(p):sha(p) for p in (CONFIG,OUT/'registration.json',OUT/'preflight.json',OUT/'results.json')},
  'runtime_log':{'path':str(log),'sha256':sha(log),'warning_messages_and_counts':dict(warnings),'warning_cause_established':False},
  'limitations':['This audit does not recompute classification/return scores or economic accounts; those need separate checks.',
   'Unchanged probability values imply no new probability estimate, not useful conditional return information.',
   'A_T is a frozen unconditional magnitude approximation, not a validated conditional magnitude forecast.',
   'All reused development masks and archive receipt-time limitations remain.']}
 REPORT.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'path':str(REPORT),'sha256':sha(REPORT),'passed':True,'counts':dict(counts),
  'maximum_absolute_mapping_difference':maximum,'distinct_hashed_files':len(verified)}))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--real-after-root-go',action='store_true')
 p.add_argument('--registered-revision');p.add_argument('--expected-results-sha256');args=p.parse_args()
 if args.real_after_root_go:
  if not args.registered_revision or not args.expected_results_sha256:p.error('Root frozen revision and completed result SHA are required')
  real(args.registered_revision,args.expected_results_sha256)
 else:synthetic()
