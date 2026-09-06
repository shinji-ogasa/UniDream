"""Stage16 DATA-ONLY source audit: no sign/magnitude forecasts, losses or orders.
Verifies completed Stage12 and its immutable ancestor chain, then counts masks
from existing saved arrays. Only the frozen data-only parity preparation is called;
no model fitter, substitution or policy helper is called.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('/tmp/oracle_sign_magnitude_source_audit_20260906.json')
SOURCE='d3b25734a34915049a327256bd9f99cd9aea8336'
HASHES={'registration':'8e25d7743e49e9df6269cb898808ee1fde691a52d56003c69b52ce9497560fc1','preflight':'ae66338b39253f88e536729948b96b9eb57abe9ba409cbfc074f339be688f4e7','results':'f5597dee653a45ee612766111da578a868969a51211f923493763b4059a18ac7'}
FOLDS=tuple(range(5,13));HALVES=('technical_half','perp_delay0_half');RULES=('utility_risk1','utility_risk1_fallback_bh')
MEANS=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')
CAUSAL=('bh','common_robust')+tuple(m+'_'+r for m in MEANS for r in RULES)
HYBRIDS=tuple(m+'_oracle_'+s for m in HALVES for s in ('return','realized_risk','both'))
RL=tuple(f'matched_rl_beam32_{rule}_risk{risk}' for rule in ('hold','fallback_bh') for risk in (0,1))
HINDSIGHT=tuple(m+'_'+r for m in HYBRIDS for r in RULES)+RL

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array(p,keys=None):
 with np.load(p,allow_pickle=False) as z:
  if keys is not None:assert set(z.files)==keys,('keys',str(p))
  a={k:z[k] for k in z.files}
 assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values())
 return a
def exact(a,b):assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
def masksha(index,mask):return hashlib.sha256(index.asi8.tobytes()+np.asarray(mask,bool).tobytes()).hexdigest()
def main():
 os.chdir(ROOT);counts=Counter();verified={};direct={}
 def verify(p,h):
  p=Path(p).resolve()
  if p not in verified:verified[p]=sha(p)
  assert verified[p]==h,('hash',str(p));counts['hash_binding_checks']+=1
 src=Path('codex_outputs/oracle_information_decomposition_v1');parity=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
 for k,h in HASHES.items():p=src/(k+'.json');verify(p,h);direct[str(p)]=h
 reg,pre,res=(read(src/(k+'.json')) for k in ('registration','preflight','results'));cfg=reg['config']
 assert reg['source_revision']==SOURCE and res['registration_sha256']==digest(reg)
 cp=Path('configs/oracle_information_decomposition_20260906.yaml');verify(cp,reg['config_sha256']);assert yaml.safe_load(cp.read_text())==cfg
 assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+str(cp)])).hexdigest()==reg['config_sha256']
 direct[str(cp)]=reg['config_sha256']
 assert reg['preflight_sha256']==cfg['preflight_sha256']==HASHES['preflight']
 assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert reg['source_bindings']==cfg['source_bindings']==pre['source_bindings']
 for p,h in cfg['source_bindings'].items():
  verify(p,h);assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+p])).hexdigest()==h;direct[p]=h
 assert cfg['source_manifest_bindings']==pre['source_manifest_bindings']
 for p,h in cfg['source_manifest_bindings'].items():verify(p,h);direct[p]=h
 verify(cfg['parity_config'],cfg['parity_config_sha256']);direct[cfg['parity_config']]=cfg['parity_config_sha256']
 assert cfg['half_means']==list(HALVES) and cfg['rules']==list(RULES) and cfg['control_ids']==list(CAUSAL) and cfg['diagnostic_ids']==list(HINDSIGHT)
 assert cfg['data_cutoff']=='2023-04-16T13:45:00Z' and cfg['replacement_support']=='existing_saved_score_support_only_keep_learned_elsewhere'
 assert res['new_models_fitted']==0 and res['new_causal_candidates']==0 and res['hindsight_only_diagnostics']==16
 for k in ('selection_performed','test_periods_used','teacher_use_allowed'):assert res[k] is False
 bindings=dict(pre['source_artifact_bindings']);assert len(bindings)==1328;own={}
 rows={(r['fold'],r['candidate_id']):r for r in res['rows']}
 assert len(rows)==len(res['rows'])==224 and set(rows)=={(f,c) for f in FOLDS for c in CAUSAL+HINDSIGHT}
 for f in FOLDS:
  p=src/f'fold_{f}.json';doc=read(p);direct[str(p)]=sha(p)
  assert doc['registration_sha256']==digest(reg) and doc['rows']==[v for v in res['rows'] if v['fold']==f]
  expected={str(src/k/f'fold{f}_{name}.{ext}') for k,names,ext in [('forecasts',HYBRIDS,'npz'),('targets',CAUSAL+HINDSIGHT,'npz'),('traces',HINDSIGHT,'json')] for name in names}
  assert len(doc['artifact_sha256'])==50 and set(doc['artifact_sha256'])==expected
  for p,h in doc['artifact_sha256'].items():
   assert p not in bindings or bindings[p]==h
   bindings[p]=h;own[p]=h
 assert len(own)==400 and len(bindings)==1728 and len({str(Path(p).resolve()) for p in bindings})==1728
 for p,h in bindings.items():verify(p,h)
 counts.update(ancestor_artifacts=1328,parent_own_artifacts=400,total_source_artifacts=1728,parent_causal_rows=96,parent_hindsight_rows=128)
 # Pin the already completed independent Stage12 audit; no old economic replay here.
 prior=Path('/tmp/oracle_information_decomposition_audit_20260906.json');verify(prior,'f3de0fa4d69da2e615a839bfb90ffc9d9852a42e5a0a775d4d5b9f1747b6c343');direct[str(prior)]=sha(prior)
 fc=yaml.safe_load(Path('configs/oracle_frontier_20260905.yaml').read_text());sp=Path(fc['data_path']);side=read(sp.with_suffix('.sha256.json'))
 for p,k in [(sp,'artifact_sha256'),(sp.with_suffix('.sha256.json'),'sidecar_sha256'),(side['availability_path'],'availability_sha256'),(side['source_ledger_path'],'ledger_sha256')]:verify(p,pre['spot_data_proof'][k]);direct[str(p)]=pre['spot_data_proof'][k]
 um=pre['um_data_proof'];up=Path(um['data_path'])
 for p,k in [(up,'data_sha256'),(up.with_suffix('.sha256.json'),'sidecar_sha256'),(um['availability_path'],'availability_sha256'),(um['source_ledger_path'],'source_ledger_sha256'),(um['registration_path'],'registration_sha256')]:verify(p,um[k]);direct[str(p)]=um[k]
 sys.path.insert(0,str(ROOT))
 from unidream.experiments.oracle_frozen_procedure_parity import prepare as prepare_parity
 pc,dc,pfc,pbars,_,_,old_y,old_masks,old_pre,*_=prepare_parity(Path(cfg['parity_config']))
 assert pfc==fc and np.array_equal(pbars.index.asi8,pd.date_range(pbars.index[0],pbars.index[-1],freq='15min').asi8)
 cut=pd.Timestamp(cfg['data_cutoff']);bars=pd.read_parquet(sp,columns=['open'],filters=[('bar_open_ts','<',cut)]);assert bars.index[-1]<cut
 keys={'timestamps','mu','variance','actual','score_support','inference_mask','fit_return_mean'};records=[];totals=Counter();regimes=Counter();forecast_bindings={}
 for f in FOLDS:
  E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);ix=pd.date_range(E,end,freq='15min',inclusive='left');ref=None
  fit=old_masks[f]['fit'];fit_returns=old_y[fit,0]
  expectedfit=(800,1034,1313,1500,1503,1634,1672,1794)[f-5]
  assert fit.dtype==bool and int(fit.sum())==expectedfit and np.isfinite(fit_returns).all()
  assert (pbars.index[fit]+pd.Timedelta(minutes=375)<E-pd.DateOffset(months=6)).all()
  fit_record={'fit_rows':expectedfit,'fit_mask_sha256':masksha(pbars.index,fit),'selected_fit_returns_sha256':digest(fit_returns.tolist()),'fit_origin_start_inclusive':(E-pd.DateOffset(months=24)).isoformat(),'fit_origin_end_exclusive':(E-pd.DateOffset(months=6)).isoformat(),'last_selected_label_maturity':(pbars.index[fit][-1]+pd.Timedelta(minutes=375)).isoformat(),'threshold_computed':False}
  assert fit_record['fit_mask_sha256']==next(v for v in old_pre['support'] if v['reference_validation_fold']==f)['mask_sha256']['fit']
  for m in MEANS:
   p=parity/'forecasts'/f'fold{f}_{m}.npz';assert str(p) in bindings;a=array(p,keys);forecast_bindings[str(p)]=bindings[str(p)]
   exact(a['timestamps'],ix.asi8)
   if ref is not None:
    for k in keys-{'mu'}:exact(a[k],ref[k])
   else:ref=a
   inf,score=a['inference_mask'],a['score_support'];assert inf.dtype==bool and score.dtype==bool and not np.any(score&~inf)
   exact(np.isfinite(a['mu']),inf);exact(np.isfinite(a['variance']),inf)
   assert np.isfinite(a['actual'][score]).all() and np.isnan(a['actual'][~score]).all()
   if m in HALVES:counts['base_half_forecast_arrays']+=1
   counts['all_five_parent_forecasts']+=1
  inf,score=ref['inference_mask'],ref['score_support'];clock=np.asarray((ix.hour%6==0)&(ix.minute==0));opening=bars.open.reindex(ix).to_numpy();known=np.isfinite(opening)&(opening>0)
  assert not np.any(inf&~clock) and not np.any(inf&~known) and (ix[score]+pd.Timedelta(minutes=375)<=end).all()
  scheduled=int(clock.sum());fallback=clock&known&~inf;missing=clock&~known;old=next(v for v in pre['support'] if v['fold']==f)
  assert old['inference_rows']==int(inf.sum()) and old['replacement_rows']==int(score.sum()) and old['learned_remainder_rows']==int((inf&~score).sum()) and old['current_open_missing_inference_rows']==0
  for m in HYBRIDS:
   p=src/'forecasts'/f'fold{f}_{m}.npz';a=array(p,keys)
   for k in keys-{'mu','variance'}:exact(a[k],ref[k])
   exact(np.isfinite(a['mu']),inf);exact(np.isfinite(a['variance']),inf);counts['old_hindsight_forecast_supports']+=1
  for cid in CAUSAL+HINDSIGHT:
   r=rows[f,cid];p=src/'targets'/f'fold{f}_{cid}.npz';assert r['targets_sha256']==bindings[str(p)]
   a=array(p,{'timestamps','targets'});exact(a['timestamps'],ix.asi8);assert len(a['targets'])==len(ix)
   assert r['hindsight_only'] is (cid in HINDSIGHT) and r['regime']==old['regime']
   if cid in HINDSIGHT:
    t=src/'traces'/f'fold{f}_{cid}.json';assert r['diagnostic_sha256']==bindings[str(t)];counts['old_hindsight_trace_bindings']+=1
   else:assert 'diagnostic_sha256' not in r
   counts['old_target_calendar_bindings']+=1
  record={'fold':f,'evaluation_start':E.isoformat(),'evaluation_end_exclusive':end.isoformat(),'full_15m_rows':len(ix),'scheduled_rows':scheduled,'inference_rows':int(inf.sum()),'replacement_rows':int(score.sum()),'learned_remainder_rows':int((inf&~score).sum()),'fallback_rows':int(fallback.sum()),'missing_current_open_rows':int(missing.sum()),'current_open_missing_inference_rows':0,'regime':old['regime']['trend'],'mask_sha256':{k:masksha(ix,v) for k,v in [('inference',inf),('replacement',score),('learned_remainder',inf&~score),('fallback',fallback),('missing_current_open',missing)]},'actual_finite_only_on_existing_replacement_support':True,'shared_variance_and_non_mean_fields_equal':True}
  record['past_fit_return_identity']=fit_record
  records.append(record);regimes[record['regime']]+=1
  for k in ('scheduled_rows','inference_rows','replacement_rows','learned_remainder_rows','fallback_rows','missing_current_open_rows'):totals[k]+=record[k]
 assert totals=={'scheduled_rows':2920,'inference_rows':2586,'replacement_rows':2574,'learned_remainder_rows':12,'fallback_rows':332,'missing_current_open_rows':2}
 assert regimes=={'bull':2,'bear':4,'sideways':2}
 proposed={'old_causal_policies':12,'old_hindsight_policies':16,'new_hindsight_policies':8,'total_policies':36,'economic_rows':288,'base_stress_accounts':576,'new_substituted_forecasts':32,'new_own_state_paths':64,'new_tail_threshold_estimates_after_freeze':8,'new_tail_threshold_models':0,'new_score_rows':192,'new_own_artifacts':392,'new_own_artifacts_per_fold':49,'new_causal_names':0,'existing_causal_ledger_count':174,'replacement_slots_per_new_forecast_family':2574,'retained_learned_slots_per_new_forecast_family':12}
 report={'passed':True,'scope':'Data-only Stage16 source chain and existing array support audit; no new sign, magnitude, mean, loss or order computation','script_sha256':sha(__file__),'parent_source_revision':SOURCE,'parent_file_sha256':HASHES,'parent_registration_canonical_sha256':digest(reg),'counts':dict(counts),'distinct_hashed_files':len(verified),'source_artifact_inventory_sha256':digest(bindings),'source_artifact_bindings':bindings,'direct_source_bindings':direct,'parent_forecast_bindings':forecast_bindings,'support':records,'totals':dict(totals),'regime_counts':dict(regimes),'regime_gate_pass':False,'proposed_fixed_counts':proposed,'tail_threshold_contract':{'past_only':True,'quantile':0.9,'method':'linear','large_move_comparison':'>=','remaining_comparison':'<','evaluation_subset_use':'future-outcome descriptive grouping only, never an order support gate','computed_in_preflight':False},'timing':{'raw_features':'completed bars through t-1','decision':'UTC six-hour schedule','fill':'immediate next 15m open; missing fill cancels','label':'saved h24 return, archive maturity t+375min','replacement':'only existing saved score_support; no removal or expansion of inference/action mask','unscored':'retain saved baseline mean and risk unchanged','missing':'same hold and known-current-open target1 fallback rules'},'limitations':['Prior Stage12 independent audit establishes actual-label and accounting arithmetic; this preflight verifies its immutable binding without rerunning it.','No quantile threshold, zero/sign/magnitude or economic summaries were newly calculated; existing canonical past fit labels are reconstructed only for identity hashes.','Retrospective common availability, reused development and missing historical receipts persist.','Stage16 is proposed until independently registered; source count is not the number of prospective Stage16 output artifacts.']}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'passed':True,'counts':dict(counts),'totals':dict(totals),'proposed':proposed}))
if __name__=='__main__':main()
