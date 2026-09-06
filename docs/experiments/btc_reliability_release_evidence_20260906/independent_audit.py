"""Independent saved-production audit. No fitter/planner/metric imports or refit."""
from pathlib import Path
import hashlib,json,math,sys,subprocess
import numpy as np
import pandas as pd
import joblib,yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/btc_reliability_release_v1';BUNDLE=OUT/'bundle'
EXPECTED='a8d2ec176b6361ea50fda2ee5d6e2b695262ca88'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def matrix(x):
 x=np.asarray(x,dtype='<f8',order='C')
 return hashlib.sha256(np.asarray([x.ndim,*x.shape],dtype='<i8').tobytes()+x.tobytes(order='C')).hexdigest()
def checked_equal(a,b,name):
 assert np.array_equal(a,b,equal_nan=True),name
 return 0.
def maxdiff(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def avg(a):
 a=list(a);return math.fsum(float(x)/len(a) for x in a)

def main():
 reg=read(OUT/'registration.json'); pre=read(OUT/'preflight.json'); manifest=read(BUNDLE/'manifest.json');done=read(OUT/'completed.json');prov=read(OUT/'fit_provenance.json')
 cal=read(BUNDLE/'calibration.json');rel=read(BUNDLE/'reliability.json');cfg=yaml.safe_load((ROOT/'configs/btc_reliability_release_20260906.yaml').read_text())
 assert done['source_revision']==reg['source_revision']==manifest['source_revision']==EXPECTED
 assert (done['new_model_fits'],done['new_reliability_weights'],done['new_scores'],done['new_policy_paths'])==(3,1,0,0)
 assert manifest['research_scores_apply_to_production_weights'] is False and manifest['rl_qualified'] is False
 assert manifest['high_probability_generalization_established'] is False and manifest['production_evaluation_scores_computed'] is False
 assert sha(BUNDLE/'manifest.json')==done['manifest_sha256']=='c62287a3f21ad2407b61382648d4c6ceda718cb7bf1c1d5bcc99926dc225abe6'
 assert cfg==reg['config'] and sha(ROOT/'configs/btc_reliability_release_20260906.yaml')==reg['config_sha256']==manifest['config_sha256']
 assert sha(OUT/'preflight.json')==cfg['preflight_sha256']==reg['preflight_sha256']==manifest['preflight_sha256']
 bindings={};checks=0
 for part in (pre['source_bindings'],pre['direct_source_bindings'],pre['raw_artifact_bindings'],done['artifact_sha256']):
  for p,h in part.items():
   path=Path(p); path=path if path.is_absolute() else ROOT/path
   assert str(path.resolve()) not in bindings or bindings[str(path.resolve())]==h
   bindings[str(path.resolve())]=h;assert sha(path)==h,p;checks+=1
 assert len(pre['source_bindings'])==23 and len(pre['raw_artifact_bindings'])==178 and len(done['artifact_sha256'])==11
 for p,h in pre['source_bindings'].items():
  old=subprocess.check_output(['git','show',EXPECTED+':'+p],cwd=ROOT)
  assert hashlib.sha256(old).hexdigest()==h,p
 for p,h in manifest['files'].items():
  assert sha(BUNDLE/p)==h==manifest['artifacts'][p]['sha256'];assert (BUNDLE/p).stat().st_size==manifest['artifacts'][p]['bytes'];checks+=1
 assert len(manifest['files'])==6
 assert canon(manifest['execution'])==manifest['execution_contract_sha256']
 assert canon(manifest['feature_contract'])==manifest['feature_contract_sha256']
 assert manifest['feature_contract']['feature_columns']==manifest['feature_columns']==pre['feature_columns']
 assert len(manifest['feature_contract']['common_mask_components'])==10
 assert manifest['source_bindings']==pre['source_bindings'] and manifest['raw_artifact_bindings']==pre['raw_artifact_bindings']
 selection=read(ROOT/cfg['selection_path']);assert sha(ROOT/cfg['selection_path'])==cfg['selection_sha256']==manifest['selection_sha256']
 assert manifest['historical_evidence']['selected']==selection['selected'] and manifest['historical_evidence']['validation_period']==selection['validation_period']
 assert manifest['historical_evidence']['source_results']==selection['source_results']
 for p,h in selection['source_results'].items():assert sha(ROOT/p)==h
 counts=pre['support']['counts'];assert counts=={'fit':2167,'scale':359,'interval':363,'predict':724,'inference':0,'scheduled':0,'score':0}
 assert cal['counts']=={k:counts[k] for k in ('fit','scale','interval')}
 snap=dict(np.load(OUT/'selected_inputs.npz',allow_pickle=False));fix=dict(np.load(BUNDLE/'prediction_fixture.npz',allow_pickle=False))
 assert set(fix)=={'timestamps','technical_features','perp_features','raw_technical_mu','raw_perp_mu','raw_log_variance','mu','variance'}
 assert set(snap)=={'fit_positions','fit_timestamps','predict_positions','predict_timestamps','fit_actual','scale_mask_on_predict','interval_mask_on_predict','calibration_actual_on_predict','fit_features_technical','fit_features_perp_delay0','predict_features_technical','predict_features_perp_delay0'}
 ts=pd.to_datetime(snap['predict_timestamps'],utc=True); fts=pd.to_datetime(snap['fit_timestamps'],utc=True)
 assert ts.is_monotonic_increasing and ts.is_unique and fts.is_monotonic_increasing and fts.is_unique
 assert np.array_equal(fix['timestamps'],snap['predict_timestamps'])
 assert len(ts)==724 and len(fts)==2167 and ts[-1]<pd.Timestamp('2026-07-16T13:45Z')
 for index in (ts,fts):assert np.all((index.hour%6==0)&(index.minute==0)&(index.second==0))
 S=snap['scale_mask_on_predict'];I=snap['interval_mask_on_predict'];actual=snap['calibration_actual_on_predict']
 assert S.dtype==I.dtype==bool and not (S&I).any() and int(S.sum())==359 and int(I.sum())==363
 assert int((~(S|I)).sum())==2 and np.isnan(actual[~(S|I)]).all()
 for name,idx,bound in [('fit',fts,'2026-01-16T13:45Z'),('scale',ts[S],'2026-04-16T13:45Z'),('interval',ts[I],'2026-07-16T13:45Z')]:
  assert idx[-1]+pd.Timedelta(minutes=375)<pd.Timestamp(bound)
  assert (idx[-1]+pd.Timedelta(minutes=375)).isoformat()==pre['support']['last_label_maturity'][name]
 for group,fx in [('technical','technical_features'),('perp_delay0','perp_features')]:
  for subset in ('fit','predict'):
   assert matrix(snap[subset+'_features_'+group])==pre['support']['feature_sha256'][subset][group]
  assert np.array_equal(snap['predict_features_'+group],fix[fx])
 assert matrix(snap['fit_actual'])==pre['support']['label_sha256']['fit']
 for name,mask in [('scale',S),('interval',I)]:assert matrix(actual[mask])==pre['support']['label_sha256'][name]
 # Independent scalar standardized linear model arithmetic on every fixture row.
 scalar_err={};serialized_err={};scaler_err={}; ridge_predictions={}
 for group,fx,target in [('technical','technical_features','raw_technical_mu'),('perp_delay0','perp_features','raw_perp_mu')]:
  model=joblib.load(BUNDLE/'models'/f'{group}_mean.joblib');scaler,ridge=(step[1] for step in model.steps)
  assert float(ridge.alpha)==100. and int(model.n_features_in_)==fix[fx].shape[1]
  train=snap['fit_features_'+group]
  mean=np.asarray([avg(train[:,j]) for j in range(train.shape[1])])
  scaler_err[group]=maxdiff(mean,scaler.mean_);assert scaler_err[group]<1e-8
  pred=np.asarray([float(ridge.intercept_)+math.fsum(((float(x)-float(m))/float(s))*float(c) for x,m,s,c in zip(row,scaler.mean_,scaler.scale_,ridge.coef_)) for row in fix[fx]])
  scalar_err[group]=maxdiff(pred,fix[target]);assert scalar_err[group]<=1e-12
  got=model.predict(fix[fx]);serialized_err[group]=checked_equal(got,fix[target],group+' serialized');ridge_predictions[group]=pred
 # Independent HGB tree traversal; the saved tree leaves include learning rate.
 hgb=joblib.load(BUNDLE/'models/technical_variance.joblib'); assert hgb.n_iter_==100
 hgb_rows=[]
 for row in fix['technical_features']:
  value=float(hgb._baseline_prediction[0,0])
  for round_trees in hgb._predictors:
   assert len(round_trees)==1
   nodes=round_trees[0].nodes;pos=0
   while not nodes[pos]['is_leaf']:
    node=nodes[pos];assert not node['is_categorical'];x=float(row[int(node['feature_idx'])])
    left=bool(node['missing_go_to_left']) if math.isnan(x) else x<=float(node['num_threshold'])
    pos=int(node['left'] if left else node['right'])
   value+=float(nodes[pos]['value'])
  hgb_rows.append(value)
 hgb_error=maxdiff(hgb_rows,fix['raw_log_variance']);assert hgb_error<=1e-12
 serialized_err['technical_variance']=checked_equal(hgb.predict(fix['technical_features']),fix['raw_log_variance'],'HGB serialized')
 # Reconstruct saved calibrations and S-only weight without importing canonical helpers.
 y=actual[:,0];anchor=avg(y[S]);assert anchor==cal['scale_mean']==rel['anchor']
 bias_error={g:abs(avg(y[S]-fix['raw_technical_mu' if g=='technical' else 'raw_perp_mu'][S])-cal['return_bias'][g]) for g in ('technical','perp_delay0')}
 assert max(bias_error.values())<=1e-15
 raw_variance=np.exp(np.clip(fix['raw_log_variance'],np.log(1e-12),0));av=np.maximum(actual[:,2]**2,1e-12)
 multiplier_error=abs(avg(av[S]/raw_variance[S])-cal['variance_multiplier']);assert multiplier_error<=1e-12
 full=fix['raw_perp_mu']+cal['return_bias']['perp_delay0'];d=full[S]-anchor;r=y[S]-anchor
 B=avg(float(v)*float(v) for v in d);C=avg(float(x)*float(y0) for x,y0 in zip(d,r))
 assert B==rel['innovation_secondmoment'] and C==rel['crossmoment'] and avg(d)==rel['mean_d'] and avg(r)==rel['mean_r']
 w=0. if B==0 or C<=0 else (1. if C>=B else C/B);assert w==rel['weight']==manifest['forecast']['weight'] and rel['n']==359
 mu=np.full(len(full),anchor) if w==0 else full.copy() if w==1 else w*full+(1-w)*anchor
 mu_error=checked_equal(mu,fix['mu'],'reliability formula');variance_error=checked_equal(np.maximum(raw_variance*cal['variance_multiplier'],1e-12),fix['variance'],'variance formula')
 assert np.isfinite(np.column_stack([fix[k] for k in ('raw_technical_mu','raw_perp_mu','raw_log_variance','mu','variance')])).all()
 log=Path('/tmp/btc-reliability-release-production-run.log').read_text();warnings=[line for line in log.splitlines() if 'RuntimeWarning:' in line]
 assert 'Traceback' not in log
 report={'schema':'btc-reliability-release-independent-audit-v1','status':'passed','source_revision':EXPECTED,'bundle_sha256':sha(BUNDLE/'manifest.json'),'completed_sha256':sha(OUT/'completed.json'),'audit_script_sha256':sha(__file__),'source_binding_count':23,'raw_artifact_binding_count':178,'completed_artifact_count':11,'bundle_payload_files':6,'distinct_checked_paths':len(bindings),'binding_verification_count':checks,'model_count':3,'reliability_weight_count':1,'new_evaluation_scores':0,'new_policy_paths':0,'fit_rows':2167,'scale_rows':359,'interval_rows':363,'fixture_rows':724,'masked_calibration_tail_rows':2,'weight':w,'weight_case':rel['weight_case'],'scalar_Ridge_prediction_max_abs_difference':scalar_err,'independent_HGB_tree_prediction_max_abs_difference':hgb_error,'serialized_prediction_max_abs_difference':serialized_err,'scalar_scaler_mean_max_abs_difference':scaler_err,'scalar_bias_max_abs_difference':bias_error,'scalar_variance_multiplier_abs_difference':multiplier_error,'independent_reliability_mu_max_abs_difference':mu_error,'independent_variance_max_abs_difference':variance_error,'reliability_moments_exact':True,'S_only_maturity_and_inputs_verified':True,'input_preflight_feature_label_hashes_exact':True,'production_run_log_sha256':sha('/tmp/btc-reliability-release-production-run.log'),'runtime_warning_count':len(warnings),'runtime_warning_classes':{msg:sum(msg in line for line in warnings) for msg in ('divide by zero','overflow','invalid value')},'limitations':['No new evaluation performance, ranking or economic paths computed.','Feature reconstruction is bound to committed input preflight hashes, not independently implemented formulas.','Official archive hashes do not establish historical receipt timing.','Warnings retained; finite scalar Ridge and tree-traversal plus serialization checks passed.','Historical selection metrics describe the procedure, not production weights or live results.']}
 dest=Path('/tmp/btc_reliability_release_audit_20260906.json');dest.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True));print('audit_json_sha256',sha(dest))
if __name__=='__main__':main()
