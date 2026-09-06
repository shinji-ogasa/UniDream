"""Independent Stage20 saved-state scalar audit; no fit/predict/project helper.

Only SHA-bound saved T/predict matrices, fitted states, probabilities, mapping
and ancestor provenance are read. No raw market data, new model or policy run.
Uses the already independent Stage18 scalar arithmetic, bound by script hash.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,os,runpy,subprocess
import numpy as np
import pandas as pd
import sklearn
import joblib
import yaml

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('codex_outputs/oracle_short_direction_decisions_v1')
OLD=Path('codex_outputs/oracle_direction_decisions_v1')
SOFT=Path('codex_outputs/oracle_soft_direction_decisions_v1')
CONFIG=Path('configs/oracle_short_direction_decisions_20260906.yaml')
ARITHMETIC=Path('/tmp/oracle_regularized_direction_model_audit_20260906.py')
REPORT=Path('/tmp/oracle_short_direction_model_audit_20260906.json')
GROUP='technical_short_both';FOLDS=tuple(range(5,13));WEIGHTS=('ordinary','magnitude')
TOL={'gradient_infinity':1e-6,'logit_atol':1e-12,'probability_atol':1e-14,
     'runtime_objective_gradient_atol':1e-14,'scaler_atol':1e-10,'scaler_rtol':1e-12}
A=runpy.run_path(str(ARITHMETIC),run_name='independent_stage18_arithmetic')
sha,read,digest,matrixsha,arrays,exact,near,sigmoid,logit_rows=[A[k] for k in (
 'sha','read','digest','matrixsha','arrays','exact','near','sigmoid','logit_rows')]


def run():
 os.chdir(ROOT);maxima={};counts=Counter();verified={};model_checks=[];mapping_checks=[];direction_checks=[]
 def verify(p,h):
  key=str(Path(p).resolve())
  if key not in verified:verified[key]=sha(p)
  assert verified[key]==h,('hash',str(p));counts['hash_bindings']+=1
 reg,pre,res=[read(OUT/(n+'.json')) for n in ('registration','preflight','results')]
 cfg=yaml.safe_load(CONFIG.read_text());revision=subprocess.check_output(['git','rev-parse','69d2cd6'],text=True).strip()
 assert reg['source_revision']==revision and res['registration_sha256']==digest(reg)
 assert cfg==reg['config'] and cfg['new_model_fits']==16 and cfg['development_folds']==list(FOLDS)
 verify(CONFIG,reg['config_sha256']);verify(OUT/'preflight.json',cfg['preflight_sha256'])
 assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
 assert (np.__version__,pd.__version__,sklearn.__version__)==('2.2.6','2.3.3','1.8.0')
 for section in ('source_bindings','direct_source_bindings','source_artifact_bindings'):
  for p,h in pre[section].items():verify(p,h)
 for p,h in cfg['source_bindings'].items():verify(p,h)
 counts['source_files']=len(cfg['source_bindings']);counts['ancestor_artifacts']=len(pre['source_artifact_bindings'])
 for section in ('parent_manifest_bindings','feature_manifest_bindings'):
  for p,h in cfg[section].items():verify(p,h)
 fullindex=pd.date_range(pd.Timestamp('2018-01-01',tz='UTC'),pd.Timestamp(cfg['data_cutoff'])-pd.Timedelta(minutes=15),freq='15min',tz='UTC')
 records={r['fold']:r for r in res['fit_records']};assert len(records)==len(res['fit_records'])==8
 dindex={(r['fold'],r['segment'],r['classifier_id']):r for r in res['direction_diagnostics']}
 mindex={(r['fold'],r['segment']):r for r in res['mapping_diagnostics']}
 assert len(dindex)==len(res['direction_diagnostics'])==32 and len(mindex)==len(res['mapping_diagnostics'])==16
 for f in FOLDS:
  fold=read(OUT/f'fold_{f}.json');assert fold['registration_sha256']==digest(reg) and len(fold['artifact_sha256'])==95
  for p,h in fold['artifact_sha256'].items():verify(p,h)
  counts['new_artifacts']+=95
  data=arrays(OUT/'fit_data'/f'fold{f}_training.npz');olddata=arrays(OLD/'fit_data'/f'fold{f}_training.npz')
  assert set(data)==set(olddata)|{'fit_features_'+GROUP,'predict_features_'+GROUP}
  for k in olddata:exact(data[k],olddata[k],'unchanged old fit pack '+k)
  record=read(OUT/'provenance'/f'fold{f}_fit.json');assert record==records[f]
  prov=record['fit_provenance'];bound=next(s for s in pre['support'] if s['fold']==f)
  oldrec=read(OLD/'provenance'/f'fold{f}_fit.json');oldprov=oldrec['fit_provenance']
  shortprov=read(bound['short_fit_provenance_path'])['fit_provenance']
  assert record['fit_source_binding']==bound and record['new_model_fits']==2
  assert not prov['evaluation_labels_used'] and not prov['support_narrowed'] and not prov['model_selection_performed']
  assert prov['feature_columns'][GROUP]==bound['short_feature_columns'] and len(bound['short_feature_columns'])==37
  fp,pp=data['fit_positions'],data['predict_positions'];nt=len(fp);npred=len(pp)
  assert nt>=512 and npred>0 and fp[-1]<pp[0] and np.all(np.diff(fp)>0) and np.all(np.diff(pp)>0)
  for name,positions,times in [('fit',fp,data['timestamps']),('predict',pp,data['predict_timestamps'])]:
   m=np.zeros(len(fullindex),bool);m[positions]=True
   exact(fullindex.asi8[positions],times,name+' time')
   assert A['position_sha'](m)==prov['mask_position_sha256'][name]
   assert A['masksha'](fullindex,m)==bound['mask_sha256'][name]
   assert len(positions)==bound['counts'][name]==prov['mask_counts'][name]
  xf,xp=[data[s+'_features_'+GROUP] for s in ('fit','predict')]
  for name,x in [('fit',xf),('predict',xp)]:
   assert x.shape==(nt if name=='fit' else npred,37) and np.isfinite(x).all()
   h=matrixsha(x);assert h==prov[name+'_features_sha256'][GROUP]==bound['short_'+name+'_features_sha256']
   assert h==shortprov[name+'_features_sha256'][GROUP]
   exact(x[:,:29],olddata[name+'_features_technical'],'old29 prefix')
  returns=data['returns'];labels=np.asarray([int(float(v)>0) for v in returns],np.int64)
  exact(labels,data['binary_labels'],'labels');assert matrixsha(labels)==prov['fit_binary_labels_sha256']
  assert matrixsha(returns)==prov['fit_return_sha256']
  assert matrixsha(np.column_stack((xf,returns)))==prov['fit_features_and_return_sha256'][GROUP]
  amp=math.fsum(abs(float(v))/nt for v in returns);mean=float(np.mean(returns))
  assert record['fit_abs_return_mean']==oldrec['fit_abs_return_mean']==amp
  assert record['fit_return_mean']==oldrec['fit_return_mean']==mean
  assert prov['sample_weights']==oldprov['sample_weights']
  center=np.asarray([math.fsum(float(v)/nt for v in xf[:,j]) for j in range(37)])
  variance=np.asarray([math.fsum((float(v)-float(center[j]))**2/nt for v in xf[:,j]) for j in range(37)])
  scale=np.sqrt(variance);eps=np.finfo(float).eps;scale[variance<=nt*eps*variance+(nt*center*eps)**2]=1.
  rawz={};rawq={}
  for w in WEIGHTS:
   mid=GROUP+'_'+w;weight=np.ones(nt) if w=='ordinary' else np.asarray([abs(float(v))/amp for v in returns])
   exact(weight,data['weights_'+w],'weight')
   total=math.fsum(float(v) for v in weight);prior=math.fsum(float(v) for v in weight[labels==1])/total
   assert record['fit_priors'][w]==oldrec['fit_priors'][w]==prior
   state=prov['fitted_state'][mid];model=joblib.load(OUT/'models'/f'fold{f}_{mid}.joblib')
   assert [k for k,_ in model.steps]==['standardscaler','logisticregression']
   scaler,lr=model.steps[0][1],model.steps[1][1]
   assert lr.get_params()==prov['parameters']['logistic']==oldprov['parameters']['logistic'] and lr.C==1.
   assert lr.n_iter_[0]<1000 and lr.n_iter_.tolist()==state['n_iter'] and lr.classes_.tolist()==[0,1]
   for value,key in [(scaler.mean_,'scaler_mean'),(scaler.var_,'scaler_variance'),(scaler.scale_,'scaler_scale'),
                     (lr.coef_,'coefficient'),(lr.intercept_,'intercept')]:
    assert np.isfinite(value).all() and value.tolist()==state[key]
   assert matrixsha(lr.coef_)==state['coefficient_sha256'] and matrixsha(lr.intercept_)==state['intercept_sha256']
   assert int(scaler.n_samples_seen_)==nt==state['scaler_rows']
   for key,oldkey in [('mean_','scaler_mean'),('var_','scaler_variance'),('scale_','scaler_scale')]:
    exact(getattr(scaler,key)[:29],np.asarray(oldprov['fitted_state']['technical_'+w][oldkey]),'old scaler29')
   near(scaler.mean_,center,'independent_scaler_mean',maxima,atol=1e-10,rtol=1e-12)
   near(scaler.var_,variance,'independent_scaler_variance',maxima,atol=1e-10,rtol=1e-12)
   near(scaler.scale_,scale,'independent_scaler_scale',maxima,atol=1e-10,rtol=1e-12)
   design,fitz=logit_rows(xf,scaler.mean_,scaler.scale_,lr.coef_[0],lr.intercept_[0])
   # Same objective as the pinned sklearn solver; scalar gradient only (no refit/Hessian eigensolve).
   W=float(np.sum(weight));l2=1/(float(lr.C)*W);residual=[];loss=[]
   for z,y,v in zip(fitz,labels,weight):
    a=float(v)/W;loss.append(a*(max(-z if y else z,0.)+math.log1p(math.exp(-abs(z)))))
    residual.append(a*(-sigmoid(-z) if y else sigmoid(z)))
   objective=math.fsum(loss)+.5*l2*math.fsum(float(b)*float(b) for b in lr.coef_[0])
   gradient=np.asarray([math.fsum(r*row[j] for r,row in zip(residual,design))+l2*float(lr.coef_[0,j]) for j in range(37)]+[math.fsum(residual)])
   assert math.isfinite(objective) and np.isfinite(gradient).all() and max(abs(gradient))<=1e-6
   check=state['scalar_verification'];assert check['checked'] and check['gradient_bound']==1e-6
   near(gradient,check['normalized_gradient'],'runtime_gradient',maxima)
   near(objective,check['normalized_objective'],'runtime_objective',maxima)
   _,z=logit_rows(xp,scaler.mean_,scaler.scale_,lr.coef_[0],lr.intercept_[0]);q=[sigmoid(v) for v in z]
   rawz[mid]=np.asarray(z);rawq[mid]=np.asarray(q)
   for seg,kind in [('interval','calibration'),('evaluation','forecasts')]:
    pred=arrays(OUT/('probabilities_'+seg)/f'fold{f}_{mid}.npz')
    old=arrays(OLD/kind/f'fold{f}_technical_{w}_direction.npz')
    maskkey='mapped_inference_mask' if seg=='interval' else 'inference_mask'
    exact(pred['timestamps'],old['timestamps'],'probability calendar')
    exact(pred['mapped_inference_mask'],old[maskkey],'I support')
    exact(pred['score_support'],old['interval_mask'] if seg=='interval' else old['score_support'],'O support')
    t=pred['timestamps'];where=(data['predict_timestamps']>=t[0])&(data['predict_timestamps']<=t[-1])
    pos=np.searchsorted(t,data['predict_timestamps'][where]);m=np.zeros(len(t),bool);m[pos]=True
    exact(m,pred['predict_mask'],'predict support')
    exact(np.isfinite(pred['logit']),m,'finite z');exact(np.isfinite(pred['probability']),m,'finite q')
    near(pred['logit'][m],rawz[mid][where],'scalar_predict_logit',maxima,atol=1e-12)
    near(pred['probability'][m],rawq[mid][where],'scalar_predict_probability',maxima,atol=1e-14)
    I=pred['mapped_inference_mask'];n=int(I.sum());zz=pred['logit'][I];qq=pred['probability'][I]
    d=dindex[f,seg,mid]
    expected={'rows':n,'zero_logit_rows':int((zz==0).sum()),'probability_zero_rows':int((qq==0).sum()),
              'probability_one_rows':int((qq==1).sum()),'sign_disagreements_vs_Technical29':int((np.sign(zz)!=np.sign(old['logit'][I])).sum()),
              'mean_abs_logit':math.fsum(abs(float(v))/n for v in zz),
              'old_mean_abs_logit':math.fsum(abs(float(v))/n for v in old['logit'][I])}
    for k,v in expected.items():assert d[k]==v,(f,seg,mid,k)
    direction_checks.append({'fold':f,'segment':seg,'model_id':mid,**expected})
    counts['probability_npz']+=1;counts['audited_probability_predict_rows']+=int(m.sum())
   counts['models']+=1;counts['fit_model_rows']+=nt;counts['predict_model_rows']+=npred
   model_checks.append({'fold':f,'model_id':mid,'fit_rows':nt,'predict_rows':npred,'gradient_infinity':float(max(abs(gradient))),
     'normalized_objective':objective,'C':float(lr.C),'l2_gradient_strength':l2,'coefficient_norm':math.sqrt(math.fsum(float(b)**2 for b in lr.coef_[0]))})
  mapping_record=read(OUT/'provenance'/f'fold{f}_mapping.json');assert mapping_record['ordinary_probability_not_mapped']
  assert mapping_record['saved_T_scalars']==bound['saved_mapping_scalars']
  for seg,kind in [('interval','calibration'),('evaluation','forecasts')]:
   name=GROUP+'_magnitude_soft';new=arrays(OUT/kind/f'fold{f}_{name}.npz')
   old=arrays(OLD/kind/f'fold{f}_technical_magnitude_direction.npz')
   assert set(new)==set(old)
   for k in old:
    if k not in ('mu','logit','probability'):exact(new[k],old[k],'unchanged risk/actual/mask '+k)
   pred=arrays(OUT/('probabilities_'+seg)/f'fold{f}_{GROUP}_magnitude.npz')
   for k in ('logit','probability'):exact(new[k],pred[k],'mapped saved '+k)
   key='mapped_inference_mask' if seg=='interval' else 'inference_mask';I=new[key];n=int(I.sum())
   assert np.isnan(new['mu'][~I]).all()
   mu=np.asarray([amp*(2.*float(q)-1.) for q in new['probability'][I]])
   exact(mu,new['mu'][I],'direct scalar mapping')
   d=mindex[f,seg];md=d['mapping_diagnostic'];priorq=bound['saved_mapping_scalars']['prior_probability']['technical']
   prior_mu=amp*(2.*priorq-1.);difference=prior_mu-mean
   for k,v in {'fit_abs_return_mean':amp,'fit_return_mean':mean,'saved_weighted_prior_probability':priorq,
      'mapped_prior':prior_mu,'prior_identity_signed_difference':difference,'prior_identity_absolute_difference':abs(difference),
      'prior_identity_tolerance':1e-14+1e-12*abs(mean),'inference_rows':n,'total_rows':len(I),
      'noninference_rows':len(I)-n,'probability_half_rows':int((new['probability'][I]==.5).sum())}.items():assert md[k]==v,(f,seg,k)
   assert abs(difference)<=md['prior_identity_tolerance'] and md['prior_identity_passed']
   oldsoft=arrays(SOFT/kind/f'fold{f}_technical_magnitude_soft.npz')
   expected={'rows':n,'mapped_zero_mean_rows':int((mu==0).sum()),
      'mapped_direction_vs_logit_disagreements':int((np.sign(mu)!=np.sign(new['logit'][I])).sum()),
      'mean_abs_new_mu':math.fsum(abs(float(v))/n for v in mu),
      'mean_abs_old_soft_mu':math.fsum(abs(float(v))/n for v in oldsoft['mu'][I])}
   for k,v in expected.items():assert d[k]==v,(f,seg,k)
   mapping_checks.append({'fold':f,'segment':seg,**expected,'prior_identity_residual':difference,'mapping_max_abs_difference':0.})
   counts['soft_npz']+=1;counts['mapped_inference_rows']+=n
  print(json.dumps({'fold':f,'audited':True,'models':2,'probability_npz':4,'soft_npz':2}),flush=True)
 assert counts['models']==16 and counts['new_artifacts']==760 and counts['probability_npz']==32 and counts['soft_npz']==16
 report={'schema':'independent-stage20-scalar-model-audit-v1','passed':True,'source_revision':revision,
   'script_sha256':sha(__file__),'arithmetic_script':str(ARITHMETIC),'arithmetic_script_sha256':sha(ARITHMETIC),
   'registration_file_sha256':sha(OUT/'registration.json'),'registration_canonical_sha256':digest(reg),
   'results_file_sha256':sha(OUT/'results.json'),'config_sha256':sha(CONFIG),'preflight_sha256':sha(OUT/'preflight.json'),
   'counts':dict(counts),'verified_distinct_files':len(verified),'max_abs_differences':maxima,'tolerances':TOL,
   'maximum_normalized_gradient_infinity':max(r['gradient_infinity'] for r in model_checks),
   'models':model_checks,'direction_diagnostics':direction_checks,'mapping_diagnostics':mapping_checks,
   'new_fits_or_estimator_predict_or_new_policy_rollouts':False,'raw_market_data_read':False,
   'feature_scope':'selected37 hashes match completed Stage15; old29 prefix/old fit pack/scaler exact; no new raw reconstruction',
   'labels_scope':'selected T returns bound exactly to Stage17; validation actual arrays checked for unchanged bytes only; no score recalculation',
   'warning_scope':'Registered runtime matmul warning categories may remain; finite scalar verification does not establish their cause.'}
 if (OUT/'run.log').exists():report['run_log_sha256']=sha(OUT/'run.log')
 REPORT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'passed':True,'report':str(REPORT),'report_sha256':sha(REPORT),'counts':dict(counts),'max_abs_differences':maxima}),flush=True)

if __name__=='__main__':run()
