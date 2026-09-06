"""Read-only old-C1 objective recheck; extends the sealed Stage18 audit scope."""
import hashlib,importlib.util,json,math
from pathlib import Path
import joblib,numpy as np
BASE_SCRIPT=Path('/tmp/oracle_regularized_direction_model_audit_20260906.py')
BASE_REPORT=Path('/tmp/oracle_regularized_direction_model_audit_20260906.json')
OUTPUT=Path('/tmp/oracle_regularized_direction_old_objective_audit_20260906.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE_SCRIPT)=='79ffa45846e3bd4bdbf6d5c93a9a0727654e5e3534a87cb25f62bc74235300df'
assert sha(BASE_REPORT)=='f8f533d70b2978ce53e1fd0951dc0b17d423589de32777ecc025c9b39f8e5159'
spec=importlib.util.spec_from_file_location('independent_stage18_audit',BASE_SCRIPT)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
root=a.ROOT;parent=root/a.PARENT
pre=a.read(root/a.OUT/'preflight.json');bindings={};rows=[];maxima={}
def bind(p):
 k=str(p.relative_to(root));h=sha(p);assert h==pre['source_artifact_bindings'][k];bindings[k]=h
for fold in a.FOLDS:
 path=parent/'fit_data'/f'fold{fold}_training.npz';bind(path);data=a.arrays(path)
 pp=parent/'provenance'/f'fold{fold}_fit.json';bind(pp);record=a.read(pp)
 for group in a.GROUPS:
  xf=data['fit_features_'+group]
  assert np.isfinite(xf).all()
  for weighting in a.WEIGHTINGS:
   mid=group+'_'+weighting;path=parent/'models'/f'fold{fold}_{mid}.joblib';bind(path)
   model=joblib.load(path);scaler,lr=model.steps[0][1],model.steps[1][1]
   assert lr.C==1. and np.isfinite(lr.coef_).all() and np.isfinite(lr.intercept_).all()
   state=record['fit_provenance']['fitted_state'][mid];check=state['scalar_verification']
   assert lr.coef_.tolist()==state['coefficient'] and lr.intercept_.tolist()==state['intercept']
   x,_=a.logit_rows(xf,scaler.mean_,scaler.scale_,lr.coef_[0],float(lr.intercept_[0]))
   objective,gradient,hessian=a.objective_gradient_hessian(x,data['binary_labels'],data['weights_'+weighting],lr.coef_[0],float(lr.intercept_[0]),lr.C)
   gradmax=float(np.max(np.abs(gradient)));eigenmin=float(np.linalg.eigvalsh(hessian).min())
   assert gradmax<=a.TOL['gradient_infinity'] and eigenmin>=a.TOL['hessian_eigenvalue_floor']
   a.near(objective,check['normalized_objective'],'old_runtime_objective',maxima,a.TOL['runtime_gradient_objective_atol'])
   a.near(gradient,check['normalized_gradient'],'old_runtime_gradient',maxima,a.TOL['runtime_gradient_objective_atol'])
   rows.append({'fold':fold,'model_id':mid,'fit_rows':len(xf),'C':lr.C,
     'actual_lambda':1/(lr.C*float(np.sum(data['weights_'+weighting]))),
     'normalized_objective':objective,'gradient_infinity':gradmax,'minimum_hessian_eigenvalue':eigenmin,
     'model_sha256':sha(path)})
assert len(rows)==32 and len(bindings)==48
result={'passed':True,'scope':'32 old Stage17 C1 objectives/gradients/Hessians independently rechecked as extension of Stage18 new/old predictor audit; no fit or policy execution',
 'script_sha256':sha(__file__),'sealed_stage18_audit_script_sha256':sha(BASE_SCRIPT),'sealed_stage18_audit_report_sha256':sha(BASE_REPORT),
 'tolerances':a.TOL,'models':32,'fit_model_rows':sum(v['fit_rows'] for v in rows),
 'max_absolute_differences':maxima,'maximum_gradient_infinity':max(v['gradient_infinity'] for v in rows),
 'minimum_hessian_eigenvalue':min(v['minimum_hessian_eigenvalue'] for v in rows),
 'model_records':rows,'source_bindings':bindings,'new_coefficients_fitted':False,
 'warning_cause_established':False}
OUTPUT.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(OUTPUT),'sha256':sha(OUTPUT),'passed':True,'models':32,
 'maximum_gradient_infinity':result['maximum_gradient_infinity'],'max_absolute_differences':maxima,
 'minimum_hessian_eigenvalue':result['minimum_hessian_eigenvalue']}))
