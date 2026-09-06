"""Independent Stage17 model audit; default synthetic, real mode only on root GO.

No project helper, estimator fit/predict or alternative optimizer is called.
Saved training matrices are tied to preflight hashes; selected T returns are
reconstructed from cutoff-filtered Spot bars. Fixed fitted coefficients enter
independent scalar logits, objective, gradient and Hessian, never a new refit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('codex_outputs/oracle_direction_decisions_v1')
CONFIG=Path('configs/oracle_direction_decisions_20260906.yaml')
PARITY=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
REPORT=Path('/tmp/oracle_direction_model_audit_20260906.json')
GROUPS=('technical','perp_delay0');WEIGHTINGS=('ordinary','magnitude');FOLDS=tuple(range(5,13))
TOL={'gradient_infinity':1e-6,'logit_atol':1e-12,'probability_atol':1e-14,
     'runtime_gradient_objective_atol':1e-14,'scaler_atol':1e-10,'scaler_rtol':1e-12,
     'hessian_eigenvalue_floor':-1e-12}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def matrixsha(a):
    a=np.asarray(a,dtype='<f8',order='C')
    return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes()).hexdigest()
def masksha(index,mask):return hashlib.sha256(index.asi8.tobytes()+np.asarray(mask,bool).tobytes()).hexdigest()
def position_sha(mask):return hashlib.sha256(np.asarray([len(mask)],dtype='<i8').tobytes()+np.asarray(mask,'u1').tobytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),(name,'exact')
def near(a,b,name,maxima,atol=1e-14,rtol=0.):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
    delta=float(np.max(np.abs(a-b))) if a.size else 0.
    maxima[name]=max(maxima.get(name,0.),delta)
    assert np.allclose(a,b,atol=atol,rtol=rtol),(name,delta,atol,rtol)
def sigmoid(z):
    e=math.exp(-abs(z));return 1/(1+e) if z>=0 else e/(1+e)
def softplus(z):return max(z,0.)+math.log1p(math.exp(-abs(z)))
def logit_rows(x,center,scale,beta,intercept):
    transformed=[[(float(v)-float(c))/float(s) for v,c,s in zip(row,center,scale)] for row in x]
    margins=[float(intercept)+math.fsum(v*float(b) for v,b in zip(row,beta)) for row in transformed]
    assert all(math.isfinite(v) for v in margins)
    return transformed,margins


def objective_gradient_hessian(x,labels,weights,beta,intercept):
    """x is independently transformed; no BLAS for objective/derivatives."""
    total=float(np.sum(weights));p=len(beta)
    margins=[float(intercept)+math.fsum(float(b)*v for b,v in zip(beta,row)) for row in x]
    residual=[];curvature=[];loss=[]
    for z,label,w in zip(margins,labels,weights):
        a=float(w)/total;q=sigmoid(z)
        residual.append(a*(-sigmoid(-z) if label else q))
        curvature.append(a*sigmoid(z)*sigmoid(-z))
        loss.append(a*softplus(-z if label else z))
    objective=math.fsum(loss)+math.fsum(float(b)**2 for b in beta)/(2*total)
    gradient=[math.fsum(r*row[j] for r,row in zip(residual,x))+float(beta[j])/total for j in range(p)]
    gradient.append(math.fsum(residual))
    design=[[*row,1.] for row in x]
    hessian=np.asarray([[math.fsum(w*row[j]*row[k] for w,row in zip(curvature,design))
                         +(1/total if j==k and j<p else 0.) for k in range(p+1)] for j in range(p+1)])
    assert math.isfinite(objective) and np.isfinite(gradient).all() and np.isfinite(hessian).all()
    return objective,np.asarray(gradient),hessian


def synthetic():
    x=[[0.],[0.],[0.],[0.]];labels=[0,1,1,1];w=np.ones(4)
    objective,g,h=objective_gradient_hessian(x,labels,w,[0.],math.log(3.))
    assert max(abs(g))<1e-15 and math.isfinite(objective) and np.linalg.eigvalsh(h).min()>0
    _,g,_=objective_gradient_hessian(x,labels,np.asarray([3.,1.,1.,1.]),[0.],0.)
    assert max(abs(g))<1e-15
    _,z=logit_rows([[2.,6.]], [1.,2.], [2.,4.], [3.,-1.], .2)
    assert abs(z[0]-.7)<1e-15
    assert sigmoid(1000.)==1. and sigmoid(-1000.)==0. and softplus(-1000.)==0.
    print(json.dumps({'synthetic_passed':True,'real_data_models_or_outcomes_read':False}))


def real(revision,expected_result):
    os.chdir(ROOT);verified={};maxima={};counts=Counter();models_report=[];weights_report=[]
    def verify(p,h):
        key=str(Path(p).resolve())
        if key not in verified:verified[key]=sha(p)
        assert verified[key]==h,('hash',str(p));counts['hash_binding_checks']+=1
    verify(OUT/'results.json',expected_result)
    reg,pre,result=[read(OUT/(name+'.json')) for name in ('registration','preflight','results')]
    cfg=yaml.safe_load(CONFIG.read_text())
    assert reg['source_revision']==revision and result['registration_sha256']==digest(reg) and reg['config']==cfg
    verify(CONFIG,reg['config_sha256']);verify(OUT/'preflight.json',cfg['preflight_sha256'])
    assert reg['preflight_sha256']==cfg['preflight_sha256']
    assert hashlib.sha256(subprocess.check_output(['git','show',revision+':'+str(CONFIG)])).hexdigest()==reg['config_sha256']
    assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert cfg['groups']==list(GROUPS) and cfg['development_folds']==list(FOLDS)
    assert (np.__version__,pd.__version__,sklearn.__version__)==('2.2.6','2.3.3','1.8.0')
    assert cfg['normalized_gradient_infinity_bound']==TOL['gradient_infinity']
    assert cfg['scalar_logit_atol']==TOL['logit_atol'] and cfg['scalar_probability_atol']==TOL['probability_atol']
    assert len(cfg['source_bindings'])==29 and len(pre['source_artifact_bindings'])==2120
    for p,h in cfg['source_bindings'].items():
        verify(p,h);assert hashlib.sha256(subprocess.check_output(['git','show',revision+':'+p])).hexdigest()==h
    for section in ('direct_source_bindings','source_artifact_bindings'):
        for p,h in pre[section].items():verify(p,h)
    fc=yaml.safe_load(Path('configs/oracle_frontier_20260905.yaml').read_text())
    verify(fc['data_path'],pre['spot_data_proof']['artifact_sha256'])
    cutoff=pd.Timestamp(cfg['data_cutoff']);assert cutoff==pd.Timestamp('2023-04-16T13:45:00Z')
    spot=pd.read_parquet(fc['data_path'],columns=['open','high','low','close'],filters=[('bar_open_ts','<',cutoff)])
    index=pd.date_range(spot.index[0],spot.index[-1],freq='15min',tz='UTC');spot=spot.reindex(index)
    assert index[-1]<cutoff;available=spot.notna().all(axis=1).to_numpy()
    runtime_bindings={};fit_records={r['fold']:r for r in result['fit_records']}
    assert len(fit_records)==len(result['fit_records'])==8
    for f in FOLDS:
        fold=read(OUT/f'fold_{f}.json');assert fold['registration_sha256']==digest(reg)
        assert len(fold['artifact_sha256'])==90
        for p,h in fold['artifact_sha256'].items():verify(p,h);runtime_bindings[p]=h
        data=arrays(OUT/'fit_data'/f'fold{f}_training.npz');record=read(OUT/'provenance'/f'fold{f}_fit.json')
        assert record==fit_records[f];prov=record['fit_provenance'];bound=next(s for s in pre['support'] if s['fold']==f)
        assert record['fit_source_binding']==bound and record['new_model_fits']==4 and record['shared_prior_estimates']==2
        expected_keys={'fit_positions','timestamps','returns','binary_labels','predict_positions','predict_timestamps'}
        expected_keys|={s+'_features_'+g for s in ('fit','predict') for g in GROUPS}|{'weights_'+w for w in WEIGHTINGS}
        assert set(data)==expected_keys
        fp,pp=data['fit_positions'],data['predict_positions'];nt=len(fp)
        assert nt>=512 and len(pp) and np.all(np.diff(fp)>0) and np.all(np.diff(pp)>0) and fp[-1]<pp[0]
        masks={}
        for name,positions,times in [('fit',fp,data['timestamps']),('predict',pp,data['predict_timestamps'])]:
            m=np.zeros(len(index),bool);m[positions]=True;masks[name]=m
            exact(index.asi8[positions],times,name+' timestamps')
            assert masksha(index,m)==bound['mask_sha256'][name]
            assert position_sha(m)==prov['mask_position_sha256'][name]
            assert len(positions)==bound['counts'][name]==prov['mask_counts'][name]
        E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3)
        assert (index[fp]+pd.Timedelta(minutes=375)<E-pd.DateOffset(months=6)).all()
        assert all(available[i+1:i+25].all() for i in fp)
        returns=np.log(spot.close.to_numpy()[fp+24]/spot.open.to_numpy()[fp+1])
        exact(data['returns'],returns,'T returns')
        assert digest(returns.tolist())==bound['fit_return_sha256'] and matrixsha(returns)==prov['fit_return_sha256']
        labels=np.asarray([int(v>0) for v in returns],dtype=np.int64)
        exact(labels,data['binary_labels'],'binary labels');assert matrixsha(labels)==prov['fit_binary_labels_sha256']
        absmean=math.fsum(abs(float(v))/nt for v in returns)
        assert record['fit_abs_return_mean']==absmean and record['fit_return_mean']==float(np.mean(returns))
        raw_logits={};raw_probability={};emitted_logits={};emitted_probability={}
        for weighting in WEIGHTINGS:
            weight=np.ones(nt) if weighting=='ordinary' else np.asarray([abs(float(v))/absmean for v in returns])
            exact(weight,data['weights_'+weighting],'normalized weights')
            total=math.fsum(float(v) for v in weight);positive=math.fsum(float(v) for v in weight[labels==1])
            prior=positive/total;prior_z=math.log(prior)-math.log1p(-prior)
            assert record['fit_priors'][weighting]==prior and record['fit_prior_logits'][weighting]==prior_z
            info=prov['sample_weights'][weighting]
            assert info['weight_sha256']==matrixsha(weight) and info['sum_fsum']==total and info['sum_numpy_used_by_solver']==float(np.sum(weight))
            assert info['zero_weight_rows']==int((weight==0).sum())
            weights_report.append({'fold':f,'weighting':weighting,'sum':total,'mean':total/nt,'maximum':max(weight),
                'zero_rows':int((weight==0).sum()),'squared_weight_effective_count':total**2/math.fsum(float(v)**2 for v in weight),
                'effective_count_is_not_independent_sample_size':True,'prior':prior,'prior_logit':prior_z})
            raw_logits['prior_'+weighting]=np.full(len(pp),prior_z)
            raw_probability['prior_'+weighting]=np.full(len(pp),sigmoid(prior_z))
        counts['fit_prior_estimates']+=2
        for group in GROUPS:
            xf,xp=(data[s+'_features_'+group] for s in ('fit','predict'))
            assert xf.shape==(nt,29 if group=='technical' else 31)
            for name,x in [('fit',xf),('predict',xp)]:
                assert np.isfinite(x).all() and digest(x.tolist())==bound[name+'_features_sha256'][group]
                assert matrixsha(x)==prov[name+'_features_sha256'][group]
            assert matrixsha(np.column_stack((xf,returns)))==prov['fit_features_and_return_sha256'][group]
            center=np.asarray([math.fsum(float(v)/nt for v in xf[:,j]) for j in range(xf.shape[1])])
            variance=np.asarray([math.fsum((float(v)-float(center[j]))**2/nt for v in xf[:,j]) for j in range(xf.shape[1])])
            epsilon=np.finfo(float).eps;constant=variance<=nt*epsilon*variance+(nt*center*epsilon)**2
            scale=np.sqrt(variance);scale[constant]=1.
            for weighting in WEIGHTINGS:
                mid=group+'_'+weighting;model=joblib.load(OUT/'models'/f'fold{f}_{mid}.joblib')
                assert [k for k,_ in model.steps]==['standardscaler','logisticregression']
                scaler,lr=model.steps[0][1],model.steps[1][1];state=prov['fitted_state'][mid]
                assert lr.get_params()==prov['parameters']['logistic']
                for k,v in {'C':1.,'l1_ratio':0.,'solver':'lbfgs','tol':1e-8,'max_iter':1000,'fit_intercept':True,'random_state':20260906}.items():assert lr.get_params()[k]==v
                assert lr.class_weight is None and not lr.warm_start and int(scaler.n_samples_seen_)==nt
                assert np.array_equal(lr.classes_,[0,1]) and np.asarray(lr.n_iter_).shape==(1,) and 0<=int(lr.n_iter_[0])<1000
                for expected,saved,name in [(center,scaler.mean_,'scaler_mean'),(variance,scaler.var_,'scaler_variance'),(scale,scaler.scale_,'scaler_scale')]:
                    near(expected,saved,name,maxima,TOL['scaler_atol'],TOL['scaler_rtol'])
                    assert list(saved)==state[name]
                assert lr.coef_.tolist()==state['coefficient'] and lr.intercept_.tolist()==state['intercept']
                assert matrixsha(lr.coef_)==state['coefficient_sha256'] and matrixsha(lr.intercept_)==state['intercept_sha256']
                beta,intercept=lr.coef_[0],float(lr.intercept_[0]);weight=data['weights_'+weighting]
                assert matrixsha(np.column_stack((xf,labels,weight)))==state['fit_features_labels_weights_sha256']
                transformed,_=logit_rows(xf,scaler.mean_,scaler.scale_,beta,intercept)
                objective,gradient,hessian=objective_gradient_hessian(transformed,labels,weight,beta,intercept)
                gradmax=float(np.max(np.abs(gradient)));assert gradmax<=TOL['gradient_infinity']
                eigenmin=float(np.linalg.eigvalsh(hessian).min());assert eigenmin>=TOL['hessian_eigenvalue_floor']
                check=state['scalar_verification'];assert check['checked'] and prov['objective']['stationarity_checked_by_this_fit_helper']
                near(objective,check['normalized_objective'],'runtime_scalar_objective',maxima,TOL['runtime_gradient_objective_atol'])
                near(gradient,check['normalized_gradient'],'runtime_scalar_gradient',maxima,TOL['runtime_gradient_objective_atol'])
                _,z=logit_rows(xp,scaler.mean_,scaler.scale_,beta,intercept)
                raw_logits[mid]=np.asarray(z);raw_probability[mid]=np.asarray([sigmoid(v) for v in z])
                emitted_logits[mid]=np.full(len(pp),np.nan);emitted_probability[mid]=np.full(len(pp),np.nan)
                models_report.append({'fold':f,'model_id':mid,'fit_rows':nt,'predict_rows':len(pp),
                    'normalized_objective':objective,'gradient_infinity':gradmax,'minimum_hessian_eigenvalue':eigenmin,
                    'runtime_scalar_objective_difference':abs(objective-check['normalized_objective']),
                    'runtime_scalar_gradient_max_difference':float(np.max(np.abs(gradient-check['normalized_gradient']))),
                    'model_sha256':sha(OUT/'models'/f'fold{f}_{mid}.joblib')})
                counts['models']+=1;counts['fit_model_rows']+=nt;counts['predict_model_rows']+=len(pp)
        for mean,mp in cfg['mapping'].items():
            mid=mp['classifier_id'];group=mp['group'];parent=arrays(PARITY/'forecasts'/f'fold{f}_{group}_half.npz')
            oldcal=arrays(PARITY/'calibration'/f'fold{f}_{group}.npz')
            oldprov=read(PARITY/'calibration'/f'fold{f}_provenance.json')['calibration']
            for kind in ('forecasts','calibration'):
                a=arrays(OUT/kind/f'fold{f}_{mean}.npz');times=pd.DatetimeIndex(pd.to_datetime(a['timestamps'],utc=True))
                selected=np.asarray((index[pp]>=times[0])&(index[pp]<=times[-1]))
                local=times.get_indexer(index[pp][selected]);assert (local>=0).all()
                predicted=np.zeros(len(times),bool);predicted[local]=True
                exact(np.isfinite(a['logit']),predicted,'raw prediction support')
                exact(np.isfinite(a['probability']),predicted,'probability support')
                near(a['logit'][local],raw_logits[mid][selected],'scalar_logits',maxima,TOL['logit_atol'])
                near(a['probability'][local],raw_probability[mid][selected],'scalar_probabilities',maxima,TOL['probability_atol'])
                exact(np.sign(a['logit'][local]),np.sign(raw_logits[mid][selected]),'scalar direction signs')
                if mid in emitted_logits:
                    emitted_logits[mid][selected]=a['logit'][local]
                    emitted_probability[mid][selected]=a['probability'][local]
                if kind=='forecasts':
                    for k in parent:
                        if k!='mu':exact(a[k],parent[k],'E inherited '+k)
                    exact(a['parent_mu'],parent['mu'],'E parent half');support=parent['inference_mask']
                else:
                    for k in ('timestamps','actual','scale_mask','interval_mask'):exact(a[k],oldcal[k],'cal inherited '+k)
                    exact(a['classifier_predict_mask'],predicted,'cal raw support')
                    support=predicted&np.asarray(times>=E-pd.DateOffset(months=3))
                    exact(a['mapped_inference_mask'],support,'I mapping support')
                    expected=np.full(len(times),np.nan)
                    expected[support]=.5*float(oldprov['scale_mean'])+.5*(oldcal['mu'][support]+float(oldprov['return_bias'][group]))
                    exact(a['parent_mu'],expected,'I parent reconstruction')
                    assert np.isnan(a['mu'][times<E-pd.DateOffset(months=3)]).all()
                mapped=np.full(len(times),np.nan)
                for i in np.flatnonzero(support):mapped[i]=(1. if a['logit'][i]>0 else -1. if a['logit'][i]<0 else 0.)*abs(float(a['parent_mu'][i]))
                exact(a['mu'],mapped,'fixed surrogate mapping');counts[kind+'_npz']+=1
        for mid in emitted_logits:
            assert np.isfinite(emitted_logits[mid]).all() and np.isfinite(emitted_probability[mid]).all()
            state=prov['fitted_state'][mid]
            assert matrixsha(emitted_logits[mid])==state['predict_logits_sha256']
            assert matrixsha(emitted_probability[mid])==state['predict_probability_sha256']
            counts['complete_emitted_prediction_hash_pairs']+=1
        counts['fit_data_and_provenance_pairs']+=1
    assert counts['models']==32 and counts['fit_prior_estimates']==16
    assert counts['forecasts_npz']==counts['calibration_npz']==64 and len(runtime_bindings)==720
    assert {r['segment'] for r in result['scores']}=={r['segment'] for r in result['classification_scores']}=={'interval','evaluation'}
    assert result['new_model_fits']==32 and result['risk_model_or_calibration_fits']==0 and not result['selection_performed']
    log=OUT/'run.log';logtext=log.read_text();runtime_warnings=Counter()
    for line in logtext.splitlines():
        if 'RuntimeWarning:' in line:runtime_warnings[line.split('RuntimeWarning:',1)[1].strip()]+=1
    completed=[json.loads(line) for line in logtext.splitlines() if line.startswith('{')]
    assert [v['fold'] for v in completed if v.get('event')=='fold_complete']==list(FOLDS)
    report={'passed':True,'scope':'Independent scalar saved-model fitting objective, stationarity, predictor and surrogate support audit; no refit or policy execution',
        'script_sha256':sha(__file__),'registered_revision':revision,'counts':dict(counts),'tolerances':TOL,
        'max_absolute_differences':maxima,'maximum_gradient_infinity':max(r['gradient_infinity'] for r in models_report),
        'minimum_hessian_eigenvalue':min(r['minimum_hessian_eigenvalue'] for r in models_report),
        'models':models_report,'fit_weight_diagnostics':weights_report,'source_bindings':cfg['source_bindings'],
        'direct_file_bindings':{str(p):sha(p) for p in (CONFIG,OUT/'registration.json',OUT/'preflight.json',OUT/'results.json')},
        'source_artifacts':2120,'runtime_artifacts':720,'distinct_hashed_files':len(verified),
        'source_artifact_inventory_sha256':digest(pre['source_artifact_bindings']),'runtime_artifact_inventory_sha256':digest(runtime_bindings),
        'runtime_log':{'path':str(log),'sha256':sha(log),'runtime_warning_categories_and_counts':dict(runtime_warnings),
            'warning_cause_established':False,'finite_final_state_and_stationarity_verified':True},
        'no_S_scores_or_new_calibration':True,'limitations':['Stationarity and numerical agreement do not establish forecast skill or convergence to a scientifically optimal model.',
            'Feature matrices are independently tied to preflight hashes, not newly recomputed by this audit; T returns are separately reconstructed on bounded Spot.',
            'Classifiers use frozen original T rows and retrospective shared support. Reused development and historical receipt limitations persist.',
            'Scalar agreement does not establish the cause of any observed RuntimeWarning. Classification/return score aggregation and own-state accounts are separately audited.']}
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'path':str(REPORT),'sha256':sha(REPORT),'passed':True,'counts':dict(counts),'maxima':maxima,
                      'maximum_gradient_infinity':report['maximum_gradient_infinity']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--real-after-root-go',action='store_true')
    p.add_argument('--registered-revision');p.add_argument('--expected-results-sha256');args=p.parse_args()
    if args.real_after_root_go:
        if not args.registered_revision or not args.expected_results_sha256:p.error('Root registered revision and completed result SHA are mandatory.')
        real(args.registered_revision,args.expected_results_sha256)
    else:synthetic()
