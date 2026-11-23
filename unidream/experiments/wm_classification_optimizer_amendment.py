"""Explicit pre-Test numerical-optimization amendment for the LR baseline.

The original registered experiment remains immutable. This separately frozen
stage replaces only the LR optimizer with an invertible, Train-fitted change of
coordinates. It retains the original worker's data access, candidate grid,
validation selector, LightGBM fitting and artifact format. Its original-space
objective and convergence certificate are recorded with every fit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time

from .wm_technical_ablation import _json, _sha
from .wm_volatility_classification import DEFAULT_CONFIG, load_config, verify


_ACTIVE_AMENDMENT = None


def source_hashes():
    directory=Path(__file__).parent
    return {name:_sha(directory/name) for name in
        ('wm_classification_optimizer_amendment.py','wm_classification_logistic_optimizer.py')}


def register(cfg):
    verify(cfg)
    out=Path(cfg['output_dir'])
    if (out/'optimizer_amendment.json').exists(): raise ValueError('amendment already frozen')
    if (out/'test_access_barrier.json').exists() or list((out/'predictions').glob('*.parquet')):
        raise ValueError('optimizer amendment must precede all new Test access')
    if list((out/'classical').glob('*/fit.json')):
        raise ValueError('archive all superseded classical fits before uniform optimizer amendment')
    meta={'registered_utc':datetime.now(timezone.utc).isoformat(),
        'original_preregistration_sha256':_sha(out/'preregistration.json'),
        'config_sha256':_sha(cfg['config_path']),'source_sha256':source_hashes(),
        'reason':'Dense 4909-column multicollinearity made the original L-BFGS candidate fits very slow. A Train-only Newton-CG timing probe also exceeded 180 seconds. No new Test predictions or performance were accessed.',
        'changes':'Numerical optimizer only: full-rank Train covariance preconditioning of multinomial L2 logistic regression. No eigenvector is removed; no new penalty is added.',
        'objective':'mean unweighted multinomial NLL + ||W||^2/(2*C*NTrain), with an unpenalized intercept, in the ORIGINAL input coordinates.',
        'transformation':'Xc=X-meanTrain; Xc.T Xc/N=V diag(e) V.T; lambda=1/(C*N); s=sqrt(max(e,0)+5*lambda); Z=Xc V / s; W=V(B/s), intercept=b-meanTrain W; transformed penalty=lambda*sum(B^2/s^2)/2.',
        'convergence':'Each candidate gets up to the original configured 1000 iterations in preconditioned coordinates. Certify finite original-space penalized gradient, including original intercept, with Linf <= original configured tol. Failure fails the task; no C candidate is silently removed. Record optimizer status, objective, gradient norm and numerical decomposition diagnostics.',
        'unchanged':{'C_candidates':cfg['tabular']['logistic_Cs'],'tol':cfg['tabular']['logistic_tol'],
            'initial_max_iter':cfg['tabular']['logistic_max_iter'],
            'data':'all 4909 features, same Train/Val/Test, same fixed labels/theta/priors and Train-only normalization',
            'selection':'highest fixed-panel Val Balanced Accuracy, ties lowest shared log loss',
            'lightgbm':'same original registered candidates and fitting implementation',
            'neural':'all existing neural workers/checkpoints and selection rules unchanged',
            'statistics':'same ten BA hypotheses per condition, primary lgbm condition, paired block bootstrap and Holm'},
        'application':'Apply uniformly to every one of the 45 LR tasks. Superseded incomplete optimizer attempts are archived and do not contribute predictions.',
        'test_access_before_amendment':False,'new_test_barrier_existed':False}
    _json(out/'optimizer_amendment.json',meta)
    return meta


def verify_amendment(cfg):
    out=Path(cfg['output_dir']); path=out/'optimizer_amendment.json'
    meta=json.loads(path.read_text())
    if (meta['original_preregistration_sha256']!=_sha(out/'preregistration.json')
            or meta['config_sha256']!=_sha(cfg['config_path']) or meta['source_sha256']!=source_hashes()):
        raise ValueError('amended optimizer source/config/provenance changed')
    if (out/'test_access_barrier.json').exists():
        barrier=json.loads((out/'test_access_barrier.json').read_text())
        if datetime.fromisoformat(meta['registered_utc'])>=datetime.fromisoformat(barrier['time_utc']):
            raise ValueError('optimizer amendment was not frozen before Test')
    return meta,_sha(path)


def _registered_logistic(x,labels,xv,labels_val,priors,config):
    from .wm_classification_logistic_optimizer import fit_logistic_candidates_preconditioned
    if _ACTIVE_AMENDMENT is None: raise RuntimeError('no registered optimizer amendment')
    model,history=fit_logistic_candidates_preconditioned(x,labels,xv,labels_val,priors,config)
    meta,sha=_ACTIVE_AMENDMENT
    history.update(optimizer_amendment_sha256=sha,optimizer_source_sha256=meta['source_sha256'],
        implementation='full-rank Train covariance preconditioned L-BFGS; original multinomial L2 objective')
    return model,history


def fit_amended_task(config_path,task):
    global _ACTIVE_AMENDMENT
    from . import wm_classification_classical as classical
    cfg=load_config(config_path); _ACTIVE_AMENDMENT=verify_amendment(cfg)
    fit_path=Path(cfg['output_dir'])/'classical'/task['job']/'fit.json'
    if fit_path.exists():
        existing=json.loads(fit_path.read_text())
        if existing['logistic'].get('optimizer_amendment_sha256')!=_ACTIVE_AMENDMENT[1]:
            raise ValueError('cannot mix original and amended LR fits')
    # This is an explicit stage-level replacement, registered above. Keeping
    # the original surrounding worker preserves its strict Train/Val-only
    # contract and the original LightGBM implementation exactly.
    classical.fit_logistic_candidates=_registered_logistic
    return classical.fit_classical_task(config_path,task)


def run(cfg):
    _,frozen=verify(cfg); verify_amendment(cfg)
    out=Path(cfg['output_dir']); tasks=frozen['classical_tasks']; errors=[]; started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(fit_amended_task,cfg['config_path'],task):task for task in tasks}
        for future in as_completed(futures):
            try: print(json.dumps(future.result()),flush=True)
            except Exception as exc: errors.append({'task':futures[future],'error':repr(exc)})
            _json(out/'classical_status.json',{'status':'failed' if errors else 'running',
                'completed':sum((out/'classical'/task['job']/'fit.json').exists() for task in tasks),
                'total':len(tasks),'errors':errors,'optimizer_amendment_sha256':_sha(out/'optimizer_amendment.json'),
                'elapsed_seconds':time.monotonic()-started})
    _json(out/'classical_status.json',{'status':'failed' if errors else 'fits_completed',
        'completed':sum((out/'classical'/task['job']/'fit.json').exists() for task in tasks),
        'total':len(tasks),'errors':errors,'optimizer_amendment_sha256':_sha(out/'optimizer_amendment.json'),
        'elapsed_seconds':time.monotonic()-started})
    if errors: raise RuntimeError(errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=('register','run'))
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    args=parser.parse_args();cfg=load_config(args.config)
    if args.command=='register':print(json.dumps(register(cfg)),flush=True)
    else:run(cfg)


if __name__=='__main__':main()
