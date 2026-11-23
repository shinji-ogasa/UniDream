"""Causal variance-threshold classification; isolated retrospective research.

Source forecasters are frozen before common labels are formed. Classifiers are
independently fitted for each threshold condition, using train-only class priors
and validation-only selection. Test predictions require a global fit barrier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .wm_technical_ablation import ROOT, _arrays, _features, _input, _json, _load_config, _sha, _verify_freeze
from . import wm_model_comparison as comparison
from .wm_classification_models import ForecastClassifier
from .wm_classification_evaluation import (make_labels, calibrate_thresholds,
    variance_thresholds, score_classification, balanced_prior_predictions,
    paired_classification_block_bootstrap)
from .wm_ablation_evaluation import holm_adjust

DEFAULT_CONFIG = ROOT / 'configs/experiments/wm_volatility_classification_20261110.yaml'


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['test_selection_allowed'] is not False or cfg['primary_horizon'] != 16
            or cfg['k'] != .5 or cfg['primary_condition'] != 'lgbm'
            or cfg['conditions'] != ['lgbm', 'wm', 'fixed']):
        raise ValueError('fixed four-hour classification contract required')
    cfg['config_path'] = str(Path(path).resolve())
    for key in ('output_dir', 'ablation_config', 'comparison_config'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def parents(cfg):
    return _load_config(cfg['ablation_config']), comparison.load_config(cfg['comparison_config'])


def source_hashes():
    names = ('wm_volatility_classification.py', 'wm_classification_models.py',
             'wm_classification_classical.py', 'wm_classification_evaluation.py',
             'wm_classification_forecasters.py')
    return {name: _sha(Path(__file__).parent / name) for name in names}


def fold_data(cfg, fold):
    return dict(np.load(Path(cfg['output_dir']) / 'prepared' / f'fold_{fold:02d}.npz'))


def get_design(cfg, ids, data):
    cache=Path(cfg['output_dir'])/'cache'
    decisions=np.load(cache/'decisions.npy',mmap_mode='r'); positions=np.searchsorted(decisions,ids)
    if np.any(positions>=len(decisions)) or not np.array_equal(decisions[positions],ids):
        raise ValueError('missing full-history tabular decisions')
    raw=np.load(cache/'tabular_raw.npy',mmap_mode='r')
    mean,std=tabular_scaler(data)
    result=(raw[positions]-mean)/std
    if not np.isfinite(result).all(): raise ValueError('nonfinite tabular input')
    return result


def tabular_design(windows):
    """All OHLCV/market history plus explicit last-bar deterministic TA."""
    if windows.ndim!=3 or windows.shape[1:]!=(256,63): raise ValueError('expected full D context')
    return np.concatenate((windows[...,:19].reshape(len(windows),-1),windows[:,-1,19:]),axis=1)


def tabular_scaler(data):
    return tuple(np.concatenate((np.tile(data[name][:19],256),data[name][19:])) for name in ('mean','std'))


def _normalization(arrays, train, parent, fold_id):
    rng = np.random.default_rng(parent['training']['normalization_seed'] + fold_id)
    chosen = rng.choice(train, min(len(train), parent['training']['normalization_windows']), replace=False)
    total = np.zeros(63, np.float64); square = np.zeros(63, np.float64); count = 0
    for start in range(0, len(chosen), 32):
        values = _features(arrays['raw'], arrays['market'], chosen[start:start+32], 256).astype(np.float64)
        total += values.sum((0, 1)); square += (values*values).sum((0, 1)); count += values.shape[0]*values.shape[1]
    mean = total/count
    std = np.sqrt(np.maximum(square/count-mean**2, 0)); std[std < 1e-7] = 1
    return mean.astype(np.float32), std.astype(np.float32)


def audit_oof_inputs(forecasts, index_ns, needed_ids, metadata):
    """Require exact input-only source coverage with selection before decisions."""
    ids = forecasts.decision_id.to_numpy(dtype=np.int64)
    if forecasts.decision_id.duplicated().any() or not np.array_equal(ids, needed_ids):
        raise ValueError('OOF support differs from frozen needed decisions')
    if not np.array_equal(pd.to_datetime(forecasts.timestamp, utc=True).astype('int64').to_numpy(), index_ns[ids]):
        raise ValueError('OOF decision timestamp mismatch')
    for info in metadata:
        part = forecasts.loc[forecasts.quarter_id == info['quarter_id']]
        if len(part) != info['n'] or not len(part): raise ValueError('OOF quarter coverage mismatch')
        if (not info['forecast_decision_strictly_after_model_selection_end']
                or info['forecast_quarter_labels_read']
                or index_ns[part.decision_id.to_numpy()].min() <= info['model_selection_end_ns']):
            raise ValueError('OOF source used future selection information')
    if set(forecasts.quarter_id.unique()) != {info['quarter_id'] for info in metadata}:
        raise ValueError('OOF metadata quarter mismatch')


def prepare(cfg):
    parent, old = parents(cfg)
    _verify_freeze(parent); comparison.verify(old)
    out = Path(cfg['output_dir'])
    if (out / 'preregistration.json').exists(): raise ValueError('already registered')
    plan = json.loads((out / 'forecasters/plan.json').read_text())
    # The source stage is a separately frozen dependency, never refitted using
    # classifier outcomes. All source fits/predictions must exist at this point.
    paths = sorted((out / 'forecasters/oof').glob('quarter_*.parquet'))
    if len(paths) != len(plan['prediction_tasks']):
        raise ValueError(f'incomplete OOF forecasts: {len(paths)}')
    forecasts = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values('decision_id')
    if forecasts.decision_id.duplicated().any(): raise ValueError('duplicate forecast decisions')
    for key in ('vhat_wm', 'vhat_lgbm'):
        if not np.isfinite(forecasts[key]).all() or not (forecasts[key] > 0).all():
            raise ValueError(f'invalid OOF forecasts: {key}')
    arrays = _arrays(parent['output_dir']); n = len(arrays['index_ns'])
    metadata = [json.loads(p.with_suffix('.json').read_text()) for p in paths]
    for path, info in zip(paths, metadata):
        if _sha(path) != info['prediction_sha256']: raise ValueError('OOF artifact changed')
    audit_oof_inputs(forecasts, arrays['index_ns'], np.load(plan['needed_ids_path']), metadata)
    cache=out/'cache'; cache.mkdir(exist_ok=True)
    decisions=forecasts.decision_id.to_numpy(dtype=np.int64); np.save(cache/'decisions.npy',decisions)
    design=np.lib.format.open_memmap(cache/'tabular_raw.npy',mode='w+',dtype='float32',shape=(len(decisions),4908))
    for start in range(0,len(decisions),128):
        ids=decisions[start:start+128]
        design[start:start+len(ids)]=tabular_design(_features(arrays['raw'],arrays['market'],ids,256))
    design.flush(); del design
    source = {}
    for name in ('lgbm', 'wm'):
        value = np.full(n, np.nan); value[forecasts.decision_id.to_numpy()] = forecasts['vhat_'+name]
        source[name] = value
    available = np.isfinite(source['lgbm']) & np.isfinite(source['wm'])
    old_folds = json.loads((Path(parent['output_dir']) / 'prepared/folds.json').read_text())
    directory = out / 'prepared'; directory.mkdir(exist_ok=True)
    folds = []
    for previous in old_folds:
        f = previous['fold']
        old_data = dict(np.load(Path(parent['output_dir']) / 'prepared' / f'fold_{f:02d}.npz'))
        data = {key: old_data[key][available[old_data[key]]] for key in ('train', 'val', 'test')}
        # New forecasting support may shorten early Train; evaluation coverage
        # stays exactly equal to the original immutable outer folds.
        for key in ('val', 'test'):
            if not np.array_equal(data[key], old_data[key]): raise ValueError(f'{key} support changed')
        train = data['train']; returns = arrays['returns'][train, 1]
        if len(train) < 10000: raise ValueError('insufficient common OOF Train')
        data['mean'], data['std'] = _normalization(arrays, train, parent, f)
        rows = min(len(data['val']), cfg['neural']['selection_rows'])
        data['selection_val'] = data['val'][np.linspace(0, len(data['val'])-1, rows, dtype=int)]
        rms = float(np.sqrt(np.mean(returns**2)))
        fold = {key: value for key,value in previous.items() if key not in ('scales','support','crash_train_prevalence')}
        fold.update(return_rms=rms, original_train_n=len(old_data['train']), train_n=len(train), priors={}, calibration={}, support={})
        fold['scales'] = {
            'return': np.maximum(np.sqrt(np.mean(arrays['returns'][train]**2, axis=0)), 1e-4).tolist(),
            'variance': np.maximum(np.mean(arrays['variance'][train], axis=0), 1e-8).tolist(),
            'downside': np.maximum(np.sqrt(np.mean(arrays['downside'][train]**2, axis=0)), 1e-4).tolist()}
        for name in cfg['conditions']:
            if name == 'fixed':
                theta = np.full(n, cfg['k']*rms)
                calibration = {'fixed_theta': cfg['k']*rms, 'n_train': len(train)}
            else:
                calibration = calibrate_thresholds(returns, source[name][train], train_returns=returns)
                theta = np.full(n, np.nan)
                theta[available] = variance_thresholds(source[name][available], calibration['a'])
            labels = np.full(n, -1, np.int8)
            needed = np.unique(np.concatenate([data[key] for key in ('train','val','test')]))
            labels[needed] = make_labels(arrays['returns'][needed, 1], theta[needed])
            priors = np.bincount(labels[train], minlength=3)/len(train)
            if not (priors > 0).all(): raise ValueError('missing Train class')
            if not np.array_equal(np.unique(labels[data['selection_val']]), np.arange(3)):
                raise ValueError('fixed validation panel does not cover three classes')
            data['theta_'+name] = theta
            data['label_'+name] = labels
            fold['priors'][name] = priors.tolist(); fold['calibration'][name] = calibration
            # Only Train calibration distribution is examined before fitting.
            z = returns/(theta[train]/cfg['k'])
            fold['calibration'][name]['train_standardized_return_quantiles'] = np.quantile(z,[.01,.05,.25,.5,.75,.95,.99]).tolist()
        for split in ('train','val','test'):
            ids = data[split]; end = pd.Timestamp(fold[split+'_end']).value
            if not np.all(arrays['index_ns'][ids] + 64*15*60*10**9 < end): raise ValueError('cross-boundary labels')
            fold['support'][split] = {'n':len(ids), 'first':str(pd.Timestamp(arrays['index_ns'][ids[0]], tz='UTC')), 'last':str(pd.Timestamp(arrays['index_ns'][ids[-1]], tz='UTC'))}
        np.savez(directory / f'fold_{f:02d}.npz', **data)
        folds.append(fold)
        print(f'Prepared classification fold {f}: Train {len(train)} (original {len(old_data["train"])})', flush=True)
    _json(directory / 'folds.json', folds)
    tasks = [{'condition':condition, 'fold':f['fold'], 'model':model, 'seed':seed,
        'job':f"fold_{f['fold']:02d}_{condition}_{model}_s{seed}"}
        for condition in cfg['conditions'] for f in folds for seed in cfg['seeds'] for model in cfg['neural']['models']]
    classical = [{'condition':condition, 'fold':f['fold'], 'job':f"fold_{f['fold']:02d}_{condition}"}
        for condition in cfg['conditions'] for f in folds]
    frozen = {'config':cfg, 'config_sha256':_sha(cfg['config_path']), 'registered_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':source_hashes(), 'prepared_sha256':{p.name:_sha(p) for p in sorted(directory.iterdir())},
        'cache_sha256':{p.name:_sha(p) for p in sorted(cache.iterdir())},
        'source_forecast_sha256':{str(p.relative_to(out)):_sha(p) for p in paths},
        'forecaster_metadata_sha256':{str(p.relative_to(out)):_sha(p) for p in
            sorted((out/'forecasters').rglob('*')) if p.is_file() and p.suffix in ('.json','.npy','.npz','.pt','.txt') and not p.name.endswith('_training.log')},
        'forecaster_registration_sha256':_sha(out/'forecasters/preregistration.json'),
        'parent_registration_sha256':_sha(Path(parent['output_dir'])/'preregistration.json'),
        'comparison_registration_sha256':_sha(Path(old['output_dir'])/'preregistration.json'),
        'regression_controls_sha256':_sha(Path(old['output_dir'])/'predictions.parquet'),
        'tasks':tasks, 'classical_tasks':classical,
        'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','pandas','scipy','lightgbm','scikit-learn')},
        'runtime':{'platform':platform.platform(),'device':cfg['device'],
            'lightgbm_library_sha256':_sha(ROOT/'.venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib'),
            'openmp_library_sha256':_sha(ROOT/'.venv/lib/python3.12/site-packages/torch/lib/libomp.dylib')},
        'label':'0 down R<-theta; 1 flat inclusive; 2 up R>theta; four-hour close-to-close log return',
        'calibration':'k=.5 fixed. a=mean(Train OOF return^2 / predicted RV). theta=.5 sqrt(a*vhat). Fixed theta=.5 Train return RMS. No Gaussian assumption.',
        'information':'D: same 256 completed bars, 63 channels. Additional current theta side feature is log(theta/Train return RMS), supplied only to last neural classification head or appended tabular vector.',
        'tabular_information':'4908 = 256 x (base5 + all market14) time-major plus final TA44; current theta side makes 4909. All original OHLCV/market history is retained. Neural uses all63 at all256 tokens, including deterministic TA history. Source variance forecasters and old regression controls retain their frozen older 1394-column recipe.',
        'classification':'unweighted CE. Primary decisions argmax(p / Train class priors), raw argmax also reported. Classifiers separately fitted per condition. WM retains full frozen auxiliary objective plus last-token CE.',
        'selection':'1536 fixed evenly-spaced Val decisions; maximize Balanced Accuracy; ties minimize log loss. Neural checks every 200, significant BA gain .001 for patience, true best selected independently.',
        'test_access':'All selections frozen before new Test predictions. Historical Test already inspected in prior research: retrospective OOS, not a pristine holdout.',
        'comparisons':'Per condition: six pairwise contrasts among four classifiers plus each versus Train-majority, ten primary BA tests Holm corrected. lgbm condition primary; wm/fixed exploratory. Do not rank raw Accuracy across different label definitions.',
        'controls':'Old regression predictions thresholded descriptively; their historical Train/selection recipe differs. EWMA/GARCH forecast magnitude only and are not direction classifiers.'}
    frozen['direction_diagnostic'] = ('Descriptive only: on true down/up outcomes, report AUROC of p_up/(p_down+p_up) '
        '(zero denominator -> .5), plus two-class BA using p_up>p_down (ties down). No additional selection or primary hypothesis. '
        'Three-class BA alone can improve through flat detection and does not establish directional skill.')
    _json(out/'preregistration.json', frozen)
    print(f'Registered {len(tasks)} neural fits, {len(classical)*2} classical fits.', flush=True)


def verify(cfg):
    parent, old = parents(cfg); _verify_freeze(parent); comparison.verify(old)
    out = Path(cfg['output_dir']); frozen = json.loads((out/'preregistration.json').read_text())
    if frozen['config_sha256'] != _sha(cfg['config_path']) or frozen['source_sha256'] != source_hashes(): raise ValueError('classification source/config changed')
    if (frozen['parent_registration_sha256'] != _sha(Path(parent['output_dir'])/'preregistration.json')
            or frozen['comparison_registration_sha256'] != _sha(Path(old['output_dir'])/'preregistration.json')
            or frozen['regression_controls_sha256'] != _sha(Path(old['output_dir'])/'predictions.parquet')):
        raise ValueError('registered parent/control dependency changed')
    for name,sha in frozen['prepared_sha256'].items():
        if _sha(out/'prepared'/name) != sha: raise ValueError('prepared data changed')
    for name,sha in frozen['cache_sha256'].items():
        if _sha(out/'cache'/name) != sha: raise ValueError('tabular cache changed')
    for name,sha in frozen['source_forecast_sha256'].items():
        if _sha(out/name) != sha: raise ValueError('source predictions changed')
    for name,sha in frozen['forecaster_metadata_sha256'].items():
        if _sha(out/name) != sha: raise ValueError('source forecaster metadata/artifact changed')
    if _sha(out/'forecasters/preregistration.json') != frozen['forecaster_registration_sha256']: raise ValueError('forecaster registration changed')
    for name,version in frozen['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('package changed')
    for name,path in (('lightgbm_library_sha256',ROOT/'.venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib'),
                      ('openmp_library_sha256',ROOT/'.venv/lib/python3.12/site-packages/torch/lib/libomp.dylib')):
        if _sha(path)!=frozen['runtime'][name]: raise ValueError('OpenMP/LightGBM runtime changed')
    return parent, frozen


def prediction_proba(cfg, model, arrays, ids, data, fold, condition):
    parent,_ = parents(cfg); model.eval(); result=[]
    theta = data['theta_'+condition]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            chosen=ids[start:start+128]
            x,_ = _input(arrays, chosen, data, 'D', 256)
            side=np.log(theta[chosen]/fold['return_rms']).astype(np.float32)[:,None]
            result.append(model.predict_proba(torch.from_numpy(x).to(cfg['device']),torch.from_numpy(side).to(cfg['device'])).cpu().numpy())
    result=np.concatenate(result)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1),1,atol=1e-5): raise ValueError('invalid probabilities')
    return result


def selection_key(score):
    return float(score['balanced_accuracy']), -float(score['log_loss'])


def train_task(config_path, task):
    cfg=load_config(config_path); parent,_=parents(cfg); out=Path(cfg['output_dir'])
    directory=out/'neural'/task['job']; directory.mkdir(parents=True,exist_ok=True)
    reg=_sha(out/'preregistration.json')
    if (directory/'fit.json').exists():
        previous=json.loads((directory/'fit.json').read_text())
        if previous['preregistration_sha256'] != reg: raise ValueError('stale fit')
        return previous
    with (directory/'training.log').open('w',buffering=1) as stream,redirect_stdout(stream),redirect_stderr(stream):
        started=time.monotonic(); torch.set_num_threads(cfg['neural_threads']); torch.use_deterministic_algorithms(True)
        random.seed(task['seed']); np.random.seed(task['seed']); torch.manual_seed(task['seed'])
        if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
        arrays=_arrays(parent['output_dir']); data=fold_data(cfg,task['fold'])
        fold=json.loads((out/'prepared/folds.json').read_text())[task['fold']]; condition=task['condition']
        priors=np.array(fold['priors'][condition]); labels=data['label_'+condition]; theta=data['theta_'+condition]
        model=ForecastClassifier(task['model'],63,parent['model'],fold['scales']).to(cfg['device'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=parent['training']['learning_rate'],weight_decay=parent['training']['weight_decay'])
        rng=np.random.default_rng(task['seed']); diagnostic_rng=np.random.default_rng(cfg['neural']['diagnostic_seed']+task['fold'])
        train_panel=np.sort(diagnostic_rng.choice(data['train'],cfg['neural']['diagnostic_rows'],replace=False))
        val_ids=data['selection_val']; best=(-float('inf'),-float('inf')); best_state=None; best_step=0
        anchor=-float('inf'); last_significant=0; history=[]; reason='max_steps'; class_counts=np.zeros(3,np.int64)
        train_start=pd.Timestamp(fold['train_start']).value; train_end=pd.Timestamp(fold['train_end']).value
        for step in range(1,cfg['neural']['max_steps']+1):
            model.train(); ids=rng.choice(data['train'],parent['training']['batch_size'],replace=True)
            x,base=_input(arrays,ids,data,'D',256)
            positions=ids[:,None]-256+np.arange(1,257)[None,:]
            ends=arrays['index_ns'][positions,None]+np.array((4,16,64))[None,None,:]*15*60*10**9
            valid=arrays['target_mask'][positions] & (ends<train_end) & (arrays['index_ns'][positions,None]>=train_start)
            targets={key:torch.from_numpy(arrays[name][positions].astype(np.float32)).to(cfg['device']) for key,name in (('return','returns'),('variance','variance'),('downside','downside'),('crash','crash'))}
            reward=np.log(arrays['raw'][positions,3]/arrays['raw'][positions-1,3]).astype(np.float32)
            side=np.log(theta[ids]/fold['return_rms']).astype(np.float32)[:,None]
            class_counts+=np.bincount(labels[ids],minlength=3)
            optimizer.zero_grad(set_to_none=True)
            loss,parts=model.loss(torch.from_numpy(x).to(cfg['device']),torch.from_numpy(base).to(cfg['device']),targets,
                torch.from_numpy(valid).to(cfg['device']),torch.from_numpy(reward).to(cfg['device']),
                torch.from_numpy(labels[ids].astype(np.int64)).to(cfg['device']),torch.from_numpy(side).to(cfg['device']))
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),parent['training']['gradient_clip'],error_if_nonfinite=True); optimizer.step()
            if step%cfg['neural']['validation_every']==0:
                score=score_classification(labels[val_ids],prediction_proba(cfg,model,arrays,val_ids,data,fold,condition),priors)
                train=score_classification(labels[train_panel],prediction_proba(cfg,model,arrays,train_panel,data,fold,condition),priors)
                key=selection_key(score)
                row={'step':step,'val':score,'train':train,'batch_loss':float(loss.detach().cpu()),'parts':{k:float(v.cpu()) for k,v in parts.items()}}
                history.append(row); print(json.dumps(row),flush=True)
                if key>best:
                    best=key; best_step=step; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if score['balanced_accuracy'] > anchor+cfg['neural']['significant_ba_improvement']:
                    anchor=score['balanced_accuracy']; last_significant=step
                if step>=cfg['neural']['min_steps'] and step-last_significant>=cfg['neural']['patience_steps']:
                    reason='validation_patience'; break
        if best_state is None or not np.all(class_counts>0): raise RuntimeError('missing checkpoint/class coverage')
        torch.save({'format':'volatility_classification_research_v1','task':task,'model':best_state,'scales':fold['scales'],
            'best_step':best_step,'preregistration_sha256':reg},directory/'research_model.pt')
        result={**task,'steps':step,'best_step':best_step,'best_val_balanced_accuracy':best[0],'best_val_log_loss':-best[1],
            'right_censored':reason=='max_steps','stop_reason':reason,'parameters':sum(p.numel() for p in model.parameters()),
            'class_counts':class_counts.tolist(),'last_significant_improvement_step':last_significant,'history':history,
            'checkpoint_sha256':_sha(directory/'research_model.pt'),'preregistration_sha256':reg,'elapsed_seconds':time.monotonic()-started}
        _json(directory/'fit.json',result); torch.mps.empty_cache()
        return {k:result[k] for k in ('job','best_step','steps','elapsed_seconds')}


def run_pool(cfg,kind):
    _,frozen=verify(cfg); out=Path(cfg['output_dir'])
    if kind=='neural': worker=train_task; tasks=frozen['tasks']; workers=cfg['parallel_workers']
    else:
        from .wm_classification_classical import fit_classical_task
        worker=fit_classical_task; tasks=frozen['classical_tasks']; workers=2
    pending=[t for t in tasks if not (out/kind/t['job']/'fit.json').exists()]; errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(worker,cfg['config_path'],t):t for t in pending}
        for future in as_completed(futures):
            try: print(json.dumps(future.result()),flush=True)
            except Exception as exc: errors.append({'task':futures[future],'error':repr(exc)})
            complete=sum((out/kind/t['job']/'fit.json').exists() for t in tasks)
            _json(out/(kind+'_status.json'),{'status':'failed' if errors else 'running','completed':complete,'total':len(tasks),'errors':errors})
    _json(out/(kind+'_status.json'),{'status':'failed' if errors else 'fits_completed','completed':sum((out/kind/t['job']/'fit.json').exists() for t in tasks),'total':len(tasks),'errors':errors})
    if errors: raise RuntimeError(errors)


def prediction_frame(arrays,ids,data,fold,condition,model,seed,split,probs,predictions=None):
    priors=np.array(fold['priors'][condition]); actual=data['label_'+condition][ids]
    pred=balanced_prior_predictions(probs,priors) if predictions is None else predictions
    raw=probs.argmax(1) if predictions is None else predictions
    return pd.DataFrame({'condition':condition,'model':model,'fold':fold['fold'],'seed':seed,'split':split,'decision_id':ids,
        'timestamp':pd.to_datetime(arrays['index_ns'][ids],utc=True),'actual':actual,'theta':data['theta_'+condition][ids],
        'return':arrays['returns'][ids,1],'p0':probs[:,0],'p1':probs[:,1],'p2':probs[:,2],'pred':pred,'pred_raw':raw})


def evaluate_fold(config_path, fold_id):
    import pickle
    from lightgbm import Booster
    cfg=load_config(config_path); parent,_=parents(cfg); out=Path(cfg['output_dir'])
    dest=out/'predictions'/f'fold_{fold_id:02d}.parquet'; dest.parent.mkdir(exist_ok=True)
    if dest.exists(): return {'fold':fold_id,'cached':True}
    torch.set_num_threads(cfg['cpu_threads']); torch.use_deterministic_algorithms(True)
    arrays=_arrays(parent['output_dir']); data=fold_data(cfg,fold_id)
    fold=json.loads((out/'prepared/folds.json').read_text())[fold_id]; pieces=[]
    reg=_sha(out/'preregistration.json')
    for condition in cfg['conditions']:
        job=f'fold_{fold_id:02d}_{condition}'; directory=out/'classical'/job
        fit=json.loads((directory/'fit.json').read_text())
        if fit['preregistration_sha256']!=reg: raise ValueError('classical identity mismatch')
        for name,sha in fit['artifact_sha256'].items():
            if _sha(directory/name)!=sha: raise ValueError('classical checkpoint changed')
        logistic=pickle.loads((directory/'logistic.pkl').read_bytes()); lgbm=Booster(model_file=str(directory/'lightgbm.txt'))
        for split in ('val','test'):
            ids=data[split]; theta=data['theta_'+condition][ids]
            x=np.column_stack((get_design(cfg,ids,data),np.log(theta/fold['return_rms']).astype(np.float32)))
            for name,prob in (('logistic',logistic.predict_proba(x)),('lightgbm',lgbm.predict(x,num_threads=cfg['cpu_threads']))):
                pieces.append(prediction_frame(arrays,ids,data,fold,condition,name,0,split,prob))
            priors=np.array(fold['priors'][condition]); prob=np.tile(priors,(len(ids),1))
            pieces.append(prediction_frame(arrays,ids,data,fold,condition,'train_majority',0,split,prob,np.full(len(ids),priors.argmax())))
            # Zero-return conversion is exactly the always-flat decision.
            flat=np.tile(np.array([0.,1.,0.]),(len(ids),1))
            pieces.append(prediction_frame(arrays,ids,data,fold,condition,'always_flat',0,split,flat,np.ones(len(ids),dtype=int)))
        for model_name in cfg['neural']['models']:
            for seed in cfg['seeds']:
                task={'condition':condition,'fold':fold_id,'model':model_name,'seed':seed,'job':f'fold_{fold_id:02d}_{condition}_{model_name}_s{seed}'}
                directory=out/'neural'/task['job']; fit=json.loads((directory/'fit.json').read_text())
                if fit['preregistration_sha256']!=reg or fit['checkpoint_sha256']!=_sha(directory/'research_model.pt'): raise ValueError('neural checkpoint identity')
                cp=torch.load(directory/'research_model.pt',map_location='cpu',weights_only=False)
                if cp['task']!=task or cp['preregistration_sha256']!=reg: raise ValueError('checkpoint task')
                model=ForecastClassifier(model_name,63,parent['model'],fold['scales']).to(cfg['device']); model.load_state_dict(cp['model'])
                for split in ('val','test'):
                    ids=data[split]; prob=prediction_proba(cfg,model,arrays,ids,data,fold,condition)
                    pieces.append(prediction_frame(arrays,ids,data,fold,condition,model_name,seed,split,prob))
                del model; torch.mps.empty_cache()
    pd.concat(pieces,ignore_index=True).to_parquet(dest,index=False)
    return {'fold':fold_id,'rows':sum(len(p) for p in pieces)}


def evaluate(cfg):
    _,frozen=verify(cfg); out=Path(cfg['output_dir'])
    for kind,tasks in (('neural',frozen['tasks']),('classical',frozen['classical_tasks'])):
        if any(not(out/kind/t['job']/'fit.json').exists() for t in tasks): raise ValueError('global selection barrier not satisfied')
    # Explicit marker prevents a report from being confused with a training run.
    _json(out/'test_access_barrier.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
        'neural_fits':len(frozen['tasks']),'classical_fits':2*len(frozen['classical_tasks']), 'preregistration_sha256':_sha(out/'preregistration.json')})
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate_fold,cfg['config_path'],fold) for fold in range(15)]
        for future in as_completed(futures): print(json.dumps(future.result()),flush=True)


def summarize(cfg):
    _,frozen=verify(cfg); out=Path(cfg['output_dir'])
    if not (out/'test_access_barrier.json').exists(): raise ValueError('missing Test barrier')
    paths=sorted((out/'predictions').glob('fold_*.parquet'))
    if len(paths)!=15: raise ValueError('incomplete predictions')
    predictions=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    metrics=[]; fold_metrics=[]
    for keys,frame in predictions.groupby(['condition','model','split']):
        score=score_classification(frame.actual.to_numpy(),frame[['p0','p1','p2']].to_numpy(),None,
            predictions=frame.pred.to_numpy(),raw_predictions=frame.pred_raw.to_numpy())
        metrics.append(dict(zip(('condition','model','split'),keys))|score)
    for keys,frame in predictions.groupby(['condition','model','split','fold','seed']):
        score=score_classification(frame.actual.to_numpy(),frame[['p0','p1','p2']].to_numpy(),None,
            predictions=frame.pred.to_numpy(),raw_predictions=frame.pred_raw.to_numpy())
        fold_metrics.append(dict(zip(('condition','model','split','fold','seed'),keys))|score)
    pd.DataFrame(metrics).to_csv(out/'aggregate_metrics.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv',index=False)
    learners=['logistic','lightgbm','direct','wm']; pairs=list(itertools.combinations(learners,2))+[('train_majority',x) for x in learners]
    tests=predictions.loc[predictions.split=='test']; inference=[]; corrections={}
    for condition in cfg['conditions']:
        family={}
        for a,b in pairs:
            selected=[]
            for name in (a,b):
                frame=tests.loc[(tests.condition==condition)&(tests.model==name)]
                if (frame.seed==0).all(): frame=pd.concat([frame.assign(seed=seed) for seed in cfg['seeds']],ignore_index=True)
                selected.append(frame)
            rows=paired_classification_block_bootstrap(pd.concat(selected,ignore_index=True),a,b,condition=condition,
                block_days=cfg['statistics']['block_days'],replicates=cfg['statistics']['bootstrap_replicates'],random_seed=cfg['statistics']['bootstrap_seed'])
            inference.extend(rows)
            # Evaluation module owns the exact inference schema; see its API.
            primary=next(row for row in rows if row['block_days']==cfg['statistics']['primary_block_days'])
            family[b+'-'+a]=primary['metrics']['balanced_accuracy_difference']['p_value']
        corrections[condition]=holm_adjust(family)
    histories=[json.loads((out/'neural'/task['job']/'fit.json').read_text()) for task in frozen['tasks']]
    summary={'metrics':metrics,'inference':inference,'holm_adjusted_p':corrections,
        'neural_runs':len(histories),'classical_fits':2*len(frozen['classical_tasks']),
        'convergence':{condition:{model:{'runs':len(rows),'max_steps_count':sum(r['right_censored'] for r in rows),
            'median_best_step':float(np.median([r['best_step'] for r in rows]))} for model in learners[2:]
            for rows in [[r for r in histories if r['condition']==condition and r['model']==model]]} for condition in cfg['conditions']},
        'preregistration_sha256':_sha(out/'preregistration.json'),'test_selection_allowed':False}
    _json(out/'summary.json',summary)
    print(pd.DataFrame(metrics).query("split=='test'")[["condition","model","accuracy","balanced_accuracy","macro_f1","log_loss"]].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=('prepare','run-neural','run-classical','evaluate','summarize'))
    parser.add_argument('--config',default=str(DEFAULT_CONFIG)); args=parser.parse_args(); cfg=load_config(args.config)
    if args.command=='run-neural': run_pool(cfg,'neural')
    elif args.command=='run-classical': run_pool(cfg,'classical')
    else: {'prepare':prepare,'evaluate':evaluate,'summarize':summarize}[args.command](cfg)


if __name__=='__main__': main()
