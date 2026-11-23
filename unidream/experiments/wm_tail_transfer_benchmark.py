"""Registered, isolated tail-information transfer research coordinator.

This entry point never changes the Plan011 v31 pipeline or previous studies.
Every forecast and allocation selection is completed before Test inference.
Sparse free liquidation samples are a separately named evaluation cohort.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/wm_tail_transfer_20261112.yaml'
PREFIX = 'wm_tail_transfer_'
_VERIFIED_HASHES = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def file_identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _verification_sha(path, full=False):
    """Reuse a content check only while the filesystem identity is unchanged."""
    key = str(Path(path).resolve())
    identity = file_identity(path)
    prior = _VERIFIED_HASHES.get(key)
    if full or prior is None or prior[0] != identity:
        prior = (identity, sha(path))
        _VERIFIED_HASHES[key] = prior
    return prior[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg['config_path'] = str(path)
    cfg['output_dir'] = str((ROOT / cfg['output_dir']).resolve())
    for key in ('raw_market_path', 'legacy_data_root'):
        cfg['data'][key] = str((ROOT / cfg['data'][key]).resolve())
    if (cfg['test_selection_allowed'] is not False or cfg['historical_seen_test'] is not True
            or cfg['independently_unused_holdout'] is not False):
        raise ValueError('known historical Test must remain report-only')
    if (cfg['targets']['horizons'] != [1, 4, 16, 48, 96]
            or cfg['targets']['primary_horizon'] != 16
            or cfg['splits']['producer_train_months'] + cfg['splits']['producer_validation_months']
            + cfg['splits']['policy_train_months'] != cfg['splits']['train_months']):
        raise ValueError('fixed horizons and non-overlapping producer/policy chronology required')
    if cfg['accounting']['reset_positions_at_segment_boundary'] is not False:
        raise ValueError('a continuous account is required')
    return cfg


def fold_definitions(cfg):
    settings = cfg['splits']
    result = []
    origin = pd.Timestamp(settings['first_train_start'])
    last = pd.Timestamp(settings['last_test_end_exclusive'])
    for fold in settings['folds']:
        train = origin + pd.DateOffset(months=int(fold) * settings['step_months'])
        val = train + pd.DateOffset(months=settings['train_months'])
        test = val + pd.DateOffset(months=settings['validation_months'])
        end = min(test + pd.DateOffset(months=settings['test_months']), last)
        prod_val = train + pd.DateOffset(months=settings['producer_train_months'])
        policy = prod_val + pd.DateOffset(months=settings['producer_validation_months'])
        if not train < prod_val < policy < val < test < end:
            raise ValueError('empty or nonchronological fold')
        result.append(dict(fold=int(fold), train_start=train.isoformat(), train_end=val.isoformat(),
            val_start=val.isoformat(), val_end=test.isoformat(), test_start=test.isoformat(), test_end=end.isoformat(),
            earlystop_start=val.isoformat(), earlystop_end=(val+pd.DateOffset(months=1)).isoformat(),
            calibration_start=(val+pd.DateOffset(months=1)).isoformat(), calibration_end=(val+pd.DateOffset(months=2)).isoformat(),
            selection_start=(val+pd.DateOffset(months=2)).isoformat(), selection_end=test.isoformat(),
            producer_train_start=train.isoformat(), producer_train_end=prod_val.isoformat(),
            producer_val_start=prod_val.isoformat(), producer_val_end=policy.isoformat(),
            producer_earlystop_start=prod_val.isoformat(), producer_earlystop_end=(prod_val+pd.DateOffset(months=1)).isoformat(),
            producer_calibration_start=(prod_val+pd.DateOffset(months=1)).isoformat(), producer_calibration_end=policy.isoformat(),
            policy_train_start=policy.isoformat(), policy_train_end=val.isoformat(),
            known_historical_test=True))
    return result


def split_ids(data, group, start, end, purge=True, *, require_labels=True, cohort=None):
    index = np.asarray(data['index_ns'], np.int64)
    valid = np.asarray(data['valid_'+group], bool)
    lower, upper = pd.Timestamp(start).value, pd.Timestamp(end).value
    eligible = valid & (index >= lower) & (index < upper)
    if cohort == 'dense_common':
        eligible &= np.asarray(data['valid_D07_dense'],bool)
    elif cohort == 'liquidation_common':
        eligible &= np.asarray(data['valid_D07'],bool)
    elif cohort == 'liquidation_only_common':
        eligible &= np.asarray(data['valid_D06'],bool)
    elif cohort not in (None,'native'):
        raise ValueError('unknown explicitly registered input cohort')
    if purge and require_labels:
        predictive = np.asarray(data.get('predictive_label_end_ns', data['label_end_ns']))
        passive = np.asarray(data.get('passive_label_end_ns', predictive))
        eligible &= (np.maximum(predictive.max(axis=1), passive.max(axis=1)) < upper)
    return np.flatnonzero(eligible)


def make_job(role, group, model, fold, seed, *, objectives='joint', **extra):
    value = dict(role=role, group=group, model=model, fold=int(fold), seed=int(seed), objectives=objectives, **extra)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:10]
    value['job'] = f'{role}_{group}_{model}_f{fold:02d}_s{seed}_{digest}'
    return value


def execution_device(cfg, task):
    """Fix device placement from identity, never validation/Test outcomes.

    Matched input and objective ablations keep the same family/fold/seed
    backend. CPU and GPU share the forecast queue; policy training stays CPU.
    """
    model=task.get('model')
    if model in ('linear','lightgbm','lgbm'):
        return 'cpu'
    fallback=cfg.get('device','mps')
    fraction=float(cfg.get('runtime',{}).get('forecast_cpu_fractions',{}).get(model,0.))
    if not 0.<=fraction<=1.:
        raise ValueError('forecast CPU assignment fraction must lie in [0,1]')
    identity=f"{int(task['fold'])}:{int(task['seed'])}".encode()
    coordinate=int.from_bytes(hashlib.sha256(identity).digest()[:8],'big')/2**64
    return 'cpu' if coordinate<fraction else fallback


def screen_jobs(cfg):
    s = cfg['screen']
    pairs=[(g,('liquidation_only_common' if g=='D06' else 'liquidation_common')
            if g in ('D06','D07') else 'dense_common')
           for g in s['input_groups']+s.get('supplementary_input_groups',[])]
    pairs += [(g,'liquidation_common')for g in s.get('sparse_control_groups',[])]
    pairs += [(g,'liquidation_only_common')for g in s.get('liquidation_only_control_groups',[])]
    return [make_job('benchmark', group, model, fold, seed,cohort=cohort)
            for group,cohort in pairs for model in s['models'] for fold in s['folds']
            for seed in (s['seeds'] if model in ('wm', 'lstm', 'transformer') else [s['seeds'][0]])]


def confirmation_jobs(cfg, groups):
    return [make_job('benchmark', group, model, fold, seed,cohort='dense_common')
            for group in groups for model in cfg['confirmation']['models'] for fold in cfg['splits']['folds']
            for seed in (cfg['seeds'] if model in ('wm', 'lstm', 'transformer') else [cfg['seeds'][0]])]


def producer_jobs(cfg):
    old = ['return', 'rv', 'legacy_downside', 'crash', 'dd_excess', 'advantage', 'recovery']
    result = []
    for fold in cfg['splits']['folds']:
        for model in ('linear', 'lightgbm'):
            result.append(make_job('producer', 'D03', model, fold, cfg['seeds'][0]))
        for seed in cfg['seeds']:
            result += [make_job('producer', 'D03', 'wm', fold, seed, objectives=old, representation='old'),
                       make_job('producer', 'D03', 'wm', fold, seed, representation='tail'),
                       make_job('producer', 'D03', 'wm', fold, seed, objectives=['T02','T03','T04','T05','T06','T08','T09','T10','T11','T12'], representation='tail_only')]
    return result


def ablation_jobs(cfg):
    s = cfg['ablation']; result = []
    for fold in s['folds']:
        for seed in s['seeds']:
            for model in ('lstm', 'transformer', 'wm'):
                for objective in ('regression', 'classification'):
                    result.append(make_job('ablation','D03',model,fold,seed,objectives=objective,item='L01'))
            for task in [f'T{i:02d}' for i in range(1,13)]:
                result.append(make_job('ablation','D03','wm',fold,seed,objectives=[task],item='L01_single_task'))
            for direction in ('up','down'):
                result.append(make_job('ablation','D03','wm',fold,seed,objectives=['T02'],direction=direction,item='L02'))
            result.append(make_job('ablation','D03','wm',fold,seed,architecture='hurdle',item='L03'))
            for loss in ('weighted_bce', 'focal'):
                result.append(make_job('ablation','D03','wm',fold,seed,loss={'classification_loss':loss},item='L04'))
            result.append(make_job('ablation','D03','wm',fold,seed,loss={'variance_loss':'log_mse'},item='L05'))
            result.append(make_job('ablation','D03','wm',fold,seed,loss={'student_tail_weight':2.},item='L08'))
        for model in ('linear','lightgbm'):
            result.append(make_job('ablation','D03',model,fold,s['seeds'][0],feature_recipe='full_flatten',item='classical_history_compression'))
    return [make_job(**{**{k:v for k,v in task.items()if k!='job'},'cohort':s['cohort']})for task in result]


def policy_tasks(cfg):
    controls = []
    for variant in cfg['policy']['variants']:
        for cap in cfg['policy_comparisons']['main_caps']:
            controls.append((variant,dict(cap=cap, speed=.08, bc_prior=.01, controller='direct', objective='net_log_dd', risk_gate=False, account_mode='cash_borrow')))
    base = dict(cap=2., speed=.08, bc_prior=.01, controller='direct', objective='net_log_dd', risk_gate=False, account_mode='cash_borrow')
    for change in [dict(speed=.24),dict(speed=.24,bc_prior=0.),dict(bc_prior=0.),dict(controller='hierarchical'),dict(controller='ml_rule'),dict(risk_gate=True),dict(objective='net_log'),dict(objective='episodic_cvar_5pct')]:
        controls.append(('R04',{**base,**change}))
    for variant in ('R00','R04'):
        for cap in cfg['perpetual']['caps']:
            controls.append((variant,{**base,'cap':cap,'account_mode':'linear_perp',
                'mmr':cfg['perpetual']['primary_maintenance_margin_rate'],'slippage':cfg['perpetual']['primary_slippage_rate']}))
    tasks=[]
    for variant, control in controls:
        for fold in cfg['splits']['folds']:
            for seed in cfg['seeds']:
                value=dict(fold=fold,seed=seed,input_variant='D03',policy_variant=variant,control=control)
                digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:12]
                value['job']=f'policy_{variant}_f{fold:02d}_s{seed}_{digest}'
                tasks.append(value)
    return tasks


class WindowCache:
    """Exact window-local base/TA cache plus causal supplementary columns."""
    def __init__(self, cfg, data, group):
        self.base = np.load(Path(cfg['output_dir'])/'data/window_cache/base_D03.npy',mmap_mode='r')
        self.data,self.group,self.context=data,group,int(cfg['model']['context'])
        self.shape=(len(data['index_ns']),self.context,len(data['group_columns'][group]))

    def __getitem__(self, ids):
        ids=np.asarray(ids,np.int64)
        scalar=ids.ndim==0
        ids=np.atleast_1d(ids)
        width={'D00':5,'D01':19,'D02':63}.get(self.group,84)
        parts=[np.asarray(self.base[ids,:,:width])]
        positions=ids[:,None]-self.context+np.arange(self.context)[None,:]
        if self.group in ('D04','D07','D07_dense'):
            parts.append(np.asarray(self.data['minute_features'][positions],np.float32))
        if self.group in ('D05','D07','D07_dense'):
            parts.append(np.asarray(self.data['depth_features'][positions],np.float32))
        if self.group in ('D06','D07'):
            current=np.asarray(self.data['liquidation_features'][ids-1],np.float32)
            parts.append(np.broadcast_to(current[:,None],(len(ids),self.context,current.shape[-1])))
        answer=np.concatenate(parts,-1)
        if answer.shape[-1]!=self.shape[-1] or not np.isfinite(answer).all():
            raise ValueError('unavailable causal window requested from cache')
        return answer[0] if scalar else answer


def _cache_worker(config, ids, group):
    from . import wm_tail_transfer_data as data_module
    data=data_module.load_data(config)
    target=Path(config['output_dir'])/'data/window_cache/base_D03.npy'
    output=np.load(target,mmap_mode='r+')
    width=84 if group=='D03' else 5
    for start in range(0,len(ids),128):
        selected=np.asarray(ids[start:start+128],np.int64)
        output[selected,:,:width]=data_module.window_inputs(data,selected,group,config['model']['context'])
    output.flush()
    return len(ids)


def prepare_window_cache(cfg):
    from . import wm_tail_transfer_data as data_module
    data=data_module.load_data(cfg)
    folder=Path(cfg['output_dir'])/'data/window_cache';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'completion.json'
    if marker.exists():
        prior=json.loads(marker.read_text())
        if prior['data_manifest_sha256']!=sha(Path(cfg['output_dir'])/'data/prepared/manifest.json') or sha(folder/'base_D03.npy')!=prior['cache_sha256']:
            raise ValueError('window cache or data manifest changed')
        return prior
    n,context=len(data['index_ns']),cfg['model']['context']
    out=np.lib.format.open_memmap(folder/'base_D03.npy',mode='w+',dtype=np.float32,shape=(n,context,84))
    out[:]=np.nan;out.flush();del out
    ids3=split_ids(data,'D03',cfg['data']['start'],cfg['data']['end_exclusive'],purge=False)
    ids0=split_ids(data,'D00',cfg['data']['start'],cfg['data']['end_exclusive'],purge=False)
    ids0=np.setdiff1d(ids0,ids3,assume_unique=True)
    jobs=[(group,ids[start:start+1024])for group,ids in [('D03',ids3),('D00',ids0)]for start in range(0,len(ids),1024)]
    consumed=0
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(_cache_worker,cfg,ids,group)for group,ids in jobs}
        while futures:
            done,futures=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
            consumed+=sum(f.result()for f in done)
            write_json(folder/'progress.json',dict(rows=consumed,total=len(ids3)+len(ids0),at=now()))
    # Compare real cache slices against independently recomputed source windows.
    for group in data['group_columns']:
        ids=split_ids(data,group,cfg['data']['start'],cfg['data']['end_exclusive'],purge=False)
        if len(ids):
            selected=ids[np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=int)]
            np.testing.assert_array_equal(WindowCache(cfg,data,group)[selected],data_module.window_inputs(data,selected,group,context))
    record=dict(complete=True,rows=consumed,context=context,window_local_technical_indicators=True,
        data_manifest_sha256=sha(Path(cfg['output_dir'])/'data/prepared/manifest.json'),cache_sha256=sha(folder/'base_D03.npy'))
    write_json(marker,record)
    return record


def source_snapshot():
    paths=[p for base in ('unidream','tests','configs')for p in (ROOT/base).rglob('*')
           if p.is_file()and p.suffix in ('.py','.yaml')and not p.name.startswith(('wm_tail_transfer_','test_wm_tail_transfer_'))]
    paths += [ROOT/name for name in ('AGENTS.md','README.md','SPEC.md','docs/README.md')]
    return {str(p.relative_to(ROOT)):sha(p)for p in sorted(set(paths))}


def new_sources():
    paths=list((ROOT/'unidream/experiments').glob('wm_tail_transfer_*.py'))+list((ROOT/'tests').glob('test_wm_tail_transfer_*.py'))+[DEFAULT_CONFIG]
    return {str(p.relative_to(ROOT)):sha(p)for p in sorted(paths)}


def preregister(cfg):
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    marker=output/'preregistration.json'
    if marker.exists():
        verify_registration(cfg)
        return sha(marker)
    if list((output/'fits').glob('*/fit.json'))or list((output/'policies').glob('*/fit.json')):
        raise ValueError('new registration cannot follow fitted result access')
    prepared=output/'data/prepared/manifest.json'
    if not json.loads(prepared.read_text()).get('complete'):
        raise ValueError('complete provenance-bound data required')
    record=dict(at=now(),config=cfg,folds=fold_definitions(cfg),source_sha256=new_sources(),
        previous_source_sha256=source_snapshot(),data_manifest_sha256=sha(prepared),
        window_cache_completion_sha256=sha(output/'data/window_cache/completion.json'),
        screen_jobs=screen_jobs(cfg),producer_jobs=producer_jobs(cfg),ablation_jobs=ablation_jobs(cfg),
        policy_tasks=policy_tasks(cfg),selection_rule=cfg['confirmation']['selection'],
        packages={name:importlib.metadata.version(name)for name in ('torch','numpy','pandas','scipy','lightgbm','scikit-learn')},
        test_report_only=True,all_Test_periods_previously_inspected=True,
        liquidation_sample_scope='free calendar-first-day observed snapshots only; no full-period claim',
        linear_perp_scope=cfg['perpetual']['scope'])
    history=output/'prior_attempts.json'
    if history.exists():
        prior=json.loads(history.read_text())
        if any(item.get('Test_read')or not item.get('all_fits_discarded')for item in prior['attempts']):
            raise ValueError('only fully discarded pre-Test attempts may precede a fresh registration')
        record['prior_attempts']=prior
        record['prior_attempts_sha256']=sha(history)
    write_json(marker,record)
    return sha(marker)


def verify_registration(cfg, full=False):
    output=Path(cfg['output_dir'])
    registration=json.loads((output/'preregistration.json').read_text())
    if registration['config']!=cfg or registration['source_sha256']!=new_sources():
        raise ValueError('registered configuration or new implementation changed')
    for relative,expected in registration['previous_source_sha256'].items():
        if _verification_sha(ROOT/relative,full=full)!=expected:
            raise ValueError('pre-existing file changed: '+relative)
    if sha(output/'data/prepared/manifest.json')!=registration['data_manifest_sha256']:
        raise ValueError('registered data manifest changed')
    if sha(output/'data/window_cache/completion.json')!=registration['window_cache_completion_sha256']:
        raise ValueError('registered causal cache manifest changed')
    if 'prior_attempts_sha256'in registration and sha(output/'prior_attempts.json')!=registration['prior_attempts_sha256']:
        raise ValueError('registered pre-Test invalidation history changed')
    if full:
        manifest=json.loads((output/'data/prepared/manifest.json').read_text())
        for name,expected in manifest['artifacts'].items():
            if sha(output/'data/prepared'/name)!=expected:
                raise ValueError('prepared data artifact changed: '+name)
        cache=json.loads((output/'data/window_cache/completion.json').read_text())
        if sha(output/'data/window_cache/base_D03.npy')!=cache['cache_sha256']:
            raise ValueError('causal window cache changed')
    return registration


def verify_test_barrier(cfg, full=False):
    verify_registration(cfg, full=full)
    out=Path(cfg['output_dir']);path=out/'test_barrier.json'
    barrier=json.loads(path.read_text())
    if barrier['registration_sha256']!=sha(out/'preregistration.json'):
        raise ValueError('wrong Test barrier registration')
    for relative,expected in barrier['fit_sha256'].items():
        identity = barrier.get('fit_file_identity', {}).get(relative)
        if identity is not None and file_identity(out/relative)!=identity:
            raise ValueError('Train/Val artifact metadata changed after global Test barrier')
        if (full or identity is None) and sha(out/relative)!=expected:
            raise ValueError('Train/Val selection changed after global Test barrier')
    for name,field in [('comparison_plan.json','comparison_plan_sha256'),('execution_manifest.json','execution_manifest_sha256'),
                       ('input_selection.json','input_selection_sha256'),('producer_selection.json','producer_selection_sha256')]:
        if sha(out/name)!=barrier[field]:
            raise ValueError('frozen comparison or selection manifest changed: '+name)
    return barrier


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--action',choices=['prepare','register','run'],default='run')
    args=parser.parse_args();config=load_config(args.config)
    if args.action=='prepare':
        print(json.dumps(prepare_window_cache(config)))
    elif args.action=='register':
        print(preregister(config))
    else:
        from .wm_tail_transfer_runner import run
        run(config)
