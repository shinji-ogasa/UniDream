"""Report-only Test replay and registered comparisons for tail transfer.

No estimator, threshold, calibrator, checkpoint or configuration is selected
here. Forecasts preserve timestamp IDs and native coordinates. Policy replays
carry the entire account across folds and score every raw calendar bar.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from . import wm_tail_transfer_benchmark as benchmark
from . import wm_tail_transfer_data as data_module
from . import wm_tail_transfer_evaluation as evaluation


MAIN_TASKS = tuple(f'T{i:02d}' for i in range(1, 13))
CALIBRATIONS = ('none', 'parametric', 'isotonic')
STATISTICAL_MODELS = ('zero_return', 'train_climatology', 'past_variance', 'ewma',
                      'garch_gaussian', 'gjr_student_t', 'historical_residual', 'garch_evt')
DAY_NS = 86_400_000_000_000


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path, value):
    benchmark.write_json(path, _safe(value))


def _identity(task):
    return {k: v for k, v in task.items() if k not in ('source_task', 'source_tasks')}


def _gate(cfg, registration_sha):
    barrier = benchmark.verify_test_barrier(cfg)
    if barrier['registration_sha256'] != registration_sha:
        raise ValueError('report-only Test barrier registration differs')
    return barrier


def _completed(path, registration_sha):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if not result.get('complete') or result['registration_sha256'] != registration_sha:
        raise ValueError('evaluation completion belongs to a different study')
    for name, expected in result['artifact_sha256'].items():
        if benchmark.sha(path.parent/name) != expected:
            raise ValueError('completed report artifact changed: '+name)
    return result


def _coordinate_filters(task, specs):
    if task.get('direction') not in ('up', 'down'):
        return None
    direction = int(task['direction'] == 'up')
    return {s['name']: [[h, direction, k] for h in range(s['shape'][0])
                        for k in range(s['shape'][2])]
            for s in specs if s['name'] in ('reach_fixed', 'reach_dynamic')}


def _unavailable_forecast(cfg, task, registration_sha, fit, destination, *, reason=None):
    """An unavailable Train/Val fit cannot be resurrected using Test outcomes."""
    from .wm_tail_transfer_models import build_specs,active_specs
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    specs=fit.get('head_specs') or active_specs(build_specs(cfg),task.get('objectives','joint'))
    pd.DataFrame(columns=['job','task','head','horizon','n','status']).to_parquet(destination/'test_metrics.parquet',index=False)
    pd.DataFrame(columns=['job','target','calibration','n','mean_loss','mean_baseline_loss']).to_csv(destination/'task_metrics.csv',index=False)
    record=dict(complete=True,available=False,status='unavailable_data',registration_sha256=registration_sha,
        task=_identity({k:v for k,v in task.items() if k!='members'}),native_head_specs=specs,
        reason=reason or fit.get('reason') or fit.get('unavailable_reason') or 'Train/Val data unavailable',
        n_prediction_rows=0,n_metric_coordinates=0,n_primary_task_outputs=0,calibrations=[],
        Test_read=False,Test_used_for_selection=False,n_new_training_fits=0,
        artifact_sha256={name:benchmark.sha(destination/name) for name in ('test_metrics.parquet','task_metrics.csv')})
    _json(destination/'evaluation.json',record)
    return record


def _write_forecast(cfg, task, registration_sha, data, ids, train_ids, specs,
                    predictions_by_calibration, destination, *, source_sha256):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    identity = _identity(task)
    metadata = {k: v if not isinstance(v, (dict, list)) else json.dumps(v, sort_keys=True)
                for k, v in identity.items()}
    filters = _coordinate_filters(task, specs)
    prediction_arrays = {'raw_ids': np.asarray(ids, np.int64),
                         'index_ns': np.asarray(data['index_ns'])[ids]}
    loss_arrays = dict(prediction_arrays)
    records, summaries = [], []
    for calibration, predictions in predictions_by_calibration.items():
        selected_specs = [s for s in specs if s['name'] in predictions]
        for name, value in predictions.items():
            prediction_arrays[f'{calibration}__{name}'] = np.asarray(value, np.float32)
        metrics = evaluation.evaluate_predictions(data, ids, predictions, selected_specs, cfg, train_ids,
                                                   identity={**metadata, 'calibration': calibration})
        if filters and 'direction' in metrics:
            direction_heads = metrics['head'].isin(filters)
            metrics = metrics.loc[~direction_heads | (metrics.direction == task['direction'])].copy()
        records.append(metrics)
        parts = evaluation.per_row_losses(data, ids, predictions, selected_specs, cfg, train_ids,
            horizon=int(cfg['targets']['primary_horizon']), coordinate_filters=filters)
        for spec in selected_specs:
            headpart = evaluation.per_row_losses(data, ids, predictions, [spec], cfg, train_ids,
                horizon=int(cfg['targets']['primary_horizon']), coordinate_filters=filters)[spec['task']]
            for key in ('model_loss', 'baseline_loss', 'valid'):
                loss_arrays[f'{calibration}__head_{spec["name"]}__{key}'] = headpart[key]
            if spec['name'] in ('reach_fixed','reach_dynamic'):
                for direction in ('down','up'):
                    direction_filters=_coordinate_filters({'direction':direction},[spec])
                    part=evaluation.per_row_losses(data,ids,predictions,[spec],cfg,train_ids,
                        horizon=int(cfg['targets']['primary_horizon']),coordinate_filters=direction_filters)[spec['task']]
                    for key in ('model_loss','baseline_loss','valid'):
                        loss_arrays[f'{calibration}__head_{spec["name"]}_{direction}__{key}']=part[key]
        for target, part in parts.items():
            for key in ('model_loss', 'baseline_loss', 'valid'):
                loss_arrays[f'{calibration}__{target}__{key}'] = part[key]
            valid = part['valid'] & np.isfinite(part['model_loss']) & np.isfinite(part['baseline_loss'])
            summaries.append({**metadata, 'calibration': calibration, 'target': target,
                              'primary_horizon': cfg['targets']['primary_horizon'],
                              'n': int(valid.sum()),
                              'mean_loss': float(np.mean(part['model_loss'][valid])) if valid.any() else np.nan,
                              'mean_baseline_loss': float(np.mean(part['baseline_loss'][valid])) if valid.any() else np.nan})
    metrics = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    np.savez_compressed(destination/'test_predictions.npz', **prediction_arrays)
    np.savez_compressed(destination/'per_row_losses.npz', **loss_arrays)
    metrics.to_parquet(destination/'test_metrics.parquet', index=False)
    pd.DataFrame(summaries).to_csv(destination/'task_metrics.csv', index=False)
    artifacts = ['test_predictions.npz', 'per_row_losses.npz', 'test_metrics.parquet', 'task_metrics.csv']
    result = dict(complete=True, registration_sha256=registration_sha, task=identity,
        n_prediction_rows=len(ids), n_metric_coordinates=len(metrics), n_primary_task_outputs=len(summaries),
        calibrations=list(predictions_by_calibration), primary_horizon=cfg['targets']['primary_horizon'],
        native_head_specs=specs, train_reference_rows=len(train_ids),
        direction_scope=task.get('direction', 'both'), source_sha256=source_sha256,
        Test_used_for_selection=False, first_timestamp_ns=int(data['index_ns'][ids[0]]) if len(ids) else None,
        last_timestamp_ns=int(data['index_ns'][ids[-1]]) if len(ids) else None,
        artifact_sha256={name: benchmark.sha(destination/name) for name in artifacts})
    _json(destination/'evaluation.json', result)
    return result


def evaluate_forecast_job(cfg, task, registration_sha, *, arrays=None):
    """Report all native coordinates and all Train/Val-fitted calibrations."""
    _gate(cfg, registration_sha)  # Must precede model, producer or Test data access.
    destination = Path(cfg['output_dir'])/'evaluations/forecast'/task['job']
    existing = _completed(destination/'evaluation.json', registration_sha)
    if existing is not None:
        return existing
    from . import wm_tail_transfer_training as training
    directory = training.fit_directory(cfg, task)
    fit_record=json.loads((directory/'fit.json').read_text())
    if not fit_record.get('complete') or fit_record['registration_sha256']!=registration_sha:
        raise ValueError('a matching frozen forecast fit is required')
    if fit_record.get('available') is False:
        return _unavailable_forecast(cfg,task,registration_sha,fit_record,destination)
    data = data_module.load_data(cfg) if arrays is None else arrays
    meta, _, splits = training.split_arrays(cfg, task, data)
    ids = benchmark.split_ids(data, task['group'], meta['test_start'], meta['test_end'],
                              cohort=task.get('cohort'))
    if not len(ids):
        return _unavailable_forecast(cfg,task,registration_sha,fit_record,destination,
                                     reason='no observed Test timestamps in the fixed registered cohort')
    _, _, fit, selected_calibrator = training.load_fit(cfg, task, registration_sha)
    prediction = training.predict(cfg, task, registration_sha, ids, split='test', arrays=data, calibration=False)
    all_path = directory/'calibrators_all.pkl'
    if all_path.exists():
        if fit['artifact_sha256'].get('calibrators_all.pkl') != benchmark.sha(all_path):
            raise ValueError('Train/Val calibrator family is not bound to the selected fit')
        with all_path.open('rb') as stream:
            calibrators = pickle.load(stream)
    else:
        calibrators = {'none': {'method': 'none', 'heads': {}},
                       selected_calibrator.get('method', 'none'): selected_calibrator}
    by_calibration = {name: training.apply_calibrators(prediction, calibrators[name])
                      for name in CALIBRATIONS if name in calibrators}
    return _write_forecast(cfg, task, registration_sha, data, ids, splits['train'], fit['head_specs'],
                           by_calibration, destination,
                           source_sha256={'fit.json': benchmark.sha(directory/'fit.json')})


def evaluate_statistical_job(cfg, task, registration_sha, *, arrays=None):
    """Evaluate a previously selected statistical bundle without any fitting."""
    _gate(cfg, registration_sha)
    from . import wm_tail_transfer_baselines as baselines
    from .wm_tail_transfer_models import build_specs
    data = data_module.load_data(cfg) if arrays is None else arrays
    directory = Path(cfg['output_dir'])/'statistical'/task['job']
    fit = json.loads((directory/'fit.json').read_text())
    if fit['registration_sha256'] != registration_sha or not fit.get('complete'):
        raise ValueError('statistical bundle must complete before Test')
    bundle_path = directory/'bundle.pkl'
    if fit['artifact_sha256']['bundle.pkl'] != benchmark.sha(bundle_path):
        raise ValueError('selected statistical bundle changed')
    with bundle_path.open('rb') as stream:
        bundle = pickle.load(stream)
    meta = next(m for m in benchmark.fold_definitions(cfg) if m['fold'] == task['fold'])
    ids = benchmark.split_ids(data, task.get('group', 'D03'), meta['test_start'], meta['test_end'],
                              cohort=task.get('cohort', 'dense_common'))
    stride=int(cfg.get('statistical_baselines', {}).get('evaluation_stride_bars', 16))
    ids=ids[ids % stride == 0]
    train_ids = benchmark.split_ids(data, task.get('group', 'D03'), meta['train_start'], meta['train_end'],
                                    cohort=task.get('cohort', 'dense_common'))
    all_predictions = baselines.predict_statistical_baselines(data, ids, bundle, cfg)
    output = []
    for name, predictions in all_predictions.items():
        report_task = {**task, 'model': name, 'seed':cfg['seeds'][0], 'group':task.get('group','D03'),
                       'cohort':task.get('cohort','dense_common'), 'job': task['job']+'__'+name}
        destination = Path(cfg['output_dir'])/'evaluations/forecast'/report_task['job']
        existing = _completed(destination/'evaluation.json', registration_sha)
        output.append(existing if existing is not None else _write_forecast(cfg, report_task, registration_sha,
            data, ids, train_ids, build_specs(cfg), {'none': predictions}, destination,
            source_sha256={'fit.json': benchmark.sha(directory/'fit.json')}))
    return output


def policy_path_id(path_spec):
    identity = {k: v for k, v in path_spec.items() if k != 'sources'}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    task = path_spec['tasks'][0]
    return path_spec.get('path_id', path_spec.get('job', f"{path_spec['stage']}_{task['policy_variant']}_s{task['seed']}_{digest}"))


def policy_metrics(ledger, *, annual_bars=35040):
    """Physical path metrics, including seed insolvency rather than dropping it."""
    if ledger.empty or np.any(np.diff(ledger.raw_id.to_numpy(np.int64)) != 1):
        raise ValueError('metrics require the entire consecutive raw calendar')
    start = float(ledger.equity_previous.iloc[0]); final = float(ledger.equity.iloc[-1])
    if start <= 0 or not np.isfinite(start):
        raise ValueError('positive initial account value required')
    equity = ledger.equity.to_numpy(float)
    if np.any(~np.isfinite(equity)) or np.any(equity < 0):
        raise ValueError('physical equity must be finite and nonnegative')
    logs = ledger.net_log_return.to_numpy(float)
    peak = np.maximum.accumulate(np.r_[start, equity])
    dd = 1-equity/peak[1:]
    exposure = ledger.close_exposure.to_numpy(float)
    times = pd.DatetimeIndex(ledger.timestamp)
    result = dict(n_rows=len(ledger), n_calendar_days=int((times[-1].normalize()-times[0].normalize()).days+1),
        first_timestamp=times[0].isoformat(), last_timestamp=times[-1].isoformat(),
        initial_equity=start, final_equity=final, net_total_return=final/start-1,
        max_drawdown=float(dd.max()), annualized_net_log_growth=float(logs.mean()*annual_bars),
        annualized_volatility=float(np.std(logs, ddof=1)*np.sqrt(annual_bars)) if np.isfinite(logs).all() and len(logs)>1 else np.nan,
        mean_exposure=float(exposure.mean()), exposure_max=float(exposure.max()),
        above1_fraction=float((exposure>1).mean()), above112_fraction=float((exposure>1.12).mean()),
        above2_fraction=float((exposure>2).mean()), above3_fraction=float((exposure>3).mean()),
        bankrupt=bool((equity<=0).any()), liquidation=bool(ledger.get('liquidation_event', pd.Series(False, index=ledger.index)).any()),
        n_resets=0, n_effective_decisions=int(ledger.decision_available.sum()),
        n_context_available=int(ledger.context_available.sum()),
        percentage_metrics_available=True, log_metric_available=bool(np.isfinite(logs).all()))
    for name in ('fee', 'borrow', 'funding', 'liquidation_fee', 'turnover'):
        result[name+'_sum'] = float(ledger[name].sum()) if name in ledger else 0.
    decision = ledger.decision_available.to_numpy(bool)
    requested = ledger.requested_position.to_numpy(float)
    for cap, suffix in ((1., '1'), (1.12, '112'), (1.5, '15'), (2., '2'), (3., '3')):
        result['requested_above'+suffix+'_decision_fraction'] = float((requested[decision]>cap).mean()) if decision.any() else np.nan
    return result


def _fold_source(path_spec, fold):
    sources = path_spec.get('sources', {})
    return sources.get(fold, sources.get(str(fold)))


def _sensitivity_config(settings, sensitivity):
    settings = copy.deepcopy(settings)
    sensitivity = sensitivity or {}
    for key, destination in (('mmr', 'maintenance_margin_rate'), ('slippage', 'slippage_rate')):
        if key in sensitivity:
            settings['accounting'][destination] = float(sensitivity[key])
    return settings


def evaluate_policy_path(cfg, path_spec, registration_sha, *, arrays=None):
    """Replay all folds without resets; sensitivity retains selected weights."""
    _gate(cfg, registration_sha)
    from . import wm_tail_transfer_training as training
    from . import wm_tail_transfer_policy as policy
    if path_spec['stage'] not in ('BC', 'RL'):
        raise ValueError('policy stage must be BC or RL')
    tasks = list(path_spec['tasks'])
    if not tasks or len({t['seed'] for t in tasks}) != 1 or np.any(np.diff([t['fold'] for t in tasks]) <= 0):
        raise ValueError('one seed and increasing folds required per continuous path')
    if {t['fold'] for t in tasks} != set(cfg['splits']['folds']):
        raise ValueError('every registered fold must appear in a policy path')
    path_id = policy_path_id(path_spec)
    destination = Path(cfg['output_dir'])/'evaluations/policy'/path_id
    existing = _completed(destination/'evaluation.json', registration_sha)
    if existing is not None:
        return existing
    data = data_module.load_data(cfg) if arrays is None else arrays
    calendar = np.asarray(data['index_ns'])
    folds = {m['fold']: m for m in benchmark.fold_definitions(cfg)}
    ledgers, state, sources = [], None, {}
    for task in tasks:
        meta = folds[task['fold']]
        ids = np.flatnonzero((calendar >= pd.Timestamp(meta['test_start']).value) &
                             (calendar < pd.Timestamp(meta['test_end']).value))
        if not len(ids) or np.any(np.diff(ids) != 1) or ledgers and ids[0] != int(ledgers[-1].raw_id.iloc[-1])+1:
            raise ValueError('folds must cover one uninterrupted Test calendar')
        producer_sources = _fold_source(path_spec, task['fold'])
        if producer_sources is None:
            raise ValueError('fixed producer source tasks are required for every fold')
        producer_manifest = training.prepare_producer_cache(cfg, task['fold'], task['seed'], registration_sha,
            producer_sources, split='test', arrays=data)
        policy_arrays = training.load_producer_cache(cfg, producer_manifest, arrays=data)
        policy_arrays['producer_manifest']=producer_manifest
        directory = Path(cfg['output_dir'])/'policies'/task['job']
        checkpoint = directory/('bc.pt' if path_spec['stage']=='BC' else 'selected.pt')
        fit = json.loads((directory/'fit.json').read_text())
        if not fit.get('complete') or fit['registration_sha256'] != registration_sha:
            raise ValueError('policy Train/Val fit must be frozen before Test')
        expected = fit['artifact_sha256'].get(checkpoint.name)
        if expected != benchmark.sha(checkpoint):
            raise ValueError('policy checkpoint changed after selection')
        model, obj = policy.load_policy(checkpoint, registration_sha=registration_sha)
        settings = _sensitivity_config(obj['config'], path_spec.get('sensitivity') or {})
        # Risk gate and actor account features depend on sensitivity settings;
        # context generation is deterministic and the checkpoint stays fixed.
        context = policy.encode_test_contexts_cached(model, policy_arrays, ids, settings,
            Path(cfg['output_dir'])/'test_market_contexts')
        ledger, state = policy.replay_cached(model, context, policy_arrays, ids, settings, initial_state=state)
        ledger['fold'] = int(task['fold']); ledger['path_id'] = path_id
        ledgers.append(ledger)
        sources[task['job']] = {'fit.json': benchmark.sha(directory/'fit.json'),
                                checkpoint.name: expected,
                                'Test_market_context_manifest_sha256':context.get('cache_manifest_sha256')}
    ledger = pd.concat(ledgers, ignore_index=True)
    metrics = policy_metrics(ledger, annual_bars=cfg['accounting'].get('annual_bars', 35040))
    destination.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(destination/'ledger.parquet', index=False)
    _json(destination/'metrics.json', metrics)
    first = tasks[0]
    result = dict(complete=True, registration_sha256=registration_sha, path_id=path_id,
        stage=path_spec['stage'], policy_variant=first['policy_variant'], seed=first['seed'],
        control=first['control'], sensitivity=path_spec.get('sensitivity') or {},
        oracle=first['policy_variant']=='R07', scope='diagnostic_oracle' if first['policy_variant']=='R07' else 'deployable_inputs_historical_research',
        tasks=tasks, checkpoint_sha256=sources, metrics=metrics,
        n_resets=0, Test_used_for_selection=False,
        artifact_sha256={name: benchmark.sha(destination/name) for name in ('ledger.parquet', 'metrics.json')})
    _json(destination/'evaluation.json', result)
    return result


def _policy_signature(path_spec, *, seed=False, stage=True):
    first = path_spec['tasks'][0]
    identity = {'variant': first['policy_variant'], 'control': first['control'],
                'sensitivity': path_spec.get('sensitivity') or {}}
    if seed:
        identity['seed'] = first['seed']
    if stage:
        identity['stage'] = path_spec['stage']
    return json.dumps(identity, sort_keys=True, separators=(',', ':'))


def _forecast_signature(task):
    ignored = {'fold', 'seed', 'job', 'source_task', 'source_tasks', 'source_job', 'source_checkpoint', 'members'}
    return json.dumps({k: v for k, v in task.items() if k not in ignored}, sort_keys=True, separators=(',', ':'))


def comparison_plan(cfg, forecast_jobs, path_specs):
    """Construct hypotheses from registered tasks, without reading outcomes.

    The returned JSON should be frozen into the global Test barrier. Pairwise
    contrasts use identical raw timestamp cohorts; unavailable hypotheses
    remain in the fixed Holm family with p1.
    """
    forecast_groups, policy_groups = defaultdict(list), defaultdict(list)
    ensemble_specs=ensemble_jobs(cfg,forecast_jobs)
    for task in list(forecast_jobs)+ensemble_specs:
        forecast_groups[_forecast_signature(task)].append(task)
    for spec in path_specs:
        policy_groups[_policy_signature(spec)].append(spec)
    statistical_heads={'zero_return':['return'],
        'train_climatology':None,'past_variance':['rv','up_semivariance','down_semivariance'],
        'ewma':['rv','up_semivariance','down_semivariance'],
        'garch_gaussian':['rv','up_semivariance','down_semivariance','return_quantiles','var_es'],
        'gjr_student_t':['rv','up_semivariance','down_semivariance','return_quantiles','var_es'],
        'historical_residual':['return_quantiles','var_es'],'garch_evt':['return_quantiles','var_es']}
    statistical_groups={}
    for model in statistical_heads:
        members=[dict(job=f'statistical_f{fold:02d}__{model}',fold=fold,seed=cfg['seeds'][0],role='statistical',
                      group='D03',cohort='dense_common',model=model,objectives='joint') for fold in cfg['splits']['folds']]
        signature=_forecast_signature(members[0]);statistical_groups[signature]=members
    from .wm_tail_transfer_models import build_specs, active_specs, CLASSIFICATION_KINDS
    specs=build_specs(cfg)
    contrasts = []
    for signature, members in sorted(forecast_groups.items()):
        series = hashlib.sha256(signature.encode()).hexdigest()[:12]
        active=active_specs(specs,members[0].get('objectives','joint'))
        available_tasks=sorted({s['task'] for s in active} & set(MAIN_TASKS))
        probability_tasks=sorted({s['task'] for s in active if s['kind'] in CLASSIFICATION_KINDS} & set(MAIN_TASKS))
        for target in available_tasks:
            contrasts.append(dict(name=f'forecast_baseline:{series}:{target}', family='forecast_vs_Train_baseline',
                kind='forecast', target=target, left=signature, right='Train_baseline',
                calibration='none', members=[t['job'] for t in members]))
        for calibration in ('parametric', 'isotonic'):
            for target in probability_tasks:
                contrasts.append(dict(name=f'calibration:{series}:{calibration}:{target}', family='probability_calibration',
                    kind='forecast', target=target, left=signature, right=signature,
                    calibration=calibration, reference_calibration='none', members=[t['job'] for t in members]))
    # Model/input/learning/representation contrasts are defined structurally,
    # before outcomes are available. Every pair evaluates the same native
    # heads, so regression-only is never scored against a different target.
    signatures=list(sorted(forecast_groups))
    configurations={key:json.loads(key) for key in signatures}
    pairs=[]
    for key in signatures:
        task=configurations[key]
        for reference in signatures:
            ref=configurations[reference]
            family=None
            if task.get('role')=='benchmark' and ref.get('role')=='benchmark':
                same={k:v for k,v in task.items() if k not in ('model','group')}
                refsame={k:v for k,v in ref.items() if k not in ('model','group')}
                if same==refsame:
                    if task.get('group')==ref.get('group') and ref.get('model')=='lightgbm' and task.get('model')!='lightgbm':
                        family='forecast_model_comparison'
                    elif task.get('model')==ref.get('model') and ref.get('group')=='D03' and task.get('group')!='D03':
                        family='forecast_input_comparison'
            if task.get('role')=='ablation' and ref.get('role')=='benchmark' and ref.get('group')=='D03' and task.get('model')==ref.get('model'):
                if task.get('cohort','dense_common')==ref.get('cohort','dense_common') and ref.get('objectives')=='joint':
                    family='forecast_learning_representation'
            if task.get('role')=='producer' and ref.get('role')=='producer' and task.get('representation') in ('tail','tail_only') and ref.get('representation')=='old':
                family='forecast_representation_information'
            if task.get('role')=='ensemble' and ref.get('role')=='benchmark' and task.get('group')==ref.get('group') and task.get('model')=='ensemble_'+str(ref.get('model')) and task.get('cohort')==ref.get('cohort'):
                family='forecast_seed_ensemble'
            if family:
                pairs.append((key,reference,family))
    for left,right,family in pairs:
        leftspecs=active_specs(specs,configurations[left].get('objectives','joint'))
        rightspecs=active_specs(specs,configurations[right].get('objectives','joint'))
        common={s['name'] for s in leftspecs}&{s['name'] for s in rightspecs}
        for target in sorted({s['task'] for s in leftspecs if s['name'] in common}&set(MAIN_TASKS)):
            heads=[s['name'] for s in leftspecs if s['task']==target and s['name'] in common]
            identifier=hashlib.sha256((left+right+target).encode()).hexdigest()[:16]
            contrasts.append(dict(name=f'forecast_pair:{identifier}:{target}',family=family,kind='forecast',
                target=target,left=left,right=right,calibration='none',shared_heads=heads,
                direction=configurations[left].get('direction'),
                members=[t['job'] for t in forecast_groups[left]],reference_members=[t['job'] for t in forecast_groups[right]]))
    for left,members in sorted(forecast_groups.items()):
        task=configurations[left]
        if task.get('role')!='benchmark' or task.get('group')!='D03':
            continue
        active=active_specs(specs,task.get('objectives','joint'))
        for right,references in statistical_groups.items():
            model=references[0]['model'];wanted=statistical_heads[model]
            for target in sorted({s['task'] for s in active if wanted is None or s['name'] in wanted}&set(MAIN_TASKS)):
                heads=[s['name'] for s in active if s['task']==target and (wanted is None or s['name'] in wanted)]
                identifier=hashlib.sha256((left+right+target).encode()).hexdigest()[:16]
                contrasts.append(dict(name=f'forecast_statistical:{identifier}:{target}',family='forecast_vs_statistical',kind='forecast',
                    target=target,left=left,right=right,calibration='none',shared_heads=heads,
                    members=[t['job'] for t in members],reference_members=[t['job'] for t in references]))
    for signature, members in sorted(policy_groups.items()):
        first = members[0]; task = first['tasks'][0]
        if task['policy_variant']=='R07':
            continue  # Oracle diagnostics are never deployable superiority claims.
        series = hashlib.sha256(signature.encode()).hexdigest()[:12]
        for name in ('BH', 'constant08', 'constant_cap', 'MA1920', 'voltarget672', 'MLrule'):
            contrasts.append(dict(name=f'policy_baseline:{series}:{name}', family='policy_vs_matched_baselines', kind='policy',
                left=signature, right='baseline:'+name, members=[policy_path_id(p) for p in members]))
        if first['stage']=='RL':
            bc = copy.deepcopy(first); bc['stage']='BC'
            reference = _policy_signature(bc)
            if reference in policy_groups:
                contrasts.append(dict(name=f'RL_vs_BC:{series}', family='RL_vs_BC', kind='policy', left=signature,
                    right=reference, members=[policy_path_id(p) for p in members],
                    reference_members=[policy_path_id(p) for p in policy_groups[reference]]))
        if task['policy_variant']!='R00':
            reference = copy.deepcopy(first)
            for row in reference['tasks']:
                row['policy_variant']='R00'
            reference = _policy_signature(reference)
            if reference in policy_groups:
                contrasts.append(dict(name=f'policy_vs_R00:{series}', family='policy_information_transfer', kind='policy',
                    left=signature, right=reference, members=[policy_path_id(p) for p in members],
                    reference_members=[policy_path_id(p) for p in policy_groups[reference]]))
        # Adjacent leverage limits and control-objective changes are explicit
        # additional families, independent of the observed best condition.
        control=task['control']
        references=[]
        if not first.get('sensitivity') and control.get('cap') in (1.5,2.,5.):
            previous={1.5:1.12,2.:1.5,5.:3.}[control['cap']]
            ref=copy.deepcopy(first)
            for item in ref['tasks']: item['control']['cap']=previous
            references.append((_policy_signature(ref),'policy_leverage_increment'))
        standard={**control,'speed':.08,'bc_prior':.01,'controller':'direct','objective':'net_log_dd','risk_gate':False}
        if control!=standard:
            ref=copy.deepcopy(first)
            for item in ref['tasks']: item['control']=standard
            references.append((_policy_signature(ref),'policy_control_objective'))
        if first.get('sensitivity'):
            ref=copy.deepcopy(first);ref['sensitivity']={}
            references.append((_policy_signature(ref),'perpetual_parameter_sensitivity'))
        for reference,family in references:
            if reference in policy_groups:
                contrastid=hashlib.sha256((signature+reference).encode()).hexdigest()[:16]
                contrasts.append(dict(name=f'policy_pair:{contrastid}',family=family,kind='policy',left=signature,right=reference,
                    members=[policy_path_id(p) for p in members],reference_members=[policy_path_id(p) for p in policy_groups[reference]]))
    for contrast in contrasts:
        if contrast['kind']=='forecast':
            contrast['family'] += ':'+contrast['target']
        elif contrast['right'].startswith('baseline:'):
            contrast['family'] += ':'+contrast['right'].split(':',1)[1]
    return dict(version=1, forecast_groups=dict(forecast_groups), policy_groups=dict(policy_groups),
        contrasts=contrasts, n_comparisons=len(contrasts),
        null='paired mean forecast loss improvement or annualized net log growth difference is zero',
        positive_estimate='left improves upon right', cohort='intersection of fixed timestamps, identical across all paired seeds',
        Test_selection=False)


def _loss_path(cfg, job):
    return Path(cfg['output_dir'])/'evaluations/forecast'/job/'per_row_losses.npz'


def _load_loss(path, calibration, target):
    with np.load(path, allow_pickle=False) as arrays:
        key = f'{calibration}__{target}__'
        if key+'model_loss' not in arrays:
            return None
        return {'timestamps': arrays['index_ns'], 'model': arrays[key+'model_loss'],
                'baseline': arrays[key+'baseline_loss'], 'valid': arrays[key+'valid'].astype(bool)}


def _load_shared_head_loss(path, calibration, heads, direction=None):
    with np.load(path,allow_pickle=False) as arrays:
        keys=[f'{calibration}__head_{name}'+('_'+direction if direction in ('up','down') and name in ('reach_fixed','reach_dynamic') else '')+'__' for name in heads]
        if any(key+'model_loss' not in arrays for key in keys):
            return None
        values=np.column_stack([arrays[key+'model_loss'] for key in keys])
        valid=np.column_stack([arrays[key+'valid'].astype(bool) for key in keys])
        counts=valid.sum(1)
        combined=np.divide(np.where(valid,values,0).sum(1),counts,out=np.full(len(counts),np.nan),where=counts>0)
        return dict(timestamps=arrays['index_ns'],model=combined,valid=counts>0)


def _series_panel(rows):
    """Concatenate folds within seeds, then align common physical timestamps."""
    by_seed = defaultdict(list)
    for seed, times, values, valid in rows:
        by_seed[int(seed)].append((np.asarray(times, np.int64), np.asarray(values, float), np.asarray(valid, bool)))
    panels = {}
    for seed, parts in by_seed.items():
        times = np.concatenate([p[0] for p in parts]); values = np.concatenate([p[1] for p in parts]); valid = np.concatenate([p[2] for p in parts])
        order = np.argsort(times, kind='stable'); times, values, valid = times[order], values[order], valid[order]
        if np.any(np.diff(times)<=0):
            raise ValueError('a forecast/policy seed has overlapping fold timestamps')
        panels[seed] = (times, values, valid)
    common = None
    for times, _, _ in panels.values():
        common = times if common is None else np.intersect1d(common, times, assume_unique=True)
    if common is None:
        common = np.empty(0, np.int64)
    values, masks = [], []
    for seed in sorted(panels):
        times, vector, valid = panels[seed]; locations = np.searchsorted(times, common)
        values.append(vector[locations]); masks.append(valid[locations])
    if not values:
        return common, np.empty((0, 1)), np.empty((0, 1), bool), []
    values, mask = np.column_stack(values), np.column_stack(masks)
    # Exclude missing labels on the SAME rows for all seeds, never nonfinite
    # observed losses: the inference engine must invalidate such a contrast.
    mask[:] = mask.all(1)[:, None]
    return common, values, mask, sorted(panels)


def _empty_comparison(family, kind, metadata=None):
    return dict(timestamps=np.empty(0, np.int64), values=np.empty((0, 1)), valid=np.empty((0, 1), bool),
                family=family, kind=kind, metadata=metadata or {})


def build_comparisons(cfg, plan, forecast_records, policy_records, baseline_records):
    """Build preregistered paired contrasts; absent cells stay unavailable."""
    fmeta = {r['task']['job']: r for r in forecast_records}
    pmeta = {r['path_id']: r for r in policy_records}
    bmeta = {r['baseline_key']: r for r in baseline_records}
    result = {}
    for contrast in plan['contrasts']:
        rows, unavailable = [], False
        if contrast['kind']=='forecast':
            reference_jobs={}
            for job in contrast.get('reference_members',[]):
                if job in fmeta:
                    task=fmeta[job]['task'];reference_jobs[(task['fold'],task['seed'])]=job
            for job in contrast['members']:
                record = fmeta.get(job)
                if record is None or record.get('available') is False:
                    unavailable=True; break
                part = (_load_shared_head_loss(_loss_path(cfg,job),contrast['calibration'],contrast['shared_heads'],contrast.get('direction'))
                        if 'shared_heads' in contrast else _load_loss(_loss_path(cfg, job), contrast['calibration'], contrast['target']))
                if part is None:
                    unavailable=True; break
                if contrast['right']=='Train_baseline':
                    values = part['baseline']-part['model']; valid=part['valid']
                elif 'shared_heads' not in contrast:
                    ref = _load_loss(_loss_path(cfg, job), contrast['reference_calibration'], contrast['target'])
                    if ref is None or not np.array_equal(part['timestamps'], ref['timestamps']):
                        unavailable=True; break
                    values = ref['model']-part['model']; valid=part['valid'] & ref['valid']
                else:
                    task=record['task'];refjob=reference_jobs.get((task['fold'],task['seed']))
                    if refjob is None:
                        candidates=[value for (fold,seed),value in reference_jobs.items() if fold==task['fold']]
                        refjob=candidates[0] if len(candidates)==1 else None
                    if refjob is None:
                        unavailable=True;break
                    if fmeta[refjob].get('available') is False:
                        unavailable=True;break
                    ref=_load_shared_head_loss(_loss_path(cfg,refjob),contrast['calibration'],contrast['shared_heads'],contrast.get('direction'))
                    if ref is None:
                        unavailable=True;break
                    common,li,ri=np.intersect1d(part['timestamps'],ref['timestamps'],assume_unique=True,return_indices=True)
                    part={**part,'timestamps':common,'model':part['model'][li],'valid':part['valid'][li]}
                    values=ref['model'][ri]-part['model'];valid=part['valid']&ref['valid'][ri]
                # Missing Train reference is unavailable rather than a false
                # improvement; model nonfinite at observed labels stays valid.
                if contrast['right']=='Train_baseline':
                    valid = valid & np.isfinite(part['baseline'])
                rows.append((record['task']['seed'], part['timestamps'], values, valid))
        else:
            references = {pmeta[x]['seed']: pmeta[x] for x in contrast.get('reference_members', []) if x in pmeta}
            for path_id in contrast['members']:
                record=pmeta.get(path_id)
                if record is None:
                    unavailable=True; break
                left = pd.read_parquet(Path(cfg['output_dir'])/'evaluations/policy'/path_id/'ledger.parquet', columns=['timestamp','net_log_return'])
                if contrast['right'].startswith('baseline:'):
                    key = baseline_key(record['control'], record.get('sensitivity', {}), contrast['right'].split(':', 1)[1])
                    baseline=bmeta.get(key)
                    if baseline is None:
                        unavailable=True; break
                    right=pd.read_parquet(baseline['ledger_path'], columns=['timestamp','net_log_return'])
                else:
                    ref=references.get(record['seed'])
                    if ref is None:
                        unavailable=True; break
                    right=pd.read_parquet(Path(cfg['output_dir'])/'evaluations/policy'/ref['path_id']/'ledger.parquet', columns=['timestamp','net_log_return'])
                lt=pd.DatetimeIndex(left.timestamp).asi8; rt=pd.DatetimeIndex(right.timestamp).asi8
                if not np.array_equal(lt,rt):
                    raise ValueError('policy contrasts require identical full market calendars')
                rows.append((record['seed'],lt,left.net_log_return.to_numpy()-right.net_log_return.to_numpy(),np.ones(len(left),bool)))
        if unavailable or not rows:
            result[contrast['name']]=_empty_comparison(contrast['family'],contrast['kind'],contrast)
        else:
            times,values,valid,seeds=_series_panel(rows)
            result[contrast['name']]=dict(timestamps=times,values=values,valid=valid,
                family=contrast['family'],kind=contrast['kind'],metadata={**contrast,'seed_ids':seeds})
    return result


def baseline_key(control, sensitivity, name):
    # Controller/objective/BC prior do not change the execution environment.
    environment={k:control[k] for k in ('cap','speed','account_mode','mmr','slippage') if k in control}
    payload=json.dumps(dict(environment=environment,sensitivity=sensitivity or {},name=name),sort_keys=True)
    return name+'_'+hashlib.sha256(payload.encode()).hexdigest()[:14]


def baseline_path_specs(cfg, path_specs):
    """One deterministic representative producer per execution environment."""
    selected={}
    for spec in path_specs:
        if spec['tasks'][0]['policy_variant']=='R07':
            continue
        key=baseline_key(spec['tasks'][0]['control'],spec.get('sensitivity') or {},'BH')
        order={seed:i for i,seed in enumerate(cfg['seeds'])}
        if key not in selected or order.get(spec['tasks'][0]['seed'],999)<order.get(selected[key]['tasks'][0]['seed'],999):
            selected[key]=copy.deepcopy(spec)
    output=[]
    for key,spec in sorted(selected.items()):
        spec['job']='baselines_'+key
        output.append(spec)
    return output


def ensemble_jobs(cfg, jobs):
    groups=defaultdict(list)
    for task in jobs:
        if task.get('role')!='benchmark' or task.get('model') not in ('lstm','transformer','wm') or task.get('group')!='D03':
            continue
        identity={k:v for k,v in task.items() if k not in ('job','seed')}
        groups[json.dumps(identity,sort_keys=True)].append(task)
    result=[]
    required=set(cfg['ablation']['ensemble_members'])
    for signature,members in sorted(groups.items()):
        selected={t['seed']:t for t in members if t['seed'] in required}
        if set(selected)!=required:
            continue
        first=next(iter(selected.values()))
        digest=hashlib.sha256(signature.encode()).hexdigest()[:14]
        result.append({**first,'role':'ensemble','model':'ensemble_'+first['model'],'seed':cfg['seeds'][0],
                       'job':'ensemble_'+digest,'members':[selected[s] for s in sorted(required)],'item':'L07'})
    return result


def rollout_jobs(cfg, jobs):
    result=[]
    for task in jobs:
        ordinary=(task.get('role')=='benchmark' and task.get('group')=='D03' and task.get('model')=='wm'
                  and task.get('cohort')=='dense_common'
                  and task.get('fold') in cfg['screen']['folds'] and task.get('seed') in cfg['screen']['seeds'])
        weighted=task.get('item')=='L08' and task.get('model')=='wm'
        if ordinary or weighted:
            result.append(dict(job='rollout_'+task['job'],source=task,fold=task['fold'],seed=task['seed'],item='W07'))
    return result


def evaluate_ensemble_job(cfg, task, registration_sha, *, arrays=None):
    """Fixed seed ensemble; predictions and calibration were already frozen."""
    _gate(cfg,registration_sha)
    destination=Path(cfg['output_dir'])/'evaluations/forecast'/task['job']
    existing=_completed(destination/'evaluation.json',registration_sha)
    if existing is not None:
        return existing
    from . import wm_tail_transfer_training as training
    members=task['members'];directories=[Path(cfg['output_dir'])/'evaluations/forecast'/member['job'] for member in members]
    manifests=[_completed(directory/'evaluation.json',registration_sha) for directory in directories]
    if any(m is None for m in manifests):
        raise ValueError('all selected ensemble members must finish report-only inference first')
    if any(m.get('available') is False for m in manifests):
        return _unavailable_forecast(cfg,task,registration_sha,{},destination,reason='one or more fixed ensemble members lack Train/Val data')
    shared=set(manifests[0]['calibrations'])
    for manifest in manifests[1:]:
        shared &= set(manifest['calibrations'])
    specs=manifests[0]['native_head_specs']
    if any(m['native_head_specs']!=specs for m in manifests[1:]):
        raise ValueError('ensemble members need identical native target contracts')
    by_calibration={};uncertainty={}
    packs=[np.load(directory/'test_predictions.npz',allow_pickle=False) for directory in directories]
    try:
        ids=packs[0]['raw_ids']
        if any(not np.array_equal(ids,p['raw_ids']) for p in packs[1:]):
            raise ValueError('ensemble members require identical timestamp cohorts')
        for calibration in CALIBRATIONS:
            if calibration not in shared:
                continue
            by_calibration[calibration]={}
            for spec in specs:
                key=calibration+'__'+spec['name']
                values=np.stack([p[key] for p in packs])
                by_calibration[calibration][spec['name']]=values.mean(0)
                uncertainty[key]=values.std(0,ddof=1).astype(np.float32)
    finally:
        for pack in packs:pack.close()
    data=data_module.load_data(cfg) if arrays is None else arrays
    _,_,splits=training.split_arrays(cfg,members[0],data)
    report_task={k:v for k,v in task.items() if k!='members'}
    report=_write_forecast(cfg,report_task,registration_sha,data,ids,splits['train'],specs,by_calibration,destination,
        source_sha256={member['job']:benchmark.sha(directory/'evaluation.json') for member,directory in zip(members,directories)})
    np.savez_compressed(destination/'member_uncertainty.npz',raw_ids=ids,**uncertainty)
    report['artifact_sha256']['member_uncertainty.npz']=benchmark.sha(destination/'member_uncertainty.npz')
    report['ensemble_members']=[member['job'] for member in members]
    report['n_new_training_fits']=0
    _json(destination/'evaluation.json',report)
    return report


def _sample_crps(samples, truth):
    samples=np.sort(np.asarray(samples,float),axis=1)
    weights=(2*np.arange(1,samples.shape[1]+1)-samples.shape[1]-1)/samples.shape[1]**2
    return np.mean(np.abs(samples-truth[:,None]),axis=1)-(samples*weights).sum(1)


def rollout_diagnostics(samples, actual_returns, cfg):
    """Close-path diagnostics; intrabar highs/lows are not reconstructed."""
    samples=np.asarray(samples,float);truth=np.asarray(actual_returns,float)
    if samples.ndim!=3 or truth.shape!=(len(samples),samples.shape[2]):
        raise ValueError('sample paths[B,draws,H] and actual close returns[B,H] required')
    rows=[]
    for horizon in cfg['targets']['horizons']:
        draws=samples[:,:,:horizon];observed=truth[:,:horizon]
        valid=np.isfinite(draws).all((1,2)) & np.isfinite(observed).all(1)
        cumulative=np.cumsum(draws,axis=-1);real=np.cumsum(observed,axis=-1)
        # Work in log price: this remains defined without clipping even when
        # an unphysical sampled path would overflow an exponentiation.
        logpeak=np.maximum.accumulate(np.concatenate([np.zeros((*cumulative.shape[:2],1)),cumulative],axis=-1),axis=-1)[:,:,1:]
        realpeak=np.maximum.accumulate(np.c_[np.zeros(len(real)),real],axis=-1)[:,1:]
        with np.errstate(invalid='ignore'):
            maxdd=np.max(-np.expm1(np.minimum(0,cumulative-logpeak)),axis=-1)
        realdd=np.max(-np.expm1(np.minimum(0,real-realpeak)),axis=-1)
        for name,forecast,actual in [('terminal_log_return',cumulative[:,:,-1],real[:,-1]),
                                     ('close_path_maxDD',maxdd,realdd),
                                     ('realized_variance',np.square(draws).sum(-1),np.square(observed).sum(-1))]:
            vv=valid & np.isfinite(forecast).all(1) & np.isfinite(actual)
            row=dict(horizon=horizon,diagnostic=name,n=int(vv.sum()),invalid_decisions=int((~vv).sum()),
                     n_sample_paths=int(np.prod(forecast.shape)),observation_basis='15m_close_to_close_not_intrabar_high_low')
            if vv.any():
                values=forecast[vv];yy=actual[vv]
                q=np.quantile(values,[.01,.05,.5,.95,.99],axis=1).T
                row.update(crps=float(_sample_crps(values,yy).mean()),
                    predicted_mean=float(values.mean()),actual_mean=float(yy.mean()),
                    coverage90=float(((yy>=q[:,1])&(yy<=q[:,3])).mean()),
                    lower5_exceedance=float((yy<=q[:,1]).mean()),upper95_exceedance=float((yy>=q[:,3]).mean()))
            rows.append(row)
        for threshold in cfg['targets']['fixed_thresholds']:
            for direction in ('down','up'):
                threshold_log=math.log1p(-threshold) if direction=='down' else math.log1p(threshold)
                probability=(cumulative.min(-1)<=threshold_log).mean(1) if direction=='down' else (cumulative.max(-1)>=threshold_log).mean(1)
                event=(real.min(-1)<=threshold_log) if direction=='down' else (real.max(-1)>=threshold_log)
                rows.append(dict(horizon=horizon,diagnostic='close_reach',direction=direction,threshold=threshold,
                    n=int(valid.sum()),invalid_decisions=int((~valid).sum()),
                    brier=float(np.mean((probability[valid]-event[valid])**2)) if valid.any() else np.nan,
                    predicted_probability=float(probability[valid].mean()) if valid.any() else np.nan,
                    observed_probability=float(event[valid].mean()) if valid.any() else np.nan,
                    observation_basis='15m_close_to_close_not_intrabar_high_low'))
        for values,source in ((draws,'model'),(observed[:,None,:],'actual')):
            longest=np.zeros(values.shape[:2],int);run=np.zeros_like(longest)
            for time in range(horizon):
                run=np.where(values[:,:,time]<0,run+1,0);longest=np.maximum(longest,run)
            rows.append(dict(horizon=horizon,diagnostic='longest_negative_return_run',source=source,
                n=int(valid.sum()),invalid_decisions=int((~valid).sum()),
                mean_longest_run=float(longest[valid].mean()) if valid.any() else np.nan,
                observation_basis='15m_close_to_close_not_intrabar_high_low'))
    return pd.DataFrame(rows)


def evaluate_rollout_job(cfg, task, registration_sha, *, arrays=None):
    """Fixed random draws from genuine WM transitions, with no future refresh."""
    _gate(cfg,registration_sha)
    from . import wm_tail_transfer_training as training
    destination=Path(cfg['output_dir'])/'evaluations/rollout'/task['job']
    existing=_completed(destination/'evaluation.json',registration_sha)
    if existing is not None:
        return existing
    source=task['source'];data=data_module.load_data(cfg) if arrays is None else arrays
    model,norm,fit,_=training.load_fit(cfg,source,registration_sha)
    if not hasattr(model,'sample_rollout') or getattr(model,'variant',None)!='wm':
        raise ValueError('W07 needs a trained stochastic WM, not interpolated point predictions')
    settings=cfg['rollout_diagnostics'];meta=next(x for x in benchmark.fold_definitions(cfg) if x['fold']==task['fold'])
    eligible=benchmark.split_ids(data,source['group'],meta['test_start'],meta['test_end'],cohort=source.get('cohort'))
    ids=eligible[np.linspace(0,len(eligible)-1,min(len(eligible),int(settings['decision_rows'])),dtype=int)]
    if not len(ids):
        raise ValueError('W07 has no fully observed eligible Test decisions')
    data=training.attach_window_cache(cfg,data,source['group'])
    torch.manual_seed(int(settings['seed'])+int(task['fold'])*100+int(task['seed']))
    model.to(cfg.get('device','mps')).eval();parts=[]
    for start in range(0,len(ids),int(settings['batch_size'])):
        selected=ids[start:start+int(settings['batch_size'])]
        x=((training._windows(data,selected,source['group'],cfg)-norm['mean'])/norm['std']).clip(-10,10).astype(np.float32)
        parts.append(model.sample_rollout(torch.from_numpy(x).to(cfg.get('device','mps')),
            int(settings['horizon']),int(settings['n_samples'])).cpu().numpy())
    samples=np.concatenate(parts);close=np.asarray(data['ohlcspot'][:,3],float)
    offsets=ids[:,None]+np.arange(int(settings['horizon']))[None,:]
    actual=np.log(close[offsets]/close[offsets-1])
    diagnostics=rollout_diagnostics(samples,actual,cfg)
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'sampled_paths.npz',raw_ids=ids,index_ns=np.asarray(data['index_ns'])[ids],sampled_log_returns=samples,actual_log_returns=actual)
    diagnostics.to_csv(destination/'diagnostics.csv',index=False)
    result=dict(complete=True,registration_sha256=registration_sha,task=task,n_rows=len(ids),
        invalid_sample_paths=int((~np.isfinite(samples).all(-1)).sum()),genuine_learned_stochastic_transitions=True,
        future_market_refresh=False,source_fit_sha256=benchmark.sha(training.fit_directory(cfg,source)/'fit.json'),
        diagnostic_scope='15mclosepaths; intrabar extremes and arrival order are not claimed',
        artifact_sha256={name:benchmark.sha(destination/name) for name in ('sampled_paths.npz','diagnostics.csv')})
    _json(destination/'evaluation.json',result)
    return result


def evaluate_baseline_paths(cfg, path_spec, registration_sha, *, arrays=None):
    """Matched BH/8%/cap/MA/vol/ML rules through the SAME account engine."""
    _gate(cfg,registration_sha)
    from . import wm_tail_transfer_policy as policy
    from . import wm_tail_transfer_training as training
    from . import wm_tail_transfer_accounting as account
    data=data_module.load_data(cfg) if arrays is None else arrays
    first=path_spec['tasks'][0]; settings=_sensitivity_config(policy.effective_config(cfg,first),path_spec.get('sensitivity',{}))
    control,sensitivity=first['control'],path_spec.get('sensitivity') or {}
    names=('BH','constant08','constant_cap','MA1920','voltarget672','MLrule')
    destinations={name:Path(cfg['output_dir'])/'evaluations/baselines'/baseline_key(control,sensitivity,name) for name in names}
    existing=[_completed(destinations[name]/'evaluation.json',registration_sha) for name in names]
    if all(r is not None for r in existing):
        return existing
    folds={m['fold']:m for m in benchmark.fold_definitions(cfg)}
    close=np.asarray(data['ohlcperp' if settings['accounting'].get('mode')=='linear_perp' else 'ohlcspot'][:,3],float)
    # shift(1) ensures the current or future bar cannot affect a request.
    trailing_mean=pd.Series(close).rolling(1920,min_periods=1920).mean().shift(1).to_numpy()
    returns=pd.Series(np.r_[np.nan,np.diff(np.log(close))])
    volatility=np.sqrt(returns.pow(2).rolling(672,min_periods=672).mean().shift(1).to_numpy()*settings['accounting'].get('annual_bars',35040))
    state=None; collected={name:[] for name in names}
    cap=float(settings['accounting']['position_max']); speed=float(settings['accounting']['max_position_step'])
    for task in path_spec['tasks']:
        meta=folds[task['fold']];calendar=np.asarray(data['index_ns'])
        ids=np.flatnonzero((calendar>=pd.Timestamp(meta['test_start']).value)&(calendar<pd.Timestamp(meta['test_end']).value))
        manifest=training.prepare_producer_cache(cfg,task['fold'],task['seed'],registration_sha,_fold_source(path_spec,task['fold']),split='test',arrays=data)
        source=training.load_producer_cache(cfg,manifest,arrays=data)
        if state is None:
            sourceprice=source['ohlcmark'] if settings['accounting'].get('mode')=='linear_perp' else source['ohlcspot']
            state=account.initial_state(len(names),'cpu',np.repeat(float(sourceprice[ids[0]-1,3]),len(names)),settings)
        for raw_id in ids:
            available=bool(source['available'][raw_id]) and np.isfinite(trailing_mean[raw_id]) and np.isfinite(volatility[raw_id])
            previous=state['prev_equity'].numpy().copy()
            sigma=float(source['risk_sigma'][raw_id]);up=float(source['up_probability'][raw_id]);down=float(source['down_probability'][raw_id])
            mlvol=sigma*math.sqrt(settings['accounting'].get('annual_bars',35040)/cfg['targets']['primary_horizon'])
            ml=min(cap,.1/max(mlvol,1e-5)) if math.isfinite(mlvol) else 0.
            ml=0. if down>=.5 else ml*(1. if up>down else .5)
            desired=np.array([1.,.08,cap,cap if close[raw_id-1]>trailing_mean[raw_id] else 0.,min(cap,.1/max(volatility[raw_id],1e-5)),ml])
            request=(state['requested']+torch.from_numpy(desired-state['requested'].numpy()).clamp(-speed,speed)).clamp(0,cap)
            active=np.full(len(names),available,bool);active[0]=False # BH keeps fixed endowed units; no rebalancing.
            state,diag=policy._advance(source,np.repeat(raw_id,len(names)),state,request,settings,torch.from_numpy(active))
            for col,name in enumerate(names):
                row={key:bool(v[col]) if v.dtype==torch.bool else float(v[col]) for key,v in diag.items()}
                row.update(raw_id=int(raw_id),timestamp=pd.Timestamp(int(calendar[raw_id]),tz='UTC'),
                    equity_previous=float(previous[col]),context_available=bool(available),fold=int(task['fold']))
                collected[name].append(row)
    result=[]
    for name in names:
        destination=destinations[name];destination.mkdir(parents=True,exist_ok=True)
        ledger=pd.DataFrame(collected[name]);ledger.to_parquet(destination/'ledger.parquet',index=False)
        metrics=policy_metrics(ledger,annual_bars=settings['accounting'].get('annual_bars',35040))
        record=dict(complete=True,registration_sha256=registration_sha,baseline=name,baseline_key=baseline_key(control,sensitivity,name),
            control=control,sensitivity=sensitivity,metrics=metrics,ledger_path=str(destination/'ledger.parquet'),
            reference_producer_seed=first['seed'],Test_used_for_selection=False,
            artifact_sha256={'ledger.parquet':benchmark.sha(destination/'ledger.parquet')})
        _json(destination/'evaluation.json',record);result.append(record)
    return result


def seed_summary(frame):
    keys=[k for k in ('stage','policy_variant','control_json','sensitivity_json','oracle') if k in frame]
    rows=[]
    for values,part in frame.groupby(keys,dropna=False,sort=False):
        row=dict(zip(keys,values if isinstance(values,tuple) else (values,)))
        row['n_seeds']=len(part);row['seed_ids']=','.join(map(str,part.seed.astype(int)))
        for name in ('net_total_return','max_drawdown','mean_exposure','exposure_max','fee_sum','borrow_sum','funding_sum','liquidation_fee_sum'):
            vector=part[name].to_numpy(float)
            for suffix,function in (('mean',np.mean),('std',lambda a:np.std(a,ddof=1) if len(a)>1 else np.nan),('min',np.min),('max',np.max)):
                row[name+'_'+suffix]=float(function(vector))
        worst=part.loc[part.max_drawdown.idxmax()]
        row.update(worst_drawdown=float(worst.max_drawdown),worst_drawdown_seed=int(worst.seed),
            n_bankrupt_seeds=int(part.bankrupt.sum()),n_liquidated_seeds=int(part.liquidation.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def _statuses(cfg, forecast_records, policy_records, requested_statuses=None, rollout_records=None):
    rows=[]
    rollout_records=rollout_records or []
    def append(item,members,note):
        available=[r for r in members if r.get('available') is not False]
        rows.append(dict(item=item,status=('completed_with_unavailable_cells' if len(available)<len(members) else 'completed') if available else 'unavailable',
            fit_count=sum(r['task'].get('role') not in ('ensemble','statistical') for r in available),
            unavailable_count=len(members)-len(available),output_count=sum(r.get('n_metric_coordinates',0) for r in available),note=note))
    for target in MAIN_TASKS:
        members=[r for r in forecast_records if any(s['task']==target for s in r['native_head_specs'])]
        append(target,members,'native outputs; coordinate/horizon/calibration counts reported separately')
    for group in [f'D{i:02d}' for i in range(8)]+['D07_dense']:
        append(group,[r for r in forecast_records if r['task'].get('group')==group],
               'fixed common cohort; sparse liquidation cohorts reported separately from dense history')
    predicates={
        'L01':lambda t:t.get('objectives') in ('regression','classification'),
        'L02':lambda t:t.get('direction') in ('up','down'),
        'L03':lambda t:t.get('architecture')=='hurdle',
        'L04':lambda t:t.get('item')=='L04',
        'L05':lambda t:t.get('item')=='L05',
        'L06':lambda t:True,
        'L07':lambda t:t.get('role')=='ensemble',
        'L08':lambda t:t.get('item')=='L08',
        'W01':lambda t:t.get('role')=='producer' and t.get('model') in ('linear','lightgbm'),
        'W02':lambda t:t.get('architecture')=='latent_probe',
        'W03':lambda t:t.get('architecture')=='frozen_probe',
        'W04':lambda t:t.get('representation')=='tail',
        'W05':lambda t:t.get('representation')=='tail_only',
        'W06':lambda t:t.get('role')=='benchmark' and t.get('model') in ('wm','transformer')}
    for item,predicate in predicates.items():
        members=[r for r in forecast_records if predicate(r['task']) and
                 (item!='L06' or set(r['calibrations'])==set(CALIBRATIONS))]
        append(item,members,'Train/Val-only calibration' if item=='L06' else 'registered screen/confirmation scope; no Test selection')
    rows.append(dict(item='W07',status='completed' if rollout_records else 'unavailable',fit_count=0,
        unavailable_count=0,output_count=len(rollout_records),note='genuine stochastic WM transitions; close-path-only diagnostics'))
    for variant in (f'R{i:02d}' for i in range(8)):
        members=[r for r in policy_records if r['policy_variant']==variant]
        rows.append(dict(item=variant,status='completed' if members else 'unavailable',fit_count=len(members),
                         output_count=len(members),note='diagnostic Oracle only' if variant=='R07' else 'continuous-calendar policy paths'))
    conditions={
        'P01':lambda r:r['stage'] in ('BC','RL'),
        'P02':lambda r:r['control'].get('controller')=='direct',
        'P03':lambda r:r['control'].get('controller')=='ml_rule',
        'P04':lambda r:r['control'].get('controller')=='hierarchical',
        'P05':lambda r:r['control'].get('risk_gate') is True,
        'P06':lambda r:r['control'].get('cap') in (1.12,1.5,2.),
        'P07':lambda r:r['control'].get('speed')==.24 or r['control'].get('bc_prior')==0.,
        'P08':lambda r:r['control'].get('objective') in ('net_log','episodic_cvar_5pct'),
        'P09':lambda r:r['control'].get('account_mode')=='linear_perp'}
    for item,predicate in conditions.items():
        members=[r for r in policy_records if predicate(r)]
        rows.append(dict(item=item,status='completed' if members else 'unavailable',fit_count=0,
            unavailable_count=0,output_count=len(members),note='policy fit count deduplicated separately; BC/RL replays and sensitivity are not new fits'))
    if requested_statuses:
        rows.extend(requested_statuses)
    return pd.DataFrame(rows)


def _validate_statistical_output_keyset(cfg, records, execution_manifest=None):
    """Require every declared fold/model output, including unavailable cells.

    A manual table aggregation may omit this check. Campaign aggregation uses
    the frozen execution manifest; standalone checks can derive the same
    deterministic job names from the registered fold configuration.
    """
    folds = [int(fold) for fold in cfg['splits']['folds']]
    jobs = (execution_manifest['statistical_jobs'] if execution_manifest is not None else
            [dict(job=f'statistical_f{fold:02d}', fold=fold) for fold in folds])
    identities = [(str(task['job']), int(task['fold'])) for task in jobs]
    if (len(set(identities)) != len(identities) or len({job for job, _ in identities}) != len(jobs)
            or sorted(fold for _, fold in identities) != sorted(folds)):
        raise ValueError('registered statistical fit jobs require unique exact fold coverage')
    expected = {(job+'__'+model, fold, model) for job, fold in identities for model in STATISTICAL_MODELS}
    actual = [(str(record['task']['job']), int(record['task']['fold']), str(record['task']['model']))
              for record in records if record['task'].get('role') == 'statistical']
    counts = defaultdict(int)
    for identity in actual:
        counts[identity] += 1
    duplicates = sorted(identity for identity, count in counts.items() if count > 1)
    missing = sorted(expected-set(actual))
    unexpected = sorted(set(actual)-expected)
    if missing or unexpected or duplicates:
        raise ValueError('registered statistical output keyset mismatch: '+json.dumps(dict(
            missing=missing, unexpected=unexpected, duplicates=duplicates), sort_keys=True))


def aggregate_report(cfg, forecast_jobs=None, path_specs=None, registration_sha=None, *, plan=None,
                     requested_statuses=None, run_inference=True):
    """Collect completed reports; no Test-driven selection occurs here."""
    if registration_sha is None and isinstance(forecast_jobs,str):
        registration_sha,forecast_jobs=forecast_jobs,None
    _gate(cfg,registration_sha)
    root=Path(cfg['output_dir']);destination=root/'final';destination.mkdir(parents=True,exist_ok=True)
    forecast=[json.loads(p.read_text()) for p in sorted((root/'evaluations/forecast').glob('*/evaluation.json'))]
    policies=[json.loads(p.read_text()) for p in sorted((root/'evaluations/policy').glob('*/evaluation.json'))]
    baselines=[json.loads(p.read_text()) for p in sorted((root/'evaluations/baselines').glob('*/evaluation.json'))]
    rollouts=[json.loads(p.read_text()) for p in sorted((root/'evaluations/rollout').glob('*/evaluation.json'))]
    for directory,records in (('forecast',forecast),('policy',policies),('baselines',baselines)):
        for record in records:
            identity=record['task']['job'] if directory=='forecast' else record['path_id'] if directory=='policy' else record['baseline_key']
            _completed(root/'evaluations'/directory/identity/'evaluation.json',registration_sha)
    expected_forecasts=({t['job'] for t in forecast_jobs}|{t['job'] for t in ensemble_jobs(cfg,forecast_jobs)}) if forecast_jobs is not None else None
    if forecast_jobs is not None and {r['task']['job'] for r in forecast if r['task'].get('role')!='statistical'}!=expected_forecasts:
        raise ValueError('registered forecast output keyset is incomplete or unexpected')
    if forecast_jobs is not None:
        execution_path = root/'execution_manifest.json'
        execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
        _validate_statistical_output_keyset(cfg, forecast, execution)
    if path_specs is not None and {r['path_id'] for r in policies}!={policy_path_id(s) for s in path_specs}:
        raise ValueError('registered policy path keyset is incomplete or unexpected')
    if path_specs is not None:
        expected_baselines={baseline_key(spec['tasks'][0]['control'],spec.get('sensitivity') or {},name)
            for spec in baseline_path_specs(cfg,path_specs) for name in ('BH','constant08','constant_cap','MA1920','voltarget672','MLrule')}
        if {r['baseline_key'] for r in baselines}!=expected_baselines:
            raise ValueError('registered matched baseline path keyset is incomplete or unexpected')
    if forecast_jobs is not None and {r['task']['job'] for r in rollouts}!={j['job'] for j in rollout_jobs(cfg,forecast_jobs)}:
        raise ValueError('registered stochastic rollout keyset is incomplete or unexpected')
    for record in rollouts:
        _completed(root/'evaluations/rollout'/record['task']['job']/'evaluation.json',registration_sha)
    metrics=pd.concat([pd.read_parquet(root/'evaluations/forecast'/r['task']['job']/'test_metrics.parquet') for r in forecast],ignore_index=True) if forecast else pd.DataFrame()
    metrics.to_parquet(destination/'forecast_metrics.parquet',index=False)
    taskmetrics=pd.concat([pd.read_csv(root/'evaluations/forecast'/r['task']['job']/'task_metrics.csv') for r in forecast],ignore_index=True) if forecast else pd.DataFrame()
    taskmetrics.to_csv(destination/'forecast_task_metrics.csv',index=False)
    policyrows=[]
    for record in policies:
        policyrows.append({k:record[k] for k in ('path_id','stage','policy_variant','seed','oracle')}|
            dict(control_json=json.dumps(record['control'],sort_keys=True),sensitivity_json=json.dumps(record['sensitivity'],sort_keys=True))|record['metrics'])
    policyframe=pd.DataFrame(policyrows);policyframe.to_csv(destination/'policy_metrics.csv',index=False)
    summary=seed_summary(policyframe) if len(policyframe) else pd.DataFrame()
    summary.to_csv(destination/'seed_summary.csv',index=False)
    baselineframe=pd.DataFrame([dict(baseline=r['baseline'],baseline_key=r['baseline_key'],control_json=json.dumps(r['control'],sort_keys=True),
        sensitivity_json=json.dumps(r['sensitivity'],sort_keys=True),**r['metrics']) for r in baselines])
    baselineframe.to_csv(destination/'baseline_metrics.csv',index=False)
    statuses=_statuses(cfg,forecast,policies,requested_statuses,rollouts);statuses.to_csv(destination/'status_matrix.csv',index=False)
    rolloutframe=pd.concat([pd.read_csv(root/'evaluations/rollout'/r['task']['job']/'diagnostics.csv').assign(
        job=r['task']['job'],fold=r['task']['fold'],seed=r['task']['seed']) for r in rollouts],ignore_index=True) if rollouts else pd.DataFrame()
    rolloutframe.to_csv(destination/'stochastic_rollout_diagnostics.csv',index=False)
    _json(destination/'path_manifest.json',[dict(path_id=r['path_id'],stage=r['stage'],seed=r['seed'],
        path=str(root/'evaluations/policy'/r['path_id']/'ledger.parquet'),sha256=r['artifact_sha256']['ledger.parquet']) for r in policies])
    if plan is None:
        frozen_plan=root/'comparison_plan.json'
        if not frozen_plan.exists() and run_inference:
            raise ValueError('comparison plan must be frozen before Test; outcome-driven discovery forbidden')
        plan=json.loads(frozen_plan.read_text()) if frozen_plan.exists() else None
    frozen_plan=root/'comparison_plan.json'
    if plan is not None and (not frozen_plan.exists() or json.loads(frozen_plan.read_text())!=plan):
        raise ValueError('comparison hypotheses differ from the pre-Test frozen plan')
    inference_metadata={}
    if run_inference and plan is not None:
        from .wm_tail_transfer_statistics import infer_comparisons
        result=infer_comparisons(build_comparisons(cfg,plan,forecast,policies,baselines),cfg)
        for key,name in (('inferenceDF','inference.csv'),('hacDF','hac_inference.csv'),('primaryDF','primary_inference.csv')):
            result[key].to_csv(destination/name,index=False)
        inference_metadata=result['metadata'];_json(destination/'inference_metadata.json',inference_metadata)
    n_unavailable=sum(r.get('available') is False for r in forecast)
    n_ensemble=sum(r['task'].get('role')=='ensemble' for r in forecast)
    n_statistical=sum(r['task'].get('role')=='statistical' for r in forecast)
    n_forecast_fits=sum(r.get('available') is not False and r['task'].get('role') not in ('ensemble','statistical') for r in forecast)
    audit=dict(registration_sha256=registration_sha,n_forecast_fits=n_forecast_fits,
        n_forecast_registered_jobs=len(forecast),n_forecast_unavailable=n_unavailable,
        n_ensemble_outputs=n_ensemble,n_statistical_outputs=n_statistical,n_forecast_output_coordinates=len(metrics),
        n_stochastic_rollout_outputs=len(rollouts),
        n_primary_task_outputs=len(taskmetrics),n_policy_paths=len(policies),n_baseline_paths=len(baselines),
        n_resets=0,calendar_continuity='every raw Test bar, including no-decision intervals',
        forecast_horizon=cfg['targets']['horizons'],primary_horizon=cfg['targets']['primary_horizon'],
        all_Test_previously_inspected=True,independently_unused_holdout=False,Test_used_for_selection=False,
        producer_chronology='producer Train10months + Val2months precede policy Train12months',
        representation_PPO='BC market encoders frozen in PPO; new forecast producers independently trained',
        derivative_environment=cfg['perpetual']['scope'],n_comparisons=inference_metadata.get('n_comparisons',0))
    if inference_metadata:
        resolution=1/(int(cfg['statistics']['bootstrap_replicates'])+1)
        audit['bootstrap_minimum_raw_p']=resolution
        audit['global_Holm_first_step_minimum_attainable_p']=min(1.,resolution*inference_metadata['n_comparisons'])
        audit['multiplicity_scope']='task-specific forecast families and baseline-specific policy families fixed before Test; all-hypothesis Holm sensitivity also reported'
    _json(destination/'audit.json',audit)
    intro='<p>既知の歴史期間を使った研究結果。Testではモデル・閾値・校正・売買設定を選び直していない。全口座を期間中連続して評価し、入力欠損区間も価格損益・負債を引き継ぐ。Oracleは診断用として分離。</p>'
    body=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>UniDream tail-transfer研究結果</title>',
        '<style>body{font-family:system-ui;max-width:1500px;margin:32px auto;padding:16px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:5px}th{background:#eee}pre{white-space:pre-wrap}</style>',
        '<h1>急騰・急落情報 → WM → BC → 過去市場遷移を使うPPO 比較</h1>',intro,
        '<p>学習fit数と、タスク・期間・閾値・校正による出力項目数は別々に記録。</p>',
        '<h2>実施状況</h2>',statuses.to_html(index=False),'<h2>全seedの売買結果</h2>',summary.to_html(index=False),
        '<h2>比較対象</h2>',baselineframe.to_html(index=False),
        '<h2>機械可読データ</h2><p><a href="policy_metrics.csv">全seed売買成績</a> / <a href="forecast_task_metrics.csv">全タスク予測</a> / <a href="primary_inference.csv">主検定</a> / <a href="inference.csv">Block Bootstrap感度分析</a> / <a href="hac_inference.csv">HAC</a></p>',
        '<h2>評価条件</h2><pre>'+html.escape(json.dumps(_safe(audit),ensure_ascii=False,indent=2))+'</pre></html>']
    (destination/'index.html').write_text('\n'.join(body))
    artifacts=[p for p in destination.iterdir() if p.is_file() and p.name!='completion.json']
    completion=dict(complete=True,**audit,
        artifact_sha256={p.name:benchmark.sha(p) for p in artifacts})
    _json(destination/'completion.json',completion)
    return completion
