"""Fixed full 4-input x 4-family x 7-target x 2-formulation research grid.

Fresh, matched-seed neural formulations; independent classical estimators
reuse frozen point-regression and binary-event fits where exactly identical.
Test is report-only and opens only after every fit and Val choice is frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from . import wm_seven_task_benchmark as parent
from .wm_technical_ablation import ROOT, _input, _json, _sha
from .wm_seven_task_evaluation import TASKS
from .wm_full_grid_models import DirectFullGrid, WMFullGrid
from .wm_full_grid_classical import fit_grid_classical, predict_grid_classical
from .wm_full_grid_evaluation import fit_label_contract, make_task_labels

DEFAULT_CONFIG = ROOT / 'configs/experiments/wm_full_grid_20261111.yaml'


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['groups'] != ['A', 'B', 'C', 'D'] or cfg['seeds'] != [7]
            or cfg['folds'] != list(range(15)) or cfg['horizons'] != [16]
            or cfg['formulations'] != ['regression', 'classification']
            or cfg['test_selection_allowed'] is not False or cfg['calibration'] != 'identity'):
        raise ValueError('registered full 224-cell, seed7, 4h, report-only scope required')
    if cfg['validation_partition'] != 'calendar_months_earlystop_calibration_selection':
        raise ValueError('disjoint purged Val calendar months required')
    if (cfg['labels']['signed_train_rms_multiplier'] != .5
            or cfg['labels']['signed_tasks'] != ['return', 'advantage', 'recovery']
            or cfg['labels']['tertile_tasks'] != ['variance', 'downside']
            or cfg['labels']['tertile_quantiles'] != [1 / 3, 2 / 3]
            or cfg['labels']['excess'] != 'zero_positive_train_median'
            or cfg['labels']['crash'] != 'existing_binary_0_1'
            or cfg['labels']['minimum_class_count'] != 200
            or cfg['labels']['minimum_class_fraction'] != .005):
        raise ValueError('label definitions must match the fixed Train-only builder')
    cfg['config_path'] = str(Path(path).resolve())
    for name in ('output_dir', 'parent_config'):
        cfg[name] = str((ROOT / cfg[name]).resolve())
    return cfg


def source_hashes():
    names = ('wm_full_grid_benchmark.py', 'wm_full_grid_models.py', 'wm_full_grid_classical.py',
             'wm_full_grid_evaluation.py', 'wm_full_grid_report.py', 'wm_full_grid_statistics.py',
             'wm_return_formulation_classical.py')
    return {str((Path(__file__).parent / name).relative_to(ROOT)): _sha(Path(__file__).parent / name)
            for name in names}


def verify(cfg, full=False):
    out = Path(cfg['output_dir'])
    reg = json.loads((out / 'preregistration.json').read_text())
    if reg['config_sha256'] != _sha(cfg['config_path']) or reg['source_sha256'] != source_hashes():
        raise ValueError('new-study registered config/source changed')
    pcfg = parent.load_config(cfg['parent_config'])
    parent.verify(pcfg, full=full)
    if _sha(Path(pcfg['output_dir']) / 'preregistration.json') != reg['parent_registration_sha256']:
        raise ValueError('immutable parent registration changed')
    if full:
        for name, expected in reg['prepared_sha256'].items():
            if _sha(out / 'prepared' / name) != expected: raise ValueError('prepared labels changed')
        for relative, expected in reg['reused_parent_sha256'].items():
            if _sha(Path(pcfg['output_dir']) / relative) != expected: raise ValueError('reused classical artifact changed')
    return reg, pcfg


def prepare(cfg):
    out = Path(cfg['output_dir']); prepared = out / 'prepared'
    prepared.mkdir(parents=True, exist_ok=True)
    if (out / 'preregistration.json').exists(): raise ValueError('registration already frozen')
    pcfg = parent.load_config(cfg['parent_config']); preg = parent.verify(pcfg, full=True)
    if cfg['model'] != pcfg['model']:
        raise ValueError('matching inherited model/context budget required')
    arrays = parent.arrays_for(pcfg)
    folds = []
    for fold in cfg['folds']:
        data = parent.fold_for(pcfg, fold)
        old = next(row for row in preg['folds'] if row['fold'] == fold)
        contract = fit_label_contract({t: arrays[t] for t in TASKS}, data['train'])
        labels = {}
        valid = np.asarray(arrays['mask'][:, 0], bool)
        for task in TASKS:
            value = np.zeros_like(arrays[task], dtype=np.int8)
            value[valid] = make_task_labels(arrays[task][valid], task, contract).astype(np.int8)
            labels[task] = value
        np.savez(prepared / f'fold_{fold:02d}_labels.npz', **labels)
        folds.append({**old, 'label_contract': contract})
    # All registered branches train from scratch; queued order spreads groups.
    jobs = [{'fold': f, 'group': g, 'seed': 7, 'model': model, 'formulation': mode,
             'job': f'fold_{f:02d}_{g}_{model}_{mode}_s7'}
            for f in cfg['folds'] for g in ('D', 'A', 'B', 'C')
            for mode in cfg['formulations'] for model in ('direct', 'wm')]
    classics = [{'fold': f, 'group': g, 'seed': 7, 'job': f'fold_{f:02d}_{g}'}
                for f in cfg['folds'] for g in ('D', 'A', 'B', 'C')]
    reused = {}
    for task in classics:
        directory = Path(pcfg['output_dir']) / 'classical' / task['job']
        fit = json.loads((directory / 'fit.json').read_text())
        for name, expected in (('models.pkl', fit['model_sha256']),
                               ('validation_predictions.parquet', fit['validation_predictions_sha256'])):
            if _sha(directory / name) != expected: raise ValueError('frozen parent classical fit changed')
        for name in ('fit.json', 'models.pkl', 'validation_predictions.parquet'):
            p = directory / name; reused[str(p.relative_to(Path(pcfg['output_dir'])))] = _sha(p)
    _json(out / 'preregistration.json', {'experiment_id': cfg['experiment_id'], 'registered_utc': now(),
        'config': cfg, 'config_sha256': _sha(cfg['config_path']), 'source_sha256': source_hashes(),
        'parent_registration_sha256': _sha(Path(pcfg['output_dir']) / 'preregistration.json'),
        'prepared_sha256': {p.name: _sha(p) for p in sorted(prepared.glob('*.npz'))},
        'reused_parent_sha256': reused, 'folds': folds, 'neural_tasks': jobs, 'classical_tasks': classics,
        'grid': {'inputs': 4, 'families': 4, 'targets': 7, 'formulations': 2, 'cells': 224, 'neural_seeds': [7]},
        'loss_weights': parent.LOSS_WEIGHTS,
        'selection': 'first Val month: shared weighted seven-objective NN checkpoint; per-head classical hyperparameters. Third Val month: fixed native target metric or BA choices before global Test barrier. Middle Val month reserved; identity calibration.',
        'regression': 'six original continuous losses and binary crash squared probability/Brier loss; native metric per task; downstream scalar-to-class projection auxiliary',
        'classification': 'six Train-only ternary class maps with CE; original binary crash BCE; native output probability; primary BA decision p/Train_prior, raw argmax auxiliary',
        'reuse': 'only independent classical six continuous regressors and binary-crash classifiers; exact selected original fits and predictions. No neural warm start or frozen-encoder probe.',
        'test_access': 'all240 NN fits and60classical bundles plus Val choices frozen before ANY new Test inference. Historical period already explored; no pristine holdout claim.',
        'limitations': ['All seven objectives change jointly; not an isolated per-task loss intervention.',
            'Classification has12extra final output logits (2316parameters); same causal body and shared hidden head.',
            'Single seed estimates cannot establish initialization stability.',
            'Native regression metrics and classification BA are different units, no global224cellranking.',
            'Binary crash comparison is Brier-vs-BCE estimation of the same event probability.',
            'Synthetic advantage/recovery are fixed surrogates, not executable PnL or time-to-recovery.',
            'Inherited conservative15min publication assumptions and common16h-purged cohort retained.']})
    status(cfg, 'registered')


def training_mask(arrays, positions, fold):
    return (arrays['mask'][positions]
        & (arrays['label_end_ns'][positions] < pd.Timestamp(fold['train_end']).value)
        & (arrays['index_ns'][positions, None] >= pd.Timestamp(fold['train_start']).value))


def supervised_objective(actual, predictions, formulation, contract, scales):
    if formulation == 'regression':
        answer = parent.objective_numpy(actual, predictions, scales)
        brier = float(np.mean((predictions['crash'] - actual['crash']) ** 2))
        answer['score'] += parent.LOSS_WEIGHTS['crash'] * (brier - answer['parts']['crash'])
        answer['parts']['crash'] = brier
        return answer
    if formulation != 'classification': raise ValueError('unknown formulation')
    parts = {}
    for task in TASKS:
        labels = make_task_labels(actual[task], task, contract).reshape(-1)
        k = contract[task]['n_classes']
        p = np.asarray(predictions[task]).reshape(-1, k)
        if (len(p) != len(labels) or not np.isfinite(p).all() or (p < 0).any()
                or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6)):
            raise ValueError('invalid selection probabilities or cohort')
        parts[task] = float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    return {'score': sum(parent.LOSS_WEIGHTS[t] * parts[t] for t in TASKS), 'parts': parts}


def directory(cfg, kind, task):
    return Path(cfg['output_dir']) / kind / task['job']


def model_for(task, cfg, scales):
    cls = WMFullGrid if task['model'] == 'wm' else DirectFullGrid
    return cls(63, cfg['model'], scales, task['formulation'])


def fold_info(reg, fold):
    return next(row for row in reg['folds'] if row['fold'] == fold)


def predict(model, arrays, ids, data, task, cfg):
    model.eval(); parts = {t: [] for t in TASKS}
    with torch.no_grad():
        for start in range(0, len(ids), cfg['training']['evaluation_batch_size']):
            batch = ids[start:start + cfg['training']['evaluation_batch_size']]
            x, _ = _input(arrays, batch, data, task['group'], cfg['model']['context'])
            values = model.predict_last(torch.from_numpy(x).to(cfg['device']))
            for target in TASKS: parts[target].append(values[target].cpu().numpy())
    result = {t: np.concatenate(value) for t, value in parts.items()}
    if any(not np.isfinite(v).all() for v in result.values()): raise ValueError('nonfinite forecast')
    return result


def prediction_frame(arrays, ids, predictions, task, split, contract):
    frame = pd.DataFrame({'decision_id': ids, 'timestamp': pd.to_datetime(arrays['index_ns'][ids], utc=True),
        'fold': task['fold'], 'group': task['group'], 'seed': 7, 'model': task['model'],
        'formulation': task['formulation'], 'split': split})
    for target in TASKS:
        frame['actual_' + target] = arrays[target][ids, 0]
        frame['label_' + target] = make_task_labels(arrays[target][ids, 0], target, contract)
        frame['scalar_' + target] = np.asarray(predictions[target]).reshape(len(ids)) if task['formulation'] == 'regression' else np.nan
        for c in range(3):
            frame[f'p_{target}_{c}'] = (np.asarray(predictions[target]).reshape(len(ids), -1)[:, c]
                if task['formulation'] == 'classification' and c < contract[target]['n_classes'] else np.nan)
    return frame


def parquet(frame, path):
    temporary = path.with_name(path.name + '.tmp')
    frame.to_parquet(temporary, index=False); temporary.replace(path)


def worker_setup(threads):
    os.nice(5); torch.set_num_threads(threads); torch.set_num_interop_threads(1)


def train_neural(config_path, task):
    cfg = load_config(config_path); reg, pcfg = verify(cfg)
    dest = directory(cfg, 'neural', task); dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'training.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        worker_setup(cfg['training']['cpu_threads']); torch.use_deterministic_algorithms(True)
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task['fold'])
        fold = fold_info(reg, task['fold'])
        with np.load(Path(cfg['output_dir']) / 'prepared' / f"fold_{task['fold']:02d}_labels.npz") as z:
            label_arrays = {t: z[t] for t in TASKS}
        model = model_for(task, cfg, fold['scales']).to(cfg['device'])
        training = cfg['training']
        optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
        rng = np.random.default_rng(7); best = anchor = np.inf
        best_step = last_significant = 0; state = None; history = []
        counts = 0; coverage = {t: 0 for t in TASKS}; reason = 'max_steps'; started = time.monotonic()
        for step in range(1, training['max_steps'] + 1):
            model.train(); ids = rng.choice(data['train'], training['batch_size'], replace=True)
            x, base = _input(arrays, ids, data, task['group'], cfg['model']['context'])
            positions = ids[:, None] - cfg['model']['context'] + np.arange(1, cfg['model']['context'] + 1)[None, :]
            valid = training_mask(arrays, positions, fold); counts += int(valid.sum())
            targets = {}
            for target in TASKS:
                values = (label_arrays[target][positions].astype(np.int64) if task['formulation'] == 'classification'
                          else arrays[target][positions].astype(np.float32))
                targets[target] = torch.from_numpy(values).to(cfg['device'])
            next_return = np.log(arrays['raw'][positions, 3] / arrays['raw'][positions - 1, 3]).astype(np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg['device']), torch.from_numpy(base).to(cfg['device']),
                targets, torch.from_numpy(valid).to(cfg['device']), torch.from_numpy(next_return).to(cfg['device']))
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite training loss')
            loss.backward()
            grad = model.supervised_output_layer.weight.grad
            if grad is None: raise RuntimeError('missing supervised gradients')
            covered = grad.detach().abs().sum(1).cpu().numpy() > 0
            for target, sl in model.task_output_slices.items(): coverage[target] += int(covered[sl].all())
            torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if step % training['validation_every'] == 0:
                panel = predict(model, arrays, data['selection_panel'], data, task, cfg)
                score = supervised_objective({t: arrays[t][data['selection_panel']] for t in TASKS}, panel,
                    task['formulation'], fold['label_contract'], fold['scales'])
                history.append({'step': step, 'selection': score, 'batch_total_loss': float(loss.detach().cpu()),
                    'batch_loss_parts': {k: float(v.cpu()) for k, v in parts.items()}, 'elapsed_seconds': time.monotonic() - started})
                print(json.dumps(history[-1]), flush=True)
                if score['score'] < best:
                    best, best_step = score['score'], step
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if not np.isfinite(anchor) or score['score'] < anchor - training['relative_improvement'] * abs(anchor):
                    anchor, last_significant = score['score'], step
                _json(dest / 'progress.json', {**task, 'step': step, 'best_step': best_step, 'best_score': best, 'elapsed_seconds': time.monotonic() - started})
                if step >= training['min_steps'] and step - last_significant >= training['patience_steps']:
                    reason = 'validation_patience'; break
        if state is None or counts == 0 or not all(coverage.values()): raise RuntimeError('missing checkpoint/target/gradient coverage')
        model.load_state_dict(state)
        torch.save({'state_dict': state, 'scales': fold['scales'], 'task': task,
            'registration_sha256': _sha(Path(cfg['output_dir']) / 'preregistration.json')}, dest / 'research_model.pt')
        frames = [prediction_frame(arrays, data[split], predict(model, arrays, data[split], data, task, cfg),
            task, split, fold['label_contract']) for split in ('earlystop', 'calibration', 'selection')]
        parquet(pd.concat(frames, ignore_index=True), dest / 'validation_predictions.parquet')
        _json(dest / 'fit.json', {**task, 'registration_sha256': _sha(Path(cfg['output_dir']) / 'preregistration.json'),
            'elapsed_seconds': time.monotonic() - started, 'steps': step, 'best_step': best_step, 'best_score': best,
            'stop_reason': reason, 'max_step_censored': reason == 'max_steps', 'history': history,
            'target_counts': counts, 'supervised_gradient_steps': coverage, 'parameters': sum(p.numel() for p in model.parameters()),
            'checkpoint_sha256': _sha(dest / 'research_model.pt'), 'validation_predictions_sha256': _sha(dest / 'validation_predictions.parquet')})
    return task


def read_parent_classical(pcfg, task, split, study_cfg=None):
    if split == 'test':
        if study_cfg is None: raise ValueError('new-study Test barrier context required before reference Test read')
        test_access(study_cfg, 'classical', task)
    dest = Path(pcfg['output_dir']) / 'classical' / task['job']
    fit = json.loads((dest / 'fit.json').read_text())
    filename = 'test_predictions.parquet' if split == 'test' else 'validation_predictions.parquet'
    expected = fit['validation_predictions_sha256']
    if split == 'test':
        evaluation = json.loads((dest / 'evaluation.json').read_text())
        if evaluation['barrier_sha256'] != _sha(Path(pcfg['output_dir']) / 'test_barrier.json'):
            raise ValueError('parent Test artifact provenance changed')
        expected = evaluation['prediction_sha256']
    if _sha(dest / filename) != expected: raise ValueError('parent predictions changed')
    frame = pd.read_parquet(dest / filename)
    return frame.loc[frame['split'] == split]


def classical_frames(bundle, arrays, ids, data, task, split, cfg, pcfg, fold):
    fresh = predict_grid_classical(bundle, parent.tabular_x(pcfg, ids, task['group']))
    old = read_parent_classical(pcfg, task, split, study_cfg=cfg)
    frames = []
    for family in ('linear', 'lightgbm'):
        reference = old.loc[old['model'] == family].sort_values('decision_id').reset_index(drop=True)
        if not np.array_equal(reference['decision_id'], ids): raise ValueError('classical reuse cohort differs')
        for target in TASKS:
            if not np.array_equal(reference['actual_' + target], arrays[target][ids, 0]):
                raise ValueError('classical reused outcomes differ')
        for mode in cfg['formulations']:
            values = {}
            for target in TASKS:
                if (mode == 'regression' and target != 'crash'):
                    values[target] = reference['pred_' + target].to_numpy()
                elif mode == 'classification' and target == 'crash':
                    p = reference['pred_crash'].to_numpy(); values[target] = np.column_stack((1 - p, p))
                else: values[target] = fresh[mode][family][target]
            frames.append(prediction_frame(arrays, ids, values, {**task, 'model': family, 'formulation': mode}, split, fold['label_contract']))
    return frames


def train_classical(config_path, task):
    cfg = load_config(config_path); reg, pcfg = verify(cfg)
    dest = directory(cfg, 'classical', task); dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'training.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(cfg['classical']['cpu_threads']):
        os.nice(5); arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task['fold'])
        fold = fold_info(reg, task['fold']); started = time.monotonic()
        labels = {t: make_task_labels(arrays[t][data['train'], 0], t, fold['label_contract']) for t in TASKS}
        val_labels = {t: make_task_labels(arrays[t][data['earlystop'], 0], t, fold['label_contract']) for t in TASKS}
        bundle = fit_grid_classical(parent.tabular_x(pcfg, data['train'], task['group']),
            {t: arrays[t][data['train'], 0] for t in TASKS}, labels,
            parent.tabular_x(pcfg, data['earlystop'], task['group']),
            {t: arrays[t][data['earlystop'], 0] for t in TASKS}, val_labels, cfg['classical'])
        with (dest / 'models.pkl').open('wb') as stream: pickle.dump(bundle, stream, protocol=5)
        frames = []
        for split in ('earlystop', 'calibration', 'selection'):
            frames.extend(classical_frames(bundle, arrays, data[split], data, task, split, cfg, pcfg, fold))
        parquet(pd.concat(frames, ignore_index=True), dest / 'validation_predictions.parquet')
        _json(dest / 'fit.json', {**task, 'registration_sha256': _sha(Path(cfg['output_dir']) / 'preregistration.json'),
            'elapsed_seconds': time.monotonic() - started, 'metadata': bundle['metadata'],
            'model_sha256': _sha(dest / 'models.pkl'), 'validation_predictions_sha256': _sha(dest / 'validation_predictions.parquet'),
            'reused_parent_fit_sha256': _sha(Path(pcfg['output_dir']) / 'classical' / task['job'] / 'fit.json')})
    return task


def status(cfg, stage=None):
    out = Path(cfg['output_dir']); old = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
    reg = json.loads((out / 'preregistration.json').read_text())
    value = {'utc': now(), 'stage': stage or old.get('stage', 'training'), 'cells': 224,
        'neural': {'complete': sum((directory(cfg, 'neural', t) / 'fit.json').exists() for t in reg['neural_tasks']), 'total': len(reg['neural_tasks'])},
        'classical': {'complete': sum((directory(cfg, 'classical', t) / 'fit.json').exists() for t in reg['classical_tasks']), 'total': len(reg['classical_tasks'])},
        'test_opened': (out / 'test_barrier.json').exists()}
    _json(out / 'status.json', value)
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>224項目の統一比較</title><style>body{{font-family:system-ui;max-width:900px;margin:40px auto;line-height:1.8}}</style><h1>入力4×モデル4×予測7×分類／回帰：224項目</h1><p>段階: {value['stage']}</p><p>ニューラル {value['neural']['complete']}/{value['neural']['total']} 学習、古典モデルセット {value['classical']['complete']}/{value['classical']['total']}。同じ15fold・seed7。</p><p>全学習とVal選択後にTestを評価。初回の116項目とは別の、両方式を揃えた追加比較。</p><p><a href="../wm_224_grid_20261111/index.html">先行結果224枠の実施状況（116実施済み）</a> / <a href="selection/index.html">Val暫定結果</a> / <a href="final/index.html">Test結果224項目</a> / <a href="campaign.log">実行ログ</a></p><p>回帰の主指標と分類のBAは異なる単位。急落は2クラス、他の6対象は3クラス。分類境界はTrainだけで固定。</p><p>更新 UTC: {value['utc']}</p></html>'''
    temp = out / 'progress.html.tmp'; temp.write_text(page, encoding='utf-8'); temp.replace(out / 'progress.html')
    print(json.dumps(value), flush=True); return value


def check_fit(cfg, kind, task):
    dest = directory(cfg, kind, task); fit = json.loads((dest / 'fit.json').read_text())
    if fit['registration_sha256'] != _sha(Path(cfg['output_dir']) / 'preregistration.json'): raise ValueError('fit registration mismatch')
    model_file, key = ('research_model.pt', 'checkpoint_sha256') if kind == 'neural' else ('models.pkl', 'model_sha256')
    if _sha(dest / model_file) != fit[key] or _sha(dest / 'validation_predictions.parquet') != fit['validation_predictions_sha256']:
        raise ValueError('selected model/Val predictions changed')
    return fit


def frozen_fit_manifest(cfg, reg):
    manifest = {}
    for kind, key in (('neural', 'neural_tasks'), ('classical', 'classical_tasks')):
        for task in reg[key]:
            path = directory(cfg, kind, task) / 'fit.json'
            if not path.exists(): raise ValueError('global Test closed: missing fit')
            check_fit(cfg, kind, task); manifest[str(path.relative_to(Path(cfg['output_dir'])))] = _sha(path)
    return manifest


def open_test_barrier(cfg):
    reg, _ = verify(cfg, full=True); out = Path(cfg['output_dir'])
    manifest = frozen_fit_manifest(cfg, reg)
    choices_path = out / 'validation_choices.json'
    if not choices_path.exists(): raise ValueError('Val choices not frozen')
    choices = json.loads(choices_path.read_text())
    if choices.get('test_used') is not False or choices.get('registration_sha256') != _sha(out / 'preregistration.json'):
        raise ValueError('invalid validation choices provenance')
    value = {'opened_utc': now(), 'registration_sha256': _sha(out / 'preregistration.json'),
        'frozen_fit_sha256': manifest, 'validation_choices_sha256': _sha(choices_path), 'test_selection_allowed': False}
    path = out / 'test_barrier.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous[key] != value[key] for key in value if key != 'opened_utc'):
            raise ValueError('frozen Test barrier artifacts changed')
    else: _json(path, value)
    return value


def test_access(cfg, kind, task):
    reg, _ = verify(cfg); out = Path(cfg['output_dir'])
    path = out / 'test_barrier.json'
    if not path.exists(): raise ValueError('global Test barrier closed')
    barrier = json.loads(path.read_text())
    if (barrier['registration_sha256'] != _sha(out / 'preregistration.json')
            or barrier['validation_choices_sha256'] != _sha(out / 'validation_choices.json')):
        raise ValueError('Test barrier/choices changed')
    required = {f"{k}/{t['job']}/fit.json" for k, key in (('neural', 'neural_tasks'), ('classical', 'classical_tasks')) for t in reg[key]}
    if set(barrier['frozen_fit_sha256']) != required: raise ValueError('incomplete global fit manifest')
    for relative, expected in barrier['frozen_fit_sha256'].items():
        frozen_path = out / relative
        if not frozen_path.exists() or _sha(frozen_path) != expected:
            raise ValueError('global fit missing or changed after Test barrier')
    fit_path = directory(cfg, kind, task) / 'fit.json'
    if barrier['frozen_fit_sha256'].get(str(fit_path.relative_to(out))) != _sha(fit_path): raise ValueError('fit changed after Test barrier')
    check_fit(cfg, kind, task)
    return barrier


def evaluate_task(config_path, kind, task):
    cfg = load_config(config_path); test_access(cfg, kind, task); reg, pcfg = verify(cfg)
    worker_setup(cfg['training']['cpu_threads'])
    arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task['fold']); fold = fold_info(reg, task['fold'])
    dest = directory(cfg, kind, task)
    if kind == 'neural':
        state = torch.load(dest / 'research_model.pt', map_location='cpu', weights_only=False)
        model = model_for(task, cfg, state['scales']).to(cfg['device']); model.load_state_dict(state['state_dict'])
        frames = [prediction_frame(arrays, data['test'], predict(model, arrays, data['test'], data, task, cfg), task, 'test', fold['label_contract'])]
    else:
        with (dest / 'models.pkl').open('rb') as stream: bundle = pickle.load(stream)
        frames = classical_frames(bundle, arrays, data['test'], data, task, 'test', cfg, pcfg, fold)
    parquet(pd.concat(frames, ignore_index=True), dest / 'test_predictions.parquet')
    _json(dest / 'evaluation.json', {'barrier_sha256': _sha(Path(cfg['output_dir']) / 'test_barrier.json'),
        'prediction_sha256': _sha(dest / 'test_predictions.parquet'), 'utc': now()})
    return task


def run_fits(cfg):
    reg, _ = verify(cfg); context = mp.get_context('spawn')
    with ProcessPoolExecutor(cfg['training']['workers'], mp_context=context, max_tasks_per_child=1) as gpu, ProcessPoolExecutor(cfg['classical']['workers'], mp_context=context, max_tasks_per_child=1) as cpu:
        futures = {}
        for kind, key, pool, fn in (('neural', 'neural_tasks', gpu, train_neural), ('classical', 'classical_tasks', cpu, train_classical)):
            for task in reg[key]:
                if (directory(cfg, kind, task) / 'fit.json').exists(): check_fit(cfg, kind, task)
                else: futures[pool.submit(fn, cfg['config_path'], task)] = (kind, task)
        for future in as_completed(futures):
            kind, task = futures[future]
            try: future.result()
            except Exception as exc:
                _json(Path(cfg['output_dir']) / 'failure.json', {'utc': now(), 'kind': kind, 'task': task, 'error': repr(exc)})
                status(cfg, 'failed')
                for pending in futures: pending.cancel()
                raise
            print(json.dumps({'completed': kind, **task}), flush=True); status(cfg, 'training')


def freeze_validation_choices(cfg):
    from .wm_full_grid_report import write_full_grid_report
    reg, _ = verify(cfg); frozen_fit_manifest(cfg, reg)
    result = write_full_grid_report(cfg['output_dir'], cfg, include_test=False)
    path = Path(cfg['output_dir']) / 'selection' / 'results_224.csv'
    table = pd.read_csv(path)
    # Choices intentionally fixed by the report's explicit metric/direction.
    choices = {'frozen_utc': now(), 'registration_sha256': _sha(Path(cfg['output_dir']) / 'preregistration.json'),
        'test_used': False, 'calibration': 'identity', 'selection_report_sha256': _sha(path),
        'rule': 'per-target/formulation native primary metric in selection CSV; retrospective descriptive Test only',
        'all_selection_rows': table.to_dict(orient='records'), 'report': result}
    # pandas converts missing native metrics to NaN; strict JSON uses null.
    choices['all_selection_rows'] = json.loads(table.to_json(orient='records'))
    _json(Path(cfg['output_dir']) / 'validation_choices.json', choices)


def campaign(cfg):
    out = Path(cfg['output_dir'])
    try:
        verify(cfg, full=True); status(cfg, 'training'); run_fits(cfg)
        status(cfg, 'freezing_validation'); freeze_validation_choices(cfg)
        open_test_barrier(cfg); status(cfg, 'test_inference')
        reg, _ = verify(cfg); context = mp.get_context('spawn')
        for kind, key, workers in (('neural', 'neural_tasks', cfg['training']['workers']), ('classical', 'classical_tasks', cfg['classical']['workers'])):
            with ProcessPoolExecutor(workers, mp_context=context, max_tasks_per_child=1) as pool:
                futures = [pool.submit(evaluate_task, cfg['config_path'], kind, task) for task in reg[key]]
                for future in as_completed(futures): future.result()
        status(cfg, 'final_report')
        from .wm_full_grid_report import write_full_grid_report
        from .wm_full_grid_statistics import write_grid_statistics
        report = write_full_grid_report(cfg['output_dir'], cfg, include_test=True)
        status(cfg, 'block_bootstrap_statistics'); statistics = write_grid_statistics(cfg['output_dir'], cfg)
        _json(out / 'completion.json', {'complete': True, 'status': 'complete', 'utc': now(),
            'registration_sha256': _sha(out / 'preregistration.json'), 'report': report, 'statistics': statistics})
        status(cfg, 'complete')
    except Exception as exc:
        _json(out / 'failure.json', {'utc': now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        status(cfg, 'failed'); raise


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('command', choices=('prepare', 'run', 'status'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.command == 'prepare': prepare(cfg)
    elif args.command == 'run': campaign(cfg)
    else: status(cfg)


if __name__ == '__main__': main()
