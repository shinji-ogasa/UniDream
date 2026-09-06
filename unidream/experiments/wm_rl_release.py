"""Bounded, validation-only WM -> BC -> learned RL comparison.

Dates, data, costs and training settings are fixed in the YAML. This runner is
an explicit experiment, not a mainline warm-start entrypoint. It never reads
test outcomes and never promotes a checkpoint based on a test score.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .alpha_dd_search import file_digest, load_bars
from .wm_rl_inputs import build_market_inputs, fit_normalizer, apply_normalizer, mask_digest, sequence_masks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_config(cfg):
    r = cfg['release']
    if r['schema'] != 'wm-rl-market31-screen-v1' or r['group'] != 'perp_delay0':
        raise ValueError('unregistered experiment or feature group')
    if r['test_scoring'] is not False or r['evaluation_split'] != 'validation':
        raise ValueError('only validation economics are permitted in this screen')
    if r['wm_selection'] != 'fixed_endpoint700_train_only':
        raise ValueError('scored validation must not select WM weights')
    if cfg['world_model']['reward_mode'] != 'market_log_return':
        raise ValueError('a newly learned market WM is required')
    if (cfg['world_model']['train_sequence_length'], cfg['data']['seq_len'],
            cfg['world_model']['max_seq_len']) != (128, 64, 128):
        raise ValueError('registered training128 / inference64 contract changed')
    if cfg['ac']['controller_state_dim'] != 4 or cfg['ac']['advantage_dim'] != 42:
        raise ValueError('persistent controller and actual WM auxiliary inputs required')
    if r['execution'] != {'one_way_cost': .00055, 'borrow_annual': .1, 'max_step': .08,
            'deadband': .01, 'position_min': .5, 'position_max': 1.12,
            'bars_per_year': 35040, 'fill_delay_bars': 1, 'decision_interval_minutes': 15}:
        raise ValueError('execution differs from fixed account/cadence contract')
    if [f['fold'] for f in r['folds']] != cfg['run']['folds']:
        raise ValueError('fold registration differs')
    for f in r['folds']:
        dates = [pd.Timestamp(f[k]) for k in ('train_start', 'train_end', 'validation_end')]
        if any(d.tzinfo is None or d.minute % 15 for d in dates) or not dates[0] < dates[1] < dates[2]:
            raise ValueError('invalid fixed fold calendar')
    for market in ('spot', 'um'):
        if file_digest(Path(r[market + '_path'])) != r[market + '_sha256']:
            raise ValueError('changed bound raw source: ' + market)
    if file_digest(Path(r['spot_availability_path'])) != r['spot_availability_sha256']:
        raise ValueError('changed physical observation sidecar')
    if file_digest(Path(r['preflight_path'])) != r['preflight_sha256']:
        raise ValueError('changed registered input preflight')
    return r


def verify_registered_inputs(cfg, fold, frame, inputs, normalizer):
    registered = json.loads(Path(cfg['release']['preflight_path']).read_text())
    matches = [item for item in registered['folds'] if item['fold'] == fold['fold']]
    if len(matches) != 1:
        raise ValueError('fold missing or duplicated in frozen input preflight')
    item = matches[0]
    for local, frozen in [('train_start', 'train_start'), ('train_end', 'train_end_exclusive'),
                          ('validation_end', 'validation_end_exclusive')]:
        if pd.Timestamp(fold[local]) != pd.Timestamp(item['calendar'][frozen]):
            raise ValueError('calendar differs from input registration')
    if normalizer != item['normalizer'] or inputs['source_contract'] != item['source_contract']:
        raise ValueError('rebuilt features/masks/scaler differ from frozen input preflight')
    index = frame.index
    full_encoder = sequence_masks(index, feature_eligible=inputs['full_feature_eligible'], seq_len=64)['endpoint_eligible']
    for name, start, end in [('train', fold['train_start'], fold['train_end']),
                              ('validation', fold['train_end'], fold['validation_end'])]:
        segment = np.asarray((index >= pd.Timestamp(start)) & (index < pd.Timestamp(end)))
        wm = sequence_masks(index, feature_eligible=inputs['full_feature_eligible'],
            target_eligible=inputs['raw_target_validity'], row_mask=segment, seq_len=128)['endpoint_eligible']
        for key, mask in [('common_feature_support', segment & inputs['full_feature_eligible']),
                          ('wm128_endpoints', wm),
                          ('encoder64_endpoint_in_split_with_prior_context', full_encoder & segment)]:
            expected = item['segments'][name][key]
            if int(mask.sum()) != expected['count'] or mask_digest(index, mask) != expected['mask_sha256']:
                raise ValueError('rebuilt segment support differs: ' + name + '/' + key)


def prepare_fold(cfg, fold):
    """Causal full-grid inputs; normalizer and all fitting masks are T-only."""
    r = cfg['release']
    start, boundary, cutoff = (pd.Timestamp(fold[k]) for k in
                              ('train_start', 'train_end', 'validation_end'))
    spot = load_bars(Path(r['spot_path']), cutoff=cutoff.isoformat())
    um = pd.read_parquet(r['um_path'])
    um.index = pd.to_datetime(um.index, utc=True)
    um = um.loc[um.index < cutoff]
    sidecar = pd.read_parquet(r['spot_availability_path']).reindex(spot.index)
    if sidecar.spot_bar_observed.isna().any() or sidecar.spot_bar_observed.dtype != bool:
        raise ValueError('physical observation sidecar is incomplete or malformed')
    inputs = build_market_inputs(spot.drop(columns=['bar_available']), um, cutoff=cutoff,
                                 spot_observed=sidecar.spot_bar_observed.to_numpy(bool))
    frame = inputs['groups'][r['group']]
    train = np.asarray((frame.index >= start) & (frame.index < boundary))
    norm = fit_normalizer(frame, train_mask=train,
                          feature_eligible=inputs['full_feature_eligible'])
    verify_registered_inputs(cfg, fold, frame, inputs, norm)
    features = apply_normalizer(frame, norm,
                               feature_eligible=inputs['full_feature_eligible'])
    # Feature recipes retain their entire raw history. Only the already-built
    # matrix is trimmed, retaining 63 pre-T rows for the first fixed64 origin.
    keep = features.index >= start - pd.Timedelta(minutes=15 * 63)
    ix = features.index[keep]
    train = np.asarray((ix >= start) & (ix < boundary))
    val = np.asarray((ix >= boundary) & (ix < cutoff))
    result = dict(features=features.loc[keep], returns=inputs['returns'].loc[keep],
                  feature_eligible=inputs['full_feature_eligible'][keep],
                  target_eligible=inputs['raw_target_validity'][keep],
                  train_mask=train, wm_val_mask=val, bars=spot.loc[keep],
                  normalizer=norm, source_contract=inputs['source_contract'])
    result['raw_features'] = frame.loc[keep]
    result['input_provenance'] = {
        'fold': fold, 'rows': len(ix), 'feature_columns': list(features.columns),
        'normalizer': norm, 'source_contract': inputs['source_contract'],
        'train_mask_sha256': mask_digest(ix, train),
        'validation_mask_sha256': mask_digest(ix, val),
        'feature_eligible_sha256': mask_digest(ix, result['feature_eligible']),
        'raw_return_valid_sha256': mask_digest(ix, result['target_eligible']),
        'test_rows_used': 0,
    }
    return result


def path_metrics(bars, equity, exposures, intents, account_state):
    closes = bars.close.to_numpy(float)
    observed = np.isfinite(closes)
    if not observed[-1] or not np.isfinite(equity[-1]) or equity[-1] <= 0:
        raise ValueError('invalid fixed evaluation endpoint')
    values = np.r_[1., equity[observed]]
    benchmark = np.r_[1., closes[observed] / float(bars.open.iloc[0])]
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('nonfinite or insolvent observed account')
    dd = float(np.max(1 - values / np.maximum.accumulate(values)))
    bh_dd = float(np.max(1 - benchmark / np.maximum.accumulate(benchmark)))
    return {'alpha_ex': float(values[-1] - benchmark[-1]),
            'maxdd_delta': dd - bh_dd, 'total_return': float(values[-1] - 1),
            'bh_total_return': float(benchmark[-1] - 1), 'maxdd': dd, 'bh_maxdd': bh_dd,
            'mean_exposure': float(np.nanmean(exposures)),
            'intent_coverage': float(np.isfinite(intents).mean()),
            'close_coverage': float(observed.mean()), 'rows': len(bars),
            'account_final': account_state}


def summarize_matrix(rows):
    """Fixed screen ranking; test values are never accepted by this function."""
    if not rows or any(row.get('split') != 'validation' for row in rows):
        raise ValueError('only nonempty validation rows can select a model')
    keys = [(row['arm'], row['fold'], row['cost_case']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate arm/fold/cost row')
    folds = {row['fold'] for row in rows}
    summary = {}
    for arm in sorted({row['arm'] for row in rows}):
        arm_rows = [row for row in rows if row['arm'] == arm]
        means = {}
        for cost in ('base', 'stress'):
            subset = [row for row in arm_rows if row['cost_case'] == cost]
            if {row['fold'] for row in subset} != folds:
                raise ValueError('arms must use identical folds and costs')
            a = np.array([row['metrics']['alpha_ex'] for row in subset])
            d = np.array([row['metrics']['maxdd_delta'] for row in subset])
            if not len(a) or not np.isfinite(a).all() or not np.isfinite(d).all():
                raise ValueError('incomplete or nonfinite screen matrix')
            means[cost] = {'folds': len(a), 'alpha_ex_mean': float(a.mean()),
                           'maxdd_delta_mean': float(d.mean()),
                           'joint_positive_folds': int(((a > 0) & (d < 0)).sum()),
                           'worst_alpha_ex': float(a.min()),
                           'worst_maxdd_delta': float(d.max())}
        means['minimum_mean_signs'] = all(
            means[c]['alpha_ex_mean'] > 0 and means[c]['maxdd_delta_mean'] < 0
            for c in ('base', 'stress'))
        means['rl_changes_observed_decisions'] = any(
            row.get('bc_intent_difference', {}).get('changed_rows', 0) > 0 for row in arm_rows)
        means['causal_regime_means'] = {}
        for regime in ('bull', 'bear', 'sideways'):
            subset = [row for row in arm_rows if row['cost_case'] == 'base'
                      and row.get('regime', {}).get('trend') == regime]
            means['causal_regime_means'][regime] = {
                'folds': len(subset),
                'alpha_ex_mean': float(np.mean([row['metrics']['alpha_ex'] for row in subset])) if subset else None,
                'maxdd_delta_mean': float(np.mean([row['metrics']['maxdd_delta'] for row in subset])) if subset else None}
        summary[arm] = means
    learned = [key for key in summary if key.startswith('ac_')]
    eligible = [key for key in learned if summary[key]['minimum_mean_signs']
                and summary[key]['rl_changes_observed_decisions']]
    def rank(key):
        v = summary[key]['stress']
        return (v['alpha_ex_mean'] - v['maxdd_delta_mean'], key)
    return {'arms': summary, 'eligible_learned_rl': eligible,
            'screen_selected': max(eligible, key=rank) if eligible else None,
            'best_diagnostic_rl': max(learned, key=rank) if learned else None,
            'selection_rule': 'both base/stress mean signs; maximize stressed mean alpha minus mean DDdelta',
            'evidence_status': 'development_validation_screen_only',
            'high_probability_trend_independence_established': False,
            'production_promotion_permitted': False, 'test_scoring': False}


def evaluate_fold(cfg, fold, prepared, trained, out):
    """Actor decisions see t-1 account state and causal WM inputs at origin t."""
    from .wm_rl_execution import CashUnitAccount
    from .wm_rl_policy import IncrementalActorPolicy
    selected = np.flatnonzero(prepared['wm_val_mask'])
    bars = prepared['bars'].iloc[selected]
    if not len(bars) or not np.isfinite(bars.open.iloc[0]) or not np.isfinite(bars.close.iloc[-1]):
        raise ValueError('fixed validation endpoint unavailable; do not shift boundary')
    encoded, auxiliary = trained['encoded'], trained['auxiliary']['standardized']
    first = prepared['raw_features'].iloc[selected[0]]
    normalized_momentum = first.momentum_90 / max(first.vol_7 * np.sqrt(90 / 365), 1e-6)
    regime = {'at': bars.index[0].isoformat(), 'basis': 'causal features at exact validation start',
              'normalized_momentum_90': float(normalized_momentum) if np.isfinite(normalized_momentum) else None,
              'trend': ('bull' if normalized_momentum > .5 else 'bear' if normalized_momentum < -.5 else 'sideways')
              if np.isfinite(normalized_momentum) else 'unavailable'}
    actors = {'bc_only': trained['bc_actor'], **trained['ac_actors'], 'actor_removed_bh': None}
    def model_digest(actor):
        h = hashlib.sha256()
        for name, value in actor.state_dict().items():
            a = value.detach().cpu().numpy()
            h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()
    bc_state = trained['bc_actor'].state_dict()
    rows, base_intents = [], {}
    for arm, actor in actors.items():
        if actor is not None:
            actor.to('cpu').eval()
            actor_sha = model_digest(actor)
            delta_to_bc = max(float((value.detach().cpu() - bc_state[name].detach().cpu()).abs().max())
                              for name, value in actor.state_dict().items())
            if arm.startswith('ac_') and delta_to_bc == 0:
                raise ValueError('RL checkpoint has no learned change from BC')
        else:
            actor_sha, delta_to_bc = None, None
        for case, multiplier in (('base', 1), ('stress', 2)):
            contract = cfg['release']['execution']
            account = CashUnitAccount(bars.index[0], float(bars.open.iloc[0]),
                one_way_cost=contract['one_way_cost'] * multiplier,
                borrow_annual=contract['borrow_annual'] * multiplier)
            policy = None if actor is None else IncrementalActorPolicy(actor, device='cpu')
            n = len(bars)
            equity, exposure, intent, fills = (np.full(n, np.nan) for _ in range(4))
            for j, origin in enumerate(selected):
                timestamp = bars.index[j]
                feedback = account.decision_feedback(timestamp)
                if not feedback['valuation_available']:
                    raise ValueError('account cannot be valued, no synthetic policy feedback')
                available = bool(encoded['available'][origin] and feedback['actor_account_available'])
                if policy is None:
                    target = 1.0  # Pre-existing B&H inventory: no dynamic model exposure.
                else:
                    decision = policy.step(timestamp, encoded['z'][origin], encoded['h'][origin],
                        available=available, advantage=auxiliary[origin],
                        actual_exposure=feedback['actual_exposure'], executed_delta=feedback['executed_delta'])
                    target = decision['target_intent']
                if target is not None:
                    intent[j] = target
                op, cl = float(bars.open.iloc[j]), float(bars.close.iloc[j])
                event = account.advance_bar(timestamp, op, cl,
                    open_observed=bool(np.isfinite(op)), close_observed=bool(np.isfinite(cl)),
                    intent_for_next_open=target)
                if event['insolvent']:
                    raise ValueError('candidate insolvent on fixed validation path: ' + arm)
                equity[j] = np.nan if event['equity'] is None else event['equity']
                exposure[j] = np.nan if event['exposure'] is None else event['exposure']
                fills[j] = event['fill']['executed_delta']
            metrics = path_metrics(bars, equity, exposure, intent, account.state.to_dict())
            row = {'fold': fold['fold'], 'arm': arm, 'cost_case': case,
                   'split': 'validation', 'regime': regime, 'metrics': metrics,
                   'actor_state_sha256': actor_sha, 'max_actor_weight_change_vs_bc': delta_to_bc,
                   'trained_artifacts': trained['artifacts']}
            if actor is not None and model_digest(actor) != actor_sha:
                raise ValueError('inference mutated learned weights')
            if case == 'base':
                base_intents[arm] = intent.copy()
                if arm.startswith('ac_'):
                    common = np.isfinite(intent) & np.isfinite(base_intents['bc_only'])
                    row['bc_intent_difference'] = {
                        'common_rows': int(common.sum()),
                        'mean_abs': float(np.abs(intent[common] - base_intents['bc_only'][common]).mean()),
                        'changed_rows': int((np.abs(intent[common] - base_intents['bc_only'][common]) > 1e-7).sum())}
            trace = out / f"fold_{fold['fold']}_{arm}_{case}.npz"
            np.savez_compressed(trace, timestamp_ns=bars.index.as_unit('ns').asi8,
                equity=equity, exposure=exposure, intent=intent, fill_delta=fills,
                open=bars.open.to_numpy(float), close=bars.close.to_numpy(float))
            row['trace_sha256'] = file_digest(trace)
            rows.append(row)
            print(f"[physical validation] fold={fold['fold']} {arm} {case} "
                  f"AlphaEX={100*metrics['alpha_ex']:+.5f}pt "
                  f"MaxDDdelta={100*metrics['maxdd_delta']:+.5f}pt", flush=True)
    return rows


def source_snapshot(config_path=None):
    # Hash all Python/config sources, not just the small adapter entrypoint.
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'unidream', 'configs', 'tests'], check=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
                                         'unidream', 'configs', 'tests'], text=True).splitlines()
    if untracked:
        raise ValueError('source freeze forbids untracked implementation/config/tests: ' + ', '.join(untracked))
    if config_path is not None:
        subprocess.run(['git', 'ls-files', '--error-unmatch', str(config_path)], check=True, stdout=subprocess.DEVNULL)
    names = subprocess.check_output(['git', 'ls-files', 'unidream', 'configs'], text=True).splitlines()
    return {'git_head': head, 'files': {p: file_digest(Path(p)) for p in names if Path(p).is_file()}}


def run(config_path):
    from .wm_rl_training import train_wm_bc_ac
    cfg = yaml.safe_load(Path(config_path).read_text())
    r = validate_config(cfg)
    snapshot = source_snapshot(config_path)
    out = Path(r['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': r['schema'], 'config': cfg, 'config_sha256': file_digest(Path(config_path)),
                'source': snapshot, 'started_at': pd.Timestamp.now(tz='UTC').isoformat(),
                'test_scoring': False, 'status': 'running', 'folds_completed': []}
    write_json(out / 'run_manifest.json', manifest)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(bool(cfg['run']['deterministic_algorithms']))
    rows = []
    for fold in r['folds']:
        began = time.monotonic()
        prepared = prepare_fold(cfg, fold)
        fold_id = fold['fold']
        write_json(out / f'fold_{fold_id}_inputs.json', prepared['input_provenance'])
        model_dir = Path(cfg['logging']['checkpoint_dir']) / f'fold_{fold_id}'
        trained = train_wm_bc_ac(
            **{k: prepared[k] for k in ('features', 'returns', 'feature_eligible',
               'target_eligible', 'train_mask')}, wm_val_mask=None,
            cfg=cfg, output_dir=model_dir, seed=r['seed'], device=r['device'])
        if source_snapshot(config_path) != snapshot:
            raise ValueError('source changed during training; stop before economic selection')
        fold_rows = evaluate_fold(cfg, fold, prepared, trained, out)
        rows.extend(fold_rows)
        write_json(out / 'validation_metrics.json', rows)
        manifest['folds_completed'].append({'fold': fold_id, 'seconds': time.monotonic() - began})
        write_json(out / 'run_manifest.json', manifest)
        del trained, prepared
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    write_json(out / 'screen_decision.json', summarize_matrix(rows))
    if source_snapshot(config_path) != snapshot:
        raise ValueError('source changed during registered comparison')
    manifest.update(status='completed', finished_at=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(out / 'run_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
