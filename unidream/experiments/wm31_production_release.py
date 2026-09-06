"""DRAFT: one fresh production-only WM31 -> BC -> selected learned AC refit.

Install this file under the research experiments package and freeze it together
with its YAML before --mode fit. No validation/test/economic runner is called.
The preflight mode checks bound raw inputs and support; it fits no scaler/model.
The fit mode fits the T-only scaler and fixed700/5/300 endpoints exactly once.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.experiments.alpha_dd_search import file_digest, load_bars
from unidream.experiments.wm_rl_inputs import (
    build_market_inputs, build_inference_inputs, fit_normalizer, apply_normalizer,
    mask_digest, sequence_masks,
)
from unidream.experiments.wm_rl_release import source_snapshot

SCHEMA = 'wm-rl-market31-production-refit-v1'
START = pd.Timestamp('2024-09-01T00:00:00Z')
CUTOFF = pd.Timestamp('2026-09-01T00:00:00Z')
BAR = pd.Timedelta(minutes=15)
FOLD = 25
TRAINING_SECTIONS = ('ac', 'actions', 'bc', 'costs', 'data', 'eval',
                     'normalization', 'oracle', 'reward', 'targets', 'world_model')


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def require_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or file_digest(Path(path)) != expected:
        raise ValueError('bound artifact changed or hash absent: ' + str(path))
    return expected


def validate_selection(selection, *, selected_arm, source_config_sha256):
    """Bind the external screen decision; never derive qualification here."""
    if (not isinstance(selection, dict)
            or selection.get('schema') != 'wm-rl-paper-selection-v1'
            or selection.get('qualified') is not True
            or selection.get('selected_arm') != selected_arm
            or selection.get('source_config_sha256') != source_config_sha256):
        raise ValueError('completed qualified selection from the bound screen is required')


def validate_config(config_path, *, fitting):
    cfg = yaml.safe_load(Path(config_path).read_text())
    r = cfg['release']
    if (r['schema'] != SCHEMA or r['group'] != 'perp_delay0'
            or r['evaluation_split'] != 'none' or r['test_scoring'] is not False
            or r['economic_scoring'] is not False or r['screen_only'] is not False
            or r['seed'] != 7 or r['wm_selection'] != 'fixed_endpoint700_train_only'
            or r['fit_market_models'] != 1 or r['learned_rl_arms_per_model'] != 1):
        raise ValueError('fixed production-only contract changed')
    if r['folds'] != [{'fold': FOLD, 'train_start': START.isoformat(), 'train_end': CUTOFF.isoformat()}]:
        raise ValueError('production training interval is fixed; no validation/test window exists')
    if cfg['run']['folds'] != [FOLD] or pd.Timestamp(cfg['run']['start']) != START or pd.Timestamp(cfg['run']['end']) != CUTOFF:
        raise ValueError('run metadata disagrees with fixed production calendar')
    if cfg['run']['clean_checkpoint_dir'] is not False:
        raise ValueError('production cannot delete/reuse a checkpoint directory')
    require_hash(r['source_screen_config'], r['source_screen_config_sha256'])
    base = yaml.safe_load(Path(r['source_screen_config']).read_text())
    for key in TRAINING_SECTIONS:
        if cfg[key] != base[key]:
            raise ValueError('registered screen training section changed: ' + key)
    if r['execution'] != base['release']['execution'] or r['device'] != base['release']['device']:
        raise ValueError('execution/device differs from registered screen configuration')
    if cfg['run']['deterministic_algorithms'] != base['run']['deterministic_algorithms']:
        raise ValueError('registered determinism setting changed')
    require_hash(r['production_input_manifest'], r['production_input_manifest_sha256'])
    manifest = json.loads(Path(r['production_input_manifest']).read_text())
    if (manifest['status'] != 'production_input_snapshot_complete'
            or pd.Timestamp(manifest['training_start']) != START
            or pd.Timestamp(manifest['cutoff_exclusive']) != CUTOFF):
        raise ValueError('wrong production acquisition snapshot/calendar')
    paths = set()
    for item in manifest['artifacts'].values():
        p = Path(item['path']).resolve()
        if p in paths:
            raise ValueError('duplicate resolved production artifact path')
        paths.add(p); require_hash(p, item['sha256'])
    for market in ('spot', 'um'):
        for suffix, key in [('_path', market + '_15m.parquet'),
                            ('_availability_path', market + '_15m_availability.parquet')]:
            item = manifest['artifacts'][key]
            if Path(r[market + suffix]).resolve() != Path(item['path']).resolve():
                raise ValueError('configured market path differs from acquired manifest')
            require_hash(r[market + suffix], r[market + suffix.replace('_path', '_sha256')])
    # Arm selection is external, completed and immutable. This runner only binds
    # bytes and qualification metadata; it never reads/compares validation scores.
    names = r['ac_arm_names']
    selected = r['selected_arm']
    if names != [selected]:
        raise ValueError('exactly the previously selected arm is required')
    from unidream.experiments import wm_rl_training as training
    if selected not in training.AC_ARMS:
        if fitting:
            raise ValueError('production arm has not yet been bound after the screen')
    if fitting:
        resolver = getattr(training, 'selected_ac_arms', None)
        if not callable(resolver):
            raise ValueError('required optional-arm trainer guard has not been installed')
        expected = {selected: training.AC_ARMS[selected]}
        if resolver(cfg) != expected:
            raise ValueError('trainer did not resolve exactly the preselected AC arm')
        require_hash(r['selection_manifest_path'], r['selection_manifest_sha256'])
        selection = json.loads(Path(r['selection_manifest_path']).read_text())
        validate_selection(selection, selected_arm=selected,
            source_config_sha256=r['source_screen_config_sha256'])
    return cfg, manifest


def build_inputs(cfg, acquisition):
    """Rebuild only the acquired pre-cutoff input snapshot; no scaler/model fit."""
    r = cfg['release']
    spot = load_bars(Path(r['spot_path']), cutoff=CUTOFF.isoformat())
    um = pd.read_parquet(r['um_path'])
    um.index = pd.to_datetime(um.index, utc=True).as_unit('ns')
    if spot.index[-1] != CUTOFF-BAR or np.any(spot.index >= CUTOFF) or np.any(um.index >= CUTOFF):
        raise ValueError('production raw files must end before the fixed cutoff')
    sav = pd.read_parquet(r['spot_availability_path'])
    uav = pd.read_parquet(r['um_availability_path'])
    if (not sav.index.equals(spot.index) or sav.spot_bar_observed.dtype != bool
            or sav.spot_bar_observed.isna().any() or not uav.index.equals(um.index)
            or uav.um_bar_observed.dtype != bool or uav.um_bar_observed.isna().any()):
        raise ValueError('physical sidecars do not exactly match the acquired grids')
    inputs = build_market_inputs(spot.drop(columns='bar_available'), um, cutoff=CUTOFF,
        spot_observed=sav.spot_bar_observed.to_numpy(bool))
    frame = inputs['groups']['perp_delay0']
    ix = frame.index
    train = np.asarray((ix >= START) & (ix < CUTOFF))
    fm, tm = inputs['full_feature_eligible'], inputs['raw_target_validity']
    encoder = sequence_masks(ix, feature_eligible=fm, seq_len=64)['endpoint_eligible']
    wm = sequence_masks(ix, feature_eligible=fm, target_eligible=tm,
                        row_mask=train, seq_len=128)
    teacher = train & fm & tm & encoder
    ac = sequence_masks(ix, feature_eligible=teacher, seq_len=64)['endpoint_eligible']
    difference = np.diff(np.r_[False, teacher, False].astype(np.int8))
    segments = list(zip(np.flatnonzero(difference == 1), np.flatnonzero(difference == -1)))
    feasibility_path = acquisition['artifacts'][acquisition['authoritative_feasibility']]['path']
    bound = json.loads(Path(feasibility_path).read_text())
    if inputs['source_contract'] != bound['source_contract']:
        raise ValueError('feature/source/full common-mask contract differs from confirmed feasibility')
    if list(frame.columns) != bound['feature_groups']['perp_delay0']['columns'] or frame.shape[1] != 31:
        raise ValueError('frozen ordered31 feature contract changed')
    masks = {'spot_observed': inputs['spot_observed'],
        'um_observed': uav.um_bar_observed.reindex(ix, fill_value=False).to_numpy(bool),
        'common_feature': fm, 'raw_market_target_validity': tm,
        'fixed64_encoder_with_prior_context': encoder,
        'wm128_endpoints_full_T': wm['endpoint_eligible'],
        'T_teacher_support_if_model_outputs_finite': teacher,
        'AC_origins_if_model_outputs_finite': ac}
    support = {}
    for name, mask in masks.items():
        selected = train & mask
        measured = {'count': int(selected.sum()), 'mask_sha256': mask_digest(ix, selected)}
        expected = bound['supports'][name]
        if measured != {'count': expected['observed_or_eligible_rows'], 'mask_sha256': expected['mask_sha256']}:
            raise ValueError('production support changed: ' + name)
        support[name] = measured
    bc_chunks = sum((b-a)//4 for a,b in segments)
    if (int(train.sum()) != acquisition['training_nominal_rows']
            or len(wm['valid_starts']) != acquisition['wm128_sequences']
            or bc_chunks != acquisition['bc_chunk4_origins_if_outputs_finite']
            or int(ac.sum()) != acquisition['ac_origins_if_outputs_finite']):
        raise ValueError('confirmed production sequence counts changed')
    # Verify the deployable context at the cutoff using inputs alone, not WM
    # outputs. Do not append this placeholder/context to the training matrix.
    latest_spot = spot.drop(columns='bar_available').iloc[-8704:]
    latest = build_inference_inputs(latest_spot, um.reindex(latest_spot.index), origin=CUTOFF)
    if (not latest['inference_available'] or int(latest['context_feature_eligible'].sum()) != 64
            or latest['source_contract'] != bound['latest8704']['input_contract']):
        raise ValueError('confirmed fixed64 cutoff input contract changed')
    keep = np.asarray(ix >= START - 63*BAR)
    kept_ix = ix[keep]
    prepared = {'raw_features': frame.loc[keep], 'returns': inputs['returns'].loc[keep],
        'feature_eligible': fm[keep], 'target_eligible': tm[keep],
        'train_mask': train[keep], 'source_contract': inputs['source_contract'],
        'full_frame': frame, 'full_train_mask': train, 'full_feature_eligible': fm,
        'latest_raw_context': latest['context_groups']['perp_delay0'],
        'wm_valid_starts': wm['valid_starts'], 'support': support}
    if kept_ix[-1] >= CUTOFF or int(prepared['train_mask'].sum()) != 70080:
        raise ValueError('trimmed training support is not the fixed two-year period')
    prepared['input_provenance'] = {'schema': SCHEMA, 'fold': r['folds'][0],
        'rows': len(kept_ix), 'feature_columns': list(frame.columns),
        'source_contract': inputs['source_contract'], 'support_on_full_source_grid': support,
        'train_mask_sha256': mask_digest(kept_ix, prepared['train_mask']),
        'feature_eligible_sha256': mask_digest(kept_ix, prepared['feature_eligible']),
        'raw_return_valid_sha256': mask_digest(kept_ix, prepared['target_eligible']),
        'validation_mask_sha256': None, 'validation_rows_used': 0, 'test_rows_used': 0,
        'raw_cutoff_exclusive': CUTOFF.isoformat(), 'context_prefix_rows': int((~prepared['train_mask']).sum()),
        'wm_sequence_length': 128, 'inference_context_length': 64,
        'bc_chunks_if_outputs_finite': int(bc_chunks),
        'production_input_manifest_sha256': r['production_input_manifest_sha256'],
        'archive_receipt_proven': False, 'economic_scoring': False,
        'pre_T_features_are_context_only': True}
    return prepared


def fit(config_path):
    """Exactly one fresh fit. Any exception preserves evidence and is not retried."""
    cfg, acquisition = validate_config(config_path, fitting=True)
    r = cfg['release']
    source = source_snapshot(config_path)
    # A /tmp draft is not an approved executable. Require the installed runner
    # itself to be tracked, in addition to the full existing source freeze.
    subprocess.run(['git', 'ls-files', '--error-unmatch', str(Path(__file__).resolve())],
                   check=True, stdout=subprocess.DEVNULL)
    out, models = Path(r['output_dir']), Path(cfg['logging']['checkpoint_dir']) / 'fold_25'
    if out.exists() or models.exists():
        raise ValueError('fresh output/checkpoint directories required; no resume/retry')
    prepared = build_inputs(cfg, acquisition)
    normalizer = fit_normalizer(prepared['full_frame'], train_mask=prepared['full_train_mask'],
        feature_eligible=prepared['full_feature_eligible'])
    features = apply_normalizer(prepared['raw_features'], normalizer,
        feature_eligible=prepared['feature_eligible'])
    prepared['input_provenance']['normalizer'] = normalizer
    prepared['input_provenance']['normalized_features_sha256'] = array_digest(features.to_numpy())
    prepared['input_provenance']['raw_returns_sha256'] = array_digest(prepared['returns'].to_numpy())
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': SCHEMA, 'config': cfg, 'config_sha256': file_digest(Path(config_path)),
        'source': source, 'started_at': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'running',
        'attempts': 1, 'fit_only': True, 'economic_scoring': False, 'test_scoring': False,
        'validation_rows_used': 0, 'test_rows_used': 0, 'folds_completed': [],
        'selected_arm': r['selected_arm'], 'selection_manifest_sha256': r['selection_manifest_sha256'],
        'source_screen_config_sha256': r['source_screen_config_sha256'],
        'external_screen_qualified': True,
        'production_input_manifest_sha256': r['production_input_manifest_sha256'],
        'production_weights_have_no_historical_performance_claim': True,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'numpy': np.__version__, 'pandas': pd.__version__}}
    write_json(out / 'fold_25_inputs.json', prepared['input_provenance'], exclusive=True)
    np.savez_compressed(out / 'fold_25_fit_inputs.npz', timestamp_ns=features.index.as_unit('ns').asi8,
        normalized_features=features.to_numpy(), raw_returns=prepared['returns'].to_numpy(),
        feature_eligible=prepared['feature_eligible'], target_eligible=prepared['target_eligible'],
        train_mask=prepared['train_mask'], feature_columns=np.asarray(features.columns, dtype='U'),
        latest_raw_context=prepared['latest_raw_context'].to_numpy(),
        latest_context_timestamp_ns=prepared['latest_raw_context'].index.as_unit('ns').asi8)
    manifest['input_artifacts'] = {name: file_digest(out/name) for name in ('fold_25_inputs.json','fold_25_fit_inputs.npz')}
    write_json(out/'run_manifest.json', manifest, exclusive=True)
    began = time.monotonic()
    try:
        if source_snapshot(config_path) != source:
            raise ValueError('source changed during production preparation')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(bool(cfg['run']['deterministic_algorithms']))
        from unidream.experiments.wm_rl_training import train_wm_bc_ac
        trained = train_wm_bc_ac(features=features, returns=prepared['returns'],
            feature_eligible=prepared['feature_eligible'], target_eligible=prepared['target_eligible'],
            train_mask=prepared['train_mask'], wm_val_mask=None, cfg=cfg,
            output_dir=models, seed=7, device=r['device'])
        if source_snapshot(config_path) != source:
            raise ValueError('source changed during the production fit')
        validate_config(config_path, fitting=True)  # raw/source/config/selection bindings after training
        arm = r['selected_arm']
        p = trained['training_provenance']
        if (set(trained['ac_actors']) != {arm} or set(p['arms']) != {arm}
                or p['wm_executed_steps'] != 700 or p['wm_val_sequences'] is not None
                or p['wm_train_sequences'] != acquisition['wm128_sequences']
                or p['bc_chunks'] != acquisition['bc_chunk4_origins_if_outputs_finite']
                or p['ac_origin_count'] != acquisition['ac_origins_if_outputs_finite']):
            raise ValueError('actual production training inventory/support changed')
        logs = json.loads((models/'training_logs.json').read_text())
        if len(logs['bc']) != 5 or set(logs['ac']) != {arm} or len(logs['ac'][arm]) != 300:
            raise ValueError('production did not complete exactly5 BC epochs and300 selected AC updates')
        for relative, expected in trained['artifacts'].items():
            require_hash(models/relative, expected)
        for name, expected in manifest['input_artifacts'].items():
            require_hash(out/name, expected)
        final_source = source_snapshot(config_path)
        if final_source != source:
            raise ValueError('source changed before completing production provenance')
        manifest.update(status='completed', finished_at=pd.Timestamp.now(tz='UTC').isoformat(),
            source_after=final_source,
            folds_completed=[{'fold': FOLD, 'seconds': time.monotonic()-began}],
            trained_artifacts={**trained['artifacts'], 'artifacts.json': file_digest(models/'artifacts.json')},
            training_dir=str(models), wm_steps=700, bc_epochs=5, ac_steps=300)
        write_json(out/'run_manifest.json', manifest)
        print(json.dumps({'status': 'completed', 'output_dir': str(out), 'training_dir': str(models),
            'selected_arm': arm, 'economic_scoring': False, 'test_scoring': False,
            'run_manifest_sha256': file_digest(out/'run_manifest.json')}, indent=2), flush=True)
    except Exception as error:
        manifest.update(status='failed_no_retry', finished_at=pd.Timestamp.now(tz='UTC').isoformat(),
                        failure_type=type(error).__name__)
        write_json(out/'run_manifest.json', manifest)
        raise


def preflight(config_path):
    cfg, acquisition = validate_config(config_path, fitting=False)
    before = source_snapshot(config_path)
    prepared = build_inputs(cfg, acquisition)
    if source_snapshot(config_path) != before:
        raise ValueError('source changed during input-only preflight')
    result = {**prepared['input_provenance'], 'config_sha256': file_digest(Path(config_path)),
        'source': before, 'mode': 'input_only', 'normalizer_fitted': False, 'model_fitted': False,
        'model_inference_performed': False, 'actual_model_gradient_coverage_unmeasured': True}
    path = Path(cfg['release']['preflight_output_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result, exclusive=True)
    print(json.dumps({'status':'input_only_pass','path':str(path),'sha256':file_digest(path)},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('preflight','fit'), default='preflight')
    args = parser.parse_args()
    (preflight if args.mode == 'preflight' else fit)(args.config)


if __name__ == '__main__':
    main()
