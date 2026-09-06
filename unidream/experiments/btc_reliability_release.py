"""One fixed production-calendar fit/export of the selected causal ML recipe.

Preflight reconstructs inputs only. A registered run fits three frozen base
models and one scale-only reliability coefficient, with no evaluation outcome,
policy path, score, parameter search or retry.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from .alpha_dd_search import digest, file_digest, load_bars, validate_data_artifact
from .oracle_additional_window_replay import SOURCES as ANCESTRAL_SOURCES
from .oracle_confirmation_contract import calendar, segment_masks
from .oracle_derivative_ablation import mask_digest, validate_um
from .oracle_derivative_delay_features import make_delayed_perp_groups
from .oracle_derivative_features import make_derivative_groups
from .oracle_frontier import outcome_frame
from .oracle_frontier_features import make_feature_groups
from .oracle_frozen_forecasts import fit_frozen_forecasts
from .oracle_mean_reliability import fit_reliability, apply_reliability
from .oracle_risk_calibration import trailing_variances
from .oracle_short_mean_fit import _index_digest, _matrix_digest

MODEL_NAMES = ('technical_mean', 'perp_delay0_mean', 'technical_variance')
GROUPS = ('technical', 'perp_delay0')
CANDIDATE = 'perp_delay0_reliability_utility_risk1'
BUNDLE_ID = 'btc-perp-reliability-20260906'
EXECUTION = {'one_way_cost': .00055, 'borrow_annual': .1, 'max_step': .08, 'deadband': .01}
BUNDLE_EXECUTION = {**EXECUTION, 'risk_aversion': 1, 'utility_cost_multiplier': 2,
    'horizon_bars': 24, 'decision_hours_utc': [0, 6, 12, 18], 'fill_delay_bars': 1,
    'intent_bounds': [.5, 1.12], 'missing_forecast': 'hold', 'initial_cash': 0.,
    'initial_equity': 1., 'initial_units': '1/initial_open', 'bars_per_year': 35040}
SOURCES = tuple(dict.fromkeys(ANCESTRAL_SOURCES + tuple('unidream/experiments/' + n for n in (
    'oracle_mean_reliability.py', 'oracle_short_mean_fit.py', 'btc_reliability_release.py'))))
FIXED = {'schema': 'btc-reliability-release-v1', 'candidate_id': CANDIDATE,
    'bundle_id': BUNDLE_ID, 'symbol': 'BTCUSDT', 'interval': '15m', 'calendar_fold': 25,
    'data_cutoff': '2026-07-16T13:45:00Z', 'output_dir': 'codex_outputs/btc_reliability_release_v1',
    'bundle_dir': 'codex_outputs/btc_reliability_release_v1/bundle',
    'fit_months': 18, 'scale_months': 3, 'interval_months': 3, 'nominal_evaluation_months': 3,
    'horizon_bars': 24, 'label_maturity_minutes': 375, 'groups': list(GROUPS),
    'group_dimensions': [29, 31], 'extra_common_mask_delays': [0, 1, 4],
    'minimum_fit_rows': 512, 'minimum_scale_rows': 64, 'minimum_interval_rows': 64,
    'execution': EXECUTION, 'utility_risk_aversion': 1, 'utility_cost_multiplier': 2,
    'missing_forecast_rule': 'hold', 'decision_cadence_hours': 6, 'next_open_delay_bars': 1,
    'feature_shift_bars': 1, 'target_min': .5, 'target_max': 1.12,
    'mean_model': 'StandardScaler_Ridge100', 'variance_model': 'frozen_technical_HGB100',
    'reliability_weight_source': 'scale_only_clipped_crossmoment_over_secondmoment',
    'new_base_model_fits': 3, 'new_reliability_weights': 1, 'evaluation_rows_permitted': 0,
    'scores_permitted': False, 'backtests_permitted': False, 'selection_permitted': False,
    'model_retry_permitted': False, 'live_receipt_provenance_established': False,
    'numpy_version': '2.2.6', 'pandas_version': '2.3.3', 'sklearn_version': '1.8.0',
    'threadpool_limit': 2, 'scalar_ridge_prediction_atol': 1e-12}
EXTRA = {'source_bindings', 'spot_path', 'spot_sha256', 'um_path', 'um_sha256',
    'data_manifest_path', 'data_manifest_sha256', 'selection_path', 'selection_sha256', 'preflight_sha256'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _bound(path, sha):
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'invalid SHA256')
    _require(file_digest(Path(path)) == sha, 'changed bound artifact: ' + str(path))


def _bindings(parts):
    result, aliases = {}, {}
    for part in parts:
        _require(isinstance(part, dict), 'binding dictionary required')
        for path, sha in part.items():
            _require(isinstance(path, str) and bool(path), 'binding path required')
            resolved = str(Path(path).resolve())
            _require(resolved not in aliases or aliases[resolved] == path, 'aliased binding')
            _require(path not in result or result[path] == sha, 'conflicting binding')
            result[path] = sha
            aliases[resolved] = path
    return result


def validate_config(cfg):
    _require(isinstance(cfg, dict) and set(cfg) == set(FIXED) | EXTRA, 'unregistered release fields')
    _require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in FIXED.items()), 'unregistered release contract')
    _require(isinstance(cfg['source_bindings'], dict) and set(cfg['source_bindings']) == set(SOURCES), 'incomplete frozen dependency set')
    _require(pd.Timestamp(cfg['data_cutoff']) == calendar(25)['evaluation_start'], 'production cutoff must equal fixed E start')
    _require(Path(cfg['bundle_dir']) == Path(cfg['output_dir']) / 'bundle', 'invalid bundle location')


def common_feature_inputs(bars, um):
    """The entire original comparison dependency set determines availability."""
    original = make_feature_groups(bars)
    derivative = make_derivative_groups(bars, um)
    groups = make_delayed_perp_groups(bars, um, delays=(0, 1, 4))
    trailing = trailing_variances(bars, 24)
    components = {'trailing_variances': trailing, 'original_flow': original['flow'],
        **{'derivative_' + name: frame for name, frame in derivative.items()},
        **{'delayed_' + name: frame for name, frame in groups.items()}}
    common = np.ones(len(bars), dtype=bool)
    for frame in components.values():
        _require(frame.index.equals(bars.index), 'unaligned feature dependency')
        common &= np.isfinite(frame.to_numpy()).all(axis=1)
    _require([len(groups[g].columns) for g in GROUPS] == [29, 31], 'changed feature dimensions')
    return {g: groups[g] for g in GROUPS}, common, components


def release_masks(index, common, valid_labels):
    _require(len(index) > 0 and index[-1] < calendar(25)['evaluation_start'], 'post-cutoff market row')
    masks = segment_masks(index, common, valid_labels, 25)
    _require(not any(masks[k].any() for k in ('scheduled', 'inference', 'score')), 'evaluation support forbidden')
    for name, minimum in (('fit', 512), ('scale', 64), ('interval', 64)):
        _require(int(masks[name].sum()) >= minimum, 'insufficient ' + name + ' support')
    _require(masks['predict'].any(), 'empty prediction fixture support')
    return masks


def prepare(config_path):
    """Read/hash data and compute input supports; no estimators or new forecasts."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    validate_config(cfg)
    _require((np.__version__, pd.__version__, sklearn.__version__) ==
        (cfg['numpy_version'], cfg['pandas_version'], cfg['sklearn_version']), 'runtime version changed')
    direct = _bindings([cfg['source_bindings'], {cfg[k + '_path']: cfg[k + '_sha256']
        for k in ('spot', 'um', 'data_manifest', 'selection')}])
    for path, sha in direct.items():
        _bound(path, sha)
    selection = _read(cfg['selection_path'])
    _require(selection['selected_candidate_id'] == CANDIDATE and selection['bundle_id'] == BUNDLE_ID
        and selection['production_refit']['calendar_fold'] == 25
        and selection['production_refit']['production_cutoff'] == cfg['data_cutoff']
        and selection['production_refit']['new_evaluation_scores_permitted'] is False
        and selection['additional_test_used_for_selection'] is False, 'changed selected production recipe')
    direct = _bindings([direct, selection['source_results']])
    for path, sha in selection['source_results'].items():
        _bound(path, sha)
    manifest = _read(cfg['data_manifest_path'])
    _require(manifest['schema'] == 'oracle-additional-window-data-manifest-v1', 'wrong raw data manifest')
    for key in ('data_cutoff', 'spot_path', 'spot_sha256', 'um_path', 'um_sha256'):
        _require(manifest[key] == cfg[key], 'changed data manifest ' + key)
    raw_bindings = _bindings([manifest['bindings']])
    for path, sha in raw_bindings.items():
        _bound(path, sha)
    resolved = {str(Path(p).resolve()) for p in raw_bindings}
    for kind in ('spot', 'um'):
        raw = Path(cfg[kind + '_path']); sidecar = raw.with_suffix('.sha256.json'); info = _read(sidecar)
        required = [raw, sidecar, Path(info['availability_path']), Path(info['source_ledger_path'])]
        if kind == 'um':
            required.append(Path(info['registration_path']))
        _require(all(str(p.resolve()) in resolved for p in required), 'missing raw provenance')
    spot_proof = validate_data_artifact(Path(cfg['spot_path']), expected_symbol='BTCUSDT')
    um, um_proof = validate_um(Path(cfg['um_path']), cfg['data_cutoff'], 'BTCUSDT')
    bars = load_bars(Path(cfg['spot_path']), cutoff=cfg['data_cutoff'])
    groups, common, components = common_feature_inputs(bars, um)
    outcomes = outcome_frame(bars, 24).to_numpy()
    masks = release_masks(bars.index, common, np.isfinite(outcomes).all(axis=1))
    dates = calendar(25)
    support = {'counts': {k: int(v.sum()) for k, v in masks.items()},
        'mask_sha256': {k: mask_digest(bars.index, v) for k, v in masks.items()},
        'last_label_maturity': {k: (bars.index[masks[k]][-1] + pd.Timedelta(minutes=375)).isoformat()
            for k in ('fit', 'scale', 'interval')},
        'feature_sha256': {name: {g: _matrix_digest(groups[g].to_numpy()[masks[name]]) for g in GROUPS}
            for name in ('fit', 'predict')},
        'label_sha256': {name: _matrix_digest(outcomes[masks[name]]) for name in ('fit', 'scale', 'interval')}}
    pre = {'schema': 'btc-reliability-release-preflight-v1',
        'config_contract_sha256': digest({k: v for k, v in cfg.items() if k != 'preflight_sha256'}),
        'source_bindings': cfg['source_bindings'], 'direct_source_bindings': direct,
        'raw_artifact_bindings': raw_bindings, 'spot_data_proof': spot_proof, 'um_data_proof': um_proof,
        'calendar': {k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k, v in dates.items()},
        'support': support, 'full_index_sha256': _index_digest(bars.index),
        'full_index_rows': len(bars), 'full_index_start': bars.index[0].isoformat(),
        'full_index_end_inclusive': bars.index[-1].isoformat(),
        'full_common_mask_sha256': mask_digest(bars.index, common),
        'common_component_mask_sha256': {name: mask_digest(bars.index, np.isfinite(frame.to_numpy()).all(axis=1))
            for name, frame in components.items()},
        'feature_columns': {g: list(groups[g].columns) for g in GROUPS},
        'selection_sha256': cfg['selection_sha256'],
        'new_models_weights_predictions_scores_or_orders_computed': False,
        'loader_scope': 'Inherited Spot full Parquet decode before strict semantic cutoff; UM filtered before feature cutoff',
        'historical_receipt_provenance_established': False,
        'historical_validation_scores_are_recipe_evidence_not_this_snapshot_performance': True}
    return {'config': cfg, 'bars': bars, 'groups': groups, 'outcomes': outcomes, 'masks': masks, 'preflight': pre}


def fit_production_inputs(groups, outcomes, masks):
    """Called only after freeze. Same three fits plus one scale-only w; no scores."""
    _require(not masks['inference'].any() and not masks['score'].any(), 'evaluation support forbidden')
    fitted = fit_frozen_forecasts(groups, outcomes, **{name + '_mask': masks[name]
        for name in ('fit', 'scale', 'interval', 'predict', 'inference')})
    _require(set(fitted['models']) == set(MODEL_NAMES), 'unexpected fitted model inventory')
    calibration = fitted['calibration']
    raw = fitted['raw_predictions']['perp_delay0']['mu']
    full = raw.copy()
    full += calibration['return_bias']['perp_delay0']
    reliability = fit_reliability(full, fitted['calibration_arrays']['actual'],
        scale_mask=masks['scale'], anchor=calibration['scale_mean'])
    predict = masks['predict']
    anchor = np.full(len(outcomes), calibration['scale_mean'])
    mu = apply_reliability(full, anchor, inference_mask=predict, weight=reliability['weight'])
    variance = np.maximum(fitted['raw_predictions']['technical']['variance'] * calibration['variance_multiplier'], 1e-12)
    _require(np.isfinite(mu[predict]).all() and np.isfinite(variance[predict]).all(), 'nonfinite production fixture')
    scalar_errors = {}
    for group in GROUPS:
        scaler, ridge = (fitted['models'][group + '_mean'].steps[i][1] for i in (0, 1))
        scalar = np.asarray([float(ridge.intercept_) + math.fsum(
            ((float(x) - float(center)) / float(scale)) * float(coef)
            for x, center, scale, coef in zip(row, scaler.mean_, scaler.scale_, ridge.coef_))
            for row in groups[group].to_numpy()[predict]])
        actual = fitted['raw_predictions'][group]['mu'][predict]
        error = float(np.max(np.abs(scalar - actual)))
        _require(np.isfinite(scalar).all() and error <= 1e-12, 'scalar Ridge prediction parity failed')
        scalar_errors[group] = error
    return {'fitted': fitted, 'reliability': reliability, 'mu': mu, 'variance': variance,
        'scalar_prediction_max_abs_difference': scalar_errors}


def _new_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def save_preflight(config_path):
    inputs = prepare(config_path)
    path = Path(inputs['config']['output_dir']) / 'preflight.json'
    if path.exists():
        _require(_read(path) == inputs['preflight'], 'changed immutable preflight')
    else:
        _new_json(path, inputs['preflight'])
    return {'path': str(path), 'sha256': file_digest(path)}


def run(config_path):
    inputs = prepare(config_path); cfg = inputs['config']; pre = inputs['preflight']
    out = Path(cfg['output_dir']); bundle = Path(cfg['bundle_dir'])
    _bound(out / 'preflight.json', cfg['preflight_sha256'])
    _require(_read(out / 'preflight.json') == pre, 'registered preflight differs')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    # The exact source/config must already exist in the checked-out commit.
    for path, sha in {**cfg['source_bindings'], str(config_path): file_digest(Path(config_path))}.items():
        committed = subprocess.check_output(['git', 'show', revision + ':' + str(Path(path).resolve().relative_to(Path.cwd().resolve()))])
        import hashlib
        _require(hashlib.sha256(committed).hexdigest() == sha, 'release source/config not frozen')
    _require(not bundle.exists(), 'bundle already exists; no repeat fitting')
    registration = {'config': cfg, 'config_sha256': file_digest(Path(config_path)),
        'preflight_sha256': cfg['preflight_sha256'], 'source_revision': revision,
        'scope': 'one three-model production fit and one S-only reliability coefficient; zero E outcomes/scores/orders'}
    _new_json(out / 'registration.json', registration)  # Exclusive creation also prevents partial retries.
    result = fit_production_inputs(inputs['groups'], inputs['outcomes'], inputs['masks'])
    fitted, reliability = result['fitted'], result['reliability']
    bundle.mkdir(parents=True)
    artifacts = {}
    def record(path):
        artifacts[str(Path(path).relative_to(bundle))] = {'sha256': file_digest(Path(path)), 'bytes': Path(path).stat().st_size}
    for name in MODEL_NAMES:
        path = bundle / 'models' / (name + '.joblib'); path.parent.mkdir(exist_ok=True)
        buf = io.BytesIO(); joblib.dump(fitted['models'][name], buf, compress=3)
        with path.open('xb') as stream:
            stream.write(buf.getvalue())
        record(path)
    for name, value in (('calibration.json', fitted['calibration']), ('reliability.json', reliability)):
        path = bundle / name; _new_json(path, value); record(path)
    masks, bars, groups = inputs['masks'], inputs['bars'], inputs['groups']
    snapshot = {'fit_positions': np.flatnonzero(masks['fit']), 'fit_timestamps': bars.index[masks['fit']].asi8,
        'predict_positions': np.flatnonzero(masks['predict']), 'predict_timestamps': bars.index[masks['predict']].asi8,
        'fit_actual': inputs['outcomes'][masks['fit']],
        'scale_mask_on_predict': masks['scale'][masks['predict']],
        'interval_mask_on_predict': masks['interval'][masks['predict']],
        'calibration_actual_on_predict': fitted['calibration_arrays']['actual'][masks['predict']]}
    for name in ('fit', 'predict'):
        for group in GROUPS:
            snapshot[name + '_features_' + group] = groups[group].to_numpy()[masks[name]]
    with (out / 'selected_inputs.npz').open('xb') as stream:
        np.savez_compressed(stream, **snapshot)
    fixture = {'timestamps': bars.index[masks['predict']].asi8,
        'technical_features': groups['technical'].to_numpy()[masks['predict']],
        'perp_features': groups['perp_delay0'].to_numpy()[masks['predict']],
        'raw_technical_mu': fitted['raw_predictions']['technical']['mu'][masks['predict']],
        'raw_perp_mu': fitted['raw_predictions']['perp_delay0']['mu'][masks['predict']],
        'raw_log_variance': fitted['raw_predictions']['technical']['log_variance'][masks['predict']],
        'mu': result['mu'][masks['predict']], 'variance': result['variance'][masks['predict']]}
    with (bundle / 'prediction_fixture.npz').open('xb') as stream:
        np.savez_compressed(stream, **fixture)
    record(bundle / 'prediction_fixture.npz')
    serialized_errors = {}
    for name in MODEL_NAMES:
        model = joblib.load(bundle / 'models' / (name + '.joblib'))
        group = 'perp_delay0' if name == 'perp_delay0_mean' else 'technical'
        expected = fitted['raw_predictions'][group]['log_variance' if name == 'technical_variance' else 'mu'][masks['predict']]
        got = model.predict(groups[group].to_numpy()[masks['predict']])
        _require(np.array_equal(got, expected), 'serialized model prediction changed')
        serialized_errors[name] = 0.
    provenance = {'source_revision': revision, 'preflight_sha256': cfg['preflight_sha256'],
        'fit_provenance': fitted['provenance'], 'support': pre['support'],
        'selected_inputs_sha256': file_digest(out / 'selected_inputs.npz'),
        'scalar_prediction_max_abs_difference': result['scalar_prediction_max_abs_difference'],
        'serialized_prediction_max_abs_difference': serialized_errors,
        'model_fit_count': 3, 'reliability_weight_count': 1, 'new_score_count': 0, 'new_policy_path_count': 0}
    _new_json(out / 'fit_provenance.json', provenance)
    feature_contract = {'schema_version': 1, 'symbol': 'BTCUSDT', 'interval': '15m',
        'bar_timestamp': 'UTC bar open', 'raw_fields': ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_quote', 'n_trades'],
        'completed_bars_only': True, 'causal_shift_bars': 1, 'minimum_history_bars': 8641,
        'feature_columns': pre['feature_columns'],
        'common_mask_components': sorted(pre['common_component_mask_sha256']),
        'common_mask_rule': 'finite intersection; no imputation or row compression',
        'missing_inference_rule': 'hold'}
    selection = _read(cfg['selection_path'])
    manifest = {'schema': 'btc-reliability-release-bundle-v1', 'schema_version': 1, 'bundle_version': 1,
        'bundle_type': 'btc_perp_reliability_v1', 'bundle_id': BUNDLE_ID, 'model_id': BUNDLE_ID,
        'candidate_id': CANDIDATE, 'model_kind': 'causal_ridge_hgb_scale_reliability',
        'status': 'research_release', 'symbol': 'BTCUSDT', 'interval': '15m',
        'source_revision': revision, 'data_cutoff_exclusive': cfg['data_cutoff'],
        'production_cutoff': cfg['data_cutoff'],
        'calendar': pre['calendar'], 'feature_columns': pre['feature_columns'],
        'model_parameters': fitted['provenance']['parameters'], 'training_counts': fitted['calibration']['counts'],
        'feature_shift_bars': 1, 'feature_history_minimum_bars': 8641,
        'common_support_contract': 'all_original_flow_all_derivative_all_delays0_1_4_all_trailing_variances',
        'feature_contract': feature_contract, 'feature_contract_sha256': digest(feature_contract),
        'execution': BUNDLE_EXECUTION, 'execution_contract_sha256': digest(BUNDLE_EXECUTION),
        'forecast': {'horizon_bars': 24, 'label_maturity_minutes': 375,
            'formula': 'w*(perp_Ridge_prediction+saved_bias)+(1-w)*saved_scale_mean',
            'variance_formula': 'max(exp(clip(technical_HGB_log_variance,log(1e-12),0))*saved_variance_multiplier,1e-12)',
            'weight': reliability['weight'], 'exact_weight_endpoints': True},
        'models': {name: 'models/' + name + '.joblib' for name in MODEL_NAMES},
        'calibration': 'calibration.json', 'reliability': 'reliability.json',
        'prediction_fixture': 'prediction_fixture.npz', 'artifacts': artifacts,
        'files': {name: value['sha256'] for name, value in artifacts.items()},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'registration_sha256': file_digest(out / 'registration.json'),
        'config_sha256': registration['config_sha256'], 'preflight_sha256': cfg['preflight_sha256'],
        'selection_path': cfg['selection_path'], 'selection_sha256': cfg['selection_sha256'],
        'source_bindings': cfg['source_bindings'], 'raw_artifact_bindings': pre['raw_artifact_bindings'],
        'selected_inputs_sha256': provenance['selected_inputs_sha256'],
        'fit_provenance_sha256': file_digest(out / 'fit_provenance.json'),
        'historical_validation_scores_are_recipe_evidence_not_this_snapshot_performance': True,
        'historical_evidence': {'selected': selection['selected'], 'validation_period': selection['validation_period'],
            'source_results': selection['source_results'], 'selection_sha256': cfg['selection_sha256']},
        'research_scores_apply_to_production_weights': False, 'high_probability_generalization_established': False,
        'rl_qualified': False, 'uncertainty_intervals_exposed': False,
        'production_evaluation_scores_computed': False, 'live_performance_claimed': False,
        'historical_receipt_provenance_established': False, 'rl_model_or_hindsight_teacher_used': False}
    _new_json(bundle / 'manifest.json', manifest)
    completion = {'schema': 'btc-reliability-release-completed-v1', 'bundle_id': BUNDLE_ID,
        'source_revision': revision, 'manifest_sha256': file_digest(bundle / 'manifest.json'),
        'artifact_sha256': {str(p): file_digest(p) for p in
            [out / name for name in ('preflight.json', 'registration.json', 'selected_inputs.npz', 'fit_provenance.json')]
            + [bundle / name for name in artifacts] + [bundle / 'manifest.json']},
        'new_model_fits': 3, 'new_reliability_weights': 1, 'new_scores': 0, 'new_policy_paths': 0}
    _new_json(out / 'completed.json', completion)
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    print(json.dumps(save_preflight(args.config) if args.preflight else run(args.config), sort_keys=True))
