"""Independent saved Stage15 audit; explicitly gated and never fits or plans.

Decimal60 summaries and independent scalar scores. No canonical experiment
helper, scorer, summary, model or planner is imported. Real execution requires
root authorization after the registered run has completed.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT = ROOT / 'codex_outputs/oracle_short_feature_decisions_v1'
PARENT = ROOT / 'codex_outputs/oracle_rolling_centering_decisions_v1'
RELIABILITY_ROOT = ROOT / 'codex_outputs/oracle_mean_reliability_decisions_v1'
FALLBACK = ROOT / 'codex_outputs/oracle_fallback_decisions_v1'
PAR = ROOT / 'codex_outputs/oracle_frozen_procedure_parity_v1'
RAW = ROOT / 'codex_outputs/oracle_derivative_delay_v1'
REPORT = Path('/tmp/oracle_short_feature_summary_audit_20260906.json')
FS = tuple(range(5, 13))
STRATA = ('all', 'bull', 'bear', 'sideways')
COSTS = ('base', 'stress_2x')
RULES = ('utility_risk1', 'utility_risk1_fallback_bh')
GROUPS = ('technical', 'technical_short_price', 'technical_short_flow', 'technical_short_both')
NEW = tuple(g + '_raw' for g in GROUPS[1:])
BASE = ('scale_mean', 'technical_scaled', 'perp_delay0_scaled', 'technical_half', 'perp_delay0_half')
RELIABILITY = ('technical_reliability', 'perp_delay0_reliability')
ROLLING = ('rolling_anchor', 'technical_rolling', 'perp_delay0_rolling')
OLD_MEANS = BASE + ('technical_raw', 'perp_delay0_raw') + RELIABILITY + ROLLING
SIMPLE = ('zero', 'fit_mean', 'technical_raw')
OLD_CONTROLS = ('bh', 'common_robust') + tuple(m + '_' + r for m in BASE + RELIABILITY + ROLLING for r in RULES)
EXTRA = tuple(m + '_' + r for m in SIMPLE for r in RULES)
CONTROLS = OLD_CONTROLS + EXTRA
NEW_IDS = tuple(m + '_' + r for m in NEW for r in RULES)
IDS = CONTROLS + NEW_IDS
SCORE_MEANS = {'interval': SIMPLE + NEW, 'evaluation': OLD_MEANS + ('zero', 'fit_mean') + NEW}
REFERENCES = {NEW[0]: ('technical_raw',), NEW[1]: ('technical_raw',),
              NEW[2]: ('technical_raw', NEW[0], NEW[1])}
EK = ('alpha_ex', 'maxdd_delta', 'turnover', 'trades')
BINDINGS = {}
CATEGORIES = Counter()
MAXDIFF = {}
LOCATIONS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bind(path, expected=None, category='direct'):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    actual = sha(path)
    assert expected is None or actual == expected, (str(path), 'hash changed')
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if key not in BINDINGS:
        BINDINGS[key] = actual
        CATEGORIES[category] += 1
    else:
        assert BINDINGS[key] == actual
    return path


def D(value):
    return Decimal(str(value))


def mean(values):
    return sum((D(x) for x in values), Decimal(0)) / Decimal(len(values))


def plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare(left, right, path, category):
    if isinstance(left, dict):
        assert set(left) == set(right), (path, 'schema')
        for key in left:
            compare(left[key], right[key], path + '/' + str(key), category)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), (path, 'length')
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path + '/' + str(i), category)
    elif isinstance(left, (int, float, Decimal)) and not isinstance(left, bool):
        assert math.isfinite(float(left)) and math.isfinite(float(right)), path
        difference = abs(D(left) - D(right))
        if category not in MAXDIFF or difference > MAXDIFF[category]:
            MAXDIFF[category] = difference
            LOCATIONS[category] = path
    else:
        assert left == right, (path, left, right)


def array_compare(left, right, path, *, exact=True):
    assert left.shape == right.shape and left.dtype == right.dtype, (path, 'shape/dtype')
    if left.dtype.kind in 'fc':
        assert np.array_equal(np.isnan(left), np.isnan(right)), (path, 'NaN support')
        assert np.array_equal(np.isfinite(left), np.isfinite(right)), (path, 'finite support')
    if exact:
        assert np.array_equal(left, right, equal_nan=True), (path, 'array changed')
    else:
        assert np.allclose(left, right, rtol=1e-12, atol=1e-14, equal_nan=True), path
        good = np.isfinite(left)
        value = float(np.max(np.abs(left[good] - right[good]))) if good.any() else 0.
        compare(value, 0., path, 'baseline_forecast_parity')


def arrays_equal(left, right, path):
    assert set(left) == set(right), (path, 'keys')
    for key in left:
        array_compare(left[key], right[key], path + '/' + key)


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def scalar_scores(actual, prediction, fit_mean):
    n = len(actual)
    assert n >= 16 and np.isfinite(actual).all() and np.isfinite(prediction).all()
    errors = [float(a) - float(b) for a, b in zip(actual, prediction)]
    ic = None
    if len(np.unique(actual)) > 1 and len(np.unique(prediction)) > 1:
        ry, rp = ranks(actual), ranks(prediction)
        center = (n + 1) / 2
        numerator = math.fsum((float(a) - center) * (float(b) - center) for a, b in zip(ry, rp))
        denominator = math.sqrt(math.fsum((float(a) - center) ** 2 for a in ry)
                                * math.fsum((float(a) - center) ** 2 for a in rp))
        ic = numerator / denominator
    return {'rows': n, 'return_mse': math.fsum(e * e for e in errors) / n,
            'return_mae': math.fsum(abs(e) for e in errors) / n,
            'zero_return_mse': math.fsum(float(y) ** 2 for y in actual) / n,
            'fit_mean_return_mse': math.fsum((float(y) - fit_mean) ** 2 for y in actual) / n,
            'return_sign_accuracy': sum((float(a) > 0) == (float(b) > 0) for a, b in zip(actual, prediction)) / n,
            'return_rank_ic': ic}


def audit():
    with localcontext() as context:
        context.prec = 60
        result = read(bind(OUT / 'results.json'))
        registration = read(bind(OUT / 'registration.json'))
        cfg = registration['config']
        pre = read(bind(OUT / 'preflight.json', cfg['preflight_sha256']))
        config_path = ROOT / 'configs/oracle_short_feature_decisions_20260906.yaml'
        bind(config_path, registration['config_sha256'], 'config')
        assert result['registration_sha256'] == canonical(registration)
        assert registration['source_bindings'] == cfg['source_bindings'] == pre['source_bindings']
        assert registration['preflight_sha256'] == cfg['preflight_sha256']
        assert pre['config_contract_sha256'] == canonical({k: v for k, v in cfg.items() if k != 'preflight_sha256'})
        assert len(cfg['source_bindings']) == 30
        for path, value in cfg['source_bindings'].items():
            bind(path, value, 'registered_source')
        for path, value in pre['direct_source_bindings'].items():
            bind(path, value, 'preflight_direct')
        for prefix in ('source_prepare', 'parity_prepare'):
            bind(cfg[prefix + '_config'], cfg[prefix + '_config_sha256'], 'prepare_config')
        ancestor = pre['source_artifact_bindings']
        assert len(ancestor) == 1792
        for path, value in ancestor.items():
            bind(path, value, 'ancestor_artifact')
        assert len({str((ROOT / p).resolve()) for p in ancestor}) == 1792
        parent = read(bind(PARENT / 'results.json', cfg['parent_results_sha256']))
        parent_reg = read(bind(PARENT / 'registration.json', cfg['parent_registration_sha256']))
        parent_pre = read(bind(PARENT / 'preflight.json', cfg['parent_preflight_sha256']))
        fallback = read(bind(FALLBACK / 'results.json', cfg['fallback_results_sha256']))
        fallback_reg = read(bind(FALLBACK / 'registration.json', cfg['fallback_registration_sha256']))
        assert parent_reg['source_revision'] == cfg['parent_source_revision']
        assert parent['registration_sha256'] == canonical(parent_reg)
        assert fallback['registration_sha256'] == canonical(fallback_reg)
        assert pre['parent_preflight_sha256'] == canonical(parent_pre)
        rows, scores = result['rows'], result['scores']
        ri = {(r['fold'], r['candidate_id']): r for r in rows}
        si = {(s['fold'], s['segment'], s['mean_id']): s for s in scores}
        assert len(rows) == len(ri) == 272 and set(ri) == {(f, p) for f in FS for p in IDS}
        assert len(scores) == len(si) == 184
        assert set(si) == {(f, seg, m) for f in FS for seg in SCORE_MEANS for m in SCORE_MEANS[seg]}
        assert dict(Counter(s['segment'] for s in scores)) == {'interval': 48, 'evaluation': 136}
        for key, expected in {'return_model_fits': 32, 'new_return_model_fits': 24,
                              'baseline_parity_fits': 8, 'risk_model_fits': 0,
                              'calibration_weight_fits': 0, 'new_causal_policy_names': 6,
                              'total_adaptively_explored_causal_names': 174}.items():
            assert result[key] == expected
        for key in ('additional_test_used_for_modeling_or_scoring', 'selection_performed',
                    'teacher_use_allowed', 'high_probability_generalization_established'):
            assert result[key] is False
        regimes = {f: ri[f, 'bh']['regime'] for f in FS}
        regime_counts = dict(Counter(value['trend'] for value in regimes.values()))
        assert regime_counts == {'bull': 2, 'bear': 4, 'sideways': 2}
        for row in rows:
            assert row['regime'] == regimes[row['fold']]
            for cost in COSTS:
                assert all(math.isfinite(row[cost][key]) for key in EK)
        for score in scores:
            assert score['regime'] == regimes[score['fold']]
            assert score['regime_known_at_scored_decisions'] is (score['segment'] == 'evaluation')
            assert score['regime_reference'] == 'evaluation_quarter_start'
            assert score['rows'] == si[score['fold'], score['segment'], 'technical_raw']['rows']
            assert score['rows'] >= 16
        assert sum(si[f, 'evaluation', 'technical_raw']['rows'] for f in FS) == 2574
        artifacts = {}
        for f in FS:
            manifest = read(bind(OUT / f'fold_{f}.json', category='fold_manifest'))
            assert manifest['registration_sha256'] == result['registration_sha256']
            for key in ('rows', 'scores'):
                assert manifest[key] == [r for r in result[key] if r['fold'] == f]
            assert manifest['baseline_parity'] == next(p for p in result['baseline_parity'] if p['fold'] == f)
            expected = {str((OUT / 'models' / f'fold{f}_{g}.joblib').relative_to(ROOT)) for g in GROUPS}
            expected |= {str((OUT / 'calibration' / f'fold{f}_{g}.npz').relative_to(ROOT)) for g in GROUPS}
            expected |= {str((OUT / 'forecasts' / f'fold{f}_{g}_raw.npz').relative_to(ROOT)) for g in GROUPS}
            expected |= {str((OUT / 'targets' / f'fold{f}_{p}.npz').relative_to(ROOT)) for p in IDS}
            expected |= {str((OUT / 'traces' / f'fold{f}_{p}.json').relative_to(ROOT)) for p in NEW_IDS}
            expected.add(str((OUT / 'provenance' / f'fold{f}_fit.json').relative_to(ROOT)))
            assert set(manifest['artifact_sha256']) == expected and len(expected) == 53
            for path, value in manifest['artifact_sha256'].items():
                assert path not in artifacts
                artifacts[path] = value
                bind(path, value, 'new_artifact')
            provenance = read(OUT / 'provenance' / f'fold{f}_fit.json')
            support = next(p for p in pre['support'] if p['fold'] == f)
            assert provenance['fit_source_binding'] == support
            assert provenance['baseline_parity'] == manifest['baseline_parity']
            assert provenance['risk_source'] == 'unchanged technical_scaled'
            fp = provenance['fit_provenance']
            assert fp['feature_columns'] == support['feature_columns']
            assert [fp['feature_counts'][g] for g in GROUPS] == [29, 34, 32, 37]
            assert fp['mask_counts'] == {k: support['counts'][k] for k in ('fit', 'predict')}
            for key in ('model_selection_performed', 'evaluation_labels_used', 'risk_or_calibration_fitted',
                        'timestamp_feature_causality_and_label_completion_verified'):
                assert fp[key] is False
        assert len(artifacts) == 424
        for row in rows:
            target = str((OUT / 'targets' / f"fold{row['fold']}_{row['candidate_id']}.npz").relative_to(ROOT))
            assert row['targets_sha256'] == artifacts[target]
            if row['candidate_id'] in NEW_IDS:
                trace = str((OUT / 'traces' / f"fold{row['fold']}_{row['candidate_id']}.json").relative_to(ROOT))
                assert row['trace_sha256'] == artifacts[trace]
        assert len(result['baseline_parity']) == 8
        assert {x['fold'] for x in result['baseline_parity']} == set(FS)
        for value in result['baseline_parity']:
            for key, difference in value['model_state'].items():
                compare(difference, 0., f"fold{value['fold']}/model/{key}", 'reported_baseline_model_difference')
            for key in ('calibration_raw_maxdiff', 'evaluation_raw_maxdiff'):
                compare(value[key], 0., f"fold{value['fold']}/{key}", 'reported_baseline_forecast_difference')
        econ, pred, paired = {}, {}, {}
        for stratum in STRATA:
            fs = [f for f in FS if stratum == 'all' or regimes[f]['trend'] == stratum]
            econ[stratum] = {p: {'quarters': len(fs), 'joint_positive_quarters_both_costs': sum(
                all(ri[f, p][c]['alpha_ex'] > 0 and ri[f, p][c]['maxdd_delta'] < 0 for c in COSTS) for f in fs),
                **{c: {k: mean([ri[f, p][c][k] for f in fs]) for k in EK} for c in COSTS}} for p in IDS}
            pred[stratum] = {}
            for seg, means in SCORE_MEANS.items():
                pred[stratum][seg] = {}
                for m in means:
                    ss = [si[f, seg, m] for f in fs]
                    n = sum(s['rows'] for s in ss)
                    pred[stratum][seg][m] = {
                        'quarters': len(fs), 'rows': n,
                        'equal_quarter_mse': mean([s['return_mse'] for s in ss]),
                        'pooled_row_mse': sum(D(s['return_mse']) * Decimal(s['rows']) for s in ss) / Decimal(n),
                        'equal_quarter_mae': mean([s['return_mae'] for s in ss]),
                        'zero_return_mse': mean([s['zero_return_mse'] for s in ss]),
                        'fit_mean_return_mse': mean([s['fit_mean_return_mse'] for s in ss]),
                        'mse_minus_zero': mean([D(s['return_mse']) - D(s['zero_return_mse']) for s in ss]),
                        'mse_minus_fit_mean': mean([D(s['return_mse']) - D(s['fit_mean_return_mse']) for s in ss]),
                        'mean_rank_ic': mean([s['return_rank_ic'] for s in ss])
                        if all(s['return_rank_ic'] is not None for s in ss) else None}
            paired[stratum] = {}
            for m in NEW:
                paired[stratum][m] = {}
                for ref in REFERENCES[m]:
                    predictions = {}
                    for seg in SCORE_MEANS:
                        differences = [D(si[f, seg, m]['return_mse']) - D(si[f, seg, ref]['return_mse']) for f in fs]
                        reference_loss = mean([si[f, seg, ref]['return_mse'] for f in fs])
                        predictions[seg] = {'mse_difference': mean(differences),
                            'relative_mse_reduction': -mean(differences) / reference_loss if reference_loss else None,
                            'improved_quarters': sum(d < 0 for d in differences),
                            'equal_quarters': sum(d == 0 for d in differences)}
                    paired[stratum][m][ref] = {'prediction': predictions,
                        'economics': {r: {c: {k: mean([D(ri[f, m + '_' + r][c][k]) -
                            D(ri[f, ref + '_' + r][c][k]) for f in fs]) for k in EK} for c in COSTS} for r in RULES}}
        direction = {}
        for m in NEW:
            predictive = {seg: all(pred[g][seg][m]['mse_minus_zero'] < 0 and
                pred[g][seg][m]['mse_minus_fit_mean'] < 0 and all(
                    paired[g][m][ref]['prediction'][seg]['mse_difference'] < 0 for ref in REFERENCES[m])
                for g in STRATA) for seg in SCORE_MEANS}
            for rule in RULES:
                cid = m + '_' + rule
                direction[cid] = {'economic_means_all_strata_both_costs': all(
                    econ[g][cid][c]['alpha_ex'] > 0 and econ[g][cid][c]['maxdd_delta'] < 0 for g in STRATA for c in COSTS),
                    'predictive_mse_vs_zero_fitmean_and_all_references_all_strata': predictive,
                    'regime_count_gate_pass': False, 'high_probability_generalization_established': False}
        summary = {'economics': econ, 'prediction': pred, 'paired': paired, 'direction': direction,
            'regime_counts': regime_counts, 'interval_regime_strata_are_retrospective_evaluation_groupings': True,
            'individual_feature_effects_identified': False, 'selection_performed': False,
            'high_probability_generalization_established': False, 'regime_count_gate_pass': False}
        for key, value in summary.items():
            compare(value, result['summary'][key], key, 'summary_' + key)
        assert set(summary) == set(result['summary'])
        cache = {}
        def npz(path):
            key = str(path.relative_to(ROOT))
            assert key in BINDINGS, (key, 'unbound consumed artifact')
            if key not in cache:
                with np.load(path, allow_pickle=False) as content:
                    cache[key] = {k: content[k] for k in content.files}
            return cache[key]
        parent_ri = {(r['fold'], r['candidate_id']): r for r in parent['rows']}
        fallback_ri = {(r['fold'], r['candidate_id']): r for r in fallback['rows']}
        parent_si = {(s['fold'], s['mean_id']): s for s in parent['scores']}
        copied_controls = copied_scores = rescored = 0
        rank_checks, support_counts = [], []
        for f in FS:
            ref = npz(PAR / 'forecasts' / f'fold{f}_scale_mean.npz')
            fit_mean = float(ref['fit_return_mean'])
            old_cal = npz(PAR / 'calibration' / f'fold{f}_technical.npz')
            stream = {m: npz(PAR / 'forecasts' / f'fold{f}_{m}.npz')['mu'] for m in BASE}
            stream.update({m: npz(RAW / 'forecasts' / f'fold{f}_{m}.npz')['mu']
                           for m in ('technical_raw', 'perp_delay0_raw')})
            stream.update({m: npz(RELIABILITY_ROOT / 'forecasts' / f'fold{f}_{m}.npz')['mu'] for m in RELIABILITY})
            stream.update({m: npz(PARENT / 'forecasts' / f'fold{f}_{m}.npz')['mu'] for m in ROLLING})
            stream.update({'zero': np.where(ref['inference_mask'], 0., np.nan),
                           'fit_mean': np.where(ref['inference_mask'], fit_mean, np.nan)})
            cal_stream = {'zero': np.where(old_cal['interval_mask'], 0., np.nan),
                          'fit_mean': np.where(old_cal['interval_mask'], fit_mean, np.nan)}
            for g in GROUPS:
                m = g + '_raw'
                z = npz(OUT / 'forecasts' / f'fold{f}_{m}.npz')
                assert set(z) == set(ref)
                for key in ref:
                    if key != 'mu':
                        array_compare(z[key], ref[key], f'{f}/{m}/shared/{key}')
                assert np.array_equal(np.isfinite(z['mu']), ref['inference_mask'])
                if g == 'technical':
                    array_compare(z['mu'], stream[m], f'{f}/technical/refit_E', exact=False)
                stream[m] = z['mu']
                cal = npz(OUT / 'calibration' / f'fold{f}_{g}.npz')
                assert set(cal) == {'timestamps', 'actual', 'mu', 'scale_mask', 'interval_mask'}
                for key in ('timestamps', 'actual', 'scale_mask', 'interval_mask'):
                    array_compare(cal[key], old_cal[key], f'{f}/{g}/cal/{key}')
                assert np.array_equal(np.isfinite(cal['mu']), np.isfinite(old_cal['mu']))
                if g == 'technical':
                    array_compare(cal['mu'], old_cal['mu'], f'{f}/technical/refit_SI', exact=False)
                cal_stream[m] = cal['mu']
            for seg, means in SCORE_MEANS.items():
                actual = ref['actual'] if seg == 'evaluation' else old_cal['actual']
                support = ref['score_support'] if seg == 'evaluation' else old_cal['interval_mask']
                predictions = stream if seg == 'evaluation' else cal_stream
                assert support.dtype == np.dtype(bool)
                for m in means:
                    measured = scalar_scores(actual[support, 0], predictions[m][support], fit_mean)
                    original = si[f, seg, m]
                    compare(measured, {k: original[k] for k in measured}, f'{f}/{seg}/{m}', 'scalar_rescore')
                    rescored += 1
                    if seg == 'evaluation' and m in OLD_MEANS:
                        compare({k: original[k] for k in measured}, {k: parent_si[f, m][k] for k in measured},
                                f'{f}/{m}/copied_score', 'unchanged_old_scores')
                        copied_scores += 1
                    rank_checks.append({'fold': f, 'segment': seg, 'mean_id': m,
                        'unique_predictions': len(np.unique(predictions[m][support])),
                        'rank_ic_is_defined': measured['return_rank_ic'] is not None})
            support_counts.append({'fold': f, 'inference': int(ref['inference_mask'].sum()),
                'evaluation_scored': int(ref['score_support'].sum()),
                'interval_scored': int(old_cal['interval_mask'].sum())})
            for cid in CONTROLS:
                source, source_ri = (FALLBACK, fallback_ri) if cid in EXTRA else (PARENT, parent_ri)
                arrays_equal(npz(OUT / 'targets' / f'fold{f}_{cid}.npz'),
                             npz(source / 'targets' / f'fold{f}_{cid}.npz'), f'{f}/{cid}/target')
                for cost in COSTS:
                    compare(ri[f, cid][cost], source_ri[f, cid][cost], f'{f}/{cid}/{cost}', 'unchanged_old_accounts')
                copied_controls += 1
        assert rescored == 184 and copied_scores == 96 and copied_controls == 224
        assert sum(x['inference'] for x in support_counts) == 2586
        assert sum(x['evaluation_scored'] for x in support_counts) == 2574
        assert all(value <= Decimal('1e-12') for value in MAXDIFF.values()), plain(MAXDIFF)
        failures = {}
        for m in NEW:
            failures[m] = {'predictive_failed_references': {}, 'economic_failed_margins': {}}
            for seg in SCORE_MEANS:
                failures[m]['predictive_failed_references'][seg] = {
                    g: [ref for ref, difference in [('zero', pred[g][seg][m]['mse_minus_zero']),
                        ('fit_mean', pred[g][seg][m]['mse_minus_fit_mean'])] +
                        [(ref, paired[g][m][ref]['prediction'][seg]['mse_difference']) for ref in REFERENCES[m]]
                        if difference >= 0] for g in STRATA}
            for rule in RULES:
                cid = m + '_' + rule
                failures[m]['economic_failed_margins'][rule] = [
                    {'stratum': g, 'cost': c, 'endpoint': k, 'value': econ[g][cid][c][k]}
                    for g in STRATA for c in COSTS for k in ('alpha_ex', 'maxdd_delta')
                    if (econ[g][cid][c][k] <= 0 if k == 'alpha_ex' else econ[g][cid][c][k] >= 0)]
        tiny_dd = [{'fold': r['fold'], 'candidate_id': r['candidate_id'], 'cost': c,
                    'maxdd_delta': r[c]['maxdd_delta'], 'registered_sign_is_negative': r[c]['maxdd_delta'] < 0}
                   for r in rows for c in COSTS if 0 < abs(r[c]['maxdd_delta']) < 1e-12]
        report = {'schema': 'independent-short-feature-summary-audit-v1', 'passed': True,
            'scope': 'Original reused-development saved artifacts only. Decimal60 summary aggregation and independent scalar184score/rank checks; no canonical summary/scorer import, fit, policy, raw market loader or additional period.',
            'source_revision': registration['source_revision'],
            'audit_script': {'path': str(Path(__file__)), 'sha256': sha(Path(__file__))},
            'inventory': {'economic_rows': 272, 'cost_account_summaries': 544, 'policies': 34,
                'scores': 184, 'interval_scores': 48, 'evaluation_scores': 136,
                'new_artifacts': 424, 'ancestral_artifacts': 1792, 'registered_sources': 30,
                'unchanged_control_rows': copied_controls, 'unchanged_control_accounts': copied_controls * 2,
                'unchanged_old_evaluation_scores': copied_scores, 'regime_counts': regime_counts,
                'registered_forecast_contrasts': 5, 'new_mean_streams': 3},
            'verified_binding_counts': dict(CATEGORIES), 'source_sha256': BINDINGS,
            'binding_scope': 'Every registered source, direct-preflight file, ancestor artifact and new fold artifact was independently rehashed. Raw archive contents not enumerated here inherit earlier bound source proofs.',
            'numeric_max_absolute_differences': MAXDIFF, 'maximum_difference_locations': LOCATIONS,
            'economics': econ, 'prediction': pred, 'paired': paired, 'direction': direction,
            'failed_margins': failures, 'rank_checks': rank_checks, 'support_counts': support_counts,
            'tiny_nonzero_dd_values': tiny_dd,
            'limitations': [
                'All8 development quarters and histories are reused; 2/4/2 regimes cannot establish independent high-probability robustness.',
                'Interval strata label the later evaluation-start regime retrospectively; interval/evaluation losses across folds overlap.',
                'Relative loss reduction is a ratio of equal-quarter losses, not an equal-quarter average of loss ratios.',
                'Raw means change while technical S-calibrated risk and execution remain frozen; economic changes do not isolate a pure return-predictability effect.',
                'Block comparisons do not identify individual feature effects; adding correlated coordinates changes Ridge regularization geometry.',
                'Small nonzero DD values are exposed without changing strict registered sign counts; B&H reference roundoff is not economic improvement.',
                'Regression reconstruction and full own-state account reconstruction belong to separate independent audits; this audit checks their stored inputs, source identities and summaries.',
                'No new model, policy, selection, significance claim, confidence interval, test-data access or promotion.']}
        REPORT.write_text(json.dumps(plain(report), ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False) + '\n')
        print(json.dumps({'output': str(REPORT), 'sha256': sha(REPORT), 'passed': True,
                          'inventory': report['inventory'], 'bindings': dict(CATEGORIES),
                          'maxdiff': plain(MAXDIFF), 'direction': direction}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run only after root authorizes completed Stage15 results.')
    parser.add_argument('--execute-saved-audit', action='store_true', required=True)
    parser.parse_args()
    audit()
