"""Independent Stage16 forecast/tail-score audit; real mode requires root GO.

No Stage16 substitution, tail, scoring or planning helper is imported. Old
SHA-bound parity preparation supplies the inherited feature-availability masks;
selected h24 returns are independently reconstructed from bounded Spot bars.
There are no fits, new strategies, alternative thresholds or policy rollouts.
Default mode is synthetic-only and never reads workspace data or outcomes.
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
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
CONFIG = Path('configs/oracle_sign_magnitude_decisions_20260906.yaml')
OUT = Path('codex_outputs/oracle_sign_magnitude_decisions_v1')
PARENT = Path('codex_outputs/oracle_information_decomposition_v1')
PARITY = Path('codex_outputs/oracle_frozen_procedure_parity_v1')
REPORT = Path('/tmp/oracle_sign_magnitude_forecast_audit_20260906.json')
FOLDS = tuple(range(5, 13))
HALVES = ('technical_half', 'perp_delay0_half')
RULES = ('utility_risk1', 'utility_risk1_fallback_bh')
CELLS = ('base', 'sign', 'magnitude', 'full')
SUBSETS = ('all', 'fit_q90_large', 'fit_q90_other')
KEYS = {'timestamps', 'mu', 'variance', 'actual', 'inference_mask', 'score_support', 'fit_return_mean'}
TOLERANCES = {'threshold_atol': 1e-15, 'score_atol': 1e-14,
              'rank_ic_atol': 1e-12, 'sign_loss_identity_atol': 1e-14}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def exact(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, (name, 'shape/dtype')
    assert np.array_equal(a, b, equal_nan=True), (name, 'values')
    if a.dtype.kind == 'f':
        exact(np.signbit(a[a == 0]), np.signbit(b[b == 0]), name + ' signed zeros')


def close(a, b, name, maxima, tol=1e-14):
    if a is None or b is None:
        assert a is b, (name, a, b)
        return
    assert math.isfinite(float(a)) and math.isfinite(float(b)), name
    delta = abs(float(a) - float(b)); maxima[name] = max(maxima.get(name, 0.), delta)
    assert delta <= tol, (name, delta, tol)


def mask_sha(index, mask):
    return hashlib.sha256(index.asi8.tobytes() + np.asarray(mask, bool).tobytes()).hexdigest()


def scalar_sign(x): return 1. if x > 0 else -1. if x < 0 else 0.


def scalar_quantile(values):
    """Fixed linear sample quantile, sorted scalars and direct interpolation."""
    ordered = sorted(abs(float(v)) for v in values)
    assert ordered and all(math.isfinite(v) for v in ordered)
    position = (len(ordered) - 1) * .9
    lo = math.floor(position); hi = math.ceil(position); fraction = position - lo
    a, b = ordered[lo], ordered[hi]
    # Both branches are the same linear interpolation, choosing the nearer end.
    return a + (b - a) * fraction if fraction < .5 else b - (b - a) * (1 - fraction)


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [0.] * len(values)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and values[order[k]] == values[order[j]]: k += 1
        rank = (j + 1 + k) / 2
        for i in order[j:k]: result[i] = rank
        j = k
    return result


def scalar_scores(y, mu, fit_mean):
    y, mu = [float(v) for v in y], [float(v) for v in mu]
    n = len(y); assert len(mu) == n
    names = ('return_mse', 'return_mae', 'return_sign_accuracy', 'zero_return_mse',
             'fit_mean_return_mse', 'return_rank_ic')
    if not n: return {'rows': 0, **{k: None for k in names}}
    ic = None
    if len(set(y)) > 1 and len(set(mu)) > 1:
        ry, rm = average_ranks(y), average_ranks(mu); center = (n + 1) / 2
        cross = math.fsum((a - center) * (b - center) for a, b in zip(ry, rm))
        dy = math.fsum((v - center) ** 2 for v in ry); dm = math.fsum((v - center) ** 2 for v in rm)
        ic = cross / math.sqrt(dy * dm)
    return {'rows': n,
        'return_mse': math.fsum((a - b) ** 2 / n for a, b in zip(y, mu)),
        'return_mae': math.fsum(abs(a - b) / n for a, b in zip(y, mu)),
        'return_sign_accuracy': sum((a > 0) == (b > 0) for a, b in zip(y, mu)) / n,
        'zero_return_mse': math.fsum(a * a / n for a in y),
        'fit_mean_return_mse': math.fsum((a - fit_mean) ** 2 / n for a in y),
        'return_rank_ic': ic}


def scalar_substitute(base, actual, score, component):
    result = base.copy()
    for i in np.flatnonzero(score):
        m, y = float(base[i]), float(actual[i, 0])
        result[i] = scalar_sign(y) * abs(m) if component == 'sign' else scalar_sign(m) * abs(y)
    return result


def synthetic():
    assert scalar_quantile([0., 1.] * 256) == 1.
    assert scalar_quantile([0.] * 512) == 0.
    assert abs(scalar_quantile(list(range(512))) - 459.9) < 1e-13
    exact(np.asarray(average_ranks([3., 1., 1., 7.])), np.asarray([3., 1.5, 1.5, 4.]), 'ranks')
    y = np.asarray([[2., 8., 9.], [-2., 8., 9.], [0., 8., 9.], [4., 8., 9.], [np.nan] * 3])
    m = np.asarray([-3., -1., 5., 0., np.nan]); score = np.asarray([True] * 4 + [False])
    s = scalar_substitute(m, y, score, 'sign'); g = scalar_substitute(m, y, score, 'magnitude')
    exact(s, np.asarray([3., -1., 0., 0., np.nan]), 'sign')
    exact(g, np.asarray([-2., -2., 0., 0., np.nan]), 'magnitude')
    for a, b, c in zip(y[score, 0], m[score], s[score]):
        gain = (a - b) ** 2 - (a - c) ** 2
        rhs = b * b if a == 0 else 4 * abs(a * b) if a * b < 0 else 0.
        assert gain >= 0 and gain == rhs
    assert scalar_scores([1., -1.], [0., 0.], 0.)['return_mse'] == 1.
    assert scalar_scores([1., -1.], [1., -1.], 0.)['return_mse'] == 0.
    assert scalar_scores([], [], 0.)['return_mse'] is None
    assert all(abs(v) >= scalar_quantile([0.] * 512) for v in [0., -0., 1.])
    print(json.dumps({'synthetic_passed': True, 'real_data_or_outputs_read': False}))


def real(revision, expected_results_sha):
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    verified = {}; counts = Counter(); maxima = {}; records = []; direct = {}

    def verify(path, expected):
        key = str(Path(path).resolve())
        if key not in verified: verified[key] = sha(path)
        assert verified[key] == expected, ('sha256', str(path))
        counts['binding_checks'] += 1

    verify(OUT / 'results.json', expected_results_sha)
    reg, pre, result = [read(OUT / (name + '.json')) for name in ('registration', 'preflight', 'results')]
    cfg = yaml.safe_load(CONFIG.read_text())
    verify(CONFIG, reg['config_sha256']); assert reg['config'] == cfg
    assert reg['source_revision'] == revision
    assert hashlib.sha256(subprocess.check_output(['git', 'show', revision + ':' + str(CONFIG)])).hexdigest() == reg['config_sha256']
    assert result['registration_sha256'] == digest(reg)
    assert cfg['development_folds'] == list(FOLDS) and cfg['halves'] == list(HALVES)
    assert cfg['data_cutoff'] == '2023-04-16T13:45:00Z' and cfg['tail_quantile'] == .9
    assert cfg['tail_quantile_method'] == 'linear' and cfg['tail_fit_only'] is True
    assert cfg['replacement_support'] == 'saved_score_only_keep_own_learned_elsewhere'
    assert cfg['components'] == ['sign', 'magnitude'] and cfg['new_model_fits'] == 0
    verify(OUT / 'preflight.json', cfg['preflight_sha256'])
    assert reg['preflight_sha256'] == cfg['preflight_sha256']
    assert pre['config_contract_sha256'] == digest({k: v for k, v in cfg.items() if k != 'preflight_sha256'})
    for p, h in cfg['source_bindings'].items():
        verify(p, h)
        assert hashlib.sha256(subprocess.check_output(['git', 'show', revision + ':' + p])).hexdigest() == h
    for p, h in pre['direct_source_bindings'].items(): verify(p, h)
    for p, h in pre['source_artifact_bindings'].items(): verify(p, h)
    assert len(pre['source_artifact_bindings']) == 1728
    for p in (CONFIG, OUT/'registration.json', OUT/'preflight.json', OUT/'results.json'):
        direct[str(p)] = sha(p)
    # This frozen helper is data-only: no fit, new mean, quantile or policy call.
    from unidream.experiments.oracle_frozen_procedure_parity import prepare as old_prepare
    _, _, fc, bars, _, _, inherited_y, masks, *_ = old_prepare(Path(cfg['parity_config']))
    cutoff = pd.Timestamp(cfg['data_cutoff'])
    proof = pre['spot_data_proof']; verify(fc['data_path'], proof['artifact_sha256'])
    bounded = pd.read_parquet(fc['data_path'], columns=['open', 'close'], filters=[('bar_open_ts', '<', cutoff)])
    assert bounded.index.max() < cutoff and bars.index.max() < cutoff
    bounded = bounded.reindex(bars.index)
    for k in ('open', 'close'): exact(bounded[k].to_numpy(), bars[k].to_numpy(), 'bounded ' + k)
    open_values, close_values = bounded.open.to_numpy(), bounded.close.to_numpy()
    available = bars.bar_available.to_numpy(bool)

    def selected_returns(positions):
        assert all(i + 24 < len(bars) and available[i+1:i+25].all() for i in positions)
        # Only supplied historical T or existing O rows are decoded into labels.
        return np.log(close_values[positions+24] / open_values[positions+1])

    old_result = read(PARENT/'results.json')
    old_rows = {(r['fold'], r['candidate_id']): r for r in old_result['rows']}
    rows = {(r['fold'], r['candidate_id']): r for r in result['rows']}
    scores = {(s['fold'], s['mean_id'], s['subset']): s for s in result['scores']}
    directions = {(d['fold'], d['mean_id'], d['subset']): d for d in result['direction_diagnostics']}
    assert len(rows) == len(result['rows']) == 288
    assert set(rows) == {(f, c) for f in FOLDS for c in cfg['control_ids'] + cfg['new_diagnostic_ids']}
    assert len(scores) == len(result['scores']) == 192
    assert set(scores) == {(f, m, u) for f in FOLDS for m in cfg['score_means'] for u in SUBSETS}
    assert len(directions) == len(result['direction_diagnostics']) == 48
    endpoints = {(r['fold'], r['candidate_id']): r for r in result['endpoint_parity']}
    expected_endpoints = {(f, h + ('_oracle_return' if cell == 'full' else '') + '_' + rule)
                          for f in FOLDS for h in HALVES for cell in ('base', 'full') for rule in RULES}
    assert len(endpoints) == len(result['endpoint_parity']) == 64 and set(endpoints) == expected_endpoints
    for (_, cid), check in endpoints.items():
        assert check['targets_exact'] is True
        assert check['full_decision_trace_matches'] is ('_oracle_return_' in cid)
    runtime_bindings = {}; algebra = Counter(); algebra_residual = 0.
    for f in FOLDS:
        fold = read(OUT/f'fold_{f}.json')
        assert fold['registration_sha256'] == digest(reg)
        assert fold['rows'] == [r for r in result['rows'] if r['fold'] == f]
        assert fold['scores'] == [r for r in result['scores'] if r['fold'] == f]
        assert fold['endpoint_parity'] == [r for r in result['endpoint_parity'] if r['fold'] == f]
        for p, h in fold['artifact_sha256'].items(): verify(p, h); runtime_bindings[p] = h
        assert len(fold['artifact_sha256']) == 49
        T = masks[f]['fit']; positions = np.flatnonzero(T)
        fitted = selected_returns(positions)
        exact(fitted, inherited_y[T, 0], 'independent T labels')
        bound = next(x for x in pre['support'] if x['fold'] == f)
        assert digest(fitted.tolist()) == bound['fit_return_sha256']
        assert mask_sha(bars.index, T) == bound['fit_mask_sha256']
        q = scalar_quantile(fitted); threshold = read(OUT/'thresholds'/f'fold{f}_fit_q90.json')
        assert threshold == fold['threshold'] == next(x for x in result['thresholds'] if x['fold'] == f)
        close(q, threshold['threshold'], 'threshold', maxima, TOLERANCES['threshold_atol'])
        assert threshold['fit_return_sha256'] == digest(fitted.tolist()) and threshold['fit_rows'] == len(fitted)
        assert threshold['quantile'] == .9 and threshold['method'] == 'linear'
        assert threshold['used_for_orders'] is False and threshold['hindsight_tail_grouping'] is True
        counts['past_fit_thresholds'] += 1
        E = pd.Timestamp('2021-04-16T13:45:00Z') + pd.DateOffset(months=3*(f-5)); end = E + pd.DateOffset(months=3)
        assert (bars.index[T] + pd.Timedelta(minutes=375) < E-pd.DateOffset(months=6)).all()
        index = pd.date_range(E, end, freq='15min', inclusive='left')
        global_positions = bars.index.get_indexer(index); assert (global_positions >= 0).all()
        observed_open = open_values[global_positions]
        clock = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
        known_open = np.isfinite(observed_open) & (observed_open > 0)
        reference = None; per_fold = {'fold': f, 'fit_rows': len(fitted), 'threshold': q, 'means': {}}
        for h in HALVES:
            base = arrays(PARITY/'forecasts'/f'fold{f}_{h}.npz'); full = arrays(PARENT/'forecasts'/f'fold{f}_{h}_oracle_return.npz')
            assert set(base) == set(full) == KEYS
            exact(base['timestamps'], index.asi8, 'E calendar')
            if reference is not None:
                for k in KEYS-{'mu'}: exact(base[k], reference[k], 'shared parent '+k)
            reference = base
            I, O = base['inference_mask'], base['score_support']
            assert I.dtype == O.dtype == bool and not (O & ~I).any()
            assert not (I & ~(clock & known_open)).any()
            exact(I, masks[f]['inference'][global_positions], 'I'); exact(O, masks[f]['score'][global_positions], 'O')
            exact(base['actual'][O, 0], selected_returns(global_positions[O]), 'independent E labels')
            assert (index[O] + pd.Timedelta(minutes=375) <= end).all()
            y = base['actual'][:, 0]
            large = np.zeros(len(O), bool)
            for i in np.flatnonzero(O): large[i] = abs(float(y[i])) >= q
            subsets = {'all': O.copy(), 'fit_q90_large': large, 'fit_q90_other': O & ~large}
            assert threshold['subset_rows'] == {u: int(m.sum()) for u, m in subsets.items()}
            assert threshold['threshold_equal_scored_rows'] == sum(abs(float(y[i])) == q for i in np.flatnonzero(O))
            assert threshold['threshold_is_zero'] == (q == 0.)
            predictions = {'base': base, 'full': full}
            for component in ('sign', 'magnitude'):
                p = OUT/'forecasts'/f'fold{f}_{h}_oracle_{component}.npz'; a = arrays(p)
                assert set(a) == KEYS and str(p) in runtime_bindings
                for k in KEYS-{'mu'}: exact(a[k], base[k], 'unchanged new '+k)
                expected = scalar_substitute(base['mu'], base['actual'], O, component)
                exact(a['mu'], expected, 'scalar '+component)
                exact(a['mu'][~O], base['mu'][~O], 'unscored remainder')
                exact(np.isfinite(a['mu']), I, 'hybrid inference support')
                predictions[component] = a; counts['new_hybrid_forecasts'] += 1
                for rule in RULES:
                    cid = h+'_oracle_'+component+'_'+rule
                    target = arrays(OUT/'targets'/f'fold{f}_{cid}.npz')
                    exact(target['timestamps'], index.asi8, 'target calendar')
                    trace = read(OUT/'traces'/f'fold{f}_{cid}.json')
                    for key, value in {'future_information_used_for_decisions': True, 'hindsight_only': True,
                                       'deployable': False, 'teacher_use_allowed': False,
                                       'teacher_actions_used': False, 'global_optimum_claimed': False}.items():
                        assert trace[key] is value, (cid, key)
                    assert trace['information_swap'] == component
                    meta = trace['information_intervention']
                    assert meta['component'] == component and meta['variance_unchanged'] is True
                    assert meta['replacement_rows'] == int(O.sum()) and meta['learned_remainder_rows'] == int((I&~O).sum())
                    td = trace['decision_trace']; ids = np.asarray(td['bar_indices'], int)
                    assert len(set(ids.tolist())) == len(ids) and np.all(np.diff(ids) > 0)
                    expected_decisions = clock & known_open if rule == RULES[1] else I & known_open
                    exact(ids, np.flatnonzero(expected_decisions), 'trace decision support')
                    assert td['hindsight_information_replaced'] == O[ids].tolist()
                    for values in ('known_open_nav', 'known_open_exposure'):
                        assert len(td[values]) == len(ids) and all(math.isfinite(v) for v in td[values])
                    assert not np.isfinite(target['targets'][~expected_decisions]).any()
                    if rule == RULES[1]:
                        reasons = ['hybrid_hindsight' if O[i] else 'learned' if I[i] else 'forecast_unavailable' for i in ids]
                        assert td['reasons'] == reasons
                        for j, i in enumerate(ids):
                            v = target['targets'][i]
                            assert td['targets'][j] == (float(v) if math.isfinite(v) else None)
                            if not I[i]:
                                assert v == 1. and td['estimated_utility_gain_over_hold'][j] is None
                                assert td['estimated_trade_turnover'][j] is None
                    counts['new_hindsight_trace_supports'] += 1
            for k in KEYS-{'mu'}: exact(full[k], base[k], 'full endpoint '+k)
            exact(full['mu'][O], y[O], 'full actual endpoint'); exact(full['mu'][~O], base['mu'][~O], 'full remainder')
            for i in np.flatnonzero(O):
                a, b, s = float(y[i]), float(base['mu'][i]), float(predictions['sign']['mu'][i])
                gain = (a-b)**2 - (a-s)**2
                rhs = b*b if a == 0 else 4*abs(a*b) if a*b < 0 else 0.
                algebra_residual = max(algebra_residual, abs(gain-rhs))
                assert gain >= -TOLERANCES['sign_loss_identity_atol'] and abs(gain-rhs) <= TOLERANCES['sign_loss_identity_atol']
                algebra['rows'] += 1; algebra['actual_zero_rows'] += a == 0; algebra['base_zero_rows'] += b == 0
                algebra['opposite_nonzero_rows'] += a*b < 0
            denominator = math.fsum(float(y[i])**2 for i in np.flatnonzero(O))
            per_fold['means'][h] = {'inference_rows': int(I.sum()), 'score_rows': int(O.sum()),
                'unscored_inference_rows': int((I&~O).sum()), 'subset_mask_sha256': {u: mask_sha(index, m) for u,m in subsets.items()}}
            for cell, a in predictions.items():
                mid = h if cell == 'base' else h+'_oracle_'+('return' if cell == 'full' else cell)
                for subset, mask in subsets.items():
                    expected = scalar_scores(y[mask], a['mu'][mask], float(base['fit_return_mean']))
                    saved = scores[f,mid,subset]
                    assert saved['rows'] == expected['rows']
                    assert saved['hindsight_only'] is (cell != 'base')
                    assert saved['tail_grouping_uses_future_labels'] is (subset != 'all')
                    for k, v in expected.items():
                        if k != 'rows': close(v, saved[k], k, maxima, TOLERANCES['rank_ic_atol'] if k=='return_rank_ic' else TOLERANCES['score_atol'])
                    if cell == 'full' and expected['rows']: assert expected['return_mse'] == expected['return_mae'] == 0.
                    counts['subset_scores'] += 1
            for subset, mask in subsets.items():
                yy, mm = y[mask], base['mu'][mask]; n = len(yy)
                opposite = [scalar_sign(float(a))*scalar_sign(float(b)) < 0 for a,b in zip(yy,mm)]
                same = [scalar_sign(float(a))*scalar_sign(float(b)) > 0 for a,b in zip(yy,mm)]
                energy = math.fsum(float(a)**2 for a in yy)
                d = directions[f,h,subset]
                for k,v in {'rows':n,'actual_zero_rows':sum(float(a)==0 for a in yy),
                    'forecast_zero_rows':sum(float(a)==0 for a in mm),'nonzero_opposite_sign_rows':sum(opposite),
                    'nonzero_same_sign_rows':sum(same)}.items(): assert d[k] == v, k
                close(energy/denominator if denominator else None, d['actual_squared_return_share_of_all'], 'energy_share', maxima)
                close(math.fsum(float(a)**2 for a,o in zip(yy,opposite) if o)/energy if energy else None,
                      d['opposite_sign_actual_squared_return_share_of_subset'], 'opposite_energy_share', maxima)
                counts['direction_diagnostics'] += 1
        for cid in cfg['control_ids']:
            a = arrays(OUT/'targets'/f'fold{f}_{cid}.npz'); b = arrays(PARENT/'targets'/f'fold{f}_{cid}.npz')
            for k in a: exact(a[k],b[k],'old control target '+k)
            for k in ('base','stress_2x','regime','hindsight_only'): assert rows[f,cid][k] == old_rows[f,cid][k]
            counts['unchanged_parent_paths'] += 1
        records.append(per_fold)
    assert len(runtime_bindings) == 392 and counts['new_hybrid_forecasts'] == 32
    assert counts['past_fit_thresholds'] == 8 and counts['subset_scores'] == 192
    assert counts['new_hindsight_trace_supports'] == 64 and counts['unchanged_parent_paths'] == 224
    assert algebra['rows'] == 5148 and counts['direction_diagnostics'] == 48
    for key in ('selection_performed','teacher_use_allowed','additional_test_used','high_probability_generalization_established'):
        assert result[key] is False
    report = {'passed': True, 'scope': 'Independent registered Stage16 hybrid forecasts, fit-only thresholds, subset scores and trace support; no policy rollout',
        'script_sha256':sha(__file__),'registered_revision':revision,'source_bindings':cfg['source_bindings'],
        'direct_bindings':direct,'source_artifact_count':len(pre['source_artifact_bindings']),
        'source_artifact_inventory_sha256':digest(pre['source_artifact_bindings']),
        'runtime_artifact_count':len(runtime_bindings),'runtime_artifact_inventory_sha256':digest(runtime_bindings),
        'distinct_hashed_files':len(verified),'counts':dict(counts),'predeclared_tolerances':TOLERANCES,
        'max_absolute_difference':maxima,'exact_hybrid_risk_support_unscored_and_parent_arrays':True,
        'sign_loss_algebra':{**dict(algebra),'max_absolute_residual':algebra_residual,
            'identity':'base squared loss minus sign squared loss = 4 abs(y mu) for opposite nonzero signs; mu^2 when y=0; zero otherwise'},
        'full_return_mse_and_mae_zero_on_every_nonempty_subset':True,'folds':records,
        'endpoint_manifest_inventory':{'target_reproductions':64,'full_trace_reproductions':32,
            'runtime_assertions_only_not_independently_rerolled_here':True},
        'limitations':['This validates a future-information diagnostic, not deployable forecasts or a training teacher.',
            'Historical availability and reused development remain; this establishes no high-probability trend generalization.',
            'Old SHA-bound parity preparation supplies inherited masks and decodes the old full Spot parquet before slicing; selected labels here are separately checked using a cutoff-filtered Spot read. No later-period outcomes are scored.',
            'New path cash/units accounting is outside this audit; old path targets and saved metrics are compared exactly without rerolling.']}
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'path':str(REPORT),'sha256':sha(REPORT),'passed':True,'counts':dict(counts),'max_absolute_difference':maxima}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-after-root-go',action='store_true')
    parser.add_argument('--registered-revision'); parser.add_argument('--expected-results-sha256')
    args=parser.parse_args()
    if args.real_after_root_go:
        if not args.registered_revision or not args.expected_results_sha256:
            parser.error('Real mode requires registered revision and completed result SHA supplied by root.')
        real(args.registered_revision,args.expected_results_sha256)
    else: synthetic()
