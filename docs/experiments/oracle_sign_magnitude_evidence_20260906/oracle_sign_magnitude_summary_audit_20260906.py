"""Independent Stage16 saved-summary audit. No fitting or policy construction.
Prepare before freeze; execute only after root authorizes completed results.
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
OUT = ROOT / 'codex_outputs/oracle_sign_magnitude_decisions_v1'
PARENT = ROOT / 'codex_outputs/oracle_information_decomposition_v1'
PAR = ROOT / 'codex_outputs/oracle_frozen_procedure_parity_v1'
REPORT = Path('/tmp/oracle_sign_magnitude_summary_audit_20260906.json')
FS = tuple(range(5, 13))
HALVES = ('technical_half', 'perp_delay0_half')
RULES = ('utility_risk1', 'utility_risk1_fallback_bh')
COSTS = ('base', 'stress_2x')
STRATA = ('all', 'bull', 'bear', 'sideways')
SUBSETS = ('all', 'fit_q90_large', 'fit_q90_other')
CELLS = {h: {'base': h, 'sign': h+'_oracle_sign', 'magnitude': h+'_oracle_magnitude',
             'full': h+'_oracle_return'} for h in HALVES}
MEANS = tuple(m for cells in CELLS.values() for m in cells.values())
BASE_MEANS = ('scale_mean', 'technical_scaled', 'perp_delay0_scaled', 'technical_half', 'perp_delay0_half')
CAUSAL = ('bh', 'common_robust') + tuple(m+'_'+r for m in BASE_MEANS for r in RULES)
OLD_ORACLES = tuple(h+'_oracle_'+s+'_'+r for h in HALVES
                    for s in ('return', 'realized_risk', 'both') for r in RULES) + tuple(
                    'matched_rl_beam32_'+r+'_risk'+str(k) for r in ('hold', 'fallback_bh') for k in (0, 1))
CONTROLS = CAUSAL + OLD_ORACLES
NEW_MEANS = tuple(CELLS[h][c] for h in HALVES for c in ('sign', 'magnitude'))
NEW_IDS = tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
IDS = CONTROLS + NEW_IDS
CONTRASTS = (('sign','base'), ('magnitude','base'), ('full','base'), ('full','sign'), ('full','magnitude'))
EK = ('alpha_ex', 'maxdd_delta', 'turnover', 'trades')
PK = ('return_mse', 'return_mae', 'return_sign_accuracy', 'return_rank_ic')
ALL_SCORE_KEYS = ('return_mse', 'return_mae', 'return_sign_accuracy', 'zero_return_mse', 'fit_mean_return_mse', 'return_rank_ic')
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
    return sum((D(x) for x in values), Decimal(0)) / Decimal(len(values)) if values and all(x is not None for x in values) else None


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
    assert n > 0 and np.isfinite(actual).all() and np.isfinite(prediction).all()
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
        result = read(bind(OUT/'results.json'))
        reg = read(bind(OUT/'registration.json'))
        cfg = reg['config']
        bind(ROOT/'configs/oracle_sign_magnitude_decisions_20260906.yaml', reg['config_sha256'], 'config')
        pre = read(bind(OUT/'preflight.json', cfg['preflight_sha256']))
        assert result['registration_sha256'] == canonical(reg)
        assert reg['preflight_sha256'] == cfg['preflight_sha256']
        assert pre['source_bindings'] == cfg['source_bindings']
        assert len(cfg['source_bindings']) == 25
        assert pre['config_contract_sha256'] == canonical({k:v for k,v in cfg.items() if k!='preflight_sha256'})
        for p,h in cfg['source_bindings'].items(): bind(p,h,'registered_source')
        for p,h in pre['direct_source_bindings'].items(): bind(p,h,'preflight_direct')
        for p,h in cfg['parent_manifest_bindings'].items(): bind(p,h,'parent_manifest')
        for key in ('parent','parity'): bind(cfg[key+'_config'],cfg[key+'_config_sha256'],'parent_config')
        ancestors = pre['source_artifact_bindings']
        assert len(ancestors) == 1728
        assert len({str((ROOT/p).resolve()) for p in ancestors}) == 1728
        for p,h in ancestors.items(): bind(p,h,'ancestor_artifact')
        parent = read(PARENT/'results.json')
        parent_reg = read(PARENT/'registration.json')
        assert parent_reg['source_revision'] == cfg['parent_source_revision']
        assert parent['registration_sha256'] == canonical(parent_reg)
        rows, scores = result['rows'], result['scores']
        ri = {(r['fold'],r['candidate_id']):r for r in rows}
        si = {(s['fold'],s['mean_id'],s['subset']):s for s in scores}
        assert len(rows) == len(ri) == 288 and set(ri) == {(f,p) for f in FS for p in IDS}
        assert len(scores) == len(si) == 192 and set(si) == {(f,m,u) for f in FS for m in MEANS for u in SUBSETS}
        ti = {t['fold']:t for t in result['thresholds']}
        di = {(d['fold'],d['mean_id'],d['subset']):d for d in result['direction_diagnostics']}
        assert len(result['thresholds']) == len(ti) == 8 and set(ti) == set(FS)
        assert len(result['direction_diagnostics']) == len(di) == 48
        assert set(di) == {(f,h,u) for f in FS for h in HALVES for u in SUBSETS}
        epi = {(x['fold'],x['candidate_id']):x for x in result['endpoint_parity']}
        endpoint_ids = tuple(CELLS[h][cell]+'_'+r for h in HALVES for cell in ('base','full') for r in RULES)
        assert len(result['endpoint_parity']) == len(epi) == 64
        assert set(epi) == {(f,p) for f in FS for p in endpoint_ids}
        for (f,cid),value in epi.items():
            assert value['targets_exact'] is True
            assert value['full_decision_trace_matches'] is ('_oracle_return_' in cid)
        for k,v in {'new_model_fits':0,'fit_distribution_thresholds':8,'new_causal_names':0,
                    'adaptive_causal_names_unchanged':174,'new_hindsight_policy_names':8}.items():
            assert result[k] == v
        for k in ('selection_performed','teacher_use_allowed','additional_test_used','high_probability_generalization_established'):
            assert result[k] is False
        regimes = {f:ri[f,'bh']['regime'] for f in FS}
        counts = dict(Counter(v['trend'] for v in regimes.values()))
        assert counts == {'bull':2,'bear':4,'sideways':2}
        for r in rows:
            assert r['regime'] == regimes[r['fold']]
            assert r['hindsight_only'] is (r['candidate_id'] in OLD_ORACLES+NEW_IDS)
            for cost in COSTS:
                assert all(math.isfinite(r[cost][k]) for k in EK)
        for s in scores:
            assert s['regime'] == regimes[s['fold']]
            assert s['hindsight_only'] is (s['mean_id'] not in HALVES)
            assert s['tail_grouping_uses_future_labels'] is (s['subset']!='all')
            assert type(s['rows']) is int and s['rows'] >= 0
            assert s['rows'] == si[s['fold'],HALVES[0],s['subset']]['rows']
            if s['rows'] == 0:
                assert all(s[k] is None for k in ALL_SCORE_KEYS)
            else:
                assert all(s[k] is not None and math.isfinite(s[k]) and s[k]>=0 for k in
                           ('return_mse','return_mae','zero_return_mse','fit_mean_return_mse'))
                assert 0 <= s['return_sign_accuracy'] <= 1
                assert s['return_rank_ic'] is None or -1.000000000001 <= s['return_rank_ic'] <= 1.000000000001
        for f in FS:
            assert si[f,HALVES[0],'all']['rows'] >= 16
            for m in MEANS:
                assert si[f,m,'all']['rows'] == si[f,m,'fit_q90_large']['rows'] + si[f,m,'fit_q90_other']['rows']
        artifacts = {}
        for f in FS:
            fold = read(bind(OUT/f'fold_{f}.json',category='fold_manifest'))
            assert fold['registration_sha256'] == result['registration_sha256']
            for key in ('rows','scores','direction_diagnostics','endpoint_parity'):
                assert fold[key] == [x for x in result[key] if x['fold']==f]
            assert fold['threshold'] == ti[f]
            expected = {str((OUT/'targets'/f'fold{f}_{p}.npz').relative_to(ROOT)) for p in IDS}
            expected |= {str((OUT/'forecasts'/f'fold{f}_{m}.npz').relative_to(ROOT)) for m in NEW_MEANS}
            expected |= {str((OUT/'traces'/f'fold{f}_{p}.json').relative_to(ROOT)) for p in NEW_IDS}
            expected.add(str((OUT/'thresholds'/f'fold{f}_fit_q90.json').relative_to(ROOT)))
            assert len(expected) == 49 and set(fold['artifact_sha256']) == expected
            for p,h in fold['artifact_sha256'].items():
                assert p not in artifacts
                artifacts[p] = h
                bind(p,h,'new_artifact')
            assert read(OUT/'thresholds'/f'fold{f}_fit_q90.json') == ti[f]
            pf = next(x for x in pre['support'] if x['fold']==f)
            assert ti[f]['fit_rows'] == pf['fit_rows'] and ti[f]['fit_return_sha256'] == pf['fit_return_sha256']
            assert ti[f]['quantile'] == .9 and ti[f]['method'] == 'linear'
            assert ti[f]['hindsight_tail_grouping'] is True and ti[f]['used_for_orders'] is False
            assert math.isfinite(ti[f]['threshold']) and ti[f]['threshold'] >= 0
        assert len(artifacts) == 392
        for r in rows:
            p = str((OUT/'targets'/f"fold{r['fold']}_{r['candidate_id']}.npz").relative_to(ROOT))
            assert r['targets_sha256'] == artifacts[p]
            if r['candidate_id'] in NEW_IDS:
                p = str((OUT/'traces'/f"fold{r['fold']}_{r['candidate_id']}.json").relative_to(ROOT))
                assert r['diagnostic_sha256'] == artifacts[p]
        econ, pred, paired, interactions = {}, {}, {}, {}
        for stratum in STRATA:
            fs = [f for f in FS if stratum=='all' or regimes[f]['trend']==stratum]
            econ[stratum] = {p:{'quarters':len(fs),'hindsight_only':p in OLD_ORACLES+NEW_IDS,
                'joint_positive_quarters_both_costs':sum(all(ri[f,p][c]['alpha_ex']>0 and ri[f,p][c]['maxdd_delta']<0 for c in COSTS) for f in fs),
                **{c:{k:mean([ri[f,p][c][k] for f in fs]) for k in EK} for c in COSTS}} for p in IDS}
            pred[stratum] = {}
            for m in MEANS:
                pred[stratum][m] = {}
                for u in SUBSETS:
                    ss = [si[f,m,u] for f in fs]
                    n = sum(s['rows'] for s in ss)
                    pred[stratum][m][u] = {'quarters':len(fs),'nonempty_quarters':sum(s['rows']>0 for s in ss),
                        'defined_rank_quarters':sum(s['return_rank_ic'] is not None for s in ss),'rows':n,
                        **{k:mean([s[k] for s in ss]) for k in PK},
                        'pooled_row_mse':sum(D(s['return_mse'])*Decimal(s['rows']) for s in ss if s['rows'])/Decimal(n) if n else None}
            paired[stratum], interactions[stratum] = {}, {}
            for h,cells in CELLS.items():
                paired[stratum][h] = {}
                for a,b in CONTRASTS:
                    mse = {}
                    for u in SUBSETS:
                        ds = [None if si[f,cells[a],u]['return_mse'] is None else
                              D(si[f,cells[a],u]['return_mse'])-D(si[f,cells[b],u]['return_mse']) for f in fs]
                        mse[u] = mean(ds)
                    paired[stratum][h][a+'_minus_'+b] = {
                        'economics':{r:{c:{k:mean([D(ri[f,cells[a]+'_'+r][c][k])-D(ri[f,cells[b]+'_'+r][c][k]) for f in fs])
                                          for k in EK} for c in COSTS} for r in RULES},'mse':mse}
                mse = {}
                for u in SUBSETS:
                    values = [None if si[f,cells['base'],u]['return_mse'] is None else sum(
                        D(si[f,cells[cell],u]['return_mse'])*sg for cell,sg in
                        (('full',1),('sign',-1),('magnitude',-1),('base',1))) for f in fs]
                    mse[u] = mean(values)
                interactions[stratum][h] = {
                    'economics':{r:{c:{k:mean([sum(D(ri[f,cells[cell]+'_'+r][c][k])*sg for cell,sg in
                        (('full',1),('sign',-1),('magnitude',-1),('base',1))) for f in fs]) for k in EK} for c in COSTS} for r in RULES},
                    'mse':mse}
        summary = {'economics':econ,'prediction':pred,'paired':paired,'interaction':interactions,
            'selection_performed':False,'teacher_use_allowed':False,'high_probability_generalization_established':False,
            'hindsight_diagnostic_not_causal_accuracy':True,'adaptive_causal_names_unchanged':174}
        # Root may add the requested explicit regime inventory before freeze;
        # either schema is independently validated rather than trusting its counts.
        if 'regime_counts' in result['summary']: summary['regime_counts'] = counts
        assert set(summary) == set(result['summary'])
        for k,v in summary.items(): compare(v,result['summary'][k],k,'summary_'+k)
        cache = {}
        def npz(path):
            key = str(path.relative_to(ROOT))
            assert key in BINDINGS, (key,'consumed artifact is not bound')
            if key not in cache:
                with np.load(path,allow_pickle=False) as z:
                    cache[key] = {k:z[k] for k in z.files}
            return cache[key]
        parent_ri = {(r['fold'],r['candidate_id']):r for r in parent['rows']}
        rescored = copied_controls = new_forecasts = trace_labels = 0
        direction_checks, rank_checks, algebra_checks, threshold_checks = [], [], [], []
        for f in FS:
            first = npz(PAR/'forecasts'/f'fold{f}_{HALVES[0]}.npz')
            sc = first['score_support']
            yy = first['actual'][:,0]
            q = ti[f]['threshold']
            assert sc.dtype == np.dtype(bool) and np.isfinite(yy[sc]).all()
            large = np.zeros(len(sc),bool)
            large[sc] = np.abs(yy[sc]) >= q
            subset_masks = {'all':sc,'fit_q90_large':large,'fit_q90_other':sc & ~large}
            assert ti[f]['subset_rows'] == {u:int(v.sum()) for u,v in subset_masks.items()}
            assert ti[f]['threshold_equal_scored_rows'] == int(np.sum(np.abs(yy[sc])==q))
            assert ti[f]['threshold_is_zero'] is (q==0)
            threshold_checks.append({'fold':f,'saved_fit_q90':q,'fit_rows':ti[f]['fit_rows'],
                'fit_return_sha256':ti[f]['fit_return_sha256'],'subset_rows':ti[f]['subset_rows'],
                'equality_rows':ti[f]['threshold_equal_scored_rows'],
                'fit_quantile_recomputed_here':False,'tail_membership_recomputed_from_saved_scored_actual':True})
            for h,cells in CELLS.items():
                original = npz(PAR/'forecasts'/f'fold{f}_{h}.npz')
                for key in first:
                    if key!='mu': array_compare(original[key],first[key],f'{f}/{h}/shared/{key}')
                streams = {'base':original,
                    'full':npz(PARENT/'forecasts'/f'fold{f}_{cells["full"]}.npz')}
                for component in ('sign','magnitude'):
                    streams[component] = npz(OUT/'forecasts'/f'fold{f}_{cells[component]}.npz')
                    new_forecasts += 1
                for cell,z in streams.items():
                    assert set(z) == set(original)
                    for key in original:
                        if key!='mu': array_compare(z[key],original[key],f'{f}/{h}/{cell}/{key}')
                    expected = original['mu'].copy()
                    if cell=='sign':
                        expected[sc] = np.asarray([math.copysign(1.,float(y))*abs(float(m)) if y!=0 else 0.
                            for y,m in zip(yy[sc],original['mu'][sc])])
                    elif cell=='magnitude':
                        expected[sc] = np.asarray([math.copysign(1.,float(m))*abs(float(y)) if m!=0 else 0.
                            for y,m in zip(yy[sc],original['mu'][sc])])
                    elif cell=='full': expected[sc] = yy[sc]
                    array_compare(z['mu'],expected,f'{f}/{h}/{cell}/substitution')
                    assert np.array_equal(np.isfinite(z['mu']),original['inference_mask'])
                    for subset,mask in subset_masks.items():
                        measured = scalar_scores(yy[mask],z['mu'][mask],float(original['fit_return_mean'])) if mask.any() else {
                            'rows':0,**{k:None for k in ALL_SCORE_KEYS}}
                        saved = si[f,cells[cell],subset]
                        compare(measured,{k:saved[k] for k in measured},f'{f}/{h}/{cell}/{subset}/score','scalar_rescore')
                        rescored += 1
                        rank_checks.append({'fold':f,'mean_id':cells[cell],'subset':subset,
                            'rows':measured['rows'],'unique_predictions':len(np.unique(z['mu'][mask])),
                            'rank_defined':measured['return_rank_ic'] is not None})
                base = original['mu']
                all_energy = math.fsum(float(y)**2 for y in yy[sc])
                for subset,mask in subset_masks.items():
                    yv,mv = yy[mask],base[mask]
                    opposite = np.asarray([float(y)*float(m)<0 for y,m in zip(yv,mv)],bool)
                    same = np.asarray([(y>0 and m>0) or (y<0 and m<0) for y,m in zip(yv,mv)],bool)
                    energy = math.fsum(float(y)**2 for y in yv)
                    wrong_energy = math.fsum(float(y)**2 for y in yv[opposite])
                    measured = {'fold':f,'mean_id':h,'subset':subset,'rows':len(yv),
                        'actual_zero_rows':sum(float(y)==0 for y in yv),
                        'forecast_zero_rows':sum(float(m)==0 for m in mv),
                        'nonzero_opposite_sign_rows':int(opposite.sum()),'nonzero_same_sign_rows':int(same.sum()),
                        'actual_squared_return_share_of_all':energy/all_energy if all_energy else None,
                        'opposite_sign_actual_squared_return_share_of_subset':wrong_energy/energy if energy else None}
                    compare(measured,di[f,h,subset],f'{f}/{h}/{subset}/direction','direction_diagnostics')
                    direction_checks.append(measured)
                    if len(yv):
                        gain = math.fsum(4*abs(float(y)*float(m)) if y*m<0 else float(m)**2 if y==0 else 0.
                                         for y,m in zip(yv,mv))/len(yv)
                        observed_gain = D(si[f,cells['base'],subset]['return_mse'])-D(si[f,cells['sign'],subset]['return_mse'])
                        compare(gain,observed_gain,f'{f}/{h}/{subset}/sign_MSE_identity','sign_mse_identity')
                        assert gain >= 0 and si[f,cells['full'],subset]['return_mse'] == 0.
                        algebra_checks.append({'fold':f,'mean_id':h,'subset':subset,
                            'sign_mse_reduction_from_identity':gain,'full_scored_mse_is_zero':True,
                            'these_are_algebraic_checks_not_predictive_skill':True})
                for component in ('sign','magnitude'):
                    for rule in RULES:
                        cid = cells[component]+'_'+rule
                        trace = read(OUT/'traces'/f'fold{f}_{cid}.json')
                        assert trace['hindsight_only'] is True and trace['future_information_used_for_decisions'] is True
                        assert trace['deployable'] is False and trace['teacher_use_allowed'] is False
                        assert trace['information_swap'] == component
                        meta = trace['information_intervention']
                        assert meta['component'] == component and meta['variance_unchanged'] is True
                        assert meta['replacement_rows'] == int(sc.sum())
                        assert meta['learned_remainder_rows'] == int((original['inference_mask'] & ~sc).sum())
                        assert meta['global_optimum_claimed'] is False
                        decision = trace['decision_trace']
                        indices = np.asarray(decision['bar_indices'],int)
                        assert decision['hindsight_information_replaced'] == [bool(sc[i]) for i in indices]
                        if 'reasons' in decision:
                            assert all(reason=='hybrid_hindsight' for i,reason in zip(indices,decision['reasons']) if sc[i])
                        trace_labels += 1
            for cid in CONTROLS:
                arrays_equal(npz(OUT/'targets'/f'fold{f}_{cid}.npz'),
                             npz(PARENT/'targets'/f'fold{f}_{cid}.npz'),f'{f}/{cid}/control_targets')
                for cost in COSTS:
                    compare(ri[f,cid][cost],parent_ri[f,cid][cost],f'{f}/{cid}/{cost}','unchanged_control_accounts')
                copied_controls += 1
        assert rescored==192 and copied_controls==224 and new_forecasts==32 and trace_labels==64
        assert len(direction_checks)==48 and len(threshold_checks)==8
        assert sum(t['subset_rows']['all'] for t in threshold_checks)==2574
        assert all(v<=Decimal('1e-12') for v in MAXDIFF.values()),plain(MAXDIFF)
        tiny_dd = [{'fold':r['fold'],'candidate_id':r['candidate_id'],'cost':c,
                    'maxdd_delta':r[c]['maxdd_delta']} for r in rows for c in COSTS
                   if 0 < abs(r[c]['maxdd_delta']) < 1e-12]
        diagnostic_economic_signs = {p:{
            'all_strata_both_costs_alpha_positive_dd_negative':all(econ[g][p][c]['alpha_ex']>0 and econ[g][p][c]['maxdd_delta']<0 for g in STRATA for c in COSTS),
            'joint_quarters_both_costs':econ['all'][p]['joint_positive_quarters_both_costs'],
            'causal_model_promotion_permitted':False} for p in NEW_IDS}
        report = {'schema':'independent-sign-magnitude-summary-audit-v1','passed':True,
            'scope':'Original reused-development saved artifacts only; independent Decimal60 summaries, scalar192score/rank checks,32 substitutions,48 direction diagnostics and fixed-q90 memberships. No canonical experiment helper/scorer/summary/planner, raw market loader, model fit or new policy.',
            'source_revision':reg['source_revision'],
            'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
            'inventory':{'economic_rows':288,'cost_account_summaries':576,'policies':36,
                'causal_control_rows':96,'hindsight_rows':192,'score_records':192,'mean_streams':8,
                'direction_records':48,'threshold_records':8,'new_artifacts':392,'ancestor_artifacts':1728,
                'registered_sources':25,'copied_control_rows':224,'copied_control_accounts':448,
                'registered_endpoint_receipts':64,'full_trace_parity_receipts':32,
                'new_forecasts_checked':32,'new_hindsight_trace_labels_checked':64,'regime_counts':counts},
            'verified_binding_counts':dict(CATEGORIES),'source_sha256':BINDINGS,
            'binding_scope':'All enumerated registered sources, direct preflight files, completed parent manifests, ancestor artifacts and new fold artifacts independently rehashed.',
            'numeric_max_absolute_differences':MAXDIFF,'maximum_difference_locations':LOCATIONS,
            'economics':econ,'prediction':pred,'paired':paired,'interaction':interactions,
            'direction_diagnostics':direction_checks,'threshold_checks':threshold_checks,
            'rank_checks':rank_checks,'algebraic_checks':algebra_checks,
            'diagnostic_economic_signs':diagnostic_economic_signs,'tiny_nonzero_dd_values':tiny_dd,
            'limitations':[
                'Partial/full cells use future information and cannot be causal predictors or teachers; base cells are unchanged causal controls without new independent confirmation.',
                'Sign-only squared-error reduction and full zero MSE are algebraic consequences, not evidence of learnable causal signal.',
                'abs(mu) is not an estimated conditional absolute return; replacing it by realized abs(y) is not a conditional-variance intervention.',
                'Tail membership uses future scored labels and never selects decisions or attributes dollar PnL.',
                'Saved fit-q90 values and fit-vector hashes are bound here; independent raw fit-quantile recomputation belongs to the separate source/substitution audit.',
                'Equal-quarter subgroup aggregates preserve all constituent quarters or are null. Pooled losses have a different explicit row denominator.',
                'Economic interaction is a symmetric metric-specific contrast, not an additive causal allocation or globally optimal policy bound.',
                '64 endpoint receipts and448 copied accounts are checked here; independent own-state/account reconstruction is a separate audit.',
                'Repeated8 quarters, overlapping fit/evaluation histories and2/4/2 regimes cannot establish high-probability trend robustness.',
                'No new model, window, teacher, variant selection, p-value, confidence interval, heldout access or promotion.']}
        REPORT.write_text(json.dumps(plain(report),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        print(json.dumps({'output':str(REPORT),'sha256':sha(REPORT),'passed':True,
            'inventory':report['inventory'],'bindings':dict(CATEGORIES),'maxdiff':plain(MAXDIFF),
            'diagnostic_economic_signs':diagnostic_economic_signs},ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run only after root authorizes completed Stage16 results.')
    parser.add_argument('--execute-saved-audit',action='store_true',required=True)
    parser.parse_args()
    audit()
