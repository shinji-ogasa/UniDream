"""Causal statistical references and disjoint-set probability calibration."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import genpareto
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .wm_tail_transfer_evaluation import _baseline
from .wm_tail_transfer_models import build_specs


def _returns(data):
    close = np.asarray(data['ohlcspot'][:, 3], float)
    return np.r_[np.nan, np.diff(np.log(close))]


def _qlike(y, p):
    good = np.isfinite(y) & np.isfinite(p) & (p > 0)
    if not good.any(): return np.inf
    ratio = np.maximum(y[good], 1e-12) / p[good]
    return float(np.mean(ratio - np.log(ratio) - 1))


def _variance_paths(data, window=None, decay=None):
    r = _returns(data)
    square = r * r
    components = (square, np.where(np.isfinite(r), np.where(r > 0, square, 0), np.nan),
                  np.where(np.isfinite(r), np.where(r < 0, square, 0), np.nan))
    result = []
    for value in components:
        series = pd.Series(value)
        if window is not None:
            prediction = series.rolling(window, min_periods=window).mean().shift(1).to_numpy()
        else:
            prediction = series.ewm(alpha=1 - decay, adjust=False, min_periods=16).mean().shift(1).to_numpy()
        result.append(prediction)
    return result


def _fit_garch(r, asymmetry=False):
    from arch import arch_model
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        fit = arch_model(r * 100, mean='Zero', vol='GARCH', p=1, o=int(asymmetry), q=1,
                         dist='StudentsT' if asymmetry else 'normal', rescale=False).fit(
                             disp='off', update_freq=0, options={'maxiter': 500})
    params = {k: float(v) for k, v in fit.params.items()}
    standardized = np.asarray(fit.std_resid)
    standardized = standardized[np.isfinite(standardized)]
    return {'params': params, 'initial_variance_percent2': float(np.var(r * 100)),
            'standardized_residuals': standardized.tolist(), 'convergence_flag': int(fit.convergence_flag),
            'warnings': [str(v.message) for v in recorded], 'fit_observations': len(r)}


def fit_statistical_baselines(data, train_ids, val_ids, cfg):
    """Fit coefficients on Train; Val selects only trailing-window/decay.

    GARCH residual paths use common fixed innovations for all decision rows.
    EVT is fitted to both standardized residual tails; multi-period quantiles
    and ES come from simulated return paths, never sqrt(h) tail extrapolation.
    """
    train_ids, val_ids = np.asarray(train_ids, int), np.asarray(val_ids, int)
    if not len(train_ids) or not len(val_ids) or np.max(train_ids) >= np.min(val_ids):
        raise ValueError('chronological disjoint Train and Val references required')
    r = _returns(data)
    sample = r[train_ids]
    sample = sample[np.isfinite(sample)]
    if len(sample) < 256:
        raise ValueError('at least256 Train returns required')
    options = cfg.get('statistical_baselines', {})
    windows = list(options.get('variance_windows', [16, 96, 672]))
    decays = list(options.get('ewma_decays', [.94, .97, .99]))
    primary = cfg['targets']['horizons'].index(cfg['targets']['primary_horizon'])
    truth = np.asarray(data['targets']['rv'][val_ids, primary])
    horizon = cfg['targets']['primary_horizon']
    choices = {'past': [], 'ewma': []}
    for window in windows:
        v = _variance_paths(data, window=int(window))[0][val_ids] * horizon
        choices['past'].append({'value': int(window), 'validation_qlike': _qlike(truth, v)})
    for decay in decays:
        v = _variance_paths(data, decay=float(decay))[0][val_ids] * horizon
        choices['ewma'].append({'value': float(decay), 'validation_qlike': _qlike(truth, v)})
    fits, failures = {}, {}
    # arch does not accept missing observations. Fit the longest finite,
    # contiguous raw Train stretch; never compress a missing interval or a
    # caller's subsampling schedule into artificial adjacent returns.
    raw_train = np.arange(int(train_ids.min()), int(train_ids.max()) + 1)
    finite = np.isfinite(r[raw_train])
    boundaries = np.r_[0, np.flatnonzero(np.diff(finite.astype(int))) + 1, len(finite)]
    stretches = [raw_train[a:b] for a, b in zip(boundaries[:-1], boundaries[1:]) if finite[a]]
    fit_ids = max(stretches, key=len)
    fit_sample = r[fit_ids]
    for name, asymmetric in [('garch_gaussian', False), ('gjr_student_t', True)]:
        try:
            fit = _fit_garch(fit_sample, asymmetric)
            fit['contiguous_fit_first_id'] = int(fit_ids[0])
            fit['contiguous_fit_last_id'] = int(fit_ids[-1])
            if fit['convergence_flag'] != 0:
                raise RuntimeError(f"arch convergence flag {fit['convergence_flag']}")
            fits[name] = fit
        except Exception as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
    evt = {}
    if 'garch_gaussian' in fits:
        residuals = np.asarray(fits['garch_gaussian']['standardized_residuals'])
        for side in (-1, 1):
            positive = side * residuals
            threshold = float(np.quantile(positive, .975))
            excess = positive[positive > threshold] - threshold
            if len(excess) >= 30:
                shape, _, scale = genpareto.fit(excess, floc=0)
                # An ES requires finite mean. A failed tail is explicitly
                # absent; empirical residual fallback retains a different name.
                if np.isfinite(shape) and shape < 1 and np.isfinite(scale) and scale > 0:
                    evt[str(side)] = {'threshold': threshold, 'shape': float(shape), 'scale': float(scale),
                                      'probability': float(len(excess) / len(residuals)), 'n_exceedances': len(excess)}
    specs = build_specs(cfg)
    references = {s['name']: _baseline(data, train_ids, s).tolist() for s in specs}
    return {'complete': True, 'train_first_id': int(train_ids.min()), 'train_last_id': int(train_ids.max()),
            'train_ids': train_ids.tolist(), 'validation_first_id': int(val_ids.min()), 'validation_last_id': int(val_ids.max()),
            'selected_window': min(choices['past'], key=lambda v: v['validation_qlike'])['value'],
            'selected_decay': min(choices['ewma'], key=lambda v: v['validation_qlike'])['value'],
            'validation_candidates': choices, 'fits': fits, 'failures': failures, 'evt': evt,
            'references': references, 'mc_paths': int(options.get('monte_carlo_paths', 512)),
            'mc_seed': int(options.get('seed', 20261112)), 'residual_block_length': int(options.get('residual_block_length', 4)),
            'evt_scope': 'standardized_two_tail_GPD_returns_MC;not_sqrt_h_quantiles',
            'missing_filter_returns': 'conditional_expected_square_update_not_zero_return',
            'source': 'Train_only_coefficients;Val_only_window_decay_selection'}


def _filtered_variance(data, fit):
    r = _returns(data) * 100
    p = fit['params']; omega = p['omega']; alpha = p.get('alpha[1]', 0)
    beta, gamma = p.get('beta[1]', 0), p.get('gamma[1]', 0)
    variance = np.empty(len(r)); variance[0] = fit['initial_variance_percent2']
    for i in range(1, len(r)):
        if np.isfinite(r[i - 1]):
            variance[i] = omega + (alpha + gamma * (r[i - 1] < 0)) * r[i - 1] ** 2 + beta * variance[i - 1]
        else:
            variance[i] = omega + (alpha + beta + .5 * gamma) * variance[i - 1]
        variance[i] = min(max(variance[i], 1e-10), 1e12)
    return variance


def _innovation_bank(bundle, fit, mode, maxh):
    rng = np.random.default_rng(bundle['mc_seed'])
    count = bundle['mc_paths']
    if mode == 'gaussian':
        return rng.normal(size=(count, maxh))
    if mode == 'student':
        nu = fit['params'].get('nu', 8)
        return rng.standard_t(nu, size=(count, maxh)) * math.sqrt((nu - 2) / nu)
    residuals = np.asarray(fit['standardized_residuals'])
    block = bundle['residual_block_length']
    starts = rng.integers(0, len(residuals), size=(count, math.ceil(maxh / block)))
    positions = (starts[..., None] + np.arange(block)) % len(residuals)
    values = residuals[positions.reshape(count, -1)[:, :maxh]].copy()
    if mode == 'evt':
        for side in (-1, 1):
            tail = bundle['evt'][str(side)]
            chosen = side * values > tail['threshold']
            tail_draw = genpareto.ppf(rng.uniform(1e-9, 1 - 1e-9, size=int(chosen.sum())),
                                     tail['shape'], loc=tail['threshold'], scale=tail['scale'])
            values[chosen] = side * tail_draw
        # The same innovation bank is shared at every date, preserving common
        # MC noise. No Test return estimates this centering/scaling constant.
        values = (values - values.mean()) / max(values.std(), 1e-8)
    return values


def _garch_predictions(data, ids, fit, bundle, cfg, mode, filtered_variance=None):
    h = np.asarray(cfg['targets']['horizons'])
    levels = np.asarray(cfg['targets']['return_quantiles'])
    alphas = np.asarray(cfg['targets']['es_alphas'])
    initial = (_filtered_variance(data, fit) if filtered_variance is None else filtered_variance)[ids]
    p = fit['params']; alpha = p.get('alpha[1]', 0); beta = p.get('beta[1]', 0); gamma = p.get('gamma[1]', 0)
    persistence = alpha + beta + gamma / 2
    analytic = np.zeros((len(ids), len(h)))
    variance = initial.copy(); total = np.zeros(len(ids))
    for step in range(1, int(h.max()) + 1):
        total += variance / 10000
        if step in h: analytic[:, list(h).index(step)] = total
        variance = p['omega'] + persistence * variance
    result = {'rv': analytic, 'up_semivariance': analytic / 2, 'down_semivariance': analytic / 2}
    quantiles = np.empty((len(ids), len(h), len(levels)))
    vares = np.empty((len(ids), len(h), len(alphas), 2))
    bank = _innovation_bank(bundle, fit, mode, int(h.max()))
    for begin in range(0, len(ids), 64):
        stop = min(begin + 64, len(ids))
        v = np.broadcast_to(initial[begin:stop, None], (stop - begin, len(bank))).copy()
        cumulative = np.zeros_like(v)
        for step in range(1, int(h.max()) + 1):
            innovation = bank[:, step - 1]
            draw = np.sqrt(np.maximum(v, 1e-10)) * innovation[None, :]
            cumulative += draw / 100
            if step in h:
                hi = list(h).index(step)
                quantiles[begin:stop, hi] = np.quantile(cumulative, levels, axis=1).T
                for ai, a in enumerate(alphas):
                    q = np.minimum(np.quantile(cumulative, a, axis=1), -1e-12)
                    selected = cumulative <= q[:, None]
                    count = selected.sum(1)
                    es = np.divide(np.where(selected, cumulative, 0).sum(1), count,
                                   out=q.copy(), where=count > 0)
                    vares[begin:stop, hi, ai, 0] = q
                    vares[begin:stop, hi, ai, 1] = np.minimum(es, q - 1e-12)
            v = p['omega'] + (alpha + gamma * (draw < 0)) * draw ** 2 + beta * v
    result['return_quantiles'] = quantiles
    result['var_es'] = vares
    return result


def predict_statistical_baselines(data, ids, bundle, cfg):
    ids = np.asarray(ids, int)
    if len(ids) and ids.min() <= bundle['train_last_id']:
        raise ValueError('statistical out-of-sample forecasts require dates after fit Train')
    horizons = np.asarray(cfg['targets']['horizons'])
    prior = {k: np.broadcast_to(np.asarray(v), (len(ids),) + np.asarray(v).shape).copy()
             for k, v in bundle['references'].items()}
    for s in build_specs(cfg):
        if s['kind'] == 'variance': prior[s['name']] = np.maximum(prior[s['name']], 1e-12)
    result = {'zero_return': {'return': np.zeros((len(ids), len(horizons)))}, 'train_climatology': prior}
    for name, options in [('past_variance', {'window': bundle['selected_window']}),
                          ('ewma', {'decay': bundle['selected_decay']})]:
        values = _variance_paths(data, **options)
        result[name] = {head: np.maximum(value[ids, None] * horizons[None, :], 1e-12)
                        for head, value in zip(('rv', 'up_semivariance', 'down_semivariance'), values)}
    filtered = {name: _filtered_variance(data, fit) for name, fit in bundle['fits'].items()}
    for name, fit in bundle['fits'].items():
        result[name] = _garch_predictions(data, ids, fit, bundle, cfg, 'student' if name == 'gjr_student_t' else 'gaussian', filtered[name])
    if 'garch_gaussian' in bundle['fits']:
        fit = bundle['fits']['garch_gaussian']
        result['historical_residual'] = _garch_predictions(data, ids, fit, bundle, cfg, 'residual', filtered['garch_gaussian'])
        for head in ('rv', 'up_semivariance', 'down_semivariance'):
            result['historical_residual'].pop(head)
        if len(bundle['evt']) == 2:
            result['garch_evt'] = _garch_predictions(data, ids, fit, bundle, cfg, 'evt', filtered['garch_gaussian'])
            # EVT residuals need not have symmetric conditional second moments.
            # Only simulated distributional heads are claimed for this model.
            for head in ('rv', 'up_semivariance', 'down_semivariance'):
                result['garch_evt'].pop(head)
    return result


def _fit_binary_calibrator(y, p, method):
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {'kind': 'identity', 'reason': 'rare_or_constant_calibration_outcome', 'n': len(y)}
    if method == 'isotonic':
        fit = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        return {'kind': 'isotonic', 'x': fit.X_thresholds_.tolist(), 'y': fit.y_thresholds_.tolist(), 'n': len(y)}
    fit = LogisticRegression(C=1e3, solver='lbfgs', max_iter=1000).fit(logit(np.clip(p, 1e-7, 1 - 1e-7))[:, None], y)
    return {'kind': 'platt', 'slope': float(fit.coef_[0, 0]), 'intercept': float(fit.intercept_[0]), 'n': len(y)}


def _apply_binary(p, fit):
    if fit['kind'] == 'identity': return p
    if fit['kind'] == 'isotonic': return np.interp(p, fit['x'], fit['y'])
    return expit(fit['slope'] * logit(np.clip(p, 1e-7, 1 - 1e-7)) + fit['intercept'])


def fit_calibrator(data, ids, predictions, specs, cfg=None, method='parametric'):
    """Fit only on caller-supplied disjoint calibration rows, never Test."""
    if method not in ('none', 'parametric', 'isotonic'):
        raise ValueError('unknown probability calibration method')
    ids = np.asarray(ids, int)
    bundle = {'method': method, 'fit_first_id': int(ids.min()) if len(ids) else None,
              'fit_last_id': int(ids.max()) if len(ids) else None, 'heads': {}}
    if method == 'none': return bundle
    for spec in specs:
        name, kind = spec['name'], spec['kind']
        if name not in predictions or kind not in ('binary', 'categorical', 'hazard'):
            continue
        y = np.asarray(data['targets'][name][ids]); m = np.asarray(data['masks'][name][ids], bool)
        p = np.asarray(predictions[name])
        cells = []
        for coord in np.ndindex(tuple(spec['shape'])):
            s = (slice(None),) + coord; valid = m[s] & np.isfinite(y[s])
            yy, pp = y[s][valid], p[s][valid]
            if kind == 'binary':
                fit = _fit_binary_calibrator(yy, pp, method)
            elif len(yy) < 20 or len(np.unique(yy)) < 2:
                fit = {'kind': 'identity', 'reason': 'rare_or_constant_calibration_outcome', 'n': len(yy)}
            elif method == 'isotonic':
                fit = {'kind': 'multiclass_isotonic', 'classes': [
                    _fit_binary_calibrator((yy == c).astype(float), pp[:, c], method) for c in range(pp.shape[-1])]}
            else:
                logs = np.log(np.clip(pp, 1e-12, 1))
                def objective(log_temperature):
                    logits = logs / math.exp(log_temperature)
                    logits -= logits.max(1, keepdims=True)
                    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
                    return float(-np.log(np.clip(probs[np.arange(len(yy)), yy.astype(int)], 1e-12, 1)).mean())
                fit = {'kind': 'temperature', 'temperature': math.exp(float(minimize_scalar(
                    objective, bounds=(-3, 3), method='bounded').x)), 'n': len(yy)}
            cells.append({'coordinate': list(coord), **fit})
        bundle['heads'][name] = {'kind': kind, 'cells': cells}
    return bundle


def apply_calibrator(predictions, bundle, specs=None):
    result = {k: np.asarray(v).copy() for k, v in predictions.items()}
    for name, head in bundle.get('heads', {}).items():
        if name not in result: continue
        for fit in head['cells']:
            s = (slice(None),) + tuple(fit['coordinate']); p = result[name][s]
            if fit['kind'] == 'identity': continue
            if head['kind'] == 'binary':
                result[name][s] = _apply_binary(p, fit)
            elif fit['kind'] == 'temperature':
                logits = np.log(np.clip(p, 1e-12, 1)) / fit['temperature']
                logits -= logits.max(-1, keepdims=True)
                probs = np.exp(logits); result[name][s] = probs / probs.sum(-1, keepdims=True)
            else:
                probs = np.column_stack([_apply_binary(p[:, c], value) for c, value in enumerate(fit['classes'])])
                denom = probs.sum(-1, keepdims=True)
                result[name][s] = np.divide(probs, denom, out=p.copy(), where=denom > 0)
    return result
