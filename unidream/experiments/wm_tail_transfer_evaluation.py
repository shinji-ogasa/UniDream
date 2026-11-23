"""Masked, Train-referenced evaluation of registered tail-transfer outputs.

No threshold, calibrator or selector is fit on evaluation outcomes. Constant
targets retain sample/event counts while undefined skill and correlations are
NaN. Per-row losses retain the market timestamp as the inference unit.
"""
from __future__ import annotations

from collections import defaultdict
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-12


def _mean(values, mask, axis=None):
    values = np.asarray(values, float)
    mask = np.broadcast_to(np.asarray(mask, bool), values.shape) & np.isfinite(values)
    total = np.where(mask, values, 0).sum(axis=axis)
    n = mask.sum(axis=axis)
    return np.divide(total, n, out=np.full(np.shape(total), np.nan, float), where=n > 0)


def _baseline(data, train_ids, spec):
    shape = tuple(spec['shape']); kind = spec['kind']
    extra = (int(spec.get('classes', 3)),) if kind in ('categorical', 'hazard') else ((2,) if kind == 'var_es' else ())
    result = np.full(shape + extra, np.nan)
    if train_ids is None or not len(train_ids):
        return result
    y = np.asarray(data['targets'][spec['name']][train_ids], float)
    m = np.asarray(data['masks'][spec['name']][train_ids], bool) & np.isfinite(y)
    for coord in np.ndindex(shape):
        value = y[(slice(None),) + coord][m[(slice(None),) + coord]]
        if not len(value):
            continue
        if kind in ('categorical', 'hazard'):
            result[coord] = np.bincount(value.astype(int), minlength=extra[0])[:extra[0]] / len(value)
        elif kind in ('quantile', 'positive_quantile'):
            result[coord] = np.quantile(value, float(spec['levels'][coord[-1]]))
        elif kind == 'var_es':
            q = min(float(np.quantile(value, spec['levels'][coord[-1]])), -EPS)
            tail = value[value <= q]
            e = min(float(tail.mean()) if len(tail) else q, q - EPS)
            result[coord] = [q, e]
        elif spec['name'] == 'return':
            result[coord] = 0  # Explicit zero-return benchmark for OOS R².
        else:
            result[coord] = value.mean()
    return result


def _scale(data, train_ids, spec):
    shape = tuple(spec['shape']); kind = spec['kind']
    if train_ids is None or not len(train_ids) or kind in ('binary', 'categorical', 'hazard', 'variance'):
        return np.ones(shape)
    y = np.asarray(data['targets'][spec['name']][train_ids], float)
    m = np.asarray(data['masks'][spec['name']][train_ids], bool)
    scale = np.sqrt(_mean(y * y, m, axis=0))
    return np.maximum(np.where(np.isfinite(scale), scale, 1), 1e-6)


def _loss(y, p, spec):
    kind = spec['kind']
    if kind in ('categorical', 'hazard'):
        safe = np.where(np.isfinite(y), y, 0).astype(int)
        return -np.log(np.take_along_axis(np.clip(p, EPS, 1), safe[..., None], axis=-1)[..., 0])
    if kind == 'binary':
        return (p - y) ** 2
    if kind == 'variance':
        ratio = np.maximum(y, EPS) / np.maximum(p, EPS)
        return ratio - np.log(ratio) - 1
    if kind in ('quantile', 'positive_quantile'):
        q = np.asarray(spec['levels'])
        err = y - p
        return np.maximum(q * err, (q - 1) * err)
    if kind == 'var_es':
        q, e = p[..., 0], p[..., 1]
        alpha = np.asarray(spec['levels'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = -(y <= q).astype(float) * (q - y) / (alpha * e) + q / e + np.log(-e) - 1
        result[(e >= 0) | (q >= 0) | (e > q)] = np.nan
        return result
    return (p - y) ** 2


def _head(data, ids, prediction, spec, train_ids):
    name, kind = spec['name'], spec['kind']
    y = np.asarray(data['targets'][name][ids], float)
    mask = np.asarray(data['masks'][name][ids], bool) & np.isfinite(y)
    p = np.asarray(prediction, float)
    expected = y.shape + ((int(spec.get('classes', 3)),) if kind in ('categorical', 'hazard')
                          else ((2,) if kind == 'var_es' else ()))
    if p.shape != expected:
        raise ValueError(f'{name} prediction shape {p.shape}, expected {expected}')
    if kind in ('categorical', 'hazard', 'var_es'):
        finite = np.isfinite(p).all(axis=-1)
    else:
        finite = np.isfinite(p)
    if np.any(mask & ~finite):
        raise ValueError(f'{name} has a missing prediction at an observed target')
    if kind in ('categorical', 'hazard'):
        if np.any(mask & ((p < 0).any(axis=-1) | (p > 1).any(axis=-1) | (np.abs(p.sum(-1) - 1) > 1e-4))):
            raise ValueError('categorical predictions must be probabilities summing to1')
    if kind == 'binary' and np.any(mask & ((p < 0) | (p > 1))):
        raise ValueError('binary probabilities outside[0,1]')
    if kind == 'variance' and np.any(mask & (p <= 0)):
        raise ValueError('variance predictions must be positive')
    if kind == 'var_es' and np.any(mask & ((p[..., 0] >= 0) | (p[..., 1] >= 0) | (p[..., 1] > p[..., 0]))):
        raise ValueError('FZ0 requires negative ordered ES<=VaR<0')
    baseline = _baseline(data, train_ids, spec)
    base = np.broadcast_to(baseline, p.shape)
    return y, p, mask, base, _loss(y, p, spec), _loss(y, base, spec), _scale(data, train_ids, spec)


def _coordinate_fields(spec, coord, cfg):
    out = {'coordinate': json.dumps(list(coord), separators=(',', ':'))}
    if spec['kind'] == 'hazard':
        out['time_bar'] = coord[0] + 1
        out['threshold'] = cfg['targets']['hazard_thresholds'][coord[-1]]
        return out
    out['horizon'] = cfg['targets']['horizons'][coord[0]]
    name = spec['name']
    if name in ('reach_fixed', 'reach_dynamic'):
        out['direction'] = ('down', 'up')[coord[1]]
        out['threshold' if name == 'reach_fixed' else 'sigma_multiplier'] = cfg['targets'][
            'fixed_thresholds' if name == 'reach_fixed' else 'dynamic_multipliers'][coord[2]]
    if spec['kind'] in ('quantile', 'positive_quantile', 'var_es'):
        out['quantile' if spec['kind'] != 'var_es' else 'alpha'] = spec['levels'][coord[-1]]
    if name in ('passive_account_return', 'loss_probability', 'ruin',
                'perp_liquidation', 'perp_passive_return'):
        out['leverage'] = cfg['targets']['passive_leverages'][coord[1]]
        if name == 'loss_probability':
            out['loss_threshold'] = cfg['targets']['passive_loss_thresholds'][coord[2]]
    return out


def _episodes(ids, labels):
    ids, labels = np.asarray(ids), np.asarray(labels, bool)
    if not len(ids):
        return 0
    return int((labels & np.r_[True, (~labels[:-1]) | (np.diff(ids) != 1)]).sum())


def _calibration(y, p, bins=10):
    cells = []
    ece = 0.
    index = np.minimum((p * bins).astype(int), bins - 1)
    for k in range(bins):
        select = index == k
        if not select.any():
            continue
        observed, predicted = float(y[select].mean()), float(p[select].mean())
        count = int(select.sum())
        ece += count / len(y) * abs(observed - predicted)
        cells.append({'bin': k, 'n': count, 'predicted': predicted, 'observed': observed})
    return ece, json.dumps(cells, separators=(',', ':'))


def _native_metric(y, p, kind, ids, row):
    row['n'] = len(y)
    if not len(y):
        row['status'] = 'no_observed_targets'
        return
    row['status'] = 'constant_target' if np.ptp(y) == 0 else 'ok'
    if kind == 'binary':
        event = y > .5; decision = p >= .5
        row.update(events=int(event.sum()), prevalence=float(event.mean()),
                   event_episodes=_episodes(ids, event), predicted_episodes=_episodes(ids, decision),
                   false_alarm_episodes=_episodes(ids, decision & ~event), brier=float(np.mean((p - y) ** 2)),
                   log_loss=float(np.mean(-(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1))))),
                   recall=float(decision[event].mean()) if event.any() else np.nan,
                   precision=float(event[decision].mean()) if decision.any() else np.nan,
                   balanced_accuracy=(float(decision[event].mean() + (~decision[~event]).mean()) / 2)
                   if event.any() and (~event).any() else np.nan,
                   pr_auc=float(average_precision_score(y, p)) if event.any() and (~event).any() else np.nan,
                   roc_auc=float(roc_auc_score(y, p)) if event.any() and (~event).any() else np.nan)
        row['false_alarm_episodes_per_observed_day'] = row['false_alarm_episodes'] / (len(y) / 96)
        row['calibration_ece'], row['calibration_bins'] = _calibration(y, p)
    elif kind in ('categorical', 'hazard'):
        pred = p.argmax(-1)
        recalls = [float((pred[y == k] == k).mean()) if (y == k).any() else np.nan for k in range(p.shape[-1])]
        row.update(accuracy=float(np.mean(pred == y)), class_counts=json.dumps(np.bincount(y.astype(int), minlength=p.shape[-1]).tolist()),
                   balanced_accuracy=float(np.mean(recalls)) if np.isfinite(recalls).all() else np.nan,
                   ba_observed_classes=float(np.nanmean(recalls)),
                   log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y.astype(int)], EPS, 1)).mean()))
    elif kind in ('quantile', 'positive_quantile'):
        row.update(pinball=row['native_loss'], coverage=float(np.mean(y <= p)),
                   quantile_error=float(np.mean(y <= p) - row['quantile']))
    elif kind == 'var_es':
        q, e = p[:, 0], p[:, 1]
        event = y <= q
        row.update(fz0=row['native_loss'], var_exceedance_rate=float(event.mean()),
                   expected_exceedance_rate=row['alpha'], tail_events=int(event.sum()),
                   tail_event_episodes=_episodes(ids, event),
                   empirical_exceedance_mean=float(y[event].mean()) if event.any() else np.nan,
                   predicted_es_at_exceedance=float(e[event].mean()) if event.any() else np.nan,
                   constrained_negative_var_es=True)
    else:
        error = p - y
        zero = float(np.mean(y * y))
        row.update(rmse=float(np.sqrt(np.mean(error * error))), mae=float(np.mean(np.abs(error))),
                   oos_r2_zero=1 - float(np.mean(error * error)) / zero if zero > EPS else np.nan,
                   ic_spearman=float(spearmanr(y, p).statistic) if len(y) > 2 and np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan,
                   actual_mean=float(y.mean()), predicted_mean=float(p.mean()))
        if kind == 'variance':
            row.update(qlike=row['native_loss'],
                       calibration_ratio=float(y.mean() / p.mean()) if p.mean() > 0 else np.nan)


def evaluate_predictions(data, ids, predictions, specs, cfg, train_ids=None, identity=None):
    """One wide row per registered output coordinate, with explicit counts."""
    ids = np.asarray(ids, dtype=np.int64)
    records = []
    for spec in specs:
        name, kind = spec['name'], spec['kind']
        if name not in predictions:
            continue
        y, p, mask, base, loss, bl, scale = _head(data, ids, predictions[name], spec, train_ids)
        for coord in np.ndindex(tuple(spec['shape'])):
            sl = (slice(None),) + coord
            valid = mask[sl]
            ml, baseline = _mean(loss[sl], valid), _mean(bl[sl], valid)
            row = dict(identity or {}, head=name, task=spec['task'], kind=kind,
                       native_loss=float(ml), baseline_loss=float(baseline))
            row.update(_coordinate_fields(spec, coord, cfg))
            row['skill'] = 1 - float(ml / baseline) if np.isfinite(baseline) and baseline > EPS and kind != 'var_es' else np.nan
            row['loss_improvement'] = float(baseline - ml) if np.isfinite(baseline) else np.nan
            _native_metric(y[sl][valid], p[sl][valid], kind, ids[valid], row)
            if kind == 'binary':
                row['brier_skill'] = row['skill']
            if name == 'reach_fixed':
                key = 'first_down_bar' if coord[1] == 0 else 'first_up_bar'
                if key in data:
                    crossing = np.asarray(data[key][ids, coord[2]])
                    detected = valid & (y[sl] > .5) & (p[sl] >= .5) & (crossing < 32767)
                    row['detected_events'] = int(detected.sum())
                    row['first_crossing_lead_minutes_lower'] = float(crossing[detected].mean() * 15) if detected.any() else np.nan
                    row['first_crossing_lead_minutes_upper'] = float((crossing[detected] + 1).mean() * 15) if detected.any() else np.nan
                    row['lead_time_basis'] = 'first_crossing15mbarbounds_since_decision_not_execution'
            records.append(row)
        if kind == 'hazard':
            # Competing cumulative incidence preserves the same right-censor
            # trajectory; direction-specific reach is not an independent event.
            for hi, horizon in enumerate(cfg['targets']['horizons']):
                prob = p[:, :horizon]
                survival_before = np.concatenate([np.ones_like(prob[:, :1, :, 0]),
                                                  np.cumprod(prob[:, :-1, :, 0], axis=1)], axis=1)
                cumulative = (survival_before[..., None] * prob[..., 1:]).sum(1)
                observed = mask[:, :horizon].any(1)
                # A trajectory must be followed to event or the requested
                # horizon, otherwise the reach outcome is censored/unknown.
                seen_event = ((y[:, :horizon] > 0) & mask[:, :horizon]).any(1)
                fully_followed = mask[:, horizon - 1] | seen_event
                for threshold in range(y.shape[-1]):
                    for direction, klass in [('down', 1), ('up', 2)]:
                        truth = (((y[:, :horizon, threshold] == klass) & mask[:, :horizon, threshold]).any(1)).astype(float)
                        vv = observed[:, threshold] & fully_followed[:, threshold]
                        pp = cumulative[:, threshold, klass - 1]
                        row = dict(identity or {}, head='time_hazard_cumulative_incidence', task=spec['task'], kind='binary',
                                   horizon=horizon, direction=direction, threshold=cfg['targets']['hazard_thresholds'][threshold],
                                   coordinate=json.dumps([hi, threshold, klass]), native_loss=float(_mean((pp - truth) ** 2, vv)),
                                   baseline_loss=np.nan, skill=np.nan,
                                   outcome='first_competing_event_not_independent_reach')
                        _native_metric(truth[vv], pp[vv], 'binary', ids[vv], row)
                        records.append(row)
    return pd.DataFrame.from_records(records)


def per_row_losses(data, ids, predictions, specs, cfg, train_ids=None, *, horizon=None,
                   names=None, coordinate_filters=None, **unused):
    """Timestamp-paired losses by task plus head details.

    Native MSE/pinball/FZ losses share Train-only physical scales before
    averaging. Set horizon=16 for the primary4h comparison. Hazard loss sums
    its observed time likelihood then divides by fixed horizon length.
    """
    ids = np.asarray(ids, np.int64)
    task_parts, head_parts = defaultdict(list), {}
    for spec in specs:
        name, kind = spec['name'], spec['kind']
        if name not in predictions or names is not None and name not in names:
            continue
        y, p, valid, base, loss, bl, scale = _head(data, ids, predictions[name], spec, train_ids)
        if kind in ('real', 'positive', 'bounded'):
            loss, bl = loss / scale ** 2, bl / scale ** 2
        elif kind in ('quantile', 'positive_quantile'):
            loss, bl = loss / scale, bl / scale
        elif kind == 'var_es':
            # FZ0(y/s,q/s,e/s)=FZ0(y,q,e)-log(s), retaining proper scoring.
            loss, bl = loss - np.log(scale), bl - np.log(scale)
        if coordinate_filters and name in coordinate_filters:
            chosen = np.zeros(tuple(spec['shape']), bool)
            for c in coordinate_filters[name]:
                chosen[tuple(c)] = True
            valid &= chosen
        if kind == 'hazard':
            stop = int(horizon or spec['shape'][0])
            valid, loss, bl = valid[:, :stop], loss[:, :stop], bl[:, :stop]
            eligible = valid.any(1)
            model_part = np.where(valid, loss, 0).sum(1) / stop
            base_part = np.where(valid & np.isfinite(bl), bl, 0).sum(1) / stop
            base_part[(valid & ~np.isfinite(bl)).any(1) | ~valid.any(1)] = np.nan
            model_part = _mean(model_part, eligible, axis=1)
            base_part = _mean(base_part, eligible, axis=1)
        else:
            if horizon is not None:
                hi = list(cfg['targets']['horizons']).index(horizon)
                valid, loss, bl = valid[:, hi], loss[:, hi], bl[:, hi]
            shape = (len(ids), -1)
            model_part = _mean(loss.reshape(shape), valid.reshape(shape), axis=1)
            base_part = _mean(bl.reshape(shape), valid.reshape(shape), axis=1)
        eligible = np.isfinite(model_part)
        item = {'model_loss': model_part, 'baseline_loss': base_part, 'valid': eligible,
                'ids': ids, 'index_ns': np.asarray(data['index_ns'][ids])}
        head_parts[name] = item
        task_parts[spec['task']].append(item)
    result = {}
    for task, parts in task_parts.items():
        ml = np.column_stack([v['model_loss'] for v in parts])
        bl = np.column_stack([v['baseline_loss'] for v in parts])
        result[task] = {'model_loss': _mean(ml, np.isfinite(ml), axis=1),
                        'baseline_loss': _mean(bl, np.isfinite(bl), axis=1),
                        'valid': np.isfinite(ml).any(1), 'ids': ids,
                        'index_ns': np.asarray(data['index_ns'][ids])}
    # Optional details are attached without polluting task iteration.
    return result


def score_predictions(data, ids, predictions, specs, cfg, train_ids=None, *, return_details=False, **kwargs):
    """Equal-task loss relative to Train-fitted baselines; lower is better."""
    if train_ids is None or not len(train_ids):
        raise ValueError('Val selection requires explicit Train-only reference rows')
    tasks = per_row_losses(data, ids, predictions, specs, cfg, train_ids, **kwargs)
    scores, counts = {}, {}
    for task, part in tasks.items():
        valid = part['valid'] & np.isfinite(part['baseline_loss'])
        model, baseline = _mean(part['model_loss'], valid), _mean(part['baseline_loss'], valid)
        if not np.isfinite(baseline) or abs(float(baseline)) <= EPS:
            scores[task] = np.nan
        else:
            denom = max(abs(float(baseline)), 1.) if task == 'T10' else abs(float(baseline))
            scores[task] = 1 + float(model - baseline) / denom
        counts[task] = int(valid.sum())
    values = np.asarray(list(scores.values()), float)
    score = float(_mean(values, np.isfinite(values)))
    if return_details:
        return {'score': score, 'by_task': scores, 'n_by_task': counts,
                'n_available_tasks': int(np.isfinite(values).sum())}
    return score
