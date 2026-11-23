"""Isolated causal inputs and path targets for the registered tail-transfer study.

Calendar row d is a decision after raw candle d-1 closes.  Predictive labels
use raw candles d..d+h-1; passive-account labels instead fill at open[d+1].
The original research data are read-only.  All repaired/derived arrays live in
the new campaign's data directory.  Liquidation data are observed exchange
snapshots on free month-first-day samples, never a full liquidation history.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml

from .wm_ablation_features import BASE_COLUMNS, TA_COLUMNS, transform_windows
from .wm_technical_ablation import MARKET_COLUMNS, market_features

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/wm_tail_transfer_20261112.yaml'
BAR_NS = 900_000_000_000
MIN_NS = 60_000_000_000
CHANGE_COLUMNS = tuple(
    [f'oi_change_{h}' for h in (1, 4, 16, 64)]
    + [f'oi_value_change_{h}' for h in (1, 4, 16, 64)]
    + [f'funding_mean_{h}' for h in (16, 64)]
    + [f'funding_change_{h}' for h in (4, 16)]
    + [f'longshort_change_{h}' for h in (4, 16)]
    + [f'basis_change_{h}' for h in (1, 4, 16)]
    + ['perp_spot_taker_divergence']
    + [f'price_oi_interaction_{h}' for h in (1, 4, 16)]
)
MINUTE_COLUMNS = tuple(f'{source}_minute_{name}' for source in ('spot', 'perp')
                       for name in ('rv', 'up_rv', 'down_rv', 'max_return', 'min_return', 'volume_concentration'))
DEPTH_COLUMNS = tuple(f'depth_{name}_{band}' for band in (1, 2, 3, 4, 5)
                      for name in ('log_bid_notional', 'log_ask_notional', 'imbalance'))
LIQ_COLUMNS = ('liq_log_sell_notional', 'liq_log_buy_notional', 'liq_imbalance', 'liq_log_count')
GROUPS = ('D00', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D07_dense')


def _cfg(cfg):
    if isinstance(cfg, (str, Path)):
        cfg = yaml.safe_load(Path(cfg).read_text())
    return cfg


def _out(cfg):
    p = Path(cfg.get('output_dir', 'codex_outputs/wm_tail_transfer_20261112'))
    return (p if p.is_absolute() else ROOT / p) / 'data'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    tmp.replace(path)


def group_columns(group):
    cols = tuple(BASE_COLUMNS)
    if group != 'D00':
        cols += tuple(MARKET_COLUMNS)
    if group not in ('D00', 'D01'):
        cols += tuple(TA_COLUMNS)
    if group in ('D03', 'D04', 'D05', 'D06', 'D07', 'D07_dense'):
        cols += CHANGE_COLUMNS
    if group in ('D04', 'D07', 'D07_dense'):
        cols += MINUTE_COLUMNS
    if group in ('D05', 'D07', 'D07_dense'):
        cols += DEPTH_COLUMNS
    if group in ('D06', 'D07'):
        cols += LIQ_COLUMNS
    if group not in GROUPS:
        raise ValueError(f'unknown group {group}')
    return cols


def _changes(market, base):
    def delta(x, h):
        j = np.maximum(np.arange(x.shape[1]) - h, 0)
        return x - x[:, j]
    def mean(x, h):
        c = np.cumsum(x, axis=1)
        s = c.copy()
        s[:, h:] -= c[:, :-h]
        return s / np.minimum(np.arange(x.shape[1]) + 1, h)
    oi, oiv, funding, ratio, basis = [market[:, :, i] for i in (11, 12, 10, 13, 6)]
    values = ([delta(oi, h) for h in (1, 4, 16, 64)]
              + [delta(oiv, h) for h in (1, 4, 16, 64)]
              + [mean(funding, h) for h in (16, 64)]
              + [delta(funding, h) for h in (4, 16)]
              + [delta(ratio, h) for h in (4, 16)]
              + [delta(basis, h) for h in (1, 4, 16)]
              + [market[:, :, 5] - market[:, :, 2]])
    cumulative_return = np.cumsum(base[:, :, 3], axis=1)
    values += [delta(cumulative_return, h) * delta(oi, h) for h in (1, 4, 16)]
    return np.stack(values, axis=-1).astype(np.float32)


def window_inputs(data, decisions, group, context=128):
    """B,T,F; all rolling features restart inside the identical raw context.

    Liquidations are a current-summary input: repeat only the last completed
    bar's liquidation summary across tokens, and explicitly flag the sparse
    month-first-day cohort. No earlier unknown liquidation counts become0.
    """
    ids = np.asarray(decisions, dtype=np.int64)
    if ids.ndim != 1 or len(ids) and (ids.min() < context or ids.max() >= len(data['index_ns'])):
        raise ValueError('decisions must have a complete preceding context')
    positions = ids[:, None] - context + np.arange(context)[None, :]
    raw = np.asarray(data['ohlcspot'][positions], dtype=np.float64)
    market = np.asarray(data['market'][positions], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('missing raw context')
    g = 'A' if group == 'D00' else ('C' if group == 'D01' else 'D')
    if g != 'A' and not np.isfinite(market).all():
        raise ValueError('missing market context')
    values = transform_windows(raw, g, market_windows=market if g != 'A' else None,
                               market_columns=MARKET_COLUMNS if g != 'A' else ())
    parts = [values]
    if group in ('D03', 'D04', 'D05', 'D06', 'D07', 'D07_dense'):
        parts.append(_changes(market, values[:, :, :5]))
    if group in ('D04', 'D07', 'D07_dense'):
        parts.append(np.asarray(data['minute_features'][positions]))
    if group in ('D05', 'D07', 'D07_dense'):
        parts.append(np.asarray(data['depth_features'][positions]))
    if group in ('D06', 'D07'):
        liq = np.asarray(data['liquidation_features'][ids - 1])
        parts.append(np.broadcast_to(liq[:, None], (len(ids), context, len(LIQ_COLUMNS))))
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if result.shape[-1] != len(group_columns(group)) or not np.isfinite(result).all():
        raise ValueError('group window missing or schema mismatch')
    return result


def _read_kline(path, minutes):
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('one CSV required')
        payload = z.read(members[0])
    first = payload.splitlines()[0].split(b',')[0]
    frame = pd.read_csv(io.BytesIO(payload), header=None if first.isdigit() else 0)
    if frame.shape[1] != 12:
        raise ValueError('unexpected kline schema')
    values = frame.to_numpy(dtype=np.float64)
    unit = 'us' if np.nanmedian(values[:, 0]) > 1e14 else 'ms'
    times = pd.to_datetime(values[:, 0].astype(np.int64), unit=unit, utc=True).asi8
    close_times = pd.to_datetime(values[:, 6].astype(np.int64), unit=unit, utc=True).asi8
    price = values[:, 1:6]
    good = ((times % (minutes * MIN_NS) == 0)
            & (close_times >= times + minutes * MIN_NS - 1_000_000)
            & (close_times < times + minutes * MIN_NS)
            & np.isfinite(price).all(1) & (price[:, :4] > 0).all(1)
            & (price[:, 4] >= 0) & (price[:, 2] <= np.minimum(price[:, 0], price[:, 3]))
            & (price[:, 1] >= np.maximum(price[:, 0], price[:, 3])))
    price[~good] = np.nan
    return times, price


def _minute_arrays(out, index_ns):
    start = pd.Timestamp('2023-01-01', tz='UTC').value
    end = int(index_ns[-1]) + BAR_NS
    n = (end - start) // MIN_NS
    results = {}
    audit = []
    for source in ('spot', 'perp', 'mark'):
        path = out / 'prepared' / f'{source}_one_minute.npy'
        values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(n, 5))
        values[:] = np.nan
        paths = sorted((out / 'archives' / f'{source}1m').glob('*.zip'))
        paths += sorted((out / 'archives' / f'{source}1m_daily').glob('*.zip'))
        for p in paths:
            t, pvals = _read_kline(p, 1)
            ids = (t - start) // MIN_NS
            valid = (ids >= 0) & (ids < n)
            positions, incoming = ids[valid], pvals[valid]
            existing = np.asarray(values[positions])
            if 'daily' in p.parent.name:
                fill = ~np.isfinite(existing).all(1) & np.isfinite(incoming).all(1)
            else:
                fill = np.ones(len(positions), bool)
            values[positions[fill]] = incoming[fill]
            audit.append({'path': str(p), 'sha256': _sha(p), 'rows': len(t),
                          'filled': int(fill.sum()), 'invalid': int((~np.isfinite(pvals).all(1)).sum())})
        values.flush()
        results[source] = values
    return start, results, audit


def _aggregate_minutes(values):
    n = len(values) // 15
    m = np.asarray(values[:n * 15]).reshape(n, 15, 5)
    result = np.column_stack([m[:, 0, 0], np.max(m[:, :, 1], axis=1), np.min(m[:, :, 2], axis=1),
                              m[:, -1, 3], np.sum(m[:, :, 4], axis=1)])
    result[~np.isfinite(m).all((1, 2))] = np.nan
    close = np.asarray(values[:, 3])
    ret = np.r_[np.nan, np.diff(np.log(close))][:n * 15].reshape(n, 15)
    rv = np.square(ret)
    vol = m[:, :, 4]
    totals = vol.sum(1)
    concentration = np.divide(vol.max(1), totals, out=np.zeros(n), where=totals > 0)
    features = np.column_stack([rv.sum(1), np.where(ret > 0, rv, 0).sum(1),
                                np.where(ret < 0, rv, 0).sum(1), ret.max(1), ret.min(1), concentration])
    features[~np.isfinite(m).all((1, 2)) | ~np.isfinite(ret).all(1)] = np.nan
    return result, features.astype(np.float32)


def _depth(out, index_ns):
    """Causal snapshots only; exclude each band after a trailing3min freeze.

    Absence of a new update for90sec is unavailable; no interpolation. Each
    raw15m token uses the last snapshot before its close, with one-second lag.
    """
    result = np.full((len(index_ns), len(DEPTH_COLUMNS)), np.nan, np.float32)
    audit = []
    for p in sorted((out / 'archives' / 'bookDepth').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            f = pd.read_csv(z.open(z.namelist()[0]))
        if list(f.columns) != ['timestamp', 'percentage', 'depth', 'notional']:
            raise ValueError('unexpected depth schema')
        f['timestamp'] = pd.to_datetime(f.timestamp, utc=True)
        frame = []
        stale_count = 0
        for band in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
            q = f.loc[f.percentage == band].set_index('timestamp')[['depth', 'notional']].sort_index()
            q = q.loc[~q.index.duplicated(keep=False)]
            if not len(q):
                continue
            # Only information through the current snapshot enters its mask.
            changed = q.ne(q.shift(1)).any(axis=1)
            change_ts = pd.Series(q.index.asi8, index=q.index).where(changed).ffill()
            stale = q.index.asi8 - change_ts.to_numpy() >= 180 * 1_000_000_000
            invalid = stale | ~np.isfinite(q).all(axis=1).to_numpy() | (q < 0).any(axis=1).to_numpy()
            stale_count += int(stale.sum())
            q.loc[invalid] = np.nan
            q.columns = [f'{c}_{band}' for c in q.columns]
            frame.append(q)
        if not frame:
            continue
        wide = pd.concat(frame, axis=1).sort_index()
        begin = int(np.searchsorted(index_ns, wide.index[0].floor('D').value))
        end = int(np.searchsorted(index_ns, (wide.index[0].floor('D') + pd.Timedelta(days=1)).value))
        rows = np.arange(begin, end)
        cutoff = index_ns[rows] + BAR_NS - 1_000_000_000
        loc = np.searchsorted(wide.index.asi8, cutoff, side='right') - 1
        valid = (loc >= 0) & (cutoff - wide.index.asi8[np.maximum(loc, 0)] <= 90 * 1_000_000_000)
        selected = wide.iloc[np.maximum(loc, 0)]
        cols = []
        for band in (1, 2, 3, 4, 5):
            bid = selected.get(f'notional_{-band}', pd.Series(np.nan, index=selected.index)).to_numpy()
            ask = selected.get(f'notional_{band}', pd.Series(np.nan, index=selected.index)).to_numpy()
            imbalance = np.divide(bid - ask, bid + ask, out=np.zeros(len(rows)), where=bid + ask > 0)
            cols += [np.log1p(bid), np.log1p(ask), imbalance]
        values = np.column_stack(cols)
        values[~valid] = np.nan
        result[rows] = values
        audit.append({'date': p.stem[-10:], 'snapshots': len(wide), 'frozen_band_snapshots': stale_count,
                      'valid_15m_rows': int(np.isfinite(values).all(1).sum()), 'sha256': _sha(p)})
    return result, audit


def _liquidation(out, index_ns):
    result = np.full((len(index_ns), len(LIQ_COLUMNS)), np.nan, np.float32)
    audit = []
    for p in sorted((out / 'archives' / 'liquidation_sample').glob('*.csv.gz')):
        day = pd.Timestamp(p.name[:10], tz='UTC')
        rows = np.flatnonzero((index_ns >= day.value) & (index_ns < day.value + 86400 * 1_000_000_000))
        f = pd.read_csv(p)
        required = {'timestamp', 'local_timestamp', 'side', 'price', 'amount'}
        if not required.issubset(f.columns):
            raise ValueError(f'liquidation schema: {list(f.columns)}')
        observed = pd.to_datetime(f.local_timestamp, unit='us', utc=True).astype('int64').to_numpy()
        # local collector observation, not exchange event time, controls use.
        ids = (observed - int(index_ns[0])) // BAR_NS
        values = np.zeros((len(rows), 4), dtype=np.float64)
        for c, side in enumerate(('sell', 'buy')):
            choose = (f.side == side).to_numpy() & np.isfinite(f.price * f.amount).to_numpy()
            series = pd.Series((f.price * f.amount).to_numpy()[choose], index=ids[choose]).groupby(level=0).sum()
            values[:, c] = series.reindex(rows, fill_value=0).to_numpy()
        counts = pd.Series(1, index=ids).groupby(level=0).sum().reindex(rows, fill_value=0).to_numpy()
        values[:, 2] = np.divide(values[:, 0] - values[:, 1], values[:, 0] + values[:, 1],
                                 out=np.zeros(len(rows)), where=values[:, 0] + values[:, 1] > 0)
        values[:, :2] = np.log1p(values[:, :2])
        values[:, 3] = np.log1p(counts)
        # Sample begins at midnight; late messages from the preceding unknown
        # day are dropped, and no unsampled day is treated as zero events.
        result[rows] = values
        audit.append({'date': str(day.date()), 'events': len(f), 'scope': 'month_first_day_observed_snapshot_sample',
                      'availability_clock': 'collector local_timestamp', 'sha256': _sha(p)})
    return result, audit


def _resolve_order(ids, offset, up_barrier, down_barrier, minute, minute_start, index_ns):
    """Return -1down/+1up/0unresolved for a same15m first hit."""
    result = np.zeros(len(ids), np.int8)
    if minute is None or not len(ids):
        return result
    starts = (index_ns[ids + offset] - minute_start) // MIN_NS
    good = (starts >= 0) & (starts + 15 <= len(minute))
    selected = np.flatnonzero(good)
    if len(selected):
        pos = starts[selected, None] + np.arange(15)[None, :]
        bars = np.asarray(minute[pos])
        finite = np.isfinite(bars[:, :, :4]).all((1, 2))
        up = bars[:, :, 1] >= up_barrier[selected, None]
        down = bars[:, :, 2] <= down_barrier[selected, None]
        a, b = up.argmax(1), down.argmax(1)
        resolved = finite & up.any(1) & down.any(1) & (a != b)
        result[selected[resolved]] = np.where(a[resolved] < b[resolved], 1, -1)
    return result


def build_targets(index_ns, ohlc, cfg, *, minute=None, minute_start=0):
    """Per-horizon masks, explicit ambiguous-order masks and fixed label ends."""
    tcfg = cfg['targets']
    horizons = list(tcfg['horizons'])
    fixed = np.asarray(tcfg['fixed_thresholds'], dtype=np.float64)
    dyn = np.asarray(tcfg['dynamic_multipliers'], dtype=np.float64)
    n, nh = len(index_ns), len(horizons)
    close, high, low = [np.asarray(ohlc[:, i]) for i in (3, 1, 2)]
    prior = np.r_[np.nan, close[:-1]]
    r = np.r_[np.nan, np.diff(np.log(close))]
    # r[d-1] is known at d. Missing observations do not become zero returns.
    ewma = pd.Series(r * r).ewm(alpha=1 - float(tcfg['ewma_decay']), adjust=False, min_periods=16).mean().shift(1).to_numpy()
    sigma = np.sqrt(np.maximum(ewma[:, None] * np.asarray(horizons)[None, :], 1e-12))
    targets, masks = {}, {}
    def put(name, a, mask=None):
        targets[name] = np.asarray(a)
        masks[name] = np.isfinite(a) if mask is None else np.asarray(mask, bool) & np.isfinite(a)
    scalar = {k: np.full((n, nh), np.nan, np.float32) for k in
              ('return', 'mfe', 'mae', 'peak_dd', 'rv', 'up_semivariance', 'down_semivariance')}
    reach = np.full((n, nh, 2, len(fixed)), np.nan, np.float32)
    reachdyn = np.full((n, nh, 2, len(dyn)), np.nan, np.float32)
    classes = np.full((n, nh, 2), np.nan, np.float32)
    barrier = np.full((n, nh, 3, 2), np.nan, np.float32)
    barrier_bounds = np.full((n, nh, 3, 2, 2), np.nan, np.float32)
    valid_path = np.isfinite(prior) & (prior > 0)
    future_max, future_min, running_peak = prior.copy(), prior.copy(), prior.copy()
    close_peak = prior.copy()
    maxdd = np.zeros(n)
    lower_dd = np.zeros(n); upper_dd = np.zeros(n)
    dd_bounds = np.full((n, nh, 2), np.nan, np.float32)
    close_min, close_max, internal_dd = close.copy(), close.copy(), np.zeros(n)
    minute_complete = np.zeros(n, bool)
    if minute is not None:
        first = int(np.searchsorted(index_ns, minute_start))
        nbar = min(len(minute) // 15, n - first)
        cpath = np.asarray(minute[:nbar * 15, 3]).reshape(nbar, 15)
        complete = np.isfinite(cpath).all(1)
        minute_complete[first:first + nbar] = complete
        close_min[first:first + nbar] = cpath.min(1)
        close_max[first:first + nbar] = cpath.max(1)
        internal_dd[first:first + nbar] = np.max(1 - cpath / np.maximum.accumulate(cpath, axis=1), axis=1)
    dd_path_valid = np.isfinite(prior) & (prior > 0)
    rv = np.zeros(n); upv = np.zeros(n); dnv = np.zeros(n)
    haz = np.full((n, max(horizons), len(fixed)), np.nan, np.float32)
    hazmask = np.zeros(haz.shape, bool)
    active = np.ones((n, len(fixed)), bool)
    fixed_up = np.full((n, len(fixed)), 32767, np.int16)
    fixed_down = fixed_up.copy()
    # Dynamic first passage depends on each horizon's fixed decision-time sigma.
    dyn_up = np.full((n, nh, len(dyn)), 32767, np.int16); dyn_down = dyn_up.copy()
    bu = np.full((n, nh, 3, 2), 32767, np.int16); bd = bu.copy()
    order = np.zeros(bu.shape, np.int8)
    same_unknown = np.zeros(bu.shape, bool)
    rows = np.arange(n)
    for offset in range(max(horizons)):
        j = rows + offset
        within = j < n
        safe = np.minimum(j, n - 1)
        hh, ll, cc, rr = high[safe], low[safe], close[safe], r[safe]
        valid_path &= within & np.isfinite(hh) & np.isfinite(ll) & np.isfinite(cc) & np.isfinite(rr)
        future_max = np.maximum(future_max, hh)
        future_min = np.minimum(future_min, ll)
        # Exact observed close-order primary target; high/low timing remains
        # represented by lower/upper bounds instead of invented intrabar order.
        dd_path_valid &= within & np.isfinite(close_min[safe]) & np.isfinite(close_max[safe])
        maxdd = np.maximum(maxdd, np.maximum(1 - close_min[safe] / close_peak, internal_dd[safe]))
        close_peak = np.maximum(close_peak, close_max[safe])
        lower_dd = np.maximum(lower_dd, np.maximum(1 - ll / running_peak,
                                                   1 - cc / np.maximum(running_peak, hh)))
        running_peak = np.maximum(running_peak, hh)
        upper_dd = np.maximum(upper_dd, 1 - ll / running_peak)
        rv += np.square(rr); upv += np.where(rr > 0, rr * rr, 0); dnv += np.where(rr < 0, rr * rr, 0)
        uh = hh[:, None] >= prior[:, None] * (1 + fixed)
        dh = ll[:, None] <= prior[:, None] * (1 - fixed)
        fixed_up[(fixed_up == 32767) & uh] = offset
        fixed_down[(fixed_down == 32767) & dh] = offset
        for k in range(len(fixed)):
            choose = active[:, k] & valid_path
            labels = np.where(dh[:, k], 1, np.where(uh[:, k], 2, 0)).astype(np.float32)
            amb = choose & uh[:, k] & dh[:, k]
            ai = np.flatnonzero(amb)
            resolution = _resolve_order(ai, offset, prior[ai] * (1 + fixed[k]),
                                        prior[ai] * (1 - fixed[k]), minute, minute_start, index_ns)
            labels[ai[resolution == 1]] = 2
            haz[:, offset, k] = labels
            hazmask[:, offset, k] = choose
            hazmask[ai[resolution == 0], offset, k] = False
            active[:, k] &= ~(uh[:, k] | dh[:, k]) & valid_path
        for hi, horizon in enumerate(horizons):
            if offset >= horizon:
                continue
            upper = prior[:, None] * (1 + sigma[:, hi, None] * dyn)
            lower = prior[:, None] * (1 - sigma[:, hi, None] * dyn)
            a, b = dyn_up[:, hi], dyn_down[:, hi]
            a[(a == 32767) & (hh[:, None] >= upper)] = offset
            b[(b == 32767) & (ll[:, None] <= lower)] = offset
            for ri, (profit, loss) in enumerate(tcfg['barrier_ratios']):
                for wi in range(2):
                    width = np.full(n, float(tcfg['barrier_fixed_base'])) if wi == 0 else sigma[:, hi]
                    ub = prior * (1 + profit * width); lb = prior * (1 - loss * width)
                    uphit, dnhit = hh >= ub, ll <= lb
                    a, b = bu[:, hi, ri, wi], bd[:, hi, ri, wi]
                    first = (a == 32767) & (b == 32767) & valid_path
                    amb = first & uphit & dnhit
                    ai = np.flatnonzero(amb)
                    resolution = _resolve_order(ai, offset, ub[ai], lb[ai], minute, minute_start, index_ns)
                    order[ai, hi, ri, wi] = resolution
                    same_unknown[ai[resolution == 0], hi, ri, wi] = True
                    a[(a == 32767) & uphit] = offset
                    b[(b == 32767) & dnhit] = offset
            if offset + 1 == horizon:
                valid = valid_path.copy()
                value = np.log(cc / prior)
                for key, array in [('return', value), ('mfe', np.maximum(0, future_max / prior - 1)),
                                    ('mae', np.maximum(0, 1 - future_min / prior)), ('peak_dd', maxdd),
                                    ('rv', rv), ('up_semivariance', upv), ('down_semivariance', dnv)]:
                    scalar[key][valid, hi] = array[valid]
                scalar['peak_dd'][~dd_path_valid, hi] = np.nan
                dd_bounds[valid, hi, 0] = lower_dd[valid]
                dd_bounds[valid, hi, 1] = upper_dd[valid]
                reach[:, hi, 0] = (fixed_down < horizon); reach[:, hi, 1] = (fixed_up < horizon)
                reach[~valid, hi] = np.nan
                reachdyn[:, hi, 0] = (dyn_down[:, hi] < horizon); reachdyn[:, hi, 1] = (dyn_up[:, hi] < horizon)
                reachdyn[~valid | ~np.isfinite(sigma[:, hi]), hi] = np.nan
                for wi in range(2):
                    threshold = float(tcfg['return_class_fixed_threshold']) if wi == 0 else float(tcfg['return_class_dynamic_multiplier']) * sigma[:, hi]
                    classes[:, hi, wi] = np.where(value < -threshold, 0, np.where(value > threshold, 2, 1))
                    classes[~valid | ~np.isfinite(np.broadcast_to(threshold, (n,))), hi, wi] = np.nan
                a, b = bu[:, hi], bd[:, hi]
                classes_b = np.where(a < b, 2, np.where(b < a, 0, 1)).astype(np.float32)
                hit_same = (a == b) & (a < horizon)
                classes_b[hit_same & (order[:, hi] > 0)] = 2
                classes_b[hit_same & (order[:, hi] < 0)] = 0
                lower_b, upper_b = classes_b.copy(), classes_b.copy()
                lower_b[same_unknown[:, hi]] = 0; upper_b[same_unknown[:, hi]] = 2
                classes_b[same_unknown[:, hi]] = np.nan
                classes_b[~valid] = np.nan
                classes_b[:, :, 1][~np.isfinite(sigma[:, hi])] = np.nan
                barrier[:, hi] = classes_b
                barrier_bounds[:, hi, :, :, 0] = lower_b
                barrier_bounds[:, hi, :, :, 1] = upper_b
    for key, value in scalar.items():
        put(key, value)
    put('reach_fixed', reach); put('reach_dynamic', reachdyn); put('return_class', classes)
    put('barrier', barrier); put('time_hazard', haz, hazmask)
    put('return_quantiles', np.repeat(scalar['return'][:, :, None], len(tcfg['return_quantiles']), axis=2))
    for source in ('mfe', 'mae', 'peak_dd'):
        put(source + '_quantiles', np.repeat(scalar[source][:, :, None], len(tcfg['positive_quantiles']), axis=2))
    put('var_es', np.repeat(scalar['return'][:, :, None], len(tcfg['es_alphas']), axis=2))
    # Preserve old close-only synthetic targets as explicit controls; MAE in
    # the new primary task uses lows and is intentionally a different target.
    from .wm_seven_task_evaluation import build_seven_targets
    legacy = build_seven_targets(pd.to_datetime(index_ns, utc=True), close,
                                  horizons=horizons, crash_threshold=float(tcfg['crash_threshold']))
    for name in ('crash', 'dd_excess', 'advantage', 'recovery'):
        put(name, legacy[name].astype(np.float32))
    put('legacy_downside', legacy['downside'].astype(np.float32))
    # Passive labels: decision d -> fill open d+1, hold fixed BTC units; fee is
    # charged on actual acquisition notional, debt accrues every scored bar.
    leverages = np.asarray(tcfg['passive_leverages'])
    losslevels = np.asarray(tcfg['passive_loss_thresholds'])
    passive = np.full((n, nh, len(leverages)), np.nan, np.float32)
    ruin = passive.copy(); losses = np.full((*passive.shape, len(losslevels)), np.nan, np.float32)
    fee = float(cfg['accounting']['cost_rate'])
    borrow = math.expm1(float(cfg['accounting']['borrow_apr']) / float(cfg['accounting']['annual_bars']))
    entry = np.r_[ohlc[1:, 0], np.nan]
    notional = leverages[None, :] / (1 + fee * leverages[None, :])
    units = notional / entry[:, None]
    cash = np.broadcast_to(1 - notional * (1 + fee), units.shape).copy()
    alive = np.ones(units.shape, bool)
    minimum_nav = np.ones(units.shape)
    valid = np.isfinite(entry) & (entry > 0)
    for offset in range(1, max(horizons) + 1):
        j = rows + offset
        within = j < n; safe = np.minimum(j, n - 1)
        oo, ll, cc = [ohlc[safe, col] for col in (0, 2, 3)]
        valid &= within & np.isfinite(oo) & np.isfinite(ll) & np.isfinite(cc)
        cash -= np.maximum(-cash, 0) * borrow
        navlow = cash + units * ll[:, None]
        navclose = cash + units * cc[:, None]
        navopen = cash + units * oo[:, None]
        alive &= (navlow > 0) & (navopen > 0) & (navclose > 0)
        minimum_nav = np.minimum(minimum_nav, np.where(alive, np.minimum(navlow, np.minimum(navopen, navclose)), 0))
        if offset in horizons:
            hi = horizons.index(offset)
            net = np.where(alive, navclose, 0) - 1
            passive[valid, hi] = net[valid]
            ruin[valid, hi] = (~alive)[valid]
            losses[valid, hi] = (minimum_nav[:, :, None] <= 1 - losslevels)[valid]
    put('passive_account_return', passive); put('ruin', ruin); put('loss_probability', losses)
    metadata = {'predictive_label_end_ns': index_ns[:, None] + np.asarray(horizons)[None, :] * BAR_NS,
                'passive_label_end_ns': index_ns[:, None] + (1 + np.asarray(horizons)[None, :]) * BAR_NS,
                'causal_sigma': sigma.astype(np.float32), 'barrier_bounds': barrier_bounds,
                'first_up_bar': fixed_up, 'first_down_bar': fixed_down,
                'peak_dd_definition': 'ordered_observed1mclose_drawdown_from2023;15mclose_before2023_diagnostic_only',
                'loss_probability_definition': 'passive_fixed_units_minimum_open_low_close_NAV_below_initial_equity_threshold',
                'peak_dd_highlow_bounds': dd_bounds,
                'minute_complete_bar': minute_complete,
                'direction_order': ['down', 'up'], 'hazard_classes': ['none', 'down', 'up'],
                'barrier_classes': ['down', 'timeout', 'up']}
    return targets, masks, metadata


def prepare(cfg):
    cfg = _cfg(cfg)
    out = _out(cfg); prepared = out / 'prepared'
    prepared.mkdir(parents=True, exist_ok=True)
    if (prepared / 'manifest.json').exists():
        manifest = json.loads((prepared / 'manifest.json').read_text())
        if manifest.get('complete'):
            return manifest
    legacy = Path(cfg['data']['legacy_data_root'])
    if not legacy.is_absolute():
        legacy = ROOT / legacy
    rawpath = Path(cfg['data']['raw_market_path'])
    if not rawpath.is_absolute():
        rawpath = ROOT / rawpath
    raw = pd.read_parquet(rawpath)
    index_ns = raw.index.asi8
    if not raw.index.is_unique or not np.all(np.diff(index_ns) == BAR_NS):
        raise ValueError('complete unique raw calendar required')
    _json(out / 'preparation_status.json', {'phase': 'minute_parse'})
    minute_start, minutes, minute_audit = _minute_arrays(out, index_ns)
    ohlcs = {}
    aggregates = {}
    features = np.full((len(raw), len(MINUTE_COLUMNS)), np.nan, np.float32)
    startrow = int(np.searchsorted(index_ns, minute_start))
    repairs = []
    for source in ('spot', 'perp'):
        values = raw[[source + '_' + c for c in ('open', 'high', 'low', 'close', 'volume')]].to_numpy(float)
        agg, feat = _aggregate_minutes(minutes[source])
        aggregates[source] = agg
        target = values[startrow:startrow + len(agg)]
        missing = ~np.isfinite(target).all(1) & np.isfinite(agg).all(1)
        target[missing] = agg[missing]
        repairs += [{'source': source, 'timestamp': str(raw.index[startrow + i]), 'method': 'complete15observed1m_aggregate'}
                    for i in np.flatnonzero(missing)]
        ohlcs[source] = values
        col = 0 if source == 'spot' else 6
        features[startrow:startrow + len(feat), col:col + 6] = feat
    mark = np.full((len(raw), 5), np.nan)
    for p in sorted((legacy / 'archives' / 'mark').glob('*.zip')):
        times, values = _read_kline(p, 15)
        loc = np.searchsorted(index_ns, times)
        good = (loc < len(index_ns)) & (index_ns[np.minimum(loc, len(index_ns) - 1)] == times)
        mark[loc[good]] = values[good]
    agg, _ = _aggregate_minutes(minutes['mark'])
    m = mark[startrow:startrow + len(agg)]
    missing = ~np.isfinite(m[:, :4]).all(1) & np.isfinite(agg[:, :4]).all(1)
    m[missing] = agg[missing]
    repairs += [{'source': 'mark', 'timestamp': str(raw.index[startrow + i]), 'method': 'complete15observed1m_aggregate'}
                for i in np.flatnonzero(missing)]
    ohlcs['mark'] = mark
    # Repair market close fields only when an actual observed aggregate exists;
    # auxiliary index/premium remain unavailable rather than becoming zeros.
    for source in ('spot', 'perp', 'mark'):
        absent = ~np.isfinite(raw[source + '_close'])
        raw.loc[absent, source + '_close'] = ohlcs[source][absent, 3]
    _json(out / 'preparation_status.json', {'phase': 'depth_parse'})
    depth, depth_audit = _depth(out, index_ns)
    liq, liq_audit = _liquidation(out, index_ns)
    market = market_features(raw)
    settlement = np.zeros(len(raw)); event_count = np.zeros(len(raw), np.int16)
    events = pd.read_parquet(legacy / 'base/funding_events.parquet')
    event_rows = (events.index.asi8 - int(index_ns[0])) // BAR_NS
    good = (event_rows >= 0) & (event_rows < len(raw))
    np.add.at(settlement, event_rows[good], events.funding_rate.to_numpy()[good])
    np.add.at(event_count, event_rows[good], 1)
    arrays = {'index_ns': index_ns, 'ohlcspot': ohlcs['spot'], 'ohlcperp': ohlcs['perp'], 'ohlcmark': mark,
              'market': market, 'minute_features': features, 'depth_features': depth,
              'liquidation_features': liq, 'funding_settlement_rate': settlement, 'funding_event_count': event_count}
    context = int(cfg['data']['context'])
    counts = np.r_[0, np.cumsum(np.isfinite(ohlcs['spot']).all(1), dtype=np.int64)]
    rowids = np.arange(len(raw)); start = np.maximum(rowids - context, 0)
    base_valid = (rowids >= context) & (counts[rowids] - counts[start] == context)
    def context_valid(a):
        v = np.isfinite(a).all(1)
        c = np.r_[0, np.cumsum(v, dtype=np.int64)]
        return (rowids >= context) & (c[rowids] - c[start] == context)
    mv, fv, dv = context_valid(market), context_valid(features), context_valid(depth)
    lv = np.r_[False, np.isfinite(liq[:-1]).all(1)]
    for group in GROUPS:
        valid = base_valid.copy()
        if group != 'D00': valid &= mv
        if group in ('D04', 'D07', 'D07_dense'): valid &= fv
        if group in ('D05', 'D07', 'D07_dense'): valid &= dv
        if group in ('D06', 'D07'): valid &= lv
        arrays['valid_' + group] = valid
    for name, value in arrays.items():
        np.save(prepared / (name + '.npy'), value)
    data = arrays.copy()
    _json(out / 'preparation_status.json', {'phase': 'window_last_token'})
    for group in GROUPS:
        output = np.lib.format.open_memmap(prepared / ('input_' + group + '.npy'), mode='w+', dtype=np.float32,
                                           shape=(len(raw), len(group_columns(group))))
        output[:] = np.nan
        ids = np.flatnonzero(arrays['valid_' + group])
        for begin in range(0, len(ids), 256):
            selected = ids[begin:begin + 256]
            output[selected] = window_inputs(data, selected, group, context)[:, -1]
        output.flush()
    _json(out / 'preparation_status.json', {'phase': 'path_targets'})
    targets, masks, metadata = build_targets(index_ns, ohlcs['spot'], cfg, minute=minutes['spot'], minute_start=minute_start)
    for name, values in targets.items():
        np.save(prepared / ('target_' + name + '.npy'), values)
        np.save(prepared / ('mask_' + name + '.npy'), masks[name])
    for name, values in metadata.items():
        if isinstance(values, np.ndarray):
            np.save(prepared / (name + '.npy'), values)
    _json(out / 'minute_parse_audit.json', minute_audit)
    _json(out / 'depth_quality_audit.json', depth_audit)
    _json(out / 'liquidation_sample_audit.json', liq_audit)
    _json(out / 'repairs.json', repairs)
    events.to_parquet(prepared / 'actual_funding_events.parquet')
    manifest = {'complete': True, 'rows': len(raw), 'start': str(raw.index[0]), 'end_exclusive': str(raw.index[-1] + pd.Timedelta(minutes=15)),
                'index_semantics': 'UTC15mbaropen', 'decision_input': 'd observes completed raw rows [d-context,d)',
                'context': context, 'groups': {g: {'columns': list(group_columns(g)), 'valid_decisions': int(arrays['valid_' + g].sum()),
                                                  'scope': 'month_first_day_current_summary' if g in ('D06', 'D07') else 'dense_calendar'} for g in GROUPS},
                'targets': {k: {'shape': list(v.shape[1:]), 'dtype': str(v.dtype), 'valid': int(masks[k].sum())} for k, v in targets.items()},
                'label_definitions': {k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)},
                'source_raw_sha256': _sha(rawpath), 'n_repaired_bars': len(repairs),
                'funding_convention': 'actual event bucketed to raw15mbar; eventclock retained; settle once afteropen beforelow',
                'historical_publication': 'unknown for archive market metrics; conservative source-time lag assumed',
                'artifacts': {p.name: _sha(p) for p in prepared.glob('*.npy')}}
    _json(prepared / 'manifest.json', manifest)
    _json(out / 'preparation_status.json', {'phase': 'complete', 'rows': len(raw)})
    return manifest


def load_data(cfg):
    cfg = _cfg(cfg)
    prepared = _out(cfg) / 'prepared'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('data preparation incomplete')
    arrays, targets, masks = {}, {}, {}
    for name in manifest['artifacts']:
        stem = Path(name).stem
        array = np.load(prepared / name, mmap_mode='r')
        if stem.startswith('target_'): targets[stem[7:]] = array
        elif stem.startswith('mask_'): masks[stem[5:]] = array
        else: arrays[stem] = array
    arrays.update(targets=targets, masks=masks, manifest=manifest,
                  label_end_ns=arrays['predictive_label_end_ns'],
                  group_columns={g: tuple(v['columns']) for g, v in manifest['groups'].items()})
    return arrays


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = prepare(args.config)
    print(json.dumps({'complete': result['complete'], 'rows': result['rows'], 'groups': result['groups']}, ensure_ascii=False))
