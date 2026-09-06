"""Causal raw-market inputs for a new WM/learned-RL procedure.

This module does not fit a neural model, predict a policy, acquire data, or
score outcomes. Features retain the inherited 29/31-column arithmetic and
broad common mask. Missing rows stay on the full grid. The unshifted market
return is a separate training target, never an input or inference-support gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from unidream.data.dataset import SequenceDataset
from .oracle_derivative_features import _raw_frame, make_derivative_groups
from .oracle_derivative_delay_features import make_delayed_perp_groups
from .oracle_frontier_features import make_feature_groups
from .oracle_risk_calibration import trailing_variances

INTERVAL = pd.Timedelta(minutes=15)
GROUPS = ("technical", "perp_delay0")
SCALE_FLOOR = 1e-8
FEATURE_SOURCES = (
    "alpha_dd_search.py", "alpha_dd_features.py", "oracle_frontier_features.py",
    "oracle_derivative_features.py", "oracle_derivative_delay_features.py",
    "oracle_risk_calibration.py", "wm_rl_inputs.py",
)


def _index(index):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("timezone-aware full-grid timestamps required")
    index = index.tz_convert("UTC")
    if (not len(index) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing
            or np.any(index.asi8 % INTERVAL.value)
            or np.any(np.diff(index.asi8) != INTERVAL.value)):
        raise ValueError("nonempty, unique, increasing complete 15-minute grid required")
    return index


def _cutoff(value):
    cutoff = pd.Timestamp(value)
    if pd.isna(cutoff) or cutoff.tz is None:
        raise ValueError("cutoff must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    if cutoff.value % INTERVAL.value:
        raise ValueError("cutoff must align to the 15-minute grid")
    return cutoff


def _mask(value, index, name):
    if isinstance(value, pd.Series) and not value.index.equals(index):
        raise ValueError(name + " Series index is unaligned")
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != (len(index),):
        raise ValueError(name + " must be an aligned one-dimensional boolean mask")
    return array.copy()


def mask_digest(index, mask):
    """Bind both nominal timestamps and every boolean value."""
    index = _index(index)
    mask = _mask(mask, index, "mask")
    digest = hashlib.sha256(index.asi8.astype("<i8").tobytes())
    digest.update(mask.astype(np.uint8).tobytes())
    return digest.hexdigest()


def _matrix_digest(array):
    array = np.asarray(array, dtype="<f8", order="C")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def _real_selected(values, mask, name):
    raw = np.asarray(values, dtype=object)
    selected = raw[mask]
    for value in selected.flat:
        if isinstance(value, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(name + " selected values must be real numbers, not bool/complex")
    result = np.asarray(selected, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(name + " selected values must be finite")
    return result


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates or not len(frame.columns):
        raise ValueError("nonempty feature DataFrame with unique columns required")
    if not all(isinstance(c, str) and c for c in frame.columns):
        raise ValueError("feature columns must be nonempty strings")
    return _index(frame.index)


def build_market_inputs(spot, um, *, cutoff, spot_observed=None):
    """Return full-grid features, raw targets, masks and source provenance.

    Inputs must already obey the caller's strict archival cutoff: no raw bar
    may open at/after it. Sparse UM rows are aligned to the full Spot grid by
    the inherited helper. An optional physical observation sidecar is retained
    separately from raw validity and feature eligibility. Receipt-time truth
    cannot be established from archived bar values.
    """
    cutoff = _cutoff(cutoff)
    spot = _raw_frame(spot, "Spot", complete=True)
    um = _raw_frame(um, "UM", complete=False)
    index = _index(spot.index)
    if index[-1] >= cutoff or (len(um) and um.index[-1] >= cutoff):
        raise ValueError("post-cutoff raw row; caller must provide only completed pre-cutoff bars")
    required = ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote", "n_trades")
    if "n_trades" not in um.columns:
        raise ValueError("UM requires the frozen n_trades raw schema")
    if any(name not in spot.columns for name in required):
        raise ValueError("Spot requires OHLCV, quote/taker and frozen n_trades raw fields")
    # Raw metadata is validated separately; no replacement is fed into the
    # inherited features, preserving their exact coverage arithmetic.
    for name, frame in (("Spot", spot), ("UM", um)):
        for column in (c for c in (*required, "n_trades", "trades") if c in frame):
            arr = frame[column].to_numpy()
            if np.iscomplexobj(arr) or pd.api.types.is_bool_dtype(arr.dtype):
                raise ValueError(name + " raw fields cannot be bool or complex")
            if arr.dtype == object:
                for value in arr:
                    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
                        raise ValueError(name + " raw fields cannot be bool or complex")
    original = make_feature_groups(spot)
    derivative = make_derivative_groups(spot, um)
    delayed = make_delayed_perp_groups(spot, um, delays=(0, 1, 4))
    components = {"trailing_variances": trailing_variances(spot, 24),
        "original_flow": original["flow"],
        **{"derivative_" + key: value for key, value in derivative.items()},
        **{"delayed_" + key: value for key, value in delayed.items()}}
    component_masks = {}
    for name, frame in components.items():
        if not frame.index.equals(index):
            raise ValueError("feature dependency changed the nominal grid")
        component_masks[name] = np.isfinite(frame.to_numpy()).all(axis=1)
    common = np.logical_and.reduce(list(component_masks.values()))
    groups = {name: delayed[name] for name in GROUPS}
    if [len(groups[g].columns) for g in GROUPS] != [29, 31]:
        raise ValueError("inherited feature dimensions changed")
    observed_values = spot.loc[:, required].notna().any(axis=1).to_numpy()
    observed = observed_values if spot_observed is None else _mask(spot_observed, index, "spot_observed")
    if np.any(~observed & observed_values):
        raise ValueError("unobserved Spot rows contain raw values")
    raw = spot.loc[:, required].astype(float)
    o, h, l, c = (raw[name] for name in ("open", "high", "low", "close"))
    price_valid = (np.isfinite(raw.loc[:, ("open", "high", "low", "close")]).all(axis=1)
        & o.gt(0) & h.gt(0) & l.gt(0) & c.gt(0)
        & h.ge(o) & h.ge(c) & l.le(o) & l.le(c) & h.ge(l)).to_numpy()
    raw_valid = (price_valid & np.isfinite(raw.volume) & raw.volume.ge(0)).to_numpy()
    quote_valid = (np.isfinite(raw.quote_volume) & raw.quote_volume.gt(0)
        & np.isfinite(raw.taker_buy_quote) & raw.taker_buy_quote.ge(0)
        & raw.taker_buy_quote.le(raw.quote_volume)).to_numpy()
    close_valid = (observed & np.isfinite(c) & c.gt(0)).to_numpy()
    returns = np.full(len(index), np.nan)
    target_valid = close_valid & np.r_[False, close_valid[:-1]]
    # Same-row market reward r[t], with both real consecutive closes required.
    positions = np.flatnonzero(target_valid)
    returns[positions] = np.log(c.to_numpy()[positions] / c.to_numpy()[positions - 1])
    target_valid &= np.isfinite(returns)
    returns[~target_valid] = np.nan
    um_presence = um.loc[:, ["close", "quote_volume", "taker_buy_quote"]].notna().any(axis=1)
    um_observed = um_presence.reindex(index, fill_value=False).to_numpy(dtype=bool)
    availability = pd.DataFrame({"spot_bar_observed": observed,
        "um_bar_observed": um_observed, "raw_ohlcv_valid": raw_valid,
        "raw_quote_flow_valid": quote_valid, "full_feature_eligible": common,
        "raw_target_validity": target_valid}, index=index)
    source_hashes = {"unidream/experiments/" + name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest() for name in FEATURE_SOURCES}
    contract = {"schema": "wm-rl-inputs-v1", "interval": "15m", "bar_timestamp": "open",
        "cutoff_exclusive": cutoff.isoformat(), "causal_shift_bars": 1,
        "groups": {name: list(frame.columns) for name, frame in groups.items()},
        "common_mask_components": list(components), "source_hashes": source_hashes,
        "raw_schema": {"Spot": list(spot.columns), "UM": list(um.columns)},
        "required_trade_column": "n_trades",
        "common_component_columns": {k: list(v.columns) for k, v in components.items()},
        "common_mask_sha256": mask_digest(index, common),
        "component_mask_sha256": {k: mask_digest(index, v) for k, v in component_masks.items()},
        "physical_observation_source": "explicit_sidecar" if spot_observed is not None else "raw_field_presence",
        "historical_receipt_provenance_established": False,
        "raw_target": "log(close[t]/close[t-1]); unshifted; not a feature",
        "normalization": "train-only population mean/std, scale floor 1e-8, no clipping"}
    return {"groups": groups, "full_feature_eligible": common.copy(),
        "spot_observed": observed.copy(), "raw_target_validity": target_valid.copy(),
        "returns": pd.Series(returns, index=index, name="market_log_return"),
        "availability": availability, "component_masks": component_masks, "source_contract": contract}



def build_inference_inputs(spot, um, *, origin):
    """Build the fixed64 causal feature context ending at decision ``origin``.

    Real raw inputs must open strictly before origin. Spot must retain every
    nominal15m row through origin-15m, including explicit NaN gaps. A single
    all-NaN Spot row is appended solely to expose the inherited one-bar shift
    at origin; no current OHLCV, labels or current-open value is consumed.
    The placeholder is physically unobserved and does not gate feature-only
    inference. Receipt timeliness and current-open eligibility remain the live
    caller's separate responsibilities.

    At least63 preceding nominal rows are required to return a64-row context.
    The90-day inherited warmup requires8704 preceding raw rows for all64
    feature rows to be eligible on a complete observation history. Shorter or
    gapped history remains unavailable; no value is filled or extrapolated.
    """
    origin = _cutoff(origin)
    if not isinstance(spot, pd.DataFrame) or not isinstance(um, pd.DataFrame):
        raise ValueError("raw Spot and UM DataFrames required")
    index = _index(spot.index)
    if not isinstance(um.index, pd.DatetimeIndex) or um.index.tz is None:
        raise ValueError("UM requires timezone-aware raw bar-open timestamps")
    # Reject supplied current/future rows before inspecting their raw values.
    if np.any(index >= origin) or np.any(um.index >= origin):
        raise ValueError("current or future raw row supplied to inference origin")
    if index[-1] != origin - INTERVAL:
        raise ValueError("Spot grid must end at origin minus15m; retain missing nominal rows")
    if len(index) < 63:
        raise ValueError("at least63 prior nominal rows required for fixed64 context")
    spot = _raw_frame(spot, "Spot", complete=True)
    um = _raw_frame(um, "UM", complete=False)
    placeholder_index = index.append(pd.DatetimeIndex([origin], name=index.name))
    extended = spot.reindex(placeholder_index)
    result = build_market_inputs(extended, um, cutoff=origin + INTERVAL)
    if result["spot_observed"][-1] or result["raw_target_validity"][-1]:
        raise ValueError("inference placeholder was incorrectly treated as observed")
    result.pop("returns")
    result.pop("raw_target_validity")
    result["availability"] = result["availability"].drop(columns="raw_target_validity")
    result["context_groups"] = {name: frame.iloc[-64:].copy() for name, frame in result["groups"].items()}
    result["context_feature_eligible"] = result["full_feature_eligible"][-64:].copy()
    result["inference_available"] = bool(result["context_feature_eligible"].all())
    contract = result["source_contract"]
    contract.pop("raw_target")
    contract.update(schema="wm-rl-inference-inputs-v1", cutoff_exclusive=origin.isoformat(),
        closed_raw_cutoff_exclusive=origin.isoformat(), decision_origin=origin.isoformat(),
        last_nominal_closed_bar_open=(origin - INTERVAL).isoformat(),
        nominal_feature_grid_end_inclusive=origin.isoformat(),
        placeholder_bar_open=origin.isoformat(), placeholder_has_market_values=False,
        placeholder_physically_observed=False, current_open_value_consumed=False,
        target_values_returned=False, context_length=64,
        complete_history_raw_rows_for64_features=8704,
        inference_available=result["inference_available"],
        live_receipt_timeliness_established=False)
    return result


def fit_normalizer(frame, *, train_mask, feature_eligible):
    """Fit only eligible training rows; labels never enter this operation."""
    index = _frame(frame)
    train = _mask(train_mask, index, "train_mask")
    eligible = _mask(feature_eligible, index, "feature_eligible")
    selected = train & eligible
    if not selected.any():
        raise ValueError("normalizer has no eligible training rows")
    values = _real_selected(frame.to_numpy(), selected, "training features")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("nonfinite normalizer moments")
    scale = np.maximum(std, SCALE_FLOOR)
    return {"schema": "wm-rl-normalizer-v1", "columns": list(frame.columns), "n": int(selected.sum()),
        "mean": mean.tolist(), "std": std.tolist(), "scale": scale.tolist(),
        "ddof": 0, "scale_floor": SCALE_FLOOR, "clip": None,
        "train_mask_sha256": mask_digest(index, train),
        "selected_mask_sha256": mask_digest(index, selected),
        "selected_features_sha256": _matrix_digest(values)}


def apply_normalizer(frame, normalizer, *, feature_eligible):
    """Use a frozen normalizer; preserve the full index and off-support NaNs."""
    index = _frame(frame)
    selected = _mask(feature_eligible, index, "feature_eligible")
    if (not isinstance(normalizer, Mapping) or normalizer.get("schema") != "wm-rl-normalizer-v1"
            or normalizer.get("columns") != list(frame.columns) or normalizer.get("ddof") != 0
            or normalizer.get("scale_floor") != SCALE_FLOOR or normalizer.get("clip") is not None):
        raise ValueError("changed normalizer schema, feature order or fixed arithmetic")
    moments = {}
    for name in ("mean", "std", "scale"):
        array = np.asarray(normalizer.get(name), dtype=object)
        if array.shape != (frame.shape[1],):
            raise ValueError("malformed normalizer " + name)
        moments[name] = _real_selected(array, np.ones(len(array), dtype=bool), "normalizer " + name)
    if np.any(moments["std"] < 0) or not np.array_equal(
            moments["scale"], np.maximum(moments["std"], SCALE_FLOOR)):
        raise ValueError("normalizer scale does not equal max(population std, 1e-8)")
    values = _real_selected(frame.to_numpy(), selected, "eligible features")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = (values - moments["mean"]) / moments["scale"]
    if not np.isfinite(scaled).all():
        raise ValueError("normalization produced nonfinite features")
    output = np.full(frame.shape, np.nan)
    output[selected] = scaled
    return pd.DataFrame(output, index=index, columns=frame.columns)


def target_horizon_mask(index, return_eligible, *, cutoff, horizon=64):
    """Require real r[t:t+horizon+1], all closed by cutoff (inclusive).

    Horizon zero is the same-row market reward; horizon64 is conservative
    support for every next64 auxiliary return. This label mask must never gate
    live inference, whose inputs require feature eligibility alone.
    """
    index = _index(index)
    valid = _mask(return_eligible, index, "return_eligible")
    cutoff = _cutoff(cutoff)
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    n = len(index)
    output = np.zeros(n, dtype=bool)
    if horizon >= n:
        return output
    starts = np.arange(n - horizon)
    invalid = np.r_[0, np.cumsum(~valid)]
    complete = invalid[starts + horizon + 1] == invalid[starts]
    mature = index[starts + horizon] + INTERVAL <= cutoff
    output[starts] = complete & mature
    return output


def sequence_masks(index, *, feature_eligible, target_eligible=None, row_mask=None, seq_len=64):
    """Return original full-grid starts/endpoints, never compressing time."""
    index = _index(index)
    if isinstance(seq_len, (bool, np.bool_)) or not isinstance(seq_len, (int, np.integer)) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    features = _mask(feature_eligible, index, "feature_eligible")
    eligible = features.copy()
    if target_eligible is not None:
        eligible &= _mask(target_eligible, index, "target_eligible")
    if row_mask is not None:
        eligible &= _mask(row_mask, index, "row_mask")
    n = len(index)
    starts = np.arange(max(n - seq_len + 1, 0), dtype=np.int64)
    invalid = np.r_[0, np.cumsum(~eligible)]
    starts = starts[invalid[starts + seq_len] == invalid[starts]]
    endpoints = np.zeros(n, dtype=bool)
    endpoints[starts + seq_len - 1] = True
    return {"row_eligible": eligible, "valid_starts": starts,
        "endpoint_eligible": endpoints, "row_mask_sha256": mask_digest(index, eligible),
        "endpoint_mask_sha256": mask_digest(index, endpoints)}


class MarketSequenceDataset(SequenceDataset):
    """Full-grid tensor storage with explicit WM/BC quality-window selection.

    Returned sample start offsets remain in the caller's original index.
    Physical Spot observation is independently checked if provided; it is not
    overwritten with model feature eligibility. Off-support NaNs remain NaNs.
    """
    def __init__(self, features, *, feature_eligible, target_eligible=None,
                 row_mask=None, seq_len=64, returns=None, actions=None,
                 regime_probs=None, spot_observed=None):
        index = _frame(features)
        masks = sequence_masks(index, feature_eligible=feature_eligible,
            target_eligible=target_eligible, row_mask=row_mask, seq_len=seq_len)
        eligible = masks["row_eligible"]
        if spot_observed is not None:
            physical = _mask(spot_observed, index, "spot_observed")
            eligible &= physical
            masks = sequence_masks(index, feature_eligible=eligible, seq_len=seq_len)
        matrix = np.full(features.shape, np.nan)
        matrix[eligible] = _real_selected(features.to_numpy(), eligible, "dataset features")
        if np.any(np.abs(matrix[eligible]) > np.finfo(np.float32).max):
            raise ValueError("dataset features overflow float32")
        def aligned(values, name):
            if values is None:
                return None
            if isinstance(values, (pd.Series, pd.DataFrame)) and not values.index.equals(index):
                raise ValueError(name + " index is unaligned")
            values = np.asarray(values, dtype=object)
            if values.ndim not in (1, 2) or len(values) != len(index):
                raise ValueError(name + " must align to every original row")
            output = np.full(values.shape, np.nan)
            output[eligible] = _real_selected(values, eligible, name)
            if np.any(np.abs(output[eligible]) > np.finfo(np.float32).max):
                raise ValueError(name + " overflows float32")
            return output
        availability = None if spot_observed is None else pd.DataFrame(
            {"spot_bar_observed": physical}, index=index)
        super().__init__(matrix, seq_len, actions=aligned(actions, "actions"),
            returns=aligned(returns, "returns"), regime_probs=aligned(regime_probs, "regime_probs"),
            timestamps=index, availability=availability, interval="15m",
            include_funding=False, include_mark=False)
        self._valid_starts = masks["valid_starts"].copy()
        self._row_eligible = eligible.copy()
        self.input_masks = masks
        self.source_timestamps = index.copy()


__all__ = ["GROUPS", "SCALE_FLOOR", "build_market_inputs", "build_inference_inputs", "fit_normalizer", "apply_normalizer",
    "mask_digest", "target_horizon_mask", "sequence_masks", "MarketSequenceDataset"]
