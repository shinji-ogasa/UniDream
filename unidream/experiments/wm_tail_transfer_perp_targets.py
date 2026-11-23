"""Execution-aware passive perpetual labels for the isolated tail study.

These are fixed-parameter research USDT-linear long-only margin labels, not
historical exchange liquidation records.  They deliberately do not rename the
separate cash-borrow ``ruin`` label.  A decision at d enters at trade open d+1,
then holds fixed units through h scored bars; there is no terminal exit.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from . import wm_tail_transfer_accounting as account
from . import wm_tail_transfer_data as data_module


VERSION = "research_linear_perp_passive_v1"


def _account_config(cfg):
    result = copy.deepcopy(cfg)
    maximum = max(1., max(float(v) for v in cfg["targets"]["passive_leverages"]))
    result.setdefault("accounting", {}).update(mode="linear_perp", position_min=0.,
        position_max=maximum, max_position_step=maximum, initial_position=1.,
        execution_delay_bars=1, reset_positions_at_segment_boundary=False,
        initial_entry_cost=False, terminal_liquidation=False, deadband=0.)
    result["accounting"]["maintenance_margin_rate"] = float(cfg["perpetual"]["primary_maintenance_margin_rate"])
    result["accounting"]["slippage_rate"] = float(cfg["perpetual"]["primary_slippage_rate"])
    return result


def build_perpetual_targets(data, cfg, *, chunk_rows=8192, progress=None):
    """Return [decision,horizon,leverage] targets and full-future masks.

    Invalid future trade/mark data invalidate all affected labels even when an
    account has already liquidated. Sanitized placeholders are used only in
    the inactive mathematical computation, never exposed as observed labels.
    """
    horizons = [int(h) for h in cfg["targets"]["horizons"]]
    leverages = np.asarray(cfg["targets"]["passive_leverages"], dtype=np.float64)
    if not horizons or min(horizons) <= 0 or len(set(horizons)) != len(horizons):
        raise ValueError("unique positive passive horizons required")
    if leverages.ndim != 1 or not len(leverages) or not np.isfinite(leverages).all() or (leverages < 0).any():
        raise ValueError("finite nonnegative passive leverages required")
    if int(chunk_rows) != chunk_rows or chunk_rows <= 0:
        raise ValueError("positive integer chunk size required")
    index = np.asarray(data["index_ns"], dtype=np.int64)
    n, nl = len(index), len(leverages)
    if not n or not np.all(np.diff(index) == data_module.BAR_NS):
        raise ValueError("complete nonempty calendar required")
    trade, mark = np.asarray(data["ohlcperp"]), np.asarray(data["ohlcmark"])
    rates = np.asarray(data["funding_settlement_rate"], dtype=np.float64)
    counts = np.asarray(data["funding_event_count"])
    if trade.shape[0] != n or mark.shape[0] != n or trade.shape[1] < 4 or mark.shape[1] < 4 or rates.shape != (n,) or counts.shape != (n,):
        raise ValueError("matched full-calendar perpetual, mark and funding arrays required")
    count_valid = np.isfinite(counts) & (counts >= 0) & (counts == counts.astype(np.int64))
    event_valid = count_valid & np.isfinite(rates) & ((counts > 0) | (rates == 0))
    prices = np.column_stack((trade[:, 0], trade[:, 2], trade[:, 3], mark[:, 0], mark[:, 2], mark[:, 3]))
    price_valid = (np.isfinite(prices).all(1) & (prices > 0).all(1)
                   & (prices[:, 1] <= np.minimum(prices[:, 0], prices[:, 2]))
                   & (prices[:, 4] <= np.minimum(prices[:, 3], prices[:, 5])))
    observed = price_valid & event_valid
    shape = (n, len(horizons), nl)
    targets = {name: np.full(shape, np.nan, np.float32) for name in ("perp_liquidation", "perp_passive_return")}
    mask = np.zeros(shape, bool)
    acfg = _account_config(cfg)
    device = "cpu"
    for begin in range(0, n, int(chunk_rows)):
        decisions = np.arange(begin, min(n, begin + int(chunk_rows)))
        first = np.minimum(decisions + 1, n - 1)
        first_price = np.where(observed[first], mark[first, 0], 1.)
        initial_price = np.repeat(first_price, nl)
        state = account.initial_state(len(initial_price), device, initial_price, acfg)
        # The shared trading account starts endowed 1x. Passive outcome labels
        # instead start entirely in collateral and queue the decision-d entry.
        # No price exposure exists before the actual open[d+1] acquisition.
        state["cash"].fill_(1.)
        for name in ("units", "executed", "entry_price"):
            state[name].zero_()
        state["requested"] = torch.from_numpy(np.tile(leverages, len(decisions)))
        state["has_pending"].fill_(True)
        valid = np.ones(len(decisions), bool)
        for offset in range(1, max(horizons) + 1):
            row = decisions + offset
            safe = np.minimum(row, n - 1)
            valid &= (row < n) & observed[safe]
            usable = observed[safe]
            values = np.where(usable[:, None], prices[safe], 1.)
            opening, low, close, mo, ml, mc = [np.repeat(values[:, k], nl) for k in range(6)]
            rate = np.repeat(np.where(usable, rates[safe], 0.), nl)
            count = np.repeat(np.where(usable, counts[safe], 0).astype(np.int64), nl)
            state, _ = account.step(state, float("nan"), opening, low, close, acfg,
                decision_available=False, mark_open=mo, mark_low=ml, mark_close=mc,
                funding_rate=rate, funding_event=np.repeat(index[safe], nl), funding_event_count=count)
            if offset in horizons:
                hi = horizons.index(offset)
                observed_here = np.broadcast_to(valid[:, None], (len(decisions), nl))
                liquidation = state["liquidated"].numpy().reshape(-1, nl)
                net = state["prev_equity"].numpy().reshape(-1, nl) - 1.
                targets["perp_liquidation"][decisions, hi] = np.where(observed_here, liquidation, np.nan)
                targets["perp_passive_return"][decisions, hi] = np.where(observed_here, net, np.nan)
                mask[decisions, hi] = observed_here
        if progress is not None:
            progress(min(n, begin + int(chunk_rows)), n)
    return targets, {name: mask.copy() for name in targets}


def prepare_perpetual_targets(cfg):
    """Append new labels to the new study manifest, preserving a snapshot."""
    cfg = data_module._cfg(cfg)
    out = data_module._out(cfg)
    prepared = out / "prepared"
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    existing = manifest.get("perpetual_targets", {})
    settings = dict(version=VERSION, horizons=cfg["targets"]["horizons"],
        leverages=cfg["targets"]["passive_leverages"], accounting=account._perp(_account_config(cfg)),
        execution="decision_d_enters_trade_open_d_plus_1_then_fixed_units_no_terminal_exit",
        initial_equity=1., initial_collateral=1., initial_units=0.,
        liquidation="mark_open_gap_then_mark_low_maintenance_crossing_trade_open_or_low_exit",
        residual="nonnegative_residual_collateral_kept_after_liquidation;NAV_zero_only_is_bankruptcy",
        funding="actual_events_bucketed_to15m;aggregate_rate_once_on_mark_open_notional_after_entry_before_low",
        intrabar_assumption="funding_before_low;independent_mark_and_trade_low_pairing_conservative_not_observed_order",
        future_availability="all_h_future_trade_and_mark_open_low_close_and_funding_valid_even_after_liquidation",
        label_end="existing_passive_label_end_ns;d_plus_1_plus_h_bar_open_right_exclusive",
        scope="fixed_parameter_research_not_historical_exchange_reconstruction",
        source_artifacts={name: manifest["artifacts"][name] for name in (
            "index_ns.npy", "ohlcperp.npy", "ohlcmark.npy", "funding_settlement_rate.npy",
            "funding_event_count.npy", "passive_label_end_ns.npy")})
    if existing:
        if existing != settings:
            raise ValueError("registered perpetual label settings changed")
        for name in ("perp_liquidation", "perp_passive_return"):
            for prefix in ("target_", "mask_"):
                filename = prefix + name + ".npy"
                if data_module._sha(prepared / filename) != manifest["artifacts"][filename]:
                    raise ValueError("perpetual target artifact changed")
        return manifest
    snapshot = out / "validation" / "manifest_before_perpetual_targets.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    original = manifest_path.read_bytes()
    if snapshot.exists() and snapshot.read_bytes() != original:
        raise ValueError("pre-perpetual manifest snapshot already exists with other bytes")
    snapshot.write_bytes(original)
    for filename, digest in settings["source_artifacts"].items():
        if data_module._sha(prepared / filename) != digest:
            raise ValueError("perpetual label source artifact changed")
    arrays = data_module.load_data(cfg)
    expected_end = np.asarray(arrays["index_ns"])[:, None] + (1 + np.asarray(cfg["targets"]["horizons"])[None]) * data_module.BAR_NS
    if not np.array_equal(arrays["passive_label_end_ns"], expected_end):
        raise ValueError("passive execution-aware label boundaries differ")
    def progress(done, total):
        data_module._json(out / "perpetual_targets_status.json", dict(completed=done, total=total, version=VERSION))
    targets, masks = build_perpetual_targets(arrays, cfg, progress=progress)
    for name, value in targets.items():
        for prefix, array in (("target_", value), ("mask_", masks[name])):
            path = prepared / (prefix + name + ".npy")
            np.save(path, array)
            manifest["artifacts"][path.name] = data_module._sha(path)
        manifest["targets"][name] = dict(shape=list(value.shape[1:]), dtype=str(value.dtype), valid=int(masks[name].sum()))
    manifest["perpetual_targets"] = settings
    manifest["label_definitions"]["perp_liquidation"] = "research_isolated_mark_maintenance_liquidation_with_residual;distinct_from_cash_borrow_ruin"
    manifest["label_definitions"]["perp_passive_return"] = "net_collateral_return_including_entry_fee_slippage_actual_funding_liquidation_fee;no_terminal_exit"
    data_module._json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(data_module.DEFAULT_CONFIG))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = prepare_perpetual_targets(args.config)
    print(json.dumps({"complete": True, "rows": result["rows"], "targets": {k: result["targets"][k]
        for k in ("perp_liquidation", "perp_passive_return")}}, ensure_ascii=False))
