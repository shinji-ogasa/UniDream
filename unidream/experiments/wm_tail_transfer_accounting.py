"""Separate tail-transfer study accounts; no production account is modified.

Cash borrowing delegates to the immutable continuous-calendar research account.
The linear-perpetual alternative is a fixed-parameter, isolated, long-only
research model, not a reconstruction of historical exchange margin rules.
Within each bar: prior pending order at open, funding, mark-low maintenance
test, mark-close valuation. A liquidation uses the trade bar low and an
additional liquidation slippage, then keeps any residual cash permanently.
"""
from __future__ import annotations

import math
import torch

from . import wm_control_release_accounting as cash_account


BASE_INVENTORY_NAMES = cash_account.INVENTORY_KEYS
EXTRA_INVENTORY_NAMES = ("cash_ratio", "debt_ratio", "collateral_ratio", "unrealized_ratio",
                         "maintenance_fraction", "margin_buffer", "has_pending", "alive")


def _mode(cfg):
    mode = cfg.get("accounting", cfg).get("mode", "cash_borrow")
    if mode not in ("cash_borrow", "linear_perp"):
        raise ValueError("account mode must be cash_borrow or linear_perp")
    return mode


def _perp(cfg):
    a, p = cfg.get("accounting", cfg), cfg.get("perpetual", {})
    cash_account._settings(cfg)
    result = {"fee": float(p.get("fee_rate", a.get("cost_rate", .00055))),
              "mmr": float(a.get("maintenance_margin_rate", p.get("primary_maintenance_margin_rate", .005))),
              "slippage": float(a.get("slippage_rate", p.get("primary_slippage_rate", .0005))),
              "liq_fee": float(p.get("liquidation_fee_rate", .005)),
              "liq_slippage": float(p.get("liquidation_slippage_rate", a.get("slippage_rate", p.get("primary_slippage_rate", .0005))))}
    if not all(math.isfinite(v) and 0 <= v < 1 for v in result.values()) or not 0 < result["mmr"] < 1:
        raise ValueError("finite fractional perpetual costs and positive maintenance margin required")
    if p.get("margin_mode", "isolated") != "isolated":
        raise ValueError("only fixed-parameter isolated research margin is supported")
    return result


def initial_state(batch, device, price, cfg, dtype=torch.float64):
    state = cash_account.initial_state(batch, device, price, cfg, dtype=dtype)
    if _mode(cfg) == "linear_perp":
        _perp(cfg)
        state.update(entry_price=state["prev_close"].clone(),
                     last_funding_event=torch.full((int(batch),), -1, device=device, dtype=torch.int64),
                     liquidated=torch.zeros(int(batch), device=device, dtype=torch.bool))
    return state


def inventory(state, cfg=None, expanded=False):
    base = cash_account.inventory(state)
    if not expanded:
        return base
    equity = state["prev_equity"].clamp_min(1e-12)
    notional = state["units"] * state["prev_close"]
    is_perp = "entry_price" in state
    collateral = state["cash"] + state["units"] * state["entry_price"] if is_perp else equity
    unrealized = notional - state["units"] * state["entry_price"] if is_perp else torch.zeros_like(equity)
    mmr = _perp(cfg)["mmr"] if is_perp and cfg is not None else 0.
    maintenance = notional * mmr / equity
    extra = torch.stack((state["cash"] / equity, (-state["cash"]).clamp_min(0) / equity if not is_perp else torch.zeros_like(equity),
                         collateral / equity, unrealized / equity, maintenance,
                         (1. - maintenance), state["has_pending"].to(equity.dtype), state["alive"].to(equity.dtype)), -1)
    return torch.cat((base, extra.clamp(-20, 20)), -1)


def step(state, new_request, open_price, low_price, close_price, cfg, *, decision_available=None,
         mark_open=None, mark_low=None, mark_close=None, funding_rate=0., funding_event=None,
         funding_event_count=None):
    if _mode(cfg) == "cash_borrow":
        updated, diagnostics = cash_account.step(state, new_request, open_price, low_price, close_price, cfg,
                                                 decision_available=decision_available)
        diagnostics.update(funding=torch.zeros_like(updated["cash"]), liquidation_fee=torch.zeros_like(updated["cash"]),
                           liquidation_event=torch.zeros_like(updated["alive"]), liquidated=torch.zeros_like(updated["alive"]))
        return updated, diagnostics
    return _perp_step(state, new_request, open_price, low_price, close_price, cfg,
                      decision_available=decision_available, mark_open=mark_open, mark_low=mark_low,
                      mark_close=mark_close, funding_rate=funding_rate, funding_event=funding_event,
                      funding_event_count=funding_event_count)


@torch.no_grad()
def _perp_step(state, new_request, open_price, low_price, close_price, cfg, *, decision_available,
               mark_open, mark_low, mark_close, funding_rate, funding_event, funding_event_count):
    a, p = cash_account._settings(cfg), _perp(cfg)
    if set(state) != set(cash_account.STATE_KEYS) | {"entry_price", "last_funding_event", "liquidated"}:
        raise ValueError("complete isolated perpetual account state required")
    cash, units, alive = state["cash"], state["units"], state["alive"]
    if cash.ndim != 1 or not len(cash) or any(v.shape != cash.shape for v in state.values()):
        raise ValueError("matched nonempty batched account tensors required")
    vector = lambda value, name: cash_account._vector(value, cash, name)
    opening, low, close = [vector(v, n) for v, n in ((open_price, "open"), (low_price, "low"), (close_price, "close"))]
    if any(v is None for v in (mark_open, mark_low, mark_close)):
        raise ValueError("perpetual maintenance requires independently supplied mark OHLC")
    mo, ml, mc = [vector(v, n) for v, n in ((mark_open, "mark_open"), (mark_low, "mark_low"), (mark_close, "mark_close"))]
    prices = torch.stack((opening, low, close, mo, ml, mc), -1)
    valid = torch.isfinite(prices).all(-1) & (prices > 0).all(-1) & (low <= torch.minimum(opening, close)) & (ml <= torch.minimum(mo, mc))
    if not bool((~alive | valid).all()):
        raise ValueError("alive perpetual accounts require valid trade and mark open/low/close")
    # Dead accounts hold their residual quote collateral; unavailable future
    # market observations must not manufacture NaNs in these inactive rows.
    opening, low, close, mo, ml, mc = [torch.where(alive, v, state["prev_close"]) for v in (opening, low, close, mo, ml, mc)]
    available = torch.ones_like(alive) if decision_available is None else torch.as_tensor(decision_available, device=cash.device)
    if available.ndim == 0:
        available = available.expand_as(alive)
    if available.dtype != torch.bool or available.shape != alive.shape:
        raise ValueError("decision availability must be boolean scalar or [B]")
    request = vector(new_request, "request")
    active = alive & available
    feasible = torch.isfinite(request) & (request >= a["position_min"]-1e-6) & (request <= a["position_max"]+1e-6) & ((request-state["requested"]).abs() <= a["max_position_step"]+1e-6)
    if not bool((~active | feasible).all()):
        raise ValueError("perpetual order target violates bounds or target-change rate")
    rate = vector(funding_rate, "funding_rate")
    count = (rate != 0).to(torch.int64) if funding_event_count is None else torch.as_tensor(funding_event_count, device=cash.device)
    if count.ndim == 0:
        count = count.expand_as(alive)
    if count.shape != alive.shape or count.dtype == torch.bool or not bool(((count >= 0) & (count == count.long())).all()):
        raise ValueError("nonnegative integer funding-event counts required")
    event = count > 0
    if not bool(torch.isfinite(rate).all()) or not bool((event | (rate == 0)).all()):
        raise ValueError("funding rates must be finite and absent events must have zero rate")
    if funding_event is None and bool((alive & event).any()):
        raise ValueError("actual funding settlements require chronological unique event ids")
    event_id = torch.full_like(state["last_funding_event"], -1) if funding_event is None else torch.as_tensor(funding_event, device=cash.device, dtype=torch.int64)
    if event_id.ndim == 0:
        event_id = event_id.expand_as(alive)
    if event_id.shape != alive.shape or not bool((~(alive & event) | (event_id > state["last_funding_event"])).all()):
        raise ValueError("funding event already settled or event ids are not chronological")
    zero, one = torch.zeros_like(cash), torch.ones_like(cash)
    nav_open = cash + units * mo
    open_maintenance = units * mo * p["mmr"]
    gap_liquidation = alive & ((nav_open <= open_maintenance) | (nav_open <= 0))
    fill = alive & ~gap_liquidation & state["has_pending"]
    numerator = state["requested"] * nav_open - units * mo
    side = torch.sign(numerator)
    fill_price = opening * (1 + p["slippage"] * side)
    denominator = mo-state["requested"]*(mo-fill_price)+state["requested"]*p["fee"]*fill_price*side
    if not bool((~fill | (torch.isfinite(denominator) & (denominator > 0))).all()):
        raise ValueError("perpetual target cannot be financed at trade/mark basis and costs")
    trade_units = torch.where(fill, numerator / torch.where(fill, denominator, one), zero)
    traded_units = (units + trade_units).clamp_min(0)
    notional = trade_units * fill_price
    fee = notional.abs() * p["fee"]
    traded_cash = cash-notional-fee
    entry = torch.where(trade_units > 0, (units*state["entry_price"]+trade_units*fill_price)/traded_units.clamp_min(1e-12), state["entry_price"])
    entry = torch.where(traded_units > 0, entry, zero)
    funding = torch.where(alive & ~gap_liquidation & event, traded_units * mo * rate, zero)
    funded_cash = traded_cash-funding
    nav_low, nav_close = funded_cash+traded_units*ml, funded_cash+traded_units*mc
    low_liquidation = alive & ~gap_liquidation & (nav_low <= p["mmr"]*traded_units*ml)
    liquidation = gap_liquidation | low_liquidation
    exit_price = torch.where(gap_liquidation, opening, low) * (1-p["liq_slippage"])
    liquidation_notional = torch.where(liquidation, traded_units*exit_price, zero)
    liquidation_fee = liquidation_notional*p["liq_fee"]
    residual = (funded_cash+liquidation_notional-liquidation_fee).clamp_min(0)
    next_alive = alive & ~liquidation
    equity = torch.where(liquidation, residual, torch.where(alive, nav_close, state["prev_equity"]))
    if not bool((torch.isfinite(equity) & (equity >= 0)).all()):
        raise ValueError("finite nonnegative perpetual equity required")
    next_cash = torch.where(liquidation, residual, funded_cash)
    next_units = torch.where(next_alive, traded_units, zero)
    ratio = torch.where(alive, equity/state["prev_equity"].clamp_min(1e-12), one)
    net_log, training_log = torch.log(ratio), torch.log(ratio.clamp_min(1e-12))
    wealth = torch.log(equity)
    peak = torch.maximum(state["peak_log_wealth"], wealth)
    dd = -torch.expm1(wealth-peak)
    max_dd = torch.maximum(state["max_drawdown"], dd)
    accepted = torch.where(active, request, state["requested"])
    accepted = torch.where(next_alive, accepted, zero)
    exposure = torch.where(next_alive, next_units*mc/equity.clamp_min(1e-12), zero)
    updated = {"requested": accepted, "executed": exposure, "previous_delta": torch.where(active & next_alive, accepted-state["requested"], zero),
               "log_wealth": wealth, "peak_log_wealth": peak, "drawdown": dd, "max_drawdown": max_dd,
               "cash": next_cash, "units": next_units, "prev_equity": equity, "prev_close": mc,
               "alive": next_alive, "has_pending": active & next_alive,
               "entry_price": torch.where(next_alive, entry, zero),
               "last_funding_event": torch.where(alive & event & ~gap_liquidation, event_id, state["last_funding_event"]),
               "liquidated": state["liquidated"] | liquidation}
    bankrupt_event = alive & (equity <= 0)
    diagnostics = {"equity": equity, "net_simple_return": ratio-1, "net_log_return": net_log,
                   "training_log_return": training_log, "executed_exposure": exposure, "executed_position": exposure,
                   "open_exposure": torch.where(alive & ~gap_liquidation, traded_units*mo/(traded_cash+traded_units*mo).clamp_min(1e-12), zero),
                   "close_exposure": exposure, "turnover": torch.where(fill, notional.abs()/nav_open.clamp_min(1e-12), zero),
                   "fee": fee, "borrow": zero, "funding": funding, "liquidation_fee": liquidation_fee,
                   "drawdown": dd, "max_drawdown": max_dd, "max_dd_increment": max_dd-state["max_drawdown"],
                   "bankrupt": equity <= 0, "bankrupt_event": bankrupt_event, "alive": next_alive,
                   "liquidation_event": liquidation, "liquidated": updated["liquidated"],
                   "new_request": accepted, "requested_position": accepted, "has_pending": updated["has_pending"],
                   "decision_available": active, "order_filled": fill, "trade_notional": notional,
                   "nav_open": nav_open, "nav_low": nav_low, "nav_close_before_absorption": nav_close,
                   "cash": next_cash, "units": next_units, "maintenance_requirement": p["mmr"]*traded_units*ml,
                   "liquidation_fill_price": torch.where(liquidation, exit_price, zero), "funding_event_count": torch.where(alive & event & ~gap_liquidation, count.long(), torch.zeros_like(count.long()))}
    return updated, diagnostics
