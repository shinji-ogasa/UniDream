"""Long-only cash/asset-units research account with fixed-APR cash borrowing.

This is not an exchange-margin or futures-liquidation simulator. Exposure caps
bound order targets, not price-induced actual leverage. A nonpositive open/low/
close NAV conservatively ends the research account at equity0; there is no
simulated liquidation fill or claim that remaining debt was economically paid.
Every call advances one15-minute calendar bar, including no-decision bars.
"""
from __future__ import annotations

import math

import torch


STATE_KEYS = ("requested", "executed", "previous_delta", "log_wealth", "peak_log_wealth", "drawdown",
              "max_drawdown", "cash", "units", "prev_equity", "prev_close", "alive", "has_pending")
INVENTORY_KEYS = ("requested", "executed", "previous_delta", "log_wealth", "drawdown", "max_drawdown")


def _settings(cfg):
    supplied = cfg.get("accounting", cfg)
    if (supplied.get("execution_delay_bars", 1) != 1
            or supplied.get("initial_entry_cost", False) is not False
            or supplied.get("terminal_liquidation", False) is not False
            or supplied.get("reset_positions_at_segment_boundary", False) is not False
            or float(supplied.get("deadband", 0.)) != 0.):
        raise ValueError("one-bar execution, no artificial entry/exit, no reset and no deadband required")
    result = {"position_min": 0., "position_max": 1.12, "max_position_step": .08,
              "initial_position": 1., "cost_rate": .00055, "borrow_apr": .10, "annual_bars": 35040.}
    result.update({key: float(supplied[key]) for key in result if key in supplied})
    if (not all(math.isfinite(value) for value in result.values()) or result["position_min"] < 0
            or result["position_min"] >= result["position_max"]
            or not result["position_min"] <= 1. <= result["position_max"]
            or result["initial_position"] != 1. or result["max_position_step"] <= 0
            or not 0 <= result["cost_rate"] < 1 or result["cost_rate"] * result["position_max"] >= 1
            or result["borrow_apr"] < 0 or result["annual_bars"] <= 0):
        raise ValueError("valid long-only target bounds, fee/borrow and endowed1x account required")
    return result


def _vector(value, like, name):
    tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand_as(like)
    if tensor.shape != like.shape:
        raise ValueError(f"{name} must be scalar or one value per account")
    return tensor


@torch.no_grad()
def initial_state(batch, device, price, cfg, dtype=torch.float64):
    """Equity1 already endowed with BTC1x, cash0, and no entry order or fee."""
    _settings(cfg)
    if isinstance(batch, bool) or int(batch) != batch or batch <= 0 or not dtype.is_floating_point:
        raise ValueError("positive integer batch and floating account dtype required")
    ones = torch.ones(int(batch), device=device, dtype=dtype)
    price = _vector(price, ones, "initial price")
    if not bool((torch.isfinite(price) & (price > 0)).all()):
        raise ValueError("positive finite initial BTC price required")
    zero = torch.zeros_like(ones)
    return {"requested": ones.clone(), "executed": ones.clone(), "previous_delta": zero.clone(),
        "log_wealth": zero.clone(), "peak_log_wealth": zero.clone(), "drawdown": zero.clone(),
        "max_drawdown": zero.clone(), "cash": zero.clone(), "units": ones / price,
        "prev_equity": ones.clone(), "prev_close": price.clone(),
        "alive": torch.ones(int(batch), device=device, dtype=torch.bool),
        "has_pending": torch.zeros(int(batch), device=device, dtype=torch.bool)}


@torch.no_grad()
def inventory(state):
    """Six own-account features; absorbed accounts have finite inactive inputs.

    The performance state retains log_wealth=-inf at ruin. Only the inactive
    actor observation substitutes0 for this undefined dead-account feature.
    """
    fields = [state[key] for key in INVENTORY_KEYS]
    fields[3] = torch.where(state["alive"], fields[3], torch.zeros_like(fields[3]))
    return torch.stack(fields, dim=-1)


@torch.no_grad()
def step(state, new_request, open_price, low_price, close_price, cfg, *, decision_available=None):
    """Fill only the prior pending target; queue today's decision for next open.

    Invalid market features stop new orders, while existing pending orders and
    all cash/units/borrowing continue. No gap or fold reset exists in this API.
    Turnover is actual trade notional / pre-trade open NAV, not target changes.
    """
    a = _settings(cfg)
    if set(state) != set(STATE_KEYS):
        raise ValueError("complete cash/units account state required")
    cash, units = state["cash"], state["units"]
    if cash.ndim != 1 or not len(cash) or any(value.shape != cash.shape for value in state.values()):
        raise ValueError("all account state values must have matched one-dimensional shapes")
    if state["alive"].dtype != torch.bool or state["has_pending"].dtype != torch.bool:
        raise ValueError("alive and pending state must be boolean")
    alive = state["alive"]
    opening, low, close = (_vector(value, cash, name) for value, name in
                          ((open_price, "open"), (low_price, "low"), (close_price, "close")))
    raw_request = _vector(new_request, cash, "new request")
    if decision_available is None:
        available = torch.ones_like(alive)
    else:
        available = torch.as_tensor(decision_available, device=cash.device)
        if available.ndim == 0:
            available = available.expand_as(alive)
        if available.dtype != torch.bool or available.shape != alive.shape:
            raise ValueError("decision availability must be boolean scalar or[B]")
    valid_prices = (torch.isfinite(opening) & torch.isfinite(low) & torch.isfinite(close)
                    & (opening > 0) & (low > 0) & (close > 0)
                    & (low <= torch.minimum(opening, close)))
    if not bool((~alive | valid_prices).all()):
        raise ValueError("alive accounts require positive finite open/low/close with valid low")
    active_decision = alive & available
    valid_requests = (torch.isfinite(raw_request)
        & (raw_request >= a["position_min"] - 1e-6) & (raw_request <= a["position_max"] + 1e-6)
        & ((raw_request - state["requested"]).abs() <= a["max_position_step"] + 1e-6))
    if not bool((~active_decision | valid_requests).all()):
        raise ValueError("new alive/available request violates finite target bounds or target-change rate")
    valid_state = (torch.isfinite(cash) & torch.isfinite(units) & (units >= 0)
                   & torch.isfinite(state["prev_equity"]) & (state["prev_equity"] > 0)
                   & torch.isfinite(state["requested"]) & torch.isfinite(state["prev_close"])
                   & (state["prev_close"] > 0))
    if not bool((~alive | valid_state).all()):
        raise ValueError("alive cash/units state must be finite with positive previous equity")
    # Inactive rows may contain unavailable external prices, but no arithmetic
    # may create NaNs that can contaminate a batched actor/critic computation.
    one, zero = torch.ones_like(cash), torch.zeros_like(cash)
    opening = torch.where(alive, opening, state["prev_close"])
    low = torch.where(alive, low, state["prev_close"])
    close = torch.where(alive, close, state["prev_close"])
    cash = torch.where(alive, cash, zero)
    units = torch.where(alive, units, zero)
    nav_open = cash + units * opening
    if not bool((~alive | torch.isfinite(nav_open)).all()):
        raise ValueError("nonfinite open NAV")
    fill = alive & state["has_pending"] & (nav_open > 0)
    target = state["requested"]
    numerator = target * nav_open - units * opening
    denominator = 1 + a["cost_rate"] * target * torch.sign(numerator)
    if not bool((~fill | (torch.isfinite(denominator) & (denominator > 0))).all()):
        raise ValueError("fee-inclusive target cannot be financed")
    trade = torch.where(fill, numerator / torch.where(fill, denominator, one), zero)
    fee = a["cost_rate"] * trade.abs()
    traded_cash = cash - trade - fee
    traded_units = units + trade / opening
    if bool((alive & (traded_units < -1e-10)).any()):
        raise ValueError("target order would create a short BTC position")
    traded_units = traded_units.clamp_min(0)
    post_fill_nav = traded_cash + traded_units * opening
    open_exposure = torch.where(alive & (post_fill_nav > 0),
        traded_units * opening / torch.where(post_fill_nav > 0, post_fill_nav, one), zero)
    # One charge for every15-minute calendar bar, whether or not an order exists.
    borrow = torch.where(alive & (nav_open > 0), (-traded_cash).clamp_min(0)
                         * math.expm1(a["borrow_apr"] / a["annual_bars"]), zero)
    financed_cash = traded_cash - borrow
    nav_low = financed_cash + traded_units * low
    nav_close = financed_cash + traded_units * close
    if not bool((~alive | (torch.isfinite(nav_low) & torch.isfinite(nav_close))).all()):
        raise ValueError("nonfinite low/close NAV")
    bankrupt_event = alive & ((nav_open <= 0) | (nav_low <= 0) | (nav_close <= 0))
    next_alive = alive & ~bankrupt_event
    equity = torch.where(next_alive, nav_close, zero)
    next_cash = torch.where(next_alive, financed_cash, zero)
    next_units = torch.where(next_alive, traded_units, zero)
    ratio = torch.where(alive, equity / torch.where(state["prev_equity"] > 0, state["prev_equity"], one), one)
    net_simple = ratio - 1
    net_log = torch.log(ratio)
    training_log = torch.log(ratio.clamp_min(1e-12))
    log_wealth = torch.log(equity)
    peak_log = torch.maximum(state["peak_log_wealth"], log_wealth)
    drawdown = -torch.expm1(log_wealth - peak_log)
    max_drawdown = torch.maximum(state["max_drawdown"], drawdown)
    dd_increment = max_drawdown - state["max_drawdown"]
    accepted = torch.where(active_decision, raw_request.clamp(a["position_min"], a["position_max"]), state["requested"])
    accepted = torch.where(next_alive, accepted, zero)
    delta = torch.where(active_decision & next_alive, accepted - state["requested"], zero)
    close_exposure = torch.where(next_alive, next_units * close / torch.where(equity > 0, equity, one), zero)
    updated = {"requested": accepted, "executed": close_exposure, "previous_delta": delta,
        "log_wealth": log_wealth, "peak_log_wealth": peak_log, "drawdown": drawdown,
        "max_drawdown": max_drawdown, "cash": next_cash, "units": next_units,
        "prev_equity": equity, "prev_close": close, "alive": next_alive,
        "has_pending": active_decision & next_alive}
    turnover = torch.where(fill, trade.abs() / torch.where(nav_open > 0, nav_open, one), zero)
    diagnostics = {"equity": equity, "net_simple_return": net_simple, "net_log_return": net_log,
        "training_log_return": training_log, "executed_exposure": close_exposure,
        "executed_position": close_exposure, "open_exposure": open_exposure, "close_exposure": close_exposure,
        "turnover": turnover, "fee": fee, "borrow": borrow,
        "drawdown": drawdown, "max_drawdown": max_drawdown, "max_dd_increment": dd_increment,
        "bankrupt": ~next_alive, "bankrupt_event": bankrupt_event, "alive": next_alive,
        "new_request": accepted, "requested_position": accepted, "has_pending": updated["has_pending"],
        "decision_available": active_decision, "order_filled": fill, "trade_notional": trade,
        "nav_open": nav_open, "nav_low": nav_low, "nav_close_before_absorption": nav_close,
        "cash": next_cash, "units": next_units}
    return updated, diagnostics
