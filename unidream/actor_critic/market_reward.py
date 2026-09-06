"""Opt-in market-WM accounting; no learned policy or market model lives here.

A step fills at the current imagined price, charges cash borrowing for one
15-minute period, then marks assets using a predicted market log return.  This
is a close-to-close imagination approximation, not an OHLC execution simulator.
Cash and marked asset value persist; passive exposure drift is never clipped.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def market_log_return_target(raw_returns: torch.Tensor) -> torch.Tensor:
    """Return the actual market target unchanged, rejecting fabricated targets."""
    _finite_tensor(raw_returns, "raw market log returns")
    if raw_returns.ndim != 2 or not raw_returns.numel():
        raise ValueError("raw market log returns must be a nonempty B x T tensor")
    return raw_returns


@dataclass(frozen=True)
class MarketExecution:
    one_way_cost: float = 0.00055
    borrow_annual: float = 0.10
    max_step: float = 0.08
    deadband: float = 0.01
    position_min: float = 0.50
    position_max: float = 1.12
    bars_per_year: int = 35040

    def __post_init__(self) -> None:
        for name in ("one_way_cost", "borrow_annual", "max_step", "deadband",
                     "position_min", "position_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 <= self.one_way_cost < 1 or self.borrow_annual < 0:
            raise ValueError("invalid trading or borrowing cost")
        if self.max_step <= 0 or not 0 <= self.deadband <= self.max_step:
            raise ValueError("invalid max_step or deadband")
        if not 0 <= self.position_min <= 1 <= self.position_max:
            raise ValueError("market positions must be long-only and include B&H")
        if self.one_way_cost * self.position_max >= 1:
            raise ValueError("fee/position combination has a nonpositive sell denominator")
        if isinstance(self.bars_per_year, bool) or self.bars_per_year != 35040:
            raise ValueError("market imagination supports only 15m / 35040 bars per year")


def market_portfolio_step(
    cash: torch.Tensor,
    asset_value: torch.Tensor,
    target_position: torch.Tensor,
    market_log_return: torch.Tensor,
    execution: MarketExecution,
) -> dict[str, torch.Tensor]:
    """Exact post-fee own-NAV target solve, borrow, then mark one market step.

    Inputs have the same one-dimensional batch shape.  Targets are intents:
    bounds, maximum step, and deadband act on them, not on the existing state.
    A positive-NAV, finite but out-of-bounds passive exposure remains valid.
    """
    for name, value in (("cash", cash), ("asset_value", asset_value),
                        ("target_position", target_position),
                        ("market_log_return", market_log_return)):
        _finite_tensor(value, name)
        if value.ndim != 1 or value.shape != cash.shape or not value.numel():
            raise ValueError("portfolio inputs must have the same nonempty batch shape")
        if value.dtype != cash.dtype or value.device != cash.device:
            raise ValueError("portfolio inputs must share dtype and device")
    if not isinstance(execution, MarketExecution):
        raise ValueError("execution must be a validated MarketExecution")
    nav = cash + asset_value
    if bool((nav <= 0).any()) or bool((asset_value < 0).any()):
        raise ValueError("market portfolio requires positive NAV and nonnegative assets")
    current = asset_value / nav
    bounded_target = target_position.clamp(execution.position_min, execution.position_max)
    change = (bounded_target - current).clamp(-execution.max_step, execution.max_step)
    trade = (change.abs() >= execution.deadband) & (change != 0)
    desired = current + change
    denominator = 1 + execution.one_way_cost * desired * torch.sign(change)
    if bool((denominator <= 0).any()):
        raise ValueError("nonpositive post-fee target denominator")
    # Exact solve: (asset+x) / (NAV-cost*abs(x)) = desired.
    trade_value = torch.where(trade, (desired * nav - asset_value) / denominator,
                              torch.zeros_like(nav))
    fee = execution.one_way_cost * trade_value.abs()
    after_cash = cash - trade_value - fee
    after_asset = asset_value + trade_value
    post_trade_nav = after_cash + after_asset
    borrow_factor = math.expm1(execution.borrow_annual / execution.bars_per_year)
    borrow = (-after_cash).clamp_min(0) * borrow_factor
    after_cash = after_cash - borrow
    marked_asset = after_asset * torch.exp(market_log_return)
    marked_nav = after_cash + marked_asset
    values = {
        "cash": after_cash,
        "asset_value": marked_asset,
        "nav": marked_nav,
        "simple_return": marked_nav / nav - 1,
        "exposure": marked_asset / marked_nav,
        "executed_position": after_asset / post_trade_nav,
        "trade_value": trade_value,
        "fee": fee,
        "borrow": borrow,
    }
    if bool((post_trade_nav <= 0).any()) or bool((marked_nav <= 0).any()):
        raise ValueError("market imagination portfolio became insolvent")
    for name, value in values.items():
        _finite_tensor(value, name)
    return values


def compound_drawdown(simple_returns: torch.Tensor) -> torch.Tensor:
    """Fractional running drawdown from compounded NAV, including initial NAV1."""
    _finite_tensor(simple_returns, "portfolio simple returns")
    if simple_returns.ndim != 2 or not simple_returns.numel():
        raise ValueError("portfolio simple returns must be nonempty B x H")
    if bool((simple_returns <= -1).any()):
        raise ValueError("portfolio simple returns must exceed -1")
    equity = (1 + simple_returns).cumprod(dim=1)
    initial = torch.ones_like(equity[:, :1])
    peaks = torch.cat([initial, equity], dim=1).cummax(dim=1).values[:, 1:]
    result = 1 - equity / peaks
    _finite_tensor(result, "compound drawdown")
    return result.clamp_min(0)


__all__ = ["MarketExecution", "market_log_return_target", "market_portfolio_step", "compound_drawdown"]
