"""Causal one-step allocation from frozen conditional return/risk forecasts.

The score is an estimated six-hour log-utility difference versus holding the
current cash/units inventory. It is neither a Bayes-optimal policy nor a global
return or drawdown optimizer. No hindsight target or future price enters a
decision. The caller is responsible for binding causal forecast provenance.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .alpha_dd_search import BARS_YEAR, _simulate, metrics, njit


DELTAS = np.array([-.08, -.04, .04, .08], dtype=float)


@njit(cache=True)
def _choose(current, nav, asset_value, mu, variance, cost, borrow_annual,
            deadband, risk_aversion, cost_multiplier, max_step=.08):
    # No trade is a separate action. A numerical target equal to current would
    # otherwise rebalance its naturally drifting inventory at the future fill.
    target, best_score, estimated_turnover = np.nan, 0.0, 0.0
    for delta in DELTAS:
        intent = min(max(current + delta, .5), 1.12)
        # Passive drift can put inventory far outside target bounds. Score
        # only the fill projection possible under max_step at the known open,
        # while issuing the bounded intent for the actual next-open projection.
        change = min(max(intent-current, -max_step), max_step)
        position = current + change
        if change == 0 or abs(change) < deadband:
            continue
        trade_value = ((position * nav - asset_value)
                       / (1 + cost * position * (1 if change > 0 else -1)))
        turnover = abs(trade_value) / nav
        score = ((position - current) * mu
                 - .5 * risk_aversion * (position * position - current * current) * variance
                 - cost_multiplier * cost * turnover
                 - (max(position - 1, 0) - max(current - 1, 0)) * borrow_annual * 24 / BARS_YEAR)
        # Strict improvement is required; no trade wins every zero-score tie.
        # Equal positive scores keep the first candidate in the fixed order.
        if score > best_score:
            target, best_score, estimated_turnover = intent, score, turnover
    return target, best_score, estimated_turnover


@njit(cache=True)
def _plan(opens, closes, mu, variance, schedule, cost, borrow_annual,
          max_step, deadband, risk_aversion, cost_multiplier):
    n = len(opens)
    targets = np.full(n, np.nan)
    equity = np.full(n, np.nan)
    exposure = np.full(n, np.nan)
    decision_nav = np.full(n, np.nan)
    decision_exposure = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    estimated_turnover = np.full(n, np.nan)
    cash, units = 0.0, 1.0 / opens[0]
    turnover, fees, borrow, trades = 0.0, 0.0, 0.0, 0
    for i in range(n):
        available_open = np.isfinite(opens[i])
        nav = cash + units * opens[i] if available_open else np.nan
        if available_open and nav <= 0:
            break
        # Execute yesterday's submitted intent before making a new decision.
        # No current close or bar_available flag may cancel this open fill.
        if available_open and i > 0 and schedule[i - 1] and np.isfinite(targets[i - 1]):
            old_exposure = units * opens[i] / nav
            desired = min(max(targets[i - 1], 0.0), 1.12)
            change = min(max(desired - old_exposure, -max_step), max_step)
            if abs(change) >= deadband:
                desired = old_exposure + change
                trade_value = ((desired * nav - units * opens[i])
                               / (1 + cost * desired * (1 if change > 0 else -1)))
                fee = cost * abs(trade_value)
                cash -= trade_value + fee
                units += trade_value / opens[i]
                turnover += abs(trade_value) / nav
                fees += fee
                trades += 1

        # Everything below used by the decision is known at this bar's open.
        # Current close and this bar's borrowing charge occur after the choice.
        if (schedule[i] and available_open and np.isfinite(mu[i])
                and np.isfinite(variance[i]) and variance[i] >= 0):
            nav = cash + units * opens[i]
            asset_value = units * opens[i]
            current = asset_value / nav
            targets[i], scores[i], estimated_turnover[i] = _choose(
                current, nav, asset_value, mu[i], variance[i], cost,
                borrow_annual, deadband, risk_aversion, cost_multiplier, max_step)
            decision_nav[i], decision_exposure[i] = nav, current

        if cash < 0:
            charge = -cash * (math.exp(borrow_annual / BARS_YEAR) - 1)
            cash -= charge
            borrow += charge
        if not np.isfinite(closes[i]):
            continue
        equity[i] = cash + units * closes[i]
        if equity[i] <= 0:
            break
        exposure[i] = units * closes[i] / equity[i]
    return (targets, equity, exposure, turnover, fees, borrow, trades,
            decision_nav, decision_exposure, scores, estimated_turnover)


def conditional_targets(bars: pd.DataFrame, mu, variance, contract: dict, *,
                        risk_aversion: float, cost_multiplier: float = 2) -> tuple[np.ndarray, dict]:
    """Return six-hour causal order intents and canonical-accounting diagnostics.

    Decisions use cash/units as of open[t] and frozen forecasts at t. Changes
    {-0.08,-0.04,+0.04,+0.08} are proposed around the actual current exposure,
    then clipped to [.5,1.12]. Holding is always allowed, including when passive
    drift has moved exposure outside those bounds. Holding is encoded as NaN,
    so it does not silently reset the exposure at the next bar.

    An accepted target is an intent: its next-bar open is unknown when scored.
    The canonical fill applies max_step and deadband to the exposure actually
    observed then. A missing next open skips that fill without looking ahead
    to suppress the original decision. Borrowing continues across missing bars.

    The returned trace exposes known-open NAV/exposure and the selected score.
    It contains no forward mark, oracle action, or future availability signal.
    """
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("timezone-aware complete 15-minute grid required")
    index = bars.index.tz_convert("UTC")
    if (not len(index) or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or np.any(index.asi8 % pd.Timedelta(minutes=15).value)
            or np.any(np.diff(index.asi8) != pd.Timedelta(minutes=15).value)):
        raise ValueError("complete ordered 15-minute grid required")
    if not {"open", "close", "bar_available"}.issubset(bars.columns):
        raise ValueError("open, close and bar_available columns required")
    opens, closes = bars.open.to_numpy(float), bars.close.to_numpy(float)
    if (np.isinf(opens).any() or np.isinf(closes).any()
            or np.any(opens[np.isfinite(opens)] <= 0)
            or np.any(closes[np.isfinite(closes)] <= 0)):
        raise ValueError("observed prices must be positive and finite; gaps use NaN")
    if (not np.isfinite(opens[0]) or not np.isfinite(closes[-1])
            or not bool(bars.bar_available.iloc[0]) or not bool(bars.bar_available.iloc[-1])):
        raise ValueError("canonical evaluation boundary bars must be present")
    mu, variance = np.asarray(mu, float), np.asarray(variance, float)
    if mu.shape != (len(bars),) or variance.shape != (len(bars),):
        raise ValueError("aligned one-dimensional forecast arrays required")
    parameters = [float(contract[k]) for k in ("one_way_cost", "borrow_annual", "max_step", "deadband")]
    if (not np.isfinite(parameters).all() or min(parameters) < 0
            or parameters[0] >= 1 / 1.12 or parameters[2] <= 0 or parameters[3] <= 0):
        raise ValueError("valid nonnegative costs and positive step/deadband required")
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("risk_aversion must be finite and nonnegative")
    if not np.isfinite(cost_multiplier) or cost_multiplier < 0:
        raise ValueError("cost_multiplier must be finite and nonnegative")
    schedule = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
    planned = _plan(opens, closes, mu, variance, schedule, *parameters,
                    float(risk_aversion), float(cost_multiplier))
    targets, equity, positions, turnover, fees, borrow, trades = planned[:7]
    replay_schedule = np.asarray(index.minute == 0)
    canonical = _simulate(opens, closes, targets, replay_schedule, *parameters)
    differences = {}
    for name, actual, expected in zip(
            ("equity", "exposure", "turnover", "fees", "borrow", "trades"),
            (equity, positions, turnover, fees, borrow, trades), canonical):
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise AssertionError(f"conditional planner/canonical replay mismatch: {name}")
        delta = np.abs(np.asarray(actual) - np.asarray(expected))
        finite = delta[np.isfinite(delta)]
        differences[name] = float(np.max(finite)) if finite.size else 0.0
    replay_bars = bars.copy(deep=False)
    replay_bars.index = index
    replay = metrics(replay_bars, targets, contract)
    decision_nav, decision_exposure, scores, estimated_turnover = planned[7:]
    decided = np.flatnonzero(np.isfinite(scores))
    trace = {"bar_indices": decided.tolist(), "known_open_nav": decision_nav[decided].tolist(),
             "known_open_exposure": decision_exposure[decided].tolist(),
             "estimated_utility_gain_over_hold": scores[decided].tolist(),
             "estimated_trade_turnover": estimated_turnover[decided].tolist()}
    diagnostic = {
        "diagnostic_kind": "causal_conditional_one_step_log_utility_planner",
        "future_information_used_for_decisions": False, "hindsight_only": False,
        "teacher_actions_used": False, "global_optimum_claimed": False,
        "bayes_optimum_claimed": False, "drawdown_optimum_claimed": False,
        "risk_aversion": float(risk_aversion), "cost_multiplier": float(cost_multiplier),
        "horizon_bars": 24, "decision_cadence_hours": 6, "execution_delay_bars": 1,
        "candidate_exposure_deltas": DELTAS.tolist(), "target_floor": .5, "target_ceiling": 1.12,
        "scheduled_decision_count": int(schedule.sum()), "valid_decision_count": len(decided),
        "positive_intent_count": int(np.isfinite(targets).sum()),
        "hold_decision_count": int(np.sum(np.isfinite(scores) & ~np.isfinite(targets))),
        "missing_open_decision_count": int(np.sum(schedule & ~np.isfinite(opens))),
        "unavailable_forecast_decision_count": int(np.sum(schedule & np.isfinite(opens)
            & (~np.isfinite(mu) | ~np.isfinite(variance) | (variance < 0)))),
        "canonical_replay_verified": True,
        "accounting_max_absolute_difference": max(differences.values()),
        "accounting_absolute_differences": differences, "decision_trace": trace, "metrics": replay,
    }
    return targets, diagnostic


__all__ = ["conditional_targets"]
