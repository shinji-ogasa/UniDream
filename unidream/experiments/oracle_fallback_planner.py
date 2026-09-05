"""Causal own-inventory decisions with a target-one forecast-unavailable fallback."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .alpha_dd_search import BARS_YEAR, _simulate, metrics, njit
from .oracle_conditional_planner import DELTAS, _choose


@njit(cache=True)
def _plan(opens, closes, mu, variance, inference, schedule, cost, borrow_annual,
          max_step, deadband, risk_aversion, cost_multiplier):
    n = len(opens)
    targets, equity, exposure = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    decision_nav, decision_exposure = np.full(n, np.nan), np.full(n, np.nan)
    scores, estimated_turnover = np.full(n, np.nan), np.full(n, np.nan)
    reasons = np.zeros(n, np.int8)  # 1 learned, 2 forecast-unavailable fallback.
    cash, units = 0., 1. / opens[0]
    turnover, fees, borrow, trades = 0., 0., 0., 0
    for i in range(n):
        available_open = np.isfinite(opens[i])
        nav = cash + units * opens[i] if available_open else np.nan
        if available_open and nav <= 0:
            break
        # A submitted intent has exactly one next-open opportunity. An absent
        # open skips it; later bars do not roll it forward. Close is irrelevant.
        if available_open and i > 0 and schedule[i - 1] and np.isfinite(targets[i - 1]):
            old_exposure = units * opens[i] / nav
            desired = min(max(targets[i - 1], 0.), 1.12)
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

        # Both branches observe the same own cash/units, after any current-open
        # fill and before current borrowing or current-close marking.
        if schedule[i] and available_open:
            nav = cash + units * opens[i]
            asset_value = units * opens[i]
            current = asset_value / nav
            decision_nav[i], decision_exposure[i] = nav, current
            if inference[i]:
                reasons[i] = 1
                targets[i], scores[i], estimated_turnover[i] = _choose(
                    current, nav, asset_value, mu[i], variance[i], cost,
                    borrow_annual, deadband, risk_aversion, cost_multiplier, max_step)
            else:
                reasons[i] = 2
                targets[i] = 1.

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
            decision_nav, decision_exposure, scores, estimated_turnover, reasons)


def _nonnegative_scalar(value, name):
    try:
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError()
        number = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return number


def fallback_targets(bars: pd.DataFrame, mu, variance, contract: dict, *,
                     inference_mask, risk_aversion: float = 1,
                     cost_multiplier: float = 2) -> tuple[np.ndarray, dict]:
    """Use the frozen learned choice when available, otherwise submit target 1.

    ``inference_mask`` is the caller's causal six-hour availability mask, with
    no scoring or future-outcome mask. Every claimed-valid forecast must have
    finite mean and finite nonnegative variance. Unavailable forecasts are
    ignored. Missing current open permits neither choice nor fallback order.

    A learned hold remains NaN. Fallback intents pass through the same next-open
    max-step, deadband, fee and borrowing rules and alter the inventory seen by
    later learned decisions. Actual exposure is never clipped to target bounds.
    Complete, observed evaluation boundaries remain the canonical requirement.
    """
    if (not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex)
            or bars.index.tz is None):
        raise ValueError("timezone-aware complete 15-minute grid required")
    index = bars.index.tz_convert("UTC")
    step = pd.Timedelta(minutes=15).value
    if (not len(index) or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or np.any(index.asi8 % step) or np.any(np.diff(index.asi8) != step)):
        raise ValueError("complete ordered 15-minute grid required")
    if bars.columns.has_duplicates or not {"open", "close", "bar_available"}.issubset(bars.columns):
        raise ValueError("unique open, close and bar_available columns required")
    if bars.bar_available.dtype != np.dtype(bool):
        raise ValueError("bar_available must be boolean")
    try:
        opens, closes = bars.open.to_numpy(float), bars.close.to_numpy(float)
        mu, variance = np.asarray(mu, float), np.asarray(variance, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric price and forecast arrays required") from exc
    if (np.isinf(opens).any() or np.isinf(closes).any()
            or np.any(opens[np.isfinite(opens)] <= 0)
            or np.any(closes[np.isfinite(closes)] <= 0)):
        raise ValueError("observed prices must be positive and finite; gaps use NaN")
    if (not np.isfinite(opens[0]) or not np.isfinite(closes[-1])
            or not bool(bars.bar_available.iloc[0]) or not bool(bars.bar_available.iloc[-1])):
        raise ValueError("canonical evaluation boundary bars must be present")
    if mu.shape != (len(bars),) or variance.shape != (len(bars),):
        raise ValueError("aligned one-dimensional forecast arrays required")
    inference = np.asarray(inference_mask)
    if inference.shape != (len(bars),) or inference.dtype != np.dtype(bool):
        raise ValueError("inference_mask must be an aligned one-dimensional boolean mask")
    schedule = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
    if np.any(inference & ~schedule):
        raise ValueError("inference_mask must use only scheduled six-hour UTC decisions")
    if (not np.isfinite(mu[inference]).all() or not np.isfinite(variance[inference]).all()
            or np.any(variance[inference] < 0)):
        raise ValueError("claimed-valid inference requires finite mean and nonnegative variance")
    try:
        parameters = [float(contract[k]) for k in ("one_way_cost", "borrow_annual", "max_step", "deadband")]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("complete numeric execution contract required") from exc
    if (not np.isfinite(parameters).all() or min(parameters) < 0
            or parameters[0] >= 1 / 1.12 or parameters[2] <= 0 or parameters[3] <= 0):
        raise ValueError("valid nonnegative costs and positive step/deadband required")
    risk = _nonnegative_scalar(risk_aversion, "risk_aversion")
    multiplier = _nonnegative_scalar(cost_multiplier, "cost_multiplier")
    planned = _plan(opens, closes, mu, variance, inference, schedule, *parameters, risk, multiplier)
    targets, equity, positions, turnover, fees, borrow, trades = planned[:7]
    canonical = _simulate(opens, closes, targets, np.asarray(index.minute == 0), *parameters)
    differences = {}
    for name, actual, expected in zip(
            ("equity", "exposure", "turnover", "fees", "borrow", "trades"),
            (equity, positions, turnover, fees, borrow, trades), canonical):
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise AssertionError(f"fallback planner/canonical replay mismatch: {name}")
        delta = np.abs(np.asarray(actual) - np.asarray(expected))
        finite = delta[np.isfinite(delta)]
        differences[name] = float(np.max(finite)) if finite.size else 0.
    replay_bars = bars.copy(deep=False)
    replay_bars.index = index
    replay = metrics(replay_bars, targets, contract)
    decision_nav, decision_exposure, scores, estimated_turnover, reasons = planned[7:]
    learned, fallback = reasons == 1, reasons == 2
    hold = learned & ~np.isfinite(targets)
    missing_open = schedule & ~np.isfinite(opens)
    decided = np.flatnonzero(learned | fallback)
    nullable = lambda array: [float(array[i]) if np.isfinite(array[i]) else None for i in decided]
    trace = {"bar_indices": decided.tolist(),
        "reasons": ["learned" if learned[i] else "forecast_unavailable" for i in decided],
        "known_open_nav": decision_nav[decided].tolist(),
        "known_open_exposure": decision_exposure[decided].tolist(),
        "targets": nullable(targets), "estimated_utility_gain_over_hold": nullable(scores),
        "estimated_trade_turnover": nullable(estimated_turnover)}
    diagnostic = {
        "diagnostic_kind": "causal_conditional_utility_with_target_one_unavailable_fallback",
        "future_information_used_for_decisions": False, "hindsight_only": False,
        "teacher_actions_used": False, "global_optimum_claimed": False,
        "bayes_optimum_claimed": False, "drawdown_optimum_claimed": False,
        "risk_aversion": risk, "cost_multiplier": multiplier,
        "horizon_bars": 24, "decision_cadence_hours": 6, "execution_delay_bars": 1,
        "candidate_exposure_deltas": DELTAS.tolist(), "target_floor": .5, "target_ceiling": 1.12,
        "fallback_target": 1., "scheduled_decision_count": int(schedule.sum()),
        "inference_available_decision_count": int(inference.sum()),
        "valid_decision_count": int(learned.sum()), "learned_decision_count": int(learned.sum()),
        "fallback_decision_count": int(fallback.sum()), "hold_decision_count": int(hold.sum()),
        "missing_open_decision_count": int(missing_open.sum()),
        "unavailable_forecast_decision_count": int(fallback.sum()),
        "learned_intent_count": int(np.sum(learned & np.isfinite(targets))),
        "positive_intent_count": int(np.isfinite(targets).sum()),
        "canonical_replay_verified": True, "accounting_max_absolute_difference": max(differences.values()),
        "accounting_absolute_differences": differences,
        "decision_masks": {"learned": learned.tolist(), "fallback": fallback.tolist(),
                           "hold": hold.tolist(), "missing_open": missing_open.tolist()},
        "decision_trace": trace, "metrics": replay,
    }
    return targets, diagnostic


__all__ = ["fallback_targets"]
