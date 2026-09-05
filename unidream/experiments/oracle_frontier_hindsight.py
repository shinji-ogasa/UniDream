"""Full-path hindsight planning reference for the Alpha/DD cash/units account.

This diagnostic deliberately reads future prices.  A finite beam finds a feasible
path, so its objective is a LOWER bound on the maximum attainable hindsight
objective, not a global optimum, causal forecast, RL policy, or training teacher.
The fixed intent set is no trade, 0.5, 1.0, and 1.12 exposure; decisions occur
every six UTC hours and fill at the next 15-minute bar's observed open.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import BARS_YEAR, metrics, njit


# State: cash, units, running close-NAV peak, MaxDD, last marked NAV,
# turnover, fees, borrow, trades.  Keep accounting order identical to _simulate.
INTENTS = np.array([np.nan, 0.5, 1.0, 1.12], dtype=np.float64)


@njit(cache=True)
def _advance(state, opens, closes, start, end, target, cost, borrow_annual,
             max_step, deadband):
    out = state.copy()
    for i in range(start, end):
        fill_available = np.isfinite(opens[i])
        nav = out[0] + out[1] * opens[i] if fill_available else np.nan
        if fill_available and nav <= 0:
            return out, False
        if i == start and i > 0 and fill_available and np.isfinite(target):
            old_exposure = out[1] * opens[i] / nav
            desired = min(max(target, 0.0), 1.12)
            change = min(max(desired - old_exposure, -max_step), max_step)
            if abs(change) >= deadband:
                desired = old_exposure + change
                x = ((desired * nav - out[1] * opens[i]) /
                     (1 + cost * desired * (1 if change > 0 else -1)))
                fee = cost * abs(x)
                out[0] -= x + fee
                out[1] += x / opens[i]
                out[5] += abs(x) / nav
                out[6] += fee
                out[8] += 1
        if out[0] < 0:
            charge = -out[0] * (math.exp(borrow_annual / BARS_YEAR) - 1)
            out[0] -= charge
            out[7] += charge
        if not np.isfinite(closes[i]):
            continue
        out[4] = out[0] + out[1] * closes[i]
        if out[4] <= 0 or not np.isfinite(out[4]):
            return out, False
        out[2] = max(out[2], out[4])
        out[3] = max(out[3], 1 - out[4] / out[2])
    return out, True


@njit(cache=True)
def _search(opens, closes, decisions, cost, borrow_annual, max_step, deadband,
            beam_width, risk_penalty):
    n, event_count = len(opens), len(decisions)
    initial = np.array([0.0, 1.0 / opens[0], 1.0, 0.0, 1.0,
                        0.0, 0.0, 0.0, 0.0])
    prefix_end = decisions[0] + 1 if event_count else n
    state, valid = _advance(initial, opens, closes, 0, prefix_end, np.nan,
                            cost, borrow_annual, max_step, deadband)
    if not valid:
        raise ValueError("insolvent initial B&H inventory")
    states = state.reshape(1, 9)
    parents = np.full((event_count, beam_width), -1, dtype=np.int64)
    chosen_intents = np.full((event_count, beam_width), np.nan)
    expanded, pruned = 0, 0
    for event in range(event_count):
        start = decisions[event] + 1
        end = decisions[event + 1] + 1 if event + 1 < event_count else n
        candidates = np.empty((len(states) * len(INTENTS), 9))
        candidate_parents = np.empty(len(candidates), dtype=np.int64)
        candidate_intents = np.empty(len(candidates))
        scores = np.empty(len(candidates))
        count = 0
        for parent in range(len(states)):
            for intent in INTENTS:
                candidate, valid = _advance(
                    states[parent], opens, closes, start, end, intent,
                    cost, borrow_annual, max_step, deadband)
                expanded += 1
                if not valid:
                    continue
                # Equivalent intents after step limits or missing fills must not
                # consume the finite beam with identical accounting states.
                duplicate = False
                for previous in range(count):
                    if np.all(candidate == candidates[previous]):
                        duplicate = True
                        break
                if duplicate:
                    continue
                candidates[count] = candidate
                candidate_parents[count] = parent
                candidate_intents[count] = intent
                scores[count] = math.log(candidate[4]) - risk_penalty * candidate[3]
                count += 1
        if not count:
            raise ValueError("all hindsight branches insolvent")
        keep = min(beam_width, count)
        pruned += count - keep
        # Stable ordering gives no-trade first when objective/state ties occur.
        order = np.argsort(-scores[:count], kind="mergesort")[:keep]
        states = candidates[order].copy()
        for slot in range(keep):
            source = order[slot]
            parents[event, slot] = candidate_parents[source]
            chosen_intents[event, slot] = candidate_intents[source]
    targets = np.full(n, np.nan)
    slot = 0
    for event in range(event_count - 1, -1, -1):
        targets[decisions[event]] = chosen_intents[event, slot]
        slot = parents[event, slot]
    best = states[0].copy()
    # Beam pruning can discard a path that later recovers.  Always retain the
    # feasible all-hold comparator in the final opportunity-reference envelope.
    hold, valid = _advance(initial, opens, closes, 0, n, np.nan, cost,
                          borrow_annual, max_step, deadband)
    hold_fallback = False
    if valid and (math.log(hold[4]) - risk_penalty * hold[3] >
                  math.log(best[4]) - risk_penalty * best[3]):
        targets[:] = np.nan
        best = hold
        hold_fallback = True
    return targets, best, expanded, pruned, hold_fallback


def hindsight_targets(bars: pd.DataFrame, contract: dict, *, beam_width: int = 32,
                      risk_penalty: float = 0.0) -> tuple[np.ndarray, dict]:
    """Return future-dependent intents and canonically replayed diagnostics.

    Pruning scores are log marked NAV minus ``risk_penalty * running MaxDD``;
    final ranking uses the same objective on terminal NAV.  Running peak starts
    at equity 1, before the first close.  Missing opens skip fills, missing closes
    skip marks, and borrowing continues over missing bars, exactly as in the
    Alpha/DD evaluator.  ``bar_available`` cannot retrospectively cancel a fill.

    The input must retain its full UTC 15-minute grid.  Callers control the
    diagnostic's registered period; this function performs no fitting/selection
    and its output must never enter a causal model's features or teacher labels.
    """
    if isinstance(beam_width, bool) or not isinstance(beam_width, Integral) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if not np.isfinite(risk_penalty) or risk_penalty < 0:
        raise ValueError("risk_penalty must be finite and nonnegative")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars require a timezone-aware 15-minute UTC grid")
    utc_index = bars.index.tz_convert("UTC")
    if (not len(bars) or not utc_index.is_monotonic_increasing or
            not utc_index.is_unique or (utc_index.minute % 15 != 0).any() or
            (utc_index.second != 0).any() or (utc_index.microsecond != 0).any() or
            (utc_index.nanosecond != 0).any() or
            (np.diff(utc_index.asi8) != pd.Timedelta(minutes=15).value).any()):
        raise ValueError("bars must retain the complete ordered 15-minute grid")
    opens = bars.open.to_numpy(dtype=np.float64)
    closes = bars.close.to_numpy(dtype=np.float64)
    if (np.isinf(opens).any() or np.isinf(closes).any() or
            (opens[np.isfinite(opens)] <= 0).any() or
            (closes[np.isfinite(closes)] <= 0).any()):
        raise ValueError("observed prices must be finite and positive; gaps use NaN")
    if (not bars.bar_available.iloc[0] or not bars.bar_available.iloc[-1] or
            not np.isfinite(opens[0]) or not np.isfinite(closes[-1])):
        raise ValueError("evaluation boundary bars must be present; no boundary shifting")
    names = ("one_way_cost", "borrow_annual", "max_step", "deadband")
    parameters = [float(contract[name]) for name in names]
    if not np.isfinite(parameters).all() or any(value < 0 for value in parameters):
        raise ValueError("execution parameters must be finite and nonnegative")
    decisions = np.flatnonzero((utc_index.minute == 0) & (utc_index.hour % 6 == 0))
    decisions = decisions[decisions + 1 < len(bars)].astype(np.int64)
    targets, state, expanded, pruned, fallback = _search(
        opens, closes, decisions, *parameters, int(beam_width), float(risk_penalty))
    # metrics schedules hourly, but all non-6h intents above are NaN.  Convert
    # the replay index to UTC so timezone representations cannot move the clock.
    replay_bars = bars.copy(deep=False)
    replay_bars.index = utc_index
    replay = metrics(replay_bars, targets, contract)
    planned = {
        "total_return": float(state[4] - 1), "maxdd": float(state[3]),
        "turnover": float(state[5]), "fees_initial_equity_units": float(state[6]),
        "borrow_initial_equity_units": float(state[7]), "trades": int(state[8]),
    }
    differences = {key: abs(planned[key] - replay[key]) for key in planned}
    if any(not np.isclose(planned[key], replay[key], rtol=1e-11, atol=1e-12)
           for key in planned):
        raise AssertionError(f"hindsight planning/canonical replay mismatch: {differences}")
    objective = math.log1p(replay["total_return"]) - risk_penalty * replay["maxdd"]
    diagnostics = {
        "diagnostic_kind": "rl_full_path_hindsight_planning",
        "future_information_used": True,
        "deployable": False, "teacher_use_allowed": False,
        "global_optimum_claimed": False,
        "bound_direction": "lower_bound_on_maximum_attainable_hindsight_objective",
        "objective_definition": "log_terminal_nav_minus_risk_penalty_times_maxdd",
        "objective": float(objective), "risk_penalty": float(risk_penalty),
        "beam_width": int(beam_width), "decision_cadence_hours": 6,
        "execution_delay_bars": 1, "decision_count": len(decisions),
        "intent_set": ["no_trade", 0.5, 1.0, 1.12],
        "expanded_branches": int(expanded), "pruned_distinct_branches": int(pruned),
        "exhaustive_for_fixed_intent_set": pruned == 0,
        "all_hold_fallback_selected": bool(fallback),
        "canonical_replay_verified": True,
        "accounting_max_absolute_difference": float(max(differences.values())),
        "metrics": replay,
    }
    return targets, diagnostics
