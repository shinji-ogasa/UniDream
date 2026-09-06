"""Finite-beam hindsight with the conditional planner's dynamic action set.

Future prices rank feasible cash/units paths. The result is a lower bound on
the maximum hindsight objective, never a deployable model, global ceiling,
causal action, or training teacher. No forecast or label is consumed here.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .alpha_dd_search import metrics, njit
from .oracle_conditional_planner import DELTAS
from .oracle_frontier_hindsight import _advance


@njit(cache=True)
def _dynamic_intents(current, max_step, deadband):
    """Include no-trade; apply the frozen known-open projection eligibility."""
    result = np.full(5, np.nan)
    count = 1
    for delta in DELTAS:
        intent = min(max(current + delta, .5), 1.12)
        change = min(max(intent - current, -max_step), max_step)
        if change == 0 or abs(change) < deadband:
            continue
        duplicate = False
        for j in range(1, count):
            if intent == result[j]:
                duplicate = True
                break
        if not duplicate:
            result[count] = intent
            count += 1
    return result[:count]


@njit(cache=True)
def _search(opens, closes, decisions, support, fallback_bh, cost, borrow_annual,
            max_step, deadband, beam_width, risk_penalty):
    n, events = len(opens), len(decisions)
    initial = np.array([0., 1. / opens[0], 1., 0., 1., 0., 0., 0., 0.])
    # States are immediately BEFORE the decision bar, not after its close or
    # borrow charge. Earlier unsupported slots advance without any new order.
    end = decisions[0] if events else n
    prefix, valid = _advance(initial, opens, closes, 0, end, np.nan,
                             cost, borrow_annual, max_step, deadband)
    if not valid:
        raise ValueError("insolvent initial inventory")
    states = prefix.reshape(1, 9)
    parents = np.full((events, beam_width), -1, dtype=np.int64)
    choices = np.full((events, beam_width), np.nan)
    expanded, pruned, duplicates, insolvent = 0, 0, 0, 0
    for event in range(events):
        decision = decisions[event]
        end = decisions[event + 1] if event + 1 < events else n
        candidates = np.empty((len(states) * 5, 9))
        origins = np.empty(len(candidates), dtype=np.int64)
        intents = np.empty(len(candidates))
        scores = np.empty(len(candidates))
        count = 0
        for parent in range(len(states)):
            known = np.isfinite(opens[decision])
            nav = states[parent, 0] + states[parent, 1] * opens[decision] if known else np.nan
            if known and (not np.isfinite(nav) or nav <= 0):
                insolvent += 1
                continue
            actions = np.full(1, np.nan)
            if known and support[decision]:
                current = states[parent, 1] * opens[decision] / nav
                actions = _dynamic_intents(current, max_step, deadband)
            elif known and fallback_bh:
                actions[0] = 1.
            marked, valid = _advance(states[parent], opens, closes, decision, decision + 1,
                                     np.nan, cost, borrow_annual, max_step, deadband)
            if not valid:
                insolvent += len(actions)
                continue
            for intent in actions:
                candidate, valid = _advance(marked, opens, closes, decision + 1, end,
                                            intent, cost, borrow_annual, max_step, deadband)
                expanded += 1
                if not valid:
                    insolvent += 1
                    continue
                duplicate = False
                for previous in range(count):
                    if np.all(candidate == candidates[previous]):
                        duplicate = True
                        break
                if duplicate:
                    duplicates += 1
                    continue
                candidates[count] = candidate
                origins[count], intents[count] = parent, intent
                scores[count] = math.log(candidate[4]) - risk_penalty * candidate[3]
                count += 1
        if not count:
            raise ValueError("all hindsight branches insolvent")
        keep = min(count, beam_width)
        pruned += count - keep
        # No-trade is first and stable sorting preserves exact objective ties.
        order = np.argsort(-scores[:count], kind="mergesort")[:keep]
        states = candidates[order].copy()
        for slot in range(keep):
            source = order[slot]
            parents[event, slot] = origins[source]
            choices[event, slot] = intents[source]
    targets = np.full(n, np.nan)
    slot = 0
    for event in range(events - 1, -1, -1):
        targets[decisions[event]] = choices[event, slot]
        slot = parents[event, slot]
    best = states[0].copy()
    beam_objective = math.log(best[4]) - risk_penalty * best[3]
    # The incumbent must obey the same missing-input rule. Under fallback_bh,
    # it holds on supported slots and submits 1.0 on every unsupported known
    # open; the unconditional all-NaN path would violate that action contract.
    incumbent_targets = np.full(n, np.nan)
    if fallback_bh:
        for decision in decisions:
            if not support[decision] and np.isfinite(opens[decision]):
                incumbent_targets[decision] = 1.
    incumbent, _, _, valid = _path_trace(opens, closes, decisions, incumbent_targets,
                                         cost, borrow_annual, max_step, deadband)
    incumbent_selected = False
    incumbent_objective = np.nan
    if valid:
        incumbent_objective = math.log(incumbent[4]) - risk_penalty * incumbent[3]
        if incumbent_objective >= beam_objective:
            targets = incumbent_targets
            best, incumbent_selected = incumbent, True
    return (targets, best, expanded, pruned, duplicates, insolvent,
            incumbent_selected, beam_objective, incumbent_objective)


@njit(cache=True)
def _path_trace(opens, closes, decisions, targets, cost, borrow_annual,
                max_step, deadband):
    navs, exposures = np.empty(len(decisions)), np.empty(len(decisions))
    state = np.array([0., 1. / opens[0], 1., 0., 1., 0., 0., 0., 0.])
    prefix_end = decisions[0] if len(decisions) else len(opens)
    state, valid = _advance(state, opens, closes, 0, prefix_end, np.nan,
                            cost, borrow_annual, max_step, deadband)
    if not valid:
        return state, navs, exposures, False
    for event in range(len(decisions)):
        decision = decisions[event]
        known = np.isfinite(opens[decision])
        navs[event] = state[0] + state[1] * opens[decision] if known else np.nan
        exposures[event] = state[1] * opens[decision] / navs[event] if known else np.nan
        state, valid = _advance(state, opens, closes, decision, decision + 1, np.nan,
                                cost, borrow_annual, max_step, deadband)
        if not valid:
            return state, navs, exposures, False
        end = decisions[event + 1] if event + 1 < len(decisions) else len(opens)
        state, valid = _advance(state, opens, closes, decision + 1, end, targets[decision],
                                cost, borrow_annual, max_step, deadband)
        if not valid:
            return state, navs, exposures, False
    return state, navs, exposures, True


def matched_hindsight_targets(bars, contract, *, decision_support, beam_width=32,
                              risk_penalty=0.0, missing_input_rule="hold") -> tuple[np.ndarray, dict]:
    """Rank dynamic feasible intents on supplied six-hour decision support.

    ``decision_support`` is a full-grid boolean decision mask, not a future-label
    mask. The caller binds that causal support. Unknown current opens cause no
    order. False support holds by default; ``fallback_bh`` instead forces intent
    1.0 at every unsupported scheduled known open. Candidate values use own inventory at current open,
    before current close/borrow; their ranking deliberately reads future prices.
    Missing next opens skip fills without rollover. No-trade stays NaN even if
    passive exposure lies outside [.5,1.12]. Last-bar orders have no fill and
    identical free branches collapse to hold; required terminal fallback intents
    remain present. Full evaluation boundaries are required. The final incumbent
    obeys the chosen missing-input rule and holds on freely supported decisions.
    """
    if isinstance(beam_width, (bool, np.bool_)) or not isinstance(beam_width, Integral) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if (isinstance(risk_penalty, (bool, np.bool_)) or not isinstance(risk_penalty, Real)
            or not np.isfinite(risk_penalty) or risk_penalty < 0):
        raise ValueError("risk_penalty must be finite and nonnegative")
    if missing_input_rule not in ("hold", "fallback_bh"):
        raise ValueError("missing_input_rule must be hold or fallback_bh")
    if not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("timezone-aware complete UTC 15-minute calendar required")
    index = bars.index.tz_convert("UTC")
    if (not len(index) or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or np.any(index.asi8 % pd.Timedelta(minutes=15).value)
            or np.any(np.diff(index.asi8) != pd.Timedelta(minutes=15).value)):
        raise ValueError("complete ordered 15-minute grid required")
    if not {"open", "close", "bar_available"}.issubset(bars.columns):
        raise ValueError("open, close and bar_available columns required")
    support = np.asarray(decision_support)
    schedule = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
    if support.dtype != np.dtype(bool) or support.shape != (len(bars),) or np.any(support & ~schedule):
        raise ValueError("aligned boolean support on the six-hour UTC clock required")
    raw_open, raw_close = bars.open.to_numpy(), bars.close.to_numpy()
    if np.iscomplexobj(raw_open) or np.iscomplexobj(raw_close):
        raise ValueError("real prices required")
    try:
        opens, closes = np.asarray(raw_open, float), np.asarray(raw_close, float)
        parameters = []
        for key in ("one_way_cost", "borrow_annual", "max_step", "deadband"):
            value = contract[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real): raise ValueError("real execution parameter required")
            parameters.append(float(value))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real prices and complete execution contract required") from exc
    if (np.isinf(opens).any() or np.isinf(closes).any()
            or np.any(opens[np.isfinite(opens)] <= 0) or np.any(closes[np.isfinite(closes)] <= 0)):
        raise ValueError("observed prices must be positive; missing prices use NaN")
    if (not np.isfinite(opens[0]) or not np.isfinite(closes[-1])
            or not bool(bars.bar_available.iloc[0]) or not bool(bars.bar_available.iloc[-1])):
        raise ValueError("canonical evaluation boundary bars must be present")
    if (not np.isfinite(parameters).all() or min(parameters) < 0 or parameters[0] >= 1 / 1.12
            or parameters[2] <= 0 or parameters[3] <= 0):
        raise ValueError("valid nonnegative costs and positive step/deadband required")
    fallback_bh = missing_input_rule == "fallback_bh"
    free = support & np.isfinite(opens)
    forced = schedule & ~support & np.isfinite(opens) if fallback_bh else np.zeros(len(bars), bool)
    eligible = free | forced
    # Preserve the existing hold search. The fallback variant advances every
    # scheduled event, including forced holds at unknown current opens.
    decisions = np.flatnonzero(schedule if fallback_bh else free).astype(np.int64)
    (targets, state, expanded, pruned, duplicates, insolvent, incumbent_selected,
     beam_objective, incumbent_objective) = _search(
        opens, closes, decisions, support, fallback_bh, *parameters, int(beam_width), float(risk_penalty))
    _, navs, exposures, valid = _path_trace(opens, closes, decisions, targets, *parameters)
    known_events = np.isfinite(opens[decisions])
    if not valid or not np.isfinite(navs[known_events]).all() or not np.isfinite(exposures[known_events]).all():
        raise AssertionError("invalid chosen-path known-open trace")
    for i, current in zip(decisions, exposures):
        if free[i] and np.isfinite(targets[i]) and targets[i] not in _dynamic_intents(current, parameters[2], parameters[3]):
            raise AssertionError("chosen intent escaped dynamic action set")
    if (np.any(np.isfinite(targets) & ~eligible) or np.any(targets[forced] != 1.)):
        raise AssertionError("chosen path violated forced missing-input actions")
    replay_bars = bars.copy(deep=False)
    replay_bars.index = index
    replay = metrics(replay_bars, targets, contract)
    planned = {"total_return": float(state[4] - 1), "maxdd": float(state[3]),
        "turnover": float(state[5]), "fees_initial_equity_units": float(state[6]),
        "borrow_initial_equity_units": float(state[7]), "trades": int(state[8])}
    differences = {key: abs(planned[key] - replay[key]) for key in planned}
    if any(not np.isclose(planned[key], replay[key], rtol=1e-11, atol=1e-12) for key in planned):
        raise AssertionError(f"matched hindsight/canonical replay mismatch: {differences}")
    traced = decisions[known_events]
    trace = {"bar_indices": traced.tolist(), "known_open_nav": navs[known_events].tolist(),
        "known_open_exposure": exposures[known_events].tolist(),
        "reasons": ["forced_fallback_bh" if forced[i] else "free_dynamic_action" for i in traced],
        "targets": [float(targets[i]) if np.isfinite(targets[i]) else None for i in traced]}
    diagnostic = {"diagnostic_kind": "matched_dynamic_action_finite_beam_hindsight",
        "future_information_used": True, "deployable": False, "teacher_use_allowed": False,
        "global_optimum_claimed": False, "support_causality_verified": False,
        "bound_direction": "lower_bound_on_maximum_attainable_hindsight_objective",
        "objective_definition": "log_terminal_nav_minus_risk_penalty_times_maxdd",
        "objective": float(math.log1p(replay["total_return"]) - risk_penalty * replay["maxdd"]),
        "risk_penalty": float(risk_penalty), "beam_width": int(beam_width),
        "missing_input_rule": missing_input_rule,
        "candidate_exposure_deltas": DELTAS.tolist(), "target_floor": .5, "target_ceiling": 1.12,
        "decision_cadence_hours": 6, "execution_delay_bars": 1,
        "scheduled_decision_count": int(schedule.sum()), "supported_decision_count": int(support.sum()),
        "decision_count": len(traced), "search_event_count": len(decisions),
        "free_branching_decision_count": int(free.sum()), "forced_fallback_decision_count": int(forced.sum()),
        "unsupported_decision_count": int((schedule & ~support).sum()),
        "missing_open_decision_count": int((schedule & ~np.isfinite(opens)).sum()),
        "supported_missing_open_decision_count": int((support & ~np.isfinite(opens)).sum()),
        "intent_count": int(np.isfinite(targets).sum()), "hold_decision_count": int(np.isnan(targets[eligible]).sum()),
        "decision_support": support.tolist(), "eligible_decisions": eligible.tolist(), "decision_trace": trace,
        "expanded_branches": int(expanded), "pruned_distinct_branches": int(pruned),
        "duplicate_states_collapsed": int(duplicates), "insolvent_branches_rejected": int(insolvent),
        "exhaustive_for_matched_dynamic_action_set": pruned == 0,
        "all_hold_envelope_selected": bool(incumbent_selected and not fallback_bh),
        "incumbent_selected": bool(incumbent_selected),
        "incumbent_rule": "hold_on_supported_fallback_bh_on_unsupported" if fallback_bh else "all_hold",
        "beam_objective_before_incumbent": float(beam_objective),
        "incumbent_objective": float(incumbent_objective) if np.isfinite(incumbent_objective) else None,
        "canonical_replay_verified": True,
        "accounting_max_absolute_difference": float(max(differences.values())),
        "accounting_absolute_differences": differences, "metrics": replay}
    return targets, diagnostic


__all__ = ["matched_hindsight_targets"]
