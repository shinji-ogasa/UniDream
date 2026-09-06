"""Incremental canonical cash/units account for delayed actor intents.

At nominal t the caller reads decision_feedback(t), computes intent[t] from
already available inputs, and then advances bar t. That advance fills only
intent[t-1], charges borrowing, marks close[t], and queues intent[t] for t+1.
No policy, feature, outcome selection, forecast or model lives in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

BAR_NS = 900_000_000_000
BARS_PER_YEAR = 35040
BORROW_ANNUAL = 0.1
MAX_STEP = 0.08
DEADBAND = 0.01
MIN_TARGET, MAX_TARGET = 0.5, 1.12
SCHEMA = "wm-rl-cash-units-v1"


def execution_contract(one_way_cost=0.00055, borrow_annual=0.1):
    cost = _real(one_way_cost, "one_way_cost")
    borrowing = _real(borrow_annual, "borrow_annual")
    if not 0 <= cost < 1 / MAX_TARGET:
        raise ValueError("nonnegative cost with positive sell denominator required")
    if borrowing < 0:
        raise ValueError("nonnegative annual borrowing rate required")
    return {"schema": SCHEMA, "one_way_cost": cost,
            "borrow_annual": borrowing, "max_step": MAX_STEP,
            "deadband": DEADBAND, "intent_bounds": [MIN_TARGET, MAX_TARGET],
            "bars_per_year": BARS_PER_YEAR, "fill_delay_bars": 1,
            "initial_cash": 0.0, "initial_units": "1/initial_open",
            "initial_equity": 1.0, "missing_next_open": "expire_pending_intent",
            "decision_account_feedback": "previous_completed_bar",
            "missing_close_hold_valuation": "last_known_mark_explicitly_stale"}


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(name + " must be a finite real scalar")
    return float(value)


def _timestamp_ns(value):
    if not isinstance(value, (str, pd.Timestamp, np.datetime64)):
        raise ValueError("explicit timezone-aware timestamp required, not an epoch integer")
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tz is None:
        raise ValueError("explicit timezone-aware timestamp required")
    ns = int(ts.tz_convert("UTC").value)
    if ns % BAR_NS:
        raise ValueError("timestamp must align to UTC 15-minute bar-open grid")
    return ns


def _iso(ns):
    return None if ns is None else pd.Timestamp(ns, unit="ns", tz="UTC").isoformat()


def _observed_price(value, observed, name):
    if not isinstance(observed, (bool, np.bool_)):
        raise ValueError(name + "_observed must be boolean")
    if not observed:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isnan(float(value)):
            raise ValueError("unobserved " + name + " must be None or NaN, never a fabricated price")
        return None
    price = _real(value, name)
    if price <= 0:
        raise ValueError(name + " must be positive when observed")
    return price


def _intent(value):
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isnan(float(value)):
        return None
    target = _real(value, "intent")
    # Native actor inference is float32; do not reject its boundary rounding.
    tolerance = 2 * np.finfo(np.float32).eps * MAX_TARGET
    if target < MIN_TARGET - tolerance or target > MAX_TARGET + tolerance:
        raise ValueError("intent outside fixed [0.5,1.12] bounds")
    return min(max(target, MIN_TARGET), MAX_TARGET)


@dataclass(frozen=True)
class AccountState:
    initial_timestamp_ns: int
    initial_open: float
    one_way_cost: float
    borrow_annual: float
    cash: float
    units: float
    last_bar_timestamp_ns: int | None = None
    pending_target: float | None = None
    pending_due_ns: int | None = None
    last_mark: float = 0.0
    last_mark_timestamp_ns: int = 0
    last_mark_source: str = "initial_open"
    last_close_observed: bool = False
    last_equity: float | None = None
    last_exposure: float | None = None
    last_fill_delta: float = 0.0
    turnover: float = 0.0
    fees: float = 0.0
    borrow: float = 0.0
    trades: int = 0
    bars_processed: int = 0
    insolvent: bool = False

    def validate(self):
        execution_contract(self.one_way_cost, self.borrow_annual)
        for name in ("initial_open", "cash", "units", "last_mark", "last_fill_delta", "turnover", "fees", "borrow"):
            _real(getattr(self, name), name)
        if self.initial_open <= 0 or self.last_mark <= 0 or self.units < 0:
            raise ValueError("positive initial/mark prices and nonnegative units required")
        if min(self.turnover, self.fees, self.borrow) < 0:
            raise ValueError("negative accounting totals")
        for name in ("initial_timestamp_ns", "last_mark_timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value % BAR_NS:
                raise ValueError("invalid grid timestamp in state")
        for name in ("trades", "bars_processed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("nonnegative integer counts required")
        if self.trades > self.bars_processed:
            raise ValueError("at most one fill per bar")
        if not isinstance(self.last_close_observed, bool) or not isinstance(self.insolvent, bool):
            raise ValueError("state flags must be boolean")
        if self.last_mark_source not in ("initial_open", "open", "close"):
            raise ValueError("invalid mark provenance")
        if self.last_bar_timestamp_ns is None:
            if (self.bars_processed or self.pending_target is not None or self.pending_due_ns is not None
                    or self.trades or self.fees or self.borrow or self.turnover):
                raise ValueError("nonempty initial state")
            if (self.cash != 0.0 or self.units != 1 / self.initial_open
                    or self.last_mark != self.initial_open
                    or self.last_mark_timestamp_ns != self.initial_timestamp_ns
                    or self.last_mark_source != "initial_open"
                    or self.last_equity is not None or self.last_exposure is not None
                    or self.last_close_observed or self.last_fill_delta or self.insolvent):
                raise ValueError("initial benchmark account mismatch")
        else:
            expected = self.initial_timestamp_ns + (self.bars_processed - 1) * BAR_NS
            if (isinstance(self.last_bar_timestamp_ns, bool)
                    or not isinstance(self.last_bar_timestamp_ns, Integral)
                    or self.bars_processed < 1 or self.last_bar_timestamp_ns != expected):
                raise ValueError("account clock differs from uncompressed bar count")
            if not self.initial_timestamp_ns <= self.last_mark_timestamp_ns <= self.last_bar_timestamp_ns + BAR_NS:
                raise ValueError("mark timestamp is outside observed history")
        if (self.pending_target is None) != (self.pending_due_ns is None):
            raise ValueError("pending intent/due timestamp must be present together")
        if self.pending_target is not None:
            if _intent(self.pending_target) != self.pending_target:
                raise ValueError("serialized intent must have exact canonical bounds")
            if (isinstance(self.pending_due_ns, bool) or not isinstance(self.pending_due_ns, Integral)
                    or self.pending_due_ns != self.last_bar_timestamp_ns + BAR_NS):
                raise ValueError("pending intent must expire at the exact next open")
        if (self.last_equity is None) != (self.last_exposure is None):
            if not (self.insolvent and self.last_equity is not None and self.last_exposure is None):
                raise ValueError("closed NAV/exposure must be present together")
        for name in ("last_equity", "last_exposure"):
            if getattr(self, name) is not None:
                _real(getattr(self, name), name)
        if not self.insolvent and self.last_close_observed != (self.last_equity is not None):
            raise ValueError("closed mark availability mismatch")
        if self.last_close_observed:
            if (self.last_mark_source != "close"
                    or self.last_mark_timestamp_ns != self.last_bar_timestamp_ns + BAR_NS
                    or self.last_equity != self.cash + self.units * self.last_mark):
                raise ValueError("closed NAV/mark provenance mismatch")
            if self.last_equity > 0 and self.last_exposure != self.units * self.last_mark / self.last_equity:
                raise ValueError("closed exposure differs from actual cash/units")
        if self.insolvent and self.pending_target is not None:
            raise ValueError("insolvent account cannot retain orders")

    def to_dict(self):
        return {"schema": SCHEMA, "contract": execution_contract(self.one_way_cost, self.borrow_annual), "account": asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema", "contract", "account"} or value["schema"] != SCHEMA:
            raise ValueError("exact serialized account envelope required")
        body = value["account"]
        if not isinstance(body, dict) or set(body) != {f.name for f in fields(cls)}:
            raise ValueError("exact serialized AccountState fields required")
        candidate = cls(**body)
        candidate.validate()
        if value["contract"] != execution_contract(candidate.one_way_cost, candidate.borrow_annual):
            raise ValueError("serialized execution contract changed")
        return candidate


class CashUnitAccount:
    def __init__(self, initial_timestamp=None, initial_open=None, *, one_way_cost=0.00055,
                 borrow_annual=0.1, state=None):
        if state is not None:
            if initial_timestamp is not None or initial_open is not None:
                raise ValueError("restore state or initialize; do not mix both")
            self.state = AccountState.from_dict(state) if isinstance(state, dict) else state
            if not isinstance(self.state, AccountState):
                raise ValueError("AccountState required")
            self.state.validate()
            if self.state.one_way_cost != _real(one_way_cost, "one_way_cost"):
                raise ValueError("restored account cost differs from caller contract")
            if self.state.borrow_annual != _real(borrow_annual, "borrow_annual"):
                raise ValueError("restored account borrowing differs from caller contract")
        else:
            timestamp = _timestamp_ns(initial_timestamp)
            price = _observed_price(initial_open, True, "initial_open")
            contract = execution_contract(one_way_cost, borrow_annual)
            self.state = AccountState(timestamp, price, contract["one_way_cost"], contract["borrow_annual"], 0.0, 1.0 / price,
                                      last_mark=price, last_mark_timestamp_ns=timestamp)
            self.state.validate()

    def _next_event(self, timestamp):
        ns = _timestamp_ns(timestamp)
        state = self.state
        expected = state.initial_timestamp_ns if state.last_bar_timestamp_ns is None else state.last_bar_timestamp_ns + BAR_NS
        if ns != expected:
            raise ValueError("exact next nominal bar required; missing bars cannot be compressed")
        if state.insolvent:
            raise ValueError("account is insolvent and halted")
        return ns

    def decision_feedback(self, timestamp):
        """Read before current bar values or current intent are processed.

        A missing previous close forces the caller to skip Actor inference.
        Last-known-mark exposure is still provided solely to advance its held
        inventory clock, with explicit mark time/age and stale-valuation flags.
        Initial feedback is the pre-existing benchmark inventory.
        """
        ns = self._next_event(timestamp)
        s = self.state
        nav = s.cash + s.units * s.last_mark
        valid = math.isfinite(nav) and nav > 0
        initial = s.last_bar_timestamp_ns is None
        return {"decision_timestamp": _iso(ns), "completed_bar_timestamp": _iso(s.last_bar_timestamp_ns),
                "cash": s.cash, "units": s.units,
                "actual_exposure": s.units * s.last_mark / nav if valid else None,
                "executed_delta": s.last_fill_delta,
                "valuation_nav": nav if math.isfinite(nav) else None,
                "valuation_available": valid, "mark": s.last_mark,
                "mark_timestamp": _iso(s.last_mark_timestamp_ns),
                "mark_age_bars": (ns - s.last_mark_timestamp_ns) // BAR_NS,
                "mark_source": s.last_mark_source,
                "current_close_observed": s.last_close_observed,
                "initial_benchmark_event": initial,
                "actor_account_available": bool(valid and (initial or s.last_close_observed)),
                "stale_valuation": bool(not initial and not s.last_close_observed)}

    def advance_bar(self, timestamp, open_price, close_price, *, open_observed,
                    close_observed, intent_for_next_open=None):
        """Fill prior intent once, borrow, mark, then queue already-computed intent.

        Open availability alone controls the due fill. Missing close does not
        undo it. Missing due open expires the intent; it never rolls forward.
        Borrowing applies on every nominal bar, including entirely missing bars.
        """
        ns = self._next_event(timestamp)
        op = _observed_price(open_price, open_observed, "open")
        cl = _observed_price(close_price, close_observed, "close")
        intent = _intent(intent_for_next_open)
        s = self.state
        if s.bars_processed == 0 and op != s.initial_open:
            raise ValueError("first observed open must match initial benchmark inventory")
        cash, units = s.cash, s.units
        turnover, fees, borrow, trades = s.turnover, s.fees, s.borrow, s.trades
        nav = cash + units * op if op is not None else None
        if nav is not None and not math.isfinite(nav):
            raise ValueError("nonfinite current-open NAV")
        fill = {"due_target": s.pending_target, "status": "none", "trade_value": 0.0,
                "fee": 0.0, "turnover": 0.0, "executed_delta": 0.0,
                "exposure_before": None, "exposure_after": None}
        halted_at_open = nav is not None and nav <= 0
        if s.pending_target is not None and not halted_at_open:
            if s.pending_due_ns != ns:
                raise ValueError("pending intent has wrong due timestamp")
            if op is None:
                fill["status"] = "expired_missing_open"
            else:
                old_exposure = units * op / nav
                desired = min(max(s.pending_target, 0.0), MAX_TARGET)
                change = min(max(desired - old_exposure, -MAX_STEP), MAX_STEP)
                fill.update(status="deadband_hold", exposure_before=old_exposure, exposure_after=old_exposure)
                if abs(change) >= DEADBAND:
                    desired = old_exposure + change
                    x = (desired * nav - units * op) / (1 + s.one_way_cost * desired * (1 if change > 0 else -1))
                    fee = s.one_way_cost * abs(x)
                    cash -= x + fee
                    units += x / op
                    turnover += abs(x) / nav
                    fees += fee
                    trades += 1
                    after = units * op / (cash + units * op)
                    fill.update(status="filled", trade_value=x, fee=fee, turnover=abs(x)/nav,
                                exposure_after=after, executed_delta=after-old_exposure)
        charge = 0.0
        if not halted_at_open and cash < 0:
            charge = -cash * (math.exp(s.borrow_annual / BARS_PER_YEAR) - 1)
            cash -= charge
            borrow += charge
        equity = None if halted_at_open or cl is None else cash + units * cl
        if equity is not None and not math.isfinite(equity):
            raise ValueError("nonfinite closed NAV")
        insolvent = bool(halted_at_open or (equity is not None and equity <= 0))
        exposure = units * cl / equity if equity is not None and equity > 0 else None
        mark, mark_ns, mark_source = s.last_mark, s.last_mark_timestamp_ns, s.last_mark_source
        if not halted_at_open:
            if cl is not None:
                mark, mark_ns, mark_source = cl, ns + BAR_NS, "close"
            elif op is not None:
                mark, mark_ns, mark_source = op, ns, "open"
        new_state = replace(s, cash=cash, units=units, last_bar_timestamp_ns=ns,
            pending_target=None if insolvent else intent,
            pending_due_ns=None if insolvent or intent is None else ns + BAR_NS,
            last_mark=mark, last_mark_timestamp_ns=mark_ns, last_mark_source=mark_source,
            last_close_observed=bool(cl is not None and not halted_at_open),
            last_equity=equity, last_exposure=exposure, last_fill_delta=fill["executed_delta"],
            turnover=turnover, fees=fees, borrow=borrow, trades=trades,
            bars_processed=s.bars_processed + 1, insolvent=insolvent)
        new_state.validate()
        self.state = new_state
        return {"bar_timestamp": _iso(ns), "equity": equity, "exposure": exposure,
                "open_nav_before_fill": nav, "fill": fill, "borrow_charge": charge,
                "cash": cash, "units": units, "insolvent": insolvent,
                "pending_target": new_state.pending_target, "pending_due": _iso(new_state.pending_due_ns),
                "turnover": turnover, "fees": fees, "borrow": borrow, "trades": trades,
                "open_observed": bool(open_observed), "close_observed": bool(close_observed)}


__all__ = ["AccountState", "CashUnitAccount", "execution_contract"]
