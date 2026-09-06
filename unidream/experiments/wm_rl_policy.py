"""Causal fixed-context WM inputs and a persistent actor intent adapter.

No accounting takes place here. Physical mode consumes execution feedback and
never treats an issued intent as a fill. The optional ideal-intent mode exists
only for structural comparison with Actor.predict_positions, not evaluation or
promotion. Features already describe information available at their origin;
this module adds no shift and never reads outcomes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
import torch

BAR_NS = 15 * 60 * 1_000_000_000
CONTEXT_LENGTH = 64


@contextmanager
def _evaluation(module):
    modes = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for child, training in modes:
            child.training = training


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values, not booleans or objects")
    return array


def _index(timestamps):
    if not isinstance(timestamps, pd.DatetimeIndex) or timestamps.tz is None:
        raise ValueError("explicit timezone-aware bar-open DatetimeIndex required")
    if timestamps.hasnans or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be finite, unique and increasing")
    index = timestamps.tz_convert("UTC")
    ns = index.as_unit("ns").asi8
    if np.any(ns % BAR_NS):
        raise ValueError("timestamps must be on the UTC 15-minute grid")
    return index, ns


def _window_mask(row_good, ns):
    result = np.zeros(len(row_good), dtype=bool)
    if len(row_good) < CONTEXT_LENGTH:
        return result
    bad = np.r_[0, np.cumsum(~row_good)]
    edge_bad = np.r_[0, np.cumsum(np.diff(ns) != BAR_NS)]
    good = (bad[CONTEXT_LENGTH:] - bad[:-CONTEXT_LENGTH]) == 0
    good &= (edge_bad[CONTEXT_LENGTH - 1:] - edge_bad[:-(CONTEXT_LENGTH - 1)]) == 0
    result[CONTEXT_LENGTH - 1:] = good
    return result


def encode_fixed_context(ensemble, features, timestamps, full_feature_eligible, *,
                         context_length=64, batch_size=64, device="cpu"):
    """Encode precisely [t-63, t], with benchmark-one action context.

    Returned z/h keep the full input index and are explicit NaN when unavailable.
    Availability requires all 64 rows to be source-eligible, finite after float32
    conversion and exactly 15 minutes apart. No interpolation or zero filling.
    """
    if isinstance(context_length, bool) or context_length != CONTEXT_LENGTH:
        raise ValueError("the frozen context length is exactly 64")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("positive integer batch_size required")
    values = _real_array(features, "features")
    index, ns = _index(timestamps)
    eligible = np.asarray(full_feature_eligible)
    if values.ndim != 2 or values.shape[0] != len(index) or values.shape[1] < 1:
        raise ValueError("features must have shape (timestamps, positive feature dimension)")
    if eligible.dtype != np.bool_ or eligible.shape != (len(index),):
        raise ValueError("full_feature_eligible must be a same-length strict boolean mask")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32)
    available = _window_mask(eligible & np.isfinite(values).all(axis=1), ns)
    z_dim, h_dim = int(ensemble.get_z_dim()), int(ensemble.get_d_model())
    if z_dim < 1 or h_dim < 1:
        raise ValueError("positive model latent dimensions required")
    z_out = np.full((len(index), z_dim), np.nan, dtype=np.float32)
    h_out = np.full((len(index), h_dim), np.nan, dtype=np.float32)
    origins = np.flatnonzero(available)
    with _evaluation(ensemble):
        for offset in range(0, len(origins), batch_size):
            selected = origins[offset:offset + batch_size]
            windows = np.stack([values[t - CONTEXT_LENGTH + 1:t + 1] for t in selected])
            observation = torch.as_tensor(windows, dtype=torch.float32, device=device)
            actions = torch.ones((len(selected), CONTEXT_LENGTH, 1), dtype=torch.float32, device=device)
            z, _ = ensemble.encode(observation)
            out = ensemble.forward(z, actions)
            h = out["h"]
            if z.shape != (len(selected), CONTEXT_LENGTH, z_dim) or h.shape != (len(selected), CONTEXT_LENGTH, h_dim):
                raise ValueError("WM output dimensions differ from its fixed-context contract")
            z_last = z[:, -1].detach().cpu().numpy()
            h_last = h[:, -1].detach().cpu().numpy()
            if not np.isfinite(z_last).all() or not np.isfinite(h_last).all():
                raise ValueError("WM produced nonfinite output for a valid context")
            z_out[selected], h_out[selected] = z_last, h_last
    return {"z": z_out, "h": h_out, "available": available,
            "timestamps": index, "context_length": CONTEXT_LENGTH,
            "context_action": 1.0, "feature_dtype": "float32"}


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


@dataclass(frozen=True)
class PolicyState:
    controller: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    step_count: int = 0
    active_count: int = 0
    underweight_count: int = 0
    long_count: int = 0
    last_timestamp_ns: int | None = None
    physical_feedback: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("exact PolicyState schema required")
        candidate = cls(**value)
        candidate.validate()
        return cls(**{**value, "controller": tuple(value["controller"])})

    def validate(self):
        if len(self.controller) != 4:
            raise ValueError("exact four-dimensional controller state required")
        for item in self.controller:
            _scalar(item, "controller")
        if not 0 <= self.controller[2] <= 1 or not 0 <= self.controller[3] <= 1:
            raise ValueError("controller duration fields must be in [0,1]")
        counts = (self.step_count, self.active_count, self.underweight_count, self.long_count)
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in counts):
            raise ValueError("nonnegative integer counters required")
        if max(counts[1:]) > self.step_count or self.underweight_count + self.long_count > self.active_count:
            raise ValueError("inconsistent policy counters")
        if not isinstance(self.physical_feedback, bool):
            raise ValueError("physical_feedback must be boolean")
        if self.last_timestamp_ns is None:
            if self.step_count:
                raise ValueError("nonempty state must include its last timestamp")
        elif (isinstance(self.last_timestamp_ns, bool) or not isinstance(self.last_timestamp_ns, Integral)
              or self.last_timestamp_ns % BAR_NS or self.step_count == 0):
            raise ValueError("last timestamp must be a nanosecond grid timestamp for a nonempty state")


class IncrementalActorPolicy:
    """Persistent native actor intents with optional actual execution feedback.

    Each call represents one consecutive 15-minute event. In physical mode,
    actual_exposure and executed_delta are both mandatory. executed_delta is the
    exposure change caused by fills since the preceding event, excluding price
    drift. The first state component always reflects actual exposure; a proposed
    target never updates it. The next three fields use the native controller
    update with this observed fill delta. A first event also advances durations.

    Rate-cap counters follow native issued intents on valid events, and held
    exposure on unavailable events. They are diagnostic decision counters, not
    fills or accounting turnover. The caller owns cash, units, costs, pending
    orders and actual limits. No-account mode is explicitly diagnostic only.
    """
    def __init__(self, actor, *, device="cpu", state=None, physical_feedback=True):
        if not isinstance(physical_feedback, bool):
            raise ValueError("physical_feedback must be boolean")
        if actor.inventory_dim != 4 or actor._benchmark_position() != 1.0:
            raise ValueError("four-state benchmark-one actor required")
        if _scalar(actor._state_hold_scale(), "hold state scale") <= 0:
            raise ValueError("positive hold state scale required")
        self.actor = actor
        self.device = torch.device(device)
        self.state = PolicyState(physical_feedback=physical_feedback) if state is None else (
            PolicyState.from_dict(state) if isinstance(state, dict) else state)
        if not isinstance(self.state, PolicyState):
            raise ValueError("PolicyState required")
        self.state.validate()
        if self.state.physical_feedback != physical_feedback:
            raise ValueError("cannot switch physical/diagnostic semantics when restoring state")

    def _vector(self, value, size, name):
        if size == 0:
            if value is not None and np.asarray(value).size:
                raise ValueError(f"{name} is not configured on this actor")
            return None
        if value is None:
            return None
        array = _real_array(value, name)
        if array.shape != (size,):
            raise ValueError(f"{name} must have exact shape ({size},)")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32)
        if not np.isfinite(array).all():
            return None
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _flags(self, position):
        overlay = position - 1.0
        eps = float(getattr(self.actor, "rate_cap_active_eps", 0.05))
        return abs(overlay) > eps, overlay < -eps, overlay > eps

    def _native_outer_intent(self, position, state):
        # Actor.predict_positions has no standalone helper for this outer loop.
        # Preserve its quantization, neutral snap and ceil-based rate caps.
        tensor = self.actor._quantize_inference(position)
        value = _scalar(float(tensor.item()), "actor intent")
        snap = float(getattr(self.actor, "benchmark_neutral_snap_eps", 0.0))
        if snap > 0.0 and abs(value - 1.0) <= snap:
            value = 1.0
        flags = self._flags(value)
        counts = (state.active_count, state.underweight_count, state.long_count)
        names = ("active_rate_max", "short_underweight_rate_max", "benchmark_overweight_long_rate_max")
        for flag, count, name in zip(flags, counts, names):
            cap = float(getattr(self.actor, name, 0.0))
            if 0 < cap < 1 and flag and count + 1 > math.ceil(cap * (state.step_count + 1)):
                return 1.0
        return value

    def step(self, timestamp, z=None, h=None, *, available, regime=None, advantage=None,
             actual_exposure=None, executed_delta=None):
        if not isinstance(available, (bool, np.bool_)):
            raise ValueError("available must be boolean")
        if not isinstance(timestamp, (pd.Timestamp, str, np.datetime64)):
            raise ValueError("explicit datetime timestamp required, not an epoch integer")
        ts, ns_array = _index(pd.DatetimeIndex([pd.Timestamp(timestamp)]))
        ns = int(ns_array[0])
        state = self.state
        if state.last_timestamp_ns is not None and ns - state.last_timestamp_ns != BAR_NS:
            raise ValueError("exact next 15-minute event required; emit unavailable events through gaps")
        physical = state.physical_feedback
        if physical:
            exposure = _scalar(actual_exposure, "actual_exposure")
            delta = _scalar(executed_delta, "executed_delta")
            if max(abs(exposure), abs(delta), abs(exposure - 1.0 - delta)) > np.finfo(np.float32).max:
                raise ValueError("physical feedback must be representable in float32")
        elif actual_exposure is not None or executed_delta is not None:
            raise ValueError("physical feedback pair is forbidden in diagnostic mode")
        else:
            exposure, delta = float(state.controller[0]) + 1.0, 0.0
        control = torch.tensor([state.controller], dtype=torch.float32, device=self.device)
        if physical:
            # Reconstruct the pre-fill overlay at this event's valuation price.
            # This excludes mark-to-market drift from the native traded flag.
            control[0, 0] = exposure - 1.0 - delta
            control = self.actor.update_controller_state(
                control, torch.tensor([[exposure]], dtype=torch.float32, device=self.device))
            # Preserve supplied exposure/delta even where subtraction roundoff
            # would manufacture a tiny fill. Durations use the explicit delta.
            control[0, 0], control[0, 1] = exposure - 1.0, delta
            eps = self.actor._trade_state_eps()
            scale = self.actor._state_hold_scale()
            control[0, 2] = 0.0 if abs(delta) > eps else min(state.controller[2] + 1.0 / scale, 1.0)
            control[0, 3] = min(state.controller[3] + 1.0 / scale, 1.0) if exposure - 1.0 < -eps else 0.0
        target = None
        reason = "feature_unavailable"
        if available:
            z_t = self._vector(z, np.asarray(z).size if z is not None else 1, "z")
            h_t = self._vector(h, np.asarray(h).size if h is not None else 1, "h")
            reg_t = self._vector(regime, self.actor.regime_dim, "regime")
            adv_t = self._vector(advantage, self.actor.advantage_dim, "advantage")
            valid = (z_t is not None and h_t is not None
                     and (self.actor.regime_dim == 0 or reg_t is not None)
                     and (self.actor.advantage_dim == 0 or adv_t is not None))
            if valid:
                with _evaluation(self.actor):
                    position = self.actor.act_greedy(z_t, h_t, inventory=control,
                                                     regime=reg_t, advantage=adv_t)
                    if position.shape != (1, 1):
                        raise ValueError("actor must return one scalar intent shaped (1,1)")
                    target = self._native_outer_intent(position, state)
                low, high = self.actor._absolute_bounds()
                # Preserve the native float32 result, including its boundary
                # rounding. The actual execution engine owns decimal bounds.
                tolerance = 2 * np.finfo(np.float32).eps * max(1.0, abs(low), abs(high))
                if not low - tolerance <= target <= high + tolerance:
                    raise ValueError("actor intent outside absolute bounds")
                reason = "actor_intent"
            else:
                reason = "model_input_unavailable"
        if not physical:
            held_or_intended = exposure if target is None else target
            control = self.actor.update_controller_state(
                control, torch.tensor([[held_or_intended]], dtype=torch.float32, device=self.device))
        flags = self._flags(exposure if target is None else target)
        next_state = PolicyState(
            controller=tuple(float(v) for v in control.detach().cpu().numpy()[0]),
            step_count=state.step_count + 1,
            active_count=state.active_count + int(flags[0]),
            underweight_count=state.underweight_count + int(flags[1]),
            long_count=state.long_count + int(flags[2]),
            last_timestamp_ns=ns, physical_feedback=physical)
        next_state.validate()
        self.state = next_state
        return {"timestamp": ts[0].isoformat(), "target_intent": target,
                "available": target is not None, "reason": reason,
                "state": next_state.to_dict(), "physical_feedback": physical,
                "rate_counter_basis": "issued_intent" if target is not None else "held_exposure",
                "intent_is_fill": False,
                "diagnostic_only": not physical}
