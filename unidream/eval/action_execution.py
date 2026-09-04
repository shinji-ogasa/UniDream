"""Single source of truth for the conditional-Oracle action contract.

The historical research path has several action grids, execution delays and
cost defaults.  This module is deliberately independent from the historical
Oracle and Backtest implementations so the new conditional-Oracle path can
opt in to one immutable contract without changing historical results.

The contract is spot-only and intentionally small:

``decision t -> fill t+1 -> earn returns[t+1:t+5]``

The final block is scored only when all four returns are present.  A trajectory
contains full-length diagnostic arrays, plus an explicit ``scored_mask``; the
Backtest adapter trims to that mask before computing metrics.  A missing
decision feature produces a scored hold commitment.  A missing fill bar
prevents execution, while a later outcome gap preserves the already executed
inventory and excludes only retrospective scoring.  The same replay geometry
is used by conditional teachers and by the upper-bound diagnostic, while their
selectors remain separate: the teacher is causal and U0 is hindsight-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np


_FLOAT_TOL = 1e-9
_CANONICAL_CANDIDATE_DELTAS = (-0.08, -0.04, 0.0, 0.04, 0.08)
_CANONICAL_POSITION_MIN = 0.50
_CANONICAL_POSITION_MAX = 1.00
_CANONICAL_H_DECISION = 4
_CANONICAL_COMMITMENT_BARS = 4
_CANONICAL_EXECUTION_DELAY_BARS = 1
_CANONICAL_P_START = 1.00
_CANONICAL_INITIAL_COUNTDOWN = 0


def _as_real(value: Any, *, name: str) -> float:
    """Accept real numeric scalars only; reject bools and numeric strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _as_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a finite numeric sequence")
    try:
        result = tuple(_as_real(value, name=name) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence") from exc
    if not result:
        raise ValueError(f"{name} must be a non-empty finite sequence")
    return result


def _require_integer(value: Any, *, name: str, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        qualifier = "positive" if minimum > 0 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _mapping_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the explicitly named new-path contract section.

    We accept the few names used by experiment manifests, but never infer the
    contract from the historical ``costs`` section.  This is important: a
    missing new-path contract must fail closed instead of silently inheriting
    the old 5/2/0.0004 defaults.
    """
    direct_keys = [
        key
        for key in ("action_execution_contract", "action_execution")
        if config.get(key) is not None
    ]
    if len(direct_keys) > 1:
        raise ValueError("action execution config contains duplicate contract sections")
    conditional = config.get("conditional_oracle")
    # A direct contract plus any conditional wrapper is ambiguous even when
    # the wrapper contains only an enable flag or unrelated fields.  Do not
    # silently select the direct section and discard the other source.
    if direct_keys and conditional is not None:
        raise ValueError("action execution config contains duplicate contract sections")
    if isinstance(conditional, Mapping):
        conditional_keys = [
            key
            for key in ("action_execution_contract", "action_execution")
            if conditional.get(key) is not None
        ]
        if len(conditional_keys) > 1 or (direct_keys and conditional_keys):
            raise ValueError("action execution config contains duplicate contract sections")
        if conditional_keys:
            key = conditional_keys[0]
            value = conditional[key]
            if not isinstance(value, Mapping):
                raise ValueError(f"conditional_oracle.{key} must be a mapping")
            return value
    if direct_keys:
        key = direct_keys[0]
        value = config[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a mapping")
        return value
    if isinstance(conditional, Mapping):
        return conditional
    raise ValueError(
        "new action-execution path requires an explicit action_execution_contract"
    )


@dataclass(frozen=True, slots=True)
class ActionExecutionContract:
    """Immutable execution/action/cost contract for the new Oracle path.

    The defaults are the P0-C canonical values.  ``from_config`` is stricter:
    it requires every semantic field to be present and rejects a non-canonical
    contract.  Direct construction remains useful for deterministic fixtures,
    but unsupported semantics (funding, partial fills, non-additive returns)
    are still rejected.
    """

    position_min: float = 0.50
    position_max: float = 1.00
    candidate_deltas: tuple[float, ...] = field(
        default_factory=lambda: _CANONICAL_CANDIDATE_DELTAS
    )
    h_decision: int = 4
    commitment_bars: int = 4
    execution_delay_bars: int = 1
    fill_policy: str = "all_or_none"
    partial_fill_policy: str = "unsupported"
    tail_policy: str = "exclude_incomplete"
    spread_bps: float = 3.0
    spread_convention: str = "full_quoted"
    slippage_bps: float = 1.0
    fee_rate: float = 0.0003
    return_unit: str = "additive_log_return"
    funding_included: bool = False
    p_start: float = 1.00
    initial_countdown: int = 0
    countdown_decrement: int = 1
    boundary_cost_policy: str = "fill_only"
    feature_unavailable_policy: str = "hold_and_score_commitment"
    outcome_unavailable_policy: str = "exclude_block"
    execution_skip_policy: str = "hold_commitment"
    eligibility_masks_required: bool = True

    def __post_init__(self) -> None:
        numeric = {
            "position_min": self.position_min,
            "position_max": self.position_max,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "fee_rate": self.fee_rate,
            "p_start": self.p_start,
        }
        for name, value in numeric.items():
            object.__setattr__(self, name, _as_real(value, name=name))
        if self.spread_bps < 0.0:
            raise ValueError("spread_bps must be non-negative")
        if self.slippage_bps < 0.0:
            raise ValueError("slippage_bps must be non-negative")
        if self.fee_rate < 0.0:
            raise ValueError("fee_rate must be non-negative")
        if float(self.position_min) > float(self.position_max):
            raise ValueError("position_min must be <= position_max")
        if not float(self.position_min) <= float(self.p_start) <= float(self.position_max):
            raise ValueError("p_start must lie within position bounds")
        _require_integer(self.h_decision, name="h_decision", minimum=1)
        _require_integer(self.commitment_bars, name="commitment_bars", minimum=1)
        _require_integer(self.execution_delay_bars, name="execution_delay_bars", minimum=0)
        _require_integer(self.initial_countdown, name="initial_countdown", minimum=0)
        _require_integer(self.countdown_decrement, name="countdown_decrement", minimum=1)
        if self.h_decision != self.commitment_bars:
            raise ValueError("h_decision must equal commitment_bars")
        if self.execution_delay_bars != 1:
            raise ValueError("execution_delay_bars must be 1")
        if self.initial_countdown > self.commitment_bars:
            raise ValueError("initial_countdown must not exceed commitment_bars")
        deltas = _as_float_tuple(self.candidate_deltas, name="candidate_deltas")
        if not any(abs(delta) <= _FLOAT_TOL for delta in deltas):
            raise ValueError("candidate_deltas must include the hold delta 0.0")
        if len({round(delta, 12) for delta in deltas}) != len(deltas):
            raise ValueError("candidate_deltas must not contain duplicate values")
        object.__setattr__(self, "candidate_deltas", deltas)
        if not np.isclose(
            self.position_min,
            _CANONICAL_POSITION_MIN,
            atol=_FLOAT_TOL,
            rtol=0.0,
        ) or not np.isclose(
            self.position_max,
            _CANONICAL_POSITION_MAX,
            atol=_FLOAT_TOL,
            rtol=0.0,
        ):
            raise ValueError("position bounds must be the canonical [0.5, 1.0]")
        if len(deltas) != len(_CANONICAL_CANDIDATE_DELTAS) or not all(
            np.isclose(a, b, atol=_FLOAT_TOL, rtol=0.0)
            for a, b in zip(deltas, _CANONICAL_CANDIDATE_DELTAS)
        ):
            raise ValueError("candidate_deltas must use the canonical P0-C grid")
        if self.h_decision != _CANONICAL_H_DECISION:
            raise ValueError("h_decision must be 4 for the registered P0-C contract")
        if self.commitment_bars != _CANONICAL_COMMITMENT_BARS:
            raise ValueError("commitment_bars must be 4 for the registered P0-C contract")
        if self.execution_delay_bars != _CANONICAL_EXECUTION_DELAY_BARS:
            raise ValueError("execution_delay_bars must be 1 for the registered P0-C contract")
        if not np.isclose(self.p_start, _CANONICAL_P_START, atol=_FLOAT_TOL, rtol=0.0):
            raise ValueError("p_start must be 1.0 for the registered P0-C contract")
        if self.initial_countdown != _CANONICAL_INITIAL_COUNTDOWN:
            raise ValueError("initial_countdown must be 0 for the registered P0-C contract")
        for name, value in (
            ("fill_policy", self.fill_policy),
            ("partial_fill_policy", self.partial_fill_policy),
            ("tail_policy", self.tail_policy),
            ("spread_convention", self.spread_convention),
            ("return_unit", self.return_unit),
            ("boundary_cost_policy", self.boundary_cost_policy),
            ("feature_unavailable_policy", self.feature_unavailable_policy),
            ("outcome_unavailable_policy", self.outcome_unavailable_policy),
            ("execution_skip_policy", self.execution_skip_policy),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.fill_policy != "all_or_none":
            raise ValueError("only all_or_none fills are supported by the new path")
        if self.partial_fill_policy != "unsupported":
            raise ValueError("partial fills are unsupported by the new path")
        if self.tail_policy != "exclude_incomplete":
            raise ValueError("only exclude_incomplete tail policy is supported")
        if self.spread_convention != "full_quoted":
            raise ValueError("spread_bps must use the full_quoted convention")
        if self.return_unit != "additive_log_return":
            raise ValueError("new path requires additive_log_return units")
        if not isinstance(self.funding_included, (bool, np.bool_)):
            raise ValueError("funding_included must be a boolean")
        if bool(self.funding_included):
            raise ValueError("funding is excluded from the spot first-pass contract")
        if self.boundary_cost_policy != "fill_only":
            raise ValueError("boundary_cost_policy must be fill_only")
        if self.countdown_decrement != 1:
            raise ValueError("countdown_decrement must be 1")
        if self.feature_unavailable_policy != "hold_and_score_commitment":
            raise ValueError(
                "feature_unavailable_policy must be hold_and_score_commitment"
            )
        if self.outcome_unavailable_policy != "exclude_block":
            raise ValueError("outcome_unavailable_policy must be exclude_block")
        if self.execution_skip_policy != "hold_commitment":
            raise ValueError("execution_skip_policy must be hold_commitment")
        if not isinstance(self.eligibility_masks_required, (bool, np.bool_)):
            raise ValueError("eligibility_masks_required must be a boolean")
        if not bool(self.eligibility_masks_required):
            raise ValueError("eligibility_masks_required must be true for the new path")

    @classmethod
    def canonical(cls) -> "ActionExecutionContract":
        """Return the canonical P0-C contract."""
        return cls()

    @property
    def H_decision(self) -> int:
        return self.h_decision

    @property
    def delay(self) -> int:
        return self.execution_delay_bars

    @property
    def position_bounds(self) -> tuple[float, float]:
        return (float(self.position_min), float(self.position_max))

    @property
    def min_position(self) -> float:
        return float(self.position_min)

    @property
    def max_position(self) -> float:
        return float(self.position_max)

    @property
    def delta_grid(self) -> tuple[float, ...]:
        return self.candidate_deltas

    @property
    def countdown_reset(self) -> int:
        return int(self.commitment_bars)

    @property
    def transition_cost_rate(self) -> float:
        """Cost per unit of absolute position change in return units."""
        return float(
            self.spread_bps / 10000.0 / 2.0
            + self.slippage_bps / 10000.0
            + self.fee_rate
        )

    @property
    def cost_per_position_delta(self) -> float:
        return self.transition_cost_rate

    def to_dict(self) -> dict[str, Any]:
        """Serialize all semantic fields and derived cost units."""
        return {
            "position_min": float(self.position_min),
            "position_max": float(self.position_max),
            "candidate_deltas": [float(value) for value in self.candidate_deltas],
            "h_decision": int(self.h_decision),
            "commitment_bars": int(self.commitment_bars),
            "commitment_countdown_reset": int(self.commitment_bars),
            "commitment_countdown_decrement": int(self.countdown_decrement),
            "countdown_decrement": int(self.countdown_decrement),
            "execution_delay_bars": int(self.execution_delay_bars),
            "fill_policy": self.fill_policy,
            "partial_fill_policy": self.partial_fill_policy,
            "tail_policy": self.tail_policy,
            "spread_bps": float(self.spread_bps),
            "spread_convention": self.spread_convention,
            "spread_side": "half_transition",
            "slippage_bps": float(self.slippage_bps),
            "fee_rate": float(self.fee_rate),
            "transition_cost_rate": self.transition_cost_rate,
            "return_unit": self.return_unit,
            "funding_included": bool(self.funding_included),
            "p_start": float(self.p_start),
            "initial_countdown": int(self.initial_countdown),
            "boundary_cost_policy": self.boundary_cost_policy,
            "feature_unavailable_policy": self.feature_unavailable_policy,
            "outcome_unavailable_policy": self.outcome_unavailable_policy,
            "execution_skip_policy": self.execution_skip_policy,
            "eligibility_masks_required": bool(self.eligibility_masks_required),
        }

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def hash(self) -> str:
        """Short alias useful in artifact metadata and tests."""
        return self.contract_hash

    @property
    def digest(self) -> str:
        return self.contract_hash

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def candidate_positions(self, current_position: float) -> np.ndarray:
        return candidate_positions(current_position, self)

    def transition_cost(self, previous_position: float, next_position: float) -> float:
        return transition_cost(previous_position, next_position, self)

    def replay(
        self,
        returns: np.ndarray | Sequence[float],
        decision_deltas: np.ndarray | Sequence[float],
        *,
        decision_eligible: np.ndarray | Sequence[bool] | None = None,
        score_eligible: np.ndarray | Sequence[bool] | None = None,
        bar_available: np.ndarray | Sequence[bool] | None = None,
        common_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> "ActionExecutionTrajectory":
        return replay_action_path(
            returns,
            decision_deltas,
            self,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            bar_available=bar_available,
            common_mask=common_mask,
        )

    def replay_absolute(
        self,
        returns: np.ndarray | Sequence[float],
        absolute_positions: np.ndarray | Sequence[float],
        *,
        decision_eligible: np.ndarray | Sequence[bool] | None = None,
        score_eligible: np.ndarray | Sequence[bool] | None = None,
        bar_available: np.ndarray | Sequence[bool] | None = None,
        forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
        common_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> "ActionExecutionTrajectory":
        return replay_contract_absolute_path(
            returns,
            absolute_positions,
            self,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            bar_available=bar_available,
            forecast_finite_mask=forecast_finite_mask,
            common_mask=common_mask,
        )

    def select_decisions(
        self,
        decision_block_scores: np.ndarray | Sequence[float],
        *,
        decision_eligible: np.ndarray | Sequence[bool] | None = None,
        score_eligible: np.ndarray | Sequence[bool] | None = None,
        bar_available: np.ndarray | Sequence[bool] | None = None,
    ) -> np.ndarray:
        return select_block_decisions(
            decision_block_scores,
            self,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            bar_available=bar_available,
        )

    def __hash__(self) -> int:
        # Explicitly hash the canonical serialized contract so future additions
        # to the dataclass cannot accidentally omit a semantic field.
        return hash(self.contract_hash)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        require_canonical: bool = True,
    ) -> "ActionExecutionContract":
        """Parse an explicit new-path contract, rejecting missing fields.

        ``config`` may be the whole manifest or the contract section itself.
        Legacy cost keys are intentionally not consulted.
        """
        if not isinstance(config, Mapping):
            raise ValueError("action execution config must be a mapping")
        if any(key in config for key in ("action_execution_contract", "action_execution", "conditional_oracle")):
            section = _mapping_section(config)
        else:
            section = config

        required = {
            "position_min",
            "position_max",
            "candidate_deltas",
            "h_decision",
            "commitment_bars",
            "execution_delay_bars",
            "fill_policy",
            "partial_fill_policy",
            "tail_policy",
            "spread_bps",
            "spread_convention",
            "slippage_bps",
            "fee_rate",
            "return_unit",
            "funding_included",
            "p_start",
            "initial_countdown",
            "countdown_decrement",
            "boundary_cost_policy",
            "feature_unavailable_policy",
            "outcome_unavailable_policy",
            "execution_skip_policy",
            "eligibility_masks_required",
        }
        aliases = {
            "H_decision": "h_decision",
            "delta_grid": "candidate_deltas",
            "min_position": "position_min",
            "max_position": "position_max",
            "delay": "execution_delay_bars",
            "countdown_reset": "commitment_bars",
        }
        if any(not isinstance(key, str) for key in section):
            raise ValueError("action execution contract keys must be strings")
        derived_fields = {
            "commitment_countdown_reset",
            "commitment_countdown_decrement",
            "spread_side",
            "transition_cost_rate",
        }
        allowed = required | set(aliases) | derived_fields
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ValueError(
                "action execution contract contains unknown fields: "
                + ", ".join(unknown)
            )
        normalized = dict(section)
        for source, target in aliases.items():
            if source in normalized and target in normalized:
                raise ValueError(
                    f"action execution contract contains duplicate alias fields {source}/{target}"
                )
            if source in normalized:
                normalized[target] = normalized.pop(source)
        missing = sorted(key for key in required if key not in normalized)
        if missing:
            raise ValueError(
                "action execution contract missing required fields: " + ", ".join(missing)
            )
        contract = cls(
            position_min=normalized["position_min"],
            position_max=normalized["position_max"],
            candidate_deltas=tuple(normalized["candidate_deltas"]),
            h_decision=normalized["h_decision"],
            commitment_bars=normalized["commitment_bars"],
            execution_delay_bars=normalized["execution_delay_bars"],
            fill_policy=normalized["fill_policy"],
            partial_fill_policy=normalized["partial_fill_policy"],
            tail_policy=normalized["tail_policy"],
            spread_bps=normalized["spread_bps"],
            spread_convention=normalized["spread_convention"],
            slippage_bps=normalized["slippage_bps"],
            fee_rate=normalized["fee_rate"],
            return_unit=normalized["return_unit"],
            funding_included=normalized["funding_included"],
            p_start=normalized["p_start"],
            initial_countdown=normalized["initial_countdown"],
            countdown_decrement=normalized.get("countdown_decrement", 1),
            boundary_cost_policy=normalized["boundary_cost_policy"],
            feature_unavailable_policy=normalized["feature_unavailable_policy"],
            outcome_unavailable_policy=normalized["outcome_unavailable_policy"],
            execution_skip_policy=normalized["execution_skip_policy"],
            eligibility_masks_required=normalized["eligibility_masks_required"],
        )
        derived_expected = {
            "commitment_countdown_reset": int(contract.commitment_bars),
            "commitment_countdown_decrement": int(contract.countdown_decrement),
            "spread_side": "half_transition",
            "transition_cost_rate": float(contract.transition_cost_rate),
        }
        for field_name, expected in derived_expected.items():
            if field_name not in section:
                continue
            actual = section[field_name]
            if isinstance(expected, float):
                try:
                    equal = bool(
                        np.isclose(
                            _as_real(actual, name=field_name),
                            expected,
                            atol=_FLOAT_TOL,
                            rtol=0.0,
                        )
                    )
                except ValueError:
                    equal = False
            else:
                equal = actual == expected and type(actual) is type(expected)
            if not equal:
                raise ValueError(
                    f"action execution contract derived field {field_name} must equal {expected!r}"
                )
        if require_canonical:
            canonical = cls.canonical()
            fields = (
                "position_min",
                "position_max",
                "candidate_deltas",
                "h_decision",
                "commitment_bars",
                "execution_delay_bars",
                "fill_policy",
                "partial_fill_policy",
                "tail_policy",
                "spread_bps",
                "spread_convention",
                "slippage_bps",
                "fee_rate",
                "return_unit",
                "funding_included",
                "p_start",
                "initial_countdown",
                "countdown_decrement",
                "boundary_cost_policy",
                "feature_unavailable_policy",
                "outcome_unavailable_policy",
                "execution_skip_policy",
                "eligibility_masks_required",
            )
            for field_name in fields:
                actual = getattr(contract, field_name)
                expected = getattr(canonical, field_name)
                if isinstance(expected, tuple):
                    equal = len(actual) == len(expected) and all(
                        np.isclose(a, b, atol=_FLOAT_TOL, rtol=0.0)
                        for a, b in zip(actual, expected)
                    )
                elif isinstance(expected, float):
                    equal = bool(np.isclose(actual, expected, atol=_FLOAT_TOL, rtol=0.0))
                else:
                    equal = actual == expected
                if not equal:
                    raise ValueError(
                        f"action execution contract field {field_name} must equal "
                        f"the canonical P0-C value ({expected!r})"
                    )
        return contract


DEFAULT_ACTION_EXECUTION_CONTRACT = ActionExecutionContract.canonical()
P0_C_ACTION_EXECUTION_CONTRACT = DEFAULT_ACTION_EXECUTION_CONTRACT


def action_execution_contract_from_config(
    config: Mapping[str, Any],
    *,
    require_canonical: bool = True,
) -> ActionExecutionContract:
    """Explicit parser used by stage adapters; never falls back to legacy config."""
    return ActionExecutionContract.from_config(config, require_canonical=require_canonical)


def configured_action_execution_contract(
    config: Mapping[str, Any],
    *,
    require_canonical: bool = True,
) -> ActionExecutionContract | None:
    """Return the new contract only when a manifest explicitly opts in.

    Ordinary historical manifests return ``None``.  A manifest that advertises
    the conditional/contract path but omits its section raises instead of
    falling back to legacy action, delay, or cost defaults.
    """
    if not isinstance(config, Mapping):
        raise ValueError("experiment config must be a mapping")
    raw_enabled = config.get("use_action_execution_contract", False)
    if not isinstance(raw_enabled, (bool, np.bool_)):
        raise ValueError("use_action_execution_contract must be a boolean")
    enabled = bool(raw_enabled)
    enabled = enabled or any(
        key in config for key in ("action_execution_contract", "action_execution")
    )
    conditional = config.get("conditional_oracle")
    enabled = enabled or conditional is not None
    if not enabled:
        return None
    return action_execution_contract_from_config(
        config,
        require_canonical=require_canonical,
    )


@dataclass(frozen=True, slots=True)
class ActionExecutionTrajectory:
    """Full replay arrays and the explicit scored-window mask."""

    contract_hash: str
    returns: np.ndarray = field(repr=False)
    # The causal request at each decision bar.  ``decision_deltas`` below is
    # the effective, clipped/executed delta (zero on a fill gap); retaining
    # the intent separately is necessary to audit that a future fill/outcome
    # gap did not alter model selection.
    intent_deltas: np.ndarray = field(repr=False)
    decision_deltas: np.ndarray = field(repr=False)
    decision_positions: np.ndarray = field(repr=False)
    fill_positions: np.ndarray = field(repr=False)
    effective_positions: np.ndarray = field(repr=False)
    transition_costs: np.ndarray = field(repr=False)
    gross_pnl: np.ndarray = field(repr=False)
    net_pnl: np.ndarray = field(repr=False)
    decision_mask: np.ndarray = field(repr=False)
    fill_mask: np.ndarray = field(repr=False)
    scored_mask: np.ndarray = field(repr=False)
    commitment_countdown: np.ndarray = field(repr=False)
    scheduled_decision_mask: np.ndarray = field(repr=False)
    decision_eligible: np.ndarray = field(repr=False)
    score_eligible: np.ndarray = field(repr=False)
    forecast_finite_mask: np.ndarray = field(repr=False)
    eligible_decision_mask: np.ndarray = field(repr=False)
    fill_block_eligible_mask: np.ndarray = field(repr=False)
    block_eligible_mask: np.ndarray = field(repr=False)
    score_block_eligible_mask: np.ndarray = field(repr=False)
    execution_skipped_mask: np.ndarray = field(repr=False)
    contract: ActionExecutionContract = field(repr=False, compare=False)
    block_masks: "ActionBlockMasks" = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze every replay array before exposing its hash/provenance.

        A trajectory's mask registry and PnL fields are used as an identity
        binding by the action artifact.  Leaving the arrays writable would
        let a caller mutate the values after the registry was inspected and
        silently desynchronise the replay from its provenance.
        """
        array_fields = (
            "returns",
            "intent_deltas",
            "decision_deltas",
            "decision_positions",
            "fill_positions",
            "effective_positions",
            "transition_costs",
            "gross_pnl",
            "net_pnl",
            "decision_mask",
            "fill_mask",
            "scored_mask",
            "commitment_countdown",
            "scheduled_decision_mask",
            "decision_eligible",
            "score_eligible",
            "forecast_finite_mask",
            "eligible_decision_mask",
            "fill_block_eligible_mask",
            "block_eligible_mask",
            "score_block_eligible_mask",
            "execution_skipped_mask",
        )
        lengths: set[int] = set()
        for name in array_fields:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != 1:
                raise ValueError(f"trajectory {name} must be a one-dimensional array")
            copied = np.array(value, copy=True, order="C")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
            lengths.add(len(copied))
        if len(lengths) != 1:
            raise ValueError("trajectory arrays must share one full-bar length")
        if not isinstance(self.contract, ActionExecutionContract):
            raise TypeError("trajectory contract must be an ActionExecutionContract")
        if not isinstance(self.block_masks, ActionBlockMasks):
            raise TypeError("trajectory block_masks must be ActionBlockMasks")

    @property
    def scored_indices(self) -> np.ndarray:
        return np.flatnonzero(self.scored_mask)

    @property
    def decision_intent_deltas(self) -> np.ndarray:
        """Alias for the causal request before delayed-fill gating."""
        return self.intent_deltas

    @property
    def scored_returns(self) -> np.ndarray:
        return self.returns[self.scored_mask]

    @property
    def scored_positions(self) -> np.ndarray:
        return self.effective_positions[self.scored_mask]

    @property
    def scored_costs(self) -> np.ndarray:
        return self.transition_costs[self.scored_mask]

    @property
    def scored_pnl(self) -> np.ndarray:
        return self.net_pnl[self.scored_mask]

    @property
    def n_filled_blocks(self) -> int:
        """Number of scheduled blocks with an actual execution fill.

        A decision-feature gap can still be a complete/scorable four-bar
        block while deliberately producing no fill.  Keep this count
        separate from the denominator used by the PnL metrics.
        """
        return int(np.count_nonzero(self.fill_mask))

    @property
    def n_complete_blocks(self) -> int:
        """Number of complete/scorable scheduled blocks.

        This compatibility name now follows the outcome window, not the
        number of fills.  Use :attr:`n_filled_blocks` for execution fills.
        """
        return self.n_scorable_blocks

    @property
    def n_scheduled_decisions(self) -> int:
        return int(np.count_nonzero(self.scheduled_decision_mask))

    @property
    def n_eligible_decisions(self) -> int:
        return int(np.count_nonzero(self.eligible_decision_mask))

    @property
    def n_eligible_blocks(self) -> int:
        return int(np.count_nonzero(self.block_eligible_mask))

    @property
    def n_fill_complete_blocks(self) -> int:
        return int(np.count_nonzero(self.fill_block_eligible_mask))

    @property
    def n_scorable_blocks(self) -> int:
        return int(np.count_nonzero(self.score_block_eligible_mask))

    @property
    def n_execution_skipped_blocks(self) -> int:
        return int(np.count_nonzero(self.execution_skipped_mask))

    @property
    def n_excluded_blocks(self) -> int:
        return self.n_scheduled_decisions - self.n_scorable_blocks

    @property
    def n_scored_bars(self) -> int:
        return int(np.count_nonzero(self.scored_mask))

    @property
    def eligibility_counts(self) -> dict[str, int]:
        """Stable mask counts for audit artifacts and parity checks."""
        return {
            "scheduled_decisions": self.n_scheduled_decisions,
            "eligible_decisions": self.n_eligible_decisions,
            "eligible_blocks": self.n_eligible_blocks,
            "fill_complete_blocks": self.n_fill_complete_blocks,
            "scorable_blocks": self.n_scorable_blocks,
            "filled_blocks": self.n_filled_blocks,
            "execution_skipped_blocks": self.n_execution_skipped_blocks,
            "excluded_blocks": self.n_excluded_blocks,
            "scored_bars": self.n_scored_bars,
        }

    @property
    def scheduled_mask(self) -> np.ndarray:
        """Alias used by audit/manifest consumers."""
        return self.scheduled_decision_mask

    @property
    def bar_available(self) -> np.ndarray:
        """Full-bar availability input used to derive fill/outcome masks."""
        return self.block_masks.bar_available

    @property
    def eligible_block_mask(self) -> np.ndarray:
        """Alias for the action-executable block mask."""
        return self.block_eligible_mask

    @property
    def scorable_block_mask(self) -> np.ndarray:
        """Alias for the outcome/scoring block mask."""
        return self.score_block_eligible_mask

    @property
    def eligibility_mask_hash(self) -> str:
        """Hash the exact full-length eligibility inputs used for replay."""
        payload = json.dumps(
            {
                "decision_eligible": self.decision_eligible.tolist(),
                "forecast_finite_mask": self.forecast_finite_mask.tolist(),
                "score_eligible": self.score_eligible.tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def action_block_mask_hash(self) -> str:
        """Hash the complete causal/fill/outcome/metric mask graph.

        ``eligibility_mask_hash`` is retained for legacy causal consumers and
        intentionally omits fill/outcome/common state.  Production action
        provenance must use this full graph hash instead.
        """
        return self.block_masks.mask_hash

    @property
    def action_block_mask_hash_registry(self) -> Mapping[str, str]:
        """Return per-mask digests for primitive↔trajectory parity checks."""
        return self.block_masks.mask_hash_registry

    @property
    def mask_hash(self) -> str:
        """Short semantic alias for artifact consumers."""
        return self.eligibility_mask_hash


def _validate_series(values: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _coerce_numeric_series(
    values: np.ndarray | Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    """Coerce a numeric path while preserving masked non-finite cells."""
    try:
        return np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc


def _strict_bool_mask(
    values: np.ndarray | Sequence[bool] | None,
    *,
    name: str,
    n_bars: int,
) -> np.ndarray:
    """Convert a mask without accepting truthy integers/strings."""
    if values is None:
        raise ValueError(f"{name} is required for the action execution contract")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional boolean mask") from exc
    if raw.ndim != 1 or len(raw) != n_bars:
        raise ValueError(f"{name} must be a one-dimensional mask of length {n_bars}")
    if not all(isinstance(value, (bool, np.bool_)) for value in raw.tolist()):
        raise ValueError(f"{name} must contain only boolean values")
    return raw.astype(bool, copy=True)


def _resolve_bar_available_alias(
    score_eligible: np.ndarray | Sequence[bool] | None,
    bar_available: np.ndarray | Sequence[bool] | None,
) -> np.ndarray | Sequence[bool] | None:
    """Resolve the legacy availability spelling without allowing drift.

    ``score_eligible`` historically carried a full-bar availability mask.  The
    causal contract names that input ``bar_available`` because outcome/scoring
    is derived from it rather than supplied by the caller.  Accept the old
    spelling only as a compatibility alias and reject any disagreement before
    coercion into the strict boolean-mask validator.
    """
    if score_eligible is not None and bar_available is not None:
        try:
            score_arr = np.asarray(score_eligible)
            available_arr = np.asarray(bar_available)
        except (TypeError, ValueError) as exc:
            raise ValueError("score_eligible and bar_available aliases are malformed") from exc
        if score_arr.shape != available_arr.shape or not np.array_equal(
            score_arr, available_arr
        ):
            raise ValueError("score_eligible and bar_available aliases disagree")
    return bar_available if bar_available is not None else score_eligible


def validate_eligibility_masks(
    decision_eligible: np.ndarray | Sequence[bool] | None,
    score_eligible: np.ndarray | Sequence[bool] | None,
    n_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the two required full-length eligibility masks."""
    if isinstance(n_bars, (bool, np.bool_)) or not isinstance(n_bars, (int, np.integer)):
        raise ValueError("n_bars must be an integer")
    n_bars = int(n_bars)
    if n_bars < 0:
        raise ValueError("n_bars must be non-negative")
    return (
        _strict_bool_mask(decision_eligible, name="decision_eligible", n_bars=n_bars),
        _strict_bool_mask(score_eligible, name="score_eligible", n_bars=n_bars),
    )


@dataclass(frozen=True, slots=True)
class ActionBlockMasks:
    """Deterministic decision/fill/outcome masks on the fixed block grid.

    Only the four causal/source inputs are accepted.  Fill, outcome,
    execution and metric masks are derived values so callers cannot provide
    mutually inconsistent versions of the execution contract.
    """

    origin_mask: np.ndarray = field(repr=False)
    forecast_finite_mask: np.ndarray = field(repr=False)
    bar_available: np.ndarray = field(repr=False)
    returns_finite_mask: np.ndarray = field(repr=False)
    scheduled_decision_mask: np.ndarray = field(repr=False)
    decision_block_mask: np.ndarray = field(repr=False)
    fill_complete_mask: np.ndarray = field(repr=False)
    outcome_complete_mask: np.ndarray = field(repr=False)
    executed_block_mask: np.ndarray = field(repr=False)
    scored_action_mask: np.ndarray = field(repr=False)
    common_mask: np.ndarray = field(repr=False)
    utility_metric_mask: np.ndarray = field(repr=False)
    action_metric_mask: np.ndarray = field(repr=False)
    starts: tuple[int, ...]

    @property
    def mask_hash(self) -> str:
        payload = {
            name: getattr(self, name).tolist()
            for name in (
                "origin_mask",
                "forecast_finite_mask",
                "bar_available",
                "returns_finite_mask",
                "scheduled_decision_mask",
                "decision_block_mask",
                "fill_complete_mask",
                "outcome_complete_mask",
                "executed_block_mask",
                "scored_action_mask",
                "common_mask",
                "utility_metric_mask",
                "action_metric_mask",
            )
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def mask_hash_registry(self) -> Mapping[str, str]:
        """Digest every persisted mask independently in canonical C order."""
        return {
            name: hashlib.sha256(
                np.ascontiguousarray(getattr(self, name), dtype=np.bool_).tobytes(
                    order="C"
                )
            ).hexdigest()
            for name in (
                "origin_mask",
                "forecast_finite_mask",
                "bar_available",
                "returns_finite_mask",
                "scheduled_decision_mask",
                "decision_block_mask",
                "fill_complete_mask",
                "outcome_complete_mask",
                "executed_block_mask",
                "scored_action_mask",
                "common_mask",
                "utility_metric_mask",
                "action_metric_mask",
            )
        }


def derive_action_block_masks(
    n_bars: int,
    contract: ActionExecutionContract,
    *,
    origin_mask: np.ndarray | Sequence[bool] | None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None,
    bar_available: np.ndarray | Sequence[bool] | None,
    realized_returns: np.ndarray | Sequence[float] | None = None,
    common_mask: np.ndarray | Sequence[bool] | None = None,
) -> ActionBlockMasks:
    """Derive the only supported causal fill/outcome mask graph.

    ``decision = origin AND finite forecast``
    ``fill = bar_available[t+1]``
    ``outcome = all(bar_available[t+1:t+5]) AND finite returns``
    ``executed = decision AND fill``
    ``scored_action = executed AND outcome``

    ``common_mask`` is block-level and affects metric reduction only.  It can
    never suppress a causal decision, fill, or chronological state update.
    """
    if not isinstance(contract, ActionExecutionContract):
        raise TypeError("contract must be an ActionExecutionContract")
    if isinstance(n_bars, (bool, np.bool_)) or not isinstance(n_bars, (int, np.integer)):
        raise ValueError("n_bars must be an integer")
    n_bars = int(n_bars)
    if n_bars < 0:
        raise ValueError("n_bars must be non-negative")
    origin = _strict_bool_mask(origin_mask, name="origin_mask", n_bars=n_bars)
    forecast = _strict_bool_mask(
        forecast_finite_mask,
        name="forecast_finite_mask",
        n_bars=n_bars,
    )
    available = _strict_bool_mask(
        bar_available,
        name="bar_available",
        n_bars=n_bars,
    )
    if realized_returns is None:
        returns_finite = np.ones(n_bars, dtype=bool)
    else:
        values = _coerce_numeric_series(realized_returns, name="realized_returns")
        if len(values) != n_bars:
            raise ValueError(f"realized_returns must have length {n_bars}")
        returns_finite = np.isfinite(values)

    starts = complete_decision_starts(n_bars, contract)
    scheduled = np.zeros(n_bars, dtype=bool)
    decision = np.zeros(n_bars, dtype=bool)
    fill_complete = np.zeros(n_bars, dtype=bool)
    outcome_complete = np.zeros(n_bars, dtype=bool)
    executed = np.zeros(n_bars, dtype=bool)
    scored_action = np.zeros(n_bars, dtype=bool)
    for start in starts:
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        scheduled[start] = True
        decision[start] = bool(origin[start] and forecast[start])
        fill_complete[start] = bool(available[fill])
        outcome_complete[start] = bool(
            available[fill:end].all() and returns_finite[fill:end].all()
        )
        executed[start] = bool(decision[start] and fill_complete[start])
        scored_action[start] = bool(executed[start] and outcome_complete[start])

    if common_mask is None:
        common = np.ones(len(starts), dtype=bool)
    else:
        raw_common = np.asarray(common_mask)
        if raw_common.ndim != 1 or len(raw_common) != len(starts):
            raise ValueError(
                f"common_mask must be a one-dimensional block mask of length {len(starts)}"
            )
        if not all(isinstance(value, (bool, np.bool_)) for value in raw_common.tolist()):
            raise ValueError("common_mask must contain only boolean values")
        common = raw_common.astype(bool, copy=True)
    common_full = np.zeros(n_bars, dtype=bool)
    for index, start in enumerate(starts):
        common_full[start] = common[index]
    utility_metric = outcome_complete & common_full
    action_metric = scored_action & common_full

    arrays = (
        origin,
        forecast,
        available,
        returns_finite,
        scheduled,
        decision,
        fill_complete,
        outcome_complete,
        executed,
        scored_action,
        common_full,
        utility_metric,
        action_metric,
    )
    for values in arrays:
        values.setflags(write=False)
    return ActionBlockMasks(
        origin_mask=origin,
        forecast_finite_mask=forecast,
        bar_available=available,
        returns_finite_mask=returns_finite,
        scheduled_decision_mask=scheduled,
        decision_block_mask=decision,
        fill_complete_mask=fill_complete,
        outcome_complete_mask=outcome_complete,
        executed_block_mask=executed,
        scored_action_mask=scored_action,
        common_mask=common_full,
        utility_metric_mask=utility_metric,
        action_metric_mask=action_metric,
        starts=starts,
    )


def _contract_block_masks(
    n_bars: int,
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None,
    score_eligible: np.ndarray | Sequence[bool] | None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
    realized_returns: np.ndarray | Sequence[float] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
]:
    decision_mask, score_mask = validate_eligibility_masks(
        decision_eligible,
        score_eligible,
        n_bars,
    )
    forecast_mask = (
        np.ones(n_bars, dtype=bool)
        if forecast_finite_mask is None
        else forecast_finite_mask
    )
    masks = derive_action_block_masks(
        n_bars,
        contract,
        origin_mask=decision_mask,
        forecast_finite_mask=forecast_mask,
        bar_available=score_mask,
        realized_returns=realized_returns,
    )
    return (
        decision_mask,
        score_mask,
        masks.scheduled_decision_mask,
        masks.decision_block_mask,
        masks.fill_complete_mask,
        masks.executed_block_mask,
        masks.outcome_complete_mask,
        masks.starts,
    )


def _validate_decision_block_scores(
    values: np.ndarray | Sequence[float],
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None,
    score_eligible: np.ndarray | Sequence[bool] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
]:
    """Validate only the scalar forecast available at each decision start.

    The full-length representation intentionally permits arbitrary/NaN values
    at blocked and outcome bars.  Those cells are not forecasts for the
    current decision and must never be inspected or used as a fallback.
    """
    arr = _coerce_numeric_series(values, name="decision_block_scores")
    (
        decision_mask,
        score_mask,
        scheduled,
        eligible_decision,
        fill_block_eligible,
        block_eligible,
        score_block_eligible,
        starts,
    ) = _contract_block_masks(
        len(arr),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        forecast_finite_mask=np.isfinite(arr),
    )
    if not starts:
        raise ValueError("decision_block_scores require at least one complete decision block")
    return (
        arr,
        decision_mask,
        score_mask,
        scheduled,
        eligible_decision,
        fill_block_eligible,
        block_eligible,
        score_block_eligible,
        starts,
    )


def _validate_lengths(decision_deltas: np.ndarray, returns: np.ndarray) -> None:
    if len(decision_deltas) != len(returns):
        raise ValueError("decision_deltas and returns must have equal lengths")


def _candidate_position(contract: ActionExecutionContract, current: float, delta: float) -> float:
    if not np.isfinite(delta):
        raise ValueError("decision delta must be finite")
    if not any(np.isclose(delta, allowed, atol=_FLOAT_TOL, rtol=0.0) for allowed in contract.candidate_deltas):
        raise ValueError(
            f"decision delta {delta!r} is not in the contract candidate grid "
            f"{contract.candidate_deltas!r}"
        )
    # The absolute-position wire value is canonicalised identically to the
    # action-primitive producer: clip first, then round to twelve decimals.
    return float(
        np.round(
            np.clip(current + delta, contract.position_min, contract.position_max),
            decimals=12,
        )
    )


def candidate_positions(
    current_position: float,
    contract: ActionExecutionContract | None = None,
) -> np.ndarray:
    """Return clip-then-unique feasible absolute positions for one decision."""
    contract = contract or ActionExecutionContract.canonical()
    current = float(current_position)
    if not np.isfinite(current) or not contract.position_min <= current <= contract.position_max:
        raise ValueError("current_position must be finite and within contract bounds")
    values = np.clip(
        current + np.asarray(contract.candidate_deltas, dtype=np.float64),
        contract.position_min,
        contract.position_max,
    )
    return np.unique(np.round(values, decimals=12))


def transition_cost(
    previous_position: float,
    next_position: float,
    contract: ActionExecutionContract | None = None,
) -> float:
    contract = contract or ActionExecutionContract.canonical()
    previous = float(previous_position)
    nxt = float(next_position)
    if not np.isfinite(previous) or not np.isfinite(nxt):
        raise ValueError("positions must be finite")
    if not contract.position_min <= previous <= contract.position_max:
        raise ValueError("previous_position must be within contract bounds")
    if not contract.position_min <= nxt <= contract.position_max:
        raise ValueError("next_position must be within contract bounds")
    return float(abs(nxt - previous) * contract.transition_cost_rate)


def complete_decision_starts(
    n_bars: int,
    contract: ActionExecutionContract | None = None,
) -> tuple[int, ...]:
    """Return decision bars whose delayed four-bar block is fully observed."""
    contract = contract or ActionExecutionContract.canonical()
    n_bars = int(n_bars)
    if n_bars < 0:
        raise ValueError("n_bars must be non-negative")
    first = int(contract.initial_countdown)
    starts = range(first, n_bars, int(contract.commitment_bars))
    return tuple(
        start
        for start in starts
        if start + contract.execution_delay_bars + contract.h_decision <= n_bars
    )


def replay_action_path(
    returns: np.ndarray | Sequence[float],
    decision_deltas: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
    common_mask: np.ndarray | Sequence[bool] | None = None,
) -> ActionExecutionTrajectory:
    """Replay deltas under the fixed delay, commitment and fill contract.

    ``decision_deltas[t]`` is the causal intent at an eligible complete
    decision bar.  The delayed fill bar gates execution and the full delayed
    outcome window gates scoring; neither future outcome availability nor an
    outcome gap can change the already selected intent.  A fill gap converts
    the intent to a zero effective delta and leaves inventory unchanged.  A
    later outcome gap keeps the executed position in chronological state but
    excludes that block from PnL.  A decision-feature gap skips execution and
    holds inventory.  The schedule is never compressed: the next decision
    remains at the next commitment boundary.  Both input masks are strict,
    full-length boolean arrays and are mandatory for the new path.
    """
    contract = contract or ActionExecutionContract.canonical()
    returns_arr = _coerce_numeric_series(returns, name="returns")
    deltas_arr = _coerce_numeric_series(decision_deltas, name="decision_deltas")
    _validate_lengths(deltas_arr, returns_arr)
    n_bars = len(returns_arr)
    replay_forecast_finite = (
        np.isfinite(deltas_arr)
        if forecast_finite_mask is None
        else _strict_bool_mask(
            forecast_finite_mask,
            name="forecast_finite_mask",
            n_bars=n_bars,
        )
    )
    supplied_bar_available = _resolve_bar_available_alias(
        score_eligible,
        bar_available,
    )
    (
        decision_eligible_arr,
        score_eligible_arr,
        scheduled_decision_mask,
        eligible_decision_mask,
        fill_block_eligible_mask,
        block_eligible_mask,
        score_block_eligible_mask,
        starts,
    ) = _contract_block_masks(
        n_bars,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=supplied_bar_available,
        forecast_finite_mask=replay_forecast_finite,
        realized_returns=returns_arr,
    )
    block_masks = derive_action_block_masks(
        n_bars,
        contract,
        origin_mask=decision_eligible_arr,
        forecast_finite_mask=replay_forecast_finite,
        bar_available=score_eligible_arr,
        realized_returns=returns_arr,
        common_mask=common_mask,
    )
    if not np.array_equal(block_masks.scheduled_decision_mask, scheduled_decision_mask):
        raise ValueError("action block schedule is inconsistent with the replay contract")
    if not np.array_equal(block_masks.decision_block_mask, eligible_decision_mask):
        raise ValueError("action block decision mask is inconsistent with replay inputs")
    if not np.array_equal(block_masks.fill_complete_mask, fill_block_eligible_mask):
        raise ValueError("action block fill mask is inconsistent with replay inputs")
    if not np.array_equal(block_masks.executed_block_mask, block_eligible_mask):
        raise ValueError("action block execution mask is inconsistent with replay inputs")
    if not np.array_equal(block_masks.outcome_complete_mask, score_block_eligible_mask):
        raise ValueError("action block outcome mask is inconsistent with replay inputs")
    decision_deltas_out = np.zeros(n_bars, dtype=np.float64)
    intent_deltas_out = np.zeros(n_bars, dtype=np.float64)
    decision_positions = np.full(n_bars, np.nan, dtype=np.float64)
    fill_positions = np.full(n_bars, np.nan, dtype=np.float64)
    effective_positions = np.full(n_bars, float(contract.p_start), dtype=np.float64)
    transition_costs = np.zeros(n_bars, dtype=np.float64)
    decision_mask = np.zeros(n_bars, dtype=bool)
    fill_mask = np.zeros(n_bars, dtype=bool)
    scored_mask = np.zeros(n_bars, dtype=bool)
    commitment_countdown = np.zeros(n_bars, dtype=np.int64)
    execution_skipped_mask = np.zeros(n_bars, dtype=bool)

    current = float(contract.p_start)
    for start in starts:
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        if not eligible_decision_mask[start]:
            # A decision-feature gap is an execution skip, not a data gap:
            # hold the current inventory for this fixed commitment block and
            # keep its finite returns scored only when the outcome is complete.
            raw_delta = deltas_arr[start]
            if np.isfinite(raw_delta) and not np.isclose(
                raw_delta,
                0.0,
                atol=_FLOAT_TOL,
                rtol=0.0,
            ):
                raise ValueError(
                    f"decision delta at ineligible block {start} must be zero"
                )
            execution_skipped_mask[start] = True
            if score_block_eligible_mask[start]:
                scored_mask[fill:end] = True
            continue
        if not fill_block_eligible_mask[start]:
            # The decision intent exists, but an all-or-none fill gap leaves
            # the executable/recorded delta at zero and cannot mutate state.
            # Validate the causal intent without treating the future fill as
            # an input to action selection.
            raw_delta = deltas_arr[start]
            _candidate_position(contract, current, float(raw_delta))
            intent_deltas_out[start] = float(raw_delta)
            decision_mask[start] = True
            decision_positions[start] = current
            continue
        # A decision is made before the delayed fill.  The state remains the
        # previous position until the fill bar, then stays fixed for H bars.
        raw_delta = float(deltas_arr[start])
        next_position = _candidate_position(contract, current, raw_delta)
        actual_delta = next_position - current
        intent_deltas_out[start] = raw_delta
        decision_mask[start] = True
        decision_deltas_out[start] = actual_delta
        decision_positions[start] = next_position
        # A hold decision still starts a scored commitment, but it is not a
        # fill/order and therefore must not inflate the filled-block count.
        if not np.isclose(actual_delta, 0.0, atol=_FLOAT_TOL, rtol=0.0):
            fill_mask[fill] = True
            fill_positions[fill] = next_position
            transition_costs[fill] = transition_cost(current, next_position, contract)
        effective_positions[fill:end] = next_position
        if score_block_eligible_mask[start]:
            scored_mask[fill:end] = True
        current = next_position

    # Fill the state traces after constructing the complete blocks.  The
    # countdown is the post-decision commitment state: H at a decision bar,
    # then decremented on each subsequent bar.  It is never used to score tail
    # bars and cannot create a partial fill.
    last_decision = float(contract.p_start)
    last_fill = float(contract.p_start)
    for bar in range(n_bars):
        if decision_mask[bar]:
            last_decision = decision_positions[bar]
        if scheduled_decision_mask[bar]:
            commitment_countdown[bar] = int(contract.commitment_bars)
        elif bar > 0:
            commitment_countdown[bar] = max(
                0,
                commitment_countdown[bar - 1] - contract.countdown_decrement,
            )
        decision_positions[bar] = last_decision
        if fill_mask[bar]:
            last_fill = fill_positions[bar]
        fill_positions[bar] = last_fill

    # A filled position remains the live inventory after its four scored bars,
    # including any unscored incomplete tail.  The tail is excluded from PnL
    # by ``scored_mask`` but must not pretend that inventory returned to
    # p_start.
    last_effective = float(contract.p_start)
    for bar in range(n_bars):
        if fill_mask[bar]:
            last_effective = fill_positions[bar]
        effective_positions[bar] = last_effective

    if np.any(scored_mask & ~np.isfinite(returns_arr)):
        raise ValueError("returns must be finite on scored eligible bars")
    scored_returns = np.where(scored_mask, returns_arr, 0.0)
    gross_pnl = effective_positions * scored_returns
    net_pnl = np.where(scored_mask, gross_pnl - transition_costs, 0.0)
    return ActionExecutionTrajectory(
        contract_hash=contract.contract_hash,
        returns=returns_arr,
        intent_deltas=intent_deltas_out,
        decision_deltas=decision_deltas_out,
        decision_positions=decision_positions,
        fill_positions=fill_positions,
        effective_positions=effective_positions,
        transition_costs=transition_costs,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        decision_mask=decision_mask,
        fill_mask=fill_mask,
        scored_mask=scored_mask,
        commitment_countdown=commitment_countdown,
        scheduled_decision_mask=scheduled_decision_mask,
        decision_eligible=decision_eligible_arr,
        score_eligible=score_eligible_arr,
        forecast_finite_mask=replay_forecast_finite,
        eligible_decision_mask=eligible_decision_mask,
        fill_block_eligible_mask=fill_block_eligible_mask,
        block_eligible_mask=block_eligible_mask,
        score_block_eligible_mask=score_block_eligible_mask,
        execution_skipped_mask=execution_skipped_mask,
        contract=contract,
        block_masks=block_masks,
    )


def decision_deltas_from_positions(
    positions: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    strict_blocked: bool = True,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Convert an absolute policy path to contract deltas without fallback.

    Only causal decision-eligible bars are allowed to express a non-zero
    intent.  ``bar_available`` gates whether that intent actually fills; a
    missing delayed fill must not erase the intent or make an otherwise valid
    absolute path fail validation.  In strict mode blocked feature bars and
    committed-bar targets must still agree with the current/intent position,
    preventing a legacy every-bar actor path from being silently clipped into
    the new contract.
    """
    contract = contract or ActionExecutionContract.canonical()
    positions_arr = _coerce_numeric_series(positions, name="positions")
    deltas = np.zeros(len(positions_arr), dtype=np.float64)
    current = float(contract.p_start)
    (
        _,
        _,
        _,
        eligible_decision_mask,
        fill_block_eligible_mask,
        _,
        _,
        starts_tuple,
    ) = _contract_block_masks(
        len(positions_arr),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
        forecast_finite_mask=(
            np.isfinite(positions_arr)
            if forecast_finite_mask is None
            else _strict_bool_mask(
                forecast_finite_mask,
                name="forecast_finite_mask",
                n_bars=len(positions_arr),
            )
        ),
    )
    starts = set(starts_tuple)
    # ``block_target`` is the selected intent for the current commitment.  A
    # fill gap leaves inventory at ``current`` but permits either a producer
    # that records the effective path (current) or one that records the
    # selected intent (block_target) across the commitment bars.
    block_target = float(contract.p_start)
    for bar, target in enumerate(positions_arr):
        if bar in starts:
            if not eligible_decision_mask[bar]:
                if strict_blocked and not np.isclose(
                    target,
                    current,
                    atol=_FLOAT_TOL,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"position path changes during ineligible block at bar {bar}"
                    )
                block_target = current
                continue
            delta = float(target - current)
            if not any(np.isclose(delta, allowed, atol=_FLOAT_TOL, rtol=0.0) for allowed in contract.candidate_deltas):
                raise ValueError(
                    f"position path at decision bar {bar} has unsupported delta {delta!r}"
                )
            candidate = _candidate_position(contract, current, delta)
            if not np.isclose(candidate, target, atol=_FLOAT_TOL, rtol=0.0):
                raise ValueError(
                    f"position path at decision bar {bar} is outside contract bounds"
                )
            deltas[bar] = delta
            block_target = candidate
            if fill_block_eligible_mask[bar]:
                current = candidate
            continue
        if strict_blocked:
            allowed_targets = (current, block_target)
            if not any(
                np.isclose(target, allowed, atol=_FLOAT_TOL, rtol=0.0)
                for allowed in allowed_targets
            ):
                raise ValueError(
                    f"position path changes during a committed block at bar {bar}"
                )
    return deltas


def select_block_decisions(
    decision_block_scores: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Select causal block actions from one scalar score per decision start.

    ``decision_block_scores`` is a full-length vector, but only
    ``decision_block_scores[t]`` is read for a complete decision start ``t``.
    Each scalar is the cumulative four-bar forecast already available at that
    decision time; blocked/outcome-bar cells are deliberately ignored.  The
    returned value is causal intent, so a future fill/outcome gap cannot alter
    it.  Chronological inventory advances only after a complete delayed fill.
    U0 uses :func:`select_hindsight_block_decisions` instead because it
    consumes realized per-bar returns.
    """
    contract = contract or ActionExecutionContract.canonical()
    (
        scores,
        _,
        _,
        _,
        eligible_decision,
        fill_block_eligible,
        block_eligible,
        _,
        starts,
    ) = _validate_decision_block_scores(
        decision_block_scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
    )

    deltas = np.zeros(len(scores), dtype=np.float64)
    current = float(contract.p_start)
    for start in starts:
        if not eligible_decision[start]:
            continue
        block_score = float(scores[start])
        candidates: list[tuple[float, float, float]] = []
        for delta in contract.candidate_deltas:
            nxt = _candidate_position(contract, current, delta)
            value = nxt * block_score - transition_cost(current, nxt, contract)
            candidates.append((value, delta, nxt))
        _, best_delta, best_next = max(
            candidates,
            key=lambda item: (item[0], -abs(item[1]), -item[1]),
        )
        deltas[start] = float(best_delta)
        if fill_block_eligible[start]:
            current = best_next
    return deltas


def select_hindsight_block_decisions(
    realized_returns: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Select the realized-future U0 path with an iterative block DP.

    Unlike the conditional teacher, U0 is permitted to inspect all complete
    future blocks.  The DP is bottom-up over the finite reachable position
    states, avoiding Python recursion depth failures on long research folds.
    It is an upper-bound diagnostic only.
    """
    contract = contract or ActionExecutionContract.canonical()
    scores = _coerce_numeric_series(realized_returns, name="realized_returns")
    (
        _,
        _,
        _,
        _,
        _,
        block_eligible,
        score_block_eligible,
        starts,
    ) = _contract_block_masks(
        len(scores),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
        realized_returns=scores,
    )
    if not starts:
        raise ValueError("action path requires at least one complete decision block")

    for start in starts:
        if not score_block_eligible[start]:
            continue
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        if not np.all(np.isfinite(scores[fill:end])):
            raise ValueError(
                f"realized_returns must be finite on eligible score block {start}"
            )

    optimizable_block = block_eligible & score_block_eligible

    # Reachable states at each block boundary are finite because every action
    # is clipped to the bounded spot allocation interval.
    states: list[tuple[float, ...]] = [(float(contract.p_start),)]
    for index, _ in enumerate(starts):
        previous_states = states[-1]
        if not optimizable_block[starts[index]]:
            states.append(previous_states)
            continue
        next_values = {
            round(_candidate_position(contract, current, delta), 12)
            for current in previous_states
            for delta in contract.candidate_deltas
        }
        states.append(tuple(sorted(next_values)))

    value_next = {round(state, 12): 0.0 for state in states[-1]}
    policy: list[dict[float, tuple[float, float]]] = [{} for _ in starts]
    for index in range(len(starts) - 1, -1, -1):
        start = starts[index]
        if not optimizable_block[start]:
            if score_block_eligible[start]:
                fill = start + contract.execution_delay_bars
                end = fill + contract.h_decision
                block_sum = float(scores[fill:end].sum())
                value_next = {
                    round(current, 12): current * block_sum
                    + value_next[round(current, 12)]
                    for current in states[index]
                }
            else:
                value_next = {
                    round(current, 12): value_next[round(current, 12)]
                    for current in states[index]
                }
            policy[index] = {
                round(current, 12): (0.0, current)
                for current in states[index]
            }
            continue
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        block_sum = float(scores[fill:end].sum())
        value_now: dict[float, float] = {}
        for current in states[index]:
            candidates: list[tuple[float, float, float]] = []
            for delta in contract.candidate_deltas:
                nxt = _candidate_position(contract, current, delta)
                next_key = round(nxt, 12)
                value = (
                    nxt * block_sum
                    - transition_cost(current, nxt, contract)
                    + value_next[next_key]
                )
                candidates.append((value, delta, nxt))
            best_value, best_delta, best_next = max(
                candidates,
                key=lambda item: (item[0], -abs(item[1]), -item[1]),
            )
            current_key = round(current, 12)
            value_now[current_key] = best_value
            policy[index][current_key] = (best_delta, best_next)
        value_next = value_now

    deltas = np.zeros(len(scores), dtype=np.float64)
    current = float(contract.p_start)
    for index, start in enumerate(starts):
        best_delta, best_next = policy[index][round(current, 12)]
        deltas[start] = float(best_delta)
        current = best_next
    return deltas


def replay_selected_path(
    decision_block_scores: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
) -> ActionExecutionTrajectory:
    """Select/replay a causal teacher from cumulative block forecasts.

    The returned replay uses a deterministic per-bar expansion of each
    decision-start scalar (score divided evenly across the four scored bars)
    solely so the trajectory can expose utility/cost arrays.  No blocked or
    outcome-bar input cells are read.
    """
    contract = contract or ActionExecutionContract.canonical()
    (
        scores,
        _,
        _,
        _,
        eligible_decision,
        _,
        _,
        score_block_eligible,
        starts,
    ) = _validate_decision_block_scores(
        decision_block_scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
    )
    deltas = select_block_decisions(
        scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
    )
    replay_returns = np.zeros(len(scores), dtype=np.float64)
    for start in starts:
        if not (eligible_decision[start] and score_block_eligible[start]):
            continue
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        replay_returns[fill:end] = float(scores[start]) / contract.h_decision
    return replay_action_path(
        replay_returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
        forecast_finite_mask=np.isfinite(scores),
    )


def replay_hindsight_selected_path(
    realized_returns: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
) -> ActionExecutionTrajectory:
    """Select and replay the realized-future U0 trajectory."""
    contract = contract or ActionExecutionContract.canonical()
    deltas = select_hindsight_block_decisions(
        realized_returns,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=_resolve_bar_available_alias(score_eligible, bar_available),
    )
    return replay_action_path(
        realized_returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        score_eligible=score_eligible,
    )


def run_contract_backtest(
    backtest_cls,
    returns: np.ndarray | Sequence[float],
    absolute_positions: np.ndarray | Sequence[float],
    *,
    benchmark_positions: np.ndarray | Sequence[float] | None,
    contract: ActionExecutionContract,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
    common_mask: np.ndarray | Sequence[bool] | None = None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
    expected_contract_hash: str | None = None,
    require_external_contract_hash: bool = False,
    **kwargs: Any,
):
    """Stage adapter: validate absolute policy paths, then invoke new Backtest.

    ``backtest_cls`` is injected by the existing pipeline.  The explicit
    contract keyword is mandatory here; a class that does not support it fails
    loudly instead of falling back to historical delay/cost defaults.
    """
    if score_eligible is not None and bar_available is not None:
        if not np.array_equal(np.asarray(score_eligible), np.asarray(bar_available)):
            raise ValueError("score_eligible and bar_available aliases disagree")
    supplied_bar_available = (
        bar_available if bar_available is not None else score_eligible
    )
    if require_external_contract_hash and bar_available is None:
        raise ValueError(
            "production contract backtest requires explicit bar_available; "
            "score_eligible is an ambiguous legacy alias"
        )
    if require_external_contract_hash and common_mask is None:
        raise ValueError(
            "production contract backtest requires an explicit paired common_mask"
        )
    decision_eligible_arr, score_eligible_arr = validate_eligibility_masks(
        decision_eligible,
        supplied_bar_available,
        len(np.asarray(returns).reshape(-1)),
    )
    # Import lazily to avoid a module cycle: ``backtest`` imports the replay
    # primitives from this module, while this stage adapter is called by it.
    from .backtest import validate_bound_action_execution_contract

    validate_bound_action_execution_contract(
        contract,
        expected_contract_hash=expected_contract_hash,
        require_external_hash=require_external_contract_hash,
    )
    decision_deltas = decision_deltas_from_positions(
        absolute_positions,
        contract,
        decision_eligible=decision_eligible_arr,
        score_eligible=score_eligible_arr,
        forecast_finite_mask=forecast_finite_mask,
    )
    benchmark_deltas = (
        np.zeros(len(decision_deltas), dtype=np.float64)
        if benchmark_positions is None
        else decision_deltas_from_positions(
            benchmark_positions,
            contract,
            decision_eligible=decision_eligible_arr,
            score_eligible=score_eligible_arr,
        )
    )
    kwargs = dict(kwargs)
    # Do not accept historical cost/delay knobs on the explicit path.  Merely
    # dropping them would hide a mixed-contract caller (and could make a
    # report claim the wrong 3/1/.0003 geometry), so fail closed even when the
    # supplied value happens to equal the canonical one.
    legacy_keys = (
        "spread_bps",
        "fee_rate",
        "slippage_bps",
        "execution_delay_bars",
        "initial_position",
        "benchmark_initial_position",
    )
    supplied_legacy = sorted(key for key in legacy_keys if key in kwargs)
    if supplied_legacy:
        raise ValueError(
            "contract path rejects legacy overrides: "
            + ", ".join(supplied_legacy)
        )
    kwargs["benchmark_positions"] = benchmark_deltas
    kwargs["action_execution_contract"] = contract
    kwargs["action_positions_are_deltas"] = True
    kwargs["decision_eligible"] = decision_eligible_arr
    kwargs["bar_available"] = score_eligible_arr
    kwargs["common_mask"] = common_mask
    kwargs["forecast_finite_mask"] = forecast_finite_mask
    kwargs["expected_contract_hash"] = expected_contract_hash
    kwargs["require_external_contract_hash"] = require_external_contract_hash
    return backtest_cls(returns, decision_deltas, **kwargs)


def replay_contract_absolute_path(
    returns: np.ndarray | Sequence[float],
    absolute_positions: np.ndarray | Sequence[float],
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
    common_mask: np.ndarray | Sequence[bool] | None = None,
) -> ActionExecutionTrajectory:
    """Strictly convert and replay an absolute policy path."""
    deltas = decision_deltas_from_positions(
        absolute_positions,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite_mask,
    )
    return replay_action_path(
        returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite_mask,
        common_mask=common_mask,
    )


def contract_pnl_attribution(
    returns: np.ndarray | Sequence[float],
    absolute_positions: np.ndarray | Sequence[float],
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
    bar_available: np.ndarray | Sequence[bool] | None = None,
    forecast_finite_mask: np.ndarray | Sequence[bool] | None = None,
    common_mask: np.ndarray | Sequence[bool] | None = None,
) -> dict[str, float]:
    """Return long/short/cost attribution from the shared contract replay."""
    trajectory = replay_contract_absolute_path(
        returns,
        absolute_positions,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite_mask,
        common_mask=common_mask,
    )
    scored = trajectory.scored_mask
    gross = trajectory.gross_pnl[scored]
    positions = trajectory.effective_positions[scored]
    costs = trajectory.transition_costs[scored]
    return {
        "long_gross": float(gross[positions > 0.0].sum()),
        "short_gross": float(gross[positions < 0.0].sum()),
        "cost_total": float(costs.sum()),
        "net_total": float(trajectory.net_pnl[scored].sum()),
    }
