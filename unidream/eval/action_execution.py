"""Single source of truth for the conditional-Oracle action contract.

The historical research path has several action grids, execution delays and
cost defaults.  This module is deliberately independent from the historical
Oracle and Backtest implementations so the new conditional-Oracle path can
opt in to one immutable contract without changing historical results.

The contract is spot-only and intentionally small:

``decision t -> fill t+1 -> earn returns[t+1:t+5]``

The final block is scored only when all four returns are present.  A trajectory
contains full-length diagnostic arrays, plus an explicit ``scored_mask``; the
Backtest adapter trims to that mask before computing metrics.  The same
replay geometry is used by conditional teachers and by the upper-bound
diagnostic, while their selectors remain separate: the teacher is causal and
U0 is hindsight-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_FLOAT_TOL = 1e-9
_CANONICAL_CANDIDATE_DELTAS = (-0.08, -0.04, 0.0, 0.04, 0.08)


def _as_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence") from exc
    if not result or not all(np.isfinite(value) for value in result):
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
    for key in ("action_execution_contract", "action_execution"):
        value = config.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping")
            return value
    conditional = config.get("conditional_oracle")
    if isinstance(conditional, Mapping):
        for key in ("action_execution_contract", "action_execution"):
            value = conditional.get(key)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(f"conditional_oracle.{key} must be a mapping")
                return value
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
    feature_unavailable_policy: str = "exclude_block"
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
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.position_min) > float(self.position_max):
            raise ValueError("position_min must be <= position_max")
        if not float(self.position_min) <= float(self.p_start) <= float(self.position_max):
            raise ValueError("p_start must lie within position bounds")
        _require_integer(self.h_decision, name="h_decision", minimum=1)
        _require_integer(self.commitment_bars, name="commitment_bars", minimum=1)
        _require_integer(self.execution_delay_bars, name="execution_delay_bars", minimum=0)
        _require_integer(self.initial_countdown, name="initial_countdown", minimum=0)
        _require_integer(self.countdown_decrement, name="countdown_decrement", minimum=1)
        deltas = _as_float_tuple(self.candidate_deltas, name="candidate_deltas")
        if not any(abs(delta) <= _FLOAT_TOL for delta in deltas):
            raise ValueError("candidate_deltas must include the hold delta 0.0")
        if len({round(delta, 12) for delta in deltas}) != len(deltas):
            raise ValueError("candidate_deltas must not contain duplicate values")
        object.__setattr__(self, "candidate_deltas", deltas)
        for name, value in (
            ("fill_policy", self.fill_policy),
            ("partial_fill_policy", self.partial_fill_policy),
            ("tail_policy", self.tail_policy),
            ("spread_convention", self.spread_convention),
            ("return_unit", self.return_unit),
            ("boundary_cost_policy", self.boundary_cost_policy),
            ("feature_unavailable_policy", self.feature_unavailable_policy),
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
        if self.feature_unavailable_policy != "exclude_block":
            raise ValueError("feature_unavailable_policy must be exclude_block")
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
    ) -> "ActionExecutionTrajectory":
        return replay_action_path(
            returns,
            decision_deltas,
            self,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

    def replay_absolute(
        self,
        returns: np.ndarray | Sequence[float],
        absolute_positions: np.ndarray | Sequence[float],
        *,
        decision_eligible: np.ndarray | Sequence[bool] | None = None,
        score_eligible: np.ndarray | Sequence[bool] | None = None,
    ) -> "ActionExecutionTrajectory":
        return replay_contract_absolute_path(
            returns,
            absolute_positions,
            self,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

    def select_decisions(
        self,
        decision_block_scores: np.ndarray | Sequence[float],
        *,
        decision_eligible: np.ndarray | Sequence[bool] | None = None,
        score_eligible: np.ndarray | Sequence[bool] | None = None,
    ) -> np.ndarray:
        return select_block_decisions(
            decision_block_scores,
            self,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
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
            "commitment_countdown_decrement": "countdown_decrement",
        }
        normalized = dict(section)
        for source, target in aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized[source]
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
            execution_skip_policy=normalized["execution_skip_policy"],
            eligibility_masks_required=normalized["eligibility_masks_required"],
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
    enabled = bool(config.get("use_action_execution_contract", False))
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
    eligible_decision_mask: np.ndarray = field(repr=False)
    block_eligible_mask: np.ndarray = field(repr=False)

    @property
    def scored_indices(self) -> np.ndarray:
        return np.flatnonzero(self.scored_mask)

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
    def n_complete_blocks(self) -> int:
        return int(np.count_nonzero(self.fill_mask))

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
    def n_excluded_blocks(self) -> int:
        return self.n_scheduled_decisions - self.n_eligible_blocks

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
            "excluded_blocks": self.n_excluded_blocks,
            "scored_bars": self.n_scored_bars,
        }

    @property
    def eligibility_mask_hash(self) -> str:
        """Hash the exact full-length eligibility inputs used for replay."""
        payload = json.dumps(
            {
                "decision_eligible": self.decision_eligible.tolist(),
                "score_eligible": self.score_eligible.tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

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


def _contract_block_masks(
    n_bars: int,
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
    tuple[int, ...],
]:
    decision_mask, score_mask = validate_eligibility_masks(
        decision_eligible,
        score_eligible,
        n_bars,
    )
    starts = complete_decision_starts(n_bars, contract)
    scheduled = np.zeros(n_bars, dtype=bool)
    eligible_decision = np.zeros(n_bars, dtype=bool)
    block_eligible = np.zeros(n_bars, dtype=bool)
    for start in starts:
        scheduled[start] = True
        eligible_decision[start] = decision_mask[start]
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        block_eligible[start] = bool(
            decision_mask[start] and score_mask[fill:end].all()
        )
    return decision_mask, score_mask, scheduled, eligible_decision, block_eligible, starts


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
        block_eligible,
        starts,
    ) = _contract_block_masks(
        len(arr),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    if not starts:
        raise ValueError("decision_block_scores require at least one complete decision block")
    for start in starts:
        if block_eligible[start] and not np.isfinite(arr[start]):
            raise ValueError(
                f"decision_block_scores[{start}] must be a finite cumulative forecast"
            )
    return (
        arr,
        decision_mask,
        score_mask,
        scheduled,
        eligible_decision,
        block_eligible,
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
    return float(np.clip(current + delta, contract.position_min, contract.position_max))


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
) -> ActionExecutionTrajectory:
    """Replay deltas under the fixed delay, commitment and fill contract.

    ``decision_deltas[t]`` is read only at eligible complete decision bars.
    A scheduled block is excluded as a whole when its decision or any delayed
    score bar is ineligible.  The schedule is not compressed: the next
    decision remains at the next commitment boundary.  Both masks are strict,
    full-length boolean arrays and are mandatory for the new contract path.
    """
    contract = contract or ActionExecutionContract.canonical()
    returns_arr = _coerce_numeric_series(returns, name="returns")
    deltas_arr = _coerce_numeric_series(decision_deltas, name="decision_deltas")
    _validate_lengths(deltas_arr, returns_arr)
    n_bars = len(returns_arr)
    (
        decision_eligible_arr,
        score_eligible_arr,
        scheduled_decision_mask,
        eligible_decision_mask,
        block_eligible_mask,
        starts,
    ) = _contract_block_masks(
        n_bars,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    decision_deltas_out = np.zeros(n_bars, dtype=np.float64)
    decision_positions = np.full(n_bars, np.nan, dtype=np.float64)
    fill_positions = np.full(n_bars, np.nan, dtype=np.float64)
    effective_positions = np.full(n_bars, float(contract.p_start), dtype=np.float64)
    transition_costs = np.zeros(n_bars, dtype=np.float64)
    decision_mask = np.zeros(n_bars, dtype=bool)
    fill_mask = np.zeros(n_bars, dtype=bool)
    scored_mask = np.zeros(n_bars, dtype=bool)
    commitment_countdown = np.zeros(n_bars, dtype=np.int64)

    current = float(contract.p_start)
    for start in starts:
        if not block_eligible_mask[start]:
            # An excluded scheduled block cannot mutate inventory.  A caller
            # may use NaN as its masked delta, but a supplied nonzero delta is
            # an explicit contract violation rather than a silent fallback.
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
            continue
        # A decision is made before the delayed fill.  The state remains the
        # previous position until the fill bar, then stays fixed for H bars.
        raw_delta = float(deltas_arr[start])
        next_position = _candidate_position(contract, current, raw_delta)
        actual_delta = next_position - current
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        decision_mask[start] = True
        decision_deltas_out[start] = actual_delta
        decision_positions[start] = next_position
        fill_mask[fill] = True
        fill_positions[fill] = next_position
        transition_costs[fill] = transition_cost(current, next_position, contract)
        effective_positions[fill:end] = next_position
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
        eligible_decision_mask=eligible_decision_mask,
        block_eligible_mask=block_eligible_mask,
    )


def decision_deltas_from_positions(
    positions: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    strict_blocked: bool = True,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Convert an absolute policy path to contract deltas without fallback.

    Only eligible complete decision bars are allowed to change.  In strict
    mode every blocked-bar target, including an excluded scheduled block, must
    equal the currently committed position, preventing a legacy every-bar
    actor path from being silently clipped into the new contract.
    """
    contract = contract or ActionExecutionContract.canonical()
    positions_arr = _coerce_numeric_series(positions, name="positions")
    deltas = np.zeros(len(positions_arr), dtype=np.float64)
    current = float(contract.p_start)
    (
        _,
        _,
        _,
        _,
        block_eligible_mask,
        starts_tuple,
    ) = _contract_block_masks(
        len(positions_arr),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    starts = set(starts_tuple)
    for bar, target in enumerate(positions_arr):
        if bar in starts:
            if not block_eligible_mask[bar]:
                if strict_blocked and not np.isclose(
                    target,
                    current,
                    atol=_FLOAT_TOL,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"position path changes during ineligible block at bar {bar}"
                    )
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
            current = candidate
        elif strict_blocked and not np.isclose(target, current, atol=_FLOAT_TOL, rtol=0.0):
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
) -> np.ndarray:
    """Select causal block actions from one scalar score per decision start.

    ``decision_block_scores`` is a full-length vector, but only
    ``decision_block_scores[t]`` is read for a complete decision start ``t``.
    Each scalar is the cumulative four-bar forecast already available at that
    decision time; blocked/outcome-bar cells are deliberately ignored.  U0
    uses :func:`select_hindsight_block_decisions` instead because it consumes
    realized per-bar returns.
    """
    contract = contract or ActionExecutionContract.canonical()
    (
        scores,
        _,
        _,
        _,
        _,
        block_eligible,
        starts,
    ) = _validate_decision_block_scores(
        decision_block_scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )

    deltas = np.zeros(len(scores), dtype=np.float64)
    current = float(contract.p_start)
    for start in starts:
        if not block_eligible[start]:
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
        current = best_next
    return deltas


def select_hindsight_block_decisions(
    realized_returns: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
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
        block_eligible,
        starts,
    ) = _contract_block_masks(
        len(scores),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    if not starts:
        raise ValueError("action path requires at least one complete decision block")

    for start in starts:
        if not block_eligible[start]:
            continue
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        if not np.all(np.isfinite(scores[fill:end])):
            raise ValueError(
                f"realized_returns must be finite on eligible score block {start}"
            )

    # Reachable states at each block boundary are finite because every action
    # is clipped to the bounded spot allocation interval.
    states: list[tuple[float, ...]] = [(float(contract.p_start),)]
    for index, _ in enumerate(starts):
        previous_states = states[-1]
        if not block_eligible[starts[index]]:
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
        if not block_eligible[start]:
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
        _,
        block_eligible,
        starts,
    ) = _validate_decision_block_scores(
        decision_block_scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    deltas = select_block_decisions(
        scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    replay_returns = np.zeros(len(scores), dtype=np.float64)
    for start in starts:
        if not block_eligible[start]:
            continue
        fill = start + contract.execution_delay_bars
        end = fill + contract.h_decision
        replay_returns[fill:end] = float(scores[start]) / contract.h_decision
    return replay_action_path(
        replay_returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )


def replay_hindsight_selected_path(
    realized_returns: np.ndarray | Sequence[float],
    contract: ActionExecutionContract | None = None,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
) -> ActionExecutionTrajectory:
    """Select and replay the realized-future U0 trajectory."""
    contract = contract or ActionExecutionContract.canonical()
    deltas = select_hindsight_block_decisions(
        realized_returns,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    return replay_action_path(
        realized_returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
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
    **kwargs: Any,
):
    """Stage adapter: validate absolute policy paths, then invoke new Backtest.

    ``backtest_cls`` is injected by the existing pipeline.  The explicit
    contract keyword is mandatory here; a class that does not support it fails
    loudly instead of falling back to historical delay/cost defaults.
    """
    decision_eligible_arr, score_eligible_arr = validate_eligibility_masks(
        decision_eligible,
        score_eligible,
        len(np.asarray(returns).reshape(-1)),
    )
    decision_deltas = decision_deltas_from_positions(
        absolute_positions,
        contract,
        decision_eligible=decision_eligible_arr,
        score_eligible=score_eligible_arr,
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
    # Do not even forward historical cost/delay knobs on the explicit path.
    # The contract is the only source of these values; accepting the legacy
    # kwargs at the stage boundary is retained solely for call-site shape
    # compatibility.
    for legacy_key in (
        "spread_bps",
        "fee_rate",
        "slippage_bps",
        "execution_delay_bars",
        "initial_position",
        "benchmark_initial_position",
    ):
        kwargs.pop(legacy_key, None)
    kwargs["benchmark_positions"] = benchmark_deltas
    kwargs["action_execution_contract"] = contract
    kwargs["action_positions_are_deltas"] = True
    kwargs["decision_eligible"] = decision_eligible_arr
    kwargs["score_eligible"] = score_eligible_arr
    return backtest_cls(returns, decision_deltas, **kwargs)


def replay_contract_absolute_path(
    returns: np.ndarray | Sequence[float],
    absolute_positions: np.ndarray | Sequence[float],
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
) -> ActionExecutionTrajectory:
    """Strictly convert and replay an absolute policy path."""
    deltas = decision_deltas_from_positions(
        absolute_positions,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    return replay_action_path(
        returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )


def contract_pnl_attribution(
    returns: np.ndarray | Sequence[float],
    absolute_positions: np.ndarray | Sequence[float],
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray | Sequence[bool] | None = None,
    score_eligible: np.ndarray | Sequence[bool] | None = None,
) -> dict[str, float]:
    """Return long/short/cost attribution from the shared contract replay."""
    trajectory = replay_contract_absolute_path(
        returns,
        absolute_positions,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
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
