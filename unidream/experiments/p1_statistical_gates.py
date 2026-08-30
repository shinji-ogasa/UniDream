"""Fixed Holm and Wilson gates for the authenticated P1 comparison family."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from .p1_result_registry import P1ResultRegistry, load_p1_result_registry


class P1StatisticalGateError(ValueError):
    """Raised when a gate input does not cover the fixed P1 family exactly."""


P1_FAMILYWISE_ALPHA = 0.05
P1_WILSON_Z = 1.959963984540054
P1_S0_COMPARISON_IDS = frozenset(
    {
        "S0__ridge__utility_vs_hold__cost_on",
        "S0__persistence__utility_vs_hold__cost_on",
    }
)


@dataclass(frozen=True)
class HolmRow:
    comparison_id: str
    raw_p: float
    rank: int
    alpha_rank: float
    adjusted_p: float
    rejected: bool


@dataclass(frozen=True)
class HolmFamily:
    alpha: float
    rows: tuple[HolmRow, ...]
    by_id: Mapping[str, HolmRow]


@dataclass(frozen=True)
class WilsonInterval:
    successes: int
    total: int
    point: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    z: float = P1_WILSON_Z


def _strict_probability(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise P1StatisticalGateError(f"{name} must be a probability")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1StatisticalGateError(f"{name} must be a probability") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise P1StatisticalGateError(f"{name} must be finite and in [0,1]")
    return result


def holm_bonferroni_fixed_family(
    raw_p_values: Mapping[str, Any],
    *,
    registry: P1ResultRegistry | None = None,
    alpha: float = P1_FAMILYWISE_ALPHA,
) -> HolmFamily:
    """Apply the preregistered 16-row Holm step-down with lexical tie breaks."""

    if registry is None:
        registry = load_p1_result_registry()
    if not isinstance(registry, P1ResultRegistry):
        raise P1StatisticalGateError("Holm requires the authenticated P1 registry")
    alpha_value = _strict_probability(alpha, name="alpha")
    if alpha_value != P1_FAMILYWISE_ALPHA:
        raise P1StatisticalGateError("P1 familywise alpha is fixed at 0.05")
    if not isinstance(raw_p_values, Mapping):
        raise P1StatisticalGateError("raw_p_values must be a mapping")
    expected_ids = tuple(row["comparison_id"] for row in registry.comparisons)
    if set(raw_p_values) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(raw_p_values))
        extra = sorted(set(raw_p_values) - set(expected_ids))
        raise P1StatisticalGateError(
            f"Holm family is not exact (missing={missing}, extra={extra})"
        )
    normalized = {
        comparison_id: _strict_probability(
            raw_p_values[comparison_id], name=f"raw_p[{comparison_id}]"
        )
        for comparison_id in expected_ids
    }
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    adjusted_running = 0.0
    step_down_open = True
    rows: list[HolmRow] = []
    for zero_rank, (comparison_id, raw_p) in enumerate(ordered):
        rank = zero_rank + 1
        remaining = family_size - zero_rank
        alpha_rank = alpha_value / remaining
        passes_rank = raw_p <= alpha_rank
        rejected = bool(step_down_open and passes_rank)
        if not passes_rank:
            step_down_open = False
        adjusted_running = max(adjusted_running, remaining * raw_p)
        rows.append(
            HolmRow(
                comparison_id=comparison_id,
                raw_p=raw_p,
                rank=rank,
                alpha_rank=alpha_rank,
                adjusted_p=min(1.0, adjusted_running),
                rejected=rejected,
            )
        )
    by_id = MappingProxyType({row.comparison_id: row for row in rows})
    return HolmFamily(alpha=alpha_value, rows=tuple(rows), by_id=by_id)


def wilson_score_interval(successes: Any, total: Any) -> WilsonInterval:
    """Return the fixed 95% Wilson interval; N/A counts fail closed."""

    for name, value in (("successes", successes), ("total", total)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise P1StatisticalGateError(f"{name} must be an integer")
    x = int(successes)
    n = int(total)
    if n <= 0 or x < 0 or x > n:
        raise P1StatisticalGateError("Wilson counts require 0 <= successes <= total and total > 0")
    point = x / n
    z2 = P1_WILSON_Z * P1_WILSON_Z
    denominator = 1.0 + z2 / n
    center = (point + z2 / (2.0 * n)) / denominator
    half = (
        P1_WILSON_Z
        * math.sqrt(point * (1.0 - point) / n + z2 / (4.0 * n * n))
        / denominator
    )
    return WilsonInterval(
        successes=x,
        total=n,
        point=point,
        lower=center - half,
        upper=center + half,
    )


def evaluate_s0_safety_bounds(
    comparison_id: str,
    bootstrap_values_by_block_length: Mapping[int, Sequence[Any]],
    *,
    holm: HolmFamily,
) -> Mapping[str, Any]:
    """Evaluate the special S0 adjusted lower-bound/no-rejection safety rule."""

    if comparison_id not in P1_S0_COMPARISON_IDS:
        raise P1StatisticalGateError("S0 safety bounds only apply to the two fixed S0 rows")
    if not isinstance(holm, HolmFamily) or comparison_id not in holm.by_id:
        raise P1StatisticalGateError("S0 safety bounds require the completed Holm family")
    if set(bootstrap_values_by_block_length) != {8, 16, 32}:
        raise P1StatisticalGateError("S0 safety bounds require block lengths 8, 16, and 32")
    row = holm.by_id[comparison_id]
    lower_bounds: dict[int, float] = {}
    for block_length in (8, 16, 32):
        try:
            values = np.asarray(
                bootstrap_values_by_block_length[block_length], dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise P1StatisticalGateError("S0 bootstrap values must be numeric") from exc
        if values.shape != (2000,) or not np.isfinite(values).all():
            raise P1StatisticalGateError(
                "S0 safety bounds require exactly 2000 finite bootstrap values per L"
            )
        lower_bounds[block_length] = float(
            np.quantile(values, row.alpha_rank, method="linear")
        )
    passed = (not row.rejected) and all(value <= 0.0 for value in lower_bounds.values())
    return MappingProxyType(
        {
            "comparison_id": comparison_id,
            "rank": row.rank,
            "alpha_rank": row.alpha_rank,
            "positive_edge_rejected": row.rejected,
            "adjusted_lower_bounds": MappingProxyType(lower_bounds),
            "passed": passed,
        }
    )


__all__ = [
    "HolmFamily",
    "HolmRow",
    "P1StatisticalGateError",
    "P1_FAMILYWISE_ALPHA",
    "P1_S0_COMPARISON_IDS",
    "P1_WILSON_Z",
    "WilsonInterval",
    "evaluate_s0_safety_bounds",
    "holm_bonferroni_fixed_family",
    "wilson_score_interval",
]
