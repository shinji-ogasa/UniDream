"""Explicit hindsight substitutions on a frozen forecast's existing score rows.

These are diagnostic hybrids, never causal forecasts or training teachers.
Unscored inference slots retain learned values and unavailable inputs remain
unavailable. Future labels never add or remove an order opportunity.
"""
from __future__ import annotations

import numpy as np


SWAPS = ("return", "realized_risk", "both")


def substitute_information(mu, variance, *, inference_mask, score_support, actual, swap):
    if swap not in SWAPS:
        raise ValueError("unregistered information substitution")
    inference, score = np.asarray(inference_mask), np.asarray(score_support)
    if (inference.ndim != 1 or score.shape != inference.shape or inference.dtype != bool
            or score.dtype != bool or np.any(score & ~inference) or not score.any()):
        raise ValueError("nonempty frozen scoring subset of boolean inference required")
    mean, risk, observed = np.asarray(mu), np.asarray(variance), np.asarray(actual)
    if (mean.shape != inference.shape or risk.shape != inference.shape
            or observed.shape != (len(inference), 3)
            or any(np.iscomplexobj(v) or v.dtype.kind not in "fiu" for v in (mean, risk, observed))):
        raise ValueError("aligned real numeric forecast and three outcome arrays required")
    if (not np.isfinite(mean[inference]).all() or not np.isfinite(risk[inference]).all()
            or np.any(risk[inference] < 0)
            or not np.isnan(mean[~inference]).all() or not np.isnan(risk[~inference]).all()):
        raise ValueError("frozen forecast support must be finite or explicitly NaN")
    # Unscored outcomes, including tail labels, are deliberately never used.
    selected = observed[score]
    if not np.isfinite(selected).all() or np.any(selected[:, 2] < 0):
        raise ValueError("scored hindsight outcomes must be finite with nonnegative realized volatility")
    out_mean, out_risk = mean.astype(float, copy=True), risk.astype(float, copy=True)
    if swap in ("return", "both"):
        out_mean[score] = selected[:, 0]
    if swap in ("realized_risk", "both"):
        with np.errstate(over="ignore"):
            realized = selected[:, 2].astype(float) ** 2
        if not np.isfinite(realized).all():
            raise ValueError("nonfinite realized risk substitution")
        # This is measured quadratic variation, not conditional variance.
        out_risk[score] = realized
    metadata = {"diagnostic_kind": "hybrid_hindsight_information_substitution", "swap": swap,
        "hindsight_only": True, "future_information_used_for_decisions": True,
        "deployable": False, "teacher_use_allowed": False, "global_optimum_claimed": False,
        "inference_rows": int(inference.sum()), "replacement_rows": int(score.sum()),
        "learned_remainder_rows": int((inference & ~score).sum()),
        "inference_and_missing_action_support_unchanged": True,
        "risk_semantics": "squared realized h24 volatility; not true conditional variance"}
    return {"mu": out_mean, "variance": out_risk, "inference_mask": inference.copy(),
            "score_support": score.copy(), "metadata": metadata}


def mark_hindsight_trace(trace, *, swap, score_support):
    """Override the causal planner's metadata when its inputs use future values."""
    support = np.asarray(score_support)
    if support.ndim != 1 or support.dtype != bool:
        raise ValueError("one-dimensional boolean replacement support required")
    result = {**trace, "diagnostic_kind": "hindsight_hybrid_conditional_utility_planner",
        "future_information_used_for_decisions": True, "hindsight_only": True,
        "deployable": False, "teacher_use_allowed": False, "information_swap": swap,
        "realized_risk_is_conditional_variance": False}
    indices = result["decision_trace"]["bar_indices"]
    result["decision_trace"] = {**result["decision_trace"],
        "hindsight_information_replaced": [bool(support[i]) for i in indices]}
    if "reasons" in result["decision_trace"]:
        result["decision_trace"]["reasons"] = [
            "hybrid_hindsight" if support[i] else reason
            for i, reason in zip(indices, result["decision_trace"]["reasons"])]
    return result
