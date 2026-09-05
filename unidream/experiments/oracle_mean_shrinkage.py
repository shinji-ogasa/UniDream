"""Fixed half shrinkage toward a supplied frozen constant mean forecast."""
from __future__ import annotations

import numpy as np


def half_mean(mu, anchor, *, inference_mask) -> np.ndarray:
    """Return 0.5 * anchor + 0.5 * mu on the unchanged inference support.

    The caller supplies the frozen scale-mean anchor; this function neither
    estimates it nor consumes outcomes or scoring support. Its selected values
    must be exactly constant. Unavailable rows remain NaN and are ignored.
    Scaling before addition avoids overflowing either a same-sign sum or an
    opposite-sign subtraction. Inputs are never modified.
    """
    inference = np.asarray(inference_mask)
    if inference.ndim != 1 or inference.dtype != np.dtype(bool):
        raise ValueError("inference_mask must be a one-dimensional boolean mask")
    if not len(inference) or not inference.any():
        raise ValueError("nonempty inference support required")
    try:
        forecast, frozen = np.asarray(mu, dtype=float), np.asarray(anchor, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric mean and anchor arrays required") from exc
    if forecast.shape != inference.shape or frozen.shape != inference.shape:
        raise ValueError("aligned one-dimensional mean, anchor and inference arrays required")
    selected_mu, selected_anchor = forecast[inference], frozen[inference]
    if not np.isfinite(selected_mu).all() or not np.isfinite(selected_anchor).all():
        raise ValueError("claimed-valid means and anchors must be finite")
    if not np.all(selected_anchor == selected_anchor[0]):
        raise ValueError("anchor must be exactly constant over inference support")
    result = np.full(len(inference), np.nan)
    result[inference] = .5 * selected_anchor + .5 * selected_mu
    return result


__all__ = ["half_mean"]
