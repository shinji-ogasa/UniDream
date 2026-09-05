"""Additional UM-flow availability delays on the unchanged Spot decision grid."""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from .oracle_derivative_features import PERP_FLOW_FEATURE_NAMES, make_derivative_groups


def make_delayed_perp_groups(
    spot: pd.DataFrame, um: pd.DataFrame, delays=(0, 1, 4),
) -> dict[str, pd.DataFrame]:
    """Keep technical29 fixed and add UM flow24/96 delayed by extra bars.

    The source helper validates raw UTC bar-open timing and the complete Spot
    15-minute grid, aligns sparse UM rows as NaN, and shifts its features once.
    Only those already-causal UM flow columns receive an additional row shift:
    delay k at decision t uses UM bars through t-k-1. Spot features, decision
    timestamps and missing rows are never shifted, filled or compressed here.
    """
    try:
        delays = tuple(delays)
    except TypeError as exc:
        raise ValueError("delays must be distinct nonnegative integers including 0") from exc
    if (not delays or
            any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0
                for k in delays) or
            len(set(delays)) != len(delays) or 0 not in delays):
        raise ValueError("delays must be distinct nonnegative integers including 0")
    original = make_derivative_groups(spot, um)
    technical = original["technical"]
    flow = original["perp_flow"].loc[:, PERP_FLOW_FEATURE_NAMES]
    result = {"technical": technical}
    for delay in delays:
        result[f"perp_delay{int(delay)}"] = pd.concat(
            [technical, flow.shift(int(delay))], axis=1)
    return result


__all__ = ["make_delayed_perp_groups"]
