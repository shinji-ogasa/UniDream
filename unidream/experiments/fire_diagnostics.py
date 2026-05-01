from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FireDangerDiagnostics:
    horizon: int
    fire_count: int
    valid_fire_count: int
    danger_fire_rate: float
    pre_dd_danger_rate: float
    future_mdd_overlap_rate: float
    global_mdd_overlap_rate: float
    safe_fire_rate: float
    fire_advantage_mean: float
    safe_fire_advantage_mean: float
    post_fire_dd_contribution_mean: float
    safe_fire_pnl: float
    danger_fire_pnl: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def empty_fire_danger_diagnostics(horizon: int) -> FireDangerDiagnostics:
    return FireDangerDiagnostics(
        horizon=int(horizon),
        fire_count=0,
        valid_fire_count=0,
        danger_fire_rate=0.0,
        pre_dd_danger_rate=0.0,
        future_mdd_overlap_rate=0.0,
        global_mdd_overlap_rate=0.0,
        safe_fire_rate=0.0,
        fire_advantage_mean=0.0,
        safe_fire_advantage_mean=0.0,
        post_fire_dd_contribution_mean=0.0,
        safe_fire_pnl=0.0,
        danger_fire_pnl=0.0,
    )


def _bar_pnl(
    ret: float,
    position: float,
    prev_position: float,
    costs_cfg: dict,
) -> float:
    spread_bps = float(costs_cfg.get("spread_bps", 5.0))
    fee_rate = float(costs_cfg.get("fee_rate", 0.0004))
    slippage_bps = float(costs_cfg.get("slippage_bps", 2.0))
    delta_pos = abs(float(position) - float(prev_position))
    cost = (spread_bps / 10000.0) * 0.5 * delta_pos
    cost += fee_rate * delta_pos
    cost += (slippage_bps / 10000.0) * delta_pos
    return float(position) * float(ret) - cost


def _bar_pnl_series(
    returns: np.ndarray,
    positions: np.ndarray,
    costs_cfg: dict,
) -> np.ndarray:
    t = min(len(returns), len(positions))
    out = np.zeros(t, dtype=np.float64)
    prev_position = 0.0
    for i in range(t):
        out[i] = _bar_pnl(float(returns[i]), float(positions[i]), prev_position, costs_cfg)
        prev_position = float(positions[i])
    return out


def _window_pnl(
    returns: np.ndarray,
    positions: np.ndarray,
    start: int,
    horizon: int,
    costs_cfg: dict,
) -> np.ndarray:
    end = min(len(returns), int(start) + int(horizon))
    out = np.zeros(max(0, end - int(start)), dtype=np.float64)
    prev_position = float(positions[int(start) - 1]) if int(start) > 0 else 0.0
    for k, i in enumerate(range(int(start), end)):
        out[k] = _bar_pnl(float(returns[i]), float(positions[i]), prev_position, costs_cfg)
        prev_position = float(positions[i])
    return out


def _maxdd_interval(equity: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(equity, dtype=np.float64)
    if len(arr) == 0:
        return {"peak": 0, "trough": 0, "maxdd": 0.0}

    peak_idx = 0
    peak_value = float(arr[0])
    best_peak = 0
    best_trough = 0
    maxdd = 0.0
    for i, value in enumerate(arr):
        value = float(value)
        if value > peak_value:
            peak_value = value
            peak_idx = i
        dd = value / max(peak_value, 1e-12) - 1.0
        if dd < maxdd:
            maxdd = dd
            best_peak = peak_idx
            best_trough = i
    return {"peak": int(best_peak), "trough": int(best_trough), "maxdd": float(maxdd)}


def _local_drawdown(pnl: np.ndarray) -> np.ndarray:
    if len(pnl) == 0:
        return np.zeros(0, dtype=np.float64)
    equity = np.exp(np.cumsum(np.asarray(pnl, dtype=np.float64)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return np.maximum(0.0, -drawdown)


def _rolling_past_vol(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    window = max(int(window), 1)
    for i in range(len(arr)):
        start = max(0, i - window)
        if i > start:
            out[i] = float(np.std(arr[start:i]))
    return out


def _fire_run_end_indices(fire: np.ndarray) -> np.ndarray:
    mask = np.asarray(fire, dtype=bool)
    out = np.arange(len(mask), dtype=np.int64)
    run_end = len(mask)
    for i in range(len(mask) - 1, -1, -1):
        if mask[i]:
            if i + 1 >= len(mask) or not mask[i + 1]:
                run_end = i + 1
            out[i] = run_end
        else:
            out[i] = i
    return out


def _global_mdd_mask(
    returns: np.ndarray,
    positions: np.ndarray,
    costs_cfg: dict,
) -> np.ndarray:
    pnl = _bar_pnl_series(returns, positions, costs_cfg)
    equity = np.exp(np.cumsum(pnl))
    interval = _maxdd_interval(equity)
    mask = np.zeros(len(pnl), dtype=bool)
    peak = int(interval["peak"])
    trough = int(interval["trough"])
    if trough >= peak:
        mask[peak : trough + 1] = True
    return mask


def evaluate_fire_danger_diagnostics(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    fire: np.ndarray,
    costs_cfg: dict,
    horizon: int = 32,
    rel_vol_window: int = 64,
    mdd_rel_threshold: float = 0.5,
    post_dd_quantile: float = 0.8,
    include_post_dd_in_danger: bool = True,
) -> FireDangerDiagnostics:
    """Historical validation diagnostics for adapter fire safety.

    Labels intentionally use future returns inside the validation/test window.
    This function is for checkpoint selection and offline diagnostics, not for
    real-time inference features.
    """
    t = min(len(returns), len(positions), len(no_adapter), len(fire))
    horizon = max(int(horizon), 1)
    if t == 0:
        return empty_fire_danger_diagnostics(horizon)

    returns_arr = np.asarray(returns[:t], dtype=np.float64)
    positions_arr = np.asarray(positions[:t], dtype=np.float64)
    no_adapter_arr = np.asarray(no_adapter[:t], dtype=np.float64)
    fire_arr = np.asarray(fire[:t], dtype=bool)
    fire_count = int(np.sum(fire_arr))
    if fire_count == 0:
        return empty_fire_danger_diagnostics(horizon)

    valid = np.zeros(t, dtype=bool)
    fire_advantage = np.full(t, np.nan, dtype=np.float64)
    post_fire_dd_contribution = np.full(t, np.nan, dtype=np.float64)
    future_mdd_overlap = np.zeros(t, dtype=bool)
    pre_dd_state = np.zeros(t, dtype=bool)
    global_mdd_overlap = _global_mdd_mask(returns_arr, positions_arr, costs_cfg)
    run_end = _fire_run_end_indices(fire_arr)
    roll_vol = _rolling_past_vol(returns_arr, rel_vol_window)

    for i in range(0, max(0, t - horizon + 1)):
        if not fire_arr[i]:
            continue
        pnl_on = _window_pnl(returns_arr, positions_arr, i, horizon, costs_cfg)
        pnl_off = _window_pnl(returns_arr, no_adapter_arr, i, horizon, costs_cfg)
        if len(pnl_on) < horizon or len(pnl_off) < horizon:
            continue
        valid[i] = True
        equity_on = np.exp(np.cumsum(pnl_on))
        interval = _maxdd_interval(equity_on)
        dd_on = _local_drawdown(pnl_on)
        dd_off = _local_drawdown(pnl_off)
        dd_diff = dd_on[: min(len(dd_on), len(dd_off))] - dd_off[: min(len(dd_on), len(dd_off))]
        mdd_on = float(np.max(dd_on)) if len(dd_on) else 0.0
        denom = max(float(roll_vol[i]) * math.sqrt(float(horizon)), 1e-8)
        threshold = float(mdd_rel_threshold) * denom
        peak = int(interval["peak"])
        trough = int(interval["trough"])
        run_stop = max(0, min(horizon - 1, int(run_end[i] - i - 1)))

        fire_advantage[i] = float(np.sum(pnl_on) - np.sum(pnl_off))
        post_fire_dd_contribution[i] = float(max(0.0, np.max(dd_diff))) if len(dd_diff) else 0.0
        future_mdd_overlap[i] = bool(mdd_on > threshold and peak <= run_stop and trough >= 0)
        pre_dd_state[i] = bool(
            mdd_on > threshold
            and peak > run_stop
            and peak <= max(run_stop + 1, horizon // 2)
        )

    valid_fire = fire_arr & valid
    valid_count = int(np.sum(valid_fire))
    if valid_count == 0:
        return empty_fire_danger_diagnostics(horizon)

    post_values = post_fire_dd_contribution[valid_fire]
    finite_post = post_values[np.isfinite(post_values)]
    post_threshold = (
        float(np.quantile(finite_post, float(post_dd_quantile)))
        if len(finite_post)
        else float("inf")
    )
    high_post_dd = post_fire_dd_contribution >= post_threshold
    danger = future_mdd_overlap | pre_dd_state
    if include_post_dd_in_danger:
        danger = danger | high_post_dd
    safe = valid_fire & (~danger) & (fire_advantage > 0.0)

    policy_pnl = _bar_pnl_series(returns_arr, positions_arr, costs_cfg)
    danger_fire = valid_fire & danger
    adv_valid = fire_advantage[valid_fire]
    adv_safe = fire_advantage[safe]
    post_valid = post_fire_dd_contribution[valid_fire]
    return FireDangerDiagnostics(
        horizon=horizon,
        fire_count=fire_count,
        valid_fire_count=valid_count,
        danger_fire_rate=float(np.mean(danger[valid_fire])),
        pre_dd_danger_rate=float(np.mean(pre_dd_state[valid_fire])),
        future_mdd_overlap_rate=float(np.mean(future_mdd_overlap[valid_fire])),
        global_mdd_overlap_rate=float(np.mean(global_mdd_overlap[valid_fire])),
        safe_fire_rate=float(np.mean(safe[valid_fire])),
        fire_advantage_mean=float(np.nanmean(adv_valid)) if len(adv_valid) else 0.0,
        safe_fire_advantage_mean=float(np.nanmean(adv_safe)) if len(adv_safe) else 0.0,
        post_fire_dd_contribution_mean=(
            float(np.nanmean(post_valid)) if len(post_valid) else 0.0
        ),
        safe_fire_pnl=float(np.sum(policy_pnl[safe])) if np.any(safe) else 0.0,
        danger_fire_pnl=float(np.sum(policy_pnl[danger_fire])) if np.any(danger_fire) else 0.0,
    )
