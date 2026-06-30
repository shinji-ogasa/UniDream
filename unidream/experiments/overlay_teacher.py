from __future__ import annotations

import numpy as np


def _risk_gate_from_advantage(
    advantage_values,
    *,
    indices: tuple[int, ...],
    length: int,
    center: float,
    scale: float,
    min_gate: float,
) -> np.ndarray | None:
    if not indices or advantage_values is None or length <= 0:
        return None
    adv = np.asarray(advantage_values, dtype=np.float32)
    if adv.ndim == 1:
        adv = adv[:, None]
    adv = adv[:length]
    valid_indices = [i for i in indices if 0 <= i < adv.shape[1]]
    if not valid_indices:
        return None
    risk_signal = np.nan_to_num(adv[:, valid_indices].mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    raw_gate = 1.0 / (1.0 + np.exp(-np.clip((risk_signal - center) * scale, -30.0, 30.0)))
    return min_gate + (1.0 - min_gate) * raw_gate


def _risk_signal_from_advantage(
    advantage_values,
    *,
    indices: tuple[int, ...],
    length: int,
) -> np.ndarray | None:
    if not indices or advantage_values is None or length <= 0:
        return None
    adv = np.asarray(advantage_values, dtype=np.float32)
    if adv.ndim == 1:
        adv = adv[:, None]
    adv = adv[:length]
    valid_indices = [i for i in indices if 0 <= i < adv.shape[1]]
    if not valid_indices:
        return None
    return np.nan_to_num(adv[:, valid_indices].mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _edge_protect_gate(
    advantage_values,
    *,
    indices: tuple[int, ...],
    length: int,
    center: float,
    scale: float,
) -> np.ndarray | None:
    edge = _risk_signal_from_advantage(advantage_values, indices=indices, length=length)
    if edge is None:
        return None
    raw = np.clip((edge - center) * scale, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-raw))


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.empty_like(x)
    prev = float(x[0])
    out[0] = prev
    for i in range(1, x.shape[0]):
        prev = (1.0 - alpha) * prev + alpha * float(x[i])
        out[i] = prev
    return out


def _causal_trailing_sum(returns, length: int, lookback: int) -> np.ndarray:
    if returns is None:
        return np.zeros(length, dtype=np.float32)
    r = np.asarray(returns, dtype=np.float32)[:length]
    if r.shape[0] < length:
        r = np.pad(r, (0, length - r.shape[0]), mode="constant")
    shifted = np.concatenate([[0.0], r[:-1]])
    csum = np.concatenate([[0.0], np.cumsum(shifted)])
    lb = max(int(lookback), 1)
    out = csum[1:] - csum[np.maximum(np.arange(length) + 1 - lb, 0)]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _causal_trailing_drawdown(returns, length: int, lookback: int) -> np.ndarray:
    if returns is None or lookback <= 0:
        return np.zeros(length, dtype=np.float32)
    r = np.asarray(returns, dtype=np.float32)[:length]
    if r.shape[0] < length:
        r = np.pad(r, (0, length - r.shape[0]), mode="constant")
    shifted = np.concatenate([[0.0], r[:-1]])
    equity = np.cumsum(np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0))
    lb = max(int(lookback), 1)
    out = np.zeros(length, dtype=np.float32)
    for i in range(length):
        start = max(0, i + 1 - lb)
        peak = float(np.max(equity[start : i + 1]))
        out[i] = max(0.0, peak - float(equity[i]))
    return out


def _lowfreq_wm_overlay(
    *,
    advantage_values,
    returns,
    bc_cfg: dict,
    length: int,
) -> np.ndarray | None:
    risk_indices = tuple(int(i) for i in bc_cfg.get("benchmark_overlay_risk_gate_indices", []) or [])
    risk = _risk_signal_from_advantage(advantage_values, indices=risk_indices, length=length)
    if risk is None:
        return None
    edge_source = bc_cfg.get("benchmark_overlay_edge_indices")
    if not edge_source:
        edge_source = bc_cfg.get("benchmark_overlay_edge_protect_indices", [])
    edge_indices = tuple(int(i) for i in edge_source or [])
    edge = _risk_signal_from_advantage(advantage_values, indices=edge_indices, length=length)
    risk_slow = _ema(risk, int(bc_cfg.get("benchmark_overlay_lowfreq_ema_span", 256)))
    edge_slow = _ema(edge, int(bc_cfg.get("benchmark_overlay_lowfreq_ema_span", 256))) if edge is not None else None
    trend = _causal_trailing_sum(
        returns,
        length,
        int(bc_cfg.get("benchmark_overlay_lowfreq_trend_lookback", 128)),
    )
    dd_depth = _causal_trailing_drawdown(
        returns,
        length,
        int(bc_cfg.get("benchmark_overlay_lowfreq_dd_lookback", 0)),
    )
    risk_hi = float(bc_cfg.get("benchmark_overlay_lowfreq_risk_hi", 1.0))
    risk_lo = float(bc_cfg.get("benchmark_overlay_lowfreq_risk_lo", 0.0))
    trend_max = float(bc_cfg.get("benchmark_overlay_lowfreq_trend_max", 0.0))
    trend_min = float(bc_cfg.get("benchmark_overlay_lowfreq_trend_min", 0.0))
    edge_hi = float(bc_cfg.get("benchmark_overlay_lowfreq_edge_hi", 0.0))
    edge_lo = float(bc_cfg.get("benchmark_overlay_lowfreq_edge_lo", 0.0))
    dd_min = float(bc_cfg.get("benchmark_overlay_lowfreq_dd_min", 0.0))
    dd_max = float(bc_cfg.get("benchmark_overlay_lowfreq_dd_max", np.inf))
    base = float(bc_cfg.get("benchmark_overlay_lowfreq_base", 0.0))
    down = abs(float(bc_cfg.get("benchmark_overlay_lowfreq_down", 0.04)))
    up = abs(float(bc_cfg.get("benchmark_overlay_lowfreq_up", 0.0)))
    base_mode = str(bc_cfg.get("benchmark_overlay_lowfreq_base_mode", "always")).lower()
    overlay = np.full(length, base, dtype=np.float32)
    down_mask = (risk_slow >= risk_hi) & (trend <= trend_max)
    up_mask = (risk_slow <= risk_lo) & (trend >= trend_min)
    if int(bc_cfg.get("benchmark_overlay_lowfreq_dd_lookback", 0)) > 0:
        down_mask = down_mask & (dd_depth >= dd_min)
        up_mask = up_mask & (dd_depth <= dd_max)
    if edge_slow is not None:
        down_mask = down_mask & (edge_slow <= edge_lo)
        up_mask = up_mask & (edge_slow >= edge_hi)
    if base_mode in {"safe", "safe_only", "risk_off"}:
        base_mask = (trend >= trend_min)
        if int(bc_cfg.get("benchmark_overlay_lowfreq_dd_lookback", 0)) > 0:
            base_mask = base_mask & (dd_depth <= dd_max)
        overlay = np.zeros(length, dtype=np.float32)
        overlay[base_mask] = base
    overlay[down_mask] = base - down
    overlay[up_mask] = base + up

    min_hold = int(bc_cfg.get("benchmark_overlay_lowfreq_min_hold", 0))
    deadzone = float(max(bc_cfg.get("benchmark_overlay_deadzone", 0.0), 0.0))
    if min_hold > 0 and overlay.size > 1:
        held = overlay.copy()
        current = 0.0
        hold = min_hold
        for i, target in enumerate(held):
            target = float(target)
            if hold < min_hold and abs(target - current) > deadzone:
                target = current
            if abs(target - current) > 1e-8:
                hold = 0
            else:
                hold += 1
            current = target
            held[i] = current
        overlay = held
    return overlay


def _affine_wm_overlay(
    *,
    advantage_values,
    returns,
    bc_cfg: dict,
    length: int,
) -> np.ndarray | None:
    """Build a continuous WM risk-budget overlay.

    Unlike the low-frequency hard gate, this keeps sizing continuous: risk only
    reduces exposure when edge is weak, while edge/utility can add a small
    positive overlay.
    """
    risk_indices = tuple(
        int(i)
        for i in (
            bc_cfg.get("benchmark_overlay_affine_risk_indices")
            or bc_cfg.get("benchmark_overlay_risk_gate_indices", [])
            or []
        )
    )
    edge_indices = tuple(
        int(i)
        for i in (
            bc_cfg.get("benchmark_overlay_affine_edge_indices")
            or bc_cfg.get("benchmark_overlay_edge_protect_indices", [])
            or []
        )
    )
    utility_low_indices = tuple(
        int(i)
        for i in bc_cfg.get("benchmark_overlay_affine_utility_low_indices", [25, 26, 27, 28])
    )
    utility_high_indices = tuple(
        int(i)
        for i in bc_cfg.get("benchmark_overlay_affine_utility_high_indices", [29, 30, 31])
    )
    risk = _risk_signal_from_advantage(advantage_values, indices=risk_indices, length=length)
    edge = _risk_signal_from_advantage(advantage_values, indices=edge_indices, length=length)
    utility_low = _risk_signal_from_advantage(advantage_values, indices=utility_low_indices, length=length)
    utility_high = _risk_signal_from_advantage(advantage_values, indices=utility_high_indices, length=length)
    if risk is None and edge is None and utility_low is None and utility_high is None:
        return None

    span = int(bc_cfg.get("benchmark_overlay_affine_ema_span", bc_cfg.get("benchmark_overlay_lowfreq_ema_span", 256)))
    zeros = np.zeros(length, dtype=np.float32)
    risk_slow = _ema(risk if risk is not None else zeros, span)
    edge_slow = _ema(edge if edge is not None else zeros, span)
    utility_low_slow = _ema(utility_low if utility_low is not None else zeros, span)
    utility_high_slow = _ema(utility_high if utility_high is not None else zeros, span)
    utility_bias = utility_high_slow - utility_low_slow
    trend = _causal_trailing_sum(
        returns,
        length,
        int(bc_cfg.get("benchmark_overlay_affine_trend_lookback", bc_cfg.get("benchmark_overlay_lowfreq_trend_lookback", 128))),
    )

    base = float(bc_cfg.get("benchmark_overlay_affine_base", 0.0))
    risk_center = float(bc_cfg.get("benchmark_overlay_affine_risk_center", 0.0))
    edge_center = float(bc_cfg.get("benchmark_overlay_affine_edge_center", 0.0))
    trend_center = float(bc_cfg.get("benchmark_overlay_affine_trend_center", 0.0))
    edge_gate_scale = float(bc_cfg.get("benchmark_overlay_affine_edge_gate_scale", 3.0))
    risk_coef = float(bc_cfg.get("benchmark_overlay_affine_risk_coef", 0.0))
    edge_coef = float(bc_cfg.get("benchmark_overlay_affine_edge_coef", 0.0))
    utility_coef = float(bc_cfg.get("benchmark_overlay_affine_utility_coef", 0.0))
    trend_down_coef = float(bc_cfg.get("benchmark_overlay_affine_trend_down_coef", 0.0))
    trend_up_coef = float(bc_cfg.get("benchmark_overlay_affine_trend_up_coef", 0.0))

    risk_excess = np.maximum(risk_slow - risk_center, 0.0)
    weak_edge_gate = 1.0 / (1.0 + np.exp(np.clip((edge_slow - edge_center) * edge_gate_scale, -30.0, 30.0)))
    bad_trend = np.maximum(trend_center - trend, 0.0)
    good_trend = np.maximum(trend - trend_center, 0.0)
    overlay = (
        base
        + edge_coef * edge_slow
        + utility_coef * utility_bias
        + trend_up_coef * good_trend
        - risk_coef * risk_excess * weak_edge_gate
        - trend_down_coef * bad_trend
    )
    return np.nan_to_num(overlay, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def benchmark_overlay_teacher_enabled(bc_cfg: dict | None, ac_cfg: dict | None = None) -> bool:
    """Return whether the benchmark-relative overlay teacher is requested."""
    bc_cfg = bc_cfg or {}
    ac_cfg = ac_cfg or {}
    return bool(
        bc_cfg.get("benchmark_overlay_teacher", False)
        or ac_cfg.get("benchmark_overlay_teacher", False)
    )


def apply_benchmark_overlay_teacher(
    positions,
    *,
    bc_cfg: dict | None,
    ac_cfg: dict | None,
    reward_cfg: dict | None,
    advantage_values=None,
    returns=None,
) -> np.ndarray:
    """Map an absolute teacher path to a small B&H-relative overlay path.

    The original hindsight oracle can ask for full de-risk / full overweight
    jumps. Plan011 keeps that information only as direction and intensity, then
    trains the actor on a bounded continuous overlay around B&H exposure=1.0.
    """
    arr = np.asarray(positions, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()
    bc_cfg = bc_cfg or {}
    ac_cfg = ac_cfg or {}
    reward_cfg = reward_cfg or {}
    if not benchmark_overlay_teacher_enabled(bc_cfg, ac_cfg):
        return arr.copy()

    benchmark = float(bc_cfg.get(
        "benchmark_overlay_base_position",
        reward_cfg.get("benchmark_position", 1.0),
    ))
    min_overlay = float(bc_cfg.get(
        "benchmark_overlay_min",
        ac_cfg.get("residual_min_overlay", ac_cfg.get("abs_min_position", 0.0) - benchmark),
    ))
    max_overlay = float(bc_cfg.get(
        "benchmark_overlay_max",
        ac_cfg.get("residual_max_overlay", ac_cfg.get("abs_max_position", 1.0) - benchmark),
    ))
    if max_overlay < min_overlay:
        min_overlay, max_overlay = max_overlay, min_overlay

    scale = float(bc_cfg.get("benchmark_overlay_scale", 1.0))
    risk_indices = tuple(int(i) for i in bc_cfg.get("benchmark_overlay_risk_gate_indices", []) or [])
    risk_gate = _risk_gate_from_advantage(
        advantage_values,
        indices=risk_indices,
        length=arr.shape[0],
        center=float(bc_cfg.get("benchmark_overlay_risk_gate_center", 0.0)),
        scale=float(bc_cfg.get("benchmark_overlay_risk_gate_scale", 1.0)),
        min_gate=float(np.clip(bc_cfg.get("benchmark_overlay_risk_gate_min", 0.0), 0.0, 1.0)),
    )
    mode = str(bc_cfg.get("benchmark_overlay_teacher_mode", "oracle_scaled")).lower()
    if mode in {"lowfreq_wm", "lowfreq_wm_overlay"}:
        overlay = _lowfreq_wm_overlay(
            advantage_values=advantage_values,
            returns=returns,
            bc_cfg=bc_cfg,
            length=arr.shape[0],
        )
        if overlay is None:
            overlay = np.zeros(arr.shape[0], dtype=np.float32)
    elif mode in {"affine_wm", "affine_wm_overlay", "continuous_wm_overlay"}:
        overlay = _affine_wm_overlay(
            advantage_values=advantage_values,
            returns=returns,
            bc_cfg=bc_cfg,
            length=arr.shape[0],
        )
        if overlay is None:
            overlay = np.zeros(arr.shape[0], dtype=np.float32)
    elif mode in {"risk_budget", "wm_risk_budget"} and risk_gate is not None:
        base_overlay = float(bc_cfg.get("benchmark_overlay_base", 0.0))
        low_risk_up = float(bc_cfg.get("benchmark_overlay_low_risk_up", 0.0))
        high_risk_down = float(bc_cfg.get("benchmark_overlay_high_risk_down", 0.0))
        overlay = base_overlay + low_risk_up * (1.0 - risk_gate) - high_risk_down * risk_gate
    else:
        down_scale = float(bc_cfg.get("benchmark_overlay_down_scale", scale))
        up_scale = float(bc_cfg.get("benchmark_overlay_up_scale", scale))
        raw_overlay = arr - benchmark
        overlay = np.where(raw_overlay < 0.0, raw_overlay * down_scale, raw_overlay * up_scale)
        if risk_gate is not None:
            overlay = np.where(overlay < 0.0, overlay * risk_gate, overlay)

    edge_indices = tuple(int(i) for i in bc_cfg.get("benchmark_overlay_edge_protect_indices", []) or [])
    edge_gate = _edge_protect_gate(
        advantage_values,
        indices=edge_indices,
        length=arr.shape[0],
        center=float(bc_cfg.get("benchmark_overlay_edge_protect_center", 0.0)),
        scale=float(bc_cfg.get("benchmark_overlay_edge_protect_scale", 1.0)),
    )
    if edge_gate is not None:
        down_reduction = float(np.clip(bc_cfg.get("benchmark_overlay_edge_down_reduction", 0.0), 0.0, 1.0))
        up_add = float(bc_cfg.get("benchmark_overlay_edge_up_add", 0.0))
        if down_reduction > 0.0:
            overlay = np.where(overlay < 0.0, overlay * (1.0 - down_reduction * edge_gate), overlay)
        if up_add > 0.0:
            overlay = np.where(overlay >= 0.0, overlay + up_add * edge_gate, overlay)
    overlay = np.clip(overlay, min_overlay, max_overlay)

    ema_alpha = float(bc_cfg.get("benchmark_overlay_ema_alpha", 0.0))
    if 0.0 < ema_alpha < 1.0 and overlay.size > 1:
        smoothed = overlay.astype(np.float32).copy()
        prev = float(smoothed[0])
        for i in range(1, smoothed.shape[0]):
            prev = (1.0 - ema_alpha) * prev + ema_alpha * float(smoothed[i])
            smoothed[i] = prev
        overlay = smoothed

    hold_band = float(max(bc_cfg.get("benchmark_overlay_hold_band", 0.0), 0.0))
    if hold_band > 0.0 and overlay.size > 1:
        held = overlay.astype(np.float32).copy()
        prev = float(held[0])
        for i in range(1, held.shape[0]):
            raw = float(held[i])
            if abs(raw - prev) < hold_band:
                held[i] = prev
            else:
                prev = raw
        overlay = held

    deadzone = float(max(bc_cfg.get("benchmark_overlay_deadzone", 0.0), 0.0))
    if deadzone > 0.0:
        overlay = np.where(np.abs(overlay) < deadzone, 0.0, overlay)

    max_step = bc_cfg.get("benchmark_overlay_max_step")
    if max_step is None:
        max_step = ac_cfg.get("max_position_step")
    max_step = 0.0 if max_step is None else float(max_step)
    if max_step > 0.0 and overlay.size > 1:
        smoothed = overlay.astype(np.float32).copy()
        prev = 0.0
        for i in range(smoothed.shape[0]):
            raw = float(smoothed[i])
            prev = prev + float(np.clip(raw - prev, -max_step, max_step))
            smoothed[i] = prev
        overlay = smoothed

    return (benchmark + overlay).astype(np.float32)


def describe_benchmark_overlay_teacher(
    original_positions,
    mapped_positions,
    *,
    reward_cfg: dict | None,
) -> str:
    reward_cfg = reward_cfg or {}
    benchmark = float(reward_cfg.get("benchmark_position", 1.0))
    original = np.asarray(original_positions, dtype=np.float32) - benchmark
    mapped = np.asarray(mapped_positions, dtype=np.float32) - benchmark
    if mapped.size == 0:
        return "overlay teacher: empty"
    active = float(np.mean(np.abs(mapped) > 1e-6))
    active05 = float(np.mean(np.abs(mapped) > 0.05))
    changed = float(np.mean(np.abs(mapped - original[: mapped.shape[0]]) > 1e-6))
    return (
        "overlay teacher: "
        f"mean={float(mapped.mean()):+.4f} "
        f"p10={float(np.quantile(mapped, 0.10)):+.4f} "
        f"p50={float(np.quantile(mapped, 0.50)):+.4f} "
        f"p90={float(np.quantile(mapped, 0.90)):+.4f} "
        f"active={active:.1%} active05={active05:.1%} remapped={changed:.1%}"
    )
