from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualAdvantageBundle:
    target_positions: np.ndarray
    weights: np.ndarray
    inventory_states: np.ndarray
    summary: dict


def _current_positions_from_path(positions: np.ndarray, benchmark_position: float) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    current = np.empty_like(positions)
    if len(current) == 0:
        return current
    current[0] = float(benchmark_position)
    if len(current) > 1:
        current[1:] = positions[:-1]
    return current


def _rolling_sum(x: np.ndarray, horizon: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 0 or len(x) < horizon:
        return out
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out[: len(x) - horizon + 1] = csum[horizon:] - csum[:-horizon]
    return out


def _rolling_vol(x: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 1 or len(x) < horizon:
        return out
    sum_x = _rolling_sum(x, horizon)
    sum_x2 = _rolling_sum(np.asarray(x, dtype=np.float64) ** 2, horizon)
    mean = sum_x / horizon
    var = np.maximum(sum_x2 / horizon - mean * mean, 0.0)
    out[: len(x) - horizon + 1] = np.sqrt(var[: len(x) - horizon + 1]) * math.sqrt(horizon)
    return out


def _rolling_drawdown_dynamic(returns: np.ndarray, overlay: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    overlay = np.asarray(overlay, dtype=np.float64)
    out = np.full(overlay.shape, np.nan, dtype=np.float64)
    if horizon <= 0 or len(returns) < horizon:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(returns, horizon)
    n = windows.shape[0]
    path = overlay[:n, :, None] * windows[:, None, :]
    cums = np.cumsum(path, axis=2)
    peak = np.maximum.accumulate(np.maximum(cums, 0.0), axis=2)
    out[:n] = -np.min(cums - peak, axis=2)
    return out


def _unit_cost(costs_cfg: dict) -> float:
    return float(
        (float(costs_cfg.get("spread_bps", 3.0)) / 10000.0) / 2.0
        + float(costs_cfg.get("fee_rate", 0.0003))
        + float(costs_cfg.get("slippage_bps", 1.0)) / 10000.0
    )


def _candidate_matrix(
    *,
    anchor_positions: np.ndarray,
    current_positions: np.ndarray,
    cfg: dict,
    benchmark_position: float,
) -> tuple[np.ndarray, list[str]]:
    bc_cfg = cfg.get("bc", {})
    ac_cfg = cfg.get("ac", {})
    anchor = np.asarray(anchor_positions, dtype=np.float32)
    current = np.asarray(current_positions, dtype=np.float32)
    n = min(len(anchor), len(current))
    anchor = anchor[:n]
    current = current[:n]
    step = float(bc_cfg.get("residual_candidate_step", 0.05))
    overweight_actions = [float(x) for x in bc_cfg.get("residual_overweight_actions", [1.05, 1.10])]
    abs_min = float(ac_cfg.get("abs_min_position", 0.0))
    abs_max = float(ac_cfg.get("abs_max_position", 1.25))

    cols = [
        anchor,
        anchor - step,
        anchor + step,
        current,
        np.full(n, float(benchmark_position), dtype=np.float32),
    ]
    labels = ["anchor", f"minus_{step:.2f}", f"plus_{step:.2f}", "hold", "benchmark"]
    for action in overweight_actions:
        cols.append(np.full(n, action, dtype=np.float32))
        labels.append(f"ow_{action:.2f}")
    return np.clip(np.stack(cols, axis=1).astype(np.float32), abs_min, abs_max), labels


def _candidate_values(
    *,
    returns: np.ndarray,
    current_positions: np.ndarray,
    candidates: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    current = np.asarray(current_positions, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    n = min(len(returns), len(current), len(candidates))
    returns = returns[:n]
    current = current[:n]
    candidates = candidates[:n]
    bc_cfg = cfg.get("bc", {})
    horizons = tuple(int(h) for h in bc_cfg.get("transition_advantage_horizons", [4, 8, 16, 32]))
    raw_weights = bc_cfg.get("transition_advantage_horizon_weights")
    weights = np.asarray(raw_weights if raw_weights is not None else np.ones(len(horizons)), dtype=np.float64)
    if weights.size != len(horizons):
        weights = np.ones(len(horizons), dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)

    overlay = candidates - float(benchmark_position)
    trade_delta = np.abs(candidates - current[:, None])
    values_h = np.full((n, candidates.shape[1], len(horizons)), np.nan, dtype=np.float64)
    for hi, horizon in enumerate(horizons):
        sum_ret = _rolling_sum(returns, horizon)
        vol = _rolling_vol(returns, horizon)
        dd = _rolling_drawdown_dynamic(returns, overlay, horizon)
        valid = np.isfinite(sum_ret)
        value = overlay * sum_ret[:, None]
        value = value - _unit_cost(costs_cfg) * trade_delta
        value = value - float(bc_cfg.get("transition_volatility_penalty_coef", 0.10)) * np.nan_to_num(vol, nan=0.0)[:, None] * np.abs(overlay)
        value = value - float(bc_cfg.get("transition_drawdown_penalty_coef", 0.25)) * np.nan_to_num(dd, nan=0.0)
        value = value - float(bc_cfg.get("transition_leverage_penalty_coef", 0.0)) * np.maximum(candidates - benchmark_position, 0.0) * horizon
        value[~valid, :] = np.nan
        values_h[:, :, hi] = value

    finite = np.isfinite(values_h)
    weighted = np.where(finite, values_h, 0.0) * weights.reshape(1, 1, -1)
    denom = np.sum(finite * weights.reshape(1, 1, -1), axis=2)
    return np.divide(weighted.sum(axis=2), denom, out=np.full(candidates.shape, np.nan), where=denom > 0).astype(np.float32)


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0


def build_realized_candidate_advantage_targets(
    *,
    actor,
    z: np.ndarray,
    h: np.ndarray,
    returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    device: str,
    regime_probs: np.ndarray | None = None,
    advantage_values: np.ndarray | None = None,
) -> ResidualAdvantageBundle | None:
    bc_cfg = cfg.get("bc", {})
    coef = float(bc_cfg.get("realized_candidate_bc_coef", 0.0))
    if coef <= 0.0:
        return None

    n = min(len(z), len(h), len(returns))
    if regime_probs is not None:
        n = min(n, len(regime_probs))
    if advantage_values is not None:
        n = min(n, len(advantage_values))
    if n <= 0:
        return None

    reg = regime_probs[:n] if regime_probs is not None else None
    adv = advantage_values[:n] if advantage_values is not None else None
    anchor = actor.predict_positions(z[:n], h[:n], regime_np=reg, advantage_np=adv, device=device).astype(np.float32)
    current = _current_positions_from_path(anchor, benchmark_position).astype(np.float32)
    candidates, labels = _candidate_matrix(
        anchor_positions=anchor,
        current_positions=current,
        cfg=cfg,
        benchmark_position=benchmark_position,
    )
    values = _candidate_values(
        returns=np.asarray(returns[:n], dtype=np.float32),
        current_positions=current,
        candidates=candidates,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )

    anchor_values = values[:, 0]
    advantage = values - anchor_values[:, None]
    finite = np.isfinite(advantage)
    margin = float(bc_cfg.get("realized_candidate_margin", bc_cfg.get("transition_advantage_margin", 0.0)))
    valid = finite & (advantage > margin)
    valid[:, 0] = False
    active = valid.any(axis=1)

    tau = max(float(bc_cfg.get("realized_candidate_tau", 0.002)), 1e-8)
    score = np.where(valid, np.clip((advantage - margin) / tau, -60.0, 60.0), -np.inf)
    row_has_valid = np.isfinite(score).any(axis=1, keepdims=True)
    row_max = np.where(row_has_valid, np.max(score, axis=1, keepdims=True), 0.0)
    exp_score = np.where(row_has_valid, np.exp(score - row_max), 0.0)
    exp_score[~np.isfinite(exp_score)] = 0.0
    denom = exp_score.sum(axis=1, keepdims=True)
    soft = np.divide(exp_score, np.clip(denom, 1e-12, None), out=np.zeros_like(exp_score), where=denom > 0.0)

    mode = str(bc_cfg.get("realized_candidate_mode", "awr")).lower()
    best_idx = np.argmax(np.where(finite, advantage, -np.inf), axis=1)
    if mode == "hard":
        target = anchor.copy()
        target[active] = candidates[np.arange(n), best_idx][active]
    else:
        soft_target = (soft * candidates).sum(axis=1)
        anchor_mix = float(np.clip(bc_cfg.get("realized_candidate_anchor_mix", 0.65), 0.0, 1.0))
        target = np.where(active, anchor_mix * anchor + (1.0 - anchor_mix) * soft_target, anchor)

    max_delta = float(bc_cfg.get("realized_candidate_max_delta", 0.05))
    if max_delta > 0.0:
        target = anchor + np.clip(target - anchor, -max_delta, max_delta)
    abs_min = float(cfg.get("ac", {}).get("abs_min_position", 0.0))
    abs_max = float(cfg.get("ac", {}).get("abs_max_position", 1.25))
    target = np.clip(target, abs_min, abs_max).astype(np.float32)

    best_improvement = np.zeros(n, dtype=np.float32)
    if active.any():
        best_improvement[active] = np.max(np.where(valid[active], advantage[active], -np.inf), axis=1)
    weight_max = max(float(bc_cfg.get("realized_candidate_weight_max", 3.0)), 1.0)
    weights = np.where(
        active,
        np.minimum(np.exp(np.clip(best_improvement / tau, 0.0, math.log(weight_max))), weight_max),
        0.0,
    ).astype(np.float32)

    summary = {
        "mode": mode,
        "active_rate": _rate(active),
        "mean_best_improvement": float(best_improvement[active].mean()) if active.any() else 0.0,
        "anchor_short": _rate(anchor < benchmark_position - 1e-6),
        "anchor_flat": _rate(np.abs(anchor - benchmark_position) <= 1e-6),
        "anchor_long": _rate(anchor > benchmark_position + 1e-6),
        "target_short": _rate(target < benchmark_position - 1e-6),
        "target_flat": _rate(np.abs(target - benchmark_position) <= 1e-6),
        "target_long": _rate(target > benchmark_position + 1e-6),
        "candidate_labels": labels,
        "active_distribution": {labels[i]: _rate(active & (best_idx == i)) for i in range(len(labels))},
    }
    return ResidualAdvantageBundle(
        target_positions=target,
        weights=weights,
        inventory_states=actor.controller_states_from_positions(anchor).astype(np.float32),
        summary=summary,
    )
