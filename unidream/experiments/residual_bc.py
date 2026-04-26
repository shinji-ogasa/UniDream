from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualBCBundle:
    target_positions: np.ndarray
    weights: np.ndarray
    inventory_states: np.ndarray
    anchor_positions: np.ndarray
    candidate_positions: np.ndarray
    candidate_values: np.ndarray
    candidate_labels: list[str]
    best_idx: np.ndarray
    improvement: np.ndarray
    summary: dict


def current_positions_from_path(positions: np.ndarray, benchmark_position: float) -> np.ndarray:
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


def _transition_unit_cost(costs_cfg: dict) -> float:
    return float(
        (float(costs_cfg.get("spread_bps", 3.0)) / 10000.0) / 2.0
        + float(costs_cfg.get("fee_rate", 0.0003))
        + float(costs_cfg.get("slippage_bps", 1.0)) / 10000.0
    )


def build_dynamic_residual_candidates(
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

    abs_min = float(ac_cfg.get("abs_min_position", 0.0))
    abs_max = float(ac_cfg.get("abs_max_position", 1.25))
    step = float(bc_cfg.get("residual_adv_bc_step", 0.05))
    overweight_actions = [float(x) for x in bc_cfg.get("residual_adv_bc_overweight_actions", [1.05, 1.10])]

    cols = [
        anchor,
        anchor - step,
        anchor + step,
        current,
        np.full(n, float(benchmark_position), dtype=np.float32),
    ]
    labels = ["bc", f"bc_minus_{step:.2f}", f"bc_plus_{step:.2f}", "hold_current", "benchmark"]
    for action in overweight_actions:
        cols.append(np.full(n, action, dtype=np.float32))
        labels.append(f"ow_{action:.2f}")

    matrix = np.stack(cols, axis=1).astype(np.float32)
    return np.clip(matrix, abs_min, abs_max), labels


def compute_dynamic_candidate_values(
    *,
    returns: np.ndarray,
    current_positions: np.ndarray,
    candidate_positions: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    current = np.asarray(current_positions, dtype=np.float64)
    candidates = np.asarray(candidate_positions, dtype=np.float64)
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
    unit_cost = _transition_unit_cost(costs_cfg)
    values_h = np.full((n, candidates.shape[1], len(horizons)), np.nan, dtype=np.float64)
    for hi, horizon in enumerate(horizons):
        sum_ret = _rolling_sum(returns, horizon)
        vol = _rolling_vol(returns, horizon)
        dd = _rolling_drawdown_dynamic(returns, overlay, horizon)
        valid = np.isfinite(sum_ret)
        value = overlay * sum_ret[:, None]
        value = value - unit_cost * trade_delta
        value = value - float(bc_cfg.get("transition_turnover_penalty_coef", 0.0)) * trade_delta
        value = value - float(bc_cfg.get("transition_volatility_penalty_coef", 0.10)) * np.nan_to_num(vol, nan=0.0)[:, None] * np.abs(overlay)
        value = value - float(bc_cfg.get("transition_drawdown_penalty_coef", 0.25)) * np.nan_to_num(dd, nan=0.0)
        value = value - float(bc_cfg.get("transition_leverage_penalty_coef", 0.0)) * np.maximum(candidates - benchmark_position, 0.0) * horizon
        value = value - float(bc_cfg.get("transition_short_penalty_coef", 0.0)) * np.maximum(-candidates, 0.0) * horizon
        value[~valid, :] = np.nan
        values_h[:, :, hi] = value

    finite = np.isfinite(values_h)
    weighted = np.where(finite, values_h, 0.0) * weights.reshape(1, 1, -1)
    denom = np.sum(finite * weights.reshape(1, 1, -1), axis=2)
    return np.divide(weighted.sum(axis=2), denom, out=np.full(candidates.shape, np.nan), where=denom > 0).astype(np.float32)


def _safe_rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0


def build_realized_advantage_residual_bc(
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
) -> ResidualBCBundle | None:
    bc_cfg = cfg.get("bc", {})
    if float(bc_cfg.get("residual_adv_bc_coef", 0.0)) <= 0.0:
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
    anchor_positions = actor.predict_positions(
        z[:n],
        h[:n],
        regime_np=reg,
        advantage_np=adv,
        device=device,
    ).astype(np.float32)
    current = current_positions_from_path(anchor_positions, benchmark_position).astype(np.float32)
    candidates, labels = build_dynamic_residual_candidates(
        anchor_positions=anchor_positions,
        current_positions=current,
        cfg=cfg,
        benchmark_position=benchmark_position,
    )
    values = compute_dynamic_candidate_values(
        returns=np.asarray(returns[:n], dtype=np.float32),
        current_positions=current,
        candidate_positions=candidates,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )

    finite = np.isfinite(values)
    safe_values = np.where(finite, values, -np.inf)
    best_idx = np.argmax(safe_values, axis=1).astype(np.int64)
    best_values = safe_values[np.arange(n), best_idx]
    anchor_values = values[:, 0]
    improvement = best_values - anchor_values
    valid = np.isfinite(anchor_values) & np.isfinite(best_values)

    margin = float(bc_cfg.get("residual_adv_bc_margin", bc_cfg.get("transition_advantage_margin", 0.0)))
    max_delta = float(bc_cfg.get("residual_adv_bc_max_delta", 0.0))
    target_positions = anchor_positions.copy()
    active = valid & (improvement > margin) & (best_idx != 0)
    if max_delta > 0.0:
        active = active & (np.abs(candidates[np.arange(n), best_idx] - anchor_positions) <= max_delta)
    target_positions[active] = candidates[np.arange(n), best_idx][active]

    tau = max(float(bc_cfg.get("residual_adv_bc_tau", 0.002)), 1e-8)
    weight_max = max(float(bc_cfg.get("residual_adv_bc_weight_max", 5.0)), 1.0)
    raw_weight = np.exp(np.clip((improvement - margin) / tau, 0.0, math.log(weight_max)))
    weights = np.where(active, np.minimum(raw_weight, weight_max), 0.0).astype(np.float32)

    inventory_states = actor.controller_states_from_positions(anchor_positions).astype(np.float32)
    best_targets = candidates[np.arange(n), best_idx]
    summary = {
        "active_rate": _safe_rate(active),
        "mean_improvement_active": float(np.nanmean(improvement[active])) if active.any() else 0.0,
        "mean_weight_active": float(weights[active].mean()) if active.any() else 0.0,
        "short_target_rate": _safe_rate(target_positions < benchmark_position - 1e-6),
        "flat_target_rate": _safe_rate(np.abs(target_positions - benchmark_position) <= 1e-6),
        "long_target_rate": _safe_rate(target_positions > benchmark_position + 1e-6),
        "anchor_short_rate": _safe_rate(anchor_positions < benchmark_position - 1e-6),
        "anchor_flat_rate": _safe_rate(np.abs(anchor_positions - benchmark_position) <= 1e-6),
        "anchor_long_rate": _safe_rate(anchor_positions > benchmark_position + 1e-6),
        "candidate_labels": labels,
        "best_distribution": {
            labels[i]: _safe_rate(best_idx == i)
            for i in range(len(labels))
        },
        "active_distribution": {
            labels[i]: _safe_rate(active & (best_idx == i))
            for i in range(len(labels))
        },
        "mean_best_target": float(np.nanmean(best_targets)) if len(best_targets) else 0.0,
    }
    return ResidualBCBundle(
        target_positions=target_positions.astype(np.float32),
        weights=weights,
        inventory_states=inventory_states,
        anchor_positions=anchor_positions,
        candidate_positions=candidates.astype(np.float32),
        candidate_values=values.astype(np.float32),
        candidate_labels=labels,
        best_idx=best_idx,
        improvement=improvement.astype(np.float32),
        summary=summary,
    )
