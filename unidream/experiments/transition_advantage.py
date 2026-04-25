from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


ROUTE_NAMES: tuple[str, ...] = ("neutral", "de_risk", "recovery", "overweight")
ROUTE_TO_ID = {name: idx for idx, name in enumerate(ROUTE_NAMES)}


@dataclass(frozen=True)
class TransitionAdvantageConfig:
    horizons: tuple[int, ...] = (4, 8, 16, 32)
    horizon_weights: tuple[float, ...] | None = None
    margin: float = 0.0
    drawdown_penalty_coef: float = 0.25
    volatility_penalty_coef: float = 0.10
    turnover_penalty_coef: float = 0.0
    leverage_penalty_coef: float = 0.0
    short_penalty_coef: float = 0.0
    candidate_actions: tuple[float, ...] = (0.0, 0.5, 1.0, 1.25)
    benchmark_position: float = 1.0
    spread_bps: float = 3.0
    fee_rate: float = 0.0003
    slippage_bps: float = 1.0


def transition_unit_cost(spread_bps: float, fee_rate: float, slippage_bps: float) -> float:
    return float((spread_bps / 10000.0) / 2.0 + fee_rate + slippage_bps / 10000.0)


def config_from_dict(
    cfg: dict,
    *,
    costs_cfg: dict,
    benchmark_position: float,
    default_actions,
) -> TransitionAdvantageConfig:
    horizons = tuple(int(h) for h in cfg.get("transition_advantage_horizons", [4, 8, 16, 32]))
    weights = cfg.get("transition_advantage_horizon_weights")
    horizon_weights = tuple(float(w) for w in weights) if weights is not None else None
    actions = cfg.get("transition_candidate_actions", default_actions)
    return TransitionAdvantageConfig(
        horizons=horizons,
        horizon_weights=horizon_weights,
        margin=float(cfg.get("transition_advantage_margin", 0.0)),
        drawdown_penalty_coef=float(cfg.get("transition_drawdown_penalty_coef", 0.25)),
        volatility_penalty_coef=float(cfg.get("transition_volatility_penalty_coef", 0.10)),
        turnover_penalty_coef=float(cfg.get("transition_turnover_penalty_coef", 0.0)),
        leverage_penalty_coef=float(cfg.get("transition_leverage_penalty_coef", 0.0)),
        short_penalty_coef=float(cfg.get("transition_short_penalty_coef", 0.0)),
        candidate_actions=tuple(float(a) for a in actions),
        benchmark_position=float(benchmark_position),
        spread_bps=float(costs_cfg.get("spread_bps", 3.0)),
        fee_rate=float(costs_cfg.get("fee_rate", 0.0003)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 1.0)),
    )


def _rolling_sum(x: np.ndarray, horizon: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 0 or len(x) < horizon:
        return out
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out[: len(x) - horizon + 1] = csum[horizon:] - csum[:-horizon]
    return out


def _rolling_vol(x: np.ndarray, horizon: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 1 or len(x) < horizon:
        return out
    sum_x = _rolling_sum(x, horizon)
    sum_x2 = _rolling_sum(x * x, horizon)
    mean = sum_x / horizon
    var = np.maximum(sum_x2 / horizon - mean * mean, 0.0)
    out[: len(x) - horizon + 1] = np.sqrt(var[: len(x) - horizon + 1]) * math.sqrt(horizon)
    return out


def _rolling_drawdown(x: np.ndarray, horizon: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 0 or len(x) < horizon:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, horizon)
    cums = np.cumsum(windows, axis=1)
    running_peak = np.maximum.accumulate(np.maximum(cums, 0.0), axis=1)
    drawdowns = cums - running_peak
    out[: windows.shape[0]] = -np.min(drawdowns, axis=1)
    return out


def transition_classes(
    current_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    benchmark_position: float,
    eps: float = 1e-6,
) -> np.ndarray:
    current = np.asarray(current_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    cls = np.full(target.shape, "neutral", dtype=object)
    under_now = current < benchmark_position - eps
    cls[target < benchmark_position - eps] = "de_risk"
    cls[(under_now) & (target < benchmark_position - eps)] = "stay_underweight"
    cls[(under_now) & (target >= benchmark_position - eps) & (target <= benchmark_position + eps)] = "recovery"
    cls[target > benchmark_position + eps] = "overweight"
    benchmark_target = np.abs(target - benchmark_position) <= eps
    cls[benchmark_target & under_now] = "recovery"
    cls[benchmark_target & (~under_now)] = "neutral"
    return cls


def route_ids_from_classes(classes: np.ndarray) -> np.ndarray:
    """Map fine-grained transition classes to the four routing heads."""
    arr = np.asarray(classes, dtype=object)
    route = np.zeros(arr.shape, dtype=np.int64)
    route[(arr == "de_risk") | (arr == "stay_underweight")] = ROUTE_TO_ID["de_risk"]
    route[arr == "recovery"] = ROUTE_TO_ID["recovery"]
    route[arr == "overweight"] = ROUTE_TO_ID["overweight"]
    return route


def compute_route_targets(
    bundle: dict,
    *,
    tau: float = 0.001,
    label_smoothing: float = 0.05,
    margin: float = 0.0,
) -> dict:
    """Build no-trade-aware soft route labels from transition advantages.

    The route score is the best non-negative cost-adjusted advantage available
    to each route.  Non-neutral routes whose advantage fails the margin are
    masked out, which turns small/noisy edge into a neutral no-trade target.
    """
    class_matrix = np.asarray(bundle["class_matrix"], dtype=object)
    adv_neutral = np.asarray(bundle["advantage_vs_neutral"], dtype=np.float64)
    adv_current = np.asarray(bundle["advantage_vs_current"], dtype=np.float64)
    valid = np.isfinite(adv_neutral) | np.isfinite(adv_current)
    route_ids = route_ids_from_classes(class_matrix)

    raw_score = np.maximum(
        np.nan_to_num(adv_neutral, nan=-np.inf),
        np.nan_to_num(adv_current, nan=-np.inf),
    )
    raw_score = np.where(valid, np.maximum(raw_score, 0.0), -np.inf)

    T = raw_score.shape[0]
    n_routes = len(ROUTE_NAMES)
    route_scores = np.full((T, n_routes), -np.inf, dtype=np.float64)
    route_action_idx = np.full((T, n_routes), -1, dtype=np.int64)
    for route_id in range(n_routes):
        masked = np.where(route_ids == route_id, raw_score, -np.inf)
        route_action_idx[:, route_id] = np.argmax(masked, axis=1)
        route_scores[:, route_id] = masked[np.arange(T), route_action_idx[:, route_id]]

    # Neutral is the benchmark/no-trade fallback.  It must remain available even
    # when all forward advantage estimates are invalid near the right edge.
    route_scores[:, ROUTE_TO_ID["neutral"]] = 0.0

    non_neutral_scores = route_scores[:, 1:]
    best_non_idx = np.argmax(non_neutral_scores, axis=1) + 1
    best_non_score = non_neutral_scores[np.arange(T), best_non_idx - 1]
    route_labels = np.where(best_non_score > float(margin), best_non_idx, ROUTE_TO_ID["neutral"]).astype(np.int64)
    route_advantage = np.where(route_labels == ROUTE_TO_ID["neutral"], 0.0, best_non_score).astype(np.float32)

    gated_scores = route_scores.copy()
    gated_scores[:, 1:] = np.where(gated_scores[:, 1:] > float(margin), gated_scores[:, 1:], -np.inf)
    gated_scores[:, ROUTE_TO_ID["neutral"]] = 0.0
    scale = max(float(tau), 1e-8)
    scaled = np.clip(gated_scores / scale, -60.0, 60.0)
    row_max = np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scaled - row_max)
    exp_scores[~np.isfinite(exp_scores)] = 0.0
    denom = exp_scores.sum(axis=1, keepdims=True)
    soft_labels = np.divide(
        exp_scores,
        np.clip(denom, 1e-12, None),
        out=np.zeros_like(exp_scores),
        where=denom > 0.0,
    )
    empty = denom.squeeze(-1) <= 0.0
    if empty.any():
        soft_labels[empty, ROUTE_TO_ID["neutral"]] = 1.0
    smoothing = float(np.clip(label_smoothing, 0.0, 0.99))
    if smoothing > 0.0:
        soft_labels = (1.0 - smoothing) * soft_labels + smoothing / n_routes

    return {
        "route_names": ROUTE_NAMES,
        "route_scores": route_scores.astype(np.float32),
        "route_labels": route_labels,
        "route_soft_labels": soft_labels.astype(np.float32),
        "route_advantage": route_advantage,
        "route_action_idx": route_action_idx,
    }


def summarize_route_targets(route_bundle: dict) -> dict:
    labels = np.asarray(route_bundle["route_labels"], dtype=np.int64)
    scores = np.asarray(route_bundle["route_scores"], dtype=np.float64)
    adv = np.asarray(route_bundle["route_advantage"], dtype=np.float64)
    rows = []
    for idx, name in enumerate(route_bundle.get("route_names", ROUTE_NAMES)):
        mask = labels == idx
        vals = adv[mask]
        rows.append(
            {
                "route": name,
                "count": int(mask.sum()),
                "rate": float(mask.mean()) if len(mask) else 0.0,
                "mean_adv": float(vals.mean()) if vals.size else 0.0,
                "top_decile_adv": float(vals[vals >= np.quantile(vals, 0.90)].mean()) if vals.size else 0.0,
                "mean_score": float(np.nanmean(scores[:, idx])) if scores.size else 0.0,
            }
        )
    non_neutral = labels != ROUTE_TO_ID["neutral"]
    return {
        "routes": rows,
        "active_rate": float(non_neutral.mean()) if len(non_neutral) else 0.0,
        "mean_route_advantage": float(np.mean(adv)) if len(adv) else 0.0,
        "top_decile_route_advantage": float(adv[adv >= np.quantile(adv, 0.90)].mean()) if len(adv) else 0.0,
    }


def compute_transition_advantage(
    returns: np.ndarray,
    current_positions: np.ndarray,
    cfg: TransitionAdvantageConfig,
) -> dict:
    returns = np.asarray(returns, dtype=np.float64)
    current = np.asarray(current_positions, dtype=np.float64)
    T = min(len(returns), len(current))
    returns = returns[:T]
    current = current[:T]
    actions = np.asarray(cfg.candidate_actions, dtype=np.float64)
    n_actions = len(actions)
    n_h = len(cfg.horizons)
    weights = np.asarray(
        cfg.horizon_weights if cfg.horizon_weights is not None else np.ones(n_h, dtype=np.float64),
        dtype=np.float64,
    )
    if weights.size != n_h:
        weights = np.ones(n_h, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)

    values_h = np.full((T, n_actions, n_h), np.nan, dtype=np.float64)
    unit_cost = transition_unit_cost(cfg.spread_bps, cfg.fee_rate, cfg.slippage_bps)
    for hi, horizon in enumerate(cfg.horizons):
        sum_ret = _rolling_sum(returns, horizon)
        vol = _rolling_vol(returns, horizon)
        valid = np.isfinite(sum_ret)
        for ai, pos in enumerate(actions):
            overlay = pos - cfg.benchmark_position
            excess_path = overlay * returns
            dd = _rolling_drawdown(excess_path, horizon)
            trade_delta = np.abs(pos - current)
            value = overlay * sum_ret
            value = value - unit_cost * trade_delta
            value = value - cfg.turnover_penalty_coef * trade_delta
            value = value - cfg.volatility_penalty_coef * np.nan_to_num(vol, nan=0.0) * abs(overlay)
            value = value - cfg.drawdown_penalty_coef * np.nan_to_num(dd, nan=0.0)
            value = value - cfg.leverage_penalty_coef * max(pos - cfg.benchmark_position, 0.0) * horizon
            value = value - cfg.short_penalty_coef * max(-pos, 0.0) * horizon
            value[~valid] = np.nan
            values_h[:, ai, hi] = value

    finite = np.isfinite(values_h)
    weighted = np.where(finite, values_h, 0.0) * weights.reshape(1, 1, -1)
    denom = np.sum(finite * weights.reshape(1, 1, -1), axis=2)
    values = np.divide(weighted.sum(axis=2), denom, out=np.full((T, n_actions), np.nan), where=denom > 0)
    neutral_idx = int(np.argmin(np.abs(actions - cfg.benchmark_position)))
    neutral_values = values[:, neutral_idx]
    advantage_vs_neutral = values - neutral_values[:, None]
    current_idx = np.argmin(np.abs(actions.reshape(1, -1) - current.reshape(-1, 1)), axis=1)
    current_values = values[np.arange(T), current_idx]
    advantage_vs_current = values - current_values[:, None]
    best_idx = np.nanargmax(np.where(np.isfinite(values), values, -np.inf), axis=1)
    best_value = values[np.arange(T), best_idx]
    best_adv_neutral = advantage_vs_neutral[np.arange(T), best_idx]
    best_adv_current = advantage_vs_current[np.arange(T), best_idx]
    best_adv = np.maximum(best_adv_neutral, best_adv_current)
    best_adv = np.maximum(best_adv, 0.0)
    target_positions = actions[best_idx].astype(np.float32)
    neutral_mask = (~np.isfinite(best_value)) | (best_adv <= cfg.margin)
    target_positions[neutral_mask] = np.float32(cfg.benchmark_position)
    best_idx[neutral_mask] = neutral_idx
    best_value[neutral_mask] = neutral_values[neutral_mask]
    best_adv[neutral_mask] = 0.0
    best_adv_neutral[neutral_mask] = 0.0
    best_adv_current[neutral_mask] = 0.0
    classes = transition_classes(
        np.repeat(current[:, None], n_actions, axis=1),
        np.repeat(actions[None, :], T, axis=0),
        benchmark_position=cfg.benchmark_position,
    )
    best_class = classes[np.arange(T), best_idx]
    return {
        "actions": actions,
        "horizons": np.asarray(cfg.horizons, dtype=np.int64),
        "values": values.astype(np.float32),
        "values_h": values_h.astype(np.float32),
        "advantage_vs_neutral": advantage_vs_neutral.astype(np.float32),
        "advantage_vs_current": advantage_vs_current.astype(np.float32),
        "target_positions": target_positions,
        "best_idx": best_idx.astype(np.int64),
        "best_value": best_value.astype(np.float32),
        "best_advantage": best_adv.astype(np.float32),
        "best_advantage_vs_neutral": best_adv_neutral.astype(np.float32),
        "best_advantage_vs_current": best_adv_current.astype(np.float32),
        "best_class": best_class,
        "class_matrix": classes,
    }


def current_positions_from_path(positions: np.ndarray, benchmark_position: float) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    current = np.empty_like(positions)
    if len(current) == 0:
        return current
    current[0] = benchmark_position
    if len(current) > 1:
        current[1:] = positions[:-1]
    return current


def summarize_transition_advantage(bundle: dict, current_positions: np.ndarray, benchmark_position: float) -> dict:
    actions = bundle["actions"]
    values = bundle["values"]
    best_class = np.asarray(bundle["best_class"], dtype=object)
    best_adv = np.asarray(bundle["best_advantage"], dtype=np.float64)
    target = np.asarray(bundle["target_positions"], dtype=np.float64)
    current = np.asarray(current_positions[: len(target)], dtype=np.float64)
    valid = np.isfinite(best_adv)

    action_rows = []
    for ai, action in enumerate(actions):
        arr = values[:, ai]
        mask = np.isfinite(arr)
        if not mask.any():
            continue
        vals = arr[mask].astype(np.float64)
        q90 = float(np.quantile(vals, 0.90))
        q10 = float(np.quantile(vals, 0.10))
        action_rows.append(
            {
                "action": float(action),
                "count": int(mask.sum()),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "top_decile_mean": float(vals[vals >= q90].mean()),
                "bottom_decile_mean": float(vals[vals <= q10].mean()),
                "positive_rate": float(np.mean(vals > 0.0)),
            }
        )

    class_rows = []
    for cls in ["neutral", "de_risk", "stay_underweight", "recovery", "overweight"]:
        mask = valid & (best_class == cls)
        if not mask.any():
            continue
        vals = best_adv[mask]
        q90 = float(np.quantile(vals, 0.90))
        q10 = float(np.quantile(vals, 0.10))
        class_rows.append(
            {
                "class": cls,
                "count": int(mask.sum()),
                "rate": float(mask.mean()),
                "mean_adv": float(vals.mean()),
                "median_adv": float(np.median(vals)),
                "top_decile_mean": float(vals[vals >= q90].mean()),
                "bottom_decile_mean": float(vals[vals <= q10].mean()),
                "positive_rate": float(np.mean(vals > 0.0)),
            }
        )

    bucket_names = ["underweight", "benchmark", "overweight"]
    def bucket(x):
        out = np.full(len(x), 1, dtype=np.int64)
        out[x < benchmark_position - 1e-6] = 0
        out[x > benchmark_position + 1e-6] = 2
        return out

    cur_b = bucket(current)
    tgt_b = bucket(target)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (cur_b, tgt_b), 1)

    recovery_mask = current < benchmark_position - 1e-6
    recovery_rate = float(np.mean(target[recovery_mask] >= benchmark_position - 1e-6)) if recovery_mask.any() else float("nan")
    return {
        "actions": action_rows,
        "classes": class_rows,
        "best_distribution": {
            cls: float(np.mean(best_class == cls))
            for cls in ["neutral", "de_risk", "stay_underweight", "recovery", "overweight"]
        },
        "transition_matrix": {
            bucket_names[i]: {bucket_names[j]: int(matrix[i, j]) for j in range(3)}
            for i in range(3)
        },
        "recovery_rate_from_underweight": recovery_rate,
        "target_short_rate": float(np.mean(target < benchmark_position - 1e-6)),
        "target_benchmark_rate": float(np.mean(np.abs(target - benchmark_position) <= 1e-6)),
        "target_overweight_rate": float(np.mean(target > benchmark_position + 1e-6)),
        "mean_best_advantage": float(np.nanmean(best_adv)),
    }


def recovery_latency(positions: np.ndarray, benchmark_position: float) -> dict:
    positions = np.asarray(positions, dtype=np.float64)
    latencies = []
    t = 0
    while t < len(positions):
        if positions[t] >= benchmark_position - 1e-6:
            t += 1
            continue
        start = t
        while t < len(positions) and positions[t] < benchmark_position - 1e-6:
            t += 1
        if t < len(positions):
            latencies.append(t - start)
    if not latencies:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
    }
