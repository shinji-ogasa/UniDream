from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.cli.market_event_label_probe import _path_max_drawdown
from unidream.cli.route_separability_probe import (
    _binary_eval,
    _fit_binary_model,
    _score_binary,
    _select_threshold,
)
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest, compute_pnl
from unidream.eval.pbo import compute_pbo
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    lane: str
    candidates: tuple[float, ...]
    horizon: int
    dd_penalty: float
    vol_penalty: float
    active_cap: float
    maxdd_cap_pt: float
    turnover_cap: float
    min_threshold: float
    mode: str
    cooldown_bars: int = 0
    hold_bars: int = 1
    guard_horizon: int = 32
    guard_vol_window: int = 64
    guard_barrier_k: float = 1.25
    cooldown_grid: tuple[int, ...] = ()
    min_val_maxdd_improvement_pt: float = 0.0
    turnover_score_coef: float = 0.05
    active_score_coef: float = 0.25


SELECTOR_SPECS = (
    SelectorSpec(
        name="B_safe_small",
        lane="B_safe_improvement",
        candidates=(0.75, 1.0, 1.05),
        horizon=32,
        dd_penalty=0.80,
        vol_penalty=0.10,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="plain",
    ),
    SelectorSpec(
        name="D_risk_sensitive",
        lane="D_risk_sensitive",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="plain",
    ),
    SelectorSpec(
        name="D_risk_sensitive_floor005",
        lane="D_risk_sensitive",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.005,
        mode="plain",
    ),
    SelectorSpec(
        name="D_risk_sensitive_floor010",
        lane="D_risk_sensitive",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.010,
        mode="plain",
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="tb_guard",
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_floor005",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.005,
        mode="tb_guard",
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_cd32",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="tb_guard",
        cooldown_bars=32,
        hold_bars=1,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_cd32_floor001",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard",
        cooldown_bars=32,
        hold_bars=1,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_cd32_floor0025",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0025,
        mode="tb_guard",
        cooldown_bars=32,
        hold_bars=1,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_cd96",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="tb_guard",
        cooldown_bars=96,
        hold_bars=1,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_h64",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="tb_guard",
        guard_horizon=64,
        guard_vol_window=128,
        guard_barrier_k=1.50,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="tb_guard",
        cooldown_grid=(0, 32),
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd_floor001",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard",
        cooldown_grid=(0, 32),
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd_floor001_valdd",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard",
        cooldown_grid=(0, 32),
        min_val_maxdd_improvement_pt=0.001,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd_floor001_pullback",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback",
        cooldown_grid=(0, 32),
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback_evalonly",
        cooldown_grid=(0, 32),
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly_tpen025",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback_evalonly",
        cooldown_grid=(0, 32),
        turnover_score_coef=0.25,
    ),
    SelectorSpec(
        name="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly_tpen050",
        lane="D_risk_sensitive_plus_A_guard",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.15,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.001,
        mode="tb_guard_pullback_evalonly",
        cooldown_grid=(0, 32),
        turnover_score_coef=0.50,
    ),
    SelectorSpec(
        name="F_listwise",
        lane="F_pairwise_listwise",
        candidates=(0.50, 0.75, 1.0, 1.05, 1.10, 1.25),
        horizon=32,
        dd_penalty=0.50,
        vol_penalty=0.05,
        active_cap=0.35,
        maxdd_cap_pt=0.25,
        turnover_cap=4.00,
        min_threshold=0.0,
        mode="plain",
    ),
    SelectorSpec(
        name="E_bootstrap_uncertainty",
        lane="E_model_uncertainty",
        candidates=(0.75, 1.0, 1.05, 1.10),
        horizon=32,
        dd_penalty=1.00,
        vol_penalty=0.10,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="uncertainty",
    ),
    SelectorSpec(
        name="G_vol_regime_safe",
        lane="G_regime_split",
        candidates=(0.75, 1.0, 1.05),
        horizon=32,
        dd_penalty=0.80,
        vol_penalty=0.10,
        active_cap=0.25,
        maxdd_cap_pt=0.00,
        turnover_cap=3.50,
        min_threshold=0.0,
        mode="regime",
    ),
)


def _unit_cost(costs_cfg: dict) -> float:
    return (
        float(costs_cfg.get("spread_bps", 3.0)) / 10000.0 / 2.0
        + float(costs_cfg.get("fee_rate", 0.0003))
        + float(costs_cfg.get("slippage_bps", 1.0)) / 10000.0
    )


def _rolling_past_sum(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    for i in range(len(arr)):
        start = max(0, i - int(window))
        out[i] = csum[i] - csum[start]
    return out


def _rolling_past_vol(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i - int(window))
        vals = arr[start:i]
        out[i] = float(np.std(vals)) if len(vals) >= 2 else 0.0
    return out


def _online_drawdown(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    equity = np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    underwater = peak - equity
    return dd, underwater


def _state_features(raw: np.ndarray, returns: np.ndarray) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    dd, underwater = _online_drawdown(ret)
    parts = [raw_arr]
    for window in (4, 16, 32, 64, 128):
        parts.append(_rolling_past_sum(ret, window).reshape(-1, 1))
        parts.append(_rolling_past_vol(ret, window).reshape(-1, 1))
    parts.extend([dd.reshape(-1, 1), underwater.reshape(-1, 1)])
    x = np.concatenate([p[: len(ret)] for p in parts], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _future_windows(returns: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(returns, dtype=np.float64)
    h = int(max(horizon, 1))
    valid = np.zeros(len(r), dtype=bool)
    if len(r) <= h:
        return np.empty((0, h), dtype=np.float64), valid
    windows = np.lib.stride_tricks.sliding_window_view(r[1:], h)
    valid_n = len(r) - h
    valid[:valid_n] = True
    return windows[:valid_n].copy(), valid


def _future_vol(windows: np.ndarray) -> np.ndarray:
    if len(windows) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.std(windows, axis=1) * np.sqrt(windows.shape[1])


def _candidate_utilities(
    returns: np.ndarray,
    *,
    candidates: tuple[float, ...],
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    dd_penalty: float,
    vol_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    windows, valid = _future_windows(returns, horizon)
    n = len(returns)
    k = len(candidates)
    values = np.full((n, k), np.nan, dtype=np.float64)
    if len(windows) == 0:
        return values, valid
    valid_n = windows.shape[0]
    candidates_arr = np.asarray(candidates, dtype=np.float64)
    bench = float(benchmark_position)
    future_sum = np.sum(windows, axis=1)
    future_vol = _future_vol(windows)
    bench_dd = _path_max_drawdown(windows, bench)
    for ci, pos in enumerate(candidates_arr):
        dd = _path_max_drawdown(windows, float(pos))
        dd_worsen = np.maximum(dd - bench_dd, 0.0)
        overlay = float(pos - bench)
        trade_cost = abs(overlay) * unit_cost
        values[:valid_n, ci] = (
            overlay * future_sum
            - trade_cost
            - float(dd_penalty) * dd_worsen
            - float(vol_penalty) * abs(overlay) * future_vol
        )
    return values, valid


def _finite_rows(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x_mask = np.all(np.isfinite(np.asarray(x, dtype=np.float64)), axis=1)
    if y is None:
        return x_mask
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim == 1:
        y_mask = np.isfinite(yy)
    else:
        y_mask = np.all(np.isfinite(yy), axis=1)
    return x_mask & y_mask


@dataclass
class RidgeModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        xx = np.concatenate([xx, np.ones((len(xx), 1), dtype=np.float64)], axis=1)
        return xx @ self.coef


def _fit_ridge_multi(x: np.ndarray, y: np.ndarray, *, l2: float) -> RidgeModel | None:
    mask = _finite_rows(x, y)
    if int(mask.sum()) < 100:
        return None
    x_train = np.asarray(x[mask], dtype=np.float64)
    y_train = np.asarray(y[mask], dtype=np.float64)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    xx = (x_train - mean) / std
    xx = np.concatenate([xx, np.ones((len(xx), 1), dtype=np.float64)], axis=1)
    gram = xx.T @ xx
    reg = float(l2) * np.eye(gram.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    rhs = xx.T @ y_train
    try:
        coef = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram + reg) @ rhs
    return RidgeModel(mean=mean, std=std, coef=coef)


def _fit_bootstrap_models(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    seed: int,
    n_models: int,
) -> list[RidgeModel]:
    mask = _finite_rows(x, y)
    idx = np.flatnonzero(mask)
    if len(idx) < 100:
        return []
    rng = np.random.default_rng(seed)
    models: list[RidgeModel] = []
    sample_size = max(100, int(len(idx) * 0.80))
    for _ in range(int(n_models)):
        sample = np.sort(rng.choice(idx, size=sample_size, replace=True))
        model = _fit_ridge_multi(x[sample], y[sample], l2=l2)
        if model is not None:
            models.append(model)
    return models


def _predict_ensemble(models: list[RidgeModel], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds = np.stack([m.predict(x) for m in models], axis=0)
    return np.mean(preds, axis=0), np.std(preds, axis=0)


def _positions_from_prediction(
    pred_values: np.ndarray,
    *,
    candidates: tuple[float, ...],
    threshold: float,
    benchmark_position: float,
    active_mask: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    uncertainty_cap: float | None = None,
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(pred_values, dtype=np.float64)
    candidates_arr = np.asarray(candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(candidates_arr - float(benchmark_position))))
    best_idx = np.argmax(pred, axis=1)
    improve = pred[np.arange(len(pred)), best_idx] - pred[:, bench_idx]
    choose = improve > float(threshold)
    if active_mask is not None:
        choose &= np.asarray(active_mask, dtype=bool)
    if uncertainty is not None and uncertainty_cap is not None:
        unc = np.asarray(uncertainty, dtype=np.float64)
        choose &= np.isfinite(unc) & (unc <= float(uncertainty_cap))
    selected = np.full(len(pred), float(benchmark_position), dtype=np.float64)
    selected[choose] = candidates_arr[best_idx[choose]]
    return selected, {
        "pred_improve_mean": float(np.nanmean(improve)),
        "pred_improve_top10": _top_fraction_mean(improve, improve, 0.10),
        "raw_active_rate": float(np.mean(choose)),
    }


def _shift_for_execution(selected: np.ndarray, benchmark_position: float) -> np.ndarray:
    sel = np.asarray(selected, dtype=np.float64)
    out = np.full(len(sel), float(benchmark_position), dtype=np.float64)
    if len(sel) > 1:
        out[1:] = sel[:-1]
    return out


def _apply_event_throttle(
    selected: np.ndarray,
    *,
    benchmark_position: float,
    cooldown_bars: int,
    hold_bars: int,
) -> np.ndarray:
    sel = np.asarray(selected, dtype=np.float64)
    if int(cooldown_bars) <= 0 and int(hold_bars) <= 1:
        return sel.copy()
    out = np.full(len(sel), float(benchmark_position), dtype=np.float64)
    i = 0
    cooldown = max(int(cooldown_bars), 0)
    hold = max(int(hold_bars), 1)
    while i < len(sel):
        if abs(float(sel[i]) - float(benchmark_position)) <= 1e-12:
            i += 1
            continue
        end = min(len(sel), i + hold)
        out[i:end] = float(sel[i])
        i = end + cooldown
    return out


def _backtest_positions(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
) -> tuple[dict, np.ndarray]:
    t = min(len(returns), len(positions))
    rets = np.asarray(returns[:t], dtype=np.float64)
    pos = np.asarray(positions[:t], dtype=np.float64)
    bench = np.full(t, float(benchmark_position), dtype=np.float64)
    metrics = Backtest(
        rets,
        pos,
        spread_bps=float(costs_cfg.get("spread_bps", 5.0)),
        fee_rate=float(costs_cfg.get("fee_rate", 0.0004)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 2.0)),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=bench,
    ).run()
    stats = _action_stats(pos, benchmark_position=benchmark_position)
    pnl = compute_pnl(
        rets,
        pos,
        spread_bps=float(costs_cfg.get("spread_bps", 5.0)),
        fee_rate=float(costs_cfg.get("fee_rate", 0.0004)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 2.0)),
    )
    out = {
        "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
        "sharpe_delta": float(metrics.sharpe_delta or 0.0),
        "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
        "period_win_rate": float(metrics.period_win_rate_vs_bh or 0.0),
        "bar_win_rate": float(metrics.win_rate_vs_bh or 0.0),
        "turnover": float(stats["turnover"]),
        "long_rate": float(stats["long"]),
        "short_rate": float(stats["short"]),
        "flat_rate": float(stats["flat"]),
        "mean_position": float(stats["mean"]),
        "n_trades": int(metrics.n_trades),
    }
    return out, pnl


def _metric_score(metrics: dict, spec: SelectorSpec) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    sharpe = float(metrics.get("sharpe_delta", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    active = 1.0 - float(metrics.get("flat_rate", 0.0))
    if turnover <= 1e-12 and active <= 1e-12:
        return 0.0
    if maxdd > float(spec.maxdd_cap_pt) or turnover > float(spec.turnover_cap) or active > float(spec.active_cap):
        return -1_000_000.0 + alpha + 2.0 * sharpe - max(turnover, 0.0)
    min_dd_improve = float(spec.min_val_maxdd_improvement_pt)
    if min_dd_improve > 0.0 and maxdd > -min_dd_improve:
        return -1_000_000.0 + alpha + 2.0 * sharpe - max(turnover, 0.0)
    penalty = 0.0
    penalty += 5.0 * max(maxdd - float(spec.maxdd_cap_pt), 0.0)
    penalty += 2.0 * max(turnover - float(spec.turnover_cap), 0.0)
    penalty += 10.0 * max(active - float(spec.active_cap), 0.0)
    return (
        alpha
        + 2.0 * sharpe
        - float(spec.turnover_score_coef) * turnover
        - float(spec.active_score_coef) * active
        - penalty
    )


def _threshold_grid(improve: np.ndarray, *, active_cap: float) -> list[float]:
    vals = np.asarray(improve, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("inf")]
    qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
    qvals = [float(np.quantile(vals, q)) for q in qs]
    cap_q = max(0.0, min(0.995, 1.0 - float(active_cap)))
    qvals.append(float(np.quantile(vals, cap_q)))
    qvals.extend([0.0, 0.00001, 0.00005, 0.00010, 0.00025, 0.00050, 0.001, 0.0025, 0.005, float("inf")])
    return sorted(set(qvals))


def _regime_masks(train_returns: np.ndarray, eval_returns: np.ndarray) -> dict[str, np.ndarray]:
    train_vol = _rolling_past_vol(train_returns, 64)
    eval_vol = _rolling_past_vol(eval_returns, 64)
    finite = train_vol[np.isfinite(train_vol)]
    if len(finite) < 20:
        return {"all": np.ones(len(eval_returns), dtype=bool)}
    q1, q2 = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "all": np.ones(len(eval_returns), dtype=bool),
        "low_vol": eval_vol <= q1,
        "mid_vol": (eval_vol > q1) & (eval_vol <= q2),
        "high_vol": eval_vol > q2,
        "not_high_vol": eval_vol <= q2,
    }


def _pullback_no_fire_mask(returns: np.ndarray) -> np.ndarray:
    ret = np.asarray(returns, dtype=np.float64)
    past_ret32 = _rolling_past_sum(ret, 32)
    past_ret64 = _rolling_past_sum(ret, 64)
    past_vol64 = _rolling_past_vol(ret, 64)
    current_dd, _uw = _online_drawdown(ret)
    vol_ref = np.nanmedian(past_vol64[np.isfinite(past_vol64)]) if np.any(np.isfinite(past_vol64)) else 0.0
    return (
        (past_ret32 < -0.005)
        & (past_ret64 > 0.005)
        & (current_dd < -0.015)
        & (current_dd > -0.120)
        & (past_vol64 >= 0.75 * max(float(vol_ref), 1e-8))
    )


def _selected_utility_stats(
    selected: np.ndarray,
    actual_values: np.ndarray,
    *,
    candidates: tuple[float, ...],
    benchmark_position: float,
) -> dict:
    candidates_arr = np.asarray(candidates, dtype=np.float64)
    idx = np.argmin(np.abs(candidates_arr[None, :] - selected[:, None]), axis=1)
    vals = np.asarray(actual_values, dtype=np.float64)
    valid = np.isfinite(vals[np.arange(len(vals)), idx])
    if not np.any(valid):
        return {"selected_utility_mean": float("nan"), "selected_utility_positive_rate": float("nan")}
    selected_util = vals[np.arange(len(vals)), idx]
    bench_idx = int(np.argmin(np.abs(candidates_arr - float(benchmark_position))))
    best_idx = np.nanargmax(np.where(np.isfinite(vals), vals, -np.inf), axis=1)
    best_improve = vals[np.arange(len(vals)), best_idx] - vals[:, bench_idx]
    return {
        "selected_utility_mean": float(np.nanmean(selected_util[valid])),
        "selected_utility_positive_rate": float(np.nanmean(selected_util[valid] > 0.0)),
        "oracle_best_improve_top10": _top_fraction_mean(best_improve, best_improve, 0.10),
    }


def _selected_event_context_stats(
    selected: np.ndarray,
    returns: np.ndarray,
    *,
    benchmark_position: float,
) -> dict:
    sel = np.asarray(selected, dtype=np.float64)
    idx = np.flatnonzero(np.abs(sel - float(benchmark_position)) > 1e-12)
    if len(idx) == 0:
        return {
            "event_count": 0,
            "event_past_ret32": float("nan"),
            "event_past_ret64": float("nan"),
            "event_past_vol64": float("nan"),
            "event_current_dd": float("nan"),
        }
    ret = np.asarray(returns, dtype=np.float64)
    past_ret32 = _rolling_past_sum(ret, 32)
    past_ret64 = _rolling_past_sum(ret, 64)
    past_vol64 = _rolling_past_vol(ret, 64)
    current_dd, _uw = _online_drawdown(ret)
    return {
        "event_count": int(len(idx)),
        "event_past_ret32": float(np.mean(past_ret32[idx])),
        "event_past_ret64": float(np.mean(past_ret64[idx])),
        "event_past_vol64": float(np.mean(past_vol64[idx])),
        "event_current_dd": float(np.mean(current_dd[idx])),
    }


def _evaluate_selector(
    *,
    spec: SelectorSpec,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    l2: float,
    seed: int,
) -> dict:
    unit_cost = _unit_cost(costs_cfg)
    y_train, train_valid = _candidate_utilities(
        train_returns,
        candidates=spec.candidates,
        horizon=spec.horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=spec.dd_penalty,
        vol_penalty=spec.vol_penalty,
    )
    y_val, val_valid = _candidate_utilities(
        val_returns,
        candidates=spec.candidates,
        horizon=spec.horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=spec.dd_penalty,
        vol_penalty=spec.vol_penalty,
    )
    y_test, test_valid = _candidate_utilities(
        test_returns,
        candidates=spec.candidates,
        horizon=spec.horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=spec.dd_penalty,
        vol_penalty=spec.vol_penalty,
    )

    uses_tb_guard = spec.mode in {"tb_guard", "tb_guard_pullback", "tb_guard_pullback_evalonly"}
    if spec.mode == "uncertainty":
        models = _fit_bootstrap_models(x_train[train_valid], y_train[train_valid], l2=l2, seed=seed, n_models=5)
        if not models:
            return {"status": "no_model"}
        pred_train, unc_train_all = _predict_ensemble(models, x_train)
        pred_val, unc_val_all = _predict_ensemble(models, x_val)
        pred_test, unc_test_all = _predict_ensemble(models, x_test)
    else:
        model = _fit_ridge_multi(x_train[train_valid], y_train[train_valid], l2=l2)
        if model is None:
            return {"status": "no_model"}
        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)
        pred_test = model.predict(x_test)
        unc_train_all = np.zeros_like(pred_train)
        unc_val_all = np.zeros_like(pred_val)
        unc_test_all = np.zeros_like(pred_test)

    danger_train = None
    danger_val = None
    danger_test = None
    danger_caps: list[float | None] = [None]
    if uses_tb_guard:
        train_tb = _triple_barrier_labels(
            train_returns,
            horizon=spec.guard_horizon,
            vol_window=spec.guard_vol_window,
            barrier_k=spec.guard_barrier_k,
        )
        val_tb = _triple_barrier_labels(
            val_returns,
            horizon=spec.guard_horizon,
            vol_window=spec.guard_vol_window,
            barrier_k=spec.guard_barrier_k,
        )
        test_tb = _triple_barrier_labels(
            test_returns,
            horizon=spec.guard_horizon,
            vol_window=spec.guard_vol_window,
            barrier_k=spec.guard_barrier_k,
        )
        tb_train_valid = np.asarray(train_tb["valid"], dtype=bool)
        tb_model = _fit_binary_model(
            x_train[tb_train_valid],
            np.asarray(train_tb["tb_down"][tb_train_valid], dtype=np.int64),
            max_train_samples=max(len(x_train), 50000),
            seed=seed + 917,
        )
        danger_train = _score_binary(tb_model, x_train)
        danger_val = _score_binary(tb_model, x_val)
        danger_test = _score_binary(tb_model, x_test)
        finite_danger = danger_val[np.asarray(val_tb["valid"], dtype=bool) & np.isfinite(danger_val)]
        if len(finite_danger):
            danger_caps = [float(np.quantile(finite_danger, q)) for q in (0.25, 0.40, 0.55, 0.70, 0.85)]
            danger_caps.append(float("inf"))

    best_idx_train = np.argmax(pred_train, axis=1)
    bench_idx = int(np.argmin(np.abs(np.asarray(spec.candidates) - benchmark_position)))
    best_idx_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_idx_val] - pred_val[:, bench_idx]
    best_idx_test = np.argmax(pred_test, axis=1)
    improve_test = pred_test[np.arange(len(pred_test)), best_idx_test] - pred_test[:, bench_idx]
    thresholds_raw = _threshold_grid(improve_val[val_valid], active_cap=spec.active_cap)
    thresholds = sorted(
        {
            float("inf") if not math.isfinite(float(t)) else max(float(t), float(spec.min_threshold))
            for t in thresholds_raw
        }
    )
    uncertainty_quantiles = [None]
    if spec.mode == "uncertainty":
        uncertainty_quantiles = [0.50, 0.70, 0.85]

    train_regimes = _regime_masks(train_returns, train_returns)
    val_regimes = _regime_masks(train_returns, val_returns)
    test_regimes = _regime_masks(train_returns, test_returns)
    regime_names = ["all"]
    if spec.mode == "regime":
        regime_names = [k for k in val_regimes.keys() if k != "all"] + ["all"]
    val_pullback_block = (
        _pullback_no_fire_mask(val_returns)
        if spec.mode == "tb_guard_pullback"
        else np.zeros(len(val_returns), dtype=bool)
    )
    train_pullback_block = (
        _pullback_no_fire_mask(train_returns)
        if spec.mode in {"tb_guard_pullback", "tb_guard_pullback_evalonly"}
        else np.zeros(len(train_returns), dtype=bool)
    )
    test_pullback_block = (
        _pullback_no_fire_mask(test_returns)
        if spec.mode in {"tb_guard_pullback", "tb_guard_pullback_evalonly"}
        else np.zeros(len(test_returns), dtype=bool)
    )

    best: dict[str, Any] | None = None
    best_val_positions: np.ndarray | None = None
    cooldown_choices = tuple(spec.cooldown_grid) if spec.cooldown_grid else (int(spec.cooldown_bars),)
    for regime_name in regime_names:
        val_active_mask = val_regimes.get(regime_name, np.ones(len(val_returns), dtype=bool))
        test_active_mask = test_regimes.get(regime_name, np.ones(len(test_returns), dtype=bool))
        for q in uncertainty_quantiles:
            if q is None:
                unc_cap = None
                unc_val = None
            else:
                unc_val = unc_val_all[np.arange(len(unc_val_all)), best_idx_val]
                finite_unc = unc_val[np.isfinite(unc_val) & val_valid]
                unc_cap = float(np.quantile(finite_unc, q)) if len(finite_unc) else float("inf")
            for danger_cap in danger_caps:
                danger_mask = np.ones(len(val_returns), dtype=bool)
                if danger_val is not None and danger_cap is not None:
                    danger_mask = np.isfinite(danger_val) & (danger_val <= float(danger_cap))
                for cooldown_choice in cooldown_choices:
                    for threshold in thresholds:
                        selected_val, diag = _positions_from_prediction(
                            pred_val,
                            candidates=spec.candidates,
                            threshold=threshold,
                            benchmark_position=benchmark_position,
                            active_mask=val_active_mask & val_valid & danger_mask & (~val_pullback_block),
                            uncertainty=unc_val,
                            uncertainty_cap=unc_cap,
                        )
                        selected_val = _apply_event_throttle(
                            selected_val,
                            benchmark_position=benchmark_position,
                            cooldown_bars=int(cooldown_choice),
                            hold_bars=spec.hold_bars,
                        )
                        val_positions = _shift_for_execution(selected_val, benchmark_position)
                        val_metrics, _val_pnl = _backtest_positions(
                            val_returns,
                            val_positions,
                            cfg=cfg,
                            costs_cfg=costs_cfg,
                            benchmark_position=benchmark_position,
                        )
                        score = _metric_score(val_metrics, spec)
                        candidate = {
                            "threshold": float(threshold) if math.isfinite(float(threshold)) else "inf",
                            "regime": regime_name,
                            "uncertainty_quantile": q,
                            "uncertainty_cap": unc_cap,
                            "cooldown_bars": int(cooldown_choice),
                            "danger_cap": (
                                float(danger_cap)
                                if danger_cap is not None and math.isfinite(float(danger_cap))
                                else ("inf" if danger_cap is not None else None)
                            ),
                            "val_score": float(score),
                            "val": {**val_metrics, **diag},
                        }
                        if best is None or score > float(best["val_score"]):
                            best = candidate
                            best_val_positions = val_positions.copy()

    if best is None:
        return {"status": "no_selection"}
    regime_name = str(best["regime"])
    train_active_mask = train_regimes.get(regime_name, np.ones(len(train_returns), dtype=bool))
    test_active_mask = test_regimes.get(regime_name, np.ones(len(test_returns), dtype=bool))
    if best["uncertainty_quantile"] is None:
        unc_train = None
        unc_test = None
        unc_cap = None
    else:
        unc_train = unc_train_all[np.arange(len(unc_train_all)), best_idx_train]
        unc_test = unc_test_all[np.arange(len(unc_test_all)), best_idx_test]
        unc_cap = float(best["uncertainty_cap"])
    threshold = float("inf") if best["threshold"] == "inf" else float(best["threshold"])
    danger_train_mask = np.ones(len(train_returns), dtype=bool)
    if danger_train is not None and best.get("danger_cap") is not None:
        danger_cap_raw = best.get("danger_cap")
        danger_cap = float("inf") if danger_cap_raw == "inf" else float(danger_cap_raw)
        danger_train_mask = np.isfinite(danger_train) & (danger_train <= danger_cap)
    danger_test_mask = np.ones(len(test_returns), dtype=bool)
    if danger_test is not None and best.get("danger_cap") is not None:
        danger_cap_raw = best.get("danger_cap")
        danger_cap = float("inf") if danger_cap_raw == "inf" else float(danger_cap_raw)
        danger_test_mask = np.isfinite(danger_test) & (danger_test <= danger_cap)
    selected_train, _diag_train = _positions_from_prediction(
        pred_train,
        candidates=spec.candidates,
        threshold=threshold,
        benchmark_position=benchmark_position,
        active_mask=train_active_mask & train_valid & danger_train_mask & (~train_pullback_block),
        uncertainty=unc_train,
        uncertainty_cap=unc_cap,
    )
    selected_train = _apply_event_throttle(
        selected_train,
        benchmark_position=benchmark_position,
        cooldown_bars=int(best.get("cooldown_bars", spec.cooldown_bars)),
        hold_bars=spec.hold_bars,
    )
    train_positions = _shift_for_execution(selected_train, benchmark_position)
    selected_test, diag_test = _positions_from_prediction(
        pred_test,
        candidates=spec.candidates,
        threshold=threshold,
        benchmark_position=benchmark_position,
        active_mask=test_active_mask & test_valid & danger_test_mask & (~test_pullback_block),
        uncertainty=unc_test,
        uncertainty_cap=unc_cap,
    )
    selected_test = _apply_event_throttle(
        selected_test,
        benchmark_position=benchmark_position,
        cooldown_bars=int(best.get("cooldown_bars", spec.cooldown_bars)),
        hold_bars=spec.hold_bars,
    )
    test_positions = _shift_for_execution(selected_test, benchmark_position)
    test_metrics, test_pnl = _backtest_positions(
        test_returns,
        test_positions,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )
    util_stats = _selected_utility_stats(
        selected_test,
        y_test,
        candidates=spec.candidates,
        benchmark_position=benchmark_position,
    )
    context_stats = _selected_event_context_stats(
        selected_test,
        test_returns,
        benchmark_position=benchmark_position,
    )
    ranking = _ranking_stats(pred_test, y_test, test_valid, bench_idx)
    return {
        "status": "ok",
        "lane": spec.lane,
        "variant": spec.name,
        "candidates": list(spec.candidates),
        "horizon": int(spec.horizon),
        "dd_penalty": float(spec.dd_penalty),
        "vol_penalty": float(spec.vol_penalty),
        "selection": best,
        "test": {**test_metrics, **diag_test, **util_stats, **context_stats, **ranking},
        "_train_positions": train_positions,
        "_val_positions": best_val_positions if best_val_positions is not None else np.full(len(val_returns), benchmark_position),
        "_test_positions": test_positions,
        "test_pnl": test_pnl,
    }


def _top_fraction_mean(score: np.ndarray, value: np.ndarray, frac: float) -> float:
    s = np.asarray(score, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(v)
    if int(mask.sum()) == 0:
        return float("nan")
    ss = s[mask]
    vv = v[mask]
    n = max(1, int(math.ceil(len(ss) * float(frac))))
    idx = np.argsort(ss, kind="mergesort")[-n:]
    return float(np.mean(vv[idx]))


def _ranking_stats(pred: np.ndarray, actual: np.ndarray, valid: np.ndarray, bench_idx: int) -> dict:
    mask = np.asarray(valid, dtype=bool)
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    n = min(len(pred), len(actual), len(mask))
    pred = pred[:n]
    actual = actual[:n]
    mask = mask[:n] & np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(actual), axis=1)
    if int(mask.sum()) < 20:
        return {
            "row_top1_match": float("nan"),
            "selected_vs_benchmark_utility": float("nan"),
            "pred_improve_actual_corr": float("nan"),
        }
    pred_m = pred[mask]
    actual_m = actual[mask]
    pred_best = np.argmax(pred_m, axis=1)
    actual_best = np.argmax(actual_m, axis=1)
    pred_improve = pred_m[np.arange(len(pred_m)), pred_best] - pred_m[:, bench_idx]
    actual_improve = actual_m[np.arange(len(actual_m)), pred_best] - actual_m[:, bench_idx]
    corr = (
        float(np.corrcoef(pred_improve, actual_improve)[0, 1])
        if np.std(pred_improve) > 1e-12 and np.std(actual_improve) > 1e-12
        else float("nan")
    )
    return {
        "row_top1_match": float(np.mean(pred_best == actual_best)),
        "selected_vs_benchmark_utility": float(np.mean(actual_improve)),
        "pred_improve_actual_corr": corr,
    }


def _triple_barrier_labels(
    returns: np.ndarray,
    *,
    horizon: int,
    vol_window: int,
    barrier_k: float,
) -> dict[str, np.ndarray]:
    windows, valid = _future_windows(returns, horizon)
    n = len(returns)
    down = np.zeros(n, dtype=np.int64)
    up_safe = np.zeros(n, dtype=np.int64)
    first_hit = np.full(n, 0, dtype=np.int64)
    if len(windows) == 0:
        return {"valid": valid.astype(np.int64), "tb_down": down, "tb_up_safe": up_safe, "first_hit": first_hit}
    past_vol = _rolling_past_vol(returns, vol_window)
    valid_n = windows.shape[0]
    path = np.cumsum(windows, axis=1)
    dd = _path_max_drawdown(windows, 1.0)
    for i in range(valid_n):
        barrier = max(float(past_vol[i]) * math.sqrt(float(horizon)) * float(barrier_k), 1e-6)
        upper_hits = np.flatnonzero(path[i] >= barrier)
        lower_hits = np.flatnonzero(path[i] <= -barrier)
        u = int(upper_hits[0]) if len(upper_hits) else horizon + 1
        l = int(lower_hits[0]) if len(lower_hits) else horizon + 1
        if l < u:
            down[i] = 1
            first_hit[i] = -1
        elif u < l:
            first_hit[i] = 1
            up_safe[i] = 1 if float(dd[i]) <= barrier * 0.75 else 0
    return {"valid": valid.astype(np.int64), "tb_down": down, "tb_up_safe": up_safe, "first_hit": first_hit}


def _evaluate_barrier_target(
    *,
    target_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_labels: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    seed: int,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict:
    train_valid = np.asarray(train_labels["valid"], dtype=bool)
    val_valid = np.asarray(val_labels["valid"], dtype=bool)
    test_valid = np.asarray(test_labels["valid"], dtype=bool)
    model = _fit_binary_model(
        x_train[train_valid],
        np.asarray(train_labels[target_name][train_valid], dtype=np.int64),
        max_train_samples=50000,
        seed=seed,
    )
    val_score = _score_binary(model, x_val[val_valid])
    threshold, val_rates = _select_threshold(
        np.asarray(val_labels[target_name][val_valid], dtype=np.int64),
        val_score,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
    )
    return {
        "target": target_name,
        "density": {
            "train": float(np.mean(train_labels[target_name][train_valid])) if np.any(train_valid) else float("nan"),
            "val": float(np.mean(val_labels[target_name][val_valid])) if np.any(val_valid) else float("nan"),
            "test": float(np.mean(test_labels[target_name][test_valid])) if np.any(test_valid) else float("nan"),
        },
        "threshold_selected_on_val": val_rates,
        "test": _binary_eval(
            model=model,
            x=x_test[test_valid],
            y=np.asarray(test_labels[target_name][test_valid], dtype=np.int64),
            threshold=threshold,
        ),
    }


def _evaluate_barriers(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    seed: int,
) -> dict:
    out: dict[str, Any] = {"lane": "A_triple_barrier"}
    configs = (
        ("h16_k100", 16, 64, 1.00),
        ("h32_k125", 32, 64, 1.25),
        ("h64_k150", 64, 128, 1.50),
    )
    for label, horizon, vol_window, barrier_k in configs:
        train_labels = _triple_barrier_labels(
            train_returns, horizon=horizon, vol_window=vol_window, barrier_k=barrier_k
        )
        val_labels = _triple_barrier_labels(
            val_returns, horizon=horizon, vol_window=vol_window, barrier_k=barrier_k
        )
        test_labels = _triple_barrier_labels(
            test_returns, horizon=horizon, vol_window=vol_window, barrier_k=barrier_k
        )
        out[f"{label}_down"] = _evaluate_barrier_target(
            target_name="tb_down",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            seed=seed + horizon,
            false_active_cap=0.15,
            pred_rate_cap=0.25,
        )
        out[f"{label}_up_safe"] = _evaluate_barrier_target(
            target_name="tb_up_safe",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            seed=seed + horizon + 13,
            false_active_cap=0.15,
            pred_rate_cap=0.25,
        )
    return out


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items() if k != "test_pnl"}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _min(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.min(vals)) if vals else float("nan")


def _max(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.max(vals)) if vals else float("nan")


def _aggregate_selectors(results: dict[str, dict], groups: dict[str, list[int]]) -> dict:
    variants = sorted(
        {
            name
            for fold_rows in results.values()
            for name, row in fold_rows.get("selectors", {}).items()
            if row.get("status") == "ok"
        }
    )
    out: dict[str, dict] = {}
    for group_name, fold_ids in groups.items():
        out[group_name] = {}
        for variant in variants:
            rows = []
            pnls = []
            for fold in fold_ids:
                row = results.get(str(fold), {}).get("selectors", {}).get(variant)
                if row and row.get("status") == "ok":
                    rows.append(row["test"])
                    if "test_pnl" in row:
                        pnls.append(row["test_pnl"])
            if not rows:
                continue
            alpha = [r["alpha_excess_pt"] for r in rows]
            maxdd = [r["maxdd_delta_pt"] for r in rows]
            sharpe = [r["sharpe_delta"] for r in rows]
            turnover = [r["turnover"] for r in rows]
            long_rate = [r["long_rate"] for r in rows]
            flat_rate = [r["flat_rate"] for r in rows]
            out[group_name][variant] = {
                "folds": int(len(rows)),
                "alpha_mean": _mean(alpha),
                "alpha_median": _median(alpha),
                "alpha_worst": _min(alpha),
                "maxdd_mean": _mean(maxdd),
                "maxdd_worst": _max(maxdd),
                "sharpe_mean": _mean(sharpe),
                "sharpe_worst": _min(sharpe),
                "turnover_max": _max(turnover),
                "long_max": _max(long_rate),
                "flat_mean": _mean(flat_rate),
                "pass_rate_alpha_pos_maxdd_nonpos": float(
                    np.mean(
                        [
                            (a > 0.0 and d <= 0.0 and to <= 3.5 and lr <= 0.03)
                            for a, d, to, lr in zip(alpha, maxdd, turnover, long_rate)
                        ]
                    )
                ),
                "pbo": compute_pbo(pnls) if len(pnls) >= 2 else float("nan"),
            }
        if out[group_name]:
            valid_variants = list(out[group_name].keys())
            for selected_fold in fold_ids:
                rank_rows = []
                for variant in valid_variants:
                    train_folds = [f for f in fold_ids if f != selected_fold]
                    train_rows = [
                        results.get(str(f), {}).get("selectors", {}).get(variant, {}).get("test", {})
                        for f in train_folds
                    ]
                    train_score = _mean(
                        [
                            float(r.get("alpha_excess_pt", 0.0))
                            - 5.0 * max(float(r.get("maxdd_delta_pt", 0.0)), 0.0)
                            - 0.1 * max(float(r.get("turnover", 0.0)) - 3.5, 0.0)
                            for r in train_rows
                            if r
                        ]
                    )
                    rank_rows.append((train_score, variant))
                if not rank_rows:
                    continue
                _score, selected_variant = max(rank_rows, key=lambda x: x[0])
                oos = results.get(str(selected_fold), {}).get("selectors", {}).get(selected_variant, {}).get("test")
                if oos is not None:
                    out[group_name].setdefault("_nested_leave_one_fold", {"selected": []})["selected"].append(
                        {
                            "oos_fold": int(selected_fold),
                            "variant": selected_variant,
                            "alpha_excess_pt": float(oos.get("alpha_excess_pt", 0.0)),
                            "maxdd_delta_pt": float(oos.get("maxdd_delta_pt", 0.0)),
                            "turnover": float(oos.get("turnover", 0.0)),
                        }
                    )
            nested = out[group_name].get("_nested_leave_one_fold")
            if nested:
                selected_rows = nested["selected"]
                nested.update(
                    {
                        "folds": int(len(selected_rows)),
                        "alpha_mean": _mean([r["alpha_excess_pt"] for r in selected_rows]),
                        "alpha_worst": _min([r["alpha_excess_pt"] for r in selected_rows]),
                        "maxdd_worst": _max([r["maxdd_delta_pt"] for r in selected_rows]),
                        "turnover_max": _max([r["turnover"] for r in selected_rows]),
                    }
                )
    return out


def _aggregate_barriers(results: dict[str, dict], groups: dict[str, list[int]]) -> dict:
    out: dict[str, dict] = {}
    for group_name, fold_ids in groups.items():
        out[group_name] = {}
        targets = sorted(
            {
                target
                for fold in fold_ids
                for target in results.get(str(fold), {}).get("barriers", {}).keys()
                if target != "lane"
            }
        )
        for target in targets:
            rows = []
            densities = []
            for fold in fold_ids:
                row = results.get(str(fold), {}).get("barriers", {}).get(target)
                if row:
                    rows.append(row["test"])
                    densities.append(row["density"]["test"])
            if rows:
                out[group_name][target] = {
                    "folds": int(len(rows)),
                    "density_mean": _mean(densities),
                    "auc_mean": _mean([r.get("auc", float("nan")) for r in rows]),
                    "auc_worst": _min([r.get("auc", float("nan")) for r in rows]),
                    "false_active_worst": _max([r.get("false_active_rate", float("nan")) for r in rows]),
                    "recall_worst": _min([r.get("recall", float("nan")) for r in rows]),
                    "pred_rate_mean": _mean([r.get("pred_rate", float("nan")) for r in rows]),
                }
    return out


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _write_md(path: str, payload: dict) -> None:
    lines = [
        "# Plan 2 Exploration Board Probe",
        "",
        f"Config: `{payload['config']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "## Selector Aggregate",
        "",
    ]
    for group, rows in payload["selector_aggregate"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                "| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, row in rows.items():
            if variant.startswith("_"):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        str(row["folds"]),
                        _fmt(row["alpha_mean"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["maxdd_mean"]),
                        _fmt(row["maxdd_worst"]),
                        _fmt(row["sharpe_mean"]),
                        _fmt(row["turnover_max"]),
                        _fmt(row["long_max"]),
                        _fmt(row["pass_rate_alpha_pos_maxdd_nonpos"]),
                        _fmt(row["pbo"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        nested = rows.get("_nested_leave_one_fold")
        if nested:
            lines.extend(
                [
                    "Nested leave-one-fold selector:",
                    "",
                    "| folds | alpha mean | alpha worst | maxdd worst | turnover max |",
                    "|---:|---:|---:|---:|---:|",
                    "| "
                    + " | ".join(
                        [
                            str(nested["folds"]),
                            _fmt(nested["alpha_mean"]),
                            _fmt(nested["alpha_worst"]),
                            _fmt(nested["maxdd_worst"]),
                            _fmt(nested["turnover_max"]),
                        ]
                    )
                    + " |",
                    "",
                ]
            )
    lines.extend(["## Triple Barrier Aggregate", ""])
    for group, rows in payload["barrier_aggregate"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                "| target | folds | density | AUC mean | AUC worst | false-active worst | recall worst | pred rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for target, row in rows.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        target,
                        str(row["folds"]),
                        _fmt(row["density_mean"]),
                        _fmt(row["auc_mean"]),
                        _fmt(row["auc_worst"]),
                        _fmt(row["false_active_worst"]),
                        _fmt(row["recall_worst"]),
                        _fmt(row["pred_rate_mean"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Fold Detail", ""])
    for fold, fold_rows in payload["results"].items():
        lines.extend([f"### Fold {fold}", ""])
        lines.append("| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for variant, row in fold_rows.get("selectors", {}).items():
            if row.get("status") != "ok":
                continue
            test = row["test"]
            sel = row["selection"]
            choice = (
                f"thr={sel['threshold']} regime={sel['regime']} "
                f"uq={sel['uncertainty_quantile']} cd={sel.get('cooldown_bars')} "
                f"danger={sel.get('danger_cap')}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        _fmt(test["alpha_excess_pt"]),
                        _fmt(test["maxdd_delta_pt"]),
                        _fmt(test["sharpe_delta"]),
                        _fmt(test["turnover"]),
                        _fmt(test["long_rate"]),
                        _fmt(test["flat_rate"]),
                        choice,
                    ]
                )
                + " |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.exploration_board_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--selector-names", default="")
    parser.add_argument("--output-json", default="documents/plan2_exploration_board.json")
    parser.add_argument("--output-md", default="documents/plan2_exploration_board.md")
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    results: dict[str, dict] = {}
    selector_filter = {x.strip() for x in str(args.selector_names or "").split(",") if x.strip()}
    selector_specs = [s for s in SELECTOR_SPECS if not selector_filter or s.name in selector_filter]
    if not selector_specs:
        raise ValueError(f"No selector matched --selector-names={args.selector_names!r}")
    for split in splits:
        print(f"[ExplorationBoard] fold={split.fold_idx} start")
        dataset = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train = _state_features(dataset.train_features, dataset.train_returns)
        x_val = _state_features(dataset.val_features, dataset.val_returns)
        x_test = _state_features(dataset.test_features, dataset.test_returns)
        fold_rows: dict[str, Any] = {
            "selectors": {},
            "barriers": _evaluate_barriers(
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                train_returns=dataset.train_returns,
                val_returns=dataset.val_returns,
                test_returns=dataset.test_returns,
                seed=args.seed + int(split.fold_idx),
            ),
        }
        for spec in selector_specs:
            print(f"[ExplorationBoard] fold={split.fold_idx} selector={spec.name}")
            fold_rows["selectors"][spec.name] = _evaluate_selector(
                spec=spec,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                train_returns=dataset.train_returns,
                val_returns=dataset.val_returns,
                test_returns=dataset.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                l2=args.ridge_l2,
                seed=args.seed + int(split.fold_idx) * 100,
            )
        results[str(split.fold_idx)] = fold_rows

    fold_ids = [int(split.fold_idx) for split in splits]
    groups = {"all": fold_ids}
    if all(f in fold_ids for f in (4, 5, 6)):
        groups["f456"] = [4, 5, 6]
    if all(f in fold_ids for f in (0, 4, 5)):
        groups["f045"] = [0, 4, 5]

    payload = {
        "config": args.config,
        "start": args.start,
        "end": args.end,
        "folds": fold_ids,
        "selector_specs": [_json_sanitize(s.__dict__) for s in selector_specs],
        "results": _json_sanitize(results),
        "selector_aggregate": _json_sanitize(_aggregate_selectors(results, groups)),
        "barrier_aggregate": _json_sanitize(_aggregate_barriers(results, groups)),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[ExplorationBoard] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
