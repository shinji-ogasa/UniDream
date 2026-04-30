from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from unidream.cli.ac_fire_dd_guard_probe import _maxdd_interval, _pnl_bar
from unidream.cli.ac_fire_timing_probe import ProbeRun, _load_actor_for_run, _parse_run
from unidream.cli.train import _action_stats, _benchmark_position_value, _benchmark_positions
from unidream.eval.backtest import Backtest, compute_pnl
from unidream.experiments.policy_fire import predict_with_policy_flags
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class LabelProbeConfig:
    horizons: tuple[int, ...]
    fire_eps: float
    harm_margin: float
    drawdown_worsen_margin: float
    trough_rebound_margin: float
    train_frac: float
    ridge_l2: float
    max_z_dim: int


def _as_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_2d(value, *, limit: int | None = None) -> np.ndarray | None:
    arr = _as_numpy(value)
    if arr is None:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = arr.astype(np.float64, copy=False)
    if limit is not None and arr.shape[1] > int(limit):
        arr = arr[:, : int(limit)]
    return arr


def _rolling_past_sum(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    if len(arr) == 0:
        return out
    csum = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    for i in range(len(arr)):
        start = max(0, i - int(window))
        out[i] = csum[i] - csum[start]
    return out


def _rolling_past_vol(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i - int(window))
        if i > start:
            out[i] = float(np.std(arr[start:i]))
    return out


def _online_state(
    returns: np.ndarray,
    positions: np.ndarray,
    costs_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State before each bar; labels may use future, features must not."""
    t = min(len(returns), len(positions))
    drawdown = np.zeros(t, dtype=np.float64)
    underwater = np.zeros(t, dtype=np.float64)
    equity_before = np.ones(t, dtype=np.float64)
    peak_before = np.ones(t, dtype=np.float64)
    equity = 1.0
    peak = 1.0
    prev_pos = 0.0
    uw = 0
    for i in range(t):
        dd = equity / max(peak, 1e-12) - 1.0
        drawdown[i] = dd
        underwater[i] = float(uw)
        equity_before[i] = equity
        peak_before[i] = peak
        pnl = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        equity *= float(math.exp(pnl))
        peak = max(peak, equity)
        uw = uw + 1 if equity < peak * (1.0 - 1e-10) else 0
        prev_pos = float(positions[i])
    return drawdown, underwater, equity_before, peak_before


def _future_drawdown_worsening(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    start: int,
    horizon: int,
    costs_cfg: dict,
    equity_before: float,
    peak_before: float,
    current_drawdown: float,
) -> float:
    equity = float(equity_before)
    peak = float(peak_before)
    min_drawdown = float(current_drawdown)
    prev_pos = float(positions[start - 1]) if start > 0 else 0.0
    end = min(len(returns), start + int(horizon))
    for i in range(start, end):
        pnl = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        equity *= float(math.exp(pnl))
        peak = max(peak, equity)
        min_drawdown = min(min_drawdown, equity / max(peak, 1e-12) - 1.0)
        prev_pos = float(positions[i])
    return float(max(0.0, float(current_drawdown) - min_drawdown))


def _window_pnl(
    returns: np.ndarray,
    positions: np.ndarray,
    start: int,
    horizon: int,
    costs_cfg: dict,
) -> np.ndarray:
    end = min(len(returns), start + int(horizon))
    out = np.zeros(max(0, end - start), dtype=np.float64)
    prev_pos = float(positions[start - 1]) if start > 0 else 0.0
    for k, i in enumerate(range(start, end)):
        out[k] = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        prev_pos = float(positions[i])
    return out


def _drawdown_magnitude_from_pnl(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    equity = np.exp(np.cumsum(np.asarray(pnl, dtype=np.float64)))
    interval = _maxdd_interval(equity)
    return float(max(0.0, -interval["maxdd"]))


def _build_labels(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    cfg: LabelProbeConfig,
    costs_cfg: dict,
) -> dict[int, dict[str, np.ndarray]]:
    t = min(len(returns), len(positions), len(no_adapter))
    current_dd, _underwater, equity_before, peak_before = _online_state(returns[:t], positions[:t], costs_cfg)
    out: dict[int, dict[str, np.ndarray]] = {}
    for horizon in cfg.horizons:
        fire_advantage = np.full(t, np.nan, dtype=np.float64)
        fire_harm = np.full(t, np.nan, dtype=np.float64)
        harm_margin = np.full(t, np.nan, dtype=np.float64)
        drawdown_worsening = np.full(t, np.nan, dtype=np.float64)
        future_drawdown = np.full(t, np.nan, dtype=np.float64)
        trough_exit = np.full(t, np.nan, dtype=np.float64)
        valid = np.zeros(t, dtype=bool)
        for i in range(0, max(0, t - int(horizon) + 1)):
            pnl_on = _window_pnl(returns, positions, i, horizon, costs_cfg)
            pnl_off = _window_pnl(returns, no_adapter, i, horizon, costs_cfg)
            if len(pnl_on) < int(horizon) or len(pnl_off) < int(horizon):
                continue
            valid[i] = True
            advantage = float(np.sum(pnl_on) - np.sum(pnl_off))
            dd_on = _drawdown_magnitude_from_pnl(pnl_on)
            dd_off = _drawdown_magnitude_from_pnl(pnl_off)
            fire_advantage[i] = advantage
            harm_gap = dd_on - dd_off
            harm_margin[i] = harm_gap
            fire_harm[i] = 1.0 if harm_gap > cfg.harm_margin else 0.0
            worsening = _future_drawdown_worsening(
                returns=returns,
                positions=positions,
                start=i,
                horizon=horizon,
                costs_cfg=costs_cfg,
                equity_before=float(equity_before[i]),
                peak_before=float(peak_before[i]),
                current_drawdown=float(current_dd[i]),
            )
            future_drawdown[i] = worsening
            drawdown_worsening[i] = 1.0 if worsening > cfg.drawdown_worsen_margin else 0.0

            cum = np.cumsum(pnl_on)
            trough_idx = int(np.argmin(cum)) if len(cum) else 0
            rebound = float(cum[-1] - cum[trough_idx]) if len(cum) else 0.0
            early = trough_idx <= max(1, int(horizon) // 3)
            trough_exit[i] = 1.0 if early and rebound > cfg.trough_rebound_margin else 0.0
        out[int(horizon)] = {
            "valid": valid,
            "fire_advantage": fire_advantage,
            "fire_harm": fire_harm,
            "fire_harm_margin": harm_margin,
            "drawdown_worsening": drawdown_worsening,
            "future_drawdown": future_drawdown,
            "trough_exit": trough_exit,
        }
    return out


def _build_feature_matrix(
    *,
    enc: dict,
    returns: np.ndarray,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    fire: np.ndarray,
    regime: np.ndarray | None,
    advantage: np.ndarray | None,
    costs_cfg: dict,
    max_z_dim: int,
) -> np.ndarray:
    t = min(len(returns), len(positions), len(no_adapter), len(fire))
    delta = np.asarray(positions[:t], dtype=np.float64) - np.asarray(no_adapter[:t], dtype=np.float64)
    current_dd, underwater, equity_before, peak_before = _online_state(returns[:t], positions[:t], costs_cfg)
    base_dd, base_underwater, _base_equity, base_peak = _online_state(returns[:t], no_adapter[:t], costs_cfg)
    engineered = [
        np.asarray(positions[:t], dtype=np.float64).reshape(-1, 1),
        np.asarray(no_adapter[:t], dtype=np.float64).reshape(-1, 1),
        delta.reshape(-1, 1),
        np.asarray(fire[:t], dtype=np.float64).reshape(-1, 1),
        current_dd.reshape(-1, 1),
        base_dd.reshape(-1, 1),
        underwater.reshape(-1, 1),
        base_underwater.reshape(-1, 1),
        equity_before.reshape(-1, 1),
        peak_before.reshape(-1, 1),
        base_peak.reshape(-1, 1),
    ]
    for window in (4, 16, 32, 64):
        engineered.append(_rolling_past_sum(returns[:t], window).reshape(-1, 1))
        engineered.append(_rolling_past_vol(returns[:t], window).reshape(-1, 1))
    z = _flatten_2d(enc.get("z"), limit=max_z_dim)
    h = _flatten_2d(enc.get("h"))
    regime_arr = _flatten_2d(regime)
    adv_arr = _flatten_2d(advantage)
    parts = [part[:t] for part in (z, h, regime_arr, adv_arr) if part is not None]
    parts.extend(engineered)
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = int(np.sum(y))
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _average_tie_ranks(s)
    return float((np.sum(ranks[y]) - pos * (pos + 1) / 2.0) / (pos * neg))


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = int(np.sum(y))
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1.0)
    return float(np.sum(precision[y_sorted]) / pos)


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    l2: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_std = np.std(x_train, axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x_tr = (x_train - x_mean) / x_std
    x_ev = (x_eval - x_mean) / x_std
    x_tr = np.concatenate([x_tr, np.ones((len(x_tr), 1), dtype=np.float64)], axis=1)
    x_ev = np.concatenate([x_ev, np.ones((len(x_ev), 1), dtype=np.float64)], axis=1)
    y = np.asarray(y_train, dtype=np.float64)
    gram = x_tr.T @ x_tr
    reg = float(l2) * np.eye(gram.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    rhs = x_tr.T @ y
    try:
        coef = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram + reg) @ rhs
    return x_tr @ coef, x_ev @ coef


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _top_decile_mask(score: np.ndarray) -> np.ndarray:
    if len(score) == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(np.percentile(score, 90.0))
    return np.asarray(score, dtype=np.float64) >= threshold


def _chronological_split(mask: np.ndarray, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask)
    if len(idx) < 20:
        return idx[:0], idx[:0]
    cut = max(10, min(len(idx) - 10, int(len(idx) * float(train_frac))))
    return idx[:cut], idx[cut:]


def _evaluate_binary(
    *,
    x: np.ndarray,
    y: np.ndarray,
    sample_mask: np.ndarray,
    train_frac: float,
    l2: float,
) -> dict:
    valid = sample_mask & np.isfinite(y)
    train_idx, eval_idx = _chronological_split(valid, train_frac)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return {"samples": int(np.sum(valid)), "eval_samples": 0, "auc": float("nan"), "pr_auc": float("nan")}
    y_train = np.asarray(y[train_idx], dtype=np.float64)
    y_eval = np.asarray(y[eval_idx], dtype=np.float64)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
        return {
            "samples": int(np.sum(valid)),
            "eval_samples": int(len(eval_idx)),
            "positive_rate": float(np.mean(y[valid])) if np.any(valid) else float("nan"),
            "eval_positive_rate": float(np.mean(y_eval)) if len(y_eval) else float("nan"),
            "auc": float("nan"),
            "pr_auc": float("nan"),
        }
    train_score, eval_score = _ridge_predict(x[train_idx], y_train, x[eval_idx], l2=l2)
    train_scale = float(np.std(train_score))
    if train_scale < 1e-8:
        train_scale = 1.0
    eval_prob = _sigmoid((eval_score - float(np.mean(train_score))) / train_scale)
    top_mask = _top_decile_mask(eval_score)
    return {
        "samples": int(np.sum(valid)),
        "train_samples": int(len(train_idx)),
        "eval_samples": int(len(eval_idx)),
        "positive_rate": float(np.mean(y[valid])),
        "eval_positive_rate": float(np.mean(y_eval)),
        "auc": _roc_auc(y_eval, eval_score),
        "pr_auc": _average_precision(y_eval, eval_score),
        "top_decile_positive_rate": float(np.mean(y_eval[top_mask])) if np.any(top_mask) else float("nan"),
        "brier": float(np.mean((eval_prob - y_eval) ** 2)),
    }


def _evaluate_regression(
    *,
    x: np.ndarray,
    y: np.ndarray,
    sample_mask: np.ndarray,
    train_frac: float,
    l2: float,
) -> dict:
    valid = sample_mask & np.isfinite(y)
    train_idx, eval_idx = _chronological_split(valid, train_frac)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return {"samples": int(np.sum(valid)), "eval_samples": 0}
    _train_score, eval_score = _ridge_predict(x[train_idx], y[train_idx], x[eval_idx], l2=l2)
    y_eval = np.asarray(y[eval_idx], dtype=np.float64)
    top_mask = _top_decile_mask(eval_score)
    bottom_mask = eval_score <= float(np.percentile(eval_score, 10.0))
    corr = float(np.corrcoef(eval_score, y_eval)[0, 1]) if np.std(eval_score) > 1e-12 and np.std(y_eval) > 1e-12 else float("nan")
    return {
        "samples": int(np.sum(valid)),
        "train_samples": int(len(train_idx)),
        "eval_samples": int(len(eval_idx)),
        "mean": float(np.mean(y[valid])),
        "eval_mean": float(np.mean(y_eval)),
        "corr": corr,
        "top_decile_mean": float(np.mean(y_eval[top_mask])) if np.any(top_mask) else float("nan"),
        "bottom_decile_mean": float(np.mean(y_eval[bottom_mask])) if np.any(bottom_mask) else float("nan"),
        "top_bottom_spread": (
            float(np.mean(y_eval[top_mask]) - np.mean(y_eval[bottom_mask]))
            if np.any(top_mask) and np.any(bottom_mask)
            else float("nan")
        ),
        "top_decile_positive_rate": float(np.mean(y_eval[top_mask] > 0.0)) if np.any(top_mask) else float("nan"),
    }


def _summary_metrics(
    *,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    fire_eps: float,
) -> dict:
    t = min(len(positions), len(no_adapter), len(returns))
    pos = np.asarray(positions[:t], dtype=np.float64)
    base = np.asarray(no_adapter[:t], dtype=np.float64)
    rets = np.asarray(returns[:t], dtype=np.float64)
    delta = pos - base
    fire = np.abs(delta) > float(fire_eps)
    metrics = Backtest(
        rets,
        pos,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=_benchmark_positions(t, cfg),
    ).run()
    stats = _action_stats(pos, benchmark_position=benchmark_position)
    pnl = compute_pnl(
        rets,
        pos,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
    )
    return {
        "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
        "sharpe_delta": float(metrics.sharpe_delta or 0.0),
        "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
        "turnover": float(stats["turnover"]),
        "long": float(stats["long"]),
        "short": float(stats["short"]),
        "flat": float(stats["flat"]),
        "fire_rate": float(np.mean(fire)),
        "fire_count": int(np.sum(fire)),
        "mean_delta": float(np.mean(delta[fire])) if np.any(fire) else 0.0,
        "fire_pnl": float(np.sum(pnl[fire])) if np.any(fire) else 0.0,
        "nonfire_pnl": float(np.sum(pnl[~fire])) if np.any(~fire) else 0.0,
    }


def _evaluate_run_fold(
    *,
    run: ProbeRun,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    probe_cfg: LabelProbeConfig,
    device: str,
) -> dict:
    payload = _load_actor_for_run(
        run=run,
        split=split,
        features_df=features_df,
        raw_returns=raw_returns,
        cfg=cfg,
        device=device,
    )
    actor = payload["actor"]
    enc = payload["enc_test"]
    returns = np.asarray(payload["test_returns"], dtype=np.float64)
    regime = payload["test_regime_probs"]
    advantage = payload["test_advantage_values"]
    costs_cfg = payload["costs_cfg"]
    benchmark_position = float(payload["benchmark_position"])
    positions = actor.predict_positions(
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
    )
    no_adapter = predict_with_policy_flags(
        actor,
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
        use_floor=bool(getattr(actor, "use_benchmark_exposure_floor", False)),
        use_adapter=False,
    )
    t = min(len(returns), len(positions), len(no_adapter))
    returns = returns[:t]
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    fire = np.abs(positions - no_adapter) > float(probe_cfg.fire_eps)
    x = _build_feature_matrix(
        enc=enc,
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        fire=fire,
        regime=regime,
        advantage=advantage,
        costs_cfg=costs_cfg,
        max_z_dim=probe_cfg.max_z_dim,
    )
    labels = _build_labels(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        cfg=probe_cfg,
        costs_cfg=costs_cfg,
    )
    summary = _summary_metrics(
        positions=positions,
        no_adapter=no_adapter,
        returns=returns,
        cfg=payload["cfg"],
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
        fire_eps=probe_cfg.fire_eps,
    )
    horizon_metrics: dict[str, dict] = {}
    for horizon, record in labels.items():
        valid = np.asarray(record["valid"], dtype=bool)
        sample = valid & fire
        horizon_metrics[str(horizon)] = {
            "sample_count": int(np.sum(sample)),
            "fire_advantage": _evaluate_regression(
                x=x,
                y=record["fire_advantage"],
                sample_mask=sample,
                train_frac=probe_cfg.train_frac,
                l2=probe_cfg.ridge_l2,
            ),
            "fire_harm": _evaluate_binary(
                x=x,
                y=record["fire_harm"],
                sample_mask=sample,
                train_frac=probe_cfg.train_frac,
                l2=probe_cfg.ridge_l2,
            ),
            "drawdown_worsening": _evaluate_binary(
                x=x,
                y=record["drawdown_worsening"],
                sample_mask=sample,
                train_frac=probe_cfg.train_frac,
                l2=probe_cfg.ridge_l2,
            ),
            "trough_exit": _evaluate_binary(
                x=x,
                y=record["trough_exit"],
                sample_mask=sample,
                train_frac=probe_cfg.train_frac,
                l2=probe_cfg.ridge_l2,
            ),
            "label_base": {
                "mean_fire_advantage": float(np.nanmean(record["fire_advantage"][sample])) if np.any(sample) else float("nan"),
                "mean_fire_harm_margin": float(np.nanmean(record["fire_harm_margin"][sample])) if np.any(sample) else float("nan"),
                "mean_future_drawdown": float(np.nanmean(record["future_drawdown"][sample])) if np.any(sample) else float("nan"),
            },
        }
    return {
        "label": run.label,
        "mode": "ac" if run.use_ac else "bc",
        "checkpoint_dir": run.checkpoint_dir,
        "fold": int(split.fold_idx),
        "summary": summary,
        "horizons": horizon_metrics,
    }


def _format_float(value: float, digits: int = 3, signed: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return "nan"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}f}"


def _pass_flags(records: list[dict]) -> dict:
    flags = {
        "fire_harm_auc_ge_0_58": [],
        "drawdown_worsening_auc_ge_0_58": [],
        "trough_exit_auc_ge_0_55": [],
        "fire_advantage_top_decile_positive": [],
    }
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            flags["fire_harm_auc_ge_0_58"].append(float(metrics["fire_harm"].get("auc", float("nan"))) >= 0.58)
            flags["drawdown_worsening_auc_ge_0_58"].append(
                float(metrics["drawdown_worsening"].get("auc", float("nan"))) >= 0.58
            )
            flags["trough_exit_auc_ge_0_55"].append(float(metrics["trough_exit"].get("auc", float("nan"))) >= 0.55)
            flags["fire_advantage_top_decile_positive"].append(
                float(metrics["fire_advantage"].get("top_decile_mean", float("nan"))) > 0.0
            )
    return {key: bool(vals) and all(vals) for key, vals in flags.items()}


def _write_markdown(path: str, *, records: list[dict], args: argparse.Namespace, probe_cfg: LabelProbeConfig) -> None:
    lines = [
        "# Plan14 Fire-Control Label Probe",
        "",
        "## Setup",
        "",
        f"- config: `{args.config}`",
        f"- folds: `{args.folds}`",
        f"- horizons: `{','.join(str(h) for h in probe_cfg.horizons)}`",
        f"- sample: adapter fire bars only, chronological {probe_cfg.train_frac:.0%}/{1.0 - probe_cfg.train_frac:.0%} probe split",
        f"- labels: fire_harm, drawdown_worsening, trough_exit, fire_advantage",
        "",
        "## Policy Summary",
        "",
        "| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        s = rec["summary"]
        lines.append(
            f"| {rec['label']} | {rec['fold']} | {_format_float(s['alpha_excess_pt'], 2, True)} | "
            f"{_format_float(s['sharpe_delta'], 3, True)} | {_format_float(s['maxdd_delta_pt'], 2, True)} | "
            f"{_format_float(s['turnover'], 2)} | {s['long']:.1%} | {s['short']:.1%} | {s['flat']:.1%} | "
            f"{s['fire_rate']:.1%}/{s['fire_count']} | {_format_float(s['fire_pnl'], 4, True)} |"
        )
    lines += [
        "",
        "## Label Quality",
        "",
        "| run | fold | h | fire samples | harm AUC | harm PR | harm top10 | DD worse AUC | DD worse PR | trough AUC | trough PR | adv corr | adv top10 | adv spread |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            harm = metrics["fire_harm"]
            dd = metrics["drawdown_worsening"]
            trough = metrics["trough_exit"]
            adv = metrics["fire_advantage"]
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {horizon} | {metrics['sample_count']} | "
                f"{_format_float(harm.get('auc', float('nan')))} | {_format_float(harm.get('pr_auc', float('nan')))} | "
                f"{_format_float(harm.get('top_decile_positive_rate', float('nan')))} | "
                f"{_format_float(dd.get('auc', float('nan')))} | {_format_float(dd.get('pr_auc', float('nan')))} | "
                f"{_format_float(trough.get('auc', float('nan')))} | {_format_float(trough.get('pr_auc', float('nan')))} | "
                f"{_format_float(adv.get('corr', float('nan')))} | {_format_float(adv.get('top_decile_mean', float('nan')), 5, True)} | "
                f"{_format_float(adv.get('top_bottom_spread', float('nan')), 5, True)} |"
            )
    pass_flags = _pass_flags(records)
    lines += [
        "",
        "## Gate Readiness Check",
        "",
        "| criterion | pass |",
        "|---|---:|",
    ]
    for key, value in pass_flags.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `fire_harm` and `drawdown_worsening` are useful only if AUC is reproducibly above the threshold across folds/horizons.",
        "- `fire_advantage` is useful only if the predicted top decile has positive realized adapter advantage.",
        "- If these fail, adding WM heads or AC freedom is not justified; the direct labels are not separable enough yet.",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.fire_control_label_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="5")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="checkpoints/data_cache")
    parser.add_argument("--run", action="append", required=True, help="label=checkpoint_dir[@ac_file][:ac|:bc]")
    parser.add_argument("--horizons", default="16,32")
    parser.add_argument("--fire-eps", type=float, default=1e-6)
    parser.add_argument("--harm-margin", type=float, default=0.00025)
    parser.add_argument("--drawdown-worsen-margin", type=float, default=0.003)
    parser.add_argument("--trough-rebound-margin", type=float, default=0.0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--ridge-l2", type=float, default=1e-2)
    parser.add_argument("--max-z-dim", type=int, default=128)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
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
        cache_dir=args.cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.folds))
    if not splits:
        raise RuntimeError(f"No folds matched: {args.folds}")
    runs = [_parse_run(spec) for spec in args.run]
    probe_cfg = LabelProbeConfig(
        horizons=tuple(int(x) for x in args.horizons.split(",") if x.strip()),
        fire_eps=float(args.fire_eps),
        harm_margin=float(args.harm_margin),
        drawdown_worsen_margin=float(args.drawdown_worsen_margin),
        trough_rebound_margin=float(args.trough_rebound_margin),
        train_frac=float(args.train_frac),
        ridge_l2=float(args.ridge_l2),
        max_z_dim=int(args.max_z_dim),
    )
    records: list[dict] = []
    for split in splits:
        for run in runs:
            try:
                print(f"[Plan14] run={run.label} fold={split.fold_idx} checkpoint={run.checkpoint_dir}")
                records.append(
                    _evaluate_run_fold(
                        run=run,
                        split=split,
                        features_df=features_df,
                        raw_returns=raw_returns,
                        cfg=cfg,
                        probe_cfg=probe_cfg,
                        device=args.device,
                    )
                )
            except FileNotFoundError:
                if not args.skip_missing:
                    raise
                print(f"[Plan14] skip missing run={run.label} fold={split.fold_idx}")
    serializable = {
        "config": args.config,
        "folds": args.folds,
        "runs": [run.__dict__ for run in runs],
        "probe": probe_cfg.__dict__,
        "records": records,
        "gate_readiness": _pass_flags(records),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, allow_nan=True)
    _write_markdown(args.output_md, records=records, args=args, probe_cfg=probe_cfg)
    print(f"[Plan14] wrote {args.output_md}")


if __name__ == "__main__":
    main()
