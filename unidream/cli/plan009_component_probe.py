"""Plan009 component probe: isolate scalable changes before full MBRL retraining.

This probe intentionally avoids training the full Transformer WM/BC/AC stack.
It tests whether the proposed Plan009 ingredients have enough signal to scale:

- action-conditioned residual utility targets
- pessimistic/uncertainty-gated residual action selection
- future drawdown danger labels for small risk-off overlays
- pullback recovery labels for small overweight overlays
- simple val-only policy selection, without test-period selection
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle,
    _backtest_positions,
    _candidate_utilities,
    _fit_bootstrap_models,
    _fit_ridge_multi,
    _future_windows,
    _positions_from_prediction,
    _sanitize_feature_matrix,
    _shift_for_execution,
    _state_features,
)
from unidream.cli.market_event_label_probe import _path_max_drawdown
from unidream.cli.plan5_laneF import make_pullback_recovery_label
from unidream.cli.route_separability_probe import _fit_binary_model, _score_binary
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class Candidate:
    name: str
    positions: np.ndarray
    diag: dict[str, Any]
    val_metrics: dict[str, Any]
    score: float


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _sample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    if n <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(max_n), replace=False))


def _future_benchmark_drawdown_label(
    returns: np.ndarray,
    *,
    horizon: int,
    threshold: float,
    benchmark_position: float,
) -> np.ndarray:
    windows, valid = _future_windows(returns, horizon)
    label = np.zeros(len(returns), dtype=np.int64)
    if len(windows) == 0:
        return label
    dd = _path_max_drawdown(windows, float(benchmark_position))
    label[np.flatnonzero(valid)[: len(dd)]] = (dd >= float(threshold)).astype(np.int64)
    return label


def _binary_auc_ap(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    yy = np.asarray(y, dtype=np.int64)
    ss = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(ss)
    yy = yy[mask]
    ss = ss[mask]
    if len(yy) < 20 or len(np.unique(yy)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "density": float(np.mean(yy)) if len(yy) else 0.0}
    return {
        "auc": float(roc_auc_score(yy, ss)),
        "ap": float(average_precision_score(yy, ss)),
        "density": float(np.mean(yy)),
    }


def _fit_hgb_regressors(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_train_samples: int,
) -> list[Any] | None:
    x_clean = _sanitize_feature_matrix(x)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.all(np.isfinite(x_clean), axis=1) & np.all(np.isfinite(y_arr), axis=1)
    idx = np.flatnonzero(mask)
    if len(idx) < 500:
        return None
    idx = _sample_idx(len(idx), int(max_train_samples), seed)
    true_idx = np.flatnonzero(mask)[idx]
    models: list[Any] = []
    for j in range(y_arr.shape[1]):
        model = make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=80,
                max_leaf_nodes=15,
                min_samples_leaf=80,
                l2_regularization=0.05,
                random_state=seed + j,
            ),
        )
        model.fit(x_clean[true_idx], y_arr[true_idx, j])
        models.append(model)
    return models


def _predict_hgb(models: list[Any] | None, x: np.ndarray) -> np.ndarray | None:
    if not models:
        return None
    x_clean = _sanitize_feature_matrix(x)
    return np.stack([m.predict(x_clean) for m in models], axis=1).astype(np.float64)


def _fit_hgb_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_train_samples: int,
) -> Any | None:
    x_clean = _sanitize_feature_matrix(x)
    y_arr = np.asarray(y, dtype=np.int64)
    mask = np.all(np.isfinite(x_clean), axis=1) & np.isfinite(y_arr)
    idx = np.flatnonzero(mask)
    if len(idx) < 500 or len(np.unique(y_arr[idx])) < 2:
        return None
    idx = _sample_idx(len(idx), int(max_train_samples), seed)
    true_idx = np.flatnonzero(mask)[idx]
    model = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=0.05,
            random_state=seed,
        ),
    )
    model.fit(x_clean[true_idx], y_arr[true_idx])
    return model


def _score_classifier(model: Any | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), np.nan, dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(_sanitize_feature_matrix(x))[:, -1].astype(np.float64)
    return np.asarray(model.decision_function(_sanitize_feature_matrix(x)), dtype=np.float64)


def _threshold_candidates(score: np.ndarray, extra: tuple[float, ...] = ()) -> list[float]:
    finite = np.asarray(score, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return [float("inf")]
    qs = (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.975, 0.99)
    values = [float(np.quantile(finite, q)) for q in qs]
    values.extend(float(x) for x in extra)
    values.append(float("inf"))
    return sorted(set(values))


def _select_policy(
    *,
    name: str,
    desired_val: np.ndarray,
    desired_test_builder,
    val_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    thresholds: list[float],
    raw_score_val: np.ndarray,
    hold_grid: tuple[int, ...],
    cooldown_grid: tuple[int, ...],
    turnover_cap: float,
    active_cap: float,
) -> Candidate:
    best: Candidate | None = None
    bench = float(benchmark_position)
    raw_score_val = np.asarray(raw_score_val, dtype=np.float64)
    for threshold in thresholds:
        active = np.isfinite(raw_score_val) & (raw_score_val > float(threshold))
        selected_val = np.full(len(desired_val), bench, dtype=np.float64)
        selected_val[active] = desired_val[active]
        for hold in hold_grid:
            for cooldown in cooldown_grid:
                throttled = _apply_event_throttle(
                    selected_val,
                    benchmark_position=bench,
                    hold_bars=int(hold),
                    cooldown_bars=int(cooldown),
                )
                exec_val = _shift_for_execution(throttled, bench)
                metrics, _ = _backtest_positions(
                    val_returns,
                    exec_val,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=bench,
                )
                active_rate = 1.0 - float(metrics.get("flat_rate", 1.0))
                turnover = float(metrics.get("turnover", 999.0))
                maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
                alpha = float(metrics.get("alpha_excess_pt", 0.0))
                sharpe = float(metrics.get("sharpe_delta", 0.0))
                if active_rate > float(active_cap):
                    score = -1_000_000.0 - 1000.0 * (active_rate - float(active_cap))
                elif turnover > float(turnover_cap):
                    score = -1_000_000.0 - 100.0 * (turnover - float(turnover_cap))
                elif turnover <= 1e-12 and active_rate <= 1e-12:
                    score = -1.0
                else:
                    score = alpha + 8.0 * sharpe + 0.75 * max(0.0, -maxdd)
                    score -= 5.0 * max(0.0, maxdd)
                    score -= 0.10 * turnover
                diag = {
                    "threshold": float(threshold),
                    "hold_bars": int(hold),
                    "cooldown_bars": int(cooldown),
                    "val_active_rate": float(active_rate),
                    "val_score": float(score),
                }
                candidate = Candidate(name=name, positions=exec_val, diag=diag, val_metrics=metrics, score=float(score))
                if best is None or candidate.score > best.score:
                    best = candidate
    assert best is not None
    test_positions, test_diag = desired_test_builder(best.diag)
    merged_diag = dict(best.diag)
    merged_diag.update(test_diag)
    return Candidate(
        name=name,
        positions=test_positions,
        diag=merged_diag,
        val_metrics=best.val_metrics,
        score=best.score,
    )


def _policy_from_pred(
    *,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    candidates: tuple[float, ...],
    variant_name: str,
    val_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    hold_grid: tuple[int, ...],
    cooldown_grid: tuple[int, ...],
    turnover_cap: float,
    active_cap: float,
) -> Candidate:
    bench = float(benchmark_position)
    cand = np.asarray(candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(cand - bench)))
    best_idx_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_idx_val] - pred_val[:, bench_idx]
    desired_val = np.full(len(pred_val), bench, dtype=np.float64)
    desired_val[:] = cand[best_idx_val]

    def _build_test(diag: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        selected, pd = _positions_from_prediction(
            pred_test,
            candidates=candidates,
            threshold=float(diag["threshold"]),
            benchmark_position=bench,
        )
        throttled = _apply_event_throttle(
            selected,
            benchmark_position=bench,
            hold_bars=int(diag["hold_bars"]),
            cooldown_bars=int(diag["cooldown_bars"]),
        )
        return _shift_for_execution(throttled, bench), pd

    return _select_policy(
        name=variant_name,
        desired_val=desired_val,
        desired_test_builder=_build_test,
        val_returns=val_returns,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=bench,
        thresholds=_threshold_candidates(improve_val, extra=(0.0,)),
        raw_score_val=improve_val,
        hold_grid=hold_grid,
        cooldown_grid=cooldown_grid,
        turnover_cap=turnover_cap,
        active_cap=active_cap,
    )


def _policy_from_score(
    *,
    score_val: np.ndarray,
    score_test: np.ndarray,
    position: float,
    variant_name: str,
    val_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    hold_grid: tuple[int, ...],
    cooldown_grid: tuple[int, ...],
    turnover_cap: float,
    active_cap: float,
) -> Candidate:
    bench = float(benchmark_position)
    desired_val = np.full(len(score_val), float(position), dtype=np.float64)

    def _build_test(diag: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        active = np.isfinite(score_test) & (score_test > float(diag["threshold"]))
        selected = np.full(len(score_test), bench, dtype=np.float64)
        selected[active] = float(position)
        throttled = _apply_event_throttle(
            selected,
            benchmark_position=bench,
            hold_bars=int(diag["hold_bars"]),
            cooldown_bars=int(diag["cooldown_bars"]),
        )
        return _shift_for_execution(throttled, bench), {
            "raw_active_rate": float(np.mean(active)),
            "score_mean": float(np.nanmean(score_test)),
        }

    return _select_policy(
        name=variant_name,
        desired_val=desired_val,
        desired_test_builder=_build_test,
        val_returns=val_returns,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=bench,
        thresholds=_threshold_candidates(score_val),
        raw_score_val=score_val,
        hold_grid=hold_grid,
        cooldown_grid=cooldown_grid,
        turnover_cap=turnover_cap,
        active_cap=active_cap,
    )


def _combined_policy(
    *,
    danger_val: np.ndarray,
    danger_test: np.ndarray,
    recovery_val: np.ndarray,
    recovery_test: np.ndarray,
    down_position: float,
    up_position: float,
    val_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    hold_grid: tuple[int, ...],
    cooldown_grid: tuple[int, ...],
    turnover_cap: float,
    active_cap: float,
) -> Candidate:
    bench = float(benchmark_position)
    best: Candidate | None = None
    for d_thr in _threshold_candidates(danger_val):
        for r_thr in _threshold_candidates(recovery_val):
            raw_active = np.maximum(danger_val - d_thr, recovery_val - r_thr)
            desired = np.full(len(danger_val), bench, dtype=np.float64)
            rec = np.isfinite(recovery_val) & (recovery_val > r_thr)
            dng = np.isfinite(danger_val) & (danger_val > d_thr)
            desired[rec] = float(up_position)
            desired[dng] = float(down_position)
            for hold in hold_grid:
                for cooldown in cooldown_grid:
                    throttled = _apply_event_throttle(
                        desired,
                        benchmark_position=bench,
                        hold_bars=int(hold),
                        cooldown_bars=int(cooldown),
                    )
                    exec_val = _shift_for_execution(throttled, bench)
                    metrics, _ = _backtest_positions(
                        val_returns,
                        exec_val,
                        cfg=cfg,
                        costs_cfg=costs_cfg,
                        benchmark_position=bench,
                    )
                    active_rate = 1.0 - float(metrics.get("flat_rate", 1.0))
                    turnover = float(metrics.get("turnover", 999.0))
                    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
                    alpha = float(metrics.get("alpha_excess_pt", 0.0))
                    sharpe = float(metrics.get("sharpe_delta", 0.0))
                    if active_rate > float(active_cap):
                        score = -1_000_000.0 - 1000.0 * (active_rate - float(active_cap))
                    elif turnover > float(turnover_cap):
                        score = -1_000_000.0 - 100.0 * (turnover - float(turnover_cap))
                    elif turnover <= 1e-12 and active_rate <= 1e-12:
                        score = -1.0
                    else:
                        score = alpha + 8.0 * sharpe + 0.75 * max(0.0, -maxdd)
                        score -= 5.0 * max(0.0, maxdd)
                        score -= 0.10 * turnover
                    candidate = Candidate(
                        name="label_combined",
                        positions=exec_val,
                        diag={
                            "danger_threshold": float(d_thr),
                            "recovery_threshold": float(r_thr),
                            "hold_bars": int(hold),
                            "cooldown_bars": int(cooldown),
                            "val_active_rate": float(active_rate),
                            "val_score": float(score),
                        },
                        val_metrics=metrics,
                        score=float(score),
                    )
                    if best is None or candidate.score > best.score:
                        best = candidate
    assert best is not None
    rec_t = np.isfinite(recovery_test) & (recovery_test > float(best.diag["recovery_threshold"]))
    dng_t = np.isfinite(danger_test) & (danger_test > float(best.diag["danger_threshold"]))
    selected = np.full(len(danger_test), bench, dtype=np.float64)
    selected[rec_t] = float(up_position)
    selected[dng_t] = float(down_position)
    throttled = _apply_event_throttle(
        selected,
        benchmark_position=bench,
        hold_bars=int(best.diag["hold_bars"]),
        cooldown_bars=int(best.diag["cooldown_bars"]),
    )
    exec_test = _shift_for_execution(throttled, bench)
    diag = dict(best.diag)
    diag.update(
        {
            "raw_active_rate": float(np.mean(rec_t | dng_t)),
            "raw_recovery_rate": float(np.mean(rec_t)),
            "raw_danger_rate": float(np.mean(dng_t)),
        }
    )
    return Candidate(name="label_combined", positions=exec_test, diag=diag, val_metrics=best.val_metrics, score=best.score)


def _run_fold(
    *,
    fold,
    features_df,
    raw_returns,
    cfg: dict,
    costs_cfg: dict,
    args,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ds = WFODataset(features_df, raw_returns, fold, seq_len=int(cfg.get("data", {}).get("seq_len", 64)))
    x_train = _state_features(ds.train_features, ds.train_returns)
    x_val = _state_features(ds.val_features, ds.val_returns)
    x_test = _state_features(ds.test_features, ds.test_returns)

    bench = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    costs = (
        float(costs_cfg.get("spread_bps", 5.0)) / 10000.0 / 2.0
        + float(costs_cfg.get("fee_rate", 0.0004))
        + float(costs_cfg.get("slippage_bps", 2.0)) / 10000.0
    )
    candidates = tuple(float(x) for x in args.candidates.split(","))
    horizon = int(args.horizon)
    seed = int(args.seed) + int(fold.fold_idx) * 17

    y_train, _ = _candidate_utilities(
        ds.train_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=bench,
        unit_cost=costs,
        dd_penalty=float(args.dd_penalty),
        vol_penalty=float(args.vol_penalty),
    )
    y_val, _ = _candidate_utilities(
        ds.val_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=bench,
        unit_cost=costs,
        dd_penalty=float(args.dd_penalty),
        vol_penalty=float(args.vol_penalty),
    )
    y_test, _ = _candidate_utilities(
        ds.test_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=bench,
        unit_cost=costs,
        dd_penalty=float(args.dd_penalty),
        vol_penalty=float(args.vol_penalty),
    )

    diagnostics: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    hold_grid = tuple(int(x) for x in args.hold_grid.split(","))
    cooldown_grid = tuple(int(x) for x in args.cooldown_grid.split(","))

    ridge = _fit_ridge_multi(x_train, y_train, l2=float(args.ridge_l2))
    if ridge is not None:
        pred_val = ridge.predict(x_val)
        pred_test = ridge.predict(x_test)
        policy = _policy_from_pred(
            pred_val=pred_val,
            pred_test=pred_test,
            candidates=candidates,
            variant_name="utility_ridge",
            val_returns=ds.val_returns,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=bench,
            hold_grid=hold_grid,
            cooldown_grid=cooldown_grid,
            turnover_cap=float(args.turnover_cap),
            active_cap=float(args.active_cap),
        )
        metrics, _ = _backtest_positions(ds.test_returns, policy.positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bench)
        rows.append({"fold": fold.fold_idx, "variant": policy.name, "test": metrics, "val": policy.val_metrics, "selection": policy.diag})
        diagnostics.append(_utility_diag(fold.fold_idx, "utility_ridge", pred_val, y_val, pred_test, y_test, candidates, bench))

    boot = _fit_bootstrap_models(
        x_train,
        y_train,
        l2=float(args.ridge_l2),
        seed=seed,
        n_models=int(args.bootstrap_models),
    )
    if boot:
        val_preds = np.stack([m.predict(x_val) for m in boot], axis=0)
        test_preds = np.stack([m.predict(x_test) for m in boot], axis=0)
        for penalty in (0.5, 1.0):
            pred_val = val_preds.mean(axis=0) - penalty * val_preds.std(axis=0)
            pred_test = test_preds.mean(axis=0) - penalty * test_preds.std(axis=0)
            policy = _policy_from_pred(
                pred_val=pred_val,
                pred_test=pred_test,
                candidates=candidates,
                variant_name=f"utility_pessimistic_k{penalty:g}",
                val_returns=ds.val_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=bench,
                hold_grid=hold_grid,
                cooldown_grid=cooldown_grid,
                turnover_cap=float(args.turnover_cap),
                active_cap=float(args.active_cap),
            )
            metrics, _ = _backtest_positions(ds.test_returns, policy.positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bench)
            rows.append({"fold": fold.fold_idx, "variant": policy.name, "test": metrics, "val": policy.val_metrics, "selection": policy.diag})
            diagnostics.append(_utility_diag(fold.fold_idx, policy.name, pred_val, y_val, pred_test, y_test, candidates, bench))

    if bool(args.hgb):
        hgb = _fit_hgb_regressors(x_train, y_train, seed=seed, max_train_samples=int(args.max_train_samples))
        pred_val = _predict_hgb(hgb, x_val)
        pred_test = _predict_hgb(hgb, x_test)
        if pred_val is not None and pred_test is not None:
            policy = _policy_from_pred(
                pred_val=pred_val,
                pred_test=pred_test,
                candidates=candidates,
                variant_name="utility_hgb",
                val_returns=ds.val_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=bench,
                hold_grid=hold_grid,
                cooldown_grid=cooldown_grid,
                turnover_cap=float(args.turnover_cap),
                active_cap=float(args.active_cap),
            )
            metrics, _ = _backtest_positions(ds.test_returns, policy.positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bench)
            rows.append({"fold": fold.fold_idx, "variant": policy.name, "test": metrics, "val": policy.val_metrics, "selection": policy.diag})
            diagnostics.append(_utility_diag(fold.fold_idx, "utility_hgb", pred_val, y_val, pred_test, y_test, candidates, bench))

    danger_train = _future_benchmark_drawdown_label(
        ds.train_returns,
        horizon=int(args.danger_horizon),
        threshold=float(args.danger_threshold),
        benchmark_position=bench,
    )
    danger_val = _future_benchmark_drawdown_label(
        ds.val_returns,
        horizon=int(args.danger_horizon),
        threshold=float(args.danger_threshold),
        benchmark_position=bench,
    )
    danger_test = _future_benchmark_drawdown_label(
        ds.test_returns,
        horizon=int(args.danger_horizon),
        threshold=float(args.danger_threshold),
        benchmark_position=bench,
    )
    danger_model = _fit_binary_model(x_train, danger_train, max_train_samples=int(args.max_train_samples), seed=seed)
    danger_score_val = _score_binary(danger_model, x_val)
    danger_score_test = _score_binary(danger_model, x_test)
    diagnostics.append(
        {
            "fold": fold.fold_idx,
            "variant": "label_danger_logistic",
            "val_label": _binary_auc_ap(danger_val, danger_score_val),
            "test_label": _binary_auc_ap(danger_test, danger_score_test),
        }
    )
    policy = _policy_from_score(
        score_val=danger_score_val,
        score_test=danger_score_test,
        position=float(args.down_position),
        variant_name="danger_riskoff_logistic",
        val_returns=ds.val_returns,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=bench,
        hold_grid=hold_grid,
        cooldown_grid=cooldown_grid,
        turnover_cap=float(args.turnover_cap),
        active_cap=float(args.active_cap),
    )
    metrics, _ = _backtest_positions(ds.test_returns, policy.positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bench)
    rows.append({"fold": fold.fold_idx, "variant": policy.name, "test": metrics, "val": policy.val_metrics, "selection": policy.diag})

    recovery_train = make_pullback_recovery_label(ds.train_returns, horizon=int(args.recovery_horizon))
    recovery_val = make_pullback_recovery_label(ds.val_returns, horizon=int(args.recovery_horizon))
    recovery_test = make_pullback_recovery_label(ds.test_returns, horizon=int(args.recovery_horizon))
    recovery_model = _fit_binary_model(x_train, recovery_train, max_train_samples=int(args.max_train_samples), seed=seed + 1)
    recovery_score_val = _score_binary(recovery_model, x_val)
    recovery_score_test = _score_binary(recovery_model, x_test)
    diagnostics.append(
        {
            "fold": fold.fold_idx,
            "variant": "label_recovery_logistic",
            "val_label": _binary_auc_ap(recovery_val, recovery_score_val),
            "test_label": _binary_auc_ap(recovery_test, recovery_score_test),
        }
    )
    policy = _policy_from_score(
        score_val=recovery_score_val,
        score_test=recovery_score_test,
        position=float(args.up_position),
        variant_name="recovery_overweight_logistic",
        val_returns=ds.val_returns,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=bench,
        hold_grid=hold_grid,
        cooldown_grid=cooldown_grid,
        turnover_cap=float(args.turnover_cap),
        active_cap=float(args.active_cap),
    )
    metrics, _ = _backtest_positions(ds.test_returns, policy.positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bench)
    rows.append({"fold": fold.fold_idx, "variant": policy.name, "test": metrics, "val": policy.val_metrics, "selection": policy.diag})

    if bool(args.hgb):
        danger_hgb = _fit_hgb_classifier(x_train, danger_train, seed=seed + 2, max_train_samples=int(args.max_train_samples))
        recovery_hgb = _fit_hgb_classifier(x_train, recovery_train, seed=seed + 3, max_train_samples=int(args.max_train_samples))
        d_val = _score_classifier(danger_hgb, x_val)
        d_test = _score_classifier(danger_hgb, x_test)
        r_val = _score_classifier(recovery_hgb, x_val)
        r_test = _score_classifier(recovery_hgb, x_test)
        diagnostics.append(
            {
                "fold": fold.fold_idx,
                "variant": "label_danger_hgb",
                "val_label": _binary_auc_ap(danger_val, d_val),
                "test_label": _binary_auc_ap(danger_test, d_test),
            }
        )
        diagnostics.append(
            {
                "fold": fold.fold_idx,
                "variant": "label_recovery_hgb",
                "val_label": _binary_auc_ap(recovery_val, r_val),
                "test_label": _binary_auc_ap(recovery_test, r_test),
            }
        )
        policy = _combined_policy(
            danger_val=d_val,
            danger_test=d_test,
            recovery_val=r_val,
            recovery_test=r_test,
            down_position=float(args.down_position),
            up_position=float(args.up_position),
            val_returns=ds.val_returns,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=bench,
            hold_grid=hold_grid,
            cooldown_grid=cooldown_grid,
            turnover_cap=float(args.turnover_cap),
            active_cap=float(args.active_cap),
        )
        metrics, _ = _backtest_positions(ds.test_returns, policy.positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bench)
        rows.append({"fold": fold.fold_idx, "variant": policy.name, "test": metrics, "val": policy.val_metrics, "selection": policy.diag})

    return rows, diagnostics


def _utility_diag(
    fold_idx: int,
    variant: str,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    pred_test: np.ndarray,
    y_test: np.ndarray,
    candidates: tuple[float, ...],
    benchmark_position: float,
) -> dict[str, Any]:
    cand = np.asarray(candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(cand - float(benchmark_position))))

    def _one(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
        pred_best = np.argmax(pred, axis=1)
        pred_improve = pred[np.arange(len(pred)), pred_best] - pred[:, bench_idx]
        truth_improve = truth[np.arange(len(truth)), pred_best] - truth[:, bench_idx]
        mask = np.isfinite(pred_improve) & np.isfinite(truth_improve)
        if int(mask.sum()) < 20:
            return {"top10_realized_improve": float("nan"), "rank_hit_rate": float("nan")}
        top_n = max(1, int(mask.sum() * 0.10))
        idx = np.flatnonzero(mask)
        top_idx = idx[np.argsort(pred_improve[mask])[-top_n:]]
        oracle_best = np.argmax(truth, axis=1)
        return {
            "top10_realized_improve": float(np.nanmean(truth_improve[top_idx])),
            "rank_hit_rate": float(np.mean(pred_best[mask] == oracle_best[mask])),
            "pred_positive_rate": float(np.mean(pred_improve[mask] > 0.0)),
        }

    return {
        "fold": fold_idx,
        "variant": variant,
        "val_utility": _one(pred_val, y_val),
        "test_utility": _one(pred_test, y_test),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({str(r["variant"]) for r in rows})
    out = []
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        tests = [r["test"] for r in subset]
        out.append(
            {
                "variant": variant,
                "folds": len(subset),
                "pass_3m3": int(
                    sum(
                        float(t.get("alpha_excess_pt", 0.0)) >= 3.0
                        and float(t.get("maxdd_delta_pt", 0.0)) <= -3.0
                        for t in tests
                    )
                ),
                "alpha_mean": float(np.mean([float(t.get("alpha_excess_pt", 0.0)) for t in tests])) if tests else 0.0,
                "alpha_median": float(np.median([float(t.get("alpha_excess_pt", 0.0)) for t in tests])) if tests else 0.0,
                "alpha_worst": float(np.min([float(t.get("alpha_excess_pt", 0.0)) for t in tests])) if tests else 0.0,
                "maxdd_mean": float(np.mean([float(t.get("maxdd_delta_pt", 0.0)) for t in tests])) if tests else 0.0,
                "maxdd_best": float(np.min([float(t.get("maxdd_delta_pt", 0.0)) for t in tests])) if tests else 0.0,
                "maxdd_worst": float(np.max([float(t.get("maxdd_delta_pt", 0.0)) for t in tests])) if tests else 0.0,
                "turnover_mean": float(np.mean([float(t.get("turnover", 0.0)) for t in tests])) if tests else 0.0,
            }
        )
    return out


def _write_outputs(path_json: str, path_md: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(path_md) or ".", exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Plan009 Component Probe",
        "",
        "Purpose: rough, no-leak component screening before scaling TransformerMBRL.",
        "",
        "## Aggregate",
        "",
        "| variant | folds | pass +3/-3 | Alpha mean | Alpha median | Alpha worst | MaxDD mean | MaxDD best | MaxDD worst | TO mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['variant']} | {row['folds']} | {row['pass_3m3']}/{row['folds']} | "
            f"{_fmt(row['alpha_mean'])} | {_fmt(row['alpha_median'])} | {_fmt(row['alpha_worst'])} | "
            f"{_fmt(row['maxdd_mean'])} | {_fmt(row['maxdd_best'])} | {_fmt(row['maxdd_worst'])} | "
            f"{_fmt(row['turnover_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Fold Detail",
            "",
            "| fold | variant | AlphaEx | SharpeD | MaxDDD | TO | active | selection |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        t = row["test"]
        active = 1.0 - float(t.get("flat_rate", 1.0))
        sel = row.get("selection", {})
        sel_short = {
            k: sel[k]
            for k in sorted(sel)
            if k
            in {
                "threshold",
                "danger_threshold",
                "recovery_threshold",
                "hold_bars",
                "cooldown_bars",
                "raw_active_rate",
                "val_active_rate",
                "val_score",
            }
        }
        lines.append(
            f"| {row['fold']} | {row['variant']} | {_fmt(t.get('alpha_excess_pt'))} | "
            f"{_fmt(t.get('sharpe_delta'))} | {_fmt(t.get('maxdd_delta_pt'))} | "
            f"{_fmt(t.get('turnover'))} | {_fmt(active)} | `{sel_short}` |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| fold | variant | val signal | test signal |",
            "|---:|---|---|---|",
        ]
    )
    for d in payload["diagnostics"]:
        val = d.get("val_label") or d.get("val_utility") or {}
        test = d.get("test_label") or d.get("test_utility") or {}
        lines.append(f"| {d['fold']} | {d['variant']} | `{val}` | `{test}` |")
    lines.extend(
        [
            "",
            "## Leak Discipline",
            "",
            "- Features are current/past state features plus shifted rolling return features already used in prior probes.",
            "- Utility and label targets use future returns only for train/validation fitting and selection.",
            "- Test targets are used only for diagnostics after policy positions are fixed by train+validation.",
            "- Positions are shifted by one bar before backtest so a bar's decision is not applied to the same bar's return.",
        ]
    )
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan009_component_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--candidates", default="0.94,1.0,1.06")
    parser.add_argument("--up-position", type=float, default=1.06)
    parser.add_argument("--down-position", type=float, default=0.94)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--danger-horizon", type=int, default=64)
    parser.add_argument("--recovery-horizon", type=int, default=32)
    parser.add_argument("--danger-threshold", type=float, default=0.015)
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--vol-penalty", type=float, default=0.25)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--bootstrap-models", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--hold-grid", default="16,32,64,96")
    parser.add_argument("--cooldown-grid", default="0,32,96")
    parser.add_argument("--turnover-cap", type=float, default=6.0)
    parser.add_argument("--active-cap", type=float, default=0.55)
    parser.add_argument("--hgb", action="store_true")
    parser.add_argument("--output-json", default="docs_local/20260527_plan009_component_probe_f456.json")
    parser.add_argument("--output-md", default="docs_local/20260527_plan009_component_probe_f456.md")
    args = parser.parse_args()

    set_seed(int(args.seed))
    cfg = load_config(args.config)
    cfg, _ = resolve_costs(cfg, None)
    dc = cfg.get("data", {})
    zw = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{dc['symbol']}_{dc['interval']}_{args.start}_{args.end}_z{zw}_v2"
    features_df, raw_returns = load_training_features(
        symbol=dc["symbol"],
        interval=dc["interval"],
        start=args.start,
        end=args.end,
        zscore_window=zw,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
    )
    splits, _ = select_wfo_splits(build_wfo_splits(features_df, dc), args.folds)

    all_rows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for split in splits:
        print(f"[Plan009] fold={split.fold_idx}")
        rows, diagnostics = _run_fold(
            fold=split,
            features_df=features_df,
            raw_returns=raw_returns,
            cfg=cfg,
            costs_cfg=cfg.get("costs", {}),
            args=args,
        )
        all_rows.extend(rows)
        all_diagnostics.extend(diagnostics)

    payload = {
        "experiment": "plan009_component_probe",
        "folds": [s.fold_idx for s in splits],
        "args": vars(args),
        "aggregate": _aggregate_rows(all_rows),
        "rows": all_rows,
        "diagnostics": all_diagnostics,
    }
    _write_outputs(args.output_json, args.output_md, payload)
    print(f"[Plan009] wrote {args.output_json}")
    print(f"[Plan009] wrote {args.output_md}")


if __name__ == "__main__":
    main()
