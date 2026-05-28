from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.cli.exploration_board_probe import (
    RidgeModel,
    _fit_ridge_multi,
    _future_vol,
    _future_windows,
    _sanitize_feature_matrix,
    _state_features,
)
from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy
from unidream.research.plan005_meta_guard_bc_ac import (
    _build_past_features,
    guard_positions_from_features,
    select_meta_guard_mode,
)


MODE_NAMES = ("core_pair", "pre_halving_rebound", "deep_bear_recovery")


@dataclass(frozen=True)
class DepthPolicy:
    name: str
    positions: np.ndarray
    val_metrics: dict[str, Any]
    diag: dict[str, Any]
    score: float


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _mask_for(index, start, end, *, inclusive_end: bool) -> np.ndarray:
    if inclusive_end:
        return np.asarray((index >= start) & (index <= end), dtype=bool)
    return np.asarray((index >= start) & (index < end), dtype=bool)


def _segment_start_idx(mask: np.ndarray) -> int:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        raise ValueError("empty segment mask")
    return int(idx[0])


def _unit_cost(costs_cfg: dict[str, Any]) -> float:
    return (
        float(costs_cfg.get("spread_bps", 3.0)) / 10000.0 / 2.0
        + float(costs_cfg.get("fee_rate", 0.0003))
        + float(costs_cfg.get("slippage_bps", 1.0)) / 10000.0
    )


def _rowwise_path_max_drawdown(windows: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if windows.size == 0:
        return np.zeros(0, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
    path = np.cumsum(np.asarray(windows, dtype=np.float64) * pos, axis=1)
    path = np.concatenate([np.zeros((path.shape[0], 1), dtype=np.float64), path], axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    return np.max(peak - path, axis=1)


def _depth_positions(
    base: np.ndarray,
    guard: np.ndarray,
    *,
    depth: float,
    benchmark_position: float,
) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float64)
    guard_arr = np.asarray(guard, dtype=np.float64)[: len(base_arr)]
    bench = float(benchmark_position)
    scaled_guard = bench - float(depth) * (bench - guard_arr)
    return np.minimum(base_arr, scaled_guard)


def _mode_one_hot(mode: str, n: int) -> np.ndarray:
    out = np.zeros((int(n), len(MODE_NAMES)), dtype=np.float64)
    if mode in MODE_NAMES:
        out[:, MODE_NAMES.index(mode)] = 1.0
    return out


def _augment_features(
    x: np.ndarray,
    *,
    base: np.ndarray,
    guard: np.ndarray,
    mode: str,
    benchmark_position: float,
) -> np.ndarray:
    n = min(len(x), len(base), len(guard))
    bench = float(benchmark_position)
    base_arr = np.asarray(base[:n], dtype=np.float64)
    guard_arr = np.asarray(guard[:n], dtype=np.float64)
    gap = bench - guard_arr
    base_overlay = base_arr - bench
    parts = [
        np.asarray(x[:n], dtype=np.float64),
        base_arr.reshape(-1, 1),
        base_overlay.reshape(-1, 1),
        guard_arr.reshape(-1, 1),
        gap.reshape(-1, 1),
        np.abs(np.diff(np.concatenate([[bench], guard_arr]))).reshape(-1, 1),
        _mode_one_hot(mode, n),
    ]
    return _sanitize_feature_matrix(np.concatenate(parts, axis=1))


def _depth_utilities(
    returns: np.ndarray,
    *,
    base_positions: np.ndarray,
    guard_positions: np.ndarray,
    depths: tuple[float, ...],
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    dd_reward: float,
    dd_worsen_penalty: float,
    vol_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    windows, valid = _future_windows(returns, horizon)
    n = len(returns)
    values = np.full((n, len(depths)), np.nan, dtype=np.float64)
    if len(windows) == 0:
        return values, valid
    valid_n = len(windows)
    base = np.asarray(base_positions[:valid_n], dtype=np.float64)
    guard = np.asarray(guard_positions[:valid_n], dtype=np.float64)
    future_sum = np.sum(windows, axis=1)
    future_vol = _future_vol(windows)
    base_dd = _rowwise_path_max_drawdown(windows, base)
    for j, depth in enumerate(depths):
        pos = _depth_positions(base, guard, depth=float(depth), benchmark_position=benchmark_position)
        pos_dd = _rowwise_path_max_drawdown(windows, pos)
        dd_improve = base_dd - pos_dd
        delta = pos - base
        trade_cost = np.abs(delta) * float(unit_cost)
        values[:valid_n, j] = (
            delta * future_sum
            + float(dd_reward) * np.maximum(dd_improve, 0.0)
            - float(dd_worsen_penalty) * np.maximum(-dd_improve, 0.0)
            - trade_cost
            - float(vol_penalty) * np.abs(delta) * future_vol
        )
    return values, valid


def _threshold_candidates(improve: np.ndarray) -> list[float]:
    vals = np.asarray(improve, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("inf")]
    qs = (0.00, 0.25, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
    out = [float(np.quantile(vals, q)) for q in qs]
    out.extend([0.0, float("inf")])
    return sorted(set(out))


def _fit_hgb_multi(
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
    if len(idx) > int(max_train_samples):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=int(max_train_samples), replace=False))
    models: list[Any] = []
    for j in range(y_arr.shape[1]):
        model = make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=120,
                l2_regularization=0.05,
                random_state=seed + j,
            ),
        )
        model.fit(x_clean[idx], y_arr[idx, j])
        models.append(model)
    return models


def _predict_hgb(models: list[Any] | None, x: np.ndarray) -> np.ndarray | None:
    if not models:
        return None
    x_clean = _sanitize_feature_matrix(x)
    return np.stack([m.predict(x_clean) for m in models], axis=1).astype(np.float64)


def _predict_ridge(model: RidgeModel | None, x: np.ndarray) -> np.ndarray | None:
    if model is None:
        return None
    return model.predict(x)


def _positions_from_depth_prediction(
    pred: np.ndarray,
    *,
    base: np.ndarray,
    guard: np.ndarray,
    depths: tuple[float, ...],
    threshold: float,
    benchmark_position: float,
    min_depth: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    depths_arr = np.asarray(depths, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(depths_arr)))
    best_idx = np.argmax(pred_arr, axis=1)
    improve = pred_arr[np.arange(len(pred_arr)), best_idx] - pred_arr[:, zero_idx]
    choose = np.isfinite(improve) & (improve > float(threshold))
    selected_depth = np.zeros(len(pred_arr), dtype=np.float64)
    selected_depth[choose] = np.maximum(depths_arr[best_idx[choose]], float(min_depth))
    positions = np.asarray(base[: len(pred_arr)], dtype=np.float64).copy()
    for depth in sorted(set(float(x) for x in selected_depth if x > 0.0)):
        mask = np.isclose(selected_depth, depth)
        positions[mask] = _depth_positions(
            np.asarray(base[: len(pred_arr)], dtype=np.float64)[mask],
            np.asarray(guard[: len(pred_arr)], dtype=np.float64)[mask],
            depth=depth,
            benchmark_position=benchmark_position,
        )
    return positions, {
        "raw_active_rate": float(np.mean(choose)),
        "mean_selected_depth": float(np.mean(selected_depth)),
        "max_selected_depth": float(np.max(selected_depth)) if len(selected_depth) else 0.0,
        "pred_improve_mean": float(np.nanmean(improve)),
    }


def _selection_score(metrics: dict[str, Any], *, turnover_cap: float, target_dd: float, dd_margin: float) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    sharpe = float(metrics.get("sharpe_delta", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    if turnover > float(turnover_cap):
        return -1_000_000.0 - 1000.0 * (turnover - float(turnover_cap))
    hard_dd = float(target_dd) - float(dd_margin)
    if maxdd > hard_dd:
        return -100_000.0 - 1000.0 * (maxdd - hard_dd) + alpha
    return alpha + 8.0 * sharpe + 4.0 * max(0.0, -maxdd) - 0.15 * turnover


def _select_policy(
    *,
    name: str,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    guard_val: np.ndarray,
    guard_test: np.ndarray,
    depths: tuple[float, ...],
    val_returns: np.ndarray,
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    benchmark_position: float,
    turnover_cap: float,
    target_dd: float,
    dd_margin: float,
    min_depth_grid: tuple[float, ...],
) -> DepthPolicy:
    depths_arr = np.asarray(depths, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(depths_arr)))
    best_idx_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_idx_val] - pred_val[:, zero_idx]
    best: DepthPolicy | None = None
    for min_depth in min_depth_grid:
        for threshold in _threshold_candidates(improve_val):
            pos_val, diag = _positions_from_depth_prediction(
                pred_val,
                base=base_val,
                guard=guard_val,
                depths=depths,
                threshold=threshold,
                benchmark_position=benchmark_position,
                min_depth=float(min_depth),
            )
            stress = _stress_metrics(
                returns=val_returns[: len(pos_val)],
                positions=pos_val,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
            metrics = stress["cost_x1"]
            score = _selection_score(
                metrics,
                turnover_cap=turnover_cap,
                target_dd=target_dd,
                dd_margin=dd_margin,
            )
            candidate = DepthPolicy(
                name=name,
                positions=pos_val,
                val_metrics=metrics,
                diag={
                    **diag,
                    "threshold": "inf" if not math.isfinite(float(threshold)) else float(threshold),
                    "min_depth": float(min_depth),
                    "val_score": float(score),
                },
                score=float(score),
            )
            if best is None or candidate.score > best.score:
                best = candidate
    assert best is not None
    threshold = float("inf") if best.diag["threshold"] == "inf" else float(best.diag["threshold"])
    pos_test, test_diag = _positions_from_depth_prediction(
        pred_test,
        base=base_test,
        guard=guard_test,
        depths=depths,
        threshold=threshold,
        benchmark_position=benchmark_position,
        min_depth=float(best.diag["min_depth"]),
    )
    return DepthPolicy(
        name=name,
        positions=pos_test,
        val_metrics=best.val_metrics,
        diag={**best.diag, **{f"test_{k}": v for k, v in test_diag.items()}},
        score=best.score,
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == group]
        metrics = [r["stress"]["cost_x1"] for r in rs]
        alpha = np.asarray([float(m["alpha_excess_pt"]) for m in metrics], dtype=np.float64)
        dd = np.asarray([float(m["maxdd_delta_pt"]) for m in metrics], dtype=np.float64)
        turnover = np.asarray([float(m["turnover"]) for m in metrics], dtype=np.float64)
        out[group] = {
            "folds": int(len(rs)),
            "pass_alpha_ge3_dd_le_neg3": int(np.sum((alpha >= 3.0) & (dd <= -3.0))),
            "pass_alpha_ge10_dd_le_neg5": int(np.sum((alpha >= 10.0) & (dd <= -5.0))),
            "alpha_mean": float(np.mean(alpha)) if len(alpha) else float("nan"),
            "alpha_median": float(np.median(alpha)) if len(alpha) else float("nan"),
            "alpha_worst": float(np.min(alpha)) if len(alpha) else float("nan"),
            "maxdd_mean": float(np.mean(dd)) if len(dd) else float("nan"),
            "maxdd_worst": float(np.max(dd)) if len(dd) else float("nan"),
            "turnover_mean": float(np.mean(turnover)) if len(turnover) else float("nan"),
            "turnover_max": float(np.max(turnover)) if len(turnover) else float("nan"),
        }
    return out


def _stress_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stresses = sorted({s for r in rows for s in r["stress"].keys()})
    groups = sorted({r["group"] for r in rows})
    out: dict[str, Any] = {}
    for stress in stresses:
        out[stress] = {}
        for group in groups:
            rs = [r for r in rows if r["group"] == group]
            metrics = [r["stress"][stress] for r in rs]
            alpha = np.asarray([float(m["alpha_excess_pt"]) for m in metrics], dtype=np.float64)
            dd = np.asarray([float(m["maxdd_delta_pt"]) for m in metrics], dtype=np.float64)
            turnover = np.asarray([float(m["turnover"]) for m in metrics], dtype=np.float64)
            out[stress][group] = {
                "folds": int(len(rs)),
                "pass_alpha_ge3_dd_le_neg3": int(np.sum((alpha >= 3.0) & (dd <= -3.0))),
                "alpha_median": float(np.median(alpha)) if len(alpha) else float("nan"),
                "alpha_worst": float(np.min(alpha)) if len(alpha) else float("nan"),
                "maxdd_worst": float(np.max(dd)) if len(dd) else float("nan"),
                "turnover_max": float(np.max(turnover)) if len(turnover) else float("nan"),
            }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan009 Depth Learner Probe",
        "",
        "Action-conditioned depth utility learner over Plan004 base + Plan005 guard state.",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        "",
        "## Aggregate: cost_x1",
        "",
        "| group | folds | pass +3/-3 | pass +10/-5 | Alpha mean | Alpha median | Alpha worst | MaxDD mean | MaxDD worst | TO mean | TO max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row["folds"]),
                    str(row["pass_alpha_ge3_dd_le_neg3"]),
                    str(row["pass_alpha_ge10_dd_le_neg5"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_mean"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                    _fmt(row["turnover_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Fold Detail: cost_x1", ""])
    lines.append("| fold | group | AlphaEx | MaxDDDelta | TO | val Alpha | val MaxDD | val TO | selection |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        d = row.get("diag", {})
        v = d.get("val", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    _fmt(m["alpha_excess_pt"]),
                    _fmt(m["maxdd_delta_pt"]),
                    _fmt(m["turnover"]),
                    _fmt(v.get("alpha_excess_pt")),
                    _fmt(v.get("maxdd_delta_pt")),
                    _fmt(v.get("turnover")),
                    f"thr={d.get('threshold')} min_depth={d.get('min_depth')}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Stress Aggregate", ""])
    for stress, groups in payload["stress_aggregate"].items():
        lines.extend(
            [
                f"### {stress}",
                "",
                "| group | folds | pass +3/-3 | Alpha median | Alpha worst | MaxDD worst | TO max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, row in groups.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        str(row["folds"]),
                        str(row["pass_alpha_ge3_dd_le_neg3"]),
                        _fmt(row["alpha_median"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["maxdd_worst"]),
                        _fmt(row["turnover_max"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Leak Discipline",
            "",
            "- Plan004 base policy is fold-local.",
            "- Plan005 guard mode is selected from shifted past features at segment start.",
            "- Depth utility labels use future returns only within train for fitting and within validation for selection.",
            "- Test positions are produced once from the train-fitted learner and validation-selected threshold.",
        ]
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan009_depth_learner_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=3.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--depths", default="0,0.3,0.5,0.75,0.94,1.0")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--dd-reward", type=float, default=2.0)
    parser.add_argument("--dd-worsen-penalty", type=float, default=4.0)
    parser.add_argument("--vol-penalty", type=float, default=0.15)
    parser.add_argument("--turnover-cap", type=float, default=45.0)
    parser.add_argument("--target-dd", type=float, default=-3.0)
    parser.add_argument("--dd-margin", type=float, default=1.5)
    parser.add_argument("--min-depth-grid", default="0,0.3,0.5,0.75,0.94")
    parser.add_argument("--hgb", action="store_true")
    parser.add_argument("--source-selection-mode", choices=("primary", "include_costx3"), default="primary")
    parser.add_argument("--output-json", default="docs_local/20260528_plan009_depth_learner.json")
    parser.add_argument("--output-md", default="docs_local/20260528_plan009_depth_learner.md")
    args = parser.parse_args()

    set_seed(int(args.seed))
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
    splits, selected_folds = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    unit_cost = _unit_cost(costs_cfg)
    full_returns = np.asarray(raw_returns, dtype=np.float64)
    past_features = _build_past_features(full_returns)
    depths = tuple(float(x) for x in str(args.depths).split(",") if x.strip())
    min_depth_grid = tuple(float(x) for x in str(args.min_depth_grid).split(",") if x.strip())

    rows: list[dict[str, Any]] = []
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[Plan009Depth] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=data_cfg.get("seq_len", 64))
        rec = run_plan004_fold_policy(
            ds=ds,
            cfg=cfg,
            costs_cfg=costs_cfg,
            fold_idx=fid,
            seed=int(args.seed),
            ridge_l2=float(args.ridge_l2),
            max_train_samples=int(args.max_train_samples),
            selection_stress_mode=str(args.source_selection_mode),
        )
        best = rec.get("best_candidate") or {}
        base_train = np.asarray(best.get("base_train", np.full(len(ds.train_returns), benchmark_position)), dtype=np.float64)
        base_val = np.asarray(best.get("base_val", np.full(len(ds.val_returns), benchmark_position)), dtype=np.float64)
        base_test = np.asarray(rec.get("positions", np.full(len(ds.test_returns), benchmark_position)), dtype=np.float64)

        train_mask = _mask_for(features_df.index, split.train_start, split.train_end, inclusive_end=True)
        val_mask = _mask_for(features_df.index, split.val_start, split.val_end, inclusive_end=False)
        test_mask = _mask_for(features_df.index, split.test_start, split.test_end, inclusive_end=True)
        train_mode, _train_diag = select_meta_guard_mode(past_features, _segment_start_idx(train_mask))
        val_mode, _val_diag = select_meta_guard_mode(past_features, _segment_start_idx(val_mask))
        test_mode, test_mode_diag = select_meta_guard_mode(past_features, _segment_start_idx(test_mask))
        guard_train = guard_positions_from_features(past_features, mode=train_mode)[train_mask][: len(ds.train_returns)]
        guard_val = guard_positions_from_features(past_features, mode=val_mode)[val_mask][: len(ds.val_returns)]
        guard_test = guard_positions_from_features(past_features, mode=test_mode)[test_mask][: len(ds.test_returns)]

        x_train = _augment_features(
            _state_features(ds.train_features, ds.train_returns),
            base=base_train,
            guard=guard_train,
            mode=train_mode,
            benchmark_position=benchmark_position,
        )
        x_val = _augment_features(
            _state_features(ds.val_features, ds.val_returns),
            base=base_val,
            guard=guard_val,
            mode=val_mode,
            benchmark_position=benchmark_position,
        )
        x_test = _augment_features(
            _state_features(ds.test_features, ds.test_returns),
            base=base_test,
            guard=guard_test,
            mode=test_mode,
            benchmark_position=benchmark_position,
        )

        y_train, train_valid = _depth_utilities(
            ds.train_returns,
            base_positions=base_train,
            guard_positions=guard_train,
            depths=depths,
            horizon=int(args.horizon),
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            dd_reward=float(args.dd_reward),
            dd_worsen_penalty=float(args.dd_worsen_penalty),
            vol_penalty=float(args.vol_penalty),
        )
        base_stress = _stress_metrics(
            returns=ds.test_returns[: len(base_test)],
            positions=base_test,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan004_base",
                "stress": base_stress,
                "diag": {"source": rec.get("selected_row", {}).get("source"), "spec": rec.get("selected_row", {}).get("spec")},
            }
        )

        ridge = _fit_ridge_multi(x_train[train_valid], y_train[train_valid], l2=float(args.ridge_l2))
        model_outputs: list[tuple[str, np.ndarray | None, np.ndarray | None]] = [
            ("depth_ridge", _predict_ridge(ridge, x_val), _predict_ridge(ridge, x_test))
        ]
        if bool(args.hgb):
            hgb = _fit_hgb_multi(
                x_train[train_valid],
                y_train[train_valid],
                seed=int(args.seed) + fid * 41,
                max_train_samples=int(args.max_train_samples),
            )
            model_outputs.append(("depth_hgb", _predict_hgb(hgb, x_val), _predict_hgb(hgb, x_test)))

        for name, pred_val, pred_test in model_outputs:
            if pred_val is None or pred_test is None:
                continue
            policy = _select_policy(
                name=name,
                pred_val=pred_val,
                pred_test=pred_test,
                base_val=base_val,
                base_test=base_test,
                guard_val=guard_val,
                guard_test=guard_test,
                depths=depths,
                val_returns=ds.val_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                turnover_cap=float(args.turnover_cap),
                target_dd=float(args.target_dd),
                dd_margin=float(args.dd_margin),
                min_depth_grid=min_depth_grid,
            )
            stress = _stress_metrics(
                returns=ds.test_returns[: len(policy.positions)],
                positions=policy.positions,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
            rows.append(
                {
                    "fold": fid,
                    "group": policy.name,
                    "stress": stress,
                    "diag": {
                        **policy.diag,
                        "val": policy.val_metrics,
                        "train_mode": train_mode,
                        "val_mode": val_mode,
                        "test_mode": test_mode,
                        **test_mode_diag,
                    },
                }
            )
            m = stress["cost_x1"]
            print(
                f"[Plan009Depth] fold={fid} {policy.name} "
                f"alpha={m['alpha_excess_pt']:+.2f} dd={m['maxdd_delta_pt']:+.2f} "
                f"to={m['turnover']:.2f} min_depth={policy.diag.get('min_depth')}"
            )

    payload = {
        "experiment": "plan009_depth_learner_probe",
        "seed": int(args.seed),
        "config": args.config,
        "folds": selected_folds,
        "depths": depths,
        "horizon": int(args.horizon),
        "dd_reward": float(args.dd_reward),
        "dd_worsen_penalty": float(args.dd_worsen_penalty),
        "vol_penalty": float(args.vol_penalty),
        "turnover_cap": float(args.turnover_cap),
        "target_dd": float(args.target_dd),
        "dd_margin": float(args.dd_margin),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "stress_aggregate": _stress_aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Plan009Depth] wrote {args.output_json}")
    print(f"[Plan009Depth] wrote {args.output_md}")


if __name__ == "__main__":
    main()
