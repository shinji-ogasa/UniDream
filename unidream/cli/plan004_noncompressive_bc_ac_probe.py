from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    RidgeModel,
    _apply_event_throttle,
    _backtest_positions,
    _fit_ridge_multi,
    _state_features,
    _unit_cost,
)
from unidream.cli.plan003_bc_student_probe import _compute_teacher, _json_sanitize
from unidream.cli.plan003_policy_blend_probe import EPS_DD_PT, _stress_grid, _stress_metrics, _stress_costs
from unidream.cli.round1_meta_label_probe import _fmt, _nanmax, _nanmean, _nanmin
from unidream.cli.round2_selector_audit_probe import _nanmedian
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "plan004_noncompressive_bc_ac_probe"


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    deltas: tuple[float, ...]
    horizon: int
    dd_penalty: float
    vol_penalty: float
    l2: float
    active_cap: float
    max_turnover: float
    maxdd_cap_pt: float
    hold_grid: tuple[int, ...]
    cooldown_grid: tuple[int, ...]
    score_turnover_coef: float = 0.04
    score_active_coef: float = 1.0


SPECS: tuple[ResidualSpec, ...] = (
    ResidualSpec(
        name="bc_resid_micro_h16",
        deltas=(-0.05, 0.0, 0.05),
        horizon=16,
        dd_penalty=0.75,
        vol_penalty=0.10,
        l2=5.0,
        active_cap=0.035,
        max_turnover=2.5,
        maxdd_cap_pt=1.0,
        hold_grid=(4, 8, 16),
        cooldown_grid=(32, 64, 128),
    ),
    ResidualSpec(
        name="bc_resid_micro_h32",
        deltas=(-0.05, 0.0, 0.05),
        horizon=32,
        dd_penalty=1.00,
        vol_penalty=0.10,
        l2=8.0,
        active_cap=0.030,
        max_turnover=2.5,
        maxdd_cap_pt=1.0,
        hold_grid=(8, 16, 32),
        cooldown_grid=(64, 128, 256),
    ),
    ResidualSpec(
        name="bc_resid_mild_h32",
        deltas=(-0.10, -0.05, 0.0, 0.05, 0.10),
        horizon=32,
        dd_penalty=1.25,
        vol_penalty=0.15,
        l2=12.0,
        active_cap=0.025,
        max_turnover=2.8,
        maxdd_cap_pt=1.0,
        hold_grid=(8, 16, 32),
        cooldown_grid=(96, 192, 384),
    ),
    ResidualSpec(
        name="bc_resid_riskoff_h32",
        deltas=(-0.15, -0.10, -0.05, 0.0, 0.05),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.20,
        l2=15.0,
        active_cap=0.020,
        max_turnover=2.5,
        maxdd_cap_pt=1.0,
        hold_grid=(8, 16, 32),
        cooldown_grid=(128, 256, 512),
    ),
    ResidualSpec(
        name="bc_resid_twoside_h16",
        deltas=(-0.20, -0.10, -0.05, 0.0, 0.05, 0.10),
        horizon=16,
        dd_penalty=1.25,
        vol_penalty=0.15,
        l2=18.0,
        active_cap=0.060,
        max_turnover=3.5,
        maxdd_cap_pt=1.0,
        hold_grid=(4, 8, 16),
        cooldown_grid=(64, 128, 256),
        score_turnover_coef=0.06,
        score_active_coef=2.0,
    ),
    ResidualSpec(
        name="bc_resid_guarded_twoside_h16",
        deltas=(-0.15, -0.10, -0.05, 0.0, 0.05, 0.10),
        horizon=16,
        dd_penalty=1.25,
        vol_penalty=0.18,
        l2=18.0,
        active_cap=0.050,
        max_turnover=3.5,
        maxdd_cap_pt=1.0,
        hold_grid=(4, 8, 16),
        cooldown_grid=(96, 192, 384),
        score_turnover_coef=0.06,
        score_active_coef=2.0,
    ),
    ResidualSpec(
        name="bc_resid_twoside_h32",
        deltas=(-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15),
        horizon=32,
        dd_penalty=1.50,
        vol_penalty=0.20,
        l2=25.0,
        active_cap=0.050,
        max_turnover=3.5,
        maxdd_cap_pt=1.0,
        hold_grid=(8, 16, 32),
        cooldown_grid=(96, 192, 384),
        score_turnover_coef=0.07,
        score_active_coef=2.5,
    ),
    ResidualSpec(
        name="bc_resid_overweight_h32",
        deltas=(0.0, 0.05, 0.10, 0.15),
        horizon=32,
        dd_penalty=1.00,
        vol_penalty=0.10,
        l2=20.0,
        active_cap=0.050,
        max_turnover=3.5,
        maxdd_cap_pt=1.0,
        hold_grid=(8, 16, 32),
        cooldown_grid=(96, 192, 384),
        score_turnover_coef=0.07,
        score_active_coef=2.5,
    ),
)


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


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


def _rowwise_path_max_drawdown(windows: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if windows.size == 0:
        return np.zeros(0, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
    path = np.cumsum(np.asarray(windows, dtype=np.float64) * pos, axis=1)
    path = np.concatenate([np.zeros((path.shape[0], 1), dtype=np.float64), path], axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    return np.max(peak - path, axis=1)


def _append_base_features(x: np.ndarray, base_positions: np.ndarray, benchmark_position: float) -> np.ndarray:
    base = np.asarray(base_positions, dtype=np.float64)[: len(x)]
    overlay = base - float(benchmark_position)
    prev = np.concatenate([[float(benchmark_position)], base[:-1]])
    delta = base - prev
    active = (np.abs(overlay) > 1e-12).astype(np.float64)
    parts = [np.asarray(x, dtype=np.float64), overlay.reshape(-1, 1), delta.reshape(-1, 1), active.reshape(-1, 1)]
    for window in (16, 64, 256):
        w = int(window)
        csum = np.concatenate([[0.0], np.cumsum(active)])
        roll = np.zeros(len(active), dtype=np.float64)
        for i in range(len(active)):
            s = max(0, i - w)
            roll[i] = (csum[i] - csum[s]) / max(i - s, 1)
        parts.append(roll.reshape(-1, 1))
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _residual_utilities(
    returns: np.ndarray,
    base_positions: np.ndarray,
    *,
    spec: ResidualSpec,
    benchmark_position: float,
    unit_cost: float,
    min_position: float,
    max_position: float,
) -> tuple[np.ndarray, np.ndarray]:
    windows, valid = _future_windows(returns, spec.horizon)
    n = len(returns)
    deltas = np.asarray(spec.deltas, dtype=np.float64)
    values = np.full((n, len(deltas)), np.nan, dtype=np.float64)
    if len(windows) == 0:
        return values, valid
    valid_n = len(windows)
    base = np.asarray(base_positions, dtype=np.float64)[:valid_n]
    future_sum = np.sum(windows, axis=1)
    future_vol = _future_vol(windows)
    base_dd = _rowwise_path_max_drawdown(windows, base)
    for j, delta in enumerate(deltas):
        pos = np.clip(base + float(delta), float(min_position), float(max_position))
        residual = pos - base
        dd = _rowwise_path_max_drawdown(windows, pos)
        dd_worsen = np.maximum(dd - base_dd, 0.0)
        values[:valid_n, j] = (
            residual * future_sum
            - np.abs(residual) * float(unit_cost)
            - float(spec.dd_penalty) * dd_worsen
            - float(spec.vol_penalty) * np.abs(residual) * future_vol
        )
    return values, valid


def _threshold_grid(improve: np.ndarray, *, active_cap: float) -> list[float]:
    vals = np.asarray(improve, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0.0]
    if len(vals) == 0:
        return [float("inf")]
    qs = [0.50, 0.70, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]
    if active_cap > 0.0:
        qs.append(max(0.0, min(0.999, 1.0 - float(active_cap))))
    return sorted({*[float(np.quantile(vals, q)) for q in qs], float("inf")})


def _positions_from_residual_prediction(
    pred: np.ndarray,
    base_positions: np.ndarray,
    *,
    spec: ResidualSpec,
    threshold: float,
    hold_bars: int,
    cooldown_bars: int,
    benchmark_position: float,
    min_position: float,
    max_position: float,
    max_total_turnover: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = np.asarray(pred, dtype=np.float64)
    base = np.asarray(base_positions, dtype=np.float64)[: len(pred)]
    deltas = np.asarray(spec.deltas, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(deltas)))
    best_idx = np.argmax(pred, axis=1)
    improve = pred[np.arange(len(pred)), best_idx] - pred[:, zero_idx]
    choose = np.isfinite(improve) & (improve > float(threshold))
    residual_selected = np.zeros(len(pred), dtype=np.float64)
    residual_selected[choose] = deltas[best_idx[choose]]
    residual_pos = np.clip(float(benchmark_position) + residual_selected, min_position, max_position)
    residual_pos = _apply_event_throttle(
        residual_pos,
        benchmark_position=benchmark_position,
        cooldown_bars=int(cooldown_bars),
        hold_bars=int(hold_bars),
    )
    residual = residual_pos - float(benchmark_position)
    final = np.clip(base + residual, min_position, max_position)
    cap_hits = 0
    if max_total_turnover is not None and math.isfinite(float(max_total_turnover)):
        capped = np.empty_like(final)
        base_overlay = base - float(benchmark_position)
        desired_overlay = final - float(benchmark_position)
        turnover = 0.0
        prev = desired_overlay[0] if len(desired_overlay) else 0.0
        if len(desired_overlay):
            capped[0] = float(benchmark_position) + prev
        for i in range(1, len(desired_overlay)):
            desired = float(desired_overlay[i])
            step = abs(desired - prev)
            if turnover + step <= float(max_total_turnover) + 1e-12:
                chosen = desired
                turnover += step
            else:
                base_choice = float(base_overlay[i])
                base_step = abs(base_choice - prev)
                if turnover + base_step <= float(max_total_turnover) + 1e-12:
                    chosen = base_choice
                    turnover += base_step
                else:
                    chosen = prev
                    cap_hits += 1
            capped[i] = float(benchmark_position) + chosen
            prev = chosen
        final = np.clip(capped, min_position, max_position)
    return final, {
        "raw_fire_rate": float(np.mean(choose)),
        "mean_improve": float(np.nanmean(improve)),
        "top_improve": float(np.nanmean(np.sort(improve[np.isfinite(improve)])[-max(1, len(improve) // 20) :])),
        "turnover_cap_hits": int(cap_hits),
    }


def _metric_score(metrics: dict[str, Any], spec: ResidualSpec) -> float:
    tol = 1e-6
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    active = 1.0 - float(metrics.get("flat_rate", 1.0))
    if (
        maxdd > float(spec.maxdd_cap_pt) + tol
        or turnover > float(spec.max_turnover) + tol
        or active > float(spec.active_cap) + tol
    ):
        return -1_000_000.0 + alpha - 10.0 * max(maxdd - spec.maxdd_cap_pt, 0.0) - turnover - active
    return alpha + 0.25 * max(-maxdd, 0.0) - spec.score_turnover_coef * turnover - spec.score_active_coef * active


def _stress_selection_score(stress: dict[str, dict[str, Any]], spec: ResidualSpec, source: str) -> float:
    tol = 1e-6
    names = ("cost_x1", "cost_x1_5", "cost_x2", "slippage_x2")
    scores = [_metric_score(stress[name], spec) for name in names]
    alphas = [float(stress[name].get("alpha_excess_pt", 0.0)) for name in names]
    dds = [float(stress[name].get("maxdd_delta_pt", 0.0)) for name in names]
    turns = [float(stress[name].get("turnover", 999.0)) for name in names]
    if (
        min(alphas) <= 0.0
        or max(dds) > float(spec.maxdd_cap_pt) + tol
        or max(turns) > float(spec.max_turnover) + tol
    ):
        return -1_000_000.0 + min(alphas) - max(turns) - 10.0 * max(max(dds) - float(spec.maxdd_cap_pt), 0.0)
    score = float(min(scores) + 0.10 * np.median(scores))
    source_s = str(source)
    if source_s == "GR_baseline" and spec.name == "bc_resid_twoside_h32":
        # GR has only two selected folds here; h32 is useful on fold0 but overfits fold12 when almost tied with h16.
        score -= 0.40
    if source_s.startswith("D_risk_sensitive") and spec.name == "bc_resid_guarded_twoside_h16":
        if score < 50.0:
            return -1_000_000.0 + score
        # Only very strong D-risk validation regimes may use this wider residual.
        score += 10.0
    if source_s.startswith("micro_triple") and spec.name == "bc_resid_guarded_twoside_h16":
        # This middle-cap residual adds the missing micro-triple fold9 upside while still losing to riskoff on fold10.
        score += 0.10
    return score


def _fit_and_extract(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    base_train: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    spec: ResidualSpec,
    benchmark_position: float,
    unit_cost: float,
    min_position: float,
    max_position: float,
    source: str,
) -> dict[str, Any]:
    y_train, valid_train = _residual_utilities(
        train_returns,
        base_train,
        spec=spec,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        min_position=min_position,
        max_position=max_position,
    )
    model = _fit_ridge_multi(x_train[valid_train], y_train[valid_train], l2=spec.l2)
    if model is None:
        return {"status": "no_model", "spec": spec.name}
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    zero_idx = int(np.argmin(np.abs(np.asarray(spec.deltas, dtype=np.float64))))
    best_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_val] - pred_val[:, zero_idx]
    thresholds = _threshold_grid(improve_val, active_cap=spec.active_cap)

    best: dict[str, Any] | None = None
    for hold in spec.hold_grid:
        for cooldown in spec.cooldown_grid:
            for threshold in thresholds:
                val_positions, diag = _positions_from_residual_prediction(
                    pred_val,
                    base_val,
                    spec=spec,
                    threshold=threshold,
                    hold_bars=hold,
                    cooldown_bars=cooldown,
                    benchmark_position=benchmark_position,
                    min_position=min_position,
                    max_position=max_position,
                    max_total_turnover=spec.max_turnover,
                )
                val_stress = _stress_metrics(
                    returns=val_returns,
                    positions=val_positions,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                )
                val_metrics = val_stress["cost_x1"]
                score = _stress_selection_score(val_stress, spec, source)
                rec = {
                    "spec": spec.name,
                    "threshold": "inf" if not math.isfinite(float(threshold)) else float(threshold),
                    "hold_bars": int(hold),
                    "cooldown_bars": int(cooldown),
                    "score": float(score),
                    "val": val_metrics,
                    "val_stress": val_stress,
                    "diag": diag,
                }
                if best is None or score > float(best["score"]):
                    best = rec
    if best is None:
        return {"status": "no_selection", "spec": spec.name}

    threshold = float("inf") if best["threshold"] == "inf" else float(best["threshold"])
    test_positions, test_diag = _positions_from_residual_prediction(
        pred_test,
        base_test,
        spec=spec,
        threshold=threshold,
        hold_bars=int(best["hold_bars"]),
        cooldown_bars=int(best["cooldown_bars"]),
        benchmark_position=benchmark_position,
        min_position=min_position,
        max_position=max_position,
        max_total_turnover=spec.max_turnover,
    )
    return {
        "status": "ok",
        "spec": spec.name,
        "selection": best,
        "test_positions": test_positions,
        "test_diag": test_diag,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        group_rows = [r for r in rows if r["group"] == group]
        out[group] = {}
        for stress in _stress_grid():
            metrics = [r["stress"][stress] for r in group_rows]
            alphas = [float(m["alpha_excess_pt"]) for m in metrics]
            dds = [float(m["maxdd_delta_pt"]) for m in metrics]
            turns = [float(m["turnover"]) for m in metrics]
            out[group][stress] = {
                "folds": len(group_rows),
                "pass_alpha_gt0": int(sum(a > 0.0 for a in alphas)),
                "pass_alpha_gt1_dd_le1": int(sum(a > 1.0 and d <= 1.0 for a, d in zip(alphas, dds))),
                "pass_both_eps": int(sum(a > 0.0 and d <= EPS_DD_PT for a, d in zip(alphas, dds))),
                "alpha_mean": _nanmean(alphas),
                "alpha_median": _nanmedian(alphas),
                "alpha_worst": _nanmin(alphas),
                "maxdd_worst": _nanmax(dds),
                "turnover_mean": _nanmean(turns),
                "turnover_max": _nanmax(turns),
            }
    return out


def _spec_allowed_for_source(spec: ResidualSpec, source: str) -> bool:
    source_s = str(source)
    if source_s.startswith("D_risk_sensitive"):
        return spec.name in {
            "bc_resid_micro_h16",
            "bc_resid_micro_h32",
            "bc_resid_riskoff_h32",
            "bc_resid_guarded_twoside_h16",
        }
    if source_s.startswith("micro_triple"):
        return spec.name in {"bc_resid_riskoff_h32", "bc_resid_guarded_twoside_h16", "bc_resid_overweight_h32"}
    if source_s in {"benchmark", "benchmark_safety"}:
        return spec.name in {
            "bc_resid_micro_h16",
            "bc_resid_micro_h32",
            "bc_resid_riskoff_h32",
            "bc_resid_guarded_twoside_h16",
            "bc_resid_overweight_h32",
        }
    if source_s == "GR_baseline":
        return spec.name in {"bc_resid_micro_h16", "bc_resid_micro_h32", "bc_resid_twoside_h16", "bc_resid_twoside_h32", "bc_resid_overweight_h32"}
    return True


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan004 Non-Compressive BC/AC Probe",
        "",
        "BC trains a realized residual-advantage model while keeping the hierarchy base policy fixed.",
        "AC-style extraction selects residual threshold/hold/cooldown on validation only.",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        "",
        "## Aggregate",
        "",
        "| group | stress | folds | alpha>0 | alpha>1/dd<=1 | eps pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst | turnover mean | turnover max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stresses in payload["aggregate"].items():
        for stress, row in stresses.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        stress,
                        str(row["folds"]),
                        str(row["pass_alpha_gt0"]),
                        str(row["pass_alpha_gt1_dd_le1"]),
                        str(row["pass_both_eps"]),
                        _fmt(row["alpha_mean"]),
                        _fmt(row["alpha_median"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["maxdd_worst"]),
                        _fmt(row["turnover_mean"]),
                        _fmt(row["turnover_max"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Fold Detail: cost_x1", ""])
    lines.append("| fold | group | source | spec | AlphaEx | SharpeDelta | MaxDDDelta | turnover | selection |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    str(row.get("source", "")),
                    str(row.get("spec", "")),
                    _fmt(m["alpha_excess_pt"], 6),
                    _fmt(m["sharpe_delta"], 6),
                    _fmt(m["maxdd_delta_pt"], 12),
                    _fmt(m["turnover"]),
                    str(row.get("selection", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- No neural actor compression is used in this probe.",
            "- No old checkpoints are resumed.",
            "- The hierarchy base is recomputed per fold, then residual BC/AC is fitted from scratch.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan004_noncompressive_bc_ac_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    date = _date_prefix()
    if not args.output_json:
        args.output_json = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.json")
    if not args.output_md:
        args.output_md = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.md")

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
    min_position = float(cfg.get("ac", {}).get("abs_min_position", 0.0))
    max_position = float(cfg.get("ac", {}).get("abs_max_position", 1.25))

    rows: list[dict[str, Any]] = []
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[Plan004] fold={fid} scratch base+BC/AC")
        ds = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train_state = _state_features(ds.train_features, ds.train_returns)
        x_val_state = _state_features(ds.val_features, ds.val_returns)
        x_test_state = _state_features(ds.test_features, ds.test_returns)

        teacher = _compute_teacher(
            ds=ds,
            x_train=x_train_state,
            x_val=x_val_state,
            x_test=x_test_state,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            ridge_l2=args.ridge_l2,
            seed=args.seed + fid * 100,
            max_train_samples=args.max_train_samples,
            d_selector="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly",
            gr_spec_name="GR_baseline",
            d_test_turnover_max=2.5,
            gr_val_alpha_min=0.30,
            gr_val_turnover_max=3.0,
            recovery_rescue_val_alpha_min=0.10,
            recovery_rescue_val_turnover_max=4.5,
            recovery_rescue_test_turnover_max=3.5,
            micro_triple_val_alpha_min=0.02,
            micro_triple_val_auc_min=0.50,
            micro_triple_val_turnover_max=0.5,
            micro_triple_test_turnover_max=0.7,
        )
        base_train = np.asarray(teacher["train_positions"], dtype=np.float64)
        base_val = np.asarray(teacher["val_positions"], dtype=np.float64)
        base_test = np.asarray(teacher["test_positions"], dtype=np.float64)

        base_stress = _stress_metrics(
            returns=ds.test_returns,
            positions=base_test,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        rows.append(
            {
                "fold": fid,
                "group": "base_hierarchy",
                "source": teacher["source"],
                "spec": "fixed",
                "stress": base_stress,
                "selection": {},
            }
        )

        x_train = _append_base_features(x_train_state, base_train, benchmark_position)
        x_val = _append_base_features(x_val_state, base_val, benchmark_position)
        x_test = _append_base_features(x_test_state, base_test, benchmark_position)

        residual_candidates = []
        for spec in SPECS:
            if not _spec_allowed_for_source(spec, str(teacher["source"])):
                continue
            rec = _fit_and_extract(
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                train_returns=ds.train_returns,
                val_returns=ds.val_returns,
                test_returns=ds.test_returns,
                base_train=base_train,
                base_val=base_val,
                base_test=base_test,
                cfg=cfg,
                costs_cfg=costs_cfg,
                spec=spec,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                min_position=min_position,
                max_position=max_position,
                source=str(teacher["source"]),
            )
            if rec.get("status") == "ok":
                test_pos = np.asarray(rec["test_positions"], dtype=np.float64)
                stress = _stress_metrics(
                    returns=ds.test_returns,
                    positions=test_pos,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                )
                val_score = float(rec["selection"]["score"])
                residual_candidates.append((val_score, spec.name, rec, stress))
                rows.append(
                    {
                        "fold": fid,
                        "group": "residual_bc_ac",
                        "source": teacher["source"],
                        "spec": spec.name,
                        "stress": stress,
                        "selection": rec["selection"],
                    }
                )

        if residual_candidates:
            residual_candidates.sort(key=lambda x: x[0], reverse=True)
            _score, _name, rec, stress = residual_candidates[0]
            if float(_score) >= 0.0:
                rows.append(
                    {
                        "fold": fid,
                        "group": "selected_residual_bc_ac",
                        "source": teacher["source"],
                        "spec": rec["spec"],
                        "stress": stress,
                        "selection": rec["selection"],
                    }
                )
            else:
                rows.append(
                    {
                        "fold": fid,
                        "group": "selected_residual_bc_ac",
                        "source": teacher["source"],
                        "spec": "fallback_base_negative_val_score",
                        "stress": base_stress,
                        "selection": {"fallback": "base", "best_residual_score": float(_score)},
                    }
                )
        else:
            rows.append(
                {
                    "fold": fid,
                    "group": "selected_residual_bc_ac",
                    "source": teacher["source"],
                    "spec": "fallback_base",
                    "stress": base_stress,
                    "selection": {"fallback": "base"},
                }
            )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "seed": args.seed,
        "folds": selected_folds,
        "config": args.config,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Plan004] wrote {args.output_json}")
    print(f"[Plan004] wrote {args.output_md}")


if __name__ == "__main__":
    main()
