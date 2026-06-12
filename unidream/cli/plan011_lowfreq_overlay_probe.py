from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from unidream.cli.train import (
    _action_stats,
    _benchmark_position_value,
    _forward_window_stats,
    _m2_scorecard,
)
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt(v: float | None, digits: int = 2, signed: bool = True) -> str:
    if v is None or not np.isfinite(float(v)):
        return "NA"
    prefix = "+" if signed else ""
    return f"{float(v):{prefix}.{digits}f}"


@dataclass(frozen=True)
class OverlaySpec:
    base_overlay: float
    base_safe_only: bool
    ema_span: int
    risk_hi: float
    risk_lo: float
    edge_hi: float
    edge_lo: float
    trend_lookback: int
    trend_max: float
    trend_min: float
    dd_lookback: int
    dd_min: float
    dd_max: float
    down_overlay: float
    up_overlay: float
    max_step: float
    min_hold: int
    deadzone: float


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.empty_like(x)
    prev = float(x[0])
    out[0] = prev
    for i in range(1, len(x)):
        prev = (1.0 - alpha) * prev + alpha * float(x[i])
        out[i] = prev
    return out


def _causal_trailing_sum(returns: np.ndarray, lookback: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], returns[:-1]])
    csum = np.concatenate([[0.0], np.cumsum(shifted)])
    lb = max(int(lookback), 1)
    out = csum[1:] - csum[np.maximum(np.arange(len(returns)) + 1 - lb, 0)]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _causal_trailing_drawdown(returns: np.ndarray, lookback: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    if lookback <= 0:
        return np.zeros(len(returns), dtype=np.float64)
    shifted = np.concatenate([[0.0], returns[:-1]])
    equity = np.cumsum(np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0))
    lb = max(int(lookback), 1)
    out = np.zeros(len(returns), dtype=np.float64)
    for i in range(len(returns)):
        start = max(0, i + 1 - lb)
        peak = float(np.max(equity[start : i + 1]))
        out[i] = max(0.0, peak - float(equity[i]))
    return out


def _lowfreq_positions(
    *,
    predictive: np.ndarray,
    returns: np.ndarray,
    spec: OverlaySpec,
    benchmark: float,
    risk_indices: tuple[int, ...],
    edge_indices: tuple[int, ...],
) -> np.ndarray:
    if predictive.ndim == 1:
        predictive = predictive[:, None]
    valid = [i for i in risk_indices if 0 <= i < predictive.shape[1]]
    if not valid:
        risk = np.zeros(len(returns), dtype=np.float64)
    else:
        risk = np.nan_to_num(predictive[:, valid].mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    edge_valid = [i for i in edge_indices if 0 <= i < predictive.shape[1]]
    if not edge_valid:
        edge_slow = None
    else:
        edge = np.nan_to_num(predictive[:, edge_valid].mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
        edge_slow = _ema(edge, spec.ema_span)
    risk_slow = _ema(risk, spec.ema_span)
    trend = _causal_trailing_sum(returns, spec.trend_lookback)
    dd_depth = _causal_trailing_drawdown(returns, spec.dd_lookback)

    raw_overlay = np.full(len(returns), float(spec.base_overlay), dtype=np.float64)
    risk_on = (risk_slow >= spec.risk_hi) & (trend <= spec.trend_max)
    risk_off = (risk_slow <= spec.risk_lo) & (trend >= spec.trend_min)
    if edge_slow is not None:
        risk_on = risk_on & (edge_slow <= spec.edge_lo)
        risk_off = risk_off & (edge_slow >= spec.edge_hi)
    if spec.dd_lookback > 0:
        risk_on = risk_on & (dd_depth >= spec.dd_min)
        risk_off = risk_off & (dd_depth <= spec.dd_max)
    if spec.base_safe_only:
        base_mask = trend >= spec.trend_min
        if spec.dd_lookback > 0:
            base_mask = base_mask & (dd_depth <= spec.dd_max)
        raw_overlay = np.zeros(len(returns), dtype=np.float64)
        raw_overlay[base_mask] = float(spec.base_overlay)
    raw_overlay[risk_on] = float(spec.base_overlay) - abs(float(spec.down_overlay))
    raw_overlay[risk_off] = float(spec.base_overlay) + abs(float(spec.up_overlay))
    if spec.deadzone > 0.0:
        raw_overlay[np.abs(raw_overlay) < spec.deadzone] = 0.0

    overlay = np.zeros_like(raw_overlay)
    current = 0.0
    hold = spec.min_hold
    for i, target in enumerate(raw_overlay):
        if hold < spec.min_hold and abs(target - current) > spec.deadzone:
            target = current
        delta = float(np.clip(target - current, -spec.max_step, spec.max_step))
        next_overlay = current + delta
        if abs(next_overlay - current) > 1e-8:
            hold = 0
        else:
            hold += 1
        current = next_overlay
        overlay[i] = current
    return (benchmark + overlay).astype(np.float32)


def _metrics_record(metrics, positions: np.ndarray, benchmark: float) -> dict[str, Any]:
    stats = _action_stats(positions, benchmark_position=benchmark)
    scorecard = _m2_scorecard(metrics, stats, {})
    return {
        "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
        "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
        "sharpe_delta": float(metrics.sharpe_delta or 0.0),
        "turnover": float(stats["turnover"]),
        "long": float(stats["long"]),
        "short": float(stats["short"]),
        "flat": float(stats["flat"]),
        "m2_pass": bool(scorecard["m2_pass"]),
        "stretch_hit": bool(scorecard["stretch_hit"]),
    }


def _score(rec: dict[str, Any], *, max_turnover: float) -> float:
    alpha = float(rec["alpha_excess_pt"])
    dd = float(rec["maxdd_delta_pt"])
    turnover = float(rec["turnover"])
    score = 2.0 * alpha + 4.0 * max(0.0, -dd) - 1.5 * turnover
    if alpha < 0.0:
        score -= 50.0 + abs(alpha)
    if dd > 0.0:
        score -= 25.0 + 5.0 * dd
    if turnover > max_turnover:
        score -= 100.0 + 10.0 * (turnover - max_turnover)
    return float(score)


def _robust_score(
    rec: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    max_turnover: float,
) -> float:
    alpha = float(rec["alpha_excess_pt"])
    dd = float(rec["maxdd_delta_pt"])
    turnover = float(rec["turnover"])
    alpha_min = min(float(first["alpha_excess_pt"]), float(second["alpha_excess_pt"]))
    dd_worst = max(float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]), dd)
    dd_best = min(float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]), dd)

    score = (
        2.5 * alpha
        + 2.0 * alpha_min
        + 20.0 * max(0.0, -dd_best)
        - 20.0 * max(0.0, dd_worst)
        - 2.0 * turnover
    )
    if alpha < 3.0:
        score -= 50.0 + 8.0 * (3.0 - alpha)
    if alpha_min < 1.0:
        score -= 120.0 + 12.0 * (1.0 - alpha_min)
    if dd_worst > 0.25:
        score -= 260.0 + 50.0 * dd_worst
    elif dd_worst > 0.0:
        score -= 160.0 + 40.0 * dd_worst
    if turnover > max_turnover:
        score -= 100.0 + 10.0 * (turnover - max_turnover)
    return float(score)


def _spec_grid(quick: bool = False) -> list[OverlaySpec]:
    specs: list[OverlaySpec] = []
    specs.append(
        OverlaySpec(
            base_overlay=0.0,
            base_safe_only=False,
            ema_span=256,
            risk_hi=99.0,
            risk_lo=-99.0,
            edge_hi=99.0,
            edge_lo=-99.0,
            trend_lookback=128,
            trend_max=0.0,
            trend_min=0.0,
            dd_lookback=0,
            dd_min=0.0,
            dd_max=np.inf,
            down_overlay=0.0,
            up_overlay=0.0,
            max_step=0.01,
            min_hold=32,
            deadzone=0.01,
        )
    )
    base_overlays = (0.0, 0.015) if quick else (0.0, 0.015, 0.02)
    base_safe_modes = (False,) if quick else (False, True)
    ema_spans = (128,) if quick else (64, 128, 256)
    risk_his = (0.5, 1.0) if quick else (0.5, 1.0, 1.5)
    risk_los = (-0.5,) if quick else (-0.5, 0.0)
    edge_his = (99.0, 0.0) if quick else (99.0, -0.25, 0.0, 0.25)
    edge_los = (-99.0, 0.25) if quick else (-99.0, -0.50, 0.0, 0.25)
    trend_lookbacks = (128,) if quick else (64, 128)
    trend_maxes = (0.02,) if quick else (-0.04, -0.02, 0.0, 0.02)
    trend_mins = (0.0,)
    dd_lookbacks = (0, 256) if quick else (0, 128, 256, 512)
    dd_mins = (0.0, 0.05) if quick else (0.0, 0.02, 0.05, 0.08)
    down_overlays = (0.12, 0.32) if quick else (0.04, 0.06, 0.12, 0.24, 0.40, 0.52)
    up_overlays = (0.02, 0.04) if quick else (0.0, 0.02, 0.04)
    max_steps = (0.02,) if quick else (0.01, 0.02)
    for base_overlay in base_overlays:
        for base_safe_only in base_safe_modes:
            for ema_span in ema_spans:
                for risk_hi in risk_his:
                    for risk_lo in risk_los:
                        for edge_hi in edge_his:
                            for edge_lo in edge_los:
                                for trend_lookback in trend_lookbacks:
                                    for trend_max in trend_maxes:
                                        for trend_min in trend_mins:
                                            for dd_lookback in dd_lookbacks:
                                                for dd_min in dd_mins:
                                                    if dd_lookback <= 0 and dd_min > 0.0:
                                                        continue
                                                    for down_overlay in down_overlays:
                                                        for up_overlay in up_overlays:
                                                            for max_step in max_steps:
                                                                specs.append(
                                                                    OverlaySpec(
                                                                        base_overlay=base_overlay,
                                                                        base_safe_only=base_safe_only,
                                                                        ema_span=ema_span,
                                                                        risk_hi=risk_hi,
                                                                        risk_lo=risk_lo,
                                                                        edge_hi=edge_hi,
                                                                        edge_lo=edge_lo,
                                                                        trend_lookback=trend_lookback,
                                                                        trend_max=trend_max,
                                                                        trend_min=trend_min,
                                                                        dd_lookback=dd_lookback,
                                                                        dd_min=dd_min,
                                                                        dd_max=0.08,
                                                                        down_overlay=down_overlay,
                                                                        up_overlay=up_overlay,
                                                                        max_step=max_step,
                                                                        min_hold=32,
                                                                        deadzone=0.01,
                                                                    )
                                                                )
    return specs


def _spec_from_config(cfg: dict) -> OverlaySpec:
    bc_cfg = cfg.get("bc", {}) or {}
    return OverlaySpec(
        base_overlay=float(bc_cfg.get("benchmark_overlay_lowfreq_base", 0.0)),
        base_safe_only=str(bc_cfg.get("benchmark_overlay_lowfreq_base_mode", "always")).lower()
        in {"safe", "safe_only", "risk_off"},
        ema_span=int(bc_cfg.get("benchmark_overlay_lowfreq_ema_span", 256)),
        risk_hi=float(bc_cfg.get("benchmark_overlay_lowfreq_risk_hi", 1.0)),
        risk_lo=float(bc_cfg.get("benchmark_overlay_lowfreq_risk_lo", 0.0)),
        edge_hi=float(bc_cfg.get("benchmark_overlay_lowfreq_edge_hi", 0.0)),
        edge_lo=float(bc_cfg.get("benchmark_overlay_lowfreq_edge_lo", 0.0)),
        trend_lookback=int(bc_cfg.get("benchmark_overlay_lowfreq_trend_lookback", 128)),
        trend_max=float(bc_cfg.get("benchmark_overlay_lowfreq_trend_max", 0.0)),
        trend_min=float(bc_cfg.get("benchmark_overlay_lowfreq_trend_min", 0.0)),
        dd_lookback=int(bc_cfg.get("benchmark_overlay_lowfreq_dd_lookback", 0)),
        dd_min=float(bc_cfg.get("benchmark_overlay_lowfreq_dd_min", 0.0)),
        dd_max=float(bc_cfg.get("benchmark_overlay_lowfreq_dd_max", np.inf)),
        down_overlay=abs(float(bc_cfg.get("benchmark_overlay_lowfreq_down", 0.0))),
        up_overlay=abs(float(bc_cfg.get("benchmark_overlay_lowfreq_up", 0.0))),
        max_step=float(bc_cfg.get("benchmark_overlay_max_step", 0.01)),
        min_hold=int(bc_cfg.get("benchmark_overlay_lowfreq_min_hold", 0)),
        deadzone=float(bc_cfg.get("benchmark_overlay_deadzone", 0.0)),
    )


def _slice_pair(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = max(len(arr) // 2, 1)
    return arr[:mid], arr[mid:]


def _eval_spec(
    *,
    returns: np.ndarray,
    predictive: np.ndarray,
    spec: OverlaySpec,
    cfg: dict,
    costs_cfg: dict,
    benchmark: float,
    risk_indices: tuple[int, ...],
    edge_indices: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    pos = _lowfreq_positions(
        predictive=predictive,
        returns=returns,
        spec=spec,
        benchmark=benchmark,
        risk_indices=risk_indices,
        edge_indices=edge_indices,
    )
    t = min(len(returns), len(pos))
    metrics = Backtest(
        returns[:t],
        pos[:t],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=np.full(t, benchmark, dtype=np.float64),
    ).run()
    return _metrics_record(metrics, pos[:t], benchmark), pos[:t]


def _run_fold(
    split,
    features_df,
    raw_returns,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
    specs: list[OverlaySpec],
    risk_indices: tuple[int, ...],
    edge_indices: tuple[int, ...],
    include_candidates: bool = False,
) -> dict:
    data_cfg = cfg.get("data", {})
    costs_cfg = cfg.get("costs", {})
    ac_cfg = cfg.get("ac", {})
    bc_cfg = cfg.get("bc", {})
    reward_cfg = cfg.get("reward", {})
    seq_len = int(data_cfg.get("seq_len", 64))
    benchmark = _benchmark_position_value(cfg)
    wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    runtime = prepare_fold_runtime(
        fold_idx=split.fold_idx,
        checkpoint_dir=checkpoint_dir,
        ac_cfg=ac_cfg,
        resume=False,
        start_from="test",
        stop_after="test",
    )
    if not runtime["has_wm_ckpt"]:
        raise FileNotFoundError(f"fold {split.fold_idx}: missing WM checkpoint: {runtime['wm_path']}")

    fold_inputs = prepare_fold_inputs(
        fold_idx=split.fold_idx,
        wfo_dataset=wfo_dataset,
        cfg=cfg,
        costs_cfg=costs_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        action_stats_fn=_action_stats,
        format_action_stats_fn=lambda s: "",
        benchmark_position=benchmark,
        forward_window_stats_fn=_forward_window_stats,
        log_ts=_ts,
    )
    ensemble, wm_trainer = prepare_world_model_stage(
        fold_idx=split.fold_idx,
        obs_dim=wfo_dataset.obs_dim,
        cfg=cfg,
        device=device,
        has_wm=True,
        wm_path=runtime["wm_path"],
        wfo_dataset=wfo_dataset,
        oracle_positions=fold_inputs["oracle_positions"],
        val_oracle_positions=fold_inputs["val_oracle_positions"],
        train_returns=fold_inputs["train_returns"],
        train_regime_probs=fold_inputs["train_regime_probs"],
        val_regime_probs=fold_inputs["val_regime_probs"],
        log_ts=_ts,
    )
    enc_train = wm_trainer.encode_sequence(wfo_dataset.train_features, actions=None, seq_len=seq_len)
    bundle = build_wm_predictive_state_bundle(
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        z_train=enc_train["z"],
        h_train=enc_train["h"],
        seq_len=seq_len,
        ac_cfg=ac_cfg,
        log_ts=_ts,
    )
    if bundle is None:
        raise RuntimeError(f"fold {split.fold_idx}: predictive state unavailable")

    max_turnover = float(ac_cfg.get("selector_max_turnover", 6.5))
    candidates = []
    for spec in specs:
        val_rec, val_pos = _eval_spec(
            returns=wfo_dataset.val_returns,
            predictive=bundle["val"],
            spec=spec,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark=benchmark,
            risk_indices=risk_indices,
            edge_indices=edge_indices,
        )
        val_ret_first, val_ret_second = _slice_pair(wfo_dataset.val_returns)
        val_pred_first, val_pred_second = _slice_pair(bundle["val"])
        val_first, _ = _eval_spec(
            returns=val_ret_first,
            predictive=val_pred_first,
            spec=spec,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark=benchmark,
            risk_indices=risk_indices,
            edge_indices=edge_indices,
        )
        val_second, _ = _eval_spec(
            returns=val_ret_second,
            predictive=val_pred_second,
            spec=spec,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark=benchmark,
            risk_indices=risk_indices,
            edge_indices=edge_indices,
        )
        candidates.append({
            "spec": asdict(spec),
            "val": val_rec,
            "val_first": val_first,
            "val_second": val_second,
            "score": _robust_score(val_rec, val_first, val_second, max_turnover=max_turnover),
            "legacy_score": _score(val_rec, max_turnover=max_turnover),
        })
    best = max(candidates, key=lambda x: x["score"])
    best_spec = OverlaySpec(**best["spec"])
    test_rec, _test_pos = _eval_spec(
        returns=wfo_dataset.test_returns,
        predictive=bundle["test"],
        spec=best_spec,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark=benchmark,
        risk_indices=risk_indices,
        edge_indices=edge_indices,
    )
    print(
        f"[Plan011LowFreq] fold={split.fold_idx} "
        f"val alpha={_fmt(best['val']['alpha_excess_pt'])} dd={_fmt(best['val']['maxdd_delta_pt'])} "
        f"split=({_fmt(best['val_first']['alpha_excess_pt'])}/{_fmt(best['val_second']['alpha_excess_pt'])}) "
        f"to={best['val']['turnover']:.2f} | "
        f"test alpha={_fmt(test_rec['alpha_excess_pt'])} dd={_fmt(test_rec['maxdd_delta_pt'])} "
        f"to={test_rec['turnover']:.2f}"
    )
    out = {
        "fold": int(split.fold_idx),
        "selected": best,
        "test": test_rec,
        "top_val": sorted(candidates, key=lambda x: x["score"], reverse=True)[:10],
    }
    if include_candidates:
        out["candidates"] = candidates
    return out


def _spec_key(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _select_global_spec(results: list[dict[str, Any]], *, max_turnover: float) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in results:
        for cand in row.get("candidates", []):
            key = _spec_key(cand["spec"])
            item = grouped.setdefault(
                key,
                {
                    "spec": cand["spec"],
                    "folds": [],
                    "score": 0.0,
                    "alpha_min": float("inf"),
                    "dd_worst": -float("inf"),
                    "turnover_sum": 0.0,
                },
            )
            val = cand["val"]
            first = cand["val_first"]
            second = cand["val_second"]
            alpha_min = min(
                float(val["alpha_excess_pt"]),
                float(first["alpha_excess_pt"]),
                float(second["alpha_excess_pt"]),
            )
            dd_worst = max(
                float(val["maxdd_delta_pt"]),
                float(first["maxdd_delta_pt"]),
                float(second["maxdd_delta_pt"]),
            )
            turnover = float(val["turnover"])
            item["folds"].append({
                "fold": row["fold"],
                "val": val,
                "val_first": first,
                "val_second": second,
                "score": cand["score"],
            })
            item["alpha_min"] = min(float(item["alpha_min"]), alpha_min)
            item["dd_worst"] = max(float(item["dd_worst"]), dd_worst)
            item["turnover_sum"] = float(item["turnover_sum"]) + turnover
            item["score"] = float(item["score"]) + float(cand["score"])

    best_item = None
    best_score = -float("inf")
    for item in grouped.values():
        n = max(len(item["folds"]), 1)
        score = float(item["score"])
        avg_turnover = float(item["turnover_sum"]) / n
        if float(item["alpha_min"]) < -1.0:
            score -= 250.0 + 25.0 * abs(float(item["alpha_min"]))
        if float(item["dd_worst"]) > 0.0:
            score -= 250.0 + 80.0 * float(item["dd_worst"])
        if avg_turnover > max_turnover:
            score -= 200.0 + 20.0 * (avg_turnover - max_turnover)
        item["global_score"] = score
        item["avg_turnover"] = avg_turnover
        if score > best_score:
            best_score = score
            best_item = item
    if best_item is None:
        raise RuntimeError("global selector found no candidates")
    return best_item


def _write_report(payload: dict[str, Any], output_md: str) -> None:
    lines = [
        "# Plan011 Low-Frequency WM Overlay Probe",
        "",
        f"- config: `{payload['config']}`",
        f"- checkpoint_dir: `{payload['checkpoint_dir']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "| fold | val AlphaEx | val MaxDDDelta | val TO | test AlphaEx | test MaxDDDelta | test TO | selected |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        spec = row["selected"]["spec"]
        lines.append(
            "| "
            f"{row['fold']} | "
            f"{_fmt(row['selected']['val']['alpha_excess_pt'])} | "
            f"{_fmt(row['selected']['val']['maxdd_delta_pt'])} | "
            f"{row['selected']['val']['turnover']:.2f} | "
            f"{_fmt(row['test']['alpha_excess_pt'])} | "
            f"{_fmt(row['test']['maxdd_delta_pt'])} | "
            f"{row['test']['turnover']:.2f} | "
            f"ema={spec['ema_span']} hi={spec['risk_hi']} lo={spec['risk_lo']} "
            f"ehi={spec['edge_hi']} elo={spec['edge_lo']} "
            f"base={spec['base_overlay']} safe={spec.get('base_safe_only', False)} tlb={spec['trend_lookback']} "
            f"ddlb={spec['dd_lookback']} ddmin={spec['dd_min']} "
            f"down={spec['down_overlay']} up={spec['up_overlay']} step={spec['max_step']} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v4_bconly.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/plan010_risk_focus_raw_wm_s007")
    parser.add_argument("--folds", default="3,4,5")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick-grid", action="store_true")
    parser.add_argument("--config-spec-only", action="store_true")
    parser.add_argument("--global-select", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
    set_seed(args.seed)
    symbol = cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_dir = resolve_cache_dir(args.checkpoint_dir, cfg)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=cfg.get("data", {}).get("extra_series_mode", "derived"),
        extra_series_include=cfg.get("data", {}).get("extra_series_include"),
        include_funding=bool(cfg.get("data", {}).get("include_funding", True)),
        include_oi=bool(cfg.get("data", {}).get("include_oi", True)),
        include_mark=bool(cfg.get("data", {}).get("include_mark", True)),
    )
    splits, selected = select_wfo_splits(build_wfo_splits(features_df, cfg.get("data", {})), args.folds)
    risk_indices = tuple(int(i) for i in cfg.get("bc", {}).get(
        "benchmark_overlay_risk_gate_indices",
        [10, 11, 12, 13, 15, 16, 17, 18],
    ))
    edge_source = cfg.get("bc", {}).get("benchmark_overlay_edge_indices")
    if not edge_source:
        edge_source = cfg.get("bc", {}).get("benchmark_overlay_edge_protect_indices", [])
    edge_indices = tuple(int(i) for i in edge_source or [])
    specs = [_spec_from_config(cfg)] if args.config_spec_only else _spec_grid(quick=bool(args.quick_grid))
    global_selected = None
    if args.global_select and not args.config_spec_only:
        val_results = [
            _run_fold(
                split,
                features_df,
                raw_returns,
                cfg,
                args.device,
                args.checkpoint_dir,
                specs,
                risk_indices,
                edge_indices,
                include_candidates=True,
            )
            for split in splits
        ]
        max_turnover = float(cfg.get("ac", {}).get("selector_max_turnover", 6.5))
        global_selected = _select_global_spec(val_results, max_turnover=max_turnover)
        spec = OverlaySpec(**global_selected["spec"])
        print(
            "[Plan011LowFreq] global selected "
            f"score={global_selected['global_score']:.2f} "
            f"alpha_min={global_selected['alpha_min']:+.2f} "
            f"dd_worst={global_selected['dd_worst']:+.2f} "
            f"spec={global_selected['spec']}"
        )
        results = [
            _run_fold(
                split,
                features_df,
                raw_returns,
                cfg,
                args.device,
                args.checkpoint_dir,
                [spec],
                risk_indices,
                edge_indices,
            )
            for split in splits
        ]
    else:
        results = [
            _run_fold(
                split,
                features_df,
                raw_returns,
                cfg,
                args.device,
                args.checkpoint_dir,
                specs,
                risk_indices,
                edge_indices,
            )
            for split in splits
        ]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output or f"codex_outputs/{ts}_plan011_lowfreq_overlay_folds{args.folds.replace(',', '')}"
    payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": selected if selected is not None else [int(s.fold_idx) for s in splits],
        "results": results,
    }
    if global_selected is not None:
        payload["global_selected"] = global_selected
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    json_path = output_base + ".json"
    md_path = output_base + ".md"
    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, md_path)
    print(f"[Plan011LowFreq] wrote {json_path}")
    print(f"[Plan011LowFreq] wrote {md_path}")


if __name__ == "__main__":
    main()
