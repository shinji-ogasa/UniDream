from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from unidream.cli.train import _action_stats, _benchmark_position_value, _forward_window_stats, _m2_scorecard
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
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
class UtilitySpec:
    ema_span: int
    margin: float
    max_step: float
    min_hold: int
    max_under: float
    max_over: float
    risk_penalty: float
    drawdown_bias: float
    trend_lookback: int
    trend_floor: float


def _ema_2d(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.empty_like(x)
    prev = x[0].astype(np.float64)
    out[0] = prev
    for i in range(1, len(x)):
        prev = (1.0 - alpha) * prev + alpha * x[i]
        out[i] = prev
    return out


def _ema_1d(x: np.ndarray, span: int) -> np.ndarray:
    return _ema_2d(np.asarray(x, dtype=np.float64)[:, None], span)[:, 0]


def _causal_trailing_sum(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0))])
    lb = max(int(lookback), 1)
    return csum[1:] - csum[np.maximum(np.arange(len(r)) + 1 - lb, 0)]


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


def _score(rec: dict[str, Any], first: dict[str, Any], second: dict[str, Any], *, max_turnover: float) -> float:
    alpha = float(rec["alpha_excess_pt"])
    dd = float(rec["maxdd_delta_pt"])
    turnover = float(rec["turnover"])
    alpha_min = min(float(first["alpha_excess_pt"]), float(second["alpha_excess_pt"]), alpha)
    dd_worst = max(float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]), dd)
    dd_best = min(float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]), dd)
    score = 3.0 * alpha + 2.0 * alpha_min + 18.0 * max(0.0, -dd_best) - 14.0 * max(0.0, dd_worst)
    score -= 2.0 * turnover
    if alpha_min < 1.0:
        score -= 80.0 + 10.0 * (1.0 - alpha_min)
    if dd_worst > 0.0:
        score -= 80.0 + 25.0 * dd_worst
    if turnover > max_turnover:
        score -= 100.0 + 20.0 * (turnover - max_turnover)
    return float(score)


def _spec_grid(quick: bool) -> list[UtilitySpec]:
    specs = [
        UtilitySpec(
            ema_span=128,
            margin=99.0,
            max_step=0.01,
            min_hold=32,
            max_under=0.0,
            max_over=0.0,
            risk_penalty=0.0,
            drawdown_bias=0.0,
            trend_lookback=128,
            trend_floor=-99.0,
        )
    ]
    ema_spans = (64, 128, 256) if not quick else (128, 256)
    margins = (0.0, 0.01, 0.03, 0.06) if not quick else (0.01, 0.03, 0.06)
    max_steps = (0.01, 0.02)
    max_unders = (0.06, 0.15, 0.30, 0.50) if not quick else (0.15, 0.30, 0.50)
    max_overs = (0.0, 0.03, 0.06)
    risk_penalties = (0.0, 0.02, 0.05, 0.10) if not quick else (0.0, 0.05, 0.10)
    drawdown_biases = (0.0, 0.02, 0.05) if not quick else (0.0, 0.05)
    trend_floors = (-99.0, -0.03, 0.0)
    for ema_span in ema_spans:
        for margin in margins:
            for max_step in max_steps:
                for max_under in max_unders:
                    for max_over in max_overs:
                        for risk_penalty in risk_penalties:
                            for drawdown_bias in drawdown_biases:
                                for trend_floor in trend_floors:
                                    specs.append(
                                        UtilitySpec(
                                            ema_span=ema_span,
                                            margin=margin,
                                            max_step=max_step,
                                            min_hold=32,
                                            max_under=max_under,
                                            max_over=max_over,
                                            risk_penalty=risk_penalty,
                                            drawdown_bias=drawdown_bias,
                                            trend_lookback=256,
                                            trend_floor=trend_floor,
                                        )
                                    )
    return specs


def _utility_positions(
    *,
    utility: np.ndarray,
    risk: np.ndarray | None,
    returns: np.ndarray,
    positions: np.ndarray,
    benchmark: float,
    spec: UtilitySpec,
) -> np.ndarray:
    util = np.nan_to_num(np.asarray(utility, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    util = _ema_2d(util, spec.ema_span)
    allowed = (positions >= benchmark - spec.max_under - 1e-9) & (positions <= benchmark + spec.max_over + 1e-9)
    if not np.any(allowed):
        allowed = np.isclose(positions, benchmark)
    util[:, ~allowed] = -1e9

    bench_idx = int(np.argmin(np.abs(positions - benchmark)))
    risk_sig = np.zeros(len(util), dtype=np.float64)
    if risk is not None and np.asarray(risk).size:
        risk_arr = np.nan_to_num(np.asarray(risk, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if risk_arr.ndim > 1:
            risk_arr = risk_arr.mean(axis=1)
        risk_sig = _ema_1d(risk_arr, spec.ema_span)
    if spec.risk_penalty > 0.0:
        util -= spec.risk_penalty * np.maximum(risk_sig, 0.0)[:, None] * np.abs(positions[None, :] - benchmark)
    if spec.drawdown_bias > 0.0:
        util += spec.drawdown_bias * np.maximum(risk_sig, 0.0)[:, None] * np.maximum(benchmark - positions[None, :], 0.0)

    best_idx = np.argmax(util, axis=1)
    best = positions[best_idx]
    bench_util = util[:, bench_idx]
    best_util = util[np.arange(len(util)), best_idx]
    target = np.where(best_util - bench_util >= spec.margin, best, benchmark)

    if spec.trend_floor > -50.0:
        trend = _causal_trailing_sum(returns, spec.trend_lookback)
        target = np.where((target > benchmark) & (trend < spec.trend_floor), benchmark, target)

    out = np.empty_like(target, dtype=np.float64)
    current = benchmark
    hold = spec.min_hold
    for i, raw in enumerate(target):
        if hold < spec.min_hold and abs(float(raw) - current) > 1e-8:
            raw = current
        delta = float(np.clip(float(raw) - current, -spec.max_step, spec.max_step))
        nxt = current + delta
        if abs(nxt - current) > 1e-8:
            hold = 0
        else:
            hold += 1
        current = nxt
        out[i] = current
    return out.astype(np.float32)


def _eval_positions(returns: np.ndarray, positions: np.ndarray, cfg: dict, costs_cfg: dict, benchmark: float) -> dict[str, Any]:
    t = min(len(returns), len(positions))
    metrics = Backtest(
        returns[:t],
        positions[:t],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=np.full(t, benchmark, dtype=np.float64),
    ).run()
    return _metrics_record(metrics, positions[:t], benchmark)


def _slice_pair(*arrs: np.ndarray) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    n = min(len(a) for a in arrs)
    mid = max(n // 2, 1)
    return tuple(a[:mid] for a in arrs), tuple(a[mid:n] for a in arrs)


def _predict_raw_aux(wm_trainer, wfo_dataset, seq_len: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, feats in (
        ("train", wfo_dataset.train_features),
        ("val", wfo_dataset.val_features),
        ("test", wfo_dataset.test_features),
    ):
        enc = wm_trainer.encode_sequence(feats, actions=None, seq_len=seq_len)
        out[name] = wm_trainer.predict_auxiliary_from_encoded(enc["z"], enc["h"], features=feats)
    return out


def _run_fold(split, features_df, raw_returns, cfg: dict, device: str, checkpoint_dir: str, specs: list[UtilitySpec]) -> dict[str, Any]:
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
    _ensemble, wm_trainer = prepare_world_model_stage(
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
    aux = _predict_raw_aux(wm_trainer, wfo_dataset, seq_len)
    positions = np.asarray(cfg.get("world_model", {}).get("position_utility_positions", []), dtype=np.float64)
    if positions.size == 0 or "position_utility" not in aux["val"]:
        raise RuntimeError("position_utility head unavailable")
    candidates = []
    max_turnover = float(ac_cfg.get("selector_max_turnover", 6.5))
    for spec in specs:
        val_pos = _utility_positions(
            utility=aux["val"]["position_utility"],
            risk=aux["val"].get("drawdown_excess", aux["val"].get("drawdown")),
            returns=wfo_dataset.val_returns,
            positions=positions,
            benchmark=benchmark,
            spec=spec,
        )
        val = _eval_positions(wfo_dataset.val_returns, val_pos, cfg, costs_cfg, benchmark)
        (r1, u1), (r2, u2) = _slice_pair(wfo_dataset.val_returns, aux["val"]["position_utility"])
        risk_arr = aux["val"].get("drawdown_excess", aux["val"].get("drawdown"))
        if risk_arr is None:
            risk1 = risk2 = None
        else:
            (_, risk1), (_, risk2) = _slice_pair(wfo_dataset.val_returns, risk_arr)
        first_pos = _utility_positions(
            utility=u1,
            risk=risk1,
            returns=r1,
            positions=positions,
            benchmark=benchmark,
            spec=spec,
        )
        second_pos = _utility_positions(
            utility=u2,
            risk=risk2,
            returns=r2,
            positions=positions,
            benchmark=benchmark,
            spec=spec,
        )
        first = _eval_positions(r1, first_pos, cfg, costs_cfg, benchmark)
        second = _eval_positions(r2, second_pos, cfg, costs_cfg, benchmark)
        candidates.append({
            "spec": asdict(spec),
            "val": val,
            "val_first": first,
            "val_second": second,
            "score": _score(val, first, second, max_turnover=max_turnover),
        })
    best = max(candidates, key=lambda x: x["score"])
    spec = UtilitySpec(**best["spec"])
    test_pos = _utility_positions(
        utility=aux["test"]["position_utility"],
        risk=aux["test"].get("drawdown_excess", aux["test"].get("drawdown")),
        returns=wfo_dataset.test_returns,
        positions=positions,
        benchmark=benchmark,
        spec=spec,
    )
    test = _eval_positions(wfo_dataset.test_returns, test_pos, cfg, costs_cfg, benchmark)
    print(
        f"[Plan011Utility] fold={split.fold_idx} "
        f"val alpha={_fmt(best['val']['alpha_excess_pt'])} dd={_fmt(best['val']['maxdd_delta_pt'])} "
        f"split=({_fmt(best['val_first']['alpha_excess_pt'])}/{_fmt(best['val_second']['alpha_excess_pt'])}) "
        f"to={best['val']['turnover']:.2f} | "
        f"test alpha={_fmt(test['alpha_excess_pt'])} dd={_fmt(test['maxdd_delta_pt'])} to={test['turnover']:.2f}"
    )
    return {
        "fold": int(split.fold_idx),
        "selected": best,
        "test": test,
        "top_val": sorted(candidates, key=lambda x: x["score"], reverse=True)[:10],
    }


def _write_report(payload: dict[str, Any], output_md: str) -> None:
    lines = [
        "# Plan011 Position Utility Allocator Probe",
        "",
        f"- config: `{payload['config']}`",
        f"- checkpoint_dir: `{payload['checkpoint_dir']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "| fold | val AlphaEx | val MaxDDDelta | val TO | test AlphaEx | test MaxDDDelta | test TO | spec |",
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
            f"ema={spec['ema_span']} margin={spec['margin']} under={spec['max_under']} over={spec['max_over']} "
            f"riskp={spec['risk_penalty']} ddbias={spec['drawdown_bias']} step={spec['max_step']} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v14_edgewm_bconly.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/plan011_overlay_actor_v14_edgewm_bconly_s007")
    parser.add_argument("--folds", default="3,4,5")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quick-grid", action="store_true")
    parser.add_argument("--output", default=None)
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
    specs = _spec_grid(quick=bool(args.quick_grid))
    results = [
        _run_fold(split, features_df, raw_returns, cfg, args.device, args.checkpoint_dir, specs)
        for split in splits
    ]
    output_base = args.output or f"codex_outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_plan011_utility_f{args.folds.replace(',', '')}"
    payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": selected if selected is not None else [int(s.fold_idx) for s in splits],
        "results": results,
    }
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    Path(output_base + ".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, output_base + ".md")
    print(f"[Plan011Utility] wrote {output_base}.json")
    print(f"[Plan011Utility] wrote {output_base}.md")


if __name__ == "__main__":
    main()
