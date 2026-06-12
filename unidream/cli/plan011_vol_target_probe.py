from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from unidream.cli.train import _action_stats, _benchmark_position_value, _m2_scorecard
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class VolSpec:
    lookback: int
    target_q: float
    min_pos: float
    max_pos: float
    max_step: float
    min_hold: int
    trend_lookback: int
    trend_floor: float
    dd_lookback: int
    dd_cut: float


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(float(v)):
        return "NA"
    return f"{float(v):+.{digits}f}"


def _causal_vol(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    out = np.zeros(len(r), dtype=np.float64)
    lb = max(int(lookback), 1)
    for i in range(len(r)):
        w = shifted[max(0, i + 1 - lb) : i + 1]
        out[i] = float(np.sqrt(np.mean(w * w))) if len(w) else 0.0
    return out


def _causal_sum(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    csum = np.concatenate([[0.0], np.cumsum(shifted)])
    lb = max(int(lookback), 1)
    return csum[1:] - csum[np.maximum(np.arange(len(r)) + 1 - lb, 0)]


def _causal_dd(returns: np.ndarray, lookback: int) -> np.ndarray:
    if lookback <= 0:
        return np.zeros(len(returns), dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    eq = np.cumsum(shifted)
    out = np.zeros(len(r), dtype=np.float64)
    lb = max(int(lookback), 1)
    for i in range(len(r)):
        w = eq[max(0, i + 1 - lb) : i + 1]
        out[i] = max(0.0, float(np.max(w)) - float(eq[i])) if len(w) else 0.0
    return out


def _positions(returns: np.ndarray, train_returns: np.ndarray, benchmark: float, spec: VolSpec) -> np.ndarray:
    train_vol = _causal_vol(train_returns, spec.lookback)
    target = float(np.quantile(train_vol[train_vol > 0.0], spec.target_q)) if np.any(train_vol > 0.0) else 0.0
    vol = _causal_vol(returns, spec.lookback)
    raw = benchmark * target / np.maximum(vol, 1e-8)
    raw = np.clip(raw, spec.min_pos, spec.max_pos)
    if spec.trend_floor > -50.0:
        trend = _causal_sum(returns, spec.trend_lookback)
        raw = np.where(trend < spec.trend_floor, np.minimum(raw, benchmark), raw)
    if spec.dd_lookback > 0 and spec.dd_cut > 0.0:
        dd = _causal_dd(returns, spec.dd_lookback)
        raw = np.where(dd >= spec.dd_cut, np.minimum(raw, spec.min_pos), raw)
    out = np.empty_like(raw, dtype=np.float64)
    current = benchmark
    hold = spec.min_hold
    for i, target_pos in enumerate(raw):
        if hold < spec.min_hold and abs(float(target_pos) - current) > 1e-8:
            target_pos = current
        delta = float(np.clip(float(target_pos) - current, -spec.max_step, spec.max_step))
        nxt = current + delta
        if abs(nxt - current) > 1e-8:
            hold = 0
        else:
            hold += 1
        current = nxt
        out[i] = current
    return out.astype(np.float32)


def _metrics(returns: np.ndarray, positions: np.ndarray, cfg: dict, costs_cfg: dict, benchmark: float) -> dict[str, Any]:
    t = min(len(returns), len(positions))
    m = Backtest(
        returns[:t],
        positions[:t],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=np.full(t, benchmark, dtype=np.float64),
    ).run()
    stats = _action_stats(positions[:t], benchmark_position=benchmark)
    scorecard = _m2_scorecard(m, stats, {})
    return {
        "alpha_excess_pt": 100.0 * float(m.alpha_excess or 0.0),
        "maxdd_delta_pt": 100.0 * float(m.maxdd_delta or 0.0),
        "sharpe_delta": float(m.sharpe_delta or 0.0),
        "turnover": float(stats["turnover"]),
        "m2_pass": bool(scorecard["m2_pass"]),
        "stretch_hit": bool(scorecard["stretch_hit"]),
    }


def _grid() -> list[VolSpec]:
    specs = []
    for lookback in (64, 128, 256):
        for target_q in (0.35, 0.50):
            for min_pos in (0.70, 0.85):
                for max_pos in (1.0, 1.06):
                    for max_step in (0.02,):
                        for trend_floor in (-99.0, 0.0):
                            for dd_cut in (0.0, 0.03):
                                specs.append(
                                    VolSpec(
                                        lookback=lookback,
                                        target_q=target_q,
                                        min_pos=min_pos,
                                        max_pos=max_pos,
                                        max_step=max_step,
                                        min_hold=32,
                                        trend_lookback=256,
                                        trend_floor=trend_floor,
                                        dd_lookback=256,
                                        dd_cut=dd_cut,
                                    )
                                )
    return specs


def _score(val: dict[str, Any], first: dict[str, Any], second: dict[str, Any]) -> float:
    alpha_min = min(float(val["alpha_excess_pt"]), float(first["alpha_excess_pt"]), float(second["alpha_excess_pt"]))
    dd_worst = max(float(val["maxdd_delta_pt"]), float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]))
    dd_best = min(float(val["maxdd_delta_pt"]), float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]))
    score = 3.0 * float(val["alpha_excess_pt"]) + 2.0 * alpha_min + 50.0 * max(0.0, -dd_best) - 60.0 * max(0.0, dd_worst)
    score -= 2.0 * float(val["turnover"])
    if alpha_min < 3.0:
        score -= 10000.0 + 50.0 * (3.0 - alpha_min)
    if dd_worst > -3.0:
        score -= 120.0 + 25.0 * (dd_worst + 3.0)
    return float(score)


def _split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = max(len(arr) // 2, 1)
    return arr[:mid], arr[mid:]


def _run_fold(split, features_df, raw_returns, cfg: dict, costs_cfg: dict) -> dict[str, Any]:
    benchmark = _benchmark_position_value(cfg)
    seq_len = int(cfg.get("data", {}).get("seq_len", 64))
    ds = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    candidates = []
    for spec in _grid():
        val_pos = _positions(ds.val_returns, ds.train_returns, benchmark, spec)
        val = _metrics(ds.val_returns, val_pos, cfg, costs_cfg, benchmark)
        r1, r2 = _split(ds.val_returns)
        first = _metrics(r1, _positions(r1, ds.train_returns, benchmark, spec), cfg, costs_cfg, benchmark)
        second = _metrics(r2, _positions(r2, ds.train_returns, benchmark, spec), cfg, costs_cfg, benchmark)
        candidates.append({"spec": asdict(spec), "val": val, "val_first": first, "val_second": second, "score": _score(val, first, second)})
    best = max(candidates, key=lambda c: c["score"])
    spec = VolSpec(**best["spec"])
    test_pos = _positions(ds.test_returns, ds.train_returns, benchmark, spec)
    test = _metrics(ds.test_returns, test_pos, cfg, costs_cfg, benchmark)
    print(
        f"[Plan011VolTarget] fold={split.fold_idx} val={_fmt(best['val']['alpha_excess_pt'])}/{_fmt(best['val']['maxdd_delta_pt'])} "
        f"test={_fmt(test['alpha_excess_pt'])}/{_fmt(test['maxdd_delta_pt'])} to={test['turnover']:.2f}"
    )
    return {"fold": int(split.fold_idx), "selected": best, "test": test, "top_val": sorted(candidates, key=lambda c: c["score"], reverse=True)[:20]}


def _write_report(payload: dict[str, Any], output_md: str) -> None:
    lines = [
        "# Plan011 Causal Vol Target Probe",
        "",
        f"- config: `{payload['config']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | test TO | spec |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        spec = row["selected"]["spec"]
        lines.append(
            f"| {row['fold']} | {_fmt(row['selected']['val']['alpha_excess_pt'])} | {_fmt(row['selected']['val']['maxdd_delta_pt'])} | "
            f"{_fmt(row['test']['alpha_excess_pt'])} | {_fmt(row['test']['maxdd_delta_pt'])} | {row['test']['turnover']:.2f} | "
            f"lb={spec['lookback']} q={spec['target_q']} min={spec['min_pos']} max={spec['max_pos']} "
            f"step={spec['max_step']} trend={spec['trend_floor']} ddcut={spec['dd_cut']} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v14_edgewm_bconly.yaml")
    parser.add_argument("--folds", default="3,4,5")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
    set_seed(args.seed)
    symbol = cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_dir = resolve_cache_dir(cfg.get("logging", {}).get("checkpoint_dir", "checkpoints"), cfg)
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
    costs_cfg = cfg.get("costs", {})
    results = [_run_fold(split, features_df, raw_returns, cfg, costs_cfg) for split in splits]
    output_base = args.output or f"codex_outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_plan011_voltarget_f{args.folds.replace(',', '')}"
    payload = {"config": args.config, "folds": selected if selected is not None else [int(s.fold_idx) for s in splits], "results": results}
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    Path(output_base + ".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, output_base + ".md")
    print(f"[Plan011VolTarget] wrote {output_base}.json")
    print(f"[Plan011VolTarget] wrote {output_base}.md")


if __name__ == "__main__":
    main()
