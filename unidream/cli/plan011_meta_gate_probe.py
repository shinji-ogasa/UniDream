from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from unidream.cli.train import _action_stats, _benchmark_position_value, _forward_window_stats, _m2_scorecard
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
class GateSpec:
    horizon: int
    prob_threshold: float
    base_overlay: float
    deep_overlay: float
    ema_span: int
    max_step: float
    min_hold: int
    trend_lookback: int
    trend_max: float
    min_prob_hold: int


def _causal_sum(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0))])
    lb = max(int(lookback), 1)
    return csum[1:] - csum[np.maximum(np.arange(len(r)) + 1 - lb, 0)]


def _causal_vol(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    lb = max(int(lookback), 1)
    out = np.zeros(len(r), dtype=np.float64)
    for i in range(len(r)):
        w = shifted[max(0, i + 1 - lb) : i + 1]
        out[i] = float(np.std(w)) if len(w) else 0.0
    return out


def _causal_dd(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], r[:-1]])
    eq = np.cumsum(np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0))
    lb = max(int(lookback), 1)
    out = np.zeros(len(r), dtype=np.float64)
    for i in range(len(r)):
        w = eq[max(0, i + 1 - lb) : i + 1]
        out[i] = max(0.0, float(np.max(w)) - float(eq[i])) if len(w) else 0.0
    return out


def _future_labels(returns: np.ndarray, horizon: int) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        end = min(n, i + 1 + horizon)
        path = np.cumsum(r[i + 1 : end])
        if path.size == 0:
            continue
        cum = float(path[-1])
        dd = max(0.0, -float(np.min(path)))
        y[i] = int((cum <= -0.015) or (dd >= 0.035 and cum <= 0.006))
    return y


def _feature_matrix(features: np.ndarray, returns: np.ndarray, predictive: np.ndarray | None) -> np.ndarray:
    parts = [np.asarray(features, dtype=np.float64)]
    if predictive is not None:
        parts.append(np.asarray(predictive, dtype=np.float64))
    for lb in (16, 64, 256, 512):
        parts.append(_causal_sum(returns, lb)[:, None])
    for lb in (64, 256):
        parts.append(_causal_vol(returns, lb)[:, None])
    for lb in (128, 512):
        parts.append(_causal_dd(returns, lb)[:, None])
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.empty_like(x)
    prev = float(x[0])
    out[0] = prev
    for i in range(1, len(x)):
        prev = (1.0 - alpha) * prev + alpha * float(x[i])
        out[i] = prev
    return out


def _positions_from_prob(prob: np.ndarray, returns: np.ndarray, benchmark: float, spec: GateSpec) -> np.ndarray:
    p = _ema(prob, spec.ema_span)
    trend = _causal_sum(returns, spec.trend_lookback)
    raw = np.full(len(p), benchmark + spec.base_overlay, dtype=np.float64)
    active = (p >= spec.prob_threshold) & (trend <= spec.trend_max)
    raw[active] = benchmark + spec.deep_overlay
    out = np.empty_like(raw)
    current = benchmark
    hold = spec.min_hold
    active_hold = 0
    for i, target in enumerate(raw):
        if active[i]:
            active_hold += 1
        else:
            active_hold = 0
        if active[i] and active_hold < spec.min_prob_hold:
            target = current
        if hold < spec.min_hold and abs(float(target) - current) > 1e-8:
            target = current
        delta = float(np.clip(float(target) - current, -spec.max_step, spec.max_step))
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


def _score(val: dict[str, Any], first: dict[str, Any], second: dict[str, Any]) -> float:
    alpha = float(val["alpha_excess_pt"])
    alpha_min = min(alpha, float(first["alpha_excess_pt"]), float(second["alpha_excess_pt"]))
    dd_worst = max(float(val["maxdd_delta_pt"]), float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]))
    dd_best = min(float(val["maxdd_delta_pt"]), float(first["maxdd_delta_pt"]), float(second["maxdd_delta_pt"]))
    score = 3.0 * alpha + 2.5 * alpha_min + 25.0 * max(0.0, -dd_best) - 18.0 * max(0.0, dd_worst)
    if alpha_min < 1.0:
        score -= 100.0 + 12.0 * (1.0 - alpha_min)
    if dd_worst > 0.0:
        score -= 120.0 + 30.0 * dd_worst
    return float(score)


def _grid() -> list[GateSpec]:
    specs = []
    for horizon in (64, 128):
        for prob_threshold in (0.45, 0.55, 0.65, 0.75):
            for base_overlay in (0.0, 0.015, 0.02):
                for deep_overlay in (-0.15, -0.30, -0.50):
                    for ema_span in (16, 64, 128):
                        for max_step in (0.01, 0.02):
                            for trend_max in (-99.0, -0.02, 0.0):
                                specs.append(
                                    GateSpec(
                                        horizon=horizon,
                                        prob_threshold=prob_threshold,
                                        base_overlay=base_overlay,
                                        deep_overlay=deep_overlay,
                                        ema_span=ema_span,
                                        max_step=max_step,
                                        min_hold=32,
                                        trend_lookback=256,
                                        trend_max=trend_max,
                                        min_prob_hold=4,
                                    )
                                )
    return specs


def _split_pair(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = max(len(arr) // 2, 1)
    return arr[:mid], arr[mid:]


def _run_fold(split, features_df, raw_returns, cfg: dict, device: str, checkpoint_dir: str) -> dict[str, Any]:
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
        raise FileNotFoundError(runtime["wm_path"])
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
    x_train = _feature_matrix(wfo_dataset.train_features, wfo_dataset.train_returns, None if bundle is None else bundle["train"])
    x_val = _feature_matrix(wfo_dataset.val_features, wfo_dataset.val_returns, None if bundle is None else bundle["val"])
    x_test = _feature_matrix(wfo_dataset.test_features, wfo_dataset.test_returns, None if bundle is None else bundle["test"])
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    candidates = []
    for horizon in sorted({s.horizon for s in _grid()}):
        y = _future_labels(wfo_dataset.train_returns, horizon)
        valid_len = max(len(y) - horizon, 1)
        y_fit = y[:valid_len]
        if y_fit.sum() < 20 or y_fit.sum() == len(y_fit):
            continue
        model = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=7 + int(split.fold_idx) + horizon,
        )
        model.fit(x_train[:valid_len], y_fit)
        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]
        for spec in [s for s in _grid() if s.horizon == horizon]:
            pos = _positions_from_prob(p_val, wfo_dataset.val_returns, benchmark, spec)
            val = _metrics(wfo_dataset.val_returns, pos, cfg, costs_cfg, benchmark)
            rv1, rv2 = _split_pair(wfo_dataset.val_returns)
            pv1, pv2 = _split_pair(p_val)
            first = _metrics(rv1, _positions_from_prob(pv1, rv1, benchmark, spec), cfg, costs_cfg, benchmark)
            second = _metrics(rv2, _positions_from_prob(pv2, rv2, benchmark, spec), cfg, costs_cfg, benchmark)
            candidates.append({"spec": asdict(spec), "val": val, "val_first": first, "val_second": second, "score": _score(val, first, second), "p_test": p_test})
    if not candidates:
        raise RuntimeError("no gate candidates")
    best = max(candidates, key=lambda x: x["score"])
    spec = GateSpec(**best["spec"])
    test_pos = _positions_from_prob(best["p_test"], wfo_dataset.test_returns, benchmark, spec)
    test = _metrics(wfo_dataset.test_returns, test_pos, cfg, costs_cfg, benchmark)
    best = {k: v for k, v in best.items() if k != "p_test"}
    print(
        f"[Plan011MetaGate] fold={split.fold_idx} "
        f"val alpha={_fmt(best['val']['alpha_excess_pt'])} dd={_fmt(best['val']['maxdd_delta_pt'])} "
        f"split=({_fmt(best['val_first']['alpha_excess_pt'])}/{_fmt(best['val_second']['alpha_excess_pt'])}) | "
        f"test alpha={_fmt(test['alpha_excess_pt'])} dd={_fmt(test['maxdd_delta_pt'])} to={test['turnover']:.2f}"
    )
    return {"fold": int(split.fold_idx), "selected": best, "test": test, "top_val": sorted([{k: v for k, v in c.items() if k != "p_test"} for c in candidates], key=lambda x: x["score"], reverse=True)[:10]}


def _write_report(payload: dict[str, Any], output_md: str) -> None:
    lines = [
        "# Plan011 Meta Gate Probe",
        "",
        f"- config: `{payload['config']}`",
        f"- checkpoint_dir: `{payload['checkpoint_dir']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | test TO | spec |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        spec = row["selected"]["spec"]
        lines.append(
            "| "
            f"{row['fold']} | {_fmt(row['selected']['val']['alpha_excess_pt'])} | {_fmt(row['selected']['val']['maxdd_delta_pt'])} | "
            f"{_fmt(row['test']['alpha_excess_pt'])} | {_fmt(row['test']['maxdd_delta_pt'])} | {row['test']['turnover']:.2f} | "
            f"h={spec['horizon']} p={spec['prob_threshold']} base={spec['base_overlay']} deep={spec['deep_overlay']} "
            f"ema={spec['ema_span']} trend={spec['trend_max']} step={spec['max_step']} |"
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
    results = [_run_fold(split, features_df, raw_returns, cfg, args.device, args.checkpoint_dir) for split in splits]
    output_base = args.output or f"codex_outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_plan011_meta_gate_f{args.folds.replace(',', '')}"
    payload = {"config": args.config, "checkpoint_dir": args.checkpoint_dir, "folds": selected if selected is not None else [int(s.fold_idx) for s in splits], "results": results}
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    Path(output_base + ".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, output_base + ".md")
    print(f"[Plan011MetaGate] wrote {output_base}.json")
    print(f"[Plan011MetaGate] wrote {output_base}.md")


if __name__ == "__main__":
    main()
