from __future__ import annotations

import argparse
import json
import os

import numpy as np

from unidream.cli.ac_fire_timing_probe import _load_actor_for_run, _parse_run
from unidream.cli.train import _action_stats, _benchmark_position_value, _benchmark_positions
from unidream.data.features import augment_with_rebound_features
from unidream.eval.backtest import Backtest, compute_pnl
from unidream.experiments.policy_fire import (
    forward_incremental_mean,
    forward_return_mean,
    predict_with_policy_flags,
)
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


def _pnl_bar(ret: float, pos: float, prev_pos: float, costs_cfg: dict) -> float:
    delta = abs(float(pos) - float(prev_pos))
    spread = (float(costs_cfg.get("spread_bps", 5.0)) / 10000.0) / 2.0 * delta
    fee = float(costs_cfg.get("fee_rate", 0.0004)) * delta
    slip = (float(costs_cfg.get("slippage_bps", 2.0)) / 10000.0) * delta
    return float(pos) * float(ret) - spread - fee - slip


def _maxdd_interval(equity: np.ndarray) -> dict:
    if len(equity) == 0:
        return {"peak": 0, "trough": 0, "maxdd": 0.0}
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / (peaks + 1e-12)
    trough = int(np.argmin(dd))
    peak = int(np.argmax(equity[: trough + 1])) if trough >= 0 else 0
    return {"peak": peak, "trough": trough, "maxdd": float(dd[trough])}


def _fire_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(np.asarray(mask, dtype=bool)):
        if active and start is None:
            start = i
        elif not active and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _guard_by_mask(positions: np.ndarray, no_adapter: np.ndarray, mask: np.ndarray) -> np.ndarray:
    guarded = np.asarray(positions, dtype=np.float64).copy()
    t = min(len(guarded), len(no_adapter), len(mask))
    guarded[:t][np.asarray(mask[:t], dtype=bool)] = np.asarray(no_adapter[:t], dtype=np.float64)[
        np.asarray(mask[:t], dtype=bool)
    ]
    return guarded


def _guard_by_delta_scale(
    positions: np.ndarray,
    no_adapter: np.ndarray,
    fire: np.ndarray,
    scale: float,
) -> np.ndarray:
    t = min(len(positions), len(no_adapter), len(fire))
    guarded = np.asarray(positions[:t], dtype=np.float64).copy()
    base = np.asarray(no_adapter[:t], dtype=np.float64)
    active = np.asarray(fire[:t], dtype=bool)
    guarded[active] = base[active] + float(scale) * (guarded[active] - base[active])
    return guarded


def _guard_by_pre_dd(
    positions: np.ndarray,
    no_adapter: np.ndarray,
    returns: np.ndarray,
    fire: np.ndarray,
    costs_cfg: dict,
    threshold: float,
) -> np.ndarray:
    t = min(len(positions), len(no_adapter), len(returns), len(fire))
    out = np.asarray(positions[:t], dtype=np.float64).copy()
    equity = 1.0
    peak = 1.0
    prev_pos = 0.0
    for i in range(t):
        dd = equity / max(peak, 1e-12) - 1.0
        if bool(fire[i]) and dd <= -float(threshold):
            out[i] = float(no_adapter[i])
        pnl = _pnl_bar(float(returns[i]), float(out[i]), prev_pos, costs_cfg)
        equity *= float(np.exp(pnl))
        peak = max(peak, equity)
        prev_pos = float(out[i])
    return out


def _guard_by_cooldown(
    positions: np.ndarray,
    no_adapter: np.ndarray,
    fire: np.ndarray,
    cooldown: int,
) -> np.ndarray:
    t = min(len(positions), len(no_adapter), len(fire))
    out = np.asarray(positions[:t], dtype=np.float64).copy()
    remaining = 0
    for i in range(t):
        if bool(fire[i]):
            if remaining > 0:
                out[i] = float(no_adapter[i])
            else:
                remaining = int(cooldown)
        if remaining > 0:
            remaining -= 1
    return out


def _predictive_names(cfg: dict) -> list[str]:
    wm_cfg = cfg.get("world_model", {})
    ac_cfg = cfg.get("ac", {})
    heads = list(ac_cfg.get("wm_predictive_state_heads", ["return", "vol", "drawdown"]))
    return_h = [int(h) for h in wm_cfg.get("return_horizons", [wm_cfg.get("return_horizon", 10)])]
    risk_h = [int(h) for h in wm_cfg.get("risk_horizons", return_h)]
    names: list[str] = []
    if "return" in heads:
        names.extend([f"wm_pred_return_h{h}" for h in return_h])
    if "vol" in heads:
        names.extend([f"wm_pred_vol_h{h}" for h in risk_h])
    if "drawdown" in heads:
        names.extend([f"wm_pred_drawdown_h{h}" for h in risk_h])
    return names


def _variant_summary(
    *,
    name: str,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    returns: np.ndarray,
    costs_cfg: dict,
    cfg: dict,
    benchmark_position: float,
) -> dict:
    t = min(len(positions), len(no_adapter), len(returns))
    pos = np.asarray(positions[:t], dtype=np.float64)
    base = np.asarray(no_adapter[:t], dtype=np.float64)
    rets = np.asarray(returns[:t], dtype=np.float64)
    delta = pos - base
    fire = np.abs(delta) > 1e-6
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
    equity = np.exp(np.cumsum(pnl))
    interval = _maxdd_interval(equity)
    in_mdd = np.zeros(t, dtype=bool)
    if interval["trough"] >= interval["peak"]:
        in_mdd[interval["peak"] : interval["trough"] + 1] = True
    return {
        "name": name,
        "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
        "sharpe_delta": float(metrics.sharpe_delta or 0.0),
        "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
        "turnover": float(stats["turnover"]),
        "long": float(stats["long"]),
        "short": float(stats["short"]),
        "flat": float(stats["flat"]),
        "fire_rate": float(np.mean(fire)),
        "fire_count": int(np.sum(fire)),
        "fire_pnl": float(np.sum(pnl[fire])) if np.any(fire) else 0.0,
        "nonfire_pnl": float(np.sum(pnl[~fire])) if np.any(~fire) else 0.0,
        "fwd_ret_16": forward_return_mean(rets, fire, 16),
        "fwd_incr_pnl_16": forward_incremental_mean(
            returns=rets,
            delta=delta,
            mask=fire,
            horizon=16,
        ),
        "mdd_peak": int(interval["peak"]),
        "mdd_trough": int(interval["trough"]),
        "mdd": float(interval["maxdd"]),
        "fire_in_mdd": int(np.sum(fire & in_mdd)),
        "fire_in_mdd_rate": float(np.mean(fire & in_mdd)) if t > 0 else 0.0,
    }


def _run_attribution(
    *,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    returns: np.ndarray,
    costs_cfg: dict,
    cfg: dict,
    benchmark_position: float,
    max_rows: int,
) -> list[dict]:
    current = _variant_summary(
        name="current",
        positions=positions,
        no_adapter=no_adapter,
        returns=returns,
        costs_cfg=costs_cfg,
        cfg=cfg,
        benchmark_position=benchmark_position,
    )
    fire = np.abs(np.asarray(positions) - np.asarray(no_adapter)) > 1e-6
    runs = _fire_runs(fire)
    rows = []
    for start, end in runs:
        mask = np.zeros_like(fire, dtype=bool)
        mask[start:end] = True
        cf = _variant_summary(
            name=f"suppress_{start}_{end}",
            positions=_guard_by_mask(positions, no_adapter, mask),
            no_adapter=no_adapter,
            returns=returns,
            costs_cfg=costs_cfg,
            cfg=cfg,
            benchmark_position=benchmark_position,
        )
        fwd16 = forward_return_mean(np.asarray(returns), mask, 16)
        incr16 = forward_incremental_mean(
            returns=np.asarray(returns),
            delta=np.asarray(positions) - np.asarray(no_adapter),
            mask=mask,
            horizon=16,
        )
        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "fwd16": float(fwd16),
                "incr16": float(incr16),
                "maxdd_improve_pt": float(current["maxdd_delta_pt"] - cf["maxdd_delta_pt"]),
                "alpha_loss_pt": float(current["alpha_excess_pt"] - cf["alpha_excess_pt"]),
                "cf_alpha_excess_pt": float(cf["alpha_excess_pt"]),
                "cf_maxdd_delta_pt": float(cf["maxdd_delta_pt"]),
            }
        )
    rows.sort(key=lambda x: (x["maxdd_improve_pt"], -abs(x["alpha_loss_pt"])), reverse=True)
    return rows[:max_rows]


def _write_markdown(path: str, *, result: dict) -> None:
    lines = [
        "# AC Fire DD Guard Probe",
        "",
        f"Run: `{result['run_label']}`",
        f"Fold: `{result['fold']}`",
        "",
        "## Variant Summary",
        "",
        "| variant | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | fire_pnl | fwd16 | incr16 | fire_in_mdd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["variants"]:
        lines.append(
            f"| {row['name']} | {row['alpha_excess_pt']:+.2f} | {row['sharpe_delta']:+.3f} "
            f"| {row['maxdd_delta_pt']:+.2f} | {row['turnover']:.2f} | "
            f"{row['long']:.1%} | {row['short']:.1%} | {row['fire_rate']:.1%} "
            f"| {row['fire_pnl']:+.4f} | {row['fwd_ret_16']:+.5f} "
            f"| {row['fwd_incr_pnl_16']:+.5f} | {row['fire_in_mdd']} |"
        )
    lines += [
        "",
        "## Top Fire Runs By MaxDD Improvement If Suppressed",
        "",
        "| start | end | len | fwd16 | incr16 | maxdd_improve | alpha_loss | cf_alpha | cf_maxddd |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["top_fire_runs"]:
        lines.append(
            f"| {row['start']} | {row['end']} | {row['length']} | {row['fwd16']:+.5f} "
            f"| {row['incr16']:+.5f} | {row['maxdd_improve_pt']:+.3f} "
            f"| {row['alpha_loss_pt']:+.2f} | {row['cf_alpha_excess_pt']:+.2f} "
            f"| {row['cf_maxdd_delta_pt']:+.2f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `pre_dd_*`, `cooldown_*`, `pred_dd_*`, and `delta_scale_*` are inference-only style guards.",
        "- `oracle_mdd_interval` and `oracle_postmin32_*` use future/test-path information and are diagnostic upper bounds, not deployable policies.",
        "- Adoption requires the normal train/test path to satisfy MaxDDDelta <= 0; this probe is a screening step.",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.ac_fire_dd_guard_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", required=True, help="label=checkpoint_dir[@ac_file][:ac|:bc]")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-runs", type=int, default=12)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "checkpoints")
    cache_dir = resolve_cache_dir(checkpoint_dir, cfg)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    feature_extras_cfg = cfg.get("feature_extras", {})
    if feature_extras_cfg.get("rebound_v1", False):
        features_df = augment_with_rebound_features(
            features_df,
            raw_returns,
            zscore_window_days=zscore_window,
            interval=interval,
            windows_hours=feature_extras_cfg.get("rebound_windows_hours", [24, 72]),
        )
        raw_returns = raw_returns.loc[features_df.index]

    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.fold))
    if not splits:
        raise RuntimeError(f"Fold {args.fold} not found")
    run = _parse_run(args.run)
    payload = _load_actor_for_run(
        run=run,
        split=splits[0],
        features_df=features_df,
        raw_returns=raw_returns,
        cfg=cfg,
        device=args.device,
    )

    actor = payload["actor"]
    enc = payload["enc_test"]
    returns = np.asarray(payload["test_returns"], dtype=np.float64)
    regime = payload["test_regime_probs"]
    advantage = payload["test_advantage_values"]
    costs_cfg = payload["costs_cfg"]
    benchmark_position = _benchmark_position_value(cfg)
    positions = actor.predict_positions(
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=args.device,
    )
    no_adapter = predict_with_policy_flags(
        actor,
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=args.device,
        use_floor=bool(getattr(actor, "use_benchmark_exposure_floor", False)),
        use_adapter=False,
    )
    t = min(len(positions), len(no_adapter), len(returns))
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    returns = returns[:t]
    fire = np.abs(positions - no_adapter) > 1e-6

    variants: list[tuple[str, np.ndarray]] = [
        ("current", positions),
        ("no_adapter", no_adapter),
    ]
    for scale in (0.75, 0.50, 0.25):
        variants.append((f"delta_scale_{scale:.2f}", _guard_by_delta_scale(positions, no_adapter, fire, scale)))
    for threshold in (
        0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
        0.22, 0.2225, 0.225, 0.2275, 0.23, 0.24,
    ):
        variants.append((f"pre_dd_{threshold:.0%}", _guard_by_pre_dd(
            positions, no_adapter, returns, fire, costs_cfg, threshold
        )))
    for cooldown in (8, 16, 32, 64):
        variants.append((f"cooldown_{cooldown}", _guard_by_cooldown(positions, no_adapter, fire, cooldown)))

    names = _predictive_names(cfg)
    if advantage is not None and len(advantage) >= t and names:
        adv = np.asarray(advantage[:t], dtype=np.float64)
        dd_cols = [i for i, name in enumerate(names[: adv.shape[1]]) if "drawdown" in name]
        if dd_cols:
            dd_risk = np.max(adv[:, dd_cols], axis=1)
            source = dd_risk[fire] if np.any(fire) else dd_risk
            for q in (0.50, 0.75, 0.90):
                thr = float(np.quantile(source, q))
                variants.append((f"pred_dd_q{int(q * 100)}", _guard_by_mask(
                    positions,
                    no_adapter,
                    fire & (dd_risk >= thr),
                )))

    current_summary = _variant_summary(
        name="current",
        positions=positions,
        no_adapter=no_adapter,
        returns=returns,
        costs_cfg=costs_cfg,
        cfg=cfg,
        benchmark_position=benchmark_position,
    )
    mdd_mask = np.zeros(t, dtype=bool)
    if current_summary["mdd_trough"] >= current_summary["mdd_peak"]:
        mdd_mask[current_summary["mdd_peak"] : current_summary["mdd_trough"] + 1] = True
    variants.append(("oracle_mdd_interval", _guard_by_mask(positions, no_adapter, fire & mdd_mask)))

    for threshold in (-0.0001, -0.00025, -0.0005):
        bad = np.zeros(t, dtype=bool)
        delta = positions - no_adapter
        for i in np.flatnonzero(fire):
            end = min(t, i + 32)
            path = np.cumsum(delta[i:end] * returns[i:end])
            if path.size and float(np.min(path)) <= threshold:
                bad[i] = True
        variants.append((f"oracle_postmin32_{threshold:+.5f}", _guard_by_mask(positions, no_adapter, bad)))

    summaries = [
        _variant_summary(
            name=name,
            positions=variant_pos,
            no_adapter=no_adapter,
            returns=returns,
            costs_cfg=costs_cfg,
            cfg=cfg,
            benchmark_position=benchmark_position,
        )
        for name, variant_pos in variants
    ]
    top_runs = _run_attribution(
        positions=positions,
        no_adapter=no_adapter,
        returns=returns,
        costs_cfg=costs_cfg,
        cfg=cfg,
        benchmark_position=benchmark_position,
        max_rows=args.max_runs,
    )
    result = {
        "run_label": run.label,
        "fold": args.fold,
        "variants": summaries,
        "top_fire_runs": top_runs,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _write_markdown(args.output_md, result=result)
    print(f"[FireDD] wrote {args.output_md}")


if __name__ == "__main__":
    main()
