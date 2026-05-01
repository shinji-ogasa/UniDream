from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from unidream.cli.train import (
    _action_stats,
    _benchmark_position_value,
    _benchmark_positions,
    _fmt_action_stats,
    _forward_window_stats,
    _m2_scorecard,
)
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest, compute_pnl
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import build_bc_trainer
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.fire_diagnostics import evaluate_fire_danger_diagnostics
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.policy_fire import predict_with_policy_flags as _predict_with_policy_flags
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage


def _ts() -> str:
    return "probe"


@dataclass(frozen=True)
class ProbeRun:
    label: str
    checkpoint_dir: str
    use_ac: bool
    ac_filename: str = "ac.pt"


def _parse_run(spec: str) -> ProbeRun:
    parts = spec.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Run spec must be label=checkpoint_dir[:bc|:ac], got: {spec}")
    label, raw_path = parts
    mode = "ac"
    if raw_path.endswith(":bc"):
        raw_path = raw_path[:-3]
        mode = "bc"
    elif raw_path.endswith(":ac"):
        raw_path = raw_path[:-3]
        mode = "ac"
    ac_filename = "ac.pt"
    if "@" in raw_path:
        raw_path, ac_filename = raw_path.rsplit("@", 1)
        if not raw_path or not ac_filename:
            raise ValueError(f"Run spec has invalid checkpoint file override: {spec}")
    return ProbeRun(label=label, checkpoint_dir=raw_path, use_ac=(mode == "ac"), ac_filename=ac_filename)


def _load_actor_for_run(
    *,
    run: ProbeRun,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    device: str,
) -> dict:
    cfg_local = json.loads(json.dumps(cfg))
    costs_cfg = cfg_local.get("costs", {})
    ac_cfg = cfg_local.get("ac", {})
    bc_cfg = cfg_local.get("bc", {})
    reward_cfg = cfg_local.get("reward", {})
    seq_len = cfg_local.get("data", {}).get("seq_len", 64)
    benchmark_position = _benchmark_position_value(cfg_local)

    wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    runtime = prepare_fold_runtime(
        fold_idx=split.fold_idx,
        checkpoint_dir=run.checkpoint_dir,
        ac_cfg=ac_cfg,
        resume=False,
        start_from="test",
        stop_after="test",
    )
    if not runtime["has_wm_ckpt"] or not runtime["has_bc_ckpt"]:
        raise FileNotFoundError(f"{run.label}: missing WM/BC checkpoint: {runtime}")
    ac_path = (
        os.path.join(runtime["fold_ckpt_dir"], run.ac_filename)
        if run.use_ac else runtime["ac_path"]
    )
    has_ac = os.path.exists(ac_path)
    if run.use_ac and not has_ac:
        raise FileNotFoundError(f"{run.label}: requested AC actor but missing {ac_path}")

    fold_inputs = prepare_fold_inputs(
        wfo_dataset=wfo_dataset,
        cfg=cfg_local,
        costs_cfg=costs_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        action_stats_fn=_action_stats,
        format_action_stats_fn=_fmt_action_stats,
        benchmark_position=benchmark_position,
        forward_window_stats_fn=_forward_window_stats,
        log_ts=_ts,
    )
    ensemble, wm_trainer = prepare_world_model_stage(
        fold_idx=split.fold_idx,
        obs_dim=wfo_dataset.obs_dim,
        cfg=cfg_local,
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
    predictive_bundle = build_wm_predictive_state_bundle(
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        z_train=enc_train["z"],
        h_train=enc_train["h"],
        seq_len=seq_len,
        ac_cfg=ac_cfg,
        log_ts=_ts,
    )
    train_advantage_values = fold_inputs.get("train_advantage_values")
    test_advantage_values = fold_inputs.get("test_advantage_values")
    if predictive_bundle is not None:
        ac_cfg["advantage_conditioned"] = True
        ac_cfg["advantage_dim"] = int(predictive_bundle["train"].shape[1])
        train_advantage_values = predictive_bundle["train"]
        test_advantage_values = predictive_bundle["test"]

    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=fold_inputs["oracle_bundle"]["oracle_action_values"],
        oracle_positions=fold_inputs["oracle_positions"],
        oracle_values=fold_inputs["oracle_bundle"]["oracle_values"],
        train_regime_probs=fold_inputs["train_regime_probs"],
        outcome_edge=fold_inputs["outcome_edge"],
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        oracle_teacher_mode=fold_inputs["oracle_bundle"]["oracle_teacher_mode"],
    )
    actor = bc_setup["actor"]
    trainer = build_bc_trainer(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=fold_inputs["oracle_cfg"],
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
    )
    trainer.load(runtime["bc_path"])
    if run.use_ac:
        ckpt = torch.load(ac_path, map_location=device, weights_only=False)
        incompatible = actor.load_state_dict(ckpt["actor"], strict=False)
        optional_missing = {
            "residual_head_a.weight",
            "residual_head_a.bias",
            "residual_head_b.weight",
            "residual_head_b.bias",
            "route_head.weight",
            "route_head.bias",
            "route_delta_head.weight",
            "route_delta_head.bias",
            "route_active_head.weight",
            "route_active_head.bias",
            "route_active_class_head.weight",
            "route_active_class_head.bias",
            "route_advantage_gate.weight",
            "benchmark_overweight_sizing_adapter.weight",
            "benchmark_overweight_sizing_adapter.bias",
            "inventory_recovery_head.weight",
            "inventory_recovery_head.bias",
        }
        missing = [key for key in incompatible.missing_keys if key not in optional_missing]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{run.label}: AC actor checkpoint mismatch: "
                f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
            )
    actor.infer_adjust_rate_scale = float((ac_cfg.get("val_adjust_rate_scale_grid") or [0.5])[0])
    actor.infer_advantage_level = float(ac_cfg.get("infer_advantage_level", 0.0))
    actor.eval()

    enc_test = wm_trainer.encode_sequence(wfo_dataset.test_features, actions=None, seq_len=seq_len)
    return {
        "actor": actor,
        "enc_test": enc_test,
        "test_returns": wfo_dataset.test_returns,
        "test_regime_probs": fold_inputs["test_regime_probs"],
        "test_advantage_values": test_advantage_values,
        "costs_cfg": costs_cfg,
        "cfg": cfg_local,
        "benchmark_position": benchmark_position,
        "split": split,
        "wm_trainer": wm_trainer,
        "predictive_bundle": predictive_bundle,
    }


def _forward_sums(returns: np.ndarray, mask: np.ndarray, horizons: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    idx = np.flatnonzero(mask)
    for horizon in horizons:
        vals = []
        for i in idx:
            end = min(len(returns), i + horizon)
            if end > i:
                vals.append(float(np.sum(returns[i:end])))
        out[f"fwd_ret_{horizon}"] = float(np.mean(vals)) if vals else 0.0
    return out


def _forward_incremental_pnl(
    *,
    returns: np.ndarray,
    delta: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    idx = np.flatnonzero(mask)
    for horizon in horizons:
        vals = []
        for i in idx:
            end = min(len(returns), i + horizon)
            if end > i:
                vals.append(float(delta[i] * np.sum(returns[i:end])))
        out[f"fwd_incr_pnl_{horizon}"] = float(np.mean(vals)) if vals else 0.0
    return out


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _summarize_run(run: ProbeRun, payload: dict, horizons: list[int], device: str) -> dict:
    actor = payload["actor"]
    enc = payload["enc_test"]
    returns = np.asarray(payload["test_returns"], dtype=np.float64)
    regime = payload["test_regime_probs"]
    advantage = payload["test_advantage_values"]
    costs_cfg = payload["costs_cfg"]
    cfg = payload["cfg"]
    benchmark_position = payload["benchmark_position"]

    positions = actor.predict_positions(
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
    )
    no_adapter = _predict_with_policy_flags(
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
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    returns = returns[:t]
    delta = positions - no_adapter
    fire = np.abs(delta) > 1e-6

    metrics = Backtest(
        returns,
        positions,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=_benchmark_positions(t, cfg),
    ).run()
    stats = _action_stats(positions, benchmark_position=benchmark_position)
    scorecard = _m2_scorecard(metrics, stats, cfg)
    pnl = compute_pnl(
        returns,
        positions,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
    )
    lengths = _run_lengths(fire)
    adv_arr = np.asarray(advantage[:t], dtype=np.float64) if advantage is not None else None
    adv_summary = {}
    if adv_arr is not None and adv_arr.ndim == 2 and adv_arr.shape[0] >= t and np.any(fire):
        adv_summary = {
            "pred_adv_mean": float(np.mean(adv_arr[:t][fire, 0])),
            "pred_adv_p75": float(np.percentile(adv_arr[:t][fire, 0], 75)),
        }

    summary = {
        "label": run.label,
        "mode": "ac" if run.use_ac else "bc",
        "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
        "sharpe_delta": float(metrics.sharpe_delta or 0.0),
        "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
        "period_win": float(getattr(metrics, "period_win_rate_vs_bh", 0.0) or 0.0),
        "turnover": float(stats["turnover"]),
        "long": float(stats["long"]),
        "short": float(stats["short"]),
        "flat": float(stats["flat"]),
        "fire_rate": float(np.mean(fire)),
        "fire_count": int(np.sum(fire)),
        "mean_delta": float(np.mean(delta[fire])) if np.any(fire) else 0.0,
        "positive_delta_rate": float(np.mean(delta[fire] > 0.0)) if np.any(fire) else 0.0,
        "fire_pnl": float(np.sum(pnl[fire])) if np.any(fire) else 0.0,
        "nonfire_pnl": float(np.sum(pnl[~fire])) if np.any(~fire) else 0.0,
        "long_state_rate": float(np.mean(positions > benchmark_position + 1e-6)),
        "fire_run_count": int(len(lengths)),
        "fire_run_mean": float(np.mean(lengths)) if lengths else 0.0,
        "fire_run_max": int(max(lengths)) if lengths else 0,
        "m2_pass": bool(scorecard["m2_pass"]),
        "stretch_hit": bool(scorecard["stretch_hit"]),
    }
    summary.update(_forward_sums(returns, fire, horizons))
    summary.update(_forward_incremental_pnl(returns=returns, delta=delta, mask=fire, horizons=horizons))
    summary.update(adv_summary)
    danger = evaluate_fire_danger_diagnostics(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        fire=fire,
        costs_cfg=costs_cfg,
        horizon=32,
        rel_vol_window=64,
        mdd_rel_threshold=0.5,
        post_dd_quantile=0.8,
        include_post_dd_in_danger=True,
    )
    summary.update(
        {
            "danger_fire_rate": danger.danger_fire_rate,
            "pre_dd_danger_rate": danger.pre_dd_danger_rate,
            "future_mdd_overlap_rate": danger.future_mdd_overlap_rate,
            "global_mdd_overlap_rate": danger.global_mdd_overlap_rate,
            "safe_fire_rate": danger.safe_fire_rate,
            "safe_fire_pnl": danger.safe_fire_pnl,
            "fire_advantage_mean": danger.fire_advantage_mean,
            "post_fire_dd_contribution_mean": danger.post_fire_dd_contribution_mean,
        }
    )
    return {
        "summary": summary,
        "fire_mask": fire,
        "delta": delta,
        "positions": positions,
    }


def _overlaps(results: dict[str, dict]) -> dict[str, dict]:
    labels = list(results.keys())
    out = {}
    for i, left in enumerate(labels):
        lmask = results[left]["fire_mask"]
        for right in labels[i + 1:]:
            rmask = results[right]["fire_mask"]
            t = min(len(lmask), len(rmask))
            a = lmask[:t]
            b = rmask[:t]
            inter = int(np.sum(a & b))
            union = int(np.sum(a | b))
            out[f"{left}__{right}"] = {
                "intersection": inter,
                "union": union,
                "jaccard": float(inter / union) if union else 1.0,
                "left_only": int(np.sum(a & ~b)),
                "right_only": int(np.sum(~a & b)),
            }
    return out


def _write_markdown(path: str, *, results: dict[str, dict], overlaps: dict[str, dict], fold: int) -> None:
    lines = [
        "# AC Fire Timing Probe",
        "",
        f"Fold: `{fold}`",
        "",
        "## Performance / Fire Summary",
        "",
        "| label | mode | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | danger | preDD | futureMDD | safe | fire_pnl | safe_pnl | fwd16 | incr16 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rec in results.items():
        s = rec["summary"]
        lines.append(
            f"| {label} | {s['mode']} | {s['alpha_excess_pt']:+.2f} | {s['sharpe_delta']:+.3f} "
            f"| {s['maxdd_delta_pt']:+.2f} | {s['turnover']:.2f} | {s['long']:.1%} | {s['short']:.1%} "
            f"| {s['fire_rate']:.1%} | {s['danger_fire_rate']:.1%} | "
            f"{s['pre_dd_danger_rate']:.1%} | {s['future_mdd_overlap_rate']:.1%} | "
            f"{s['safe_fire_rate']:.1%} | {s['fire_pnl']:+.4f} | "
            f"{s['safe_fire_pnl']:+.4f} | {s.get('fwd_ret_16', 0.0):+.5f} "
            f"| {s.get('fwd_incr_pnl_16', 0.0):+.5f} |"
        )
    lines += [
        "",
        "## Fire Overlap",
        "",
        "| pair | jaccard | intersection | left_only | right_only |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, rec in overlaps.items():
        lines.append(
            f"| {pair} | {rec['jaccard']:.3f} | {rec['intersection']} | "
            f"{rec['left_only']} | {rec['right_only']} |"
        )
    lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.ac_fire_timing_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=checkpoint_dir[@ac_file][:ac|:bc]",
    )
    parser.add_argument("--horizons", default="4,8,16,32")
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
    cache_dir = "checkpoints/data_cache"
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
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.fold))
    if not splits:
        raise RuntimeError(f"Fold {args.fold} not found")
    split = splits[0]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    runs = [_parse_run(spec) for spec in args.run]

    results = {}
    for run in runs:
        print(f"[FireProbe] loading {run.label}: {run.checkpoint_dir} mode={'ac' if run.use_ac else 'bc'}")
        payload = _load_actor_for_run(
            run=run,
            split=split,
            features_df=features_df,
            raw_returns=raw_returns,
            cfg=cfg,
            device=args.device,
        )
        results[run.label] = _summarize_run(run, payload, horizons, args.device)
    overlaps = _overlaps(results)
    serializable = {
        "fold": args.fold,
        "runs": {label: rec["summary"] for label, rec in results.items()},
        "overlaps": overlaps,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    _write_markdown(args.output_md, results=results, overlaps=overlaps, fold=args.fold)
    print(f"[FireProbe] wrote {args.output_md}")


if __name__ == "__main__":
    main()
