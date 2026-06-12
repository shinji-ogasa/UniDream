from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from unidream.cli.train import _forward_window_stats
from unidream.data.dataset import WFODataset
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _future_sum(returns: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    out = np.full(len(returns), np.nan, dtype=np.float64)
    if len(returns) <= horizon:
        return out
    csum = np.concatenate([[0.0], np.cumsum(returns)])
    idx = np.arange(0, len(returns) - horizon)
    out[idx] = csum[idx + 1 + horizon] - csum[idx + 1]
    return out


def _future_drawdown(returns: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    out = np.full(len(returns), np.nan, dtype=np.float64)
    for t in range(0, max(0, len(returns) - horizon)):
        path = np.cumsum(returns[t + 1 : t + 1 + horizon])
        path = np.concatenate([[0.0], path])
        peak = np.maximum.accumulate(path)
        out[t] = -float((path - peak).min())
    return out


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return float("nan")
    value = spearmanr(x[mask], y[mask]).correlation
    return float(value) if value is not None else float("nan")


def _decile_spread(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 100:
        return float("nan")
    xm = x[mask]
    ym = y[mask]
    lo = np.quantile(xm, 0.10)
    hi = np.quantile(xm, 0.90)
    return float(np.mean(ym[xm >= hi]) - np.mean(ym[xm <= lo]))


def _eval_matrix(pred: np.ndarray, names: list[str], returns: np.ndarray, horizons: list[int]) -> list[dict[str, Any]]:
    pred = np.asarray(pred, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    feature_names = names[: pred.shape[1]]
    if len(feature_names) < pred.shape[1]:
        feature_names += [f"feature_{i}" for i in range(len(feature_names), pred.shape[1])]
    composites = {
        "risk_mean": np.nanmean(pred, axis=1),
    }
    for i, name in enumerate(feature_names):
        composites[name] = pred[:, i]

    for name, x in composites.items():
        row: dict[str, Any] = {"name": name}
        for h in horizons:
            ret = _future_sum(returns, h)
            dd = _future_drawdown(returns, h)
            row[f"ret_ic_h{h}"] = _rank_ic(x, ret)
            row[f"ret_decile_h{h}"] = _decile_spread(x, ret)
            row[f"dd_ic_h{h}"] = _rank_ic(x, dd)
            row[f"dd_decile_h{h}"] = _decile_spread(x, dd)
        rows.append(row)
    return rows


def _run_fold(split, features_df, raw_returns, cfg: dict, device: str, checkpoint_dir: str, horizons: list[int]) -> dict:
    data_cfg = cfg.get("data", {})
    costs_cfg = cfg.get("costs", {})
    ac_cfg = cfg.get("ac", {})
    bc_cfg = cfg.get("bc", {})
    reward_cfg = cfg.get("reward", {})
    seq_len = int(data_cfg.get("seq_len", 64))
    wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    runtime = prepare_fold_runtime(
        fold_idx=split.fold_idx,
        checkpoint_dir=checkpoint_dir,
        ac_cfg=ac_cfg,
        resume=False,
        start_from="test",
        stop_after="test",
    )
    fold_inputs = prepare_fold_inputs(
        fold_idx=split.fold_idx,
        wfo_dataset=wfo_dataset,
        cfg=cfg,
        costs_cfg=costs_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        action_stats_fn=lambda positions, benchmark_position=1.0: {},
        format_action_stats_fn=lambda s: "",
        benchmark_position=float(reward_cfg.get("benchmark_position", 1.0)),
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
    return {
        "fold": int(split.fold_idx),
        "names": bundle.get("names", []),
        "val": _eval_matrix(bundle["val"], bundle.get("names", []), wfo_dataset.val_returns, horizons),
        "test": _eval_matrix(bundle["test"], bundle.get("names", []), wfo_dataset.test_returns, horizons),
    }


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    return f"{v:+.4f}"


def _write_md(payload: dict[str, Any], path: str) -> None:
    h = payload["horizons"][-1]
    lines = [
        "# Plan011 Predictive Feature Probe",
        "",
        f"- config: `{payload['config']}`",
        f"- checkpoint_dir: `{payload['checkpoint_dir']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        "",
    ]
    for fold in payload["results"]:
        lines.extend([
            f"## Fold {fold['fold']}",
            "",
            f"### Top test DD IC h{h}",
            "",
            "| name | val dd_ic | test dd_ic | test dd_decile | test ret_ic | test ret_decile |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        val_by_name = {row["name"]: row for row in fold["val"]}
        top = sorted(fold["test"], key=lambda r: r.get(f"dd_ic_h{h}", float("nan")), reverse=True)[:8]
        for row in top:
            val = val_by_name.get(row["name"], {})
            lines.append(
                f"| `{row['name']}` | {_fmt(float(val.get(f'dd_ic_h{h}', float('nan'))))} | "
                f"{_fmt(float(row.get(f'dd_ic_h{h}', float('nan'))))} | "
                f"{_fmt(float(row.get(f'dd_decile_h{h}', float('nan'))))} | "
                f"{_fmt(float(row.get(f'ret_ic_h{h}', float('nan'))))} | "
                f"{_fmt(float(row.get(f'ret_decile_h{h}', float('nan'))))} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v11_lowfreq_bconly.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/plan010_risk_focus_raw_wm_s007")
    parser.add_argument("--folds", default="3,4,5")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizons", default="16,32,64")
    parser.add_argument("--output", default="codex_outputs/plan011_predictive_feature_f345")
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
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    splits, selected = select_wfo_splits(build_wfo_splits(features_df, cfg.get("data", {})), args.folds)
    results = [
        _run_fold(split, features_df, raw_returns, cfg, args.device, args.checkpoint_dir, horizons)
        for split in splits
    ]
    payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": selected if selected is not None else [int(s.fold_idx) for s in splits],
        "horizons": horizons,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json_path = args.output + ".json"
    md_path = args.output + ".md"
    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(payload, md_path)
    print(f"[Plan011PredFeature] wrote {json_path}")
    print(f"[Plan011PredFeature] wrote {md_path}")


if __name__ == "__main__":
    main()
