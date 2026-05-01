from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from unidream.cli.ac_fire_timing_probe import _load_actor_for_run, _parse_run
from unidream.data.dataset import WFODataset
from unidream.experiments.fold_inputs import _normalized_feature_stress_signal
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


def _ts() -> str:
    return "export"


def _json_safe_time(value) -> str:
    return str(value)


def _copy_if_exists(src: str | os.PathLike, dst: str | os.PathLike) -> str | None:
    if not os.path.exists(src):
        return None
    os.makedirs(os.path.dirname(str(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _stress_stats_for_split(*, wfo_dataset: WFODataset, cfg: dict, benchmark_position: float) -> dict | None:
    oracle_cfg = cfg.get("oracle", {})
    eval_cfg = cfg.get("eval", {})
    if str(eval_cfg.get("regime_source", "")).lower() != "feature_stress_tri":
        return None
    ac_cfg = cfg.get("ac", {})
    train_signal, stats = _normalized_feature_stress_signal(
        train_features=wfo_dataset.train_features,
        fold_features=wfo_dataset.train_features,
        feature_columns=wfo_dataset.feature_columns,
        oracle_cfg=oracle_cfg,
        benchmark_position=benchmark_position,
        abs_min_position=float(ac_cfg.get("abs_min_position", 0.0)),
        abs_max_position=float(ac_cfg.get("abs_max_position", 1.0)),
    )
    centers = tuple(float(x) for x in oracle_cfg.get("stress_regime_centers", [0.0, 0.5, 1.0]))
    return {
        "source": "feature_stress_tri",
        "centers": list(centers),
        "stress_center": float(stats[0]),
        "stress_scale": float(stats[1]),
        "train_signal_mean": float(np.mean(train_signal)),
        "train_signal_std": float(np.std(train_signal)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.export_inference_bundle")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--run",
        default="Best=checkpoints/acplan7_sizing_adapter_s007@ac_best.pt:ac",
        help="label=checkpoint_dir[@ac_file][:ac|:bc]",
    )
    parser.add_argument("--sample-bars", type=int, default=0, help="0 exports the full WFO test split")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, profile = resolve_costs(cfg)
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
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.fold))
    if not splits:
        raise RuntimeError(f"Fold {args.fold} not found")
    split = splits[0]
    run = _parse_run(args.run)
    payload = _load_actor_for_run(
        run=run,
        split=split,
        features_df=features_df,
        raw_returns=raw_returns,
        cfg=cfg,
        device=args.device,
    )
    actor = payload["actor"]
    enc = payload["enc_test"]
    regime = payload["test_regime_probs"]
    advantage = payload["test_advantage_values"]
    positions = actor.predict_positions(
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=args.device,
    )

    seq_len = cfg.get("data", {}).get("seq_len", 64)
    wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    test_index = features_df.loc[(features_df.index >= split.test_start) & (features_df.index <= split.test_end)].index
    test_features = wfo_dataset.test_features
    test_returns = wfo_dataset.test_returns
    t = min(len(test_features), len(test_returns), len(positions), len(test_index))
    start_idx = 0 if int(args.sample_bars) <= 0 else max(0, t - int(args.sample_bars))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(actor, output_dir / "actor_full.pt")

    runtime_dir = Path(run.checkpoint_dir) / f"fold_{split.fold_idx}"
    wm_src = runtime_dir / "world_model.pt"
    bc_src = runtime_dir / "bc_actor.pt"
    ac_src = runtime_dir / run.ac_filename
    _copy_if_exists(wm_src, output_dir / "checkpoints" / "world_model.pt")
    _copy_if_exists(bc_src, output_dir / "checkpoints" / "bc_actor.pt")
    _copy_if_exists(ac_src, output_dir / "checkpoints" / "ac.pt")

    with open(output_dir / "model_config.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    predictive_bundle = payload.get("predictive_bundle")
    if predictive_bundle is not None:
        np.savez_compressed(
            output_dir / "predictive_state.npz",
            mean=predictive_bundle["mean"],
            std=predictive_bundle["std"],
            names=np.asarray(predictive_bundle["names"], dtype=object),
        )
    else:
        np.savez_compressed(
            output_dir / "predictive_state.npz",
            mean=np.zeros((1, 0), dtype=np.float32),
            std=np.ones((1, 0), dtype=np.float32),
            names=np.asarray([], dtype=object),
        )

    np.savez_compressed(
        output_dir / "sample_input.npz",
        features=test_features[start_idx:t].astype(np.float32),
        returns=np.asarray(test_returns[start_idx:t], dtype=np.float32),
        regime=np.asarray(regime[start_idx:t], dtype=np.float32) if regime is not None else np.zeros((t - start_idx, 0), dtype=np.float32),
        advantage=np.asarray(advantage[start_idx:t], dtype=np.float32) if advantage is not None else np.zeros((t - start_idx, 0), dtype=np.float32),
        expected_positions=np.asarray(positions[start_idx:t], dtype=np.float32),
        timestamps=np.asarray([_json_safe_time(x) for x in test_index[start_idx:t]], dtype=object),
    )
    manifest = {
        "bundle_version": 1,
        "created_by": "unidream.cli.export_inference_bundle",
        "run": {
            "label": run.label,
            "checkpoint_dir": run.checkpoint_dir,
            "mode": "ac" if run.use_ac else "bc",
            "ac_filename": run.ac_filename,
            "fold": int(split.fold_idx),
        },
        "data": {
            "symbol": symbol,
            "interval": interval,
            "start": args.start,
            "end": args.end,
            "zscore_window_days": int(zscore_window),
            "feature_columns": list(features_df.columns),
            "obs_dim": int(test_features.shape[1]),
            "seq_len": int(seq_len),
            "cost_profile": profile,
        },
        "split": {
            "train_start": _json_safe_time(split.train_start),
            "train_end": _json_safe_time(split.train_end),
            "val_start": _json_safe_time(split.val_start),
            "val_end": _json_safe_time(split.val_end),
            "test_start": _json_safe_time(split.test_start),
            "test_end": _json_safe_time(split.test_end),
        },
        "policy": {
            "benchmark_position": float(payload["benchmark_position"]),
            "last_sample_position": float(positions[t - 1]) if t else None,
            "sample_bars": int(t - start_idx),
        },
        "regime": _stress_stats_for_split(
            wfo_dataset=wfo_dataset,
            cfg=cfg,
            benchmark_position=float(payload["benchmark_position"]),
        ),
        "artifacts": {
            "actor": "actor_full.pt",
            "world_model": "checkpoints/world_model.pt",
            "predictive_state": "predictive_state.npz",
            "sample_input": "sample_input.npz",
            "config": "model_config.yaml",
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
    sample_out = {
        "positions_tail": [float(x) for x in positions[max(0, t - 10):t]],
        "last_position": float(positions[t - 1]) if t else None,
        "n_positions": int(t),
    }
    with open(output_dir / "sample_output.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sample_out, f, indent=2)
    print(f"[export] wrote inference bundle: {output_dir}")


if __name__ == "__main__":
    main()
