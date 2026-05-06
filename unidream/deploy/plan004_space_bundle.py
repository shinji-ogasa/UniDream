from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from unidream.cli.ac_fire_timing_probe import _load_actor_for_run, _parse_run
from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"required artifact not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_time(value: Any) -> str:
    return str(value)


def export_plan004_space_bundle(
    *,
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    start: str,
    end: str,
    fold: int,
    seed: int,
    device: str,
    ac_filename: str = "ac.pt",
    sample_bars: int = 0,
) -> dict[str, Any]:
    set_seed(seed)
    cfg = load_config(config_path)
    cfg, profile = resolve_costs(cfg)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{start}_{end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        zscore_window=zscore_window,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(fold))
    if selected != [fold] or len(splits) != 1:
        raise RuntimeError(f"failed to select fold {fold}: got {selected}")
    split = splits[0]
    seq_len = int(data_cfg.get("seq_len", 64))
    wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)

    run = _parse_run(f"Plan004={checkpoint_dir}@{ac_filename}:ac")
    payload = _load_actor_for_run(
        run=run,
        split=split,
        features_df=features_df,
        raw_returns=raw_returns,
        cfg=cfg,
        device=device,
    )

    runtime_dir = Path(checkpoint_dir) / f"fold_{fold}"
    policy_src = runtime_dir / "plan004_policy.npz"
    summary_src = runtime_dir / "plan004_summary.json"
    with np.load(policy_src, allow_pickle=False) as policy:
        expected_positions = np.asarray(policy["positions"], dtype=np.float32)
        source = str(policy["source"][0]) if "source" in policy.files else ""
        spec = str(policy["spec"][0]) if "spec" in policy.files else ""
        status = str(policy["status"][0]) if "status" in policy.files else ""

    test_index = features_df.loc[(features_df.index >= split.test_start) & (features_df.index <= split.test_end)].index
    test_features = np.asarray(wfo_dataset.test_features, dtype=np.float32)
    test_returns = np.asarray(wfo_dataset.test_returns, dtype=np.float32)
    t = min(len(test_features), len(test_returns), len(expected_positions), len(test_index))
    start_idx = 0 if int(sample_bars) <= 0 else max(0, t - int(sample_bars))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)

    actor = payload["actor"]
    torch.save(actor, output / "actor_full.pt")
    _copy_required(runtime_dir / "world_model.pt", output / "checkpoints" / "world_model.pt")
    _copy_required(runtime_dir / "bc_actor.pt", output / "checkpoints" / "bc_actor.pt")
    _copy_required(runtime_dir / ac_filename, output / "checkpoints" / "ac.pt")
    _copy_required(policy_src, output / "policy_model.npz")
    _copy_required(summary_src, output / "plan004_summary.json")

    predictive_bundle = payload.get("predictive_bundle")
    if predictive_bundle is not None:
        np.savez_compressed(
            output / "predictive_state.npz",
            mean=predictive_bundle["mean"],
            std=predictive_bundle["std"],
            names=np.asarray(predictive_bundle["names"], dtype="<U64"),
        )
    else:
        np.savez_compressed(
            output / "predictive_state.npz",
            mean=np.zeros((1, 0), dtype=np.float32),
            std=np.ones((1, 0), dtype=np.float32),
            names=np.asarray([], dtype="<U64"),
        )

    with open(output / "model_config.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    np.savez_compressed(
        output / "sample_input.npz",
        features=test_features[start_idx:t],
        returns=test_returns[start_idx:t],
        expected_positions=expected_positions[start_idx:t],
        timestamps=np.asarray([_safe_time(x) for x in test_index[start_idx:t]], dtype="<U64"),
    )
    sample_output = {
        "last_position": float(expected_positions[t - 1]) if t else None,
        "n_positions": int(t - start_idx),
        "source": source,
        "spec": spec,
    }
    with open(output / "sample_output.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sample_output, f, ensure_ascii=False, indent=2)

    manifest = {
        "bundle_version": 3,
        "bundle_type": "plan004_residual_bc_ac",
        "created_by": "unidream.cli.export_plan004_space_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "checkpoint_dir": checkpoint_dir,
            "fold": int(fold),
            "seed": int(seed),
            "status": status,
            "source": source,
            "spec": spec,
            "ac_filename": ac_filename,
            "no_leak_scope": "mainline train generated WM/BC/AC checkpoints; Plan004 model fit uses train split and extraction settings use validation split",
        },
        "data": {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end,
            "zscore_window_days": int(zscore_window),
            "feature_columns": list(features_df.columns),
            "obs_dim": int(test_features.shape[1]),
            "seq_len": int(seq_len),
            "cost_profile": profile,
        },
        "split": {
            "train_start": _safe_time(split.train_start),
            "train_end": _safe_time(split.train_end),
            "val_start": _safe_time(split.val_start),
            "val_end": _safe_time(split.val_end),
            "test_start": _safe_time(split.test_start),
            "test_end": _safe_time(split.test_end),
        },
        "policy": {
            "benchmark_position": float(payload["benchmark_position"]),
            "last_sample_position": sample_output["last_position"],
            "sample_bars": int(t - start_idx),
        },
        "artifacts": {
            "actor": "actor_full.pt",
            "policy_model": "policy_model.npz",
            "world_model": "checkpoints/world_model.pt",
            "bc_actor": "checkpoints/bc_actor.pt",
            "ac": "checkpoints/ac.pt",
            "predictive_state": "predictive_state.npz",
            "sample_input": "sample_input.npz",
            "sample_output": "sample_output.json",
            "plan004_summary": "plan004_summary.json",
            "config": "model_config.yaml",
        },
    }
    with open(output / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(manifest), f, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.export_plan004_space_bundle")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--fold", type=int, default=13)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ac-filename", default="ac.pt")
    parser.add_argument("--sample-bars", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_dir = args.checkpoint_dir or str(cfg.get("logging", {}).get("checkpoint_dir", "checkpoints"))
    manifest = export_plan004_space_bundle(
        config_path=args.config,
        checkpoint_dir=checkpoint_dir,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        fold=args.fold,
        seed=args.seed,
        device=args.device,
        ac_filename=args.ac_filename,
        sample_bars=args.sample_bars,
    )
    print(f"[export] wrote Plan004 bundle: {args.output_dir}")
    print(json.dumps(_json_sanitize(manifest["run"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
