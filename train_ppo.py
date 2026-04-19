"""PPO ベースライン学習スクリプト.

Model-free PPO を WFO データ上で学習・評価する。
UniDream（世界モデル + AC）との比較対象として使用する。

Usage:
    python train_ppo.py [--config configs/trading.yaml]
                        [--symbol BTCUSDT] [--start 2018-01-01] [--end 2024-01-01]
                        [--device auto] [--seed 42] [--resume]
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.data.download import fetch_binance_ohlcv
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.dataset import get_wfo_splits, WFODataset
from unidream.device import add_device_argument, resolve_device
from unidream.baselines.ppo import PPOTrainer, TradingEnv
from unidream.eval.backtest import Backtest
from unidream.eval.pbo import compute_pbo


_CACHE_STALE_DAYS = 7


def _cache_is_fresh(path: str, stale_days: int = _CACHE_STALE_DAYS) -> bool:
    """キャッシュファイルが存在し、stale_days 以内に更新されていれば True."""
    if not os.path.exists(path):
        return False
    import time
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < stale_days


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_costs(cfg: dict, cost_profile: str | None = None) -> tuple[dict, str]:
    resolved_cfg = dict(cfg)
    profile_name = cost_profile or cfg.get("cost_profile") or "default"
    profiles = cfg.get("cost_profiles")

    if profiles:
        if profile_name == "default":
            profile_name = "base" if "base" in profiles else next(iter(profiles))
        if profile_name not in profiles:
            available = ", ".join(profiles.keys())
            raise KeyError(f"Unknown cost profile '{profile_name}'. Available: {available}")
        resolved_cfg["costs"] = dict(profiles[profile_name])
        resolved_cfg["cost_profile"] = profile_name
    else:
        resolved_cfg["costs"] = dict(cfg.get("costs", {}))
        resolved_cfg["cost_profile"] = profile_name

    return resolved_cfg, resolved_cfg["cost_profile"]


def main():
    parser = argparse.ArgumentParser(description="PPO Baseline Training")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    add_device_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints/ppo")
    parser.add_argument("--cost-profile", default=None,
                        help="Named cost profile from config cost_profiles")
    parser.add_argument("--resume", action="store_true",
                        help="保存済みチェックポイントから再開する")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg, active_cost_profile = resolve_costs(cfg, args.cost_profile)

    set_seed(args.seed)

    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    ppo_cfg = cfg.get("ppo", {})
    costs_cfg = cfg.get("costs", {})
    data_cfg = cfg.get("data", {})

    print(f"PPO Baseline | {symbol} {interval} | {args.start} → {args.end}")
    print(f"Device: {args.device} | Resume: {args.resume}")
    total_cost_bps = (
        costs_cfg.get("spread_bps", 0.0) / 2
        + costs_cfg.get("fee_rate", 0.0) * 10000
        + costs_cfg.get("slippage_bps", 0.0)
    )
    print(
        "Costs: "
        f"profile={active_cost_profile} | "
        f"spread={costs_cfg.get('spread_bps', 0.0):.2f}bps "
        f"fee={costs_cfg.get('fee_rate', 0.0) * 10000:.2f}bps "
        f"slip={costs_cfg.get('slippage_bps', 0.0):.2f}bps "
        f"=> one-way Δpos=1 cost={total_cost_bps:.2f}bps"
    )

    # データ取得・特徴量計算（キャッシュ対応）
    cache_dir = os.path.join(args.checkpoint_dir, "data_cache")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}"
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")

    if _cache_is_fresh(features_cache) and _cache_is_fresh(returns_cache):
        print("\n[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
        print(f"  Cached: {features_df.shape}")
    else:
        print("\n[Data] Fetching OHLCV...")
        df = fetch_binance_ohlcv(symbol, interval, args.start, args.end)

        print("[Data] Computing features...")
        features_df = compute_features(
            df,
            zscore_window_days=cfg.get("normalization", {}).get("zscore_window_days", 60),
            interval=interval,
        )
        raw_returns = get_raw_returns(df)
        common_idx = features_df.index.intersection(raw_returns.index)
        features_df = features_df.loc[common_idx]
        raw_returns = raw_returns.loc[common_idx]
        print(f"  Features: {features_df.shape}")

        os.makedirs(cache_dir, exist_ok=True)
        features_df.to_parquet(features_cache)
        raw_returns.to_frame().to_parquet(returns_cache)
        print(f"  Cached to {cache_dir}")

    obs_dim = features_df.shape[1]

    print("[Data] WFO splits...")
    splits = get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )
    print(f"  {len(splits)} folds")

    fold_results = {}
    for split in splits:
        print(f"\n--- Fold {split.fold_idx} ---")
        ckpt_path = os.path.join(args.checkpoint_dir, f"ppo_fold_{split.fold_idx}.pt")
        wfo_ds = WFODataset(features_df, raw_returns, split,
                            seq_len=data_cfg.get("seq_len", 64))

        trainer = PPOTrainer(
            obs_dim=obs_dim,
            act_dim=cfg.get("actions", {}).get("n", 5),
            hidden_dim=256,
            lr=ppo_cfg.get("lr", 3e-4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_eps=ppo_cfg.get("clip_eps", 0.2),
            value_coef=ppo_cfg.get("value_coef", 0.5),
            entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            batch_size=ppo_cfg.get("batch_size", 64),
            grad_clip=ppo_cfg.get("grad_clip", 0.5),
            device=args.device,
        )

        if args.resume and os.path.exists(ckpt_path):
            print(f"  Loading checkpoint: {ckpt_path}")
            trainer.load(ckpt_path)
        else:
            env = TradingEnv(
                wfo_ds.train_returns,
                wfo_ds.train_features,
                spread_bps=costs_cfg.get("spread_bps", 5.0),
                fee_rate=costs_cfg.get("fee_rate", 0.0004),
                slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                dsr_eta=cfg.get("reward", {}).get("dsr_eta", 0.01),
                beta=cfg.get("reward", {}).get("beta", 0.1),
            )
            trainer.train(env, max_steps=ppo_cfg.get("max_steps", 500_000))
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save(ckpt_path)

        # テスト推論
        positions = trainer.predict(wfo_ds.test_dataset().features.numpy())
        test_returns = wfo_ds.test_returns
        T_min = min(len(test_returns), len(positions))

        bt = Backtest(
            test_returns[:T_min],
            positions[:T_min],
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            interval=interval,
        )
        metrics = bt.run()
        fold_results[split.fold_idx] = {"metrics": metrics}

        print(f"  Sharpe={metrics.sharpe:.3f} | MaxDD={metrics.max_drawdown:.3f} | "
              f"Calmar={metrics.calmar:.3f}")

    # サマリー
    print("\n" + "="*60)
    print("PPO Baseline Summary")
    print("="*60)
    sharpes = [r["metrics"].sharpe for r in fold_results.values()]
    for fold_idx, r in fold_results.items():
        m = r["metrics"]
        print(f"  Fold {fold_idx}: Sharpe={m.sharpe:.3f}, MaxDD={m.max_drawdown:.3f}")
    print(f"  Mean Sharpe: {np.mean(sharpes):.3f}")

    pnl_list = [r["metrics"].pnl_series for r in fold_results.values()]
    pbo = compute_pbo(pnl_list)
    print(f"  PBO: {pbo:.4f}")


if __name__ == "__main__":
    main()
