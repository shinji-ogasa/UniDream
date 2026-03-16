"""UniDream メイン学習スクリプト.

SPEC.md の実装順序に従って以下を実行する:
  1. データ取得・特徴量計算
  2. WFO 分割
  3. train 期間で Hindsight Oracle 計算
  4. BC 初期化（Actor を oracle に模倣させる）
  5. Transformer 世界モデル学習
  6. Imagination AC fine-tune
  7. test 期間バックテスト
  8. PBO / Deflated Sharpe による過学習検出
  9. HMM レジーム別メトリクス

Usage:
    python train.py [--config configs/trading.yaml] [--symbol BTCUSDT]
                    [--start 2020-01-01] [--end 2024-01-01]
                    [--device cuda] [--seed 42]
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import yaml

from unidream.data.download import fetch_binance_ohlcv
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.oracle import hindsight_oracle_dp, oracle_to_dataset
from unidream.data.dataset import get_wfo_splits, WFODataset, SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble
from unidream.actor_critic.actor import Actor
from unidream.actor_critic.critic import Critic
from unidream.actor_critic.bc_pretrain import BCPretrainer
from unidream.actor_critic.imagination_ac import ImagACTrainer
from unidream.eval.backtest import Backtest
from unidream.eval.wfo import aggregate_wfo_results
from unidream.eval.pbo import compute_pbo, deflated_sharpe
from unidream.eval.regime import RegimeDetector, regime_metrics, print_regime_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_fold(
    fold_idx: int,
    wfo_dataset: WFODataset,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
) -> dict:
    """1 WFO fold の学習・評価を実行する.

    Returns:
        {"fold": fold_idx, "metrics": BacktestMetrics, "positions": np.ndarray}
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: train {wfo_dataset.split.train_start.date()} → "
          f"{wfo_dataset.split.train_end.date()} | "
          f"test {wfo_dataset.split.test_start.date()} → "
          f"{wfo_dataset.split.test_end.date()}")
    print(f"{'='*60}")

    ac_cfg = cfg.get("ac", {})
    bc_cfg = cfg.get("bc", {})
    wm_cfg = cfg.get("world_model", {})
    costs_cfg = cfg.get("costs", {})

    obs_dim = wfo_dataset.obs_dim

    # --------- Step 1: Hindsight Oracle ---------
    print("\n[Step 1] Hindsight Oracle DP...")
    train_returns = wfo_dataset.train_returns
    oracle_actions, oracle_values = hindsight_oracle_dp(
        train_returns,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        discount=cfg.get("oracle", {}).get("discount", 1.0),
    )
    print(f"  Oracle computed: {len(oracle_actions)} steps, "
          f"mean value={oracle_values.mean():.4f}")

    # --------- Step 2: 世界モデル学習 ---------
    print("\n[Step 2] World Model Training...")
    ensemble = build_ensemble(obs_dim, cfg)
    wm_trainer = WorldModelTrainer(ensemble, cfg, device=device)

    train_ds = wfo_dataset.train_dataset()
    val_ds = wfo_dataset.val_dataset()

    # train_ds に oracle_actions を付与して再作成
    train_ds_with_actions = SequenceDataset(
        wfo_dataset.train_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=oracle_actions[:len(wfo_dataset.train_features)],
        returns=train_returns,
    )

    fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    wm_path = os.path.join(fold_ckpt_dir, "world_model.pt")

    wm_trainer.train_on_dataset(
        train_ds_with_actions,
        val_dataset=val_ds,
        checkpoint_path=wm_path,
    )

    # train 期間の全シーケンスをエンコード（AC の初期状態として使用）
    encoded = wm_trainer.encode_sequence(
        wfo_dataset.train_features,
        actions=oracle_actions[:len(wfo_dataset.train_features)],
        seq_len=cfg.get("data", {}).get("seq_len", 64),
    )
    z_train = encoded["z"]
    h_train = encoded["h"]

    # --------- Step 3: BC 初期化 ---------
    print("\n[Step 3] BC Pre-training...")
    actor = Actor(
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        act_dim=cfg.get("actions", {}).get("n", 5),
        hidden_dim=ac_cfg.get("actor_hidden", 256),
        n_layers=ac_cfg.get("ac_layers", 2),
    )
    bc_trainer = BCPretrainer(
        actor=actor,
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        lr=bc_cfg.get("lr", 3e-4),
        batch_size=bc_cfg.get("batch_size", 256),
        n_epochs=bc_cfg.get("n_epochs", 5),
        sirl_hidden=bc_cfg.get("sirl_hidden", 128),
        device=device,
    )

    T_enc = min(len(z_train), len(oracle_actions))
    bc_trainer.train(
        z=z_train[:T_enc],
        h=h_train[:T_enc],
        oracle_actions=oracle_actions[:T_enc],
    )
    bc_path = os.path.join(fold_ckpt_dir, "bc_actor.pt")
    bc_trainer.save(bc_path)

    # --------- Step 4: Imagination AC Fine-tune ---------
    print("\n[Step 4] Imagination AC Fine-tuning...")
    critic = Critic(
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        hidden_dim=ac_cfg.get("critic_hidden", 256),
        n_layers=ac_cfg.get("ac_layers", 2),
        n_bins=wm_cfg.get("n_bins", 255),
        ema_decay=ac_cfg.get("ema_decay", 0.98),
    )

    ac_trainer = ImagACTrainer(
        actor=actor,
        critic=critic,
        ensemble=ensemble,
        cfg=cfg,
        device=device,
    )
    ac_trainer.set_oracle_data(
        z=z_train[:T_enc],
        h=h_train[:T_enc],
        oracle_actions=oracle_actions[:T_enc],
    )

    encoded_list = [{"z": z_train, "h": h_train}]
    ac_trainer.train(
        encoded_sequences=encoded_list,
        batch_size=ac_cfg.get("batch_size", 32),
    )
    ac_path = os.path.join(fold_ckpt_dir, "ac.pt")
    ac_trainer.save(ac_path)

    # --------- Step 5: Test バックテスト ---------
    print("\n[Step 5] Test Backtest...")
    test_features = wfo_dataset.test_dataset().features.numpy()
    test_returns = wfo_dataset.test_returns
    seq_len = cfg.get("data", {}).get("seq_len", 64)

    # Two-pass encoding: 学習時は oracle action 付きで h を計算しているため、
    # テスト時もactor の予測 action を渡して h を整合させる。
    # Pass 1: action=0 でエンコード → z（obs依存のみ）は正確、h は近似
    enc_pass1 = wm_trainer.encode_sequence(test_features, seq_len=seq_len)
    # Actor の greedy action を予測
    predicted_actions = actor.predict_positions(
        enc_pass1["z"], enc_pass1["h"], device=device,
    )
    # positions → action indices に変換
    from unidream.data.oracle import ACTIONS as _ACTIONS
    action_indices = np.array([
        int(np.argmin(np.abs(_ACTIONS - p))) for p in predicted_actions
    ])
    # Pass 2: 予測 action でエンコードし直して h を整合させる
    enc_test = wm_trainer.encode_sequence(
        test_features, actions=action_indices, seq_len=seq_len,
    )
    positions = actor.predict_positions(enc_test["z"], enc_test["h"], device=device)

    T_min = min(len(test_returns), len(positions))
    bt = Backtest(
        test_returns[:T_min],
        positions[:T_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
    )
    metrics = bt.run()

    print(f"  Sharpe:   {metrics.sharpe:.3f}")
    print(f"  MaxDD:    {metrics.max_drawdown:.3f}")
    print(f"  Calmar:   {metrics.calmar:.3f}")
    print(f"  TotalRet: {metrics.total_return:.4f}")

    return {
        "fold": fold_idx,
        "metrics": metrics,
        "positions": positions[:T_min],
        "test_returns": test_returns[:T_min],
    }


def main():
    parser = argparse.ArgumentParser(description="UniDream Training Script")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--symbol", default=None, help="Binance symbol (overrides config)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")

    print(f"UniDream Training | {symbol} {interval} | {args.start} → {args.end}")
    print(f"Device: {args.device} | Seed: {args.seed}")

    # --------- データ取得 ---------
    print("\n[Data] Fetching OHLCV...")
    df = fetch_binance_ohlcv(symbol, interval, args.start, args.end)
    print(f"  Raw data: {len(df)} bars ({df.index[0]} → {df.index[-1]})")

    # --------- 特徴量計算 ---------
    print("[Data] Computing features...")
    features_df = compute_features(
        df,
        zscore_window_days=cfg.get("normalization", {}).get("zscore_window_days", 60),
        interval=interval,
    )
    raw_returns = get_raw_returns(df)
    # features と returns のインデックスを揃える
    common_idx = features_df.index.intersection(raw_returns.index)
    features_df = features_df.loc[common_idx]
    raw_returns = raw_returns.loc[common_idx]
    print(f"  Features: {features_df.shape} | obs_dim={features_df.shape[1]}")

    # --------- WFO 分割 ---------
    print("[Data] WFO splits...")
    data_cfg = cfg.get("data", {})
    splits = get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )
    print(f"  {len(splits)} folds")

    if len(splits) == 0:
        print("ERROR: WFO splits が空です。データ期間が短すぎます。")
        return

    # --------- 各 Fold の学習・評価 ---------
    fold_results = {}
    for split in splits:
        wfo_ds = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=data_cfg.get("seq_len", 64),
        )
        result = run_fold(
            fold_idx=split.fold_idx,
            wfo_dataset=wfo_ds,
            cfg=cfg,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )
        fold_results[split.fold_idx] = result

    # --------- PBO / Deflated Sharpe ---------
    print("\n[Eval] PBO & Deflated Sharpe...")
    pnl_list = [r["metrics"].pnl_series for r in fold_results.values()]
    eval_cfg = cfg.get("eval", {})
    pbo = compute_pbo(
        pnl_list,
        n_combinations=eval_cfg.get("pbo_n_trials"),
    )
    print(f"  PBO: {pbo:.4f} (< 0.5 が望ましい)")

    all_sharpes = [r["metrics"].sharpe for r in fold_results.values()]
    best_sharpe = max(all_sharpes)
    n_trials = len(splits)
    T_avg = int(np.mean([len(r["metrics"].pnl_series) for r in fold_results.values()]))
    dsr = deflated_sharpe(best_sharpe, n_trials=n_trials, T=T_avg)
    print(f"  Deflated Sharpe: {dsr:.4f} (> 0 が望ましい)")

    # --------- レジーム別メトリクス ---------
    print("\n[Eval] Regime Analysis...")
    all_test_returns = np.concatenate([r["test_returns"] for r in fold_results.values()])
    all_positions = np.concatenate([r["positions"] for r in fold_results.values()])

    try:
        detector = RegimeDetector(n_states=cfg.get("eval", {}).get("hmm_n_states", 3))
        regimes = detector.fit_predict(all_test_returns)
        regime_results = regime_metrics(
            all_test_returns,
            all_positions,
            regimes,
            n_states=detector.n_states,
            interval=interval,
            spread_bps=cfg.get("costs", {}).get("spread_bps", 5.0),
            fee_rate=cfg.get("costs", {}).get("fee_rate", 0.0004),
            slippage_bps=cfg.get("costs", {}).get("slippage_bps", 2.0),
        )
        print_regime_report(regime_results, detector)
    except Exception as e:
        print(f"  Regime analysis skipped: {e}")

    # --------- サマリー ---------
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for fold_idx, r in fold_results.items():
        m = r["metrics"]
        print(f"  Fold {fold_idx}: Sharpe={m.sharpe:.3f}, MaxDD={m.max_drawdown:.3f}, "
              f"Calmar={m.calmar:.3f}, TotalRet={m.total_return:.4f}")
    print(f"  Mean Sharpe: {np.mean(all_sharpes):.3f}")
    print(f"  PBO: {pbo:.4f} | DSR: {dsr:.4f}")


if __name__ == "__main__":
    main()
