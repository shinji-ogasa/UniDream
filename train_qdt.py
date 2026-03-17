"""QDT (Q-conditioned Decision Transformer) 学習スクリプト.

oracle_values（Hindsight Oracle DP の状態価値）を return-to-go ラベルとして
Decision Transformer を学習する。UniDream AC との比較対象。

Usage:
    python train_qdt.py --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01
"""
from __future__ import annotations
import argparse, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from unidream.data.download import fetch_binance_ohlcv
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.dataset import get_wfo_splits, WFODataset
from unidream.data.oracle import hindsight_oracle_dp, ACTIONS as _ACTIONS
from unidream.eval.backtest import Backtest
from unidream.eval.pbo import compute_pbo


_CACHE_STALE_DAYS = 7


def _cache_is_fresh(path, stale_days=_CACHE_STALE_DAYS):
    if not os.path.exists(path):
        return False
    import time
    return (time.time() - os.path.getmtime(path)) / 86400 < stale_days


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class QDTModel(nn.Module):
    """Q-conditioned Decision Transformer.

    入力: [RTG_t, obs_t] の時系列
    出力: action logits (t ごと)

    RTG (Return-to-Go) = oracle_values（DP による将来価値の合計）
    """

    def __init__(self, obs_dim: int, act_dim: int = 5, rtg_dim: int = 1,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()
        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.rtg_emb = nn.Linear(rtg_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, act_dim)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, obs: torch.Tensor, rtg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, T, obs_dim)
            rtg: (B, T, 1) return-to-go
        Returns:
            logits: (B, T, act_dim)
        """
        B, T, _ = obs.shape
        T = min(T, self.max_seq_len)
        obs = obs[:, :T]
        rtg = rtg[:, :T]

        x = self.obs_emb(obs) + self.rtg_emb(rtg)
        pos = torch.arange(T, device=obs.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=obs.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)


class QDTTrainer:
    def __init__(self, model: QDTModel, lr: float = 1e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, obs_seqs: np.ndarray, rtg_seqs: np.ndarray,
              action_seqs: np.ndarray, n_epochs: int = 5, batch_size: int = 64) -> None:
        """
        Args:
            obs_seqs: (N, T, obs_dim)
            rtg_seqs: (N, T) oracle values as RTG
            action_seqs: (N, T) oracle action indices
        """
        obs_t = torch.tensor(obs_seqs, dtype=torch.float32)
        rtg_t = torch.tensor(rtg_seqs, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        act_t = torch.tensor(action_seqs, dtype=torch.long)

        dataset = TensorDataset(obs_t, rtg_t, act_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            total_loss = 0.0; count = 0
            for obs_b, rtg_b, act_b in loader:
                obs_b = obs_b.to(self.device)
                rtg_b = rtg_b.to(self.device)
                act_b = act_b.to(self.device)
                logits = self.model(obs_b, rtg_b)  # (B, T, act_dim)
                B, T, A = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, A), act_b.reshape(B * T))
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item(); count += 1
            print(f"[QDT] Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/max(count,1):.4f}")

    @torch.no_grad()
    def predict(self, obs_seqs: np.ndarray, target_rtg: float) -> np.ndarray:
        """スライディングウィンドウで全テスト期間の action を予測する.

        Args:
            obs_seqs: (T, obs_dim) test observations
            target_rtg: desired return-to-go (75th percentile of training RTGs)
        Returns:
            positions: (T,) ∈ {-1, -0.5, 0, 0.5, 1}
        """
        T = len(obs_seqs)
        S = self.model.max_seq_len
        all_actions = np.zeros(T, dtype=np.int64)

        for start in range(0, T, S):
            end = min(start + S, T)
            chunk = obs_seqs[start:end]
            L = end - start
            obs_t = torch.tensor(chunk, dtype=torch.float32, device=self.device).unsqueeze(0)
            rtg_t = torch.full((1, L, 1), target_rtg, dtype=torch.float32, device=self.device)
            logits = self.model(obs_t, rtg_t)           # (1, L, act_dim)
            all_actions[start:end] = logits.squeeze(0).argmax(-1).cpu().numpy()

        return _ACTIONS[all_actions]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


def _make_sequences(features: np.ndarray, oracle_actions: np.ndarray,
                    oracle_values: np.ndarray, seq_len: int) -> tuple:
    """特徴量・oracle データからシーケンスバッチを生成する."""
    T = len(features)
    N = max(0, T - seq_len + 1)
    obs_seqs = np.zeros((N, seq_len, features.shape[1]), dtype=np.float32)
    rtg_seqs = np.zeros((N, seq_len), dtype=np.float32)
    act_seqs = np.zeros((N, seq_len), dtype=np.int64)
    for i in range(N):
        obs_seqs[i] = features[i:i+seq_len]
        rtg_seqs[i] = oracle_values[i:i+seq_len]
        act_seqs[i] = oracle_actions[i:i+seq_len]
    return obs_seqs, rtg_seqs, act_seqs


def main():
    parser = argparse.ArgumentParser(description="QDT Baseline Training")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints/qdt")
    parser.add_argument("--data_cache_dir", default="checkpoints/data_cache",
                        help="メインの train.py と同じキャッシュを共有する場合に指定")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)

    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    data_cfg = cfg.get("data", {})
    costs_cfg = cfg.get("costs", {})
    seq_len = data_cfg.get("seq_len", 64)

    print(f"QDT Training | {symbol} {interval} | {args.start} → {args.end}")
    print(f"Device: {args.device}")

    cache_dir = args.data_cache_dir
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}"
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")

    if _cache_is_fresh(features_cache) and _cache_is_fresh(returns_cache):
        print("[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
    else:
        print("[Data] Fetching OHLCV...")
        df = fetch_binance_ohlcv(symbol, interval, args.start, args.end)
        print(f"  Raw data: {len(df)} bars")
        features_df = compute_features(df, zscore_window_days=zscore_window, interval=interval)
        raw_returns = get_raw_returns(df)
        common_idx = features_df.index.intersection(raw_returns.index)
        features_df = features_df.loc[common_idx]
        raw_returns = raw_returns.loc[common_idx]
        os.makedirs(cache_dir, exist_ok=True)
        features_df.to_parquet(features_cache)
        raw_returns.to_frame().to_parquet(returns_cache)
        print(f"  Cached to {cache_dir}")

    obs_dim = features_df.shape[1]

    splits = get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )
    print(f"[Data] {len(splits)} WFO folds")

    fold_results = {}
    for split in splits:
        print(f"\n--- Fold {split.fold_idx} ---")
        wfo_ds = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
        ckpt_path = os.path.join(args.checkpoint_dir, f"qdt_fold_{split.fold_idx}.pt")

        oracle_actions, oracle_values, _ = hindsight_oracle_dp(
            wfo_ds.train_returns,
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            discount=cfg.get("oracle", {}).get("discount", 1.0),
        )
        target_rtg = float(np.percentile(oracle_values, 75))
        print(f"  Oracle: mean={oracle_values.mean():.3f}, target_rtg={target_rtg:.3f}")

        model = QDTModel(
            obs_dim=obs_dim,
            act_dim=cfg.get("actions", {}).get("n", 5),
            d_model=cfg.get("world_model", {}).get("d_model", 512) // 4,
            n_heads=4, n_layers=2, d_ff=256,
            max_seq_len=seq_len,
        )
        trainer = QDTTrainer(model, lr=cfg.get("qdt", {}).get("lr", 3e-4), device=args.device)

        if args.resume and os.path.exists(ckpt_path):
            print(f"  Loading checkpoint: {ckpt_path}")
            trainer.load(ckpt_path)
        else:
            obs_seqs, rtg_seqs, act_seqs = _make_sequences(
                wfo_ds.train_features, oracle_actions, oracle_values, seq_len
            )
            trainer.train(obs_seqs, rtg_seqs, act_seqs,
                         n_epochs=cfg.get("qdt", {}).get("n_epochs", cfg.get("bc", {}).get("n_epochs", 20)),
                         batch_size=cfg.get("qdt", {}).get("batch_size", cfg.get("bc", {}).get("batch_size", 64)))
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save(ckpt_path)

        positions = trainer.predict(wfo_ds.test_dataset().features.numpy(), target_rtg)
        test_returns = wfo_ds.test_returns
        T_min = min(len(test_returns), len(positions))

        bt = Backtest(
            test_returns[:T_min], positions[:T_min],
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            interval=interval,
        )
        metrics = bt.run()
        fold_results[split.fold_idx] = {"metrics": metrics}
        print(f"  Sharpe={metrics.sharpe:.3f} | MaxDD={metrics.max_drawdown:.3f} | Calmar={metrics.calmar:.3f}")

    print("\n" + "="*60)
    print("QDT Summary")
    print("="*60)
    sharpes = [r["metrics"].sharpe for r in fold_results.values()]
    for fold_idx, r in fold_results.items():
        m = r["metrics"]
        print(f"  Fold {fold_idx}: Sharpe={m.sharpe:.3f}, MaxDD={m.max_drawdown:.3f}")
    print(f"  Mean Sharpe: {np.mean(sharpes):.3f}")
    pnl_list = [r["metrics"].pnl_series for r in fold_results.values()]
    if len(pnl_list) > 1:
        pbo = compute_pbo(pnl_list)
        print(f"  PBO: {pbo:.4f}")


if __name__ == "__main__":
    main()
