"""BC (Behavior Cloning) 事前学習モジュール.

Hindsight Oracle の最適行動列を教師ラベルとして、
Actor を KL-divergence 損失で数エポック学習する。

SIRL (State-dependent Importance-weighted RL) スタイルの
状態依存重み w(s) で各サンプルの損失を重み付けする。
サンプルに等しい重みを与えたい場合は sirl_hidden=0 で無効化可能。

References:
    SIRL: https://arxiv.org/abs/2209.02276
    DAgger は実装コストが高いため初版ではスキップ。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from unidream.actor_critic.actor import Actor


class SIRLWeightNet(nn.Module):
    """状態依存 BC 重み w(s) を出力するネットワーク（SIRL）.

    w(s) = softmax(MLP(s)) * N で確率的なリサンプリング重みを学習する。
    N は batch size（重みの合計が N になるよう正規化）。

    Args:
        in_dim: 入力状態次元（z_dim + h_dim）
        hidden_dim: 隠れ層次元
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, in_dim)

        Returns:
            weights: (B,) 正規化済み重み（合計 B）
        """
        logits = self.net(state).squeeze(-1)  # (B,)
        weights = F.softmax(logits, dim=0) * logits.shape[0]
        return weights


class BCPretrainer:
    """Behavior Cloning 事前学習.

    Args:
        actor: Actor モジュール
        z_dim: 潜在次元
        h_dim: Transformer hidden 次元
        lr: 学習率
        batch_size: ミニバッチサイズ
        n_epochs: 学習エポック数
        sirl_hidden: SIRL 重みネット隠れ層次元（0 で無効）
        device: 計算デバイス
    """

    def __init__(
        self,
        actor: Actor,
        z_dim: int,
        h_dim: int,
        lr: float = 3e-4,
        batch_size: int = 256,
        n_epochs: int = 5,
        sirl_hidden: int = 128,
        device: str = "cpu",
    ):
        self.actor = actor
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # SIRL 重みネット
        self.use_sirl = sirl_hidden > 0
        if self.use_sirl:
            self.weight_net = SIRLWeightNet(z_dim + h_dim, sirl_hidden).to(self.device)
            params = list(actor.parameters()) + list(self.weight_net.parameters())
        else:
            self.weight_net = None
            params = list(actor.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _bc_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        oracle_actions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """KL-divergence ベースの BC 損失.

        KL(oracle_onehot || actor_dist) = -log P_actor(oracle_action)

        Args:
            z: (B, z_dim)
            h: (B, h_dim)
            oracle_actions: (B,) oracle 行動インデックス
            weights: (B,) 状態依存重み
            regime: (B, regime_dim) レジーム確率ベクトル（省略可）

        Returns:
            loss: スカラー
        """
        dist = self.actor(z, h, regime=regime)
        log_prob = dist.log_prob(oracle_actions)  # (B,)

        if weights is not None:
            loss = -(weights * log_prob).mean()
        else:
            loss = -log_prob.mean()

        return loss

    def train(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_actions: np.ndarray,
        verbose: bool = True,
        regime_probs: "np.ndarray | None" = None,
    ) -> list[dict]:
        """BC 事前学習を実行する.

        Args:
            z: (T, z_dim) エンコードされた潜在
            h: (T, h_dim) Transformer hidden
            oracle_actions: (T,) oracle 行動インデックス
            regime_probs: (T, regime_dim) レジーム確率ベクトル（省略可）

        Returns:
            各エポックのロスログ
        """
        T = min(len(z), len(h), len(oracle_actions))
        z_t = torch.tensor(z[:T], dtype=torch.float32)
        h_t = torch.tensor(h[:T], dtype=torch.float32)
        a_t = torch.tensor(oracle_actions[:T], dtype=torch.long)

        use_regime = regime_probs is not None
        if use_regime:
            reg_t = torch.tensor(regime_probs[:T], dtype=torch.float32)
            dataset = TensorDataset(z_t, h_t, a_t, reg_t)
        else:
            dataset = TensorDataset(z_t, h_t, a_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logs = []
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            count = 0

            for batch in loader:
                if use_regime:
                    z_b, h_b, a_b, reg_b = batch
                    reg_b = reg_b.to(self.device)
                else:
                    z_b, h_b, a_b = batch
                    reg_b = None
                z_b = z_b.to(self.device)
                h_b = h_b.to(self.device)
                a_b = a_b.to(self.device)

                # SIRL 重み
                if self.use_sirl:
                    state = torch.cat([z_b, h_b], dim=-1)
                    weights = self.weight_net(state)
                else:
                    weights = None

                loss = self._bc_loss(z_b, h_b, a_b, weights, regime=reg_b)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) +
                    (list(self.weight_net.parameters()) if self.use_sirl else []),
                    max_norm=10.0,
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1

            avg_loss = epoch_loss / max(count, 1)
            logs.append({"epoch": epoch, "bc_loss": avg_loss})

            if verbose:
                print(f"[BC] Epoch {epoch + 1}/{self.n_epochs} | Loss: {avg_loss:.4f}")

        return logs

    def save(self, path: str) -> None:
        payload = {"actor": self.actor.state_dict()}
        if self.use_sirl:
            payload["weight_net"] = self.weight_net.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        if self.use_sirl and "weight_net" in ckpt:
            self.weight_net.load_state_dict(ckpt["weight_net"])
