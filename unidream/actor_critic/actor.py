"""Actor モジュール.

離散 5 行動（ポジション比率）を出力する方策ネットワーク。
入力: 世界モデルの潜在 z + Transformer hidden h （+ オプションの regime ベクトル）
出力: 5 行動の logits（Categorical 分布）

DreamerV3 スタイル: entropy 正則化付き。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from unidream.data.oracle import ACTIONS, N_ACTIONS


class Actor(nn.Module):
    """離散行動 Actor.

    Args:
        z_dim: 潜在次元（n_categoricals × n_classes = 1024）
        h_dim: Transformer hidden 次元
        act_dim: 行動数（デフォルト 5）
        hidden_dim: MLP 隠れ層次元
        n_layers: MLP 層数
        unimix_ratio: 行動分布の均一混合比率（exploration 用）
        regime_dim: レジーム確率ベクトルの次元（0 で無効）
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        act_dim: int = N_ACTIONS,
        hidden_dim: int = 256,
        n_layers: int = 2,
        unimix_ratio: float = 0.01,
        regime_dim: int = 0,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.unimix_ratio = unimix_ratio
        self.regime_dim = regime_dim

        in_dim = z_dim + h_dim + regime_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        regime: "torch.Tensor | None" = None,
    ) -> Categorical:
        """潜在表現から行動分布を返す.

        Args:
            z: (..., z_dim)
            h: (..., h_dim)
            regime: (..., regime_dim) レジーム確率ベクトル（省略可）

        Returns:
            Categorical 分布（unimix 済み）
        """
        if self.regime_dim > 0:
            if regime is None:
                regime = torch.zeros(*z.shape[:-1], self.regime_dim, dtype=z.dtype, device=z.device)
            x = torch.cat([z, h, regime], dim=-1)
        else:
            x = torch.cat([z, h], dim=-1)
        logits = self.net(x)

        if self.unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - self.unimix_ratio) + self.unimix_ratio / self.act_dim
            return Categorical(probs=probs)

        return Categorical(logits=logits)

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        regime: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動サンプル・log_prob・entropy を返す.

        Returns:
            action: (...,) 行動インデックス
            log_prob: (...,) 対数確率
            entropy: (...,) エントロピー
        """
        dist = self.forward(z, h, regime=regime)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def act_greedy(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        regime: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """最頻値行動を返す（推論時 greedy）."""
        dist = self.forward(z, h, regime=regime)
        return dist.probs.argmax(dim=-1)

    @torch.no_grad()
    def predict_positions(
        self,
        z_np: "np.ndarray",
        h_np: "np.ndarray",
        regime_np: "np.ndarray | None" = None,
        device: str = "cpu",
    ) -> "np.ndarray":
        """numpy 配列から ポジション比率列を返す.

        Args:
            z_np: (T, z_dim)
            h_np: (T, h_dim)
            regime_np: (T, regime_dim) レジーム確率ベクトル（省略可）

        Returns:
            positions: (T,) ∈ {-1, -0.5, 0, 0.5, 1}
        """
        import numpy as np
        dev = torch.device(device)
        z = torch.tensor(z_np, dtype=torch.float32, device=dev)
        h = torch.tensor(h_np, dtype=torch.float32, device=dev)
        regime = None
        if regime_np is not None and self.regime_dim > 0:
            regime = torch.tensor(regime_np, dtype=torch.float32, device=dev)
        action_indices = self.act_greedy(z, h, regime=regime).cpu().numpy()
        return ACTIONS[action_indices]
