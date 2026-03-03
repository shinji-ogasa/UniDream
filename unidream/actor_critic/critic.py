"""Critic（Value function）モジュール.

DreamerV3 スタイル:
- 出力: 255 bins の twohot 分布（symlog + twohot encoding）
- slow target: EMA で更新し、学習を安定化

入力: 世界モデルの潜在 z + Transformer hidden h
出力: value の twohot logits（255 bins）

References:
    NM512/dreamerv3-torch - Critic, slow target
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unidream.world_model.transformer import twohot_encode, twohot_decode, symlog


class Critic(nn.Module):
    """Twohot value function Critic.

    Args:
        z_dim: 潜在次元
        h_dim: Transformer hidden 次元
        hidden_dim: MLP 隠れ層次元
        n_layers: MLP 層数
        n_bins: twohot ビン数（デフォルト 255）
        ema_decay: slow target の EMA 減衰率
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_bins: int = 255,
        ema_decay: float = 0.98,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.ema_decay = ema_decay

        in_dim = z_dim + h_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, n_bins))
        self.net = nn.Sequential(*layers)

        # slow target ネットワーク（EMA）
        self.slow_net = copy.deepcopy(self.net)
        for p in self.slow_net.parameters():
            p.requires_grad_(False)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """value の twohot logits を返す.

        Args:
            z: (..., z_dim)
            h: (..., h_dim)

        Returns:
            logits: (..., n_bins)
        """
        x = torch.cat([z, h], dim=-1)
        return self.net(x)

    def slow_forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Slow target の logits を返す."""
        x = torch.cat([z, h], dim=-1)
        return self.slow_net(x)

    def value(self, z: torch.Tensor, h: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """期待値 value（symexp 変換済みスカラー）を返す."""
        logits = self.forward(z, h)
        return twohot_decode(logits, bins)

    def slow_value(self, z: torch.Tensor, h: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Slow target の期待値 value を返す."""
        logits = self.slow_forward(z, h)
        return twohot_decode(logits, bins)

    @torch.no_grad()
    def update_slow_target(self) -> None:
        """Slow target を EMA で更新する."""
        for fast, slow in zip(self.net.parameters(), self.slow_net.parameters()):
            slow.data.copy_(self.ema_decay * slow.data + (1.0 - self.ema_decay) * fast.data)

    def loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        targets: torch.Tensor,
        bins: torch.Tensor,
    ) -> torch.Tensor:
        """Twohot cross-entropy loss を計算する.

        Args:
            z: (..., z_dim)
            h: (..., h_dim)
            targets: (...,) ターゲット value（生スケール）
            bins: (n_bins,) twohot ビン境界

        Returns:
            loss: スカラー
        """
        logits = self.forward(z, h)
        targets_symlog = symlog(targets)
        target_twohot = twohot_encode(targets_symlog, bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_twohot * log_probs).sum(-1).mean()


class RewardEMANorm:
    """報酬の EMA 正規化（DreamerV3 スタイル）.

    5th / 95th percentile EMA でスケーリングする。
    critic の target を安定化するために使用。
    """

    def __init__(self, decay: float = 0.99, low_pct: float = 5.0, high_pct: float = 95.0):
        self.decay = decay
        self.low_pct = low_pct
        self.high_pct = high_pct
        self._low = None
        self._high = None

    def update(self, values: torch.Tensor) -> None:
        """EMA を更新する."""
        v = values.detach().float()
        lo = torch.quantile(v, self.low_pct / 100.0).item()
        hi = torch.quantile(v, self.high_pct / 100.0).item()
        if self._low is None:
            self._low, self._high = lo, hi
        else:
            self._low = self.decay * self._low + (1 - self.decay) * lo
            self._high = self.decay * self._high + (1 - self.decay) * hi

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """値を正規化する."""
        if self._low is None:
            return values
        scale = max(1.0, self._high - self._low)
        return values / scale

    @property
    def scale(self) -> float:
        if self._low is None:
            return 1.0
        return max(1.0, self._high - self._low)
