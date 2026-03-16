"""観測エンコーダモジュール.

MLP 2 層（256-256）で観測を離散カテゴリカル潜在表現に変換する。
DreamerV3 の straight-through gradient を使用。

References:
    NM512/dreamerv3-torch - OneHotDist, RSSM encoder
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd


# --- 補助関数 ---

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm transform (DreamerV3)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# --- Straight-Through OneHot 分布 ---

class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """Straight-through gradient 付き OneHot Categorical 分布.

    DreamerV3 の実装に基づく。
    - sample(): straight-through estimator
    - mode(): straight-through estimator
    - unimix_ratio で均一分布との混合によりエントロピー正則化

    References:
        NM512/dreamerv3-torch tools.py
    """

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        unimix_ratio: float = 0.01,
    ):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
        if logits is not None:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        super().__init__(logits=logits, probs=probs if logits is None else None, validate_args=False)

    def mode(self) -> torch.Tensor:
        """最頻値を straight-through で返す."""
        _mode = F.one_hot(
            torch.argmax(super().logits, dim=-1),
            super().logits.shape[-1],
        ).float()
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape: tuple = ()) -> torch.Tensor:
        """サンプルを straight-through で返す."""
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs.unsqueeze(0)
        return sample + probs - probs.detach()


# --- 観測エンコーダ ---

class ObsEncoder(nn.Module):
    """観測 → 離散カテゴリカル潜在 エンコーダ.

    MLP 2 層（256-256）で obs_dim → n_categoricals × n_classes の logits を出力。
    OneHotDist を通じて straight-through gradient で z を得る。

    Args:
        obs_dim: 入力観測次元
        n_categoricals: カテゴリカル分布の数（デフォルト 32）
        n_classes: 各カテゴリの離散値数（デフォルト 32）
        hidden_dim: 隠れ層次元（デフォルト 256）
        unimix_ratio: エントロピー正則化のための均一混合比率
    """

    def __init__(
        self,
        obs_dim: int,
        n_categoricals: int = 32,
        n_classes: int = 32,
        hidden_dim: int = 256,
        unimix_ratio: float = 0.01,
    ):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.z_dim = n_categoricals * n_classes  # 1024
        self.unimix_ratio = unimix_ratio

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_categoricals * n_classes),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, ..., obs_dim) または (B, T, obs_dim)

        Returns:
            z: (B, ..., n_categoricals * n_classes) straight-through サンプル
            logits: (B, ..., n_categoricals, n_classes) raw logits（KL 計算用）
        """
        logits_flat = self.net(obs)  # (..., n_cats * n_classes)
        shape = logits_flat.shape[:-1] + (self.n_categoricals, self.n_classes)
        logits = logits_flat.reshape(shape)

        dist = OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio)
        z_onehot = dist.sample()  # (..., n_cats, n_classes) straight-through

        # flatten
        z = z_onehot.reshape(logits_flat.shape)  # (..., n_cats * n_classes)
        return z, logits

    def get_dist(self, logits: torch.Tensor) -> torchd.Distribution:
        """logits から分布を返す（KL 計算に使用）."""
        return torchd.independent.Independent(
            OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio),
            reinterpreted_batch_ndims=1,
        )

    @property
    def latent_dim(self) -> int:
        return self.z_dim


class ObsDecoder(nn.Module):
    """潜在 z → 観測再構成 デコーダ.

    Args:
        z_dim: 潜在次元（n_categoricals * n_classes）
        h_dim: Transformer hidden 次元（z と結合する）
        obs_dim: 出力観測次元
        hidden_dim: 隠れ層次元
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = z_dim + h_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (..., z_dim)
            h: (..., h_dim)

        Returns:
            obs_recon: (..., obs_dim)
        """
        x = torch.cat([z, h], dim=-1)
        return self.net(x)
