"""Actor モジュール.

離散 5 行動（ポジション比率）に加えて、`trade / hold` の補助 head を持つ。
入力: 世界モデルの潜在 z + Transformer hidden h + 現在 inventory
      （+ オプションの regime ベクトル）
出力:
  - target inventory 用 5 行動 logits（Categorical 分布）
  - trade / hold 用 logit
  - no-trade band 幅

DreamerV3 スタイル: entropy 正則化付き。
"""
from __future__ import annotations

import numpy as np
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
        dropout_p: float = 0.0,
        inventory_dim: int = 1,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.unimix_ratio = unimix_ratio
        self.regime_dim = regime_dim
        self.inventory_dim = inventory_dim

        in_dim = z_dim + h_dim + regime_dim + inventory_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.trunk = nn.Sequential(*layers[:-1])
        self.policy_head = layers[-1]
        self.trade_head = nn.Linear(hidden_dim, 1)
        self.band_head = nn.Linear(hidden_dim, 1)

    def _encode_state(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """方策ヘッド共有の状態表現を作る."""
        if inventory is None:
            inventory = torch.zeros(*z.shape[:-1], self.inventory_dim, dtype=z.dtype, device=z.device)
        elif inventory.ndim == z.ndim - 1:
            inventory = inventory.unsqueeze(-1)

        parts = [z, h, inventory]
        if self.regime_dim > 0:
            if regime is None:
                regime = torch.zeros(*z.shape[:-1], self.regime_dim, dtype=z.dtype, device=z.device)
            parts.append(regime)
        x = torch.cat(parts, dim=-1)
        return self.trunk(x)

    def policy_outputs(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
        temperature: float = 1.0,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """行動分布と trade logit と no-trade band 幅を返す."""
        hidden = self._encode_state(z, h, inventory=inventory, regime=regime)
        logits = self.policy_head(hidden)
        trade_logits = self.trade_head(hidden).squeeze(-1)
        band_width = F.softplus(self.band_head(hidden).squeeze(-1))

        if temperature != 1.0:
            logits = logits / temperature

        if self.unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - self.unimix_ratio) + self.unimix_ratio / self.act_dim
            return Categorical(probs=probs), trade_logits, band_width

        return Categorical(logits=logits), trade_logits, band_width

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
        temperature: float = 1.0,
    ) -> Categorical:
        """潜在表現から行動分布を返す."""
        dist, _, _ = self.policy_outputs(
            z, h, inventory=inventory, regime=regime, temperature=temperature
        )
        return dist

    def trade_logits(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """trade / hold の logit を返す."""
        _, trade_logits, _ = self.policy_outputs(z, h, inventory=inventory, regime=regime)
        return trade_logits

    def band_width(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """learned no-trade band 幅を返す."""
        _, _, band_width = self.policy_outputs(z, h, inventory=inventory, regime=regime)
        return band_width

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動サンプル・log_prob・entropy を返す.

        Returns:
            action: (...,) 行動インデックス
            log_prob: (...,) 対数確率
            entropy: (...,) エントロピー
        """
        dist = self.forward(z, h, inventory=inventory, regime=regime)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def act_greedy(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: "torch.Tensor | None" = None,
        regime: "torch.Tensor | None" = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """最頻値行動を返す（推論時 greedy）."""
        dist = self.forward(z, h, inventory=inventory, regime=regime, temperature=temperature)
        return dist.probs.argmax(dim=-1)

    @torch.no_grad()
    def predict_positions(
        self,
        z_np: "np.ndarray",
        h_np: "np.ndarray",
        regime_np: "np.ndarray | None" = None,
        device: str = "cpu",
        temperature: "float | None" = None,
    ) -> "np.ndarray":
        """numpy 配列から ポジション比率列を返す.

        Args:
            z_np: (T, z_dim)
            h_np: (T, h_dim)
            regime_np: (T, regime_dim) レジーム確率ベクトル（省略可）
            temperature: softmax 温度（None のとき self.infer_temperature を使用）

        Returns:
            positions: (T,) ∈ {-1, -0.5, 0, 0.5, 1}
        """
        t = temperature if temperature is not None else getattr(self, "infer_temperature", 1.0)
        switch_margin = float(getattr(self, "switch_margin", 0.0))
        max_position_step = float(getattr(self, "max_position_step", 10.0))
        was_training = self.training
        self.eval()
        dev = torch.device(device)
        z = torch.tensor(z_np, dtype=torch.float32, device=dev)
        h = torch.tensor(h_np, dtype=torch.float32, device=dev)
        regime = None
        if regime_np is not None and self.regime_dim > 0:
            regime = torch.tensor(regime_np, dtype=torch.float32, device=dev)
        action_indices = np.zeros(len(z_np), dtype=np.int64)
        prev_idx = int(np.where(ACTIONS == 0.0)[0][0])
        for i in range(len(z_np)):
            inv_t = torch.tensor([[ACTIONS[prev_idx]]], dtype=torch.float32, device=dev)
            z_t = z[i:i + 1]
            h_t = h[i:i + 1]
            reg_t = regime[i:i + 1] if regime is not None else None
            dist, trade_logits, band_width = self.policy_outputs(
                z_t, h_t, inventory=inv_t, regime=reg_t, temperature=t
            )
            p = dist.probs.squeeze(0).detach().cpu().numpy()
            best_idx = int(np.argmax(p))
            chosen_idx = best_idx
            trade_prob = float(torch.sigmoid(trade_logits).item())
            band = float(band_width.item())
            trade_threshold = float(getattr(self, "infer_trade_threshold", 0.5))
            target_gap = float(abs(ACTIONS[best_idx] - ACTIONS[prev_idx]))

            if trade_prob < trade_threshold or target_gap <= band:
                chosen_idx = prev_idx

            if chosen_idx != prev_idx and switch_margin > 0.0:
                if p[best_idx] - p[prev_idx] < switch_margin:
                    chosen_idx = prev_idx

            if chosen_idx != prev_idx and max_position_step < 10.0:
                prev_pos = ACTIONS[prev_idx]
                allowed = np.where(np.abs(ACTIONS - prev_pos) <= max_position_step + 1e-8)[0]
                if chosen_idx not in allowed:
                    chosen_idx = int(allowed[np.argmax(p[allowed])])

            action_indices[i] = chosen_idx
            prev_idx = chosen_idx
        if was_training:
            self.train()
        return ACTIONS[action_indices]
