"""Actor モジュール.

`trade / target inventory / no-trade band` を主出力に持つ inventory controller。
policy が学習するのは inventory path で、世界モデルにも executed inventory を渡す。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from unidream.data.oracle import N_ACTIONS


class Actor(nn.Module):
    """Inventory controller actor."""

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        act_dim: int = N_ACTIONS,
        hidden_dim: int = 256,
        n_layers: int = 2,
        unimix_ratio: float = 0.0,
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
        self.trunk = nn.Sequential(*layers)
        self.trade_head = nn.Linear(hidden_dim, 1)
        self.target_mean_head = nn.Linear(hidden_dim, 1)
        self.target_std_head = nn.Linear(hidden_dim, 1)
        self.band_head = nn.Linear(hidden_dim, 1)

        # 初期状態は「まず hold、必要なときだけ動く」に寄せる。
        nn.init.constant_(self.trade_head.bias, -0.5)
        nn.init.constant_(self.band_head.bias, 0.0)
        nn.init.constant_(self.target_std_head.bias, -2.0)

    def _prepare_inputs(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return self.trunk(x), inventory

    def controller_outputs(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """trade logit, target mean/std, band width, current inventory."""
        hidden, inventory_t = self._prepare_inputs(z, h, inventory=inventory, regime=regime)
        trade_logits = self.trade_head(hidden).squeeze(-1)
        min_band = float(getattr(self, "min_band", 0.02))
        max_band = float(getattr(self, "max_band", 0.20))
        band_width = min_band + max_band * torch.sigmoid(self.band_head(hidden).squeeze(-1))
        target_mean = torch.tanh(self.target_mean_head(hidden).squeeze(-1))
        min_std = float(getattr(self, "min_target_std", 0.05))
        max_std = float(getattr(self, "max_target_std", 0.35))
        target_std = min_std + max_std * torch.sigmoid(self.target_std_head(hidden).squeeze(-1))
        if temperature != 1.0:
            target_std = target_std * temperature
        return trade_logits, target_mean, target_std, band_width, inventory_t.squeeze(-1)

    def _target_dist(
        self,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> TransformedDistribution:
        base = Normal(target_mean, target_std.clamp_min(1e-4))
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    def _step_limited_target(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        max_position_step = float(getattr(self, "max_position_step", 10.0))
        if max_position_step >= 10.0:
            return target
        delta = (target - current).clamp(min=-max_position_step, max=max_position_step)
        return current + delta

    def execute_controller(
        self,
        trade_signal: torch.Tensor,
        target_inventory: torch.Tensor,
        band_width: torch.Tensor,
        current_inventory: torch.Tensor,
        trade_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Controller 出力を executed inventory に変換する."""
        target_gap = torch.abs(target_inventory - current_inventory)
        will_trade = (trade_signal >= trade_threshold) & (target_gap > band_width)
        bounded_target = self._step_limited_target(current_inventory, target_inventory)
        next_inventory = torch.where(will_trade, bounded_target, current_inventory)
        return next_inventory.clamp(min=-1.0, max=1.0)

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """executed inventory・log_prob・entropy を返す."""
        trade_logits, target_mean, target_std, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_dist = Bernoulli(logits=trade_logits)
        target_dist = self._target_dist(target_mean, target_std)

        trade_sample = trade_dist.sample()
        target_inventory = target_dist.rsample()

        next_inventory = self.execute_controller(
            trade_signal=trade_sample,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
            trade_threshold=0.5,
        )
        trade_log_prob = trade_dist.log_prob(trade_sample)
        target_log_prob = target_dist.log_prob(target_inventory) * trade_sample
        log_prob = trade_log_prob + target_log_prob
        entropy = trade_dist.entropy() + target_dist.base_dist.entropy() * trade_dist.probs
        return next_inventory.unsqueeze(-1), log_prob, entropy

    @torch.no_grad()
    def act_greedy(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """greedy execution に対応する executed inventory を返す."""
        del temperature
        trade_logits, target_mean, target_std, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        _ = target_std
        trade_prob = torch.sigmoid(trade_logits)
        target_inventory = target_mean
        next_inventory = self.execute_controller(
            trade_signal=trade_prob,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
            trade_threshold=float(getattr(self, "infer_trade_threshold", 0.5)),
        )
        return next_inventory.unsqueeze(-1)

    @torch.no_grad()
    def predict_positions(
        self,
        z_np: np.ndarray,
        h_np: np.ndarray,
        regime_np: np.ndarray | None = None,
        device: str = "cpu",
        temperature: float | None = None,
    ) -> np.ndarray:
        """numpy 配列からポジション比率列を返す."""
        del temperature
        was_training = self.training
        self.eval()
        dev = torch.device(device)
        z = torch.tensor(z_np, dtype=torch.float32, device=dev)
        h = torch.tensor(h_np, dtype=torch.float32, device=dev)
        regime = None
        if regime_np is not None and self.regime_dim > 0:
            regime = torch.tensor(regime_np, dtype=torch.float32, device=dev)

        positions = np.zeros(len(z_np), dtype=np.float32)
        prev_inventory = torch.zeros(1, 1, dtype=torch.float32, device=dev)
        for i in range(len(z_np)):
            reg_t = regime[i:i + 1] if regime is not None else None
            next_inventory = self.act_greedy(
                z[i:i + 1],
                h[i:i + 1],
                inventory=prev_inventory,
                regime=reg_t,
            )
            positions[i] = float(next_inventory.item())
            prev_inventory = next_inventory.reshape(1, 1)

        if was_training:
            self.train()
        return positions
