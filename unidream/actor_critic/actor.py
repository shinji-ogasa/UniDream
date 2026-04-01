"""Actor モジュール.

`trade / target inventory / no-trade band` を主出力に持つ inventory controller。
世界モデルとは execution rule を介して接続し、最終的には離散 action index へ写像する。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

from unidream.data.oracle import ACTIONS, N_ACTIONS

_ACTIONS_T = torch.tensor(ACTIONS, dtype=torch.float32)


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
        self.target_head = nn.Linear(hidden_dim, 1)
        self.band_head = nn.Linear(hidden_dim, 1)
        self.target_log_std = nn.Parameter(torch.tensor(np.log(0.15), dtype=torch.float32))

        # Conservative initialization: narrow band but not zero.
        nn.init.constant_(self.band_head.bias, -2.0)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """trade logit, target inventory mean, band width, current inventory."""
        hidden, inventory_t = self._prepare_inputs(z, h, inventory=inventory, regime=regime)
        trade_logits = self.trade_head(hidden).squeeze(-1)
        target_mean = torch.tanh(self.target_head(hidden).squeeze(-1))
        min_band = float(getattr(self, "min_band", 0.02))
        band_width = min_band + F.softplus(self.band_head(hidden).squeeze(-1))
        return trade_logits, target_mean, band_width, inventory_t.squeeze(-1)

    def target_std(self) -> torch.Tensor:
        """Shared exploration std for target inventory sampling."""
        min_std = float(getattr(self, "min_target_std", 0.05))
        max_std = float(getattr(self, "max_target_std", 0.35))
        std = torch.exp(self.target_log_std)
        return std.clamp(min=min_std, max=max_std)

    def _nearest_action_idx(self, inventory: torch.Tensor) -> torch.Tensor:
        action_values = _ACTIONS_T.to(device=inventory.device, dtype=inventory.dtype)
        return torch.argmin(torch.abs(inventory.unsqueeze(-1) - action_values), dim=-1)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Controller 出力を inventory と discrete action に変換する."""
        target_gap = torch.abs(target_inventory - current_inventory)
        will_trade = (trade_signal >= trade_threshold) & (target_gap > band_width)
        bounded_target = self._step_limited_target(current_inventory, target_inventory)
        next_inventory = torch.where(will_trade, bounded_target, current_inventory).clamp(min=-1.0, max=1.0)
        action_idx = self._nearest_action_idx(next_inventory)
        snapped_inventory = _ACTIONS_T.to(device=next_inventory.device, dtype=next_inventory.dtype)[action_idx]
        return snapped_inventory, action_idx

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動サンプル・log_prob・entropy を返す."""
        trade_logits, target_mean, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_dist = Bernoulli(logits=trade_logits)
        target_dist = Normal(target_mean, self.target_std().to(device=target_mean.device, dtype=target_mean.dtype))

        trade_sample = trade_dist.sample()
        raw_target = target_dist.rsample()
        target_sample = raw_target.clamp(min=-1.0, max=1.0)

        next_inventory, action_idx = self.execute_controller(
            trade_signal=trade_sample,
            target_inventory=target_sample,
            band_width=band_width,
            current_inventory=current_inventory,
            trade_threshold=0.5,
        )
        trade_log_prob = trade_dist.log_prob(trade_sample)
        target_log_prob = target_dist.log_prob(target_sample) * trade_sample
        log_prob = trade_log_prob + target_log_prob
        entropy = trade_dist.entropy() + target_dist.entropy() * trade_dist.probs
        _ = next_inventory
        return action_idx, log_prob, entropy

    @torch.no_grad()
    def act_greedy(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """greedy execution に対応する discrete action index を返す."""
        del temperature
        trade_logits, target_mean, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_prob = torch.sigmoid(trade_logits)
        _, action_idx = self.execute_controller(
            trade_signal=trade_prob,
            target_inventory=target_mean,
            band_width=band_width,
            current_inventory=current_inventory,
            trade_threshold=float(getattr(self, "infer_trade_threshold", 0.5)),
        )
        return action_idx

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

        action_indices = np.zeros(len(z_np), dtype=np.int64)
        prev_inventory = torch.zeros(1, 1, dtype=torch.float32, device=dev)
        for i in range(len(z_np)):
            reg_t = regime[i:i + 1] if regime is not None else None
            action_idx = self.act_greedy(
                z[i:i + 1],
                h[i:i + 1],
                inventory=prev_inventory,
                regime=reg_t,
            )
            action_indices[i] = int(action_idx.item())
            prev_inventory = _ACTIONS_T.to(device=dev).to(dtype=torch.float32)[action_idx].reshape(1, 1)

        if was_training:
            self.train()
        return ACTIONS[action_indices]
