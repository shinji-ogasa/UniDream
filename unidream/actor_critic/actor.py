"""Actor モジュール.

`trade / target inventory / no-trade band` を主出力に持つ inventory controller。
policy が学習するのは inventory path で、世界モデルにも executed inventory を渡す。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

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
        self.target_head = nn.Linear(hidden_dim, act_dim)
        self.band_head = nn.Linear(hidden_dim, 1)

        # 初期状態は「まず hold、必要なときだけ動く」に寄せる。
        nn.init.constant_(self.trade_head.bias, -0.5)
        nn.init.constant_(self.band_head.bias, 0.0)

    def _benchmark_position(self) -> float:
        return float(getattr(self, "benchmark_position", 0.0))

    def _absolute_bounds(self) -> tuple[float, float]:
        abs_min = float(getattr(self, "abs_min_position", -1.0))
        abs_max = float(getattr(self, "abs_max_position", 1.0))
        return abs_min, abs_max

    def _overlay_bounds(self) -> tuple[float, float]:
        bench = self._benchmark_position()
        abs_min, abs_max = self._absolute_bounds()
        return abs_min - bench, abs_max - bench

    def _position_to_overlay(self, position: torch.Tensor) -> torch.Tensor:
        return position - self._benchmark_position()

    def _overlay_to_position(self, overlay: torch.Tensor) -> torch.Tensor:
        abs_min, abs_max = self._absolute_bounds()
        return (overlay + self._benchmark_position()).clamp(min=abs_min, max=abs_max)

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
    ) -> tuple[torch.Tensor, Categorical, torch.Tensor, torch.Tensor]:
        """trade logit, target distribution, band width, current inventory."""
        hidden, inventory_t = self._prepare_inputs(z, h, inventory=inventory, regime=regime)
        trade_logits = self.trade_head(hidden).squeeze(-1)
        min_band = float(getattr(self, "min_band", 0.02))
        max_band = float(getattr(self, "max_band", 0.20))
        band_width = min_band + max_band * torch.sigmoid(self.band_head(hidden).squeeze(-1))
        if temperature != 1.0:
            target_logits = self.target_head(hidden) / temperature
        else:
            target_logits = self.target_head(hidden)
        if self.unimix_ratio > 0.0:
            probs = F.softmax(target_logits, dim=-1)
            probs = probs * (1.0 - self.unimix_ratio) + self.unimix_ratio / self.act_dim
            target_dist = Categorical(probs=probs)
        else:
            target_dist = Categorical(logits=target_logits)
        return trade_logits, target_dist, band_width, inventory_t.squeeze(-1)

    def _target_values_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = getattr(self, "target_values", None)
        if values is None:
            abs_min, abs_max = self._absolute_bounds()
            values = np.linspace(abs_min, abs_max, self.act_dim, dtype=np.float32)
        return torch.as_tensor(values, dtype=dtype, device=device)

    def target_indices(self, absolute_positions: torch.Tensor) -> torch.Tensor:
        target_values = self._target_values_tensor(absolute_positions.device, absolute_positions.dtype)
        return torch.argmin(torch.abs(absolute_positions.unsqueeze(-1) - target_values), dim=-1)

    def _bounded_step(self, gap: torch.Tensor) -> torch.Tensor:
        max_position_step = float(getattr(self, "max_position_step", 10.0))
        if max_position_step >= 10.0:
            return gap
        return gap.clamp(min=-max_position_step, max=max_position_step)

    def _trade_score(
        self,
        trade_signal: torch.Tensor,
        target_gap: torch.Tensor,
        band_width: torch.Tensor,
    ) -> torch.Tensor:
        """band を超えた gap が大きいほど発火しやすい score を作る."""
        gap_boost = float(getattr(self, "infer_gap_boost", 0.0))
        if gap_boost <= 0.0:
            return trade_signal

        overlay_low, overlay_high = self._overlay_bounds()
        overlay_span = max(float(overlay_high - overlay_low), 1e-6)
        gap_excess = (torch.abs(target_gap) - band_width).clamp(min=0.0)
        gap_scale = (gap_excess / overlay_span).clamp(max=1.0)
        return trade_signal * (1.0 + gap_boost * gap_scale)

    def execute_controller(
        self,
        trade_signal: torch.Tensor,
        target_inventory: torch.Tensor,
        band_width: torch.Tensor,
        current_inventory: torch.Tensor,
        trade_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Controller 出力を executed inventory に変換する."""
        target_gap = target_inventory - current_inventory
        trade_score = self._trade_score(trade_signal, target_gap, band_width)
        will_trade = (trade_score >= trade_threshold) & (torch.abs(target_gap) > band_width)
        bounded_step = self._bounded_step(target_gap)
        next_inventory = torch.where(will_trade, current_inventory + bounded_step, current_inventory)
        overlay_low, overlay_high = self._overlay_bounds()
        return next_inventory.clamp(min=overlay_low, max=overlay_high)

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """executed inventory・log_prob・entropy を返す."""
        trade_logits, target_dist, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_dist = Bernoulli(logits=trade_logits)

        trade_sample = trade_dist.sample()
        target_idx = target_dist.sample()
        target_values = self._target_values_tensor(current_inventory.device, current_inventory.dtype)
        target_inventory = target_values[target_idx] - self._benchmark_position()

        next_inventory = self.execute_controller(
            trade_signal=trade_sample,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
            trade_threshold=0.5,
        )
        trade_log_prob = trade_dist.log_prob(trade_sample)
        target_log_prob = target_dist.log_prob(target_idx) * trade_sample
        log_prob = trade_log_prob + target_log_prob
        entropy = trade_dist.entropy() + target_dist.entropy() * trade_dist.probs
        next_position = self._overlay_to_position(next_inventory)
        return next_position.unsqueeze(-1), log_prob, entropy

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
        trade_logits, target_dist, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_prob = torch.sigmoid(trade_logits)
        target_idx = target_dist.probs.argmax(dim=-1)
        target_values = self._target_values_tensor(current_inventory.device, current_inventory.dtype)
        target_inventory = target_values[target_idx] - self._benchmark_position()
        next_inventory = self.execute_controller(
            trade_signal=trade_prob,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
            trade_threshold=float(getattr(self, "infer_trade_threshold", 0.5)),
        )
        next_position = self._overlay_to_position(next_inventory)
        return next_position.unsqueeze(-1)

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
        prev_overlay = torch.zeros(1, 1, dtype=torch.float32, device=dev)
        for i in range(len(z_np)):
            reg_t = regime[i:i + 1] if regime is not None else None
            next_position = self.act_greedy(
                z[i:i + 1],
                h[i:i + 1],
                inventory=prev_overlay,
                regime=reg_t,
            )
            positions[i] = float(next_position.item())
            prev_overlay = self._position_to_overlay(next_position).reshape(1, 1)

        if was_training:
            self.train()
        return positions
