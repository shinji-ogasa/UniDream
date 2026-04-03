"""Actor モジュール.

`trade / target inventory / no-trade band` を主出力に持つ inventory controller。
policy が学習するのは inventory path で、世界モデルにも executed inventory を渡す。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

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

    def _state_hold_scale(self) -> float:
        return float(getattr(self, "hold_state_scale", 64.0))

    def _trade_state_eps(self) -> float:
        return float(getattr(self, "trade_state_eps", 1e-6))

    def _ensure_controller_state(
        self,
        inventory: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if inventory is None:
            return torch.zeros(*ref.shape[:-1], self.inventory_dim, dtype=ref.dtype, device=ref.device)
        if inventory.ndim == ref.ndim - 1:
            inventory = inventory.unsqueeze(-1)
        if inventory.shape[-1] == self.inventory_dim:
            return inventory
        if inventory.shape[-1] > self.inventory_dim:
            return inventory[..., :self.inventory_dim]
        pad_shape = list(inventory.shape)
        pad_shape[-1] = self.inventory_dim - inventory.shape[-1]
        pad = torch.zeros(*pad_shape, dtype=inventory.dtype, device=inventory.device)
        return torch.cat([inventory, pad], dim=-1)

    def _current_inventory_from_state(self, controller_state: torch.Tensor) -> torch.Tensor:
        if controller_state.ndim == 0:
            return controller_state
        if controller_state.shape[-1] == 0:
            return torch.zeros(*controller_state.shape[:-1], dtype=controller_state.dtype, device=controller_state.device)
        return controller_state[..., 0]

    def make_controller_state(
        self,
        current_inventory: torch.Tensor,
        prev_delta: torch.Tensor | None = None,
        hold_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces = [current_inventory]
        if self.inventory_dim >= 2:
            if prev_delta is None:
                prev_delta = torch.zeros_like(current_inventory)
            pieces.append(prev_delta)
        if self.inventory_dim >= 3:
            if hold_feature is None:
                hold_feature = torch.zeros_like(current_inventory)
            pieces.append(hold_feature)
        if self.inventory_dim > len(pieces):
            for _ in range(self.inventory_dim - len(pieces)):
                pieces.append(torch.zeros_like(current_inventory))
        return torch.stack(pieces, dim=-1)

    def update_controller_state(
        self,
        controller_state: torch.Tensor | None,
        next_position: torch.Tensor,
    ) -> torch.Tensor:
        next_overlay = self._position_to_overlay(next_position.squeeze(-1) if next_position.ndim > 1 else next_position)
        state = self._ensure_controller_state(controller_state, next_overlay.unsqueeze(-1))
        current_inventory = self._current_inventory_from_state(state)
        next_delta = next_overlay - current_inventory
        traded = next_delta.abs() > self._trade_state_eps()

        hold_feature = None
        if self.inventory_dim >= 3:
            prev_hold = state[..., 2]
            hold_feature = torch.where(
                traded,
                torch.zeros_like(prev_hold),
                (prev_hold + 1.0 / self._state_hold_scale()).clamp(max=1.0),
            )
        return self.make_controller_state(next_overlay, next_delta, hold_feature)

    def controller_states_from_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        T = len(positions)
        states = np.zeros((T, self.inventory_dim), dtype=np.float32)
        if T == 0:
            return states
        bench = self._benchmark_position()
        hold = 0.0
        hold_step = 1.0 / self._state_hold_scale()
        for t in range(1, T):
            prev_overlay = float(positions[t - 1] - bench)
            states[t, 0] = prev_overlay
            if self.inventory_dim >= 2:
                prev_prev_overlay = float(positions[t - 2] - bench) if t > 1 else 0.0
                prev_delta = prev_overlay - prev_prev_overlay
                states[t, 1] = prev_delta
                traded = abs(prev_delta) > self._trade_state_eps()
            else:
                traded = abs(prev_overlay) > self._trade_state_eps()
            if self.inventory_dim >= 3:
                hold = 0.0 if traded else min(1.0, hold + hold_step)
                states[t, 2] = hold
        return states

    def controller_state_from_history(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        B, T = positions.shape
        bench = self._benchmark_position()
        current_overlay = positions[:, -1] - bench
        prev_delta = torch.zeros_like(current_overlay)
        if T >= 2 and self.inventory_dim >= 2:
            prev_overlay = positions[:, -2] - bench
            prev_delta = current_overlay - prev_overlay
        hold_feature = torch.zeros_like(current_overlay)
        if self.inventory_dim >= 3:
            eps = self._trade_state_eps()
            hold_step = 1.0 / self._state_hold_scale()
            for b in range(B):
                hold = 0.0
                for t in range(max(1, T - 256), T):
                    prev_overlay = float(positions[b, t - 1] - bench)
                    prev_prev_overlay = float(positions[b, t - 2] - bench) if t > 1 else 0.0
                    traded = abs(prev_overlay - prev_prev_overlay) > eps
                    hold = 0.0 if traded else min(1.0, hold + hold_step)
                hold_feature[b] = hold
        return self.make_controller_state(current_overlay, prev_delta, hold_feature)

    def _prepare_inputs(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inventory = self._ensure_controller_state(inventory, z)

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
        """adjust logit, target mean/std, band width, current inventory."""
        hidden, inventory_t = self._prepare_inputs(z, h, inventory=inventory, regime=regime)
        trade_logits = self.trade_head(hidden).squeeze(-1)
        min_band = float(getattr(self, "min_band", 0.02))
        max_band = float(getattr(self, "max_band", 0.20))
        band_width = min_band + max_band * torch.sigmoid(self.band_head(hidden).squeeze(-1))
        overlay_low, overlay_high = self._overlay_bounds()
        overlay_low_t = torch.as_tensor(overlay_low, dtype=hidden.dtype, device=hidden.device)
        overlay_high_t = torch.as_tensor(overlay_high, dtype=hidden.dtype, device=hidden.device)
        overlay_center = 0.5 * (overlay_low_t + overlay_high_t)
        overlay_half_range = 0.5 * (overlay_high_t - overlay_low_t)
        target_mean = overlay_center + overlay_half_range * torch.tanh(
            self.target_mean_head(hidden).squeeze(-1) / max(temperature, 1e-6)
        )
        min_target_std = float(getattr(self, "min_target_std", 0.05))
        max_target_std = float(getattr(self, "max_target_std", 0.25))
        target_std = min_target_std + (max_target_std - min_target_std) * torch.sigmoid(
            self.target_std_head(hidden).squeeze(-1)
        )
        return trade_logits, target_mean, target_std, band_width, self._current_inventory_from_state(inventory_t)

    def _target_values_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = getattr(self, "target_values", None)
        if values is None:
            abs_min, abs_max = self._absolute_bounds()
            values = np.linspace(abs_min, abs_max, self.act_dim, dtype=np.float32)
        return torch.as_tensor(values, dtype=dtype, device=device)

    def target_indices(self, absolute_positions: torch.Tensor) -> torch.Tensor:
        target_values = self._target_values_tensor(absolute_positions.device, absolute_positions.dtype)
        return torch.argmin(torch.abs(absolute_positions.unsqueeze(-1) - target_values), dim=-1)

    def target_distribution(
        self,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> Normal:
        return Normal(loc=target_mean, scale=target_std.clamp_min(1e-4))

    def _bounded_step(self, gap: torch.Tensor) -> torch.Tensor:
        max_position_step = float(getattr(self, "max_position_step", 10.0))
        if max_position_step >= 10.0:
            return gap
        return gap.clamp(min=-max_position_step, max=max_position_step)

    def _quantize_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy rollout の device 差分を減らすために軽く量子化する."""
        step = float(getattr(self, "infer_quantize_step", 0.0))
        if step <= 0.0:
            return x
        return torch.round(x / step) * step

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

        gap_excess = (torch.abs(target_gap) - band_width).clamp(min=0.0)
        min_band = float(getattr(self, "min_band", 0.02))
        band_scale = band_width.clamp(min=min_band)
        # 固定 threshold 一本では fold ごとに no-trade / over-trade が切り替わりやすい。
        # gap が learned band をどれだけ上回っているかで追加点を与えて、
        # 明確な de-risk シグナルだけを通しやすくする。
        gap_scale = torch.tanh(gap_excess / band_scale)
        return (trade_signal + gap_boost * gap_scale).clamp(max=1.0)

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

    def execute_controller_greedy(
        self,
        trade_signal: torch.Tensor,
        target_inventory: torch.Tensor,
        band_width: torch.Tensor,
        current_inventory: torch.Tensor,
        trade_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Greedy rollout では target へ部分調整して threshold 依存を滑らかにする."""
        target_gap = target_inventory - current_inventory
        trade_score = self._trade_score(trade_signal, target_gap, band_width)
        above_band = torch.abs(target_gap) > band_width
        denom = max(1e-6, 1.0 - trade_threshold)
        trade_frac = ((trade_score - trade_threshold) / denom).clamp(min=0.0, max=1.0)
        bounded_step = self._bounded_step(target_gap)
        next_inventory = torch.where(
            above_band,
            current_inventory + trade_frac * bounded_step,
            current_inventory,
        )
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
        trade_logits, target_mean, target_std, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_dist = Bernoulli(logits=trade_logits)
        target_dist = self.target_distribution(target_mean, target_std)

        trade_sample = trade_dist.sample()
        overlay_low, overlay_high = self._overlay_bounds()
        target_inventory = target_dist.rsample().clamp(min=overlay_low, max=overlay_high)

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
        trade_logits, target_mean, _target_std, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        trade_prob = torch.sigmoid(trade_logits)
        target_inventory = target_mean
        trade_prob = self._quantize_inference(trade_prob)
        band_width = self._quantize_inference(band_width)
        target_inventory = self._quantize_inference(target_inventory)
        current_inventory = self._quantize_inference(current_inventory)
        next_inventory = self.execute_controller_greedy(
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
        controller_state = torch.zeros(1, self.inventory_dim, dtype=torch.float32, device=dev)
        for i in range(len(z_np)):
            reg_t = regime[i:i + 1] if regime is not None else None
            next_position = self.act_greedy(
                z[i:i + 1],
                h[i:i + 1],
                inventory=controller_state,
                regime=reg_t,
            )
            next_position = self._quantize_inference(next_position)
            positions[i] = float(next_position.item())
            controller_state = self.update_controller_state(controller_state, next_position)

        if was_training:
            self.train()
        return positions
