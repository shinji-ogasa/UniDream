"""Actor モジュール.

`trade / target inventory / no-trade band` を主出力に持つ inventory controller。
policy が学習するのは inventory path で、世界モデルにも executed inventory を渡す。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, RelaxedBernoulli

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
        advantage_dim: int = 0,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.unimix_ratio = unimix_ratio
        self.regime_dim = regime_dim
        self.inventory_dim = inventory_dim
        self.advantage_dim = advantage_dim

        in_dim = z_dim + h_dim + regime_dim + inventory_dim + advantage_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
        self.trunk = nn.Sequential(*layers)
        self.trade_head = nn.Linear(hidden_dim, 1)
        self.execution_head = nn.Linear(hidden_dim, 1)
        self.target_logits_head = nn.Linear(hidden_dim, act_dim)
        self.regime_target_bias_head = (
            nn.Linear(regime_dim, act_dim, bias=False) if regime_dim > 0 else None
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.target_std_head = nn.Linear(hidden_dim, 1)
        self.band_head = nn.Linear(hidden_dim, 1)

        # 初期状態は「まず hold、必要なときだけ動く」に寄せる。
        nn.init.constant_(self.trade_head.bias, -0.5)
        nn.init.constant_(self.execution_head.bias, -0.5)
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

    def _use_residual_controller(self) -> bool:
        return bool(getattr(self, "use_residual_controller", False))

    def _use_separate_execution_head(self) -> bool:
        return bool(getattr(self, "separate_execution_head", False))

    def _ensure_advantage(
        self,
        advantage: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.advantage_dim <= 0:
            return None
        if advantage is None:
            default_adv = float(getattr(self, "infer_advantage_level", 0.0))
            return torch.full(
                (*ref.shape[:-1], self.advantage_dim),
                default_adv,
                dtype=ref.dtype,
                device=ref.device,
            )
        if advantage.ndim == ref.ndim - 1:
            advantage = advantage.unsqueeze(-1)
        if advantage.shape[-1] == self.advantage_dim:
            return advantage
        if advantage.shape[-1] > self.advantage_dim:
            return advantage[..., :self.advantage_dim]
        pad_shape = list(advantage.shape)
        pad_shape[-1] = self.advantage_dim - advantage.shape[-1]
        pad = torch.zeros(*pad_shape, dtype=advantage.dtype, device=advantage.device)
        return torch.cat([advantage, pad], dim=-1)

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
        advantage: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inventory = self._ensure_controller_state(inventory, z)
        advantage = self._ensure_advantage(advantage, z)

        parts = [z, h, inventory]
        if self.regime_dim > 0:
            if regime is None:
                regime = torch.zeros(*z.shape[:-1], self.regime_dim, dtype=z.dtype, device=z.device)
            parts.append(regime)
        if advantage is not None:
            parts.append(advantage)
        x = torch.cat(parts, dim=-1)
        return self.trunk(x), inventory

    def _target_overlay_values_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._target_values_tensor(device, dtype) - self._benchmark_position()

    def controller_outputs_full(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        advantage: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """trade logit, target logits/mean/std, band width, current inventory."""
        hidden, inventory_t = self._prepare_inputs(z, h, inventory=inventory, regime=regime, advantage=advantage)
        trade_logits = self.trade_head(hidden).squeeze(-1)
        min_band = float(getattr(self, "min_band", 0.02))
        max_band = float(getattr(self, "max_band", 0.20))
        band_width = min_band + max_band * torch.sigmoid(self.band_head(hidden).squeeze(-1))
        min_target_std = float(getattr(self, "min_target_std", 0.05))
        max_target_std = float(getattr(self, "max_target_std", 0.25))
        target_std = min_target_std + (max_target_std - min_target_std) * torch.sigmoid(
            self.target_std_head(hidden).squeeze(-1)
        )
        target_values = self._target_overlay_values_tensor(hidden.device, hidden.dtype)
        if self._use_residual_controller():
            overlay_low, overlay_high = self._overlay_bounds()
            residual_min = float(getattr(self, "residual_min_overlay", overlay_low))
            residual_max = float(getattr(self, "residual_max_overlay", overlay_high))
            residual_min = max(overlay_low, residual_min)
            residual_max = min(overlay_high, residual_max)
            if residual_max <= residual_min + 1e-6:
                target_mean = torch.full_like(trade_logits, residual_max)
            else:
                residual_frac = torch.sigmoid(self.residual_head(hidden).squeeze(-1))
                target_mean = residual_min + (residual_max - residual_min) * residual_frac
            regime_caps = getattr(self, "regime_overlay_caps", None)
            if regime_caps is not None and regime is not None and regime.shape[-1] > 0:
                cap_tensor = torch.as_tensor(regime_caps, dtype=target_mean.dtype, device=target_mean.device)
                cap_tensor = cap_tensor[:regime.shape[-1]]
                dynamic_min = (regime[..., : cap_tensor.shape[0]] * cap_tensor).sum(dim=-1)
                target_mean = torch.maximum(target_mean, dynamic_min)
            logit_scale = target_std.clamp_min(1e-4).unsqueeze(-1) * max(temperature, 1e-6)
            target_logits = -0.5 * ((target_values.unsqueeze(0) - target_mean.unsqueeze(-1)) / logit_scale) ** 2
        else:
            target_logits = self.target_logits_head(hidden) / max(temperature, 1e-6)
        if (
            bool(getattr(self, "use_regime_target_bias", False))
            and self.regime_target_bias_head is not None
            and regime is not None
            and regime.shape[-1] > 0
        ):
            bias_scale = float(getattr(self, "regime_target_bias_scale", 1.0))
            target_logits = target_logits + bias_scale * self.regime_target_bias_head(regime)
        if not self._use_residual_controller():
            target_probs = F.softmax(target_logits, dim=-1)
            target_mean = (target_probs * target_values).sum(dim=-1)
        return (
            trade_logits,
            target_logits,
            target_mean,
            target_std,
            band_width,
            self._current_inventory_from_state(inventory_t),
        )

    def controller_outputs(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        advantage: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """adjust logit, target mean/std, band width, current inventory."""
        trade_logits, _target_logits, target_mean, target_std, band_width, current_inventory = (
            self.controller_outputs_full(
                z,
                h,
                inventory=inventory,
                regime=regime,
                advantage=advantage,
                temperature=temperature,
            )
        )
        return trade_logits, target_mean, target_std, band_width, current_inventory

    def execution_logits(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        advantage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, _inventory_t = self._prepare_inputs(z, h, inventory=inventory, regime=regime, advantage=advantage)
        return self.execution_head(hidden).squeeze(-1)

    def _target_values_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = getattr(self, "target_values", None)
        if values is None:
            abs_min, abs_max = self._absolute_bounds()
            values = np.linspace(abs_min, abs_max, self.act_dim, dtype=np.float32)
        return torch.as_tensor(values, dtype=dtype, device=device)

    def target_indices(self, absolute_positions: torch.Tensor) -> torch.Tensor:
        target_values = self._target_values_tensor(absolute_positions.device, absolute_positions.dtype)
        return torch.argmin(torch.abs(absolute_positions.unsqueeze(-1) - target_values), dim=-1)

    def target_soft_labels(self, absolute_positions: torch.Tensor) -> torch.Tensor:
        target_values = self._target_values_tensor(absolute_positions.device, absolute_positions.dtype)
        sorted_values, sort_idx = torch.sort(target_values)
        flat_pos = absolute_positions.reshape(-1).contiguous()
        upper = torch.bucketize(flat_pos, sorted_values)
        upper = upper.clamp(max=len(sorted_values) - 1)
        lower = (upper - 1).clamp(min=0)

        lower_val = sorted_values[lower]
        upper_val = sorted_values[upper]
        denom = (upper_val - lower_val).abs()
        exact = denom < 1e-8
        upper_w = torch.where(
            exact,
            torch.zeros_like(flat_pos),
            ((flat_pos - lower_val) / (upper_val - lower_val)).clamp(0.0, 1.0),
        )
        lower_w = torch.where(exact, torch.ones_like(flat_pos), 1.0 - upper_w)

        labels_sorted = torch.zeros(flat_pos.shape[0], len(sorted_values), dtype=absolute_positions.dtype, device=absolute_positions.device)
        labels_sorted.scatter_add_(1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
        labels_sorted.scatter_add_(1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))
        labels = torch.zeros_like(labels_sorted)
        labels[:, sort_idx] = labels_sorted
        return labels.reshape(*absolute_positions.shape, len(target_values))

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

    def _adjustment_rate(
        self,
        trade_signal: torch.Tensor,
        target_gap: torch.Tensor,
        band_width: torch.Tensor,
    ) -> torch.Tensor:
        """連続 adjustment-rate を返す."""
        rate_scale = float(getattr(self, "infer_adjust_rate_scale", 1.0))
        base_rate = (trade_signal * rate_scale).clamp(min=0.0, max=1.0)
        gap_boost = float(getattr(self, "infer_gap_boost", 0.0))
        if gap_boost <= 0.0:
            return base_rate

        gap_excess = (torch.abs(target_gap) - band_width).clamp(min=0.0)
        min_band = float(getattr(self, "min_band", 0.02))
        band_scale = band_width.clamp(min=min_band)
        gap_scale = torch.tanh(gap_excess / band_scale)
        return (base_rate + gap_boost * gap_scale).clamp(max=1.0)

    def execute_controller(
        self,
        trade_signal: torch.Tensor,
        target_inventory: torch.Tensor,
        band_width: torch.Tensor,
        current_inventory: torch.Tensor,
        band_sharpness: float = 32.0,
    ) -> torch.Tensor:
        """連続 adjustment-rate で executed inventory に変換する."""
        target_gap = target_inventory - current_inventory
        adjustment_rate = self._adjustment_rate(trade_signal, target_gap, band_width)
        band_frac = torch.sigmoid(band_sharpness * (torch.abs(target_gap) - band_width))
        bounded_step = self._bounded_step(target_gap)
        next_inventory = current_inventory + adjustment_rate * band_frac * bounded_step
        overlay_low, overlay_high = self._overlay_bounds()
        return next_inventory.clamp(min=overlay_low, max=overlay_high)

    def execute_controller_greedy(
        self,
        trade_signal: torch.Tensor,
        target_inventory: torch.Tensor,
        band_width: torch.Tensor,
        current_inventory: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy rollout でも連続 adjustment-rate をそのまま使う."""
        return self.soft_execute_controller(
            trade_signal=trade_signal,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
        )

    def soft_execute_controller(
        self,
        trade_signal: torch.Tensor,
        target_inventory: torch.Tensor,
        band_width: torch.Tensor,
        current_inventory: torch.Tensor,
        band_sharpness: float = 16.0,
    ) -> torch.Tensor:
        """微分可能な近似実行則.

        trade gate と band gate を連続化して、BC/AC の補助損失に使う。
        """
        target_gap = target_inventory - current_inventory
        trade_frac = self._adjustment_rate(trade_signal, target_gap, band_width)
        band_frac = torch.sigmoid(band_sharpness * (torch.abs(target_gap) - band_width))
        bounded_step = self._bounded_step(target_gap)
        next_inventory = current_inventory + trade_frac * band_frac * bounded_step
        overlay_low, overlay_high = self._overlay_bounds()
        return next_inventory.clamp(min=overlay_low, max=overlay_high)

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        advantage: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """executed inventory・log_prob・entropy を返す."""
        trade_logits, target_mean, target_std, band_width, current_inventory = self.controller_outputs(
            z, h, inventory=inventory, regime=regime, advantage=advantage
        )
        trade_base = Bernoulli(logits=trade_logits)
        relax_temp = torch.as_tensor(
            float(getattr(self, "adjustment_temperature", 0.25)),
            dtype=trade_logits.dtype,
            device=trade_logits.device,
        )
        trade_dist = RelaxedBernoulli(temperature=relax_temp, logits=trade_logits)
        target_dist = self.target_distribution(target_mean, target_std)

        trade_sample = trade_dist.rsample().clamp(min=1e-5, max=1.0 - 1e-5)
        overlay_low, overlay_high = self._overlay_bounds()
        target_inventory = target_dist.rsample().clamp(min=overlay_low, max=overlay_high)

        next_inventory = self.execute_controller(
            trade_signal=trade_sample,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
        )
        trade_log_prob = trade_dist.log_prob(trade_sample)
        target_log_prob = target_dist.log_prob(target_inventory) * trade_sample.detach()
        log_prob = trade_log_prob + target_log_prob
        entropy = trade_base.entropy() + target_dist.entropy() * torch.sigmoid(trade_logits)
        next_position = self._overlay_to_position(next_inventory)
        return next_position.unsqueeze(-1), log_prob, entropy

    @torch.no_grad()
    def act_greedy(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor | None = None,
        regime: torch.Tensor | None = None,
        advantage: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """greedy execution に対応する executed inventory を返す."""
        del temperature
        inventory_state = self._ensure_controller_state(inventory, z)
        trade_logits, target_logits, target_mean, target_std, band_width, current_inventory = self.controller_outputs_full(
            z, h, inventory=inventory_state, regime=regime, advantage=advantage
        )
        trade_prob = torch.sigmoid(trade_logits)
        execution_prob = trade_prob
        if self._use_separate_execution_head():
            execution_prob = torch.sigmoid(
                self.execution_logits(
                    z,
                    h,
                    inventory=inventory_state,
                    regime=regime,
                    advantage=advantage,
                )
            )
        target_inventory = target_mean
        target_probs = F.softmax(target_logits, dim=-1)
        baseline_target_idx = int(getattr(self, "baseline_target_index", self.act_dim - 1))
        selected_target_idx = target_probs.argmax(dim=-1)
        baseline_target_prob = target_probs[..., baseline_target_idx]
        if bool(getattr(self, "infer_target_from_logits", False)):
            target_overlays = self._target_overlay_values_tensor(target_logits.device, target_logits.dtype)
            modal_target = target_overlays[selected_target_idx]
            blend = float(getattr(self, "infer_logits_target_blend", 1.0))
            target_inventory = (1.0 - blend) * target_inventory + blend * modal_target
        bootstrap_prob = float(getattr(self, "infer_bootstrap_target_prob", 0.0))
        bootstrap_std = float(getattr(self, "infer_bootstrap_target_std", 0.0))
        bootstrap_trade_signal = float(getattr(self, "infer_bootstrap_trade_signal", 0.0))
        baseline_margin = float(getattr(self, "infer_bootstrap_baseline_margin", 0.0))
        support_min_count = float(getattr(self, "infer_support_min_count", 0.0))
        support_min_ratio = float(getattr(self, "infer_support_min_ratio", 0.0))
        current_abs_position = self._overlay_to_position(current_inventory)
        current_target_idx = self.target_indices(current_abs_position)
        support_bootstrap = torch.zeros_like(target_inventory, dtype=torch.bool)
        support_counts = getattr(self, "support_transition_counts", None)
        if support_counts is not None and (support_min_count > 0.0 or support_min_ratio > 0.0):
            support_tensor = torch.as_tensor(
                support_counts,
                dtype=target_probs.dtype,
                device=target_probs.device,
            )
            if regime is not None and regime.shape[-1] > 0:
                regime_idx = regime.argmax(dim=-1).to(dtype=torch.long)
            else:
                regime_idx = torch.zeros_like(selected_target_idx, dtype=torch.long)
            regime_idx = regime_idx.clamp(min=0, max=support_tensor.shape[0] - 1)
            current_target_idx = current_target_idx.to(dtype=torch.long).clamp(min=0, max=support_tensor.shape[1] - 1)
            selected_target_idx = selected_target_idx.to(dtype=torch.long).clamp(min=0, max=support_tensor.shape[2] - 1)
            support_count = support_tensor[regime_idx, current_target_idx, selected_target_idx]
            support_total = support_tensor[regime_idx, current_target_idx].sum(dim=-1).clamp_min(1.0)
            support_ratio = support_count / support_total
            support_bootstrap = (support_count < support_min_count) | (support_ratio < support_min_ratio)
        if bootstrap_prob > 0.0 or bootstrap_std > 0.0:
            top_prob = target_probs.max(dim=-1).values
            should_bootstrap = target_inventory < 0.0
            if bootstrap_prob > 0.0:
                should_bootstrap = should_bootstrap & (top_prob < bootstrap_prob)
            if bootstrap_std > 0.0:
                should_bootstrap = should_bootstrap & (target_std > bootstrap_std)
            if baseline_margin > 0.0:
                selected_prob = target_probs.gather(-1, selected_target_idx.unsqueeze(-1)).squeeze(-1)
                should_bootstrap = should_bootstrap | (
                    (target_inventory < 0.0) & ((selected_prob - baseline_target_prob) < baseline_margin)
                )
            should_bootstrap = should_bootstrap | ((target_inventory < 0.0) & support_bootstrap)
            if should_bootstrap.any():
                target_inventory = torch.where(should_bootstrap, torch.zeros_like(target_inventory), target_inventory)
                if bootstrap_trade_signal > 0.0:
                    bootstrap_signal = torch.full_like(trade_prob, bootstrap_trade_signal)
                    execution_prob = torch.where(
                        should_bootstrap,
                        torch.maximum(execution_prob, bootstrap_signal),
                        execution_prob,
                    )
        regime_active_threshold = float(getattr(self, "infer_regime_active_threshold", 0.0))
        regime_active_state = int(getattr(self, "infer_regime_active_state", 0))
        if regime is not None and regime_active_threshold > 0.0 and regime.shape[-1] > 0:
            regime_idx = max(0, min(regime.shape[-1] - 1, regime_active_state))
            allow_active = regime[:, regime_idx] >= regime_active_threshold
            target_inventory = torch.where(
                (target_inventory < 0.0) & (~allow_active),
                torch.zeros_like(target_inventory),
                target_inventory,
            )
        active_std_max = float(getattr(self, "infer_active_std_max", 0.0))
        active_zscore_min = float(getattr(self, "infer_active_zscore_min", 0.0))
        if active_std_max > 0.0:
            target_inventory = torch.where(
                (target_inventory != 0.0) & (target_std > active_std_max),
                torch.zeros_like(target_inventory),
                target_inventory,
            )
        if active_zscore_min > 0.0:
            gap_to_target = torch.abs(target_inventory - current_inventory)
            signal_z = gap_to_target / target_std.clamp_min(1e-4)
            target_inventory = torch.where(
                (target_inventory != 0.0) & (signal_z < active_zscore_min),
                torch.zeros_like(target_inventory),
                target_inventory,
            )
        event_entry_gap = float(getattr(self, "infer_event_entry_gap", 0.0))
        event_exit_gap = float(getattr(self, "infer_event_exit_gap", 0.0))
        event_trade_prob = float(getattr(self, "infer_event_trade_prob", 0.0))
        event_target_overlay = getattr(self, "infer_event_target_overlay", None)
        event_min_hold_bars = float(getattr(self, "infer_event_min_hold_bars", 0.0))
        if event_entry_gap > 0.0 or event_exit_gap > 0.0:
            hold_feature = (
                inventory_state[..., 2]
                if self.inventory_dim >= 3 and inventory_state.shape[-1] >= 3
                else torch.zeros_like(current_inventory)
            )
            min_hold_progress = event_min_hold_bars / max(self._state_hold_scale(), 1.0)
            hold_ready = hold_feature >= min_hold_progress
            in_hedge = current_inventory < -self._trade_state_eps()
            entry_signal = target_inventory <= -event_entry_gap
            if event_trade_prob > 0.0:
                entry_signal = entry_signal & (trade_prob >= event_trade_prob)
            fixed_overlay = (
                torch.full_like(target_inventory, float(event_target_overlay))
                if event_target_overlay is not None
                else target_inventory
            )
            stay_hedged = (~hold_ready) | (target_inventory <= -event_exit_gap)
            target_inventory = torch.where(
                in_hedge,
                torch.where(
                    stay_hedged,
                    torch.minimum(current_inventory, fixed_overlay),
                    torch.zeros_like(target_inventory),
                ),
                torch.where(
                    entry_signal,
                    torch.minimum(fixed_overlay, torch.zeros_like(target_inventory)),
                    torch.zeros_like(target_inventory),
                ),
            )
        underweight_adjust_scale = float(getattr(self, "infer_underweight_adjust_scale", 1.0))
        if underweight_adjust_scale < 1.0:
            more_underweight = (target_inventory < 0.0) & (target_inventory < current_inventory)
            trade_prob = torch.where(
                more_underweight,
                execution_prob * underweight_adjust_scale,
                execution_prob,
            )
        min_trade_floor = float(getattr(self, "infer_min_trade_floor", 0.0))
        min_trade_gap = float(getattr(self, "infer_min_trade_gap", 0.0))
        min_trade_scale = float(getattr(self, "infer_min_trade_scale", 0.0))
        if min_trade_floor > 0.0:
            target_gap_abs = torch.abs(target_inventory - current_inventory)
            gap_excess = (target_gap_abs - min_trade_gap).clamp(min=0.0)
            if min_trade_scale > 0.0:
                floor_strength = torch.tanh(gap_excess / max(min_trade_scale, 1e-6))
            else:
                floor_strength = (gap_excess > 0.0).to(trade_prob.dtype)
            trade_floor = min_trade_floor * floor_strength
            execution_prob = torch.maximum(execution_prob, trade_floor)
        if bool(getattr(self, "infer_direct_target_track", False)):
            direct_scale = float(getattr(self, "infer_direct_track_scale", 1.0))
            target_gap = target_inventory - current_inventory
            bounded_step = self._bounded_step(target_gap)
            next_inventory = current_inventory + direct_scale * bounded_step
            overlay_low, overlay_high = self._overlay_bounds()
            next_inventory = next_inventory.clamp(min=overlay_low, max=overlay_high)
            next_position = self._overlay_to_position(next_inventory)
            return next_position.unsqueeze(-1)
        execution_prob = self._quantize_inference(execution_prob)
        band_width = self._quantize_inference(band_width)
        target_inventory = self._quantize_inference(target_inventory)
        current_inventory = self._quantize_inference(current_inventory)
        next_inventory = self.execute_controller_greedy(
            trade_signal=execution_prob,
            target_inventory=target_inventory,
            band_width=band_width,
            current_inventory=current_inventory,
        )
        next_position = self._overlay_to_position(next_inventory)
        return next_position.unsqueeze(-1)

    @torch.no_grad()
    def predict_positions(
        self,
        z_np: np.ndarray,
        h_np: np.ndarray,
        regime_np: np.ndarray | None = None,
        advantage_np: np.ndarray | float | None = None,
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
        advantage = None
        if self.advantage_dim > 0:
            if advantage_np is None:
                advantage = torch.full((len(z_np), self.advantage_dim), float(getattr(self, "infer_advantage_level", 0.0)), dtype=torch.float32, device=dev)
            elif np.isscalar(advantage_np):
                advantage = torch.full((len(z_np), self.advantage_dim), float(advantage_np), dtype=torch.float32, device=dev)
            else:
                advantage = torch.tensor(advantage_np, dtype=torch.float32, device=dev)
                if advantage.ndim == 1:
                    advantage = advantage.unsqueeze(-1)

        positions = np.zeros(len(z_np), dtype=np.float32)
        controller_state = torch.zeros(1, self.inventory_dim, dtype=torch.float32, device=dev)
        for i in range(len(z_np)):
            reg_t = regime[i:i + 1] if regime is not None else None
            adv_t = advantage[i:i + 1] if advantage is not None else None
            next_position = self.act_greedy(
                z[i:i + 1],
                h[i:i + 1],
                inventory=controller_state,
                regime=reg_t,
                advantage=adv_t,
            )
            next_position = self._quantize_inference(next_position)
            positions[i] = float(next_position.item())
            controller_state = self.update_controller_state(controller_state, next_position)

        if was_training:
            self.train()
        return positions
