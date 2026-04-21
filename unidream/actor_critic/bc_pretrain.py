"""BC (Behavior Cloning) 事前学習モジュール.

Hindsight Oracle の最適 inventory path を教師ラベルとして、
Actor の `trade / target / band` を数エポック学習する。

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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from unidream.actor_critic.actor import Actor
from unidream.device import resolve_device
from unidream.data.oracle import smooth_aim_positions

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


class ControllerPathDataset(Dataset):
    """短い将来 window の controller path を返す dataset."""

    def __init__(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        oracle_positions: torch.Tensor,
        inventory: torch.Tensor,
        horizon: int,
        regime: torch.Tensor | None = None,
        soft_labels: torch.Tensor | None = None,
        sample_quality: torch.Tensor | None = None,
        advantage: torch.Tensor | None = None,
    ):
        self.z = z
        self.h = h
        self.oracle_positions = oracle_positions
        self.inventory = inventory
        self.horizon = max(1, int(horizon))
        self.regime = regime
        self.soft_labels = soft_labels
        self.sample_quality = sample_quality
        self.advantage = advantage
        self.length = max(0, len(z) - self.horizon + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        s = idx
        e = idx + self.horizon
        items = [self.z[s:e], self.h[s:e], self.oracle_positions[s:e], self.inventory[s]]
        if self.regime is not None:
            items.append(self.regime[s:e])
        if self.soft_labels is not None:
            items.append(self.soft_labels[s:e])
        if self.sample_quality is not None:
            items.append(self.sample_quality[s:e])
        if self.advantage is not None:
            items.append(self.advantage[s:e])
        return tuple(items)


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
        label_smoothing: ラベル平滑化係数
        entropy_coef: エントロピー正則化係数
        chunk_size: Action Chunking のチャンクサイズ（1 で無効）
        class_balanced: クラス頻度逆数による損失重み付けを有効化
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
        label_smoothing: float = 0.0,
        entropy_coef: float = 0.0,
        chunk_size: int = 1,
        class_balanced: bool = False,
        target_aux_coef: float = 1.0,
        trade_aux_coef: float = 0.5,
        band_aux_coef: float = 0.25,
        execution_aux_coef: float = 0.0,
        path_aux_coef: float = 0.0,
        path_horizon: int = 1,
        path_position_coef: float = 1.0,
        path_turnover_coef: float = 0.0,
        path_shortfall_coef: float = 0.0,
        soft_trade_targets: bool = True,
        trade_target_scale: float | None = None,
        self_condition_prob: float = 0.0,
        self_condition_interval: int = 1,
        self_condition_warmup_epochs: int = 0,
        self_condition_mode: str = "mix",
        self_condition_blend: float = 0.0,
        self_condition_max_position_gap: float | None = None,
        self_condition_max_underweight_gap: float | None = None,
        self_condition_relabel_step: float | None = None,
        self_condition_relabel_band: float = 0.0,
        relabel_aim_max_step: float = 0.125,
        relabel_aim_band: float = 0.0,
        relabel_min_position: float = -1.0,
        relabel_max_position: float = 1.0,
        relabel_benchmark_position: float = 0.0,
        relabel_underweight_confirm_bars: int = 0,
        relabel_underweight_min_scale: float = 0.0,
        relabel_underweight_floor_position: float | None = None,
        relabel_underweight_step_scale: float = 1.0,
        residual_target_coef: float = 1.0,
        residual_aux_ce_coef: float = 0.0,
        target_dist_match_coef: float = 0.0,
        position_mean_match_coef: float = 0.0,
        target_regime_dist_match_coef: float = 0.0,
        short_mass_match_coef: float = 0.0,
        support_prior_coef: float = 0.0,
        support_prior_underweight_only: bool = True,
        mode_target_coef: float = 0.0,
        mode_target_margin: float = 0.05,
        mode_target_neutral_margin: float = 0.0,
        mode_target_gap_min: float = 0.0,
        mode_target_positive_only: bool = False,
        mode_rate_match_coef: float = 0.0,
        mode_regime_rate_match_coef: float = 0.0,
        dual_head_anchor_coef: float = 0.0,
        dual_head_underweight_coef: float = 0.0,
        dual_head_separation_coef: float = 0.0,
        dual_head_neutral_margin: float = 0.0,
        dual_head_underweight_margin: float = 0.05,
        dual_head_separation_margin: float = 0.05,
        direct_band_target_coef: float = 0.0,
        direct_band_margin: float = 0.05,
        direct_hold_band_margin: float = 0.02,
        band_aux_trade_only: bool = False,
        direct_band_trade_only: bool = False,
        direct_band_gap_min: float = 0.0,
        recovery_trade_coef: float = 0.0,
        recovery_band_coef: float = 0.0,
        recovery_target_coef: float = 0.0,
        recovery_execution_coef: float = 0.0,
        recovery_underweight_margin: float = 0.05,
        recovery_target_margin: float = 0.05,
        sample_quality_coef: float = 0.0,
        sample_quality_clip: float = 4.0,
        device: str = "cpu",
    ):
        self.actor = actor
        self.device = torch.device(resolve_device(device))
        self.actor.to(self.device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.entropy_coef = entropy_coef
        self.chunk_size = max(1, chunk_size)
        self.class_balanced = class_balanced
        self.target_aux_coef = target_aux_coef
        self.trade_aux_coef = trade_aux_coef
        self.band_aux_coef = band_aux_coef
        self.execution_aux_coef = execution_aux_coef
        self.path_aux_coef = path_aux_coef
        self.path_horizon = max(1, int(path_horizon))
        self.path_position_coef = path_position_coef
        self.path_turnover_coef = path_turnover_coef
        self.path_shortfall_coef = path_shortfall_coef
        self.soft_trade_targets = soft_trade_targets
        self.trade_target_scale = trade_target_scale
        self.self_condition_prob = float(self_condition_prob)
        self.self_condition_interval = max(1, int(self_condition_interval))
        self.self_condition_warmup_epochs = max(0, int(self_condition_warmup_epochs))
        self.self_condition_mode = str(self_condition_mode)
        self.self_condition_blend = float(np.clip(self_condition_blend, 0.0, 1.0))
        self.self_condition_max_position_gap = (
            None if self_condition_max_position_gap is None else max(0.0, float(self_condition_max_position_gap))
        )
        self.self_condition_max_underweight_gap = (
            None
            if self_condition_max_underweight_gap is None
            else max(0.0, float(self_condition_max_underweight_gap))
        )
        self.self_condition_relabel_step = (
            None if self_condition_relabel_step is None else max(0.0, float(self_condition_relabel_step))
        )
        self.self_condition_relabel_band = max(0.0, float(self_condition_relabel_band))
        self.relabel_aim_max_step = float(max(relabel_aim_max_step, 1e-6))
        self.relabel_aim_band = float(max(relabel_aim_band, 0.0))
        self.relabel_min_position = float(relabel_min_position)
        self.relabel_max_position = float(relabel_max_position)
        self.relabel_benchmark_position = float(relabel_benchmark_position)
        self.relabel_underweight_confirm_bars = int(max(relabel_underweight_confirm_bars, 0))
        self.relabel_underweight_min_scale = float(np.clip(relabel_underweight_min_scale, 0.0, 1.0))
        self.relabel_underweight_floor_position = (
            None if relabel_underweight_floor_position is None else float(relabel_underweight_floor_position)
        )
        self.relabel_underweight_step_scale = float(np.clip(relabel_underweight_step_scale, 0.0, 1.0))
        self.residual_target_coef = float(max(residual_target_coef, 0.0))
        self.residual_aux_ce_coef = float(max(residual_aux_ce_coef, 0.0))
        self.target_dist_match_coef = float(max(target_dist_match_coef, 0.0))
        self.position_mean_match_coef = float(max(position_mean_match_coef, 0.0))
        self.target_regime_dist_match_coef = float(max(target_regime_dist_match_coef, 0.0))
        self.short_mass_match_coef = float(max(short_mass_match_coef, 0.0))
        self.support_prior_coef = float(max(support_prior_coef, 0.0))
        self.support_prior_underweight_only = bool(support_prior_underweight_only)
        self.mode_target_coef = float(max(mode_target_coef, 0.0))
        self.mode_target_margin = float(max(mode_target_margin, 0.0))
        self.mode_target_neutral_margin = float(max(mode_target_neutral_margin, 0.0))
        self.mode_target_gap_min = float(max(mode_target_gap_min, 0.0))
        self.mode_target_positive_only = bool(mode_target_positive_only)
        self.mode_rate_match_coef = float(max(mode_rate_match_coef, 0.0))
        self.mode_regime_rate_match_coef = float(max(mode_regime_rate_match_coef, 0.0))
        self.dual_head_anchor_coef = float(max(dual_head_anchor_coef, 0.0))
        self.dual_head_underweight_coef = float(max(dual_head_underweight_coef, 0.0))
        self.dual_head_separation_coef = float(max(dual_head_separation_coef, 0.0))
        self.dual_head_neutral_margin = float(max(dual_head_neutral_margin, 0.0))
        self.dual_head_underweight_margin = float(max(dual_head_underweight_margin, 0.0))
        self.dual_head_separation_margin = float(max(dual_head_separation_margin, 0.0))
        self.direct_band_target_coef = float(max(direct_band_target_coef, 0.0))
        self.direct_band_margin = float(max(direct_band_margin, 0.0))
        self.direct_hold_band_margin = float(max(direct_hold_band_margin, 0.0))
        self.band_aux_trade_only = bool(band_aux_trade_only)
        self.direct_band_trade_only = bool(direct_band_trade_only)
        self.direct_band_gap_min = float(max(direct_band_gap_min, 0.0))
        self.recovery_trade_coef = float(max(recovery_trade_coef, 0.0))
        self.recovery_band_coef = float(max(recovery_band_coef, 0.0))
        self.recovery_target_coef = float(max(recovery_target_coef, 0.0))
        self.recovery_execution_coef = float(max(recovery_execution_coef, 0.0))
        self.recovery_underweight_margin = float(max(recovery_underweight_margin, 0.0))
        self.recovery_target_margin = float(max(recovery_target_margin, 0.0))
        self.sample_quality_coef = float(max(sample_quality_coef, 0.0))
        self.sample_quality_clip = float(max(sample_quality_clip, 0.0))

        # SIRL 重みネット
        self.use_sirl = sirl_hidden > 0
        if self.use_sirl:
            self.weight_net = SIRLWeightNet(z_dim + h_dim, sirl_hidden).to(self.device)
            params = list(actor.parameters()) + list(self.weight_net.parameters())
        else:
            self.weight_net = None
            params = list(actor.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr)

    @staticmethod
    def _normalized_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.float32)
        return mask / mask.mean().clamp_min(1e-6)

    def _mixed_controller_states(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_positions: np.ndarray,
        regime_probs: np.ndarray | None = None,
        rollout_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        teacher_states = self.actor.controller_states_from_positions(oracle_positions)
        if self.self_condition_prob <= 0.0:
            return teacher_states, rollout_positions, np.asarray(oracle_positions, dtype=np.float32)

        if rollout_positions is None:
            rollout_positions = self.actor.predict_positions(
                z,
                h,
                regime_np=regime_probs,
                device=str(self.device),
            )
        stabilized_rollout = np.asarray(rollout_positions, dtype=np.float32).copy()
        if self.self_condition_max_position_gap is not None and self.self_condition_max_position_gap > 0.0:
            gap = np.full_like(stabilized_rollout, self.self_condition_max_position_gap, dtype=np.float32)
            stabilized_rollout = np.clip(
                stabilized_rollout,
                np.asarray(oracle_positions, dtype=np.float32) - gap,
                np.asarray(oracle_positions, dtype=np.float32) + gap,
            )
        if self.self_condition_max_underweight_gap is not None and self.self_condition_max_underweight_gap > 0.0:
            benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
            teacher_underweight = np.clip(benchmark_position - np.asarray(oracle_positions, dtype=np.float32), 0.0, None)
            min_rollout_position = benchmark_position - (teacher_underweight + self.self_condition_max_underweight_gap)
            stabilized_rollout = np.maximum(stabilized_rollout, min_rollout_position.astype(np.float32))
        rollout_states = self.actor.controller_states_from_positions(stabilized_rollout)
        if self.self_condition_blend > 0.0:
            rollout_states = (
                (1.0 - self.self_condition_blend) * teacher_states
                + self.self_condition_blend * rollout_states
            ).astype(np.float32)
        if self.self_condition_mode == "trajectory":
            use_rollout = np.random.rand() < self.self_condition_prob
            if not use_rollout:
                return teacher_states, stabilized_rollout, np.asarray(oracle_positions, dtype=np.float32)
            relabeled_targets = smooth_aim_positions(
                np.asarray(oracle_positions, dtype=np.float32),
                max_step=self.relabel_aim_max_step,
                band=self.relabel_aim_band,
                initial_position=float(stabilized_rollout[0]),
                min_position=self.relabel_min_position,
                max_position=self.relabel_max_position,
                benchmark_position=self.relabel_benchmark_position,
                underweight_confirm_bars=self.relabel_underweight_confirm_bars,
                underweight_min_scale=self.relabel_underweight_min_scale,
                underweight_floor_position=self.relabel_underweight_floor_position,
                underweight_step_scale=self.relabel_underweight_step_scale,
            ).astype(np.float32)
            return rollout_states.astype(np.float32), stabilized_rollout, relabeled_targets
        if self.self_condition_mode == "dagger":
            use_rollout = np.random.rand() < self.self_condition_prob
            if not use_rollout:
                return teacher_states, stabilized_rollout, np.asarray(oracle_positions, dtype=np.float32)
            benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
            min_position = float(getattr(self.actor, "abs_min_position", -1.0))
            max_position = float(getattr(self.actor, "abs_max_position", 1.0))
            current_positions = rollout_states[:, 0] + benchmark_position
            relabel_step = self.self_condition_relabel_step
            if relabel_step is None:
                relabel_step = float(getattr(self.actor, "max_position_step", 0.125))
            relabel_step = max(0.0, relabel_step)
            relabel_gap = np.asarray(oracle_positions, dtype=np.float32) - current_positions
            relabeled_targets = current_positions + np.clip(relabel_gap, -relabel_step, relabel_step)
            if self.self_condition_relabel_band > 0.0:
                relabeled_targets = np.where(
                    np.abs(relabel_gap) <= self.self_condition_relabel_band,
                    current_positions,
                    relabeled_targets,
                )
            relabeled_targets = np.clip(relabeled_targets, min_position, max_position).astype(np.float32)
            return rollout_states.astype(np.float32), stabilized_rollout, relabeled_targets
        mask = np.random.rand(len(teacher_states)) < self.self_condition_prob
        mixed_states = teacher_states.copy()
        mixed_states[mask] = rollout_states[mask]
        mixed_targets = np.asarray(oracle_positions, dtype=np.float32).copy()
        relabel_step = self.self_condition_relabel_step
        if relabel_step is None:
            relabel_step = float(getattr(self.actor, "max_position_step", 0.125))
        relabel_step = max(0.0, relabel_step)
        if relabel_step > 0.0 and mask.any():
            benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
            min_position = float(getattr(self.actor, "abs_min_position", -1.0))
            max_position = float(getattr(self.actor, "abs_max_position", 1.0))
            current_positions = mixed_states[:, 0] + benchmark_position
            relabel_gap = mixed_targets[mask] - current_positions[mask]
            relabeled = current_positions[mask] + np.clip(relabel_gap, -relabel_step, relabel_step)
            if self.self_condition_relabel_band > 0.0:
                keep_current = np.abs(relabel_gap) <= self.self_condition_relabel_band
                relabeled = np.where(keep_current, current_positions[mask], relabeled)
            mixed_targets[mask] = np.clip(relabeled, min_position, max_position).astype(np.float32)
        return mixed_states, stabilized_rollout, mixed_targets

    def _bc_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        oracle_positions: torch.Tensor,
        inventory: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        soft_labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        trade_pos_weight: Optional[torch.Tensor] = None,
        sample_quality: Optional[torch.Tensor] = None,
        advantage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inventory controller 向けの BC 損失."""
        trade_logits, target_logits, target_mean, target_std, band_width, current_inventory = (
            self.actor.controller_outputs_full(
            z, h, inventory=inventory, regime=regime, advantage=advantage
            )
        )
        benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
        oracle_target = oracle_positions - benchmark_position
        target_gap = torch.abs(oracle_target - current_inventory)
        trade_mask = (target_gap > 1e-8).float()
        if self.soft_trade_targets:
            scale = self.trade_target_scale
            if scale is None or scale <= 0.0:
                target_values = getattr(self.actor, "target_values", None)
                if target_values is not None:
                    unique_vals = np.unique(np.asarray(target_values, dtype=np.float32))
                    diffs = np.diff(np.sort(unique_vals))
                    positive_diffs = diffs[diffs > 1e-8]
                    if len(positive_diffs) > 0:
                        scale = float(np.median(positive_diffs))
                if scale is None or scale <= 0.0:
                    scale = 0.5
            trade_targets = (target_gap / float(scale)).clamp(0.0, 1.0)
        else:
            trade_targets = trade_mask
        target_reg_loss = F.smooth_l1_loss(target_mean, oracle_target, reduction="none")
        target_dist_penalty = None
        position_mean_penalty = None
        target_regime_penalty = None
        short_mass_penalty = None
        support_prior_penalty = None
        if self.actor._use_residual_controller():
            target_loss = self.residual_target_coef * target_reg_loss
            target_soft_labels = self.actor.target_soft_labels(oracle_positions).to(target_logits.dtype)
            if self.residual_aux_ce_coef > 0.0:
                if soft_labels is not None and soft_labels.shape[-1] == target_logits.shape[-1]:
                    provided_soft_labels = soft_labels.to(target_logits.dtype)
                    on_grid = target_soft_labels.max(dim=-1).values > 1.0 - 1e-6
                    blended_soft_labels = 0.5 * provided_soft_labels + 0.5 * target_soft_labels
                    target_soft_labels = torch.where(on_grid.unsqueeze(-1), blended_soft_labels, target_soft_labels)
                log_probs = F.log_softmax(target_logits, dim=-1)
                target_ce_loss = -(target_soft_labels * log_probs).sum(dim=-1)
                target_loss = target_loss + self.residual_aux_ce_coef * target_ce_loss
            if self.target_dist_match_coef > 0.0:
                pred_probs = F.softmax(target_logits, dim=-1)
                target_dist_penalty = torch.abs(
                    pred_probs.mean(dim=0) - target_soft_labels.mean(dim=0)
                ).mean()
            if self.position_mean_match_coef > 0.0:
                position_mean_penalty = torch.abs(target_mean.mean() - oracle_target.mean())
            if self.target_regime_dist_match_coef > 0.0 or self.short_mass_match_coef > 0.0:
                pred_probs = F.softmax(target_logits, dim=-1)
        else:
            target_indices = self.actor.target_indices(oracle_positions).to(dtype=torch.long)
            target_soft_labels = self.actor.target_soft_labels(oracle_positions).to(target_logits.dtype)
            if soft_labels is not None and soft_labels.shape[-1] == target_logits.shape[-1]:
                provided_soft_labels = soft_labels.to(target_logits.dtype)
                on_grid = target_soft_labels.max(dim=-1).values > 1.0 - 1e-6
                blended_soft_labels = 0.5 * provided_soft_labels + 0.5 * target_soft_labels
                target_soft_labels = torch.where(on_grid.unsqueeze(-1), blended_soft_labels, target_soft_labels)
            log_probs = F.log_softmax(target_logits, dim=-1)
            target_loss = -(target_soft_labels * log_probs).sum(dim=-1) + 0.25 * target_reg_loss
            if self.target_dist_match_coef > 0.0:
                pred_probs = F.softmax(target_logits, dim=-1)
                target_dist_penalty = torch.abs(
                    pred_probs.mean(dim=0) - target_soft_labels.mean(dim=0)
                ).mean()
            if self.position_mean_match_coef > 0.0:
                position_mean_penalty = torch.abs(target_mean.mean() - oracle_target.mean())
            if self.target_regime_dist_match_coef > 0.0 or self.short_mass_match_coef > 0.0:
                pred_probs = F.softmax(target_logits, dim=-1)
            if class_weights is not None:
                sample_class_w = class_weights.to(
                    device=target_loss.device,
                    dtype=target_loss.dtype,
                )[target_indices]
                target_loss = target_loss * sample_class_w
        if self.target_regime_dist_match_coef > 0.0 and regime is not None and regime.ndim == 2:
            regime_w = regime.to(dtype=target_logits.dtype)
            denom = regime_w.sum(dim=0).clamp_min(1e-6)
            pred_regime_mean = (regime_w.T @ pred_probs) / denom.unsqueeze(-1)
            target_regime_mean = (regime_w.T @ target_soft_labels) / denom.unsqueeze(-1)
            target_regime_penalty = torch.abs(pred_regime_mean - target_regime_mean).mean()
        if self.short_mass_match_coef > 0.0:
            target_values = self.actor._target_values_tensor(target_logits.device, target_logits.dtype)
            short_mask = target_values < benchmark_position - 1e-6
            if bool(short_mask.any()):
                pred_short_mass = pred_probs[:, short_mask].sum(dim=-1)
                target_short_mass = target_soft_labels[:, short_mask].sum(dim=-1)
                short_mass_penalty = torch.abs(pred_short_mass - target_short_mass).mean()
        if self.support_prior_coef > 0.0:
            support_counts = getattr(self.actor, "support_transition_counts", None)
            if support_counts is not None:
                support_tensor = torch.as_tensor(
                    support_counts,
                    dtype=target_logits.dtype,
                    device=target_logits.device,
                )
                if regime is not None and regime.ndim == 2 and regime.shape[-1] > 0:
                    regime_idx = regime.argmax(dim=-1).to(dtype=torch.long)
                else:
                    regime_idx = torch.zeros_like(oracle_target, dtype=torch.long)
                current_abs_position = current_inventory + benchmark_position
                current_target_idx = self.actor.target_indices(current_abs_position).to(dtype=torch.long)
                regime_idx = regime_idx.clamp(min=0, max=support_tensor.shape[0] - 1)
                current_target_idx = current_target_idx.clamp(min=0, max=support_tensor.shape[1] - 1)
                support_row = support_tensor[regime_idx, current_target_idx]
                support_probs = support_row + 1e-4
                support_probs = support_probs / support_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                support_mask = torch.ones_like(oracle_target, dtype=torch.bool)
                if self.support_prior_underweight_only:
                    support_mask = oracle_target < -1e-6
                support_prior_penalty = F.kl_div(
                    F.log_softmax(target_logits, dim=-1),
                    support_probs,
                    reduction="none",
                ).sum(dim=-1)
                support_prior_penalty = support_prior_penalty * self._normalized_mask(
                    support_mask.to(target_logits.dtype)
                )
        recovery_mask_f = None
        if (
            self.recovery_trade_coef > 0.0
            or self.recovery_band_coef > 0.0
            or self.recovery_target_coef > 0.0
            or self.recovery_execution_coef > 0.0
        ):
            recovery_from_underweight = current_inventory < -self.recovery_underweight_margin
            recovery_to_benchmark = oracle_target > (current_inventory + self.recovery_target_margin)
            underweight_recovery_mask = recovery_from_underweight & recovery_to_benchmark
            recovery_mask_f = self._normalized_mask(underweight_recovery_mask.to(target_loss.dtype))
            if self.recovery_target_coef > 0.0:
                target_loss = target_loss + self.recovery_target_coef * target_reg_loss * recovery_mask_f
        if trade_pos_weight is not None:
            target_w = torch.where(
                trade_mask > 0.5,
                trade_pos_weight.to(device=target_loss.device, dtype=target_loss.dtype),
                torch.ones_like(target_loss),
            )
            target_loss = target_loss * target_w

        trade_pred = torch.sigmoid(trade_logits)
        trade_loss = F.smooth_l1_loss(trade_pred, trade_targets, reduction="none")
        recovery_trade_loss = None
        recovery_band_loss = None
        mode_loss = None
        mode_rate_penalty = None
        mode_regime_rate_penalty = None
        dual_head_anchor_loss = None
        dual_head_underweight_loss = None
        dual_head_separation_loss = None
        if recovery_mask_f is not None:
            if self.recovery_trade_coef > 0.0:
                recovery_trade_targets = torch.ones_like(trade_logits)
                recovery_trade_loss = F.binary_cross_entropy_with_logits(
                    trade_logits,
                    recovery_trade_targets,
                    reduction="none",
                )
                recovery_trade_loss = recovery_trade_loss * recovery_mask_f
            if self.recovery_band_coef > 0.0:
                min_band = float(getattr(self.actor, "min_band", 0.02))
                recovery_band_targets = torch.full_like(band_width, min_band)
                recovery_band_loss = F.smooth_l1_loss(
                    band_width,
                    recovery_band_targets,
                    reduction="none",
                )
                recovery_band_loss = recovery_band_loss * recovery_mask_f
        if (
            self.mode_target_coef > 0.0
            and (
                bool(getattr(self.actor, "use_dual_regime_target_bias", False))
                or bool(getattr(self.actor, "use_dual_residual_controller", False))
            )
        ):
            mode_logits = self.actor.target_mode_logits(
                z,
                h,
                inventory=inventory,
                regime=regime,
                advantage=advantage,
            )
            positive_mode_mask = oracle_target < -self.mode_target_margin
            if self.mode_target_positive_only:
                mode_supervision_mask = positive_mode_mask
            else:
                neutral_margin = self.mode_target_neutral_margin
                if neutral_margin > 0.0:
                    neutral_mode_mask = oracle_target > -neutral_margin
                    mode_supervision_mask = positive_mode_mask | neutral_mode_mask
                else:
                    mode_supervision_mask = torch.ones_like(positive_mode_mask, dtype=torch.bool)
            if self.mode_target_gap_min > 0.0:
                mode_supervision_mask = mode_supervision_mask & (target_gap >= self.mode_target_gap_min)
            mode_targets = positive_mode_mask.to(dtype=mode_logits.dtype)
            mode_loss = F.binary_cross_entropy_with_logits(mode_logits, mode_targets, reduction="none")
            mode_loss = mode_loss * self._normalized_mask(mode_supervision_mask.to(mode_loss.dtype))
            mode_probs = torch.sigmoid(mode_logits)
            if self.mode_rate_match_coef > 0.0:
                supervised_mask = mode_supervision_mask.to(mode_probs.dtype)
                supervised_denom = supervised_mask.sum().clamp_min(1.0)
                pred_mode_rate = (mode_probs * supervised_mask).sum() / supervised_denom
                target_mode_rate = (mode_targets * supervised_mask).sum() / supervised_denom
                mode_rate_penalty = torch.abs(pred_mode_rate - target_mode_rate)
            if (
                self.mode_regime_rate_match_coef > 0.0
                and regime is not None
                and regime.ndim == 2
                and regime.shape[-1] > 0
            ):
                supervised_mask = mode_supervision_mask.to(mode_probs.dtype)
                regime_w = regime.to(dtype=mode_probs.dtype) * supervised_mask.unsqueeze(-1)
                denom = regime_w.sum(dim=0).clamp_min(1e-6)
                pred_regime_rate = (regime_w * mode_probs.unsqueeze(-1)).sum(dim=0) / denom
                target_regime_rate = (regime_w * mode_targets.unsqueeze(-1)).sum(dim=0) / denom
                mode_regime_rate_penalty = torch.abs(pred_regime_rate - target_regime_rate).mean()
        if (
            bool(getattr(self.actor, "use_dual_residual_controller", False))
            and (
                self.dual_head_anchor_coef > 0.0
                or self.dual_head_underweight_coef > 0.0
                or self.dual_head_separation_coef > 0.0
            )
        ):
            target_mean_a, target_mean_b, _mode_gate = self.actor.dual_residual_components(
                z,
                h,
                inventory=inventory,
                regime=regime,
                advantage=advantage,
            )
            if self.dual_head_anchor_coef > 0.0:
                neutral_mask = oracle_target > -self.dual_head_neutral_margin
                dual_head_anchor_loss = F.smooth_l1_loss(
                    target_mean_a,
                    torch.zeros_like(target_mean_a),
                    reduction="none",
                )
                dual_head_anchor_loss = dual_head_anchor_loss * self._normalized_mask(
                    neutral_mask.to(target_mean_a.dtype)
                )
            if self.dual_head_underweight_coef > 0.0:
                underweight_mask = oracle_target < -self.dual_head_underweight_margin
                dual_head_underweight_loss = F.smooth_l1_loss(
                    target_mean_b,
                    oracle_target,
                    reduction="none",
                )
                dual_head_underweight_loss = dual_head_underweight_loss * self._normalized_mask(
                    underweight_mask.to(target_mean_b.dtype)
                )
            if self.dual_head_separation_coef > 0.0:
                separation_margin = self.dual_head_separation_margin
                separation_violation = F.relu(target_mean_b - (target_mean_a - separation_margin))
                separation_mask = oracle_target < -self.dual_head_underweight_margin
                dual_head_separation_loss = separation_violation * self._normalized_mask(
                    separation_mask.to(separation_violation.dtype)
                )
        if trade_pos_weight is not None:
            trade_w = torch.where(
                trade_mask > 0.5,
                trade_pos_weight.to(device=trade_loss.device, dtype=trade_loss.dtype),
                torch.ones_like(trade_loss),
            )
            trade_loss = trade_loss * trade_w

        loss_terms = self.target_aux_coef * target_loss
        if self.trade_aux_coef > 0.0:
            loss_terms = loss_terms + self.trade_aux_coef * trade_loss
        if recovery_trade_loss is not None:
            loss_terms = loss_terms + self.recovery_trade_coef * recovery_trade_loss
        if mode_loss is not None:
            if trade_pos_weight is not None:
                mode_w = torch.where(
                    trade_mask > 0.5,
                    trade_pos_weight.to(device=mode_loss.device, dtype=mode_loss.dtype),
                    torch.ones_like(mode_loss),
                )
                mode_loss = mode_loss * mode_w
            loss_terms = loss_terms + self.mode_target_coef * mode_loss
        if mode_rate_penalty is not None:
            loss_terms = loss_terms + self.mode_rate_match_coef * mode_rate_penalty
        if mode_regime_rate_penalty is not None:
            loss_terms = loss_terms + self.mode_regime_rate_match_coef * mode_regime_rate_penalty
        if dual_head_anchor_loss is not None:
            loss_terms = loss_terms + self.dual_head_anchor_coef * dual_head_anchor_loss
        if dual_head_underweight_loss is not None:
            loss_terms = loss_terms + self.dual_head_underweight_coef * dual_head_underweight_loss
        if dual_head_separation_loss is not None:
            loss_terms = loss_terms + self.dual_head_separation_coef * dual_head_separation_loss
        if target_dist_penalty is not None:
            loss_terms = loss_terms + self.target_dist_match_coef * target_dist_penalty
        if position_mean_penalty is not None:
            loss_terms = loss_terms + self.position_mean_match_coef * position_mean_penalty
        if target_regime_penalty is not None:
            loss_terms = loss_terms + self.target_regime_dist_match_coef * target_regime_penalty
        if short_mass_penalty is not None:
            loss_terms = loss_terms + self.short_mass_match_coef * short_mass_penalty
        if support_prior_penalty is not None:
            loss_terms = loss_terms + self.support_prior_coef * support_prior_penalty
        if recovery_band_loss is not None:
            loss_terms = loss_terms + self.recovery_band_coef * recovery_band_loss

        if self.band_aux_coef > 0.0:
            trade_margin = 0.05
            hold_band_min = 0.05
            trade_penalty = F.softplus(band_width - (target_gap - trade_margin).clamp(min=0.0))
            if self.band_aux_trade_only:
                band_mask = trade_mask
                if self.direct_band_gap_min > 0.0:
                    band_mask = band_mask * (target_gap >= self.direct_band_gap_min).to(trade_mask.dtype)
                band_penalty = trade_penalty * self._normalized_mask(band_mask)
            else:
                hold_penalty = F.softplus(hold_band_min - band_width)
                band_penalty = torch.where(trade_mask > 0.5, trade_penalty, hold_penalty)
            if trade_pos_weight is not None:
                band_sample_w = torch.where(
                    trade_mask > 0.5,
                    trade_pos_weight.to(device=band_penalty.device, dtype=band_penalty.dtype),
                    torch.ones_like(band_penalty),
                )
                band_penalty = band_penalty * band_sample_w
            loss_terms = loss_terms + self.band_aux_coef * band_penalty

        if self.direct_band_target_coef > 0.0:
            min_band = float(getattr(self.actor, "min_band", 0.02))
            band_cap = min_band + float(getattr(self.actor, "max_band", 0.20))
            trade_band_target = (target_gap - self.direct_band_margin).clamp(min=min_band, max=band_cap)
            if self.direct_band_trade_only:
                band_targets = trade_band_target
                band_target_mask = trade_mask
                if self.direct_band_gap_min > 0.0:
                    band_target_mask = band_target_mask * (target_gap >= self.direct_band_gap_min).to(trade_mask.dtype)
            else:
                hold_band_target = (target_gap + self.direct_hold_band_margin).clamp(min=min_band, max=band_cap)
                band_targets = torch.where(trade_mask > 0.5, trade_band_target, hold_band_target)
                band_target_mask = torch.ones_like(trade_mask)
            band_target_loss = F.smooth_l1_loss(band_width, band_targets, reduction="none")
            band_target_loss = band_target_loss * self._normalized_mask(band_target_mask)
            if trade_pos_weight is not None:
                band_target_w = torch.where(
                    trade_mask > 0.5,
                    trade_pos_weight.to(device=band_target_loss.device, dtype=band_target_loss.dtype),
                    torch.ones_like(band_target_loss),
                )
                band_target_loss = band_target_loss * band_target_w
            loss_terms = loss_terms + self.direct_band_target_coef * band_target_loss

        if self.execution_aux_coef > 0.0:
            execution_signal = trade_pred
            if self.actor._use_separate_execution_head():
                execution_logits = self.actor.execution_logits(
                    z,
                    h,
                    inventory=inventory,
                    regime=regime,
                    advantage=advantage,
                )
                execution_signal = torch.sigmoid(execution_logits)
            predicted_next_inventory = self.actor.soft_execute_controller(
                trade_signal=execution_signal,
                target_inventory=target_mean,
                band_width=band_width,
                current_inventory=current_inventory,
            )
            execution_loss = F.smooth_l1_loss(
                predicted_next_inventory,
                oracle_target,
                reduction="none",
            )
            if recovery_mask_f is not None and self.recovery_execution_coef > 0.0:
                execution_loss = execution_loss + self.recovery_execution_coef * execution_loss * recovery_mask_f
            if trade_pos_weight is not None:
                exec_w = torch.where(
                    trade_mask > 0.5,
                    trade_pos_weight.to(device=execution_loss.device, dtype=execution_loss.dtype),
                    torch.ones_like(execution_loss),
                )
                execution_loss = execution_loss * exec_w
            loss_terms = loss_terms + self.execution_aux_coef * execution_loss

        if sample_quality is not None and self.sample_quality_coef > 0.0:
            quality_weights = sample_quality.to(device=loss_terms.device, dtype=loss_terms.dtype)
            quality_weights = 1.0 + self.sample_quality_coef * quality_weights.clamp(
                min=0.0,
                max=self.sample_quality_clip,
            )
            weights = quality_weights if weights is None else weights * quality_weights

        if weights is not None:
            loss = (weights * loss_terms).mean()
        else:
            loss = loss_terms.mean()

        return loss

    def _bc_path_loss(
        self,
        z_seq: torch.Tensor,
        h_seq: torch.Tensor,
        oracle_positions_seq: torch.Tensor,
        inventory0: torch.Tensor,
        regime_seq: Optional[torch.Tensor] = None,
        advantage_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        horizon = z_seq.shape[1]
        if horizon <= 0:
            return torch.tensor(0.0, device=self.device)

        state = inventory0
        pos_losses = []
        turnover_losses = []
        shortfall_losses = []
        benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
        prev_pred_abs = inventory0[:, 0] + benchmark_position
        prev_oracle_abs = inventory0[:, 0] + benchmark_position
        for t in range(horizon):
            reg_t = regime_seq[:, t] if regime_seq is not None else None
            trade_logits, target_mean, _target_std, band_width, current_inventory = self.actor.controller_outputs(
                z_seq[:, t],
                h_seq[:, t],
                inventory=state,
                regime=reg_t,
                advantage=advantage_seq[:, t] if advantage_seq is not None else None,
            )
            execution_signal = torch.sigmoid(trade_logits)
            if self.actor._use_separate_execution_head():
                execution_logits = self.actor.execution_logits(
                    z_seq[:, t],
                    h_seq[:, t],
                    inventory=state,
                    regime=reg_t,
                    advantage=advantage_seq[:, t] if advantage_seq is not None else None,
                )
                execution_signal = torch.sigmoid(execution_logits)
            next_inventory = self.actor.soft_execute_controller(
                trade_signal=execution_signal,
                target_inventory=target_mean,
                band_width=band_width,
                current_inventory=current_inventory,
            )
            pred_abs_position = next_inventory + benchmark_position
            step_loss = F.smooth_l1_loss(
                pred_abs_position,
                oracle_positions_seq[:, t],
                reduction="none",
            )
            pos_losses.append(step_loss)
            if self.path_turnover_coef > 0.0:
                pred_delta = pred_abs_position - prev_pred_abs
                oracle_delta = oracle_positions_seq[:, t] - prev_oracle_abs
                excess_turnover = F.relu(torch.abs(pred_delta) - torch.abs(oracle_delta))
                turnover_losses.append(excess_turnover)
                prev_pred_abs = pred_abs_position
                prev_oracle_abs = oracle_positions_seq[:, t]
            if self.path_shortfall_coef > 0.0:
                oracle_shortfall = F.relu(benchmark_position - oracle_positions_seq[:, t])
                pred_shortfall = F.relu(benchmark_position - pred_abs_position)
                shortfall_losses.append(F.relu(pred_shortfall - oracle_shortfall))
            state = self.actor.update_controller_state(state, pred_abs_position.unsqueeze(-1))

        loss = self.path_position_coef * torch.stack(pos_losses, dim=0).mean()
        if turnover_losses:
            loss = loss + self.path_turnover_coef * torch.stack(turnover_losses, dim=0).mean()
        if shortfall_losses:
            loss = loss + self.path_shortfall_coef * torch.stack(shortfall_losses, dim=0).mean()
        return loss

    def train(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_positions: np.ndarray,
        verbose: bool = True,
        regime_probs: "np.ndarray | None" = None,
        soft_labels: "np.ndarray | None" = None,
        sample_quality: "np.ndarray | None" = None,
        advantage_values: "np.ndarray | None" = None,
    ) -> list[dict]:
        """BC 事前学習を実行する.

        chunk_size > 1 の場合は Action Chunking を適用する。
        各チャンクの先頭ステップの (z, h) で k ステップ分の oracle 行動を予測し、
        損失を k ステップの平均とすることで高頻度ラベルノイズを平滑化する。

        Args:
            z: (T, z_dim) エンコードされた潜在
            h: (T, h_dim) Transformer hidden
            oracle_positions: (T,) oracle ポジション比率
            regime_probs: (T, regime_dim) レジーム確率ベクトル（省略可）
            soft_labels: (T, K) Boltzmann soft target（省略可）

        Returns:
            各エポックのロスログ
        """
        T = min(len(z), len(h), len(oracle_positions))
        benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
        teacher_state_all = self.actor.controller_states_from_positions(oracle_positions[:T])
        inv_all = teacher_state_all[:, 0]
        trade_mask = (np.abs((oracle_positions[:T] - benchmark_position) - inv_all[:T]) > 1e-8).astype(np.float32)
        n_pos = float(trade_mask.sum())
        n_neg = float(T - n_pos)
        trade_pos_weight_t = None
        if n_pos > 0 and n_neg > 0:
            trade_pos_weight_t = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=self.device)
        class_weights_t = None
        if self.class_balanced:
            oracle_pos_t = torch.tensor(oracle_positions[:T], dtype=torch.float32)
            target_idx_all = self.actor.target_indices(oracle_pos_t).to(dtype=torch.long)
            class_counts = torch.bincount(target_idx_all, minlength=self.actor.act_dim).to(dtype=torch.float32)
            class_weights_t = class_counts.sum() / class_counts.clamp_min(1.0)
            class_weights_t = class_weights_t / class_weights_t.mean().clamp_min(1e-8)
            class_weights_t = class_weights_t.to(self.device)

        # --- Action Chunking: データをチャンク単位に再構築 ---
        k = self.chunk_size
        if k > 1:
            logs = []
            rollout_positions = None
            use_regime = regime_probs is not None
            use_soft = soft_labels is not None
            use_adv = advantage_values is not None
            for epoch in range(self.n_epochs):
                state_all = teacher_state_all
                target_all = np.asarray(oracle_positions[:T], dtype=np.float32)
                if self.self_condition_prob > 0.0 and epoch >= self.self_condition_warmup_epochs:
                    if rollout_positions is None or (epoch - self.self_condition_warmup_epochs) % self.self_condition_interval == 0:
                        rollout_positions = None
                    state_all, rollout_positions, target_all = self._mixed_controller_states(
                        z[:T],
                        h[:T],
                        oracle_positions[:T],
                        regime_probs=regime_probs[:T] if use_regime else None,
                        rollout_positions=rollout_positions,
                    )

                n_chunks = T // k
                T_use = n_chunks * k
                z_arr = z[:T_use].reshape(n_chunks, k, -1)[:, 0, :]
                h_arr = h[:T_use].reshape(n_chunks, k, -1)[:, 0, :]
                pos_arr = target_all[:T_use].reshape(n_chunks, k)
                state_arr = state_all[:T_use].reshape(n_chunks, k, -1)
                cur_abs_arr = state_arr[:, 0, 0] + benchmark_position
                switch_mask = np.abs(pos_arr - cur_abs_arr[:, None]) > 1e-8
                has_switch = switch_mask.any(axis=1)
                first_switch_idx = switch_mask.argmax(axis=1)
                repr_idx = np.where(has_switch, first_switch_idx, 0)
                repr_pos_arr = pos_arr[np.arange(n_chunks), repr_idx]
                repr_state_arr = state_arr[:, 0, :]

                tensors = [
                    torch.tensor(z_arr, dtype=torch.float32),
                    torch.tensor(h_arr, dtype=torch.float32),
                    torch.tensor(repr_pos_arr, dtype=torch.float32),
                    torch.tensor(repr_state_arr, dtype=torch.float32),
                ]
                if use_regime:
                    r_arr = regime_probs[:T_use].reshape(n_chunks, k, -1)[:, 0, :]
                    tensors.append(torch.tensor(r_arr, dtype=torch.float32))
                if use_soft:
                    sl_arr = soft_labels[:T_use].reshape(n_chunks, k, -1)
                    sl_repr_arr = sl_arr[np.arange(n_chunks), repr_idx]
                    tensors.append(torch.tensor(sl_repr_arr, dtype=torch.float32))
                if sample_quality is not None:
                    q_arr = np.asarray(sample_quality[:T_use], dtype=np.float32).reshape(n_chunks, k)
                    q_repr_arr = q_arr[np.arange(n_chunks), repr_idx]
                    tensors.append(torch.tensor(q_repr_arr, dtype=torch.float32))
                if use_adv:
                    adv_arr = np.asarray(advantage_values[:T_use], dtype=np.float32).reshape(n_chunks, k)
                    adv_repr_arr = adv_arr[np.arange(n_chunks), repr_idx]
                    tensors.append(torch.tensor(adv_repr_arr, dtype=torch.float32))

                dataset = TensorDataset(*tensors)
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                epoch_loss = 0.0
                count = 0
                for batch in loader:
                    z_b = batch[0].to(self.device)       # (B, z_dim)
                    h_b = batch[1].to(self.device)       # (B, h_dim)
                    a_repr = batch[2].to(self.device)    # (B,)
                    inv_now = batch[3].to(self.device)   # (B, S)
                    bi = 4
                    reg_b = batch[bi].to(self.device) if use_regime else None
                    if use_regime:
                        bi += 1
                    sl_repr = batch[bi].to(self.device) if use_soft else None
                    if use_soft:
                        bi += 1
                    q_repr = batch[bi].to(self.device) if sample_quality is not None else None
                    if sample_quality is not None:
                        bi += 1
                    adv_repr = batch[bi].to(self.device) if use_adv else None

                    if self.use_sirl:
                        state = torch.cat([z_b, h_b], dim=-1)
                        sirl_w = self.weight_net(state)
                    else:
                        sirl_w = None

                    # chunk 先頭 state から「最初に起こる実 switch」を学ばせる。
                    loss = self._bc_loss(
                        z_b, h_b, a_repr,
                        inventory=inv_now,
                        weights=sirl_w,
                        regime=reg_b,
                        soft_labels=sl_repr,
                        class_weights=class_weights_t,
                        trade_pos_weight=trade_pos_weight_t,
                        sample_quality=q_repr,
                        advantage=adv_repr,
                    )

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

        # --- 通常の per-step 学習（chunk_size=1）---
        use_regime = regime_probs is not None
        use_soft = soft_labels is not None
        use_adv = advantage_values is not None
        logs = []
        rollout_positions = None
        for epoch in range(self.n_epochs):
            state_all = teacher_state_all
            target_all = np.asarray(oracle_positions[:T], dtype=np.float32)
            if self.self_condition_prob > 0.0 and epoch >= self.self_condition_warmup_epochs:
                if rollout_positions is None or (epoch - self.self_condition_warmup_epochs) % self.self_condition_interval == 0:
                    rollout_positions = None
                state_all, rollout_positions, target_all = self._mixed_controller_states(
                    z[:T],
                    h[:T],
                    oracle_positions[:T],
                    regime_probs=regime_probs[:T] if use_regime else None,
                    rollout_positions=rollout_positions,
                )

            z_t = torch.tensor(z[:T], dtype=torch.float32)
            h_t = torch.tensor(h[:T], dtype=torch.float32)
            a_t = torch.tensor(target_all[:T], dtype=torch.float32)
            inv_t = torch.tensor(state_all[:T], dtype=torch.float32)

            regime_t = torch.tensor(regime_probs[:T], dtype=torch.float32) if use_regime else None
            soft_t = torch.tensor(soft_labels[:T], dtype=torch.float32) if use_soft else None
            quality_t = torch.tensor(sample_quality[:T], dtype=torch.float32) if sample_quality is not None else None
            adv_t = torch.tensor(advantage_values[:T], dtype=torch.float32) if use_adv else None
            if self.path_aux_coef > 0.0 and self.path_horizon > 1:
                dataset = ControllerPathDataset(
                    z_t,
                    h_t,
                    a_t,
                    inv_t,
                    horizon=self.path_horizon,
                    regime=regime_t,
                    soft_labels=soft_t,
                    sample_quality=quality_t,
                    advantage=adv_t,
                )
            else:
                tensors = [z_t, h_t, a_t, inv_t]
                if use_regime:
                    tensors.append(regime_t)
                if use_soft:
                    tensors.append(soft_t)
                if quality_t is not None:
                    tensors.append(quality_t)
                if adv_t is not None:
                    tensors.append(adv_t)
                dataset = TensorDataset(*tensors)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            epoch_loss = 0.0
            count = 0

            for batch in loader:
                if self.path_aux_coef > 0.0 and self.path_horizon > 1:
                    z_seq, h_seq, a_seq, inv_b = batch[0], batch[1], batch[2], batch[3]
                    z_b, h_b, a_b = z_seq[:, 0], h_seq[:, 0], a_seq[:, 0]
                else:
                    z_b, h_b, a_b, inv_b = batch[0], batch[1], batch[2], batch[3]
                bi = 4
                reg_b = batch[bi].to(self.device) if use_regime else None
                if use_regime:
                    if self.path_aux_coef > 0.0 and self.path_horizon > 1:
                        reg_b = reg_b[:, 0]
                    bi += 1
                sl_b = batch[bi].to(self.device) if use_soft else None
                if use_soft and self.path_aux_coef > 0.0 and self.path_horizon > 1:
                    sl_b = sl_b[:, 0]
                if use_soft:
                    bi += 1
                q_b = batch[bi].to(self.device) if sample_quality is not None else None
                if q_b is not None and self.path_aux_coef > 0.0 and self.path_horizon > 1:
                    q_b = q_b[:, 0]
                if sample_quality is not None:
                    bi += 1
                adv_b = batch[bi].to(self.device) if use_adv else None
                if adv_b is not None and self.path_aux_coef > 0.0 and self.path_horizon > 1:
                    adv_b = adv_b[:, 0]

                z_b = z_b.to(self.device)
                h_b = h_b.to(self.device)
                a_b = a_b.to(self.device)
                inv_b = inv_b.to(self.device)

                if self.use_sirl:
                    state = torch.cat([z_b, h_b], dim=-1)
                    weights = self.weight_net(state)
                else:
                    weights = None

                loss = self._bc_loss(
                    z_b, h_b, a_b,
                    inventory=inv_b,
                    weights=weights,
                    regime=reg_b,
                    soft_labels=sl_b,
                    class_weights=class_weights_t,
                    trade_pos_weight=trade_pos_weight_t,
                    sample_quality=q_b,
                    advantage=adv_b,
                )
                if self.path_aux_coef > 0.0 and self.path_horizon > 1:
                    z_seq = z_seq.to(self.device)
                    h_seq = h_seq.to(self.device)
                    a_seq = a_seq.to(self.device)
                    reg_seq = batch[4].to(self.device) if use_regime else None
                    adv_seq_idx = 4 + (1 if use_regime else 0) + (1 if use_soft else 0) + (1 if sample_quality is not None else 0)
                    adv_seq = batch[adv_seq_idx].to(self.device) if use_adv else None
                    path_loss = self._bc_path_loss(
                        z_seq=z_seq,
                        h_seq=h_seq,
                        oracle_positions_seq=a_seq,
                        inventory0=inv_b,
                        regime_seq=reg_seq,
                        advantage_seq=adv_seq,
                    )
                    loss = loss + self.path_aux_coef * path_loss

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
        incompatible = self.actor.load_state_dict(ckpt["actor"], strict=False)
        optional_missing = {
            "execution_head.weight",
            "execution_head.bias",
            "residual_head_a.weight",
            "residual_head_a.bias",
            "residual_head_b.weight",
            "residual_head_b.bias",
        }
        missing = [key for key in incompatible.missing_keys if key not in optional_missing]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"BC checkpoint incompatibility while loading {path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self.use_sirl and "weight_net" in ckpt:
            self.weight_net.load_state_dict(ckpt["weight_net"])
