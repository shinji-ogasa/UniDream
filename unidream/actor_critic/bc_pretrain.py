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
    ):
        self.z = z
        self.h = h
        self.oracle_positions = oracle_positions
        self.inventory = inventory
        self.horizon = max(1, int(horizon))
        self.regime = regime
        self.soft_labels = soft_labels
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
        device: str = "cpu",
    ):
        self.actor = actor
        self.device = torch.device(device)
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

        # SIRL 重みネット
        self.use_sirl = sirl_hidden > 0
        if self.use_sirl:
            self.weight_net = SIRLWeightNet(z_dim + h_dim, sirl_hidden).to(self.device)
            params = list(actor.parameters()) + list(self.weight_net.parameters())
        else:
            self.weight_net = None
            params = list(actor.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _mixed_controller_states(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_positions: np.ndarray,
        regime_probs: np.ndarray | None = None,
        rollout_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        teacher_states = self.actor.controller_states_from_positions(oracle_positions)
        if self.self_condition_prob <= 0.0:
            return teacher_states, rollout_positions

        if rollout_positions is None:
            rollout_positions = self.actor.predict_positions(
                z,
                h,
                regime_np=regime_probs,
                device=str(self.device),
            )
        rollout_states = self.actor.controller_states_from_positions(rollout_positions)
        mask = np.random.rand(len(teacher_states)) < self.self_condition_prob
        mixed_states = teacher_states.copy()
        mixed_states[mask] = rollout_states[mask]
        return mixed_states, rollout_positions

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
    ) -> torch.Tensor:
        """Inventory controller 向けの BC 損失."""
        trade_logits, target_logits, target_mean, target_std, band_width, current_inventory = (
            self.actor.controller_outputs_full(
            z, h, inventory=inventory, regime=regime
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
        target_indices = self.actor.target_indices(oracle_positions).to(dtype=torch.long)
        if soft_labels is not None and soft_labels.shape[-1] == target_logits.shape[-1]:
            log_probs = F.log_softmax(target_logits, dim=-1)
            target_loss = -(soft_labels.to(log_probs.dtype) * log_probs).sum(dim=-1)
        else:
            target_loss = F.cross_entropy(
                target_logits,
                target_indices,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
        target_reg_loss = F.smooth_l1_loss(target_mean, oracle_target, reduction="none")
        target_loss = target_loss + 0.25 * target_reg_loss
        if class_weights is not None:
            sample_class_w = class_weights.to(
                device=target_loss.device,
                dtype=target_loss.dtype,
            )[target_indices]
            target_loss = target_loss * sample_class_w
        if trade_pos_weight is not None:
            target_w = torch.where(
                trade_mask > 0.5,
                trade_pos_weight.to(device=target_loss.device, dtype=target_loss.dtype),
                torch.ones_like(target_loss),
            )
            target_loss = target_loss * target_w

        trade_pred = torch.sigmoid(trade_logits)
        trade_loss = F.smooth_l1_loss(trade_pred, trade_targets, reduction="none")
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

        if self.band_aux_coef > 0.0:
            trade_margin = 0.05
            hold_band_min = 0.05
            trade_penalty = F.softplus(band_width - (target_gap - trade_margin).clamp(min=0.0))
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

        if self.execution_aux_coef > 0.0:
            predicted_next_inventory = self.actor.soft_execute_controller(
                trade_signal=torch.sigmoid(trade_logits),
                target_inventory=target_mean,
                band_width=band_width,
                current_inventory=current_inventory,
            )
            execution_loss = F.smooth_l1_loss(
                predicted_next_inventory,
                oracle_target,
                reduction="none",
            )
            if trade_pos_weight is not None:
                exec_w = torch.where(
                    trade_mask > 0.5,
                    trade_pos_weight.to(device=execution_loss.device, dtype=execution_loss.dtype),
                    torch.ones_like(execution_loss),
                )
                execution_loss = execution_loss * exec_w
            loss_terms = loss_terms + self.execution_aux_coef * execution_loss

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
            )
            next_inventory = self.actor.soft_execute_controller(
                trade_signal=torch.sigmoid(trade_logits),
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
                shortfall_losses.append(F.relu(benchmark_position - pred_abs_position))
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
            for epoch in range(self.n_epochs):
                state_all = teacher_state_all
                if self.self_condition_prob > 0.0 and epoch >= self.self_condition_warmup_epochs:
                    if rollout_positions is None or (epoch - self.self_condition_warmup_epochs) % self.self_condition_interval == 0:
                        rollout_positions = None
                    state_all, rollout_positions = self._mixed_controller_states(
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
                pos_arr = oracle_positions[:T_use].reshape(n_chunks, k)
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
        logs = []
        rollout_positions = None
        for epoch in range(self.n_epochs):
            state_all = teacher_state_all
            if self.self_condition_prob > 0.0 and epoch >= self.self_condition_warmup_epochs:
                if rollout_positions is None or (epoch - self.self_condition_warmup_epochs) % self.self_condition_interval == 0:
                    rollout_positions = None
                state_all, rollout_positions = self._mixed_controller_states(
                    z[:T],
                    h[:T],
                    oracle_positions[:T],
                    regime_probs=regime_probs[:T] if use_regime else None,
                    rollout_positions=rollout_positions,
                )

            z_t = torch.tensor(z[:T], dtype=torch.float32)
            h_t = torch.tensor(h[:T], dtype=torch.float32)
            a_t = torch.tensor(oracle_positions[:T], dtype=torch.float32)
            inv_t = torch.tensor(state_all[:T], dtype=torch.float32)

            regime_t = torch.tensor(regime_probs[:T], dtype=torch.float32) if use_regime else None
            soft_t = torch.tensor(soft_labels[:T], dtype=torch.float32) if use_soft else None
            if self.path_aux_coef > 0.0 and self.path_horizon > 1:
                dataset = ControllerPathDataset(
                    z_t,
                    h_t,
                    a_t,
                    inv_t,
                    horizon=self.path_horizon,
                    regime=regime_t,
                    soft_labels=soft_t,
                )
            else:
                tensors = [z_t, h_t, a_t, inv_t]
                if use_regime:
                    tensors.append(regime_t)
                if use_soft:
                    tensors.append(soft_t)
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
                )
                if self.path_aux_coef > 0.0 and self.path_horizon > 1:
                    z_seq = z_seq.to(self.device)
                    h_seq = h_seq.to(self.device)
                    a_seq = a_seq.to(self.device)
                    reg_seq = batch[4].to(self.device) if use_regime else None
                    path_loss = self._bc_path_loss(
                        z_seq=z_seq,
                        h_seq=h_seq,
                        oracle_positions_seq=a_seq,
                        inventory0=inv_b,
                        regime_seq=reg_seq,
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
        self.actor.load_state_dict(ckpt["actor"])
        if self.use_sirl and "weight_net" in ckpt:
            self.weight_net.load_state_dict(ckpt["weight_net"])
