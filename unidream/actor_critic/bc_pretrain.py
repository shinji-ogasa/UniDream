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
from torch.utils.data import DataLoader, TensorDataset

from unidream.actor_critic.actor import Actor
from unidream.data.oracle import ACTIONS

_ACTIONS_T = torch.tensor(ACTIONS, dtype=torch.float32)


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

        # SIRL 重みネット
        self.use_sirl = sirl_hidden > 0
        if self.use_sirl:
            self.weight_net = SIRLWeightNet(z_dim + h_dim, sirl_hidden).to(self.device)
            params = list(actor.parameters()) + list(self.weight_net.parameters())
        else:
            self.weight_net = None
            params = list(actor.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _bc_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        oracle_actions: torch.Tensor,
        inventory: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        soft_labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        trade_pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inventory controller 向けの BC 損失."""
        trade_logits, target_mean, target_std, band_width, current_inventory = self.actor.controller_outputs(
            z, h, inventory=inventory, regime=regime
        )
        benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
        abs_min = float(getattr(self.actor, "abs_min_position", -1.0))
        abs_max = float(getattr(self.actor, "abs_max_position", 1.0))
        oracle_position = _ACTIONS_T.to(device=current_inventory.device, dtype=current_inventory.dtype)[oracle_actions]
        oracle_position = oracle_position.clamp(min=abs_min, max=abs_max)
        oracle_target = oracle_position - benchmark_position
        target_gap = torch.abs(oracle_target - current_inventory)
        trade_targets = (target_gap > 1e-8).float()

        if soft_labels is not None:
            soft_actions = _ACTIONS_T.to(device=soft_labels.device, dtype=soft_labels.dtype)
            oracle_target_soft = (soft_labels * soft_actions.unsqueeze(0)).sum(dim=-1) - benchmark_position
            target_loss = F.smooth_l1_loss(target_mean, oracle_target_soft, reduction="none")
        else:
            target_loss = F.smooth_l1_loss(target_mean, oracle_target, reduction="none")

        if self.label_smoothing > 0.0:
            target_loss = target_loss + self.label_smoothing * target_std

        if class_weights is not None:
            sample_class_w = class_weights[oracle_actions]
            weights = weights * sample_class_w if weights is not None else sample_class_w
        if trade_pos_weight is not None:
            target_w = torch.where(
                trade_targets > 0.5,
                trade_pos_weight.to(device=target_loss.device, dtype=target_loss.dtype),
                torch.ones_like(target_loss),
            )
            target_loss = target_loss * target_w

        trade_loss = F.binary_cross_entropy_with_logits(
            trade_logits,
            trade_targets,
            reduction="none",
            pos_weight=trade_pos_weight,
        )

        loss_terms = self.target_aux_coef * target_loss
        if self.trade_aux_coef > 0.0:
            loss_terms = loss_terms + self.trade_aux_coef * trade_loss

        if self.band_aux_coef > 0.0:
            trade_margin = 0.05
            hold_band_min = 0.05
            trade_penalty = F.softplus(band_width - (target_gap - trade_margin).clamp(min=0.0))
            hold_penalty = F.softplus(hold_band_min - band_width)
            band_penalty = torch.where(trade_targets > 0.5, trade_penalty, hold_penalty)
            if trade_pos_weight is not None:
                band_sample_w = torch.where(
                    trade_targets > 0.5,
                    trade_pos_weight.to(device=band_penalty.device, dtype=band_penalty.dtype),
                    torch.ones_like(band_penalty),
                )
                band_penalty = band_penalty * band_sample_w
            loss_terms = loss_terms + self.band_aux_coef * band_penalty

        if weights is not None:
            loss = (weights * loss_terms).mean()
        else:
            loss = loss_terms.mean()

        return loss

    def train(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_actions: np.ndarray,
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
            oracle_actions: (T,) oracle 行動インデックス
            regime_probs: (T, regime_dim) レジーム確率ベクトル（省略可）
            soft_labels: (T, K) Boltzmann soft target（省略可）

        Returns:
            各エポックのロスログ
        """
        T = min(len(z), len(h), len(oracle_actions))
        benchmark_position = float(getattr(self.actor, "benchmark_position", 0.0))
        abs_min = float(getattr(self.actor, "abs_min_position", -1.0))
        abs_max = float(getattr(self.actor, "abs_max_position", 1.0))
        clipped_actions = np.clip(ACTIONS[oracle_actions[:T]], abs_min, abs_max)
        inv_all = np.zeros(T, dtype=np.float32)
        if T > 1:
            inv_all[1:] = clipped_actions[:T - 1] - benchmark_position
        trade_mask = (np.abs((clipped_actions - benchmark_position) - inv_all[:T]) > 1e-8).astype(np.float32)
        n_pos = float(trade_mask.sum())
        n_neg = float(T - n_pos)
        trade_pos_weight_t = None
        if n_pos > 0 and n_neg > 0:
            trade_pos_weight_t = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=self.device)

        # --- Action Chunking: データをチャンク単位に再構築 ---
        k = self.chunk_size
        if k > 1:
            n_chunks = T // k
            T_use = n_chunks * k

            # チャンク先頭の (z, h, regime) を取得
            z_arr = z[:T_use].reshape(n_chunks, k, -1)[:, 0, :]         # (n_chunks, z_dim)
            h_arr = h[:T_use].reshape(n_chunks, k, -1)[:, 0, :]         # (n_chunks, h_dim)
            a_arr = oracle_actions[:T_use].reshape(n_chunks, k)          # (n_chunks, k)
            inv_arr = inv_all[:T_use].reshape(n_chunks, k)               # (n_chunks, k)

            tensors = [
                torch.tensor(z_arr, dtype=torch.float32),
                torch.tensor(h_arr, dtype=torch.float32),
                torch.tensor(a_arr, dtype=torch.long),
                torch.tensor(inv_arr, dtype=torch.float32),
            ]
            use_regime = regime_probs is not None
            use_soft = soft_labels is not None
            if use_regime:
                r_arr = regime_probs[:T_use].reshape(n_chunks, k, -1)[:, 0, :]
                tensors.append(torch.tensor(r_arr, dtype=torch.float32))
            if use_soft:
                sl_arr = soft_labels[:T_use].reshape(n_chunks, k, -1)   # (n_chunks, k, K)
                tensors.append(torch.tensor(sl_arr, dtype=torch.float32))

            dataset = TensorDataset(*tensors)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            logs = []
            for epoch in range(self.n_epochs):
                epoch_loss = 0.0
                count = 0
                for batch in loader:
                    z_b = batch[0].to(self.device)       # (B, z_dim)
                    h_b = batch[1].to(self.device)       # (B, h_dim)
                    a_chunk = batch[2].to(self.device)   # (B, k)
                    inv_chunk = batch[3].to(self.device) # (B, k)
                    bi = 4
                    reg_b = batch[bi].to(self.device) if use_regime else None
                    if use_regime:
                        bi += 1
                    sl_chunk = batch[bi].to(self.device) if use_soft else None  # (B, k, K) or None

                    if self.use_sirl:
                        state = torch.cat([z_b, h_b], dim=-1)
                        sirl_w = self.weight_net(state)
                    else:
                        sirl_w = None

                    # k ステップ分の損失を平均
                    step_losses = []
                    for step in range(k):
                        a_step = a_chunk[:, step]        # (B,)
                        inv_step = inv_chunk[:, step]
                        sl_step = sl_chunk[:, step, :] if sl_chunk is not None else None
                        step_loss = self._bc_loss(
                            z_b, h_b, a_step,
                            inventory=inv_step,
                            weights=sirl_w,
                            regime=reg_b,
                            soft_labels=sl_step,
                            trade_pos_weight=trade_pos_weight_t,
                        )
                        step_losses.append(step_loss)
                    loss = torch.stack(step_losses).mean()

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
        z_t = torch.tensor(z[:T], dtype=torch.float32)
        h_t = torch.tensor(h[:T], dtype=torch.float32)
        a_t = torch.tensor(oracle_actions[:T], dtype=torch.long)
        inv_t = torch.tensor(inv_all[:T], dtype=torch.float32)

        use_regime = regime_probs is not None
        use_soft = soft_labels is not None
        tensors = [z_t, h_t, a_t, inv_t]
        if use_regime:
            tensors.append(torch.tensor(regime_probs[:T], dtype=torch.float32))
        if use_soft:
            tensors.append(torch.tensor(soft_labels[:T], dtype=torch.float32))
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logs = []
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            count = 0

            for batch in loader:
                z_b, h_b, a_b, inv_b = batch[0], batch[1], batch[2], batch[3]
                bi = 4
                reg_b = batch[bi].to(self.device) if use_regime else None
                if use_regime:
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
