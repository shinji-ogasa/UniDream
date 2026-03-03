"""Imagination Actor-Critic 学習モジュール.

DreamerV3 ベースの Imagination AC + BC 損失減衰混合。

アルゴリズム:
  1. 実軌跡からの世界モデル学習（train_wm.py で実施済みを前提）
  2. 現在の観測から z, h をエンコード
  3. Imagination: Actor が z_t を入力に行動 a_t を選択
     → 世界モデルが next_z_{t+1}, r_{t+1}, done_{t+1} を予測
     → horizon=3 まで繰り返す
  4. λ-return（symlog 空間）で advantage を計算
  5. Actor loss: α·BC_loss + (1-α)·AC_loss（α は 1→0 線形減衰）
  6. Critic loss: twohot cross-entropy
  7. TD3+BC 的な保守的制約を Actor loss に付加

References:
    DreamerV3 Actor-Critic (ICLR 2023)
    TD3+BC: https://arxiv.org/abs/2106.06860
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from unidream.actor_critic.actor import Actor
from unidream.actor_critic.critic import Critic, RewardEMANorm
from unidream.world_model.ensemble import EnsembleWorldModel
from unidream.world_model.transformer import symlog, symexp, twohot_decode


class ImagACTrainer:
    """Imagination Actor-Critic 学習ループ.

    Args:
        actor: Actor モジュール
        critic: Critic モジュール
        ensemble: 学習済み EnsembleWorldModel（imagination に使用）
        cfg: config 辞書
        device: 計算デバイス
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        ensemble: EnsembleWorldModel,
        cfg: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.actor = actor
        self.critic = critic
        self.ensemble = ensemble
        self.device = torch.device(device)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.ensemble.to(self.device)
        self.ensemble.eval()  # WM は固定

        cfg = cfg or {}
        ac_cfg = cfg.get("ac", {})

        self.horizon = ac_cfg.get("horizon", 3)
        self.lam = ac_cfg.get("lam", 0.95)
        self.gamma = ac_cfg.get("gamma", 0.99)
        self.entropy_scale = ac_cfg.get("entropy_scale", 3e-4)
        self.td3bc_alpha = ac_cfg.get("td3bc_alpha", 2.5)
        self.alpha_init = ac_cfg.get("alpha_init", 1.0)
        self.alpha_final = ac_cfg.get("alpha_final", 0.0)
        self.alpha_decay_steps = ac_cfg.get("alpha_decay_steps", 50_000)
        self.max_steps = ac_cfg.get("max_steps", 200_000)
        self.grad_clip = ac_cfg.get("grad_clip", 100.0)
        self.log_interval = cfg.get("logging", {}).get("log_interval", 1000)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=ac_cfg.get("actor_lr", 3e-5)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=ac_cfg.get("critic_lr", 3e-4)
        )

        self.bins = self.ensemble.get_bins().to(self.device)
        self.reward_ema = RewardEMANorm()
        self.global_step = 0
        self.loss_history: list[dict] = []

        # BC 損失用の oracle データ（bc_pretrain 後に set_oracle_data で設定）
        self._oracle_z: Optional[torch.Tensor] = None
        self._oracle_h: Optional[torch.Tensor] = None
        self._oracle_actions: Optional[torch.Tensor] = None

    def set_oracle_data(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_actions: np.ndarray,
    ) -> None:
        """BC 損失用の Oracle データを設定する.

        Args:
            z: (T, z_dim)
            h: (T, h_dim)
            oracle_actions: (T,) oracle 行動インデックス
        """
        T = min(len(z), len(h), len(oracle_actions))
        self._oracle_z = torch.tensor(z[:T], dtype=torch.float32, device=self.device)
        self._oracle_h = torch.tensor(h[:T], dtype=torch.float32, device=self.device)
        self._oracle_actions = torch.tensor(oracle_actions[:T], dtype=torch.long, device=self.device)

    def _get_alpha(self) -> float:
        """現在の BC/AC 混合比率 α を返す（線形減衰: 1→0）."""
        t = min(self.global_step, self.alpha_decay_steps)
        return self.alpha_init + (self.alpha_final - self.alpha_init) * (t / self.alpha_decay_steps)

    def _imagination_rollout(
        self,
        z0: torch.Tensor,
        h0: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Imagination rollout を実行する（horizon ステップ）.

        Args:
            z0: (B, z_dim) 初期潜在
            h0: (B, h_dim) 初期 hidden（コンテキスト用）
            past_zs: (B, t, z_dim) 過去の潜在履歴
            past_as: (B, t) 過去の行動履歴

        Returns:
            {zs, hs, actions, log_probs, entropies, rewards, dones}
            各 (B, horizon) または (B, horizon, dim)
        """
        B = z0.shape[0]
        zs, hs, acts, log_probs_list, entropies_list, rewards_list, dones_list = [], [], [], [], [], [], []

        z = z0
        h = h0
        pzs = past_zs
        pas = past_as

        for _ in range(self.horizon):
            # Actor が行動を選択
            action, log_prob, entropy = self.actor.get_action(z, h)

            # Imagination step
            with torch.no_grad():
                result = self.ensemble.imagine_step(z, h, action, pzs, pas)

            zs.append(z)
            hs.append(h)
            acts.append(action)
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)
            rewards_list.append(result["reward"])
            dones_list.append(result["done"])

            z = result["next_z"].detach()
            h = result["next_h"].detach()
            pzs = result["past_zs"]
            pas = result["past_as"]

        return {
            "zs": torch.stack(zs, dim=1),                      # (B, H, z_dim)
            "hs": torch.stack(hs, dim=1),                      # (B, H, h_dim)
            "actions": torch.stack(acts, dim=1),               # (B, H)
            "log_probs": torch.stack(log_probs_list, dim=1),   # (B, H)
            "entropies": torch.stack(entropies_list, dim=1),   # (B, H)
            "rewards": torch.stack(rewards_list, dim=1),       # (B, H)
            "dones": torch.stack(dones_list, dim=1),           # (B, H)
            "last_z": z,
            "last_h": h,
        }

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_value: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """λ-return を計算する（symlog 空間）.

        G_t^λ = r_t + γ(1-d_t)[(1-λ)V(s_{t+1}) + λ G_{t+1}^λ]

        Args:
            rewards: (B, H) 各ステップの報酬（正規化済み）
            values: (B, H) 各ステップの value（symlog 空間）
            last_value: (B,) H+1 ステップ目の value（bootstrap）
            dones: (B, H) 終了フラグ

        Returns:
            returns: (B, H) λ-return（symlog 空間）
        """
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        G = symlog(last_value)

        for t in reversed(range(H)):
            r_t = symlog(rewards[:, t])
            d_t = dones[:, t]
            G = r_t + self.gamma * (1 - d_t) * (
                (1 - self.lam) * values[:, t] + self.lam * G
            )
            returns[:, t] = G

        return returns

    def _bc_loss_batch(self, batch_size: int = 128) -> torch.Tensor:
        """Oracle データからランダムサンプルして BC 損失を計算する."""
        if self._oracle_z is None:
            return torch.tensor(0.0, device=self.device)

        T = self._oracle_z.shape[0]
        idx = torch.randint(0, T, (min(batch_size, T),), device=self.device)
        z_b = self._oracle_z[idx]
        h_b = self._oracle_h[idx]
        a_b = self._oracle_actions[idx]

        dist = self.actor(z_b, h_b)
        return -dist.log_prob(a_b).mean()

    def train_step(
        self,
        z0: torch.Tensor,
        h0: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """1 ステップの Actor-Critic 更新.

        Args:
            z0: (B, z_dim) バッチの初期潜在
            h0: (B, h_dim) バッチの初期 hidden
            past_zs: (B, t, z_dim) コンテキスト潜在列（省略可）
            past_as: (B, t) コンテキスト行動列（省略可）

        Returns:
            {actor_loss, critic_loss, bc_loss, entropy}
        """
        B = z0.shape[0]

        # --- Imagination rollout ---
        rollout = self._imagination_rollout(z0, h0, past_zs, past_as)

        zs = rollout["zs"]            # (B, H, z_dim)
        hs = rollout["hs"]            # (B, H, h_dim)
        rewards = rollout["rewards"]  # (B, H)
        dones = rollout["dones"]      # (B, H)
        log_probs = rollout["log_probs"]
        entropies = rollout["entropies"]
        last_z = rollout["last_z"]
        last_h = rollout["last_h"]

        # --- Reward 正規化（EMA）---
        self.reward_ema.update(rewards)
        rewards_norm = rewards / self.reward_ema.scale

        # --- Critic の value 推定 ---
        # (B, H, z_dim) → flatten して value 計算
        zs_flat = zs.reshape(B * self.horizon, -1)
        hs_flat = hs.reshape(B * self.horizon, -1)
        values_flat = self.critic.slow_value(zs_flat, hs_flat, self.bins)
        values = values_flat.reshape(B, self.horizon)  # symexp 済み

        # bootstrap value
        with torch.no_grad():
            last_val = self.critic.slow_value(last_z, last_h, self.bins)

        # --- λ-return ---
        returns = self._compute_lambda_returns(
            symlog(rewards_norm + 1e-8),  # symlog 空間で計算
            symlog(values + 1e-8),
            last_val,
            dones,
        )

        # --- Critic 損失 ---
        critic_logits_flat = self.critic(zs_flat.detach(), hs_flat.detach())
        critic_logits = critic_logits_flat.reshape(B, self.horizon, -1)

        from unidream.world_model.transformer import twohot_encode
        targets_symlog = returns.detach()
        critic_loss = 0.0
        for t in range(self.horizon):
            target_twohot = twohot_encode(targets_symlog[:, t], self.bins)
            import torch.nn.functional as F
            log_p = F.log_softmax(critic_logits[:, t], dim=-1)
            critic_loss = critic_loss - (target_twohot * log_p).sum(-1).mean()
        critic_loss = critic_loss / self.horizon

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        self.critic.update_slow_target()

        # --- Actor 損失（AC + BC 混合）---
        alpha = self._get_alpha()

        # Advantage = λ-return - value（symlog 空間）
        advantage = (returns - symlog(values + 1e-8)).detach()
        # TD3+BC スタイルの保守的スケーリング
        norm_q = self.td3bc_alpha / (advantage.abs().mean() + 1e-8)

        ac_loss = -(norm_q * advantage * log_probs).mean() - self.entropy_scale * entropies.mean()

        bc_loss = self._bc_loss_batch()

        actor_loss = alpha * bc_loss + (1 - alpha) * ac_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.global_step += 1

        return {
            "actor_loss": actor_loss.item(),
            "ac_loss": ac_loss.item(),
            "bc_loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss,
            "critic_loss": critic_loss.item(),
            "entropy": entropies.mean().item(),
            "alpha": alpha,
            "reward_mean": rewards.mean().item(),
            "reward_scale": self.reward_ema.scale,
        }

    def train(
        self,
        encoded_sequences: list[dict],
        max_steps: Optional[int] = None,
        batch_size: int = 32,
    ) -> list[dict]:
        """学習ループを実行する.

        Args:
            encoded_sequences: [{"z": (T, z_dim), "h": (T, h_dim)}] のリスト
                               世界モデルの encode_sequence 出力
            max_steps: 最大ステップ数
            batch_size: imagination 初期状態のバッチサイズ

        Returns:
            ステップごとのログリスト
        """
        max_steps = max_steps or self.max_steps
        logs = []

        # 全シーケンスから初期状態を収集
        all_z = np.concatenate([s["z"] for s in encoded_sequences], axis=0)
        all_h = np.concatenate([s["h"] for s in encoded_sequences], axis=0)
        T_total = len(all_z)

        while self.global_step < max_steps:
            # ランダムにバッチサイズ分の初期状態をサンプル
            idx = np.random.randint(0, T_total, size=batch_size)
            z0 = torch.tensor(all_z[idx], dtype=torch.float32, device=self.device)
            h0 = torch.tensor(all_h[idx], dtype=torch.float32, device=self.device)

            step_log = self.train_step(z0, h0)
            logs.append({"step": self.global_step, **step_log})
            self.loss_history.append(logs[-1])

            if self.global_step % self.log_interval == 0:
                print(
                    f"[AC] Step {self.global_step}/{max_steps} | "
                    f"Actor: {step_log['actor_loss']:.4f} | "
                    f"AC: {step_log['ac_loss']:.4f} | "
                    f"BC: {step_log['bc_loss']:.4f} | "
                    f"Critic: {step_log['critic_loss']:.4f} | "
                    f"α={step_log['alpha']:.3f}"
                )

        return logs

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.global_step = ckpt.get("global_step", 0)
