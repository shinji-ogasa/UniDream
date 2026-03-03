"""Model-free PPO ベースライン.

世界モデルを使わず直接方策勾配で学習する。
報酬: DSR(r_t - costs_t) - β·ΔDD_t
世界モデルの寄与を差分で確認するための比較対象として使用する。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from unidream.data.oracle import ACTIONS


# --- ネットワーク ---

class PPOActor(nn.Module):
    """離散行動 Actor.

    Args:
        obs_dim: 観測次元
        act_dim: 行動数（=5）
        hidden_dim: 隠れ層次元
    """

    def __init__(self, obs_dim: int, act_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        """観測から行動分布を返す."""
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """行動サンプルと log prob を返す."""
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class PPOCritic(nn.Module):
    """Value function Critic.

    Args:
        obs_dim: 観測次元
        hidden_dim: 隠れ層次元
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# --- DSR 報酬 ---

class DSRReward:
    """Differential Sharpe Ratio 報酬の EMA 計算器.

    R_t = DSR(r_t - costs_t) - β·ΔDD_t
    """

    def __init__(self, eta: float = 0.01, beta: float = 0.1):
        self.eta = eta
        self.beta = beta
        self.reset()

    def reset(self):
        self.mu = 0.0
        self.sigma2 = 1e-4
        self.peak = 0.0
        self.cumulative = 0.0

    def step(self, net_return: float) -> float:
        """1 ステップの DSR 報酬を計算する.

        Args:
            net_return: コスト控除後のリターン

        Returns:
            DSR 報酬
        """
        # Differential Sharpe Ratio
        delta_mu = net_return - self.mu
        dsr = (self.sigma2 * delta_mu - 0.5 * delta_mu ** 2 * self.mu) / (
            self.sigma2 ** 1.5 + 1e-8
        )

        # EMA 更新
        self.mu = (1 - self.eta) * self.mu + self.eta * net_return
        self.sigma2 = (1 - self.eta) * self.sigma2 + self.eta * (net_return - self.mu) ** 2

        # ドローダウン計算
        self.cumulative += net_return
        prev_peak = self.peak
        self.peak = max(self.peak, self.cumulative)
        delta_dd = max(0.0, self.peak - prev_peak - (self.cumulative - prev_peak))

        reward = dsr - self.beta * delta_dd
        return float(reward)

    def batch_rewards(
        self,
        net_returns: np.ndarray,
    ) -> np.ndarray:
        """バッチでリターン列から報酬列を計算する."""
        rewards = np.zeros_like(net_returns)
        self.reset()
        for t, r in enumerate(net_returns):
            rewards[t] = self.step(float(r))
        return rewards


# --- トレーディング環境 ---

class TradingEnv:
    """バックテスト形式のトレーディング環境.

    Args:
        returns: 対数リターン列 (T,)
        features: 特徴量行列 (T, feat_dim)
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ
        dsr_eta: DSR EMA 係数
        beta: ドローダウンペナルティ係数
    """

    def __init__(
        self,
        returns: np.ndarray,
        features: np.ndarray,
        spread_bps: float = 5.0,
        fee_rate: float = 0.0004,
        slippage_bps: float = 2.0,
        dsr_eta: float = 0.01,
        beta: float = 0.1,
    ):
        self.returns = returns
        self.features = features
        self.T = len(returns)
        self.spread_bps = spread_bps
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.dsr_reward = DSRReward(eta=dsr_eta, beta=beta)
        self.action_values = ACTIONS
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position = 0.0
        self.dsr_reward.reset()
        return self.features[0]

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        """1 ステップ実行する.

        Returns:
            (next_obs, reward, done)
        """
        new_position = self.action_values[action_idx]
        r_t = self.returns[self.t]

        # コスト計算
        delta_pos = abs(new_position - self.position)
        cost = (
            (self.spread_bps / 10000) / 2 +
            self.fee_rate +
            (self.slippage_bps / 10000)
        ) * delta_pos

        net_return = new_position * r_t - cost
        reward = self.dsr_reward.step(net_return)

        self.position = new_position
        self.t += 1
        done = self.t >= self.T - 1

        next_obs = self.features[min(self.t, self.T - 1)]
        return next_obs, reward, done


# --- PPO Trainer ---

class PPOTrainer:
    """PPO 学習ループ.

    Args:
        obs_dim: 観測次元
        act_dim: 行動数
        hidden_dim: 隠れ層次元
        lr: 学習率
        gamma: 割引率
        gae_lambda: GAE λ
        clip_eps: PPO clip ε
        value_coef: value loss 係数
        entropy_coef: entropy 係数
        n_epochs: 各ロールアウト後の更新エポック数
        batch_size: ミニバッチサイズ
        grad_clip: 勾配クリップ
        device: 計算デバイス
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 5,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
        grad_clip: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.actor = PPOActor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic = PPOCritic(obs_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip

    def _collect_rollout(self, env: TradingEnv) -> dict[str, torch.Tensor]:
        """1 エピソード分のロールアウトを収集する."""
        obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

        obs = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist = self.actor(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.critic(obs_t)

            next_obs, reward, done = env.step(int(action.item()))

            obs_list.append(obs)
            act_list.append(int(action.item()))
            logp_list.append(float(log_prob.item()))
            rew_list.append(reward)
            val_list.append(float(value.item()))
            done_list.append(done)

            obs = next_obs

        # GAE 計算
        T = len(rew_list)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = val_list[t + 1] if t + 1 < T else 0.0
            delta = rew_list[t] + self.gamma * next_val * (1 - done_list[t]) - val_list[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - done_list[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(val_list, dtype=np.float32)

        return {
            "obs": torch.tensor(np.array(obs_list), dtype=torch.float32),
            "actions": torch.tensor(act_list, dtype=torch.long),
            "log_probs_old": torch.tensor(logp_list, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "returns": torch.tensor(returns, dtype=torch.float32),
        }

    def _update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        """PPO アップデートを実行する."""
        dataset = TensorDataset(
            rollout["obs"],
            rollout["actions"],
            rollout["log_probs_old"],
            rollout["advantages"],
            rollout["returns"],
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs_b, act_b, logp_old_b, adv_b, ret_b in loader:
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                logp_old_b = logp_old_b.to(self.device)
                adv_b = adv_b.to(self.device)
                ret_b = ret_b.to(self.device)

                # Actor loss
                dist = self.actor(obs_b)
                logp_new = dist.log_prob(act_b)
                ratio = torch.exp(logp_new - logp_old_b)
                adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_norm
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values = self.critic(obs_b)
                critic_loss = F.mse_loss(values, ret_b)

                # Entropy
                entropy = dist.entropy().mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.grad_clip,
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def train(
        self,
        env: TradingEnv,
        max_steps: int = 500000,
        log_interval: int = 10000,
    ) -> list[dict]:
        """学習ループを実行する.

        Args:
            env: TradingEnv
            max_steps: 最大ステップ数
            log_interval: ログ出力間隔

        Returns:
            各ロールアウトのメトリクスリスト
        """
        total_steps = 0
        logs = []

        while total_steps < max_steps:
            rollout = self._collect_rollout(env)
            update_stats = self._update(rollout)
            episode_return = float(rollout["returns"].sum().item())
            n_steps = len(rollout["obs"])
            total_steps += n_steps

            log = {
                "steps": total_steps,
                "episode_return": episode_return,
                **update_stats,
            }
            logs.append(log)

            if total_steps % log_interval < n_steps:
                print(f"[PPO] Steps: {total_steps}/{max_steps} | "
                      f"Return: {episode_return:.4f} | "
                      f"ActorLoss: {update_stats['actor_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy']:.4f}")

        return logs

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """特徴量列からポジション比率列を返す（greedy）.

        Args:
            features: (T, feat_dim)

        Returns:
            ポジション比率列 (T,)
        """
        self.actor.eval()
        obs_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        dists = self.actor(obs_t)
        action_indices = dists.probs.argmax(dim=-1).cpu().numpy()
        return ACTIONS[action_indices]

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
