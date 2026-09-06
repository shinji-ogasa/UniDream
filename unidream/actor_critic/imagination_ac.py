"""Imagination Actor-Critic 学習モジュール.

DreamerV3 ベースの Imagination AC + BC 損失減衰混合。

アルゴリズム:
  1. 実軌跡からの世界モデル学習（train_wm.py で実施済みを前提）
  2. 現在の観測から z, h をエンコード
  3. Imagination: Actor が z_t を入力に行動 a_t を選択
     → 世界モデルが next_z_{t+1}, net_return_{t+1}, done_{t+1} を予測
     → horizon=3 まで繰り返す
  4. λ-return（symlog 空間）で advantage を計算
     ★ _compute_lambda_returns は報酬・value を原スケールで受け取り
        内部で symlog を一度だけ適用する（二重 symlog を防ぐ）
  5. Actor loss: α·BC_loss + (1-α)·AC_loss（α は 1→0 線形減衰）
  6. Critic loss: twohot cross-entropy（targets は symlog 空間）
  7. TD3+BC 的な保守的制約を Actor loss に付加

References:
    DreamerV3 Actor-Critic (ICLR 2023)
    TD3+BC: https://arxiv.org/abs/2106.06860
"""
from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unidream.actor_critic.actor import Actor
from unidream.actor_critic.critic import Critic, RewardEMANorm
from unidream.actor_critic.market_reward import (
    MarketExecution,
    compound_drawdown,
    market_portfolio_step,
)
from unidream.device import resolve_device
from unidream.experiments.checkpointing import atomic_torch_save
from unidream.eval.policy_stats import action_stats as _action_stats
from unidream.eval.policy_stats import format_action_stats as _fmt_action_stats
from unidream.world_model.ensemble import EnsembleWorldModel
from unidream.world_model.transformer import symlog, symexp, twohot_decode, twohot_encode

def _ac_alerts(label: str, s: dict, bc_loss: float | None = None) -> None:
    """ポジション偏り・turnover・BC loss の異常を検出してアラートを出す."""
    directional_collapse = max(s["long"], s["short"]) > 0.80 and s["switches"] <= 5 and s["turnover"] < 1.0
    if directional_collapse and s["long"] > 0.80:
        print(f"  ⚠️  [{label}] long 比率 {s['long']:.0%} > 80%")
    if directional_collapse and s["short"] > 0.80:
        print(f"  ⚠️  [{label}] short 比率 {s['short']:.0%} > 80%")
    if s["avg_hold"] < 2.0:
        print(f"  ⚠️  [{label}] avg_hold={s['avg_hold']:.1f}b — 高 turnover")
    if bc_loss is not None and bc_loss > 0.05:
        print(f"  ⚠️  [{label}] BC loss {bc_loss:.4f} > 0.05")


def _ac_alerts_ascii(label: str, s: dict, bc_loss: float | None = None) -> None:
    """ASCII-safe alert logging for Windows cp932 terminals."""
    directional_collapse = max(s["long"], s["short"]) > 0.80 and s["switches"] <= 5 and s["turnover"] < 1.0
    if directional_collapse and s["long"] > 0.80:
        print(f"  [WARN] [{label}] long ratio {s['long']:.0%} > 80%")
    if directional_collapse and s["short"] > 0.80:
        print(f"  [WARN] [{label}] short ratio {s['short']:.0%} > 80%")
    if s["avg_hold"] < 2.0:
        print(f"  [WARN] [{label}] avg_hold={s['avg_hold']:.1f}b high turnover")
    if bc_loss is not None and bc_loss > 0.05:
        print(f"  [WARN] [{label}] BC loss {bc_loss:.4f} > 0.05")


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
        self.device = torch.device(resolve_device(device))

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.ensemble.to(self.device)
        self.ensemble.eval()  # WM は固定
        self.actor_prior = copy.deepcopy(self.actor).to(self.device).eval()
        for p in self.actor_prior.parameters():
            p.requires_grad_(False)

        cfg = cfg or {}
        ac_cfg = cfg.get("ac", {})
        reward_cfg = cfg.get("reward", {})
        wm_cfg = cfg.get("world_model", {})
        self.market_reward_mode = (
            wm_cfg.get("reward_mode", reward_cfg.get("mode", "absolute")) == "market_log_return"
        )
        self.market_execution: MarketExecution | None = None
        self.market_ignore_done = False
        self.market_reward_contract = None
        if self.market_reward_mode:
            # The old action-dependent WM objective remains untouched.  This
            # opt-in path has a different, explicit physical reward contract.
            from unidream.world_model.train_wm import world_model_action_context

            if world_model_action_context(cfg) != "actionless":
                raise ValueError("market_log_return requires action_context: actionless")
            benchmark = reward_cfg.get("benchmark_position", 1.0)
            if isinstance(benchmark, bool) or benchmark != 1.0:
                raise ValueError("market_log_return requires benchmark_position: 1.0")
            if str(ac_cfg.get("reward_objective", "legacy")).lower() not in {
                "benchmark_absolute_constraint", "absolute_bh_constraint", "final_metric"
            }:
                raise ValueError("market_log_return requires benchmark_absolute_constraint")
            if cfg.get("data", {}).get("interval", "15m") != "15m":
                raise ValueError("market_log_return supports only the 15m execution contract")
            costs = cfg.get("costs", {})
            rates = [costs.get("fee_rate", .0003), costs.get("spread_bps", 3.0),
                     costs.get("slippage_bps", 1.0)]
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not np.isfinite(x) or x < 0 for x in rates):
                raise ValueError("market execution costs must be finite nonnegative numbers")
            self.market_execution = MarketExecution(
                one_way_cost=rates[0] + rates[1] / 20000 + rates[2] / 10000,
                borrow_annual=costs.get("borrow_annual", .10),
                max_step=ac_cfg.get("max_position_step", .08),
                deadband=ac_cfg.get("market_deadband", .01),
                position_min=ac_cfg.get("abs_min_position", .50),
                position_max=ac_cfg.get("abs_max_position", 1.12),
            )
            done_scale = wm_cfg.get("done_scale", 1.0)
            if isinstance(done_scale, bool) or not isinstance(done_scale, (int, float)) \
                    or not np.isfinite(done_scale) or done_scale < 0:
                raise ValueError("market done_scale must be finite and nonnegative")
            self.market_ignore_done = done_scale == 0
            self.market_reward_contract = {
                "mode": "market_log_return", "context_action": 1.0,
                "execution": vars(self.market_execution).copy(),
                "ignore_done": self.market_ignore_done,
                "drawdown": "compound_running_maxdd_initial_nav1",
            }

        self.horizon = ac_cfg.get("horizon", 3)
        self.context_len = cfg.get("data", {}).get("seq_len", 64)
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
        self.target_aux_coef = ac_cfg.get("target_aux_coef", 1.0)
        self.trade_aux_coef = ac_cfg.get("trade_aux_coef", 0.5)
        self.band_aux_coef = ac_cfg.get("band_aux_coef", 0.25)
        self.execution_aux_coef = ac_cfg.get("execution_aux_coef", 0.0)
        self.prior_kl_coef = ac_cfg.get("prior_kl_coef", 0.0)
        self.prior_trade_coef = ac_cfg.get("prior_trade_coef", 0.0)
        self.prior_band_coef = ac_cfg.get("prior_band_coef", 0.0)
        self.prior_flow_coef = ac_cfg.get("prior_flow_coef", 0.0)
        self.turnover_coef = ac_cfg.get("turnover_coef", 0.0)
        self.flow_change_coef = ac_cfg.get("flow_change_coef", 0.0)
        self.active_deviation_coef = ac_cfg.get("active_deviation_coef", 0.0)
        self.underweight_exposure_coef = ac_cfg.get("underweight_exposure_coef", 0.0)
        self.underweight_floor = ac_cfg.get("underweight_floor", 0.0)
        self.upside_miss_coef = ac_cfg.get("upside_miss_coef", 0.0)
        self.downside_hedge_coef = ac_cfg.get("downside_hedge_coef", 0.0)
        self.nn_anchor_coef = ac_cfg.get("nn_anchor_coef", 0.0)
        self.nn_anchor_flow_coef = ac_cfg.get("nn_anchor_flow_coef", 0.0)
        self.nn_anchor_bank_size = ac_cfg.get("nn_anchor_bank_size", 4096)
        self.positive_advantages = ac_cfg.get("positive_advantages", False)
        self.critic_only = bool(ac_cfg.get("critic_only", False))
        raw_prefixes = ac_cfg.get("trainable_actor_prefixes")
        self.trainable_actor_prefixes = tuple(raw_prefixes or [])

        # SPEC: R_t = DSR(r_t - costs_t) - β·DD_t
        # WM は net_return（コスト控除済み）を予測するため、
        # imagination reward には EMA 正規化のみ適用（DSR の近似）。
        # DD_t は rollout 内 running peak からの累積ドローダウンレベル。
        self.beta = reward_cfg.get("beta", 0.1)

        self._apply_actor_trainable_mask()
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        if not actor_params:
            actor_params = list(self.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=ac_cfg.get("actor_lr", 3e-5))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=ac_cfg.get("critic_lr", 3e-4)
        )

        self.bins = self.ensemble.get_bins().to(self.device)
        self.reward_ema = RewardEMANorm()
        self.global_step = 0
        self.loss_history: list[dict] = []
        self.checkpoint_interval = ac_cfg.get("checkpoint_interval", 10_000)
        self.save_step_checkpoints = bool(ac_cfg.get("save_step_checkpoints", False))
        self.step_checkpoint_prefix = str(ac_cfg.get("step_checkpoint_prefix", "ac_step"))
        self.critic_pretrain_steps = ac_cfg.get("critic_pretrain_steps", 0)

        # Early stopping
        # val_patience: val Sharpe が N 回連続で best 更新なければ停止（0 で無効）
        self.val_patience = ac_cfg.get("val_patience", 0)
        # bc_loss_threshold: BC loss がこの値を超えた状態が bc_loss_patience 回続けば停止
        self.bc_loss_threshold = ac_cfg.get("bc_loss_threshold", 0.0)
        self.bc_loss_patience = ac_cfg.get("bc_loss_patience", 3)

        # EMA of |advantage| for stable norm_q (TD3+BC)
        self._adv_ema: float = 1.0

        # α が単調非増加になるよう到達済み最大 t を追跡する
        self._max_alpha_t: float = 0.0
        # Curriculum stage ごとに α schedule を再開始できるようにする。
        # 非 curriculum 実行では 0 のままなので従来挙動と同じ。
        self._alpha_stage_start_step: int = 0
        self.gradient_diagnostics_interval: int = int(
            ac_cfg.get("gradient_diagnostics_interval", 0)
        )
        self.actor_runtime_overrides: dict[str, object] = {}
        self.actor_runtime_defaults: dict[str, object] = {}
        self.checkpoint_metadata: dict[str, object] = {}
        self.last_train_best_candidate: dict[str, object] | None = None

        # BC 損失用の oracle データ（bc_pretrain 後に set_oracle_data で設定）
        self._oracle_z: Optional[torch.Tensor] = None
        self._oracle_h: Optional[torch.Tensor] = None
        self._oracle_positions: Optional[torch.Tensor] = None
        self._oracle_inventory: Optional[torch.Tensor] = None
        self._oracle_trade_pos_weight: Optional[torch.Tensor] = None
        self._oracle_anchor_h: Optional[torch.Tensor] = None
        self._oracle_anchor_inventory: Optional[torch.Tensor] = None
        self._oracle_anchor_regime: Optional[torch.Tensor] = None
        self._oracle_anchor_advantage: Optional[torch.Tensor] = None
        self._oracle_anchor_overlay: Optional[torch.Tensor] = None

        # DSR EMA trackers
        self._dsr_A: float = 0.0    # EMA of reward (running mean)
        self._dsr_B: float = 1e-4   # EMA of reward^2 (running variance proxy)
        self._dsr_eta: float = reward_cfg.get("dsr_eta", 0.01)
        self.use_dsr: bool = ac_cfg.get("use_dsr", False)
        self.benchmark_position: float = reward_cfg.get("benchmark_position", 1.0)
        self.abs_min_position: float = ac_cfg.get("abs_min_position", -1.0)
        self.abs_max_position: float = ac_cfg.get("abs_max_position", 1.0)

        self.reward_objective = str(ac_cfg.get("reward_objective", "legacy")).lower()
        self.logwealth_coef = float(ac_cfg.get("logwealth_coef", 1.0))
        self.relative_dd_coef = float(ac_cfg.get("relative_dd_coef", 0.0))
        self.relative_dd_budget = float(ac_cfg.get("relative_dd_budget", 0.0))
        self.relative_terminal_dd_coef = float(ac_cfg.get("relative_terminal_dd_coef", 0.0))
        self.relative_dd_improve_coef = float(ac_cfg.get("relative_dd_improve_coef", 0.0))
        self.relative_terminal_dd_improve_coef = float(ac_cfg.get("relative_terminal_dd_improve_coef", 0.0))
        self.alpha_floor_coef = float(ac_cfg.get("alpha_floor_coef", 0.0))
        self.alpha_floor = float(ac_cfg.get("alpha_floor", 0.0))
        self.relative_cvar_coef = float(ac_cfg.get("relative_cvar_coef", 0.0))
        self.dd_level_coef = float(ac_cfg.get("dd_level_coef", 0.0))
        self.dd_budget_coef = float(ac_cfg.get("dd_budget_coef", 0.0))
        self.dd_budget = float(ac_cfg.get("dd_budget", 0.0))
        self.terminal_dd_coef = float(ac_cfg.get("terminal_dd_coef", 0.0))
        self.downside_coef = float(ac_cfg.get("downside_coef", 0.0))
        self.tail_coef = float(ac_cfg.get("tail_coef", 0.0))
        self.tail_margin = float(ac_cfg.get("tail_margin", 0.0))
        self.overlay_l2_coef = float(ac_cfg.get("overlay_l2_coef", 0.0))
        self.abs_exposure_l2_coef = float(ac_cfg.get("abs_exposure_l2_coef", 0.0))
        self.risk_state_exposure_coef = float(ac_cfg.get("risk_state_exposure_coef", 0.0))
        self.risk_tilt_coef = float(ac_cfg.get("risk_tilt_coef", 0.0))
        self.risk_state_center = float(ac_cfg.get("risk_state_center", 0.0))
        self.risk_state_scale = float(ac_cfg.get("risk_state_scale", 1.0))
        raw_risk_state_indices = ac_cfg.get("risk_state_indices", [])
        self.risk_state_indices = tuple(int(i) for i in raw_risk_state_indices)
        self.edge_overlay_coef = float(ac_cfg.get("edge_overlay_coef", 0.0))
        self.edge_state_center = float(ac_cfg.get("edge_state_center", 0.0))
        self.edge_state_scale = float(ac_cfg.get("edge_state_scale", 1.0))
        raw_edge_state_indices = ac_cfg.get("edge_state_indices", [])
        self.edge_state_indices = tuple(int(i) for i in raw_edge_state_indices)
        self.short_l1_coef = float(ac_cfg.get("short_l1_coef", 0.0))
        self.overweight_l1_coef = float(ac_cfg.get("overweight_l1_coef", 0.0))

        # Adaptive BC
        self.adaptive_bc: bool = ac_cfg.get("adaptive_bc", False)
        self._alpha_speed: float = 1.0   # multiplier on alpha decay speed
        self._last_val_sharpe: Optional[float] = None

        # Regime conditioning
        self.regime_dim: int = 0  # set later via set_regime_dim()
        self._oracle_regime: Optional[torch.Tensor] = None
        self._oracle_advantage: Optional[torch.Tensor] = None

        # Online WM update interval
        self.online_wm_interval: int = ac_cfg.get("online_wm_interval", 0)
        self.restore_best_val_checkpoint: bool = bool(ac_cfg.get("restore_best_val_checkpoint", True))

    def _apply_actor_trainable_mask(self) -> None:
        if self.critic_only:
            for param in self.actor.parameters():
                param.requires_grad_(False)
            return
        if not self.trainable_actor_prefixes:
            for param in self.actor.parameters():
                param.requires_grad_(True)
            return
        prefixes = tuple(str(prefix) for prefix in self.trainable_actor_prefixes)
        for name, param in self.actor.named_parameters():
            param.requires_grad_(name.startswith(prefixes))

    def set_regime_dim(self, regime_dim: int) -> None:
        """regime_dim を後から設定する（Actor が外部で構築される場合用）."""
        self.regime_dim = regime_dim

    def set_oracle_data(
        self,
        z: np.ndarray,
        h: np.ndarray,
        oracle_positions: np.ndarray,
        regime_probs: "np.ndarray | None" = None,
        advantage_values: "np.ndarray | None" = None,
    ) -> None:
        """BC 損失用の Oracle データを設定する."""
        T = min(len(z), len(h), len(oracle_positions))
        self._oracle_z = torch.tensor(z[:T], dtype=torch.float32, device=self.device)
        self._oracle_h = torch.tensor(h[:T], dtype=torch.float32, device=self.device)
        clipped_positions = np.clip(oracle_positions[:T], self.abs_min_position, self.abs_max_position)
        self._oracle_positions = torch.tensor(clipped_positions, dtype=torch.float32, device=self.device)
        oracle_inventory = np.zeros(T, dtype=np.float32)
        if T > 1:
            oracle_inventory[1:] = clipped_positions[:T - 1] - self.benchmark_position
        self._oracle_inventory = torch.tensor(
            oracle_inventory, dtype=torch.float32, device=self.device
        )
        trade_targets = (
            np.abs((clipped_positions - self.benchmark_position) - oracle_inventory) > 1e-8
        ).astype(np.float32)
        n_pos = float(trade_targets.sum())
        n_neg = float(T - n_pos)
        if n_pos > 0 and n_neg > 0:
            self._oracle_trade_pos_weight = torch.tensor(
                n_neg / n_pos, dtype=torch.float32, device=self.device
            )
        else:
            self._oracle_trade_pos_weight = None
        if regime_probs is not None:
            self._oracle_regime = torch.tensor(
                regime_probs[:T], dtype=torch.float32, device=self.device
            )
        else:
            self._oracle_regime = None
        if advantage_values is not None and self.actor.advantage_dim > 0:
            advantage_arr = np.asarray(advantage_values[:T], dtype=np.float32)
            if advantage_arr.ndim == 1:
                advantage_arr = advantage_arr[:, None]
            self._oracle_advantage = torch.tensor(
                advantage_arr[:, : self.actor.advantage_dim],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._oracle_advantage = None

        bank_size = int(self.nn_anchor_bank_size)
        if bank_size > 0 and T > 0:
            if T <= bank_size:
                anchor_idx = np.arange(T, dtype=np.int64)
            else:
                anchor_idx = np.linspace(0, T - 1, num=bank_size, dtype=np.int64)
            anchor_idx_t = torch.tensor(anchor_idx, dtype=torch.long, device=self.device)
            self._oracle_anchor_h = self._oracle_h.index_select(0, anchor_idx_t)
            self._oracle_anchor_inventory = self._oracle_inventory.index_select(0, anchor_idx_t)
            self._oracle_anchor_overlay = (
                self._oracle_positions.index_select(0, anchor_idx_t) - self.benchmark_position
            )
            if self._oracle_regime is not None:
                self._oracle_anchor_regime = self._oracle_regime.index_select(0, anchor_idx_t)
            else:
                self._oracle_anchor_regime = None
            if self._oracle_advantage is not None:
                self._oracle_anchor_advantage = self._oracle_advantage.index_select(0, anchor_idx_t)
            else:
                self._oracle_anchor_advantage = None
        else:
            self._oracle_anchor_h = None
            self._oracle_anchor_inventory = None
            self._oracle_anchor_overlay = None
            self._oracle_anchor_regime = None
            self._oracle_anchor_advantage = None

    def _get_alpha(self) -> float:
        """現在の BC/AC 混合比率 α を返す（stage 内で単調非増加）.

        adaptive_bc で _alpha_speed が下がっても α が増加しないよう
        _max_alpha_t で到達済み最大 t を追跡する。
        """
        decay_steps = max(float(self.alpha_decay_steps), 1.0)
        stage_step = max(self.global_step - self._alpha_stage_start_step, 0)
        t = min(stage_step * self._alpha_speed, decay_steps)
        self._max_alpha_t = max(self._max_alpha_t, t)   # 単調増加を強制
        return self.alpha_init + (self.alpha_final - self.alpha_init) * (self._max_alpha_t / decay_steps)

    def begin_alpha_stage(self) -> None:
        """現在 step を起点に stage-local α schedule を開始する."""
        self._alpha_stage_start_step = int(self.global_step)
        self._max_alpha_t = 0.0
        self._alpha_speed = 1.0
        self._last_val_sharpe = None

    def _restore_actor_runtime_config(
        self,
        *,
        defaults: dict[str, object],
        overrides: dict[str, object],
    ) -> None:
        """Actor runtime属性を既定値へ戻してからcheckpoint差分を適用する."""
        self.actor_runtime_defaults.update(defaults)
        for name, value in self.actor_runtime_defaults.items():
            if not hasattr(self.actor, name):
                raise RuntimeError(
                    f"AC checkpoint has unknown actor runtime default: {name}"
                )
            setattr(self.actor, name, value)
            if name in {"abs_min_position", "abs_max_position"}:
                setattr(self, name, float(value))

        self.actor_runtime_overrides = dict(overrides)
        for name, value in self.actor_runtime_overrides.items():
            if not hasattr(self.actor, name):
                raise RuntimeError(
                    f"AC checkpoint has unknown actor runtime override: {name}"
                )
            setattr(self.actor, name, value)
            if name in {"abs_min_position", "abs_max_position"}:
                setattr(self, name, float(value))

    def _actor_gradient_diagnostics(
        self,
        bc_component: torch.Tensor,
        ac_component: torch.Tensor,
    ) -> dict[str, float]:
        """BC と AC の actor gradient norm/cosine を副作用なしで測る."""
        params = [p for p in self.actor.parameters() if p.requires_grad]
        if not params:
            return {}
        bc_grads = torch.autograd.grad(
            bc_component,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        ac_grads = torch.autograd.grad(
            ac_component,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        bc_sq = torch.zeros((), device=self.device)
        ac_sq = torch.zeros((), device=self.device)
        dot = torch.zeros((), device=self.device)
        for bc_grad, ac_grad in zip(bc_grads, ac_grads):
            if bc_grad is not None:
                bc_sq = bc_sq + bc_grad.detach().square().sum()
            if ac_grad is not None:
                ac_sq = ac_sq + ac_grad.detach().square().sum()
            if bc_grad is not None and ac_grad is not None:
                dot = dot + (bc_grad.detach() * ac_grad.detach()).sum()
        bc_norm = bc_sq.sqrt()
        ac_norm = ac_sq.sqrt()
        denom = (bc_norm * ac_norm).clamp_min(1e-12)
        return {
            "bc_grad_norm": float(bc_norm.item()),
            "ac_grad_norm": float(ac_norm.item()),
            "bc_ac_grad_cosine": float((dot / denom).item()),
        }

    def _compute_dsr_rewards(self, net_returns: torch.Tensor) -> torch.Tensor:
        """DSR 報酬を計算する（インクリメンタル Sharpe 改善量）."""
        B, H = net_returns.shape
        dsr_rewards = torch.zeros_like(net_returns)
        A = self._dsr_A
        Bsq = self._dsr_B
        eta = self._dsr_eta
        for t in range(H):
            r_t = net_returns[:, t]
            denom = (Bsq - A * A + 1e-8) ** 1.5
            dsr_t = (Bsq * (r_t - A) - 0.5 * A * (r_t ** 2 - Bsq)) / denom
            dsr_rewards[:, t] = dsr_t
            r_mean = r_t.detach().mean().item()
            A = A + eta * (r_mean - A)
            Bsq = Bsq + eta * (r_mean ** 2 - Bsq)
        self._dsr_A = A
        self._dsr_B = Bsq
        return dsr_rewards

    def _imagination_rollout(
        self,
        z0: torch.Tensor,
        h0: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
        inventory0: Optional[torch.Tensor] = None,
        regime0: Optional[torch.Tensor] = None,
        advantage0: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Imagination rollout を実行する（horizon ステップ）.

        Returns:
            rewards: WM の reward head が予測した net_return（原スケール）
        """
        if getattr(self, "market_reward_mode", False):
            return self._market_imagination_rollout(
                z0, h0, past_zs, past_as, inventory0, regime0, advantage0
            )
        zs, hs, inventories, acts, log_probs_list, entropies_list, rewards_list, dones_list = [], [], [], [], [], [], [], []

        z = torch.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0)
        h = torch.nan_to_num(h0, nan=0.0, posinf=0.0, neginf=0.0)
        pzs = past_zs
        pas = past_as
        if pzs is not None:
            pzs = torch.nan_to_num(pzs, nan=0.0, posinf=0.0, neginf=0.0)
        if pas is not None:
            pas = torch.nan_to_num(pas, nan=0.0, posinf=0.0, neginf=0.0)
        if regime0 is not None:
            regime0 = torch.nan_to_num(regime0, nan=0.0, posinf=0.0, neginf=0.0)
        if advantage0 is not None:
            advantage0 = torch.nan_to_num(advantage0, nan=0.0, posinf=0.0, neginf=0.0)
        if inventory0 is None:
            inventory = torch.zeros(z0.shape[0], 1, dtype=z0.dtype, device=z0.device)
        elif inventory0.ndim == 1:
            inventory = inventory0.unsqueeze(-1)
        else:
            inventory = inventory0
        inventory = torch.nan_to_num(inventory, nan=0.0, posinf=0.0, neginf=0.0)

        for _ in range(self.horizon):
            next_inventory, log_prob, entropy = self.actor.get_action(
                z, h, inventory=inventory, regime=regime0, advantage=advantage0
            )
            next_inventory = torch.nan_to_num(next_inventory, nan=self.benchmark_position, posinf=self.abs_max_position, neginf=self.abs_min_position)
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                result = self.ensemble.imagine_step(z, h, next_inventory, pzs, pas)
            next_overlay = next_inventory.squeeze(-1) - self.benchmark_position

            zs.append(z)
            hs.append(h)
            inventories.append(inventory.squeeze(-1))
            acts.append(next_overlay)
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)
            rewards_list.append(torch.nan_to_num(result["reward"], nan=0.0, posinf=0.0, neginf=0.0))   # net_return（原スケール）
            dones_list.append(torch.nan_to_num(result["done"], nan=1.0, posinf=1.0, neginf=0.0))

            z = torch.nan_to_num(result["next_z"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
            h = torch.nan_to_num(result["next_h"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
            pzs = torch.nan_to_num(result["past_zs"], nan=0.0, posinf=0.0, neginf=0.0)
            pas = torch.nan_to_num(result["past_as"], nan=0.0, posinf=0.0, neginf=0.0)
            inventory = next_overlay.unsqueeze(-1)

        return {
            "zs": torch.stack(zs, dim=1),                      # (B, H, z_dim)
            "hs": torch.stack(hs, dim=1),                      # (B, H, h_dim)
            "inventories": torch.stack(inventories, dim=1),    # (B, H)
            "actions": torch.stack(acts, dim=1),               # (B, H)
            "log_probs": torch.stack(log_probs_list, dim=1),   # (B, H)
            "entropies": torch.stack(entropies_list, dim=1),   # (B, H)
            "rewards": torch.stack(rewards_list, dim=1),       # (B, H) 原スケール
            "dones": torch.stack(dones_list, dim=1),           # (B, H)
            "last_z": z,
            "last_h": h,
        }

    def _market_imagination_rollout(
        self, z0, h0, past_zs=None, past_as=None, inventory0=None,
        regime0=None, advantage0=None,
    ) -> dict[str, torch.Tensor]:
        """Market-only WM path plus a persistent, self-financing actor account.

        Initial NAV/price are one, with pre-existing exposure 1+inventory0.
        The actor receives un-clipped actual exposure after passive drift.
        Its intent is filled now, then borrowing and a 15m market mark occur.
        No future actual return or auxiliary label enters this path.
        """
        def require_finite(value, name):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() \
                    or not bool(torch.isfinite(value).all()):
                raise ValueError(f"market rollout {name} must be a finite floating tensor")

        for name, value in (("z0", z0), ("h0", h0), ("past_zs", past_zs),
                            ("regime0", regime0), ("advantage0", advantage0)):
            if value is not None:
                require_finite(value, name)
        batch = z0.shape[0]
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("market horizon must be a positive integer")
        if inventory0 is None:
            inventory = torch.zeros(batch, dtype=z0.dtype, device=z0.device)
        else:
            require_finite(inventory0, "inventory0")
            if inventory0.shape not in {(batch,), (batch, 1), (batch, 4)}:
                raise ValueError("market inventory0 must have B, B x 1 or B x 4 shape")
            inventory = (inventory0[:, 0] if inventory0.ndim == 2 else inventory0).detach().to(z0)
        full_controller = int(getattr(self.actor, "inventory_dim", 1)) == 4
        controller = inventory.unsqueeze(-1)
        if full_controller:
            controller = self.actor.make_controller_state(inventory)
            if inventory0 is not None and inventory0.shape == (batch, 4):
                controller = inventory0.detach().to(z0).clone()
        asset = 1 + inventory
        cash = 1 - asset
        if bool((asset < 0).any()):
            raise ValueError("market rollout cannot begin with short assets")
        z, h, pzs = z0, h0, past_zs
        # Ignore any actor/hindsight action values attached to a context.  The
        # trained market context is fixed B&H at every history and future step.
        pas = None
        if pzs is not None:
            if pzs.ndim != 3 or pzs.shape[0] != batch:
                raise ValueError("market past_zs must have B x T x Z shape")
            pas = torch.ones((*pzs.shape[:2], 1), dtype=z0.dtype, device=z0.device)
        fixed_action = torch.ones(batch, 1, dtype=z0.dtype, device=z0.device)
        values = {name: [] for name in (
            "zs", "hs", "inventories", "controller_states", "actions", "log_probs", "entropies",
            "rewards", "dones", "benchmark_rewards", "market_log_returns",
            "cash", "asset_values", "nav", "fees", "borrow", "trade_values",
        )}
        for _ in range(self.horizon):
            nav = cash + asset
            inventory = (asset / nav - 1).detach()
            controller = controller.clone()
            controller[:, 0] = inventory
            target, log_prob, entropy = self.actor.get_action(
                z, h, inventory=controller, regime=regime0, advantage=advantage0
            )
            for name, value in (("target", target), ("log_prob", log_prob), ("entropy", entropy)):
                require_finite(value, name)
                if value.shape not in {(batch,), (batch, 1)}:
                    raise ValueError(f"market {name} must have B or B x 1 shape")
            with torch.no_grad():
                # Each decision uses the same fixed context as observed
                # inference; generated steps replace old rows, not extend it.
                if pzs is not None:
                    keep = max(int(self.context_len) - 1, 0)
                    pzs = pzs[:, -keep:] if keep else pzs[:, :0]
                    pas = torch.ones((*pzs.shape[:2], 1), dtype=z0.dtype, device=z0.device)
                result = self.ensemble.imagine_step(z, h, fixed_action, pzs, pas)
                market_log = result["reward"]
                require_finite(market_log, "predicted market return")
                if market_log.shape != (batch,):
                    raise ValueError("predicted market return must have B shape")
                # Ensemble.imagine_step subtracts an optional uncertainty term.
                # Restore the actual mean market-head output for physical price
                # accounting; uncertainty is not a realized price movement.
                if "disagreement" in result:
                    disagreement = result["disagreement"]
                    require_finite(disagreement, "disagreement")
                    if disagreement.shape != (batch,) or bool((disagreement < 0).any()):
                        raise ValueError("invalid market disagreement")
                    market_log = market_log + float(getattr(self.ensemble, "disagree_scale", 0)) * disagreement
                elif getattr(self.ensemble, "disagree_scale", 0) != 0:
                    raise ValueError("market ensemble must expose its subtracted disagreement")
                account = market_portfolio_step(
                    cash, asset, target.reshape(batch).detach(), market_log, self.market_execution
                )
                benchmark_return = torch.expm1(market_log)
                require_finite(benchmark_return, "benchmark return")
                if self.market_ignore_done:
                    done = torch.zeros_like(market_log)
                else:
                    done = result["done"]
                    require_finite(done, "done")
                    if done.shape != (batch,) or bool(((done < 0) | (done > 1)).any()):
                        raise ValueError("market done probabilities must lie in [0, 1]")
            current_values = {
                "zs": z, "hs": h, "inventories": inventory,
                "controller_states": controller,
                "actions": account["executed_position"] - 1,
                "log_probs": log_prob.reshape(batch), "entropies": entropy.reshape(batch),
                "rewards": account["simple_return"], "dones": done,
                "benchmark_rewards": benchmark_return, "market_log_returns": market_log,
                "cash": account["cash"], "asset_values": account["asset_value"],
                "nav": account["nav"], "fees": account["fee"], "borrow": account["borrow"],
                "trade_values": account["trade_value"],
            }
            for name, value in current_values.items():
                values[name].append(value)
            if full_controller:
                # Native duration update, using actual filled exposure change;
                # passive market drift is not a trade. Match live feedback.
                filled_delta = torch.where(account["trade_value"] != 0,
                    account["executed_position"] - (inventory + 1), torch.zeros_like(inventory))
                next_exposure = account["exposure"]
                proxy = controller.clone()
                proxy[:, 0] = next_exposure - 1 - filled_delta
                updated = self.actor.update_controller_state(proxy, next_exposure.unsqueeze(-1))
                updated[:, 0], updated[:, 1] = next_exposure - 1, filled_delta
                eps, scale = self.actor._trade_state_eps(), self.actor._state_hold_scale()
                updated[:, 2] = torch.where(filled_delta.abs() > eps,
                    torch.zeros_like(inventory), (controller[:, 2] + 1 / scale).clamp(max=1))
                updated[:, 3] = torch.where(next_exposure - 1 < -eps,
                    (controller[:, 3] + 1 / scale).clamp(max=1), torch.zeros_like(inventory))
                controller = updated.detach()
            cash, asset = account["cash"], account["asset_value"]
            for name in ("next_z", "next_h", "past_zs"):
                require_finite(result[name], name)
            z, h, pzs = result["next_z"].detach(), result["next_h"].detach(), result["past_zs"].detach()
            pas = torch.ones((*pzs.shape[:2], 1), dtype=z0.dtype, device=z0.device)
        return {**{name: torch.stack(items, dim=1) for name, items in values.items()},
                "last_z": z, "last_h": h}

    @torch.no_grad()
    def _benchmark_rollout_rewards(
        self,
        z0: torch.Tensor,
        h0: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Roll out fixed B&H exposure from the same initial states."""
        if getattr(self, "market_reward_mode", False):
            raise ValueError("market B&H rewards must come from the shared actor market rollout")
        rewards = []
        z = torch.nan_to_num(z0.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        h = torch.nan_to_num(h0.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        pzs = torch.nan_to_num(past_zs.detach(), nan=0.0, posinf=0.0, neginf=0.0) if past_zs is not None else None
        pas = torch.nan_to_num(past_as.detach(), nan=0.0, posinf=0.0, neginf=0.0) if past_as is not None else None
        action = torch.full(
            (z.shape[0], 1),
            float(self.benchmark_position),
            dtype=z.dtype,
            device=z.device,
        )
        for _ in range(self.horizon):
            result = self.ensemble.imagine_step(z, h, action, pzs, pas)
            rewards.append(torch.nan_to_num(result["reward"], nan=0.0, posinf=0.0, neginf=0.0))
            z = torch.nan_to_num(result["next_z"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
            h = torch.nan_to_num(result["next_h"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
            pzs = torch.nan_to_num(result["past_zs"], nan=0.0, posinf=0.0, neginf=0.0)
            pas = torch.nan_to_num(result["past_as"], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.stack(rewards, dim=1)

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_value: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """λ-return を symlog 空間で計算する.

        ★ 引数はすべて原スケール（symlog 変換前）で渡すこと。
           内部で symlog を一度だけ適用する。

        G_t^λ = symlog(r_t) + γ(1-d_t)[(1-λ)·symlog(V_{t+1}) + λ·G_{t+1}^λ]

        Args:
            rewards: (B, H) 各ステップの報酬（原スケール・EMA 正規化済み）
            values: (B, H) 各ステップの slow target value（原スケール）
            last_value: (B,) H+1 ステップ目の bootstrap value（原スケール）
            dones: (B, H) 終了フラグ

        Returns:
            returns: (B, H) λ-return（symlog 空間）
        """
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        G = symlog(last_value)  # bootstrap を symlog 空間へ

        for t in reversed(range(H)):
            r_t = symlog(rewards[:, t])   # ★ ここで一度だけ symlog 適用
            v_t = symlog(values[:, t])    # ★ ここで一度だけ symlog 適用
            d_t = dones[:, t]
            G = r_t + self.gamma * (1 - d_t) * (
                (1 - self.lam) * v_t + self.lam * G
            )
            returns[:, t] = G

        return returns  # symlog 空間

    def _compute_drawdown(self, net_returns: torch.Tensor) -> torch.Tensor:
        """imagination 軌跡の累積ドローダウンレベルを計算する.

        ΔDD（増分）ではなく、rollout 内の peak からの累積下落幅を返す。
        短い horizon でも意味のあるペナルティを与えるため、
        ドローダウン「状態」にいること自体をペナルティ対象とする。

        Args:
            net_returns: (B, H) 原スケールの net_returns

        Returns:
            drawdown: (B, H) ≥ 0  rollout 内の running peak からの下落幅
        """
        cum_rets = net_returns.cumsum(dim=1)                       # (B, H)
        peak = cum_rets.cummax(dim=1).values                       # running max
        return (peak - cum_rets).clamp(min=0.0)                    # DD レベル ≥ 0

    def _risk_state_from_advantage(
        self,
        advantage: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """WM predictive state から高リスク度合いを作る。

        risk_state_indices は標準化済み predictive state のうち、vol/DD 系列を
        指す想定。正の側だけを使い、高ボラ/高DD予測時の absolute exposure を
        落とす圧として使う。
        """
        if (
            advantage is None
            or (self.risk_state_exposure_coef <= 0.0 and self.risk_tilt_coef <= 0.0)
            or not self.risk_state_indices
        ):
            return torch.zeros_like(ref)
        adv = advantage
        if adv.ndim == 1:
            adv = adv.unsqueeze(-1)
        if adv.shape[-1] <= 0:
            return torch.zeros_like(ref)
        valid_indices = [i for i in self.risk_state_indices if 0 <= i < adv.shape[-1]]
        if not valid_indices:
            return torch.zeros_like(ref)
        idx = torch.tensor(valid_indices, dtype=torch.long, device=adv.device)
        risk_raw = adv.index_select(-1, idx).mean(dim=-1)
        risk = F.relu((risk_raw - self.risk_state_center) * self.risk_state_scale)
        while risk.ndim < ref.ndim:
            risk = risk.unsqueeze(-1)
        return risk.expand_as(ref)

    def _signed_state_from_advantage(
        self,
        advantage: Optional[torch.Tensor],
        ref: torch.Tensor,
        indices: tuple[int, ...],
        *,
        center: float = 0.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if advantage is None or not indices:
            return torch.zeros_like(ref)
        adv = advantage
        if adv.ndim == 1:
            adv = adv.unsqueeze(-1)
        valid_indices = [i for i in indices if 0 <= i < adv.shape[-1]]
        if not valid_indices:
            return torch.zeros_like(ref)
        idx = torch.tensor(valid_indices, dtype=torch.long, device=adv.device)
        signal = adv.index_select(-1, idx).mean(dim=-1)
        signal = torch.nan_to_num((signal - center) * scale, nan=0.0, posinf=0.0, neginf=0.0)
        while signal.ndim < ref.ndim:
            signal = signal.unsqueeze(-1)
        return signal.expand_as(ref)

    def _risk_budget_rewards(
        self,
        *,
        net_returns: torch.Tensor,
        next_inventory: torch.Tensor,
        rewards_norm: torch.Tensor,
        advantage0: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Risk-budget overlay 用の報酬を作る。

        WM reward は benchmark-relative net return なので、ここでは
        absolute wealth ではなく B&H relative wealth growth を近似する。
        """
        reward_scale = max(float(self.reward_ema.scale), 1e-8)
        logwealth = torch.log1p(net_returns.clamp(min=-0.95))
        rewards = self.logwealth_coef * (logwealth / reward_scale)
        if self.logwealth_coef == 0.0:
            rewards = rewards_norm

        drawdown = self._compute_drawdown(net_returns)
        if self.dd_level_coef > 0.0:
            rewards = rewards - self.dd_level_coef * (drawdown / reward_scale)
        if self.dd_budget_coef > 0.0:
            dd_excess = F.relu(drawdown - self.dd_budget)
            rewards = rewards - self.dd_budget_coef * (dd_excess / reward_scale)
        if self.terminal_dd_coef > 0.0 and drawdown.shape[1] > 0:
            terminal_dd = drawdown[:, -1:].expand_as(drawdown)
            rewards = rewards - self.terminal_dd_coef * (terminal_dd / reward_scale)
        if self.downside_coef > 0.0:
            rewards = rewards - self.downside_coef * F.relu(-rewards_norm)
        if self.tail_coef > 0.0:
            tail = F.relu((-net_returns) - self.tail_margin)
            rewards = rewards - self.tail_coef * (tail / reward_scale)
        if self.overlay_l2_coef > 0.0:
            rewards = rewards - self.overlay_l2_coef * next_inventory.pow(2)
        if self.short_l1_coef > 0.0:
            rewards = rewards - self.short_l1_coef * F.relu(-next_inventory)
        if self.overweight_l1_coef > 0.0:
            rewards = rewards - self.overweight_l1_coef * F.relu(next_inventory)

        abs_position = next_inventory + self.benchmark_position
        if self.abs_exposure_l2_coef > 0.0:
            rewards = rewards - self.abs_exposure_l2_coef * abs_position.pow(2)
        risk_state = self._risk_state_from_advantage(advantage0, next_inventory)
        if self.risk_state_exposure_coef > 0.0:
            rewards = rewards - self.risk_state_exposure_coef * risk_state * abs_position.pow(2)
        if self.risk_tilt_coef > 0.0:
            rewards = rewards - self.risk_tilt_coef * risk_state * next_inventory
        edge_state = self._signed_state_from_advantage(
            advantage0,
            next_inventory,
            self.edge_state_indices,
            center=self.edge_state_center,
            scale=self.edge_state_scale,
        )
        if self.edge_overlay_coef > 0.0:
            rewards = rewards + self.edge_overlay_coef * edge_state * next_inventory

        diagnostics = {
            "rb_logwealth": float(logwealth.detach().mean().item()),
            "rb_dd": float(drawdown.detach().mean().item()),
            "rb_risk_state": float(risk_state.detach().mean().item()),
            "rb_edge_state": float(edge_state.detach().mean().item()),
            "rb_abs_exposure": float(abs_position.detach().abs().mean().item()),
        }
        return rewards, diagnostics

    def _benchmark_absolute_constraint_rewards(
        self,
        *,
        strategy_returns: torch.Tensor,
        benchmark_returns: torch.Tensor,
        next_inventory: torch.Tensor,
        rewards_norm: torch.Tensor,
        advantage0: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Final-metric aligned reward using absolute WM rewards.

        Legacy mode uses absolute WM rewards and a parallel imagination path.
        Opt-in market mode instead receives actual portfolio simple returns
        from self-financing accounts on one shared generated market path; its
        drawdown includes initial NAV1 and is compounded, in fractional units.
        """
        reward_scale = max(float(self.reward_ema.scale), 1e-8)
        horizon = max(1, int(strategy_returns.shape[1]))
        if getattr(self, "market_reward_mode", False):
            strategy_dd = compound_drawdown(strategy_returns).cummax(dim=1).values
            benchmark_dd = compound_drawdown(benchmark_returns).cummax(dim=1).values
            strategy_log = torch.log1p(strategy_returns)
            benchmark_log = torch.log1p(benchmark_returns)
        else:
            strategy_log = torch.log1p(strategy_returns.clamp(min=-0.95))
            benchmark_log = torch.log1p(benchmark_returns.clamp(min=-0.95))
            strategy_dd = self._compute_drawdown(strategy_log)
            benchmark_dd = self._compute_drawdown(benchmark_log)
        excess_logwealth = strategy_log - benchmark_log
        rewards = self.logwealth_coef * (excess_logwealth / reward_scale)
        if self.logwealth_coef == 0.0:
            rewards = rewards_norm

        cum_excess = excess_logwealth.cumsum(dim=1)
        dd_delta = strategy_dd - benchmark_dd

        if self.relative_dd_coef > 0.0:
            dd_worse = F.relu(dd_delta - self.relative_dd_budget)
            rewards = rewards - self.relative_dd_coef * (dd_worse / reward_scale)
        if self.relative_dd_improve_coef > 0.0:
            dd_better = F.relu((-dd_delta) - self.relative_dd_budget)
            rewards = rewards + self.relative_dd_improve_coef * (dd_better / reward_scale)
        if self.relative_terminal_dd_coef > 0.0 and dd_delta.shape[1] > 0:
            terminal_worse = F.relu(dd_delta[:, -1:] - self.relative_dd_budget).expand_as(dd_delta)
            rewards = rewards - self.relative_terminal_dd_coef * (terminal_worse / reward_scale)
        if self.relative_terminal_dd_improve_coef > 0.0 and dd_delta.shape[1] > 0:
            terminal_better = F.relu((-dd_delta[:, -1:]) - self.relative_dd_budget).expand_as(dd_delta)
            rewards = rewards + self.relative_terminal_dd_improve_coef * (terminal_better / reward_scale)
        if self.alpha_floor_coef > 0.0 and cum_excess.shape[1] > 0:
            terminal_excess = cum_excess[:, -1:]
            alpha_shortfall = F.relu(self.alpha_floor - terminal_excess).expand_as(cum_excess)
            rewards = rewards - self.alpha_floor_coef * (alpha_shortfall / reward_scale) / float(horizon)
        if self.relative_cvar_coef > 0.0:
            tail = F.relu((-excess_logwealth) - self.tail_margin)
            rewards = rewards - self.relative_cvar_coef * (tail / reward_scale)

        if self.overlay_l2_coef > 0.0:
            rewards = rewards - self.overlay_l2_coef * next_inventory.pow(2)
        if self.short_l1_coef > 0.0:
            rewards = rewards - self.short_l1_coef * F.relu(-next_inventory)
        if self.overweight_l1_coef > 0.0:
            rewards = rewards - self.overweight_l1_coef * F.relu(next_inventory)

        abs_position = next_inventory + self.benchmark_position
        if self.abs_exposure_l2_coef > 0.0:
            rewards = rewards - self.abs_exposure_l2_coef * abs_position.pow(2)
        risk_state = self._risk_state_from_advantage(advantage0, next_inventory)
        if self.risk_state_exposure_coef > 0.0:
            rewards = rewards - self.risk_state_exposure_coef * risk_state * abs_position.pow(2)
        if self.risk_tilt_coef > 0.0:
            rewards = rewards - self.risk_tilt_coef * risk_state * next_inventory
        edge_state = self._signed_state_from_advantage(
            advantage0,
            next_inventory,
            self.edge_state_indices,
            center=self.edge_state_center,
            scale=self.edge_state_scale,
        )
        if self.edge_overlay_coef > 0.0:
            rewards = rewards + self.edge_overlay_coef * edge_state * next_inventory

        terminal_excess_mean = cum_excess[:, -1].detach().mean().item() if cum_excess.shape[1] > 0 else 0.0
        terminal_dd_delta = dd_delta[:, -1].detach().mean().item() if dd_delta.shape[1] > 0 else 0.0
        diagnostics = {
            "bac_excess": float(excess_logwealth.detach().mean().item()),
            "bac_strategy_dd": float(strategy_dd.detach().mean().item()),
            "bac_benchmark_dd": float(benchmark_dd.detach().mean().item()),
            "bac_dd_delta": float(dd_delta.detach().mean().item()),
            "bac_terminal_excess": float(terminal_excess_mean),
            "bac_terminal_dd_delta": float(terminal_dd_delta),
            "rb_logwealth": float(excess_logwealth.detach().mean().item()),
            "rb_dd": float(dd_delta.detach().mean().item()),
            "rb_risk_state": float(risk_state.detach().mean().item()),
            "rb_edge_state": float(edge_state.detach().mean().item()),
            "rb_abs_exposure": float(abs_position.detach().abs().mean().item()),
        }
        return rewards, diagnostics

    def _relative_constraint_rewards(
        self,
        *,
        net_returns: torch.Tensor,
        next_inventory: torch.Tensor,
        rewards_norm: torch.Tensor,
        advantage0: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """B&H relative overlay 用の制約付き報酬を作る。

        reward_mode=excess_bh の WM reward を前提に、rollout 内の相対
        log wealth を主目的、相対DD/終端alpha shortfallを制約として扱う。
        """
        reward_scale = max(float(self.reward_ema.scale), 1e-8)
        horizon = max(1, int(net_returns.shape[1]))
        excess_logwealth = torch.log1p(net_returns.clamp(min=-0.95))
        rewards = self.logwealth_coef * (excess_logwealth / reward_scale)
        if self.logwealth_coef == 0.0:
            rewards = rewards_norm

        cum_excess = excess_logwealth.cumsum(dim=1)
        zero = torch.zeros_like(cum_excess[:, :1])
        path = torch.cat([zero, cum_excess], dim=1)
        running_peak = path.cummax(dim=1).values[:, 1:]
        relative_dd = (running_peak - cum_excess).clamp(min=0.0)

        if self.relative_dd_coef > 0.0:
            dd_excess = F.relu(relative_dd - self.relative_dd_budget)
            rewards = rewards - self.relative_dd_coef * (dd_excess / reward_scale)
        if self.relative_terminal_dd_coef > 0.0 and relative_dd.shape[1] > 0:
            terminal_dd = relative_dd[:, -1:].expand_as(relative_dd)
            rewards = rewards - self.relative_terminal_dd_coef * (terminal_dd / reward_scale)
        if self.alpha_floor_coef > 0.0 and cum_excess.shape[1] > 0:
            terminal_excess = cum_excess[:, -1:]
            alpha_shortfall = F.relu(self.alpha_floor - terminal_excess).expand_as(cum_excess)
            rewards = rewards - self.alpha_floor_coef * (alpha_shortfall / reward_scale) / float(horizon)
        if self.relative_cvar_coef > 0.0:
            tail = F.relu((-net_returns) - self.tail_margin)
            rewards = rewards - self.relative_cvar_coef * (tail / reward_scale)

        if self.overlay_l2_coef > 0.0:
            rewards = rewards - self.overlay_l2_coef * next_inventory.pow(2)
        if self.short_l1_coef > 0.0:
            rewards = rewards - self.short_l1_coef * F.relu(-next_inventory)
        if self.overweight_l1_coef > 0.0:
            rewards = rewards - self.overweight_l1_coef * F.relu(next_inventory)

        abs_position = next_inventory + self.benchmark_position
        if self.abs_exposure_l2_coef > 0.0:
            rewards = rewards - self.abs_exposure_l2_coef * abs_position.pow(2)
        risk_state = self._risk_state_from_advantage(advantage0, next_inventory)
        if self.risk_state_exposure_coef > 0.0:
            rewards = rewards - self.risk_state_exposure_coef * risk_state * abs_position.pow(2)
        if self.risk_tilt_coef > 0.0:
            rewards = rewards - self.risk_tilt_coef * risk_state * next_inventory
        edge_state = self._signed_state_from_advantage(
            advantage0,
            next_inventory,
            self.edge_state_indices,
            center=self.edge_state_center,
            scale=self.edge_state_scale,
        )
        if self.edge_overlay_coef > 0.0:
            rewards = rewards + self.edge_overlay_coef * edge_state * next_inventory

        terminal_excess_mean = cum_excess[:, -1].detach().mean().item() if cum_excess.shape[1] > 0 else 0.0
        alpha_shortfall_mean = (
            F.relu(self.alpha_floor - cum_excess[:, -1]).detach().mean().item()
            if cum_excess.shape[1] > 0
            else 0.0
        )
        diagnostics = {
            "rc_excess": float(excess_logwealth.detach().mean().item()),
            "rc_rel_dd": float(relative_dd.detach().mean().item()),
            "rc_terminal_excess": float(terminal_excess_mean),
            "rc_alpha_shortfall": float(alpha_shortfall_mean),
            "rb_logwealth": float(excess_logwealth.detach().mean().item()),
            "rb_dd": float(relative_dd.detach().mean().item()),
            "rb_risk_state": float(risk_state.detach().mean().item()),
            "rb_edge_state": float(edge_state.detach().mean().item()),
            "rb_abs_exposure": float(abs_position.detach().abs().mean().item()),
        }
        return rewards, diagnostics

    def _bc_loss_batch(self, batch_size: int = 128) -> torch.Tensor:
        """Oracle データからランダムサンプルして BC 損失を計算する."""
        if self._oracle_z is None:
            return torch.tensor(0.0, device=self.device)

        T = self._oracle_z.shape[0]
        idx = torch.randint(0, T, (min(batch_size, T),), device=self.device)
        regime_batch = self._oracle_regime[idx] if self._oracle_regime is not None else None
        advantage_batch = self._oracle_advantage[idx] if self._oracle_advantage is not None else None
        inventory_batch = self._oracle_inventory[idx] if self._oracle_inventory is not None else None
        trade_logits, target_mean, target_std, band_width, current_inventory = self.actor.controller_outputs(
            self._oracle_z[idx],
            self._oracle_h[idx],
            inventory=inventory_batch,
            regime=regime_batch,
            advantage=advantage_batch,
        )
        oracle_pos = self._oracle_positions[idx].to(device=self.device, dtype=current_inventory.dtype)
        oracle_overlay = oracle_pos - self.benchmark_position
        target_gap = torch.abs(oracle_overlay - current_inventory)
        trade_targets = (target_gap > 1e-8).float()
        target_dist = self.actor.target_distribution(target_mean, target_std)
        target_loss = -target_dist.log_prob(oracle_overlay)
        if self._oracle_trade_pos_weight is not None:
            target_w = torch.where(
                trade_targets > 0.5,
                self._oracle_trade_pos_weight.to(device=target_loss.device, dtype=target_loss.dtype),
                torch.ones_like(target_loss),
            )
            target_loss = target_loss * target_w

        loss = self.target_aux_coef * target_loss.mean()
        if self.trade_aux_coef > 0.0:
            trade_pred = torch.sigmoid(trade_logits)
            trade_loss = F.smooth_l1_loss(trade_pred, trade_targets)
            if self._oracle_trade_pos_weight is not None:
                trade_w = torch.where(
                    trade_targets > 0.5,
                    self._oracle_trade_pos_weight.to(device=trade_pred.device, dtype=trade_pred.dtype),
                    torch.ones_like(trade_pred),
                )
                trade_loss = (F.smooth_l1_loss(trade_pred, trade_targets, reduction="none") * trade_w).mean()
            loss = loss + self.trade_aux_coef * trade_loss
            if self.band_aux_coef > 0.0:
                trade_margin = 0.05
                hold_band_min = 0.05
                trade_penalty = F.softplus(band_width - (target_gap - trade_margin).clamp(min=0.0))
                hold_penalty = F.softplus(hold_band_min - band_width)
                band_penalty = torch.where(trade_targets > 0.5, trade_penalty, hold_penalty)
                if self._oracle_trade_pos_weight is not None:
                    band_penalty = torch.where(
                        trade_targets > 0.5,
                        band_penalty * self._oracle_trade_pos_weight.to(
                            device=band_penalty.device, dtype=band_penalty.dtype
                        ),
                        band_penalty,
                    )
                band_loss = band_penalty.mean()
                loss = loss + self.band_aux_coef * band_loss
        if self.execution_aux_coef > 0.0:
            pred_next_inventory = self.actor.soft_execute_controller(
                trade_signal=torch.sigmoid(trade_logits),
                target_inventory=target_mean,
                band_width=band_width,
                current_inventory=current_inventory,
            )
            exec_loss = F.smooth_l1_loss(pred_next_inventory, oracle_overlay)
            loss = loss + self.execution_aux_coef * exec_loss
        return loss

    def _prior_anchor_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
        advantage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BC 初期 policy からの trust-region 正則化."""
        if (
            self.prior_kl_coef <= 0.0
            and self.prior_trade_coef <= 0.0
            and self.prior_band_coef <= 0.0
            and self.prior_flow_coef <= 0.0
        ):
            return torch.tensor(0.0, device=self.device)

        cur_trade_logits, cur_target_mean, cur_target_std, cur_band, _ = self.actor.controller_outputs(
            z, h, inventory=inventory, regime=regime, advantage=advantage
        )
        with torch.no_grad():
            ref_trade_logits, ref_target_mean, ref_target_std, ref_band, _ = self.actor_prior.controller_outputs(
                z, h, inventory=inventory, regime=regime, advantage=advantage
            )

        loss = torch.tensor(0.0, device=self.device)
        if self.prior_kl_coef > 0.0:
            cur_target_dist = self.actor.target_distribution(cur_target_mean, cur_target_std)
            ref_target_dist = self.actor_prior.target_distribution(ref_target_mean, ref_target_std)
            target_kl = torch.distributions.kl_divergence(ref_target_dist, cur_target_dist).mean()
            loss = loss + self.prior_kl_coef * target_kl
        if self.prior_trade_coef > 0.0:
            ref_trade_prob = torch.sigmoid(ref_trade_logits)
            trade_anchor = F.binary_cross_entropy_with_logits(cur_trade_logits, ref_trade_prob)
            loss = loss + self.prior_trade_coef * trade_anchor
        if self.prior_band_coef > 0.0:
            band_anchor = F.smooth_l1_loss(cur_band, ref_band)
            loss = loss + self.prior_band_coef * band_anchor
        if self.prior_flow_coef > 0.0:
            cur_trade_prob = torch.sigmoid(cur_trade_logits)
            ref_trade_prob = torch.sigmoid(ref_trade_logits)
            cur_target_inventory = cur_target_mean
            ref_target_inventory = ref_target_mean
            inventory_now = inventory.squeeze(-1) if inventory.ndim > 1 else inventory
            cur_next_inventory = self.actor.soft_execute_controller(
                trade_signal=cur_trade_prob,
                target_inventory=cur_target_inventory,
                band_width=cur_band,
                current_inventory=inventory_now,
            )
            ref_next_inventory = self.actor_prior.soft_execute_controller(
                trade_signal=ref_trade_prob,
                target_inventory=ref_target_inventory,
                band_width=ref_band,
                current_inventory=inventory_now,
            )
            cur_flow = cur_next_inventory - inventory_now
            ref_flow = ref_next_inventory - inventory_now
            flow_anchor = F.smooth_l1_loss(cur_flow, ref_flow)
            loss = loss + self.prior_flow_coef * flow_anchor
        return loss

    def _nearest_oracle_anchor_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        inventory: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
        advantage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Nearest-dataset action anchor, inspired by dataset-constrained offline RL."""
        if (
            (self.nn_anchor_coef <= 0.0 and self.nn_anchor_flow_coef <= 0.0)
            or self._oracle_anchor_h is None
            or self._oracle_anchor_inventory is None
            or self._oracle_anchor_overlay is None
        ):
            return torch.tensor(0.0, device=self.device)

        cur_trade_logits, cur_target_mean, _, cur_band, _ = self.actor.controller_outputs(
            z, h, inventory=inventory, regime=regime, advantage=advantage
        )

        query_parts = [F.normalize(h, dim=-1), inventory]
        bank_parts = [
            F.normalize(self._oracle_anchor_h, dim=-1),
            self._oracle_anchor_inventory.unsqueeze(-1),
        ]
        if regime is not None and self._oracle_anchor_regime is not None:
            query_parts.append(regime)
            bank_parts.append(self._oracle_anchor_regime)
        if advantage is not None and self._oracle_anchor_advantage is not None:
            query_parts.append(advantage)
            bank_parts.append(self._oracle_anchor_advantage)
        query = torch.cat(query_parts, dim=-1)
        bank = torch.cat(bank_parts, dim=-1)

        with torch.no_grad():
            dist = torch.cdist(query, bank)
            nn_idx = dist.argmin(dim=-1)
            anchor_overlay = self._oracle_anchor_overlay.index_select(0, nn_idx)

        loss = torch.tensor(0.0, device=self.device)
        if self.nn_anchor_coef > 0.0:
            target_anchor = F.smooth_l1_loss(cur_target_mean, anchor_overlay)
            loss = loss + self.nn_anchor_coef * target_anchor
        if self.nn_anchor_flow_coef > 0.0:
            cur_trade_prob = torch.sigmoid(cur_trade_logits)
            inventory_now = inventory.squeeze(-1) if inventory.ndim > 1 else inventory
            cur_next_inventory = self.actor.soft_execute_controller(
                trade_signal=cur_trade_prob,
                target_inventory=cur_target_mean,
                band_width=cur_band,
                current_inventory=inventory_now,
            )
            flow_anchor = F.smooth_l1_loss(cur_next_inventory, anchor_overlay)
            loss = loss + self.nn_anchor_flow_coef * flow_anchor
        return loss

    def train_step(
        self,
        z0: torch.Tensor,
        h0: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
        regime0: Optional[torch.Tensor] = None,
        advantage0: Optional[torch.Tensor] = None,
        controller_state0: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """1 ステップの Actor-Critic 更新."""
        B = z0.shape[0]
        if past_as is not None and past_as.shape[1] > 0:
            inventory0 = past_as[:, -1] - self.benchmark_position
            if inventory0.ndim == 1:
                inventory0 = inventory0.unsqueeze(-1)
        else:
            inventory0 = torch.zeros(B, 1, dtype=z0.dtype, device=z0.device)

        if controller_state0 is not None:
            if not getattr(self, "market_reward_mode", False):
                raise ValueError("controller_state0 is only supported by market_log_return")
            if controller_state0.shape != (B, 4) or not controller_state0.is_floating_point() \
                    or not bool(torch.isfinite(controller_state0).all()):
                raise ValueError("controller_state0 must be finite B x 4")
            inventory0 = controller_state0.detach().to(z0)

        # --- Imagination rollout ---
        rollout = self._imagination_rollout(
            z0,
            h0,
            past_zs,
            past_as,
            inventory0=inventory0,
            regime0=regime0,
            advantage0=advantage0,
        )

        zs = rollout["zs"]            # (B, H, z_dim)
        hs = rollout["hs"]            # (B, H, h_dim)
        inventories = rollout["inventories"]  # (B, H)
        net_returns = torch.nan_to_num(rollout["rewards"], nan=0.0, posinf=0.0, neginf=0.0)  # (B, H) 原スケール（WM 予測の net_return）
        dones = torch.nan_to_num(rollout["dones"], nan=1.0, posinf=1.0, neginf=0.0)      # (B, H)
        next_inventory = torch.nan_to_num(rollout["actions"], nan=0.0, posinf=self.abs_max_position - self.benchmark_position, neginf=self.abs_min_position - self.benchmark_position)
        delta_inventory = torch.abs(next_inventory - inventories)
        flow_change = torch.zeros_like(delta_inventory)
        if self.horizon > 1:
            flow_change[:, 1:] = torch.abs(delta_inventory[:, 1:] - delta_inventory[:, :-1])
        log_probs = rollout["log_probs"]
        entropies = rollout["entropies"]
        last_z = rollout["last_z"]
        last_h = rollout["last_h"]

        # --- 報酬計算 ---
        reward_diag: dict[str, float] = {}
        benchmark_returns = None
        if self.reward_objective in {"benchmark_absolute_constraint", "absolute_bh_constraint", "final_metric"}:
            if getattr(self, "market_reward_mode", False):
                benchmark_returns = rollout["benchmark_rewards"]
            else:
                benchmark_returns = self._benchmark_rollout_rewards(
                    z0,
                    h0,
                    past_zs,
                    past_as,
                )
            reward_basis = net_returns - benchmark_returns
        else:
            reward_basis = net_returns
        self.reward_ema.update(reward_basis)
        rewards_norm = reward_basis / self.reward_ema.scale
        if self.reward_objective in {"risk_budget", "risk_budget_overlay"}:
            rewards_for_ac, reward_diag = self._risk_budget_rewards(
                net_returns=net_returns,
                next_inventory=next_inventory,
                rewards_norm=rewards_norm,
                advantage0=advantage0,
            )
        elif self.reward_objective in {"relative_constraint", "bh_relative_constraint", "constrained_relative"}:
            rewards_for_ac, reward_diag = self._relative_constraint_rewards(
                net_returns=net_returns,
                next_inventory=next_inventory,
                rewards_norm=rewards_norm,
                advantage0=advantage0,
            )
        elif self.reward_objective in {"benchmark_absolute_constraint", "absolute_bh_constraint", "final_metric"}:
            rewards_for_ac, reward_diag = self._benchmark_absolute_constraint_rewards(
                strategy_returns=net_returns,
                benchmark_returns=benchmark_returns,
                next_inventory=next_inventory,
                rewards_norm=rewards_norm,
                advantage0=advantage0,
            )
        elif self.use_dsr:
            rewards_for_ac = self._compute_dsr_rewards(net_returns)
        else:
            # SPEC 準拠の報酬: R_t ≈ net_return / EMA_scale - β·DD_t
            self.reward_ema.update(net_returns)
            rewards_norm = net_returns / self.reward_ema.scale          # EMA 正規化（DSR の近似）
            drawdown = self._compute_drawdown(net_returns)
            rewards_for_ac = rewards_norm - self.beta * drawdown        # (B, H) 原スケール

        if self.turnover_coef > 0.0:
            rewards_for_ac = rewards_for_ac - self.turnover_coef * delta_inventory
        if self.flow_change_coef > 0.0:
            rewards_for_ac = rewards_for_ac - self.flow_change_coef * flow_change
        if self.active_deviation_coef > 0.0:
            rewards_for_ac = rewards_for_ac - self.active_deviation_coef * next_inventory.abs()
        if self.underweight_exposure_coef > 0.0:
            underweight_excess = F.relu((-next_inventory) - float(self.underweight_floor))
            rewards_for_ac = rewards_for_ac - self.underweight_exposure_coef * underweight_excess
        if self.upside_miss_coef > 0.0 or self.downside_hedge_coef > 0.0:
            underweight_size = F.relu(-next_inventory)
            upside_returns = F.relu(rewards_norm)
            downside_returns = F.relu(-rewards_norm)
            if self.upside_miss_coef > 0.0:
                rewards_for_ac = rewards_for_ac - self.upside_miss_coef * upside_returns * underweight_size
            if self.downside_hedge_coef > 0.0:
                rewards_for_ac = rewards_for_ac + self.downside_hedge_coef * downside_returns * underweight_size

        # --- Slow Critic の value 推定（原スケール）---
        zs_flat = zs.reshape(B * self.horizon, -1)
        hs_flat = hs.reshape(B * self.horizon, -1)
        with torch.no_grad():
            values_flat = self.critic.slow_value(zs_flat, hs_flat, self.bins)  # 原スケール
            values = values_flat.reshape(B, self.horizon)
            last_val = self.critic.slow_value(last_z, last_h, self.bins)       # 原スケール

        # --- λ-return（symlog 空間）---
        # ★ rewards_for_ac / values / last_val はすべて原スケールで渡す
        returns = self._compute_lambda_returns(
            rewards_for_ac,  # 原スケール → 内部で symlog
            values,          # 原スケール → 内部で symlog
            last_val,        # 原スケール → 内部で symlog
            dones,
        )
        # returns は symlog 空間 (B, H)

        # --- Critic 損失（twohot cross-entropy）---
        critic_logits_flat = self.critic(zs_flat.detach(), hs_flat.detach())
        critic_logits = critic_logits_flat.reshape(B, self.horizon, -1)

        targets_symlog = returns.detach()  # symlog 空間のターゲット
        critic_loss = torch.tensor(0.0, device=self.device)
        for t in range(self.horizon):
            target_twohot = twohot_encode(targets_symlog[:, t], self.bins)
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

        # Advantage = λ-return（symlog）- value（symlog）
        # values は原スケールなので symlog 変換してから引く
        # symlog(0)=0 なので ε 加算は不要（バイアスを避ける）
        advantage = (returns - symlog(values)).detach()            # symlog 空間
        adv_scale = advantage.abs().mean().item()
        self._adv_ema = 0.99 * self._adv_ema + 0.01 * adv_scale
        norm_q = self.td3bc_alpha / max(self._adv_ema, 0.1)

        pg_advantage = F.relu(advantage) if self.positive_advantages else advantage
        ac_loss = -(norm_q * pg_advantage * log_probs).mean() - self.entropy_scale * entropies.mean()
        prior_loss = self._prior_anchor_loss(z0, h0, inventory0, regime=regime0, advantage=advantage0)
        nn_anchor_loss = self._nearest_oracle_anchor_loss(z0, h0, inventory0, regime=regime0, advantage=advantage0)
        bc_loss = self._bc_loss_batch()
        bc_component = alpha * bc_loss
        ac_component = (1.0 - alpha) * ac_loss
        actor_loss = bc_component + ac_component + prior_loss + nn_anchor_loss

        gradient_diag: dict[str, float] = {}
        if (
            self.gradient_diagnostics_interval > 0
            and (self.global_step + 1) % self.gradient_diagnostics_interval == 0
        ):
            gradient_diag = self._actor_gradient_diagnostics(bc_component, ac_component)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.global_step += 1

        return {
            "actor_loss": actor_loss.item(),
            "ac_loss": ac_loss.item(),
            "bc_loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else float(bc_loss),
            "prior_loss": prior_loss.item() if isinstance(prior_loss, torch.Tensor) else float(prior_loss),
            "nn_anchor_loss": nn_anchor_loss.item() if isinstance(nn_anchor_loss, torch.Tensor) else float(nn_anchor_loss),
            "critic_loss": critic_loss.item(),
            "entropy": entropies.mean().item(),
            "alpha": alpha,
            "reward_mean": net_returns.mean().item(),
            "reward_scale": self.reward_ema.scale,
            **gradient_diag,
            **reward_diag,
        }

    def pretrain_critic(
        self,
        encoded_sequences: list[dict],
        n_steps: int = 2000,
        batch_size: int = 32,
    ) -> None:
        """Actor を固定して Critic だけ事前学習する (Actor-Critic Alignment).

        BC 後・AC 前に呼び出す。Critic が収束してから Actor の更新を開始することで
        advantage 推定の不安定さによる Actor 崩壊を防ぐ。
        """
        all_z = np.concatenate([s["z"] for s in encoded_sequences], axis=0)
        all_h = np.concatenate([s["h"] for s in encoded_sequences], axis=0)
        T_total = len(all_z)
        z_dim = all_z.shape[1]
        L = self.context_len

        prev_requires_grad = [p.requires_grad for p in self.actor.parameters()]
        for p in self.actor.parameters():
            p.requires_grad_(False)

        print(f"[AC] Critic pre-training ({n_steps} steps, actor frozen)...")
        log_every = max(1, n_steps // 5)
        last_loss = 0.0

        for step in range(n_steps):
            idx = np.random.randint(0, T_total, size=batch_size)
            z0 = torch.tensor(all_z[idx], dtype=torch.float32, device=self.device)
            h0 = torch.tensor(all_h[idx], dtype=torch.float32, device=self.device)

            past_zs_np = np.zeros((batch_size, L, z_dim), dtype=np.float32)
            past_as_np = np.full((batch_size, L, 1), self.benchmark_position, dtype=np.float32)
            for b, i in enumerate(idx):
                start = max(0, i - L)
                length = i - start
                if length > 0:
                    past_zs_np[b, L - length:] = all_z[start:i]
            past_zs = torch.tensor(past_zs_np, device=self.device)
            past_as = torch.tensor(past_as_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                inventory0 = past_as[:, -1]
                if getattr(self, "market_reward_mode", False):
                    inventory0 = inventory0 - self.benchmark_position
                rollout = self._imagination_rollout(
                    z0, h0, past_zs, past_as, inventory0=inventory0
                )

            zs = rollout["zs"]
            hs = rollout["hs"]
            net_returns = rollout["rewards"]
            dones = rollout["dones"]
            last_z = rollout["last_z"]
            last_h = rollout["last_h"]
            B = z0.shape[0]

            if getattr(self, "market_reward_mode", False):
                benchmark_returns = rollout["benchmark_rewards"]
                basis = net_returns - benchmark_returns
                self.reward_ema.update(basis)
                rewards_norm = basis / self.reward_ema.scale
                rewards_for_ac, _ = self._benchmark_absolute_constraint_rewards(
                    strategy_returns=net_returns, benchmark_returns=benchmark_returns,
                    next_inventory=rollout["actions"], rewards_norm=rewards_norm,
                    advantage0=None,
                )
                delta = (rollout["actions"] - rollout["inventories"]).abs()
                flow_change = torch.zeros_like(delta)
                flow_change[:, 1:] = (delta[:, 1:] - delta[:, :-1]).abs()
                rewards_for_ac = rewards_for_ac - self.turnover_coef * delta - self.flow_change_coef * flow_change
                rewards_for_ac = rewards_for_ac - self.active_deviation_coef * rollout["actions"].abs()
                rewards_for_ac = rewards_for_ac - self.underweight_exposure_coef * F.relu(-rollout["actions"] - self.underweight_floor)
                underweight = F.relu(-rollout["actions"])
                rewards_for_ac = rewards_for_ac - self.upside_miss_coef * F.relu(rewards_norm) * underweight
                rewards_for_ac = rewards_for_ac + self.downside_hedge_coef * F.relu(-rewards_norm) * underweight
            else:
                self.reward_ema.update(net_returns)
                rewards_norm = net_returns / self.reward_ema.scale
                drawdown = self._compute_drawdown(net_returns)
                rewards_for_ac = rewards_norm - self.beta * drawdown

            zs_flat = zs.reshape(B * self.horizon, -1)
            hs_flat = hs.reshape(B * self.horizon, -1)
            with torch.no_grad():
                values_flat = self.critic.slow_value(zs_flat, hs_flat, self.bins)
                values = values_flat.reshape(B, self.horizon)
                last_val = self.critic.slow_value(last_z, last_h, self.bins)

            returns = self._compute_lambda_returns(rewards_for_ac, values, last_val, dones)

            critic_logits_flat = self.critic(zs_flat.detach(), hs_flat.detach())
            critic_logits = critic_logits_flat.reshape(B, self.horizon, -1)
            targets_symlog = returns.detach()
            critic_loss = torch.tensor(0.0, device=self.device)
            for t in range(self.horizon):
                target_twohot = twohot_encode(targets_symlog[:, t], self.bins)
                log_p = F.log_softmax(critic_logits[:, t], dim=-1)
                critic_loss = critic_loss - (target_twohot * log_p).sum(-1).mean()
            critic_loss = critic_loss / self.horizon

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()
            self.critic.update_slow_target()
            last_loss = critic_loss.item()

            if (step + 1) % log_every == 0:
                print(f"[AC] Critic pretrain step {step+1}/{n_steps} | Loss: {last_loss:.4f}")

        for p, requires_grad in zip(self.actor.parameters(), prev_requires_grad):
            p.requires_grad_(requires_grad)
        self._apply_actor_trainable_mask()
        print(f"[AC] Critic pre-training done.")

    def train(
        self,
        encoded_sequences: list[dict],
        max_steps: Optional[int] = None,
        batch_size: int = 32,
        checkpoint_path: Optional[str] = None,
        val_eval_fn=None,
        checkpoint_eval_fn=None,
        val_baseline_sharpe: float = -float("inf"),
        online_wm_callback=None,
    ) -> list[dict]:
        """学習ループを実行する."""
        max_steps = max_steps or self.max_steps
        logs = []

        all_z = np.concatenate([s["z"] for s in encoded_sequences], axis=0)
        all_h = np.concatenate([s["h"] for s in encoded_sequences], axis=0)
        T_total = len(all_z)
        z_dim = all_z.shape[1]
        L = self.context_len

        # Regime 配列の抽出
        has_regime = all(
            "regime" in s and s["regime"] is not None for s in encoded_sequences
        )
        if has_regime:
            all_regime = np.concatenate([s["regime"] for s in encoded_sequences], axis=0)
        else:
            all_regime = None
        has_advantage = all(
            "advantage" in s and s["advantage"] is not None for s in encoded_sequences
        )
        if has_advantage:
            all_advantage = np.concatenate([s["advantage"] for s in encoded_sequences], axis=0)
        else:
            all_advantage = None

        # context action 配列: flat（no-action）で統一する。
        # oracle actions を使うと WM は oracle context 空間で imagination するが、
        # test 時は no-action context のため train/test 分布が大きくずれる。
        # no-action context に統一することで imagination の分布を test と揃える。
        context_actions_np = None

        # val Sharpe tracking for best checkpoint selection
        best_val_sharpe = val_baseline_sharpe
        best_ckpt_path = (
            checkpoint_path.replace(".pt", "_best.pt")
            if checkpoint_path is not None and val_eval_fn is not None
            else None
        )
        best_checkpoint_score = -float("inf")
        best_checkpoint_path = (
            checkpoint_path.replace(".pt", "_fire_best.pt")
            if checkpoint_path is not None and checkpoint_eval_fn is not None
            else None
        )
        if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
            os.remove(best_checkpoint_path)
        # AC が一度も BC を超えなかった場合の fallback として
        # 学習開始時点（BC 状態）を _best.pt に必ず保存する
        if best_ckpt_path is not None:
            self.save(best_ckpt_path)
        self.last_train_best_candidate = None

        # Early stop カウンター
        _no_improve_count = 0
        _bc_loss_exceed_count = 0

        while self.global_step < max_steps:
            idx = np.random.randint(0, T_total, size=batch_size)
            z0 = torch.tensor(all_z[idx], dtype=torch.float32, device=self.device)
            h0 = torch.tensor(all_h[idx], dtype=torch.float32, device=self.device)

            # Regime バッチ
            if all_regime is not None:
                regime0 = torch.tensor(
                    all_regime[idx], dtype=torch.float32, device=self.device
                )
            else:
                regime0 = None
            if all_advantage is not None:
                advantage0 = torch.tensor(
                    all_advantage[idx], dtype=torch.float32, device=self.device
                )
            else:
                advantage0 = None

            # 各サンプルの直前 L ステップを context として取得（左端はゼロパディング）
            past_zs_np = np.zeros((batch_size, L, z_dim), dtype=np.float32)
            past_as_np = np.full((batch_size, L, 1), self.benchmark_position, dtype=np.float32)
            for b, i in enumerate(idx):
                start = max(0, i - L)
                length = i - start
                if length > 0:
                    past_zs_np[b, L - length:] = all_z[start:i]
                    if context_actions_np is not None:
                        act_end = min(i, len(context_actions_np))
                        act_start = max(0, act_end - length)
                        past_as_np[b, L - (act_end - act_start):, 0] = context_actions_np[act_start:act_end]
            past_zs = torch.tensor(past_zs_np, device=self.device)
            past_as = torch.tensor(past_as_np, dtype=torch.float32, device=self.device)

            step_log = self.train_step(
                z0,
                h0,
                past_zs=past_zs,
                past_as=past_as,
                regime0=regime0,
                advantage0=advantage0,
            )
            logs.append({"step": self.global_step, **step_log})
            self.loss_history.append(logs[-1])

            if self.global_step % self.log_interval == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                rb_bits = ""
                if "rb_dd" in step_log:
                    rb_bits = (
                        f" | rb_dd={step_log['rb_dd']:.5f}"
                        f" risk={step_log['rb_risk_state']:.3f}"
                        f" edge={step_log.get('rb_edge_state', 0.0):+.3f}"
                        f" exp={step_log['rb_abs_exposure']:.3f}"
                    )
                print(
                    f"[{ts}] [AC] Step {self.global_step}/{max_steps} | "
                    f"Actor: {step_log['actor_loss']:.4f} | "
                    f"AC: {step_log['ac_loss']:.4f} | "
                    f"BC: {step_log['bc_loss']:.4f} | "
                    f"Critic: {step_log['critic_loss']:.4f} | "
                    f"α={step_log['alpha']:.3f}"
                    f"{rb_bits}"
                )

            # Online WM callback
            if (
                online_wm_callback is not None
                and self.online_wm_interval > 0
                and self.global_step % self.online_wm_interval == 0
            ):
                online_wm_callback(self.global_step)

            if (
                checkpoint_path is not None
                and self.checkpoint_interval > 0
                and self.global_step % self.checkpoint_interval == 0
            ):
                self.save(checkpoint_path)
                print(f"[AC] Checkpoint saved: {checkpoint_path} (step={self.global_step})")
                if "bc_grad_norm" in step_log:
                    print(
                        "[AC] Gradient diag: "
                        f"BC={step_log['bc_grad_norm']:.4e} "
                        f"AC={step_log['ac_grad_norm']:.4e} "
                        f"cos={step_log['bc_ac_grad_cosine']:+.3f}"
                    )
                if self.save_step_checkpoints:
                    base_dir = os.path.dirname(checkpoint_path)
                    step_path = os.path.join(
                        base_dir,
                        f"{self.step_checkpoint_prefix}{self.global_step}.pt",
                    )
                    self.save(step_path)
                    print(f"[AC] Step checkpoint saved: {step_path}")

                # --- Train 行動分布ログ（oracle z/h 上の greedy 予測）---
                if self._oracle_z is not None:
                    n_sample = min(5000, self._oracle_z.shape[0])
                    _tr_pos = self.actor.predict_positions(
                        self._oracle_z[:n_sample].cpu().numpy(),
                        self._oracle_h[:n_sample].cpu().numpy(),
                        regime_np=(
                            self._oracle_regime[:n_sample].cpu().numpy()
                            if self._oracle_regime is not None else None
                        ),
                        advantage_np=(
                            self._oracle_advantage[:n_sample].cpu().numpy()
                            if self._oracle_advantage is not None else None
                        ),
                        device=str(self.device),
                    )
                    _tr_s = _action_stats(_tr_pos, benchmark_position=self.benchmark_position)
                    print(f"[AC] Train dist: {_fmt_action_stats(_tr_s)}")
                    _ac_alerts_ascii(
                        f"train/step{self.global_step}",
                        _tr_s,
                        bc_loss=step_log.get("bc_loss"),
                    )

                # BC loss early stop チェック
                cur_bc_loss = step_log["bc_loss"]
                if self.bc_loss_threshold > 0:
                    if cur_bc_loss > self.bc_loss_threshold:
                        _bc_loss_exceed_count += 1
                    else:
                        _bc_loss_exceed_count = 0
                    if _bc_loss_exceed_count >= self.bc_loss_patience:
                        print(f"[AC] Early stop: BC loss {cur_bc_loss:.4f} > {self.bc_loss_threshold} "
                              f"for {_bc_loss_exceed_count} consecutive checkpoints")
                        break

                if checkpoint_eval_fn is not None:
                    checkpoint_result = checkpoint_eval_fn()
                    if isinstance(checkpoint_result, tuple):
                        checkpoint_score, checkpoint_label = checkpoint_result
                        checkpoint_accepted = True
                    else:
                        checkpoint_score = float(checkpoint_result["score"])
                        checkpoint_label = str(checkpoint_result.get("label", checkpoint_score))
                        checkpoint_accepted = bool(checkpoint_result.get("accepted", True))
                    marker = ""
                    if (
                        checkpoint_accepted
                        and best_checkpoint_path is not None
                        and checkpoint_score > best_checkpoint_score
                    ):
                        best_checkpoint_score = checkpoint_score
                        self.save(best_checkpoint_path)
                        marker = " ★ fire_best"
                    print(f"[AC] Fire Selector: {checkpoint_label}{marker}")

                if val_eval_fn is not None:
                    val_result = val_eval_fn()
                    if isinstance(val_result, tuple):
                        val_sharpe, val_label = val_result
                    else:
                        val_sharpe, val_label = val_result, f"{val_result:.3f}"
                    marker = ""
                    if val_sharpe > best_val_sharpe:
                        best_val_sharpe = val_sharpe
                        self.save(best_ckpt_path)
                        self.last_train_best_candidate = {
                            "score": float(val_sharpe),
                            "label": str(val_label),
                            "path": str(best_ckpt_path),
                        }
                        marker = " ★ best"
                        _no_improve_count = 0
                    else:
                        _no_improve_count += 1
                    print(f"[AC] Val Score: {val_label}{marker}")

                    # Val patience early stop チェック
                    if self.val_patience > 0 and _no_improve_count >= self.val_patience:
                        print(f"[AC] Early stop: val Sharpe no improvement for "
                              f"{_no_improve_count} consecutive checkpoints")
                        break

                    # Adaptive BC: 直前の val Sharpe と比較して alpha 減衰速度を調整
                    # tolerance=0.01: 微小変動で speed を下げ続けないようにする
                    if self.adaptive_bc and self._last_val_sharpe is not None:
                        if val_sharpe > self._last_val_sharpe + 0.01:
                            self._alpha_speed = min(self._alpha_speed * 1.2, 3.0)
                        elif val_sharpe < self._last_val_sharpe - 0.01:
                            self._alpha_speed = max(self._alpha_speed * 0.8, 0.3)
                    self._last_val_sharpe = val_sharpe

        # 最良 val checkpoint に復元
        if not self.restore_best_val_checkpoint:
            pass
        elif best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
            print(f"[AC] Restoring best fire checkpoint (score={best_checkpoint_score:.3f})")
            saved_step = self.global_step
            self.load(best_checkpoint_path)
            self.global_step = saved_step  # resume のため step は保持
        elif best_ckpt_path is not None and os.path.exists(best_ckpt_path):
            print(f"[AC] Restoring best val checkpoint (Sharpe={best_val_sharpe:.3f})")
            saved_step = self.global_step
            self.load(best_ckpt_path)
            self.global_step = saved_step  # resume のため step は保持

        return logs

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        atomic_torch_save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
            "adv_ema": self._adv_ema,
            "dsr_A": self._dsr_A,
            "dsr_B": self._dsr_B,
            "alpha_speed": self._alpha_speed,
            "max_alpha_t": self._max_alpha_t,
            "alpha_stage_start_step": self._alpha_stage_start_step,
            "actor_runtime_overrides": self.actor_runtime_overrides,
            "actor_runtime_defaults": self.actor_runtime_defaults,
            "checkpoint_metadata": self.checkpoint_metadata,
            **({"market_reward_contract": self.market_reward_contract}
               if getattr(self, "market_reward_contract", None) is not None else {}),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if getattr(self, "market_reward_mode", False) and (
            ckpt.get("market_reward_contract") != self.market_reward_contract
        ):
            raise ValueError("refusing legacy or mismatched AC market reward contract")
        incompatible = self.actor.load_state_dict(ckpt["actor"], strict=False)
        optional_missing = {
            "residual_head_a.weight",
            "residual_head_a.bias",
            "residual_head_b.weight",
            "residual_head_b.bias",
            "route_head.weight",
            "route_head.bias",
            "route_delta_head.weight",
            "route_delta_head.bias",
            "route_active_head.weight",
            "route_active_head.bias",
            "route_active_class_head.weight",
            "route_active_class_head.bias",
            "route_advantage_gate.weight",
            "benchmark_overweight_sizing_adapter.weight",
            "benchmark_overweight_sizing_adapter.bias",
            "ac_residual_adapter.weight",
            "ac_residual_adapter.bias",
            "inventory_recovery_head.weight",
            "inventory_recovery_head.bias",
        }
        missing = [key for key in incompatible.missing_keys if key not in optional_missing]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"AC checkpoint incompatibility while loading {path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._restore_actor_runtime_config(
            defaults=dict(ckpt.get("actor_runtime_defaults") or {}),
            overrides=dict(ckpt.get("actor_runtime_overrides") or {}),
        )
        self.checkpoint_metadata = dict(ckpt.get("checkpoint_metadata") or {})
        self.critic.load_state_dict(ckpt["critic"])
        try:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        except ValueError as exc:
            # Curriculum stages can change the trainable actor subset. The actor
            # weights are still valid; the stage runner rebuilds the optimizer.
            print(f"[AC] Actor optimizer state skipped while loading {path}: {exc}")
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self._adv_ema = ckpt.get("adv_ema", 1.0)
        self._dsr_A = ckpt.get("dsr_A", 0.0)
        self._dsr_B = ckpt.get("dsr_B", 1e-4)
        self._alpha_speed = ckpt.get("alpha_speed", 1.0)
        self._max_alpha_t = ckpt.get("max_alpha_t", 0.0)
        self._alpha_stage_start_step = ckpt.get("alpha_stage_start_step", 0)
