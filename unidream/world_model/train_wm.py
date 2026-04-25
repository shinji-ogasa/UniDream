"""世界モデル学習エントリポイント.

EnsembleWorldModel を WFO データ上で学習する。
損失: reconstruction + KL (free bits) + reward (twohot) + done (BCE)
     + IDM (Inverse Dynamics Model) auxiliary loss
     + N-step return prediction auxiliary loss
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unidream.data.dataset import SequenceDataset
from unidream.device import resolve_device
from unidream.world_model.ensemble import EnsembleWorldModel


class IDMHead(nn.Module):
    """Inverse Dynamics Model: (z_t, z_{t+1}) → action logits.

    エンコーダが行動識別に有用な情報を保持することを強制する。
    NeurIPS 2023 で BC 事前学習の表現学習に有効と示された手法。
    """

    def __init__(self, z_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t:  (B, T, z_dim)
            z_t1: (B, T, z_dim)
        Returns:
            logits: (B, T, n_actions)
        """
        return self.net(torch.cat([z_t, z_t1], dim=-1))


class ReturnHead(nn.Module):
    """N-step return prediction: (z_t, h_t) → scalar.

    WM の潜在表現に将来リターンの情報を埋め込む。
    """

    def __init__(self, z_dim: int, d_model: int, hidden: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + d_model, hidden),
            nn.ELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, z_dim)
            h: (B, T, d_model)
        Returns:
            pred: (B, T, out_dim)
        """
        return self.net(torch.cat([z, h], dim=-1))


class RegimeHead(nn.Module):
    """Regime probability prediction from latent state."""

    def __init__(self, z_dim: int, d_model: int, hidden: int, regime_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + d_model, hidden),
            nn.ELU(),
            nn.Linear(hidden, regime_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h], dim=-1))


def build_ensemble(obs_dim: int, cfg: dict) -> EnsembleWorldModel:
    """config dict から EnsembleWorldModel を構築する."""
    wm_cfg = cfg.get("world_model", {})
    return EnsembleWorldModel(
        n_models=wm_cfg.get("n_ensemble", 3),
        disagree_scale=wm_cfg.get("disagree_scale", 0.1),
        obs_dim=obs_dim,
        act_dim=cfg.get("actions", {}).get("dim", 1),
        n_categoricals=wm_cfg.get("n_categoricals", 32),
        n_classes=wm_cfg.get("n_classes", 32),
        d_model=wm_cfg.get("d_model", 512),
        n_heads=wm_cfg.get("n_heads", 8),
        n_layers=wm_cfg.get("n_layers", 4),
        d_ff=wm_cfg.get("d_ff", 2048),
        dropout=wm_cfg.get("dropout", 0.1),
        max_seq_len=wm_cfg.get("max_seq_len", 256),
        n_bins=wm_cfg.get("n_bins", 255),
        bin_low=wm_cfg.get("bin_range", [-20.0, 20.0])[0],
        bin_high=wm_cfg.get("bin_range", [-20.0, 20.0])[1],
        unimix_ratio=wm_cfg.get("unimix_ratio", 0.01),
        encoder_hidden=wm_cfg.get("encoder_hidden", 256),
        encoder_layers=wm_cfg.get("encoder_layers", 2),
    )


class WorldModelTrainer:
    """世界モデルの学習ループ.

    Args:
        ensemble: EnsembleWorldModel
        cfg: config 辞書
        device: 計算デバイス
    """

    def __init__(
        self,
        ensemble: EnsembleWorldModel,
        cfg: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.ensemble = ensemble
        self.device = torch.device(resolve_device(device))
        self.ensemble.to(self.device)
        cfg = cfg or {}
        wm_cfg = cfg.get("world_model", {})

        self.lr = wm_cfg.get("lr", 1e-4)
        self.batch_size = wm_cfg.get("batch_size", 32)
        self.max_steps = wm_cfg.get("max_steps", 100_000)
        self.grad_clip = wm_cfg.get("grad_clip", 100.0)
        self.log_interval = cfg.get("logging", {}).get("log_interval", 1000)

        # 損失ハイパーパラメータ
        self.free_bits = wm_cfg.get("free_bits", 1.0)
        self.dyn_scale = wm_cfg.get("dyn_scale", 0.5)
        self.rep_scale = wm_cfg.get("rep_scale", 0.1)
        self.recon_scale = wm_cfg.get("recon_scale", 1.0)
        self.reward_scale = wm_cfg.get("reward_scale", 1.0)
        self.done_scale = wm_cfg.get("done_scale", 1.0)

        # Auxiliary loss スケール
        self.idm_scale = wm_cfg.get("idm_scale", 0.0)
        self.return_scale = wm_cfg.get("return_scale", 0.0)
        self.return_horizon = wm_cfg.get("return_horizon", 10)
        self.return_horizons = [
            int(h) for h in wm_cfg.get("return_horizons", [self.return_horizon])
        ]
        self.return_include_current = bool(wm_cfg.get("return_include_current", True))
        self.return_target_scale = float(wm_cfg.get("return_target_scale", 1.0))
        self.vol_scale = float(wm_cfg.get("vol_scale", 0.0))
        self.drawdown_scale = float(wm_cfg.get("drawdown_scale", 0.0))
        self.risk_horizons = [
            int(h) for h in wm_cfg.get("risk_horizons", self.return_horizons)
        ]
        self.risk_target_scale = float(wm_cfg.get("risk_target_scale", 1.0))
        self.regime_aux_scale = wm_cfg.get("regime_aux_scale", 0.0)
        self.regime_dim = int(wm_cfg.get("regime_dim", 0))

        # コストパラメータ（net_return 計算に使用）
        costs_cfg = cfg.get("costs", {})
        self.cost_rate = (
            (costs_cfg.get("spread_bps", 5.0) / 10000) / 2
            + costs_cfg.get("fee_rate", 0.0004)
            + (costs_cfg.get("slippage_bps", 2.0) / 10000)
        )
        reward_cfg = cfg.get("reward", {})
        self.reward_mode = reward_cfg.get("mode", "absolute")
        self.benchmark_position = reward_cfg.get("benchmark_position", 1.0)

        # Auxiliary heads（スケール > 0 の場合のみ構築）
        z_dim = ensemble.get_z_dim()
        d_model = ensemble.get_d_model()
        n_actions = cfg.get("actions", {}).get("n", 5)
        self.action_values = torch.tensor(
            cfg.get("actions", {}).get("values", [-1.0, -0.5, 0.0, 0.5, 1.0]),
            dtype=torch.float32,
            device=self.device,
        )
        aux_params: list[nn.Parameter] = []

        if self.idm_scale > 0.0:
            self.idm_head = IDMHead(z_dim, hidden=256, n_actions=n_actions).to(self.device)
            aux_params.extend(self.idm_head.parameters())
        else:
            self.idm_head = None

        if self.return_scale > 0.0:
            self.return_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.return_horizons),
            ).to(self.device)
            aux_params.extend(self.return_head.parameters())
        else:
            self.return_head = None

        if self.vol_scale > 0.0:
            self.vol_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
            ).to(self.device)
            aux_params.extend(self.vol_head.parameters())
        else:
            self.vol_head = None

        if self.drawdown_scale > 0.0:
            self.drawdown_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
            ).to(self.device)
            aux_params.extend(self.drawdown_head.parameters())
        else:
            self.drawdown_head = None

        if self.regime_aux_scale > 0.0 and self.regime_dim > 0:
            self.regime_head = RegimeHead(z_dim, d_model, hidden=256, regime_dim=self.regime_dim).to(self.device)
            aux_params.extend(self.regime_head.parameters())
        else:
            self.regime_head = None

        self._all_params = list(self.ensemble.parameters()) + aux_params
        self.optimizer = torch.optim.Adam(
            self._all_params,
            lr=self.lr,
        )

        self.global_step = 0
        self.loss_history: list[dict] = []

    def _loader_options(self, num_workers: int) -> dict:
        workers = max(0, int(num_workers))
        return {
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": workers > 0,
        }

    def _compute_net_returns(
        self,
        actions: torch.Tensor,
        raw_returns: torch.Tensor,
    ) -> torch.Tensor:
        """行動インデックスと生リターンからネットリターン（コスト控除後）を計算する.

        net_return[t] = position[t] * raw_returns[t]
                        - cost_rate * |position[t] - position[t-1]|

        初期ポジション = 0.0（フラット）。

        Args:
            actions: (B, T, 1) position path
            raw_returns: (B, T) 生の対数リターン

        Returns:
            net_returns: (B, T) コスト控除後リターン
        """
        positions = actions.squeeze(-1)                             # (B, T)

        # 前ステップポジション（初期 = 0.0 = フラット）
        prev_positions = torch.cat([
            torch.zeros_like(positions[:, :1]),
            positions[:, :-1],
        ], dim=1)                                                    # (B, T)

        delta_pos = (positions - prev_positions).abs()
        costs = self.cost_rate * delta_pos                           # (B, T)
        net_returns = positions * raw_returns - costs               # (B, T)
        if self.reward_mode == "excess_bh":
            benchmark_returns = self.benchmark_position * raw_returns
            return net_returns - benchmark_returns
        return net_returns

    def _future_return_targets(
        self,
        raw_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build multi-horizon return targets and valid masks."""
        B, T = raw_returns.shape
        offset0 = 0 if self.return_include_current else 1
        targets = []
        masks = []
        for horizon in self.return_horizons:
            horizon = max(1, int(horizon))
            target = torch.zeros_like(raw_returns)
            for k in range(offset0, offset0 + horizon):
                if k < T:
                    target[:, : T - k] += raw_returns[:, k:]
            valid_len = T - (offset0 + horizon - 1)
            mask = torch.zeros((B, T), dtype=torch.bool, device=raw_returns.device)
            if valid_len > 0:
                mask[:, :valid_len] = True
            targets.append(target * self.return_target_scale)
            masks.append(mask)
        return torch.stack(targets, dim=-1), torch.stack(masks, dim=-1)

    def _future_risk_targets(
        self,
        raw_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build future realized volatility and drawdown-risk targets."""
        B, T = raw_returns.shape
        vol_targets = []
        dd_targets = []
        masks = []
        for horizon in self.risk_horizons:
            horizon = max(1, int(horizon))
            cum = torch.zeros_like(raw_returns)
            min_cum = torch.zeros_like(raw_returns)
            sq_sum = torch.zeros_like(raw_returns)
            for k in range(1, horizon + 1):
                if k < T:
                    shifted = torch.zeros_like(raw_returns)
                    shifted[:, : T - k] = raw_returns[:, k:]
                    cum = cum + shifted
                    min_cum = torch.minimum(min_cum, cum)
                    sq_sum = sq_sum + shifted.square()
            valid_len = T - horizon
            mask = torch.zeros((B, T), dtype=torch.bool, device=raw_returns.device)
            if valid_len > 0:
                mask[:, :valid_len] = True
            vol_targets.append(torch.sqrt(sq_sum / float(horizon) + 1e-12) * self.risk_target_scale)
            dd_targets.append((-min_cum) * self.risk_target_scale)
            masks.append(mask)
        return (
            torch.stack(vol_targets, dim=-1),
            torch.stack(dd_targets, dim=-1),
            torch.stack(masks, dim=-1),
        )

    @staticmethod
    def _masked_smooth_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not torch.any(valid):
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred[valid], target[valid])

    def train_on_dataset(
        self,
        dataset: SequenceDataset,
        val_dataset: Optional[SequenceDataset] = None,
        max_steps: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        patience: int = 10,
    ) -> list[dict]:
        """データセット上で世界モデルを学習する.

        val_dataset がある場合、log_interval ごとに val loss を計算し、
        best model を保持する。patience 回連続で改善しなければ early stop する。

        Args:
            dataset: 学習用 SequenceDataset
            val_dataset: 検証用 SequenceDataset（省略可、あれば early stopping に使用）
            max_steps: 最大ステップ数（None の場合は self.max_steps）
            checkpoint_path: チェックポイント保存先（省略可）
            patience: early stopping の忍耐回数（val 評価回数単位）

        Returns:
            各ステップのロスログリスト
        """
        max_steps = max_steps or self.max_steps

        if len(dataset) == 0:
            print("[WM] WARNING: dataset is empty, skipping training")
            return []

        # dataset が batch_size 未満の場合 drop_last=True で loader が空になり無限ループする
        drop_last = len(dataset) >= self.batch_size
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            **self._loader_options(4),
        )
        self.ensemble.train()
        step = 0
        logs = []

        # Early stopping 用の状態
        best_val_loss = float("inf")
        best_state_dict = None
        no_improve_count = 0

        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break

                obs = batch["obs"].to(self.device)            # (B, T, obs_dim)
                obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

                # actions がない場合はゼロ埋め（WM 事前学習時はランダムポリシーで収集した軌跡を想定）
                if "actions" in batch:
                    actions = batch["actions"].to(self.device)  # (B, T, 1) or (B, T)
                else:
                    actions = torch.full(
                        (*obs.shape[:2], 1),
                        fill_value=self.benchmark_position if self.reward_mode == "excess_bh" else 0.0,
                        dtype=torch.float32,
                        device=self.device,
                    )
                if actions.ndim == 2 and not torch.is_floating_point(actions):
                    actions = self.action_values[actions].unsqueeze(-1)
                elif actions.ndim == 2:
                    actions = actions.unsqueeze(-1)

                # SPEC 準拠: WM の reward head は net_return（コスト控除後）を学習する
                # raw return がある場合のみ net_return を計算、なければゼロ埋め
                raw_returns = batch.get("returns")
                if raw_returns is not None:
                    raw_returns = raw_returns.to(self.device)   # (B, T)
                    rewards = self._compute_net_returns(actions, raw_returns)
                else:
                    rewards = torch.zeros(obs.shape[:2], device=self.device)

                # dones はゼロ埋め（継続的トレーディングでは done=False が多い）
                dones = torch.zeros_like(rewards)

                # 損失計算
                loss_dict = self.ensemble.compute_losses(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    free_bits=self.free_bits,
                    dyn_scale=self.dyn_scale,
                    rep_scale=self.rep_scale,
                    recon_scale=self.recon_scale,
                    reward_scale=self.reward_scale,
                    done_scale=self.done_scale,
                )

                total_loss = loss_dict["loss"]

                # --- Auxiliary losses ---
                idm_loss_val = 0.0
                return_loss_val = 0.0
                vol_loss_val = 0.0
                drawdown_loss_val = 0.0
                regime_loss_val = 0.0

                has_predictive_head = (
                    self.return_head is not None
                    or self.vol_head is not None
                    or self.drawdown_head is not None
                    or self.regime_head is not None
                )
                if self.idm_head is not None or has_predictive_head:
                    z, _ = self.ensemble.encode(obs)  # (B, T, z_dim)

                    if self.idm_head is not None and "actions" in batch and not torch.is_floating_point(batch["actions"]):
                        z_t = z[:, :-1, :]   # (B, T-1, z_dim)
                        z_t1 = z[:, 1:, :]   # (B, T-1, z_dim)
                        idm_logits = self.idm_head(z_t, z_t1)  # (B, T-1, n_actions)
                        oracle_acts = batch["actions"].to(self.device)[:, :-1]  # (B, T-1)
                        B_, T_, A_ = idm_logits.shape
                        idm_loss = F.cross_entropy(
                            idm_logits.reshape(B_ * T_, A_),
                            oracle_acts.reshape(B_ * T_),
                        )
                        total_loss = total_loss + self.idm_scale * idm_loss
                        idm_loss_val = idm_loss.item()

                    h = None
                    if has_predictive_head:
                        out_h = self.ensemble.forward(z, actions)
                        h = out_h["h"]  # (B, T, d_model)
                    if self.return_head is not None and raw_returns is not None and h is not None:
                        target, mask = self._future_return_targets(raw_returns)
                        pred = self.return_head(z, h)
                        return_loss = self._masked_smooth_l1(pred, target, mask)
                        total_loss = total_loss + self.return_scale * return_loss
                        return_loss_val = return_loss.item()

                    if (self.vol_head is not None or self.drawdown_head is not None) and raw_returns is not None and h is not None:
                        vol_target, dd_target, risk_mask = self._future_risk_targets(raw_returns)
                        if self.vol_head is not None:
                            vol_pred = self.vol_head(z, h)
                            vol_loss = self._masked_smooth_l1(vol_pred, vol_target, risk_mask)
                            total_loss = total_loss + self.vol_scale * vol_loss
                            vol_loss_val = vol_loss.item()
                        if self.drawdown_head is not None:
                            dd_pred = self.drawdown_head(z, h)
                            drawdown_loss = self._masked_smooth_l1(dd_pred, dd_target, risk_mask)
                            total_loss = total_loss + self.drawdown_scale * drawdown_loss
                            drawdown_loss_val = drawdown_loss.item()

                    regime_probs = batch.get("regime")
                    if self.regime_head is not None and regime_probs is not None and h is not None:
                        regime_probs = regime_probs.to(self.device)
                        regime_logits = self.regime_head(z, h)
                        log_probs = F.log_softmax(regime_logits, dim=-1)
                        regime_loss = -(regime_probs * log_probs).sum(dim=-1).mean()
                        total_loss = total_loss + self.regime_aux_scale * regime_loss
                        regime_loss_val = regime_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self._all_params, self.grad_clip)
                self.optimizer.step()

                step += 1
                self.global_step += 1

                log = {
                    "step": self.global_step,
                    "loss": total_loss.item(),
                    "base_loss": loss_dict["base_loss"].item(),
                    "disagreement": loss_dict["disagreement"].item(),
                    "idm_loss": idm_loss_val,
                    "return_loss": return_loss_val,
                    "vol_loss": vol_loss_val,
                    "drawdown_loss": drawdown_loss_val,
                    "regime_loss": regime_loss_val,
                }
                logs.append(log)
                self.loss_history.append(log)

                if step % self.log_interval == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    aux_str = ""
                    if self.idm_head is not None:
                        aux_str += f" | IDM: {log['idm_loss']:.4f}"
                    if self.return_head is not None:
                        aux_str += f" | Ret: {log['return_loss']:.4f}"
                    if self.vol_head is not None:
                        aux_str += f" | Vol: {log['vol_loss']:.4f}"
                    if self.drawdown_head is not None:
                        aux_str += f" | DD: {log['drawdown_loss']:.4f}"
                    if self.regime_head is not None:
                        aux_str += f" | Reg: {log['regime_loss']:.4f}"
                    print(
                        f"[{ts}] [WM] Step {self.global_step}/{max_steps} | "
                        f"Loss: {log['loss']:.4f} | "
                        f"BaseLoss: {log['base_loss']:.4f} | "
                        f"Disagree: {log['disagreement']:.4f}"
                        + aux_str
                    )

                    # Validation loss + early stopping
                    if val_dataset is not None:
                        val_loss = self._eval_loss(val_dataset)
                        print(f"       Val Loss: {val_loss:.4f}", end="")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state_dict = {
                                k: v.cpu().clone()
                                for k, v in self.ensemble.state_dict().items()
                            }
                            no_improve_count = 0
                            print(" ★ best")
                        else:
                            no_improve_count += 1
                            print(f" (no improve {no_improve_count}/{patience})")

                        if no_improve_count >= patience:
                            print(f"[WM] Early stopping at step {self.global_step} "
                                  f"(best val loss: {best_val_loss:.4f})")
                            step = max_steps  # ループ脱出
                            break

            # エポック終了後にチェックポイント保存
            if checkpoint_path is not None:
                self.save(checkpoint_path)

        # Best model を復元（val_dataset があり、改善があった場合）
        if best_state_dict is not None:
            self.ensemble.load_state_dict(best_state_dict)
            self.ensemble.to(self.device)
            print(f"[WM] Restored best model (val loss: {best_val_loss:.4f})")
            if checkpoint_path is not None:
                self.save(checkpoint_path)

        return logs

    @torch.no_grad()
    def _eval_loss(self, dataset: SequenceDataset, n_batches: int = 10) -> float:
        """Validation loss を計算する（n_batches 分のミニバッチ平均）.

        学習時と同じ net_return（コスト控除後）を reward として使用する。
        """
        self.ensemble.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self._loader_options(2),
        )
        total = 0.0
        count = 0
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            obs = batch["obs"].to(self.device)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)  # training と同一処理
            default_action = self.benchmark_position if self.reward_mode == "excess_bh" else 0.0
            actions = batch.get("actions", torch.full((*obs.shape[:2], 1), default_action, dtype=torch.float32))
            actions = actions.to(self.device)
            if actions.ndim == 2 and not torch.is_floating_point(actions):
                actions = self.action_values[actions].unsqueeze(-1)
            elif actions.ndim == 2:
                actions = actions.unsqueeze(-1)

            # 学習時と同じ: raw_returns → net_returns を reward として使用
            raw_returns = batch.get("returns")
            if raw_returns is not None:
                raw_returns = raw_returns.to(self.device)
                raw_returns = torch.nan_to_num(raw_returns, nan=0.0, posinf=0.0, neginf=0.0)
                rewards = self._compute_net_returns(actions, raw_returns)
            else:
                rewards = torch.zeros(obs.shape[:2], device=self.device)

            dones = torch.zeros_like(rewards)

            loss_dict = self.ensemble.compute_losses(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                free_bits=self.free_bits,
                dyn_scale=self.dyn_scale,
                rep_scale=self.rep_scale,
                recon_scale=self.recon_scale,
                reward_scale=self.reward_scale,
                done_scale=self.done_scale,
            )
            total_loss = loss_dict["loss"]
            has_predictive_head = (
                self.return_head is not None
                or self.vol_head is not None
                or self.drawdown_head is not None
                or self.regime_head is not None
            )
            if has_predictive_head:
                z, _ = self.ensemble.encode(obs)
                h = self.ensemble.forward(z, actions)["h"]
                if self.return_head is not None and raw_returns is not None:
                    target, mask = self._future_return_targets(raw_returns)
                    pred = self.return_head(z, h)
                    total_loss = total_loss + self.return_scale * self._masked_smooth_l1(pred, target, mask)
                if (self.vol_head is not None or self.drawdown_head is not None) and raw_returns is not None:
                    vol_target, dd_target, risk_mask = self._future_risk_targets(raw_returns)
                    if self.vol_head is not None:
                        vol_pred = self.vol_head(z, h)
                        total_loss = total_loss + self.vol_scale * self._masked_smooth_l1(
                            vol_pred,
                            vol_target,
                            risk_mask,
                        )
                    if self.drawdown_head is not None:
                        dd_pred = self.drawdown_head(z, h)
                        total_loss = total_loss + self.drawdown_scale * self._masked_smooth_l1(
                            dd_pred,
                            dd_target,
                            risk_mask,
                        )
            if self.regime_head is not None and "regime" in batch:
                regime_probs = batch["regime"].to(self.device)
                regime_logits = self.regime_head(z, h)
                log_probs = F.log_softmax(regime_logits, dim=-1)
                regime_loss = -(regime_probs * log_probs).sum(dim=-1).mean()
                total_loss = total_loss + self.regime_aux_scale * regime_loss
            total += total_loss.item()
            count += 1

        self.ensemble.train()
        return total / max(count, 1)

    @torch.no_grad()
    def encode_sequence(
        self,
        features: np.ndarray,
        actions: Optional[np.ndarray] = None,
        seq_len: int = 64,
    ) -> dict[str, np.ndarray]:
        """特徴量列をエンコードして潜在・hidden を返す.

        Actor-Critic の入力として使用する。

        Args:
            features: (T, obs_dim)
            actions: (T,) 行動インデックス
            seq_len: バッチ長

        Returns:
            {z: (T, z_dim), h: (T, d_model)}
        """
        self.ensemble.eval()
        T, obs_dim = features.shape
        z_dim = self.ensemble.get_z_dim()
        d_model = self.ensemble.get_d_model()

        if T == 0:
            return {"z": np.zeros((0, z_dim)), "h": np.zeros((0, d_model))}

        z_arr = np.zeros((T, z_dim), dtype=np.float32)
        h_arr = np.zeros((T, d_model), dtype=np.float32)
        covered = 0

        # 各チャンクの直前 seq_len ステップをウォームアップ文脈として追加する。
        # これにより Transformer がブロック境界でコンテキストをリセットする問題を防ぐ。
        # 合計シーケンス長は最大 2*seq_len <= max_seq_len（configs で保証）。
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            # 最後のチャンクが短い場合、末尾揃えにする
            if end - start < seq_len and T >= seq_len:
                start = T - seq_len
                end = T

            # ウォームアップ文脈: start の直前 seq_len ステップ
            ctx_start = max(0, start - seq_len)

            obs_t = torch.tensor(
                features[ctx_start:end], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)

            if actions is not None:
                act_t = torch.tensor(
                    actions[ctx_start:end], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            else:
                act_t = torch.full(
                    (1, end - ctx_start, 1),
                    fill_value=self.benchmark_position if self.reward_mode == "excess_bh" else 0.0,
                    dtype=torch.float32,
                    device=self.device,
                )
            if act_t.ndim == 2:
                act_t = act_t.unsqueeze(-1)

            z, _ = self.ensemble.encode(obs_t)
            out = self.ensemble.forward(z, act_t)

            # ウォームアップ部分を除いた本体のみを書き込む
            prefix_len = start - ctx_start
            z_np = z.squeeze(0)[prefix_len:].cpu().numpy()
            h_np = out["h"].squeeze(0)[prefix_len:].cpu().numpy()

            write_start = max(start, covered)
            offset = write_start - start
            z_arr[write_start:end] = z_np[offset:]
            h_arr[write_start:end] = h_np[offset:]
            covered = end

            if end == T:
                break

        return {"z": z_arr, "h": h_arr}

    def save(self, path: str) -> None:
        """チェックポイントを保存する."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        ckpt = {
            "ensemble": self.ensemble.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.idm_head is not None:
            ckpt["idm_head"] = self.idm_head.state_dict()
        if self.return_head is not None:
            ckpt["return_head"] = self.return_head.state_dict()
        if self.vol_head is not None:
            ckpt["vol_head"] = self.vol_head.state_dict()
        if self.drawdown_head is not None:
            ckpt["drawdown_head"] = self.drawdown_head.state_dict()
        if self.regime_head is not None:
            ckpt["regime_head"] = self.regime_head.state_dict()
        torch.save(ckpt, path)
        print(f"[WM] Checkpoint saved: {path}")

    def load(self, path: str) -> None:
        """チェックポイントをロードする."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ensemble.load_state_dict(ckpt["ensemble"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(f"[WM] Optimizer state skipped: {exc}")
        self.global_step = ckpt.get("global_step", 0)
        if self.idm_head is not None and "idm_head" in ckpt:
            self.idm_head.load_state_dict(ckpt["idm_head"])
        if self.return_head is not None and "return_head" in ckpt:
            self.return_head.load_state_dict(ckpt["return_head"])
        if self.vol_head is not None and "vol_head" in ckpt:
            self.vol_head.load_state_dict(ckpt["vol_head"])
        if self.drawdown_head is not None and "drawdown_head" in ckpt:
            self.drawdown_head.load_state_dict(ckpt["drawdown_head"])
        if self.regime_head is not None and "regime_head" in ckpt:
            self.regime_head.load_state_dict(ckpt["regime_head"])
        print(f"[WM] Checkpoint loaded: {path} (step={self.global_step})")

    @torch.no_grad()
    def predict_auxiliary_from_encoded(
        self,
        z: np.ndarray,
        h: np.ndarray,
        batch_size: int = 8192,
    ) -> dict[str, np.ndarray]:
        """Return predictive auxiliary head outputs for already encoded states."""
        heads = {
            "return": self.return_head,
            "vol": self.vol_head,
            "drawdown": self.drawdown_head,
        }
        active = {name: head for name, head in heads.items() if head is not None}
        if not active:
            return {}

        was_training = self.ensemble.training
        self.ensemble.eval()
        for head in active.values():
            head.eval()

        z_arr = np.asarray(z, dtype=np.float32)
        h_arr = np.asarray(h, dtype=np.float32)
        outputs: dict[str, list[np.ndarray]] = {name: [] for name in active}
        n = min(len(z_arr), len(h_arr))
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            z_t = torch.as_tensor(z_arr[start:end], dtype=torch.float32, device=self.device)
            h_t = torch.as_tensor(h_arr[start:end], dtype=torch.float32, device=self.device)
            for name, head in active.items():
                pred = head(z_t, h_t)
                if pred.ndim == 1:
                    pred = pred.unsqueeze(-1)
                outputs[name].append(pred.detach().cpu().numpy().astype(np.float32))

        if was_training:
            self.ensemble.train()
        return {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}

    def predictive_feature_names(self) -> list[str]:
        names: list[str] = []
        if self.return_head is not None:
            names.extend([f"wm_pred_return_h{h}" for h in self.return_horizons])
        if self.vol_head is not None:
            names.extend([f"wm_pred_vol_h{h}" for h in self.risk_horizons])
        if self.drawdown_head is not None:
            names.extend([f"wm_pred_drawdown_h{h}" for h in self.risk_horizons])
        return names
