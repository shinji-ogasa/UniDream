"""世界モデル学習エントリポイント.

EnsembleWorldModel を WFO データ上で学習する。
損失: reconstruction + KL (free bits) + reward (twohot) + done (BCE)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unidream.data.dataset import SequenceDataset
from unidream.data.oracle import ACTIONS as ORACLE_ACTIONS
from unidream.world_model.ensemble import EnsembleWorldModel


def build_ensemble(obs_dim: int, cfg: dict) -> EnsembleWorldModel:
    """config dict から EnsembleWorldModel を構築する."""
    wm_cfg = cfg.get("world_model", {})
    return EnsembleWorldModel(
        n_models=wm_cfg.get("n_ensemble", 3),
        disagree_scale=wm_cfg.get("disagree_scale", 0.1),
        obs_dim=obs_dim,
        act_dim=cfg.get("actions", {}).get("n", 5),
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
        self.device = torch.device(device)
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

        # コストパラメータ（net_return 計算に使用）
        costs_cfg = cfg.get("costs", {})
        self.cost_rate = (
            (costs_cfg.get("spread_bps", 5.0) / 10000) / 2
            + costs_cfg.get("fee_rate", 0.0004)
            + (costs_cfg.get("slippage_bps", 2.0) / 10000)
        )

        self.optimizer = torch.optim.Adam(
            self.ensemble.parameters(),
            lr=self.lr,
        )

        self.global_step = 0
        self.loss_history: list[dict] = []

    def _compute_net_returns(
        self,
        actions: torch.Tensor,
        raw_returns: torch.Tensor,
    ) -> torch.Tensor:
        """行動インデックスと生リターンからネットリターン（コスト控除後）を計算する.

        net_return[t] = ACTIONS[actions[t]] * raw_returns[t]
                        - cost_rate * |ACTIONS[actions[t]] - ACTIONS[actions[t-1]]|

        初期ポジション = 0.0（フラット）。

        Args:
            actions: (B, T) 行動インデックス
            raw_returns: (B, T) 生の対数リターン

        Returns:
            net_returns: (B, T) コスト控除後リターン
        """
        action_vals = torch.tensor(
            ORACLE_ACTIONS, dtype=raw_returns.dtype, device=raw_returns.device
        )
        positions = action_vals[actions]                             # (B, T)

        # 前ステップポジション（初期 = 0.0 = フラット）
        prev_positions = torch.cat([
            torch.zeros_like(positions[:, :1]),
            positions[:, :-1],
        ], dim=1)                                                    # (B, T)

        delta_pos = (positions - prev_positions).abs()
        costs = self.cost_rate * delta_pos                           # (B, T)
        return positions * raw_returns - costs                       # (B, T)

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
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
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

                # actions がない場合はゼロ埋め（WM 事前学習時はランダムポリシーで収集した軌跡を想定）
                if "actions" in batch:
                    actions = batch["actions"].to(self.device)  # (B, T)
                else:
                    actions = torch.zeros(obs.shape[:2], dtype=torch.long, device=self.device)

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

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ensemble.parameters(), self.grad_clip)
                self.optimizer.step()

                step += 1
                self.global_step += 1

                log = {
                    "step": self.global_step,
                    "loss": total_loss.item(),
                    "base_loss": loss_dict["base_loss"].item(),
                    "disagreement": loss_dict["disagreement"].item(),
                }
                logs.append(log)
                self.loss_history.append(log)

                if step % self.log_interval == 0:
                    print(
                        f"[WM] Step {self.global_step}/{max_steps} | "
                        f"Loss: {log['loss']:.4f} | "
                        f"BaseLoss: {log['base_loss']:.4f} | "
                        f"Disagree: {log['disagreement']:.4f}"
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
        """Validation loss を計算する（n_batches 分のミニバッチ平均）."""
        self.ensemble.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        total = 0.0
        count = 0
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            obs = batch["obs"].to(self.device)
            actions = batch.get("actions", torch.zeros(obs.shape[:2], dtype=torch.long))
            actions = actions.to(self.device)
            rewards = batch.get("returns", torch.zeros(obs.shape[:2]))
            rewards = rewards.to(self.device)
            dones = torch.zeros_like(rewards)

            loss_dict = self.ensemble.compute_losses(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            total += loss_dict["loss"].item()
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

        # 重複なしストライドでエンコード
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            # 最後のチャンクが短い場合、末尾揃えにする
            if end - start < seq_len and T >= seq_len:
                start = T - seq_len
                end = T

            obs_t = torch.tensor(
                features[start:end], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if actions is not None:
                act_t = torch.tensor(
                    actions[start:end], dtype=torch.long, device=self.device
                ).unsqueeze(0)
            else:
                act_t = torch.zeros(1, end - start, dtype=torch.long, device=self.device)

            z, _ = self.ensemble.encode(obs_t)
            out = self.ensemble.forward(z, act_t)

            z_np = z.squeeze(0).cpu().numpy()
            h_np = out["h"].squeeze(0).cpu().numpy()

            # 重複区間は後のウィンドウの後半のみ書き込む
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
        torch.save({
            "ensemble": self.ensemble.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        print(f"[WM] Checkpoint saved: {path}")

    def load(self, path: str) -> None:
        """チェックポイントをロードする."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ensemble.load_state_dict(ckpt["ensemble"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"[WM] Checkpoint loaded: {path} (step={self.global_step})")
