"""Ensemble World Model モジュール.

3〜5 個の TransformerWorldModel を独立に学習し、
不一致（disagreement）ペナルティを計算する。

disagreement = 各モデルの prior logits の分散の平均
→ 世界モデルが不確かな領域への探索を抑制する
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unidream.world_model.transformer import TransformerWorldModel


class EnsembleWorldModel(nn.Module):
    """複数の TransformerWorldModel のアンサンブル.

    Args:
        n_models: アンサンブル数（デフォルト 3）
        disagree_scale: disagreement ペナルティの係数
        **wm_kwargs: TransformerWorldModel のコンストラクタ引数
    """

    def __init__(
        self,
        n_models: int = 3,
        disagree_scale: float = 0.1,
        **wm_kwargs,
    ):
        super().__init__()
        self.n_models = n_models
        self.disagree_scale = disagree_scale
        self.models = nn.ModuleList([
            TransformerWorldModel(**wm_kwargs)
            for _ in range(n_models)
        ])

    def compute_losses(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        **loss_kwargs,
    ) -> dict[str, torch.Tensor]:
        """全モデルの損失を計算する.

        Args:
            obs: (B, T, obs_dim)
            actions: (B, T)
            rewards: (B, T)
            dones: (B, T)

        Returns:
            各モデルの losses と集計値
        """
        all_losses = []
        all_prior_logits = []

        for model in self.models:
            losses = model.compute_losses(obs, actions, rewards, dones, **loss_kwargs)
            all_losses.append(losses["loss"])
            # prior logits を集めて disagreement を計算
            all_prior_logits.append(losses["prior_logits"])  # (B, T, n_cats, n_classes)

        # 各モデルの total loss の平均
        total_loss = torch.stack(all_losses).mean()

        # Disagreement penalty: prior logits の分散
        # shape: (n_models, B, T, n_cats, n_classes) → variance over n_models
        stacked_priors = torch.stack(all_prior_logits, dim=0)  # (n_models, B, T, n_cats, n_classes)
        prior_probs = F.softmax(stacked_priors, dim=-1)
        disagreement = prior_probs.var(dim=0).mean()  # scalar

        return {
            "loss": total_loss + self.disagree_scale * disagreement,
            "base_loss": total_loss,
            "disagreement": disagreement,
            "model_losses": [l.item() for l in all_losses],
        }

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """全モデルで forward を実行し、平均を取る.

        Args:
            z: (B, T, z_dim)
            actions: (B, T)

        Returns:
            平均化された出力辞書
        """
        all_h = []
        all_prior_logits = []
        all_reward_logits = []
        all_done_logits = []

        for model in self.models:
            out = model.forward(z, actions)
            all_h.append(out["h"])
            all_prior_logits.append(out["prior_logits"])
            all_reward_logits.append(out["reward_logits"])
            all_done_logits.append(out["done_logits"])

        return {
            "h": torch.stack(all_h).mean(0),
            "prior_logits": torch.stack(all_prior_logits).mean(0),
            "reward_logits": torch.stack(all_reward_logits).mean(0),
            "done_logits": torch.stack(all_done_logits).mean(0),
        }

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """先頭モデルのエンコーダを使用して観測を変換する.

        全モデルで encode するとメモリコストが大きいため、
        先頭モデルのエンコーダを共用する。
        （学習時は各モデルの compute_losses 内で encode が呼ばれる）
        """
        return self.models[0].encode(obs)

    @torch.no_grad()
    def imagine_step(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """全モデルで imagination を実行し、平均化する.

        disagreement（モデル間の prior の不一致）を reward に組み込む。
        """
        all_results = []
        for model in self.models:
            result = model.imagine_step(z, h, action, past_zs, past_as)
            all_results.append(result)

        # 各モデルの prior_logits の softmax
        all_priors = torch.stack([r["prior_logits"] for r in all_results], dim=0)
        prior_probs = F.softmax(all_priors, dim=-1)
        disagreement = prior_probs.var(dim=0).mean(-1).mean(-1)  # (B,)

        # 平均化した next_z（先頭モデルのサンプルを使用）
        next_z = all_results[0]["next_z"]
        next_h = torch.stack([r["next_h"] for r in all_results]).mean(0)
        reward = torch.stack([r["reward"] for r in all_results]).mean(0)
        done = torch.stack([r["done"] for r in all_results]).mean(0)

        return {
            "next_z": next_z,
            "next_h": next_h,
            "reward": reward - self.disagree_scale * disagreement,  # disagreement ペナルティ
            "done": done,
            "disagreement": disagreement,
            "prior_logits": all_priors.mean(0),
            "past_zs": all_results[0]["past_zs"],
            "past_as": all_results[0]["past_as"],
        }

    def get_bins(self) -> torch.Tensor:
        """twohot bins を返す（先頭モデルから）."""
        return self.models[0].bins

    def get_z_dim(self) -> int:
        return self.models[0].z_dim

    def get_d_model(self) -> int:
        return self.models[0].d_model
