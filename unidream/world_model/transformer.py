"""Block-Causal Transformer 世界モデル.

GPT スタイルの causal Transformer で時系列のダイナミクスをモデル化する。
各タイムステップを 1 ブロックとして causal masking で未来リークを防ぐ。

設計:
  入力: [z_t (1024), a_t (5)] × T → embed → Transformer → h_t
  出力ヘッド:
    - Prior: p(z_{t+1} | h_t) → logits(32, 32)
    - Decoder: obs_recon(t) → obs_dim
    - Reward: r_t → 255 bins (symlog + twohot)
    - Done: terminal(t) → sigmoid

References:
    nicklashansen/dreamer4 - BlockCausalLayer 設計
    NM512/dreamerv3-torch - symlog/twohot/KL loss
    eloialonso/iris - causal Transformer as world model
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd

from unidream.world_model.encoder import symlog, symexp, OneHotDist, ObsEncoder, ObsDecoder


# --- Twohot encoding ---

def make_twohot_bins(n_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Twohot encoding 用のビン境界を生成する（symlog 空間）."""
    return torch.linspace(low, high, n_bins)


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """連続値を twohot 表現にエンコードする.

    Args:
        x: (...,) 値列
        bins: (n_bins,) ビン境界

    Returns:
        twohot: (..., n_bins) 2 要素が非ゼロの one-hot 的ベクトル
    """
    n_bins = bins.shape[0]
    bins = bins.to(x.device)
    x_clamped = x.clamp(bins[0], bins[-1])

    # 左のビンインデックス
    lower_idx = (x_clamped.unsqueeze(-1) >= bins).sum(-1) - 1
    lower_idx = lower_idx.clamp(0, n_bins - 2)
    upper_idx = lower_idx + 1

    lower_val = bins[lower_idx]
    upper_val = bins[upper_idx]

    # 線形補間重み
    upper_weight = (x_clamped - lower_val) / (upper_val - lower_val + 1e-8)
    lower_weight = 1.0 - upper_weight

    twohot = torch.zeros(*x.shape, n_bins, device=x.device, dtype=x.dtype)
    twohot.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    twohot.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return twohot


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """twohot 分布の期待値を計算する.

    Args:
        logits: (..., n_bins)
        bins: (n_bins,)

    Returns:
        expected_value: (...,) symexp を適用したスカラー値
    """
    probs = F.softmax(logits, dim=-1)
    bins = bins.to(logits.device)
    symlog_val = (probs * bins).sum(-1)
    return symexp(symlog_val)


def twohot_loss(logits: torch.Tensor, targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Twohot cross-entropy loss.

    Args:
        logits: (..., n_bins)
        targets: (...,) 実際の値（symlog 変換前）
        bins: (n_bins,) ビン境界

    Returns:
        loss: スカラー
    """
    targets_symlog = symlog(targets)
    target_twohot = twohot_encode(targets_symlog, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_twohot * log_probs).sum(-1).mean()


# --- Sinusoidal Positional Encoding ---

class SinusoidalPE(nn.Module):
    """Sinusoidal 位置埋め込み."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


# --- Causal Transformer Block ---

class CausalTransformerBlock(nn.Module):
    """GPT スタイルの Causal Transformer ブロック.

    Pre-LayerNorm 構造: LN → Attention → + → LN → FFN → +
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            attn_mask: (T, T) causal mask（is_causal=True 使用時は None で可）

        Returns:
            (B, T, d_model)
        """
        # Self-attention with causal mask
        # PyTorch 2.0+ では is_causal=True が内部で causal mask を生成するため、
        # attn_mask との併用は不要。is_causal のみ使用する。
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, is_causal=True)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.ln2(x))
        return x


# --- Transformer World Model ---

class TransformerWorldModel(nn.Module):
    """Block-Causal Transformer 世界モデル.

    DreamerV3 の RSSM を GPT スタイルの Transformer に置換。

    処理フロー:
        z_t (1024), a_t (5) → Linear embed → PE →
        CausalTransformer × n_layers → h_t →
        [Prior, Decoder, Reward, Done] heads

    Args:
        obs_dim: 観測次元
        act_dim: 行動次元（one-hot または embedding）
        n_categoricals: カテゴリカル数（デフォルト 32）
        n_classes: クラス数（デフォルト 32）
        d_model: Transformer 隠れ次元
        n_heads: Attention ヘッド数
        n_layers: Transformer 層数
        d_ff: FFN 中間次元
        dropout: dropout 率
        max_seq_len: 最大シーケンス長
        n_bins: twohot encoding のビン数（reward/critic）
        bin_low: ビン下限（symlog 空間）
        bin_high: ビン上限（symlog 空間）
        unimix_ratio: OneHotDist の uniform mix 比率
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 5,
        n_categoricals: int = 32,
        n_classes: int = 32,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        n_bins: int = 255,
        bin_low: float = -20.0,
        bin_high: float = 20.0,
        unimix_ratio: float = 0.01,
        encoder_hidden: int = 256,
    ):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.z_dim = n_categoricals * n_classes
        self.act_dim = act_dim
        self.d_model = d_model

        # --- Encoder/Decoder ---
        self.encoder = ObsEncoder(
            obs_dim, n_categoricals, n_classes, encoder_hidden, unimix_ratio
        )
        self.decoder = ObsDecoder(self.z_dim, d_model, obs_dim, encoder_hidden)

        # --- Input embedding: z + a → d_model ---
        self.z_embed = nn.Linear(self.z_dim, d_model)
        self.a_embed = nn.Embedding(act_dim, d_model)
        self.input_proj = nn.Linear(d_model, d_model)  # z+a 結合後の射影

        # --- Positional Encoding ---
        self.pe = SinusoidalPE(d_model, max_len=max_seq_len, dropout=dropout)

        # --- Transformer ---
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # --- Output Heads ---
        # Prior: p(z_{t+1} | h_t)
        self.prior_head = nn.Linear(d_model, n_categoricals * n_classes)

        # Reward head: twohot over symlog bins
        self.reward_head = nn.Linear(d_model, n_bins)

        # Done head: binary
        self.done_head = nn.Linear(d_model, 1)

        # Twohot bins
        self.register_buffer("bins", make_twohot_bins(n_bins, bin_low, bin_high))
        self.unimix_ratio = unimix_ratio

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """(T, T) の causal attention mask を生成する（上三角を -inf）."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """観測を離散潜在に変換する.

        Args:
            obs: (B, T, obs_dim)

        Returns:
            z: (B, T, z_dim) straight-through サンプル
            post_logits: (B, T, n_cats, n_classes) posterior logits
        """
        return self.encoder(obs)

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Transformer の forward pass.

        Args:
            z: (B, T, z_dim) 現時点の潜在（encoder 出力）
            actions: (B, T) 行動インデックス

        Returns:
            h: (B, T, d_model) Transformer hidden states
            prior_logits: (B, T, n_cats, n_classes) 次時刻の prior
            reward_logits: (B, T, n_bins)
            done_logits: (B, T, 1)
        """
        B, T, _ = z.shape

        # Input embedding: z と action を加算
        z_emb = self.z_embed(z)                           # (B, T, d_model)
        a_emb = self.a_embed(actions)                     # (B, T, d_model)
        x = self.input_proj(z_emb + a_emb)               # (B, T, d_model)

        # Positional encoding
        x = self.pe(x)

        # Causal Transformer
        mask = self._causal_mask(T, z.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        h = self.ln_final(x)  # (B, T, d_model)

        # Output heads
        prior_logits_flat = self.prior_head(h)            # (B, T, n_cats*n_classes)
        prior_logits = prior_logits_flat.reshape(
            B, T, self.n_categoricals, self.n_classes
        )
        reward_logits = self.reward_head(h)               # (B, T, n_bins)
        done_logits = self.done_head(h)                   # (B, T, 1)

        return {
            "h": h,
            "prior_logits": prior_logits,
            "reward_logits": reward_logits,
            "done_logits": done_logits,
        }

    def compute_losses(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        free_bits: float = 1.0,
        dyn_scale: float = 0.5,
        rep_scale: float = 0.1,
        recon_scale: float = 1.0,
        reward_scale: float = 1.0,
        done_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """世界モデルの全損失を計算する.

        Args:
            obs: (B, T, obs_dim)
            actions: (B, T) 行動インデックス
            rewards: (B, T) 実際のリターン
            dones: (B, T) 終了フラグ（0/1）

        Returns:
            losses と中間変数の辞書
        """
        B, T, _ = obs.shape

        # --- Posterior: q(z_t | obs_t) ---
        z, post_logits = self.encode(obs)  # (B, T, z_dim), (B, T, n_cats, n_classes)

        # --- Transformer forward ---
        out = self.forward(z, actions)
        h = out["h"]
        prior_logits = out["prior_logits"]
        reward_logits = out["reward_logits"]
        done_logits = out["done_logits"]

        # --- Reconstruction loss (Gaussian NLL 近似 = MSE) ---
        obs_recon = self.decoder(z, h)
        recon_loss = F.mse_loss(obs_recon, symlog(obs))

        # --- KL loss with free bits (DreamerV3 スタイル) ---
        # post: q(z_t | obs_t), prior: p(z_t | h_{t-1})
        # prior は t=1 から T まで（先読み shift なし。h_{t-1} は block t から得られる）
        # 簡略化: prior[t] は h[t] から計算されたもの（次ステップの予測）
        # posterior[1:] と prior[:T-1] を対比
        if T > 1:
            post_logits_shifted = post_logits[:, 1:]    # (B, T-1, n_cats, n_classes) ← 実際の次ステップ
            prior_logits_shifted = prior_logits[:, :-1]  # (B, T-1, n_cats, n_classes) ← 予測

            post_dist = self.encoder.get_dist(post_logits_shifted)
            prior_dist = self.encoder.get_dist(prior_logits_shifted)

            # stop_gradient variants
            kl_dyn = torchd.kl_divergence(
                self.encoder.get_dist(post_logits_shifted.detach()),
                prior_dist,
            )
            kl_rep = torchd.kl_divergence(
                post_dist,
                self.encoder.get_dist(prior_logits_shifted.detach()),
            )

            kl_dyn = kl_dyn.clamp(min=free_bits).mean()
            kl_rep = kl_rep.clamp(min=free_bits).mean()
            kl_loss = dyn_scale * kl_dyn + rep_scale * kl_rep
        else:
            kl_loss = torch.tensor(0.0, device=obs.device)
            kl_dyn = kl_loss
            kl_rep = kl_loss

        # --- Reward loss (twohot) ---
        # h[t] が r[t] を予測する
        reward_loss = twohot_loss(reward_logits[:, :-1], rewards[:, 1:], self.bins)

        # --- Done loss ---
        done_loss = F.binary_cross_entropy_with_logits(
            done_logits[:, :-1].squeeze(-1),
            dones[:, 1:].float(),
        )

        total_loss = (
            recon_scale * recon_loss
            + kl_loss
            + reward_scale * reward_loss
            + done_scale * done_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_dyn": kl_dyn,
            "kl_rep": kl_rep,
            "reward_loss": reward_loss,
            "done_loss": done_loss,
            "z": z,
            "h": h,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
        }

    @torch.no_grad()
    def imagine_step(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
        past_zs: Optional[torch.Tensor] = None,
        past_as: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """1 ステップの imagination を実行する.

        Args:
            z: (B, z_dim) 現在の潜在
            h: (B, d_model) 現在の hidden（使用済みコンテキスト）
            action: (B,) 行動インデックス
            past_zs: (B, t, z_dim) 過去の潜在列（None の場合は z のみ）
            past_as: (B, t) 過去の行動列

        Returns:
            next_z, next_h, reward, done
        """
        if past_zs is None:
            past_zs = z.unsqueeze(1)   # (B, 1, z_dim)
            past_as = action.unsqueeze(1)  # (B, 1)
        else:
            past_zs = torch.cat([past_zs, z.unsqueeze(1)], dim=1)
            past_as = torch.cat([past_as, action.unsqueeze(1)], dim=1)

        out = self.forward(past_zs, past_as)
        h_new = out["h"][:, -1]          # (B, d_model) 最後のタイムステップ
        prior_logits = out["prior_logits"][:, -1]  # (B, n_cats, n_classes)
        reward_logits = out["reward_logits"][:, -1]
        done_logits = out["done_logits"][:, -1]

        # next_z をサンプル（imagination では prior からサンプル）
        next_z_dist = OneHotDist(logits=prior_logits)
        next_z_onehot = next_z_dist.sample()
        next_z = next_z_onehot.reshape(z.shape[0], -1)

        # 予測報酬・done
        reward = twohot_decode(reward_logits, self.bins)
        done = torch.sigmoid(done_logits.squeeze(-1))

        return {
            "next_z": next_z,
            "next_h": h_new,
            "reward": reward,
            "done": done,
            "prior_logits": prior_logits,
            "past_zs": past_zs,
            "past_as": past_as,
        }
