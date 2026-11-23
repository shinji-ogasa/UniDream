"""Isolated Plan011-style market WM for the preregistered feature ablation.

This does not change the trading pipeline or its checkpoint/bundle format.
All arms share the same architecture; excluded input channels are explicitly
masked after a common train-only normalization. Reconstruction always targets
the same five OHLCV channels. No raw-feature shortcut enters forecast heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from unidream.world_model.encoder import ObsDecoder, symlog
from unidream.world_model.transformer import (
    CausalTransformerBlock,
    TransformerWorldModel,
    twohot_loss,
)


class _AttentionBlock(CausalTransformerBlock):
    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        # Attention weights are unused; avoid materializing them on MPS.
        attended, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attended
        return x + self.ffn(self.ln2(x))


class FeatureAblationWM(nn.Module):
    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__()
        self.wm = TransformerWorldModel(
            obs_dim=input_dim,
            act_dim=1,
            n_categoricals=int(model_cfg.get("n_categoricals", 32)),
            n_classes=int(model_cfg.get("n_classes", 32)),
            d_model=int(model_cfg.get("d_model", 192)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            d_ff=int(model_cfg.get("d_ff", 768)),
            dropout=float(model_cfg.get("dropout", .1)),
            max_seq_len=int(model_cfg.get("context", 256)),
            encoder_hidden=int(model_cfg.get("encoder_hidden", 192)),
            encoder_layers=2,
        )
        # Preserve the existing model equations and parameters, with a more
        # economical attention call. All arms execute the identical path.
        for block in self.wm.blocks:
            block.__class__ = _AttentionBlock
        self.wm.decoder = ObsDecoder(self.wm.z_dim, self.wm.d_model, 5, 192)
        self.head = nn.Sequential(
            nn.Linear(self.wm.z_dim + self.wm.d_model, 192),
            nn.ELU(),
            nn.Linear(192, 12),
        )
        for key in ("return", "variance", "downside"):
            self.register_buffer(key + "_scale", torch.tensor(scales[key], dtype=torch.float32))
        self.register_buffer("variance_floor", torch.tensor(1e-12, dtype=torch.float32))

    def forward(self, x):
        z, posterior = self.wm.encode(x)
        dynamics = self.wm(z, torch.ones((*x.shape[:2], 1), device=x.device, dtype=x.dtype))
        raw = self.head(torch.cat((z, dynamics["h"]), dim=-1))
        predictions = {
            "return": raw[..., :3] * self.return_scale,
            "variance": (F.softplus(raw[..., 3:6]) + 1e-4) * self.variance_scale,
            "downside": F.softplus(raw[..., 6:9]) * self.downside_scale,
            "crash": torch.sigmoid(raw[..., 9:12]),
        }
        return predictions, {"z": z, "posterior": posterior, "raw": raw, **dynamics}

    def loss(self, x, base_target, targets, valid, next_return):
        prediction, state = self(x)
        # NaN placeholders must be removed before arithmetic: masking a NaN
        # loss afterwards can still yield NaN gradients in autograd.
        targets = {key: torch.where(valid, value, torch.zeros_like(value))
                   for key, value in targets.items()}
        valid_float = valid.to(x.dtype)
        denominator = valid_float.sum().clamp_min(1)
        def masked(value):
            return (torch.where(valid, value, torch.zeros_like(value))).sum() / denominator

        return_loss = masked(((prediction["return"] - targets["return"]) / self.return_scale).square())
        # QLIKE is applied to SUM of squared future returns (a variance proxy),
        # never to standard deviation or to its mean per 15-minute bar.
        ratio = targets["variance"].clamp_min(self.variance_floor) / prediction["variance"].clamp_min(self.variance_floor)
        variance_loss = masked(ratio - torch.log(ratio) - 1)
        downside_loss = masked(((prediction["downside"] - targets["downside"]) / self.downside_scale).square())
        crash_loss = masked(F.binary_cross_entropy_with_logits(state["raw"][..., 9:12], targets["crash"], reduction="none"))
        reconstruction = self.wm.decoder(state["z"], state["h"])
        reconstruction_loss = F.mse_loss(reconstruction, symlog(base_target))

        post_next = state["posterior"][:, 1:]
        prior = state["prior_logits"][:, :-1]
        dyn_kl = td.kl_divergence(self.wm.encoder.get_dist(post_next.detach()), self.wm.encoder.get_dist(prior)).clamp_min(1).mean()
        rep_kl = td.kl_divergence(self.wm.encoder.get_dist(post_next), self.wm.encoder.get_dist(prior.detach())).clamp_min(1).mean()
        # Fixed action is a market-model context, not hindsight inventory.
        # Reward predicts the next market return, rather than identically zero
        # B&H-relative reward under the trading pipeline's actionless setting.
        reward_valid = torch.isfinite(next_return)
        reward_loss = twohot_loss(
            state["reward_logits"][reward_valid], next_return[reward_valid], self.wm.bins
        )
        loss = (return_loss + .2 * variance_loss + .2 * downside_loss + .1 * crash_loss
                + .3 * reconstruction_loss + .2 * dyn_kl + .05 * rep_kl + .3 * reward_loss)
        return loss, {
            "return": return_loss.detach(), "variance": variance_loss.detach(),
            "downside": downside_loss.detach(), "crash": crash_loss.detach(),
            "reconstruction": reconstruction_loss.detach(),
            "kl_dyn": dyn_kl.detach(), "kl_rep": rep_kl.detach(),
        }

    @torch.no_grad()
    def predict_last(self, x):
        predictions, _ = self(x)
        return {key: value[:, -1] for key, value in predictions.items()}
