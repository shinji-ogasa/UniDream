"""Continuous direct forecaster for the frozen WM comparison's extension.

This is a separate research model: it does not alter the registered WM or
load its weights.  It shares that experiment's input windows and attention
capacity while directly learning future return and realized variance.  No
categorical bottleneck, reconstruction, dynamics KL or reward loss is used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from unidream.world_model.transformer import CausalTransformerBlock


class _DirectAttentionBlock(CausalTransformerBlock):
    """Same pre-normalized attention/FFN equations without unused weights."""

    def forward(self, x, attn_mask=None):
        normalized = self.ln1(x)
        attended, _ = self.attn(
            normalized, normalized, normalized,
            attn_mask=attn_mask, need_weights=False,
        )
        x = x + attended
        return x + self.ffn(self.ln2(x))


class DirectForecastTransformer(nn.Module):
    """Predict three return/variance horizons from continuous causal states.

    ``forward`` returns ``{"return": [B,T,3], "variance": [B,T,3]}``.
    The ``loss`` signature is compatible with ``FeatureAblationWM``; the
    reconstruction and next-return arguments are accepted but unused.
    All preprocessing and outcome-scale fitting belong to the train-only
    experiment preparation, outside this model.
    """

    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(model_cfg.get("d_model", 192))
        self.context = int(model_cfg.get("context", 256))
        n_heads = int(model_cfg.get("n_heads", 4))
        n_layers = int(model_cfg.get("n_layers", 2))
        hidden = int(model_cfg.get("encoder_hidden", self.d_model))
        dropout = float(model_cfg.get("dropout", .1))
        if min(self.input_dim, self.d_model, self.context, n_heads, n_layers, hidden) < 1:
            raise ValueError("input/model/context dimensions and layer counts must be positive")
        if self.d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.d_model),
            nn.ELU(),
        )
        self.position_embedding = nn.Parameter(torch.empty(1, self.context, self.d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=.02)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _DirectAttentionBlock(
                self.d_model, n_heads, int(model_cfg.get("d_ff", 768)), dropout
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ELU(),
            nn.Linear(self.d_model, 6),
        )
        for key in ("return", "variance"):
            scale = torch.as_tensor(scales[key], dtype=torch.float32).clone()
            if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"{key} scales must contain three finite positive values")
            self.register_buffer(key + "_scale", scale)
        self.register_buffer("variance_floor", torch.tensor(1e-12, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("x must have shape [batch, time, input_dim]")
        if not 0 < x.shape[1] <= self.context:
            raise ValueError("input sequence length must be between 1 and context")
        hidden = self.encoder(x)
        hidden = self.input_dropout(hidden + self.position_embedding[:, :x.shape[1]])
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, attn_mask=mask)
        raw = self.head(self.ln_final(hidden))
        return {
            "return": raw[..., :3] * self.return_scale,
            "variance": (F.softplus(raw[..., 3:6]) + 1e-4) * self.variance_scale,
        }

    def loss(self, x, base_target, targets, valid, next_return):
        """All-token return MSE + 0.2 variance QLIKE on valid labels only."""
        prediction = self(x)
        if valid.dtype != torch.bool or valid.shape != prediction["return"].shape:
            raise ValueError("valid must be boolean with shape [batch, time, 3]")
        clean = {
            key: torch.where(valid, targets[key], torch.zeros_like(targets[key]))
            for key in ("return", "variance")
        }
        denominator = valid.to(x.dtype).sum().clamp_min(1)

        def masked_mean(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum() / denominator

        return_loss = masked_mean(
            ((prediction["return"] - clean["return"]) / self.return_scale).square()
        )
        ratio = clean["variance"].clamp_min(self.variance_floor) / prediction["variance"].clamp_min(self.variance_floor)
        variance_loss = masked_mean(ratio - torch.log(ratio) - 1.0)
        loss = return_loss + .2 * variance_loss
        return loss, {"return": return_loss.detach(), "variance": variance_loss.detach()}

    @torch.no_grad()
    def predict_last(self, x):
        return {key: value[:, -1] for key, value in self(x).items()}
