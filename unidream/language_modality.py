"""
LanguageConditionedDynamics — Dreamer 4 Dynamics subclass that
concatenates a language token into the world-model token sequence.

Design:
  * Adds Modality.LANGUAGE=8 to the TokenLayout (one token per timestep).
  * Rebuilds BlockCausalTransformer with the extended modality_ids so
    the attention mask is aware of the new modality.
  * Accepts lang_tokens: (B, T, 1, lang_dim) in forward(); None → zeros.
  * lang_proj (Linear lang_dim → d_model) lives here; LanguageEncoder
    stays in language_encoder.py and is not imported here.
"""

from __future__ import annotations

import sys
import os

# Make dreamer4 importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer4"))

from typing import Optional, Tuple

import torch
import torch.nn as nn

from dreamer4.model import (
    BlockCausalTransformer,
    Dynamics,
    Modality,
    TokenLayout,
    add_sinusoidal_positions,
)
from .language_encoder import MINECLIP_DIM


class LanguageConditionedDynamics(Dynamics):
    """
    Drop-in replacement for Dynamics that adds a single language token
    per timestep to the world-model token sequence.

    Extra constructor args:
        lang_dim: dimensionality of the language embedding fed in
                  (default 512, matching MineCLIP text encoder).

    Extra forward arg:
        lang_tokens: (B, T, 1, lang_dim) or None
                     When None a zero token is used (unlabeled steps).
    """

    def __init__(
        self,
        *,
        lang_dim: int = MINECLIP_DIM,
        # Dreamer 4 Dynamics kwargs — must be forwarded explicitly so
        # we can re-use them when rebuilding the transformer.
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        depth: int,
        k_max: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        space_mode: str = "wm_agent_isolated",
    ):
        super().__init__(
            d_model=d_model,
            d_bottleneck=d_bottleneck,
            d_spatial=d_spatial,
            n_spatial=n_spatial,
            n_register=n_register,
            n_agent=n_agent,
            n_heads=n_heads,
            depth=depth,
            k_max=k_max,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            space_mode=space_mode,
        )

        # Extend the token layout with one LANGUAGE token at the end
        new_segments = list(self.layout.segments) + [(Modality.LANGUAGE, 1)]
        self.layout = TokenLayout(n_latents=0, segments=tuple(new_segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))
        self.lang_slice = sl[Modality.LANGUAGE]
        modality_ids = self.layout.modality_ids()

        # Rebuild transformer so SpaceSelfAttentionModality sees the
        # correct modality_ids (including LANGUAGE=8).
        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=0,
            modality_ids=modality_ids,
            space_mode=space_mode,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=False,
        )

        # Project language embedding into the world-model hidden space
        self.lang_proj = nn.Linear(lang_dim, self.d_model)

    # ------------------------------------------------------------------

    def forward(
        self,
        actions: Optional[torch.Tensor],          # (B, T, 16) or None
        step_idxs: torch.Tensor,                  # (B, T)
        signal_idxs: torch.Tensor,                # (B, T)
        packed_enc_tokens: torch.Tensor,          # (B, T, n_spatial, d_spatial)
        *,
        act_mask: Optional[torch.Tensor] = None,
        agent_tokens: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,  # (B, T, 1, lang_dim)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = packed_enc_tokens.shape[:2]

        # --- standard Dynamics tokens ---
        spatial_tokens = self.spatial_proj(packed_enc_tokens)   # (B,T,n_spatial,D)
        action_tokens = self.action_encoder(
            actions, batch_time_shape=(B, T), act_mask=act_mask, as_tokens=True
        )                                                        # (B,T,1,D)
        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)
        step_tok = self.step_embed(step_idxs.to(torch.long))[:, :, None, :]
        sig_tok = self.signal_embed(signal_idxs.to(torch.long))[:, :, None, :]

        # --- language token ---
        if lang_tokens is None:
            lang_tok = torch.zeros(
                B, T, 1, self.d_model,
                device=spatial_tokens.device,
                dtype=spatial_tokens.dtype,
            )
        else:
            lang_tok = self.lang_proj(lang_tokens)               # (B,T,1,D)

        # --- assemble & run transformer ---
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(
                    B, T, self.n_agent, self.d_model,
                    device=spatial_tokens.device,
                    dtype=spatial_tokens.dtype,
                )
            toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg, agent_tokens, lang_tok]
        else:
            toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg, lang_tok]

        tokens = torch.cat(toks, dim=2)          # (B, T, S, D)
        tokens = add_sinusoidal_positions(tokens)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)   # (B, T, n_spatial, d_spatial)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]   # (B, T, n_agent, d_model)

        return x1_hat, h_t
