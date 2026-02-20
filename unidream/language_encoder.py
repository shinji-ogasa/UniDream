"""
MineCLIP text encoder wrapper.

Input:  list of B strings (one instruction per batch item)
Output: (B, T, 1, 512) float tensor — raw MineCLIP text embeddings,
        broadcast across T timesteps.

The 512→d_model projection lives in LanguageConditionedDynamics.lang_proj
so the encoder stays swappable (MineCLIP → T5, etc.) in Phase 2.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

MINECLIP_DIM = 512


class LanguageEncoder(nn.Module):
    """
    Wraps MineCLIP's text encoder to produce per-step language embeddings.

    If MineCLIP is not installed or no checkpoint is provided the encoder
    returns zero tensors of the correct shape so the rest of the forward
    pass can be smoke-tested without weights.
    """

    def __init__(
        self,
        mineclip_ckpt: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = MINECLIP_DIM
        self._model = None

        if mineclip_ckpt is not None:
            self._model = self._load_mineclip(mineclip_ckpt, device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        texts: List[str],
        T: int,
    ) -> torch.Tensor:
        """
        Args:
            texts: list of B instruction strings
            T:     number of timesteps to broadcast across

        Returns:
            (B, T, 1, 512) float tensor on the same device as the model
            (or CPU zeros when no checkpoint is loaded).
        """
        B = len(texts)

        if self._model is None:
            # Smoke-test stub: correct shape, all zeros
            return torch.zeros(B, T, 1, self.dim)

        with torch.no_grad():
            emb = self._model.encode_text(texts)  # (B, 512)

        # (B, 512) → (B, T, 1, 512)
        emb = emb[:, None, None, :].expand(B, T, 1, self.dim)
        return emb.float()

    @property
    def out_dim(self) -> int:
        return self.dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mineclip(ckpt_path: str, device: str):
        try:
            from mineclip import MineCLIP  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mineclip is not installed. "
                "Install with: pip install git+https://github.com/MineDojo/MineCLIP"
            ) from e

        cfg = dict(
            arch="vit_base_p16_fz.v2.t2",
            hidden_dim=512,
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            pool_type="attn.d2.ks0.m0.l0.e1",
            resolution=[160, 256],
        )
        model = MineCLIP(**cfg)
        model.load_ckpt(ckpt_path, strict=True)
        model.eval()
        model.to(device)
        # Freeze — text encoder is used as a fixed feature extractor in Phase 1
        for p in model.parameters():
            p.requires_grad_(False)
        return model
