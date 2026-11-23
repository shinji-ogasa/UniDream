"""Fresh direct/WM classifiers for threshold-defined down/flat/up labels.

The threshold side input describes the *current* decision only. It is joined
to the final representation, after all causal sequence processing; it never
enters earlier token states or the WM auxiliary forecast/reconstruction loss.
Class probabilities use ordinary unweighted cross entropy. Any decision rule
that adjusts for train class priors belongs to the evaluation runner.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .wm_ablation_model import FeatureAblationWM
from .wm_comparison_models import DirectForecastTransformer


class _ContinuousClassifierCore(DirectForecastTransformer):
    """Reuse the continuous encoder and attention, removing forecast heads."""

    def __init__(self, input_dim, model_cfg, scales):
        super().__init__(input_dim, model_cfg, scales)
        self.head = nn.Identity()

    def forward(self, x):
        hidden = self.encoder(x)
        hidden = self.input_dropout(hidden + self.position_embedding[:, :x.shape[1]])
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, attn_mask=mask)
        return self.ln_final(hidden)


class ForecastClassifier(nn.Module):
    """Three-class model with a current-threshold side input.

    ``kind`` is ``direct`` or ``wm``. Labels are 0=down, 1=flat, 2=up.
    ``threshold_side`` must be ``[B,1]``, holding the externally prepared
    log(theta / outer-train return RMS). Direct uses final continuous state
    -> linear three-class logits. WM uses final (z,h) -> 192 ELU -> logits,
    and retains the registered full WM loss with coefficient one.

    Every constructor creates fresh parameters. No pretrained variance
    generator, research checkpoint or warm-start weights are loaded here.
    """

    def __init__(self, kind: str, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__()
        if kind not in ("direct", "wm"):
            raise ValueError("kind must be 'direct' or 'wm'")
        self.kind = kind
        self.input_dim = int(input_dim)
        self.context = int(model_cfg.get("context", 256))
        if kind == "direct":
            self.direct_core = _ContinuousClassifierCore(input_dim, model_cfg, scales)
            self.classification_head = nn.Linear(self.direct_core.d_model + 1, 3)
        else:
            self.wm = FeatureAblationWM(input_dim, model_cfg, scales)
            representation_dim = self.wm.wm.z_dim + self.wm.wm.d_model
            self.classification_head = nn.Sequential(
                nn.Linear(representation_dim + 1, 192),
                nn.ELU(),
                nn.Linear(192, 3),
            )

    def _validate_input(self, x, threshold_side):
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[-1] != self.input_dim:
            raise ValueError("x must have shape [nonempty batch, time, input_dim]")
        if not 0 < x.shape[1] <= self.context:
            raise ValueError("input sequence length must be between 1 and context")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("x must contain finite floating-point observations")
        if threshold_side.shape != (x.shape[0], 1):
            raise ValueError("threshold_side must have shape [batch, 1]")
        if (not torch.is_floating_point(threshold_side)
                or threshold_side.dtype != x.dtype or threshold_side.device != x.device
                or not torch.isfinite(threshold_side).all()):
            raise ValueError("threshold_side must be finite and match x dtype/device")

    def _representations(self, x):
        """Causal states before the decision-specific threshold is supplied."""
        if self.kind == "direct":
            return self.direct_core(x)
        _, state = self.wm(x)
        return torch.cat((state["z"], state["h"]), dim=-1)

    def _logits(self, representations, threshold_side):
        return self.classification_head(torch.cat(
            (representations[:, -1], threshold_side), dim=-1
        ))

    def forward(self, x, threshold_side):
        self._validate_input(x, threshold_side)
        return self._logits(self._representations(x), threshold_side)

    def _validate_auxiliary(self, x, base_target, targets_aux, valid_aux, next_return):
        expected = (*x.shape[:2], 3)
        if valid_aux.dtype != torch.bool or valid_aux.shape != expected:
            raise ValueError("valid_aux must be boolean with shape [batch, time, 3]")
        if base_target.shape != (*x.shape[:2], 5) or not torch.isfinite(base_target).all():
            raise ValueError("base_target must be finite with shape [batch, time, 5]")
        for key in ("return", "variance", "downside", "crash"):
            value = targets_aux[key]
            if value.shape != expected or not torch.isfinite(value[valid_aux]).all():
                raise ValueError(f"{key} auxiliary target must be finite at valid positions with shape {expected}")
        if next_return.shape != x.shape[:2] or not torch.isfinite(next_return).any():
            raise ValueError("next_return must have shape [batch, time] and a finite reward target")

    def loss(self, x, base_target, targets_aux, valid_aux, next_return, labels, threshold_side):
        """Direct: CE. WM: unchanged full WM auxiliary loss + 1.0 CE."""
        self._validate_input(x, threshold_side)
        if (labels.shape != (x.shape[0],) or labels.dtype != torch.long
                or labels.device != x.device or not ((labels >= 0) & (labels < 3)).all()):
            raise ValueError("labels must be int64 [batch] on x.device with values 0, 1 or 2")
        if self.kind == "direct":
            logits = self._logits(self._representations(x), threshold_side)
            ce = F.cross_entropy(logits, labels)
            return ce, {"ce": ce.detach(), "auxiliary": ce.detach().new_zeros(())}

        self._validate_auxiliary(x, base_target, targets_aux, valid_aux, next_return)
        # Observe the existing WM loss's single forward pass. A temporary hook
        # preserves its exact auxiliary equations and shares the SAME sampled
        # latent/dropout realization with CE, without a second encoder pass.
        captured = []

        def capture(_module, _arguments, output):
            captured.append(output[1])

        handle = self.wm.register_forward_hook(capture)
        try:
            auxiliary, auxiliary_parts = self.wm.loss(
                x, base_target, targets_aux, valid_aux, next_return
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("WM auxiliary loss must execute one forward pass")
        state = captured[0]
        representations = torch.cat((state["z"], state["h"]), dim=-1)
        ce = F.cross_entropy(self._logits(representations, threshold_side), labels)
        parts = {"ce": ce.detach(), "auxiliary": auxiliary.detach()}
        parts.update({"aux_" + key: value.detach() for key, value in auxiliary_parts.items()})
        return auxiliary + ce, parts

    @torch.no_grad()
    def predict_proba(self, x, threshold_side):
        return F.softmax(self(x, threshold_side), dim=-1)
