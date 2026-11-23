"""Causal direct/WM predictors for the isolated seven-task benchmark.

The models share targets, horizon-specific Train scales, supervised losses,
attention width/depth and forecast-head structure. The WM additionally has
its categorical encoder, latent dynamics, reconstruction and next-return
objectives; its total parameter count is consequently not identical to the
continuous direct model. Neither model reads a checkpoint or selects on Test.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

from unidream.world_model.encoder import ObsDecoder, symlog
from unidream.world_model.transformer import TransformerWorldModel, twohot_loss
from .wm_ablation_model import _AttentionBlock


TASKS = ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery")
SUPERVISED_WEIGHTS = {
    "return": 1., "variance": .2, "downside": .2, "crash": .1,
    "dd_excess": .2, "advantage": .2, "recovery": .2,
}


class _SevenTaskModel(nn.Module):
    """Common task ordering, constrained outputs and loss implementation."""

    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context = int(model_cfg.get("context", 256))
        self.d_model = int(model_cfg.get("d_model", 192))
        self.n_heads = int(model_cfg.get("n_heads", 4))
        self.n_layers = int(model_cfg.get("n_layers", 2))
        self.encoder_hidden = int(model_cfg.get("encoder_hidden", self.d_model))
        self.d_ff = int(model_cfg.get("d_ff", 768))
        self.dropout = float(model_cfg.get("dropout", .1))
        if min(self.input_dim, self.context, self.d_model, self.n_heads,
               self.n_layers, self.encoder_hidden, self.d_ff) < 1:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        return_scale = torch.as_tensor(scales["return"], dtype=torch.float32)
        if return_scale.ndim != 1 or return_scale.numel() < 1:
            raise ValueError("return scales must be a nonempty horizon vector")
        self.horizons = int(return_scale.numel())
        for task in TASKS:
            if task == "crash":
                continue
            scale = torch.as_tensor(scales[task], dtype=torch.float32).clone()
            if (scale.shape != (self.horizons,) or not torch.isfinite(scale).all()
                    or not (scale > 0).all()):
                raise ValueError(f"{task} scales must contain H finite positive values")
            self.register_buffer(task + "_scale", scale.clamp_min(1e-8 if task == "variance" else 1e-4))
        self.register_buffer("variance_floor", torch.tensor(1e-12, dtype=torch.float32))

    def _initialize_head(self, representation_dim: int):
        self.head = nn.Sequential(
            nn.Linear(representation_dim, self.d_model), nn.ELU(),
            nn.Linear(self.d_model, len(TASKS) * self.horizons),
        )

    @property
    def supervised_output_layer(self) -> nn.Linear:
        """Rows are task-major: TASKS order, H consecutive rows per task."""
        return self.head[-1]

    def supervised_gradient_coverage(self) -> dict[str, bool]:
        """Report whether every horizon output has a finite nonzero gradient."""
        gradient = self.supervised_output_layer.weight.grad
        if gradient is None:
            return dict.fromkeys(TASKS, False)
        coverage = torch.isfinite(gradient).all(dim=1) & (gradient.abs().sum(dim=1) > 0)
        return {task: bool(coverage[i * self.horizons:(i + 1) * self.horizons].all())
                for i, task in enumerate(TASKS)}

    def _validate_input(self, x):
        if (x.ndim != 3 or x.shape[0] < 1 or x.shape[-1] != self.input_dim
                or not torch.is_floating_point(x)):
            raise ValueError("x must be floating [nonempty batch, time, input_dim]")
        if not 0 < x.shape[1] <= self.context:
            raise ValueError("sequence length must be between 1 and context")

    def _decode(self, raw):
        result = {}
        for index, task in enumerate(TASKS):
            value = raw[..., index * self.horizons:(index + 1) * self.horizons]
            if task == "crash":
                result[task] = torch.sigmoid(value)
            elif task in ("variance", "downside", "dd_excess"):
                result[task] = (F.softplus(value) + (1e-4 if task == "variance" else 0.)) * getattr(self, task + "_scale")
            elif task == "recovery":
                # The registered synthetic recovery outcome is in [-1,1].
                # Apply the same domain projection as the classical models.
                result[task] = (value * self.recovery_scale).clamp(-1., 1.)
            else:
                result[task] = value * getattr(self, task + "_scale")
        return result

    def _predict_with_state(self, x):
        self._validate_input(x)
        representation, state = self._representations(x)
        raw = self.head(representation)
        return self._decode(raw), raw, state

    def forward(self, x) -> dict[str, torch.Tensor]:
        """Seven predictions of shape [B,T,H]; crash is a probability."""
        return self._predict_with_state(x)[0]

    @torch.no_grad()
    def predict_last(self, x) -> dict[str, torch.Tensor]:
        return {task: value[:, -1] for task, value in self(x).items()}

    @torch.no_grad()
    def latent_last(self, x) -> torch.Tensor:
        """Final causal state; deterministic when the module is in eval mode."""
        self._validate_input(x)
        return self._representations(x)[0][:, -1]

    def _supervised_loss(self, prediction, raw, targets, valid):
        expected = prediction["return"].shape
        if valid.dtype != torch.bool or valid.shape != expected or valid.device != raw.device:
            raise ValueError("valid must be boolean [batch, time, H] on the model device")
        clean = {}
        for task in TASKS:
            if targets[task].shape != expected or targets[task].device != raw.device:
                raise ValueError(f"{task} target must have shape [batch, time, H] on the model device")
            # Remove missing labels BEFORE arithmetic, avoiding NaN gradients
            # even when a whole horizon has no valid observations in a batch.
            clean[task] = torch.where(valid, targets[task], torch.zeros_like(targets[task]))
        denominator = valid.sum().clamp_min(1).to(raw.dtype)

        def masked_mean(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum() / denominator

        losses = {}
        for task in TASKS:
            if task == "variance":
                ratio = clean[task].clamp_min(self.variance_floor) / prediction[task].clamp_min(self.variance_floor)
                losses[task] = masked_mean(ratio - torch.log(ratio) - 1.)
            elif task == "crash":
                crash_index = TASKS.index("crash") * self.horizons
                losses[task] = masked_mean(F.binary_cross_entropy_with_logits(
                    raw[..., crash_index:crash_index + self.horizons], clean[task], reduction="none"))
            else:
                losses[task] = masked_mean(((prediction[task] - clean[task]) / getattr(self, task + "_scale")).square())
        total = sum(SUPERVISED_WEIGHTS[task] * losses[task] for task in TASKS)
        return total, {task: value.detach() for task, value in losses.items()}


class DirectSevenTaskTransformer(_SevenTaskModel):
    """Continuous causal Transformer with the common seven objectives."""

    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__(input_dim, model_cfg, scales)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.encoder_hidden), nn.ELU(),
            nn.Linear(self.encoder_hidden, self.d_model), nn.ELU(),
        )
        self.position_embedding = nn.Parameter(torch.empty(1, self.context, self.d_model))
        nn.init.normal_(self.position_embedding, mean=0., std=.02)
        self.input_dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList([
            _AttentionBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.ln_final = nn.LayerNorm(self.d_model)
        self._initialize_head(self.d_model)

    def _representations(self, x):
        hidden = self.input_dropout(self.encoder(x) + self.position_embedding[:, :x.shape[1]])
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            hidden = block(hidden, attn_mask=mask)
        return self.ln_final(hidden), {}

    def loss(self, x, base_target, targets, valid, next_return):
        prediction, raw, _ = self._predict_with_state(x)
        loss, parts = self._supervised_loss(prediction, raw, targets, valid)
        return loss, {**parts, "supervised": loss.detach()}


class SevenTaskWM(_SevenTaskModel):
    """Market WM with seven common heads plus the existing WM objectives."""

    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__(input_dim, model_cfg, scales)
        self.wm = TransformerWorldModel(
            obs_dim=self.input_dim, act_dim=1,
            n_categoricals=int(model_cfg.get("n_categoricals", 32)),
            n_classes=int(model_cfg.get("n_classes", 32)),
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            d_ff=self.d_ff, dropout=self.dropout, max_seq_len=self.context,
            encoder_hidden=self.encoder_hidden, encoder_layers=2,
        )
        for block in self.wm.blocks:
            block.__class__ = _AttentionBlock
        self.wm.decoder = ObsDecoder(self.wm.z_dim, self.d_model, 5, self.encoder_hidden)
        self._initialize_head(self.wm.z_dim + self.d_model)

    def _representations(self, x):
        z, posterior = self.wm.encode(x)
        dynamics = self.wm(z, x.new_ones((*x.shape[:2], 1)))
        return torch.cat((z, dynamics["h"]), dim=-1), {"z": z, "posterior": posterior, **dynamics}

    def loss(self, x, base_target, targets, valid, next_return):
        prediction, raw, state = self._predict_with_state(x)
        supervised, parts = self._supervised_loss(prediction, raw, targets, valid)
        if base_target.shape != (*x.shape[:2], 5) or next_return.shape != x.shape[:2]:
            raise ValueError("base_target must be [B,T,5] and next_return [B,T]")
        reconstruction = F.mse_loss(self.wm.decoder(state["z"], state["h"]), symlog(base_target))
        if x.shape[1] > 1:
            posterior = state["posterior"][:, 1:]
            prior = state["prior_logits"][:, :-1]
            dyn_kl = td.kl_divergence(self.wm.encoder.get_dist(posterior.detach()),
                                    self.wm.encoder.get_dist(prior)).clamp_min(1).mean()
            rep_kl = td.kl_divergence(self.wm.encoder.get_dist(posterior),
                                    self.wm.encoder.get_dist(prior.detach())).clamp_min(1).mean()
        else:
            dyn_kl = state["prior_logits"].sum() * 0.
            rep_kl = state["posterior"].sum() * 0.
        reward_valid = torch.isfinite(next_return)
        if bool(reward_valid.any()):
            next_return_loss = twohot_loss(state["reward_logits"][reward_valid],
                                          next_return[reward_valid], self.wm.bins)
        else:
            next_return_loss = state["reward_logits"].sum() * 0.
        auxiliary = .3 * reconstruction + .2 * dyn_kl + .05 * rep_kl + .3 * next_return_loss
        parts.update({"supervised": supervised.detach(), "reconstruction": reconstruction.detach(),
                      "kl_dyn": dyn_kl.detach(), "kl_rep": rep_kl.detach(),
                      "next_return": next_return_loss.detach(), "auxiliary": auxiliary.detach()})
        return supervised + auxiliary, parts
