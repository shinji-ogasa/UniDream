"""Frozen-representation and zero-inflated hurdle ablations, research only."""
from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn

from .wm_tail_transfer_models import TailModel, active_specs, build_specs


def parameter_digest(parameters):
    """Tensor-content identity, independent of torch serialization metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenTailProbe(TailModel):
    """A fresh nonlinear readout on an exactly frozen causal market encoder.

    Native heads and their common forecast MLP are initialized afresh. Encoder
    operations run in eval/no-grad mode even while the readout is training.
    WM auxiliary terms remain constant diagnostic terms, with zero gradients;
    the underlying market model, prior, decoder and Student-t head are fixed.
    """
    def __init__(self, base_model, cfg, specs=None, scales=None, objectives="joint"):
        super().__init__(base_model.input_dim, cfg, specs, scales, base_model.variant, objectives)
        self._backbone_names = ["wm", "student_head"] if self.variant == "wm" else ["encoder", "drop", "final_norm"]
        self._backbone_names += ["recurrent"] if self.variant == "lstm" else ([] if self.variant == "wm" else ["blocks"])
        for name in self._backbone_names:
            setattr(self, name, copy.deepcopy(getattr(base_model, name)))
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(False)
        if self.variant == "transformer":
            self.position = nn.Parameter(base_model.position.detach().clone(), requires_grad=False)
        if self.variant == "wm":
            self.one_return_scale.copy_(base_model.one_return_scale)
        self._source_backbone_sha256 = self.backbone_digest()
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for name in getattr(self, "_backbone_names", []):
            getattr(self, name).eval()
        return self

    def _represent(self, x):
        with torch.no_grad():
            return super()._represent(x)

    def backbone_digest(self):
        selected = {name: value for name, value in self.state_dict().items()
                    if name.split(".")[0] in self._backbone_names or name == "position" or name == "one_return_scale"}
        return parameter_digest(selected)

    def assert_backbone_unchanged(self):
        if self.backbone_digest() != self._source_backbone_sha256:
            raise ValueError("frozen representation content changed during probe fitting")
        if any(p.requires_grad for name in self._backbone_names for p in getattr(self, name).parameters()):
            raise ValueError("a frozen representation parameter was enabled for training")
        return True

    def metadata(self):
        return {**super().metadata(), "probe": "fresh nonlinear readout on frozen source encoder",
            "source_backbone_sha256": self._source_backbone_sha256,
            "frozen_market_encoder": True, "source_native_heads_reused": False,
            "wm_auxiliary_terms_trainable": False,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


class HurdleTailModel(nn.Module):
    """Shared causal features, Bernoulli occurrence plus conditional magnitude.

    This is a genuine zero-inflated hurdle likelihood, not a pair of models
    trained on identical unconditional targets. Positive point heads predict
    E[Y | Y>0]; separate event heads predict P[Y>0]. Their product is the
    unconditional point forecast. Quantiles retain their original full-data
    pinball objectives; multiplying a quantile by a probability is avoided.
    """
    def __init__(self, input_dim, cfg, specs=None, scales=None, variant="transformer", objectives="joint",
                 hurdle_names=("mfe", "mae", "peak_dd", "dd_excess")):
        super().__init__()
        self.specs = active_specs(build_specs(cfg) if specs is None else specs, objectives)
        self.hurdle_names = [s["name"] for s in self.specs if s["name"] in hurdle_names and s["kind"] == "positive"]
        internal = [dict(s) for s in self.specs]
        self.event_names = {}
        for spec in self.specs:
            if spec["name"] not in self.hurdle_names:
                continue
            event = spec["name"] + "__occurs"
            self.event_names[spec["name"]] = event
            internal.append(dict(name=event, kind="binary", shape=list(spec["shape"]), task=spec["task"], levels=None))
        self.model = TailModel(input_dim, cfg, internal, scales, variant, "joint")
        self.input_dim, self.context, self.variant = input_dim, self.model.context, variant
        self.representation_dim = self.model.representation_dim
        self.objectives = objectives

    def _conditional_targets(self, targets, masks):
        transformed = dict(targets)
        transformed_mask = dict(masks)
        for name, event in self.event_names.items():
            valid = masks[name]
            observed = torch.where(valid, targets[name], torch.zeros_like(targets[name]))
            occurs = observed > 0
            transformed[event] = occurs.to(observed.dtype)
            transformed_mask[event] = valid
            transformed_mask[name] = valid & occurs
        return transformed, transformed_mask

    def loss(self, x, targets, masks, base_target=None, next_return=None, next_return_mask=None):
        target, mask = self._conditional_targets(targets, masks)
        return self.model.loss(x, target, mask, base_target, next_return, next_return_mask)

    def _combine(self, prediction):
        result = {spec["name"]: prediction[spec["name"]] for spec in self.specs}
        for name, event in self.event_names.items():
            result[name] = prediction[event] * prediction[name]
        return result

    def forward(self, x):
        return self._combine(self.model(x))

    @torch.no_grad()
    def predict_last(self, x):
        return self._combine(self.model.predict_last(x))

    @torch.no_grad()
    def occurrence_last(self, x):
        result = self.model.predict_last(x)
        return {name: result[event] for name, event in self.event_names.items()}

    def encode(self, x):
        return self.model.encode(x)

    @torch.no_grad()
    def latent_last(self, x):
        return self.model.latent_last(x)

    def supervised_gradient_coverage(self):
        return self.model.supervised_gradient_coverage()

    def metadata(self):
        return {**self.model.metadata(), "specs": self.specs, "objectives": self.objectives,
            "hurdle": True, "hurdle_names": self.hurdle_names,
            "internal_event_names": self.event_names,
            "point_forecast": "P(Y>0) * E[Y | Y>0]",
            "quantile_forecast": "ordinary unconditional quantile; no probability multiplication",
            "shared_hurdle_encoder": True,
            "conditional_magnitude_availability": "only observed strictly positive Train targets"}
