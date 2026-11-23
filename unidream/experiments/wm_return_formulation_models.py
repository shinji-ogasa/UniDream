"""Direct and WM arms replacing return regression with three-class CE.

The frozen seven-task predictors are reused without modification. Their six
other supervised terms, causal representations and WM auxiliary objectives
are retained. The original scalar-return output remains only for state-dict
compatibility with that architecture and receives no supervised objective;
it is deliberately excluded from the public prediction dictionary.

Class meanings/thresholds belong to the external Train-only label builder.
No class-prior correction, calibration, checkpoint loading or selection is
performed inside these modules.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .wm_seven_task_models import (
    TASKS, SUPERVISED_WEIGHTS, DirectSevenTaskTransformer, SevenTaskWM,
)


OTHER_TASKS = tuple(task for task in TASKS if task != "return")
_LABEL_KEY = "_return_class_labels"


class _ReturnClassificationMixin:
    """Share the SAME representation realization with all supervised heads."""

    trained_prediction_tasks = OTHER_TASKS
    unused_prediction_tasks = ("return",)

    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__(input_dim, model_cfg, scales)
        self.classification_head = nn.Sequential(
            nn.Linear(self.head[0].in_features, self.d_model), nn.ELU(),
            nn.Linear(self.d_model, self.horizons * 3),
        )

    @property
    def classification_output_layer(self) -> nn.Linear:
        """H consecutive blocks of three logits, ordered down/flat/up."""
        return self.classification_head[-1]

    def _classification_logits(self, raw):
        return raw[..., len(TASKS) * self.horizons:].reshape(*raw.shape[:-1], self.horizons, 3)

    def _predict_with_state(self, x):
        self._validate_input(x)
        representation, state = self._representations(x)
        original_raw = self.head(representation)
        # Append logits to the private raw tensor so the inherited loss can
        # call our objective without a second stochastic WM encoder pass.
        raw = torch.cat((original_raw, self.classification_head(representation)), dim=-1)
        return self._decode(original_raw), raw, state

    def forward(self, x):
        prediction, raw, _ = self._predict_with_state(x)
        return {**{task: prediction[task] for task in OTHER_TASKS},
                "class_probabilities": F.softmax(self._classification_logits(raw), dim=-1)}

    @torch.no_grad()
    def predict_proba(self, x):
        """Final-token probabilities [B,3] for H=1; [B,H,3] otherwise."""
        probability = self(x)["class_probabilities"][:, -1]
        return probability[:, 0] if self.horizons == 1 else probability

    @torch.no_grad()
    def predict_last(self, x):
        """Only the six supervised auxiliary outcomes, each [B,H]."""
        prediction, _, _ = self._predict_with_state(x)
        return {task: prediction[task][:, -1] for task in OTHER_TASKS}

    def loss(self, x, base_target, targets, valid, next_return, labels):
        expected = (*x.shape[:2], self.horizons)
        if labels.shape != expected or labels.dtype != torch.long or labels.device != x.device:
            raise ValueError("labels must be int64 [batch,time,H] on x.device")
        # The inherited Direct/WM loss performs one forward and invokes this
        # mixin's supervised objective. WM auxiliary equations stay identical.
        return super().loss(x, base_target, {**targets, _LABEL_KEY: labels}, valid, next_return)

    def _supervised_loss(self, prediction, raw, targets, valid):
        expected = prediction["return"].shape
        if valid.dtype != torch.bool or valid.shape != expected or valid.device != raw.device:
            raise ValueError("valid must be boolean [batch,time,H] on the model device")
        denominator = valid.sum().clamp_min(1).to(raw.dtype)

        def masked_mean(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum() / denominator

        losses = {}
        for task in OTHER_TASKS:
            value = targets[task]
            if value.shape != expected or value.device != raw.device:
                raise ValueError(f"{task} target must be [batch,time,H] on the model device")
            clean = torch.where(valid, value, torch.zeros_like(value))
            if task == "variance":
                ratio = clean.clamp_min(self.variance_floor) / prediction[task].clamp_min(self.variance_floor)
                losses[task] = masked_mean(ratio - torch.log(ratio) - 1.)
            elif task == "crash":
                start = TASKS.index(task) * self.horizons
                losses[task] = masked_mean(F.binary_cross_entropy_with_logits(
                    raw[..., start:start + self.horizons], clean, reduction="none"))
            else:
                losses[task] = masked_mean(((prediction[task] - clean) / getattr(self, task + "_scale")).square())
        labels = torch.where(valid, targets[_LABEL_KEY], torch.zeros_like(targets[_LABEL_KEY]))
        logits = self._classification_logits(raw)
        ce = masked_mean(F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), reduction="none").reshape(expected))
        total = ce + sum(SUPERVISED_WEIGHTS[task] * losses[task] for task in OTHER_TASKS)
        return total, {**{task: value.detach() for task, value in losses.items()}, "classification": ce.detach()}

    def supervised_gradient_coverage(self):
        coverage = super().supervised_gradient_coverage()
        coverage.pop("return")
        gradient = self.classification_output_layer.weight.grad
        coverage["classification"] = bool(
            gradient is not None and torch.isfinite(gradient).all()
            and (gradient.abs().sum(dim=1) > 0).all())
        return coverage


class DirectReturnClassifier(_ReturnClassificationMixin, DirectSevenTaskTransformer):
    """Continuous seven-task Transformer with CE replacing return MSE."""


class WMReturnClassifier(_ReturnClassificationMixin, SevenTaskWM):
    """Same replacement with unchanged reconstruction/KL/reward objectives."""
