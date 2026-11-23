"""Fresh Return-only controls and joint regression/classification predictors.

All three arms instantiate the same shared hidden MLP and four output rows per
horizon: one scalar return plus three class logits.  Only the registered
active losses train their rows.  Public forecasts exclude inactive heads.
The six other supervised outcomes are absent.  WM retains exactly the same
reconstruction, latent KL and next-one-bar-return auxiliary objectives in all
three arms, so its term "Return-only" refers to supervised forecast tasks.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .wm_seven_task_models import TASKS, DirectSevenTaskTransformer, SevenTaskWM


ARMS = ("reg_only", "class_only", "joint")
ARM_WEIGHTS = {"reg_only": {"return_mse": 1., "return_ce": 0.},
               "class_only": {"return_mse": 0., "return_ce": 1.},
               "joint": {"return_mse": .5, "return_ce": .5}}
_LABEL_KEY = "_return_class_labels"


class _ReturnMultitask:
    def __init__(self, input_dim: int, model_cfg: dict, scales: dict, arm: str):
        if arm not in ARMS:
            raise ValueError("arm must be reg_only, class_only or joint")
        if not isinstance(scales, Mapping) or "return" not in scales:
            raise ValueError("a positive Train-only return scale per horizon is required")
        # Frozen backbones share a parent constructor expecting seven scale
        # buffers; no other targets or scale estimates are actually used.
        # Supply return scales temporarily, then remove every unused buffer.
        temporary_scales = {task: scales["return"] for task in TASKS if task != "crash"}
        super().__init__(input_dim, model_cfg, temporary_scales)
        for task in TASKS:
            if task not in ("return", "crash"):
                delattr(self, task + "_scale")
        delattr(self, "variance_floor")
        self.arm = arm
        self.loss_weights = dict(ARM_WEIGHTS[arm])
        self.task_output_slices = {"regression": slice(0, self.horizons),
                                   "classification": slice(self.horizons, 4 * self.horizons)}
        self.prediction_availability = {"return": self.loss_weights["return_mse"] > 0,
                                        "class_probabilities": self.loss_weights["return_ce"] > 0}
        self.head_training_status = {name: "trained" if active else "inactive_untrained"
                                    for name, active in self.prediction_availability.items()}
        self.trained_prediction_tasks = tuple(name for name, active in (
            ("regression", self.prediction_availability["return"]),
            ("classification", self.prediction_availability["class_probabilities"])) if active)

    def _initialize_head(self, representation_dim: int):
        # Virtual dispatch during the inherited backbone constructor ensures
        # exactly identical RNG consumption/parameters for every arm.
        self.head = nn.Sequential(nn.Linear(representation_dim, self.d_model), nn.ELU(),
                                  nn.Linear(self.d_model, 4 * self.horizons))

    def _decode(self, raw):
        scalar = raw[..., :self.horizons] * self.return_scale
        logits = raw[..., self.horizons:].reshape(*raw.shape[:-1], self.horizons, 3)
        return {"return": scalar, "class_probabilities": F.softmax(logits, dim=-1)}

    def forward(self, x):
        predictions, _, _ = self._predict_with_state(x)
        return {name: value for name, value in predictions.items() if self.prediction_availability[name]}

    def loss(self, x, base_target, targets, valid, next_return, labels=None):
        """Return-only [B,T,H] targets with optional matching int64 class labels.

        Classification-only may pass an empty targets mapping; regression-only
        does not require labels.  Invalid placeholders outside ``valid`` are
        removed before arithmetic/cross entropy.  Inactive losses are never
        evaluated, avoiding both accidental supervision and zero-times-NaN.
        """
        if not isinstance(targets, Mapping) or any(name != "return" for name in targets):
            raise ValueError("supervised targets may contain only the return task")
        supplied = dict(targets)
        if self.prediction_availability["class_probabilities"]:
            expected = (*x.shape[:2], self.horizons)
            if labels is None or labels.shape != expected or labels.dtype != torch.long or labels.device != x.device:
                raise ValueError("active class labels must be int64 [B,T,H] on the model device")
            supplied[_LABEL_KEY] = labels
        return super().loss(x, base_target, supplied, valid, next_return)

    def _supervised_loss(self, prediction, raw, targets, valid):
        expected = (*raw.shape[:2], self.horizons)
        if valid.dtype != torch.bool or valid.shape != expected or valid.device != raw.device:
            raise ValueError("valid must be boolean [B,T,H] on the model device")
        denominator = valid.sum().clamp_min(1).to(raw.dtype)

        def masked_mean(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum() / denominator

        losses = {}
        if self.prediction_availability["return"]:
            if "return" not in targets:
                raise ValueError("active regression requires return targets")
            values = targets["return"]
            if values.shape != expected or values.device != raw.device or not torch.is_floating_point(values):
                raise ValueError("return targets must be floating [B,T,H] on the model device")
            clean = torch.where(valid, values, torch.zeros_like(values))
            losses["return_mse"] = masked_mean(((prediction["return"] - clean) / self.return_scale).square())
        if self.prediction_availability["class_probabilities"]:
            labels = torch.where(valid, targets[_LABEL_KEY], torch.zeros_like(targets[_LABEL_KEY]))
            logits = raw[..., self.horizons:].reshape(*raw.shape[:-1], self.horizons, 3)
            ce = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), reduction="none").reshape(expected)
            losses["return_ce"] = masked_mean(ce)
        total = sum(self.loss_weights[name] * value for name, value in losses.items())
        return total, {name: value.detach() for name, value in losses.items()}

    def supervised_gradient_coverage(self):
        """Coverage of active heads only; inactive output rows must stay unused."""
        gradient = self.supervised_output_layer.weight.grad
        if gradient is None:
            return dict.fromkeys(self.trained_prediction_tasks, False)
        covered = torch.isfinite(gradient).all(dim=1) & (gradient.abs().sum(dim=1) > 0)
        return {name: bool(covered[self.task_output_slices[name]].all()) for name in self.trained_prediction_tasks}


class DirectReturnMultitask(_ReturnMultitask, DirectSevenTaskTransformer):
    """Continuous causal Transformer: regression-only, class-only, or joint."""


class WMReturnMultitask(_ReturnMultitask, SevenTaskWM):
    """Same return arms plus unchanged WM reconstruction/KL/next-bar losses."""
