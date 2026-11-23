"""Shared-backbone joint regression/classification arms of the 336-cell grid.

Each of the seven registered outcomes has its scalar regression head and its
native classification head.  The supervised objective is exactly one half
of each frozen full-grid formulation's weighted objective.  The WM's existing
auxiliary objective is evaluated once and is not multiplied by that half.

The final output layer has 26 rows per horizon: seven scalar outputs followed
by nineteen classification logits (one binary crash logit, otherwise three).
Its regression and classification rows start at the same seeded values as
their respective frozen controls.  The larger output layer is an explicitly
reported capacity difference; all representation and hidden-head parameters
are shared and retain the controls' initial values.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .wm_seven_task_models import (
    TASKS, SUPERVISED_WEIGHTS, DirectSevenTaskTransformer, SevenTaskWM,
)


FORMULATION = "joint"
FORMULATION_WEIGHTS = {"regression": .5, "classification": .5}


class _JointGridObjectives:
    def __init__(self, input_dim: int, model_cfg: dict, scales: dict):
        super().__init__(input_dim, model_cfg, scales)
        self.formulation = FORMULATION
        self.formulation_weights = dict(FORMULATION_WEIGHTS)
        regression_layer = self.head[-1]
        regression_width = len(TASKS) * self.horizons
        classification_width = 19 * self.horizons
        # FullGrid's classification constructor makes this exact allocation
        # immediately after the common seven-output parent constructor.
        classification_layer = nn.Linear(self.d_model, classification_width)
        # Allocation of the concatenated layer must not change the RNG state
        # left by the two frozen constructors' common initialization sequence.
        with torch.random.fork_rng(devices=[]):
            joint_layer = nn.Linear(self.d_model, regression_width + classification_width)
        with torch.no_grad():
            joint_layer.weight[:regression_width].copy_(regression_layer.weight)
            joint_layer.bias[:regression_width].copy_(regression_layer.bias)
            joint_layer.weight[regression_width:].copy_(classification_layer.weight)
            joint_layer.bias[regression_width:].copy_(classification_layer.bias)
        self.head[-1] = joint_layer
        self.regression_task_output_slices = {
            task: slice(index * self.horizons, (index + 1) * self.horizons)
            for index, task in enumerate(TASKS)
        }
        self.classification_task_output_slices = {}
        offset = regression_width
        for task in TASKS:
            width = self.horizons * (1 if task == "crash" else 3)
            self.classification_task_output_slices[task] = slice(offset, offset + width)
            offset += width
        self.task_output_slices = {
            "regression": self.regression_task_output_slices,
            "classification": self.classification_task_output_slices,
        }

    def _decode(self, raw):
        regression = super()._decode(raw[..., :len(TASKS) * self.horizons])
        classification = {}
        for task in TASKS:
            value = raw[..., self.classification_task_output_slices[task]]
            if task == "crash":
                event_probability = torch.sigmoid(value)
                classification[task] = torch.stack((1. - event_probability, event_probability), dim=-1)
            else:
                logits = value.reshape(*value.shape[:-1], self.horizons, 3)
                classification[task] = F.softmax(logits, dim=-1)
        return {"regression": regression, "classification": classification}

    @torch.no_grad()
    def predict_last(self, x):
        return {formulation: {task: value[:, -1] for task, value in predictions.items()}
                for formulation, predictions in self(x).items()}

    def loss(self, x, base_target, targets, valid, next_return, labels=None):
        """Accept both native target mappings under a common boolean mask.

        ``targets`` may be ``{'regression': floats, 'classification': labels}``
        or the seven floating targets with a separate ``labels=`` mapping.
        All target tensors are [B,T,H].  The class labels are int64 and come
        from the same frozen Train-only threshold contract as the controls.
        """
        if not isinstance(targets, Mapping):
            raise ValueError("targets must be a mapping of regression and classification targets")
        if labels is None:
            if set(targets) != {"regression", "classification"}:
                raise ValueError("joint supervision requires regression and classification targets")
            supplied = targets
        else:
            supplied = {"regression": targets, "classification": labels}
        for formulation in FORMULATION_WEIGHTS:
            if not isinstance(supplied[formulation], Mapping) or set(supplied[formulation]) != set(TASKS):
                raise ValueError(f"{formulation} targets must contain exactly all seven registered tasks")
        return super().loss(x, base_target, supplied, valid, next_return)

    def _supervised_loss(self, prediction, raw, targets, valid):
        expected = (*raw.shape[:2], self.horizons)
        if valid.dtype != torch.bool or valid.shape != expected or valid.device != raw.device:
            raise ValueError("valid must be boolean [batch,time,H] on the model device")
        denominator = valid.sum().clamp_min(1).to(raw.dtype)

        def masked_mean(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum() / denominator

        losses = {}
        totals = {}
        for formulation, task_targets in targets.items():
            task_losses = {}
            for task in TASKS:
                target = task_targets[task]
                if target.shape != expected or target.device != raw.device:
                    raise ValueError(f"{formulation} {task} target must be [batch,time,H] on the model device")
                if formulation == "classification":
                    if target.dtype != torch.long:
                        raise ValueError(f"{task} class labels must have int64 dtype")
                elif not torch.is_floating_point(target):
                    raise ValueError(f"{task} regression targets must be floating point")
                # Remove NaNs and out-of-range class placeholders before any
                # arithmetic, matching the frozen control's mask semantics.
                clean = torch.where(valid, target, torch.zeros_like(target))
                if formulation == "classification":
                    logits = raw[..., self.classification_task_output_slices[task]]
                    if task == "crash":
                        if not bool(((clean >= 0) & (clean <= 1)).all()):
                            raise ValueError("crash class labels must be binary 0/1 at valid positions")
                        value = F.binary_cross_entropy_with_logits(logits, clean.to(raw.dtype), reduction="none")
                    else:
                        value = F.cross_entropy(logits.reshape(-1, 3), clean.reshape(-1), reduction="none").reshape(expected)
                elif task == "variance":
                    ratio = clean.clamp_min(self.variance_floor) / prediction[formulation][task].clamp_min(self.variance_floor)
                    value = ratio - torch.log(ratio) - 1.
                elif task == "crash":
                    value = (prediction[formulation][task] - clean).square()
                else:
                    value = ((prediction[formulation][task] - clean) / getattr(self, task + "_scale")).square()
                task_losses[task] = masked_mean(value)
                losses[f"{formulation}_{task}"] = task_losses[task].detach()
            totals[formulation] = sum(SUPERVISED_WEIGHTS[task] * task_losses[task] for task in TASKS)
            losses[f"{formulation}_supervised"] = totals[formulation].detach()
        total = sum(FORMULATION_WEIGHTS[formulation] * totals[formulation] for formulation in FORMULATION_WEIGHTS)
        return total, losses

    def supervised_gradient_coverage(self):
        """Every task/horizon row in both native heads must be trained."""
        gradient = self.supervised_output_layer.weight.grad
        names = {f"{formulation}_{task}": output_slice
                 for formulation, task_slices in self.task_output_slices.items()
                 for task, output_slice in task_slices.items()}
        if gradient is None:
            return dict.fromkeys(names, False)
        # One device transfer avoids fourteen separate MPS synchronizations
        # when the runner checks every native head after each optimizer step.
        covered = (torch.isfinite(gradient).all(dim=1)
                   & (gradient.abs().sum(dim=1) > 0)).detach().cpu()
        return {name: bool(covered[output_slice].all()) for name, output_slice in names.items()}

    def checkpoint_metadata(self):
        """Explicit architecture and capacity contract for research artifacts."""
        per_row = self.d_model + 1
        return {
            "formulation": FORMULATION,
            "tasks": list(TASKS),
            "shared_parameters": True,
            "shared_representation_for_both_formulations": True,
            "representation_passes_per_batch": 1,
            "supervised_formulation_weights": dict(FORMULATION_WEIGHTS),
            "regression_output_rows": 7 * self.horizons,
            "classification_output_rows": 19 * self.horizons,
            "joint_output_rows": 26 * self.horizons,
            "additional_parameters_vs_regression": 19 * self.horizons * per_row,
            "additional_parameters_vs_classification": 7 * self.horizons * per_row,
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "initial_backbone_and_hidden_head_match_frozen_controls": True,
            "initial_native_output_rows_match_corresponding_frozen_controls": True,
            "wm_auxiliary_evaluations_per_batch": int(isinstance(self, SevenTaskWM)),
        }


class DirectJointGrid(_JointGridObjectives, DirectSevenTaskTransformer):
    """One direct causal representation, seven scalar and seven class heads."""


class WMJointGrid(_JointGridObjectives, SevenTaskWM):
    """One market WM representation with the unchanged WM auxiliary losses."""
