"""Fresh all-regression/all-classification arms of the full task grid.

These isolated models reuse the frozen causal direct/WM representations and
unchanged WM auxiliary objectives. All seven supervised objectives change
together; there is no checkpoint warm start or auxiliary scalar-return head.
Both formulations use exactly one logit for binary crash probability.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .wm_seven_task_models import (
    TASKS, SUPERVISED_WEIGHTS, DirectSevenTaskTransformer, SevenTaskWM,
)


FORMULATIONS = ("regression", "classification")


class _FullGridObjectives:
    def __init__(self, input_dim: int, model_cfg: dict, scales: dict, formulation: str):
        if formulation not in FORMULATIONS:
            raise ValueError("formulation must be 'regression' or 'classification'")
        super().__init__(input_dim, model_cfg, scales)
        self.formulation = formulation
        self.task_output_slices = {}
        offset = 0
        for task in TASKS:
            width = self.horizons * (3 if formulation == "classification" and task != "crash" else 1)
            self.task_output_slices[task] = slice(offset, offset + width)
            offset += width
        if formulation == "classification":
            # The same shared representation -> d_model -> output MLP.
            # Only the final width changes from 7H scalar logits to 19H.
            self.head[-1] = nn.Linear(self.d_model, offset)

    def _decode(self, raw):
        if self.formulation == "regression":
            return super()._decode(raw)
        predictions = {}
        for task in TASKS:
            value = raw[..., self.task_output_slices[task]]
            if task == "crash":
                p = torch.sigmoid(value)
                predictions[task] = torch.stack((1. - p, p), dim=-1)
            else:
                logits = value.reshape(*value.shape[:-1], self.horizons, 3)
                predictions[task] = F.softmax(logits, dim=-1)
        return predictions

    def _supervised_loss(self, prediction, raw, targets, valid):
        expected = (*raw.shape[:2], self.horizons)
        if valid.dtype != torch.bool or valid.shape != expected or valid.device != raw.device:
            raise ValueError("valid must be boolean [batch,time,H] on the model device")
        denominator = valid.sum().clamp_min(1).to(raw.dtype)

        def masked_mean(value):
            return torch.where(valid, value, torch.zeros_like(value)).sum() / denominator

        losses = {}
        for task in TASKS:
            target = targets[task]
            if target.shape != expected or target.device != raw.device:
                raise ValueError(f"{task} target must be [batch,time,H] on the model device")
            if self.formulation == "classification" and target.dtype != torch.long:
                raise ValueError(f"{task} class labels must have int64 dtype")
            # Missing continuous values and out-of-range class placeholders
            # are removed before arithmetic or cross entropy.
            clean = torch.where(valid, target, torch.zeros_like(target))
            if self.formulation == "classification":
                logits = raw[..., self.task_output_slices[task]]
                if task == "crash":
                    if not bool(((clean >= 0) & (clean <= 1)).all()):
                        raise ValueError("crash class labels must be binary 0/1 at valid positions")
                    value = F.binary_cross_entropy_with_logits(logits, clean.to(raw.dtype), reduction="none")
                else:
                    value = F.cross_entropy(logits.reshape(-1, 3), clean.reshape(-1), reduction="none").reshape(expected)
            elif task == "variance":
                ratio = clean.clamp_min(self.variance_floor) / prediction[task].clamp_min(self.variance_floor)
                value = ratio - torch.log(ratio) - 1.
            elif task == "crash":
                value = (prediction[task] - clean).square()
            else:
                value = ((prediction[task] - clean) / getattr(self, task + "_scale")).square()
            losses[task] = masked_mean(value)
        total = sum(SUPERVISED_WEIGHTS[task] * losses[task] for task in TASKS)
        return total, {task: value.detach() for task, value in losses.items()}

    def supervised_gradient_coverage(self):
        """Every exposed task/horizon logit has a finite nonzero gradient."""
        gradient = self.supervised_output_layer.weight.grad
        if gradient is None:
            return dict.fromkeys(TASKS, False)
        covered = torch.isfinite(gradient).all(dim=1) & (gradient.abs().sum(dim=1) > 0)
        return {task: bool(covered[output_slice].all())
                for task, output_slice in self.task_output_slices.items()}


class DirectFullGrid(_FullGridObjectives, DirectSevenTaskTransformer):
    """Direct causal Transformer with seven MSE/QLIKE or CE/BCE heads.

    ``forward`` emits task -> [B,T,H] in regression mode, or [B,T,H,K]
    class probabilities in classification mode (K=2 crash, otherwise 3).
    ``predict_last`` removes only T. The inherited loss accepts continuous
    targets or int64 labels, respectively, with the same boolean [B,T,H] mask.
    """


class WMFullGrid(_FullGridObjectives, SevenTaskWM):
    """Same supervised contract plus unchanged reconstruction/KL/reward loss."""
