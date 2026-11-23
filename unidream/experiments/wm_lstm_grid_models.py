"""Causal LSTM controls for the frozen seven-target prediction contract.

Only the temporal backbone changes. Regression, classification and shared
joint objectives, output constraints, masking and task weights are reused
from the registered full/joint grids. Every window starts from a fresh zero
recurrent state, and no bidirectional or future-input operation is present.

The default two-layer 256-wide LSTM is approximately parameter matched to
the 192-wide two-layer direct Transformer; exact counts are exported. The
common input encoder and 192-wide forecast MLP keep the existing dimensions.
"""
from __future__ import annotations

import torch
from torch import nn

from .wm_full_grid_models import _FullGridObjectives
from .wm_joint_grid_models import _JointGridObjectives
from .wm_seven_task_models import TASKS, SUPERVISED_WEIGHTS, _SevenTaskModel


FORMULATIONS = ("regression", "classification", "joint")


class _LSTMBackbone(_SevenTaskModel):
    def __init__(self, input_dim, model_cfg, scales):
        super().__init__(input_dim, model_cfg, scales)
        self.lstm_hidden_size = int(model_cfg.get("lstm_hidden_size", 256))
        self.lstm_layers = int(model_cfg.get("lstm_layers", 2))
        if self.lstm_hidden_size < 1 or self.lstm_layers < 1:
            raise ValueError("LSTM hidden size and layer count must be positive")
        if model_cfg.get("bidirectional", False) or model_cfg.get("lstm_bidirectional", False):
            raise ValueError("the registered forecasting LSTM must be unidirectional")
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.encoder_hidden), nn.ELU(),
            nn.Linear(self.encoder_hidden, self.d_model), nn.ELU(),
        )
        self.input_dropout = nn.Dropout(self.dropout)
        self.recurrent = nn.LSTM(
            input_size=self.d_model, hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers, batch_first=True, bidirectional=False,
            dropout=self.dropout if self.lstm_layers > 1 else 0.,
        )
        self.ln_final = nn.LayerNorm(self.lstm_hidden_size)
        self._initialize_head(self.lstm_hidden_size)

    def _representations(self, x):
        # No h/c argument: every call uses a zero initial state. The returned
        # state is deliberately discarded and never carried between windows.
        hidden, _ = self.recurrent(self.input_dropout(self.encoder(x)))
        return self.ln_final(hidden), {}

    def loss(self, x, base_target, targets, valid, next_return):
        # The two unused arguments retain the frozen direct-model interface.
        # LSTM has no WM reconstruction, latent-KL or next-return objective.
        prediction, raw, _ = self._predict_with_state(x)
        supervised, parts = self._supervised_loss(prediction, raw, targets, valid)
        return supervised, {**parts, "supervised": supervised.detach()}

    def _architecture_metadata(self):
        return {"backbone": "lstm", "input_dim": self.input_dim, "context": self.context,
            "encoder_hidden": self.encoder_hidden, "encoder_output_dim": self.d_model,
            "lstm_hidden_size": self.lstm_hidden_size, "lstm_layers": self.lstm_layers,
            "bidirectional": False, "recurrent_initial_state": "fresh zeros per input window",
            "persistent_recurrent_state": False, "forecast_mlp_hidden_dim": self.d_model,
            "representation_passes_per_batch": 1, "wm_auxiliary_evaluations_per_batch": 0,
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "capacity_policy": "approximately parameter matched; exact counts reported, no Test-based capacity selection",
            "same_seed_backbone_and_hidden_head_match_lstm_controls": True,
            "same_seed_native_output_rows_match_corresponding_lstm_controls": True,
            "transformer_checkpoint_reused": False,
            "supervised_task_weights": dict(SUPERVISED_WEIGHTS)}

    def checkpoint_metadata(self):
        return {**self._architecture_metadata(), "formulation": self.formulation,
            "tasks": list(TASKS), "output_rows": self.supervised_output_layer.out_features,
            "shared_parameters": True,
            "supervised_formulation_weights": {self.formulation: 1.}}


class LSTMFullGrid(_FullGridObjectives, _LSTMBackbone):
    """Seven scalar or seven native class heads on a causal recurrent state."""

    def supervised_gradient_coverage(self):
        gradient = self.supervised_output_layer.weight.grad
        if gradient is None:
            return dict.fromkeys(TASKS, False)
        covered = (torch.isfinite(gradient).all(dim=1)
                   & (gradient.abs().sum(dim=1) > 0)).detach().cpu()
        return {task: bool(covered[output_slice].all())
                for task, output_slice in self.task_output_slices.items()}


class LSTMJointGrid(_JointGridObjectives, _LSTMBackbone):
    """Both native head sets share exactly one recurrent representation."""

    def checkpoint_metadata(self):
        result = super().checkpoint_metadata()
        result.update(self._architecture_metadata())
        # The reused mixin describes its own Transformer controls as frozen;
        # here the initialization comparisons are within the new LSTM arms.
        result["initial_backbone_and_hidden_head_match_frozen_controls"] = False
        result["initial_native_output_rows_match_corresponding_frozen_controls"] = False
        result["initialization_control_scope"] = "same-seed LSTM regression/classification arms in this study"
        return result


def model_for(task, cfg, scales):
    """Construct a fresh study model; never load a checkpoint or infer mode."""
    if task.get("model") != "lstm" or task.get("formulation") not in FORMULATIONS:
        raise ValueError("an explicitly registered LSTM formulation is required")
    if task["formulation"] == "joint":
        return LSTMJointGrid(63, cfg["model"], scales)
    return LSTMFullGrid(63, cfg["model"], scales, task["formulation"])
