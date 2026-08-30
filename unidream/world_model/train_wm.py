"""世界モデル学習エントリポイント.

EnsembleWorldModel を WFO データ上で学習する。
損失: reconstruction + KL (free bits) + reward (twohot) + done (BCE)
     + IDM (Inverse Dynamics Model) auxiliary loss
     + N-step return prediction auxiliary loss
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unidream.data.dataset import SequenceDataset
from unidream.device import resolve_device
from unidream.experiments.checkpointing import atomic_torch_save
from unidream.experiments.chronological_oof import (
    conditional_path_or_artifact_enabled,
    strict_bool_value,
)
from unidream.world_model.ensemble import EnsembleWorldModel


_ACTIONLESS_CONTEXTS = frozenset(
    {
        "actionless",
        "benchmark",
        "benchmark_equivalent",
        "deployable",
        "deployment",
        "none",
    }
)
_ORACLE_CONTEXTS = frozenset({"dataset", "observed", "oracle"})


class TargetGradientCoverageError(RuntimeError):
    """Raised when a promotion-gated WM run lacks target/gradient coverage."""


def world_model_action_context(cfg: Optional[dict] = None) -> str:
    """Resolve whether WM action inputs come from a dataset or stay actionless.

    ``action_context`` is deliberately explicit for deployable configurations:
    an actionless WM uses the same benchmark-equivalent fallback as
    :meth:`WorldModelTrainer.encode_sequence(actions=None)`.  The historical
    Oracle/dataset behavior remains available when callers opt into
    ``action_context: oracle`` (or one of its aliases).
    """
    wm_cfg = (cfg or {}).get("world_model", {})
    raw_context = wm_cfg.get("action_context")
    if raw_context is None and "use_oracle_actions" in wm_cfg:
        raw_context = "oracle" if bool(wm_cfg["use_oracle_actions"]) else "actionless"
    if raw_context is None:
        # Keep callers that predate the explicit setting compatible.  Plan011
        # sets action_context explicitly to actionless in its YAML configs.
        raw_context = "oracle"
    if isinstance(raw_context, bool):
        raw_context = "oracle" if raw_context else "actionless"
    context = str(raw_context).strip().lower().replace("-", "_").replace(" ", "_")
    if context in _ACTIONLESS_CONTEXTS:
        return "actionless"
    if context in _ORACLE_CONTEXTS:
        return "oracle"
    valid = ", ".join(sorted(_ACTIONLESS_CONTEXTS | _ORACLE_CONTEXTS))
    raise ValueError(f"Unsupported world_model.action_context={raw_context!r}; expected one of: {valid}")


def world_model_uses_dataset_actions(cfg: Optional[dict] = None) -> bool:
    """Return whether WM training/encoding may consume dataset actions."""
    return world_model_action_context(cfg) == "oracle"


class IDMHead(nn.Module):
    """Inverse Dynamics Model: (z_t, z_{t+1}) → action logits.

    エンコーダが行動識別に有用な情報を保持することを強制する。
    NeurIPS 2023 で BC 事前学習の表現学習に有効と示された手法。
    """

    def __init__(self, z_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t:  (B, T, z_dim)
            z_t1: (B, T, z_dim)
        Returns:
            logits: (B, T, n_actions)
        """
        return self.net(torch.cat([z_t, z_t1], dim=-1))


class ReturnHead(nn.Module):
    """N-step return prediction: (z_t, h_t) → scalar.

    WM の潜在表現に将来リターンの情報を埋め込む。
    """

    def __init__(self, z_dim: int, d_model: int, hidden: int, out_dim: int = 1, obs_dim: int = 0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + d_model + self.obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: (B, T, z_dim)
            h: (B, T, d_model)
        Returns:
            pred: (B, T, out_dim)
        """
        parts = [z, h]
        if self.obs_dim > 0:
            if obs is None:
                obs_part = torch.zeros(*z.shape[:-1], self.obs_dim, dtype=z.dtype, device=z.device)
            else:
                obs_part = torch.nan_to_num(obs[..., : self.obs_dim].to(dtype=z.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(obs_part)
        return self.net(torch.cat(parts, dim=-1))


class RegimeHead(nn.Module):
    """Regime probability prediction from latent state."""

    def __init__(self, z_dim: int, d_model: int, hidden: int, regime_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + d_model, hidden),
            nn.ELU(),
            nn.Linear(hidden, regime_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h], dim=-1))


def build_ensemble(obs_dim: int, cfg: dict) -> EnsembleWorldModel:
    """config dict から EnsembleWorldModel を構築する."""
    wm_cfg = cfg.get("world_model", {})
    return EnsembleWorldModel(
        n_models=wm_cfg.get("n_ensemble", 3),
        disagree_scale=wm_cfg.get("disagree_scale", 0.1),
        obs_dim=obs_dim,
        act_dim=cfg.get("actions", {}).get("dim", 1),
        n_categoricals=wm_cfg.get("n_categoricals", 32),
        n_classes=wm_cfg.get("n_classes", 32),
        d_model=wm_cfg.get("d_model", 512),
        n_heads=wm_cfg.get("n_heads", 8),
        n_layers=wm_cfg.get("n_layers", 4),
        d_ff=wm_cfg.get("d_ff", 2048),
        dropout=wm_cfg.get("dropout", 0.1),
        max_seq_len=wm_cfg.get("max_seq_len", 256),
        n_bins=wm_cfg.get("n_bins", 255),
        bin_low=wm_cfg.get("bin_range", [-20.0, 20.0])[0],
        bin_high=wm_cfg.get("bin_range", [-20.0, 20.0])[1],
        unimix_ratio=wm_cfg.get("unimix_ratio", 0.01),
        encoder_hidden=wm_cfg.get("encoder_hidden", 256),
        encoder_layers=wm_cfg.get("encoder_layers", 2),
    )


class WorldModelTrainer:
    """世界モデルの学習ループ.

    Args:
        ensemble: EnsembleWorldModel
        cfg: config 辞書
        device: 計算デバイス
        coverage_context: optional run/fold/phase provenance for coverage rows
    """

    def __init__(
        self,
        ensemble: EnsembleWorldModel,
        cfg: Optional[dict] = None,
        device: str = "cpu",
        coverage_context: Mapping[str, object] | None = None,
    ):
        self.ensemble = ensemble
        self.checkpoint_metadata: dict[str, object] = {}
        self.device = torch.device(resolve_device(device))
        self.ensemble.to(self.device)
        cfg = cfg or {}
        wm_cfg = cfg.get("world_model", {})
        # A strict artifact request is itself a hard stop while this legacy
        # trainer remains the only implementation.  Do not let a top-level
        # ``require_conditional_oof_artifact`` silently select this path.
        conditional_enabled = conditional_path_or_artifact_enabled(cfg)
        explicit_coverage_gate = False
        option_sections = [("config", cfg), ("world_model", wm_cfg)]
        for section_name in ("oracle", "ac", "bc"):
            section = cfg.get(section_name)
            if isinstance(section, Mapping):
                option_sections.append((section_name, section))
        for section_name, section in option_sections:
            if "require_target_gradient_coverage" in section:
                explicit_coverage_gate = explicit_coverage_gate or strict_bool_value(
                    section["require_target_gradient_coverage"],
                    name=f"{section_name}.require_target_gradient_coverage",
                )
        self.require_target_gradient_coverage = conditional_enabled or explicit_coverage_gate
        self._coverage_context: dict[str, object] = {}
        for section_name, section in (("config", cfg), ("world_model", wm_cfg)):
            configured_context = section.get("coverage_context")
            if configured_context is None:
                continue
            if not isinstance(configured_context, Mapping):
                raise ValueError(f"{section_name}.coverage_context must be a mapping")
            self._coverage_context.update(dict(configured_context))
        if coverage_context is not None:
            if not isinstance(coverage_context, Mapping):
                raise ValueError("coverage_context must be a mapping")
            self._coverage_context.update(dict(coverage_context))
        self.action_context = world_model_action_context(cfg)
        self.use_dataset_actions = self.action_context == "oracle"

        self.lr = wm_cfg.get("lr", 1e-4)
        self.batch_size = wm_cfg.get("batch_size", 32)
        self.max_steps = wm_cfg.get("max_steps", 100_000)
        self.grad_clip = wm_cfg.get("grad_clip", 100.0)
        # Keep the default portable across macOS spawn and Linux fork.  A
        # caller can opt into workers explicitly in a self-contained config.
        self.num_workers = int(wm_cfg.get("num_workers", 0))
        self.log_interval = cfg.get("logging", {}).get("log_interval", 1000)
        val_max_batches = wm_cfg.get("val_max_batches")
        if val_max_batches is None:
            self.val_max_batches: int | None = None
        else:
            self.val_max_batches = int(val_max_batches)
            if self.val_max_batches <= 0:
                raise ValueError("world_model.val_max_batches must be positive when configured")

        # 損失ハイパーパラメータ
        self.free_bits = wm_cfg.get("free_bits", 1.0)
        self.dyn_scale = wm_cfg.get("dyn_scale", 0.5)
        self.rep_scale = wm_cfg.get("rep_scale", 0.1)
        self.recon_scale = wm_cfg.get("recon_scale", 1.0)
        self.reward_scale = wm_cfg.get("reward_scale", 1.0)
        self.done_scale = wm_cfg.get("done_scale", 1.0)
        self.aux_use_raw_features = bool(wm_cfg.get("aux_use_raw_features", False))
        model0 = self.ensemble.models[0] if getattr(self.ensemble, "models", None) else None
        self.aux_raw_obs_dim = int(getattr(model0, "obs_dim", 0)) if self.aux_use_raw_features else 0

        # Auxiliary loss スケール
        self.idm_scale = wm_cfg.get("idm_scale", 0.0)
        self.return_scale = wm_cfg.get("return_scale", 0.0)
        self.return_horizon = wm_cfg.get("return_horizon", 10)
        self.return_horizons = [
            int(h) for h in wm_cfg.get("return_horizons", [self.return_horizon])
        ]
        self.return_include_current = bool(wm_cfg.get("return_include_current", True))
        self.return_target_scale = float(wm_cfg.get("return_target_scale", 1.0))
        self.vol_scale = float(wm_cfg.get("vol_scale", 0.0))
        self.drawdown_scale = float(wm_cfg.get("drawdown_scale", 0.0))
        self.crash_scale = float(wm_cfg.get("crash_scale", 0.0))
        self.crash_threshold = float(wm_cfg.get("crash_threshold", 0.012))
        self.crash_pos_weight = float(wm_cfg.get("crash_pos_weight", 1.0))
        self.drawdown_excess_scale = float(wm_cfg.get("drawdown_excess_scale", 0.0))
        self.drawdown_excess_threshold = float(
            wm_cfg.get("drawdown_excess_threshold", self.crash_threshold)
        )
        self.position_utility_scale = float(wm_cfg.get("position_utility_scale", 0.0))
        self.position_utility_positions = [
            float(x) for x in wm_cfg.get("position_utility_positions", [0.0, 0.5, 0.85, 1.0, 1.06])
        ]
        self.position_utility_horizon = int(wm_cfg.get("position_utility_horizon", 32))
        self.position_utility_dd_penalty = float(wm_cfg.get("position_utility_dd_penalty", 1.0))
        self.position_utility_dd_improve_reward = float(wm_cfg.get("position_utility_dd_improve_reward", 0.0))
        self.position_utility_vol_penalty = float(wm_cfg.get("position_utility_vol_penalty", 0.25))
        self.position_utility_target_scale = float(
            wm_cfg.get("position_utility_target_scale", self.return_target_scale)
        )
        self.position_utility_positive_weight = float(wm_cfg.get("position_utility_positive_weight", 0.0))
        self.position_utility_nonbench_weight = float(wm_cfg.get("position_utility_nonbench_weight", 0.0))
        self.position_utility_rank_scale = float(wm_cfg.get("position_utility_rank_scale", 0.0))
        self.position_utility_rank_margin = float(wm_cfg.get("position_utility_rank_margin", 0.0))
        self.overweight_advantage_scale = float(wm_cfg.get("overweight_advantage_scale", 0.0))
        self.recovery_scale = float(wm_cfg.get("recovery_scale", 0.0))
        self.risk_horizons = [
            int(h) for h in wm_cfg.get("risk_horizons", self.return_horizons)
        ]
        self.risk_target_scale = float(wm_cfg.get("risk_target_scale", 1.0))
        self.control_target_scale = float(wm_cfg.get("control_target_scale", self.risk_target_scale))
        self.overweight_delta = float(wm_cfg.get("overweight_delta", 0.25))
        self.overweight_drawdown_penalty = float(wm_cfg.get("overweight_drawdown_penalty", 0.35))
        self.recovery_drawdown_penalty = float(wm_cfg.get("recovery_drawdown_penalty", 0.50))
        self.regime_aux_scale = wm_cfg.get("regime_aux_scale", 0.0)
        self.regime_dim = int(wm_cfg.get("regime_dim", 0))

        # コストパラメータ（net_return 計算に使用）
        costs_cfg = cfg.get("costs", {})
        self.cost_rate = (
            (costs_cfg.get("spread_bps", 5.0) / 10000) / 2
            + costs_cfg.get("fee_rate", 0.0004)
            + (costs_cfg.get("slippage_bps", 2.0) / 10000)
        )
        reward_cfg = cfg.get("reward", {})
        self.reward_mode = wm_cfg.get("reward_mode", reward_cfg.get("mode", "absolute"))
        self.benchmark_position = reward_cfg.get("benchmark_position", 1.0)

        # Auxiliary heads（スケール > 0 の場合のみ構築）
        z_dim = ensemble.get_z_dim()
        d_model = ensemble.get_d_model()
        self.action_values = torch.tensor(
            cfg.get("actions", {}).get("values", [-1.0, -0.5, 0.0, 0.5, 1.0]),
            dtype=torch.float32,
            device=self.device,
        )
        n_actions = int(cfg.get("actions", {}).get("n", len(self.action_values)))
        aux_params: list[nn.Parameter] = []

        if self.idm_scale > 0.0:
            self.idm_head = IDMHead(z_dim, hidden=256, n_actions=n_actions).to(self.device)
            aux_params.extend(self.idm_head.parameters())
        else:
            self.idm_head = None

        if self.return_scale > 0.0:
            self.return_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.return_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.return_head.parameters())
        else:
            self.return_head = None

        if self.vol_scale > 0.0:
            self.vol_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.vol_head.parameters())
        else:
            self.vol_head = None

        if self.drawdown_scale > 0.0:
            self.drawdown_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.drawdown_head.parameters())
        else:
            self.drawdown_head = None

        if self.crash_scale > 0.0:
            self.crash_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.crash_head.parameters())
        else:
            self.crash_head = None

        if self.drawdown_excess_scale > 0.0:
            self.drawdown_excess_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.drawdown_excess_head.parameters())
        else:
            self.drawdown_excess_head = None

        if self.position_utility_scale > 0.0:
            self.position_utility_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.position_utility_positions),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.position_utility_head.parameters())
        else:
            self.position_utility_head = None

        if self.overweight_advantage_scale > 0.0:
            self.overweight_advantage_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.overweight_advantage_head.parameters())
        else:
            self.overweight_advantage_head = None

        if self.recovery_scale > 0.0:
            self.recovery_head = ReturnHead(
                z_dim,
                d_model,
                hidden=256,
                out_dim=len(self.risk_horizons),
                obs_dim=self.aux_raw_obs_dim,
            ).to(self.device)
            aux_params.extend(self.recovery_head.parameters())
        else:
            self.recovery_head = None

        if self.regime_aux_scale > 0.0 and self.regime_dim > 0:
            self.regime_head = RegimeHead(z_dim, d_model, hidden=256, regime_dim=self.regime_dim).to(self.device)
            aux_params.extend(self.regime_head.parameters())
        else:
            self.regime_head = None

        if bool(wm_cfg.get("freeze_ensemble", False)):
            for param in self.ensemble.parameters():
                param.requires_grad_(False)
        if bool(wm_cfg.get("freeze_standard_predictive_heads", False)):
            for head in (
                self.return_head,
                self.vol_head,
                self.drawdown_head,
                self.crash_head,
                self.drawdown_excess_head,
                self.position_utility_head,
                self.regime_head,
            ):
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad_(False)

        self._all_params = list(self.ensemble.parameters()) + aux_params
        self.optimizer = torch.optim.Adam(
            [p for p in self._all_params if p.requires_grad],
            lr=self.lr,
        )

        self.global_step = 0
        self.loss_history: list[dict] = []
        self._coverage_sequence_length: int | None = None
        self._coverage_stats: dict[tuple[str, object, int | None], dict[str, object]] = {}
        self._coverage_step_executed: set[tuple[str, object, int | None]] = set()
        self._coverage_step_finite: dict[tuple[str, object, int | None], bool] = {}
        self._active_coverage_context: dict[str, object] = {}

    def _active_auxiliary_heads(self) -> dict[str, nn.Module]:
        """Return all auxiliary heads that participate in this trainer."""
        return {
            name: head
            for name in (
                "idm_head",
                "return_head",
                "vol_head",
                "drawdown_head",
                "crash_head",
                "drawdown_excess_head",
                "position_utility_head",
                "overweight_advantage_head",
                "recovery_head",
                "regime_head",
            )
            if (head := getattr(self, name, None)) is not None
        }

    def _coverage_specs(self) -> list[dict[str, object]]:
        """Return one machine-readable coverage row for every enabled output.

        The historical training loop used to report a finite aggregate loss as
        evidence that a future head was trained.  That is insufficient for a
        sequence whose right edge has no future label (for example horizon 64
        in a future-only sequence of length 64).  Coverage is tracked per
        output so a single valid horizon cannot hide a zero-coverage sibling.
        """
        specs: list[dict[str, object]] = []

        def add(head: str, horizon: object, output_index: int | None = None, **extra: object) -> None:
            row = {
                "head": head,
                "module": f"{head}_head",
                "horizon": horizon,
                "output_index": output_index,
            }
            row.update(extra)
            specs.append(row)

        if self.idm_head is not None:
            add("idm", 1)
        if self.return_head is not None:
            for idx, horizon in enumerate(self.return_horizons):
                add("return", int(horizon), idx)
        for head_name, horizons in (
            ("vol", self.risk_horizons),
            ("drawdown", self.risk_horizons),
            ("crash", self.risk_horizons),
            ("drawdown_excess", self.risk_horizons),
            ("overweight_advantage", self.risk_horizons),
            ("recovery", self.risk_horizons),
        ):
            if getattr(self, f"{head_name}_head", None) is not None:
                for idx, horizon in enumerate(horizons):
                    add(head_name, int(horizon), idx)
        if self.position_utility_head is not None:
            for idx, position in enumerate(self.position_utility_positions):
                add(
                    "position_utility",
                    int(self.position_utility_horizon),
                    idx,
                    position=float(position),
                )
        if self.regime_head is not None:
            add("regime", "current", None, regime_dim=int(self.regime_dim))
        return specs

    def _start_target_gradient_coverage(
        self,
        sequence_length: int | None = None,
        coverage_context: Mapping[str, object] | None = None,
    ) -> None:
        """Reset target/gradient counters for one training invocation."""
        self._coverage_sequence_length = None if sequence_length is None else int(sequence_length)
        self._coverage_stats: dict[tuple[str, object, int | None], dict[str, object]] = {}
        self._coverage_step_executed = set()
        self._coverage_step_finite = {}
        self._active_coverage_context = dict(self._coverage_context)
        if coverage_context is not None:
            if not isinstance(coverage_context, Mapping):
                raise ValueError("coverage_context must be a mapping")
            self._active_coverage_context.update(dict(coverage_context))
        for spec in self._coverage_specs():
            key = (str(spec["head"]), spec["horizon"], spec.get("output_index"))
            row = dict(spec)
            row.update(
                {
                    "sequence_length": self._coverage_sequence_length,
                    "total_target_slots": 0,
                    "masked_target_slots": 0,
                    "valid_targets": 0,
                    "finite_targets": 0,
                    "finite_masked_targets": 0,
                    "finite_loss_steps": 0,
                    "gradient_steps": 0,
                    "nonzero_gradient_steps": 0,
                }
            )
            if self._active_coverage_context:
                row["context"] = dict(self._active_coverage_context)
                for field in ("run", "run_id", "fold", "phase"):
                    if field in self._active_coverage_context:
                        row[field] = self._active_coverage_context[field]
            self._coverage_stats[key] = row

    @staticmethod
    def _coverage_tensor_values(
        target: torch.Tensor,
        mask: torch.Tensor,
        output_index: int | None,
    ) -> tuple[int, int, int, int, int]:
        """Count slots, masks, and finite values without hiding invalid tails."""
        target = target.detach()
        mask = mask.detach().to(dtype=torch.bool)
        if output_index is not None:
            target = target[..., output_index]
            mask = mask[..., output_index]
        finite = torch.isfinite(target)
        return (
            int(target.numel()),
            int(mask.sum().item()),
            int((mask & finite).sum().item()),
            int(finite.sum().item()),
            int((mask & finite).sum().item()),
        )

    def _accumulate_target_coverage(
        self,
        head: str,
        target: torch.Tensor,
        mask: torch.Tensor,
        horizons: list[object] | tuple[object, ...] | None = None,
    ) -> None:
        """Accumulate target/mask counts for one auxiliary head."""
        if not getattr(self, "_coverage_stats", None):
            return
        if horizons is None:
            horizons = [key[1] for key in self._coverage_stats if key[0] == head]
        for output_index, horizon in enumerate(horizons):
            key = (head, horizon, output_index)
            # IDM and regime have no per-output horizon dimension.  Their
            # rows use ``output_index=None`` and are accumulated as a whole.
            if key not in self._coverage_stats:
                key = (head, horizon, None)
            row = self._coverage_stats.get(key)
            if row is None:
                continue
            slots, masked, valid, finite, finite_masked = self._coverage_tensor_values(
                target,
                mask,
                output_index if key[2] is not None else None,
            )
            row["total_target_slots"] = int(row["total_target_slots"]) + slots
            row["masked_target_slots"] = int(row["masked_target_slots"]) + masked
            row["valid_targets"] = int(row["valid_targets"]) + valid
            row["finite_targets"] = int(row["finite_targets"]) + finite
            row["finite_masked_targets"] = int(row["finite_masked_targets"]) + finite_masked
            self._coverage_step_executed.add(key)

    def _mark_head_loss_coverage(
        self,
        head: str,
        loss: torch.Tensor,
        horizons: list[object] | tuple[object, ...] | None = None,
    ) -> None:
        """Record per-head execution and loss finiteness for this step."""
        if not getattr(self, "_coverage_stats", None):
            return
        if horizons is None:
            keys = [key for key in self._coverage_stats if key[0] == head]
        else:
            keys = []
            for output_index, horizon in enumerate(horizons):
                key = (head, horizon, output_index)
                if key not in self._coverage_stats:
                    key = (head, horizon, None)
                if key in self._coverage_stats:
                    keys.append(key)
        finite = bool(torch.isfinite(loss.detach()))
        for key in keys:
            self._coverage_step_executed.add(key)
            self._coverage_step_finite[key] = finite

    @staticmethod
    def _has_nonzero_parameter_gradient(
        module: nn.Module,
        output_index: int | None = None,
        atol: float = 1e-12,
    ) -> bool:
        if output_index is not None:
            # Multi-output heads share hidden layers.  Looking at any shared
            # parameter would falsely give a zero-mask horizon credit because
            # another horizon trained it.  The final projection row isolates
            # the output whose target/mask is being audited.
            linear_layers = [submodule for submodule in module.modules() if isinstance(submodule, nn.Linear)]
            if linear_layers:
                final = linear_layers[-1]
                if output_index >= int(final.out_features):
                    return False
                for parameter in (final.weight, final.bias):
                    if parameter is not None and parameter.grad is not None:
                        row = parameter.grad[output_index]
                        if bool(torch.any(torch.isfinite(row) & (row.abs() > float(atol)))):
                            return True
                return False
        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            return False
        return any(
            param.grad is not None
            and bool(torch.any(torch.isfinite(param.grad) & (param.grad.abs() > float(atol))))
            for param in params
        )

    def _record_gradient_coverage(self, loss: torch.Tensor | None = None) -> None:
        """Record gradients only for heads that executed on this step."""
        del loss  # Per-head loss finiteness is tracked by _mark_head_loss_coverage.
        executed = getattr(self, "_coverage_step_executed", set())
        for key, row in getattr(self, "_coverage_stats", {}).items():
            if key not in executed:
                continue
            head_name = str(key[0])
            module = getattr(self, f"{head_name}_head", None)
            output_index = key[2]
            row["gradient_steps"] = int(row["gradient_steps"]) + 1
            if self._coverage_step_finite.get(key, False):
                row["finite_loss_steps"] = int(row["finite_loss_steps"]) + 1
            if module is not None and self._has_nonzero_parameter_gradient(module, output_index=output_index):
                row["nonzero_gradient_steps"] = int(row["nonzero_gradient_steps"]) + 1
        self._coverage_step_executed.clear()
        self._coverage_step_finite.clear()

    def target_gradient_coverage(self) -> list[dict[str, object]]:
        """Return finalized per-head target and nonzero-gradient coverage.

        ``status=pass`` requires both a positive number of valid, finite target
        values and at least one observed nonzero gradient step.  A finite loss
        alone never promotes a row.
        """
        rows: list[dict[str, object]] = []
        for row in getattr(self, "_coverage_stats", {}).values():
            item = dict(row)
            total = int(item.get("total_target_slots", 0))
            masked = int(item.get("masked_target_slots", 0))
            valid = int(item.get("valid_targets", 0))
            finite = int(item.get("finite_targets", 0))
            gradients = int(item.get("gradient_steps", 0))
            nonzero = int(item.get("nonzero_gradient_steps", 0))
            finite_loss_steps = int(item.get("finite_loss_steps", 0))
            finite_masked = int(item.get("finite_masked_targets", 0))
            item["mask_fraction"] = float(masked / total) if total else 0.0
            item["mask_rate"] = item["mask_fraction"]
            item["invalid_mask_fraction"] = float(1.0 - item["mask_fraction"]) if total else 1.0
            item["target_coverage"] = float(valid / total) if total else 0.0
            item["gradient_coverage"] = float(nonzero / gradients) if gradients else 0.0
            item["target_count"] = valid
            item["valid_targets"] = valid
            item["finite_target_count"] = finite
            item["gradient_steps"] = gradients
            item["nonzero_gradient_steps"] = nonzero
            item["finite_loss_steps"] = finite_loss_steps
            if valid <= 0:
                reason = "zero_valid_targets"
            elif finite_masked < masked:
                reason = "nonfinite_target_present"
            elif finite_loss_steps <= 0:
                reason = "nonfinite_loss"
            elif nonzero <= 0:
                reason = "zero_nonzero_gradient_steps"
            else:
                reason = None
            item["pass"] = reason is None
            item["status"] = "pass" if reason is None else "block"
            item["block_reason"] = reason
            rows.append(item)
        return rows

    def target_gradient_coverage_passes(self) -> bool:
        """Whether every enabled target head has usable targets and gradients."""
        rows = self.target_gradient_coverage()
        return bool(rows) and all(bool(row.get("pass")) for row in rows)

    def _write_target_gradient_coverage(self, path: str | None) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in self.target_gradient_coverage():
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        blocked = [
            f"{row['head']}:{row['horizon']}({row['block_reason']})"
            for row in self.target_gradient_coverage()
            if row.get("status") != "pass"
        ]
        suffix = f" blocked={','.join(blocked)}" if blocked else ""
        print(f"[WM] Target/gradient coverage artifact: {path}{suffix}")

    def _write_target_gradient_coverage_block_marker(
        self,
        checkpoint_path: str | None,
        coverage_path: str | None,
    ) -> str | None:
        """Mark a saved checkpoint as non-promotable after a failed gate."""
        marker_path = (
            f"{checkpoint_path}.blocked.json"
            if checkpoint_path
            else f"{coverage_path}.blocked.json"
            if coverage_path
            else None
        )
        if marker_path is None:
            return None
        os.makedirs(os.path.dirname(marker_path) or ".", exist_ok=True)
        blocked_rows = [
            {
                "head": row.get("head"),
                "horizon": row.get("horizon"),
                "output_index": row.get("output_index"),
                "block_reason": row.get("block_reason"),
            }
            for row in self.target_gradient_coverage()
            if row.get("status") != "pass"
        ]
        with open(marker_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "blocked",
                    "promotable": False,
                    "reason": "target_gradient_coverage",
                    "checkpoint": checkpoint_path,
                    "coverage_artifact": coverage_path,
                    "blocked_rows": blocked_rows,
                },
                handle,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        return marker_path

    def _enforce_target_gradient_coverage_gate(
        self,
        checkpoint_path: str | None,
        coverage_path: str | None,
    ) -> None:
        if not self.require_target_gradient_coverage:
            return
        if self.target_gradient_coverage_passes():
            return
        marker_path = self._write_target_gradient_coverage_block_marker(
            checkpoint_path,
            coverage_path,
        )
        blocked = [
            f"{row['head']}:{row['horizon']}({row['block_reason']})"
            for row in self.target_gradient_coverage()
            if row.get("status") != "pass"
        ]
        marker_text = f"; marker={marker_path}" if marker_path else ""
        raise TargetGradientCoverageError(
            "target/gradient coverage gate blocked promotion: "
            f"{','.join(blocked) or 'no enabled head passed'}{marker_text}"
        )

    @staticmethod
    def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        """Clone a module state to CPU so a best snapshot cannot be mutated."""
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    def _capture_model_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Capture one coherent snapshot of the ensemble and active heads."""
        state = {"ensemble": self._clone_state_dict(self.ensemble)}
        state.update(
            {
                name: self._clone_state_dict(head)
                for name, head in self._active_auxiliary_heads().items()
            }
        )
        return state

    def _restore_model_state(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore an ensemble/head snapshot captured by ``_capture_model_state``."""
        self.ensemble.load_state_dict(state["ensemble"])
        for name, head in self._active_auxiliary_heads().items():
            if name in state:
                head.load_state_dict(state[name])
        self.ensemble.to(self.device)
        for head in self._active_auxiliary_heads().values():
            head.to(self.device)

    def _loader_options(self, num_workers: int) -> dict:
        workers = max(0, int(num_workers))
        return {
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": workers > 0,
        }

    def _compute_net_returns(
        self,
        actions: torch.Tensor,
        raw_returns: torch.Tensor,
    ) -> torch.Tensor:
        """行動インデックスと生リターンからネットリターン（コスト控除後）を計算する.

        net_return[t] = position[t] * raw_returns[t]
                        - cost_rate * |position[t] - position[t-1]|

        初期ポジション = 0.0（フラット）。

        Args:
            actions: (B, T, 1) position path
            raw_returns: (B, T) 生の対数リターン

        Returns:
            net_returns: (B, T) コスト控除後リターン
        """
        positions = actions.squeeze(-1)                             # (B, T)

        # 前ステップポジション（初期 = 0.0 = フラット）
        prev_positions = torch.cat([
            torch.zeros_like(positions[:, :1]),
            positions[:, :-1],
        ], dim=1)                                                    # (B, T)

        delta_pos = (positions - prev_positions).abs()
        costs = self.cost_rate * delta_pos                           # (B, T)
        net_returns = positions * raw_returns - costs               # (B, T)
        if self.reward_mode == "excess_bh":
            benchmark_returns = self.benchmark_position * raw_returns
            return net_returns - benchmark_returns
        return net_returns

    def _future_return_targets(
        self,
        raw_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build multi-horizon return targets and valid masks."""
        B, T = raw_returns.shape
        offset0 = 0 if self.return_include_current else 1
        targets = []
        masks = []
        for horizon in self.return_horizons:
            horizon = max(1, int(horizon))
            target = torch.zeros_like(raw_returns)
            for k in range(offset0, offset0 + horizon):
                if k < T:
                    target[:, : T - k] += raw_returns[:, k:]
            valid_len = T - (offset0 + horizon - 1)
            mask = torch.zeros((B, T), dtype=torch.bool, device=raw_returns.device)
            if valid_len > 0:
                mask[:, :valid_len] = True
            targets.append(target * self.return_target_scale)
            masks.append(mask)
        return torch.stack(targets, dim=-1), torch.stack(masks, dim=-1)

    def _future_risk_targets(
        self,
        raw_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build future realized volatility, drawdown, and crash-risk targets."""
        B, T = raw_returns.shape
        vol_targets = []
        dd_targets = []
        crash_targets = []
        dd_excess_targets = []
        masks = []
        for horizon in self.risk_horizons:
            horizon = max(1, int(horizon))
            cum = torch.zeros_like(raw_returns)
            min_cum = torch.zeros_like(raw_returns)
            sq_sum = torch.zeros_like(raw_returns)
            for k in range(1, horizon + 1):
                if k < T:
                    shifted = torch.zeros_like(raw_returns)
                    shifted[:, : T - k] = raw_returns[:, k:]
                    cum = cum + shifted
                    min_cum = torch.minimum(min_cum, cum)
                    sq_sum = sq_sum + shifted.square()
            valid_len = T - horizon
            mask = torch.zeros((B, T), dtype=torch.bool, device=raw_returns.device)
            if valid_len > 0:
                mask[:, :valid_len] = True
            drawdown = -min_cum
            vol_targets.append(torch.sqrt(sq_sum / float(horizon) + 1e-12) * self.risk_target_scale)
            dd_targets.append(drawdown * self.risk_target_scale)
            crash_targets.append((drawdown >= float(self.crash_threshold)).to(raw_returns.dtype))
            dd_excess_targets.append(
                torch.clamp(drawdown - float(self.drawdown_excess_threshold), min=0.0)
                * self.risk_target_scale
            )
            masks.append(mask)
        return (
            torch.stack(vol_targets, dim=-1),
            torch.stack(dd_targets, dim=-1),
            torch.stack(crash_targets, dim=-1),
            torch.stack(dd_excess_targets, dim=-1),
            torch.stack(masks, dim=-1),
        )

    def _future_control_targets(
        self,
        raw_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build AC-control-oriented targets from forward return paths.

        `overweight_advantage` approximates the incremental risk-adjusted value
        of holding a small overweight versus benchmark exposure. `recovery`
        is positive when forward return dominates downside over the horizon.
        Both targets are non-leaky labels used only during WM training.
        """
        B, T = raw_returns.shape
        ow_targets = []
        recovery_targets = []
        masks = []
        for horizon in self.risk_horizons:
            horizon = max(1, int(horizon))
            cum = torch.zeros_like(raw_returns)
            min_cum = torch.zeros_like(raw_returns)
            sq_sum = torch.zeros_like(raw_returns)
            for k in range(1, horizon + 1):
                if k < T:
                    shifted = torch.zeros_like(raw_returns)
                    shifted[:, : T - k] = raw_returns[:, k:]
                    cum = cum + shifted
                    min_cum = torch.minimum(min_cum, cum)
                    sq_sum = sq_sum + shifted.square()
            downside = -min_cum
            valid_len = T - horizon
            mask = torch.zeros((B, T), dtype=torch.bool, device=raw_returns.device)
            if valid_len > 0:
                mask[:, :valid_len] = True

            delta = float(self.overweight_delta)
            one_way_cost = float(self.cost_rate) * abs(delta)
            ow_value = delta * cum - delta * float(self.overweight_drawdown_penalty) * downside - one_way_cost
            vol = torch.sqrt(sq_sum / float(horizon) + 1e-12)
            recovery_value = (cum - float(self.recovery_drawdown_penalty) * downside) / (vol * horizon**0.5 + 1e-6)
            recovery_value = recovery_value.clamp(-5.0, 5.0) / 5.0

            ow_targets.append(ow_value * self.control_target_scale)
            recovery_targets.append(recovery_value)
            masks.append(mask)
        return (
            torch.stack(ow_targets, dim=-1),
            torch.stack(recovery_targets, dim=-1),
            torch.stack(masks, dim=-1),
        )

    def _future_position_utility_targets(
        self,
        raw_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build candidate absolute-position utility targets.

        The target is benchmark-relative utility over a forward window:
        overlay return minus one-way rebalance cost, drawdown worsening, and
        volatility exposure. It is used only for supervised WM auxiliary
        learning; train/val/test policy selection still uses past/current
        features and validation thresholds.
        """
        B, T = raw_returns.shape
        horizon = max(1, int(self.position_utility_horizon))
        cum = torch.zeros_like(raw_returns)
        sq_sum = torch.zeros_like(raw_returns)
        utility_parts = []
        path_cums: dict[float, torch.Tensor] = {}
        positions = sorted(set([float(self.benchmark_position), *self.position_utility_positions]))
        for pos in positions:
            path_cums[pos] = torch.zeros((B, T, horizon + 1), dtype=raw_returns.dtype, device=raw_returns.device)

        running = {pos: torch.zeros_like(raw_returns) for pos in positions}
        for k in range(1, horizon + 1):
            if k < T:
                shifted = torch.zeros_like(raw_returns)
                shifted[:, : T - k] = raw_returns[:, k:]
                cum = cum + shifted
                sq_sum = sq_sum + shifted.square()
                for pos in positions:
                    running[pos] = running[pos] + shifted * float(pos)
                    path_cums[pos][:, :, k] = running[pos]

        bench_path = path_cums[float(self.benchmark_position)]
        bench_peak = torch.cummax(bench_path, dim=-1).values
        bench_dd = (bench_peak - bench_path).amax(dim=-1)
        future_vol = torch.sqrt(sq_sum / float(horizon) + 1e-12)

        for pos in self.position_utility_positions:
            pos = float(pos)
            path = path_cums[pos]
            peak = torch.cummax(path, dim=-1).values
            dd = (peak - path).amax(dim=-1)
            dd_worsen = F.relu(dd - bench_dd)
            dd_improve = F.relu(bench_dd - dd)
            overlay = pos - float(self.benchmark_position)
            trade_cost = abs(overlay) * float(self.cost_rate)
            utility = (
                overlay * cum
                - trade_cost
                - float(self.position_utility_dd_penalty) * dd_worsen
                + float(self.position_utility_dd_improve_reward) * dd_improve
                - float(self.position_utility_vol_penalty) * abs(overlay) * future_vol
            )
            utility_parts.append(utility * self.position_utility_target_scale)

        valid_len = T - horizon
        mask = torch.zeros((B, T), dtype=torch.bool, device=raw_returns.device)
        if valid_len > 0:
            mask[:, :valid_len] = True
        return torch.stack(utility_parts, dim=-1), mask.unsqueeze(-1).expand(-1, -1, len(utility_parts))

    @staticmethod
    def _masked_smooth_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not torch.any(valid):
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred[valid], target[valid])

    def _position_utility_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not torch.any(valid):
            return pred.sum() * 0.0

        per_elem = F.smooth_l1_loss(pred, target, reduction="none")
        weights = torch.ones_like(per_elem)
        if self.position_utility_positive_weight > 0.0:
            weights = weights + float(self.position_utility_positive_weight) * (
                target > float(self.position_utility_rank_margin)
            ).to(weights.dtype)

        bench_idx = int(
            min(
                range(len(self.position_utility_positions)),
                key=lambda i: abs(float(self.position_utility_positions[i]) - float(self.benchmark_position)),
            )
        )
        row_valid = torch.all(valid, dim=-1)
        target_best = torch.argmax(target, dim=-1)
        target_best_value = torch.gather(target, -1, target_best.unsqueeze(-1)).squeeze(-1)
        bench_value = target[..., bench_idx]
        best_improvement = target_best_value - bench_value
        row_weight = torch.ones_like(best_improvement)
        if self.position_utility_nonbench_weight > 0.0:
            nonbench = target_best != bench_idx
            actionable = best_improvement > float(self.position_utility_rank_margin)
            row_weight = row_weight + float(self.position_utility_nonbench_weight) * (nonbench & actionable).to(
                row_weight.dtype
            )
        weights = weights * row_weight.unsqueeze(-1)

        denom = torch.clamp(weights[valid].sum(), min=1e-6)
        regression_loss = (per_elem[valid] * weights[valid]).sum() / denom
        if self.position_utility_rank_scale <= 0.0 or not torch.any(row_valid):
            return regression_loss

        logits = pred[row_valid]
        labels = target_best[row_valid]
        ce = F.cross_entropy(logits, labels, reduction="none")
        ce_weight = row_weight[row_valid]
        rank_loss = (ce * ce_weight).sum() / torch.clamp(ce_weight.sum(), min=1e-6)
        return regression_loss + float(self.position_utility_rank_scale) * rank_loss

    @staticmethod
    def _masked_bce_with_logits(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pos_weight: float = 1.0,
    ) -> torch.Tensor:
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not torch.any(valid):
            return pred.sum() * 0.0
        weight = None
        if float(pos_weight) > 0.0 and abs(float(pos_weight) - 1.0) > 1e-8:
            weight = torch.where(
                target[valid] > 0.5,
                torch.full_like(target[valid], float(pos_weight)),
                torch.ones_like(target[valid]),
            )
        return F.binary_cross_entropy_with_logits(pred[valid], target[valid], weight=weight)

    def train_on_dataset(
        self,
        dataset: SequenceDataset,
        val_dataset: Optional[SequenceDataset] = None,
        max_steps: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        patience: int = 10,
        coverage_context: Mapping[str, object] | None = None,
    ) -> list[dict]:
        """データセット上で世界モデルを学習する.

        val_dataset がある場合、log_interval ごとに val loss を計算し、
        best model を保持する。patience 回連続で改善しなければ early stop する。

        Args:
            dataset: 学習用 SequenceDataset
            val_dataset: 検証用 SequenceDataset（省略可、あれば early stopping に使用）
            max_steps: 最大ステップ数（None の場合は self.max_steps）
            checkpoint_path: チェックポイント保存先（省略可）
            patience: early stopping の忍耐回数（val 評価回数単位）
            coverage_context: 任意の run/fold/phase provenance fields for the
                machine-readable target coverage artifact.

        Returns:
            各ステップのロスログリスト
        """
        max_steps = max_steps or self.max_steps

        sequence_length = int(getattr(dataset, "seq_len", 0)) if dataset is not None else None
        self._start_target_gradient_coverage(sequence_length, coverage_context)

        if len(dataset) == 0:
            print("[WM] WARNING: dataset is empty, skipping training")
            coverage_path = (
                os.path.join(os.path.dirname(checkpoint_path), "target_gradient_coverage.jsonl")
                if checkpoint_path
                else None
            )
            self._write_target_gradient_coverage(coverage_path)
            self._enforce_target_gradient_coverage_gate(None, coverage_path)
            return []

        # dataset が batch_size 未満の場合 drop_last=True で loader が空になり無限ループする
        drop_last = len(dataset) >= self.batch_size
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            **self._loader_options(self.num_workers),
        )
        self.ensemble.train()
        step = 0
        logs = []

        # Early stopping 用の状態
        best_val_loss = float("inf")
        best_state_dict = None
        no_improve_count = 0

        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break

                self._coverage_step_executed.clear()
                self._coverage_step_finite.clear()

                obs = batch["obs"].to(self.device)            # (B, T, obs_dim)
                obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

                # actionless/deployable mode intentionally ignores any action
                # column accidentally attached to the dataset.  This keeps
                # training on the same benchmark-equivalent context as
                # encode_sequence(actions=None).
                batch_actions = batch.get("actions") if self.use_dataset_actions else None
                if batch_actions is not None:
                    actions = batch_actions.to(self.device)  # (B, T, 1) or (B, T)
                else:
                    actions = torch.full(
                        (*obs.shape[:2], 1),
                        fill_value=self.benchmark_position if self.reward_mode == "excess_bh" else 0.0,
                        dtype=torch.float32,
                        device=self.device,
                    )
                if actions.ndim == 2 and not torch.is_floating_point(actions):
                    actions = self.action_values[actions].unsqueeze(-1)
                elif actions.ndim == 2:
                    actions = actions.unsqueeze(-1)

                # SPEC 準拠: WM の reward head は net_return（コスト控除後）を学習する
                # raw return がある場合のみ net_return を計算、なければゼロ埋め
                raw_returns = batch.get("returns")
                if raw_returns is not None:
                    raw_returns = raw_returns.to(self.device)   # (B, T)
                    rewards = self._compute_net_returns(actions, raw_returns)
                else:
                    rewards = torch.zeros(obs.shape[:2], device=self.device)

                # dones はゼロ埋め（継続的トレーディングでは done=False が多い）
                dones = torch.zeros_like(rewards)

                # 損失計算
                loss_dict = self.ensemble.compute_losses(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    free_bits=self.free_bits,
                    dyn_scale=self.dyn_scale,
                    rep_scale=self.rep_scale,
                    recon_scale=self.recon_scale,
                    reward_scale=self.reward_scale,
                    done_scale=self.done_scale,
                )

                total_loss = loss_dict["loss"]

                # --- Auxiliary losses ---
                idm_loss_val = 0.0
                return_loss_val = 0.0
                vol_loss_val = 0.0
                drawdown_loss_val = 0.0
                crash_loss_val = 0.0
                drawdown_excess_loss_val = 0.0
                position_utility_loss_val = 0.0
                overweight_advantage_loss_val = 0.0
                recovery_loss_val = 0.0
                regime_loss_val = 0.0

                has_predictive_head = (
                    self.return_head is not None
                    or self.vol_head is not None
                    or self.drawdown_head is not None
                    or self.crash_head is not None
                    or self.drawdown_excess_head is not None
                    or self.position_utility_head is not None
                    or self.overweight_advantage_head is not None
                    or self.recovery_head is not None
                    or self.regime_head is not None
                )
                if self.idm_head is not None or has_predictive_head:
                    z, _ = self.ensemble.encode(obs)  # (B, T, z_dim)

                    if self.idm_head is not None and batch_actions is not None and not torch.is_floating_point(batch_actions):
                        self._accumulate_target_coverage(
                            "idm",
                            batch_actions.to(self.device)[:, :-1],
                            torch.ones_like(batch_actions[:, :-1], dtype=torch.bool, device=self.device),
                            horizons=[1],
                        )
                        z_t = z[:, :-1, :]   # (B, T-1, z_dim)
                        z_t1 = z[:, 1:, :]   # (B, T-1, z_dim)
                        idm_logits = self.idm_head(z_t, z_t1)  # (B, T-1, n_actions)
                        oracle_acts = batch_actions.to(self.device)[:, :-1]  # (B, T-1)
                        B_, T_, A_ = idm_logits.shape
                        idm_loss = F.cross_entropy(
                            idm_logits.reshape(B_ * T_, A_),
                            oracle_acts.reshape(B_ * T_),
                        )
                        total_loss = total_loss + self.idm_scale * idm_loss
                        idm_loss_val = idm_loss.item()
                        self._mark_head_loss_coverage("idm", idm_loss, [1])

                    h = None
                    if has_predictive_head:
                        out_h = self.ensemble.forward(z, actions)
                        h = out_h["h"]  # (B, T, d_model)
                    if self.return_head is not None and raw_returns is not None and h is not None:
                        target, mask = self._future_return_targets(raw_returns)
                        self._accumulate_target_coverage("return", target, mask, self.return_horizons)
                        pred = self.return_head(z, h, obs)
                        return_loss = self._masked_smooth_l1(pred, target, mask)
                        total_loss = total_loss + self.return_scale * return_loss
                        return_loss_val = return_loss.item()
                        self._mark_head_loss_coverage("return", return_loss, self.return_horizons)

                    if (
                        (
                            self.vol_head is not None
                            or self.drawdown_head is not None
                            or self.crash_head is not None
                            or self.drawdown_excess_head is not None
                        )
                        and raw_returns is not None
                        and h is not None
                    ):
                        vol_target, dd_target, crash_target, dd_excess_target, risk_mask = (
                            self._future_risk_targets(raw_returns)
                        )
                        for head_name, target_tensor in (
                            ("vol", vol_target),
                            ("drawdown", dd_target),
                            ("crash", crash_target),
                            ("drawdown_excess", dd_excess_target),
                        ):
                            if getattr(self, f"{head_name}_head", None) is not None:
                                self._accumulate_target_coverage(
                                    head_name,
                                    target_tensor,
                                    risk_mask,
                                    self.risk_horizons,
                                )
                        if self.vol_head is not None:
                            vol_pred = self.vol_head(z, h, obs)
                            vol_loss = self._masked_smooth_l1(vol_pred, vol_target, risk_mask)
                            total_loss = total_loss + self.vol_scale * vol_loss
                            vol_loss_val = vol_loss.item()
                            self._mark_head_loss_coverage("vol", vol_loss, self.risk_horizons)
                        if self.drawdown_head is not None:
                            dd_pred = self.drawdown_head(z, h, obs)
                            drawdown_loss = self._masked_smooth_l1(dd_pred, dd_target, risk_mask)
                            total_loss = total_loss + self.drawdown_scale * drawdown_loss
                            drawdown_loss_val = drawdown_loss.item()
                            self._mark_head_loss_coverage("drawdown", drawdown_loss, self.risk_horizons)
                        if self.crash_head is not None:
                            crash_pred = self.crash_head(z, h, obs)
                            crash_loss = self._masked_bce_with_logits(
                                crash_pred,
                                crash_target,
                                risk_mask,
                                pos_weight=self.crash_pos_weight,
                            )
                            total_loss = total_loss + self.crash_scale * crash_loss
                            crash_loss_val = crash_loss.item()
                            self._mark_head_loss_coverage("crash", crash_loss, self.risk_horizons)
                        if self.drawdown_excess_head is not None:
                            dd_excess_pred = self.drawdown_excess_head(z, h, obs)
                            dd_excess_loss = self._masked_smooth_l1(
                                dd_excess_pred,
                                dd_excess_target,
                                risk_mask,
                            )
                            total_loss = total_loss + self.drawdown_excess_scale * dd_excess_loss
                            drawdown_excess_loss_val = dd_excess_loss.item()
                            self._mark_head_loss_coverage(
                                "drawdown_excess",
                                dd_excess_loss,
                                self.risk_horizons,
                            )

                    if self.position_utility_head is not None and raw_returns is not None and h is not None:
                        utility_target, utility_mask = self._future_position_utility_targets(raw_returns)
                        self._accumulate_target_coverage(
                            "position_utility",
                            utility_target,
                            utility_mask,
                            [self.position_utility_horizon] * len(self.position_utility_positions),
                        )
                        utility_pred = self.position_utility_head(z, h, obs)
                        utility_loss = self._position_utility_loss(utility_pred, utility_target, utility_mask)
                        total_loss = total_loss + self.position_utility_scale * utility_loss
                        position_utility_loss_val = utility_loss.item()
                        self._mark_head_loss_coverage(
                            "position_utility",
                            utility_loss,
                            [self.position_utility_horizon] * len(self.position_utility_positions),
                        )

                    if (
                        (self.overweight_advantage_head is not None or self.recovery_head is not None)
                        and raw_returns is not None
                        and h is not None
                    ):
                        ow_target, recovery_target, control_mask = self._future_control_targets(raw_returns)
                        if self.overweight_advantage_head is not None:
                            self._accumulate_target_coverage(
                                "overweight_advantage",
                                ow_target,
                                control_mask,
                                self.risk_horizons,
                            )
                        if self.recovery_head is not None:
                            self._accumulate_target_coverage(
                                "recovery",
                                recovery_target,
                                control_mask,
                                self.risk_horizons,
                            )
                        if self.overweight_advantage_head is not None:
                            ow_pred = self.overweight_advantage_head(z, h, obs)
                            ow_loss = self._masked_smooth_l1(ow_pred, ow_target, control_mask)
                            total_loss = total_loss + self.overweight_advantage_scale * ow_loss
                            overweight_advantage_loss_val = ow_loss.item()
                            self._mark_head_loss_coverage(
                                "overweight_advantage",
                                ow_loss,
                                self.risk_horizons,
                            )
                        if self.recovery_head is not None:
                            recovery_pred = self.recovery_head(z, h, obs)
                            recovery_loss = self._masked_smooth_l1(recovery_pred, recovery_target, control_mask)
                            total_loss = total_loss + self.recovery_scale * recovery_loss
                            recovery_loss_val = recovery_loss.item()
                            self._mark_head_loss_coverage(
                                "recovery",
                                recovery_loss,
                                self.risk_horizons,
                            )

                    regime_probs = batch.get("regime")
                    if self.regime_head is not None and regime_probs is not None and h is not None:
                        regime_probs = regime_probs.to(self.device)
                        self._accumulate_target_coverage(
                            "regime",
                            regime_probs,
                            torch.ones_like(regime_probs, dtype=torch.bool, device=self.device),
                            horizons=["current"],
                        )
                        regime_logits = self.regime_head(z, h)
                        log_probs = F.log_softmax(regime_logits, dim=-1)
                        regime_loss = -(regime_probs * log_probs).sum(dim=-1).mean()
                        total_loss = total_loss + self.regime_aux_scale * regime_loss
                        regime_loss_val = regime_loss.item()
                        self._mark_head_loss_coverage("regime", regime_loss, ["current"])

                self.optimizer.zero_grad()
                total_loss.backward()
                self._record_gradient_coverage(total_loss)
                nn.utils.clip_grad_norm_(self._all_params, self.grad_clip)
                self.optimizer.step()

                step += 1
                self.global_step += 1

                log = {
                    "step": self.global_step,
                    "loss": total_loss.item(),
                    "base_loss": loss_dict["base_loss"].item(),
                    "disagreement": loss_dict["disagreement"].item(),
                    "idm_loss": idm_loss_val,
                    "return_loss": return_loss_val,
                    "vol_loss": vol_loss_val,
                    "drawdown_loss": drawdown_loss_val,
                    "crash_loss": crash_loss_val,
                    "drawdown_excess_loss": drawdown_excess_loss_val,
                    "position_utility_loss": position_utility_loss_val,
                    "overweight_advantage_loss": overweight_advantage_loss_val,
                    "recovery_loss": recovery_loss_val,
                    "regime_loss": regime_loss_val,
                }
                logs.append(log)
                self.loss_history.append(log)

                if step % self.log_interval == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    aux_str = ""
                    if self.idm_head is not None:
                        aux_str += f" | IDM: {log['idm_loss']:.4f}"
                    if self.return_head is not None:
                        aux_str += f" | Ret: {log['return_loss']:.4f}"
                    if self.vol_head is not None:
                        aux_str += f" | Vol: {log['vol_loss']:.4f}"
                    if self.drawdown_head is not None:
                        aux_str += f" | DD: {log['drawdown_loss']:.4f}"
                    if self.crash_head is not None:
                        aux_str += f" | Crash: {log['crash_loss']:.4f}"
                    if self.drawdown_excess_head is not None:
                        aux_str += f" | DDEx: {log['drawdown_excess_loss']:.4f}"
                    if self.position_utility_head is not None:
                        aux_str += f" | PosU: {log['position_utility_loss']:.4f}"
                    if self.overweight_advantage_head is not None:
                        aux_str += f" | OWA: {log['overweight_advantage_loss']:.4f}"
                    if self.recovery_head is not None:
                        aux_str += f" | Rec: {log['recovery_loss']:.4f}"
                    if self.regime_head is not None:
                        aux_str += f" | Reg: {log['regime_loss']:.4f}"
                    print(
                        f"[{ts}] [WM] Step {self.global_step}/{max_steps} | "
                        f"Loss: {log['loss']:.4f} | "
                        f"BaseLoss: {log['base_loss']:.4f} | "
                        f"Disagree: {log['disagreement']:.4f}"
                        + aux_str
                    )

                    # Validation loss + early stopping
                    if val_dataset is not None:
                        if self.val_max_batches is None:
                            val_loss = self._eval_loss(val_dataset)
                        else:
                            val_loss = self._eval_loss(val_dataset, n_batches=self.val_max_batches)
                        print(f"       Val Loss: {val_loss:.4f}", end="")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state_dict = self._capture_model_state()
                            no_improve_count = 0
                            print(" ★ best")
                        else:
                            no_improve_count += 1
                            print(f" (no improve {no_improve_count}/{patience})")

                        if no_improve_count >= patience:
                            print(f"[WM] Early stopping at step {self.global_step} "
                                  f"(best val loss: {best_val_loss:.4f})")
                            step = max_steps  # ループ脱出
                            break

            # エポック終了後にチェックポイント保存
            if checkpoint_path is not None:
                self.save(checkpoint_path)

        # Best model を復元（val_dataset があり、改善があった場合）
        if best_state_dict is not None:
            self._restore_model_state(best_state_dict)
            print(f"[WM] Restored best model (val loss: {best_val_loss:.4f})")
            if checkpoint_path is not None:
                self.save(checkpoint_path)

        coverage_path = (
            os.path.join(os.path.dirname(checkpoint_path), "target_gradient_coverage.jsonl")
            if checkpoint_path
            else None
        )
        self._write_target_gradient_coverage(coverage_path)
        self._enforce_target_gradient_coverage_gate(checkpoint_path, coverage_path)

        return logs

    @torch.no_grad()
    def _eval_loss(self, dataset: SequenceDataset, n_batches: Optional[int] = None) -> float:
        """Validation loss を計算する（既定では全ミニバッチの平均）.

        ``n_batches`` が明示された場合だけ先頭からその数に制限する。
        学習時と同じ net_return（コスト控除後）を reward として使用する。
        """
        self.ensemble.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_options(min(self.num_workers, 2)),
        )
        total = 0.0
        count = 0
        for i, batch in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break
            obs = batch["obs"].to(self.device)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)  # training と同一処理
            default_action = self.benchmark_position if self.reward_mode == "excess_bh" else 0.0
            batch_actions = batch.get("actions") if self.use_dataset_actions else None
            actions = (
                batch_actions
                if batch_actions is not None
                else torch.full((*obs.shape[:2], 1), default_action, dtype=torch.float32)
            )
            actions = actions.to(self.device)
            if actions.ndim == 2 and not torch.is_floating_point(actions):
                actions = self.action_values[actions].unsqueeze(-1)
            elif actions.ndim == 2:
                actions = actions.unsqueeze(-1)

            # 学習時と同じ: raw_returns → net_returns を reward として使用
            raw_returns = batch.get("returns")
            if raw_returns is not None:
                raw_returns = raw_returns.to(self.device)
                raw_returns = torch.nan_to_num(raw_returns, nan=0.0, posinf=0.0, neginf=0.0)
                rewards = self._compute_net_returns(actions, raw_returns)
            else:
                rewards = torch.zeros(obs.shape[:2], device=self.device)

            dones = torch.zeros_like(rewards)

            loss_dict = self.ensemble.compute_losses(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                free_bits=self.free_bits,
                dyn_scale=self.dyn_scale,
                rep_scale=self.rep_scale,
                recon_scale=self.recon_scale,
                reward_scale=self.reward_scale,
                done_scale=self.done_scale,
            )
            total_loss = loss_dict["loss"]
            has_predictive_head = (
                self.return_head is not None
                or self.vol_head is not None
                or self.drawdown_head is not None
                or self.crash_head is not None
                or self.drawdown_excess_head is not None
                or self.position_utility_head is not None
                or self.overweight_advantage_head is not None
                or self.recovery_head is not None
                or self.regime_head is not None
            )
            has_idm_target = (
                self.idm_head is not None
                and batch_actions is not None
                and not torch.is_floating_point(batch_actions)
            )
            if has_idm_target or has_predictive_head:
                z, _ = self.ensemble.encode(obs)
                if has_idm_target:
                    z_t = z[:, :-1, :]
                    z_t1 = z[:, 1:, :]
                    idm_logits = self.idm_head(z_t, z_t1)
                    oracle_acts = batch_actions.to(self.device)[:, :-1]
                    B_, T_, A_ = idm_logits.shape
                    idm_loss = F.cross_entropy(
                        idm_logits.reshape(B_ * T_, A_),
                        oracle_acts.reshape(B_ * T_),
                    )
                    total_loss = total_loss + self.idm_scale * idm_loss
                h = None
                if has_predictive_head:
                    h = self.ensemble.forward(z, actions)["h"]
                if self.return_head is not None and raw_returns is not None:
                    target, mask = self._future_return_targets(raw_returns)
                    pred = self.return_head(z, h, obs)
                    total_loss = total_loss + self.return_scale * self._masked_smooth_l1(pred, target, mask)
                if (
                    (
                        self.vol_head is not None
                        or self.drawdown_head is not None
                        or self.crash_head is not None
                        or self.drawdown_excess_head is not None
                    )
                    and raw_returns is not None
                ):
                    vol_target, dd_target, crash_target, dd_excess_target, risk_mask = (
                        self._future_risk_targets(raw_returns)
                    )
                    if self.vol_head is not None:
                        vol_pred = self.vol_head(z, h, obs)
                        total_loss = total_loss + self.vol_scale * self._masked_smooth_l1(
                            vol_pred,
                            vol_target,
                            risk_mask,
                        )
                    if self.drawdown_head is not None:
                        dd_pred = self.drawdown_head(z, h, obs)
                        total_loss = total_loss + self.drawdown_scale * self._masked_smooth_l1(
                            dd_pred,
                            dd_target,
                            risk_mask,
                        )
                    if self.crash_head is not None:
                        crash_pred = self.crash_head(z, h, obs)
                        total_loss = total_loss + self.crash_scale * self._masked_bce_with_logits(
                            crash_pred,
                            crash_target,
                            risk_mask,
                            pos_weight=self.crash_pos_weight,
                        )
                    if self.drawdown_excess_head is not None:
                        dd_excess_pred = self.drawdown_excess_head(z, h, obs)
                        total_loss = total_loss + self.drawdown_excess_scale * self._masked_smooth_l1(
                            dd_excess_pred,
                            dd_excess_target,
                            risk_mask,
                        )
                if self.position_utility_head is not None and raw_returns is not None:
                    utility_target, utility_mask = self._future_position_utility_targets(raw_returns)
                    utility_pred = self.position_utility_head(z, h, obs)
                    total_loss = total_loss + self.position_utility_scale * self._position_utility_loss(
                        utility_pred,
                        utility_target,
                        utility_mask,
                    )
                if (
                    (self.overweight_advantage_head is not None or self.recovery_head is not None)
                    and raw_returns is not None
                ):
                    ow_target, recovery_target, control_mask = self._future_control_targets(raw_returns)
                    if self.overweight_advantage_head is not None:
                        ow_pred = self.overweight_advantage_head(z, h, obs)
                        total_loss = total_loss + self.overweight_advantage_scale * self._masked_smooth_l1(
                            ow_pred,
                            ow_target,
                            control_mask,
                        )
                    if self.recovery_head is not None:
                        recovery_pred = self.recovery_head(z, h, obs)
                        total_loss = total_loss + self.recovery_scale * self._masked_smooth_l1(
                            recovery_pred,
                            recovery_target,
                            control_mask,
                        )
            if self.regime_head is not None and "regime" in batch and h is not None:
                regime_probs = batch["regime"].to(self.device)
                regime_logits = self.regime_head(z, h)
                log_probs = F.log_softmax(regime_logits, dim=-1)
                regime_loss = -(regime_probs * log_probs).sum(dim=-1).mean()
                total_loss = total_loss + self.regime_aux_scale * regime_loss
            total += total_loss.item()
            count += 1

        self.ensemble.train()
        return total / max(count, 1)

    @torch.no_grad()
    def encode_sequence(
        self,
        features: np.ndarray,
        actions: Optional[np.ndarray] = None,
        seq_len: int = 64,
    ) -> dict[str, np.ndarray]:
        """特徴量列をエンコードして潜在・hidden を返す.

        Actor-Critic の入力として使用する。

        Args:
            features: (T, obs_dim)
            actions: (T,) 行動インデックス
            seq_len: バッチ長

        Returns:
            {z: (T, z_dim), h: (T, d_model)}
        """
        self.ensemble.eval()
        T, obs_dim = features.shape
        z_dim = self.ensemble.get_z_dim()
        d_model = self.ensemble.get_d_model()

        if T == 0:
            return {"z": np.zeros((0, z_dim)), "h": np.zeros((0, d_model))}

        z_arr = np.zeros((T, z_dim), dtype=np.float32)
        h_arr = np.zeros((T, d_model), dtype=np.float32)
        covered = 0

        # 各チャンクの直前 seq_len ステップをウォームアップ文脈として追加する。
        # これにより Transformer がブロック境界でコンテキストをリセットする問題を防ぐ。
        # 合計シーケンス長は最大 2*seq_len <= max_seq_len（configs で保証）。
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            # 最後のチャンクが短い場合、末尾揃えにする
            if end - start < seq_len and T >= seq_len:
                start = T - seq_len
                end = T

            # ウォームアップ文脈: start の直前 seq_len ステップ
            ctx_start = max(0, start - seq_len)

            obs_t = torch.tensor(
                features[ctx_start:end], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)

            if actions is not None and self.use_dataset_actions:
                act_t = torch.tensor(
                    actions[ctx_start:end], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            else:
                act_t = torch.full(
                    (1, end - ctx_start, 1),
                    fill_value=self.benchmark_position if self.reward_mode == "excess_bh" else 0.0,
                    dtype=torch.float32,
                    device=self.device,
                )
            if act_t.ndim == 2:
                act_t = act_t.unsqueeze(-1)

            z, _ = self.ensemble.encode(obs_t)
            out = self.ensemble.forward(z, act_t)

            # ウォームアップ部分を除いた本体のみを書き込む
            prefix_len = start - ctx_start
            z_np = z.squeeze(0)[prefix_len:].cpu().numpy()
            h_np = out["h"].squeeze(0)[prefix_len:].cpu().numpy()

            write_start = max(start, covered)
            offset = write_start - start
            z_arr[write_start:end] = z_np[offset:]
            h_arr[write_start:end] = h_np[offset:]
            covered = end

            if end == T:
                break

        return {"z": z_arr, "h": h_arr}

    def save(self, path: str) -> None:
        """チェックポイントを保存する."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        ckpt = {
            "ensemble": self.ensemble.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "checkpoint_metadata": self.checkpoint_metadata,
        }
        if self.idm_head is not None:
            ckpt["idm_head"] = self.idm_head.state_dict()
        if self.return_head is not None:
            ckpt["return_head"] = self.return_head.state_dict()
        if self.vol_head is not None:
            ckpt["vol_head"] = self.vol_head.state_dict()
        if self.drawdown_head is not None:
            ckpt["drawdown_head"] = self.drawdown_head.state_dict()
        if self.crash_head is not None:
            ckpt["crash_head"] = self.crash_head.state_dict()
        if self.drawdown_excess_head is not None:
            ckpt["drawdown_excess_head"] = self.drawdown_excess_head.state_dict()
        if self.position_utility_head is not None:
            ckpt["position_utility_head"] = self.position_utility_head.state_dict()
        if self.overweight_advantage_head is not None:
            ckpt["overweight_advantage_head"] = self.overweight_advantage_head.state_dict()
        if self.recovery_head is not None:
            ckpt["recovery_head"] = self.recovery_head.state_dict()
        if self.regime_head is not None:
            ckpt["regime_head"] = self.regime_head.state_dict()
        atomic_torch_save(ckpt, path)
        print(f"[WM] Checkpoint saved: {path}")

    def load(self, path: str, *, allow_blocked_legacy: bool = False) -> None:
        """チェックポイントをロードする.

        A failed target/gradient coverage gate leaves ``path +
        ".blocked.json"`` beside the checkpoint.  Normal promotion/evaluation
        consumers must not silently read that checkpoint.  The only escape
        hatch is the explicit, strict-boolean ``allow_blocked_legacy=True``
        used by a caller that is intentionally replaying a historical or
        diagnostic artifact; it does not make the checkpoint promotable.
        """
        allow_blocked_legacy = strict_bool_value(
            allow_blocked_legacy,
            name="allow_blocked_legacy",
        )
        marker_path = f"{path}.blocked.json"
        if os.path.exists(marker_path) and not allow_blocked_legacy:
            raise TargetGradientCoverageError(
                "refusing to load coverage-blocked checkpoint for normal "
                f"promotion/evaluation: {path}; marker={marker_path}. "
                "Pass allow_blocked_legacy=True only for explicit historical "
                "diagnostics."
            )
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ensemble.load_state_dict(ckpt["ensemble"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(f"[WM] Optimizer state skipped: {exc}")
        self.global_step = ckpt.get("global_step", 0)
        self.checkpoint_metadata = dict(ckpt.get("checkpoint_metadata") or {})
        if self.idm_head is not None and "idm_head" in ckpt:
            self.idm_head.load_state_dict(ckpt["idm_head"])
        if self.return_head is not None and "return_head" in ckpt:
            self.return_head.load_state_dict(ckpt["return_head"])
        if self.vol_head is not None and "vol_head" in ckpt:
            self.vol_head.load_state_dict(ckpt["vol_head"])
        if self.drawdown_head is not None and "drawdown_head" in ckpt:
            self.drawdown_head.load_state_dict(ckpt["drawdown_head"])
        if self.crash_head is not None and "crash_head" in ckpt:
            self.crash_head.load_state_dict(ckpt["crash_head"])
        if self.drawdown_excess_head is not None and "drawdown_excess_head" in ckpt:
            self.drawdown_excess_head.load_state_dict(ckpt["drawdown_excess_head"])
        if self.position_utility_head is not None and "position_utility_head" in ckpt:
            self.position_utility_head.load_state_dict(ckpt["position_utility_head"])
        if self.overweight_advantage_head is not None and "overweight_advantage_head" in ckpt:
            self.overweight_advantage_head.load_state_dict(ckpt["overweight_advantage_head"])
        if self.recovery_head is not None and "recovery_head" in ckpt:
            self.recovery_head.load_state_dict(ckpt["recovery_head"])
        if self.regime_head is not None and "regime_head" in ckpt:
            self.regime_head.load_state_dict(ckpt["regime_head"])
        print(f"[WM] Checkpoint loaded: {path} (step={self.global_step})")

    @torch.no_grad()
    def predict_auxiliary_from_encoded(
        self,
        z: np.ndarray,
        h: np.ndarray,
        features: np.ndarray | None = None,
        batch_size: int = 8192,
    ) -> dict[str, np.ndarray]:
        """Return predictive auxiliary head outputs for already encoded states."""
        heads = {
            "return": self.return_head,
            "vol": self.vol_head,
            "drawdown": self.drawdown_head,
            "crash": self.crash_head,
            "drawdown_excess": self.drawdown_excess_head,
            "position_utility": self.position_utility_head,
            "overweight_advantage": self.overweight_advantage_head,
            "recovery": self.recovery_head,
        }
        active = {name: head for name, head in heads.items() if head is not None}
        if not active:
            return {}

        was_training = self.ensemble.training
        self.ensemble.eval()
        for head in active.values():
            head.eval()

        z_arr = np.asarray(z, dtype=np.float32)
        h_arr = np.asarray(h, dtype=np.float32)
        feat_arr = None if features is None else np.asarray(features, dtype=np.float32)
        outputs: dict[str, list[np.ndarray]] = {name: [] for name in active}
        n = min(len(z_arr), len(h_arr))
        if feat_arr is not None:
            n = min(n, len(feat_arr))
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            z_t = torch.as_tensor(z_arr[start:end], dtype=torch.float32, device=self.device)
            h_t = torch.as_tensor(h_arr[start:end], dtype=torch.float32, device=self.device)
            obs_t = (
                None
                if feat_arr is None
                else torch.as_tensor(feat_arr[start:end], dtype=torch.float32, device=self.device)
            )
            for name, head in active.items():
                pred = head(z_t, h_t, obs_t)
                if pred.ndim == 1:
                    pred = pred.unsqueeze(-1)
                outputs[name].append(pred.detach().cpu().numpy().astype(np.float32))

        if was_training:
            self.ensemble.train()
        return {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}

    def predictive_feature_names(self) -> list[str]:
        names: list[str] = []
        if self.return_head is not None:
            names.extend([f"wm_pred_return_h{h}" for h in self.return_horizons])
        if self.vol_head is not None:
            names.extend([f"wm_pred_vol_h{h}" for h in self.risk_horizons])
        if self.drawdown_head is not None:
            names.extend([f"wm_pred_drawdown_h{h}" for h in self.risk_horizons])
        if self.crash_head is not None:
            names.extend([f"wm_pred_crash_h{h}" for h in self.risk_horizons])
        if self.drawdown_excess_head is not None:
            names.extend([f"wm_pred_drawdown_excess_h{h}" for h in self.risk_horizons])
        if self.position_utility_head is not None:
            names.extend([f"wm_pred_position_utility_p{pos:g}" for pos in self.position_utility_positions])
        if self.overweight_advantage_head is not None:
            names.extend([f"wm_pred_overweight_advantage_h{h}" for h in self.risk_horizons])
        if self.recovery_head is not None:
            names.extend([f"wm_pred_recovery_h{h}" for h in self.risk_horizons])
        return names
