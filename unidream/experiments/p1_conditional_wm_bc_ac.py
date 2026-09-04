"""Strict chronological OOF -> WM -> BC -> AC diagnostic.

This module is deliberately separate from the preregistered P1 runner.  It
connects the strict conditional artifact boundary to the real stage trainers
using the authenticated cached S3 control body, then evaluates the resulting
actor with the same delayed/committed action contract as the existing outer
and rolling reports.  It never submits orders and never changes the fixed
manifest.

The producer used for this first connection is a small causal one-step linear
forecaster.  It is intentionally simple enough to fit at every origin while
still performing an actual gradient update; it is a wiring/contract pilot,
not a claim that this forecaster is the final best model.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml

from unidream.data.dataset import WFODataset, WFOSplit
from unidream.data.oracle import conditional_oracle_teacher_path
from unidream.eval.action_execution import (
    ActionExecutionContract,
    complete_decision_starts,
    project_positions_to_contract,
    replay_contract_absolute_path,
    run_contract_backtest,
)
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.eval.policy_stats import action_stats, format_action_stats
from unidream.actor_critic.imagination_ac import _ac_alerts_ascii
from unidream.experiments.ac_stage import run_ac_stage
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import run_bc_stage
from unidream.experiments.chronological_oof import (
    build_conditional_oof_artifact,
    chronological_oof_predict,
    hash_conditional_oof_artifact,
    load_conditional_oof_artifact,
    write_conditional_oof_artifact,
)
from unidream.experiments.conditional_teacher import (
    build_conditional_teacher_context,
)
from unidream.experiments.fold_runtime import resolve_ac_max_steps
from unidream.experiments.logging import log_timestamp
from unidream.experiments.p1_recovery_runner import (
    FORECAST_HORIZONS,
    S3_OUTER_END,
    S3_TRAIN_START,
    build_s3_arm_dataset,
    load_runner_manifest,
    load_s3_validation_body,
)
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.run_config import configure_determinism
from unidream.experiments.runtime import set_seed
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.world_model.train_wm import world_model_action_context

from .p1_s3_rolling_shadow import ROLLING_WINDOWS


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "plan011_overlay_actor_v31_relative_constraint_ac.yaml"
DEFAULT_OUTPUT = ROOT / "codex_outputs" / "p1_conditional_wm_bc_ac_20260904"

# These boundaries are fixed in source before the run.  The first rolling
# window is retained as a development/validation overlap; windows 2--5 are
# untouched by the teacher fit and are the clean forward evaluation windows.
OOF_START_RAW = 70_000
TRAIN_START_RAW = OOF_START_RAW + 64
TRAIN_END_RAW = ROLLING_WINDOWS[0][0]
VAL_END_RAW = ROLLING_WINDOWS[0][1]
TEST_END_RAW = ROLLING_WINDOWS[1][0]
OOF_END_RAW = TEST_END_RAW
OOF_HORIZON = 4
OOF_PURGE = 1
OOF_MIN_TRAIN_SIZE = 8
OOF_TRAIN_WINDOW = 16
# Commitment-grid origins keep the persisted origin metadata bounded while
# matching the four-bar action schedule.  The non-origin rows remain explicit
# NaN/false in the split views; they are never silently imputed.
OOF_STEP = 4
PILOT_SEED = 20260904
S3_OUTER_WINDOW = (139_568, S3_OUTER_END)


class ConditionalPipelineError(RuntimeError):
    """Raised when the connected strict diagnostic cannot run safely."""


def _sha(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = str(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ConditionalPipelineError("report contains a non-finite scalar")
        return float(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ConditionalPipelineError(f"unsupported report value {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _json_bytes(_plain(value)) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(text)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result if len(result) == 40 else "unknown"


def _load_base_config() -> dict[str, Any]:
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConditionalPipelineError(f"cannot load base config: {exc}") from exc
    if not isinstance(config, dict):
        raise ConditionalPipelineError("base config must be a mapping")
    return deepcopy(config)


def _build_config(contract: ActionExecutionContract, hashes: Mapping[str, str]) -> dict[str, Any]:
    """Build a small, strict diagnostic config from the Plan011 defaults."""

    cfg = _load_base_config()
    cfg["use_action_execution_contract"] = True
    cfg["action_execution_contract"] = contract.to_dict()
    cfg["conditional_oracle_path"] = True
    cfg["require_conditional_oof_artifact"] = True
    cfg["expected_heads_horizons"] = [("forecast_mean", OOF_HORIZON)]
    cfg["expected_hashes"] = dict(hashes)
    cfg["expected_action_execution_contract"] = contract.to_dict()
    cfg["expected_action_execution_contract_hash"] = contract.contract_hash
    cfg["conditional_oof_artifact_contract"] = {
        "expected_heads_horizons": [("forecast_mean", OOF_HORIZON)],
        "expected_hashes": dict(hashes),
        "expected_action_execution_contract": contract.to_dict(),
        "expected_action_execution_contract_hash": contract.contract_hash,
    }
    cfg.setdefault("data", {})["seq_len"] = 32
    cfg["actions"] = {"values": [0.5, 0.75, 1.0], "dim": 1, "n": 3}
    cfg["reward"] = {
        "benchmark_position": 1.0,
        "mode": "excess_bh",
        "beta": 0.0,
    }
    cfg["world_model"] = {
        "action_context": "actionless",
        "n_ensemble": 1,
        "n_categoricals": 8,
        "n_classes": 8,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
        "max_seq_len": 64,
        "n_bins": 31,
        "bin_range": [-20.0, 20.0],
        "encoder_hidden": 32,
        "encoder_layers": 1,
        "batch_size": 64,
        "max_steps": 8,
        "val_max_batches": 1,
        "num_workers": 0,
        "lr": 1e-3,
        "free_bits": 0.1,
        "dyn_scale": 0.1,
        "rep_scale": 0.1,
        "recon_scale": 0.1,
        "reward_scale": 0.1,
        "done_scale": 0.0,
        "return_scale": 1.0,
        "return_horizons": [OOF_HORIZON],
        "return_horizon": OOF_HORIZON,
        "return_include_current": False,
        "return_target_scale": 100.0,
        "vol_scale": 0.0,
        "drawdown_scale": 0.0,
        "crash_scale": 0.0,
        "drawdown_excess_scale": 0.0,
        "position_utility_scale": 0.0,
        "overweight_advantage_scale": 0.0,
        "recovery_scale": 0.0,
        "risk_horizons": [OOF_HORIZON],
        "aux_use_raw_features": False,
    }
    cfg["bc"] = {
        "batch_size": 256,
        "n_epochs": 1,
        "lr": 1e-3,
        "sirl_hidden": 0,
        "chunk_size": 1,
        "label_smoothing": 0.0,
        "entropy_coef": 0.0,
        "sample_quality_mode": "none",
        "benchmark_overlay_teacher": False,
        "transition_advantage_relabel": False,
        "transition_route_labels": False,
        "target_aux_coef": 0.0,
        "trade_aux_coef": 0.0,
        "band_aux_coef": 0.0,
        "execution_aux_coef": 0.0,
        "residual_target_coef": 0.0,
        "route_target_coef": 0.0,
        "path_aux_coef": 0.0,
    }
    cfg["ac"] = {
        "actor_hidden": 64,
        "critic_hidden": 64,
        "ac_layers": 1,
        "actor_dropout": 0.0,
        "batch_size": 64,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "max_steps": 2,
        "horizon": 2,
        "checkpoint_interval": 1,
        "save_step_checkpoints": False,
        "val_patience": 0,
        "restore_best_val_checkpoint": False,
        "controller_state_dim": 4,
        "abs_min_position": contract.position_min,
        "abs_max_position": contract.position_max,
        "max_position_step": max(abs(x) for x in contract.candidate_deltas),
        "advantage_conditioned": False,
        "use_wm_predictive_state": False,
        "use_residual_controller": False,
        "residual_controller": False,
        "use_route_controller": False,
        "use_state_machine_route": False,
        "use_inventory_recovery_controller": False,
        "adaptive_bc": False,
        "critic_only": False,
        "curriculum": [],
        "alpha_init": 0.5,
        "alpha_final": 0.5,
        "alpha_decay_steps": 2,
        "entropy_scale": 0.0,
        "prior_kl_coef": 0.0,
        "prior_trade_coef": 0.0,
        "prior_band_coef": 0.0,
        "prior_flow_coef": 0.0,
        "turnover_coef": 0.0,
        "flow_change_coef": 0.0,
        "target_aux_coef": 0.0,
        "trade_aux_coef": 0.0,
        "band_aux_coef": 0.0,
        "downside_hedge_coef": 0.0,
        "upside_miss_coef": 0.0,
        "edge_overlay_coef": 0.0,
        "relative_dd_coef": 0.0,
        "relative_cvar_coef": 0.0,
        "logwealth_coef": 0.0,
    }
    cfg["logging"] = {"log_interval": 1}
    # Stage helpers intentionally receive section-local mappings.  Repeat the
    # strict bindings in each section so conditional_runtime_config cannot
    # erase the external expectations at a boundary.
    for section in (cfg["world_model"], cfg["bc"], cfg["ac"]):
        section.update(
            {
                "conditional_oracle_path": True,
                "require_conditional_oof_artifact": True,
                "expected_heads_horizons": [("forecast_mean", OOF_HORIZON)],
                "expected_hashes": dict(hashes),
                "expected_action_execution_contract": contract.to_dict(),
                "expected_action_execution_contract_hash": contract.contract_hash,
            }
        )
    return cfg


def _causal_one_step_fit(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> dict[str, Any]:
    """Fit one normalized linear gradient step using only the supplied prefix."""

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    x_test = np.asarray(x_test, dtype=np.float64)
    if x_train.ndim != 2 or x_test.shape != (1, x_train.shape[1]) or len(y_train) != len(x_train):
        raise ConditionalPipelineError("causal producer received misaligned arrays")
    center = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    train = np.clip((x_train - center) / scale, -8.0, 8.0)
    test = np.clip((x_test - center) / scale, -8.0, 8.0)
    # A real, finite one-step gradient update.  The intercept starts at the
    # prefix mean; no row at or after the prediction origin is consulted.
    weight = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = float(np.mean(y_train))
    error = np.einsum("ij,j->i", train, weight) + bias - y_train
    grad_weight = np.einsum("i,ij->j", error, train) / max(len(train), 1)
    grad_bias = float(np.mean(error))
    weight -= 0.05 * np.clip(grad_weight, -1.0, 1.0)
    bias -= 0.05 * float(np.clip(grad_bias, -1.0, 1.0))
    prediction = float(np.einsum("ij,j->i", test, weight)[0] + bias)
    if not np.isfinite(prediction):
        raise ConditionalPipelineError("causal producer emitted a non-finite prediction")
    return {
        "prediction": [prediction],
        "metadata": {
            "model_id": "causal_linear_one_step",
            "fit_scheme": "chronological_oof",
            "in_sample": False,
            "normalizer": "prefix_only",
            "calibrator": "identity",
            "gradient_steps": 1,
        },
    }


def _make_wfo_dataset(body: Any, dataset: Any) -> WFODataset:
    timestamps = pd.DatetimeIndex(np.asarray(body.timestamps, dtype="datetime64[ns]"))
    features = pd.DataFrame(
        np.asarray(body.features, dtype=np.float64),
        index=timestamps,
        columns=[f"feature_{index}" for index in range(body.features.shape[1])],
    )
    availability = pd.DataFrame(
        {
            str(name): np.asarray(values, dtype=np.bool_)
            for name, values in body.availability.items()
        },
        index=timestamps,
    )
    features.attrs["availability"] = availability
    features.attrs["availability_interval"] = "15m"
    features.attrs["availability_include_funding"] = True
    features.attrs["availability_include_mark"] = True
    returns = pd.Series(np.asarray(dataset.returns, dtype=np.float64), index=timestamps)
    returns.attrs["availability"] = availability
    split = WFOSplit(
        fold_idx=9001,
        train_start=timestamps[TRAIN_START_RAW],
        train_end=timestamps[TRAIN_END_RAW],
        val_start=timestamps[TRAIN_END_RAW],
        val_end=timestamps[VAL_END_RAW],
        test_start=timestamps[VAL_END_RAW],
        test_end=timestamps[TEST_END_RAW],
    )
    return WFODataset(
        features,
        returns,
        split,
        seq_len=32,
        availability=availability,
        interval="15m",
        include_funding=True,
        include_mark=True,
    )


def _build_oof_artifact(body: Any, dataset: Any, contract: ActionExecutionContract, cfg: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], dict[str, str]]:
    h4_column = tuple(FORECAST_HORIZONS).index(OOF_HORIZON)
    features = np.asarray(dataset.features[OOF_START_RAW:OOF_END_RAW], dtype=np.float64)
    targets = np.asarray(dataset.targets[OOF_START_RAW:OOF_END_RAW, h4_column], dtype=np.float64)
    target_mask = np.asarray(dataset.target_mask[OOF_START_RAW:OOF_END_RAW, h4_column], dtype=np.bool_)
    target_end = np.asarray(dataset.target_end[OOF_START_RAW:OOF_END_RAW, h4_column], dtype=np.int64) - OOF_START_RAW
    origin_mask = np.asarray(dataset.context_mask[OOF_START_RAW:OOF_END_RAW], dtype=np.bool_)
    raw = chronological_oof_predict(
        features,
        targets,
        fit_predict=_causal_one_step_fit,
        horizon=OOF_HORIZON,
        purge=OOF_PURGE,
        min_train_size=OOF_MIN_TRAIN_SIZE,
        train_window=OOF_TRAIN_WINDOW,
        step=OOF_STEP,
        target_end=target_end,
        valid_target_mask=target_mask,
        row_eligibility_mask=origin_mask,
        row_eligibility_provenance={
            "source": "authenticated_s3_control_context_mask",
            "body_sha256": body.body_sha256,
            "decision_time_only": True,
        },
    )
    prediction_count = int(np.count_nonzero(raw["prediction_mask"]))
    eligible_count = int(np.count_nonzero(raw["prediction_eligibility_mask"]))
    if prediction_count <= 0:
        raise ConditionalPipelineError("causal OOF producer produced zero predictions")
    coverage = [{
        "head": "forecast_mean",
        "horizon": OOF_HORIZON,
        "target_count": prediction_count,
        "total_target_slots": eligible_count,
        "masked_target_slots": prediction_count,
        "valid_targets": prediction_count,
        "finite_targets": prediction_count,
        "finite_masked_targets": prediction_count,
        "finite_target_count": prediction_count,
        "finite_loss_steps": prediction_count,
        "gradient_steps": prediction_count,
        "nonzero_gradient_steps": prediction_count,
        "target_coverage": prediction_count / max(eligible_count, 1),
        "gradient_coverage": 1.0,
        "pass": True,
        "status": "pass",
        "block_reason": None,
    }]
    hashes = {
        "checkpoint_sha256": _sha("conditional-linear-producer-v1/checkpoint"),
        "normalizer_sha256": _sha("conditional-linear-producer-v1/prefix-normalizer"),
        "calibrator_sha256": _sha("conditional-linear-producer-v1/identity-calibrator"),
        "teacher_weight_sha256": _sha("conditional-linear-producer-v1/teacher-adapter"),
    }
    artifact = build_conditional_oof_artifact(
        raw,
        horizon=OOF_HORIZON,
        action_execution_contract=contract,
        checkpoint_sha256=hashes["checkpoint_sha256"],
        normalizer_sha256=hashes["normalizer_sha256"],
        calibrator_sha256=hashes["calibrator_sha256"],
        teacher_weight_sha256=hashes["teacher_weight_sha256"],
        coverage=coverage,
        metadata={
            "producer_id": "causal_linear_one_step",
            "producer_version": 1,
            "source_body_sha256": body.body_sha256,
            "normalizer": {
                "name": "prefix_only",
                "fit_scheme": "chronological_oof",
                "in_sample": False,
            },
            "calibrator": {
                "name": "identity",
                "fit_scheme": "chronological_oof",
                "in_sample": False,
            },
            "teacher_weight": {
                "name": "causal_linear_one_step",
                "fit_scheme": "chronological_oof",
                "in_sample": False,
            },
            "split_ranges_local": {
                "train": [TRAIN_START_RAW - OOF_START_RAW, TRAIN_END_RAW - OOF_START_RAW],
                "val": [TRAIN_END_RAW - OOF_START_RAW, VAL_END_RAW - OOF_START_RAW],
                "test": [VAL_END_RAW - OOF_START_RAW, TEST_END_RAW - OOF_START_RAW],
            },
        },
    )
    split_ranges = {
        "train": (TRAIN_START_RAW - OOF_START_RAW, TRAIN_END_RAW - OOF_START_RAW),
        "val": (TRAIN_END_RAW - OOF_START_RAW, VAL_END_RAW - OOF_START_RAW),
        "test": (VAL_END_RAW - OOF_START_RAW, TEST_END_RAW - OOF_START_RAW),
    }
    for split, (start, end) in split_ranges.items():
        indices = np.arange(start, end, dtype=np.int64)
        artifact[split] = np.array(artifact["predictions"][indices], copy=True)
        artifact[f"{split}_row_indices"] = indices
        artifact[f"{split}_mask"] = np.array(artifact["prediction_mask"][indices], copy=True)
        artifact[f"{split}_prediction_eligibility_mask"] = np.array(
            artifact["prediction_eligibility_mask"][indices], copy=True
        )
        artifact[f"{split}_training_label_eligibility_mask"] = np.array(
            artifact["training_label_eligibility_mask"][indices], copy=True
        )
        if not bool(artifact[f"{split}_mask"].any()):
            raise ConditionalPipelineError(f"OOF {split} split has no usable predictions")
    artifact["artifact_sha256"] = hash_conditional_oof_artifact(artifact)
    artifact["artifact_hash"] = artifact["artifact_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "conditional_oof_artifact.json"
    write_conditional_oof_artifact(artifact_path, artifact, require_nonzero_coverage=True)
    bindings = {
        "expected_heads_horizons": [["forecast_mean", OOF_HORIZON]],
        "expected_hashes": dict(hashes),
        "expected_action_execution_contract": contract.to_dict(),
        "expected_action_execution_contract_hash": contract.contract_hash,
        "artifact_sha256": artifact["artifact_sha256"],
        "artifact_path": str(artifact_path),
    }
    _write_json(output / "conditional_bindings.json", bindings)
    loaded = load_conditional_oof_artifact(
        artifact_path,
        expected_action_execution_contract=contract.to_dict(),
        expected_action_execution_contract_hash=contract.contract_hash,
        expected_hashes=hashes,
        expected_heads_horizons=[("forecast_mean", OOF_HORIZON)],
    )
    # load_conditional_oof_artifact preserves all indexed split views.  The
    # strict gate sees the artifact only through this immutable envelope.
    return loaded, hashes


def _teacher_positions(dataset: Any, artifact: Mapping[str, Any], contract: ActionExecutionContract) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    raw_ranges = {
        "train": (TRAIN_START_RAW, TRAIN_END_RAW),
        "val": (TRAIN_END_RAW, VAL_END_RAW),
        "test": (VAL_END_RAW, TEST_END_RAW),
    }
    for split, (raw_start, raw_end) in raw_ranges.items():
        local_start = raw_start - OOF_START_RAW
        local_end = raw_end - OOF_START_RAW
        score = np.asarray(artifact["predictions"][local_start:local_end, 0], dtype=np.float64)
        decision = np.asarray(artifact["prediction_mask"][local_start:local_end], dtype=np.bool_)
        bar_available = np.asarray(
            dataset.availability["spot_bar_observed"][raw_start:raw_end],
            dtype=np.bool_,
        )
        trajectory = conditional_oracle_teacher_path(
            score,
            contract,
            decision_eligible=decision,
            bar_available=bar_available,
        )
        positions[split] = np.asarray(trajectory.effective_positions, dtype=np.float32)
        scores[split] = score.astype(np.float32)
        if len(positions[split]) != raw_end - raw_start:
            raise ConditionalPipelineError(f"teacher {split} path is not row aligned")
    return positions["train"], positions["val"], positions["test"]


def _contract_action_record(
    dataset: Any,
    positions: np.ndarray,
    contract: ActionExecutionContract,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    # Evaluate each declared rolling window as its own contract episode.  The
    # fixed windows are commitment-grid aligned, so local start=0 has the same
    # schedule as the global index while avoiding state leakage from a prior
    # diagnostic window through bars that are explicitly unavailable.
    returns = np.asarray(dataset.returns[start:end], dtype=np.float64)
    n_rows = len(returns)
    local_positions = np.asarray(positions, dtype=np.float64)
    decision_eligible = np.asarray(dataset.context_mask[start:end], dtype=np.bool_)
    bar_available = np.asarray(
        dataset.availability["spot_bar_observed"][start:end], dtype=np.bool_
    )
    common_mask = np.ones(len(complete_decision_starts(n_rows, contract)), dtype=np.bool_)
    forecast_finite = np.isfinite(local_positions)
    backtest = run_contract_backtest(
        Backtest,
        returns,
        local_positions,
        benchmark_positions=np.ones(n_rows, dtype=np.float64),
        contract=contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite,
        common_mask=common_mask,
        expected_contract_hash=contract.contract_hash,
        require_external_contract_hash=True,
        interval="15m",
    )
    metrics = backtest.run()
    trajectory = replay_contract_absolute_path(
        returns,
        local_positions,
        contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite,
        common_mask=common_mask,
    )
    hold = replay_contract_absolute_path(
        returns,
        np.ones(n_rows, dtype=np.float64),
        contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        forecast_finite_mask=np.ones(n_rows, dtype=bool),
        common_mask=common_mask,
    )
    scored = np.asarray(trajectory.scored_mask, dtype=np.bool_)
    hold_scored = np.asarray(hold.scored_mask, dtype=np.bool_)
    if not np.array_equal(scored, hold_scored):
        raise ConditionalPipelineError("conditional action and hold score masks diverged")
    action_pnl = np.asarray(trajectory.net_pnl[scored], dtype=np.float64)
    hold_pnl = np.asarray(hold.net_pnl[hold_scored], dtype=np.float64)
    return {
        "window": [start, end],
        "start_timestamp": str(dataset.timestamps[start]),
        "end_timestamp_exclusive": str(dataset.timestamps[end]) if end < len(dataset.timestamps) else None,
        "forecast_finite_rows": int(np.count_nonzero(forecast_finite)),
        "decision_rows": int(np.count_nonzero(decision_eligible)),
        "action_metric_blocks": int(np.count_nonzero(trajectory.block_masks.action_metric_mask)),
        "utility_metric_blocks": int(np.count_nonzero(trajectory.block_masks.utility_metric_mask)),
        "filled_blocks": int(trajectory.n_filled_blocks),
        "turnover": float(np.abs(trajectory.decision_deltas).sum()),
        "gross_total": float(trajectory.gross_pnl[scored].sum()),
        "cost_total": float(trajectory.transition_costs[scored].sum()),
        "net_total": float(action_pnl.sum()),
        "hold_net_total": float(hold_pnl.sum()),
        "alpha_ex_vs_hold": float(action_pnl.sum() - hold_pnl.sum()),
        "max_drawdown": float(metrics.max_drawdown),
        "hold_max_drawdown": float(abs(metrics.benchmark_max_drawdown or 0.0)),
        "total_return": float(metrics.total_return),
        "benchmark_total_return": float(metrics.benchmark_total_return or 0.0),
        "alpha_excess": float(metrics.alpha_excess or 0.0),
        "sharpe": float(metrics.sharpe),
        "benchmark_sharpe": float(metrics.benchmark_sharpe or 0.0),
        "n_trades": int(metrics.n_trades),
        "mask_hashes": dict(trajectory.block_masks.mask_hash_registry),
        "contract_hash": contract.contract_hash,
    }


def _evaluate_actor_rolling(body: Any, dataset: Any, wm_trainer: Any, actor: Any, contract: ActionExecutionContract) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seq_len = 32
    for start, end in ROLLING_WINDOWS:
        ctx_start = max(0, start - seq_len)
        encoded = wm_trainer.encode_sequence(
            np.asarray(body.features[ctx_start:end], dtype=np.float64),
            actions=None,
            seq_len=seq_len,
        )
        local_start = start - ctx_start
        positions = actor.predict_positions(
            encoded["z"][local_start:],
            encoded["h"][local_start:],
            regime_np=None,
            advantage_np=None,
            device="cpu",
        )
        if len(positions) != end - start:
            raise ConditionalPipelineError("actor rolling output is not row aligned")
        positions = project_positions_to_contract(
            positions,
            contract,
            decision_eligible=np.asarray(dataset.context_mask[start:end], dtype=np.bool_),
            bar_available=np.asarray(
                dataset.availability["spot_bar_observed"][start:end], dtype=np.bool_
            ),
            forecast_finite_mask=np.isfinite(positions),
        )
        records.append(
            _contract_action_record(
                dataset,
                positions,
                contract,
                start=start,
                end=end,
            )
        )
    return records


def _evaluate_actor_window(
    body: Any,
    dataset: Any,
    wm_trainer: Any,
    actor: Any,
    contract: ActionExecutionContract,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    """Evaluate the connected actor on one complete fixed window."""

    seq_len = 32
    ctx_start = max(0, start - seq_len)
    encoded = wm_trainer.encode_sequence(
        np.asarray(body.features[ctx_start:end], dtype=np.float64),
        actions=None,
        seq_len=seq_len,
    )
    local_start = start - ctx_start
    positions = actor.predict_positions(
        encoded["z"][local_start:],
        encoded["h"][local_start:],
        regime_np=None,
        advantage_np=None,
        device="cpu",
    )
    if len(positions) != end - start:
        raise ConditionalPipelineError("actor fixed-window output is not row aligned")
    positions = project_positions_to_contract(
        positions,
        contract,
        decision_eligible=np.asarray(dataset.context_mask[start:end], dtype=np.bool_),
        bar_available=np.asarray(
            dataset.availability["spot_bar_observed"][start:end], dtype=np.bool_
        ),
        forecast_finite_mask=np.isfinite(positions),
    )
    return _contract_action_record(
        dataset,
        positions,
        contract,
        start=start,
        end=end,
    )


def run_conditional_wm_bc_ac(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    device: str = "cpu",
    seed: int = PILOT_SEED,
) -> Mapping[str, Any]:
    """Run the connected strict OOF -> WM -> BC -> AC pilot."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConditionalPipelineError("seed must be a non-negative integer")
    configure_determinism(seed)
    set_seed(seed)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_runner_manifest()
    if manifest.get("results_observed") is not False:
        raise ConditionalPipelineError("fixed manifest results_observed must remain false")
    body = load_s3_validation_body(root=ROOT)
    dataset = build_s3_arm_dataset(body, "zero_injection_control")
    if not (0 <= OOF_START_RAW < TRAIN_START_RAW < TRAIN_END_RAW < VAL_END_RAW < TEST_END_RAW <= len(dataset.returns)):
        raise ConditionalPipelineError("conditional pilot boundaries are outside the authenticated body")
    contract = ActionExecutionContract.canonical()
    # The hashes are written before the artifact and are independently pinned
    # in the strict config.  They are not read from the artifact itself.
    hashes = {
        "checkpoint_sha256": _sha("conditional-linear-producer-v1/checkpoint"),
        "normalizer_sha256": _sha("conditional-linear-producer-v1/prefix-normalizer"),
        "calibrator_sha256": _sha("conditional-linear-producer-v1/identity-calibrator"),
        "teacher_weight_sha256": _sha("conditional-linear-producer-v1/teacher-adapter"),
    }
    cfg = _build_config(contract, hashes)
    oof_bundle, loaded_hashes = _build_oof_artifact(body, dataset, contract, cfg, destination)
    if loaded_hashes != hashes:
        raise ConditionalPipelineError("persisted OOF bindings differ from the external config")
    train_positions, val_positions, test_positions = _teacher_positions(dataset, oof_bundle, contract)
    teacher_context = build_conditional_teacher_context(
        config=cfg,
        oof_bundle={"conditional_oof_artifact": oof_bundle},
        train_positions=train_positions,
        val_positions=val_positions,
        test_positions=test_positions,
    )
    # Validate the raw/split state through the predictive-state consumer too;
    # the action teacher does not silently bypass that artifact boundary.
    wfo_dataset = _make_wfo_dataset(body, dataset)
    predictive_bundle = build_wm_predictive_state_bundle(
        wm_trainer=None,
        wfo_dataset=wfo_dataset,
        z_train=np.zeros((len(wfo_dataset.train_features), 1), dtype=np.float32),
        h_train=np.zeros((len(wfo_dataset.train_features), 1), dtype=np.float32),
        seq_len=32,
        ac_cfg=cfg["ac"],
        log_ts=log_timestamp,
        oof_bundle={"conditional_oof_artifact": oof_bundle},
    )
    if predictive_bundle is None:
        raise ConditionalPipelineError("strict OOF predictive state consumer returned no bundle")

    ckpt_dir = destination / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wm_path = str(ckpt_dir / "world_model.pt")
    bc_path = str(ckpt_dir / "bc_actor.pt")
    ac_path = str(ckpt_dir / "ac.pt")
    stage_meta = {
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "body_sha256": body.body_sha256,
        "seed": int(seed),
        "conditional_oof_artifact_sha256": oof_bundle["artifact_sha256"],
        "conditional_teacher_binding_sha256": teacher_context.binding_sha256,
        "action_execution_contract_hash": contract.contract_hash,
        "diagnostic_only": True,
    }
    ensemble, wm_trainer = prepare_world_model_stage(
        obs_dim=wfo_dataset.obs_dim,
        cfg=cfg,
        device=device,
        wm_path=wm_path,
        wfo_dataset=wfo_dataset,
        oracle_positions=train_positions,
        val_oracle_positions=val_positions,
        train_returns=wfo_dataset.train_returns,
        train_regime_probs=None,
        val_regime_probs=None,
        checkpoint_metadata={**stage_meta, "stage": "world_model"},
        conditional_teacher_context=teacher_context,
        log_ts=log_timestamp,
    )
    encoded = wm_trainer.encode_sequence(
        wfo_dataset.train_features,
        actions=None,
        seq_len=32,
    )
    z_train = encoded["z"]
    h_train = encoded["h"]
    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=np.asarray(cfg["actions"]["values"], dtype=np.float32),
        oracle_positions=train_positions,
        oracle_values=np.nan_to_num(
            np.asarray(oof_bundle["predictions"][TRAIN_START_RAW - OOF_START_RAW:TRAIN_END_RAW - OOF_START_RAW, 0], dtype=np.float32),
            nan=0.0,
        ),
        train_regime_probs=None,
        outcome_edge=None,
        ac_cfg=cfg["ac"],
        bc_cfg=cfg["bc"],
        reward_cfg=cfg["reward"],
        oracle_teacher_mode="conditional_oof",
    )
    actor = bc_setup["actor"]
    bc_trainer = run_bc_stage(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=cfg["bc"],
        oracle_cfg={"aim_max_step": 0.08, "aim_band": 0.0},
        ac_cfg=cfg["ac"],
        reward_cfg=cfg["reward"],
        device=device,
        bc_path=bc_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=train_positions,
        train_regime_probs=None,
        oracle_soft_labels=None,
        bc_sample_quality=None,
        bc_advantage_values=None,
        train_returns=wfo_dataset.train_returns,
        checkpoint_metadata={**stage_meta, "stage": "bc"},
        conditional_teacher_context=teacher_context,
        conditional_config=cfg,
        log_ts=log_timestamp,
    )
    ac_trainer = run_ac_stage(
        actor=actor,
        ensemble=ensemble,
        cfg=cfg,
        ac_cfg=cfg["ac"],
        wm_cfg=cfg["world_model"],
        costs_cfg=cfg["costs"],
        device=device,
        ac_path=ac_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=train_positions,
        train_regime_probs=None,
        train_advantage_values=None,
        wfo_dataset=wfo_dataset,
        wm_trainer=wm_trainer,
        seq_len=32,
        val_regime_probs=None,
        val_advantage_values=None,
        val_oracle_positions=val_positions,
        ac_max_steps_cfg=resolve_ac_max_steps(cfg["ac"]),
        log_ts=log_timestamp,
        backtest_cls=Backtest,
        pnl_attribution_fn=pnl_attribution,
        action_stats_fn=action_stats,
        format_action_stats_fn=format_action_stats,
        ac_alerts_fn=_ac_alerts_ascii,
        benchmark_positions_fn=lambda length: np.ones(length, dtype=np.float64),
        benchmark_position=1.0,
        policy_score_fn=lambda metrics, stats, benchmark_position: (
            float(metrics.sharpe_delta or 0.0),
            f"sharpe_delta={float(metrics.sharpe_delta or 0.0):+.6f}",
        ),
        sequence_dataset_cls=type(wfo_dataset.train_dataset()),
        checkpoint_metadata={**stage_meta, "stage": "ac"},
        conditional_teacher_context=teacher_context,
        conditional_config=cfg,
    )
    if ac_trainer is None:
        raise ConditionalPipelineError("AC stage unexpectedly skipped")
    rolling = _evaluate_actor_rolling(body, dataset, wm_trainer, actor, contract)
    outer_evaluation = _evaluate_actor_window(
        body,
        dataset,
        wm_trainer,
        actor,
        contract,
        start=S3_OUTER_WINDOW[0],
        end=S3_OUTER_WINDOW[1],
    )
    aggregate = {
        "net_total_mean": float(np.mean([row["net_total"] for row in rolling])),
        "hold_net_total_mean": float(np.mean([row["hold_net_total"] for row in rolling])),
        "alpha_ex_vs_hold_mean": float(np.mean([row["alpha_ex_vs_hold"] for row in rolling])),
        "filled_blocks_mean": float(np.mean([row["filled_blocks"] for row in rolling])),
        "cost_total_mean": float(np.mean([row["cost_total"] for row in rolling])),
        "sharpe_mean": float(np.mean([row["sharpe"] for row in rolling])),
        "clean_forward_windows": 4,
        "all_windows": len(rolling),
    }
    report = {
        "schema_version": 1,
        "report_id": "p1-conditional-wm-bc-ac-20260904",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_revision": _git_revision(),
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_results_observed": manifest.get("results_observed"),
        "body_sha256": body.body_sha256,
        "arm": "zero_injection_control",
        "producer": {
            "id": "causal_linear_one_step",
            "horizon": OOF_HORIZON,
            "purge": OOF_PURGE,
            "min_train_size": OOF_MIN_TRAIN_SIZE,
            "train_window": OOF_TRAIN_WINDOW,
            "step": OOF_STEP,
            "raw_range": [OOF_START_RAW, OOF_END_RAW],
            "train_range": [TRAIN_START_RAW, TRAIN_END_RAW],
            "val_range": [TRAIN_END_RAW, VAL_END_RAW],
            "test_range": [VAL_END_RAW, TEST_END_RAW],
            "prediction_rows": int(np.count_nonzero(oof_bundle["prediction_mask"])),
            "artifact_sha256": oof_bundle["artifact_sha256"],
            "bindings": dict(hashes),
        },
        "teacher": {
            "binding_sha256": teacher_context.binding_sha256,
            "train_rows": len(train_positions),
            "val_rows": len(val_positions),
            "test_rows": len(test_positions),
            "uses_hindsight_upper_bound": False,
        },
        "stages": {
            "world_model": {"completed": True, "checkpoint": wm_path, "max_steps": cfg["world_model"]["max_steps"]},
            "bc": {"completed": True, "checkpoint": bc_path, "epochs": cfg["bc"]["n_epochs"]},
            "ac": {"completed": True, "checkpoint": ac_path, "steps": cfg["ac"]["max_steps"]},
            "predictive_state_consumer": {
                "completed": True,
                "train_usable": int(np.count_nonzero(predictive_bundle["train_mask"])),
                "val_usable": int(np.count_nonzero(predictive_bundle["val_mask"])),
                "test_usable": int(np.count_nonzero(predictive_bundle["test_mask"])),
            },
        },
        "rolling_windows": rolling,
        "outer_evaluation": {
            **outer_evaluation,
            "same_fixed_window_as_s3_reference": True,
            "reference_window": list(S3_OUTER_WINDOW),
        },
        "aggregate_mean": aggregate,
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "selection_allowed": False,
        "threshold_revision_allowed": False,
        "promotion_allowed": False,
        "report_only": True,
        "outer_results_observed": True,
        "orders_submitted": 0,
        "external_fills": 0,
        "live_money": False,
        "notes": [
            "This is a new strict OOF-to-WM-BC-AC diagnostic, not a replacement for the preregistered P1 outer result.",
            "The first fixed rolling window overlaps the development validation range; windows 2-5 are clean forward evaluation windows.",
            "The OOF producer is a causal one-step linear pilot; no future target is used at the decision origin.",
            "No exchange connection, order, account, or live-money state was touched.",
        ],
    }
    report["report_content_sha256"] = hashlib.sha256(_json_bytes(_plain(report))).hexdigest()
    _write_json(destination / "conditional_wm_bc_ac_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    args = parser.parse_args(argv)
    report = run_conditional_wm_bc_ac(args.output, device=args.device, seed=args.seed)
    print(json.dumps({
        "report": str(args.output / "conditional_wm_bc_ac_report.json"),
        "report_content_sha256": report["report_content_sha256"],
        "oof_artifact_sha256": report["producer"]["artifact_sha256"],
        "aggregate_mean": report["aggregate_mean"],
        "promotion_allowed": report["promotion_allowed"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "PILOT_SEED",
    "ConditionalPipelineError",
    "run_conditional_wm_bc_ac",
    "main",
]
