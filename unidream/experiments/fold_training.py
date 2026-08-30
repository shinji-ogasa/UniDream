"""Application orchestration for one current-spec WFO fold.

The CLI owns argument parsing only.  This module wires the fold stages while
keeping model implementations, metrics, and selector policy in their own
modules.
"""
from __future__ import annotations

from copy import deepcopy
import os

from unidream.actor_critic.imagination_ac import _ac_alerts_ascii as ac_alerts
from unidream.data.dataset import SequenceDataset, WFODataset
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.eval.policy_stats import action_stats, format_action_stats
from unidream.eval.selector import (
    benchmark_positions,
    candidate_to_text,
    policy_score,
    select_policy_candidate,
    selector_candidate,
    selector_config,
)
from unidream.experiments.ac_stage import run_ac_stage
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import run_bc_stage
from unidream.experiments.checkpointing import (
    checkpoint_metadata_for_fold,
    snapshot_actor_inference_settings,
)
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.logging import log_timestamp
from unidream.experiments.m2 import (
    benchmark_position_value,
    format_m2_scorecard,
    m2_scorecard,
)
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.fold_runtime import resolve_ac_max_steps
from unidream.experiments.chronological_oof import (
    ConditionalPathBlocked,
    conditional_path_or_artifact_enabled,
    conditional_runtime_config,
)
from unidream.experiments.test_stage import run_test_stage
from unidream.experiments.val_selector_stage import run_val_selector_stage
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.data.oracle import _forward_window_stats


def run_fold(
    fold_idx: int,
    wfo_dataset: WFODataset,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
    run_manifest: dict | None = None,
) -> dict:
    """Train and evaluate one configured WFO fold from scratch."""
    print(f"\n{'=' * 60}")
    print(
        f"Fold {fold_idx}: train {wfo_dataset.split.train_start.date()} → "
        f"{wfo_dataset.split.train_end.date()} | "
        f"test {wfo_dataset.split.test_start.date()} → "
        f"{wfo_dataset.split.test_end.date()}"
    )
    print(f"{'=' * 60}")

    # Predictive-state construction adds fold-local Actor dimensions to the
    # AC config. Keep derived values out of the immutable run config stored in
    # the manifest so replay can use the same config hash in this process too.
    fold_cfg = deepcopy(cfg)
    # Propagate affirmative conditional flags from the complete manifest into
    # every stage-local mapping.  Otherwise a top-level strict OOF request can
    # be lost when a legacy stage receives only ``ac``/``bc``/``world_model``.
    ac_cfg = conditional_runtime_config(fold_cfg, fold_cfg.get("ac", {}))
    bc_cfg = conditional_runtime_config(fold_cfg, fold_cfg.get("bc", {}))
    wm_cfg = conditional_runtime_config(fold_cfg, fold_cfg.get("world_model", {}))
    costs_cfg = fold_cfg.get("costs", {})
    reward_cfg = fold_cfg.get("reward", {})
    obs_dim = wfo_dataset.obs_dim
    seq_len = fold_cfg.get("data", {}).get("seq_len", 64)

    if (
        conditional_path_or_artifact_enabled(fold_cfg)
        or conditional_path_or_artifact_enabled(ac_cfg)
        or conditional_path_or_artifact_enabled(bc_cfg)
        or conditional_path_or_artifact_enabled(wm_cfg)
    ):
        raise ConditionalPathBlocked(
            "run_fold is blocked for conditional Oracle until chronological OOF WM "
            "retraining, normalizer/calibrator provenance, and replay inventory are supplied"
        )

    fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    wm_path = os.path.join(fold_ckpt_dir, "world_model.pt")
    bc_path = os.path.join(fold_ckpt_dir, "bc_actor.pt")
    ac_path = os.path.join(fold_ckpt_dir, "ac.pt")
    wm_checkpoint_metadata = checkpoint_metadata_for_fold(
        run_manifest,
        fold_idx=fold_idx,
        stage="world_model",
    )
    bc_checkpoint_metadata = checkpoint_metadata_for_fold(
        run_manifest,
        fold_idx=fold_idx,
        stage="bc_actor",
    )
    ac_checkpoint_metadata = checkpoint_metadata_for_fold(
        run_manifest,
        fold_idx=fold_idx,
        stage="ac",
    )

    fold_inputs = prepare_fold_inputs(
        fold_idx=fold_idx,
        wfo_dataset=wfo_dataset,
        cfg=fold_cfg,
        costs_cfg=costs_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        action_stats_fn=action_stats,
        format_action_stats_fn=format_action_stats,
        benchmark_position=benchmark_position_value(fold_cfg),
        forward_window_stats_fn=_forward_window_stats,
        log_ts=log_timestamp,
    )
    train_returns = fold_inputs["train_returns"]
    oracle_cfg = fold_inputs["oracle_cfg"]
    oracle_bundle = fold_inputs["oracle_bundle"]
    oracle_action_values = oracle_bundle["oracle_action_values"]
    oracle_soft_labels = oracle_bundle["oracle_soft_labels"]
    oracle_positions = fold_inputs["oracle_positions"]
    val_oracle_positions = fold_inputs["val_oracle_positions"]
    outcome_edge = fold_inputs["outcome_edge"]
    train_regime_probs = fold_inputs["train_regime_probs"]
    val_regime_probs = fold_inputs["val_regime_probs"]
    test_regime_probs = fold_inputs["test_regime_probs"]
    train_advantage_values = fold_inputs.get("train_advantage_values")
    val_advantage_values = fold_inputs.get("val_advantage_values")
    test_advantage_values = fold_inputs.get("test_advantage_values")

    ensemble, wm_trainer = prepare_world_model_stage(
        obs_dim=obs_dim,
        cfg=fold_cfg,
        device=device,
        wm_path=wm_path,
        wfo_dataset=wfo_dataset,
        oracle_positions=oracle_positions,
        val_oracle_positions=val_oracle_positions,
        train_returns=train_returns,
        train_regime_probs=train_regime_probs,
        val_regime_probs=val_regime_probs,
        checkpoint_metadata=wm_checkpoint_metadata,
        log_ts=log_timestamp,
    )

    encoded = wm_trainer.encode_sequence(
        wfo_dataset.train_features,
        actions=None,
        seq_len=seq_len,
    )
    z_train = encoded["z"]
    h_train = encoded["h"]
    predictive_bundle = build_wm_predictive_state_bundle(
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        z_train=z_train,
        h_train=h_train,
        seq_len=seq_len,
        ac_cfg=ac_cfg,
        log_ts=log_timestamp,
    )
    if predictive_bundle is not None:
        ac_cfg["advantage_conditioned"] = True
        ac_cfg["advantage_dim"] = int(predictive_bundle["train"].shape[1])
        train_advantage_values = predictive_bundle["train"]
        val_advantage_values = predictive_bundle["val"]
        test_advantage_values = predictive_bundle["test"]

    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=oracle_action_values,
        oracle_positions=oracle_positions,
        oracle_values=oracle_bundle["oracle_values"],
        train_regime_probs=train_regime_probs,
        outcome_edge=outcome_edge,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        oracle_teacher_mode=oracle_bundle["oracle_teacher_mode"],
    )
    actor = bc_setup["actor"]
    bc_sample_quality = bc_setup["bc_sample_quality"]
    bc_advantage_values = (
        train_advantage_values
        if train_advantage_values is not None
        else bc_setup["bc_advantage_values"]
    )
    bc_trainer = run_bc_stage(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
        bc_path=bc_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=oracle_positions,
        train_regime_probs=train_regime_probs,
        oracle_soft_labels=oracle_soft_labels,
        bc_sample_quality=bc_sample_quality,
        bc_advantage_values=bc_advantage_values,
        train_returns=train_returns,
        train_route_labels=fold_inputs.get("train_route_labels"),
        train_route_soft_labels=fold_inputs.get("train_route_soft_labels"),
        train_route_advantage=fold_inputs.get("train_route_advantage"),
        checkpoint_metadata=bc_checkpoint_metadata,
        log_ts=log_timestamp,
    )

    ac_trainer = run_ac_stage(
        actor=actor,
        ensemble=ensemble,
        cfg=fold_cfg,
        ac_cfg=ac_cfg,
        wm_cfg=wm_cfg,
        costs_cfg=costs_cfg,
        device=device,
        ac_path=ac_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=oracle_positions,
        train_regime_probs=train_regime_probs,
        train_advantage_values=train_advantage_values,
        wfo_dataset=wfo_dataset,
        wm_trainer=wm_trainer,
        seq_len=seq_len,
        val_regime_probs=val_regime_probs,
        val_advantage_values=val_advantage_values,
        val_oracle_positions=val_oracle_positions,
        ac_max_steps_cfg=resolve_ac_max_steps(ac_cfg),
        log_ts=log_timestamp,
        backtest_cls=Backtest,
        pnl_attribution_fn=pnl_attribution,
        action_stats_fn=action_stats,
        format_action_stats_fn=format_action_stats,
        ac_alerts_fn=ac_alerts,
        benchmark_positions_fn=lambda length: benchmark_positions(length, fold_cfg),
        benchmark_position=benchmark_position_value(fold_cfg),
        policy_score_fn=policy_score,
        sequence_dataset_cls=SequenceDataset,
        checkpoint_metadata=ac_checkpoint_metadata,
    )

    inference_selection = run_val_selector_stage(
        actor=actor,
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        seq_len=seq_len,
        val_regime_probs=val_regime_probs,
        val_advantage_values=val_advantage_values,
        device=device,
        cfg=fold_cfg,
        ac_cfg=ac_cfg,
        costs_cfg=costs_cfg,
        backtest_cls=Backtest,
        action_stats_fn=action_stats,
        selector_cfg_fn=selector_config,
        selector_candidate_fn=selector_candidate,
        select_policy_candidate_fn=select_policy_candidate,
        candidate_to_text_fn=candidate_to_text,
        benchmark_positions_fn=lambda length: benchmark_positions(length, fold_cfg),
        benchmark_position=benchmark_position_value(fold_cfg),
    )

    # Validation selector は AC（または BC-only）の学習後に実行される。
    # selector 結果を反映した actor と設定を最終 policy artifact に再保存し、
    # 学習直後と後日 replay の推論経路を同一にする。
    final_policy_trainer = ac_trainer or bc_trainer
    final_policy_path = ac_path if ac_trainer is not None else bc_path
    if final_policy_trainer is not None and final_policy_path:
        final_policy_trainer.checkpoint_metadata["inference_selection"] = inference_selection or {
            "source": "config_default",
            "adjust_rate_scale": float(getattr(actor, "infer_adjust_rate_scale", 1.0)),
            "advantage_level": float(getattr(actor, "infer_advantage_level", 0.0)),
        }
        final_policy_trainer.checkpoint_metadata["inference_settings"] = snapshot_actor_inference_settings(actor)
        final_policy_trainer.save(final_policy_path)

    test_result = run_test_stage(
        actor=actor,
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        seq_len=seq_len,
        test_regime_probs=test_regime_probs,
        test_advantage_values=test_advantage_values,
        device=device,
        cfg=fold_cfg,
        costs_cfg=costs_cfg,
        backtest_cls=Backtest,
        pnl_attribution_fn=pnl_attribution,
        action_stats_fn=action_stats,
        format_action_stats_fn=format_action_stats,
        ac_alerts_fn=ac_alerts,
        benchmark_positions_fn=lambda length: benchmark_positions(length, fold_cfg),
        benchmark_position=benchmark_position_value(fold_cfg),
        m2_scorecard_fn=m2_scorecard,
        format_m2_scorecard_fn=format_m2_scorecard,
        log_ts=log_timestamp,
        fold_idx=fold_idx,
        override_positions=None,
        override_policy_name=None,
    )
    test_result["fold"] = fold_idx
    return test_result
