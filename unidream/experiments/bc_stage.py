from __future__ import annotations

import math

import numpy as np
import torch

from unidream.actor_critic.bc_pretrain import BCPretrainer


def build_bc_trainer(
    *,
    actor,
    ensemble,
    bc_cfg: dict,
    oracle_cfg: dict,
    ac_cfg: dict,
    reward_cfg: dict,
    device: str,
) -> BCPretrainer:
    return BCPretrainer(
        actor=actor,
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        lr=bc_cfg.get("lr", 3e-4),
        batch_size=bc_cfg.get("batch_size", 256),
        n_epochs=bc_cfg.get("n_epochs", 5),
        sirl_hidden=bc_cfg.get("sirl_hidden", 128),
        label_smoothing=bc_cfg.get("label_smoothing", 0.0),
        entropy_coef=bc_cfg.get("entropy_coef", 0.0),
        chunk_size=bc_cfg.get("chunk_size", 1),
        class_balanced=bc_cfg.get("class_balanced", False),
        target_aux_coef=bc_cfg.get("target_aux_coef", 1.0),
        trade_aux_coef=bc_cfg.get("trade_aux_coef", 0.5),
        band_aux_coef=bc_cfg.get("band_aux_coef", 0.25),
        execution_aux_coef=bc_cfg.get("execution_aux_coef", 0.0),
        path_aux_coef=bc_cfg.get("path_aux_coef", 0.0),
        path_horizon=bc_cfg.get("path_horizon", 1),
        path_position_coef=bc_cfg.get("path_position_coef", 1.0),
        path_turnover_coef=bc_cfg.get("path_turnover_coef", 0.0),
        path_shortfall_coef=bc_cfg.get("path_shortfall_coef", 0.0),
        soft_trade_targets=bc_cfg.get("soft_trade_targets", True),
        trade_target_scale=bc_cfg.get("trade_target_scale"),
        self_condition_prob=bc_cfg.get("self_condition_prob", 0.0),
        self_condition_interval=bc_cfg.get("self_condition_interval", 1),
        self_condition_warmup_epochs=bc_cfg.get("self_condition_warmup_epochs", 0),
        self_condition_mode=bc_cfg.get("self_condition_mode", "mix"),
        self_condition_blend=bc_cfg.get("self_condition_blend", 0.0),
        self_condition_max_position_gap=bc_cfg.get("self_condition_max_position_gap"),
        self_condition_max_underweight_gap=bc_cfg.get("self_condition_max_underweight_gap"),
        self_condition_relabel_step=bc_cfg.get("self_condition_relabel_step"),
        self_condition_relabel_band=bc_cfg.get("self_condition_relabel_band", 0.0),
        relabel_aim_max_step=oracle_cfg.get("aim_max_step", 0.125),
        relabel_aim_band=oracle_cfg.get("aim_band", 0.0),
        relabel_min_position=ac_cfg.get("abs_min_position", -1.0),
        relabel_max_position=ac_cfg.get("abs_max_position", 1.0),
        relabel_benchmark_position=reward_cfg.get("benchmark_position", 0.0),
        relabel_underweight_confirm_bars=oracle_cfg.get("aim_underweight_confirm_bars", 0),
        relabel_underweight_min_scale=oracle_cfg.get("aim_underweight_min_scale", 0.0),
        relabel_underweight_floor_position=oracle_cfg.get("aim_underweight_floor_position"),
        relabel_underweight_step_scale=oracle_cfg.get("aim_underweight_step_scale", 1.0),
        residual_target_coef=bc_cfg.get("residual_target_coef", 1.0),
        residual_aux_ce_coef=bc_cfg.get("residual_aux_ce_coef", 0.0),
        target_dist_match_coef=bc_cfg.get("target_dist_match_coef", 0.0),
        position_mean_match_coef=bc_cfg.get("position_mean_match_coef", 0.0),
        target_regime_dist_match_coef=bc_cfg.get("target_regime_dist_match_coef", 0.0),
        short_mass_match_coef=bc_cfg.get("short_mass_match_coef", 0.0),
        support_prior_coef=bc_cfg.get("support_prior_coef", 0.0),
        support_prior_underweight_only=bc_cfg.get("support_prior_underweight_only", True),
        mode_target_coef=bc_cfg.get("mode_target_coef", 0.0),
        mode_target_margin=bc_cfg.get("mode_target_margin", 0.05),
        mode_target_neutral_margin=bc_cfg.get("mode_target_neutral_margin", 0.0),
        mode_target_gap_min=bc_cfg.get("mode_target_gap_min", 0.0),
        mode_target_positive_only=bc_cfg.get("mode_target_positive_only", False),
        mode_target_use_underweight_strength=bc_cfg.get("mode_target_use_underweight_strength", False),
        mode_rate_match_coef=bc_cfg.get("mode_rate_match_coef", 0.0),
        mode_regime_rate_match_coef=bc_cfg.get("mode_regime_rate_match_coef", 0.0),
        dual_head_anchor_coef=bc_cfg.get("dual_head_anchor_coef", 0.0),
        dual_head_underweight_coef=bc_cfg.get("dual_head_underweight_coef", 0.0),
        dual_head_separation_coef=bc_cfg.get("dual_head_separation_coef", 0.0),
        dual_head_neutral_margin=bc_cfg.get("dual_head_neutral_margin", 0.0),
        dual_head_underweight_margin=bc_cfg.get("dual_head_underweight_margin", 0.05),
        dual_head_separation_margin=bc_cfg.get("dual_head_separation_margin", 0.05),
        direct_band_target_coef=bc_cfg.get("direct_band_target_coef", 0.0),
        direct_band_margin=bc_cfg.get("direct_band_margin", 0.05),
        direct_hold_band_margin=bc_cfg.get("direct_hold_band_margin", 0.02),
        band_aux_trade_only=bc_cfg.get("band_aux_trade_only", False),
        direct_band_trade_only=bc_cfg.get("direct_band_trade_only", False),
        direct_band_gap_min=bc_cfg.get("direct_band_gap_min", 0.0),
        recovery_trade_coef=bc_cfg.get("recovery_trade_coef", 0.0),
        recovery_band_coef=bc_cfg.get("recovery_band_coef", 0.0),
        recovery_target_coef=bc_cfg.get("recovery_target_coef", 0.0),
        recovery_execution_coef=bc_cfg.get("recovery_execution_coef", 0.0),
        recovery_underweight_margin=bc_cfg.get("recovery_underweight_margin", 0.05),
        recovery_target_margin=bc_cfg.get("recovery_target_margin", 0.05),
        sample_quality_coef=bc_cfg.get("sample_quality_coef", 0.0),
        sample_quality_clip=bc_cfg.get("sample_quality_clip", 4.0),
        trainable_actor_prefixes=bc_cfg.get("trainable_actor_prefixes"),
        device=device,
    )


def _reinitialize_actor_prefixes(actor, prefixes: list[str] | tuple[str, ...]) -> None:
    prefix_tuple = tuple(prefixes)
    if not prefix_tuple:
        return
    for name, module in actor.named_modules():
        if not name or not any(name.startswith(prefix) for prefix in prefix_tuple):
            continue
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    if any(prefix.startswith("target_mode_gate") for prefix in prefix_tuple):
        torch.nn.init.constant_(actor.target_mode_gate.bias, -0.5)
        torch.nn.init.zeros_(actor.target_mode_gate.weight)
    residual_min = float(getattr(actor, "residual_min_overlay", -1.0))
    residual_max = float(getattr(actor, "residual_max_overlay", 0.0))
    if residual_max > residual_min + 1e-6:
        init_overlay = float(getattr(actor, "residual_init_overlay", 0.0))
        init_frac = np.clip((init_overlay - residual_min) / (residual_max - residual_min), 1e-4, 1.0 - 1e-4)
        residual_bias = float(math.log(init_frac / (1.0 - init_frac)))
        if any(prefix.startswith("residual_head_b") for prefix in prefix_tuple):
            torch.nn.init.constant_(actor.residual_head_b.bias, residual_bias)
            torch.nn.init.zeros_(actor.residual_head_b.weight)
        if any(prefix.startswith("residual_head_a") for prefix in prefix_tuple):
            bench_overlay = 0.0
            bench_frac = np.clip((bench_overlay - residual_min) / (residual_max - residual_min), 1e-4, 1.0 - 1e-4)
            bench_bias = float(math.log(bench_frac / (1.0 - bench_frac)))
            torch.nn.init.constant_(actor.residual_head_a.bias, bench_bias)
            torch.nn.init.zeros_(actor.residual_head_a.weight)


def run_bc_stage(
    *,
    actor,
    ensemble,
    bc_cfg: dict,
    oracle_cfg: dict,
    ac_cfg: dict,
    reward_cfg: dict,
    device: str,
    has_bc: bool,
    start_idx: int,
    bc_stage_idx: int,
    bc_path: str,
    z_train,
    h_train,
    oracle_positions,
    train_regime_probs,
    oracle_soft_labels,
    bc_sample_quality,
    bc_advantage_values,
    log_ts,
):
    bc_trainer = build_bc_trainer(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
    )
    if has_bc:
        print(f"\n[{log_ts()}] [Step 3] BC - loading checkpoint: {bc_path}")
        bc_trainer.load(bc_path)
        return bc_trainer

    if start_idx <= bc_stage_idx:
        init_bc_path = bc_cfg.get("init_checkpoint")
        if init_bc_path:
            print(f"\n[{log_ts()}] [Step 3] BC - warm start from: {init_bc_path}")
            bc_trainer.load(init_bc_path)
        reinit_prefixes = bc_cfg.get("reinit_actor_prefixes", [])
        if reinit_prefixes:
            _reinitialize_actor_prefixes(actor, reinit_prefixes)
        print(f"\n[{log_ts()}] [Step 3] BC Pre-training...")
        t_enc = min(len(z_train), len(oracle_positions))
        bc_trainer.train(
            z=z_train[:t_enc],
            h=h_train[:t_enc],
            oracle_positions=oracle_positions[:t_enc],
            regime_probs=train_regime_probs[:t_enc] if train_regime_probs is not None else None,
            soft_labels=oracle_soft_labels[:t_enc] if oracle_soft_labels is not None else None,
            sample_quality=bc_sample_quality[:t_enc] if bc_sample_quality is not None else None,
            advantage_values=bc_advantage_values[:t_enc] if bc_advantage_values is not None else None,
        )
        bc_trainer.save(bc_path)
        return bc_trainer

    print(f"\n[{log_ts()}] [Step 3] BC - skipped (AC checkpoint will provide actor weights)")
    return bc_trainer
