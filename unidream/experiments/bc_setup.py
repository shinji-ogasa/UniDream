import numpy as np
import torch

from unidream.actor_critic.actor import Actor


def _safe_logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-4, 1.0 - 1e-4)
    return np.log(prob / (1.0 - prob))


def _initialize_regime_mode_gate_from_teacher(
    *,
    actor,
    oracle_positions,
    train_regime_probs,
    benchmark_position: float,
    bc_cfg: dict,
) -> None:
    if actor.regime_mode_gate_head is None or train_regime_probs is None:
        return
    if len(oracle_positions) == 0 or len(train_regime_probs) == 0:
        return

    underweight_margin = float(
        bc_cfg.get(
            "init_regime_mode_gate_margin",
            bc_cfg.get("dual_head_underweight_margin", bc_cfg.get("mode_target_margin", 0.10)),
        )
    )
    min_samples = float(max(bc_cfg.get("init_regime_mode_gate_min_samples", 64.0), 1.0))
    prior_scale = float(max(bc_cfg.get("init_regime_mode_gate_weight_scale", 1.0), 0.0))

    oracle_positions = np.asarray(oracle_positions, dtype=np.float32)
    regime_probs = np.asarray(train_regime_probs[: len(oracle_positions)], dtype=np.float32)
    if regime_probs.ndim != 2 or regime_probs.shape[1] == 0:
        return

    underweight_mask = (oracle_positions < (benchmark_position - underweight_margin)).astype(np.float32)
    regime_mass = regime_probs.sum(axis=0)
    total_mass = float(regime_mass.sum())
    if total_mass <= 0.0:
        return

    global_rate = float((regime_probs * underweight_mask[:, None]).sum() / max(total_mass, 1e-6))
    global_logit = float(_safe_logit(np.asarray(global_rate, dtype=np.float32)))

    raw_regime_rate = (regime_probs * underweight_mask[:, None]).sum(axis=0) / np.clip(regime_mass, 1e-6, None)
    shrink = regime_mass / (regime_mass + min_samples)
    shrunk_rate = shrink * raw_regime_rate + (1.0 - shrink) * global_rate
    regime_logits = _safe_logit(shrunk_rate)

    occupancy = regime_mass / total_mass
    centered_logits = regime_logits - float(np.sum(occupancy * regime_logits))

    torch.nn.init.constant_(actor.target_mode_gate.bias, global_logit)
    torch.nn.init.zeros_(actor.target_mode_gate.weight)
    with torch.no_grad():
        actor.regime_mode_gate_head.weight.zero_()
        actor.regime_mode_gate_head.weight[0, : centered_logits.shape[0]] = (
            torch.as_tensor(centered_logits, dtype=actor.regime_mode_gate_head.weight.dtype)
            * prior_scale
        )


def _initialize_regime_residual_shift_from_teacher(
    *,
    actor,
    oracle_positions,
    train_regime_probs,
    benchmark_position: float,
    bc_cfg: dict,
) -> None:
    if actor.regime_residual_shift_head is None or train_regime_probs is None:
        return
    shift_scale = float(getattr(actor, "regime_residual_shift_scale", 0.0))
    if shift_scale <= 0.0:
        return
    if len(oracle_positions) == 0 or len(train_regime_probs) == 0:
        return

    min_samples = float(max(bc_cfg.get("init_regime_residual_shift_min_samples", 64.0), 1.0))
    clip_ratio = float(np.clip(bc_cfg.get("init_regime_residual_shift_clip_ratio", 0.9), 1e-3, 0.999))

    oracle_overlay = np.asarray(oracle_positions, dtype=np.float32) - float(benchmark_position)
    regime_probs = np.asarray(train_regime_probs[: len(oracle_overlay)], dtype=np.float32)
    if regime_probs.ndim != 2 or regime_probs.shape[1] == 0:
        return

    regime_mass = regime_probs.sum(axis=0)
    total_mass = float(regime_mass.sum())
    if total_mass <= 0.0:
        return

    global_overlay = float((regime_probs * oracle_overlay[:, None]).sum() / max(total_mass, 1e-6))
    raw_regime_overlay = (regime_probs * oracle_overlay[:, None]).sum(axis=0) / np.clip(regime_mass, 1e-6, None)
    shrink = regime_mass / (regime_mass + min_samples)
    shrunk_overlay = shrink * raw_regime_overlay + (1.0 - shrink) * global_overlay
    centered_shift = shrunk_overlay - float(np.sum((regime_mass / total_mass) * shrunk_overlay))

    max_abs_shift = max(shift_scale * clip_ratio, 1e-6)
    clipped_shift = np.clip(centered_shift, -max_abs_shift, max_abs_shift)
    normalized_shift = np.clip(clipped_shift / shift_scale, -clip_ratio, clip_ratio)
    raw_weights = np.arctanh(normalized_shift)

    with torch.no_grad():
        actor.regime_residual_shift_head.weight.zero_()
        actor.regime_residual_shift_head.weight[0, : raw_weights.shape[0]] = torch.as_tensor(
            raw_weights,
            dtype=actor.regime_residual_shift_head.weight.dtype,
        )


def prepare_bc_setup(
    *,
    ensemble,
    oracle_action_values,
    oracle_positions,
    oracle_values,
    train_regime_probs,
    outcome_edge,
    ac_cfg: dict,
    bc_cfg: dict,
    reward_cfg: dict,
    oracle_teacher_mode: str,
):
    actor = Actor(
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        act_dim=int(len(oracle_action_values)),
        hidden_dim=ac_cfg.get("actor_hidden", 256),
        n_layers=ac_cfg.get("ac_layers", 2),
        regime_dim=0 if train_regime_probs is None else int(train_regime_probs.shape[1]),
        dropout_p=ac_cfg.get("actor_dropout", 0.0),
        inventory_dim=ac_cfg.get("controller_state_dim", 1),
        advantage_dim=1 if ac_cfg.get("advantage_conditioned", False) else 0,
    )
    benchmark_position = float(reward_cfg.get("benchmark_position", 1.0))
    actor.target_values = oracle_action_values.astype(np.float32)
    actor.benchmark_position = benchmark_position
    actor.baseline_target_index = int(np.argmin(np.abs(oracle_action_values - benchmark_position)))
    actor.abs_min_position = ac_cfg.get("abs_min_position", -1.0)
    actor.abs_max_position = ac_cfg.get("abs_max_position", 1.0)
    actor.infer_temperature = ac_cfg.get("infer_temperature", 1.0)
    actor.infer_advantage_level = ac_cfg.get("infer_advantage_level", 0.0)
    actor.infer_gap_boost = ac_cfg.get("infer_gap_boost", 0.0)
    actor.infer_adjust_rate_scale = ac_cfg.get("infer_adjust_rate_scale", 1.0)
    actor.adjustment_temperature = ac_cfg.get("adjustment_temperature", 0.25)
    actor.max_position_step = ac_cfg.get("max_position_step", 10.0)
    actor.min_band = ac_cfg.get("min_band", 0.02)
    actor.max_band = ac_cfg.get("max_band", 0.20)
    actor.min_target_std = ac_cfg.get("min_target_std", 0.05)
    actor.max_target_std = ac_cfg.get("max_target_std", 0.35)
    actor.hold_state_scale = ac_cfg.get("hold_state_scale", 64.0)
    actor.trade_state_eps = ac_cfg.get("trade_state_eps", 1e-6)
    actor.infer_quantize_step = ac_cfg.get("infer_quantize_step", 0.0)
    actor.use_residual_controller = bool(ac_cfg.get("residual_controller", False))
    actor.use_dual_residual_controller = bool(ac_cfg.get("use_dual_residual_controller", False))
    actor.use_regime_mode_gate_bias = bool(ac_cfg.get("use_regime_mode_gate_bias", False))
    actor.regime_mode_gate_scale = float(ac_cfg.get("regime_mode_gate_scale", 1.0))
    actor.separate_execution_head = bool(ac_cfg.get("separate_execution_head", False))
    actor.use_regime_target_bias = bool(ac_cfg.get("use_regime_target_bias", False))
    actor.regime_target_bias_scale = float(ac_cfg.get("regime_target_bias_scale", 1.0))
    actor.use_dual_regime_target_bias = bool(ac_cfg.get("use_dual_regime_target_bias", False))
    actor.dual_regime_target_bias_scale = float(ac_cfg.get("dual_regime_target_bias_scale", 1.0))
    actor.use_regime_trade_bias = bool(ac_cfg.get("use_regime_trade_bias", False))
    actor.regime_trade_bias_scale = float(ac_cfg.get("regime_trade_bias_scale", 1.0))
    actor.use_regime_execution_bias = bool(ac_cfg.get("use_regime_execution_bias", False))
    actor.regime_execution_bias_scale = float(ac_cfg.get("regime_execution_bias_scale", 1.0))
    actor.use_regime_residual_shift = bool(ac_cfg.get("use_regime_residual_shift", False))
    actor.regime_residual_shift_scale = float(ac_cfg.get("regime_residual_shift_scale", 0.0))
    actor.use_regime_band_bias = bool(ac_cfg.get("use_regime_band_bias", False))
    actor.regime_band_bias_scale = float(ac_cfg.get("regime_band_bias_scale", 0.0))
    actor.residual_min_overlay = ac_cfg.get(
        "residual_min_overlay",
        ac_cfg.get("abs_min_position", -1.0) - benchmark_position,
    )
    actor.residual_max_overlay = ac_cfg.get("residual_max_overlay", 0.0)
    actor.regime_overlay_caps = ac_cfg.get("regime_overlay_caps")
    actor.infer_bootstrap_target_prob = ac_cfg.get("infer_bootstrap_target_prob", 0.0)
    actor.infer_bootstrap_target_std = ac_cfg.get("infer_bootstrap_target_std", 0.0)
    actor.infer_bootstrap_trade_signal = ac_cfg.get("infer_bootstrap_trade_signal", 0.0)
    actor.infer_bootstrap_baseline_margin = ac_cfg.get("infer_bootstrap_baseline_margin", 0.0)
    actor.infer_regime_active_threshold = ac_cfg.get("infer_regime_active_threshold", 0.0)
    actor.infer_regime_active_state = ac_cfg.get("infer_regime_active_state", 0)
    actor.infer_active_std_max = ac_cfg.get("infer_active_std_max", 0.0)
    actor.infer_active_zscore_min = ac_cfg.get("infer_active_zscore_min", 0.0)
    actor.infer_event_entry_gap = ac_cfg.get("infer_event_entry_gap", 0.0)
    actor.infer_event_exit_gap = ac_cfg.get("infer_event_exit_gap", 0.0)
    actor.infer_event_trade_prob = ac_cfg.get("infer_event_trade_prob", 0.0)
    actor.infer_event_target_overlay = ac_cfg.get("infer_event_target_overlay")
    actor.infer_event_min_hold_bars = ac_cfg.get("infer_event_min_hold_bars", 0.0)
    actor.infer_target_from_logits = ac_cfg.get("infer_target_from_logits", False)
    actor.infer_logits_target_blend = ac_cfg.get("infer_logits_target_blend", 1.0)
    actor.infer_direct_target_track = ac_cfg.get("infer_direct_target_track", False)
    actor.infer_direct_track_scale = ac_cfg.get("infer_direct_track_scale", 1.0)
    actor.infer_underweight_adjust_scale = ac_cfg.get("infer_underweight_adjust_scale", 1.0)
    actor.infer_support_min_count = ac_cfg.get("infer_support_min_count", 0.0)
    actor.infer_support_min_ratio = ac_cfg.get("infer_support_min_ratio", 0.0)
    actor.infer_min_trade_floor = ac_cfg.get("infer_min_trade_floor", 0.0)
    actor.infer_min_trade_gap = ac_cfg.get("infer_min_trade_gap", 0.0)
    actor.infer_min_trade_scale = ac_cfg.get("infer_min_trade_scale", 0.0)
    actor.support_transition_counts = None
    if actor.use_residual_controller:
        residual_min = float(actor.residual_min_overlay)
        residual_max = float(actor.residual_max_overlay)
        init_overlay = float(ac_cfg.get("residual_init_overlay", 0.0))
        if residual_max > residual_min + 1e-6:
            init_frac = np.clip((init_overlay - residual_min) / (residual_max - residual_min), 1e-4, 1.0 - 1e-4)
            residual_bias = float(np.log(init_frac / (1.0 - init_frac)))
            torch.nn.init.constant_(actor.residual_head.bias, residual_bias)
            torch.nn.init.zeros_(actor.residual_head.weight)
            if actor.use_dual_residual_controller:
                bench_overlay = 0.0
                bench_frac = np.clip((bench_overlay - residual_min) / (residual_max - residual_min), 1e-4, 1.0 - 1e-4)
                bench_bias = float(np.log(bench_frac / (1.0 - bench_frac)))
                torch.nn.init.constant_(actor.residual_head_a.bias, bench_bias)
                torch.nn.init.zeros_(actor.residual_head_a.weight)
                torch.nn.init.constant_(actor.residual_head_b.bias, residual_bias)
                torch.nn.init.zeros_(actor.residual_head_b.weight)
                torch.nn.init.constant_(actor.target_mode_gate.bias, -0.5)
                torch.nn.init.zeros_(actor.target_mode_gate.weight)
                if actor.regime_mode_gate_head is not None:
                    torch.nn.init.zeros_(actor.regime_mode_gate_head.weight)
    if bc_cfg.get("init_regime_mode_gate_from_teacher", False):
        _initialize_regime_mode_gate_from_teacher(
            actor=actor,
            oracle_positions=oracle_positions,
            train_regime_probs=train_regime_probs,
            benchmark_position=benchmark_position,
            bc_cfg=bc_cfg,
        )
    if bc_cfg.get("init_regime_residual_shift_from_teacher", False):
        _initialize_regime_residual_shift_from_teacher(
            actor=actor,
            oracle_positions=oracle_positions,
            train_regime_probs=train_regime_probs,
            benchmark_position=benchmark_position,
            bc_cfg=bc_cfg,
        )
    try:
        current_abs_positions = np.empty_like(oracle_positions)
        current_abs_positions[0] = benchmark_position
        if len(oracle_positions) > 1:
            current_abs_positions[1:] = oracle_positions[:-1]
        current_idx = actor.target_indices(torch.tensor(current_abs_positions, dtype=torch.float32)).cpu().numpy()
        next_idx = actor.target_indices(torch.tensor(oracle_positions, dtype=torch.float32)).cpu().numpy()
        if train_regime_probs is not None:
            regime_idx = np.argmax(train_regime_probs[:len(next_idx)], axis=1).astype(np.int64)
            n_regimes = train_regime_probs.shape[1]
        else:
            regime_idx = np.zeros(len(next_idx), dtype=np.int64)
            n_regimes = 1
        support_counts = np.zeros((n_regimes, len(oracle_action_values), len(oracle_action_values)), dtype=np.float32)
        np.add.at(support_counts, (regime_idx, current_idx, next_idx), 1.0)
        actor.support_transition_counts = support_counts
    except Exception as e:
        print(f"[SPIBB] support table skipped: {e}")

    bc_sample_quality = None
    bc_advantage_values = None
    bc_quality_mode = str(bc_cfg.get("sample_quality_mode", "none")).lower()
    if oracle_teacher_mode == "signal_aim" and bc_quality_mode != "none":
        signal_values = np.asarray(oracle_values, dtype=np.float32)
        raw_signal = np.asarray(outcome_edge if outcome_edge is not None else np.zeros(len(oracle_positions)), dtype=np.float32)
        if bc_quality_mode == "abs_signal":
            bc_sample_quality = np.abs(signal_values)
        elif bc_quality_mode == "underweight_edge":
            underweight_size = np.clip(benchmark_position - np.asarray(oracle_positions, dtype=np.float32), 0.0, None)
            negative_signal = np.clip(-raw_signal, 0.0, None)
            raw_edge = underweight_size * negative_signal
            positive_edge = raw_edge[raw_edge > 0.0]
            if positive_edge.size > 0:
                edge_quantile = float(np.clip(bc_cfg.get("sample_quality_quantile", 0.75), 0.0, 0.99))
                edge_floor = float(np.quantile(positive_edge, edge_quantile))
                edge_scale = float(np.quantile(positive_edge, 0.90)) - edge_floor
                edge_scale = max(edge_scale, 1e-6)
                bc_sample_quality = np.clip((raw_edge - edge_floor) / edge_scale, 0.0, bc_cfg.get("sample_quality_clip", 4.0))
            else:
                bc_sample_quality = np.zeros_like(raw_edge, dtype=np.float32)
        elif bc_quality_mode in {"outcome_edge", "outcome_edge_relabel"}:
            raw_edge = raw_signal
            positive_edge = raw_edge[raw_edge > 0.0]
            if positive_edge.size > 0:
                edge_quantile = float(np.clip(bc_cfg.get("sample_quality_quantile", 0.75), 0.0, 0.99))
                edge_floor = float(np.quantile(positive_edge, edge_quantile))
                edge_scale = float(np.quantile(positive_edge, 0.90)) - edge_floor
                edge_scale = max(edge_scale, 1e-6)
                bc_sample_quality = np.clip((raw_edge - edge_floor) / edge_scale, 0.0, bc_cfg.get("sample_quality_clip", 4.0))
            else:
                bc_sample_quality = np.zeros_like(raw_edge, dtype=np.float32)
    if ac_cfg.get("advantage_conditioned", False):
        raw_adv = outcome_edge if outcome_edge is not None else bc_sample_quality
        if raw_adv is None:
            raw_adv = np.zeros(len(oracle_positions), dtype=np.float32)
        raw_adv = np.asarray(raw_adv, dtype=np.float32)
        positive_adv = raw_adv[raw_adv > 0.0]
        if positive_adv.size > 0:
            adv_scale = float(np.quantile(positive_adv, 0.90))
            adv_scale = max(adv_scale, 1e-6)
            bc_advantage_values = np.clip(raw_adv / adv_scale, 0.0, 1.0).astype(np.float32)
        else:
            bc_advantage_values = np.zeros_like(raw_adv, dtype=np.float32)

    return {
        "actor": actor,
        "bc_sample_quality": bc_sample_quality,
        "bc_advantage_values": bc_advantage_values,
    }
