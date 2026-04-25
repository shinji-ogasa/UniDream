import numpy as np

from unidream.data.oracle import ACTIONS as DEFAULT_ACTIONS
from unidream.data.oracle import feature_stress_teacher
from unidream.data.oracle import smooth_aim_positions

from .oracle_post import apply_oracle_postprocess
from .oracle_stage import compute_base_oracle
from .oracle_teacher import compute_teacher_oracle
from .regime_runtime import fit_fold_regimes
from .transition_advantage import (
    compute_transition_advantage,
    config_from_dict as transition_advantage_config_from_dict,
    current_positions_from_path,
    summarize_transition_advantage,
)


def _normalized_feature_stress_signal(
    *,
    train_features,
    fold_features,
    feature_columns,
    oracle_cfg: dict,
    benchmark_position: float,
    abs_min_position: float,
    abs_max_position: float,
    train_signal_stats: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    _positions_train, train_signal = feature_stress_teacher(
        train_features,
        feature_columns=feature_columns,
        benchmark_position=benchmark_position,
        min_position=oracle_cfg.get("stress_floor_position", abs_min_position),
        max_position=oracle_cfg.get("stress_ceiling_position", abs_max_position),
        fast_vol_col=oracle_cfg.get("stress_fast_vol_col", "rv_16"),
        slow_vol_col=oracle_cfg.get("stress_slow_vol_col", "rv_96"),
        shock_col=oracle_cfg.get("stress_shock_col", "atr_norm_ret"),
        drift_col=oracle_cfg.get("stress_drift_col", "open_ret"),
        macd_col=oracle_cfg.get("stress_macd_col", "macd"),
        macd_signal_col=oracle_cfg.get("stress_macd_signal_col", "macd_signal"),
        funding_col=oracle_cfg.get("stress_funding_col", "funding_rate"),
        fast_vol_threshold=oracle_cfg.get("stress_fast_vol_threshold", 0.8),
        slow_vol_threshold=oracle_cfg.get("stress_slow_vol_threshold", 0.8),
        shock_threshold=oracle_cfg.get("stress_shock_threshold", 0.8),
        drift_threshold=oracle_cfg.get("stress_drift_threshold", 0.2),
        trend_threshold=oracle_cfg.get("stress_trend_threshold", 0.1),
        funding_threshold=oracle_cfg.get("stress_funding_threshold", -0.5),
        fast_vol_weight=oracle_cfg.get("stress_fast_vol_weight", 0.45),
        slow_vol_weight=oracle_cfg.get("stress_slow_vol_weight", 0.25),
        shock_weight=oracle_cfg.get("stress_shock_weight", 0.20),
        drift_weight=oracle_cfg.get("stress_drift_weight", 0.10),
        trend_weight=oracle_cfg.get("stress_trend_weight", 0.15),
        funding_weight=oracle_cfg.get("stress_funding_weight", 0.05),
        entry_threshold=oracle_cfg.get("stress_entry_threshold", 0.15),
        signal_scale=oracle_cfg.get("stress_signal_scale", 1.0),
    )
    _positions_fold, fold_signal = feature_stress_teacher(
        fold_features,
        feature_columns=feature_columns,
        benchmark_position=benchmark_position,
        min_position=oracle_cfg.get("stress_floor_position", abs_min_position),
        max_position=oracle_cfg.get("stress_ceiling_position", abs_max_position),
        fast_vol_col=oracle_cfg.get("stress_fast_vol_col", "rv_16"),
        slow_vol_col=oracle_cfg.get("stress_slow_vol_col", "rv_96"),
        shock_col=oracle_cfg.get("stress_shock_col", "atr_norm_ret"),
        drift_col=oracle_cfg.get("stress_drift_col", "open_ret"),
        macd_col=oracle_cfg.get("stress_macd_col", "macd"),
        macd_signal_col=oracle_cfg.get("stress_macd_signal_col", "macd_signal"),
        funding_col=oracle_cfg.get("stress_funding_col", "funding_rate"),
        fast_vol_threshold=oracle_cfg.get("stress_fast_vol_threshold", 0.8),
        slow_vol_threshold=oracle_cfg.get("stress_slow_vol_threshold", 0.8),
        shock_threshold=oracle_cfg.get("stress_shock_threshold", 0.8),
        drift_threshold=oracle_cfg.get("stress_drift_threshold", 0.2),
        trend_threshold=oracle_cfg.get("stress_trend_threshold", 0.1),
        funding_threshold=oracle_cfg.get("stress_funding_threshold", -0.5),
        fast_vol_weight=oracle_cfg.get("stress_fast_vol_weight", 0.45),
        slow_vol_weight=oracle_cfg.get("stress_slow_vol_weight", 0.25),
        shock_weight=oracle_cfg.get("stress_shock_weight", 0.20),
        drift_weight=oracle_cfg.get("stress_drift_weight", 0.10),
        trend_weight=oracle_cfg.get("stress_trend_weight", 0.15),
        funding_weight=oracle_cfg.get("stress_funding_weight", 0.05),
        entry_threshold=oracle_cfg.get("stress_entry_threshold", 0.15),
        signal_scale=oracle_cfg.get("stress_signal_scale", 1.0),
    )
    if train_signal_stats is None:
        train_pos = train_signal[train_signal > 0.0]
        if train_pos.size > 0:
            center = float(np.quantile(train_pos, 0.50))
            scale = float(np.quantile(train_pos, 0.90) - center)
        else:
            center = 0.0
            scale = 1.0
        train_signal_stats = (center, max(scale, 1e-6))
    center, scale = train_signal_stats
    fold_stress = np.clip((np.asarray(fold_signal, dtype=np.float32) - center) / scale, 0.0, 1.0).astype(np.float32)
    return fold_stress, train_signal_stats


def _triangular_regime_probs(values: np.ndarray, centers: tuple[float, ...]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)[:, None]
    centers_arr = np.asarray(centers, dtype=np.float32)[None, :]
    if centers_arr.shape[1] <= 1:
        return np.ones((len(values), 1), dtype=np.float32)
    spacing = float(np.min(np.diff(np.asarray(centers, dtype=np.float32))))
    spacing = max(spacing, 1e-6)
    weights = np.clip(1.0 - np.abs(x - centers_arr) / spacing, 0.0, 1.0)
    denom = weights.sum(axis=1, keepdims=True)
    fallback = np.zeros_like(weights)
    fallback[:, int(np.argmin(np.abs(np.asarray(centers) - 0.5)))] = 1.0
    probs = np.where(denom > 1e-6, weights / np.clip(denom, 1e-6, None), fallback)
    return probs.astype(np.float32)


def _build_feature_stress_regimes(
    *,
    wfo_dataset,
    oracle_cfg: dict,
    benchmark_position: float,
    abs_min_position: float,
    abs_max_position: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    feature_columns = getattr(wfo_dataset, "feature_columns", [])
    train_signal, stress_stats = _normalized_feature_stress_signal(
        train_features=wfo_dataset.train_features,
        fold_features=wfo_dataset.train_features,
        feature_columns=feature_columns,
        oracle_cfg=oracle_cfg,
        benchmark_position=benchmark_position,
        abs_min_position=abs_min_position,
        abs_max_position=abs_max_position,
    )
    val_signal, stress_stats = _normalized_feature_stress_signal(
        train_features=wfo_dataset.train_features,
        fold_features=wfo_dataset.val_features,
        feature_columns=feature_columns,
        oracle_cfg=oracle_cfg,
        benchmark_position=benchmark_position,
        abs_min_position=abs_min_position,
        abs_max_position=abs_max_position,
        train_signal_stats=stress_stats,
    )
    test_signal, _ = _normalized_feature_stress_signal(
        train_features=wfo_dataset.train_features,
        fold_features=wfo_dataset.test_features,
        feature_columns=feature_columns,
        oracle_cfg=oracle_cfg,
        benchmark_position=benchmark_position,
        abs_min_position=abs_min_position,
        abs_max_position=abs_max_position,
        train_signal_stats=stress_stats,
    )
    centers = tuple(oracle_cfg.get("stress_regime_centers", [0.0, 0.5, 1.0]))
    train_probs = _triangular_regime_probs(train_signal, centers)
    val_probs = _triangular_regime_probs(val_signal, centers)
    test_probs = _triangular_regime_probs(test_signal, centers)
    return train_probs, val_probs, test_probs, int(train_probs.shape[1])


def _append_exogenous_stress_signal(
    regime_probs: np.ndarray | None,
    *,
    train_features,
    fold_features,
    feature_columns,
    oracle_cfg: dict,
    benchmark_position: float,
    abs_min_position: float,
    abs_max_position: float,
    train_signal_stats: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    fold_stress, train_signal_stats = _normalized_feature_stress_signal(
        train_features=train_features,
        fold_features=fold_features,
        feature_columns=feature_columns,
        oracle_cfg=oracle_cfg,
        benchmark_position=benchmark_position,
        abs_min_position=abs_min_position,
        abs_max_position=abs_max_position,
        train_signal_stats=train_signal_stats,
    )
    fold_stress = fold_stress[:, None]
    if regime_probs is None:
        augmented = fold_stress
    else:
        augmented = np.concatenate([regime_probs.astype(np.float32), fold_stress], axis=1)
    return augmented, train_signal_stats


def prepare_fold_inputs(
    *,
    wfo_dataset,
    cfg: dict,
    costs_cfg: dict,
    ac_cfg: dict,
    bc_cfg: dict,
    reward_cfg: dict,
    action_stats_fn,
    format_action_stats_fn,
    benchmark_position: float,
    forward_window_stats_fn,
    log_ts,
):
    print(f"\n[{log_ts()}] [Step 1] Hindsight Oracle DP...")
    train_returns = wfo_dataset.train_returns
    oracle_cfg = cfg.get("oracle", {})
    oracle_bundle = compute_base_oracle(
        train_returns=train_returns,
        val_returns=wfo_dataset.val_returns,
        oracle_cfg=oracle_cfg,
        reward_cfg=reward_cfg,
        costs_cfg=costs_cfg,
        default_action_values=cfg.get("actions", {}).get("values", DEFAULT_ACTIONS),
    )
    oracle_action_values = oracle_bundle["oracle_action_values"]
    oracle_values = oracle_bundle["oracle_values"]
    oracle_reward_mode = oracle_bundle["oracle_reward_mode"]
    oracle_benchmark_position = oracle_bundle["oracle_benchmark_position"]
    oracle_teacher_mode = oracle_bundle["oracle_teacher_mode"]
    oracle_actions = oracle_bundle["oracle_actions"]
    print(f"  Oracle computed: {len(oracle_actions)} steps, mean value={oracle_values.mean():.4f}")
    print(f"  Oracle objective: {oracle_reward_mode} (benchmark={oracle_benchmark_position:+.2f})")
    oracle_stats = action_stats_fn(oracle_action_values[oracle_actions], benchmark_position=benchmark_position)
    print(f"  Oracle dist: {format_action_stats_fn(oracle_stats)}")

    oracle_positions = oracle_bundle["oracle_positions"]
    val_oracle_positions = oracle_bundle["val_oracle_positions"]
    teacher_bundle = compute_teacher_oracle(
        teacher_mode=oracle_teacher_mode,
        base_oracle_positions=oracle_positions,
        base_val_oracle_positions=val_oracle_positions,
        base_oracle_values=oracle_values,
        train_returns=train_returns,
        val_returns=wfo_dataset.val_returns,
        train_features=wfo_dataset.train_features,
        val_features=wfo_dataset.val_features,
        feature_columns=getattr(wfo_dataset, "feature_columns", []),
        oracle_action_values=oracle_action_values,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        oracle_benchmark_position=oracle_benchmark_position,
    )
    oracle_positions = teacher_bundle["oracle_positions"]
    val_oracle_positions = teacher_bundle["val_oracle_positions"]
    if teacher_bundle["oracle_values"] is not None:
        oracle_values = teacher_bundle["oracle_values"]
    if teacher_bundle["teacher_message"]:
        print(teacher_bundle["teacher_message"])

    oracle_positions, val_oracle_positions, outcome_edge = apply_oracle_postprocess(
        oracle_positions=oracle_positions,
        val_oracle_positions=val_oracle_positions,
        oracle_action_values=oracle_action_values,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        oracle_reward_mode=oracle_reward_mode,
        oracle_benchmark_position=oracle_benchmark_position,
        oracle_teacher_mode=oracle_teacher_mode,
        train_returns=train_returns,
        forward_window_stats_fn=forward_window_stats_fn,
    )
    if bc_cfg.get("transition_advantage_relabel", False):
        ta_cfg = transition_advantage_config_from_dict(
            bc_cfg,
            costs_cfg=costs_cfg,
            benchmark_position=oracle_benchmark_position,
            default_actions=oracle_action_values,
        )
        current_train = current_positions_from_path(oracle_positions, oracle_benchmark_position)
        train_transition_bundle = compute_transition_advantage(train_returns, current_train, ta_cfg)
        oracle_positions = train_transition_bundle["target_positions"]
        if bc_cfg.get("transition_relabel_smooth", False):
            oracle_positions = smooth_aim_positions(
                oracle_positions,
                max_step=bc_cfg.get("transition_relabel_max_step", oracle_cfg.get("aim_max_step", 0.10)),
                band=bc_cfg.get("transition_relabel_band", oracle_cfg.get("aim_band", 0.02)),
                initial_position=oracle_benchmark_position,
                min_position=ac_cfg.get("abs_min_position", 0.0),
                max_position=ac_cfg.get("abs_max_position", 1.0),
                benchmark_position=oracle_benchmark_position,
                underweight_confirm_bars=bc_cfg.get("transition_relabel_underweight_confirm_bars", 8),
                underweight_min_scale=bc_cfg.get("transition_relabel_underweight_min_scale", 0.25),
                underweight_step_scale=bc_cfg.get("transition_relabel_underweight_step_scale", 0.5),
            ).astype(np.float32)
        outcome_edge = train_transition_bundle["best_advantage"]
        oracle_bundle["oracle_soft_labels"] = None

        current_val = current_positions_from_path(val_oracle_positions, oracle_benchmark_position)
        val_transition_bundle = compute_transition_advantage(wfo_dataset.val_returns, current_val, ta_cfg)
        val_oracle_positions = val_transition_bundle["target_positions"]
        if bc_cfg.get("transition_relabel_smooth", False):
            val_oracle_positions = smooth_aim_positions(
                val_oracle_positions,
                max_step=bc_cfg.get("transition_relabel_max_step", oracle_cfg.get("aim_max_step", 0.10)),
                band=bc_cfg.get("transition_relabel_band", oracle_cfg.get("aim_band", 0.02)),
                initial_position=oracle_benchmark_position,
                min_position=ac_cfg.get("abs_min_position", 0.0),
                max_position=ac_cfg.get("abs_max_position", 1.0),
                benchmark_position=oracle_benchmark_position,
                underweight_confirm_bars=bc_cfg.get("transition_relabel_underweight_confirm_bars", 8),
                underweight_min_scale=bc_cfg.get("transition_relabel_underweight_min_scale", 0.25),
                underweight_step_scale=bc_cfg.get("transition_relabel_underweight_step_scale", 0.5),
            ).astype(np.float32)

        transition_summary = summarize_transition_advantage(
            train_transition_bundle,
            current_train,
            oracle_benchmark_position,
        )
        dist = (
            f"short={transition_summary['target_short_rate']:.0%} "
            f"bench={transition_summary['target_benchmark_rate']:.0%} "
            f"ow={transition_summary['target_overweight_rate']:.0%}"
        )
        print(
            "  Transition relabel: "
            f"{dist} mean_adv={transition_summary['mean_best_advantage']:.6f} "
            f"recovery_rate={transition_summary['recovery_rate_from_underweight']:.1%}"
        )
    if oracle_cfg.get("use_aim_targets", False):
        aim_stats = action_stats_fn(oracle_positions, benchmark_position=benchmark_position)
        print(f"  Oracle aim dist: {format_action_stats_fn(aim_stats)}")
    if bc_cfg.get("outcome_relabel_bad_to_benchmark", False) and outcome_edge is not None:
        relabeled = int(np.sum(oracle_positions >= oracle_benchmark_position - 1e-6))
        print(f"  Outcome relabel active; benchmark-or-higher targets={relabeled}")

    n_states = cfg.get("eval", {}).get("hmm_n_states", 3)
    hmm_det = None
    regime_dim = 0
    train_regime_probs = None
    val_regime_probs = None
    test_regime_probs = None
    train_advantage_values = None
    val_advantage_values = None
    test_advantage_values = None
    regime_source = str(cfg.get("eval", {}).get("regime_source", "hmm")).lower()
    if regime_source == "feature_stress_tri":
        train_regime_probs, val_regime_probs, test_regime_probs, regime_dim = _build_feature_stress_regimes(
            wfo_dataset=wfo_dataset,
            oracle_cfg=oracle_cfg,
            benchmark_position=oracle_benchmark_position,
            abs_min_position=ac_cfg.get("abs_min_position", 0.0),
            abs_max_position=ac_cfg.get("abs_max_position", 1.0),
        )
        print(f"[Regime] Feature-stress tri regime built, regime_dim={regime_dim}")
    else:
        try:
            regime_bundle = fit_fold_regimes(
                train_returns=wfo_dataset.train_returns,
                val_returns=wfo_dataset.val_returns,
                test_returns=wfo_dataset.test_returns,
                n_states=n_states,
            )
            hmm_det = regime_bundle["detector"]
            regime_dim = regime_bundle["regime_dim"]
            train_regime_probs = regime_bundle["train_regime_probs"]
            val_regime_probs = regime_bundle["val_regime_probs"]
            test_regime_probs = regime_bundle["test_regime_probs"]
            print(f"[Regime] HMM fitted, regime_dim={regime_dim}")
        except Exception as e:
            print(f"[Regime] HMM skipped: {e}")

    if oracle_cfg.get("append_stress_regime_signal", False):
        feature_columns = getattr(wfo_dataset, "feature_columns", [])
        stress_stats = None
        train_regime_probs, stress_stats = _append_exogenous_stress_signal(
            train_regime_probs,
            train_features=wfo_dataset.train_features,
            fold_features=wfo_dataset.train_features,
            feature_columns=feature_columns,
            oracle_cfg=oracle_cfg,
            benchmark_position=oracle_benchmark_position,
            abs_min_position=ac_cfg.get("abs_min_position", 0.0),
            abs_max_position=ac_cfg.get("abs_max_position", 1.0),
        )
        val_regime_probs, stress_stats = _append_exogenous_stress_signal(
            val_regime_probs,
            train_features=wfo_dataset.train_features,
            fold_features=wfo_dataset.val_features,
            feature_columns=feature_columns,
            oracle_cfg=oracle_cfg,
            benchmark_position=oracle_benchmark_position,
            abs_min_position=ac_cfg.get("abs_min_position", 0.0),
            abs_max_position=ac_cfg.get("abs_max_position", 1.0),
            train_signal_stats=stress_stats,
        )
        test_regime_probs, _ = _append_exogenous_stress_signal(
            test_regime_probs,
            train_features=wfo_dataset.train_features,
            fold_features=wfo_dataset.test_features,
            feature_columns=feature_columns,
            oracle_cfg=oracle_cfg,
            benchmark_position=oracle_benchmark_position,
            abs_min_position=ac_cfg.get("abs_min_position", 0.0),
            abs_max_position=ac_cfg.get("abs_max_position", 1.0),
            train_signal_stats=stress_stats,
        )
        regime_dim = int(train_regime_probs.shape[1]) if train_regime_probs is not None else 1
        print(f"[Regime] Stress signal appended, regime_dim={regime_dim}")

    if ac_cfg.get("advantage_conditioned", False):
        advantage_source = str(ac_cfg.get("advantage_source", "default")).lower()
        if advantage_source == "feature_stress":
            feature_columns = getattr(wfo_dataset, "feature_columns", [])
            train_advantage_values, stress_stats = _normalized_feature_stress_signal(
                train_features=wfo_dataset.train_features,
                fold_features=wfo_dataset.train_features,
                feature_columns=feature_columns,
                oracle_cfg=oracle_cfg,
                benchmark_position=oracle_benchmark_position,
                abs_min_position=ac_cfg.get("abs_min_position", 0.0),
                abs_max_position=ac_cfg.get("abs_max_position", 1.0),
            )
            val_advantage_values, stress_stats = _normalized_feature_stress_signal(
                train_features=wfo_dataset.train_features,
                fold_features=wfo_dataset.val_features,
                feature_columns=feature_columns,
                oracle_cfg=oracle_cfg,
                benchmark_position=oracle_benchmark_position,
                abs_min_position=ac_cfg.get("abs_min_position", 0.0),
                abs_max_position=ac_cfg.get("abs_max_position", 1.0),
                train_signal_stats=stress_stats,
            )
            test_advantage_values, _ = _normalized_feature_stress_signal(
                train_features=wfo_dataset.train_features,
                fold_features=wfo_dataset.test_features,
                feature_columns=feature_columns,
                oracle_cfg=oracle_cfg,
                benchmark_position=oracle_benchmark_position,
                abs_min_position=ac_cfg.get("abs_min_position", 0.0),
                abs_max_position=ac_cfg.get("abs_max_position", 1.0),
                train_signal_stats=stress_stats,
            )
            print("[Advantage] Feature-stress signal prepared")

    oracle_bundle["oracle_values"] = oracle_values
    return {
        "train_returns": train_returns,
        "oracle_cfg": oracle_cfg,
        "oracle_bundle": oracle_bundle,
        "oracle_positions": oracle_positions,
        "val_oracle_positions": val_oracle_positions,
        "outcome_edge": outcome_edge,
        "hmm_det": hmm_det,
        "regime_dim": regime_dim,
        "train_regime_probs": train_regime_probs,
        "val_regime_probs": val_regime_probs,
        "test_regime_probs": test_regime_probs,
        "train_advantage_values": train_advantage_values,
        "val_advantage_values": val_advantage_values,
        "test_advantage_values": test_advantage_values,
    }
