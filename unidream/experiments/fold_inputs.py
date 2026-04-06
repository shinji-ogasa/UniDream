import numpy as np

from unidream.data.oracle import ACTIONS as DEFAULT_ACTIONS

from .oracle_post import apply_oracle_postprocess
from .oracle_stage import compute_base_oracle
from .oracle_teacher import compute_teacher_oracle
from .regime_runtime import fit_fold_regimes


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
    }
