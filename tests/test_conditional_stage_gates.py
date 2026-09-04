"""Conditional Oracle stage boundaries must not fall back to legacy labels."""
from __future__ import annotations

import unittest

from unidream.experiments.ac_stage import run_ac_stage
from unidream.experiments.chronological_oof import ConditionalPathBlocked
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.experiments.bc_stage import run_bc_stage


class ConditionalStageGateTest(unittest.TestCase):
    def test_world_model_stage_blocks_without_oof_teacher(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            prepare_world_model_stage(
                obs_dim=1,
                cfg={"conditional_oracle_path": True},
                device="cpu",
                wm_path="unused",
                wfo_dataset=None,
                oracle_positions=None,
                val_oracle_positions=None,
                train_returns=None,
                log_ts=lambda: "test",
            )

    def test_bc_stage_blocks_legacy_oracle_positions(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            run_bc_stage(
                actor=None,
                ensemble=None,
                bc_cfg={},
                oracle_cfg={},
                ac_cfg={"conditional_oracle_path": True},
                reward_cfg={},
                device="cpu",
                bc_path="unused",
                z_train=None,
                h_train=None,
                oracle_positions=None,
                train_regime_probs=None,
                oracle_soft_labels=None,
                bc_sample_quality=None,
                bc_advantage_values=None,
                log_ts=lambda: "test",
            )

    def test_ac_stage_cannot_bypass_oof_gate(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            run_ac_stage(
                actor=None,
                ensemble=None,
                cfg={"conditional_oracle_path": True},
                ac_cfg={},
                wm_cfg={},
                costs_cfg={},
                device="cpu",
                ac_path="unused",
                z_train=None,
                h_train=None,
                oracle_positions=None,
                train_regime_probs=None,
                train_advantage_values=None,
                wfo_dataset=None,
                wm_trainer=None,
                seq_len=1,
                val_regime_probs=None,
                val_advantage_values=None,
                val_oracle_positions=None,
                ac_max_steps_cfg=0,
                log_ts=lambda: "test",
                backtest_cls=None,
                pnl_attribution_fn=None,
                action_stats_fn=None,
                format_action_stats_fn=None,
                ac_alerts_fn=None,
                benchmark_positions_fn=None,
                benchmark_position=1.0,
                policy_score_fn=None,
                sequence_dataset_cls=None,
            )


if __name__ == "__main__":
    unittest.main()
