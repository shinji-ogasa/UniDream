from __future__ import annotations

import unittest

import numpy as np

from unidream.experiments.chronological_oof import ConditionalPathBlocked
from unidream.experiments.fold_inputs import (
    TeacherInventoryContractError,
    current_inventory_from_replay,
    prepare_fold_inputs,
    validate_current_inventory_source,
)
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle


class TeacherInventoryContractTest(unittest.TestCase):
    def test_teacher_and_hindsight_inventory_sources_are_rejected(self) -> None:
        for source in ("hindsight_teacher", "teacher", "oracle", "signal_aim"):
            with self.subTest(source=source):
                with self.assertRaises(TeacherInventoryContractError):
                    validate_current_inventory_source(source)
        with self.assertRaises(TeacherInventoryContractError):
            validate_current_inventory_source("policy_replay", provenance={"producer": "hindsight_oracle"})

    def test_allowed_replay_sources_are_explicit_and_shifted(self) -> None:
        path = np.asarray([0.8, 1.0, 0.6], dtype=np.float32)
        current = current_inventory_from_replay(
            source="policy_replay",
            positions=path,
            benchmark_position=1.0,
            initial_position=1.0,
        )
        np.testing.assert_allclose(current, [1.0, 0.8, 1.0])
        benchmark = current_inventory_from_replay(
            source="benchmark_replay",
            length=3,
            benchmark_position=1.0,
        )
        np.testing.assert_array_equal(benchmark, [1.0, 1.0, 1.0])
        with self.assertRaises(TeacherInventoryContractError):
            current_inventory_from_replay(
                source="benchmark_replay",
                length=3,
                positions=path,
                benchmark_position=1.0,
            )
        with self.assertRaises(TeacherInventoryContractError):
            current_inventory_from_replay(
                source="policy_replay",
                positions=path,
                benchmark_position=1.0,
            )

    def test_conditional_predictive_state_does_not_fall_back_to_legacy_state(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((2, 1), dtype=np.float32),
                h_train=np.zeros((2, 1), dtype=np.float32),
                seq_len=2,
                ac_cfg={
                    "conditional_oracle_path": True,
                    "use_wm_predictive_state": True,
                },
                log_ts=lambda: "00:00:00",
            )

    def test_conditional_fold_input_builder_blocks_before_hindsight_dp(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            prepare_fold_inputs(
                wfo_dataset=None,
                cfg={"conditional_oracle_path": True},
                costs_cfg={},
                ac_cfg={},
                bc_cfg={},
                reward_cfg={},
                action_stats_fn=None,
                format_action_stats_fn=None,
                benchmark_position=1.0,
                forward_window_stats_fn=None,
                log_ts=lambda: "00:00:00",
            )


if __name__ == "__main__":
    unittest.main()
