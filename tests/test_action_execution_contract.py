import dataclasses
import unittest

import numpy as np

from unidream.data.oracle import (
    conditional_oracle_teacher_path,
    hindsight_upper_bound_path,
)
from unidream.eval.action_execution import (
    ActionExecutionContract,
    candidate_positions,
    complete_decision_starts,
    configured_action_execution_contract,
    decision_deltas_from_positions,
    replay_action_path,
    replay_contract_absolute_path,
    transition_cost,
)
from unidream.eval.backtest import ActionExecutionBacktest, Backtest
from unidream.experiments.transition_advantage import (
    compute_transition_advantage,
    config_from_dict,
)


class ActionExecutionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ActionExecutionContract.canonical()

    def test_contract_is_hashable_and_round_trips_without_legacy_defaults(self) -> None:
        self.assertAlmostEqual(self.contract.transition_cost_rate, 0.00055, places=15)
        self.assertEqual(hash(self.contract), hash(self.contract))
        self.assertEqual(
            self.contract.contract_hash,
            ActionExecutionContract.from_config(self.contract.to_dict()).contract_hash,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.contract.p_start = 0.5

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            configured_action_execution_contract(
                {
                    "use_action_execution_contract": True,
                    "costs": {"spread_bps": 5.0, "slippage_bps": 2.0, "fee_rate": 0.0004},
                }
            )

    def test_candidate_grid_clips_then_deduplicates(self) -> None:
        np.testing.assert_allclose(
            candidate_positions(0.52, self.contract),
            np.asarray([0.50, 0.52, 0.56, 0.60]),
        )
        np.testing.assert_allclose(
            candidate_positions(0.98, self.contract),
            np.asarray([0.90, 0.94, 0.98, 1.00]),
        )

    def test_cost_uses_full_spread_as_half_transition_plus_slippage_and_fee(self) -> None:
        self.assertAlmostEqual(
            transition_cost(1.0, 0.92, self.contract),
            0.00055 * 0.08,
            places=15,
        )
        self.assertEqual(self.contract.to_dict()["spread_convention"], "full_quoted")
        self.assertEqual(self.contract.to_dict()["spread_side"], "half_transition")
        self.assertFalse(self.contract.funding_included)

    def test_delay_fill_commitment_and_tail_mask_are_explicit(self) -> None:
        returns = np.ones(10, dtype=np.float64)
        deltas = np.zeros(10, dtype=np.float64)
        deltas[0] = -0.08
        deltas[1] = 0.08  # blocked-bar input must be ignored
        deltas[4] = 0.04
        trajectory = replay_action_path(returns, deltas, self.contract)

        np.testing.assert_array_equal(
            trajectory.decision_mask,
            [True, False, False, False, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.fill_mask,
            [False, True, False, False, False, True, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.scored_mask,
            [False, True, True, True, True, True, True, True, True, False],
        )
        np.testing.assert_allclose(
            trajectory.effective_positions,
            [1.0, 0.92, 0.92, 0.92, 0.92, 0.96, 0.96, 0.96, 0.96, 1.0],
        )
        self.assertAlmostEqual(trajectory.transition_costs[1], 0.00055 * 0.08)
        self.assertAlmostEqual(trajectory.transition_costs[5], 0.00055 * 0.04)
        np.testing.assert_array_equal(
            trajectory.commitment_countdown,
            [4, 3, 2, 1, 4, 3, 2, 1, 0, 0],
        )
        self.assertEqual(trajectory.n_complete_blocks, 2)
        self.assertEqual(trajectory.n_scored_bars, 8)

    def test_complete_starts_exclude_incomplete_final_block(self) -> None:
        self.assertEqual(complete_decision_starts(9, self.contract), (0, 4))
        self.assertEqual(complete_decision_starts(10, self.contract), (0, 4))
        self.assertEqual(complete_decision_starts(4, self.contract), ())
        with self.assertRaisesRegex(ValueError, "complete decision block"):
            ActionExecutionBacktest(
                np.ones(4),
                np.zeros(4),
                contract=self.contract,
            ).run()

    def test_backtest_uses_contract_cost_and_return_alignment(self) -> None:
        returns = np.zeros(6, dtype=np.float64)
        deltas = np.zeros(6, dtype=np.float64)
        deltas[0] = -0.08
        metrics = Backtest(
            returns,
            deltas,
            # These historical values must not override the explicit contract.
            spread_bps=99.0,
            fee_rate=0.9,
            slippage_bps=99.0,
            benchmark_positions=np.zeros(6),
            action_execution_contract=self.contract,
            interval="1d",
        ).run()
        self.assertEqual(len(metrics.pnl_series), 4)
        self.assertAlmostEqual(metrics.pnl_series[0], -0.00055 * 0.08)
        self.assertEqual(metrics.action_execution_contract_hash, self.contract.contract_hash)
        self.assertEqual(metrics.scored_bars, 4)
        self.assertEqual(metrics.complete_blocks, 1)

    def test_backtest_accepts_historical_absolute_position_shape_when_contract_is_opted_in(self) -> None:
        returns = np.zeros(9, dtype=np.float64)
        positions = np.asarray([0.92, 0.92, 0.92, 0.92, 0.96, 0.96, 0.96, 0.96, 0.96])
        metrics = Backtest(
            returns,
            positions,
            benchmark_positions=np.ones(9, dtype=np.float64),
            action_execution_contract=self.contract,
            interval="1d",
        ).run()
        self.assertEqual(metrics.scored_bars, 8)
        self.assertAlmostEqual(metrics.pnl_series[0], -0.00055 * 0.08)

    def test_absolute_path_adapter_rejects_changes_inside_commitment(self) -> None:
        positions = np.asarray([0.92, 0.92, 0.92, 0.92, 0.96, 0.96, 0.96, 0.96, 0.96])
        deltas = decision_deltas_from_positions(positions, self.contract)
        np.testing.assert_allclose(deltas[[0, 4]], [-0.08, 0.04])
        trajectory = replay_contract_absolute_path(np.ones(len(positions)), positions, self.contract)
        self.assertEqual(trajectory.n_scored_bars, 8)

        bad = positions.copy()
        bad[2] = 0.88
        with self.assertRaisesRegex(ValueError, "committed block"):
            decision_deltas_from_positions(bad, self.contract)

    def test_u0_teacher_and_backtest_share_the_same_trajectory(self) -> None:
        returns = np.asarray(
            [0.0, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01],
            dtype=np.float64,
        )
        u0 = hindsight_upper_bound_path(returns, self.contract)
        teacher = conditional_oracle_teacher_path(returns, self.contract)
        np.testing.assert_array_equal(u0.decision_deltas, teacher.decision_deltas)
        np.testing.assert_array_equal(u0.scored_mask, teacher.scored_mask)
        np.testing.assert_allclose(u0.effective_positions, teacher.effective_positions)
        self.assertEqual(u0.contract_hash, self.contract.contract_hash)

        metrics = ActionExecutionBacktest(
            returns,
            teacher.decision_deltas,
            contract=self.contract,
            interval="1d",
        ).run()
        np.testing.assert_allclose(metrics.pnl_series, teacher.scored_pnl)

    def test_contract_transition_advantage_has_only_complete_decision_rows(self) -> None:
        cfg = config_from_dict(
            {"action_execution_contract": self.contract.to_dict()},
            costs_cfg={"spread_bps": 99.0, "fee_rate": 0.9, "slippage_bps": 99.0},
            benchmark_position=1.0,
            default_actions=np.asarray([0.0, 1.0]),
        )
        result = compute_transition_advantage(
            np.ones(10, dtype=np.float64) * 0.001,
            np.ones(10, dtype=np.float64),
            cfg,
        )
        self.assertEqual(result["action_execution_contract_hash"], self.contract.contract_hash)
        self.assertEqual(tuple(result["actions"]), self.contract.candidate_deltas)
        self.assertTrue(np.all(np.isnan(result["values"][1:4])))
        self.assertTrue(np.all(np.isnan(result["values"][5:])))
        self.assertEqual(result["trajectory"].n_complete_blocks, 2)

    def test_unsupported_semantics_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "funding"):
            dataclasses.replace(self.contract, funding_included=True)
        with self.assertRaisesRegex(ValueError, "partial"):
            dataclasses.replace(self.contract, partial_fill_policy="pro_rata")
        with self.assertRaisesRegex(ValueError, "additive_log_return"):
            dataclasses.replace(self.contract, return_unit="simple_return")


if __name__ == "__main__":
    unittest.main()
