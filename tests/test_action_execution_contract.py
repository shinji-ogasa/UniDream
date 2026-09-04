import dataclasses
import json
from pathlib import Path
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
    run_contract_backtest,
    select_block_decisions,
    transition_cost,
    _candidate_position,
)
from unidream.eval.backtest import ActionExecutionBacktest, Backtest
from unidream.eval.backtest import validate_bound_action_execution_contract
from unidream.experiments.transition_advantage import (
    compute_transition_advantage,
    compute_hindsight_transition_advantage,
    config_from_dict,
)


class ActionExecutionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ActionExecutionContract.canonical()

    @staticmethod
    def _all_masks(n_bars: int) -> tuple[np.ndarray, np.ndarray]:
        mask = np.ones(n_bars, dtype=bool)
        return mask.copy(), mask.copy()

    def test_contract_is_hashable_and_round_trips_without_legacy_defaults(self) -> None:
        self.assertAlmostEqual(self.contract.transition_cost_rate, 0.00055, places=15)
        self.assertEqual(hash(self.contract), hash(self.contract))
        self.assertEqual(
            self.contract.contract_hash,
            ActionExecutionContract.from_config(self.contract.to_dict()).contract_hash,
        )
        self.assertEqual(
            self.contract.contract_hash,
            "6f5beb7865fceac5ecbcfbb31dd11e8fdada02e1841fecac1c17e22377bb624f",
        )
        self.assertEqual(
            self.contract.to_dict()["feature_unavailable_policy"],
            "hold_and_score_commitment",
        )
        self.assertEqual(self.contract.to_dict()["outcome_unavailable_policy"], "exclude_block")
        self.assertEqual(self.contract.to_dict()["execution_skip_policy"], "hold_commitment")
        self.assertTrue(self.contract.to_dict()["eligibility_masks_required"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.contract.p_start = 0.5

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            configured_action_execution_contract(
                {
                    "use_action_execution_contract": True,
                    "costs": {"spread_bps": 5.0, "slippage_bps": 2.0, "fee_rate": 0.0004},
                }
            )
        with self.assertRaisesRegex(ValueError, "outcome_unavailable_policy"):
            ActionExecutionContract.from_config(
                {
                    key: value
                    for key, value in self.contract.to_dict().items()
                    if key != "outcome_unavailable_policy"
                }
            )

    def test_tracked_contract_artifact_round_trips_to_canonical_hash(self) -> None:
        artifact = Path(__file__).parents[1] / "docs" / "experiments" / "action_execution_contract.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        loaded = ActionExecutionContract.from_config(payload)
        self.assertEqual(loaded.contract_hash, self.contract.contract_hash)

    def test_contract_rejects_ambiguous_numeric_and_flag_config(self) -> None:
        for field in ("position_min", "spread_bps", "fee_rate", "p_start"):
            with self.subTest(field=field):
                config = self.contract.to_dict()
                config[field] = str(config[field])
                with self.assertRaisesRegex(ValueError, "finite real number"):
                    ActionExecutionContract.from_config(config)
        config = self.contract.to_dict()
        config["position_min"] = True
        with self.assertRaisesRegex(ValueError, "finite real number"):
            ActionExecutionContract.from_config(config)
        config = self.contract.to_dict()
        config["candidate_deltas"] = ["-0.08", -0.04, 0.0, 0.04, 0.08]
        with self.assertRaisesRegex(ValueError, "finite numeric sequence"):
            ActionExecutionContract.from_config(config)
        config = self.contract.to_dict()
        config["eligibility_masks_required"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            ActionExecutionContract.from_config(config)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            configured_action_execution_contract(
                {
                    "use_action_execution_contract": "false",
                    "action_execution_contract": self.contract.to_dict(),
                }
            )

    def test_contract_rejects_unknown_duplicate_alias_and_derived_overrides(self) -> None:
        unknown = self.contract.to_dict()
        unknown["future_override"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ActionExecutionContract.from_config(unknown)

        duplicate_alias = self.contract.to_dict()
        duplicate_alias["delay"] = duplicate_alias["execution_delay_bars"]
        with self.assertRaisesRegex(ValueError, "duplicate alias"):
            ActionExecutionContract.from_config(duplicate_alias)

        for field, value in (
            ("commitment_countdown_reset", 99),
            ("commitment_countdown_decrement", 2),
            ("spread_side", "full_transition"),
            ("transition_cost_rate", 0.0),
        ):
            with self.subTest(field=field):
                overridden = self.contract.to_dict()
                overridden[field] = value
                with self.assertRaisesRegex(ValueError, "derived field"):
                    ActionExecutionContract.from_config(overridden)

        with self.assertRaisesRegex(ValueError, "duplicate contract sections"):
            ActionExecutionContract.from_config(
                {
                    "action_execution_contract": self.contract.to_dict(),
                    "action_execution": self.contract.to_dict(),
                }
            )
        with self.assertRaisesRegex(ValueError, "duplicate contract sections"):
            ActionExecutionContract.from_config(
                {
                    "action_execution_contract": self.contract.to_dict(),
                    "conditional_oracle": {"enabled": True},
                }
            )

    def test_contract_rejects_invalid_cost_and_timing_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dataclasses.replace(self.contract, fee_rate=-0.001)
        with self.assertRaisesRegex(ValueError, "h_decision"):
            dataclasses.replace(self.contract, h_decision=8)
        with self.assertRaisesRegex(ValueError, "execution_delay_bars"):
            dataclasses.replace(self.contract, execution_delay_bars=0)
        with self.assertRaisesRegex(ValueError, "initial_countdown"):
            dataclasses.replace(self.contract, initial_countdown=5)
        with self.assertRaisesRegex(ValueError, "position bounds"):
            dataclasses.replace(self.contract, position_min=0.4)
        with self.assertRaisesRegex(ValueError, "canonical P0-C grid"):
            dataclasses.replace(self.contract, candidate_deltas=(-0.08, -0.04, 0.0, 0.04, 0.06))
        with self.assertRaisesRegex(ValueError, "p_start"):
            dataclasses.replace(self.contract, p_start=0.9)

    def test_candidate_grid_clips_then_deduplicates(self) -> None:
        self.assertEqual(self.contract.position_bounds, (0.5, 1.0))
        self.assertEqual(self.contract.min_position, 0.5)
        self.assertEqual(self.contract.max_position, 1.0)
        np.testing.assert_allclose(
            candidate_positions(0.52, self.contract),
            np.asarray([0.50, 0.52, 0.56, 0.60]),
        )
        np.testing.assert_allclose(
            candidate_positions(0.98, self.contract),
            np.asarray([0.90, 0.94, 0.98, 1.00]),
        )

    def test_absolute_position_canonicalises_clip_round12_before_replay(self) -> None:
        near_one = np.nextafter(1.0, 0.0)
        self.assertEqual(_candidate_position(self.contract, near_one, -0.08), 0.92)
        self.assertEqual(
            candidate_positions(near_one, self.contract).tolist(),
            [0.92, 0.96, 1.0],
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
        decision_eligible, score_eligible = self._all_masks(len(returns))
        trajectory = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

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
            [1.0, 0.92, 0.92, 0.92, 0.92, 0.96, 0.96, 0.96, 0.96, 0.96],
        )
        self.assertAlmostEqual(trajectory.transition_costs[1], 0.00055 * 0.08)
        self.assertAlmostEqual(trajectory.transition_costs[5], 0.00055 * 0.04)
        np.testing.assert_array_equal(
            trajectory.commitment_countdown,
            [4, 3, 2, 1, 4, 3, 2, 1, 0, 0],
        )
        self.assertEqual(trajectory.n_complete_blocks, 2)
        self.assertEqual(trajectory.n_scored_bars, 8)
        self.assertAlmostEqual(trajectory.effective_positions[-1], 0.96)
        with self.assertRaises(ValueError):
            trajectory.effective_positions[1] = 1.0
        with self.assertRaises(ValueError):
            trajectory.block_masks.common_mask[0] = False

    def test_noop_hold_is_scorable_but_not_a_filled_block(self) -> None:
        returns = np.zeros(8, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(len(returns))
        trajectory = replay_action_path(
            returns,
            np.zeros(len(returns), dtype=np.float64),
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertEqual(trajectory.n_complete_blocks, 1)
        self.assertEqual(trajectory.n_scorable_blocks, 1)
        self.assertEqual(trajectory.n_filled_blocks, 0)
        self.assertEqual(trajectory.n_execution_skipped_blocks, 0)
        self.assertEqual(trajectory.n_scored_bars, 4)
        self.assertFalse(trajectory.fill_mask[1])
        self.assertTrue(trajectory.decision_mask[0])

    def test_complete_starts_exclude_incomplete_final_block(self) -> None:
        self.assertEqual(complete_decision_starts(9, self.contract), (0, 4))
        self.assertEqual(complete_decision_starts(10, self.contract), (0, 4))
        self.assertEqual(complete_decision_starts(4, self.contract), ())
        with self.assertRaisesRegex(ValueError, "complete decision block"):
            decision_eligible, score_eligible = self._all_masks(4)
            ActionExecutionBacktest(
                np.ones(4),
                np.zeros(4),
                contract=self.contract,
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
            ).run()

    def test_backtest_uses_contract_cost_and_return_alignment(self) -> None:
        returns = np.zeros(6, dtype=np.float64)
        deltas = np.zeros(6, dtype=np.float64)
        deltas[0] = -0.08
        decision_eligible, score_eligible = self._all_masks(len(returns))
        with self.assertRaisesRegex(ValueError, "legacy override"):
            Backtest(
                returns,
                deltas,
                # Conflicting historical values must fail closed rather than
                # being silently ignored by the explicit contract path.
                spread_bps=99.0,
                fee_rate=0.9,
                slippage_bps=99.0,
                benchmark_positions=np.zeros(6),
                action_execution_contract=self.contract,
                action_positions_are_deltas=True,
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
                interval="1d",
            )
        with self.assertRaisesRegex(ValueError, "execution_delay_bars legacy override"):
            Backtest(
                returns,
                deltas,
                execution_delay_bars=0,
                benchmark_positions=np.zeros(6),
                action_execution_contract=self.contract,
                action_positions_are_deltas=True,
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
                interval="1d",
            )
        metrics = Backtest(
            returns,
            deltas,
            benchmark_positions=np.zeros(6),
            action_execution_contract=self.contract,
            action_positions_are_deltas=True,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
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
        decision_eligible, score_eligible = self._all_masks(len(returns))
        metrics = Backtest(
            returns,
            positions,
            benchmark_positions=np.ones(9, dtype=np.float64),
            action_execution_contract=self.contract,
            action_positions_are_deltas=False,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            interval="1d",
        ).run()
        self.assertEqual(metrics.scored_bars, 8)
        self.assertAlmostEqual(metrics.pnl_series[0], -0.00055 * 0.08)

    def test_contract_backtest_rejects_ambiguous_position_semantics(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_positions_are_deltas"):
            Backtest(
                np.zeros(8, dtype=np.float64),
                np.zeros(8, dtype=np.float64),
                action_execution_contract=self.contract,
            )

    def test_absolute_path_adapter_rejects_changes_inside_commitment(self) -> None:
        positions = np.asarray([0.92, 0.92, 0.92, 0.92, 0.96, 0.96, 0.96, 0.96, 0.96])
        decision_eligible, score_eligible = self._all_masks(len(positions))
        deltas = decision_deltas_from_positions(
            positions,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        np.testing.assert_allclose(deltas[[0, 4]], [-0.08, 0.04])
        trajectory = replay_contract_absolute_path(
            np.ones(len(positions)),
            positions,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertEqual(trajectory.n_scored_bars, 8)

        bad = positions.copy()
        bad[2] = 0.88
        with self.assertRaisesRegex(ValueError, "committed block"):
            decision_deltas_from_positions(
                bad,
                self.contract,
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
            )

    def test_u0_teacher_and_backtest_share_the_same_trajectory(self) -> None:
        returns = np.asarray(
            [0.0, 0.00125, 0.00125, 0.00125, 0.00125, -0.005, -0.005, -0.005, -0.005],
            dtype=np.float64,
        )
        decision_eligible, score_eligible = self._all_masks(len(returns))
        u0 = hindsight_upper_bound_path(
            returns,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        teacher = conditional_oracle_teacher_path(
            returns,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        # The local teacher is causal at t=0 (hold on the small positive
        # block), while hindsight U0 can pre-position for the next negative
        # block. They intentionally need not select the same action.
        self.assertEqual(teacher.decision_deltas[0], 0.0)
        self.assertAlmostEqual(u0.decision_deltas[0], -0.08)
        np.testing.assert_array_equal(u0.scored_mask, teacher.scored_mask)
        self.assertEqual(u0.contract_hash, self.contract.contract_hash)
        for trajectory in (u0, teacher):
            self.assertTrue(np.all(trajectory.effective_positions >= self.contract.position_min))
            self.assertTrue(np.all(trajectory.effective_positions <= self.contract.position_max))
            np.testing.assert_array_equal(trajectory.scored_mask, u0.scored_mask)

        metrics = ActionExecutionBacktest(
            teacher.returns,
            teacher.decision_deltas,
            contract=self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            interval="1d",
        ).run()
        np.testing.assert_allclose(metrics.pnl_series, teacher.scored_pnl)

    def test_causal_teacher_does_not_read_future_decision_block_scores(self) -> None:
        scores = np.zeros(9, dtype=np.float64)
        scores[0] = -0.01
        decision_eligible, score_eligible = self._all_masks(len(scores))
        baseline = conditional_oracle_teacher_path(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

        perturbed = scores.copy()
        perturbed[1:5] = 1_000_000.0
        changed = conditional_oracle_teacher_path(
            perturbed,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

        self.assertEqual(baseline.decision_deltas[0], changed.decision_deltas[0])
        self.assertEqual(baseline.decision_positions[0], changed.decision_positions[0])

        changed_start_score = scores.copy()
        changed_start_score[0] = 0.01
        score_changed = conditional_oracle_teacher_path(
            changed_start_score,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertNotEqual(baseline.decision_deltas[0], score_changed.decision_deltas[0])

        sparse = np.full(9, np.nan, dtype=np.float64)
        sparse[0] = scores[0]
        sparse[4] = 0.0
        sparse_teacher = conditional_oracle_teacher_path(
            sparse,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertEqual(sparse_teacher.decision_deltas[0], baseline.decision_deltas[0])

    def test_eligibility_masks_are_required_and_strict_boolean(self) -> None:
        n_bars = 9
        returns = np.zeros(n_bars, dtype=np.float64)
        deltas = np.zeros(n_bars, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(n_bars)

        with self.assertRaisesRegex(ValueError, "decision_eligible is required"):
            replay_action_path(returns, deltas, self.contract)
        with self.assertRaisesRegex(ValueError, "score_eligible is required"):
            replay_action_path(
                returns,
                deltas,
                self.contract,
                decision_eligible=decision_eligible,
            )

        invalid_masks = (
            np.ones(n_bars, dtype=np.int64),
            np.asarray(["true"] * n_bars),
            np.asarray([True, np.nan] + [True] * (n_bars - 2), dtype=object),
            np.ones(n_bars - 1, dtype=bool),
        )
        for invalid in invalid_masks:
            with self.subTest(mask=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "boolean|length"):
                    replay_action_path(
                        returns,
                        deltas,
                        self.contract,
                        decision_eligible=invalid,
                        score_eligible=score_eligible,
                    )

        with self.assertRaisesRegex(ValueError, "boolean"):
            replay_action_path(
                returns,
                deltas,
                self.contract,
                decision_eligible=decision_eligible,
                score_eligible=np.ones(n_bars, dtype=np.int64),
            )

    def test_outcome_gap_is_unscored_without_cancelling_execution(self) -> None:
        n_bars = 10
        returns = np.zeros(n_bars, dtype=np.float64)
        returns[1:5] = np.nan  # Must not be read for the excluded first block.
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[4] = -0.08
        decision_eligible, score_eligible = self._all_masks(n_bars)
        score_eligible[2] = False

        trajectory = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        np.testing.assert_array_equal(
            trajectory.scheduled_decision_mask,
            [True, False, False, False, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.eligible_decision_mask,
            [True, False, False, False, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.block_eligible_mask,
            [True, False, False, False, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.decision_mask,
            [True, False, False, False, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.fill_mask,
            [False, False, False, False, False, True, False, False, False, False],
        )
        np.testing.assert_array_equal(
            trajectory.scored_mask,
            [False, False, False, False, False, True, True, True, True, False],
        )
        np.testing.assert_allclose(
            trajectory.effective_positions,
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.92, 0.92, 0.92, 0.92],
        )
        self.assertEqual(trajectory.n_scheduled_decisions, 2)
        self.assertEqual(trajectory.n_eligible_decisions, 2)
        self.assertEqual(trajectory.n_eligible_blocks, 2)
        self.assertEqual(trajectory.n_fill_complete_blocks, 2)
        self.assertEqual(trajectory.n_excluded_blocks, 1)
        self.assertEqual(trajectory.n_scored_bars, 4)
        self.assertEqual(
            trajectory.eligibility_counts,
            {
                "scheduled_decisions": 2,
                "eligible_decisions": 2,
                "eligible_blocks": 2,
                "fill_complete_blocks": 2,
                "scorable_blocks": 1,
                "filled_blocks": 1,
                "execution_skipped_blocks": 0,
                "excluded_blocks": 1,
                "scored_bars": 4,
            },
        )
        self.assertEqual(len(trajectory.eligibility_mask_hash), 64)
        self.assertEqual(trajectory.mask_hash, trajectory.eligibility_mask_hash)
        self.assertAlmostEqual(trajectory.transition_costs[5], 0.00055 * 0.08)
        self.assertTrue(np.all(np.isfinite(trajectory.net_pnl)))
        self.assertEqual(len(trajectory.action_block_mask_hash), 64)
        self.assertEqual(
            set(trajectory.action_block_mask_hash_registry),
            {
                "origin_mask",
                "forecast_finite_mask",
                "bar_available",
                "returns_finite_mask",
                "scheduled_decision_mask",
                "decision_block_mask",
                "fill_complete_mask",
                "outcome_complete_mask",
                "executed_block_mask",
                "scored_action_mask",
                "common_mask",
                "utility_metric_mask",
                "action_metric_mask",
            },
        )
        self.assertNotEqual(
            trajectory.action_block_mask_hash,
            trajectory.eligibility_mask_hash,
        )

        executing_deltas = deltas.copy()
        executing_deltas[0] = -0.08
        executing = replay_action_path(
            returns,
            executing_deltas,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertAlmostEqual(executing.decision_deltas[0], -0.08)
        self.assertAlmostEqual(executing.effective_positions[1], 0.92)
        self.assertAlmostEqual(executing.effective_positions[5], 0.84)
        self.assertFalse(executing.scored_mask[1:5].any())

        absolute = np.ones(n_bars, dtype=np.float64)
        absolute[4:] = 0.92
        converted = decision_deltas_from_positions(
            absolute,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertAlmostEqual(converted[0], 0.0)
        self.assertAlmostEqual(converted[4], -0.08)
        executing_absolute = np.full(n_bars, 0.92, dtype=np.float64)
        executing_absolute[4:] = 0.84
        executing_converted = decision_deltas_from_positions(
            executing_absolute,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        np.testing.assert_allclose(executing_converted[[0, 4]], [-0.08, -0.08])

    def test_block_common_mask_is_metric_only_and_is_bound_in_full_hash(self) -> None:
        n_bars = 13
        returns = np.zeros(n_bars, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(n_bars)
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[0] = -0.08
        common = np.asarray([False, True, True], dtype=np.bool_)
        trajectory = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            common_mask=common,
        )
        # Common only gates metric masks; causal decision/fill/state remain
        # unchanged and the full registry records the supplied block mask.
        self.assertTrue(trajectory.decision_mask[0])
        self.assertTrue(trajectory.fill_mask[1])
        self.assertFalse(trajectory.block_masks.common_mask[0])
        self.assertTrue(trajectory.block_masks.common_mask[4])
        self.assertFalse(trajectory.block_masks.utility_metric_mask[0])
        self.assertTrue(trajectory.block_masks.utility_metric_mask[4])

    def test_future_outcome_availability_cannot_change_causal_selection(self) -> None:
        n_bars = 9
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[[0, 4]] = (-1.0, 1.0)
        decision_eligible, complete = self._all_masks(n_bars)
        outcome_gap = complete.copy()
        outcome_gap[2] = False  # fill t+1 exists; only a later outcome is absent

        complete_intent = select_block_decisions(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=complete,
        )
        gapped_intent = select_block_decisions(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=outcome_gap,
        )
        np.testing.assert_array_equal(gapped_intent, complete_intent)

        returns = np.zeros(n_bars, dtype=np.float64)
        returns[2] = np.nan
        complete_path = replay_action_path(
            np.nan_to_num(returns),
            complete_intent,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=complete,
        )
        gapped_path = replay_action_path(
            returns,
            gapped_intent,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=outcome_gap,
        )
        np.testing.assert_allclose(
            gapped_path.decision_deltas[[0, 4]],
            complete_path.decision_deltas[[0, 4]],
        )
        np.testing.assert_allclose(
            gapped_path.effective_positions,
            complete_path.effective_positions,
        )
        self.assertFalse(gapped_path.scored_mask[1:5].any())
        self.assertTrue(gapped_path.scored_mask[5:9].all())

    def test_fill_gap_preserves_intent_but_prevents_state_mutation(self) -> None:
        n_bars = 9
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[[0, 4]] = (-1.0, 1.0)
        decision_eligible, complete = self._all_masks(n_bars)
        fill_gap = complete.copy()
        fill_gap[1] = False

        complete_intent = select_block_decisions(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=complete,
        )
        gapped_intent = select_block_decisions(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=fill_gap,
        )
        self.assertEqual(gapped_intent[0], complete_intent[0])
        self.assertNotEqual(gapped_intent[0], 0.0)

        trajectory = replay_action_path(
            np.zeros(n_bars, dtype=np.float64),
            gapped_intent,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=fill_gap,
        )
        self.assertTrue(trajectory.decision_mask[0])
        self.assertFalse(trajectory.fill_block_eligible_mask[0])
        self.assertEqual(trajectory.intent_deltas[0], -0.08)
        self.assertEqual(trajectory.decision_deltas[0], 0.0)
        np.testing.assert_allclose(trajectory.effective_positions[1:5], 1.0)

    def test_absolute_path_adapter_accepts_intent_across_fill_gap(self) -> None:
        n_bars = 9
        returns = np.zeros(n_bars, dtype=np.float64)
        decision_eligible, complete = self._all_masks(n_bars)
        fill_gap = complete.copy()
        fill_gap[1] = False
        # This path records the selected target across the unfilled block.  It
        # is still a valid causal intent; replay must keep actual inventory at
        # p_start until a later block fills.
        intended_positions = np.full(n_bars, 0.92, dtype=np.float64)
        deltas = decision_deltas_from_positions(
            intended_positions,
            self.contract,
            decision_eligible=decision_eligible,
            bar_available=fill_gap,
        )
        self.assertAlmostEqual(deltas[0], -0.08)
        trajectory = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision_eligible,
            bar_available=fill_gap,
        )
        self.assertEqual(trajectory.decision_deltas[0], 0.0)
        np.testing.assert_allclose(trajectory.effective_positions[:5], 1.0)
        self.assertFalse(trajectory.fill_mask[:5].any())

    def test_absolute_path_adapter_preserves_explicit_forecast_gap(self) -> None:
        n_bars = 9
        returns = np.zeros(n_bars, dtype=np.float64)
        decision_eligible, available = self._all_masks(n_bars)
        forecast_finite = np.ones(n_bars, dtype=np.bool_)
        forecast_finite[4] = False
        positions = np.full(n_bars, 0.92, dtype=np.float64)
        deltas = decision_deltas_from_positions(
            positions,
            self.contract,
            decision_eligible=decision_eligible,
            bar_available=available,
            forecast_finite_mask=forecast_finite,
        )
        trajectory = replay_contract_absolute_path(
            returns,
            positions,
            self.contract,
            decision_eligible=decision_eligible,
            bar_available=available,
            forecast_finite_mask=forecast_finite,
        )
        self.assertAlmostEqual(deltas[0], -0.08)
        self.assertEqual(deltas[4], 0.0)
        self.assertFalse(trajectory.decision_mask[4])
        self.assertTrue(trajectory.execution_skipped_mask[4])
        np.testing.assert_allclose(trajectory.effective_positions[1:9], 0.92)

    def test_explicit_bar_available_alias_matches_legacy_spelling(self) -> None:
        n_bars = 9
        returns = np.zeros(n_bars, dtype=np.float64)
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[0] = -0.08
        decision, available = self._all_masks(n_bars)
        legacy = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision,
            score_eligible=available,
        )
        explicit = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision,
            bar_available=available,
        )
        self.assertEqual(
            legacy.action_block_mask_hash_registry,
            explicit.action_block_mask_hash_registry,
        )

    def test_decision_feature_gap_holds_commitment_but_keeps_finite_outcomes_scored(self) -> None:
        n_bars = 13
        returns = np.zeros(n_bars, dtype=np.float64)
        returns[1:5] = -0.01
        returns[5:9] = 0.02
        returns[9:13] = -0.01
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[0] = -0.08
        deltas[4] = 0.0  # execution skip is an explicit hold, never a fallback action
        decision_eligible, score_eligible = self._all_masks(n_bars)
        decision_eligible[4] = False

        trajectory = replay_action_path(
            returns,
            deltas,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

        np.testing.assert_array_equal(
            trajectory.scheduled_decision_mask[[0, 4, 8]], [True, True, True]
        )
        np.testing.assert_array_equal(
            trajectory.block_eligible_mask[[0, 4, 8]], [True, False, True]
        )
        np.testing.assert_array_equal(
            trajectory.score_block_eligible_mask[[0, 4, 8]], [True, True, True]
        )
        np.testing.assert_array_equal(
            trajectory.execution_skipped_mask[[0, 4, 8]], [False, True, False]
        )
        self.assertEqual(trajectory.n_scored_bars, 12)
        self.assertEqual(trajectory.n_scorable_blocks, 3)
        self.assertEqual(trajectory.n_complete_blocks, 3)
        self.assertEqual(trajectory.n_filled_blocks, 1)
        self.assertEqual(trajectory.n_execution_skipped_blocks, 1)
        self.assertEqual(trajectory.n_excluded_blocks, 0)
        np.testing.assert_allclose(trajectory.effective_positions[1:13], 0.92)
        np.testing.assert_allclose(trajectory.transition_costs[[1, 5, 9]], [0.00055 * 0.08, 0.0, 0.0])
        np.testing.assert_array_equal(trajectory.scored_mask[1:13], np.ones(12, dtype=bool))
        self.assertEqual(trajectory.commitment_countdown[4], 4)

    def test_strategy_and_benchmark_parity_includes_execution_skips(self) -> None:
        n_bars = 13
        returns = np.zeros(n_bars, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(n_bars)
        decision_eligible[4] = False
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[0] = -0.08
        metrics = ActionExecutionBacktest(
            returns,
            deltas,
            contract=self.contract,
            benchmark_decision_deltas=np.zeros(n_bars),
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        ).run()
        self.assertEqual(metrics.complete_blocks, 3)
        self.assertEqual(metrics.filled_blocks, 1)
        self.assertEqual(metrics.scorable_blocks, 3)
        self.assertEqual(metrics.execution_skipped_blocks, 1)
        self.assertEqual(metrics.excluded_blocks, 0)

    def test_cost_off_backtest_counts_position_changing_fills(self) -> None:
        artifact = Path(__file__).parents[1] / "docs" / "experiments" / "action_execution_contract_cost_off.json"
        cost_off = ActionExecutionContract.from_config(
            json.loads(artifact.read_text(encoding="utf-8")),
            require_canonical=False,
        )
        returns = np.zeros(5, dtype=np.float64)
        deltas = np.zeros(5, dtype=np.float64)
        deltas[0] = -0.08
        decision_eligible, score_eligible = self._all_masks(5)
        metrics = ActionExecutionBacktest(
            returns,
            deltas,
            contract=cost_off,
            benchmark_decision_deltas=np.zeros(5, dtype=np.float64),
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        ).run()
        self.assertEqual(metrics.n_trades, 1)
        self.assertEqual(metrics.filled_blocks, 1)

    def test_production_backtest_binds_contract_and_full_mask_registry(self) -> None:
        n_bars = 9
        returns = np.zeros(n_bars, dtype=np.float64)
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[0] = -0.08
        decision, available = self._all_masks(n_bars)
        common = np.ones(2, dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "externally pinned contract hash"):
            ActionExecutionBacktest(
                returns,
                deltas,
                contract=self.contract,
                decision_eligible=decision,
                bar_available=available,
                require_external_contract_hash=True,
            )
        with self.assertRaisesRegex(ValueError, "ambiguous legacy alias"):
            ActionExecutionBacktest(
                returns,
                deltas,
                contract=self.contract,
                decision_eligible=decision,
                score_eligible=available,
                expected_contract_hash=self.contract.contract_hash,
                require_external_contract_hash=True,
            )
        with self.assertRaisesRegex(ValueError, "explicit bar_available"):
            run_contract_backtest(
                Backtest,
                returns,
                deltas,
                benchmark_positions=np.zeros(n_bars),
                contract=self.contract,
                decision_eligible=decision,
                score_eligible=available,
                expected_contract_hash=self.contract.contract_hash,
                require_external_contract_hash=True,
            )
        with self.assertRaisesRegex(ValueError, "explicit paired common_mask"):
            ActionExecutionBacktest(
                returns,
                deltas,
                contract=self.contract,
                decision_eligible=decision,
                bar_available=available,
                expected_contract_hash=self.contract.contract_hash,
                require_external_contract_hash=True,
            )
        metrics = ActionExecutionBacktest(
            returns,
            deltas,
            contract=self.contract,
            decision_eligible=decision,
            bar_available=available,
            common_mask=common,
            expected_contract_hash=self.contract.contract_hash,
            require_external_contract_hash=True,
        ).run()
        self.assertEqual(metrics.action_execution_contract_hash, self.contract.contract_hash)
        self.assertEqual(len(metrics.action_block_mask_hash), 64)
        self.assertEqual(
            set(metrics.action_block_mask_hash_registry),
            {
                "origin_mask",
                "forecast_finite_mask",
                "bar_available",
                "returns_finite_mask",
                "scheduled_decision_mask",
                "decision_block_mask",
                "fill_complete_mask",
                "outcome_complete_mask",
                "executed_block_mask",
                "scored_action_mask",
                "common_mask",
                "utility_metric_mask",
                "action_metric_mask",
            },
        )
        altered = dataclasses.replace(self.contract, fee_rate=0.1)
        with self.assertRaisesRegex(ValueError, "does not match its external hash"):
            validate_bound_action_execution_contract(
                altered,
                expected_contract_hash=self.contract.contract_hash,
                require_external_hash=True,
            )

    def test_teacher_and_u0_share_skip_vs_exclusion_classification(self) -> None:
        n_bars = 13
        decision_eligible, score_eligible = self._all_masks(n_bars)
        decision_eligible[4] = False
        score_eligible[10] = False
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[0] = -0.01
        scores[8] = 0.01
        realized = np.zeros(n_bars, dtype=np.float64)
        realized[1:5] = -0.01
        realized[5:9] = 0.02
        realized[9:13] = np.nan

        teacher = conditional_oracle_teacher_path(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        u0 = hindsight_upper_bound_path(
            realized,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        np.testing.assert_array_equal(teacher.score_block_eligible_mask, u0.score_block_eligible_mask)
        np.testing.assert_array_equal(teacher.execution_skipped_mask, u0.execution_skipped_mask)
        np.testing.assert_array_equal(teacher.scored_mask, u0.scored_mask)
        self.assertEqual(teacher.n_execution_skipped_blocks, 1)
        self.assertEqual(teacher.n_excluded_blocks, 1)
        self.assertEqual(teacher.n_scored_bars, 8)

    def test_stage_contract_adapter_rejects_legacy_overrides(self) -> None:
        n_bars = 8
        returns = np.zeros(n_bars, dtype=np.float64)
        positions = np.ones(n_bars, dtype=np.float64)
        benchmark = np.ones(n_bars, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(n_bars)
        for key, value in {
            "spread_bps": self.contract.spread_bps,
            "fee_rate": self.contract.fee_rate,
            "slippage_bps": self.contract.slippage_bps,
            "execution_delay_bars": 0,
            "initial_position": self.contract.p_start,
            "benchmark_initial_position": self.contract.p_start,
        }.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, rf"rejects legacy overrides.*{key}"):
                    run_contract_backtest(
                        Backtest,
                        returns,
                        positions,
                        benchmark_positions=benchmark,
                        contract=self.contract,
                        decision_eligible=decision_eligible,
                        score_eligible=score_eligible,
                        **{key: value},
                    )

    def test_teacher_and_u0_apply_identical_masks_without_reading_gap_values(self) -> None:
        n_bars = 10
        decision_eligible, score_eligible = self._all_masks(n_bars)
        score_eligible[2] = False
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[0] = 0.0
        scores[4] = -0.01
        realized_returns = np.zeros(n_bars, dtype=np.float64)
        realized_returns[1:5] = np.nan
        realized_returns[5:9] = -0.01

        teacher = conditional_oracle_teacher_path(
            scores,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        u0 = hindsight_upper_bound_path(
            realized_returns,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        np.testing.assert_array_equal(teacher.scored_mask, u0.scored_mask)
        np.testing.assert_array_equal(teacher.block_eligible_mask, u0.block_eligible_mask)
        np.testing.assert_array_equal(teacher.scheduled_decision_mask, u0.scheduled_decision_mask)
        self.assertEqual(teacher.decision_deltas[0], 0.0)
        self.assertEqual(u0.decision_deltas[0], 0.0)
        self.assertNotEqual(teacher.decision_deltas[4], 0.0)
        self.assertNotEqual(u0.decision_deltas[4], 0.0)
        np.testing.assert_allclose(teacher.effective_positions[:5], 1.0)
        np.testing.assert_allclose(u0.effective_positions[:5], 1.0)

    def test_u0_persists_target_inventory_across_excluded_schedule(self) -> None:
        n_bars = 13
        returns = np.zeros(n_bars, dtype=np.float64)
        returns[1:5] = -0.01  # U0 should de-risk in the first block.
        returns[5:9] = np.nan  # Excluded block must not be read by U0.
        decision_eligible, score_eligible = self._all_masks(n_bars)
        score_eligible[6] = False

        u0 = hindsight_upper_bound_path(
            returns,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )

        np.testing.assert_array_equal(
            u0.scheduled_decision_mask[[0, 4, 8]], [True, True, True]
        )
        np.testing.assert_array_equal(
            u0.score_block_eligible_mask[[0, 4, 8]], [True, False, True]
        )
        np.testing.assert_array_equal(
            u0.block_eligible_mask[[0, 4, 8]], [True, True, True]
        )
        np.testing.assert_allclose(
            u0.decision_deltas[[0, 4, 8]], [-0.08, 0.0, 0.0]
        )
        np.testing.assert_allclose(u0.effective_positions[1:13], 0.92)
        np.testing.assert_allclose(u0.effective_positions[9:13], 0.92)
        self.assertEqual(u0.n_complete_blocks, 2)
        self.assertEqual(u0.n_filled_blocks, 1)
        self.assertEqual(u0.n_excluded_blocks, 1)
        self.assertEqual(u0.n_scored_bars, 8)
        self.assertAlmostEqual(u0.transition_costs[1], 0.00055 * 0.08)
        self.assertEqual(float(u0.transition_costs[5]), 0.0)
        self.assertEqual(float(u0.transition_costs[9]), 0.0)

    def test_strategy_and_benchmark_use_the_same_eligibility_window(self) -> None:
        n_bars = 10
        returns = np.zeros(n_bars, dtype=np.float64)
        deltas = np.zeros(n_bars, dtype=np.float64)
        deltas[4] = -0.08
        decision_eligible, score_eligible = self._all_masks(n_bars)
        score_eligible[2] = False

        direct = ActionExecutionBacktest(
            returns,
            deltas,
            contract=self.contract,
            benchmark_decision_deltas=np.zeros(n_bars),
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            interval="1d",
        ).run()
        wrapped = Backtest(
            returns,
            deltas,
            benchmark_positions=np.zeros(n_bars),
            action_execution_contract=self.contract,
            action_positions_are_deltas=True,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            interval="1d",
        ).run()
        self.assertEqual(direct.scheduled_decisions, 2)
        self.assertEqual(direct.eligible_blocks, 2)
        self.assertEqual(direct.excluded_blocks, 1)
        self.assertEqual(direct.scored_bars, 4)
        self.assertEqual(wrapped.to_dict(), direct.to_dict())
        np.testing.assert_allclose(wrapped.pnl_series, direct.pnl_series)

    def test_stage_contract_adapter_requires_and_forwards_masks(self) -> None:
        n_bars = 10
        returns = np.zeros(n_bars, dtype=np.float64)
        positions = np.ones(n_bars, dtype=np.float64)
        positions[4:] = 0.92
        benchmark = np.ones(n_bars, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(n_bars)
        score_eligible[2] = False

        with self.assertRaisesRegex(ValueError, "decision_eligible is required"):
            run_contract_backtest(
                Backtest,
                returns,
                positions,
                benchmark_positions=benchmark,
                contract=self.contract,
            )

        metrics = run_contract_backtest(
            Backtest,
            returns,
            positions,
            benchmark_positions=benchmark,
            contract=self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            interval="1d",
        ).run()
        self.assertEqual(metrics.scheduled_decisions, 2)
        self.assertEqual(metrics.eligible_blocks, 2)
        self.assertEqual(metrics.excluded_blocks, 1)
        self.assertEqual(metrics.scored_bars, 4)

    def test_contract_transition_advantage_is_explicit_hindsight_diagnostic(self) -> None:
        cfg = config_from_dict(
            {"action_execution_contract": self.contract.to_dict()},
            costs_cfg={},
            benchmark_position=1.0,
            default_actions=np.asarray([0.0]),
        )
        n_bars = 10
        returns = np.zeros(n_bars, dtype=np.float64)
        returns[1:5] = np.nan
        returns[5:9] = -0.01
        current = np.ones(n_bars, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(n_bars)
        score_eligible[2] = False
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            compute_transition_advantage(
                returns,
                current,
                cfg,
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
            )
        result = compute_hindsight_transition_advantage(
            returns,
            current,
            cfg,
            diagnostic_only=True,
            current_position_source="benchmark_replay",
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertEqual(result["decision_deltas"][0], 0.0)
        self.assertEqual(result["best_idx"][0], -1)
        self.assertAlmostEqual(result["target_positions"][0], 1.0)
        self.assertAlmostEqual(result["target_positions"][4], 0.92)
        self.assertIsNone(result["trajectory"])
        self.assertFalse(result["replayable"])
        self.assertTrue(result["future_derived"])
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["role"], "hindsight_transition_diagnostic")
        self.assertTrue(result["provenance"]["future_derived"])
        self.assertEqual(result["excluded_blocks"], 1)
        np.testing.assert_array_equal(result["score_block_eligible_mask"], [False, False, False, False, True, False, False, False, False, False])

    def test_hindsight_selector_is_iterative_for_long_windows(self) -> None:
        # More than Python's usual recursion limit in decision blocks: U0 must
        # remain a valid diagnostic without recursive stack growth.
        returns = np.zeros(1 + 4 * 1_100, dtype=np.float64)
        decision_eligible, score_eligible = self._all_masks(len(returns))
        trajectory = hindsight_upper_bound_path(
            returns,
            self.contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertEqual(trajectory.n_complete_blocks, 1_100)
        self.assertEqual(len(trajectory.decision_deltas), len(returns))

    def test_contract_transition_advantage_has_only_complete_decision_rows(self) -> None:
        cfg = config_from_dict(
            {"action_execution_contract": self.contract.to_dict()},
            costs_cfg={"spread_bps": 99.0, "fee_rate": 0.9, "slippage_bps": 99.0},
            benchmark_position=1.0,
            default_actions=np.asarray([0.0, 1.0]),
        )
        result = compute_hindsight_transition_advantage(
            np.ones(10, dtype=np.float64) * 0.001,
            np.ones(10, dtype=np.float64),
            cfg,
            diagnostic_only=True,
            current_position_source="benchmark_replay",
            decision_eligible=np.ones(10, dtype=bool),
            score_eligible=np.ones(10, dtype=bool),
        )
        self.assertEqual(result["action_execution_contract_hash"], self.contract.contract_hash)
        self.assertEqual(tuple(result["actions"]), self.contract.candidate_deltas)
        self.assertTrue(np.all(np.isnan(result["values"][1:4])))
        self.assertTrue(np.all(np.isnan(result["values"][5:])))
        self.assertIsNone(result["trajectory"])

    def test_hindsight_transition_diagnostic_uses_independent_current_rows(self) -> None:
        cfg = config_from_dict(
            {"action_execution_contract": self.contract.to_dict()},
            costs_cfg={},
            benchmark_position=1.0,
            default_actions=np.asarray([0.0]),
        )
        returns = np.asarray(
            [0.0, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.0],
            dtype=np.float64,
        )
        # The diagnostic evaluates each row from an independently supplied
        # benchmark inventory.  It must not require (or manufacture) a
        # hindsight-selected inventory path for the next row.
        current = np.ones_like(returns)
        decision_eligible, score_eligible = self._all_masks(len(returns))
        result = compute_hindsight_transition_advantage(
            returns,
            current,
            cfg,
            diagnostic_only=True,
            current_position_source="benchmark_replay",
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        self.assertEqual(result["decision_deltas"][0], -0.08)
        self.assertEqual(result["decision_deltas"][4], -0.08)
        self.assertAlmostEqual(result["current_positions"][4], 1.0)
        self.assertAlmostEqual(result["current_positions_at_decision"][4], 1.0)
        self.assertAlmostEqual(result["target_positions"][0], 0.92)
        self.assertAlmostEqual(result["target_positions"][4], 0.92)
        self.assertIsNone(result["trajectory"])

        with self.assertRaisesRegex(ValueError, "diagnostic_only=True"):
            compute_hindsight_transition_advantage(
                returns,
                current,
                cfg,
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
                diagnostic_only=False,
                current_position_source="benchmark_replay",
            )
        with self.assertRaisesRegex(ValueError, "benchmark_replay or causal_policy_replay"):
            compute_hindsight_transition_advantage(
                returns,
                current,
                cfg,
                diagnostic_only=True,
                current_position_source="hindsight_teacher",
                decision_eligible=decision_eligible,
                score_eligible=score_eligible,
            )

    def test_unsupported_semantics_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "funding"):
            dataclasses.replace(self.contract, funding_included=True)
        with self.assertRaisesRegex(ValueError, "partial"):
            dataclasses.replace(self.contract, partial_fill_policy="pro_rata")
        with self.assertRaisesRegex(ValueError, "additive_log_return"):
            dataclasses.replace(self.contract, return_unit="simple_return")


if __name__ == "__main__":
    unittest.main()
