"""Delayed execution, teacher/accounting identity and paired market inference."""
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_policy_ablation_evaluation import (
    ARMS, STAGES, accounting, teacher_target, bounded_requests, execute_positions,
    portfolio_step, score_path, evaluate_policy_ledger, paired_policy_inference,
    select_constant_exposure, volatility_target_requests,
)


def policy_fixture(days=16, seeds=(7, 19, 42)):
    n = days * 96
    time = pd.date_range("2025-04-01", periods=n, freq="15min", tz="UTC")
    returns = np.random.default_rng(15).normal(.00003, .001, n)
    rows = []
    for seed in seeds:
        for arm in ARMS:
            for stage in STAGES:
                target = 1 + .01 * (ord(arm) - ord("A")) + (.01 if stage == "rl" else 0)
                rows.append(pd.DataFrame({"fold": np.repeat([0, 1], n // 2), "seed": seed, "arm": arm,
                    "stage": stage, "timestamp": time, "decision_id": np.arange(n),
                    "bar_log_return": returns, "requested_position": target}))
    return pd.concat(rows, ignore_index=True), {"folds": [0, 1], "seeds": list(seeds),
        "statistics": {"bootstrap_replicates": 31, "block_days": [1, 7, 28], "seed": 17}}


class ExecutionTests(unittest.TestCase):
    def test_one_bar_delay_exact_cost_borrow_and_compounding(self):
        returns = np.array([.01, -.02, .03])
        request = np.array([1.08, 1., 1.08])
        ledger = execute_positions(returns, request)
        np.testing.assert_allclose(ledger.executed_position, [1., 1.08, 1.])
        cost = np.array([0., .08, .08]) * .00055
        borrow = np.array([0., .08, 0.]) * .10 / 35040
        expected = (1 - cost) * (1 + np.array([1, 1.08, 1]) * np.expm1(returns) - borrow)
        np.testing.assert_allclose(np.exp(ledger.net_log_return), expected, rtol=1e-14)
        np.testing.assert_allclose(ledger.equity, np.cumprod(expected), rtol=1e-14)

    def test_first_exposure_and_last_unexecuted_request_are_masked(self):
        first = execute_positions([.01, .02, -.01], [1.08, 1., 1.08])
        second = execute_positions([.01, .02, -.01], [1.08, 1., .92])
        np.testing.assert_array_equal(first.net_log_return, second.net_log_return)
        np.testing.assert_array_equal(first.execution_has_prior_request, [False, True, True])
        np.testing.assert_array_equal(first.request_will_execute, [True, True, False])
        self.assertAlmostEqual(first.requested_turnover_executable.sum(), first.turnover.sum())

    def test_gap_cancels_pending_request_and_resets_inventory_for_every_policy(self):
        time = pd.to_datetime(["2025-04-01T00:00Z", "2025-04-01T00:15Z", "2025-04-01T01:00Z"])
        ledger = execute_positions([.01, -.01, .02], [.92, .84, 1.08], time)
        np.testing.assert_allclose(ledger.executed_position, [1., .92, 1.])
        np.testing.assert_array_equal(ledger.reset, [True, False, True])
        np.testing.assert_allclose(ledger.cost_fraction, [0., .08 * .00055, 0.])
        np.testing.assert_array_equal(ledger.request_will_execute, [True, False, False])

    def test_buy_and_hold_has_zero_cost_borrow_and_exact_common_benchmark(self):
        returns = np.random.default_rng(9).normal(0, .01, 100)
        ledger = execute_positions(returns, np.ones(100))
        scores = score_path(ledger)
        np.testing.assert_allclose(ledger.net_log_return, returns, atol=3e-16)
        self.assertAlmostEqual(scores["alpha_excess"], 0, places=13)
        self.assertAlmostEqual(scores["sharpe_delta"], 0, places=12)
        self.assertAlmostEqual(scores["maxdd_delta"], 0, places=13)
        self.assertEqual(scores["turnover"], 0)
        self.assertEqual(scores["borrow_fraction_sum"], 0)

    def test_bounds_rate_causality_and_nonpositive_wealth_are_checked(self):
        with self.assertRaisesRegex(ValueError, "maximum step"):
            execute_positions([0., 0.], [.5, .5])
        with self.assertRaisesRegex(ValueError, "inventory bounds"):
            execute_positions([0.], [1.2])
        with self.assertRaisesRegex(ValueError, "non-positive"):
            portfolio_step(-100., 1.12, 1.12)
        first = execute_positions([.01, .02], [1.08, 1.])
        extended = execute_positions([.01, .02, -10.], [1.08, 1., 1.08])
        np.testing.assert_array_equal(first.net_log_return, extended.net_log_return.iloc[:2])

    def test_incremental_drawdown_reward_telescopes_within_episode(self):
        returns = np.array([.01, -.03, .005, -.04, .05])
        path = execute_positions(returns, np.ones(len(returns)))
        score = score_path(path)
        self.assertAlmostEqual(path.increment_max_drawdown.sum(), score["max_drawdown"])
        self.assertAlmostEqual(path.reward.sum(), -.25 * score["max_drawdown"])

    def test_projected_controls_obey_same_step_bound_and_only_pay_executable_changes(self):
        target = np.full(20, .5)
        requests = bounded_requests(target)
        np.testing.assert_allclose(requests[:3], [.92, .84, .76])
        ledger = execute_positions(np.zeros(20), requests)
        self.assertAlmostEqual(ledger.turnover.sum(), .5)
        self.assertAlmostEqual(ledger.cost_fraction.sum(), .5 * .00055)

    def test_borrow_is_applied_only_above_one_and_uses_fixed_annual_assumption(self):
        step = portfolio_step([0., 0., 0.], [.5, 1., 1.12], [.5, 1., 1.12])
        np.testing.assert_allclose(step["borrow_fraction"], [0., 0., .12 * .1 / 35040])
        self.assertEqual(accounting()["annual_bars"], 35040)


class TeacherAndControlTests(unittest.TestCase):
    def test_teacher_utility_matches_exact_constant_position_replay(self):
        per_bar = np.random.default_rng(23).normal(.001, .01, (4, 16))
        cumulative = np.cumsum(per_bar, axis=1)
        current = np.array([.5, .94, 1., 1.12])
        result = teacher_target(cumulative, current, return_details=True)
        for row, target in enumerate(result["target_position"]):
            previous = current[row]
            lognet = []
            for r in per_bar[row]:
                lognet.append(float(portfolio_step(r, target, previous)["net_log_return"]))
                previous = target
            equity = np.r_[1., np.exp(np.cumsum(lognet))]
            dd = np.max(1 - equity / np.maximum.accumulate(equity))
            utility = np.sum(lognet) - cumulative[row, -1] - .25 * dd
            self.assertAlmostEqual(result["utility"][row], utility, places=13)
            self.assertLessEqual(abs(target - current[row]), .08 + 1e-12)

    def test_cumulative_paths_are_differenced_not_treated_as_independent_bar_returns(self):
        bars = np.tile([.04, -.03], 8)
        path = np.cumsum(bars)
        details = teacher_target(path, 1., return_details=True)
        chosen = details["target_position"][0]
        expected = np.log1p(-.00055 * abs(chosen - 1)) + np.log(1 + chosen * np.expm1(bars) - max(chosen - 1, 0) * .1 / 35040).sum()
        self.assertNotAlmostEqual(expected, np.log1p(chosen * np.expm1(path[-1])), places=5)

    def test_float32_actor_boundaries_remain_legal_teacher_inputs(self):
        target = teacher_target(np.zeros((2, 16)), np.array([.5, 1.12], np.float32))
        self.assertTrue(np.isfinite(target).all())
        self.assertTrue((target >= .5).all() and (target <= 1.12).all())

    def test_constant_selection_has_no_Test_argument_and_records_Val_only(self):
        time = pd.date_range("2025-01-01", periods=100, freq="15min", tz="UTC")
        selected = select_constant_exposure(np.full(100, .001), time)
        self.assertEqual(selected["selection_split"], "validation_only")
        self.assertFalse(selected["test_used"])
        requests = volatility_target_requests(np.full(100, .0001), .008, time)
        self.assertTrue((requests >= .5).all() and (requests <= 1.12).all())


class PolicyInferenceTests(unittest.TestCase):
    def test_complete_fold_seed_stage_grid_and_nonduplicated_market_sample_count(self):
        frame, cfg = policy_fixture()
        executed, folds, pooled, aggregate = evaluate_policy_ledger(frame, cfg)
        self.assertEqual(len(folds), 2 * 3 * 4 * 2)
        self.assertEqual(len(pooled), 24)
        self.assertEqual(len(aggregate), 8)
        inference = paired_policy_inference(executed, cfg)
        self.assertEqual(len(inference), 36)
        self.assertEqual(inference.primary_inference.sum(), 12)
        self.assertEqual(set(inference.n_unique_market_rows), {16 * 96})
        self.assertEqual(set(inference.n_training_seeds), {3})
        self.assertTrue(inference.loc[inference.primary_inference, "p_holm_12"].between(0, 1).all())

    def test_different_market_return_or_missing_condition_is_rejected(self):
        frame, cfg = policy_fixture(days=2)
        frame.loc[0, "bar_log_return"] += .01
        with self.assertRaisesRegex(ValueError, "exactly the same"):
            evaluate_policy_ledger(frame, cfg)
        frame, cfg = policy_fixture(days=2)
        frame = frame.loc[~(frame.arm.eq("D") & frame.stage.eq("rl"))]
        with self.assertRaisesRegex(ValueError, "every registered"):
            evaluate_policy_ledger(frame, cfg)

    def test_duplicate_identical_training_seeds_do_not_triple_market_precision(self):
        frame, cfg = policy_fixture()
        executed, *_ = evaluate_policy_ledger(frame, cfg)
        multiple = paired_policy_inference(executed, cfg)
        cfg_single = {**cfg, "seeds": [7]}
        single = paired_policy_inference(executed.loc[executed.seed.eq(7)], cfg_single)
        np.testing.assert_allclose(multiple.ci_low, single.ci_low, atol=1e-13, equal_nan=True)
        np.testing.assert_allclose(multiple.ci_high, single.ci_high, atol=1e-13, equal_nan=True)

    def test_short_segments_remain_fixed_and_have_no_fictitious_seven_day_CI(self):
        frame, cfg = policy_fixture(days=2)
        executed, *_ = evaluate_policy_ledger(frame, cfg)
        rows = paired_policy_inference(executed, cfg)
        primary = rows.loc[rows.primary_inference]
        self.assertFalse(primary.inference_available.any())
        self.assertTrue(primary.p_value.isna().all())
        self.assertTrue(primary.p_holm_12.eq(1).all())


if __name__ == "__main__":
    unittest.main()
