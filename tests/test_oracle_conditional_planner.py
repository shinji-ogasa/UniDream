import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import _simulate, metrics
from unidream.experiments.oracle_conditional_planner import _choose, conditional_targets


CONTRACT = {"one_way_cost": .00055, "borrow_annual": .1, "max_step": .08, "deadband": .01}


def bars(n=145):
    return pd.DataFrame({"open": 100., "close": 100., "bar_available": True},
                        index=pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC"))


class OracleConditionalPlannerTests(unittest.TestCase):
    def test_score_projects_large_passive_drift_to_actual_max_step(self):
        target, score, turnover = _choose(.30, 1., .30, .02, 0., .00055, .1, .01, 0., 2.)
        self.assertEqual(target, .5)
        notional = .08/(1+.00055*.38)
        self.assertAlmostEqual(turnover, notional)
        self.assertAlmostEqual(score, .08*.02-2*.00055*notional)

    def test_zero_signal_and_zero_risk_hold_without_rebalancing(self):
        data = bars()
        target, diagnostic = conditional_targets(data, np.zeros(len(data)), np.zeros(len(data)),
                                                 CONTRACT, risk_aversion=0)
        self.assertTrue(np.isnan(target).all())
        self.assertEqual(diagnostic["metrics"]["trades"], 0)
        self.assertEqual(diagnostic["metrics"]["alpha_ex"], 0)
        self.assertEqual(diagnostic["hold_decision_count"], 7)

    def test_fee_threshold_rejects_weak_signal_and_accepts_strong_signal(self):
        data = bars(25)
        low, _ = conditional_targets(data, np.full(len(data), .0005), np.zeros(len(data)),
                                     CONTRACT, risk_aversion=0)
        high, _ = conditional_targets(data, np.full(len(data), .01), np.zeros(len(data)),
                                      CONTRACT, risk_aversion=0)
        self.assertTrue(np.isnan(low).all())
        self.assertAlmostEqual(high[0], 1.08)

    def test_score_uses_exact_fee_notional_and_borrow_penalty(self):
        target, score, turnover = _choose(1., 1., 1., .02, .0004, .00055, .1, .01, 1., 2.)
        self.assertAlmostEqual(target, 1.08)
        exact_turnover = .08 / (1 + .00055 * 1.08)
        expected = .08 * .02 - .5 * (1.08 ** 2 - 1) * .0004 - 2 * .00055 * exact_turnover - .08 * .1 * 24 / 35040
        self.assertAlmostEqual(turnover, exact_turnover)
        self.assertAlmostEqual(score, expected)

    def test_inventory_is_endogenous_and_decision_precedes_current_borrow(self):
        data = bars(73)
        signal = np.full(len(data), .02)
        signal[48:] = -.02
        target, diagnostic = conditional_targets(data, signal, np.zeros(len(data)), CONTRACT, risk_aversion=0)
        self.assertAlmostEqual(target[0], 1.08)
        self.assertAlmostEqual(target[24], 1.12)
        trace = diagnostic["decision_trace"]
        j = trace["bar_indices"].index(48)
        current = trace["known_open_exposure"][j]
        self.assertGreater(current, 1.12)  # Borrowing has changed actual inventory exposure.
        self.assertAlmostEqual(target[48], current - .08)
        # Borrowing on bar48 is still in the future of its decision.
        _, positions, *_ = _simulate(data.open.to_numpy(), data.close.to_numpy(), target,
                                    np.asarray(data.index.minute == 0), *CONTRACT.values())
        self.assertAlmostEqual(current, positions[47])
        self.assertGreater(positions[48], current)

    def test_current_close_and_future_prices_forecasts_cannot_change_current_target(self):
        data = bars()
        mu, variance = np.full(len(data), .02), np.full(len(data), .0001)
        before, _ = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=1)
        changed = data.copy()
        changed.loc[changed.index[48], "close"] = np.nan
        changed.loc[changed.index[48], "bar_available"] = False
        changed.loc[changed.index[49]:, ["open", "close"]] *= 1.5
        future_mu, future_var = mu.copy(), variance.copy()
        future_mu[49:], future_var[49:] = -.1, .01
        after, _ = conditional_targets(changed, future_mu, future_var, CONTRACT, risk_aversion=1)
        np.testing.assert_array_equal(before[:49], after[:49])
        truncated, _ = conditional_targets(data.iloc[:49], mu[:49], variance[:49], CONTRACT, risk_aversion=1)
        np.testing.assert_array_equal(before[:49], truncated)

    def test_unknown_next_open_cannot_cancel_current_intent_and_close_gap_does_not_cancel_fill(self):
        data = bars(73)
        mu = np.full(len(data), .02)
        variance = np.zeros(len(data))
        clean, clean_diag = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=0)
        missing_open = data.copy()
        missing_open.loc[missing_open.index[1], ["open", "close"]] = np.nan
        missing_open.loc[missing_open.index[1], "bar_available"] = False
        target, diagnostic = conditional_targets(missing_open, mu, variance, CONTRACT, risk_aversion=0)
        self.assertEqual(target[0], clean[0])
        self.assertAlmostEqual(target[24], 1.08)  # First intent did not fill.
        missing_close = data.copy()
        missing_close.loc[missing_close.index[1], "close"] = np.nan
        missing_close.loc[missing_close.index[1], "bar_available"] = False
        other, other_diag = conditional_targets(missing_close, mu, variance, CONTRACT, risk_aversion=0)
        self.assertEqual(other[0], clean[0])
        self.assertEqual(other_diag["metrics"]["trades"], clean_diag["metrics"]["trades"])
        self.assertGreater(diagnostic["metrics"]["borrow_initial_equity_units"], 0)
        self.assertTrue(diagnostic["canonical_replay_verified"])

    def test_actual_fill_step_and_gap_borrow_match_canonical_account(self):
        data = bars()
        mu, variance = np.full(len(data), .02), np.zeros(len(data))
        mu[:24] = -.02
        data.loc[data.index[25], "open"] = 50.
        data.loc[data.index[25], "close"] = np.nan
        data.loc[data.index[25], "bar_available"] = False
        data.loc[data.index[75:78], ["open", "close"]] = np.nan
        data.loc[data.index[75:78], "bar_available"] = False
        target, diagnostic = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=0)
        self.assertAlmostEqual(target[0], .92)
        self.assertAlmostEqual(target[24], 1.)
        self.assertGreater(diagnostic["metrics"]["borrow_initial_equity_units"], 0)
        self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)
        self.assertEqual(diagnostic["metrics"], metrics(data, target, CONTRACT))

    def test_drift_outside_bounds_may_hold_and_missing_known_inputs_skip_only_those_decisions(self):
        target, score, turnover = _choose(1.2, 1., 1.2, .1, 0., .00055, .1, .01, 0., 2.)
        self.assertTrue(np.isnan(target))
        self.assertEqual((score, turnover), (0., 0.))
        data = bars()
        mu, variance = np.full(len(data), .02), np.zeros(len(data))
        mu[24] = np.nan
        variance[48] = -.1
        data.loc[data.index[72], "open"] = np.nan
        data.loc[data.index[72], "bar_available"] = False
        target, diagnostic = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=0)
        self.assertTrue(np.isnan(target[[24, 48, 72]]).all())
        self.assertEqual(diagnostic["unavailable_forecast_decision_count"], 2)
        self.assertEqual(diagnostic["missing_open_decision_count"], 1)

    def test_utc_schedule_is_invariant_to_timezone_representation(self):
        data = bars()
        shifted = data.copy()
        shifted.index = shifted.index.tz_convert("Asia/Tokyo")
        mu, variance = np.full(len(data), .02), np.zeros(len(data))
        original, _ = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=0)
        converted, _ = conditional_targets(shifted, mu, variance, CONTRACT, risk_aversion=0)
        np.testing.assert_array_equal(original, converted)


if __name__ == "__main__":
    unittest.main()
