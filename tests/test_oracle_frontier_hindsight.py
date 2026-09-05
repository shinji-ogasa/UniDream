import itertools
import math
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import metrics
from unidream.experiments.oracle_frontier_hindsight import INTENTS, hindsight_targets


CONTRACT = {"one_way_cost": 0.00055, "borrow_annual": 0.1,
            "max_step": 0.08, "deadband": 0.01}


def bars(prices):
    closes = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": np.r_[100.0, closes[:-1]], "close": closes,
                         "bar_available": True},
                        index=pd.date_range("2022-01-01", periods=len(closes),
                                            freq="15min", tz="UTC"))


class OracleFrontierHindsightTests(unittest.TestCase):
    def test_wide_beam_matches_independent_exhaustive_intent_replay(self):
        data = bars(100 * np.exp(.08 * np.sin(np.arange(50) / 7)))
        decisions = [0, 24, 48]
        for penalty in (0.0, 1.0):
            values = []
            for sequence in itertools.product(INTENTS, repeat=len(decisions)):
                target = np.full(len(data), np.nan)
                target[decisions] = sequence
                result = metrics(data, target, CONTRACT)
                values.append(math.log1p(result["total_return"]) - penalty * result["maxdd"])
            target, diagnostic = hindsight_targets(data, CONTRACT, beam_width=128,
                                                   risk_penalty=penalty)
            self.assertAlmostEqual(diagnostic["objective"], max(values), places=13)
            self.assertTrue(diagnostic["exhaustive_for_fixed_intent_set"])
            self.assertTrue(diagnostic["canonical_replay_verified"])
            self.assertTrue(set(np.flatnonzero(np.isfinite(target))).issubset(decisions))

    def test_scalar_accounting_replays_cost_borrow_drift_and_first_drawdown(self):
        data = bars(80 * np.exp(np.arange(75) * .013))
        targets, diagnostic = hindsight_targets(data, CONTRACT, beam_width=32)
        result = metrics(data, targets, CONTRACT)
        self.assertGreater(result["borrow_initial_equity_units"], 0)
        self.assertGreater(result["fees_initial_equity_units"], 0)
        self.assertAlmostEqual(result["maxdd"], .2)
        self.assertLess(diagnostic["accounting_max_absolute_difference"], 1e-12)
        self.assertEqual(diagnostic["metrics"], result)

    def test_future_dependence_is_explicit_and_changes_first_intent(self):
        up = bars(100 * np.exp(np.arange(24) * .01))
        down = bars(100 * np.exp(-np.arange(24) * .01))
        first, diagnostic = hindsight_targets(up, CONTRACT)
        second, _ = hindsight_targets(down, CONTRACT)
        self.assertEqual(first[0], 1.12)
        self.assertEqual(second[0], .5)
        self.assertTrue(diagnostic["future_information_used"])
        self.assertFalse(diagnostic["deployable"])
        self.assertFalse(diagnostic["teacher_use_allowed"])
        self.assertFalse(diagnostic["global_optimum_claimed"])

    def test_missing_close_on_fill_cannot_cancel_order_and_missing_open_skips_it(self):
        data = bars(100 * np.exp(-np.arange(24) * .01))
        data.loc[data.index[1], ["close", "bar_available"]] = [np.nan, False]
        target, diagnostic = hindsight_targets(data, CONTRACT)
        self.assertEqual(target[0], .5)
        self.assertEqual(diagnostic["metrics"]["trades"], 1)
        self.assertLess(diagnostic["accounting_max_absolute_difference"], 1e-12)
        data.loc[data.index[1], "open"] = np.nan
        _, missing = hindsight_targets(data, CONTRACT)
        self.assertEqual(missing["metrics"]["trades"], 0)

    def test_intervening_gap_keeps_inventory_and_accrues_borrow(self):
        data = bars(100 * np.exp(np.arange(74) * .01))
        data.loc[data.index[2:7], ["open", "close"]] = np.nan
        data.loc[data.index[2:7], "bar_available"] = False
        target, diagnostic = hindsight_targets(data, CONTRACT, risk_penalty=1)
        replay = metrics(data, target, CONTRACT)
        self.assertGreater(replay["borrow_initial_equity_units"], 0)
        self.assertLess(diagnostic["accounting_max_absolute_difference"], 1e-12)

    def test_hold_is_no_rebalance_and_terminal_decision_has_no_fill(self):
        data = bars([100, 100, 100, 100, 100])
        target, diagnostic = hindsight_targets(data, CONTRACT)
        self.assertTrue(np.isnan(target).all())
        self.assertEqual(diagnostic["metrics"]["turnover"], 0)
        single = data.iloc[:1]
        target, diagnostic = hindsight_targets(single, CONTRACT)
        self.assertTrue(np.isnan(target).all())
        self.assertEqual(diagnostic["decision_count"], 0)

    def test_timezone_representation_keeps_utc_decision_clock(self):
        data = bars(100 * np.exp(np.arange(50) * .01))
        first, diagnostic = hindsight_targets(data, CONTRACT)
        changed = data.tz_convert("Asia/Tokyo")
        second, other = hindsight_targets(changed, CONTRACT)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(diagnostic["metrics"], other["metrics"])

    def test_narrow_beam_is_labeled_and_never_worse_than_full_hold_objective(self):
        data = bars(100 * np.exp(.2 * np.sin(np.arange(150) / 15)))
        _, diagnostic = hindsight_targets(data, CONTRACT, beam_width=1, risk_penalty=1)
        hold = metrics(data, np.full(len(data), np.nan), CONTRACT)
        self.assertGreater(diagnostic["pruned_distinct_branches"], 0)
        self.assertFalse(diagnostic["exhaustive_for_fixed_intent_set"])
        self.assertGreaterEqual(diagnostic["objective"],
                                math.log1p(hold["total_return"]) - hold["maxdd"])

    def test_invalid_contract_grid_and_boundaries_fail_closed(self):
        data = bars(np.ones(30) * 100)
        for width in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                hindsight_targets(data, CONTRACT, beam_width=width)
        for penalty in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                hindsight_targets(data, CONTRACT, risk_penalty=penalty)
        with self.assertRaises(ValueError):
            hindsight_targets(data.drop(data.index[3]), CONTRACT)
        with self.assertRaises(ValueError):
            hindsight_targets(data.tz_localize(None), CONTRACT)
        with self.assertRaises(ValueError):
            hindsight_targets(data, {**CONTRACT, "one_way_cost": np.nan})
        data.iloc[0, data.columns.get_loc("open")] = np.nan
        with self.assertRaises(ValueError):
            hindsight_targets(data, CONTRACT)


if __name__ == "__main__":
    unittest.main()
