import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_frontier import (
    fit_mask, forecast_scores, map_outcomes, outcome_frame, summarize,
)


class OracleFrontierTests(unittest.TestCase):
    def bars(self):
        index = pd.date_range("2020-01-01", periods=8, freq="15min", tz="UTC")
        return pd.DataFrame({"open": [100, 102, 104, 101, 98, 101, 104, 105],
                             "close": [101, 103, 100, 99, 100, 103, 105, 106],
                             "bar_available": True}, index=index)

    def test_delayed_fill_and_future_path_by_hand(self):
        bars = self.bars()
        y = outcome_frame(bars, 2)
        self.assertAlmostEqual(y.iloc[0, 0], np.log(100 / 102))
        self.assertAlmostEqual(y.iloc[0, 1], -np.log(100 / 102))
        self.assertAlmostEqual(y.iloc[0, 2],
                               np.sqrt(np.log(103 / 102)**2 + np.log(100 / 103)**2))
        changed = bars.copy()
        changed.iloc[0, changed.columns.get_loc("close")] = 900
        np.testing.assert_allclose(outcome_frame(changed, 2).iloc[0], y.iloc[0])
        self.assertTrue(y.iloc[-2:].isna().all().all())

    def test_missing_target_bar_is_never_zero_filled(self):
        bars = self.bars()
        bars.iloc[2, bars.columns.get_loc("bar_available")] = False
        bars.iloc[2, bars.columns.get_loc("close")] = np.nan
        y = outcome_frame(bars, 2)
        self.assertTrue(y.iloc[:2].isna().all().all())
        self.assertTrue(y.iloc[3].notna().all())

    def test_training_purge_uses_outcome_close_time(self):
        index = pd.date_range("2020-01-01", periods=97, freq="15min", tz="UTC")
        mask = fit_mask(index, np.ones(len(index), bool), start=index[0],
                        end=index[49], horizon=24, cadence_hours=6)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0])
        # Origin 24's last outcome is bar 48 closing exactly at end=49.
        # Strictly-before-end training excludes it.

    def test_mapper_preserves_unavailable_and_exposure_bounds(self):
        pred = np.array([[.1, .05, .02], [-.1, .2, .03], [0, 0, 0], [np.nan, .1, .1]])
        ret = map_outcomes(pred, "return")
        risk = map_outcomes(pred, "downside")
        self.assertTrue(np.isnan(ret[-1]))
        self.assertTrue(np.all((ret[:3] >= .5) & (ret[:3] <= 1.12)))
        self.assertEqual(ret[2], 1.)
        self.assertTrue(np.all(risk[:3] <= ret[:3]))

    def test_forecast_baseline_skill_zero(self):
        y = np.tile(np.array([[.01, .02, .03], [-.02, .04, .05]]), (16, 1))
        mean = y.mean(axis=0)
        result = forecast_scores(y, np.tile(mean, (len(y), 1)), mean)
        self.assertIsNone(result["return_rank_ic"])
        for outcome in result["outcomes"].values():
            self.assertAlmostEqual(outcome["mse_skill"], 0.)

    def test_missing_regime_prevents_robust_pass(self):
        rows = [{"candidate_id": "a", "regime": {"trend": "bull"},
                 "base": {"alpha_ex": .03, "maxdd_delta": -.05},
                 "stress_2x": {"alpha_ex": .02, "maxdd_delta": -.04}} for _ in range(4)]
        result = summarize(rows, 3)["a"]
        self.assertTrue(result["direction_pass"])
        self.assertFalse(result["exploratory_regime_direction_pass"])
        self.assertFalse(result["high_probability_generalization_established"])


if __name__ == "__main__":
    unittest.main()
