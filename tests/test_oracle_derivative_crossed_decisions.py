import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_crossed_decisions import load_paired_forecasts, crossed_targets


class CrossedForecastContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name)/name for name in ("mean.npz", "risk.npz")]
        self.index = pd.date_range("2021-01-01", periods=32, freq="15min", tz="UTC")
        inference = np.zeros(32, bool); inference[[0, 24]] = True
        score = np.zeros(32, bool); score[0] = True
        actual = np.full((32, 3), np.nan); actual[0] = [.01, 0., .02]
        common = dict(timestamps=self.index.asi8, inference_mask=inference, score_support=score, actual=actual)
        self.mean = {**common, "mu": np.full(32, .01), "variance": np.full(32, .04)}
        self.risk = {**common, "mu": np.full(32, -.2), "variance": np.full(32, .0004)}

    def write_pair(self):
        np.savez(self.paths[0], **self.mean); np.savez(self.paths[1], **self.risk)

    def test_crosses_components_without_canceling_unscored_orders(self):
        self.write_pair()
        mu, variance, inference, score = load_paired_forecasts(*self.paths, expected_index=self.index)
        np.testing.assert_array_equal(mu[inference], [.01, .01])
        np.testing.assert_array_equal(variance[inference], [.0004, .0004])
        self.assertTrue(np.isnan(mu[~inference]).all())
        self.assertTrue(np.isnan(variance[~inference]).all())
        self.assertFalse(score[24]); self.assertTrue(np.isfinite(mu[24]))

    def test_future_labels_do_not_change_point_decisions(self):
        self.write_pair()
        def targets():
            mu, variance, _, _ = load_paired_forecasts(*self.paths, expected_index=self.index)
            return crossed_targets(None, mu, variance, {}, policy="point", cost_multiplier=2)[0]
        before = targets()
        self.mean["actual"][0] = [-.9, .95, .8]
        self.risk["actual"] = self.mean["actual"].copy()
        self.write_pair()
        np.testing.assert_allclose(targets(), before, equal_nan=True)

    def test_calendar_or_support_mismatch_is_rejected(self):
        for key in ("timestamps", "inference_mask", "score_support", "actual"):
            with self.subTest(key=key):
                original = self.risk[key]
                changed = original.copy()
                if key == "timestamps": changed[0] += pd.Timedelta(minutes=15).value
                elif key == "actual": changed[0, 0] += .01
                else: changed[0] = not changed[0]
                self.risk[key] = changed; self.write_pair()
                with self.assertRaises(ValueError):
                    load_paired_forecasts(*self.paths, expected_index=self.index)
                self.risk[key] = original


if __name__ == "__main__":
    unittest.main()
