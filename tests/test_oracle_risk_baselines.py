import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import file_digest
from unidream.experiments.oracle_risk_baselines import (
    common_score_mask, loss_arrays, persistence_forecasts, summarize, verify_digest,
)


def bars(n=1000):
    i = np.arange(n)
    return pd.DataFrame({"close": 100 * np.exp(.001 * i + .02 * np.sin(i / 29))},
                        index=pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"))


class OracleRiskBaselinesTests(unittest.TestCase):
    def test_known_constant_return_scale_and_nominal_warmup(self):
        data = bars()
        data["close"] = 100 * np.exp(.001 * np.arange(len(data)))
        out = persistence_forecasts(data)
        for window in (24, 96, 672):
            name = f"persistence_w{window}"
            self.assertEqual(out[name].first_valid_index(), data.index[window + 1])
            np.testing.assert_allclose(out[name].iloc[window + 1:], np.sqrt(24) * .001, rtol=1e-10)

    def test_current_and_future_mutation_and_prefix_invariance(self):
        data = bars()
        boundary = 800
        changed = data.copy()
        changed.iloc[boundary:] *= 100
        changed.iloc[boundary + 1] = np.nan
        before = persistence_forecasts(data)
        pd.testing.assert_frame_equal(before.iloc[:boundary + 1], persistence_forecasts(changed).iloc[:boundary + 1])
        pd.testing.assert_frame_equal(before.iloc[:boundary + 1], persistence_forecasts(data.iloc[:boundary + 1]))

    def test_missing_timestamp_equals_missing_row_without_bridge(self):
        data = bars()
        boundary = 800
        missing = data.copy()
        missing.iloc[boundary] = np.nan
        out = persistence_forecasts(missing)
        pd.testing.assert_frame_equal(out, persistence_forecasts(data.drop(data.index[boundary])))
        self.assertTrue(np.isnan(out.persistence_w24.iloc[boundary + 1]))
        self.assertTrue(np.isnan(out.persistence_w24.iloc[boundary + 25]))
        self.assertTrue(np.isfinite(out.persistence_w24.iloc[boundary + 26]))

    def test_995_threshold_counts_returns_not_missing_prices(self):
        data = bars()
        one_gap = data.copy()
        one_gap.iloc[500] = np.nan  # Two unavailable returns: 670/672 passes.
        two_gaps = one_gap.copy()
        two_gaps.iloc[600] = np.nan  # Four unavailable returns: 668/672 fails.
        self.assertTrue(np.isfinite(persistence_forecasts(one_gap).persistence_w672.iloc[900]))
        self.assertTrue(np.isnan(persistence_forecasts(two_gaps).persistence_w672.iloc[900]))

    def test_measured_zero_and_invalid_price_are_distinct(self):
        data = bars()
        data["close"] = 100
        self.assertEqual(persistence_forecasts(data).persistence_w24.iloc[-1], 0)
        invalid = data.copy().astype(float)
        missing = invalid.copy()
        invalid.iloc[-2] = 0
        missing.iloc[-2] = np.nan
        pd.testing.assert_frame_equal(persistence_forecasts(invalid), persistence_forecasts(missing))
        self.assertTrue(np.isnan(persistence_forecasts(invalid).persistence_w24.iloc[-1]))

    def test_loss_scale_qlike_zero_floor_and_missing(self):
        actual = np.array([2.0, 0.0, np.nan])
        pred = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]])
        loss = loss_arrays(actual, pred)
        self.assertEqual(loss["variance_mse"][0, 0], 9)
        self.assertEqual(loss["rms_mse"][0, 0], 1)
        self.assertAlmostEqual(loss["qlike"][0, 0], 4 - np.log(4) - 1)
        for value in loss.values():
            self.assertEqual(value[0, 1], 0)
            np.testing.assert_array_equal(value[1], 0)
            self.assertTrue(np.isnan(value[2]).all())

    def test_common_mask_drops_any_missing_forecast_and_boundary_crossing(self):
        index = bars(40).index
        actual = np.ones(40)
        pred = np.ones((40, 6))
        pred[3, 5] = np.nan
        end = index[30]
        support = np.ones(40, dtype=bool)
        support[1] = False
        got = common_score_mask(index, actual=actual, predictions=pred,
                                source_support=support, end=end, horizon=24)
        self.assertEqual(np.flatnonzero(got).tolist(), [0, 2, 4, 5])

    def test_digest_refuses_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.npz"
            path.write_bytes(b"registered input")
            expected = file_digest(path)
            self.assertEqual(verify_digest(path, expected), expected)
            path.write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_digest(path, expected)

    def test_summary_separates_equal_quarter_from_pooled_and_does_not_rank(self):
        rows = []
        for n, m, b in ((1, 1.0, 2.0), (3, 4.0, 2.0)):
            rows.append({"rows": n, "regime": "bull",
                         "losses": {"m": dict.fromkeys(("variance_mse", "qlike", "rms_mse"), m),
                                    "b": dict.fromkeys(("variance_mse", "qlike", "rms_mse"), b)}})
        result = summarize(rows, model_ids=["m"], baseline_ids=["b"])
        metric = result["all"]["models"]["m"]["variance_mse"]
        self.assertEqual(metric["equal_quarter_mean"], 2.5)
        self.assertEqual(metric["pooled_mean"], 3.25)
        pair = result["all"]["paired_comparisons"][0]
        self.assertEqual(pair["equal_quarter_loss_ratio"], 1.25)
        self.assertEqual(pair["quarters_model_better"], 1)
        self.assertEqual(result["bear"]["quarters"], 0)
        self.assertNotIn("ranking", result)


if __name__ == "__main__":
    unittest.main()
