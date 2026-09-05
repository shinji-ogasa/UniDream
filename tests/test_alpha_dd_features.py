import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_features import (
    BASE_FEATURE_NAMES,
    BARS_DAY,
    COVERAGE_FEATURE_NAMES,
    FEATURE_NAMES,
    OBSERVED_COVERAGE_THRESHOLD,
    make_features,
)

EXPECTED_BASE_FEATURE_NAMES = (
    "momentum_1",
    "momentum_7",
    "momentum_30",
    "momentum_90",
    "vol_1",
    "vol_7",
    "vol_30",
    "drawdown_7",
    "drawdown_30",
    "drawdown_90",
    "vol_ratio",
    "flow_1",
    "flow_7",
)


def bars(n=9000):
    index = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    close = 100.0 * np.exp(0.00002 * np.arange(n) + 0.03 * np.sin(np.arange(n) / 233.0))
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 100.0,
            "quote_volume": 100.0,
            "n_trades": 10.0,
            "taker_buy_base": 50.0,
            "taker_buy_quote": 50.0,
        },
        index=index,
    )


class AlphaDDFeaturesTests(unittest.TestCase):
    def test_shape_order_and_nominal_history(self):
        data = bars()
        features = make_features(data)

        self.assertEqual(features.shape, (len(data), 16))
        self.assertEqual(tuple(features.columns), FEATURE_NAMES)
        self.assertEqual(tuple(BASE_FEATURE_NAMES), EXPECTED_BASE_FEATURE_NAMES)
        self.assertEqual(tuple(FEATURE_NAMES[:13]), EXPECTED_BASE_FEATURE_NAMES)
        self.assertEqual(tuple(FEATURE_NAMES[13:]), COVERAGE_FEATURE_NAMES)

        # Row t has only t-1 and earlier.  Rolling values do not become
        # eligible merely because min_periods is below the nominal window.
        self.assertIsNone(features["flow_7"].iloc[: 7 * BARS_DAY].first_valid_index())
        self.assertIsNone(features["price_coverage_90"].iloc[: 90 * BARS_DAY].first_valid_index())
        self.assertEqual(
            features["price_coverage_90"].first_valid_index(),
            data.index[90 * BARS_DAY],
        )
        self.assertEqual(
            features["momentum_90"].first_valid_index(),
            data.index[90 * BARS_DAY + 1],
        )

    def test_prefix_current_and_future_mutations_are_causal(self):
        data = bars()
        boundary = 8500
        mutated = data.copy()
        columns = ["close", "quote_volume", "taker_buy_quote"]
        mutated.loc[mutated.index[boundary]:, columns] *= 5.0
        mutated.loc[mutated.index[boundary + 4], "close"] = np.nan

        before = make_features(data)
        after = make_features(mutated)
        pd.testing.assert_frame_equal(
            before.iloc[: boundary + 1],
            after.iloc[: boundary + 1],
            check_exact=True,
        )

    def test_current_missing_close_does_not_change_current_row(self):
        data = bars()
        boundary = 8700
        mutated = data.copy()
        mutated.iloc[boundary, mutated.columns.get_loc("close")] = np.nan

        before = make_features(data)
        after = make_features(mutated)
        pd.testing.assert_frame_equal(
            before.iloc[[boundary]], after.iloc[[boundary]], check_exact=True
        )
        self.assertTrue(np.isnan(after["momentum_1"].iloc[boundary + 1]))
        self.assertLess(after["return_coverage_7"].iloc[boundary + 1], 1.0)

    def test_missing_and_zero_are_unavailable_not_zero_filled(self):
        data = bars()
        boundary = 8700

        missing_close = data.copy()
        zero_close = data.copy()
        missing_close.iloc[boundary, missing_close.columns.get_loc("close")] = np.nan
        zero_close.iloc[boundary, zero_close.columns.get_loc("close")] = 0.0
        pd.testing.assert_frame_equal(
            make_features(missing_close), make_features(zero_close), check_exact=True
        )

        missing_flow = data.copy()
        zero_flow = data.copy()
        missing_flow.iloc[boundary, missing_flow.columns.get_loc("quote_volume")] = np.nan
        zero_flow.iloc[boundary, zero_flow.columns.get_loc("quote_volume")] = 0.0
        flow_columns = ["flow_1", "flow_7", "flow_coverage_7"]
        pd.testing.assert_frame_equal(
            make_features(missing_flow)[flow_columns],
            make_features(zero_flow)[flow_columns],
            check_exact=True,
        )
        self.assertTrue(np.isnan(make_features(zero_flow)["flow_1"].iloc[boundary + 1]))
        self.assertLess(
            make_features(zero_flow)["flow_coverage_7"].iloc[boundary + 1], 1.0
        )

    def test_ceil_995_threshold_and_explicit_coverage(self):
        self.assertEqual(
            int(np.ceil(OBSERVED_COVERAGE_THRESHOLD * (7 * BARS_DAY))),
            669,
        )
        data = bars()
        boundary = 8500
        window_start = boundary - 7 * BARS_DAY

        three_flow_gaps = data.copy()
        four_flow_gaps = data.copy()
        for offset in range(3):
            three_flow_gaps.iloc[window_start + offset,
                                 three_flow_gaps.columns.get_loc("quote_volume")] = np.nan
        for offset in range(4):
            four_flow_gaps.iloc[window_start + offset,
                                four_flow_gaps.columns.get_loc("quote_volume")] = np.nan
        three_features = make_features(three_flow_gaps)
        four_features = make_features(four_flow_gaps)
        self.assertTrue(np.isfinite(three_features["flow_7"].iloc[boundary]))
        self.assertTrue(np.isnan(four_features["flow_7"].iloc[boundary]))
        self.assertAlmostEqual(
            three_features["flow_coverage_7"].iloc[boundary], 669 / 672
        )
        self.assertAlmostEqual(
            four_features["flow_coverage_7"].iloc[boundary], 668 / 672
        )

        three_price_gaps = data.copy()
        four_price_gaps = data.copy()
        for offset in range(3):
            three_price_gaps.iloc[window_start + offset,
                                  three_price_gaps.columns.get_loc("close")] = np.nan
        for offset in range(4):
            four_price_gaps.iloc[window_start + offset,
                                 four_price_gaps.columns.get_loc("close")] = np.nan
        three_features = make_features(three_price_gaps)
        four_features = make_features(four_price_gaps)
        self.assertTrue(np.isfinite(three_features["drawdown_7"].iloc[boundary]))
        self.assertTrue(np.isnan(four_features["drawdown_7"].iloc[boundary]))


if __name__ == "__main__":
    unittest.main()
