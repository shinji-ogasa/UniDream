import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_features import FEATURE_NAMES, make_features
from unidream.experiments.oracle_frontier_features import (
    FLOW_FEATURE_NAMES, TECHNICAL_FEATURE_NAMES, TRADE_FEATURE_NAMES, make_feature_groups,
)


def bars(n=9000):
    index = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    step = np.arange(n)
    close = 100.0 * np.exp(0.00002 * step + 0.03 * np.sin(step / 233.0))
    quote = 1000.0 + 100.0 * np.cos(step / 107.0)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "quote_volume": quote,
        "taker_buy_quote": quote * (0.5 + 0.1 * np.sin(step / 59.0)),
        "n_trades": 10.0 + 2.0 * np.cos(step / 43.0),
    }, index=index)


class OracleFrontierFeaturesTests(unittest.TestCase):
    def test_group_columns_and_base_parity(self):
        data = bars()
        groups = make_feature_groups(data)
        self.assertEqual(list(groups), ["base16", "technical", "flow"])
        self.assertEqual({name: frame.shape[1] for name, frame in groups.items()},
                         {"base16": 16, "technical": 29, "flow": 24})
        pd.testing.assert_frame_equal(groups["base16"], make_features(data))
        self.assertEqual(tuple(groups["technical"].columns), FEATURE_NAMES + TECHNICAL_FEATURE_NAMES)
        self.assertEqual(tuple(groups["flow"].columns), FEATURE_NAMES + FLOW_FEATURE_NAMES + TRADE_FEATURE_NAMES)
        for frame in groups.values():
            self.assertTrue(np.isfinite(frame.iloc[8641:].to_numpy()).all())

    def test_current_future_mutation_and_truncation_invariance(self):
        data = bars()
        boundary = 8700
        changed = data.copy()
        changed.iloc[boundary:] *= 5.0
        changed.iloc[boundary + 3] = np.nan
        original = make_feature_groups(data)
        mutated = make_feature_groups(changed)
        truncated = make_feature_groups(data.iloc[:boundary + 1])
        for name in original:
            pd.testing.assert_frame_equal(original[name].iloc[:boundary + 1], mutated[name].iloc[:boundary + 1])
            pd.testing.assert_frame_equal(original[name].iloc[:boundary + 1], truncated[name])

    def test_missing_timestamps_are_gap_rows_not_shorter_windows(self):
        data = bars()
        boundary = 8700
        omitted = make_feature_groups(data.drop(data.index[boundary]))
        explicit = data.copy()
        explicit.iloc[boundary] = np.nan
        explicit_groups = make_feature_groups(explicit)
        for name in omitted:
            pd.testing.assert_frame_equal(omitted[name], explicit_groups[name])
        technical = omitted["technical"]
        self.assertTrue(np.isnan(technical["rsi14"].iloc[boundary + 1]))
        self.assertTrue(np.isnan(technical["channel_position96"].iloc[boundary + 1]))
        self.assertTrue(np.isnan(technical["weighted_flow96"].iloc[boundary + 1]))

    def test_coverage_and_nominal_history(self):
        data = bars()
        clean = make_feature_groups(data)["technical"]
        self.assertEqual(clean["rsi14"].first_valid_index(), data.index[15])
        self.assertEqual(clean["atr14_relative"].first_valid_index(), data.index[15])
        self.assertEqual(clean["channel_position672"].first_valid_index(), data.index[672])
        self.assertEqual(clean["efficiency_ratio672"].first_valid_index(), data.index[673])
        self.assertEqual(clean["weighted_flow672"].first_valid_index(), data.index[672])
        boundary = 8700
        for missing_count, eligible in ((3, True), (4, False)):
            changed = data.copy()
            changed.iloc[boundary - 600:boundary - 600 + missing_count] = np.nan
            technical = make_feature_groups(changed)["technical"]
            self.assertEqual(bool(np.isfinite(technical["weighted_flow672"].iloc[boundary])), eligible)
            self.assertEqual(bool(np.isfinite(technical["channel_position672"].iloc[boundary])), eligible)

    def test_measured_zero_ranges_and_one_sided_prices(self):
        data = bars(800)
        data[["open", "high", "low", "close"]] = 100.0
        technical = make_feature_groups(data)["technical"]
        expected = {"rsi14": 50.0, "rsi96": 50.0, "atr14_relative": 0.0,
                    "channel_position96": 0.5, "price_zscore96": 0.0,
                    "efficiency_ratio96": 0.0, "downside_upside_log_vol_ratio96": 0.0}
        for column, value in expected.items():
            self.assertEqual(technical[column].iloc[-1], value)
        for direction, rsi in ((1.0, 100.0), (-1.0, 0.0)):
            close = 100.0 * np.exp(direction * 0.001 * np.arange(len(data)))
            data[["open", "high", "low", "close"]] = np.column_stack([close] * 4)
            technical = make_feature_groups(data)["technical"]
            self.assertEqual(technical["rsi14"].iloc[-1], rsi)
            self.assertTrue(np.isfinite(technical.loc[:, TECHNICAL_FEATURE_NAMES].iloc[673:].to_numpy()).all())

    def test_optional_trade_columns(self):
        data = bars()
        without = make_feature_groups(data.drop(columns="n_trades"))["flow"]
        self.assertEqual(without.shape[1], 20)
        renamed = make_feature_groups(data.rename(columns={"n_trades": "trades"}))["flow"]
        original = make_feature_groups(data)["flow"]
        pd.testing.assert_frame_equal(renamed, original)

    def test_weighted_flow_uses_matching_observed_volume(self):
        data = bars(800)
        data["taker_buy_quote"] = 0.75 * data["quote_volume"]
        technical = make_feature_groups(data)["technical"]
        self.assertAlmostEqual(technical["weighted_flow672"].iloc[-1], 0.5)
        data.iloc[-2, data.columns.get_loc("taker_buy_quote")] = -1.0
        technical = make_feature_groups(data)["technical"]
        self.assertTrue(np.isnan(technical["weighted_flow96"].iloc[-1]))
        self.assertAlmostEqual(technical["weighted_flow672"].iloc[-1], 0.5)

    def test_rejects_irregular_grid_and_duplicate_timestamps(self):
        data = bars(10)
        with self.assertRaisesRegex(ValueError, "15-minute grid"):
            make_feature_groups(data.set_axis(data.index + pd.Timedelta(minutes=1)))
        with self.assertRaisesRegex(ValueError, "unique"):
            make_feature_groups(pd.concat([data, data.iloc[[-1]]]))


if __name__ == "__main__":
    unittest.main()
