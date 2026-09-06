import math
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_features import make_derivative_groups
from unidream.experiments.oracle_short_features import (
    FLOW_FEATURE_NAMES, PRICE_FEATURE_NAMES, make_short_feature_groups,
)


def raw_bars(n=900):
    index = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC", name="bar_open_ts")
    close = 100 * np.exp(np.arange(n) * .0001)
    return pd.DataFrame({"open": close / 1.001, "close": close,
        "high": close * 1.01, "low": close * .99,
        "quote_volume": 100., "taker_buy_quote": 60., "n_trades": 10.}, index=index)


class OracleShortFeaturesTests(unittest.TestCase):
    def test_exact_base_schema_and_one_shift_hand_calculation(self):
        spot, um = raw_bars(), raw_bars()
        spot.loc[spot.index[799], ["open", "high", "low", "close"]] = [100., 110., 90., 105.]
        spot.loc[spot.index[796:800], "quote_volume"] = [10., 20., 30., 40.]
        spot.loc[spot.index[796:800], "taker_buy_quote"] = [10., 10., 0., 40.]
        um.loc[um.index[796:800], "taker_buy_quote"] = [0., 25., 50., 100.]
        result = make_short_feature_groups(spot, um)
        self.assertEqual({k: len(v.columns) for k, v in result.items()},
            {"technical": 29, "technical_short_price": 34, "technical_short_flow": 32, "technical_short_both": 37})
        pd.testing.assert_frame_equal(result["technical"], make_derivative_groups(spot, um)["technical"])
        for frame in result.values():
            pd.testing.assert_frame_equal(frame.iloc[:, :29], result["technical"])
        both = result["technical_short_both"]
        self.assertEqual(tuple(both.columns[29:]), PRICE_FEATURE_NAMES + FLOW_FEATURE_NAMES)
        r = both.iloc[800]
        self.assertEqual(r.spot_body_sign1, 1.)
        self.assertEqual(r.spot_close_location1, .5)
        self.assertAlmostEqual(r.spot_weighted_flow4, .2)
        self.assertAlmostEqual(r.perp_weighted_flow4, -.125)
        for k in (4, 16, 48):
            self.assertAlmostEqual(r[f"spot_log_return{k}"], math.log(105. / spot.close.iloc[799 - k]), places=14)
        self.assertAlmostEqual(r.spot_quote_activity24_672, math.log((2100. / 24) / (66900. / 672)), places=14)

    def test_return_full_close_history_boundary_48_and_gap_propagation(self):
        spot, um = raw_bars(160), raw_bars(160)
        new = make_short_feature_groups(spot, um)["technical_short_price"]
        for k in (4, 16, 48):
            col = new[f"spot_log_return{k}"]
            self.assertTrue(col.iloc[:k + 1].isna().all())
            self.assertAlmostEqual(col.iloc[k + 1], k * .0001, places=14)
        spot.loc[spot.index[60], "close"] = np.nan
        after = make_short_feature_groups(spot, um)["technical_short_price"]
        for k in (4, 16, 48):
            col = after[f"spot_log_return{k}"]
            self.assertTrue(col.iloc[61:62 + k].isna().all())
            self.assertTrue(math.isfinite(col.iloc[62 + k]))
            self.assertTrue(math.isfinite(col.iloc[60]))

    def test_future_mutation_and_prefix_invariance(self):
        spot, um = raw_bars(), raw_bars()
        original = make_short_feature_groups(spot, um)
        changed_spot, changed_um = spot.copy(), um.copy()
        for frame in (changed_spot, changed_um):
            frame.loc[frame.index[800]:, ["open", "high", "low", "close"]] *= 1.2
            frame.loc[frame.index[800]:, "taker_buy_quote"] = 0.
            frame.loc[frame.index[800]:, "quote_volume"] = 200.
        changed = make_short_feature_groups(changed_spot, changed_um)
        truncated = make_short_feature_groups(spot.iloc[:801], um.iloc[:801])
        for name in original:
            pd.testing.assert_frame_equal(changed[name].iloc[:801], original[name].iloc[:801])
            pd.testing.assert_frame_equal(truncated[name], original[name].iloc[:801])

    def test_latest_um_bar_enters_after_exactly_one_shift(self):
        spot, um = raw_bars(), raw_bars()
        original = make_short_feature_groups(spot, um)
        um.loc[um.index[799], "taker_buy_quote"] = 0.
        changed = make_short_feature_groups(spot, um)
        col = "perp_weighted_flow4"
        pd.testing.assert_series_equal(changed["technical_short_flow"][col].iloc[:800], original["technical_short_flow"][col].iloc[:800])
        self.assertNotEqual(changed["technical_short_flow"][col].iloc[800], original["technical_short_flow"][col].iloc[800])
        pd.testing.assert_frame_equal(changed["technical_short_price"], original["technical_short_price"])

    def test_sparse_um_and_zero_invalid_taker_are_missing_not_neutral(self):
        spot, um = raw_bars(), raw_bars()
        sparse = um.drop(um.index[700])
        result = make_short_feature_groups(spot, sparse)["technical_short_flow"]
        self.assertTrue(result.perp_weighted_flow4.iloc[701:705].isna().all())
        self.assertTrue(math.isfinite(result.perp_weighted_flow4.iloc[705]))
        self.assertEqual(len(result), len(spot))
        for quote, buy in [(0., 0.), (100., -1.), (100., 101.), (np.nan, 50.), (100., np.nan)]:
            with self.subTest(quote=quote, buy=buy):
                altered = spot.copy()
                altered.loc[altered.index[700], ["quote_volume", "taker_buy_quote"]] = [quote, buy]
                flow = make_short_feature_groups(altered, um)["technical_short_flow"].spot_weighted_flow4
                self.assertTrue(flow.iloc[701:705].isna().all())
        for buy, expected in [(0., -1.), (100., 1.)]:
            um["taker_buy_quote"] = buy
            flow = make_short_feature_groups(spot, um)["technical_short_flow"].perp_weighted_flow4
            self.assertEqual(flow.iloc[4], expected)

    def test_volume_nominal_history_and_669_of_672_coverage(self):
        spot, um = raw_bars(), raw_bars()
        original = make_short_feature_groups(spot, um)["technical_short_flow"].spot_quote_activity24_672
        self.assertTrue(original.iloc[:672].isna().all())
        self.assertEqual(original.iloc[672], 0.)
        spot.loc[spot.index[[1, 2, 3]], "quote_volume"] = np.nan
        accepted = make_short_feature_groups(spot, um)["technical_short_flow"].spot_quote_activity24_672
        self.assertEqual(accepted.iloc[672], 0.)
        spot.loc[spot.index[4], "quote_volume"] = np.nan
        rejected = make_short_feature_groups(spot, um)["technical_short_flow"].spot_quote_activity24_672
        self.assertTrue(np.isnan(rejected.iloc[672]))
        # The 24-bar mean still requires 24/24, even if the 672 mean allows gaps.
        spot, um = raw_bars(), raw_bars()
        spot.loc[spot.index[670], "quote_volume"] = np.nan
        short_gap = make_short_feature_groups(spot, um)["technical_short_flow"].spot_quote_activity24_672
        self.assertTrue(short_gap.iloc[672:695].isna().all())
        self.assertEqual(short_gap.iloc[695], 0.)

    def test_flat_candle_neutral_and_inconsistent_or_missing_ohlc_rejected(self):
        spot, um = raw_bars(), raw_bars()
        spot.loc[spot.index[799], ["open", "high", "low", "close"]] = 100.
        flat = make_short_feature_groups(spot, um)["technical_short_price"].iloc[800]
        self.assertEqual(flat.spot_body_sign1, 0.)
        self.assertEqual(flat.spot_close_location1, 0.)
        for column, value in [("open", np.nan), ("open", 101.), ("high", 99.), ("low", 101.), ("close", 0.)]:
            with self.subTest(column=column, value=value):
                changed = spot.copy()
                changed.loc[changed.index[799], column] = value
                row = make_short_feature_groups(changed, um)["technical_short_price"].iloc[800]
                self.assertTrue(np.isnan(row.spot_body_sign1))
                self.assertTrue(np.isnan(row.spot_close_location1))

    def test_raw_metadata_validation_and_no_input_mutation(self):
        spot, um = raw_bars(), raw_bars()
        for frame in (spot, um):
            frame["bar_close_ts"] = frame.index + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)
            frame["decision_ts"] = frame.index + pd.Timedelta(minutes=15)
        before_spot, before_um = spot.copy(deep=True), um.copy(deep=True)
        make_short_feature_groups(spot, um)
        pd.testing.assert_frame_equal(spot, before_spot)
        pd.testing.assert_frame_equal(um, before_um)
        bad = um.copy()
        bad.loc[bad.index[800], "decision_ts"] += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "timing contract"):
            make_short_feature_groups(spot, bad)
        for frame in [spot.rename_axis("decision_ts"), spot.set_axis(spot.index.tz_localize(None)),
                      spot.drop(spot.index[30]), spot.iloc[::-1], pd.concat([spot, spot.iloc[-1:]])]:
            with self.subTest(index=frame.index.name), self.assertRaises(ValueError):
                make_short_feature_groups(frame, um)
        with self.assertRaises(ValueError):
            make_short_feature_groups(spot.drop(columns="open"), um)


if __name__ == "__main__":
    unittest.main()
