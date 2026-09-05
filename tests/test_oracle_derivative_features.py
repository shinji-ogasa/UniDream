import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_features import (
    DERIVATIVE_FEATURE_NAMES, PERP_FLOW_FEATURE_NAMES, make_derivative_groups,
)
from unidream.experiments.oracle_frontier_features import make_feature_groups


def bars(n=9000, *, timing=False):
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC", name="bar_open_ts")
    step = np.arange(n)
    close = 100 * np.exp(.00001 * step + .02 * np.sin(step / 233))
    q = 100 + .01 * step
    frame = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * .999,
                          "close": close, "volume": 10., "quote_volume": q,
                          "n_trades": 10., "taker_buy_base": 5., "taker_buy_quote": q * .5}, index=index)
    if timing:
        frame["bar_close_ts"] = index + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)
        frame["decision_ts"] = index + pd.Timedelta(minutes=15)
    return frame


def extras(spot, um):
    return make_derivative_groups(spot, um)["derivative"].loc[:, DERIVATIVE_FEATURE_NAMES]


class OracleDerivativeFeaturesTests(unittest.TestCase):
    def test_group_schema_and_exact_existing_feature_parity(self):
        spot, um = bars(), bars(timing=True)
        original, grouped = make_feature_groups(spot), make_derivative_groups(spot, um)
        self.assertEqual({k: len(v.columns) for k, v in grouped.items()},
                         {"base16": 16, "technical": 29, "perp_flow": 31, "derivative": 37})
        pd.testing.assert_frame_equal(grouped["base16"], original["base16"])
        pd.testing.assert_frame_equal(grouped["technical"], original["technical"])
        self.assertEqual(tuple(grouped["perp_flow"].columns[-2:]), PERP_FLOW_FEATURE_NAMES)
        self.assertEqual(tuple(grouped["derivative"].columns[-8:]), DERIVATIVE_FEATURE_NAMES)
        self.assertTrue(np.isfinite(grouped["derivative"].iloc[8641:].to_numpy()).all())

    def test_future_mutation_and_truncation_are_invariant_in_both_markets(self):
        spot, um = bars(), bars(timing=True)
        boundary = 8700
        original = make_derivative_groups(spot, um)
        changed_spot, changed_um = spot.copy(), um.copy()
        raw_columns = list(spot.columns)
        changed_spot.loc[spot.index[boundary]:, raw_columns] *= 2
        changed_um.loc[um.index[boundary]:, raw_columns] *= 3
        changed_um.loc[um.index[boundary + 1], raw_columns] = np.nan
        changed = make_derivative_groups(changed_spot, changed_um)
        prefix = make_derivative_groups(spot.iloc[:boundary + 1], um.iloc[:boundary + 1])
        for name in original:
            pd.testing.assert_frame_equal(original[name].iloc[:boundary + 1], changed[name].iloc[:boundary + 1])
            pd.testing.assert_frame_equal(original[name].iloc[:boundary + 1], prefix[name])

    def test_one_shift_and_exact_nominal_warmup(self):
        spot, um = bars(800), bars(800)
        um["close"] = 1.02 * spot.close
        out = extras(spot, um)
        expected = {"perp_weighted_flow24": 24, "perp_weighted_flow96": 96,
                    "relative_quote_activity24_672": 672, "traded_close_basis": 1,
                    "traded_close_basis_change24": 25, "relative_realized_variance24": 25}
        for name, row in expected.items():
            self.assertEqual(out[name].first_valid_index(), spot.index[row])
        self.assertAlmostEqual(out.traded_close_basis.iloc[1], np.log(1.02))

    def test_missing_um_timestamp_is_missing_row_without_fill_or_return_bridge(self):
        spot, um = bars(900), bars(900, timing=True)
        missing = um.copy()
        missing.loc[um.index[800], list(spot.columns)] = np.nan
        pd.testing.assert_frame_equal(extras(spot, missing), extras(spot, um.drop(um.index[800])))
        out = extras(spot, missing)
        self.assertTrue(out.loc[spot.index[801], ["perp_weighted_flow24", "traded_close_basis",
                                                 "relative_realized_variance24"]].isna().all())
        self.assertTrue(np.isnan(out.relative_realized_variance24.iloc[825]))
        self.assertTrue(np.isfinite(out.relative_realized_variance24.iloc[826]))

    def test_activity_uses_independent_672_volume_masks(self):
        spot, um = bars(1000), bars(1000)
        um["quote_volume"] *= 2
        spot.loc[spot.index[400:403], "quote_volume"] = np.nan
        um.loc[um.index[410:413], "quote_volume"] = np.nan
        # Each market has 669/672 valid quotes. A joint mask would have only 666.
        out = extras(spot, um)
        self.assertTrue(np.isfinite(out.relative_quote_activity24_672.iloc[999]))
        sp, pp = spot.quote_volume.iloc[327:999], um.quote_volume.iloc[327:999]
        expected = np.log((pp.iloc[-24:].mean() / pp.mean()) / (sp.iloc[-24:].mean() / sp.mean()))
        self.assertAlmostEqual(out.relative_quote_activity24_672.iloc[999], expected)
        um.loc[um.index[413], "quote_volume"] = np.nan
        self.assertTrue(np.isnan(extras(spot, um).relative_quote_activity24_672.iloc[999]))

    def test_late_um_start_retains_preflight_669_observation_boundary(self):
        spot, um = bars(1000), bars(1000, timing=True).iloc[200:]
        out = extras(spot, um)
        self.assertEqual(out.relative_quote_activity24_672.first_valid_index(), spot.index[869])
        self.assertEqual(out.traded_close_basis.first_valid_index(), spot.index[201])

    def test_invalid_flow_is_unavailable_and_does_not_mask_quote_activity(self):
        spot, um = bars(900), bars(900)
        for bad in (-1., np.inf, 1e9):
            changed = um.copy()
            changed.loc[um.index[800], "taker_buy_quote"] = bad
            out = extras(spot, changed)
            self.assertTrue(np.isnan(out.perp_weighted_flow24.iloc[801]))
            self.assertEqual(out.relative_quote_activity24_672.iloc[801], extras(spot, um).relative_quote_activity24_672.iloc[801])
        for bad in (0., -1., np.inf):
            invalid, missing = um.copy(), um.copy()
            invalid.loc[um.index[800], "quote_volume"] = bad
            missing.loc[um.index[800], "quote_volume"] = np.nan
            pd.testing.assert_frame_equal(extras(spot, invalid), extras(spot, missing))

    def test_signed_flow_bounds_and_flat_variance(self):
        spot, um = bars(900), bars(900)
        spot["taker_buy_quote"] = 0.
        um["taker_buy_quote"] = um.quote_volume
        spot[["open", "high", "low", "close"]] = 100.
        um[["open", "high", "low", "close"]] = 100.
        out = extras(spot, um)
        for window in (24, 96):
            self.assertAlmostEqual(out[f"perp_weighted_flow{window}"].iloc[-1], 1.)
            self.assertAlmostEqual(out[f"perp_minus_spot_flow{window}"].iloc[-1], 2.)
            self.assertTrue(out[f"perp_weighted_flow{window}"].dropna().between(-1, 1).all())
        self.assertEqual(out.relative_realized_variance24.iloc[-1], 0.)
        um.loc[um.index[-2], "close"] = 0.
        out = extras(spot, um)
        self.assertTrue(np.isnan(out.relative_realized_variance24.iloc[-1]))
        self.assertTrue(np.isnan(out.traded_close_basis.iloc[-1]))

    def test_timing_columns_reject_double_shift_or_naive_clock(self):
        spot, um = bars(100), bars(100, timing=True)
        shifted = um.copy()
        shifted.index += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "timing contract"):
            make_derivative_groups(spot, shifted)
        with self.assertRaisesRegex(ValueError, "raw bar-open time"):
            make_derivative_groups(spot, um.set_index("decision_ts", drop=False))
        naive = um.copy()
        naive["decision_ts"] = naive.decision_ts.dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_derivative_groups(spot, naive)
        absent = um.copy()
        absent.loc[um.index[5], "bar_close_ts"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "missing bar_close_ts"):
            make_derivative_groups(spot, absent)

    def test_raw_schema_grid_and_timezone_validation(self):
        spot, um = bars(100), bars(100, timing=True)
        with self.assertRaisesRegex(ValueError, "missing raw input"):
            make_derivative_groups(spot, um.drop(columns="quote_volume"))
        with self.assertRaisesRegex(ValueError, "complete"):
            make_derivative_groups(spot.drop(spot.index[5]), um)
        with self.assertRaisesRegex(ValueError, "unique"):
            make_derivative_groups(spot, pd.concat([um, um.iloc[[-1]]]))
        naive = um.copy()
        naive.index = naive.index.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_derivative_groups(spot, naive)
        offgrid = um.copy()
        offgrid.index += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "15-minute grid"):
            make_derivative_groups(spot, offgrid)
        japan = um.copy()
        japan.index = japan.index.tz_convert("Asia/Tokyo")
        pd.testing.assert_frame_equal(extras(spot, um), extras(spot, japan))


if __name__ == "__main__":
    unittest.main()
