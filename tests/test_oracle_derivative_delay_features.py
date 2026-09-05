import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_delay_features import make_delayed_perp_groups
from unidream.experiments.oracle_derivative_features import (
    PERP_FLOW_FEATURE_NAMES, make_derivative_groups,
)


def bars(n=600, *, timing=False):
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC", name="bar_open_ts")
    step = np.arange(n)
    close = 100 * np.exp(.00001 * step + .02 * np.sin(step / 233))
    quote = 100 + .01 * step
    frame = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * .999,
        "close": close, "volume": 10., "quote_volume": quote,
        "n_trades": 10., "taker_buy_base": 5.,
        "taker_buy_quote": quote * (.45 + .2 * np.sin(step / 13))}, index=index)
    if timing:
        frame["bar_close_ts"] = index + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)
        frame["decision_ts"] = index + pd.Timedelta(minutes=15)
    return frame


class OracleDerivativeDelayFeaturesTests(unittest.TestCase):
    def test_zero_delay_exact_identity_and_group_schema(self):
        spot, um = bars(9000), bars(9000, timing=True)
        source = make_derivative_groups(spot, um)
        result = make_delayed_perp_groups(spot, um)
        self.assertEqual({k: len(v.columns) for k, v in result.items()},
                         {"technical": 29, "perp_delay0": 31, "perp_delay1": 31, "perp_delay4": 31})
        pd.testing.assert_frame_equal(result["technical"], source["technical"])
        pd.testing.assert_frame_equal(result["perp_delay0"], source["perp_flow"])
        for frame in result.values():
            self.assertTrue(frame.index.equals(spot.index))
            self.assertTrue(np.isfinite(frame.iloc[8641:].to_numpy()).all())

    def test_only_um_flow_gets_additional_delay_and_exact_warmup(self):
        spot, um = bars(), bars()
        result = make_delayed_perp_groups(spot, um)
        flow = result["perp_delay0"].loc[:, PERP_FLOW_FEATURE_NAMES]
        for delay in (0, 1, 4):
            frame = result[f"perp_delay{delay}"]
            pd.testing.assert_frame_equal(frame.loc[:, PERP_FLOW_FEATURE_NAMES], flow.shift(delay))
            for window in (24, 96):
                self.assertEqual(frame[f"perp_weighted_flow{window}"].first_valid_index(),
                                 spot.index[window + delay])

    def test_latest_permitted_um_bar_and_future_mutation_boundary(self):
        spot, um = bars(), bars(timing=True)
        t = 300
        original = make_delayed_perp_groups(spot, um)
        for delay in (0, 1, 4):
            name = f"perp_delay{delay}"
            latest = t - delay - 1
            future = um.copy()
            future.loc[um.index[latest + 1]:, "taker_buy_quote"] = (
                future.loc[um.index[latest + 1]:, "quote_volume"] * .95)
            changed = make_delayed_perp_groups(spot, future)
            pd.testing.assert_frame_equal(original[name].iloc[:t + 1], changed[name].iloc[:t + 1])
            # The boundary bar itself must still be used: over-delaying also fails.
            boundary = um.copy()
            boundary.loc[um.index[latest], "taker_buy_quote"] = um.quote_volume.iloc[latest]
            changed = make_delayed_perp_groups(spot, boundary)
            pd.testing.assert_frame_equal(original[name].iloc[:t], changed[name].iloc[:t])
            for col in PERP_FLOW_FEATURE_NAMES:
                self.assertNotEqual(original[name][col].iloc[t], changed[name][col].iloc[t])

    def test_spot_features_remain_at_original_timestamps(self):
        spot, um = bars(9000), bars(9000)
        t = 8700
        original = make_delayed_perp_groups(spot, um)
        changed_spot = spot.copy()
        changed_spot.loc[spot.index[t - 1], ["open", "high", "low", "close"]] *= 1.02
        changed = make_delayed_perp_groups(changed_spot, um)
        technical = list(original["technical"].columns)
        self.assertNotEqual(original["technical"].momentum_1.iloc[t],
                            changed["technical"].momentum_1.iloc[t])
        for delay in (0, 1, 4):
            name = f"perp_delay{delay}"
            pd.testing.assert_frame_equal(changed[name].loc[:, technical], changed["technical"])
            pd.testing.assert_frame_equal(original[name].loc[:, PERP_FLOW_FEATURE_NAMES],
                                          changed[name].loc[:, PERP_FLOW_FEATURE_NAMES])
            pd.testing.assert_frame_equal(original[name].iloc[:t], changed[name].iloc[:t])

    def test_missing_um_bar_stays_on_full_grid_and_delays_nan_windows(self):
        spot, um = bars(), bars(timing=True)
        gap = 250
        absent = um.drop(um.index[gap])
        explicit = um.copy()
        explicit.loc[um.index[gap], ["close", "quote_volume", "taker_buy_quote"]] = np.nan
        sparse_result = make_delayed_perp_groups(spot, absent)
        explicit_result = make_delayed_perp_groups(spot, explicit)
        for name in sparse_result:
            pd.testing.assert_frame_equal(sparse_result[name], explicit_result[name])
            self.assertTrue(sparse_result[name].index.equals(spot.index))
        for delay in (0, 1, 4):
            frame = sparse_result[f"perp_delay{delay}"]
            for window in (24, 96):
                col = frame[f"perp_weighted_flow{window}"]
                start, end = gap + 1 + delay, gap + window + delay
                self.assertTrue(np.isfinite(col.iloc[start - 1]))
                self.assertTrue(col.iloc[start:end + 1].isna().all())
                self.assertTrue(np.isfinite(col.iloc[end + 1]))

    def test_valid_custom_delay_order_and_numpy_integer(self):
        spot, um = bars(), bars()
        result = make_delayed_perp_groups(spot, um, delays=(np.int64(2), 0))
        self.assertEqual(list(result), ["technical", "perp_delay2", "perp_delay0"])
        pd.testing.assert_frame_equal(result["perp_delay2"].loc[:, PERP_FLOW_FEATURE_NAMES],
                                      result["perp_delay0"].loc[:, PERP_FLOW_FEATURE_NAMES].shift(2))

    def test_invalid_delays_rejected(self):
        spot, um = bars(100), bars(100)
        for delays in ((), (1, 4), (0, -1), (0, 0), (False, 1), (0, True),
                       (0, np.bool_(True)), (0., 1), (0, 1.), (0, "1"), None, 0):
            with self.subTest(delays=delays):
                with self.assertRaisesRegex(ValueError, "distinct nonnegative integers including 0"):
                    make_delayed_perp_groups(spot, um, delays=delays)

    def test_original_grid_timezone_and_timing_contract_still_applies(self):
        spot, um = bars(100), bars(100, timing=True)
        with self.assertRaisesRegex(ValueError, "complete"):
            make_delayed_perp_groups(spot.drop(spot.index[5]), um)
        naive = um.copy()
        naive.index = naive.index.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_delayed_perp_groups(spot, naive)
        shifted = um.copy()
        shifted.index += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "timing contract"):
            make_delayed_perp_groups(spot, shifted)
        japan = um.copy()
        japan.index = japan.index.tz_convert("Asia/Tokyo")
        expected, actual = make_delayed_perp_groups(spot, um), make_delayed_perp_groups(spot, japan)
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], actual[name])


if __name__ == "__main__":
    unittest.main()
