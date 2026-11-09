import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_ablation_features import (
    BASE_COLUMNS,
    FEATURE_VERSION,
    RAW_COLUMNS,
    TA_COLUMNS,
    feature_columns,
    make_window_features,
    transform_windows,
)


def sample_bars(n=300):
    time = np.arange(n)
    close = 100.0 * np.exp(0.001 * time + 0.02 * np.sin(time / 7.0))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 10.0 + time % 17,
            "funding_rate": 0.0001 * np.cos(time / 15.0),
            "open_interest": 1000.0 + time,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )


class WMAbalationFeaturesTests(unittest.TestCase):
    def test_registry_and_groups_have_exactly_shared_channels(self):
        frame = sample_bars(256)
        raw = frame.loc[:, list(RAW_COLUMNS)].to_numpy()[None]
        market_columns = ("funding_rate", "open_interest")
        market = frame.loc[:, list(market_columns)].to_numpy()[None]
        a = transform_windows(raw, "A")
        b = transform_windows(raw, "B")
        c = transform_windows(raw, "C", market, market_columns)
        d = transform_windows(raw, "D", market, market_columns)
        self.assertEqual(FEATURE_VERSION, "window_local_ohlcv_ta44_v1")
        self.assertEqual(len(TA_COLUMNS), 44)
        self.assertEqual(d.shape, (1, 256, 51))
        self.assertEqual(d.dtype, np.float32)
        self.assertEqual(feature_columns("D", market_columns), BASE_COLUMNS + market_columns + TA_COLUMNS)
        np.testing.assert_array_equal(a, b[:, :, :5])
        np.testing.assert_array_equal(a, c[:, :, :5])
        np.testing.assert_array_equal(c, d[:, :, :7])
        np.testing.assert_array_equal(b[:, :, 5:], d[:, :, 7:])

    def test_current_and_future_bars_do_not_change_earlier_features(self):
        frame = sample_bars(256)
        cut = 140
        before = make_window_features(frame, "B")
        changed = frame.copy()
        changed.loc[changed.index[cut]:, list(RAW_COLUMNS[:4])] *= 2.0
        changed.loc[changed.index[cut]:, "volume"] *= 11.0
        after = make_window_features(changed, "B")
        pd.testing.assert_frame_equal(before.iloc[:cut], after.iloc[:cut], check_exact=True)
        self.assertFalse(np.array_equal(before.iloc[cut:].values, after.iloc[cut:].values))
        # Prefix computation must agree with a computation that has later bars.
        prefix = make_window_features(frame.iloc[:cut], "B")
        pd.testing.assert_frame_equal(before.iloc[:cut], prefix, check_exact=True)

    def test_completed_bar_boundary_and_history_isolation(self):
        frame = sample_bars(300)
        window = frame.iloc[-256:]
        decision = frame.index[-1] + pd.Timedelta(minutes=15)
        before = make_window_features(window, "B", decision_time=decision)
        # EMA, RSI and OBV must never depend on history outside this window.
        changed = frame.copy()
        changed.iloc[:44, changed.columns.get_indexer(list(RAW_COLUMNS[:4]))] *= 10.0
        changed.iloc[:44, changed.columns.get_loc("volume")] *= 100.0
        after = make_window_features(changed.iloc[-256:], "B", decision_time=decision)
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        with self.assertRaisesRegex(ValueError, "last completed"):
            make_window_features(window, "B", decision_time=frame.index[-1])

    def test_each_window_is_independent_of_other_batch_examples(self):
        frame = sample_bars(300)
        raw = frame.loc[:, list(RAW_COLUMNS)].to_numpy()
        batch = np.stack((raw[:256], raw[-256:]))
        together = transform_windows(batch, "B")
        for i in range(2):
            alone = transform_windows(batch[i:i + 1], "B")
            np.testing.assert_array_equal(together[i:i + 1], alone)

    def test_flat_zero_volume_has_finite_and_neutral_indicators(self):
        raw = np.ones((2, 256, 5), dtype=float) * 100.0
        raw[:, :, 4] = 0.0
        result = transform_windows(raw, "B")
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result[:, :, :5] == 0.0).all())
        names = feature_columns("B")
        for name in ("rsi_14", "rsi_28", "stochastic_14", "mfi_14", "donchian_position_16"):
            np.testing.assert_array_equal(result[:, :, names.index(name)], 0.5)
        for name in ("adx_14", "cci_20", "realized_vol_16", "obv_flow_16", "vwap_gap_16", "relative_volume_16"):
            np.testing.assert_array_equal(result[:, :, names.index(name)], 0.0)

    def test_hand_calculated_return_momentum_variance_and_flow(self):
        raw = np.array([[
            [100.0, 102.0, 99.0, 101.0, 10.0],
            [101.0, 103.0, 100.0, 102.0, 20.0],
            [102.0, 103.0, 100.0, 101.0, 30.0],
        ]])
        values = transform_windows(raw, "B")[0]
        columns = feature_columns("B")
        np.testing.assert_allclose(values[0, :4], np.log(np.array([100., 102., 99., 101.]) / 100.0))
        np.testing.assert_allclose(values[:, columns.index("momentum_4")], np.log(np.array([101., 102., 101.]) / 101.0))
        expected_rv = np.sqrt((np.log(102.0 / 101.0) ** 2 + np.log(101.0 / 102.0) ** 2) / 3.0)
        self.assertAlmostEqual(float(values[-1, columns.index("realized_vol_4")]), expected_rv, places=8)
        self.assertAlmostEqual(float(values[-1, columns.index("obv_flow_4")]), (20.0 - 30.0) / 60.0, places=7)

    def test_cci_uses_current_window_mean_absolute_deviation(self):
        raw = np.array([[[value, value, value, value, 1.] for value in (1., 2., 4., 8.)]])
        result = transform_windows(raw, "B")
        expected = (8.0 - 3.75) / (0.015 * np.mean(np.abs(np.array([1., 2., 4., 8.]) - 3.75)))
        self.assertAlmostEqual(float(result[0, -1, feature_columns("B").index("cci_20")]), expected, places=4)

    def test_price_scaling_preserves_dimensionless_features(self):
        raw = sample_bars(256).loc[:, list(RAW_COLUMNS)].to_numpy()[None]
        scaled = raw.copy()
        scaled[:, :, :4] *= 7.0
        np.testing.assert_allclose(
            transform_windows(raw, "B"), transform_windows(scaled, "B"),
            rtol=1e-5, atol=2e-6,
        )

    def test_rejects_missing_invalid_or_misaligned_inputs(self):
        raw = sample_bars(32).loc[:, list(RAW_COLUMNS)].to_numpy()[None]
        invalid = raw.copy()
        invalid[0, 4, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            transform_windows(invalid, "B")
        with self.assertRaisesRegex(ValueError, "require market_columns"):
            transform_windows(raw, "C")
        with self.assertRaisesRegex(ValueError, "shape"):
            transform_windows(raw, "C", np.zeros((1, 31, 1)), ("funding",))
        frame = sample_bars(32)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            make_window_features(frame.drop(frame.index[5]), "B")
        with self.assertRaisesRegex(ValueError, "timezone"):
            make_window_features(frame.set_axis(frame.index.tz_localize(None)), "B")


if __name__ == "__main__":
    unittest.main()
