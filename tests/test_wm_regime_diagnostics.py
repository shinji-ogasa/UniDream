import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_regime_diagnostics import (
    contiguous_windows, correlation_summary, decision_metrics,
    distribution_shift, feature_name, past_variance,
)


class RegimeDiagnosticTest(unittest.TestCase):
    def test_past_volatility_does_not_read_the_decision_or_future_candles(self):
        close = np.exp(np.arange(40) * .01)
        original = past_variance(close)
        changed = close.copy()
        changed[20:] *= np.linspace(2, 3, 20)
        perturbed = past_variance(changed)
        self.assertAlmostEqual(original[20], 16 * .01 ** 2)
        np.testing.assert_allclose(original[:21], perturbed[:21], equal_nan=True)
        self.assertNotEqual(original[21], perturbed[21])

    def test_windows_never_cross_missing_bars(self):
        full = pd.date_range('2026-01-01', periods=20, freq='15min', tz='UTC')
        index = full.delete(8)
        sections = contiguous_windows(index, 5, 3)
        self.assertEqual([(s.start, s.stop) for s in sections], [(0, 5), (3, 8), (8, 13), (11, 16), (14, 19)])
        for section in sections:
            self.assertTrue((np.diff(index[section].asi8) == pd.Timedelta(minutes=15).value).all())

    def test_absent_class_does_not_invent_a_three_class_ba(self):
        score = decision_metrics([0, 0, 1], [0, 1, 1])
        self.assertIsNone(score['balanced_accuracy'])
        self.assertEqual(score['recall_down'], .5)
        self.assertEqual(score['recall_flat'], 1)
        with self.assertRaises(ValueError):
            decision_metrics([0, 1], [0, 3])

    def test_feature_decoder_preserves_all_history_and_latest_ta(self):
        self.assertEqual(feature_name(0), ('log_open_previous_close', 256, 'OHLCV'))
        self.assertEqual(feature_name(4863), ('log_long_short_ratio', 1, 'market'))
        self.assertEqual(feature_name(4864), ('momentum_4', 1, 'technical'))
        self.assertEqual(feature_name(4908), ('threshold_side', 0, 'threshold'))
        with self.assertRaises(ValueError):
            feature_name(4909)

    def test_shift_bins_are_fitted_from_train_and_capture_tail_departures(self):
        train = np.linspace(0, 1, 100)
        same = distribution_shift(train, train)
        shifted = distribution_shift(train, np.full(30, 2.))
        self.assertEqual(same['psi'], 0)
        self.assertGreater(shifted['psi'], same['psi'])
        self.assertEqual(shifted['outside_train_range'], 1)
        self.assertEqual(shifted['outside_train_01_99'], 1)

    def test_correlations_report_sensitivity_without_an_iid_p_value(self):
        result = correlation_summary(np.arange(15), np.arange(15))
        self.assertAlmostEqual(result['spearman'], 1)
        self.assertAlmostEqual(result['leave_one_out_min'], 1)
        self.assertNotIn('p_value', result)
        self.assertIsNone(correlation_summary(np.ones(15), np.arange(15))['spearman'])


if __name__ == '__main__':
    unittest.main()
