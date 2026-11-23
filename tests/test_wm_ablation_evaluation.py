import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_ablation_evaluation import (
    BAR_INTERVAL,
    build_future_targets,
    evaluate_predictions,
    holm_adjust,
    make_walk_forward_folds,
    paired_block_bootstrap,
    split_mask,
    variance_qlike,
)


class WMInputAblationTargetsTest(unittest.TestCase):
    def test_calendar_folds_and_strict_boundary_purge(self):
        folds = make_walk_forward_folds()
        self.assertEqual(len(folds), 15)
        first, last = folds[0], folds[-1]
        self.assertEqual(first.train_start, pd.Timestamp("2020-10-01", tz="UTC"))
        self.assertEqual(first.train_end, pd.Timestamp("2022-10-01", tz="UTC"))
        self.assertEqual(first.val_end, first.test_start)
        self.assertEqual(last.test_start, pd.Timestamp("2026-07-01", tz="UTC"))
        self.assertEqual(last.test_end, pd.Timestamp("2026-10-01", tz="UTC"))
        for previous, following in zip(folds, folds[1:]):
            self.assertEqual(previous.test_end, following.test_start)
        index = pd.date_range("2022-12-31", "2023-01-02", freq="15min", tz="UTC")
        targets = build_future_targets(index, np.ones(len(index)), horizons=(4, 64))
        val = split_mask(index, targets, first, "val")
        # The outcome close at exactly test_start is deliberately excluded.
        self.assertFalse(val[index.get_loc("2022-12-31 08:00:00+00:00")])
        self.assertTrue(val[index.get_loc("2022-12-31 07:45:00+00:00")])
        test = split_mask(index, targets, first, "test", context_valid=np.ones(len(index), bool))
        self.assertTrue(test[index.get_loc("2023-01-01 00:00:00+00:00")])
        self.assertFalse(test[index.get_loc("2022-12-31 23:45:00+00:00")])

    def test_future_labels_use_reference_close_and_future_closes_only(self):
        index = pd.date_range("2023-01-01", periods=6, freq="15min", tz="UTC")
        close = np.array([100.0, 110.0, 99.0, 108.0, 100.0, 101.0])
        targets = build_future_targets(index, close, horizons=(2,))
        # Decision t=1 sees 100, then future closes 110,99: 1% adverse
        # from its reference, versus 10% intra-window peak-to-trough DD.
        self.assertAlmostEqual(targets.returns[1, 0], np.log(99 / 100))
        self.assertAlmostEqual(targets.variance[1, 0], np.log(110 / 100) ** 2 + np.log(99 / 110) ** 2)
        self.assertAlmostEqual(targets.downside[1, 0], 0.01)
        self.assertEqual(targets.crash[1, 0], 0.0)
        self.assertEqual(targets.label_end[1, 0], np.datetime64("2023-01-01T00:45:00"))
        self.assertFalse(targets.mask[0, 0])
        self.assertFalse(targets.mask[-1, 0])
        changed = close.copy()
        changed[3:] = 999.0
        other = build_future_targets(index, changed, horizons=(2,))
        self.assertEqual(targets.returns[1, 0], other.returns[1, 0])

    def test_missing_future_or_reference_invalidates_full_path(self):
        index = pd.date_range("2023-01-01", periods=9, freq="15min", tz="UTC")
        observed = np.ones(len(index), dtype=bool)
        observed[3] = False
        targets = build_future_targets(index, np.ones(len(index)), observed, horizons=(2,))
        np.testing.assert_array_equal(targets.mask[:, 0], [False, True, False, False, False, True, True, True, False])
        self.assertTrue(np.isnan(targets.variance[2, 0]))
        with self.assertRaisesRegex(ValueError, "complete 15-minute grid"):
            build_future_targets(index.delete(3), np.ones(len(index) - 1))

    def test_variance_baseline_is_scaled_past16_not_future(self):
        index = pd.date_range("2023-01-01", periods=90, freq="15min", tz="UTC")
        close = np.exp(np.arange(len(index)) * 0.01)
        targets = build_future_targets(index, close)
        np.testing.assert_allclose(targets.baseline_variance[17], np.array([4, 16, 64]) * 0.01 ** 2)
        self.assertTrue(np.isnan(targets.baseline_variance[16]).all())
        close[18:] *= 10
        changed = build_future_targets(index, close)
        np.testing.assert_array_equal(targets.baseline_variance[17], changed.baseline_variance[17])


class WMInputAblationMetricsTest(unittest.TestCase):
    @staticmethod
    def predictions():
        return pd.DataFrame({
            "actual_return": [-0.02, -0.01, 0.01, 0.02],
            "pred_return": [-0.01, 0.0, 0.0, 0.01],
            "actual_variance": [1e-4, 2e-4, 1e-4, 2e-4],
            "pred_variance": [1e-4, 2e-4, 1e-4, 2e-4],
            "actual_downside": [0.02, 0.01, 0, 0],
            "pred_downside": [0.01, 0.01, 0, 0],
            "actual_crash": [1, 0, 0, 0],
            "pred_crash": [0.8, 0.2, 0.1, 0.1],
        })

    def test_interpretable_metrics_and_zero_return_baseline(self):
        frame = self.predictions()
        scores = evaluate_predictions(frame)
        self.assertEqual(scores["n"], 4)
        self.assertAlmostEqual(scores["return_rmse"], 0.01)
        self.assertAlmostEqual(scores["return_oos_r2"], 0.6)
        self.assertEqual(scores["variance_qlike"], 0)
        self.assertEqual(scores["crash_pr_auc"], 1)
        self.assertAlmostEqual(scores["crash_brier"], 0.025)
        frame["pred_return"] = 0.0
        scores = evaluate_predictions(frame)
        self.assertEqual(scores["return_oos_r2"], 0)
        self.assertIsNone(scores["return_spearman_ic"])

    def test_qlike_uses_variance_and_fixed_numerical_floor(self):
        np.testing.assert_allclose(variance_qlike(np.array([4.0]), np.array([2.0])), [1 - np.log(2)])
        self.assertEqual(variance_qlike(np.array([0.0]), np.array([0.0]))[0], 0)
        with self.assertRaises(ValueError):
            variance_qlike(np.array([-1.0]), np.array([1.0]))

    def test_same_finite_rows_and_undefined_r2(self):
        frame = self.predictions()
        frame.loc[0, "pred_downside"] = np.nan
        frame["actual_return"] = 0.0
        scores = evaluate_predictions(frame)
        self.assertEqual(scores["n"], 3)
        self.assertEqual(scores["n_nonfinite_excluded"], 1)
        self.assertIsNone(scores["return_oos_r2"])


class WMInputAblationBootstrapTest(unittest.TestCase):
    @staticmethod
    def predictions(days=35, seeds=(0, 1)):
        # One daily-changing shock is repeated on all 96 intraday rows.
        index = pd.date_range("2023-01-01", periods=days * 96, freq="15min", tz="UTC")
        day = np.arange(len(index)) // 96
        y = 0.01 * np.where(day % 2, -1, 1)
        parts = []
        for group in ("A", "B"):
            for seed in seeds:
                pred = np.zeros(len(index)) if group == "A" else y * 0.5
                parts.append(pd.DataFrame({
                    "timestamp": index, "fold": 0, "seed": seed,
                    "group": group, "horizon": 16,
                    "actual_return": y, "pred_return": pred,
                    "actual_variance": np.full(len(index), 0.0001),
                    "pred_variance": np.full(len(index), 0.0002 if group == "A" else 0.0001),
                }))
        return pd.concat(parts, ignore_index=True)

    def test_pooled_paired_rmse_and_seed_count_not_independent_samples(self):
        frame = self.predictions()
        result = paired_block_bootstrap(frame, replicates=49)
        self.assertEqual([part["block_days"] for part in result], [1, 7, 28])
        for part in result:
            self.assertEqual(part["n_paired_timestamps"], 35 * 96)
            self.assertEqual(part["n_seeds"], 2)
            self.assertAlmostEqual(part["metrics"]["return_rmse_improvement_pct"]["estimate"], 50)
            self.assertAlmostEqual(part["metrics"]["return_oos_r2_difference"]["estimate"], 0.75)
            self.assertGreater(part["metrics"]["variance_qlike_improvement"]["estimate"], 0)
        one_seed = paired_block_bootstrap(frame[frame["seed"] == 0], replicates=49)
        self.assertEqual(result[1]["metrics"], one_seed[1]["metrics"])
        self.assertEqual(result, paired_block_bootstrap(frame, replicates=49))

    def test_day_blocks_cannot_bridge_data_gaps_or_fold_boundaries(self):
        frame = self.predictions(days=6)
        frame = frame[frame["timestamp"] != pd.Timestamp("2023-01-03 12:00", tz="UTC")].copy()
        frame.loc[frame["timestamp"] >= pd.Timestamp("2023-01-05", tz="UTC"), "fold"] = 1
        part = paired_block_bootstrap(frame, block_days=(7,), replicates=29)[0]
        self.assertEqual(part["n_segments"], 3)
        self.assertEqual(part["n_segments_shorter_than_block"], 3)
        self.assertFalse(part["inference_available"])
        self.assertIsNone(part["metrics"]["return_mse_improvement"]["p_value"])
        self.assertIsNone(part["metrics"]["return_mse_improvement"]["ci_low"])

    def test_week_blocks_preserve_serially_dependent_daily_losses(self):
        frame = self.predictions(days=84, seeds=(0,))
        decision_day = (frame["timestamp"] - frame["timestamp"].min()).dt.days.to_numpy()
        use = frame["group"].to_numpy() == "B"
        # Two-week good/bad regimes: consecutive losses share the same shock.
        gain = np.where((decision_day // 14) % 2, 0.2, 0.8)
        frame.loc[use, "pred_return"] = frame.loc[use, "actual_return"].to_numpy() * gain[use]
        result = paired_block_bootstrap(frame, block_days=(1, 7), replicates=799)
        intervals = [part["metrics"]["return_mse_improvement"] for part in result]
        self.assertGreater(intervals[1]["ci_high"] - intervals[1]["ci_low"],
                           intervals[0]["ci_high"] - intervals[0]["ci_low"])

    def test_pairing_rejects_different_test_rows_or_labels(self):
        frame = self.predictions(days=2)
        with self.assertRaisesRegex(ValueError, "identical timestamp"):
            paired_block_bootstrap(frame.iloc[1:], replicates=9)
        frame.loc[frame["group"] == "B", "actual_return"] += 0.001
        with self.assertRaisesRegex(ValueError, "identical outcomes"):
            paired_block_bootstrap(frame, replicates=9)

    def test_holm_preserves_four_test_family(self):
        adjusted = holm_adjust({"return_BA": 0.01, "risk_BA": 0.03, "return_DC": 0.02, "risk_DC": 0.20})
        self.assertEqual(adjusted, {"return_BA": 0.04, "risk_BA": 0.06, "return_DC": 0.06, "risk_DC": 0.20})
        self.assertEqual(holm_adjust({"a": 0.01, "b": None})["a"], 0.02)


if __name__ == "__main__":
    unittest.main()
