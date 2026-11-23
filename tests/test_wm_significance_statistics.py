from __future__ import annotations

import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from unidream.experiments.wm_full_grid_statistics import _block_sums
from unidream.experiments.wm_significance_statistics import (
    compile_rank_backend, holm_fixed_family, infer_panel, moving_day_weights,
    weighted_spearman,
)


def panel(days=16):
    rng = np.random.default_rng(17)
    n = days * 96
    y = rng.normal(size=n)
    label = np.arange(n) % 3
    return pd.DataFrame({"fold": np.repeat(np.arange(2), n // 2),
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "actual": y, "label": label, "prediction": y + .1 * rng.normal(size=n),
        "baseline_prediction": 0., "predicted_class": label,
        "baseline_class": np.zeros(n, dtype=int)})


class ExactRankTests(unittest.TestCase):
    def test_frequency_mid_ranks_match_expanded_scipy_with_ties_and_zeros(self):
        actual = np.array([2., 1., 2., 3., 4., 4.])
        predicted = np.array([1., 3., 3., 0., 2., 2.])
        day = np.array([0, 0, 1, 1, 2, 2])
        weights = np.array([[1, 1, 1], [2, 0, 3], [0, 0, 4], [0, 2, 0], [0, 0, 0]])
        got = weighted_spearman(actual, predicted, day, weights)
        expected = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sample in weights:
                w = sample[day]
                expected.append(spearmanr(np.repeat(actual, w), np.repeat(predicted, w)).statistic)
        np.testing.assert_allclose(got, expected, atol=1e-14, equal_nan=True)

    @unittest.skipUnless(shutil.which("clang++"), "local C++ compiler unavailable")
    def test_compiled_exact_backend_matches_python_and_cache(self):
        rng = np.random.default_rng(19)
        y = rng.integers(0, 7, 45).astype(float)
        prediction = rng.integers(0, 9, 45).astype(float)
        day = np.arange(45) % 9
        weights = rng.integers(0, 5, (21, 9))
        expected = weighted_spearman(y, prediction, day, weights)
        with tempfile.TemporaryDirectory() as td:
            backend = compile_rank_backend(td)
            got = weighted_spearman(y, prediction, day, weights, backend_path=backend, cache_dir=td)
            cached = weighted_spearman(y, prediction, day, weights, backend_path=backend, cache_dir=td)
        np.testing.assert_allclose(got, expected, atol=5e-15, equal_nan=True)
        np.testing.assert_array_equal(got, cached)

    def test_invalid_fractional_weights_rejected(self):
        with self.assertRaisesRegex(ValueError, "integers"):
            weighted_spearman([1, 2], [2, 1], [0, 1], [[.5, 1]])

    def test_constant_rank_input_is_undefined(self):
        got = weighted_spearman([1, 2], [1, 1], [0, 1], [[1, 1], [2, 1]])
        self.assertTrue(np.isnan(got).all())


class MovingBlockTests(unittest.TestCase):
    def test_weight_draws_exactly_match_existing_non_circular_truncated_block_sums(self):
        rng = np.random.default_rng(31)
        values = rng.normal(size=(31, 4))
        weights = moving_day_weights([np.arange(31)], n_days=31, block_days=7, replicates=25,
                                     rng=np.random.default_rng(12))
        old = _block_sums(values, 7, 25, np.random.default_rng(12))
        np.testing.assert_allclose(weights @ values, old, atol=1e-14)
        np.testing.assert_array_equal(weights.sum(axis=1), 31)

    def test_folds_gaps_and_short_segments_preserve_separate_support(self):
        pieces = [np.arange(0, 4), np.arange(4, 15), np.arange(15, 22)]
        weights = moving_day_weights(pieces, n_days=22, block_days=7, replicates=23,
                                     rng=np.random.default_rng(7))
        np.testing.assert_array_equal(weights[:, :4], 1)
        np.testing.assert_array_equal(weights[:, 15:], 1)
        np.testing.assert_array_equal(weights[:, 4:15].sum(axis=1), 11)
        self.assertGreater(np.ptp(weights[:, 4:15], axis=0).sum(), 0)

    def test_segment_partition_must_be_exact(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            moving_day_weights([np.array([0, 1]), np.array([1, 2])], n_days=3,
                               block_days=1, replicates=4, rng=np.random.default_rng(1))


class PanelInferenceTests(unittest.TestCase):
    def test_paired_native_and_ba_and_absolute_cis(self):
        rows = infer_panel(panel(), task="return", blocks=(1, 7, 28), replicates=41,
                           metrics=("balanced_accuracy", "zero_oos_r2", "pearson_ic"))
        ba = next(r for r in rows if r["metric"] == "balanced_accuracy" and r["block_days"] == 7)
        self.assertEqual(ba["model_estimate"], 1.)
        self.assertAlmostEqual(ba["baseline_estimate"], 1 / 3)
        self.assertAlmostEqual(ba["improvement"], 2 / 3)
        self.assertEqual(ba["n_rows"], 16 * 96)
        self.assertEqual(ba["chance_baseline"], 1 / 3)
        self.assertGreater(ba["n_valid_replicates"], 1)
        r2 = next(r for r in rows if r["metric"] == "zero_oos_r2" and r["block_days"] == 7)
        self.assertGreater(r2["improvement"], .95)
        self.assertGreater(r2["improvement_ci_low"], 0)
        pearson = next(r for r in rows if r["metric"] == "pearson_ic" and r["block_days"] == 7)
        self.assertGreater(pearson["model_ci_low"], .9)
        self.assertIsNone(pearson["baseline_estimate"])
        self.assertIsNone(pearson["p_value"])
        long = [r for r in rows if r["block_days"] == 28]
        self.assertTrue(all(not r["inference_available"] for r in long))
        self.assertTrue(all(r["model_ci_low"] is None for r in long))

    def test_identical_predictions_are_zero_paired_difference(self):
        data = panel()
        data["baseline_prediction"] = data.prediction
        data["baseline_class"] = data.predicted_class
        rows = infer_panel(data, task="advantage", blocks=(1,), replicates=17)
        self.assertTrue(all(abs(r["improvement"]) < 1e-14 for r in rows))
        self.assertTrue(all(r["p_value"] == 1 for r in rows))

    def test_fixed_class_universe_never_silently_excludes_missing_class(self):
        data = panel()
        data["label"] = data.predicted_class = data.baseline_class = 0
        rows = infer_panel(data, task="return", blocks=(1,), replicates=12,
                           metrics=("balanced_accuracy",))
        self.assertIsNone(rows[0]["model_estimate"])
        self.assertEqual(rows[0]["n_valid_replicates"], 0)
        self.assertIsNone(rows[0]["p_value"])

    def test_gap_splits_day_segments_without_dropping_any_rows(self):
        data = panel().drop([10, 11])
        rows = infer_panel(data, task="return", blocks=(7,), replicates=12,
                           metrics=("balanced_accuracy",))
        row = rows[0]
        self.assertEqual(row["n_rows"], len(data))
        self.assertEqual(row["n_segments"], 3)
        self.assertGreater(row["n_fixed_rows"], 0)
        self.assertEqual(row["n_fixed_rows"] + row["n_resampleable_rows"], len(data))

    def test_overlap_duplicate_and_nonfinite_are_errors(self):
        original = panel()
        for bad in (pd.concat([original, original.iloc[:1]], ignore_index=True),
                    original.assign(actual=np.nan)):
            with self.assertRaises(ValueError):
                infer_panel(bad, task="return", blocks=(1,), replicates=3)

    def test_class_only_panel_and_per_block_replicate_counts(self):
        data = panel().drop(columns=["prediction", "baseline_prediction"])
        rows = infer_panel(data, task="return", blocks=(1, 7), replicates=10,
                           replicates_by_block={7: 21}, metrics=("balanced_accuracy",))
        self.assertEqual([r["replicates"] for r in rows], [10, 21])
        self.assertEqual({r["metric"] for r in rows}, {"balanced_accuracy"})

    def test_return_zero_baseline_has_exact_zero_oos_r2(self):
        rows = infer_panel(panel(), task="return", blocks=(1,), replicates=10,
                           metrics=("zero_oos_r2",))
        self.assertEqual(rows[0]["baseline_estimate"], 0.)
        self.assertEqual(rows[0]["baseline_ci_low"], 0.)
        self.assertEqual(rows[0]["baseline_ci_high"], 0.)

    def test_qlike_improvement_is_positive_when_model_loss_is_lower(self):
        data = panel()
        data["actual"] = np.exp(data.actual) * 1e-5
        data["prediction"] = data.actual
        data["baseline_prediction"] = 1e-5
        rows = infer_panel(data, task="variance", blocks=(1,), replicates=13, metrics=("qlike",))
        self.assertAlmostEqual(rows[0]["model_estimate"], 0., places=12)
        self.assertGreater(rows[0]["baseline_estimate"], 0)
        self.assertGreater(rows[0]["improvement"], 0)
        self.assertGreater(rows[0]["improvement_ci_low"], 0)

    def test_brier_requires_actual_events_and_probability_forecasts(self):
        data = panel()
        data["actual"] = data["label"] = np.arange(len(data)) % 2
        data["prediction"] = .1 + .8 * data.actual
        data["baseline_prediction"] = .3
        data["predicted_class"] = data.actual
        rows = infer_panel(data, task="crash", blocks=(1,), replicates=13,
                           metrics=("brier", "balanced_accuracy"))
        brier = next(row for row in rows if row["metric"] == "brier")
        self.assertAlmostEqual(brier["model_estimate"], .01)
        self.assertGreater(brier["improvement"], 0)
        data["prediction"] = 1.1
        with self.assertRaisesRegex(ValueError, "probabilities"):
            infer_panel(data, task="crash", blocks=(1,), replicates=3)


class MultiplicityTests(unittest.TestCase):
    def test_missing_hypotheses_and_aliases_do_not_reduce_search_family(self):
        adjusted = holm_fixed_family({"a": .0001, "alias_a": .0001, "undefined": None}, family_size=336)
        self.assertAlmostEqual(adjusted["a"], .0336)
        self.assertAlmostEqual(adjusted["alias_a"], .0336)
        self.assertIsNone(adjusted["undefined"])

    def test_family_cannot_be_smaller_than_reported_hypotheses(self):
        with self.assertRaisesRegex(ValueError, "cover"):
            holm_fixed_family({"a": .1, "b": .2}, family_size=1)


if __name__ == "__main__":
    unittest.main()
