"""Frozen baseline access/fit-cohort/scoring contracts."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_ablation_baselines import (
    even_fit_indices, ridge_design, score_predictions,
)


class TestWMAblationBaselines(unittest.TestCase):
    def test_fit_subset_is_deterministic_unique_and_train_only(self):
        train = np.arange(10_000, 40_000, 3)
        selected = even_fit_indices(train, 8192)
        np.testing.assert_array_equal(selected, even_fit_indices(train, 8192))
        self.assertEqual(len(selected), 8192)
        self.assertEqual(len(np.unique(selected)), 8192)
        self.assertEqual(selected[0], train[0])
        self.assertEqual(selected[-1], train[-1])
        self.assertTrue(np.isin(selected, train).all())
        np.testing.assert_array_equal(even_fit_indices(train[:3], 8192), train[:3])

    def test_recipe_has_expected_dimensions_and_exact_channel_order(self):
        windows = np.arange(2 * 256 * 63, dtype=np.float32).reshape(2, 256, 63)
        expected_widths = {"A": 1280, "B": 1324, "C": 1294, "D": 1338}
        for group, width in expected_widths.items():
            design = ridge_design(windows, group)
            self.assertEqual(design.shape, (2, width))
            np.testing.assert_array_equal(design[:, :1280], windows[:, :, :5].reshape(2, -1))
        np.testing.assert_array_equal(ridge_design(windows, "B")[:, 1280:], windows[:, -1, 19:])
        np.testing.assert_array_equal(ridge_design(windows, "C")[:, 1280:], windows[:, -1, 5:19])
        np.testing.assert_array_equal(ridge_design(windows, "D")[:, 1280:], windows[:, -1, 5:])

    def test_linear_access_uses_only_last_extra_token(self):
        original = np.ones((1, 256, 63), dtype=np.float32)
        changed = original.copy()
        changed[:, :-1, 5:] = 2
        for group in "ABCD":
            np.testing.assert_array_equal(ridge_design(original, group), ridge_design(changed, group))
        changed[:, 0, :5] = 3
        self.assertFalse(np.array_equal(ridge_design(original, "A"), ridge_design(changed, "A")))

    def test_zero_return_is_zero_oos_r2_and_constant_ic_undefined(self):
        frame = pd.DataFrame({"actual_return": [.1, -.2], "pred_return": [0., 0.],
                              "actual_variance": [np.nan, np.nan], "pred_variance": [np.nan, np.nan]})
        result = score_predictions(frame)
        self.assertEqual(result["return_oos_r2"], 0)
        self.assertIsNone(result["return_spearman_ic"])
        self.assertEqual(result["n_return"], 2)
        self.assertEqual(result["n_variance"], 0)
        self.assertIsNone(result["variance_qlike"])

    def test_variance_baseline_is_scored_without_fake_return_predictions(self):
        frame = pd.DataFrame({"actual_return": [.1, -.2], "pred_return": [np.nan, np.nan],
                              "actual_variance": [1., 4.], "pred_variance": [1., 4.]})
        result = score_predictions(frame)
        self.assertEqual(result["variance_qlike"], 0)
        self.assertEqual(result["n_variance"], 2)
        self.assertEqual(result["n_return"], 0)
        self.assertIsNone(result["return_rmse"])


if __name__ == "__main__":
    unittest.main()
