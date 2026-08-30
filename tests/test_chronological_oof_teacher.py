from __future__ import annotations

import unittest

import numpy as np

from unidream.experiments.chronological_oof import (
    ChronologicalOOFError,
    chronological_oof_predict,
    chronological_oof_standardize,
    validate_oof_result,
)


class ChronologicalOOFTeacherTest(unittest.TestCase):
    @staticmethod
    def _fit_predict(x_train, y_train, x_test):
        # Deliberately simple teacher: the test row is not used as a target.
        # Returning metadata exercises the provenance channel used for
        # normalizer/calibrator/teacher-weight hashes in a real fit.
        return {
            "prediction": np.asarray([float(np.mean(y_train))]),
            "metadata": {"fit_scheme": "chronological_oof", "teacher_weight": "prefix"},
        }

    def test_early_rows_are_unavailable_and_same_row_label_cannot_change_oof_state(self) -> None:
        features = np.arange(10, dtype=np.float64).reshape(-1, 1)
        labels = (features[:, 0] * 2.0 + 1.0).astype(np.float64)
        first = chronological_oof_predict(
            features,
            labels,
            fit_predict=self._fit_predict,
            horizon=1,
            purge=0,
            min_train_size=3,
        )

        perturbed_labels = labels.copy()
        perturbed_labels[6] += 10_000.0
        second = chronological_oof_predict(
            features,
            perturbed_labels,
            fit_predict=self._fit_predict,
            horizon=1,
            purge=0,
            min_train_size=3,
        )

        self.assertTrue(np.array_equal(first["prediction_mask"][:3], [False, False, False]))
        self.assertTrue(np.isnan(first["predictions"][:3]).all())
        self.assertTrue(first["prediction_mask"][6])
        # y[6] cannot be in the training prefix for prediction 6; it may affect
        # later rows, which is the intended chronological behavior.
        np.testing.assert_array_equal(first["predictions"][6], second["predictions"][6])
        self.assertNotEqual(first["predictions"][7, 0], second["predictions"][7, 0])
        for origin in first["origins"]:
            self.assertLessEqual(
                origin["train_end_exclusive"], origin["prediction_index"]
            )
        self.assertFalse(first["provenance"]["in_sample"])
        self.assertEqual(first["metadata_by_row"][6]["fit_scheme"], "chronological_oof")

    def test_horizon_and_purge_exclude_incomplete_or_overlapping_labels(self) -> None:
        features = np.arange(20, dtype=np.float64).reshape(-1, 1)
        labels = np.arange(20, dtype=np.float64)
        result = chronological_oof_predict(
            features,
            labels,
            fit_predict=self._fit_predict,
            horizon=4,
            purge=2,
            min_train_size=2,
        )
        for origin in result["origins"]:
            t = origin["prediction_index"]
            cutoff = t - 2
            train_indices = np.arange(origin["train_start"], origin["train_end_exclusive"])
            self.assertTrue(np.all(train_indices + 4 <= cutoff))

    def test_expanding_oof_standardizer_does_not_self_normalize_or_fill_early_rows(self) -> None:
        values = np.asarray([[np.nan], [np.nan], [1.0], [3.0], [5.0]], dtype=np.float64)
        mask = np.asarray([False, False, True, True, True])
        result = chronological_oof_standardize(values, mask, min_history=1)
        self.assertFalse(result["mask"][0])
        self.assertFalse(result["mask"][2])  # no prior OOF state yet
        self.assertTrue(result["mask"][3])
        self.assertEqual(float(result["values"][3, 0]), 2.0)
        self.assertTrue(np.isnan(result["values"][0]).all())
        self.assertEqual(result["provenance"]["normalizer"], "expanding_prefix")

    def test_standardizer_rejects_finite_values_outside_oof_mask(self) -> None:
        with self.assertRaises(ChronologicalOOFError):
            chronological_oof_standardize(
                np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64),
                np.asarray([False, True]),
            )

    def test_oof_validator_rejects_partial_finite_values_outside_mask(self) -> None:
        with self.assertRaises(ChronologicalOOFError):
            validate_oof_result(
                {
                    "predictions": np.asarray([[1.0, np.nan], [np.nan, np.nan]]),
                    "prediction_mask": np.asarray([False, False]),
                    "provenance": {"in_sample": False},
                }
            )

    def test_invalid_callback_shape_fails_closed(self) -> None:
        with self.assertRaises(ChronologicalOOFError):
            chronological_oof_predict(
                np.ones((5, 1)),
                np.ones(5),
                fit_predict=lambda x_train, y_train, x_test: np.asarray([1.0, 2.0]),
                min_train_size=1,
            )


if __name__ == "__main__":
    unittest.main()
