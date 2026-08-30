from __future__ import annotations

import unittest

import numpy as np

from unidream.experiments.chronological_oof import (
    ChronologicalOOFError,
    conditional_path_enabled,
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

    def test_origin_eligibility_is_independent_of_own_target_mask_or_value(self) -> None:
        features = np.arange(10, dtype=np.float64).reshape(-1, 1)
        labels = np.arange(10, dtype=np.float64)
        valid_target_mask = np.ones(10, dtype=bool)
        baseline = chronological_oof_predict(
            features,
            labels,
            fit_predict=self._fit_predict,
            min_train_size=2,
            valid_target_mask=valid_target_mask,
        )

        masked_target = valid_target_mask.copy()
        masked_target[5] = False
        masked = chronological_oof_predict(
            features,
            labels,
            fit_predict=self._fit_predict,
            min_train_size=2,
            valid_target_mask=masked_target,
        )

        nan_target = labels.copy()
        nan_target[5] = np.nan
        nan_value = chronological_oof_predict(
            features,
            nan_target,
            fit_predict=self._fit_predict,
            min_train_size=2,
            valid_target_mask=valid_target_mask,
        )

        # Row 5 is decision-time eligible from its finite feature, even when
        # its future target mask/value is unavailable.  The own row is never
        # admitted to its training prefix, so its OOF state is invariant.
        for changed in (masked, nan_value):
            self.assertTrue(changed["prediction_eligibility_mask"][5])
            self.assertTrue(changed["prediction_mask"][5])
            self.assertEqual(baseline["train_count"][5], changed["train_count"][5])
            np.testing.assert_array_equal(
                baseline["predictions"][5], changed["predictions"][5]
            )
        self.assertEqual(
            baseline["provenance"]["prediction_eligibility"]["count"],
            masked["provenance"]["prediction_eligibility"]["count"],
        )
        self.assertEqual(
            masked["provenance"]["training_label_eligibility"]["count"],
            nan_value["provenance"]["training_label_eligibility"]["count"],
        )
        # The changed row is available to later training prefixes only in the
        # baseline; this is the intended delayed effect of label availability.
        self.assertNotEqual(
            baseline["predictions"][6, 0], masked["predictions"][6, 0]
        )
        self.assertNotEqual(
            baseline["predictions"][6, 0], nan_value["predictions"][6, 0]
        )

        incomplete_tail_mask = valid_target_mask.copy()
        incomplete_tail_mask[-1] = False
        incomplete_tail = chronological_oof_predict(
            features,
            labels,
            fit_predict=self._fit_predict,
            min_train_size=2,
            valid_target_mask=incomplete_tail_mask,
        )
        # Target-completeness for scoring is a downstream contract; it must not
        # suppress a decision-time prediction at an otherwise valid tail row.
        self.assertTrue(incomplete_tail["prediction_eligibility_mask"][-1])
        self.assertTrue(incomplete_tail["prediction_mask"][-1])

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

    def test_masks_require_strict_bool_dtype(self) -> None:
        features = np.ones((4, 1), dtype=np.float64)
        targets = np.ones(4, dtype=np.float64)
        invalid_masks = (
            np.asarray([1, 0, 1, 0], dtype=np.int64),
            np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray(["true", "false", "true", "false"]),
            np.asarray([True, np.nan, True, False], dtype=object),
        )
        for invalid in invalid_masks:
            with self.subTest(dtype=invalid.dtype):
                with self.assertRaises(ChronologicalOOFError):
                    chronological_oof_predict(
                        features,
                        targets,
                        fit_predict=self._fit_predict,
                        valid_target_mask=invalid,
                    )
                with self.assertRaises(ChronologicalOOFError):
                    chronological_oof_standardize(
                        np.ones((4, 1), dtype=np.float64),
                        invalid,
                    )
                with self.assertRaises(ChronologicalOOFError):
                    validate_oof_result(
                        {
                            "predictions": np.ones((4, 1), dtype=np.float64),
                            "prediction_mask": invalid,
                            "provenance": {"in_sample": False},
                        }
                    )

    def test_row_eligibility_blocks_origins_and_training_rows(self) -> None:
        features = np.arange(12, dtype=np.float64).reshape(-1, 1)
        features[4, 0] = np.nan
        labels = np.arange(12, dtype=np.float64)
        row_mask = np.ones(12, dtype=bool)
        row_mask[[3, 8]] = False
        calls: list[int] = []

        def callback(x_train, y_train, x_test):
            calls.append(int(x_test[0, 0]))
            return {"prediction": [float(np.mean(y_train))]}

        first = chronological_oof_predict(
            features,
            labels,
            fit_predict=callback,
            min_train_size=2,
            row_eligibility_mask=row_mask,
            row_eligibility_provenance={"source": "p0_a_availability"},
        )
        self.assertNotIn(3, calls)
        self.assertNotIn(4, calls)  # non-finite feature row is unavailable
        self.assertNotIn(8, calls)
        self.assertFalse(first["prediction_mask"][3])
        self.assertFalse(first["prediction_mask"][4])
        self.assertFalse(first["prediction_mask"][8])
        self.assertTrue(np.isnan(first["predictions"][3]).all())
        self.assertTrue(first["provenance"]["row_eligibility_mask_supplied"])
        self.assertEqual(first["provenance"]["row_eligibility_source"], "p0_a_availability")
        self.assertEqual(
            first["provenance"]["row_eligibility_provenance"]["source"],
            "p0_a_availability",
        )

        perturbed = labels.copy()
        perturbed[8] = 100_000.0
        second = chronological_oof_predict(
            features,
            perturbed,
            fit_predict=callback,
            min_train_size=2,
            row_eligibility_mask=row_mask,
        )
        np.testing.assert_array_equal(first["prediction_mask"], second["prediction_mask"])
        usable = first["prediction_mask"]
        np.testing.assert_allclose(first["predictions"][usable], second["predictions"][usable])

    def test_integer_and_conditional_options_are_strict(self) -> None:
        features = np.ones((6, 1), dtype=np.float64)
        targets = np.ones(6, dtype=np.float64)
        invalid_integer_options = {
            "horizon": 1.0,
            "purge": "0",
            "min_train_size": True,
            "train_window": 2.5,
            "step": "1",
        }
        for name, value in invalid_integer_options.items():
            with self.subTest(option=name):
                with self.assertRaises(ChronologicalOOFError):
                    chronological_oof_predict(
                        features,
                        targets,
                        fit_predict=self._fit_predict,
                        **{name: value},
                    )
        with self.assertRaises(ChronologicalOOFError):
            chronological_oof_predict(
                features,
                targets,
                fit_predict=self._fit_predict,
                target_end=np.asarray([1.0] * 6),
            )
        with self.assertRaises(ChronologicalOOFError):
            chronological_oof_standardize(
                np.ones((6, 1), dtype=np.float64),
                np.ones(6, dtype=bool),
                min_history="1",
            )
        with self.assertRaises(ChronologicalOOFError):
            conditional_path_enabled({"conditional_oracle_path": "false"})
        with self.assertRaises(ChronologicalOOFError):
            conditional_path_enabled({"conditional_oracle": {"enabled": "false"}})


if __name__ == "__main__":
    unittest.main()
