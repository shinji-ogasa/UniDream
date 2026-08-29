"""Regression tests for gap-aware sequence eligibility."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from unidream.data.window_quality import (
    WindowQualityError,
    valid_sequence_starts,
    window_is_gap_free,
)


class WindowQualityTest(unittest.TestCase):
    def test_gap_excludes_only_windows_crossing_the_missing_bar(self) -> None:
        full = pd.date_range("2018-01-01", periods=8, freq="15min")
        index = full.delete(4)
        starts = valid_sequence_starts(index, 3)
        # The break is between rows 3 and 4.  The first two windows end before
        # it and the final window starts after it; starts 2 and 3 cross it.
        np.testing.assert_array_equal(starts, np.array([0, 1, 4], dtype=np.int64))
        self.assertTrue(window_is_gap_free(index, 0, 3))
        self.assertFalse(window_is_gap_free(index, 2, 3))

    def test_v4_observation_mask_rejects_false_rows_without_compaction(self) -> None:
        index = pd.date_range("2018-01-01", periods=6, freq="15min")
        observed = np.array([True, True, False, True, True, True], dtype=bool)
        np.testing.assert_array_equal(
            valid_sequence_starts(index, 2, spot_bar_observed=observed),
            np.array([0, 3, 4], dtype=np.int64),
        )

    def test_duplicate_unsorted_and_bad_masks_fail_closed(self) -> None:
        index = pd.date_range("2018-01-01", periods=4, freq="15min")
        with self.assertRaisesRegex(WindowQualityError, "duplicate"):
            valid_sequence_starts(index.insert(2, index[2]), 2)
        with self.assertRaisesRegex(WindowQualityError, "not strictly increasing"):
            valid_sequence_starts(index[[0, 2, 1, 3]], 2)
        with self.assertRaisesRegex(WindowQualityError, "boolean dtype"):
            valid_sequence_starts(index, 2, spot_bar_observed=np.ones(len(index), dtype=np.int8))

    def test_short_and_singleton_windows_have_explicit_offsets(self) -> None:
        index = pd.date_range("2018-01-01", periods=2, freq="15min")
        np.testing.assert_array_equal(valid_sequence_starts(index, 3), np.empty(0, dtype=np.int64))
        np.testing.assert_array_equal(valid_sequence_starts(index, 1), np.array([0, 1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
