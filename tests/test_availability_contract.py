"""P0-A regression tests for sidecar propagation and canonical observations."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from unidream.data.dataset import SequenceDataset, WFODataset, WFOSplit


def _split(index: pd.DatetimeIndex) -> WFOSplit:
    return WFOSplit(
        fold_idx=0,
        train_start=index[0],
        train_end=index[4],
        val_start=index[4],
        val_end=index[6],
        test_start=index[6],
        test_end=index[-1] + pd.Timedelta(minutes=15),
    )


class AvailabilityContractTest(unittest.TestCase):
    def test_sequence_dataset_preserves_body_rows_and_uses_original_offsets(self) -> None:
        index = pd.date_range("2024-01-01", periods=8, freq="15min")
        features = np.arange(8 * 17, dtype=np.float32).reshape(8, 17)
        availability = pd.DataFrame(
            {
                "spot_bar_observed": True,
                "funding_rate_available": [True, True, False, True, True, True, True, True],
                "mark_close_available": True,
            },
            index=index,
        )
        dataset = SequenceDataset(
            features,
            seq_len=2,
            timestamps=index,
            availability=availability,
        )
        self.assertEqual(dataset.T, 8)
        self.assertEqual(dataset.features.shape[-1], 17)
        np.testing.assert_array_equal(dataset.row_eligible, [True, True, False, True, True, True, True, True])
        np.testing.assert_array_equal(dataset.valid_starts, [0, 3, 4, 5, 6])
        # Dataset index 1 maps to original body offset 3, proving that the
        # false row was not removed/compacted before window selection.
        np.testing.assert_array_equal(dataset[1]["obs"].numpy(), features[3:5])

    def test_wfo_propagates_sidecar_from_dataframe_attrs(self) -> None:
        index = pd.date_range("2024-01-01", periods=8, freq="15min")
        features = pd.DataFrame(
            np.arange(8 * 17, dtype=np.float32).reshape(8, 17),
            index=index,
            columns=[f"f{i}" for i in range(17)],
        )
        availability = pd.DataFrame(
            {
                "spot_bar_observed": True,
                "funding_rate_available": True,
                "mark_close_available": True,
            },
            index=index,
        )
        availability.loc[index[2], "mark_close_available"] = False
        features.attrs.update(
            {
                "availability": availability,
                "availability_interval": "15m",
                "availability_include_funding": True,
                "availability_include_mark": True,
            }
        )
        returns = pd.Series(np.zeros(8), index=index, name="returns")
        dataset = WFODataset(features, returns, _split(index), seq_len=2)
        self.assertEqual(dataset.obs_dim, 17)
        np.testing.assert_array_equal(dataset.train_dataset().valid_starts, [0])
        np.testing.assert_array_equal(dataset.val_dataset().valid_starts, [0])


if __name__ == "__main__":
    unittest.main()
