from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from unidream.data.dataset import WFOSplit
from unidream.experiments import train_pipeline


def _split() -> WFOSplit:
    ts = pd.Timestamp
    return WFOSplit(
        fold_idx=3,
        train_start=ts("2025-01-01", tz="UTC"),
        train_end=ts("2025-01-01 01:00", tz="UTC"),
        val_start=ts("2025-01-01 01:00", tz="UTC"),
        val_end=ts("2025-01-01 02:00", tz="UTC"),
        test_start=ts("2025-01-01 02:00", tz="UTC"),
        test_end=ts("2025-01-01 03:00", tz="UTC"),
    )


class RunWfoFoldsTestPathTest(unittest.TestCase):
    def _run(self, logging_cfg: dict, checkpoint_dir: Path):
        index = pd.date_range("2025-01-01", periods=16, freq="15min", tz="UTC")
        features = pd.DataFrame({"x": np.arange(16.0)}, index=index)
        positions = np.array([1.0, 0.9, 1.1, 1.0])
        returns = np.array([0.01, -0.02, 0.005, 0.0])
        pnl = positions * returns

        def run_fold_fn(*, fold_idx, **_kwargs):
            (checkpoint_dir / f"fold_{fold_idx}").mkdir(parents=True, exist_ok=True)
            return {
                "positions": positions,
                "test_returns": returns,
                "metrics": SimpleNamespace(pnl_series=pnl),
            }

        with mock.patch.object(train_pipeline, "WFODataset"):
            train_pipeline.run_wfo_folds(
                features_df=features,
                raw_returns=np.zeros(16),
                splits=[_split()],
                data_cfg={},
                cfg={"logging": logging_cfg},
                device="cpu",
                checkpoint_dir=str(checkpoint_dir),
                run_fold_fn=run_fold_fn,
            )
        return index[8:12], positions, returns, pnl

    def test_saves_test_path_aligned_to_test_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, positions, returns, pnl = self._run({"save_test_path": True}, root)
            dest = root / "fold_3" / "test_path.parquet"
            frame = pd.read_parquet(dest)
            self.assertEqual(list(frame.index), list(index))
            np.testing.assert_allclose(frame["position"], positions)
            np.testing.assert_allclose(frame["market_log_return"], returns)
            np.testing.assert_allclose(frame["net_log_return"], pnl)
            self.assertFalse(dest.with_suffix(".tmp").exists())

    def test_does_not_save_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run({}, root)
            self.assertFalse((root / "fold_3" / "test_path.parquet").exists())


if __name__ == "__main__":
    unittest.main()
