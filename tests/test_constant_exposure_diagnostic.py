import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.data.dataset import WFOSplit
from unidream.eval.constant_exposure_diagnostic import (
    ConstantExposureData,
    WAVE_FOLDS,
    _net_paths,
    run_constant_exposure_diagnostic,
    validate_wave3d_folds,
)


def _cfg(delay=1):
    return {
        "data": {"interval": "15m"},
        "eval": {"forecast_execution_delay_bars": delay},
        "costs": {"spread_bps": 0.0, "fee_rate": 0.0, "slippage_bps": 0.0},
    }


def _synthetic_data() -> ConstantExposureData:
    timestamps = []
    values = []
    splits = []
    cursor = pd.Timestamp("2018-01-01")
    for fold in WAVE_FOLDS:
        train_start = cursor
        train_end = train_start + pd.Timedelta(minutes=15)
        val_start = train_end
        val_end = val_start + pd.Timedelta(minutes=30)
        test_start = val_end
        test_end = test_start + pd.Timedelta(minutes=60)
        splits.append(
            WFOSplit(
                fold_idx=fold,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        timestamps.extend(
            [
                val_start,
                val_start + pd.Timedelta(minutes=15),
                test_start,
                test_start + pd.Timedelta(minutes=15),
                test_start + pd.Timedelta(minutes=30),
                test_start + pd.Timedelta(minutes=45),
            ]
        )
        values.extend([0.001, -0.0002, 0.0005, -0.0001, 0.0004, -0.0001])
        cursor = test_end + pd.Timedelta(minutes=15)
    series = pd.Series(values, index=pd.DatetimeIndex(timestamps), name="returns")
    return ConstantExposureData(
        returns=series,
        splits=tuple(splits),
        returns_path=Path("synthetic_returns.parquet"),
        train_years=2,
        val_months=3,
        test_months=3,
        evaluation_cutoff_exclusive=splits[-1].test_end,
        source_rows=len(series),
        excluded_future_rows=0,
    )


class ConstantExposureDiagnosticContractTest(unittest.TestCase):
    def test_exact_even_wave_folds_reject_missing_duplicates_and_holdout(self):
        self.assertEqual(validate_wave3d_folds(), WAVE_FOLDS)
        with self.assertRaises(ValueError):
            validate_wave3d_folds(tuple(range(11)))
        with self.assertRaises(ValueError):
            validate_wave3d_folds((*range(11), 11, 11))
        with self.assertRaises(ValueError):
            validate_wave3d_folds((*range(11), 12))
        with self.assertRaises(ValueError):
            validate_wave3d_folds([0.0, *range(1, 12)])
        with self.assertRaises(ValueError):
            validate_wave3d_folds([True, *range(1, 12)])

    def test_fixed_delay_path_is_right_aligned_without_padding(self):
        returns = np.asarray([0.0, 1.0e-3, 0.0])
        positions = np.asarray([0.5, 1.12, 0.5])
        path = _net_paths(returns, positions, _cfg(), execution_delay=1)
        np.testing.assert_array_equal(path["returns"], [1.0e-3, 0.0])
        np.testing.assert_array_equal(path["effective_positions"], [0.5, 1.12])
        self.assertEqual(len(path["alpha_excess"]), 2)

    def test_delay_is_fixed_and_public_float_is_rejected(self):
        data = _synthetic_data()
        with self.assertRaisesRegex(ValueError, "fixes.*1"):
            run_constant_exposure_diagnostic(
                data=data,
                cfg=_cfg(delay=0),
                config_path="synthetic.yaml",
                output_dir=tempfile.mkdtemp(),
            )
        with self.assertRaisesRegex(ValueError, "execution_delay_bars"):
            run_constant_exposure_diagnostic(
                data=data,
                cfg=_cfg(delay=1.5),
                config_path="synthetic.yaml",
                output_dir=tempfile.mkdtemp(),
            )

    def test_run_writes_paths_ledger_and_never_reports_fold12(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "returns.parquet"
            data = replace(_synthetic_data(), returns_path=cache_path)
            data.returns.to_frame().to_parquet(cache_path)
            result = run_constant_exposure_diagnostic(
                data=data,
                cfg=_cfg(),
                config_path="synthetic.yaml",
                seed=7,
                output_dir=directory,
            )
            self.assertEqual(result["folds"], list(range(12)))
            self.assertFalse(result["fold12_or_later_evaluated"])
            self.assertEqual(result["gate"]["promotion_eligible"], False)
            diagnostics = result["statistical_diagnostics"]
            self.assertEqual(diagnostics["deflated_sharpe"]["selected_candidate"], "selected_constant")
            self.assertEqual(diagnostics["deflated_sharpe"]["n_trials"], 7)
            self.assertEqual(diagnostics["cscv_pbo"]["n_candidates"], 6)
            self.assertEqual(diagnostics["cscv_pbo"]["n_subperiods"], 12)
            self.assertEqual(
                {item["name"] for item in diagnostics["stress"]["records"]["cost"]},
                {"cost_1x", "cost_1.5x", "cost_2x"},
            )
            self.assertEqual(result["selected_vs_previous"]["folds"], list(range(1, 12)))
            self.assertEqual(result["next_wave_candidates"], [])
            self.assertEqual(result["path_artifacts"]["entries"], 107)
            self.assertTrue(Path(result["path_artifacts"]["npz_path"]).exists())
            self.assertTrue(Path(result["ledger_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())
            rows = [json.loads(line) for line in Path(result["ledger_path"]).read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row["folds"] == list(range(12)) for row in rows))
            self.assertNotIn(12, [row.get("fold") for row in rows if row.get("fold") is not None])
            def assert_finite(value):
                if isinstance(value, dict):
                    for item in value.values():
                        assert_finite(item)
                elif isinstance(value, list):
                    for item in value:
                        assert_finite(item)
                elif isinstance(value, float):
                    self.assertTrue(np.isfinite(value))
            assert_finite(result)
            for row in rows:
                assert_finite(row)
            report = Path(result["report_path"]).read_text()
            self.assertIn("low-frequency constant-exposure baseline", report)
            self.assertIn("not a forecast-accuracy result", report)


if __name__ == "__main__":
    unittest.main()
