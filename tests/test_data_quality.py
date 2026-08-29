"""Regression tests for the development-cache data quality gate."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unidream.eval.data_quality import (
    EXTERNAL_FEATURES,
    FULL17_FEATURES,
    OHLCV_FEATURES,
    DataQualityError,
    external_coverage,
    inspect_feature_contract,
    run_causality_probes,
    same_row_fairness,
    validate_feature_contract,
)
from unidream.cli.verify_data_quality import main as verify_data_quality


def _metadata(columns: list[str] | None = None) -> dict:
    columns = columns or list(FULL17_FEATURES)
    return {
        "schema_version": 1,
        "cache_tag": "synthetic-development-v3",
        "parameters": {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "start": "2018-01-01",
            "end": "2024-01-01",
            "zscore_window_days": 60,
            "extra_series_mode": "derived",
            "extra_series_include": [],
            "include_funding": True,
            "include_oi": False,
            "include_mark": True,
        },
        "feature_columns": columns,
        "rows": 8,
        "first_timestamp": "2018-01-01 00:00:00",
        "last_timestamp": "2018-01-01 01:45:00",
        "provenance": "synthetic",
    }


def _full17(rows: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2018-01-01", periods=rows, freq="15min")
    values = np.arange(rows * len(FULL17_FEATURES), dtype=np.float64).reshape(rows, -1) + 1.0
    features = pd.DataFrame(values, index=index, columns=FULL17_FEATURES)
    # Explicit zero values are valid observations but still fail the mask gate.
    features["funding_rate"] = np.linspace(0.0, 0.001, rows)
    features["basis"] = 0.0
    features["basis_mom"] = np.linspace(-0.1, 0.1, rows)
    features["basis_abs"] = np.abs(features["basis_mom"])
    returns = pd.Series(np.linspace(-0.01, 0.01, rows), index=index, name="returns")
    return features, returns


class DataQualityContractTest(unittest.TestCase):
    def test_normal_full17_contract_is_aligned_and_digestable(self) -> None:
        features, returns = _full17()
        result = validate_feature_contract(features, returns, _metadata())
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["alignment"]["same_index"])
        self.assertEqual(result["schema"]["actual_feature_columns"], list(FULL17_FEATURES))
        self.assertEqual(len(result["schema"]["schema_digest"]), 64)

    def test_missing_funding_and_basis_are_named_and_fail_closed(self) -> None:
        features, returns = _full17()
        features = features.drop(columns=["funding_rate", "basis"])
        result = inspect_feature_contract(features, returns, _metadata())
        self.assertEqual(result["status"], "fail")
        message = " | ".join(result["issues"])
        self.assertIn("funding_rate", message)
        self.assertIn("basis", message)
        with self.assertRaisesRegex(DataQualityError, "funding_rate"):
            validate_feature_contract(features, returns, _metadata())

    def test_nan_and_inf_are_reported_without_repair(self) -> None:
        features, returns = _full17()
        features.iloc[2, 0] = np.nan
        features.iloc[3, 1] = np.inf
        returns.iloc[4] = -np.inf
        result = inspect_feature_contract(features, returns, _metadata())
        self.assertEqual(result["features"]["numeric"]["nonfinite_count"], 2)
        self.assertEqual(result["returns"]["numeric"]["nonfinite_count"], 1)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("non-finite" in issue for issue in result["issues"]))

    def test_duplicate_and_non_15m_gap_are_detected(self) -> None:
        features, returns = _full17()
        duplicate_features = pd.concat([features.iloc[:3], features.iloc[2:]])
        duplicate_returns = pd.concat([returns.iloc[:3], returns.iloc[2:]])
        duplicate = inspect_feature_contract(duplicate_features, duplicate_returns, _metadata())
        self.assertGreater(duplicate["features"]["index"]["duplicate_count"], 0)
        self.assertIn("duplicate timestamps", " | ".join(duplicate["issues"]))

        gap_features = features.drop(index=features.index[3])
        gap_returns = returns.drop(index=returns.index[3])
        gap = inspect_feature_contract(gap_features, gap_returns, _metadata())
        self.assertGreater(gap["features"]["index"]["missing_bar_count"], 0)
        self.assertIn("non-15m gaps", " | ".join(gap["issues"]))

    def test_metadata_column_order_is_a_contract(self) -> None:
        features, returns = _full17()
        reordered = features[list(reversed(features.columns))]
        result = inspect_feature_contract(reordered, returns, _metadata())
        self.assertEqual(result["status"], "fail")
        self.assertIn("column order mismatch", " | ".join(result["issues"]))

    def test_external_zero_missing_ambiguity_is_a_failed_gate(self) -> None:
        features, _ = _full17()
        features.loc[features.index[1], "basis"] = np.nan
        result = external_coverage(features)
        self.assertEqual(result["external"]["basis"]["zero_count"], len(features) - 1)
        self.assertEqual(result["external"]["basis"]["missing_count"], 1)
        self.assertFalse(result["availability_gate"]["availability_mask_present"])
        self.assertEqual(result["availability_gate"]["status"], "fail")
        self.assertIn("indistinguishable", result["availability_gate"]["reason"])

    def test_same_row_fairness_uses_full17_intersection(self) -> None:
        features, _ = _full17()
        fair = same_row_fairness(features)
        self.assertTrue(fair["same_row_eligibility"])
        self.assertEqual(fair["ohlcv13_eligible_rows"], fair["full17_eligible_rows"])

        features.loc[features.index[3], "basis"] = np.nan
        unfair = same_row_fairness(features)
        self.assertFalse(unfair["same_row_eligibility"])
        self.assertEqual(unfair["full17_eligible_rows"], len(features) - 1)

    def test_causality_probes_cover_prefix_and_external_offsets(self) -> None:
        result = run_causality_probes()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            set(result["checks"]),
            {
                "future_perturbation_prefix",
                "prefix_invariance",
                "mark_offset_no_future_bfill",
                "funding_offset_asof",
            },
        )
        self.assertTrue(all(check["status"] == "pass" for check in result["checks"].values()))

    def test_cli_writes_ledger_and_report_while_preserving_failed_mask_gate(self) -> None:
        features, returns = _full17()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            tag = "synthetic-development-v3"
            features.to_parquet(cache_dir / f"{tag}_features.parquet")
            returns.to_frame().to_parquet(cache_dir / f"{tag}_returns.parquet")
            (cache_dir / f"{tag}_metadata.json").write_text(
                json.dumps(_metadata()), encoding="utf-8"
            )
            config = {
                "run": {
                    "start": "2018-01-01",
                    "end": "2024-01-01",
                    "folds": [0],
                },
                "data": {"symbol": "BTCUSDT", "interval": "15m"},
                "normalization": {"zscore_window_days": 60},
                "logging": {"cache_dir": str(cache_dir)},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            ledger = root / "quality.jsonl"
            report = root / "quality.md"
            exit_code = verify_data_quality(
                [
                    "--config", str(config_path),
                    "--features", str(cache_dir / f"{tag}_features.parquet"),
                    "--returns", str(cache_dir / f"{tag}_returns.parquet"),
                    "--metadata", str(cache_dir / f"{tag}_metadata.json"),
                    "--ledger", str(ledger),
                    "--report", str(report),
                    "--allow-quality-gate-fail",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(report.exists())
            self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 11)
            self.assertIn("external availability mask | fail", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
