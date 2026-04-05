from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd
import yaml

from build_coinmetrics_source_cache import _apply_transform as apply_coinmetrics_transform
from check_config_source_requirements import collect_missing_requirements
from select_best_source_family import select_best_family
from source_rollout_plan import (
    build_rollout_snapshot,
    dedupe_missing_targets,
    fetch_command_hint,
    parse_cache_tag,
)
from validate_source_manifest import validate_manifest
from validate_source_rollout_suite import validate_suite


class SourceRolloutHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._test_root = Path("checkpoints") / "_test_source_rollout_helpers" / uuid.uuid4().hex
        self._test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self._test_root.exists():
            shutil.rmtree(self._test_root, ignore_errors=True)

    def test_dedupe_missing_targets(self) -> None:
        missing = [
            "a -> missing TAG_series_signed_order_flow.parquet",
            "b -> missing TAG_series_signed_order_flow.parquet",
            "c -> missing TAG_series_exchange_netflow.parquet",
        ]
        self.assertEqual(
            dedupe_missing_targets(missing),
            [
                "TAG_series_signed_order_flow.parquet",
                "TAG_series_exchange_netflow.parquet",
            ],
        )

    def test_parse_cache_tag(self) -> None:
        self.assertEqual(
            parse_cache_tag("BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2"),
            ("BTCUSDT", "15m", "2021-01-01", "2023-06-01"),
        )

    def test_coinmetrics_logdiff_transform(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"active_address_growth": [100.0, 110.0, 121.0, 133.1]}, index=idx)
        out = apply_coinmetrics_transform(df, "active_address_growth", "logdiff")
        values = out["active_address_growth"].round(6).tolist()
        self.assertTrue(pd.isna(values[0]))
        self.assertAlmostEqual(values[1], 0.09531, places=5)
        self.assertAlmostEqual(values[2], 0.09531, places=5)
        self.assertAlmostEqual(values[3], 0.09531, places=5)

    def test_collect_missing_requirements(self) -> None:
        cache_dir = self._test_root
        cache_tag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2"
        (cache_dir / f"{cache_tag}_mark.parquet").write_text("", encoding="utf-8")
        (cache_dir / f"{cache_tag}_funding.parquet").write_text("", encoding="utf-8")

        config_path = cache_dir / "cfg.yaml"
        cfg = {
            "risk_controller": {
                "feature_subset": [
                    "basis",
                    "funding_rate",
                    "signed_order_flow_z_96",
                ]
            }
        }
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        missing = collect_missing_requirements(str(cache_dir), cache_tag, str(config_path))
        self.assertEqual(
            missing,
            [f"signed_order_flow_z_96 -> missing {cache_tag}_series_signed_order_flow.parquet"],
        )

    def test_fetch_command_hint(self) -> None:
        cmd = fetch_command_hint("checkpoints\\x", "TAG", "TAG_series_exchange_netflow.parquet")
        self.assertIn("build_glassnode_source_cache.py", cmd)
        self.assertIn("--pit", cmd)

    def test_validate_suite_accepts_consistent_layout(self) -> None:
        cfg_dir = self._test_root / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for name in ("a.yaml", "b.yaml"):
            (cfg_dir / name).write_text("{}", encoding="utf-8")

        suite_path = cfg_dir / "suite.yaml"
        suite = {
            "ordered_configs": [
                "configs\\a.yaml",
                "configs\\b.yaml",
            ],
            "stages": [
                {"name": "stage1", "configs": ["configs\\a.yaml"]},
                {"name": "stage2", "configs": ["configs\\b.yaml"]},
            ],
        }
        suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
        self.assertEqual(validate_suite(str(suite_path)), [])

    def test_validate_suite_rejects_stage_mismatch(self) -> None:
        cfg_dir = self._test_root / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "a.yaml").write_text("{}", encoding="utf-8")

        suite_path = cfg_dir / "suite.yaml"
        suite = {
            "ordered_configs": ["configs\\a.yaml"],
            "stages": [
                {"name": "stage1", "configs": ["configs\\missing.yaml"]},
            ],
        }
        suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
        errors = validate_suite(str(suite_path))
        self.assertIn("ordered config missing from stages: configs\\a.yaml", errors)
        self.assertIn("stage config missing from ordered_configs: configs\\missing.yaml", errors)

    def test_validate_manifest_accepts_remote_example_shape(self) -> None:
        manifest_path = self._test_root / "manifest.yaml"
        manifest = {
            "cache_dir": "checkpoints/basis_source_cache",
            "cache_tag": "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
            "binance": {
                "symbol": "BTCUSDT",
                "interval": "15m",
                "start": "2021-01-01",
                "end": "2023-06-01",
                "taker_flow": True,
            },
            "coinmetrics": {
                "asset": "btc",
                "start": "2021-01-01",
                "end": "2023-06-01",
                "metrics": {
                    "active_address_growth": {
                        "metric": "AdrActCnt",
                        "transform": "logdiff",
                    }
                },
            },
            "glassnode": {
                "asset": "BTC",
                "api_key": "<glassnode_api_key>",
                "metrics": {
                    "exchange_netflow": "transactions/transfers_volume_exchanges_net",
                },
            },
        }
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        self.assertEqual(validate_manifest(str(manifest_path)), [])

    def test_validate_manifest_rejects_bad_transform(self) -> None:
        manifest_path = self._test_root / "bad_manifest.yaml"
        manifest = {
            "cache_dir": "checkpoints/basis_source_cache",
            "cache_tag": "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
            "coinmetrics": {
                "asset": "btc",
                "start": "2021-01-01",
                "end": "2023-06-01",
                "metrics": {
                    "active_address_growth": {
                        "metric": "AdrActCnt",
                        "transform": "bad_transform",
                    }
                },
            },
        }
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        errors = validate_manifest(str(manifest_path))
        self.assertIn(
            "coinmetrics.metrics.active_address_growth.transform is unsupported: bad_transform",
            errors,
        )

    def test_build_rollout_snapshot_reports_next_stage(self) -> None:
        cache_dir = self._test_root
        cache_tag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2"
        (cache_dir / f"{cache_tag}_mark.parquet").write_text("", encoding="utf-8")
        (cache_dir / f"{cache_tag}_funding.parquet").write_text("", encoding="utf-8")

        cfg_a = cache_dir / "a.yaml"
        cfg_a.write_text(
            yaml.safe_dump({"risk_controller": {"feature_subset": ["basis", "funding_rate"]}}),
            encoding="utf-8",
        )
        cfg_b = cache_dir / "b.yaml"
        cfg_b.write_text(
            yaml.safe_dump({"risk_controller": {"feature_subset": ["signed_order_flow_z_96"]}}),
            encoding="utf-8",
        )

        snapshot = build_rollout_snapshot(
            str(cache_dir),
            cache_tag,
            [str(cfg_a), str(cfg_b)],
        )
        self.assertEqual(snapshot["unlocked"], ["a.yaml"])
        self.assertEqual(snapshot["next_stage"]["config"], "b.yaml")
        self.assertEqual(
            snapshot["next_stage"]["missing_targets"],
            [f"{cache_tag}_series_signed_order_flow.parquet"],
        )

    def test_select_best_family_prefers_m2_progress(self) -> None:
        summary_path = self._test_root / "suite_summary.csv"
        pd.DataFrame(
            [
                {
                    "config": "basis",
                    "m2_pass_count": 0,
                    "test_alpha_pt_mean": 1.0,
                    "test_sharpe_delta_mean": 0.05,
                    "test_maxdd_delta_pt_mean": -1.0,
                    "test_win_rate_mean": 0.51,
                },
                {
                    "config": "orderflow",
                    "m2_pass_count": 1,
                    "test_alpha_pt_mean": 4.5,
                    "test_sharpe_delta_mean": 0.18,
                    "test_maxdd_delta_pt_mean": -8.0,
                    "test_win_rate_mean": 0.58,
                },
            ]
        ).to_csv(summary_path, index=False)

        ranked, best = select_best_family(str(summary_path))
        self.assertEqual(ranked.iloc[0]["config"], "orderflow")
        self.assertEqual(best["config"], "orderflow")


if __name__ == "__main__":
    unittest.main()
