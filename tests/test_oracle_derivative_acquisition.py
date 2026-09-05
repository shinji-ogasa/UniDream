import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unidream.experiments.oracle_derivative_acquisition import run


class OracleDerivativeAcquisitionTests(unittest.TestCase):
    def config(self, root):
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump({"source": "um_klines", "symbol": "BTCUSDT",
            "interval": "15m", "start_month": "2020-01", "end_month": "2020-02",
            "feature_decision_cutoff": "2020-01-01T00:45:00Z", "timeout_seconds": 1,
            "output_dir": str(root / "out")}))
        return path

    def downloader(self, calls, *, bad_checksum=False, bad_time=False):
        def fetch(source, *, month, raw_dir, **kwargs):
            calls.append(month)
            if month == "2020-02":
                return pd.DataFrame(), {"month": month, "source": source, "http_status": 404}
            raw = Path(raw_dir) / source / f"BTCUSDT-15m-{month}.zip"
            raw.parent.mkdir(parents=True)
            payload = b"fixture raw payload"
            raw.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            idx = pd.date_range(month + "-01", periods=4, freq="15min", tz="UTC", name="bar_open_ts")
            frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                "volume": 10., "quote_volume": 1000., "n_trades": 5,
                "taker_buy_base": 4., "taker_buy_quote": 400.,
                "bar_close_ts": idx + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)}, index=idx)
            if bad_time:
                frame.loc[idx[0], "bar_close_ts"] += pd.Timedelta(milliseconds=1)
            return frame, {"month": month, "source": source, "http_status": 200,
                "checksum_verified": not bad_checksum, "response_sha256": digest,
                "checksum_expected_sha256": digest, "parsed_rows": len(frame)}
        return fetch

    def test_fixed_grid_missing_month_cutoff_hashes_and_immutable_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.config(root); calls = []
            result = run(config, downloader=self.downloader(calls))
            self.assertEqual(calls, ["2020-01", "2020-02"])
            self.assertEqual(result["observed_rows"], 4)
            self.assertEqual(result["rows"], 60 * 96)
            self.assertEqual(result["unavailable_months"], ["2020-02"])
            self.assertEqual(result["feature_eligible_observed_rows"], 2)
            self.assertEqual(result["retained_post_cutoff_observed_rows"], 2)
            data = pd.read_parquet(result["data_path"])
            availability = pd.read_parquet(result["availability_path"])
            self.assertTrue(data.loc["2020-02", "open"].isna().all())
            self.assertFalse(availability.loc["2020-02", "um_bar_observed"].any())
            self.assertEqual(result["data_sha256"], hashlib.sha256(Path(result["data_path"]).read_bytes()).hexdigest())
            again = run(config, downloader=lambda *a, **k: self.fail("resume downloaded again"))
            self.assertEqual(result, again)
            self.assertEqual(len(Path(result["source_ledger_path"]).read_text().splitlines()), 2)
            raw = root / "out/raw/um_klines/BTCUSDT-15m-2020-01.zip"
            raw.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                run(config)

    def test_checksum_failure_is_recorded_and_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.config(root)
            with self.assertRaisesRegex(ValueError, "failed source"):
                run(config, downloader=self.downloader([], bad_checksum=True))
            record = json.loads((root / "out/monthly/2020-01.json").read_text())
            self.assertEqual(record["status"], "failed_source_contract")
            self.assertFalse((root / "out/um_15m.parquet").exists())

    def test_timestamp_and_registration_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.config(root)
            with self.assertRaisesRegex(ValueError, "inclusive close"):
                run(config, downloader=self.downloader([], bad_time=True))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.config(root)
            run(config, downloader=self.downloader([]))
            value = yaml.safe_load(config.read_text());value["timeout_seconds"] = 2
            config.write_text(yaml.safe_dump(value))
            with self.assertRaisesRegex(ValueError, "registration differs"):
                run(config)


if __name__ == "__main__":
    unittest.main()
