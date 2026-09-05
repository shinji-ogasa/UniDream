"""Tests for the audited, data-only Spot alpha/DD archive acquisition."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from unidream.experiments.alpha_dd_data import (
    AVAILABILITY_COLUMN,
    AcquisitionError,
    OUTPUT_COLUMNS,
    expected_bar_grid,
    run_acquisition,
)
from unidream.data.d1_signed_flow import _parse_kline_archive_bytes


def _frame(month: str, positions: list[int]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{month}-01", tz="UTC") + pd.Timedelta(minutes=15 * n) for n in positions],
        name="bar_open_ts",
    )
    rows = []
    for number in positions:
        value = float(number + 100)
        rows.append(
            {
                "open": value,
                "high": value + 1.0,
                "low": value - 1.0,
                "close": value + 0.5,
                "volume": 10.0,
                "quote_volume": 100.0,
                "n_trades": 7,
                "taker_buy_base": 6.0,
                "taker_buy_quote": 60.0,
            }
        )
    return pd.DataFrame(rows, index=index)


def _record(month: str, *, status: int = 200, checksum: bool = True, error: str | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": "d1_archive_download",
        "source": "spot_klines",
        "symbol": "BTCUSDT",
        "interval": "15m",
        "month": month,
        "archive_url": f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{month}.zip",
        "final_url": f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{month}.zip",
        "http_status": status,
        "response_sha256": (month.replace("-", "") * 32)[:64],
        "archive_revision_id": (month.replace("-", "") * 32)[:64],
        "checksum_verified": checksum,
        "checksum_expected_sha256": (month.replace("-", "") * 32)[:64],
        "archive_published_ts": None,
        "collector_observed_ts": None,
        "exchange_available_ts": None,
        "live_causal_eligible": False,
    }
    if error:
        record["error"] = error
    return record


def _zip_csv(name: str, rows: list[list[str]]) -> bytes:
    csv_buffer = io.StringIO()
    csv.writer(csv_buffer, lineterminator="\n").writerows(rows)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, csv_buffer.getvalue().encode("utf-8"))
    return output.getvalue()


def _kline_row(open_epoch: int, close_epoch: int) -> list[str]:
    return [
        str(open_epoch),
        "99",
        "101",
        "98",
        "100",
        "10",
        str(close_epoch),
        "1000",
        "7",
        "6",
        "600",
        "0",
    ]


class AlphaDDDataTest(unittest.TestCase):
    def test_timing_quarantine_does_not_hide_numeric_corruption(self) -> None:
        from unidream.data.official_v4_sources import OfficialSourceError
        start = 1_735_689_600_000_000
        valid = _kline_row(start, start + 900_000_000 - 1)
        for bad_value in ("not-a-number", "inf", "-1"):
            with self.subTest(value=bad_value):
                bad = _kline_row(start + 900_000_000, start + 900_000_000 + 1)
                bad[4] = bad_value
                payload = _zip_csv("BTCUSDT-15m-2025-01.csv", [valid, bad])
                with self.assertRaises(OfficialSourceError):
                    _parse_kline_archive_bytes(payload, source="spot_klines", symbol="BTCUSDT",
                        interval="15m", month="2025-01", quarantine_invalid_rows=True, timestamp_unit="auto")

    def test_opt_in_parser_detects_microseconds_and_quarantines_bad_timing_row(self) -> None:
        open_us = 1_735_689_600_000_000
        valid = _kline_row(open_us, open_us + 900_000_000 - 1)
        bad = _kline_row(open_us + 15 * 60 * 1_000_000, open_us + 15 * 60 * 1_000_000 + 899_999_000)
        payload = _zip_csv("BTCUSDT-15m-2025-01.csv", [valid, bad])
        frame, metadata = _parse_kline_archive_bytes(
            payload,
            source="spot_klines",
            symbol="BTCUSDT",
            interval="15m",
            month="2025-01",
            quarantine_invalid_rows=True,
            timestamp_unit="auto",
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(metadata["timestamp_unit"], "us")
        self.assertEqual(metadata["quarantined_rows"], 1)
        self.assertIn("bar_close_does_not_match_15m", metadata["quarantine_records"][0]["reasons"])
        self.assertEqual(len(metadata["quarantine_records"][0]["raw_row_sha256"]), 64)

    def test_expected_grid_is_utc_bar_open_and_inclusive_by_month(self) -> None:
        grid = expected_bar_grid("2024-01", "2024-02")
        self.assertEqual(grid[0], pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(grid[-1], pd.Timestamp("2024-02-29 23:45", tz="UTC"))
        self.assertEqual(grid.name, "bar_open_ts")
        self.assertEqual(str(grid.tz), "UTC")
        self.assertEqual(len(grid), 60 * 24 * 4)

    def test_missing_rows_are_nan_and_have_explicit_availability_mask(self) -> None:
        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            return _frame(month, [0, 2]), _record(month)

        with tempfile.TemporaryDirectory() as temporary:
            result = run_acquisition(
                Path(temporary) / "spot_15m.parquet",
                "2024-01",
                "2024-01",
                max_workers=1,
                downloader=downloader,
            )
            output = pd.read_parquet(result["output_path"])
            availability = pd.read_parquet(result["availability_path"])
            self.assertEqual(tuple(output.columns), OUTPUT_COLUMNS)
            self.assertTrue(output.index.tz is not None)
            self.assertTrue(pd.isna(output.iloc[1]["close"]))
            self.assertFalse(bool(availability.iloc[1][AVAILABILITY_COLUMN]))
            self.assertTrue(bool(availability.iloc[2][AVAILABILITY_COLUMN]))
            self.assertEqual(result["available_rows"], 2)

    def test_checksum_is_mandatory_and_error_is_appended_before_failure(self) -> None:
        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            return _frame(month, [0]), _record(month, checksum=False)

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "spot_15m.parquet"
            with self.assertRaises(AcquisitionError) as raised:
                run_acquisition(
                    output_path,
                    "2024-01",
                    "2024-01",
                    max_workers=1,
                    downloader=downloader,
                )
            payload = raised.exception.result
            self.assertTrue(Path(payload["output_path"]).exists())
            records = [
                json.loads(line)
                for line in Path(payload["ledger_path"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(record.get("checksum_failure") for record in records))
            self.assertEqual(records[-1]["status"], "failed")
            availability = pd.read_parquet(payload["availability_path"])
            self.assertFalse(bool(availability[AVAILABILITY_COLUMN].any()))

    def test_numeric_parser_failure_is_fail_closed_and_appended(self) -> None:
        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            return pd.DataFrame(), _record(
                month,
                error="OfficialSourceError: archive contains non-positive OHLC prices",
            )

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "spot_15m.parquet"
            with self.assertRaises(AcquisitionError) as raised:
                run_acquisition(
                    output_path,
                    "2024-01",
                    "2024-01",
                    max_workers=1,
                    downloader=downloader,
                )
            payload = raised.exception.result
            records = [
                json.loads(line)
                for line in Path(payload["ledger_path"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[-1]["status"], "failed")
            self.assertEqual(records[-1]["fatal_months"], ["2024-01"])
            self.assertEqual(records[-2]["parser_error_kind"], "integrity_or_structure")

    def test_verified_months_resume_without_downloader_calls(self) -> None:
        calls: list[str] = []

        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            calls.append(month)
            return _frame(month, [0]), _record(month)

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "spot_15m.parquet"
            first = run_acquisition(output_path, "2024-01", "2024-02", max_workers=1, downloader=downloader)
            self.assertEqual(calls, ["2024-01", "2024-02"])
            self.assertTrue(Path(first["monthly_dir"], "2024-01.parquet").exists())
            self.assertTrue(Path(first["monthly_dir"], "2024-01.json").exists())

            def should_not_download(*args: object, **kwargs: object):
                raise AssertionError("verified monthly checkpoint was downloaded again")

            resumed = run_acquisition(
                output_path,
                "2024-01",
                "2024-02",
                max_workers=1,
                downloader=should_not_download,
            )
            self.assertEqual(resumed["available_rows"], 2)
            sidecar = json.loads(Path(resumed["sha_sidecar"]).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["artifact_sha256"], resumed["output_sha256"])
            self.assertEqual(sidecar["source_ledger_sha256"], resumed["ledger_sha256"])

    def test_corrupt_checkpoint_is_not_trusted_and_is_recorded(self) -> None:
        calls: list[str] = []

        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            calls.append(month)
            return _frame(month, [0]), _record(month)

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "spot_15m.parquet"
            first = run_acquisition(output_path, "2024-01", "2024-01", max_workers=1, downloader=downloader)
            monthly_path = Path(first["monthly_dir"], "2024-01.parquet")
            monthly_path.write_bytes(monthly_path.read_bytes() + b"tampered")
            run_acquisition(output_path, "2024-01", "2024-01", max_workers=1, downloader=downloader)
            self.assertEqual(calls, ["2024-01", "2024-01"])
            records = [
                json.loads(line)
                for line in Path(first["ledger_path"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(record.get("record_type") == "alpha_dd_cache_invalid" for record in records))

    def test_unavailable_suffix_and_historical_hole_are_distinct(self) -> None:
        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            if month in {"2024-01", "2024-03"}:
                return pd.DataFrame(), _record(month, status=404, checksum=False, error="official source returned HTTP 404")
            return _frame(month, [0]), _record(month)

        with tempfile.TemporaryDirectory() as temporary:
            result = run_acquisition(
                Path(temporary) / "spot_15m.parquet",
                "2024-01",
                "2024-03",
                max_workers=1,
                now="2024-04-01",
                downloader=downloader,
            )
            self.assertEqual(result["historical_gap_months"], ["2024-01"])
            self.assertEqual(result["unavailable_tail_months"], ["2024-03"])
            self.assertEqual(result["status"], "complete_with_gaps")

    def test_future_unavailable_tail_stops_later_requests(self) -> None:
        calls: list[str] = []

        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            calls.append(month)
            if month >= "2024-03":
                return pd.DataFrame(), _record(month, status=404, checksum=False, error="official source returned HTTP 404")
            return _frame(month, [0]), _record(month)

        with tempfile.TemporaryDirectory() as temporary:
            result = run_acquisition(
                Path(temporary) / "spot_15m.parquet",
                "2024-01",
                "2024-05",
                max_workers=1,
                now="2024-03-01",
                downloader=downloader,
            )
            self.assertEqual(calls, ["2024-01", "2024-02", "2024-03"])
            self.assertEqual(result["unavailable_tail_months"], ["2024-03", "2024-04", "2024-05"])

    def test_concurrency_is_hard_capped_at_six(self) -> None:
        lock = threading.Lock()
        active = 0
        maximum = 0

        def downloader(source: str, *, symbol: str, interval: str, month: str, raw_dir: object, timeout: float):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return _frame(month, [0]), _record(month)

        with tempfile.TemporaryDirectory() as temporary:
            run_acquisition(
                Path(temporary) / "spot_15m.parquet",
                "2024-01",
                "2024-08",
                max_workers=MAX_WORKERS_FOR_TEST,
                downloader=downloader,
            )
        self.assertLessEqual(maximum, 6)


MAX_WORKERS_FOR_TEST = 6


if __name__ == "__main__":
    unittest.main()
