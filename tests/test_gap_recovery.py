"""Tests for official-only development-cache gap probes."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from unidream.eval.gap_recovery import (
    detect_gaps,
    probe_official_gap_recovery,
)


class _Response:
    def __init__(self, payload: list, url: str = "https://data-api.binance.vision/api/v3/klines"):
        self.content = str(payload).encode("utf-8")
        self.status_code = 200
        self.url = url

    def json(self) -> list:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payload: list):
        self.payload = payload
        self.calls: list[tuple[str, dict, float]] = []

    def get(self, url: str, params: dict | None = None, timeout: float = 0.0) -> _Response:
        self.calls.append((url, dict(params or {}), timeout))
        response = _Response(self.payload, url=url)
        response._payload = self.payload
        return response


class GapRecoveryTest(unittest.TestCase):
    def test_detect_gaps_reports_expected_timestamps_without_repair(self) -> None:
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2018-01-01 00:00"),
                pd.Timestamp("2018-01-01 00:15"),
                pd.Timestamp("2018-01-01 01:00"),
            ]
        )
        gaps = detect_gaps(index)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["expected_missing_count"], 2)
        self.assertEqual(
            gaps[0]["expected_missing_timestamps"],
            ["2018-01-01 00:30:00", "2018-01-01 00:45:00"],
        )
        self.assertEqual(len(index), 3)

    def test_probe_uses_official_rest_and_recovers_gap_without_writing_cache(self) -> None:
        index = pd.date_range("2018-01-01", periods=5, freq="15min").delete(3)
        timestamps = [
            int(value.timestamp() * 1000)
            for value in pd.date_range("2018-01-01", periods=5, freq="15min")
        ]
        payload = [[timestamp, "1", "1", "1", "1", "1", timestamp + 899999, "1", 1, "1", "1", "0"] for timestamp in timestamps]
        session = _Session(payload)
        features = pd.DataFrame({"close_ret": range(len(index))}, index=index)
        result = probe_official_gap_recovery(
            features,
            symbol="BTCUSDT",
            interval="15m",
            session=session,
            use_archive_fallback=False,
        )
        self.assertEqual(result["summary"]["status"], "pass")
        self.assertEqual(result["summary"]["expected_missing_bars"], 1)
        self.assertEqual(result["summary"]["official_covered_bars"], 1)
        self.assertEqual(result["summary"]["official_missing_after_probe"], 0)
        self.assertFalse(result["source_policy"]["non_official_provider_used"])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0][0], "https://data-api.binance.vision/api/v3/klines")

    def test_non_official_rest_host_is_rejected(self) -> None:
        index = pd.date_range("2018-01-01", periods=2, freq="15min")
        features = pd.DataFrame({"close_ret": [0.0, 0.1]}, index=index)
        with self.assertRaisesRegex(ValueError, "non-official"):
            probe_official_gap_recovery(
                features,
                rest_base_url="https://example.invalid",
                use_archive_fallback=False,
            )

    def test_archive_source_record_is_unique_per_month(self) -> None:
        index = pd.date_range("2018-01-01", periods=5, freq="15min").delete([2, 3])
        features = pd.DataFrame({"close_ret": range(len(index))}, index=index)
        archive_record = {
            "source": "official_spot_monthly_archive",
            "month": "2018-01",
            "http_status": 200,
        }
        with patch(
            "unidream.eval.gap_recovery._request_archive_month",
            return_value=(archive_record, []),
        ) as archive_probe:
            result = probe_official_gap_recovery(
                features,
                session=_Session([]),
                use_archive_fallback=True,
            )
        self.assertEqual(archive_probe.call_count, 1)
        self.assertEqual(len(result["gaps"][0]["source_records"]), 2)
        self.assertEqual(
            [record["source"] for record in result["gaps"][0]["source_records"]],
            ["official_spot_rest", "official_spot_monthly_archive"],
        )


if __name__ == "__main__":
    unittest.main()
