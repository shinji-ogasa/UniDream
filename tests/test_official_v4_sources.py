"""Tests for official-only v4 source URLs, schemas, and response evidence."""
from __future__ import annotations

import csv
import io
import unittest
import zipfile

import pandas as pd

from unidream.data.official_v4_sources import (
    FUNDING_COLUMNS,
    OfficialSourceError,
    fetch_archive_month,
    fetch_spot_rest_window,
    official_archive_url,
    probe_official_sources,
)


def _zip_csv(name: str, rows: list[list[str]]) -> bytes:
    raw = io.StringIO()
    writer = csv.writer(raw, lineterminator="\n")
    writer.writerows(rows)
    payload = raw.getvalue().encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return output.getvalue()


def _kline_row(timestamp: int) -> list[str]:
    return [
        str(timestamp), "1", "2", "0.5", "1.5", "10", str(timestamp + 899999),
        "15", "3", "4", "6", "0",
    ]


MARK_HEADER = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class _Response:
    def __init__(self, content: bytes, *, url: str, status_code: int = 200, payload=None):
        self.content = content
        self.url = url
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload


class _Session:
    def __init__(self, responses: dict[str, _Response]):
        self.responses = responses
        self.calls: list[tuple[str, dict, float]] = []

    def get(self, url: str, params=None, timeout: float = 0.0) -> _Response:
        self.calls.append((url, dict(params or {}), timeout))
        return self.responses[url]


class OfficialV4SourcesTest(unittest.TestCase):
    def test_archive_urls_and_csv_schemas_are_distinct_and_official(self) -> None:
        spot_url = official_archive_url("spot_klines", "BTCUSDT", "15m", "2018-01")
        mark_url = official_archive_url("um_mark_price_klines", "BTCUSDT", "15m", "2020-01")
        funding_url = official_archive_url("um_funding_rate", "BTCUSDT", "15m", "2020-01")
        self.assertIn("/spot/monthly/klines/", spot_url)
        self.assertIn("/futures/um/monthly/markPriceKlines/", mark_url)
        self.assertIn("/futures/um/monthly/fundingRate/", funding_url)

        spot_frame, spot_record = fetch_archive_month(
            "spot_klines",
            symbol="BTCUSDT",
            interval="15m",
            month="2018-01",
            session=_Session({spot_url: _Response(_zip_csv("spot.csv", [_kline_row(0)]), url=spot_url)}),
        )
        self.assertEqual(spot_frame.columns.tolist(), ["open", "high", "low", "close", "volume"])
        self.assertFalse(spot_record["schema"]["header_present"])
        self.assertEqual(spot_record["schema"]["column_count"], 12)

        mark_header = [MARK_HEADER]
        mark_frame, mark_record = fetch_archive_month(
            "um_mark_price_klines",
            symbol="BTCUSDT",
            interval="15m",
            month="2020-01",
            session=_Session(
                {mark_url: _Response(_zip_csv("mark.csv", mark_header + [_kline_row(0)]), url=mark_url)}
            ),
        )
        self.assertEqual(len(mark_frame), 1)
        self.assertTrue(mark_record["schema"]["header_present"])

        funding_rows = [list(FUNDING_COLUMNS), ["0", "8", "0.001"]]
        funding_frame, funding_record = fetch_archive_month(
            "um_funding_rate",
            symbol="BTCUSDT",
            interval="15m",
            month="2020-01",
            session=_Session(
                {funding_url: _Response(_zip_csv("funding.csv", funding_rows), url=funding_url)}
            ),
        )
        self.assertEqual(funding_frame.columns.tolist(), ["funding_rate"])
        self.assertEqual(funding_record["schema"]["columns"], list(FUNDING_COLUMNS))

    def test_official_404_is_recorded_and_non_official_redirect_is_rejected(self) -> None:
        url = official_archive_url("um_mark_price_klines", "BTCUSDT", "15m", "2019-12")
        frame, record = fetch_archive_month(
            "um_mark_price_klines",
            symbol="BTCUSDT",
            interval="15m",
            month="2019-12",
            session=_Session({url: _Response(b"", url=url, status_code=404)}),
        )
        self.assertTrue(frame.empty)
        self.assertEqual(record["http_status"], 404)
        self.assertIn("HTTP 404", record["error"])

        redirected = _Session(
            {url: _Response(b"", url="https://example.invalid/redirect", status_code=200)}
        )
        with self.assertRaisesRegex(OfficialSourceError, "non-official"):
            fetch_archive_month(
                "um_mark_price_klines",
                symbol="BTCUSDT",
                interval="15m",
                month="2019-12",
                session=redirected,
            )

    def test_spot_rest_uses_utc_window_and_records_hash(self) -> None:
        start = pd.Timestamp("2018-01-01 00:00:00")
        end = pd.Timestamp("2018-01-01 00:15:00")
        url = "https://data-api.binance.vision/api/v3/klines"
        payload = [_kline_row(0)]
        session = _Session({url: _Response(b"payload", url=url, payload=payload)})
        frame, record = fetch_spot_rest_window(
            symbol="BTCUSDT",
            interval="15m",
            start=start,
            end=end,
            session=session,
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(session.calls[0][1]["startTime"], 1514764800000)
        self.assertEqual(record["source"], "spot_klines_rest")
        self.assertEqual(len(record["response_sha256"]), 64)

    def test_probe_records_all_three_sources(self) -> None:
        # A real-source probe is covered by the reproducibility CLI; this test
        # checks the report shape using a tiny fake official response map.
        month = "2020-01"
        responses: dict[str, _Response] = {}
        for source in ("spot_klines", "um_mark_price_klines"):
            url = official_archive_url(source, "BTCUSDT", "15m", month)
            header = [] if source == "spot_klines" else [MARK_HEADER]
            responses[url] = _Response(_zip_csv(f"{source}.csv", header + [_kline_row(0)]), url=url)
        url = official_archive_url("um_funding_rate", "BTCUSDT", "15m", month)
        responses[url] = _Response(_zip_csv("funding.csv", [list(FUNDING_COLUMNS), ["0", "8", "0.001"]]), url=url)
        report = probe_official_sources(months=[month], session=_Session(responses))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(set(report["sources"]), {"spot_klines", "um_mark_price_klines", "um_funding_rate"})


if __name__ == "__main__":
    unittest.main()
