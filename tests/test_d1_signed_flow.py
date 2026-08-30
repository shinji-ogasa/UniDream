"""Tests for the data-only D1 signed-flow pilot."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.data.d1_signed_flow import (
    D1_AVAILABILITY_COLUMNS,
    D1_FEATURE_COLUMNS,
    _parse_kline_archive_bytes,
    aggtrade_archive_url,
    build_d1_features,
    classify_archive_revisions,
    download_d1_kline_month,
    estimate_aggtrade_archive_storage,
)
from unidream.data.official_v4_sources import official_archive_url
from unidream.cli.acquire_d1_signed_flow import run_pilot


def _pilot_args(tmp: str | Path) -> Namespace:
    root = Path(tmp)
    return Namespace(
        months=["2024-01"],
        symbol="BTCUSDT",
        interval="15m",
        raw_dir=None,
        timeout=1.0,
        ledger=root / "ledger.jsonl",
        capacity_start="2024-01",
        capacity_end="2024-02",
        features=root / "features.csv",
        availability=root / "availability.csv",
        capacity_json=root / "capacity.json",
        report=root / "report.md",
    )


def _source_record(source: str, *, error: str | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": "d1_archive_download",
        "source": source,
        "symbol": "BTCUSDT",
        "interval": "15m",
        "month": "2024-01",
        "archive_revision_id": "a" * 64,
        "checksum_verified": True,
        "archive_published_ts": None,
        "collector_observed_ts": None,
        "exchange_available_ts": None,
        "live_causal_eligible": False,
    }
    if error:
        record["error"] = error
    return record


def _zip_csv(name: str, rows: list[list[str]]) -> bytes:
    raw = io.StringIO()
    writer = csv.writer(raw, lineterminator="\n")
    writer.writerows(rows)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, raw.getvalue().encode("utf-8"))
    return output.getvalue()


def _kline_row(open_ms: int, close: float, quote: float = 100.0, buy_quote: float = 60.0) -> list[str]:
    return [
        str(open_ms),
        str(close - 1.0),
        str(close + 1.0),
        str(close - 2.0),
        str(close),
        "10",
        str(open_ms + 899_999),
        str(quote),
        "7",
        "6",
        str(buy_quote),
        "0",
    ]


class _Response:
    def __init__(
        self,
        content: bytes,
        *,
        url: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}


class _Session:
    def __init__(self, get_responses: dict[str, _Response], head_responses: dict[str, _Response] | None = None):
        self.get_responses = get_responses
        self.head_responses = head_responses or {}
        self.get_calls: list[tuple[str, float]] = []
        self.head_calls: list[tuple[str, float]] = []

    def get(self, url: str, timeout: float = 0.0) -> _Response:
        self.get_calls.append((url, timeout))
        return self.get_responses[url]

    def head(self, url: str, allow_redirects: bool = False, timeout: float = 0.0) -> _Response:
        self.head_calls.append((url, timeout))
        return self.head_responses[url]


def _frame(closes: list[float], *, missing: set[int] | None = None) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="15min")
    rows = []
    missing = missing or set()
    for idx, close in enumerate(closes):
        if idx in missing:
            continue
        rows.append(
            {
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 10.0,
                "quote_volume": 100.0,
                "n_trades": 7,
                "taker_buy_base": 6.0,
                "taker_buy_quote": 60.0,
            }
        )
    return pd.DataFrame(rows, index=index.delete(list(missing)))


class D1SignedFlowTest(unittest.TestCase):
    def test_um_kline_archive_url_is_distinct_and_official(self) -> None:
        url = official_archive_url("um_klines", "BTCUSDT", "15m", "2024-01")
        self.assertIn("/futures/um/monthly/klines/BTCUSDT/15m/", url)
        self.assertTrue(url.endswith("BTCUSDT-15m-2024-01.zip"))
        self.assertIn("/spot/monthly/aggTrades/", aggtrade_archive_url("spot_aggTrades", "BTCUSDT", "2024-01"))

    def test_download_verifies_checksum_and_keeps_archive_live_times_separate(self) -> None:
        archive_url = official_archive_url("spot_klines", "BTCUSDT", "15m", "2024-01")
        open_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        payload = _zip_csv("BTCUSDT-15m-2024-01.csv", [_kline_row(open_ms, 100.0)])
        digest = hashlib.sha256(payload).hexdigest()
        checksum = f"{digest}  {Path(archive_url).name}\n".encode("utf-8")
        session = _Session(
            {
                archive_url: _Response(payload, url=archive_url),
                archive_url + ".CHECKSUM": _Response(checksum, url=archive_url + ".CHECKSUM"),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            frame, record = download_d1_kline_month(
                "spot_klines",
                symbol="BTCUSDT",
                interval="15m",
                month="2024-01",
                raw_dir=tmp,
                session=session,
            )
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.index[0], pd.Timestamp("2024-01-01 00:00:00", tz="UTC"))
            self.assertEqual(
                frame.iloc[0]["bar_close_ts"],
                pd.Timestamp("2024-01-01 00:14:59.999000", tz="UTC"),
            )
            self.assertTrue(record["checksum_verified"])
            self.assertEqual(record["archive_revision_id"], digest)
            self.assertIsNone(record["archive_published_ts"])
            self.assertIsNone(record["collector_observed_ts"])
            self.assertFalse(record["live_causal_eligible"])
            self.assertTrue(Path(record["raw_path"]).exists())

    def test_build_uses_completed_bars_and_future_mutation_does_not_change_prefix(self) -> None:
        spot = _frame([100.0, 101.0, 102.0])
        perp = _frame([100.5, 101.5, 103.0])
        features, availability = build_d1_features(
            spot,
            perp,
            bar_open_start="2024-01-01 00:00:00",
            bar_open_end="2024-01-01 00:45:00",
        )
        self.assertEqual(features.index[0], pd.Timestamp("2024-01-01 00:15:00", tz="UTC"))
        self.assertEqual(features.index[-1], pd.Timestamp("2024-01-01 00:45:00", tz="UTC"))
        self.assertEqual(features.index.name, "decision_ts")
        self.assertEqual(set(features.columns), set(D1_FEATURE_COLUMNS))
        self.assertEqual(set(availability.columns), set(D1_AVAILABILITY_COLUMNS))
        self.assertAlmostEqual(features.iloc[0]["spot_taker_imbalance"], 0.2)
        self.assertAlmostEqual(features.iloc[0]["perp_taker_imbalance"], 0.2)
        self.assertAlmostEqual(features.iloc[0]["spot_perp_basis"], np.log(100.5 / 100.0))
        self.assertTrue(pd.isna(features.iloc[0]["spot_perp_return_divergence"]))
        self.assertFalse(availability.iloc[0]["spot_perp_return_divergence_available"])

        changed = perp.copy()
        changed.iloc[-1, changed.columns.get_loc("close")] = 10_000.0
        changed_features, _ = build_d1_features(
            spot,
            changed,
            bar_open_start="2024-01-01 00:00:00",
            bar_open_end="2024-01-01 00:45:00",
        )
        pd.testing.assert_frame_equal(features.iloc[:2], changed_features.iloc[:2])

    def test_missing_and_zero_are_distinct_masks(self) -> None:
        spot = _frame([100.0, 101.0, 102.0])
        spot.iloc[1, spot.columns.get_loc("quote_volume")] = 0.0
        spot.iloc[1, spot.columns.get_loc("taker_buy_quote")] = 0.0
        perp = _frame([100.5, 101.5, 103.0], missing={1})
        features, availability = build_d1_features(
            spot,
            perp,
            bar_open_start="2024-01-01 00:00:00",
            bar_open_end="2024-01-01 00:45:00",
        )
        # The real zero quote volume is retained, but imbalance is unavailable.
        self.assertEqual(features.iloc[1]["spot_quote_volume"], 0.0)
        self.assertTrue(pd.isna(features.iloc[1]["spot_taker_imbalance"]))
        self.assertFalse(availability.iloc[1]["spot_taker_imbalance_available"])
        # The missing perp row remains NaN and is not silently converted to zero.
        self.assertTrue(pd.isna(features.iloc[1]["perp_quote_volume"]))
        self.assertFalse(availability.iloc[1]["perp_bar_observed"])
        self.assertFalse(availability.iloc[1]["d1_features_available"])

    def test_parser_rejects_invalid_member_range_grid_and_numeric_integrity(self) -> None:
        jan_open_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        valid = _kline_row(jan_open_ms, 100.0)
        cases = {
            "member_name": ("unexpected.csv", [valid]),
            "month_range": (
                "BTCUSDT-15m-2024-01.csv",
                [_kline_row(int(pd.Timestamp("2023-12-31 23:45", tz="UTC").timestamp() * 1000), 100.0)],
            ),
            "grid_alignment": (
                "BTCUSDT-15m-2024-01.csv",
                [_kline_row(int(pd.Timestamp("2024-01-01 00:01", tz="UTC").timestamp() * 1000), 100.0)],
            ),
            "non_finite": (
                "BTCUSDT-15m-2024-01.csv",
                [[*valid[:5], "nan", *valid[6:]]],
            ),
            "negative_volume": (
                "BTCUSDT-15m-2024-01.csv",
                [[*valid[:5], "-1", *valid[6:]]],
            ),
            "non_positive_price": (
                "BTCUSDT-15m-2024-01.csv",
                [[*valid[:4], "0", *valid[5:]]],
            ),
            "ohlc_consistency": (
                "BTCUSDT-15m-2024-01.csv",
                [[*valid[:2], "50", *valid[3:]]],
            ),
            "fractional_trade_count": (
                "BTCUSDT-15m-2024-01.csv",
                [[*valid[:8], "7.5", *valid[9:]]],
            ),
            "taker_buy_exceeds_volume": (
                "BTCUSDT-15m-2024-01.csv",
                [[*valid[:9], "11", *valid[10:]]],
            ),
        }
        for name, (member, rows) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _parse_kline_archive_bytes(
                        _zip_csv(member, rows),
                        source="spot_klines",
                        symbol="BTCUSDT",
                        interval="15m",
                        month="2024-01",
                    )

    def test_pilot_writes_independent_availability_hash_and_all_ledger_record_types(self) -> None:
        capacity = {
            "method": "HTTP HEAD Content-Length; no aggregate-trade payload downloaded",
            "estimated_compressed_bytes_known": 0,
            "sources": {
                source: {
                    "months_requested": 1,
                    "http_200_count": 0,
                    "http_404_count": 1,
                    "known_size_months": 0,
                    "unknown_size_months": 1,
                    "estimated_compressed_bytes": 0,
                    "records": [
                        {
                            "record_type": "d1_aggtrade_head_probe",
                            "source": source,
                            "month": "2024-01",
                            "http_status": 404,
                            "content_length_bytes": 999,
                            "known_size": False,
                            "payload_downloaded": False,
                        }
                    ],
                }
                for source in ("spot_aggTrades", "um_aggTrades")
            },
            "records": [
                {
                    "record_type": "d1_aggtrade_head_probe",
                    "source": "spot_aggTrades",
                    "month": "2024-01",
                    "http_status": 404,
                    "content_length_bytes": 999,
                    "known_size": False,
                    "payload_downloaded": False,
                },
                {
                    "record_type": "d1_aggtrade_head_probe",
                    "source": "um_aggTrades",
                    "month": "2024-01",
                    "http_status": 404,
                    "content_length_bytes": 999,
                    "known_size": False,
                    "payload_downloaded": False,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            args = _pilot_args(tmp)
            with patch(
                "unidream.cli.acquire_d1_signed_flow.download_d1_kline_month",
                side_effect=[
                    (_frame([100.0, 101.0, 102.0]), _source_record("spot_klines")),
                    (_frame([100.5, 101.5, 103.0]), _source_record("um_klines")),
                ],
            ), patch(
                "unidream.cli.acquire_d1_signed_flow.estimate_aggtrade_archive_storage",
                return_value=capacity,
            ):
                result = run_pilot(args)
            availability_path = Path(result["availability"])
            self.assertTrue(availability_path.exists())
            availability_sha = hashlib.sha256(availability_path.read_bytes()).hexdigest()
            records = [
                json.loads(line)
                for line in Path(result["ledger"]).read_text(encoding="utf-8").splitlines()
            ]
            run_record = records[0]
            self.assertEqual(run_record["availability_sha256"], availability_sha)
            self.assertEqual(run_record["availability_path"], str(availability_path))
            self.assertEqual(
                Counter(record["record_type"] for record in records),
                Counter(run_record["ledger_record_counts"]),
            )
            self.assertEqual(
                run_record["ledger_record_counts"]["d1_aggtrade_head_probe"],
                2,
            )
            self.assertIn(availability_sha, Path(result["report"]).read_text(encoding="utf-8"))
            self.assertIn("HTTP 404", Path(result["report"]).read_text(encoding="utf-8"))

    def test_download_failure_is_appended_before_pilot_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _pilot_args(tmp)
            with patch(
                "unidream.cli.acquire_d1_signed_flow.download_d1_kline_month",
                side_effect=[
                    (pd.DataFrame(), _source_record("spot_klines", error="checksum mismatch")),
                    (_frame([100.5, 101.5, 103.0]), _source_record("um_klines")),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    run_pilot(args)
            records = [
                json.loads(line)
                for line in Path(args.ledger).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                [record["source"] for record in records],
                ["spot_klines", "um_klines"],
            )
            self.assertEqual(records[0]["revision_status"], "initial")
            self.assertEqual(records[0]["error"], "checksum mismatch")

    def test_capacity_probe_uses_head_only_and_sums_known_lengths(self) -> None:
        start, end = "2024-01", "2024-03"
        responses: dict[str, _Response] = {}
        for source in ("spot_aggTrades", "um_aggTrades"):
            for month, size in (("2024-01", 100), ("2024-02", 200)):
                url = aggtrade_archive_url(source, "BTCUSDT", month)
                status = 404 if source == "um_aggTrades" and month == "2024-02" else 200
                responses[url] = _Response(
                    b"",
                    url=url,
                    status_code=status,
                    headers={"Content-Length": str(size)},
                )
        session = _Session({}, responses)
        report = estimate_aggtrade_archive_storage(
            symbol="BTCUSDT",
            start=start,
            end=end,
            session=session,
        )
        self.assertEqual(report["estimated_compressed_bytes_known"], 400)
        self.assertEqual(len(session.head_calls), 4)
        self.assertTrue(all(not call[0].endswith(".CHECKSUM") for call in session.head_calls))
        self.assertTrue(all(not call[0].startswith("https://data-api") for call in session.head_calls))
        rejected = report["sources"]["um_aggTrades"]["records"][1]
        self.assertEqual(rejected["http_status"], 404)
        self.assertEqual(rejected["content_length_bytes"], 200)
        self.assertFalse(rejected["known_size"])
        self.assertEqual(report["sources"]["um_aggTrades"]["unknown_size_months"], 1)

    def test_revision_classification_preserves_replacement(self) -> None:
        record = {
            "record_type": "d1_archive_download",
            "source": "spot_klines",
            "month": "2024-01",
            "archive_revision_id": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            first = classify_archive_revisions([record], ledger_path=ledger)
            ledger.write_text(json_line(first[0]) + "\n", encoding="utf-8")
            same = classify_archive_revisions([record], ledger_path=ledger)
            self.assertEqual(same[0]["revision_status"], "unchanged")
            replacement = dict(record, archive_revision_id="b" * 64)
            changed = classify_archive_revisions([replacement], ledger_path=ledger)
            self.assertEqual(changed[0]["revision_status"], "replaced")
            self.assertEqual(changed[0]["previous_archive_revision_id"], "a" * 64)


def json_line(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    unittest.main()
