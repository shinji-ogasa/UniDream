"""Tests for official raw-to-v4 rebuild helpers without network access."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from unidream.data.official_v4_sources import OFFICIAL_SPOT_REST_BASE
from unidream.data.rebuild_v4 import (
    OfficialSourceError,
    _quarantine_off_grid_spot,
    build_full_grid_availability,
    compute_v4_frames,
    recover_spot_gaps,
)
from unidream.cli.rebuild_official_v4_cache import build_parser


def _spot_row(timestamp: int) -> list:
    return [
        timestamp,
        "1",
        "1.1",
        "0.9",
        "1.05",
        "10",
        timestamp + 899999,
        "10",
        1,
        "5",
        "5",
        "0",
    ]


class _Response:
    def __init__(self, payload: list, url: str):
        self.content = b"official-rest-payload"
        self.status_code = 200
        self.url = url
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload: list):
        self.payload = payload
        self.calls: list[tuple[str, dict, float]] = []

    def get(self, url: str, params=None, timeout: float = 0.0):
        self.calls.append((url, dict(params or {}), timeout))
        return _Response(self.payload, url)


class RebuildV4Test(unittest.TestCase):
    def test_off_grid_quarantine_requires_explicit_cli_flag(self) -> None:
        self.assertFalse(build_parser().parse_args([]).allow_off_grid_quarantine)
        self.assertTrue(
            build_parser().parse_args(["--allow-off-grid-quarantine"]).allow_off_grid_quarantine
        )

    def test_availability_uses_asof_external_data_and_keeps_spot_gap_false(self) -> None:
        expected = pd.date_range("2020-01-01", periods=4, freq="15min")
        availability = build_full_grid_availability(
            expected,
            spot_index=expected.delete(2),
            mark_index=pd.DatetimeIndex([expected[1], expected[2]]),
            funding_index=pd.DatetimeIndex([expected[1]]),
            interval="15m",
        )
        self.assertEqual(availability["spot_bar_observed"].tolist(), [True, True, False, True])
        self.assertEqual(availability["mark_close_available"].tolist(), [False, False, True, True])
        self.assertEqual(availability["funding_rate_available"].tolist(), [False, False, True, True])
        self.assertTrue(all(dtype == bool for dtype in availability.dtypes))

    def test_mark_availability_requires_exact_causal_decision_timestamp(self) -> None:
        expected = pd.date_range("2020-01-01", periods=4, freq="15min")
        base = dict(
            expected=expected,
            spot_index=expected,
            funding_index=pd.DatetimeIndex([]),
            interval="15m",
        )
        mark_at_previous = build_full_grid_availability(
            **base,
            mark_index=pd.DatetimeIndex([expected[1]]),
        )
        mark_at_target = build_full_grid_availability(
            **base,
            mark_index=pd.DatetimeIndex([expected[2]]),
        )
        # Row t=00:30 consumes the exact mark observation at decision time
        # t-15m=00:15; a mark stamped at t itself is not causal evidence.
        self.assertTrue(mark_at_previous.loc[expected[2], "mark_close_available"])
        self.assertFalse(mark_at_target.loc[expected[2], "mark_close_available"])

    def test_funding_availability_accepts_exact_eight_hour_asof_boundary(self) -> None:
        expected = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-01 08:15:00",
                "2020-01-01 08:30:00",
            ]
        )
        availability = build_full_grid_availability(
            expected,
            spot_index=expected,
            mark_index=expected - pd.Timedelta(minutes=15),
            funding_index=pd.DatetimeIndex(["2020-01-01 00:00:00"]),
            interval="15m",
        )
        # t=08:15 has decision time 08:00, exactly 8h after publication;
        # 08:30 is 8h15m old and must be unavailable.
        self.assertEqual(availability["funding_rate_available"].tolist(), [False, True, False])

    def test_rest_recovers_only_expected_gap_rows_without_interpolation(self) -> None:
        expected = pd.date_range("2018-01-01", periods=4, freq="15min")
        spot = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.1, 1.1, 1.1],
                "low": [0.9, 0.9, 0.9],
                "close": [1.05, 1.05, 1.05],
                "volume": [10.0, 10.0, 10.0],
            },
            index=expected.delete(2),
        )
        payload = [_spot_row(int(value.timestamp() * 1000)) for value in expected]
        session = _Session(payload)
        merged, gaps, records = recover_spot_gaps(
            spot,
            expected=expected,
            symbol="BTCUSDT",
            interval="15m",
            timeout=2.0,
            session=session,
        )
        self.assertEqual(len(merged), len(expected))
        self.assertEqual(gaps[0]["official_rest_covered_count"], 1)
        self.assertEqual(gaps[0]["official_rest_missing_count"], 0)
        self.assertEqual(records[0]["source"], "spot_klines_rest")
        self.assertEqual(session.calls[0][0], OFFICIAL_SPOT_REST_BASE + "/api/v3/klines")

    def test_rest_end_boundary_is_recorded_and_excluded(self) -> None:
        expected = pd.date_range("2018-01-01", periods=4, freq="15min")
        spot = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.1, 1.1, 1.1],
                "low": [0.9, 0.9, 0.9],
                "close": [1.05, 1.05, 1.05],
                "volume": [10.0, 10.0, 10.0],
            },
            index=expected.delete(2),
        )
        payload = [_spot_row(int(value.timestamp() * 1000)) for value in expected]
        payload.append(_spot_row(int((expected[-1] + pd.Timedelta(minutes=15)).timestamp() * 1000)))
        merged, _gaps, records = recover_spot_gaps(
            spot,
            expected=expected,
            symbol="BTCUSDT",
            interval="15m",
            timeout=2.0,
            session=_Session(payload),
        )
        self.assertTrue(merged.index.equals(expected))
        self.assertEqual(records[0]["outside_expected_count"], 1)
        self.assertEqual(records[0]["outside_expected_allowed_timestamps"], [str(expected[-1] + pd.Timedelta(minutes=15))])

    def test_off_grid_quarantine_lists_each_row_without_timestamp_remap(self) -> None:
        expected = pd.date_range("2018-02-09", periods=4, freq="15min")
        off_grid_a = expected[1] + pd.Timedelta(minutes=1)
        off_grid_b = expected[2] + pd.Timedelta(minutes=2)
        index = pd.DatetimeIndex([expected[0], off_grid_a, off_grid_b, expected[3]])
        spot = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [1.1, 2.1, 3.1, 4.1],
                "low": [0.9, 1.9, 2.9, 3.9],
                "close": [1.05, 2.05, 3.05, 4.05],
                "volume": [10.0, 20.0, 30.0, 40.0],
            },
            index=index,
        )
        records = [
            {
                "month": "2018-02",
                "parsed_first_timestamp": str(index[0]),
                "parsed_last_timestamp": str(index[-1]),
            }
        ]
        with self.assertRaisesRegex(OfficialSourceError, "off-grid timestamp"):
            _quarantine_off_grid_spot(
                spot,
                expected=expected,
                source_records=records,
                interval="15m",
                scope_start=expected[0],
                scope_end=expected[-1] + pd.Timedelta(minutes=15),
            )
        quarantined, entries = _quarantine_off_grid_spot(
            spot,
            expected=expected,
            source_records=records,
            interval="15m",
            scope_start=expected[0],
            scope_end=expected[-1] + pd.Timedelta(minutes=15),
            allow_off_grid_quarantine=True,
        )
        self.assertEqual(quarantined.index.tolist(), [expected[0], expected[3]])
        self.assertEqual(entries[0]["quarantined_count"], 2)
        self.assertEqual([row["timestamp"] for row in entries[0]["rows"]], [str(off_grid_a), str(off_grid_b)])
        self.assertEqual(len(entries[0]["row_sha256"]), 2)
        self.assertIn("delta_from_previous", entries[0]["rows"][0])
        self.assertIn("delta_to_next", entries[0]["rows"][1])

    def test_feature_rebuild_keeps_causal_17_column_contract(self) -> None:
        index = pd.date_range("2020-01-01", periods=240, freq="15min")
        close = pd.Series(100.0 + np.arange(len(index)) * 0.01, index=index)
        spot = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 100.0 + np.arange(len(index)),
            },
            index=index,
        )
        mark = pd.DataFrame({"mark_close": close + 0.05}, index=index)
        funding = pd.DataFrame(
            {"funding_rate": np.zeros(len(index) // 32 + 1)},
            index=pd.date_range(index[0], periods=len(index) // 32 + 1, freq="8h"),
        )
        features, returns = compute_v4_frames(
            spot,
            funding=funding,
            mark=mark,
            zscore_window_days=1,
            interval="15m",
        )
        self.assertEqual(features.columns.tolist(), [
            "open_ret", "high_ret", "low_ret", "close_ret", "vol_ret",
            "RSI_14", "macd", "macd_signal", "atr_norm_ret", "atr",
            "rv_4", "rv_16", "rv_96", "funding_rate", "basis",
            "basis_mom", "basis_abs",
        ])
        self.assertTrue(features.index.equals(returns.index))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertTrue(np.isfinite(returns.to_numpy()).all())

    def test_scope_must_be_development_interval(self) -> None:
        with self.assertRaisesRegex(OfficialSourceError, "restricted"):
            # Avoid network: validation happens before source fetches.
            from unidream.data.rebuild_v4 import rebuild_official_v4_frames

            rebuild_official_v4_frames(start="2019-01-01", end="2020-01-01")


if __name__ == "__main__":
    unittest.main()
