"""Targeted causal-source checks for the isolated WM ablation data contract."""
from __future__ import annotations

import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from unidream.experiments.wm_ablation_data import (
    ArchiveJob, causal_asof, deduplicate, parse_archive,
)


def _zip(text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("BTCUSDT.csv", text)
    return output.getvalue()


class TestWMAblationData(unittest.TestCase):
    def test_asof_is_lagged_from_close_and_expires(self) -> None:
        grid = pd.date_range("2020-09-01", periods=5, freq="15min", tz="UTC", name="bar_open")
        # The second observation is after the first bar OPEN, although before
        # its decision time. It must not be included in that first decision.
        event_times = pd.DatetimeIndex([grid[0] - pd.Timedelta(minutes=5), grid[0] + pd.Timedelta(minutes=5)])
        events = pd.DataFrame({"value": [1.0, 2.0]}, index=event_times)
        result, times = causal_asof(events, grid, pd.Timedelta(minutes=30))
        np.testing.assert_equal(result["value"].to_numpy(), [1.0, 2.0, 2.0, np.nan, np.nan])
        self.assertTrue((times.dropna() <= grid[times.notna()]).all())
        self.assertTrue((times.dropna() < grid[times.notna()] + pd.Timedelta(minutes=15)).all())

    def test_conflicting_timestamps_are_quarantined(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(["2020-09-01", "2020-09-01", "2020-09-02", "2020-09-02"], tz="UTC", name="bar_open"))
        clean, conflicts, audit = deduplicate(frame)
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(audit["identical_duplicates_removed"], 1)
        self.assertEqual(audit["conflicting_timestamps_quarantined"], 1)

    def test_latest_partial_record_does_not_fill_from_older_record(self) -> None:
        grid = pd.date_range("2020-09-01", periods=2, freq="15min", tz="UTC", name="bar_open")
        times = pd.DatetimeIndex([grid[0], grid[0] + pd.Timedelta(minutes=5)])
        events = pd.DataFrame({"oi": [1.0, 2.0], "ratio": [1.5, np.nan]}, index=times)
        aligned, _ = causal_asof(events, grid, pd.Timedelta(minutes=30))
        self.assertEqual(aligned.iloc[1]["oi"], 2.0)
        self.assertTrue(pd.isna(aligned.iloc[1]["ratio"]))

    def test_spot_microseconds_and_invalid_candles(self) -> None:
        start = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp() * 1_000_000)
        good = f"{start},100,101,99,100,2,{start+899999999},200,3,1,100,0"
        # Invalid OHLC must remain missing rather than silently clipped.
        bad = f"{start+900000000},100,90,99,100,2,{start+1799999999},200,3,1,100,0"
        frame, audit, conflicts = parse_archive(ArchiveJob("spot", "2026-10", "unused"), _zip(good + "\n" + bad + "\n"))
        self.assertEqual(frame.index[0], pd.Timestamp("2026-10-01", tz="UTC"))
        self.assertEqual(frame.iloc[0]["spot_taker_buy_quote"], 100)
        self.assertTrue(frame.iloc[1].isna().all())
        self.assertEqual(audit["invalid_candle_rows"], 1)
        self.assertTrue(conflicts.empty)

    def test_metrics_sort_identical_duplicate_and_conflict(self) -> None:
        header = "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        a = "2020-09-01 00:00:00,BTCUSDT,1,2,3,4,5,6\n"
        b = "2020-09-01 00:05:00,BTCUSDT,2,2,3,4,5,6\n"
        c = "2020-09-01 00:05:00,BTCUSDT,3,2,3,4,5,6\n"
        frame, audit, conflicts = parse_archive(ArchiveJob("metrics", "2020-09-01", "unused"), _zip(header + b + a + a + c))
        self.assertEqual(len(frame), 1)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(audit["identical_duplicates_removed"], 1)
        self.assertEqual(frame.iloc[0]["metrics_sum_open_interest"], 1)

    def test_partial_metric_missingness_preserves_other_columns(self) -> None:
        header = "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        row = "2022-02-01 00:00:00,BTCUSDT,100,200,,,1.2,\n"
        frame, audit, _ = parse_archive(ArchiveJob("metrics", "2022-02-01", "unused"), _zip(header + row))
        self.assertEqual(frame.iloc[0]["metrics_sum_open_interest"], 100)
        self.assertEqual(frame.iloc[0]["metrics_count_long_short_ratio"], 1.2)
        self.assertTrue(pd.isna(frame.iloc[0]["metrics_count_toptrader_long_short_ratio"]))
        self.assertEqual(audit["nonfinite_rows"], 1)

    def test_records_outside_requested_archive_period_are_excluded(self) -> None:
        header = "calc_time,funding_interval_hours,last_funding_rate\n"
        previous = int(pd.Timestamp("2020-08-31", tz="UTC").timestamp() * 1000)
        current = int(pd.Timestamp("2020-09-01", tz="UTC").timestamp() * 1000)
        rows = f"{previous},8,0.1\n{current},8,0.2\n"
        frame, audit, _ = parse_archive(ArchiveJob("funding", "2020-09", "unused"), _zip(header + rows))
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["funding_rate"], 0.2)
        self.assertEqual(audit["outside_requested_period_rows"], 1)


if __name__ == "__main__":
    unittest.main()
