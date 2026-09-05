import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_receipt_support import receipt_support


STEP = pd.Timedelta(minutes=15)


def metadata(n=3):
    opens = pd.date_range("2024-01-01 12:00", periods=n, freq="15min", tz="UTC")
    return opens, opens + STEP - pd.Timedelta(milliseconds=1), opens + STEP


class OracleReceiptSupportTests(unittest.TestCase):
    def test_deadline_admits_delayed_final_receipt_without_admitting_current_bar(self):
        opens, closes, ends = metadata(2)
        deadline = ends[0] + pd.Timedelta(seconds=60)
        result = receipt_support(opens, closes, [ends[0] + pd.Timedelta(seconds=1), None],
                                 ends[0], decision_deadline=deadline)
        self.assertEqual(result.receipt_eligible.tolist(), [True, False])
        self.assertEqual(result.reason.tolist(), ["ELIGIBLE", "BAR_NOT_CLOSED"])
        self.assertTrue((result.decision_deadline == deadline).all())
        late = receipt_support(opens[:1], closes[:1], deadline + pd.Timedelta(nanoseconds=1),
                               ends[0], decision_deadline=deadline)
        self.assertEqual(late.reason.iloc[0], "RECEIPT_LATE")

    def test_deadline_must_precede_unknown_next_fill_and_not_precede_decision(self):
        opens, closes, ends = metadata(1)
        for deadline in (ends[0] - pd.Timedelta(nanoseconds=1), ends[0] + STEP,
                         ends[0] + 2 * STEP, pd.NaT, ends[0].tz_localize(None)):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                receipt_support(opens, closes, ends, ends, decision_deadline=deadline)

    def test_exact_receipt_deadline_and_one_nanosecond_late(self):
        opens, closes, decisions = metadata()
        receipts = pd.DatetimeIndex([decisions[0], decisions[1] + pd.Timedelta(nanoseconds=1), decisions[2]])
        result = receipt_support(opens, closes, receipts, decisions)
        self.assertEqual(result.receipt_eligible.tolist(), [True, False, True])
        self.assertEqual(result.reason.tolist(), ["ELIGIBLE", "RECEIPT_LATE", "ELIGIBLE"])
        self.assertTrue(result.event_time_eligible.all())
        self.assertTrue(result.receipt_known.all())
        self.assertEqual(result.receipt_late.tolist(), [False, True, False])
        self.assertEqual(result.received_at.iloc[1], receipts[1])

    def test_current_and_future_bars_are_unclosed_features_not_missing_current_open(self):
        opens, closes, ends = metadata()
        result = receipt_support(opens, closes, [ends[0], None, None], ends[0])
        self.assertEqual(result.is_prior_bar.tolist(), [True, False, False])
        self.assertEqual(result.bar_closed_by_decision.tolist(), [True, False, False])
        self.assertEqual(result.reason.tolist(), ["ELIGIBLE", "BAR_NOT_CLOSED", "BAR_NOT_CLOSED"])
        self.assertFalse(result.attrs["current_open_availability_evaluated"])
        self.assertEqual(result.index[1], result.decision_at.iloc[1])  # Current open has its own observation contract.

    def test_unknown_receipt_is_archive_event_time_only_and_never_neighbor_filled(self):
        opens, closes, ends = metadata()
        result = receipt_support(opens, closes, [ends[0], pd.NaT, ends[2]], ends[-1])
        self.assertTrue(result.event_time_eligible.all())
        self.assertEqual(result.receipt_eligible.tolist(), [True, False, True])
        self.assertEqual(result.archive_event_time_only.tolist(), [False, True, False])
        self.assertEqual(result.reason.iloc[1], "RECEIPT_UNKNOWN")
        self.assertTrue(pd.isna(result.received_at.iloc[1]))
        unknown = receipt_support(opens, closes, None, ends[-1])
        self.assertFalse(unknown.receipt_eligible.any())
        self.assertTrue(unknown.archive_event_time_only.all())
        self.assertFalse(unknown.attrs["receipt_authenticity_verified"])

    def test_later_arrival_does_not_retroactively_backfill_earlier_decision(self):
        opens, closes, ends = metadata(1)
        arrival = ends[0] + pd.Timedelta(minutes=1)
        earlier = receipt_support(opens, closes, arrival, ends[0])
        earlier_snapshot = earlier.copy(deep=True)
        later = receipt_support(opens, closes, arrival, ends[0] + STEP)
        self.assertFalse(earlier.receipt_eligible.iloc[0])
        self.assertEqual(earlier.reason.iloc[0], "RECEIPT_LATE")
        self.assertTrue(later.receipt_eligible.iloc[0])
        pd.testing.assert_frame_equal(earlier, earlier_snapshot)

    def test_scalar_broadcast_and_aligned_arrays_are_equivalent(self):
        opens, closes, ends = metadata()
        scalar = receipt_support(opens, closes, ends[-1], ends[-1])
        aligned = receipt_support(opens, closes, [ends[-1]] * 3, [ends[-1]] * 3)
        pd.testing.assert_frame_equal(scalar, aligned)
        self.assertTrue(scalar.receipt_eligible.all())

    def test_timezone_conversion_and_nanosecond_metadata_preserve_input_arrays(self):
        opens, closes, ends = metadata()
        receipts = ends + pd.Timedelta(microseconds=500)
        original = [v.copy() for v in (opens, closes, receipts, ends)]
        expected = receipt_support(opens, closes, receipts, ends + STEP)
        converted = receipt_support(opens.tz_convert("Asia/Tokyo"), closes.tz_convert("Asia/Tokyo"),
                                    receipts.tz_convert("Asia/Tokyo"), (ends + STEP).tz_convert("Asia/Tokyo"))
        pd.testing.assert_frame_equal(expected, converted)
        for actual, before in zip((opens, closes, receipts, ends), original):
            self.assertTrue(actual.equals(before))
        self.assertEqual(str(expected.index.tz), "UTC")
        self.assertTrue(expected.receipt_eligible.all())

    def test_sparse_supplied_rows_do_not_claim_a_complete_feature_window(self):
        opens, closes, ends = metadata(4)
        chosen = [0, 3]
        result = receipt_support(opens[chosen], closes[chosen], ends[chosen], ends[-1])
        self.assertEqual(len(result), 2)
        self.assertTrue(result.receipt_eligible.all())
        self.assertFalse(result.attrs["full_feature_history_verified"])
        self.assertFalse(result.attrs["bar_values_validated"])
        self.assertEqual(result.index.tolist(), opens[chosen].tolist())

    def test_future_metadata_changes_cannot_change_existing_row_eligibility(self):
        opens, closes, ends = metadata(4)
        before = receipt_support(opens, closes, ends, ends)
        changed_receipts = list(ends)
        changed_receipts[2:] = [None, ends[3] + STEP]
        changed_decisions = list(ends)
        changed_decisions[3] += STEP
        after = receipt_support(opens, closes, changed_receipts, changed_decisions)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
        self.assertFalse(after.receipt_eligible.iloc[2])

    def test_inconsistent_bar_close_or_premature_final_receipt_is_rejected(self):
        opens, closes, ends = metadata()
        for changed_close in (ends, opens, closes + pd.Timedelta(milliseconds=1),
                              pd.DatetimeIndex([closes[0], pd.NaT, closes[2]])):
            with self.subTest(close=changed_close), self.assertRaises(ValueError):
                receipt_support(opens, changed_close, ends, ends)
        for premature in (closes, opens, ends - pd.Timedelta(nanoseconds=1)):
            with self.subTest(received=premature), self.assertRaisesRegex(ValueError, "finalized-bar receipt"):
                receipt_support(opens, closes, premature, ends)

    def test_duplicate_unordered_naive_or_off_grid_metadata_fails_closed(self):
        opens, closes, ends = metadata()
        valid = dict(event_open=opens, event_close=closes, received_at=ends, decision_at=ends)
        cases = [("event_open", opens[[0, 0, 2]]), ("event_open", opens[::-1]),
                 ("event_open", opens + pd.Timedelta(seconds=1)),
                 ("decision_at", ends + pd.Timedelta(nanoseconds=1)),
                 ("decision_at", pd.NaT)]
        for name in valid:
            cases.append((name, valid[name].tz_localize(None)))
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                receipt_support(**{**valid, name: value})

    def test_empty_malformed_misaligned_inputs_and_unregistered_step_are_rejected(self):
        opens, closes, ends = metadata()
        valid = dict(event_open=opens, event_close=closes, received_at=ends, decision_at=ends)
        for name in valid:
            for value in ([], [ends[0]], [[ends[0], ends[1], ends[2]]]):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    receipt_support(**{**valid, name: value})
        for name in ("event_open", "event_close"):
            with self.assertRaisesRegex(ValueError, "array"):
                receipt_support(**{**valid, name: valid[name][0]})
        for step in ("1min", "30min", 0, None, "invalid"):
            with self.subTest(step=step), self.assertRaises(ValueError):
                receipt_support(**valid, step=step)


if __name__ == "__main__":
    unittest.main()
