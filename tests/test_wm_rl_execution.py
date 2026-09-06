import copy
import json
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import _simulate
from unidream.experiments.wm_rl_execution import (
    AccountState, CashUnitAccount, execution_contract,
)


def fixture(n=160):
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC")
    k = np.arange(n, dtype=float)
    opens = 100 * np.exp(0.002 * np.sin(k/3) + 0.0005*k)
    closes = opens * np.exp(0.002 * np.cos(k/5))
    targets = np.where((np.arange(n)//16) % 2 == 0, 0.5, 1.12)
    schedule = np.arange(n) % 4 == 0
    return index, opens, closes, targets, schedule


def incremental(index, opens, closes, targets, schedule, cost=0.00055, restart=False):
    borrow_annual = 0.2 if cost == 0.0011 else 0.1
    account = CashUnitAccount(index[0], opens[0], one_way_cost=cost, borrow_annual=borrow_annual)
    rows = []
    feedback = []
    for i, timestamp in enumerate(index):
        feedback.append(account.decision_feedback(timestamp))
        rows.append(account.advance_bar(timestamp, opens[i], closes[i],
            open_observed=bool(np.isfinite(opens[i])), close_observed=bool(np.isfinite(closes[i])),
            intent_for_next_open=targets[i] if schedule[i] else None))
        if restart:
            account = CashUnitAccount(state=json.loads(json.dumps(account.state.to_dict(), allow_nan=False)),
                                      one_way_cost=cost, borrow_annual=borrow_annual)
    return account, rows, feedback


class IncrementalAccountTests(unittest.TestCase):
    def assert_canonical(self, opens, closes, targets, schedule, cost, rows, state):
        expected = _simulate(opens, closes, targets, schedule, cost, 0.2 if cost == 0.0011 else 0.1, 0.08, 0.01)
        observed_equity = np.array([np.nan if r["equity"] is None else r["equity"] for r in rows])
        observed_exposure = np.array([np.nan if r["exposure"] is None else r["exposure"] for r in rows])
        np.testing.assert_allclose(observed_equity, expected[0], rtol=0, atol=2e-14, equal_nan=True)
        np.testing.assert_allclose(observed_exposure, expected[1], rtol=0, atol=2e-14, equal_nan=True)
        np.testing.assert_allclose([state.turnover, state.fees, state.borrow], expected[2:5], rtol=0, atol=2e-14)
        self.assertEqual(state.trades, expected[5])

    def test_complete_price_paths_match_canonical_both_costs(self):
        index, opens, closes, targets, schedule = fixture()
        for cost in (0.00055, 0.0011):
            account, rows, _ = incremental(index, opens, closes, targets, schedule, cost)
            self.assert_canonical(opens, closes, targets, schedule, cost, rows, account.state)
            self.assertGreater(account.state.trades, 0)
            self.assertGreater(account.state.borrow, 0)

    def test_gap_paths_match_canonical_without_compressing_time(self):
        index, opens, closes, targets, schedule = fixture()
        opens[[5, 6, 21, 70, 71]] = np.nan
        closes[[6, 17, 18, 25, 70, 71]] = np.nan
        for cost in (0.00055, 0.0011):
            account, rows, _ = incremental(index, opens, closes, targets, schedule, cost)
            self.assert_canonical(opens, closes, targets, schedule, cost, rows, account.state)
            self.assertEqual(rows[5]["fill"]["status"], "expired_missing_open")
            self.assertEqual(rows[6]["fill"]["status"], "none")
            self.assertEqual(rows[17]["fill"]["status"], "filled")
            self.assertIsNone(rows[17]["equity"])

    def test_initial_inventory_and_one_bar_delay(self):
        index, opens, closes, _, _ = fixture(3)
        account = CashUnitAccount(index[0], opens[0])
        feedback = account.decision_feedback(index[0])
        self.assertEqual(feedback["actual_exposure"], 1.0)
        self.assertTrue(feedback["actor_account_available"])
        first = account.advance_bar(index[0], opens[0], closes[0], open_observed=True,
            close_observed=True, intent_for_next_open=0.5)
        self.assertEqual(first["trades"], 0)
        self.assertEqual(first["cash"], 0)
        second = account.advance_bar(index[1], opens[1], closes[1], open_observed=True, close_observed=True)
        self.assertEqual(second["trades"], 1)
        self.assertAlmostEqual(second["fill"]["exposure_before"], 1)
        self.assertAlmostEqual(second["fill"]["exposure_after"], 0.92)
        self.assertAlmostEqual(second["fill"]["executed_delta"], -0.08)

    def test_missing_due_open_expires_without_later_fill(self):
        index, _, _, _, _ = fixture(3)
        account = CashUnitAccount(index[0], 100)
        account.advance_bar(index[0], 100, 100, open_observed=True, close_observed=True, intent_for_next_open=0.5)
        missing = account.advance_bar(index[1], None, None, open_observed=False, close_observed=False)
        self.assertEqual(missing["fill"]["status"], "expired_missing_open")
        self.assertIsNone(account.state.pending_target)
        restored = account.advance_bar(index[2], 110, 111, open_observed=True, close_observed=True)
        self.assertEqual(restored["trades"], 0)
        self.assertEqual(restored["units"], 0.01)

    def test_missing_close_does_not_suppress_fill_or_borrow(self):
        index, _, _, _, _ = fixture(4)
        account = CashUnitAccount(index[0], 100)
        account.advance_bar(index[0], 100, 100, open_observed=True, close_observed=True, intent_for_next_open=1.12)
        row = account.advance_bar(index[1], 100, None, open_observed=True, close_observed=False)
        self.assertEqual(row["fill"]["status"], "filled")
        self.assertGreater(row["borrow_charge"], 0)
        feedback = account.decision_feedback(index[2])
        self.assertFalse(feedback["actor_account_available"])
        self.assertFalse(feedback["current_close_observed"])
        self.assertTrue(feedback["valuation_available"])
        self.assertTrue(feedback["stale_valuation"])
        self.assertEqual(feedback["mark_source"], "open")
        self.assertEqual(feedback["mark_age_bars"], 1)
        gap = account.advance_bar(index[2], None, None, open_observed=False, close_observed=False)
        self.assertGreater(gap["borrow_charge"], 0)
        self.assertGreater(gap["borrow_charge"], row["borrow_charge"])
        self.assertEqual(account.decision_feedback(index[3])["mark_age_bars"], 2)

    def test_fill_delta_excludes_price_drift_and_is_known_only_next_event(self):
        index, _, _, _, _ = fixture(4)
        account = CashUnitAccount(index[0], 100)
        account.advance_bar(index[0], 100, 100, open_observed=True, close_observed=True, intent_for_next_open=1.12)
        self.assertEqual(account.decision_feedback(index[1])["executed_delta"], 0)
        account.advance_bar(index[1], 100, 105, open_observed=True, close_observed=True)
        feedback = account.decision_feedback(index[2])
        self.assertAlmostEqual(feedback["executed_delta"], 0.08)
        self.assertNotEqual(feedback["actual_exposure"], 1.08)
        account.advance_bar(index[2], 95, 90, open_observed=True, close_observed=True)
        self.assertEqual(account.decision_feedback(index[3])["executed_delta"], 0)

    def test_restart_has_exact_canonical_state_and_events(self):
        index, opens, closes, targets, schedule = fixture(50)
        closes[17] = np.nan
        opens[21] = np.nan
        uninterrupted, rows, feedback = incremental(index, opens, closes, targets, schedule)
        restarted, other_rows, other_feedback = incremental(index, opens, closes, targets, schedule, restart=True)
        self.assertEqual(rows, other_rows)
        self.assertEqual(feedback, other_feedback)
        self.assertEqual(uninterrupted.state, restarted.state)

    def test_future_prices_and_intents_do_not_change_prefix(self):
        index, opens, closes, targets, schedule = fixture(60)
        _, full, feedback = incremental(index, opens, closes, targets, schedule)
        _, prefix, prefix_feedback = incremental(index[:30], opens[:30], closes[:30], targets[:30], schedule[:30])
        self.assertEqual(full[:30], prefix)
        self.assertEqual(feedback[:30], prefix_feedback)
        opens[30:], closes[30:], targets[30:] = 200, 210, 1
        _, changed, changed_feedback = incremental(index, opens, closes, targets, schedule)
        self.assertEqual(full[:30], changed[:30])
        self.assertEqual(feedback[:31], changed_feedback[:31])

    def test_no_intents_is_exact_existing_buy_and_hold(self):
        index, opens, closes, _, schedule = fixture(20)
        account, rows, _ = incremental(index, opens, closes, np.full(20, np.nan), schedule)
        np.testing.assert_array_equal([r["equity"] for r in rows], closes * (1 / opens[0]))
        self.assertEqual(account.state.trades, 0)
        self.assertEqual(account.state.borrow, 0)
        self.assertEqual(account.state.fees, 0)

    def test_malformed_prices_time_and_intents_are_atomic_rejections(self):
        t = pd.Timestamp("2021-01-01T00:00Z")
        account = CashUnitAccount(t, 100)
        prior = account.state
        for kwargs in (dict(open_price=100, close_price=100, open_observed=False, close_observed=True),
                       dict(open_price=100, close_price=True, open_observed=True, close_observed=True),
                       dict(open_price=100, close_price=np.inf, open_observed=True, close_observed=True),
                       dict(open_price=100, close_price=100, open_observed=True, close_observed=True, intent_for_next_open=0.4)):
            with self.assertRaises(ValueError):
                account.advance_bar(t, **kwargs)
            self.assertEqual(prior, account.state)
        for bad in (t + pd.Timedelta(minutes=15), t.tz_localize(None), t.value):
            with self.assertRaises(ValueError):
                account.decision_feedback(bad)
        with self.assertRaises(ValueError):
            account.advance_bar(t, 101, 102, open_observed=True, close_observed=True)
        self.assertEqual(prior, account.state)

    def test_duplicate_or_compressed_gap_rejected(self):
        t = pd.Timestamp("2021-01-01T00:00Z")
        account = CashUnitAccount(t, 100)
        account.advance_bar(t, 100, 100, open_observed=True, close_observed=True)
        for bad in (t, t + pd.Timedelta(minutes=30)):
            with self.assertRaises(ValueError):
                account.advance_bar(bad, 100, 100, open_observed=True, close_observed=True)

    def test_restore_pins_contract_and_pending_due_clock(self):
        t = pd.Timestamp("2021-01-01T00:00Z")
        account = CashUnitAccount(t, 100)
        account.advance_bar(t, 100, 100, open_observed=True, close_observed=True, intent_for_next_open=0.5)
        envelope = account.state.to_dict()
        for key, value in (("max_step", 0.2), ("fill_delay_bars", 0)):
            bad = copy.deepcopy(envelope)
            bad["contract"][key] = value
            with self.assertRaises(ValueError):
                AccountState.from_dict(bad)
        for key, value in (("pending_due_ns", t.value), ("bars_processed", 5), ("trades", True),
                           ("last_equity", 2.0), ("last_exposure", 0.5),
                           ("last_bar_timestamp_ns", float(t.value))):
            bad = copy.deepcopy(envelope)
            bad["account"][key] = value
            with self.assertRaises(ValueError):
                AccountState.from_dict(bad)
        with self.assertRaises(ValueError):
            CashUnitAccount(state=envelope, one_way_cost=0.0011)
        with self.assertRaises(ValueError):
            CashUnitAccount(state=envelope, borrow_annual=0.2)
        with self.assertRaises(ValueError):
            execution_contract(borrow_annual=-0.1)

    def test_native_float32_bound_is_canonicalized_without_expanding_bounds(self):
        t = pd.Timestamp("2021-01-01T00:00Z")
        account = CashUnitAccount(t, 100)
        account.advance_bar(t, 100, 100, open_observed=True, close_observed=True,
                            intent_for_next_open=float(np.float32(1.12)))
        self.assertEqual(account.state.pending_target, 1.12)
        self.assertEqual(execution_contract()["fill_delay_bars"], 1)

    def test_drift_above_target_bound_rebalances_by_only_max_step(self):
        index = pd.date_range("2021-01-01", periods=4, freq="15min", tz="UTC")
        account = CashUnitAccount(index[0], 100)
        for t in index[:3]:
            account.advance_bar(t, 100, 100, open_observed=True, close_observed=True,
                                intent_for_next_open=1.12)
        row = account.advance_bar(index[3], 45, 45, open_observed=True, close_observed=True)
        self.assertGreater(row["fill"]["exposure_before"], 1.2)
        self.assertAlmostEqual(row["fill"]["executed_delta"], -0.08)
        self.assertGreater(row["fill"]["exposure_after"], 1.12)

    def test_open_insolvency_halts_without_pending_trade_or_new_order(self):
        index = pd.date_range("2021-01-01", periods=4, freq="15min", tz="UTC")
        account = CashUnitAccount(index[0], 100)
        for t in index[:2]:
            account.advance_bar(t, 100, 100, open_observed=True, close_observed=True,
                                intent_for_next_open=1.12)
        prior_trades = account.state.trades
        row = account.advance_bar(index[2], 1, 1, open_observed=True, close_observed=True,
                                  intent_for_next_open=0.5)
        self.assertTrue(row["insolvent"])
        self.assertIsNone(row["equity"])
        self.assertIsNone(account.state.pending_target)
        self.assertEqual(account.state.trades, prior_trades)
        with self.assertRaises(ValueError):
            account.decision_feedback(index[3])


if __name__ == "__main__":
    unittest.main()
