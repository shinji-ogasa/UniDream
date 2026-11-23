import copy
import math
import unittest

import torch

from unidream.experiments import wm_tail_transfer_accounting as account


def config(mode="linear_perp", **overrides):
    a = {"mode": mode, "position_min": 0., "position_max": 5., "max_position_step": 5.,
         "initial_position": 1., "cost_rate": 0., "borrow_apr": 0.,
         "execution_delay_bars": 1, "reset_positions_at_segment_boundary": False}
    a.update(overrides)
    return {"accounting": a, "perpetual": {"fee_rate": 0., "liquidation_fee_rate": .005,
            "primary_maintenance_margin_rate": .005, "primary_slippage_rate": 0.}}


def advance(state, request, op=100., low=100., close=100., cfg=None, **kwargs):
    return account.step(state, request, op, low, close, cfg or config(),
                        mark_open=kwargs.pop("mark_open", op), mark_low=kwargs.pop("mark_low", low),
                        mark_close=kwargs.pop("mark_close", close), **kwargs)


class TailTransferAccountingTests(unittest.TestCase):
    def test_perpetual_one_bar_delay_and_inventory_drift(self):
        cfg = config()
        state = account.initial_state(1, "cpu", 100., cfg)
        state, first = advance(state, 2., cfg=cfg)
        self.assertFalse(first["order_filled"].item())
        self.assertEqual(state["units"].item(), .01)
        state, second = advance(state, 2., close=110., cfg=cfg)
        self.assertTrue(second["order_filled"].item())
        self.assertAlmostEqual(state["units"].item(), .02)
        self.assertAlmostEqual(state["prev_equity"].item(), 1.2)
        self.assertAlmostEqual(state["executed"].item(), 2.2/1.2)

    def test_fill_target_is_fee_inclusive_with_different_mark_price(self):
        cfg = config()
        cfg["perpetual"].update(fee_rate=.001, primary_slippage_rate=.002)
        state = account.initial_state(1, "cpu", 100., cfg)
        state, _ = advance(state, 3., cfg=cfg)
        state, diag = advance(state, 3., cfg=cfg, mark_open=101., mark_low=101., mark_close=101.)
        self.assertAlmostEqual(diag["open_exposure"].item(), 3., places=10)
        self.assertAlmostEqual(diag["fee"].item(), abs(diag["trade_notional"].item())*.001, places=12)
        self.assertAlmostEqual(state["cash"].item()+state["units"].item()*101, state["prev_equity"].item(), places=12)

    def test_funding_is_signed_actual_event_once_and_chronological(self):
        cfg = config()
        state = account.initial_state(1, "cpu", 100., cfg)
        state, diag = advance(state, 1., cfg=cfg, funding_rate=.01, funding_event=10, funding_event_count=1)
        self.assertAlmostEqual(diag["funding"].item(), .01)
        self.assertAlmostEqual(state["prev_equity"].item(), .99)
        with self.assertRaisesRegex(ValueError, "already settled"):
            advance(state, 1., cfg=cfg, funding_rate=.01, funding_event=10, funding_event_count=1)
        state, diag = advance(state, 1., cfg=cfg, funding_rate=-.01, funding_event=11, funding_event_count=1,
                              decision_available=False)
        self.assertLess(diag["funding"].item(), 0.)
        with self.assertRaisesRegex(ValueError, "zero rate"):
            advance(state, 1., cfg=cfg, funding_rate=.01, funding_event_count=0)

    def test_missing_decision_keeps_pending_and_price_pnl(self):
        cfg = config()
        state = account.initial_state(1, "cpu", 100., cfg)
        state, _ = advance(state, 2., cfg=cfg)
        state, diag = advance(state, float("nan"), close=110., cfg=cfg, decision_available=False)
        self.assertTrue(diag["order_filled"].item())
        self.assertFalse(state["has_pending"].item())
        self.assertAlmostEqual(state["prev_equity"].item(), 1.2)
        state, diag = advance(state, float("nan"), op=110., low=105., close=105., cfg=cfg, decision_available=False)
        self.assertFalse(diag["order_filled"].item())
        self.assertAlmostEqual(state["prev_equity"].item(), 1.1)

    def test_mark_low_liquidation_uses_trade_low_and_retains_residual(self):
        cfg = config()
        state = account.initial_state(1, "cpu", 100., cfg)
        state, _ = advance(state, 5., cfg=cfg)
        state, diag = advance(state, 5., low=80.3, close=100., mark_low=80.3, cfg=cfg)
        self.assertTrue(diag["liquidation_event"].item())
        self.assertFalse(state["alive"].item())
        expected = max(0., -4+5*.803-5*.803*.005)
        self.assertAlmostEqual(state["prev_equity"].item(), expected)
        self.assertEqual(state["units"].item(), 0.)
        state, after = advance(state, float("nan"), op=float("nan"), low=float("nan"), close=float("nan"), cfg=cfg,
                                decision_available=False)
        self.assertEqual(state["prev_equity"].item(), expected)
        self.assertEqual(after["net_log_return"].item(), 0.)

    def test_mark_maintenance_differs_from_trade_price(self):
        cfg = config()
        state = account.initial_state(1, "cpu", 100., cfg)
        state, _ = advance(state, 5., cfg=cfg)
        state, diag = advance(state, 5., low=100., close=100., mark_low=80., cfg=cfg)
        self.assertTrue(diag["liquidation_event"].item())
        self.assertAlmostEqual(diag["liquidation_fill_price"].item(), 100.)

    def test_cash_mode_is_exact_delegation_and_expanded_inventory_is_finite(self):
        cfg = config("cash_borrow", cost_rate=.00055, borrow_apr=.1)
        state = account.initial_state(2, "cpu", [100., 200.], cfg)
        expected = copy.deepcopy(state)
        state, diag = account.step(state, torch.tensor([2., .5]), [100., 200.], [99., 199.], [101., 201.], cfg)
        expected, old_diag = account.cash_account.step(expected, torch.tensor([2., .5]), [100., 200.], [99., 199.], [101., 201.], cfg)
        self.assertTrue(all(torch.equal(state[k], expected[k]) for k in state))
        self.assertTrue(all(torch.equal(diag[k], v) for k, v in old_diag.items()))
        self.assertEqual(account.inventory(state, cfg, expanded=True).shape, (2, 14))
        self.assertTrue(torch.isfinite(account.inventory(state, cfg, expanded=True)).all())

    def test_invalid_prices_and_missing_mark_are_not_skipped(self):
        cfg = config()
        state = account.initial_state(1, "cpu", 100., cfg)
        with self.assertRaisesRegex(ValueError, "mark OHLC"):
            account.step(state, 1., 100., 99., 100., cfg)
        with self.assertRaisesRegex(ValueError, "valid trade"):
            advance(state, 1., low=float("nan"), cfg=cfg)
        with self.assertRaisesRegex(ValueError, "bounds"):
            advance(state, 6., cfg=cfg)

    def test_batch_events_are_account_specific(self):
        cfg = config()
        state = account.initial_state(2, "cpu", 100., cfg)
        state, diag = advance(state, torch.tensor([1., 1.]), cfg=cfg, funding_rate=torch.tensor([.01, 0.]),
                              funding_event=torch.tensor([10, -1]), funding_event_count=torch.tensor([1, 0]))
        self.assertEqual(state["last_funding_event"].tolist(), [10, -1])
        self.assertAlmostEqual(diag["funding"].sum().item(), .01, places=8)


if __name__ == "__main__":
    unittest.main()
