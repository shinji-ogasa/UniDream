"""Independent scalar cash/units checks for full-calendar control research."""
import copy
import math
import unittest

import numpy as np
import torch

from unidream.experiments import wm_control_release_accounting as account


def scalar_initial(price):
    return dict(requested=1., cash=0., units=1 / price, equity=1., peak=1., max_dd=0., alive=True, pending=False)


def scalar_step(state, request, opening, low, close, settings, available=True):
    """NumPy-free scalar ledger; derives target units directly after fee."""
    s = copy.deepcopy(state)
    if not s["alive"]:
        return s, dict(equity=0., fee=0., borrow=0., turnover=0., net_simple=0.)
    before = s["cash"] + s["units"] * opening
    fee = trade = borrow = 0.
    if before > 0 and s["pending"]:
        old_value = s["units"] * opening
        difference = s["requested"] * before - old_value
        direction = 1 if difference > 0 else -1 if difference < 0 else 0
        # E_after=(E_before + fee_rate*direction*old_value)/(1+fee_rate*direction*target)
        after = (before + settings["cost_rate"] * direction * old_value) / (1 + settings["cost_rate"] * direction * s["requested"])
        new_units = s["requested"] * after / opening
        trade = (new_units - s["units"]) * opening
        fee = settings["cost_rate"] * abs(trade)
        s["units"] = new_units
        s["cash"] -= trade + fee
    if before > 0:
        borrow = max(-s["cash"], 0.) * math.expm1(settings["borrow_apr"] / settings["annual_bars"])
        s["cash"] -= borrow
    at_low = s["cash"] + s["units"] * low
    nav = s["cash"] + s["units"] * close
    dead = min(before, at_low, nav) <= 0
    equity = 0. if dead else nav
    result = dict(equity=equity, fee=fee, borrow=borrow, turnover=abs(trade) / before if before > 0 else 0.,
                  net_simple=equity / s["equity"] - 1)
    s.update(equity=equity, alive=not dead, pending=bool(available and not dead))
    s["peak"] = max(s["peak"], equity)
    s["max_dd"] = max(s["max_dd"], 1 - equity / s["peak"])
    if dead:
        s.update(cash=0., units=0., requested=0.)
    elif available:
        s["requested"] = request
    return s, result


class ControlReleaseAccountingTests(unittest.TestCase):
    def setUp(self):
        self.a = {"position_min": 0., "position_max": 2., "max_position_step": 1., "initial_position": 1.,
                  "cost_rate": .00055, "borrow_apr": .10, "annual_bars": 35040.}
        self.cfg = {"accounting": self.a}

    def initial(self, price=100., batch=1):
        return account.initial_state(batch, "cpu", price, self.cfg)

    def test_initial_is_existing_1x_without_order_or_entry_cost(self):
        state = self.initial(100., 2)
        self.assertEqual(set(state), set(account.STATE_KEYS))
        torch.testing.assert_close(state["cash"], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(state["units"], torch.full((2,), .01, dtype=torch.float64))
        self.assertFalse(state["has_pending"].any())
        self.assertTrue(state["alive"].all())
        self.assertEqual(account.inventory(state).shape, (2, 6))

    def test_one_bar_lag_never_executes_current_request_at_current_open(self):
        state, first = account.step(self.initial(), 2., 100., 100., 110., self.cfg)
        self.assertAlmostEqual(first["equity"].item(), 1.1)
        self.assertEqual(first["fee"].item(), 0.)
        self.assertFalse(first["order_filled"].item())
        self.assertTrue(state["has_pending"].item())
        state, second = account.step(state, 1., 110., 110., 110., self.cfg)
        self.assertTrue(second["order_filled"].item())
        self.assertAlmostEqual(second["open_exposure"].item(), 2., places=12)
        self.assertAlmostEqual(state["requested"].item(), 1.)
        self.assertGreater(second["fee"].item(), 0.)

    def test_missing_decision_preserves_previous_pending_then_stops_new_orders(self):
        state, _ = account.step(self.initial(), 1.5, 100., 100., 100., self.cfg)
        state, second = account.step(state, float("nan"), 100., 100., 100., self.cfg, decision_available=False)
        self.assertTrue(second["order_filled"].item())
        self.assertAlmostEqual(second["open_exposure"].item(), 1.5)
        self.assertFalse(state["has_pending"].item())
        old_units, old_cash = state["units"].clone(), state["cash"].clone()
        state, third = account.step(state, float("nan"), 105., 104., 105., self.cfg, decision_available=False)
        torch.testing.assert_close(state["units"], old_units)
        self.assertLess(state["cash"].item(), old_cash.item())
        self.assertEqual(third["fee"].item(), 0.)
        self.assertEqual(third["turnover"].item(), 0.)
        self.assertGreater(third["borrow"].item(), 0.)

    def test_constant_units_and_constant_exposure_have_different_turnover(self):
        self.a["borrow_apr"] = 0.
        base, _ = account.step(self.initial(), 1.5, 100., 100., 100., self.cfg)
        holding, _ = account.step(base, 1.5, 100., 100., 110., self.cfg, decision_available=False)
        targeting, _ = account.step(base, 1.5, 100., 100., 110., self.cfg)
        self.assertLess(holding["executed"].item(), 1.5)
        holding, h = account.step(holding, 1.5, 110., 110., 110., self.cfg, decision_available=False)
        targeting, t = account.step(targeting, 1.5, 110., 110., 110., self.cfg)
        self.assertEqual(h["fee"].item(), 0.)
        self.assertGreater(t["fee"].item(), 0.)
        self.assertGreater(t["turnover"].item(), 0.)
        self.assertAlmostEqual(t["open_exposure"].item(), 1.5, places=12)

    def test_price_gap_carries_units_cash_and_loss_instead_of_resetting(self):
        state, _ = account.step(self.initial(), 1.5, 100., 100., 100., self.cfg)
        state, _ = account.step(state, 1.5, 100., 100., 100., self.cfg, decision_available=False)
        previous = {key: value.clone() for key, value in state.items()}
        state, d = account.step(state, 1.5, 80., 75., 85., self.cfg, decision_available=False)
        torch.testing.assert_close(state["units"], previous["units"])
        self.assertAlmostEqual(d["equity"].item(), (previous["cash"] - d["borrow"] + previous["units"] * 85).item())
        self.assertLess(d["equity"].item(), previous["prev_equity"].item())
        self.assertGreater(d["close_exposure"].item(), 1.5)
        self.assertGreater(d["drawdown"].item(), 0.)

    def test_intrabar_negative_nav_is_absorbed_even_if_close_recovers(self):
        self.a["borrow_apr"] = 0.
        state, _ = account.step(self.initial(), 2., 100., 100., 100., self.cfg)
        state, d = account.step(state, 2., 100., 40., 120., self.cfg)
        self.assertTrue(d["bankrupt_event"].item())
        self.assertLess(d["nav_low"].item(), 0.)
        self.assertGreater(d["nav_close_before_absorption"].item(), 0.)
        self.assertEqual(d["equity"].item(), 0.)
        self.assertEqual(d["net_simple_return"].item(), -1.)
        self.assertTrue(torch.isneginf(d["net_log_return"]).item())
        self.assertAlmostEqual(d["training_log_return"].item(), math.log(1e-12))
        self.assertEqual(d["max_drawdown"].item(), 1.)
        self.assertTrue(torch.isfinite(account.inventory(state)).all())
        self.assertTrue(torch.isneginf(state["log_wealth"]).all())
        state, later = account.step(state, float("nan"), float("nan"), float("nan"), float("nan"), self.cfg)
        self.assertTrue(later["bankrupt"].item())
        self.assertFalse(later["bankrupt_event"].item())
        for key in ("equity", "fee", "borrow", "net_simple_return", "net_log_return", "training_log_return"):
            self.assertEqual(later[key].item(), 0.)

    def test_open_insolvency_prevents_pending_fill(self):
        self.a["borrow_apr"] = 0.
        state, _ = account.step(self.initial(), 2., 100., 100., 100., self.cfg)
        state, _ = account.step(state, 2., 100., 100., 100., self.cfg)
        state, d = account.step(state, 2., 30., 30., 30., self.cfg)
        self.assertTrue(d["bankrupt_event"].item())
        self.assertFalse(d["order_filled"].item())
        self.assertEqual(d["fee"].item(), 0.)
        self.assertEqual(state["cash"].item(), 0.)
        self.assertEqual(state["units"].item(), 0.)

    def test_request_checks_only_live_available_rows_and_accepts_float32_roundoff(self):
        self.a["max_position_step"] = .08
        state = self.initial(batch=2)
        with self.assertRaisesRegex(ValueError, "target-change rate"):
            account.step(state, [1.2, 1.], 100., 100., 100., self.cfg)
        updated, _ = account.step(state, torch.tensor([1.08, float("nan")], dtype=torch.float32),
            100., 100., 100., self.cfg, decision_available=torch.tensor([True, False]))
        self.assertTrue(updated["has_pending"][0])
        self.assertFalse(updated["has_pending"][1])
        with self.assertRaisesRegex(ValueError, "target-change rate"):
            account.step(state, [1.08, float("nan")], 100., 100., 100., self.cfg)

    def test_positive_finite_ohlc_and_valid_low_required_for_live_accounts(self):
        for opening, low, close in [(float("nan"), 99., 100.), (100., 101., 100.), (100., 0., 100.)]:
            with self.assertRaisesRegex(ValueError, "open/low/close"):
                account.step(self.initial(), 1., opening, low, close, self.cfg)

    def test_no_implicit_gap_reset_or_delay_flag_override(self):
        for key, value in (("execution_delay_bars", 2), ("initial_entry_cost", True),
                           ("terminal_liquidation", True), ("reset_positions_at_segment_boundary", True), ("deadband", .01)):
            cfg = copy.deepcopy(self.cfg)
            cfg["accounting"][key] = value
            with self.assertRaisesRegex(ValueError, "one-bar execution"):
                account.initial_state(1, "cpu", 100., cfg)

    def test_input_state_is_not_mutated_and_bh_units_hold_exactly(self):
        self.a["max_position_step"] = .08
        state = self.initial()
        snapshot = {key: value.clone() for key, value in state.items()}
        updated, d = account.step(state, 1., 100., 95., 101., self.cfg)
        for key in snapshot:
            torch.testing.assert_close(snapshot[key], state[key])
        for opening, low, close in ((105., 99., 102.), (102., 100., 108.), (108., 103., 104.)):
            updated, d = account.step(updated, 1., opening, low, close, self.cfg)
            self.assertEqual(d["fee"].item(), 0.)
            self.assertEqual(d["borrow"].item(), 0.)
            self.assertEqual(d["turnover"].item(), 0.)
            self.assertEqual(updated["units"].item(), .01)
            self.assertAlmostEqual(d["equity"].item(), close / 100.)

    def test_batched_path_agrees_with_independent_scalar_fee_inclusive_formula(self):
        rng = np.random.default_rng(20261112)
        torch_state = self.initial([100., 120., 80.], batch=3)
        scalar = [scalar_initial(price) for price in [100., 120., 80.]]
        previous_close = np.array([100., 120., 80.])
        for _ in range(55):
            opening = previous_close * np.exp(rng.normal(0., .012, 3))
            close = opening * np.exp(rng.normal(0., .018, 3))
            low = np.minimum(opening, close) * (1 - rng.uniform(0., .02, 3))
            requests = np.array([np.clip(s["requested"] + rng.uniform(-.5, .5), 0., 2.) for s in scalar])
            available = rng.uniform(size=3) > .2
            torch_state, diagnostics = account.step(torch_state, requests, opening, low, close, self.cfg,
                decision_available=torch.from_numpy(available))
            for i in range(3):
                scalar[i], expected = scalar_step(scalar[i], requests[i], opening[i], low[i], close[i], self.a, bool(available[i]))
                for key in ("cash", "units", "requested"):
                    self.assertAlmostEqual(torch_state[key][i].item(), scalar[i][key], places=11)
                for key in ("equity", "fee", "borrow", "turnover"):
                    self.assertAlmostEqual(diagnostics[key][i].item(), expected[key], places=11)
                self.assertAlmostEqual(diagnostics["net_simple_return"][i].item(), expected["net_simple"], places=11)
            previous_close = close
        self.assertTrue(all(not value.requires_grad for value in torch_state.values()))


if __name__ == "__main__":
    unittest.main()
