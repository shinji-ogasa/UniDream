"""Synthetic action-set and independent cash-account checks, without fits."""
import json
import math
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from unidream.experiments.alpha_dd_search import BARS_YEAR, metrics
from unidream.experiments.oracle_matched_hindsight import _dynamic_intents, matched_hindsight_targets


CONTRACT = {"one_way_cost": .00055, "borrow_annual": .1, "max_step": .08, "deadband": .01}


def bars(prices):
    close = np.asarray(prices, float)
    return pd.DataFrame({"open": np.r_[100., close[:-1]], "close": close, "bar_available": True},
                        index=pd.date_range("2001-01-01", periods=len(close), freq="15min", tz="UTC"))


def support(data):
    utc = data.index.tz_convert("UTC")
    return np.asarray((utc.hour % 6 == 0) & (utc.minute == 0))


def exhaustive_paths(data, allowed, contract, missing_input_rule="hold"):
    """Enumerate dynamic choices using an independent scalar cash/units loop."""
    paths = [(0., 1. / float(data.open.iloc[0]), np.full(len(data), np.nan))]
    for i, (opening, closing) in enumerate(zip(data.open, data.close)):
        following = []
        for cash, units, targets in paths:
            if np.isfinite(opening):
                nav = cash + units * opening
                if nav <= 0: continue
                if i > 0 and np.isfinite(targets[i - 1]):
                    exposure = units * opening / nav
                    desired = float(np.clip(targets[i - 1], 0., 1.12))
                    change = float(np.clip(desired - exposure, -contract["max_step"], contract["max_step"]))
                    if abs(change) >= contract["deadband"]:
                        projected = exposure + change
                        trade = (projected * nav - units * opening) / (1 + contract["one_way_cost"] * projected * (1 if change > 0 else -1))
                        cash -= trade + contract["one_way_cost"] * abs(trade)
                        units += trade / opening
            choices = [np.nan]
            if allowed[i] and np.isfinite(opening):
                current = units * opening / (cash + units * opening)
                for delta in (-.08, -.04, .04, .08):
                    intent = float(np.clip(current + delta, .5, 1.12))
                    projected_change = float(np.clip(intent - current, -contract["max_step"], contract["max_step"]))
                    if projected_change != 0 and abs(projected_change) >= contract["deadband"]:
                        choices.append(intent)
            elif missing_input_rule == "fallback_bh" and support(data)[i] and np.isfinite(opening):
                choices = [1.]
            for choice in choices:
                candidate = targets.copy(); candidate[i] = choice
                next_cash = cash
                if next_cash < 0:
                    next_cash -= -next_cash * (math.exp(contract["borrow_annual"] / BARS_YEAR) - 1)
                if np.isfinite(closing) and next_cash + units * closing <= 0: continue
                following.append((next_cash, units, candidate))
        paths = following
    return [path[2] for path in paths]


def scalar_known_open_trace(data, targets):
    cash, units = 0., 1. / float(data.open.iloc[0])
    known = {}
    for i, (opening, closing) in enumerate(zip(data.open, data.close)):
        if np.isfinite(opening):
            nav = cash + units * opening
            if i > 0 and np.isfinite(targets[i - 1]):
                position = units * opening / nav
                change = float(np.clip(np.clip(targets[i - 1], 0., 1.12) - position, -.08, .08))
                if abs(change) >= .01:
                    desired = position + change
                    value = (desired * nav - units * opening) / (1 + .00055 * desired * (1 if change > 0 else -1))
                    cash -= value + .00055 * abs(value)
                    units += value / opening
            known[i] = (cash + units * opening, units * opening / (cash + units * opening))
        if cash < 0: cash -= -cash * (math.exp(.1 / BARS_YEAR) - 1)
    return known


class MatchedHindsightTests(unittest.TestCase):
    def test_large_beam_matches_independent_exhaustive_dynamic_paths(self):
        data = bars(100 * np.exp(.08 * np.sin(np.arange(50) / 7)))
        allowed = support(data)
        paths = exhaustive_paths(data, allowed, CONTRACT)
        # Dynamic clipping/deadband remove six otherwise nominal 5^3 paths.
        self.assertEqual(len(paths), 119)
        for penalty in (0., 1.):
            expected = max(math.log1p(m["total_return"]) - penalty * m["maxdd"]
                           for m in (metrics(data, path, CONTRACT) for path in paths))
            target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=allowed,
                                                     beam_width=256, risk_penalty=penalty)
            self.assertAlmostEqual(diag["objective"], expected, places=13)
            self.assertTrue(diag["exhaustive_for_matched_dynamic_action_set"])
            self.assertTrue(diag["canonical_replay_verified"])
            self.assertEqual(diag["metrics"], metrics(data, target, CONTRACT))

    def test_dynamic_intents_match_passive_inventory_and_deadband_projection(self):
        assert_array_equal(_dynamic_intents(1., .08, .01), [np.nan, .92, .96, 1.04, 1.08])
        assert_array_equal(_dynamic_intents(.3, .08, .01), [np.nan, .5])
        assert_array_equal(_dynamic_intents(1.4, .08, .01), [np.nan, 1.12])
        self.assertTrue(np.isnan(_dynamic_intents(1., .005, .01)).all())

    def test_current_close_cannot_define_candidate_values_but_future_rank_is_explicit(self):
        up = bars(100 * np.exp(np.arange(24) * .015))
        down = bars(100 * np.exp(-np.arange(24) * .015))
        first, diag = matched_hindsight_targets(up, CONTRACT, decision_support=support(up))
        second, _ = matched_hindsight_targets(down, CONTRACT, decision_support=support(down))
        self.assertEqual(first[0], 1.08)
        self.assertEqual(second[0], .92)
        self.assertTrue(diag["future_information_used"])
        self.assertFalse(diag["deployable"])
        self.assertFalse(diag["teacher_use_allowed"])
        self.assertFalse(diag["global_optimum_claimed"])
        self.assertFalse(diag["support_causality_verified"])

    def test_trace_uses_endogenous_open_inventory_before_current_borrow_or_close(self):
        data = bars(100 * np.exp(np.arange(75) * .004))
        data.loc[data.index[24], "close"] = 70.
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data))
        known = scalar_known_open_trace(data, target)
        for i, nav, exposure in zip(diag["decision_trace"]["bar_indices"],
                                    diag["decision_trace"]["known_open_nav"], diag["decision_trace"]["known_open_exposure"]):
            self.assertAlmostEqual(nav, known[i][0], places=14)
            self.assertAlmostEqual(exposure, known[i][1], places=14)
            if np.isfinite(target[i]): self.assertIn(target[i], _dynamic_intents(exposure, .08, .01))
        self.assertLess(diag["accounting_max_absolute_difference"], 1e-12)

    def test_unsupported_slots_force_hold_and_do_not_reset_inventory(self):
        data = bars(100 * np.exp(np.arange(75) * .005))
        allowed = support(data); allowed[24:] = False
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=allowed)
        self.assertEqual(target[0], 1.08)
        self.assertTrue(np.isnan(target[24:]).all())
        self.assertEqual(diag["metrics"]["trades"], 1)
        self.assertEqual(diag["unsupported_decision_count"], 3)
        self.assertEqual(diag["decision_trace"]["bar_indices"], [0])

    def test_missing_current_open_skips_decision_but_missing_close_does_not(self):
        data = bars(100 * np.exp(-np.arange(50) * .008))
        data.loc[data.index[24], ["open", "bar_available"]] = [np.nan, False]
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data))
        self.assertTrue(np.isnan(target[24]))
        self.assertEqual(diag["missing_open_decision_count"], 1)
        data = bars(100 * np.exp(-np.arange(24) * .01))
        data.loc[data.index[1], ["close", "bar_available"]] = [np.nan, False]
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data))
        self.assertEqual(target[0], .92)
        self.assertEqual(diag["metrics"]["trades"], 1)

    def test_missing_next_open_cancels_fill_without_rollover(self):
        data = bars(100 * np.exp(-np.arange(24) * .01))
        data.loc[data.index[1], ["open", "close", "bar_available"]] = [np.nan, np.nan, False]
        _, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data))
        self.assertEqual(diag["metrics"]["trades"], 0)
        self.assertEqual(diag["metrics"]["turnover"], 0)
        self.assertEqual(diag["metrics"]["alpha_ex"], 0)

    def test_gap_borrow_and_next_open_step_use_canonical_account(self):
        data = bars(100 * np.exp(np.arange(75) * .009))
        data.loc[data.index[2:7], ["open", "close"]] = np.nan
        data.loc[data.index[2:7], "bar_available"] = False
        data.loc[data.index[25], "open"] *= 1.2
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data), risk_penalty=1)
        self.assertGreater(diag["metrics"]["borrow_initial_equity_units"], 0)
        self.assertGreater(diag["metrics"]["fees_initial_equity_units"], 0)
        self.assertEqual(diag["metrics"], metrics(data, target, CONTRACT))
        self.assertLess(diag["accounting_max_absolute_difference"], 1e-12)

    def test_all_hold_envelope_and_empty_support_never_force_a_trade(self):
        data = bars(100 * np.exp(.2 * np.sin(np.arange(150) / 15)))
        _, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data), beam_width=1, risk_penalty=1)
        hold = metrics(data, np.full(len(data), np.nan), CONTRACT)
        self.assertGreater(diag["pruned_distinct_branches"], 0)
        self.assertGreaterEqual(diag["objective"], math.log1p(hold["total_return"]) - hold["maxdd"])
        target, empty = matched_hindsight_targets(data, CONTRACT, decision_support=np.zeros(len(data), bool))
        self.assertTrue(np.isnan(target).all())
        self.assertEqual(empty["metrics"], hold)
        self.assertEqual(empty["decision_count"], 0)
        json.dumps(empty, allow_nan=False)

    def test_fallback_large_beam_matches_independent_exhaustive_rule_paths(self):
        data = bars(100 * np.exp(.08 * np.sin(np.arange(75) / 7)))
        allowed = support(data); allowed[[24, 72]] = False
        paths = exhaustive_paths(data, allowed, CONTRACT, "fallback_bh")
        for penalty in (0., 1.):
            expected = max(math.log1p(m["total_return"]) - penalty * m["maxdd"]
                           for m in (metrics(data, path, CONTRACT) for path in paths))
            target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=allowed,
                beam_width=256, risk_penalty=penalty, missing_input_rule="fallback_bh")
            self.assertAlmostEqual(diag["objective"], expected, places=13)
            assert_array_equal(target[[24, 72]], [1., 1.])
            self.assertEqual(diag["forced_fallback_decision_count"], 2)
            self.assertTrue(diag["exhaustive_for_matched_dynamic_action_set"])

    def test_fallback_state_changes_later_free_inventory_and_trace_reasons(self):
        data = bars(100 * np.exp(np.arange(75) * .006))
        allowed = support(data); allowed[24] = False
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=allowed, missing_input_rule="fallback_bh")
        self.assertEqual(target[0], 1.08)
        self.assertEqual(target[24], 1.)
        self.assertAlmostEqual(target[48], 1.08)
        known = scalar_known_open_trace(data, target)
        for i, nav, exposure, reason in zip(diag["decision_trace"]["bar_indices"],
                diag["decision_trace"]["known_open_nav"], diag["decision_trace"]["known_open_exposure"],
                diag["decision_trace"]["reasons"]):
            self.assertAlmostEqual(nav, known[i][0], places=14)
            self.assertAlmostEqual(exposure, known[i][1], places=14)
            self.assertEqual(reason, "forced_fallback_bh" if i == 24 else "free_dynamic_action")
        _, hold = matched_hindsight_targets(data, CONTRACT, decision_support=allowed)
        j = hold["decision_trace"]["bar_indices"].index(48)
        self.assertNotAlmostEqual(hold["decision_trace"]["known_open_exposure"][j], known[48][1])

    def test_fallback_incumbent_obeys_forced_intents_even_terminal_and_no_fill(self):
        data = bars(np.full(49, 100.)); allowed = np.zeros(len(data), bool)
        data.loc[data.index[24], ["open", "bar_available"]] = [np.nan, False]
        data.loc[data.index[1], ["open", "close", "bar_available"]] = [np.nan, np.nan, False]
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=allowed,
            beam_width=1, missing_input_rule="fallback_bh")
        assert_array_equal(target[[0, 48]], [1., 1.])
        self.assertTrue(np.isnan(target[24]))
        self.assertEqual(diag["search_event_count"], 3)
        self.assertEqual(diag["missing_open_decision_count"], 1)
        self.assertTrue(diag["incumbent_selected"])
        self.assertFalse(diag["all_hold_envelope_selected"])
        self.assertEqual(diag["incumbent_rule"], "hold_on_supported_fallback_bh_on_unsupported")
        self.assertEqual(diag["objective"], diag["incumbent_objective"])
        self.assertEqual(diag["metrics"]["trades"], 0)
        json.dumps(diag, allow_nan=False)

    def test_fallback_narrow_beam_preserves_a_rule_feasible_incumbent(self):
        data = bars(100 * np.exp(.2 * np.sin(np.arange(150) / 15)))
        allowed = support(data); allowed[[24, 72, 120]] = False
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=allowed,
            beam_width=1, risk_penalty=1, missing_input_rule="fallback_bh")
        self.assertGreater(diag["pruned_distinct_branches"], 0)
        self.assertGreaterEqual(diag["objective"], diag["incumbent_objective"])
        assert_array_equal(target[[24, 72, 120]], [1., 1., 1.])
        self.assertEqual(diag["metrics"], metrics(data, target, CONTRACT))

    def test_single_terminal_slot_has_no_fill_and_timezone_is_only_representation(self):
        data = bars([101.])
        target, diag = matched_hindsight_targets(data, CONTRACT, decision_support=support(data))
        self.assertTrue(np.isnan(target).all())
        self.assertEqual(diag["metrics"]["trades"], 0)
        longer = bars(100 * np.exp(np.arange(50) * .005))
        first, diag = matched_hindsight_targets(longer, CONTRACT, decision_support=support(longer))
        second, other = matched_hindsight_targets(longer.tz_convert("Asia/Tokyo"), CONTRACT, decision_support=support(longer))
        assert_array_equal(first, second)
        self.assertEqual(diag, other)

    def test_inputs_preserved_and_future_scoring_mask_is_not_an_argument(self):
        data = bars(100 * np.exp(np.arange(26) * .005))
        allowed, original = support(data), data.copy(deep=True)
        allowed.setflags(write=False)
        before = allowed.copy()
        matched_hindsight_targets(data, CONTRACT, decision_support=allowed)
        pd.testing.assert_frame_equal(data, original)
        assert_array_equal(allowed, before)
        with self.assertRaises(TypeError):
            matched_hindsight_targets(data, CONTRACT, decision_support=allowed, score_mask=allowed)

    def test_invalid_support_contract_calendar_and_boundaries_rejected(self):
        data = bars(np.full(30, 100.)); allowed = support(data)
        off_clock = allowed.copy(); off_clock[1] = True
        for mask in (allowed.astype(int), allowed[:-1], np.ones((30, 1), bool), off_clock):
            with self.assertRaises(ValueError): matched_hindsight_targets(data, CONTRACT, decision_support=mask)
        for width in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError): matched_hindsight_targets(data, CONTRACT, decision_support=allowed, beam_width=width)
        for penalty in (True, -1., np.nan, np.inf):
            with self.assertRaises(ValueError): matched_hindsight_targets(data, CONTRACT, decision_support=allowed, risk_penalty=penalty)
        with self.assertRaises(ValueError):
            matched_hindsight_targets(data, CONTRACT, decision_support=allowed, missing_input_rule="zero")
        for change in ({"one_way_cost": 1.}, {"borrow_annual": -1.}, {"max_step": 0.}, {"deadband": 0.}, {"one_way_cost": True}):
            with self.assertRaises(ValueError): matched_hindsight_targets(data, {**CONTRACT, **change}, decision_support=allowed)
        for bad in (data.drop(data.index[3]), data.tz_localize(None)):
            with self.assertRaises(ValueError): matched_hindsight_targets(bad, CONTRACT, decision_support=support(data))
        data.iloc[0, data.columns.get_loc("open")] = np.nan
        with self.assertRaises(ValueError): matched_hindsight_targets(data, CONTRACT, decision_support=allowed)


if __name__ == "__main__":
    unittest.main()
