import inspect
import json
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import _simulate, metrics
from unidream.experiments.oracle_conditional_planner import conditional_targets
from unidream.experiments.oracle_fallback_planner import fallback_targets


CONTRACT = {"one_way_cost": .00055, "borrow_annual": .1, "max_step": .08, "deadband": .01}


def bars(n=145):
    return pd.DataFrame({"open": 100., "close": 100., "bar_available": True},
                        index=pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC"))


def scheduled(data):
    index = data.index.tz_convert("UTC")
    return np.asarray((index.hour % 6 == 0) & (index.minute == 0))


def trace_at(diagnostic, bar):
    trace = diagnostic["decision_trace"]
    i = trace["bar_indices"].index(bar)
    return {key: values[i] for key, values in trace.items()}


class OracleFallbackPlannerTests(unittest.TestCase):
    def test_all_valid_matches_frozen_parent_targets_and_accounting(self):
        data = bars()
        data.loc[data.index[37]:, ["open", "close"]] = 97.
        mu, variance = np.full(len(data), .02), np.full(len(data), .0001)
        mu[48:96] = -.02
        expected, parent = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=1)
        actual, diagnostic = fallback_targets(data, mu, variance, CONTRACT,
                                               inference_mask=scheduled(data))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(diagnostic["metrics"], parent["metrics"])
        for key, values in parent["decision_trace"].items():
            self.assertEqual(diagnostic["decision_trace"][key], values)
        self.assertEqual(diagnostic["fallback_decision_count"], 0)
        self.assertEqual(diagnostic["learned_decision_count"], 7)
        self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)
        json.dumps(diagnostic, allow_nan=False)

    def test_valid_zero_utility_preserves_nan_hold_and_trace(self):
        data = bars(49)
        targets, diagnostic = fallback_targets(data, np.zeros(49), np.zeros(49), CONTRACT,
                                               inference_mask=scheduled(data), risk_aversion=0)
        self.assertTrue(np.isnan(targets).all())
        self.assertEqual(diagnostic["hold_decision_count"], 3)
        self.assertEqual(diagnostic["fallback_decision_count"], 0)
        self.assertEqual(diagnostic["positive_intent_count"], 0)
        self.assertEqual(diagnostic["decision_trace"]["targets"], [None] * 3)
        self.assertEqual(diagnostic["decision_trace"]["estimated_utility_gain_over_hold"], [0.] * 3)
        self.assertEqual(diagnostic["decision_trace"]["reasons"], ["learned"] * 3)
        self.assertEqual(diagnostic["decision_masks"]["learned"], scheduled(data).tolist())
        self.assertEqual(diagnostic["decision_masks"]["hold"], scheduled(data).tolist())

    def test_all_unavailable_has_bh_economics_with_explicit_noop_intents(self):
        data = bars()
        data[["open", "close"]] = (100 + 10 * np.sin(np.arange(len(data)) / 19))[:, None].repeat(2, axis=1)
        data.loc[data.index[30:34], ["open", "close"]] = np.nan
        data.loc[data.index[30:34], "bar_available"] = False
        targets, diagnostic = fallback_targets(data, np.full(len(data), np.nan), np.full(len(data), np.nan),
                                               CONTRACT, inference_mask=np.zeros(len(data), bool))
        np.testing.assert_array_equal(targets[scheduled(data)], np.ones(7))
        self.assertTrue(np.isnan(targets[~scheduled(data)]).all())
        benchmark = metrics(data, np.full(len(data), np.nan), CONTRACT)
        for key, value in benchmark.items():
            if key != "intent_coverage":
                self.assertEqual(diagnostic["metrics"][key], value, key)
        self.assertEqual(diagnostic["fallback_decision_count"], 7)
        self.assertEqual(diagnostic["learned_decision_count"], 0)
        self.assertEqual(diagnostic["hold_decision_count"], 0)
        self.assertEqual(diagnostic["metrics"]["trades"], 0)
        self.assertGreater(diagnostic["metrics"]["intent_coverage"], 0)
        trace = diagnostic["decision_trace"]
        self.assertEqual(trace["reasons"], ["forecast_unavailable"] * 7)
        self.assertEqual(trace["estimated_utility_gain_over_hold"], [None] * 7)
        self.assertEqual(trace["estimated_trade_turnover"], [None] * 7)
        json.dumps(diagnostic, allow_nan=False)

    def test_fallback_changes_own_inventory_before_later_learned_choice(self):
        data = bars(73)
        infer = scheduled(data)
        infer[24] = False
        mu, variance = np.full(len(data), .02), np.zeros(len(data))
        mu[24] = np.nan
        targets, diagnostic = fallback_targets(data, mu, variance, CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        parent_targets, parent = conditional_targets(data, mu, variance, CONTRACT, risk_aversion=0)
        self.assertAlmostEqual(targets[0], 1.08)
        self.assertEqual(targets[24], 1.)
        self.assertTrue(np.isnan(parent_targets[24]))
        own = trace_at(diagnostic, 48)["known_open_exposure"]
        parent_current = trace_at(parent, 48)["known_open_exposure"]
        self.assertLess(own, parent_current - .07)
        self.assertAlmostEqual(targets[48], own + .08)
        self.assertLess(targets[48], parent_targets[48])
        self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)

    def test_outside_bound_passive_drift_is_not_clipped_on_hold_or_fallback_fill(self):
        data = bars(97)
        data.loc[data.index[30]:, ["open", "close"]] = 50.
        infer = scheduled(data)
        infer[72] = False
        targets, diagnostic = fallback_targets(data, np.full(97, .1), np.zeros(97), CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        self.assertAlmostEqual(targets[24], 1.12)
        self.assertGreater(trace_at(diagnostic, 48)["known_open_exposure"], 1.2)
        self.assertTrue(np.isnan(targets[48]))
        self.assertEqual(targets[72], 1.)
        _, exposure, *_ = _simulate(data.open.to_numpy(), data.close.to_numpy(), targets,
                                   np.asarray(data.index.minute == 0), *CONTRACT.values())
        self.assertGreater(exposure[73], 1.12)
        self.assertAlmostEqual(exposure[73], exposure[72] - CONTRACT["max_step"], places=5)
        self.assertGreater(trace_at(diagnostic, 96)["known_open_exposure"], 1.12)
        self.assertTrue(np.isnan(targets[96]))
        self.assertGreater(diagnostic["metrics"]["borrow_initial_equity_units"], 0.)
        self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)

    def test_missing_next_open_keeps_current_intent_but_does_not_roll_it_forward(self):
        data = bars(73)
        infer = scheduled(data)
        infer[24] = False
        mu, variance = np.zeros(73), np.zeros(73)
        mu[0] = .02
        clean_targets, clean = fallback_targets(data, mu, variance, CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        data.loc[data.index[25], ["open", "close"]] = np.nan
        data.loc[data.index[25], "bar_available"] = False
        targets, diagnostic = fallback_targets(data, mu, variance, CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        self.assertEqual(targets[24], clean_targets[24])
        self.assertEqual(targets[24], 1.)
        self.assertEqual(diagnostic["metrics"]["trades"], 1)
        self.assertEqual(clean["metrics"]["trades"], 2)
        self.assertGreater(trace_at(diagnostic, 48)["known_open_exposure"],
                           trace_at(clean, 48)["known_open_exposure"] + .07)
        self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)

    def test_missing_close_does_not_cancel_known_open_decision_or_fill(self):
        data = bars(73)
        infer = scheduled(data)
        infer[24] = False
        mu, variance = np.full(73, .02), np.zeros(73)
        clean_targets, clean = fallback_targets(data, mu, variance, CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        data.loc[data.index[[1, 24, 25, 48]], "close"] = np.nan
        data.loc[data.index[[1, 24, 25, 48]], "bar_available"] = False
        targets, diagnostic = fallback_targets(data, mu, variance, CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        np.testing.assert_array_equal(targets, clean_targets)
        self.assertEqual(diagnostic["decision_trace"], clean["decision_trace"])
        for key in ("trades", "turnover", "fees_initial_equity_units", "borrow_initial_equity_units"):
            self.assertEqual(diagnostic["metrics"][key], clean["metrics"][key])
        self.assertEqual(targets[24], 1.)
        self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)

    def test_missing_current_open_is_separate_from_unavailable_forecast(self):
        data = bars(97)
        infer = scheduled(data)
        infer[24], infer[72] = False, False
        data.loc[data.index[[24, 48]], "open"] = np.nan
        data.loc[data.index[[24, 48]], "bar_available"] = False
        targets, diagnostic = fallback_targets(data, np.zeros(97), np.zeros(97), CONTRACT,
                                               inference_mask=infer, risk_aversion=0)
        self.assertTrue(np.isnan(targets[[24, 48]]).all())
        self.assertEqual(targets[72], 1.)
        self.assertEqual(diagnostic["missing_open_decision_count"], 2)
        self.assertEqual(diagnostic["fallback_decision_count"], 1)
        self.assertEqual(diagnostic["learned_decision_count"], 2)
        self.assertEqual(diagnostic["hold_decision_count"], 2)
        self.assertEqual(diagnostic["decision_trace"]["bar_indices"], [0, 72, 96])
        masks = diagnostic["decision_masks"]
        np.testing.assert_array_equal(np.asarray(masks["learned"], int) + masks["fallback"] + masks["missing_open"],
                                      scheduled(data).astype(int))

    def test_all_missing_opens_are_rejected_but_interior_open_gaps_preserve_bh(self):
        data = bars(97)
        data["open"] = np.nan
        with self.assertRaisesRegex(ValueError, "boundary"):
            fallback_targets(data, np.full(97, np.nan), np.full(97, np.nan), CONTRACT,
                             inference_mask=np.zeros(97, bool))
        data.loc[data.index[[0, -1]], "open"] = 100.
        data.loc[data.index[1:-1], "bar_available"] = False
        targets, diagnostic = fallback_targets(data, np.full(97, np.nan), np.full(97, np.nan), CONTRACT,
                                               inference_mask=np.zeros(97, bool))
        self.assertEqual(diagnostic["missing_open_decision_count"], 3)
        self.assertEqual(diagnostic["fallback_decision_count"], 2)
        self.assertEqual(diagnostic["metrics"]["trades"], 0)
        self.assertEqual(diagnostic["metrics"]["alpha_ex"], 0.)
        self.assertTrue(np.isnan(targets[24:73]).all())

    def test_current_close_future_prices_and_future_availability_cannot_change_current_targets(self):
        data = bars()
        infer = scheduled(data)
        infer[48] = False
        mu, variance = np.full(145, .02), np.full(145, .0001)
        targets, _ = fallback_targets(data, mu, variance, CONTRACT, inference_mask=infer)
        changed = data.copy()
        changed.loc[changed.index[48], "close"] = np.nan
        changed.loc[changed.index[48], "bar_available"] = False
        changed.loc[changed.index[49]:, ["open", "close"]] *= 1.5
        new_mu, new_variance, new_infer = mu.copy(), variance.copy(), infer.copy()
        new_mu[49:], new_variance[49:], new_infer[49:] = -.1, .01, False
        after, _ = fallback_targets(changed, new_mu, new_variance, CONTRACT, inference_mask=new_infer)
        np.testing.assert_array_equal(after[:49], targets[:49])
        truncated, _ = fallback_targets(data.iloc[:49], mu[:49], variance[:49], CONTRACT,
                                        inference_mask=infer[:49])
        np.testing.assert_array_equal(truncated, targets[:49])

    def test_score_support_is_not_an_input_and_cannot_cancel_inference(self):
        data = bars(49)
        infer = scheduled(data)
        mu, variance = np.full(49, .02), np.zeros(49)
        targets, diagnostic = fallback_targets(data, mu, variance, CONTRACT, inference_mask=infer)
        data["score_support"], data["future_label_available"] = False, np.nan
        other, other_diagnostic = fallback_targets(data, mu, variance, CONTRACT, inference_mask=infer)
        np.testing.assert_array_equal(other, targets)
        self.assertEqual(other_diagnostic, diagnostic)
        self.assertNotIn("score_mask", inspect.signature(fallback_targets).parameters)
        with self.assertRaises(TypeError):
            fallback_targets(data, mu, variance, CONTRACT, inference_mask=infer, score_mask=infer)

    def test_timezone_representation_does_not_change_utc_schedule(self):
        data = bars(49)
        infer = scheduled(data)
        infer[24] = False
        expected, diagnostic = fallback_targets(data, np.full(49, .02), np.zeros(49), CONTRACT,
                                                inference_mask=infer)
        data.index = data.index.tz_convert("Asia/Tokyo")
        actual, converted = fallback_targets(data, np.full(49, .02), np.zeros(49), CONTRACT,
                                            inference_mask=infer)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(converted, diagnostic)

    def test_forecasts_only_claimed_valid_rows_are_required_to_be_finite(self):
        data = bars(49)
        infer = scheduled(data)
        infer[24] = False
        mu, variance = np.zeros(49), np.zeros(49)
        mu[~infer], variance[~infer] = np.inf, -np.inf
        targets, _ = fallback_targets(data, mu, variance, CONTRACT, inference_mask=infer)
        self.assertEqual(targets[24], 1.)
        for which, value in (("mu", np.nan), ("mu", np.inf), ("variance", np.nan),
                             ("variance", np.inf), ("variance", -.1)):
            bad_mu, bad_variance = mu.copy(), variance.copy()
            (bad_mu if which == "mu" else bad_variance)[0] = value
            with self.subTest(which=which, value=value), self.assertRaisesRegex(ValueError, "claimed-valid"):
                fallback_targets(data, bad_mu, bad_variance, CONTRACT, inference_mask=infer)
        data.loc[data.index[0], "close"] = np.nan  # Even an absent mark cannot excuse invalid inference.
        with self.assertRaisesRegex(ValueError, "claimed-valid"):
            fallback_targets(data, np.full(49, np.nan), variance, CONTRACT, inference_mask=infer)

    def test_malformed_masks_forecast_shapes_and_execution_contracts_fail_closed(self):
        data = bars(49)
        infer = scheduled(data)
        offschedule = infer.copy()
        offschedule[1] = True
        for bad in (infer.astype(float), infer[:-1], infer[:, None], offschedule):
            with self.subTest(mask_shape=bad.shape), self.assertRaises(ValueError):
                fallback_targets(data, np.zeros(49), np.zeros(49), CONTRACT, inference_mask=bad)
        for mu, variance in ((np.zeros((49, 1)), np.zeros(49)), (np.zeros(49), np.zeros(48))):
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                fallback_targets(data, mu, variance, CONTRACT, inference_mask=infer)
        contracts = [{}, None] + [dict(CONTRACT, **{key: value}) for key, value in
                                  (("one_way_cost", -.1), ("one_way_cost", 1), ("borrow_annual", -.1),
                                   ("max_step", 0), ("deadband", 0), ("one_way_cost", np.nan))]
        for contract in contracts:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                fallback_targets(data, np.zeros(49), np.zeros(49), contract, inference_mask=infer)
        for name, value in (("risk_aversion", -.1), ("risk_aversion", np.nan),
                            ("cost_multiplier", np.inf), ("cost_multiplier", [2.])):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                fallback_targets(data, np.zeros(49), np.zeros(49), CONTRACT,
                                 inference_mask=infer, **{name: value})

    def test_malformed_grid_prices_and_availability_are_rejected(self):
        original = bars(49)
        invalid = [original.iloc[0:0], original.drop(original.index[3]), original.iloc[::-1],
                   original.drop(columns="close")]
        naive = original.copy()
        naive.index = naive.index.tz_localize(None)
        invalid.append(naive)
        offgrid = original.copy()
        offgrid.index += pd.Timedelta(minutes=1)
        invalid.append(offgrid)
        for column, row, value in (("open", 0, 0.), ("close", 24, np.inf),
                                   ("open", 0, np.nan), ("close", 48, np.nan),
                                   ("bar_available", 0, False)):
            data = original.copy()
            data.loc[data.index[row], column] = value
            invalid.append(data)
        nonbool = original.copy()
        nonbool["bar_available"] = 1
        invalid.append(nonbool)
        for data in invalid:
            with self.subTest(rows=len(data), columns=data.columns.tolist()), self.assertRaises(ValueError):
                fallback_targets(data, np.zeros(len(data)), np.zeros(len(data)), CONTRACT,
                                 inference_mask=np.zeros(len(data), bool))


if __name__ == "__main__":
    unittest.main()
