import copy
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import _simulate
from unidream.experiments.oracle_conditional_planner import conditional_targets
from unidream.experiments.oracle_fallback_planner import fallback_targets
from unidream.experiments.oracle_rolling_centering import (
    rolling_centered_forecasts, score_decomposition,
)


class Poison:
    def __float__(self):
        raise AssertionError("unavailable value inspected")


class RollingCenteringTests(unittest.TestCase):
    def fixture(self, decision="2020-04-01T00:00:00Z", end=None):
        t = pd.Timestamp(decision)
        index = pd.date_range(pd.Timestamp("2020-01-01T00:00:00Z"), pd.Timestamp(end) if end else t, freq="15min")
        n = len(index)
        origins = np.asarray((index.hour % 6 == 0) & (index.minute == 0))
        return dict(history_index=index,
            raw_predictions={"technical": np.full(n, 2., dtype=object),
                             "perp_delay0": np.full(n, 4., dtype=object)},
            actual=np.full((n, 3), 1., dtype=object), current_index=pd.DatetimeIndex([t]),
            history_forecast_mask=origins, label_available_mask=np.ones(n, dtype=bool),
            inference_mask=np.array([True]), weights={"technical": .5, "perp_delay0": 1.})

    def test_hand_computed_shared_history_and_audit_trace(self):
        args = self.fixture()
        index = args["history_index"]
        origins = pd.DatetimeIndex(["2020-01-01T00:00Z", "2020-02-01T00:00Z", "2020-03-31T12:00Z"])
        selected = index.isin(origins)
        args["history_forecast_mask"][:] = selected
        args["history_forecast_mask"][-1] = True
        args["actual"][selected, 0] = [1., 3., 5.]
        args["raw_predictions"]["technical"][selected] = [1., 2., 3.]
        args["raw_predictions"]["perp_delay0"][selected] = [4., 2., 0.]
        args["raw_predictions"]["technical"][-1] = 6.
        args["raw_predictions"]["perp_delay0"][-1] = 5.
        result = rolling_centered_forecasts(**args, minimum_pairs=1)
        self.assertEqual(result["means"]["rolling_anchor"][0], 3.)
        self.assertEqual(result["means"]["technical_rolling"][0], 5.)
        self.assertEqual(result["means"]["perp_delay0_rolling"][0], 6.)
        trace = result["trace"][0]
        self.assertEqual(trace["history_count"], 3)
        self.assertEqual(trace["raw_means"], {"technical": 2., "perp_delay0": 2.})
        self.assertEqual(trace["history_timestamp_sha256"], hashlib.sha256(origins.asi8.tobytes()).hexdigest())
        self.assertEqual(trace["latest_maturity"], "2020-03-31T18:15:00+00:00")
        self.assertEqual(trace["window_start"], "2020-01-01T00:00:00+00:00")
        json.dumps(result["trace"], allow_nan=False)

    def test_default_minimum_64_fails_closed_without_extending_window(self):
        args = self.fixture()
        index, t = args["history_index"], args["current_index"][0]
        eligible = np.flatnonzero(args["history_forecast_mask"] & (index <= t - pd.Timedelta(minutes=375)))
        args["label_available_mask"][:] = False
        args["label_available_mask"][eligible[-63:]] = True
        result = rolling_centered_forecasts(**args)
        self.assertFalse(result["available"][0])
        self.assertEqual(result["paired_count"][0], 63)
        self.assertEqual(result["reason_code"][0], "insufficient_history")
        self.assertTrue(all(np.isnan(v[0]) for v in result["means"].values()))
        self.assertIsNone(result["trace"][0]["rolling_anchor"])
        args["label_available_mask"][eligible[-64]] = True
        self.assertTrue(rolling_centered_forecasts(**args)["available"][0])

    def test_calendar_month_lower_boundary_is_inclusive_not_90_days(self):
        args = self.fixture(decision="2020-06-01T00:00:00Z")
        index = args["history_index"]
        args["history_forecast_mask"][:] = False
        for ts in ["2020-02-29T18:00Z", "2020-03-01T00:00Z", "2020-05-31T12:00Z", "2020-06-01T00:00Z"]:
            args["history_forecast_mask"][index.get_loc(pd.Timestamp(ts))] = True
        result = rolling_centered_forecasts(**args, minimum_pairs=1)
        trace = result["trace"][0]
        self.assertEqual(trace["window_start"], "2020-03-01T00:00:00+00:00")
        self.assertEqual(trace["oldest_origin"], "2020-03-01T00:00:00+00:00")
        self.assertEqual(trace["history_count"], 2)
        self.assertNotEqual(pd.Timestamp(trace["window_start"]), args["current_index"][0] - pd.Timedelta(days=90))

    def test_maturity_excludes_prior_six_hour_origin_and_current_label(self):
        args = self.fixture()
        index, t = args["history_index"], args["current_index"][0]
        expected = rolling_centered_forecasts(**args)
        # At the six-hour schedule, tau=t-6h matures at t+15min; t-12h is latest.
        immature = index > t - pd.Timedelta(minutes=375)
        args["actual"][immature, :] = Poison()
        args["actual"][:, 1:] = Poison()
        args["label_available_mask"][immature] = True
        result = rolling_centered_forecasts(**args)
        self.assertEqual(result["trace"], expected["trace"])
        trace = result["trace"][0]
        self.assertEqual(trace["latest_origin"], "2020-03-31T12:00:00+00:00")
        self.assertEqual(trace["maturity_limit_origin"], "2020-03-31T17:45:00+00:00")
        # Equality cannot occur between two UTC six-hour origins plus 375min;
        # the audited limit nevertheless uses <=, not an extra-bar purge.
        self.assertLessEqual(pd.Timestamp(trace["latest_maturity"]), t)

    def test_future_extension_and_immature_poison_do_not_change_prefix(self):
        args = self.fixture(end="2020-04-03T00:00:00Z")
        expected = rolling_centered_forecasts(**args)
        t = args["current_index"][0]
        future = args["history_index"] > t
        args["actual"][future, :] = Poison()
        for raw in args["raw_predictions"].values():
            raw[future] = Poison()
        args["label_available_mask"][future] = False
        self.assertEqual(rolling_centered_forecasts(**args)["trace"], expected["trace"])
        keep = args["history_index"] <= t
        args["history_index"] = args["history_index"][keep]
        args["actual"] = args["actual"][keep]
        args["raw_predictions"] = {g: v[keep] for g, v in args["raw_predictions"].items()}
        for key in ["history_forecast_mask", "label_available_mask"]:
            args[key] = args[key][keep]
        self.assertEqual(rolling_centered_forecasts(**args)["trace"], expected["trace"])

    def test_shared_origin_and_label_support_ignore_unavailable_poison(self):
        args = self.fixture()
        index = args["history_index"]
        dropped_origin = index.get_loc(pd.Timestamp("2020-03-01T00:00Z"))
        dropped_label = index.get_loc(pd.Timestamp("2020-03-02T00:00Z"))
        args["history_forecast_mask"][dropped_origin] = False
        args["label_available_mask"][dropped_label] = False
        expected = rolling_centered_forecasts(**args)
        for j in [dropped_origin, dropped_label]:
            args["actual"][j] = Poison()
            for raw in args["raw_predictions"].values():
                raw[j] = Poison()
        result = rolling_centered_forecasts(**args)
        self.assertEqual(result["trace"], expected["trace"])
        self.assertEqual(result["trace"][0]["mature_label_missing_count"], 1)

    def test_zero_weight_exact_anchor_and_current_inference_ignores_current_label(self):
        args = self.fixture()
        args["weights"] = {"technical": 0., "perp_delay0": 0.}
        args["label_available_mask"][-1] = False
        args["actual"][-1] = Poison()
        result = rolling_centered_forecasts(**args)
        self.assertTrue(result["available"][0])
        for group in ["technical", "perp_delay0"]:
            np.testing.assert_array_equal(result["means"][group + "_rolling"], result["means"]["rolling_anchor"])
        args["history_forecast_mask"][-1] = False
        with self.assertRaisesRegex(ValueError, "current inference"):
            rolling_centered_forecasts(**args)

    def test_unselected_evaluation_rows_remain_nan_and_inputs_unchanged(self):
        args = self.fixture()
        args["current_index"] = args["history_index"][-25:]
        args["inference_mask"] = np.zeros(25, dtype=bool)
        args["inference_mask"][-1] = True
        before = copy.deepcopy(args)
        result = rolling_centered_forecasts(**args)
        for value in result["means"].values():
            self.assertTrue(np.isnan(value[:-1]).all())
        self.assertTrue((result["reason_code"][:-1] == "not_inference").all())
        np.testing.assert_array_equal(args["actual"], before["actual"])
        for group in args["raw_predictions"]:
            np.testing.assert_array_equal(args["raw_predictions"][group], before["raw_predictions"][group])
        for key in ["history_forecast_mask", "label_available_mask", "inference_mask"]:
            np.testing.assert_array_equal(args[key], before[key])

    def test_selected_nonfinite_bool_complex_or_object_is_rejected(self):
        for bad in [float("nan"), float("inf"), True, 1 + 0j, Poison()]:
            for source in ["actual", "raw", "current"]:
                with self.subTest(bad=type(bad).__name__, source=source):
                    args = self.fixture()
                    if source == "actual":
                        args["actual"][0, 0] = bad
                    else:
                        args["raw_predictions"]["technical"][-1 if source == "current" else 0] = bad
                    with self.assertRaises(ValueError):
                        rolling_centered_forecasts(**args)

    def test_weight_shape_grid_and_mask_guards(self):
        for bad in [-.01, 1.01, float("nan"), True, 1 + 0j, [.5]]:
            args = self.fixture()
            args["weights"]["technical"] = bad
            with self.subTest(weight=bad), self.assertRaises(ValueError):
                rolling_centered_forecasts(**args)
        for key in ["history_forecast_mask", "label_available_mask", "inference_mask"]:
            args = self.fixture()
            args[key] = args[key].astype(int)
            with self.subTest(mask=key), self.assertRaises(ValueError):
                rolling_centered_forecasts(**args)
        for bad in [True, 1., 0]:
            with self.subTest(minimum=bad), self.assertRaises(ValueError):
                rolling_centered_forecasts(**self.fixture(), minimum_pairs=bad)
        for key, transform in [("history_index", lambda ix: ix.tz_localize(None)),
                               ("history_index", lambda ix: ix.delete(3)),
                               ("current_index", lambda ix: ix + pd.Timedelta(minutes=1))]:
            args = self.fixture()
            args[key] = transform(args[key])
            with self.subTest(index=key), self.assertRaises(ValueError):
                rolling_centered_forecasts(**args)
        args = self.fixture()
        args["history_forecast_mask"][1] = True
        with self.assertRaisesRegex(ValueError, "six-hour"):
            rolling_centered_forecasts(**args)
        args = self.fixture(decision="2020-03-31T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "full three-calendar-month"):
            rolling_centered_forecasts(**args)

    def test_overflowing_centered_forecast_fails(self):
        args = self.fixture()
        args["raw_predictions"]["technical"][:] = -1.7e308
        args["raw_predictions"]["technical"][-1] = 1.7e308
        with self.assertRaisesRegex(ValueError, "overflowing rolling forecast"):
            rolling_centered_forecasts(**args)

    def test_zero_weight_own_inventory_targets_and_accounts_match_anchor_with_gaps(self):
        args = self.fixture(end="2020-04-03T00:00:00Z")
        history = args["history_index"]
        args["current_index"] = history[history >= pd.Timestamp("2020-04-01T00:00Z")]
        current = args["current_index"]
        inference = np.asarray((current.hour % 6 == 0) & (current.minute == 0))
        inference[[24, 96]] = False
        args["inference_mask"] = inference
        args["weights"] = {"technical": 0., "perp_delay0": 0.}
        args["actual"][:, 0] = np.linspace(.015, .025, len(history))
        # Future, unscored tail labels are immaterial to origin availability.
        args["actual"][-25:] = Poison()
        args["label_available_mask"][-25:] = False
        result = rolling_centered_forecasts(**args)
        np.testing.assert_array_equal(result["available"], inference)
        self.assertTrue(result["available"][-1])
        data = pd.DataFrame({"open": 100., "close": 100., "bar_available": True}, index=current)
        data.loc[current[130]:, ["open", "close"]] = 75.
        data.loc[current[[25, 48]], "open"] = np.nan
        data.loc[current[[1, 25, 96, 97, 120]], "close"] = np.nan
        data["bar_available"] = data[["open", "close"]].notna().all(axis=1)
        # These columns are deliberately unusable as current inference gates.
        data["score_support"], data["future_label_available"] = False, np.nan
        contract = {"one_way_cost": .00055, "borrow_annual": .1, "max_step": .08, "deadband": .01}
        variance = np.where(inference, .0001, np.nan)
        anchor = result["means"]["rolling_anchor"]
        by_rule = {}
        for rule in ("hold", "fallback_bh"):
            def plan(mu):
                if rule == "hold":
                    return conditional_targets(data, mu, variance, contract, risk_aversion=1, cost_multiplier=2)
                return fallback_targets(data, mu, variance, contract, inference_mask=inference,
                                        risk_aversion=1, cost_multiplier=2)
            expected, diagnostic = plan(anchor)
            by_rule[rule] = (expected, diagnostic)
            replay_args = (data.open.to_numpy(), data.close.to_numpy())
            schedule = np.asarray(current.minute == 0)
            expected_account = _simulate(*replay_args, expected, schedule, *contract.values())
            self.assertGreater(diagnostic["metrics"]["trades"], 0)
            self.assertGreater(diagnostic["metrics"]["borrow_initial_equity_units"], 0.)
            self.assertEqual(diagnostic["accounting_max_absolute_difference"], 0.)
            self.assertTrue(np.isnan(expected[48]))  # current open is unavailable
            for group in ("technical_rolling", "perp_delay0_rolling"):
                with self.subTest(rule=rule, group=group):
                    actual, actual_diagnostic = plan(result["means"][group])
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual_diagnostic, diagnostic)
                    account = _simulate(*replay_args, actual, schedule, *contract.values())
                    for observed, reference in zip(account, expected_account):
                        np.testing.assert_array_equal(observed, reference)
        hold_targets, hold = by_rule["hold"]
        fallback_targets_result, fallback = by_rule["fallback_bh"]
        self.assertTrue(np.isnan(hold_targets[[24, 96]]).all())
        np.testing.assert_array_equal(fallback_targets_result[[24, 96]], [1., 1.])
        # The first fallback has no next open; the second fills despite no close.
        self.assertEqual(fallback["fallback_decision_count"], 2)
        self.assertEqual(fallback["missing_open_decision_count"], 1)
        def known_exposure(diagnostic, bar):
            trace = diagnostic["decision_trace"]
            return trace["known_open_exposure"][trace["bar_indices"].index(bar)]
        self.assertEqual(known_exposure(fallback, 72), known_exposure(hold, 72))
        self.assertLess(known_exposure(fallback, 120), known_exposure(hold, 120) - .07)


class VariableAnchorDecompositionTests(unittest.TestCase):
    def test_hand_moments_with_variable_anchor_and_nonzero_drift(self):
        anchor = np.arange(16, dtype=float)
        d, r = np.tile([1., 3.], 8), np.tile([2., 4.], 8)
        actual = np.column_stack([anchor + r, np.zeros(16), np.zeros(16)])
        result = score_decomposition(actual, anchor + d, anchor, np.ones(16, dtype=bool))
        expected = {"n": 16, "candidate_mse": 1., "anchor_mse": 10., "lossdiff": -9.,
            "mean_d": 2., "mean_r": 3., "innovation_secondmoment": 5., "crossmoment": 7.,
            "centered_variance_d": 1., "centered_covariance": 1., "centered_component": -1.,
            "drift_component": -8., "identityresidual": 0.}
        self.assertEqual(result, expected)

    def test_masked_poison_ignored_and_selected_types_checked(self):
        anchor = np.arange(20, dtype=object)
        mu = np.asarray([float(i + 1) for i in range(20)], dtype=object)
        actual = np.full((20, 3), Poison(), dtype=object)
        actual[:16, 0] = np.arange(16) + 2.
        mask = np.arange(20) < 16
        expected = score_decomposition(actual, mu, anchor, mask)
        mu[~mask], anchor[~mask] = Poison(), Poison()
        self.assertEqual(score_decomposition(actual, mu, anchor, mask), expected)
        for bad in [True, 1 + 0j, np.nan, np.inf]:
            broken = mu.copy()
            broken[0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_decomposition(actual, broken, anchor, mask)
        with self.assertRaises(ValueError):
            score_decomposition(actual, mu, anchor, mask.astype(int))
        with self.assertRaisesRegex(ValueError, "16 scored"):
            score_decomposition(actual, mu, anchor, np.arange(20) < 15)

    def test_zero_innovation_and_extreme_finite_moment_overflow(self):
        anchor = np.arange(16, dtype=float)
        actual = np.column_stack([anchor + 2., np.zeros(16), np.zeros(16)])
        result = score_decomposition(actual, anchor.copy(), anchor, np.ones(16, dtype=bool))
        self.assertEqual(result["lossdiff"], 0.)
        self.assertEqual(result["identityresidual"], 0.)
        self.assertEqual(result["innovation_secondmoment"], 0.)
        with self.assertRaisesRegex(ValueError, "overflow"):
            score_decomposition(actual, np.full(16, 1e200), anchor, np.ones(16, dtype=bool))


if __name__ == "__main__":
    unittest.main()
