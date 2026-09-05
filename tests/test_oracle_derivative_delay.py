import copy
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_delay import (
    FOLDS, FROZEN, GROUPS, PAIRS, POLICIES, VERSIONS,
    paired_results, segment_masks, validate_config,
)


def registered_config():
    return {"schema": "oracle-derivative-delay-v1", "development_folds": list(range(5, 13)),
        "extra_delay_bars": [0, 1, 4], "horizon_bars": 24,
        "fit_months": 18, "scale_months": 3, "interval_months": 3,
        "versions": ["raw", "scaled"], "utility_risk_aversions": [1],
        "utility_cost_multiplier": 2, "return_ridge_alpha": 100.0}


def paired_fixture():
    """Eight complete quarters, with pairs of 1/100 rows in every regime."""
    values = {"technical": (2., 8.), "perp_delay0": (4., 2.),
              "perp_delay1": (5., 3.), "perp_delay4": (6., 4.),
              "frozen_delay1": (7., 5.), "frozen_delay4": (8., 6.)}
    forecast_factors = {"return_mse": 1., "qlike": 2., "variance_mse": 3., "rms_mse": 4.}
    economic_factors = {"alpha_ex": 1., "maxdd_delta": -2., "turnover": 3., "trades": 4.}
    scores, rows = [], []
    for i, fold in enumerate(FOLDS):
        trend = "bull" if i < 2 else "bear" if i < 6 else "sideways"
        regime = {"trend": trend}
        count = 1 if i % 2 == 0 else 100
        for group in GROUPS + FROZEN:
            for version in VERSIONS:
                version_factor = 1 if version == "raw" else 3
                value = values[group][i % 2] * version_factor
                mid = group + "_" + version
                scores.append({"fold": fold, "model_id": mid, "regime": regime.copy(), "rows": count,
                    **{metric: value * factor for metric, factor in forecast_factors.items()}})
                for policy in POLICIES:
                    policy_factor = 1 if policy == "point" else 3
                    rows.append({"fold": fold, "candidate_id": mid + "_" + policy,
                        "regime": regime.copy(),
                        "metadata": {"counts": {"inference": count, "score": count}},
                        **{cost: {metric: value * factor * policy_factor * cost_factor
                                  for metric, factor in economic_factors.items()}
                           for cost, cost_factor in (("base", 1), ("stress_2x", 10))}})
        for control in ("bh", "common_robust"):
            rows.append({"fold": fold, "candidate_id": control, "regime": regime.copy(),
                **{cost: {metric: 0. for metric in economic_factors}
                   for cost in ("base", "stress_2x")}})
    return scores, rows


class OracleDerivativeDelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Boundary at 06:15 makes midnight's 24-bar label end exactly at it.
        cls.val_start = pd.Timestamp("2022-07-01T06:15:00Z")
        cls.val_end = pd.Timestamp("2022-10-01T06:15:00Z")
        cls.index = pd.date_range("2020-07-01T06:15:00Z", cls.val_end, freq="15min")

    def masks(self, common=None, valid=None):
        n = len(self.index)
        return segment_masks(self.index, np.ones(n, bool) if common is None else common,
            np.ones(n, bool) if valid is None else valid, registered_config(), self.val_start, self.val_end)

    def test_future_label_availability_cannot_cancel_causal_inference_or_predict(self):
        original, _ = self.masks()
        valid = np.ones(len(self.index), bool)
        valid[self.index >= self.val_start] = False
        changed, _ = self.masks(valid=valid)
        self.assertGreater(original["score"].sum(), 0)
        self.assertEqual(changed["score"].sum(), 0)
        for name in ("inference", "predict", "fit", "scale", "interval"):
            np.testing.assert_array_equal(original[name], changed[name])
        all_missing, _ = self.masks(valid=np.zeros(len(self.index), bool))
        for name in ("fit", "scale", "interval", "score"):
            self.assertEqual(all_missing[name].sum(), 0)
        for name in ("inference", "predict"):
            np.testing.assert_array_equal(original[name], all_missing[name])

    def test_fitting_segments_purge_labels_strictly_before_each_end(self):
        masks, dates = self.masks()
        self.assertEqual(pd.Timestamp(dates["train_start"]), pd.Timestamp("2020-07-01T06:15:00Z"))
        label_age = pd.Timedelta(minutes=375)
        for name, end_key in (("fit", "train_end"), ("scale", "scale_end"), ("interval", "interval_end")):
            end = pd.Timestamp(dates[end_key])
            boundary_decision = end - label_age
            equal = self.index.get_loc(boundary_decision)
            earlier = self.index.get_loc(boundary_decision - pd.Timedelta(hours=6))
            self.assertEqual(boundary_decision.hour, 0)
            self.assertFalse(masks[name][equal], name)
            self.assertTrue(masks[name][earlier], name)
            self.assertTrue((self.index[masks[name]] + label_age < end).all(), name)
        self.assertFalse((masks["fit"] & masks["scale"]).any())
        self.assertFalse((masks["scale"] & masks["interval"]).any())

    def test_scoring_allows_label_end_equal_val_end_but_orders_ignore_future_end(self):
        masks, _ = self.masks()
        exact = self.index.get_loc(self.val_end - pd.Timedelta(minutes=375))
        late = self.index.get_loc(self.val_end - pd.Timedelta(minutes=15))
        self.assertTrue(masks["score"][exact])
        self.assertTrue(masks["inference"][exact])
        self.assertFalse(masks["score"][late])
        self.assertTrue(masks["inference"][late])
        valid = np.ones(len(self.index), bool)
        valid[exact] = False
        changed, _ = self.masks(valid=valid)
        self.assertFalse(changed["score"][exact])
        self.assertTrue(changed["inference"][exact])

    def test_known_feature_gap_and_fixed_clock_restrict_inference(self):
        decision = self.index.get_loc(self.val_start.normalize() + pd.Timedelta(hours=12))
        common = np.ones(len(self.index), bool)
        common[decision] = False
        masks, _ = self.masks(common=common)
        for name in ("inference", "predict", "score"):
            self.assertFalse(masks[name][decision])
            self.assertFalse(masks[name][decision + 1])  # 12:15 is not the six-hour clock.
        inferred = self.index[masks["inference"]]
        self.assertTrue(((inferred.hour % 6 == 0) & (inferred.minute == 0)).all())
        self.assertTrue(((inferred >= self.val_start) & (inferred < self.val_end)).all())

    def test_configuration_rejects_unregistered_family_before_data_or_fitting(self):
        validate_config(registered_config())
        changes = {"schema": "other", "development_folds": [5, 6],
            "extra_delay_bars": [0, 1, 8], "horizon_bars": 96, "fit_months": 24,
            "scale_months": 2, "interval_months": 4, "versions": ["scaled"],
            "utility_risk_aversions": [0, 1], "utility_cost_multiplier": 1,
            "return_ridge_alpha": 1.0}
        for key, value in changes.items():
            with self.subTest(key=key):
                config = registered_config()
                config[key] = value
                with self.assertRaisesRegex(ValueError, "unregistered delay family"):
                    validate_config(config)
        config = registered_config()
        del config["horizon_bars"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_paired_complete_inventory_direction_and_equal_quarter_weighting(self):
        scores, rows = paired_fixture()
        self.assertEqual(len(scores), 96)
        self.assertEqual(len(rows), 208)
        paired = paired_results(scores, rows)
        self.assertEqual(set(paired), {c + "_" + v + "_vs_" + r + "_" + v
                                      for c, r in PAIRS for v in VERSIONS})
        result = paired["perp_delay0_raw_vs_technical_raw"]["regimes"]
        for regime, quarters in (("all", 8), ("bull", 2), ("bear", 4), ("sideways", 2)):
            entry = result[regime]
            self.assertEqual(entry["quarters"], quarters)
            mse = entry["forecast"]["return_mse"]
            # Per pair, [4-2, 2-8] = [2,-6], equal-quarter mean -2.
            self.assertEqual(mse["mean_difference"], -2.)
            self.assertAlmostEqual(mse["relative_loss_reduction"], .4)
            self.assertEqual(mse["improved_quarters"], quarters // 2)
            self.assertEqual(entry["forecast"]["qlike"]["mean_difference"], -4.)
            self.assertEqual(set(entry["policies"]), {"point", "utility_risk1"})
            self.assertEqual(entry["policies"]["point"]["base"],
                             {"alpha_ex": -2., "maxdd_delta": 4., "turnover": -6., "trades": -8.})
            self.assertEqual(entry["policies"]["utility_risk1"]["stress_2x"]["alpha_ex"], -60.)
        weighted_delta = (2. * 1 - 6. * 100) / 101
        self.assertNotAlmostEqual(result["all"]["forecast"]["return_mse"]["mean_difference"], weighted_delta)
        scaled = paired["perp_delay0_scaled_vs_technical_scaled"]["regimes"]["all"]
        self.assertEqual(scaled["forecast"]["return_mse"]["mean_difference"], -6.)
        self.assertEqual(scaled["policies"]["utility_risk1"]["stress_2x"]["alpha_ex"], -180.)
        frozen = paired["perp_delay1_raw_vs_frozen_delay1_raw"]["regimes"]["all"]
        self.assertEqual(frozen["forecast"]["return_mse"]["mean_difference"], -2.)

    def test_missing_duplicate_and_unexpected_forecast_or_policy_reject_entire_result(self):
        original_scores, original_rows = paired_fixture()
        for collection in ("scores", "rows"):
            for fault in ("missing", "duplicate", "unexpected"):
                with self.subTest(collection=collection, fault=fault):
                    scores, rows = copy.deepcopy(original_scores), copy.deepcopy(original_rows)
                    target = scores if collection == "scores" else rows
                    if fault == "missing":
                        target.pop()
                    elif fault == "duplicate":
                        target.append(copy.deepcopy(target[0]))
                    else:
                        target[0]["model_id" if collection == "scores" else "candidate_id"] = "unregistered"
                    with self.assertRaisesRegex(ValueError, "missing, duplicate or unexpected"):
                        paired_results(scores, rows)

    def test_forecast_pairing_rejects_different_sample_count_or_regime(self):
        for field, value in (("rows", 999), ("regime", {"trend": "sideways"})):
            with self.subTest(field=field):
                scores, rows = paired_fixture()
                candidate = next(s for s in scores if s["fold"] == 5 and s["model_id"] == "perp_delay1_scaled")
                candidate[field] = value
                with self.assertRaisesRegex(ValueError, "unpaired"):
                    paired_results(scores, rows)

    def test_policy_regime_mismatch_rejects_entire_result(self):
        scores, rows = paired_fixture()
        rows[0]["regime"] = {"trend": "unavailable"}
        with self.assertRaisesRegex(ValueError, "unpaired result regimes"):
            paired_results(scores, rows)

    def test_zero_reference_loss_retains_absolute_difference_without_ratio(self):
        scores, rows = paired_fixture()
        for score in scores:
            if score["model_id"] == "technical_raw":
                for metric in ("return_mse", "qlike", "variance_mse", "rms_mse"):
                    score[metric] = 0.
        result = paired_results(scores, rows)["perp_delay0_raw_vs_technical_raw"]["regimes"]["all"]["forecast"]["return_mse"]
        self.assertEqual(result["mean_difference"], 3.)
        self.assertEqual(result["improved_quarters"], 0)
        self.assertIsNone(result["relative_loss_reduction"])


if __name__ == "__main__":
    unittest.main()
