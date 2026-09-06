"""Synthetic family completeness and nonselecting paired-summary tests."""
import copy
import unittest

from unidream.experiments.oracle_short_feature_decisions import (
    FIXED, EXTRA, FOLDS, POLICIES, NEW_MEANS, SCORE_MEANS, SOURCES, summarize, validate_config,
)


def family():
    rows, scores = [], []
    for f in FOLDS:
        regime = {"trend": "bull" if f in (5, 7) else "sideways" if f in (9, 12) else "bear"}
        for p in POLICIES:
            rows.append({"fold": f, "candidate_id": p, "regime": regime,
                **{c: {"alpha_ex": .01, "maxdd_delta": -.02, "turnover": 2., "trades": 4.}
                   for c in ("base", "stress_2x")}})
        for seg, means in SCORE_MEANS.items():
            for m in means:
                scores.append({"fold": f, "segment": seg, "mean_id": m, "regime": regime,
                    "rows": 16 * (f - 4), "return_mse": float(f - 4), "return_mae": 1.,
                    "zero_return_mse": 10., "fit_mean_return_mse": 11., "return_rank_ic": None})
    return rows, scores


class ShortFeatureDecisionsTests(unittest.TestCase):
    def test_fixed_family_rejects_calibration_window_model_and_type_changes(self):
        cfg = {**copy.deepcopy(FIXED), **{k: "abc" for k in EXTRA},
            "source_bindings": {s: "abc" for s in SOURCES}, "preflight_sha256": None}
        validate_config(cfg)
        for k, v in (("development_folds", [15]), ("selection_permitted", True), ("new_causal_policy_names", True),
                     ("return_calibration", "scaled"), ("ridge_alpha", 10.), ("feature_shift_bars", 0),
                     ("group_dimensions", [29, 34, 32, 37, 8]), ("weight_fitting_permitted", True)):
            with self.subTest(key=k), self.assertRaises(ValueError): validate_config({**cfg, k: v})
        with self.assertRaises(ValueError): validate_config({**cfg, "extra": 1})

    def test_equal_quarter_not_pooled_and_undefined_rank_not_dropped(self):
        r = summarize(*family()); p = r["prediction"]["all"]["evaluation"]["technical_raw"]
        self.assertEqual(p["equal_quarter_mse"], 4.5)
        self.assertAlmostEqual(p["pooled_row_mse"], 204 / 36)
        self.assertIsNone(p["mean_rank_ic"])
        self.assertTrue(r["interval_regime_strata_are_retrospective_evaluation_groupings"])
        for d in r["direction"].values():
            self.assertTrue(d["economic_means_all_strata_both_costs"])
            self.assertFalse(any(d["predictive_mse_vs_zero_fitmean_and_all_references_all_strata"].values()))
            self.assertFalse(d["high_probability_generalization_established"])

    def test_missing_duplicate_unpaired_nonfinite_or_missing_regime_rejected(self):
        for target in (0, 1):
            a = family(); a[target].pop()
            with self.subTest(missing=target), self.assertRaises(ValueError): summarize(*a)
            a = family(); a[target].append(a[target][0])
            with self.subTest(duplicate=target), self.assertRaises(ValueError): summarize(*a)
        for k, v in (("rows", 17), ("regime", {"trend": "sideways"}), ("return_mse", float("nan"))):
            a = family(); a[1][0][k] = v
            with self.subTest(key=k), self.assertRaises(ValueError): summarize(*a)
        a = family()
        for r in a[0]+a[1]: r["regime"] = {"trend": "bear"}
        with self.assertRaises(ValueError): summarize(*a)

    def test_both_block_must_beat_each_single_and_every_stratum(self):
        a = family()
        for s in a[1]:
            if s["mean_id"] in NEW_MEANS:
                s["return_mse"] = .2 if s["mean_id"] == NEW_MEANS[2] else .3
        r = summarize(*a)
        for d in r["direction"].values():
            self.assertTrue(all(d["predictive_mse_vs_zero_fitmean_and_all_references_all_strata"].values()))
        for s in a[1]:
            if s["fold"] == 9 and s["segment"] == "interval" and s["mean_id"] == NEW_MEANS[2]: s["return_mse"] = .9
        r = summarize(*a); d = r["direction"][NEW_MEANS[2]+"_utility_risk1"]
        self.assertFalse(d["predictive_mse_vs_zero_fitmean_and_all_references_all_strata"]["interval"])
        self.assertTrue(d["predictive_mse_vs_zero_fitmean_and_all_references_all_strata"]["evaluation"])
        self.assertFalse(r["selection_performed"])


if __name__ == "__main__": unittest.main()
