"""Synthetic registration and complete-family summary checks; no real fits."""
import copy
import unittest

from unidream.experiments.oracle_mean_reliability_decisions import (
    FIXED, FOLDS, GROUPS, MEANS, POLICIES, SEGMENTS, SOURCES, summarize, validate_config,
)


def family():
    rows, scores, fits = [], [], []
    for f in FOLDS:
        regime = {"trend": "bull" if f in (5, 7) else "sideways" if f in (9, 12) else "bear"}
        for p in POLICIES:
            rows.append({"fold": f, "candidate_id": p, "regime": regime,
                **{c: {"alpha_ex": .01, "maxdd_delta": -.02, "turnover": 2., "trades": 4.}
                   for c in ("base", "stress_2x")}})
        for seg in SEGMENTS:
            for m in MEANS:
                scores.append({"fold": f, "segment": seg, "mean_id": m, "regime": regime,
                    "rows": 16 * (f - 4), "return_mse": float(f - 4), "return_mae": 1.,
                    "zero_return_mse": 10., "fit_mean_return_mse": 11., "return_rank_ic": None,
                    "decomposition": {k: 0. for k in ("lossdiff", "innovation_secondmoment", "crossmoment",
                        "centered_component", "drift_component", "identityresidual")}})
        for g in GROUPS:
            fits.append({"fold": f, "group": g, "fit": {"weight": .5}})
    return rows, scores, fits


class ReliabilityDecisionsTests(unittest.TestCase):
    def test_fixed_registration_rejects_scope_or_type_changes(self):
        cfg = {**copy.deepcopy(FIXED), "source_bindings": {s: "abc" for s in SOURCES},
               "source_prepare_config_sha256": "abc", "preflight_sha256": None}
        validate_config(cfg)
        for key, value in (("development_folds", [15]), ("selection_permitted", True),
                           ("new_causal_policy_names", True), ("fit_segment", "interval"),
                           ("weight_bounds", [-1., 1.])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**cfg, key: value})
        with self.assertRaises(ValueError): validate_config({**cfg, "extra": 1})

    def test_equal_quarter_is_not_pooled_and_constant_rank_is_undefined(self):
        result = summarize(*family())
        p = result["prediction"]["all"]["evaluation"]["scale_mean"]
        self.assertEqual(p["equal_quarter_mse"], 4.5)
        self.assertAlmostEqual(p["pooled_row_mse"], 204 / 36)
        self.assertIsNone(p["mean_rank_ic"])
        self.assertEqual(p["mse_minus_zero"], -5.5)
        for conditions in result["direction"].values():
            self.assertTrue(conditions["economic_means_all_strata_both_costs"])
            self.assertFalse(conditions["predictive_mse_vs_zero_scale_full_half_all_strata"]["evaluation"])
            self.assertFalse(conditions["regime_count_gate_pass"])
        self.assertFalse(result["high_probability_generalization_established"])

    def test_incomplete_duplicate_or_unpaired_family_fails_closed(self):
        for target in (0, 1, 2):
            args = family(); args[target].pop()
            with self.subTest(missing=target), self.assertRaises(ValueError): summarize(*args)
            args = family(); args[target].append(args[target][0])
            with self.subTest(duplicate=target), self.assertRaises(ValueError): summarize(*args)
        args = family(); args[1][1]["rows"] += 1
        with self.assertRaises(ValueError): summarize(*args)
        args = family(); args[1][1]["regime"] = {"trend": "sideways"}
        with self.assertRaises(ValueError): summarize(*args)

    def test_missing_regime_nonfinite_or_invalid_weight_is_not_zero_success(self):
        args = family()
        for r in args[0] + args[1]: r["regime"] = {"trend": "bear"}
        with self.assertRaises(ValueError): summarize(*args)
        args = family(); args[1][0]["return_mse"] = float("nan")
        with self.assertRaises(ValueError): summarize(*args)
        args = family(); args[2][0]["fit"]["weight"] = 2.
        with self.assertRaises(ValueError): summarize(*args)


if __name__ == "__main__": unittest.main()
