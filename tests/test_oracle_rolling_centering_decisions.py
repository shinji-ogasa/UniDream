"""Synthetic complete-family accounting and registration checks; no real run."""
import copy
import unittest

from unidream.experiments.oracle_rolling_centering_decisions import (
    FIXED, EXTRA, FOLDS, GROUPS, MEANS, NEW_MEANS, POLICIES, SOURCES, summarize, validate_config,
)


def family():
    rows, scores, weights = [], [], []
    for f in FOLDS:
        regime = {"trend": "bull" if f in (5, 7) else "sideways" if f in (9, 12) else "bear"}
        for p in POLICIES:
            rows.append({"fold": f, "candidate_id": p, "regime": regime,
                **{c: {"alpha_ex": .01, "maxdd_delta": -.02, "turnover": 2., "trades": 4.}
                   for c in ("base", "stress_2x")}})
        for m in MEANS:
            scores.append({"fold": f, "segment": "evaluation", "mean_id": m, "regime": regime,
                "rows": 16 * (f - 4), "return_mse": float(f - 4), "return_mae": 1.,
                "zero_return_mse": 10., "fit_mean_return_mse": 11., "return_rank_ic": None,
                "decomposition": {k: 0. for k in ("lossdiff", "innovation_secondmoment", "crossmoment",
                    "centered_component", "drift_component", "identityresidual")}})
        for g in GROUPS: weights.append({"fold": f, "group": g, "fit": {"weight": .5}})
    return rows, scores, weights


class RollingDecisionsTests(unittest.TestCase):
    def test_registration_rejects_window_update_scope_or_type_changes(self):
        cfg = {**copy.deepcopy(FIXED), **{k: "abc" for k in EXTRA},
               "source_bindings": {s: "abc" for s in SOURCES}, "preflight_sha256": None}
        validate_config(cfg)
        for key, value in (("development_folds", [15]), ("selection_permitted", True),
                           ("new_causal_policy_names", True), ("history_calendar_months", 6),
                           ("minimum_history_pairs", 32), ("new_weight_fitting_permitted", True),
                           ("maturity_minutes", 360), ("segments", ["interval", "evaluation"])):
            with self.subTest(key=key), self.assertRaises(ValueError): validate_config({**cfg, key: value})
        with self.assertRaises(ValueError): validate_config({**cfg, "extra": 1})

    def test_equal_quarter_vs_pooled_and_strict_direction(self):
        result = summarize(*family()); p = result["prediction"]["all"]["scale_mean"]
        self.assertEqual(p["equal_quarter_mse"], 4.5)
        self.assertAlmostEqual(p["pooled_row_mse"], 204 / 36)
        self.assertIsNone(p["mean_rank_ic"])
        self.assertEqual(p["mse_minus_zero"], -5.5)
        for c in result["direction"].values():
            self.assertTrue(c["economic_means_all_strata_both_costs"])
            self.assertFalse(c["predictive_mse_vs_zero_fitmean_and_all_registered_references_all_strata"])
            self.assertFalse(c["regime_count_gate_pass"])
        for e in result["economics"]["all"].values(): self.assertEqual(e["joint_positive_quarters_both_costs"], 8)
        self.assertFalse(result["high_probability_generalization_established"])

    def test_missing_duplicate_unpaired_or_nonfinite_fails(self):
        for target in (0, 1, 2):
            args = family(); args[target].pop()
            with self.subTest(missing=target), self.assertRaises(ValueError): summarize(*args)
            args = family(); args[target].append(args[target][0])
            with self.subTest(duplicate=target), self.assertRaises(ValueError): summarize(*args)
        for key, value in (("rows", 17), ("regime", {"trend": "sideways"}),
                           ("segment", "interval"), ("return_mse", float("nan"))):
            args = family(); args[1][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): summarize(*args)
        args = family(); args[2][0]["fit"]["weight"] = 2.
        with self.assertRaises(ValueError): summarize(*args)
        args = family()
        for r in args[0] + args[1]: r["regime"] = {"trend": "bear"}
        with self.assertRaises(ValueError): summarize(*args)

    def test_rolling_anchor_incremental_reference_and_adverse_stratum_retained(self):
        args = family()
        for s in args[1]:
            if s["mean_id"] in NEW_MEANS:
                s["return_mse"] = .5 if s["mean_id"] == "rolling_anchor" else .25
        result = summarize(*args)
        for c in result["direction"].values():
            self.assertTrue(c["predictive_mse_vs_zero_fitmean_and_all_registered_references_all_strata"])
        for s in args[1]:
            if s["fold"] == 9 and s["mean_id"] == "technical_rolling": s["return_mse"] = 9.
        result = summarize(*args)
        self.assertFalse(result["direction"]["technical_rolling_utility_risk1"][
            "predictive_mse_vs_zero_fitmean_and_all_registered_references_all_strata"])
        self.assertGreater(result["paired"]["sideways"]["technical_rolling"]["rolling_anchor"]["prediction"]["mse_difference"], 0)


if __name__ == "__main__": unittest.main()
