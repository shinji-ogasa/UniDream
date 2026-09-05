import copy
import unittest

from unidream.experiments.oracle_mean_control_decisions import (
    FOLDS, MEANS, POLICIES, paired_summaries, prediction_summaries, validate_config,
)


def registered_config():
    return {"schema": "oracle-mean-control-decisions-v1", "development_folds": list(range(5, 13)),
        "mean_sources": ["zero", "fit_mean", "scale_mean", "technical_raw", "technical_scaled",
                         "perp_delay0_raw", "perp_delay0_scaled"],
        "variance_source": "technical_scaled", "policies": ["point", "utility_risk1"],
        "utility_cost_multiplier": 2, "minimum_quarters_per_regime": 3,
        "inference_rows": 2586, "score_rows": 2574}


def fixture():
    """Every regime has paired small/large quarters, deliberately 1/100 rows."""
    values = {"zero": (2., 8.), "fit_mean": (4., 2.), "scale_mean": (1., 9.),
              "technical_raw": (3., 5.), "technical_scaled": (1., 6.),
              "perp_delay0_raw": (4., 3.), "perp_delay0_scaled": (2., 7.)}
    economic_factors = {"alpha_ex": 1., "maxdd_delta": -2., "turnover": 3., "trades": 4.,
                        "fees_initial_equity_units": .5, "borrow_initial_equity_units": .25}
    scores, rows = [], []
    for i, fold in enumerate(FOLDS):
        regime = {"trend": "bull" if i < 2 else "bear" if i < 6 else "sideways"}
        count = 1 if i % 2 == 0 else 100
        for mean in MEANS:
            value = values[mean][i % 2]
            scores.append({"fold": fold, "mean_id": mean, "regime": regime.copy(), "rows": count,
                "return_mse": value, "return_mae": value / 4,
                "return_sign_accuracy": .25 if i % 2 == 0 else .75,
                "zero_return_mse": values["zero"][i % 2],
                "fit_mean_return_mse": values["fit_mean"][i % 2],
                "return_rank_ic": None if mean in ("zero", "fit_mean", "scale_mean")
                                  else -.5 if i % 2 == 0 else .5})
            for policy in POLICIES:
                policy_factor = 1 if policy == "point" else 3
                rows.append({"fold": fold, "candidate_id": mean + "_" + policy,
                    "regime": regime.copy(),
                    **{cost: {metric: value * factor * policy_factor * cost_factor
                              for metric, factor in economic_factors.items()}
                       for cost, cost_factor in (("base", 1), ("stress_2x", 10))}})
        for control in ("bh", "common_robust"):
            rows.append({"fold": fold, "candidate_id": control, "regime": regime.copy(),
                         **{cost: {metric: 0. for metric in economic_factors}
                            for cost in ("base", "stress_2x")}})
    return scores, rows


class OracleMeanControlDecisionsTests(unittest.TestCase):
    def test_configuration_requires_exact_mean_variance_policy_and_support_family(self):
        validate_config(registered_config())
        changes = {"schema": "unregistered", "development_folds": [5, 6],
            "mean_sources": ["zero", "technical_scaled"], "variance_source": "perp_delay0_scaled",
            "policies": ["point", "utility_risk0"], "utility_cost_multiplier": 1,
            "minimum_quarters_per_regime": 2, "inference_rows": 2587, "score_rows": 2575}
        for key, value in changes.items():
            with self.subTest(key=key):
                cfg = registered_config()
                cfg[key] = value
                with self.assertRaisesRegex(ValueError, "unregistered mean control family"):
                    validate_config(cfg)
        cfg = registered_config()
        del cfg["variance_source"]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_complete_inventory_has_exact_21_directed_pairs(self):
        scores, rows = fixture()
        self.assertEqual(len(scores), 56)
        self.assertEqual(len(rows), 128)
        expected = {
            "fit_mean_vs_zero", "scale_mean_vs_zero", "technical_raw_vs_zero",
            "technical_scaled_vs_zero", "perp_delay0_raw_vs_zero", "perp_delay0_scaled_vs_zero",
            "scale_mean_vs_fit_mean", "technical_raw_vs_fit_mean", "technical_scaled_vs_fit_mean",
            "perp_delay0_raw_vs_fit_mean", "perp_delay0_scaled_vs_fit_mean",
            "technical_raw_vs_scale_mean", "technical_scaled_vs_scale_mean",
            "perp_delay0_raw_vs_scale_mean", "perp_delay0_scaled_vs_scale_mean",
            "technical_scaled_vs_technical_raw", "perp_delay0_raw_vs_technical_raw",
            "perp_delay0_scaled_vs_technical_raw", "perp_delay0_raw_vs_technical_scaled",
            "perp_delay0_scaled_vs_technical_scaled", "perp_delay0_scaled_vs_perp_delay0_raw",
        }
        paired = paired_summaries(scores, rows)
        self.assertEqual(set(paired), expected)
        for name, entry in paired.items():
            self.assertEqual(name, entry["candidate"] + "_vs_" + entry["reference"])
            self.assertEqual(entry["difference_convention"], "candidate minus reference")

    def test_paired_losses_and_economics_use_equal_quarter_differences(self):
        scores, rows = fixture()
        result = paired_summaries(scores, rows)["fit_mean_vs_zero"]["regimes"]
        for regime, quarters in (("all", 8), ("bull", 2), ("bear", 4), ("sideways", 2)):
            entry = result[regime]
            self.assertEqual(entry["quarters"], quarters)
            loss = entry["prediction"]["return_mse"]
            # [+2,-6] per quarter pair averages to -2, independent of 1/100 rows.
            self.assertEqual(loss["mean_difference"], -2.)
            self.assertAlmostEqual(loss["relative_loss_reduction"], .4)
            self.assertEqual(loss["improved_quarters"], quarters // 2)
            self.assertEqual(entry["prediction"]["return_mae"]["mean_difference"], -.5)
            self.assertEqual(set(entry["policies"]), {"point", "utility_risk1"})
            self.assertEqual(entry["policies"]["point"]["base"],
                {"alpha_ex": -2., "maxdd_delta": 4., "turnover": -6., "trades": -8.,
                 "fees_initial_equity_units": -1., "borrow_initial_equity_units": -.5})
            self.assertEqual(entry["policies"]["utility_risk1"]["stress_2x"]["alpha_ex"], -60.)
        self.assertNotAlmostEqual(result["all"]["prediction"]["return_mse"]["mean_difference"], -598 / 101)

    def test_prediction_summary_separates_equal_quarter_and_pooled_row_means(self):
        scores, _ = fixture()
        result = prediction_summaries(scores)
        self.assertEqual(set(result), set(MEANS))
        for regime, quarters, count in (("all", 8, 404), ("bull", 2, 101),
                                        ("bear", 4, 202), ("sideways", 2, 101)):
            fit = result["fit_mean"][regime]
            self.assertEqual((fit["quarters"], fit["rows"]), (quarters, count))
            self.assertEqual(fit["metrics"]["return_mse"]["equal_quarter_mean"], 3.)
            self.assertAlmostEqual(fit["metrics"]["return_mse"]["pooled_row_mean"], 204 / 101)
            self.assertEqual(fit["metrics"]["return_mae"]["equal_quarter_mean"], .75)
            self.assertAlmostEqual(fit["metrics"]["return_mae"]["pooled_row_mean"], 51 / 101)
            self.assertEqual(fit["metrics"]["return_sign_accuracy"]["equal_quarter_mean"], .5)
            self.assertAlmostEqual(fit["metrics"]["return_sign_accuracy"]["pooled_row_mean"], 75.25 / 101)
            zero = result["zero"][regime]
            pooled_difference = fit["metrics"]["return_mse"]["pooled_row_mean"] - zero["metrics"]["return_mse"]["pooled_row_mean"]
            self.assertAlmostEqual(pooled_difference, -598 / 101)

    def test_constant_or_unavailable_rank_ic_is_retained_as_none(self):
        scores, _ = fixture()
        result = prediction_summaries(scores)
        for mean in ("zero", "fit_mean", "scale_mean"):
            for regime in ("all", "bull", "bear", "sideways"):
                self.assertIsNone(result[mean][regime]["mean_return_rank_ic"])
        self.assertEqual(result["technical_scaled"]["all"]["mean_return_rank_ic"], 0.)
        next(s for s in scores if s["fold"] == 5 and s["mean_id"] == "technical_scaled")["return_rank_ic"] = None
        changed = prediction_summaries(scores)
        self.assertIsNone(changed["technical_scaled"]["all"]["mean_return_rank_ic"])
        self.assertIsNone(changed["technical_scaled"]["bull"]["mean_return_rank_ic"])
        self.assertEqual(changed["technical_scaled"]["bear"]["mean_return_rank_ic"], 0.)

    def test_pairing_rejects_missing_duplicate_and_unknown_score_or_policy(self):
        for collection in ("scores", "rows"):
            for fault in ("missing", "duplicate", "unknown"):
                with self.subTest(collection=collection, fault=fault):
                    scores, rows = fixture()
                    target = scores if collection == "scores" else rows
                    if fault == "missing":
                        target.pop()
                    elif fault == "duplicate":
                        target.append(copy.deepcopy(target[0]))
                    else:
                        target[0]["mean_id" if collection == "scores" else "candidate_id"] = "unknown"
                    with self.assertRaises(ValueError):
                        paired_summaries(scores, rows)

    def test_pairing_rejects_score_support_and_control_or_policy_regime_mismatch(self):
        for field, value in (("rows", 999), ("regime", {"trend": "sideways"})):
            with self.subTest(score_field=field):
                scores, rows = fixture()
                next(s for s in scores if s["fold"] == 5 and s["mean_id"] == "fit_mean")[field] = value
                with self.assertRaisesRegex(ValueError, "unpaired mean control support"):
                    paired_summaries(scores, rows)
        for policy in ("bh", "common_robust", "technical_scaled_utility_risk1"):
            with self.subTest(policy=policy):
                scores, rows = fixture()
                next(r for r in rows if r["fold"] == 5 and r["candidate_id"] == policy)["regime"] = {"trend": "sideways"}
                with self.assertRaisesRegex(ValueError, "unpaired mean control support"):
                    paired_summaries(scores, rows)

    def test_prediction_summary_rejects_incomplete_duplicate_or_unpaired_scores(self):
        for fault in ("missing", "duplicate", "unknown", "regime", "count"):
            with self.subTest(fault=fault):
                scores, _ = fixture()
                if fault == "missing":
                    scores.pop()
                elif fault == "duplicate":
                    scores.append(copy.deepcopy(scores[0]))
                elif fault == "unknown":
                    scores[0]["mean_id"] = "unknown"
                elif fault == "regime":
                    scores[1]["regime"] = {"trend": "sideways"}
                else:
                    scores[1]["rows"] = 999
                with self.assertRaises(ValueError):
                    prediction_summaries(scores)

    def test_zero_reference_loss_has_no_ratio_but_retains_absolute_difference(self):
        scores, rows = fixture()
        for score in scores:
            if score["mean_id"] == "zero":
                score["return_mse"] = score["return_mae"] = 0.
        result = paired_summaries(scores, rows)["fit_mean_vs_zero"]["regimes"]["all"]["prediction"]
        self.assertEqual(result["return_mse"]["mean_difference"], 3.)
        self.assertEqual(result["return_mae"]["mean_difference"], .75)
        for metric in result.values():
            self.assertIsNone(metric["relative_loss_reduction"])
            self.assertEqual(metric["improved_quarters"], 0)


if __name__ == "__main__":
    unittest.main()
