import copy
from pathlib import Path
import unittest

import yaml

from unidream.experiments.oracle_mean_shrinkage_decisions import (
    CONTROL_IDS, ENDPOINTS, FOLDS, HALVES, NEW_IDS, PREDICTION_PAIRS, RULES, compare, validate_config,
)


def fixture():
    values = {"scale_mean": (2., 8.), "technical_scaled": (4., 2.), "perp_delay0_scaled": (3., 5.),
              "technical_half": (1., 6.), "perp_delay0_half": (2., 7.)}
    scores, rows = [], []
    factors = {"alpha_ex": 1., "maxdd_delta": -2., "turnover": 3., "trades": 4.,
               "fees_initial_equity_units": .5, "borrow_initial_equity_units": .25}
    for i, fold in enumerate(FOLDS):
        regime = {"trend": "bull" if i < 2 else "bear" if i < 6 else "sideways"}
        for mean in ENDPOINTS + tuple(HALVES):
            value = values[mean][i % 2]
            scores.append({"fold": fold, "mean_id": mean, "regime": regime.copy(),
                "rows": 1 if i % 2 == 0 else 100, "return_mse": value, "return_mae": value / 4,
                "return_sign_accuracy": .25 if i % 2 == 0 else .75,
                "zero_return_mse": 5., "fit_mean_return_mse": 6.,
                "return_rank_ic": None if mean == "scale_mean" else -.5 if i % 2 == 0 else .5})
            for rule, weight in zip(RULES, (1., 3.)):
                rows.append({"fold": fold, "candidate_id": mean + "_" + rule, "regime": regime.copy(),
                    **{cost: {key: value * factor * weight * mult for key, factor in factors.items()}
                       for cost, mult in (("base", 1.), ("stress_2x", 10.))}})
        for cid in ("bh", "common_robust"):
            rows.append({"fold": fold, "candidate_id": cid, "regime": regime.copy(),
                         **{cost: {key: 0. for key in factors} for cost in ("base", "stress_2x")}})
    return scores, rows


class OracleMeanShrinkageDecisionsTests(unittest.TestCase):
    def test_registered_half_family_forbids_weight_selection_or_weaker_support(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/oracle_mean_shrinkage_decisions_20260905.yaml").read_text())
        validate_config(cfg)
        for key, value in {"weight": .25, "half_sources": {"perp_delay0_half": "perp_delay0_scaled"},
                "anchor_source": "fit_mean", "variance_source": "perp_delay0_scaled", "rules": [RULES[0]],
                "minimum_quarters_per_regime": 2, "score_rows": 2586,
                "fallback_eligible_rows": 0, "adaptive_prior_policy_names_observed": 0}.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(cfg); changed[key] = value
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    validate_config(changed)

    def test_exact_five_prediction_pairs_and_twelve_economic_comparisons(self):
        scores, rows = fixture()
        self.assertEqual((len(scores), len(rows)), (40, 96))
        pairs, predictions, rules = compare(scores, rows)
        self.assertEqual(set(pairs), {a + "_vs_" + b for a, b in PREDICTION_PAIRS})
        self.assertEqual(len(pairs), 5)
        self.assertEqual(set(predictions), set(ENDPOINTS) | set(HALVES))
        self.assertEqual(set(rules), set(HALVES))
        self.assertEqual(len(CONTROL_IDS), 8); self.assertEqual(len(NEW_IDS), 4)

    def test_pairs_use_equal_quarter_losses_and_own_rule_cost_differences(self):
        scores, rows = fixture(); pairs, _, rules = compare(scores, rows)
        for regime, quarters in (("all", 8), ("bull", 2), ("bear", 4), ("sideways", 2)):
            p = pairs["technical_half_vs_scale_mean"]["regimes"][regime]
            self.assertEqual(p["quarters"], quarters)
            self.assertEqual(p["prediction"]["return_mse"]["mean_difference"], -1.5)
            self.assertAlmostEqual(p["prediction"]["return_mse"]["relative_loss_reduction"], .3)
            self.assertEqual(p["prediction"]["return_mse"]["improved_quarters"], quarters)
            self.assertEqual(p["policies"][RULES[0]]["base"]["alpha_ex"], -1.5)
            self.assertEqual(p["policies"][RULES[1]]["stress_2x"]["alpha_ex"], -45.)
            self.assertEqual(rules["technical_half"]["regimes"][regime]["base"]["alpha_ex"], 7.)

    def test_summary_keeps_row_pooled_loss_separate_and_constant_ic_undefined(self):
        scores, rows = fixture(); _, predictions, _ = compare(scores, rows)
        p = predictions["technical_half"]["all"]
        self.assertEqual(p["metrics"]["return_mse"]["equal_quarter_mean"], 3.5)
        self.assertAlmostEqual(p["metrics"]["return_mse"]["pooled_row_mean"], 601 / 101)
        self.assertEqual(p["mean_return_rank_ic"], 0.)
        self.assertIsNone(predictions["scale_mean"]["all"]["mean_return_rank_ic"])

    def test_missing_duplicate_unknown_or_unpaired_scores_and_policies_fail(self):
        for collection in (0, 1):
            for fault in ("missing", "duplicate", "unknown", "regime"):
                scores, rows = fixture(); target = (scores, rows)[collection]
                if fault == "missing": target.pop()
                elif fault == "duplicate": target.append(copy.deepcopy(target[0]))
                elif fault == "unknown": target[0]["candidate_id" if collection else "mean_id"] = "unknown"
                else: target[0]["regime"] = {"trend": "bear"}
                with self.subTest(collection=collection, fault=fault), self.assertRaises(ValueError):
                    compare(scores, rows)
        scores, rows = fixture(); scores[0]["rows"] += 1
        with self.assertRaises(ValueError): compare(scores, rows)

    def test_zero_reference_loss_preserves_absolute_difference_without_ratio(self):
        scores, rows = fixture()
        for s in scores:
            if s["mean_id"] == "scale_mean": s["return_mse"] = s["return_mae"] = 0.
        pairs, _, _ = compare(scores, rows)
        p = pairs["technical_half_vs_scale_mean"]["regimes"]["all"]["prediction"]
        self.assertEqual(p["return_mse"]["mean_difference"], 3.5)
        self.assertIsNone(p["return_mse"]["relative_loss_reduction"])


if __name__ == "__main__":
    unittest.main()
