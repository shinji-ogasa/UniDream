import copy
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_ablation import (
    mask_digest, paired_summaries, score_forecast,
)


def paired_fixture():
    """Two deliberately unequal-size quarters with hand-computable differences."""
    losses = {"technical": [2., 8.], "perp_flow": [4., 2.], "derivative": [1., 5.]}
    forecast_factors = {"return_mse": 1., "qlike": 2., "variance_mse": 3., "rms_mse": 4.}
    policy_factors = {"alpha_ex": 1., "maxdd_delta": -2., "turnover": 3., "trades": 4.}
    scores, rows = [], []
    for group, values in losses.items():
        for version, version_factor in (("raw", 1.), ("scaled", 3.)):
            for j, (fold, sample_count) in enumerate(((5, 1), (6, 100))):
                value = values[j] * version_factor
                mid = group + "_" + version
                scores.append({"fold": fold, "model_id": mid, "rows": sample_count,
                               **{metric: value * factor for metric, factor in forecast_factors.items()}})
                for policy, policy_factor in (("point", 1.), ("utility_risk0", 2.), ("utility_risk1", 3.)):
                    rows.append({
                        "fold": fold, "candidate_id": mid + "_" + policy,
                        "metadata": {"inference_rows": sample_count, "score_rows": sample_count},
                        **{cost: {metric: value * factor * policy_factor * cost_factor
                                  for metric, factor in policy_factors.items()}
                           for cost, cost_factor in (("base", 1.), ("stress_2x", 10.))},
                    })
    return scores, rows


class OracleDerivativeAblationTests(unittest.TestCase):
    def test_mask_digest_binds_calendar_and_each_boolean_slot(self):
        index = pd.date_range("2022-01-01", periods=4, freq="6h", tz="UTC")
        mask = np.array([True, False, True, False])
        original = mask_digest(index, mask)
        self.assertEqual(original, mask_digest(index.tz_convert("Asia/Tokyo"), mask.tolist()))
        self.assertNotEqual(original, mask_digest(index + pd.Timedelta(hours=6), mask))
        # Equal counts at different timestamps must have different bindings.
        self.assertNotEqual(original, mask_digest(index, np.array([False, True, True, False])))

    def test_score_references_are_zero_fit_mean_and_variance_persistence(self):
        # Return error is [1,0]; variance forecasts [1,4] exactly match outcomes.
        y = np.array([[-1., 0., 1.], [1., 0., 2.]])
        scored = score_forecast(y, np.array([0., 1.]), np.array([1., 4.]),
                                qr=2., qv=2., fit_mean=.5, persistence=np.ones(2))
        self.assertEqual(scored["rows"], 2)
        self.assertEqual(scored["return_mse"], .5)
        self.assertEqual(scored["zero_return_mse"], 1.)
        self.assertEqual(scored["fit_mean_return_mse"], 1.25)
        self.assertEqual(scored["return_skill_vs_zero"], .5)
        self.assertAlmostEqual(scored["return_skill_vs_fit_mean"], .6)
        self.assertEqual(scored["return_sign_accuracy"], 1.)
        self.assertAlmostEqual(scored["return_rank_ic"], 1.)
        self.assertEqual(scored["variance_mse"], 0.)
        self.assertEqual(scored["persistence96"]["variance_mse"], 4.5)
        self.assertEqual(scored["persistence96"]["rms_mse"], .5)
        self.assertAlmostEqual(scored["persistence96"]["qlike"], (3. - np.log(4.)) / 2.)

    def test_zero_mse_reference_and_constant_rank_are_undefined_not_infinite(self):
        y = np.array([[0., 0., 1.], [0., 0., 1.]])
        for mu in (np.zeros(2), np.ones(2)):
            scored = score_forecast(y, mu, np.ones(2), qr=2., qv=2., fit_mean=0., persistence=np.ones(2))
            self.assertIsNone(scored["return_skill_vs_zero"])
            self.assertIsNone(scored["return_skill_vs_fit_mean"])
            self.assertIsNone(scored["return_rank_ic"])

    def test_paired_direction_equal_quarter_weighting_versions_and_policy_keys(self):
        scores, rows = paired_fixture()
        paired = paired_summaries(scores, rows)
        self.assertEqual(len(paired), 6)
        raw = paired["perp_flow_raw_vs_technical_raw"]
        self.assertEqual(raw["folds"], 2)
        self.assertEqual(raw["difference_convention"], "candidate minus reference")
        self.assertEqual(set(raw["policies"]), {"point", "utility_risk0", "utility_risk1"})
        # [4-2, 2-8] averages to -2, irrespective of quarter sample sizes 1/100.
        self.assertEqual(raw["forecast"]["return_mse"]["mean_difference"], -2.)
        self.assertAlmostEqual(raw["forecast"]["return_mse"]["relative_loss_reduction"], .4)
        self.assertEqual(raw["forecast"]["return_mse"]["improved_quarters"], 1)
        self.assertEqual(raw["policies"]["point"]["base"],
                         {"alpha_ex": -2., "maxdd_delta": 4., "turnover": -6., "trades": -8.})
        self.assertEqual(raw["policies"]["utility_risk1"]["stress_2x"]["alpha_ex"], -60.)
        self.assertEqual(paired["perp_flow_scaled_vs_technical_scaled"]["forecast"]["return_mse"]["mean_difference"], -6.)
        tied = paired["derivative_raw_vs_perp_flow_raw"]["forecast"]["return_mse"]
        self.assertEqual(tied["mean_difference"], 0.)
        self.assertEqual(tied["improved_quarters"], 1)

    def test_paired_forecasts_reject_missing_fold(self):
        scores, rows = paired_fixture()
        scores = [r for r in scores if not (r["model_id"] == "derivative_scaled" and r["fold"] == 6)]
        with self.assertRaises(ValueError):
            paired_summaries(scores, rows)

    def test_paired_policies_reject_missing_candidate_fold_instead_of_shortening_average(self):
        scores, rows = paired_fixture()
        rows = [r for r in rows if not (r["candidate_id"] == "perp_flow_raw_point" and r["fold"] == 6)]
        with self.assertRaises(ValueError):
            paired_summaries(scores, rows)

    def test_paired_zero_reference_loss_retains_difference_but_no_relative_ratio(self):
        scores, rows = paired_fixture()
        scores = copy.deepcopy(scores)
        for score in scores:
            if score["model_id"].startswith("technical_"):
                for key in ("return_mse", "qlike", "variance_mse", "rms_mse"):
                    score[key] = 0.
        result = paired_summaries(scores, rows)["perp_flow_raw_vs_technical_raw"]["forecast"]["return_mse"]
        self.assertEqual(result["mean_difference"], 3.)
        self.assertEqual(result["improved_quarters"], 0)
        self.assertIsNone(result["relative_loss_reduction"])


if __name__ == "__main__":
    unittest.main()
