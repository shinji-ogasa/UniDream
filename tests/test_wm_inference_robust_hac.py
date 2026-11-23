"""Independent reference and prospective-planning tests on synthetic series."""
import json
import unittest

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.stattools import acf

from unidream.experiments.wm_inference_robust_hac import (
    hac_mean, normal_power, power_scenarios, serial_diagnostics,
)


class RobustHACTests(unittest.TestCase):
    def series(self, n=180):
        rng = np.random.default_rng(3124)
        shock = rng.normal(size=n)
        values = np.zeros(n)
        for i in range(1, n):
            values[i] = .4*values[i-1] + shock[i]
        return values

    def test_matches_independent_statsmodels_intercept_hac_equal_counts(self):
        values = self.series()
        counts = np.full(len(values), 96)
        sums = np.column_stack((values-.2, values, values+.3)) * counts[:, None]
        mean = sums.mean(1)/counts
        fit = sm.OLS(mean, np.ones((len(mean), 1))).fit()
        for lag in (0, 6, 13, 27, 29):
            with self.subTest(lag=lag):
                result = hac_mean(sums, counts, np.zeros(len(mean), int), np.arange(len(mean)), lag=lag, annual_bars=1)
                self.assertAlmostEqual(result["estimate"], fit.params[0], places=13)
                self.assertAlmostEqual(result["market_variance"], cov_hac(fit, nlags=lag, use_correction=True)[0, 0], places=13)
                self.assertAlmostEqual(result["standard_error"], np.sqrt(cov_hac(fit, nlags=lag)[0, 0]), places=13)

    def test_uneven_counts_match_independent_weighted_intercept_hac(self):
        y = self.series(120)
        counts = np.resize([96, 13, 70, 40, 95], len(y))
        sums = np.column_stack((y*.9, y*1.1, y)) * counts[:, None]
        fit = sm.WLS(sums.mean(1)/counts, np.ones((len(y), 1)), weights=counts).fit()
        for corrected in (True, False):
            result = hac_mean(sums, counts, np.zeros(len(y), int), np.arange(len(y)), lag=6,
                annual_bars=35040, finite_sample=corrected)
            self.assertAlmostEqual(result["estimate"], fit.params[0]*35040, places=9)
            self.assertAlmostEqual(result["market_variance"]/35040**2,
                cov_hac(fit, nlags=6, use_correction=corrected)[0, 0], places=13)
            self.assertEqual(result["n_market_bars"], int(counts.sum()))

    def test_mean_is_ratio_of_sums_not_unweighted_daily_average(self):
        sums = np.array([[1., 2., 3.], [1., 2., 3.], [2., 3., 4.]])
        counts = np.array([96, 12, 48])
        result = hac_mean(sums, counts, [0, 0, 0], [0, 1, 2], lag=0, annual_bars=1)
        self.assertAlmostEqual(result["estimate"], sums.mean(1).sum()/counts.sum())
        self.assertNotAlmostEqual(result["estimate"], (sums.mean(1)/counts).mean())

    def test_no_covariance_pairs_cross_segment_boundary(self):
        values = np.array([-3., -1., 1., 3.])
        result = hac_mean(values[:, None], np.ones(4), [0, 0, 1, 1], [0, 1, 2, 3], lag=1, annual_bars=1)
        self.assertEqual(result["lag_pairs"][0]["n_pairs"], 2)
        self.assertAlmostEqual(result["score_long_run_sum"], 26.)
        self.assertAlmostEqual(result["market_variance"], 26./16*4/3)

    def test_lag_is_exact_calendar_gap_not_adjacent_observed_rows(self):
        values = np.array([-3., -1., 1., 3.])
        result = hac_mean(values[:, None], np.ones(4), np.zeros(4, int), [0, 2, 3, 7], lag=2, annual_bars=1)
        self.assertEqual([r["n_pairs"] for r in result["lag_pairs"]], [1, 1])
        self.assertAlmostEqual(result["score_long_run_sum"], 20+2/3)

    def test_replicating_seeds_never_multiplies_market_sample_or_precision(self):
        y = self.series(80)
        arguments = (np.ones(80), np.zeros(80, int), np.arange(80))
        one = hac_mean(y[:, None], *arguments, annual_bars=1)
        three = hac_mean(np.repeat(y[:, None], 3, axis=1), *arguments, annual_bars=1)
        self.assertEqual(three["n_market_bars"], 80)
        self.assertEqual(three["n_seeds"], 3)
        self.assertAlmostEqual(one["standard_error"], three["standard_error"], places=14)
        self.assertEqual(three["approximate_seed_variance_floor"], 0.)

    def test_seed_dispersion_is_separate_approximate_floor(self):
        y = self.series(100)*.001
        sums = np.column_stack((y-.001, y, y+.001))
        result = hac_mean(sums, np.ones(100), np.zeros(100, int), np.arange(100), annual_bars=1)
        floor = np.var([r["estimate"] for r in result["per_seed"]], ddof=1)/3
        self.assertAlmostEqual(result["approximate_seed_variance_floor"], floor)
        self.assertAlmostEqual(result["approximate_seed_augmented"]["standard_error"]**2,
            result["market_variance"]+floor)
        self.assertFalse(result["approximate_seed_augmented"]["primary"])
        self.assertTrue(result["approximate_seed_augmented"]["approximate"])

    def test_normal_p_and_confidence_match_independent_formula(self):
        y = self.series(100)+.15
        result = hac_mean(y, np.ones(100), np.zeros(100, int), np.arange(100), annual_bars=1)
        z = result["estimate"]/result["standard_error"]
        self.assertAlmostEqual(result["p_value"], 2*norm.sf(abs(z)))
        self.assertAlmostEqual(result["ci_low"], result["estimate"]-norm.ppf(.975)*result["standard_error"])

    def test_zero_variance_fails_closed_without_certain_significance(self):
        result = hac_mean(np.ones((12, 3)), np.ones(12), np.zeros(12, int), np.arange(12), annual_bars=1)
        self.assertFalse(result["inference_available"])
        self.assertIsNone(result["p_value"])
        self.assertFalse(power_scenarios(result)["available"])

    def test_invalid_counts_duplicate_day_and_missing_day_identifiers_reject(self):
        y = np.arange(4.)[:, None]
        bad = (([1, 0, 1, 1], [0, 1, 2, 3]), ([1, 1.5, 1, 1], [0, 1, 2, 3]),
            ([1]*4, [0, 1, 1, 3]), ([1]*4, [0., 1., 2., 3.]))
        for counts, days in bad:
            with self.subTest(counts=counts, days=days), self.assertRaises(ValueError):
                hac_mean(y, counts, [0]*4, days)
        for lag in (-1, .5, True):
            with self.subTest(lag=lag), self.assertRaises(ValueError):
                hac_mean(y, [1]*4, [0]*4, [0, 1, 2, 3], lag=lag)

    def test_same_utc_day_allowed_across_two_distinct_segments(self):
        result = hac_mean(np.arange(4.)[:, None], [20, 15, 20, 15], [0, 0, 1, 1], [0, 1, 1, 2], annual_bars=1)
        self.assertEqual(result["n_daily_rows"], 4)
        self.assertEqual(result["n_unique_calendar_days"], 3)
        self.assertEqual(result["n_segments"], 2)
        json.dumps(result, allow_nan=False)

    def test_normal_power_is_exact_two_sided_and_symmetric(self):
        self.assertAlmostEqual(normal_power(0., .1, .05), .05, places=13)
        self.assertAlmostEqual(normal_power(.2, .1), norm.sf(norm.ppf(.975)-2)+norm.cdf(-norm.ppf(.975)-2))
        self.assertAlmostEqual(normal_power(-.2, .1), normal_power(.2, .1))

    def result(self):
        return {"inference_available": True, "market_variance": .04, "eligible_years": 2.,
            "estimate": .1, "approximate_seed_variance_floor": None}

    def test_mde_sqrt_time_and_required_years_agree_with_exact_power(self):
        result = self.result()
        planning = power_scenarios(result, effects=(.05,), years=(1, 2, 4), alphas=(.05,), powers=(.8,), include_observed=False)
        mde = planning["mde"]
        self.assertAlmostEqual(mde[0]["mde_annual_log_growth"]/mde[2]["mde_annual_log_growth"], 2.)
        required = planning["required_years"][0]["required_eligible_years"]
        se = np.sqrt(.04*2/required)
        self.assertAlmostEqual(normal_power(.05, se, .05), .8, places=10)
        self.assertEqual(planning["current_eligible_years"], 2.)
        json.dumps(planning, allow_nan=False)

    def test_bonferroni_planning_is_more_demanding_and_observed_scenario_caveated(self):
        planning = power_scenarios(self.result(), effects=(.05,), years=(2,), alphas=(.05, .05/8), powers=(.9,))
        mde = planning["mde"]
        self.assertGreater(mde[1]["mde_annual_log_growth"], mde[0]["mde_annual_log_growth"])
        observed = [r for r in planning["power"] if r["effect_scenario"].startswith("observed")]
        self.assertTrue(observed)
        self.assertTrue(all(not r["observed_scenario_is_evidence"] for r in observed))
        self.assertIn("optimistic", planning["observed_effect_caveat"])

    def test_fixed_seed_floor_is_not_divided_by_more_market_years(self):
        result = self.result()
        result["approximate_seed_variance_floor"] = .01
        planning = power_scenarios(result, effects=(.02,), years=(1, 1000000), alphas=(.05,), powers=(.8,), include_observed=False)
        floor_mde = [r for r in planning["mde"] if r["mode"].startswith("approximate")][-1]
        self.assertGreater(floor_mde["mde_annual_log_growth"], .28)
        self.assertGreater(floor_mde["fixed_seed_floor_mde_limit"], .28)
        required = [r for r in planning["required_years"] if r["mode"].startswith("approximate")][0]
        self.assertTrue(required["unattainable_with_fixed_seed_floor"])
        self.assertIsNone(required["required_eligible_years"])
        self.assertIsNotNone(planning["required_years"][0]["required_eligible_years"])

    def test_acf_matches_independent_statsmodels_without_gaps(self):
        y = self.series(100)
        diagnostics = serial_diagnostics(y[:, None], np.ones(100), np.zeros(100, int), np.arange(100))
        reference = acf(y, nlags=30, fft=False, adjusted=False)
        for row in diagnostics["acf"]:
            if row["series"] == "diff":
                self.assertAlmostEqual(row["acf"], reference[row["lag_days"]], places=13)
        self.assertFalse(diagnostics["inferential_claim"])

    def test_acf_segments_and_fold_demeaning_are_explicit(self):
        y = np.r_[np.array([-.1, .1, -.1, .1]), np.array([.9, 1.1, .9, 1.1])]
        diagnostics = serial_diagnostics(y[:, None], np.ones(8), [0]*4+[1]*4, np.arange(8),
            fold_ids=[0]*4+[1]*4, lags=(1, 7))
        diff = [r for r in diagnostics["acf"] if r["series"] == "diff"]
        self.assertEqual(diff[0]["n_pairs"], 6)
        self.assertEqual(diff[1]["n_pairs"], 0)
        self.assertIsNone(diff[1]["acf"])
        centered = [r for r in diagnostics["acf"] if r["series"] == "within_fold_diff" and r["lag_days"] == 1][0]
        self.assertAlmostEqual(centered["acf"], -.75)
        self.assertEqual([r["mean_daily_per_bar_log_excess"] for r in diagnostics["folds"]], [0., 1.])
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
