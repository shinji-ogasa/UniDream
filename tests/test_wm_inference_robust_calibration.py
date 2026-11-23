import math
import unittest
from unittest.mock import patch

import numpy as np

from unidream.experiments import wm_inference_robust_calibration as calibration


class NullGeneratorTests(unittest.TestCase):
    def test_iid_draws_are_not_demeaned_or_standardized(self):
        seed, n, burn = 17, 37, 12
        expected = np.random.default_rng(seed).standard_normal(n + burn)[burn:]
        actual = calibration.simulate_null("iid_normal", n, np.random.default_rng(seed), burnin=burn)
        np.testing.assert_array_equal(actual, expected)
        self.assertNotEqual(float(actual.mean()), 0.)
        self.assertNotEqual(float(actual.std()), 1.)

    def test_AR_recursion_preserves_theoretical_unit_variance(self):
        phi, n = .3, 20
        shocks = np.random.default_rng(21).standard_normal(n)
        expected = []
        previous = 0.
        for shock in shocks:
            previous = phi * previous + math.sqrt(1 - phi ** 2) * shock
            expected.append(previous)
        actual = calibration.simulate_null("ar1", n, np.random.default_rng(21), burnin=0, ar_phi=phi)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_GARCH_recursion_and_AR_GARCH_innovation_scaling(self):
        n, alpha, beta, phi = 30, .05, .9, .3
        shocks = np.random.default_rng(31).standard_normal(n)
        expected, expected_ar = [], []
        previous_error, variance, previous_ar = 0., 1., 0.
        for shock in shocks:
            variance = 1 - alpha - beta + alpha * previous_error ** 2 + beta * variance
            self.assertGreater(variance, 0.)
            error = math.sqrt(variance) * shock
            expected.append(error)
            previous_ar = phi * previous_ar + math.sqrt(1 - phi ** 2) * error
            expected_ar.append(previous_ar)
            previous_error = error
        garch = calibration.simulate_null("garch", n, np.random.default_rng(31), burnin=0)
        ar_garch = calibration.simulate_null("ar1_garch", n, np.random.default_rng(31), burnin=0)
        np.testing.assert_allclose(garch, expected, rtol=0, atol=0)
        np.testing.assert_allclose(ar_garch, expected_ar, rtol=0, atol=0)

    def test_stationary_long_paths_have_expected_variance_and_AR_dependence(self):
        for number, scenario in enumerate(calibration.SCENARIOS):
            values = calibration.simulate_null(scenario, 100000, np.random.default_rng(100 + number))
            self.assertLess(abs(float(values.mean())), .025)
            self.assertAlmostEqual(float(values.var()), 1., delta=.055)
            lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]
            self.assertAlmostEqual(float(lag1), .3 if scenario in ("ar1", "ar1_garch") else 0., delta=.025)

    def test_nonstationary_parameters_and_unknown_scenarios_are_rejected(self):
        for scenario, kwargs in (("unknown", {}), ("garch", {"garch_alpha": .2, "garch_beta": .9}),
                                 ("ar1", {"ar_phi": 1.})):
            with self.assertRaises(ValueError):
                calibration.simulate_null(scenario, 10, np.random.default_rng(1), **kwargs)


class CalibrationAuditTests(unittest.TestCase):
    def test_exact_legacy_edge_weights_and_bias_are_reported(self):
        audit = calibration.legacy_inclusion_audit()
        np.testing.assert_array_equal(audit["legacy_expected_multiplicity"], [.5, 1, 1.5, 1.75, 1.5, 1.25, 1, .75, .5, .25])
        self.assertEqual(audit["input_mean"], 4.5)
        self.assertEqual(audit["legacy_expected_mean"], 3.9)
        self.assertAlmostEqual(audit["legacy_expected_mean_minus_point"], -.6)
        self.assertEqual(sum(audit["legacy_expected_multiplicity"]), 10.)
        self.assertTrue(audit["synthetic_only"])

    def test_wilson_intervals_remain_valid_at_both_endpoints(self):
        low0, high0 = calibration.wilson_interval(0, 300)
        low1, high1 = calibration.wilson_interval(300, 300)
        self.assertEqual(low0, 0.)
        self.assertGreater(high0, 0.)
        self.assertLess(low1, 1.)
        self.assertEqual(high1, 1.)
        self.assertAlmostEqual(high0, 1 - low1)
        for success in (0, 15, 300):
            lo, hi = calibration.wilson_interval(success, 300)
            self.assertLessEqual(lo, success / 300)
            self.assertGreaterEqual(hi, success / 300)

    def test_bootstrap_CI_conventions_and_tail_count_are_explicit(self):
        settings = calibration._settings({"calibration": {"inner_replicates": 3}})
        fixture = {"point": np.array([2.]), "market_distribution": np.array([[1.], [2.], [5.]]),
                   "market_null_distribution": np.array([[-.5], [0.], [1.]])}
        with patch("unidream.experiments.wm_inference_robust_bootstrap.bootstrap_distributions", return_value=fixture):
            legacy = calibration._bootstrap_result(np.arange(10.), "legacy_non_circular", settings, 1)
            circular = calibration._bootstrap_result(np.arange(10.), "circular", settings, 1)
        np.testing.assert_allclose(legacy[:2], np.quantile([1, 2, 5], [.025, .975]))
        np.testing.assert_allclose(circular[:2], 2 - np.quantile([-.5, 0, 1], [.975, .025]))
        self.assertEqual(legacy[2], .5)  # (one extreme + one)/(three + one)
        self.assertEqual(circular[2], .25)
        self.assertAlmostEqual(legacy[3], 2 / 3)

    def test_real_kernel_integration_has_fixed_grid_and_reproducible_outputs(self):
        cfg = {"calibration": {"outer_replicates": 3, "inner_replicates": 39,
                               "days": 40, "burnin": 20, "block_days": 7}}
        first = calibration.run_calibration(cfg)
        second = calibration.run_calibration(cfg)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(first), 16)
        self.assertEqual(set(zip(first.scenario, first.method)),
                         {(scenario, method) for scenario in calibration.SCENARIOS for method in calibration.METHODS})
        self.assertTrue(first.no_sample_demeaning_or_normalization.all())
        self.assertTrue(first.true_mean.eq(0).all())
        self.assertTrue(first.n_market_series.eq(1).all())
        self.assertTrue(first.n_training_seeds.eq(1).all())
        self.assertTrue(first.rejection_count.between(0, 3).all())
        self.assertTrue(first.coverage_count.between(0, 3).all())
        np.testing.assert_allclose(first.empirical_type1_error, first.rejection_count / 3)
        np.testing.assert_allclose(first.empirical_coverage_95, first.coverage_count / 3)
        self.assertTrue((first.type1_mc_ci_low <= first.empirical_type1_error).all())
        self.assertTrue((first.type1_mc_ci_high >= first.empirical_type1_error).all())
        self.assertEqual(set(first[first.method.eq("circular")].ci_method), {"null_residual_basic"})
        self.assertEqual(set(first[first.method.eq("legacy_non_circular")].ci_method), {"historical_percentile"})

    def test_invalid_run_settings_are_rejected_before_inference(self):
        for settings in ({"outer_replicates": 1}, {"inner_replicates": 2.5}, {"ar_phi": 1.},
                         {"garch_alpha": .2, "garch_beta": .9}, {"block_days": 540},
                         {"segment_layouts": []}, {"segment_layouts": ["single", "single"]},
                         {"segment_layouts": ["unknown"]}, {"days": 40, "segment_layouts": ["six_equal"]}):
            with self.assertRaises(ValueError):
                calibration._settings({"calibration": settings})

    def test_two_layouts_use_same_paths_and_six_fixed_segments(self):
        cfg = {"calibration": {"outer_replicates": 2, "inner_replicates": 29,
                               "days": 60, "burnin": 20, "block_days": 7,
                               "segment_layouts": ["single", "six_equal"]}}
        captured = []
        original = calibration._bootstrap_result

        def capture(values, method, settings, seed, *, segment_ids=None):
            captured.append((values.copy(), segment_ids.copy()))
            return original(values, method, settings, seed, segment_ids=segment_ids)

        with patch.object(calibration, "_bootstrap_result", side_effect=capture):
            result = calibration.run_calibration(cfg)
        self.assertEqual(len(result), 32)
        self.assertEqual(set(result.layout), {"single", "six_equal"})
        self.assertTrue(result[result.layout.eq("six_equal")].n_segments.eq(6).all())
        self.assertTrue(result[result.layout.eq("single")].n_segments.eq(1).all())
        for offset in range(0, len(captured), 4):
            for values, _ in captured[offset:offset + 4]:
                np.testing.assert_array_equal(values, captured[offset][0])
            np.testing.assert_array_equal(captured[offset][1], np.zeros(60, dtype=int))
            np.testing.assert_array_equal(captured[offset + 2][1], np.arange(60) * 6 // 60)
        for _, pair in result.groupby(["scenario", "method"]):
            self.assertEqual(pair.mean_point_estimate_across_simulations.nunique(), 1)
        # IID t deliberately ignores the synthetic segmentation. Its identical
        # results are another check that layouts did not get different data.
        iid = result[result.method.eq("iid_t")]
        for _, pair in iid.groupby("scenario"):
            self.assertEqual(pair.rejection_count.nunique(), 1)
            self.assertEqual(pair.mean_interval_width.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
