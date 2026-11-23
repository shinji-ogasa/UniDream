"""Synthetic checks of centering, segment resampling and max-mean inference."""
import unittest

import numpy as np

from unidream.experiments.wm_full_grid_statistics import _block_sums
from unidream.experiments.wm_inference_robust_bootstrap import (
    _weights, bootstrap_distributions, expected_inclusion_weights, infer_daily,
)


class RobustBootstrapTests(unittest.TestCase):
    def test_exact_legacy_expected_inclusion_bias_with_truncated_block(self):
        expected = np.array([.5, 1, 1.5, 1.75, 1.5, 1.25, 1, .75, .5, .25])
        np.testing.assert_allclose(expected_inclusion_weights(10, 7), expected, atol=1e-14)
        self.assertAlmostEqual(expected @ np.arange(10) / 10, 3.9)
        np.testing.assert_equal(expected_inclusion_weights(10, 7, "circular"), 1)
        np.testing.assert_equal(expected_inclusion_weights(10, 7, "stationary"), 1)
        np.testing.assert_equal(expected_inclusion_weights(7, 7), 1)

    def test_non_circular_kernel_exactly_replays_frozen_block_sums(self):
        values = np.random.default_rng(3).normal(size=(19, 4))
        for block in (1, 7, 14):
            seed = 17
            weights = _weights(19, block, 101, np.random.default_rng(seed), "legacy_non_circular")
            reference = _block_sums(values, block, 101, np.random.default_rng(seed))
            np.testing.assert_allclose(np.einsum("bt,tc->bc", weights, values), reference, atol=1e-13)
            np.testing.assert_equal(weights.sum(axis=1), 19)

    def test_circular_uniform_marginal_inclusion_and_stationary_count(self):
        for method in ("circular", "stationary"):
            weights = _weights(10, 7, 30000, np.random.default_rng(11), method)
            np.testing.assert_equal(weights.sum(axis=1), 10)
            np.testing.assert_allclose(weights.mean(axis=0), 1, atol=.025)
            self.assertTrue(np.any(weights[:, -1] > 0))

    def test_actual_h0_centering_equals_subtracting_estimate_with_partial_counts(self):
        rng = np.random.default_rng(4)
        counts = rng.integers(1, 97, 45)
        y = rng.normal(.001, .01, size=(45, 2, 3)) * counts[:, None, None]
        for method in ("legacy_non_circular", "circular", "stationary"):
            result = bootstrap_distributions(y, counts, np.zeros(45), method=method, block_days=7, replicates=199)
            for key in ("market", "hierarchical"):
                np.testing.assert_allclose(result[key + "_null_distribution"],
                    result[key + "_distribution"] - result["point"], atol=1e-16)

    def test_duplicate_identical_seeds_do_not_create_more_market_precision(self):
        rng = np.random.default_rng(21)
        y = rng.normal(0, .01, (45, 2, 1))
        counts = np.full(45, 96)
        one = bootstrap_distributions(y, counts, np.zeros(45), replicates=299)
        three = bootstrap_distributions(np.repeat(y, 3, axis=2), counts, np.zeros(45), replicates=299)
        for key in ("point", "market_distribution", "market_null_distribution", "hierarchical_distribution"):
            np.testing.assert_allclose(one[key], three[key], atol=1e-17)
        self.assertEqual(one["n_unique_market_rows"], three["n_unique_market_rows"])

    def test_hierarchical_shared_center_preserves_seed_mean_dispersion(self):
        y = np.broadcast_to(np.array([-1., 0., 1.])[None, None, :], (40, 1, 3)).copy()
        result = bootstrap_distributions(y, np.full(40, 96), np.zeros(40), replicates=299)
        np.testing.assert_allclose(result["market_null_distribution"], 0, atol=1e-17)
        self.assertGreater(result["hierarchical_null_distribution"].std(), .001)
        np.testing.assert_allclose(result["hierarchical_null_distribution"], result["hierarchical_distribution"])

    def test_segments_never_mix_and_short_segments_are_fixed(self):
        counts = np.full(25, 96)
        y = np.r_[np.full(20, .1), np.full(5, 10.)][:, None, None]
        result = bootstrap_distributions(y, counts, np.r_[np.zeros(20), np.ones(5)], replicates=101)
        np.testing.assert_allclose(result["market_distribution"], y.mean() / 96, atol=1e-16)
        self.assertEqual(result["n_fixed_rows"], 5 * 96)
        self.assertEqual(result["n_resampleable_rows"], 20 * 96)
        self.assertEqual(result["n_segments"], 2)
        fixed = bootstrap_distributions(y, counts, np.arange(25), replicates=101)
        self.assertFalse(fixed["inference_available"])
        self.assertEqual(fixed["fixed_row_fraction"], 1)

    def test_all_null_paths_produce_zero_intervals_and_p_one(self):
        cfg = {"condition_names": list("ABCDEFGH"), "statistics": {"methods": ["circular"],
            "block_days": [1, 7, 14, 28, 30], "bootstrap_replicates": 99}}
        result = infer_daily(np.zeros((40, 8, 3)), np.full(40, 96), np.zeros(40), cfg)
        rows, rc = result["inference"], result["reality_check"]
        self.assertEqual(len(rows), 80)
        self.assertEqual(rows.primary_inference.sum(), 8)
        self.assertTrue(rows.ci_low.eq(0).all())
        self.assertTrue(rows.ci_high.eq(0).all())
        self.assertTrue(rows.p_value.eq(1).all())
        self.assertTrue(rc.global_reality_check_p.eq(1).all())
        self.assertFalse(rows.significant_difference.any())

    def test_basic_and_percentile_ci_are_separate_and_rc_is_max_adjusted(self):
        rng = np.random.default_rng(51)
        y = rng.exponential(.02, size=(80, 8, 1)) - .018
        cfg = {"statistics": {"methods": ["circular"], "block_days": [7], "bootstrap_replicates": 999}}
        result = infer_daily(y, np.full(80, 96), np.zeros(80), cfg)
        rows, rc = result["inference"], result["reality_check"]
        self.assertTrue((rc.max_adjusted_one_sided_p >= rc.one_sided_p).all())
        self.assertEqual(rc.global_reality_check_p.nunique(), 1)
        self.assertGreater(np.max(np.abs(rows.ci_low - rows.percentile_ci_low)), 1e-5)
        self.assertLess(rows.null_data_vs_subtract_estimate_max_error.max(), 1e-12)
        self.assertEqual(set(rc.family_size), {8})

    def test_bad_counts_nonconsecutive_days_and_reused_segments_fail_closed(self):
        y = np.zeros((20, 1, 1)); counts = np.full(20, 96)
        with self.assertRaisesRegex(ValueError, "integer counts"):
            bootstrap_distributions(y, counts * 3, np.zeros(20))
        with self.assertRaisesRegex(ValueError, "contiguous run"):
            bootstrap_distributions(y, counts, np.arange(20) % 2)
        with self.assertRaisesRegex(ValueError, "consecutive UTC"):
            bootstrap_distributions(y, counts, np.zeros(20), day_ids=np.arange(20) * 2)
        bootstrap_distributions(y, counts, np.zeros(20), day_ids=np.arange(20))


if __name__ == "__main__":
    unittest.main()
