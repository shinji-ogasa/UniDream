import unittest

import numpy as np

from unidream.experiments.wm_tail_transfer_statistics import (
    DAY_NS, holm_adjust, infer_comparisons, _sampling_plan,
)


def config(**overrides):
    settings = dict(bootstrap_replicates=399, primary_block_days=7,
                    sensitivity_block_days=[14, 30], hac_lags=[0, 6, 13, 29],
                    seed=71, min_observed_days=14, min_observations=20)
    settings.update(overrides)
    return dict(statistics=settings, accounting={"annual_bars": 35040})


def comparison(values, *, times=None, valid=None, family="forecast_tasks", kind="forecast"):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    return dict(timestamps=np.arange(len(values), dtype=np.int64) * DAY_NS if times is None else times,
                values=values, valid=np.ones_like(values, dtype=bool) if valid is None else valid,
                family=family, kind=kind)


class TailStatisticsTest(unittest.TestCase):
    def test_holm_known_values_and_invalid_family_retention(self):
        np.testing.assert_allclose(holm_adjust([.01, .04, .03, np.nan]), [.04, .09, .09, 1.])
        rng = np.random.default_rng(9)
        valid = comparison(rng.normal(.6, .1, 90))
        dead = comparison(np.r_[rng.normal(size=89), -np.inf])
        constant = comparison(np.ones(90))
        short = comparison(rng.normal(size=5))
        result = infer_comparisons(dict(valid=valid, bankrupt=dead, constant=constant, short=short), config())
        table = result["primaryDF"].set_index("comparison")
        self.assertEqual(len(table), 4)
        for name in ("bankrupt", "constant", "short"):
            self.assertEqual(table.loc[name, "p_value"], 1.)
            self.assertFalse(table.loc[name, "available"])
            self.assertTrue(np.isnan(table.loc[name, "ci_low"]))
        self.assertTrue((table.family_size == 4).all())
        self.assertTrue((table.global_size == 4).all())
        self.assertEqual(result["metadata"]["family_sizes"], {"forecast_tasks": 4})

    def test_centered_null_is_translation_equivariant_not_uncentered_p_half(self):
        x = np.random.default_rng(8).normal(size=180)
        x -= x.mean()
        result = infer_comparisons({"zero": comparison(x), "shifted": comparison(x + 2.)}, config())
        primary = result["primaryDF"].set_index("comparison")
        self.assertGreater(primary.loc["zero", "p_value"], .9)
        self.assertLess(primary.loc["shifted", "p_value"], .01)
        for col in ("estimate", "ci_low", "ci_high"):
            self.assertAlmostEqual(primary.loc["shifted", col] - primary.loc["zero", col], 2., places=12)
        self.assertAlmostEqual(primary.loc["shifted", "standard_error"], primary.loc["zero", "standard_error"], places=12)

    def test_basic_interval_and_ratio_match_explicit_circular_calendar_draws(self):
        days = 45
        rng = np.random.default_rng(42)
        counts = rng.integers(0, 4, days)
        counts[0] = counts[-1] = 2
        day_values = rng.normal(size=days)
        times = np.concatenate([d * DAY_NS + np.arange(counts[d], dtype=np.int64) for d in range(days)])
        values = np.repeat(day_values, counts)
        cfg = config(min_observed_days=5, min_observations=10)
        result = infer_comparisons({"ratio": comparison(values, times=times)}, cfg)
        row = result["primaryDF"].iloc[0]
        plan = _sampling_plan(days, 7, 399, 71)
        indices = (plan["starts"][:, :, None] + np.arange(7)) % days
        indices = indices.reshape(399, -1)
        if plan["remainder"]:
            indices = np.column_stack((indices, (plan["tail"][:, None] + np.arange(plan["remainder"])) % days))
        sampled = (day_values * counts)[indices].sum(1) / counts[indices].sum(1)
        estimate = values.mean()
        lo, hi = np.quantile(sampled, [.025, .975])
        self.assertAlmostEqual(row.ci_low, 2 * estimate - hi, places=12)
        self.assertAlmostEqual(row.ci_high, 2 * estimate - lo, places=12)
        expected_p = (1 + np.count_nonzero(np.abs(sampled - estimate) >= abs(estimate))) / 400
        self.assertAlmostEqual(row.p_value, expected_p)

    def test_dependent_blocks_have_wider_interval_than_iid_days(self):
        rng = np.random.default_rng(11)
        series = np.zeros(500)
        innovations = rng.normal(size=500)
        for i in range(1, 500):
            series[i] = .92 * series[i - 1] + innovations[i]
        result = infer_comparisons({"AR": comparison(series)}, config(sensitivity_block_days=[1, 14, 30]))
        table = result["inferenceDF"]
        table = table.loc[table.estimand == "fixed_seed_mean"].set_index("block_days")
        width = table.ci_high - table.ci_low
        self.assertGreater(width.loc[7], width.loc[1] * 1.7)

    def test_repeated_seeds_do_not_create_independent_markets(self):
        x = np.random.default_rng(29).normal(.1, 1, 120)
        result = infer_comparisons({"one": comparison(x), "ten_identical": comparison(np.repeat(x[:, None], 10, axis=1))}, config())
        primary = result["primaryDF"].set_index("comparison")
        for col in ("estimate", "ci_low", "ci_high", "standard_error", "p_value"):
            self.assertAlmostEqual(primary.loc["one", col], primary.loc["ten_identical", col], places=12)
        self.assertEqual(primary.loc["one", "n_market_observations"], primary.loc["ten_identical", "n_market_observations"])
        self.assertEqual(primary.loc["ten_identical", "n_seeds"], 10)
        hierarchy = result["inferenceDF"].query("comparison == 'ten_identical' and estimand == 'hierarchical_seed_sensitivity' and block_days == 7").iloc[0]
        self.assertAlmostEqual(hierarchy.standard_error, primary.loc["one", "standard_error"], places=12)

    def test_sparse_zero_count_days_remain_on_shared_calendar(self):
        observed_days = np.array([0, 15, 30, 45, 60, 75, 89])
        sparse_values = np.array([.1, -.2, .4, -.1, .3, -.3, .2])
        sparse = comparison(sparse_values, times=observed_days * DAY_NS)
        dense_values = np.full(90, np.nan)
        dense_values[observed_days] = sparse_values
        dense_mask = np.isfinite(dense_values)[:, None]
        dense = comparison(dense_values, valid=dense_mask)
        sparse["event_counts"] = {"crashes": np.int64(3), "non_crashes": 4}
        result = infer_comparisons({"sparse": sparse, "dense_masked": dense}, config(min_observed_days=5, min_observations=5))
        p = result["primaryDF"].set_index("comparison")
        for column in ("estimate", "ci_low", "ci_high", "p_value"):
            self.assertAlmostEqual(p.loc["sparse", column], p.loc["dense_masked", column], places=12)
        detail = result["metadata"]["comparisons"]["sparse"]
        self.assertEqual(detail["n_zero_count_days"], 83)
        self.assertEqual(detail["n_observed_days"], 7)
        self.assertEqual(detail["event_counts"]["crashes"], 3)
        self.assertEqual(result["metadata"]["n_global_calendar_days"], 90)

    def test_hac_ratio_influence_and_policy_annualization(self):
        x = np.random.default_rng(18).normal(.2, 1, 100)
        result = infer_comparisons({"forecast": comparison(x), "policy": comparison(x, kind="policy")}, config())
        hac = result["hacDF"].query("hac_lag_days == 0").set_index("comparison")
        expected_se = np.sqrt(np.square(x - x.mean()).sum()) / len(x)
        self.assertAlmostEqual(hac.loc["forecast", "standard_error"], expected_se, places=12)
        self.assertAlmostEqual(hac.loc["policy", "standard_error"], expected_se * 35040, places=8)
        self.assertAlmostEqual(hac.loc["policy", "p_value"], hac.loc["forecast", "p_value"])
        p = result["primaryDF"].set_index("comparison")
        self.assertAlmostEqual(p.loc["policy", "estimate"], x.mean() * 35040, places=8)

    def test_input_order_preserves_shared_draws_and_seed_cohort_mismatch_fails(self):
        rng = np.random.default_rng(39)
        a, b = comparison(rng.normal(size=(90, 3))), comparison(rng.normal(size=(90, 3)))
        first = infer_comparisons({"a": a, "b": b}, config())
        second = infer_comparisons({"b": b, "a": a}, config())
        self.assertEqual(first["metadata"]["shared_sampling_plan_sha256"], second["metadata"]["shared_sampling_plan_sha256"])
        np.testing.assert_array_equal(first["primaryDF"].p_value, second["primaryDF"].p_value)
        b["valid"][0, 1] = False
        with self.assertRaisesRegex(ValueError, "same timestamp cohort"):
            infer_comparisons({"a": a, "b": b}, config())

    def test_too_short_for_large_blocks_is_unavailable_and_family_retained(self):
        values = np.random.default_rng(6).normal(size=20)
        result = infer_comparisons({"short": comparison(values)}, config())
        rows = result["inferenceDF"].query("block_days == 30")
        self.assertTrue((rows.p_value == 1).all())
        self.assertFalse(rows.available.any())
        self.assertTrue((rows.family_size == 1).all())
        self.assertTrue(rows.reason.str.contains("fewer_than_two_blocks").all())


if __name__ == "__main__":
    unittest.main()
