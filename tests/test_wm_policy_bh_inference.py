"""Independent paired B&H inference checks; all paths are synthetic."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments.wm_full_grid_statistics import _block_sums
from unidream.experiments.wm_policy_bh_inference import infer_bh


def fixture(days=40, seeds=(7, 19, 42), effect=None, gap=False):
    time = pd.date_range("2025-01-01", periods=days * 96, freq="15min", tz="UTC")
    if gap:
        time = time + pd.to_timedelta((np.arange(len(time)) >= len(time) // 2) * 86400, unit="s")
    n = len(time)
    returns = np.random.default_rng(19).normal(.00003, .001, n)
    bh = pd.DataFrame({"fold": 0, "decision_id": np.arange(n), "timestamp": time,
                       "bar_log_return": returns, "net_log_return": returns, "baseline": "bh"})
    frames = []
    for seed in seeds:
        values = np.zeros(n) if effect is None else np.broadcast_to(effect(seed, np.arange(n) // 96), (n,))
        for stage in ("bc", "rl"):
            for arm in "ABCD":
                part = bh.drop(columns="baseline").copy()
                part["net_log_return"] += values
                frames.append(part.assign(seed=seed, arm=arm, stage=stage))
    cfg = {"folds": [0], "seeds": list(seeds), "statistics": {
        "bootstrap_replicates": 199, "block_days": [1, 7, 28], "seed": 17}}
    return pd.concat(frames, ignore_index=True), bh, cfg


class BHPairedInferenceTests(unittest.TestCase):
    def test_identical_bh_is_exact_zero_and_never_significant(self):
        policy, bh, cfg = fixture()
        rows = infer_bh(policy, bh, cfg)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows.primary_inference.sum(), 8)
        self.assertTrue(rows.estimate.eq(0).all())
        self.assertTrue(rows.ci_low.eq(0).all())
        self.assertTrue(rows.ci_high.eq(0).all())
        self.assertTrue(rows.p_value.eq(1).all())
        self.assertTrue(rows.loc[rows.primary_inference, "p_holm_8"].eq(1).all())
        self.assertFalse(rows.significant_difference.any())
        self.assertEqual(set(rows.n_unique_market_rows), {40 * 96})

    def test_known_constant_log_improvement_and_underperformance(self):
        for sign in (1, -1):
            policy, bh, cfg = fixture(effect=lambda seed, day: sign * .00001)
            cfg["statistics"]["bootstrap_replicates"] = 999
            rows = infer_bh(policy, bh, cfg)
            primary = rows.loc[rows.primary_inference]
            np.testing.assert_allclose(rows.estimate, sign * .3504, atol=1e-14)
            np.testing.assert_allclose(rows.ci_low, rows.estimate, atol=1e-13)
            np.testing.assert_allclose(rows.ci_high, rows.estimate, atol=1e-13)
            self.assertTrue(primary.significant_difference.all())
            self.assertTrue((primary.significant_improvement if sign > 0 else primary.significant_underperformance).all())
            self.assertTrue(rows.loc[~rows.primary_inference, "p_holm_8"].isna().all())
            self.assertFalse(rows.loc[~rows.primary_inference, "significant_difference"].any())

    def test_mismatched_rows_returns_missing_conditions_and_duplicates_rejected(self):
        policy, bh, cfg = fixture(days=8)
        variants = []
        wrong = policy.copy(); wrong.loc[0, "bar_log_return"] += .01; variants.append(wrong)
        variants.append(policy.iloc[1:])
        variants.append(pd.concat([policy, policy.iloc[:1]], ignore_index=True))
        variants.append(policy.loc[~(policy.arm.eq("D") & policy.stage.eq("rl"))])
        for wrong in variants:
            with self.subTest(rows=len(wrong)), self.assertRaises(ValueError):
                infer_bh(wrong, bh, cfg)
        bad_bh = bh.copy(); bad_bh.loc[0, "net_log_return"] += .001
        with self.assertRaisesRegex(ValueError, "unit-exposure"):
            infer_bh(policy, bad_bh, cfg)

    def test_identical_seed_duplicates_do_not_multiply_market_precision(self):
        effect = lambda seed, day: .00001 * np.sin(day * .7)
        policy, bh, cfg = fixture(effect=effect)
        multiple = infer_bh(policy, bh, cfg)
        single = infer_bh(policy.loc[policy.seed.eq(7)], bh, {**cfg, "seeds": [7]})
        for name in ("estimate", "ci_low", "ci_high", "p_value"):
            np.testing.assert_allclose(multiple[name], single[name], atol=1e-13, equal_nan=True)
        self.assertEqual(set(multiple.n_unique_market_rows), set(single.n_unique_market_rows))

    def test_seed_uncertainty_is_retained_instead_of_averaging_seeds_before_resampling(self):
        policy, bh, cfg = fixture(effect=lambda seed, day: {7: -.0001, 19: 0., 42: .0001}[seed])
        rows = infer_bh(policy, bh, cfg)
        primary = rows.loc[rows.primary_inference]
        self.assertTrue((primary.ci_high - primary.ci_low > 1.).all())
        self.assertFalse(primary.significant_difference.any())

    def test_short_segments_remain_fixed_without_iid_or_shorter_block_fallback(self):
        policy, bh, cfg = fixture(days=12, gap=True, effect=lambda seed, day: day * .000001)
        rows = infer_bh(policy, bh, cfg)
        primary = rows.loc[rows.primary_inference]
        self.assertFalse(primary.inference_available.any())
        self.assertTrue(primary.p_value.isna().all())
        self.assertTrue(primary.ci_low.isna().all())
        self.assertTrue(primary.p_holm_8.eq(1).all())
        self.assertTrue(primary.n_fixed_rows.eq(len(bh)).all())
        self.assertEqual(set(rows.n_reset_segments), {2})

    def test_same_market_block_draws_are_paired_across_seeds_exactly(self):
        effect = lambda seed, day: (seed - 19) * np.sin(day / 3) * .000001
        policy, bh, cfg = fixture(days=40, effect=effect)
        rows = infer_bh(policy, bh, cfg)
        daily = np.column_stack([96 * effect(seed, np.arange(40)) for seed in cfg["seeds"]]
                                + [np.full(40, 96.)])
        for block in (1, 7, 28):
            rng = np.random.default_rng(np.random.SeedSequence([17, block]))
            draws = _block_sums(daily, block, 199, rng)
            seed_ids = rng.integers(0, 3, size=(199, 3))
            values = np.take_along_axis(draws[:, :3], seed_ids, axis=1).mean(1) / draws[:, 3] * 35040
            row = rows.loc[rows.comparison.eq("bc_A_minus_bh") & rows.block_days.eq(block)].iloc[0]
            np.testing.assert_allclose([row.ci_low, row.ci_high], np.quantile(values, [.025, .975]), atol=1e-13)

    def test_blocks_never_cross_gap_or_fold_and_fixed_support_is_reported(self):
        policy, bh, cfg = fixture(days=40, gap=True)
        policy.loc[policy.decision_id.ge(30 * 96), "fold"] = 1
        bh.loc[bh.decision_id.ge(30 * 96), "fold"] = 1
        cfg["folds"] = [0, 1]
        seen = []
        def observe(values, days, reps, rng):
            seen.append((len(values), values.shape[1], days))
            return _block_sums(values, days, reps, rng)
        with patch("unidream.experiments.wm_policy_bh_inference._block_sums", side_effect=observe):
            rows = infer_bh(policy, bh, cfg)
        self.assertEqual(set(length for length, _, _ in seen), {10, 20})
        self.assertEqual(set(columns for _, columns, _ in seen), {4})
        self.assertEqual(set(rows.n_reset_segments), {3})
        self.assertTrue(rows.loc[rows.block_days.eq(28), "n_fixed_rows"].eq(40 * 96).all())
        self.assertFalse(rows.loc[rows.block_days.eq(28), "inference_available"].any())

    def test_registered_row_count_and_reset_mismatch_fail_closed(self):
        policy, bh, cfg = fixture(days=8)
        cfg["statistics"]["expected_market_rows"] = 1
        with self.assertRaisesRegex(ValueError, "row count"):
            infer_bh(policy, bh, cfg)
        del cfg["statistics"]["expected_market_rows"]
        bh["reset"] = False
        with self.assertRaisesRegex(ValueError, "resets"):
            infer_bh(policy, bh, cfg)


if __name__ == "__main__":
    unittest.main()
