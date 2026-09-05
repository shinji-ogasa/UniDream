import json
import unittest

import numpy as np

from unidream.experiments.oracle_paired_loss_bootstrap import (
    _moving_block_indices,
    paired_quarter_block_bootstrap,
)


class PairedLossBootstrapTests(unittest.TestCase):
    def run_bootstrap(self, folds, comparisons=None, **kwargs):
        options = {"block_lengths": (4,), "primary_block_length": 4,
                   "n_bootstrap": 128, "seed": 7}
        options.update(kwargs)
        return paired_quarter_block_bootstrap(
            folds, comparisons=comparisons or {"a_minus_b": ("a", "b")}, **options)

    def test_equal_quarter_weight_and_paired_gaps(self):
        # Unequal lengths cannot turn this into a pooled-row mean of 0.5.
        first = np.full(8, 3.0)
        first[2] = np.nan
        reference = np.full(8, 2.0)
        reference[5] = np.nan
        result = self.run_bootstrap({
            "short": {"a": first, "b": reference},
            "long": {"a": np.ones(24), "b": np.full(24, 2.0)},
        })
        obs = result["observed"]["a_minus_b"]
        self.assertEqual(obs["folds"]["short"]["grid_rows"], 8)
        self.assertEqual(obs["folds"]["short"]["paired_rows"], 6)
        self.assertEqual(obs["equal_quarter_mean_difference"], 0.0)
        self.assertEqual(result["blocks"]["4"]["comparisons"]["a_minus_b"]
                         ["centered_interval"], [0.0, 0.0])
        json.dumps(result, allow_nan=False)

    def test_same_indices_preserve_pairing_and_reversal(self):
        series = np.repeat(np.array([1., 8., 2., 6.]), 8)
        result = self.run_bootstrap(
            {"q": {"a": series, "b": np.ones(32), "copy": series.copy()}},
            {"forward": ("a", "b"), "reverse": ("b", "a"),
             "identical": ("a", "copy")})
        values = result["blocks"]["4"]["comparisons"]
        np.testing.assert_allclose(values["forward"]["centered_interval"],
                                   -np.array(values["reverse"]["centered_interval"])[::-1],
                                   atol=1e-14, rtol=0)
        self.assertEqual(values["identical"]["centered_interval"], [0., 0.])
        self.assertEqual(values["identical"]["bootstrap_standard_error"], 0.)

    def test_original_blocks_never_wrap_or_compress_missing_slots(self):
        indices = _moving_block_indices(11, 4, 200, np.random.default_rng(9))
        self.assertEqual(indices.shape, (200, 11))
        self.assertTrue((indices >= 0).all() and (indices < 11).all())
        for start in (0, 4, 8):
            self.assertTrue((np.diff(indices[:, start:start + 4], axis=1) == 1).all())
        # Missing scheduled slot 5 travels inside contiguous original blocks.
        values = np.arange(11, dtype=float)
        values[5] = np.nan
        sampled = values[indices]
        self.assertEqual(int(np.isnan(sampled).sum()), int((indices == 5).sum()))
        self.assertGreater(int(np.isnan(sampled).sum()), 0)

    def test_seed_and_primary_are_stable_to_order_and_sensitivity_additions(self):
        folds = {"q2": {"a": np.arange(32.), "b": np.ones(32)},
                 "q1": {"a": np.arange(24.), "b": np.zeros(24)}}
        first = self.run_bootstrap(folds)
        second = self.run_bootstrap(dict(reversed(list(folds.items()))), block_lengths=(8, 4))
        self.assertEqual(first["blocks"]["4"], second["blocks"]["4"])

    def test_scalar_replay_of_uncertainty_with_missing_slots(self):
        folds = {"q": {"a": np.array([2., np.nan, 5., 6., 3., 2., 1., 4., 9.]),
                       "b": np.ones(9)}}
        result = self.run_bootstrap(folds)
        # Independently form blocks and scalar paired means from the fixed RNG.
        rng = np.random.default_rng(np.random.SeedSequence([7, 4]))
        starts = rng.integers(0, 6, size=(128, 3))
        draws = []
        for row in starts:
            slots = [slot for start in row for slot in range(start, start + 4)][:9]
            valid = [slot for slot in slots if np.isfinite(folds["q"]["a"][slot])]
            draws.append(sum(folds["q"]["a"][slot] - 1 for slot in valid) / len(valid))
        observed = sum([1., 4., 5., 2., 1., 0., 3., 8.]) / 8
        centered = np.array(draws) - np.mean(draws)
        expected = observed - np.quantile(centered, [.975, .025])
        got = result["blocks"]["4"]["comparisons"]["a_minus_b"]
        np.testing.assert_allclose(got["centered_interval"], expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(got["bootstrap_mean_minus_observed"], np.mean(draws) - observed)

    def test_invalid_support_or_parameters_fail_closed(self):
        valid = {"q": {"a": np.ones(8), "b": np.zeros(8)}}
        for kwargs in ({"block_lengths": (9,), "primary_block_length": 9},
                       {"block_lengths": (4, 4)}, {"n_bootstrap": 1},
                       {"seed": -1}, {"confidence": 1}, {"n_bootstrap": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_bootstrap(valid, **kwargs)
        for bad in ({"q": {"a": np.ones(8), "b": np.zeros(7)}},
                    {"q": {"a": np.full(8, np.nan), "b": np.zeros(8)}},
                    {"q": {"a": np.full(8, np.inf), "b": np.zeros(8)}}):
            with self.assertRaises(ValueError):
                self.run_bootstrap(bad)

    def test_empty_quarter_replicates_are_counted_and_never_reweighted(self):
        sparse = np.full(8, np.nan)
        sparse[0] = 1.
        result = self.run_bootstrap({
            "sparse": {"a": sparse, "b": np.zeros(8)},
            "dense": {"a": np.zeros(8), "b": np.ones(8)},
        })
        block = result["blocks"]["4"]
        value = block["comparisons"]["a_minus_b"]
        invalid = value["invalid_bootstrap_replicates"]
        self.assertGreater(invalid, 0)
        self.assertLess(invalid, 128)
        self.assertEqual(value["valid_bootstrap_replicates"] + invalid, 128)
        self.assertEqual(block["fold_grids"]["sparse"]["all_missing_replicates"]["a_minus_b"], invalid)
        self.assertEqual(block["fold_grids"]["dense"]["all_missing_replicates"]["a_minus_b"], 0)
        self.assertEqual(value["centered_interval"], [0., 0.])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
