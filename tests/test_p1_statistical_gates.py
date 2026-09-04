import unittest

import numpy as np

from unidream.experiments import p1_statistical_gates as gates


class P1StatisticalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = gates.load_p1_result_registry()
        cls.ids = [row["comparison_id"] for row in cls.registry.comparisons]

    def test_holm_uses_exact_family_step_down_and_lexical_ties(self):
        values = {comparison_id: 1.0 for comparison_id in self.ids}
        tied = sorted(self.ids[:2])
        values[tied[0]] = 0.001
        values[tied[1]] = 0.001
        family = gates.holm_bonferroni_fixed_family(
            values,
            registry=self.registry,
        )
        self.assertEqual([row.comparison_id for row in family.rows[:2]], tied)
        self.assertTrue(family.rows[0].rejected)
        self.assertTrue(family.rows[1].rejected)
        self.assertFalse(family.rows[2].rejected)
        self.assertAlmostEqual(family.rows[0].adjusted_p, 0.016)
        self.assertAlmostEqual(family.rows[1].adjusted_p, 0.016)

        missing = dict(values)
        missing.pop(self.ids[-1])
        with self.assertRaisesRegex(gates.P1StatisticalGateError, "not exact"):
            gates.holm_bonferroni_fixed_family(missing, registry=self.registry)

    def test_wilson_interval_is_fixed_and_rejects_na_counts(self):
        result = gates.wilson_score_interval(90, 100)
        self.assertEqual(result.point, 0.9)
        self.assertAlmostEqual(result.lower, 0.8256343384950865)
        self.assertAlmostEqual(result.upper, 0.9447708629393249)
        with self.assertRaises(gates.P1StatisticalGateError):
            gates.wilson_score_interval(0, 0)
        with self.assertRaises(gates.P1StatisticalGateError):
            gates.wilson_score_interval(True, 1)

    def test_s0_uses_holm_rank_alpha_at_all_three_lengths(self):
        raw_p = {comparison_id: 1.0 for comparison_id in self.ids}
        family = gates.holm_bonferroni_fixed_family(
            raw_p,
            registry=self.registry,
        )
        comparison_id = "S0__ridge__utility_vs_hold__cost_on"
        negative = {length: np.full(2000, -0.01) for length in (8, 16, 32)}
        passed = gates.evaluate_s0_safety_bounds(
            comparison_id,
            negative,
            holm=family,
        )
        self.assertTrue(passed["passed"])
        positive = dict(negative)
        positive[16] = np.full(2000, 0.01)
        failed = gates.evaluate_s0_safety_bounds(
            comparison_id,
            positive,
            holm=family,
        )
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
