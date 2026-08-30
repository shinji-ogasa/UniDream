"""Fixture-free tests for the fixed P1 validation forecast registry boundary."""
from __future__ import annotations

import unittest

from unidream.experiments import p1_validation_forecast as forecast


class P1ValidationForecastSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = forecast.authenticate_p1_forecast_contract()

    def test_authenticates_manifest_and_both_exact_registries(self) -> None:
        self.assertEqual(self.contract.manifest_sha256, forecast.REGISTERED_MANIFEST_SHA256)
        self.assertEqual(len(self.contract.registry.trials), 56)
        self.assertEqual(len(self.contract.registry.comparisons), 16)
        self.assertEqual(len(self.contract.specs), 7)

    def test_exact_arm_seed_beta_origin_and_support_schedule(self) -> None:
        synthetic = {
            ("S0", "zero_signal"): 0.0,
            ("S1", "known_high_snr_dgp"): 0.004,
            ("S2-high", "high"): 0.004,
            ("S2-medium", "medium"): 0.001,
            ("S2-low", "low"): 0.00025,
        }
        for key, beta in synthetic.items():
            spec = self.contract.spec(*key)
            self.assertEqual(spec.beta, beta)
            self.assertEqual(spec.seeds, tuple(range(20260830, 20260840)))
            self.assertEqual(spec.fit_origin, 90000)
            self.assertEqual(spec.train_start, 0)
            self.assertEqual(spec.support_range, (90000, 100000))
            self.assertEqual(spec.support_id, "synthetic_validation")
        for arm, beta in (("injected", 0.0005), ("zero_injection_control", 0.0)):
            spec = self.contract.spec("S3", arm)
            self.assertEqual(spec.beta, beta)
            self.assertEqual(spec.seeds, (20260830,))
            self.assertEqual(spec.fit_origin, 104528)
            self.assertEqual(spec.train_start, 52492)
            self.assertEqual(spec.support_range, (104528, 139568))
            self.assertEqual(spec.support_id, "s3_validation")

    def test_production_grid_is_complete_and_outer_is_blocked(self) -> None:
        self.assertEqual(self.contract.horizons, (1, 4, 8, 16))
        self.assertEqual(
            self.contract.model_task_keys,
            (
                ("zero_return", "continuous"),
                ("zero_return", "binary"),
                ("persistence_last_observed", "continuous"),
                ("persistence_last_observed", "binary"),
                ("ridge", "continuous"),
                ("logistic", "binary"),
            ),
        )
        with self.assertRaises(forecast.P1ForecastOuterBlocked):
            forecast.execute_p1_outer_report()


if __name__ == "__main__":
    unittest.main()
