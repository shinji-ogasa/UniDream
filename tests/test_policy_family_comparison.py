from __future__ import annotations

import unittest

import numpy as np

from unidream.cli.compare_policy_families import ExecutionSpec, _causal_vol, _execute_targets


class PolicyFamilyComparisonTest(unittest.TestCase):
    def test_causal_vol_is_unchanged_before_future_perturbation(self) -> None:
        returns = np.linspace(-0.01, 0.01, 32)
        perturbed = returns.copy()
        perturbed[20:] += 1.0
        np.testing.assert_allclose(_causal_vol(returns, 8)[:21], _causal_vol(perturbed, 8)[:21])

    def test_execution_respects_position_and_step_limits(self) -> None:
        positions = _execute_targets(
            np.asarray([2.0, 2.0, -1.0, -1.0]),
            benchmark=1.0,
            min_position=0.5,
            max_position=1.12,
            spec=ExecutionSpec(blend=1.0, max_step=0.04, min_hold=0),
        )
        self.assertTrue(np.all(positions >= 0.5))
        self.assertTrue(np.all(positions <= 1.12))
        self.assertTrue(np.all(np.abs(np.diff(np.concatenate([[1.0], positions]))) <= 0.040001))


if __name__ == "__main__":
    unittest.main()
