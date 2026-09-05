import unittest
import numpy as np
from unidream.experiments.oracle_risk_controller import risk_targets


class RiskControllerTests(unittest.TestCase):
    def test_equal_future_and_past_risk_preserves_backbone(self):
        base = np.array([.5, 1, 1.12])
        vol = np.array([.5, 1, 2])
        got = risk_targets(base, vol*np.sqrt(24/35040), vol, strength=.5)
        np.testing.assert_allclose(got, base)

    def test_high_risk_reduces_and_missing_risk_uses_declared_fallback(self):
        got = risk_targets([1, 1.12, np.nan, .6], [.1, np.nan, .01, np.nan],
                           [.5, .5, .5, .5], strength=1.)
        self.assertEqual(got[0], .5)
        self.assertEqual(got[1], 1.12)
        self.assertTrue(np.isnan(got[2]))
        self.assertEqual(got[3], .6)

    def test_future_mutation_cannot_change_past_intents(self):
        base = np.ones(20)
        vol = np.full(20, .5)
        prediction = np.full(20, .02)
        before = risk_targets(base, prediction, vol, strength=.25)
        prediction[10:] = 2
        after = risk_targets(base, prediction, vol, strength=.25)
        np.testing.assert_array_equal(before[:10], after[:10])


if __name__ == "__main__":
    unittest.main()
