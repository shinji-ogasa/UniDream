import unittest

import numpy as np
import pandas as pd

from unidream.experiments.robust_overlay import build_targets


class RobustOverlayTests(unittest.TestCase):
    def _features(self, n=32):
        index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "momentum_7": np.ones(n),
                "momentum_30": np.ones(n),
                "momentum_90": np.ones(n),
                "vol_7": np.ones(n),
            },
            index=index,
        )

    def test_only_six_hour_decisions_are_emitted(self):
        targets = build_targets(self._features())
        decision = np.array(
            [ts.hour % 6 == 0 and ts.minute == 0 for ts in self._features().index]
        )
        self.assertTrue(np.isfinite(targets[decision]).all())
        self.assertTrue(np.isnan(targets[~decision]).all())

    def test_disagreement_returns_to_benchmark_tactical_leg(self):
        features = self._features()
        features.loc[:, "momentum_7"] = -0.1
        features.loc[:, "momentum_30"] = 0.1
        features.loc[:, "momentum_90"] = 0.1
        targets = build_targets(features)
        # The slow leg is overweight; the mixed tactical leg is exactly 1.0.
        # With 50/50 blending the target is therefore 1.06.
        self.assertAlmostEqual(float(targets[0]), 1.06, places=6)

    def test_missing_features_fail_closed(self):
        features = self._features()
        features.loc[features.index[0], "vol_7"] = np.nan
        targets = build_targets(features)
        self.assertTrue(np.isnan(targets[0]))


if __name__ == "__main__":
    unittest.main()
