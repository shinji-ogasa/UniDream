import unittest

from unidream.experiments.alpha_dd_features import FEATURE_NAMES
from unidream.experiments.alpha_dd_ml import RECIPE_NAME, registered_candidates
from unidream.experiments.alpha_dd_search import candidate_universe


class RegisteredMLRecipeTests(unittest.TestCase):
    def test_exact_original_ml_universe(self):
        candidates = registered_candidates()
        self.assertEqual(len(candidates), 25)
        self.assertEqual(candidates[0].family, "hold")
        expected = [c for c in candidate_universe()
                    if c.family in ("hold", "ridge", "hgb", "logistic")]
        self.assertEqual(candidates, expected)
        self.assertEqual(len({c.id for c in candidates}), 25)
        self.assertEqual(len(FEATURE_NAMES), 16)
        self.assertEqual(RECIPE_NAME, "gap_aware_ml_v1")


if __name__ == "__main__":
    unittest.main()
