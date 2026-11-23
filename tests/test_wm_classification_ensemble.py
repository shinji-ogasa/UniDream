import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from unidream.experiments.wm_classification_ensemble import (
    DEFAULT_CONFIG, fit_task, load_config, quick_score, select_weights,
    simplex_grid, stacking_split,
)


class EnsembleSelectionTest(unittest.TestCase):
    def test_weight_grid_includes_single_models_and_respects_simplex(self):
        for size, count in ((3, 66), (4, 286)):
            weights = simplex_grid(size)
            self.assertEqual(weights.shape, (count, size))
            np.testing.assert_allclose(weights.sum(1), 1)
            self.assertTrue((weights >= 0).all())
            for single in np.eye(size):
                self.assertTrue(np.any(np.all(weights == single, axis=1)))

    def test_validation_selection_can_ignore_a_harmful_model(self):
        y = np.arange(60) % 3
        perfect = .05 + .85 * np.eye(3)[y]
        wrong = .05 + .85 * np.eye(3)[(y + 1) % 3]
        p = np.stack([perfect, wrong, wrong], axis=1)
        chosen, history = select_weights(y, p, np.full(3, 1 / 3))
        np.testing.assert_array_equal(chosen, [1, 0, 0])
        self.assertEqual(len(history), 66)

    def test_balanced_decisions_and_probability_loss_remain_separate(self):
        y = np.array([0, 1, 2])
        p = np.array([[.25, .5, .25], [.1, .8, .1], [.2, .3, .5]])
        score = quick_score(y, p, np.array([.1, .8, .1]))
        self.assertEqual(score['balanced_accuracy'], 2 / 3)
        self.assertAlmostEqual(score['log_loss'], -np.log([.25, .8, .5]).mean())

    def test_stacking_chronology_purges_full_auxiliary_horizon(self):
        times = pd.date_range('2026-04-01', '2026-07-01', freq='15min', inclusive='left', tz='UTC')
        fit, select = stacking_split(times, '2026-04-01', 16)
        self.assertLess(times[fit[-1]] + pd.Timedelta(hours=16), times[select[0]])
        self.assertEqual(times[select[0]], pd.Timestamp('2026-06-01', tz='UTC'))
        with self.assertRaises(ValueError):
            stacking_split(times[::-1], '2026-04-01', 16)
        with self.assertRaises(ValueError):
            stacking_split(times, '2026-07-01', 16)

    def test_test_rows_cannot_enter_an_ensemble_fit(self):
        with TemporaryDirectory() as folder:
            cfg = {'output_dir': folder, 'aligned_predictions': 'dummy.parquet'}
            frame = pd.DataFrame({'split': ['test'], 'timestamp': [pd.Timestamp('2026-01-01', tz='UTC')]})
            with patch('unidream.experiments.wm_classification_ensemble.load_config', return_value=cfg), \
                 patch('unidream.experiments.wm_classification_ensemble.pd.read_parquet', return_value=frame) as read:
                with self.assertRaisesRegex(ValueError, 'Val rows only'):
                    fit_task('unused', 14, 7)
                self.assertIn(('split', '==', 'val'), read.call_args.kwargs['filters'])

    def test_test_selection_training_or_overlapping_holdout_cannot_be_enabled(self):
        original = yaml.safe_load(DEFAULT_CONFIG.read_text())
        for changes in ({'test_selection_allowed': True}, {'base_model_training': True}, {'fresh_start': '2026-10-01'}):
            with TemporaryDirectory() as folder:
                path = Path(folder) / 'config.yaml'
                path.write_text(yaml.safe_dump(original | changes))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
