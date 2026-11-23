import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_classification_evaluation import (
    balanced_prior_predictions,
    calibrate_thresholds,
    make_labels,
    paired_classification_block_bootstrap,
    score_classification,
    variance_thresholds,
)


class ClassificationTargetTest(unittest.TestCase):
    def test_threshold_boundaries_are_inclusive_neutral(self):
        values = np.array([-0.021, -0.02, 0, 0.02, 0.021])
        np.testing.assert_array_equal(make_labels(values, .02), [0, 1, 1, 1, 2])
        np.testing.assert_array_equal(make_labels(values, np.full(5, .02)), [0, 1, 1, 1, 2])

    def test_calibration_uses_mean_ratios_and_separate_common_train_returns(self):
        fitted = calibrate_thresholds(np.array([.02, .01]), np.array([.0001, .0001]), train_returns=np.array([.04, -.04]))
        self.assertAlmostEqual(fitted['a'], 2.5)
        self.assertAlmostEqual(fitted['fixed_theta'], .02)
        self.assertEqual(fitted['n_oof'], 2)
        actual = variance_thresholds(np.array([.0004, .0001]), fitted['a'])
        np.testing.assert_allclose(actual, .5 * np.sqrt(np.array([.001, .00025])))
        original = fitted.copy()
        variance_thresholds(np.array([100., 200.]), fitted['a'])
        self.assertEqual(original, fitted)

    def test_zero_variance_uses_fixed_floor_without_refitting(self):
        fitted = calibrate_thresholds(np.array([1e-6]), np.array([0.]), train_returns=np.array([.01]))
        self.assertEqual(fitted['a'], 1)
        self.assertEqual(variance_thresholds(np.array([0.]), fitted['a'])[0], 5e-7)

    def test_invalid_and_degenerate_inputs_fail_explicitly(self):
        for theta in (0, -.1, np.inf, np.nan):
            with self.assertRaises(ValueError): make_labels(np.array([.1]), theta)
        with self.assertRaises(ValueError): make_labels(np.ones(3), np.ones((3, 1)))
        with self.assertRaises(ValueError): make_labels(np.array([np.nan]), .01)
        with self.assertRaises(ValueError): calibrate_thresholds([0.], [1.], train_returns=[0.])
        with self.assertRaises(ValueError): calibrate_thresholds([1.], [-1.], train_returns=[1.])
        with self.assertRaises(ValueError): variance_thresholds([1.], 0)


class ClassificationScoreTest(unittest.TestCase):
    def test_prior_rule_and_raw_rule_are_separate(self):
        probs = np.array([[.25, .5, .25], [.2, .3, .5]])
        priors = np.array([.1, .8, .1])
        np.testing.assert_array_equal(balanced_prior_predictions(probs, priors), [0, 2])
        score = score_classification([0, 2], probs, priors)
        self.assertEqual(score['accuracy'], 1)
        self.assertEqual(score['raw']['accuracy'], .5)
        self.assertIsNone(score['balanced_accuracy'])
        self.assertAlmostEqual(score['log_loss'], (-np.log(.25) - np.log(.5)) / 2)

    def test_true_three_class_metrics_and_brier_sum_convention(self):
        probs = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7]])
        score = score_classification([0, 1, 2], probs, np.full(3, 1 / 3))
        self.assertEqual(score['accuracy'], 1)
        self.assertEqual(score['balanced_accuracy'], 1)
        self.assertEqual(score['macro_f1'], 1)
        self.assertEqual(score['confusion_matrix'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(score['multiclass_brier'], (.06 + .14 + .14) / 3)
        self.assertEqual(score['neutral_fraction'], 1 / 3)

    def test_balanced_prior_ties_and_row_specific_priors(self):
        probs = np.full((3, 3), 1 / 3)
        np.testing.assert_array_equal(balanced_prior_predictions(probs, np.full(3, 1 / 3)), [0, 0, 0])
        priors = np.array([[.1, .8, .1], [.8, .1, .1], [.8, .15, .05]])
        np.testing.assert_array_equal(balanced_prior_predictions(probs, priors), [0, 1, 2])

    def test_majority_baseline_explicit_decisions_keep_proper_probabilities(self):
        actual = np.array([0, 1, 2])
        prior = np.array([.2, .6, .2])
        probs = np.tile(prior, (3, 1))
        score = score_classification(actual, probs, None, predictions=np.ones(3, dtype=int))
        self.assertEqual(score['balanced_accuracy'], 1 / 3)
        self.assertEqual(score['neutral_fraction'], 1)
        self.assertAlmostEqual(score['log_loss'], -np.mean(np.log(prior)))
        self.assertEqual(score['per_class_precision'], [0, 1 / 3, 0])
        self.assertEqual(score['per_class_recall'], [0, 1, 0])

    def test_absent_actual_class_is_not_silently_a_two_class_task(self):
        score = score_classification([0, 0], [[1, 0, 0], [1, 0, 0]], np.full(3, 1 / 3))
        self.assertIsNone(score['balanced_accuracy'])
        self.assertEqual(score['macro_f1'], 1 / 3)
        self.assertEqual(score['per_class_recall'], [1, None, None])

    def test_invalid_probabilities_and_priors_are_rejected(self):
        for probs in ([[.2, .2, .2]], [[.2, -.1, .9]], [[np.nan, .5, .5]]):
            with self.assertRaises(ValueError): score_classification([0], probs, np.full(3, 1 / 3))
        with self.assertRaises(ValueError): score_classification([0], [[.1, .8, .1]], [0, .5, .5])
        with self.assertRaises(ValueError): score_classification([0], [[.1, .8, .1]], None)
        with self.assertRaises(ValueError): score_classification([3], [[.1, .8, .1]], np.full(3, 1 / 3))


class ClassificationBootstrapTest(unittest.TestCase):
    @staticmethod
    def frame(days=35, seeds=(7, 17, 27)):
        times = pd.date_range('2023-01-01', periods=96 * days, freq='15min', tz='UTC')
        actual = np.arange(len(times)) % 3
        parts = []
        for model in ('a', 'b'):
            for seed in seeds:
                pred = np.zeros(len(times), dtype=int) if model == 'a' else actual.copy()
                probs = np.full((len(times), 3), .1)
                probs[np.arange(len(times)), pred] = .8
                parts.append(pd.DataFrame({'timestamp': times, 'fold': 0, 'seed': seed, 'condition': 'lgbm_theta', 'model': model,
                    'actual': actual, 'p0': probs[:, 0], 'p1': probs[:, 1], 'p2': probs[:, 2], 'pred': pred, 'pred_raw': pred}))
        return pd.concat(parts, ignore_index=True)

    def test_pooled_metrics_and_seed_averaging(self):
        frame = self.frame()
        results = paired_classification_block_bootstrap(frame, 'a', 'b', condition='lgbm_theta', replicates=39)
        self.assertEqual([r['block_days'] for r in results], [1, 7, 28])
        for row in results:
            self.assertEqual(row['n_paired_timestamps'], 35 * 96)
            self.assertEqual(row['n_seeds'], 3)
            self.assertAlmostEqual(row['metrics']['balanced_accuracy_difference']['estimate'], 2 / 3)
            self.assertAlmostEqual(row['metrics']['accuracy_difference']['estimate'], 2 / 3)
            self.assertGreater(row['metrics']['log_loss_improvement']['estimate'], 0)
        single = paired_classification_block_bootstrap(frame[frame.seed == 7], 'a', 'b', condition='lgbm_theta', replicates=39)
        for key in results[1]['metrics']:
            for name in ('estimate', 'ci_low', 'ci_high'):
                self.assertAlmostEqual(results[1]['metrics'][key][name], single[1]['metrics'][key][name])

    def test_seed_pool_is_not_a_probability_ensemble(self):
        frame = self.frame(days=14)
        mask = frame.model == 'b'
        actual = frame.loc[mask, 'actual'].to_numpy()
        seed = frame.loc[mask, 'seed'].to_numpy()
        pred = np.where(seed == 7, actual, (actual + 1) % 3)
        probs = np.full((len(pred), 3), .05)
        probs[np.arange(len(pred)), pred] = .9
        frame.loc[mask, ['p0', 'p1', 'p2']] = probs
        frame.loc[mask, 'pred'] = pred
        frame.loc[mask, 'pred_raw'] = pred
        result = paired_classification_block_bootstrap(frame, 'a', 'b', condition='lgbm_theta', block_days=(7,), replicates=39)[0]
        self.assertAlmostEqual(result['metrics']['balanced_accuracy_difference']['estimate'], 0)
        # Averaging probabilities instead would predict the wrong class on
        # every timestamp; this would have an entirely different BA of zero.

    def test_fold_boundaries_and_gaps_are_separate_segments(self):
        frame = self.frame(days=6)
        frame = frame[frame.timestamp != pd.Timestamp('2023-01-03 12:00', tz='UTC')].copy()
        frame.loc[frame.timestamp >= pd.Timestamp('2023-01-05', tz='UTC'), 'fold'] = 1
        result = paired_classification_block_bootstrap(frame, 'a', 'b', condition='lgbm_theta', block_days=(7,), replicates=39)[0]
        self.assertEqual(result['n_segments'], 3)
        self.assertFalse(result['inference_available'])
        self.assertIsNone(result['metrics']['balanced_accuracy_difference']['p_value'])
        self.assertIsNone(result['metrics']['balanced_accuracy_difference']['ci_low'])

    def test_common_condition_actuals_and_complete_seed_sets_are_required(self):
        frame = self.frame(days=2)
        with self.assertRaisesRegex(ValueError, 'identical timestamp'):
            paired_classification_block_bootstrap(frame.iloc[1:], 'a', 'b', condition='lgbm_theta', replicates=9)
        altered = frame.copy()
        altered.loc[altered.model == 'b', 'actual'] = (altered.loc[altered.model == 'b', 'actual'] + 1) % 3
        with self.assertRaisesRegex(ValueError, 'identical actual'):
            paired_classification_block_bootstrap(altered, 'a', 'b', condition='lgbm_theta', replicates=9)
        with self.assertRaises(ValueError):
            paired_classification_block_bootstrap(frame, 'a', 'b', condition='other_theta', replicates=9)

    def test_absent_class_resamples_are_explicitly_undefined(self):
        frame = self.frame(days=14)
        frame['actual'] = 0
        result = paired_classification_block_bootstrap(frame, 'a', 'b', condition='lgbm_theta', block_days=(7,), replicates=39)[0]
        metric = result['metrics']['balanced_accuracy_difference']
        self.assertIsNone(metric['estimate'])
        self.assertIsNone(metric['p_value'])
        self.assertEqual(metric['n_undefined_replicates'], 39)
        self.assertEqual(metric['n_valid_replicates'], 0)

    def test_week_blocks_preserve_persistent_classification_errors(self):
        frame = self.frame(days=84, seeds=(7,))
        bad = ((frame.timestamp - frame.timestamp.min()).dt.days // 14) % 2 == 1
        use = (frame.model == 'b') & bad
        pred = (frame.loc[use, 'actual'].to_numpy() + 1) % 3
        probs = np.full((len(pred), 3), .1)
        probs[np.arange(len(pred)), pred] = .8
        frame.loc[use, ['p0', 'p1', 'p2']] = probs
        frame.loc[use, 'pred'] = pred
        frame.loc[use, 'pred_raw'] = pred
        results = paired_classification_block_bootstrap(frame, 'a', 'b', condition='lgbm_theta', block_days=(1, 7), replicates=599)
        intervals = [r['metrics']['balanced_accuracy_difference'] for r in results]
        self.assertGreater(intervals[1]['ci_high'] - intervals[1]['ci_low'], intervals[0]['ci_high'] - intervals[0]['ci_low'])


if __name__ == '__main__':
    unittest.main()
