import unittest

import numpy as np

from unidream.experiments.wm_tail_transfer_evaluation import (
    evaluate_predictions, per_row_losses, score_predictions,
)


class TailTransferEvaluationTests(unittest.TestCase):
    def fixture(self, name='return', kind='real', shape=(2,), task='T01', levels=None):
        cfg = {'targets': {'horizons': [1, 16], 'fixed_thresholds': [.01, .03],
                           'dynamic_multipliers': [1, 2, 3], 'hazard_thresholds': [.01, .03],
                           'passive_leverages': [.5, 1, 2], 'passive_loss_thresholds': [.05, .1]}}
        rng = np.random.default_rng(123)
        y = rng.normal(0, .01, (80,) + shape)
        if kind == 'binary': y = (y > 0).astype(float)
        elif kind in ('categorical', 'hazard'): y = rng.integers(0, 3, y.shape).astype(float)
        elif kind in ('positive', 'variance', 'positive_quantile'): y = abs(y)
        data = {'index_ns': np.arange(80) * 900_000_000_000,
                'targets': {name: y}, 'masks': {name: np.ones(y.shape, bool)}}
        spec = dict(name=name, kind=kind, shape=list(shape), task=task, levels=levels, classes=3)
        return cfg, data, [spec], np.arange(40), np.arange(40, 80)

    def test_perfect_regression_has_zero_loss_and_unit_skill(self):
        cfg, data, specs, train, ids = self.fixture()
        pred = {'return': data['targets']['return'][ids].copy()}
        result = evaluate_predictions(data, ids, pred, specs, cfg, train)
        np.testing.assert_array_equal(result.rmse, 0)
        np.testing.assert_array_equal(result.oos_r2_zero, 1)
        np.testing.assert_array_equal(result.skill, 1)
        self.assertAlmostEqual(score_predictions(data, ids, pred, specs, cfg, train), 0)

    def test_reference_return_predictor_has_score_one(self):
        cfg, data, specs, train, ids = self.fixture()
        pred = {'return': np.zeros_like(data['targets']['return'][ids])}
        self.assertAlmostEqual(score_predictions(data, ids, pred, specs, cfg, train), 1)

    def test_eval_labels_do_not_fit_baseline(self):
        cfg, data, specs, train, ids = self.fixture('mae', 'positive', task='T05')
        pred = {'mae': np.full((len(ids), 2), .01)}
        first = evaluate_predictions(data, ids, pred, specs, cfg, train)
        data['targets']['mae'][ids] += 100
        second = evaluate_predictions(data, ids, pred, specs, cfg, train)
        # The empirical Train mean is unchanged; recover it from baseline MSE
        # to show evaluation outcomes never become a fitted climatology.
        mean = data['targets']['mae'][train].mean(0)
        expected = ((data['targets']['mae'][ids] - mean) ** 2).mean(0)
        np.testing.assert_allclose(second.baseline_loss, expected)
        self.assertFalse(np.allclose(first.baseline_loss, second.baseline_loss))

    def test_binary_probabilities_report_episodes_and_calibration(self):
        cfg, data, specs, train, ids = self.fixture('crash', 'binary', task='legacy_crash')
        pred = {'crash': data['targets']['crash'][ids] * .8 + .1}
        result = evaluate_predictions(data, ids, pred, specs, cfg, train)
        np.testing.assert_allclose(result.brier, .01)
        np.testing.assert_allclose(result.balanced_accuracy, 1)
        self.assertTrue((result.event_episodes > 0).all())
        self.assertTrue((result.calibration_ece > 0).all())

    def test_constant_target_retains_counts_but_undefined_skill(self):
        cfg, data, specs, train, ids = self.fixture('crash', 'binary', task='legacy_crash')
        data['targets']['crash'][:] = 0
        pred = {'crash': np.full((len(ids), 2), .1)}
        result = evaluate_predictions(data, ids, pred, specs, cfg, train)
        self.assertTrue(result.skill.isna().all())
        self.assertTrue(result.balanced_accuracy.isna().all())
        self.assertTrue((result.events == 0).all())
        self.assertTrue((result.n == 40).all())

    def test_invalid_probabilities_fail_instead_of_being_clipped(self):
        cfg, data, specs, train, ids = self.fixture('crash', 'binary')
        with self.assertRaises(ValueError):
            evaluate_predictions(data, ids, {'crash': np.full((len(ids), 2), 1.01)}, specs, cfg, train)

    def test_missing_prediction_at_observed_target_fails(self):
        cfg, data, specs, train, ids = self.fixture()
        pred = data['targets']['return'][ids].copy(); pred[0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_predictions(data, ids, {'return': pred}, specs, cfg, train)

    def test_quantile_pinball_and_coverage(self):
        cfg, data, specs, train, ids = self.fixture('return_quantiles', 'quantile', (2, 2), 'T09', [.05, .95])
        pred = {'return_quantiles': np.zeros((len(ids), 2, 2))}
        result = evaluate_predictions(data, ids, pred, specs, cfg, train)
        self.assertEqual(len(result), 4)
        self.assertTrue(np.isfinite(result.pinball).all())
        self.assertTrue(result.coverage.between(0, 1).all())

    def test_var_es_fz0_finite_for_negative_ordered_predictions(self):
        cfg, data, specs, train, ids = self.fixture('var_es', 'var_es', (2, 2), 'T10', [.01, .05])
        pred = np.empty((len(ids), 2, 2, 2)); pred[..., 0] = -.02; pred[..., 1] = -.03
        result = evaluate_predictions(data, ids, {'var_es': pred}, specs, cfg, train)
        self.assertTrue(np.isfinite(result.fz0).all())
        self.assertTrue(np.isfinite(score_predictions(data, ids, {'var_es': pred}, specs, cfg, train)))

    def test_hazard_is_trajectory_log_likelihood_not_risk_step_mean(self):
        cfg, data, specs, train, ids = self.fixture('time_hazard', 'hazard', (16, 2), 'T11')
        y = data['targets']['time_hazard']; mask = data['masks']['time_hazard']
        y[:] = 0; y[:, 2] = 1; mask[:] = False; mask[:, :3] = True
        pred = np.tile([.8, .1, .1], (len(ids), 16, 2, 1))
        part = per_row_losses(data, ids, {'time_hazard': pred}, specs, cfg, train)['T11']
        expected = (-2 * np.log(.8) - np.log(.1)) / 16
        np.testing.assert_allclose(part['model_loss'], expected)
        report = evaluate_predictions(data, ids, {'time_hazard': pred}, specs, cfg, train)
        cif = report[report['head'] == 'time_hazard_cumulative_incidence']
        self.assertEqual(len(cif), 8)
        self.assertTrue((cif.n == 40).all())

    def test_horizon_filter_preserves_only_requested_cells(self):
        cfg, data, specs, train, ids = self.fixture()
        pred = data['targets']['return'][ids].copy(); pred[:, 0] += 1
        part = per_row_losses(data, ids, {'return': pred}, specs, cfg, train, horizon=16)['T01']
        np.testing.assert_array_equal(part['model_loss'], 0)
        self.assertTrue((part['ids'] == ids).all())

    def test_task_weight_does_not_depend_on_coordinate_count(self):
        cfg, data, specs, train, ids = self.fixture()
        data['targets']['other'] = data['targets']['return'].copy()
        data['masks']['other'] = data['masks']['return'].copy()
        specs.append(dict(name='other', kind='real', shape=[2], task='T02', levels=None))
        pred = {'return': np.zeros((len(ids), 2)), 'other': data['targets']['other'][ids].copy()}
        result = score_predictions(data, ids, pred, specs, cfg, train, return_details=True)
        self.assertEqual(result['n_available_tasks'], 2)
        self.assertAlmostEqual(result['score'], .5)


if __name__ == '__main__':
    unittest.main()
