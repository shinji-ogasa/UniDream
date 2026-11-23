import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from unidream.experiments.wm_tail_transfer_data import build_targets
from unidream.experiments.wm_tail_transfer_models import build_specs
from unidream.experiments.wm_tail_transfer_baselines import (
    _filtered_variance, _innovation_bank, _variance_paths,
    apply_calibrator, fit_calibrator, fit_statistical_baselines, predict_statistical_baselines,
)


class TailStatisticalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                  'configs/experiments/wm_tail_transfer_20261112.yaml').read_text())
        cls.cfg['targets']['horizons'] = [1, 4, 16]
        cls.cfg['targets']['include_perpetual'] = False  # This fixture supplies spot path labels only.
        cls.cfg['statistical_baselines'] = {'monte_carlo_paths': 64, 'variance_windows': [16, 96],
                                           'ewma_decays': [.94, .97], 'seed': 123}
        rng = np.random.default_rng(123)
        close = 100 * np.exp(np.cumsum(rng.normal(0, .003, 1400)))
        opening = np.r_[close[0], close[:-1]]
        raw = np.column_stack([opening, np.maximum(opening, close) * 1.001,
                                np.minimum(opening, close) * .999, close, np.ones(len(close))])
        ix = np.arange(len(close), dtype=np.int64) * 900_000_000_000
        target, mask, _ = build_targets(ix, raw, cls.cfg)
        cls.data = {'index_ns': ix, 'ohlcspot': raw, 'targets': target, 'masks': mask}
        cls.train = np.arange(100, 900)
        cls.val = np.arange(950, 1100)

    @staticmethod
    def fakefit(r, asymmetric=False):
        return {'params': {'omega': .004, 'alpha[1]': .05, 'beta[1]': .9,
                            'gamma[1]': .02 if asymmetric else 0, 'nu': 8},
                'initial_variance_percent2': .1, 'standardized_residuals': np.linspace(-3, 3, 1000).tolist(),
                'convergence_flag': 0, 'warnings': [], 'fit_observations': len(r)}

    def bundle(self, data=None):
        with patch('unidream.experiments.wm_tail_transfer_baselines._fit_garch', self.fakefit):
            return fit_statistical_baselines(data or self.data, self.train, self.val, self.cfg)

    def test_fit_is_train_only_and_val_selects_fixed_candidates(self):
        b = self.bundle()
        self.assertEqual(b['train_last_id'], 899)
        self.assertEqual(b['validation_first_id'], 950)
        self.assertIn(b['selected_window'], [16, 96])
        self.assertEqual(b['fits']['garch_gaussian']['fit_observations'], 800)

    def test_fit_does_not_compress_missing_returns(self):
        data = copy.deepcopy(self.data)
        data['ohlcspot'][400, 3] = np.nan
        b = self.bundle(data)
        fitted = b['fits']['garch_gaussian']
        self.assertEqual(fitted['contiguous_fit_first_id'], 402)
        self.assertEqual(fitted['contiguous_fit_last_id'], 899)
        self.assertEqual(fitted['fit_observations'], 498)

    def test_future_price_changes_cannot_change_forecast(self):
        b = self.bundle(); ids = np.array([1150, 1160])
        first = predict_statistical_baselines(self.data, ids, b, self.cfg)
        data = copy.deepcopy(self.data); data['ohlcspot'][1160:, :4] *= 1.3
        second = predict_statistical_baselines(data, ids, b, self.cfg)
        for name in first:
            for head in first[name]:
                np.testing.assert_allclose(first[name][head], second[name][head], equal_nan=True)

    def test_mc_forecasts_are_invariant_to_chunk_and_order(self):
        b = self.bundle()
        a = predict_statistical_baselines(self.data, [1150, 1160], b, self.cfg)
        z = predict_statistical_baselines(self.data, [1160], b, self.cfg)
        for name in ['garch_gaussian', 'gjr_student_t', 'historical_residual']:
            for head in a[name]:
                np.testing.assert_allclose(a[name][head][1], z[name][head][0])

    def test_variances_positive_and_quantiles_ordered(self):
        b = self.bundle(); result = predict_statistical_baselines(self.data, [1150, 1160], b, self.cfg)
        for model in result.values():
            if 'rv' in model: self.assertTrue((model['rv'] > 0).all())
            if 'return_quantiles' in model: self.assertTrue((np.diff(model['return_quantiles'], axis=-1) >= 0).all())
            if 'var_es' in model:
                self.assertTrue((model['var_es'][..., 1] <= model['var_es'][..., 0]).all())
                self.assertTrue((model['var_es'][..., 0] < 0).all())

    def test_fixed_params_filter_uses_only_previous_return(self):
        fitted = self.fakefit(np.zeros(500))
        a = _filtered_variance(self.data, fitted)
        data = copy.deepcopy(self.data); data['ohlcspot'][1200:, :4] *= 2
        b = _filtered_variance(data, fitted)
        np.testing.assert_array_equal(a[:1201], b[:1201])

    def test_in_sample_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_statistical_baselines(self.data, [899], self.bundle(), self.cfg)

    def test_calibrators_do_not_invent_classes_for_constant_labels(self):
        data = {'targets': {'x': np.zeros((40, 1))}, 'masks': {'x': np.ones((40, 1), bool)}}
        p = {'x': np.full((40, 1), .3)}
        specs = [{'name': 'x', 'kind': 'binary', 'shape': [1]}]
        for method in ['parametric', 'isotonic']:
            b = fit_calibrator(data, np.arange(40), p, specs, method=method)
            self.assertEqual(b['heads']['x']['cells'][0]['kind'], 'identity')
            np.testing.assert_array_equal(apply_calibrator(p, b)['x'], p['x'])

    def test_binary_calibration_uses_disjoint_fit_arrays(self):
        y = np.r_[np.zeros(40), np.ones(40)][:, None]
        data = {'targets': {'x': y}, 'masks': {'x': np.ones(y.shape, bool)}}
        p = {'x': np.linspace(.2, .8, 80)[:, None]}
        specs = [{'name': 'x', 'kind': 'binary', 'shape': [1]}]
        for method in ['parametric', 'isotonic']:
            b = fit_calibrator(data, np.arange(80), p, specs, method=method)
            out = apply_calibrator({'x': np.array([[.25], [.75]])}, b)['x']
            self.assertLess(out[0, 0], out[1, 0])
            self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_multiclass_calibration_stays_normalized(self):
        y = (np.arange(90) % 3)[:, None].astype(float)
        data = {'targets': {'x': y}, 'masks': {'x': np.ones(y.shape, bool)}}
        p = np.full((90, 1, 3), .2); p[np.arange(90), 0, y[:, 0].astype(int)] = .6
        specs = [{'name': 'x', 'kind': 'categorical', 'shape': [1], 'classes': 3}]
        for method in ['parametric', 'isotonic']:
            b = fit_calibrator(data, np.arange(90), {'x': p}, specs, method=method)
            out = apply_calibrator({'x': p}, b)['x']
            np.testing.assert_allclose(out.sum(-1), 1)
            self.assertTrue((out >= 0).all())


if __name__ == '__main__':
    unittest.main()
