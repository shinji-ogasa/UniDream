"""Synthetic tests for the fixed short37 feature-by-direction comparison."""
import hashlib
import json
import math
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from unidream.experiments.oracle_direction_fit import fit_direction_family
from unidream.experiments.oracle_short_direction_fit import (
    FEATURE_NAMES, GROUP, MODEL_IDS, WEIGHTINGS, fit_short_direction_family,
)


class Poison:
    def __float__(self):
        raise AssertionError('unselected value was inspected')


def fixture():
    rng = np.random.default_rng(20200119)
    n = 608
    index = pd.date_range('2000-01-01', periods=n, freq='15min', tz='UTC')
    x = rng.normal(size=(n, 37))
    x[:, 2] = 1.
    sign = np.where(.45*x[:, 0] + .2*x[:, -1] + rng.normal(size=n) > 0, 1., -1.)
    y = sign * (.001 + .002*np.abs(x[:, 1]))
    features = pd.DataFrame(x, index=index, columns=FEATURE_NAMES)
    outcomes = pd.DataFrame(np.column_stack((y, np.abs(y), np.full(n, .02))),
                            index=index, columns=['return', 'adverse', 'volatility'])
    masks = {'fit_mask': np.arange(n) < 512,
             'predict_mask': (np.arange(n) >= 528) & (np.arange(n) < 592)}
    return features, outcomes, masks


def sigmoid(value):
    if value >= 0:
        return 1/(1+math.exp(-value))
    e = math.exp(value)
    return e/(1+e)


class ShortDirectionFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features, cls.outcomes, cls.masks = fixture()
        cls.result = fit_short_direction_family(cls.features, cls.outcomes, **cls.masks)

    def assert_same(self, result):
        for mid in MODEL_IDS:
            for name in ('logits', 'probabilities'):
                assert_array_equal(result[name][mid], self.result[name][mid])
        for name in ('provenance', 'fit_priors', 'fit_return_mean', 'fit_abs_return_mean'):
            self.assertEqual(result[name], self.result[name])
        assert_array_equal(result['fit_labels'], self.result['fit_labels'])
        for weighting in WEIGHTINGS:
            assert_array_equal(result['fit_weights'][weighting], self.result['fit_weights'][weighting])

    def test_two_models_exact_weights_priors_hashes_and_fixed_parameters(self):
        r = self.result
        fit, predict = self.masks['fit_mask'], self.masks['predict_mask']
        self.assertEqual(tuple(r['models']), MODEL_IDS)
        self.assertEqual(len(MODEL_IDS), 2)
        self.assertEqual(len(FEATURE_NAMES), 37)
        y = self.outcomes.iloc[fit, 0].to_numpy()
        a = math.fsum(abs(float(v))/len(y) for v in y)
        self.assertEqual(r['fit_abs_return_mean'], a)
        self.assertEqual(r['fit_return_mean'], float(np.mean(y)))
        assert_array_equal(r['fit_weights']['magnitude'], np.abs(y)/a)
        assert_array_equal(r['fit_weights']['ordinary'], np.ones(len(y)))
        assert_array_equal(r['fit_labels'], (y > 0).astype(np.int64))
        for weighting in WEIGHTINGS:
            w = r['fit_weights'][weighting]
            total = math.fsum(float(v) for v in w)
            self.assertEqual(r['fit_priors'][weighting], math.fsum(float(v) for v in w[y > 0])/total)
            self.assertAlmostEqual(total/len(y), 1., places=14)
            header = np.asarray([1, len(y)], dtype='<i8').tobytes()
            digest = hashlib.sha256(header + np.asarray(w, dtype='<f8').tobytes()).hexdigest()
            self.assertEqual(r['provenance']['sample_weights'][weighting]['weight_sha256'], digest)
        for mid in MODEL_IDS:
            for name in ('logits', 'probabilities'):
                assert_array_equal(np.isfinite(r[name][mid]), predict)
            lr = r['models'][mid]['logisticregression']
            for key, value in {'C': 1., 'l1_ratio': 0., 'solver': 'lbfgs', 'tol': 1e-8,
                               'max_iter': 1000, 'fit_intercept': True,
                               'random_state': 20260906}.items():
                self.assertEqual(lr.get_params()[key], value)
            self.assertIsNone(lr.class_weight)
            self.assertFalse(lr.warm_start)
        self.assertEqual(r['provenance']['feature_columns'], {GROUP: list(FEATURE_NAMES)})
        self.assertEqual(r['provenance']['feature_counts'], {GROUP: 37})
        self.assertTrue(r['provenance']['canonical_feature_order_required'])
        self.assertFalse(r['provenance']['support_narrowed'])
        self.assertFalse(r['provenance']['evaluation_labels_used'])
        self.assertFalse(r['provenance']['model_selection_performed'])
        self.assertFalse(r['provenance']['timestamp_feature_causality_and_label_completion_verified'])
        json.dumps(r['provenance'], allow_nan=False)

    def test_exact_synthetic_parity_with_original_fit_on_same_selected_37_coordinates(self):
        old = fit_direction_family({'technical': self.features,
                                    'perp_delay0': self.features.copy()},
                                   self.outcomes, **self.masks)
        for weighting in WEIGHTINGS:
            newmid, oldmid = GROUP+'_'+weighting, 'technical_'+weighting
            for key in ('logits', 'probabilities'):
                assert_array_equal(self.result[key][newmid], old[key][oldmid])
            newstate = self.result['provenance']['fitted_state'][newmid].copy()
            oldstate = old['provenance']['fitted_state'][oldmid].copy()
            newstate.pop('group'); oldstate.pop('group')
            self.assertEqual(newstate, oldstate)
            assert_array_equal(self.result['fit_weights'][weighting], old['fit_weights'][weighting])
        for key in ('fit_priors', 'fit_return_mean', 'fit_abs_return_mean'):
            self.assertEqual(self.result[key], old[key])
        assert_array_equal(self.result['fit_labels'], old['fit_labels'])

    def test_scalar_prediction_and_stationarity_are_checked_for_both_models(self):
        predict = self.masks['predict_mask']
        x = self.features.iloc[predict].to_numpy()
        for mid in MODEL_IDS:
            model = self.result['models'][mid]
            scaler, lr = model['standardscaler'], model['logisticregression']
            expected = [float(lr.intercept_[0]) + math.fsum(
                ((float(v)-float(center))/float(scale))*float(beta)
                for v, center, scale, beta in zip(row, scaler.mean_, scaler.scale_, lr.coef_[0]))
                for row in x]
            assert_allclose(self.result['logits'][mid][predict], expected, atol=1e-12, rtol=0)
            assert_allclose(self.result['probabilities'][mid][predict],
                            [sigmoid(v) for v in expected], atol=1e-14, rtol=0)
            checked = self.result['provenance']['fitted_state'][mid]['scalar_verification']
            self.assertTrue(checked['checked'])
            self.assertEqual((checked['fit_rows'], checked['predict_rows']), (512, 64))
            self.assertTrue(math.isfinite(checked['normalized_objective']))
            self.assertLessEqual(checked['normalized_gradient_infinity'], 1e-6)
            self.assertLessEqual(checked['max_abs_logit_difference'], 1e-12)
            self.assertLessEqual(checked['max_abs_probability_difference'], 1e-14)
        self.assertTrue(self.result['provenance']['objective']['stationarity_checked_by_this_fit_helper'])

    def test_fit_only_labels_auxiliary_poison_and_unused_features_cannot_change_results(self):
        fit, predict = self.masks['fit_mask'], self.masks['predict_mask']
        features = self.features.astype(object)
        outcomes = self.outcomes.astype(object)
        outcomes.iloc[~fit, 0] = Poison()
        outcomes.iloc[:, 1] = complex(1, 3)
        outcomes.iloc[:, 2] = True
        features.iloc[~(fit | predict), :] = Poison()
        self.assert_same(fit_short_direction_family(features, outcomes, **self.masks))
        self.assert_same(fit_short_direction_family(features, outcomes.to_numpy(), **self.masks))

    def test_selected_new_feature_or_return_invalidity_fails_before_any_fit_without_mask_change(self):
        bad = (True, np.bool_(False), complex(1, 0), '0.1', Poison(), None,
               pd.NA, np.nan, np.inf, -np.inf)
        original_masks = {k: v.copy() for k, v in self.masks.items()}
        for value in bad:
            for row in (0, 530):
                features = self.features.astype(object)
                features.iloc[row, -1] = value
                with self.subTest(value=repr(value), row=row), patch(
                        'unidream.experiments.oracle_short_direction_fit.make_pipeline') as make:
                    with self.assertRaises(ValueError):
                        fit_short_direction_family(features, self.outcomes, **self.masks)
                    make.assert_not_called()
            outcomes = self.outcomes.astype(object)
            outcomes.iloc[0, 0] = value
            with patch('unidream.experiments.oracle_short_direction_fit.make_pipeline') as make:
                with self.assertRaises(ValueError):
                    fit_short_direction_family(self.features, outcomes, **self.masks)
                make.assert_not_called()
        for key, mask in original_masks.items():
            assert_array_equal(self.masks[key], mask)

    def test_canonical_feature_names_count_order_and_index_are_required(self):
        features = self.features
        cases = [features.to_numpy(), {GROUP: features}, features.iloc[:, :-1],
                 features.assign(extra=0.), features.iloc[:, ::-1],
                 features.rename(columns={FEATURE_NAMES[-1]: 'renamed'}),
                 features.set_axis([FEATURE_NAMES[0]]*37, axis=1), features.iloc[:-1],
                 features.iloc[::-1], features.set_axis(pd.Index([0]*len(features))),
                 features.set_axis(features.index + pd.Timedelta(minutes=15))]
        for candidate in cases:
            with self.subTest(kind=type(candidate).__name__), patch(
                    'unidream.experiments.oracle_short_direction_fit.make_pipeline') as make:
                with self.assertRaises(ValueError):
                    fit_short_direction_family(candidate, self.outcomes, **self.masks)
                make.assert_not_called()
        for outcomes in (np.zeros(608), np.zeros((608, 2)), np.zeros((0, 3)),
                         self.outcomes.to_numpy().tolist()):
            with self.assertRaises(ValueError):
                fit_short_direction_family(features, outcomes, **self.masks)

    def test_strict_masks_minimum_and_fit_before_prediction(self):
        cases = []
        short = self.masks['fit_mask'].copy(); short[511] = False
        cases.append({**self.masks, 'fit_mask': short})
        cases.append({**self.masks, 'predict_mask': np.zeros(608, bool)})
        overlap = self.masks['predict_mask'].copy(); overlap[511] = True
        cases.append({**self.masks, 'predict_mask': overlap})
        late = self.masks['fit_mask'].copy(); late[550] = True
        cases.append({**self.masks, 'fit_mask': late})
        for key, mask in self.masks.items():
            for bad in (mask.astype(int), mask.astype(object), mask[:-1], mask[:, None]):
                cases.append({**self.masks, key: bad})
        for masks in cases:
            with self.assertRaises(ValueError), patch(
                    'unidream.experiments.oracle_short_direction_fit.make_pipeline') as make:
                fit_short_direction_family(self.features, self.outcomes, **masks)
            make.assert_not_called()

    def test_class_degeneracy_and_zero_magnitude_fail_before_partial_family(self):
        for y in (np.zeros(512), np.ones(512), -np.ones(512),
                  np.r_[np.zeros(256), np.ones(256)]):
            outcomes = self.outcomes.copy(); outcomes.iloc[:512, 0] = y
            with patch('unidream.experiments.oracle_short_direction_fit.make_pipeline') as make:
                with self.assertRaises(ValueError):
                    fit_short_direction_family(self.features, outcomes, **self.masks)
                make.assert_not_called()
        outcomes = self.outcomes.copy(); outcomes.iloc[0, 0] = 0.
        r = fit_short_direction_family(self.features, outcomes, **self.masks)
        self.assertEqual(r['fit_labels'][0], 0)
        self.assertEqual(r['fit_weights']['magnitude'][0], 0.)
        self.assertEqual(r['provenance']['sample_weights']['magnitude']['zero_weight_rows'], 1)

    def test_constant_features_recover_the_two_distinct_known_prior_objectives(self):
        features = self.features.copy(); features.iloc[:, :] = 3.
        outcomes = self.outcomes.copy()
        outcomes.iloc[:512, 0] = np.tile([-.01, .02], 256)
        r = fit_short_direction_family(features, outcomes, **self.masks)
        self.assertEqual(r['fit_priors']['ordinary'], .5)
        self.assertAlmostEqual(r['fit_priors']['magnitude'], 2/3, places=15)
        for weighting in WEIGHTINGS:
            mid = GROUP+'_'+weighting
            assert_array_equal(r['models'][mid]['logisticregression'].coef_, np.zeros((1, 37)))
            assert_allclose(r['probabilities'][mid][self.masks['predict_mask']],
                            r['fit_priors'][weighting], atol=1e-8, rtol=0)

    def test_scalers_unweighted_return_rescaling_cancels_and_inputs_do_not_alias(self):
        seen = []
        original = StandardScaler.fit
        def record(scaler, X, y=None, sample_weight=None):
            seen.append(sample_weight)
            return original(scaler, X, y, sample_weight=sample_weight)
        with patch.object(StandardScaler, 'fit', record):
            self.assert_same(fit_short_direction_family(self.features, self.outcomes, **self.masks))
        self.assertEqual(seen, [None, None])
        features, outcomes, masks = fixture()
        f0, y0 = features.copy(deep=True), outcomes.copy(deep=True)
        m0 = {k: v.copy() for k, v in masks.items()}
        for mask in masks.values():
            mask.setflags(write=False)
        r = fit_short_direction_family(features, outcomes, **masks)
        r['masks']['fit'][:] = False
        r['fit_labels'][:] = 0
        r['fit_weights']['ordinary'][:] = 0
        pd.testing.assert_frame_equal(features, f0)
        pd.testing.assert_frame_equal(outcomes, y0)
        for key in masks:
            assert_array_equal(masks[key], m0[key])
        outcomes.iloc[:, 0] *= 2
        scaled = fit_short_direction_family(features, outcomes, **masks)
        for mid in MODEL_IDS:
            assert_array_equal(scaled['logits'][mid], self.result['logits'][mid])
            assert_array_equal(scaled['probabilities'][mid], self.result['probabilities'][mid])

    def test_convergence_warning_and_iteration_limit_fail_without_retry(self):
        def warn(*args, **kwargs):
            warnings.warn('synthetic convergence failure', ConvergenceWarning)
        with patch.object(LogisticRegression, 'fit', side_effect=warn) as fit:
            with self.assertRaisesRegex(ValueError, 'convergence'):
                fit_short_direction_family(self.features, self.outcomes, **self.masks)
        self.assertEqual(fit.call_count, 1)
        original = LogisticRegression.fit
        calls = []
        def at_limit(model, *args, **kwargs):
            calls.append(1); original(model, *args, **kwargs)
            model.n_iter_[:] = 1000
            return model
        with patch.object(LogisticRegression, 'fit', at_limit):
            with self.assertRaisesRegex(ValueError, 'iteration limit'):
                fit_short_direction_family(self.features, self.outcomes, **self.masks)
        self.assertEqual(len(calls), 1)

    def test_finite_nonstationary_model_fails_original_scalar_guard_without_retry(self):
        original = LogisticRegression.fit
        calls = []
        def corrupt(model, *args, **kwargs):
            calls.append(1); original(model, *args, **kwargs)
            model.intercept_ += .1
            return model
        with patch.object(LogisticRegression, 'fit', corrupt):
            with self.assertRaisesRegex(ValueError, 'stationarity bound'):
                fit_short_direction_family(self.features, self.outcomes, **self.masks)
        self.assertEqual(len(calls), 1)

    def test_scalar_guard_rejects_finite_probability_corruption(self):
        original_fit = LogisticRegression.fit
        original_prob = Pipeline.predict_proba
        calls = []
        def track(model, *args, **kwargs):
            calls.append(1)
            return original_fit(model, *args, **kwargs)
        def corrupt(model, *args, **kwargs):
            p = original_prob(model, *args, **kwargs)
            p[:, 1] += 1e-5; p[:, 0] -= 1e-5
            return p
        with patch.object(LogisticRegression, 'fit', track), patch.object(Pipeline, 'predict_proba', corrupt):
            with self.assertRaisesRegex(ValueError, 'scalar predictor parity'):
                fit_short_direction_family(self.features, self.outcomes, **self.masks)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
