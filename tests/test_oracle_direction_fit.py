"""Synthetic-only chronological logistic fits, weights and objective checks."""
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

from unidream.experiments.oracle_direction_fit import (
    GROUPS, MODEL_IDS, WEIGHTINGS, STATIONARITY_GRADIENT_BOUND, fit_direction_family,
)


def fixture():
    rng = np.random.default_rng(917)
    n = 608; index = pd.date_range('2000-01-01', periods=n, freq='15min', tz='UTC')
    x = rng.normal(size=(n, 3)); x[:, 2] = 1.
    um = rng.normal(size=n)
    groups = {
        'technical': pd.DataFrame(x, index=index, columns=['z', 'a', 'constant']),
        'perp_delay0': pd.DataFrame(np.column_stack((x, um)), index=index,
                                   columns=['z', 'a', 'constant', 'um']),
    }
    direction = np.where(.7*x[:, 0] + .2*um + rng.normal(size=n) > 0, 1., -1.)
    y = direction * (.001 + .002*np.abs(x[:, 1]))
    outcomes = pd.DataFrame(np.column_stack((y, np.abs(y), np.full(n, .02))),
                            index=index, columns=['return', 'adverse', 'volatility'])
    masks = {'fit_mask': np.arange(n) < 512,
             'predict_mask': (np.arange(n) >= 528) & (np.arange(n) < 592)}
    return groups, outcomes, masks


def sigmoid(z):
    if z >= 0: return 1/(1+math.exp(-z))
    ez = math.exp(z); return ez/(1+ez)


class Poison:
    def __float__(self): raise AssertionError('unselected poison was converted')


class DirectionFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.outcomes, cls.masks = fixture()
        cls.result = fit_direction_family(cls.groups, cls.outcomes, **cls.masks)

    def assert_same(self, actual):
        for mid in MODEL_IDS:
            for key in ('logits', 'probabilities'):
                assert_array_equal(actual[key][mid], self.result[key][mid])
        for key in ('provenance', 'fit_priors', 'fit_return_mean', 'fit_abs_return_mean'):
            self.assertEqual(actual[key], self.result[key])
        assert_array_equal(actual['fit_labels'], self.result['fit_labels'])
        for name in WEIGHTINGS:
            assert_array_equal(actual['fit_weights'][name], self.result['fit_weights'][name])

    def test_fixed_inventory_support_normalization_and_exact_selected_hashes(self):
        r = self.result; fit = self.masks['fit_mask']; predict = self.masks['predict_mask']
        self.assertEqual(tuple(r['models']), MODEL_IDS)
        y = self.outcomes.iloc[fit, 0].to_numpy()
        avg = math.fsum(abs(float(v))/512 for v in y)
        self.assertEqual(r['fit_abs_return_mean'], avg)
        assert_array_equal(r['fit_weights']['magnitude'], np.abs(y)/avg)
        assert_array_equal(r['fit_labels'], (y > 0).astype(np.int64))
        for name in WEIGHTINGS:
            w = r['fit_weights'][name]
            self.assertAlmostEqual(math.fsum(w)/512, 1., places=14)
            self.assertEqual(r['fit_priors'][name], math.fsum(w[y>0])/math.fsum(w))
            raw = np.asarray(w, dtype='<f8')
            h = hashlib.sha256(np.asarray([1,512],dtype='<i8').tobytes()+raw.tobytes()).hexdigest()
            self.assertEqual(r['provenance']['sample_weights'][name]['weight_sha256'], h)
        for mid in MODEL_IDS:
            for key in ('logits', 'probabilities'):
                assert_array_equal(np.isfinite(r[key][mid]), predict)
            logistic = r['models'][mid]['logisticregression']
            params = logistic.get_params()
            for key,value in {'C':1.,'l1_ratio':0.,'solver':'lbfgs','tol':1e-8,
                              'max_iter':1000,'fit_intercept':True,'random_state':20260906}.items():
                self.assertEqual(params[key], value)
            self.assertIsNone(params['class_weight'])
            self.assertFalse(params['warm_start'])
            self.assertEqual(r['models'][mid]['standardscaler'].get_params(), StandardScaler().get_params())
        self.assertFalse(r['provenance']['evaluation_labels_used'])
        self.assertFalse(r['provenance']['timestamp_feature_causality_and_label_completion_verified'])
        self.assertTrue(r['provenance']['objective']['stationarity_checked_by_this_fit_helper'])
        for mid in MODEL_IDS:
            check=r['provenance']['fitted_state'][mid]['scalar_verification']
            self.assertTrue(check['checked'])
            self.assertEqual((check['fit_rows'],check['predict_rows']),(512,64))
            self.assertTrue(math.isfinite(check['normalized_objective']))
            self.assertLessEqual(check['normalized_gradient_infinity'],1e-6)
            self.assertLessEqual(check['max_abs_logit_difference'],1e-12)
            self.assertLessEqual(check['max_abs_probability_difference'],1e-14)
        json.dumps(r['provenance'], allow_nan=False)

    def test_scalar_score_probability_and_normalized_objective_stationarity(self):
        fit, predict = self.masks['fit_mask'], self.masks['predict_mask']
        labels = self.result['fit_labels']
        for group in GROUPS:
            x = self.groups[group].to_numpy()
            for weighting in WEIGHTINGS:
                mid = group+'_'+weighting; model = self.result['models'][mid]
                scaler, lr = model['standardscaler'], model['logisticregression']
                beta, intercept = lr.coef_[0], float(lr.intercept_[0])
                transformed = (x-scaler.mean_)/scaler.scale_
                z = [intercept+math.fsum(float(a)*float(b) for a,b in zip(row,beta))
                     for row in transformed]
                prob = [sigmoid(v) for v in z]
                assert_allclose(self.result['logits'][mid][predict], np.asarray(z)[predict], atol=1e-14, rtol=0)
                assert_allclose(self.result['probabilities'][mid][predict], np.asarray(prob)[predict], atol=1e-15, rtol=0)
                weight = self.result['fit_weights'][weighting]; total = float(np.sum(weight))
                residual = weight*(np.asarray(prob)[fit]-labels)/total
                xx = transformed[fit]
                gradient = [math.fsum(float(r)*float(v) for r,v in zip(residual,xx[:,j]))+float(beta[j])/total
                            for j in range(len(beta))]
                gradient.append(math.fsum(residual))
                self.assertLessEqual(max(abs(v) for v in gradient), STATIONARITY_GRADIENT_BOUND)
                check=self.result['provenance']['fitted_state'][mid]['scalar_verification']
                assert_allclose(check['normalized_gradient'],gradient,atol=1e-15,rtol=0)
                # The known weighted logistic Hessian includes no intercept penalty.
                q = np.asarray(prob)[fit]; hweights = weight*q*(1-q)/total
                design = np.column_stack((xx,np.ones(len(xx))))
                hessian = np.einsum('i,ij,ik->jk',hweights,design,design,optimize=False)
                hessian[np.arange(len(beta)),np.arange(len(beta))] += 1/total
                self.assertGreaterEqual(float(np.linalg.eigvalsh(hessian).min()), -1e-14)

    def test_constant_feature_known_optimum_differs_by_weighted_prior(self):
        groups, outcomes, masks = fixture()
        for frame in groups.values(): frame.iloc[:,:] = 0.
        outcomes.iloc[:128,0] = -3.; outcomes.iloc[128:512,0] = 1.
        r = fit_direction_family(groups,outcomes,**masks)
        self.assertEqual(r['fit_priors'], {'ordinary':.75,'magnitude':.5})
        for group in GROUPS:
            for weighting, expected_logit in [('ordinary',math.log(3.)),('magnitude',0.)]:
                mid = group+'_'+weighting; lr = r['models'][mid]['logisticregression']
                assert_array_equal(lr.coef_,np.zeros_like(lr.coef_))
                assert_allclose(r['logits'][mid][masks['predict_mask']],expected_logit,atol=1e-7,rtol=0)
                assert_allclose(r['probabilities'][mid][masks['predict_mask']],r['fit_priors'][weighting],atol=1e-8,rtol=0)

    def test_weighted_classifier_never_weights_scaler_and_scale_of_returns_cancels(self):
        seen=[]; original=StandardScaler.fit
        def record(scaler, X, y=None, sample_weight=None):
            seen.append(sample_weight)
            return original(scaler,X,y,sample_weight=sample_weight)
        with patch.object(StandardScaler,'fit',record):
            self.assert_same(fit_direction_family(self.groups,self.outcomes,**self.masks))
        self.assertEqual(seen,[None]*4)
        changed=self.outcomes.copy();changed.iloc[:,0] *= 2
        r=fit_direction_family(self.groups,changed,**self.masks)
        for mid in MODEL_IDS:
            assert_array_equal(r['logits'][mid],self.result['logits'][mid])
            assert_array_equal(r['probabilities'][mid],self.result['probabilities'][mid])
        for group in GROUPS:
            scaler=self.result['models'][group+'_magnitude']['standardscaler']
            assert_allclose(scaler.mean_,self.groups[group].iloc[self.masks['fit_mask']].mean().to_numpy(),atol=1e-15,rtol=0)

    def test_future_labels_and_unselected_features_do_not_change_fit_or_prediction(self):
        fit,predict=self.masks['fit_mask'],self.masks['predict_mask']
        groups={k:v.astype(object) for k,v in self.groups.items()}
        outcomes=self.outcomes.astype(object)
        outcomes.iloc[~fit,0]=Poison();outcomes.iloc[:,1]=complex(1,3);outcomes.iloc[:,2]=True
        for frame in groups.values(): frame.iloc[~(fit|predict),:]=Poison()
        self.assert_same(fit_direction_family(groups,outcomes,**self.masks))
        self.assert_same(fit_direction_family(groups,outcomes.to_numpy(),**self.masks))

    def test_selected_values_and_input_schemas_fail_before_any_fit(self):
        bad_values=(True,np.bool_(False),complex(1,0),'0.1',Poison(),None,pd.NA,np.nan,np.inf,-np.inf)
        for value in bad_values:
            for row in (0,530):
                groups={k:v.astype(object) for k,v in self.groups.items()}
                groups['perp_delay0'].iloc[row,0]=value
                with self.subTest(value=repr(value),row=row),self.assertRaises(ValueError),patch(
                        'unidream.experiments.oracle_direction_fit.make_pipeline') as make:
                    fit_direction_family(groups,self.outcomes,**self.masks)
                make.assert_not_called()
            outcomes=self.outcomes.astype(object);outcomes.iloc[0,0]=value
            with self.assertRaises(ValueError): fit_direction_family(self.groups,outcomes,**self.masks)
        for outcomes in (np.zeros(608),np.zeros((608,2)),np.zeros((0,3)),self.outcomes.to_numpy().tolist()):
            with self.assertRaises(ValueError):fit_direction_family(self.groups,outcomes,**self.masks)
        for groups in ({'technical':self.groups['technical']},{**self.groups,'extra':self.groups['technical']}):
            with self.assertRaises(ValueError):fit_direction_family(groups,self.outcomes,**self.masks)

    def test_class_and_magnitude_degeneracy_fail_before_partial_family(self):
        for returns in (np.zeros(512),np.ones(512),-np.ones(512),np.r_[np.zeros(256),np.ones(256)]):
            outcomes=self.outcomes.copy();outcomes.iloc[:512,0]=returns
            with self.assertRaises(ValueError),patch('unidream.experiments.oracle_direction_fit.make_pipeline') as make:
                fit_direction_family(self.groups,outcomes,**self.masks)
            make.assert_not_called()
        outcomes=self.outcomes.copy();outcomes.iloc[0,0]=0.
        r=fit_direction_family(self.groups,outcomes,**self.masks)
        self.assertEqual(r['fit_labels'][0],0)
        self.assertEqual(r['fit_weights']['magnitude'][0],0.)
        self.assertEqual(r['provenance']['sample_weights']['magnitude']['zero_weight_rows'],1)

    def test_strict_masks_minimum_and_chronology(self):
        cases=[]
        short=self.masks['fit_mask'].copy();short[511]=False
        cases.append({**self.masks,'fit_mask':short})
        cases.append({**self.masks,'predict_mask':np.zeros(608,bool)})
        overlap=self.masks['predict_mask'].copy();overlap[511]=True
        cases.append({**self.masks,'predict_mask':overlap})
        late=self.masks['fit_mask'].copy();late[550]=True
        cases.append({**self.masks,'fit_mask':late})
        for name,mask in self.masks.items():
            for bad in (mask.astype(int),mask.astype(object),mask[:-1],mask[:,None]):
                cases.append({**self.masks,name:bad})
        for masks in cases:
            with self.assertRaises(ValueError):fit_direction_family(self.groups,self.outcomes,**masks)

    def test_alignment_and_feature_column_order_are_bound(self):
        frame=self.groups['perp_delay0']
        cases=[frame.iloc[::-1],frame.set_axis(pd.Index([0]*608)),
               frame.set_axis(frame.index+pd.Timedelta(minutes=15)),frame.iloc[:-1],frame.iloc[:,:0],
               frame.set_axis(['same']*4,axis=1),frame.set_axis(['z','a','',3],axis=1)]
        for bad in cases:
            with self.assertRaises(ValueError):fit_direction_family({**self.groups,'perp_delay0':bad},self.outcomes,**self.masks)
        r=fit_direction_family({**self.groups,'perp_delay0':frame.iloc[:,::-1]},self.outcomes,**self.masks)
        self.assertEqual(r['provenance']['feature_columns']['perp_delay0'],list(frame.columns[::-1]))
        self.assertNotEqual(r['provenance']['fit_features_sha256']['perp_delay0'],
                            self.result['provenance']['fit_features_sha256']['perp_delay0'])

    def test_inputs_preserved_and_returned_arrays_do_not_alias(self):
        groups,outcomes,masks=fixture()
        g0={k:v.copy(deep=True) for k,v in groups.items()};y0=outcomes.copy(deep=True)
        m0={k:v.copy() for k,v in masks.items()}
        for mask in masks.values():mask.setflags(write=False)
        r=fit_direction_family(groups,outcomes,**masks)
        r['masks']['fit'][:]=False;r['fit_labels'][:]=0;r['fit_weights']['ordinary'][:]=0
        for k,v in groups.items():pd.testing.assert_frame_equal(v,g0[k])
        pd.testing.assert_frame_equal(outcomes,y0)
        for k,v in masks.items():assert_array_equal(v,m0[k])

    def test_convergence_warning_iteration_limit_and_nonfinite_predict_fail_without_retry(self):
        def warn(*args,**kwargs): warnings.warn('synthetic failure',ConvergenceWarning)
        with patch.object(LogisticRegression,'fit',side_effect=warn) as fit:
            with self.assertRaisesRegex(ValueError,'convergence'):
                fit_direction_family(self.groups,self.outcomes,**self.masks)
        self.assertEqual(fit.call_count,1)
        original=LogisticRegression.fit;calls=[]
        def at_limit(model,*args,**kwargs):
            calls.append(1);original(model,*args,**kwargs);model.n_iter_[:]=1000;return model
        with patch.object(LogisticRegression,'fit',at_limit),self.assertRaisesRegex(ValueError,'iteration limit'):
            fit_direction_family(self.groups,self.outcomes,**self.masks)
        self.assertEqual(len(calls),1)
        with patch('sklearn.pipeline.Pipeline.decision_function',return_value=np.full(64,np.nan)):
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                fit_direction_family(self.groups,self.outcomes,**self.masks)

    def test_finite_but_nonstationary_model_is_rejected_without_retry(self):
        original=LogisticRegression.fit;calls=[]
        def corrupt(model,*args,**kwargs):
            calls.append(1);original(model,*args,**kwargs)
            model.intercept_ += .1
            return model
        with patch.object(LogisticRegression,'fit',corrupt):
            with self.assertRaisesRegex(ValueError,'stationarity bound'):
                fit_direction_family(self.groups,self.outcomes,**self.masks)
        self.assertEqual(len(calls),1)

    def test_finite_predictions_with_nonfinite_scalar_objective_are_rejected(self):
        original=LogisticRegression.fit;calls=[]
        def corrupt(model,*args,**kwargs):
            calls.append(1);original(model,*args,**kwargs)
            # This feature is identically zero after scaling: logits remain
            # finite, but the L2 penalty of its corrupted coefficient does not.
            model.coef_[0,2]=1e200
            return model
        with patch.object(LogisticRegression,'fit',corrupt):
            with self.assertRaisesRegex(ValueError,'nonfinite scalar objective'):
                fit_direction_family(self.groups,self.outcomes,**self.masks)
        self.assertEqual(len(calls),1)

    def test_scalar_predictor_rejects_logit_or_probability_corruption(self):
        original_fit=LogisticRegression.fit
        original_logit,original_probability=Pipeline.decision_function,Pipeline.predict_proba
        for kind in ('logit','probability'):
            calls=[]
            def track(model,*args,**kwargs):
                calls.append(1);return original_fit(model,*args,**kwargs)
            def wrong_logit(model,*args,**kwargs):
                return original_logit(model,*args,**kwargs)+.001
            def wrong_probability(model,*args,**kwargs):
                p=original_probability(model,*args,**kwargs)
                p[:,1] += 1e-5;p[:,0] -= 1e-5
                return p
            method='decision_function' if kind=='logit' else 'predict_proba'
            replacement=wrong_logit if kind=='logit' else wrong_probability
            with self.subTest(kind=kind),patch.object(LogisticRegression,'fit',track),patch.object(Pipeline,method,replacement):
                with self.assertRaisesRegex(ValueError,'scalar predictor parity'):
                    fit_direction_family(self.groups,self.outcomes,**self.masks)
            self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
