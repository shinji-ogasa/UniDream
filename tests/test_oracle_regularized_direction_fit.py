"""Synthetic-only checks of the single fixed average-loss L2 schedule."""
import copy
import inspect
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

from unidream.experiments import oracle_direction_fit as parent
from unidream.experiments.oracle_regularized_direction_fit import (
    GROUPS, MODEL_IDS, WEIGHTINGS, REGULARIZATION_RULE,
    _regularization_normalizer, fit_regularized_direction_family,
)


MODULE = 'unidream.experiments.oracle_regularized_direction_fit'


def fixture():
    rng = np.random.default_rng(917)
    n = 608
    index = pd.date_range('2000-01-01', periods=n, freq='15min', tz='UTC')
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
    if z >= 0:
        return 1/(1+math.exp(-z))
    ez = math.exp(z)
    return ez/(1+ez)


def scalar_objective_gradient(x, labels, weights, beta, intercept, c):
    """Independent normalized objective, using the actual per-model C."""
    total = float(np.sum(weights))
    l2 = 1/(c*total)
    logits = [intercept+math.fsum(float(a)*float(b) for a,b in zip(row,beta))
              for row in x]
    losses, residuals = [], []
    for z, label, w in zip(logits, labels, weights):
        signed = -z if label else z
        losses.append(float(w)/total*(max(signed,0.)+math.log1p(math.exp(-abs(signed)))))
        residuals.append(float(w)/total*(-sigmoid(-z) if label else sigmoid(z)))
    objective = math.fsum(losses)+.5*l2*math.fsum(float(b)*float(b) for b in beta)
    gradient = [math.fsum(r*float(row[j]) for r,row in zip(residuals,x))+l2*float(beta[j])
                for j in range(len(beta))]
    gradient.append(math.fsum(residuals))
    return objective, gradient


class Poison:
    def __float__(self):
        raise AssertionError('unselected value was converted')


class RegularizedDirectionFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.outcomes, cls.masks = fixture()
        cls.result = fit_regularized_direction_family(cls.groups, cls.outcomes, **cls.masks)

    def assert_same(self, actual):
        for mid in MODEL_IDS:
            for key in ('logits', 'probabilities'):
                assert_array_equal(actual[key][mid], self.result[key][mid])
        for key in ('provenance', 'fit_priors', 'fit_return_mean', 'fit_abs_return_mean'):
            self.assertEqual(actual[key], self.result[key])
        assert_array_equal(actual['fit_labels'], self.result['fit_labels'])
        for name in WEIGHTINGS:
            assert_array_equal(actual['fit_weights'][name], self.result['fit_weights'][name])

    def test_one_fixed_family_weight_arithmetic_and_recorded_actual_regularization(self):
        r = self.result; fit, predict = self.masks['fit_mask'], self.masks['predict_mask']
        self.assertEqual(tuple(r['models']), MODEL_IDS)
        y = self.outcomes.iloc[fit, 0].to_numpy()
        avg = math.fsum(abs(float(v))/len(y) for v in y)
        assert_array_equal(r['fit_labels'], (y > 0).astype(np.int64))
        assert_array_equal(r['fit_weights']['ordinary'], np.ones(512))
        assert_array_equal(r['fit_weights']['magnitude'], np.abs(y)/avg)
        self.assertEqual(r['fit_abs_return_mean'], avg)
        self.assertEqual(r['fit_return_mean'], float(np.mean(y)))
        for name in WEIGHTINGS:
            w = r['fit_weights'][name]
            self.assertEqual(r['fit_priors'][name], math.fsum(w[y>0])/math.fsum(w))
        for mid in MODEL_IDS:
            model = r['models'][mid]; params = model['logisticregression'].get_params()
            state = r['provenance']['fitted_state'][mid]
            reg = state['regularization']; w = r['fit_weights'][state['weighting']]
            total = float(np.sum(w)); expected_c = 1.0/total
            self.assertEqual(params['C'], expected_c)
            self.assertEqual(reg['C'], expected_c)
            self.assertEqual(reg['solver_weight_sum'], total)
            self.assertEqual(reg['actual_l2_strength'], 1/(expected_c*total))
            self.assertEqual(reg['normalizer'], REGULARIZATION_RULE)
            self.assertEqual(reg['weight_sha256'],
                             r['provenance']['sample_weights'][state['weighting']]['weight_sha256'])
            self.assertEqual(state['logistic_parameters'], params)
            self.assertEqual(r['provenance']['parameters']['logistic_by_model'][mid], params)
            for key, value in parent.LOGISTIC_PARAMETERS.items():
                if key != 'C':
                    self.assertEqual(params[key], value)
            self.assertIsNone(params['class_weight']); self.assertFalse(params['warm_start'])
            self.assertEqual(model['standardscaler'].get_params(), StandardScaler().get_params())
            for key in ('logits','probabilities'):
                assert_array_equal(np.isfinite(r[key][mid]), predict)
        self.assertNotIn('C', r['provenance']['parameters']['logistic_shared_parameters'])
        self.assertFalse(r['provenance']['model_selection_performed'])
        self.assertFalse(r['provenance']['regularization']['search_performed'])
        self.assertFalse(r['provenance']['evaluation_labels_used'])
        self.assertFalse(r['provenance']['risk_or_calibration_fitted'])
        self.assertTrue(r['provenance']['objective']['stationarity_checked_by_this_fit_helper'])
        self.assertEqual(tuple(inspect.signature(fit_regularized_direction_family).parameters),
                         ('groups','outcomes','fit_mask','predict_mask'))
        json.dumps(r['provenance'], allow_nan=False)

    def test_all_four_scalar_predictions_and_correct_normalized_l2_stationarity(self):
        fit, predict = self.masks['fit_mask'], self.masks['predict_mask']
        r = self.result
        for group in GROUPS:
            x = self.groups[group].to_numpy()
            for weighting in WEIGHTINGS:
                mid = group+'_'+weighting; model = r['models'][mid]
                scaler, lr = model['standardscaler'], model['logisticregression']
                beta, intercept = lr.coef_[0], float(lr.intercept_[0])
                transformed = (x-scaler.mean_)/scaler.scale_
                z = [intercept+math.fsum(float(a)*float(b) for a,b in zip(row,beta))
                     for row in transformed[predict]]
                assert_allclose(r['logits'][mid][predict], z, atol=1e-12, rtol=0)
                assert_allclose(r['probabilities'][mid][predict], [sigmoid(v) for v in z],
                                atol=1e-14, rtol=0)
                objective, grad = scalar_objective_gradient(
                    transformed[fit], r['fit_labels'], r['fit_weights'][weighting],
                    beta, intercept, lr.C)
                check = r['provenance']['fitted_state'][mid]['scalar_verification']
                self.assertTrue(math.isfinite(objective))
                self.assertLessEqual(max(abs(v) for v in grad), 1e-6)
                self.assertEqual(objective, check['normalized_objective'])
                assert_allclose(grad, check['normalized_gradient'], atol=1e-15, rtol=0)
                # The old C=1 penalty is a different objective at this solution.
                _, old_grad = scalar_objective_gradient(
                    transformed[fit], r['fit_labels'], r['fit_weights'][weighting],
                    beta, intercept, 1.)
                self.assertGreater(max(abs(v) for v in old_grad), 1e-3)

    def test_replicated_training_rows_preserve_average_loss_objective_and_solution(self):
        groups, outcomes, masks = fixture()
        new_index = pd.date_range('2000-01-01', periods=1120, freq='15min', tz='UTC')
        def replicate(frame):
            a = frame.to_numpy()
            return pd.DataFrame(np.concatenate((np.repeat(a[:512],2,axis=0),a[512:])),
                                index=new_index,columns=frame.columns)
        gg = {k:replicate(v) for k,v in groups.items()}; yy=replicate(outcomes)
        mm = {'fit_mask':np.arange(1120)<1024,
              'predict_mask':(np.arange(1120)>=1040)&(np.arange(1120)<1104)}
        doubled = fit_regularized_direction_family(gg,yy,**mm)
        for mid in MODEL_IDS:
            group = self.result['provenance']['fitted_state'][mid]['group']
            weighting = self.result['provenance']['fitted_state'][mid]['weighting']
            old_model, new_model = self.result['models'][mid], doubled['models'][mid]
            old_c, new_c = old_model['logisticregression'].C, new_model['logisticregression'].C
            self.assertEqual(new_c, old_c/2)
            # Evaluate an arbitrary fixed parameter vector, independently of both fits.
            beta = np.linspace(-.23,.31,groups[group].shape[1]); intercept=.17
            objectives=[]; gradients=[]
            for result, frame, mask, model in ((self.result,groups[group],masks['fit_mask'],old_model),
                                               (doubled,gg[group],mm['fit_mask'],new_model)):
                s=model['standardscaler']; x=(frame.to_numpy()[mask]-s.mean_)/s.scale_
                obj,grad=scalar_objective_gradient(x,result['fit_labels'],result['fit_weights'][weighting],
                                                   beta,intercept,model['logisticregression'].C)
                objectives.append(obj); gradients.append(grad)
            assert_allclose(objectives[0],objectives[1],atol=1e-14,rtol=0)
            assert_allclose(gradients[0],gradients[1],atol=1e-14,rtol=0)
            assert_allclose(new_model['logisticregression'].coef_,old_model['logisticregression'].coef_,
                            atol=1e-7,rtol=0)
            assert_allclose(doubled['logits'][mid][mm['predict_mask']],
                            self.result['logits'][mid][masks['predict_mask']],atol=1e-6,rtol=0)

    def test_known_constant_feature_optimum_keeps_intercept_unpenalized(self):
        groups, outcomes, masks = fixture()
        for frame in groups.values(): frame.iloc[:,:] = 0.
        outcomes.iloc[:128,0] = -3.; outcomes.iloc[128:512,0] = 1.
        result = fit_regularized_direction_family(groups,outcomes,**masks)
        self.assertEqual(result['fit_priors'], {'ordinary':.75,'magnitude':.5})
        for group in GROUPS:
            for weighting, expected in [('ordinary',math.log(3.)),('magnitude',0.)]:
                mid=group+'_'+weighting
                assert_array_equal(result['models'][mid]['logisticregression'].coef_,
                                   np.zeros((1,groups[group].shape[1])))
                assert_allclose(result['logits'][mid][masks['predict_mask']],expected,atol=1e-7,rtol=0)

    def test_unweighted_scaler_and_old_globals_unchanged(self):
        before=copy.deepcopy(parent.LOGISTIC_PARAMETERS); seen=[]
        original=StandardScaler.fit
        def record(scaler,X,y=None,sample_weight=None):
            seen.append(sample_weight)
            return original(scaler,X,y,sample_weight=sample_weight)
        with patch.object(StandardScaler,'fit',record):
            self.assert_same(fit_regularized_direction_family(self.groups,self.outcomes,**self.masks))
        self.assertEqual(seen,[None]*4); self.assertEqual(parent.LOGISTIC_PARAMETERS,before)
        self.assertEqual(parent.LOGISTIC_PARAMETERS['C'],1.)

    def test_fit_only_poison_does_not_filter_or_change_predictions(self):
        fit,predict=self.masks['fit_mask'],self.masks['predict_mask']
        groups={k:v.astype(object) for k,v in self.groups.items()}
        outcomes=self.outcomes.astype(object)
        outcomes.iloc[~fit,0]=Poison(); outcomes.iloc[:,1]=complex(1,3); outcomes.iloc[:,2]=True
        for frame in groups.values(): frame.iloc[~(fit|predict),:]=Poison()
        self.assert_same(fit_regularized_direction_family(groups,outcomes,**self.masks))
        self.assert_same(fit_regularized_direction_family(groups,outcomes.to_numpy(),**self.masks))

    def test_normalizer_rejects_invalid_weights_without_fit(self):
        bad=[[],np.zeros(2),np.ones((2,2)),np.array([1.,-1.]),np.ones(2)*2,
             [np.nan,1.],[np.inf,1.],[1e308,1e308],np.ones(2,dtype=complex),
             np.ones(2,dtype=bool),['1','1'],np.array([Poison(),1.],dtype=object)]
        for weights in bad:
            with self.subTest(weights=repr(weights)), self.assertRaises(ValueError):
                _regularization_normalizer(weights)
        valid=np.array([0.,.5,1.5,2.]); before=valid.copy()
        reg=_regularization_normalizer(valid)
        self.assertEqual((reg['C'],reg['solver_weight_sum'],reg['actual_l2_strength']),(.25,4.,1.))
        assert_array_equal(valid,before)
        with patch(MODULE+'._regularization_normalizer',side_effect=ValueError('normalizer failure')):
            with patch(MODULE+'.make_pipeline') as make, self.assertRaisesRegex(ValueError,'normalizer'):
                fit_regularized_direction_family(self.groups,self.outcomes,**self.masks)
        make.assert_not_called()

    def test_strict_selected_values_masks_and_chronology_fail_before_fitting(self):
        for bad in (True,complex(1,0),'1',Poison(),np.nan,np.inf):
            groups={k:v.astype(object) for k,v in self.groups.items()}
            groups['technical'].iloc[528,0]=bad
            with patch(MODULE+'.make_pipeline') as make, self.assertRaises(ValueError):
                fit_regularized_direction_family(groups,self.outcomes,**self.masks)
            make.assert_not_called()
            y=self.outcomes.astype(object);y.iloc[0,0]=bad
            with self.assertRaises(ValueError):
                fit_regularized_direction_family(self.groups,y,**self.masks)
        cases=[]
        short=self.masks['fit_mask'].copy();short[511]=False
        overlap=self.masks['predict_mask'].copy();overlap[511]=True
        cases.extend([{'fit_mask':short},{'predict_mask':overlap},
                      {'predict_mask':np.zeros(608,bool)}])
        for key,value in self.masks.items():
            for bad in (value.astype(int),value.astype(object),value[:-1],value[:,None]):
                cases.append({key:bad})
        for changed in cases:
            with self.assertRaises(ValueError):
                fit_regularized_direction_family(self.groups,self.outcomes,**{**self.masks,**changed})
        with self.assertRaises(TypeError):
            fit_regularized_direction_family(self.groups,self.outcomes,**self.masks,C=.1)

    def test_degenerate_class_weight_support_fails_before_any_model(self):
        for values in (np.zeros(512),np.ones(512),-np.ones(512),np.r_[np.zeros(256),np.ones(256)]):
            y=self.outcomes.copy();y.iloc[:512,0]=values
            with patch(MODULE+'.make_pipeline') as make, self.assertRaises(ValueError):
                fit_regularized_direction_family(self.groups,y,**self.masks)
            make.assert_not_called()

    def test_input_arrays_and_masks_are_preserved(self):
        groups,y,masks=fixture(); gg={k:v.copy(deep=True) for k,v in groups.items()}
        yy=y.copy(deep=True); mm={k:v.copy() for k,v in masks.items()}
        for mask in masks.values():mask.setflags(write=False)
        result=fit_regularized_direction_family(groups,y,**masks)
        result['masks']['fit'][:]=False;result['fit_weights']['ordinary'][:]=0;result['fit_labels'][:]=0
        for k in groups:pd.testing.assert_frame_equal(groups[k],gg[k])
        pd.testing.assert_frame_equal(y,yy)
        for k in masks:assert_array_equal(masks[k],mm[k])

    def test_convergence_and_stationarity_failure_stop_after_one_attempt(self):
        def warning(*args,**kwargs):warnings.warn('synthetic nonconvergence',ConvergenceWarning)
        with patch.object(LogisticRegression,'fit',side_effect=warning) as fit:
            with self.assertRaisesRegex(ValueError,'convergence'):
                fit_regularized_direction_family(self.groups,self.outcomes,**self.masks)
        self.assertEqual(fit.call_count,1)
        original=LogisticRegression.fit;calls=[]
        def corrupt(model,*args,**kwargs):
            calls.append(1);original(model,*args,**kwargs);model.intercept_[0]+=.1;return model
        with patch.object(LogisticRegression,'fit',corrupt),self.assertRaisesRegex(ValueError,'stationarity'):
            fit_regularized_direction_family(self.groups,self.outcomes,**self.masks)
        self.assertEqual(len(calls),1)

    def test_scalar_predictor_disagreement_is_rejected(self):
        original=Pipeline.decision_function
        def changed(model,*args,**kwargs):return original(model,*args,**kwargs)+.001
        with patch.object(Pipeline,'decision_function',changed),self.assertRaisesRegex(ValueError,'predictor parity'):
            fit_regularized_direction_family(self.groups,self.outcomes,**self.masks)
        original_probability=Pipeline.predict_proba
        def changed_probability(model,*args,**kwargs):
            result=original_probability(model,*args,**kwargs)
            result[:,0]-=1e-5;result[:,1]+=1e-5
            return result
        with patch.object(Pipeline,'predict_proba',changed_probability),self.assertRaisesRegex(ValueError,'predictor parity'):
            fit_regularized_direction_family(self.groups,self.outcomes,**self.masks)


if __name__ == '__main__':
    unittest.main()
