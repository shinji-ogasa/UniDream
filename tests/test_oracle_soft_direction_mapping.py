"""Synthetic-only mapping checks; no market data, fitting or policy execution."""
import inspect
import json
import math
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from unidream.experiments.oracle_soft_direction_mapping import (
    MEAN_IDS, PRIOR_IDENTITY_ATOL, PRIOR_IDENTITY_RTOL, map_soft_direction,
)


class Poison:
    def __float__(self):
        raise AssertionError("unselected value was converted")


def call(q, mask=None, **changes):
    if mask is None:
        mask = np.ones(len(q), dtype=bool)
    arguments = {"inference_mask": mask, "fit_abs_return_mean": .04,
                 "saved_weighted_prior_probability": .625, "fit_return_mean": .01}
    return map_soft_direction(q, **{**arguments, **changes})


class SoftDirectionMappingTests(unittest.TestCase):
    def test_hand_computed_mean_and_all_three_constant_controls(self):
        r=call([0.,.25,.5,.75,1.])
        self.assertEqual(tuple(r['means']),MEAN_IDS)
        assert_array_equal(r['means']['soft'],[-.04,-.02,0.,.02,.04])
        assert_array_equal(r['means']['mapped_prior'],np.full(5,.01))
        assert_array_equal(r['means']['fit_mean'],np.full(5,.01))
        assert_array_equal(r['means']['zero'],np.zeros(5))
        self.assertEqual(r['diagnostic']['probability_half_rows'],1)
        self.assertEqual(r['diagnostic']['prior_identity_signed_difference'],0.)
        self.assertFalse(r['diagnostic']['probabilities_recomputed'])
        self.assertFalse(r['diagnostic']['future_outcomes_or_score_support_used'])
        self.assertEqual((r['diagnostic']['model_fits'],r['diagnostic']['calibration_fits']),(0,0))
        json.dumps(r['diagnostic'],allow_nan=False)

    def test_masked_poison_and_invalid_probabilities_are_not_inspected_or_filled(self):
        mask=np.array([True,False,True,False,False,False,False])
        q=np.array([.25,Poison(),.75,np.inf,complex(1,2),True,'invalid'],dtype=object)
        r=call(q,mask)
        for name in MEAN_IDS:
            assert_array_equal(np.isfinite(r['means'][name]),mask)
        assert_array_equal(r['means']['soft'][mask],[-.02,.02])
        changed=q.copy();changed[~mask]=[np.nan,-100.,None,False,Poison()]
        other=call(changed,mask)
        for name in MEAN_IDS:assert_array_equal(r['means'][name],other['means'][name])
        self.assertEqual(r['diagnostic'],other['diagnostic'])

    def test_prior_identity_keeps_rounding_difference_and_stored_value_exact(self):
        amplitude=.5;prior=.6;mapped=amplitude*(2.*prior-1.)
        saved=np.nextafter(mapped,np.inf)
        r=call([prior,.5],fit_abs_return_mean=amplitude,
               saved_weighted_prior_probability=prior,fit_return_mean=saved)
        assert_array_equal(r['means']['mapped_prior'],np.full(2,mapped))
        assert_array_equal(r['means']['fit_mean'],np.full(2,saved))
        self.assertNotEqual(mapped,saved)
        d=r['diagnostic']
        self.assertEqual(d['prior_identity_signed_difference'],mapped-saved)
        self.assertEqual(d['prior_identity_absolute_difference'],abs(mapped-saved))
        self.assertEqual(d['prior_identity_tolerance'],PRIOR_IDENTITY_ATOL+PRIOR_IDENTITY_RTOL*abs(saved))
        self.assertTrue(d['prior_identity_passed'])
        with self.assertRaisesRegex(ValueError,'identity tolerance'):
            call([prior],fit_abs_return_mean=amplitude,saved_weighted_prior_probability=prior,fit_return_mean=saved+1e-6)

    def test_both_fixed_absolute_and_relative_identity_tolerance_terms(self):
        call([.5],fit_abs_return_mean=1.,saved_weighted_prior_probability=.5,fit_return_mean=PRIOR_IDENTITY_ATOL)
        with self.assertRaisesRegex(ValueError,'identity tolerance'):
            call([.5],fit_abs_return_mean=1.,saved_weighted_prior_probability=.5,fit_return_mean=2*PRIOR_IDENTITY_ATOL)
        # Large finite means require the fixed relative term; no tolerance fitting.
        call([1.],fit_abs_return_mean=1e6,saved_weighted_prior_probability=1.,fit_return_mean=1e6+5e-7)
        with self.assertRaisesRegex(ValueError,'identity tolerance'):
            call([1.],fit_abs_return_mean=1e6,saved_weighted_prior_probability=1.,fit_return_mean=1e6+2e-6)

    def test_saved_prior_probability_is_used_despite_statistical_prior_rounding(self):
        statistical_prior=.6; saved_prior=np.nextafter(statistical_prior,1.)
        amplitude=.5; fit_mean=amplitude*(2.*statistical_prior-1.)
        expected=amplitude*(2.*saved_prior-1.)
        r=call([saved_prior],fit_abs_return_mean=amplitude,
               saved_weighted_prior_probability=saved_prior,fit_return_mean=fit_mean)
        self.assertNotEqual(expected,fit_mean)
        self.assertEqual(r['means']['soft'][0],expected)
        self.assertEqual(r['means']['mapped_prior'][0],expected)
        self.assertEqual(r['means']['fit_mean'][0],fit_mean)
        self.assertEqual(r['diagnostic']['saved_weighted_prior_probability'],saved_prior)
        self.assertEqual(r['diagnostic']['prior_identity_signed_difference'],expected-fit_mean)

    def test_tie_and_nearest_floats_use_saved_probability_without_epsilon(self):
        q=np.array([np.nextafter(.5,0.),.5,np.nextafter(.5,1.)])
        r=call(q,fit_abs_return_mean=1.,saved_weighted_prior_probability=.5,fit_return_mean=-0.)
        expected=2.*q-1.
        assert_array_equal(r['means']['soft'],expected)
        self.assertLess(r['means']['soft'][0],0.);self.assertGreater(r['means']['soft'][2],0.)
        self.assertEqual(r['means']['soft'][1],0.)
        self.assertFalse(np.signbit(r['means']['soft'][1]))
        self.assertTrue(np.signbit(r['means']['fit_mean']).all())
        self.assertFalse(np.signbit(r['means']['mapped_prior']).any())

    def test_extreme_finite_amplitudes_are_bounded_without_overflow_or_clipping(self):
        q=np.array([0.,.125,.5,.875,1.])
        for amplitude in (np.finfo(float).max,np.nextafter(0.,1.)):
            r=call(q,fit_abs_return_mean=amplitude,saved_weighted_prior_probability=.5,fit_return_mean=0.)
            mu=r['means']['soft']
            self.assertTrue(np.isfinite(mu).all());self.assertTrue((np.abs(mu)<=amplitude).all())
            self.assertEqual(mu[0],-amplitude);self.assertEqual(mu[-1],amplitude)
            self.assertEqual(mu[2],0.)
        with self.assertRaisesRegex(ValueError,'identity tolerance'):
            call([1.],fit_abs_return_mean=np.finfo(float).max,saved_weighted_prior_probability=1.,
                 fit_return_mean=-np.finfo(float).max)

    def test_probability_and_scalar_types_ranges_and_shapes_are_strict(self):
        for value in (True,np.bool_(False),1+0j,'0.5',Poison(),None,np.nan,np.inf,-np.inf,-.01,1.01):
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):call([.25,value])
        for name in ('fit_abs_return_mean','saved_weighted_prior_probability','fit_return_mean'):
            for value in (True,np.bool_(False),1+0j,'0.5',Poison(),None,np.nan,np.inf,np.array(.5),[.5]):
                with self.subTest(name=name,value=repr(value)),self.assertRaises(ValueError):
                    call([.5],**{name:value})
        for amplitude in (0.,-0.,-.1):
            with self.assertRaises(ValueError):call([.5],fit_abs_return_mean=amplitude)
        for prior in (-.01,1.01):
            with self.assertRaises(ValueError):call([.5],saved_weighted_prior_probability=prior)
        for q in (np.ones((2,1)),[.5],.5):
            with self.assertRaises(ValueError):call(q,np.ones(2,bool))

    def test_strict_nonempty_aligned_masks_and_no_scoring_or_fit_parameters(self):
        masks=([],np.zeros(2,bool),np.ones(3,bool),np.ones((2,1),bool),
               np.ones(2,int),np.ones(2,dtype=object),[True,1])
        for mask in masks:
            with self.subTest(mask=repr(mask)),self.assertRaises(ValueError):call([.25,.75],mask)
        self.assertEqual(tuple(inspect.signature(map_soft_direction).parameters),
                         ('probabilities','inference_mask','fit_abs_return_mean','saved_weighted_prior_probability','fit_return_mean'))
        for name in ('actual','score_support','weight','temperature'):
            with self.assertRaises(TypeError):call([.5],**{name:1.})

    def test_future_prefix_invariance_and_inputs_and_result_arrays_do_not_alias(self):
        q=np.array([.25,.5,.75,np.nan]);mask=np.array([True,True,True,False])
        old_q=q.copy();old_mask=mask.copy();q.setflags(write=False);mask.setflags(write=False)
        short=call(q,mask)
        long=call(np.r_[q,.9,.1],np.r_[mask,True,True])
        for name in MEAN_IDS:assert_array_equal(short['means'][name],long['means'][name][:4])
        short['means']['soft'][0]=123.;short['inference_mask'][:]=False
        self.assertEqual(short['means']['mapped_prior'][0],.01)
        self.assertEqual(short['means']['fit_mean'][0],.01)
        self.assertEqual(short['means']['zero'][0],0.)
        assert_array_equal(q,old_q);assert_array_equal(mask,old_mask)
        for left in MEAN_IDS:
            for right in MEAN_IDS:
                if left!=right:self.assertFalse(np.shares_memory(short['means'][left],short['means'][right]))


if __name__ == '__main__':
    unittest.main()
