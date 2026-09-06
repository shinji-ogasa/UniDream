import copy
import unittest
import numpy as np
from unidream.experiments.oracle_soft_direction_decisions import (
    FIXED,SOURCES,PARENT_ROOT,FOLDS,CONTROLS,POLICIES,MEANS,OLD_MEANS,CLASSIFIERS,NEW_MEANS,
    SOFT_MEANS,CONSTANT_MEANS,LEARNED_IDS,SEGMENTS,RULES,MAPPING,REFERENCES,PROBABILITY_IDS,
    validate_config,map_saved_inputs,summarize,exact_tree,
)


class SoftDirectionDecisionTests(unittest.TestCase):
    def config(self):
        return {**copy.deepcopy(FIXED),'source_bindings':{p:'0'*64 for p in SOURCES},
            'parent_config_sha256':'0'*64,'preflight_sha256':'0'*64,
            'parent_manifest_bindings':{str(PARENT_ROOT/n):'0'*64 for n in
                ['registration.json','results.json','preflight.json']+[f'fold_{f}.json' for f in FOLDS]}}

    def test_fixed_inventory_no_fit_and_no_candidate_deletion(self):
        c=self.config();validate_config(c)
        self.assertEqual((len(POLICIES),len(MEANS),len(CLASSIFIERS),len(SOURCES)),(80,24,10,33))
        self.assertEqual((len(SOFT_MEANS),len(CONSTANT_MEANS),len(LEARNED_IDS)),(4,6,8))
        self.assertEqual(len(set(SOURCES)),33)
        for k,v in [('new_model_fits',1),('new_unique_fit_priors',1),('new_causal_names',8),
            ('surrogate_mean','a*tanh(z/2)'),('prior_probability_source','raw_fit_prior'),
            ('development_folds',[15]),('selection_permitted',True),('teacher_use_allowed',True),
            ('adaptive_total_causal_names',206),('prior_identity_absolute_tolerance',1e-6)]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_config({**c,k:v})
        for m in SOFT_MEANS:self.assertEqual(len(REFERENCES[m]),5)
        c['parent_manifest_bindings'].pop(next(iter(c['parent_manifest_bindings'])))
        with self.assertRaises(ValueError):validate_config(c)

    def test_mapping_boundary_never_reads_outcomes_or_score_mask_and_preserves_unscored(self):
        class OnlyInputs(dict):
            def __getitem__(self,k):
                if k not in ('probability','inference_mask'):raise AssertionError('forbidden mapping input '+k)
                return super().__getitem__(k)
        a=OnlyInputs(probability=np.array([.75,.25,.5,.625,np.nan]),inference_mask=np.array([1,1,1,1,0],bool))
        st={'fit_abs_return_mean':.125,'fit_return_mean':.03125,'prior_probability':{'technical':.625}}
        mapped=map_saved_inputs(a,st,'technical','inference_mask')['means']
        np.testing.assert_array_equal(mapped['soft'][:4],[.0625,-.0625,0.,.03125])
        self.assertTrue(np.isnan(mapped['soft'][-1]));self.assertEqual(mapped['soft'][2],0.)
        self.assertTrue(np.isfinite(mapped['soft'][3])) # This may be a future-unscored origin.

    def family(self):
        rows=[];scores=[];cs=[]
        for f in FOLDS:
            regime={'trend':'bull' if f<7 else 'bear' if f<11 else 'sideways'}
            for cid in POLICIES:
                x={'alpha_ex':.3 if cid in LEARNED_IDS else .1,'maxdd_delta':-.04 if cid in LEARNED_IDS else -.02,'turnover':1.,'trades':2}
                rows.append({'fold':f,'candidate_id':cid,'regime':regime,'base':x.copy(),'stress_2x':x.copy()})
            for seg in SEGMENTS:
                for m in MEANS:
                    scores.append({'fold':f,'segment':seg,'mean_id':m,'regime':regime,'rows':20,
                        'return_mse':1. if m in SOFT_MEANS else 3. if m in CONSTANT_MEANS else 2.,
                        'return_mae':1.,'return_sign_accuracy':.5,'return_rank_ic':None,
                        'zero_return_mse':3.,'fit_mean_return_mse':3.})
                for mid in CLASSIFIERS:
                    loss=.3 if mid in PROBABILITY_IDS else .2
                    cs.append({'fold':f,'segment':seg,'classifier_id':mid,'regime':regime,'rows':20,
                        'log_loss':loss,'brier':loss,'binary_accuracy':.5,'signed_return_mean':.01,
                        'weighted_log_loss':loss,'weighted_brier':loss,'weighted_binary_accuracy':.5,
                        'absolute_return_sum':2.,'absolute_return_mean':.1,'zero_actual_rows':0,'zero_logit_rows':0})
        return rows,scores,cs

    def test_mapping_predictive_and_economic_gains_do_not_claim_probability_gain(self):
        d=self.family();s=summarize(*d);m=SOFT_MEANS[0];cid=m+'_'+RULES[0];g=s['soft'][cid]
        self.assertTrue(g['economic_means_all_strata_both_costs'])
        self.assertTrue(g['economic_improvement_vs_all_five_references_all_strata_both_costs'])
        self.assertTrue(all(g['mapped_mse_vs_all_five_references_improved_all_strata'].values()))
        self.assertFalse(any(g['inherited_source_weighted_losses_below_prior_all_strata'].values()))
        self.assertFalse(g['new_probability_accuracy_improvement']);self.assertFalse(s['high_probability_generalization_established'])
        self.assertEqual(set(s['paired']['all'][m]),set(REFERENCES[m]))
        self.assertEqual(len(s['soft']),8)
        self.assertIsNone(s['prediction']['all']['evaluation'][m]['return_rank_ic'])

    def test_every_reference_including_zero_and_fitmean_can_falsify_paired_economics(self):
        m=SOFT_MEANS[0];cid=m+'_'+RULES[0]
        for ref in REFERENCES[m]:
            d=self.family()
            for r in d[0]:
                if r['candidate_id']==ref+'_'+RULES[0] and r['fold']>=11:r['stress_2x']['alpha_ex']=.4
            g=summarize(*d)['soft'][cid]
            self.assertTrue(g['economic_means_all_strata_both_costs'])
            self.assertFalse(g['economic_improvement_vs_all_five_references_all_strata_both_costs'])
            self.assertTrue(g['mapped_mse_vs_all_five_references_improved_all_strata']['evaluation'])

    def test_mse_strict_tie_in_one_stratum_segment_fails_only_that_segment(self):
        m=SOFT_MEANS[0];cid=m+'_'+RULES[0]
        for ref in REFERENCES[m]:
            d=self.family()
            for r in d[1]:
                if r['fold']>=11 and r['segment']=='interval':
                    if r['mean_id']==ref:r['return_mse']=1.
                    if ref.endswith('_soft_zero'):
                        r['zero_return_mse']=1.
                        if r['mean_id'].endswith('_soft_zero'):r['return_mse']=1.
                    if ref.endswith('_soft_fit_mean'):
                        r['fit_mean_return_mse']=1.
                        if r['mean_id'].endswith('_soft_fit_mean'):r['return_mse']=1.
            g=summarize(*d)['soft'][cid]
            self.assertFalse(g['mapped_mse_vs_all_five_references_improved_all_strata']['interval'])
            self.assertTrue(g['mapped_mse_vs_all_five_references_improved_all_strata']['evaluation'])

    def test_preserved_tree_must_be_exact_not_merely_within_old_parity_tolerance(self):
        exact_tree({'x':.2},{'x':.2},name='synthetic')
        with self.assertRaises(ValueError):exact_tree({'x':.2+1e-15},{'x':.2},name='synthetic')

    def test_constant_reference_mse_and_undefined_rank_enforced(self):
        for kind,key,val in [('zero','return_mse',2.),('fit_mean','return_mse',2.),('mapped_prior','return_rank_ic',0.)]:
            d=self.family();r=next(x for x in d[1] if x['mean_id']=='technical_soft_'+kind);r[key]=val
            with self.assertRaises(ValueError):summarize(*d)

    def test_probability_weighted_null_is_retained_and_cannot_pass_inherited_gate(self):
        d=self.family()
        for r in d[2]:
            if r['fold']==5 and r['segment']=='interval':r.update(absolute_return_sum=0.,absolute_return_mean=0.,zero_actual_rows=20,
                weighted_log_loss=None,weighted_brier=None,weighted_binary_accuracy=None)
        s=summarize(*d)
        self.assertIsNone(s['classification']['all']['interval'][PROBABILITY_IDS[0]]['weighted_log_loss'])
        self.assertFalse(s['soft'][LEARNED_IDS[0]]['inherited_source_weighted_losses_below_prior_all_strata']['interval'])

    def test_inventory_support_invalid_new_and_old_scores_fail_closed(self):
        for i in range(3):
            d=self.family();d[i].pop()
            with self.assertRaises(ValueError):summarize(*d)
            d=self.family();d[i].append(d[i][0])
            with self.assertRaises(ValueError):summarize(*d)
        for key,val in [('rows',19),('rows',True),('return_mse',float('nan')),('return_sign_accuracy',1.1),
                        ('return_rank_ic',2.),('zero_return_mse',2.),('fit_mean_return_mse',2.),('regime',{'trend':'bull'})]:
            d=self.family();d[1][-1][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):summarize(*d)


if __name__=='__main__':unittest.main()
