import copy
import unittest
import numpy as np
from unidream.experiments.oracle_direction_decisions import (
    FIXED,SOURCES,PARENT_ROOT,FOLDS,POLICIES,MEANS,CLASSIFIERS,MODEL_IDS,NEW_MEANS,
    SEGMENTS,RULES,MAPPING,validate_config,map_direction,summarize,
)

class DirectionDecisionTests(unittest.TestCase):
    def config(self):
        return {**copy.deepcopy(FIXED),'source_bindings':{p:'0'*64 for p in SOURCES},
            'parent_config_sha256':'0'*64,'parity_config_sha256':'0'*64,'preflight_sha256':'0'*64,
            'parent_manifest_bindings':{str(PARENT_ROOT/n):'0'*64 for n in
                ['registration.json','results.json','preflight.json']+[f'fold_{f}.json' for f in FOLDS]}}

    def test_frozen_inventory_and_no_adaptive_tuning(self):
        c=self.config();validate_config(c)
        self.assertEqual(len(POLICIES),52);self.assertEqual(len(MEANS),10);self.assertEqual(len(CLASSIFIERS),6)
        self.assertEqual(len(SOURCES),29);self.assertEqual(len(set(SOURCES)),29)
        for k,v in [('C',2.),('teacher_use_allowed',True),('development_folds',[15]),('new_model_fits',16),
                    ('group_dimensions',[34,31]),('adaptive_total_causal_names',174),('selection_permitted',True)]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_config({**c,k:v})
        c['parent_manifest_bindings'].pop(next(iter(c['parent_manifest_bindings'])))
        with self.assertRaises(ValueError):validate_config(c)

    def test_direction_mapping_uses_all_inference_and_preserves_magnitude(self):
        z=np.array([1.,-1.,0.,2.,np.nan]);mu=np.array([-.2,.3,-.4,0.,np.inf]);mask=np.array([1,1,1,1,0],bool)
        got=map_direction(z,mu,mask)
        np.testing.assert_array_equal(got[:4],[.2,-.3,0.,0.]);self.assertTrue(np.isnan(got[-1]))
        np.testing.assert_array_equal(mu,[-.2,.3,-.4,0.,np.inf])
        # No future labels or score-support argument exists: every inference row is replaced.
        z[-1]=1e300;mu[-1]=-1e300;np.testing.assert_array_equal(map_direction(z,mu,mask),got)
        for zz,mm in [(np.array([1.,-1.,np.nan,2.,3.]),mask),(z,mask.astype(int)),(z,np.zeros(5,bool))]:
            with self.assertRaises(ValueError):map_direction(zz,mu,mm)

    def family(self):
        rows=[];scores=[];cs=[]
        for f in FOLDS:
            regime={'trend':'bull' if f<7 else 'bear' if f<11 else 'sideways'}
            for cid in POLICIES:
                x={'alpha_ex':.1,'maxdd_delta':-.02,'turnover':1.,'trades':2}
                rows.append({'fold':f,'candidate_id':cid,'regime':regime,'base':x.copy(),'stress_2x':x.copy()})
            for seg in SEGMENTS:
                for m in MEANS:
                    mse=1. if m.endswith('_direction') and not m.endswith('_prior_direction') else 2.
                    scores.append({'fold':f,'segment':seg,'mean_id':m,'regime':regime,'rows':20,
                        'return_mse':mse,'return_mae':1.,'return_sign_accuracy':.5,'return_rank_ic':None,
                        'zero_return_mse':3.,'fit_mean_return_mse':3.})
                for mid in CLASSIFIERS:
                    loss=.1 if mid in MODEL_IDS else .2
                    cs.append({'fold':f,'segment':seg,'classifier_id':mid,'regime':regime,'rows':20,
                        'log_loss':loss,'brier':loss,'binary_accuracy':.5,'signed_return_mean':.01,
                        'weighted_log_loss':loss,'weighted_brier':loss,'weighted_binary_accuracy':.5,
                        'absolute_return_sum':2.,'absolute_return_mean':.1,'zero_actual_rows':0,'zero_logit_rows':0})
        return rows,scores,cs

    def test_three_separate_gates_and_matched_counterfactuals(self):
        rows,scores,cs=self.family();s=summarize(rows,scores,cs)
        mid=MODEL_IDS[0];m=mid+'_direction';cid=m+'_'+RULES[0]
        self.assertTrue(s['direction'][cid]['economic_means_all_strata_both_costs'])
        self.assertTrue(all(s['direction'][cid]['matched_probability_losses_improved_all_strata'].values()))
        self.assertTrue(all(s['direction'][cid]['mapped_mse_vs_zero_fitmean_parent_and_matched_prior_all_strata'].values()))
        self.assertFalse(s['high_probability_generalization_established']);self.assertFalse(s['selection_performed'])
        self.assertEqual(s['paired']['all'][m][MAPPING[m]['parent_mean']]['prediction']['evaluation']['mse_difference'],-1.)
        # One sideways stress sign failure must fail the economics gate while predictive flags remain true.
        for r in rows:
            if r['candidate_id']==cid and r['fold']>=11:r['stress_2x']['alpha_ex']=-.1
        s=summarize(rows,scores,cs);self.assertFalse(s['direction'][cid]['economic_means_all_strata_both_costs'])
        self.assertTrue(s['direction'][cid]['matched_probability_losses_improved_all_strata']['evaluation'])

    def test_weighted_null_quarter_is_retained_and_fails_weighted_gate(self):
        rows,scores,cs=self.family()
        for r in cs:
            if r['fold']==5 and r['segment']=='interval':
                r.update(absolute_return_sum=0.,absolute_return_mean=0.,zero_actual_rows=20,
                    weighted_log_loss=None,weighted_brier=None,weighted_binary_accuracy=None)
        s=summarize(rows,scores,cs);mid='technical_magnitude';cid=mid+'_direction_'+RULES[0]
        self.assertIsNone(s['classification']['all']['interval'][mid]['weighted_log_loss'])
        self.assertEqual(s['classification']['all']['interval'][mid]['quarters'],8)
        self.assertFalse(s['direction'][cid]['matched_probability_losses_improved_all_strata']['interval'])
        self.assertTrue(s['direction']['technical_ordinary_direction_'+RULES[0]]['matched_probability_losses_improved_all_strata']['interval'])

    def test_missing_duplicate_or_unpaired_records_fail(self):
        rows,scores,cs=self.family()
        for a,b,c in [(rows[:-1],scores,cs),(rows+rows[:1],scores,cs),(rows,scores[:-1],cs),
                      (rows,scores,cs[:-1]),(rows,scores,cs+cs[:1])]:
            with self.assertRaises(ValueError):summarize(a,b,c)
        for family,key,value in [(0,'regime',{'trend':'bull'}),(1,'rows',19),(2,'absolute_return_sum',3.)]:
            inputs=self.family();inputs[family][-1][key]=value
            with self.assertRaises(ValueError):summarize(*inputs)

    def test_invalid_or_asymmetric_scores_cannot_be_silently_dropped(self):
        for family,key,value in [(1,'return_mse',None),(1,'return_mse',float('nan')),
            (1,'return_rank_ic',2.),(2,'log_loss',float('inf')),(2,'weighted_brier',None),
            (2,'binary_accuracy',1.1),(2,'rows',True),(2,'zero_logit_rows',-1)]:
            inputs=self.family();inputs[family][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):summarize(*inputs)
        rows,scores,cs=self.family()
        for r in rows+scores+cs:r['regime']={'trend':'bull'}
        with self.assertRaises(ValueError):summarize(rows,scores,cs)

if __name__=='__main__':unittest.main()
