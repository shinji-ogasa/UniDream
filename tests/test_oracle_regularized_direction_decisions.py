import copy
import unittest
import numpy as np
import pandas as pd
from unidream.experiments.oracle_regularized_direction_decisions import (
    FIXED,SOURCES,PARENT_ROOT,FOLDS,POLICIES,MEANS,CLASSIFIERS,MODEL_IDS,NEW_MEANS,
    SEGMENTS,RULES,MAPPING,PARTS,OLD_MODEL_IDS,REFERENCES,
    validate_config,reconstruct_masks,fit_inputs,summarize,
)

class RegularizedDirectionDecisionTests(unittest.TestCase):
    def config(self):
        return {**copy.deepcopy(FIXED),'source_bindings':{p:'0'*64 for p in SOURCES},
            'parent_config_sha256':'0'*64,'preflight_sha256':'0'*64,
            'parent_manifest_bindings':{str(PARENT_ROOT/n):'0'*64 for n in
                ['registration.json','results.json','preflight.json']+[f'fold_{f}.json' for f in FOLDS]}}

    def test_frozen_single_schedule_inventory_and_no_new_priors(self):
        c=self.config();validate_config(c)
        self.assertEqual((len(POLICIES),len(MEANS),len(CLASSIFIERS),len(SOURCES)),(60,14,10,31))
        self.assertEqual(len(set(SOURCES)),31)
        for k,v in [('regularization_C','1/n'),('new_unique_fit_priors',16),('new_model_fits',64),
            ('teacher_use_allowed',True),('development_folds',[15]),('adaptive_total_causal_names',190),
            ('selection_permitted',True),('group_dimensions',[34,31]),('scalar_logit_atol',1e-6)]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_config({**c,k:v})
        c['parent_manifest_bindings'].pop(next(iter(c['parent_manifest_bindings'])))
        with self.assertRaises(ValueError):validate_config(c)

    def snapshots(self):
        index=pd.date_range('2020-01-01',periods=10,freq='15min',tz='UTC')
        fp=np.array([0,1]);pp=np.array([3,5,7,8,9])
        pack={'fit_positions':fp,'predict_positions':pp,'timestamps':index.asi8[fp],
            'predict_timestamps':index.asi8[pp],'returns':np.array([.1,-.2])}
        for g in ['technical','perp_delay0']:
            pack['fit_features_'+g]=np.arange(4.).reshape(2,2)
            pack['predict_features_'+g]=np.arange(10.).reshape(5,2)
        cal={'timestamps':index.asi8[3:7],'scale_mask':np.array([1,0,0,0],bool),'interval_mask':np.array([0,0,1,0],bool)}
        e={'timestamps':index.asi8[7:],'inference_mask':np.ones(3,bool),'score_support':np.array([1,1,0],bool)}
        return index,pack,cal,e

    def test_snapshot_reconstruction_retains_all_six_masks_and_unscored_origin(self):
        index,pack,cal,e=self.snapshots();m=reconstruct_masks(index,pack,cal,e)
        self.assertEqual({k:int(v.sum()) for k,v in m.items()},dict(fit=2,predict=5,scale=1,interval=1,inference=3,score=2))
        self.assertTrue(m['predict'][-1] and m['inference'][-1] and not m['score'][-1])
        for key,val in [('fit_positions',np.array([-1,1])),('predict_positions',np.array([3,3,7,8,9])),
                        ('predict_timestamps',pack['predict_timestamps']+1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):reconstruct_masks(index,{**pack,key:val},cal,e)
        cal['scale_mask'][1]=True
        with self.assertRaises(ValueError):reconstruct_masks(index,pack,cal,e)

    def test_new_fit_input_has_no_nonfit_or_auxiliary_outcomes(self):
        index,pack,cal,e=self.snapshots();m=reconstruct_masks(index,pack,cal,e)
        pack['future_actuals']=np.full((10,3),1e300)
        frames,y=fit_inputs(index,pack,m,{g:['a','b'] for g in ['technical','perp_delay0']})
        np.testing.assert_array_equal(y[m['fit'],0],pack['returns'])
        self.assertTrue(np.isnan(y[~m['fit']]).all());self.assertTrue(np.isnan(y[:,1:]).all())
        for g,a in frames.items():
            self.assertTrue(a.index.equals(index));np.testing.assert_array_equal(a[m['fit']],pack['fit_features_'+g])
            np.testing.assert_array_equal(a[m['predict']],pack['predict_features_'+g])
            self.assertTrue(np.isnan(a[~(m['fit']|m['predict'])]).all().all())

    def family(self):
        rows=[];scores=[];cs=[]
        for f in FOLDS:
            regime={'trend':'bull' if f<7 else 'bear' if f<11 else 'sideways'}
            for cid in POLICIES:
                new=any(cid.startswith(m+'_') for m in NEW_MEANS)
                x={'alpha_ex':.2 if new else .1,'maxdd_delta':-.03 if new else -.02,'turnover':1.,'trades':2}
                rows.append({'fold':f,'candidate_id':cid,'regime':regime,'base':x.copy(),'stress_2x':x.copy()})
            for seg in SEGMENTS:
                for m in MEANS:
                    scores.append({'fold':f,'segment':seg,'mean_id':m,'regime':regime,'rows':20,
                        'return_mse':1. if m in NEW_MEANS else 2.,'return_mae':1.,'return_sign_accuracy':.5,'return_rank_ic':None,
                        'zero_return_mse':3.,'fit_mean_return_mse':3.})
                for mid in CLASSIFIERS:
                    loss=.1 if mid in MODEL_IDS else .2 if mid in OLD_MODEL_IDS else .3
                    cs.append({'fold':f,'segment':seg,'classifier_id':mid,'regime':regime,'rows':20,
                        'log_loss':loss,'brier':loss,'binary_accuracy':.5,'signed_return_mean':.01,
                        'weighted_log_loss':loss,'weighted_brier':loss,'weighted_binary_accuracy':.5,
                        'absolute_return_sum':2.,'absolute_return_mean':.1,'zero_actual_rows':0,'zero_logit_rows':0})
        return rows,scores,cs

    def test_absolute_paired_probability_and_return_gates_are_separate(self):
        data=self.family();s=summarize(*data);mid=MODEL_IDS[0];m=mid+'_direction';cid=m+'_'+RULES[0]
        d=s['direction'][cid]
        self.assertTrue(d['economic_means_all_strata_both_costs']);self.assertTrue(d['economic_improvement_vs_all_references_all_strata_both_costs'])
        self.assertTrue(all(d['matched_probability_losses_vs_C1_and_prior_improved_all_strata'].values()))
        self.assertTrue(all(d['mapped_mse_vs_zero_fitmean_and_all_references_improved_all_strata'].values()))
        self.assertEqual(set(s['paired']['all'][m]),set(REFERENCES[m]));self.assertFalse(s['high_probability_generalization_established'])
        for r in data[0]:
            if r['candidate_id']==cid and r['fold']>=11:r['stress_2x']['alpha_ex']=.05
        d=summarize(*data)['direction'][cid]
        self.assertTrue(d['economic_means_all_strata_both_costs']);self.assertFalse(d['economic_improvement_vs_all_references_all_strata_both_costs'])
        self.assertTrue(d['matched_probability_losses_vs_C1_and_prior_improved_all_strata']['evaluation'])

    def test_beating_only_prior_or_half_cannot_pass_strong_reference_gates(self):
        data=self.family();mid=MODEL_IDS[0];m=mid+'_direction';cid=m+'_'+RULES[0]
        for r in data[2]:
            if r['classifier_id']==PARTS[mid]['old_classifier'] and r['segment']=='evaluation':r['brier']=.05
        for r in data[1]:
            if r['mean_id']==MAPPING[m]['old_mean'] and r['segment']=='interval':r['return_mse']=.5
        d=summarize(*data)['direction'][cid]
        self.assertFalse(d['matched_probability_losses_vs_C1_and_prior_improved_all_strata']['evaluation'])
        self.assertTrue(d['matched_probability_losses_vs_C1_and_prior_improved_all_strata']['interval'])
        self.assertFalse(d['mapped_mse_vs_zero_fitmean_and_all_references_improved_all_strata']['interval'])

    def test_weighted_null_quarter_retained_without_changing_ordinary_gate(self):
        data=self.family()
        for r in data[2]:
            if r['fold']==5 and r['segment']=='interval':r.update(absolute_return_sum=0.,absolute_return_mean=0.,zero_actual_rows=20,
                weighted_log_loss=None,weighted_brier=None,weighted_binary_accuracy=None)
        s=summarize(*data)
        self.assertIsNone(s['classification']['all']['interval']['technical_magnitude_l2unit']['weighted_log_loss'])
        key='matched_probability_losses_vs_C1_and_prior_improved_all_strata'
        self.assertFalse(s['direction']['technical_magnitude_l2unit_direction_'+RULES[0]][key]['interval'])
        self.assertTrue(s['direction']['technical_ordinary_l2unit_direction_'+RULES[0]][key]['interval'])

    def test_inventory_regime_score_support_and_nonfinite_fail_closed(self):
        for i in range(3):
            data=self.family();data[i].pop()
            with self.assertRaises(ValueError):summarize(*data)
            data=self.family();data[i].append(data[i][0])
            with self.assertRaises(ValueError):summarize(*data)
        for i,key,val in [(0,'regime',{'trend':'bull'}),(1,'rows',19),(1,'return_mse',float('nan')),
                        (2,'rows',True),(2,'weighted_log_loss',None),(2,'zero_actual_rows',21)]:
            data=self.family();data[i][-1][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):summarize(*data)

if __name__=='__main__':unittest.main()
