import copy
import unittest
import numpy as np
from unidream.experiments.oracle_sign_magnitude_decisions import (
    FIXED,SOURCES,PARENT_ROOT,FOLDS,POLICIES,SCORE_MEANS,SUBSETS,CELLS,NEW_IDS,
    validate_config,tail_groups,score_forecast,summarize,
)

class SignMagnitudeDecisionTests(unittest.TestCase):
    def config(self):
        return {**copy.deepcopy(FIXED),'source_bindings':{p:'0'*64 for p in SOURCES},
            'parent_config_sha256':'0'*64,'parity_config_sha256':'0'*64,'preflight_sha256':'0'*64,
            'parent_manifest_bindings':{str(PARENT_ROOT/n):'0'*64 for n in
                ['registration.json','results.json','preflight.json']+[f'fold_{f}.json' for f in FOLDS]}}

    def test_predeclared_original_support_and_no_new_causal_winner(self):
        c=self.config();validate_config(c)
        for k,v in [('tail_quantile',.95),('teacher_use_allowed',True),('development_folds',[15]),
                    ('new_model_fits',1),('components',['sign']),('adaptive_causal_names_unchanged',182)]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_config({**c,k:v})
        c['parent_manifest_bindings'].pop(next(iter(c['parent_manifest_bindings'])))
        with self.assertRaises(ValueError):validate_config(c)

    def test_tail_threshold_is_fit_only_and_equality_is_large(self):
        fit=np.tile([0.,1.],256);y=np.array([[0.,np.nan,np.inf],[1.,0,0],[3.,0,0],[np.nan,0,0]])
        mask=np.array([True,True,True,False]);q,groups=tail_groups(fit,y,mask)
        self.assertEqual(q,1.);np.testing.assert_array_equal(groups['fit_q90_large'],[False,True,True,False])
        poisoned=y.copy();poisoned[:3,0]=[-8.,0,5.];poisoned[3,0]=1e300
        self.assertEqual(tail_groups(fit,poisoned,mask)[0],q)
        objects=y.astype(object);objects[:,1]='ignored';objects[:,2]=object()
        self.assertEqual(tail_groups(fit,objects,mask)[0],q)
        _,empty=tail_groups(np.zeros(512),y,mask)
        self.assertFalse(empty['fit_q90_other'].any())
        out=score_forecast({'actual':y,'fit_return_mean':0.},np.zeros(4),empty['fit_q90_other'])
        self.assertEqual(out['rows'],0);self.assertIsNone(out['return_mse'])

    def family(self):
        rows=[];scores=[]
        for f in FOLDS:
            regime={'trend':('bull' if f<7 else 'bear' if f<11 else 'sideways')}
            for cid in POLICIES:
                x={'alpha_ex':.1,'maxdd_delta':-.02,'turnover':1.,'trades':2}
                rows.append({'fold':f,'candidate_id':cid,'regime':regime,'base':x,'stress_2x':x})
            for m in SCORE_MEANS:
                cell=next(c for cells in CELLS.values() for c,v in cells.items() if v==m)
                value={'base':5.,'sign':1.,'magnitude':4.,'full':0.}[cell]
                for u in SUBSETS:
                    scores.append({'fold':f,'mean_id':m,'regime':regime,'subset':u,'rows':4 if u=='all' else 2,
                        'return_mse':value,'return_mae':1.,'return_sign_accuracy':.5,'return_rank_ic':None,'zero_return_mse':10.,'fit_mean_return_mse':10.})
        return rows,scores

    def test_complete_factorial_contrasts_and_empty_subgroups_are_not_dropped(self):
        rows,scores=self.family();s=summarize(rows,scores)
        for h in CELLS:
            self.assertEqual(s['paired']['all'][h]['sign_minus_base']['mse']['all'],-4.)
            self.assertEqual(s['interaction']['all'][h]['mse']['all'],0.)
        self.assertFalse(s['selection_performed']);self.assertTrue(s['hindsight_diagnostic_not_causal_accuracy'])
        for x in scores:
            if x['fold']==5 and x['subset']=='fit_q90_other':
                x.update(rows=0,return_mse=None,return_mae=None,return_sign_accuracy=None,return_rank_ic=None,zero_return_mse=None,fit_mean_return_mse=None)
            if x['fold']==5 and x['subset']=='fit_q90_large':x['rows']=4
        s=summarize(rows,scores)
        for m in SCORE_MEANS:
            x=s['prediction']['all'][m]['fit_q90_other'];self.assertIsNone(x['return_mse']);self.assertEqual(x['nonempty_quarters'],7)
            self.assertIsNotNone(x['pooled_row_mse']);self.assertEqual(x['defined_rank_quarters'],0)
        for h in CELLS:self.assertIsNone(s['paired']['all'][h]['sign_minus_base']['mse']['fit_q90_other'])

    def test_missing_counterfactual_or_changed_regime_is_rejected(self):
        rows,scores=self.family()
        for rr,ss in [(rows[:-1],scores),(rows+rows[:1],scores),(rows,scores[:-1]),(rows,scores+scores[:1])]:
            with self.assertRaises(ValueError):summarize(rr,ss)
        rows[-1]['regime']={'trend':'bull'}
        with self.assertRaises(ValueError):summarize(rows,scores)
        rows,scores=self.family();scores[-1]['rows']=1
        with self.assertRaises(ValueError):summarize(rows,scores)

    def test_nonfinite_asymmetric_nulls_and_relabelled_inventory_fail(self):
        for field,value in [('return_mse',None),('return_mse',float('inf')),('return_mae',-1.),
                            ('return_sign_accuracy',1.1),('return_rank_ic',float('nan')),('rows',True)]:
            rows,scores=self.family();scores[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):summarize(rows,scores)
        rows,scores=self.family()
        for row in rows+scores:row['regime']={'trend':'bull'}
        with self.assertRaises(ValueError):summarize(rows,scores)

if __name__=='__main__':unittest.main()
