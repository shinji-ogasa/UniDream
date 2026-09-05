import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from unidream.experiments.oracle_additional_window_replay import (
    CANDIDATES, ECON_KEYS, FOLDS, MEANS, POLICIES, artifact_inventory,
    describe_additional_family, plan, validate_config, validate_completed_fold,
)


def fixture(regimes=None):
    regimes=regimes or ['bull']*4+['bear']*3+['sideways']*3
    rows,scores=[],[]
    for i,f in enumerate(FOLDS):
        regime={'trend':regimes[i]}
        for cid in POLICIES:
            value={k:0. for k in ECON_KEYS}
            value.update(alpha_ex=.02,maxdd_delta=-.03)
            rows.append({'fold':f,'candidate_id':cid,'regime':regime,'base':value.copy(),'stress_2x':value.copy()})
        for mid in MEANS:
            scores.append({'fold':f,'mean_id':mid,'regime':regime,'rows':10+i,
                'return_mse':1. if mid.endswith('half') else 2.,'return_mae':.5,
                'zero_return_mse':2.,'fit_mean_return_mse':2.,'return_sign_accuracy':.5,
                'return_rank_ic':None if mid=='scale_mean' else .1})
    return rows,scores


class AdditionalWindowReplayTests(unittest.TestCase):
    def test_fixed_literal_calendar_family_and_contract_cannot_change(self):
        cfg=yaml.safe_load(Path('configs/oracle_additional_window_replay_20260906.yaml').read_text())
        validate_config(cfg)
        for key,value in {'evaluation_folds':list(range(26,36)),'mean_weight':.75,
                'selection_permitted':True,'independent_confirmation':True,'minimum_fit_rows':1,
                'performance_early_stopping':True,'utility_cost_multiplier':1,'mean_weight_type':.5}.items():
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(cfg);bad['source_bindings'].pop(next(iter(bad['source_bindings'])))
        with self.assertRaises(ValueError):validate_config(bad)

    def test_complete_family_has80_descriptive_components_and_no_inference_claim(self):
        rows,scores=fixture();result=describe_additional_family(rows,scores)
        self.assertEqual(result['policy_rows'],120);self.assertEqual(result['forecast_rows'],50)
        self.assertEqual(result['component_count'],80)
        self.assertEqual(len({c['id'] for c in result['descriptive_components']}),80)
        self.assertTrue(result['regime_coverage'])
        self.assertTrue(all(x['observed_metric_and_coverage_conditions_met'] for x in result['candidates'].values()))
        self.assertFalse(result['high_probability_generalization_established'])
        self.assertFalse(result['selection_performed']);self.assertIsNone(result['p_values'])
        self.assertEqual(len(result['paired_policies']),10)

    def test_absent_regime_stays_null_and_fails_without_dropping_quarters(self):
        rows,scores=fixture(['bull']*10);result=describe_additional_family(rows,scores)
        self.assertEqual(result['policy_rows'],120);self.assertFalse(result['regime_coverage'])
        for cid in CANDIDATES:
            self.assertIsNone(result['policies'][cid]['bear']['base']['alpha_ex_mean'])
            self.assertFalse(result['candidates'][cid]['observed_metric_and_coverage_conditions_met'])

    def test_missing_duplicate_mismatched_or_nonfinite_rows_are_rejected(self):
        for mutation in ('missing','duplicate','regime','denominator','nan','bool','complex','constant_ic'):
            rows,scores=fixture()
            if mutation=='missing':rows.pop()
            elif mutation=='duplicate':rows.append(rows[0])
            elif mutation=='regime':scores[0]['regime']={'trend':'bear'}
            elif mutation=='denominator':scores[0]['rows']+=1
            elif mutation=='nan':rows[0]['base']['alpha_ex']=np.nan
            elif mutation=='bool':scores[0]['return_mse']=True
            elif mutation=='complex':scores[0]['return_mse']=1+0j
            else:scores[0]['return_rank_ic']=0.
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):describe_additional_family(rows,scores)

    def test_equal_quarter_relative_loss_is_ratio_of_means(self):
        rows,scores=fixture()
        for row in scores:
            if row['mean_id']=='scale_mean':row['return_mse']=1. if row['fold']==15 else 100.
            if row['mean_id']=='technical_half':row['return_mse']=2.
        result=describe_additional_family(rows,scores)
        got=result['paired_mse']['technical_half_vs_scale_mean']['all']['return_mse_relative_loss_reduction']
        self.assertAlmostEqual(got,1-2/90.1)
        self.assertNotAlmostEqual(got,1-(2+9*.02)/10)

    def test_cross_feature_descriptive_loss_is_not_an_extra_candidate_gate(self):
        rows,scores=fixture()
        for row in scores:
            if row['mean_id']=='perp_delay0_half':row['return_mse']=1.1
        result=describe_additional_family(rows,scores)
        self.assertLess(result['paired_mse']['perp_delay0_half_vs_technical_half']['all']['return_mse_relative_loss_reduction'],0)
        self.assertTrue(result['candidates']['perp_delay0_half_utility_risk1']['observed_predictive_signs'])

    def test_missing_input_rules_use_own_state_and_only_fallback_at_known_open(self):
        ix=pd.date_range('2024-01-01',periods=49,freq='15min',tz='UTC')
        frame=pd.DataFrame({'open':100.,'close':100.,'bar_available':True},index=ix)
        inf=np.zeros(len(ix),bool);inf[0]=True
        pred={'mu':np.where(inf,.04,np.nan),'variance':np.where(inf,.0001,np.nan),'inference_mask':inf}
        predictions={m:pred for m in MEANS}
        execution={'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01}
        hold,_=plan(frame,'technical_half_utility_risk1',predictions,execution,np.ones(len(ix)))
        fallback,_=plan(frame,'technical_half_utility_risk1_fallback_bh',predictions,execution,np.ones(len(ix)))
        self.assertTrue(np.isnan(hold[24]));self.assertEqual(fallback[24],1.)
        frame.loc[ix[24],'open']=np.nan
        fallback,_=plan(frame,'technical_half_utility_risk1_fallback_bh',predictions,execution,np.ones(len(ix)))
        self.assertTrue(np.isnan(fallback[24]))

    def test_incomplete_resume_inventory_cannot_succeed(self):
        rows,scores=fixture()
        saved={'registration_sha256':'fixed','rows':[r for r in rows if r['fold']==15],
            'scores':[s for s in scores if s['fold']==15],'artifact_sha256':{}}
        self.assertEqual(len(artifact_inventory(Path('out'),15)),33)
        with self.assertRaisesRegex(ValueError,'inventory'):
            validate_completed_fold(saved,15,Path('out'),'fixed',None,None,{'regime':{'trend':'bull'}},None,None,None,None)


if __name__=='__main__':unittest.main()
