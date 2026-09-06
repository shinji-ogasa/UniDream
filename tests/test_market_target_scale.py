"""Synthetic main-reward unit correction; no real model fits or outputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tests.test_market_wm_reward import config, _WmSpy, ac_trainer
from unidream.data.dataset import SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer, resolve_market_target_scale


def scaled_config(scale):
    cfg=config();cfg['world_model']['market_target_scale']=scale
    return cfg


class MarketTargetScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_scale1_exact_original_target_object_and_checkpoint_contract(self):
        default=WorldModelTrainer(_WmSpy(),config(),device='cpu')
        explicit=WorldModelTrainer(_WmSpy(),scaled_config(1),device='cpu')
        raw=torch.tensor([[.01,-.02,0.]])
        for trainer in (default,explicit):
            self.assertIs(trainer._compute_net_returns(None,raw),raw)
            self.assertEqual(trainer.market_reward_contract,{'mode':'market_log_return',
                'action_context':'actionless','context_action':1.,'target':'actual_raw_market_log_return'})
        a,b=ac_trainer(config()),ac_trainer(scaled_config(1.))
        self.assertEqual(a.market_reward_contract,b.market_reward_contract)
        self.assertNotIn('market_target_scale',a.market_reward_contract)

    def test100_train_and_eval_loss_target_ignores_actions_without_touching_raw(self):
        wm=_WmSpy();trainer=WorldModelTrainer(wm,scaled_config(100),device='cpu')
        raw=np.array([.01,-.02,.03,-.04],dtype=np.float32)
        features=np.column_stack([raw,raw*2]);saved=raw.copy()
        ds=SequenceDataset(features,seq_len=2,returns=raw,actions=np.full(4,np.nan,dtype=np.float32))
        trainer.train_on_dataset(ds,checkpoint_path=None);trainer._eval_loss(ds)
        self.assertGreaterEqual(len(wm.calls),3)
        for obs,actions,rewards in wm.calls:
            torch.testing.assert_close(actions,torch.ones_like(actions),rtol=0,atol=0)
            torch.testing.assert_close(rewards,obs[:,:,0]*100.,rtol=0,atol=0)
        np.testing.assert_array_equal(raw,saved)
        for actions in (None,torch.full((2,4,1),np.nan)):
            value=torch.tensor([[.01,-.02,0.,.0001]])
            torch.testing.assert_close(trainer._compute_net_returns(actions,value),value*100,rtol=0,atol=0)

    def test_all_eight_auxiliary_targets_and_masks_are_unscaled(self):
        a=WorldModelTrainer(_WmSpy(),config(),device='cpu')
        b=WorldModelTrainer(_WmSpy(),scaled_config(100.),device='cpu')
        raw=torch.sin(torch.arange(256,dtype=torch.float32).reshape(2,128))*.003
        for name in ('_future_return_targets','_future_risk_targets','_future_control_targets','_future_position_utility_targets'):
            expected=getattr(a,name)(raw);actual=getattr(b,name)(raw)
            self.assertEqual(len(expected),len(actual))
            for x,y in zip(expected,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_disagreement_restored_before_dividing_and_account_matches_raw_units(self):
        # Dyadic values make scaling/recovery exact rather than tolerance-based.
        raw=(.015625,-.03125,.0078125)
        scaled=tuple(x*100 for x in raw)
        a=ac_trainer(config(),targets=(1.08,.92,1.04),returns=raw,disagreement=.625)
        b=ac_trainer(scaled_config(100.),targets=(1.08,.92,1.04),returns=scaled,disagreement=.625)
        z=torch.zeros(2,2,dtype=torch.float64)
        first=a._imagination_rollout(z,z);second=b._imagination_rollout(z,z)
        for key in ('market_log_returns','benchmark_rewards','rewards','cash','asset_values','nav','fees','borrow','trade_values'):
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0,msg=key)
        self.assertEqual(b.market_reward_contract['market_target_scale'],100.)
        self.assertEqual(b.ensemble.calls[0][0].unique().tolist(),[1.])

    def test_scaled_market_checkpoint_mismatch_rejected_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind,factory in [('wm',lambda c:WorldModelTrainer(_WmSpy(),c,device='cpu')),('ac',ac_trainer)]:
                a,b=factory(config()),factory(scaled_config(100.));path=Path(tmp)/(kind+'.pt')
                a.save(str(path));before=torch.load(path,weights_only=False)['market_reward_contract']
                self.assertNotIn('market_target_scale',before)
                a.load(str(path))
                with self.assertRaisesRegex(ValueError,'contract'):b.load(str(path))
                b.save(str(path));after=torch.load(path,weights_only=False)['market_reward_contract']
                self.assertEqual(after,before|{'market_target_scale':100.})
                b.load(str(path))
                with self.assertRaisesRegex(ValueError,'contract'):a.load(str(path))

    def test_invalid_scale_and_nonmarket_optin_fail(self):
        for value in (True,False,0,-1,np.inf,np.nan,None,'100',1j):
            for factory in (lambda c:WorldModelTrainer(_WmSpy(),c,device='cpu'),ac_trainer):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'market_target_scale'):
                    factory(scaled_config(value))
        for mode in ('absolute','excess_bh'):
            cfg=scaled_config(100.);cfg['world_model']['reward_mode']=mode
            for factory in (lambda c:WorldModelTrainer(_WmSpy(),c,device='cpu'),ac_trainer):
                with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'only supported'):
                    factory(cfg)
            cfg['world_model']['market_target_scale']=1.
            self.assertEqual(resolve_market_target_scale(cfg),1.)
        for key,value in [('action_context','oracle')]:
            cfg=scaled_config(100.);cfg['world_model'][key]=value
            for factory in (lambda c:WorldModelTrainer(_WmSpy(),c,device='cpu'),ac_trainer):
                with self.assertRaisesRegex(ValueError,'actionless'):factory(cfg)

    def test_scaled_target_overflow_and_decoded_underflow_scale_overflow_fail(self):
        trainer=WorldModelTrainer(_WmSpy(),scaled_config(100.),device='cpu')
        with self.assertRaises(ValueError):
            trainer._compute_net_returns(None,torch.tensor([[torch.finfo(torch.float32).max]]))
        ac=ac_trainer(scaled_config(1e-300),returns=(1.,1.,1.))
        with self.assertRaises(ValueError):ac._imagination_rollout(torch.zeros(1,2),torch.zeros(1,2))


if __name__=='__main__':unittest.main()
