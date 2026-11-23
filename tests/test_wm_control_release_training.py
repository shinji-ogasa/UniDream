"""Real tiny head PPO, split-gated caches, and continuous account replay."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_control_release_training as training
from unidream.experiments import wm_control_release_accounting as account
from unidream.experiments.wm_policy_ablation_models import PolicyActorCritic


class ControlHeadTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();source=self.base/'source';source.mkdir()
        prepared=source/'prepared';prepared.mkdir()
        output=self.base/'output';output.mkdir()
        self.kernel={'output_dir':str(source),'device':'cpu',
            'splits':{'start':'2023-01-01','end':'2026-10-01'},
            'policy':{'context':4,'normalizer_clip':10.,'purge_bars':64,'horizon':16,
                'train_years':2,'val_months':3,'test_months':3},
            'actor':{'context':4,'lstm_hidden_size':4,'lstm_layers':1,'projection_dim':4,
                'hidden_size':8,'dropout':0.,'min_log_std':-4.,'max_log_std':0.},
            'accounting':{'position_min':.5,'position_max':1.12,'max_position_step':.08}}
        rng=np.random.default_rng(712);nraw=360
        ids=np.r_[np.arange(10,90),np.arange(200,240),np.arange(300,340)]
        self.raw_times=np.r_[pd.date_range('2023-01-01',periods=200,freq='15min',tz='UTC').asi8,
            pd.date_range('2025-01-01',periods=100,freq='15min',tz='UTC').asi8,
            pd.date_range('2025-04-01',periods=60,freq='15min',tz='UTC').asi8]
        close=100*np.exp(np.cumsum(rng.normal(0,.002,nraw)))
        opening=np.r_[close[0],close[:-1]]
        self.ohlc=np.column_stack([opening,np.maximum(opening,close)*1.001,np.minimum(opening,close)*.999,close])
        self.price_path=self.base/'ohlc.npy';np.save(self.price_path,self.ohlc)
        time_path=self.base/'index.npy';np.save(time_path,self.raw_times)
        arrays={'decision_id':ids,'index_ns':self.raw_times[ids],
            'x_grid':rng.normal(size=(nraw,2)).astype(np.float32),'f':rng.normal(size=(len(ids),3)).astype(np.float32),
            'context_valid':np.ones(len(ids),bool)}
        manifest={'complete':True,'arrays':{}}
        for name,value in arrays.items():
            path=prepared/(name+'.npy');np.save(path,value)
            manifest['arrays'][name]={'path':str(path),'sha256':training.sha(path)}
        training.write_json(prepared/'cache_manifest.json',manifest)
        fold={'train':np.arange(80),'val':np.arange(80,120),'earlystop':np.arange(80,90),'test':np.arange(120,160),
            'x_mean':np.zeros(2),'x_std':np.ones(2),'f_mean':np.zeros(3),'f_std':np.ones(3)}
        fold_path=prepared/'fold_00.npz';np.savez(fold_path,**fold)
        latent=rng.normal(size=(len(ids),5)).astype(np.float32)
        train_latent=latent.copy();train_latent[120:]=np.nan
        test_latent=latent.copy();test_latent[:120]=np.nan
        latent_path=source/'latent_train_val.npy';np.save(latent_path,train_latent)
        test_latent_path=source/'latent_test.npy';np.save(test_latent_path,test_latent)
        norm_path=source/'latent_norm.npz';np.savez(norm_path,mean=np.zeros(5),std=np.ones(5))
        training._cpu_seed(6)
        actor=PolicyActorCritic(2,3,5,self.kernel,'C')
        bc_path=source/'bc.pt';training.source_policy._save_checkpoint(bc_path,actor,'b'*64,100,'bc')
        with torch.no_grad():
            actor.x_encoder[0].weight.add_(.3)
            actor.policy_head[-1].bias.sub_(.4)
        actor.position_min=0.
        actor_path=source/'floor_gradual.pt';training.source_policy._save_checkpoint(actor_path,actor,'a'*64,125,'rl_continuation')
        fit_path=source/'fit.json';training.write_json(fit_path,{'complete':True,'test_used':False})
        pair={'kernel_config':self.kernel,'fold_path':str(fold_path),'initial_checkpoint_path':str(actor_path),
            'initial_checkpoint_registration_sha256':'a'*64,'initial_fit_path':str(fit_path),
            'bc_checkpoint_path':str(bc_path),'bc_checkpoint_registration_sha256':'b'*64,
            'latent_train_val_path':str(latent_path),'latent_test_path':str(test_latent_path),
            'latent_norm_path':str(norm_path),'latent_dim':5}
        names=[actor_path,bc_path,fit_path,fold_path,latent_path,test_latent_path,norm_path]
        pair['source_artifact_sha256']={str(path):training.sha(path)for path in names}
        stage=lambda end,cap,step,prior:{'through_update':end,'position_max':cap,'max_position_step':step,'bc_prior_coefficient':prior}
        self.cfg={'output_dir':str(output),'source_pairs':{'fold_00_C_s3':pair},'conditions':['control','free'],
            'accounting':{'position_min':0.,'position_max':1.12,'max_position_step':.08,'initial_position':1.,
                'cost_rate':.00055,'borrow_apr':.1,'annual_bars':35040,'drawdown_penalty':.25,
                'execution_delay_bars':1,'initial_entry_cost':False,'terminal_liquidation':False,'reset_positions_at_segment_boundary':False},
            'training':{'max_updates':8,'selection_updates':[6,8],'rollout_batch_size':2,'rollout_horizon':4,
                'minibatch_size':4,'ppo_epochs':1,'learning_rate':.001,'critic_learning_rate':.001,
                'gamma':.99,'gae_lambda':.95,'clip_ratio':.2,'value_coefficient':.5,'entropy_coefficient':.001,
                'bc_prior_coefficient':.01,'reward_scale':100.,'gradient_clip':10.,'context_batch_size':16},
            'runtime':{'cpu_threads':1,'encoder_device':'cpu'},
            'data':{'raw_ohlc_path':str(self.price_path),'raw_index_ns_path':str(time_path)},
            'schedules':{'control':[stage(8,1.12,.08,.01)],
                'free':[stage(2,1.12,.08,.01),stage(4,1.5,.16,.003),stage(8,2.,.24,0.)]}}
        self.tasks=[{'job':'fold_00_C_s3_'+condition,'fold':0,'seed':3,'arm':'C','condition':condition}for condition in self.cfg['conditions']]
        training.write_json(output/'preregistration.json',{'config':self.cfg,'jobs':self.tasks})
        self.reg=training.sha(output/'preregistration.json');self.output=output

    def test_actor_and_prior_caches_use_distinct_frozen_encoders_and_train_normalization(self):
        pair=self.cfg['source_pairs']['fold_00_C_s3'];before=dict(pair['source_artifact_sha256'])
        cache=training.prepare_context(self.cfg,self.tasks[0],self.reg)
        arrays=training._cached(cache)
        self.assertEqual(cache['n_rows'],120);self.assertEqual(cache['normalizer_fit_split'],'original policy Train only')
        self.assertNotEqual(cache['actor_encoder_sha256'],cache['prior_encoder_sha256'])
        self.assertFalse(np.allclose(arrays['contexts'],arrays['prior_contexts']))
        self.assertTrue(np.array_equal(arrays['raw_ids'],np.r_[np.arange(10,90),np.arange(200,240)]))
        self.assertNotIn(pair['latent_test_path'],cache['source_artifact_sha256'])
        for filename,digest in before.items():self.assertEqual(training.sha(filename),digest)
        with patch.object(training,'_load_source',side_effect=AssertionError('encoding must be reused')):
            self.assertEqual(training.prepare_context(self.cfg,self.tasks[1],self.reg),cache)

    def test_registration_and_new_test_barrier_precede_source_or_test_latent_reads(self):
        with patch.object(training,'_load_source')as load:
            with self.assertRaises(ValueError):training.prepare_context(self.cfg,self.tasks[0],'wrong')
            load.assert_not_called()
        with patch.object(training,'_barrier',side_effect=ValueError('global barrier closed')),patch.object(training,'_source_hashes')as source:
            with self.assertRaisesRegex(ValueError,'global barrier closed'):
                training.prepare_context(self.cfg,self.tasks[0],self.reg,'test')
            source.assert_not_called()

    def test_cpu_fit_seed_never_calls_MPS_seed_and_is_reproducible(self):
        with patch.object(torch.mps,'manual_seed',side_effect=AssertionError('CPU fit must not touch MPS')):
            training._cpu_seed(19);left=torch.randn(10)
            training._cpu_seed(19);right=torch.randn(10)
        torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_real_head_ppo_changes_only_heads_uses_full_Val_and_matches_common_prefix(self):
        fits=[];histories=[]
        for task in self.tasks:
            fit=training.train(self.cfg,task,self.reg);fits.append(fit)
            path=self.output/'neural'/task['job']
            history=json.loads((path/'training_history.json').read_text());histories.append(history)
            self.assertTrue(fit['encoder_unchanged']);self.assertTrue(fit['head_changed']);self.assertFalse(fit['test_used'])
            self.assertEqual(fit['historical_transitions'],64);self.assertEqual(fit['updates'],8)
            self.assertEqual(fit['validation_calendar_rows'],100)
            self.assertEqual([row['update']for row in history if'validation_score'in row],[6,8])
            ledger=pd.read_parquet(path/'validation_predictions.parquet')
            self.assertEqual(len(ledger),100);self.assertEqual(int(ledger.context_available.sum()),40)
            self.assertFalse(ledger.iloc[40:].decision_available.any())
            self.assertEqual(fit['head_gradient_coverage'],{'policy':True,'critic':True,'encoder_has_no_gradients':True})
            with patch.object(training,'prepare_context',side_effect=AssertionError('completed fit must not retrain')):
                self.assertEqual(training.train(self.cfg,task,self.reg),fit)
        for key in ('initial_state_sha256','immutable_bc_prior_state_sha256','market_start_sequence_sha256'):
            self.assertEqual(fits[0][key],fits[1][key])
        strip=lambda value:{key:item for key,item in value.items()if key!='elapsed_seconds'}
        self.assertEqual([strip(row)for row in histories[0][:2]],[strip(row)for row in histories[1][:2]])
        with patch.object(training,'_barrier',return_value={'registration_sha256':self.reg}):
            selected,settings,cache=training.load_selected(self.cfg,self.tasks[1],self.reg)
            frozen,_,_=training.load_selected(self.cfg,self.tasks[1],self.reg,frozen=True)
        self.assertEqual(settings['accounting']['position_max'],2.)
        self.assertEqual(selected.max_position_step,.24)
        self.assertEqual(training._state_sha(frozen),fits[1]['initial_state_sha256'])
        self.assertNotEqual(training._state_sha(selected),training._state_sha(frozen))
        self.assertTrue(all(not p.requires_grad for p in selected.parameters()))

    def test_calendar_replay_carries_inventory_pending_orders_and_drawdown_through_feature_gaps(self):
        cache=training.prepare_context(self.cfg,self.tasks[0],self.reg)
        model=training._head_only(training._load_source(self.cfg['source_pairs']['fold_00_C_s3']),False)
        settings=training.condition_config(self.cfg,'control');training._apply(model,settings)
        whole,end=training.replay(model,cache,np.arange(200,300),settings)
        first,middle=training.replay(model,cache,np.arange(200,240),settings)
        self.assertTrue(bool(middle['has_pending'][0]))
        second,split_end=training.replay(model,cache,np.arange(240,300),settings,initial_state=middle)
        self.assertTrue(second.order_filled.iloc[0]);self.assertFalse(second.decision_available.any())
        self.assertFalse(second.has_pending.any())
        merged=pd.concat([first,second],ignore_index=True)
        for name in ('equity','net_log_return','fee','borrow','max_drawdown','units','cash'):
            np.testing.assert_allclose(whole[name],merged[name],rtol=0,atol=1e-14)
        for name in end:torch.testing.assert_close(end[name],split_end[name],rtol=0,atol=1e-14)
        self.assertNotEqual(float(second.executed_position.iloc[0]),1.)

    def test_bankruptcy_is_retained_and_absorbing_in_replay_and_train_surrogate(self):
        cache=training.prepare_context(self.cfg,self.tasks[0],self.reg)
        settings=training.condition_config(self.cfg,'free');model=training._head_only(training._load_source(self.cfg['source_pairs']['fold_00_C_s3']),False)
        training._apply(model,settings)
        shocked=self.ohlc.copy();shocked[200,2]=1.
        path=self.base/'shock.npy';np.save(path,shocked)
        shock_cfg={**settings,'data':{**settings['data'],'raw_ohlc_path':str(path)}}
        price=shocked[199,3];state=account.initial_state(1,'cpu',price,settings)
        state['cash'].fill_(-1);state['units'].fill_(2/price);state['requested'].fill_(2);state['executed'].fill_(2)
        ledger,end=training.replay(model,cache,np.arange(200,205),shock_cfg,initial_state=state)
        self.assertTrue(ledger.bankrupt.all());self.assertEqual(ledger.equity.iloc[-1],0)
        self.assertTrue(np.isneginf(ledger.net_log_return.iloc[0]));self.assertTrue(ledger.net_log_return.iloc[1:].eq(0).all())
        self.assertTrue(np.isfinite(ledger.training_log_return).all());self.assertTrue(ledger.unscaled_reward.iloc[1:].eq(0).all())
        metrics=training.replay_metrics(ledger)
        self.assertTrue(metrics['bankrupt']);self.assertEqual(metrics['net_total_return'],-1)
        self.assertEqual(metrics['selection_score'],-1e30)

    def test_absorbing_training_states_have_no_policy_entropy_or_prior_weight(self):
        pair=self.cfg['source_pairs']['fold_00_C_s3'];settings=training.condition_config(self.cfg,'free')
        model=training._head_only(training._load_source(pair),True);prior=training._head_only(training._load_source(pair,True),False)
        training._apply(model,settings)
        with torch.no_grad():model.policy_head[-1].weight.zero_();model.policy_head[-1].bias.fill_(20.)
        contexts=torch.randn(2,5,12);prior_contexts=torch.randn(2,5,12)
        prices=self.ohlc.copy();prices[11,2]=1.
        rollout=training._collect(model,contexts,prior_contexts,prices,np.tile(np.arange(10,15),(2,1)),settings)
        self.assertEqual(rollout['bankrupt_rollouts'],2)
        expected=torch.tensor([True,True,False,False,True,True,False,False])
        torch.testing.assert_close(rollout['batch']['active'],expected)
        self.assertTrue(torch.isfinite(rollout['batch']['returns']).all())
        self.assertTrue(rollout['batch']['returns'][~expected].eq(0).all())
        diagnostics=training._ppo(model,prior,training._optimizer(model,settings),rollout,settings)
        self.assertTrue(all(np.isfinite(value)for value in diagnostics.values()))
        self.assertTrue(training._coverage(model)['encoder_has_no_gradients'])


if __name__=='__main__':unittest.main()
