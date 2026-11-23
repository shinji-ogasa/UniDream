import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_tail_transfer_report as report
from unidream.experiments import wm_tail_transfer_policy as policy
from unidream.experiments import wm_tail_transfer_benchmark as benchmark


class TailTransferReportTests(unittest.TestCase):
    def test_Test_gate_precedes_data_or_checkpoint_access(self):
        with patch.object(benchmark,'verify_test_barrier',side_effect=ValueError('not frozen')):
            with patch.object(report.data_module,'load_data') as load:
                with self.assertRaisesRegex(ValueError,'not frozen'):
                    report.evaluate_forecast_job({}, {'job':'missing'},'registration')
                with self.assertRaisesRegex(ValueError,'not frozen'):
                    report.evaluate_policy_path({}, {'tasks':[]},'registration')
                load.assert_not_called()

    def test_forecast_retains_all_calibrations_coordinates_and_physical_ids(self):
        rng=np.random.default_rng(7);ids=np.arange(40,80);train=np.arange(40)
        targets=rng.normal(0,.01,(80,2))
        data={'index_ns':np.arange(80,dtype=np.int64)*900_000_000_000,
              'targets':{'return':targets},'masks':{'return':np.ones_like(targets,bool)}}
        cfg={'targets':{'horizons':[1,16],'primary_horizon':16}}
        spec={'name':'return','kind':'real','task':'T01','shape':[2],'levels':None}
        predictions={name:{'return':targets[ids].copy()} for name in report.CALIBRATIONS}
        with tempfile.TemporaryDirectory() as tmp:
            record=report._write_forecast(cfg,dict(job='fit1',fold=0,seed=7,model='linear'),
                'reg',data,ids,train,[spec],predictions,Path(tmp),source_sha256={})
            self.assertEqual(record['n_metric_coordinates'],6)
            with np.load(Path(tmp)/'per_row_losses.npz') as losses:
                np.testing.assert_array_equal(losses['raw_ids'],ids)
                for calibration in report.CALIBRATIONS:
                    np.testing.assert_array_equal(losses[calibration+'__T01__model_loss'],0)
                    self.assertIn(calibration+'__head_return__model_loss',losses.files)
            self.assertEqual(report._completed(Path(tmp)/'evaluation.json','reg'),record)
            (Path(tmp)/'task_metrics.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                report._completed(Path(tmp)/'evaluation.json','reg')

    def test_training_direction_filters_exclude_untrained_opposite_output(self):
        cfg={'targets':{'horizons':[16],'primary_horizon':16,'fixed_thresholds':[.03]}}
        truth=np.zeros((10,1,2,1));truth[::2]=1
        pred=truth[5:]*.8+.1;pred[:,:,1]=.99
        data={'index_ns':np.arange(10,dtype=np.int64)*900_000_000_000,
              'targets':{'reach_fixed':truth},'masks':{'reach_fixed':np.ones_like(truth,bool)}}
        spec=dict(name='reach_fixed',kind='binary',task='T02',shape=[1,2,1])
        with tempfile.TemporaryDirectory() as tmp:
            report._write_forecast(cfg,dict(job='down',direction='down'), 'reg',data,np.arange(5,10),np.arange(5),
                [spec],{'none':{'reach_fixed':pred}},tmp,source_sha256={})
            metrics=pd.read_parquet(Path(tmp)/'test_metrics.parquet')
            self.assertEqual(metrics.direction.tolist(),['down'])
            with np.load(Path(tmp)/'per_row_losses.npz') as losses:
                np.testing.assert_allclose(losses['none__T02__model_loss'],.01)

    def test_seed_panel_keeps_nonfinite_observations_and_common_mask(self):
        times=np.arange(6,dtype=np.int64)*900_000_000_000
        rows=[(7,times[:3],[1,2,np.nan],[True,True,True]),(7,times[3:],[4,5,6],[True,True,True]),
              (19,times,[2,3,4,5,6,7],[True,False,True,True,True,True])]
        returned,values,mask,seeds=report._series_panel(rows)
        np.testing.assert_array_equal(returned,times)
        self.assertEqual(seeds,[7,19]);self.assertTrue(np.isnan(values[2,0]))
        self.assertTrue(mask[2,0]);self.assertFalse(mask[1].any())
        with self.assertRaisesRegex(ValueError,'overlapping'):
            report._series_panel([(7,times,[1]*6,[True]*6),(7,times,[2]*6,[True]*6)])

    def test_worst_seed_drawdown_and_mean_not_best_seed(self):
        rows=[]
        for seed,ret,dd in [(7,.1,.2),(42,-.1,.5),(19,.3,.1)]:
            rows.append(dict(stage='RL',policy_variant='R04',control_json='{}',sensitivity_json='{}',oracle=False,
                seed=seed,net_total_return=ret,max_drawdown=dd,mean_exposure=.5,exposure_max=2.,fee_sum=.01,
                borrow_sum=.02,funding_sum=0.,liquidation_fee_sum=0.,bankrupt=False,liquidation=False))
        result=report.seed_summary(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(result.net_total_return_mean,.1)
        self.assertEqual(result.worst_drawdown_seed,42)
        self.assertEqual(result.worst_drawdown,.5)

    def test_unavailable_fit_never_loads_Test_data_or_makes_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'fits/missing';folder.mkdir(parents=True)
            (folder/'fit.json').write_text(json.dumps(dict(complete=True,available=False,status='unavailable_data',
                registration_sha256='reg',reason='empty calibration month',artifact_sha256={})))
            cfg={'output_dir':tmp,'targets':{'horizons':[1,16],'primary_horizon':16}}
            with patch.object(benchmark,'verify_test_barrier',return_value={'registration_sha256':'reg'}),\
                 patch.object(report.data_module,'load_data') as load:
                result=report.evaluate_forecast_job(cfg,dict(job='missing',model='wm',objectives='joint'),'reg')
                load.assert_not_called()
            self.assertFalse(result['available']);self.assertEqual(result['n_prediction_rows'],0)
            self.assertFalse((Path(tmp)/'evaluations/forecast/missing/test_predictions.npz').exists())

    def test_ensemble_means_probabilities_and_retains_member_disagreement(self):
        rng=np.random.default_rng(7);ids=np.arange(40,80);train=np.arange(40)
        targets=rng.normal(0,.01,(80,1))
        data={'index_ns':np.arange(80,dtype=np.int64)*900_000_000_000,
              'targets':{'return':targets},'masks':{'return':np.ones_like(targets,bool)}}
        cfg={'targets':{'horizons':[16],'primary_horizon':16},'seeds':[7,19,42]}
        spec=dict(name='return',kind='real',task='T01',shape=[1],levels=None)
        members=[dict(job=f'member{s}',fold=0,seed=s,group='D03',model='wm') for s in [7,19,42]]
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir']=tmp
            for member,bias in zip(members,[-.01,0,.01]):
                report._write_forecast(cfg,member,'reg',data,ids,train,[spec],
                    {'none':{'return':targets[ids]+bias}},Path(tmp)/'evaluations/forecast'/member['job'],source_sha256={})
            task=dict(job='ensemble',model='ensemble_wm',role='ensemble',members=members,fold=0,seed=7,group='D03')
            with patch.object(benchmark,'verify_test_barrier',return_value={'registration_sha256':'reg'}),\
                 patch('unidream.experiments.wm_tail_transfer_training.split_arrays',return_value=({}, {}, {'train':train})):
                result=report.evaluate_ensemble_job(cfg,task,'reg',arrays=data)
            with np.load(Path(tmp)/'evaluations/forecast/ensemble/per_row_losses.npz') as arrays:
                np.testing.assert_allclose(arrays['none__T01__model_loss'],0,atol=1e-12)
            with np.load(Path(tmp)/'evaluations/forecast/ensemble/member_uncertainty.npz') as arrays:
                np.testing.assert_allclose(arrays['none__return'],.01,atol=1e-8)
            self.assertEqual(result['n_new_training_fits'],0)

    def test_preregistered_plan_has_fixed_families_and_never_reads_outputs(self):
        cfg=benchmark.load_config();jobs=[]
        for fold in [0,3,5]:
            for seed in [7,19,42]:
                for model in ['wm','transformer']:
                    jobs.append(benchmark.make_job('benchmark','D03',model,fold,seed,cohort='dense_common'))
        with patch.object(Path,'read_text',side_effect=AssertionError('no outcome reads allowed')):
            plan=report.comparison_plan(cfg,jobs,[])
        self.assertTrue(plan['contrasts'])
        self.assertTrue(all(c['family'].endswith(c['target']) for c in plan['contrasts']))
        self.assertEqual(len({c['name'] for c in plan['contrasts']}),len(plan['contrasts']))
        self.assertEqual(len(report.rollout_jobs(cfg,jobs)),9)
        self.assertEqual(len(report.ensemble_jobs(cfg,jobs)),6)

    def test_baseline_specs_are_unique_environment_and_fixed_reference_seed(self):
        cfg={'seeds':[7,19,42,3]};paths=[]
        for seed in [3,7,19]:
            for variant in ['R00','R04']:
                control=dict(cap=2.,speed=.08,account_mode='cash_borrow',controller='direct',bc_prior=.01)
                paths.append(dict(job=f'{seed}_{variant}',stage='RL',sensitivity=None,
                    tasks=[dict(fold=0,seed=seed,policy_variant=variant,control=control)]))
        selected=report.baseline_path_specs(cfg,paths)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['tasks'][0]['seed'],7)
        self.assertEqual(report.policy_path_id(selected[0]),selected[0]['job'])

    def test_rollout_diagnostics_are_close_path_and_nonfinite_not_clipped(self):
        cfg={'targets':{'horizons':[1,4],'fixed_thresholds':[.01,.03]}}
        actual=np.array([[.01,-.02,.03,-.04],[.01,.01,.01,.01]])
        samples=np.repeat(actual[:,None,:],8,axis=1)
        result=report.rollout_diagnostics(samples,actual,cfg)
        terminal=result[result.diagnostic=='terminal_log_return']
        np.testing.assert_allclose(terminal.crps,0)
        np.testing.assert_allclose(terminal.coverage90,1)
        self.assertTrue(result.observation_basis.str.contains('not_intrabar').all())
        samples[0,0,0]=np.inf
        invalid=report.rollout_diagnostics(samples,actual,cfg)
        self.assertTrue((invalid.invalid_decisions==1).all())

    def test_real_policy_replay_carries_units_pending_and_peak_across_folds(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        n=48;index=pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC').asi8
        close=100*np.exp(np.arange(n)*.001);opening=np.r_[close[0],close[:-1]]
        bars=np.column_stack([opening,np.maximum(opening,close)*1.001,np.minimum(opening,close)*.999,close,np.ones(n)])
        rng=np.random.default_rng(9)
        source=dict(index_ns=index,ohlcspot=bars,ohlcperp=bars.copy(),ohlcmark=bars.copy(),
            market={name:rng.normal(size=(n,width)).astype(np.float32) for name,width in
                    [('raw',3),('old_z',5),('new_z',5),('forecasts',4),('uncertainty',2),('oracle',4)]},
            available=np.ones(n,bool),risk_sigma=np.full(n,.01),up_probability=np.full(n,.3),
            down_probability=np.full(n,.2),uncertainty_score=np.full(n,.1),
            train_ids=np.arange(4,16),val_ids=np.arange(16,24),test_ids=np.arange(24,48))
        source['available'][29:32]=False
        cfg=dict(policy=dict(context=4,hidden_size=8,projection_dim=4,head_hidden=16),
            accounting=dict(mode='cash_borrow',position_min=0.,position_max=2.,max_position_step=.08,
                initial_position=1.,cost_rate=.00055,borrow_apr=.1,reset_positions_at_segment_boundary=False),
            splits=dict(folds=[0,1]),targets=dict(primary_horizon=16),runtime=dict(cpu_threads=1))
        control=dict(cap=2.,speed=.08,account_mode='cash_borrow')
        tasks=[dict(job=f'policy{f}',fold=f,seed=7,policy_variant='R00',control=control) for f in range(2)]
        folds=[dict(fold=f,test_start=pd.Timestamp(index[start],tz='UTC').isoformat(),
            test_end=pd.Timestamp(index[end] if end<n else index[-1]+900_000_000_000,tz='UTC').isoformat())
            for f,start,end in [(0,24,36),(1,36,48)]]
        model=policy.CachedTailPolicy(policy._dims(source,cfg),cfg,'R00')
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir']=tmp
            for task in tasks:
                folder=Path(tmp)/'policies'/task['job'];folder.mkdir(parents=True)
                policy._save(folder/'selected.pt',model,cfg,task,'reg','RL',1)
                (folder/'fit.json').write_text(json.dumps(dict(complete=True,registration_sha256='reg',
                    artifact_sha256={'selected.pt':benchmark.sha(folder/'selected.pt')})))
            spec=dict(job='continuous',stage='RL',tasks=tasks,sources={'0':{},'1':{}},sensitivity=None)
            with patch.object(benchmark,'verify_test_barrier',return_value={'registration_sha256':'reg'}),\
                 patch.object(benchmark,'fold_definitions',return_value=folds),\
                 patch('unidream.experiments.wm_tail_transfer_training.prepare_producer_cache',return_value=dict(
                     complete=True,split='test',registration_sha256='reg',artifact_sha256={})),\
                 patch('unidream.experiments.wm_tail_transfer_training.load_producer_cache',return_value=source):
                result=report.evaluate_policy_path(cfg,spec,'reg',arrays=source)
            ledger=pd.read_parquet(Path(tmp)/'evaluations/policy/continuous/ledger.parquet')
            self.assertEqual(len(ledger),24);self.assertEqual(result['n_resets'],0)
            self.assertAlmostEqual(ledger.equity_previous.iloc[12],ledger.equity.iloc[11])
            self.assertEqual(ledger.raw_id.tolist(),list(range(24,48)))
            self.assertGreater(ledger.loc[~ledger.context_available,'equity'].std(),0)


if __name__=='__main__':
    unittest.main()
