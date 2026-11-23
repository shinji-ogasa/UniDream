"""Synthetic causality, execution and fixed-family ten-seed evaluation tests."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import numpy as np
import pandas as pd

from unidream.experiments import wm_seed_robustness_evaluation as evaluation
from unidream.experiments.wm_policy_ablation_evaluation import bounded_requests,execute_positions,accounting


SEEDS=[7,19,42,3,11,23,31,53,67,89]


def config():
    return {"seeds":SEEDS,"reused_seeds":[7,19,42],"new_seeds":[3,11,23,31,53,67,89],
        "folds":[0],"conditions":list(evaluation.CONDITIONS),"accounting":{**accounting(),"position_min":0.},
        "baselines":{"constant_exposure":.08,"ma_window_bars":1920,
                     "volatility_window_bars":672,"volatility_target_annual":.10},
        "statistics":{"methods":["circular"],"block_days":[7,14,30],"bootstrap_replicates":99,
                      "hac_lags":[6,13,29],"seed":17}}


def fixture(days=40):
    cfg=config();n=days*96;start=2000
    times=pd.date_range('2024-01-01',periods=start+n+200,freq='15min',tz='UTC').asi8
    returns=np.random.default_rng(23).normal(-.00001,.001,len(times))
    close=100*np.exp(np.cumsum(returns));ids=np.arange(start,start+n)
    # A real gap in the cohort, not in the raw information history.
    ids[n//2:]+=100
    cohort=pd.DataFrame({'fold':0,'decision_id':ids,'timestamp':pd.to_datetime(times[ids],utc=True),
                         'bar_log_return':np.log(close[ids]/close[ids-1])})
    cfg['statistics']['expected_market_rows']=n
    baselines,metrics,audit=evaluation.build_baselines(close,times,cohort,cfg)
    configs=evaluation.condition_configs(cfg);paths=[]
    for condition in evaluation.CONDITIONS:
        for seed in SEEDS:
            target=.6 if condition=='C_control' else (.85 if seed==42 else .1+(seed%4)*.01)
            request=bounded_requests(np.full(n,target),cohort.timestamp,cohort.fold,configs[condition])
            path=execute_positions(cohort.bar_log_return,request,cohort.timestamp,cohort.fold,configs[condition])
            path['decision_id']=ids;paths.append(path.assign(condition=condition,seed=seed))
    return close,times,cohort,pd.concat(paths,ignore_index=True),baselines,metrics,audit,cfg,configs


class SeedBaselineCausalityTests(unittest.TestCase):
    def test_no_current_or_future_close_enters_ma_or_volatility_signal(self):
        cfg=config();close=np.full(4000,100.);times=np.arange(4000,dtype=np.int64)*evaluation.BAR_NS
        ids=np.array([2000,2001,2002]);before=evaluation.baseline_targets(close,times,ids,cfg)
        changed=close.copy();changed[2000:]=150.
        after=evaluation.baseline_targets(changed,times,ids,cfg)
        np.testing.assert_array_equal(before.loc[0,list(evaluation.BASELINES)],after.loc[0,list(evaluation.BASELINES)])
        self.assertEqual(before.loc[0,'ma20d'],0) # Equality is flat.
        self.assertEqual(after.loc[1,'ma20d'],1)
        self.assertEqual(before.loc[0,'voltarget10'],1.12) # Zero variance ->cap.
        self.assertLess(after.loc[1,'voltarget10'],before.loc[1,'voltarget10'])

    def test_exact_full_raw_history_windows_and_annual_sigma(self):
        cfg=config();n=5000;returns=np.full(n,.0001);close=100*np.exp(np.cumsum(returns))
        times=np.arange(n,dtype=np.int64)*evaluation.BAR_NS;ids=np.array([2000,2001,4500])
        values=evaluation.baseline_targets(close,times,ids,cfg)
        np.testing.assert_allclose(values.causal_sma,[close[i-1920:i].mean() for i in ids],rtol=1e-13)
        np.testing.assert_allclose(values.causal_annual_sigma,np.sqrt(35040)*.0001,rtol=1e-11)
        np.testing.assert_allclose(values.voltarget10,np.minimum(1.12,.10/(np.sqrt(35040)*.0001)),rtol=1e-11)
        self.assertTrue(values.ma20d.eq(1).all())

    def test_missing_history_never_filled_or_removed_and_bad_grid_rejected(self):
        cfg=config();close=np.full(4000,100.);times=np.arange(4000,dtype=np.int64)*evaluation.BAR_NS
        close[1500]=np.nan
        with self.assertRaisesRegex(ValueError,'complete causal raw history'):
            evaluation.baseline_targets(close,times,np.array([2000]),cfg)
        with self.assertRaises(ValueError):evaluation.baseline_targets(np.ones(4000),times,np.array([100]),cfg)
        times[10]+=1
        with self.assertRaisesRegex(ValueError,'regular'):
            evaluation.baseline_targets(np.ones(4000),times,np.array([2000]),cfg)

    def test_execution_has_rate_cap_lag_fees_and_episode_resets(self):
        close,times,cohort,frame,baselines,metrics,audit,cfg,configs=fixture(days=2)
        constant=baselines.loc[baselines.baseline.eq('constant08')].reset_index(drop=True)
        np.testing.assert_allclose(constant.executed_position.iloc[:3],[1,.92,.84])
        np.testing.assert_allclose(constant.requested_position.iloc[:3],[.92,.84,.76])
        self.assertAlmostEqual(constant.cost_fraction.iloc[1],.08*.00055)
        self.assertTrue(constant.loc[constant.reset,'executed_position'].eq(1).all())
        self.assertTrue(constant.loc[constant.reset,'cost_fraction'].eq(0).all())
        self.assertAlmostEqual(constant.executed_position.iloc[20],.08)
        bh=baselines.loc[baselines.baseline.eq('bh')]
        np.testing.assert_allclose(bh.net_log_return,bh.bar_log_return,atol=1e-15)
        self.assertFalse(metrics.set_index('baseline').loc['constant08','control_floor_feasible_for_all_targets'])
        self.assertFalse(audit['full_calendar_backtest'])

    def test_volatility_cap_charges_borrowing_on_actual_lagged_exposure(self):
        cfg=config();times=np.arange(4000,dtype=np.int64)*evaluation.BAR_NS
        close=np.full(4000,100.);ids=np.arange(2000,2020)
        cohort=pd.DataFrame({'fold':0,'decision_id':ids,'timestamp':pd.to_datetime(times[ids],utc=True),'bar_log_return':0.})
        paths,_,_=evaluation.build_baselines(close,times,cohort,cfg)
        vol=paths.loc[paths.baseline.eq('voltarget10')].reset_index(drop=True)
        self.assertAlmostEqual(vol.executed_position.iloc[1],1.08)
        self.assertAlmostEqual(vol.borrow_fraction.iloc[1],.08*.1/35040)
        self.assertAlmostEqual(vol.cost_fraction.iloc[1],.08*.00055)


class TenSeedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.close,cls.times,cls.cohort,cls.frame,cls.baselines,cls.metrics,cls.audit,cls.cfg,cls.configs=fixture()
        cls.result=evaluation.evaluate_ledgers(cls.frame,cls.baselines,cls.cfg,cls.configs)

    def test_all_seeds_not_independent_markets_and_worst_seed_disclosed(self):
        r=self.result;self.assertEqual(len(r['pooled_metrics']),30)
        self.assertEqual(len(r['primary_inference']),14)
        self.assertEqual(r['daily_panel']['sums'].shape,(42,14,10))
        self.assertEqual(r['audit']['n_market_rows'],40*96)
        self.assertFalse(r['audit']['seed_count_multiplies_market_rows'])
        rows=r['seed_summary'];self.assertEqual(set(rows.seed_subset),{'all_seeds','reused_seeds','new_seeds'})
        for row in rows.itertuples():
            ids=SEEDS if row.seed_subset=='all_seeds' else self.cfg[row.seed_subset]
            source=r['pooled_metrics'].loc[lambda x:x.condition.eq(row.condition)&x.seed.isin(ids)]
            self.assertAlmostEqual(row.net_total_return_mean,source.net_total_return.mean())
            self.assertAlmostEqual(row.net_total_return_median,source.net_total_return.median())
            self.assertAlmostEqual(row.net_total_return_std,source.net_total_return.std(ddof=1))
            self.assertAlmostEqual(row.worst_drawdown,source.max_drawdown.max())
            self.assertIn(row.worst_drawdown_seed,ids)

    def test_full12_plus2_families_with_all14_adjustment(self):
        rows=self.result['primary_inference']
        self.assertEqual(rows.groupby('family').size().to_dict(),{'baseline_primary':12,'policy_secondary':2})
        self.assertTrue(rows.p_holm_all14.ge(rows.p_holm_family).all())
        self.assertEqual(len(self.result['hac_inference']),42)
        self.assertEqual(set(rows.n_unique_market_rows),{40*96})

    def test_period_returns_compound_per_seed_before_summary(self):
        for key in ['monthly_by_seed','quarterly_by_seed']:
            table=self.result[key]
            for (condition,seed),part in table.groupby(['condition','seed']):
                whole=self.result['pooled_metrics'].loc[lambda x:x.condition.eq(condition)&x.seed.eq(seed)].iloc[0]
                self.assertAlmostEqual(np.prod(1+part.net_total_return)-1,whole.net_total_return,places=12)

    def test_corrupt_baseline_accounting_cohort_and_missing_seed_fail_closed(self):
        wrong=self.baselines.copy();wrong.loc[0,'net_log_return']+=.01
        with self.assertRaises(ValueError):evaluation.evaluate_ledgers(self.frame,wrong,self.cfg,self.configs)
        with self.assertRaises(ValueError):evaluation.evaluate_ledgers(self.frame.iloc[1:],self.baselines,self.cfg,self.configs)
        with self.assertRaises(ValueError):evaluation.evaluate_ledgers(self.frame.loc[lambda x:x.seed.ne(89)],self.baselines,self.cfg,self.configs)

    def test_report_keeps_worst_risk_and_exploratory_scope(self):
        result={**self.result,'baseline_metrics':self.metrics}
        with tempfile.TemporaryDirectory() as temp:
            evaluation._report(Path(temp),result)
            text=(Path(temp)/'index.html').read_text()
            for expected in ['worst_drawdown_seed','標準偏差','未使用Holdout','post-hoc','全14','seed追加は市場標本数を増やしません']:
                self.assertIn(expected,text)

    def test_manifest_barrier_hash_provenance_to_report_roundtrip(self):
        from unidream.experiments import wm_seed_robustness_benchmark as benchmark
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg={**self.cfg,'output_dir':str(root)}
            close_path=root/'raw_close.npy';time_path=root/'raw_index_ns.npy'
            np.save(close_path,self.close);np.save(time_path,self.times)
            cfg['data']={'raw_close_path':str(close_path),'raw_index_ns_path':str(time_path)}
            (root/'preregistration.json').write_text('{"synthetic_only":true}')
            (root/'test_barrier.json').write_text('{"synthetic_only":true}')
            reg=evaluation._sha(root/'preregistration.json');jobs=[]
            for (condition,seed),part in self.frame.groupby(['condition','seed']):
                path=root/f'{condition}_{seed}.parquet';part.to_parquet(path,index=False)
                jobs.append({'job':f'{condition}_{seed}','condition':condition,'fold':0,'seed':int(seed),
                    'path':str(path),'sha256':evaluation._sha(path),'reused':seed in [7,19,42]})
            manifest={'complete':True,'registration_sha256':reg,
                'barrier_sha256':evaluation._sha(root/'test_barrier.json'),'jobs':jobs}
            manifest_path=root/'evaluation_manifest.json';manifest_path.write_text(json.dumps(manifest))
            with patch.object(benchmark,'verify_test_barrier',return_value={'synthetic_only':True}) as guard:
                result=evaluation.aggregate(cfg,reg)
                self.assertEqual(guard.call_count,2)
                self.assertTrue(result['complete']);self.assertEqual(result['n_evaluations'],30)
                self.assertEqual(result['n_seeds'],10);self.assertEqual(result['n_market_rows'],40*96)
                for name,digest in result['artifact_sha256'].items():self.assertEqual(evaluation._sha(root/name),digest)
                audit=json.loads((root/'final/audit.json').read_text())
                self.assertEqual(audit['n_reused_evaluations'],9);self.assertEqual(audit['n_new_evaluations'],21)
                self.assertEqual(len(pd.read_csv(root/'final/primary_inference.csv')),14)
                manifest['barrier_sha256']='changed';manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'complete registered evaluation manifest'):
                    evaluation.aggregate(cfg,reg)

    def test_test_barrier_precedes_any_ledger_reads(self):
        from unidream.experiments import wm_seed_robustness_benchmark as benchmark
        with patch.object(benchmark,'verify_test_barrier',side_effect=ValueError('missing barrier')),patch.object(pd,'read_parquet') as read:
            with self.assertRaisesRegex(ValueError,'missing barrier'):evaluation.aggregate({'output_dir':'/not-read'},'reg')
            read.assert_not_called()


if __name__=='__main__':unittest.main()
