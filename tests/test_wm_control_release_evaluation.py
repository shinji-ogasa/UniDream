"""Synthetic continuous-account, causal-control and fixed-family evaluation tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_control_release_evaluation as evaluation


def config(n=40*96):
    return {"seeds":[7,19],"reused_seeds":[7],"new_seeds":[19],
        "accounting":{"position_min":0.,"position_max":1.12,"max_position_step":.08,
            "initial_position":1.,"cost_rate":.00055,"borrow_apr":.10,"annual_bars":35040},
        "condition_definitions":dict(zip(evaluation.CONDITIONS,[
            {"position_max":cap,"max_position_step":speed,"bc_prior_coefficient":prior}
            for cap,speed,prior in [(1.12,.08,.01),(1.5,.08,.01),(2.,.08,.01),(2.,.24,.01),(2.,.24,0.)]])),
        "baselines":{"ma_window_bars":8,"volatility_window_bars":4,"volatility_target_annual":.10},
        "statistics":{"methods":["circular"],"block_days":[7,14,30],"bootstrap_replicates":99,
            "seed":23,"hac_lags":[6,13,29],"expected_market_rows":n},"runtime":{"cpu_threads":1}}


def ledger(returns,start=20):
    r=np.asarray(returns,float);n=len(r);equity=np.cumprod(1+r)
    death=r==-1;bankrupt=equity==0
    logs=np.zeros(n);logs[death]=-np.inf;logs[~death]=np.log1p(r[~death])
    return pd.DataFrame({"timestamp":pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC'),
        "decision_id":np.arange(start,start+n),"fold":np.arange(n)//max(n//2,1),
        "equity":equity,"net_simple_return":r,"net_log_return":logs,
        "executed_exposure":np.where(bankrupt,0.,.5),"turnover":0.,"fee":0.,"borrow":0.,
        "bankrupt":bankrupt,"bankrupt_event":death,"alive":~bankrupt,"decision_available":~bankrupt})


def fixture(days=40):
    n=days*96;cfg=config(n);rng=np.random.default_rng(23)
    market=rng.normal(.00001,.0003,n);paths={}
    for kind in ['trained','frozen']:
        for arm in evaluation.ARMS:
            for j,condition in enumerate(evaluation.CONDITIONS):
                for seed in cfg['seeds']:
                    r=market*(.4+.02*j)+(.0000005*j if kind=='trained' else 0.)
                    if arm=='C':r=r+.0000001
                    paths[(kind,arm,condition,seed)]=ledger(r)
    bases={(c,b):ledger(market*.3) for c in evaluation.CONDITIONS for b in evaluation.BASELINES}
    bases[('descriptive','bh')]=ledger(market)
    bases[('descriptive','constant08')]=ledger(market*.08)
    return cfg,paths,bases


class ContinuousMetricsTests(unittest.TestCase):
    def test_direct_product_drawdown_fees_and_fold_boundary_no_reset(self):
        path=ledger([.1,-.2,.1,.05]);path['fee']=[.001,.002,.003,.004];path['borrow']=[.01,.02,.03,.04]
        m=evaluation.metrics(path,config(4))
        self.assertAlmostEqual(m['net_total_return'],np.prod([1.1,.8,1.1,1.05])-1)
        self.assertAlmostEqual(m['max_drawdown'],.2)
        self.assertAlmostEqual(m['fee_quote'],.01);self.assertAlmostEqual(m['borrow_quote'],.1)
        self.assertEqual(m['n_resets'],0);self.assertEqual(m['n_rows'],4)
        wrong=path.copy();wrong.loc[2,'equity']=1.
        with self.assertRaisesRegex(ValueError,'resets/injections'):evaluation.metrics(wrong,config(4))

    def test_ruin_retains_minus100_and_minusinf_and_absorbs(self):
        path=ledger([.1,-1.,0.,0.]);m=evaluation.metrics(path,config(4))
        self.assertEqual(m['net_total_return'],-1.);self.assertEqual(m['annualized_net_log_growth'],-np.inf)
        self.assertEqual(m['max_drawdown'],1.);self.assertTrue(m['insolvent']);self.assertFalse(m['log_metric_available'])
        wrong=path.copy();wrong.loc[1,'net_log_return']=-27.
        with self.assertRaisesRegex(ValueError,'-inf'):evaluation.metrics(wrong,config(4))
        wrong=path.copy();wrong.loc[3,'fee']=.01
        with self.assertRaisesRegex(ValueError,'dead accounts'):evaluation.metrics(wrong,config(4))

    def test_period_after_ruin_not_zero_percent_profit(self):
        path=ledger([.1,-1.,0.,0.])
        m=evaluation.metrics(path.iloc[2:],{**config(2),'_period_start_equity':0.})
        self.assertIsNone(m['net_total_return']);self.assertIsNone(m['max_drawdown'])
        self.assertFalse(m['percentage_metrics_available'])

    def test_actual_exposure_can_drift_beyond_target_cap(self):
        path=ledger([.01,-.05,.01]);path['executed_exposure']=[1.12,1.4,2.5]
        m=evaluation.metrics(path,config(3))
        self.assertEqual(m['exposure_max'],2.5);self.assertAlmostEqual(m['above15_fraction'],1/3)

    def test_missing_calendar_bar_rejected_instead_of_cohort_compression(self):
        path=ledger(np.zeros(10)).drop(index=4)
        with self.assertRaisesRegex(ValueError,'every consecutive'):evaluation.metrics(path,config(10))


class CausalContinuousControlTests(unittest.TestCase):
    def test_current_and_future_close_never_enters_baseline_signal(self):
        cfg=config(5);close=100*np.exp(np.arange(60)*.0001);ohlc=np.repeat(close[:,None],4,axis=1)
        times=np.arange(60,dtype=np.int64)*evaluation.BAR_NS;ids=np.arange(20,25)
        before=evaluation.baseline_targets(ohlc,times,ids,cfg)
        changed=ohlc.copy();changed[20:,3]*=1.5
        after=evaluation.baseline_targets(changed,times,ids,cfg)
        for condition in evaluation.CONDITIONS:
            for baseline in evaluation.BASELINES:
                self.assertEqual(before[condition][baseline][0],after[condition][baseline][0])
        sigma=np.sqrt(35040)*.0001
        self.assertAlmostEqual(before['cap20']['voltarget10'][0],min(2.,.1/sigma),places=10)
        self.assertTrue((before['cap20']['ma20d']==2).all())

    def test_missing_raw_history_not_filled(self):
        cfg=config(5);ohlc=np.full((60,4),100.);times=np.arange(60,dtype=np.int64)*evaluation.BAR_NS
        ohlc[18,3]=np.nan
        with self.assertRaisesRegex(ValueError,'complete causal raw'):evaluation.baseline_targets(ohlc,times,np.arange(20,25),cfg)

    def test_pending_orders_survive_unavailable_bar_and_bh_is_true_hold(self):
        cfg=config(12);close=100*np.exp(np.arange(60)*.001);ohlc=np.column_stack([close,close,close*.99,close])
        times=np.arange(60,dtype=np.int64)*evaluation.BAR_NS;ids=np.arange(20,32)
        reference=pd.DataFrame({'decision_id':ids,'timestamp':pd.to_datetime(times[ids],utc=True),'fold':np.r_[np.zeros(6,int),np.ones(6,int)]})
        available=np.ones(60,bool);available[21:23]=False
        bases=evaluation.build_baselines(ohlc,times,available,reference,cfg)
        path=bases[('control','constant_cap')]
        self.assertEqual(path.fee.iloc[0],0.)
        self.assertGreater(path.fee.iloc[1],0.) # Prior pending fills even when today's decision is absent.
        self.assertEqual(path.turnover.iloc[2],0.)
        self.assertFalse(path.decision_available.iloc[1]);self.assertFalse(path.decision_available.iloc[2])
        self.assertNotEqual(path.equity.iloc[6],1.) # Fold change does not recapitalize.
        bh=bases[('descriptive','bh')]
        self.assertEqual(bh.fee.sum(),0.);self.assertEqual(bh.borrow.sum(),0.);self.assertEqual(bh.turnover.sum(),0.)
        self.assertAlmostEqual(bh.equity.iloc[-1],close[ids[-1]]/close[ids[0]-1],places=12)
        constant=bases[('descriptive','constant08')]
        self.assertAlmostEqual(constant.executed_exposure.iloc[0],1.)
        self.assertGreater(constant.executed_exposure.iloc[1],.08)


class FixedFamilyInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.paths,cls.bases=fixture();cls.result=evaluation.evaluate_paths(cls.paths,cls.bases,cls.cfg)

    def test_all53_hypotheses_and_single_market_calendar_retained(self):
        r=self.result;p=r['primary_inference']
        self.assertEqual(len(p),53);self.assertEqual(p.groupby('family').size().to_dict(),{'primary':13,'secondary':40})
        self.assertEqual(len(r['inference']),318);self.assertEqual(len(r['hac_inference']),159)
        self.assertEqual(set(p.n_unique_market_rows),{40*96});self.assertEqual(set(p.n_training_seeds),{2})
        self.assertEqual(set(p.n_segments),{1});self.assertFalse(r['audit']['seed_count_multiplies_market_rows'])
        self.assertTrue((p.p_holm_all53+1e-14>=p.p_holm_family).all())

    def test_duplicating_seed_does_not_shrink_market_confidence_interval(self):
        cfg=copy.deepcopy(self.cfg);cfg['seeds']=[7];cfg['new_seeds']=[]
        paths={key:path for key,path in self.paths.items() if key[-1]==7}
        other=evaluation.evaluate_paths(paths,self.bases.copy(),cfg)['primary_inference'].set_index('condition')
        main=self.result['primary_inference'].set_index('condition')
        for column in ['estimate','ci_low','ci_high','p_value']:
            np.testing.assert_allclose(main.loc[other.index,column],other[column],rtol=0,atol=1e-14)

    def test_one_ruined_seed_disables_whole_paired_contrast_not_family(self):
        paths=self.paths.copy();key=('trained','C','free',19)
        path=paths[key].copy();death=20*96
        path.loc[death,'net_simple_return']=-1.;path.loc[death+1:,'net_simple_return']=0.
        path.loc[death,'net_log_return']=-np.inf;path.loc[death+1:,'net_log_return']=0.
        path.loc[death:,'equity']=0.;path.loc[death:,'executed_exposure']=0.
        path.loc[death:,'bankrupt']=True;path.loc[death,'bankrupt_event']=True
        path.loc[death:,'alive']=False;path.loc[death:,'decision_available']=False
        paths[key]=path;r=evaluation.evaluate_paths(paths,self.bases.copy(),self.cfg)
        main=r['primary_inference'];na=main.loc[~main.inference_available]
        self.assertEqual(len(main),53);self.assertEqual(len(na),6)
        self.assertTrue(na.p_value.eq(1).all());self.assertTrue(na.p_holm_all53.eq(1).all())
        self.assertTrue(na.estimate.isna().all());self.assertFalse(na.significant_improvement.any())
        pooled=r['pooled_metrics'].loc[lambda x:x.kind.eq('trained')&x.arm.eq('C')&x.condition.eq('free')&x.seed.eq(19)].iloc[0]
        self.assertEqual(pooled.net_total_return,-1.);self.assertEqual(pooled.annualized_net_log_growth,-np.inf)
        report_summary=r['seed_summary'].loc[lambda x:x.kind.eq('trained')&x.arm.eq('C')&x.condition.eq('free')&x.seed_subset.eq('all_seeds')].iloc[0]
        self.assertEqual(report_summary.n_insolvent_seeds,1);self.assertEqual(report_summary.worst_return,-1.)

    def test_calendar_or_condition_missing_fails_closed(self):
        wrong=self.paths.copy();wrong.pop(next(iter(wrong)))
        with self.assertRaisesRegex(ValueError,'every registered'):evaluation.evaluate_paths(wrong,self.bases,self.cfg)
        wrong=self.paths.copy();key=next(iter(wrong));wrong[key]=wrong[key].drop(index=100)
        with self.assertRaisesRegex(ValueError,'every consecutive'):evaluation.evaluate_paths(wrong,self.bases,self.cfg)

    def test_fold_and_month_periods_compound_without_account_reset(self):
        pooled=self.result['pooled_metrics'];folded=self.result['fold_metrics']
        for key,part in folded.groupby(['kind','arm','condition','seed']):
            p=pooled.loc[(pooled[['kind','arm','condition','seed']]==key).all(axis=1)].iloc[0]
            self.assertAlmostEqual(np.prod(1+part.net_total_return)-1,p.net_total_return,places=12)
            self.assertEqual(part.n_resets.sum(),0)

    def test_html_preserves_ruin_and_exploratory_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation._report(Path(directory),self.result);page=(Path(directory)/'index.html').read_text()
            for text in ['N/A','破産','未使用Holdout','無料リセットはありません','53','95%市場CIではありません']:
                self.assertIn(text,page)

    def test_all_ruined_hypotheses_remain_unavailable_and_p1(self):
        cfg,paths,bases=fixture(days=2)
        for key,path in paths.items():
            path.loc[1,'net_simple_return']=-1.;path.loc[2:,'net_simple_return']=0.
            path.loc[1,'net_log_return']=-np.inf;path.loc[2:,'net_log_return']=0.
            path.loc[1:,'equity']=0.;path.loc[1:,'executed_exposure']=0.
            path.loc[1:,'bankrupt']=True;path.loc[1,'bankrupt_event']=True
            path.loc[1:,'alive']=False;path.loc[1:,'decision_available']=False
        r=evaluation.evaluate_paths(paths,bases,cfg)
        self.assertEqual(len(r['inference']),318)
        self.assertFalse(r['inference'].inference_available.any())
        self.assertTrue(r['inference'].p_value.eq(1).all());self.assertTrue(r['inference'].p_holm_all53.eq(1).all())
        self.assertFalse(r['inference'].significant_improvement.any())

    def test_test_barrier_precedes_all_result_reads(self):
        from unidream.experiments import wm_control_release_benchmark as benchmark
        with patch.object(benchmark,'verify_test_barrier',side_effect=ValueError('closed')),patch.object(pd,'read_parquet',side_effect=AssertionError('Forbidden before barrier')):
            with self.assertRaisesRegex(ValueError,'closed'):evaluation.aggregate({'output_dir':'/does/not/exist'},'no')

    def test_guarded_manifest_to_real_synthetic_report_and_tamper_detection(self):
        cfg,paths,bases=fixture(days=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg['output_dir']=str(root)
            (root/'preregistration.json').write_text('{"synthetic":true}')
            registration_sha=evaluation._sha(root/'preregistration.json')
            (root/'test_barrier.json').write_text('{"synthetic":true}')
            entries=[]
            for i,(key,path) in enumerate(paths.items()):
                filename=root/f'path_{i}.parquet';path.to_parquet(filename,index=False)
                entries.append(dict(zip(('kind','arm','condition','seed'),key),path_id=str(i),path=str(filename),
                    initial_equity=1.,sha256=evaluation._sha(filename)))
            reference=next(iter(paths.values()));nraw=int(reference.decision_id.max())+1
            times=(np.arange(nraw)-20)*evaluation.BAR_NS+reference.timestamp.iloc[0].value
            raw={'ohlc':np.full((nraw,4),100.),'index_ns':times,'decision_available':np.ones(nraw,bool)}
            cfg['data']={};raw_info={}
            for name,key in [('ohlc','raw_ohlc_path'),('index_ns','raw_index_ns_path'),('decision_available','decision_available_path')]:
                filename=root/f'{name}.npy';np.save(filename,raw[name]);cfg['data'][key]=str(filename)
                raw_info[name]={'path':str(filename),'sha256':evaluation._sha(filename)}
            manifest={'complete':True,'registration_sha256':registration_sha,'barrier_sha256':evaluation._sha(root/'test_barrier.json'),
                      'paths':entries,'raw_arrays':raw_info}
            (root/'path_manifest.json').write_text(json.dumps(manifest))
            from unidream.experiments import wm_control_release_benchmark as benchmark
            with patch.object(benchmark,'verify_test_barrier',return_value=None),patch.object(evaluation,'build_baselines',return_value=bases):
                result=evaluation.aggregate(cfg,registration_sha)
                self.assertTrue(result['complete']);self.assertEqual(result['n_paths'],40)
                self.assertEqual(result['n_market_rows'],192)
                self.assertTrue((root/'final/index.html').exists())
                for name,digest in result['artifact_sha256'].items():self.assertEqual(evaluation._sha(root/name),digest)
                with (root/'path_0.parquet').open('ab') as stream:stream.write(b'tampered')
                with self.assertRaisesRegex(ValueError,'SHA changed'):evaluation.aggregate(cfg,registration_sha)


if __name__=='__main__':unittest.main()
