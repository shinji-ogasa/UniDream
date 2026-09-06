"""Read-only Stage13 source audit. No slope, covariance, loss, target, or policy calculation."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, subprocess
import numpy as np
import pandas as pd
import yaml

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
PARENT_REVISION='f28412454deaada9cb81fabdceb0e853e451a876'
PARENT_RESULT_SHA256='c19b772e4f53b5a749839553b82c439ef4684a2ec9efe523464c533e8beed366'
PARENT_PREFLIGHT_SHA256='4dcf2dacf2ca6a43d23d90ffc0c8cd4d8bdfd3be6015978e3a98baae221a86ae'
FOLDS=tuple(range(5,13))
GROUPS=('technical','perp_delay0')
MEANS=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();root=args.root.resolve()
    parent=root/'codex_outputs/oracle_frozen_procedure_parity_v1';delay=root/'codex_outputs/oracle_derivative_delay_v1'
    verified={};bindings={};maxima={};counts=Counter();records=[]
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def verify(p,h):
        path=Path(p).resolve()
        if path not in verified:verified[path]=sha(path)
        assert verified[path]==h,('hash mismatch',str(path));counts['hash_binding_checks']+=1
    def arr(p,keys=None):
        with np.load(p,allow_pickle=False) as z:
            if keys is not None:assert set(z.files)==set(keys),('schema',str(p),z.files)
            a={k:z[k] for k in z.files}
        assert all(v.dtype.kind in 'bifu' and not np.isinf(v).any() for v in a.values()),('invalid array',str(p))
        return a
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True),('exact mismatch',name)
    def close(name,a,b,tol=1e-12):
        a,b=np.asarray(a,float),np.asarray(b,float)
        assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),('invalid comparison',name)
        fin=np.isfinite(a);d=float(np.max(np.abs(a[fin]-b[fin]))) if fin.any() else 0.
        maxima[name]=max(maxima.get(name,0.),d);assert d<=tol,('numeric mismatch',name,d)
    def mask_digest(index,mask):return hashlib.sha256(index.asi8.astype('<i8').tobytes()+np.asarray(mask,'u1').tobytes()).hexdigest()
    reg,pre,result=(read(parent/(n+'.json')) for n in ('registration','preflight','results'))
    assert reg['source_revision']==PARENT_REVISION and digest(reg)==result['registration_sha256'] and result['parity_pass'] is True
    verify(parent/'results.json',PARENT_RESULT_SHA256);verify(parent/'preflight.json',PARENT_PREFLIGHT_SHA256)
    cfg=reg['config'];configpath=root/'configs/oracle_frozen_procedure_parity_20260906.yaml'
    verify(configpath,reg['config_sha256']);assert yaml.safe_load(configpath.read_text())==cfg
    assert reg['preflight_sha256']==cfg['preflight_sha256']==PARENT_PREFLIGHT_SHA256
    assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert cfg['development_folds']==list(FOLDS) and cfg['calendar_evaluation_folds']==list(range(4,12)) and cfg['additional_periods_permitted'] is False and cfg['selection_permitted'] is False
    assert pre['additional_periods_accessed'] is False and pre['new_forecast_or_policy_computed'] is False
    assert pre['source_sha256']==reg['source_sha256']
    for p,h in cfg['metadata_bindings'].items():verify(root/p,h)
    for name,h in reg['source_sha256'].items():
        path='unidream/experiments/'+name;verify(root/path,h)
        assert hashlib.sha256(subprocess.check_output(['git','show',PARENT_REVISION+':'+path],cwd=root)).hexdigest()==h
    bindings=dict(pre['source_artifact_sha256']);assert len(bindings)==1064
    for f in FOLDS:
        fold=read(parent/f'fold_{f}.json');assert fold['registration_sha256']==digest(reg) and len(fold['artifact_sha256'])==33
        for p,h in fold['artifact_sha256'].items():
            assert p not in bindings or bindings[p]==h
            bindings[p]=h
    assert len(bindings)==1328
    for p,h in bindings.items():verify(root/p,h)
    dreg,dpre=(read(delay/(n+'.json')) for n in ('registration','preflight'))
    assert dreg['source_revision']=='a75b55a9994aacbacd69ff130cb79d8293292033'
    dc=dreg['config'];verify(root/'configs/oracle_derivative_delay_20260905.yaml',dreg['config_sha256'])
    assert yaml.safe_load((root/'configs/oracle_derivative_delay_20260905.yaml').read_text())==dc
    fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());cut=pd.Timestamp(fc['data_cutoff']);assert cut==pd.Timestamp('2023-04-16T13:45:00Z')
    sp=Path(fc['data_path']);side=read(sp.with_suffix('.sha256.json'));proof=pre['spot_data_proof'];assert proof==dpre['spot_data_proof']
    for path,key in [(sp,'artifact_sha256'),(sp.with_suffix('.sha256.json'),'sidecar_sha256'),(side['availability_path'],'availability_sha256'),(side['source_ledger_path'],'ledger_sha256')]:verify(path,proof[key])
    um=pre['um_data_proof'];assert um==dpre['um_data_proof'];up=root/um['data_path']
    for path,key in [(up,'data_sha256'),(up.with_suffix('.sha256.json'),'sidecar_sha256'),(root/um['availability_path'],'availability_sha256'),(root/um['source_ledger_path'],'source_ledger_sha256'),(root/um['registration_path'],'registration_sha256')]:verify(path,um[key])
    print(json.dumps({'phase':'all_1328_source_bindings_verified','distinct_files':len(verified)}),flush=True)
    bars=pd.read_parquet(sp,filters=[('bar_open_ts','<',cut)]);assert bars.index.is_unique and bars.index.is_monotonic_increasing and bars.index.max()<cut
    fullindex=pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC');bars=bars.reindex(fullindex)
    bars['bar_available']=bars[['open','high','low','close']].notna().all(axis=1)
    opens=bars.open.to_numpy();closes=bars.close.to_numpy();available=bars.bar_available.to_numpy()
    def existing_actual(index,selected):
        # Reconstruct only previously retained calibration/scoring labels, never new tail outcomes.
        result=np.full((len(index),3),np.nan);positions=fullindex.get_indexer(index)
        assert (positions>=0).all()
        for local in np.flatnonzero(selected):
            t=int(positions[local]);entry=float(opens[t+1]);path=[float(v) for v in closes[t+1:t+25]]
            assert t+24<len(fullindex) and len(path)==24 and available[t+1:t+25].all() and math.isfinite(entry) and entry>0 and all(math.isfinite(v) and v>0 for v in path)
            terms=[math.log(path[0]/entry)**2]+[math.log(path[j]/path[j-1])**2 for j in range(1,24)]
            result[local]=[math.log(path[-1]/entry),max(-math.log(min(path)/entry),0.),math.sqrt(math.fsum(terms))]
        return result
    keys={'timestamps','actual','mu','variance','fit_return_mean','score_support','inference_mask'}
    rawkeys=keys|{'raw_log_variance','persistence96_variance'}
    for f in FOLDS:
        E=pd.Timestamp('2021-04-16T13:45:00Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3)
        S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3);ix=pd.date_range(E,end,freq='15min',inclusive='left');ci=pd.date_range(S,E,freq='15min',inclusive='left')
        assert end<=cut
        support=next(s for s in pre['support'] if s['reference_validation_fold']==f);old=next(s for s in dpre['folds'] if s['fold']==f)
        assert support['adapter_calendar_test_fold']==f-1 and support['same_already_observed_interval'] is True
        for k,v in [('fit_start',E-pd.DateOffset(months=24)),('fit_end',S),('scale_start',S),('scale_end',I),('interval_start',I),('interval_end',E),('evaluation_start',E),('evaluation_end',end)]:assert pd.Timestamp(support[k])==v
        forecasts={m:arr(parent/'forecasts'/f'fold{f}_{m}.npz',keys) for m in MEANS};anchor=forecasts['scale_mean'];inf=anchor['inference_mask'];score=anchor['score_support']
        assert inf.dtype==bool and score.dtype==bool and not (score&~inf).any();clock=np.asarray((ix.hour%6==0)&(ix.minute==0));assert not (inf&~clock).any()
        assert (ix[score]+pd.Timedelta(minutes=375)<=end).all()
        expected=existing_actual(ix,score)
        for name,p in forecasts.items():
            exact('forecast calendar',p['timestamps'],ix.asi8)
            for k in ('inference_mask','score_support','actual','variance','fit_return_mean'):exact('shared forecast '+k,p[k],anchor[k])
            assert np.array_equal(np.isfinite(p['mu']),inf) and np.array_equal(np.isfinite(p['variance']),inf) and (p['variance'][inf]>=0).all() and np.isnan(p['actual'][~score]).all()
            close('retained_evaluation_actual_vs_raw',p['actual'],expected);counts['parent_forecasts_verified']+=1
        for name,mask in [('inference',inf),('score',score)]:
            complete=np.zeros(len(fullindex),bool);complete[fullindex.get_indexer(ix)]=mask
            assert mask_digest(fullindex,complete)==support['mask_sha256'][name] and int(mask.sum())==support['counts'][name]
        doc=read(parent/'calibration'/f'fold{f}_provenance.json');cal=doc['calibration'];assert cal['counts']=={k:old['counts'][k] for k in ('fit','scale','interval')}
        assert doc['provenance']['mask_counts']=={k:old['counts'][k] for k in ('fit','scale','interval','predict','inference')}
        assert len(anchor['mu'][inf]) and np.all(anchor['mu'][inf]==anchor['mu'][inf][0]);a=float(anchor['mu'][inf][0]);assert cal['scale_mean']==a
        cps={}
        for group in GROUPS:
            ck={'timestamps','actual','scale_mask','interval_mask','mu'}|({'log_variance','variance'} if group=='technical' else set())
            path=parent/'calibration'/f'fold{f}_{group}.npz';cp=arr(path,ck);cps[group]=cp;exact('calibration calendar',cp['timestamps'],ci.asi8)
            scale,interval=cp['scale_mask'],cp['interval_mask'];assert scale.dtype==bool and interval.dtype==bool and not (scale&interval).any()
            schedule=np.asarray((ci.hour%6==0)&(ci.minute==0))
            for label,mask,start,stop in [('scale',scale,S,I),('interval',interval,I,E)]:
                assert not (mask&~schedule).any() and (ci[mask]>=start).all() and (ci[mask]+pd.Timedelta(minutes=375)<stop).all()
                assert mask.sum()>=64 and int(mask.sum())==support['counts'][label]==cal['counts'][label]
                complete=np.zeros(len(fullindex),bool);complete[fullindex.get_indexer(ci)]=mask
                assert mask_digest(fullindex,complete)==support['mask_sha256'][label]
            selected=scale|interval;assert np.isfinite(cp['actual'][selected]).all() and np.isnan(cp['actual'][~selected]).all() and np.isfinite(cp['mu'][selected]).all()
            close('retained_calibration_actual_vs_raw',cp['actual'],existing_actual(ci,selected))
            oldcal=arr(delay/'calibration'/f'fold{f}_{group}.npz')
            for k,v in cp.items():
                if k in ('timestamps','actual','scale_mask','interval_mask'):exact('calibration ancestral '+k,v,oldcal[k])
                else:close('calibration ancestral '+k,v,oldcal[k],1e-14)
            # Existing anchors and bias only: no slope or residual/covariance statistic is calculated.
            recreated_anchor=math.fsum(float(v)/int(scale.sum()) for v in cp['actual'][scale,0]);assert recreated_anchor==a
            bias=float(cal['return_bias'][group]);assert math.isfinite(bias)
            close('saved_bias_vs_scale_residual_mean',bias,float(np.mean(cp['actual'][scale,0]-cp['mu'][scale])),1e-14)
            raw=arr(delay/'forecasts'/f'fold{f}_{group}_raw.npz',rawkeys)
            exact('raw forecast calendar',raw['timestamps'],ix.asi8)
            for k in ('actual','score_support','inference_mask','fit_return_mean'):exact('raw forecast pairing '+k,raw[k],anchor[k])
            scaled=forecasts[group+'_scaled'];close('raw_plus_saved_bias_vs_full_endpoint',raw['mu']+bias,scaled['mu'],1e-14)
            half='technical_half' if group=='technical' else 'perp_delay0_half';exact('existing half endpoint identity',forecasts[half]['mu'],.5*scaled['mu']+.5*anchor['mu'])
            # Raw mean is an out-of-fit prediction on both retained calibration segments.
            assert doc['provenance']['mask_ranges']['fit'][1]<doc['provenance']['mask_ranges']['predict'][0]
            records.append({'fold':f,'group':group,'fit_count':cal['counts']['fit'],'scale_count':int(scale.sum()),'interval_count':int(interval.sum()),'inference_count':int(inf.sum()),'score_count':int(score.sum()),'scale_start':S.isoformat(),'scale_end_exclusive':I.isoformat(),'interval_end_exclusive':E.isoformat(),'evaluation_end_exclusive':end.isoformat(),'calibration_sha256':sha(path),'raw_evaluation_forecast_sha256':sha(delay/'forecasts'/f'fold{f}_{group}_raw.npz'),'saved_endpoint_sha256':sha(parent/'forecasts'/f'fold{f}_{group}_scaled.npz'),'anchor_sha256':sha(parent/'forecasts'/f'fold{f}_scale_mean.npz'),'saved_bias_source_sha256':sha(parent/'calibration'/f'fold{f}_provenance.json')})
            counts['calibration_streams_verified']+=1;counts['raw_evaluation_streams_paired']+=1;counts['exact_saved_scale_anchors']+=1;counts['exact_existing_half_identities']+=1
        for k in ('timestamps','actual','scale_mask','interval_mask'):exact('technical/perp calibration pairing '+k,cps['technical'][k],cps['perp_delay0'][k])
        counts['fold_indexed_scale_rows']+=int(cps['technical']['scale_mask'].sum());counts['fold_indexed_interval_rows']+=int(cps['technical']['interval_mask'].sum());counts['inference_rows_across_folds']+=int(inf.sum());counts['score_rows_across_folds']+=int(score.sum())
        print(json.dumps({'phase':'existing_calibration_and_forecast_contract_verified','fold':f}),flush=True)
    assert counts['calibration_streams_verified']==16 and counts['parent_forecasts_verified']==40 and counts['raw_evaluation_streams_paired']==16
    assert counts['inference_rows_across_folds']==2586 and counts['score_rows_across_folds']==2574
    # Counts sum fold-indexed supports; reused calendar timestamps are not independent observations.
    report={'status':'pass','scope':'Data-only inherited source, immutable prediction and calibration support audit; no Stage13 slope, loss, covariance, new forecast or policy calculation.','audit_script_sha256':sha(__file__),'parent_source_revision':PARENT_REVISION,'parent_results_sha256':PARENT_RESULT_SHA256,'parent_preflight_sha256':PARENT_PREFLIGHT_SHA256,'verified_source_artifacts':1328,'distinct_hashed_files':len(verified),'counts':dict(counts),'max_absolute_differences':maxima,'stream_records':records,'source_artifact_inventory_sha256':digest(bindings),'spot_data_proof':proof,'um_data_proof':um,'data_cutoff_exclusive':cut.isoformat(),'new_slope_computed':False,'new_loss_computed':False,'new_policy_computed':False,'new_model_fitted':False,'additional_period_values_read':False,'limitations':['Source bytes are fully hashed, while semantic raw Spot reads are filtered strictly before the original development cutoff.','This audit inherits the established retrospective common feature-availability mask; it verifies saved mask hashes and label timing, not historical receipt evidence.','Scale and interval arrays are already observed and overlap other folds calendar periods; row sums do not count independent observations.','Saved endpoint bias uses mean(y-raw), while anchor uses math.fsum(y/n); raw-plus-bias is preserved and is not asserted bit-identical to raw-centered OLS.']}
    path=Path('/tmp/oracle_mean_reliability_source_audit_20260906.json');path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'pass','report':str(path),'counts':dict(counts),'maxima':maxima}),flush=True)

if __name__=='__main__':main()
