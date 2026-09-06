"""Stage14 DATA-ONLY source and rolling-history availability audit.
Never calculates return labels, means, covariance, weights, losses or forecasts.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,subprocess
import numpy as np
import pandas as pd
import yaml

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
SOURCE_REVISION='ff9bbb92d588b4615ae8352b353a1301751202a7'
RESULT_SHA='333f88d4bc06f671d552d8ca70470ee60ecd67074812f6aa248b26f1b94562f1'
PREFLIGHT_SHA='d1304dc98df0595b68992b6605c78bd3527d3b811209071df12cb6594d294370'
CONFIG_SHA='71b0965691099074a690085ac22cbe6808b2b430fea305ec2b75480db1c2f094'
GROUPS=('technical','perp_delay0');FOLDS=tuple(range(5,13))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();root=args.root.resolve()
    src=root/'codex_outputs/oracle_mean_reliability_decisions_v1';parity=root/'codex_outputs/oracle_frozen_procedure_parity_v1';delay=root/'codex_outputs/oracle_derivative_delay_v1'
    counts=Counter();verified={};records=[];stream_bindings=[]
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def verify(p,h):
        p=Path(p).resolve()
        if p not in verified:verified[p]=sha(p)
        assert verified[p]==h,('hash',str(p));counts['hash_binding_checks']+=1
    def mask_sha(ix,m):return hashlib.sha256(ix.asi8.astype('<i8').tobytes()+np.asarray(m,'u1').tobytes()).hexdigest()
    def exact(a,b,name):
        assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True),('array',name)
    reg,pre,res=(read(src/(p+'.json')) for p in ('registration','preflight','results'));cfg=reg['config']
    assert reg['source_revision']==SOURCE_REVISION and digest(reg)==res['registration_sha256']
    verify(src/'results.json',RESULT_SHA);verify(src/'preflight.json',PREFLIGHT_SHA)
    cfgpath=root/'configs/oracle_mean_reliability_decisions_20260906.yaml';verify(cfgpath,CONFIG_SHA)
    assert reg['config_sha256']==CONFIG_SHA and yaml.safe_load(cfgpath.read_text())==cfg and cfg['preflight_sha256']==reg['preflight_sha256']==PREFLIGHT_SHA
    assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert reg['source_bindings']==cfg['source_bindings']==pre['source_bindings']
    assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE_REVISION+':'+str(cfgpath.relative_to(root))],cwd=root)).hexdigest()==CONFIG_SHA
    for p,h in cfg['source_bindings'].items():
        verify(root/p,h);assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE_REVISION+':'+p],cwd=root)).hexdigest()==h
    parent_pre=pre['parent_prepare_preflight'];bindings=dict(pre['source_artifact_bindings']);assert len(bindings)==1328 and bindings==parent_pre['source_artifact_bindings']
    source_cfg=root/cfg['source_prepare_config'];verify(source_cfg,cfg['source_prepare_config_sha256']);pc=yaml.safe_load(source_cfg.read_text())
    assert parent_pre['config_contract_sha256']==digest({k:v for k,v in pc.items() if k!='preflight_sha256'})
    manifests={}
    for p,h in pc['source_manifest_bindings'].items():verify(root/p,h);manifests[p]=h
    own={}
    for f in FOLDS:
        path=src/f'fold_{f}.json';fold=read(path);assert fold['registration_sha256']==digest(reg)
        for k in ('rows','scores','fits'):assert fold[k]==[v for v in res[k] if v['fold']==f]
        assert len(fold['artifact_sha256'])==26
        for p,h in fold['artifact_sha256'].items():
            assert p not in bindings or bindings[p]==h
            own[p]=h;bindings[p]=h
        manifests[str(path.relative_to(root))]=sha(path)
    assert len(own)==208 and len(bindings)==1536
    for p,h in bindings.items():verify(root/p,h)
    for n in ('registration','preflight','results'):manifests[str((src/(n+'.json')).relative_to(root))]=sha(src/(n+'.json'))
    pr,pp=(read(parity/(p+'.json')) for p in ('registration','preflight'))
    for p,h in pr['config']['metadata_bindings'].items():verify(root/p,h)
    dreg,dpre=(read(delay/(p+'.json')) for p in ('registration','preflight'))
    assert dreg['source_revision']=='a75b55a9994aacbacd69ff130cb79d8293292033'
    fc=yaml.safe_load((root/'configs/oracle_frontier_20260905.yaml').read_text());cut=pd.Timestamp(fc['data_cutoff']);assert cut==pd.Timestamp('2023-04-16T13:45Z')
    sp=Path(fc['data_path']);side=read(sp.with_suffix('.sha256.json'));proof=parent_pre['spot_data_proof'];assert proof==pp['spot_data_proof']==dpre['spot_data_proof']
    for path,key in [(sp,'artifact_sha256'),(sp.with_suffix('.sha256.json'),'sidecar_sha256'),(side['availability_path'],'availability_sha256'),(side['source_ledger_path'],'ledger_sha256')]:verify(path,proof[key])
    um=parent_pre['um_data_proof'];assert um==pp['um_data_proof']==dpre['um_data_proof'];up=root/um['data_path']
    for path,key in [(up,'data_sha256'),(up.with_suffix('.sha256.json'),'sidecar_sha256'),(root/um['availability_path'],'availability_sha256'),(root/um['source_ledger_path'],'source_ledger_sha256'),(root/um['registration_path'],'registration_sha256')]:verify(path,um[key])
    print(json.dumps({'phase':'1536_artifact_sources_verified'}),flush=True)
    # Read OHLC only for finite/positive observation status. No price arithmetic or return outcomes.
    bars=pd.read_parquet(sp,columns=['open','high','low','close'],filters=[('bar_open_ts','<',cut)])
    assert bars.index.is_unique and bars.index.is_monotonic_increasing and bars.index.max()<cut
    full=pd.date_range(bars.index[0],bars.index[-1],freq='15min',tz='UTC');bars=bars.reindex(full)
    values=bars.to_numpy();assert not np.isinf(values).any() and (values[np.isfinite(values)]>0).all()
    available=bars.notna().all(axis=1).to_numpy();n=len(bars);label=np.zeros(n,bool)
    # Independent boolean window accounting: each previously defined h24 label needs rows tau+1..tau+24.
    missing=np.r_[0,np.cumsum(~available,dtype=np.int64)];ids=np.arange(n-24)
    label[ids]=((missing[ids+25]-missing[ids+1])==0)&np.isfinite(bars.open.to_numpy()[ids+1])
    assert not label[-24:].any()
    totals=Counter();restored_union={}
    for f in FOLDS:
        E=pd.Timestamp('2021-04-16T13:45Z')+pd.DateOffset(months=3*(f-5));end=E+pd.DateOffset(months=3);S=E-pd.DateOffset(months=6);I=E-pd.DateOffset(months=3)
        ci=pd.date_range(S,E,freq='15min',inclusive='left');vi=pd.date_range(E,end,freq='15min',inclusive='left');ix=ci.append(vi)
        assert end<=cut and ix.is_unique and ix.is_monotonic_increasing and np.all(np.diff(ix.asi8)==pd.Timedelta(minutes=15).value)
        phase=np.r_[np.zeros(len(ci),int),np.ones(len(vi),int)];assert len(phase)==len(ix)
        fold=read(delay/f'fold_{f}.json');assert fold['registration_sha256']==digest(dreg)
        scoremeta={s['model_id']:s for s in fold['scores']};streams={};cal_masks={};eval_masks={}
        fit_support=next(s for s in pp['support'] if s['reference_validation_fold']==f)
        assert pd.Timestamp(fit_support['fit_end'])==S and pd.Timestamp(fit_support['evaluation_start'])==E
        for g in GROUPS:
            cp=delay/'calibration'/f'fold{f}_{g}.npz';vp=delay/'forecasts'/f'fold{f}_{g}_raw.npz';pcp=parity/'calibration'/f'fold{f}_{g}.npz';npred=src/'forecasts'/f'fold{f}_{g}_reliability.npz'
            metadata=scoremeta[g+'_raw'];prov=metadata['provenance'];mp=prov['models']['return']
            assert prov['training_group']==prov['input_group']==g and prov['intervention']=='refit_with_available_inputs'
            assert prov['calibration']['path']==str(cp.relative_to(root)) and prov['calibration']['sha256']==bindings[str(cp.relative_to(root))] and metadata['forecast_sha256']==bindings[str(vp.relative_to(root))]
            assert mp['sha256']==bindings[mp['path']];verify(root/mp['path'],mp['sha256'])
            with np.load(cp,allow_pickle=False) as c,np.load(vp,allow_pickle=False) as v,np.load(pcp,allow_pickle=False) as pcals,np.load(npred,allow_pickle=False) as nc:
                assert set(c.files)=={'timestamps','mu','log_variance','variance','actual','scale_mask','interval_mask'}
                assert set(v.files)=={'timestamps','actual','mu','variance','raw_log_variance','persistence96_variance','fit_return_mean','score_support','inference_mask'}
                exact(c['timestamps'],ci.asi8,'calibration timestamps');exact(v['timestamps'],vi.asi8,'evaluation timestamps');exact(nc['timestamps'],vi.asi8,'Stage13 timestamps')
                exact(c['mu'],pcals['mu'],'parity raw calibration prediction identity')
                for key in ('scale_mask','interval_mask'):exact(c[key],pcals[key],key)
                for key in ('inference_mask','score_support'):exact(v[key],nc[key],key)
                raw=np.r_[c['mu'],v['mu']];assert raw.dtype.kind=='f' and not np.isinf(raw).any()
                finite=np.isfinite(raw);clock=np.asarray((ix.hour%6==0)&(ix.minute==0));assert not (finite&~clock).any()
                inf,score=v['inference_mask'].copy(),v['score_support'].copy();scale,interval=c['scale_mask'].copy(),c['interval_mask'].copy();assert all(m.dtype==bool for m in (inf,score,scale,interval)) and not (scale&interval).any() and not (score&~inf).any()
                assert np.array_equal(np.isfinite(v['mu']),inf)
                for name,mask,start,stop in [('scale',scale,S,I),('interval',interval,I,E)]:
                    assert (ci[mask]>=start).all() and (ci[mask]+pd.Timedelta(minutes=375)<stop).all()
                assert (vi[score]+pd.Timedelta(minutes=375)<=end).all()
                old=np.r_[scale|interval,score];assert not (old&~finite).any()
                # Old actual arrays supply only boolean persisted availability for mask consistency; numerical labels are never calculated.
                old_actual=np.r_[np.isfinite(c['actual']).all(axis=1),np.isfinite(v['actual']).all(axis=1)]
                exact(old,old_actual,'old persisted label support')
                streams[g]=finite;cal_masks[g]=(scale,interval);eval_masks[g]=(inf,score,old)
            stream_bindings.append({'fold':f,'group':g,'return_model':mp,'raw_calibration_path':str(cp.relative_to(root)),'raw_calibration_sha256':bindings[str(cp.relative_to(root))],'raw_evaluation_path':str(vp.relative_to(root)),'raw_evaluation_sha256':bindings[str(vp.relative_to(root))],'parity_calibration_path':str(pcp.relative_to(root)),'parity_calibration_sha256':bindings[str(pcp.relative_to(root))],'stage13_weight_path':str((src/'weights'/f'fold{f}_{g}.json').relative_to(root)),'stage13_weight_sha256':bindings[str((src/'weights'/f'fold{f}_{g}.json').relative_to(root))],'raw_availability_sha256':mask_sha(ix,finite)})
            counts['same_model_calibration_evaluation_streams']+=1
        exact(streams['technical'],streams['perp_delay0'],'shared raw history availability')
        for left,right in zip(eval_masks['technical'],eval_masks['perp_delay0']):exact(left,right,'shared current masks')
        inf,score,old=eval_masks['technical'];paired=streams['technical']&streams['perp_delay0']&label[full.get_indexer(ix)]
        assert not (old&~paired).any()
        history_positions=np.flatnonzero(paired);history_times=ix.asi8[history_positions];old_positions=np.flatnonzero(paired&old);old_times=ix.asi8[old_positions]
        frame=bars.loc[vi];clock=np.asarray((vi.hour%6==0)&(vi.minute==0));known=np.isfinite(frame.open.to_numpy());fallback=clock&known&~inf;missing_open=clock&~known
        numbers=[];restored_counts=[];first_tau=[];last_tau=[];restored_used=set();lookback_lower=[]
        for t in vi[inf]:
            lower=(t-pd.DateOffset(months=3)).value;upper=(t-pd.Timedelta(minutes=375)).value
            left=np.searchsorted(history_times,lower,side='left');right=np.searchsorted(history_times,upper,side='right');used=history_positions[left:right]
            number=len(used);assert number>=64
            assert (ix[used]>=t-pd.DateOffset(months=3)).all() and (ix[used]<t).all() and (ix[used]+pd.Timedelta(minutes=375)<=t).all()
            assert not np.any(ix[used]==t-pd.Timedelta(hours=6))
            prior_left=np.searchsorted(old_times,lower,side='left');prior_right=np.searchsorted(old_times,upper,side='right');restored=used[~old[used]]
            assert len(restored)==number-(prior_right-prior_left)
            numbers.append(number);restored_counts.append(len(restored));first_tau.append(int(ix.asi8[used[0]]));last_tau.append(int(ix.asi8[used[-1]]));lookback_lower.append(int(lower));restored_used.update(int(ix.asi8[v]) for v in restored)
        restored_union[str(f)]=[pd.Timestamp(t,tz='UTC').isoformat() for t in sorted(restored_used)]
        record={'fold':f,'same_raw_history_support':True,'raw_timeline_start':ix[0].isoformat(),'raw_timeline_end_exclusive':end.isoformat(),'raw_prediction_lookback_available_before_first_decision':bool(ix[0]<=vi[inf][0]-pd.DateOffset(months=3)),'current_inference_rows':int(inf.sum()),'current_score_rows':int(score.sum()),'current_fallback_rows':int(fallback.sum()),'current_missing_open_rows':int(missing_open.sum()),'minimum_history_rows':min(numbers),'maximum_history_rows':max(numbers),'first_history_rows':numbers[0],'last_history_rows':numbers[-1],'below64_history_count':sum(v<64 for v in numbers),'decisions_with_restored_boundary_history':sum(v>0 for v in restored_counts),'max_restored_per_history':max(restored_counts),'used_restored_history_timestamps':restored_union[str(f)],'history_eligibility_sha256':mask_sha(ix,paired),'raw_availability_sha256':mask_sha(ix,streams['technical']),'current_inference_sha256':mask_sha(vi,inf),'current_score_sha256':mask_sha(vi,score),'decision_timestamps_ns':vi.asi8[inf].tolist(),'history_row_counts':numbers,'history_first_tau_ns':first_tau,'history_last_tau_ns':last_tau,'restored_counts':restored_counts,'calendar_lower_bounds_ns':lookback_lower}
        records.append(record)
        for key,value in [('inference',inf.sum()),('score',score.sum()),('fallback',fallback.sum()),('missing_current_open',missing_open.sum()),('rolling_history_queries',len(numbers)),('restored_history_queries',sum(v>0 for v in restored_counts)),('distinct_fold_local_restored_timestamps',len(restored_used))]:totals[key]+=int(value)
        print(json.dumps({'phase':'paired_mature_history_counts_verified','fold':f,'minimum':min(numbers),'maximum':max(numbers),'restored_query_count':sum(v>0 for v in restored_counts)}),flush=True)
    assert totals=={'inference':2586,'score':2574,'fallback':332,'missing_current_open':2,'rolling_history_queries':2586,'restored_history_queries':2578,'distinct_fold_local_restored_timestamps':8}
    assert [r['minimum_history_rows'] for r in records]==[179,215,215,354,351,354,363,330]
    assert counts['same_model_calibration_evaluation_streams']==16
    report={'status':'pass','schema':'oracle-rolling-centering-source-audit-v1','script_sha256':sha(__file__),'scope':'Availability, chronology and immutable provenance only; no return values, means, slopes, losses or real rolling forecast/policy computations.','stage13_source_revision':SOURCE_REVISION,'stage13_config_sha256':CONFIG_SHA,'stage13_preflight_sha256':PREFLIGHT_SHA,'stage13_results_sha256':RESULT_SHA,'ancestor_artifacts':1328,'stage13_artifacts':208,'total_artifacts_verified':1536,'source_artifact_bindings':bindings,'source_artifact_inventory_sha256':digest(bindings),'source_manifest_bindings':manifests,'source_bindings':cfg['source_bindings'],'spot_data_proof':proof,'um_data_proof':um,'distinct_hashed_files':len(verified),'counts':dict(counts),'support_totals':dict(totals),'stream_bindings':stream_bindings,'folds':records,'rolling_contract':{'history_clock':'six-hour UTC','lookback':'tau >= t - DateOffset(months=3), tau < t','label_maturity':'tau + 375 minutes <= t','latest_possible_scheduled_tau':'t - 12 hours; t - 6 hours label is immature','label_availability':'canonical OHLC available for tau+1 through tau+24; positive observed entry open tau+1','shared_raw_support':'same-fold technical AND perp raw finite on inherited shared feature mask','minimum_history_rows':64,'current_support':'unchanged original inference and score masks; rolling anchor also limited to inference','boundary_restoration':'reconstructed matured labels; no reuse of fixed calibration/evaluation label masks','source_identity':'raw calibration and raw evaluation bind the identical fold/group return model; no cross-fold forecast stitching'},'new_mean_computed':False,'new_weight_computed':False,'new_loss_computed':False,'new_forecast_or_policy_computed':False,'additional_period_values_accessed':False,'limitations':['Retrospective raw observation availability does not prove exchange publication or historical receipt time.','No new returns or rolling values are calculated here; only numeric price positivity/finite observation status and saved forecast finiteness are inspected.','Restored history counts refer to labels actually eligible in at least one registered inference query, not every unscored future tail.','Same calendar timestamps reused under different fold models are not independent observations.','Entire source bytes are hashed; semantic OHLC reads are filtered strictly before 2023-04-16T13:45Z.']}
    path=Path('/tmp/oracle_rolling_centering_source_audit_20260906.json');path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'pass','report':str(path),'support':dict(totals),'distinct_hashed_files':len(verified)}),flush=True)

if __name__=='__main__':main()
