"""Stage15 independent data-only bindings, feature formula and support audit.
No model fitting, model coefficients, new return forecast, forecast loss or policy
calculation. The pinned old parity.prepare reconstructs existing labels/masks.
New feature equations below are independently written; no new runner is called.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,os,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=Path('/tmp/oracle_short_feature_source_audit_20260906.json')
SOURCE='845a9fd167533599c7189019b38dc9ca2edf0f41'
RESULT='d5adfc39e2822bfb77aaa519202b9949d664f772d704bcd63fa535147da9b620'
PREFLIGHT='8d9da4c4c04952e01b5194336605abe6a2480b72d8b4b68b0cf18c4a942347d1'
FOLDS=range(5,13)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def masksha(ix,m):return hashlib.sha256(ix.asi8.tobytes()+np.asarray(m,bool).tobytes()).hexdigest()
def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));checks=Counter();verified={}
    def verify(p,h):
        p=Path(p).resolve()
        if p not in verified:verified[p]=sha(p)
        assert verified[p]==h,('hash',str(p));checks['hash_binding_checks']+=1
    src=Path('codex_outputs/oracle_rolling_centering_decisions_v1')
    reg,pre,result=(read(src/(k+'.json')) for k in ('registration','preflight','results'))
    verify(src/'results.json',RESULT);verify(src/'preflight.json',PREFLIGHT)
    assert reg['source_revision']==SOURCE and result['registration_sha256']==digest(reg)
    cfg=reg['config'];cp=Path('configs/oracle_rolling_centering_decisions_20260906.yaml')
    verify(cp,reg['config_sha256']);assert yaml.safe_load(cp.read_text())==cfg
    assert reg['preflight_sha256']==PREFLIGHT==cfg['preflight_sha256']
    assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    assert pre['source_bindings']==cfg['source_bindings']==reg['source_bindings']
    for field in ('source_bindings','direct_source_bindings'):
        for p,h in pre[field].items():verify(p,h)
    bindings=dict(pre['source_artifact_bindings']);assert len(bindings)==1536
    own={};fold_bindings={}
    for f in FOLDS:
        p=src/f'fold_{f}.json';fold=read(p);fold_bindings[str(p)]=sha(p)
        assert fold['registration_sha256']==digest(reg)
        for key in ('rows','scores','fixed_weights'):assert fold[key]==[x for x in result[key] if x['fold']==f]
        assert len(fold['artifact_sha256'])==32
        for p,h in fold['artifact_sha256'].items():
            assert p not in bindings or bindings[p]==h
            bindings[p]=h;own[p]=h
    assert len(own)==256 and len(bindings)==1792
    assert len({str(Path(p).resolve()) for p in bindings})==len(bindings)
    for p,h in bindings.items():verify(p,h)
    checks.update(ancestor_artifacts=1792,stage14_artifacts=256)
    fallback=Path('codex_outputs/oracle_fallback_decisions_v1');fr=read(fallback/'results.json');fg=read(fallback/'registration.json')
    assert fr['registration_sha256']==digest(fg)
    rows={(r['fold'],r['candidate_id']):r for r in fr['rows']};assert len(rows)==len(fr['rows'])
    controls={}
    for f in FOLDS:
        for mean in ('zero','fit_mean','technical_raw'):
            for rule in ('utility_risk1','utility_risk1_fallback_bh'):
                cid=mean+'_'+rule;p=fallback/'targets'/f'fold{f}_{cid}.npz';r=rows[f,cid]
                assert r['targets_sha256']==bindings[str(p)]
                a=npz(p);assert set(a)=={'timestamps','targets'} and len(a['timestamps'])==len(a['targets'])
                controls[f,cid]=a;checks['old_additional_controls_verified']+=1
    # Only old, already frozen data/feature/mask preparation is executed.
    from unidream.experiments.oracle_frozen_procedure_parity import prepare,delay_um
    pc,dc,fc,bars,old_groups,_,y,masks,pp,_,_,_=prepare(Path('configs/oracle_frozen_procedure_parity_20260906.yaml'))
    assert len(bars) and bars.index[-1]<pd.Timestamp('2023-04-16T13:45:00Z')
    um=delay_um(pc,fc).reindex(bars.index)
    from unidream.experiments.oracle_short_features import make_short_feature_groups,PRICE_FEATURE_NAMES,FLOW_FEATURE_NAMES
    groups=make_short_feature_groups(bars,um)
    assert list(groups)==['technical','technical_short_price','technical_short_flow','technical_short_both']
    assert groups['technical'].equals(old_groups['technical'])
    for g,n in zip(groups,(29,34,32,37)):
        assert len(groups[g].columns)==n and groups[g].iloc[:,:29].equals(old_groups['technical'])
    def finite(series,positive=False):
        a=series.astype(float);return a.where(np.isfinite(a)&(a.gt(0) if positive else True))
    c,o,h,l=(finite(bars[k],True) for k in ('close','open','high','low'))
    independent=pd.DataFrame(index=bars.index)
    for k in (4,16,48):
        complete=c.notna().rolling(k+1,min_periods=k+1).sum().eq(k+1)
        independent[f'spot_log_return{k}']=np.log(c/c.shift(k)).where(complete)
    candle=(o.notna()&c.notna()&h.notna()&l.notna()&h.ge(l)&h.ge(o)&h.ge(c)&l.le(o)&l.le(c))
    independent['spot_body_sign1']=np.sign(c-o).where(candle)
    span=h-l
    independent['spot_close_location1']=(2*(c-l)/span.where(span.gt(0))-1).where(candle).mask(candle&span.eq(0),0.)
    for frame,prefix in ((bars,'spot'),(um,'perp')):
        q=finite(frame['quote_volume'],True);b=finite(frame['taker_buy_quote']);good=b.ge(0)&b.le(q)
        q=q.where(good);b=b.where(good)
        qs=q.rolling(4,min_periods=4).sum();bs=b.rolling(4,min_periods=4).sum()
        independent[prefix+'_weighted_flow4']=2*bs/qs-1
    q=finite(bars['quote_volume'],True)
    small=q.rolling(24,min_periods=24).mean();large=q.rolling(672,min_periods=669).mean().where(np.arange(len(q))>=671)
    independent['spot_quote_activity24_672']=np.log(small/large)
    independent=independent.loc[:,PRICE_FEATURE_NAMES+FLOW_FEATURE_NAMES].replace([np.inf,-np.inf],np.nan).shift(1)
    actual=groups['technical_short_both'].iloc[:,29:]
    assert np.array_equal(np.isfinite(independent),np.isfinite(actual))
    maxdiff={}
    for col in independent:
        a,b=independent[col].to_numpy(),actual[col].to_numpy();m=np.isfinite(a)
        err=float(np.max(np.abs(a[m]-b[m]))) if m.any() else 0.;maxdiff[col]=err
        assert np.allclose(a[m],b[m],rtol=1e-12,atol=1e-12),(col,err)
    records=[];regimes=Counter();availability_pass=True
    for f in FOLDS:
        m=masks[f];old=next(x for x in pp['support'] if x['reference_validation_fold']==f)
        dates={k:pd.Timestamp(old[k]) for k in ('fit_start','scale_start','interval_start','evaluation_start','evaluation_end')}
        # Hash geometry comes from the unchanged old full dependency intersection.
        for name in m:assert masksha(bars.index,m[name])==old['mask_sha256'][name]
        finite_counts={name:{g:int(np.isfinite(x.to_numpy()[mask]).all(axis=1).sum()) for g,x in groups.items()} for name,mask in m.items()}
        availability_pass &= all(n==int(m[name].sum()) for name,gs in finite_counts.items() if name!='scheduled' for n in gs.values())
        unavailable={name:{col:{'count':int((mask & ~np.isfinite(actual[col].to_numpy())).sum()),'timestamps':[t.isoformat() for t in bars.index[mask & ~np.isfinite(actual[col].to_numpy())]]} for col in actual if np.any(mask & ~np.isfinite(actual[col].to_numpy()))} for name,mask in m.items()}
        print(json.dumps({'fold':f,'unavailable_counts':{name:{col:v['count'] for col,v in cs.items()} for name,cs in unavailable.items()}}),flush=True)
        ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        reference=npz(Path('codex_outputs/oracle_derivative_delay_v1/forecasts')/f'fold{f}_technical_raw.npz')
        assert np.array_equal(reference['timestamps'],bars.index[ix].asi8)
        for name,key in (('inference','inference_mask'),('score','score_support')):assert np.array_equal(m[name][ix],reference[key])
        opening=bars.open.to_numpy()[ix];idx=bars.index[ix];clock=(idx.hour%6==0)&(idx.minute==0)&(idx.second==0)
        known=np.isfinite(opening)&(opening>0);fallback_mask=clock&known&~reference['inference_mask'];missing=clock&~known
        for (fold,cid),a in controls.items():
            if fold==f:assert np.array_equal(a['timestamps'],reference['timestamps'])
        model=Path('codex_outputs/oracle_derivative_delay_v1/models')/f'fold{f}_technical_return.joblib'
        cal=Path('codex_outputs/oracle_frozen_procedure_parity_v1/calibration')/f'fold{f}_technical.npz'
        assert str(model) in bindings and str(cal) in bindings
        z=npz(cal);cx=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        assert np.array_equal(z['timestamps'],bars.index[cx].asi8)
        for name,key in (('scale','scale_mask'),('interval','interval_mask')):assert np.array_equal(z[key],m[name][cx])
        assert np.array_equal(np.isfinite(z['mu']),m['predict'][cx])
        regime=next(r['regime']['trend'] for r in result['rows'] if r['fold']==f);regimes[regime]+=1
        records.append({'fold':f,'calendar':{k:v.isoformat() for k,v in dates.items()},'counts':{k:int(v.sum()) for k,v in m.items()},
            'mask_sha256':old['mask_sha256'],'finite_rows_on_each_original_mask':finite_counts,'unavailable_new_features':unavailable,
            'fallback_rows':int(fallback_mask.sum()),'missing_current_open_rows':int(missing.sum()),'regime':regime,
            'old_baseline_model':{'path':str(model),'sha256':bindings[str(model)]},'old_baseline_calibration':{'path':str(cal),'sha256':bindings[str(cal)]}})
    assert sum(r['counts']['inference'] for r in records)==2586 and sum(r['counts']['score'] for r in records)==2574
    assert sum(r['fallback_rows'] for r in records)==332 and sum(r['missing_current_open_rows'] for r in records)==2
    assert dict(regimes)=={'bull':2,'bear':4,'sideways':2}
    report={'passed':bool(availability_pass),'source_and_formula_checks_passed':True,'finite_requirement_masks':['fit','scale','interval','predict','inference','score'],'scheduled_mask_finiteness_is_not_required':True,'scope':'Data-only source chain, feature formulas and unchanged supports; no new model fit, return forecast, loss or policy',
        'source_revision':SOURCE,'results_sha256':RESULT,'preflight_sha256':PREFLIGHT,'script_sha256':sha(__file__),
        'source_artifact_inventory_sha256':digest(bindings),'source_artifact_bindings':bindings,'direct_stage14_fold_bindings':fold_bindings,
        'checks':dict(checks),'distinct_hashed_files':len(verified),'feature_max_abs_difference_independent_equations':maxdiff,
        'features':{g:list(x.columns) for g,x in groups.items()},'support':records,'regime_counts':dict(regimes),'regime_count_gate_pass':False,
        'new_feature_source_sha256':sha('unidream/experiments/oracle_short_features.py'),
        'limitations':['The old parity data-only loader reconstructs existing development labels/masks.','Historical event timestamps do not establish contemporary receipt.','The common availability mask remains retrospective.','Raw model numerical state/prediction parity is reserved for the frozen run.','No additional test data enter features, labels or scoring; inherited parquet decoding occurs before cutoff.']}
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'checks':dict(checks),'max_feature_difference':max(maxdiff.values()),'passed':bool(availability_pass)}))
if __name__=='__main__':main()
