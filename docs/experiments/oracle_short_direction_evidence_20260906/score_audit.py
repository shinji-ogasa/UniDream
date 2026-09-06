"""Independent Stage20 saved score/summary audit. No fit/predict/planner calls.
Uses only our sealed Stage19 audit's independent scalar/Decimal routines.
"""
import argparse, copy, hashlib, importlib.util, json, math
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
HELPER=Path('/tmp/oracle_soft_direction_score_audit_20260906.py')
HELPER_SHA='0f6e72b37c2e5bae906beeb7eba652152e1ae73cf8c7c0346a0a420425008907'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
spec=importlib.util.spec_from_file_location('independent_stage19',HELPER)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
ROOT=a.ROOT; PARENT=a.OUT; OUT=ROOT/'codex_outputs/oracle_short_direction_decisions_v1'
REPORT=Path('/tmp/oracle_short_direction_score_audit_20260906.json')
FS=a.FS; SEGMENTS=a.SEGMENTS; STRATA=a.STRATA; RULES=a.RULES; COSTS=a.COSTS
EK=a.EK;RK=a.RK;CK=a.CK[:7];OLD_MEANS=a.MEANS;OLD_CLASSIFIERS=a.CLASSIFIERS
MODELS=('technical_short_both_ordinary','technical_short_both_magnitude')
NEW_MEAN='technical_short_both_magnitude_soft';MEANS=OLD_MEANS+(NEW_MEAN,)
CLASSIFIERS=OLD_CLASSIFIERS+MODELS;NEW_IDS=tuple(NEW_MEAN+'_'+r for r in RULES)
REFERENCES=('technical_magnitude_soft','technical_magnitude_direction','technical_half',
 'technical_soft_mapped_prior','technical_soft_fit_mean','technical_soft_zero')
bind=a.bind;read=a.read;npz=a.npz;D=a.D;mean=a.mean;compare=a.compare

def stream(f,m,seg):
    kind='calibration' if seg=='interval' else 'forecasts'
    if m==NEW_MEAN:return npz(OUT/kind/f'fold{f}_{m}.npz')
    if m in a.NEW_MEANS:return npz(PARENT/kind/f'fold{f}_{m}.npz')
    if m in a.HALVES:
        if seg=='evaluation':return npz(a.PARITY/kind/f'fold{f}_{m}.npz')
        v=npz(a.DIRECTION/kind/f'fold{f}_{m.removesuffix("_half")}_ordinary_direction.npz')
        return {**v,'mu':v['parent_mu']}
    return npz((a.PARENT if m in a.REG_MEANS else a.DIRECTION)/kind/f'fold{f}_{m}.npz')

def summarize(rows,scores,cs,parent_summary,controls,hindsight):
    ri={(r['fold'],r['candidate_id']):r for r in rows}
    si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
    ci={(r['fold'],r['segment'],r['classifier_id']):r for r in cs}
    regimes={f:ri[f,'bh']['regime'] for f in FS}
    assert Counter(v['trend'] for v in regimes.values())=={'bull':2,'bear':4,'sideways':2}
    out={'economics':{},'prediction':{},'classification':{},'classification_paired':{},'paired':{},
         'probability_gates':{},'short_direction':{},'inherited_Stage19_soft_flags':parent_summary['soft'],
         'inherited_Stage19_summary':parent_summary,'regime_counts':{'bull':2,'bear':4,'sideways':2},
         'interval_regime_strata_are_retrospective_evaluation_groupings':True,
         'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    for g in STRATA:
        ff=[f for f in FS if g=='all' or regimes[f]['trend']==g]
        out['economics'][g]={c:{'quarters':len(ff),'hindsight_only':c in hindsight,
          'joint_positive_quarters_both_costs':sum(all(ri[f,c][co]['alpha_ex']>0 and ri[f,c][co]['maxdd_delta']<0 for co in COSTS) for f in ff),
          **{co:{k:mean([ri[f,c][co][k] for f in ff]) for k in EK} for co in COSTS}} for c in controls+NEW_IDS}
        out['prediction'][g]={};out['classification'][g]={};out['classification_paired'][g]={}
        for s in SEGMENTS:
            out['prediction'][g][s]={m:{'quarters':len(ff),'rows':sum(si[f,s,m]['rows'] for f in ff),
              **{k:mean([si[f,s,m][k] for f in ff]) for k in RK},
              'pooled_row_mse':sum(D(si[f,s,m]['return_mse'])*si[f,s,m]['rows'] for f in ff)/Decimal(sum(si[f,s,m]['rows'] for f in ff))} for m in MEANS}
            out['classification'][g][s]={m:{'quarters':len(ff),'rows':sum(ci[f,s,m]['rows'] for f in ff),
              **{k:mean([ci[f,s,m][k] for f in ff]) for k in CK},
              'zero_actual_rows':sum(ci[f,s,m]['zero_actual_rows'] for f in ff),
              'zero_logit_rows':sum(ci[f,s,m]['zero_logit_rows'] for f in ff),
              'absolute_return_sum':sum(D(ci[f,s,m]['absolute_return_sum']) for f in ff)} for m in CLASSIFIERS}
            out['classification_paired'][g][s]={m:{ref:{k:mean([
              None if ci[f,s,m][k] is None or ci[f,s,ref][k] is None else D(ci[f,s,m][k])-D(ci[f,s,ref][k]) for f in ff]) for k in CK}
              for ref in ('technical_'+w,'prior_'+w)} for w,m in zip(('ordinary','magnitude'),MODELS)}
        out['paired'][g]={ref:{'prediction':{s:{
          'mse_difference':mean([D(si[f,s,NEW_MEAN]['return_mse'])-D(si[f,s,ref]['return_mse']) for f in ff]),
          'improved_quarters':sum(si[f,s,NEW_MEAN]['return_mse']<si[f,s,ref]['return_mse'] for f in ff),
          'equal_quarters':sum(si[f,s,NEW_MEAN]['return_mse']==si[f,s,ref]['return_mse'] for f in ff)} for s in SEGMENTS},
          'economics':{rule:{co:{k:mean([D(ri[f,NEW_MEAN+'_'+rule][co][k])-D(ri[f,ref+'_'+rule][co][k]) for f in ff]) for k in EK} for co in COSTS} for rule in RULES}}
          for ref in REFERENCES}
    for w,m in zip(('ordinary','magnitude'),MODELS):
        keys=('brier','log_loss') if w=='ordinary' else ('weighted_brier','weighted_log_loss')
        out['probability_gates'][m]={s:all(out['classification_paired'][g][s][m][ref][k] is not None and out['classification_paired'][g][s][m][ref][k]<0
          for g in STRATA for ref in ('technical_'+w,'prior_'+w) for k in keys) for s in SEGMENTS}
    out['both_classifier_families_improve_matched_losses_all_strata_both_segments']=all(v for v0 in out['probability_gates'].values() for v in v0.values())
    for rule in RULES:
        cid=NEW_MEAN+'_'+rule
        out['short_direction'][cid]={
          'economic_means_all_strata_both_costs':all(out['economics'][g][cid][co]['alpha_ex']>0 and out['economics'][g][cid][co]['maxdd_delta']<0 for g in STRATA for co in COSTS),
          'economic_improvement_vs_all_six_references_all_strata_both_costs':all(out['paired'][g][ref]['economics'][rule][co]['alpha_ex']>0 and out['paired'][g][ref]['economics'][rule][co]['maxdd_delta']<0 for g in STRATA for ref in REFERENCES for co in COSTS),
          'mapped_mse_vs_all_six_references_improved_all_strata':{s:all(out['paired'][g][ref]['prediction'][s]['mse_difference']<0 for g in STRATA for ref in REFERENCES) for s in SEGMENTS},
          'magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata':out['probability_gates'][MODELS[1]],
          'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out

def audit(expected_sha):
    with localcontext() as ctx:
        ctx.prec=60
        result=read(bind(OUT/'results.json',expected_sha));reg=read(bind(OUT/'registration.json'));cfg=reg['config']
        bind(ROOT/'configs/oracle_short_direction_decisions_20260906.yaml',reg['config_sha256'],'config')
        pre=read(bind(OUT/'preflight.json',reg['preflight_sha256']))
        assert result['registration_sha256']==a.canonical(reg)
        assert cfg['preflight_sha256']==reg['preflight_sha256']
        assert cfg['source_bindings']==pre['source_bindings'] and len(cfg['source_bindings'])==42
        assert pre['config_contract_sha256']==a.canonical({k:v for k,v in cfg.items() if k!='preflight_sha256'})
        for k in ('source_bindings','direct_source_bindings','source_artifact_bindings'):
            for p,h in pre[k].items():bind(p,h,k)
        for k in ('parent_manifest_bindings','feature_manifest_bindings'):
            for p,h in cfg[k].items():bind(p,h,k)
        assert len(pre['source_artifact_bindings'])==5344
        bind(HELPER,HELPER_SHA,'independent_audit_helper')
        parent=read(PARENT/'results.json');preg=read(PARENT/'registration.json')
        assert parent['registration_sha256']==a.canonical(preg)==pre['parent_registration_canonical_sha256']
        assert preg['source_revision']==cfg['parent_source_revision']=='9b4f6a0e5606831a26a8f2a7c401e05c52d41f6f'
        feature_reg=read(ROOT/'codex_outputs/oracle_short_feature_decisions_v1/registration.json')
        assert a.canonical(feature_reg)==pre['feature_registration_canonical_sha256']
        rows,scores,cs=result['rows'],result['scores'],result['classification_scores']
        ri={(r['fold'],r['candidate_id']):r for r in rows};si={(r['fold'],r['segment'],r['mean_id']):r for r in scores};ci={(r['fold'],r['segment'],r['classifier_id']):r for r in cs}
        controls=tuple(cfg['control_ids']);ids=controls+NEW_IDS
        assert len(controls)==80 and len(ids)==82 and len(set(ids))==82
        assert tuple(cfg['return_score_means'])==MEANS and tuple(cfg['classifiers'])==CLASSIFIERS
        assert tuple(cfg['references'])==REFERENCES and tuple(cfg['new_policy_ids'])==NEW_IDS
        assert len(rows)==len(ri)==656 and set(ri)=={(f,c) for f in FS for c in ids}
        assert len(scores)==len(si)==400 and set(si)=={(f,s,m) for f in FS for s in SEGMENTS for m in MEANS}
        assert len(cs)==len(ci)==192 and set(ci)=={(f,s,m) for f in FS for s in SEGMENTS for m in CLASSIFIERS}
        fits={r['fold']:r for r in result['fit_records']};assert len(fits)==8 and set(fits)==set(FS)
        assert len(result['direction_diagnostics'])==32 and len(result['mapping_diagnostics'])==16
        assert {(r['fold'],r['segment'],r['classifier_id']) for r in result['direction_diagnostics']}=={(f,s,m) for f in FS for s in SEGMENTS for m in MODELS}
        assert {(r['fold'],r['segment'],r['mean_id']) for r in result['mapping_diagnostics']}=={(f,s,NEW_MEAN) for f in FS for s in SEGMENTS}
        for k,v in {'new_model_fits':16,'new_unique_priors':0,'new_causal_policy_names':2,'total_adaptively_explored_causal_names':220,'risk_model_or_calibration_fits':0}.items():assert result[k]==v
        for k in ('additional_test_used_for_modeling_or_scoring','selection_performed','teacher_use_allowed','high_probability_generalization_established'):assert result[k] is False
        artifacts={}
        for f in FS:
            fold=read(bind(OUT/f'fold_{f}.json',category='fold_manifest'));assert fold['registration_sha256']==result['registration_sha256']
            for k in ('rows','scores','classification_scores','direction_diagnostics','mapping_diagnostics'):assert fold[k]==[r for r in result[k] if r['fold']==f]
            expected={str((OUT/'targets'/f'fold{f}_{c}.npz').relative_to(ROOT)) for c in ids}
            expected|={str((OUT/'traces'/f'fold{f}_{c}.json').relative_to(ROOT)) for c in NEW_IDS}
            expected|={str((OUT/k/f'fold{f}_{NEW_MEAN}.npz').relative_to(ROOT)) for k in ('forecasts','calibration')}
            expected|={str((OUT/k/f'fold{f}_{m}.{ext}').relative_to(ROOT)) for k,ext in [('models','joblib'),('probabilities_interval','npz'),('probabilities_evaluation','npz')] for m in MODELS}
            expected|={str((OUT/'provenance'/f'fold{f}_{kind}.json').relative_to(ROOT)) for kind in ('fit','mapping')}
            expected|={str((OUT/'fit_data'/f'fold{f}_training.npz').relative_to(ROOT))}
            assert len(expected)==95 and set(fold['artifact_sha256'])==expected
            for p,h in fold['artifact_sha256'].items():bind(p,h,'new_artifact');artifacts[p]=h
        assert len(artifacts)==760
        old_ri={(r['fold'],r['candidate_id']):r for r in parent['rows']};old_si={(r['fold'],r['segment'],r['mean_id']):r for r in parent['scores']};old_ci={(r['fold'],r['segment'],r['classifier_id']):r for r in parent['classification_scores']}
        assert len(old_ri)==640 and len(old_si)==384 and len(old_ci)==160
        for key,r in old_ri.items():compare(ri[key],r,str(key),'exact_old_row','0')
        for key,r in old_si.items():compare(si[key],r,str(key),'exact_old_return','0')
        for key,r in old_ci.items():compare(ci[key],r,str(key),'exact_old_classification','0')
        assert result['summary']['inherited_Stage19_summary']==parent['summary']
        assert result['summary']['inherited_Stage19_soft_flags']==parent['summary']['soft']
        hindsight={r['candidate_id'] for r in parent['rows'] if r['hindsight_only']};assert len(hindsight)==24
        regimes={f:ri[f,'bh']['regime'] for f in FS};assert Counter(v['trend'] for v in regimes.values())=={'bull':2,'bear':4,'sideways':2}
        for r in rows:
            assert r['regime']==regimes[r['fold']] and r['hindsight_only']==(r['candidate_id'] in hindsight)
            for co in COSTS:assert all(type(v) in (float,int) and math.isfinite(v) for v in r[co].values())
        for r in scores+cs:
            assert r['regime']==regimes[r['fold']] and r['regime_reference']=='evaluation_quarter_start'
            assert r['regime_known_at_scored_decisions']==(r['segment']=='evaluation')
        # Reconstruct inherited summaries with our sealed independent Decimal routines.
        parent18=read(a.PARENT/'results.json');reg18=read(a.PARENT/'registration.json')
        old18=a.build_inherited_summary(parent18['rows'],parent18['scores'],parent18['classification_scores'],tuple(reg18['config']['control_ids']),hindsight)
        compare(old18,parent18['summary'],'stage18','independent_ancestor_summary')
        old19=a.build_summary(parent['rows'],parent['scores'],parent['classification_scores'],tuple(preg['config']['control_ids']),hindsight,old18)
        compare(old19,parent['summary'],'stage19','independent_parent_summary')
        summary=summarize(rows,scores,cs,old19,controls,hindsight)
        for k,v in summary.items():compare(v,result['summary'][k],k,'summary_'+k)
        assert set(summary)==set(result['summary'])
        return_checks=[];classification_checks=[];supports=[];constant_checks=[]
        for f in FS:
            fit=fits[f];assert fit==read(OUT/'provenance'/f'fold{f}_fit.json')
            assert fit['new_model_fits']==2 and fit['new_unique_prior_estimates']==0 and fit['fit_labels_weights_priors_and_scalars_match_parent_exactly'] is True
            oldfit=read(a.DIRECTION/'provenance'/f'fold{f}_fit.json')
            assert fit['fit_priors']==oldfit['fit_priors']
            for k in ('fit_return_mean','fit_abs_return_mean'):assert fit[k]==oldfit[k]
            fm=float(fit['fit_return_mean'])
            for s in SEGMENTS:
                for m in MEANS:
                    v=stream(f,m,s);sm=v['interval_mask'] if s=='interval' else v['score_support'];im=v['mapped_inference_mask'] if s=='interval' else v['inference_mask']
                    assert sm.dtype==im.dtype==bool and sm.shape==im.shape and np.all(sm<=im)
                    assert float(v['fit_return_mean'])==fm and type(si[f,s,m]['rows']) is int
                    measured=a.return_scores(v['actual'][sm,0],v['mu'][sm],fm)
                    compare(measured,{k:si[f,s,m][k] for k in measured},f'{f}/{s}/{m}','scalar_return_score')
                    for k in ('zero_return_mse','fit_mean_return_mse'):assert si[f,s,m][k]==si[f,s,'technical_half'][k]
                    if m in a.CONSTANT_MEANS:
                        assert si[f,s,m]['return_rank_ic'] is None and np.all(v['mu'][im]==v['mu'][im][0])
                        if m.endswith('_zero'):assert si[f,s,m]['return_mse']==si[f,s,m]['zero_return_mse']
                        if m.endswith('_fit_mean'):assert si[f,s,m]['return_mse']==si[f,s,m]['fit_mean_return_mse']
                        constant_checks.append({'fold':f,'segment':s,'mean_id':m,'null_rank':True})
                    if m==NEW_MEAN:
                        old=stream(f,'technical_magnitude_direction',s);assert set(v)==set(old)
                        for k in old:
                            if k not in ('mu','logit','probability'):a.exact_array(v[k],old[k],f'{f}/{s}/new_source/{k}')
                        assert np.array_equal(np.isfinite(v['mu']),im)
                        supports.append({'fold':f,'segment':s,'infer':int(im.sum()),'score':int(sm.sum()),'unscored_infer':int((im&~sm).sum())})
                    return_checks.append({'fold':f,'segment':s,'mean_id':m,'rows':measured['rows'],'rank_null':measured['return_rank_ic'] is None})
                for mid in CLASSIFIERS:
                    if mid in MODELS:
                        p=npz(OUT/('probabilities_'+s)/f'fold{f}_{mid}.npz');v=stream(f,'technical_magnitude_direction',s)
                        sm=v['interval_mask'] if s=='interval' else v['score_support'];im=v['mapped_inference_mask'] if s=='interval' else v['inference_mask']
                        assert np.array_equal(p['score_support'],sm) and np.array_equal(p['mapped_inference_mask'],im)
                        a.exact_array(p['timestamps'],v['timestamps'],f'{f}/{s}/{mid}/calendar')
                        assert p['predict_mask'].dtype==bool and np.all(im<=p['predict_mask'])
                        z=p['logit'];q=p['probability']
                        if mid==MODELS[1]:
                            nv=stream(f,NEW_MEAN,s)
                            for k in ('logit','probability'):a.exact_array(nv[k],p[k],f'{f}/{s}/{mid}/{k}')
                    else:
                        m=mid+'_direction' if not mid.startswith('prior_') else 'technical_'+mid.removeprefix('prior_')+'_prior_direction'
                        v=stream(f,m,s);sm=v['interval_mask'] if s=='interval' else v['score_support'];im=v['mapped_inference_mask'] if s=='interval' else v['inference_mask'];z=v['logit'];q=v['probability']
                    measured=a.classification_scores(v['actual'][sm,0],z[sm])
                    compare(measured,{k:ci[f,s,mid][k] for k in measured},f'{f}/{s}/{mid}','scalar_classification_score')
                    assert np.isfinite(q[im]).all() and np.all((q[im]>=0)&(q[im]<=1))
                    qp=np.asarray([a.scalar_sigmoid(zz) for zz in z[im]])
                    compare(float(np.max(np.abs(q[im]-qp))),0,f'{f}/{s}/{mid}/sigmoid','scalar_probability','1e-14')
                    classification_checks.append({'fold':f,'segment':s,'classifier_id':mid,'rows':measured['rows'],
                      'zero_actual':measured['zero_actual_rows'],'zero_logit':measured['zero_logit_rows'],'weighted_null':measured['weighted_log_loss'] is None})
            for c in controls:
                # Complete row equality above; bytes also bind copied intents.
                assert a.sha(OUT/'targets'/f'fold{f}_{c}.npz')==a.sha(PARENT/'targets'/f'fold{f}_{c}.npz')
        assert len(return_checks)==400 and len(classification_checks)==192 and len(constant_checks)==96 and len(supports)==16
        gate_components=[]
        for g in STRATA:
            for s in SEGMENTS:
                for w,m in zip(('ordinary','magnitude'),MODELS):
                    for ref in ('technical_'+w,'prior_'+w):
                        for k in (('brier','log_loss') if w=='ordinary' else ('weighted_brier','weighted_log_loss')):
                            val=summary['classification_paired'][g][s][m][ref][k]
                            gate_components.append({'type':'classification','stratum':g,'segment':s,'classifier':m,'reference':ref,'metric':k,'difference':val,'pass':val is not None and val<0})
                for ref in REFERENCES:
                    val=summary['paired'][g][ref]['prediction'][s]['mse_difference']
                    gate_components.append({'type':'return_mse','stratum':g,'segment':s,'reference':ref,'difference':val,'pass':val<0})
            for rule in RULES:
                cid=NEW_MEAN+'_'+rule
                for co in COSTS:
                    for k in ('alpha_ex','maxdd_delta'):
                        val=summary['economics'][g][cid][co][k]
                        gate_components.append({'type':'absolute_economics','stratum':g,'candidate':cid,'cost':co,'metric':k,'value':val,'pass':val>0 if k=='alpha_ex' else val<0})
                        for ref in REFERENCES:
                            dv=summary['paired'][g][ref]['economics'][rule][co][k]
                            gate_components.append({'type':'paired_economics','stratum':g,'candidate':cid,'cost':co,'metric':k,'reference':ref,'difference':dv,'pass':dv>0 if k=='alpha_ex' else dv<0})
        assert Counter(c['type'] for c in gate_components)=={'classification':64,'return_mse':48,'absolute_economics':32,'paired_economics':192}
        report={'schema':'independent-short-direction-score-summary-audit-v1','passed':True,
          'scope':'Read-only saved arrays and metadata. Independent scalar 400 return/192 classification scores and sigmoid check; Decimal60 complete 82-policy, all25mean/all12classifier, inherited Stage18/19 and six-reference summaries/gates. No fits, sklearn predict, training-objective replay, feature/raw-market reconstruction, canonical scoring/summary or policy/account rollouts. Separate audits own model-fit, mapper/diagnostics, and account paths.',
          'source_revision':reg['source_revision'],'source_sha256':a.BINDINGS,'verified_binding_counts':dict(a.CATEGORIES),
          'audit_script':{'path':str(Path(__file__)),'sha256':a.sha(Path(__file__))},
          'inventory':{'folds':8,'policies':82,'economic_rows':656,'accounts':1312,'return_means':25,'return_scores':400,'classifiers':12,'classification_scores':192,
            'new_model_artifacts':16,'new_artifacts':760,'ancestor_artifacts':5344,'source_files':42,
            'unchanged_old_economic_rows':640,'unchanged_old_return_scores':384,'unchanged_old_classification_scores':160,
            'constant_score_records':96,'fit_records':8,'direction_diagnostic_inventory':32,'mapping_diagnostic_inventory':16,
            'audit_fit_calls':0,'audit_model_predict_calls':0,'audit_policy_rollouts':0},
          'numeric_max_absolute_differences':a.MAXDIFF,'maximum_difference_locations':a.LOCATIONS,
          'summary':summary,'gate_components':gate_components,'new_support_checks':supports,
          'return_score_checks':return_checks,'classification_score_checks':classification_checks,'constant_control_checks':constant_checks,
          'tiny_nonzero_dd_values':[{'fold':r['fold'],'candidate_id':r['candidate_id'],'cost':co,'maxdd_delta':r[co]['maxdd_delta']} for r in rows for co in COSTS if 0<abs(r[co]['maxdd_delta'])<1e-12],
          'limitations':['Reused original development quarters; no untouched-forward confirmation, P-values, selection correction or high-probability claim.',
            'Regimes retain 2bull/4bear/2side counts. Interval scores are grouped retrospectively by evaluation-quarter-start regime.',
            'Magnitude-weighted probabilities target a tilted conditional distribution; frozen mean absolute return is not an established conditional magnitude.',
            'Four proper-score gates and two economic/MSE policy gates are distinct; overall economic signs do not establish all-regime or predictive improvement.',
            'All previously retained classifier scores and economics remain exactly preserved; new16 model fits are asserted artifact inventory, not independently refitted.',
            'Technical37 jointly changes eight columns within fixed C1; effect does not identify individual feature causality or exclude other architectures.']}
        assert not REPORT.exists(),'preserve existing audit'
        REPORT.write_text(json.dumps(a.plain(report),sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'passed':True,'path':str(REPORT),'sha256':a.sha(REPORT),'script_sha256':a.sha(Path(__file__)),
          'inventory':report['inventory'],'max_differences':a.plain(a.MAXDIFF),'probability_gates':summary['probability_gates'],
          'new_policy_gates':summary['short_direction']},sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-results-sha',required=True);args=parser.parse_args()
    assert len(args.expected_results_sha)==64
    audit(args.expected_results_sha)
