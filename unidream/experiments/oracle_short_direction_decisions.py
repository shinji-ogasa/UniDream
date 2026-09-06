"""Fixed Technical37 C1 direction fits and one weighted-probability soft mean."""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from . import oracle_soft_direction_decisions as parent_module
from . import oracle_short_feature_decisions as feature_module
from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import FOLDS, RULES, compare_array
from .oracle_mean_controls import return_scores
from .oracle_direction_decisions import arrays, HINDSIGHT_IDS, SEGMENTS
from .oracle_direction_scores import direction_scores
from .oracle_short_mean_fit import _matrix_digest
from .oracle_soft_direction_decisions import exact_tree, average
from .oracle_soft_direction_mapping import map_soft_direction
from .oracle_short_direction_fit import fit_short_direction_family
from .oracle_short_direction_inputs import prepare_sources

PARENT_ROOT=Path('codex_outputs/oracle_soft_direction_decisions_v1')
FEATURE_ROOT=Path('codex_outputs/oracle_short_feature_decisions_v1')
DIRECTION_ROOT=Path('codex_outputs/oracle_direction_decisions_v1')
GROUP='technical_short_both'
WEIGHTINGS=('ordinary','magnitude')
MODEL_IDS=tuple(GROUP+'_'+w for w in WEIGHTINGS)
CONTROLS=parent_module.POLICIES
OLD_MEANS=parent_module.MEANS
OLD_CLASSIFIERS=parent_module.CLASSIFIERS
CLASSIFIERS=OLD_CLASSIFIERS+MODEL_IDS
NEW_MEAN=GROUP+'_magnitude_soft'
NEW_MEANS=(NEW_MEAN,)
MEANS=OLD_MEANS+NEW_MEANS
NEW_IDS=tuple(NEW_MEAN+'_'+r for r in RULES)
POLICIES=CONTROLS+NEW_IDS
REFERENCES=('technical_magnitude_soft','technical_magnitude_direction','technical_half',
    'technical_soft_mapped_prior','technical_soft_fit_mean','technical_soft_zero')
SOURCES=tuple(dict.fromkeys(parent_module.SOURCES+feature_module.SOURCES+tuple('unidream/experiments/'+n for n in (
    'oracle_short_direction_fit.py','oracle_short_direction_inputs.py','oracle_short_direction_decisions.py'))))
FIXED={'schema':'oracle-short-direction-decisions-v1','development_folds':list(FOLDS),
    'data_cutoff':'2023-04-16T13:45:00Z','parent_config':'configs/oracle_soft_direction_decisions_20260906.yaml',
    'feature_config':'configs/oracle_short_feature_decisions_20260906.yaml',
    'parent_root':str(PARENT_ROOT),'feature_root':str(FEATURE_ROOT),
    'parent_source_revision':'9b4f6a0e5606831a26a8f2a7c401e05c52d41f6f',
    'parent_publication_revision':'10262e7c95ba1375444d6db6142057ab4fc8f122',
    'output_dir':'codex_outputs/oracle_short_direction_decisions_v1','group':GROUP,'group_dimension':37,
    'weightings':list(WEIGHTINGS),'model_ids':list(MODEL_IDS),'classifiers':list(CLASSIFIERS),
    'control_ids':list(CONTROLS),'new_mean_ids':list(NEW_MEANS),'return_score_means':list(MEANS),
    'new_policy_ids':list(NEW_IDS),'references':list(REFERENCES),'rules':list(RULES),'segments':list(SEGMENTS),
    'logistic_settings':'exact_Stage17_C1','scaler':'unweighted_T_only_StandardScaler',
    'surrogate_mean':'saved_fit_abs_return_mean*(2.0*new_magnitude_probability-1.0)',
    'ordinary_probability_mapped_to_returns':False,'feature_source':'exact_Stage15_technical_short_both',
    'additional_support_removal_permitted':False,'new_model_fits':16,'new_unique_fit_priors':0,
    'shared_prior_verification_recomputations':16,'new_causal_names':2,'adaptive_prior_causal_names':218,
    'adaptive_total_causal_names':220,'score_classification_records':192,'score_return_records':400,
    'direction_diagnostic_records':32,'mapping_diagnostic_records':16,'fold_artifacts':95,
    'economic_rows':656,'economic_accounts':1312,'risk_source':'unchanged_saved_technical_scaled',
    'new_calibration_permitted':False,'utility_risk_aversion':1,'utility_cost_multiplier':2,
    'execution':{'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01},
    'inference_rows':2586,'score_rows':2574,'interval_inference_rows':2537,'interval_score_rows':2523,
    'fallback_rows':332,'missing_current_open_rows':2,
    'selection_permitted':False,'teacher_use_allowed':False,'additional_test_permitted':False,
    'numpy_version':'2.2.6','pandas_version':'2.3.3','sklearn_version':'1.8.0',
    'normalized_gradient_infinity_bound':1e-6,'scalar_logit_atol':1e-12,'scalar_probability_atol':1e-14,
    'partial_retry_policy':'full_replay_verify_existing_no_live_restart'}
EXTRA={'source_bindings','parent_config_sha256','feature_config_sha256','parent_manifest_bindings','feature_manifest_bindings','preflight_sha256'}


def manifest_paths(root):
    return {str(root/n) for n in ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in FOLDS]}


def validate_config(cfg):
    if (set(cfg)!=set(FIXED)|EXTRA or any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in FIXED.items())
            or set(cfg['source_bindings'])!=set(SOURCES)):
        raise ValueError('unregistered short direction family')
    for key,root in [('parent_manifest_bindings',PARENT_ROOT),('feature_manifest_bindings',FEATURE_ROOT)]:
        if set(cfg[key])!=manifest_paths(root):raise ValueError('incomplete '+key)


def prepare(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());validate_config(cfg)
    if (np.__version__,pd.__version__,sklearn.__version__)!=(cfg['numpy_version'],cfg['pandas_version'],cfg['sklearn_version']):
        raise ValueError('registered runtime changed')
    inputs=prepare_sources(cfg)
    return (cfg,inputs['bars'],inputs['features37'],inputs['packs'],inputs['masks'],inputs['scalars'],
        inputs['old_provenance'],inputs['evaluation'],inputs['calibration'],inputs['controls'],inputs['parent'],inputs['preflight'])


def summarize(rows,scores,classification_scores):
    ri={(r['fold'],r['candidate_id']):r for r in rows}
    si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
    ci={(r['fold'],r['segment'],r['classifier_id']):r for r in classification_scores}
    if (len(ri)!=len(rows) or set(ri)!={(f,c) for f in FOLDS for c in POLICIES}
            or len(si)!=len(scores) or set(si)!={(f,s,m) for f in FOLDS for s in SEGMENTS for m in MEANS}
            or len(ci)!=len(classification_scores) or set(ci)!={(f,s,m) for f in FOLDS for s in SEGMENTS for m in CLASSIFIERS}):
        raise ValueError('incomplete short direction family')
    old=parent_module.summarize([r for r in rows if r['candidate_id'] in CONTROLS],
        [r for r in scores if r['mean_id'] in OLD_MEANS],[r for r in classification_scores if r['classifier_id'] in OLD_CLASSIFIERS])
    regimes=('all','bull','bear','sideways');costs=('base','stress_2x')
    pk=('return_mse','return_mae','return_sign_accuracy','return_rank_ic','zero_return_mse','fit_mean_return_mse')
    ck=('log_loss','brier','binary_accuracy','signed_return_mean','weighted_log_loss','weighted_brier','weighted_binary_accuracy')
    for f in FOLDS:
        regime=ri[f,'bh']['regime']
        for c in NEW_IDS:
            if ri[f,c]['regime']!=regime or any(type(ri[f,c][co][k]) not in (int,float) or not math.isfinite(ri[f,c][co][k]) or (k in ('turnover','trades') and ri[f,c][co][k]<0)
                    for co in costs for k in ('alpha_ex','maxdd_delta','turnover','trades')):raise ValueError('invalid new economics')
        for seg in SEGMENTS:
            ref=si[f,seg,OLD_MEANS[0]];r=si[f,seg,NEW_MEAN];n=ref['rows']
            if type(r['rows']) is not int or r['rows']!=n or r['regime']!=regime:raise ValueError('unpaired return support')
            for k in pk:
                v=r[k]
                if v is None and k=='return_rank_ic':continue
                if type(v) not in (int,float) or not math.isfinite(v) or (k!='return_rank_ic' and v<0) or (k=='return_rank_ic' and abs(v)>1) or (k=='return_sign_accuracy' and v>1):
                    raise ValueError('invalid new return score')
            for k in ('zero_return_mse','fit_mean_return_mse'):
                if r[k]!=ref[k]:raise ValueError('new return baseline changed')
            cr=ci[f,seg,'technical_ordinary']
            for mid in MODEL_IDS:
                r=ci[f,seg,mid]
                if type(r['rows']) is not int or r['rows']!=n or r['regime']!=regime:raise ValueError('unpaired classifier support')
                for k in ('absolute_return_sum','absolute_return_mean','zero_actual_rows'):
                    if type(r[k]) is not type(cr[k]) or r[k]!=cr[k]:raise ValueError('new classifier label denominator changed')
                if type(r['zero_logit_rows']) is not int or not 0<=r['zero_logit_rows']<=n:raise ValueError('invalid zero logit count')
                for k in ck:
                    v=r[k]
                    if k.startswith('weighted_') and r['absolute_return_sum']==0:
                        if v is not None:raise ValueError('zero weight score must be null')
                        continue
                    if type(v) not in (int,float) or not math.isfinite(v) or (k!='signed_return_mean' and v<0) or (('accuracy' in k or 'brier' in k) and v>1):
                        raise ValueError('invalid new classifier score')
    out={'economics':copy.deepcopy(old['economics']),'prediction':copy.deepcopy(old['prediction']),
        'classification':copy.deepcopy(old['classification']),'classification_paired':{},'paired':{},
        'probability_gates':{},'short_direction':{},'inherited_Stage19_soft_flags':old['soft'],
        'inherited_Stage19_summary':copy.deepcopy(old),
        'regime_counts':old['regime_counts'],'interval_regime_strata_are_retrospective_evaluation_groupings':True,
        'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    for g in regimes:
        ff=[f for f in FOLDS if g=='all' or ri[f,'bh']['regime']['trend']==g]
        out['economics'][g].update({c:{'quarters':len(ff),'hindsight_only':False,
            'joint_positive_quarters_both_costs':sum(all(ri[f,c][co]['alpha_ex']>0 and ri[f,c][co]['maxdd_delta']<0 for co in costs) for f in ff),
            **{co:{k:average([ri[f,c][co][k] for f in ff]) for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs}} for c in NEW_IDS})
        out['classification_paired'][g]={}
        for seg in SEGMENTS:
            out['prediction'][g][seg][NEW_MEAN]={'quarters':len(ff),'rows':sum(si[f,seg,NEW_MEAN]['rows'] for f in ff),
                **{k:average([si[f,seg,NEW_MEAN][k] for f in ff]) for k in pk},
                'pooled_row_mse':math.fsum(si[f,seg,NEW_MEAN]['return_mse']*si[f,seg,NEW_MEAN]['rows']/sum(si[t,seg,NEW_MEAN]['rows'] for t in ff) for f in ff)}
            out['classification'][g][seg].update({mid:{'quarters':len(ff),'rows':sum(ci[f,seg,mid]['rows'] for f in ff),
                **{k:average([ci[f,seg,mid][k] for f in ff]) for k in ck},
                'zero_actual_rows':sum(ci[f,seg,mid]['zero_actual_rows'] for f in ff),
                'zero_logit_rows':sum(ci[f,seg,mid]['zero_logit_rows'] for f in ff),
                'absolute_return_sum':math.fsum(ci[f,seg,mid]['absolute_return_sum'] for f in ff)} for mid in MODEL_IDS})
            out['classification_paired'][g][seg]={GROUP+'_'+w:{ref:{k:average([
                None if ci[f,seg,GROUP+'_'+w][k] is None or ci[f,seg,ref][k] is None else ci[f,seg,GROUP+'_'+w][k]-ci[f,seg,ref][k]
                for f in ff]) for k in ck} for ref in ('technical_'+w,'prior_'+w)} for w in WEIGHTINGS}
        out['paired'][g]={ref:{'prediction':{seg:{
            'mse_difference':average([si[f,seg,NEW_MEAN]['return_mse']-si[f,seg,ref]['return_mse'] for f in ff]),
            'improved_quarters':sum(si[f,seg,NEW_MEAN]['return_mse']<si[f,seg,ref]['return_mse'] for f in ff),
            'equal_quarters':sum(si[f,seg,NEW_MEAN]['return_mse']==si[f,seg,ref]['return_mse'] for f in ff)} for seg in SEGMENTS},
            'economics':{rule:{co:{k:average([ri[f,NEW_MEAN+'_'+rule][co][k]-ri[f,ref+'_'+rule][co][k] for f in ff])
                for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs} for rule in RULES}} for ref in REFERENCES}
    for w in WEIGHTINGS:
        keys=('brier','log_loss') if w=='ordinary' else ('weighted_brier','weighted_log_loss')
        out['probability_gates'][GROUP+'_'+w]={seg:all(out['classification_paired'][g][seg][GROUP+'_'+w][ref][k] is not None
            and out['classification_paired'][g][seg][GROUP+'_'+w][ref][k]<0
            for g in regimes for ref in ('technical_'+w,'prior_'+w) for k in keys) for seg in SEGMENTS}
    out['both_classifier_families_improve_matched_losses_all_strata_both_segments']=all(
        flag for flags in out['probability_gates'].values() for flag in flags.values())
    for rule in RULES:
        out['short_direction'][NEW_MEAN+'_'+rule]={
            'economic_means_all_strata_both_costs':all(out['economics'][g][NEW_MEAN+'_'+rule][co]['alpha_ex']>0 and out['economics'][g][NEW_MEAN+'_'+rule][co]['maxdd_delta']<0 for g in regimes for co in costs),
            'economic_improvement_vs_all_six_references_all_strata_both_costs':all(out['paired'][g][ref]['economics'][rule][co]['alpha_ex']>0 and out['paired'][g][ref]['economics'][rule][co]['maxdd_delta']<0 for g in regimes for ref in REFERENCES for co in costs),
            'mapped_mse_vs_all_six_references_improved_all_strata':{seg:all(out['paired'][g][ref]['prediction'][seg]['mse_difference']<0 for g in regimes for ref in REFERENCES) for seg in SEGMENTS},
            'magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata':out['probability_gates'][GROUP+'_magnitude'].copy(),
            'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out


def run(config_path):
    cfg,bars,features,packs,masks,scalars,old_provenance,evaluation,calibration,controls,parent,pre=prepare(config_path)
    out=Path(cfg['output_dir'])
    if (out/'results.json').exists():raise ValueError('immutable short direction run completed')
    if file_digest(out/'preflight.json')!=cfg['preflight_sha256'] or json.loads((out/'preflight.json').read_text())!=pre:
        raise ValueError('registered short direction preflight changed')
    reg={'config':cfg,'config_sha256':file_digest(config_path),'preflight_sha256':cfg['preflight_sha256'],
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'16 new Technical37 C1 fits;0 new unique priors;1 soft mean;2 policies;80 old controls'}
    _immutable_json(out/'registration.json',reg)
    ex=cfg['execution'];stress={**ex,'one_way_cost':2*ex['one_way_cost'],'borrow_annual':2*ex['borrow_annual']}
    old_scores={(r['fold'],r['segment'],r['mean_id']):r for r in parent['scores']}
    old_cs={(r['fold'],r['segment'],r['classifier_id']):r for r in parent['classification_scores']}
    all_rows=[];all_scores=[];all_cs=[];all_fits=[];all_diagnostics=[];all_mapping=[]
    for f in FOLDS:
        dates=calendar(f-1);mm=masks[f];pack=packs[f];old_prov=old_provenance[f]
        cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        eval_ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        window=bars.loc[eval_ix];regime=controls[f,'bh']['regime'];ref=evaluation[f,'technical_half']
        y=np.full((len(bars),3),np.nan);y[mm['fit'],0]=pack['returns']
        fitted=fit_short_direction_family(features,y,fit_mask=mm['fit'],predict_mask=mm['predict']);del y
        compare_array(fitted['fit_labels'],pack['binary_labels'],name='unchanged labels',exact=True)
        for w in WEIGHTINGS:
            compare_array(fitted['fit_weights'][w],pack['weights_'+w],name='unchanged weights',exact=True)
            if fitted['fit_priors'][w]!=old_prov['fit_priors'][w]:raise ValueError('shared fit prior changed')
        for k in ('fit_return_mean','fit_abs_return_mean'):
            if fitted[k]!=old_prov[k]:raise ValueError('original T statistic changed')
        rows=[];scores=[];cs=[];diagnostics=[];mapping_diagnostics=[];bindings={};model_state={}
        def save(kind,name,value,extension='npz'):
            p=out/kind/f'fold{f}_{name}.{extension}';p.parent.mkdir(parents=True,exist_ok=True)
            if extension=='json':_immutable_json(p,value)
            elif extension=='joblib':
                buf=io.BytesIO();joblib.dump(value,buf,compress=3);data=buf.getvalue()
                if p.exists():
                    if p.read_bytes()!=data:raise ValueError('partial model differs')
                else:p.write_bytes(data)
            elif p.exists():
                old=arrays(p)
                if set(old)!=set(value):raise ValueError('partial array schema differs')
                for k in old:compare_array(old[k],value[k],name=str(p)+k,exact=True)
            else:np.savez_compressed(p,**value)
            bindings[str(p)]=file_digest(p);return p
        save('fit_data','training',{**pack,'fit_features_'+GROUP:features.to_numpy()[mm['fit']],
            'predict_features_'+GROUP:features.to_numpy()[mm['predict']]})
        for w,mid in zip(WEIGHTINGS,MODEL_IDS):
            model=fitted['models'][mid];old=joblib.load(DIRECTION_ROOT/'models'/f'fold{f}_technical_{w}.joblib')
            if model.named_steps['logisticregression'].get_params()!=old.named_steps['logisticregression'].get_params():raise ValueError('C1 logistic settings changed')
            scaler=model.named_steps['standardscaler'];old_scaler=old.named_steps['standardscaler']
            if scaler.get_params()!=old_scaler.get_params():raise ValueError('scaler settings changed')
            for k in ('mean_','var_','scale_'):
                compare_array(getattr(scaler,k)[:29],getattr(old_scaler,k),name='old scaler columns '+k,exact=True)
            compare_array(scaler.n_samples_seen_,old_scaler.n_samples_seen_,name='old scaler row count',exact=True)
            model_state[mid]={'all_estimator_parameters_unchanged':True,'old29_scaler_columns_exact':True,
                'old_model_path':str(DIRECTION_ROOT/'models'/f'fold{f}_technical_{w}.joblib'),
                'feature_columns':list(features.columns),'C':float(model.named_steps['logisticregression'].C)}
            save('models',mid,model,'joblib')
            for seg,select,container,maskkey in [('interval',cal_ix,calibration,'mapped_inference_mask'),('evaluation',eval_ix,evaluation,'inference_mask')]:
                original=container[f,'technical_'+w+'_direction'];mask=original[maskkey];z=fitted['logits'][mid][select];q=fitted['probabilities'][mid][select]
                scoremask=original['score_support'] if seg=='evaluation' else mm['interval'][select]
                save('probabilities_'+seg,mid,{'timestamps':bars.index[select].asi8,'logit':z,'probability':q,
                    'predict_mask':mm['predict'][select],'mapped_inference_mask':mask,'score_support':scoremask})
                n=int(mask.sum());oz=original['logit'][mask]
                diagnostics.append({'fold':f,'segment':seg,'classifier_id':mid,'regime':regime,'rows':n,
                    'uses_all_inference_not_score_support':True,'sign_disagreements_vs_Technical29':int((np.sign(z[mask])!=np.sign(oz)).sum()),
                    'zero_logit_rows':int((z[mask]==0).sum()),'probability_zero_rows':int((q[mask]==0).sum()),'probability_one_rows':int((q[mask]==1).sum()),
                    'mean_abs_logit':math.fsum(abs(float(v))/n for v in z[mask]),'old_mean_abs_logit':math.fsum(abs(float(v))/n for v in oz)})
        fit_record={'fold':f,'fit_provenance':fitted['provenance'],'model_state':model_state,
            'fit_priors':fitted['fit_priors'],'fit_return_mean':fitted['fit_return_mean'],'fit_abs_return_mean':fitted['fit_abs_return_mean'],
            'fit_source_binding':pre['support'][FOLDS.index(f)],'fit_labels_weights_priors_and_scalars_match_parent_exactly':True,
            'new_unique_prior_estimates':0,'new_model_fits':2,'risk_source':'unchanged_saved_technical_scaled'}
        save('provenance','fit',fit_record,'json');all_fits.append(fit_record)
        eval_means={m:evaluation[f,m]['mu'] for m in OLD_MEANS};cal_means={m:calibration[f,m]['mu'] for m in OLD_MEANS}
        for seg,select,container,means,maskkey,kind in [('interval',cal_ix,calibration,cal_means,'mapped_inference_mask','calibration'),('evaluation',eval_ix,evaluation,eval_means,'inference_mask','forecasts')]:
            original=container[f,'technical_magnitude_direction'];q=fitted['probabilities'][GROUP+'_magnitude'][select];z=fitted['logits'][GROUP+'_magnitude'][select]
            mapped=map_soft_direction(q,inference_mask=original[maskkey],fit_abs_return_mean=scalars[f]['fit_abs_return_mean'],
                saved_weighted_prior_probability=scalars[f]['prior_probability']['technical'],fit_return_mean=scalars[f]['fit_return_mean'])
            means[NEW_MEAN]=mapped['means']['soft'];save(kind,NEW_MEAN,{**original,'mu':means[NEW_MEAN],'logit':z,'probability':q})
            for const in ('mapped_prior','fit_mean','zero'):
                compare_array(mapped['means'][const],container[f,'technical_soft_'+const]['mu'],name='unchanged constant '+const,exact=True)
            mask=original[maskkey];n=int(mask.sum());mu=means[NEW_MEAN][mask]
            mapping_diagnostics.append({'fold':f,'segment':seg,'mean_id':NEW_MEAN,'regime':regime,'rows':n,
                'uses_all_inference_not_score_support':True,'mapping_diagnostic':mapped['diagnostic'],
                'mapped_direction_vs_logit_disagreements':int((np.sign(mu)!=np.sign(z[mask])).sum()),
                'mapped_zero_mean_rows':int((mu==0).sum()),'mean_abs_new_mu':math.fsum(abs(float(v))/n for v in mu),
                'mean_abs_old_soft_mu':math.fsum(abs(float(v))/n for v in container[f,'technical_magnitude_soft']['mu'][mask])})
        save('provenance','mapping',{'fold':f,'saved_T_scalars':scalars[f],'diagnostics':mapping_diagnostics,'ordinary_probability_not_mapped':True},'json')
        for seg in SEGMENTS:
            actual,support,means,select=(ref['actual'],ref['score_support'],eval_means,eval_ix) if seg=='evaluation' else (calibration[f,'technical_half']['actual'],mm['interval'][cal_ix],cal_means,cal_ix)
            meta={'fold':f,'segment':seg,'regime':regime,'regime_known_at_scored_decisions':seg=='evaluation','regime_reference':'evaluation_quarter_start'}
            for m in MEANS:
                score=return_scores(actual,means[m],support,float(ref['fit_return_mean']));record={**meta,'mean_id':m,**score}
                if m in OLD_MEANS:exact_tree(record,old_scores[f,seg,m],name='complete unchanged old return score')
                scores.append(record)
            for mid in CLASSIFIERS:
                if mid in MODEL_IDS:z=fitted['logits'][mid][select]
                else:
                    old_mean=mid+'_direction' if not mid.startswith('prior_') else 'technical_'+mid.removeprefix('prior_')+'_prior_direction'
                    z=(evaluation if seg=='evaluation' else calibration)[f,old_mean]['logit']
                score=direction_scores(actual,z,support);record={**meta,'classifier_id':mid,**score}
                if mid in OLD_CLASSIFIERS:exact_tree(record,old_cs[f,seg,mid],name='complete unchanged old classifier score')
                cs.append(record)
        for cid in POLICIES:
            trace=None
            if cid in CONTROLS:
                saved=arrays(PARENT_ROOT/'targets'/f'fold{f}_{cid}.npz');compare_array(saved['timestamps'],window.index.asi8,name='old target calendar',exact=True);target=saved['targets']
            else:
                original=evaluation[f,'technical_magnitude_direction'];am=action_masks(window.index,window.open.to_numpy(),original['inference_mask'])
                if cid.endswith(RULES[1]):
                    target,trace=fallback_targets(window,eval_means[NEW_MEAN],original['variance'],ex,inference_mask=original['inference_mask'],risk_aversion=1,cost_multiplier=2)
                    check_action_support(target,am);check_trace_support(target,am,trace)
                else:
                    target,trace=conditional_targets(window,eval_means[NEW_MEAN],original['variance'],ex,risk_aversion=1,cost_multiplier=2)
                    if np.any(np.isfinite(target)&~am['learned_eligible']):raise ValueError('hold order escaped inference support')
                trace['short_direction_mapping']={'mean_id':NEW_MEAN,'formula':cfg['surrogate_mean'],'saved_T_scalars':scalars[f],
                    'future_labels_used_for_orders':False,'probability_calibration':False,'risk_fits':0}
            p=save('targets',cid,{'timestamps':window.index.asi8,'targets':target})
            row={'fold':f,'candidate_id':cid,'regime':regime,'hindsight_only':cid in HINDSIGHT_IDS,'targets_sha256':bindings[str(p)],
                **{co:metrics(window,target,c) for co,c in [('base',ex),('stress_2x',stress)]}}
            if cid in CONTROLS:
                for co in ('base','stress_2x'):exact_tree(row[co],controls[f,cid][co],name='unchanged old account')
                if row['targets_sha256']!=controls[f,cid]['targets_sha256']:raise ValueError('old target bytes changed')
                row=controls[f,cid].copy()
            else:
                p=save('traces',cid,trace,'json');row['trace_sha256']=bindings[str(p)]
            rows.append(row)
        if (len(bindings),len(rows),len(scores),len(cs),len(diagnostics),len(mapping_diagnostics))!=(95,82,50,24,4,2):raise ValueError('incomplete short direction fold')
        _immutable_json(out/f'fold_{f}.json',{'registration_sha256':digest(reg),'rows':rows,'scores':scores,
            'classification_scores':cs,'direction_diagnostics':diagnostics,'mapping_diagnostics':mapping_diagnostics,'artifact_sha256':bindings})
        all_rows.extend(rows);all_scores.extend(scores);all_cs.extend(cs);all_diagnostics.extend(diagnostics);all_mapping.extend(mapping_diagnostics)
        print(json.dumps({'event':'fold_complete','fold':f,'new_model_fits':2,'new_unique_priors':0,'policies':82,'return_scores':50,'classification_scores':24,'artifacts':95}),flush=True)
    summary=summarize(all_rows,all_scores,all_cs)
    exact_tree(summary['inherited_Stage19_summary'],parent['summary'],name='complete inherited Stage19 summary')
    result={'registration_sha256':digest(reg),'rows':all_rows,'scores':all_scores,'classification_scores':all_cs,'fit_records':all_fits,
        'direction_diagnostics':all_diagnostics,'mapping_diagnostics':all_mapping,'summary':summary,
        'new_model_fits':16,'new_unique_priors':0,'new_causal_policy_names':2,'total_adaptively_explored_causal_names':220,
        'additional_test_used_for_modeling_or_scoring':False,'selection_performed':False,'teacher_use_allowed':False,
        'high_probability_generalization_established':False,'risk_model_or_calibration_fits':0}
    _immutable_json(out/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        cfg,*_,pre=prepare(args.config);path=Path(cfg['output_dir'])/'preflight.json';_immutable_json(path,pre)
        print(json.dumps({'path':str(path),'sha256':file_digest(path),'new_real_statistics_fits_logits_mappings_or_orders_computed':False}))
    else:run(args.config)
