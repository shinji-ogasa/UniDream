"""Fixed causal direction losses with frozen parent magnitudes and trading rules."""
from __future__ import annotations

import argparse
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

from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import FOLDS, RULES, compare_array, compare_tree, prepare as prepare_parity
from .oracle_mean_controls import return_scores
from .oracle_mean_shrinkage import half_mean
from .oracle_sign_magnitude_decisions import (
    prepare as prepare_parent, SOURCES as PARENT_SOURCES, POLICIES as CONTROLS,
    OLD_ORACLES, NEW_IDS as PARENT_NEW_ORACLES, arrays,
)
from .oracle_direction_fit import fit_direction_family
from .oracle_direction_scores import direction_scores

PARENT_ROOT=Path('codex_outputs/oracle_sign_magnitude_decisions_v1')
PARITY_ROOT=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
GROUPS=('technical','perp_delay0')
WEIGHTINGS=('ordinary','magnitude')
PARENTS={g:g+'_half' for g in GROUPS}
MODEL_IDS=tuple(g+'_'+w for g in GROUPS for w in WEIGHTINGS)
PRIOR_IDS=tuple('prior_'+w for w in WEIGHTINGS)
CLASSIFIERS=MODEL_IDS+PRIOR_IDS
LEARNED_MEANS=tuple(m+'_direction' for m in MODEL_IDS)
PRIOR_MEANS=tuple(g+'_'+w+'_prior_direction' for g in GROUPS for w in WEIGHTINGS)
NEW_MEANS=LEARNED_MEANS+PRIOR_MEANS
MEANS=tuple(PARENTS.values())+NEW_MEANS
NEW_IDS=tuple(m+'_'+rule for m in NEW_MEANS for rule in RULES)
POLICIES=CONTROLS+NEW_IDS
HINDSIGHT_IDS=OLD_ORACLES+PARENT_NEW_ORACLES
SEGMENTS=('interval','evaluation')
MAPPING={g+'_'+w+suffix:{'group':g,'weighting':w,'classifier_id':('prior_'+w if suffix=='_prior_direction' else g+'_'+w),
    'parent_mean':PARENTS[g],'prior_mean':g+'_'+w+'_prior_direction'}
    for suffix in ('_direction','_prior_direction') for g in GROUPS for w in WEIGHTINGS}
SOURCES=PARENT_SOURCES+tuple('unidream/experiments/'+n for n in
    ('oracle_short_mean_fit.py','oracle_direction_fit.py','oracle_direction_scores.py','oracle_direction_decisions.py'))
FIXED={'schema':'oracle-direction-decisions-v1','development_folds':list(FOLDS),
    'data_cutoff':'2023-04-16T13:45:00Z','parent_config':'configs/oracle_sign_magnitude_decisions_20260906.yaml',
    'parity_config':'configs/oracle_frozen_procedure_parity_20260906.yaml','parent_root':str(PARENT_ROOT),
    'parent_source_revision':'b44c211dccc38f719b6f893a95c0d1a2d4cbf638',
    'output_dir':'codex_outputs/oracle_direction_decisions_v1','groups':list(GROUPS),'group_dimensions':[29,31],
    'weightings':list(WEIGHTINGS),'model_ids':list(MODEL_IDS),'classifiers':list(CLASSIFIERS),
    'new_mean_ids':list(NEW_MEANS),'return_score_means':list(MEANS),'control_ids':list(CONTROLS),
    'new_policy_ids':list(NEW_IDS),'mapping':MAPPING,'rules':list(RULES),'segments':list(SEGMENTS),
    'model':'unweighted_StandardScaler_then_weighted_L2_logistic','C':1.0,'solver':'lbfgs',
    'tol':1e-8,'max_iter':1000,'l1_ratio':0.0,'random_state':20260906,'threadpool_limit':2,
    'weight_normalization':'abs_fit_return_divided_by_fit_mean_abs_return',
    'new_model_fits':32,'fit_prior_probabilities':16,'new_causal_names':16,'adaptive_prior_causal_names':174,
    'adaptive_total_causal_names':190,'score_classification_records':96,'score_return_records':160,
    'surrogate_mean':'sign(logit)*abs(own_frozen_half_mu)','zero_logit_direction':0,
    'probability_calibration_permitted':False,'mean_risk_or_weight_calibration_permitted':False,
    'risk_source':'unchanged_saved_technical_scaled','utility_risk_aversion':1,'utility_cost_multiplier':2,
    'inference_rows':2586,'score_rows':2574,'fallback_rows':332,'missing_current_open_rows':2,
    'selection_permitted':False,'teacher_use_allowed':False,'additional_test_permitted':False,
    'numpy_version':'2.2.6','pandas_version':'2.3.3','sklearn_version':'1.8.0',
    'normalized_gradient_infinity_bound':1e-6,'scalar_logit_atol':1e-12,'scalar_probability_atol':1e-14,
    'fold_artifacts':90,'economic_rows':416,'economic_accounts':832,
    'prior_logit':'math.log(pi)-math.log1p(-pi)','partial_retry_policy':'full_replay_verify_existing_no_live_restart'}
EXTRA={'source_bindings','parent_config_sha256','parity_config_sha256','parent_manifest_bindings','preflight_sha256'}


def validate_config(cfg):
    if (set(cfg)!=set(FIXED)|EXTRA or any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in FIXED.items())
            or set(cfg['source_bindings'])!=set(SOURCES)):raise ValueError('unregistered direction family')
    expected={str(PARENT_ROOT/n) for n in ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in FOLDS]}
    if set(cfg['parent_manifest_bindings'])!=expected:raise ValueError('incomplete parent manifest inventory')


def map_direction(logits,parent_mu,mask):
    z,mu,m=np.asarray(logits),np.asarray(parent_mu),np.asarray(mask)
    if (m.ndim!=1 or m.dtype!=bool or not m.any() or z.shape!=m.shape or mu.shape!=m.shape
            or z.dtype.kind not in 'fiu' or mu.dtype.kind not in 'fiu'
            or not np.isfinite(z[m]).all() or not np.isfinite(mu[m]).all()):
        raise ValueError('finite aligned causal direction inputs required')
    result=np.full(len(m),np.nan);result[m]=np.sign(z[m])*np.abs(mu[m]);return result


def prepare(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());validate_config(cfg)
    if (np.__version__,pd.__version__,sklearn.__version__)!=(cfg['numpy_version'],cfg['pandas_version'],cfg['sklearn_version']):
        raise ValueError('registered runtime changed')
    direct={**cfg['source_bindings'],**cfg['parent_manifest_bindings'],cfg['parent_config']:cfg['parent_config_sha256'],
            cfg['parity_config']:cfg['parity_config_sha256']}
    for p,sha in direct.items():
        if file_digest(Path(p))!=sha:raise ValueError('source changed: '+p)
    pc,fc,bars,forecasts,_,parent_fit_returns,pp=prepare_parent(Path(cfg['parent_config']))
    parent=json.loads((PARENT_ROOT/'results.json').read_text());reg=json.loads((PARENT_ROOT/'registration.json').read_text())
    if (reg['config']!=pc or reg['config_sha256']!=cfg['parent_config_sha256']
            or reg['source_revision']!=cfg['parent_source_revision']
            or reg['preflight_sha256']!=file_digest(PARENT_ROOT/'preflight.json')
            or pp!=json.loads((PARENT_ROOT/'preflight.json').read_text()) or parent['registration_sha256']!=digest(reg)):
        raise ValueError('completed Stage16 chain changed')
    controls={(r['fold'],r['candidate_id']):r for r in parent['rows']}
    if len(controls)!=288 or set(controls)!={(f,c) for f in FOLDS for c in CONTROLS}:raise ValueError('incomplete old family')
    par=prepare_parity(Path(cfg['parity_config']));_,_,pfc,pbars,all_groups,_,y,masks,_,_,_,_=par
    if pfc!=fc or not pbars.equals(bars) or fc['data_cutoff']!=cfg['data_cutoff']:raise ValueError('original data changed')
    groups={g:all_groups[g] for g in GROUPS};bindings=dict(pp['source_artifact_bindings']);calibration={};support=[]
    if [len(groups[g].columns) for g in GROUPS]!=[29,31]:raise ValueError('original feature schemas changed')
    for f in FOLDS:
        fold=json.loads((PARENT_ROOT/f'fold_{f}.json').read_text())
        if fold['registration_sha256']!=digest(reg) or len(fold['artifact_sha256'])!=49:raise ValueError('parent fold changed')
        for k in ('rows','scores','direction_diagnostics','endpoint_parity'):
            if fold[k]!=[r for r in parent[k] if r['fold']==f]:raise ValueError('parent fold records differ')
        if fold['threshold']!=next(q for q in parent['thresholds'] if q['fold']==f):raise ValueError('parent threshold changed')
        for p,sha in fold['artifact_sha256'].items():
            if file_digest(Path(p))!=sha or (p in bindings and bindings[p]!=sha):raise ValueError('ancestor changed')
            bindings[p]=sha
        m=masks[f];dates=calendar(f-1);cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        for name in ('fit','scale','interval','predict','inference','score'):
            if any(not np.isfinite(groups[g].to_numpy()[m[name]]).all() for g in GROUPS):raise ValueError('feature support changed')
        compare_array(y[m['fit'],0],parent_fit_returns[f],name='parent fit returns',exact=True)
        provenance_path=PARITY_ROOT/'calibration'/f'fold{f}_provenance.json'
        if str(provenance_path) not in bindings:raise ValueError('unbound original calibration')
        prov=json.loads(provenance_path.read_text())['calibration'];cal_times=bars.index[cal_ix]
        for g in GROUPS:
            p=PARITY_ROOT/'calibration'/f'fold{f}_{g}.npz'
            if str(p) not in bindings:raise ValueError('unbound original S/I forecast')
            ca=arrays(p);compare_array(ca['timestamps'],cal_times.asi8,name='calibration times',exact=True)
            for seg in ('scale','interval'):compare_array(ca[seg+'_mask'],m[seg][cal_ix],name='calibration mask',exact=True)
            expected=y[cal_ix].copy();expected[~(m['scale']|m['interval'])[cal_ix]]=np.nan
            compare_array(ca['actual'],expected,name='calibration outcomes',exact=True)
            mapped_mask=m['predict'][cal_ix]&np.asarray(cal_times>=dates['interval_start'])
            # S-fitted bias/anchor are used only from I onward. Raw classifier S predictions are saved but never scored.
            ca['parent_mu']=half_mean(ca['mu']+float(prov['return_bias'][g]),np.full(len(cal_times),float(prov['scale_mean'])),inference_mask=mapped_mask)
            ca['classifier_predict_mask']=m['predict'][cal_ix];ca['mapped_inference_mask']=mapped_mask
            calibration[f,g]=ca
        support.append({'fold':f,'regime':controls[f,'bh']['regime'],
            'counts':{n:int(m[n].sum()) for n in ('fit','scale','interval','predict','inference','score')},
            'mask_sha256':{n:mask_digest(bars.index,m[n]) for n in ('fit','scale','interval','predict','inference','score')},
            'fit_return_sha256':digest(y[m['fit'],0].tolist()),
            'feature_columns':{g:list(groups[g].columns) for g in GROUPS},
            'fit_features_sha256':{g:digest(groups[g].to_numpy()[m['fit']].tolist()) for g in GROUPS},
            'predict_features_sha256':{g:digest(groups[g].to_numpy()[m['predict']].tolist()) for g in GROUPS},
            'new_feature_rows_removed':0})
    if len(bindings)!=2120 or len({str(Path(p).resolve()) for p in bindings})!=2120:raise ValueError('ancestor inventory changed')
    pre={'schema':'oracle-direction-preflight-v1','config_contract_sha256':digest({k:v for k,v in cfg.items() if k!='preflight_sha256'}),
        'source_bindings':cfg['source_bindings'],'direct_source_bindings':direct,'source_artifact_bindings':bindings,
        'support':support,'spot_data_proof':pp['spot_data_proof'],'um_data_proof':pp['um_data_proof'],
        'new_class_labels_priors_weights_fits_logits_mapped_predictions_or_orders_computed':False,
        'original_I_half_magnitudes_reconstructed_from_frozen_S_calibration':True}
    return cfg,fc,bars,groups,y,masks,forecasts,controls,calibration,pre


def average(values):
    return None if any(v is None for v in values) else math.fsum(v/len(values) for v in values)


def summarize(rows,scores,classification_scores):
    ri={(r['fold'],r['candidate_id']):r for r in rows}
    si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
    ci={(r['fold'],r['segment'],r['classifier_id']):r for r in classification_scores}
    if (len(ri)!=len(rows) or set(ri)!={(f,c) for f in FOLDS for c in POLICIES}
            or len(si)!=len(scores) or set(si)!={(f,s,m) for f in FOLDS for s in SEGMENTS for m in MEANS}
            or len(ci)!=len(classification_scores) or set(ci)!={(f,s,m) for f in FOLDS for s in SEGMENTS for m in CLASSIFIERS}):
        raise ValueError('incomplete direction family')
    regimes=('all','bull','bear','sideways');costs=('base','stress_2x')
    counts={g:sum(ri[f,'bh']['regime']['trend']==g for f in FOLDS) for g in regimes[1:]}
    if counts!={'bull':2,'bear':4,'sideways':2}:raise ValueError('regime inventory changed')
    ck=('log_loss','brier','binary_accuracy','signed_return_mean','weighted_log_loss','weighted_brier','weighted_binary_accuracy')
    pk=('return_mse','return_mae','return_sign_accuracy','return_rank_ic','zero_return_mse','fit_mean_return_mse')
    for f in FOLDS:
        regime=ri[f,'bh']['regime']
        for cid in POLICIES:
            r=ri[f,cid]
            if r['regime']!=regime or any(type(r[c][k]) not in (float,int) or not math.isfinite(r[c][k])
                    for c in costs for k in ('alpha_ex','maxdd_delta','turnover','trades')):raise ValueError('invalid economics')
        for seg in SEGMENTS:
            n=si[f,seg,MEANS[0]]['rows']
            if type(n) is not int or n<16:raise ValueError('invalid score count')
            for m in MEANS:
                r=si[f,seg,m]
                if type(r['rows']) is not int or r['rows']!=n or r['regime']!=regime:raise ValueError('unpaired return support')
                for k in pk:
                    v=r[k]
                    if v is None and k=='return_rank_ic':continue
                    if (type(v) not in (int,float) or not math.isfinite(v) or
                        (k!='return_rank_ic' and v<0) or (k=='return_sign_accuracy' and v>1) or
                        (k=='return_rank_ic' and abs(v)>1)):raise ValueError('invalid return score')
            denominators=[]
            for m in CLASSIFIERS:
                r=ci[f,seg,m]
                if type(r['rows']) is not int or r['rows']!=n or r['regime']!=regime:raise ValueError('unpaired classifier support')
                for k in ('zero_actual_rows','zero_logit_rows'):
                    if type(r[k]) is not int or not 0<=r[k]<=n:raise ValueError('invalid zero count')
                for k in ('absolute_return_sum','absolute_return_mean'):
                    if type(r[k]) not in (int,float) or not math.isfinite(r[k]) or r[k]<0:raise ValueError('invalid absolute-return denominator')
                if not math.isclose(r['absolute_return_sum']/n,r['absolute_return_mean'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('inconsistent weight denominator')
                denominators.append((r['absolute_return_sum'],r['absolute_return_mean'],r['zero_actual_rows']))
                for k in ck:
                    v=r[k]
                    if k.startswith('weighted_') and r['absolute_return_sum']==0:
                        if v is not None:raise ValueError('zero-weight score must be null')
                        continue
                    if (type(v) not in (int,float) or not math.isfinite(v) or (k!='signed_return_mean' and v<0)
                            or (('accuracy' in k or 'brier' in k) and v>1)):raise ValueError('invalid classification score')
            if len(set(denominators))!=1:raise ValueError('classification label supports changed')
    out={'economics':{},'prediction':{},'classification':{},'paired':{},'classification_paired':{},'direction':{},
        'regime_counts':counts,'interval_regime_strata_are_retrospective_evaluation_groupings':True,
        'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    for g in regimes:
        ff=[f for f in FOLDS if g=='all' or ri[f,'bh']['regime']['trend']==g]
        out['economics'][g]={c:{'quarters':len(ff),'hindsight_only':c in HINDSIGHT_IDS,
            'joint_positive_quarters_both_costs':sum(all(ri[f,c][co]['alpha_ex']>0 and ri[f,c][co]['maxdd_delta']<0 for co in costs) for f in ff),
            **{co:{k:average([ri[f,c][co][k] for f in ff]) for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs}} for c in POLICIES}
        out['prediction'][g]={};out['classification'][g]={};out['classification_paired'][g]={}
        for seg in SEGMENTS:
            out['prediction'][g][seg]={m:{'quarters':len(ff),'rows':sum(si[f,seg,m]['rows'] for f in ff),
                **{k:average([si[f,seg,m][k] for f in ff]) for k in pk},
                'pooled_row_mse':math.fsum(si[f,seg,m]['return_mse']*si[f,seg,m]['rows']/sum(si[t,seg,m]['rows'] for t in ff) for f in ff)} for m in MEANS}
            out['classification'][g][seg]={m:{'quarters':len(ff),'rows':sum(ci[f,seg,m]['rows'] for f in ff),
                **{k:average([ci[f,seg,m][k] for f in ff]) for k in ck},
                'zero_actual_rows':sum(ci[f,seg,m]['zero_actual_rows'] for f in ff),
                'zero_logit_rows':sum(ci[f,seg,m]['zero_logit_rows'] for f in ff),
                'absolute_return_sum':math.fsum(ci[f,seg,m]['absolute_return_sum'] for f in ff)} for m in CLASSIFIERS}
            out['classification_paired'][g][seg]={}
            for mid in MODEL_IDS:
                group,weighting=next((gg,w) for gg in GROUPS for w in WEIGHTINGS if gg+'_'+w==mid)
                refs=PRIOR_IDS+(group+'_'+('magnitude' if weighting=='ordinary' else 'ordinary'),)
                out['classification_paired'][g][seg][mid]={ref:{k:average([None if ci[f,seg,mid][k] is None or ci[f,seg,ref][k] is None else
                    ci[f,seg,mid][k]-ci[f,seg,ref][k] for f in ff]) for k in ck} for ref in refs}
        paired={}
        for m in NEW_MEANS:
            refs=tuple(dict.fromkeys((MAPPING[m]['parent_mean'],MAPPING[m]['prior_mean'])))
            paired[m]={ref:{'prediction':{seg:{'mse_difference':average([si[f,seg,m]['return_mse']-si[f,seg,ref]['return_mse'] for f in ff]),
                'improved_quarters':sum(si[f,seg,m]['return_mse']<si[f,seg,ref]['return_mse'] for f in ff),
                'equal_quarters':sum(si[f,seg,m]['return_mse']==si[f,seg,ref]['return_mse'] for f in ff)} for seg in SEGMENTS},
                'economics':{rule:{co:{k:average([ri[f,m+'_'+rule][co][k]-ri[f,ref+'_'+rule][co][k] for f in ff])
                    for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs} for rule in RULES}} for ref in refs}
        out['paired'][g]=paired
    for m in NEW_MEANS:
        mapping=MAPPING[m];mid=mapping['classifier_id'];weighting=mapping['weighting']
        score_keys=('brier','log_loss') if weighting=='ordinary' else ('weighted_brier','weighted_log_loss')
        classifier_gate={seg:mid in MODEL_IDS and all(out['classification_paired'][g][seg][mid]['prior_'+weighting][k] is not None
            and out['classification_paired'][g][seg][mid]['prior_'+weighting][k]<0 for g in regimes for k in score_keys) for seg in SEGMENTS}
        mapped_gate={seg:all(out['prediction'][g][seg][m]['return_mse']<out['prediction'][g][seg][m][control]
            and all(out['paired'][g][m][ref]['prediction'][seg]['mse_difference']<0 for ref in
                tuple(dict.fromkeys((mapping['parent_mean'],mapping['prior_mean']))))
            for g in regimes for control in ('zero_return_mse','fit_mean_return_mse')) for seg in SEGMENTS}
        for rule in RULES:
            cid=m+'_'+rule
            out['direction'][cid]={'economic_means_all_strata_both_costs':all(out['economics'][g][cid][co]['alpha_ex']>0
                and out['economics'][g][cid][co]['maxdd_delta']<0 for g in regimes for co in costs),
                'matched_probability_losses_improved_all_strata':classifier_gate,
                'mapped_mse_vs_zero_fitmean_parent_and_matched_prior_all_strata':mapped_gate,
                'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out


def run(config_path):
    cfg,fc,bars,groups,y,masks,forecasts,controls,calibration,pre=prepare(config_path);out=Path(cfg['output_dir'])
    if (out/'results.json').exists():raise ValueError('immutable direction run already completed')
    if file_digest(out/'preflight.json')!=cfg['preflight_sha256'] or json.loads((out/'preflight.json').read_text())!=pre:
        raise ValueError('registered direction preflight changed')
    reg={'config':cfg,'config_sha256':file_digest(config_path),'preflight_sha256':cfg['preflight_sha256'],
         'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'scope':'32 fixed linear classifier fits,16 shared fit priors;16 new causal policies;36 old controls'}
    _immutable_json(out/'registration.json',reg)
    ex=fc['execution'];stress={**ex,'one_way_cost':2*ex['one_way_cost'],'borrow_annual':2*ex['borrow_annual']}
    all_rows=[];all_scores=[];all_cs=[];all_fits=[]
    parent_scores={(r['fold'],r['mean_id']):r for r in json.loads((PARENT_ROOT/'results.json').read_text())['scores'] if r['subset']=='all'}
    for f in FOLDS:
        dates=calendar(f-1);mask=masks[f]
        cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        eval_ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        window=bars.loc[eval_ix];regime=controls[f,'bh']['regime'];ref=forecasts[f,PARENTS[GROUPS[0]]]
        fitted=fit_direction_family(groups,y,fit_mask=mask['fit'],predict_mask=mask['predict'])
        if fitted['fit_return_mean']!=float(ref['fit_return_mean']):raise ValueError('fit mean changed')
        rows=[];scores=[];cs=[];bindings={}
        def save(kind,name,value,extension='npz'):
            p=out/kind/f'fold{f}_{name}.{extension}';p.parent.mkdir(parents=True,exist_ok=True)
            if extension=='json':_immutable_json(p,value)
            elif extension=='joblib':
                buf=io.BytesIO();joblib.dump(value,buf,compress=3);data=buf.getvalue()
                if p.exists():
                    if p.read_bytes()!=data:raise ValueError('partial model differs')
                else:p.write_bytes(data)
            elif p.exists():
                prior=arrays(p)
                if set(prior)!=set(value):raise ValueError('partial array schema differs')
                for k in prior:compare_array(prior[k],value[k],name=str(p)+k,exact=True)
            else:np.savez_compressed(p,**value)
            bindings[str(p)]=file_digest(p);return p
        logits=dict(fitted['logits']);probabilities=dict(fitted['probabilities']);prior_logits={}
        for w in WEIGHTINGS:
            pi=fitted['fit_priors'][w];z=math.log(pi)-math.log1p(-pi);prior_logits[w]=z
            logits['prior_'+w]=np.where(mask['predict'],z,np.nan)
            # Save the sigmoid actually scored from the frozen prior logit, not a rounded alternative.
            probabilities['prior_'+w]=np.where(mask['predict'],1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z)),np.nan)
        for mid in MODEL_IDS:save('models',mid,fitted['models'][mid],'joblib')
        fitdata={'fit_positions':np.flatnonzero(mask['fit']),'timestamps':bars.index[mask['fit']].asi8,
            'returns':y[mask['fit'],0],'binary_labels':fitted['fit_labels'],
            'predict_positions':np.flatnonzero(mask['predict']),'predict_timestamps':bars.index[mask['predict']].asi8,
            **{'fit_features_'+g:groups[g].to_numpy()[mask['fit']] for g in GROUPS},
            **{'predict_features_'+g:groups[g].to_numpy()[mask['predict']] for g in GROUPS},
            **{'weights_'+w:fitted['fit_weights'][w] for w in WEIGHTINGS}}
        save('fit_data','training',fitdata)
        fit_record={'fold':f,'fit_provenance':fitted['provenance'],'fit_priors':fitted['fit_priors'],
            'fit_prior_logits':prior_logits,'fit_return_mean':fitted['fit_return_mean'],
            'fit_abs_return_mean':fitted['fit_abs_return_mean'],'fit_source_binding':pre['support'][FOLDS.index(f)],
            'risk_source':'unchanged_saved_technical_scaled','new_model_fits':4,'shared_prior_estimates':2}
        save('provenance','fit',fit_record,'json');all_fits.append(fit_record)
        eval_means={m:forecasts[f,m]['mu'] for m in PARENTS.values()}
        cal_means={PARENTS[g]:calibration[f,g]['parent_mu'] for g in GROUPS}
        for m in NEW_MEANS:
            mp=MAPPING[m];mid=mp['classifier_id'];original=forecasts[f,mp['parent_mean']];ca=calibration[f,mp['group']]
            eval_means[m]=map_direction(logits[mid][eval_ix],original['mu'],original['inference_mask'])
            cal_means[m]=map_direction(logits[mid][cal_ix],ca['parent_mu'],ca['mapped_inference_mask'])
            save('forecasts',m,{**original,'mu':eval_means[m],'parent_mu':original['mu'],
                'logit':logits[mid][eval_ix],'probability':probabilities[mid][eval_ix]})
            save('calibration',m,{'timestamps':ca['timestamps'],'actual':ca['actual'],
                'scale_mask':ca['scale_mask'],'interval_mask':ca['interval_mask'],
                'classifier_predict_mask':ca['classifier_predict_mask'],'mapped_inference_mask':ca['mapped_inference_mask'],
                'mu':cal_means[m],'parent_mu':ca['parent_mu'],'logit':logits[mid][cal_ix],
                'probability':probabilities[mid][cal_ix],'fit_return_mean':ref['fit_return_mean']})
        for seg in SEGMENTS:
            actual,support,mm,select=(ref['actual'],ref['score_support'],eval_means,eval_ix) if seg=='evaluation' else (
                calibration[f,GROUPS[0]]['actual'],mask['interval'][cal_ix],cal_means,cal_ix)
            meta={'fold':f,'segment':seg,'regime':regime,'regime_known_at_scored_decisions':seg=='evaluation',
                'regime_reference':'evaluation_quarter_start'}
            for m in MEANS:
                score=return_scores(actual,mm[m],support,float(ref['fit_return_mean']))
                if seg=='evaluation' and m in PARENTS.values():
                    compare_tree(score,{k:parent_scores[f,m][k] for k in score},name='unchanged parent return score')
                scores.append({**meta,'mean_id':m,**score})
            for mid in CLASSIFIERS:cs.append({**meta,'classifier_id':mid,**direction_scores(actual,logits[mid][select],support)})
        for cid in POLICIES:
            trace=None
            if cid in CONTROLS:
                saved=arrays(PARENT_ROOT/'targets'/f'fold{f}_{cid}.npz')
                compare_array(saved['timestamps'],window.index.asi8,name='old target calendar',exact=True);target=saved['targets']
            else:
                rule=RULES[1] if cid.endswith(RULES[1]) else RULES[0];m=cid[:-(len(rule)+1)]
                original=forecasts[f,MAPPING[m]['parent_mean']]
                am=action_masks(window.index,window.open.to_numpy(),original['inference_mask'])
                if rule==RULES[1]:
                    target,trace=fallback_targets(window,eval_means[m],original['variance'],ex,inference_mask=original['inference_mask'],risk_aversion=1,cost_multiplier=2)
                    check_action_support(target,am);check_trace_support(target,am,trace)
                else:
                    target,trace=conditional_targets(window,eval_means[m],original['variance'],ex,risk_aversion=1,cost_multiplier=2)
                    if np.any(np.isfinite(target)&~am['learned_eligible']):raise ValueError('hold orders escaped inference support')
                trace['direction_mapping']={**MAPPING[m],'surrogate_mean':cfg['surrogate_mean'],'future_labels_used_for_orders':False}
            p=save('targets',cid,{'timestamps':window.index.asi8,'targets':target})
            row={'fold':f,'candidate_id':cid,'regime':regime,'hindsight_only':cid in HINDSIGHT_IDS,'targets_sha256':bindings[str(p)],
                **{co:metrics(window,target,c) for co,c in [('base',ex),('stress_2x',stress)]}}
            if cid in CONTROLS:
                for co in ('base','stress_2x'):compare_tree(row[co],controls[f,cid][co],name='unchanged old account')
            else:
                p=save('traces',cid,trace,'json');row['trace_sha256']=bindings[str(p)]
            rows.append(row)
        if len(bindings)!=90 or len(rows)!=52 or len(scores)!=20 or len(cs)!=12:raise ValueError('incomplete direction fold')
        _immutable_json(out/f'fold_{f}.json',{'registration_sha256':digest(reg),'rows':rows,'scores':scores,
            'classification_scores':cs,'artifact_sha256':bindings})
        all_rows.extend(rows);all_scores.extend(scores);all_cs.extend(cs)
        print(json.dumps({'event':'fold_complete','fold':f,'new_model_fits':4,'shared_prior_estimates':2,
            'policies':52,'return_scores':20,'classification_scores':12,'artifacts':90}),flush=True)
    result={'registration_sha256':digest(reg),'rows':all_rows,'scores':all_scores,'classification_scores':all_cs,'fit_records':all_fits,
        'summary':summarize(all_rows,all_scores,all_cs),'new_model_fits':32,'shared_prior_estimates':16,
        'new_causal_policy_names':16,'total_adaptively_explored_causal_names':190,
        'additional_test_used_for_modeling_or_scoring':False,'selection_performed':False,'teacher_use_allowed':False,
        'high_probability_generalization_established':False,'risk_model_or_calibration_fits':0}
    _immutable_json(out/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        cfg,*_,pre=prepare(args.config);path=Path(cfg['output_dir'])/'preflight.json';_immutable_json(path,pre)
        print(json.dumps({'path':str(path),'sha256':file_digest(path),'new_labels_priors_weights_fits_or_orders_computed':False}))
    else:run(args.config)
