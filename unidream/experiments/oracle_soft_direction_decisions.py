"""No-fit continuous mapping of four frozen magnitude-weighted probabilities."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from . import oracle_regularized_direction_decisions as parent_module
from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import FOLDS, RULES, compare_array, compare_tree
from .oracle_mean_controls import return_scores
from .oracle_direction_decisions import arrays, GROUPS, PARENTS, HINDSIGHT_IDS, SEGMENTS
from .oracle_direction_scores import direction_scores
from .oracle_soft_direction_mapping import map_soft_direction

PARENT_ROOT=Path('codex_outputs/oracle_regularized_direction_decisions_v1')
DIRECTION_ROOT=Path('codex_outputs/oracle_direction_decisions_v1')
CONTROLS=parent_module.POLICIES
OLD_MEANS=parent_module.MEANS
CLASSIFIERS=parent_module.CLASSIFIERS
PROBABILITY_IDS=tuple(g+'_magnitude'+suffix for g in GROUPS for suffix in ('','_l2unit'))
SOFT_MEANS=tuple(m+'_soft' for m in PROBABILITY_IDS)
CONSTANT_KINDS=('mapped_prior','fit_mean','zero')
CONSTANT_MEANS=tuple(g+'_soft_'+k for g in GROUPS for k in CONSTANT_KINDS)
NEW_MEANS=SOFT_MEANS+CONSTANT_MEANS
MEANS=OLD_MEANS+NEW_MEANS
NEW_IDS=tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
LEARNED_IDS=tuple(m+'_'+r for m in SOFT_MEANS for r in RULES)
POLICIES=CONTROLS+NEW_IDS
MAPPING={m+'_soft':{'group':g,'kind':'soft','role':'learned_mapping',
    'source_classifier':m,'source_mean':m+'_direction','parent_mean':PARENTS[g]}
    for g in GROUPS for m in PROBABILITY_IDS if m.startswith(g+'_')}
MAPPING.update({g+'_soft_'+kind:{'group':g,'kind':kind,'role':'constant_control',
    'source_classifier':g+'_magnitude','source_mean':g+'_magnitude_direction','parent_mean':PARENTS[g]}
    for g in GROUPS for kind in CONSTANT_KINDS})
REFERENCES={m:(MAPPING[m]['source_mean'],MAPPING[m]['parent_mean'],
    *(MAPPING[m]['group']+'_soft_'+k for k in CONSTANT_KINDS)) for m in SOFT_MEANS}
SOURCES=parent_module.SOURCES+tuple('unidream/experiments/'+n for n in (
    'oracle_soft_direction_mapping.py','oracle_soft_direction_decisions.py'))
FIXED={'schema':'oracle-soft-direction-decisions-v1','development_folds':list(FOLDS),
    'data_cutoff':'2023-04-16T13:45:00Z','parent_config':'configs/oracle_regularized_direction_decisions_20260906.yaml',
    'parent_root':str(PARENT_ROOT),'parent_source_revision':'5a82c270c64a342ab7e9df8105b7d23d1336d876',
    'parent_publication_revision':'96447c4600979c4ce5c66140fe60ddd27448c2d2',
    'data_path':'/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/checkpoints/alpha_dd_data/spot_15m.parquet',
    'output_dir':'codex_outputs/oracle_soft_direction_decisions_v1','groups':list(GROUPS),
    'probability_ids':list(PROBABILITY_IDS),'classifiers':list(CLASSIFIERS),
    'new_mean_ids':list(NEW_MEANS),'return_score_means':list(MEANS),'control_ids':list(CONTROLS),
    'new_policy_ids':list(NEW_IDS),'learned_policy_ids':list(LEARNED_IDS),
    'mapping':MAPPING,'references':{m:list(v) for m,v in REFERENCES.items()},'rules':list(RULES),'segments':list(SEGMENTS),
    'surrogate_mean':'saved_fit_abs_return_mean*(2.0*saved_probability-1.0)',
    'prior_probability_source':'bound_Stage17_prior_NPZ_probability_not_raw_fit_prior_or_new_sigmoid',
    'prior_identity_absolute_tolerance':1e-14,'prior_identity_relative_tolerance':1e-12,
    'new_model_fits':0,'new_unique_fit_priors':0,'new_causal_names':20,'new_learned_policy_names':8,
    'new_constant_policy_names':12,'adaptive_prior_causal_names':198,'adaptive_total_causal_names':218,
    'score_classification_records':160,'score_return_records':384,'mapping_diagnostic_records':64,
    'fold_artifacts':121,'economic_rows':640,'economic_accounts':1280,
    'new_mean_risk_probability_or_weight_calibration_permitted':False,
    'risk_source':'unchanged_saved_technical_scaled','utility_risk_aversion':1,'utility_cost_multiplier':2,
    'execution':{'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01},
    'inference_rows':2586,'score_rows':2574,'interval_inference_rows':2537,'interval_score_rows':2523,
    'fallback_rows':332,'missing_current_open_rows':2,
    'selection_permitted':False,'teacher_use_allowed':False,'additional_test_permitted':False,
    'numpy_version':'2.2.6','pandas_version':'2.3.3','sklearn_version':'1.8.0',
    'partial_retry_policy':'full_replay_verify_existing_no_live_restart'}
EXTRA={'source_bindings','parent_config_sha256','parent_manifest_bindings','preflight_sha256'}


def validate_config(cfg):
    if (set(cfg)!=set(FIXED)|EXTRA or any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in FIXED.items())
            or set(cfg['source_bindings'])!=set(SOURCES)):
        raise ValueError('unregistered soft direction family')
    expected={str(PARENT_ROOT/n) for n in ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in FOLDS]}
    if set(cfg['parent_manifest_bindings'])!=expected:raise ValueError('incomplete parent manifest inventory')


def source_root(mean):
    return PARENT_ROOT if mean in parent_module.NEW_MEANS else DIRECTION_ROOT


def prepare(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());validate_config(cfg)
    direct={**cfg['source_bindings'],**cfg['parent_manifest_bindings'],cfg['parent_config']:cfg['parent_config_sha256']}
    for p,sha in direct.items():
        if file_digest(Path(p))!=sha:raise ValueError('bound source changed: '+p)
    pc,bars,packs,masks,provenance,evaluation,calibration,_,grandparent,pp=parent_module.prepare(cfg['parent_config'])
    parent=json.loads((PARENT_ROOT/'results.json').read_text());reg=json.loads((PARENT_ROOT/'registration.json').read_text())
    if (reg['config']!=pc or reg['config_sha256']!=cfg['parent_config_sha256'] or reg['source_revision']!=cfg['parent_source_revision']
            or reg['preflight_sha256']!=file_digest(PARENT_ROOT/'preflight.json') or pc['preflight_sha256']!=reg['preflight_sha256']
            or parent['registration_sha256']!=digest(reg) or json.loads((PARENT_ROOT/'preflight.json').read_text())!=pp
            or pc['data_cutoff']!=cfg['data_cutoff']):raise ValueError('completed Stage18 chain changed')
    for p,sha in pp['direct_source_bindings'].items():
        if p in direct and direct[p]!=sha:raise ValueError('conflicting source binding')
        direct[p]=sha
    bindings=dict(pp['source_artifact_bindings'])
    if len(bindings)!=2840:raise ValueError('incomplete Stage18 ancestry')
    for f in FOLDS:
        fold=json.loads((PARENT_ROOT/f'fold_{f}.json').read_text())
        if fold['registration_sha256']!=digest(reg) or len(fold['artifact_sha256'])!=81:raise ValueError('parent fold changed')
        for key in ('rows','scores','classification_scores','direction_diagnostics'):
            if fold[key]!=[r for r in parent[key] if r['fold']==f]:raise ValueError('parent fold record changed')
        for p,sha in fold['artifact_sha256'].items():
            if p in bindings:raise ValueError('duplicate inherited artifact')
            bindings[p]=sha
    if len(bindings)!=3488 or len({str(Path(p).resolve()) for p in bindings})!=3488:raise ValueError('ancestor inventory/aliases changed')
    for p,sha in {**direct,**bindings}.items():
        if file_digest(Path(p))!=sha:raise ValueError('bound artifact changed: '+p)
    controls={(r['fold'],r['candidate_id']):r for r in parent['rows']}
    if len(controls)!=480 or set(controls)!={(f,c) for f in FOLDS for c in CONTROLS}:raise ValueError('old control inventory changed')
    if len(parent['scores'])!=224 or len(parent['classification_scores'])!=160:raise ValueError('old score inventory changed')
    support=[];scalars={}
    for f in FOLDS:
        dates=calendar(f-1)
        cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        eval_ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        for m in parent_module.NEW_MEANS:
            for kind,destination,ix in (('forecasts',evaluation,eval_ix),('calibration',calibration,cal_ix)):
                path=PARENT_ROOT/kind/f'fold{f}_{m}.npz'
                if str(path) not in bindings:raise ValueError('unbound parent prediction')
                a=arrays(path);destination[f,m]=a
                ref=destination[f,parent_module.MAPPING[m]['old_mean']]
                if set(a)!=set(ref):raise ValueError('parent prediction schema changed')
                for k in a:
                    if k not in ('mu','probability','logit'):compare_array(a[k],ref[k],name='parent paired '+k,exact=True)
                compare_array(a['timestamps'],bars.index[ix].asi8,name='parent prediction calendar',exact=True)
        prov=provenance[f]
        scalars[f]={'fit_abs_return_mean':prov['fit_abs_return_mean'],'fit_return_mean':prov['fit_return_mean'],
            'fit_statistical_magnitude_prior':prov['fit_priors']['magnitude'],'prior_probability':{},'prior_paths':{}}
        q_inputs={}
        for g in GROUPS:
            prior=g+'_magnitude_prior_direction';ps=[]
            for seg,destination,kind,maskkey in (('interval',calibration,'calibration','mapped_inference_mask'),('evaluation',evaluation,'forecasts','inference_mask')):
                a=destination[f,prior];mask=a[maskkey];q=a['probability']
                if mask.dtype!=bool or not mask.any() or not np.isfinite(q[mask]).all():raise ValueError('invalid saved prior support')
                v=float(q[mask][0])
                if not 0<v<1 or not np.all(q[mask]==v):raise ValueError('saved prior is not constant')
                ps.append(v);path=DIRECTION_ROOT/kind/f'fold{f}_{prior}.npz'
                q_inputs[f'{g}/{seg}/prior']={'path':str(path),'sha256':bindings[str(path)]}
            if ps[0]!=ps[1]:raise ValueError('saved prior changed between I/E')
            scalars[f]['prior_probability'][g]=ps[0]
            scalars[f]['prior_paths'][g]={seg:q_inputs[f'{g}/{seg}/prior'] for seg in SEGMENTS}
        if len(set(scalars[f]['prior_probability'].values()))!=1:raise ValueError('saved prior differs between groups')
        for mid in PROBABILITY_IDS:
            m=mid+'_direction'
            for seg,destination,kind,maskkey in (('interval',calibration,'calibration','mapped_inference_mask'),('evaluation',evaluation,'forecasts','inference_mask')):
                a=destination[f,m];mask=a[maskkey];q=a['probability']
                if (q.dtype!=np.float64 or q.shape!=mask.shape or a['logit'].shape!=mask.shape
                        or not np.isfinite(a['logit'][mask]).all() or not np.isfinite(q[mask]).all() or np.any((q[mask]<0)|(q[mask]>1))):
                    raise ValueError('invalid frozen probability')
                if float(a['fit_return_mean'])!=prov['fit_return_mean']:raise ValueError('saved fit mean mismatch')
                path=source_root(m)/kind/f'fold{f}_{m}.npz';q_inputs[f'{mid}/{seg}']={'path':str(path),'sha256':bindings[str(path)]}
        support.append({**pp['support'][FOLDS.index(f)],'saved_probability_inputs':q_inputs,'saved_mapping_scalars':scalars[f],
            'new_fits_or_feature_construction':False})
    pre={'schema':'oracle-soft-direction-preflight-v1','config_contract_sha256':digest({k:v for k,v in cfg.items() if k!='preflight_sha256'}),
        'source_bindings':cfg['source_bindings'],'direct_source_bindings':direct,'source_artifact_bindings':bindings,'support':support,
        'spot_data_proof':pp['spot_data_proof'],'um_data_proof':pp['um_data_proof'],'parent_registration_canonical_sha256':digest(reg),
        'new_statistics_mapped_predictions_losses_or_orders_computed':False,'no_estimator_fit_or_predict_called':True,
        'loader_scope':pp['loader_scope']}
    return cfg,bars,masks,scalars,evaluation,calibration,controls,parent,pre


def exact_tree(got,expected,*,name):
    if compare_tree(got,expected,name=name)!=0.:raise ValueError('exact preserved value changed: '+name)


def average(values):
    return None if any(v is None for v in values) else math.fsum(v/len(values) for v in values)


def summarize(rows,scores,classification_scores):
    ri={(r['fold'],r['candidate_id']):r for r in rows};si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
    if (len(ri)!=len(rows) or set(ri)!={(f,c) for f in FOLDS for c in POLICIES}
            or len(si)!=len(scores) or set(si)!={(f,s,m) for f in FOLDS for s in SEGMENTS for m in MEANS}):
        raise ValueError('incomplete soft direction family')
    old=parent_module.summarize([r for r in rows if r['candidate_id'] in CONTROLS],
        [r for r in scores if r['mean_id'] in OLD_MEANS],classification_scores)
    regimes=('all','bull','bear','sideways');costs=('base','stress_2x')
    pk=('return_mse','return_mae','return_sign_accuracy','return_rank_ic','zero_return_mse','fit_mean_return_mse')
    for f in FOLDS:
        regime=ri[f,'bh']['regime']
        for c in NEW_IDS:
            if ri[f,c]['regime']!=regime or any(type(ri[f,c][co][k]) not in (int,float) or not math.isfinite(ri[f,c][co][k])
                    for co in costs for k in ('alpha_ex','maxdd_delta','turnover','trades')):raise ValueError('invalid new economics')
        for seg in SEGMENTS:
            n=si[f,seg,OLD_MEANS[0]]['rows']
            for m in NEW_MEANS:
                r=si[f,seg,m]
                if type(r['rows']) is not int or r['rows']!=n or r['regime']!=regime:raise ValueError('unpaired new return support')
                for k in pk:
                    v=r[k]
                    if v is None and k=='return_rank_ic':continue
                    if (type(v) not in (int,float) or not math.isfinite(v) or (k!='return_rank_ic' and v<0)
                            or (k=='return_rank_ic' and abs(v)>1) or (k=='return_sign_accuracy' and v>1)):
                        raise ValueError('invalid new return score')
                for k in ('zero_return_mse','fit_mean_return_mse'):
                    if r[k]!=si[f,seg,OLD_MEANS[0]][k]:raise ValueError('new score baseline changed')
                kind=MAPPING[m]['kind']
                if kind in ('zero','fit_mean') and r['return_mse']!=r['zero_return_mse' if kind=='zero' else 'fit_mean_return_mse']:
                    raise ValueError('constant reference score does not match its named baseline')
                if kind!='soft' and r['return_rank_ic'] is not None:raise ValueError('constant reference rank must be undefined')
    out={'economics':{},'prediction':{},'paired':{},'soft':{},'classification':old['classification'],
        'inherited_classification_paired':old['classification_paired'],'inherited_Stage18_direction_flags':old['direction'],
        'regime_counts':old['regime_counts'],'interval_regime_strata_are_retrospective_evaluation_groupings':True,
        'probability_predictions_and_scores_unchanged':True,'new_probability_accuracy_improvement':False,
        'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    for g in regimes:
        ff=[f for f in FOLDS if g=='all' or ri[f,'bh']['regime']['trend']==g]
        out['economics'][g]={c:{'quarters':len(ff),'hindsight_only':c in HINDSIGHT_IDS,
            'joint_positive_quarters_both_costs':sum(all(ri[f,c][co]['alpha_ex']>0 and ri[f,c][co]['maxdd_delta']<0 for co in costs) for f in ff),
            **{co:{k:average([ri[f,c][co][k] for f in ff]) for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs}} for c in POLICIES}
        out['prediction'][g]={seg:{m:{'quarters':len(ff),'rows':sum(si[f,seg,m]['rows'] for f in ff),
            **{k:average([si[f,seg,m][k] for f in ff]) for k in pk},
            'pooled_row_mse':math.fsum(si[f,seg,m]['return_mse']*si[f,seg,m]['rows']/sum(si[t,seg,m]['rows'] for t in ff) for f in ff)}
            for m in MEANS} for seg in SEGMENTS}
        out['paired'][g]={m:{ref:{'prediction':{seg:{
            'mse_difference':average([si[f,seg,m]['return_mse']-si[f,seg,ref]['return_mse'] for f in ff]),
            'improved_quarters':sum(si[f,seg,m]['return_mse']<si[f,seg,ref]['return_mse'] for f in ff),
            'equal_quarters':sum(si[f,seg,m]['return_mse']==si[f,seg,ref]['return_mse'] for f in ff)} for seg in SEGMENTS},
            'economics':{rule:{co:{k:average([ri[f,m+'_'+rule][co][k]-ri[f,ref+'_'+rule][co][k] for f in ff])
                for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs} for rule in RULES}}
            for ref in REFERENCES[m]} for m in SOFT_MEANS}
    for m in SOFT_MEANS:
        mid=MAPPING[m]['source_classifier']
        source_gate={seg:all(out['classification'][g][seg][mid][k] is not None
            and out['classification'][g][seg]['prior_magnitude'][k] is not None
            and out['classification'][g][seg][mid][k]<out['classification'][g][seg]['prior_magnitude'][k]
            for g in regimes for k in ('weighted_brier','weighted_log_loss')) for seg in SEGMENTS}
        mse_gate={seg:all(out['paired'][g][m][ref]['prediction'][seg]['mse_difference']<0
            for g in regimes for ref in REFERENCES[m]) for seg in SEGMENTS}
        for rule in RULES:
            cid=m+'_'+rule
            out['soft'][cid]={'economic_means_all_strata_both_costs':all(out['economics'][g][cid][co]['alpha_ex']>0
                and out['economics'][g][cid][co]['maxdd_delta']<0 for g in regimes for co in costs),
                'economic_improvement_vs_all_five_references_all_strata_both_costs':all(
                    out['paired'][g][m][ref]['economics'][rule][co]['alpha_ex']>0
                    and out['paired'][g][m][ref]['economics'][rule][co]['maxdd_delta']<0
                    for g in regimes for ref in REFERENCES[m] for co in costs),
                'mapped_mse_vs_all_five_references_improved_all_strata':mse_gate,
                'inherited_source_weighted_losses_below_prior_all_strata':source_gate,
                'new_probability_accuracy_improvement':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out


def map_saved_inputs(original,scalars,group,mask_key):
    """The mapper sees saved probabilities and T scalars only, never outcomes."""
    return map_soft_direction(original['probability'],inference_mask=original[mask_key],
        fit_abs_return_mean=scalars['fit_abs_return_mean'],
        saved_weighted_prior_probability=scalars['prior_probability'][group],fit_return_mean=scalars['fit_return_mean'])


def run(config_path):
    cfg,bars,masks,scalars,evaluation,calibration,controls,parent,pre=prepare(config_path);out=Path(cfg['output_dir'])
    if (out/'results.json').exists():raise ValueError('immutable soft direction run completed')
    if file_digest(out/'preflight.json')!=cfg['preflight_sha256'] or json.loads((out/'preflight.json').read_text())!=pre:
        raise ValueError('registered soft preflight changed')
    reg={'config':cfg,'config_sha256':file_digest(config_path),'preflight_sha256':cfg['preflight_sha256'],
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'0 fits;4 frozen probability mappings and6 constant means;20 new causal policies;60 old controls'}
    _immutable_json(out/'registration.json',reg)
    ex=cfg['execution'];stress={**ex,'one_way_cost':2*ex['one_way_cost'],'borrow_annual':2*ex['borrow_annual']}
    old_scores={(r['fold'],r['segment'],r['mean_id']):r for r in parent['scores']}
    old_cs={(r['fold'],r['segment'],r['classifier_id']):r for r in parent['classification_scores']}
    all_rows=[];all_scores=[];all_cs=[];all_records=[];all_diagnostics=[]
    for f in FOLDS:
        dates=calendar(f-1);mm=masks[f]
        cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        eval_ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        window=bars.loc[eval_ix];regime=controls[f,'bh']['regime'];ref=evaluation[f,PARENTS[GROUPS[0]]]
        rows=[];scores=[];cs=[];diagnostics=[];bindings={}
        def save(kind,name,value,extension='npz'):
            p=out/kind/f'fold{f}_{name}.{extension}';p.parent.mkdir(parents=True,exist_ok=True)
            if extension=='json':_immutable_json(p,value)
            elif p.exists():
                old=arrays(p)
                if set(old)!=set(value):raise ValueError('partial array schema differs')
                for k in old:compare_array(old[k],value[k],name=str(p)+k,exact=True)
            else:np.savez_compressed(p,**value)
            bindings[str(p)]=file_digest(p);return p
        eval_means={m:evaluation[f,m]['mu'] for m in OLD_MEANS};cal_means={m:calibration[f,m]['mu'] for m in OLD_MEANS}
        mapped={};mapping_record={'fold':f,'saved_T_scalars':scalars[f],'new_fits':0,'new_unique_priors':0,
            'frozen_input_bindings':pre['support'][FOLDS.index(f)]['saved_probability_inputs'],
            'mean_records':{},'mapping_formula':cfg['surrogate_mean'],'probability_arrays_unchanged':True,'caller_saved_input_provenance_and_calendar_verified':True}
        for mid in PROBABILITY_IDS:
            m=mid+'_soft';mp=MAPPING[m];original=evaluation[f,mp['source_mean']];ca=calibration[f,mp['source_mean']]
            mapped[mid,'evaluation']=map_saved_inputs(original,scalars[f],mp['group'],'inference_mask')
            mapped[mid,'interval']=map_saved_inputs(ca,scalars[f],mp['group'],'mapped_inference_mask')
            for seg,a,maskkey in (('evaluation',original,'inference_mask'),('interval',ca,'mapped_inference_mask')):
                support=a[maskkey];q=a['probability'][support];z=a['logit'][support];mu=mapped[mid,seg]['means']['soft'][support];n=len(mu)
                diagnostics.append({'fold':f,'segment':seg,'mean_id':m,'source_classifier':mid,'regime':regime,'rows':n,
                    'uses_all_inference_not_score_support':True,'probability_half_rows':int((q==.5).sum()),
                    'probability_zero_rows':int((q==0).sum()),'probability_one_rows':int((q==1).sum()),
                    'source_zero_logit_rows':int((z==0).sum()),'mapped_zero_mean_rows':int((mu==0).sum()),
                    'probability_direction_vs_logit_disagreements':int((np.sign(2.*q-1.)!=np.sign(z)).sum()),
                    'mapped_direction_vs_logit_disagreements':int((np.sign(mu)!=np.sign(z)).sum()),
                    'mean_abs_new_mu':math.fsum(abs(float(v))/n for v in mu),
                    'mean_abs_hard_mu':math.fsum(abs(float(v))/n for v in a['mu'][support]),
                    'mean_abs_parent_mu':math.fsum(abs(float(v))/n for v in a['parent_mu'][support]),
                    'new_abs_mu_greater_than_hard_rows':int((np.abs(mu)>np.abs(a['mu'][support])).sum()),
                    'new_abs_mu_equal_to_hard_rows':int((np.abs(mu)==np.abs(a['mu'][support])).sum())})
        for m in NEW_MEANS:
            mp=MAPPING[m];mid=mp['source_classifier'];original=evaluation[f,mp['source_mean']];ca=calibration[f,mp['source_mean']]
            eval_means[m]=mapped[mid,'evaluation']['means'][mp['kind']]
            cal_means[m]=mapped[mid,'interval']['means'][mp['kind']]
            save('forecasts',m,{**original,'mu':eval_means[m]})
            save('calibration',m,{**ca,'mu':cal_means[m]})
            mapping_record['mean_records'][m]={**mp,'source_probability_and_logit_fields_are_preserved_source_evidence':True,
                'segment_diagnostics':{seg:mapped[mid,seg]['diagnostic'] for seg in SEGMENTS}}
        save('provenance','mapping',mapping_record,'json');all_records.append(mapping_record)
        for seg in SEGMENTS:
            actual,support,means=(ref['actual'],ref['score_support'],eval_means) if seg=='evaluation' else (
                calibration[f,PARENTS[GROUPS[0]]]['actual'],mm['interval'][cal_ix],cal_means)
            meta={'fold':f,'segment':seg,'regime':regime,'regime_known_at_scored_decisions':seg=='evaluation','regime_reference':'evaluation_quarter_start'}
            for m in MEANS:
                score=return_scores(actual,means[m],support,float(ref['fit_return_mean']))
                if m in OLD_MEANS:exact_tree(score,{k:old_scores[f,seg,m][k] for k in score},name='unchanged old return score')
                record={**meta,'mean_id':m,**score}
                if m in OLD_MEANS and record!=old_scores[f,seg,m]:raise ValueError('complete old score changed')
                scores.append(record)
            for mid in CLASSIFIERS:
                mean=mid+'_direction' if not mid.startswith('prior_') else GROUPS[0]+'_'+mid.removeprefix('prior_')+'_prior_direction'
                z=(evaluation if seg=='evaluation' else calibration)[f,mean]['logit']
                score=direction_scores(actual,z,support)
                exact_tree(score,{k:old_cs[f,seg,mid][k] for k in score},name='unchanged classification score')
                record={**meta,'classifier_id':mid,**score}
                if record!=old_cs[f,seg,mid]:raise ValueError('complete classification score changed')
                cs.append(record)
        for cid in POLICIES:
            trace=None
            if cid in CONTROLS:
                saved=arrays(PARENT_ROOT/'targets'/f'fold{f}_{cid}.npz')
                compare_array(saved['timestamps'],window.index.asi8,name='old target calendar',exact=True);target=saved['targets']
            else:
                rule=RULES[1] if cid.endswith(RULES[1]) else RULES[0];m=cid[:-(len(rule)+1)];mp=MAPPING[m]
                original=evaluation[f,mp['source_mean']];am=action_masks(window.index,window.open.to_numpy(),original['inference_mask'])
                if rule==RULES[1]:
                    target,trace=fallback_targets(window,eval_means[m],original['variance'],ex,inference_mask=original['inference_mask'],risk_aversion=1,cost_multiplier=2)
                    check_action_support(target,am);check_trace_support(target,am,trace)
                else:
                    target,trace=conditional_targets(window,eval_means[m],original['variance'],ex,risk_aversion=1,cost_multiplier=2)
                    if np.any(np.isfinite(target)&~am['learned_eligible']):raise ValueError('hold order escaped inference support')
                trace['soft_direction_mapping']={**mp,'mapping_formula':cfg['surrogate_mean'],
                    'saved_T_scalars':scalars[f],'future_labels_used_for_orders':False,'new_fit_or_calibration':False}
            p=save('targets',cid,{'timestamps':window.index.asi8,'targets':target})
            row={'fold':f,'candidate_id':cid,'regime':regime,'hindsight_only':cid in HINDSIGHT_IDS,'targets_sha256':bindings[str(p)],
                **{co:metrics(window,target,c) for co,c in [('base',ex),('stress_2x',stress)]}}
            if cid in CONTROLS:
                for co in ('base','stress_2x'):exact_tree(row[co],controls[f,cid][co],name='unchanged old account')
                if row['targets_sha256']!=controls[f,cid]['targets_sha256']:raise ValueError('old target file bytes changed')
                row=controls[f,cid].copy()
            else:
                p=save('traces',cid,trace,'json');row['trace_sha256']=bindings[str(p)]
            rows.append(row)
        if len(bindings)!=121 or len(rows)!=80 or len(scores)!=48 or len(cs)!=20 or len(diagnostics)!=8:raise ValueError('incomplete soft direction fold')
        _immutable_json(out/f'fold_{f}.json',{'registration_sha256':digest(reg),'rows':rows,'scores':scores,
            'classification_scores':cs,'mapping_diagnostics':diagnostics,'artifact_sha256':bindings})
        all_rows.extend(rows);all_scores.extend(scores);all_cs.extend(cs);all_diagnostics.extend(diagnostics)
        print(json.dumps({'event':'fold_complete','fold':f,'new_model_fits':0,'new_unique_priors':0,
            'policies':80,'return_scores':48,'classification_scores':20,'mapping_diagnostics':8,'artifacts':121}),flush=True)
    result={'registration_sha256':digest(reg),'rows':all_rows,'scores':all_scores,'classification_scores':all_cs,
        'mapping_records':all_records,'mapping_diagnostics':all_diagnostics,'summary':summarize(all_rows,all_scores,all_cs),
        'new_model_fits':0,'new_unique_priors':0,'new_causal_policy_names':20,'new_learned_policy_names':8,
        'new_constant_control_policy_names':12,'total_adaptively_explored_causal_names':218,
        'additional_test_used_for_modeling_or_scoring':False,'selection_performed':False,'teacher_use_allowed':False,
        'high_probability_generalization_established':False,'risk_model_or_calibration_fits':0,
        'probability_predictions_and_scores_unchanged':True,'new_probability_accuracy_improvement':False}
    _immutable_json(out/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        cfg,*_,pre=prepare(args.config);path=Path(cfg['output_dir'])/'preflight.json';_immutable_json(path,pre)
        print(json.dumps({'path':str(path),'sha256':file_digest(path),'new_real_statistics_mapped_predictions_or_orders_computed':False}))
    else:run(args.config)
