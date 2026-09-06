"""Frozen sign/magnitude hindsight diagnostics; no new causal model or selection."""
from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks, check_action_support, check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import FOLDS, RULES, compare_array, compare_tree, prepare as prepare_parity
from .oracle_information_decomposition import (
    HALVES, POLICIES as CONTROLS, ORACLES as OLD_ORACLES, SOURCES as PARENT_SOURCES, prepare as prepare_parent,
)
from .oracle_information_interventions import mark_hindsight_trace
from .oracle_mean_controls import return_scores
from .oracle_sign_magnitude_interventions import substitute_return_component

PARENT_ROOT = Path('codex_outputs/oracle_information_decomposition_v1')
PARITY_ROOT = Path('codex_outputs/oracle_frozen_procedure_parity_v1')
COMPONENTS = ('sign', 'magnitude')
CELLS = {h: {'base': h, 'sign': h+'_oracle_sign', 'magnitude': h+'_oracle_magnitude',
             'full': h+'_oracle_return'} for h in HALVES}
NEW_MEANS = tuple(CELLS[h][c] for h in HALVES for c in COMPONENTS)
SCORE_MEANS = tuple(m for cells in CELLS.values() for m in cells.values())
NEW_IDS = tuple(m+'_'+rule for m in NEW_MEANS for rule in RULES)
POLICIES = CONTROLS + NEW_IDS
SUBSETS = ('all', 'fit_q90_large', 'fit_q90_other')
CONTRASTS = (('sign','base'), ('magnitude','base'), ('full','base'), ('full','sign'), ('full','magnitude'))
SOURCES = PARENT_SOURCES + tuple('unidream/experiments/'+n for n in
    ('oracle_sign_magnitude_interventions.py','oracle_sign_magnitude_decisions.py'))
FIXED = {'schema':'oracle-sign-magnitude-decisions-v1', 'development_folds':list(FOLDS),
    'data_cutoff':'2023-04-16T13:45:00Z', 'parent_root':str(PARENT_ROOT),
    'parent_config':'configs/oracle_information_decomposition_20260906.yaml',
    'parity_config':'configs/oracle_frozen_procedure_parity_20260906.yaml',
    'parent_source_revision':'d3b25734a34915049a327256bd9f99cd9aea8336',
    'output_dir':'codex_outputs/oracle_sign_magnitude_decisions_v1',
    'halves':list(HALVES), 'components':list(COMPONENTS), 'cells':CELLS,
    'control_ids':list(CONTROLS), 'new_diagnostic_ids':list(NEW_IDS), 'rules':list(RULES),
    'score_means':list(SCORE_MEANS), 'score_subsets':list(SUBSETS),
    'contrasts':[list(c) for c in CONTRASTS], 'interaction':'full-sign-magnitude+base',
    'tail_quantile':0.9, 'tail_quantile_method':'linear', 'tail_fit_only':True,
    'tail_membership':'abs_actual_return_ge_fit_q90_is_hindsight_diagnostic_only',
    'new_model_fits':0, 'fit_distribution_thresholds':8, 'new_causal_names':0,
    'adaptive_causal_names_unchanged':174, 'new_hindsight_policy_names':8,
    'risk_source':'unchanged_saved_technical_scaled', 'utility_risk_aversion':1,
    'utility_cost_multiplier':2, 'inference_rows':2586, 'score_rows':2574,
    'fallback_rows':332, 'missing_current_open_rows':2,
    'replacement_support':'saved_score_only_keep_own_learned_elsewhere',
    'selection_permitted':False, 'teacher_use_allowed':False, 'additional_test_permitted':False,
    'numpy_version':'2.2.6', 'pandas_version':'2.3.3',
    'partial_retry_policy':'full_deterministic_replay_no_live_restart'}
EXTRA = {'source_bindings','parent_config_sha256','parity_config_sha256','parent_manifest_bindings','preflight_sha256'}


def validate_config(cfg):
    if (set(cfg)!=set(FIXED)|EXTRA or any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in FIXED.items())
            or set(cfg['source_bindings'])!=set(SOURCES)):
        raise ValueError('unregistered sign/magnitude diagnostic family')
    expected={str(PARENT_ROOT/n) for n in ['registration.json','results.json','preflight.json']+
              [f'fold_{f}.json' for f in FOLDS]}
    if set(cfg['parent_manifest_bindings'])!=expected: raise ValueError('incomplete parent manifest inventory')


def arrays(path):
    with np.load(path,allow_pickle=False) as saved: return {k:saved[k] for k in saved.files}


def prepare(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text()); validate_config(cfg)
    if np.__version__!=cfg['numpy_version'] or pd.__version__!=cfg['pandas_version']:
        raise ValueError('registered runtime changed')
    direct={**cfg['source_bindings'],**cfg['parent_manifest_bindings'],
            cfg['parent_config']:cfg['parent_config_sha256'],cfg['parity_config']:cfg['parity_config_sha256']}
    for p,h in direct.items():
        if file_digest(Path(p))!=h: raise ValueError('registered source changed: '+p)
    pc,fc,bars,forecasts,_,pp=prepare_parent(Path(cfg['parent_config']))
    parent=json.loads((PARENT_ROOT/'results.json').read_text()); reg=json.loads((PARENT_ROOT/'registration.json').read_text())
    if (reg['config']!=pc or reg['config_sha256']!=cfg['parent_config_sha256']
            or reg['source_revision']!=cfg['parent_source_revision']
            or reg['preflight_sha256']!=file_digest(PARENT_ROOT/'preflight.json')
            or pp!=json.loads((PARENT_ROOT/'preflight.json').read_text())
            or parent['registration_sha256']!=digest(reg)):
        raise ValueError('Stage12 completed chain changed')
    controls={(r['fold'],r['candidate_id']):r for r in parent['rows']}
    if len(controls)!=224 or set(controls)!={(f,c) for f in FOLDS for c in CONTROLS}:
        raise ValueError('incomplete old diagnostic/control family')
    bindings=dict(pp['source_artifact_bindings'])
    par=prepare_parity(Path(cfg['parity_config']))
    _,_,pfc,pbars,_,_,y,masks,_,_,_,_=par
    if pfc!=fc or not pbars.equals(bars) or fc['data_cutoff']!=cfg['data_cutoff']:
        raise ValueError('fit/evaluation data differs from parent')
    fit_returns={};support=[]
    for f in FOLDS:
        fold=json.loads((PARENT_ROOT/f'fold_{f}.json').read_text())
        if (fold['registration_sha256']!=digest(reg) or len(fold['artifact_sha256'])!=50
                or fold['rows']!=[r for r in parent['rows'] if r['fold']==f]):
            raise ValueError('completed Stage12 fold changed')
        for p,h in fold['artifact_sha256'].items():
            if file_digest(Path(p))!=h or (p in bindings and bindings[p]!=h): raise ValueError('parent artifact mismatch')
            bindings[p]=h
        dates=calendar(f-1);ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        for h in HALVES:
            ref=forecasts[f,h]
            for k,v in [('inference_mask',masks[f]['inference'][ix]),('score_support',masks[f]['score'][ix])]:
                compare_array(ref[k],v,name='original '+k,exact=True)
            full=arrays(PARENT_ROOT/'forecasts'/f'fold{f}_{h}_oracle_return.npz')
            for k in ref:
                if k!='mu': compare_array(full[k],ref[k],name='full return oracle '+k,exact=True)
            compare_array(full['mu'][ref['score_support']],ref['actual'][ref['score_support'],0],name='saved full oracle',exact=True)
            compare_array(full['mu'][~ref['score_support']],ref['mu'][~ref['score_support']],name='saved full remainder',exact=True)
            forecasts[f,CELLS[h]['full']]=full
        v=np.asarray(y[masks[f]['fit'],0],dtype=float).copy()
        if len(v)<512 or not np.isfinite(v).all(): raise ValueError('invalid original fit return support')
        fit_returns[f]=v
        support.append({'fold':f,'fit_rows':len(v),'fit_return_sha256':digest(v.tolist()),
            'fit_mask_sha256':mask_digest(bars.index,masks[f]['fit']),
            'inference_mask_sha256':mask_digest(bars.index,masks[f]['inference']),
            'score_mask_sha256':mask_digest(bars.index,masks[f]['score']),
            'inference_rows':int(masks[f]['inference'].sum()),'score_rows':int(masks[f]['score'].sum()),
            'regime':controls[f,'bh']['regime']})
    if len(bindings)!=1728 or len({str(Path(p).resolve()) for p in bindings})!=1728:
        raise ValueError('ancestor count or alias changed')
    if sum(x['inference_rows'] for x in support)!=2586 or sum(x['score_rows'] for x in support)!=2574:
        raise ValueError('original supports changed')
    pre={'schema':'oracle-sign-magnitude-preflight-v1','config_contract_sha256':digest({k:v for k,v in cfg.items() if k!='preflight_sha256'}),
        'source_bindings':cfg['source_bindings'],'direct_source_bindings':direct,'source_artifact_bindings':bindings,
        'support':support,'spot_data_proof':pp['spot_data_proof'],'um_data_proof':pp['um_data_proof'],
        'new_forecasts_orders_scores_or_quantiles_computed':False,'new_model_fits':0}
    return cfg,fc,bars,forecasts,controls,fit_returns,pre


def tail_groups(fit_returns,actual,score_support):
    fitted=np.asarray(fit_returns);actual=np.asarray(actual);score=np.asarray(score_support)
    if (fitted.ndim!=1 or len(fitted)<512 or fitted.dtype.kind not in 'fiu' or not np.isfinite(fitted).all()
            or score.ndim!=1 or score.dtype!=bool or actual.shape!=(len(score),3) or not score.any()):
        raise ValueError('registered fit/scoring support required for tail diagnostic')
    selected=actual[score,0]
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) or not math.isfinite(float(v)) for v in selected):
        raise ValueError('finite real scored return required')
    selected=np.asarray(selected,dtype=float)
    q=float(np.quantile(np.abs(fitted),.9,method='linear'))
    large=np.zeros(len(score),bool);large[score]=np.abs(selected)>=q
    return q,{'all':score.copy(),'fit_q90_large':large,'fit_q90_other':score&~large}


def score_forecast(original,mu,mask):
    if mask.any(): return return_scores(original['actual'],mu,mask,float(original['fit_return_mean']))
    return {'rows':0,**{k:None for k in ('return_mse','return_mae','return_sign_accuracy','zero_return_mse',
            'fit_mean_return_mse','return_rank_ic')}}


def average(values):
    return math.fsum(x/len(values) for x in values) if values and all(x is not None for x in values) else None


def summarize(rows,scores):
    index={(r['fold'],r['candidate_id']):r for r in rows};si={(r['fold'],r['mean_id'],r['subset']):r for r in scores}
    if (len(index)!=len(rows) or set(index)!={(f,c) for f in FOLDS for c in POLICIES}
            or len(si)!=len(scores) or set(si)!={(f,m,u) for f in FOLDS for m in SCORE_MEANS for u in SUBSETS}):
        raise ValueError('incomplete diagnostic or score family')
    metric_keys=('return_mse','return_mae','return_sign_accuracy','return_rank_ic','zero_return_mse','fit_mean_return_mse')
    def finite(v):return isinstance(v,Real) and not isinstance(v,(bool,np.bool_)) and math.isfinite(float(v))
    regime_counts={g:sum(index[f,'bh']['regime']['trend']==g for f in FOLDS) for g in ('bull','bear','sideways')}
    if regime_counts!={'bull':2,'bear':4,'sideways':2}:raise ValueError('original regime inventory changed')
    for row in rows:
        for co in ('base','stress_2x'):
            if any(not finite(row[co][k]) for k in ('alpha_ex','maxdd_delta','turnover','trades')):
                raise ValueError('nonfinite economic metric')
            if row[co]['turnover']<0 or row[co]['trades']<0:raise ValueError('negative execution count')
    for row in scores:
        n=row['rows']
        if type(n) is not int or n<0:raise ValueError('invalid score count')
        if n==0:
            if any(row[k] is not None for k in metric_keys):raise ValueError('empty score must be null')
        else:
            if any(not finite(row[k]) or row[k]<0 for k in ('return_mse','return_mae','zero_return_mse','fit_mean_return_mse')):
                raise ValueError('nonfinite or negative loss')
            if not finite(row['return_sign_accuracy']) or not 0<=row['return_sign_accuracy']<=1:raise ValueError('invalid sign accuracy')
            if row['return_rank_ic'] is not None and (not finite(row['return_rank_ic']) or not -1<=row['return_rank_ic']<=1):
                raise ValueError('invalid rank correlation')
    for f in FOLDS:
        for m in SCORE_MEANS:
            if si[f,m,'all']['rows']<=0 or si[f,m,'all']['rows']!=si[f,m,'fit_q90_large']['rows']+si[f,m,'fit_q90_other']['rows']:
                raise ValueError('tail rows do not partition original scored support')
        regime=index[f,'bh']['regime']
        if any(index[f,c]['regime']!=regime for c in POLICIES) or any(si[f,m,u]['regime']!=regime for m in SCORE_MEANS for u in SUBSETS):
            raise ValueError('unpaired regime classifications')
        for u in SUBSETS:
            if len({si[f,m,u]['rows'] for m in SCORE_MEANS})!=1: raise ValueError('unpaired score supports')
    out={'economics':{},'prediction':{},'paired':{},'interaction':{},'selection_performed':False,
         'teacher_use_allowed':False,'high_probability_generalization_established':False,
         'hindsight_diagnostic_not_causal_accuracy':True,'adaptive_causal_names_unchanged':174}
    for regime in ('all','bull','bear','sideways'):
        fs=[f for f in FOLDS if regime=='all' or index[f,'bh']['regime']['trend']==regime]
        econ={};pred={}
        for cid in POLICIES:
            econ[cid]={'quarters':len(fs),'hindsight_only':cid in OLD_ORACLES+NEW_IDS,
                'joint_positive_quarters_both_costs':sum(all(index[f,cid][co]['alpha_ex']>0 and index[f,cid][co]['maxdd_delta']<0
                    for co in ('base','stress_2x')) for f in fs),
                **{co:{k:average([index[f,cid][co][k] for f in fs]) for k in ('alpha_ex','maxdd_delta','turnover','trades')}
                   for co in ('base','stress_2x')}}
        for m in SCORE_MEANS:
            pred[m]={}
            for u in SUBSETS:
                selected=[si[f,m,u] for f in fs];n=sum(x['rows'] for x in selected)
                pred[m][u]={'quarters':len(fs),'nonempty_quarters':sum(x['rows']>0 for x in selected),'defined_rank_quarters':sum(x['return_rank_ic'] is not None for x in selected),'rows':n,
                    **{k:average([x[k] for x in selected]) for k in ('return_mse','return_mae','return_sign_accuracy','return_rank_ic')},
                    'pooled_row_mse':math.fsum(x['return_mse']*x['rows']/n for x in selected if x['rows']) if n else None}
        paired={};interaction={}
        for h,cells in CELLS.items():
            paired[h]={};interaction[h]={}
            for a,b in CONTRASTS:
                name=a+'_minus_'+b
                paired[h][name]={'economics':{rule:{co:{k:average([index[f,cells[a]+'_'+rule][co][k]-index[f,cells[b]+'_'+rule][co][k] for f in fs])
                    for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in ('base','stress_2x')} for rule in RULES},
                    'mse':{u:average([None if si[f,cells[a],u]['return_mse'] is None else
                            si[f,cells[a],u]['return_mse']-si[f,cells[b],u]['return_mse'] for f in fs]) for u in SUBSETS}}
            interaction[h]={'economics':{rule:{co:{k:average([math.fsum(sg*index[f,cells[cell]+'_'+rule][co][k]
                    for cell,sg in (('full',1),('sign',-1),('magnitude',-1),('base',1))) for f in fs])
                    for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in ('base','stress_2x')} for rule in RULES},
                'mse':{u:average([None if si[f,cells['base'],u]['return_mse'] is None else math.fsum(
                    sg*si[f,cells[cell],u]['return_mse'] for cell,sg in (('full',1),('sign',-1),('magnitude',-1),('base',1))) for f in fs]) for u in SUBSETS}}
        out['economics'][regime]=econ;out['prediction'][regime]=pred;out['paired'][regime]=paired;out['interaction'][regime]=interaction
    return out


def run(config_path):
    cfg,fc,bars,forecasts,controls,fit_returns,pre=prepare(config_path);output=Path(cfg['output_dir'])
    if (output/'results.json').exists(): raise ValueError('immutable sign/magnitude diagnostic completed')
    if file_digest(output/'preflight.json')!=cfg['preflight_sha256'] or json.loads((output/'preflight.json').read_text())!=pre:
        raise ValueError('registered preflight changed')
    reg={'config':cfg,'config_sha256':file_digest(config_path),'preflight_sha256':cfg['preflight_sha256'],
         'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'scope':'36 policies:12 old causal+16 old hindsight+8 new hindsight;8 fit-only descriptive thresholds;0model fits'}
    _immutable_json(output/'registration.json',reg)
    execution=fc['execution'];stress={**execution,'one_way_cost':2*execution['one_way_cost'],'borrow_annual':2*execution['borrow_annual']}
    rows=[];scores=[];thresholds=[];direction=[];endpoint_parity=[]
    for f in FOLDS:
        dates=calendar(f-1);ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        window=bars.loc[ix];regime=controls[f,'bh']['regime'];fr=[];fs=[];fd=[];fe=[];bindings={}
        def save_npz(kind,name,a):
            p=output/kind/f'fold{f}_{name}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                old=arrays(p)
                if set(old)!=set(a): raise ValueError('partial artifact schema changed')
                for k,v in a.items(): compare_array(old[k],v,name=str(p)+k,exact=True)
            else: np.savez_compressed(p,**a)
            bindings[str(p)]=file_digest(p);return p
        def evaluate(cid,target,trace=None):
            p=save_npz('targets',cid,{'timestamps':window.index.asi8,'targets':target})
            row={'fold':f,'candidate_id':cid,'regime':regime,'hindsight_only':cid in OLD_ORACLES+NEW_IDS,
                 **{co:metrics(window,target,c) for co,c in [('base',execution),('stress_2x',stress)]},'targets_sha256':bindings[str(p)]}
            if trace is not None:
                p=output/'traces'/f'fold{f}_{cid}.json';_immutable_json(p,trace);bindings[str(p)]=file_digest(p);row['diagnostic_sha256']=bindings[str(p)]
            fr.append(row)
        q,common_subsets=tail_groups(fit_returns[f],forecasts[f,HALVES[0]]['actual'],forecasts[f,HALVES[0]]['score_support'])
        if not math.isfinite(q) or q<0: raise ValueError('invalid fit magnitude threshold')
        threshold={'fold':f,'quantile':.9,'method':'linear','threshold':q,'fit_rows':len(fit_returns[f]),
            'fit_return_sha256':digest(fit_returns[f].tolist()),'hindsight_tail_grouping':True,'used_for_orders':False,
            'subset_rows':{u:int(m.sum()) for u,m in common_subsets.items()},
            'threshold_equal_scored_rows':int((np.abs(forecasts[f,HALVES[0]]['actual'][common_subsets['all'],0])==q).sum()),
            'threshold_is_zero':q==0.}
        p=output/'thresholds'/f'fold{f}_fit_q90.json';_immutable_json(p,threshold);bindings[str(p)]=file_digest(p)
        for cid in CONTROLS:
            old=arrays(PARENT_ROOT/'targets'/f'fold{f}_{cid}.npz')
            compare_array(old['timestamps'],window.index.asi8,name='control calendar',exact=True);evaluate(cid,old['targets'])
            for co in ('base','stress_2x'):compare_tree(fr[-1][co],controls[f,cid][co],name='old control account')
        for h in HALVES:
            original=forecasts[f,h]
            for cell in ('base','full'):
                a=forecasts[f,CELLS[h][cell]]
                for rule in RULES:
                    if rule==RULES[1]:target,trace=fallback_targets(window,a['mu'],a['variance'],execution,
                        inference_mask=a['inference_mask'],risk_aversion=1,cost_multiplier=2)
                    else:target,trace=conditional_targets(window,a['mu'],a['variance'],execution,risk_aversion=1,cost_multiplier=2)
                    cid=CELLS[h][cell]+'_'+rule
                    compare_array(target,arrays(PARENT_ROOT/'targets'/f'fold{f}_{cid}.npz')['targets'],name='endpoint target',exact=True)
                    if cell=='full':
                        trace=mark_hindsight_trace(trace,swap='return',score_support=a['score_support'])
                        old_trace=json.loads((PARENT_ROOT/'traces'/f'fold{f}_{cid}.json').read_text())
                        compare_tree(trace['decision_trace'],old_trace['decision_trace'],name='full endpoint trace')
                    fe.append({'fold':f,'candidate_id':cid,'targets_exact':True,'full_decision_trace_matches':cell=='full'})
            for component in COMPONENTS:
                intervention=substitute_return_component(original['mu'],original['variance'],inference_mask=original['inference_mask'],
                    score_support=original['score_support'],actual=original['actual'],component=component)
                mid=CELLS[h][component];a={**original,'mu':intervention['mu'],'variance':intervention['variance']};forecasts[f,mid]=a
                save_npz('forecasts',mid,a)
                for rule in RULES:
                    if rule==RULES[1]:target,trace=fallback_targets(window,a['mu'],a['variance'],execution,
                        inference_mask=a['inference_mask'],risk_aversion=1,cost_multiplier=2)
                    else:target,trace=conditional_targets(window,a['mu'],a['variance'],execution,risk_aversion=1,cost_multiplier=2)
                    am=action_masks(window.index,window.open.to_numpy(),a['inference_mask'])
                    if rule==RULES[1]:
                        check_action_support(target,am);check_trace_support(target,am,trace)
                    elif np.any(np.isfinite(target)&~am['learned_eligible']):
                        raise ValueError('hold order escaped learned availability')
                    trace=mark_hindsight_trace(trace,swap=component,score_support=a['score_support'])
                    trace['information_intervention']=intervention['metadata'];evaluate(mid+'_'+rule,target,trace)
            observed=original['actual'][:,0];base=original['mu'];score=original['score_support']
            subsets=common_subsets
            for cell,mid in CELLS[h].items():
                for subset,mask in subsets.items():
                    fs.append({'fold':f,'mean_id':mid,'regime':regime,'subset':subset,'hindsight_only':cell!='base',
                        'tail_grouping_uses_future_labels':subset!='all',**score_forecast(original,forecasts[f,mid]['mu'],mask)})
            denom=math.fsum(float(v)**2 for v in observed[score])
            for subset,mask in subsets.items():
                yv=observed[mask];mv=base[mask];n=int(mask.sum())
                opposite=(np.sign(yv)*np.sign(mv))<0
                fd.append({'fold':f,'mean_id':h,'subset':subset,'rows':n,
                    'actual_zero_rows':int((yv==0).sum()),'forecast_zero_rows':int((mv==0).sum()),
                    'nonzero_opposite_sign_rows':int(opposite.sum()),'nonzero_same_sign_rows':int(((np.sign(yv)*np.sign(mv))>0).sum()),
                    'actual_squared_return_share_of_all':math.fsum(float(v)**2 for v in yv)/denom if denom else None,
                    'opposite_sign_actual_squared_return_share_of_subset':math.fsum(float(v)**2 for v in yv[opposite])/math.fsum(float(v)**2 for v in yv) if n and np.any(yv) else None})
        if len(bindings)!=49 or len(fr)!=36 or len(fs)!=24 or len(fd)!=6:raise ValueError('incomplete fold output')
        fold={'registration_sha256':digest(reg),'rows':fr,'scores':fs,'threshold':threshold,'direction_diagnostics':fd,'endpoint_parity':fe,'artifact_sha256':bindings}
        _immutable_json(output/f'fold_{f}.json',fold);rows.extend(fr);scores.extend(fs);direction.extend(fd);thresholds.append(threshold);endpoint_parity.extend(fe)
        print(json.dumps({'event':'fold_complete','fold':f,'policies':36,'scores':24,'artifacts':49,'new_model_fits':0}),flush=True)
    result={'registration_sha256':digest(reg),'rows':rows,'scores':scores,'thresholds':thresholds,'direction_diagnostics':direction,
        'summary':summarize(rows,scores),'endpoint_parity':endpoint_parity,'new_model_fits':0,'fit_distribution_thresholds':8,'new_causal_names':0,
        'adaptive_causal_names_unchanged':174,'new_hindsight_policy_names':8,'selection_performed':False,
        'teacher_use_allowed':False,'additional_test_used':False,'high_probability_generalization_established':False}
    _immutable_json(output/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        cfg,*_,pre=prepare(args.config);path=Path(cfg['output_dir'])/'preflight.json';_immutable_json(path,pre)
        print(json.dumps({'path':str(path),'sha256':file_digest(path),'new_model_fits':0,'new_substitution_or_quantiles_computed':False}))
    else:run(args.config)
