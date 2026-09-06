"""Fixed unit average-loss L2 direction fits on bound original input snapshots."""
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

from .alpha_dd_search import digest,file_digest,load_bars,validate_data_artifact,metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_ablation import mask_digest
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_decisions import action_masks,check_action_support,check_trace_support
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import FOLDS,RULES,compare_array,compare_tree
from .oracle_mean_controls import return_scores
from .oracle_short_mean_fit import _index_digest,_mask_digest,_matrix_digest
from .oracle_direction_scores import direction_scores
from .oracle_direction_decisions import (
    SOURCES as PARENT_SOURCES,POLICIES as CONTROLS,MEANS as OLD_MEANS,
    MODEL_IDS as OLD_MODEL_IDS,CLASSIFIERS as OLD_CLASSIFIERS,GROUPS,WEIGHTINGS,
    PARENTS,PRIOR_IDS,HINDSIGHT_IDS,SEGMENTS,arrays,map_direction,validate_config as validate_parent_config,
)
from .oracle_regularized_direction_fit import fit_regularized_direction_family

PARENT_ROOT=Path('codex_outputs/oracle_direction_decisions_v1')
PARITY_ROOT=Path('codex_outputs/oracle_frozen_procedure_parity_v1')
MODEL_IDS=tuple(m+'_l2unit' for m in OLD_MODEL_IDS)
CLASSIFIERS=OLD_CLASSIFIERS+MODEL_IDS
NEW_MEANS=tuple(m+'_direction' for m in MODEL_IDS)
MEANS=OLD_MEANS+NEW_MEANS
NEW_IDS=tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
POLICIES=CONTROLS+NEW_IDS
PARTS={g+'_'+w+'_l2unit':{'group':g,'weighting':w,'old_classifier':g+'_'+w} for g in GROUPS for w in WEIGHTINGS}
MAPPING={m+'_direction':{**PARTS[m],'classifier_id':m,'parent_mean':PARENTS[PARTS[m]['group']],
    'old_mean':PARTS[m]['old_classifier']+'_direction',
    'prior_mean':PARTS[m]['group']+'_'+PARTS[m]['weighting']+'_prior_direction'} for m in MODEL_IDS}
REFERENCES={m:(MAPPING[m]['parent_mean'],MAPPING[m]['old_mean'],MAPPING[m]['prior_mean']) for m in NEW_MEANS}
SOURCES=PARENT_SOURCES+tuple('unidream/experiments/'+n for n in ('oracle_regularized_direction_fit.py','oracle_regularized_direction_decisions.py'))
FIXED={'schema':'oracle-regularized-direction-decisions-v1','development_folds':list(FOLDS),
    'data_cutoff':'2023-04-16T13:45:00Z','parent_config':'configs/oracle_direction_decisions_20260906.yaml',
    'parent_root':str(PARENT_ROOT),'parent_source_revision':'6ae673fcdfeed29280256450c05eb8905af77ee3',
    'data_path':'/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/checkpoints/alpha_dd_data/spot_15m.parquet',
    'output_dir':'codex_outputs/oracle_regularized_direction_decisions_v1','groups':list(GROUPS),'group_dimensions':[29,31],
    'weightings':list(WEIGHTINGS),'model_ids':list(MODEL_IDS),'classifiers':list(CLASSIFIERS),
    'new_mean_ids':list(NEW_MEANS),'return_score_means':list(MEANS),'control_ids':list(CONTROLS),
    'new_policy_ids':list(NEW_IDS),'mapping':MAPPING,'references':{m:list(v) for m,v in REFERENCES.items()},
    'rules':list(RULES),'segments':list(SEGMENTS),
    'regularization_C':'1.0/float(np.sum(frozen_fit_weights))','normalized_l2_strength_in_real_arithmetic':1.0,
    'fixed_logistic_settings_except_C':'unchanged_Stage17','new_model_fits':32,'new_unique_fit_priors':0,
    'shared_prior_verification_recomputations':16,'new_causal_names':8,'adaptive_prior_causal_names':190,
    'adaptive_total_causal_names':198,'score_classification_records':160,'score_return_records':224,
    'direction_diagnostic_records':64,'fold_artifacts':81,'economic_rows':480,'economic_accounts':960,
    'surrogate_mean':'sign(logit)*abs(own_frozen_half_mu)','zero_logit_direction':0,
    'new_mean_risk_probability_or_weight_calibration_permitted':False,
    'risk_source':'unchanged_saved_technical_scaled','utility_risk_aversion':1,'utility_cost_multiplier':2,
    'execution':{'one_way_cost':.00055,'borrow_annual':.1,'max_step':.08,'deadband':.01},
    'inference_rows':2586,'score_rows':2574,'fallback_rows':332,'missing_current_open_rows':2,
    'selection_permitted':False,'teacher_use_allowed':False,'additional_test_permitted':False,
    'numpy_version':'2.2.6','pandas_version':'2.3.3','sklearn_version':'1.8.0',
    'normalized_gradient_infinity_bound':1e-6,'scalar_logit_atol':1e-12,'scalar_probability_atol':1e-14,
    'partial_retry_policy':'full_replay_verify_existing_no_live_restart'}
EXTRA={'source_bindings','parent_config_sha256','parent_manifest_bindings','preflight_sha256'}


def validate_config(cfg):
    if (set(cfg)!=set(FIXED)|EXTRA or any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in FIXED.items())
            or set(cfg['source_bindings'])!=set(SOURCES)):raise ValueError('unregistered regularized direction family')
    expected={str(PARENT_ROOT/n) for n in ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in FOLDS]}
    if set(cfg['parent_manifest_bindings'])!=expected:raise ValueError('incomplete parent manifest inventory')


def reconstruct_masks(index,pack,cal,forecast):
    """Recover the same full-grid six supports from immutable selected snapshots."""
    n=len(index);masks={k:np.zeros(n,bool) for k in ('fit','predict','scale','interval','inference','score')}
    for name,positions_key,times_key in (('fit','fit_positions','timestamps'),('predict','predict_positions','predict_timestamps')):
        pos=np.asarray(pack[positions_key]);ts=np.asarray(pack[times_key])
        if (pos.ndim!=1 or not len(pos) or pos.dtype.kind not in 'iu' or np.any(pos<0) or np.any(pos>=n)
                or np.any(np.diff(pos)<=0)):raise ValueError('invalid selected snapshot positions')
        compare_array(index.asi8[pos],ts,name='snapshot calendar',exact=True);masks[name][pos]=True
    if np.flatnonzero(masks['fit'])[-1]>=np.flatnonzero(masks['predict'])[0]:raise ValueError('fit/predict chronology changed')
    for a,fields in ((cal,{'scale':'scale_mask','interval':'interval_mask'}),(forecast,{'inference':'inference_mask','score':'score_support'})):
        times=np.asarray(a['timestamps']);pos=index.get_indexer(pd.to_datetime(times,utc=True))
        if np.any(pos<0) or len(np.unique(pos))!=len(pos):raise ValueError('snapshot calendar outside original grid')
        for name,key in fields.items():
            v=np.asarray(a[key])
            if v.dtype!=bool or v.shape!=pos.shape:raise ValueError('snapshot mask type/shape changed')
            masks[name][pos]=v
    if any(np.any(masks[k]&~masks['predict']) for k in ('scale','interval','inference','score')):raise ValueError('saved selected features do not cover original masks')
    return masks


def fit_inputs(index,pack,masks,columns):
    """Only frozen fit returns and fit/predict features enter the new fitter."""
    frames={};y=np.full((len(index),3),np.nan);y[masks['fit'],0]=pack['returns']
    for g in GROUPS:
        a=np.full((len(index),len(columns[g])),np.nan)
        a[masks['fit']]=pack['fit_features_'+g];a[masks['predict']]=pack['predict_features_'+g]
        frames[g]=pd.DataFrame(a,index=index,columns=columns[g])
    return frames,y


def prepare(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());validate_config(cfg)
    if (np.__version__,pd.__version__,sklearn.__version__)!=(cfg['numpy_version'],cfg['pandas_version'],cfg['sklearn_version']):
        raise ValueError('registered runtime changed')
    direct={**cfg['source_bindings'],**cfg['parent_manifest_bindings'],cfg['parent_config']:cfg['parent_config_sha256']}
    for p,sha in direct.items():
        if file_digest(Path(p))!=sha:raise ValueError('bound source changed: '+p)
    pc=yaml.safe_load(Path(cfg['parent_config']).read_text());validate_parent_config(pc)
    parent=json.loads((PARENT_ROOT/'results.json').read_text());reg=json.loads((PARENT_ROOT/'registration.json').read_text());pp=json.loads((PARENT_ROOT/'preflight.json').read_text())
    if (reg['config']!=pc or reg['config_sha256']!=cfg['parent_config_sha256'] or reg['source_revision']!=cfg['parent_source_revision']
            or reg['preflight_sha256']!=file_digest(PARENT_ROOT/'preflight.json') or pc['preflight_sha256']!=reg['preflight_sha256']
            or parent['registration_sha256']!=digest(reg) or pp['source_bindings']!=pc['source_bindings']
            or pp['config_contract_sha256']!=digest({k:v for k,v in pc.items() if k!='preflight_sha256'})
            or pc['data_cutoff']!=cfg['data_cutoff']):raise ValueError('completed Stage17 chain changed')
    for p,sha in pp['direct_source_bindings'].items():
        if p in direct and direct[p]!=sha:raise ValueError('conflicting inherited source binding')
        direct[p]=sha
    bindings=dict(pp['source_artifact_bindings'])
    if len(bindings)!=2120:raise ValueError('incomplete Stage17 ancestry')
    for f in FOLDS:
        fold=json.loads((PARENT_ROOT/f'fold_{f}.json').read_text())
        if fold['registration_sha256']!=digest(reg) or len(fold['artifact_sha256'])!=90:raise ValueError('parent fold changed')
        for key in ('rows','scores','classification_scores'):
            if fold[key]!=[r for r in parent[key] if r['fold']==f]:raise ValueError('parent fold records differ')
        for p,sha in fold['artifact_sha256'].items():
            if p in bindings and bindings[p]!=sha:raise ValueError('conflicting ancestor binding')
            bindings[p]=sha
    if len(bindings)!=2840 or len({str(Path(p).resolve()) for p in bindings})!=2840:raise ValueError('ancestor inventory or aliases changed')
    for p,sha in {**direct,**bindings}.items():
        if file_digest(Path(p))!=sha:raise ValueError('inherited artifact changed: '+p)
    controls={(r['fold'],r['candidate_id']):r for r in parent['rows']}
    if len(controls)!=416 or set(controls)!={(f,c) for f in FOLDS for c in CONTROLS}:raise ValueError('old control inventory changed')
    spot_proof=validate_data_artifact(Path(cfg['data_path']),expected_symbol='BTCUSDT')
    if spot_proof!=pp['spot_data_proof']:raise ValueError('original Spot provenance changed')
    ump=pp['um_data_proof']
    if file_digest(Path(ump['data_path']))!=ump['data_sha256']:raise ValueError('original derivative raw artifact changed')
    bars=load_bars(Path(cfg['data_path']),cutoff=cfg['data_cutoff'])
    packs={};masks={};provenance={};evaluation={};calibration={};support=[]
    def bound_arrays(path):
        if str(path) not in bindings:raise ValueError('unbound saved input')
        return arrays(path)
    for f in FOLDS:
        dates=calendar(f-1);cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        eval_ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        pack_path=PARENT_ROOT/'fit_data'/f'fold{f}_training.npz';pack=bound_arrays(pack_path);packs[f]=pack
        prov_path=PARENT_ROOT/'provenance'/f'fold{f}_fit.json'
        if str(prov_path) not in bindings:raise ValueError('unbound saved fit provenance')
        prov=json.loads(prov_path.read_text());provenance[f]=prov
        if prov!=next(x for x in parent['fit_records'] if x['fold']==f):raise ValueError('parent fit record changed')
        fp=prov['fit_provenance'];old_support=next(x for x in pp['support'] if x['fold']==f)
        if fp['index_sha256']!=_index_digest(bars.index):raise ValueError('full original input grid changed')
        for m in OLD_MEANS:
            if m in PARENTS.values():
                original=bound_arrays(PARITY_ROOT/'forecasts'/f'fold{f}_{m}.npz')
                g=next(g for g in GROUPS if PARENTS[g]==m)
                ca=bound_arrays(PARENT_ROOT/'calibration'/f'fold{f}_{g}_ordinary_direction.npz')
                ca={**ca,'mu':ca['parent_mu']}
            else:
                original=bound_arrays(PARENT_ROOT/'forecasts'/f'fold{f}_{m}.npz')
                ca=bound_arrays(PARENT_ROOT/'calibration'/f'fold{f}_{m}.npz')
            compare_array(original['timestamps'],bars.index[eval_ix].asi8,name='E snapshot times',exact=True)
            compare_array(ca['timestamps'],bars.index[cal_ix].asi8,name='S/I snapshot times',exact=True)
            evaluation[f,m]=original;calibration[f,m]=ca
        ref=evaluation[f,PARENTS[GROUPS[0]]];ca=calibration[f,PARENTS[GROUPS[0]]]
        mm=reconstruct_masks(bars.index,pack,ca,ref);masks[f]=mm
        for k in mm:
            if int(mm[k].sum())!=old_support['counts'][k] or mask_digest(bars.index,mm[k])!=old_support['mask_sha256'][k]:raise ValueError('original six-mask contract changed')
        for k in ('fit','predict'):
            if _mask_digest(mm[k])!=fp['mask_position_sha256'][k]:raise ValueError('fit input positional support changed')
            for g in GROUPS:
                values=pack[k+'_features_'+g]
                if (values.shape!=(int(mm[k].sum()),len(fp['feature_columns'][g])) or not np.isfinite(values).all()
                        or _matrix_digest(values)!=fp[k+'_features_sha256'][g]
                        or digest(values.tolist())!=old_support[k+'_features_sha256'][g]):raise ValueError('frozen selected feature matrix changed')
        if [len(fp['feature_columns'][g]) for g in GROUPS]!=[29,31]:raise ValueError('original feature schema changed')
        for key,hashkey in (('returns','fit_return_sha256'),('binary_labels','fit_binary_labels_sha256')):
            if _matrix_digest(pack[key])!=fp[hashkey]:raise ValueError('frozen fit return/label hash changed')
        for w in WEIGHTINGS:
            if _matrix_digest(pack['weights_'+w])!=fp['sample_weights'][w]['weight_sha256']:raise ValueError('frozen fit weights changed')
        for m in OLD_MEANS:
            e,c=evaluation[f,m],calibration[f,m]
            for key in ('actual','variance','inference_mask','score_support','fit_return_mean'):
                compare_array(e[key],ref[key],name='paired E '+key,exact=True)
            for key in ('actual','scale_mask','interval_mask','classifier_predict_mask','mapped_inference_mask'):
                compare_array(c[key],ca[key],name='paired S/I '+key,exact=True)
        support.append({**old_support,'fit_data_path':str(pack_path),'fit_data_sha256':bindings[str(pack_path)],
            'fit_provenance_path':str(prov_path),'fit_provenance_sha256':bindings[str(prov_path)],
            'features_reconstructed_from_market_bars':False,'new_feature_rows_removed':0})
    pre={'schema':'oracle-regularized-direction-preflight-v1','config_contract_sha256':digest({k:v for k,v in cfg.items() if k!='preflight_sha256'}),
        'source_bindings':cfg['source_bindings'],'direct_source_bindings':direct,'source_artifact_bindings':bindings,'support':support,
        'spot_data_proof':spot_proof,'um_data_proof':ump,'parent_registration_canonical_sha256':digest(reg),
        'new_class_statistics_fits_logits_mapped_predictions_losses_or_orders_computed':False,
        'inputs_from_hash_bound_original_selected_snapshots':True,
        'loader_scope':'Inherited Spot full Parquet decode then strict semantic cutoff; UM file hashed without feature decoding'}
    return cfg,bars,packs,masks,provenance,evaluation,calibration,controls,parent,pre

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
                group,weighting=PARTS[mid]['group'],PARTS[mid]['weighting']
                refs=PRIOR_IDS+(PARTS[mid]['old_classifier'],group+'_'+('magnitude' if weighting=='ordinary' else 'ordinary')+'_l2unit')
                out['classification_paired'][g][seg][mid]={ref:{k:average([None if ci[f,seg,mid][k] is None or ci[f,seg,ref][k] is None else
                    ci[f,seg,mid][k]-ci[f,seg,ref][k] for f in ff]) for k in ck} for ref in refs}
        paired={}
        for m in NEW_MEANS:
            refs=REFERENCES[m]
            paired[m]={ref:{'prediction':{seg:{'mse_difference':average([si[f,seg,m]['return_mse']-si[f,seg,ref]['return_mse'] for f in ff]),
                'improved_quarters':sum(si[f,seg,m]['return_mse']<si[f,seg,ref]['return_mse'] for f in ff),
                'equal_quarters':sum(si[f,seg,m]['return_mse']==si[f,seg,ref]['return_mse'] for f in ff)} for seg in SEGMENTS},
                'economics':{rule:{co:{k:average([ri[f,m+'_'+rule][co][k]-ri[f,ref+'_'+rule][co][k] for f in ff])
                    for k in ('alpha_ex','maxdd_delta','turnover','trades')} for co in costs} for rule in RULES}} for ref in refs}
        out['paired'][g]=paired
    for m in NEW_MEANS:
        mapping=MAPPING[m];mid=mapping['classifier_id'];weighting=mapping['weighting']
        score_keys=('brier','log_loss') if weighting=='ordinary' else ('weighted_brier','weighted_log_loss')
        classifier_gate={seg:all(out['classification_paired'][g][seg][mid][ref][k] is not None
            and out['classification_paired'][g][seg][mid][ref][k]<0
            for g in regimes for ref in ('prior_'+weighting,mapping['old_classifier']) for k in score_keys) for seg in SEGMENTS}
        mapped_gate={seg:all(out['prediction'][g][seg][m]['return_mse']<out['prediction'][g][seg][m][control]
            and all(out['paired'][g][m][ref]['prediction'][seg]['mse_difference']<0 for ref in REFERENCES[m])
            for g in regimes for control in ('zero_return_mse','fit_mean_return_mse')) for seg in SEGMENTS}
        for rule in RULES:
            cid=m+'_'+rule
            out['direction'][cid]={'economic_means_all_strata_both_costs':all(out['economics'][g][cid][co]['alpha_ex']>0
                and out['economics'][g][cid][co]['maxdd_delta']<0 for g in regimes for co in costs),
                'economic_improvement_vs_all_references_all_strata_both_costs':all(
                    out['paired'][g][m][ref]['economics'][rule][co]['alpha_ex']>0
                    and out['paired'][g][m][ref]['economics'][rule][co]['maxdd_delta']<0
                    for g in regimes for ref in REFERENCES[m] for co in costs),
                'matched_probability_losses_vs_C1_and_prior_improved_all_strata':classifier_gate,
                'mapped_mse_vs_zero_fitmean_and_all_references_improved_all_strata':mapped_gate,
                'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out


def run(config_path):
    cfg,bars,packs,masks,old_provenance,evaluation,calibration,controls,parent,pre=prepare(config_path);out=Path(cfg['output_dir'])
    if (out/'results.json').exists():raise ValueError('immutable regularized direction run completed')
    if file_digest(out/'preflight.json')!=cfg['preflight_sha256'] or json.loads((out/'preflight.json').read_text())!=pre:
        raise ValueError('registered regularized preflight changed')
    reg={'config':cfg,'config_sha256':file_digest(config_path),'preflight_sha256':cfg['preflight_sha256'],
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'32 new fixed unit-normalized-L2 fits,0 new priors;8 new causal policies;52 old controls'}
    _immutable_json(out/'registration.json',reg)
    ex=cfg['execution'];stress={**ex,'one_way_cost':2*ex['one_way_cost'],'borrow_annual':2*ex['borrow_annual']}
    old_scores={(r['fold'],r['segment'],r['mean_id']):r for r in parent['scores']}
    old_cs={(r['fold'],r['segment'],r['classifier_id']):r for r in parent['classification_scores']}
    all_rows=[];all_scores=[];all_cs=[];all_fits=[];all_diagnostics=[]
    for f in FOLDS:
        dates=calendar(f-1);mm=masks[f];pack=packs[f];old_prov=old_provenance[f]
        cal_ix=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
        eval_ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']))
        window=bars.loc[eval_ix];regime=controls[f,'bh']['regime'];ref=evaluation[f,PARENTS[GROUPS[0]]]
        groups,y=fit_inputs(bars.index,pack,mm,old_prov['fit_provenance']['feature_columns'])
        fitted=fit_regularized_direction_family(groups,y,fit_mask=mm['fit'],predict_mask=mm['predict'])
        del groups,y
        compare_array(fitted['fit_labels'],pack['binary_labels'],name='unchanged labels',exact=True)
        for w in WEIGHTINGS:
            compare_array(fitted['fit_weights'][w],pack['weights_'+w],name='unchanged weights',exact=True)
            if fitted['fit_priors'][w]!=old_prov['fit_priors'][w]:raise ValueError('shared fit prior changed')
        for k in ('fit_return_mean','fit_abs_return_mean'):
            if fitted[k]!=old_prov[k]:raise ValueError('original fit return statistic changed')
        rows=[];scores=[];cs=[];diagnostics=[];bindings={};model_state={}
        def save(kind,name,value,extension='npz'):
            p=out/kind/f'fold{f}_{name}.{extension}';p.parent.mkdir(parents=True,exist_ok=True)
            if extension=='json':_immutable_json(p,value)
            elif extension=='joblib':
                buf=io.BytesIO();joblib.dump(value,buf,compress=3);data=buf.getvalue()
                if p.exists():
                    if p.read_bytes()!=data:raise ValueError('partial fitted model differs')
                else:p.write_bytes(data)
            elif p.exists():
                old=arrays(p)
                if set(old)!=set(value):raise ValueError('partial array schema differs')
                for k in old:compare_array(old[k],value[k],name=str(p)+k,exact=True)
            else:np.savez_compressed(p,**value)
            bindings[str(p)]=file_digest(p);return p
        for mid in MODEL_IDS:
            old_mid=PARTS[mid]['old_classifier'];weighting=PARTS[mid]['weighting']
            model=fitted['models'][old_mid];old=joblib.load(PARENT_ROOT/'models'/f'fold{f}_{old_mid}.joblib')
            W=float(np.sum(pack['weights_'+weighting]));C=1./W
            if model.named_steps['logisticregression'].get_params()!={**old.named_steps['logisticregression'].get_params(),'C':C}:
                raise ValueError('a logistic setting other than C changed')
            for key in ('mean_','var_','scale_','n_features_in_','n_samples_seen_'):
                compare_array(getattr(model.named_steps['standardscaler'],key),getattr(old.named_steps['standardscaler'],key),name='unchanged scaler '+key,exact=True)
            if model.named_steps['standardscaler'].get_params()!=old.named_steps['standardscaler'].get_params():raise ValueError('scaler settings changed')
            new_beta=model.named_steps['logisticregression'].coef_[0];old_beta=old.named_steps['logisticregression'].coef_[0]
            model_state[mid]={'C':C,'solver_weight_sum':W,'actual_l2_strength':1./(C*W),
                'coefficient_l2_norm':math.sqrt(math.fsum(float(v)**2 for v in new_beta)),
                'old_coefficient_l2_norm':math.sqrt(math.fsum(float(v)**2 for v in old_beta)),
                'intercept':float(model.named_steps['logisticregression'].intercept_[0]),
                'old_intercept':float(old.named_steps['logisticregression'].intercept_[0]),
                'unchanged_scaler_exact':True,'only_C_setting_changed':True,
                'old_model_path':str(PARENT_ROOT/'models'/f'fold{f}_{old_mid}.joblib')}
            save('models',mid,model,'joblib')
        fit_record={'fold':f,'fit_provenance':fitted['provenance'],'model_state':model_state,
            'fit_priors':old_prov['fit_priors'],'fit_prior_logits':old_prov['fit_prior_logits'],
            'fit_source_binding':pre['support'][FOLDS.index(f)],'frozen_fit_data_reused':True,
            'fit_labels_weights_and_priors_match_parent_exactly':True,'new_unique_prior_estimates':0,
            'risk_source':'unchanged_saved_technical_scaled','new_model_fits':4}
        save('provenance','fit',fit_record,'json');all_fits.append(fit_record)
        eval_means={m:evaluation[f,m]['mu'] for m in OLD_MEANS};cal_means={m:calibration[f,m]['mu'] for m in OLD_MEANS}
        for m in NEW_MEANS:
            mp=MAPPING[m];mid=mp['classifier_id'];old_mid=mp['old_classifier']
            original=evaluation[f,mp['old_mean']];ca=calibration[f,mp['old_mean']]
            z=fitted['logits'][old_mid];prob=fitted['probabilities'][old_mid]
            eval_means[m]=map_direction(z[eval_ix],original['parent_mu'],original['inference_mask'])
            cal_means[m]=map_direction(z[cal_ix],ca['parent_mu'],ca['mapped_inference_mask'])
            save('forecasts',m,{**original,'mu':eval_means[m],'logit':z[eval_ix],'probability':prob[eval_ix]})
            save('calibration',m,{**ca,'mu':cal_means[m],'logit':z[cal_ix],'probability':prob[cal_ix]})
            for seg,select,old_array,support in (('interval',cal_ix,ca,ca['mapped_inference_mask']),('evaluation',eval_ix,original,original['inference_mask'])):
                new_z=z[select][support];old_z=old_array['logit'][support];n=len(new_z)
                diagnostics.append({'fold':f,'segment':seg,'classifier_id':mid,'regime':regime,'rows':n,
                    'uses_all_inference_not_score_support':True,
                    'sign_disagreements_vs_C1':int((np.sign(new_z)!=np.sign(old_z)).sum()),
                    'sign_matches_matched_prior':int((np.sign(new_z)==np.sign(old_prov['fit_prior_logits'][mp['weighting']])).sum()),
                    'zero_logit_rows':int((new_z==0).sum()),'old_zero_logit_rows':int((old_z==0).sum()),
                    'mean_abs_logit':math.fsum(abs(float(v))/n for v in new_z),
                    'old_mean_abs_logit':math.fsum(abs(float(v))/n for v in old_z),**model_state[mid]})
        for seg in SEGMENTS:
            actual,support,means,select=(ref['actual'],ref['score_support'],eval_means,eval_ix) if seg=='evaluation' else (
                calibration[f,PARENTS[GROUPS[0]]]['actual'],mm['interval'][cal_ix],cal_means,cal_ix)
            meta={'fold':f,'segment':seg,'regime':regime,'regime_known_at_scored_decisions':seg=='evaluation','regime_reference':'evaluation_quarter_start'}
            for m in MEANS:
                score=return_scores(actual,means[m],support,float(ref['fit_return_mean']))
                if m in OLD_MEANS:compare_tree(score,{k:old_scores[f,seg,m][k] for k in score},name='unchanged old return score')
                scores.append({**meta,'mean_id':m,**score})
            for mid in CLASSIFIERS:
                if mid in MODEL_IDS:z=fitted['logits'][PARTS[mid]['old_classifier']][select]
                else:
                    old_mean=mid+'_direction' if mid in OLD_MODEL_IDS else GROUPS[0]+'_'+mid.removeprefix('prior_')+'_prior_direction'
                    z=(evaluation if seg=='evaluation' else calibration)[f,old_mean]['logit']
                score=direction_scores(actual,z,support)
                if mid in OLD_CLASSIFIERS:compare_tree(score,{k:old_cs[f,seg,mid][k] for k in score},name='unchanged old classifier score')
                cs.append({**meta,'classifier_id':mid,**score})
        for cid in POLICIES:
            trace=None
            if cid in CONTROLS:
                saved=arrays(PARENT_ROOT/'targets'/f'fold{f}_{cid}.npz')
                compare_array(saved['timestamps'],window.index.asi8,name='old target calendar',exact=True);target=saved['targets']
            else:
                rule=RULES[1] if cid.endswith(RULES[1]) else RULES[0];m=cid[:-(len(rule)+1)]
                original=evaluation[f,MAPPING[m]['old_mean']];am=action_masks(window.index,window.open.to_numpy(),original['inference_mask'])
                if rule==RULES[1]:
                    target,trace=fallback_targets(window,eval_means[m],original['variance'],ex,inference_mask=original['inference_mask'],risk_aversion=1,cost_multiplier=2)
                    check_action_support(target,am);check_trace_support(target,am,trace)
                else:
                    target,trace=conditional_targets(window,eval_means[m],original['variance'],ex,risk_aversion=1,cost_multiplier=2)
                    if np.any(np.isfinite(target)&~am['learned_eligible']):raise ValueError('hold order escaped inference support')
                trace['direction_mapping']={**MAPPING[m],'C_schedule':cfg['regularization_C'],'surrogate_mean':cfg['surrogate_mean'],'future_labels_used_for_orders':False}
            p=save('targets',cid,{'timestamps':window.index.asi8,'targets':target})
            row={'fold':f,'candidate_id':cid,'regime':regime,'hindsight_only':cid in HINDSIGHT_IDS,'targets_sha256':bindings[str(p)],
                **{co:metrics(window,target,c) for co,c in [('base',ex),('stress_2x',stress)]}}
            if cid in CONTROLS:
                for co in ('base','stress_2x'):compare_tree(row[co],controls[f,cid][co],name='unchanged old account')
            else:
                p=save('traces',cid,trace,'json');row['trace_sha256']=bindings[str(p)]
            rows.append(row)
        if len(bindings)!=81 or len(rows)!=60 or len(scores)!=28 or len(cs)!=20 or len(diagnostics)!=8:raise ValueError('incomplete regularized direction fold')
        _immutable_json(out/f'fold_{f}.json',{'registration_sha256':digest(reg),'rows':rows,'scores':scores,
            'classification_scores':cs,'direction_diagnostics':diagnostics,'artifact_sha256':bindings})
        all_rows.extend(rows);all_scores.extend(scores);all_cs.extend(cs);all_diagnostics.extend(diagnostics)
        print(json.dumps({'event':'fold_complete','fold':f,'new_model_fits':4,'new_unique_priors':0,
            'policies':60,'return_scores':28,'classification_scores':20,'direction_diagnostics':8,'artifacts':81}),flush=True)
    result={'registration_sha256':digest(reg),'rows':all_rows,'scores':all_scores,'classification_scores':all_cs,'fit_records':all_fits,
        'direction_diagnostics':all_diagnostics,'summary':summarize(all_rows,all_scores,all_cs),'new_model_fits':32,'new_unique_priors':0,
        'new_causal_policy_names':8,'total_adaptively_explored_causal_names':198,
        'additional_test_used_for_modeling_or_scoring':False,'selection_performed':False,'teacher_use_allowed':False,
        'high_probability_generalization_established':False,'risk_model_or_calibration_fits':0}
    _immutable_json(out/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        cfg,*_,pre=prepare(args.config);path=Path(cfg['output_dir'])/'preflight.json';_immutable_json(path,pre)
        print(json.dumps({'path':str(path),'sha256':file_digest(path),'new_real_model_statistics_fits_or_orders_computed':False}))
    else:run(args.config)
