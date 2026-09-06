"""Independent Stage19 saved-score/summary audit; no fits or policy rollouts.

Prepared before freeze. Real artifacts are opened only with the explicit
--execute-saved-audit switch, after the root authorizes a completed run.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT = ROOT / 'codex_outputs/oracle_soft_direction_decisions_v1'
PARENT = ROOT / 'codex_outputs/oracle_regularized_direction_decisions_v1'
DIRECTION = ROOT / 'codex_outputs/oracle_direction_decisions_v1'
PARITY = ROOT / 'codex_outputs/oracle_frozen_procedure_parity_v1'
REPORT = Path('/tmp/oracle_soft_direction_score_audit_20260906.json')
FS = tuple(range(5, 13))
GROUPS = ('technical', 'perp_delay0')
WEIGHTINGS = ('ordinary', 'magnitude')
OLD_MODELS = tuple(g+'_'+w for g in GROUPS for w in WEIGHTINGS)
REG_MODELS = tuple(m+'_l2unit' for m in OLD_MODELS)
CLASSIFIERS = OLD_MODELS + tuple('prior_'+w for w in WEIGHTINGS) + REG_MODELS
HALVES = tuple(g+'_half' for g in GROUPS)
DIRECTION_MEANS = tuple(m+'_direction' for m in OLD_MODELS) + tuple(
    g+'_'+w+'_prior_direction' for g in GROUPS for w in WEIGHTINGS)
REG_MEANS = tuple(m+'_direction' for m in REG_MODELS)
OLD_MEANS = HALVES + DIRECTION_MEANS + REG_MEANS
PROBABILITY_IDS = tuple(g+'_magnitude'+suffix for g in GROUPS for suffix in ('','_l2unit'))
SOFT_MEANS = tuple(m+'_soft' for m in PROBABILITY_IDS)
CONSTANT_KINDS = ('mapped_prior','fit_mean','zero')
CONSTANT_MEANS = tuple(g+'_soft_'+kind for g in GROUPS for kind in CONSTANT_KINDS)
NEW_MEANS = SOFT_MEANS + CONSTANT_MEANS
MEANS = OLD_MEANS + NEW_MEANS
MAPPING = {mid+'_soft': {'group':g,'kind':'soft','role':'learned_mapping',
    'source_classifier':mid,'source_mean':mid+'_direction','parent_mean':g+'_half'}
    for g in GROUPS for mid in PROBABILITY_IDS if mid.startswith(g+'_')}
MAPPING.update({g+'_soft_'+kind:{'group':g,'kind':kind,'role':'constant_control',
    'source_classifier':g+'_magnitude','source_mean':g+'_magnitude_direction','parent_mean':g+'_half'}
    for g in GROUPS for kind in CONSTANT_KINDS})
REFERENCES = {m:(MAPPING[m]['source_mean'],MAPPING[m]['parent_mean'],
    *(MAPPING[m]['group']+'_soft_'+kind for kind in CONSTANT_KINDS)) for m in SOFT_MEANS}
REG_MAPPING = {g+'_'+w+'_l2unit_direction': {'group':g, 'weighting':w,
    'classifier_id':g+'_'+w+'_l2unit', 'old_classifier':g+'_'+w,
    'parent_mean':g+'_half', 'old_mean':g+'_'+w+'_direction',
    'prior_mean':g+'_'+w+'_prior_direction'} for g in GROUPS for w in WEIGHTINGS}
REG_REFERENCES = {m:tuple(REG_MAPPING[m][k] for k in ('parent_mean','old_mean','prior_mean')) for m in REG_MEANS}
RULES = ('utility_risk1', 'utility_risk1_fallback_bh')
COSTS = ('base', 'stress_2x')
SEGMENTS = ('interval', 'evaluation')
STRATA = ('all', 'bull', 'bear', 'sideways')
EK = ('alpha_ex', 'maxdd_delta', 'turnover', 'trades')
RK = ('return_mse', 'return_mae', 'return_sign_accuracy', 'zero_return_mse',
      'fit_mean_return_mse', 'return_rank_ic')
CK = ('log_loss', 'brier', 'binary_accuracy', 'signed_return_mean',
      'weighted_log_loss', 'weighted_brier', 'weighted_binary_accuracy',
      'absolute_return_sum', 'absolute_return_mean')
BINDINGS, MAXDIFF, LOCATIONS = {}, {}, {}
CATEGORIES = Counter()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bind(path, expected=None, category='direct'):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    value = sha(path)
    assert expected is None or value == expected, (str(path), 'hash changed')
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if key not in BINDINGS:
        BINDINGS[key] = value
        CATEGORIES[category] += 1
    else:
        assert BINDINGS[key] == value
    return path


def npz(path):
    path = Path(path)
    assert str(path.relative_to(ROOT)) in BINDINGS, (path, 'unbound artifact')
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k] for k in value.files}


def matrix_sha(values):
    a = np.asarray(values, dtype='<f8', order='C')
    return hashlib.sha256(np.asarray([a.ndim, *a.shape], dtype='<i8').tobytes()
                          + a.tobytes(order='C')).hexdigest()


def D(x):
    return Decimal(str(x))


def mean(values):
    if not values or any(x is None for x in values):
        return None
    return sum((D(x) for x in values), Decimal(0))/Decimal(len(values))


def plain(value):
    if isinstance(value, Decimal): return float(value)
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def compare(left, right, path, category, tolerance='1e-12'):
    if isinstance(left, dict):
        assert set(left) == set(right), (path, 'schema', set(left)^set(right))
        for key in left: compare(left[key], right[key], path+'/'+str(key), category, tolerance)
    elif isinstance(left, (tuple,list)):
        assert len(left) == len(right), (path, 'length')
        for i,(a,b) in enumerate(zip(left,right)): compare(a,b,path+'/'+str(i),category,tolerance)
    elif isinstance(left, (int,float,Decimal)) and not isinstance(left,bool):
        assert right is not None and not isinstance(right,bool), path
        assert math.isfinite(float(left)) and math.isfinite(float(right)), path
        difference = abs(D(left)-D(right))
        if category not in MAXDIFF or difference > MAXDIFF[category]:
            MAXDIFF[category], LOCATIONS[category] = difference, path
        assert difference <= D(tolerance), (path, float(difference), tolerance)
    else:
        assert left == right, (path,left,right)


def exact_array(a, b, path):
    assert a.shape == b.shape and a.dtype == b.dtype, (path,'shape/dtype')
    assert np.array_equal(a,b,equal_nan=True), (path,'array changed')


def scalar_sigmoid(z):
    z = float(z)
    if z >= 0: return 1/(1+math.exp(-z))
    e = math.exp(z)
    return e/(1+e)


def classification_scores(y, logits):
    ys, zs = [float(v) for v in y], [float(v) for v in logits]
    n = len(ys)
    assert n > 0 and len(zs)==n and all(math.isfinite(v) for v in ys+zs)
    label = [int(v>0) for v in ys]
    probability = [scalar_sigmoid(z) for z in zs]
    # Independent class-conditional softplus arithmetic avoids the canonical
    # scorer's logaddexp(0,z)-label*z cancellation route.
    ll = [math.log1p(math.exp(-abs(z))) + (max(-z,0.) if b else max(z,0.))
          for z,b in zip(zs,label)]
    bs = [(p-b)**2 for p,b in zip(probability,label)]
    correct = [float((z>0)==bool(b)) for z,b in zip(zs,label)]
    absolute = [abs(y) for y in ys]
    total = math.fsum(absolute)
    weighted = lambda values: math.fsum(a/total*v for a,v in zip(absolute,values)) if total else None
    return {'rows':n, 'zero_actual_rows':sum(y==0 for y in ys),
        'zero_logit_rows':sum(z==0 for z in zs),
        'log_loss':math.fsum(ll)/n,'brier':math.fsum(bs)/n,
        'binary_accuracy':math.fsum(correct)/n,
        'signed_return_mean':math.fsum((int(z>0)-int(z<0))*y for z,y in zip(zs,ys))/n,
        'weighted_log_loss':weighted(ll),'weighted_brier':weighted(bs),
        'weighted_binary_accuracy':weighted(correct),
        'absolute_return_sum':total,'absolute_return_mean':total/n}


def ranks(values):
    # Ordinal grouping with explicit average ties, independent of scipy.rankdata.
    order = sorted(range(len(values)),key=lambda i:float(values[i]))
    result = np.empty(len(order),float)
    i=0
    while i<len(order):
        end=i+1
        while end<len(order) and values[order[end]]==values[order[i]]: end+=1
        rank=(i+1+end)/2
        for position in order[i:end]:result[position]=rank
        i=end
    return result


def return_scores(y, mu, fit_mean):
    n=len(y)
    assert n>0 and len(mu)==n and np.isfinite(y).all() and np.isfinite(mu).all()
    residual=[float(a)-float(b) for a,b in zip(y,mu)]
    ry,rp=ranks(y),ranks(mu); center=(n+1)/2
    normy=math.fsum((float(v)-center)**2 for v in ry)
    normp=math.fsum((float(v)-center)**2 for v in rp)
    rank=(math.fsum((float(a)-center)*(float(b)-center) for a,b in zip(ry,rp))
          /math.sqrt(normy*normp)) if normy and normp else None
    return {'rows':n,'return_mse':math.fsum(v*v for v in residual)/n,
        'return_mae':math.fsum(abs(v) for v in residual)/n,
        'return_sign_accuracy':sum((a>0)==(b>0) for a,b in zip(y,mu))/n,
        'zero_return_mse':math.fsum(float(v)**2 for v in y)/n,
        'fit_mean_return_mse':math.fsum((float(v)-fit_mean)**2 for v in y)/n,
        'return_rank_ic':rank}


def build_inherited_summary(rows, scores, class_scores, controls, hindsight):
    MEANS=OLD_MEANS;NEW_MEANS=REG_MEANS;MODELS=REG_MODELS;MAPPING=REG_MAPPING;REFERENCES=REG_REFERENCES
    ri={(r['fold'],r['candidate_id']):r for r in rows}
    si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
    ci={(r['fold'],r['segment'],r['classifier_id']):r for r in class_scores}
    ids=controls+tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
    mapping=MAPPING
    regime={f:ri[f,'bh']['regime'] for f in FS}
    assert Counter(r['trend'] for r in regime.values())=={'bull':2,'bear':4,'sideways':2}
    ck=CK[:7]
    out={'economics':{},'prediction':{},'classification':{},'paired':{},'classification_paired':{},'direction':{},
        'regime_counts':{'bull':2,'bear':4,'sideways':2},
        'interval_regime_strata_are_retrospective_evaluation_groupings':True,
        'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    for g in STRATA:
        fs=[f for f in FS if g=='all' or regime[f]['trend']==g]
        out['economics'][g]={cid:{'quarters':len(fs),'hindsight_only':cid in hindsight,
            'joint_positive_quarters_both_costs':sum(all(ri[f,cid][c]['alpha_ex']>0 and ri[f,cid][c]['maxdd_delta']<0 for c in COSTS) for f in fs),
            **{c:{k:mean([ri[f,cid][c][k] for f in fs]) for k in EK} for c in COSTS}} for cid in ids}
        out['prediction'][g]={};out['classification'][g]={};out['classification_paired'][g]={};out['paired'][g]={}
        for seg in SEGMENTS:
            out['prediction'][g][seg]={m:{'quarters':len(fs),'rows':sum(si[f,seg,m]['rows'] for f in fs),
                **{k:mean([si[f,seg,m][k] for f in fs]) for k in RK},
                'pooled_row_mse':sum(D(si[f,seg,m]['return_mse'])*si[f,seg,m]['rows'] for f in fs)
                    /Decimal(sum(si[f,seg,m]['rows'] for f in fs))} for m in MEANS}
            out['classification'][g][seg]={m:{'quarters':len(fs),'rows':sum(ci[f,seg,m]['rows'] for f in fs),
                **{k:mean([ci[f,seg,m][k] for f in fs]) for k in ck},
                'zero_actual_rows':sum(ci[f,seg,m]['zero_actual_rows'] for f in fs),
                'zero_logit_rows':sum(ci[f,seg,m]['zero_logit_rows'] for f in fs),
                'absolute_return_sum':sum(D(ci[f,seg,m]['absolute_return_sum']) for f in fs)} for m in CLASSIFIERS}
            out['classification_paired'][g][seg]={}
            for group in GROUPS:
                for weighting in WEIGHTINGS:
                    mid=group+'_'+weighting+'_l2unit'
                    refs=('prior_ordinary','prior_magnitude',group+'_'+weighting,group+'_'+('magnitude' if weighting=='ordinary' else 'ordinary')+'_l2unit')
                    out['classification_paired'][g][seg][mid]={ref:{k:mean([
                        None if ci[f,seg,mid][k] is None or ci[f,seg,ref][k] is None else
                        D(ci[f,seg,mid][k])-D(ci[f,seg,ref][k]) for f in fs]) for k in ck} for ref in refs}
        for m in NEW_MEANS:
            refs=REFERENCES[m]
            out['paired'][g][m]={ref:{'prediction':{seg:{
                'mse_difference':mean([D(si[f,seg,m]['return_mse'])-D(si[f,seg,ref]['return_mse']) for f in fs]),
                'improved_quarters':sum(si[f,seg,m]['return_mse']<si[f,seg,ref]['return_mse'] for f in fs),
                'equal_quarters':sum(si[f,seg,m]['return_mse']==si[f,seg,ref]['return_mse'] for f in fs)} for seg in SEGMENTS},
                'economics':{r:{c:{k:mean([D(ri[f,m+'_'+r][c][k])-D(ri[f,ref+'_'+r][c][k]) for f in fs])
                    for k in EK} for c in COSTS} for r in RULES}} for ref in refs}
    for m in NEW_MEANS:
        mp=mapping[m];mid=mp['classifier_id']
        sk=('brier','log_loss') if mp['weighting']=='ordinary' else ('weighted_brier','weighted_log_loss')
        cg={seg:all(out['classification_paired'][g][seg][mid][ref][k] is not None
            and out['classification_paired'][g][seg][mid][ref][k]<0
            for g in STRATA for ref in ('prior_'+mp['weighting'],mp['old_classifier']) for k in sk) for seg in SEGMENTS}
        mg={seg:all(out['prediction'][g][seg][m]['return_mse']<out['prediction'][g][seg][m][ref]
            and all(out['paired'][g][m][p]['prediction'][seg]['mse_difference']<0 for p in
                REFERENCES[m])
            for g in STRATA for ref in ('zero_return_mse','fit_mean_return_mse')) for seg in SEGMENTS}
        for r in RULES:
            cid=m+'_'+r
            out['direction'][cid]={'economic_means_all_strata_both_costs':all(
                out['economics'][g][cid][c]['alpha_ex']>0 and out['economics'][g][cid][c]['maxdd_delta']<0 for g in STRATA for c in COSTS),
                'economic_improvement_vs_all_references_all_strata_both_costs':all(
                    out['paired'][g][m][ref]['economics'][r][c]['alpha_ex']>0
                    and out['paired'][g][m][ref]['economics'][r][c]['maxdd_delta']<0
                    for g in STRATA for ref in REFERENCES[m] for c in COSTS),
                'matched_probability_losses_vs_C1_and_prior_improved_all_strata':cg,
                'mapped_mse_vs_zero_fitmean_and_all_references_improved_all_strata':mg,
                'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out


def build_summary(rows,scores,class_scores,controls,hindsight,old_summary):
    ri={(r['fold'],r['candidate_id']):r for r in rows}
    si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
    ids=controls+tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
    regime={f:ri[f,'bh']['regime'] for f in FS}
    assert Counter(r['trend'] for r in regime.values())=={'bull':2,'bear':4,'sideways':2}
    out={'economics':{},'prediction':{},'paired':{},'soft':{},'classification':old_summary['classification'],
        'inherited_classification_paired':old_summary['classification_paired'],
        'inherited_Stage18_direction_flags':old_summary['direction'],
        'regime_counts':{'bull':2,'bear':4,'sideways':2},
        'interval_regime_strata_are_retrospective_evaluation_groupings':True,
        'probability_predictions_and_scores_unchanged':True,'new_probability_accuracy_improvement':False,
        'selection_performed':False,'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    for g in STRATA:
        fs=[f for f in FS if g=='all' or regime[f]['trend']==g]
        out['economics'][g]={cid:{'quarters':len(fs),'hindsight_only':cid in hindsight,
            'joint_positive_quarters_both_costs':sum(all(ri[f,cid][c]['alpha_ex']>0 and ri[f,cid][c]['maxdd_delta']<0 for c in COSTS) for f in fs),
            **{c:{k:mean([ri[f,cid][c][k] for f in fs]) for k in EK} for c in COSTS}} for cid in ids}
        out['prediction'][g]={seg:{m:{'quarters':len(fs),'rows':sum(si[f,seg,m]['rows'] for f in fs),
            **{k:mean([si[f,seg,m][k] for f in fs]) for k in RK},
            'pooled_row_mse':sum(D(si[f,seg,m]['return_mse'])*si[f,seg,m]['rows'] for f in fs)
                /Decimal(sum(si[f,seg,m]['rows'] for f in fs))} for m in MEANS} for seg in SEGMENTS}
        out['paired'][g]={m:{ref:{'prediction':{seg:{
            'mse_difference':mean([D(si[f,seg,m]['return_mse'])-D(si[f,seg,ref]['return_mse']) for f in fs]),
            'improved_quarters':sum(si[f,seg,m]['return_mse']<si[f,seg,ref]['return_mse'] for f in fs),
            'equal_quarters':sum(si[f,seg,m]['return_mse']==si[f,seg,ref]['return_mse'] for f in fs)} for seg in SEGMENTS},
            'economics':{rule:{cost:{k:mean([D(ri[f,m+'_'+rule][cost][k])-D(ri[f,ref+'_'+rule][cost][k]) for f in fs])
                for k in EK} for cost in COSTS} for rule in RULES}}
            for ref in REFERENCES[m]} for m in SOFT_MEANS}
    for m in SOFT_MEANS:
        mid=MAPPING[m]['source_classifier']
        source_gate={seg:all(out['classification'][g][seg][mid][k] is not None
            and out['classification'][g][seg]['prior_magnitude'][k] is not None
            and out['classification'][g][seg][mid][k]<out['classification'][g][seg]['prior_magnitude'][k]
            for g in STRATA for k in ('weighted_brier','weighted_log_loss')) for seg in SEGMENTS}
        mse_gate={seg:all(out['paired'][g][m][ref]['prediction'][seg]['mse_difference']<0
            for g in STRATA for ref in REFERENCES[m]) for seg in SEGMENTS}
        for rule in RULES:
            cid=m+'_'+rule
            out['soft'][cid]={'economic_means_all_strata_both_costs':all(out['economics'][g][cid][cost]['alpha_ex']>0
                and out['economics'][g][cid][cost]['maxdd_delta']<0 for g in STRATA for cost in COSTS),
                'economic_improvement_vs_all_five_references_all_strata_both_costs':all(
                    out['paired'][g][m][ref]['economics'][rule][cost]['alpha_ex']>0
                    and out['paired'][g][m][ref]['economics'][rule][cost]['maxdd_delta']<0
                    for g in STRATA for ref in REFERENCES[m] for cost in COSTS),
                'mapped_mse_vs_all_five_references_improved_all_strata':mse_gate,
                'inherited_source_weighted_losses_below_prior_all_strata':source_gate,
                'new_probability_accuracy_improvement':False,'high_probability_generalization_established':False,
                'regime_count_gate_pass':False}
    return out


def audit(expected_results_sha):
    with localcontext() as context:
        context.prec=60
        result=read(bind(OUT/'results.json',expected_results_sha));reg=read(bind(OUT/'registration.json'))
        cfg=reg['config'];config_path=ROOT/'configs/oracle_soft_direction_decisions_20260906.yaml'
        bind(config_path,reg['config_sha256'],'config')
        pre=read(bind(OUT/'preflight.json',cfg['preflight_sha256']))
        assert result['registration_sha256']==canonical(reg)
        assert reg['preflight_sha256']==cfg['preflight_sha256']
        assert pre['source_bindings']==cfg['source_bindings'] and len(cfg['source_bindings'])==33
        assert pre['config_contract_sha256']==canonical({k:v for k,v in cfg.items() if k!='preflight_sha256'})
        for p,h in cfg['source_bindings'].items():bind(p,h,'registered_source')
        for p,h in pre['direct_source_bindings'].items():bind(p,h,'preflight_direct')
        for p,h in cfg['parent_manifest_bindings'].items():bind(p,h,'parent_manifest')
        assert len(pre['source_artifact_bindings'])==3488
        for p,h in pre['source_artifact_bindings'].items():bind(p,h,'ancestor_artifact')
        parent=read(PARENT/'results.json');preg=read(PARENT/'registration.json')
        assert preg['source_revision']==cfg['parent_source_revision']=='5a82c270c64a342ab7e9df8105b7d23d1336d876'
        assert parent['registration_sha256']==canonical(preg)==pre['parent_registration_canonical_sha256']
        controls=tuple(cfg['control_ids'])
        assert len(controls)==60 and set(controls)=={r['candidate_id'] for r in parent['rows']}
        ids=controls+tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
        learned=tuple(m+'_'+r for m in SOFT_MEANS for r in RULES)
        hindsight={r['candidate_id'] for r in parent['rows'] if r['hindsight_only']}
        assert len(hindsight)==24 and len(ids)==80
        assert cfg['mapping']==MAPPING and cfg['references']=={m:list(v) for m,v in REFERENCES.items()}
        assert tuple(cfg['return_score_means'])==MEANS and tuple(cfg['classifiers'])==CLASSIFIERS
        assert tuple(cfg['new_policy_ids'])==ids[len(controls):] and tuple(cfg['learned_policy_ids'])==learned
        assert cfg['surrogate_mean']=='saved_fit_abs_return_mean*(2.0*saved_probability-1.0)'
        rows,scores,cs=result['rows'],result['scores'],result['classification_scores']
        ri={(r['fold'],r['candidate_id']):r for r in rows}
        si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
        ci={(r['fold'],r['segment'],r['classifier_id']):r for r in cs}
        mr={r['fold']:r for r in result['mapping_records']}
        ds=result['mapping_diagnostics'];di={(r['fold'],r['segment'],r['mean_id']):r for r in ds}
        assert len(rows)==len(ri)==640 and set(ri)=={(f,p) for f in FS for p in ids}
        assert len(scores)==len(si)==384 and set(si)=={(f,s,m) for f in FS for s in SEGMENTS for m in MEANS}
        assert len(cs)==len(ci)==160 and set(ci)=={(f,s,c) for f in FS for s in SEGMENTS for c in CLASSIFIERS}
        assert len(ds)==len(di)==64 and set(di)=={(f,s,m) for f in FS for s in SEGMENTS for m in SOFT_MEANS}
        assert len(result['mapping_records'])==len(mr)==8 and set(mr)==set(FS)
        for k,v in {'new_model_fits':0,'new_unique_priors':0,'new_causal_policy_names':20,
            'new_learned_policy_names':8,'new_constant_control_policy_names':12,
            'total_adaptively_explored_causal_names':218,'risk_model_or_calibration_fits':0}.items():assert result[k]==v
        for k in ('additional_test_used_for_modeling_or_scoring','selection_performed','teacher_use_allowed',
                  'high_probability_generalization_established','new_probability_accuracy_improvement'):assert result[k] is False
        assert result['probability_predictions_and_scores_unchanged'] is True
        artifacts={}
        for f in FS:
            fold=read(bind(OUT/f'fold_{f}.json',category='fold_manifest'))
            assert fold['registration_sha256']==result['registration_sha256']
            for key in ('rows','scores','classification_scores','mapping_diagnostics'):
                assert fold[key]==[r for r in result[key] if r['fold']==f]
            expected={str((OUT/'targets'/f'fold{f}_{p}.npz').relative_to(ROOT)) for p in ids}
            expected|={str((OUT/'traces'/f'fold{f}_{p}.json').relative_to(ROOT)) for p in ids[len(controls):]}
            expected|={str((OUT/k/f'fold{f}_{m}.npz').relative_to(ROOT)) for k in ('forecasts','calibration') for m in NEW_MEANS}
            expected|={str((OUT/'provenance'/f'fold{f}_mapping.json').relative_to(ROOT))}
            assert len(expected)==121 and set(fold['artifact_sha256'])==expected
            for p,h in fold['artifact_sha256'].items():bind(p,h,'new_artifact');artifacts[p]=h
        assert len(artifacts)==968
        regime={f:ri[f,'bh']['regime'] for f in FS}
        assert Counter(r['trend'] for r in regime.values())=={'bull':2,'bear':4,'sideways':2}
        for r in rows:
            assert r['regime']==regime[r['fold']] and r['hindsight_only']==(r['candidate_id'] in hindsight)
            for cost in COSTS:
                assert all(type(v) in (int,float) and math.isfinite(v) for v in r[cost].values())
        for r in scores+cs:
            assert r['regime']==regime[r['fold']]
            assert r['regime_known_at_scored_decisions']==(r['segment']=='evaluation')
            assert r['regime_reference']=='evaluation_quarter_start'
        for r in ds:
            assert r['regime']==regime[r['fold']] and r['source_classifier']==MAPPING[r['mean_id']]['source_classifier']
            assert r['uses_all_inference_not_score_support'] is True
        old_ri={(r['fold'],r['candidate_id']):r for r in parent['rows']}
        old_si={(r['fold'],r['segment'],r['mean_id']):r for r in parent['scores']}
        old_ci={(r['fold'],r['segment'],r['classifier_id']):r for r in parent['classification_scores']}
        for key,r in old_ri.items():assert ri[key]==r,(key,'complete old economic row changed')
        for key,r in old_si.items():assert si[key]==r,(key,'complete old return score changed')
        for key,r in old_ci.items():assert ci[key]==r,(key,'complete probability score changed')
        old_summary=build_inherited_summary([ri[k] for k in old_ri],[si[k] for k in old_si],cs,
            tuple(preg['config']['control_ids']),hindsight)
        for k,v in old_summary.items():compare(v,parent['summary'][k],k,'inherited_summary_'+k)
        summary=build_summary(rows,scores,cs,controls,hindsight,old_summary)
        assert set(summary)==set(result['summary'])
        for k,v in summary.items():compare(v,result['summary'][k],k,'summary_'+k)
        for key,parentkey in [('classification','classification'),('inherited_classification_paired','classification_paired'),
                              ('inherited_Stage18_direction_flags','direction')]:
            assert result['summary'][key]==parent['summary'][parentkey],(key,'inherited exact summary changed')
        scalar_return_count=scalar_class_count=control_count=old_return_count=old_class_count=0
        score_checks=[];constant_checks=[];support_checks=[];probability_checks=[]
        for f in FS:
            record=mr[f];assert record==read(OUT/'provenance'/f'fold{f}_mapping.json')
            support=next(p for p in pre['support'] if p['fold']==f)
            assert record['frozen_input_bindings']==support['saved_probability_inputs']
            assert record['saved_T_scalars']==support['saved_mapping_scalars']
            assert record['new_fits']==record['new_unique_priors']==0
            assert record['probability_arrays_unchanged'] is True
            assert record['caller_saved_input_provenance_and_calendar_verified'] is True
            assert record['mapping_formula']==cfg['surrogate_mean'] and set(record['mean_records'])==set(NEW_MEANS)
            oldfit=read(DIRECTION/'provenance'/f'fold{f}_fit.json')
            scalars=record['saved_T_scalars'];fit_mean=float(scalars['fit_return_mean'])
            assert scalars['fit_return_mean']==oldfit['fit_return_mean']
            assert scalars['fit_abs_return_mean']==oldfit['fit_abs_return_mean']
            assert scalars['fit_statistical_magnitude_prior']==oldfit['fit_priors']['magnitude']
            assert math.isfinite(scalars['fit_abs_return_mean']) and scalars['fit_abs_return_mean']>0
            saved={};source_classes={}
            def old_stream(m,seg):
                kind='calibration' if seg=='interval' else 'forecasts'
                if m in HALVES:
                    if seg=='evaluation':return npz(PARITY/kind/f'fold{f}_{m}.npz')
                    a=npz(DIRECTION/kind/f'fold{f}_{m.removesuffix("_half")}_ordinary_direction.npz')
                    return {**a,'mu':a['parent_mu']}
                root=PARENT if m in REG_MEANS else DIRECTION
                return npz(root/kind/f'fold{f}_{m}.npz')
            for m in MEANS:
                new=m in NEW_MEANS
                for seg in SEGMENTS:
                    kind='calibration' if seg=='interval' else 'forecasts'
                    a=npz(OUT/kind/f'fold{f}_{m}.npz') if new else old_stream(m,seg)
                    saved[seg,m]=a
                    mask=a['mapped_inference_mask'] if seg=='interval' else a['inference_mask']
                    sm=a['interval_mask'] if seg=='interval' else a['score_support']
                    assert mask.dtype==sm.dtype==bool and mask.shape==sm.shape
                    assert np.all(sm<=mask) and np.isfinite(a['mu'][sm]).all()
                    assert np.isfinite(a['actual'][sm,0]).all() and float(a['fit_return_mean'])==fit_mean
                    if new:
                        mp=MAPPING[m];old=old_stream(mp['source_mean'],seg)
                        assert set(a)==set(old)
                        for key in old:
                            if key!='mu':exact_array(a[key],old[key],f'{f}/{m}/{seg}/unchanged_source/{key}')
                        assert np.array_equal(np.isfinite(a['mu']),mask)
                        assert a['mu'].dtype==np.float64
                        if seg=='interval':assert not np.any(mask & a['scale_mask'])
                        assert record['mean_records'][m]['source_probability_and_logit_fields_are_preserved_source_evidence'] is True
                        for k,v in mp.items():assert record['mean_records'][m][k]==v
                        if mp['kind']!='soft':
                            selected=a['mu'][mask]
                            assert len(selected)>0 and np.all(selected==selected[0]),(f,m,seg,'not constant')
                            if mp['kind']=='zero':assert np.all(selected==0.)
                            if mp['kind']=='fit_mean':assert np.all(selected==fit_mean)
                            constant_checks.append({'fold':f,'mean_id':m,'segment':seg,
                                'kind':mp['kind'],'inference_rows':len(selected),'constant_value':float(selected[0]),
                                'constant_rank_is_null':si[f,seg,m]['return_rank_ic'] is None})
                        else:
                            diagnostic=di[f,seg,m]
                            assert diagnostic['rows']==int(mask.sum())
                        support_checks.append({'fold':f,'mean_id':m,'segment':seg,
                            'inference_rows':int(mask.sum()),'scored_rows':int(sm.sum()),
                            'unscored_inference_rows':int((mask&~sm).sum()),'all_inference_support_exact':True})
                    measured=return_scores(a['actual'][sm,0],a['mu'][sm],fit_mean)
                    expected=si[f,seg,m]
                    compare(measured,{k:expected[k] for k in measured},f'{f}/{m}/{seg}/return_score','scalar_return_score')
                    scalar_return_count+=1
                    assert type(expected['rows']) is int and expected['rows']==int(sm.sum())>=16
                    for key in ('zero_return_mse','fit_mean_return_mse'):
                        assert expected[key]==si[f,seg,HALVES[0]][key],(f,m,seg,'changed baseline')
                    if m in CONSTANT_MEANS:
                        assert expected['return_rank_ic'] is None
                        mp=MAPPING[m]
                        if mp['kind'] in ('zero','fit_mean'):
                            baseline='zero_return_mse' if mp['kind']=='zero' else 'fit_mean_return_mse'
                            assert expected['return_mse']==expected[baseline],(f,m,seg,'baseline identity')
                    if not new:
                        compare(expected,old_si[f,seg,m],f'{f}/{m}/{seg}/old_return','unchanged_old_return_score',tolerance='0')
                        old_return_count+=1
                    score_checks.append({'fold':f,'mean_id':m,'segment':seg,'rows':measured['rows'],
                        'return_rank_null':measured['return_rank_ic'] is None,'new_mean':new})
            for seg in SEGMENTS:
                for mid in CLASSIFIERS:
                    m=mid+'_direction' if not mid.startswith('prior_') else GROUPS[0]+'_'+mid.removeprefix('prior_')+'_prior_direction'
                    a=old_stream(m,seg);sm=a['interval_mask'] if seg=='interval' else a['score_support']
                    measured=classification_scores(a['actual'][sm,0],a['logit'][sm]);expected=ci[f,seg,mid]
                    compare(measured,{k:expected[k] for k in measured},f'{f}/{mid}/{seg}/classification','scalar_classification_score')
                    compare(expected,old_ci[f,seg,mid],f'{f}/{mid}/{seg}/old_classification','unchanged_classification_score',tolerance='0')
                    assert np.isfinite(a['probability'][sm]).all() and np.all((a['probability'][sm]>=0)&(a['probability'][sm]<=1))
                    scalar_class_count+=1;old_class_count+=1
                    probability_checks.append({'fold':f,'classifier_id':mid,'segment':seg,'rows':measured['rows'],
                        'weighted_null':measured['weighted_log_loss'] is None,'zero_actual_rows':measured['zero_actual_rows'],
                        'zero_logit_rows':measured['zero_logit_rows'],'complete_record_exact':True})
            for g in GROUPS:
                prior=g+'_magnitude_prior_direction'
                for seg in SEGMENTS:
                    a=old_stream(prior,seg);mask=a['mapped_inference_mask'] if seg=='interval' else a['inference_mask']
                    assert np.all(a['probability'][mask]==scalars['prior_probability'][g])
            for kind in CONSTANT_KINDS:
                for seg in SEGMENTS:
                    a=saved[seg,GROUPS[0]+'_soft_'+kind];b=saved[seg,GROUPS[1]+'_soft_'+kind]
                    exact_array(a['mu'],b['mu'],f'{f}/{kind}/{seg}/group_constant_identity')
            for cid in controls:
                old,new=npz(PARENT/'targets'/f'fold{f}_{cid}.npz'),npz(OUT/'targets'/f'fold{f}_{cid}.npz')
                assert set(old)==set(new)
                for key in old:exact_array(old[key],new[key],f'{f}/{cid}/old_target/{key}')
                assert sha(PARENT/'targets'/f'fold{f}_{cid}.npz')==sha(OUT/'targets'/f'fold{f}_{cid}.npz')
                compare(ri[f,cid],old_ri[f,cid],f'{f}/{cid}/entire_old_row','unchanged_old_economic_row',tolerance='0')
                control_count+=1
        assert scalar_return_count==384 and scalar_class_count==160 and control_count==480
        assert old_return_count==224 and old_class_count==160
        assert len(constant_checks)==96 and len(support_checks)==160
        assert len(summary['soft'])==8 and set(summary['soft'])==set(learned)
        report={'schema':'independent-soft-direction-score-summary-audit-v1','passed':True,
            'scope':'Independent complete file bindings; exact 480 old rows/224 old return/160 old classification records; scalar 384 return and 160 classifier scores; constant baseline/null identities; Decimal60 all 80-policy and five-reference summaries/gates. No fits, model predictor/objective replay, raw market loader, canonical scorer/summary/planner, or new policy. Mapping arithmetic and detailed diagnostic values are delegated to the separate mapping auditor.',
            'source_revision':reg['source_revision'],'source_sha256':BINDINGS,'verified_binding_counts':dict(CATEGORIES),
            'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
            'inventory':{'economic_rows':640,'accounts':1280,'policies':80,'learned_gated_policies':8,
                'new_constant_control_policies':12,'return_scores':384,'classification_scores':160,'mapping_records':8,
                'mapping_diagnostic_inventory':64,'new_mapped_npz':160,'new_artifacts':968,'ancestor_artifacts':3488,
                'unchanged_control_rows':480,'unchanged_control_accounts':960,'unchanged_return_scores':224,
                'unchanged_classification_scores':160,'constant_score_records':96,'new_models':0,'new_prediction_calls':0},
            'numeric_max_absolute_differences':MAXDIFF,'maximum_difference_locations':LOCATIONS,
            'constant_control_checks':constant_checks,'new_prediction_support_checks':support_checks,
            'return_score_checks':score_checks,'probability_score_checks':probability_checks,
            'summary':summary,
            'tiny_nonzero_dd_values':[{'fold':r['fold'],'candidate_id':r['candidate_id'],'cost':c,'maxdd_delta':r[c]['maxdd_delta']}
                for r in rows for c in COSTS if 0<abs(r[c]['maxdd_delta'])<1e-12],
            'limitations':['All observations are reused original development quarters, not independent confirmation.',
                'A frozen mean absolute return does not establish constant conditional magnitude across features or time.',
                'All probability streams/scores are unchanged. Any mapped MSE or economic change cannot be called new probability accuracy.',
                'Weighted sigmoid remains a magnitude-tilted score, not an ordinary up probability.',
                'Interval strata use subsequent evaluation-start regimes retrospectively.',
                'Constant controls repeated across groups are not independent forecasts or replications.',
                'Zero mean is a valid utility input, not B&H or forecast abstention.',
                'Original input causality is hash-bound to previous audits and not rebuilt here.',
                'New mapping/diagnostic arithmetic and own-state/accounts are covered by separate independent audits.',
                'No p-value, selection adjustment, all-trend guarantee, formal P1 result or deployment follows.']}
        assert not REPORT.exists(), 'Preserve prior audit; write a separately named explicitly authorized revision.'
        REPORT.write_text(json.dumps(plain(report),sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'passed':True,'path':str(REPORT),'sha256':sha(REPORT),'inventory':report['inventory'],
            'max_absolute_differences':plain(MAXDIFF)},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-saved-audit',action='store_true')
    parser.add_argument('--expected-results-sha')
    args=parser.parse_args()
    if not args.execute_saved_audit or not args.expected_results_sha or len(args.expected_results_sha)!=64:
        parser.error('Wait for root GO after freeze and completion, with exact results SHA.')
    audit(args.expected_results_sha)

if __name__=='__main__':main()
