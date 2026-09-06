"""Independent Stage18 saved-forecast/score/diagnostic audit; no fits or policy rollouts.

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

import joblib
import numpy as np

ROOT = Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT = ROOT / 'codex_outputs/oracle_regularized_direction_decisions_v1'
PARENT = ROOT / 'codex_outputs/oracle_direction_decisions_v1'
PARITY = ROOT / 'codex_outputs/oracle_frozen_procedure_parity_v1'
REPORT = Path('/tmp/oracle_regularized_direction_score_audit_20260906.json')
FS = tuple(range(5, 13))
GROUPS = ('technical', 'perp_delay0')
WEIGHTINGS = ('ordinary', 'magnitude')
OLD_MODELS = tuple(g+'_'+w for g in GROUPS for w in WEIGHTINGS)
OLD_CLASSIFIERS = OLD_MODELS + tuple('prior_'+w for w in WEIGHTINGS)
MODELS = tuple(m+'_l2unit' for m in OLD_MODELS)
CLASSIFIERS = OLD_CLASSIFIERS + MODELS
HALVES = tuple(g+'_half' for g in GROUPS)
OLD_NEW_MEANS = tuple(m+'_direction' for m in OLD_MODELS) + tuple(
    g+'_'+w+'_prior_direction' for g in GROUPS for w in WEIGHTINGS)
OLD_MEANS = HALVES + OLD_NEW_MEANS
NEW_MEANS = tuple(m+'_direction' for m in MODELS)
MEANS = OLD_MEANS + NEW_MEANS
MAPPING = {g+'_'+w+'_l2unit_direction': {'group':g, 'weighting':w,
    'classifier_id':g+'_'+w+'_l2unit', 'old_classifier':g+'_'+w,
    'parent_mean':g+'_half', 'old_mean':g+'_'+w+'_direction',
    'prior_mean':g+'_'+w+'_prior_direction'} for g in GROUPS for w in WEIGHTINGS}
REFERENCES = {m:tuple(MAPPING[m][k] for k in ('parent_mean','old_mean','prior_mean')) for m in NEW_MEANS}
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


def scalar_model_predict(model, features):
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    assert model.steps[0][0]=='standardscaler' and model.steps[1][0]=='logisticregression'
    assert np.array_equal(logistic.classes_,[0,1])
    assert np.isfinite(features).all() and features.shape[1]==len(scaler.mean_)
    values=[]
    for row in features:
        terms=[float(c)*((float(x)-float(m))/float(s)) for x,m,s,c in
               zip(row,scaler.mean_,scaler.scale_,logistic.coef_[0])]
        values.append(math.fsum([float(logistic.intercept_[0]),*terms]))
    result=np.asarray(values,float)
    return result,np.asarray([scalar_sigmoid(v) for v in result],float)


def mapped_mean(logits, parent, mask):
    result=np.full(len(mask),np.nan)
    result[mask]=[float(int(z>0)-int(z<0))*abs(float(mu))
                  for z,mu in zip(logits[mask],parent[mask])]
    return result


def build_summary(rows, scores, class_scores, controls, hindsight):
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


def audit(expected_results_sha):
    with localcontext() as context:
        context.prec=60
        result=read(bind(OUT/'results.json',expected_results_sha));reg=read(bind(OUT/'registration.json'))
        cfg=reg['config'];config_path=ROOT/'configs/oracle_regularized_direction_decisions_20260906.yaml'
        bind(config_path,reg['config_sha256'],'config')
        pre=read(bind(OUT/'preflight.json',cfg['preflight_sha256']))
        assert result['registration_sha256']==canonical(reg)
        assert reg['preflight_sha256']==cfg['preflight_sha256']
        assert pre['source_bindings']==cfg['source_bindings'] and len(cfg['source_bindings'])==31
        assert pre['config_contract_sha256']==canonical({k:v for k,v in cfg.items() if k!='preflight_sha256'})
        for p,h in cfg['source_bindings'].items():bind(p,h,'registered_source')
        for p,h in pre['direct_source_bindings'].items():bind(p,h,'preflight_direct')
        for p,h in cfg['parent_manifest_bindings'].items():bind(p,h,'parent_manifest')
        assert len(pre['source_artifact_bindings'])==2840
        for p,h in pre['source_artifact_bindings'].items():bind(p,h,'ancestor_artifact')
        parent=read(PARENT/'results.json');preg=read(PARENT/'registration.json')
        assert preg['source_revision']==cfg['parent_source_revision']=='6ae673fcdfeed29280256450c05eb8905af77ee3'
        assert parent['registration_sha256']==canonical(preg)==pre['parent_registration_canonical_sha256']
        controls=tuple(cfg['control_ids'])
        assert len(controls)==52 and set(controls)=={r['candidate_id'] for r in parent['rows']}
        ids=controls+tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
        hindsight={r['candidate_id'] for r in parent['rows'] if r['hindsight_only']}
        assert len(hindsight)==24 and len(ids)==60
        assert cfg['mapping']==MAPPING and cfg['references']=={m:list(v) for m,v in REFERENCES.items()}
        assert tuple(cfg['return_score_means'])==MEANS and tuple(cfg['classifiers'])==CLASSIFIERS
        assert tuple(cfg['new_policy_ids'])==ids[len(controls):]
        assert cfg['regularization_C']=='1.0/float(np.sum(frozen_fit_weights))'
        rows,scores,cs=result['rows'],result['scores'],result['classification_scores']
        ri={(r['fold'],r['candidate_id']):r for r in rows}
        si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
        ci={(r['fold'],r['segment'],r['classifier_id']):r for r in cs}
        fi={r['fold']:r for r in result['fit_records']}
        ds=result['direction_diagnostics'];di={(r['fold'],r['segment'],r['classifier_id']):r for r in ds}
        assert len(rows)==len(ri)==480 and set(ri)=={(f,p) for f in FS for p in ids}
        assert len(scores)==len(si)==224 and set(si)=={(f,s,m) for f in FS for s in SEGMENTS for m in MEANS}
        assert len(cs)==len(ci)==160 and set(ci)=={(f,s,c) for f in FS for s in SEGMENTS for c in CLASSIFIERS}
        assert len(ds)==len(di)==64 and set(di)=={(f,s,c) for f in FS for s in SEGMENTS for c in MODELS}
        assert len(result['fit_records'])==len(fi)==8 and set(fi)==set(FS)
        for k,v in {'new_model_fits':32,'new_unique_priors':0,'new_causal_policy_names':8,
            'total_adaptively_explored_causal_names':198,'risk_model_or_calibration_fits':0}.items():assert result[k]==v
        for k in ('additional_test_used_for_modeling_or_scoring','selection_performed','teacher_use_allowed',
                  'high_probability_generalization_established'):assert result[k] is False
        artifacts={}
        for f in FS:
            fold=read(bind(OUT/f'fold_{f}.json',category='fold_manifest'))
            assert fold['registration_sha256']==result['registration_sha256']
            for key in ('rows','scores','classification_scores','direction_diagnostics'):
                assert fold[key]==[r for r in result[key] if r['fold']==f]
            expected={str((OUT/'targets'/f'fold{f}_{p}.npz').relative_to(ROOT)) for p in ids}
            expected|={str((OUT/'traces'/f'fold{f}_{p}.json').relative_to(ROOT)) for p in ids[len(controls):]}
            expected|={str((OUT/k/f'fold{f}_{m}.npz').relative_to(ROOT)) for k in ('forecasts','calibration') for m in NEW_MEANS}
            expected|={str((OUT/'models'/f'fold{f}_{m}.joblib').relative_to(ROOT)) for m in MODELS}
            expected|={str((OUT/'provenance'/f'fold{f}_fit.json').relative_to(ROOT))}
            assert len(expected)==81 and set(fold['artifact_sha256'])==expected
            for p,h in fold['artifact_sha256'].items():bind(p,h,'new_artifact');artifacts[p]=h
        assert len(artifacts)==648
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
        for r in ds:assert r['regime']==regime[r['fold']]
        summary=build_summary(rows,scores,cs,controls,hindsight)
        for k,v in summary.items():compare(v,result['summary'][k],k,'summary_'+k)
        assert set(summary)==set(result['summary'])
        old_ri={(r['fold'],r['candidate_id']):r for r in parent['rows']}
        old_si={(r['fold'],r['segment'],r['mean_id']):r for r in parent['scores']}
        old_ci={(r['fold'],r['segment'],r['classifier_id']):r for r in parent['classification_scores']}
        old_fi={r['fold']:r for r in parent['fit_records']}
        fit_checks=[];model_checks=[];mapping_checks=[];diagnostic_checks=[];score_checks=[]
        scalar_return_count=scalar_class_count=control_count=old_return_count=old_class_count=0
        for f in FS:
            fit=fi[f];assert fit==read(OUT/'provenance'/f'fold{f}_fit.json')
            prov=fit['fit_provenance'];old_fit=old_fi[f];old_prov=old_fit['fit_provenance']
            data=npz(PARENT/'fit_data'/f'fold{f}_training.npz')
            expected={'fit_positions','timestamps','returns','binary_labels','weights_ordinary','weights_magnitude',
                'predict_positions','predict_timestamps'}|{k+'_'+g for k in ('fit_features','predict_features') for g in GROUPS}
            assert set(data)==expected
            n=len(data['returns']);assert n>=512 and data['returns'].shape==(n,)
            assert np.isfinite(data['returns']).all() and np.all(np.diff(data['timestamps'])>0)
            assert np.all(np.diff(data['fit_positions'])>0) and np.all(np.diff(data['predict_positions'])>0)
            assert data['fit_positions'][-1]<data['predict_positions'][0]
            assert data['timestamps'][-1]<data['predict_timestamps'][0]
            support=next(p for p in pre['support'] if p['fold']==f)
            assert fit['fit_source_binding']==support and support['counts']['fit']==n
            assert fit['frozen_fit_data_reused'] is True and fit['fit_labels_weights_and_priors_match_parent_exactly'] is True
            assert fit['new_unique_prior_estimates']==0 and fit['new_model_fits']==4
            assert fit['fit_priors']==old_fit['fit_priors'] and fit['fit_prior_logits']==old_fit['fit_prior_logits']
            assert canonical(data['returns'].tolist())==support['fit_return_sha256']
            assert matrix_sha(data['returns'])==prov['fit_return_sha256']==old_prov['fit_return_sha256']
            assert len(data['predict_timestamps'])==support['counts']['predict']
            for key in ('feature_columns','feature_counts','index_sha256','mask_counts','mask_ranges',
                        'mask_position_sha256','fit_return_sha256','fit_binary_labels_sha256','fit_class_counts',
                        'fit_features_sha256','predict_features_sha256','fit_features_and_return_sha256','sample_weights'):
                assert prov[key]==old_prov[key],(f,key,'frozen fit input provenance changed')
            assert prov['regularization']['search_performed'] is False
            labels=np.asarray([int(float(y)>0) for y in data['returns']],dtype=np.int64)
            exact_array(labels,data['binary_labels'],f'{f}/labels')
            assert matrix_sha(labels)==prov['fit_binary_labels_sha256']
            abs_mean=math.fsum(abs(float(y))/n for y in data['returns'])
            compare(abs_mean,old_fit['fit_abs_return_mean'],f'{f}/absmean','fit_statistics',tolerance='0')
            compare(float(np.mean(data['returns'])),old_fit['fit_return_mean'],f'{f}/fitmean','fit_statistics',tolerance='0')
            prior_z={};weightchecks={};normalizers={}
            for w in WEIGHTINGS:
                weights=np.ones(n) if w=='ordinary' else np.asarray([abs(float(y))/abs_mean for y in data['returns']])
                exact_array(weights,data['weights_'+w],f'{f}/{w}/weights')
                total=math.fsum(float(v) for v in weights);solver_total=float(np.sum(weights))
                masses=[math.fsum(float(v) for v,b in zip(weights,labels) if b==c) for c in (0,1)]
                assert all(v>0 for v in masses) and math.isfinite(total)
                pi=masses[1]/total;z=math.log(pi)-math.log1p(-pi)
                compare(pi,fit['fit_priors'][w],f'{f}/{w}/prior','fit_priors',tolerance='0')
                compare(z,fit['fit_prior_logits'][w],f'{f}/{w}/logit','fit_priors',tolerance='0')
                C=1./solver_total;l2=1./(C*solver_total)
                record=prov['regularization']['by_weighting'][w]
                expected_norm={'C':C,'solver_weight_sum':solver_total,'actual_l2_strength':l2,
                    'normalizer':prov['regularization']['rule'],'sum_fsum':total,'mean_fsum':total/n,
                    'rows':n,'weight_sha256':matrix_sha(weights)}
                compare(expected_norm,record,f'{f}/{w}/normalizer','normalizer',tolerance='0')
                assert matrix_sha(weights)==prov['sample_weights'][w]['weight_sha256']
                prior_z[w]=z;normalizers[w]=expected_norm
                weightchecks[w]={'prior':pi,'prior_logit':z,'sum_fsum':total,'sum_numpy':solver_total,
                    'sum_minus_rows':total-n,'C':C,'actual_l2_strength':l2,'actual_l2_minus_one':l2-1.,
                    'old_actual_l2_strength':1./solver_total,'maximum_weight':float(max(weights)),
                    'zero_weight_rows':int((weights==0).sum()),
                    'weight_concentration_effective_rows':total*total/math.fsum(float(v)**2 for v in weights),
                    'not_independent_sample_size':True}
            pred_z={};pred_p={};new_states={}
            for g in GROUPS:
                xf,xp=data['fit_features_'+g],data['predict_features_'+g]
                assert xf.shape==(n,29 if g=='technical' else 31) and xp.shape==(len(data['predict_timestamps']),xf.shape[1])
                assert np.isfinite(xf).all() and np.isfinite(xp).all()
                assert canonical(xf.tolist())==support['fit_features_sha256'][g]
                assert canonical(xp.tolist())==support['predict_features_sha256'][g]
                assert matrix_sha(xf)==prov['fit_features_sha256'][g]
                assert matrix_sha(xp)==prov['predict_features_sha256'][g]
                for w in WEIGHTINGS:
                    old_mid=g+'_'+w;mid=old_mid+'_l2unit'
                    new_model=joblib.load(OUT/'models'/f'fold{f}_{mid}.joblib')
                    old_model=joblib.load(PARENT/'models'/f'fold{f}_{old_mid}.joblib')
                    scaler,logistic=new_model.steps[0][1],new_model.steps[1][1]
                    old_scaler,old_logistic=old_model.steps[0][1],old_model.steps[1][1]
                    state=prov['fitted_state'][old_mid]
                    assert int(scaler.n_samples_seen_)==n and np.array_equal(logistic.classes_,[0,1])
                    assert len(logistic.n_iter_)==1 and 0<=int(logistic.n_iter_[0])<1000
                    pars={**old_logistic.get_params(),'C':normalizers[w]['C']}
                    compare(logistic.get_params(),pars,f'{f}/{mid}/only_C','model_parameters',tolerance='0')
                    compare(logistic.get_params(),state['logistic_parameters'],f'{f}/{mid}/saved_params','model_parameters',tolerance='0')
                    compare(logistic.get_params(),prov['parameters']['logistic_by_model'][old_mid],f'{f}/{mid}/all_params','model_parameters',tolerance='0')
                    assert scaler.get_params()==old_scaler.get_params()
                    for attr in ('mean_','var_','scale_','n_features_in_','n_samples_seen_'):
                        exact_array(np.asarray(getattr(scaler,attr)),np.asarray(getattr(old_scaler,attr)),f'{f}/{mid}/same_scaler/{attr}')
                    for field,value in [('scaler_mean',scaler.mean_),('scaler_variance',scaler.var_),('scaler_scale',scaler.scale_),
                                        ('coefficient',logistic.coef_),('intercept',logistic.intercept_)]:
                        compare(value.tolist(),state[field],f'{f}/{mid}/{field}','persisted_model_state',tolerance='0')
                    assert state['regularization']==normalizers[w]
                    assert state['fit_features_labels_weights_sha256']==matrix_sha(np.column_stack((xf,labels,data['weights_'+w])))
                    assert state['coefficient_sha256']==matrix_sha(logistic.coef_)
                    assert state['intercept_sha256']==matrix_sha(logistic.intercept_)
                    for j in range(xf.shape[1]):
                        mu=math.fsum(float(v) for v in xf[:,j])/n
                        var=math.fsum((float(v)-mu)**2 for v in xf[:,j])/n
                        for name,actual,expected in [('mean',float(scaler.mean_[j]),mu),('variance',float(scaler.var_[j]),var)]:
                            compare(abs(actual-expected)/max(1.,abs(expected)),0.,f'{f}/{mid}/unweighted_{name}/{j}','unweighted_scaler_relative')
                    for model_id,model in ((mid,new_model),(old_mid,old_model)):
                        zz,pp=scalar_model_predict(model,xp);pred_z[model_id],pred_p[model_id]=zz,pp
                    ns={'C':normalizers[w]['C'],'solver_weight_sum':normalizers[w]['solver_weight_sum'],
                        'actual_l2_strength':normalizers[w]['actual_l2_strength'],
                        'coefficient_l2_norm':math.sqrt(math.fsum(float(v)**2 for v in logistic.coef_[0])),
                        'old_coefficient_l2_norm':math.sqrt(math.fsum(float(v)**2 for v in old_logistic.coef_[0])),
                        'intercept':float(logistic.intercept_[0]),'old_intercept':float(old_logistic.intercept_[0]),
                        'unchanged_scaler_exact':True,'only_C_setting_changed':True,
                        'old_model_path':str((PARENT/'models'/f'fold{f}_{old_mid}.joblib').relative_to(ROOT))}
                    compare(ns,fit['model_state'][mid],f'{f}/{mid}/model_state','model_state',tolerance='0')
                    new_states[mid]=ns
                    model_checks.append({'fold':f,'model_id':mid,'predict_rows':len(xp),'n_iter':int(logistic.n_iter_[0]),
                        'C':ns['C'],'actual_l2_strength':ns['actual_l2_strength'],
                        'coefficient_l2_norm':ns['coefficient_l2_norm'],'old_coefficient_l2_norm':ns['old_coefficient_l2_norm'],
                        'model_fit_repeated':False,'both_new_and_old_scalar_predictions_recomputed':True})
            for w in WEIGHTINGS:
                pred_z['prior_'+w]=np.full(len(data['predict_timestamps']),prior_z[w])
                pred_p['prior_'+w]=np.full(len(data['predict_timestamps']),scalar_sigmoid(prior_z[w]))
            time_to_position={int(t):i for i,t in enumerate(data['predict_timestamps'])}
            def expand(mid,times,which):
                source=pred_z[mid] if which=='z' else pred_p[mid]
                return np.asarray([source[time_to_position[int(t)]] if int(t) in time_to_position else np.nan for t in times])
            saved_classifier={};saved_means={}
            for m in OLD_NEW_MEANS+NEW_MEANS:
                is_new=m in NEW_MEANS;root=OUT if is_new else PARENT
                if is_new:mp=MAPPING[m];mid=mp['classifier_id'];g=mp['group']
                else:
                    g=next(g for g in GROUPS if m.startswith(g+'_'))
                    w=next(w for w in WEIGHTINGS if m.startswith(g+'_'+w+'_'))
                    mid=('prior_'+w) if '_prior_direction' in m else g+'_'+w
                ev=npz(root/'forecasts'/f'fold{f}_{m}.npz')
                ca=npz(root/'calibration'/f'fold{f}_{m}.npz')
                if is_new:
                    for kind,newstream in (('forecasts',ev),('calibration',ca)):
                        oldstream=npz(PARENT/kind/f"fold{f}_{mp['old_mean']}.npz")
                        assert set(newstream)==set(oldstream)
                        for key in oldstream:
                            if key not in ('mu','logit','probability'):
                                exact_array(newstream[key],oldstream[key],f'{f}/{m}/frozen/{kind}/{key}')
                for seg,stream in [('interval',ca),('evaluation',ev)]:
                    mask=stream['mapped_inference_mask'] if seg=='interval' else stream['inference_mask']
                    z,p=expand(mid,stream['timestamps'],'z'),expand(mid,stream['timestamps'],'p')
                    assert np.array_equal(np.isfinite(stream['logit']),np.isfinite(z))
                    assert np.array_equal(np.isfinite(stream['probability']),np.isfinite(p))
                    good=np.isfinite(z)
                    for k,a,b,tol in [('logit',stream['logit'][good],z[good],'1e-12'),('probability',stream['probability'][good],p[good],'1e-14')]:
                        compare(float(np.max(np.abs(a-b))),0.,f'{f}/{m}/{seg}/{k}','scalar_model_'+k,tolerance=tol)
                    if seg=='interval':
                        assert np.array_equal(good,stream['classifier_predict_mask'])
                        assert np.all(mask<=good) and not np.any(mask & stream['scale_mask'])
                    else:assert np.array_equal(good,mask)
                    expected=mapped_mean(stream['logit'],stream['parent_mu'],mask)
                    exact_array(stream['mu'],expected,f'{f}/{m}/{seg}/mapped_mu')
                    assert np.array_equal(np.isfinite(stream['mu']),mask)
                    sm=stream['interval_mask'] if seg=='interval' else stream['score_support']
                    assert np.all(sm<=mask)
                    if (seg,mid) in saved_classifier:
                        other=saved_classifier[seg,mid]
                        for k in ('logit','probability','timestamps','actual'):
                            exact_array(stream[k],other[k],f'{f}/{mid}/{seg}/shared/{k}')
                    else:saved_classifier[seg,mid]=stream
                    saved_means[seg,m]=stream
                    measured=return_scores(stream['actual'][sm,0],stream['mu'][sm],float(stream['fit_return_mean']))
                    saved=si[f,seg,m]
                    compare(measured,{k:saved[k] for k in measured},f'{f}/{m}/{seg}/return_score','scalar_return_score')
                    scalar_return_count+=1
                    if not is_new:
                        compare(saved,old_si[f,seg,m],f'{f}/{m}/{seg}/old_score','unchanged_old_return_score',tolerance='0')
                        old_return_count+=1
                    else:
                        oldstream=npz(PARENT/('calibration' if seg=='interval' else 'forecasts')/f"fold{f}_{mp['old_mean']}.npz")
                        nz=stream['logit'][mask];oz=oldstream['logit'][mask];dn=len(nz)
                        measured_diagnostic={'fold':f,'segment':seg,'classifier_id':mid,'regime':regime[f],'rows':dn,
                            'uses_all_inference_not_score_support':True,
                            'sign_disagreements_vs_C1':sum((int(a>0)-int(a<0))!=(int(b>0)-int(b<0)) for a,b in zip(nz,oz)),
                            'sign_matches_matched_prior':sum((int(a>0)-int(a<0))==(int(prior_z[mp['weighting']]>0)-int(prior_z[mp['weighting']]<0)) for a in nz),
                            'zero_logit_rows':sum(a==0 for a in nz),'old_zero_logit_rows':sum(a==0 for a in oz),
                            'mean_abs_logit':math.fsum(abs(float(v)) for v in nz)/dn,
                            'old_mean_abs_logit':math.fsum(abs(float(v)) for v in oz)/dn,**new_states[mid]}
                        compare(measured_diagnostic,di[f,seg,mid],f'{f}/{mid}/{seg}/diagnostic','direction_diagnostic')
                        diagnostic_checks.append(measured_diagnostic)
                        mapping_checks.append({'fold':f,'mean_id':m,'segment':seg,'inference_rows':int(mask.sum()),
                            'scored_rows':int(sm.sum()),'unscored_inference_rows':int((mask&~sm).sum()),
                            'zero_parent_mean_rows':int((stream['parent_mu'][mask]==0).sum()),
                            'zero_logit_rows':int((stream['logit'][mask]==0).sum()),'all_inference_mapping_exact':True})
            for g in GROUPS:
                m=g+'_half';ev=npz(PARITY/'forecasts'/f'fold{f}_{m}.npz')
                ca=saved_means['interval',g+'_ordinary_direction']
                for seg,stream,mu in [('evaluation',ev,ev['mu']),('interval',ca,ca['parent_mu'])]:
                    sm=stream['interval_mask'] if seg=='interval' else stream['score_support']
                    measured=return_scores(stream['actual'][sm,0],mu[sm],old_fit['fit_return_mean'])
                    compare(measured,{k:si[f,seg,m][k] for k in measured},f'{f}/{m}/{seg}/return_score','scalar_return_score')
                    compare(si[f,seg,m],old_si[f,seg,m],f'{f}/{m}/{seg}/old_score','unchanged_old_return_score',tolerance='0')
                    scalar_return_count+=1;old_return_count+=1
            for seg in SEGMENTS:
                for mid in CLASSIFIERS:
                    stream=saved_classifier[seg,mid];sm=stream['interval_mask'] if seg=='interval' else stream['score_support']
                    measured=classification_scores(stream['actual'][sm,0],stream['logit'][sm]);saved=ci[f,seg,mid]
                    compare(measured,{k:saved[k] for k in measured},f'{f}/{mid}/{seg}/classification','scalar_classification_score')
                    scalar_class_count+=1
                    if mid in OLD_CLASSIFIERS:
                        compare(saved,old_ci[f,seg,mid],f'{f}/{mid}/{seg}/old_classification','unchanged_old_classification',tolerance='0')
                        old_class_count+=1
                    score_checks.append({'fold':f,'classifier_id':mid,'segment':seg,'rows':measured['rows'],
                        'weighted_null':measured['weighted_log_loss'] is None,
                        'zero_actual_rows':measured['zero_actual_rows'],'zero_logit_rows':measured['zero_logit_rows']})
            for cid in controls:
                old,new=npz(PARENT/'targets'/f'fold{f}_{cid}.npz'),npz(OUT/'targets'/f'fold{f}_{cid}.npz')
                assert set(old)==set(new)
                for k in old:exact_array(old[k],new[k],f'{f}/{cid}/old_target/{k}')
                for cost in COSTS:compare(ri[f,cid][cost],old_ri[f,cid][cost],f'{f}/{cid}/{cost}','unchanged_control_accounts',tolerance='0')
                control_count+=1
            fit_checks.append({'fold':f,'fit_rows':n,'predict_rows':len(data['predict_timestamps']),
                'class_counts':[int((labels==c).sum()) for c in (0,1)],'actual_zero_rows':int((data['returns']==0).sum()),
                'fit_abs_return_mean':abs_mean,'weights':weightchecks})
        assert scalar_return_count==224 and scalar_class_count==160 and control_count==416
        assert old_return_count==160 and old_class_count==96
        assert len(model_checks)==32 and len(mapping_checks)==64 and len(diagnostic_checks)==64
        report={'schema':'independent-regularized-direction-score-audit-v1','passed':True,
            'scope':'Independent saved-source hashes; reused fit labels/weights/priors; exact C and scaler identities; scalar new/old model predictions; mappings; 224 return scores; 160 classifier scores; 64 all-inference diagnostics; Decimal60 summaries. No refitting, raw market loader, canonical fitter/scorer/summary/planner or new policy.',
            'source_revision':reg['source_revision'],'source_sha256':BINDINGS,'verified_binding_counts':dict(CATEGORIES),
            'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
            'inventory':{'economic_rows':480,'accounts':960,'policies':60,'return_scores':224,'classification_scores':160,
                'fit_records':8,'new_models':32,'old_models_scalar_rechecked':32,'unique_prior_verifications':16,
                'direction_diagnostics':64,'new_mapped_array_pairs':32,'new_artifacts':648,'ancestor_artifacts':2840,
                'unchanged_control_rows':416,'unchanged_control_accounts':832,'unchanged_return_scores':160,'unchanged_classification_scores':96},
            'numeric_max_absolute_differences':MAXDIFF,'maximum_difference_locations':LOCATIONS,
            'fit_checks':fit_checks,'model_checks':model_checks,'mapping_checks':mapping_checks,
            'classification_score_checks':score_checks,'direction_diagnostics':diagnostic_checks,'summary':summary,
            'tiny_nonzero_dd_values':[{'fold':r['fold'],'candidate_id':r['candidate_id'],'cost':c,'maxdd_delta':r[c]['maxdd_delta']}
                for r in rows for c in COSTS if 0<abs(r[c]['maxdd_delta'])<1e-12],
            'limitations':['All observations are reused original development quarters, not independent confirmation.',
                'A constant normalized penalty does not establish sample independence or future regime stability.',
                'Weighted sigmoid is a magnitude-tilted score, not an ordinary up-probability claim.',
                'Interval strata use subsequent evaluation-start regimes retrospectively.',
                'Mapped mean is sign(logit) times frozen parent mean magnitude, not a newly calibrated conditional mean.',
                'Fit-matrix causality and original market-data construction are bound to earlier audits, not rebuilt here.',
                'Optimization stationarity and new own-state/account rollouts are covered by separate independent audits.',
                'No p-value, selection adjustment, all-trend probability guarantee, formal P1 result or deployment follows.']}
        assert not REPORT.exists(), 'Preserve the prior saved audit; use a separately named explicitly authorized revision.'
        REPORT.write_text(json.dumps(plain(report),sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'passed':True,'path':str(REPORT),'sha256':sha(REPORT),'inventory':report['inventory'],
            'max_absolute_differences':plain(MAXDIFF)},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-saved-audit',action='store_true')
    parser.add_argument('--expected-results-sha')
    args=parser.parse_args()
    if not args.execute_saved_audit or not args.expected_results_sha or len(args.expected_results_sha)!=64:
        parser.error('Execution remains gated until root authorizes the completed frozen run and its results SHA.')
    audit(args.expected_results_sha)


if __name__=='__main__':main()
