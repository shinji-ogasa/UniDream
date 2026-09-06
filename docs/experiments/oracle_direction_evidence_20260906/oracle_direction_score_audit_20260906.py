"""Independent Stage17 saved-forecast/score audit; no fits or policy rollouts.

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
OUT = ROOT / 'codex_outputs/oracle_direction_decisions_v1'
PARENT = ROOT / 'codex_outputs/oracle_sign_magnitude_decisions_v1'
PARITY = ROOT / 'codex_outputs/oracle_frozen_procedure_parity_v1'
REPORT = Path('/tmp/oracle_direction_score_audit_20260906.json')
FS = tuple(range(5, 13))
GROUPS = ('technical', 'perp_delay0')
WEIGHTINGS = ('ordinary', 'magnitude')
MODELS = tuple(g+'_'+w for g in GROUPS for w in WEIGHTINGS)
CLASSIFIERS = MODELS + tuple('prior_'+w for w in WEIGHTINGS)
HALVES = tuple(g+'_half' for g in GROUPS)
NEW_MEANS = tuple(m+'_direction' for m in MODELS) + tuple(
    g+'_'+w+'_prior_direction' for g in GROUPS for w in WEIGHTINGS)
MEANS = HALVES + NEW_MEANS
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
    mapping={g+'_'+w+s:{'group':g,'weighting':w,'classifier_id':('prior_'+w if s=='_prior_direction' else g+'_'+w),
        'parent_mean':g+'_half','prior_mean':g+'_'+w+'_prior_direction'}
        for s in ('_direction','_prior_direction') for g in GROUPS for w in WEIGHTINGS}
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
                    mid=group+'_'+weighting
                    refs=('prior_ordinary','prior_magnitude',group+'_'+('magnitude' if weighting=='ordinary' else 'ordinary'))
                    out['classification_paired'][g][seg][mid]={ref:{k:mean([
                        None if ci[f,seg,mid][k] is None or ci[f,seg,ref][k] is None else
                        D(ci[f,seg,mid][k])-D(ci[f,seg,ref][k]) for f in fs]) for k in ck} for ref in refs}
        for m in NEW_MEANS:
            refs=tuple(dict.fromkeys((mapping[m]['parent_mean'],mapping[m]['prior_mean'])))
            out['paired'][g][m]={ref:{'prediction':{seg:{
                'mse_difference':mean([D(si[f,seg,m]['return_mse'])-D(si[f,seg,ref]['return_mse']) for f in fs]),
                'improved_quarters':sum(si[f,seg,m]['return_mse']<si[f,seg,ref]['return_mse'] for f in fs),
                'equal_quarters':sum(si[f,seg,m]['return_mse']==si[f,seg,ref]['return_mse'] for f in fs)} for seg in SEGMENTS},
                'economics':{r:{c:{k:mean([D(ri[f,m+'_'+r][c][k])-D(ri[f,ref+'_'+r][c][k]) for f in fs])
                    for k in EK} for c in COSTS} for r in RULES}} for ref in refs}
    for m in NEW_MEANS:
        mp=mapping[m];mid=mp['classifier_id']
        sk=('brier','log_loss') if mp['weighting']=='ordinary' else ('weighted_brier','weighted_log_loss')
        cg={seg:mid in MODELS and all(out['classification_paired'][g][seg][mid]['prior_'+mp['weighting']][k] is not None
            and out['classification_paired'][g][seg][mid]['prior_'+mp['weighting']][k]<0 for g in STRATA for k in sk) for seg in SEGMENTS}
        mg={seg:all(out['prediction'][g][seg][m]['return_mse']<out['prediction'][g][seg][m][ref]
            and all(out['paired'][g][m][p]['prediction'][seg]['mse_difference']<0 for p in
                tuple(dict.fromkeys((mp['parent_mean'],mp['prior_mean']))))
            for g in STRATA for ref in ('zero_return_mse','fit_mean_return_mse')) for seg in SEGMENTS}
        for r in RULES:
            cid=m+'_'+r
            out['direction'][cid]={'economic_means_all_strata_both_costs':all(
                out['economics'][g][cid][c]['alpha_ex']>0 and out['economics'][g][cid][c]['maxdd_delta']<0 for g in STRATA for c in COSTS),
                'matched_probability_losses_improved_all_strata':cg,
                'mapped_mse_vs_zero_fitmean_parent_and_matched_prior_all_strata':mg,
                'high_probability_generalization_established':False,'regime_count_gate_pass':False}
    return out


def audit():
    with localcontext() as context:
        context.prec=60
        result=read(bind(OUT/'results.json'));reg=read(bind(OUT/'registration.json'))
        cfg=reg['config']
        bind(ROOT/'configs/oracle_direction_decisions_20260906.yaml',reg['config_sha256'],'config')
        pre=read(bind(OUT/'preflight.json',cfg['preflight_sha256']))
        assert result['registration_sha256']==canonical(reg)
        assert reg['preflight_sha256']==cfg['preflight_sha256']
        assert pre['source_bindings']==cfg['source_bindings'] and len(cfg['source_bindings'])==29
        assert pre['config_contract_sha256']==canonical({k:v for k,v in cfg.items() if k!='preflight_sha256'})
        for p,h in cfg['source_bindings'].items():bind(p,h,'registered_source')
        for p,h in pre['direct_source_bindings'].items():bind(p,h,'preflight_direct')
        for p,h in cfg['parent_manifest_bindings'].items():bind(p,h,'parent_manifest')
        assert len(pre['source_artifact_bindings'])==2120
        for p,h in pre['source_artifact_bindings'].items():bind(p,h,'ancestor_artifact')
        parent=read(PARENT/'results.json');parent_reg=read(PARENT/'registration.json')
        assert parent_reg['source_revision']==cfg['parent_source_revision']=='b44c211dccc38f719b6f893a95c0d1a2d4cbf638'
        assert parent['registration_sha256']==canonical(parent_reg)
        controls=tuple(cfg['control_ids'])
        assert len(controls)==36 and set(controls)=={r['candidate_id'] for r in parent['rows']}
        ids=controls+tuple(m+'_'+r for m in NEW_MEANS for r in RULES)
        hindsight={r['candidate_id'] for r in parent['rows'] if r['hindsight_only']}
        assert len(hindsight)==24
        assert tuple(cfg['return_score_means'])==MEANS and tuple(cfg['classifiers'])==CLASSIFIERS
        assert tuple(cfg['new_policy_ids'])==ids[len(controls):]
        rows,scores,cs=result['rows'],result['scores'],result['classification_scores']
        ri={(r['fold'],r['candidate_id']):r for r in rows}
        si={(r['fold'],r['segment'],r['mean_id']):r for r in scores}
        ci={(r['fold'],r['segment'],r['classifier_id']):r for r in cs}
        fi={r['fold']:r for r in result['fit_records']}
        assert len(rows)==len(ri)==416 and set(ri)=={(f,p) for f in FS for p in ids}
        assert len(scores)==len(si)==160 and set(si)=={(f,s,m) for f in FS for s in SEGMENTS for m in MEANS}
        assert len(cs)==len(ci)==96 and set(ci)=={(f,s,c) for f in FS for s in SEGMENTS for c in CLASSIFIERS}
        assert len(result['fit_records'])==len(fi)==8 and set(fi)==set(FS)
        for k,v in {'new_model_fits':32,'shared_prior_estimates':16,'new_causal_policy_names':16,
            'total_adaptively_explored_causal_names':190,'risk_model_or_calibration_fits':0}.items():assert result[k]==v
        for k in ('additional_test_used_for_modeling_or_scoring','selection_performed','teacher_use_allowed',
                  'high_probability_generalization_established'):assert result[k] is False
        artifacts={}
        for f in FS:
            fold=read(bind(OUT/f'fold_{f}.json',category='fold_manifest'))
            assert fold['registration_sha256']==result['registration_sha256']
            for key in ('rows','scores','classification_scores'):
                assert fold[key]==[r for r in result[key] if r['fold']==f]
            expected={str((OUT/'targets'/f'fold{f}_{p}.npz').relative_to(ROOT)) for p in ids}
            expected|={str((OUT/'traces'/f'fold{f}_{p}.json').relative_to(ROOT)) for p in ids[len(controls):]}
            expected|={str((OUT/k/f'fold{f}_{m}.npz').relative_to(ROOT)) for k in ('forecasts','calibration') for m in NEW_MEANS}
            expected|={str((OUT/'models'/f'fold{f}_{m}.joblib').relative_to(ROOT)) for m in MODELS}
            expected|={str((OUT/'fit_data'/f'fold{f}_training.npz').relative_to(ROOT)),str((OUT/'provenance'/f'fold{f}_fit.json').relative_to(ROOT))}
            assert len(expected)==90 and set(fold['artifact_sha256'])==expected
            for p,h in fold['artifact_sha256'].items():bind(p,h,'new_artifact');artifacts[p]=h
        assert len(artifacts)==720
        regime={f:ri[f,'bh']['regime'] for f in FS}
        assert Counter(r['trend'] for r in regime.values())=={'bull':2,'bear':4,'sideways':2}
        for r in rows:
            assert r['regime']==regime[r['fold']] and r['hindsight_only']==(r['candidate_id'] in hindsight)
        for r in scores+cs:
            assert r['regime']==regime[r['fold']]
            assert r['regime_known_at_scored_decisions']==(r['segment']=='evaluation')
            assert r['regime_reference']=='evaluation_quarter_start'
        summary=build_summary(rows,scores,cs,controls,hindsight)
        for k,v in summary.items():compare(v,result['summary'][k],k,'summary_'+k)
        assert set(summary)==set(result['summary'])
        fit_checks=[];model_checks=[];score_checks=[];mapping_checks=[]
        old_ri={(r['fold'],r['candidate_id']):r for r in parent['rows']}
        old_si={(r['fold'],r['mean_id']):r for r in parent['scores'] if r['subset']=='all'}
        scalar_return_count=scalar_class_count=control_count=0
        for f in FS:
            fit=fi[f];assert fit==read(OUT/'provenance'/f'fold{f}_fit.json')
            prov=fit['fit_provenance'];data=npz(OUT/'fit_data'/f'fold{f}_training.npz')
            expected={'fit_positions','timestamps','returns','binary_labels','weights_ordinary','weights_magnitude',
                'predict_positions','predict_timestamps'}|{k+'_'+g for k in ('fit_features','predict_features') for g in GROUPS}
            assert set(data)==expected
            n=len(data['returns']);assert n>=512 and data['returns'].shape==(n,)
            assert np.isfinite(data['returns']).all() and np.all(np.diff(data['timestamps'])>0)
            assert np.all(np.diff(data['fit_positions'])>0) and np.all(np.diff(data['predict_positions'])>0)
            assert data['fit_positions'][-1]<data['predict_positions'][0]
            assert data['timestamps'][-1]<data['predict_timestamps'][0]
            binding=next(p for p in pre['support'] if p['fold']==f)
            assert fit['fit_source_binding']==binding and binding['counts']['fit']==n
            assert canonical(data['returns'].tolist())==binding['fit_return_sha256']
            assert matrix_sha(data['returns'])==prov['fit_return_sha256']
            assert len(data['predict_timestamps'])==binding['counts']['predict']
            labels=np.asarray([int(float(y)>0) for y in data['returns']],dtype=np.int64)
            exact_array(labels,data['binary_labels'],f'{f}/labels')
            assert matrix_sha(labels)==prov['fit_binary_labels_sha256']
            abs_mean=math.fsum(abs(float(y))/n for y in data['returns'])
            compare(abs_mean,fit['fit_abs_return_mean'],f'{f}/absmean','fit_statistics',tolerance='0')
            compare(float(np.mean(data['returns'])),fit['fit_return_mean'],f'{f}/fitmean','fit_statistics',tolerance='0')
            priors={};prior_z={};weightchecks={}
            for w in WEIGHTINGS:
                weights=np.ones(n) if w=='ordinary' else np.asarray([abs(float(y))/abs_mean for y in data['returns']])
                exact_array(weights,data['weights_'+w],f'{f}/{w}/weights')
                total=math.fsum(float(v) for v in weights)
                masses=[math.fsum(float(v) for v,b in zip(weights,labels) if b==c) for c in (0,1)]
                assert all(v>0 for v in masses) and math.isfinite(total)
                pi=masses[1]/total;z=math.log(pi)-math.log1p(-pi)
                compare(pi,fit['fit_priors'][w],f'{f}/{w}/prior','fit_priors',tolerance='0')
                compare(z,fit['fit_prior_logits'][w],f'{f}/{w}/logit','fit_priors',tolerance='0')
                record=prov['sample_weights'][w]
                assert matrix_sha(weights)==record['weight_sha256']
                compare(total,record['sum_fsum'],f'{f}/{w}/sum','fit_statistics',tolerance='0')
                compare(float(np.sum(weights)),record['sum_numpy_used_by_solver'],f'{f}/{w}/solver_sum','fit_statistics',tolerance='0')
                compare(masses,record['positive_weight_by_class'],f'{f}/{w}/masses','fit_statistics',tolerance='0')
                priors[w]=pi;prior_z[w]=z
                weightchecks[w]={'prior':pi,'prior_logit':z,'sum_fsum':total,'sum_numpy':float(np.sum(weights)),
                    'sum_minus_rows':total-n,'maximum_weight':float(max(weights)),
                    'zero_weight_rows':int((weights==0).sum()),
                    'weight_concentration_effective_rows':total*total/math.fsum(float(v)**2 for v in weights),
                    'not_independent_sample_size':True}
            pred_z={};pred_p={}
            for g in GROUPS:
                xf,xp=data['fit_features_'+g],data['predict_features_'+g]
                assert xf.shape==(n,29 if g=='technical' else 31) and xp.shape==(len(data['predict_timestamps']),xf.shape[1])
                assert np.isfinite(xf).all() and np.isfinite(xp).all()
                assert canonical(xf.tolist())==binding['fit_features_sha256'][g]
                assert canonical(xp.tolist())==binding['predict_features_sha256'][g]
                assert matrix_sha(xf)==prov['fit_features_sha256'][g]
                assert matrix_sha(xp)==prov['predict_features_sha256'][g]
                for w in WEIGHTINGS:
                    mid=g+'_'+w
                    model=joblib.load(OUT/'models'/f'fold{f}_{mid}.joblib')
                    scaler,logistic=model.steps[0][1],model.steps[1][1]
                    state=prov['fitted_state'][mid]
                    assert state['feature_counts'] if 'feature_counts' in state else True
                    assert int(scaler.n_samples_seen_)==n and np.array_equal(logistic.classes_,[0,1])
                    assert len(logistic.n_iter_)==1 and 0<=int(logistic.n_iter_[0])<1000
                    compare(logistic.get_params(),prov['parameters']['logistic'],f'{f}/{mid}/params','model_parameters',tolerance='0')
                    for field,value in [('scaler_mean',scaler.mean_),('scaler_variance',scaler.var_),('scaler_scale',scaler.scale_),
                                        ('coefficient',logistic.coef_),('intercept',logistic.intercept_)]:
                        compare(value.tolist(),state[field],f'{f}/{mid}/{field}','persisted_model_state',tolerance='0')
                    for j in range(xf.shape[1]):
                        mu=math.fsum(float(v) for v in xf[:,j])/n
                        var=math.fsum((float(v)-mu)**2 for v in xf[:,j])/n
                        for name,actual,expected in [('mean',float(scaler.mean_[j]),mu),('variance',float(scaler.var_[j]),var)]:
                            difference=abs(actual-expected)/max(1.,abs(expected))
                            compare(difference,0.,f'{f}/{mid}/unweighted_{name}/{j}','unweighted_scaler_relative')
                    z,p=scalar_model_predict(model,xp)
                    pred_z[mid],pred_p[mid]=z,p
                    assert state['fit_features_labels_weights_sha256']==matrix_sha(np.column_stack((xf,labels,data['weights_'+w])))
                    model_checks.append({'fold':f,'model_id':mid,'predict_rows':len(z),'n_iter':int(logistic.n_iter_[0]),
                        'model_fit_repeated':False,'scalar_prediction_recomputed':True})
            for w in WEIGHTINGS:
                pred_z['prior_'+w]=np.full(len(data['predict_timestamps']),prior_z[w])
                pred_p['prior_'+w]=np.full(len(data['predict_timestamps']),scalar_sigmoid(prior_z[w]))
            time_to_position={int(t):i for i,t in enumerate(data['predict_timestamps'])}
            def expand(mid,times,which):
                source=pred_z[mid] if which=='z' else pred_p[mid]
                return np.asarray([source[time_to_position[int(t)]] if int(t) in time_to_position else np.nan for t in times])
            eval_streams={};cal_streams={};saved_classifier={}
            original_cal={};original_eval={}
            for g in GROUPS:
                original_eval[g]=npz(PARITY/'forecasts'/f'fold{f}_{g}_half.npz')
                original_cal[g]=npz(PARITY/'calibration'/f'fold{f}_{g}.npz')
            cal_prov=read(PARITY/'calibration'/f'fold{f}_provenance.json')['calibration']
            for m in NEW_MEANS:
                mp=cfg['mapping'][m];mid=mp['classifier_id'];g=mp['group']
                ev=npz(OUT/'forecasts'/f'fold{f}_{m}.npz')
                ca=npz(OUT/'calibration'/f'fold{f}_{m}.npz')
                oe,oc=original_eval[g],original_cal[g]
                assert set(ev)==set(oe)|{'parent_mu','logit','probability'}
                for key in oe:
                    if key!='mu':exact_array(ev[key],oe[key],f'{f}/{m}/old_E/{key}')
                exact_array(ev['parent_mu'],oe['mu'],f'{f}/{m}/E_parent')
                for key in ('timestamps','actual','scale_mask','interval_mask'):
                    exact_array(ca[key],oc[key],f'{f}/{m}/old_I/{key}')
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
                        expected_parent=np.full(len(mask),np.nan)
                        anchor=float(cal_prov['scale_mean']);bias=float(cal_prov['return_bias'][g])
                        expected_parent[mask]=.5*anchor+.5*(oc['mu'][mask]+bias)
                        exact_array(stream['parent_mu'],expected_parent,f'{f}/{m}/I_parent_half')
                    else:assert np.array_equal(good,mask)
                    # Exact saved-logit mapping; scalar model arithmetic is checked above.
                    expected=mapped_mean(stream['logit'],stream['parent_mu'],mask)
                    exact_array(stream['mu'],expected,f'{f}/{m}/{seg}/mapped_mu')
                    assert np.array_equal(np.isfinite(stream['mu']),mask)
                    scoremask=stream['interval_mask'] if seg=='interval' else stream['score_support']
                    assert np.all(scoremask<=mask)
                    if (seg,mid) in saved_classifier:
                        other=saved_classifier[seg,mid]
                        for k in ('logit','probability','timestamps','actual'):
                            exact_array(stream[k],other[k],f'{f}/{mid}/{seg}/shared_classifier/{k}')
                    else:saved_classifier[seg,mid]=stream
                    measured=return_scores(stream['actual'][scoremask,0],stream['mu'][scoremask],float(stream['fit_return_mean']))
                    saved=si[f,seg,m]
                    compare(measured,{k:saved[k] for k in measured},f'{f}/{m}/{seg}/return_score','scalar_return_score')
                    scalar_return_count+=1
                    mapping_checks.append({'fold':f,'mean_id':m,'segment':seg,'inference_rows':int(mask.sum()),
                        'scored_rows':int(scoremask.sum()),'unscored_inference_rows':int((mask&~scoremask).sum()),
                        'zero_parent_mean_rows':int((stream['parent_mu'][mask]==0).sum()),
                        'zero_logit_rows':int((stream['logit'][mask]==0).sum()),'all_inference_mapping_exact':True})
                eval_streams[m],cal_streams[m]=ev,ca
            for g in GROUPS:
                m=g+'_half'
                example=cal_streams[g+'_ordinary_direction']
                for seg,a,mask,mu in [('evaluation',original_eval[g]['actual'],original_eval[g]['score_support'],original_eval[g]['mu']),
                    ('interval',example['actual'],example['interval_mask'],example['parent_mu'])]:
                    measured=return_scores(a[mask,0],mu[mask],fit['fit_return_mean'])
                    saved=si[f,seg,m]
                    compare(measured,{k:saved[k] for k in measured},f'{f}/{m}/{seg}/return_score','scalar_return_score')
                    scalar_return_count+=1
                    if seg=='evaluation':compare({k:saved[k] for k in measured},{k:old_si[f,m][k] for k in measured},
                        f'{f}/{m}/old_return_score','unchanged_parent_score',tolerance='0')
            for seg in SEGMENTS:
                for mid in CLASSIFIERS:
                    stream=saved_classifier[seg,mid]
                    sm=stream['interval_mask'] if seg=='interval' else stream['score_support']
                    measured=classification_scores(stream['actual'][sm,0],stream['logit'][sm])
                    saved=ci[f,seg,mid]
                    compare(measured,{k:saved[k] for k in measured},f'{f}/{mid}/{seg}/classification','scalar_classification_score')
                    scalar_class_count+=1
                    score_checks.append({'fold':f,'classifier_id':mid,'segment':seg,
                        'rows':measured['rows'],'weighted_null':measured['weighted_log_loss'] is None,
                        'zero_actual_rows':measured['zero_actual_rows'],'zero_logit_rows':measured['zero_logit_rows']})
            for cid in controls:
                old,new=npz(PARENT/'targets'/f'fold{f}_{cid}.npz'),npz(OUT/'targets'/f'fold{f}_{cid}.npz')
                assert set(old)==set(new)
                for k in old:exact_array(old[k],new[k],f'{f}/{cid}/old_target/{k}')
                for c in COSTS:compare(ri[f,cid][c],old_ri[f,cid][c],f'{f}/{cid}/{c}','unchanged_control_accounts',tolerance='0')
                control_count+=1
            fit_checks.append({'fold':f,'fit_rows':n,'predict_rows':len(data['predict_timestamps']),
                'class_counts':[int((labels==c).sum()) for c in (0,1)],'actual_zero_rows':int((data['returns']==0).sum()),
                'fit_abs_return_mean':abs_mean,'weights':weightchecks})
        assert scalar_return_count==160 and scalar_class_count==96 and control_count==288
        assert len(model_checks)==32 and len(mapping_checks)==128
        report={'schema':'independent-direction-score-audit-v1','passed':True,
            'scope':'Independent saved-source hashes, fit labels/weights/priors, scalar model prediction, mappings, 160 return scores, 96 classifier scores and Decimal60 summaries. No refitting, raw market loader, canonical fitter/scorer/summary/planner or new policy.',
            'source_revision':reg['source_revision'],'source_sha256':BINDINGS,'verified_binding_counts':dict(CATEGORIES),
            'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
            'inventory':{'economic_rows':416,'accounts':832,'policies':52,'return_scores':160,'classification_scores':96,
                'fit_records':8,'models':32,'unique_fit_priors':16,'mapped_array_pairs':64,
                'new_artifacts':720,'ancestor_artifacts':2120,'unchanged_control_rows':288,'unchanged_control_accounts':576},
            'numeric_max_absolute_differences':MAXDIFF,'maximum_difference_locations':LOCATIONS,
            'fit_checks':fit_checks,'model_checks':model_checks,'mapping_checks':mapping_checks,
            'classification_score_checks':score_checks,'summary':summary,
            'tiny_nonzero_dd_values':[{'fold':r['fold'],'candidate_id':r['candidate_id'],'cost':c,'maxdd_delta':r[c]['maxdd_delta']}
                for r in rows for c in COSTS if 0<abs(r[c]['maxdd_delta'])<1e-12],
            'limitations':['All observations are the reused original development quarters, not independent confirmation.',
                'Weighted sigmoid is a magnitude-tilted score, not an ordinary up-probability claim.',
                'Interval strata use subsequent evaluation-start regimes retrospectively.',
                'The mapped mean is sign(logit) times the frozen parent mean magnitude, not a newly calibrated conditional mean.',
                'Fit-matrix causality and original market-data construction are hash-bound to earlier audits; raw data is not recomputed here.',
                'Model optimization stationarity and new own-state/account rollouts are covered by separate independent audits.',
                'No p-value, selection adjustment, all-trend probability guarantee, formal P1 result or deployment follows.']}
        REPORT.write_text(json.dumps(plain(report),sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'passed':True,'path':str(REPORT),'sha256':sha(REPORT),'inventory':report['inventory'],
                          'max_absolute_differences':plain(MAXDIFF)},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-saved-audit',action='store_true')
    args=parser.parse_args()
    if not args.execute_saved_audit:
        parser.error('Execution remains gated until root authorizes the completed frozen run.')
    audit()


if __name__=='__main__':main()
