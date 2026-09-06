"""Independent, saved-results-only Decimal aggregation; never fits or rolls out."""
import hashlib, json, math
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np

ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
OUT=ROOT/'codex_outputs/oracle_information_decomposition_v1'
REPORT=Path('/tmp/oracle_information_decomposition_summary_audit_20260906.json')
FOLDS=tuple(range(5,13)); COSTS=('base','stress_2x'); STRATA=('all','bull','bear','sideways')
MEANS=('scale_mean','technical_scaled','perp_delay0_scaled','technical_half','perp_delay0_half')
HALVES=('technical_half','perp_delay0_half'); RULES=('utility_risk1','utility_risk1_fallback_bh')
SWAPS=('return','realized_risk','both'); KEYS=('alpha_ex','maxdd_delta','turnover','trades')
CONTROLS=('bh','common_robust')+tuple(m+'_'+q for m in MEANS for q in RULES)
HYBRIDS=tuple(m+'_oracle_'+s+'_'+q for m in HALVES for s in SWAPS for q in RULES)
BEAMS=tuple(f'matched_rl_beam32_{q}_risk{p}' for q in ('hold','fallback_bh') for p in (0,1))
IDS=CONTROLS+HYBRIDS+BEAMS

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(),parse_float=Decimal)
def dec(v):return v if isinstance(v,Decimal) else Decimal(v)
def avg(v):return sum(map(dec,v),Decimal(0))/Decimal(len(v)) if v else None
def plain(x):
    if isinstance(x,Decimal):return float(x)
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

bindings={};verified=Counter()
def bind(path, expected=None, category='direct'):
    p=path if path.is_absolute() else ROOT/path
    h=sha(p)
    if expected is not None:assert h==expected,(str(p),'hash mismatch')
    label=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if label in bindings:assert bindings[label]==h
    else:bindings[label]=h;verified[category]+=1
    return p

with localcontext() as ctx:
    ctx.prec=60
    result=read(bind(OUT/'results.json'))
    registration=read(bind(OUT/'registration.json'))
    pre=read(bind(OUT/'preflight.json',registration['preflight_sha256']))
    cfg=registration['config']
    assert cfg['development_validation_folds']==list(FOLDS)
    assert tuple(cfg['control_ids'])==CONTROLS and tuple(cfg['diagnostic_ids'])==HYBRIDS+BEAMS
    assert result['new_models_fitted']==result['new_causal_candidates']==0
    assert not result['selection_performed'] and not result['teacher_use_allowed'] and not result['test_periods_used']
    for p,h in cfg['source_bindings'].items():bind(Path(p),h,'registered_source')
    for p,h in cfg['source_manifest_bindings'].items():bind(Path(p),h,'parent_manifest')
    bind(Path(cfg['parity_config']),cfg['parity_config_sha256'],'parent_config')
    rows=result['rows']; index={(x['fold'],x['candidate_id']):x for x in rows}
    assert len(rows)==len(index)==224 and set(index)=={(f,c) for f in FOLDS for c in IDS}
    regime={f:index[f,'bh']['regime'] for f in FOLDS}
    for x in rows:
        assert x['regime']==regime[x['fold']]
        assert x['hindsight_only']==(x['candidate_id'] in HYBRIDS+BEAMS)
        for cost in COSTS:
            assert all(dec(x[cost][k]).is_finite() for k in x[cost])
    counts=dict(Counter(v['trend'] for v in regime.values()))
    assert counts=={'bull':2,'bear':4,'sideways':2}
    artifacts={}
    for f in FOLDS:
        fold=read(bind(OUT/f'fold_{f}.json',category='fold_manifest'))
        assert fold['rows']==[x for x in rows if x['fold']==f]
        assert fold['registration_sha256']==result['registration_sha256']
        assert len(fold['artifact_sha256'])==50
        for p,h in fold['artifact_sha256'].items():
            assert p not in artifacts;artifacts[p]=h
            bind(Path(p),h,'new_artifact')
    assert len(artifacts)==400
    def fs(g):return [f for f in FOLDS if g=='all' or regime[f]['trend']==g]
    def cid_summary(cid,keys=KEYS):
        return {g:{'quarters':len(fs(g)),**{cost:{k:avg([index[f,cid][cost][k] for f in fs(g)]) for k in keys} for cost in COSTS}} for g in STRATA}
    policies={cid:cid_summary(cid) for cid in IDS}
    pairs={}
    for m in HALVES:
        for swap in SWAPS:
            for rule in RULES:
                cid=m+'_oracle_'+swap+'_'+rule;ref=m+'_'+rule
                pairs[cid]={'reference_id':ref,'strata':{g:{'quarters':len(fs(g)),**{cost:{k:avg([dec(index[f,cid][cost][k])-dec(index[f,ref][cost][k]) for f in fs(g)]) for k in KEYS} for cost in COSTS}} for g in STRATA}}
    errors={};maxima={}
    def compare(a,b,path,category):
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
            for k in a:compare(a[k],b[k],path+'/'+k,category)
        elif isinstance(a,(Decimal,int)) and not isinstance(a,bool):
            d=abs(dec(a)-dec(b));errors[category]=max(errors.get(category,Decimal(0)),d)
            if category not in maxima or d>maxima[category]['absolute_difference']:
                maxima[category]={'path':path,'absolute_difference':d}
        else:assert a==b,(path,a,b)
    compare(policies,result['summary']['policies'],'policies','policy_equal_quarter_means')
    compare(pairs,result['summary']['oracle_minus_own_learned'],'pairs','paired_equal_quarter_means')
    assert all(x<Decimal('1e-12') for x in errors.values())
    # Existing numerical paths only: no target generation or economic replay.
    matched_paths=[]
    for f in FOLDS:
        for swap in ('return','both'):
            for rule in RULES:
                cid1=HALVES[0]+'_oracle_'+swap+'_'+rule;cid2=HALVES[1]+'_oracle_'+swap+'_'+rule
                p1=OUT/'targets'/f'fold{f}_{cid1}.npz';p2=OUT/'targets'/f'fold{f}_{cid2}.npz'
                with np.load(p1,allow_pickle=False) as a,np.load(p2,allow_pickle=False) as b:
                    equal=all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
                matched_paths.append({'fold':f,'swap':swap,'rule':rule,'exact_targets_equal':equal})
    rl={};objective_error=0.;objective_same_lambda_comparisons={}
    for cid in BEAMS:
        penalty=int(cid[-1]); per_fold=[]
        for f in FOLDS:
            p=OUT/'traces'/f'fold{f}_{cid}.json';t=read(p);r=index[f,cid]
            assert sha(p)==r['diagnostic_sha256'] and t['metrics']==r['base']
            j0=math.log1p(float(r['base']['total_return']));j1=j0-float(r['base']['maxdd'])
            objective=j0-penalty*float(r['base']['maxdd'])
            objective_error=max(objective_error,abs(objective-float(t['objective'])))
            assert t['future_information_used'] and not t['global_optimum_claimed'] and not t['deployable']
            assert t['bound_direction']=='lower_bound_on_maximum_attainable_hindsight_objective'
            assert t['pruned_distinct_branches']>0 and not t['exhaustive_for_matched_dynamic_action_set']
            if t['incumbent_objective'] is not None:assert objective+1e-12>=float(t['incumbent_objective'])
            per_fold.append({'fold':f,'log_terminal_nav':j0,'log_terminal_nav_minus_maxdd':j1,
                'objective':objective,'maxdd':r['base']['maxdd'],'pruned_distinct_branches':t['pruned_distinct_branches'],
                'incumbent_selected':t['incumbent_selected']})
        rl[cid]={'base_and_stress':cid_summary(cid,KEYS+('total_return','maxdd','fees_initial_equity_units','borrow_initial_equity_units')),
            'base_common_objective_means':{'lambda0':math.fsum(x['log_terminal_nav'] for x in per_fold)/8,
                'lambda1':math.fsum(x['log_terminal_nav_minus_maxdd'] for x in per_fold)/8},'per_fold':per_fold}
    for rule in ('hold','fallback_bh'):
        c0=f'matched_rl_beam32_{rule}_risk0';c1=f'matched_rl_beam32_{rule}_risk1'
        objective_same_lambda_comparisons[rule]={'risk1_minus_risk0':{cost:{k:avg([dec(index[f,c1][cost][k])-dec(index[f,c0][cost][k]) for f in FOLDS]) for k in ('alpha_ex','maxdd','maxdd_delta','turnover','trades')} for cost in COSTS},
            'common_lambda0_difference':rl[c1]['base_common_objective_means']['lambda0']-rl[c0]['base_common_objective_means']['lambda0'],
            'common_lambda1_difference':rl[c1]['base_common_objective_means']['lambda1']-rl[c0]['base_common_objective_means']['lambda1'],
            'per_fold_maxdd_increased': [f for f in FOLDS if index[f,c1]['base']['maxdd']>index[f,c0]['base']['maxdd']],
            'per_fold_lambda1_objective_worse': [a['fold'] for a,b in zip(rl[c1]['per_fold'],rl[c0]['per_fold']) if a['log_terminal_nav_minus_maxdd']<b['log_terminal_nav_minus_maxdd']-1e-12]}
    swaps={}
    for m in HALVES:
        for rule in RULES:
            group=m+'_'+rule; part={}
            for swap in SWAPS:
                cid=m+'_oracle_'+swap+'_'+rule
                part[swap]={'levels':policies[cid],'minus_own_learned':pairs[cid]['strata'],
                    'quarter_paired_sign_counts':{cost:{'alpha_improved':sum(index[f,cid][cost]['alpha_ex']>index[f,group][cost]['alpha_ex'] for f in FOLDS),
                        'dd_improved':sum(index[f,cid][cost]['maxdd_delta']<index[f,group][cost]['maxdd_delta'] for f in FOLDS),
                        'joint_improved':sum(index[f,cid][cost]['alpha_ex']>index[f,group][cost]['alpha_ex'] and index[f,cid][cost]['maxdd_delta']<index[f,group][cost]['maxdd_delta'] for f in FOLDS)} for cost in COSTS}}
            both=m+'_oracle_both_'+rule;ret=m+'_oracle_return_'+rule;rv=m+'_oracle_realized_risk_'+rule
            part['both_minus_return']={cost:{k:avg([dec(index[f,both][cost][k])-dec(index[f,ret][cost][k]) for f in FOLDS]) for k in KEYS} for cost in COSTS}
            part['descriptive_factorial_interaction']={cost:{k:avg([dec(index[f,both][cost][k])-dec(index[f,ret][cost][k])-dec(index[f,rv][cost][k])+dec(index[f,group][cost][k]) for f in FOLDS]) for k in KEYS} for cost in COSTS}
            swaps[group]=part
    report={'schema':'independent-oracle-information-decomposition-summary-audit-v1',
        'scope':'Saved original development folds5-12 only; Decimal60 independent aggregation; no repository helper imports, fits, policy rollouts, later test reads, selection or inference.',
        'passed':True,'source_revision':registration['source_revision'],
        'audit_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
        'inventory':{'folds':list(FOLDS),'regime_counts':counts,'controls':12,'hybrid_diagnostics':12,'beam_diagnostics':4,'economic_rows':224,'cost_accounts':448,'new_artifacts':400},
        'verified_binding_counts':dict(verified),'source_sha256':bindings,
        'ancestor_binding_scope':'Preflight with 1328 ancestral bindings is itself hash-bound; ancestral artifacts not independently rehashed in this summary-only audit.',
        'numeric_max_absolute_differences':{**errors,'rl_objective_from_saved_account':objective_error},'maximum_difference_locations':maxima,
        'equal_quarter_policies':policies,'paired_hybrids':pairs,'intervention_interpretation_data':swaps,
        'beam_interpretation_data':rl,'beam_penalty_tradeoffs':objective_same_lambda_comparisons,
        'return_and_both_technical_perpetual_existing_target_comparisons':matched_paths,
        'all32_return_both_target_pairs_coincide':all(x['exact_targets_equal'] for x in matched_paths),
        'bh_reference_max_absolute_roundoff':max(abs(dec(index[f,'bh'][cost][k])) for f in FOLDS for cost in COSTS for k in ('alpha_ex','maxdd_delta')),
        'limitations':['All eight quarters are repeatedly reused, with only2bull/4bear/2sideways quarters.',
            'Return substitutions reveal realized outcomes, not attainable conditional means; RV substitutions are noisy realized quadratic-variation targets.',
            'Only pre-existing score support is replaced. Learned remainder and missing-input rules stay fixed; paths evolve in their own inventory.',
            'Observed headroom does not distinguish absent causal signal, model misspecification, calibration or intrinsically unpredictable returns.',
            'Beam reads the full future path and uses a different objective/horizon. All32 searches prune and cannot certify a global upper bound.',
            'No new causal forecast accuracy, generalization probability, causal model selection or training-teacher authorization follows from this audit.']}
    REPORT.write_text(json.dumps(plain(report),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'output':str(REPORT),'sha256':sha(REPORT),'passed':True,'verified_binding_counts':dict(verified),'max_differences':plain(report['numeric_max_absolute_differences']),
        'beam_tradeoffs':plain(objective_same_lambda_comparisons),'all32_target_pairs_coincide':report['all32_return_both_target_pairs_coincide']},ensure_ascii=False))
