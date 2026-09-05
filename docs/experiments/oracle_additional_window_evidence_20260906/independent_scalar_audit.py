"""Independent scalar audit of all fixed reused test15-24 outputs; no fitting."""
from pathlib import Path
from collections import Counter
import json, hashlib, math
import numpy as np
import pandas as pd
import yaml

def scalar_account(frame, targets, contract, forecast=None, inference=None, fallback_enabled=True):
    fee_rate = contract['one_way_cost']
    annual = contract['borrow_annual']
    step = contract['max_step']
    deadband = contract['deadband']
    op = frame.open.to_numpy()
    cl = frame.close.to_numpy()
    schedule = (frame.index.hour % 6 == 0) & (frame.index.minute == 0)
    cash = 0.0
    units = 1.0 / float(op[0])
    equity = []
    exposure = []
    turnover = fees = borrow = 0.0
    trades = 0
    planned = np.full(len(frame), np.nan)
    trace = []
    submitted = planned if forecast is not None else targets
    for t in range(len(frame)):
        price = float(op[t])
        mark = float(cl[t])
        known = math.isfinite(price)
        nav = cash + units * price if known else math.nan
        if known:
            assert nav > 0
        if known and t and schedule[t - 1] and math.isfinite(submitted[t - 1]):
            current = units * price / nav
            intent = min(max(float(submitted[t - 1]), 0.0), 1.12)
            change = max(-step, min(step, intent - current))
            if abs(change) >= deadband:
                desired = current + change
                trade = (desired * nav - units * price) / (1 + fee_rate * desired * (1 if change > 0 else -1))
                fee = fee_rate * abs(trade)
                cash -= trade + fee
                units += trade / price
                turnover += abs(trade) / nav
                fees += fee
                trades += 1
        if forecast is not None and schedule[t] and known and (inference[t] or fallback_enabled):
            nav = cash + units * price
            asset = units * price
            current = asset / nav
            if inference[t]:
                mu = float(forecast['mu'][t])
                variance = float(forecast['variance'][t])
                assert math.isfinite(mu) and math.isfinite(variance) and (variance >= 0)
                best = 0.0
                chosen = math.nan
                estimated_turnover = 0.0
                reason = 'learned'
                for delta in (-0.08, -0.04, 0.04, 0.08):
                    intent = min(max(current + delta, 0.5), 1.12)
                    change = max(-step, min(step, intent - current))
                    desired = current + change
                    if change == 0 or abs(change) < deadband:
                        continue
                    trade = (desired * nav - asset) / (1 + fee_rate * desired * (1 if change > 0 else -1))
                    tv = abs(trade) / nav
                    score = (desired - current) * mu - 0.5 * (desired * desired - current * current) * variance - 2 * fee_rate * tv - (max(desired - 1, 0) - max(current - 1, 0)) * annual * 24 / 35040
                    if score > best:
                        best = score
                        chosen = intent
                        estimated_turnover = tv
            else:
                chosen = 1.0
                best = math.nan
                estimated_turnover = math.nan
                reason = 'forecast_unavailable'
            planned[t] = chosen
            trace.append((t, nav, current, best, estimated_turnover, chosen, reason))
        if cash < 0:
            charge = -cash * (math.exp(annual / 35040) - 1)
            cash -= charge
            borrow += charge
        if math.isfinite(mark):
            value = cash + units * mark
            assert value > 0
            equity.append(value)
            exposure.append(units * mark / value)

    def dd(values):
        peak = values[0]
        worst = 0.0
        for value in values:
            peak = max(peak, value)
            worst = max(worst, 1 - value / peak)
        return worst
    benchmark = [1.0] + [float(v) / float(op[0]) for v in cl if math.isfinite(v)]
    mdd = dd([1.0] + equity)
    bhdd = dd(benchmark)
    result = {'alpha_ex': equity[-1] - benchmark[-1], 'maxdd_delta': mdd - bhdd, 'maxdd': mdd, 'bh_maxdd': bhdd, 'total_return': equity[-1] - 1, 'bh_total_return': benchmark[-1] - 1, 'turnover': turnover, 'trades': trades, 'fees_initial_equity_units': fees, 'borrow_initial_equity_units': borrow, 'mean_exposure': math.fsum(exposure) / len(exposure), 'rows': len(frame), 'close_coverage': sum((math.isfinite(v) for v in cl)) / len(frame), 'bar_coverage': float(frame.bar_available.mean()), 'intent_coverage': sum((math.isfinite(v) for v in submitted)) / len(frame)}
    return (result, planned, trace)

def mean(x):
    return math.fsum(x) / len(x)

def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    j = 0
    while j < len(v):
        k = j + 1
        while k < len(v) and v[order[k]] == v[order[j]]:
            k += 1
        for t in range(j, k):
            r[order[t]] = (j + 1 + k) / 2
        j = k
    return r

def spearman(x, y):
    a, b = (rank(x), rank(y))
    am, bm = (mean(a), mean(b))
    aa = [v - am for v in a]
    bb = [v - bm for v in b]
    den = math.sqrt(math.fsum((v * v for v in aa)) * math.fsum((v * v for v in bb)))
    return math.fsum((x * y for x, y in zip(aa, bb))) / den if den else None

def scalar_scores(arrays):
    mask = arrays['score_support']
    observed, predicted = (arrays['actual'][mask, 0], arrays['mu'][mask])
    n = len(observed)
    if n == 0 or not np.isfinite(observed).all() or (not np.isfinite(predicted).all()):
        raise ValueError('empty or nonfinite scored values')
    fit = float(arrays['fit_return_mean'])
    losses = [float(a) - float(b) for a, b in zip(observed, predicted)]
    result = {'rows': n, 'return_mse': math.fsum((v * v for v in losses)) / n, 'return_mae': math.fsum((abs(v) for v in losses)) / n, 'return_sign_accuracy': sum(((a > 0) == (b > 0) for a, b in zip(observed.tolist(), predicted.tolist()))) / n, 'zero_return_mse': math.fsum((float(v) ** 2 for v in observed)) / n, 'fit_mean_return_mse': math.fsum(((float(v) - fit) ** 2 for v in observed)) / n, 'return_rank_ic': None}
    if len(set(observed.tolist())) > 1 and len(set(predicted.tolist())) > 1:
        center = (n + 1) / 2
        a = [v - center for v in ranks(observed)]
        b = [v - center for v in ranks(predicted)]
        result['return_rank_ic'] = math.fsum((x * y for x, y in zip(a, b))) / math.sqrt(math.fsum((x * x for x in a)) * math.fsum((y * y for y in b)))
    return result

ranks = rank

ROOT = Path('codex_outputs/oracle_additional_window_replay_v1')
FOLDS = tuple(range(15, 25))
MEANS = ('scale_mean', 'technical_scaled', 'perp_delay0_scaled', 'technical_half', 'perp_delay0_half')
RULES = ('utility_risk1', 'utility_risk1_fallback_bh')
POLICIES = ('bh', 'common_robust') + tuple(m+'_'+r for m in MEANS for r in RULES)

def audit():
    bindings, maxima, counts = {}, {}, Counter()
    def bind(path, expected=None):
        path=Path(path); value=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None: assert value == expected, str(path)
        if str(path) in bindings: assert bindings[str(path)]==value
        bindings[str(path)]=value
        return path
    def read(path, expected=None): return json.loads(bind(path,expected).read_text())
    def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def check(kind,a,b,exact=False):
        if a is None or b is None: assert a is b; return
        aa,bb=np.asarray(a),np.asarray(b)
        assert aa.shape==bb.shape, kind
        assert np.array_equal(np.isnan(aa),np.isnan(bb)),kind
        assert not np.isinf(aa).any() and not np.isinf(bb).any(),kind
        assert np.array_equal(aa,bb,equal_nan=True) if exact else np.allclose(aa,bb,rtol=1e-12,atol=1e-12,equal_nan=True),kind
        valid=np.isfinite(aa)&np.isfinite(bb)
        error=float(np.max(np.abs(aa[valid].astype(float)-bb[valid].astype(float)))) if valid.any() else 0.
        maxima[kind]=max(maxima.get(kind,0.),error)
    cfgpath=Path('configs/oracle_additional_window_replay_20260906.yaml')
    cfg=yaml.safe_load(bind(cfgpath).read_text())
    reg=read(ROOT/'registration.json'); result=read(ROOT/'results.json')
    assert result['registration_sha256']==digest(reg)
    assert reg['config']==cfg and reg['config_sha256']==bindings[str(cfgpath)]
    pre=read(ROOT/'preflight.json',cfg['preflight_sha256'])
    assert pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'})
    for p,h in cfg['source_bindings'].items(): bind(p,h)
    manifest=read(cfg['data_manifest_path'],cfg['data_manifest_sha256'])
    for p,h in manifest['bindings'].items(): bind(p,h)
    fc=yaml.safe_load(bind(cfg['source_config'],cfg['source_config_sha256']).read_text())
    family=read(cfg['family_path'],cfg['family_sha256'])
    assert family['candidate_ids']==cfg['candidate_ids'] and family['control_ids']==cfg['control_ids']
    rows={(r['fold'],r['candidate_id']):r for r in result['rows']}
    scores={(r['fold'],r['mean_id']):r for r in result['scores']}
    assert len(rows)==len(result['rows'])==120 and set(rows)=={(f,c) for f in FOLDS for c in POLICIES}
    assert len(scores)==len(result['scores'])==50 and set(scores)=={(f,m) for f in FOLDS for m in MEANS}
    raw=pd.read_parquet(cfg['spot_path'])
    raw=raw.loc[raw.index<pd.Timestamp(cfg['data_cutoff'])]
    raw['bar_available']=raw[['open','high','low','close']].notna().all(axis=1)
    for f in FOLDS:
        start=pd.Timestamp('2020-04-16T13:45:00Z')+pd.DateOffset(months=3*f)
        end=start+pd.DateOffset(months=3)
        frame=raw.loc[(raw.index>=start)&(raw.index<end)]
        assert frame.index.equals(pd.date_range(start,end,freq='15min',inclusive='left'))
        fold=read(ROOT/f'fold_{f}.json')
        assert fold['registration_sha256']==digest(reg)
        assert fold['rows']==[r for r in result['rows'] if r['fold']==f]
        assert fold['scores']==[r for r in result['scores'] if r['fold']==f]
        assert len(fold['artifact_sha256'])==33
        for p,h in fold['artifact_sha256'].items():bind(p,h)
        predictions={}
        for m in MEANS:
            p=ROOT/'forecasts'/f'fold{f}_{m}.npz'
            with np.load(bind(p,scores[f,m]['forecast_sha256']),allow_pickle=False) as saved:
                assert set(saved.files)=={'timestamps','mu','variance','inference_mask','score_support','actual','fit_return_mean'}
                pred={k:saved[k] for k in saved.files}
            predictions[m]=pred
            check('forecast_calendar',pred['timestamps'],frame.index.asi8,True)
            assert pred['inference_mask'].dtype==bool and pred['score_support'].dtype==bool
            assert not np.any(pred['score_support']&~pred['inference_mask'])
            assert np.isnan(pred['mu'][~pred['inference_mask']]).all()
            assert np.isnan(pred['actual'][~pred['score_support']]).all()
            for key,value in scalar_scores(pred).items():check('scalar_score_'+key,value,scores[f,m][key])
            for key in ('timestamps','inference_mask','score_support','actual','variance','fit_return_mean'):
                check('common_'+key,pred[key],predictions['scale_mean'][key],True)
            counts['forecast_scores']+=1
        for half,full in [('technical_half','technical_scaled'),('perp_delay0_half','perp_delay0_scaled')]:
            expected=.5*predictions['scale_mean']['mu']+.5*predictions[full]['mu']
            check('half_formula',predictions[half]['mu'],expected,True)
            check('half_rank_ic',scores[f,half]['return_rank_ic'],scores[f,full]['return_rank_ic'])
            counts['half_forecasts']+=1
        for cid in POLICIES:
            row=rows[f,cid];p=ROOT/'targets'/f'fold{f}_{cid}.npz'
            with np.load(bind(p,row['targets_sha256']),allow_pickle=False) as saved:
                assert set(saved.files)=={'timestamps','targets'}
                check('target_calendar',saved['timestamps'],frame.index.asi8,True)
                target=saved['targets'].copy()
            if cid not in ('bh','common_robust'):
                rule=RULES[1] if cid.endswith(RULES[1]) else RULES[0]
                m=cid[:-(len(rule)+1)];pred=predictions[m]
                account,planned,tr=scalar_account(frame,target,fc['execution'],pred,pred['inference_mask'],fallback_enabled=rule==RULES[1])
                check('scalar_own_state_targets',planned,target)
                trace=read(ROOT/'traces'/f'fold{f}_{cid}.json')
                dt=trace['decision_trace']
                check('trace_indices',dt['bar_indices'],[r[0] for r in tr],True)
                for column,key in [(1,'known_open_nav'),(2,'known_open_exposure'),(3,'estimated_utility_gain_over_hold'),(4,'estimated_trade_turnover')]:
                    check('trace_'+key,np.array(dt[key],dtype=float),[r[column] for r in tr])
                for key,value in account.items():check('trace_account_'+key,value,trace['metrics'][key])
                counts['own_state_paths']+=1;counts['own_state_decisions']+=len(tr)
            elif cid=='bh': assert np.isnan(target).all()
            for cost,factor in [('base',1),('stress_2x',2)]:
                contract={**fc['execution'],'one_way_cost':factor*fc['execution']['one_way_cost'],'borrow_annual':factor*fc['execution']['borrow_annual']}
                account,_,_=scalar_account(frame,target,contract)
                for key,value in account.items():check('scalar_account_'+key,value,row[cost][key])
                counts['account_paths']+=1
        counts['folds']+=1
        print(json.dumps({'event':'independent_scalar_fold_pass','fold':f}),flush=True)
    assert counts['own_state_paths']==100 and counts['account_paths']==240 and counts['forecast_scores']==50
    output={'schema':'oracle-additional-window-independent-scalar-audit-v1','status':'pass','source_revision':reg['source_revision'],'source_sha256':bindings,'counts':dict(counts),'max_absolute_differences':maxima,'scope':'All10 originaltest quarters,50 saved forecast scalar scores,20 affine half forecasts,100 independent own-state decisions and240 independent base/stress account paths; no fit or canonical planner/account/scoring helper calls.','limitations':['Reused historical periods; no independent confirmation or model selection.','Saved prediction correctness is checked by loss/half geometry, not independent Ridge/HGB training here.','Common robust targets are accounted independently but their original feature rule is not rebuilt by this scalar audit.','No historical receipt or live execution claim.']}
    dest=Path('/tmp/oracle_additional_window_scalar_audit_20260906.json')
    dest.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'path':str(dest),'counts':dict(counts),'maximum_difference':max(maxima.values())}),flush=True)

if __name__=='__main__':audit()
