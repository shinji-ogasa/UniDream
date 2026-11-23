"""Causal controls and report-only ten-seed robustness evaluation.

Parameters are fixed in the new registration, never selected from these Test
outcomes. The eight-percent comparator is a post-hoc gross-exposure reference,
not exact risk matching. The historical cohort contains gaps/free inventory
resets and is not a continuous self-financing backtest or an unused holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_inference_robust_bootstrap import infer_daily
from .wm_inference_robust_hac import hac_mean
from .wm_policy_ablation_evaluation import accounting, bounded_requests, execute_positions, reset_mask, score_path
from .wm_rl_release_evaluation import _path, _metrics
from .wm_significance_statistics import holm_fixed_family


CONDITIONS = ("C_floor_gradual", "A_floor_gradual", "C_control")
BASELINES = ("bh", "constant08", "ma20d", "voltarget10")
BAR_NS = 900 * 10**9
COHORT = ("fold", "decision_id", "timestamp", "bar_log_return")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _baseline_cfg(cfg):
    return {**cfg, "accounting": {**cfg.get("accounting", {}), "position_min": 0.}}


def condition_configs(cfg):
    return {condition: {**cfg, "accounting": {**cfg.get("accounting", {}),
        "position_min": .5 if condition == "C_control" else 0.}} for condition in CONDITIONS}


def _prices(raw_close, raw_index_ns):
    close = np.asarray(raw_close, np.float64)
    times = np.asarray(raw_index_ns)
    if (close.ndim != 1 or times.shape != close.shape or len(close) < 2
            or not np.issubdtype(times.dtype, np.integer) or np.any(np.diff(times) != BAR_NS)):
        raise ValueError("raw prices must use the complete regular integer-timestamp 15-minute grid")
    return close, times.astype(np.int64, copy=False)


def _past_sum(values, ids, window):
    valid = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    missing = np.r_[0, np.cumsum(~valid)]
    if np.any(ids < window) or np.any(missing[ids] - missing[ids - window]):
        raise ValueError("registered baseline requires complete causal raw history; no fill or cohort compression")
    return sums[ids] - sums[ids - window]


def baseline_targets(raw_close, raw_index_ns, decision_ids, cfg):
    """Signals at d use only raw close <=d-1, even across cohort omissions.

    SMA uses close[d-W:d]. Volatility uses squared raw log returns[d-W:d],
    where return[j]=log(close[j]/close[j-1]); no mean subtraction. Annualized
    sigma=sqrt(mean(r**2)*annual_bars). A zero sigma chooses the fixed cap.
    These are desired targets; rate cap and execution lag are applied later.
    """
    close, _ = _prices(raw_close, raw_index_ns)
    ids = np.asarray(decision_ids)
    settings = cfg["baselines"]
    ma = settings["ma_window_bars"]
    vol = settings["volatility_window_bars"]
    target_sigma = float(settings["volatility_target_annual"])
    constant = float(settings["constant_exposure"])
    a = accounting(_baseline_cfg(cfg))
    if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 1) or np.any(ids >= len(close)) or np.any(np.diff(ids) <= 0)
            or int(ma) != ma or int(vol) != vol or min(ma, vol) < 1
            or not np.isfinite([target_sigma, constant]).all() or target_sigma <= 0
            or not 0 <= constant <= a["position_max"]):
        raise ValueError("valid registered causal windows, targets and unique chronological decisions required")
    ma, vol = int(ma), int(vol)
    clean_close = np.where(np.isfinite(close) & (close > 0), close, np.nan)
    sma = _past_sum(clean_close, ids, ma) / ma
    returns = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns[1:] = np.log(clean_close[1:] / clean_close[:-1])
    rv = _past_sum(returns**2, ids, vol) / vol
    sigma = np.sqrt(np.maximum(rv, 0.) * a["annual_bars"])
    vol_target = np.divide(target_sigma, sigma, out=np.full(len(ids), a["position_max"]), where=sigma > 0)
    return pd.DataFrame({"decision_id": ids, "bh": 1., "constant08": constant,
        "ma20d": (clean_close[ids - 1] > sma).astype(float),
        "voltarget10": np.clip(vol_target, 0., a["position_max"]),
        "causal_sma": sma, "causal_annual_sigma": sigma})


def build_baselines(raw_close, raw_index_ns, cohort_frame, cfg):
    """Return executed controls, metrics and a causal/accounting audit."""
    close, times = _prices(raw_close, raw_index_ns)
    if not set(COHORT).issubset(cohort_frame) or cohort_frame.empty:
        raise ValueError("nonempty full market cohort required")
    cohort = cohort_frame[list(COHORT)].copy()
    cohort["timestamp"] = pd.to_datetime(cohort.timestamp, utc=True, errors="coerce")
    cohort = cohort.sort_values("timestamp").reset_index(drop=True)
    ids = cohort.decision_id.to_numpy()
    if (not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 1) or np.any(ids >= len(close))
            or not np.array_equal(times[ids], cohort.timestamp.astype("int64").to_numpy())
            or cohort.duplicated(["fold", "decision_id"]).any()):
        raise ValueError("cohort decisions must match the original raw timestamp grid exactly")
    actual = np.log(close[ids] / close[ids - 1])
    if not np.isfinite(actual).all() or not np.array_equal(actual, cohort.bar_log_return.to_numpy(float)):
        raise ValueError("cohort bar returns must equal the immutable raw close returns")
    desired = baseline_targets(close, times, ids, cfg)
    base_cfg = _baseline_cfg(cfg)
    paths, scores = [], []
    for baseline in BASELINES:
        request = bounded_requests(desired[baseline], cohort.timestamp, cohort.fold, base_cfg)
        path = execute_positions(cohort.bar_log_return, request, cohort.timestamp, cohort.fold, base_cfg)
        path["decision_id"] = ids
        path["baseline"] = baseline
        path["desired_position"] = desired[baseline].to_numpy()
        paths.append(path)
        scores.append({"baseline": baseline, "control_floor_feasible_for_all_targets": bool(desired[baseline].ge(.5).all()),
                       **_metrics(path, base_cfg)})
    return pd.concat(paths, ignore_index=True), pd.DataFrame(scores), {
        "signal_information": "raw close through decision d-1 only; full raw causal history across cohort gaps",
        "ma_rule": "close[d-1] > mean(close[d-W:d]) ->1 else0; equality->0",
        "vol_rule": "min(cap,target_sigma/sqrt(mean(raw_return[d-W:d]**2)*annual_bars)); sigma0->cap",
        "execution": "causal target ->same rate cap ->one-bar request lag ->same fee/borrow",
        "constant08_scope": "user-selected post-hoc gross-exposure approximation, not exact risk matching",
        "n_unique_market_rows": len(cohort), "n_reset_segments": int(reset_mask(cohort.timestamp, cohort.fold).sum()),
        "baseline_parameters": dict(cfg["baselines"]), "full_calendar_backtest": False,
        "cohort_bh_return":float(np.expm1(cohort.bar_log_return.sum())),
        "full_span_endpoint_bh_reference_return":float(close[ids[-1]]/close[ids[0]-1]-1),
        "full_span_raw_bars":int(ids[-1]-ids[0]+1), "omitted_within_span_bars":int(ids[-1]-ids[0]+1-len(ids))}


def _summary(frame, keys):
    rows = []
    for group, part in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(keys, group))
        row.update(n_seeds=len(part), n_unique_market_rows=int(part.n_rows.iloc[0]))
        columns = [c for c in part.select_dtypes(include=np.number) if c not in {*keys, "seed", "n_rows", "n_resets"}]
        for column in columns:
            values = part[column]
            for suffix, value in (("mean", values.mean()), ("median", values.median()),
                    ("std", values.std(ddof=1)), ("min", values.min()), ("max", values.max())):
                row[column + "_" + suffix] = float(value) if pd.notna(value) else None
        worst_dd = part.loc[part.max_drawdown.idxmax()]
        worst_return = part.loc[part.net_total_return.idxmin()]
        row.update(worst_drawdown_seed=int(worst_dd.seed), worst_drawdown=float(worst_dd.max_drawdown),
                   worst_return_seed=int(worst_return.seed), worst_return=float(worst_return.net_total_return))
        rows.append(row)
    return pd.DataFrame(rows)


def _definitions():
    rows = [(f"{condition}_minus_{baseline}", "baseline_primary", condition, baseline)
            for condition in CONDITIONS for baseline in BASELINES]
    return rows + [("C_floor_gradual_minus_A_floor_gradual", "policy_secondary", "C_floor_gradual", "A_floor_gradual"),
                   ("C_floor_gradual_minus_C_control", "policy_secondary", "C_floor_gradual", "C_control")]


def _correct(frame, *, hac=False):
    frame = frame.copy()
    frame["p_holm_family"] = np.nan
    frame["p_holm_all14"] = np.nan
    grouping = ["lag_days"] if hac else ["method", "block_days", "estimand"]
    for _, panel in frame.groupby(grouping, sort=False):
        all_p = {str(i): float(row.p_value) if pd.notna(row.p_value) else 1. for i, row in panel.iterrows()}
        q14 = holm_fixed_family(all_p, family_size=14)
        for family, part in panel.groupby("family", sort=False):
            size = 12 if family == "baseline_primary" else 2
            if len(part) != size:
                raise ValueError("the full fixed12+2 hypothesis family must be retained")
            corrected = holm_fixed_family({str(i): all_p[str(i)] for i in part.index}, family_size=size)
            for i in part.index:
                frame.loc[i, "p_holm_family"] = corrected[str(i)]
                frame.loc[i, "p_holm_all14"] = q14[str(i)]
    frame["significant_difference"] = frame.primary_inference & frame.p_holm_family.lt(.05)
    frame["significant_improvement"] = frame.significant_difference & frame.estimate.gt(0)
    return frame


def evaluate_ledgers(frame, baseline_ledger, cfg, configs):
    """Aggregate frozen seed paths, with paired market/seed uncertainty."""
    seeds, folds = list(cfg["seeds"]), list(cfg["folds"])
    if list(cfg["conditions"]) != list(CONDITIONS) or len(set(seeds)) != len(seeds):
        raise ValueError("the complete registered conditions and distinct training seeds required")
    reused,new=set(cfg.get("reused_seeds",[])),set(cfg.get("new_seeds",[]))
    if (reused or new) and (reused & new or reused | new != set(seeds)):
        raise ValueError("reused and new seed groups must be a disjoint complete partition")
    expected = {(fold, seed, condition) for fold in folds for seed in seeds for condition in CONDITIONS}
    if set(frame[["fold", "seed", "condition"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("every registered seed/fold/condition must share the common cohort")
    reference, paths, pooled, monthly, quarterly, folded = None, {}, [], [], [], []
    for condition in CONDITIONS:
        for seed in seeds:
            path = _path(frame.loc[frame.condition.eq(condition) & frame.seed.eq(seed)], configs[condition])
            if reference is not None and not path[list(COHORT)].equals(reference):
                raise ValueError("all seeds and policies must share exactly the same market bars")
            reference = path[list(COHORT)]
            paths[(condition, seed)] = path
            identity = {"condition": condition, "seed": seed}
            pooled.append({**identity, **_metrics(path, configs[condition])})
            for fold, part in path.groupby("fold", sort=True):
                folded.append({**identity, "fold": fold, **_metrics(part, configs[condition])})
            for frequency, destination in (("M", monthly), ("Q", quarterly)):
                labels = path.timestamp.dt.tz_localize(None).dt.to_period(frequency).astype(str)
                for period, part in path.groupby(labels, sort=True):
                    destination.append({**identity, "period": period, **_metrics(part, configs[condition])})
    if len(reference) != cfg["statistics"].get("expected_market_rows", len(reference)):
        raise ValueError("registered market support changed")
    base_paths = {}
    for baseline in BASELINES:
        path = _path(baseline_ledger.loc[baseline_ledger.baseline.eq(baseline)], _baseline_cfg(cfg))
        if not path[list(COHORT)].equals(reference):
            raise ValueError("every baseline must score the same complete policy cohort")
        base_paths[baseline] = path
    if set(baseline_ledger.baseline) != set(BASELINES):
        raise ValueError("all and only the four registered controls required")
    resets = reset_mask(reference.timestamp, reference.fold)
    daily = pd.DataFrame({"fold": reference.fold, "segment": np.cumsum(resets)-1,
                          "day": reference.timestamp.dt.floor("D"), "count": 1})
    grouped = daily.groupby(["fold", "segment", "day"], sort=True).sum().reset_index()
    definitions = _definitions()
    sums = np.empty((len(grouped), 14, len(seeds)))
    for j, (_, _, left, right) in enumerate(definitions):
        for s, seed in enumerate(seeds):
            lhs = paths[(left, seed)]
            rhs = base_paths[right] if right in BASELINES else paths[(right, seed)]
            diff = lhs.net_log_return.to_numpy() - rhs.net_log_return.to_numpy()
            sums[:, j, s] = daily.assign(value=diff).groupby(["fold", "segment", "day"], sort=True).value.sum().to_numpy()
    family = {name: role for name, role, _, _ in definitions}
    names = [row[0] for row in definitions]
    day_ids = grouped.day.astype("int64").to_numpy() // (86400*10**9)
    result = infer_daily(sums, grouped["count"].to_numpy(), grouped.segment.to_numpy(),
                         {**cfg, "condition_names": names}, day_ids=day_ids)
    inference = _correct(result["inference"].assign(family=lambda x: x.condition.map(family)))
    hac = []
    for j, name in enumerate(names):
        for lag in cfg["statistics"].get("hac_lags", [6,13,29]):
            value = hac_mean(sums[:,j], grouped["count"].to_numpy(), grouped.segment.to_numpy(), day_ids,
                             lag=lag, annual_bars=accounting(cfg)["annual_bars"])
            hac.append({"condition": name, "family": family[name], "lag_days": lag, "primary_inference": lag==6,
                        **{key:value[key] for key in ("estimate","standard_error","ci_low","ci_high","p_value","inference_available")}})
    pooled, monthly, quarterly = map(pd.DataFrame, (pooled, monthly, quarterly))
    subsets = {"all_seeds": seeds, "reused_seeds": cfg.get("reused_seeds", []), "new_seeds": cfg.get("new_seeds", [])}
    summaries = []
    for name, selected in subsets.items():
        if selected:
            summaries.append(_summary(pooled.loc[pooled.seed.isin(selected)], ["condition"]).assign(seed_subset=name))
    return {"pooled_metrics": pooled, "seed_summary": pd.concat(summaries,ignore_index=True),
        "fold_metrics":pd.DataFrame(folded), "monthly_by_seed":monthly, "quarterly_by_seed":quarterly,
        "monthly_summary":_summary(monthly,["condition","period"]),
        "quarterly_summary":_summary(quarterly,["condition","period"]),
        "inference":inference, "primary_inference":inference.loc[inference.primary_inference].copy(),
        "hac_inference":_correct(pd.DataFrame(hac),hac=True),
        "daily_panel":{"sums":sums,"counts":grouped["count"].to_numpy(),"segment_ids":grouped.segment.to_numpy(),"day_ids":day_ids},
        "audit":{"n_market_rows":len(reference),"n_seeds":len(seeds),"n_reset_segments":int(resets.sum()),
            "historical_seen_test":True,"independently_unused_holdout":False,"seed_count_multiplies_market_rows":False,
            "families":{"baseline_primary":12,"policy_secondary":2},"combined_family":14,
            "period_returns":"compound within each seed/period before averaging; mean monthly returns are not one compoundable strategy",
            "worst_risk":"worst individual seed maxDD and its seed ID reported alongside mean/median/std",
            "design":result["design"]}}


def aggregate(cfg, registration_sha):
    """Read all registered reused/new ledgers only after the global barrier."""
    from . import wm_seed_robustness_benchmark as benchmark
    benchmark.verify_test_barrier(cfg)
    root = Path(cfg["output_dir"])
    if _sha(root/"preregistration.json") != registration_sha:
        raise ValueError("registered seed study changed before report access")
    manifest_path = root/"evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    barrier_path=root/"test_barrier.json"
    if (not manifest.get("complete") or manifest.get("registration_sha256") != registration_sha
            or manifest.get("barrier_sha256") != _sha(barrier_path)):
        raise ValueError("a complete registered evaluation manifest is required")
    jobs = manifest["jobs"]
    expected = {(fold,seed,condition) for fold in cfg["folds"] for seed in cfg["seeds"] for condition in CONDITIONS}
    if (len(jobs)!=len(expected) or {(j["fold"],j["seed"],j["condition"]) for j in jobs}!=expected):
        raise ValueError("all180 registered reused/new evaluations must be present without duplicates")
    frames=[];inputs={str(manifest_path):_sha(manifest_path),str(barrier_path):_sha(barrier_path),
                      str(root/"preregistration.json"):registration_sha}
    for job in jobs:
        if bool(job["reused"]) != (job["seed"] in cfg.get("reused_seeds",[])):
            raise ValueError("reused evaluation provenance disagrees with the fixed seed groups")
        path=Path(job["path"])
        if _sha(path)!=job["sha256"]:
            raise ValueError("an immutable evaluation ledger changed")
        inputs[str(path)]=job["sha256"]
        part=pd.read_parquet(path)
        arm,suffix=job["condition"].split("_",1)
        for key,value in (("fold",job["fold"]),("seed",job["seed"]),("arm",arm)):
            if key in part and not part[key].eq(value).all():
                raise ValueError("ledger job metadata disagrees with registration")
            part[key]=value
        if "condition" in part and not part.condition.isin([suffix,job["condition"]]).all():
            raise ValueError("ledger condition differs from its registered reused/new identity")
        part["condition"]=job["condition"]
        frames.append(part)
    frame=pd.concat(frames,ignore_index=True)
    reference=frame.loc[frame.condition.eq(CONDITIONS[0])&frame.seed.eq(cfg["seeds"][0]),list(COHORT)].sort_values("timestamp")
    raw_paths=[Path(cfg["data"][name]) for name in ("raw_close_path","raw_index_ns_path")]
    for path in raw_paths:inputs[str(path)]=_sha(path)
    close,times=(np.load(path,mmap_mode="r") for path in raw_paths)
    baselines,baseline_metrics,baseline_audit=build_baselines(close,times,reference,cfg)
    result=evaluate_ledgers(frame,baselines,cfg,condition_configs(cfg))
    result["baseline_metrics"]=baseline_metrics
    final=root/"final";final.mkdir(parents=True,exist_ok=True)
    for name,value in result.items():
        if isinstance(value,pd.DataFrame):value.to_csv(final/(name+".csv"),index=False)
    baselines.to_parquet(final/"executed_baselines.parquet",index=False)
    ids=reference.decision_id.to_numpy(np.int64)
    calendar=np.arange(ids[0],ids[-1]+1)
    pd.DataFrame({"timestamp":pd.to_datetime(np.asarray(times)[calendar],utc=True),
        "equity_from_100":100*np.asarray(close)[calendar]/close[ids[0]-1]}).to_csv(final/"calendar_bh_reference.csv",index=False)
    np.savez(final/"daily_panel.npz",**result["daily_panel"])
    result["audit"].update(registration_sha256=registration_sha,input_sha256=inputs,baselines=baseline_audit,
        n_reused_evaluations=sum(bool(j["reused"]) for j in jobs),n_new_evaluations=sum(not j["reused"] for j in jobs))
    if any(_sha(path)!=digest for path,digest in inputs.items()):
        raise ValueError("registered inputs changed during aggregation")
    benchmark.verify_test_barrier(cfg)
    (final/"audit.json").write_text(json.dumps(result["audit"],ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    _report(final,result)
    return {"complete":True,"registration_sha256":registration_sha,"n_evaluations":len(jobs),
        "n_market_rows":result["audit"]["n_market_rows"],"n_seeds":len(cfg["seeds"]),
        "artifact_sha256":{str(p.relative_to(root)):_sha(p) for p in final.iterdir() if p.is_file()},
        "report_path":str(final/"index.html"),"independently_unused_holdout":False}


def _report(final,result):
    summary=result["seed_summary"]
    main=summary.loc[summary.seed_subset.eq("all_seeds")]
    columns=["condition","n_seeds","net_total_return_mean","net_total_return_median","net_total_return_std",
             "net_total_return_min","net_total_return_max","max_drawdown_mean","worst_drawdown","worst_drawdown_seed",
             "mean_exposure_mean","near_cash_fraction_mean"]
    primary=result["primary_inference"]
    body=["<!doctype html><html lang='ja'><meta charset='utf-8'><title>10seedと因果baselineの比較</title>",
        "<style>body{font:15px system-ui;line-height:1.7;margin:30px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}.scroll{overflow:auto}</style>",
        "<h1>10seedのばらつきと単純baseline</h1>",
        "<p>条件ID：C_floor_gradual＝WM潜在あり・最低保有率の段階解除、A_floor_gradual＝WM潜在なし・最低保有率の段階解除、C_control＝WM潜在あり・制限維持です。</p>",
        "<p>既に選ばれた既知Test条件の頑健性検証です。seed追加は市場標本数を増やしません。元3seed・追加7seedの成績も分けて保存しています。",
        "同じ欠落バー・保有率1への無料リセットを含むcohortであり、連続自己資金運用や未使用Holdoutの結果ではありません。</p>",
        "<p>calendar_bh_reference.csvは全raw期間の価格比だけの別参照です。cohort B&Hとは欠落期間の扱いが異なり、RLとの同条件の優劣比較には使いません。全カレンダーRLは未評価です。</p>",
        "<p>全seedの平均・中央値・標準偏差・最小最大と、最悪DD/seedIDを併記します。全体DDは各seedの最大DDの集計です。平均資産曲線の最大DDではありません。</p>",
        "<p>収益・DD・保有率は割合表記です。0.10＝10%、worst_drawdownはseed別最大下落率の最悪値、worst_drawdown_seedはその学習seed番号です。</p>",
        "<div class='scroll'>",main[columns].to_html(index=False,escape=True,float_format=lambda x:f"{x:.6g}"),"</div>",
        "<h2>固定した4つの対照</h2><p>MA20日・vol過去7日はcohortを詰めた窓ではなく全raw価格の過去だけを参照します。全対照に1バー遅延、速度制限、同じ費用/借入を適用します。",
        "8%目標対照は既知結果に基づくpost-hoc曝露近似で、厳密risk matchingではありません。各区間の初期1倍から、1本あたり0.08の変更速度制限を守って8%へ移行します。同じ初期・リセット条件を使うため、最初の数本は8%を超えます。低保有対照はcontrolのfloor制約では実現できない参考値を含みます。</p>",
        result["baseline_metrics"].to_html(index=False,escape=True,float_format=lambda x:f"{x:.6g}"),
        "<h2>対応あり推論</h2><p>主判定は7日循環block・固定10seed平均。対照比較12のHolm12、方策比較2のHolm2、全14補正を保存します。14/30日・階層seed・HACは感度分析です。",
        "値とBasic95%区間は年率自然対数成長率の差で、累積収益率差ではありません。</p>",
        primary[["condition","family","estimate","ci_low","ci_high","p_value","p_holm_family","p_holm_all14"]].to_html(index=False,escape=True,float_format=lambda x:f"{x:.6g}"),
        "<p><a href='seed_summary.csv'>全/元/新seed集計</a> / <a href='pooled_metrics.csv'>seed別</a> / <a href='monthly_by_seed.csv'>月別</a> / <a href='quarterly_by_seed.csv'>四半期別</a> / <a href='inference.csv'>全推論</a> / <a href='hac_inference.csv'>HAC</a> / <a href='audit.json'>評価監査</a></p>",
        "<h2>時系列分割・除外期間の監査</h2><p>同じfold内のTrain/Val/Test漏洩監査と、過去Testが後続Trainに入るwalk-forwardの通常の期間重複を区別します。",
        "予測入力Fの欠落について、個々の除外に至った原因はすべて確定したわけではありません。確定事項と未確定事項、対象バーとリセットの扱いは監査JSONを参照してください。</p>",
        "<p><a href='../split_audit/audit.json'>分割監査JSON</a> / <a href='../split_audit/split_audit.csv'>fold内分割</a> / <a href='../split_audit/cross_fold_overlaps.csv'>fold間重複</a> / <a href='../split_audit/exclusion_counts.csv'>除外件数</a> / <a href='../split_audit/exclusion_bars.csv'>除外バー</a> / <a href='../split_audit/reset_audit.csv'>リセット監査</a></p></html>"]
    (final/"index.html").write_text("\n".join(body),encoding="utf-8")
