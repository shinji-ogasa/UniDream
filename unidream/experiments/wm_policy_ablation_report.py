"""Auditable policy BC/RL comparisons on the saved conditional cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_policy_ablation_evaluation import (ARMS, STAGES, accounting, execute_positions,
    evaluate_policy_ledger, paired_policy_inference, score_path, bounded_requests)


BASELINE_ALIASES = {"buy_and_hold": "bh", "buy_hold": "bh", "b&h": "bh", "val_constant": "constant",
    "constant_exposure": "constant", "ewma_voltarget": "ewma_vol", "ewma": "ewma_vol",
    "forecast_voltarget": "forecast_vol", "lightgbm_voltarget": "forecast_vol"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _gate(root, cfg, include_test):
    path = root / "preregistration.json"
    if not path.exists():
        raise ValueError("the policy analysis must be registered before reporting")
    registration_sha = _sha(path)
    if not include_test:
        return {"registration_sha256": registration_sha, "test_opened": False}
    barrier_path, choices_path = root / "test_barrier.json", root / "validation_choices.json"
    if not barrier_path.exists() or not choices_path.exists():
        raise ValueError("the complete policy global Test barrier is closed")
    barrier = json.loads(barrier_path.read_text())
    if (barrier.get("registration_sha256") != registration_sha
            or barrier.get("validation_choices_sha256") != _sha(choices_path)):
        raise ValueError("frozen policy registration or Val choices changed")
    choices = json.loads(choices_path.read_text())
    if choices.get("test_used", choices.get("test_selection_allowed")) is not False:
        raise ValueError("policy Val selection must explicitly exclude Test")
    fits = barrier.get("frozen_fit_sha256", {})
    minimum = len(cfg.get("folds", range(6))) * len(cfg.get("seeds", [7, 19, 42])) * len(ARMS)
    if len(fits) < minimum:
        raise ValueError("all registered policy pipelines must freeze before Test")
    for filename, expected in fits.items():
        if _sha(root / filename) != expected:
            raise ValueError("a policy fit changed after the global Test barrier")
    return {"registration_sha256": registration_sha, "test_opened": True,
        "test_barrier_sha256": _sha(barrier_path), "validation_choices_sha256": _sha(choices_path), "n_frozen_fits": len(fits)}


def verify_normalizers(provenance, cfg):
    """Require the same explicit Train source across arms/seeds of each fold."""
    if not isinstance(provenance, dict) or not isinstance(provenance.get("normalizers"), list):
        raise ValueError("Train-only normalizer source provenance is required")
    records = provenance["normalizers"]
    expected = {(fold, seed, arm) for fold in cfg.get("folds", range(6))
        for seed in cfg.get("seeds", [7, 19, 42]) for arm in ARMS}
    actual = [(int(row["fold"]), int(row["seed"]), row["arm"]) for row in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("normalizer provenance must cover every registered policy pipeline exactly once")
    sources = {}
    for row in records:
        source, expected_source = row.get("source_id"), row.get("expected_source_id")
        if row.get("fit_split", "").lower() != "train" or not source or source != expected_source:
            raise ValueError("normalizer source must match its purged Train decisions only")
        fold = int(row["fold"])
        if fold in sources and sources[fold] != expected_source:
            raise ValueError("all arms and seeds of a fold must share the same Train source")
        sources[fold] = expected_source
    return {"verified": True, "n_pipeline_normalizers": len(records), "train_source_by_fold": sources}


def evaluate_baselines(baseline_ledger, executed, cfg):
    reference = executed.loc[(executed.seed == cfg.get("seeds", [7, 19, 42])[0])
        & executed.arm.eq("A") & executed.stage.eq("bc")].sort_values("timestamp").reset_index(drop=True)
    cohort_columns = ["fold", "decision_id", "timestamp", "bar_log_return"]
    reference = reference[cohort_columns]
    controls = []
    bh = reference.copy()
    bh["baseline"] = "bh"
    bh["requested_position"] = 1.
    controls.append(bh)
    if baseline_ledger is not None:
        data = baseline_ledger.copy()
        if "baseline" not in data and "arm" in data:
            data["baseline"] = data["arm"]
        if "baseline" not in data:
            raise ValueError("baseline paths need an explicit baseline name")
        data["baseline"] = data.baseline.str.lower().replace(BASELINE_ALIASES)
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        for baseline, panel in data.groupby("baseline", sort=True):
            panel = panel.sort_values("timestamp").reset_index(drop=True)
            if not panel[cohort_columns].equals(reference):
                raise ValueError("baseline and policy paths must score exactly the same bars")
            if baseline == "bh":
                if "requested_position" in panel and not np.allclose(panel.requested_position, 1, rtol=0, atol=1e-12):
                    raise ValueError("B&H inventory must be exactly1 on the common cohort")
                continue
            if "requested_position" not in panel:
                if "target_position" not in panel:
                    raise ValueError("baseline target/requested inventory is required")
                panel["requested_position"] = bounded_requests(panel.target_position, panel.timestamp, panel.fold, cfg)
            controls.append(panel)
    required = set(cfg.get("required_baselines", ["bh", "constant", "ewma_vol"]))
    if not required.issubset({panel.baseline.iloc[0] for panel in controls}):
        raise ValueError("B&H, Val-fixed constant and causal volatility controls are required")
    paths, folded, pooled = [], [], []
    for panel in controls:
        name = panel.baseline.iloc[0]
        ledger = execute_positions(panel.bar_log_return, panel.requested_position, panel.timestamp, panel.fold, cfg)
        ledger["decision_id"] = panel.decision_id.to_numpy()
        ledger["baseline"] = name
        paths.append(ledger)
        pooled.append({"baseline": name, **score_path(ledger, cfg)})
        for fold, part in ledger.groupby("fold", sort=True):
            folded.append({"fold": fold, "baseline": name, **score_path(part, cfg)})
    return pd.concat(paths, ignore_index=True), pd.DataFrame(folded), pd.DataFrame(pooled)


def _html(aggregate, inference, baselines):
    columns = ["arm", "stage", "n_seeds", "net_total_return_seed_mean", "alpha_excess_seed_mean",
        "sharpe_seed_mean", "max_drawdown_seed_mean", "maxdd_delta_seed_mean", "turnover_seed_mean",
        "mean_exposure_seed_mean", "fee_fraction_sum_seed_mean", "borrow_fraction_sum_seed_mean",
        "collapse_constant_seed_fraction", "collapse_boundary_seed_fraction"]
    primary = inference.loc[inference.primary_inference]
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>政策A/B/C/D：BC・RL比較</title>'
        '<style>body{font-family:system-ui;margin:24px;line-height:1.7}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccc}.table{overflow:auto;max-height:70vh}</style>'
        '<h1>A/B/C/D：生データ・予測値・WM潜在表現とBC/RL</h1>'
        '<p>A=生データ、B=生データ+予測値F、C=生データ+WM潜在表現Z、D=すべて。'
        '全4条件でBCと実履歴上のRLを同じ教師・報酬・売買制約・費用で比較します。WMの想像軌道だけで利益を評価しません。</p>'
        '<p>この結果は保存された予測が利用できる共通の連続区間に条件づけた研究です。欠けたバーは埋めず、'
        'ギャップとfoldの境界ではすべての条件・基準を在庫1から再開します。全カレンダーの運用損益やライブ時点の情報利用可能性を証明しません。'
        '過去に確認済みの市場期間に対する探索的比較で、新規の未使用Holdoutではありません。</p>'
        '<p>予測特徴Fの採用には、同じ市場期間について以前に確認した予測実験の結果も利用しています。'
        'この特徴選択と予測データの利用可能性は完全に未知だった条件ではなく、時系列分割や有意性検定だけでこの探索の影響を除くことはできません。</p>'
        '<p>要求した在庫は次のバーで約定します。範囲0.50〜1.12、1回の変化上限0.08、片道費用5.5bps。'
        '1を超える借入部分には年率10%を仮定します。年率指標は35040本への観測バー換算です。'
        '境界の在庫1は初期保有として扱い、人工的な参入・終了時決済は追加しません。借入・分数エクスポージャーの簡略モデルです。</p>'
        '<p><a href="fold_metrics.csv">fold別144行</a> / <a href="pooled_metrics.csv">seed別24条件</a> / '
        '<a href="aggregate_metrics.csv">seed平均8条件</a> / <a href="paired_inference.csv">対応ありBlock Bootstrap</a> / '
        '<a href="baseline_metrics.csv">単純売買基準</a> / <a href="control_comparisons.csv">基準との差</a> / '
        '<a href="report_contract.json">入力hash・実行契約</a></p>'
        '<h2>3学習seedの平均</h2><p>同じ市場を3回観測した独立標本としては扱いません。seedごとの成績と最小・最大もCSVに保存します。'
        '低い変動・境界への張り付きを示すcollapse欄は診断であり、定数基準が良い成績になる場合もあります。</p><div class="table">'
        + aggregate.reindex(columns=columns).to_html(index=False, escape=True) + '</div>'
        '<h2>12の事前固定比較</h2><p>BC/RLそれぞれのB−A、C−A、D−B、D−Cと、4条件のRL−BCです。'
        '主指標は年率換算した純対数成長率の差で、正が改善。7日ブロックを同じ市場日・対応するseedで抽出し、'
        'seed IDも対応を保って再標本化します。1日・28日は感度確認。短すぎる連続区間は固定して保持し、短いブロックへ変更しません。'
        '区間は各比較の95%区間で同時信頼区間ではありません。主比較12組はHolm補正します。</p><div class="table">'
        + primary.to_html(index=False, escape=True) + '</div>'
        '<h2>単純な売買基準</h2><p>B&H、Valだけで固定した一定エクスポージャー、因果的な分散予測による変動調整を、'
        '同じ遅延・費用・借入・対象バーで採点します。基準との差の表は記述統計で、12の特徴量比較とは別です。</p><div class="table">'
        + baselines.to_html(index=False, escape=True) + '</div>')


def write_policy_report(output_dir, cfg, ledger=None, *, include_test=False, provenance=None, baseline_ledger=None):
    """Export full fold/seed/stage accounting without any Test selection."""
    root = Path(output_dir).resolve()
    gate = _gate(root, cfg, include_test)
    if provenance is None and (root / "normalizer_provenance.json").exists():
        provenance = json.loads((root / "normalizer_provenance.json").read_text())
    normalizers = verify_normalizers(provenance, cfg)
    destination = root / ("final" if include_test else "selection")
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / "policy_ledger.parquet"
    supplied_ledger = ledger is not None
    if ledger is None:
        ledger = pd.read_parquet(ledger_path)
    if "split" in ledger:
        if include_test and not ledger.split.eq("test").all():
            raise ValueError("final policy report requires the unchanged Test split")
        if not include_test and ledger.split.eq("test").any():
            raise ValueError("Test rows cannot enter validation policy selection")
    if supplied_ledger and ledger_path.exists():
        original = pd.read_parquet(ledger_path)
        if not ledger.reset_index(drop=True).equals(original.reset_index(drop=True)):
            raise ValueError("supplied policy ledger differs from its immutable input artifact")
    elif supplied_ledger:
        ledger.to_parquet(ledger_path, index=False)
    executed, folded, pooled, aggregate = evaluate_policy_ledger(ledger, cfg)
    executed.to_parquet(destination / "executed_policy_ledger.parquet", index=False)
    folded.to_csv(destination / "fold_metrics.csv", index=False)
    pooled.to_csv(destination / "pooled_metrics.csv", index=False)
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    inference = paired_policy_inference(executed, cfg)
    inference.to_csv(destination / "paired_inference.csv", index=False)
    control_paths, control_folds, controls = evaluate_baselines(baseline_ledger, executed, cfg)
    control_paths.to_parquet(destination / "executed_baseline_ledger.parquet", index=False)
    control_folds.to_csv(destination / "baseline_fold_metrics.csv", index=False)
    controls.to_csv(destination / "baseline_metrics.csv", index=False)
    comparisons = []
    for row in pooled.to_dict("records"):
        for baseline in controls.to_dict("records"):
            comparisons.append({"seed": row["seed"], "arm": row["arm"], "stage": row["stage"], "baseline": baseline["baseline"],
                "net_total_return_difference": row["net_total_return"] - baseline["net_total_return"],
                "sharpe_difference": row["sharpe"] - baseline["sharpe"],
                "max_drawdown_improvement": baseline["max_drawdown"] - row["max_drawdown"],
                "annualized_net_log_growth_difference": row["annualized_net_log_growth"] - baseline["annualized_net_log_growth"],
                "descriptive_only_not_one_of_12_hypotheses": True})
    pd.DataFrame(comparisons).to_csv(destination / "control_comparisons.csv", index=False)
    (destination / "index.html").write_text(_html(aggregate, inference, controls), encoding="utf-8")
    contract = {"complete": True, "test_report_only": include_test, **gate, "normalizer_audit": normalizers,
        "accounting": accounting(cfg), "fold_rows": len(folded), "pooled_seed_conditions": len(pooled),
        "seed_aggregate_conditions": len(aggregate), "n_primary_hypotheses": 12,
        "policy_ledger_input_sha256": _sha(ledger_path) if ledger_path.exists() else None,
        "executed_ledger_sha256": _sha(destination / "executed_policy_ledger.parquet"),
        "inference_sha256": _sha(destination / "paired_inference.csv"),
        "report_source_sha256": _sha(Path(__file__)), "provenance": provenance,
        "conditional_saved_forecast_cohort": True, "full_calendar_pnl_claim": False,
        "independently_unused_holdout": False, "seed_market_independence_assumed": False,
        "baseline_comparisons_are_descriptive": True, "test_selection_allowed": False}
    _json(destination / "report_contract.json", contract)
    completion = {"complete": True, "report_complete": True, "include_test": include_test,
        "fold_rows": len(folded), "pooled_conditions": len(pooled), "aggregate_conditions": len(aggregate),
        "n_primary_hypotheses": 12, "registration_sha256": gate["registration_sha256"],
        "report_dir": str(destination), "contract_sha256": _sha(destination / "report_contract.json")}
    _json(destination / "report_completion.json", completion)
    return completion
