"""Post-hoc eight-condition B&H inference on frozen policy results.

Separate names deliberately avoid changing the completed ablation's frozen
source glob. No training, checkpoint choice, or original artifact is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from .wm_significance_statistics import holm_fixed_family

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_policy_bh_inference_20261112.yaml"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    for name in ("source_config", "output_dir"):
        cfg[name] = str((ROOT / cfg[name]).resolve())
    if (cfg["arms"] != list("ABCD") or cfg["stages"] != ["bc", "rl"]
            or cfg["seeds"] != [7, 19, 42] or cfg["folds"] != list(range(6))
            or cfg.get("post_hoc") is not True or cfg.get("source_test_results_previously_seen") is not True
            or cfg.get("test_selection_allowed") is not False or cfg.get("independently_unused_holdout") is not False
            or cfg["statistics"]["metric"] != "annualized_net_log_growth_difference"
            or cfg["statistics"]["primary_family_size"] != 8 or cfg["statistics"]["extended_family_size"] != 20
            or cfg["statistics"]["primary_block_days"] != 7 or cfg["statistics"]["block_days"] != [1, 7, 28]
            or cfg["statistics"]["alternative"] != "two_sided"
            or cfg["statistics"]["bootstrap_replicates"] != 10000):
        raise ValueError("fixed complete8 post-hoc BH comparisons and declared two-sided Holm8/20 required")
    return cfg


def source_hashes(cfg):
    files = sorted(Path(__file__).parent.glob("wm_policy_bh_*.py"))
    files += sorted((ROOT / "tests").glob("test_wm_policy_bh_*.py"))
    files += [Path(__file__).with_name(name) for name in (
        "wm_full_grid_statistics.py", "wm_significance_statistics.py", "wm_policy_ablation_evaluation.py")]
    files.append(Path(cfg["config_path"]))
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def original_sources(cfg):
    """Verify the completed original study and bind its saved result bytes."""
    from . import wm_policy_ablation_benchmark as parent
    pcfg = parent.load_config(cfg["source_config"])
    preg = parent.verify(pcfg, cache=True)
    barrier = parent.verify_test_barrier(pcfg)
    root = Path(pcfg["output_dir"])
    completion = json.loads((root / "completion.json").read_text())
    contract = json.loads((root / "final/report_contract.json").read_text())
    if (not completion.get("complete") or not contract.get("test_report_only")
            or len(barrier["frozen_fit_sha256"]) != 90 or len(preg["jobs"]) != 90
            or completion["report"]["contract_sha256"] != sha(root / "final/report_contract.json")
            or contract["executed_ledger_sha256"] != sha(root / "final/executed_policy_ledger.parquet")
            or contract["inference_sha256"] != sha(root / "final/paired_inference.csv")):
        raise ValueError("complete unchanged original policy fit/barrier/report required")
    names = ["preregistration.json", "test_barrier.json", "validation_choices.json", "completion.json",
        "final/report_contract.json", "final/executed_policy_ledger.parquet",
        "final/executed_baseline_ledger.parquet", "final/pooled_metrics.csv", "final/paired_inference.csv"]
    return {"source_root": str(root), "original_registration_sha256": sha(root / "preregistration.json"),
        "original_source_verified": True, "source_artifact_sha256": {name: sha(root / name) for name in names},
        "original_primary_family_size": 12, "new_posthoc_family_size": 8,
        "expected_market_rows": 50038, "all_original90_fits_frozen": True}


def register(cfg):
    """Freeze this additional analysis BEFORE computing its new BH tests.

    Outcomes were already seen, so this is an analysis amendment, never an
    independent confirmatory preregistration or a previously unused holdout.
    """
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    sources = original_sources(cfg)
    path = root / "posthoc_registration.json"
    if path.exists():
        registration = json.loads(path.read_text())
        if registration["config"] != cfg or registration["source_sha256"] != source_hashes(cfg) or registration["sources"] != sources:
            raise ValueError("frozen BH analysis configuration, sources, or input results changed")
        return registration
    registration = {"registered_at": datetime.now(timezone.utc).isoformat(), "config": cfg,
        "source_sha256": source_hashes(cfg), "sources": sources,
        "post_hoc": True, "original_test_results_previously_seen": True,
        "primary_comparisons": [f"{stage}_{arm}_minus_bh" for stage in cfg["stages"] for arm in cfg["arms"]],
        "metric": "annualized mean net log return difference; positive favors policy",
        "inference": "two-sided, 95% pointwise percentile CI; paired 7day moving blocks and paired seed resampling; Holm8",
        "expanded_family": "Holm20 over original12 and new8 raw primary p-values; sensitivity only",
        "secondary_blocks": "1 and28day sensitivity, no selecting a favorable block length",
        "benchmark": "unchanged actual executed BH ledger, inventory1, broadcast same market path across seeds",
        "cumulative_return_difference": "arithmetic mean of three seed total returns minus BH total return; descriptive only",
        "scope": "same conditional saved-forecast cohort, gap/fold resets, endowed1x start, original fees and borrow; not full-calendar return",
        "model_or_validation_choices_changed": False, "independently_unused_holdout": False}
    write_json(path, registration)
    return registration


def _validate_bh(ledger):
    bh = ledger.loc[ledger.baseline.eq("bh")].sort_values("timestamp").reset_index(drop=True)
    if (not len(bh) or bh.duplicated(["fold", "decision_id"]).any()
            or not np.allclose(bh.executed_position, 1., rtol=0, atol=1e-12)
            or not np.allclose(bh.cost_fraction, 0., rtol=0, atol=1e-12)
            or not np.allclose(bh.borrow_fraction, 0., rtol=0, atol=1e-12)
            or not np.allclose(bh.net_log_return, bh.bar_log_return, rtol=0, atol=1e-12)):
        raise ValueError("one unchanged actual1x B&H cohort without fees or borrow required")
    return bh


def extended_correction(inference, original):
    result = inference.copy()
    primary = result.loc[result.primary_inference]
    old = original.loc[original.primary_inference]
    if len(primary) != 8 or len(old) != 12 or primary.comparison.duplicated().any() or old.comparison.duplicated().any():
        raise ValueError("Holm20 must retain exactly all8 new and all12 original hypotheses")
    family = {"original:" + row.comparison: float(row.p_value) if pd.notna(row.p_value) else 1.
              for row in old.itertuples()}
    family.update({"bh:" + row.comparison: float(row.p_value) if pd.notna(row.p_value) else 1.
                   for row in primary.itertuples()})
    adjusted = holm_fixed_family(family, family_size=20)
    result["p_holm_20_sensitivity"] = [adjusted["bh:" + row.comparison] if row.primary_inference else np.nan for row in result.itertuples()]
    result["significant_improvement_holm20"] = (result.primary_inference & result.estimate.gt(0)
        & result.p_holm_20_sensitivity.lt(.05) & result.inference_available)
    combined = pd.DataFrame([{"family": key.split(":", 1)[0], "comparison": key.split(":", 1)[1],
        "p_value": value, "p_holm_20": adjusted[key]} for key, value in family.items()])
    return result, combined


def descriptive_tables(policies, bh, cfg):
    """Arithmetic seed-average returns stay distinct from the log-growth test."""
    rows, folds = [], []
    annual = float(cfg["accounting"]["annual_bars"])
    bh_total = float(np.expm1(bh.net_log_return.sum()))
    bh_growth = float(bh.net_log_return.mean() * annual)
    for (stage, arm, seed), part in policies.groupby(["stage", "arm", "seed"], sort=True):
        total = float(np.expm1(part.net_log_return.sum()))
        growth = float(part.net_log_return.mean() * annual)
        rows.append({"stage": stage, "arm": arm, "seed": seed, "n_market_rows": len(part),
            "policy_net_total_return": total, "bh_net_total_return": bh_total,
            "cumulative_return_difference_descriptive": total - bh_total,
            "policy_annualized_log_growth": growth, "bh_annualized_log_growth": bh_growth,
            "annualized_log_growth_difference": growth - bh_growth,
            "mean_exposure": float(part.executed_position.mean())})
        for fold, section in part.groupby("fold", sort=True):
            baseline = bh.loc[bh.fold.eq(fold)]
            p_total = float(np.expm1(section.net_log_return.sum()))
            b_total = float(np.expm1(baseline.net_log_return.sum()))
            folds.append({"stage": stage, "arm": arm, "seed": seed, "fold": fold, "n_market_rows": len(section),
                "policy_net_total_return": p_total, "bh_net_total_return": b_total,
                "cumulative_return_difference_descriptive": p_total - b_total,
                "annualized_log_growth_difference": float((section.net_log_return.mean() - baseline.net_log_return.mean()) * annual)})
    seed_table = pd.DataFrame(rows)
    aggregate = seed_table.groupby(["stage", "arm"], as_index=False).agg(
        policy_net_total_return=("policy_net_total_return", "mean"),
        bh_net_total_return=("bh_net_total_return", "first"),
        cumulative_return_difference_descriptive=("cumulative_return_difference_descriptive", "mean"),
        mean_exposure=("mean_exposure", "mean"), n_market_rows=("n_market_rows", "first"))
    return seed_table, pd.DataFrame(folds), aggregate


def write_report(root, inference, aggregate):
    primary = inference.loc[inference.primary_inference].merge(aggregate, on=["stage", "arm"], validate="one_to_one")
    primary = primary.sort_values(["stage", "arm"]).reset_index(drop=True)
    primary.to_csv(root / "summary.csv", index=False)
    labels = {"A": "市場データ", "B": "市場＋ML予測", "C": "市場＋WM潜在", "D": "市場＋ML予測＋WM潜在"}
    table = pd.DataFrame({"条件": primary.arm.map(labels), "段階": primary.stage.str.upper(),
        "リターン(3seed平均)": primary.policy_net_total_return.map(lambda v: f"{v:+.2%}"),
        "B&Hリターン": primary.bh_net_total_return.map(lambda v: f"{v:+.2%}"),
        "差(記述値・pt)": primary.cumulative_return_difference_descriptive.map(lambda v: f"{v * 100:+.2f}"),
        "年率対数成長率差(pt)": primary.estimate.map(lambda v: f"{v * 100:+.2f}"),
        "95%信頼区間(pt)": [f"[{lo*100:+.2f}, {hi*100:+.2f}]" if pd.notna(lo) else "算出不可" for lo, hi in zip(primary.ci_low, primary.ci_high)],
        "p(補正前)": primary.p_value.map(lambda v: f"{v:.4f}"),
        "p(Holm8)": primary.p_holm_8.map(lambda v: f"{v:.4f}"),
        "p(Holm20・参考)": primary.p_holm_20_sensitivity.map(lambda v: f"{v:.4f}"),
        "有意な改善(Holm8)": primary.significant_improvement.map({True: "あり", False: "なし"})})
    n_significant = int(primary.significant_improvement.sum())
    source_link = Path("../wm_policy_ablation_20261112/final/index.html")
    body = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>B&Hとの追加検定</title><style>'
    body += 'body{font:16px system-ui;line-height:1.8;margin:40px;max-width:1600px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #ccc}th{background:#edf2f7}.scroll{overflow-x:auto}a{color:#1769aa}</style>'
    body += '<h1>B&Hとの差：8条件の追加検定</h1>'
    body += f'<p>7日Block Bootstrap・Holm8補正後、有意な改善は <strong>{n_significant}/8条件</strong>。</p>'
    body += '<p>既存のTest結果を見た後に追加した事後検証です。モデル・チェックポイント・Val選択・売買条件は変更していません。</p>'
    body += '<p>主指標は年率換算した平均ネット対数リターンの差。95%信頼区間は各比較の区間、p値は両側検定です。'
    body += '同じ市場日を全seedに用いる時系列Block Bootstrapとseedの対応を保った再標本化を10,000回行いました。'
    body += 'Holm8は追加8比較、Holm20は元の12比較も含める感度分析です。</p>'
    body += '<p>累積リターン差は3seedの算術平均に基づく記述値で、信頼区間の対象となる年率対数成長率差とは単位が異なります。'
    body += '1日・28日ブロックは感度分析として全件保存しています。</p><div class="scroll">' + table.to_html(index=False, escape=True) + '</div>'
    body += '<p>対象は同じ50,038本の15分足・6fold。欠損区間とfold境界で口座状態をリセットした共通区間の開発評価です。'
    body += 'B&Hは初期保有1倍で開始します。市場への投資比率が異なるため、B&Hとの差だけでは売買タイミング能力を特定できません。</p>'
    body += '<p><a href="inference.csv">全検定CSV</a> · <a href="summary.csv">要約CSV</a> · <a href="per_seed.csv">seed別</a> · '
    body += '<a href="per_fold.csv">fold別</a> · <a href="extended_family20.csv">拡張20比較</a> · '
    body += f'<a href="{html.escape(str(source_link))}">元の実験結果</a></p></html>'
    (root / "index.html").write_text(body, encoding="utf-8")
    return primary


def run(cfg):
    registration = register(cfg)
    root = Path(cfg["output_dir"])
    completion_path = root / "completion.json"
    if completion_path.exists():
        previous = json.loads(completion_path.read_text())
        if previous["registration_sha256"] != sha(root / "posthoc_registration.json"):
            raise ValueError("completed BH analysis registration changed")
        for name, expected in previous["artifact_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("completed BH output changed")
        return previous
    source = Path(registration["sources"]["source_root"])
    policies = pd.read_parquet(source / "final/executed_policy_ledger.parquet")
    bh = _validate_bh(pd.read_parquet(source / "final/executed_baseline_ledger.parquet"))
    if len(bh) != registration["sources"]["expected_market_rows"]:
        raise ValueError("registered original common market cohort size changed")
    from .wm_policy_bh_inference import infer_bh
    inference = infer_bh(policies, bh, cfg)
    original = pd.read_csv(source / "final/paired_inference.csv")
    inference, expanded = extended_correction(inference, original)
    seed_table, fold_table, aggregate = descriptive_tables(policies, bh, cfg)
    inference.to_csv(root / "inference.csv", index=False)
    expanded.to_csv(root / "extended_family20.csv", index=False)
    seed_table.to_csv(root / "per_seed.csv", index=False)
    fold_table.to_csv(root / "per_fold.csv", index=False)
    primary = write_report(root, inference, aggregate)
    # Recheck original inputs/source after inference, before publishing completion.
    if original_sources(cfg) != registration["sources"] or source_hashes(cfg) != registration["source_sha256"]:
        raise ValueError("source results changed during the BH analysis")
    artifacts = {name: sha(root / name) for name in ("inference.csv", "extended_family20.csv", "per_seed.csv", "per_fold.csv", "summary.csv", "index.html")}
    completion = {"complete": True, "at": datetime.now(timezone.utc).isoformat(), "post_hoc": True,
        "registration_sha256": sha(root / "posthoc_registration.json"), "artifact_sha256": artifacts,
        "n_primary_comparisons": len(primary), "n_total_inference_rows": len(inference),
        "significant_improvements_holm8": int(primary.significant_improvement.sum()),
        "significant_improvements_holm20": int(primary.significant_improvement_holm20.sum()),
        "n_unique_market_rows": len(bh), "test_selection_changed": False,
        "original_study_unchanged": True, "report": str(root / "index.html")}
    write_json(completion_path, completion)
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
