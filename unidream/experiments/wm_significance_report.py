"""Read-only presentation of the registered 336-cell baseline inference.

This report never fits a model, chooses a threshold, adjusts a P value, or
changes an existing study.  A joint condition retains two separate native
heads; its regression and classification hypotheses are never collapsed.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = tuple("ABCD")
MODELS = ("linear", "lightgbm", "direct", "wm")
TASKS = ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery")
FORMULATIONS = ("regression", "classification", "joint")
KEYS = ("group", "model", "task", "formulation")
TASK_LABELS = {"return": "リターン", "variance": "実現分散", "downside": "現在価格からの最大下落幅",
    "crash": "急落イベント", "dd_excess": "下落閾値の超過量", "advantage": "買い増し優位性（合成指標）",
    "recovery": "回復度（合成指標）"}
METRIC_ALIASES = {"ba": "balanced_accuracy", "oos_r2": "zero_oos_r2",
    "pearson": "pearson_ic", "spearman": "spearman_ic", "ic": "pearson_ic"}
HEAD_ALIASES = {"native": "regression", "scalar": "regression", "point": "regression",
    "class": "classification", "probability": "classification", "ba": "classification"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _true(value):
    return bool(value is True or str(value).lower() == "true" or value == 1)


def _normalize(frame):
    frame = frame.copy()
    aliases = {"ci_lower": "ci_low", "ci_upper": "ci_high", "model_ci_low": "ci_low", "model_ci_high": "ci_high", "metric_name": "metric",
        "baseline_name": "baseline", "model_estimate": "estimate", "model_value": "estimate",
        "baseline_value": "baseline_estimate", "p_holm_448": "p_holm_primary448",
        "p_adjusted_primary": "p_holm_primary448", "p_adjusted_global": "p_holm_global"}
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
    required = {*KEYS, "head", "metric", "role", "baseline", "block_days", "estimate", "ci_low", "ci_high"}
    if not required.issubset(frame.columns):
        raise ValueError("inference table lacks condition/head/metric/baseline/CI columns")
    frame["head"] = frame["head"].replace(HEAD_ALIASES)
    frame["metric"] = frame["metric"].replace(METRIC_ALIASES)
    frame.loc[frame["head"].eq("auxiliary") & frame.metric.eq("balanced_accuracy"), "head"] = "classification"
    for name in ("estimate", "ci_low", "ci_high", "baseline_estimate", "difference", "improvement",
                 "improvement_ci_low", "improvement_ci_high", "p_value", "p_holm_primary448", "p_holm_global"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["block_days"] = pd.to_numeric(frame["block_days"], errors="coerce")
    if frame.block_days.isna().any() or (frame.block_days < 1).any():
        raise ValueError("inference block lengths must be positive finite days")
    if "split" in frame and not frame.split.eq("test").all():
        raise ValueError("significance report may contain only frozen Test inference")
    return frame


def _validate_points(points):
    expected = {(g, m, t, f) for g in GROUPS for m in MODELS for t in TASKS for f in FORMULATIONS}
    if not set(KEYS).issubset(points.columns):
        raise ValueError("point table lacks registered condition columns")
    actual = set(points[list(KEYS)].itertuples(index=False, name=None))
    if len(points) != 336 or points.duplicated(list(KEYS)).any() or actual != expected:
        raise ValueError("exactly 336 unique registered conditions required")


def _select_primary(frame, primary_days):
    selected = frame.loc[frame.role.eq("primary") & frame.block_days.eq(primary_days)].copy()
    expected = {(g, m, t, f, h) for g in GROUPS for m in MODELS for t in TASKS
        for f in FORMULATIONS for h in (("regression", "classification") if f == "joint" else (f,))}
    columns = [*KEYS, "head"]
    actual = set(selected[columns].itertuples(index=False, name=None))
    if len(selected) != 448 or selected.duplicated(columns).any() or actual != expected:
        raise ValueError("exactly 448 unique primary native-head comparisons required")
    available = (selected.inference_available.map(_true).to_numpy() if "inference_available" in selected
        else np.ones(len(selected), dtype=bool))
    for name in ("estimate", "ci_low", "ci_high", "baseline_estimate", "improvement",
                 "improvement_ci_low", "improvement_ci_high", "p_value", "p_holm_primary448"):
        if name not in selected:
            raise ValueError("primary inference lacks " + name)
        values = pd.to_numeric(selected[name], errors="coerce").to_numpy(float)
        invalid = ~np.isfinite(values)
        if (invalid & available).any() or (name == "p_holm_primary448" and invalid.any()):
            raise ValueError("every available primary comparison requires finite " + name)
    for low, high in (("ci_low", "ci_high"), ("improvement_ci_low", "improvement_ci_high")):
        if (selected[low] > selected[high]).any():
            raise ValueError("confidence interval endpoints are reversed")
    for name in ("p_value", "p_holm_primary448", "p_holm_global"):
        if name in selected:
            values = pd.to_numeric(selected[name], errors="coerce").dropna()
            if ((values < 0) | (values > 1)).any():
                raise ValueError("P values must lie in [0,1]")
    class_rows = selected["head"].eq("classification")
    if not selected.loc[class_rows, "metric"].eq("balanced_accuracy").all():
        raise ValueError("native classification primary metric must remain balanced accuracy")
    return selected


def _copy_metric(row, name, inference):
    row[name + "_metric"] = inference["metric"]
    for field in ("estimate", "ci_low", "ci_high", "baseline_estimate", "difference", "improvement",
                  "improvement_ci_low", "improvement_ci_high", "p_value", "p_holm_primary448", "p_holm_global"):
        row[name + "_" + field] = _number(inference.get(field))
    row[name + "_baseline"] = inference.get("baseline")
    row[name + "_inference_available"] = _true(inference.get("inference_available", True))
    row[name + "_inference_unavailable_reason"] = inference.get("inference_unavailable_reason")
    adjusted = _number(inference.get("p_holm_primary448"))
    improvement = _number(inference.get("improvement"))
    row[name + "_significant_improvement_holm448"] = bool(adjusted is not None and adjusted < .05 and improvement is not None and improvement > 0)
    global_p = _number(inference.get("p_holm_global"))
    row[name + "_significant_improvement_global"] = bool(global_p is not None and global_p < .05 and improvement is not None and improvement > 0)


def _match_frozen_point(source, selected, head):
    if source["formulation"] == "joint":
        field = "regression_primary_value" if head == "regression" else "classification_primary_value"
        expected = _number(source.get(field, source.get("balanced_accuracy") if head == "classification" else None))
    else:
        expected = _number(source.get("primary_value"))
    observed = _number(selected.get("estimate"))
    if expected is not None and observed is not None and not np.isclose(expected, observed, rtol=1e-8, atol=1e-12):
        raise ValueError("new primary estimates differ from the frozen336 point table")


def _combine(points, inference, primary, primary_days):
    table = []
    indexed = primary.set_index([*KEYS, "head"])
    for source in points.to_dict("records"):
        row = dict(source)
        key = tuple(source[name] for name in KEYS)
        formulation = source["formulation"]
        heads = ("regression", "classification") if formulation == "joint" else (formulation,)
        for head in heads:
            selected = indexed.loc[(*key, head)]
            _match_frozen_point(source, selected, head)
            _copy_metric(row, "scalar_primary" if head == "regression" else "class_BA", selected)
        row["classification_decision_origin"] = ("scalar_forecast_projected_by_frozen_Train_label_map"
            if formulation == "regression" else "native_probability_head_argmax_divided_by_Train_priors")
        panel = inference.loc[inference.block_days.eq(primary_days)]
        for name, value in zip(KEYS, key):
            panel = panel.loc[panel[name].eq(value)]
        if formulation == "regression":
            proxy = panel.loc[panel["head"].eq("classification") & panel.metric.eq("balanced_accuracy")]
            if len(proxy) > 1:
                preferred = (proxy.loc[proxy.family.eq("ba_chance")] if "family" in proxy
                    else proxy.loc[proxy.role.eq("auxiliary")])
                proxy = preferred if len(preferred) == 1 else proxy
            if len(proxy) == 1:
                expected = _number(source.get("balanced_accuracy"))
                observed = _number(proxy.iloc[0].get("estimate"))
                if expected is not None and observed is not None and not np.isclose(expected, observed, rtol=1e-8, atol=1e-12):
                    raise ValueError("scalar BA proxy differs from the frozen336 point table")
                _copy_metric(row, "class_BA", proxy.iloc[0])
                row["class_BA_is_auxiliary_scalar_proxy"] = True
            elif len(proxy) > 1:
                raise ValueError("regression scalar BA proxy has ambiguous baseline comparisons")
        for metric in ("pearson_ic", "spearman_ic"):
            if formulation == "classification":
                continue
            current = panel.loc[panel["head"].eq("regression") & panel.metric.eq(metric)]
            preferred = current.loc[current.role.eq("primary")]
            if len(preferred) == 1:
                current = preferred
            elif len(current) > 1:
                # An absolute-IC row can accompany paired IC comparisons.
                absolute = current.loc[current.baseline.fillna("").isin(("", "none", "zero_correlation", "zero"))]
                if len(absolute) == 1:
                    current = absolute
            if len(current) == 1:
                result = current.iloc[0]
                for field in ("estimate", "ci_low", "ci_high"):
                    row["native_" + metric + "_" + field] = _number(result[field])
        row.update(primary_block_days=primary_days, historical_test_exploratory=True, independently_unused_holdout=False)
        table.append(row)
    result = pd.DataFrame(table)
    _validate_points(result)
    return result


def _format(value, percent=False):
    number = _number(value)
    if number is None:
        return "—"
    return f"{number * 100:.2f}%" if percent else f"{number:.5g}"


def _focus(summary, task):
    candidates = summary.loc[summary.task.eq(task) & summary.formulation.eq("classification")]
    row = candidates.sort_values("class_BA_estimate", ascending=False, kind="stable").iloc[0]
    delta = _number(row.get("class_BA_improvement"))
    low, high = _number(row.get("class_BA_improvement_ci_low")), _number(row.get("class_BA_improvement_ci_high"))
    difference = "—" if delta is None else f"{delta * 100:+.2f} pt"
    ci = "—" if low is None or high is None else f"[{low * 100:+.2f}, {high * 100:+.2f}] pt"
    return {"task": task, "task_label": TASK_LABELS[task], "group": row.group, "model": row.model,
        "ba": _number(row.get("class_BA_estimate")), "ba_ci_low": _number(row.get("class_BA_ci_low")),
        "ba_ci_high": _number(row.get("class_BA_ci_high")), "baseline": row.get("class_BA_baseline"),
        "baseline_ba": _number(row.get("class_BA_baseline_estimate")), "improvement": delta,
        "improvement_ci_low": low, "improvement_ci_high": high,
        "p_holm_primary448": _number(row.get("class_BA_p_holm_primary448")),
        "p_holm_global": _number(row.get("class_BA_p_holm_global")),
        "significant_improvement_holm448": _true(row.get("class_BA_significant_improvement_holm448", False)),
        "display_difference": difference, "display_difference_ci": ci,
        "selection": "largest native classification BA within previously inspected Test; exploratory"}


def _html(summary, focus, stats):
    rows = json.loads(summary.to_json(orient="records"))
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    focus_rows = []
    for row in focus:
        focus_rows.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in (
            row["task_label"], row["group"] + " / " + row["model"], _format(row["ba"], True),
            f'[{_format(row["ba_ci_low"], True)}, {_format(row["ba_ci_high"], True)}]', row["baseline"],
            _format(row["baseline_ba"], True), row["display_difference"], row["display_difference_ci"],
            _format(row["p_holm_primary448"]), _format(row["p_holm_global"]))) + "</tr>")
    counts = ", ".join(f"{html.escape(TASK_LABELS[r['task']])}: {r['n_significant_improvements_global']}/{r['n_primary_tests']}"
        for r in stats["per_task"])
    return '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>336条件のベースライン差・信頼区間</title>
<style>body{font-family:system-ui;margin:24px;line-height:1.65}p{max-width:1200px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccc;text-align:right}th{position:sticky;top:0;background:white}select{margin:5px;padding:6px}.table{overflow:auto;max-height:65vh}.note{background:#f2f5f8;padding:12px}</style>
<h1>336条件：単純ベースラインとの差を検証</h1>
<p>同じTest時点を組にした時系列Block Bootstrap。主解析は7日ブロック、1日・28日は感度確認です。
表の95%信頼区間は各比較の区間であり、同時信頼区間ではありません。優位性の判定には補正済みP値を使います。
改善量はすべて「正ならモデルが優れる」向きです。分類の差はパーセントポイント(pt)、回帰は対象指標の単位です。</p>
<p class="note">過去に確認したTest期間に対する探索的な事後解析です。新しい未使用Holdoutでの再現確認ではありません。
モデル・閾値・checkpointは変更していません。学習seedは1つで、区間は再学習seedの不確実性を含みません。
336条件中、線形・LightGBMのjoint 56条件は既存の独立した回帰・分類headを束ねた対照で、予測を再利用しています。
各joint条件には回帰・分類の2つの主検定があり、主解析の検定数は448です。対照の重複も含めた保守的なHolm補正を適用します。</p>
<p><a href="results_336_significance.csv">全336条件CSV</a> / <a href="all_inference.csv">全CI・検定CSV</a> /
<a href="summary.json">集計・注記JSON</a> / <a href="report_contract.json">入力hash・集計契約</a></p>
<h2>特に確認したい分散・急落分類</h2><p>既存Testの中でBAが最大だった分類条件を表示します。この選択も探索的です。
比較相手は単純な定数予測ではなく、Trainだけで学習した少数の因果的特徴による分類ベースラインです。
BAの偶然基準は3クラス33.33%、急落の2クラス50%ですが、主判定はベースラインとの差です。</p>
<div class="table"><table><thead><tr><th>対象</th><th>入力 / モデル</th><th>BA</th><th>95%CI</th><th>比較相手</th><th>基準BA</th><th>改善量</th><th>改善量95%CI</th><th>Holm448 P</th><th>全追加検定Holm P</th></tr></thead><tbody>''' + "".join(focus_rows) + '''</tbody></table></div>
<p>全追加検定のHolm補正後に有意な正の改善（主検定448組の内訳）：''' + html.escape(counts) + '''。</p>
<h2>全336条件</h2><p>回帰からクラス化したBAは補助評価です。分類headのBAと混同しないよう判定元を表示します。
Pearson/Spearman ICの区間はCSVに併記します。合成指標の優位性・回復度は実取引の損益や回復時間の予測を意味しません。</p>
<div id="filters"></div><p id="count"></p><div class="table"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<h2>既存検定との違い</h2><p>今回追加したのは各条件の絶対指標の区間と、固定した単純ベースラインとの差です。
既存の「分類−回帰のBA差」112組と「joint−単独head」112組のニューラル比較は、別の仮説・別のHolm112検定群です。
<a href="existing_comparisons.csv">既存比較の一覧</a>は元の検定群を保持し、今回のベースライン検定に混ぜません。</p>
<p>分散は未来16本の15分リターン二乗和、下落幅は現在価格から未来の最安closeまでの低下率です。
急落は下落幅1.2%以上の2クラスです。下落閾値超過量は発生確率ではなく超過幅です。</p>
<script>const data=''' + payload + ''';
const dimensions=['group','model','task','formulation'];const filters={};
for(const key of dimensions){const label=document.createElement('label');label.textContent=key;const select=document.createElement('select');for(const value of ['すべて',...new Set(data.map(x=>x[key]))]){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option)}select.onchange=draw;filters[key]=select;label.append(select);document.getElementById('filters').append(label)}
const columns=['group','model','task','formulation','scalar_primary_metric','scalar_primary_estimate','scalar_primary_ci_low','scalar_primary_ci_high','scalar_primary_baseline','scalar_primary_improvement','scalar_primary_improvement_ci_low','scalar_primary_improvement_ci_high','scalar_primary_p_holm_primary448','scalar_primary_p_holm_global','class_BA_estimate','class_BA_ci_low','class_BA_ci_high','class_BA_baseline','class_BA_improvement','class_BA_improvement_ci_low','class_BA_improvement_ci_high','class_BA_p_holm_primary448','class_BA_p_holm_global','classification_decision_origin'];
function draw(){const rows=data.filter(row=>dimensions.every(key=>filters[key].value==='すべて'||row[key]===filters[key].value));document.getElementById('count').textContent=rows.length+' 条件';document.getElementById('head').replaceChildren();const header=document.createElement('tr');for(const key of columns){const cell=document.createElement('th');cell.textContent=key;header.append(cell)}document.getElementById('head').append(header);const body=document.getElementById('body');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');for(const key of columns){const td=document.createElement('td');const value=row[key];td.textContent=value===null||value===undefined?'—':typeof value==='number'?value.toPrecision(6):String(value);tr.append(td)}body.append(tr)}}draw();</script></html>'''


def write_significance_report(output_dir, config):
    """Write exactly336 rows,448 native-head comparisons, and a linked HTML.

    The runner must supply already-adjusted inference. No claim of a newly
    unused Holdout is permitted even when all baseline differences are strong.
    """
    root = Path(output_dir).resolve()
    destination = root / "final"
    source = destination / "all_inference.csv"
    points_path = Path(config.get("point_table_path", config.get("source_results_csv",
        "codex_outputs/wm_joint_grid_20261111/final/results_336.csv"))).resolve()
    points = pd.read_csv(points_path)
    _validate_points(points)
    inference = _normalize(pd.read_csv(source))
    primary_days = int(config.get("statistics", {}).get("primary_block_days", 7))
    if primary_days != 7:
        raise ValueError("the registered primary block length is seven days")
    primary = _select_primary(inference, primary_days)
    summary = _combine(points, inference, primary, primary_days)
    result_path = destination / "results_336_significance.csv"
    summary.to_csv(result_path, index=False)
    focus = [_focus(summary, task) for task in ("variance", "crash")]
    positive = pd.to_numeric(primary.improvement, errors="coerce").gt(0)
    rejected = pd.to_numeric(primary.p_holm_primary448, errors="coerce").lt(.05)
    global_rejected = (pd.to_numeric(primary.p_holm_global, errors="coerce").lt(.05) if "p_holm_global" in primary
        else pd.Series(False, index=primary.index))
    primary = primary.assign(_improved=positive & rejected, _global_improved=positive & global_rejected)
    per_task = [{"task": task, "n_primary_tests": len(part), "n_significant_improvements": int(part._improved.sum()),
        "n_significant_improvements_global": int(part._global_improved.sum())}
        for task, part in primary.groupby("task", sort=False)]
    stats = {"complete": True, "cells": 336, "native_primary_tests": 448,
        "n_significant_improvements_holm448": int(primary._improved.sum()), "per_task": per_task,
        "n_significant_improvements_global": int(primary._global_improved.sum()),
        "focus": focus, "primary_block_days": primary_days,
        "confidence_interval": "individual95%; not simultaneous intervals", "test_selection_allowed": False,
        "historical_seen_test": True, "independently_unused_holdout": False,
        "single_training_seed": True, "classical_joint_reuses_existing_heads": True,
        "n_reused_classical_joint_conditions": 56, "n_distinct_empirical_conditions": 280,
        "primary_family": "448 native-head baseline tests, including duplicated classical joint controls",
        "global_holm_family_size": int(config.get("statistics", {}).get("global_family_size", 1056)),
        "existing_comparisons_retained_as_separate_families": True}
    _json(destination / "summary.json", stats)
    (destination / "index.html").write_text(_html(summary, focus, stats), encoding="utf-8")
    contract = {**stats, "inference_csv_sha256": _sha(source), "source_results_336_sha256": _sha(points_path),
        "results_336_significance_sha256": _sha(result_path), "report_source_sha256": _sha(Path(__file__)),
        "source_results_336_path": str(points_path), "inference_csv_path": str(source),
        "no_models_fitted": True, "no_test_based_threshold_selection": True,
        "no_p_values_adjusted_by_report": True}
    _json(destination / "report_contract.json", contract)
    completion = {"complete": True, "report_complete": True, "cells": 336,
        "n_primary_tests": 448, "report_dir": str(destination), "results_csv": str(result_path),
        "results_csv_sha256": _sha(result_path), "html_sha256": _sha(destination / "index.html"),
        "summary_sha256": _sha(destination / "summary.json"), "contract_sha256": _sha(destination / "report_contract.json")}
    _json(destination / "report_completion.json", completion)
    return completion
