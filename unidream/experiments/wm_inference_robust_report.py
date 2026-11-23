"""Japanese report and exportable figures for the registered robustness audit.

Only already-written final CSVs are read, after the new analysis registration.
This module neither trains a model nor publishes a completion marker.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSVS = ("bootstrap_inference.csv", "reality_check.csv", "hac_inference.csv", "summary.csv",
    "power.csv", "mde.csv", "required_years.csv", "calibration.csv", "acf.csv",
    "fold_diagnostics.csv", "legacy_replay.csv")
OPTIONAL_CSVS = ("ablation_robustness.csv", "ablation_hac.csv", "ablation_primary.csv", "expanded_family20.csv", "automatic_blocks.csv")
LABELS = {"bc_A": "BC A：市場", "bc_B": "BC B：市場＋予測", "bc_C": "BC C：市場＋WM潜在",
    "bc_D": "BC D：すべて", "rl_A": "RL A：市場", "rl_B": "RL B：市場＋予測",
    "rl_C": "RL C：市場＋WM潜在", "rl_D": "RL D：すべて"}
BENCHMARKS = {"bh": "B&H（1.0倍）", "fixed50": "0.50目標の一定エクスポージャー"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(series):
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "1.0"))


def _primary(frame):
    return frame.loc[_boolean(frame["primary_inference"])].copy() if "primary_inference" in frame else frame.copy()


def _number(value, digits=3):
    try:
        return f"{float(value):.{digits}g}" if np.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return "—"


def _pct(value):
    try:
        return f"{float(value)*100:+.2f}" if np.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return "—"


def _table(frame, columns, *, percent=(), limit=None):
    present = [column for column in columns if column in frame]
    if not present or frame.empty:
        return '<p class="muted">この項目は記録されていません。CSVと監査情報を参照してください。</p>'
    selected = frame[present].head(limit).copy() if limit else frame[present].copy()
    for column in present:
        if column == "condition":
            selected[column] = selected[column].map(lambda value: LABELS.get(value, str(value)))
        elif column == "benchmark":
            selected[column] = selected[column].map(lambda value: BENCHMARKS.get(value, str(value)))
        elif column in percent:
            selected[column] = selected[column].map(_pct)
        elif pd.api.types.is_numeric_dtype(selected[column]) and not pd.api.types.is_bool_dtype(selected[column]):
            selected[column] = selected[column].map(_number)
    selected.rename(columns={key: columns[key] for key in present}, inplace=True)
    return '<div class="scroll">' + selected.to_html(index=False, escape=True, na_rep="—", border=0) + '</div>'


def _count_significant(frame, benchmark="bh"):
    panel = frame.loc[frame.benchmark.eq(benchmark)] if "benchmark" in frame else frame.iloc[0:0]
    n_conditions = len(panel)
    if "inference_available" in panel:
        panel = panel.loc[_boolean(panel.inference_available)]
    if "p_holm_family" not in panel:
        return 0, n_conditions, 0
    corrected = pd.to_numeric(panel.p_holm_family, errors="coerce")
    estimate = pd.to_numeric(panel.estimate, errors="coerce")
    return int(((corrected < .05) & (estimate > 0)).sum()), n_conditions, int(((corrected < .05) & (estimate < 0)).sum())


def _figures(final, primary, hac, power):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures = []
    names = list(LABELS)
    available = [name for name in ("bh", "fixed50") if "benchmark" in primary and primary.benchmark.eq(name).any()]
    if available:
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
            fig, axes = plt.subplots(1, len(available), figsize=(6.2*len(available), 6.2), squeeze=False, sharey=True)
            for axis, benchmark in zip(axes[0], available):
                for offset, frame, color, marker in ((-.12, primary, "#1565c0", "o"), (.12, hac, "#d66b00", "s")):
                    panel = frame.loc[frame.benchmark.eq(benchmark)] if "benchmark" in frame else frame.iloc[0:0]
                    for y, condition in enumerate(names):
                        row = panel.loc[panel.condition.eq(condition)] if "condition" in panel else panel.iloc[0:0]
                        if len(row) != 1:
                            continue
                        point, low, high = [pd.to_numeric(row[col], errors="coerce").iloc[0] for col in ("estimate", "ci_low", "ci_high")]
                        if np.isfinite([point, low, high]).all():
                            axis.hlines(y+offset, low*100, high*100, color=color, linewidth=1.5)
                            axis.plot(point*100, y+offset, marker, color=color, markersize=5)
                axis.axvline(0, color="#555", linewidth=.9)
                axis.set_title("vs B&H (1x)" if benchmark == "bh" else "vs 0.50 target exposure (secondary)")
                axis.set_xlabel("Annualized net log-growth difference (percentage points)")
                axis.grid(axis="x", alpha=.2)
                axis.set_yticks(np.arange(len(names)), [name.replace("_", " ").upper() for name in names])
                axis.set_ylim(len(names)-.5, -.5)
            fig.legend(handles=[Line2D([0], [0], color="#1565c0", marker="o", label="Circular 7-day basic 95% CI"),
                Line2D([0], [0], color="#d66b00", marker="s", label="Bartlett HAC lag 6 normal 95% CI")],
                loc="lower center", ncol=2, frameon=False)
            fig.suptitle("Fixed trained-seed average; pointwise intervals, not simultaneous intervals")
            fig.tight_layout(rect=(0, .06, 1, .95))
            for extension in ("png", "svg"):
                path = final / ("primary_forest."+extension)
                fig.savefig(path, dpi=180, bbox_inches="tight")
                figures.append(path)
            plt.close(fig)
    needed = {"benchmark", "condition", "mode", "alpha", "effect_scenario", "annual_log_growth_effect", "eligible_years", "prospective_power"}
    if needed.issubset(power) and len(power):
        panel = power.loc[power["mode"].eq("market_only_fixed_seed_average")
            & power.effect_scenario.eq("prespecified_effect")
            & np.isclose(pd.to_numeric(power.alpha, errors="coerce"), .05/8)].copy()
        benchmarks = [name for name in ("bh", "fixed50") if panel.benchmark.eq(name).any()]
        effects = [effect for effect in (.02, .05, .10) if np.isclose(pd.to_numeric(panel.annual_log_growth_effect, errors="coerce"), effect).any()]
        if benchmarks and effects:
            with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
                fig, axes = plt.subplots(len(benchmarks), len(effects), figsize=(4.7*len(effects), 3.5*len(benchmarks)), squeeze=False, sharey=True)
                colors = plt.get_cmap("tab10")
                for i, benchmark in enumerate(benchmarks):
                    for j, effect in enumerate(effects):
                        axis = axes[i, j]
                        for k, condition in enumerate(names):
                            values = panel.loc[panel.benchmark.eq(benchmark) & panel.condition.eq(condition)
                                & np.isclose(panel.annual_log_growth_effect, effect)].sort_values("eligible_years")
                            if len(values):
                                axis.plot(values.eligible_years, values.prospective_power, marker="o", markersize=3,
                                    linewidth=1.2, color=colors(k), label=condition.replace("_", " ").upper())
                        axis.axhline(.8, color="#777", linestyle="--", linewidth=.8)
                        axis.axhline(.9, color="#aaa", linestyle=":", linewidth=.8)
                        axis.set_ylim(0, 1.02)
                        axis.set_title(f"{'B&H' if benchmark == 'bh' else '0.50 target'}; effect={effect*100:.0f} log-growth pp/year")
                        axis.set_xlabel("Eligible observed-bar-equivalent years")
                        axis.set_ylabel("Prospective normal-approximation power")
                        axis.grid(alpha=.2)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False)
                fig.suptitle("Planning only: stationary plug-in LRV; two-sided alpha=0.05/8; not future-performance evidence")
                fig.tight_layout(rect=(0, .06, 1, .94))
                for extension in ("png", "svg"):
                    path = final / ("power_curves."+extension)
                    fig.savefig(path, dpi=180, bbox_inches="tight")
                    figures.append(path)
                plt.close(fig)
    return figures


def write_report(output_dir, cfg):
    """Read registered final tables, export HTML/figures, and return their SHAs."""
    root = Path(output_dir).resolve()
    registration_path = root / "posthoc_registration.json"
    if not registration_path.exists():
        raise ValueError("new robustness registration required before any result-table access")
    registration_sha = _sha(registration_path)
    registration = json.loads(registration_path.read_text())
    if registration.get("config") is not None and registration["config"] != cfg:
        raise ValueError("report config differs from the registered robustness analysis")
    final = root / "final"
    csv_names = list(INPUT_CSVS) + [name for name in OPTIONAL_CSVS if (final / name).exists()]
    files = [final / name for name in csv_names] + [final / "audit.json"]
    before = {str(path): _sha(path) for path in files}
    tables = {}
    for name in csv_names:
        try:
            tables[name.removesuffix(".csv")] = pd.read_csv(final / name, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            tables[name.removesuffix(".csv")] = pd.DataFrame()
    audit = json.loads((final / "audit.json").read_text())
    bootstrap = tables["bootstrap_inference"]
    needed = {"benchmark", "condition", "primary_inference", "estimate", "ci_low", "ci_high", "p_value", "p_holm_family"}
    if not needed.issubset(bootstrap):
        raise ValueError("registered bootstrap primary inference columns required")
    primary = _primary(bootstrap)
    if primary.duplicated(["benchmark", "condition"]).any():
        raise ValueError("each benchmark/condition must have one primary bootstrap inference row")
    expected = {(benchmark, condition) for benchmark in ("bh", "fixed50") for condition in LABELS}
    if set(primary[["benchmark", "condition"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("complete eight-condition B&H and fixed50 primary panels required")
    hac = tables["hac_inference"]
    if "lag_days" in hac:
        hac = hac.loc[pd.to_numeric(hac.lag_days, errors="coerce").eq(6)]
    if "estimand" in hac:
        hac = hac.loc[hac.estimand.eq("market_fixed_seeds")]
    hac = _primary(hac)
    if (not {"benchmark", "condition"}.issubset(hac)
            or hac.duplicated(["benchmark", "condition"]).any()
            or set(hac[["benchmark", "condition"]].itertuples(index=False, name=None)) != expected):
        raise ValueError("complete unique lag6 fixed-seed HAC panels required")
    figures = _figures(final, primary, hac, tables["power"])
    bh_positive, bh_n, bh_negative = _count_significant(primary)
    hac_positive, hac_n, _ = _count_significant(hac)
    fixed_positive, fixed_n, _ = _count_significant(primary, "fixed50")
    panel = audit.get("panel", {})
    n_bars, n_days, n_segments, n_seeds = (panel.get(key, "—") for key in
        ("n_unique_market_rows", "n_daily_units", "n_segments", "n_training_seeds"))
    years = _number(panel.get("eligible_years"), 4)
    body = ['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>B&H差の頑健性・検出力の検証</title><style>body{font:16px system-ui;line-height:1.8;margin:32px auto;padding:0 24px;max-width:1500px;color:#17212e}h1,h2{line-height:1.4}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:8px 11px;border-bottom:1px solid #d7dde3;text-align:right}th{background:#edf2f7}.scroll{overflow:auto;max-height:680px}.muted{color:#596675}.lead{padding:16px;background:#edf5ff;border-left:4px solid #2868ad}.note{padding:12px;background:#fff8e6}img{max-width:100%;height:auto}a{color:#175fa2}code{font-size:.9em}</style>',
        '<h1>B&Hとの差は、推定法を変えても支持されるか</h1>',
        f'<p class="lead">主判定は、固定した学習seedの平均に対する7日循環Block Bootstrapです。B&Hを有意に上回る条件は <strong>{bh_positive}/{bh_n}</strong>。有意に下回る条件は {bh_negative}/{bh_n}。HAC（lag 6）の改善は {hac_positive}/{hac_n}。副基準の0.50目標との比較では改善 {fixed_positive}/{fixed_n} でした。いずれも該当する比較族のHolm補正後です。</p>',
        '<p>既存の結果を見た後に追加した頑健性検証です。モデル学習・チェックポイント・Val選択・売買条件を固定し、推定方法と不確実性を検査しています。有意でないことは、効果がゼロである証明や、十分な検出力があったことの証明にはなりません。</p>',
        f'<p>共通の市場観測は <strong>{html.escape(str(n_bars))}本</strong>、{html.escape(str(n_days))}日単位、{html.escape(str(n_segments))}区間、学習seedは{html.escape(str(n_seeds))}個。{years}年は観測バー数を年35040本で割った<strong>適格観測時間の換算</strong>で、連続したカレンダー運用年数ではありません。全seedは同じ市場を観測しており、市場標本数をseed数倍にはしません。</p>',
        '<h2>主判定とHAC</h2><p>B&Hとの8比較を主族としてHolm補正します。0.50目標との8比較は別の副族です。16比較をまとめた補正は追加の厳しい感度確認です。図の95%区間は<strong>各比較の点ごとの区間</strong>で、同時信頼区間ではありません。p値の多重比較補正とは役割が異なります。</p>']
    columns = {"benchmark": "比較基準", "condition": "条件", "estimate": "年率対数成長率差(pt)",
        "ci_low": "95%下限(pt)", "ci_high": "95%上限(pt)", "p_value": "両側p", "p_holm_family": "族内Holm p"}
    body.append(_table(primary, columns, percent=("estimate", "ci_low", "ci_high")))
    if any(path.name == "primary_forest.png" for path in figures):
        body.append('<p><a href="primary_forest.png"><img alt="循環bootstrapとHACの点ごとの95パーセント信頼区間" src="primary_forest.png"></a> <a href="primary_forest.svg">SVG</a></p>')
    body += ['<p>主判定は、既に学習した3seedを固定した市場系列の不確実性です。階層seed再標本化は、別の学習seed集団へ一般化する仮定を加えた感度分析で、同じ推定対象として混同しません。HACも固定seed平均が主で、seed分散を加える方法は近似的な感度確認です。後者の仮定だけで厳密なp値や被覆率を保証できません。</p>',
        '<p>HACは日ごとの合計超過対数リターンと共通の観測バー数を使う比率推定です。Bartlett重みで同一区間・同じカレンダー日差の共分散を集計し、欠損日やfold境界はまたぎません。主lag 6とlag 0/13/27/29を保存し、正規近似とD/(D−1)の補正を明記しています。有限標本補正は正規近似の正確さを保証しません。</p>',
        '<h2>旧実装の再現とブロック端点の重み</h2>',
        '<p>旧実装にも、観測推定値を引く形の帰無仮説中心化がありました。今回点検したのは「中心化をし忘れていた」という問題ではありません。非循環の移動ブロックは端の観測が選ばれる回数が少なく、最終ブロックの切り詰めでも有限標本の重みが変わります。そのため、中心化した再標本分布の平均が厳密にゼロになるとは限りません。</p>',
        '<p>循環法は区間内の端点を折り返し、stationary法は平均長を固定した幾何分布のブロックを使います。実際の欠損やfold境界をまたがず、同一区間の終端と先頭を人工的につなぐ仮定は残ります。ブロック以下の短い区間は固定した条件付き推論で、短いブロックへの切り替えは行いません。ブロック長や推定法を結果が良いものへ選び直すことはしません。</p>']
    replay = audit.get("legacy_p_replay", {})
    body.append(f'<p>旧p値の再現最大絶対差：<strong>{_number(replay.get("max_absolute_difference"), 6)}</strong>。実際の全件は<a href="legacy_replay.csv">再現CSV</a>、端点重みの合成例は<a href="audit.json">監査JSON</a>に保存しています。</p>')
    summary_columns = {"benchmark": "基準", "condition": "条件", "estimate": "新・年率差(pt)",
        "originallegacy_p_value": "旧p", "original_legacy_p_value": "旧p", "legacy_p_value": "旧p",
        "originallegacy_ci_low": "旧区間下限", "originallegacy_ci_high": "旧区間上限",
        "p_holm_16_primary_controls": "16比較Holm p", "cumulative_return_difference_descriptive": "累積収益差・記述値(pt)"}
    body.append(_table(tables["summary"], summary_columns, percent=("estimate", "cumulative_return_difference_descriptive")))
    if "ablation_primary" in tables:
        body += ['<h2>元の12比較も同じ新手法で再評価</h2>',
            '<p>B−A、C−A、D−B、D−CをBC/RLそれぞれで比較し、各条件のRL−BCを加えた元の12比較にも、同じ日単位・同じ対応seed・新しいブロック法を適用しています。主判定は7日循環法・固定seed平均です。モデルや売買方策を追加学習した結果ではありません。</p>',
            '<p>20比較の拡張補正は、ここで再計算した新しい12比較とB&Hとの差8比較を合わせます。旧非循環法の12個のp値をそのまま混ぜる補正ではありません。この補正も420条件や57ヘッドの歴史的選択を取り消しません。</p>']
        body.append(_table(tables["ablation_primary"], {"condition": "比較", "comparison": "比較",
            "estimate": "年率対数成長率差(pt)", "ci_low": "95%下限(pt)", "ci_high": "95%上限(pt)",
            "p_value": "両側p", "p_holm_family": "12比較Holm p", "p_holm_20": "拡張20比較p", "p_holm20_sensitivity": "拡張20比較p"},
            percent=("estimate", "ci_low", "ci_high")))
    if "expanded_family20" in tables:
        body.append(_table(tables["expanded_family20"], {column: {"condition": "比較", "comparison": "比較",
            "p_value": "新・補正前p", "p_holm_20": "拡張20比較Holm p", "family": "比較族"}.get(column, column)
            for column in tables["expanded_family20"].columns}, limit=24))
    body += ['<h2>Reality Checkと特徴選択の範囲</h2>',
        '<p>Reality Checkは、同じ市場日とseedを対応させた8条件の最大改善に対する片側検定です。B&Hそのもののゼロ差も含むため、全候補の観測差が非正なら全体p値は1になります。二側の個別Holm検定とは問いが異なります。</p>',
        '<p class="note">この8条件の補正は、過去の420予測条件から57ヘッド・121列を選んだ探索を取り消しません。入力特徴や研究方針を既存のTest結果から選んだ影響は残ります。独立した未使用Holdoutによる確認ではありません。</p>']
    rc = _primary(tables["reality_check"])
    body.append(_table(rc, {"benchmark": "基準", "condition": "条件", "global_reality_check_p": "8条件全体・片側p",
        "max_adjusted_one_sided_p": "最大値補正・片側p", "one_sided_p": "個別・片側p"}))
    body += ['<h2>帰無仮説が正しい合成系列での校正</h2>',
        '<p>真の平均がゼロのIID・AR(1)・GARCH・AR(1)+GARCHを使い、単一区間と6区間で同じ系列を比較します。各標本の平均を強制的にゼロへ変えることはしません。名目5%での棄却率、95%区間の被覆率、Wilson区間を保存しています。合成校正は実データの優位性、全ての市場レジーム、多重比較やseed不確実性の正しさまで保証しません。</p>']
    body.append(_table(tables["calibration"], {"layout": "区間構成", "scenario": "真の系列", "method": "方法",
        "outer_replicates": "反復数", "empirical_type1_error": "第一種過誤率", "type1_mc_ci_low": "MC下限",
        "type1_mc_ci_high": "MC上限", "empirical_coverage_95": "95%被覆率", "coverage_mc_ci_low": "MC下限",
        "coverage_mc_ci_high": "MC上限"}))
    body += ['<h2>どれほどの効果を検出できる設計か</h2>',
        '<p>以下は将来の設計を考えるための検出力・最小検出効果（MDE）です。年率対数成長率差2/5/10ポイントを想定し、検出力80%/90%、二側α=0.05と0.05/8を計算します。後者はHolmの計画用の保守的なBonferroni近似です。</p>',
        '<p>現在推定した長期分散が今後も一定で、標準誤差が観測時間の平方根に反比例するという仮定に基づきます。レジームが変われば崩れます。必要年数は適格観測時間であり、その年数でデータが集まる約束ではありません。観測効果を使った欄は同じデータで選ばれた楽観的なシナリオになり得るため、「観測後の検出力」を帰無仮説への証拠にしません。</p>',
        '<p>固定した少数学習seedの分散を加える近似では、追加の市場年数でその床を割りません。その仮定の下で達成不能な目標は明示し、必要年数を単純に外挿しません。</p>']
    if any(path.name == "power_curves.png" for path in figures):
        body.append('<p><a href="power_curves.png"><img alt="所定の効果に対する将来設計用の検出力曲線" src="power_curves.png"></a> <a href="power_curves.svg">SVG</a></p>')
    mde = tables["mde"]
    if {"benchmark", "mode", "alpha", "target_power", "eligible_years"}.issubset(mde):
        current_year = panel.get("eligible_years", np.nan)
        mde = mde.loc[mde.benchmark.eq("bh") & mde["mode"].eq("market_only_fixed_seed_average")
            & np.isclose(mde.alpha, .05/8) & np.isclose(mde.target_power, .8)
            & (mde.eligible_years.isin([1, 3, 5, 10]) | np.isclose(mde.eligible_years, current_year))]
    body.append(_table(mde, {"benchmark": "基準", "condition": "条件", "eligible_years": "適格観測年",
        "target_power": "検出力", "mde_annual_log_growth": "MDE・年率対数差(pt)"}, percent=("mde_annual_log_growth",), limit=40))
    required = tables["required_years"]
    needed_required = {"benchmark", "mode", "target_power", "effect_scenario", "annual_log_growth_effect"}
    if needed_required.issubset(required):
        required = required.loc[required.benchmark.eq("bh") & required["mode"].eq("market_only_fixed_seed_average")
            & np.isclose(required.target_power, .8)
            & (required.effect_scenario.eq("observed_selected_effect_scenario_not_power_proof")
                | np.isclose(required.annual_log_growth_effect, .05))]
    body += ['<h3>80%検出力に必要な適格観測年数のシナリオ</h3>',
        '<p>年率対数成長率差5ptをあらかじめ想定する設計と、今回観測した差を仮に真の差と置く感度分析です。alpha=0.05は将来1比較を固定する場合、0.00625は8比較を保守的に計画する場合です。観測差のシナリオは過大評価の可能性があり、有意性の証拠には使いません。分散・市場分布が続く仮定による計算値で、将来のカレンダー期間の予言ではありません。</p>']
    body.append(_table(required, {"condition": "条件", "alpha": "両側alpha", "effect_scenario": "効果の仮定",
        "annual_log_growth_effect": "想定年率対数差(pt)", "required_eligible_years": "必要適格観測年",
        "unattainable_with_fixed_seed_floor": "固定seed分散で到達不可"}, percent=("annual_log_growth_effect",), limit=32))
    body += ['<h2>系列依存・区間・売買会計の監査</h2>',
        '<p>超過リターンの日平均、その二乗・絶対値、fold内で平均を引いた系列のACFをlag 1/2/3/7/14/30で確認します。共通観測数が異なる部分日は日平均のばらつきにも影響するため、これらは診断値として扱います。ACFを追加の優位性検定にはしません。</p>',
        '<p>B&Hは初期保有1倍、0.50目標の副基準も初期保有1倍から開始し、同じ一足遅延・変化上限・費用・借入条件で目標へ移ります。欠損区間とfold境界は全条件でリセットします。費用控除後の系列を同じバーで照合した監査をJSONに保存しています。市場への平均投資比率が違うため、基準との差だけで売買タイミング能力を特定できません。</p>']
    folds = tables["fold_diagnostics"]
    body.append(_table(folds, {"benchmark": "基準", "condition": "条件", "fold": "fold", "n_market_bars": "観測バー",
        "mean_daily_per_bar_log_excess": "平均超過対数リターン", "daily_per_bar_std": "日平均の標準偏差"}, limit=32))
    if "automatic_blocks" in tables:
        body.append('<p>補正済みPolitis–White型の自動ブロック長推定も、各連続区間で診断として計算しました。短い区間では推定が不安定になり得ます。主判定の7日や感度分析の長さをこの結果で選び直していません。</p>')
        body.append(_table(tables["automatic_blocks"], {"benchmark": "基準", "condition": "条件", "segment": "区間",
            "n_days": "観測日数", "stationary_days": "stationary推奨日", "circular_days": "循環法推奨日"}, limit=16))
    body.append('<h2>全件データと出典</h2><ul>' + ''.join(f'<li><a href="{name}">{name}</a></li>' for name in csv_names) + '<li><a href="audit.json">audit.json</a></li></ul>')
    references = cfg.get("references", [])
    if references:
        body.append('<ul>' + ''.join(f'<li><a href="{html.escape(str(ref.get("url", "")), quote=True)}">{html.escape(str(ref.get("title", ref.get("url", "出典"))))}</a></li>' for ref in references if isinstance(ref, dict)) + '</ul>')
    body.append('</html>')
    for path in files:
        if _sha(path) != before[str(path)]:
            raise ValueError("result table changed during report rendering; refusing inconsistent report")
    if _sha(registration_path) != registration_sha:
        raise ValueError("registration changed while rendering")
    path = final / "index.html"
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(''.join(body), encoding="utf-8")
    temporary.replace(path)
    outputs = [path] + figures
    return {"report_path": str(path), "report_sha256": _sha(path), "figures": {p.name: {"path": str(p), "sha256": _sha(p)} for p in figures},
        "input_sha256": before, "primary_bh_significant_improvements": bh_positive,
        "primary_bh_conditions": bh_n, "completion_written": False,
        "artifact_sha256": {p.name: _sha(p) for p in outputs}}
