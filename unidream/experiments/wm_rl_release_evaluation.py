"""Report-only evaluation of the registered continued-RL release experiment.

The historical Test interval has already informed earlier research. This is an
exploratory conditional-cohort comparison, not an unused holdout. Shared market
days are resampled once across all arms and three training seeds; seed copies
never increase the independent market sample count.
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
from .wm_significance_statistics import holm_fixed_family


ARMS = ("A", "C")
CONDITIONS = ("control", "floor_abrupt", "floor_gradual", "prior_abrupt", "prior_gradual")
COHORT = ("fold", "decision_id", "timestamp", "bar_log_return")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job(fold, arm, seed, condition):
    return f"fold_{fold:02d}_{arm}_s{seed}_{condition}"


def _path(frame, config):
    required = {*COHORT, "requested_position", "executed_position", "turnover", "cost_fraction",
                "borrow_fraction", "net_log_return", "reset"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("a complete nonempty executed release ledger is required")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, errors="coerce")
    for name in ("fold", "decision_id"):
        values = pd.to_numeric(result[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError("integer finite fold/decision identifiers required")
        result[name] = values.astype(np.int64)
    result = result.sort_values("timestamp").reset_index(drop=True)
    numeric = ["bar_log_return", "requested_position", "executed_position", "turnover", "cost_fraction",
               "borrow_fraction", "net_log_return"]
    if (not np.isfinite(result[numeric].to_numpy(float)).all()
            or result.duplicated(["fold", "decision_id"]).any()):
        raise ValueError("unique complete finite executed cohort required")
    reconstructed = execute_positions(result.bar_log_return, result.requested_position,
                                      result.timestamp, result.fold, config)
    if not np.array_equal(result.reset.to_numpy(), reconstructed.reset.to_numpy()):
        raise ValueError("reset flags must match all gaps and fold boundaries")
    for name in ("executed_position", "turnover", "cost_fraction", "borrow_fraction", "net_log_return"):
        tolerance = 1e-7 if name in ("executed_position", "turnover") else 1e-12
        if not np.allclose(result[name], reconstructed[name], rtol=0, atol=tolerance):
            raise ValueError(f"recorded {name} differs from lagged fee/borrow accounting")
    return result


def _metrics(path, config):
    result = score_path(path, config)
    settings = accounting(config)
    position = path.executed_position.to_numpy(float)
    result.update(near_cash_fraction=float(np.mean(position <= .01)),
        floor_fraction=float(np.mean(position <= settings["position_min"] + .001)),
        upper_fraction=float(np.mean(position >= settings["position_max"] - .001)),
        leveraged_fraction=float(np.mean(position > 1.001)),
        final_position_min=settings["position_min"], final_position_max=settings["position_max"])
    return result


def _seed_summary(frame, keys):
    excluded = {"seed", "fold", "n_rows", "n_resets"}
    numeric = [key for key in frame.select_dtypes(include=[np.number]).columns if key not in excluded]
    rows = []
    for group, part in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(keys, group))
        row.update(n_training_seeds=len(part), n_unique_market_rows=int(part.n_rows.iloc[0]))
        for key in numeric:
            row[key + "_seed_mean"] = float(part[key].mean())
            row[key + "_seed_min"] = float(part[key].min())
            row[key + "_seed_max"] = float(part[key].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _contrasts():
    result = []
    for arm in ARMS:
        for left, right in (("floor_abrupt", "control"), ("floor_gradual", "control"),
                ("prior_abrupt", "control"), ("prior_gradual", "control"),
                ("floor_gradual", "floor_abrupt"), ("prior_gradual", "prior_abrupt")):
            result.append((f"{arm}_{left}_minus_{right}", "release_primary", (arm, left), (arm, right)))
    for condition in CONDITIONS:
        result.append((f"C_minus_A_{condition}", "wm_secondary", ("C", condition), ("A", condition)))
    for arm in ARMS:
        for condition in CONDITIONS:
            result.append((f"{arm}_{condition}_minus_val_exposure_matched", "risk_matched_secondary",
                           (arm, condition), (arm, condition, "matched")))
    return result


def _correct(frame, *, hac=False):
    result = frame.copy()
    grouping = ["lag_days"] if hac else ["method", "block_days", "estimand"]
    result["p_holm_family"] = np.nan
    result["p_holm_all17"] = np.nan
    result["p_holm_all27"] = np.nan
    for _, subset in result.groupby(grouping, sort=False):
        all_mapping = {str(i): float(row.p_value) if pd.notna(row.p_value) else 1.
                       for i, row in subset.iterrows()}
        all_q = holm_fixed_family(all_mapping, family_size=27)
        main = subset.loc[~subset.family.eq("risk_matched_secondary")]
        q17 = holm_fixed_family({str(i): all_mapping[str(i)] for i in main.index}, family_size=17)
        for _, group in subset.groupby("family", sort=False):
            expected = {"release_primary": 12, "wm_secondary": 5, "risk_matched_secondary": 10}[group.family.iloc[0]]
            if len(group) != expected:
                raise ValueError("all registered hypotheses must be retained in each comparison family")
            q = holm_fixed_family({str(i): all_mapping[str(i)] for i in group.index}, family_size=expected)
            for i in group.index:
                result.loc[i, "p_holm_family"] = q[str(i)]
                result.loc[i, "p_holm_all27"] = all_q[str(i)]
                if i in main.index:
                    result.loc[i, "p_holm_all17"] = q17[str(i)]
    result["significant_difference"] = result.primary_inference & result.p_holm_family.lt(.05)
    result["significant_improvement"] = result.significant_difference & result.estimate.gt(0)
    return result


def evaluate_ledgers(frame, cfg, condition_configs, validation_exposure, *, known_original=None):
    """Pure aggregation/inference from already-executed report-only paths.

    validation_exposure maps (fold,seed,arm,condition) to a Val-only frozen
    exposure target. It must never be computed from the Test path's exposure.
    Static0/0.5/1 controls use floor0 accounting, endowed1 at episode start,
    original rate cap, one-bar execution delay, turnover costs and borrowing.
    """
    folds, seeds = list(cfg["folds"]), list(cfg["seeds"])
    if list(cfg["arms"]) != list(ARMS) or list(cfg["conditions"]) != list(CONDITIONS):
        raise ValueError("registered A/C arms and all five release conditions required")
    expected = {(fold, seed, arm, condition) for fold in folds for seed in seeds
                for arm in ARMS for condition in CONDITIONS}
    if (not {"fold", "seed", "arm", "condition"}.issubset(frame)
            or set(frame[["fold", "seed", "arm", "condition"]].itertuples(index=False, name=None)) != expected
            or set(validation_exposure) != expected):
        raise ValueError("every registered fold/seed/arm/condition and frozen Val target is required")
    paths, matched, pooled, folded, matched_scores = {}, {}, [], [], []
    reference = None
    for arm in ARMS:
        for condition in CONDITIONS:
            config = condition_configs[condition]
            settings = accounting(config)
            for seed in seeds:
                path = _path(frame.loc[frame.arm.eq(arm) & frame.condition.eq(condition) & frame.seed.eq(seed)], config)
                if reference is not None and not path[list(COHORT)].equals(reference):
                    raise ValueError("all release conditions and seeds must retain the exact same market cohort")
                reference = path[list(COHORT)]
                paths[(arm, condition, seed)] = path
                pooled.append({"arm": arm, "condition": condition, "seed": seed, **_metrics(path, config)})
                for fold, part in path.groupby("fold", sort=True):
                    folded.append({"fold": fold, "arm": arm, "condition": condition, "seed": seed, **_metrics(part, config)})
                targets = np.array([validation_exposure[(int(fold), seed, arm, condition)] for fold in path.fold])
                if (not np.isfinite(targets).all() or np.any(targets < settings["position_min"] - 1e-7)
                        or np.any(targets > settings["position_max"] + 1e-7)):
                    raise ValueError("Val exposure targets must be finite and feasible under the final condition bounds")
                targets = np.clip(targets, settings["position_min"], settings["position_max"])
                request = bounded_requests(targets, path.timestamp, path.fold, config)
                control = execute_positions(path.bar_log_return, request, path.timestamp, path.fold, config)
                control["decision_id"] = path.decision_id.to_numpy()
                control["baseline"] = "val_exposure_matched"
                control["arm"], control["condition"], control["seed"] = arm, condition, seed
                matched[(arm, condition, seed)] = control
                matched_scores.append({"arm": arm, "condition": condition, "seed": seed,
                    "target_source": "selected policy Val mean exposure, fixed before Test", **_metrics(control, config)})
    expected_rows = cfg.get("statistics", {}).get("expected_market_rows")
    if expected_rows is not None and len(reference) != int(expected_rows):
        raise ValueError("registered common market row count changed")
    resets = reset_mask(reference.timestamp, reference.fold)
    day_key = pd.DataFrame({"fold": reference.fold, "segment": np.cumsum(resets) - 1,
        "day": reference.timestamp.dt.floor("D"), "count": 1})
    grouped = day_key.groupby(["fold", "segment", "day"], sort=True).sum().reset_index()
    definitions = _contrasts()
    daily_sums = np.empty((len(grouped), len(definitions), len(seeds)))
    for j, (_, _, left, right) in enumerate(definitions):
        for s, seed in enumerate(seeds):
            lhs = paths[(*left, seed)]
            rhs = matched[(*right[:2], seed)] if len(right) == 3 else paths[(*right, seed)]
            difference = lhs.net_log_return.to_numpy() - rhs.net_log_return.to_numpy()
            daily_sums[:, j, s] = day_key.assign(difference=difference).groupby(
                ["fold", "segment", "day"], sort=True).difference.sum().to_numpy()
    names = [item[0] for item in definitions]
    inference_cfg = {**cfg, "condition_names": names}
    stats = infer_daily(daily_sums, grouped["count"].to_numpy(), grouped.segment.to_numpy(), inference_cfg,
                        day_ids=grouped.day.astype("int64").to_numpy() // (86400 * 10**9))
    families = {name: family for name, family, _, _ in definitions}
    inference = stats["inference"].assign(family=lambda f: f.condition.map(families))
    inference = _correct(inference)
    hac = []
    for j, name in enumerate(names):
        for lag in cfg["statistics"].get("hac_lags", [6, 13, 29]):
            value = hac_mean(daily_sums[:, j], grouped["count"].to_numpy(), grouped.segment.to_numpy(),
                grouped.day.astype("int64").to_numpy() // (86400 * 10**9), lag=lag,
                annual_bars=accounting(condition_configs["control"])["annual_bars"])
            hac.append({"condition": name, "family": families[name], "lag_days": lag,
                "primary_inference": lag == 6, **{key: value[key] for key in
                ("estimate", "standard_error", "ci_low", "ci_high", "p_value", "inference_available")}})
    hac = _correct(pd.DataFrame(hac), hac=True)
    # Shared nonadaptive controls. Unit inventory is B&H; static0 is an
    # unattainable reference under the unchanged control/prior-branch floor.
    base = condition_configs["control"]
    static_cfg = {**base, "accounting": {**base.get("accounting", {}), "position_min": 0.}}
    static_paths, static_scores = [], []
    for target, name in ((0., "static0"), (.5, "static50"), (1., "bh")):
        request = bounded_requests(np.full(len(reference), target), reference.timestamp, reference.fold, static_cfg)
        path = execute_positions(reference.bar_log_return, request, reference.timestamp, reference.fold, static_cfg)
        path["decision_id"] = reference.decision_id.to_numpy()
        path["baseline"] = name
        static_paths.append(path)
        static_scores.append({"baseline": name, "target_position": target,
            "feasible_for_control_and_prior": target >= .5, **_metrics(path, static_cfg)})
    original_scores = []
    if known_original is not None:
        original = known_original.loc[known_original.arm.isin(ARMS)].copy()
        if "stage" in original:
            original = original.loc[original.stage.eq("rl")]
        for arm in ARMS:
            for seed in seeds:
                path = _path(original.loc[original.arm.eq(arm) & original.seed.eq(seed)], base)
                if not path[list(COHORT)].equals(reference):
                    raise ValueError("known original RL must share the identical scored cohort")
                original_scores.append({"arm": arm, "seed": seed, "baseline": "known_original_rl",
                                        **_metrics(path, base)})
    pooled = pd.DataFrame(pooled)
    matched_scores = pd.DataFrame(matched_scores)
    return {"fold_metrics": pd.DataFrame(folded), "pooled_metrics": pooled,
        "aggregate_metrics": _seed_summary(pooled, ["arm", "condition"]),
        "matched_metrics": matched_scores, "matched_aggregate": _seed_summary(matched_scores, ["arm", "condition"]),
        "static_metrics": pd.DataFrame(static_scores), "known_original_metrics": pd.DataFrame(original_scores),
        "inference": inference, "primary_inference": inference.loc[inference.primary_inference].copy(),
        "hac_inference": hac, "static_ledger": pd.concat(static_paths, ignore_index=True),
        "matched_ledger": pd.concat(matched.values(), ignore_index=True),
        "daily_panel": {"sums": daily_sums, "counts": grouped["count"].to_numpy(),
            "segment_ids": grouped.segment.to_numpy(), "day_ids": grouped.day.astype("int64").to_numpy() // (86400 * 10**9)},
        "audit": {"n_unique_market_rows": len(reference), "n_training_seeds": len(seeds),
            "n_folds": len(folds), "n_reset_segments": int(resets.sum()), "n_daily_units": len(grouped),
            "families": {"release_primary": 12, "wm_secondary": 5, "risk_matched_secondary": 10},
            "seed_market_independence": False, "independently_unused_holdout": False,
            "conditional_saved_cohort": True, "test_selection_allowed": False,
            "risk_matching": "Val mean exposure fixed separately per fold/seed/arm/condition; no Test fitting",
            "static0": "reference infeasible under control/prior floor; endowed1 ramp and all costs retained",
            "costs": "original executed ratio turnover cost and excess1x borrowing APR; no separately measured spread/slippage",
            "design": stats["design"]}}


def aggregate(cfg, registration_sha):
    """Verify the all-fit Test barrier, then save complete registered reports."""
    from . import wm_rl_release_benchmark as benchmark
    from .wm_rl_release_policy import condition_cfg
    from . import wm_policy_ablation_benchmark as original
    benchmark.verify_test_barrier(cfg)
    root = Path(cfg["output_dir"])
    if _sha(root / "preregistration.json") != registration_sha:
        raise ValueError("release registration SHA changed before aggregation")
    barrier_sha = _sha(root / "selection_barrier.json")
    source_cfg = original.load_config(cfg["source_config"])
    inputs, frames, targets, target_sources, choices = {}, [], {}, {}, []
    for fold in cfg["folds"]:
        for seed in cfg["seeds"]:
            for arm in ARMS:
                for condition in CONDITIONS:
                    job = _job(fold, arm, seed, condition)
                    directory = root / "evaluations" / job
                    marker = json.loads((directory / "evaluation.json").read_text())
                    ledger_path = directory / "executed_ledger.parquet"
                    ledger_sha = _sha(ledger_path)
                    if (not marker.get("complete") or marker.get("registration_sha256") != registration_sha
                            or marker.get("job") != job
                            or marker.get("barrier_sha256") != barrier_sha
                            or marker.get("artifact_sha256", {}).get("executed_ledger.parquet") != ledger_sha):
                        raise ValueError("every registered Test ledger must have a complete immutable evaluation marker")
                    inputs[str(ledger_path)] = ledger_sha
                    frame = pd.read_parquet(ledger_path)
                    for key, value in (("fold", fold), ("seed", seed), ("arm", arm), ("condition", condition)):
                        if key in frame and not frame[key].eq(value).all():
                            raise ValueError("Test ledger metadata differs from its registered job")
                        frame[key] = value
                    frames.append(frame)
                    fit_path = root / "neural" / job / "fit.json"
                    inputs[str(fit_path)] = _sha(fit_path)
                    fit = json.loads(fit_path.read_text())
                    field = "full_validation_metrics" if "full_validation_metrics" in fit else "validation_metrics"
                    if field not in fit or "mean_exposure" not in fit[field]:
                        raise ValueError("selected Val mean exposure is required for the frozen risk-matched control")
                    targets[(fold, seed, arm, condition)] = float(fit[field]["mean_exposure"])
                    target_sources[job] = field
                    choices.append({"job": job, "fold": fold, "seed": seed, "arm": arm, "condition": condition,
                        "selected_update": fit.get("selected_update"), "updates": fit.get("updates"),
                        "val_exposure_target": targets[(fold, seed, arm, condition)], "target_source": field,
                        "test_used_for_selection": False})
    known_path = Path(source_cfg["output_dir"]) / "final/executed_policy_ledger.parquet"
    inputs[str(known_path)] = _sha(known_path)
    known = pd.read_parquet(known_path)
    configs = {condition: condition_cfg(cfg, condition) for condition in CONDITIONS}
    result = evaluate_ledgers(pd.concat(frames, ignore_index=True), cfg, configs, targets, known_original=known)
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(choices).to_csv(final / "selection_metrics.csv", index=False)
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            if name.endswith("ledger"):
                value.to_parquet(final / (name + ".parquet"), index=False)
            else:
                value.to_csv(final / (name + ".csv"), index=False)
    np.savez(final / "daily_panel.npz", **result["daily_panel"])
    result["audit"].update(registration_sha256=registration_sha, input_sha256=inputs,
                            validation_target_metric_sources=target_sources)
    if any(_sha(path) != expected for path, expected in inputs.items()):
        raise ValueError("frozen release inputs changed while reporting")
    benchmark.verify_test_barrier(cfg)
    (final / "audit.json").write_text(json.dumps(result["audit"], indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write_report(final, result)
    artifacts = {path.name: _sha(path) for path in final.iterdir() if path.is_file()}
    return {"complete": True, "registration_sha256": registration_sha, "n_release_fits": len(frames),
        "n_market_rows": result["audit"]["n_unique_market_rows"], "test_report_only": True,
        "independently_unused_holdout": False, "artifact_sha256": artifacts, "report_path": str(final / "index.html")}


def _write_report(final, result):
    import html
    primary = result["primary_inference"]
    main = primary.loc[primary.family.eq("release_primary")]
    positive = int(main.significant_improvement.sum())
    columns = ["condition", "family", "estimate", "ci_low", "ci_high", "p_value", "p_holm_family", "p_holm_all17", "p_holm_all27"]
    tables = []
    for title, frame in (("主比較12：制限緩和と継続control", main),
            ("WMの追加効果5比較：C−A", primary.loc[primary.family.eq("wm_secondary")]),
            ("Valで保有率を固定した対照との10比較", primary.loc[primary.family.eq("risk_matched_secondary")])):
        tables.append("<h2>" + html.escape(title) + "</h2>" + frame[columns].to_html(index=False, escape=True, float_format=lambda x: f"{x:.5g}"))
    metrics = result["aggregate_metrics"]
    selected = ["arm", "condition", "net_total_return_seed_mean", "annualized_net_log_growth_seed_mean",
        "max_drawdown_seed_mean", "mean_exposure_seed_mean", "near_cash_fraction_seed_mean",
        "floor_fraction_seed_mean", "turnover_seed_mean"]
    body = ["<!doctype html><html lang='ja'><meta charset='utf-8'><title>RL制限緩和の比較</title>",
        "<style>body{font:15px system-ui;line-height:1.7;margin:30px auto;max-width:1600px;padding:20px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}h2{margin-top:32px}.scroll{overflow:auto}</style>",
        "<h1>既存RLからの追加学習：制限緩和とWMの比較</h1>",
        f"<p>7日循環Block Bootstrap・固定seed平均・主12比較のHolm補正後、有意な改善は{positive}/12です。",
        "主比較12、WM比較5、Val保有率対応対照10は別々の族として補正し、全17・全27比較の感度確認も保存しています。</p>",
        "<p>このTest期間は既に研究で使った既知の探索区間です。未使用Holdoutでの確認ではありません。3seedは同じ市場系列を見ており、独立した市場を3倍に増やしません。",
        "欠損・fold境界で保有率1から再開する共通cohortの比較で、全カレンダーの実運用成績ではありません。手数料・借入費用・1バー遅延を含みます。</p>",
        "<p>効果と95%区間は年率の自然対数成長率差です。累積収益率の差ではありません。95%区間は点ごとのBasic区間で、同時信頼区間ではありません。14/30日・HACは感度分析です。</p>",
        "<h2>10条件の実際の保有率・収益・リスク</h2>", metrics[selected].to_html(index=False, escape=True, float_format=lambda x: f"{x:.5g}"),
        "<h2>固定保有対照</h2><p>static0も初期保有1から速度制限を守って現金へ移り、売却費用を払います。control/prior枝ではfloor制約により実行不能な参照対照です。",
        "保有率対応対照は各fold・seed・条件で選択済みValの平均保有率を固定し、Testで調整しません。平均保有率だけを合わせても時変リスクが完全に一致するわけではありません。</p>",
        result["static_metrics"].to_html(index=False, escape=True, float_format=lambda x: f"{x:.5g}"), *tables,
        "<p><a href='aggregate_metrics.csv'>全指標</a> / <a href='inference.csv'>全Bootstrap</a> / <a href='hac_inference.csv'>HAC</a> / <a href='known_original_metrics.csv'>追加学習前RL</a> / <a href='audit.json'>監査</a></p></html>"]
    (final / "index.html").write_text("\n".join(body), encoding="utf-8")
