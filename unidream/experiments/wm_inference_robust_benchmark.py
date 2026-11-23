"""Registered post-hoc audit of financial mean-return inference methods.

Only new result namespaces are written. Original models, costs, selected
checkpoints, all observed return ledgers and earlier inference remain frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .wm_significance_statistics import holm_fixed_family

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_inference_robust_20261112.yaml"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    for key in ("output_dir", "source_config", "previous_inference_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if (cfg.get("post_hoc") is not True or cfg.get("test_selection_allowed") is not False
            or cfg.get("independently_unused_holdout") is not False
            or cfg["arms"] != list("ABCD") or cfg["stages"] != ["bc", "rl"]
            or cfg["folds"] != list(range(6)) or cfg["seeds"] != [7, 19, 42]
            or cfg["benchmarks"] != ["bh", "fixed50"]
            or cfg["statistics"]["methods"] != ["legacy_non_circular", "circular", "stationary"]
            or cfg["statistics"]["block_days"] != [1, 7, 14, 28, 30]
            or cfg["statistics"]["bootstrap_replicates"] != 10000
            or cfg["statistics"]["primary_method"] != "circular"
            or cfg["statistics"]["primary_estimand"] != "market_fixed_seeds"
            or cfg["statistics"]["primary_block_days"] != 7):
        raise ValueError("complete fixed post-hoc robustness grid and unchanged Test selection required")
    cfg["condition_names"] = [f"{stage}_{arm}" for stage in cfg["stages"] for arm in cfg["arms"]]
    return cfg


def source_hashes(cfg):
    files = sorted(Path(__file__).parent.glob("wm_inference_robust_*.py"))
    files += sorted((ROOT / "tests").glob("test_wm_inference_robust_*.py"))
    files += [Path(__file__).with_name(name) for name in (
        "wm_policy_bh_benchmark.py", "wm_policy_bh_inference.py", "wm_policy_ablation_evaluation.py",
        "wm_full_grid_statistics.py", "wm_significance_statistics.py")]
    files.append(Path(cfg["config_path"]))
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def sources(cfg):
    from . import wm_policy_bh_benchmark as previous
    pcfg = previous.load_config(cfg["previous_inference_config"])
    # Existing registration is verified, never replaced or amended.
    if not (Path(pcfg["output_dir"]) / "posthoc_registration.json").exists():
        raise ValueError("completed prior B&H analysis required")
    prior_registration = previous.register(pcfg)
    prior_root = Path(pcfg["output_dir"])
    done = json.loads((prior_root / "completion.json").read_text())
    if not done.get("complete") or done["registration_sha256"] != sha(prior_root / "posthoc_registration.json"):
        raise ValueError("prior BH analysis completion changed")
    for name, expected in done["artifact_sha256"].items():
        if sha(prior_root / name) != expected:
            raise ValueError("prior BH analysis result changed")
    result = dict(prior_registration["sources"])
    result["prior_bh_output_dir"] = str(prior_root)
    result["prior_bh_sha256"] = {name: sha(prior_root / name) for name in (
        "posthoc_registration.json", "completion.json", "inference.csv", "summary.csv")}
    return result


def register(cfg):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "posthoc_registration.json"
    bound_sources = sources(cfg)
    version = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "arch", "matplotlib")}
    if path.exists():
        registration = json.loads(path.read_text())
        if (registration["config"] != cfg or registration["source_sha256"] != source_hashes(cfg)
                or registration["sources"] != bound_sources or registration["package_versions"] != version):
            raise ValueError("registered robustness methods, versions, or frozen data changed")
        return registration
    registration = {"registered_at": datetime.now(timezone.utc).isoformat(), "config": cfg,
        "source_sha256": source_hashes(cfg), "sources": bound_sources, "package_versions": version,
        "post_hoc": True, "original_results_previously_seen": True, "test_selection_changed": False,
        "primary": "BH8, circular7day, fixed-observed-seed mean, actual H0 residuals, basic CI, two-sided p, Holm8",
        "estimand_change": "old primary included random training-seed uncertainty; report both market-fixed-seeds and hierarchical side-by-side",
        "secondary": "fixed50 controls8; joint16 Holm; HAC6; method/block/seed sensitivities; White maxmean8; re-evaluate original12 and combined20 using new circular primary",
        "unsupported_segments": "length<=block retained fixed; conditional CI and fixed fraction explicitly reported",
        "fees": "reuse exact original net ledger; cost0.00055*executed ratio turnover; APR10% assumption on excess1x; no separate measured spread/slippage",
        "account_start": "all paths endowed with BTC1x at every gap/fold boundary; firstentry and terminalexit not charged",
        "fixed50": "position-ratio target0.5, original step0.08/onebar lag/fees; no Test-dependent target fit; fractional exposure approximation",
        "calibration": "IID/AR1/GARCH/AR1-GARCH true zero mean; no sample demeaning; 1 vs6 synthetic segments; Wilson MonteCarlo intervals",
        "power": "prospective effect scenarios and MDE; observed selected effect only sensitivity, not posthoc power evidence; constant LRV assumption, eligible years",
        "multiple_search_limit": "RC8/Holm8/16/20 cannot correct historical420 cells,57head F selection and unrecorded design search",
        "DM_scope": "actual portfolio PnL differences, not forecast loss differences; use mean-HAC, do not relabel it DM",
        "independently_unused_holdout": False}
    write_json(path, registration)
    return registration


def _cost_audit(policies, cfg):
    a = cfg["accounting"]
    names = ["turnover", "executed_position", "bar_log_return", "cost_fraction", "borrow_fraction", "net_log_return"]
    if not np.isfinite(policies[names].to_numpy(float)).all():
        raise ValueError("nonfinite original net-return accounting inputs")
    cost = policies.turnover.to_numpy(float) * a["cost_rate"]
    borrow = np.maximum(policies.executed_position.to_numpy(float) - 1, 0) * a["borrow_apr"] / a["annual_bars"]
    expected = np.log1p(policies.executed_position.to_numpy(float) * np.expm1(policies.bar_log_return.to_numpy(float)) - borrow) + np.log1p(-cost)
    errors = {"cost_max_error": float(np.max(np.abs(cost - policies.cost_fraction))),
        "borrow_max_error": float(np.max(np.abs(borrow - policies.borrow_fraction))),
        "net_log_return_max_error": float(np.max(np.abs(expected - policies.net_log_return)))}
    if max(errors.values()) > 1e-12:
        raise ValueError("original recorded returns do not match the declared fee/borrow accounting")
    return {**errors, "cost_rate": a["cost_rate"], "borrow_apr_assumption": a["borrow_apr"],
        "separately_measured_spread_slippage": False, "initial_endowed_position": 1.,
        "initial_entry_fee": False, "terminal_liquidation_fee": False, "accounting": "fixed fractional exposure approximation"}


def build_panel(cfg, registration):
    from . import wm_policy_ablation_benchmark as parent
    from .wm_policy_ablation_evaluation import bounded_requests, execute_positions, reset_mask, score_path
    from .wm_policy_bh_benchmark import _validate_bh
    original_cfg = parent.load_config(cfg["source_config"])
    source = Path(registration["sources"]["source_root"]) / "final"
    policies = pd.read_parquet(source / "executed_policy_ledger.parquet")
    expected_grid = {(fold, seed, arm, stage) for fold in cfg["folds"] for seed in cfg["seeds"]
        for arm in cfg["arms"] for stage in cfg["stages"]}
    if set(policies[["fold", "seed", "arm", "stage"]].itertuples(index=False, name=None)) != expected_grid:
        raise ValueError("exactly the original registered fold/seed/arm/stage grid required")
    bh = _validate_bh(pd.read_parquet(source / "executed_baseline_ledger.parquet"))
    if len(bh) != cfg["statistics"]["expected_market_rows"]:
        raise ValueError("same registered market cohort required")
    half_request = bounded_requests(np.full(len(bh), .5), bh.timestamp, bh.fold, original_cfg)
    half = execute_positions(bh.bar_log_return, half_request, bh.timestamp, bh.fold, original_cfg)
    half["decision_id"] = bh.decision_id.to_numpy()
    half["baseline"] = "fixed50"
    controls = {"bh": bh, "fixed50": half}
    cohort = ["fold", "decision_id", "timestamp", "bar_log_return"]
    reset = reset_mask(bh.timestamp, bh.fold)
    daily_key = pd.DataFrame({"fold": bh.fold, "segment": np.cumsum(reset) - 1,
        "day": bh.timestamp.dt.floor("D"), "count": 1})
    grouped = daily_key.groupby(["fold", "segment", "day"], sort=True).sum().reset_index()
    sums = {key: np.empty((len(grouped), 8, 3), float) for key in controls}
    descriptions, per_folds = [], []
    for j, condition in enumerate(cfg["condition_names"]):
        stage, arm = condition.split("_")
        for s, seed in enumerate(cfg["seeds"]):
            part = policies.loc[policies.arm.eq(arm) & policies.stage.eq(stage) & policies.seed.eq(seed)].sort_values("timestamp").reset_index(drop=True)
            if not part[cohort].equals(bh[cohort]):
                raise ValueError("every model, seed, and control must match exactly the same market cohort")
            for name, control in controls.items():
                diff = part.net_log_return.to_numpy() - control.net_log_return.to_numpy()
                daily = daily_key.assign(difference=diff).groupby(["fold", "segment", "day"], sort=True).sum().reset_index()
                sums[name][:, j, s] = daily.difference.to_numpy()
                descriptions.append({"benchmark": name, "condition": condition, "arm": arm, "stage": stage, "seed": seed,
                    "policy_net_total_return": float(np.expm1(part.net_log_return.sum())),
                    "benchmark_net_total_return": float(np.expm1(control.net_log_return.sum())),
                    "cumulative_return_difference_descriptive": float(np.expm1(part.net_log_return.sum()) - np.expm1(control.net_log_return.sum())),
                    "annualized_log_growth_difference": float(diff.mean() * cfg["accounting"]["annual_bars"]),
                    "mean_exposure": float(part.executed_position.mean())})
                for fold in cfg["folds"]:
                    mask = part.fold.eq(fold).to_numpy()
                    per_folds.append({"benchmark": name, "condition": condition, "fold": fold, "seed": seed,
                        "n_rows": int(mask.sum()), "estimate": float(diff[mask].mean() * cfg["accounting"]["annual_bars"])})
    counts = grouped["count"].to_numpy(float)
    days = grouped.day.astype("int64").to_numpy() // (86400 * 10**9)
    panel = {"counts": counts, "segments": grouped.segment.to_numpy(np.int64), "days": days,
        "folds": grouped.fold.to_numpy(np.int64), **{f"sum_{k}": v for k, v in sums.items()}}
    audit = {"panel": {"n_unique_market_rows": len(bh), "n_daily_units": len(grouped),
        "n_segments": int(reset.sum()), "n_training_seeds": 3, "eligible_years": len(bh) / 35040,
        "first_timestamp": str(bh.timestamp.min()), "last_timestamp": str(bh.timestamp.max()),
        "common_daily_draws_all_seeds": True}, "net_cost_audit": _cost_audit(policies, original_cfg),
        "baseline_metrics": {name: score_path(control, original_cfg) for name, control in controls.items()},
        "selection_scope": {"forecast_cells": 420, "selected_heads": 57, "selected_forecast_columns": 121,
            "unused_holdout": False, "old_test_already_inspected": True,
            "historical_search_corrected_by_RC8": False}}
    return panel, pd.DataFrame(descriptions), pd.DataFrame(per_folds), controls, policies, audit


def ablation_panel(bh_day_sums, condition_names):
    """Re-evaluate the same original12 contrasts; the common BH cancels."""
    positions = {name: i for i, name in enumerate(condition_names)}
    pairs = [(f"{stage}_{left}", f"{stage}_{right}") for stage in ("bc", "rl")
        for left, right in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C"))]
    pairs += [(f"rl_{arm}", f"bc_{arm}") for arm in "ABCD"]
    return np.stack([bh_day_sums[:, positions[left]] - bh_day_sums[:, positions[right]] for left, right in pairs], axis=1), [f"{left}_minus_{right}" for left, right in pairs]


def _holm_column(frame, groups, output):
    frame[output] = np.nan
    for _, part in frame.groupby(groups, sort=False):
        mapping = {str(i): float(row.p_value) if pd.notna(row.p_value) else 1. for i, row in part.iterrows()}
        corrected = holm_fixed_family(mapping, family_size=len(part))
        for i in part.index:
            frame.loc[i, output] = corrected[str(i)]


def _original_replay(policies, bh, cfg, registration):
    from .wm_policy_bh_benchmark import load_config as previous_config
    from .wm_policy_bh_inference import infer_bh
    prior_cfg = previous_config(cfg["previous_inference_config"])
    old = pd.read_csv(Path(registration["sources"]["prior_bh_output_dir"]) / "inference.csv", float_precision="round_trip")
    new = infer_bh(policies, bh, prior_cfg)
    joined = old.merge(new, on=["comparison", "block_days"], suffixes=("_original", "_replayed"), validate="one_to_one")
    errors = {}
    for col in ("estimate", "ci_low", "ci_high", "p_value", "p_holm_8"):
        left, right = joined[col + "_original"].to_numpy(float), joined[col + "_replayed"].to_numpy(float)
        errors[col] = float(np.nanmax(np.abs(left - right)))
    if max(errors.values()) > 1e-12:
        raise ValueError("the original BH test cannot be reproduced from its frozen inputs")
    return joined, {"max_absolute_difference": max(errors.values()), "by_statistic": errors,
        "explicit_centering_present": True, "formula": "abs(theta_star-theta_hat)>=abs(theta_hat)",
        "caveat": "this implements H0 centering but does not remove finite noncircular endpoint weighting bias"}


def _automatic_blocks(panel, cfg):
    from arch.bootstrap import optimal_block_length
    rows = []
    for benchmark in cfg["benchmarks"]:
        means = panel["sum_" + benchmark].mean(axis=2) / panel["counts"][:, None]
        for segment in np.unique(panel["segments"]):
            mask = panel["segments"] == segment
            if mask.sum() < 10:
                continue
            result = optimal_block_length(pd.DataFrame(means[mask], columns=cfg["condition_names"]))
            for condition, row in result.iterrows():
                rows.append({"benchmark": benchmark, "condition": condition, "segment": int(segment),
                    "n_days": int(mask.sum()), "stationary_days": float(row["stationary"]), "circular_days": float(row["circular"]),
                    "used_for_primary_selection": False, "scope": "diagnostic per contiguous observed segment; short samples may be noisy"})
    return pd.DataFrame(rows)


def calculate(cfg, registration):
    from .wm_inference_robust_bootstrap import infer_daily
    from .wm_inference_robust_hac import hac_mean, power_scenarios, serial_diagnostics
    from .wm_inference_robust_calibration import run_calibration, legacy_inclusion_audit
    out = Path(cfg["output_dir"]) / "final"
    out.mkdir(exist_ok=True)
    panel, descriptions, folds, controls, policies, audit = build_panel(cfg, registration)
    np.savez(Path(cfg["output_dir"]) / "prepared_panel.npz", **panel)
    descriptions.to_csv(out / "per_seed.csv", index=False)
    folds.to_csv(out / "per_fold.csv", index=False)
    pd.concat([part.assign(baseline=name) for name, part in controls.items()], ignore_index=True).to_parquet(out / "controls.parquet", index=False)
    replay, replay_audit = _original_replay(policies, controls["bh"], cfg, registration)
    replay.to_csv(out / "legacy_replay.csv", index=False)
    audit["legacy_p_replay"] = replay_audit
    audit["legacy_inclusion_example"] = legacy_inclusion_audit(cfg)
    all_boot, all_rc, hac_rows, acf_rows, fold_diag, power_rows, mde_rows, year_rows = [], [], [], [], [], [], [], []
    for benchmark in cfg["benchmarks"]:
        y = panel["sum_" + benchmark]
        result = infer_daily(y, panel["counts"], panel["segments"], cfg, day_ids=panel["days"])
        all_boot.append(result["inference"].assign(benchmark=benchmark))
        all_rc.append(result["reality_check"].assign(benchmark=benchmark))
        for j, condition in enumerate(cfg["condition_names"]):
            for lag in cfg["statistics"]["hac_lags"]:
                h = hac_mean(y[:, j], panel["counts"], panel["segments"], panel["days"], lag=lag,
                    annual_bars=cfg["accounting"]["annual_bars"])
                for estimand, row in (("market_fixed_seeds", h), ("seed_augmented_approx", h["approximate_seed_augmented"])):
                    hac_rows.append({"benchmark": benchmark, "condition": condition, "lag_days": lag, "estimand": estimand,
                        **{k: row[k] for k in ("estimate", "standard_error", "ci_low", "ci_high", "p_value", "inference_available")},
                        "primary_inference": lag == 6 and estimand == "market_fixed_seeds",
                        "approximate_seed_variance_floor": h["approximate_seed_variance_floor"],
                        "eligible_years": h["eligible_years"], "n_market_bars": h["n_market_bars"], "n_seeds": 3})
                if lag == cfg["statistics"]["primary_hac_lag"]:
                    settings = cfg["power"]
                    years = sorted(set(settings["eligible_years"] + [h["eligible_years"]]))
                    power = power_scenarios(h, effects=settings["annual_log_effects"], years=years,
                        alphas=settings["alpha"], powers=settings["target_power"], include_observed=True)
                    for name, dest in (("power", power_rows), ("mde", mde_rows), ("required_years", year_rows)):
                        dest.extend({"benchmark": benchmark, "condition": condition, **row} for row in power[name])
            diagnostic = serial_diagnostics(y[:, j], panel["counts"], panel["segments"], panel["days"], fold_ids=panel["folds"])
            acf_rows.extend({"benchmark": benchmark, "condition": condition, **row} for row in diagnostic["acf"])
            fold_diag.extend({"benchmark": benchmark, "condition": condition, **row} for row in diagnostic["folds"])
    bootstrap, rc, hac = pd.concat(all_boot, ignore_index=True), pd.concat(all_rc, ignore_index=True), pd.DataFrame(hac_rows)
    ablation_y, ablation_names = ablation_panel(panel["sum_bh"], cfg["condition_names"])
    ablation_cfg = {**cfg, "condition_names": ablation_names}
    ablation_result = infer_daily(ablation_y, panel["counts"], panel["segments"], ablation_cfg, day_ids=panel["days"])
    ablation = ablation_result["inference"]
    ablation_primary = ablation.loc[ablation.primary_inference].copy()
    ablation_hac_rows = []
    for j, condition in enumerate(ablation_names):
        for lag in cfg["statistics"]["hac_lags"]:
            h = hac_mean(ablation_y[:, j], panel["counts"], panel["segments"], panel["days"], lag=lag,
                annual_bars=cfg["accounting"]["annual_bars"])
            ablation_hac_rows.append({"condition": condition, "lag_days": lag,
                **{k: h[k] for k in ("estimate", "standard_error", "ci_low", "ci_high", "p_value", "inference_available")}})
    _holm_column(bootstrap, ["benchmark", "method", "block_days", "estimand"], "p_holm_within_sensitivity")
    _holm_column(ablation, ["method", "block_days", "estimand"], "p_holm_within_sensitivity")
    all_p = {"control:" + str(i): float(row.p_value) if pd.notna(row.p_value) else 1. for i, row in bootstrap.iterrows()}
    all_p.update({"ablation:" + str(i): float(row.p_value) if pd.notna(row.p_value) else 1. for i, row in ablation.iterrows()})
    all_family = holm_fixed_family(all_p, family_size=len(all_p))
    bootstrap["p_holm_all_bootstrap_choices"] = [all_family["control:" + str(i)] for i in bootstrap.index]
    ablation["p_holm_all_bootstrap_choices"] = [all_family["ablation:" + str(i)] for i in ablation.index]
    _holm_column(hac, ["benchmark", "lag_days", "estimand"], "p_holm_family")
    aggregate = descriptions.groupby(["benchmark", "condition"], as_index=False).agg(
        policy_net_total_return=("policy_net_total_return", "mean"), benchmark_net_total_return=("benchmark_net_total_return", "first"),
        cumulative_return_difference_descriptive=("cumulative_return_difference_descriptive", "mean"), mean_exposure=("mean_exposure", "mean"))
    summary = bootstrap.loc[bootstrap.primary_inference].merge(aggregate, on=["benchmark", "condition"], validate="one_to_one")
    joined = holm_fixed_family({str(i): float(row.p_value) if pd.notna(row.p_value) else 1.
        for i, row in summary.iterrows()}, family_size=16)
    summary["p_holm_16_primary_controls"] = [joined[str(i)] for i in summary.index]
    old = pd.read_csv(Path(registration["sources"]["prior_bh_output_dir"]) / "inference.csv", float_precision="round_trip")
    old = old.loc[old.primary_inference].assign(condition=lambda f: f.stage + "_" + f.arm)
    for metric in ("p_value", "ci_low", "ci_high"):
        lookup = old.set_index("condition")[metric]
        summary["original_legacy_" + metric] = [lookup.get(row.condition) if row.benchmark == "bh" else np.nan for row in summary.itertuples()]
    new8 = summary.loc[summary.benchmark.eq("bh")]
    if len(ablation_primary) != 12 or len(new8) != 8:
        raise ValueError("expanded20 must retain the twelve original hypotheses and all eight BH comparisons")
    family20 = {"ablation:" + row.condition: float(row.p_value) if pd.notna(row.p_value) else 1. for row in ablation_primary.itertuples()}
    family20.update({"newBH:" + row.condition: float(row.p_value) if pd.notna(row.p_value) else 1. for row in new8.itertuples()})
    corrected20 = holm_fixed_family(family20, family_size=20)
    summary["p_holm20_prior12_sensitivity"] = [corrected20["newBH:" + row.condition] if row.benchmark == "bh" else np.nan for row in summary.itertuples()]
    ablation_primary["p_holm20_sensitivity"] = [corrected20["ablation:" + row.condition] for row in ablation_primary.itertuples()]
    pd.DataFrame([{"hypothesis": key, "p_value": value, "p_holm20": corrected20[key]} for key, value in family20.items()]).to_csv(out / "expanded_family20.csv", index=False)
    for name, frame in (("bootstrap_inference", bootstrap), ("reality_check", rc), ("hac_inference", hac), ("summary", summary),
            ("ablation_robustness", ablation), ("ablation_primary", ablation_primary), ("ablation_hac", pd.DataFrame(ablation_hac_rows)),
            ("power", pd.DataFrame(power_rows)), ("mde", pd.DataFrame(mde_rows)), ("required_years", pd.DataFrame(year_rows)),
            ("acf", pd.DataFrame(acf_rows)), ("fold_diagnostics", pd.DataFrame(fold_diag)), ("automatic_blocks", _automatic_blocks(panel, cfg))):
        frame.to_csv(out / f"{name}.csv", index=False)
    print(json.dumps({"phase": "observed_results_complete", "bootstrap_rows": len(bootstrap), "hac_rows": len(hac)}), flush=True)
    calibration = run_calibration(cfg)
    calibration.to_csv(out / "calibration.csv", index=False)
    audit["new_inference_design"] = result["design"]
    audit["calibration_scope"] = "known synthetic null only; limited300 Monte Carlo precision; no guarantee for real nonstationary market or seed population"
    audit["method_choices_frozen_before_new_results"] = True
    write_json(out / "audit.json", audit)
    return out


def run(cfg):
    registration = register(cfg)
    root = Path(cfg["output_dir"])
    path = root / "completion.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["registration_sha256"] != sha(root / "posthoc_registration.json"):
            raise ValueError("robustness completion belongs to a different analysis registration")
        for filename, digest in old["artifact_sha256"].items():
            if sha(root / filename) != digest:
                raise ValueError("completed robustness result changed")
        return old
    out = calculate(cfg, registration)
    from .wm_inference_robust_report import write_report
    report = write_report(root, cfg)
    if registration["sources"] != sources(cfg) or registration["source_sha256"] != source_hashes(cfg):
        raise ValueError("frozen sources changed during robustness calculations")
    artifacts = {str(p.relative_to(root)): sha(p) for p in out.iterdir() if p.is_file()}
    artifacts["prepared_panel.npz"] = sha(root / "prepared_panel.npz")
    completed = {"complete": True, "at": datetime.now(timezone.utc).isoformat(), "post_hoc": True,
        "registration_sha256": sha(root / "posthoc_registration.json"), "artifact_sha256": artifacts,
        "original_models_costs_or_Test_selection_changed": False, "original_artifacts_unchanged": True,
        "report": report, "independently_unused_holdout": False}
    write_json(path, completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
