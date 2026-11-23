"""Paired UTC-day block inference for the joint heads and their controls.

The fixed confirmatory family is112 neural contrasts:56 native regression
contrasts and56 probability-head BA contrasts. The112 classical contrasts
are exact-zero algebraic controls, not evidence for transfer learning.
For advantage/recovery the reported primary Spearman IC has no approximate
CI: secondary RMSE receives exact paired block resampling instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_ablation_evaluation import holm_adjust
from .wm_full_grid_statistics import _block_sums, paired_day_count_inference
from .wm_joint_grid_report import TASKS, MODELS, NATIVE_PRIMARY, _sha, _json


KEYS = ("fold", "seed", "utc_day", "segment_id")
SUMS = ("sum_squared_error", "sum_absolute_error", "sum_qlike", "sum_actual_squared", "sum_brier", "n_rows")
INFERENCE_METRIC = {**NATIVE_PRIMARY, "advantage": "rmse", "recovery": "rmse"}


def _metric(values, metric):
    sse, sae, qlike, actual_sse, brier, count = np.moveaxis(np.asarray(values, float), -1, 0)
    if metric == "zero_oos_r2":
        return np.where(actual_sse > 0, 1 - np.divide(sse, actual_sse, out=np.zeros_like(sse), where=actual_sse > 0), np.nan)
    if metric == "rmse":
        return np.sqrt(sse / count)
    return {"mae": sae, "qlike": qlike, "brier": brier}[metric] / count


def _prepare_native(frame):
    required = {"split", "group", "model", "task", "formulation", *KEYS, *SUMS}
    if not required.issubset(frame.columns):
        raise ValueError("native daily table lacks registered sufficient statistics")
    selected = frame.loc[frame.split == "test"].copy()
    if selected.empty or any(selected[key].nunique() != 1 for key in ("group", "model", "task")):
        raise ValueError("one nonempty group/model/target Test panel required")
    if set(selected.formulation) != {"regression", "joint"}:
        raise ValueError("native comparison requires exactly regression and joint")
    if selected.seed.nunique() != 1 or selected.duplicated(["formulation", *KEYS]).any():
        raise ValueError("native comparison needs a single matched seed and unique daily rows")
    for key in SUMS:
        values = pd.to_numeric(selected[key], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (key != "sum_qlike" and np.any(values < 0)):
            raise ValueError("native sufficient statistics must be finite and nonnegative")
        selected[key] = values
    if np.any(selected.n_rows <= 0) or np.any(selected.n_rows != np.floor(selected.n_rows)):
        raise ValueError("native daily supports must be positive integers")
    for key in ("fold", "seed", "segment_id"):
        values = pd.to_numeric(selected[key], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values < 0):
            raise ValueError("native identifiers must be nonnegative integers")
    dates = pd.to_datetime(selected.utc_day, utc=True, errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.floor("D")).all():
        raise ValueError("native utc_day must be UTC midnight")
    selected["utc_day"] = dates
    if (selected.groupby(["formulation", "fold", "seed", "utc_day"]).n_rows.sum() > 96).any():
        raise ValueError("native daily support exceeds15-minute grid")
    baseline = selected.loc[selected.formulation == "regression"]
    joint = selected.loc[selected.formulation == "joint"]
    paired = baseline.merge(joint, on=list(KEYS), suffixes=("_baseline", "_joint"),
        how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("native joint and control daily coverage differ")
    for key in ("n_rows", "sum_actual_squared"):
        if not np.array_equal(paired[key + "_baseline"], paired[key + "_joint"]):
            raise ValueError("native joint and control outcomes/support differ")
    paired = paired.sort_values(["fold", "segment_id", "utc_day"]).reset_index(drop=True)
    for _, part in paired.groupby(["fold", "segment_id"], sort=True):
        if len(part) > 1 and not part.utc_day.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError("native contiguous segment bridges missing UTC days")
    return paired


def paired_native_day_inference(frame, *, block_days=(1, 7, 28), replicates=10000, random_seed=20261111):
    if int(replicates) != replicates or replicates < 2:
        raise ValueError("at least two whole bootstrap replicates required")
    if not block_days or len(set(block_days)) != len(block_days) or any(int(x) != x or x < 1 for x in block_days):
        raise ValueError("distinct positive whole-day block lengths required")
    if int(random_seed) != random_seed or random_seed < 0:
        raise ValueError("nonnegative whole random seed required")
    paired = _prepare_native(frame)
    task = frame.task.iloc[0]
    if task not in TASKS:
        raise ValueError("unknown native task")
    metric = INFERENCE_METRIC[task]
    columns = [key + suffix for suffix in ("_baseline", "_joint") for key in SUMS]
    total = paired[columns].sum().to_numpy(float)
    baseline_value = float(_metric(total[:6], metric))
    joint_value = float(_metric(total[6:], metric))
    estimate = joint_value - baseline_value
    segments = [part[columns].to_numpy(float) for _, part in paired.groupby(["fold", "segment_id"], sort=True)]
    rows = []
    for days in map(int, block_days):
        rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), days]))
        samples = np.zeros((int(replicates), len(columns)))
        varying = shorter = equal = 0
        supported_rows = unsupported_rows = 0
        for segment in segments:
            shorter += int(len(segment) < days); equal += int(len(segment) == days)
            if len(segment) > days:
                varying += 1
                supported_rows += int(segment[:, 5].sum())
                samples += _block_sums(segment, days, int(replicates), rng)
            else:
                unsupported_rows += int(segment[:, 5].sum())
                samples += segment.sum(axis=0)
        distribution = _metric(samples[:, 6:], metric) - _metric(samples[:, :6], metric)
        finite = distribution[np.isfinite(distribution)]
        available = bool(varying and np.isfinite(estimate) and len(finite) >= 2)
        low = high = p = None
        if available:
            low, high = map(float, np.quantile(finite, [.025, .975]))
            p = float((1 + np.sum(np.abs(finite - estimate) >= abs(estimate))) / (len(finite) + 1))
        rows.append({"split": "test", "group": frame.group.iloc[0], "model": frame.model.iloc[0], "task": task,
            "comparison": "joint_minus_regression_native", "head": "regression", "inference_metric": metric,
            "report_primary_metric": NATIVE_PRIMARY[task], "secondary_diagnostic": task in ("advantage", "recovery"),
            "metric_direction": "higher" if metric == "zero_oos_r2" else "lower",
            "block_days": days, "replicates": int(replicates), "estimate": estimate if np.isfinite(estimate) else None,
            "improvement": estimate if metric == "zero_oos_r2" else -estimate,
            "observed_joint_value": joint_value, "observed_baseline_value": baseline_value,
            "ci_low": low, "ci_high": high, "p_value": p,
            "n_paired_rows": int(total[5]), "n_folds": int(paired.fold.nunique()), "n_seeds": 1,
            "n_segments": len(segments), "n_resampleable_segments": varying,
            "n_segments_shorter_than_block": shorter, "n_segments_equal_to_block": equal,
            "n_resampleable_rows": supported_rows, "n_nonresampleable_rows": unsupported_rows,
            "resampleable_row_fraction": supported_rows / total[5], "n_valid_replicates": len(finite),
            "inference_available": available, "inference_unavailable_reason": None if available else "no varying supported block segments or undefined estimate",
            "unsupported_segment_policy": "retained in full estimate and fixed in every replicate; never shorten blocks"})
    return rows


def _classification_rows(frame, blocks, reps, seed):
    selected = frame.loc[frame.formulation.isin(("classification", "joint"))].copy()
    selected["formulation"] = selected.formulation.map({"classification": "regression", "joint": "classification"})
    rows = paired_day_count_inference(selected, block_days=blocks, replicates=reps, random_seed=seed)
    for row in rows:
        row.update(comparison="joint_minus_classification_balanced_accuracy", head="classification",
            inference_metric="balanced_accuracy", report_primary_metric="balanced_accuracy", secondary_diagnostic=False,
            metric_direction="higher", observed_joint_value=row.pop("observed_classification_ba"),
            observed_baseline_value=row.pop("observed_regression_ba"), improvement=row["estimate"])
    return rows


def _assert_control(native, classification):
    paired = _prepare_native(native)
    for name in SUMS:
        if not np.array_equal(paired[name + "_baseline"], paired[name + "_joint"]):
            raise ValueError("independent classical head control must exactly reuse regression outputs")
    keys = [*KEYS, "n_classes", "n_rows", *[f"{kind}_c{i}" for kind in ("actual", "correct") for i in range(3)]]
    baseline = classification.loc[classification.formulation == "classification", keys].sort_values(list(KEYS)).reset_index(drop=True)
    joint = classification.loc[classification.formulation == "joint", keys].sort_values(list(KEYS)).reset_index(drop=True)
    if not baseline.equals(joint):
        raise ValueError("independent classical head control must exactly reuse classification outputs")


def write_joint_grid_statistics(output_dir, config):
    """Write112 neural contrasts plus112 explicitly deterministic controls."""
    root = Path(output_dir).resolve()
    final = root / "final"
    completion = json.loads((final / "report_completion.json").read_text())
    if completion.get("cells") != 336 or not completion.get("report_complete"):
        raise ValueError("complete336 report required before joint inference")
    native_path, counts_path = final / "day_native_statistics.csv", final / "day_classification_counts.csv"
    native, counts = pd.read_csv(native_path), pd.read_csv(counts_path)
    native = native.loc[native.split == "test"]; counts = counts.loc[counts.split == "test"]
    expected = {(g, m, t) for g in "ABCD" for m in MODELS for t in TASKS}
    for frame in (native, counts):
        if set(frame[["group", "model", "task"]].itertuples(index=False, name=None)) != expected:
            raise ValueError("inference requires all112 input/model/target panels")
    settings = config.get("statistics", {})
    blocks = tuple(settings.get("block_days", [1, 7, 28]))
    primary = int(settings.get("primary_block_days", 7))
    if primary != 7 or primary not in blocks:
        raise ValueError("primary block length must be7days")
    reps = int(settings.get("bootstrap_replicates", 10000)); seed = int(settings.get("seed", 20261111))
    records = []
    for number, (group, model, task) in enumerate(sorted(expected)):
        def panel(frame):
            selected = frame.loc[(frame.group == group) & (frame.model == model) & (frame.task == task)]
            if "folds" in config and set(selected.fold) != set(config["folds"]):
                raise ValueError("joint inference is missing a registered fold")
            if "seeds" in config and set(selected.seed) != set(config["seeds"]):
                raise ValueError("joint inference requires matched registered seed")
            return selected
        n, c = panel(native), panel(counts)
        control = model in ("linear", "lightgbm")
        if control:
            _assert_control(n, c)
        # Exact-zero controls need no10000rep simulation. Reuse rigorous
        # validation and point estimates with two replicates, suppress CIs.
        actual_reps = 2 if control else reps
        current = paired_native_day_inference(n, block_days=blocks, replicates=actual_reps, random_seed=seed + 1009 * number)
        current += _classification_rows(c, blocks, actual_reps, seed + 1009 * number + 503)
        for row in current:
            row.update(control_only=control, shared_parameters=not control, primary_inference=row["block_days"] == primary,
                holm_family_size=112, fresh_training=not control)
            if control:
                if row["estimate"] != 0:
                    raise ValueError("algebraic independent-head control must have exactly zero difference")
                row.update(inference_available=False, inference_unavailable_reason="deterministic reused independent-head control; no new joint learning",
                    ci_low=None, ci_high=None, p_value=None, replicates=0, n_valid_replicates=0)
        records.extend(current)
    def key(row):
        return "/".join(str(row[name]) for name in ("group", "model", "task", "head"))
    family = {key(row): row["p_value"] for row in records if row["primary_inference"] and not row["control_only"]}
    if len(family) != 112:
        raise ValueError("registered Holm family must contain exactly112 neural contrasts")
    adjusted = holm_adjust(family)
    for row in records:
        row["p_holm_112"] = adjusted.get(key(row)) if row["primary_inference"] and not row["control_only"] else None
    target = final / "joint_inference.csv"
    pd.DataFrame(records).to_csv(target, index=False)
    summary = {"complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_comparisons": 224, "n_neural_inferential_comparisons": 112, "n_deterministic_classical_controls": 112,
        "n_primary_comparisons": 112,
        "n_rows": len(records), "primary_block_days": primary, "block_days": list(blocks), "bootstrap_replicates": reps,
        "holm_family": "112 neural contrasts; deterministic classical controls excluded from inferential family",
        "rank_task_policy": "Spearman point scores retained; secondary RMSE receives exact block inference; no approximate Spearman CI",
        "resampling": "paired non-circular UTC-day blocks separately within fold and contiguous15min segment",
        "unsupported_segments": "retain observations fixed in each replicate; no shorter-block fallback",
        "test_report_only": True, "test_selection_allowed": False, "historical_seen_test": True,
        "daily_native_sha256": _sha(native_path), "daily_counts_sha256": _sha(counts_path),
        "inference_sha256": _sha(target), "statistics_source_sha256": _sha(Path(__file__)), "inference_csv": str(target)}
    _json(root / "statistics_completion.json", summary)
    return summary
