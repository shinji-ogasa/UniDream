"""Paired native-head inference for joint versus single-task return training.

Non-circular UTC-day blocks are sampled separately within every fold and
contiguous 15-minute segment. Additive daily statistics retain exact pooled
RMSE, zero-reference OOS R2, Pearson IC, proper probability scores and BA.
"""
from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_ablation_evaluation import holm_adjust
from .wm_full_grid_statistics import _block_sums
from .wm_return_multitask_evaluation import MODELS
from .wm_return_multitask_report import CLASSIFICATION_SUMS, REGRESSION_SUMS, _sha


KEYS = ["fold", "seed", "utc_day", "segment_id"]
SHARED = ["n_rows", "actual_sum", "actual2_sum", *(f"actual_c{c}" for c in range(3))]
METRICS = {"regression": ("zero_oos_r2", "rmse", "pearson_ic"),
           "classification": ("balanced_accuracy", "raw_balanced_accuracy", "log_loss", "multiclass_brier")}
PRIMARY = {"regression": "zero_oos_r2", "classification": "balanced_accuracy"}


def _prepare_pair(frame, model, task):
    columns = REGRESSION_SUMS if task == "regression" else CLASSIFICATION_SUMS
    required = {"split", "model", "arm", "group", *KEYS, *SHARED, *columns}
    if required - set(frame):
        raise ValueError("daily statistics lack required pairing/sufficient-statistic columns")
    baseline = "reg_only" if task == "regression" else "class_only"
    panel = frame.loc[(frame.split == "test") & (frame.model == model) & frame.arm.isin((baseline, "joint"))].copy()
    if panel.empty or set(panel.arm) != {baseline, "joint"} or set(panel.group) != {"D"}:
        raise ValueError("paired Test statistics require group D and both registered native arms")
    if set(panel.seed) != {7}:
        raise ValueError("paired inference requires the one registered matched seed 7")
    panel["utc_day"] = pd.to_datetime(panel.utc_day, utc=True, errors="coerce")
    if panel.utc_day.isna().any() or not panel.utc_day.eq(panel.utc_day.dt.floor("D")).all():
        raise ValueError("utc_day must contain UTC midnight days")
    if panel.duplicated(["arm", *KEYS]).any():
        raise ValueError("duplicate native-arm day/segment rows")
    numbers = panel[[*SHARED, *columns]].to_numpy(float)
    if not np.isfinite(numbers).all():
        raise ValueError("active native sufficient statistics must be finite")
    integer = ["fold", "segment_id", "n_rows", *(f"actual_c{c}" for c in range(3))]
    if task == "classification":
        integer += [f"{rule}_correct_c{c}" for rule in ("prior", "raw") for c in range(3)]
    for column in integer:
        values = panel[column].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("day identifiers/support/correct counts must be nonnegative integers")
    support = panel[[f"actual_c{c}" for c in range(3)]].to_numpy(float)
    if np.any(panel.n_rows <= 0) or not np.array_equal(support.sum(1), panel.n_rows.to_numpy()):
        raise ValueError("class/day supports must sum to positive n_rows")
    if np.any(panel.actual2_sum < 0):
        raise ValueError("return-square totals must be nonnegative")
    if task == "classification":
        for rule in ("prior", "raw"):
            if np.any(panel[[f"{rule}_correct_c{c}" for c in range(3)]].to_numpy(float) > support):
                raise ValueError("correct counts cannot exceed actual support")
        if np.any(panel[["log_loss_sum", "brier_sum"]].to_numpy(float) < 0):
            raise ValueError("proper loss totals must be nonnegative")
    elif np.any(panel[["pred2_sum", "sse_sum"]].to_numpy(float) < 0):
        raise ValueError("prediction-square/error totals must be nonnegative")
    if (panel.groupby(["arm", "fold", "seed", "utc_day"]).n_rows.sum() > 96).any():
        raise ValueError("daily support exceeds the 15-minute grid")
    left = panel.loc[panel.arm == baseline]
    right = panel.loc[panel.arm == "joint"]
    paired = left.merge(right, on=KEYS, suffixes=("_base", "_joint"), how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("native arms need identical fold/segment/day coverage")
    for column in SHARED:
        if not np.array_equal(paired[column + "_base"], paired[column + "_joint"]):
            raise ValueError("paired native arms must have identical outcomes/day support")
    paired = paired.sort_values(["fold", "segment_id", "utc_day"]).reset_index(drop=True)
    for _, segment in paired.groupby(["fold", "segment_id"], sort=True):
        if len(segment) > 1 and not segment.utc_day.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError("a contiguous segment cannot bridge missing UTC days")
    names = [column + "_base" for column in SHARED] + [column + suffix for suffix in ("_base", "_joint") for column in columns]
    return paired, names, columns, baseline


def _scores(sums, task, names, suffix):
    values = {name: sums[..., index] for index, name in enumerate(names)}
    n = values["n_rows_base"]
    if task == "regression":
        sse, y2 = values["sse_sum" + suffix], values["actual2_sum_base"]
        ysum, xsum = values["actual_sum_base"], values["pred_sum" + suffix]
        yy = np.maximum(0., y2 - ysum ** 2 / n)
        xx = np.maximum(0., values["pred2_sum" + suffix] - xsum ** 2 / n)
        cross = values["actual_pred_sum" + suffix] - ysum * xsum / n
        denominator = np.sqrt(xx * yy)
        # Prefix summation can leave tiny positive variance for an exactly
        # constant forecast. Such numerical cancellation is not an IC signal.
        tolerance = 64 * np.finfo(np.float64).eps
        variable = ((xx > tolerance * np.maximum(values["pred2_sum" + suffix], np.finfo(float).tiny))
                    & (yy > tolerance * np.maximum(y2, np.finfo(float).tiny)))
        with np.errstate(divide="ignore", invalid="ignore"):
            return {"zero_oos_r2": np.where(y2 > 0, 1. - sse / y2, np.nan),
                    "rmse": np.sqrt(sse / n),
                    "pearson_ic": np.where(variable, np.clip(cross / denominator, -1., 1.), np.nan)}
    support = np.stack([values[f"actual_c{c}_base"] for c in range(3)], axis=-1)
    result = {}
    for rule, metric in (("prior", "balanced_accuracy"), ("raw", "raw_balanced_accuracy")):
        correct = np.stack([values[f"{rule}_correct_c{c}" + suffix] for c in range(3)], axis=-1)
        recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
        result[metric] = np.where((support > 0).all(axis=-1), recall.mean(axis=-1), np.nan)
    result["log_loss"] = values["log_loss_sum" + suffix] / n
    result["multiclass_brier"] = values["brier_sum" + suffix] / n
    return result


def _improvements(base, joint):
    return {name: (base[name] - joint[name] if name in ("rmse", "log_loss", "multiclass_brier")
                   else joint[name] - base[name]) for name in base}


def paired_native_inference(frame, model, task, *, block_days=(1, 7, 28), replicates=10000, random_seed=20261111):
    """All positive estimates mean joint improves on the corresponding native arm."""
    if model not in MODELS or task not in METRICS:
        raise ValueError("unknown native model/task")
    if isinstance(replicates, bool) or int(replicates) != replicates or replicates < 2:
        raise ValueError("at least two integer bootstrap replicates required")
    if not block_days or any(isinstance(d, bool) or int(d) != d or d < 1 for d in block_days) or len(set(block_days)) != len(block_days):
        raise ValueError("block lengths must be distinct positive integer days")
    paired, names, _, baseline = _prepare_pair(frame, model, task)
    all_values = paired[names].to_numpy(float)
    total = all_values.sum(0)
    base = _scores(total, task, names, "_base")
    joint = _scores(total, task, names, "_joint")
    observed = _improvements(base, joint)
    fold_differences = {name: [] for name in METRICS[task]}
    for _, part in paired.groupby("fold", sort=True):
        fold_sum = part[names].to_numpy(float).sum(0)
        differences = _improvements(_scores(fold_sum, task, names, "_base"), _scores(fold_sum, task, names, "_joint"))
        for name, value in differences.items():
            fold_differences[name].append(float(value))
    segments = [part[names].to_numpy(float) for _, part in paired.groupby(["fold", "segment_id"], sort=True)]
    rows = []
    for block in block_days:
        block = int(block)
        rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), block]))
        resampled = np.zeros((int(replicates), len(names)), dtype=np.float64)
        varying = short = equal = supported = unsupported = 0
        for segment in segments:
            short += int(len(segment) < block)
            equal += int(len(segment) == block)
            if len(segment) > block:
                varying += 1
                supported += int(segment[:, 0].sum())
                resampled += _block_sums(segment, block, int(replicates), rng)
            else:
                unsupported += int(segment[:, 0].sum())
                resampled += segment.sum(0)
        distributions = _improvements(_scores(resampled, task, names, "_base"), _scores(resampled, task, names, "_joint"))
        for metric, distribution in distributions.items():
            estimate = float(observed[metric])
            finite = distribution[np.isfinite(distribution)]
            available = bool(varying and np.isfinite(estimate) and len(finite) >= 2)
            ci_low = ci_high = p_value = None
            if available:
                ci_low, ci_high = map(float, np.quantile(finite, [.025, .975]))
                p_value = float((1 + np.sum(np.abs(finite - estimate) >= abs(estimate))) / (len(finite) + 1))
            defined_folds = [v for v in fold_differences[metric] if np.isfinite(v)]
            rows.append({"split": "test", "group": "D", "model": model, "task": task,
                "baseline_arm": baseline, "candidate_arm": "joint", "metric": metric,
                "comparison": f"joint_vs_{baseline}_{metric}_improvement", "positive_means": "joint improves",
                "block_days": block, "replicates": int(replicates),
                "estimate": estimate if np.isfinite(estimate) else None,
                "observed_baseline": float(base[metric]) if np.isfinite(base[metric]) else None,
                "observed_joint": float(joint[metric]) if np.isfinite(joint[metric]) else None,
                "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
                "n_paired_rows": int(total[0]), "n_paired_day_segments": len(paired),
                "n_folds": int(paired.fold.nunique()), "n_seeds": 1,
                "fold_mean_improvement": float(np.mean(defined_folds)) if defined_folds else None,
                "n_folds_with_undefined_metric": len(fold_differences[metric]) - len(defined_folds),
                "n_segments": len(segments), "n_resampleable_segments": varying,
                "n_segments_shorter_than_block": short, "n_segments_equal_to_block": equal,
                "n_resampleable_rows": supported, "n_nonresampleable_rows": unsupported,
                "resampleable_row_fraction": supported / total[0],
                "n_valid_replicates": int(len(finite)), "n_undefined_replicates": int(replicates - len(finite)),
                "inference_available": available,
                "inference_unavailable_reason": None if available else "no varying segment or insufficient defined metric replicates",
                "unsupported_segment_policy": "retain all rows in point estimate; unsupported segments replay fixed; conditional uncertainty",
            })
    return rows


def write_multitask_statistics(output_dir, config):
    output = Path(output_dir)
    source = output / "final/day_statistics.csv"
    frame = pd.read_csv(source)
    test = frame.loc[frame.split == "test"]
    if set(test.model) != set(MODELS) or set(test.arm) != {"reg_only", "class_only", "joint"}:
        raise ValueError("final inference requires all six native model/arm conditions")
    statistics = config.get("statistics", {})
    blocks = tuple(statistics.get("block_days", (1, 7, 28)))
    primary = int(statistics.get("primary_block_days", 7))
    if primary != 7 or primary not in blocks:
        raise ValueError("registered primary inference uses seven-day blocks")
    replicates = statistics.get("bootstrap_replicates", 10000)
    seed = int(statistics.get("seed", 20261111))
    records = []
    for i, model in enumerate(MODELS):
        for j, task in enumerate(METRICS):
            panel = test.loc[test.model == model]
            if "folds" in config and set(panel.fold) != set(config["folds"]):
                raise ValueError("native inference is missing a registered fold")
            records.extend(paired_native_inference(panel, model, task, block_days=blocks,
                replicates=replicates, random_seed=seed + i * 1009 + j * 31))
    pvalues = {f"{row['model']}/{row['task']}": row["p_value"] for row in records
               if row["block_days"] == primary and row["metric"] == PRIMARY[row["task"]]}
    if len(pvalues) != 4:
        raise ValueError("all four registered primary comparisons required")
    adjusted = holm_adjust(pvalues)
    for row in records:
        row["primary_inference"] = row["block_days"] == primary and row["metric"] == PRIMARY[row["task"]]
        row["holm_family_size"] = 4
        row["p_holm_4"] = adjusted[f"{row['model']}/{row['task']}"] if row["primary_inference"] else None
        row["auxiliary_unadjusted"] = not row["primary_inference"]
    destination = output / "final/multitask_inference.csv"
    table = pd.DataFrame(records); table.to_csv(destination, index=False)
    note = ("Positive differences favor joint training. Primary tests: OOS R² versus reg_only and prior-rule BA versus class_only, "
            "for direct/WM; Holm family 4 at seven-day blocks. Other metrics/block lengths are auxiliary and unadjusted. "
            "All paired Test rows enter point estimates. Short segments stay fixed, so uncertainty is conditional on their contribution. "
            "Blocks never cross folds or missing-bar gaps. Historical Test was seen in previous studies; seed 7 only.")
    (output / "final/statistics.html").write_text("<!doctype html><meta charset='utf-8'><title>Return multitask inference</title><p>"
        + html.escape(note) + "</p>" + table.to_html(index=False, na_rep="—", float_format=lambda v: f"{v:.6g}"), encoding="utf-8")
    summary = {"complete": True, "utc": datetime.now(timezone.utc).isoformat(),
        "n_primary_comparisons": 4, "n_inference_rows": len(records), "primary_block_days": primary,
        "n_primary_available": sum(r["primary_inference"] and r["inference_available"] for r in records),
        "block_days": list(blocks), "bootstrap_replicates": int(replicates), "test_report_only": True,
        "historical_seen_test": True, "seed_stability_claim": False, "native_heads_only": True,
        "daily_statistics_sha256": _sha(source), "inference_sha256": _sha(destination), "note": note}
    (output / "statistics_completion.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
