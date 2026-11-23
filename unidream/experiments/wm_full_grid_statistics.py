"""Paired classification-versus-regression inference from frozen daily counts.

Only Test counts enter inference. Non-circular UTC-day blocks remain inside
each fold and contiguous 15-minute segment. Days unsupported at a requested
block length remain in the observed estimate and are replayed unchanged;
they never trigger shorter-block or IID resampling.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .wm_ablation_evaluation import holm_adjust


GROUPS = ("A", "B", "C", "D")
MODELS = ("linear", "lightgbm", "direct", "wm")
TASKS = ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery")
KEYS = ("fold", "seed", "utc_day", "segment_id")
COUNT_COLUMNS = tuple(f"{kind}_c{category}" for kind in ("actual", "correct") for category in range(3))
REQUIRED_COLUMNS = {"split", "group", "model", "task", "formulation", "n_classes", "n_rows", *KEYS, *COUNT_COLUMNS}


def _ba(support, correct):
    """Fixed class universe: any absent class makes BA undefined."""
    support, correct = np.asarray(support, float), np.asarray(correct, float)
    valid = (support > 0).all(axis=-1)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    return np.where(valid, recall.mean(axis=-1), np.nan)


def _prepare_pair(frame):
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError("daily table lacks required full-grid count columns")
    selected = frame.loc[frame["split"] == "test"].copy()
    if selected.empty:
        raise ValueError("paired inference requires Test day counts")
    for name in ("group", "model", "task"):
        if selected[name].nunique() != 1:
            raise ValueError("one input/model/target comparison is required")
    task = selected["task"].iloc[0]
    if task not in TASKS:
        raise ValueError("unknown prediction target")
    n_classes = 2 if task == "crash" else 3
    integer_columns = ("fold", "seed", "segment_id", "n_classes", "n_rows", *COUNT_COLUMNS)
    for name in integer_columns:
        values = pd.to_numeric(selected[name], errors="coerce").to_numpy(float)
        if (not np.isfinite(values).all() or np.any(values != np.floor(values))
                or (name != "seed" and np.any(values < 0))):
            raise ValueError(f"{name} must contain finite nonnegative integer counts/identifiers")
        selected[name] = values.astype(np.int64)
    if not selected.n_classes.eq(n_classes).all():
        raise ValueError("n_classes must be two for crash and three for all other targets")
    if selected.seed.nunique() != 1:
        raise ValueError("this preregistered comparison requires one matched seed, not independent seed resampling")
    dates = pd.to_datetime(selected["utc_day"], utc=True, errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.floor("D")).all():
        raise ValueError("utc_day must identify UTC midnight calendar days")
    selected["utc_day"] = dates
    if set(selected.formulation) != {"classification", "regression"}:
        raise ValueError("both and only classification/regression formulations are required")
    if selected.duplicated(["formulation", *KEYS]).any():
        raise ValueError("duplicate formulation/fold/segment/day rows")
    actual = selected[[f"actual_c{i}" for i in range(3)]].to_numpy(float)
    correct = selected[[f"correct_c{i}" for i in range(3)]].to_numpy(float)
    if (np.any(correct > actual) or np.any(selected.n_rows <= 0)
            or not np.array_equal(actual.sum(1), selected.n_rows.to_numpy())):
        raise ValueError("correct counts must not exceed support, and class supports must sum to n_rows")
    if n_classes == 2 and (np.any(actual[:, 2]) or np.any(correct[:, 2])):
        raise ValueError("binary crash counts must have zero third-class support and correct counts")
    # An aggregation over multiple disjoint pieces of one UTC day must still
    # represent at most the 96 possible 15-minute decision rows.
    if (selected.groupby(["formulation", "fold", "seed", "utc_day"]).n_rows.sum() > 96).any():
        raise ValueError("daily support exceeds the 15-minute grid")
    left = selected.loc[selected.formulation == "regression"]
    right = selected.loc[selected.formulation == "classification"]
    paired = left.merge(right, on=list(KEYS), suffixes=("_regression", "_classification"),
                        how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("formulations must share identical fold/segment/day coverage")
    for name in ("n_rows", "n_classes", *(f"actual_c{i}" for i in range(3))):
        if not np.array_equal(paired[name + "_regression"], paired[name + "_classification"]):
            raise ValueError("paired formulations must have identical class/day support")
    paired = paired.sort_values(["fold", "segment_id", "utc_day"]).reset_index(drop=True)
    for _, part in paired.groupby(["fold", "segment_id"], sort=True):
        if len(part) > 1 and not part.utc_day.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError("a contiguous segment cannot bridge missing UTC days")
    columns = ([f"actual_c{i}_regression" for i in range(n_classes)]
               + [f"correct_c{i}_regression" for i in range(n_classes)]
               + [f"correct_c{i}_classification" for i in range(n_classes)] + ["n_rows_regression"])
    return paired, columns, n_classes


def _block_sums(values, days, replicates, rng):
    """Sum sampled blocks with prefixes, exactly matching truncate-to-L MBB."""
    length = len(values)
    full_blocks, remainder = divmod(length, days)
    count = full_blocks + int(remainder > 0)
    starts = rng.integers(0, length - days + 1, size=(replicates, count))
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    result = np.zeros((replicates, values.shape[1]), dtype=np.float64)
    if full_blocks:
        full = starts[:, :full_blocks]
        result += (prefix[full + days] - prefix[full]).sum(axis=1)
    if remainder:
        last = starts[:, -1]
        result += prefix[last + remainder] - prefix[last]
    return result


def paired_day_count_inference(frame, *, block_days=(1, 7, 28), replicates=2000, random_seed=20261111):
    """Return K-aware paired BA intervals; positive favors classification.

    Point estimates always use the entire paired cohort. Unsupported segments
    contribute fixed counts to every replicate, making the interval conditional
    on those observations. No varying block starts means no CI or P value.
    """
    if int(replicates) != replicates or replicates < 2:
        raise ValueError("replicates must be an integer of at least two")
    if (not block_days or len(set(block_days)) != len(block_days)
            or any(int(days) != days or days < 1 for days in block_days)):
        raise ValueError("block_days must contain distinct positive whole days")
    if int(random_seed) != random_seed or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    paired, columns, n_classes = _prepare_pair(frame)
    total = paired[columns].sum().to_numpy(float)
    actual = total[:n_classes]
    regression_ba = float(_ba(actual, total[n_classes:2 * n_classes]))
    classification_ba = float(_ba(actual, total[2 * n_classes:3 * n_classes]))
    difference = classification_ba - regression_ba
    fold_differences = []
    for _, part in paired.groupby("fold", sort=True):
        totals = part[columns].sum().to_numpy(float)
        fold_differences.append(float(_ba(totals[:n_classes], totals[2 * n_classes:3 * n_classes])
                                      - _ba(totals[:n_classes], totals[n_classes:2 * n_classes])))
    valid_fold_differences = [value for value in fold_differences if np.isfinite(value)]
    segments = [part[columns].to_numpy(float) for _, part in paired.groupby(["fold", "segment_id"], sort=True)]
    rows = []
    for supplied in block_days:
        days = int(supplied)
        rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), days]))
        resampled = np.zeros((int(replicates), len(columns)), dtype=np.float64)
        varying = shorter = equal = starts = 0
        supported_rows = unsupported_rows = 0.
        for segment in segments:
            length = len(segment)
            shorter += int(length < days)
            equal += int(length == days)
            starts += max(0, length - days + 1)
            if length > days:
                varying += 1
                supported_rows += float(segment[:, -1].sum())
                resampled += _block_sums(segment, days, int(replicates), rng)
            else:
                # Preserve all observations with their original contribution;
                # this is NOT a fallback to shorter blocks or IID sampling.
                unsupported_rows += float(segment[:, -1].sum())
                resampled += segment.sum(axis=0)
        boot_actual = resampled[:, :n_classes]
        distribution = (_ba(boot_actual, resampled[:, 2 * n_classes:3 * n_classes])
                        - _ba(boot_actual, resampled[:, n_classes:2 * n_classes]))
        finite = distribution[np.isfinite(distribution)]
        available = bool(varying and np.isfinite(difference) and len(finite) >= 2)
        ci_low = ci_high = p_value = None
        if available:
            ci_low, ci_high = map(float, np.quantile(finite, [.025, .975]))
            p_value = float((1 + np.sum(np.abs(finite - difference) >= abs(difference))) / (len(finite) + 1))
        reason = (None if available else "no segment has more days than the requested block" if not varying
                  else "observed cohort lacks a required class" if not np.isfinite(difference)
                  else "fewer than two complete-class bootstrap replicates")
        rows.append({
            "split": "test", "group": frame.loc[frame.split == "test", "group"].iloc[0],
            "model": frame.loc[frame.split == "test", "model"].iloc[0],
            "task": frame.loc[frame.split == "test", "task"].iloc[0],
            "comparison": "classification_minus_regression_balanced_accuracy",
            "n_classes": n_classes, "block_days": days, "replicates": int(replicates),
            "estimate": float(difference) if np.isfinite(difference) else None,
            "observed_classification_ba": classification_ba if np.isfinite(classification_ba) else None,
            "observed_regression_ba": regression_ba if np.isfinite(regression_ba) else None,
            "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
            "observed_fold_mean_ba_difference": float(np.mean(valid_fold_differences)) if valid_fold_differences else None,
            "n_folds": len(fold_differences), "n_folds_with_defined_ba": len(valid_fold_differences),
            "n_folds_with_undefined_ba": len(fold_differences) - len(valid_fold_differences),
            "n_paired_rows": int(total[-1]), "n_paired_day_segments": len(paired),
            "n_seeds": 1, "n_segments": len(segments),
            "n_segments_shorter_than_block": shorter, "n_segments_equal_to_block": equal,
            "n_resampleable_segments": varying, "n_nonresampleable_segments": len(segments) - varying,
            "n_possible_block_starts": starts,
            "n_resampleable_rows": int(supported_rows), "n_nonresampleable_rows": int(unsupported_rows),
            "resampleable_row_fraction": supported_rows / total[-1],
            "n_valid_replicates": int(len(finite)), "n_undefined_replicates": int(replicates - len(finite)),
            "inference_available": available, "inference_unavailable_reason": reason,
            "unsupported_segment_policy": "retained in full point estimate and replayed fixed in every replicate; conditional uncertainty",
        })
    return rows


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_grid_statistics(output_dir, config):
    """Write the preregistered112 paired Test comparisons from final counts."""
    output = Path(output_dir).resolve()
    source = output / "final/day_classification_counts.csv"
    frame = pd.read_csv(source)
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError("daily table lacks required full-grid count columns")
    test = frame.loc[frame.split == "test"]
    expected = {(group, model, task) for group in GROUPS for model in MODELS for task in TASKS}
    actual = set(test[["group", "model", "task"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("final inference requires all112 input/model/target comparisons")
    statistics = config.get("statistics", {})
    blocks = tuple(statistics.get("block_days", [1, 7, 28]))
    primary = int(statistics.get("primary_block_days", 7))
    if primary != 7 or primary not in blocks:
        raise ValueError("the primary preregistered block length must be seven days")
    replicates = statistics.get("bootstrap_replicates", 2000)
    seed = int(statistics.get("seed", 20261111))
    records = []
    for number, keys in enumerate(sorted(expected)):
        group, model, task = keys
        panel = test.loc[(test.group == group) & (test.model == model) & (test.task == task)]
        if "folds" in config and set(panel.fold) != set(config["folds"]):
            raise ValueError("inference cohort is missing a registered fold")
        if "seeds" in config and set(panel.seed) != set(config["seeds"]):
            raise ValueError("inference cohort differs from the registered matched seed")
        records.extend(paired_day_count_inference(panel, block_days=blocks, replicates=replicates,
                                                  random_seed=seed + number * 1009))
    values = {f"{row['group']}/{row['model']}/{row['task']}": row["p_value"]
              for row in records if row["block_days"] == primary}
    adjusted = holm_adjust(values)
    for row in records:
        row["primary_inference"] = row["block_days"] == primary
        row["holm_family_size"] = 112
        row["p_holm_112"] = adjusted[f"{row['group']}/{row['model']}/{row['task']}"] if row["primary_inference"] else None
    destination = output / "final/formulation_inference.csv"
    temporary = destination.with_name(destination.name + ".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(destination)
    summary = {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_comparisons": 112, "n_rows": len(records), "primary_block_days": primary,
        "block_days": list(blocks), "bootstrap_replicates": int(replicates),
        "n_primary_available": sum(row["primary_inference"] and row["inference_available"] for row in records),
        "primary_family": "112 group/model/target classification-minus-regression BA contrasts; Holm112",
        "observed_estimate": "pooled class recalls from ALL paired Test rows; fixed K2 crash/K3 other targets",
        "auxiliary_fold_mean": "unweighted mean over folds with every required class; missing-class fold count explicit",
        "resampling": "paired non-circular moving UTC-day blocks, separately within fold and contiguous15minsegment",
        "unsupported_segments": "retain full contribution fixed in every replicate; never shorten block; CI/P unavailable without varying segments",
        "absent_class_replicates": "excluded from fixed-K BA distribution; valid/undefined counts explicit",
        "no_training_or_selection": True, "test_report_only": True, "historical_seen_test": True,
        "daily_counts_sha256": _sha(source), "inference_sha256": _sha(destination),
        "statistics_source_sha256": _sha(Path(__file__)), "inference_csv": str(destination),
    }
    marker = output / "statistics_completion.json"
    marker_tmp = marker.with_name(marker.name + ".tmp")
    marker_tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    marker_tmp.replace(marker)
    return summary
