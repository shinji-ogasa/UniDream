"""Read-only post-hoc inference for frozen prediction panels.

All metrics use the same non-circular moving UTC-day blocks within each
fold and contiguous 15-minute segment.  Unsupported segments are retained
unchanged in every replicate.  Rank bootstrap samples are re-ranked exactly
with frequency weights; bootstrapping a correlation of fixed ranks is never
reported as a Spearman confidence interval.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import os

import numpy as np
import pandas as pd

from .wm_ablation_evaluation import variance_qlike


BAR_NS = 15 * 60 * 10**9
NATIVE_PRIMARY = {"return": "zero_oos_r2", "variance": "qlike", "downside": "mae",
                  "crash": "brier", "dd_excess": "mae", "advantage": "spearman_ic",
                  "recovery": "spearman_ic"}
HIGHER = {"balanced_accuracy", "zero_oos_r2", "pearson_ic", "spearman_ic"}
_LIBRARIES: dict[str, object] = {}


def compile_rank_backend(directory: str | Path) -> Path:
    """Compile the auditable local helper; no package install or network I/O."""
    source = Path(__file__).with_name("wm_significance_rank.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"weighted_spearman_{digest[:16]}.dylib"
    if not target.exists():
        temporary = target.with_suffix(".tmp.dylib")
        subprocess.run(["clang++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(source), "-o", str(temporary)], check=True,
                       capture_output=True, text=True)
        temporary.replace(target)
    return target


def _order_and_ends(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable").astype(np.int64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return order, np.repeat(ends, ends - starts).astype(np.int64)


def _weighted_ranks(values, weights):
    order, ends = _order_and_ends(values)
    starts = np.r_[0, np.flatnonzero(ends[1:] != ends[:-1]) + 1]
    group_weights = np.add.reduceat(np.asarray(weights, float)[order], starts)
    ranks = np.cumsum(group_weights) - (group_weights - 1) / 2
    result = np.empty(len(order), dtype=float)
    result[order] = np.repeat(ranks, ends[starts] - starts)
    return result


def weighted_spearman(actual, prediction, row_day, day_weights, *, backend_path=None, cache_dir=None):
    """Exact Spearman for every expanded integer-weight bootstrap sample.

    Zero-weight rows do not contribute. Ties, including those introduced by
    repeating a sampled row, receive the midpoint of their expanded ranks.
    Python fallback supports small validation fixtures; large production
    panels require an explicitly compiled backend to prevent accidental
    unbounded Python execution.
    """
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    day_raw, weight_raw = np.asarray(row_day), np.asarray(day_weights)
    if (actual.ndim != 1 or prediction.shape != actual.shape or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(prediction).all()):
        raise ValueError("rank inputs must be nonempty matched finite vectors")
    if (day_raw.shape != actual.shape or weight_raw.ndim != 2 or not len(weight_raw)
            or not np.isfinite(day_raw).all() or not np.isfinite(weight_raw).all()
            or np.any(day_raw != np.floor(day_raw)) or np.any(weight_raw != np.floor(weight_raw))
            or np.any(day_raw < 0) or np.any(weight_raw < 0)
            or np.max(day_raw) >= weight_raw.shape[1]
            or np.any(weight_raw > np.iinfo(np.int32).max)):
        raise ValueError("rank day identifiers and replicate weights must be valid nonnegative integers")
    days = np.ascontiguousarray(day_raw, dtype=np.int32)
    weights = np.ascontiguousarray(weight_raw, dtype=np.int32)
    result = np.full(len(weights), np.nan, dtype=np.float64)
    # A truly constant original vector remains constant under any resampling.
    if np.ptp(actual) == 0 or np.ptp(prediction) == 0:
        return result
    cache_path = None
    if cache_dir is not None:
        digest = hashlib.sha256(b"exact_frequency_weighted_spearman_v1")
        for array in (actual, prediction, days, weights):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.dtype.str.encode())
            digest.update(memoryview(contiguous).cast("B"))
        cache_path = Path(cache_dir).resolve() / (digest.hexdigest() + ".npy")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if cached.shape != result.shape or np.isinf(cached).any() or np.any(np.isfinite(cached) & (np.abs(cached) > 1)):
                raise ValueError("invalid cached exact rank distribution")
            return cached
    if backend_path is None:
        if len(actual) * len(weights) > 2_000_000:
            raise ValueError("large exact rank inference requires the compiled backend")
        for index, sample in enumerate(weights):
            w = sample[days].astype(float)
            n = w.sum()
            if n < 2:
                continue
            x, y = _weighted_ranks(actual, w), _weighted_ranks(prediction, w)
            x -= (n + 1) / 2
            y -= (n + 1) / 2
            denominator = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
            if denominator > 0:
                result[index] = np.sum(w * x * y) / denominator
        if cache_path is not None:
            _cache_rank(cache_path, result)
        return result
    path = str(Path(backend_path).resolve())
    if path not in _LIBRARIES:
        library = ctypes.CDLL(path)
        library.unidream_weighted_spearman.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 7
        library.unidream_weighted_spearman.restype = ctypes.c_int
        _LIBRARIES[path] = library
    x_order, x_ends = _order_and_ends(actual)
    y_order, y_ends = _order_and_ends(prediction)
    arrays = (x_order, x_ends, y_order, y_ends, days, weights, result)
    error = _LIBRARIES[path].unidream_weighted_spearman(
        len(actual), weights.shape[1], len(weights),
        *[ctypes.c_void_p(array.ctypes.data) for array in arrays])
    if error:
        raise RuntimeError(f"exact rank helper failed with status {error}")
    # Arithmetic rounding may produce a few ulps outside the correlation range.
    result = np.clip(result, -1, 1)
    if cache_path is not None:
        _cache_rank(cache_path, result)
    return result


def _cache_rank(path, values):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def moving_day_weights(segments, *, n_days, block_days, replicates, rng):
    """Frequency counts for exact truncate-to-length non-circular MBB."""
    if (int(n_days) != n_days or n_days < 1 or int(block_days) != block_days or block_days < 1
            or int(replicates) != replicates or replicates < 2):
        raise ValueError("positive whole day/block counts and at least two replicates required")
    pieces = [np.asarray(piece) for piece in segments]
    if (not pieces or any(p.ndim != 1 or not len(p) or not np.issubdtype(p.dtype, np.integer) for p in pieces)
            or not np.array_equal(np.sort(np.concatenate(pieces)), np.arange(n_days))):
        raise ValueError("segments must partition the unique daily rows exactly once")
    result = np.zeros((replicates, n_days), dtype=np.int32)
    for ids in pieces:
        length = len(ids)
        if length <= block_days:
            result[:, ids] = 1
            continue
        whole, remainder = divmod(length, block_days)
        count = whole + bool(remainder)
        starts = rng.integers(0, length - block_days + 1, size=(replicates, count))
        differences = np.zeros((replicates, length + 1), dtype=np.int32)
        rows = np.arange(replicates)[:, None]
        if whole:
            np.add.at(differences, (rows, starts[:, :whole]), 1)
            np.add.at(differences, (rows, starts[:, :whole] + block_days), -1)
        if remainder:
            np.add.at(differences, (np.arange(replicates), starts[:, -1]), 1)
            np.add.at(differences, (np.arange(replicates), starts[:, -1] + remainder), -1)
        result[:, ids] = np.cumsum(differences[:, :-1], axis=1)
    return result


def _prepare(frame, task):
    required = {"fold", "timestamp", "actual", "label", "predicted_class", "baseline_class"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("one nonempty matched row panel with outcomes and both class decisions required")
    selected = frame.copy()
    times = pd.to_datetime(selected.timestamp, utc=True, errors="coerce")
    if times.isna().any() or (times.astype("int64") % BAR_NS).any():
        raise ValueError("timestamps must lie on the finite UTC15-minute grid")
    selected["timestamp"] = times
    fold_values = pd.to_numeric(selected.fold, errors="coerce").to_numpy(float)
    if not np.isfinite(fold_values).all() or np.any(fold_values < 0) or np.any(fold_values != np.floor(fold_values)):
        raise ValueError("fold identifiers must be nonnegative integers")
    if selected.duplicated(["fold", "timestamp"]).any() or selected.timestamp.duplicated().any():
        raise ValueError("decision rows must be unique and folds may not overlap")
    selected = selected.sort_values(["fold", "timestamp"]).reset_index(drop=True)
    k = 2 if task == "crash" else 3
    for name in ("actual", "label", "predicted_class", "baseline_class"):
        values = pd.to_numeric(selected[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("all paired outcomes and predictions must be finite; rows are never dropped")
        if name != "actual" and (np.any(values != np.floor(values)) or not np.isin(values, np.arange(k)).all()):
            raise ValueError("class values must belong to the task's fixed class universe")
        selected[name] = values
    point = "prediction" in selected or "baseline_prediction" in selected
    if point:
        if not {"prediction", "baseline_prediction"}.issubset(selected):
            raise ValueError("both model and baseline scalar vectors are required")
        if not np.isfinite(selected[["prediction", "baseline_prediction"]].to_numpy(float)).all():
            raise ValueError("scalar forecasts must be finite on the complete paired cohort")
    if task in ("variance", "downside", "dd_excess") and np.any(selected.actual < 0):
        raise ValueError("nonnegative task outcomes required")
    if task == "crash" and not np.isin(selected.actual, (0, 1)).all():
        raise ValueError("crash outcomes must be binary")
    if task == "crash" and not np.array_equal(selected.actual.to_numpy(), selected.label.to_numpy()):
        raise ValueError("crash labels must equal the binary event outcomes")
    if task == "crash" and point and not ((selected[["prediction", "baseline_prediction"]] >= 0)
            & (selected[["prediction", "baseline_prediction"]] <= 1)).to_numpy().all():
        raise ValueError("crash Brier forecasts must be probabilities in[0,1]")
    folds = selected.fold.to_numpy()
    ns = selected.timestamp.astype("int64").to_numpy()
    split = np.r_[True, (folds[1:] != folds[:-1]) | (np.diff(ns) != BAR_NS)]
    segment = np.cumsum(split) - 1
    selected["segment_id"] = segment
    selected["utc_day"] = selected.timestamp.dt.floor("D")
    daily = selected[["fold", "segment_id", "utc_day"]].drop_duplicates().reset_index(drop=True)
    mapping = {(f, s, d): i for i, (f, s, d) in enumerate(daily.itertuples(index=False, name=None))}
    row_day = np.array([mapping[x] for x in selected[["fold", "segment_id", "utc_day"]].itertuples(index=False, name=None)], dtype=np.int32)
    daily["n_rows"] = np.bincount(row_day, minlength=len(daily))
    if (daily.groupby(["fold", "utc_day"]).n_rows.sum() > 96).any():
        raise ValueError("daily supports may not exceed96 decision rows")
    segments = [p.index.to_numpy(np.int64) for _, p in daily.groupby(["fold", "segment_id"], sort=False)]
    return selected, daily, row_day, segments, k, point


def _pearson(sx, sy, sxx, syy, sxy, count):
    covariance = sxy - sx * sy / count
    vx = np.maximum(0., sxx - sx * sx / count)
    vy = np.maximum(0., syy - sy * sy / count)
    denominator = np.sqrt(vx * vy)
    return np.divide(covariance, denominator, out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)


def _statistics(selected, row_day, n_days, classes, point, task):
    actual, label = selected.actual.to_numpy(float), selected.label.to_numpy(np.int64)
    centered_actual = actual - actual.mean()
    values = {"n": np.ones(len(actual)), "y": centered_actual, "yc2": centered_actual ** 2,
              "y2": actual ** 2}
    for c in range(classes):
        values[f"support{c}"] = (label == c).astype(float)
    for prefix, class_name, pred_name in (("model", "predicted_class", "prediction"),
                                          ("baseline", "baseline_class", "baseline_prediction")):
        decision = selected[class_name].to_numpy(np.int64)
        for c in range(classes):
            values[f"{prefix}_correct{c}"] = ((label == c) & (decision == c)).astype(float)
        if point:
            prediction = selected[pred_name].to_numpy(float)
            centered_prediction = prediction - prediction.mean()
            delta = prediction - actual
            values.update({f"{prefix}_sse": delta ** 2, f"{prefix}_sae": np.abs(delta),
                           f"{prefix}_p": centered_prediction, f"{prefix}_p2": centered_prediction ** 2,
                           f"{prefix}_py": centered_prediction * centered_actual})
            if task == "variance":
                values[f"{prefix}_qlike"] = variance_qlike(actual, prediction)
    names = list(values)
    aggregated = np.column_stack([np.bincount(row_day, weights=values[name], minlength=n_days) for name in names])
    if not np.isfinite(aggregated).all():
        raise ValueError("nonfinite daily sufficient statistics; no overflow or silent row removal allowed")
    return names, aggregated


def _metrics(sums, names, classes, point, task, prefix):
    values = dict(zip(names, np.moveaxis(np.asarray(sums, float), -1, 0)))
    n = values["n"]
    supports = np.stack([values[f"support{c}"] for c in range(classes)], axis=-1)
    correct = np.stack([values[f"{prefix}_correct{c}"] for c in range(classes)], axis=-1)
    recall = np.divide(correct, supports, out=np.zeros_like(correct, dtype=float), where=supports > 0)
    result = {"balanced_accuracy": np.where((supports > 0).all(axis=-1), recall.mean(axis=-1), np.nan)}
    if point:
        sse = values[f"{prefix}_sse"]
        result.update(rmse=np.sqrt(sse / n), mae=values[f"{prefix}_sae"] / n,
                      pearson_ic=_pearson(values["y"], values[f"{prefix}_p"], values["yc2"],
                                          values[f"{prefix}_p2"], values[f"{prefix}_py"], n))
        if task == "return":
            result["zero_oos_r2"] = np.where(values["y2"] > 0, 1 - sse / np.maximum(values["y2"], np.finfo(float).tiny), np.nan)
        if task == "variance":
            result["qlike"] = values[f"{prefix}_qlike"] / n
        if task == "crash":
            result["brier"] = sse / n
    return result


def _interval(distribution):
    finite = np.asarray(distribution, float)
    finite = finite[np.isfinite(finite)]
    return (tuple(map(float, np.quantile(finite, [.025, .975]))) if len(finite) >= 2 else (None, None), len(finite))


def infer_panel(frame, *, task, blocks=(1, 7, 28), replicates=10000, random_seed=20261112,
                rank_backend_path=None, include_spearman=True, metrics=None,
                replicates_by_block=None, rank_cache_dir=None):
    """Return absolute metric CIs and paired, direction-aware baseline tests.

    The caller identifies the condition and baseline and assigns explicit
    multiplicity families. Every returned p value is a centered, two-sided
    paired bootstrap test of zero improvement, with the finite-simulation
    correction (1+extreme)/(1+valid). P values are unadjusted here.
    """
    if task not in NATIVE_PRIMARY or int(replicates) != replicates or replicates < 2:
        raise ValueError("known task and at least two integer bootstrap replicates required")
    if (not blocks or len(set(blocks)) != len(blocks) or any(int(b) != b or b < 1 for b in blocks)
            or int(random_seed) != random_seed or random_seed < 0):
        raise ValueError("distinct positive whole block lengths and nonnegative whole seed required")
    selected, daily, row_day, segments, classes, point = _prepare(frame, task)
    names, sums = _statistics(selected, row_day, len(daily), classes, point, task)
    total = sums.sum(axis=0)
    observed = {prefix: _metrics(total, names, classes, point, task, prefix) for prefix in ("model", "baseline")}
    constant_ic = {prefix: point and (np.ptp(selected.actual.to_numpy(float)) == 0
        or np.ptp(selected[column].to_numpy(float)) == 0)
        for prefix, column in (("model", "prediction"), ("baseline", "baseline_prediction"))}
    for prefix in observed:
        if constant_ic[prefix]:
            observed[prefix]["pearson_ic"] = np.asarray(np.nan)
    if metrics is not None:
        if not metrics or len(set(metrics)) != len(metrics):
            raise ValueError("requested metric names must be nonempty and distinct")
        available = set(observed["model"]) | ({"spearman_ic"} if point and include_spearman else set())
        if not set(metrics).issubset(available):
            raise ValueError("requested metric is unavailable for this panel")
    use_rank = point and include_spearman and (metrics is None or "spearman_ic" in metrics)
    if use_rank:
        for prefix, column in (("model", "prediction"), ("baseline", "baseline_prediction")):
            observed[prefix]["spearman_ic"] = weighted_spearman(selected.actual, selected[column], row_day,
                np.ones((1, len(daily)), dtype=np.int32), backend_path=rank_backend_path, cache_dir=rank_cache_dir)[0]
    if metrics is not None:
        observed = {prefix: {name: values[name] for name in metrics} for prefix, values in observed.items()}
    records = []
    for days in map(int, blocks):
        current_reps = (replicates_by_block or {}).get(days, replicates)
        if int(current_reps) != current_reps or current_reps < 2:
            raise ValueError("per-block replicate counts must be integers of at least two")
        current_reps = int(current_reps)
        rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), days]))
        weights = moving_day_weights(segments, n_days=len(daily), block_days=days, replicates=current_reps, rng=rng)
        # Accelerate BLAS on macOS may leave spurious FP status flags even
        # for finite products. Explicitly check the actual output instead.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            bootstrap = weights.astype(float) @ sums
        if not np.isfinite(bootstrap).all():
            raise ValueError("nonfinite resampled sufficient statistics")
        varying = [ids for ids in segments if len(ids) > days]
        supported = int(daily.iloc[np.concatenate(varying)].n_rows.sum()) if varying else 0
        distributions = {prefix: _metrics(bootstrap, names, classes, point, task, prefix) for prefix in ("model", "baseline")}
        for prefix in distributions:
            if constant_ic[prefix]:
                distributions[prefix]["pearson_ic"] = np.full(current_reps, np.nan)
        if use_rank:
            for prefix, column in (("model", "prediction"), ("baseline", "baseline_prediction")):
                distributions[prefix]["spearman_ic"] = weighted_spearman(selected.actual, selected[column], row_day,
                    weights, backend_path=rank_backend_path, cache_dir=rank_cache_dir)
        for metric in observed["model"]:
            sign = 1 if metric in HIGHER else -1
            model, baseline = float(observed["model"][metric]), float(observed["baseline"][metric])
            estimate = sign * (model - baseline)
            model_boot, baseline_boot = distributions["model"][metric], distributions["baseline"][metric]
            differences = sign * (model_boot - baseline_boot)
            finite = differences[np.isfinite(differences)]
            absolute_ci, absolute_valid = _interval(model_boot)
            baseline_ci, baseline_valid = _interval(baseline_boot)
            improvement_ci, _ = _interval(finite)
            available = bool(varying and np.isfinite(estimate) and len(finite) >= 2)
            extreme = int(np.sum(np.abs(finite - estimate) >= abs(estimate))) if available else None
            p_value = float((1 + extreme) / (len(finite) + 1)) if available else None
            chance_p = chance_low = chance_high = chance_improvement = None
            if metric == "balanced_accuracy" and np.isfinite(model):
                chance_improvement = model - 1 / classes
            if metric == "balanced_accuracy" and varying and np.isfinite(model) and absolute_valid >= 2:
                chance_finite = np.asarray(model_boot)[np.isfinite(model_boot)] - 1 / classes
                chance_p = float((1 + np.sum(np.abs(chance_finite - chance_improvement) >= abs(chance_improvement))) / (len(chance_finite) + 1))
                (chance_low, chance_high), _ = _interval(chance_finite)
            # Zero baseline-difference with identical vectors is a valid zero
            # contrast, not bootstrap evidence that a new model was learned.
            records.append({"task": task, "metric": metric, "higher_is_better": metric in HIGHER,
                "native_primary": metric == NATIVE_PRIMARY[task], "block_days": days,
                "primary_block": days == 7, "replicates": current_reps, "n_rows": len(selected),
                "n_folds": int(selected.fold.nunique()), "n_days": len(daily), "n_day_segments": len(daily),
                "n_utc_days": int(selected.utc_day.nunique()), "n_segments": len(segments),
                "n_resampleable_segments": len(varying), "n_resampleable_rows": supported,
                "n_fixed_rows": len(selected) - supported, "resampleable_row_fraction": supported / len(selected),
                "n_valid_replicates": len(finite), "n_undefined_replicates": current_reps - len(finite),
                "model_estimate": model if np.isfinite(model) else None,
                "model_ci_low": absolute_ci[0] if varying and np.isfinite(model) else None,
                "model_ci_high": absolute_ci[1] if varying and np.isfinite(model) else None,
                "model_valid_replicates": absolute_valid,
                "baseline_estimate": baseline if np.isfinite(baseline) else None,
                "baseline_ci_low": baseline_ci[0] if varying and np.isfinite(baseline) else None,
                "baseline_ci_high": baseline_ci[1] if varying and np.isfinite(baseline) else None,
                "baseline_valid_replicates": baseline_valid,
                "improvement": estimate if np.isfinite(estimate) else None,
                "improvement_ci_low": improvement_ci[0] if available else None,
                "improvement_ci_high": improvement_ci[1] if available else None,
                "p_value": p_value, "inference_available": available,
                "bootstrap_extreme_count": extreme,
                "p_monte_carlo_floor": 1 / (len(finite) + 1) if available else None,
                "p_monte_carlo_standard_error": float(np.sqrt(p_value * (1 - p_value) / (len(finite) + 1))) if available else None,
                "chance_baseline": 1 / classes if metric == "balanced_accuracy" else None,
                "chance_improvement": chance_improvement, "chance_improvement_ci_low": chance_low,
                "chance_improvement_ci_high": chance_high, "chance_p_value": chance_p,
                "inference_unavailable_reason": None if available else "no varying supported segment or undefined paired metric",
                "unsupported_segment_policy": "fixed contribution; no shorter-block fallback",
                "rank_method": "exact frequency-weighted re-ranking within every replicate" if metric == "spearman_ic" else None})
    return records


def holm_fixed_family(p_values, *, family_size):
    """Holm correction retaining unavailable/unused hypotheses as p=1.

    Supply one distinct stable identifier per explored hypothesis. Aliases
    may be included conservatively but must never silently reduce the fixed
    search family. Missing p values remain unavailable in the returned map.
    """
    if int(family_size) != family_size or family_size < len(p_values) or family_size < 1:
        raise ValueError("fixed Holm family size must cover all supplied hypotheses")
    for value in p_values.values():
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("p values must be None or finite probabilities")
    ordered = sorted(p_values, key=lambda key: 1. if p_values[key] is None else p_values[key])
    adjusted, previous = {}, 0.
    for rank, key in enumerate(ordered):
        value = 1. if p_values[key] is None else float(p_values[key])
        previous = max(previous, min(1., value * (int(family_size) - rank)))
        adjusted[key] = previous if p_values[key] is not None else None
    return adjusted
