"""Calendar-paired inference for the isolated tail-information study.

The market day is the sampling unit. Sparse observation days retain their
calendar gaps, and each bootstrap estimates sum(differences)/count rather
than averaging a compressed list of observed days. Identical market blocks
are used for every declared comparison. Seeds never multiply market days.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm


DAY_NS = 86_400_000_000_000


def holm_adjust(p_values):
    """Step-down Holm; invalid hypotheses stay in the declared family as p=1."""
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), np.clip(p, 0., 1.), 1.)
    if not len(p):
        return p
    order = np.argsort(p, kind="stable")
    corrected = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(corrected)
    result[order] = corrected
    return result


def _array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validate(name, item):
    times = np.asarray(item["timestamps"])
    if np.issubdtype(times.dtype, np.datetime64):
        times = times.astype("datetime64[ns]").astype(np.int64)
    elif not np.issubdtype(times.dtype, np.integer):
        raise ValueError(f"{name}: timestamps must contain integer UTC nanoseconds")
    times = times.astype(np.int64, copy=False)
    if times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}: unique increasing market timestamps required")
    if np.any(times == np.iinfo(np.int64).min):
        raise ValueError(f"{name}: missing timestamps are not allowed")
    values = np.asarray(item["values"], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    valid = np.asarray(item["valid"])
    if valid.ndim == 1:
        valid = valid[:, None]
    if values.ndim != 2 or len(values) != len(times) or values.shape[1] < 1:
        raise ValueError(f"{name}: values must be [timestamps,paired seeds]")
    if valid.dtype != bool or valid.shape != values.shape:
        raise ValueError(f"{name}: valid must be a same-shape boolean array")
    if np.any(valid != valid[:, :1]):
        raise ValueError(f"{name}: each paired seed must have exactly the same timestamp cohort")
    family = item.get("family")
    kind = item.get("kind")
    if not isinstance(family, str) or not family or kind not in ("forecast", "policy"):
        raise ValueError(f"{name}: explicit family and forecast/policy kind required")
    return dict(name=str(name), timestamps=times, values=values, valid=valid,
                family=family, kind=kind, supplied_metadata=item.get("metadata", {}),
                event_counts=item.get("event_counts"))


def _daily(item, first_day, days, min_observed_days, min_observations):
    values, mask = item["values"], item["valid"]
    observed = mask[:, 0]
    seeds = values.shape[1]
    sums, counts = np.zeros((days, seeds), float), np.zeros(days, np.int64)
    reason = None
    if np.any(~np.isfinite(values[mask])):
        reason = "nonfinite_observed_difference_including_insolvency"
    finite_rows = observed & np.isfinite(values).all(1)
    locations = item["timestamps"] // DAY_NS - first_day
    # Counts retain even an insolvent/nonfinite observed timestamp. The whole
    # comparison is unavailable; no observation is removed to salvage a test.
    np.add.at(counts, locations[observed], 1)
    np.add.at(sums, locations[finite_rows], values[finite_rows])
    n = int(observed.sum())
    observed_days = int(np.count_nonzero(counts))
    observed_locations = np.flatnonzero(counts)
    observed_span = int(observed_locations[-1] - observed_locations[0] + 1) if len(observed_locations) else 0
    if reason is None and (n < min_observations or observed_days < min_observed_days):
        reason = "insufficient_observed_market_times_or_days"
    if reason is None:
        if np.ptp(values[observed].mean(1)) <= 1e-15:
            reason = "constant_market_mean_difference"
    estimate = float(sums.sum() / (seeds * counts.sum())) if counts.sum() and reason is None else np.nan
    metadata = dict(n_market_observations=n, n_observed_days=observed_days,
        n_observed_calendar_span_days=observed_span,
        n_calendar_days=days, n_zero_count_days=int(np.count_nonzero(counts == 0)),
        n_seeds=seeds, n_independent_market_days=observed_days,
        n_observed_nonfinite=int(np.count_nonzero(mask & ~np.isfinite(values))),
        market_days_not_multiplied_by_seed=True, reason=reason,
        n_observations_by_seed=[n] * seeds, daily_counts_sha256=_array_sha(counts),
        first_observed_timestamp_ns=int(item["timestamps"][observed][0]) if n else None,
        last_observed_timestamp_ns=int(item["timestamps"][observed][-1]) if n else None)
    metadata.update(event_counts=_json_value(item["event_counts"]),
                    event_counts_inferred_from_loss_differences=False,
                    supplied_metadata=_json_value(item["supplied_metadata"]))
    return sums, counts, estimate, metadata


def _sampling_plan(days, block_days, replicates, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(block_days), days, 9137]))
    blocks, remainder = divmod(days, block_days)
    starts = rng.integers(0, days, (replicates, blocks), dtype=np.int64) if days else np.empty((replicates, 0), np.int64)
    tail = rng.integers(0, days, replicates, dtype=np.int64) if remainder else None
    return dict(starts=starts, tail=tail, remainder=remainder,
                sha256=_array_sha(starts) + (":" + _array_sha(tail) if tail is not None else ""))


def _block_totals(daily, block_days, plan):
    """Circular moving-block totals, truncating the final block to calendar D."""
    original_ndim = daily.ndim
    if original_ndim == 1:
        daily = daily[:, None]
    days = len(daily)
    if not days:
        return np.zeros((len(plan["starts"]), daily.shape[1]), float)
    # A block may exceed the calendar length in a sensitivity calculation.
    repeats = (max(block_days, plan["remainder"]) + days - 1) // days + 1
    repeated = np.tile(daily, (repeats, 1))
    prefix = np.concatenate((np.zeros((1, daily.shape[1])), np.cumsum(repeated, axis=0)), axis=0)
    starts = np.arange(days)
    full = prefix[starts + block_days] - prefix[starts]
    answer = full[plan["starts"]].sum(axis=1)
    if plan["remainder"]:
        tail = plan["tail"]
        answer += prefix[tail + plan["remainder"]] - prefix[tail]
    return answer[:, 0] if original_ndim == 1 else answer


def _bootstrap(sums, counts, estimate, block_days, plan, seed_draws=None):
    boot_counts = _block_totals(counts, block_days, plan)
    if seed_draws is None:
        numerator = _block_totals(sums.mean(1), block_days, plan)
    else:
        seed_totals = _block_totals(sums, block_days, plan)
        numerator = np.take_along_axis(seed_totals, seed_draws, axis=1).mean(1)
    sampled = np.divide(numerator, boot_counts, out=np.full_like(numerator, np.nan), where=boot_counts > 0)
    # For a ratio estimator, subtracting theta*count in every daily sum is
    # exactly equivalent to sampled_ratio-theta. This is an H0-centered null.
    return sampled, sampled - estimate, boot_counts


def _inference_row(item, meta, estimate, scale, method, block_days=None, lag_days=None, estimand="fixed_seed_mean"):
    return dict(comparison=item["name"], family=item["family"], kind=item["kind"],
        method=method, estimand=estimand, block_days=block_days, hac_lag_days=lag_days,
        estimate=estimate * scale if np.isfinite(estimate) else np.nan,
        standard_error=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=1.,
        available=False, reason=meta["reason"], n_bootstrap_valid=0,
        annualization_factor=scale, **{k: meta[k] for k in (
            "n_market_observations", "n_observed_days", "n_calendar_days", "n_zero_count_days", "n_seeds")})


def _correct(frame, group_columns):
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["p_holm_family"] = 1.
    frame["p_holm_global"] = 1.
    frame["family_size"] = 0
    frame["global_size"] = 0
    for _, selection in frame.groupby(group_columns, dropna=False, sort=False).groups.items():
        selection = list(selection)
        frame.loc[selection, "p_holm_global"] = holm_adjust(frame.loc[selection, "p_value"].to_numpy())
        frame.loc[selection, "global_size"] = len(selection)
        for _, family_ids in frame.loc[selection].groupby("family", sort=False).groups.items():
            family_ids = list(family_ids)
            frame.loc[family_ids, "p_holm_family"] = holm_adjust(frame.loc[family_ids, "p_value"].to_numpy())
            frame.loc[family_ids, "family_size"] = len(family_ids)
    frame["significant_family"] = frame.available & (frame.p_holm_family < .05)
    frame["significant_global"] = frame.available & (frame.p_holm_global < .05)
    return frame


def infer_comparisons(comparisons, cfg):
    """Run predeclared, paired market-calendar comparisons after the Test gate.

    ``values[:,s]`` is already the within-seed paired contrast. Valid cohorts
    must be identical across seeds. Nonfinite observed paths invalidate the
    entire contrast and remain in all Holm families at p=1. No bad bars, dead
    seeds, hypotheses, or sparse zero-count calendar days are silently dropped.
    """
    settings = cfg.get("statistics", cfg)
    replicates = int(settings.get("bootstrap_replicates", 10000))
    primary_block = int(settings.get("primary_block_days", 7))
    blocks = list(dict.fromkeys([primary_block, *settings.get("sensitivity_block_days", [14, 30])]))
    lags = list(dict.fromkeys(settings.get("hac_lags", [6, 13, 29])))
    seed = int(settings.get("seed", 20261112))
    minimum_days = int(settings.get("min_observed_days", 14))
    minimum_observations = int(settings.get("min_observations", 30))
    minimum_fraction = float(settings.get("minimum_bootstrap_fraction", .95))
    annual_bars = float(cfg.get("accounting", {}).get("annual_bars", 35040.))
    if replicates < 100 or any(int(b) < 1 for b in blocks) or any(int(lag) < 0 for lag in lags):
        raise ValueError("positive block sizes, nonnegative HAC lags, at least100 replicates required")
    if not 0 < minimum_fraction <= 1 or minimum_days < 2 or minimum_observations < 2:
        raise ValueError("invalid inference adequacy thresholds")
    items = [_validate(name, comparisons[name]) for name in sorted(comparisons)]
    observed_times = [item["timestamps"] for item in items if len(item["timestamps"])]
    first = min(int(t[0] // DAY_NS) for t in observed_times) if observed_times else 0
    last = max(int(t[-1] // DAY_NS) for t in observed_times) if observed_times else -1
    days = last - first + 1
    plans = {b: _sampling_plan(days, int(b), replicates, seed) for b in blocks}
    seed_draws = {}
    rows, hac_rows, details = [], [], {}
    critical = float(norm.ppf(.975))
    for item in items:
        sums, counts, estimate, metadata = _daily(item, first, days, minimum_days, minimum_observations)
        details[item["name"]] = metadata
        seeds = item["values"].shape[1]
        if seeds not in seed_draws:
            seed_draws[seeds] = np.random.default_rng(np.random.SeedSequence([seed, seeds, 1907])).integers(0, seeds, (replicates, seeds))
        scale = annual_bars if item["kind"] == "policy" else 1.
        for block in blocks:
            fixed = None
            for estimand in ("fixed_seed_mean", "hierarchical_seed_sensitivity"):
                row = _inference_row(item, metadata, estimate, scale, "circular_day_block_bootstrap", block_days=int(block), estimand=estimand)
                if metadata["reason"] is None:
                    if metadata["n_observed_calendar_span_days"] < 2 * int(block):
                        row["reason"] = "fewer_than_two_blocks_in_observed_calendar_span"
                        rows.append(row)
                        continue
                    if estimand == "fixed_seed_mean":
                        fixed = _bootstrap(sums, counts, estimate, int(block), plans[block])
                        sampled, centered, boot_counts = fixed
                    elif seeds == 1:
                        sampled, centered, boot_counts = fixed
                    else:
                        sampled, centered, boot_counts = _bootstrap(sums, counts, estimate, int(block), plans[block], seed_draws[seeds])
                    finite = np.isfinite(centered)
                    valid_n = int(finite.sum())
                    row["n_bootstrap_valid"] = valid_n
                    row["n_zero_count_bootstrap"] = int(np.count_nonzero(boot_counts == 0))
                    if valid_n >= minimum_fraction * replicates:
                        quantiles = np.quantile(sampled[finite], [.025, .975])
                        row.update(available=True, reason=None,
                            p_value=float((1 + np.count_nonzero(np.abs(centered[finite]) >= abs(estimate))) / (valid_n + 1)),
                            ci_low=float((2 * estimate - quantiles[1]) * scale),
                            ci_high=float((2 * estimate - quantiles[0]) * scale),
                            standard_error=float(np.std(sampled[finite], ddof=1) * scale),
                            null_centered_mean=float(centered[finite].mean() * scale))
                    else:
                        row["reason"] = "too_many_zero_count_bootstrap_replicates"
                rows.append(row)
        for lag in lags:
            row = _inference_row(item, metadata, estimate, scale, "daily_ratio_Newey_West_HAC", lag_days=int(lag))
            if metadata["reason"] is None and metadata["n_observed_calendar_span_days"] > int(lag) + 1:
                influence = sums.mean(1) - estimate * counts
                covariance = float(influence @ influence / days)
                for shift in range(1, int(lag) + 1):
                    covariance += 2 * (1 - shift / (int(lag) + 1)) * float(influence[shift:] @ influence[:-shift] / days)
                variance = covariance / (days * float(counts.mean()) ** 2)
                if np.isfinite(variance) and variance > 0:
                    se = float(np.sqrt(variance))
                    row.update(available=True, reason=None, standard_error=se * scale,
                        ci_low=(estimate - critical * se) * scale, ci_high=(estimate + critical * se) * scale,
                        p_value=float(2 * norm.sf(abs(estimate) / se)))
                else:
                    row["reason"] = "degenerate_HAC_variance"
            elif metadata["reason"] is None:
                row["reason"] = "insufficient_calendar_length_for_HAC_lag"
            hac_rows.append(row)
    inference = _correct(pd.DataFrame(rows), ["method", "block_days", "estimand"])
    hac = _correct(pd.DataFrame(hac_rows), ["method", "hac_lag_days", "estimand"])
    primary = inference.loc[(inference.block_days == primary_block) & (inference.estimand == "fixed_seed_mean")].copy() if len(inference) else inference.copy()
    family_sizes = dict(Counter(item["family"] for item in items))
    report = dict(n_comparisons=len(items), comparison_names=[item["name"] for item in items],
        family_sizes=family_sizes, global_family_size=len(items), n_global_calendar_days=days,
        first_calendar_day_ns=first * DAY_NS if days else None,
        last_calendar_day_ns=last * DAY_NS if days else None,
        primary_block_days=primary_block, sensitivity_block_days=[int(b) for b in blocks if b != primary_block],
        bootstrap_replicates=replicates, random_seed=seed,
        minimum_resolvable_bootstrap_p=1. / (replicates + 1),
        smallest_global_Holm_at_minimum_p=min(1., len(items) / (replicates + 1)),
        shared_sampling_plan_sha256={str(b): plans[b]["sha256"] for b in blocks},
        shared_seed_sampling_sha256={str(s): _array_sha(v) for s, v in seed_draws.items()},
        null="H0 mean paired difference=0; common observed grand mean times daily count removed before ratio bootstrap",
        ci="95% basic bootstrap [2*theta-q97.5,2*theta-q2.5]",
        estimator="sum daily paired-seed-mean differences / sum daily observation counts",
        sparse_days="all UTC calendar gaps and zero-count days retained; zero-count draws marked unavailable",
        seed_independence="fixed seed mean primary; hierarchical seed resampling sensitivity; no multiplication of market observations",
        HAC="daily ratio influence=sumdiff-theta*count; Bartlett Newey-West daily lags; asymptotic normal",
        multiplicity="Holm within predeclared family and all comparisons, separately per prespecified sensitivity stratum",
        invalid="nonfinite observed, constant mean, insufficient observations => p1, CI unavailable, family retained",
        thresholds=dict(min_observed_days=minimum_days, min_observations=minimum_observations,
                        minimum_bootstrap_fraction=minimum_fraction), comparisons=details)
    # JSON encoding here verifies that provenance remains machine-readable.
    json.dumps(report, allow_nan=False)
    return dict(inferenceDF=inference, hacDF=hac, primaryDF=primary, metadata=report)
