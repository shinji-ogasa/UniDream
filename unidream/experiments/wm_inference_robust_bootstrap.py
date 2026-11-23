"""Registered robustness kernels for conditional daily policy-minus-B&H sums.

No fitting, model selection or input-file access occurs here. The caller must
freeze the post-hoc protocol before supplying real outcomes. Circular and
stationary sampling wrap *inside* an observed segment, introducing a synthetic
end-to-start seam; neither method bridges an actual gap or fold boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .wm_significance_statistics import holm_fixed_family


METHODS = ("legacy_non_circular", "circular", "stationary")


def _validate(day_sums, counts, segment_ids, day_ids):
    y = np.asarray(day_sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    segments = np.asarray(segment_ids)
    if (y.ndim != 3 or min(y.shape) < 1 or counts.shape != (len(y),)
            or segments.shape != counts.shape or not np.isfinite(y).all()
            or not np.isfinite(counts).all() or np.any(counts < 1)
            or np.any(counts > 96) or np.any(counts != np.floor(counts))):
        raise ValueError("finite daily sums[T,K,S], unique-market integer counts[1,96] and segment IDs required")
    if pd.isna(segments).any():
        raise ValueError("finite nonmissing segment IDs required")
    groups = []
    for segment in dict.fromkeys(segments.tolist()):
        ids = np.flatnonzero(segments == segment)
        if np.any(np.diff(ids) != 1):
            raise ValueError("each segment must occupy one contiguous run of daily units")
        groups.append(ids)
    if day_ids is not None:
        dates = np.asarray(day_ids)
        if dates.shape != counts.shape or not np.isfinite(dates).all() or np.any(dates != np.floor(dates)):
            raise ValueError("day_ids must be finite integer UTC epoch days")
        if any(np.any(np.diff(dates[ids]) != 1) for ids in groups):
            raise ValueError("daily units inside a segment must have consecutive UTC days")
    return y, counts, groups


def expected_inclusion_weights(length, block_days, method="legacy_non_circular"):
    """Exact expected multiplicity, including the truncated final block.

    Circular/stationary uniform starts give multiplicity one. A segment with
    <=L days is replayed fixed by all methods, never shortened or IID sampled.
    """
    if method not in METHODS or int(length) != length or int(block_days) != block_days or min(length, block_days) < 1:
        raise ValueError("positive integer lengths and a supported bootstrap method required")
    n, block = int(length), int(block_days)
    if n <= block or method != "legacy_non_circular":
        return np.ones(n)
    full, remainder = divmod(n, block)
    starts = np.arange(n - block + 1)
    diff = np.zeros(n + 1)
    np.add.at(diff, starts, full / len(starts))
    np.add.at(diff, starts + block, -full / len(starts))
    if remainder:
        np.add.at(diff, starts, 1 / len(starts))
        np.add.at(diff, starts + remainder, -1 / len(starts))
    return np.cumsum(diff[:-1])


def _weights(n, block, replicates, rng, method):
    if n <= block:
        return np.ones((replicates, n))
    row = np.arange(replicates)
    if method == "stationary":
        result = np.zeros((replicates, n))
        index = rng.integers(0, n, size=replicates)
        for position in range(n):
            if position:
                restart = rng.random(replicates) < 1 / block
                fresh = rng.integers(0, n, size=replicates)
                index = np.where(restart, fresh, (index + 1) % n)
            result[row, index] += 1
        return result
    full, remainder = divmod(n, block)
    pieces = full + bool(remainder)
    # Same random-number matrix and order as the frozen noncircular helper.
    starts = rng.integers(0, n - block + 1 if method == "legacy_non_circular" else n,
                          size=(replicates, pieces))
    diff = np.zeros((replicates, n + 1))
    for piece in range(pieces):
        size = block if piece < full else remainder
        start = starts[:, piece]
        end = start + size
        diff[row, start] += 1
        if method == "legacy_non_circular":
            diff[row, end] -= 1
        else:
            wraps = end > n
            diff[row, np.minimum(end, n)] -= 1
            diff[row[wraps], 0] += 1
            diff[row[wraps], end[wraps] - n] -= 1
    return np.cumsum(diff[:, :-1], axis=1)


def bootstrap_distributions(day_sums, counts, segment_ids, *, method="circular", block_days=7,
                            replicates=10000, random_seed=20261112, day_ids=None):
    """Return unannualized paired bootstrap means and explicitly H0 residuals.

    Day sums are [UTC-day units, conditions, training seeds]. All conditions and
    seeds receive the SAME day multiplicities and sampled count denominator.
    Hierarchical draws additionally resample the SAME seed IDs across all
    conditions. H0 subtraction uses one overall mean per condition (not one
    per seed), preserving observed between-seed dispersion in that sensitivity.
    """
    y, counts, groups = _validate(day_sums, counts, segment_ids, day_ids)
    if (method not in METHODS or int(block_days) != block_days or block_days < 1
            or int(replicates) != replicates or replicates < 2
            or int(random_seed) != random_seed or random_seed < 0):
        raise ValueError("valid method, block length, replicates and random seed required")
    block, reps = int(block_days), int(replicates)
    rng = np.random.default_rng(int(random_seed))
    n, k, s = y.shape
    point = y.sum(axis=0).mean(axis=1) / counts.sum()
    residuals = y - counts[:, None, None] * point[None, :, None]
    raw_sums = np.zeros((reps, k * s))
    null_sums = np.zeros_like(raw_sums)
    sampled_count = np.zeros(reps)
    expected_weights = np.ones(n)
    supported = fixed = varying = 0
    for ids in groups:
        weights = _weights(len(ids), block, reps, rng, method)
        # Explicit reductions avoid platform BLAS floating-exception side
        # effects and do not create another independent set of market draws.
        raw_sums += np.einsum("bt,tc->bc", weights, y[ids].reshape(len(ids), -1))
        null_sums += np.einsum("bt,tc->bc", weights, residuals[ids].reshape(len(ids), -1))
        sampled_count += np.einsum("bt,t->b", weights, counts[ids])
        expected_weights[ids] = expected_inclusion_weights(len(ids), block, method)
        if len(ids) > block:
            supported += int(counts[ids].sum())
            varying += 1
        else:
            fixed += int(counts[ids].sum())
    raw = raw_sums.reshape(reps, k, s) / sampled_count[:, None, None]
    null = null_sums.reshape(reps, k, s) / sampled_count[:, None, None]
    chosen = rng.integers(0, s, size=(reps, s))
    hierarchical = np.take_along_axis(raw, chosen[:, None, :], axis=2).mean(axis=2)
    hierarchical_null = np.take_along_axis(null, chosen[:, None, :], axis=2).mean(axis=2)
    expected_ratio = ((expected_weights[:, None, None] * y).sum(axis=0).mean(axis=1)
                      / np.sum(expected_weights * counts))
    if not (np.isfinite(raw).all() and np.isfinite(null).all() and np.all(sampled_count > 0)):
        raise FloatingPointError("bootstrap sums/counts must remain finite with positive market support")
    return {"point": point, "market_distribution": raw.mean(axis=2),
        "market_null_distribution": null.mean(axis=2),
        "hierarchical_distribution": hierarchical, "hierarchical_null_distribution": hierarchical_null,
        "expected_multiplicity": expected_weights, "sampled_count": sampled_count,
        "ratio_of_expected_sums": expected_ratio,
        "analytic_inclusion_ratio_bias": expected_ratio - point,
        "n_unique_market_rows": int(counts.sum()), "n_daily_units": n, "n_training_seeds": s,
        "n_segments": len(groups), "n_resampleable_segments": varying,
        "n_resampleable_rows": supported, "n_fixed_rows": fixed,
        "fixed_row_fraction": fixed / counts.sum(), "inference_available": bool(varying)}


def _p_value(null, point):
    return (1 + np.count_nonzero(np.abs(null) >= np.abs(point)[None, :], axis=0)) / (len(null) + 1)


def infer_daily(day_sums, counts, segment_ids, cfg=None, *, day_ids=None):
    """Return inference/Reality-Check tables plus explicit design metadata.

    Basic intervals invert centered bootstrap roots. Percentile intervals are
    retained separately; asymmetric roots do not make either interval exactly
    equivalent to an equal-absolute-tail test. Reality Check uses the shared
    least-favorable centered MAX across the supplied candidates only; it cannot
    correct an unrepresented historical feature/model specification search.
    """
    cfg = cfg or {}
    settings = cfg.get("statistics", {})
    y, _, _ = _validate(day_sums, counts, segment_ids, day_ids)
    names = list(cfg.get("condition_names", [f"condition_{i}" for i in range(y.shape[1])]))
    if len(names) != y.shape[1] or len(set(names)) != len(names):
        raise ValueError("one distinct condition name per candidate required")
    blocks = tuple(settings.get("block_days", [1, 7, 14, 28, 30]))
    methods = tuple(settings.get("methods", METHODS))
    if len(set(blocks)) != len(blocks) or 7 not in blocks or len(set(methods)) != len(methods) or "circular" not in methods:
        raise ValueError("unique methods/blocks including circular seven-day primary required")
    reps = int(settings.get("bootstrap_replicates", 10000))
    seed = int(settings.get("seed", 20261112))
    annual = float(cfg.get("accounting", {}).get("annual_bars", 35040))
    if not np.isfinite(annual) or annual <= 0:
        raise ValueError("positive annualization required")
    inference, rc = [], []
    for method in methods:
        for block in blocks:
            method_index = METHODS.index(method) if method in METHODS else -1
            job_seed = int(np.random.SeedSequence([seed, method_index, int(block)]).generate_state(1)[0])
            result = bootstrap_distributions(y, counts, segment_ids, method=method, block_days=block,
                replicates=reps, random_seed=job_seed, day_ids=day_ids)
            point = result["point"] * annual
            support = {name: result[name] for name in ("n_unique_market_rows", "n_daily_units", "n_training_seeds",
                "n_segments", "n_resampleable_segments", "n_resampleable_rows", "n_fixed_rows", "fixed_row_fraction")}
            for estimand, key in (("market_fixed_seeds", "market"), ("hierarchical_seeds", "hierarchical")):
                distribution = result[key + "_distribution"] * annual
                null = result[key + "_null_distribution"] * annual
                p = _p_value(null, point)
                corrected_p = _p_value(null - null.mean(axis=0), point)
                primary = method == "circular" and block == 7 and estimand == "market_fixed_seeds"
                available = result["inference_available"]
                q_null = np.quantile(null, [.025, .975], axis=0)
                q_raw = np.quantile(distribution, [.025, .975], axis=0)
                for i, name in enumerate(names):
                    inference.append({"condition": name, "method": method, "block_days": int(block), "estimand": estimand,
                        "primary_inference": primary, "replicates": reps, "metric": "annualized_net_log_growth_difference",
                        "estimate": float(point[i]), "ci_low": float(point[i] - q_null[1, i]) if available else None,
                        "ci_high": float(point[i] - q_null[0, i]) if available else None,
                        "ci_method": "basic inversion of actual H0-centered roots",
                        "percentile_ci_low": float(q_raw[0, i]) if available else None,
                        "percentile_ci_high": float(q_raw[1, i]) if available else None,
                        "p_value": float(p[i]) if available else None,
                        "p_expectation_centered_diagnostic": float(corrected_p[i]) if available else None,
                        "bootstrap_mean_minus_point": float(distribution[:, i].mean() - point[i]),
                        "null_bootstrap_mean": float(null[:, i].mean()),
                        "null_mean_monte_carlo_se": float(null[:, i].std(ddof=1) / np.sqrt(reps)),
                        "analytic_inclusion_ratio_bias": float(result["analytic_inclusion_ratio_bias"][i] * annual),
                        "null_data_vs_subtract_estimate_max_error": float(np.max(np.abs(null[:, i] - (distribution[:, i] - point[i])))),
                        "inference_available": available, **support})
                # Including B&H (zero) makes all-nonpositive observed candidates
                # produce p=1. The common sample preserves cross-strategy dependence.
                observed_max = max(0., float(point.max()))
                null_max = np.maximum(0., null.max(axis=1))
                global_p = float((1 + np.count_nonzero(null_max >= observed_max)) / (reps + 1))
                for i, name in enumerate(names):
                    adjusted = float((1 + np.count_nonzero(null_max >= point[i])) / (reps + 1))
                    rc.append({"condition": name, "method": method, "block_days": int(block), "estimand": estimand,
                        "primary_inference": primary, "replicates": reps, "estimate": float(point[i]),
                        "observed_family_max": observed_max, "family_size": len(names),
                        "global_reality_check_p": global_p if available else None,
                        "max_adjusted_one_sided_p": adjusted if available else None,
                        "one_sided_p": float((1 + np.count_nonzero(null[:, i] >= point[i])) / (reps + 1)) if available else None,
                        "scope": "supplied candidates only; historical feature/model search uncorrected",
                        "inference_available": available, **support})
    primary_rows = [row for row in inference if row["primary_inference"]]
    holm = holm_fixed_family({row["condition"]: 1. if row["p_value"] is None else row["p_value"] for row in primary_rows},
                             family_size=len(names))
    for row in inference:
        row["p_holm_family"] = holm[row["condition"]] if row["primary_inference"] else None
        row["significant_difference"] = bool(row["primary_inference"] and row["p_holm_family"] < .05)
        row["significant_improvement"] = bool(row["significant_difference"] and row["estimate"] > 0)
    return {"inference": pd.DataFrame(inference), "reality_check": pd.DataFrame(rc), "design": {
        "primary": "circular seven-day basic CI and actual H0-centered two-sided p, fixed-seed market uncertainty",
        "short_segments": "<=block_days retained fixed, conditional uncertainty; no shorter-block fallback",
        "circular_seams": "wrap observed end to start within one segment only; synthetic seam disclosed",
        "stationary_blocks": "geometric length mean block_days, circular within segment",
        "seed_estimands": "fixed observed seed mean primary; hierarchical seed-ID resampling sensitivity",
        "centering": "overall strategy mean shared across seeds; explicit residual sums divided by own sampled bar count",
        "expectation_centering": "empirical null-mean recentering diagnostic only, never primary",
        "analytic_bias": "ratio of expected sums, not expectation of a random-denominator ratio",
        "reality_check": "least-favorable max-mean family of supplied paths only; not historical420/F-search correction"}}
