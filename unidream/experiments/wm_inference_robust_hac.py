"""Segment-aware calendar-day HAC and explicitly prospective precision planning.

Pure array functions: no files, market data, model fits, or Test access occur.
The primary estimand averages a FIXED set of trained seeds on the same market
bars. More seeds never multiply the number of observed market bars. HAC uses
normal asymptotics; segment resets and a finite-sample multiplier do not make
that approximation exact or remove feature selection / nonstationarity bias.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


DEFAULT_LAGS = (0, 6, 13, 27, 29)
ACF_LAGS = (1, 2, 3, 7, 14, 30)


def _panel(day_sums, counts, segment_ids, day_ids):
    sums = np.asarray(day_sums, dtype=np.float64)
    if sums.ndim == 1:
        sums = sums[:, None]
    counts = np.asarray(counts, dtype=np.float64)
    segments, days = np.asarray(segment_ids), np.asarray(day_ids)
    n = len(sums)
    if (sums.ndim != 2 or not n or not sums.shape[1] or counts.shape != (n,)
            or segments.shape != (n,) or days.shape != (n,)
            or not np.issubdtype(days.dtype, np.integer)
            or not np.isfinite(sums).all() or not np.isfinite(counts).all()
            or np.any(counts <= 0) or np.any(counts != np.floor(counts))):
        raise ValueError("finite daily sums[D,S], positive integer common counts, segment IDs and integer UTC epoch-days required")
    if np.any(np.diff(days) < 0):
        raise ValueError("daily panel must be chronological; same day may repeat only across different segments")
    seen = set()
    for segment, day in zip(segments.tolist(), days.tolist()):
        if segment is None or (isinstance(segment, float) and not np.isfinite(segment)):
            raise ValueError("finite, nonmissing segment identities required")
        key = (segment, int(day))
        if key in seen:
            raise ValueError("each calendar day must occur at most once within a segment")
        seen.add(key)
    return sums, counts, segments, days.astype(np.int64, copy=False)


def _pairs(segments, days, lag):
    if isinstance(lag, (bool, np.bool_)) or int(lag) != lag or lag < 0:
        raise ValueError("HAC and ACF lag must be a nonnegative integer number of calendar days")
    lookup = {(segment, int(day)): row for row, (segment, day) in enumerate(zip(segments.tolist(), days.tolist()))}
    right, left = [], []
    for row, (segment, day) in enumerate(zip(segments.tolist(), days.tolist())):
        previous = lookup.get((segment, int(day) - int(lag)))
        if previous is not None:
            right.append(row)
            left.append(previous)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _variance(scores, counts, segments, days, lag, annual, finite_sample):
    n = len(scores)
    diagonal = float(scores @ scores)
    gamma = diagonal
    pairs = []
    for k in range(1, int(lag) + 1):
        left, right = _pairs(segments, days, k)
        cross = float(scores[left] @ scores[right])
        weight = 1. - k / (int(lag) + 1.)
        gamma += 2. * weight * cross
        pairs.append({"lag_days": k, "n_pairs": len(left), "bartlett_weight": weight, "score_cross_sum": cross})
    tolerance = 128 * np.finfo(float).eps * max(diagonal, np.finfo(float).tiny) * (int(lag) + 1)
    if gamma < -tolerance:
        raise ValueError("materially negative Bartlett HAC score variance; invalid numerical covariance")
    gamma = max(0., gamma)
    correction = n / (n - 1.) if finite_sample and n > 1 else 1.
    variance = float(annual ** 2 * gamma * correction / counts.sum() ** 2)
    return variance, gamma, correction, pairs


def _normal_inference(estimate, variance, confidence):
    if not np.isfinite(variance) or variance <= 0:
        return {"estimate": float(estimate), "standard_error": None, "ci_low": None, "ci_high": None,
            "p_value": None, "z_value": None, "inference_available": False,
            "inference_unavailable_reason": "nonpositive_or_nonfinite_HAC_variance"}
    se = float(np.sqrt(variance))
    z = float(estimate / se)
    critical = float(norm.ppf((1 + confidence) / 2))
    return {"estimate": float(estimate), "standard_error": se, "ci_low": float(estimate - critical * se),
        "ci_high": float(estimate + critical * se), "p_value": float(2 * norm.sf(abs(z))),
        "z_value": z, "inference_available": True, "inference_unavailable_reason": None}


def hac_mean(day_sums, counts, segment_ids, day_ids, *, lag=6, annual_bars=35040.,
             finite_sample=True, confidence=.95):
    """HAC for annualized mean per-bar log excess, allowing uneven daily counts.

    For Y_ds = SUM of seed s's net log excess on a segment/day and n_d shared
    eligible bars, mu = sum_d mean_s(Y_ds) / sum_d n_d. Its daily intercept
    score is u_d = mean_s(Y_ds) - mu*n_d. We use:

      Var(annual*mu) = annual^2 / N^2 * D/(D-1) *
        [sum u_d^2 + 2 sum_k (1-k/(L+1)) sum_(same segment, day gap=k) u_d*u_e].

    D/(D-1) is the optional conventional one-intercept correction, not a
    degrees-of-freedom claim for independent days. Missing calendar days and
    fold/market segment boundaries never create spurious adjacent lag pairs.
    """
    sums, counts, segments, days = _panel(day_sums, counts, segment_ids, day_ids)
    if (not np.isfinite(annual_bars) or annual_bars <= 0 or not 0 < confidence < 1
            or isinstance(lag, (bool, np.bool_)) or int(lag) != lag or lag < 0):
        raise ValueError("positive annual bars, confidence in(0,1), and nonnegative integer lag required")
    annual = float(annual_bars)
    mean_sums = sums.mean(axis=1)
    mu = float(mean_sums.sum() / counts.sum())
    scores = mean_sums - mu * counts
    variance, gamma, correction, pairs = _variance(scores, counts, segments, days, lag, annual, finite_sample)
    result = _normal_inference(annual * mu, variance, confidence)
    per_seed = []
    for seed in range(sums.shape[1]):
        seed_mu = float(sums[:, seed].sum() / counts.sum())
        seed_variance, _, _, _ = _variance(sums[:, seed] - seed_mu * counts, counts, segments, days, lag, annual, finite_sample)
        per_seed.append({"seed_index": seed, **_normal_inference(annual * seed_mu, seed_variance, confidence)})
    seed_means = np.asarray([row["estimate"] for row in per_seed], dtype=np.float64)
    seed_floor = float(seed_means.var(ddof=1) / len(seed_means)) if len(seed_means) > 1 else None
    # This additive approximation is a sensitivity, not the primary sampling
    # scheme and not proof of conservative coverage. It must not be shrunk by
    # future market years when the number of training seeds remains fixed.
    approximate = _normal_inference(annual * mu, variance + seed_floor, confidence) if seed_floor is not None else None
    if approximate is not None:
        approximate.update({"approximate": True, "primary": False,
            "seed_variance_floor": seed_floor,
            "assumptions": "add Var_s(annualized seed mean)/S to market HAC; covariance/separate seed population assumptions unverified; no3N independent market observations"})
    result.update({"lag_days": int(lag), "annual_bars": annual, "confidence_level": float(confidence),
        "alternative": "two_sided", "estimand": "fixed_training_seed_average_annualized_per_bar_log_excess",
        "n_daily_rows": len(sums), "n_unique_calendar_days": len(np.unique(days)), "n_market_bars": int(counts.sum()),
        "n_seeds": sums.shape[1], "n_segments": len(set(segments.tolist())),
        "eligible_years": float(counts.sum() / annual), "time_scope": "eligible_observed_bar_equivalent_years_not_calendar_years",
        "market_variance": variance, "score_long_run_sum": gamma, "score_diagonal_sum": float(scores @ scores),
        "finite_sample_factor": correction, "finite_sample_correction": "D/(D-1),one intercept" if finite_sample else "none",
        "lag_pairs": pairs, "per_seed": per_seed, "approximate_seed_variance_floor": seed_floor,
        "approximate_seed_augmented": approximate,
        "normal_asymptotics": True, "stationarity_or_weak_dependence_guaranteed": False})
    return result


def normal_power(effect, standard_error, alpha=.05):
    """Exact two-sided noncentral-normal power, conditional on a known SE."""
    if not np.isfinite(effect) or not np.isfinite(standard_error) or standard_error <= 0 or not 0 < alpha < 1:
        raise ValueError("finite effect, positive SE, and alpha in(0,1) required")
    delta, critical = abs(float(effect)) / float(standard_error), float(norm.ppf(1-alpha/2))
    return float(norm.sf(critical-delta) + norm.cdf(-critical-delta))


def _required_signal(power, alpha):
    if not 0 < alpha < power < 1:
        raise ValueError("planned target power must exceed alpha and lie below one")
    return float(brentq(lambda delta: normal_power(delta, 1., alpha) - power, 0., 32., xtol=1e-12))


def power_scenarios(result, *, effects=(.02, .05, .10), years=(1, 2, 3, 5, 10),
                    alphas=(.05, .05/8), powers=(.8, .9), include_observed=True,
                    include_seed_floor=True):
    """Prospective precision under unchanged stationary long-run variance.

    The plug-in LRV is uncertain and can change with market regime. Projected
    years mean eligible observed-bar exposure, not calendar acquisition time.
    The observed-effect scenario was selected/estimated from the same outcomes:
    its power is NOT evidence against H0 or a post-hoc power proof.
    """
    if (not result.get("inference_available") or not np.isfinite(result["market_variance"])
            or result["market_variance"] <= 0 or result["eligible_years"] <= 0):
        return {"available": False, "reason": "positive_available_market_HAC_variance_required", "power": [], "mde": [], "required_years": []}
    if (not effects or any(not np.isfinite(v) or v <= 0 for v in effects)
            or not years or any(not np.isfinite(v) or v <= 0 for v in years)):
        raise ValueError("positive finite prospective effects and eligible-year horizons required")
    variance_time = float(result["market_variance"] * result["eligible_years"])
    scenarios = [(float(effect), "prespecified_effect") for effect in effects]
    if include_observed:
        scenarios.append((abs(float(result["estimate"])), "observed_selected_effect_scenario_not_power_proof"))
    modes = [("market_only_fixed_seed_average", 0.)]
    floor = result.get("approximate_seed_variance_floor")
    if include_seed_floor and floor is not None and np.isfinite(floor) and floor >= 0:
        modes.append(("approximate_fixed_training_seed_variance_floor_sensitivity", float(floor)))
    power_rows, mde_rows, required_rows = [], [], []
    for mode, seed_floor in modes:
        for alpha in alphas:
            if not 0 < alpha < 1:
                raise ValueError("planning alpha must lie in(0,1)")
            label = "unadjusted_two_sided_alpha05" if np.isclose(alpha, .05) else "Bonferroni8_conservative_Holm_planning" if np.isclose(alpha, .05/8) else "custom_two_sided_alpha"
            for year in years:
                se = float(np.sqrt(variance_time / year + seed_floor))
                for effect, effect_label in scenarios:
                    power_rows.append({"mode": mode, "alpha": float(alpha), "alpha_interpretation": label,
                        "eligible_years": float(year), "annual_log_growth_effect": effect, "effect_scenario": effect_label,
                        "prospective_power": normal_power(effect, se, alpha), "projected_se": se,
                        "observed_scenario_is_evidence": False})
                for target_power in powers:
                    signal = _required_signal(target_power, alpha)
                    mde_rows.append({"mode": mode, "alpha": float(alpha), "alpha_interpretation": label,
                        "eligible_years": float(year), "target_power": float(target_power),
                        "mde_annual_log_growth": signal * se, "fixed_seed_floor_mde_limit": signal * np.sqrt(seed_floor)})
            for effect, effect_label in scenarios:
                for target_power in powers:
                    signal = _required_signal(target_power, alpha)
                    target_variance = (effect / signal) ** 2
                    unattainable = target_variance <= seed_floor
                    required = None if unattainable else float(variance_time / (target_variance-seed_floor))
                    required_rows.append({"mode": mode, "alpha": float(alpha), "alpha_interpretation": label,
                        "annual_log_growth_effect": effect, "effect_scenario": effect_label,
                        "target_power": float(target_power), "required_eligible_years": required,
                        "unattainable_with_fixed_seed_floor": bool(unattainable),
                        "observed_scenario_is_evidence": False})
    return {"available": True, "current_eligible_years": float(result["eligible_years"]),
        "plug_in_variance_times_eligible_year": variance_time,
        "assumptions": "prospective stationary LRV and sqrt(T) scaling; estimates uncertain under regime shifts; no guaranteed future data availability",
        "time_scope": "eligible observed bars/annual_bars,not calendar elapsed years",
        "observed_effect_caveat": "same-data selected observed effect may be optimistic; observed power is not a test of H0",
        "seed_floor_caveat": "fixed approximate seed variance is not divided by additional market years; assumes same finite set/number of seeds",
        "power": power_rows, "mde": mde_rows, "required_years": required_rows}


def _acf(series, segments, days, lag):
    values = np.asarray(series, np.float64)
    centered = values - values.mean()
    denominator = float(centered @ centered)
    left, right = _pairs(segments, days, lag)
    value = float(centered[left] @ centered[right] / denominator) if len(left) and denominator > 0 else None
    return {"lag_days": int(lag), "acf": value, "n_pairs": len(left), "n_rows": len(values),
        "defined": value is not None}


def serial_diagnostics(day_sums, counts, segment_ids, day_ids, *, fold_ids=None, lags=ACF_LAGS):
    """Descriptive ACF of daily mean log excess, its square and absolute value.

    ACF is the usual biased sample autocovariance/full centered sum-of-squares
    ratio, except numerator pairs cannot cross segments or missing calendar
    lags. These diagnostics are not additional significance tests.
    """
    sums, counts, segments, days = _panel(day_sums, counts, segment_ids, day_ids)
    daily = sums.mean(1) / counts
    series = {"diff": daily, "squared": daily ** 2, "absolute": np.abs(daily)}
    segment_centered = daily.copy()
    for segment in set(segments.tolist()):
        mask = segments == segment
        segment_centered[mask] -= np.average(daily[mask], weights=counts[mask])
    series.update({"within_segment_diff": segment_centered, "within_segment_squared": segment_centered ** 2,
        "within_segment_absolute": np.abs(segment_centered)})
    summaries = []
    if fold_ids is not None:
        folds = np.asarray(fold_ids)
        if folds.shape != counts.shape:
            raise ValueError("fold_ids must match daily rows")
        fold_centered = daily.copy()
        for fold in sorted(set(folds.tolist())):
            mask = folds == fold
            mu = float(sums[mask].mean(1).sum() / counts[mask].sum())
            fold_centered[mask] -= mu
            summaries.append({"fold": fold.item() if isinstance(fold, np.generic) else fold,
                "n_daily_rows": int(mask.sum()), "n_market_bars": int(counts[mask].sum()),
                "mean_daily_per_bar_log_excess": mu, "daily_per_bar_std": float(daily[mask].std()),
                "mean_absolute_daily_per_bar_log_excess": float(np.abs(daily[mask]).mean())})
        series.update({"within_fold_diff": fold_centered, "within_fold_squared": fold_centered ** 2,
            "within_fold_absolute": np.abs(fold_centered)})
    rows = [{"series": name, **_acf(values, segments, days, lag)}
            for name, values in series.items() for lag in lags]
    return {"acf": rows, "folds": summaries, "n_daily_rows": len(daily), "n_market_bars": int(counts.sum()),
        "definition": "daily sumdiff/count averaged over fixed seeds; demean by series mean for ACF; withinfold/segment means are count-weighted",
        "acf_normalization": "matched-calendar withinsegment centered cross-sum divided by full centered sum-of-squares; unadjusted biased ACF",
        "inferential_claim": False}
