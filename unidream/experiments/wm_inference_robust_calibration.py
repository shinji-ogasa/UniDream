"""Independent, synthetic-only calibration of mean-excess inference.

These simulations preserve the known zero population mean.  They never load
study ledgers, fit policies, choose a block length, or select an inferential
method from the observed trading results.  Monte Carlo intervals describe the
precision of the simulation, not uncertainty about actual trading performance.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCENARIOS = ("iid_normal", "ar1", "garch", "ar1_garch")
METHODS = ("legacy_non_circular", "circular", "hac_lag6", "iid_t")


def _settings(cfg):
    settings = (cfg or {}).get("calibration", {})
    result = {
        "outer_replicates": 300, "inner_replicates": 999, "days": 540,
        "block_days": 7, "seed": 20261112, "burnin": 1000,
        "ar_phi": .3, "garch_alpha": .05, "garch_beta": .90,
        "segment_layouts": ["single"],
    }
    result.update({key: settings[key] for key in result if key in settings})
    for key in ("outer_replicates", "inner_replicates", "days", "block_days", "seed", "burnin"):
        value = result[key]
        minimum = 0 if key in ("seed", "burnin") else 2
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"invalid synthetic calibration {key}")
        result[key] = int(value)
    if result["block_days"] >= result["days"]:
        raise ValueError("calibration requires a resampleable synthetic segment")
    layouts = result["segment_layouts"]
    if (not isinstance(layouts, (list, tuple)) or not layouts
            or len(set(layouts)) != len(layouts)
            or any(layout not in ("single", "six_equal") for layout in layouts)):
        raise ValueError("distinct supported synthetic segment layouts required")
    if "six_equal" in layouts and result["days"] < 6 * (result["block_days"] + 1):
        raise ValueError("six_equal calibration requires six resampleable segments")
    result["segment_layouts"] = list(layouts)
    phi, alpha, beta = map(float, (result["ar_phi"], result["garch_alpha"], result["garch_beta"]))
    if not (abs(phi) < 1 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        raise ValueError("calibration DGP must have finite stationary variance")
    result.update(ar_phi=phi, garch_alpha=alpha, garch_beta=beta)
    return result


def simulate_null(scenario, days, rng, *, burnin=1000, ar_phi=.3,
                  garch_alpha=.05, garch_beta=.90):
    """Known-mean-zero, unconditional-variance-one daily processes.

    AR innovations are scaled by sqrt(1-phi**2).  GARCH's omega is
    1-alpha-beta; innovations are martingale differences.  No sample mean or
    variance is subtracted/divided out, which would invalidate size checks.
    """
    if scenario not in SCENARIOS or days < 2 or burnin < 0:
        raise ValueError("invalid synthetic null scenario or length")
    if not (abs(ar_phi) < 1 and garch_alpha >= 0 and garch_beta >= 0
            and garch_alpha + garch_beta < 1):
        raise ValueError("stationary DGP parameters required")
    draws = rng.standard_normal(int(days + burnin))
    use_ar = scenario in ("ar1", "ar1_garch")
    use_garch = scenario in ("garch", "ar1_garch")
    phi = ar_phi if use_ar else 0.
    scale = math.sqrt(1. - phi * phi)
    omega = 1. - garch_alpha - garch_beta
    variance, previous_innovation, previous_value = 1., 0., 0.
    values = np.empty(len(draws))
    for index, innovation in enumerate(draws):
        if use_garch:
            variance = omega + garch_alpha * previous_innovation ** 2 + garch_beta * variance
        else:
            variance = 1.
        error = math.sqrt(variance) * innovation
        previous_value = phi * previous_value + scale * error
        values[index] = previous_value
        previous_innovation = error
    return values[int(burnin):]


def wilson_interval(successes, trials, *, z=1.959963984540054):
    """Two-sided 95% binomial interval; defined even for zero/all successes."""
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("valid binomial successes and trials required")
    proportion = successes / trials
    denominator = 1. + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z / denominator * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return (0. if successes == 0 else max(0., center - radius),
            1. if successes == trials else min(1., center + radius))


def _iid_result(values):
    n = len(values)
    point = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    if se == 0:
        p = 1. if point == 0 else 0.
        return point, point, p, 0.
    critical = float(student_t.ppf(.975, n - 1))
    p = float(2 * student_t.sf(abs(point / se), n - 1))
    return point - critical * se, point + critical * se, p, 0.


def _bootstrap_result(values, method, settings, seed, *, segment_ids=None):
    from .wm_inference_robust_bootstrap import bootstrap_distributions

    n = len(values)
    result = bootstrap_distributions(
        values[:, None, None], np.ones(n), np.zeros(n, dtype=int) if segment_ids is None else segment_ids,
        day_ids=np.arange(n), method=method, block_days=settings["block_days"],
        replicates=settings["inner_replicates"], random_seed=seed,
    )
    point = float(np.asarray(result["point"]).reshape(-1)[0])
    raw = np.asarray(result["market_distribution"])[:, 0]
    if method == "legacy_non_circular":
        # Exact historical convention: raw percentile interval and centered
        # tail probability.  This comparator is not silently repaired.
        null = raw - point
        low, high = np.quantile(raw, [.025, .975])
    else:
        null = np.asarray(result["market_null_distribution"])[:, 0]
        quantiles = np.quantile(null, [.025, .975])
        low, high = point - quantiles[1], point - quantiles[0]
    if not (np.isfinite(raw).all() and np.isfinite(null).all()):
        raise ValueError("synthetic bootstrap unexpectedly produced undefined draws")
    p = (1 + np.count_nonzero(np.abs(null) >= abs(point))) / (len(null) + 1)
    return float(low), float(high), float(p), float(raw.mean() - point)


def run_calibration(cfg=None):
    """Return 16 rows per segment layout; no trading data are accessed.

    Each scenario uses the same generated realization for all four methods.
    Only one synthetic market series/seed is used: training-seed uncertainty
    and multiple-candidate family control are tested separately by unit tests.
    Legacy intervals use their original percentile convention; new circular
    intervals are basic intervals from the null-residual distribution.
    """
    from .wm_inference_robust_hac import hac_mean

    settings = _settings(cfg)
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        outcomes = {(layout, method): [] for layout in settings["segment_layouts"] for method in METHODS}
        estimates = []
        for outer in range(settings["outer_replicates"]):
            data_seed = np.random.SeedSequence([settings["seed"], scenario_index, outer, 0])
            boot_seed = int(np.random.SeedSequence([settings["seed"], scenario_index, outer, 1]).generate_state(1)[0])
            values = simulate_null(scenario, settings["days"], np.random.default_rng(data_seed),
                burnin=settings["burnin"], ar_phi=settings["ar_phi"],
                garch_alpha=settings["garch_alpha"], garch_beta=settings["garch_beta"])
            estimates.append(float(values.mean()))
            # Exactly the SAME realization is tested under each layout.
            # Six fixed segments reproduce a stratified fold bootstrap, and
            # neither HAC pairs nor sampled blocks cross their boundaries.
            for layout in settings["segment_layouts"]:
                n = len(values)
                segments = np.zeros(n, dtype=int) if layout == "single" else np.arange(n) * 6 // n
                for method in METHODS:
                    if method in ("legacy_non_circular", "circular"):
                        outcome = _bootstrap_result(values, method, settings, boot_seed, segment_ids=segments)
                    elif method == "hac_lag6":
                        result = hac_mean(values[:, None], np.ones(n), segments,
                            np.arange(n), lag=6, annual_bars=1., finite_sample=True)
                        if not result["inference_available"]:
                            raise ValueError("synthetic HAC inference unexpectedly unavailable")
                        outcome = (result["ci_low"], result["ci_high"], result["p_value"], 0.)
                    else:
                        outcome = _iid_result(values)
                    low, high, p, bias = outcome
                    outcomes[(layout, method)].append((p < .05, low <= 0 <= high, high - low, p, bias))
        for (layout, method), outcome_list in outcomes.items():
            results = np.asarray(outcome_list, dtype=float)
            rejects, covered = map(int, results[:, :2].sum(axis=0))
            total = settings["outer_replicates"]
            reject_low, reject_high = wilson_interval(rejects, total)
            cover_low, cover_high = wilson_interval(covered, total)
            rows.append({
                "scenario": scenario, "method": method, "true_mean": 0.,
                "layout": layout, "n_segments": 1 if layout == "single" else 6,
                "theoretical_unconditional_variance": 1., "n_days": settings["days"],
                "outer_replicates": total, "inner_replicates": settings["inner_replicates"] if method in METHODS[:2] else None,
                "block_days": settings["block_days"] if method in METHODS[:2] else None,
                "hac_lag": 6 if method == "hac_lag6" else None,
                "n_market_series": 1, "n_training_seeds": 1,
                "rejection_count": rejects, "empirical_type1_error": rejects / total,
                "type1_mc_ci_low": reject_low, "type1_mc_ci_high": reject_high,
                "coverage_count": covered, "empirical_coverage_95": covered / total,
                "coverage_mc_ci_low": cover_low, "coverage_mc_ci_high": cover_high,
                "mean_interval_width": float(results[:, 2].mean()), "mean_p_value": float(results[:, 3].mean()),
                "mean_bootstrap_mean_minus_point": float(results[:, 4].mean()) if method in METHODS[:2] else None,
                "mean_absolute_bootstrap_mean_minus_point": float(np.abs(results[:, 4]).mean()) if method in METHODS[:2] else None,
                "mean_point_estimate_across_simulations": float(np.mean(estimates)),
                "ci_method": "historical_percentile" if method == "legacy_non_circular" else "null_residual_basic" if method == "circular" else "normal_HAC" if method == "hac_lag6" else "Student_t_IID",
                "nominal_alpha": .05, "monte_carlo_ci_method": "Wilson95",
                "no_sample_demeaning_or_normalization": True,
                "calibration_scope": "synthetic known-null market uncertainty only; not actual-study evidence or family/seed calibration",
                "size_nominal_outside_mc_interval": bool(not reject_low <= .05 <= reject_high),
                "seed": settings["seed"], "burnin": settings["burnin"],
                "ar_phi": settings["ar_phi"] if scenario in ("ar1", "ar1_garch") else 0.,
                "garch_alpha": settings["garch_alpha"] if scenario in ("garch", "ar1_garch") else 0.,
                "garch_beta": settings["garch_beta"] if scenario in ("garch", "ar1_garch") else 0.,
            })
    return pd.DataFrame(rows)


def legacy_inclusion_audit(cfg=None):
    """Exact synthetic proof of noncircular/truncated-block edge weights."""
    settings = (cfg or {}).get("calibration", {})
    n = int(settings.get("inclusion_example_days", 10))
    block = int(settings.get("inclusion_example_block", 7))
    if not 1 <= block < n:
        raise ValueError("edge-weight example requires 1 <= block < days")
    full, remainder = divmod(n, block)
    starts = n - block + 1
    weights = np.zeros(n)
    for start in range(starts):
        weights[start:start + block] += full / starts
        if remainder:
            weights[start:start + remainder] += 1. / starts
    values = np.arange(n, dtype=float)
    return {
        "synthetic_only": True, "n_days": n, "block_days": block,
        "legacy_expected_multiplicity": weights.tolist(), "circular_expected_multiplicity": np.ones(n).tolist(),
        "input_mean": float(values.mean()), "legacy_expected_mean": float(np.dot(weights, values) / n),
        "circular_expected_mean": float(values.mean()), "legacy_expected_mean_minus_point": float(np.dot(weights, values) / n - values.mean()),
        "example_values": values.tolist(), "denominator_fixed_in_example": True,
        "scope": "finite-sample edge weighting of the exact historical noncircular truncate-to-length sampler; no actual ledger results",
    }
