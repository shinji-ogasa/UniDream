"""Descriptive paired loss uncertainty on fixed validation-quarter paths.

The caller supplies losses on each *complete scheduled decision grid*, retaining
unavailable slots as NaN. This module does not load outcomes, train models, choose
features, or perform a hypothesis test. Resampling is conditional on the supplied
quarters and fitted forecasts. It does not include selection, retraining,
cross-quarter dependence, or uncertainty about future regimes.
"""

from collections.abc import Mapping, Sequence
from operator import index

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _moving_block_indices(n_rows, block_length, n_bootstrap, rng):
    """Sample contiguous original blocks; never wrap from quarter end to start."""
    if not 1 <= block_length <= n_rows:
        raise ValueError("block_length must be between 1 and the grid length")
    n_blocks = (n_rows + block_length - 1) // block_length
    starts = rng.integers(0, n_rows - block_length + 1,
                          size=(n_bootstrap, n_blocks))
    sampled = starts[:, :, None] + np.arange(block_length)
    return sampled.reshape(n_bootstrap, -1)[:, :n_rows]


def paired_quarter_block_bootstrap(
    fold_losses: Mapping[str, Mapping[str, np.ndarray]],
    *,
    comparisons: Mapping[str, tuple[str, str]],
    block_lengths: Sequence[int] = (4, 28, 112),
    primary_block_length: int = 28,
    n_bootstrap: int = 2000,
    seed: int = 20260905,
    confidence: float = 0.95,
) -> dict:
    """Return equal-quarter paired contrasts and centered bootstrap intervals.

    ``comparisons[name] = (candidate_loss_name, reference_loss_name)`` defines
    candidate minus reference loss: negative values favor the candidate. Each
    contrast uses only slots where both losses are finite; NaNs remain on the
    original grid during resampling. All comparisons share sampled indices.
    Quarter lengths may differ, but named losses within a quarter must align.
    The caller must validate timestamps and prohibit compressed or reordered
    grids; arrays alone cannot establish calendar alignment.

    A non-circular moving-block sample has the original quarter's grid length;
    its last sampled block is truncated if needed. Quarters are independently
    resampled and receive equal weight, regardless of paired observation count.
    A quarter with no original paired observations is rejected. If a resampled
    quarter has no paired observations, that entire replicate of that contrast
    is excluded and counted. Quarters are never dropped or reweighted; invalid
    replicates are never retried. Intervals require two complete replicates.

    Let D be the observed equal-quarter mean loss difference and D* the bootstrap
    replicate. The interval is [D-q_high(e*), D-q_low(e*)], where
    e* = D* - mean(D*). Centering removes finite-sample edge reweighting from the
    location of this descriptive interval; mean(D*)-D is reported separately.
    Weak within-quarter stationarity/dependence and sufficient block information
    are working assumptions, not validated by this function. Fixed block lengths
    give a sensitivity diagnostic, not exact coverage or adjusted p-values.
    """
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 2)
    seed = _integer(seed, "seed", 0)
    primary_block_length = _integer(primary_block_length, "primary_block_length", 1)
    lengths = tuple(_integer(x, "block_length", 1) for x in block_lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("block_lengths must be nonempty and unique")
    if primary_block_length not in lengths:
        raise ValueError("primary_block_length must be in block_lengths")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    if not isinstance(fold_losses, Mapping) or not fold_losses:
        raise ValueError("fold_losses must be a nonempty mapping")
    if not isinstance(comparisons, Mapping) or not comparisons:
        raise ValueError("comparisons must be a nonempty mapping")
    if any(not isinstance(k, str) for k in fold_losses):
        raise ValueError("fold names must be strings")
    if any(not isinstance(k, str) for k in comparisons):
        raise ValueError("comparison names must be strings")
    names = sorted(comparisons)
    pairs = []
    for name in names:
        pair = comparisons[name]
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(k, str) for k in pair)):
            raise ValueError(f"invalid candidate/reference pair for {name}")
        pairs.append(tuple(pair))
    required = sorted({key for pair in pairs for key in pair})
    folds = []
    observed = {name: {"candidate": pair[0], "reference": pair[1], "folds": {}}
                for name, pair in zip(names, pairs)}
    for fold_name in sorted(fold_losses):
        losses = fold_losses[fold_name]
        if not isinstance(losses, Mapping):
            raise ValueError(f"losses for {fold_name} must be a mapping")
        arrays = {}
        for key in required:
            if key not in losses:
                raise ValueError(f"missing loss {key} in {fold_name}")
            values = np.asarray(losses[key], dtype=np.float64)
            if values.ndim != 1 or not len(values) or np.isinf(values).any():
                raise ValueError(f"loss {key} in {fold_name} must be 1d, nonempty, finite or NaN")
            arrays[key] = values
        n_rows = len(arrays[required[0]])
        if any(len(a) != n_rows for a in arrays.values()):
            raise ValueError(f"loss arrays do not align in {fold_name}")
        if max(lengths) > n_rows:
            raise ValueError(f"block length exceeds scheduled grid in {fold_name}")
        differences = np.zeros((n_rows, len(names)), dtype=np.float64)
        paired = np.zeros((n_rows, len(names)), dtype=bool)
        for column, (name, (candidate, reference)) in enumerate(zip(names, pairs)):
            valid = np.isfinite(arrays[candidate]) & np.isfinite(arrays[reference])
            if not valid.any():
                raise ValueError(f"no paired observations for {name} in {fold_name}")
            with np.errstate(over="raise", invalid="raise"):
                differences[valid, column] = arrays[candidate][valid] - arrays[reference][valid]
            paired[:, column] = valid
            observed[name]["folds"][fold_name] = {
                "grid_rows": n_rows,
                "paired_rows": int(valid.sum()),
                "paired_fraction": float(valid.mean()),
                "candidate_mean_loss": float(np.mean(arrays[candidate][valid])),
                "reference_mean_loss": float(np.mean(arrays[reference][valid])),
                "mean_difference": float(np.mean(differences[valid, column])),
            }
        folds.append((fold_name, differences, paired))
    estimates = np.array([
        np.mean([f["mean_difference"] for f in observed[name]["folds"].values()])
        for name in names
    ])
    if not np.isfinite(estimates).all():
        raise ValueError("nonfinite observed loss means")
    for column, name in enumerate(names):
        observed[name]["equal_quarter_mean_difference"] = float(estimates[column])
    alpha = 1 - confidence
    block_results = {}
    for length in lengths:
        # A block length's stream does not depend on which sensitivities are run.
        rng = np.random.default_rng(np.random.SeedSequence([seed, length]))
        draws = np.zeros((n_bootstrap, len(names)), dtype=np.float64)
        complete = np.ones((n_bootstrap, len(names)), dtype=bool)
        fold_grid = {}
        for fold_name, differences, paired in folds:
            n_rows = len(differences)
            indices = _moving_block_indices(n_rows, length, n_bootstrap, rng)
            counts = paired[indices].sum(axis=1)
            has_pair = counts > 0
            complete &= has_pair
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                means = np.divide(differences[indices].sum(axis=1), counts,
                                  out=np.zeros_like(draws), where=has_pair)
                draws += means / len(folds)
            fold_grid[fold_name] = {
                "grid_rows": n_rows,
                "sampled_blocks_per_replicate": (n_rows + length - 1) // length,
                "grid_length_over_block_length": n_rows / length,
                "all_missing_replicates": {
                    name: int((~has_pair[:, column]).sum())
                    for column, name in enumerate(names)
                },
            }
        if not np.isfinite(draws).all():
            raise ValueError("nonfinite bootstrap contrasts")
        per_comparison = {}
        for column, name in enumerate(names):
            values = draws[complete[:, column], column]
            available = len(values) >= 2
            if available:
                draw_mean = np.mean(values)
                lo, hi = np.quantile(values - draw_mean, [alpha / 2, 1 - alpha / 2])
            per_comparison[name] = {
                "equal_quarter_mean_difference": float(estimates[column]),
                "bootstrap_standard_error": float(np.std(values, ddof=1)) if available else None,
                "bootstrap_mean_minus_observed": float(draw_mean - estimates[column]) if available else None,
                "centered_interval": [float(estimates[column] - hi),
                                      float(estimates[column] - lo)] if available else None,
                "bootstrap_replicates": n_bootstrap,
                "valid_bootstrap_replicates": len(values),
                "invalid_bootstrap_replicates": n_bootstrap - len(values),
                "interval_available": available,
            }
        block_results[str(length)] = {"primary": length == primary_block_length,
                                      "fold_grids": fold_grid,
                                      "comparisons": per_comparison}
    return {
        "method": "quarter_stratified_non_circular_moving_block_bootstrap",
        "scope": "descriptive_conditional_on_observed_quarters_and_fitted_forecasts",
        "difference": "candidate_loss_minus_reference_loss; negative favors candidate",
        "aggregation": "equal_quarter_mean_of_paired_finite_loss_differences",
        "interval": "observed_minus_quantiles_of_bootstrap_mean_centered_errors",
        "invalid_replicates": "count_and_exclude_entire_contrast_replicate_if_any_quarter_has_no_pair; no_retry",
        "confidence": confidence,
        "primary_block_length": primary_block_length,
        "block_lengths": list(lengths),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "fold_count": len(folds),
        "selection_adjusted": False,
        "includes_retraining_uncertainty": False,
        "includes_cross_quarter_dependence": False,
        "includes_future_regime_uncertainty": False,
        "observed": observed,
        "blocks": block_results,
    }
