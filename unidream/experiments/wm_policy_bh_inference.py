"""Post-hoc policy-versus-B&H inference on an immutable executed cohort.

This module never fits, selects or executes a policy. Its caller must register
the post-hoc protocol and source/input hashes before reading real Test ledgers.
Market blocks are sampled once per replicate for all training seeds. The seed
IDs are then sampled with replacement; seed paths are not independent markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .wm_full_grid_statistics import _block_sums
from .wm_policy_ablation_evaluation import ARMS, STAGES, accounting, reset_mask
from .wm_significance_statistics import holm_fixed_family


COHORT = ("fold", "decision_id", "timestamp", "bar_log_return")


def _panel(frame):
    required = {*COHORT, "net_log_return"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("executed ledgers require a nonempty complete return/time cohort")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, errors="coerce")
    for name in ("fold", "decision_id", "bar_log_return", "net_log_return"):
        values = pd.to_numeric(result[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("cohort identifiers and executed returns must be finite")
        if name in ("fold", "decision_id"):
            if np.any(values != np.floor(values)):
                raise ValueError("cohort identifiers must be integers")
            result[name] = values.astype(np.int64)
        else:
            result[name] = values
    result = result.sort_values("timestamp").reset_index(drop=True)
    resets = reset_mask(result.timestamp, result.fold)
    if result.duplicated(["fold", "decision_id"]).any():
        raise ValueError("each market decision must occur exactly once in one path")
    if "reset" in result and not np.array_equal(result.reset.to_numpy(), resets):
        raise ValueError("executed cohort resets must match every gap/fold boundary exactly")
    return result


def infer_bh(executed_policies, executed_bh, cfg=None):
    """Return eight paired two-sided tests and 1/28-day sensitivity intervals.

    Inputs are *executed* paths; the B&H path has unit inventory and identical
    bar returns. The positive estimate favors the policy, in annualized natural
    log-growth units. It is not an arithmetic or compounded return difference.
    Seven-day p values receive fixed-family Holm8 correction. Other block
    lengths are descriptive sensitivity analyses, without significance claims.
    Segments with <= block_days observed UTC days remain fixed in every draw.
    Their contribution makes intervals conditional on unsupported observations.
    """
    cfg = cfg or {}
    settings = cfg.get("statistics", {})
    reps = settings.get("bootstrap_replicates", 10000)
    blocks = tuple(settings.get("block_days", [1, 7, 28]))
    seeds = list(cfg.get("seeds", [7, 19, 42]))
    folds = list(cfg.get("folds", range(6)))
    random_seed = settings.get("seed", 20261112)
    if (int(reps) != reps or reps < 2 or not blocks or 7 not in blocks
            or len(set(blocks)) != len(blocks) or any(int(b) != b or b < 1 for b in blocks)
            or not seeds or len(set(seeds)) != len(seeds)
            or not folds or len(set(folds)) != len(folds)
            or int(random_seed) != random_seed or random_seed < 0):
        raise ValueError("valid registered folds/seeds, replicates and seven-day primary required")
    reps, random_seed = int(reps), int(random_seed)
    data = executed_policies.copy()
    if not {"seed", "arm", "stage"}.issubset(data.columns):
        raise ValueError("policies require seed/arm/stage identifiers")
    # No silent omission of an explored condition or training seed.
    expected = {(fold, seed, arm, stage) for fold in folds for seed in seeds
                for arm in ARMS for stage in STAGES}
    actual = set(data[["fold", "seed", "arm", "stage"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("exactly the registered full fold/seed/arm/stage grid is required")
    bh_data = executed_bh
    if "baseline" in bh_data:
        bh_data = bh_data.loc[bh_data.baseline.eq("bh")]
    bh = _panel(bh_data)
    if set(bh.fold) != set(folds):
        raise ValueError("B&H must cover exactly the registered folds")
    if not np.allclose(bh.net_log_return, bh.bar_log_return, rtol=0, atol=1e-12):
        raise ValueError("B&H executed net returns must equal the unit-exposure market returns")
    if "executed_position" in bh and not np.allclose(bh.executed_position, 1, rtol=0, atol=1e-12):
        raise ValueError("B&H must have unit executed inventory")
    if any(name in bh and not np.allclose(bh[name], 0, rtol=0, atol=1e-12)
           for name in ("cost_fraction", "borrow_fraction")):
        raise ValueError("B&H must have zero turnover fees and borrowing costs")
    expected_n = settings.get("expected_market_rows")
    if expected_n is not None and len(bh) != int(expected_n):
        raise ValueError("the common cohort row count differs from registration")
    ref = bh[list(COHORT)]
    resets = reset_mask(ref.timestamp, ref.fold)
    day_frame = pd.DataFrame({"fold": ref.fold, "segment": np.cumsum(resets) - 1,
                              "day": ref.timestamp.dt.floor("D"), "count": 1})
    annual = accounting(cfg)["annual_bars"]
    rows = []
    for stage in STAGES:
        for arm in ARMS:
            differences = []
            for seed in seeds:
                part = _panel(data.loc[data.seed.eq(seed) & data.arm.eq(arm) & data.stage.eq(stage)])
                if not part[list(COHORT)].equals(ref):
                    raise ValueError("all policies, seeds and B&H require the identical full cohort")
                differences.append(part.net_log_return.to_numpy() - bh.net_log_return.to_numpy())
            differences = np.column_stack(differences)
            daily = day_frame.copy()
            columns = [f"seed_{i}" for i in range(len(seeds))]
            daily[columns] = differences
            grouped = daily.groupby(["fold", "segment", "day"], sort=True).sum().reset_index()
            values = [*columns, "count"]
            segments = [part[values].to_numpy(float) for _, part in grouped.groupby(["fold", "segment"], sort=True)]
            estimate = float(differences.mean() * annual)
            for block in blocks:
                # Common random numbers also pair the eight contrasts; adding
                # identical training seeds cannot create extra market samples.
                rng = np.random.default_rng(np.random.SeedSequence([random_seed, int(block)]))
                samples = np.zeros((reps, len(values)))
                varying = supported = fixed = 0
                for segment in segments:
                    if len(segment) > block:
                        samples += _block_sums(segment, int(block), reps, rng)
                        varying += 1
                        supported += int(segment[:, -1].sum())
                    else:
                        samples += segment.sum(axis=0)
                        fixed += int(segment[:, -1].sum())
                selected_seeds = rng.integers(0, len(seeds), size=(reps, len(seeds)))
                sums = np.take_along_axis(samples[:, :-1], selected_seeds, axis=1).mean(axis=1)
                distribution = sums / samples[:, -1] * annual
                available = bool(varying and np.isfinite(distribution).all())
                low = high = p = None
                if available:
                    low, high = map(float, np.quantile(distribution, [.025, .975]))
                    p = float((1 + np.count_nonzero(np.abs(distribution - estimate) >= abs(estimate))) / (reps + 1))
                rows.append({"comparison": f"{stage}_{arm}_minus_bh", "arm": arm, "stage": stage,
                    "left_stage": stage, "left_arm": arm, "right_stage": "baseline", "right_arm": "bh",
                    "baseline": "bh", "metric": "annualized_net_log_growth_difference", "estimate": estimate,
                    "ci_low": low, "ci_high": high, "p_value": p, "block_days": int(block),
                    "primary_inference": int(block) == 7, "replicates": reps,
                    "n_unique_market_rows": len(ref), "n_training_seeds": len(seeds), "n_folds": len(folds),
                    "n_observed_market_days": int(ref.timestamp.dt.floor("D").nunique()),
                    "n_fold_segment_day_units": len(grouped), "n_reset_segments": int(resets.sum()),
                    "n_resampleable_segments": varying, "n_resampleable_rows": supported, "n_fixed_rows": fixed,
                    "resampleable_row_fraction": supported / len(ref), "inference_available": available,
                    "seed_resampling": "seed IDs with replacement; identical sampled market days for all seeds",
                    "market_sample_count": "unique market bars, not bars multiplied by training seeds",
                    "unsupported_segment_policy": "retained fixed; no shorter-block or IID fallback",
                    "estimand": "seed mean annualized mean net-log-return minus B&H on common saved cohort",
                    "inference_scope": "post-hoc conditional cohort; not full-calendar or live-return evidence"})
    family = {row["comparison"]: 1. if row["p_value"] is None else row["p_value"]
              for row in rows if row["primary_inference"]}
    if len(family) != 8:
        raise ValueError("exactly eight post-hoc policy-versus-B&H hypotheses required")
    adjusted = holm_fixed_family(family, family_size=8)
    for row in rows:
        value = adjusted[row["comparison"]] if row["primary_inference"] else None
        row["p_holm_8"] = value
        row["significant_difference"] = bool(value is not None and value < .05)
        row["significant_improvement"] = bool(row["significant_difference"] and row["estimate"] > 0)
        row["significant_underperformance"] = bool(row["significant_difference"] and row["estimate"] < 0)
    return pd.DataFrame(rows)
