"""Execution, common teacher and paired inference for the policy ablation.

This is a conditional saved-forecast-cohort experiment. Gaps/fold boundaries
start fresh episodes endowed with inventory1; no unobserved return is filled.
All policies and controls use exactly the same scored bars and execution lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .wm_full_grid_statistics import _block_sums
from .wm_significance_statistics import holm_fixed_family


BAR_NS = 15 * 60 * 10**9
ARMS = tuple("ABCD")
STAGES = ("bc", "rl")
CONTRASTS = (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C"))


def accounting(cfg=None):
    settings = (cfg or {}).get("accounting", {})
    result = {"position_min": .50, "position_max": 1.12, "max_position_step": .08,
        "cost_rate": .00055, "initial_position": 1., "drawdown_penalty": .25,
        "annual_bars": 35040., "borrow_apr": .10}
    result.update({name: float(settings[name]) for name in result if name in settings})
    if (not np.isfinite(list(result.values())).all() or result["position_min"] < 0
            or not result["position_min"] <= result["initial_position"] <= result["position_max"]
            or result["position_max"] <= result["position_min"] or result["max_position_step"] <= 0
            or not 0 <= result["cost_rate"] < 1 or result["drawdown_penalty"] < 0 or result["annual_bars"] <= 0
            or result["borrow_apr"] < 0):
        raise ValueError("finite valid execution/accounting constants required")
    return result


def teacher_target(future_path_log_returns, current_requested, cfg=None, *, return_details=False):
    """Choose one feasible next-bar request using Train-only future labels.

    The16 cumulative log returns run from close[t] to close[t+1..16]. Each projected target
    is held for those16 bars, paying its entry change once. This label helper
    must never supply inventory or other actor inputs from teacher rollouts.
    """
    settings = accounting(cfg)
    future = np.asarray(future_path_log_returns, dtype=np.float64)
    current = np.asarray(current_requested, dtype=np.float64)
    scalar = future.ndim == 1
    if scalar:
        future = future[None, :]
    if future.ndim != 2 or future.shape[1] != 16 or not len(future) or not np.isfinite(future).all():
        raise ValueError("future teacher labels must be finite[rows,16] cumulative log returns")
    bar_returns = np.diff(future, axis=1, prepend=np.zeros((len(future), 1)))
    current = np.broadcast_to(current, (len(future),))
    if not np.isfinite(current).all() or np.any(current < settings["position_min"] - 1e-7) or np.any(current > settings["position_max"] + 1e-7):
        raise ValueError("current requested inventory must be finite and feasible")
    current = np.clip(current, settings["position_min"], settings["position_max"])
    grid = np.asarray((cfg or {}).get("teacher", {}).get("position_grid", [.50, .70, .85, .94, 1., 1.06, 1.12]), float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError("teacher position grid must be nonempty and finite")
    targets = np.clip(grid[None, :], current[:, None] - settings["max_position_step"], current[:, None] + settings["max_position_step"])
    targets = np.clip(targets, settings["position_min"], settings["position_max"])
    cost = settings["cost_rate"] * np.abs(targets - current[:, None])
    borrow = np.maximum(targets - 1, 0) * settings["borrow_apr"] / settings["annual_bars"]
    factors = 1 + targets[:, :, None] * np.expm1(bar_returns[:, None, :]) - borrow[:, :, None]
    if np.any(factors <= 0) or np.any(cost >= 1):
        raise ValueError("teacher path reaches a non-positive wealth factor")
    logs = np.log1p(-cost)[:, :, None] + np.cumsum(np.log(factors), axis=-1)
    path = np.concatenate([np.ones((*logs.shape[:2], 1)), np.exp(logs)], axis=-1)
    drawdown = np.max(1 - path / np.maximum.accumulate(path, axis=-1), axis=-1)
    utility = logs[:, :, -1] - future[:, -1, None] - settings["drawdown_penalty"] * drawdown
    best = np.argmax(utility, axis=1)
    chosen = targets[np.arange(len(future)), best]
    if return_details:
        return {"target_position": chosen, "utility": utility[np.arange(len(future)), best],
            "max_drawdown": drawdown[np.arange(len(future)), best], "candidate_targets": targets, "candidate_utility": utility}
    return float(chosen[0]) if scalar else chosen


def reset_mask(timestamps, folds=None):
    time = pd.to_datetime(timestamps, utc=True)
    if time.isna().any() or len(time) < 1:
        raise ValueError("nonempty finite UTC timestamps required")
    ns = np.asarray(time.astype("int64"))
    if np.any(ns % BAR_NS):
        raise ValueError("timestamps must lie on the15-minute UTC grid")
    if np.any(np.diff(ns) <= 0):
        raise ValueError("one path must have unique increasing timestamps")
    reset = np.r_[True, np.diff(ns) != BAR_NS]
    if folds is not None:
        fold = np.asarray(folds)
        if fold.shape != ns.shape:
            raise ValueError("fold identifiers must match timestamps")
        reset[1:] |= fold[1:] != fold[:-1]
    return reset


def bounded_requests(target_positions, timestamps=None, folds=None, cfg=None, *, resets=None):
    settings = accounting(cfg)
    targets = np.asarray(target_positions, dtype=float)
    if targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("finite nonempty one-dimensional targets required")
    reset = _resets(len(targets), timestamps, folds, resets)
    result = np.empty(len(targets))
    previous = settings["initial_position"]
    for index, target in enumerate(targets):
        if reset[index]:
            previous = settings["initial_position"]
        previous = float(np.clip(np.clip(target, settings["position_min"], settings["position_max"]),
            previous - settings["max_position_step"], previous + settings["max_position_step"]))
        result[index] = previous
    return result


def _resets(n, timestamps, folds, resets):
    required = reset_mask(timestamps, folds) if timestamps is not None else np.r_[True, np.zeros(n - 1, dtype=bool)]
    if resets is not None:
        supplied = np.asarray(resets)
        if supplied.dtype != bool or supplied.shape != (n,) or np.any(required & ~supplied):
            raise ValueError("explicit episode resets must include all gaps/fold boundaries")
        return supplied
    return required


def portfolio_step(bar_log_return, executed_position, previous_executed, cfg=None):
    """Pure scalar/vector accounting shared with independent Torch checks."""
    settings = accounting(cfg)
    returns, position, previous = np.broadcast_arrays(np.asarray(bar_log_return, float),
        np.asarray(executed_position, float), np.asarray(previous_executed, float))
    if not all(np.isfinite(values).all() for values in (returns, position, previous)):
        raise ValueError("finite returns and inventories required")
    turnover = np.abs(position - previous)
    cost = settings["cost_rate"] * turnover
    borrow = np.maximum(position - 1, 0) * settings["borrow_apr"] / settings["annual_bars"]
    gross = position * np.expm1(returns)
    financed_factor = 1 + gross - borrow
    if np.any(financed_factor <= 0) or np.any(cost >= 1):
        raise ValueError("policy reaches a non-positive wealth factor")
    log_net = np.log(financed_factor) + np.log1p(-cost)
    return {"net_log_return": log_net, "net_simple_return": np.expm1(log_net), "cost_fraction": cost,
        "turnover": turnover, "borrow_fraction": borrow, "gross_simple_return": gross}


def execute_positions(bar_log_return, requested_position, timestamps=None, folds=None, cfg=None, *, resets=None):
    """Execute requests one bar later, with exact multiplicative fees/wealth."""
    settings = accounting(cfg)
    returns, requests = np.asarray(bar_log_return, float), np.asarray(requested_position, float)
    if returns.ndim != 1 or requests.shape != returns.shape or not len(returns) or not np.isfinite(returns).all() or not np.isfinite(requests).all():
        raise ValueError("matching nonempty finite one-dimensional returns/requests required")
    if np.any(requests < settings["position_min"] - 1e-8) or np.any(requests > settings["position_max"] + 1e-8):
        raise ValueError("requested positions violate the common inventory bounds")
    reset = _resets(len(returns), timestamps, folds, resets)
    executed = np.r_[settings["initial_position"], requests[:-1]].copy()
    executed[reset] = settings["initial_position"]
    if np.any(np.abs(requests - executed) > settings["max_position_step"] + 1e-7):
        raise ValueError("requested position changes violate the common maximum step")
    previous_executed = np.r_[settings["initial_position"], executed[:-1]]
    previous_executed[reset] = settings["initial_position"]
    steps = portfolio_step(returns, executed, previous_executed, cfg)
    turnover, cost, borrow, log_net = (steps[name] for name in ("turnover", "cost_fraction", "borrow_fraction", "net_log_return"))
    delta_mdd = np.empty(len(returns))
    starts = np.flatnonzero(reset)
    for start, end in zip(starts, np.r_[starts[1:], len(returns)]):
        logs = np.cumsum(log_net[start:end])
        peaks = np.maximum.accumulate(np.r_[0., logs])[1:]
        running = np.maximum.accumulate(-np.expm1(logs - peaks))
        delta_mdd[start:end] = np.maximum(0, np.diff(running, prepend=0.))
    reward = log_net - returns - settings["drawdown_penalty"] * delta_mdd
    frame = pd.DataFrame({"bar_log_return": returns, "requested_position": requests, "executed_position": executed,
        "turnover": turnover, "cost_fraction": cost, "net_log_return": log_net, "net_simple_return": np.expm1(log_net),
        "borrow_fraction": borrow,
        "gross_simple_return": executed * np.expm1(returns), "reset": reset,
        "increment_max_drawdown": delta_mdd, "reward": reward, "equity": np.exp(np.cumsum(log_net))})
    will_execute = np.r_[~reset[1:], False]
    prior_requests = np.r_[settings["initial_position"], requests[:-1]]
    prior_requests[reset] = settings["initial_position"]
    frame["execution_has_prior_request"] = ~reset
    frame["request_will_execute"] = will_execute
    frame["requested_turnover_executable"] = np.abs(requests - prior_requests) * will_execute
    if timestamps is not None:
        frame["timestamp"] = pd.to_datetime(timestamps, utc=True)
    if folds is not None:
        frame["fold"] = folds
    return frame


def _risk_metrics(log_returns, annual_bars):
    values = np.asarray(log_returns, float)
    simple = np.expm1(values)
    wealth = np.r_[1., np.exp(np.cumsum(values))]
    maximum_dd = float(np.max(1 - wealth / np.maximum.accumulate(wealth)))
    sigma = float(np.std(simple, ddof=0))
    return {"net_total_return": float(np.expm1(values.sum())),
        "annualized_net_log_growth": float(values.mean() * annual_bars),
        "observed_bar_annual_return": float(np.expm1(values.mean() * annual_bars)),
        "sharpe": float(simple.mean() / sigma * np.sqrt(annual_bars)) if sigma > 1e-12 else 0.,
        "max_drawdown": maximum_dd}


def score_path(ledger, cfg=None):
    settings = accounting(cfg)
    required = {"bar_log_return", "net_log_return", "executed_position", "turnover", "cost_fraction"}
    if not required.issubset(ledger):
        raise ValueError("an executed policy ledger is required")
    if not len(ledger) or not np.isfinite(ledger[list(required)].to_numpy(float)).all():
        raise ValueError("the complete finite common cohort must be retained")
    score = _risk_metrics(ledger.net_log_return, settings["annual_bars"])
    bh = _risk_metrics(ledger.bar_log_return, settings["annual_bars"])
    position = ledger.executed_position.to_numpy(float)
    n = len(position)
    turnover = float(ledger.turnover.sum())
    days = n / 96
    sigma = float(np.std(position))
    boundary = float(np.mean((position <= settings["position_min"] + 1e-3) | (position >= settings["position_max"] - 1e-3)))
    previous_wealth = np.exp(np.r_[0., np.cumsum(ledger.net_log_return.to_numpy())[:-1]])
    score.update(n_rows=n, benchmark_total_return=bh["net_total_return"], alpha_excess=score["net_total_return"] - bh["net_total_return"],
        annualized_log_excess=score["annualized_net_log_growth"] - bh["annualized_net_log_growth"],
        benchmark_sharpe=bh["sharpe"], sharpe_delta=score["sharpe"] - bh["sharpe"],
        benchmark_max_drawdown=bh["max_drawdown"], maxdd_delta=bh["max_drawdown"] - score["max_drawdown"],
        turnover=turnover, turnover_per_observed_day=turnover / days, mean_exposure=float(position.mean()),
        exposure_std=sigma, exposure_min=float(position.min()), exposure_max=float(position.max()),
        fee_fraction_sum=float(ledger.cost_fraction.sum()), fee_equity_units=float(np.sum(previous_wealth * ledger.cost_fraction)),
        borrow_fraction_sum=float(ledger.borrow_fraction.sum()) if "borrow_fraction" in ledger else 0.,
        cost_drag_log=float(-np.log1p(-ledger.cost_fraction).sum()), boundary_fraction=boundary,
        near_bh_fraction=float(np.mean(np.abs(position - 1) <= .01)),
        collapse_constant=bool(sigma < .005 and turnover / days < .01), collapse_boundary=bool(boundary >= .95),
        n_resets=int(ledger.reset.sum()) if "reset" in ledger else None,
        conditional_saved_cohort=True, full_calendar_profit_claim=False)
    if "reward" in ledger:
        score["reward_total"] = float(ledger.reward.sum())
    if "requested_turnover_executable" in ledger:
        score["requested_turnover_executable"] = float(ledger.requested_turnover_executable.sum())
    return score


def evaluate_policy_ledger(frame, cfg=None):
    """Return executed paths,144 fold rows,24 seed rows and8 seed means."""
    cfg = cfg or {}
    data = frame.copy()
    if "arm" not in data and "group" in data:
        data["arm"] = data["group"]
    required = {"fold", "seed", "arm", "stage", "timestamp", "decision_id", "bar_log_return", "requested_position"}
    if not required.issubset(data):
        raise ValueError("policy ledger lacks condition/time/return/request columns")
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data["stage"] = data.stage.str.lower()
    folds = cfg.get("folds", list(range(6)))
    seeds = cfg.get("seeds", [7, 19, 42])
    expected = {(fold, seed, arm, stage) for fold in folds for seed in seeds for arm in ARMS for stage in STAGES}
    actual = set(data[["fold", "seed", "arm", "stage"]].itertuples(index=False, name=None))
    if actual != expected or data.duplicated(["fold", "seed", "arm", "stage", "decision_id"]).any():
        raise ValueError("every registered fold/seed/arm/stage must share one unique complete cohort")
    reference, paths, folded, pooled = None, [], [], []
    for (seed, arm, stage), panel in data.groupby(["seed", "arm", "stage"], sort=True):
        panel = panel.sort_values("timestamp").reset_index(drop=True)
        cohort = panel[["fold", "decision_id", "timestamp", "bar_log_return"]]
        if reference is not None and not cohort.equals(reference):
            raise ValueError("all seeds/arms/stages must use exactly the same return/time cohort")
        reference = cohort
        executed = execute_positions(panel.bar_log_return, panel.requested_position, panel.timestamp, panel.fold, cfg)
        for key in ("decision_id", "seed", "arm", "stage"):
            executed[key] = panel[key].to_numpy()
        paths.append(executed)
        pooled.append({"seed": seed, "arm": arm, "stage": stage, **score_path(executed, cfg)})
        for fold, part in executed.groupby("fold", sort=True):
            folded.append({"fold": fold, "seed": seed, "arm": arm, "stage": stage, **score_path(part, cfg)})
    pooled_frame = pd.DataFrame(pooled)
    aggregate = []
    metrics = [name for name in pooled_frame.select_dtypes(include=[np.number]).columns if name not in ("seed", "n_rows", "n_resets")]
    for (arm, stage), panel in pooled_frame.groupby(["arm", "stage"], sort=True):
        row = {"arm": arm, "stage": stage, "n_seeds": len(panel), "n_unique_market_rows": int(panel.n_rows.iloc[0])}
        for metric in metrics:
            row[metric + "_seed_mean"] = float(panel[metric].mean())
            row[metric + "_seed_min"] = float(panel[metric].min())
            row[metric + "_seed_max"] = float(panel[metric].max())
        row["collapse_constant_seed_fraction"] = float(panel.collapse_constant.mean())
        row["collapse_boundary_seed_fraction"] = float(panel.collapse_boundary.mean())
        aggregate.append(row)
    return pd.concat(paths, ignore_index=True), pd.DataFrame(folded), pooled_frame, pd.DataFrame(aggregate)


def paired_policy_inference(executed, cfg=None):
    """Same days for paired seeds, with paired seed-ID resampling, Holm12."""
    cfg = cfg or {}
    settings = cfg.get("statistics", {})
    reps = int(settings.get("bootstrap_replicates", 10000))
    blocks = tuple(settings.get("block_days", [1, 7, 28]))
    seeds = list(cfg.get("seeds", [7, 19, 42]))
    if reps < 2 or 7 not in blocks or not seeds:
        raise ValueError("multiple replicates, registered paired seeds and seven-day primary required")
    contrasts = [(stage, left, stage, right) for stage in STAGES for left, right in CONTRASTS]
    contrasts += [("rl", arm, "bc", arm) for arm in ARMS]
    rows = []
    for number, (left_stage, left_arm, right_stage, right_arm) in enumerate(contrasts):
        columns = []
        ref = None
        for seed in seeds:
            def choose(stage, arm):
                panel = executed.loc[(executed.seed == seed) & (executed.stage == stage) & (executed.arm == arm)]
                return panel.sort_values("timestamp").reset_index(drop=True)
            left, right = choose(left_stage, left_arm), choose(right_stage, right_arm)
            keys = ["fold", "decision_id", "timestamp", "bar_log_return"]
            if left.empty or not left[keys].equals(right[keys]) or (ref is not None and not left[keys].equals(ref)):
                raise ValueError("paired ablations require the identical full cohort and same seed IDs")
            ref = left[keys]
            columns.append(left.net_log_return.to_numpy() - right.net_log_return.to_numpy())
        differences = np.column_stack(columns)
        reset = reset_mask(ref.timestamp, ref.fold)
        daily = pd.DataFrame({"fold": ref.fold, "segment": np.cumsum(reset) - 1,
            "day": ref.timestamp.dt.floor("D"), "count": 1})
        for index in range(len(seeds)):
            daily[f"seed_{index}"] = differences[:, index]
        grouped = daily.groupby(["fold", "segment", "day"], sort=True).sum().reset_index()
        values = [f"seed_{index}" for index in range(len(seeds))] + ["count"]
        segments = [part[values].to_numpy(float) for _, part in grouped.groupby(["fold", "segment"], sort=True)]
        annual = accounting(cfg)["annual_bars"]
        estimate = float(differences.mean() * annual)
        for block in blocks:
            rng = np.random.default_rng(np.random.SeedSequence([int(settings.get("seed", 20261112)), number, int(block)]))
            samples = np.zeros((reps, len(values)))
            resampleable_rows = fixed_rows = varying = 0
            for segment in segments:
                if len(segment) > block:
                    samples += _block_sums(segment, int(block), reps, rng)
                    resampleable_rows += int(segment[:, -1].sum())
                    varying += 1
                else:
                    samples += segment.sum(axis=0)
                    fixed_rows += int(segment[:, -1].sum())
            selected_seed_ids = rng.integers(0, len(seeds), size=(reps, len(seeds)))
            logs = np.take_along_axis(samples[:, :-1], selected_seed_ids, axis=1).mean(axis=1)
            distribution = logs / samples[:, -1] * annual
            available = bool(varying and np.isfinite(distribution).all())
            low = high = p = None
            if available:
                low, high = map(float, np.quantile(distribution, [.025, .975]))
                p = float((1 + np.sum(np.abs(distribution - estimate) >= abs(estimate))) / (reps + 1))
            rows.append({"comparison": f"{left_stage}_{left_arm}_minus_{right_stage}_{right_arm}",
                "left_stage": left_stage, "left_arm": left_arm, "right_stage": right_stage, "right_arm": right_arm,
                "metric": "annualized_net_log_growth_difference", "estimate": estimate, "ci_low": low, "ci_high": high,
                "p_value": p, "block_days": int(block), "primary_inference": int(block) == 7, "replicates": reps,
                "n_unique_market_rows": len(differences), "n_training_seeds": len(seeds), "n_folds": int(ref.fold.nunique()),
                "n_resampleable_segments": varying, "n_resampleable_rows": resampleable_rows, "n_fixed_rows": fixed_rows,
                "resampleable_row_fraction": resampleable_rows / len(differences), "inference_available": available,
                "seed_resampling": "paired seed IDs with replacement; identical sampled market days for all seeds",
                "unsupported_segment_policy": "retained fixed; no shorter-block or IID fallback"})
    family = {row["comparison"]: 1. if row["p_value"] is None else row["p_value"] for row in rows if row["primary_inference"]}
    if len(family) != 12:
        raise ValueError("exactly12 preregistered ablation/stage hypotheses required")
    adjusted = holm_fixed_family(family, family_size=12)
    for row in rows:
        row["p_holm_12"] = adjusted[row["comparison"]] if row["primary_inference"] else None
        row["significant_improvement"] = bool(row["primary_inference"] and row["p_holm_12"] < .05 and row["estimate"] > 0)
    return pd.DataFrame(rows)


def select_constant_exposure(bar_log_return, timestamps, folds=None, cfg=None, *, candidates=None):
    """Select a fixed exposure from validation only; Test is not an argument."""
    candidates = np.asarray(candidates if candidates is not None else [.50, .70, .85, 1., 1.06, 1.12], float)
    scores = []
    for position in candidates:
        request = bounded_requests(np.full(len(bar_log_return), position), timestamps, folds, cfg)
        path = execute_positions(bar_log_return, request, timestamps, folds, cfg)
        scores.append({"position": float(position), "score": score_path(path, cfg)["reward_total"]})
    chosen = max(scores, key=lambda row: row["score"])
    return {"position": chosen["position"], "selection_split": "validation_only", "test_used": False,
        "selection_metric": "sum_relative_log_reward_minus_incremental_episode_maxDD_penalty", "candidates": scores}


def volatility_target_requests(variance_forecast, target_sigma, timestamps=None, folds=None, cfg=None):
    variance = np.asarray(variance_forecast, float)
    if variance.ndim != 1 or not np.isfinite(variance).all() or np.any(variance < 0) or not np.isfinite(target_sigma) or target_sigma <= 0:
        raise ValueError("finite causal variance forecasts and positive Train/Val-fixed risk target required")
    return bounded_requests(float(target_sigma) / np.sqrt(np.maximum(variance, 1e-12)), timestamps, folds, cfg)
