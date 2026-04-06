from __future__ import annotations

import numpy as np

from unidream.eval.pbo import compute_pbo, deflated_sharpe


def compute_overfitting_diagnostics(fold_results: dict, eval_cfg: dict) -> tuple[float, float, list[float]]:
    pnl_list = [r["metrics"].pnl_series for r in fold_results.values()]
    pbo = compute_pbo(
        pnl_list,
        n_combinations=eval_cfg.get("pbo_n_trials"),
    )
    all_sharpes = [r["metrics"].sharpe for r in fold_results.values()]
    best_sharpe = max(all_sharpes)
    t_avg = int(np.mean([len(r["metrics"].pnl_series) for r in fold_results.values()]))
    dsr = deflated_sharpe(best_sharpe, n_trials=1, T=t_avg)
    return pbo, dsr, all_sharpes


def aggregate_scorecards(scorecards: list[dict]) -> dict | None:
    if not scorecards:
        return None

    aggregate_scorecard = {
        "alpha_excess_pt": float(np.mean([s["alpha_excess_pt"] for s in scorecards])),
        "sharpe_delta": float(np.mean([s["sharpe_delta"] for s in scorecards])),
        "maxdd_delta_pt": float(np.mean([s["maxdd_delta_pt"] for s in scorecards])),
        "win_rate_vs_bh": float(np.mean([s["win_rate_vs_bh"] for s in scorecards])),
        "collapse_guard_pass": all(s["collapse_guard_pass"] for s in scorecards),
        "collapse_guard_reasons": sorted(
            {reason for s in scorecards for reason in s["collapse_guard_reasons"]}
        ),
        "required": {
            "alpha_excess": all(s["required"]["alpha_excess"] for s in scorecards),
            "sharpe_delta": all(s["required"]["sharpe_delta"] for s in scorecards),
            "maxdd_delta": all(s["required"]["maxdd_delta"] for s in scorecards),
            "win_rate_vs_bh": all(s["required"]["win_rate_vs_bh"] for s in scorecards),
            "collapse_guard": all(s["required"]["collapse_guard"] for s in scorecards),
        },
        "stretch": {
            "alpha_excess": any(s["stretch"]["alpha_excess"] for s in scorecards),
            "maxdd_delta": any(s["stretch"]["maxdd_delta"] for s in scorecards),
        },
    }
    aggregate_scorecard["m2_pass"] = all(aggregate_scorecard["required"].values())
    aggregate_scorecard["stretch_hit"] = any(aggregate_scorecard["stretch"].values())
    return aggregate_scorecard


def print_stage_summary(fold_results: dict, default_stage: str) -> None:
    print("\n" + "=" * 60)
    print("Stage Summary")
    print("=" * 60)
    for fold_idx, result in fold_results.items():
        print(f"  Fold {fold_idx}: completed_stage={result.get('completed_stage', default_stage)}")


def print_training_summary(
    fold_results: dict,
    all_sharpes: list[float],
    aggregate_scorecard: dict | None,
    pbo: float,
    dsr: float,
    format_scorecard,
) -> None:
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for fold_idx, result in fold_results.items():
        metrics = result["metrics"]
        scorecard = result.get("scorecard")
        extra = f" | {format_scorecard(scorecard)}" if scorecard is not None else ""
        print(
            f"  Fold {fold_idx}: Sharpe={metrics.sharpe:.3f}, MaxDD={metrics.max_drawdown:.3f}, "
            f"Calmar={metrics.calmar:.3f}, TotalRet={metrics.total_return:.4f}{extra}"
        )
    print(f"  Mean Sharpe: {np.mean(all_sharpes):.3f}")
    if aggregate_scorecard is not None:
        print(f"  Aggregate M2: {format_scorecard(aggregate_scorecard)}")
    dsr_summary = f"{dsr:.4f}" if np.isfinite(dsr) else "N/A"
    print(f"  PBO (simplified): {pbo:.4f} | Sharpe t-stat: {dsr_summary}")
