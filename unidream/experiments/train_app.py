import os

import numpy as np

from unidream.data.features import augment_with_rebound_features
from unidream.eval.regime import RegimeDetector, regime_metrics, print_regime_report

from .runtime import load_training_features
from .train_pipeline import run_wfo_folds
from .train_reporting import (
    aggregate_scorecards,
    compute_overfitting_diagnostics,
    print_stage_summary,
    print_training_summary,
)
from .wfo_runtime import build_wfo_splits, select_wfo_splits


def resolve_cache_dir(checkpoint_dir: str, cfg: dict) -> str:
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("cache_dir"):
        return str(logging_cfg["cache_dir"])

    candidate_dirs = [
        os.path.join(checkpoint_dir, "data_cache"),
        os.path.join(os.path.dirname(checkpoint_dir), "data_cache"),
        "checkpoints/data_cache",
    ]
    for path in candidate_dirs:
        if os.path.exists(path):
            return path
    return candidate_dirs[0]


def run_training_app(
    *,
    args,
    cfg: dict,
    active_cost_profile: str,
    run_fold_fn,
    format_m2_scorecard_fn,
    parser_error_fn,
):
    symbol = cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")

    print(f"UniDream Training | {symbol} {interval} | {args.start} → {args.end}")
    print(f"Device: {args.device} | Seed: {args.seed} | Resume: {args.resume}")
    print(f"Stages: {args.start_from} -> {args.stop_after}")
    costs_cfg = cfg.get("costs", {})
    total_cost_bps = (
        costs_cfg.get("spread_bps", 0.0) / 2
        + costs_cfg.get("fee_rate", 0.0) * 10000
        + costs_cfg.get("slippage_bps", 0.0)
    )
    print(
        "Costs: "
        f"profile={active_cost_profile} | "
        f"spread={costs_cfg.get('spread_bps', 0.0):.2f}bps "
        f"fee={costs_cfg.get('fee_rate', 0.0) * 10000:.2f}bps "
        f"slip={costs_cfg.get('slippage_bps', 0.0):.2f}bps "
        f"=> one-way Δpos=1 cost={total_cost_bps:.2f}bps"
    )

    cache_dir = resolve_cache_dir(args.checkpoint_dir, cfg)
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    data_cfg = cfg.get("data", {})
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )

    feature_extras_cfg = cfg.get("feature_extras", {})
    if feature_extras_cfg.get("rebound_v1", False):
        features_df = augment_with_rebound_features(
            features_df,
            raw_returns,
            zscore_window_days=cfg.get("normalization", {}).get("zscore_window_days", 60),
            interval=interval,
            windows_hours=feature_extras_cfg.get("rebound_windows_hours", [24, 72]),
        )
        raw_returns = raw_returns.loc[features_df.index]
        print(f"[Data] Rebound features added -> {features_df.shape}")

    print("[Data] WFO splits...")
    splits = build_wfo_splits(features_df, data_cfg)
    print(f"  {len(splits)} folds")
    if len(splits) == 0:
        print("ERROR: WFO splits が空です。データ期間が短すぎます。")
        return

    try:
        splits, selected_folds = select_wfo_splits(splits, args.folds)
    except ValueError as exc:
        parser_error_fn(str(exc))
        return
    if selected_folds is not None:
        print(f"  Running selected folds only: {selected_folds}")

    fold_results = run_wfo_folds(
        features_df=features_df,
        raw_returns=raw_returns,
        splits=splits,
        data_cfg=data_cfg,
        cfg=cfg,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        start_from=args.start_from,
        stop_after=args.stop_after,
        run_fold_fn=run_fold_fn,
    )

    if args.stop_after != "test":
        print_stage_summary(fold_results, args.stop_after)
        return

    print("\n[Eval] Overfitting Diagnostics (simplified)...")
    eval_cfg = cfg.get("eval", {})
    pbo, dsr, all_sharpes = compute_overfitting_diagnostics(fold_results, eval_cfg)
    print(f"  PBO (simplified): {pbo:.4f} (< 0.5 is better)")
    dsr_str = f"{dsr:.4f}" if np.isfinite(dsr) else f"N/A ({dsr})"
    print(f"  Sharpe t-stat (DSR, n_trials=1): {dsr_str} (> 0 is better)")

    print("\n[Eval] Regime Analysis...")
    all_test_returns = np.concatenate([r["test_returns"] for r in fold_results.values()])
    all_positions = np.concatenate([r["positions"] for r in fold_results.values()])
    try:
        detector = RegimeDetector(n_states=cfg.get("eval", {}).get("hmm_n_states", 3))
        regimes = detector.fit_predict(all_test_returns)
        regime_results = regime_metrics(
            all_test_returns,
            all_positions,
            regimes,
            n_states=detector.n_states,
            interval=interval,
            spread_bps=cfg.get("costs", {}).get("spread_bps", 5.0),
            fee_rate=cfg.get("costs", {}).get("fee_rate", 0.0004),
            slippage_bps=cfg.get("costs", {}).get("slippage_bps", 2.0),
        )
        print_regime_report(regime_results, detector)
    except Exception as e:
        print(f"  Regime analysis skipped: {e}")

    scorecards = [r["scorecard"] for r in fold_results.values() if "scorecard" in r]
    aggregate_scorecard = aggregate_scorecards(scorecards)
    print_training_summary(
        fold_results,
        all_sharpes,
        aggregate_scorecard,
        pbo,
        dsr,
        format_m2_scorecard_fn,
    )
