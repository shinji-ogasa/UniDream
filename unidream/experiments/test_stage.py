from __future__ import annotations


def run_test_stage(
    *,
    actor,
    wm_trainer,
    wfo_dataset,
    seq_len: int,
    test_regime_probs,
    device: str,
    cfg: dict,
    costs_cfg: dict,
    backtest_cls,
    pnl_attribution_fn,
    action_stats_fn,
    format_action_stats_fn,
    ac_alerts_fn,
    benchmark_positions_fn,
    benchmark_position: float,
    m2_scorecard_fn,
    format_m2_scorecard_fn,
    log_ts,
) -> dict:
    print(f"\n[{log_ts()}] [Step 5] Test Backtest...")
    test_features = wfo_dataset.test_dataset().features.numpy()
    test_returns = wfo_dataset.test_returns

    enc_test = wm_trainer.encode_sequence(test_features, seq_len=seq_len)
    positions = actor.predict_positions(
        enc_test["z"], enc_test["h"], regime_np=test_regime_probs, device=device
    )

    t_min = min(len(test_returns), len(positions))
    metrics = backtest_cls(
        test_returns[:t_min],
        positions[:t_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=benchmark_positions_fn(t_min),
    ).run()

    test_attr = pnl_attribution_fn(
        test_returns[:t_min],
        positions[:t_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
    )
    test_stats = action_stats_fn(positions[:t_min], benchmark_position=benchmark_position)
    test_scorecard = m2_scorecard_fn(metrics, test_stats, cfg)

    print(f"  Sharpe:   {metrics.sharpe:.3f}")
    print(f"  Sortino:  {metrics.sortino:.3f}")
    print(f"  MaxDD:    {metrics.max_drawdown:.3f}")
    print(f"  Calmar:   {metrics.calmar:.3f}")
    print(f"  TotalRet: {metrics.total_return:.4f}")
    if metrics.alpha_excess is not None:
        print(f"  AlphaEx:  {100.0 * metrics.alpha_excess:+.2f} pt/yr")
        print(f"  SharpeΔ:  {(metrics.sharpe_delta or 0.0):+.3f}")
        print(f"  MaxDDΔ:   {100.0 * (metrics.maxdd_delta or 0.0):+.2f} pt")
        print(f"  WinRate:  {(metrics.win_rate_vs_bh or 0.0):.1%}")
        print(f"  Score:    {format_m2_scorecard_fn(test_scorecard)}")
    print(
        f"  PnL attr: long={test_attr['long_gross']:+.4f}  "
        f"short={test_attr['short_gross']:+.4f}  "
        f"cost={test_attr['cost_total']:.4f}  "
        f"net={test_attr['net_total']:+.4f}"
    )
    print(f"  Test dist: {format_action_stats_fn(test_stats)}")
    ac_alerts_fn("test", test_stats)

    return {
        "fold": None,
        "metrics": metrics,
        "scorecard": test_scorecard,
        "positions": positions[:t_min],
        "test_returns": test_returns[:t_min],
        "completed_stage": "test",
    }
