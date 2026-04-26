from __future__ import annotations

import numpy as np
import torch


ROUTE_NAMES = ("neutral", "de_risk", "recovery", "overweight")


def run_test_stage(
    *,
    actor,
    wm_trainer,
    wfo_dataset,
    seq_len: int,
    test_regime_probs,
    test_advantage_values,
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
        enc_test["z"],
        enc_test["h"],
        regime_np=test_regime_probs,
        advantage_np=test_advantage_values,
        device=device,
    )
    if getattr(actor, "route_head", None) is not None:
        with torch.no_grad():
            dev = torch.device(device)
            z_t = torch.tensor(enc_test["z"], dtype=torch.float32, device=dev)
            h_t = torch.tensor(enc_test["h"], dtype=torch.float32, device=dev)
            regime_t = (
                torch.tensor(test_regime_probs, dtype=torch.float32, device=dev)
                if test_regime_probs is not None and getattr(actor, "regime_dim", 0) > 0
                else None
            )
            advantage_t = (
                torch.tensor(test_advantage_values, dtype=torch.float32, device=dev)
                if test_advantage_values is not None and getattr(actor, "advantage_dim", 0) > 0
                else None
            )
            controller_state = torch.zeros(1, actor.inventory_dim, dtype=torch.float32, device=dev)
            pred_routes = []
            route_conf = []
            t_route = min(len(z_t), len(positions))
            if regime_t is not None:
                t_route = min(t_route, len(regime_t))
            if advantage_t is not None:
                t_route = min(t_route, len(advantage_t))
            for i in range(t_route):
                reg_i = regime_t[i:i + 1] if regime_t is not None else None
                adv_i = advantage_t[i:i + 1] if advantage_t is not None else None
                probs = actor.route_controller_probs(
                    z_t[i:i + 1],
                    h_t[i:i + 1],
                    inventory=controller_state,
                    regime=reg_i,
                    advantage=adv_i,
                )
                pred_routes.append(int(torch.argmax(probs, dim=-1).item()))
                route_conf.append(float(torch.max(probs, dim=-1).values.item()))
                pos_i = torch.tensor([[float(positions[i])]], dtype=torch.float32, device=dev)
                controller_state = actor.update_controller_state(controller_state, pos_i)
            if pred_routes:
                counts = np.bincount(np.asarray(pred_routes, dtype=np.int64), minlength=len(ROUTE_NAMES))
                rates = counts / max(counts.sum(), 1)
                route_dist = " ".join(f"{name}={rates[idx]:.0%}" for idx, name in enumerate(ROUTE_NAMES))
                print(
                    f"  Route dist: {route_dist} "
                    f"active={1.0 - rates[0]:.1%} conf={np.mean(route_conf):.3f}"
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
