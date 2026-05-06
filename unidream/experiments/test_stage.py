from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from unidream.eval.backtest import compute_pnl
from unidream.experiments.policy_fire import predict_with_policy_flags as _predict_with_policy_flags


ROUTE_NAMES = ("neutral", "de_risk", "recovery", "overweight")
EXPOSURE_ROUTE_NAMES = ("neutral", "de_risk", "overweight")


def _load_external_policy_positions(cfg: dict, fold_idx: int | None, split: str) -> np.ndarray | None:
    ac_cfg = cfg.get("ac", {})
    source = str(ac_cfg.get("test_policy_source", ac_cfg.get("policy_source", "actor"))).lower()
    if source not in {"hierarchy_bundle", "external_hierarchy_bundle"}:
        return None
    if fold_idx is None:
        raise ValueError("External hierarchy policy requires fold_idx")
    oracle_cfg = cfg.get("oracle", {})
    bundle_dir = ac_cfg.get("external_policy_bundle_dir") or oracle_cfg.get("external_teacher_bundle_dir")
    if not bundle_dir:
        raise ValueError("External hierarchy policy requires external_policy_bundle_dir or oracle.external_teacher_bundle_dir")
    bundle_path = Path(bundle_dir) / f"fold{int(fold_idx):02d}_teacher.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"External hierarchy policy bundle not found: {bundle_path}")
    key = f"{split}_positions"
    with np.load(bundle_path) as data:
        if key not in data:
            raise KeyError(f"{bundle_path} does not contain {key}")
        positions = np.asarray(data[key], dtype=np.float32)
        source_id = int(np.asarray(data["source_id"]).reshape(-1)[0]) if "source_id" in data else -1
    print(
        f"  External policy: hierarchy_bundle fold={fold_idx} source_id={source_id} "
        f"split={split} path={bundle_path}"
    )
    return positions


def _active_incremental_pnl(
    *,
    returns,
    current: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    costs_cfg: dict,
) -> float:
    t = min(len(returns), len(current), len(baseline), len(mask))
    if t <= 0:
        return 0.0
    active = np.asarray(mask[:t], dtype=bool)
    if not np.any(active):
        return 0.0
    cost_kwargs = {
        "spread_bps": costs_cfg.get("spread_bps", 5.0),
        "fee_rate": costs_cfg.get("fee_rate", 0.0004),
        "slippage_bps": costs_cfg.get("slippage_bps", 2.0),
    }
    current_pnl = compute_pnl(np.asarray(returns[:t]), np.asarray(current[:t]), **cost_kwargs)
    baseline_pnl = compute_pnl(np.asarray(returns[:t]), np.asarray(baseline[:t]), **cost_kwargs)
    return float((current_pnl[active] - baseline_pnl[active]).sum())


def _masked_net_pnl(
    *,
    returns,
    positions: np.ndarray,
    mask: np.ndarray,
    costs_cfg: dict,
) -> float:
    t = min(len(returns), len(positions), len(mask))
    if t <= 0:
        return 0.0
    active = np.asarray(mask[:t], dtype=bool)
    if not np.any(active):
        return 0.0
    cost_kwargs = {
        "spread_bps": costs_cfg.get("spread_bps", 5.0),
        "fee_rate": costs_cfg.get("fee_rate", 0.0004),
        "slippage_bps": costs_cfg.get("slippage_bps", 2.0),
    }
    pnl = compute_pnl(np.asarray(returns[:t]), np.asarray(positions[:t]), **cost_kwargs)
    return float(pnl[active].sum())


def _component_diagnostics(
    *,
    actor,
    positions: np.ndarray,
    z,
    h,
    test_returns,
    test_regime_probs,
    test_advantage_values,
    device: str,
    costs_cfg: dict,
    cfg: dict,
    benchmark_positions_fn,
    benchmark_position: float,
    backtest_cls,
    action_stats_fn,
) -> dict:
    use_floor = bool(getattr(actor, "use_benchmark_exposure_floor", False))
    use_adapter = bool(getattr(actor, "use_benchmark_overweight_adapter", False))
    if not (use_floor or use_adapter):
        return {}

    variants: dict[str, np.ndarray] = {"current": positions}
    if use_floor:
        variants["no_floor"] = _predict_with_policy_flags(
            actor,
            z,
            h,
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
            use_floor=False,
            use_adapter=use_adapter,
        )
    if use_adapter:
        variants["no_adapter"] = _predict_with_policy_flags(
            actor,
            z,
            h,
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
            use_floor=use_floor,
            use_adapter=False,
        )
    if use_floor and use_adapter:
        variants["floor_only"] = _predict_with_policy_flags(
            actor,
            z,
            h,
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
            use_floor=True,
            use_adapter=False,
        )
        variants["adapter_only"] = _predict_with_policy_flags(
            actor,
            z,
            h,
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
            use_floor=False,
            use_adapter=True,
        )
        variants["neither"] = _predict_with_policy_flags(
            actor,
            z,
            h,
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
            use_floor=False,
            use_adapter=False,
        )

    uses_advantage_gate = (
        getattr(actor, "route_advantage_gate", None) is not None
        or int(getattr(actor, "benchmark_overweight_advantage_index", -1)) >= 0
    )
    if uses_advantage_gate:
        variants["gate_off"] = _predict_with_policy_flags(
            actor,
            z,
            h,
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
            use_floor=use_floor,
            use_adapter=use_adapter,
            route_advantage_gate_scale=0.0,
            overweight_advantage_index=-1,
        )

    result = {}
    for name, pos in variants.items():
        t = min(len(test_returns), len(pos))
        metrics = backtest_cls(
            test_returns[:t],
            pos[:t],
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            interval=cfg.get("data", {}).get("interval", "15m"),
            benchmark_positions=benchmark_positions_fn(t),
        ).run()
        stats = action_stats_fn(pos[:t], benchmark_position=benchmark_position)
        result[name] = {
            "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
            "sharpe_delta": float(metrics.sharpe_delta or 0.0),
            "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
            "turnover": float(stats["turnover"]),
            "long": float(stats["long"]),
            "short": float(stats["short"]),
            "flat": float(stats["flat"]),
        }

    current = variants["current"][: len(positions)]
    if "no_floor" in variants:
        delta = current - variants["no_floor"][: len(current)]
        active = delta > 1e-6
        result["floor_effect_rate"] = float(np.mean(delta > 1e-6))
        result["floor_mean_increment"] = float(np.mean(delta[delta > 1e-6])) if np.any(delta > 1e-6) else 0.0
        result["floor_incremental_net"] = _active_incremental_pnl(
            returns=test_returns,
            current=current,
            baseline=variants["no_floor"][: len(current)],
            mask=active,
            costs_cfg=costs_cfg,
        )
    if "no_adapter" in variants:
        delta = current - variants["no_adapter"][: len(current)]
        active = np.abs(delta) > 1e-6
        result["adapter_effect_rate"] = float(np.mean(np.abs(delta) > 1e-6))
        result["adapter_mean_abs_delta"] = float(np.mean(np.abs(delta[np.abs(delta) > 1e-6]))) if np.any(np.abs(delta) > 1e-6) else 0.0
        result["adapter_mean_delta"] = float(np.mean(delta[active])) if np.any(active) else 0.0
        result["adapter_positive_delta_rate"] = float(np.mean(delta[active] > 0.0)) if np.any(active) else 0.0
        result["adapter_long_state_rate"] = float(np.mean(current > (benchmark_position + 1e-6)))
        result["adapter_fire_net"] = _masked_net_pnl(
            returns=test_returns,
            positions=current,
            mask=active,
            costs_cfg=costs_cfg,
        )
        result["adapter_nonfire_net"] = _masked_net_pnl(
            returns=test_returns,
            positions=current,
            mask=~active,
            costs_cfg=costs_cfg,
        )
        result["adapter_incremental_net"] = _active_incremental_pnl(
            returns=test_returns,
            current=current,
            baseline=variants["no_adapter"][: len(current)],
            mask=active,
            costs_cfg=costs_cfg,
        )
    if "gate_off" in variants:
        delta = current - variants["gate_off"][: len(current)]
        active = np.abs(delta) > 1e-6
        result["adv_gate_effect_rate"] = float(np.mean(active))
        result["adv_gate_mean_abs_delta"] = float(np.mean(np.abs(delta[active]))) if np.any(active) else 0.0
        result["adv_gate_incremental_net"] = _active_incremental_pnl(
            returns=test_returns,
            current=current,
            baseline=variants["gate_off"][: len(current)],
            mask=active,
            costs_cfg=costs_cfg,
        )
    return result


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
    fold_idx: int | None = None,
) -> dict:
    print(f"\n[{log_ts()}] [Step 5] Test Backtest...")
    test_features = wfo_dataset.test_dataset().features.numpy()
    test_returns = wfo_dataset.test_returns

    enc_test = wm_trainer.encode_sequence(test_features, seq_len=seq_len)
    external_positions = _load_external_policy_positions(cfg, fold_idx, "test")
    use_external_policy = external_positions is not None
    if use_external_policy:
        positions = external_positions
    else:
        positions = actor.predict_positions(
            enc_test["z"],
            enc_test["h"],
            regime_np=test_regime_probs,
            advantage_np=test_advantage_values,
            device=device,
        )
    if not use_external_policy and getattr(actor, "route_head", None) is not None:
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
            recovery_gate = []
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
                if (
                    bool(getattr(actor, "use_inventory_recovery_controller", False))
                    and getattr(actor, "inventory_recovery_head", None) is not None
                ):
                    rec_logit = actor.inventory_recovery_logits(
                        z_t[i:i + 1],
                        h_t[i:i + 1],
                        inventory=controller_state,
                        regime=reg_i,
                        advantage=adv_i,
                    )
                    recovery_gate.append(float(torch.sigmoid(rec_logit).item()))
                pos_i = torch.tensor([[float(positions[i])]], dtype=torch.float32, device=dev)
                controller_state = actor.update_controller_state(controller_state, pos_i)
            if pred_routes:
                route_names = EXPOSURE_ROUTE_NAMES if getattr(actor, "route_dim", len(ROUTE_NAMES)) == 3 else ROUTE_NAMES
                counts = np.bincount(np.asarray(pred_routes, dtype=np.int64), minlength=len(route_names))
                rates = counts / max(counts.sum(), 1)
                route_dist = " ".join(f"{name}={rates[idx]:.0%}" for idx, name in enumerate(route_names))
                print(
                    f"  Route dist: {route_dist} "
                    f"active={1.0 - rates[0]:.1%} conf={np.mean(route_conf):.3f}"
                )
                if recovery_gate:
                    rec_arr = np.asarray(recovery_gate, dtype=np.float64)
                    print(
                        f"  Recovery gate: mean={rec_arr.mean():.3f} "
                        f"active={(rec_arr >= 0.5).mean():.1%}"
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
    policy_diagnostics = {}
    if not use_external_policy:
        policy_diagnostics = _component_diagnostics(
            actor=actor,
            positions=positions[:t_min],
            z=enc_test["z"],
            h=enc_test["h"],
            test_returns=test_returns,
            test_regime_probs=test_regime_probs,
            test_advantage_values=test_advantage_values,
            device=device,
            costs_cfg=costs_cfg,
            cfg=cfg,
            benchmark_positions_fn=benchmark_positions_fn,
            benchmark_position=benchmark_position,
            backtest_cls=backtest_cls,
            action_stats_fn=action_stats_fn,
        )

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
        print(f"  PeriodWin:{(getattr(metrics, 'period_win_rate_vs_bh', None) or 0.0):.1%}")
        if getattr(metrics, "upside_capture", None) is not None:
            print(
                f"  Capture:  up={metrics.upside_capture:.3f} "
                f"down={metrics.downside_capture if metrics.downside_capture is not None else float('nan'):.3f} "
                f"under_streak={metrics.max_underperformance_streak}"
            )
        print(f"  Score:    {format_m2_scorecard_fn(test_scorecard)}")
    if policy_diagnostics:
        print("  Component attribution:")
        for name in ("neither", "adapter_only", "floor_only", "gate_off", "current"):
            if name not in policy_diagnostics:
                continue
            rec = policy_diagnostics[name]
            print(
                f"    {name}: alpha={rec['alpha_excess_pt']:+.2f}pt "
                f"sharpeD={rec['sharpe_delta']:+.3f} "
                f"maxddD={rec['maxdd_delta_pt']:+.2f}pt "
                f"turnover={rec['turnover']:.2f} "
                f"long={rec['long']:.0%} short={rec['short']:.0%} flat={rec['flat']:.0%}"
            )
        effect_bits = []
        if "floor_effect_rate" in policy_diagnostics:
            effect_bits.append(
                f"floor_effect={policy_diagnostics['floor_effect_rate']:.1%}"
                f"/+{policy_diagnostics['floor_mean_increment']:.3f}"
                f"/pnl={policy_diagnostics['floor_incremental_net']:+.4f}"
            )
        if "adapter_effect_rate" in policy_diagnostics:
            effect_bits.append(
                f"adapter_effect={policy_diagnostics['adapter_effect_rate']:.1%}"
                f"/{policy_diagnostics['adapter_mean_abs_delta']:.3f}"
                f"/pnl={policy_diagnostics['adapter_incremental_net']:+.4f}"
            )
        if "adv_gate_effect_rate" in policy_diagnostics:
            effect_bits.append(
                f"adv_gate_effect={policy_diagnostics['adv_gate_effect_rate']:.1%}"
                f"/{policy_diagnostics['adv_gate_mean_abs_delta']:.3f}"
                f"/pnl={policy_diagnostics['adv_gate_incremental_net']:+.4f}"
            )
        if effect_bits:
            print(f"    effects: {' '.join(effect_bits)}")
        if "adapter_effect_rate" in policy_diagnostics:
            print(
                "    adapter_detail: "
                f"fire={policy_diagnostics['adapter_effect_rate']:.1%} "
                f"mean_delta={policy_diagnostics['adapter_mean_delta']:+.4f} "
                f"positive={policy_diagnostics['adapter_positive_delta_rate']:.1%} "
                f"long_state={policy_diagnostics['adapter_long_state_rate']:.1%} "
                f"fire_pnl={policy_diagnostics['adapter_fire_net']:+.4f} "
                f"nonfire_pnl={policy_diagnostics['adapter_nonfire_net']:+.4f}"
            )
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
        "policy_diagnostics": policy_diagnostics,
        "completed_stage": "test",
    }
