from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unidream.eval.backtest import compute_pnl


@dataclass(frozen=True)
class FireMetrics:
    alpha_excess_pt: float
    sharpe_delta: float
    maxdd_delta_pt: float
    turnover: float
    long: float
    short: float
    flat: float
    fire_rate: float
    fire_count: int
    mean_delta: float
    fire_pnl: float
    nonfire_pnl: float
    fwd_ret_16: float
    fwd_incr_pnl_16: float
    accepted: bool
    reject_reason: str | None
    score: float


def predict_with_policy_flags(
    actor,
    z,
    h,
    *,
    regime_np,
    advantage_np,
    device: str,
    use_floor: bool,
    use_adapter: bool,
    route_advantage_gate_scale: float | None = None,
    overweight_advantage_index: int | None = None,
) -> np.ndarray:
    """Predict positions while temporarily toggling policy safety components."""
    sentinel = object()
    saved = {
        "use_benchmark_exposure_floor": getattr(actor, "use_benchmark_exposure_floor", sentinel),
        "use_benchmark_overweight_adapter": getattr(actor, "use_benchmark_overweight_adapter", sentinel),
        "route_advantage_gate_scale": getattr(actor, "route_advantage_gate_scale", sentinel),
        "benchmark_overweight_advantage_index": getattr(actor, "benchmark_overweight_advantage_index", sentinel),
    }
    actor.use_benchmark_exposure_floor = use_floor
    actor.use_benchmark_overweight_adapter = use_adapter
    if route_advantage_gate_scale is not None:
        actor.route_advantage_gate_scale = route_advantage_gate_scale
    if overweight_advantage_index is not None:
        actor.benchmark_overweight_advantage_index = overweight_advantage_index
    try:
        return actor.predict_positions(
            z,
            h,
            regime_np=regime_np,
            advantage_np=advantage_np,
            device=device,
        )
    finally:
        for key, value in saved.items():
            if value is sentinel:
                try:
                    delattr(actor, key)
                except AttributeError:
                    pass
            else:
                setattr(actor, key, value)


def forward_return_mean(returns: np.ndarray, mask: np.ndarray, horizon: int) -> float:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    vals = []
    for i in idx:
        end = min(len(returns), i + horizon)
        if end > i:
            vals.append(float(np.sum(returns[i:end])))
    return float(np.mean(vals)) if vals else 0.0


def forward_incremental_mean(
    *,
    returns: np.ndarray,
    delta: np.ndarray,
    mask: np.ndarray,
    horizon: int,
) -> float:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    vals = []
    for i in idx:
        end = min(len(returns), i + horizon)
        if end > i:
            vals.append(float(delta[i] * np.sum(returns[i:end])))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_fire_metrics(
    *,
    actor,
    z,
    h,
    returns,
    regime_np,
    advantage_np,
    device: str,
    cfg: dict,
    costs_cfg: dict,
    benchmark_positions_fn,
    benchmark_position: float,
    backtest_cls,
    action_stats_fn,
    selector_cfg: dict,
) -> FireMetrics:
    positions = actor.predict_positions(
        z,
        h,
        regime_np=regime_np,
        advantage_np=advantage_np,
        device=device,
    )
    no_adapter = predict_with_policy_flags(
        actor,
        z,
        h,
        regime_np=regime_np,
        advantage_np=advantage_np,
        device=device,
        use_floor=bool(getattr(actor, "use_benchmark_exposure_floor", False)),
        use_adapter=False,
    )
    t = min(len(returns), len(positions), len(no_adapter))
    returns_arr = np.asarray(returns[:t], dtype=np.float64)
    positions_arr = np.asarray(positions[:t], dtype=np.float64)
    no_adapter_arr = np.asarray(no_adapter[:t], dtype=np.float64)
    delta = positions_arr - no_adapter_arr
    fire = np.abs(delta) > float(selector_cfg.get("fire_eps", 1e-6))

    metrics = backtest_cls(
        returns_arr,
        positions_arr,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=benchmark_positions_fn(t),
    ).run()
    stats = action_stats_fn(positions_arr, benchmark_position=benchmark_position)
    pnl = compute_pnl(
        returns_arr,
        positions_arr,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
    )
    horizon = int(selector_cfg.get("forward_horizon", 16))
    alpha_excess_pt = 100.0 * float(metrics.alpha_excess or 0.0)
    sharpe_delta = float(metrics.sharpe_delta or 0.0)
    maxdd_delta_pt = 100.0 * float(metrics.maxdd_delta or 0.0)
    fire_pnl = float(np.sum(pnl[fire])) if np.any(fire) else 0.0
    nonfire_pnl = float(np.sum(pnl[~fire])) if np.any(~fire) else 0.0
    fwd_ret = forward_return_mean(returns_arr, fire, horizon)
    fwd_incr = forward_incremental_mean(
        returns=returns_arr,
        delta=delta,
        mask=fire,
        horizon=horizon,
    )

    reject_reason = None
    if maxdd_delta_pt > float(selector_cfg.get("maxdd_delta_max_pt", 0.0)):
        reject_reason = f"maxdd>{float(selector_cfg.get('maxdd_delta_max_pt', 0.0)):+.2f}pt"
    elif alpha_excess_pt <= float(selector_cfg.get("alpha_floor_pt", 0.0)):
        reject_reason = f"alpha<={float(selector_cfg.get('alpha_floor_pt', 0.0)):+.2f}pt"
    elif sharpe_delta < float(selector_cfg.get("sharpe_floor", 0.0)):
        reject_reason = f"sharpe<{float(selector_cfg.get('sharpe_floor', 0.0)):+.3f}"
    elif fire_pnl <= float(selector_cfg.get("fire_pnl_floor", 0.0)):
        reject_reason = "fire_pnl<=0"
    elif fwd_ret <= float(selector_cfg.get("forward_return_floor", 0.0)):
        reject_reason = "fwd<=0"
    elif fwd_incr <= float(selector_cfg.get("forward_incremental_floor", 0.0)):
        reject_reason = "incr<=0"
    elif float(stats["long"]) > float(selector_cfg.get("long_max", 0.03)):
        reject_reason = f"long>{float(selector_cfg.get('long_max', 0.03)):.1%}"
    elif float(stats["turnover"]) > float(selector_cfg.get("turnover_max", 3.5)):
        reject_reason = f"turnover>{float(selector_cfg.get('turnover_max', 3.5)):.2f}"
    elif float(stats["short"]) > float(selector_cfg.get("short_max", 0.0)):
        reject_reason = f"short>{float(selector_cfg.get('short_max', 0.0)):.1%}"

    score = (
        alpha_excess_pt
        + 20.0 * sharpe_delta
        - 4.0 * max(0.0, maxdd_delta_pt)
        + 2.0 * max(0.0, -maxdd_delta_pt)
        + 30.0 * fire_pnl
        + 5000.0 * fwd_incr
        + 1000.0 * fwd_ret
        - 1.0 * float(stats["turnover"])
    )
    if reject_reason is not None:
        score -= 1000.0

    return FireMetrics(
        alpha_excess_pt=alpha_excess_pt,
        sharpe_delta=sharpe_delta,
        maxdd_delta_pt=maxdd_delta_pt,
        turnover=float(stats["turnover"]),
        long=float(stats["long"]),
        short=float(stats["short"]),
        flat=float(stats["flat"]),
        fire_rate=float(np.mean(fire)),
        fire_count=int(np.sum(fire)),
        mean_delta=float(np.mean(delta[fire])) if np.any(fire) else 0.0,
        fire_pnl=fire_pnl,
        nonfire_pnl=nonfire_pnl,
        fwd_ret_16=fwd_ret,
        fwd_incr_pnl_16=fwd_incr,
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        score=float(score),
    )


def format_fire_metrics(metrics: FireMetrics) -> str:
    label = (
        f"alpha={metrics.alpha_excess_pt:+.2f}pt sharpeD={metrics.sharpe_delta:+.3f} "
        f"maxddD={metrics.maxdd_delta_pt:+.2f}pt turnover={metrics.turnover:.2f} "
        f"long={metrics.long:.1%} short={metrics.short:.1%} flat={metrics.flat:.1%} "
        f"fire={metrics.fire_rate:.1%}/{metrics.fire_count} "
        f"mean_delta={metrics.mean_delta:+.4f} fire_pnl={metrics.fire_pnl:+.4f} "
        f"fwd16={metrics.fwd_ret_16:+.5f} incr16={metrics.fwd_incr_pnl_16:+.5f} "
        f"score={metrics.score:+.3f}"
    )
    if metrics.reject_reason is not None:
        label += f" reject={metrics.reject_reason}"
    return label
