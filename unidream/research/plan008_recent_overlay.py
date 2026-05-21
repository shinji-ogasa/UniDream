from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.experiments.runtime import load_config, resolve_costs, set_seed


EXPERIMENT_NAME = "plan008_recent_overlay_probe"


@dataclass(frozen=True)
class Plan008OverlayConfig:
    up_amp: float = 0.06
    down_amp: float = 0.06
    fast_window: int = 48
    slow_window: int = 192
    drawdown_window: int = 96
    up_fast_min: float = 0.002
    down_fast_max: float = 0.0
    slow_floor: float = -0.02
    drawdown_max: float = -0.05
    confirm_bars: int = 2
    hold_bars: int = 96
    cooldown_bars: int = 24
    min_position: float = 0.80
    max_position: float = 1.12


def _shifted_log_price(returns: np.ndarray) -> np.ndarray:
    log_price = np.cumsum(np.asarray(returns, dtype=np.float64))
    if len(log_price) == 0:
        return log_price
    return np.concatenate([[log_price[0]], log_price[:-1]])


def _rolling_past_momentum(prev_log_price: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(prev_log_price, dtype=np.float64)
    w = int(max(window, 1))
    lag = np.empty_like(x)
    if len(x) == 0:
        return lag
    lag[:w] = x[0]
    lag[w:] = x[:-w]
    return x - lag


def _rolling_past_drawdown(prev_log_price: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(prev_log_price, dtype=np.float64)
    out = np.zeros(len(x), dtype=np.float64)
    dq: deque[int] = deque()
    w = int(max(window, 1))
    for i, value in enumerate(x):
        while dq and dq[0] < i - w + 1:
            dq.popleft()
        while dq and x[dq[-1]] <= value:
            dq.pop()
        dq.append(i)
        out[i] = value - x[dq[0]]
    return out


def _confirmed(signal: np.ndarray, bars: int) -> np.ndarray:
    sig = np.asarray(signal, dtype=bool)
    n = int(max(bars, 1))
    if n <= 1 or len(sig) == 0:
        return sig
    counts = np.convolve(sig.astype(np.int16), np.ones(n, dtype=np.int16), mode="full")[: len(sig)]
    return counts >= n


def _event_hold(desired: np.ndarray, *, hold_bars: int, cooldown_bars: int, benchmark_position: float) -> np.ndarray:
    target = np.asarray(desired, dtype=np.float64)
    out = np.full(len(target), float(benchmark_position), dtype=np.float64)
    hold = max(int(hold_bars), 1)
    cooldown = max(int(cooldown_bars), 0)
    i = 0
    while i < len(target):
        if abs(float(target[i]) - float(benchmark_position)) <= 1e-12:
            i += 1
            continue
        end = min(len(target), i + hold)
        out[i:end] = float(target[i])
        i = end + cooldown
    return out


def apply_plan008_overlay(
    returns: np.ndarray,
    *,
    benchmark_position: float = 1.0,
    overlay_cfg: Plan008OverlayConfig = Plan008OverlayConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    ret = np.asarray(returns, dtype=np.float64)
    prev_log_price = _shifted_log_price(ret)
    fast = _rolling_past_momentum(prev_log_price, int(overlay_cfg.fast_window))
    slow = _rolling_past_momentum(prev_log_price, int(overlay_cfg.slow_window))
    drawdown = _rolling_past_drawdown(prev_log_price, int(overlay_cfg.drawdown_window))

    bull = (fast > float(overlay_cfg.up_fast_min)) & (slow > float(overlay_cfg.slow_floor))
    bear = (
        ((fast < float(overlay_cfg.down_fast_max)) & (slow < 0.0))
        | ((drawdown < float(overlay_cfg.drawdown_max)) & (fast < 0.0))
    )

    bull = _confirmed(bull, int(overlay_cfg.confirm_bars))
    bear = _confirmed(bear, int(overlay_cfg.confirm_bars))
    desired = np.full(len(ret), float(benchmark_position), dtype=np.float64)
    desired[bull] = float(benchmark_position) + float(overlay_cfg.up_amp)
    desired[bear] = float(benchmark_position) - float(overlay_cfg.down_amp)
    desired[bull & bear] = float(benchmark_position)
    desired = np.clip(desired, float(overlay_cfg.min_position), float(overlay_cfg.max_position))
    positions = _event_hold(
        desired,
        hold_bars=int(overlay_cfg.hold_bars),
        cooldown_bars=int(overlay_cfg.cooldown_bars),
        benchmark_position=float(benchmark_position),
    )
    overlay = positions - float(benchmark_position)
    changes = np.flatnonzero(np.abs(np.diff(positions, prepend=positions[0] if len(positions) else benchmark_position)) > 1e-12)
    diag = {
        "plan008_overlay_version": "recent_overlay_v2",
        "fast_window": int(overlay_cfg.fast_window),
        "slow_window": int(overlay_cfg.slow_window),
        "drawdown_window": int(overlay_cfg.drawdown_window),
        "bull_rate": float(np.mean(bull)) if len(bull) else 0.0,
        "bear_rate": float(np.mean(bear)) if len(bear) else 0.0,
        "active_rate": float(np.mean(np.abs(overlay) > 0.05)) if len(overlay) else 0.0,
        "turnover": float(np.abs(np.diff(overlay)).sum()) if len(overlay) > 1 else 0.0,
        "last_position": float(positions[-1]) if len(positions) else None,
        "last_changes": [int(x) for x in changes[-10:]],
    }
    return positions.astype(np.float32), diag


def _read_returns(path: str) -> pd.Series:
    data = pd.read_parquet(path).squeeze()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{path} must have a DatetimeIndex")
    return data.sort_index()


def run_recent_probe(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    returns = _read_returns(args.returns)
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    overlay_cfg = Plan008OverlayConfig()
    _positions, diag = apply_plan008_overlay(
        returns.to_numpy(dtype=np.float64),
        benchmark_position=benchmark_position,
        overlay_cfg=overlay_cfg,
    )
    windows = [int(x) for x in str(args.windows).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for days in windows:
        mask = returns.index >= returns.index.max() - pd.Timedelta(days=int(days))
        ret_w = returns.loc[mask].to_numpy(dtype=np.float64)
        pos_w, diag_w = apply_plan008_overlay(
            ret_w,
            benchmark_position=benchmark_position,
            overlay_cfg=overlay_cfg,
        )
        stress = _stress_metrics(
            returns=ret_w,
            positions=pos_w,
            cfg=cfg,
            costs_cfg=cfg.get("costs", {}),
            benchmark_position=benchmark_position,
        )
        rows.append(
            {
                "days": int(days),
                "start": str(returns.loc[mask].index.min()),
                "end": str(returns.loc[mask].index.max()),
                "diag": diag_w,
                "stress": stress,
            }
        )
    return {
        "experiment": EXPERIMENT_NAME,
        "returns": args.returns,
        "period": {
            "start": str(returns.index.min()),
            "end": str(returns.index.max()),
            "n_bars": int(len(returns)),
        },
        "overlay_config": asdict(overlay_cfg),
        "diag": diag,
        "rows": rows,
        "leak_note": "Parameters are tuned on recent demo data. Signals are shifted trailing-return features only, but this is not a pristine holdout result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan008_recent_overlay_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--returns",
        default="checkpoints/data_cache/BTCUSDT_15m_2024-01-01_2026-05-21_z60_v2_plan008_latest_returns.parquet",
    )
    parser.add_argument("--windows", default="30,45,60,90,180,365")
    parser.add_argument("--output-json", default="docs_local/20260521_plan008_recent_overlay_latest.json")
    args = parser.parse_args()
    payload = run_recent_probe(args)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    print(f"[Plan008] wrote {args.output_json}")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        print(
            f"[Plan008] {row['days']}d alpha={m['alpha_excess_pt']:+.2f} "
            f"dd={m['maxdd_delta_pt']:+.2f} turnover={m['turnover']:.2f}"
        )


if __name__ == "__main__":
    main()
