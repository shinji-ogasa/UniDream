"""Plot Plan011 fold trades and equity curves from saved checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unidream.cli.train import (
    _action_stats,
    _benchmark_position_value,
    _candidate_to_text,
    _select_policy_candidate,
    _selector_candidate,
    _selector_cfg,
)
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest, compute_pnl
from unidream.experiments.checkpoint_eval import load_actor_state_checkpoint, load_fold_model_context
from unidream.experiments.run_config import configure_determinism
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.val_selector_stage import run_val_selector_stage
from unidream.experiments.wfo_runtime import build_wfo_splits, select_configured_wfo_splits


@dataclass(frozen=True)
class FoldPlotResult:
    fold: int
    test_start: str
    test_end: str
    metrics: dict[str, float]
    selected_checkpoint: str
    selected_scale: float
    figure_path: Path
    trade_count: int
    active_blocks: int


def _parse_folds(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return default
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("--folds did not contain any fold ids")
    return tuple(dict.fromkeys(values))


def _benchmark_positions(length: int, benchmark: float) -> np.ndarray:
    return np.full(length, benchmark, dtype=np.float64)


def _test_index(features: pd.DataFrame, split) -> pd.DatetimeIndex:
    mask = (features.index >= split.test_start) & (features.index <= split.test_end)
    return pd.DatetimeIndex(features.index[mask])


def _equity_series(
    returns: np.ndarray,
    positions: np.ndarray,
    benchmark: float,
    costs_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = min(len(returns), len(positions))
    returns_t = np.asarray(returns[:t], dtype=np.float64)
    positions_t = np.asarray(positions[:t], dtype=np.float64)
    bench_positions = _benchmark_positions(t, benchmark)
    pnl = compute_pnl(
        returns_t,
        positions_t,
        spread_bps=float(costs_cfg.get("spread_bps", 5.0)),
        fee_rate=float(costs_cfg.get("fee_rate", 0.0004)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 2.0)),
    )
    bench_pnl = compute_pnl(
        returns_t,
        bench_positions,
        spread_bps=float(costs_cfg.get("spread_bps", 5.0)),
        fee_rate=float(costs_cfg.get("fee_rate", 0.0004)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 2.0)),
    )
    return returns_t, positions_t, np.exp(np.cumsum(pnl)), np.exp(np.cumsum(bench_pnl))


def trade_indices(positions: np.ndarray, trade_eps: float) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return np.zeros(0, dtype=np.int64)
    delta = np.diff(positions, prepend=positions[0])
    return np.flatnonzero(np.abs(delta) > float(trade_eps)).astype(np.int64)


def active_blocks(positions: np.ndarray, benchmark: float, active_eps: float) -> list[tuple[int, int]]:
    active = np.abs(np.asarray(positions, dtype=np.float64) - float(benchmark)) > float(active_eps)
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_active in enumerate(active):
        if bool(is_active) and start is None:
            start = idx
        elif not bool(is_active) and start is not None:
            blocks.append((start, idx - 1))
            start = None
    if start is not None:
        blocks.append((start, len(active) - 1))
    return blocks


def _metrics_record(
    returns: np.ndarray,
    positions: np.ndarray,
    cfg: dict[str, Any],
    benchmark: float,
) -> dict[str, float]:
    t = min(len(returns), len(positions))
    metrics = Backtest(
        np.asarray(returns[:t], dtype=np.float64),
        np.asarray(positions[:t], dtype=np.float64),
        spread_bps=float(cfg["costs"].get("spread_bps", 5.0)),
        fee_rate=float(cfg["costs"].get("fee_rate", 0.0004)),
        slippage_bps=float(cfg["costs"].get("slippage_bps", 2.0)),
        interval=str(cfg.get("data", {}).get("interval", "15m")),
        benchmark_positions=_benchmark_positions(t, benchmark),
    ).run()
    stats = _action_stats(positions[:t], benchmark_position=benchmark)
    total_return_dec = float(metrics.total_return)
    bench_return_dec = float(metrics.benchmark_total_return or 0.0)
    return {
        "alpha_excess_pt": 100.0 * float(metrics.alpha_excess or 0.0),
        "final_excess_pt": 100.0 * (total_return_dec - bench_return_dec),
        "sharpe_delta": float(metrics.sharpe_delta or 0.0),
        "maxdd_delta_pt": 100.0 * float(metrics.maxdd_delta or 0.0),
        "total_return_pt": 100.0 * total_return_dec,
        "benchmark_total_return_pt": 100.0 * bench_return_dec,
        "max_drawdown_pt": 100.0 * abs(float(metrics.max_drawdown)),
        "benchmark_max_drawdown_pt": 100.0 * abs(float(metrics.benchmark_max_drawdown or 0.0)),
        "turnover": float(stats["turnover"]),
        "active_rate": float(1.0 - stats["flat"]),
        "mean_overlay": float(stats["mean"]),
        "avg_hold_bars": float(stats["avg_hold"]),
    }


def _plot_fold(
    *,
    times: pd.DatetimeIndex,
    positions: np.ndarray,
    strategy_equity: np.ndarray,
    benchmark_equity: np.ndarray,
    benchmark: float,
    metrics: dict[str, float],
    fold: int,
    output_path: Path,
    trade_eps: float,
    active_eps: float,
) -> tuple[int, int]:
    trade_idx = trade_indices(positions, trade_eps)
    pos_delta = np.diff(positions, prepend=positions[0])
    blocks = active_blocks(positions, benchmark, active_eps)

    fig, (ax_eq, ax_pos) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    fig.suptitle(
        (
            f"Plan011 v31 fold {fold} test | "
            f"FinalExcess {metrics['final_excess_pt']:+.2f}pt | "
            f"MaxDDDelta {metrics['maxdd_delta_pt']:+.2f}pt | "
            f"turnover {metrics['turnover']:.2f}"
        ),
        fontsize=13,
    )

    for start, end in blocks:
        ax_eq.axvspan(times[start], times[end], color="#8a8f98", alpha=0.08, linewidth=0, zorder=1)

    ax_eq.plot(times, benchmark_equity, color="#6b7280", linewidth=1.8, label="B&H equity", zorder=2)
    ax_eq.plot(times, strategy_equity, color="#2563eb", linewidth=1.9, label="Strategy equity", zorder=3)

    if len(trade_idx) > 0:
        up_idx = trade_idx[pos_delta[trade_idx] > 0]
        down_idx = trade_idx[pos_delta[trade_idx] < 0]
        if len(up_idx) > 0:
            ax_eq.scatter(
                times[up_idx],
                strategy_equity[up_idx],
                marker="^",
                s=32,
                color="#16a34a",
                alpha=0.9,
                label="position up",
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
            )
        if len(down_idx) > 0:
            ax_eq.scatter(
                times[down_idx],
                strategy_equity[down_idx],
                marker="v",
                s=32,
                color="#dc2626",
                alpha=0.9,
                label="position down",
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
            )

    ax_eq.set_ylabel("Equity multiple")
    ax_eq.grid(True, alpha=0.25)
    ax_eq.legend(loc="best", fontsize=9)

    ax_pos.plot(times, positions, color="#111827", linewidth=1.2, label="exposure")
    ax_pos.axhline(benchmark, color="#6b7280", linestyle="--", linewidth=1.0, label="B&H=1.0")
    ax_pos.fill_between(
        times,
        benchmark,
        positions,
        where=positions >= benchmark,
        color="#16a34a",
        alpha=0.14,
        interpolate=True,
    )
    ax_pos.fill_between(
        times,
        benchmark,
        positions,
        where=positions < benchmark,
        color="#dc2626",
        alpha=0.14,
        interpolate=True,
    )
    ax_pos.set_ylabel("Exposure")
    ax_pos.set_xlabel("Test time")
    ax_pos.grid(True, alpha=0.25)
    ax_pos.legend(loc="best", fontsize=9)
    ax_pos.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax_pos.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_pos.xaxis.get_major_locator()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return int(len(trade_idx)), int(len(blocks))


def _write_markdown(results: list[FoldPlotResult], output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan011 v31 Fold0-12 Trade Charts",
        "",
        "保存済み checkpoint から実モデル推論を再実行し、test split の資産推移、B&H、position 変更点を可視化した。",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python -m unidream.cli.plot_plan011_fold_trades \\",
        f"  --config {payload['config']} \\",
        f"  --checkpoint-dir {payload['checkpoint_dir']} \\",
        "  --folds 0-12 \\",
        "  --seed 7 \\",
        "  --device cpu \\",
        f"  --output-dir {output_dir.as_posix()}",
        "```",
        "",
        "## Files",
        "",
        f"- metrics: `{(output_dir / 'metrics.csv').as_posix()}`",
        f"- trades: `{(output_dir / 'trades.csv').as_posix()}`",
        f"- compressed time series: `{(output_dir / 'timeseries.npz').as_posix()}`",
        f"- trade_eps: `{payload['trade_eps']}`",
        f"- active_eps: `{payload['active_eps']}`",
        "",
        "## Fold Summary",
        "",
        "| fold | period | AlphaEx | MaxDDDelta | turnover | trade points | active blocks | chart |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        fig_name = result.figure_path.name
        lines.append(
            f"| {result.fold} | {result.test_start} to {result.test_end} | "
            f"{result.metrics['alpha_excess_pt']:+.2f}pt | "
            f"{result.metrics['maxdd_delta_pt']:+.2f}pt | "
            f"{result.metrics['turnover']:.2f} | "
            f"{result.trade_count} | {result.active_blocks} | "
            f"[{fig_name}]({fig_name}) |"
        )
    lines.extend([
        "",
        "MaxDDDelta は strategy の絶対MaxDD minus B&Hの絶対MaxDD。マイナスが改善。",
        "取引点は exposure が前バーから `trade_eps` 以上変化したバー。active block は `abs(exposure - 1.0) > active_eps` の連続区間。",
        "全バーのposition系列は `timeseries.npz` に保存している。",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--folds", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--actor-checkpoint", default="ac.pt", choices=("ac.pt", "ac_best.pt"))
    parser.add_argument("--output-dir", default="docs/figures/plan011_v31_folds0_12")
    parser.add_argument("--trade-eps", type=float, default=5e-4)
    parser.add_argument("--active-eps", type=float, default=0.005)
    args = parser.parse_args()

    configure_determinism(args.seed)
    set_seed(args.seed)
    cfg, cost_profile = resolve_costs(load_config(args.config))
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    configured_folds = tuple(int(value) for value in run_cfg.get("folds") or ())
    folds = _parse_folds(args.folds, configured_folds)
    checkpoint_dir = Path(args.checkpoint_dir or cfg.get("logging", {}).get("checkpoint_dir", "checkpoints"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol = str(data_cfg["symbol"])
    interval = str(data_cfg["interval"])
    zscore_window = int(cfg["normalization"]["zscore_window_days"])
    cache_tag = f"{symbol}_{interval}_{run_cfg['start']}_{run_cfg['end']}_z{zscore_window}_v2"
    features, returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=str(run_cfg["start"]),
        end=str(run_cfg["end"]),
        zscore_window=zscore_window,
        cache_dir=str(cfg["logging"]["cache_dir"]),
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg["include_funding"]),
        include_oi=bool(data_cfg["include_oi"]),
        include_mark=bool(data_cfg["include_mark"]),
    )
    splits, fold_ids = select_configured_wfo_splits(build_wfo_splits(features, data_cfg), folds)
    benchmark = _benchmark_position_value(cfg)

    metrics_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    timeseries_payload: dict[str, np.ndarray] = {}
    plot_results: list[FoldPlotResult] = []

    for split in splits:
        fold_idx = int(split.fold_idx)
        print(f"\n[Plot] fold={fold_idx} test={split.test_start} -> {split.test_end}")
        dataset = WFODataset(features, returns, split, seq_len=int(data_cfg["seq_len"]))
        context = load_fold_model_context(
            fold_idx=fold_idx,
            dataset=dataset,
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            device=args.device,
            benchmark_position=benchmark,
        )

        actor_ckpt = checkpoint_dir / f"fold_{fold_idx}" / args.actor_checkpoint
        if not actor_ckpt.exists() and args.actor_checkpoint == "ac.pt":
            actor_ckpt = checkpoint_dir / f"fold_{fold_idx}" / "ac_best.pt"
        if not actor_ckpt.exists():
            raise FileNotFoundError(f"missing actor checkpoint: {actor_ckpt}")
        load_actor_state_checkpoint(context["actor"], actor_ckpt, args.device)

        run_val_selector_stage(
            actor=context["actor"],
            wm_trainer=context["wm_trainer"],
            wfo_dataset=dataset,
            seq_len=int(data_cfg["seq_len"]),
            val_regime_probs=context["fold_inputs"]["val_regime_probs"],
            val_advantage_values=context["val_advantage"],
            device=args.device,
            cfg=cfg,
            ac_cfg=cfg["ac"],
            costs_cfg=cfg["costs"],
            backtest_cls=Backtest,
            action_stats_fn=_action_stats,
            selector_cfg_fn=_selector_cfg,
            selector_candidate_fn=_selector_candidate,
            select_policy_candidate_fn=_select_policy_candidate,
            candidate_to_text_fn=_candidate_to_text,
            benchmark_positions_fn=lambda length, bench=benchmark: _benchmark_positions(length, bench),
            benchmark_position=benchmark,
        )

        positions = context["actor"].predict_positions(
            context["encoded_test"]["z"],
            context["encoded_test"]["h"],
            regime_np=context["fold_inputs"]["test_regime_probs"],
            advantage_np=context["test_advantage"],
            device=args.device,
        )
        returns_t, positions_t, strategy_equity, benchmark_equity = _equity_series(
            dataset.test_returns,
            positions,
            benchmark,
            cfg["costs"],
        )
        times = _test_index(features, split)[: len(positions_t)]
        if len(times) != len(positions_t):
            raise RuntimeError(f"fold {fold_idx} time/position length mismatch: {len(times)} vs {len(positions_t)}")
        metrics = _metrics_record(returns_t, positions_t, cfg, benchmark)
        figure_path = output_dir / f"fold_{fold_idx:02d}_equity_trades.png"
        trade_count, block_count = _plot_fold(
            times=times,
            positions=positions_t,
            strategy_equity=strategy_equity,
            benchmark_equity=benchmark_equity,
            benchmark=benchmark,
            metrics=metrics,
            fold=fold_idx,
            output_path=figure_path,
            trade_eps=float(args.trade_eps),
            active_eps=float(args.active_eps),
        )

        result = FoldPlotResult(
            fold=fold_idx,
            test_start=str(split.test_start),
            test_end=str(split.test_end),
            metrics=metrics,
            selected_checkpoint=str(actor_ckpt),
            selected_scale=float(getattr(context["actor"], "infer_adjust_rate_scale", 1.0)),
            figure_path=figure_path,
            trade_count=trade_count,
            active_blocks=block_count,
        )
        plot_results.append(result)

        row = {
            "fold": fold_idx,
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
            "checkpoint": str(actor_ckpt),
            "selected_scale": result.selected_scale,
            "trade_count": trade_count,
            "active_blocks": block_count,
            **metrics,
        }
        metrics_rows.append(row)

        trade_idx = trade_indices(positions_t, float(args.trade_eps))
        deltas = np.diff(positions_t, prepend=positions_t[0])
        for idx in trade_idx:
            prev_pos = float(positions_t[idx - 1]) if idx > 0 else float(benchmark)
            trade_rows.append({
                "fold": fold_idx,
                "timestamp": str(times[idx]),
                "delta": float(deltas[idx]),
                "position_before": prev_pos,
                "position_after": float(positions_t[idx]),
                "strategy_equity": float(strategy_equity[idx]),
                "bh_equity": float(benchmark_equity[idx]),
            })

        key = f"fold_{fold_idx:02d}"
        timeseries_payload[f"{key}_time_ns"] = times.view("int64")
        timeseries_payload[f"{key}_returns"] = returns_t.astype(np.float32)
        timeseries_payload[f"{key}_positions"] = positions_t.astype(np.float32)
        timeseries_payload[f"{key}_strategy_equity"] = strategy_equity.astype(np.float32)
        timeseries_payload[f"{key}_bh_equity"] = benchmark_equity.astype(np.float32)

        print(
            f"[Plot] fold={fold_idx} alpha={metrics['alpha_excess_pt']:+.2f}pt "
            f"maxddD={metrics['maxdd_delta_pt']:+.2f}pt trades={trade_count} "
            f"chart={figure_path}"
        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_rows)
    with (output_dir / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["fold", "timestamp", "delta", "position_before", "position_after", "strategy_equity", "bh_equity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trade_rows)
    np.savez_compressed(output_dir / "timeseries.npz", **timeseries_payload)

    payload = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "actor_checkpoint": args.actor_checkpoint,
        "seed": args.seed,
        "device": args.device,
        "cost_profile": cost_profile,
        "costs": cfg["costs"],
        "benchmark_position": benchmark,
        "folds": fold_ids,
        "trade_eps": float(args.trade_eps),
        "active_eps": float(args.active_eps),
        "results": [
            {
                "fold": result.fold,
                "test_start": result.test_start,
                "test_end": result.test_end,
                "metrics": result.metrics,
                "checkpoint": result.selected_checkpoint,
                "selected_scale": result.selected_scale,
                "figure": str(result.figure_path),
                "trade_count": result.trade_count,
                "active_blocks": result.active_blocks,
            }
            for result in plot_results
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(plot_results, output_dir, payload)
    print(f"\n[Plot] wrote {output_dir / 'README.md'}")
    print(f"[Plot] wrote {output_dir / 'metrics.csv'}")
    print(f"[Plot] wrote {output_dir / 'trades.csv'}")
    print(f"[Plot] wrote {output_dir / 'timeseries.npz'}")


if __name__ == "__main__":
    main()
