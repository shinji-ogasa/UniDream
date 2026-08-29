"""バックテストモジュール.

コスト・スリッページモデル込みのバックテスト実装。
Sharpe / Sortino / MaxDD / Calmar を計算する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# 暗号資産は 24h/365d 取引可能 → 365 日で年換算
# 株式は 252 営業日
ANNUALIZATION_CRYPTO = {
    "1m": 365 * 1440,
    "5m": 365 * 288,
    "15m": 365 * 96,
    "30m": 365 * 48,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}
ANNUALIZATION_EQUITY = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "1h": 252 * 6.5,
    "4h": 252 * 1.625,
    "1d": 252,
}
# デフォルトは暗号資産（BTCUSDT 対象のため）
ANNUALIZATION = ANNUALIZATION_CRYPTO


def align_execution_path(
    returns: np.ndarray,
    positions: np.ndarray,
    benchmark_positions: np.ndarray | None = None,
    execution_delay_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Align decisions with returns that are available after execution delay.

    A delay of ``d > 0`` evaluates only the right-aligned window
    ``positions[:-d]`` against ``returns[d:]``.  The leading returns have no
    decision yet, and the final ``d`` decisions have no return in the
    requested window, so neither is padded or scored.  Benchmark positions
    are trimmed to the same return window.  Invalid delays are rejected
    explicitly rather than producing an empty or fabricated path.
    """
    returns_arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    positions_arr = np.asarray(positions, dtype=np.float64).reshape(-1)
    if len(returns_arr) != len(positions_arr):
        raise ValueError("returns and positions must have equal lengths")
    benchmark_arr = None
    if benchmark_positions is not None:
        benchmark_arr = np.asarray(benchmark_positions, dtype=np.float64).reshape(-1)
        if len(benchmark_arr) != len(returns_arr):
            raise ValueError("benchmark_positions and returns must have equal lengths")

    delay = int(execution_delay_bars)
    if delay < 0:
        raise ValueError("execution_delay_bars must be non-negative")
    if delay >= len(returns_arr):
        raise ValueError(
            "execution_delay_bars must be smaller than the number of return bars"
        )
    if delay == 0:
        return returns_arr, positions_arr, benchmark_arr
    benchmark_window = benchmark_arr[delay:] if benchmark_arr is not None else None
    return returns_arr[delay:], positions_arr[:-delay], benchmark_window


@dataclass
class BacktestMetrics:
    """バックテスト結果メトリクス."""
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    total_return: float
    annual_return: float
    n_trades: int
    avg_holding: float
    equity_curve: np.ndarray = field(repr=False)
    pnl_series: np.ndarray = field(repr=False)
    benchmark_total_return: float | None = None
    benchmark_annual_return: float | None = None
    benchmark_sharpe: float | None = None
    benchmark_max_drawdown: float | None = None
    final_excess: float | None = None
    alpha_excess: float | None = None
    annual_alpha_excess: float | None = None
    sharpe_delta: float | None = None
    maxdd_delta: float | None = None
    win_rate_vs_bh: float | None = None
    period_win_rate_vs_bh: float | None = None
    upside_capture: float | None = None
    downside_capture: float | None = None
    max_underperformance_streak: int | None = None

    def to_dict(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "n_trades": self.n_trades,
            "avg_holding": self.avg_holding,
            "benchmark_total_return": self.benchmark_total_return,
            "benchmark_annual_return": self.benchmark_annual_return,
            "benchmark_sharpe": self.benchmark_sharpe,
            "benchmark_max_drawdown": self.benchmark_max_drawdown,
            "final_excess": self.final_excess,
            "alpha_excess": self.alpha_excess,
            "annual_alpha_excess": self.annual_alpha_excess,
            "sharpe_delta": self.sharpe_delta,
            "maxdd_delta": self.maxdd_delta,
            "win_rate_vs_bh": self.win_rate_vs_bh,
            "period_win_rate_vs_bh": self.period_win_rate_vs_bh,
            "upside_capture": self.upside_capture,
            "downside_capture": self.downside_capture,
            "max_underperformance_streak": self.max_underperformance_streak,
        }


def compute_costs(
    positions: np.ndarray,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
) -> np.ndarray:
    """各ステップのトランザクションコストを計算する.

    Args:
        positions: ポジション比率列 (T,) ∈ {-1, -0.5, 0, 0.5, 1}
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ (basis points)

    Returns:
        コスト列 (T,)
    """
    delta_pos = np.abs(np.diff(positions, prepend=0.0))
    spread_cost = (spread_bps / 10000) / 2 * delta_pos
    fee_cost = fee_rate * delta_pos
    slippage_cost = (slippage_bps / 10000) * delta_pos
    return spread_cost + fee_cost + slippage_cost


def compute_pnl(
    returns: np.ndarray,
    positions: np.ndarray,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
) -> np.ndarray:
    """コスト控除後の PnL 系列を計算する.

    Args:
        returns: 対数リターン列 (T,)
        positions: ポジション比率列 (T,)

    Returns:
        コスト控除後の PnL 列 (T,)
    """
    gross_pnl = positions * returns
    costs = compute_costs(positions, spread_bps, fee_rate, slippage_bps)
    return gross_pnl - costs


def compute_sharpe(pnl: np.ndarray, ann_factor: float) -> float:
    """年換算 Sharpe Ratio を計算する."""
    if pnl.std() < 1e-10:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(ann_factor))


def compute_sortino(pnl: np.ndarray, ann_factor: float) -> float:
    """年換算 Sortino Ratio を計算する.

    分母は標準的な下方偏差 sqrt(mean(min(pnl, 0)^2))（target=0、全期間で平均）。
    負リターンのみの標本標準偏差を使う定義は外部と比較できないため使わない。
    """
    downside_dev = float(np.sqrt(np.mean(np.square(np.minimum(pnl, 0.0)))))
    if downside_dev < 1e-10:
        return 99.0 if pnl.mean() > 0 else 0.0
    return float(pnl.mean() / downside_dev * np.sqrt(ann_factor))


def compute_max_drawdown(equity: np.ndarray) -> float:
    """最大ドローダウンを計算する（0〜1 の比率）."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-8)
    return float(drawdown.min())


def compute_calmar(total_return: float, max_dd: float, period_years: float = 1.0) -> float:
    """Calmar Ratio を計算する.

    分子は annual_return と同じ幾何年率換算（算術換算だと他指標と不整合になる）。
    """
    if abs(max_dd) < 1e-10:
        return 99.0 if total_return > 0 else 0.0
    return float(compute_annual_return(total_return, period_years) / abs(max_dd))


def compute_annual_return(total_return: float, period_years: float) -> float:
    """累積リターンから年率リターンを計算する."""
    if period_years <= 0:
        return 0.0
    equity_end = max(1e-12, 1.0 + total_return)
    return float(equity_end ** (1.0 / period_years) - 1.0)


def compute_period_win_rate(pnl: np.ndarray, benchmark_pnl: np.ndarray, period_bars: int) -> float:
    if len(pnl) == 0:
        return 0.0
    period_bars = max(int(period_bars), 1)
    wins = []
    for start in range(0, len(pnl), period_bars):
        rel = float(pnl[start:start + period_bars].sum() - benchmark_pnl[start:start + period_bars].sum())
        wins.append(rel > 0.0)
    return float(np.mean(wins)) if wins else 0.0


def compute_capture_ratio(pnl: np.ndarray, benchmark_pnl: np.ndarray, positive_benchmark: bool) -> float | None:
    mask = benchmark_pnl > 0.0 if positive_benchmark else benchmark_pnl < 0.0
    if not np.any(mask):
        return None
    denom = float(benchmark_pnl[mask].sum())
    if abs(denom) < 1e-12:
        return None
    return float(pnl[mask].sum() / denom)


def max_consecutive_underperformance(pnl: np.ndarray, benchmark_pnl: np.ndarray) -> int:
    rel_under = pnl <= benchmark_pnl
    longest = 0
    current = 0
    for under in rel_under:
        if bool(under):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


class Backtest:
    """バックテスト実行クラス.

    Args:
        returns: 対数リターン列 (T,)
        positions: ポジション比率列 (T,) ∈ {-1, -0.5, 0, 0.5, 1}
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ (basis points)
        interval: 足種（年換算係数計算に使用）
        execution_delay_bars: 決定から執行までの遅延バー数（感度分析用、デフォルト 0）。
            d > 0 では positions[:-d] と returns[d:] を右整列し、未予測の
            先頭リターンと期間外の末尾決定を評価しない。benchmark も同じ期間に切り詰める。
    """

    def __init__(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        spread_bps: float = 5.0,
        fee_rate: float = 0.0004,
        slippage_bps: float = 2.0,
        interval: str = "15m",
        benchmark_positions: np.ndarray | None = None,
        execution_delay_bars: int = 0,
    ):
        assert len(returns) == len(positions), "returns と positions の長さが一致しない"
        self.returns = np.asarray(returns, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.spread_bps = spread_bps
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.ann_factor = ANNUALIZATION.get(interval, 252 * 96)
        self.execution_delay_bars = int(execution_delay_bars)
        self.benchmark_positions = (
            np.asarray(benchmark_positions, dtype=np.float64)
            if benchmark_positions is not None else None
        )
        if self.benchmark_positions is not None:
            assert len(self.benchmark_positions) == len(self.positions), (
                "benchmark_positions と positions の長さが一致しない"
            )
        if self.execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be non-negative")
        if self.execution_delay_bars >= len(self.returns):
            raise ValueError(
                "execution_delay_bars must be smaller than the number of return bars"
            )

    def run(self) -> BacktestMetrics:
        """バックテストを実行してメトリクスを返す."""
        returns, positions, benchmark_positions = align_execution_path(
            self.returns,
            self.positions,
            self.benchmark_positions,
            self.execution_delay_bars,
        )
        pnl = compute_pnl(returns, positions, self.spread_bps, self.fee_rate, self.slippage_bps)
        # position * log_return ≈ log(1 + position * simple_return) for small returns
        # 15分足では十分な近似。厳密な対数リターンは position=1.0 の場合のみ。
        equity = np.exp(np.cumsum(pnl))  # 累積 PnL → equity curve

        # equity[-1] = exp(sum(log_returns)) なので、実リターン = equity[-1] - 1.0
        total_return = float(equity[-1] - 1.0)
        sharpe = compute_sharpe(pnl, self.ann_factor)
        sortino = compute_sortino(pnl, self.ann_factor)
        max_dd = compute_max_drawdown(equity)

        period_years = len(pnl) / self.ann_factor
        annual_return = compute_annual_return(total_return, period_years)
        calmar = compute_calmar(total_return, max_dd, period_years)

        # トレード数・平均保有期間
        pos_changes = np.diff(positions, prepend=0.0) != 0
        n_trades = int(pos_changes.sum())

        # 連続して同じポジションを保持した期間の平均
        holding_lengths = []
        current_len = 1
        for i in range(1, len(positions)):
            if positions[i] == positions[i - 1]:
                current_len += 1
            else:
                holding_lengths.append(current_len)
                current_len = 1
        holding_lengths.append(current_len)
        avg_holding = float(np.mean(holding_lengths)) if holding_lengths else 0.0

        benchmark_total_return = None
        benchmark_annual_return = None
        benchmark_sharpe = None
        benchmark_max_drawdown = None
        final_excess = None
        alpha_excess = None
        annual_alpha_excess = None
        sharpe_delta = None
        maxdd_delta = None
        win_rate_vs_bh = None
        period_win_rate_vs_bh = None
        upside_capture = None
        downside_capture = None
        max_underperformance_streak = None
        if benchmark_positions is not None:
            bench_pnl = compute_pnl(
                returns,
                benchmark_positions,
                self.spread_bps,
                self.fee_rate,
                self.slippage_bps,
            )
            bench_equity = np.exp(np.cumsum(bench_pnl))
            benchmark_total_return = float(bench_equity[-1] - 1.0)
            benchmark_annual_return = compute_annual_return(benchmark_total_return, period_years)
            benchmark_sharpe = compute_sharpe(bench_pnl, self.ann_factor)
            benchmark_max_drawdown = compute_max_drawdown(bench_equity)
            final_excess = total_return - benchmark_total_return
            alpha_excess = final_excess
            annual_alpha_excess = annual_return - benchmark_annual_return
            sharpe_delta = sharpe - benchmark_sharpe
            maxdd_delta = abs(max_dd) - abs(benchmark_max_drawdown)
            win_rate_vs_bh = float(np.mean(pnl > bench_pnl))
            period_bars = max(int(round(self.ann_factor / 12)), 1)
            period_win_rate_vs_bh = compute_period_win_rate(pnl, bench_pnl, period_bars)
            upside_capture = compute_capture_ratio(pnl, bench_pnl, positive_benchmark=True)
            downside_capture = compute_capture_ratio(pnl, bench_pnl, positive_benchmark=False)
            max_underperformance_streak = max_consecutive_underperformance(pnl, bench_pnl)

        return BacktestMetrics(
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            calmar=calmar,
            total_return=total_return,
            annual_return=annual_return,
            n_trades=n_trades,
            avg_holding=avg_holding,
            benchmark_total_return=benchmark_total_return,
            benchmark_annual_return=benchmark_annual_return,
            benchmark_sharpe=benchmark_sharpe,
            benchmark_max_drawdown=benchmark_max_drawdown,
            final_excess=final_excess,
            alpha_excess=alpha_excess,
            annual_alpha_excess=annual_alpha_excess,
            sharpe_delta=sharpe_delta,
            maxdd_delta=maxdd_delta,
            win_rate_vs_bh=win_rate_vs_bh,
            period_win_rate_vs_bh=period_win_rate_vs_bh,
            upside_capture=upside_capture,
            downside_capture=downside_capture,
            max_underperformance_streak=max_underperformance_streak,
            equity_curve=equity,
            pnl_series=pnl,
        )


def pnl_attribution(
    returns: np.ndarray,
    positions: np.ndarray,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
) -> dict:
    """PnL を long / short / コスト 起因に分解する.

    Returns:
        {"long_gross": float, "short_gross": float, "cost_total": float, "net_total": float}
    """
    gross = positions * returns
    costs = compute_costs(positions, spread_bps, fee_rate, slippage_bps)
    long_mask = positions > 0
    short_mask = positions < 0
    return {
        "long_gross":  float(gross[long_mask].sum())  if long_mask.any()  else 0.0,
        "short_gross": float(gross[short_mask].sum()) if short_mask.any() else 0.0,
        "cost_total":  float(costs.sum()),
        "net_total":   float((gross - costs).sum()),
    }
