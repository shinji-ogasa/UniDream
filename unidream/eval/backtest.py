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
    alpha_excess: float | None = None
    sharpe_delta: float | None = None
    maxdd_delta: float | None = None
    win_rate_vs_bh: float | None = None

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
            "alpha_excess": self.alpha_excess,
            "sharpe_delta": self.sharpe_delta,
            "maxdd_delta": self.maxdd_delta,
            "win_rate_vs_bh": self.win_rate_vs_bh,
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
    """年換算 Sortino Ratio を計算する."""
    downside = pnl[pnl < 0]
    if len(downside) == 0 or downside.std() < 1e-10:
        return 99.0 if pnl.mean() > 0 else 0.0
    return float(pnl.mean() / downside.std() * np.sqrt(ann_factor))


def compute_max_drawdown(equity: np.ndarray) -> float:
    """最大ドローダウンを計算する（0〜1 の比率）."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-8)
    return float(drawdown.min())


def compute_calmar(total_return: float, max_dd: float, period_years: float = 1.0) -> float:
    """Calmar Ratio を計算する."""
    if abs(max_dd) < 1e-10:
        return 99.0 if total_return > 0 else 0.0
    return float((total_return / period_years) / abs(max_dd))


def compute_annual_return(total_return: float, period_years: float) -> float:
    """累積リターンから年率リターンを計算する."""
    if period_years <= 0:
        return 0.0
    equity_end = max(1e-12, 1.0 + total_return)
    return float(equity_end ** (1.0 / period_years) - 1.0)


class Backtest:
    """バックテスト実行クラス.

    Args:
        returns: 対数リターン列 (T,)
        positions: ポジション比率列 (T,) ∈ {-1, -0.5, 0, 0.5, 1}
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ (basis points)
        interval: 足種（年換算係数計算に使用）
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
    ):
        assert len(returns) == len(positions), "returns と positions の長さが一致しない"
        self.returns = np.asarray(returns, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.spread_bps = spread_bps
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.ann_factor = ANNUALIZATION.get(interval, 252 * 96)
        self.benchmark_positions = (
            np.asarray(benchmark_positions, dtype=np.float64)
            if benchmark_positions is not None else None
        )
        if self.benchmark_positions is not None:
            assert len(self.benchmark_positions) == len(self.positions), (
                "benchmark_positions と positions の長さが一致しない"
            )

    def run(self) -> BacktestMetrics:
        """バックテストを実行してメトリクスを返す."""
        pnl = compute_pnl(self.returns, self.positions, self.spread_bps, self.fee_rate, self.slippage_bps)
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
        pos_changes = np.diff(self.positions, prepend=0.0) != 0
        n_trades = int(pos_changes.sum())

        # 連続して同じポジションを保持した期間の平均
        holding_lengths = []
        current_len = 1
        for i in range(1, len(self.positions)):
            if self.positions[i] == self.positions[i - 1]:
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
        alpha_excess = None
        sharpe_delta = None
        maxdd_delta = None
        win_rate_vs_bh = None
        if self.benchmark_positions is not None:
            bench_pnl = compute_pnl(
                self.returns,
                self.benchmark_positions,
                self.spread_bps,
                self.fee_rate,
                self.slippage_bps,
            )
            bench_equity = np.exp(np.cumsum(bench_pnl))
            benchmark_total_return = float(bench_equity[-1] - 1.0)
            benchmark_annual_return = compute_annual_return(benchmark_total_return, period_years)
            benchmark_sharpe = compute_sharpe(bench_pnl, self.ann_factor)
            benchmark_max_drawdown = compute_max_drawdown(bench_equity)
            alpha_excess = annual_return - benchmark_annual_return
            sharpe_delta = sharpe - benchmark_sharpe
            maxdd_delta = abs(max_dd) - abs(benchmark_max_drawdown)
            win_rate_vs_bh = float(np.mean(pnl > bench_pnl))

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
            alpha_excess=alpha_excess,
            sharpe_delta=sharpe_delta,
            maxdd_delta=maxdd_delta,
            win_rate_vs_bh=win_rate_vs_bh,
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


def regime_backtest(
    returns: np.ndarray,
    positions: np.ndarray,
    regimes: np.ndarray,
    n_regimes: int = 3,
    **backtest_kwargs,
) -> dict[int, BacktestMetrics]:
    """レジーム別にバックテストを実行する.

    Args:
        regimes: レジームラベル列 (T,) ∈ {0, 1, ..., n_regimes-1}

    Returns:
        {regime_id: BacktestMetrics}
    """
    results = {}
    for r in range(n_regimes):
        mask = regimes == r
        if mask.sum() < 10:
            continue
        bt = Backtest(returns[mask], positions[mask], **backtest_kwargs)
        results[r] = bt.run()
    return results
