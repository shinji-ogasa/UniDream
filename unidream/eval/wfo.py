"""Walk-Forward Optimization 評価モジュール.

各 WFO fold に対してモデルの train/val/test サイクルを実行し、
test fold のバックテスト結果を集計する。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from unidream.data.dataset import WFOSplit
from unidream.eval.backtest import Backtest, BacktestMetrics


def walk_forward_eval(
    splits: list[WFOSplit],
    train_fn: Callable[[WFOSplit], None],
    predict_fn: Callable[[WFOSplit, np.ndarray], np.ndarray],
    returns_getter: Callable[[WFOSplit], np.ndarray],
    backtest_kwargs: Optional[dict] = None,
) -> dict:
    """Walk-Forward Optimization を実行する.

    各 fold に対して:
    1. train_fn(split) でモデルを学習
    2. predict_fn(split, returns) でテスト期間の position を取得
    3. Backtest でメトリクスを計算

    Args:
        splits: WFOSplit のリスト
        train_fn: fold を受け取ってモデルを学習する関数
        predict_fn: fold と観測を受け取って positions を返す関数
        returns_getter: fold を受け取って test リターン列を返す関数
        backtest_kwargs: Backtest に渡す追加引数

    Returns:
        {fold_idx: {"split": WFOSplit, "metrics": BacktestMetrics, "positions": np.ndarray}}
    """
    if backtest_kwargs is None:
        backtest_kwargs = {}

    results = {}
    for split in splits:
        print(f"[WFO] Fold {split.fold_idx}: train {split.train_start.date()}→{split.train_end.date()}, "
              f"test {split.test_start.date()}→{split.test_end.date()}")

        # 学習
        train_fn(split)

        # テスト期間の推論
        test_returns = returns_getter(split)
        positions = predict_fn(split, test_returns)

        # 長さ調整
        min_len = min(len(test_returns), len(positions))
        test_returns = test_returns[:min_len]
        positions = positions[:min_len]

        # バックテスト
        bt = Backtest(test_returns, positions, **backtest_kwargs)
        metrics = bt.run()

        results[split.fold_idx] = {
            "split": split,
            "metrics": metrics,
            "positions": positions,
        }

        print(f"  → Sharpe={metrics.sharpe:.3f}, MaxDD={metrics.max_drawdown:.3f}, "
              f"Calmar={metrics.calmar:.3f}")

    return results


def aggregate_wfo_results(results: dict) -> pd.DataFrame:
    """WFO 結果を DataFrame に集計する."""
    rows = []
    for fold_idx, r in results.items():
        m = r["metrics"]
        s = r["split"]
        rows.append({
            "fold": fold_idx,
            "test_start": s.test_start,
            "test_end": s.test_end,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "calmar": m.calmar,
            "total_return": m.total_return,
            "n_trades": m.n_trades,
        })
    return pd.DataFrame(rows)


def collect_all_test_returns(results: dict) -> tuple[np.ndarray, np.ndarray]:
    """全 fold のテスト期間 PnL を結合する.

    PBO 計算に使用する。

    Returns:
        (returns_matrix, positions_matrix): 各 fold のリターン/ポジション
    """
    all_pnl = []
    for fold_idx in sorted(results.keys()):
        pnl = results[fold_idx]["metrics"].pnl_series
        all_pnl.append(pnl)
    return all_pnl
