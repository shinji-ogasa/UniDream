"""HMM レジーム検出モジュール.

hmmlearn の GaussianHMM でレジームを推定し、レジーム別メトリクスを計算する。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

from unidream.eval.backtest import Backtest, BacktestMetrics, ANNUALIZATION


class RegimeDetector:
    """Hidden Markov Model によるレジーム検出.

    Args:
        n_states: HMM の状態数（2 = bull/bear、3 = bull/bear/sideways）
        covariance_type: HMM の共分散タイプ（'full' / 'diag' / 'tied'）
        n_iter: EM アルゴリズムの最大イテレーション数
        random_state: 乱数シード
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        if not HAS_HMMLEARN:
            raise ImportError("hmmlearn が必要です: pip install hmmlearn")
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, returns: np.ndarray | pd.Series) -> "RegimeDetector":
        """リターン系列に HMM をフィットさせる.

        fit 後、状態を平均リターンの昇順にソートする（ラベルスイッチング対策）。
        state 0 = 最低平均リターン（bearish）、state n-1 = 最高（bullish）。
        """
        r = np.asarray(returns).reshape(-1, 1)
        self.model.fit(r)
        self._fitted = True

        # ラベルを平均リターン昇順に固定（fold 間の一貫性確保）
        order = np.argsort(self.model.means_.flatten())
        self._state_perm = order          # new_i → old_i（モデル内部インデックス）
        self._inv_perm = np.argsort(order) # old_i → new_i（外部ラベル）

        return self

    def predict(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """レジームラベルを予測する（平均リターン昇順にソート済み）.

        Returns:
            レジームラベル列 (T,) ∈ {0, 1, ..., n_states-1}
        """
        if not self._fitted:
            raise RuntimeError("fit() を先に呼んでください")
        r = np.asarray(returns).reshape(-1, 1)
        raw = self.model.predict(r)
        return self._inv_perm[raw]

    def predict_proba(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """各レジームの事後確率を返す（平均リターン昇順にソート済み）.

        Returns:
            事後確率行列 (T, n_states)  ─ 列 0 = bearish、列 n-1 = bullish
        """
        if not self._fitted:
            raise RuntimeError("fit() を先に呼んでください")
        r = np.asarray(returns).reshape(-1, 1)
        raw = self.model.predict_proba(r)          # (T, n_states) 元の順
        return raw[:, self._state_perm]             # ソート済み列順に並べ替え

    def fit_predict(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """fit と predict を一度に実行する."""
        return self.fit(returns).predict(returns)

    def expected_returns(self) -> np.ndarray:
        """各ソート済みレジームの平均リターンを返す (n_states,)."""
        if not self._fitted:
            raise RuntimeError("fit() を先に呼んでください")
        return np.array([self.model.means_[old_i][0] for old_i in self._state_perm])

    @property
    def regime_stats(self) -> pd.DataFrame:
        """各レジームの平均リターン・ボラティリティを返す（ソート済み）."""
        rows = []
        for new_i, old_i in enumerate(self._state_perm):
            mean = float(self.model.means_[old_i][0])
            std = float(np.sqrt(self.model.covars_[old_i][0][0]))
            rows.append({"regime": new_i, "mean": mean, "std": std, "sharpe": mean / (std + 1e-8)})
        return pd.DataFrame(rows)


def regime_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
    regimes: np.ndarray,
    n_states: int = 3,
    interval: str = "15m",
    **backtest_kwargs,
) -> dict[int, dict]:
    """レジーム別にバックテストメトリクスを計算する.

    Args:
        returns: 対数リターン列 (T,)
        positions: ポジション比率列 (T,)
        regimes: レジームラベル列 (T,)

    Returns:
        {regime_id: {"metrics": BacktestMetrics, "n_bars": int, "fraction": float}}
    """
    T = len(returns)
    results = {}
    for r in range(n_states):
        mask = regimes == r
        n_bars = int(mask.sum())
        if n_bars < 10:
            continue
        bt = Backtest(
            returns[mask],
            positions[mask],
            interval=interval,
            **backtest_kwargs,
        )
        metrics = bt.run()
        results[r] = {
            "metrics": metrics,
            "n_bars": n_bars,
            "fraction": n_bars / T,
        }
    return results


def print_regime_report(
    regime_results: dict[int, dict],
    regime_detector: Optional[RegimeDetector] = None,
) -> None:
    """レジーム別レポートを表示する."""
    print("=" * 60)
    print("Regime-based Backtest Report")
    print("=" * 60)

    if regime_detector is not None and regime_detector._fitted:
        stats_df = regime_detector.regime_stats
        print("\nRegime Statistics (returns):")
        print(stats_df.to_string(index=False))

    print("\nBacktest Metrics by Regime:")
    for regime_id, r in sorted(regime_results.items()):
        m = r["metrics"]
        print(f"\n  Regime {regime_id} ({r['fraction']:.1%} of bars, {r['n_bars']} bars):")
        print(f"    Sharpe:    {m.sharpe:.3f}")
        print(f"    Sortino:   {m.sortino:.3f}")
        print(f"    MaxDD:     {m.max_drawdown:.3f}")
        print(f"    Calmar:    {m.calmar:.3f}")
        print(f"    TotalRet:  {m.total_return:.4f}")
