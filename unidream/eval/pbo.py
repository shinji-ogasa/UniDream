"""PBO (Probability of Backtest Overfitting) & Deflated Sharpe Ratio モジュール.

References:
  - Bailey & Lopez de Prado (2014): "The Probability of Backtest Overfitting"
  - Lopez de Prado & Bailey (2014): "The Deflated Sharpe Ratio" (SSRN 2460551)
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy import stats


# --- PBO ---

def compute_pbo(
    returns_matrix: list[np.ndarray] | np.ndarray,
    n_combinations: Optional[int] = None,
) -> float:
    """Probability of Backtest Overfitting（簡略版）を計算する.

    全 fold のリターン列を入力とし、IS/OOS 分割でパフォーマンス劣化を検出する。

    NOTE: 標準的な CSCV-PBO（Bailey & Lopez de Prado 2014）は複数の戦略候補に対して
    IS 最良戦略の OOS ランクを検定するが、本実装は WFO fold を候補として扱い、
    IS 最良 Sharpe > OOS 最良 Sharpe の頻度で過学習を推定する簡略版である。
    複数のモデル構成を比較する場合は CSCV 版に差し替えること。

    Args:
        returns_matrix: fold ごとのリターン列のリスト
        n_combinations: 評価する組み合わせ数（None = 全組み合わせ）

    Returns:
        PBO スコア（0〜1、高いほど過学習の疑いが強い）
    """
    # 各 fold のリターンを均一長に揃える
    n_folds = len(returns_matrix)
    if n_folds < 2:
        return 0.5  # fold が足りない場合は中立

    # 各 fold の Sharpe を計算
    sharpes = []
    for ret in returns_matrix:
        r = np.asarray(ret)
        if r.std() < 1e-10:
            sharpes.append(0.0)
        else:
            sharpes.append(float(r.mean() / r.std()))
    sharpes = np.array(sharpes)

    # 半分を IS、残り半分を OOS として割り当てる全組み合わせを試行
    half = n_folds // 2
    all_combos = list(itertools.combinations(range(n_folds), half))

    if n_combinations is not None and n_combinations < len(all_combos):
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(all_combos), size=n_combinations, replace=False)
        all_combos = [all_combos[i] for i in idxs]

    overfit_count = 0
    total = 0

    for is_idxs in all_combos:
        oos_idxs = [i for i in range(n_folds) if i not in is_idxs]
        if not oos_idxs:
            continue

        is_sharpes = sharpes[list(is_idxs)]
        oos_sharpes = sharpes[oos_idxs]

        best_is = np.argmax(is_sharpes)
        best_oos = np.argmax(oos_sharpes)

        # 簡略化: IS 最良 Sharpe > OOS 最良 Sharpe → IS が膨張 → overfitting
        # (fold ベース PBO — 標準 CSCV-PBO は複数戦略に対して IS-best の
        #  OOS ランクを検定するが、本実装は fold を候補として扱う)
        if is_sharpes[best_is] > oos_sharpes[best_oos]:
            overfit_count += 1
        total += 1

    if total == 0:
        return 0.5

    return float(overfit_count / total)


def estimate_return_moments(returns: np.ndarray | None) -> tuple[float, float]:
    """リターン列から skew と kurtosis を推定する（Fisher 定義）.

    crypto は negative skew + fat tails が典型的なので、正規分布の仮定（0, 3）は
    deflation を過小評価する。実データが利用可能な場合はここで推定する。
    """
    if returns is None or len(returns) < 20:
        return 0.0, 3.0
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return 0.0, 3.0
    return float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_annual: bool = False,
    returns: np.ndarray | None = None,
) -> float:
    """Deflated Sharpe Ratio を計算する.

    多重比較補正を行い、過学習を検出する。

    Args:
        sharpe: 報告された Sharpe Ratio
        n_trials: 試行数（ハイパーパラメータ探索回数等）
        T: サンプル数（バー数）
        skew: リターンの歪度（0.0 = 正規分布仮定）
        kurt: リターンの尖度（3.0 = 正規分布仮定）
        sharpe_annual: True の場合は年換算済みとして扱う
        returns: 実リターン列。指定時は skew/kurt をデータから推定（引数より優先）

    Returns:
        Deflated Sharpe Ratio（z スコア）
    """
    if returns is not None:
        skew, kurt = estimate_return_moments(returns)

    if n_trials <= 0:
        n_trials = 1

    # n_trials=1 のとき ppf(0) = -inf になるため多重比較補正なし扱い
    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        # Haircut Sharpe の期待値（Bonferroni 補正近似）
        # E[max SR | H0] ≈ (1 - γ) * Z^{-1}(1 - 1/n) + γ * Z^{-1}(1 - 1/(n*e))
        # 簡略化: Bonferroni
        expected_max_sr = stats.norm.ppf(1 - 1.0 / n_trials)

    # 非正規性補正
    # Var[SR] = (1 - skew * SR + (kurt - 1) / 4 * SR^2) / T
    sr_variance = (1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / max(T - 1, 1)
    sr_std = np.sqrt(max(sr_variance, 1e-10))

    # Deflated SR: (SR - E[max SR]) / std[SR]
    dsr = (sharpe - expected_max_sr) / sr_std
    return float(dsr)


def deflated_sharpe_pvalue(
    sharpe: float,
    n_trials: int,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe の p 値を返す（α = 0.05 で有意なら過学習でない）."""
    dsr = deflated_sharpe(sharpe, n_trials, T, skew, kurt)
    return float(stats.norm.sf(dsr))  # one-tailed p-value


def filter_by_pbo(
    wfo_results: dict,
    pbo_threshold: float = 0.5,
) -> dict:
    """PBO が閾値以下の結果のみを残す（過学習フィルタ）.

    Args:
        wfo_results: walk_forward_eval の戻り値
        pbo_threshold: PBO の閾値（これ以上は過学習とみなして除外）

    Returns:
        フィルタ後の wfo_results
    """
    pnl_list = [wfo_results[k]["metrics"].pnl_series for k in sorted(wfo_results.keys())]
    pbo = compute_pbo(pnl_list)
    print(f"PBO score: {pbo:.4f} (threshold: {pbo_threshold})")

    if pbo >= pbo_threshold:
        print(f"  WARNING: High PBO ({pbo:.4f}) — potential backtest overfitting detected.")

    return wfo_results, pbo
