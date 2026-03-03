"""Hindsight Oracle モジュール.

train 期間のみで後ろ向き DP を使って最適行動列を計算する。
テスト期間の未来情報は一切使わない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 離散行動: ポジション比率
ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
N_ACTIONS = len(ACTIONS)


def compute_step_reward(
    position: float,
    next_return: float,
    prev_position: float,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
) -> float:
    """1 ステップの報酬（コスト控除後のリターン）を計算する.

    Args:
        position: 現在のポジション比率
        next_return: 次の対数リターン
        prev_position: 前のポジション比率（コスト計算に使用）
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ (basis points per unit position change)

    Returns:
        コスト控除後のリターン
    """
    pnl = position * next_return
    delta_pos = abs(position - prev_position)
    spread_cost = (spread_bps / 10000) / 2 * delta_pos
    fee_cost = fee_rate * delta_pos
    slippage_cost = (slippage_bps / 10000) * delta_pos
    total_cost = spread_cost + fee_cost + slippage_cost
    return pnl - total_cost


def hindsight_oracle_dp(
    returns: np.ndarray | pd.Series,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
    discount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """後ろ向き DP で最適行動列を計算する（Hindsight Oracle）.

    train 期間のリターン系列を受け取り、取引コストを考慮した上で
    最大累積リターンを達成する離散行動列を返す。

    Args:
        returns: 対数リターン系列（shift(1) 適用済み）
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ (basis points)
        discount: DP の割引率

    Returns:
        actions: 最適行動インデックス列 (T,)
        values: 各時点の最適期待価値 (T,)
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    T = len(returns)
    # V[t, a] = time t に action a を取ったときの最適累積リターン
    V = np.full((T + 1, N_ACTIONS), 0.0)

    # バックトラッキング用
    policy = np.zeros((T, N_ACTIONS), dtype=int)

    # 後ろ向き DP
    for t in range(T - 1, -1, -1):
        r_next = returns[t] if t < T else 0.0
        for a_idx, pos in enumerate(ACTIONS):
            best_val = -np.inf
            best_next_a = 0
            for next_a_idx, next_pos in enumerate(ACTIONS):
                step_r = compute_step_reward(
                    position=pos,
                    next_return=r_next,
                    prev_position=pos,  # 同一 t 内では前ポジ = 現ポジ（初期化）
                    spread_bps=spread_bps,
                    fee_rate=fee_rate,
                    slippage_bps=slippage_bps,
                )
                # コスト: a → next_a のポジション変化
                transition_cost = (
                    (spread_bps / 10000) / 2 +
                    fee_rate +
                    (slippage_bps / 10000)
                ) * abs(next_pos - pos)
                val = step_r - transition_cost + discount * V[t + 1, next_a_idx]
                if val > best_val:
                    best_val = val
                    best_next_a = next_a_idx
            V[t, a_idx] = best_val
            policy[t, a_idx] = best_next_a

    # 最初の行動を greedily 選択
    actions = np.zeros(T, dtype=int)
    actions[0] = np.argmax(V[0])
    for t in range(T - 1):
        actions[t + 1] = policy[t, actions[t]]

    values = np.array([V[t, actions[t]] for t in range(T)])
    return actions, values


def oracle_positions(
    returns: np.ndarray | pd.Series,
    **kwargs,
) -> np.ndarray:
    """最適ポジション比率の系列を返す（oracle の便利 wrapper）."""
    action_indices, _ = hindsight_oracle_dp(returns, **kwargs)
    return ACTIONS[action_indices]


def oracle_to_dataset(
    returns: np.ndarray | pd.Series,
    features: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """BC 学習用データセットを作成する.

    Args:
        returns: 対数リターン系列
        features: 特徴量行列 (T, feat_dim)

    Returns:
        {states, actions, values}
    """
    action_indices, values = hindsight_oracle_dp(returns, **kwargs)
    T_min = min(len(action_indices), len(features))
    return {
        "states": features[:T_min],
        "actions": action_indices[:T_min],
        "values": values[:T_min],
    }
