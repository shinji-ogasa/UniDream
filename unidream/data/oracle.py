"""Hindsight Oracle モジュール.

train 期間のみで後ろ向き DP を使って最適行動列を計算する。
テスト期間の未来情報は一切使わない。

DP の状態設計:
  V[t, prev_a] = step t 以降の最大期待累積リターン
                 ただし step t-1 に ACTIONS[prev_a] を保有していた場合
  行動: step t でどのポジション a を保有するか
  コスト: prev_pos → pos の遷移コスト（step t の開始時点で支払い）
  収益: ACTIONS[a] * returns[t]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 離散行動: ポジション比率
ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
N_ACTIONS = len(ACTIONS)

# 初期ポジション = フラット (0.0)
_FLAT_IDX = int(np.where(ACTIONS == 0.0)[0][0])  # = 2


def _transition_cost(
    prev_pos: float,
    pos: float,
    spread_bps: float,
    fee_rate: float,
    slippage_bps: float,
) -> float:
    """遷移コスト: prev_pos → pos のポジション変化に伴う取引コスト."""
    delta = abs(pos - prev_pos)
    return ((spread_bps / 10000) / 2 + fee_rate + (slippage_bps / 10000)) * delta


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

    DP 状態: (t, prev_a_idx)
      - t: 時刻
      - prev_a_idx: step t-1 で保有していたポジションのインデックス
                    （初期: FLAT_IDX = 2 ← ACTIONS[2] == 0.0）

    Args:
        returns: 対数リターン系列（shift(1) 適用済み、長さ T）
        spread_bps: スプレッド (basis points)
        fee_rate: 手数料率
        slippage_bps: スリッページ (basis points per unit |Δpos|)
        discount: DP の割引率

    Returns:
        actions: 最適行動インデックス列 (T,)
        values:  V[t, actions[t-1]] の列 (T,) — 各 step の最適期待価値
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    T = len(returns)

    # V[t, prev_a] = step t 以降の最大期待累積リターン（prev_a = 直前のポジション）
    V = np.zeros((T + 1, N_ACTIONS))  # 終端 V[T, :] = 0
    policy = np.zeros((T, N_ACTIONS), dtype=int)  # policy[t, prev_a] = 最適 a

    # --- 後ろ向き DP ---
    for t in range(T - 1, -1, -1):
        r_t = returns[t]
        for prev_a in range(N_ACTIONS):
            prev_pos = ACTIONS[prev_a]
            best_val = -np.inf
            best_a = 0
            for a in range(N_ACTIONS):
                pos = ACTIONS[a]
                cost = _transition_cost(prev_pos, pos, spread_bps, fee_rate, slippage_bps)
                net_pnl = pos * r_t - cost
                val = net_pnl + discount * V[t + 1, a]
                if val > best_val:
                    best_val = val
                    best_a = a
            V[t, prev_a] = best_val
            policy[t, prev_a] = best_a

    # --- 前向きパス: 初期ポジション = フラット ---
    actions = np.zeros(T, dtype=int)
    prev_a = _FLAT_IDX
    for t in range(T):
        actions[t] = policy[t, prev_a]
        prev_a = actions[t]

    # values[t] = V[t, 直前のポジション] — step t 時点の最適期待累積リターン
    values = np.zeros(T)
    prev_a = _FLAT_IDX
    for t in range(T):
        values[t] = V[t, prev_a]
        prev_a = actions[t]

    return actions, values


def compute_net_returns(
    returns: np.ndarray | pd.Series,
    action_indices: np.ndarray,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
) -> np.ndarray:
    """行動列と生リターン列からネットリターン（コスト控除後）を計算する.

    net_return[t] = ACTIONS[action_indices[t]] * returns[t]
                    - cost(prev_pos → pos)

    Args:
        returns: 対数リターン系列 (T,)
        action_indices: 行動インデックス列 (T,)

    Returns:
        net_returns: コスト控除後のリターン列 (T,)
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    T = len(returns)
    net_returns = np.zeros(T)
    prev_pos = ACTIONS[_FLAT_IDX]  # 初期ポジション = 0.0

    for t in range(T):
        pos = ACTIONS[action_indices[t]]
        cost = _transition_cost(prev_pos, pos, spread_bps, fee_rate, slippage_bps)
        net_returns[t] = pos * returns[t] - cost
        prev_pos = pos

    return net_returns


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
        {states, actions, values, net_returns}
    """
    action_indices, values = hindsight_oracle_dp(returns, **kwargs)
    net_rets = compute_net_returns(
        returns,
        action_indices,
        spread_bps=kwargs.get("spread_bps", 5.0),
        fee_rate=kwargs.get("fee_rate", 0.0004),
        slippage_bps=kwargs.get("slippage_bps", 2.0),
    )
    T_min = min(len(action_indices), len(features))
    return {
        "states": features[:T_min],
        "actions": action_indices[:T_min],
        "values": values[:T_min],
        "net_returns": net_rets[:T_min],
    }
