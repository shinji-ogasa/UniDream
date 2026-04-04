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


def _enforce_min_hold(actions: np.ndarray, min_hold: int) -> np.ndarray:
    """最短保有期間を強制する後処理（greedy 前向きパス）.

    min_hold バー未満で切り替わるポジションは前のポジションへ戻す。
    oracle が avg_hold=3.3b の高頻度ラベルを生成するのを抑制し、
    BC が模倣できる安定したラベル列を生成するために使用する。
    """
    if min_hold <= 1:
        return actions
    result = actions.copy()
    prev_a = result[0]
    hold = 1
    for t in range(1, len(result)):
        cur = result[t]
        if cur == prev_a:
            hold += 1
        else:
            if hold >= min_hold:
                prev_a = cur
                hold = 1
            else:
                result[t] = prev_a
                hold += 1
    return result


def hindsight_oracle_dp(
    returns: np.ndarray | pd.Series,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
    discount: float = 1.0,
    min_hold: int = 0,
    soft_label_temp: float = 0.0,
    reward_mode: str = "absolute",
    benchmark_position: float = 1.0,
    action_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, "np.ndarray | None"]:
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
        min_hold: 最短保有バー数（後処理で強制。0 で無効）
        soft_label_temp: Boltzmann 温度（> 0 で Q 値から soft label を生成）
        reward_mode: "absolute" または "excess_bh"
        benchmark_position: excess_bh 時の benchmark inventory

    Returns:
        actions: 最適行動インデックス列 (T,)
        values:  V[t, actions[t-1]] の列 (T,) — 各 step の最適期待価値
        soft_labels: (T, N_ACTIONS) Boltzmann soft label（soft_label_temp=0 なら None）
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    T = len(returns)
    action_values = np.asarray(action_values if action_values is not None else ACTIONS, dtype=np.float64)
    n_actions = len(action_values)
    flat_idx = int(np.argmin(np.abs(action_values - 0.0)))

    # V[t, prev_a] = step t 以降の最大期待累積リターン（prev_a = 直前のポジション）
    V = np.zeros((T + 1, n_actions))  # 終端 V[T, :] = 0
    policy = np.zeros((T, n_actions), dtype=int)  # policy[t, prev_a] = 最適 a

    # --- 後ろ向き DP ---
    for t in range(T - 1, -1, -1):
        r_t = returns[t]
        for prev_a in range(n_actions):
            prev_pos = action_values[prev_a]
            best_val = -np.inf
            best_a = 0
            for a in range(n_actions):
                pos = action_values[a]
                cost = _transition_cost(prev_pos, pos, spread_bps, fee_rate, slippage_bps)
                net_pnl = pos * r_t - cost
                if reward_mode == "excess_bh":
                    net_pnl = net_pnl - benchmark_position * r_t
                val = net_pnl + discount * V[t + 1, a]
                if val > best_val:
                    best_val = val
                    best_a = a
            V[t, prev_a] = best_val
            policy[t, prev_a] = best_a

    # --- 前向きパス: 初期ポジション = フラット ---
    actions = np.zeros(T, dtype=int)
    prev_a = flat_idx
    for t in range(T):
        actions[t] = policy[t, prev_a]
        prev_a = actions[t]

    # values[t] = V[t, 直前のポジション] — step t 時点の最適期待累積リターン
    values = np.zeros(T)
    prev_a = flat_idx
    for t in range(T):
        values[t] = V[t, prev_a]
        prev_a = actions[t]

    if min_hold > 1:
        actions = _enforce_min_hold(actions, min_hold)

    # --- Value-based Soft Labels（Boltzmann）---
    soft_labels: "np.ndarray | None" = None
    if soft_label_temp > 0.0:
        # Q(t, a | prev_a_actual) = ACTIONS[a]*r_t - cost(prev→a) + discount*V[t+1, a]
        # prev_a_actual は forward pass の実際の行動列から構築
        prev_actions = np.empty(T, dtype=int)
        prev_actions[0] = flat_idx
        prev_actions[1:] = actions[:-1]
        prev_pos_arr = action_values[prev_actions]  # (T,)

        # 各 action について Q 値を vectorized 計算
        q = np.zeros((T, n_actions), dtype=np.float64)
        for a in range(n_actions):
            pos = action_values[a]
            delta = np.abs(pos - prev_pos_arr)
            cost_arr = ((spread_bps / 10000) / 2 + fee_rate + (slippage_bps / 10000)) * delta
            q[:, a] = pos * returns - cost_arr + discount * V[1:, a]
            if reward_mode == "excess_bh":
                q[:, a] = q[:, a] - benchmark_position * returns

        # Advantage: A(t,a) = Q(t,a) - max_a Q(t,a)  → max = 0、他は ≤ 0
        adv = q - q.max(axis=1, keepdims=True)

        # Per-timestep scale normalize: A_norm = A / (std(A) + ε)
        # これにより τ が Q の絶対スケールに依存しなくなる
        adv_std = adv.std(axis=1, keepdims=True)
        adv_norm = adv / (adv_std + 1e-8)

        # Boltzmann softmax with normalized advantage
        soft_labels = np.exp(adv_norm / soft_label_temp)
        soft_labels /= soft_labels.sum(axis=1, keepdims=True)

    return actions, values, soft_labels


def compute_net_returns(
    returns: np.ndarray | pd.Series,
    action_indices: np.ndarray,
    spread_bps: float = 5.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 2.0,
    reward_mode: str = "absolute",
    benchmark_position: float = 1.0,
    action_values: np.ndarray | None = None,
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
    action_values = np.asarray(action_values if action_values is not None else ACTIONS, dtype=np.float64)
    flat_idx = int(np.argmin(np.abs(action_values - 0.0)))
    net_returns = np.zeros(T)
    prev_pos = action_values[flat_idx]  # 初期ポジション = 0.0

    for t in range(T):
        pos = action_values[action_indices[t]]
        cost = _transition_cost(prev_pos, pos, spread_bps, fee_rate, slippage_bps)
        net_returns[t] = pos * returns[t] - cost
        if reward_mode == "excess_bh":
            net_returns[t] = net_returns[t] - benchmark_position * returns[t]
        prev_pos = pos

    return net_returns


def oracle_positions(
    returns: np.ndarray | pd.Series,
    **kwargs,
) -> np.ndarray:
    """最適ポジション比率の系列を返す（oracle の便利 wrapper）."""
    action_values = np.asarray(kwargs.get("action_values", ACTIONS), dtype=np.float64)
    action_indices, _, _ = hindsight_oracle_dp(returns, **kwargs)
    return action_values[action_indices]


def smooth_aim_positions(
    target_positions: np.ndarray | pd.Series,
    max_step: float = 0.25,
    band: float = 0.0,
    initial_position: float = 0.0,
    min_position: float = -1.0,
    max_position: float = 1.0,
    benchmark_position: float = 0.0,
    underweight_confirm_bars: int = 0,
    underweight_min_scale: float = 0.0,
) -> np.ndarray:
    """離散 oracle path を滑らかな aim-portfolio path へ変換する.

    各バーで一気に target へ飛ばず、最大 `max_step` だけ部分調整する。
    `band` 以内のズレは無視して no-trade region を作る。
    """
    target_positions = np.asarray(target_positions, dtype=np.float32)
    if len(target_positions) == 0:
        return target_positions.copy()

    max_step = float(max(max_step, 1e-6))
    band = float(max(band, 0.0))
    aim_positions = np.empty_like(target_positions)
    current = float(np.clip(initial_position, min_position, max_position))
    benchmark_position = float(benchmark_position)
    underweight_confirm_bars = int(max(underweight_confirm_bars, 0))
    underweight_min_scale = float(np.clip(underweight_min_scale, 0.0, 1.0))
    underweight_streak = 0

    for t, target in enumerate(target_positions):
        target = float(np.clip(target, min_position, max_position))
        if underweight_confirm_bars > 0 and benchmark_position != 0.0 and target < benchmark_position:
            underweight_streak += 1
            progress = min(1.0, underweight_streak / float(underweight_confirm_bars))
            progress = max(progress, underweight_min_scale)
            target = benchmark_position + (target - benchmark_position) * progress
        else:
            underweight_streak = 0
        gap = target - current
        if abs(gap) <= band:
            next_pos = current
        else:
            next_pos = current + float(np.clip(gap, -max_step, max_step))
        current = float(np.clip(next_pos, min_position, max_position))
        aim_positions[t] = current

    return aim_positions


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
    action_values = np.asarray(kwargs.get("action_values", ACTIONS), dtype=np.float64)
    action_indices, values, _ = hindsight_oracle_dp(returns, **kwargs)
    net_rets = compute_net_returns(
        returns,
        action_indices,
        spread_bps=kwargs.get("spread_bps", 5.0),
        fee_rate=kwargs.get("fee_rate", 0.0004),
        slippage_bps=kwargs.get("slippage_bps", 2.0),
        reward_mode=kwargs.get("reward_mode", "absolute"),
        benchmark_position=kwargs.get("benchmark_position", 1.0),
        action_values=action_values,
    )
    T_min = min(len(action_indices), len(features))
    return {
        "states": features[:T_min],
        "actions": action_indices[:T_min],
        "positions": action_values[action_indices[:T_min]],
        "values": values[:T_min],
        "net_returns": net_rets[:T_min],
    }
