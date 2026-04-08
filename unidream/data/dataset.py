"""データローダーモジュール.

WFO (Walk-Forward Optimization) 分割対応の PyTorch Dataset。
スライディングウィンドウでシーケンスを生成する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class WFOSplit:
    """WFO の 1 fold を表すデータクラス."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def get_wfo_splits(
    df: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    test_months: int = 3,
    min_folds: int = 3,
) -> list[WFOSplit]:
    """Walk-Forward Optimization の分割リストを生成する.

    四半期ロール（test_months ずつずらしていく）。

    Args:
        df: インデックスが datetime の DataFrame
        train_years: train 期間（年）
        val_months: val 期間（月）
        test_months: test 期間（月）（= ロール幅）
        min_folds: 最低 fold 数

    Returns:
        WFOSplit のリスト
    """
    idx = df.index
    start = idx[0]
    end = idx[-1]

    train_delta = pd.DateOffset(years=train_years)
    val_delta = pd.DateOffset(months=val_months)
    test_delta = pd.DateOffset(months=test_months)

    splits = []
    fold_idx = 0
    test_start = start + train_delta + val_delta

    while test_start + test_delta <= end:
        train_start = test_start - train_delta - val_delta
        train_end = test_start - val_delta
        val_start = train_end
        val_end = test_start
        test_end = test_start + test_delta

        # データが実際に存在するか確認
        if (
            len(df.loc[train_start:train_end]) > 0
            and len(df.loc[val_start:val_end]) > 0
            and len(df.loc[test_start:test_end]) > 0
        ):
            splits.append(WFOSplit(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            ))
            fold_idx += 1

        test_start += test_delta

    return splits


class SequenceDataset(Dataset):
    """スライディングウィンドウで時系列シーケンスを生成する Dataset.

    Args:
        features: 特徴量行列 (T, feat_dim)
        seq_len: シーケンス長（コンテキスト窓）
        actions: 行動列 (T,) 省略可（BC 学習時は必要）
        returns: リターン列 (T,) 省略可（報酬計算用）
    """

    def __init__(
        self,
        features: np.ndarray,
        seq_len: int = 64,
        actions: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        regime_probs: Optional[np.ndarray] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.seq_len = seq_len
        if actions is not None:
            action_arr = np.asarray(actions)
            if np.issubdtype(action_arr.dtype, np.integer):
                self.actions = torch.tensor(action_arr, dtype=torch.long)
            else:
                self.actions = torch.tensor(action_arr, dtype=torch.float32)
        else:
            self.actions = None
        self.returns = torch.tensor(returns, dtype=torch.float32) if returns is not None else None
        self.regime_probs = (
            torch.tensor(regime_probs, dtype=torch.float32) if regime_probs is not None else None
        )
        self.T = len(features)

    def __len__(self) -> int:
        return max(0, self.T - self.seq_len + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = idx
        e = idx + self.seq_len
        item = {"obs": self.features[s:e]}  # (seq_len, feat_dim)
        if self.actions is not None:
            item["actions"] = self.actions[s:e]  # (seq_len,)
        if self.returns is not None:
            item["returns"] = self.returns[s:e]  # (seq_len,)
        if self.regime_probs is not None:
            item["regime"] = self.regime_probs[s:e]  # (seq_len, regime_dim)
        return item


class WFODataset:
    """Walk-Forward 分割に対応したデータセット管理クラス.

    一つの WFOSplit に対して train/val/test の SequenceDataset を生成する。
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        returns: pd.Series,
        split: WFOSplit,
        seq_len: int = 64,
        oracle_actions: Optional[np.ndarray] = None,
    ):
        """
        Args:
            features_df: 特徴量 DataFrame（rolling z-score 正規化済み）
            returns: 生リターン Series（Oracle/バックテスト用）
            split: WFOSplit（train/val/test の時刻境界）
            seq_len: シーケンス長
            oracle_actions: Oracle から得た行動列（BC 学習時）
        """
        self.split = split
        self.seq_len = seq_len
        self._feature_columns = list(features_df.columns)

        # 各 split のデータを切り出す
        # train/val: 右端 exclusive（次の期間の開始バーと重複を防ぐ）
        # test: 右端 inclusive（後続期間がないため末尾バーを含める）
        def _slice(start, end, right_inclusive=False):
            if right_inclusive:
                mask = (features_df.index >= start) & (features_df.index <= end)
            else:
                mask = (features_df.index >= start) & (features_df.index < end)
            feat = features_df[mask].to_numpy()
            ret = returns[mask].to_numpy()
            return feat, ret

        self._train_feat, self._train_ret = _slice(split.train_start, split.train_end)
        self._val_feat, self._val_ret = _slice(split.val_start, split.val_end)
        self._test_feat, self._test_ret = _slice(split.test_start, split.test_end, right_inclusive=True)

        # Oracle 行動列（train 期間のみ）
        if oracle_actions is not None:
            train_len = len(self._train_feat)
            self._train_actions = oracle_actions[:train_len]
        else:
            self._train_actions = None

    def train_dataset(self) -> SequenceDataset:
        return SequenceDataset(
            self._train_feat,
            self.seq_len,
            actions=self._train_actions,
            returns=self._train_ret,
        )

    def val_dataset(self) -> SequenceDataset:
        return SequenceDataset(self._val_feat, self.seq_len, returns=self._val_ret)

    def test_dataset(self) -> SequenceDataset:
        return SequenceDataset(self._test_feat, self.seq_len, returns=self._test_ret)

    @property
    def train_returns(self) -> np.ndarray:
        return self._train_ret

    @property
    def val_returns(self) -> np.ndarray:
        return self._val_ret

    @property
    def test_returns(self) -> np.ndarray:
        return self._test_ret

    @property
    def train_features(self) -> np.ndarray:
        return self._train_feat

    @property
    def val_features(self) -> np.ndarray:
        return self._val_feat

    @property
    def test_features(self) -> np.ndarray:
        return self._test_feat

    @property
    def obs_dim(self) -> int:
        return self._train_feat.shape[1]

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns
