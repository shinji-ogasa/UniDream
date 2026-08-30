"""データローダーモジュール.

WFO (Walk-Forward Optimization) 分割対応の PyTorch Dataset。
スライディングウィンドウでシーケンスを生成する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .availability_contract import (
    AvailabilityContractError,
    validate_availability,
)
from .window_quality import valid_sequence_starts


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


class FeatureArray(np.ndarray):
    """Numpy feature view carrying v4 metadata without adding model columns.

    The training pipeline historically passes numpy arrays between stages.  A
    small ndarray subclass lets those existing call sites carry the timestamp
    and sidecar metadata into ``SequenceDataset`` while preserving normal
    numpy/scikit-learn behaviour and the canonical observation dimension.
    """

    def __new__(
        cls,
        values: Any,
        *,
        index: pd.DatetimeIndex,
        availability: pd.DataFrame | Mapping[str, np.ndarray],
        interval: str,
        include_funding: bool,
        include_mark: bool,
        row_eligible: np.ndarray,
    ):
        obj = np.asarray(values).view(cls)
        obj._unidream_index = index
        obj._unidream_availability = availability
        obj._unidream_interval = interval
        obj._unidream_include_funding = bool(include_funding)
        obj._unidream_include_mark = bool(include_mark)
        obj._unidream_row_eligible = np.asarray(row_eligible, dtype=bool)
        return obj

    def __array_finalize__(self, source: Any) -> None:
        if source is None:
            return
        for name in (
            "_unidream_index",
            "_unidream_availability",
            "_unidream_interval",
            "_unidream_include_funding",
            "_unidream_include_mark",
            "_unidream_row_eligible",
        ):
            setattr(self, name, getattr(source, name, None))


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
        *,
        timestamps: pd.DatetimeIndex | None = None,
        availability: pd.DataFrame | Mapping[str, Any] | None = None,
        interval: str | None = None,
        include_funding: bool | None = None,
        include_mark: bool | None = None,
    ):
        # Inspect metadata before converting to a tensor.  FeatureArray is
        # used by WFODataset to preserve this information through the legacy
        # numpy-valued stage interfaces; a DataFrame is accepted for direct
        # callers as well.
        source_attrs = getattr(features, "attrs", {})
        if timestamps is None:
            timestamps = getattr(features, "_unidream_index", None)
        if timestamps is None and isinstance(features, pd.DataFrame):
            timestamps = features.index
        if availability is None:
            availability = getattr(features, "_unidream_availability", None)
        if availability is None and isinstance(source_attrs, Mapping):
            availability = source_attrs.get("availability")
        if interval is None:
            interval = getattr(features, "_unidream_interval", None)
        if interval is None and isinstance(source_attrs, Mapping):
            interval = source_attrs.get("availability_interval")
        interval = str(interval or "15m")
        if include_funding is None:
            include_funding = getattr(features, "_unidream_include_funding", None)
        if include_funding is None and isinstance(source_attrs, Mapping):
            include_funding = source_attrs.get("availability_include_funding")
        if include_mark is None:
            include_mark = getattr(features, "_unidream_include_mark", None)
        if include_mark is None and isinstance(source_attrs, Mapping):
            include_mark = source_attrs.get("availability_include_mark")
        if include_funding is None:
            include_funding = True
        if include_mark is None:
            include_mark = True

        feature_values = np.asarray(features)
        if feature_values.ndim != 2:
            raise ValueError(
                f"features must be a two-dimensional array, got shape {feature_values.shape}"
            )
        self.features = torch.tensor(feature_values, dtype=torch.float32)
        self.seq_len = seq_len
        self.interval = interval
        self.include_funding = bool(include_funding)
        self.include_mark = bool(include_mark)
        self.timestamps = timestamps
        self.availability = availability
        self._row_eligible = np.ones(len(feature_values), dtype=bool)
        if timestamps is not None:
            if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) != len(feature_values):
                raise ValueError("timestamps must be a DatetimeIndex aligned to features")
        if availability is not None:
            if timestamps is None:
                raise ValueError(
                    "timestamps are required when availability sidecar is supplied"
                )
            try:
                selected = validate_availability(
                    availability,
                    timestamps,
                    include_funding=self.include_funding,
                    include_mark=self.include_mark,
                )
            except AvailabilityContractError as exc:
                raise ValueError(str(exc)) from exc
            self.availability = selected.sidecar
            self._row_eligible = selected.row_eligible
        if availability is not None:
            self._valid_starts = valid_sequence_starts(
                timestamps,
                seq_len,
                interval=self.interval,
                availability=availability,
                include_funding=self.include_funding,
                include_mark=self.include_mark,
            )
        elif timestamps is not None:
            self._valid_starts = valid_sequence_starts(
                timestamps,
                seq_len,
                interval=self.interval,
            )
        else:
            if not isinstance(seq_len, (int, np.integer)) or isinstance(seq_len, bool) or seq_len <= 0:
                raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")
            self._valid_starts = np.arange(
                max(0, len(feature_values) - int(seq_len) + 1),
                dtype=np.int64,
            )
        if actions is not None:
            action_arr = np.asarray(actions)
            if action_arr.ndim == 0 or len(action_arr) != len(feature_values):
                raise ValueError("actions must be aligned one-for-one with features")
            if np.issubdtype(action_arr.dtype, np.integer):
                self.actions = torch.tensor(action_arr, dtype=torch.long)
            else:
                self.actions = torch.tensor(action_arr, dtype=torch.float32)
        else:
            self.actions = None
        if returns is not None and len(returns) != len(feature_values):
            raise ValueError("returns must be aligned one-for-one with features")
        self.returns = torch.tensor(returns, dtype=torch.float32) if returns is not None else None
        if regime_probs is not None and len(regime_probs) != len(feature_values):
            raise ValueError("regime_probs must be aligned one-for-one with features")
        self.regime_probs = (
            torch.tensor(regime_probs, dtype=torch.float32) if regime_probs is not None else None
        )
        self.T = len(feature_values)

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool):
            raise TypeError(f"dataset index must be an integer, got {idx!r}")
        s = int(self._valid_starts[idx])
        e = s + self.seq_len
        item = {"obs": self.features[s:e]}  # (seq_len, feat_dim)
        if self.actions is not None:
            item["actions"] = self.actions[s:e]  # (seq_len,)
        if self.returns is not None:
            item["returns"] = self.returns[s:e]  # (seq_len,)
        if self.regime_probs is not None:
            item["regime"] = self.regime_probs[s:e]  # (seq_len, regime_dim)
        return item

    @property
    def valid_starts(self) -> np.ndarray:
        """Original row offsets whose complete sequence is eligible."""
        return self._valid_starts.copy()

    @property
    def row_eligible(self) -> np.ndarray:
        """Required-source eligibility for each original body row."""
        return self._row_eligible.copy()


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
        availability: pd.DataFrame | Mapping[str, Any] | None = None,
        interval: str | None = None,
        include_funding: bool | None = None,
        include_mark: bool | None = None,
    ):
        """
        Args:
            features_df: 特徴量 DataFrame（rolling z-score 正規化済み）
            returns: 生リターン Series（Oracle/バックテスト用）
            split: WFOSplit（train/val/test の時刻境界）
            seq_len: シーケンス長
            oracle_actions: Oracle から得た行動列（BC 学習時）
            availability: v4 availability sidecar.  When omitted, a sidecar
                attached to ``features_df.attrs`` is consumed automatically.
            interval: bar interval used for contiguous-window checks.
            include_funding/include_mark: required external availability flags.
        """
        self.split = split
        self.seq_len = seq_len
        self._feature_columns = list(features_df.columns)

        attrs = getattr(features_df, "attrs", {})
        if availability is None and isinstance(attrs, Mapping):
            availability = attrs.get("availability")
        if availability is None and isinstance(getattr(returns, "attrs", {}), Mapping):
            availability = returns.attrs.get("availability")
        if interval is None and isinstance(attrs, Mapping):
            interval = attrs.get("availability_interval")
        self.interval = str(interval or "15m")
        if include_funding is None and isinstance(attrs, Mapping):
            include_funding = attrs.get("availability_include_funding")
        if include_mark is None and isinstance(attrs, Mapping):
            include_mark = attrs.get("availability_include_mark")
        self.include_funding = True if include_funding is None else bool(include_funding)
        self.include_mark = True if include_mark is None else bool(include_mark)
        self._availability = availability
        self._full_row_eligible: np.ndarray | None = None
        if availability is not None:
            try:
                selected = validate_availability(
                    availability,
                    features_df.index,
                    include_funding=self.include_funding,
                    include_mark=self.include_mark,
                )
            except AvailabilityContractError as exc:
                raise ValueError(str(exc)) from exc
            self._availability = selected.sidecar
            self._full_row_eligible = selected.row_eligible

        # 各 split のデータを切り出す
        # train/val/test すべて右端 exclusive。
        # test を inclusive にすると test_end == 次 fold の test_start のバーが
        # 両方の fold で二重計上される（最終 fold の末尾 1 バー欠落より悪い）。
        def _slice(start, end):
            mask = (features_df.index >= start) & (features_df.index < end)
            selected_index = features_df.index[mask]
            feat = features_df.loc[mask].to_numpy()
            ret = returns.loc[mask].to_numpy()
            selected_availability = None
            selected_eligible = None
            if self._availability is not None:
                if isinstance(self._availability, pd.DataFrame):
                    selected_availability = self._availability.loc[selected_index]
                else:
                    selected_availability = {
                        column: np.asarray(values)[mask]
                        for column, values in self._availability.items()
                    }
                selected_eligible = self._full_row_eligible[mask]
                feat = FeatureArray(
                    feat,
                    index=selected_index,
                    availability=selected_availability,
                    interval=self.interval,
                    include_funding=self.include_funding,
                    include_mark=self.include_mark,
                    row_eligible=selected_eligible,
                )
            return feat, ret, selected_index, selected_availability, selected_eligible

        (
            self._train_feat,
            self._train_ret,
            self._train_index,
            self._train_availability,
            self._train_row_eligible,
        ) = _slice(split.train_start, split.train_end)
        (
            self._val_feat,
            self._val_ret,
            self._val_index,
            self._val_availability,
            self._val_row_eligible,
        ) = _slice(split.val_start, split.val_end)
        (
            self._test_feat,
            self._test_ret,
            self._test_index,
            self._test_availability,
            self._test_row_eligible,
        ) = _slice(split.test_start, split.test_end)

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
            timestamps=self._train_index if self._availability is not None else None,
            availability=self._train_availability,
            interval=self.interval,
            include_funding=self.include_funding,
            include_mark=self.include_mark,
        )

    def val_dataset(self) -> SequenceDataset:
        return SequenceDataset(
            self._val_feat,
            self.seq_len,
            returns=self._val_ret,
            timestamps=self._val_index if self._availability is not None else None,
            availability=self._val_availability,
            interval=self.interval,
            include_funding=self.include_funding,
            include_mark=self.include_mark,
        )

    def test_dataset(self) -> SequenceDataset:
        return SequenceDataset(
            self._test_feat,
            self.seq_len,
            returns=self._test_ret,
            timestamps=self._test_index if self._availability is not None else None,
            availability=self._test_availability,
            interval=self.interval,
            include_funding=self.include_funding,
            include_mark=self.include_mark,
        )

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

    @property
    def availability(self) -> pd.DataFrame | Mapping[str, np.ndarray] | None:
        """Validated v4 sidecar, or ``None`` for historical v3 data."""
        return self._availability

    @property
    def train_availability(self):
        return self._train_availability

    @property
    def val_availability(self):
        return self._val_availability

    @property
    def test_availability(self):
        return self._test_availability

    @property
    def train_row_eligible(self) -> np.ndarray:
        return (
            np.ones(len(self._train_feat), dtype=bool)
            if self._train_row_eligible is None
            else self._train_row_eligible.copy()
        )

    @property
    def val_row_eligible(self) -> np.ndarray:
        return (
            np.ones(len(self._val_feat), dtype=bool)
            if self._val_row_eligible is None
            else self._val_row_eligible.copy()
        )

    @property
    def test_row_eligible(self) -> np.ndarray:
        return (
            np.ones(len(self._test_feat), dtype=bool)
            if self._test_row_eligible is None
            else self._test_row_eligible.copy()
        )
