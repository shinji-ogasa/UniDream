"""Risk Gate モジュール（Phase 2: LoRe）.

重大イベント検知時に Actor のポジションを強制縮小 / フラット化する。

Phase 1 で alpha が確認できてから実装する。
現在はスケルトンのみ。
"""
from __future__ import annotations

import numpy as np


class RiskGate:
    """Uncertainty gating による リスク制御.

    Phase 2 で実装予定。
    """

    def __init__(self, uncertainty_threshold: float = 0.8):
        raise NotImplementedError("Phase 2 で実装予定")

    def gate(self, positions: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        """不確実性が高い場合にポジションをゼロに強制する.

        Args:
            positions: (T,) ポジション比率
            uncertainty: (T,) モデルの不確実性スコア

        Returns:
            gated_positions: (T,)
        """
        raise NotImplementedError
