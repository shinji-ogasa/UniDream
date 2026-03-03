"""LLM Embedding モジュール（Phase 2: LoRe）.

ニュース・イベント・センチメントを LLM で embed し、
世界モデルの入力に結合する。

Phase 1 で alpha が確認できてから実装する。
現在はスケルトンのみ。
"""
from __future__ import annotations

import numpy as np


class LLMEmbedder:
    """ニュース/イベントテキスト → LLM embedding.

    Phase 2 で実装予定。
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        raise NotImplementedError("Phase 2 で実装予定")

    def embed(self, texts: list[str]) -> np.ndarray:
        """テキストリストを embedding に変換する.

        Returns:
            embeddings: (N, embed_dim)
        """
        raise NotImplementedError
