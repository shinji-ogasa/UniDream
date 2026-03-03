"""オンライン Fine-tune モジュール（Phase 3）.

バックテスト結果確認後、実環境データで世界モデルを逐次更新する。

現在はスケルトンのみ。
"""
from __future__ import annotations


class OnlineFinetuner:
    """実環境データによる世界モデルの逐次更新.

    Phase 3 で実装予定。
    バックテスト結果を見てから検討する。
    """

    def __init__(self):
        raise NotImplementedError("Phase 3 で実装予定")

    def update(self, obs, action, reward, done):
        """1 ステップのオンライン更新."""
        raise NotImplementedError
