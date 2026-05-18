# UniDream — Development Guide

開発時の方針・規約をまとめたファイル。詳細仕様は SPEC.md を参照。

---

## プロジェクトの目的

Transformer 世界モデル + Imagination Actor-Critic による暗号資産トレーディング方策の学習。
DreamerV3 ベースで RSSM を Transformer に置換し、Hindsight Oracle で初期化した上で AC fine-tune する。

## 開発方針

- **バックテスト基盤を最初に固める**: WFO・コスト・PBO が正しく動かないと学習結果が意味をなさない
- **model-free PPO をベースラインとして先に回す**: 世界モデルの寄与を差分で確認
- **変数を最小化して検証する**: LoRe（LLM/Risk Gate）は Phase 1 で alpha 確認後に導入
- **未来情報リークを防ぐ**: shift(1) を必ず適用、oracle は train 期間のみで計算

## DL Framework

PyTorch (>= 2.0)。世界モデル・Actor・Critic すべて統一。

## 参照ドキュメント

- `SPEC.md` — アーキテクチャ・パイプライン・評価の詳細仕様
- `README.md` — プロジェクト概要
