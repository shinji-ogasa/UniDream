# UniDream — Development Guide

開発時の方針・規約をまとめたファイル。詳細仕様は SPEC.md を参照。

## プロジェクトの目的

Plan009 depth calibrator による BTCUSDT 15分足のリスクオフ制御を検証し、HF Space のリアルタイムデモへ反映する。

## 開発方針

- 現行仕様は Plan009 depth-calibrated past-only guard + execution compression。
- Runtime signal は shifted trailing-return features のみを使う。
- fold0-12 は開発セットとして扱い、pristine holdout の主張をしない。
- 変更後は `unidream-space` の sample parity を必ず確認する。
- README / SPEC / bundle manifest は current bundle と同じ仕様に揃える。

## 参照ドキュメント

- `SPEC.md` — 現行仕様・entrypoint・評価条件
- `README.md` — プロジェクト概要と使い方
