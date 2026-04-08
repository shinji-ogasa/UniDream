# Optimization Loop: Issue 3 AC Support Drift

## 概要

Issue 3 では、`BC prior` から `AC` がどの程度離れているかを先に監査する。
ここでは full 実験を先に広げず、既存 checkpoint だけを使って

- `BC -> AC` の行動分布差
- regime 別 mismatch
- `teacher -> AC` の乖離
- turnover の増加

を確認する。

## 診断コード

- [audit_ac_support.py](../audit_ac_support.py)
- [ac_support_audit.py](../unidream/experiments/ac_support_audit.py)

この監査は既存の checkpoint

- `world_model.pt`
- `bc_actor.pt`
- `ac_best.pt` または `ac.pt`

を読み、train / val で `teacher / BC / AC` の位置系列を比較する。

## 実行コマンド

まずは `medium_v2` の既存 checkpoint で issue3 を切る。

```powershell
uv run python audit_ac_support.py `
  --config configs/medium_v2.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --cache-dir checkpoints/data_cache `
  --checkpoint-dir checkpoints `
  --folds 4 `
  --device cuda
```

## 見る指標

- `teacher_short_ratio`
- `bc_short_ratio`
- `ac_short_ratio`
- `bc_flat_ratio`
- `ac_flat_ratio`
- `bc_to_ac_mean_abs_gap`
- `teacher_to_ac_mean_abs_gap`
- `bc_to_ac_short_mismatch`
- `bc_to_ac_flat_mismatch`
- `teacher_to_ac_short_mismatch`
- `bc_turnover`
- `ac_turnover`

## 判定

次のどちらかで判定する。

1. `AC` が `BC` から大きく離れていて、しかも `teacher` にも近づいていない  
   - `AC support drift` が主因寄り
2. `AC` は `BC` から大きく離れていない  
   - 先に `WM` か `BC prior` 側を疑う

## 候補アルゴリズム

Issue 3 の候補はこの 3 本に絞る。

### 1. KL budget を強めた constrained AC

- 目的:
  - `BC prior` からの逸脱を抑える
  - `val` でだけ派手な action に飛ぶのを防ぐ

### 2. SPIBB 風 support 制約

- 目的:
  - dataset support が薄い状態では baseline / BC に戻す
  - offline RL の OOD drift を抑える

### 3. IQL / advantage clipping 寄りの保守化

- 目的:
  - value の楽観を抑えながら改善する
  - `BC prior` を急に捨てない

## 優先順位

Issue 3 の比較順はこの順で固定する。

1. `KL-constrained AC`
2. `SPIBB / support budget`
3. `IQL / clipped-advantage`

理由:

- まず今の AC が `BC` から離れすぎているかを切る
- その後に、最も実装差分の小さい制約強化から試す

## 次の分岐

AC support audit の結果で次を決める。

- `BC -> AC` の drift が大きい:
  - Issue 3 の候補比較へ進む
- drift が小さい:
  - Issue 4 の `WM regime 補助目的` を先に見る
