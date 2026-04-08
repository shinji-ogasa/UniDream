# Optimization Loop: Issue 2 BC Prior Audit

## 目的

Issue 2 では、`teacher は動いているのに BC prior がそれを潰しているのか` を確認する。

見るものは次の通り。

- teacher と BC の `short / flat` 比率差
- regime 別 mismatch
- `mean_abs_gap / rmse_gap`
- turnover 差

ここでは full 実験ではなく、既存 checkpoint からの局所診断を優先する。

## 診断スクリプト

- [audit_bc_prior.py](../audit_bc_prior.py)
- [bc_prior_audit.py](../unidream/experiments/bc_prior_audit.py)

この診断は既存の

- `world_model.pt`
- `bc_actor.pt`

を読み、`train / val` の teacher positions と BC positions を直接比較する。

## 使う checkpoint

まずは `medium_v2` の既存 checkpoint を対象にする。

想定コマンド:

```powershell
uv run python audit_bc_prior.py `
  --config configs/medium_v2.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --cache-dir checkpoints/data_cache `
  --checkpoint-dir checkpoints `
  --folds 4 `
  --device cuda
```

## 候補アルゴリズム

Issue 2 で試す候補は次の 3 本。

### 1. weighted BC

- config: [medium_l1_bc_weighted.yaml](../configs/medium_l1_bc_weighted.yaml)
- 狙い:
  - 良い teacher 点を重く学ぶ
  - 弱い teacher 点で policy が平均化するのを防ぐ

### 2. sequence BC

- config: [medium_l1_bc_sequence.yaml](../configs/medium_l1_bc_sequence.yaml)
- 狙い:
  - 1-step imitation ではなく短い path を合わせる
  - compounding mismatch を減らす

### 3. residual target tracking

- config: [medium_l1_bc_residual.yaml](../configs/medium_l1_bc_residual.yaml)
- 狙い:
  - continuous target を離散 CE に潰しすぎない
  - target inventory の追従を強める

## 優先順位

現時点の優先順位は次の通り。

1. `weighted BC`
2. `sequence BC`
3. `residual target tracking`

理由:

- issue1 で teacher 側の regime 感度不足は切れた
- issue2 の本丸は `teacher -> BC` の圧縮過程にある可能性が高い
- その場合、まず効きやすいのは sample weighting と sequence mismatch 対策

## 次の判定

BC prior audit の結果で次を決める。

- teacher と BC の乖離が大きい:
  - `weighted BC / sequence BC` に進む
- teacher と BC の乖離は小さいが test で崩れる:
  - issue3 の `AC support 逸脱` に進む

## ここでの出口条件

Issue 2 の出口条件は次のどちらか。

- BC が teacher をかなり潰していると確認できる
- BC は teacher を再現しているので、主因は AC 側だと切れる
