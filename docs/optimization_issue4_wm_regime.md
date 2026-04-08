# Optimization Loop: Issue 4 WM Regime Representation

## 概要

Issue 4 では、`world model` の latent が regime をどこまで表現できているかを先に監査する。
ここでは full 学習を増やす前に、既存 checkpoint を使って

- 現在 regime の線形分離性
- 次 regime の予測可能性

を確認する。

## 診断コード

- [audit_wm_regime.py](../audit_wm_regime.py)
- [wm_regime_audit.py](../unidream/experiments/wm_regime_audit.py)

この監査は `world_model.pt` を読み、encoded latent (`z`, `h`) に対する線形 probe を作る。
まず train latent で ridge 多クラス分類器を学習し、その probe で train / val の

- `current_regime`
- `next_regime`

を予測する。

## 実行コマンド

```powershell
uv run python audit_wm_regime.py `
  --config configs/medium_v2.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --cache-dir checkpoints/data_cache `
  --checkpoint-dir checkpoints `
  --folds 4 `
  --device cuda
```

## 見る指標

- `accuracy`
- `balanced_accuracy`
- `macro_f1`

とくに `val/current_regime` と `val/next_regime` を重視する。

## 判定

次のどちらかで判定する。

1. `current_regime` も `next_regime` も低い  
   - WM latent が regime transition を十分に持てていない
2. `current_regime` は取れるが `next_regime` が低い  
   - static state は持てているが transition 表現が弱い

## 候補アルゴリズム

Issue 4 の候補はこの 3 本に絞る。

### 1. regime prediction 補助タスク

- 目的:
  - latent に regime ラベルを明示的に持たせる

### 2. next-regime prediction 補助タスク

- 目的:
  - transition 情報を latent に押し込む

### 3. regime-conditioned latent / capacity 増強

- 目的:
  - regime 別の状態分布をより分離して持たせる

## 優先順位

1. `current_regime / next_regime` の線形 probe
2. `regime prediction` 補助
3. `next-regime prediction`
4. 必要なら capacity 増強
