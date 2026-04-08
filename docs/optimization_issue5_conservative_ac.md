# Optimization Loop: Issue 5 Conservative Offline AC

## 概要

Issue 5 では、`AC` をそのまま延長するのではなく、今の実装ノブでより保守的な offline AC に寄せる。
対象は

- `KL budget`
- `support budget`
- `conservative actor update`

の 3 本に絞る。

## 候補

### 1. KL budget

- config: [medium_l1_ac_klbudget.yaml](../configs/medium_l1_ac_klbudget.yaml)
- 狙い:
  - `BC prior` からの離れすぎを抑える
  - `alpha` をゆっくり落とし、prior を長く残す

### 2. support budget

- config: [medium_l1_ac_supportbudget.yaml](../configs/medium_l1_ac_supportbudget.yaml)
- 狙い:
  - turnover / flow change / prior flow を強める
  - support 外 action を抑える

### 3. conservative AC

- config: [medium_l1_ac_conservative.yaml](../configs/medium_l1_ac_conservative.yaml)
- 狙い:
  - `positive_advantages=false`
  - `alpha_final` を高く保つ
  - `td3bc_alpha` を下げて actor 更新を保守化する

## 実行順

1. `KL budget`
2. `support budget`
3. `conservative AC`

## 判定

- `alpha_excess`
- `sharpe_delta`
- `win_rate_vs_bh`
- `collapse_guard`
- `BC -> AC` の drift

をセットで見る。
