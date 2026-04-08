# Optimization Loop: Issue 2 BC Prior の再現性診断

## 狙い
Issue 2 では、teacher がある程度動いていても BC prior がそれを再現できず、
`short 100%` に潰しているのかを局所診断で確かめる。

見るものは次の通り。

- teacher と BC の `short / flat` 比率差
- regime 別 mismatch
- `mean_abs_gap / rmse_gap`
- turnover 差

ここでは full 実験は回さず、既存 checkpoint と L0 の軽い学習だけで切る。

## 診断スクリプト

- [audit_bc_prior.py](../audit_bc_prior.py)
- [bc_prior_audit.py](../unidream/experiments/bc_prior_audit.py)

診断は `world_model.pt` と `bc_actor.pt` を読み、train / val の teacher positions と BC positions を比較する。

## 既存 checkpoint の真偽確認

最初に、既存の `medium_v2_fix` checkpoint を raw-only の feature family で監査した。

- config: [medium_ext_sources_rawonly.yaml](../configs/medium_ext_sources_rawonly.yaml)
- 出力:
  - [medium_ext_sources_rawonly summary](../checkpoints/medium_v2_fix/bc_prior_audit/medium_ext_sources_rawonly_bc_prior_audit_summary.csv)

### fold 4 / val(all)

- teacher: `short 50.3% / flat 49.7%`
- BC: `short 100.0% / flat 0.0%`
- `mean_abs_gap = 0.497`
- `rmse_gap = 0.516`
- teacher turnover: `89.0`
- BC turnover: `543.9`

ここで、teacher が `50/50` で動いているのに BC が `short 100%` に潰していることが確認できた。
したがって、Issue 2 の真偽判定は **true**。

## Web を挟んだ候補整理

同系統 3 本で改善が薄かったので Web を挟んだ。
文献上、いまの症状に近い候補は次の 3 本だった。

1. `weighted BC`
   - mixed / noisy teacher で良いサンプルを重く学ぶ
   - `AWR / AWAC / DWBC` 系の発想
2. `sequence BC`
   - 1-step imitation の compounding mismatch を減らす
3. `class-balanced / support-aware BC`
   - support の薄い側に潰れる挙動を class weighting で抑える

Issue 2 の中では、まず次の 5 本を fold 4 / L0 で比較した。

## L0 比較候補

### 1. weighted BC

- config: [medium_l0_bc_weighted.yaml](../configs/medium_l0_bc_weighted.yaml)

### 2. sequence BC

- config: [medium_l0_bc_sequence.yaml](../configs/medium_l0_bc_sequence.yaml)

### 3. residual target tracking

- config: [medium_l0_bc_residual.yaml](../configs/medium_l0_bc_residual.yaml)

### 4. class-balanced BC

- config: [medium_l0_bc_balanced.yaml](../configs/medium_l0_bc_balanced.yaml)

### 5. class-balanced residual

- config: [medium_l0_bc_balanced_residual.yaml](../configs/medium_l0_bc_balanced_residual.yaml)

teacher は issue1 で採用した `feature_stress` に固定した。

## L0 結果

各設定の出力:

- [weighted summary](../checkpoints/medium_l0_bc_weighted_fold4/bc_prior_audit/medium_l0_bc_weighted_bc_prior_audit_summary.csv)
- [sequence summary](../checkpoints/medium_l0_bc_sequence_fold4/bc_prior_audit/medium_l0_bc_sequence_bc_prior_audit_summary.csv)
- [residual summary](../checkpoints/medium_l0_bc_residual_fold4/bc_prior_audit/medium_l0_bc_residual_bc_prior_audit_summary.csv)
- [balanced summary](../checkpoints/medium_l0_bc_balanced_fold4/bc_prior_audit/medium_l0_bc_balanced_bc_prior_audit_summary.csv)
- [balanced residual summary](../checkpoints/medium_l0_bc_balanced_residual_fold4/bc_prior_audit/medium_l0_bc_balanced_residual_bc_prior_audit_summary.csv)

### fold 4 / val(all) 比較

| config | teacher short | BC short | mean_abs_gap | rmse_gap | BC turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sequence` | 39.6% | 99.97% | 0.1406 | 0.1818 | 17.6 |
| `residual` | 39.6% | 99.99% | 0.1730 | 0.1951 | 25.3 |
| `weighted` | 39.6% | 99.99% | 0.1731 | 0.1951 | 25.3 |
| `balanced` | 39.6% | 99.99% | 0.1879 | 0.2043 | 66.8 |
| `balanced_residual` | 39.6% | 99.99% | 0.1884 | 0.2048 | 66.9 |

### 主要な観察

- 5 本とも BC はほぼ `short 100%` に潰れた
- 5 本とも `regime_1` / `regime_2` では BC short が実質 `100%`
- `sequence BC` が gap は最小だったが、崩壊自体は止められていない
- `class-balanced` は loss は下がったが、turnover が増えて mismatch も悪化した

## Issue 2 の判定

Issue 2 の結論は次の通り。

- **BC prior が teacher を再現できていない**: true
- **weighted / sequence / residual / balanced の L0 だけで改善できる**: false

つまり、Issue 2 の本質は

- teacher が動いていても
- BC の現在の出力設計と学習の仕方では
- support の外ではなく、むしろ `short 100%` 側へ collapse してしまう

ということ。

## 次の遷移

Issue 2 の枝はここで一段閉じる。
次は `issue3: AC の support 逸脱診断` に移り、

- BC -> AC でさらに崩しているのか
- それとも BC の時点でほぼ勝負が決まっているのか

を切る。
