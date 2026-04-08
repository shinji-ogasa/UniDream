# Optimization Loop: Issue 2 BC Prior Audit

## 目的
- teacher が動いているのに BC prior が teacher 構造を壊しているかを定量確認する
- regime 別 mismatch を見て、崩壊が全 regime 共通かを切り分ける

## 診断
- 実行: `audit_bc_prior.py`
- 見るもの:
  - `teacher_short_ratio / bc_short_ratio`
  - `teacher_flat_ratio / bc_flat_ratio`
  - `mean_abs_gap / rmse_gap`
  - `short_mismatch / flat_mismatch`
  - turnover 差

## baseline 判定
- `medium_v2` fold 4 の既存 checkpoint では
  - teacher: `short 50% / flat 50%`
  - BC: `short 100% / flat 0%`
- つまり issue2 は `true` で、teacher の前に BC prior が構造を壊していた

## pre-Web の 3 本
teacher は共通で `signal_aim` を使い、fold 4 の BC-only tiny で比較した。

1. `weighted BC`
2. `sequence BC`
3. `residual BC`

### 結果
- teacher: `short 39.6% / flat 60.4%`
- `weighted BC tiny`
  - BC: `short 99.9986% / flat 0.0014%`
  - `mean_abs_gap 0.1909`
- `sequence BC tiny`
  - BC: `short 99.9986% / flat 0.0014%`
  - `mean_abs_gap 0.1909`
- `residual BC tiny`
  - BC: `short 99.9971% / flat 0.0029%`
  - `mean_abs_gap 0.2358`

3 本とも `teacher -> BC` の collapse を止められなかったため、ルール通り Web を挟んだ。

## post-Web の 2 本
Web 後は「teacher marginal を直接守る」方向に寄せた。

4. `class_balanced BC`
5. `class_balanced + residual BC`

### 結果
- `class_balanced BC tiny`
  - BC: `short 99.9986% / flat 0.0014%`
  - `mean_abs_gap 0.3138`
- `class_balanced + residual BC tiny`
  - BC: `short 99.9986% / flat 0.0014%`
  - `mean_abs_gap 0.3125`

loss 自体は下がったが、行動分布はむしろ悪化した。

## 結論
- issue2 の主因判定は `true`
- ただし、既存 BC family の改善余地はこの枝ではかなり薄い
- `weighted / sequence / residual` の 3 本でも改善なし
- Web 後の `class_balanced` 系 2 本でも改善なし
- 現時点では、BC objective そのものが teacher marginal を維持できず、全 regime 共通で `short 100%` に崩壊している

## 次の分岐
- `issue3`: AC が BC prior からどこでどれだけ逸脱しているかを計測する
- `issue4`: WM の regime 表現不足が collapse を助長していないかを計測する

## 採否
- keep:
  - `signal_aim` teacher は維持
- kill:
  - `weighted BC`
  - `sequence BC`
  - `residual BC`
  - `class_balanced BC`
  - `class_balanced + residual BC`
