# Optimization Loop: Issue 3 AC Support Drift

## 目的
- AC が BC prior からどの時点でどれだけ逸脱しているかを局所診断する
- `teacher -> BC` 崩壊のあと、さらに AC が悪化させているのかを切り分ける

## 軽量診断
- 実行: `audit_ac_support.py`
- 軽量化:
  - `--splits val`
  - `--max-bars 4096`

## baseline 結果
- config: `medium_v2`
- checkpoint: `checkpoints/fold_4`
- val 4096 bars の結果:
  - teacher short: `49.98%`
  - BC short: `100%`
  - AC short: `100%`
  - `bc_to_ac_short_mismatch = 0.0`
  - `bc_to_ac_flat_mismatch = 0.0`

## 判定
- baseline では AC drift は主因として薄い
- すでに BC が `short 100%` に崩壊しており、AC はその崩壊をほぼそのまま引き継いでいる
- issue3 単独では、まず BC collapse を超える主要ボトルネックには見えない

## 次の分岐
- `issue4`: WM が regime を十分に持てているか
- `issue5`: conservative AC の tiny 比較
