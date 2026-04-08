# Optimization Loop: Issue 6 External Source Evaluation

## 位置づけ
- issue1 から issue5 まで見た時点で、主因は `teacher / BC / AC` の学習原理側にある可能性が高い。
- それでも source family に改善余地があるかを切り分けるため、既存の source family suite を比較した。
- 比較軸は `basis` と `orderflow` の 2 系統。

## 既存 suite summary
`checkpoints/source_family_suite/suite_summary.csv`

- `basis`
  - `m2_pass_count = 0`
  - `test_alpha_pt_mean = +1.0`
  - `test_sharpe_delta_mean = +0.05`
  - `test_win_rate_mean = 0.51`
- `orderflow`
  - `m2_pass_count = 1`
  - `test_alpha_pt_mean = +4.5`
  - `test_sharpe_delta_mean = +0.18`
  - `test_win_rate_mean = 0.58`

## 判定
- 現時点の source family 比較では `orderflow > basis`
- 劇的改善ではないが、既存 source の中では `orderflow` がいちばん生き残っている
- したがって、次に learner family を切るなら `basis` ではなく `orderflow` を優先する

## 含意
- source が主因ではないという整理は維持
- ただし、source family を選ぶ必要があるなら `orderflow` を使う方が筋が良い
- 次の本命は `orderflow` を入れたまま `BC collapse` を起こさない learner を探すこと

## 実行入口
- suite: `scripts/run_issue6_external_source_loop.ps1`
- best family selection: `scripts/select_best_source_family.ps1`
