# Optimization Loop: Issue 6 External Source Evaluation

## 概要

Issue 6 では、外部ソースの寄与を `basis -> orderflow -> onchain -> hybrid` の順で比較する。
ここは学習原理側の issue を先に見た後、必要な場合だけ進める。

## 実行対象

- [source_rollout_suite_free.yaml](../configs/source_rollout_suite_free.yaml)
- [run_free_source_rollout_end_to_end.ps1](../scripts/run_free_source_rollout_end_to_end.ps1)
- [select_best_source_family.ps1](../scripts/select_best_source_family.ps1)

## 実行順

1. free source rollout を通す
2. source family suite を回す
3. `best_source_family.md` を作る
4. orderflow / onchain / hybrid のどれが最も OOS でマシかを判定する

## 実行コマンド

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_issue6_external_source_loop.ps1
```

## 判定

- `test_alpha_pt_mean`
- `test_sharpe_delta_mean`
- `win_rate_vs_bh_mean`
- `m2_pass_count`

だけでなく、`basis` 比でどの程度マシかを見る。

## 次の分岐

- 明確に良い family がある:
  - その source family を固定して学習側の issue に戻す
- 差が小さい:
  - 外部ソースは補助要因とみなし、主戦場は learning principle 側に置く
