# Optimization Loop: Issue 2 BC Prior の再現性診断

## 目的

teacher が動いていても、BC prior がそれを再現できず `short 100%` に潰れている可能性を切る。

見るもの:
- teacher と BC の `short / flat` 比率差
- `teacher_to_bc_mean_abs_gap`
- regime 別 mismatch
- turnover 差

## 過去の L0 で分かったこと

既存の L0 比較では、以下の BC 側手当てを入れても根本改善は出なかった。

- `weighted BC`
- `sequence BC`
- `residual`
- `class-balanced`
- `class-balanced residual`

結論:
- BC family は teacher marginal を保持できず、ほぼ `short 100%` に collapse する

## 今回の正式確認

比較条件:
- teacher: `signal_aim`
- learner family: `continuous target head + regime gate + execution_aux`
- 実行: fold 4 / `--stop-after bc`
- audit: `policy_collapse_audit`

### `medium_l1_bc_continuous_regimegate_exec`

- `teacher_short 0.3530`
- `teacher_flat 0.6470`
- `bc_short 0.9985`
- `bc_flat 0.0015`
- `trade_prob_mean 0.0884`
- `short_target_mass_mean 0.1393`
- `baseline_target_mass_mean 0.8607`
- `teacher_to_bc_mean_abs_gap 0.1453`

## 判定

issue2 の判定は **true**。

理由:
- issue1 で teacher は `signal_aim` に改善しても
- current best learner family で BC を回すと
  `teacher_short 0.353 -> bc_short 0.999`
  まで潰れる
- つまり主因は teacher 単独ではなく、BC prior の再現失敗

補足:
- `short_target_mass_mean 0.139` までは出ている
- それでも final BC action は `short 0.999`
- なので collapse は target marginal より後段、つまり actor / inference 側で増幅されている

## 結論

- issue2 は確定
- BC prior は teacher を再現できていない
- しかも単純な weighted / sequence / residual だけでは直っていない

## 次

- issue3 `AC support audit`
- ここで見るのは
  - AC が BC よりさらに support 外へ出ているか
  - それとも主に BC collapse の持ち越しなのか
