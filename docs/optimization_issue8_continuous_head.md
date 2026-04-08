# Optimization Loop: Issue 8 Continuous Target Head

## 位置づけ

issue7 で、既存の 1-step CE 系 actor family はほぼ全部 `BC short ≈ 1.0` に collapse すると分かった。
そのため issue8 では、`continuous target head` を使って teacher marginal を保持できる learner family を探している。

## 試した枝

### 1. continuous target head

- `medium_l0_bc_continuous`
  - `teacher_short 0.163`
  - `bc_short 0.992`
  - `teacher_to_bc_mean_abs_gap 0.0607`

### 2. continuous + regime gate

- `medium_l0_bc_continuous_regimegate`
  - `teacher_short 0.163`
  - `bc_short 0.989`
  - `gap 0.0595`

### 3. continuous + regime gate + execution_aux

- `medium_l0_bc_continuous_regimegate_exec`
  - `teacher_short 0.163`
  - `bc_short 0.969`
  - `gap 0.0576`

これは issue8 の current best。

## 棄却した枝

- `direct target track`
  - current best を超えず棄却
- `path_aux`
  - `flat 100%` に反転して棄却
- `signal_aim + raw-only orderflow`
  - `bc_short 0.999`, `gap 0.1492`
- baseline source の `signal_aim + regime gate + execution_aux`
  - `bc_short 0.998`, `gap 0.1424`
- `controller_state_dim=3`
  - `flat 100%`, `gap 0.0537`
- code-level split execution head
  - `bc_short 0.999`, `gap 0.0593`

## inference-only regime gate branch

current best checkpoint に対して、学習なしで `policy_collapse_audit` だけを当てる
inference-only 5 本を試した。

- `medium_l0_bc_continuous_regimegate_exec_r0only`
  - `bc_short 0.001`
  - `bc_flat 0.999`
  - `gap 0.0289`
- `medium_l0_bc_continuous_regimegate_exec_r0support`
  - `bc_short 0.001`
  - `bc_flat 0.999`
  - `gap 0.0289`
- `medium_l0_bc_continuous_regimegate_exec_r0uncertainty`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0289`
- `medium_l0_bc_continuous_regimegate_exec_r0soft`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0300`
- `medium_l0_bc_continuous_regimegate_exec_r0soft_uncertainty`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0289`

### 結論

- hard/soft どちらの regime gate でも、collapse は減るが flat へ過補正した
- support / uncertainty を足しても current best を超えなかった
- inference-only family はここで打ち切る

## 現時点の結論

- issue8 の current best は引き続き
  `continuous target head + regime gate + execution_aux`
- 次の本命は、`execution_aux` を維持した learner-loss branch
