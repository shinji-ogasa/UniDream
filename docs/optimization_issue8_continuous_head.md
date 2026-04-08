# Optimization Loop: Issue 8 Continuous Target Head

## 位置づけ

issue7 で既存の 1-step CE 系 actor family はほぼ全部 `BC short ≈ 1.0` に collapse すると分かった。
そのため issue8 では、`continuous target head` を使って teacher marginal を保てる learner family を探した。

## current best

- `medium_l0_bc_continuous_regimegate_exec`
  - `teacher_short 0.163`
  - `bc_short 0.969`
  - `teacher_to_bc_mean_abs_gap 0.0576`

これは issue8 の current best。

## ここまでに切った枝

- `medium_l0_bc_continuous`
  - `gap 0.0607`
- `medium_l0_bc_continuous_regimegate`
  - `bc_short 0.989`
  - `gap 0.0595`
- `direct target track`
  - current best を超えず棄却
- `path_aux`
  - `flat 100%` で棄却
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
inference-only 7 本を試した。

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
- `medium_l0_bc_continuous_regimegate_exec_bootstrap`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0289`
- `medium_l0_bc_continuous_regimegate_exec_damped`
  - `bc_short 0.969`
  - `gap 0.0576`

### 結論

- hard/soft どちらの regime gate でも collapse は減るが flat へ過補正した
- damped branch は current best と同等で改善なし
- inference-only family はここで打ち切り

## learner-loss branch

`execution_aux` を維持したまま、continuous head の loss を 3 本切った。

- `medium_l0_bc_continuous_regimegate_exec_regimedist`
  - `bc_short 0.993`
  - `bc_flat 0.007`
  - `gap 0.0590`
  - current best より悪化
- `medium_l0_bc_continuous_regimegate_exec_shortmass`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0538`
  - over-flat で棄却
- `medium_l0_bc_continuous_regimegate_exec_distcombo`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0530`
  - over-flat で棄却

### 結論

- learner-loss 3 本でも current best を超えなかった
- `regime-dist` は short collapse を維持
- `short-mass / dist-combo` は flat collapse に反転

## post-Web 2 tries

Web では multi-modal action head が有望だったため、repo で近い既存 knob として
`target_from_logits` を 2 本試した。

- `medium_l0_bc_continuous_regimegate_exec_logitsblend`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0419`
- `medium_l0_bc_continuous_regimegate_exec_logitsfull`
  - `bc_short 0.000`
  - `bc_flat 1.000`
  - `gap 0.0289`

### 結論

- post-Web 2 本も collapse は減るが flat へ過補正した
- issue8 は current best を保持したまま打ち切る

## 最終結論

- issue8 の current best は引き続き
  `continuous target head + regime gate + execution_aux`
- ただし current best でも `bc_short 0.969` で collapse は強い
- 次は `sequence / multimodal policy family` に移る
