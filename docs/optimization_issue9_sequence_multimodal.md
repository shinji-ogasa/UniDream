# Optimization Loop: Issue 9 Sequence / Multimodal Policy Family

## 位置づけ

issue8 までで、`continuous target head + regime gate + execution_aux` が current best になった。
ただし `bc_short 0.969` で collapse 自体はまだ強い。

issue9 では、1-step mapping をさらに崩して

- chunked policy
- self-conditioning
- small path auxiliary

を組み合わせた sequence-style learner family を試す。

## 候補

1. `medium_l0_bc_seqchunk_exec`
   - chunked policy
   - self-conditioning
   - small `path_aux`
2. `medium_l0_bc_exec_selfcond`
   - current best に self-conditioning だけ追加
   - heavy 化を避ける軽量枝

## 判定基準

- current best を超える最低条件:
  - `bc_short_ratio < 0.969`
  - `teacher_to_bc_mean_abs_gap < 0.0576`
- ただし `flat 100%` は棄却

## 実行結果

### `medium_l0_bc_seqchunk_exec`

- `fold4 / stop-after bc` でも 240 秒超
- heavy 禁止に反するため、この枝は runtime 理由で棄却
- 次は `self-conditioning only` へ縮める

### `medium_l0_bc_exec_selfcond`

- `fold4 / resume / start-from bc / stop-after bc` でも 150 秒超
- `bc_actor.pt` まで到達せず、heavy 制約に反する

## 中間結論

- issue9 family は現状の実装だと runtime cost が高すぎる
- 重い sequence branch は一旦保留
- 次は `current best` を維持したまま、より軽い learner/inference 枝へ戻す
