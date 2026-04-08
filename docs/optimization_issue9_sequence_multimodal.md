# Optimization Loop: Issue 9 Sequence / Multimodal Policy Family

## 位置づけ

issue8 までで、`continuous target head + regime gate + execution_aux` が current best になった。
ただし `bc_short 0.969` で collapse 自体はまだ強い。

issue9 では、1-step mapping をさらに崩して

- chunked policy
- self-conditioning
- small path auxiliary

を組み合わせた sequence-style learner family を試す。

## 最初の候補

1. `medium_l0_bc_seqchunk_exec`
   - chunked policy
   - self-conditioning
   - small `path_aux`

## 判定基準

- current best を超える最低条件:
  - `bc_short_ratio < 0.969`
  - `teacher_to_bc_mean_abs_gap < 0.0576`
- ただし `flat 100%` は棄却
