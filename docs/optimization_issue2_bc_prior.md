# Optimization Loop: Issue 2 BC Prior の再現性診断

## 課題
- teacher は `signal_aim` まで改善した
- それでも BC prior が teacher を再現できず `short 100%` へ collapse しているかを確認する

## baseline 診断

### `medium_l1_bc_continuous_regimegate_exec`
- `teacher_short 0.353`
- `teacher_flat 0.647`
- `bc_short 0.999`
- `bc_flat 0.001`
- `teacher_to_bc_mean_abs_gap 0.145`

判定:
- issue2 は true
- 主因は BC collapse

## current learner family の比較

### `medium_l1_bc_continuous_exec_shortmass_align`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1287`
- `bc_short_ratio = 0.9980`
- `bc_flat_ratio = 0.0020`

判定:
- 現時点の current keep
- まだ collapse は強いが baseline よりは改善

### `medium_l0_bc_continuous_signalaim_regimegate_exec`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1424`
- `bc_short_ratio = 0.9980`
- `bc_flat_ratio = 0.0020`

判定:
- `medium_l1_bc_continuous_exec_shortmass_align` を更新できない
- reject

## 結論
- issue2 は主因のまま
- `signal_aim teacher` を使っても BC は still `short 100%` 近辺へ collapse する
- 現時点の learner keep は `medium_l1_bc_continuous_exec_shortmass_align`

## 次
- issue10 の action-head 側を優先
- `logits blend 0.50` を基準に、training-side で同等の改善を出せる learner 枝を探す
