# Optimization Loop: Issue 3 AC の support 逸脱診断

## 狙い
Issue 3 では、`BC prior -> AC` の遷移で policy がさらに崩れているのかを局所診断する。

見るものは次の通り。

- `teacher_short_ratio`
- `bc_short_ratio`
- `ac_short_ratio`
- `bc_flat_ratio`
- `ac_flat_ratio`
- `bc_to_ac_mean_abs_gap`
- `teacher_to_ac_mean_abs_gap`
- `bc_to_ac_short_mismatch`
- `bc_to_ac_flat_mismatch`
- `teacher_to_ac_short_mismatch`
- `bc_turnover`
- `ac_turnover`

## 診断スクリプト

- [audit_ac_support.py](../audit_ac_support.py)
- [ac_support_audit.py](../unidream/experiments/ac_support_audit.py)

既存 checkpoint の

- `world_model.pt`
- `bc_actor.pt`
- `ac_best.pt` または `ac.pt`

を読み、teacher / BC / AC の行動分布を比較する。

## 真偽確認

対象は `medium_v2_fix` の fold 4。
feature family は checkpoint に合わせて raw-only の config を使う。

- config: [medium_ext_sources_rawonly.yaml](../configs/medium_ext_sources_rawonly.yaml)
- 出力:
  - [AC support summary](../checkpoints/medium_v2_fix/ac_support_audit/medium_ext_sources_rawonly_ac_support_audit_summary.csv)

実行は軽く切るため、`val` の末尾 `4096` bars に限定した。

## 結果

### fold 4 / val(all)

- teacher: `short 49.7%`
- BC: `short 100.0% / flat 0.0%`
- AC: `short 100.0% / flat 0.0%`
- `bc_to_ac_mean_abs_gap = 0.092`
- `teacher_to_ac_mean_abs_gap = 0.500`
- `bc_to_ac_short_mismatch = 0.000`
- teacher turnover: `394.0`
- AC turnover: `20.4`

### regime 別

- `regime_0`: BC short `100%`, AC short `100%`
- `regime_1`: BC short `100%`, AC short `100%`
- `regime_2`: BC short `100%`, AC short `100%`

## 判定

Issue 3 の真偽判定は次の通り。

- **AC が BC をさらに壊している**: false
- **BC の時点でほぼ崩壊している**: true

つまり、いまの失敗は `BC -> AC drift` より前に起きている。
AC はむしろ turnover を落としているが、行動分布自体は BC の `short 100%` をそのまま引き継いでいる。

## 次の遷移

Issue 3 はここで一段閉じる。
次は `issue4: WM の regime 表現` を確認し、

- latent が regime をほとんど持てていないのか
- それとも WM はそこそこだが BC 出力設計が潰しているのか

を切る。
