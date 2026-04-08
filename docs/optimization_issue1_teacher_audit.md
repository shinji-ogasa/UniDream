# Optimization Loop: Issue 1 Teacher Audit

## 目的
`teacher / oracle が本当に弱いか` を full 学習なしで先に切る。

見たかった点は次の3つ。
- regime 別に `long / flat / short` がどう動いているか
- `min_hold` を振っても teacher 分布が変わるか
- bearish regime で flat 化する teacher に改善できるか

## 対象
- baseline: [medium_v2.yaml](../configs/medium_v2.yaml)
- external source 版 baseline: [medium_ext_sources.yaml](../configs/medium_ext_sources.yaml)
- L1 比較候補:
  - [medium_l1_base.yaml](../configs/medium_l1_base.yaml)
  - [medium_l1_signal_aim.yaml](../configs/medium_l1_signal_aim.yaml)
  - [medium_l1_signal_downside.yaml](../configs/medium_l1_signal_downside.yaml)
  - [medium_l1_feature_stress.yaml](../configs/medium_l1_feature_stress.yaml)
  - [medium_l1_feature_dual.yaml](../configs/medium_l1_feature_dual.yaml)

実装:
- [audit_teacher_regimes.py](../audit_teacher_regimes.py)
- [teacher_audit.py](../unidream/experiments/teacher_audit.py)

出力:
- [medium_v2 audit summary](../checkpoints/teacher_audit/medium_v2/medium_v2_teacher_audit_summary.csv)
- [medium_ext_sources audit summary](../checkpoints/teacher_audit/medium_ext_sources/medium_ext_sources_teacher_audit_summary.csv)
- [teacher_audit_l1](../checkpoints/teacher_audit_l1)

## baseline の結論
- `medium_v2` と `medium_ext_sources` の baseline teacher は、ほぼ `long 0% / short 50% / flat 50%`
- bearish regime でもほぼ同じ
- `min_hold = 16, 32, 64` を振っても、主に `avg_hold` が変わるだけで行動分布はほぼ不変

つまり baseline teacher は
- regime 0/1 で十分に flat 化しない
- `min_hold` 調整だけでは改善しない

## fold 4 の L1 teacher 比較

### baseline
- all: `short 50.1% / flat 49.9%`
- bearish: `short 50.4% / flat 49.6%`
- avg_hold: `23.2 bars`

### signal_aim
- all: `short 36.6% / flat 63.4%`
- bearish: `short 36.2% / flat 63.8%`
- avg_hold: `3.3 bars`

### signal_aim + downside
- all: `short 66.6% / flat 33.4%`
- bearish: `short 66.5% / flat 33.5%`
- avg_hold: `1.7 bars`

### feature_stress
- all: `short 17.2% / flat 82.8%`
- bearish: `short 34.7% / flat 65.3%`
- regime 0 val: `short 55.6% / flat 44.4%`
- regime 1 val: `short 13.8% / flat 86.2%`
- regime 2 val: `short 14.0% / flat 86.0%`
- avg_hold: `4.6 bars`

### feature_dual
- all: `short 9.2% / flat 90.8%`
- bearish: `short 24.0% / flat 76.0%`
- regime 0 val: `short 42.3% / flat 57.7%`
- regime 1 val: `short 5.8% / flat 94.2%`
- regime 2 val: `short 6.8% / flat 93.2%`
- avg_hold: `7.2 bars`

## 解釈
- `signal_aim` は baseline より明確に flat を増やすが、regime ごとの差はまだ弱い
- `signal_aim + downside` は short に寄りすぎていて棄却
- `feature_stress` は regime 0 の高ボラ局面でだけ short を強め、regime 1/2 では flat を多く保てている
- `feature_dual` はさらに保守的だが、de-risk teacher としては弱すぎる

## 判定
- `teacher / oracle が本当に弱いか`: `true`
- issue1 の次候補 teacher:
  1. `feature_stress`
  2. `signal_aim`
  3. `feature_dual`
- 棄却:
  - `signal_aim + downside`

## 次
issue1 はここで一段閉じる。

次は [optimization_issue2_bc_prior.md](./optimization_issue2_bc_prior.md) に進み、`teacher は動いているのに BC が潰しているか` を fold 4 の局所診断で切る。
