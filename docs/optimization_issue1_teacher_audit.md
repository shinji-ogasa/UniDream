# Optimization Loop: Issue 1 Teacher Audit

## 目的
まず `teacher / oracle` が本当に主因かを full 学習なしで確かめる。
確認したい点は次の 3 つ。

- regime 別に `long / flat / short` がどう出ているか
- `min_hold` が teacher 分布そのものを変えているか
- bearish regime で flat 化する teacher になっているか

## 実施対象

- baseline: [medium_v2.yaml](../configs/medium_v2.yaml)
- external-source 版: [medium_ext_sources.yaml](../configs/medium_ext_sources.yaml)
- L1 候補:
  - [medium_l1_base.yaml](../configs/medium_l1_base.yaml)
  - [medium_l1_signal_aim.yaml](../configs/medium_l1_signal_aim.yaml)
  - [medium_l1_signal_downside.yaml](../configs/medium_l1_signal_downside.yaml)
  - [medium_l1_feature_stress.yaml](../configs/medium_l1_feature_stress.yaml)
  - [medium_l1_feature_dual.yaml](../configs/medium_l1_feature_dual.yaml)

スクリプト:

- [audit_teacher_regimes.py](../audit_teacher_regimes.py)
- [teacher_audit.py](../unidream/experiments/teacher_audit.py)

## baseline 監査結果

出力:

- [medium_v2 audit summary](../checkpoints/teacher_audit/medium_v2/medium_v2_teacher_audit_summary.csv)
- [medium_ext_sources audit summary](../checkpoints/teacher_audit/medium_ext_sources/medium_ext_sources_teacher_audit_summary.csv)

確認結果:

- `medium_v2` と `medium_ext_sources` はどちらもほぼ同じ分布
- overall はほぼ `long 0% / short 50% / flat 50%`
- bearish regime でも大きくは変わらない
- `min_hold = 16 / 32 / 64` を振っても、主に変わるのは `avg_hold` だけ

この時点での結論:

- `teacher / oracle が主因か`: **true**
- `min_hold` 単独では解決しない

## L1 候補の初回比較

fold 4 の teacher 監査で baseline / `signal_aim` / `signal_aim + downside` を比較した。

### baseline

- all: `short 50.1% / flat 49.9%`
- bearish: `short 50.4% / flat 49.6%`
- avg_hold: `23.2 bars`

### signal_aim

- all: `short 39.6% / flat 60.4%`
- bearish: `short 40.3% / flat 59.7%`
- avg_hold: `2.6 bars`

### signal_aim + downside

- all: `short 66.6% / flat 33.4%`
- bearish: `short 66.5% / flat 33.5%`
- avg_hold: `1.7 bars`

初回判定:

- `signal_aim`: 採用候補
- `signal_aim + downside`: short に寄りすぎるので棄却

## 追加候補の再監査

issue1 の teacher 候補をもう一段絞るため、`feature_stress` と `feature_dual` を fold 4 の teacher 監査に追加した。

出力:

- [teacher_audit_l1](../checkpoints/teacher_audit_l1)

### fold 4 / train(all) 比較

| config | short | flat | turnover |
| --- | ---: | ---: | ---: |
| `signal_aim` | 39.6% | 60.4% | 2166.0 |
| `feature_stress` | 16.2% | 83.8% | 2419.0 |
| `feature_dual` | 8.4% | 91.6% | 1846.1 |

### fold 4 / bearish(train 平均) 比較

| config | short | flat |
| --- | ---: | ---: |
| `signal_aim` | 40.3% | 59.7% |
| `feature_stress` | 23.1% | 76.9% |
| `feature_dual` | 13.3% | 86.7% |

### 追加候補の判定

- `signal_aim`
  - aim portfolio 型として最も自然
  - bearish で flat を増やすが、teacher が死にすぎない
  - **採用**
- `feature_stress`
  - stress / downside 寄り teacher として補完価値がある
  - `signal_aim` より保守的で、issue2 の比較対象として価値がある
  - **採用**
- `feature_dual`
  - bearish では flat 化するが、全体でも flat に寄りすぎる
  - issue2 に持ち込む teacher としては保守的すぎる
  - **棄却**

## issue1 の確定結論

issue1 を閉じる時点の結論は次の通り。

- `teacher / oracle が主因に弱いか`: **true**
- 主因の形:
  - regime 横断で `short / flat` 固定 teacher に近い
  - bearish regime に対する flat 化が弱い
  - `min_hold` は hold 長を変えるだけで、teacher の質はほとんど変えない
- 次段に持ち込む teacher 候補:
  - `signal_aim`
  - `feature_stress`
- 次段に持ち込まない teacher 候補:
  - `signal_aim + downside`
  - `feature_dual`

## 次の課題

次は `BC prior の再現性診断` に移る。
見るもの:

- teacher と BC の `long / flat / short` 分布差
- regime 別 mismatch
- val での再現と test での崩れの関係

issue2 は、`signal_aim` と `feature_stress` を teacher 候補として比較する。
