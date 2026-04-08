# Optimization Loop: Issue 1 Teacher Audit

## 目的

最初の課題として、`teacher / oracle が本当に弱いか` を full 学習ではなく局所診断で確認した。

確認したかった点は次の 3 つ。

- regime 別に `long / flat / short` がどう分布しているか
- `min_hold` が teacher 分布そのものに効いているか
- teacher 改善候補が regime 0/1 の flat 化を増やせるか

## 監査対象

- baseline: [medium_v2.yaml](../configs/medium_v2.yaml)
- external-source 版: [medium_ext_sources.yaml](../configs/medium_ext_sources.yaml)
- L1 比較用:
  - [medium_l1_base.yaml](../configs/medium_l1_base.yaml)
  - [medium_l1_signal_aim.yaml](../configs/medium_l1_signal_aim.yaml)
  - [medium_l1_signal_downside.yaml](../configs/medium_l1_signal_downside.yaml)

監査スクリプト:

- [audit_teacher_regimes.py](../audit_teacher_regimes.py)
- [teacher_audit.py](../unidream/experiments/teacher_audit.py)

出力:

- [medium_v2 audit summary](../checkpoints/teacher_audit/medium_v2/medium_v2_teacher_audit_summary.csv)
- [medium_ext_sources audit summary](../checkpoints/teacher_audit/medium_ext_sources/medium_ext_sources_teacher_audit_summary.csv)
- [L1 teacher audit summaries](../checkpoints/teacher_audit_l1)

## 結果

### 1. baseline teacher は regime にほぼ反応していない

`medium_v2` と `medium_ext_sources` の両方で、baseline teacher は次の形に張り付いた。

- overall: `long 0% / short 50% / flat 50%`
- bearish regimes でもほぼ同じ
- `min_hold = 16, 32, 64` を振っても、変わるのは `avg_hold` だけ

要するに、現行の baseline teacher は

- regime 0/1 で flat を増やす
- regime 2 で risk を戻す

のような構造を持てていない。

### 2. teacher 改善候補の L1 比較

fold 4 の teacher 監査だけを比較した。

#### baseline

- all: `short 50.1% / flat 49.9%`
- bearish: `short 50.4% / flat 49.6%`
- avg_hold: `23.2 bars`

#### signal_aim

- all: `short 39.6% / flat 60.4%`
- bearish: `short 40.3% / flat 59.7%`
- avg_hold: `2.6 bars`

#### signal_aim + downside

- all: `short 66.6% / flat 33.4%`
- bearish: `short 66.5% / flat 33.5%`
- avg_hold: `1.7 bars`

### 3. 判定

- baseline teacher は主因の 1 つとみなしてよい
- `min_hold` 単独では改善しない
- `signal_aim` は bearish regime で flat を増やせており、次段に持ち込む価値がある
- `signal_aim + downside` は short に寄りすぎで、今の段階ではやりすぎ

## 結論

Issue 1 の結論は次の通り。

- `teacher / oracle が本当に弱いか`: `true`
- 主因の形:
  - regime 非依存の `short / flat` 二値 teacher に近い
  - `min_hold` は分布ではなく hold 長だけを変えている
- 次に持ち込む teacher 候補:
  - `signal_aim`
- 今は持ち込まない候補:
  - `signal_aim + downside`

## 次の課題

次は `BC prior の再現性診断` に進む。

見るもの:

- teacher と BC の `long / flat / short` 分布差
- regime 別 mismatch
- val での再現と test での崩れ方

入口としては、`signal_aim teacher` を teacher 候補にした BC 比較から始める。
