# AC Plan16 / Plan15-C Fire Type Report

対象計画: `documents/acplan_16.md`

実行範囲:

```text
Plan15-C only:
  fire type / regime clustering probe
  no inference guard
  no WM head v2
  no AC unlock
  no configs/trading.yaml change
```

## 実装

追加:

```text
unidream/cli/fire_type_cluster_probe.py
```

出力:

```text
documents/20260501_plan15c_fire_type_Hbest_folds456.md
documents/plan15c_fire_type_Hbest_folds456.json
```

実行コマンド:

```powershell
uv run python -u -m unidream.cli.fire_type_cluster_probe `
  --config configs/trading_wm_control_headonly.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 4,5,6 `
  --seed 11 `
  --device cuda `
  --run Hbest=checkpoints/acplan13_wm_control_headonly_s011@ac_best.pt:ac `
  --output-json documents/plan15c_fire_type_Hbest_folds456.json `
  --output-md documents/20260501_plan15c_fire_type_Hbest_folds456.md
```

fire barごとに出したもの:

```text
current drawdown depth
underwater duration
trailing return/slope 16/32
trailing vol 64
equity slope
benchmark-relative equity slope
position / adapter delta
fire_advantage_h32
post_fire_dd_contribution
future_mdd_overlap
global MDD overlap / MDD phase
regime bucket
```

## Policy Summary

対象はHbest fold4/5/6。

| fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | fire_pnl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 11.4% / 994 | -0.0110 |
| 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 12.1% / 1065 | -0.0407 |
| 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 28.4% / 2506 | -0.1802 |

Hbest自体は複数foldで不採用。
今回の目的はpolicy採用ではなく、fire typeが分離できるかの診断。

## Main Findings

### 1. pre_dd_danger_fire は一貫して悪い

| fold | count | adv | futureMDD | globalMDD | fire_pnl |
|---:|---:|---:|---:|---:|---:|
| 4 | 550 | -0.00034 | 0.884 | 0.540 | -0.1206 |
| 5 | 625 | -0.00031 | 0.934 | 0.274 | -0.1588 |
| 6 | 1446 | -0.00015 | 0.948 | 0.603 | -0.5534 |

これはかなり重要。

```text
fireの大半を占める
3foldすべてで fire_advantage が負
future MDD overlap が非常に高い
fire_pnl も全foldで負
```

Plan16をやるなら、まず「良いfireだけ許可」ではなく、

```text
pre_dd_danger_fire を検出して抑制する
```

が第一候補。

### 2. mdd_inside_profitable_fire はalpha源だがDDリスクも高い

| fold | count | adv | postDD | futureMDD | globalMDD | fire_pnl |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 207 | +0.00046 | +0.00062 | 0.754 | 1.000 | +0.1647 |
| 5 | 93 | +0.00039 | +0.00085 | 0.774 | 1.000 | +0.1358 |
| 6 | 484 | +0.00016 | +0.00024 | 0.888 | 1.000 | +0.5143 |

これは単純に消せない。

```text
fire_advantage と fire_pnl は正
しかし futureMDD/globalMDD overlap が高い
```

つまり、DD guardでこれを全消しするとalphaを削りすぎる可能性が高い。
Plan16 Guard D のように、消すより `delta 0.25〜0.5倍` の縮小候補。

### 3. recovery/trend/profitable_low_dd はfutureMDDが低いがglobalMDDが安定しない

3foldすべてで以下は成立:

```text
fire_advantage > 0
adv positive rate >= 60%
post_fire_dd_contribution low
future_mdd_overlap low
```

ただし `globalMDD overlap` が低いとは言えなかった。

| fire_type | strict Plan16 candidate |
|---|---:|
| profitable_low_dd_fire | false |
| recovery_fire | false |
| trend_continuation_fire | false |

理由:

```text
profitable_low_dd_fire: fold4/fold6でglobalMDD overlapが高い
recovery_fire: fold6でglobalMDD overlapが高い
trend_continuation_fire: fold4でglobalMDD overlapが高い
```

acplan_16の合格条件は「MDD overlapが低い」なので、ここはまだ通せない。

## Readiness

厳格条件:

```text
enough samples in every fold
fire_advantage > 0 in every fold
adv positive rate >= 60% in every fold
post_fire_dd_contribution low in every fold
future_mdd_overlap low in every fold
global_mdd_overlap low in every fold
```

結果:

```text
Plan16-ready fire type: none
```

したがって、まだ inference-only fire guard は実装しない。

## 判断

Plan15-Cの結論:

```text
fire type分解は有効。
ただし、AlphaとDDが両立するtypeをそのままPlan16 guardに使える段階ではない。
```

一番強い発見:

```text
pre_dd_danger_fire は3foldで明確に悪い。
```

次にやるべきこと:

```text
Plan15-D:
  allow_fire score ではなく、
  danger_fire_score を先に作る。

目的:
  pre_dd_danger_fire を減らす
  mdd_inside_profitable_fire は全消ししない
  recovery/trend/profitable_low_dd はMDD phaseでさらに分割する
```

次のprobe案:

```text
danger_fire_score =
  future_mdd_overlap_score
  + pre_dd_state_score
  + post_fire_dd_contribution_score
  + global_mdd_phase_score
  - fire_advantage_score
```

Plan16に進む条件:

```text
danger score top decileが pre_dd_danger_fire を3foldで拾う
danger suppressionで fire_advantage の損失が限定的
MaxDD寄与が下がる
```

現時点では本流変更なし。
