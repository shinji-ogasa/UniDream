# AC Plan15: Fire-Control Label V2 Probe

Run completed: 2026-05-01 JST

## Purpose

`acplan_15.md` に従い、Plan14 の直接fire-control labelを作り直した。

今回の目的は、guardやWM head v2やAC unlockではない。

```text
Plan15 scope:
  fire_advantage_h32 を主score化
  trough_exit を廃止
  recovery_slope / post_trough_momentum / underwater_recovery を追加
  drawdown_worsening を rolling-vol / fold-quantile 相対化
  fire_harm を分類ではなく low-harm ranking として評価
  combined score の候補品質を見る
```

`configs/trading.yaml` は変更していない。

## Implementation

追加:

```text
unidream/cli/fire_control_label_v2_probe.py
```

出力:

```text
documents/20260501_plan15_fire_control_v2_Hbest_folds456.md
documents/plan15_fire_control_v2_Hbest_folds456.json
documents/20260501_plan15_fire_control_v2_E_fold5.md
documents/plan15_fire_control_v2_E_fold5.json
```

## Calculation

同じactorから adapter on/off を作る。

```text
pos_on  = actor current policy
pos_off = actor with benchmark overweight adapter disabled
fire    = abs(pos_on - pos_off) > 1e-6
```

主ラベル:

```text
fire_advantage_h
  = cost込み sum_h pnl(pos_on) - sum_h pnl(pos_off)
```

追加ラベル:

```text
dd_rel_h
  = future_drawdown_deepen / (rolling_vol_64 * sqrt(h))

recovery_slope_h
  = slope(cumsum(pnl_on)) / (rolling_vol_64 * sqrt(h))

relative_recovery_slope_h
  = slope(cumsum(pnl_on - pnl_off)) / (rolling_vol_64 * sqrt(h))

post_trough_momentum_h
  = horizon内のprice path trough後のrebound

underwater_recovery_h
  = future_end_drawdown - current_drawdown

harm_margin_h
  = local_future_maxdd(pos_on) - local_future_maxdd(pos_off)
```

Probe:

```text
input:
  WM latent z/h
  predictive state
  regime
  position / adapter delta
  current drawdown / underwater duration
  trailing return / vol
  equity / peak state

fit:
  chronological 60% train / 40% eval
  ridge probe

metrics:
  top10/top20 realized fire_advantage
  top-bottom spread
  recovery / DD AUC
  low-harm ranking
  combined score top10/top20
```

## Hbest Fold4/5/6 Result

Command:

```text
uv run python -m unidream.cli.fire_control_label_v2_probe --config configs/trading_wm_control_headonly.yaml --start 2018-01-01 --end 2024-01-01 --folds 4,5,6 --seed 11 --device cuda --run Hbest=checkpoints/acplan13_wm_control_headonly_s011@ac_best.pt:ac --output-json documents/plan15_fire_control_v2_Hbest_folds456.json --output-md documents/20260501_plan15_fire_control_v2_Hbest_folds456.md
```

Policy:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 97.3% | 11.4%/994 | -0.0110 |
| 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 97.0% | 12.1%/1065 | -0.0407 |
| 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 99.1% | 28.4%/2506 | -0.1802 |

### Fire Advantage h32

| fold | corr | top10 adv | top20 adv | bottom10 adv | spread | top10 harm | top10 dd_rel | top10 mdd |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.307 | +0.00152 | +0.00090 | +0.00011 | +0.00141 | +0.00050 | 0.094 | 0.150 |
| 5 | 0.230 | +0.00088 | +0.00072 | +0.00021 | +0.00067 | +0.00052 | 0.302 | 0.357 |
| 6 | 0.312 | +0.00006 | +0.00006 | -0.00030 | +0.00035 | +0.00011 | 0.273 | 0.673 |

判断:

```text
fire_advantage_h32:
  top10 positive: pass
  top20 positive: pass
  top-bottom spread positive: pass

ただし fold6 はadv絶対値が小さく、MDD区間率が高い。
```

### Recovery / Relative DD

| fold | h32 recovery AUC | h32 rel recovery AUC | h32 post-trough AUC | h32 underwater AUC | h32 dd k0.5 AUC | h32 dd q80 AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.599 | 0.615 | 0.447 | 0.597 | 0.534 | 0.336 |
| 5 | 0.526 | 0.441 | 0.628 | 0.632 | 0.367 | 0.408 |
| 6 | 0.595 | 0.610 | 0.575 | 0.616 | 0.756 | 0.789 |

判断:

```text
recovery_slope / underwater_recovery:
  一部は0.6前後まで出るが、fold間で安定しない。

relative DD:
  fold6では強いが、fold4/5で弱い。

post_trough_momentum:
  fold5/6は悪くないが、fold4が弱い。
```

### Low-Harm Ranking h32

| fold | low10 adv | low10 harm | low10 dd_rel | low10 mdd | low10 harm<=0 | high10 harm | high10 dd_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.00035 | +0.00082 | 0.342 | 0.225 | 0.050 | +0.00030 | 0.236 |
| 5 | +0.00056 | +0.00034 | 0.447 | 0.071 | 0.024 | +0.00038 | 0.146 |
| 6 | +0.00000 | +0.00007 | 0.317 | 0.733 | 0.218 | +0.00064 | 0.898 |

判断:

```text
low-harm ranking:
  fail

理由:
  low-harm top10 が realized harm<=0 にならない。
  fold4では low-harm top10 の harm が high-harm top10 より悪い。
  fold6では low-harm top10 のMDD区間率が73.3%で高すぎる。
```

### Combined Score h32

Score:

```text
A = fire_advantage only
B = fire_advantage - harm
C = fire_advantage - DD + recovery
D = all combined
```

| fold | score | top10 adv | top10 harm | top10 dd_rel | top10 mdd | top20 adv | top20 harm |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | A_adv_only | +0.00152 | +0.00050 | 0.094 | 0.150 | +0.00090 | +0.00042 |
| 4 | D_all | +0.00133 | +0.00040 | 0.118 | 0.125 | +0.00078 | +0.00066 |
| 5 | A_adv_only | +0.00088 | +0.00052 | 0.302 | 0.357 | +0.00072 | +0.00052 |
| 5 | D_all | +0.00092 | +0.00053 | 0.392 | 0.238 | +0.00075 | +0.00050 |
| 6 | A_adv_only | +0.00006 | +0.00011 | 0.273 | 0.673 | +0.00006 | +0.00012 |
| 6 | D_all | +0.00004 | +0.00010 | 0.263 | 0.594 | +0.00003 | +0.00013 |

判断:

```text
combined score:
  not ready for Plan16 guard

理由:
  A_adv_only が一番素直。
  harm/DD/recoveryを足しても realized harm を非悪化にできない。
  fold6のMDD区間率が高く、guardとしては危険。
```

## E Fold5 Reference

Command:

```text
uv run python -m unidream.cli.fire_control_label_v2_probe --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 5 --seed 11 --device cuda --run E=checkpoints/acplan10_fire_selector_s011_on_wmbc_s011:ac --output-json documents/plan15_fire_control_v2_E_fold5.json --output-md documents/20260501_plan15_fire_control_v2_E_fold5.md
```

Policy:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 98.1% | 3.3%/288 | +0.1059 |

Fire advantage h32:

| corr | top10 adv | top20 adv | bottom10 adv | spread | top10 harm | top10 dd_rel | top10 mdd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.416 | +0.00323 | +0.00270 | +0.00030 | +0.00293 | +0.00117 | 0.122 | 0.000 |

Combined h32:

| score | top10 adv | top10 harm | top10 dd_rel | top10 mdd | top20 adv | top20 harm |
|---|---:|---:|---:|---:|---:|---:|
| A_adv_only | +0.00323 | +0.00117 | 0.122 | 0.000 | +0.00270 | +0.00105 |
| B_adv_minus_harm | +0.00005 | +0.00018 | 0.307 | 0.000 | +0.00062 | +0.00036 |
| C_adv_minus_dd_plus_recovery | +0.00295 | +0.00087 | 0.149 | 0.000 | +0.00245 | +0.00095 |
| D_all | +0.00149 | +0.00058 | 0.206 | 0.000 | +0.00165 | +0.00069 |

Reference判断:

```text
E fold5でも A_adv_only / C が強い。
ただし low-harm ranking は h32 top10 adv が -0.00002 で失敗。
harmを強く避けるとalphaを削る。
```

## Readiness

| criterion | Hbest fold4/5/6 |
|---|---:|
| primary_adv_top10_all_positive | true |
| primary_adv_top20_all_positive | true |
| primary_adv_spread_all_positive | true |
| combined_D_top10_positive_and_nonharm_all | false |

## Decision

採用:

```text
fire_control_label_v2_probe.py
```

研究診断ツールとして採用。

まだ採用しない:

```text
inference-only fire guard
WM control head v2
AC unlock
low-harm guard
combined D guard
```

理由:

```text
fire_advantage_h32 は使える。
ただし harm/DD/recovery はまだ guard にするほど安定していない。
特に low-harm ranking が実際の harm<=0 を作れていない。
```

## Technical Conclusion

Plan15の結論はこれ。

```text
「fireしたら儲かるか」は h32 で読める。
「fireしても火事にならないか」はまだ読めていない。
```

なので、次に Plan16 guard へ進むなら、現状の combined score ではなく、

```text
fire_advantage_h32 score を主軸
MDD区間 / fold6 のDD寄与を直接避ける別ラベル
harm_margin分類ではなく maxDD-window fire exposure label
```

が必要。

次の実験候補:

```text
Plan15-B:
  maxdd_window_fire_label を作る
  future MDD interval overlap / pre-DD state / post-fire drawdown contribution を直接予測

Plan16:
  その後に inference-only guard を試す
```

現時点で ACを広げるのは早い。
