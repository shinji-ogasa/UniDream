# AC Plan15-B: MDD Fire Label Probe

Run completed: 2026-05-01 JST

## Purpose

Plan15 の続きとして、`harm_margin` / `dd_rel` より直接的に「fireが最大DDに巻き込まれるか」「fire後にDDを悪化させるか」をラベル化した。

今回も本流変更ではない。

```text
No change:
  configs/trading.yaml
  fire guard
  WM head v2
  AC unlock
```

## Implementation

追加:

```text
unidream/cli/fire_mdd_label_probe.py
```

出力:

```text
documents/20260501_plan15b_mdd_fire_Hbest_folds456.md
documents/plan15b_mdd_fire_Hbest_folds456.json
documents/20260501_plan15b_mdd_fire_E_fold5.md
documents/plan15b_mdd_fire_E_fold5.json
```

## Labels

同じactorから adapter on/off を作る。

```text
pos_on  = actor current policy
pos_off = actor with benchmark overweight adapter disabled
fire    = abs(pos_on - pos_off) > 1e-6
```

追加したMDD系ラベル:

```text
future_mdd_overlap_h
  fire run が horizon内の local maxDD peak-trough 区間に重なるか

post_fire_dd_contribution_h
  max_t(maxDD_on_t - maxDD_off_t, 0)

mdd_contribution_h
  local_maxDD_on - local_maxDD_off

pre_dd_state_h
  fire後、horizon前半にDD peakが来て、その後troughへ落ちるか

global_mdd_overlap
  test period全体の最大DD区間にfire barが入っているか
```

評価:

```text
chronological 60% train / 40% eval
ridge probe
binary: AUC / PR-AUC / top10 positive rate
ranking: low postDD, low future-MDD-overlap
combined:
  A = fire_advantage only
  B = fire_advantage - postDD
  C = fire_advantage - future_overlap - preDD
  D = fire_advantage - postDD - future_overlap - preDD - global_overlap
```

## Hbest Fold4/5/6

Command:

```text
uv run python -m unidream.cli.fire_mdd_label_probe --config configs/trading_wm_control_headonly.yaml --start 2018-01-01 --end 2024-01-01 --folds 4,5,6 --seed 11 --device cuda --run Hbest=checkpoints/acplan13_wm_control_headonly_s011@ac_best.pt:ac --output-json documents/plan15b_mdd_fire_Hbest_folds456.json --output-md documents/20260501_plan15b_mdd_fire_Hbest_folds456.md
```

Policy:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | fire | fire_pnl | global MDD | fire in MDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.05 | +0.017 | +0.35 | 5.69 | 11.4%/994 | -0.0110 | -53.55pt | 7.4% |
| 5 | -28.10 | -0.064 | +0.75 | 2.62 | 12.1%/1065 | -0.0407 | -24.35pt | 4.0% |
| 6 | -0.60 | -0.021 | +0.61 | 4.72 | 28.4%/2506 | -0.1802 | -42.55pt | 19.9% |

### Predictability

Primary h32:

| fold | future MDD AUC | future MDD top10 | pre-DD AUC | global MDD AUC | postDD q AUC |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.572 | 0.800 | 0.507 | 0.695 | 0.403 |
| 5 | 0.538 | 0.476 | 0.284 | 0.591 | 0.426 |
| 6 | 0.662 | 0.941 | 0.554 | 0.378 | 0.768 |

Interpretation:

```text
future_mdd_overlap_h32:
  fold4/6は読める
  fold5が閾値未満

global_mdd_overlap:
  fold4/5は多少読める
  fold6は読めない

post_fire_dd_contribution:
  fold6は読める
  fold4/5は弱い

pre_dd_state:
  h32では安定しない
```

### Advantage vs MDD Risk Ranking

Primary h32:

| fold | adv top10 | adv postDD | adv futureMDD | adv globalMDD | low postDD adv | low postDD | low overlap adv | low overlap futureMDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.00152 | +0.00056 | 0.550 | 0.150 | +0.00036 | +0.00045 | +0.00033 | 0.500 |
| 5 | +0.00088 | +0.00052 | 0.238 | 0.357 | +0.00061 | +0.00032 | +0.00053 | 0.238 |
| 6 | +0.00006 | +0.00013 | 0.584 | 0.673 | +0.00001 | +0.00008 | +0.00000 | 0.446 |

Interpretation:

```text
low_postDD:
  postDDは下がる
  ただしadvantageも大きく削る
  fold6はほぼゼロ優位

low_overlap:
  fold6のfutureMDD overlapは下がる
  ただしadvantageもほぼゼロ
```

### Combined Score

Primary h32:

| fold | score | top10 adv | postDD | futureMDD | preDD | globalMDD | top20 adv |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | A_adv_only | +0.00152 | +0.00056 | 0.550 | 0.050 | 0.150 | +0.00090 |
| 4 | D_adv_minus_all_mdd | +0.00099 | +0.00041 | 0.475 | 0.050 | 0.050 | +0.00088 |
| 5 | A_adv_only | +0.00088 | +0.00052 | 0.238 | 0.000 | 0.357 | +0.00072 |
| 5 | D_adv_minus_all_mdd | +0.00061 | +0.00034 | 0.071 | 0.024 | 0.476 | +0.00055 |
| 6 | A_adv_only | +0.00006 | +0.00013 | 0.584 | 0.050 | 0.673 | +0.00006 |
| 6 | D_adv_minus_all_mdd | +0.00004 | +0.00009 | 0.515 | 0.069 | 0.653 | +0.00003 |

Interpretation:

```text
D_adv_minus_all_mdd:
  postDD / futureMDD は少し下がる
  ただしglobalMDDがfold5で悪化
  fold6のglobalMDDも高すぎる
  alpha advantageも削る
```

Readiness:

| criterion | result |
|---|---:|
| adv_top10_positive_all | true |
| low_postdd_keeps_positive_adv_all | true |
| combined_D_improves_mdd_risk_all | false |
| future_mdd_overlap_auc_ge_0_55_all | false |

## E Fold5 Reference

Command:

```text
uv run python -m unidream.cli.fire_mdd_label_probe --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 5 --seed 11 --device cuda --run E=checkpoints/acplan10_fire_selector_s011_on_wmbc_s011:ac --output-json documents/plan15b_mdd_fire_E_fold5.json --output-md documents/20260501_plan15b_mdd_fire_E_fold5.md
```

Policy:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | fire | fire_pnl | global MDD | fire in MDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | +62.09 | +0.050 | +0.04 | 1.18 | 3.3%/288 | +0.1059 | -23.63pt | 1.5% |

Primary h32:

| future MDD AUC | future MDD top10 | pre-DD AUC | global MDD AUC | postDD q AUC |
|---:|---:|---:|---:|---:|
| 0.642 | 0.917 | 0.725 | nan | 0.833 |

Combined h32:

| score | top10 adv | postDD | futureMDD | preDD | globalMDD | top20 adv |
|---|---:|---:|---:|---:|---:|---:|
| A_adv_only | +0.00323 | +0.00129 | 0.583 | 0.000 | 0.000 | +0.00270 |
| B_adv_minus_postdd | +0.00004 | +0.00020 | 0.833 | 0.000 | 0.000 | +0.00060 |
| C_adv_minus_overlap_predd | +0.00265 | +0.00095 | 0.417 | 0.000 | 0.000 | +0.00254 |
| D_adv_minus_all_mdd | +0.00326 | +0.00124 | 0.583 | 0.000 | 0.000 | +0.00240 |

Interpretation:

```text
E fold5ではMDD系ラベルがかなり読める。
ただし低postDDだけを狙うとadvantageがほぼ消える。
CはfutureMDD overlapを少し下げつつadvantageを残す。
```

## Decision

採用:

```text
fire_mdd_label_probe.py
```

診断ツールとして採用。

まだ採用しない:

```text
inference-only fire guard
WM control head v2
AC unlock
MDD-window guard
postDD guard
combined D guard
```

理由:

```text
Hbest fold4/5/6でMDD-window labelは安定していない。
future MDD overlapはfold5でAUC 0.538。
postDD contributionはfold4/5で弱い。
combined DはMDDリスクを全foldで改善できていない。
```

## Technical Conclusion

Plan15-B の結論:

```text
「fireしたら儲かるか」は依然として fire_advantage_h32 が一番まとも。
「そのfireが最大DD区間に入るか」は一部読めるがmulti-foldで不安定。
「そのfireがDDを悪化させるか」はpostDD label単体だとalphaを削りすぎる。
```

次にやるなら、Plan16 guardへ直行ではなく、もう一段ラベルを絞るべき。

候補:

```text
Plan15-C:
  C_adv_minus_overlap_predd を中心に、futureMDD overlapだけを軽く抑える
  postDD penaltyは強くしない
  fold6のglobalMDD overlapを直接下げる専用labelを追加
  h32だけでなく h48/h64 を見る
```

現時点でAC拡張はまだ早い。
