# Plan15-B MDD Fire Label Probe

## Setup

- config: `configs/trading.yaml`
- folds: `5`
- horizons: `16,32`
- primary horizon: `32`
- scope: MDD-window label quality only; no guard, WM head, or AC unlock.

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl | global MDD | fire in MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 98.1% | 3.3%/288 | +0.1059 | -23.63pt | 1.5% |

## MDD Label Predictability

| run | fold | h | samples | eval | future MDD AUC | future MDD top10 | pre-DD AUC | global MDD AUC | postDD q AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | 288 | 116 | 0.587 | 0.583 | 0.444 | nan | 0.858 |
| E | 5 | 32 | 288 | 116 | 0.642 | 0.917 | 0.725 | nan | 0.833 |

## Advantage vs MDD Risk Ranking

| run | fold | h | adv top10 | adv postDD | adv futureMDD | adv globalMDD | low postDD adv | low postDD | low overlap adv | low overlap futureMDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | +0.00043 | +0.00023 | 0.333 | 0.000 | +0.00001 | +0.00011 | +0.00106 | 0.500 |
| E | 5 | 32 | +0.00323 | +0.00129 | 0.583 | 0.000 | -0.00001 | +0.00013 | +0.00068 | 0.667 |

## Combined Score Top-Decile

| run | fold | h | score | top10 adv | postDD | mdd contrib | futureMDD | preDD | globalMDD | top20 adv | top20 postDD |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | A_adv_only | +0.00043 | +0.00023 | +0.00017 | 0.333 | 0.167 | 0.000 | +0.00032 | +0.00019 |
| E | 5 | 16 | B_adv_minus_postdd | -0.00000 | +0.00007 | +0.00006 | 0.333 | 0.167 | 0.000 | +0.00007 | +0.00011 |
| E | 5 | 16 | C_adv_minus_overlap_predd | +0.00035 | +0.00030 | +0.00024 | 0.500 | 0.000 | 0.000 | +0.00038 | +0.00028 |
| E | 5 | 16 | D_adv_minus_all_mdd | +0.00048 | +0.00030 | +0.00019 | 0.250 | 0.083 | 0.000 | +0.00035 | +0.00027 |
| E | 5 | 32 | A_adv_only | +0.00323 | +0.00129 | +0.00117 | 0.583 | 0.000 | 0.000 | +0.00270 | +0.00113 |
| E | 5 | 32 | B_adv_minus_postdd | +0.00004 | +0.00020 | +0.00019 | 0.833 | 0.000 | 0.000 | +0.00060 | +0.00044 |
| E | 5 | 32 | C_adv_minus_overlap_predd | +0.00265 | +0.00095 | +0.00086 | 0.417 | 0.000 | 0.000 | +0.00254 | +0.00101 |
| E | 5 | 32 | D_adv_minus_all_mdd | +0.00326 | +0.00124 | +0.00115 | 0.583 | 0.000 | 0.000 | +0.00240 | +0.00100 |

## Readiness

| criterion | pass |
|---|---:|
| adv_top10_positive_all | True |
| low_postdd_keeps_positive_adv_all | False |
| combined_D_improves_mdd_risk_all | True |
| future_mdd_overlap_auc_ge_0_55_all | True |

## Interpretation

- If MDD-window labels are not separable, Plan16 guard should not use them.
- A useful risk score must lower post-fire DD contribution or MDD overlap without killing fire_advantage.
- This probe is diagnostic only and does not alter `configs/trading.yaml`.
