# Plan15 Fire-Control Label V2 Probe

## Setup

- config: `configs/trading.yaml`
- folds: `5`
- horizons: `16,32`
- primary horizon: `32`
- scope: label/ranking/combined-score probe only; no fire guard, WM head v2, or AC unlock.

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 98.1% | 3.3%/288 | +0.1059 |

## Fire Advantage Ranking

| run | fold | h | samples | eval | corr | top10 adv | top20 adv | bottom10 adv | spread | top10 harm | top10 dd_rel | top10 mdd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | 288 | 116 | -0.078 | +0.00043 | +0.00032 | +0.00037 | +0.00007 | +0.00017 | 0.180 | 0.000 |
| E | 5 | 32 | 288 | 116 | 0.416 | +0.00323 | +0.00270 | +0.00030 | +0.00293 | +0.00117 | 0.122 | 0.000 |

## Recovery / Relative DD Labels

| run | fold | h | recovery AUC | rel recovery AUC | post-trough AUC | underwater AUC | dd k0.5 AUC | dd q80 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | 0.437 | 0.557 | 0.517 | 0.467 | 0.441 | 0.388 |
| E | 5 | 32 | 0.323 | 0.397 | 0.211 | 0.550 | 0.487 | 0.571 |

## Low-Harm Ranking

| run | fold | h | low10 adv | low10 harm | low10 dd_rel | low10 mdd | low10 harm<=0 | high10 harm | high10 dd_rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | +0.00005 | +0.00010 | 0.174 | 0.000 | 0.333 | +0.00126 | 0.143 |
| E | 5 | 32 | -0.00002 | +0.00012 | 0.327 | 0.000 | 0.167 | +0.00173 | 0.085 |

## Combined Score Top-Decile

| run | fold | h | score | top10 adv | top10 harm | top10 dd_rel | top10 recovery | top10 mdd | top20 adv | top20 harm |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | A_adv_only | +0.00043 | +0.00017 | 0.180 | +0.01911 | 0.000 | +0.00032 | +0.00016 |
| E | 5 | 16 | B_adv_minus_harm | -0.00001 | +0.00006 | 0.258 | +0.01275 | 0.000 | +0.00007 | +0.00010 |
| E | 5 | 16 | C_adv_minus_dd_plus_recovery | +0.00071 | +0.00018 | 0.267 | +0.01424 | 0.000 | +0.00043 | +0.00037 |
| E | 5 | 16 | D_all | +0.00037 | +0.00018 | 0.272 | +0.00971 | 0.000 | +0.00024 | +0.00016 |
| E | 5 | 32 | A_adv_only | +0.00323 | +0.00117 | 0.122 | +0.02790 | 0.000 | +0.00270 | +0.00105 |
| E | 5 | 32 | B_adv_minus_harm | +0.00005 | +0.00018 | 0.307 | +0.00281 | 0.000 | +0.00062 | +0.00036 |
| E | 5 | 32 | C_adv_minus_dd_plus_recovery | +0.00295 | +0.00087 | 0.149 | +0.02897 | 0.000 | +0.00245 | +0.00095 |
| E | 5 | 32 | D_all | +0.00149 | +0.00058 | 0.206 | +0.01407 | 0.000 | +0.00165 | +0.00069 |

## Readiness

| criterion | pass |
|---|---:|
| primary_adv_top10_all_positive | True |
| primary_adv_top20_all_positive | True |
| primary_adv_spread_all_positive | True |
| combined_D_top10_positive_and_nonharm_all | False |

## Interpretation

- Plan15 does not change the production training flow.
- `fire_advantage_h32` is the primary signal. It must be positive in top10/top20 and have positive top-bottom spread across folds.
- Low-harm ranking is useful only if low predicted harm has positive fire advantage and non-positive realized harm margin.
- Combined scores are candidates for Plan16 inference-only guard, not adoption by themselves.
