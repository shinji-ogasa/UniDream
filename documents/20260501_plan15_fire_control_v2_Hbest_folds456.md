# Plan15 Fire-Control Label V2 Probe

## Setup

- config: `configs/trading_wm_control_headonly.yaml`
- folds: `4,5,6`
- horizons: `16,32`
- primary horizon: `32`
- scope: label/ranking/combined-score probe only; no fire guard, WM head v2, or AC unlock.

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 97.3% | 11.4%/994 | -0.0110 |
| Hbest | 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 97.0% | 12.1%/1065 | -0.0407 |
| Hbest | 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 99.1% | 28.4%/2506 | -0.1802 |

## Fire Advantage Ranking

| run | fold | h | samples | eval | corr | top10 adv | top20 adv | bottom10 adv | spread | top10 harm | top10 dd_rel | top10 mdd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | 994 | 398 | 0.323 | +0.00118 | +0.00073 | +0.00004 | +0.00114 | +0.00042 | 0.202 | 0.150 |
| Hbest | 4 | 32 | 984 | 394 | 0.307 | +0.00152 | +0.00090 | +0.00011 | +0.00141 | +0.00050 | 0.094 | 0.150 |
| Hbest | 5 | 16 | 1050 | 420 | 0.151 | +0.00060 | +0.00038 | -0.00002 | +0.00061 | +0.00052 | 0.486 | 0.310 |
| Hbest | 5 | 32 | 1047 | 419 | 0.230 | +0.00088 | +0.00072 | +0.00021 | +0.00067 | +0.00052 | 0.302 | 0.357 |
| Hbest | 6 | 16 | 2506 | 1003 | 0.241 | +0.00004 | +0.00003 | -0.00019 | +0.00023 | +0.00014 | 0.350 | 0.693 |
| Hbest | 6 | 32 | 2506 | 1003 | 0.312 | +0.00006 | +0.00006 | -0.00030 | +0.00035 | +0.00011 | 0.273 | 0.673 |

## Recovery / Relative DD Labels

| run | fold | h | recovery AUC | rel recovery AUC | post-trough AUC | underwater AUC | dd k0.5 AUC | dd q80 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | 0.549 | 0.481 | 0.594 | 0.572 | 0.534 | 0.655 |
| Hbest | 4 | 32 | 0.599 | 0.615 | 0.447 | 0.597 | 0.534 | 0.336 |
| Hbest | 5 | 16 | 0.448 | 0.499 | 0.529 | 0.488 | 0.336 | 0.316 |
| Hbest | 5 | 32 | 0.526 | 0.441 | 0.628 | 0.632 | 0.367 | 0.408 |
| Hbest | 6 | 16 | 0.565 | 0.587 | 0.574 | 0.586 | 0.637 | 0.667 |
| Hbest | 6 | 32 | 0.595 | 0.610 | 0.575 | 0.616 | 0.756 | 0.789 |

## Low-Harm Ranking

| run | fold | h | low10 adv | low10 harm | low10 dd_rel | low10 mdd | low10 harm<=0 | high10 harm | high10 dd_rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | +0.00023 | +0.00024 | 0.248 | 0.175 | 0.000 | +0.00050 | 0.277 |
| Hbest | 4 | 32 | +0.00035 | +0.00082 | 0.342 | 0.225 | 0.050 | +0.00030 | 0.236 |
| Hbest | 5 | 16 | +0.00045 | +0.00045 | 0.573 | 0.167 | 0.000 | +0.00029 | 0.194 |
| Hbest | 5 | 32 | +0.00056 | +0.00034 | 0.447 | 0.071 | 0.024 | +0.00038 | 0.146 |
| Hbest | 6 | 16 | -0.00001 | +0.00008 | 0.396 | 0.792 | 0.158 | +0.00048 | 0.784 |
| Hbest | 6 | 32 | +0.00000 | +0.00007 | 0.317 | 0.733 | 0.218 | +0.00064 | 0.898 |

## Combined Score Top-Decile

| run | fold | h | score | top10 adv | top10 harm | top10 dd_rel | top10 recovery | top10 mdd | top20 adv | top20 harm |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | A_adv_only | +0.00118 | +0.00042 | 0.202 | +0.03921 | 0.150 | +0.00073 | +0.00031 |
| Hbest | 4 | 16 | B_adv_minus_harm | +0.00078 | +0.00029 | 0.155 | +0.02698 | 0.125 | +0.00049 | +0.00034 |
| Hbest | 4 | 16 | C_adv_minus_dd_plus_recovery | +0.00106 | +0.00030 | 0.152 | +0.03838 | 0.100 | +0.00065 | +0.00030 |
| Hbest | 4 | 16 | D_all | +0.00100 | +0.00032 | 0.189 | +0.03310 | 0.150 | +0.00051 | +0.00033 |
| Hbest | 4 | 32 | A_adv_only | +0.00152 | +0.00050 | 0.094 | +0.02250 | 0.150 | +0.00090 | +0.00042 |
| Hbest | 4 | 32 | B_adv_minus_harm | +0.00135 | +0.00045 | 0.117 | +0.02038 | 0.125 | +0.00074 | +0.00059 |
| Hbest | 4 | 32 | C_adv_minus_dd_plus_recovery | +0.00151 | +0.00046 | 0.108 | +0.02238 | 0.125 | +0.00088 | +0.00058 |
| Hbest | 4 | 32 | D_all | +0.00133 | +0.00040 | 0.118 | +0.01760 | 0.125 | +0.00078 | +0.00066 |
| Hbest | 5 | 16 | A_adv_only | +0.00060 | +0.00052 | 0.486 | +0.04153 | 0.310 | +0.00038 | +0.00057 |
| Hbest | 5 | 16 | B_adv_minus_harm | +0.00053 | +0.00049 | 0.524 | +0.03245 | 0.238 | +0.00035 | +0.00047 |
| Hbest | 5 | 16 | C_adv_minus_dd_plus_recovery | +0.00051 | +0.00052 | 0.627 | +0.02688 | 0.190 | +0.00038 | +0.00044 |
| Hbest | 5 | 16 | D_all | +0.00047 | +0.00048 | 0.571 | +0.02293 | 0.190 | +0.00033 | +0.00042 |
| Hbest | 5 | 32 | A_adv_only | +0.00088 | +0.00052 | 0.302 | +0.02684 | 0.357 | +0.00072 | +0.00052 |
| Hbest | 5 | 32 | B_adv_minus_harm | +0.00092 | +0.00049 | 0.329 | +0.03298 | 0.310 | +0.00079 | +0.00050 |
| Hbest | 5 | 32 | C_adv_minus_dd_plus_recovery | +0.00090 | +0.00054 | 0.347 | +0.02502 | 0.286 | +0.00074 | +0.00051 |
| Hbest | 5 | 32 | D_all | +0.00092 | +0.00053 | 0.392 | +0.02954 | 0.238 | +0.00075 | +0.00050 |
| Hbest | 6 | 16 | A_adv_only | +0.00004 | +0.00014 | 0.350 | +0.00833 | 0.693 | +0.00003 | +0.00016 |
| Hbest | 6 | 16 | B_adv_minus_harm | +0.00003 | +0.00010 | 0.344 | +0.00567 | 0.743 | +0.00002 | +0.00009 |
| Hbest | 6 | 16 | C_adv_minus_dd_plus_recovery | +0.00004 | +0.00018 | 0.338 | +0.01275 | 0.733 | +0.00003 | +0.00018 |
| Hbest | 6 | 16 | D_all | +0.00002 | +0.00014 | 0.367 | +0.00604 | 0.733 | +0.00003 | +0.00015 |
| Hbest | 6 | 32 | A_adv_only | +0.00006 | +0.00011 | 0.273 | +0.00476 | 0.673 | +0.00006 | +0.00012 |
| Hbest | 6 | 32 | B_adv_minus_harm | +0.00001 | +0.00007 | 0.306 | +0.00299 | 0.673 | +0.00002 | +0.00008 |
| Hbest | 6 | 32 | C_adv_minus_dd_plus_recovery | +0.00005 | +0.00014 | 0.253 | +0.00435 | 0.634 | +0.00004 | +0.00018 |
| Hbest | 6 | 32 | D_all | +0.00004 | +0.00010 | 0.263 | +0.00481 | 0.594 | +0.00003 | +0.00013 |

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
