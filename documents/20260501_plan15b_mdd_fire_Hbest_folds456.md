# Plan15-B MDD Fire Label Probe

## Setup

- config: `configs/trading_wm_control_headonly.yaml`
- folds: `4,5,6`
- horizons: `16,32`
- primary horizon: `32`
- scope: MDD-window label quality only; no guard, WM head, or AC unlock.

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl | global MDD | fire in MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 97.3% | 11.4%/994 | -0.0110 | -53.55pt | 7.4% |
| Hbest | 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 97.0% | 12.1%/1065 | -0.0407 | -24.35pt | 4.0% |
| Hbest | 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 99.1% | 28.4%/2506 | -0.1802 | -42.55pt | 19.9% |

## MDD Label Predictability

| run | fold | h | samples | eval | future MDD AUC | future MDD top10 | pre-DD AUC | global MDD AUC | postDD q AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | 994 | 398 | 0.586 | 0.700 | 0.596 | 0.413 | 0.715 |
| Hbest | 4 | 32 | 984 | 394 | 0.572 | 0.800 | 0.507 | 0.695 | 0.403 |
| Hbest | 5 | 16 | 1050 | 420 | 0.316 | 0.381 | 0.760 | 0.597 | 0.300 |
| Hbest | 5 | 32 | 1047 | 419 | 0.538 | 0.476 | 0.284 | 0.591 | 0.426 |
| Hbest | 6 | 16 | 2506 | 1003 | 0.563 | 0.762 | 0.572 | 0.378 | 0.773 |
| Hbest | 6 | 32 | 2506 | 1003 | 0.662 | 0.941 | 0.554 | 0.378 | 0.768 |

## Advantage vs MDD Risk Ranking

| run | fold | h | adv top10 | adv postDD | adv futureMDD | adv globalMDD | low postDD adv | low postDD | low overlap adv | low overlap futureMDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | +0.00118 | +0.00042 | 0.500 | 0.150 | +0.00023 | +0.00018 | -0.00004 | 0.350 |
| Hbest | 4 | 32 | +0.00152 | +0.00056 | 0.550 | 0.150 | +0.00036 | +0.00045 | +0.00033 | 0.500 |
| Hbest | 5 | 16 | +0.00060 | +0.00052 | 0.786 | 0.310 | +0.00048 | +0.00048 | +0.00023 | 0.762 |
| Hbest | 5 | 32 | +0.00088 | +0.00052 | 0.238 | 0.357 | +0.00061 | +0.00032 | +0.00053 | 0.238 |
| Hbest | 6 | 16 | +0.00004 | +0.00014 | 0.614 | 0.693 | -0.00000 | +0.00007 | -0.00001 | 0.614 |
| Hbest | 6 | 32 | +0.00006 | +0.00013 | 0.584 | 0.673 | +0.00001 | +0.00008 | +0.00000 | 0.446 |

## Combined Score Top-Decile

| run | fold | h | score | top10 adv | postDD | mdd contrib | futureMDD | preDD | globalMDD | top20 adv | top20 postDD |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | A_adv_only | +0.00118 | +0.00042 | +0.00042 | 0.500 | 0.000 | 0.150 | +0.00073 | +0.00031 |
| Hbest | 4 | 16 | B_adv_minus_postdd | +0.00060 | +0.00025 | +0.00024 | 0.375 | 0.000 | 0.100 | +0.00053 | +0.00027 |
| Hbest | 4 | 16 | C_adv_minus_overlap_predd | +0.00104 | +0.00037 | +0.00037 | 0.450 | 0.000 | 0.150 | +0.00054 | +0.00036 |
| Hbest | 4 | 16 | D_adv_minus_all_mdd | +0.00096 | +0.00037 | +0.00036 | 0.425 | 0.000 | 0.125 | +0.00051 | +0.00032 |
| Hbest | 4 | 32 | A_adv_only | +0.00152 | +0.00056 | +0.00050 | 0.550 | 0.050 | 0.150 | +0.00090 | +0.00046 |
| Hbest | 4 | 32 | B_adv_minus_postdd | +0.00135 | +0.00050 | +0.00043 | 0.550 | 0.050 | 0.125 | +0.00072 | +0.00053 |
| Hbest | 4 | 32 | C_adv_minus_overlap_predd | +0.00108 | +0.00043 | +0.00040 | 0.525 | 0.075 | 0.050 | +0.00087 | +0.00053 |
| Hbest | 4 | 32 | D_adv_minus_all_mdd | +0.00099 | +0.00041 | +0.00036 | 0.475 | 0.050 | 0.050 | +0.00088 | +0.00052 |
| Hbest | 5 | 16 | A_adv_only | +0.00060 | +0.00052 | +0.00052 | 0.786 | 0.000 | 0.310 | +0.00038 | +0.00057 |
| Hbest | 5 | 16 | B_adv_minus_postdd | +0.00055 | +0.00050 | +0.00050 | 0.810 | 0.000 | 0.238 | +0.00035 | +0.00047 |
| Hbest | 5 | 16 | C_adv_minus_overlap_predd | +0.00041 | +0.00057 | +0.00056 | 0.762 | 0.000 | 0.357 | +0.00030 | +0.00063 |
| Hbest | 5 | 16 | D_adv_minus_all_mdd | +0.00046 | +0.00049 | +0.00049 | 0.881 | 0.000 | 0.214 | +0.00032 | +0.00055 |
| Hbest | 5 | 32 | A_adv_only | +0.00088 | +0.00052 | +0.00052 | 0.238 | 0.000 | 0.357 | +0.00072 | +0.00055 |
| Hbest | 5 | 32 | B_adv_minus_postdd | +0.00094 | +0.00050 | +0.00049 | 0.286 | 0.000 | 0.310 | +0.00077 | +0.00049 |
| Hbest | 5 | 32 | C_adv_minus_overlap_predd | +0.00062 | +0.00040 | +0.00037 | 0.024 | 0.071 | 0.405 | +0.00046 | +0.00046 |
| Hbest | 5 | 32 | D_adv_minus_all_mdd | +0.00061 | +0.00034 | +0.00032 | 0.071 | 0.024 | 0.476 | +0.00055 | +0.00038 |
| Hbest | 6 | 16 | A_adv_only | +0.00004 | +0.00014 | +0.00014 | 0.614 | 0.020 | 0.693 | +0.00003 | +0.00016 |
| Hbest | 6 | 16 | B_adv_minus_postdd | +0.00003 | +0.00010 | +0.00010 | 0.554 | 0.059 | 0.743 | +0.00002 | +0.00009 |
| Hbest | 6 | 16 | C_adv_minus_overlap_predd | +0.00001 | +0.00020 | +0.00020 | 0.604 | 0.040 | 0.713 | +0.00001 | +0.00021 |
| Hbest | 6 | 16 | D_adv_minus_all_mdd | +0.00004 | +0.00013 | +0.00013 | 0.584 | 0.059 | 0.713 | +0.00001 | +0.00015 |
| Hbest | 6 | 32 | A_adv_only | +0.00006 | +0.00013 | +0.00011 | 0.584 | 0.050 | 0.673 | +0.00006 | +0.00013 |
| Hbest | 6 | 32 | B_adv_minus_postdd | +0.00002 | +0.00009 | +0.00007 | 0.554 | 0.079 | 0.713 | +0.00002 | +0.00010 |
| Hbest | 6 | 32 | C_adv_minus_overlap_predd | +0.00005 | +0.00018 | +0.00016 | 0.554 | 0.030 | 0.634 | +0.00004 | +0.00018 |
| Hbest | 6 | 32 | D_adv_minus_all_mdd | +0.00004 | +0.00009 | +0.00008 | 0.515 | 0.069 | 0.653 | +0.00003 | +0.00013 |

## Readiness

| criterion | pass |
|---|---:|
| adv_top10_positive_all | True |
| low_postdd_keeps_positive_adv_all | True |
| combined_D_improves_mdd_risk_all | False |
| future_mdd_overlap_auc_ge_0_55_all | False |

## Interpretation

- If MDD-window labels are not separable, Plan16 guard should not use them.
- A useful risk score must lower post-fire DD contribution or MDD overlap without killing fire_advantage.
- This probe is diagnostic only and does not alter `configs/trading.yaml`.
