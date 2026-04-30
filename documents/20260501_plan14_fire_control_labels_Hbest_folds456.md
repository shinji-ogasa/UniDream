# Plan14 Fire-Control Label Probe

## Setup

- config: `configs/trading_wm_control_headonly.yaml`
- folds: `4,5,6`
- horizons: `16,32`
- sample: adapter fire bars only, chronological 60%/40% probe split
- labels: fire_harm, drawdown_worsening, trough_exit, fire_advantage

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 97.3% | 11.4%/994 | -0.0110 |
| Hbest | 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 97.0% | 12.1%/1065 | -0.0407 |
| Hbest | 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 99.1% | 28.4%/2506 | -0.1802 |

## Label Quality

| run | fold | h | fire samples | harm AUC | harm PR | harm top10 | DD worse AUC | DD worse PR | trough AUC | trough PR | adv corr | adv top10 | adv spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | 16 | 994 | 0.455 | 0.336 | 0.300 | 0.514 | 0.666 | 0.548 | 0.454 | 0.323 | +0.00118 | +0.00114 |
| Hbest | 4 | 32 | 984 | 0.458 | 0.409 | 0.225 | 0.603 | 0.836 | 0.587 | 0.482 | 0.307 | +0.00152 | +0.00141 |
| Hbest | 5 | 16 | 1050 | 0.619 | 0.585 | 0.595 | 0.426 | 0.633 | 0.450 | 0.490 | 0.151 | +0.00060 | +0.00061 |
| Hbest | 5 | 32 | 1047 | 0.620 | 0.655 | 0.833 | 0.421 | 0.699 | 0.502 | 0.608 | 0.230 | +0.00088 | +0.00067 |
| Hbest | 6 | 16 | 2506 | 0.739 | 0.571 | 0.713 | 0.558 | 0.632 | 0.549 | 0.485 | 0.241 | +0.00004 | +0.00023 |
| Hbest | 6 | 32 | 2506 | 0.743 | 0.605 | 0.723 | 0.551 | 0.747 | 0.532 | 0.469 | 0.312 | +0.00006 | +0.00035 |

## Gate Readiness Check

| criterion | pass |
|---|---:|
| fire_harm_auc_ge_0_58 | False |
| drawdown_worsening_auc_ge_0_58 | False |
| trough_exit_auc_ge_0_55 | False |
| fire_advantage_top_decile_positive | True |

## Interpretation

- `fire_harm` and `drawdown_worsening` are useful only if AUC is reproducibly above the threshold across folds/horizons.
- `fire_advantage` is useful only if the predicted top decile has positive realized adapter advantage.
- If these fail, adding WM heads or AC freedom is not justified; the direct labels are not separable enough yet.
