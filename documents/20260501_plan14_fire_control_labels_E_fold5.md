# Plan14 Fire-Control Label Probe

## Setup

- config: `configs/trading.yaml`
- folds: `5`
- horizons: `16,32`
- sample: adapter fire bars only, chronological 60%/40% probe split
- labels: fire_harm, drawdown_worsening, trough_exit, fire_advantage

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 98.1% | 3.3%/288 | +0.1059 |

## Label Quality

| run | fold | h | fire samples | harm AUC | harm PR | harm top10 | DD worse AUC | DD worse PR | trough AUC | trough PR | adv corr | adv top10 | adv spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 5 | 16 | 288 | 0.856 | 0.817 | 0.750 | 0.573 | 0.652 | 0.532 | 0.613 | -0.078 | +0.00043 | +0.00007 |
| E | 5 | 32 | 288 | 0.918 | 0.905 | 1.000 | 0.632 | 0.725 | 0.453 | 0.625 | 0.416 | +0.00323 | +0.00293 |

## Gate Readiness Check

| criterion | pass |
|---|---:|
| fire_harm_auc_ge_0_58 | True |
| drawdown_worsening_auc_ge_0_58 | False |
| trough_exit_auc_ge_0_55 | False |
| fire_advantage_top_decile_positive | True |

## Interpretation

- `fire_harm` and `drawdown_worsening` are useful only if AUC is reproducibly above the threshold across folds/horizons.
- `fire_advantage` is useful only if the predicted top decile has positive realized adapter advantage.
- If these fail, adding WM heads or AC freedom is not justified; the direct labels are not separable enough yet.
