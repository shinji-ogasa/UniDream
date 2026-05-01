# Plan15-D Danger Fire Score Probe

## Setup

- config: `configs/trading_wm_control_headonly.yaml`
- folds: `4,5,6`
- horizon: `32`
- scope: diagnostic score/guard simulation only; no production guard, WM head, AC unlock, or config adoption.

## Base Policy

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | fire_pnl | eval fires |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 11.4%/994 | -0.0110 | 394 |
| Hbest | 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 12.1%/1065 | -0.0407 | 419 |
| Hbest | 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 28.4%/2506 | -0.1802 | 1003 |

## Score Predictability

| run | fold | target | positive | AUC | PR-AUC | top10 positive |
|---|---:|---|---:|---:|---:|---:|
| Hbest | 4 | pre_dd_type | 0.706 | 0.763 | 0.895 | 1.000 |
| Hbest | 4 | future_mdd | 0.703 | 0.804 | 0.906 | 0.975 |
| Hbest | 4 | global_mdd | 0.142 | 0.775 | 0.347 | 0.450 |
| Hbest | 5 | pre_dd_type | 0.489 | 0.695 | 0.587 | 0.452 |
| Hbest | 5 | future_mdd | 0.465 | 0.703 | 0.661 | 0.833 |
| Hbest | 5 | global_mdd | 0.110 | 0.861 | 0.500 | 0.429 |
| Hbest | 6 | pre_dd_type | 0.532 | 0.681 | 0.678 | 0.703 |
| Hbest | 6 | future_mdd | 0.669 | 0.774 | 0.889 | 1.000 |
| Hbest | 6 | global_mdd | 0.830 | 0.651 | 0.909 | 1.000 |

## Variant Mean

| variant | AlphaEx mean | SharpeD mean | MaxDDD mean | turnover mean | long max | short max |
|---|---:|---:|---:|---:|---:|---:|
| danger_strict_top30_scale0 | +0.33 | +0.010 | +0.33 | 5.55 | 2.3% | 0.0% |
| predd_only_top30_scale0 | -4.38 | +0.009 | +0.33 | 5.97 | 2.6% | 0.0% |
| predd_only_top20_scale0 | -3.62 | +0.005 | +0.37 | 5.74 | 2.7% | 0.0% |
| danger_strict_top20_scale0 | -6.49 | -0.001 | +0.37 | 5.62 | 2.4% | 0.0% |
| danger_minus_adv_top20_scale0 | -5.24 | -0.001 | +0.36 | 5.47 | 2.4% | 0.0% |
| danger_minus_adv_top30_scale0 | -6.20 | -0.001 | +0.36 | 5.43 | 2.3% | 0.0% |
| predd_only_top10_scale0 | -6.68 | -0.005 | +0.45 | 5.51 | 2.9% | 0.0% |
| danger_strict_top30_scale0.5 | -4.55 | -0.006 | +0.45 | 4.88 | 2.3% | 0.0% |
| predd_only_top30_scale0.5 | -6.95 | -0.007 | +0.45 | 5.10 | 2.6% | 0.0% |
| danger_strict_top10_scale0 | -7.33 | -0.008 | +0.45 | 5.20 | 2.6% | 0.0% |
| predd_only_top20_scale0.5 | -6.56 | -0.009 | +0.47 | 5.00 | 2.7% | 0.0% |
| danger_minus_adv_top10_scale0 | -7.66 | -0.010 | +0.44 | 5.05 | 2.6% | 0.0% |
| danger_strict_top20_scale0.5 | -7.92 | -0.012 | +0.47 | 4.92 | 2.4% | 0.0% |
| danger_minus_adv_top20_scale0.5 | -7.28 | -0.012 | +0.46 | 4.84 | 2.4% | 0.0% |
| danger_minus_adv_top30_scale0.5 | -7.75 | -0.012 | +0.47 | 4.81 | 2.3% | 0.0% |
| predd_only_top10_scale0.5 | -8.10 | -0.014 | +0.51 | 4.89 | 2.9% | 0.0% |
| oracle_not_lowrisk_scale0 | -6.66 | -0.014 | +0.30 | 3.81 | 2.1% | 0.0% |
| danger_strict_top10_scale0.5 | -8.36 | -0.015 | +0.51 | 4.72 | 2.6% | 0.0% |
| danger_minus_adv_top10_scale0.5 | -8.52 | -0.016 | +0.50 | 4.64 | 2.6% | 0.0% |
| future_mdd_only_runmean_top20_scale0 | -8.36 | -0.017 | +0.53 | 4.02 | 2.7% | 0.0% |

## Best Variant By Fold

| variant | fold | AlphaEx | SharpeD | MaxDDD | turnover | selected | selected adv | selected fire_pnl | selected preDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| danger_strict_top30_scale0 | 4 | +0.13 | +0.025 | +0.28 | 6.45 | 119 | +0.00013 | -0.0932 | 116 |
| danger_strict_top30_scale0 | 5 | +0.57 | -0.027 | +0.61 | 4.02 | 126 | +0.00015 | -0.1300 | 82 |
| danger_strict_top30_scale0 | 6 | +0.30 | +0.031 | +0.09 | 6.19 | 301 | -0.00020 | -0.3602 | 228 |
| predd_only_top30_scale0 | 4 | +0.11 | +0.023 | +0.28 | 6.66 | 119 | +0.00015 | -0.0785 | 115 |
| predd_only_top30_scale0 | 5 | -13.91 | -0.046 | +0.75 | 3.12 | 126 | -0.00020 | -0.0528 | 77 |
| predd_only_top30_scale0 | 6 | +0.64 | +0.051 | -0.06 | 8.12 | 301 | -0.00017 | -0.5950 | 223 |
| predd_only_top20_scale0 | 4 | +0.10 | +0.022 | +0.28 | 6.44 | 79 | +0.00021 | -0.0758 | 78 |
| predd_only_top20_scale0 | 5 | -11.36 | -0.044 | +0.75 | 3.60 | 84 | -0.00029 | -0.0653 | 48 |
| predd_only_top20_scale0 | 6 | +0.39 | +0.037 | +0.09 | 7.18 | 201 | -0.00016 | -0.4678 | 150 |
| danger_strict_top20_scale0 | 4 | +0.06 | +0.017 | +0.35 | 6.25 | 79 | +0.00018 | -0.0384 | 79 |
| danger_strict_top20_scale0 | 5 | -19.85 | -0.052 | +0.69 | 3.81 | 84 | +0.00010 | -0.0469 | 55 |
| danger_strict_top20_scale0 | 6 | +0.33 | +0.033 | +0.08 | 6.81 | 201 | -0.00024 | -0.3230 | 149 |
| danger_minus_adv_top20_scale0 | 4 | +0.04 | +0.015 | +0.35 | 5.94 | 79 | +0.00018 | -0.0228 | 79 |
| danger_minus_adv_top20_scale0 | 5 | -16.00 | -0.046 | +0.61 | 4.08 | 84 | +0.00026 | -0.0625 | 47 |
| danger_minus_adv_top20_scale0 | 6 | +0.25 | +0.028 | +0.12 | 6.39 | 201 | -0.00027 | -0.3026 | 153 |

## Interpretation

- Best ranked variant: `danger_strict_top30_scale0` with mean AlphaEx +0.33, SharpeD +0.010, MaxDDD +0.33.
- A usable guard needs MaxDDD <= 0, turnover <= 3.5, long <= 3%, short = 0%, and no severe AlphaEx loss across folds.
- If every variant still has MaxDDD > 0 or kills alpha, do not proceed to Plan16 adoption.