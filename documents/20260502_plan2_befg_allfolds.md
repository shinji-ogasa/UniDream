# Plan 2 Exploration Board Probe

Config: `configs/trading.yaml`
Folds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13`

## Selector Aggregate

### all

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_safe_small | 14 | 0.032 | -1.736 | 0.045 | 0.499 | -0.005 | 1.000 | 0.000 | 0.000 | 0.500 |
| E_bootstrap_uncertainty | 14 | -7.106 | -99.479 | -0.000 | 0.000 | -0.001 | 9.000 | 0.000 | 0.000 | 0.500 |
| F_listwise | 14 | -0.221 | -5.334 | -0.148 | 0.110 | 0.004 | 42.000 | 0.000 | 0.071 | 0.500 |
| G_vol_regime_safe | 14 | -1.493 | -14.487 | 0.062 | 0.466 | -0.018 | 7.000 | 0.000 | 0.143 | 0.500 |

Nested leave-one-fold selector:

| folds | alpha mean | alpha worst | maxdd worst | turnover max |
|---:|---:|---:|---:|---:|
| 14 | -7.073 | -99.479 | 0.499 | 9.000 |

### f456

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_safe_small | 3 | -0.008 | -0.024 | 0.005 | 0.014 | -0.001 | 0.500 | 0.000 | 0.000 | 0.333 |
| E_bootstrap_uncertainty | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| F_listwise | 3 | 0.494 | -0.049 | -0.580 | 0.028 | 0.040 | 42.000 | 0.000 | 0.000 | 0.333 |
| G_vol_regime_safe | 3 | -0.079 | -0.212 | 0.005 | 0.014 | -0.010 | 1.000 | 0.000 | 0.000 | 0.333 |

Nested leave-one-fold selector:

| folds | alpha mean | alpha worst | maxdd worst | turnover max |
|---:|---:|---:|---:|---:|
| 3 | -0.008 | -0.024 | 0.014 | 0.500 |

### f045

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_safe_small | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| E_bootstrap_uncertainty | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| F_listwise | 3 | 0.510 | 0.000 | -0.589 | 0.000 | 0.042 | 42.000 | 0.000 | 0.000 | 0.333 |
| G_vol_regime_safe | 3 | -0.071 | -0.212 | -0.000 | 0.000 | -0.009 | 1.000 | 0.000 | 0.000 | 0.333 |

Nested leave-one-fold selector:

| folds | alpha mean | alpha worst | maxdd worst | turnover max |
|---:|---:|---:|---:|---:|
| 3 | 0.000 | 0.000 | 0.000 | 0.000 |

## Triple Barrier Aggregate

### all

| target | folds | density | AUC mean | AUC worst | false-active worst | recall worst | pred rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| h16_k100_down | 14 | 0.200 | 0.588 | 0.550 | 0.420 | 0.031 | 0.187 |
| h16_k100_up_safe | 14 | 0.124 | 0.544 | 0.492 | 0.631 | 0.001 | 0.184 |
| h32_k125_down | 14 | 0.161 | 0.600 | 0.572 | 0.501 | 0.031 | 0.185 |
| h32_k125_up_safe | 14 | 0.104 | 0.563 | 0.503 | 0.644 | 0.000 | 0.234 |
| h64_k150_down | 14 | 0.128 | 0.627 | 0.569 | 0.434 | 0.044 | 0.210 |
| h64_k150_up_safe | 14 | 0.091 | 0.629 | 0.557 | 0.482 | 0.000 | 0.240 |

### f456

| target | folds | density | AUC mean | AUC worst | false-active worst | recall worst | pred rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| h16_k100_down | 3 | 0.221 | 0.598 | 0.590 | 0.415 | 0.031 | 0.193 |
| h16_k100_up_safe | 3 | 0.133 | 0.536 | 0.492 | 0.631 | 0.008 | 0.244 |
| h32_k125_down | 3 | 0.189 | 0.604 | 0.599 | 0.256 | 0.031 | 0.138 |
| h32_k125_up_safe | 3 | 0.109 | 0.570 | 0.514 | 0.644 | 0.000 | 0.236 |
| h64_k150_down | 3 | 0.140 | 0.627 | 0.576 | 0.240 | 0.044 | 0.144 |
| h64_k150_up_safe | 3 | 0.081 | 0.626 | 0.580 | 0.426 | 0.033 | 0.190 |

### f045

| target | folds | density | AUC mean | AUC worst | false-active worst | recall worst | pred rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| h16_k100_down | 3 | 0.209 | 0.584 | 0.553 | 0.415 | 0.031 | 0.206 |
| h16_k100_up_safe | 3 | 0.135 | 0.536 | 0.492 | 0.631 | 0.008 | 0.263 |
| h32_k125_down | 3 | 0.173 | 0.605 | 0.601 | 0.256 | 0.031 | 0.150 |
| h32_k125_up_safe | 3 | 0.112 | 0.541 | 0.514 | 0.644 | 0.000 | 0.307 |
| h64_k150_down | 3 | 0.136 | 0.624 | 0.576 | 0.257 | 0.044 | 0.185 |
| h64_k150_up_safe | 3 | 0.093 | 0.602 | 0.580 | 0.426 | 0.033 | 0.264 |

## Fold Detail

### Fold 0

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.004351070907715938 regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.004351070907715938 regime=high_vol uq=None cd=0 danger=None |

### Fold 1

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | 0.540 | -0.045 | 0.005 | 0.500 | 0.000 | 1.000 | thr=0.0021141923142814116 regime=low_vol uq=None cd=0 danger=None |

### Fold 2

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0144827034318092 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | -99.479 | -0.000 | -0.007 | 9.000 | 0.000 | 0.998 | thr=0.0012234802990879776 regime=all uq=0.7 cd=0 danger=None |
| G_vol_regime_safe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.002379934065514388 regime=low_vol uq=None cd=0 danger=None |

### Fold 3

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 2.704 | 0.000 | 0.005 | 1.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | 2.704 | 0.000 | 0.005 | 1.000 | 0.000 | 1.000 | thr=0.005 regime=high_vol uq=None cd=0 danger=None |

### Fold 4

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| F_listwise | 1.530 | -1.768 | 0.126 | 42.000 | 0.000 | 0.980 | thr=0.0058306094422053275 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -0.212 | -0.000 | -0.027 | 1.000 | 0.000 | 0.997 | thr=0.0 regime=low_vol uq=None cd=0 danger=None |

### Fold 5

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.010462949153978718 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=high_vol uq=None cd=0 danger=None |

### Fold 6

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | -0.024 | 0.014 | -0.003 | 0.500 | 0.000 | 0.998 | thr=0.0026074601032685728 regime=all uq=None cd=0 danger=None |
| F_listwise | -0.049 | 0.028 | -0.006 | 1.000 | 0.000 | 0.998 | thr=0.006353515870307415 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -0.024 | 0.014 | -0.003 | 0.500 | 0.000 | 0.998 | thr=0.0026074601032685728 regime=high_vol uq=None cd=0 danger=None |

### Fold 7

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.003707281177708597 regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.008709838674281913 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.003707281177708597 regime=high_vol uq=None cd=0 danger=None |

### Fold 8

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | 0.056 | -0.082 | 0.010 | 0.500 | 0.000 | 1.000 | thr=0.001 regime=low_vol uq=None cd=0 danger=None |

### Fold 9

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.589 | 0.060 | 0.014 | 1.000 | 0.000 | 1.000 | thr=0.000698209045354932 regime=all uq=None cd=0 danger=None |
| F_listwise | 1.298 | -0.003 | 0.032 | 1.500 | 0.000 | 1.000 | thr=0.0034487628384363977 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -0.306 | 0.060 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0006216875441248286 regime=low_vol uq=None cd=0 danger=None |

### Fold 10

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | -0.807 | 0.000 | -0.012 | 0.500 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=None |
| F_listwise | -5.334 | -0.432 | -0.080 | 17.000 | 0.000 | 0.997 | thr=0.0025 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -7.499 | 0.466 | -0.113 | 6.500 | 0.000 | 0.997 | thr=0.0005 regime=high_vol uq=None cd=0 danger=None |

### Fold 11

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.001220910114288054 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -14.487 | -0.000 | -0.061 | 7.000 | 0.000 | 0.997 | thr=0.000532181382309143 regime=mid_vol uq=None cd=0 danger=None |

### Fold 12

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0020577712036322525 regime=all uq=None cd=0 danger=None |
| F_listwise | -0.544 | 0.110 | -0.013 | 1.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0020577712036322525 regime=high_vol uq=None cd=0 danger=None |

### Fold 13

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | -1.736 | 0.499 | -0.067 | 0.500 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=None |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0015427461232760146 regime=all uq=0.5 cd=0 danger=None |
| G_vol_regime_safe | -1.394 | 0.400 | -0.055 | 0.500 | 0.000 | 0.998 | thr=0.001348528847379298 regime=high_vol uq=None cd=0 danger=None |

