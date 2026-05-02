# Plan 2 Exploration Board Probe

Config: `configs/trading.yaml`
Folds: `0, 4, 5, 6`

## Selector Aggregate

### all

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_safe_small | 4 | -0.006 | -0.024 | 0.004 | 0.014 | -0.001 | 0.500 | 0.000 | 0.000 | 0.500 |
| D_risk_sensitive | 4 | 0.100 | -0.024 | -0.162 | 0.014 | 0.011 | 2.500 | 0.000 | 0.250 | 0.500 |
| D_risk_sensitive_floor005 | 4 | 0.096 | -0.041 | -0.159 | 0.024 | 0.012 | 2.500 | 0.000 | 0.250 | 0.500 |
| D_risk_sensitive_floor010 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| D_risk_sensitive_tbguard | 4 | 0.114 | 0.000 | -0.170 | 0.000 | 0.012 | 2.500 | 0.000 | 0.500 | 0.500 |
| D_risk_sensitive_tbguard_floor005 | 4 | 0.096 | -0.041 | -0.159 | 0.024 | 0.012 | 2.500 | 0.000 | 0.250 | 0.500 |
| E_bootstrap_uncertainty | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| F_listwise | 4 | 0.370 | -0.049 | -0.435 | 0.028 | 0.030 | 42.000 | 0.000 | 0.000 | 0.500 |
| G_vol_regime_safe | 4 | -0.059 | -0.212 | 0.004 | 0.014 | -0.008 | 1.000 | 0.000 | 0.000 | 0.500 |

### f456

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_safe_small | 3 | -0.008 | -0.024 | 0.005 | 0.014 | -0.001 | 0.500 | 0.000 | 0.000 | 0.333 |
| D_risk_sensitive | 3 | 0.133 | -0.024 | -0.216 | 0.014 | 0.015 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_floor005 | 3 | 0.127 | -0.041 | -0.212 | 0.024 | 0.015 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_floor010 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| D_risk_sensitive_tbguard | 3 | 0.152 | 0.000 | -0.227 | 0.000 | 0.016 | 2.500 | 0.000 | 0.667 | 0.333 |
| D_risk_sensitive_tbguard_floor005 | 3 | 0.127 | -0.041 | -0.212 | 0.024 | 0.015 | 2.500 | 0.000 | 0.333 | 0.333 |
| E_bootstrap_uncertainty | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| F_listwise | 3 | 0.494 | -0.049 | -0.580 | 0.028 | 0.040 | 42.000 | 0.000 | 0.000 | 0.333 |
| G_vol_regime_safe | 3 | -0.079 | -0.212 | 0.005 | 0.014 | -0.010 | 1.000 | 0.000 | 0.000 | 0.333 |

### f045

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_safe_small | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| D_risk_sensitive | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_floor005 | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_floor010 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| D_risk_sensitive_tbguard | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_tbguard_floor005 | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| E_bootstrap_uncertainty | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| F_listwise | 3 | 0.510 | 0.000 | -0.589 | 0.000 | 0.042 | 42.000 | 0.000 | 0.000 | 0.333 |
| G_vol_regime_safe | 3 | -0.071 | -0.212 | -0.000 | 0.000 | -0.009 | 1.000 | 0.000 | 0.000 | 0.333 |

## Triple Barrier Aggregate

### all

| target | folds | density | AUC mean | AUC worst | false-active worst | recall worst | pred rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| h16_k100_down | 4 | 0.209 | 0.586 | 0.553 | 0.415 | 0.031 | 0.186 |
| h16_k100_up_safe | 4 | 0.130 | 0.545 | 0.492 | 0.631 | 0.008 | 0.217 |
| h32_k125_down | 4 | 0.177 | 0.603 | 0.599 | 0.256 | 0.031 | 0.145 |
| h32_k125_up_safe | 4 | 0.109 | 0.570 | 0.514 | 0.644 | 0.000 | 0.244 |
| h64_k150_down | 4 | 0.133 | 0.625 | 0.576 | 0.257 | 0.044 | 0.176 |
| h64_k150_up_safe | 4 | 0.086 | 0.628 | 0.580 | 0.426 | 0.033 | 0.230 |

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
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.004351070907715938 regime=all uq=None danger=None |
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None danger=None |
| D_risk_sensitive_floor005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive_floor010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.01 regime=all uq=None danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None danger=0.43078277740968884 |
| D_risk_sensitive_tbguard_floor005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=0.43078277740968884 |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=0.5 danger=None |
| G_vol_regime_safe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.004351070907715938 regime=high_vol uq=None danger=None |

### Fold 4

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None danger=None |
| D_risk_sensitive | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive_floor005 | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive_floor010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.01 regime=all uq=None danger=None |
| D_risk_sensitive_tbguard | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None danger=0.37949841729851214 |
| D_risk_sensitive_tbguard_floor005 | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None danger=0.37949841729851214 |
| F_listwise | 1.530 | -1.768 | 0.126 | 42.000 | 0.000 | 0.980 | thr=0.0058306094422053275 regime=all uq=None danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 danger=None |
| G_vol_regime_safe | -0.212 | -0.000 | -0.027 | 1.000 | 0.000 | 0.997 | thr=0.0 regime=low_vol uq=None danger=None |

### Fold 5

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive_floor005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive_floor010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.01 regime=all uq=None danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=0.3589105208245268 |
| D_risk_sensitive_tbguard_floor005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=0.3589105208245268 |
| F_listwise | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.010462949153978718 regime=all uq=None danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 danger=None |
| G_vol_regime_safe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=high_vol uq=None danger=None |

### Fold 6

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| B_safe_small | -0.024 | 0.014 | -0.003 | 0.500 | 0.000 | 0.998 | thr=0.0026074601032685728 regime=all uq=None danger=None |
| D_risk_sensitive | -0.024 | 0.014 | -0.003 | 0.500 | 0.000 | 0.998 | thr=0.0020324664081611582 regime=all uq=None danger=None |
| D_risk_sensitive_floor005 | -0.041 | 0.024 | -0.002 | 0.500 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=None |
| D_risk_sensitive_floor010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.01 regime=all uq=None danger=None |
| D_risk_sensitive_tbguard | 0.032 | -0.019 | -0.000 | 0.500 | 0.000 | 0.998 | thr=0.0010893076946445368 regime=all uq=None danger=0.46034551494833514 |
| D_risk_sensitive_tbguard_floor005 | -0.041 | 0.024 | -0.002 | 0.500 | 0.000 | 1.000 | thr=0.005 regime=all uq=None danger=0.4095169925216591 |
| F_listwise | -0.049 | 0.028 | -0.006 | 1.000 | 0.000 | 0.998 | thr=0.006353515870307415 regime=all uq=None danger=None |
| E_bootstrap_uncertainty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0 regime=all uq=0.5 danger=None |
| G_vol_regime_safe | -0.024 | 0.014 | -0.003 | 0.500 | 0.000 | 0.998 | thr=0.0026074601032685728 regime=high_vol uq=None danger=None |

