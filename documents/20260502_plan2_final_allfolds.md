# Plan 2 Exploration Board Probe

Config: `configs/trading.yaml`
Folds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13`

## Selector Aggregate

### all

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_risk_sensitive | 14 | 1.652 | -1.759 | -0.012 | 0.506 | -0.001 | 11.700 | 0.000 | 0.071 | 0.500 |
| D_risk_sensitive_tbguard | 14 | 3.027 | -7.360 | -0.033 | 0.383 | -0.000 | 8.900 | 0.000 | 0.143 | 0.500 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 14 | 3.643 | -0.434 | -0.067 | 0.055 | 0.012 | 4.200 | 0.000 | 0.214 | 0.500 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 14 | -0.303 | -8.165 | -0.067 | 0.055 | 0.005 | 4.200 | 0.000 | 0.214 | 0.500 |
| D_risk_sensitive_tbguard_cd32_floor001 | 14 | 1.056 | -1.295 | -0.043 | 0.055 | 0.004 | 3.000 | 0.000 | 0.286 | 0.500 |

Nested leave-one-fold selector:

| folds | alpha mean | alpha worst | maxdd worst | turnover max |
|---:|---:|---:|---:|---:|
| 14 | 3.643 | -0.434 | 0.055 | 4.200 |

### f456

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_risk_sensitive | 3 | 0.133 | -0.024 | -0.216 | 0.014 | 0.015 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_tbguard | 3 | 0.152 | 0.000 | -0.227 | 0.000 | 0.016 | 2.500 | 0.000 | 0.667 | 0.333 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 3 | 0.153 | 0.000 | -0.228 | 0.000 | 0.017 | 2.500 | 0.000 | 0.667 | 0.333 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 3 | 0.153 | 0.000 | -0.228 | 0.000 | 0.017 | 2.500 | 0.000 | 0.667 | 0.333 |
| D_risk_sensitive_tbguard_cd32_floor001 | 3 | 0.081 | 0.000 | -0.116 | 0.000 | 0.009 | 1.000 | 0.000 | 0.667 | 0.333 |

Nested leave-one-fold selector:

| folds | alpha mean | alpha worst | maxdd worst | turnover max |
|---:|---:|---:|---:|---:|
| 3 | 0.133 | -0.024 | 0.014 | 2.500 |

### f045

| variant | folds | alpha mean | alpha worst | maxdd mean | maxdd worst | sharpe mean | turnover max | long max | pass rate | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_risk_sensitive | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_tbguard | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 3 | 0.141 | 0.000 | -0.220 | 0.000 | 0.016 | 2.500 | 0.000 | 0.333 | 0.333 |
| D_risk_sensitive_tbguard_cd32_floor001 | 3 | 0.069 | 0.000 | -0.108 | 0.000 | 0.008 | 1.000 | 0.000 | 0.333 | 0.333 |

Nested leave-one-fold selector:

| folds | alpha mean | alpha worst | maxdd worst | turnover max |
|---:|---:|---:|---:|---:|
| 3 | 0.141 | 0.000 | 0.000 | 2.500 |

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
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None cd=0 danger=0.43078277740968884 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None cd=32 danger=0.43078277740968884 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None cd=0 danger=0.43078277740968884 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0035639667547724056 regime=all uq=None cd=0 danger=0.43078277740968884 |

### Fold 1

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 3.996 | 0.000 | 0.037 | 1.000 | 0.000 | 0.997 | thr=0.005361361398900163 regime=all uq=None cd=0 danger=0.5281163995719594 |
| D_risk_sensitive_tbguard_cd32_floor001 | -1.295 | -0.000 | -0.011 | 0.500 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=32 danger=0.5281163995719594 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 3.996 | 0.000 | 0.037 | 1.000 | 0.000 | 0.997 | thr=0.005361361398900163 regime=all uq=None cd=0 danger=0.5281163995719594 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 3.996 | 0.000 | 0.037 | 1.000 | 0.000 | 0.997 | thr=0.005361361398900163 regime=all uq=None cd=0 danger=0.5281163995719594 |

### Fold 2

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=0.5152027784192731 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=32 danger=0.5152027784192731 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=0.5152027784192731 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=0.5152027784192731 |

### Fold 3

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 26.629 | 0.000 | 0.043 | 1.500 | 0.000 | 0.999 | thr=0.00331526170634212 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 41.381 | 0.000 | 0.063 | 1.000 | 0.000 | 0.999 | thr=0.00331526170634212 regime=all uq=None cd=0 danger=0.43195860287266696 |
| D_risk_sensitive_tbguard_cd32_floor001 | 14.491 | 0.000 | 0.025 | 1.000 | 0.000 | 1.000 | thr=0.002994630275459714 regime=all uq=None cd=32 danger=0.43195860287266696 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 41.381 | 0.000 | 0.063 | 1.000 | 0.000 | 0.999 | thr=0.00331526170634212 regime=all uq=None cd=0 danger=0.43195860287266696 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | -8.165 | -0.000 | -0.012 | 0.500 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=0.43195860287266696 |

### Fold 4

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None cd=0 danger=0.37949841729851214 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.206 | -0.325 | 0.024 | 1.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=32 danger=0.37949841729851214 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None cd=0 danger=0.37949841729851214 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.423 | -0.661 | 0.049 | 2.500 | 0.000 | 0.999 | thr=0.005 regime=all uq=None cd=0 danger=0.37949841729851214 |

### Fold 5

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=0.3589105208245268 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.003263331378229843 regime=all uq=None cd=32 danger=0.45903161246546936 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=0.3589105208245268 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=0 danger=0.3589105208245268 |

### Fold 6

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | -0.024 | 0.014 | -0.003 | 0.500 | 0.000 | 0.998 | thr=0.0020324664081611582 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.032 | -0.019 | -0.000 | 0.500 | 0.000 | 0.998 | thr=0.0010893076946445368 regime=all uq=None cd=0 danger=0.46034551494833514 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.037 | -0.022 | 0.002 | 0.500 | 0.000 | 1.000 | thr=0.0010893076946445368 regime=all uq=None cd=32 danger=0.5355399956231328 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.037 | -0.022 | 0.002 | 0.500 | 0.000 | 1.000 | thr=0.0010893076946445368 regime=all uq=None cd=32 danger=0.5355399956231328 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.037 | -0.022 | 0.002 | 0.500 | 0.000 | 1.000 | thr=0.0010893076946445368 regime=all uq=None cd=32 danger=0.5355399956231328 |

### Fold 7

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4559089439110864 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.005 regime=all uq=None cd=32 danger=0.4137328165213179 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4559089439110864 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4559089439110864 |

### Fold 8

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4104117737313217 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=32 danger=0.4104117737313217 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4104117737313217 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4104117737313217 |

### Fold 9

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | -0.306 | 0.060 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0011655598641181829 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0011655598641181829 regime=all uq=None cd=0 danger=0.1533333322232151 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0011655598641181829 regime=all uq=None cd=32 danger=0.1533333322232151 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0011655598641181829 regime=all uq=None cd=0 danger=0.1533333322232151 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.1533333322232151 |

### Fold 10

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | -1.558 | -0.139 | -0.022 | 11.700 | 0.000 | 0.994 | thr=0.0005 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | -7.360 | -0.222 | -0.110 | 8.900 | 0.000 | 0.996 | thr=0.00025 regime=all uq=None cd=0 danger=0.42582450413661294 |
| D_risk_sensitive_tbguard_cd32_floor001 | 1.438 | -0.222 | 0.022 | 2.200 | 0.000 | 0.999 | thr=0.001 regime=all uq=None cd=32 danger=0.3859640884368034 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | -0.434 | -0.222 | -0.006 | 4.200 | 0.000 | 0.998 | thr=0.001 regime=all uq=None cd=0 danger=0.3859640884368034 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | -0.434 | -0.222 | -0.006 | 4.200 | 0.000 | 0.998 | thr=0.001 regime=all uq=None cd=0 danger=0.3859640884368034 |

### Fold 11

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=inf regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | 5.689 | 0.000 | 0.027 | 1.000 | 0.000 | 1.000 | thr=0.001 regime=all uq=None cd=0 danger=0.4201156816096647 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=32 danger=0.4201156816096647 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 5.689 | 0.000 | 0.027 | 1.000 | 0.000 | 1.000 | thr=0.001 regime=all uq=None cd=0 danger=0.4201156816096647 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.4201156816096647 |

### Fold 12

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0017570828254766795 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0017570828254766795 regime=all uq=None cd=0 danger=0.48512494647834126 |
| D_risk_sensitive_tbguard_cd32_floor001 | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0017570828254766795 regime=all uq=None cd=32 danger=0.48512494647834126 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0017570828254766795 regime=all uq=None cd=32 danger=0.48512494647834126 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | -0.272 | 0.055 | -0.007 | 0.500 | 0.000 | 1.000 | thr=0.0025 regime=all uq=None cd=0 danger=0.48512494647834126 |

### Fold 13

| variant | alpha | maxdd | sharpe | turnover | long | flat | val choice |
|---|---:|---:|---:|---:|---:|---:|---|
| D_risk_sensitive | -1.759 | 0.506 | -0.068 | 1.500 | 0.000 | 0.999 | thr=0.0025 regime=all uq=None cd=0 danger=None |
| D_risk_sensitive_tbguard | -1.515 | 0.383 | -0.060 | 7.500 | 0.000 | 0.995 | thr=0.0012323373983317031 regime=all uq=None cd=0 danger=0.6352122591284148 |
| D_risk_sensitive_tbguard_cd32_floor001 | 0.178 | -0.085 | 0.006 | 3.000 | 0.000 | 0.999 | thr=0.0012323373983317031 regime=all uq=None cd=32 danger=0.6352122591284148 |
| D_risk_sensitive_tbguard_auto_cd_floor001 | 0.178 | -0.085 | 0.006 | 3.000 | 0.000 | 0.999 | thr=0.0012323373983317031 regime=all uq=None cd=32 danger=0.6352122591284148 |
| D_risk_sensitive_tbguard_auto_cd_floor001_valdd | 0.178 | -0.085 | 0.006 | 3.000 | 0.000 | 0.999 | thr=0.0012323373983317031 regime=all uq=None cd=32 danger=0.6352122591284148 |

