# Route Separability Probe

Config: `configs/acplan17_det_retrain_s007_tmp.yaml`
Checkpoint dir: `checkpoints/acplan17_det_retrain_s007`
Folds: `0, 4, 5`
Label mode: `benchmark`

## Aggregate Active/No-Active Separability

| feature set | folds | test AUC mean | test AUC worst | test AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 3 | 0.516 | 0.488 | 0.464 | 0.118 | 0.015 | 0.121 | 0.191 | 0.120 |
| raw | 3 | 0.527 | 0.502 | 0.479 | 0.202 | 0.060 | 0.177 | 0.412 | 0.187 |
| wm | 3 | 0.525 | 0.512 | 0.476 | 0.161 | 0.095 | 0.135 | 0.255 | 0.146 |
| wm_context | 3 | 0.525 | 0.514 | 0.475 | 0.158 | 0.094 | 0.134 | 0.250 | 0.144 |

## Fold Detail

Thresholds are selected on validation with the configured false-active and predicted-active caps, then applied to test.

### Fold 0

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.492 | 0.566 | 0.426 | 0.106 | 0.053 | 0.072 | 0.377 | 0.437 |
| wm | 0.496 | 0.548 | 0.399 | 0.102 | 0.062 | 0.076 | 0.372 | 0.454 |
| context | 0.510 | 0.548 | 0.390 | 0.015 | 0.016 | 0.016 | 0.368 | 0.345 |
| wm_context | 0.497 | 0.546 | 0.398 | 0.098 | 0.062 | 0.075 | 0.375 | 0.446 |

### Fold 4

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.516 | 0.502 | 0.543 | 0.060 | 0.066 | 0.063 | 0.346 | 0.336 |
| wm | 0.524 | 0.516 | 0.555 | 0.095 | 0.088 | 0.092 | 0.345 | 0.582 |
| context | 0.513 | 0.513 | 0.553 | 0.201 | 0.191 | 0.196 | 0.343 | 0.490 |
| wm_context | 0.523 | 0.517 | 0.555 | 0.094 | 0.089 | 0.092 | 0.343 | 0.585 |

### Fold 5

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.495 | 0.514 | 0.469 | 0.440 | 0.412 | 0.425 | 0.336 | 0.644 |
| wm | 0.515 | 0.512 | 0.473 | 0.288 | 0.255 | 0.270 | 0.340 | 0.633 |
| context | 0.515 | 0.488 | 0.450 | 0.138 | 0.156 | 0.147 | 0.327 | 0.566 |
| wm_context | 0.516 | 0.514 | 0.473 | 0.281 | 0.250 | 0.264 | 0.339 | 0.632 |

## One-Vs-Rest Test AUC


### Fold 0

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| raw | 0.549 | NA | 0.617 |
| wm | 0.538 | NA | 0.591 |
| context | 0.523 | NA | 0.581 |
| wm_context | 0.536 | NA | 0.591 |

### Fold 4

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| raw | 0.508 | NA | 0.535 |
| wm | 0.498 | NA | 0.538 |
| context | 0.496 | NA | 0.549 |
| wm_context | 0.497 | NA | 0.540 |

### Fold 5

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| raw | 0.522 | NA | 0.525 |
| wm | 0.535 | NA | 0.526 |
| context | 0.488 | NA | 0.526 |
| wm_context | 0.534 | NA | 0.526 |
