# AC Candidate State-Action Critic Probe

Config: `configs/bcplan5_phase8_state_machine_s007.yaml`
Candidate actions: `0.00, 0.50, 1.00, 1.05, 1.10, 1.25`

## Summary

This probe trains candidate Q(s, position_candidate) from realized transition values. Actor updates are not performed.

## Fold 4

### Baseline Policy

| split | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | long | short | flat | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -32.89 | -0.075 | -0.62 | 0.000 | 0.177 | 0.823 | 213.05 |
| val | -98.87 | -0.060 | -1.03 | 0.000 | 0.166 | 0.834 | 25.40 |
| test | 0.50 | -0.050 | -1.15 | 0.000 | 0.181 | 0.819 | 27.30 |

### Candidate Q Metrics

| variant | split | RMSE | flat corr | row Spearman | top1 match | selected adv vs anchor | best possible adv | top-decile selected adv | selected short | selected flat | selected long | mean abs delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mse_ensemble_mean | train | 0.006394 | 0.3084 | 0.3389 | 0.257 | -0.000080 | 0.002607 | -0.000283 | 0.000 | 1.000 | 0.000 | 0.0641 |
| mse_ensemble_mean | val | 0.008231 | 0.2984 | 0.2927 | 0.225 | -0.000126 | 0.004022 | -0.000426 | 0.000 | 1.000 | 0.000 | 0.0616 |
| mse_ensemble_mean | test | 0.008786 | 0.2408 | 0.2416 | 0.181 | -0.000214 | 0.005107 | -0.000635 | 0.000 | 1.000 | 0.000 | 0.0676 |
| cql_lite_minq | train | 0.006379 | 0.3076 | 0.3532 | 0.369 | -0.000013 | 0.002607 | -0.000013 | 0.000 | 1.000 | 0.000 | 0.0490 |
| cql_lite_minq | val | 0.008260 | 0.2940 | 0.3052 | 0.315 | -0.000009 | 0.004022 | -0.000009 | 0.000 | 1.000 | 0.000 | 0.0474 |
| cql_lite_minq | test | 0.008784 | 0.2440 | 0.2611 | 0.301 | -0.000049 | 0.005107 | -0.000049 | 0.000 | 1.000 | 0.000 | 0.0500 |

### Reading

- Positive row Spearman means the critic can rank candidate actions inside a state.
- `selected adv vs anchor` should be positive before actor residual updates are allowed.
- High selected short/long concentration indicates one-sided Q bias.
