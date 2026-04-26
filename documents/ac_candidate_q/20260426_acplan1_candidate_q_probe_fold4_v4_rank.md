# AC Candidate State-Action Critic Probe

Config: `configs/bcplan5_phase8_state_machine_s007.yaml`
Candidate actions: `0.00, 0.50, 1.00, 1.05, 1.10, 1.25`

## Summary

This probe trains candidate Q(s, position_candidate) from realized transition values. Actor updates are not performed.

## Fold 4

### Baseline Policy

| split | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | long | short | flat | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -19.60 | -0.007 | -1.72 | 0.000 | 0.164 | 0.836 | 20.60 |
| val | -75.09 | -0.014 | -1.11 | 0.000 | 0.150 | 0.850 | 2.46 |
| test | 0.89 | -0.011 | -1.58 | 0.000 | 0.165 | 0.835 | 2.65 |

### Candidate Q Metrics

| variant | split | RMSE | flat corr | row Spearman | top1 match | selected adv vs anchor | best possible adv | top-decile selected adv | selected short | selected flat | selected long | mean abs delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mse_ensemble_mean | train | 0.006405 | 0.2964 | 0.3092 | 0.051 | -0.000187 | 0.002608 | -0.000335 | 0.000 | 1.000 | 0.000 | 0.0932 |
| mse_ensemble_mean | val | 0.008276 | 0.2982 | 0.2702 | 0.038 | -0.000266 | 0.004023 | -0.000500 | 0.000 | 1.000 | 0.000 | 0.0928 |
| mse_ensemble_mean | test | 0.008801 | 0.2390 | 0.2145 | 0.018 | -0.000419 | 0.005108 | -0.000632 | 0.000 | 1.000 | 0.000 | 0.0960 |
| cql_lite_minq | train | 0.006405 | 0.2951 | 0.3270 | 0.190 | -0.000121 | 0.002608 | -0.000310 | 0.000 | 1.000 | 0.000 | 0.0760 |
| cql_lite_minq | val | 0.008257 | 0.2957 | 0.2833 | 0.162 | -0.000182 | 0.004023 | -0.000485 | 0.000 | 1.000 | 0.000 | 0.0738 |
| cql_lite_minq | test | 0.008801 | 0.2370 | 0.2298 | 0.121 | -0.000290 | 0.005108 | -0.000598 | 0.000 | 1.000 | 0.000 | 0.0795 |
| anchor_adv_mse_mean | train | 0.006397 | 0.3005 | 0.3135 | 0.081 | -0.000172 | 0.002608 | -0.000320 | 0.000 | 1.000 | 0.000 | 0.0895 |
| anchor_adv_mse_mean | val | 0.008281 | 0.2960 | 0.2733 | 0.065 | -0.000244 | 0.004023 | -0.000481 | 0.000 | 1.000 | 0.000 | 0.0884 |
| anchor_adv_mse_mean | test | 0.008793 | 0.2414 | 0.2169 | 0.033 | -0.000396 | 0.005108 | -0.000625 | 0.000 | 1.000 | 0.000 | 0.0934 |
| anchor_adv_cql_minq | train | 0.006400 | 0.3006 | 0.3240 | 0.154 | -0.000137 | 0.002608 | -0.000311 | 0.000 | 1.000 | 0.000 | 0.0804 |
| anchor_adv_cql_minq | val | 0.008246 | 0.2975 | 0.2807 | 0.127 | -0.000203 | 0.004023 | -0.000489 | 0.000 | 1.000 | 0.000 | 0.0792 |
| anchor_adv_cql_minq | test | 0.008782 | 0.2427 | 0.2269 | 0.093 | -0.000328 | 0.005108 | -0.000646 | 0.000 | 1.000 | 0.000 | 0.0839 |
| anchor_rank_ce_mean | train | 0.006567 | 0.2123 | 0.2311 | 0.119 | -0.000532 | 0.002608 | -0.001568 | 0.032 | 0.677 | 0.291 | 0.1847 |
| anchor_rank_ce_mean | val | 0.008509 | 0.1813 | 0.1994 | 0.097 | -0.000844 | 0.004023 | -0.002760 | 0.028 | 0.722 | 0.250 | 0.1725 |
| anchor_rank_ce_mean | test | 0.008925 | 0.1849 | 0.1289 | 0.116 | -0.001286 | 0.005108 | -0.003291 | 0.006 | 0.622 | 0.371 | 0.1782 |
| anchor_rank_ce_cql_minq | train | 0.006578 | 0.2187 | 0.2601 | 0.051 | -0.000349 | 0.002608 | -0.001470 | 0.035 | 0.883 | 0.082 | 0.1449 |
| anchor_rank_ce_cql_minq | val | 0.008504 | 0.1869 | 0.2241 | 0.035 | -0.000549 | 0.004023 | -0.002554 | 0.032 | 0.915 | 0.053 | 0.1358 |
| anchor_rank_ce_cql_minq | test | 0.008939 | 0.1813 | 0.1719 | 0.047 | -0.000767 | 0.005108 | -0.003060 | 0.008 | 0.873 | 0.119 | 0.1286 |

### Reading

- Positive row Spearman means the critic can rank candidate actions inside a state.
- `selected adv vs anchor` should be positive before actor residual updates are allowed.
- High selected short/long concentration indicates one-sided Q bias.
