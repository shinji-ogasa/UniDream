# Transition Advantage Probe

Generated: 2026-04-25 22:15:29

## Fold Summary

| fold | train period | target short | target benchmark | target overweight | mean best adv | recovery rate | teacher recovery median |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | 2021-01-16 -> 2023-01-16 | 34.4% | 32.9% | 32.7% | 0.003225 | 41.2% | 4.0 |

## Fold 4

- Train: `2021-01-16 -> 2023-01-16`
- Teacher dist: `long=0% short=40% flat=60% mean=-0.122 switches=22794 avg_hold=3.1b turnover=2166.04`
- Candidate actions: `0.0, 0.5, 1.0, 1.25`
- Horizons: `4, 8, 16, 32`

### Best Transition Class

| class | count | rate | mean_adv | median_adv | top_decile | bottom_decile | positive_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 12618 | 18.0% | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0% |
| de_risk | 2320 | 3.3% | 0.002715 | 0.001661 | 0.009953 | 0.000117 | 100.0% |
| stay_underweight | 21778 | 31.1% | 0.007966 | 0.005170 | 0.028054 | 0.000307 | 100.0% |
| recovery | 10421 | 14.9% | 0.000206 | 0.000000 | 0.001904 | 0.000000 | 15.4% |
| overweight | 22874 | 32.7% | 0.001917 | 0.001254 | 0.006831 | 0.000090 | 100.0% |

### Candidate Action Value

| action | mean | median | top_decile | bottom_decile | positive_rate |
|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.004288 | -0.002996 | 0.017357 | -0.030428 | 33.5% |
| 0.50 | -0.002120 | -0.001486 | 0.008729 | -0.015213 | 33.8% |
| 1.00 | -0.000067 | -0.000003 | 0.000000 | -0.000320 | 0.0% |
| 1.25 | -0.001224 | -0.000825 | 0.004220 | -0.008266 | 32.5% |

### Transition Matrix

| current -> target | underweight | benchmark | overweight |
|---|---:|---:|---:|
| underweight | 21778 | 10421 | 4842 |
| benchmark | 2320 | 12618 | 18032 |
| overweight | 0 | 0 | 0 |

## Sources

- IQL: https://arxiv.org/abs/2110.06169
- AWAC: https://arxiv.org/abs/2006.09359
- TD3+BC: https://arxiv.org/abs/2106.06860
- CQL: https://arxiv.org/abs/2006.04779
