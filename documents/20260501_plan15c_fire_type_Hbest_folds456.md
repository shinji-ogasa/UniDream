# Plan15-C Fire Type / Regime Split Probe

## Setup

- config: `configs/trading_wm_control_headonly.yaml`
- folds: `4,5,6`
- horizon: `32`
- scope: diagnostic fire type split only; no inference guard, WM head, AC unlock, or config adoption.

## Policy Summary

| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl | global MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 97.3% | 11.4%/994 | -0.0110 | -53.55pt |
| Hbest | 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 97.0% | 12.1%/1065 | -0.0407 | -24.35pt |
| Hbest | 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 99.1% | 28.4%/2506 | -0.1802 | -42.55pt |

## Fire Type Summary

| run | fold | fire_type | count | rate | adv | adv+ | postDD | futureMDD | globalMDD | fire_pnl | dd_depth | trail_ret32 | vol64 | delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | mdd_inside_profitable_fire | 207 | 21.0% | +0.00046 | 1.000 | +0.00062 | 0.754 | 1.000 | +0.1647 | 0.330 | -0.02685 | 0.00636 | +0.0308 |
| Hbest | 4 | noise_fire | 88 | 8.9% | -0.00011 | 0.057 | +0.00016 | 0.000 | 0.534 | -0.2128 | 0.385 | +0.00785 | 0.00771 | +0.0188 |
| Hbest | 4 | pre_dd_danger_fire | 550 | 55.9% | -0.00034 | 0.282 | +0.00086 | 0.884 | 0.540 | -0.1206 | 0.373 | -0.01342 | 0.00567 | +0.0312 |
| Hbest | 4 | profitable_low_dd_fire | 85 | 8.6% | +0.00030 | 1.000 | +0.00009 | 0.000 | 0.918 | +0.1393 | 0.373 | -0.02127 | 0.00815 | +0.0240 |
| Hbest | 4 | recovery_fire | 26 | 2.6% | +0.00063 | 1.000 | +0.00019 | 0.000 | 0.038 | -0.0204 | 0.483 | +0.02275 | 0.00835 | +0.0139 |
| Hbest | 4 | trend_continuation_fire | 28 | 2.8% | +0.00010 | 1.000 | +0.00006 | 0.000 | 0.571 | +0.0434 | 0.418 | +0.02542 | 0.00861 | +0.0125 |
| Hbest | 5 | mdd_inside_profitable_fire | 93 | 8.9% | +0.00039 | 1.000 | +0.00085 | 0.774 | 1.000 | +0.1358 | 0.161 | -0.03366 | 0.00505 | +0.0433 |
| Hbest | 5 | noise_fire | 116 | 11.1% | +0.00020 | 0.241 | +0.00035 | 0.000 | 0.328 | -0.2006 | 0.152 | -0.01596 | 0.00601 | +0.0269 |
| Hbest | 5 | pre_dd_danger_fire | 625 | 59.7% | -0.00031 | 0.349 | +0.00070 | 0.934 | 0.274 | -0.1588 | 0.083 | -0.00546 | 0.00435 | +0.0313 |
| Hbest | 5 | profitable_low_dd_fire | 88 | 8.4% | +0.00050 | 1.000 | +0.00020 | 0.000 | 0.136 | +0.0491 | 0.069 | -0.01800 | 0.00443 | +0.0350 |
| Hbest | 5 | recovery_fire | 95 | 9.1% | +0.00032 | 1.000 | +0.00017 | 0.000 | 0.347 | +0.1084 | 0.192 | -0.00392 | 0.00659 | +0.0190 |
| Hbest | 5 | trend_continuation_fire | 30 | 2.9% | +0.00014 | 1.000 | +0.00004 | 0.000 | 0.033 | +0.0345 | 0.021 | +0.03115 | 0.00399 | +0.0156 |
| Hbest | 6 | mdd_inside_profitable_fire | 484 | 19.3% | +0.00016 | 1.000 | +0.00024 | 0.888 | 1.000 | +0.5143 | 0.247 | -0.01045 | 0.00344 | +0.0217 |
| Hbest | 6 | noise_fire | 272 | 10.9% | -0.00000 | 0.074 | +0.00011 | 0.000 | 0.768 | -0.3579 | 0.240 | -0.00127 | 0.00405 | +0.0143 |
| Hbest | 6 | pre_dd_danger_fire | 1446 | 57.7% | -0.00015 | 0.180 | +0.00036 | 0.948 | 0.603 | -0.5534 | 0.201 | -0.00505 | 0.00318 | +0.0219 |
| Hbest | 6 | profitable_low_dd_fire | 112 | 4.5% | +0.00017 | 1.000 | +0.00009 | 0.000 | 0.741 | +0.0596 | 0.143 | -0.01302 | 0.00381 | +0.0208 |
| Hbest | 6 | recovery_fire | 153 | 6.1% | +0.00007 | 1.000 | +0.00007 | 0.000 | 0.693 | +0.1151 | 0.362 | -0.00383 | 0.00419 | +0.0160 |
| Hbest | 6 | trend_continuation_fire | 39 | 1.6% | +0.00018 | 1.000 | +0.00004 | 0.000 | 0.000 | +0.0420 | 0.061 | +0.01418 | 0.00225 | +0.0144 |

## Regime Summary

| run | fold | regime | count | adv | postDD | futureMDD | globalMDD | fire_pnl |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Hbest | 4 | regime_0 | 510 | -0.00014 | +0.00066 | 0.784 | 0.586 | -0.1833 |
| Hbest | 4 | regime_1 | 208 | -0.00007 | +0.00070 | 0.543 | 0.635 | -0.0938 |
| Hbest | 4 | regime_2 | 266 | +0.00010 | +0.00055 | 0.485 | 0.808 | +0.2707 |
| Hbest | 5 | regime_0 | 523 | -0.00016 | +0.00053 | 0.719 | 0.214 | -0.0048 |
| Hbest | 5 | regime_1 | 189 | -0.00011 | +0.00057 | 0.725 | 0.270 | -0.0055 |
| Hbest | 5 | regime_2 | 335 | +0.00016 | +0.00062 | 0.427 | 0.552 | -0.0213 |
| Hbest | 6 | regime_0 | 1713 | -0.00003 | +0.00025 | 0.755 | 0.653 | -0.0785 |
| Hbest | 6 | regime_1 | 492 | -0.00005 | +0.00027 | 0.648 | 0.778 | -0.0390 |
| Hbest | 6 | regime_2 | 301 | -0.00010 | +0.00041 | 0.628 | 0.837 | -0.0628 |

## Type x Regime Summary

| run | fold | type_regime | count | adv | postDD | futureMDD | globalMDD |
|---|---:|---|---:|---:|---:|---:|---:|
| Hbest | 4 | mdd_inside_profitable_fire|regime_0 | 85 | +0.00033 | +0.00078 | 0.871 | 1.000 |
| Hbest | 4 | mdd_inside_profitable_fire|regime_1 | 53 | +0.00029 | +0.00075 | 0.755 | 1.000 |
| Hbest | 4 | mdd_inside_profitable_fire|regime_2 | 69 | +0.00074 | +0.00031 | 0.609 | 1.000 |
| Hbest | 4 | noise_fire|regime_0 | 34 | -0.00002 | +0.00008 | 0.000 | 0.412 |
| Hbest | 4 | noise_fire|regime_1 | 22 | -0.00004 | +0.00014 | 0.000 | 0.364 |
| Hbest | 4 | noise_fire|regime_2 | 32 | -0.00025 | +0.00027 | 0.000 | 0.781 |
| Hbest | 4 | pre_dd_danger_fire|regime_0 | 348 | -0.00029 | +0.00076 | 0.937 | 0.474 |
| Hbest | 4 | pre_dd_danger_fire|regime_1 | 92 | -0.00051 | +0.00104 | 0.793 | 0.587 |
| Hbest | 4 | pre_dd_danger_fire|regime_2 | 110 | -0.00035 | +0.00102 | 0.791 | 0.709 |
| Hbest | 4 | profitable_low_dd_fire|regime_0 | 29 | +0.00016 | +0.00009 | 0.000 | 0.862 |
| Hbest | 4 | profitable_low_dd_fire|regime_1 | 17 | +0.00026 | +0.00014 | 0.000 | 0.824 |
| Hbest | 4 | profitable_low_dd_fire|regime_2 | 39 | +0.00042 | +0.00008 | 0.000 | 1.000 |
| Hbest | 4 | recovery_fire|regime_0 | 5 | +0.00016 | +0.00009 | 0.000 | 0.200 |
| Hbest | 4 | recovery_fire|regime_1 | 17 | +0.00080 | +0.00023 | 0.000 | 0.000 |
| Hbest | 4 | recovery_fire|regime_2 | 4 | +0.00048 | +0.00015 | 0.000 | 0.000 |
| Hbest | 4 | trend_continuation_fire|regime_0 | 9 | +0.00002 | +0.00006 | 0.000 | 1.000 |
| Hbest | 4 | trend_continuation_fire|regime_1 | 7 | +0.00002 | +0.00004 | 0.000 | 0.429 |
| Hbest | 4 | trend_continuation_fire|regime_2 | 12 | +0.00020 | +0.00007 | 0.000 | 0.333 |
| Hbest | 5 | mdd_inside_profitable_fire|regime_0 | 30 | +0.00011 | +0.00051 | 0.767 | 1.000 |
| Hbest | 5 | mdd_inside_profitable_fire|regime_1 | 16 | +0.00024 | +0.00070 | 1.000 | 1.000 |
| Hbest | 5 | mdd_inside_profitable_fire|regime_2 | 47 | +0.00062 | +0.00112 | 0.702 | 1.000 |
| Hbest | 5 | noise_fire|regime_0 | 28 | -0.00002 | +0.00003 | 0.000 | 0.179 |
| Hbest | 5 | noise_fire|regime_1 | 12 | +0.00010 | +0.00034 | 0.000 | 0.083 |
| Hbest | 5 | noise_fire|regime_2 | 76 | +0.00029 | +0.00046 | 0.000 | 0.421 |
| Hbest | 5 | pre_dd_danger_fire|regime_0 | 374 | -0.00031 | +0.00066 | 0.944 | 0.190 |
| Hbest | 5 | pre_dd_danger_fire|regime_1 | 130 | -0.00033 | +0.00066 | 0.931 | 0.246 |
| Hbest | 5 | pre_dd_danger_fire|regime_2 | 121 | -0.00027 | +0.00085 | 0.909 | 0.562 |
| Hbest | 5 | profitable_low_dd_fire|regime_0 | 62 | +0.00044 | +0.00017 | 0.000 | 0.097 |
| Hbest | 5 | profitable_low_dd_fire|regime_1 | 16 | +0.00070 | +0.00027 | 0.000 | 0.125 |
| Hbest | 5 | profitable_low_dd_fire|regime_2 | 10 | +0.00060 | +0.00026 | 0.000 | 0.400 |
| Hbest | 5 | recovery_fire|regime_0 | 6 | +0.00003 | +0.00001 | 0.000 | 0.000 |
| Hbest | 5 | recovery_fire|regime_1 | 11 | +0.00034 | +0.00022 | 0.000 | 0.000 |
| Hbest | 5 | recovery_fire|regime_2 | 78 | +0.00035 | +0.00018 | 0.000 | 0.423 |
| Hbest | 5 | trend_continuation_fire|regime_0 | 23 | +0.00011 | +0.00003 | 0.000 | 0.000 |
| Hbest | 5 | trend_continuation_fire|regime_1 | 4 | +0.00028 | +0.00004 | 0.000 | 0.000 |
| Hbest | 5 | trend_continuation_fire|regime_2 | 3 | +0.00024 | +0.00016 | 0.000 | 0.333 |
| Hbest | 6 | mdd_inside_profitable_fire|regime_0 | 324 | +0.00015 | +0.00021 | 0.920 | 1.000 |
| Hbest | 6 | mdd_inside_profitable_fire|regime_1 | 88 | +0.00019 | +0.00025 | 0.909 | 1.000 |
| Hbest | 6 | mdd_inside_profitable_fire|regime_2 | 72 | +0.00014 | +0.00033 | 0.722 | 1.000 |
| Hbest | 6 | noise_fire|regime_0 | 141 | +0.00002 | +0.00008 | 0.000 | 0.688 |
| Hbest | 6 | noise_fire|regime_1 | 73 | -0.00003 | +0.00014 | 0.000 | 0.877 |
| Hbest | 6 | noise_fire|regime_2 | 58 | -0.00003 | +0.00017 | 0.000 | 0.828 |
| Hbest | 6 | pre_dd_danger_fire|regime_0 | 1050 | -0.00012 | +0.00032 | 0.948 | 0.558 |
| Hbest | 6 | pre_dd_danger_fire|regime_1 | 257 | -0.00019 | +0.00036 | 0.930 | 0.681 |
| Hbest | 6 | pre_dd_danger_fire|regime_2 | 139 | -0.00030 | +0.00062 | 0.986 | 0.799 |
| Hbest | 6 | profitable_low_dd_fire|regime_0 | 61 | +0.00019 | +0.00008 | 0.000 | 0.557 |
| Hbest | 6 | profitable_low_dd_fire|regime_1 | 39 | +0.00017 | +0.00011 | 0.000 | 0.949 |
| Hbest | 6 | profitable_low_dd_fire|regime_2 | 12 | +0.00003 | +0.00004 | 0.000 | 1.000 |
| Hbest | 6 | recovery_fire|regime_0 | 105 | +0.00005 | +0.00006 | 0.000 | 0.743 |
| Hbest | 6 | recovery_fire|regime_1 | 30 | +0.00010 | +0.00011 | 0.000 | 0.633 |
| Hbest | 6 | recovery_fire|regime_2 | 18 | +0.00012 | +0.00009 | 0.000 | 0.500 |
| Hbest | 6 | trend_continuation_fire|regime_0 | 32 | +0.00019 | +0.00005 | 0.000 | 0.000 |
| Hbest | 6 | trend_continuation_fire|regime_1 | 5 | +0.00015 | +0.00003 | 0.000 | 0.000 |
| Hbest | 6 | trend_continuation_fire|regime_2 | 2 | +0.00011 | +0.00001 | 0.000 | 0.000 |

## Plan16 Readiness By Type

| fire_type | allowed | enough count | adv > 0 all | adv+ >= 60% all | low postDD all | low futureMDD all | low globalMDD all | candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mdd_inside_profitable_fire | False | True | True | True | False | False | False | False |
| noise_fire | False | True | False | False | True | True | False | False |
| pre_dd_danger_fire | False | True | False | False | False | False | False | False |
| profitable_low_dd_fire | True | True | True | True | True | True | False | False |
| recovery_fire | True | True | True | True | True | True | False | False |
| trend_continuation_fire | True | True | True | True | True | True | False | False |

## Interpretation

- Plan16-ready types: `none`
- A type is Plan16-ready only if it has enough samples, positive fire_advantage, low post-fire DD contribution, low future-MDD overlap, and low global-MDD overlap in every evaluated fold.
- If no type passes, do not add an inference guard yet; continue feature/type design.
