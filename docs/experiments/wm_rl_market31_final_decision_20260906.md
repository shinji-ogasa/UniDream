# WM + learned RL: final bounded development decision

The scale100 ac_decay_dd25 recipe meets the requested minimum means and is selected for a paper demo refit. This is a small practical improvement, not a strongest-model or high-probability trend-independent claim. Only one of three reused development validation periods improves; two exactly match B&H accounting. One seed was used. No test or outer score selected this recipe.

| Recipe | Base AlphaEX (pt) | Base MaxDDdelta (pt) | 2x AlphaEX (pt) | 2x MaxDDdelta (pt) |
|---|---:|---:|---:|---:|
| R2 scale1 | -0.358725 | -0.412091 | -0.360076 | -0.411624 |
| R3 scale100 selected | +0.053549 | -0.069770 | +0.053292 | -0.069544 |

| Fold | Start-observed state | Base AlphaEX (pt) | Base MaxDDdelta (pt) | 2x AlphaEX (pt) | 2x MaxDDdelta (pt) | Base fills |
|---|---|---:|---:|---:|---:|---:|
| 6 | bear | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 |
| 7 | bull | +0.160648 | -0.209310 | +0.159876 | -0.208633 | 2 |
| 9 | sideways | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 |

Start-observed state uses only features available at the exact validation start. It does not describe the realized full-quarter trend. The periods are July19–October19 2021, October19 2021–January19 2022, and April19–July19 2022 (17:00 UTC boundaries); the intervening quarter is not included. BC-only and Actor-removed B&H produce zero fills and zero economic differences in every period. The learned AC changes intents and executes only in fold7.

The single factor100 contrast keeps the 31 features, 64-bar causal inference context, Transformer WM architecture, 42 auxiliary outputs, teacher, WM700/BC5/AC300 endpoints, seed7, bounds0.5–1.12, step0.08, deadband0.01 and next-open execution fixed. Base one-way cost0.00055/annual borrowing0.1; stress reruns the same Actor with cost0.0011/borrowing0.2 and its resulting separate inventory. Missing bars keep the nominal grid. This is fresh retraining, not only a recomputation of legacy checkpoints.

On the same128 training-only diagnostic origins, raw-return bias shrank from+26.3629bp to+1.09657bp per15m; all300 saved imagined updates now record nonzero downside. This establishes a unit/calibration improvement within training data, not future forecast accuracy. Fold6 still stays inside the execution deadband.

The earlier ML comparison (Ridge return/HGB variance with deterministic utility) met its8-period means at+4.450249pt/−5.143249pt (base),+4.247804pt/−5.055918pt (stress). It is not learned RL and is not the selected VC demo model. Its different8-period/cadence evidence is not a controlled ranking against this3-period WM+RL run.

The production refit uses the same selected training procedure on2024-09-01 through2026-09-01 T-only data. Its new weights do not inherit these historical returns. Actual export parity, registered HF/DB hashes, current inference and subsequent15-minute paper records must pass before the public cutover. This report records selection; it does not attest deployment completion.

Evidence: [selection](wm_rl_market31_selection_20260906.json), [full decision](wm_rl_market31_r3_screen_decision.json), [per-period metrics](wm_rl_market31_r3_validation_metrics.json), [run source bindings](wm_rl_market31_r3_run_manifest.json), [T-only mechanism diagnostic](wm_rl_market31_r3_T_head_diagnostic.json), [registration and primary sources](wm_rl_market31_r3_registration_20260906.md).
