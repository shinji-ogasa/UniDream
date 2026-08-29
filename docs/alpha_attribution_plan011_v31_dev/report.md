# Plan011 v31 Alpha Attribution and Forecast Diagnostics

## Scope and provenance

- folds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12`
- selection split: `validation-only selection; test report-only; holdout reference-only`
- config SHA-256: `87e9860921c5ce1b973933fce8b17c21714545e7a92812ab9e79a9fd006fe6bf`
- data SHA-256: `bf9b1a78710782cf6be053e099e999be21c8e75795e7d14060a6a74d1f6ddd9d`
- data-contract SHA-256: `99ae4e29719b06a950835769a15b80bca749898613d41e342443a75a73c13b0c`
- commit: `3d9e3abdf920c08f3444e066783aba2cedd61836`
- costs: `{"fee_rate": 0.0003, "slippage_bps": 1.0, "spread_bps": 3.0}`

Attribution/model selection uses development folds only; holdout folds 15-23 are never used for selection. When present, the saved fold23 bundle diagnostics below are reference-only and do not alter a candidate, threshold, or test result.
Validation-only selection is recorded when validation paths are supplied. The retained historical artifact has no validation actor path, so its actor-mean row is diagnostic-only.

## Attribution summary

Constant exposure component is the actor-mean constant path; timing component is actor-sequence AlphaEx minus that constant path under the same cost contract.

| method | variant | folds | mean AlphaEx | mean MaxDDDelta | mean SharpeDelta | mean turnover |
|---|---|---:|---:|---:|---:|---:|
| actor_lag | lag_1 | 13 | +0.413pt | +0.204pt | -0.0007 | 0.4409 |
| actor_lag | lag_16 | 13 | +0.401pt | +0.209pt | -0.0013 | 0.4409 |
| actor_lag | lag_4 | 13 | +0.407pt | +0.206pt | -0.0010 | 0.4409 |
| actor_mean | constant_actor_mean | 13 | +0.391pt | +0.228pt | +0.0000 | 0.0000 |
| actor_sequence | raw_actor_path | 13 | +0.414pt | +0.204pt | -0.0007 | 0.4409 |
| bnh | exposure_1.0 | 13 | +0.000pt | +0.000pt | +0.0000 | 0.0000 |
| fixed_exposure | exposure_1.000 | 13 | +0.000pt | +0.000pt | +0.0000 | 0.0000 |
| fixed_exposure | exposure_1.005 | 13 | +0.236pt | +0.118pt | +0.0000 | 0.0000 |
| fixed_exposure | exposure_1.010 | 13 | +0.474pt | +0.235pt | +0.0000 | 0.0000 |
| fixed_exposure | exposure_1.015 | 13 | +0.712pt | +0.353pt | -0.0000 | 0.0000 |
| null_circular_shift | shift_1 | 13 | +0.412pt | +0.204pt | -0.0008 | 0.4525 |
| null_circular_shift | shift_16 | 13 | +0.405pt | +0.209pt | -0.0011 | 0.4526 |
| null_circular_shift | shift_64 | 13 | +0.324pt | +0.215pt | -0.0033 | 0.4526 |

## Constant versus timing

| fold | constant AlphaEx | timing increment | actor sequence AlphaEx | mean selection status |
|---:|---:|---:|---:|---|
| 0 | +0.072pt | +0.208pt | +0.280pt | diagnostic_only_test_mean |
| 1 | +0.167pt | +0.045pt | +0.212pt | diagnostic_only_test_mean |
| 2 | +2.852pt | +0.628pt | +3.480pt | diagnostic_only_test_mean |
| 3 | +0.866pt | +0.195pt | +1.061pt | diagnostic_only_test_mean |
| 4 | -0.330pt | -0.024pt | -0.353pt | diagnostic_only_test_mean |
| 5 | +1.650pt | -0.246pt | +1.405pt | diagnostic_only_test_mean |
| 6 | -0.249pt | +0.016pt | -0.232pt | diagnostic_only_test_mean |
| 7 | -0.072pt | -0.312pt | -0.384pt | diagnostic_only_test_mean |
| 8 | -0.380pt | -0.024pt | -0.404pt | diagnostic_only_test_mean |
| 9 | -0.087pt | -0.084pt | -0.171pt | diagnostic_only_test_mean |
| 10 | +0.077pt | +0.080pt | +0.158pt | diagnostic_only_test_mean |
| 11 | +0.510pt | -0.067pt | +0.442pt | diagnostic_only_test_mean |
| 12 | +0.002pt | -0.115pt | -0.112pt | diagnostic_only_test_mean |

## Feature coverage

Coverage is measured on each fold with `[test_start, test_end)` and is report-only. The current cache has no availability mask, so a zero cannot be distinguished from a missing/imputed value; this is retained as an explicit quality flag.

| fold | rows | funding nonzero | basis nonzero | basis_mom nonzero | basis_abs nonzero | quality/status |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 8712 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 1 | 8832 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 2 | 8808 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 3 | 8623 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 4 | 8708 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 5 | 8806 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 6 | 8832 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 7 | 8640 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 8 | 8736 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 9 | 8832 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 10 | 8832 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 11 | 8629 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 12 | 8736 | 100.00% | 100.00% | 100.00% | 100.00% | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |

### Cache diagnostic (not a model-selection input)

- rows: `208299`; observed range: `[2018-01-16 13:45:00, 2023-12-31 23:45:00]`
- Year-level derived-series rates are retained below. These diagnostics do not select, tune, or re-adjust any holdout result.

| year | rows | funding nonzero | basis nonzero | basis_mom nonzero | basis_abs nonzero |
|---:|---:|---:|---:|---:|---:|
| 2018 | 33290 | 0.00% | 0.00% | 0.00% | 0.00% |
| 2019 | 34918 | 30.92% | 2.34% | 2.33% | 2.33% |
| 2020 | 35053 | 100.00% | 100.00% | 100.00% | 100.00% |
| 2021 | 34969 | 100.00% | 100.00% | 100.00% | 100.00% |
| 2022 | 35040 | 100.00% | 100.00% | 100.00% | 100.00% |
| 2023 | 35029 | 100.00% | 100.00% | 100.00% | 100.00% |

## Predictive-head diagnostics

| fold | head | status | n | metrics / reason |
|---:|---|---|---:|---|
| 23 | wm_pred_return_h4 | ok | 8630 | `{"mae": 0.4189175998485518, "rmse": 0.6154958471648249, "sign_accuracy": 0.5044032444959444, "spearman_ic": 0.058738725800113314}` |
| 23 | wm_pred_return_h8 | ok | 8629 | `{"mae": 0.6585321917338647, "rmse": 0.9096657395204494, "sign_accuracy": 0.49009155174411867, "spearman_ic": 0.056129604484488956}` |
| 23 | wm_pred_return_h16 | ok | 8622 | `{"mae": 0.9656827391851436, "rmse": 1.3103009963132644, "sign_accuracy": 0.4846903270702853, "spearman_ic": 0.060281098823626235}` |
| 23 | wm_pred_return_h32 | ok | 8604 | `{"mae": 1.413130424099982, "rmse": 1.8748913931392872, "sign_accuracy": 0.46397024639702467, "spearman_ic": 0.07238370701784813}` |
| 23 | wm_pred_return_h64 | ok | 8554 | `{"mae": 1.6937729423067176, "rmse": 2.350382753583751, "sign_accuracy": 0.4500818330605565, "spearman_ic": -0.0025854409752260276}` |
| 23 | wm_pred_vol_h4 | ok | 8605 | `{"mae": 0.10184292287690894, "rmse": 0.1665396280546597, "sign_accuracy": null, "spearman_ic": 0.49687562029371446}` |
| 23 | wm_pred_vol_h8 | ok | 8602 | `{"mae": 0.09459198463496678, "rmse": 0.15057118734318917, "sign_accuracy": null, "spearman_ic": 0.5255839908089746}` |
| 23 | wm_pred_vol_h16 | ok | 8591 | `{"mae": 0.09140370036855865, "rmse": 0.13738949327881803, "sign_accuracy": null, "spearman_ic": 0.5016794367836913}` |
| 23 | wm_pred_vol_h32 | ok | 8583 | `{"mae": 0.09289549527993234, "rmse": 0.13061217740450742, "sign_accuracy": null, "spearman_ic": 0.4138052268627674}` |
| 23 | wm_pred_vol_h64 | ok | 8555 | `{"mae": 0.46279263738728915, "rmse": 0.4789113585573544, "sign_accuracy": null, "spearman_ic": 0.19448600298446664}` |
| 23 | wm_pred_drawdown_h4 | ok | 8626 | `{"mae": 0.24097282705093007, "rmse": 0.4285343901011403, "sign_accuracy": null, "spearman_ic": 0.1576550495107604}` |
| 23 | wm_pred_drawdown_h8 | ok | 8620 | `{"mae": 0.36750813839929836, "rmse": 0.6332034473390351, "sign_accuracy": null, "spearman_ic": 0.16953208026997288}` |
| 23 | wm_pred_drawdown_h16 | ok | 8608 | `{"mae": 0.571567597954623, "rmse": 0.9510136154872451, "sign_accuracy": null, "spearman_ic": 0.16342775849042732}` |
| 23 | wm_pred_drawdown_h32 | ok | 8591 | `{"mae": 0.8910248401956345, "rmse": 1.433907744692821, "sign_accuracy": null, "spearman_ic": 0.16719587245510284}` |
| 23 | wm_pred_drawdown_h64 | ok | 8553 | `{"mae": 1.7414195265469452, "rmse": 2.5307194662402255, "sign_accuracy": null, "spearman_ic": -0.03595613373661185}` |
| 23 | wm_pred_crash_h4 | ok | 8626 | `{"balanced_accuracy": 0.501325954858296, "brier": 0.030567087673494718, "ece": 0.014696478521455206, "mcc": 0.014481600029523476}` |
| 23 | wm_pred_crash_h8 | ok | 8625 | `{"balanced_accuracy": 0.5100686154274358, "brier": 0.07494675345289334, "ece": 0.0479179532389831, "mcc": 0.07153799269271556}` |
| 23 | wm_pred_crash_h16 | ok | 8618 | `{"balanced_accuracy": 0.5089465050186611, "brier": 0.14531562716193383, "ece": 0.09760293203847825, "mcc": 0.049591136626017954}` |
| 23 | wm_pred_crash_h32 | ok | 8603 | `{"balanced_accuracy": 0.5112467318404001, "brier": 0.22909031662022392, "ece": 0.14944329033834775, "mcc": 0.050565630867615854}` |
| 23 | wm_pred_crash_h64 | ok | 8568 | `{"balanced_accuracy": 0.5036710549731543, "brier": 0.248323038141558, "ece": 0.026377645874234468, "mcc": 0.03555872046403463}` |
| 23 | wm_pred_drawdown_excess_h4 | ok | 8602 | `{"mae": 0.030140859486907885, "rmse": 0.17883602182852879, "sign_accuracy": null, "spearman_ic": 0.10449750211737617}` |
| 23 | wm_pred_drawdown_excess_h8 | ok | 8599 | `{"mae": 0.07744676829338026, "rmse": 0.310860051182509, "sign_accuracy": null, "spearman_ic": 0.1008891240360231}` |
| 23 | wm_pred_drawdown_excess_h16 | ok | 8583 | `{"mae": 0.18877060125010203, "rmse": 0.5566129875203061, "sign_accuracy": null, "spearman_ic": 0.10594417118196123}` |
| 23 | wm_pred_drawdown_excess_h32 | ok | 8567 | `{"mae": 0.4143881628045722, "rmse": 0.9864513821657888, "sign_accuracy": null, "spearman_ic": 0.12483668165250157}` |
| 23 | wm_pred_drawdown_excess_h64 | ok | 8563 | `{"mae": 0.8395072089738064, "rmse": 1.7106441087079092, "sign_accuracy": null, "spearman_ic": -0.025229618570788065}` |
| 23 | wm_pred_position_utility_p0.5 | ok | 8526 | `{"mae": 3.0399336826650023, "rmse": 4.230123961650893, "sign_accuracy": 0.4106263194933146, "spearman_ic": 0.11024603098055154}` |
| 23 | wm_pred_position_utility_p0.7 | ok | 8526 | `{"mae": 1.6543488703138425, "rmse": 2.403267424600662, "sign_accuracy": 0.9256392212057236, "spearman_ic": 0.018785403406179308}` |
| 23 | wm_pred_position_utility_p0.85 | ok | 8526 | `{"mae": 1.054357420842999, "rmse": 1.3900408956246282, "sign_accuracy": 0.07353976073187896, "spearman_ic": -0.0705601613003438}` |
| 23 | wm_pred_position_utility_p0.94 | ok | 8526 | `{"mae": 0.25422690529695297, "rmse": 0.382799400529124, "sign_accuracy": 0.926460239268121, "spearman_ic": -0.04491818584513952}` |
| 23 | wm_pred_position_utility_p1 | ok | 8526 | `{"mae": 0.04843175634419592, "rmse": 0.0587347022331176, "sign_accuracy": 0.0, "spearman_ic": null}` |
| 23 | wm_pred_position_utility_p1.06 | ok | 8526 | `{"mae": 0.19743312085924047, "rmse": 0.26024076077776065, "sign_accuracy": 0.3647665962936899, "spearman_ic": 0.059454114345937704}` |
| 23 | wm_pred_position_utility_p1.12 | ok | 8526 | `{"mae": 0.3008021572343551, "rmse": 0.43800973502458546, "sign_accuracy": 0.7477128782547502, "spearman_ic": -0.07037509577879757}` |
| 23 | position_utility_argmax | ok | 8526 | `{"class_summary": {"confusion_matrix": [[20, 1774, 0, 5957, 0, 3, 2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [7, 165, 0, 598, 0, 0, 0]], "predicted_distribution": {"0.5": 27, "0.7": 1939, "0.85": 0, "0.94": 6555, "1": 0, "1.06": 3, "1.12": 2}, "target_all_classes_present": false, "target_distribution": {"0.5": 7756, "0.7": 0, "0.85": 0, "0.94": 0, "1": 0, "1.06": 0, "1.12": 770}}, "metrics": {"accuracy": 0.002345765892563922, "balanced_accuracy": null, "majority_class_accuracy": 0.909688013136289}}` |
| 23 | wm_pred_overweight_advantage_h4 | ok | 8635 | `{"mae": 0.12823961290237906, "rmse": 0.19052483553059943, "sign_accuracy": 0.4393746381007527, "spearman_ic": 0.04017349216407444}` |
| 23 | wm_pred_overweight_advantage_h8 | ok | 8632 | `{"mae": 0.19234859767978474, "rmse": 0.27398961638920355, "sign_accuracy": 0.4301436515291937, "spearman_ic": 0.054622132517434856}` |
| 23 | wm_pred_overweight_advantage_h16 | ok | 8623 | `{"mae": 0.2812866777993541, "rmse": 0.3954622949682793, "sign_accuracy": 0.41702423750434886, "spearman_ic": 0.0601698926406959}` |
| 23 | wm_pred_overweight_advantage_h32 | ok | 8604 | `{"mae": 0.437270231480854, "rmse": 0.5912341624051932, "sign_accuracy": 0.38819153881915386, "spearman_ic": 0.05487801666518602}` |
| 23 | wm_pred_overweight_advantage_h64 | ok | 8553 | `{"mae": 0.5024576972950697, "rmse": 0.7090819786086745, "sign_accuracy": 0.6223547293347363, "spearman_ic": -0.015802501207674857}` |
| 23 | wm_pred_recovery_h4 | ok | 8636 | `{"mae": 0.21187104086660571, "rmse": 0.256112445169723, "sign_accuracy": 0.5072950440018527, "spearman_ic": 0.07790586571887193}` |
| 23 | wm_pred_recovery_h8 | ok | 8632 | `{"mae": 0.21119503713783233, "rmse": 0.2617300957283643, "sign_accuracy": 0.4838971269694161, "spearman_ic": 0.08579848263935258}` |
| 23 | wm_pred_recovery_h16 | ok | 8623 | `{"mae": 0.20951108684135214, "rmse": 0.2627184303523474, "sign_accuracy": 0.45471413661138815, "spearman_ic": 0.07329220548117377}` |
| 23 | wm_pred_recovery_h32 | ok | 8607 | `{"mae": 0.24981488744414088, "rmse": 0.3105483121765963, "sign_accuracy": 0.38503543627280123, "spearman_ic": 0.05303853273656376}` |
| 23 | wm_pred_recovery_h64 | ok | 8569 | `{"mae": 0.19575472704169142, "rmse": 0.24008782516923638, "sign_accuracy": 0.6202590734041312, "spearman_ic": -0.020903055514626456}` |
| 23 | regime | N/A |  | `saved artifact has regime probabilities but no regime target labels` |

### Position-utility argmax audit

The target and predicted distributions use the configured action-position order; confusion rows are target classes and columns are predicted classes.

| fold | target distribution | predicted distribution | majority baseline | balanced accuracy | confusion matrix |
|---:|---|---|---:|---:|---|
| 23 | `{"0.5": 7756, "0.7": 0, "0.85": 0, "0.94": 0, "1": 0, "1.06": 0, "1.12": 770}` | `{"0.5": 27, "0.7": 1939, "0.85": 0, "0.94": 6555, "1": 0, "1.06": 3, "1.12": 2}` | 0.909688013136289 | None | `[[20,1774,0,5957,0,3,2],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[7,165,0,598,0,0,0]]` |

## Metric and leakage contract

- AlphaEx = strategy final total return minus B&H final total return; MaxDDDelta = strategy absolute MaxDD minus B&H absolute MaxDD; SharpeDelta and turnover come from the shared Backtest/action_stats implementations.
- All strategy paths use the configured costs and execution delay. B&H is the benchmark path and is not delayed by Backtest's strategy-only delay.
- Actor lags use deterministic strategy execution delay; nulls are deterministic circular shifts and are not candidates for selection.
- Fold bounds are validated as right-exclusive `[test_start, test_end)`; predictive targets mask the final unavailable future horizon.
- Saved `sample_input.npz` advantage values are standardized/clipped/scaled actor inputs; diagnostics invert the affine transform and exclude clip-boundary rows from raw-head metrics. Volatility/drawdown sign accuracy is N/A because those targets are one-sided.
- Position-utility outputs are smooth-L1/ranking regression scores per action, not logits; diagnostics use per-action regression plus an argmax decision accuracy and do not report softmax/Brier classification for that head.
- A diagnostic marked `N/A` has no computable target/label; unavailable metrics are never replaced with zero.

Machine-readable ledger: `docs/alpha_attribution_plan011_v31_dev/alpha_attribution_ledger.jsonl`
