# Wave3D constant-exposure development diagnostic

- Status: **FAIL** (promotion eligible: `False`)
- Diagnostic type: `low_frequency_constant_exposure_baseline` (low-frequency constant-exposure baseline)
- This is not a forecast-accuracy result and cannot promote a model; Wave3C forecast candidates remain a separate failed screen.

## Scope and fixed contract

- Exact WFO folds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`; each appears once. Fold 12 and folds 15+ are excluded.
- Selection: fixed grid `[0.5, 0.75, 1.0, 1.05, 1.1, 1.12]` on validation only; test is report-only.
- Benchmark: constant B&H position `1.0`.
- Execution delay: fixed `1` bar; no delay tuning.
- Each fold test window is independently costed from a flat initial position (`initial_position=None`), matching the shared per-fold backtest convention; state is not carried across folds.
- Costs: `{'spread_bps': 3.0, 'fee_rate': 0.0003, 'slippage_bps': 1.0}`.
- Returns cache rows `208299`; evaluated rows `183394`; excluded at cutoff `24905`.
- Exclusive evaluation cutoff: `2023-04-16 13:45:00`.
- Features and model artifacts were not used; this screen makes no external-feature quality claim.

## Fold and exposure results

| method | folds | median AlphaEx (pt) | positive folds | median MaxDDDelta (pt) | median turnover |
|---|---:|---:|---:|---:|---:|
| `bnh` | 12 | 0.0000 | 0 | 0.0000 | 0.0000 |
| `fixed_0p500` | 12 | -8.7851 | 5 | -12.1448 | 0.0000 |
| `fixed_0p750` | 12 | -4.4949 | 5 | -5.8409 | 0.0000 |
| `fixed_1p000` | 12 | 0.0000 | 0 | 0.0000 | 0.0000 |
| `fixed_1p050` | 12 | 0.9245 | 7 | 1.1157 | 0.0000 |
| `fixed_1p100` | 12 | 1.8577 | 7 | 2.2144 | 0.0000 |
| `fixed_1p120` | 12 | 2.2335 | 7 | 2.6493 | 0.0000 |
| `selected_constant` | 12 | 1.5644 | 6 | -4.1566 | 0.0000 |
| `previous_fold_selected_constant` | 11 | 4.0307 | 6 | -9.5789 | 0.0000 |

Selected validation exposures: `[0.5, 1.12, 1.12, 1.12, 1.12, 0.5, 1.12, 0.5, 0.5, 0.5, 0.5, 1.0]`; mode `0.5` (fraction `0.5000`); distinct `3`; switches `5`.

## Statistical diagnostics

- Selected-constant block-bootstrap additive AlphaEx CI: `{'estimate_pt': 1.7773716656953926, 'lower_pt': -66.07534937366471, 'upper_pt': 75.34197279766992, 'replicates': 200}`; sensitivity: `[{'block_length': 8, 'seed': 8, 'alpha_excess_pt': {'estimate_pt': 1.7773716656953926, 'lower_pt': -81.19073053054174, 'upper_pt': 68.4004456917868, 'replicates': 200}}, {'block_length': 16, 'seed': 9, 'alpha_excess_pt': {'estimate_pt': 1.7773716656953926, 'lower_pt': -81.16824454990878, 'upper_pt': 84.9376674121271, 'replicates': 200}}, {'block_length': 32, 'seed': 10, 'alpha_excess_pt': {'estimate_pt': 1.7773716656953926, 'lower_pt': -68.19125807316499, 'upper_pt': 81.1605718445117, 'replicates': 200}}]`.
- Selected-constant fold sign test: `{'status': 'N/A', 'passed': False, 'reason': 'selected constant alpha fold totals has 11 non-zero folds; need at least 12', 'folds': 12, 'effective_folds': 11, 'positive_folds': 6, 'negative_folds': 5, 'zero_folds': 1, 'p_value': None}`.
- DSR: `{'status': 'ok', 'passed': False, 'promotion_eligible': True, 'selected_candidate': 'selected_constant', 'n_trials': 7, 'trial_count_source': 'config', 'trial_count_warning': None, 'n_observations': 104978, 'annualization_bars_per_year': 35040.0, 'observed_sharpe_per_bar': 0.003926575659015515, 'observed_sharpe_annualized': 0.735014689693545, 'expected_max_sharpe_per_bar': 0.0001731295990358279, 'expected_max_sharpe_annualized': 0.032408085202665415, 'trial_sharpes_per_bar': {'fixed_0p500': 0.0035962711456299402, 'fixed_0p750': 0.00359627114562994, 'fixed_1p000': 0.0035962711456299402, 'fixed_1p050': 0.00359627114562994, 'fixed_1p100': 0.0035962711456299385, 'fixed_1p120': 0.003596271145629941, 'selected_constant': 0.003926575659015515}, 'variance_across_trial_sharpes': 1.558586736612591e-08, 'skewness': 0.04200595331112481, 'kurtosis_pearson': 56.60712756926511, 'deflated_sharpe_z': 1.2160922798401328, 'dsr_probability': 0.8880251144451954, 'dsr_p_value': 0.11197488555480462, 'probability_threshold': 0.95, 'formula': 'Phi((SR_hat-SR_star)*sqrt(T-1)/sqrt(1-g3*SR_hat+((g4-1)/4)*SR_hat^2)); SR_star uses Euler-weighted expected max over N trials', 'note': 'DSR is a diagnostic over fixed grid plus the adaptive selector; it is not used to select exposure'}`; fixed `n_trials=7`.
- CSCV/PBO: `{'status': 'ok', 'passed': True, 'pbo': 0.4523809523809524, 'pbo_max': 0.5, 'n_candidates': 6, 'n_subperiods': 12, 'n_combinations': 924, 'overfit_combinations': 418, 'mean_oos_rank_logit': 0.22837097836174944, 'selected_candidate_counts': {'fixed_0p500': 209, 'fixed_0p750': 0, 'fixed_1p000': 0, 'fixed_1p050': 0, 'fixed_1p100': 0, 'fixed_1p120': 715}, 'definition': 'PBO = fraction of CSCV IS winners with OOS rank below the median', 'note': 'CSCV/PBO is report-only over twelve test subperiods and never feeds validation selection'}` over twelve even test subperiods; report-only and not used for selection.
- Stress diagnostics: `{'status': 'ok', 'passed': False, 'required_kinds': ['cost', 'regime'], 'cases': [{'name': 'cost_1x', 'kind': 'cost', 'alpha_excess_pt': 1.7773716656953826, 'timing_increment_pt': 0.0, 'passed': True}, {'name': 'cost_1.5x', 'kind': 'cost', 'alpha_excess_pt': 1.8433716656953794, 'timing_increment_pt': 0.0, 'passed': True}, {'name': 'cost_2x', 'kind': 'cost', 'alpha_excess_pt': 1.9093716656953814, 'timing_increment_pt': 0.0, 'passed': True}, {'name': 'regime_positive_return', 'kind': 'regime', 'alpha_excess_pt': -1986.0060200307564, 'timing_increment_pt': 0.0, 'passed': False}, {'name': 'regime_negative_return', 'kind': 'regime', 'alpha_excess_pt': 1987.7833916964523, 'timing_increment_pt': 0.0, 'passed': True}, {'name': 'regime_high_volatility', 'kind': 'regime', 'alpha_excess_pt': 51.8735984623785, 'timing_increment_pt': 0.0, 'passed': True}], 'groups': {'cost': {'status': 'ok', 'passed': True, 'cases': 3, 'passed_cases': 3, 'pass_rate': 1.0}, 'regime': {'status': 'ok', 'passed': False, 'cases': 3, 'passed_cases': 2, 'pass_rate': 0.6666666666666666}}, 'thresholds': {'alpha_excess_floor_pt': 0.0, 'timing_increment_floor_pt': 0.0, 'minimum_pass_rate': 1.0}, 'records': {'cost': [{'name': 'cost_1x', 'rows': 104978, 'alpha_excess_pt': 1.7773716656953826}, {'name': 'cost_1.5x', 'rows': 104978, 'alpha_excess_pt': 1.8433716656953794}, {'name': 'cost_2x', 'rows': 104978, 'alpha_excess_pt': 1.9093716656953814}], 'regime': [{'name': 'regime_positive_return', 'rows': 52850, 'alpha_excess_pt': -1986.0060200307564, 'threshold_source': 'fold train+validation abs-return 75th percentile'}, {'name': 'regime_negative_return', 'rows': 52128, 'alpha_excess_pt': 1987.7833916964523, 'threshold_source': 'fold train+validation abs-return 75th percentile'}, {'name': 'regime_high_volatility', 'rows': 26397, 'alpha_excess_pt': 51.8735984623785, 'threshold_source': 'fold train+validation abs-return 75th percentile'}], 'definition': 'report-only stress diagnostics'}, 'note': 'cost paths recompute the same selected decisions; regime thresholds use train+validation only'}`; cost cases recompute 1x/1.5x/2x costs and regime thresholds use train+validation only.

## Gate decision

- Passed criteria: `['exact_development_folds', 'unique_development_folds', 'validation_selection_complete', 'selected_median_alpha_positive', 'selected_median_maxdd_delta_nonpositive', 'selection_mode_fraction_at_least_0.5', 'selection_distinct_at_most_3', 'selected_positive_folds_not_lower_than_previous', 'cscv_contract']`.
- Failed criteria: `['selected_positive_folds_at_least_8', 'selected_bootstrap_ci_lower_positive', 'selected_bootstrap_sensitivity_lower_positive', 'selected_fold_sign_test', 'selected_median_superior_to_previous', 'selected_mean_superior_to_previous', 'dsr_contract', 'stress_contract']`.
- A passing constant baseline would remain a low-frequency allocation finding, not evidence of predictive precision or a reason to advance DLinear/Transformer/RL.

## Artifacts and provenance

- Result: `/Users/sophie/Documents/UniDream/UniDream/docs/constant_exposure_plan011_dev/result.json`
- Ledger: `/Users/sophie/Documents/UniDream/UniDream/docs/constant_exposure_plan011_dev/ledger.jsonl`
- Per-bar NPZ: `/Users/sophie/Documents/UniDream/UniDream/docs/constant_exposure_plan011_dev/constant_exposure_paths.npz`
- Per-bar index: `/Users/sophie/Documents/UniDream/UniDream/docs/constant_exposure_plan011_dev/constant_exposure_paths.json`
- Config SHA256: `5c9599bdb2b5495ad525715291fba7812232b24353c4ccc16d42f1d5ad15a944`
- Data-contract SHA256: `ee9c1326477670918a5d5d1fff83be011fbd2bfe15a7fec704f7f3050e745a9c`
- Git commit at run: `f6d273b89d1df77778f9022e31fd7c87981fc05a`
