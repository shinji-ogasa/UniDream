# Plan011 Wave3C causal-context forecast tournament

## Scope and fixed contracts

- development folds only: `[0, 2, 8]` (exactly once each)
- seed: `7`; horizons: `[4, 16, 64]`
- causal context lags: `[1, 4, 16, 64]`; rolling windows: `[4, 16, 64, 256]`; statistics: `['mean', 'std', 'slope']`
- commit: `1c1cfd8d727683ab3cfabc43c01b4d9de8446df4`
- config SHA-256: `5c9599bdb2b5495ad525715291fba7812232b24353c4ccc16d42f1d5ad15a944`
- data SHA-256: `027b476f209d41f477623d92cd3e2f5061976840fd682874588fdcebdeab8704`; source SHA-256: `088387389249a37e2569f005d975e55c60ad628bd31f1406f0014fbf0cd183cd`
- fixed operational execution delay: `1` bar; sensitivity lags: `[1, 16]`; deterministic null shifts: `[1, 16, 64]`

Feature rows are pre-shifted and context uses only prior rows. Direct targets exclude the current reward and use `t+1..t+h`; therefore the primary economic contract applies a decision at `t` to `returns[t+1]` with delay 1. Timestamp gaps invalidate context/target windows but do not delete returns; ineligible policy rows emit the selected constant baseline.

Candidates are fixed before execution: Ridge and HistGradientBoosting direct return/risk regressors over causal lag/rolling context, plus a train-quantile future-downside classifier. Validation alone selects horizon, validation scales, constant exposure, threshold, overlay magnitude, hysteresis, and minimum hold. Development test is report-only and is used only as the explicitly labeled development tournament screen.

## Successive-halving gate

A candidate must have exact folds, stable forecast quality on at least two of three folds, positive median AlphaEx, positive median dynamic-minus-constant timing increment, and must beat the same-baseline constant, lag16, and shift64 temporal-destruction null by the fixed 0.02pt margin on at least two folds. Lag1, shift1, and shift16 remain robustness diagnostics. The median MaxDDDelta must be no more than 0.05pt worse than the constant and dynamic turnover must be at most 6.5. Full17 can be measured but is never promotion-eligible because the cache has no availability mask.

| feature set | candidate | folds | median quality | quality+ folds | median AlphaEx | Δconstant | Δlag16 | Δshift64 | median timing increment | median MaxDDDelta | median turnover | status |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| full17 | context_downside_classifier | 3 | +0.6264 | 3 | +19.7149pt | -0.4526pt; 1/+33% | +0.0086pt; 1/+33% | +0.0123pt; 1/+33% | -0.4526pt | -8.3131pt | +2.9912 | blocked_by_data_quality |
| full17 | context_histgb_risk_adjusted | 3 | +0.0611 | 3 | +19.7810pt | -0.3865pt; 1/+33% | -0.2289pt; 1/+33% | -0.3328pt; 1/+33% | -0.3865pt | -8.0649pt | +5.4677 | blocked_by_data_quality |
| full17 | context_ridge_risk_adjusted | 3 | +0.0537 | 3 | +20.0761pt | -0.0914pt; 1/+33% | +0.6513pt; 2/+67% | +3.6757pt; 2/+67% | -0.0914pt | -8.3844pt | +9.6829 | blocked_by_data_quality |
| ohlcv13 | context_downside_classifier | 3 | +0.6453 | 3 | +20.0930pt | -0.0745pt; 1/+33% | +0.5041pt; 2/+67% | -0.4351pt; 1/+33% | -0.0745pt | -8.3131pt | +3.1306 | fail |
| ohlcv13 | context_histgb_risk_adjusted | 3 | +0.0295 | 2 | +18.4964pt | -1.6711pt; 1/+33% | +1.5364pt; 2/+67% | +0.4656pt; 2/+67% | -1.6711pt | -8.3131pt | +8.9161 | fail |
| ohlcv13 | context_ridge_risk_adjusted | 3 | +0.0331 | 2 | +20.0591pt | -0.1084pt; 1/+33% | +0.5633pt; 2/+67% | +0.1740pt; 2/+67% | -0.1084pt | -8.2378pt | +10.0651 | fail |

Next-wave candidates (OHLCV13 only; development tournament screen): `[]`
Formal gate result: `all candidates failed`

## Selected per-fold paths

| fold | feature set | candidate | baseline | horizon | constant AlphaEx | timing increment | dynamic AlphaEx | MaxDDDelta | turnover |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | full17 | context_downside_classifier | +0.500 | 16 | -15.7590pt | +0.1678pt | -15.5912pt | -8.3131pt | +2.9912 |
| 0 | full17 | context_histgb_risk_adjusted | +0.500 | 64 | -15.7590pt | +2.6423pt | -13.1166pt | -8.0649pt | +4.9807 |
| 0 | full17 | context_ridge_risk_adjusted | +0.500 | 16 | -15.7590pt | +2.1589pt | -13.6001pt | -8.3844pt | +9.6829 |
| 0 | ohlcv13 | context_downside_classifier | +0.500 | 16 | -15.7590pt | +0.0384pt | -15.7206pt | -8.3131pt | +2.1526 |
| 0 | ohlcv13 | context_histgb_risk_adjusted | +0.500 | 4 | -15.7590pt | +0.0322pt | -15.7268pt | -8.3131pt | +8.9161 |
| 0 | ohlcv13 | context_ridge_risk_adjusted | +0.500 | 16 | -15.7590pt | +0.4646pt | -15.2944pt | -8.2378pt | +10.0651 |
| 2 | full17 | context_downside_classifier | +1.120 | 64 | +50.4569pt | -17.3641pt | +33.0928pt | +1.2532pt | +5.4971 |
| 2 | full17 | context_histgb_risk_adjusted | +1.120 | 16 | +50.4569pt | -25.1415pt | +25.3155pt | +1.6869pt | +10.5825 |
| 2 | full17 | context_ridge_risk_adjusted | +1.120 | 16 | +50.4569pt | -17.9095pt | +32.5474pt | +1.4512pt | +11.0987 |
| 2 | ohlcv13 | context_downside_classifier | +1.120 | 4 | +50.4569pt | -17.0090pt | +33.4479pt | +0.8888pt | +5.1431 |
| 2 | ohlcv13 | context_histgb_risk_adjusted | +1.120 | 16 | +50.4569pt | -7.1432pt | +43.3137pt | +2.0052pt | +10.9789 |
| 2 | ohlcv13 | context_ridge_risk_adjusted | +1.120 | 64 | +50.4569pt | -24.9346pt | +25.5224pt | +0.4728pt | +10.7358 |
| 8 | full17 | context_downside_classifier | +0.500 | 64 | +20.1675pt | -0.4526pt | +19.7149pt | -22.5117pt | +2.2959 |
| 8 | full17 | context_histgb_risk_adjusted | +0.500 | 64 | +20.1675pt | -0.3865pt | +19.7810pt | -22.1483pt | +5.4677 |
| 8 | full17 | context_ridge_risk_adjusted | +0.500 | 64 | +20.1675pt | -0.0914pt | +20.0761pt | -22.8106pt | +1.9980 |
| 8 | ohlcv13 | context_downside_classifier | +0.500 | 4 | +20.1675pt | -0.0745pt | +20.0930pt | -22.8474pt | +3.1306 |
| 8 | ohlcv13 | context_histgb_risk_adjusted | +0.500 | 64 | +20.1675pt | -1.6711pt | +18.4964pt | -21.2732pt | +6.1651 |
| 8 | ohlcv13 | context_ridge_risk_adjusted | +0.500 | 64 | +20.1675pt | -0.1084pt | +20.0591pt | -22.6797pt | +1.6635 |

## Forecast and downside diagnostics

Regression metrics are MAE, RMSE, Spearman IC, and return sign accuracy. Risk sign accuracy is N/A because realized risk is one-sided. The classifier reports AUC, Brier, precision, and recall where the split has finite labels; AUC is N/A for one-class splits with the reason preserved in the ledger.

| fold | feature set | candidate | split | horizon | target | quality | MAE | RMSE | sign/AUC | Brier | precision | recall | reason |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | full17 | context_downside_classifier | validation | 4 | downside_event | +0.6302 | N/A | N/A | +0.6302 | +0.2020 | +0.0000 | +0.0000 |  |
| 0 | full17 | context_downside_classifier | validation | 16 | downside_event | +0.6527 | N/A | N/A | +0.6527 | +0.2011 | +0.7368 | +0.0115 |  |
| 0 | full17 | context_downside_classifier | validation | 64 | downside_event | +0.6341 | N/A | N/A | +0.6341 | +0.2071 | +0.0000 | +0.0000 |  |
| 0 | full17 | context_downside_classifier | development_test | 4 | downside_event | +0.6380 | N/A | N/A | +0.6380 | +0.1420 | +0.0000 | +0.0000 |  |
| 0 | full17 | context_downside_classifier | development_test | 16 | downside_event | +0.6264 | N/A | N/A | +0.6264 | +0.1425 | +0.0000 | +0.0000 |  |
| 0 | full17 | context_downside_classifier | development_test | 64 | downside_event | +0.6043 | N/A | N/A | +0.6043 | +0.1320 | +0.0000 | +0.0000 |  |
| 0 | full17 | context_histgb_risk_adjusted | validation | 4 | return | +0.0292 | +0.0059 | +0.0123 | +0.5143 | N/A | N/A | N/A |  |
| 0 | full17 | context_histgb_risk_adjusted | validation | 16 | return | +0.0723 | +0.0120 | +0.0231 | +0.5440 | N/A | N/A | N/A |  |
| 0 | full17 | context_histgb_risk_adjusted | validation | 64 | return | +0.0831 | +0.0254 | +0.0445 | +0.5074 | N/A | N/A | N/A |  |
| 0 | full17 | context_histgb_risk_adjusted | development_test | 4 | return | +0.0463 | +0.0035 | +0.0061 | +0.5184 | N/A | N/A | N/A |  |
| 0 | full17 | context_histgb_risk_adjusted | development_test | 16 | return | +0.0917 | +0.0071 | +0.0120 | +0.5232 | N/A | N/A | N/A |  |
| 0 | full17 | context_histgb_risk_adjusted | development_test | 64 | return | +0.0626 | +0.0158 | +0.0242 | +0.5405 | N/A | N/A | N/A |  |
| 0 | full17 | context_histgb_risk_adjusted | validation | 4 | risk | +0.4130 | +0.0020 | +0.0048 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_histgb_risk_adjusted | validation | 16 | risk | +0.4125 | +0.0019 | +0.0045 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_histgb_risk_adjusted | validation | 64 | risk | +0.4251 | +0.0019 | +0.0044 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_histgb_risk_adjusted | development_test | 4 | risk | +0.4207 | +0.0013 | +0.0022 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_histgb_risk_adjusted | development_test | 16 | risk | +0.4225 | +0.0013 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_histgb_risk_adjusted | development_test | 64 | risk | +0.4160 | +0.0012 | +0.0017 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_ridge_risk_adjusted | validation | 4 | return | +0.0012 | +0.0066 | +0.0127 | +0.4993 | N/A | N/A | N/A |  |
| 0 | full17 | context_ridge_risk_adjusted | validation | 16 | return | +0.0121 | +0.0157 | +0.0252 | +0.4940 | N/A | N/A | N/A |  |
| 0 | full17 | context_ridge_risk_adjusted | validation | 64 | return | -0.0286 | +0.0438 | +0.0613 | +0.4880 | N/A | N/A | N/A |  |
| 0 | full17 | context_ridge_risk_adjusted | development_test | 4 | return | +0.0110 | +0.0044 | +0.0067 | +0.5094 | N/A | N/A | N/A |  |
| 0 | full17 | context_ridge_risk_adjusted | development_test | 16 | return | +0.0080 | +0.0123 | +0.0157 | +0.5215 | N/A | N/A | N/A |  |
| 0 | full17 | context_ridge_risk_adjusted | development_test | 64 | return | -0.0364 | +0.0381 | +0.0439 | +0.5308 | N/A | N/A | N/A |  |
| 0 | full17 | context_ridge_risk_adjusted | validation | 4 | risk | +0.3609 | +0.0020 | +0.0042 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_ridge_risk_adjusted | validation | 16 | risk | +0.4562 | +0.0019 | +0.0039 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_ridge_risk_adjusted | validation | 64 | risk | +0.6170 | +0.0019 | +0.0035 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_ridge_risk_adjusted | development_test | 4 | risk | +0.1787 | +0.0012 | +0.0023 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_ridge_risk_adjusted | development_test | 16 | risk | +0.1299 | +0.0013 | +0.0021 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | full17 | context_ridge_risk_adjusted | development_test | 64 | risk | +0.1300 | +0.0015 | +0.0020 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_downside_classifier | validation | 4 | downside_event | +0.6321 | N/A | N/A | +0.6321 | +0.2017 | +0.0000 | +0.0000 |  |
| 0 | ohlcv13 | context_downside_classifier | validation | 16 | downside_event | +0.6558 | N/A | N/A | +0.6558 | +0.2002 | +0.7447 | +0.0144 |  |
| 0 | ohlcv13 | context_downside_classifier | validation | 64 | downside_event | +0.6144 | N/A | N/A | +0.6144 | +0.2046 | +1.0000 | +0.0030 |  |
| 0 | ohlcv13 | context_downside_classifier | development_test | 4 | downside_event | +0.6380 | N/A | N/A | +0.6380 | +0.1420 | +0.0000 | +0.0000 |  |
| 0 | ohlcv13 | context_downside_classifier | development_test | 16 | downside_event | +0.6226 | N/A | N/A | +0.6226 | +0.1429 | +0.0000 | +0.0000 |  |
| 0 | ohlcv13 | context_downside_classifier | development_test | 64 | downside_event | +0.6126 | N/A | N/A | +0.6126 | +0.1320 | +0.0000 | +0.0000 |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | validation | 4 | return | +0.0292 | +0.0059 | +0.0123 | +0.5143 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | validation | 16 | return | +0.0813 | +0.0120 | +0.0231 | +0.5388 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | validation | 64 | return | +0.0467 | +0.0253 | +0.0445 | +0.5320 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | development_test | 4 | return | +0.0463 | +0.0035 | +0.0061 | +0.5184 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | development_test | 16 | return | +0.0937 | +0.0071 | +0.0120 | +0.5343 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | development_test | 64 | return | +0.0283 | +0.0158 | +0.0243 | +0.5099 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_histgb_risk_adjusted | validation | 4 | risk | +0.4197 | +0.0020 | +0.0048 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_histgb_risk_adjusted | validation | 16 | risk | +0.3989 | +0.0020 | +0.0045 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_histgb_risk_adjusted | validation | 64 | risk | +0.4384 | +0.0019 | +0.0043 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_histgb_risk_adjusted | development_test | 4 | risk | +0.4220 | +0.0013 | +0.0022 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_histgb_risk_adjusted | development_test | 16 | risk | +0.4260 | +0.0013 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_histgb_risk_adjusted | development_test | 64 | risk | +0.4127 | +0.0012 | +0.0017 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_ridge_risk_adjusted | validation | 4 | return | +0.0230 | +0.0060 | +0.0123 | +0.5073 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_ridge_risk_adjusted | validation | 16 | return | +0.0735 | +0.0123 | +0.0229 | +0.5213 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_ridge_risk_adjusted | validation | 64 | return | +0.0216 | +0.0271 | +0.0467 | +0.5143 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_ridge_risk_adjusted | development_test | 4 | return | +0.0262 | +0.0036 | +0.0062 | +0.5105 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_ridge_risk_adjusted | development_test | 16 | return | +0.0331 | +0.0075 | +0.0122 | +0.5039 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_ridge_risk_adjusted | development_test | 64 | return | +0.0116 | +0.0168 | +0.0250 | +0.4998 | N/A | N/A | N/A |  |
| 0 | ohlcv13 | context_ridge_risk_adjusted | validation | 4 | risk | +0.3770 | +0.0020 | +0.0042 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_ridge_risk_adjusted | validation | 16 | risk | +0.3856 | +0.0019 | +0.0039 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_ridge_risk_adjusted | validation | 64 | risk | +0.3855 | +0.0020 | +0.0037 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_ridge_risk_adjusted | development_test | 4 | risk | +0.1905 | +0.0013 | +0.0023 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_ridge_risk_adjusted | development_test | 16 | risk | +0.1635 | +0.0013 | +0.0020 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 0 | ohlcv13 | context_ridge_risk_adjusted | development_test | 64 | risk | +0.1559 | +0.0012 | +0.0018 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_downside_classifier | validation | 4 | downside_event | +0.6866 | N/A | N/A | +0.6866 | +0.1473 | +0.5349 | +0.0141 |  |
| 2 | full17 | context_downside_classifier | validation | 16 | downside_event | +0.6937 | N/A | N/A | +0.6937 | +0.1430 | +0.4111 | +0.0467 |  |
| 2 | full17 | context_downside_classifier | validation | 64 | downside_event | +0.6310 | N/A | N/A | +0.6310 | +0.1368 | +0.3990 | +0.0591 |  |
| 2 | full17 | context_downside_classifier | development_test | 4 | downside_event | +0.6293 | N/A | N/A | +0.6293 | +0.2175 | +0.5941 | +0.0213 |  |
| 2 | full17 | context_downside_classifier | development_test | 16 | downside_event | +0.6287 | N/A | N/A | +0.6287 | +0.2160 | +0.4985 | +0.0598 |  |
| 2 | full17 | context_downside_classifier | development_test | 64 | downside_event | +0.5799 | N/A | N/A | +0.5799 | +0.2093 | +0.4294 | +0.0629 |  |
| 2 | full17 | context_histgb_risk_adjusted | validation | 4 | return | +0.0190 | +0.0030 | +0.0049 | +0.5101 | N/A | N/A | N/A |  |
| 2 | full17 | context_histgb_risk_adjusted | validation | 16 | return | +0.0469 | +0.0062 | +0.0094 | +0.5324 | N/A | N/A | N/A |  |
| 2 | full17 | context_histgb_risk_adjusted | validation | 64 | return | -0.0168 | +0.0160 | +0.0231 | +0.5010 | N/A | N/A | N/A |  |
| 2 | full17 | context_histgb_risk_adjusted | development_test | 4 | return | +0.0261 | +0.0060 | +0.0093 | +0.5246 | N/A | N/A | N/A |  |
| 2 | full17 | context_histgb_risk_adjusted | development_test | 16 | return | +0.0190 | +0.0122 | +0.0181 | +0.5060 | N/A | N/A | N/A |  |
| 2 | full17 | context_histgb_risk_adjusted | development_test | 64 | return | +0.0080 | +0.0263 | +0.0378 | +0.5055 | N/A | N/A | N/A |  |
| 2 | full17 | context_histgb_risk_adjusted | validation | 4 | risk | +0.5715 | +0.0013 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_histgb_risk_adjusted | validation | 16 | risk | +0.6119 | +0.0015 | +0.0020 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_histgb_risk_adjusted | validation | 64 | risk | +0.5656 | +0.0017 | +0.0024 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_histgb_risk_adjusted | development_test | 4 | risk | +0.5170 | +0.0018 | +0.0028 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_histgb_risk_adjusted | development_test | 16 | risk | +0.5381 | +0.0018 | +0.0026 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_histgb_risk_adjusted | development_test | 64 | risk | +0.5001 | +0.0018 | +0.0025 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_ridge_risk_adjusted | validation | 4 | return | +0.0035 | +0.0042 | +0.0063 | +0.4960 | N/A | N/A | N/A |  |
| 2 | full17 | context_ridge_risk_adjusted | validation | 16 | return | +0.0311 | +0.0072 | +0.0104 | +0.4956 | N/A | N/A | N/A |  |
| 2 | full17 | context_ridge_risk_adjusted | validation | 64 | return | +0.0415 | +0.0153 | +0.0211 | +0.4895 | N/A | N/A | N/A |  |
| 2 | full17 | context_ridge_risk_adjusted | development_test | 4 | return | +0.0007 | +0.0075 | +0.0110 | +0.4832 | N/A | N/A | N/A |  |
| 2 | full17 | context_ridge_risk_adjusted | development_test | 16 | return | +0.0646 | +0.0134 | +0.0191 | +0.4777 | N/A | N/A | N/A |  |
| 2 | full17 | context_ridge_risk_adjusted | development_test | 64 | return | +0.0983 | +0.0291 | +0.0404 | +0.5090 | N/A | N/A | N/A |  |
| 2 | full17 | context_ridge_risk_adjusted | validation | 4 | risk | +0.2127 | +0.0020 | +0.0026 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_ridge_risk_adjusted | validation | 16 | risk | +0.2039 | +0.0023 | +0.0028 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_ridge_risk_adjusted | validation | 64 | risk | +0.0820 | +0.0027 | +0.0031 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_ridge_risk_adjusted | development_test | 4 | risk | +0.4625 | +0.0026 | +0.0034 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_ridge_risk_adjusted | development_test | 16 | risk | +0.5008 | +0.0028 | +0.0035 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | full17 | context_ridge_risk_adjusted | development_test | 64 | risk | +0.4867 | +0.0034 | +0.0041 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_downside_classifier | validation | 4 | downside_event | +0.6887 | N/A | N/A | +0.6887 | +0.1465 | +0.4375 | +0.0043 |  |
| 2 | ohlcv13 | context_downside_classifier | validation | 16 | downside_event | +0.7040 | N/A | N/A | +0.7040 | +0.1416 | +0.5644 | +0.0720 |  |
| 2 | ohlcv13 | context_downside_classifier | validation | 64 | downside_event | +0.6620 | N/A | N/A | +0.6620 | +0.1314 | +0.5263 | +0.0598 |  |
| 2 | ohlcv13 | context_downside_classifier | development_test | 4 | downside_event | +0.6453 | N/A | N/A | +0.6453 | +0.2153 | +0.6275 | +0.0227 |  |
| 2 | ohlcv13 | context_downside_classifier | development_test | 16 | downside_event | +0.6714 | N/A | N/A | +0.6714 | +0.2099 | +0.6426 | +0.1346 |  |
| 2 | ohlcv13 | context_downside_classifier | development_test | 64 | downside_event | +0.6643 | N/A | N/A | +0.6643 | +0.1967 | +0.6090 | +0.1227 |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | validation | 4 | return | +0.0124 | +0.0030 | +0.0049 | +0.5088 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | validation | 16 | return | +0.0375 | +0.0061 | +0.0092 | +0.5233 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | validation | 64 | return | -0.0286 | +0.0135 | +0.0191 | +0.5189 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | development_test | 4 | return | +0.0271 | +0.0060 | +0.0093 | +0.5198 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | development_test | 16 | return | +0.0295 | +0.0119 | +0.0179 | +0.5199 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | development_test | 64 | return | -0.0438 | +0.0256 | +0.0367 | +0.4999 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_histgb_risk_adjusted | validation | 4 | risk | +0.5715 | +0.0012 | +0.0018 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_histgb_risk_adjusted | validation | 16 | risk | +0.6274 | +0.0013 | +0.0016 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_histgb_risk_adjusted | validation | 64 | risk | +0.6236 | +0.0012 | +0.0015 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_histgb_risk_adjusted | development_test | 4 | risk | +0.5815 | +0.0017 | +0.0028 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_histgb_risk_adjusted | development_test | 16 | risk | +0.6305 | +0.0015 | +0.0024 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_histgb_risk_adjusted | development_test | 64 | risk | +0.6433 | +0.0015 | +0.0021 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_ridge_risk_adjusted | validation | 4 | return | -0.0138 | +0.0033 | +0.0052 | +0.5005 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_ridge_risk_adjusted | validation | 16 | return | +0.0171 | +0.0066 | +0.0099 | +0.5116 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_ridge_risk_adjusted | validation | 64 | return | -0.0115 | +0.0137 | +0.0196 | +0.5339 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_ridge_risk_adjusted | development_test | 4 | return | +0.0237 | +0.0062 | +0.0095 | +0.5034 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_ridge_risk_adjusted | development_test | 16 | return | +0.0568 | +0.0122 | +0.0180 | +0.5297 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_ridge_risk_adjusted | development_test | 64 | return | +0.0711 | +0.0259 | +0.0366 | +0.5428 | N/A | N/A | N/A |  |
| 2 | ohlcv13 | context_ridge_risk_adjusted | validation | 4 | risk | +0.5443 | +0.0013 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_ridge_risk_adjusted | validation | 16 | risk | +0.6119 | +0.0012 | +0.0017 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_ridge_risk_adjusted | validation | 64 | risk | +0.5781 | +0.0012 | +0.0016 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_ridge_risk_adjusted | development_test | 4 | risk | +0.5852 | +0.0021 | +0.0029 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_ridge_risk_adjusted | development_test | 16 | risk | +0.6489 | +0.0019 | +0.0026 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 2 | ohlcv13 | context_ridge_risk_adjusted | development_test | 64 | risk | +0.6440 | +0.0019 | +0.0026 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_downside_classifier | validation | 4 | downside_event | +0.6282 | N/A | N/A | +0.6282 | +0.1691 | +0.0000 | +0.0000 |  |
| 8 | full17 | context_downside_classifier | validation | 16 | downside_event | +0.6145 | N/A | N/A | +0.6145 | +0.1766 | +0.0000 | +0.0000 |  |
| 8 | full17 | context_downside_classifier | validation | 64 | downside_event | +0.5552 | N/A | N/A | +0.5552 | +0.1736 | +0.3600 | +0.0047 |  |
| 8 | full17 | context_downside_classifier | development_test | 4 | downside_event | +0.6660 | N/A | N/A | +0.6660 | +0.1828 | +1.0000 | +0.0034 |  |
| 8 | full17 | context_downside_classifier | development_test | 16 | downside_event | +0.6332 | N/A | N/A | +0.6332 | +0.1940 | +0.5755 | +0.0243 |  |
| 8 | full17 | context_downside_classifier | development_test | 64 | downside_event | +0.6405 | N/A | N/A | +0.6405 | +0.2047 | +0.6379 | +0.0832 |  |
| 8 | full17 | context_histgb_risk_adjusted | validation | 4 | return | +0.0352 | +0.0047 | +0.0071 | +0.5108 | N/A | N/A | N/A |  |
| 8 | full17 | context_histgb_risk_adjusted | validation | 16 | return | +0.0593 | +0.0095 | +0.0141 | +0.5103 | N/A | N/A | N/A |  |
| 8 | full17 | context_histgb_risk_adjusted | validation | 64 | return | +0.1308 | +0.0205 | +0.0292 | +0.5258 | N/A | N/A | N/A |  |
| 8 | full17 | context_histgb_risk_adjusted | development_test | 4 | return | +0.0004 | +0.0055 | +0.0086 | +0.4959 | N/A | N/A | N/A |  |
| 8 | full17 | context_histgb_risk_adjusted | development_test | 16 | return | +0.0463 | +0.0112 | +0.0168 | +0.4982 | N/A | N/A | N/A |  |
| 8 | full17 | context_histgb_risk_adjusted | development_test | 64 | return | +0.0611 | +0.0239 | +0.0338 | +0.4869 | N/A | N/A | N/A |  |
| 8 | full17 | context_histgb_risk_adjusted | validation | 4 | risk | +0.4425 | +0.0014 | +0.0020 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_histgb_risk_adjusted | validation | 16 | risk | +0.4492 | +0.0012 | +0.0016 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_histgb_risk_adjusted | validation | 64 | risk | +0.4900 | +0.0010 | +0.0013 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_histgb_risk_adjusted | development_test | 4 | risk | +0.5215 | +0.0015 | +0.0024 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_histgb_risk_adjusted | development_test | 16 | risk | +0.4862 | +0.0014 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_histgb_risk_adjusted | development_test | 64 | risk | +0.5035 | +0.0012 | +0.0017 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_ridge_risk_adjusted | validation | 4 | return | +0.0263 | +0.0050 | +0.0074 | +0.5155 | N/A | N/A | N/A |  |
| 8 | full17 | context_ridge_risk_adjusted | validation | 16 | return | +0.0214 | +0.0103 | +0.0148 | +0.5070 | N/A | N/A | N/A |  |
| 8 | full17 | context_ridge_risk_adjusted | validation | 64 | return | +0.0634 | +0.0220 | +0.0306 | +0.5149 | N/A | N/A | N/A |  |
| 8 | full17 | context_ridge_risk_adjusted | development_test | 4 | return | +0.0120 | +0.0058 | +0.0089 | +0.5060 | N/A | N/A | N/A |  |
| 8 | full17 | context_ridge_risk_adjusted | development_test | 16 | return | +0.0175 | +0.0119 | +0.0174 | +0.5111 | N/A | N/A | N/A |  |
| 8 | full17 | context_ridge_risk_adjusted | development_test | 64 | return | +0.0537 | +0.0244 | +0.0335 | +0.5324 | N/A | N/A | N/A |  |
| 8 | full17 | context_ridge_risk_adjusted | validation | 4 | risk | +0.4005 | +0.0015 | +0.0022 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_ridge_risk_adjusted | validation | 16 | risk | +0.4610 | +0.0013 | +0.0018 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_ridge_risk_adjusted | validation | 64 | risk | +0.4951 | +0.0011 | +0.0015 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_ridge_risk_adjusted | development_test | 4 | risk | +0.4502 | +0.0017 | +0.0026 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_ridge_risk_adjusted | development_test | 16 | risk | +0.4667 | +0.0016 | +0.0022 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | full17 | context_ridge_risk_adjusted | development_test | 64 | risk | +0.4684 | +0.0014 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_downside_classifier | validation | 4 | downside_event | +0.6403 | N/A | N/A | +0.6403 | +0.1678 | +0.0000 | +0.0000 |  |
| 8 | ohlcv13 | context_downside_classifier | validation | 16 | downside_event | +0.6156 | N/A | N/A | +0.6156 | +0.1763 | +0.0000 | +0.0000 |  |
| 8 | ohlcv13 | context_downside_classifier | validation | 64 | downside_event | +0.5653 | N/A | N/A | +0.5653 | +0.1745 | +0.2000 | +0.0021 |  |
| 8 | ohlcv13 | context_downside_classifier | development_test | 4 | downside_event | +0.6785 | N/A | N/A | +0.6785 | +0.1815 | +0.0000 | +0.0000 |  |
| 8 | ohlcv13 | context_downside_classifier | development_test | 16 | downside_event | +0.6568 | N/A | N/A | +0.6568 | +0.1914 | +0.7907 | +0.0271 |  |
| 8 | ohlcv13 | context_downside_classifier | development_test | 64 | downside_event | +0.6110 | N/A | N/A | +0.6110 | +0.2079 | +0.7834 | +0.0618 |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | validation | 4 | return | +0.0302 | +0.0047 | +0.0071 | +0.5117 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | validation | 16 | return | +0.0282 | +0.0095 | +0.0142 | +0.4988 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | validation | 64 | return | -0.0205 | +0.0211 | +0.0312 | +0.5065 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | development_test | 4 | return | -0.0143 | +0.0055 | +0.0086 | +0.4940 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | development_test | 16 | return | -0.0040 | +0.0112 | +0.0168 | +0.4778 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | development_test | 64 | return | -0.0280 | +0.0235 | +0.0336 | +0.4970 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_histgb_risk_adjusted | validation | 4 | risk | +0.4377 | +0.0014 | +0.0020 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_histgb_risk_adjusted | validation | 16 | risk | +0.4416 | +0.0012 | +0.0016 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_histgb_risk_adjusted | validation | 64 | risk | +0.4810 | +0.0010 | +0.0012 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_histgb_risk_adjusted | development_test | 4 | risk | +0.5580 | +0.0015 | +0.0023 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_histgb_risk_adjusted | development_test | 16 | risk | +0.5383 | +0.0013 | +0.0019 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_histgb_risk_adjusted | development_test | 64 | risk | +0.5603 | +0.0011 | +0.0016 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_ridge_risk_adjusted | validation | 4 | return | +0.0288 | +0.0048 | +0.0073 | +0.5112 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_ridge_risk_adjusted | validation | 16 | return | +0.0023 | +0.0100 | +0.0147 | +0.4977 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_ridge_risk_adjusted | validation | 64 | return | +0.0529 | +0.0214 | +0.0304 | +0.5133 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_ridge_risk_adjusted | development_test | 4 | return | +0.0246 | +0.0057 | +0.0088 | +0.5068 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_ridge_risk_adjusted | development_test | 16 | return | +0.0198 | +0.0116 | +0.0172 | +0.5111 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_ridge_risk_adjusted | development_test | 64 | return | -0.0024 | +0.0243 | +0.0340 | +0.4960 | N/A | N/A | N/A |  |
| 8 | ohlcv13 | context_ridge_risk_adjusted | validation | 4 | risk | +0.4085 | +0.0014 | +0.0022 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_ridge_risk_adjusted | validation | 16 | risk | +0.4773 | +0.0012 | +0.0017 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_ridge_risk_adjusted | validation | 64 | risk | +0.4893 | +0.0010 | +0.0014 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_ridge_risk_adjusted | development_test | 4 | risk | +0.4981 | +0.0017 | +0.0025 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_ridge_risk_adjusted | development_test | 16 | risk | +0.5376 | +0.0015 | +0.0021 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |
| 8 | ohlcv13 | context_ridge_risk_adjusted | development_test | 64 | risk | +0.5425 | +0.0013 | +0.0017 | N/A | N/A | N/A | N/A | sign accuracy is N/A for a one-sided non-negative realized-risk target |

## Feature coverage and timestamp eligibility

External coverage counts deliberately separate finite zero, finite nonzero, and missing values. Because the current cache has no availability mask, a zero cannot be asserted to be observed rather than imputed; full17 is therefore secondary and `blocked_by_data_quality`.

| fold | feature set | split | rows | external nonzero/missing | context excluded | non-15m transitions | return valid h4/h16/h64 | return gap-excluded h4/h16/h64 | promotion |
|---:|---|---|---:|---|---:|---:|---|---|---|
| 0 | ohlcv13 | train | 69703 | funding_rate=0/69703, basis=0/69703, basis_mom=0/69703, basis_abs=0/69703 | 3340 | 14 | 69643/69463/68743 | 56/224/896 | eligible_for_promotion |
| 0 | ohlcv13 | validation | 8701 | funding_rate=0/8701, basis=0/8701, basis_mom=0/8701, basis_abs=0/8701 | 768 | 3 | 8685/8637/8445 | 12/48/192 | eligible_for_promotion |
| 0 | ohlcv13 | test | 8712 | funding_rate=0/8712, basis=0/8712, basis_mom=0/8712, basis_abs=0/8712 | 512 | 2 | 8700/8664/8520 | 8/32/128 | eligible_for_promotion |
| 2 | ohlcv13 | train | 69953 | funding_rate=0/69953, basis=0/69953, basis_mom=0/69953, basis_abs=0/69953 | 3328 | 13 | 69897/69729/69057 | 52/208/832 | eligible_for_promotion |
| 2 | ohlcv13 | validation | 8832 | funding_rate=0/8832, basis=0/8832, basis_mom=0/8832, basis_abs=0/8832 | 0 | 0 | 8828/8816/8768 | 0/0/0 | eligible_for_promotion |
| 2 | ohlcv13 | test | 8808 | funding_rate=0/8808, basis=0/8808, basis_mom=0/8808, basis_abs=0/8808 | 768 | 3 | 8792/8744/8552 | 12/48/192 | eligible_for_promotion |
| 8 | ohlcv13 | train | 70022 | funding_rate=0/70022, basis=0/70022, basis_mom=0/70022, basis_abs=0/70022 | 3584 | 14 | 69962/69782/69062 | 56/224/896 | eligible_for_promotion |
| 8 | ohlcv13 | validation | 8640 | funding_rate=0/8640, basis=0/8640, basis_mom=0/8640, basis_abs=0/8640 | 0 | 0 | 8636/8624/8576 | 0/0/0 | eligible_for_promotion |
| 8 | ohlcv13 | test | 8736 | funding_rate=0/8736, basis=0/8736, basis_mom=0/8736, basis_abs=0/8736 | 0 | 0 | 8732/8720/8672 | 0/0/0 | eligible_for_promotion |
| 0 | full17 | train | 69703 | funding_rate=12293/0, basis=2311/0, basis_mom=2309/0, basis_abs=2310/0 | 3340 | 14 | 69643/69463/68743 | 56/224/896 | blocked_by_data_quality |
| 0 | full17 | validation | 8701 | funding_rate=8701/0, basis=8701/0, basis_mom=8701/0, basis_abs=8701/0 | 768 | 3 | 8685/8637/8445 | 12/48/192 | blocked_by_data_quality |
| 0 | full17 | test | 8712 | funding_rate=8712/0, basis=8712/0, basis_mom=8712/0, basis_abs=8712/0 | 512 | 2 | 8700/8664/8520 | 8/32/128 | blocked_by_data_quality |
| 2 | full17 | train | 69953 | funding_rate=29706/0, basis=19724/0, basis_mom=19722/0, basis_abs=19723/0 | 3328 | 13 | 69897/69729/69057 | 52/208/832 | blocked_by_data_quality |
| 2 | full17 | validation | 8832 | funding_rate=8832/0, basis=8832/0, basis_mom=8832/0, basis_abs=8832/0 | 0 | 0 | 8828/8816/8768 | 0/0/0 | blocked_by_data_quality |
| 2 | full17 | test | 8808 | funding_rate=8808/0, basis=8808/0, basis_mom=8808/0, basis_abs=8808/0 | 768 | 3 | 8792/8744/8552 | 12/48/192 | blocked_by_data_quality |
| 8 | full17 | train | 70022 | funding_rate=70022/0, basis=70022/0, basis_mom=70022/0, basis_abs=70022/0 | 3584 | 14 | 69962/69782/69062 | 56/224/896 | blocked_by_data_quality |
| 8 | full17 | validation | 8640 | funding_rate=8640/0, basis=8640/0, basis_mom=8640/0, basis_abs=8640/0 | 0 | 0 | 8636/8624/8576 | 0/0/0 | blocked_by_data_quality |
| 8 | full17 | test | 8736 | funding_rate=8736/0, basis=8736/0, basis_mom=8736/0, basis_abs=8736/0 | 0 | 0 | 8732/8720/8672 | 0/0/0 | blocked_by_data_quality |

## Frozen Wave3A comparison

Wave3A output is not overwritten. Its Ridge/HistGB OHLCV13 rows are replayed here under the Wave3C validation-selected-constant comparator and common right-side delay alignment. This replay is report-only and excluded from every Wave3C choice.

| source | status | rows | median dynamic AlphaEx | median constant AlphaEx | median timing increment |
|---|---|---:|---:|---:|---:|
| docs/forecast_tournament_plan011_dev/result.json | complete | 6 | +18.9746pt | +20.1675pt | -1.1929pt |

## Artifacts

- machine-readable ledger: `docs/forecast_context_tournament_plan011_dev/forecast_context_tournament_ledger.jsonl`
- result JSON: `docs/forecast_context_tournament_plan011_dev/result.json`
- Wave3A errata: `docs/forecast_context_tournament_plan011_dev/wave3a_errata.md`

No holdout folds 15–23 or future fold 24 were loaded, selected, or inspected by this screen.
