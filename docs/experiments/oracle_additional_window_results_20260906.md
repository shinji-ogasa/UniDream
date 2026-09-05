# Fixed half-mean candidates: additional reused-window results

**No half-mean candidate meets the requested regime-wide economic and predictive conditions.**
The ten additional quarters have adequate regime counts (bull4/bear3/sideways3),
but all four half candidates have negative mean AlphaEx in the start-bull stratum
and positive MaxDDDelta in the start-sideways stratum. Neither half mean beats
its scale-period constant mean on aggregate MSE. No model is selected or promoted.

The half-weight procedure does reduce aggregate MSE versus its own full forecasts:
0.800% for technical and 0.781% for perpetual, improving 8/10 quarters for each.
That improvement reverses slightly in the bull stratum. The earlier development
signal therefore does not establish improvement independent of trend.

## Scope and execution

This is the separately registered, **report-only reused original test15–24**
evaluation from January 16, 2024 13:45 UTC to July 16, 2026 13:45 UTC, right exclusive.
It is not an unread holdout or prospective confirmation. Prior research accessed
these dates; the old `fresh` label on test24 does not imply independence.

The complete frozen family (four half candidates, eight controls) was replayed
without changing weights, features, fit/calibration windows, models, risk,
execution, missing-input rules or costs. Thirty models were fitted using the
same 18/3/3-month procedure already reproduced exactly on development data.
This adds no policy to the 158-name exploratory history and modifies no lock.

- [Original registration](oracle_additional_window_registration_20260906.md)
- [Execution and data-only binding](oracle_additional_window_execution_20260906.md)
- [Full results](oracle_additional_window_evidence_20260906/results.json)
- [Preflight](oracle_additional_window_evidence_20260906/preflight.json)

Acquisition was committed at `f2e1faf`, and the complete adapter/config/preflight
at **`73cb806bffb7dc39d7454886e9bcfce9119dc435`**, before new fits. The run completed
normally (session8868, exit0), with 120 policy-quarter rows, 50 forecast scores,
100 utility traces, 30 models and 330 model/forecast/target/calibration artifacts.
All ten quarters completed; no subset was selected or dropped.

## Economic outcomes

Values below are **percentage points, equally weighted quarterly means**.
AlphaEx is strategy return minus B&H; MaxDDDelta is strategy drawdown minus B&H
drawdown. These drawdown averages are not the drawdown of a concatenated path.
Stress doubles fees and borrowing while replaying the same base intents.

| Fixed policy | Base AlphaEx | Base MaxDDDelta | Stress AlphaEx | Stress MaxDDDelta | Joint quarters, base/stress |
|---|---:|---:|---:|---:|---:|
| B&H | 0.000 | 0.000 | 0.000 | 0.000 | — |
| Common robust | +1.645 | -4.752 | +1.229 | -4.655 | 5/10 / 5/10 |
| Scale mean / hold | +0.310 | +0.362 | +0.260 | +0.357 | 2/10 / 2/10 |
| Scale mean / fallback | +0.310 | +0.362 | +0.260 | +0.357 | 2/10 / 2/10 |
| Technical full / hold | +0.509 | -2.294 | +0.169 | -2.222 | 4/10 / 4/10 |
| Technical full / fallback | +1.365 | -2.294 | +1.021 | -2.222 | 4/10 / 4/10 |
| Perpetual full / hold | +1.499 | -2.674 | +1.117 | -2.614 | 4/10 / 4/10 |
| Perpetual full / fallback | +2.319 | -2.674 | +1.934 | -2.614 | 4/10 / 4/10 |
| Technical half / hold | +1.336 | -2.992 | +1.162 | -2.971 | 4/10 / 4/10 |
| Technical half / fallback | +1.542 | -3.015 | +1.371 | -2.995 | 4/10 / 4/10 |
| Perpetual half / hold | +0.048 | -2.826 | -0.130 | -2.804 | 4/10 / 4/10 |
| Perpetual half / fallback | +0.866 | -2.853 | +0.680 | -2.831 | 4/10 / 4/10 |

B&H's raw relative metrics contain machine-roundoff terms around 1e−17; these
are displayed as zero. Its raw strict-sign joint count is not evidence of an
advantage over itself. Candidate economic conclusions are far larger than that
roundoff. Three of four half candidates have favorable aggregate mean signs
under both costs, but none has favorable signs in every regime.

The pre-existing **common robust rule control** does have favorable economic
mean signs in all three additional-window regimes under both costs. Its aggregate
is +1.645pt / −4.752pt at base cost and +1.229pt / −4.655pt at doubled cost.
This is useful benchmark evidence, but it is not a newly selected winning model:
in the original eight development quarters its aggregate AlphaEx was −0.317pt
and bear-stratum AlphaEx −6.610pt (−0.651pt / −7.002pt at doubled costs).
The [separate-window comparison](oracle_additional_window_evidence_20260906/common_robust_context.json)
keeps both results visible without pooling away the earlier failure or using
the additional test outcomes for selection.

| Candidate | Start regime | Quarters | Base AlphaEx / DD delta | Stress AlphaEx / DD delta |
|---|---|---:|---:|---:|
| Technical half / hold | bull | 4 | -0.435 / -5.042 | -0.522 / -5.060 |
| Technical half / hold | bear | 3 | +5.787 / -5.304 | +5.563 / -5.252 |
| Technical half / hold | sideways | 3 | -0.752 / +2.054 | -0.993 / +2.096 |
| Technical half / fallback | bull | 4 | -0.435 / -5.042 | -0.522 / -5.060 |
| Technical half / fallback | bear | 3 | +5.787 / -5.304 | +5.563 / -5.252 |
| Technical half / fallback | sideways | 3 | -0.065 / +1.976 | -0.296 / +2.016 |
| Perpetual half / hold | bull | 4 | -1.378 / -4.789 | -1.469 / -4.805 |
| Perpetual half / hold | bear | 3 | +5.831 / -4.802 | +5.596 / -4.751 |
| Perpetual half / hold | sideways | 3 | -3.834 / +1.768 | -4.070 / +1.811 |
| Perpetual half / fallback | bull | 4 | -1.378 / -4.789 | -1.469 / -4.805 |
| Perpetual half / fallback | bear | 3 | +5.831 / -4.802 | +5.596 / -4.751 |
| Perpetual half / fallback | sideways | 3 | -1.106 / +1.679 | -1.370 / +1.722 |

Regimes are fixed from past information at the first scheduled decision, not
realized quarterly return. For example, test19 starts in the bull stratum but
B&H subsequently returns −15.440%; test20 starts bear but returns +40.820%.
Do not reinterpret the grouped results as perfect knowledge of future trends.

## Forecast accuracy

There are 3,610 common scored observations per mean stream. MSE and MAE below
are equal-quarter means. Constant scale-mean rank IC is undefined. Positive
affine half shrinkage leaves each quarter's rank IC exactly unchanged.

| Fixed mean | Return MSE | Return MAE | Mean rank IC | Sign accuracy |
|---|---:|---:|---:|---:|
| scale_mean | 0.0001534800 | 0.00822783 | undefined | 51.127% |
| technical_scaled | 0.0001548045 | 0.00832045 | 0.02210 | 50.496% |
| technical_half | 0.0001535668 | 0.00824763 | 0.02210 | 51.077% |
| perp_delay0_scaled | 0.0001547112 | 0.00833277 | 0.02051 | 50.268% |
| perp_delay0_half | 0.0001535027 | 0.00825120 | 0.02051 | 50.571% |

Improvement is `1 − mean quarterly candidate loss / mean quarterly reference
loss`, not the mean of per-quarter percentage improvements. Positive is better.
Row-pooled values are separately retained in JSON.

| Forecast contrast | Aggregate MSE improvement | MAE improvement | MSE-improved quarters | Bull / bear / sideways MSE improvement |
|---|---:|---:|---:|---:|
| perp_delay0_half_vs_perp_delay0_scaled | +0.781% | +0.979% | 8/10 | -0.014% / +2.168% / +0.651% |
| perp_delay0_half_vs_scale_mean | -0.015% | -0.284% | 5/10 | +0.483% / -0.670% / -0.122% |
| perp_delay0_half_vs_technical_half | +0.042% | -0.043% | 5/10 | -0.027% / -0.103% / +0.240% |
| technical_half_vs_scale_mean | -0.057% | -0.241% | 5/10 | +0.510% / -0.566% / -0.363% |
| technical_half_vs_technical_scaled | +0.800% | +0.875% | 8/10 | -0.090% / +2.052% / +0.893% |

Technical half MSE remains 0.057% worse than the scale mean; perpetual half is
0.015% worse. Both are also worse than zero-return and fit-period-mean forecasts
on aggregate MSE. The small perpetual-half MSE edge over technical-half is only
0.042%, wins 5/10 quarters, and reverses on aggregate MAE. It is not a stable
feature advantage. The full 80 descriptive economic/predictive components have
null p-values and no confidence intervals. All four candidate conjunctions fail.

## Every additional quarter

The next table preserves all ten outcomes for both half means and missing-input
rules. Each cell is base-cost AlphaEx / MaxDDDelta in percentage points.

| Original test fold | Start regime | Actual B&H return | Technical hold | Technical fallback | Perpetual hold | Perpetual fallback |
|---|---|---:|---:|---:|---:|---:|
| 15 | bull | +45.536% | -14.705 / -8.187 | -14.705 / -8.187 | -18.252 / -7.780 | -18.252 / -7.780 |
| 16 | bull | +2.137% | -0.514 / +3.082 | -0.514 / +3.082 | -0.514 / +3.082 | -0.514 / +3.082 |
| 17 | sideways | +5.237% | +1.001 / +1.884 | +1.001 / +1.884 | +0.039 / +2.710 | +0.039 / +2.710 |
| 18 | sideways | +47.829% | -1.562 / +0.830 | +0.497 / +0.597 | -9.975 / -0.851 | -1.791 / -1.120 |
| 19 | bull | -15.440% | +11.003 / -8.320 | +11.003 / -8.320 | +10.707 / -7.650 | +10.707 / -7.650 |
| 20 | bear | +40.820% | +5.166 / +1.411 | +5.166 / +1.411 | +5.166 / +1.411 | +5.166 / +1.411 |
| 21 | bull | -5.943% | +2.475 / -6.745 | +2.475 / -6.745 | +2.547 / -6.809 | +2.547 / -6.809 |
| 22 | sideways | -14.281% | -1.694 / +3.447 | -1.694 / +3.447 | -1.568 / +3.447 | -1.568 / +3.447 |
| 23 | bear | -21.819% | +2.501 / -2.308 | +2.501 / -2.308 | +3.310 / -0.951 | +3.310 / -0.951 |
| 24 | bear | -14.000% | +9.693 / -15.016 | +9.693 / -15.016 | +9.016 / -14.865 | +9.016 / -14.865 |

## Availability and reproducibility

All evaluation Spot price bars are present. The inherited feature common mask
still excludes 28 scheduled forecasts in test18, leaving 3,620 inference rows
from 3,648 scheduled decisions. The economic path retains these periods.
Fallback versus hold differences arise in that quarter and are reported for
both rules, without selecting the favorable rule. The known current opens
allow the frozen target-one action; this does not mean 28 price bars are missing.

The UM acquisition retained and verified all 55 monthly ZIPs (January2022–July2026),
160,608 rows, with no missing raw rows. Every ZIP was reparsed and compared with
the monthly and assembled frames. Spot uses the previously bound full-history
artifact. Its raw ZIPs are not retained; the provenance records this limitation.
[Binance archive documentation](https://github.com/binance/binance-public-data)
documents official checksums, possible revisions and Spot microseconds from
January2025; [UM general information](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info)
specifies milliseconds. Archive timestamps/checksums do not prove historical
receipt, execution latency or prospective availability.

- **589 tests pass**; full-suite command and logs are bound in the pre-outcome
  verification. Code and all registered input/source hashes remained unchanged.
- The [independent scalar audit](oracle_additional_window_evidence_20260906/independent_scalar_audit.json)
  reconstructs all 100 own-state decision paths (36,340 decisions), 240
  base/stress accounting paths, 50 forecast scores and 20 half identities.
  Maximum absolute difference is **4.441e−16**. It calls no canonical planner,
  simulator or scoring helper. Common robust targets are accounted independently
  but their feature rule is not independently rebuilt in this audit.
- The numerical run emitted **120 RuntimeWarnings** in linear-model matrix
  multiplication. A separate [scalar Ridge audit](oracle_additional_window_evidence_20260906/scalar_ridge_audit.json)
  checks all 20 saved mean models using Python scalar arithmetic: 14,400 raw
  calibration predictions and 7,240 calibrated inference predictions.
  All parameters are finite with positive scaler scales; maximum forecast
  difference is **3.469e−18**. This agreement does not identify the warning cause
  or independently validate HGB training numerics.
- The [independent summary audit](oracle_additional_window_evidence_20260906/independent_summary_audit.json)
  uses 60-digit Decimal sums to check all equal-quarter and row-pooled summaries,
  80 components, five forecast contrasts, ten policy contrasts and two rule
  contrasts. Maximum aggregate difference is 7.105e−15 (trade-count mean);
  component differences are at most 6.939e−18.

Registration-file SHA256:
`b1838eca2efc6523537a49415803eb5f9a07a7603257fd11f6abaa1c4d5653eb`.
Results-file SHA256:
`8579e40b5be9ed737acf6c633c92d33b383d0fb338c75832f7b3a125bdd474d1`.

## Consequence for the active goal

The fixed half shrinkage reduces forecast amplitude and aggregate forecast loss,
but this alone does not produce a trend-robust predictive or economic advantage.
Perpetual half / hold, previously favorable in every observed development
regime mean, now has only +0.048pt aggregate AlphaEx and turns negative at
−0.130pt under doubled costs. This weakens the earlier development evidence.

The frozen family is not promoted. Keep these test outcomes report-only and
retain all original locks; do not change the weight or missing-input rule using
these results. The next research step should return to the registered
development scope and test whether a small, justified forecast/control change
beats simple constant forecasts across regimes before any model architecture
search. A separate prospective receipt-aware confirmation remains unimplemented
and no high-probability generalization or investment-performance claim is made.
