# Matched ML information and RL hindsight decomposition — completed 2026-09-06

The fixed controllers can exploit substantially more return information on the
original development periods. Revealing realized h24 return raises AlphaEx by
**19.585–25.004 percentage points versus each own half-mean controller**;
revealing only realized risk raises it by **0.370–0.587 points**. This supports
prioritizing causal return information and its reliability before changing model
architecture. It does **not** establish that available technical indicators can
predict those future returns, or that a deployable model has improved.

All 28 policies × 8 quarters completed. Independent scalar accounting,
controller-state, substitution and saved-path feasibility audits passed. No
model was fitted, no causal policy was added or selected, and no strongest
model was declared. The 158 explored causal names and frozen candidates remain
unchanged. Additional test15–24 was not used in this diagnostic.

## Scope and definitions

The [registration](oracle_information_decomposition_registration_20260906.md),
[source/config/preflight commit](https://github.com/shinji-ogasa/UniDream/commit/d3b25734a34915049a327256bd9f99cd9aea8336)
and required 612 passing tests preceded the new real rollouts. This study uses
original validation5–12, which alias original test4–11, covering
**[2021-04-16T13:45Z, 2023-04-16T13:45Z)**. These are reused development periods,
not untouched confirmation. The JSON `test_periods_used=false` means no separate
additional-test evaluation was performed; it does not erase that alias/history.

AlphaEx = policy total return minus B&H total return; MaxDDDelta = policy MaxDD
minus B&H MaxDD. Numbers below are **equal-quarter means in percentage points**,
not annualized or a compounded two-year return. Positive AlphaEx and negative
MaxDDDelta are desirable. B&H-relative outperformance does not require a positive
absolute policy return. Turnover is summed absolute traded notional divided by
contemporaneous NAV; trades are mean filled-trade counts per quarter. Stress
uses twice the fee and borrowing rate on the exact base intents, without
replanning. B&H numerical residuals below 1e−12 are displayed as zero.

Start regimes use information available at the first scheduled decision; they
are not labels derived from each quarter's later return. There are 2 bull,
4 bear and 2 sideways quarters, insufficient for the prior regime-count gate.

## Matched interventions and action constraints

The 12 causal controls reproduce their parent targets and accounting exactly.
For each technical/perpetual half mean and hold/fallback rule, only the original
score support receives one of three fixed substitutions: realized h24 return,
realized h24 volatility squared, or both. The 2,574 substituted rows are a
subset of 2,586 available inference rows per mean stream. All 12 remaining
inference rows retain the learned forecast (8 quarter-tail maturities, 4 future
gaps); missing forecasts remain missing. These counts are not independent
samples multiplied by the number of variants. All 332 known-open fallback
opportunities and 2 missing-current-open scheduled rows are retained.

Return is `log(close[t+24]/open[t+1])`; risk replacement is the saved realized
volatility squared, without a floor, annualization, factor24 or a second half
weight. Neither realization is the true conditional expectation/variance.
Mechanically substituted target error is not reported as learned accuracy.
These are explicitly marked hindsight hybrids, with teacher use disabled.

Every controller uses its own cash and units. On supported six-hour decisions,
intents come from own current-open exposure ±0.04/±0.08 clipped to [0.5,1.12],
plus no trade, with step0.08 and deadband0.01. Current state is read before that
bar's close mark and borrowing. Fallback forces target1 only for unsupported
known-open decisions. Orders fill at the immediately following bar's open, when observed; a missing
next open skips the order without rollover. Full price paths, borrowing across gaps,
initial B&H inventory, fee0.00055 and annual borrowing0.10 remain fixed.

The four hindsight searches use beam32 and terminal objective
`log(NAV) − penalty × MaxDD`, with penalty0/1 and hold/fallback. Each has a
rule-matched feasible incumbent. All32 real searches pruned distinct branches;
none selected its incumbent. These results are **feasible lower bounds on the
maximum hindsight objective within the registered action set**, not global
optima or upper bounds on achievable causal model performance. Their horizon
and full-path objective differ from the h24 local-utility hybrids, so the gap
cannot isolate decision quality.

## Complete aggregate economic inventory

The 12 original controls appear first, followed by all 16 hindsight diagnostics.
Each metric pair below is AlphaEx / MaxDDDelta in points. No row is selected as
a winner. Exact IDs, all quarters and all account fields are retained in the
[complete results JSON](oracle_information_decomposition_evidence_20260906/results.json).

| Policy | Base Alpha / DD | Stress Alpha / DD | Base turnover | Stress turnover | Base trades | Stress trades |
|---|---:|---:|---:|---:|---:|---:|
| B&H | +0.000 / +0.000 | +0.000 / +0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Common robust | -0.317 / -5.347 | -0.651 / -5.230 | 4.781 | 4.781 | 78.625 | 78.625 |
| Scale mean / hold | +3.128 / -6.531 | +3.061 / -6.494 | 0.247 | 0.246 | 4.000 | 3.875 |
| Scale mean / fallback | +3.541 / -6.716 | +3.486 / -6.688 | 0.558 | 0.557 | 8.250 | 8.125 |
| Technical full / hold | +0.595 / -4.323 | +0.277 / -4.211 | 3.187 | 3.187 | 44.500 | 44.500 |
| Technical full / fallback | +0.705 / -5.079 | +0.410 / -4.981 | 3.644 | 3.643 | 50.625 | 50.625 |
| Perp full / hold | +2.131 / -3.130 | +1.723 / -3.001 | 4.232 | 4.231 | 57.625 | 57.625 |
| Perp full / fallback | +1.918 / -3.777 | +1.546 / -3.664 | 4.563 | 4.563 | 62.125 | 62.125 |
| Technical half / hold | +1.149 / -6.031 | +0.943 / -5.953 | 1.622 | 1.621 | 23.750 | 23.625 |
| Technical half / fallback | +1.007 / -6.783 | +0.821 / -6.718 | 2.093 | 2.091 | 30.250 | 30.125 |
| Perp half / hold | +3.613 / -5.388 | +3.349 / -5.305 | 2.089 | 2.087 | 29.750 | 29.500 |
| Perp half / fallback | +3.379 / -6.160 | +3.144 / -6.090 | 2.455 | 2.454 | 34.375 | 34.125 |
| Technical half + realized return / hold | +23.198 / -11.805 | +21.767 / -11.604 | 20.920 | 20.920 | 267.125 | 267.125 |
| Technical half + realized return / fallback | +26.011 / -11.835 | +24.547 / -11.599 | 21.100 | 21.100 | 270.125 | 270.125 |
| Technical half + realized risk / hold | +1.663 / -6.315 | +1.430 / -6.238 | 1.742 | 1.742 | 25.375 | 25.250 |
| Technical half + realized risk / fallback | +1.376 / -6.940 | +1.158 / -6.873 | 2.206 | 2.206 | 31.750 | 31.625 |
| Technical half + both / hold | +23.128 / -11.543 | +21.693 / -11.341 | 20.947 | 20.946 | 267.375 | 267.375 |
| Technical half + both / fallback | +25.884 / -11.547 | +24.418 / -11.312 | 21.107 | 21.107 | 270.125 | 270.125 |
| Perp half + realized return / hold | +23.198 / -11.805 | +21.767 / -11.604 | 20.920 | 20.920 | 267.125 | 267.125 |
| Perp half + realized return / fallback | +26.011 / -11.835 | +24.547 / -11.599 | 21.100 | 21.100 | 270.125 | 270.125 |
| Perp half + realized risk / hold | +4.200 / -5.844 | +3.907 / -5.763 | 2.159 | 2.159 | 30.625 | 30.375 |
| Perp half + realized risk / fallback | +3.821 / -6.500 | +3.554 / -6.429 | 2.530 | 2.530 | 35.625 | 35.375 |
| Perp half + both / hold | +23.128 / -11.543 | +21.693 / -11.341 | 20.947 | 20.946 | 267.375 | 267.375 |
| Perp half + both / fallback | +25.884 / -11.547 | +24.418 / -11.312 | 21.107 | 21.107 | 270.125 | 270.125 |
| RL beam32 / hold / penalty 0 | +36.775 / -16.266 | +35.259 / -16.032 | 19.863 | 19.863 | 270.875 | 270.875 |
| RL beam32 / hold / penalty 1 | +35.380 / -17.220 | +33.876 / -16.972 | 19.992 | 19.992 | 272.375 | 272.375 |
| RL beam32 / fallback / penalty 0 | +37.050 / -13.502 | +35.519 / -13.288 | 20.062 | 20.062 | 274.125 | 274.125 |
| RL beam32 / fallback / penalty 1 | +36.263 / -13.936 | +34.739 / -13.723 | 20.129 | 20.129 | 273.875 | 273.875 |

## Each information intervention minus its own learned control

These are paired changes, not the absolute levels in the preceding table.
Negative DD change means a further drawdown improvement. All 12 registered
interventions are retained, including adverse incremental effects.

| Half mean, replacement and missing rule | Base ΔAlpha / ΔDD | Stress ΔAlpha / ΔDD |
|---|---:|---:|
| Technical half + realized return / hold | +22.049 / -5.773 | +20.824 / -5.651 |
| Technical half + realized return / fallback | +25.004 / -5.052 | +23.725 / -4.881 |
| Technical half + realized risk / hold | +0.514 / -0.284 | +0.486 / -0.285 |
| Technical half + realized risk / fallback | +0.370 / -0.157 | +0.337 / -0.155 |
| Technical half + both / hold | +21.979 / -5.512 | +20.750 / -5.388 |
| Technical half + both / fallback | +24.878 / -4.764 | +23.596 / -4.594 |
| Perp half + realized return / hold | +19.585 / -6.417 | +18.418 / -6.299 |
| Perp half + realized return / fallback | +22.631 / -5.675 | +21.403 / -5.509 |
| Perp half + realized risk / hold | +0.587 / -0.457 | +0.558 / -0.459 |
| Perp half + realized risk / fallback | +0.442 / -0.340 | +0.410 / -0.339 |
| Perp half + both / hold | +19.515 / -6.155 | +18.344 / -6.037 |
| Perp half + both / fallback | +22.505 / -5.387 | +21.274 / -5.222 |

Return-only intervention levels are +23.198 to +26.011 AlphaEx; risk-only
levels are +1.376 to +4.200. Those levels must not be confused with the paired
improvements. Adding realized risk to the realized-return intervention makes
aggregate AlphaEx **0.070–0.126 points worse** and MaxDDDelta **0.262–0.288
points worse**. The observed risk realization is not an ideal conditional-risk
forecast, and the fixed local utility need not improve when given it.

Technical/perpetual return-only and both-swap targets happened to coincide in
all 32 paired fold comparisons after execution. They were separately computed
and audited; this is not a guarantee that retained learned remainder rows or
own-inventory paths generally coincide.

## All hindsight diagnostics by start regime

Each cell is base AlphaEx / MaxDDDelta, followed by the same pair under doubled
costs in parentheses, all in points. The full JSON also contains turnover and
trades for every policy, regime and cost setting. Regime means are descriptive;
future information and reused 2/4/2-quarter coverage cannot establish a high
probability of causal, trend-independent improvement.

| Diagnostic | Bull (2) | Bear (4) | Sideways (2) |
|---|---:|---:|---:|
| Technical half + realized return / hold | +21.251 / -11.358 (+20.272 / -11.194) | +21.545 / -9.569 (+19.939 / -9.439) | +28.452 / -16.722 (+26.920 / -16.343) |
| Technical half + realized return / fallback | +21.563 / -12.278 (+20.595 / -11.983) | +27.278 / -9.392 (+25.612 / -9.262) | +27.922 / -16.278 (+26.368 / -15.890) |
| Technical half + realized risk / hold | +4.838 / -7.084 (+4.672 / -6.974) | +1.143 / -3.442 (+0.843 / -3.372) | -0.471 / -11.292 (-0.640 / -11.234) |
| Technical half + realized risk / fallback | +6.492 / -8.693 (+6.387 / -8.632) | -0.671 / -3.732 (-0.942 / -3.670) | +0.354 / -11.600 (+0.128 / -11.521) |
| Technical half + both / hold | +20.604 / -10.985 (+19.630 / -10.822) | +21.651 / -9.236 (+20.034 / -9.105) | +28.606 / -16.714 (+27.075 / -16.333) |
| Technical half + both / fallback | +20.974 / -11.904 (+20.011 / -11.613) | +27.308 / -9.059 (+25.630 / -8.928) | +27.946 / -16.167 (+26.399 / -15.780) |
| Perp half + realized return / hold | +21.251 / -11.358 (+20.272 / -11.194) | +21.545 / -9.569 (+19.939 / -9.439) | +28.452 / -16.722 (+26.920 / -16.343) |
| Perp half + realized return / fallback | +21.563 / -12.278 (+20.595 / -11.983) | +27.278 / -9.392 (+25.612 / -9.262) | +27.922 / -16.278 (+26.368 / -15.890) |
| Perp half + realized risk / hold | +4.648 / -6.897 (+4.479 / -6.785) | +2.225 / -3.173 (+1.868 / -3.104) | +7.703 / -10.134 (+7.415 / -10.060) |
| Perp half + realized risk / fallback | +6.350 / -8.552 (+6.242 / -8.488) | +0.409 / -3.464 (+0.085 / -3.403) | +8.117 / -10.521 (+7.802 / -10.425) |
| Perp half + both / hold | +20.604 / -10.985 (+19.630 / -10.822) | +21.651 / -9.236 (+20.034 / -9.105) | +28.606 / -16.714 (+27.075 / -16.333) |
| Perp half + both / fallback | +20.974 / -11.904 (+20.011 / -11.613) | +27.308 / -9.059 (+25.630 / -8.928) | +27.946 / -16.167 (+26.399 / -15.780) |
| RL beam32 / hold / penalty 0 | +38.059 / -25.396 (+37.013 / -24.962) | +34.828 / -10.249 (+33.119 / -10.178) | +39.383 / -19.169 (+37.785 / -18.810) |
| RL beam32 / hold / penalty 1 | +38.476 / -25.818 (+37.416 / -25.385) | +32.541 / -11.801 (+30.855 / -11.701) | +37.961 / -19.459 (+36.377 / -19.102) |
| RL beam32 / fallback / penalty 0 | +28.762 / -15.120 (+27.771 / -14.775) | +39.862 / -9.888 (+38.100 / -9.816) | +39.714 / -19.113 (+38.105 / -18.743) |
| RL beam32 / fallback / penalty 1 | +29.045 / -15.352 (+28.050 / -15.009) | +38.419 / -10.341 (+36.666 / -10.269) | +39.167 / -19.710 (+37.574 / -19.346) |

## Finite-search tradeoffs and limits

Increasing the drawdown penalty from0 to1 changes base aggregate AlphaEx /
MaxDDDelta by **−1.395 / −0.954 points** under hold and **−0.787 / −0.434 points**
under fallback. The average drawdown improves while average return falls.
However, evaluating both saved paths under the same penalty1 objective shows
that the penalty1 search is worse than the feasible penalty0 path in **4/8
quarters for each rule** (hold folds5,8,11,12; fallback6,8,11,12). This directly
exposes finite-search suboptimality; no stronger optimality claim is warranted.
No post-result replacement, cross-penalty winner or wider search was substituted
into the registered results.

[Brown, Smith and Sun (2010)](https://doi.org/10.1287/opre.1090.0796) derive
information-relaxation bounds under properly specified relaxed optimization;
[Brown and Smith (2011)](https://doi.org/10.1287/mnsc.1110.1377) apply dual bounds
to portfolio optimization with transaction costs. A finite feasible beam path
alone does not provide those upper bounds. Neither reference validates the
BTC performance or learnability reported here.

## Verification and reproducibility

- Required full command `uv run python -m unittest discover -s tests -v` passed
  **612 tests in 59.254s** before source freeze and execution. This includes
  16 matched-beam synthetic cases and 7 intervention/runner cases. Synthetic
  wide-beam comparisons include independent exhaustive toy paths; they do not
  establish real-data global optimality. `git diff --check` passed.
- Real run session27200 terminated with exit0 after all8 folds. No fitting or
  new data acquisition occurred. The completed namespace is
  `codex_outputs/oracle_information_decomposition_v1`; do not restart it.
  There are **224 economic rows, 48 hybrid forecasts, 224 target vectors,
  128 hindsight traces = 400 new artifacts**, plus8 fold JSON manifests.
- The [independent scalar audit](oracle_information_decomposition_evidence_20260906/independent_scalar_audit.json)
  and [source](oracle_information_decomposition_evidence_20260906/independent_scalar_audit.py)
  verify1,328 ancestral and400 new artifacts, with2,159 binding checks across
  1,787 distinct hashed files. All448 base/stress accounts match AlphaEx,
  MaxDDDelta, turnover, fees, borrowing and trades exactly; maximum mean
  exposure difference is2.22e−16. All96 hybrid own-state paths/33,024 decisions,
  all32 RL paths/11,008 decisions,32 rule-matched incumbent accounts,48
  substitutions,96 controls and1,440 summary scalars pass. Independent h24
  actual reconstruction differs by at most3.96e−15. It imports no canonical
  planner/simulator/metrics/search implementation. It does not rerun beam
  expansion/pruning or prove an optimum.
- The separate [Decimal60 summary audit](oracle_information_decomposition_evidence_20260906/independent_summary_audit.json)
  and [source](oracle_information_decomposition_evidence_20260906/independent_summary_audit.py)
  recompute all28 policy summaries and12 paired swaps. Maximum difference is
  3.88e−15 in turnover; RL objectives match exactly. This audit verifies446
  direct files, including23 source modules and400 new artifacts; its ancestor
  scope is the hash-bound preflight, not an independent rehash of all ancestors.
- [Pre-outcome verification](oracle_information_decomposition_evidence_20260906/pre_outcome_verification.json)
  preserves the full-test log hash and the separate data-only source audit.
  Published registration/results and audit scripts are exact byte copies;
  [publication verification](oracle_information_decomposition_evidence_20260906/publication_verification.json)
  binds the report, source, outputs and copied evidence. Raw/model/trace
  artifacts remain in the ignored local output directories; the tracked
  evidence is not a substitute for those files when rerunning audits.

The source revision is `d3b25734a34915049a327256bd9f99cd9aea8336`. File hashes:

| Evidence | SHA256 |
|---|---|
| Config | `f10c14e65c9be4edd1d56c303dce9262a79f22fe5c5a3686454b66e6202512d4` |
| Preflight | `ae66338b39253f88e536729948b96b9eb57abe9ba409cbfc074f339be688f4e7` |
| Runtime registration file | `8e25d7743e49e9df6269cb898808ee1fde691a52d56003c69b52ce9497560fc1` |
| Results file | `f5597dee653a45ee612766111da578a868969a51211f923493763b4059a18ac7` |
| Independent scalar audit | `f3de0fa4d69da2e615a839bfb90ffc9d9852a42e5a0a775d4d5b9f1747b6c343` |
| Independent summary audit | `b1f486c3147e86835ed93bb07817b761ff8333772df673239043330b1a753cf6` |

The embedded `registration_sha256` is a canonical-content binding, distinct
from the runtime registration file's byte hash above.

## Consequence for the active goal

The result locates substantial value in return information under the current
controller and cost contract. It does not identify a causal feature that
supplies that information. Next work should use original development data to
predeclare a small return-information/reliability study: test technical and
Spot/perpetual inputs against constant and simple causal baselines, isolate
forecast improvement from its economic effect, and preserve all failed
comparisons. New feature transformations must have causal availability and
cross-period evidence; more transformations alone do not demonstrate new
information. No weight, fallback rule, regime cutoff or model is retuned from
additional test15–24. Model-architecture optimization remains deferred.

The previous [additional-window result](oracle_additional_window_results_20260906.md)
still shows that all four frozen half candidates fail the trend-wide economic
and predictive conjunctions. This Oracle diagnosis does not overturn that
causal evidence. No independent confirmation, prospective receipt proof,
selection-adjusted probability or deployable strongest model has been obtained.
The broader goal remains active.
