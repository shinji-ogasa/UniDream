# Scale-only return reliability — completed 2026-09-06

All four new causal calibration policies have **positive AlphaEx and negative
MaxDDDelta in each evaluation-start regime's observed mean, under both cost
settings** on the original eight development quarters. However, each satisfies
the two economic signs under both costs in only **3/8 individual quarters**.
Relative to each own half policy, aggregate AlphaEx improves but MaxDDDelta
becomes less favorable by0.244–0.405 points at base costs (up to0.407 under
stress); this is not joint domination of
the parent. Their forecast MSE remains worse than a constant scale mean and zero, and the
2bull/4bear/2sideways regime-count gate still fails. **No strongest model or
high-probability, trend-independent improvement has been established.**

The continuous past-only calibration improves evaluation MSE versus the own full
forecast by5.162% technical /3.990% perpetual, and versus the own fixed half by
1.063% /0.304%. Yet MSE is **0.809% /1.236% worse than the scale-mean anchor**.
The exact loss decomposition identifies mean-drift terms larger than this total
excess loss, while the centered component is favorable. This narrows the next
question to causal adaptation of forecast/return means; it does not prove the
cause of drift or that another correction will work.

## Fixed procedure and scope

The [registration](oracle_mean_reliability_registration_20260906.md),
[research note](oracle_mean_reliability_research_20260906.md), config, new source,
tests and data-only preflight were committed and pushed as
[`ff9bbb9`](https://github.com/shinji-ogasa/UniDream/commit/ff9bbb92d588b4615ae8352b353a1301751202a7)
before any new coefficient was estimated. No parameter, missing rule, feature,
base model or source was changed after these outcomes. No policy was selected.

Use original validation5–12, alias original test4–11:
[2021-04-16T13:45Z,2023-04-16T13:45Z), right-exclusive. These are reused
development data. No additional-test15–24 semantic modeling or scoring occurs;
the inherited loader decodes the full Spot parquet before slicing, so that
scope is not a claim that later bytes were never read. Prior test findings and
all earlier locks remain unchanged.

The base Ridge models still fit [E−24months,E−6months). Only scale
S=[E−6months,E−3months) estimates the new scalar weight; the next interval
I=[E−3months,E) and evaluation [E,E+3months) hold it fixed. All h24 labels
mature at t+375minutes, strictly before calibration segment ends and no later
than evaluation end. No online coefficient update occurs. The study estimates
**16 scalar calibration weights and refits 0 base models**.

With exact saved scale mean a and full endpoint p=raw+saved_bias, on S:

    d=p−a, r=y−a, B=mean(d²), C=mean(d*r)
    w=0 if B==0 or C<=0; w=1 if C>=B; otherwise w=C/B.

Issue the convex sum w*p+(1−w)*a, with exact copies at w=0/1. This minimizes
scale squared loss on the registered anchor/full segment; scale improvement
is fitted and cannot count as success. It preserves prior bias arithmetic,
not exactly raw-centered unconstrained OLS. The maximum absolute scale mean(d)
residual is1.09e−18 and mean(r) is4.62e−19. Positive weights preserve ranks
apart from possible floating-point ties. No old interval widths are reused to
claim calibrated uncertainty for the changed mean.

All128economic rows,216score records and16weights are preserved in the
[complete results](oracle_mean_reliability_evidence_20260906/results.json).
The adaptive causal-name ledger rises from158 to162; the prior frozen
candidate family is not rewritten.

## Weights learned only on scale

| Original validation fold | Technical weight | Perpetual weight |
|---:|---:|---:|
| 5 | 0.264776 | 0.472445 |
| 6 | 0.368151 | 0.271626 |
| 7 | 1.000000 | 1.000000 |
| 8 | 0.114233 | 0.061155 |
| 9 | 0.255469 | 0.270464 |
| 10 | 0.581460 | 0.649504 |
| 11 | 0.081524 | 0.078631 |
| 12 | 0.093329 | 0.000000 |

There are13interior weights,2upper endpoints (both fold7), and1zero endpoint
(perpetual fold12, nonpositive crossmoment). No zero-dispersion case occurs.
The zero endpoint is the original scale forecast and its two existing policy
paths, not a newly discovered signal. Upper endpoints similarly reproduce the
own full forecast and both policies.

## All economic policies

Numbers are equal-quarter mean **percentage points**, AlphaEx / MaxDDDelta.
AlphaEx subtracts B&H total return; MaxDDDelta subtracts B&H maximum drawdown.
They are not annualized or compounded across quarters. All policies start with
B&H inventory. The exact shared technical risk, own cash/units, six-hour
schedule, next-bar open fills, missing-price skips without rollover, step.08,
deadband.01, fee.00055 and annual borrow.10 remain fixed. Stress doubles both
costs on the same base intents without replanning. Turnover is mean quarterly
sum of absolute traded notional/current NAV; trades are mean filled-trade counts.
B&H floating-point residuals below1e−12 display as zero.

| Policy | Base Alpha / DD | Stress Alpha / DD | Base turnover | Stress turnover | Base trades | Stress trades |
|---|---:|---:|---:|---:|---:|---:|
| B&H | +0.000 / +0.000 | +0.000 / +0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Common robust | -0.317 / -5.347 | -0.651 / -5.230 | 4.781 | 4.781 | 78.625 | 78.625 |
| Scale mean / hold | +3.128 / -6.531 | +3.061 / -6.494 | 0.247 | 0.246 | 4.000 | 3.875 |
| Scale mean / fallback | +3.541 / -6.716 | +3.486 / -6.688 | 0.558 | 0.557 | 8.250 | 8.125 |
| Technical full / hold | +0.595 / -4.323 | +0.277 / -4.211 | 3.187 | 3.187 | 44.500 | 44.500 |
| Technical full / fallback | +0.705 / -5.079 | +0.410 / -4.981 | 3.644 | 3.643 | 50.625 | 50.625 |
| Perpetual full / hold | +2.131 / -3.130 | +1.723 / -3.001 | 4.232 | 4.231 | 57.625 | 57.625 |
| Perpetual full / fallback | +1.918 / -3.777 | +1.546 / -3.664 | 4.563 | 4.563 | 62.125 | 62.125 |
| Technical half / hold | +1.149 / -6.031 | +0.943 / -5.953 | 1.622 | 1.621 | 23.750 | 23.625 |
| Technical half / fallback | +1.007 / -6.783 | +0.821 / -6.718 | 2.093 | 2.091 | 30.250 | 30.125 |
| Perpetual half / hold | +3.613 / -5.388 | +3.349 / -5.305 | 2.089 | 2.087 | 29.750 | 29.500 |
| Perpetual half / fallback | +3.379 / -6.160 | +3.144 / -6.090 | 2.455 | 2.454 | 34.375 | 34.125 |
| Technical reliability / hold | +3.750 / -5.695 | +3.531 / -5.641 | 1.364 | 1.361 | 19.625 | 19.375 |
| Technical reliability / fallback | +3.704 / -6.378 | +3.546 / -6.311 | 1.739 | 1.736 | 24.625 | 24.375 |
| Perpetual reliability / hold | +4.450 / -5.143 | +4.248 / -5.056 | 1.568 | 1.566 | 22.750 | 22.625 |
| Perpetual reliability / fallback | +4.208 / -5.854 | +4.039 / -5.779 | 1.953 | 1.952 | 28.250 | 28.125 |

## New policies by evaluation-start regime

Each cell shows base AlphaEx / MaxDDDelta followed by doubled costs in
parentheses, in points. Regimes use past information at the first evaluation
six-hour decision, not the subsequent quarter's return. Passing these means
is descriptive and does not mean every quarter or every possible trend passes.

| New policy | Bull (2) | Bear (4) | Sideways (2) |
|---|---:|---:|---:|
| Technical reliability / hold | +3.773 / -5.352 (+3.523 / -5.175) | +2.221 / -3.921 (+1.936 / -3.910) | +6.784 / -9.586 (+6.728 / -9.568) |
| Technical reliability / fallback | +5.763 / -7.310 (+5.590 / -7.194) | +0.447 / -4.220 (+0.260 / -4.164) | +8.161 / -9.761 (+8.072 / -9.723) |
| Perpetual reliability / hold | +4.063 / -4.749 (+3.777 / -4.548) | +2.895 / -3.967 (+2.660 / -3.904) | +7.949 / -7.890 (+7.895 / -7.867) |
| Perpetual reliability / fallback | +6.021 / -6.682 (+5.813 / -6.542) | +0.921 / -4.229 (+0.733 / -4.171) | +8.967 / -8.278 (+8.880 / -8.233) |

All4economic mean conjunctions pass, but all4predictive conjunctions fail
separately on both interval and evaluation. The predictive conjunction requires
strictly smaller mean MSE than zero, scale_mean, own full and own half in all
four summary strata. No gate or selection lock is relaxed after results.
The low individual-quarter success rate and deficient regime counts remain.

## Forecast losses versus all three registered endpoint controls

Positive percentage means MSE reduction relative to that control; negative
means worse MSE. These percentages are loss reductions, not economic points.
Scale-fit results are in the JSON and explicitly marked in-sample. The table
shows later intervals/evaluation only; neither segment changed a coefficient.

| New mean | Reference | Interval MSE reduction | Evaluation MSE reduction | Eval improved / equal quarters |
|---|---|---:|---:|---:|
| technical_reliability | scale_mean | -0.527% | -0.809% | 1 / 0 of8 |
| technical_reliability | technical_half | -0.159% | +1.063% | 5 / 0 of8 |
| technical_reliability | technical_scaled | +1.630% | +5.162% | 7 / 1 of8 |
| perp_delay0_reliability | perp_delay0_half | -0.110% | +0.304% | 4 / 0 of8 |
| perp_delay0_reliability | perp_delay0_scaled | +1.189% | +3.990% | 6 / 1 of8 |
| perp_delay0_reliability | scale_mean | -0.104% | -1.236% | 1 / 1 of8 |

Technical reliability has worse interval MSE than half by0.159%; perpetual is
worse by0.110%. Both lose to the anchor in aggregate in both later segments.
In evaluation technical beats the anchor in only1/8quarters; perpetual in1/8
with1tie (its zero-weight endpoint). Evaluation MSE versus zero is also worse,
by4.761e−6 technical /6.118e−6 perpetual in squared-log-return units. The
fit-mean reference and MAE are retained in the full summaries.

Technical reliability's equal-quarter evaluation rank IC is0.037392. The
perpetual aggregate rank IC is undefined because fold12 is constant; the other
seven quarter values remain available. An undefined rank is not replaced by
zero or silently averaged over fewer quarters. Shrinkage does not create a
new ranking signal.

## Mean drift explains the net excess loss algebraically

For each issued mean mu, let d=mu−a and r=y−a on the scored rows. The identity is

    MSE(mu)−MSE(a) = E[d²]−2E[d*r]
                  = [Var(d)−2Cov(d,r)] + [E[d]²−2E[d]E[r]].

The first bracket is the centered component; the second is the mean-drift
component. The following values are **squared log return ×1e6**, equal-quarter
means; they are not return points. A negative component favors the new forecast.

| New mean | Segment | Centered component | Mean-drift component | Total MSE minus anchor |
|---|---|---:|---:|---:|
| technical_reliability | scale | -1.927209 | +0.000000 | -1.927209 |
| technical_reliability | interval | -0.939076 | +2.891621 | +1.952545 |
| technical_reliability | evaluation | -0.228515 | +2.797574 | +2.569058 |
| perp_delay0_reliability | scale | -2.208115 | +0.000000 | -2.208115 |
| perp_delay0_reliability | interval | -0.637650 | +1.021983 | +0.384333 |
| perp_delay0_reliability | evaluation | -0.609196 | +4.534918 | +3.925722 |

The centered contribution is favorable for the learned weights on both later
segments; the mean-drift term more than offsets it. This is an accounting
identity for forecast losses, not a causal intervention isolating stale bias,
feature drift, changing expected return, or model misspecification. A centered
benefit does not prove that the conditional signal will transfer prospectively.

The original full forecast's evaluation excess MSE versus the anchor is
19.9995e−6 technical and17.2908e−6 perpetual; its drift components are17.4861e−6
and14.9228e−6. Raw full-forecast excess losses are smaller,11.4932e−6 and
9.9928e−6, but still positive. Thus the old scale-period bias correction does
not transfer favorably to these evaluation means. Dropping or retuning it after
seeing this result is not performed in this study. All9mean decompositions and
raw/scaled comparisons remain available, including adverse effects.

Scale/interval summaries stratified by the **later evaluation-start regime**
are retrospective groupings. That regime was not known at their own forecast
times; these groupings cannot establish causal regime-conditioned calibration
skill. Each record and summary marks that distinction. Calibration windows also
overlap across successive folds; pooled counts are not independent evidence.

## Validation, hashes and execution status

Full626tests passed in56.696s before coefficient fitting. The10helper tests
cover clipping, zero dispersion, exact endpoints/half, future-poison exclusion,
input guards and nonzero drift. Four runner tests cover registration, complete
and paired inventories, unequal-quarter versus pooled weighting, undefined
ranks and fail-closed regimes. The data-only source audit verifies1328ancestral
artifacts,16calibration files,40parent forecasts and16raw evaluation streams;
raw+saved_bias endpoint differences are0, source h24actual error≤3.96e−15.

Real run session54020 completed with exit0 and all8folds. It saved208new
artifacts:16weightJSON,16evaluation forecastNPZ,16calibrationNPZ,128targetNPZ
and32newtraceJSON, plus8fold manifests. Runtime introduced no warnings/errors
in its log. The completed namespace is
`codex_outputs/oracle_mean_reliability_decisions_v1`; do not restart it.

The [independent scalar audit](oracle_mean_reliability_evidence_20260906/independent_scalar_audit.json)
and [audit source](oracle_mean_reliability_evidence_20260906/independent_scalar_audit.py)
passed. They rehash all1328ancestral and208new artifacts, recompute all16S-only
weights/16evaluation and16calibration forecasts,216scores/decompositions,
40parent scores and96controls, and verify all32new own-state paths/11008
decisions plus256base/stress accounts. Weight, prediction, target, utility,
AlphaEx, DDdelta, turnover and cost differences are0; mean-exposure difference
is2.22e−16. All6observed endpoint policy paths match original targets/accounts.
The48unscored inference decisions and664fallback decisions remain. No canonical
helper, planner, simulator or scoring implementation is imported. The maximum
algebraic decomposition residual is9.32e−20. This proves saved-output
consistency, not generalization or a superiority probability.

A separate [Decimal60 summary audit](oracle_mean_reliability_evidence_20260906/independent_summary_audit.json)
and [source](oracle_mean_reliability_evidence_20260906/independent_summary_audit.py)
pass all128rows,216scores and16weights, with maximum summary difference
1.51e−15 and scalar score difference4e−18; fitted weight algebra matches exactly.
It verifies302direct/consumed files, including208new artifacts and25sources.
It rehashes56consumed ancestral files; the remaining ancestor chain is bound
through preflight, independently rehashed by the scalar/source audit above.
All45positive-weight group/segment rank comparisons preserve the parent ranks;
the3zero-weight segments in perpetual fold12 are constant. No new model,
coefficient choice, policy variant, p-value or confirmation data were added.

[Publication verification](oracle_mean_reliability_evidence_20260906/publication_verification.json)
binds the report, copied evidence,25registered sources and208new artifacts.

| Evidence | SHA256 |
|---|---|
| Frozen config | `71b0965691099074a690085ac22cbe6808b2b430fea305ec2b75480db1c2f094` |
| Data-only preflight | `d1304dc98df0595b68992b6605c78bd3527d3b811209071df12cb6594d294370` |
| Runtime registration file | `d15c59fe44489076fe332f0e0b4ee511723d19b312952dd20b0f684ae638bf97` |
| Results file | `333f88d4bc06f671d552d8ca70470ee60ecd67074812f6aa248b26f1b94562f1` |
| Independent scalar audit | `8d8f14bfd821ad83dab0eba44d53466ff2c6e900dfcd3debd4ab10ac67d5da7a` |
| Independent summary audit | `8bd15dbdddca697cc020208db44e1e117b8b3d8f66c53eb6c002a270b7b022d0` |

The embedded registration hash is canonical-content and differs from the
registration file's byte hash above. Tracked evidence preserves summary/results
and verification bindings; full binary inputs and raw traces remain in ignored
local directories and are needed to rerun the independent audits.

## Research consequence

[Gneiting–Resin](https://arxiv.org/abs/2108.03210v3) motivates separating
calibration from information content; [Dimitriadis–Puke's 2026 preprint](https://arxiv.org/abs/2603.04275)
connects linear recalibration and score components under assumptions not verified
here. Their inferential results are not applied to these nonstationary,
boundary-clipped reused data. [Smith–Wallis](https://doi.org/10.1111/j.1468-0084.2008.00541.x)
explains why estimating weights may fail to beat simple averaging. The observed
interval losses show that failure can occur in this procedure too.

The next development-only experiment should address **causal mean/forecast
centering adaptation**, with every update using already matured labels and
past forecasts, and with matched constant/rule controls. The decomposition
supplies a hypothesis, not a successful update rule. Do not search another
static weight or promote one of these4 variants from the current means. Keep
additional-test15–24 report-only, preserve162explored causal names and every
prior lock, and defer architecture optimization. No production or paid run,
new live-data receipt proof, independent confirmation or high-probability result
has been established. The goal remains active.
