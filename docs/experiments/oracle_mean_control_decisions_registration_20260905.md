# Mean forecast decision-value controls

This diagnostic is fixed before its new policy outputs. It follows133 adaptive
policy names on reused development validation. There is no model retraining,
new calibration choice, new confidence interval, winner selection, test/outer
tuning or deployment. Preserve all previous failed gates and selection locks.

## Question and exact seven means

The saved return forecasts still have larger MSE than zero/fitmean. Correcting
their mean can worsen MSE while improving some economic averages. The parent
mean-bias diagnosis also found drift between the correction period and later
validation. Test the economic value of learned, time-varying mean information
before adding architecture or another correction rule.

Hold the variance series fixed to `technical_scaled` for every arm. Use the
immutable delay experiment from source commit `a75b55a`, with its same common
support. The means are exactly:

| Identifier | Mean supplied to both decisions |
| --- | --- |
| zero | constant0 |
| fit_mean | common mean return on the parent's purged18month fit rows |
| scale_mean | mean actual return on the parent's purged3month scale rows |
| technical_raw | saved technical mean before its bias correction |
| technical_scaled | saved technical mean after its bias correction |
| perp_delay0_raw | saved additional-delay0 UM mean before bias correction |
| perp_delay0_scaled | saved additional-delay0 UM mean after bias correction |

`scale_mean` refers to the scale segment, **validation start minus6months
through minus3months**, with labels ending strictly before its end. It is not
the most recent three months. Do not include interval-calibration actuals,
unscored validation actuals, or future validation returns in either constant.
The arithmetic sample mean uses finite scale returns only, calculated with
scalar compensated summation. Save the exact value supplied to the policy.

The existing correction is
`mu_scaled(t) = mean(y_scale) + mu_raw(t) − mean(mu_raw_scale)`.
Thus scaled versus scale_mean directly compares the time-varying component
with that historical drift anchor. It does not isolate all economic causal
effects because each policy develops its own inventory trajectory.

Both technical_raw and perp_delay0_raw now use the fixed scaled technical
variance. They are new mean/variance combinations. Only technical_scaled
point andutilityrisk1 must reproduce the parent delay experiment exactly.
No15/60minute variant is chosen from the favorable delay outcomes.

## Frozen support, timing and provenance

Use original folds5–12, 2021-04-16 13:45 UTC through2023-04-16 13:45 UTC
exclusive. The source chronology remains18months fit,3months scale,
3months interval calibration,3months validation. Six-hour forecast horizon
(24bars), next-bar-open entry and the original final mark remain fixed.
Scale labels must end strictly before the scale boundary; all source masks
must match the pinned parent preflight before a policy is computed.

The config pins the parent registration, results and preflight SHA-256,
32 source forecasts and16 calibration files. Verify the parent config and
source helpers, complete496-file artifact manifest, all calibration calendars,
matching outcomes and masks, common fitreturnmean and Spot execution-data proof.
Record the new runner/helper hashes, current commit, parent source bindings,
data proof and runtime versions before policy outcomes. Mean-source and fixed
variance-source provenance accompany the output forecasts and policies.
Raw UM provenance is pinned through the parent preflight and immutable
forecast artifacts; this frozen-forecast runner does not read UM bars or
recompute UM features. The Spot execution-data proof is revalidated directly.

All7 means retain2,586 inference and2,574 scoring rows of2,920 scheduled
decisions. Forecast scoring masks never suppress an otherwise admissible
order. Keep the original complete15minute calendar and missing intervals.
The inherited retrospective common mask includes undelayed UM availability;
it is not an operational delayed-feed/outage policy. Archive receipt/version
provenance remains unestablished. The2bull/4bear/2sideways quarters cannot
pass the unchanged minimum3quarters-per-regime gate.

## Decisions, controls and output inventory

Each mean feeds exactly the existing point mapper and the existing conditional
utility controller with risk aversion1 and cost multiplier2. The fixed variance
is identical point by point for all means, including constants. Keep canonical
cash/units accounting, initial B&H inventory, exposure intent range[.5,1.12],
maxstep.08,deadband.01,one-wayfee.00055,annualborrow.10 and next-open execution.
Generate targets under base costs once; replay identical targets under base
and doubled fees/borrowing, without replanning for stress costs.

Copy/replay B&H and common_robust on the same support. Zero-mean point should
match B&H economically: its target1 intents differ from B&H's no-target
intent coverage. Zero-mean utilityrisk1 can reduce risk and is not defined as
a B&H duplicate. Check technical_scaled target and economic parity with the
parent on both costs. Retain the original parent target/trace proof.

Inventory:7means×2policies=14 named policy variants, plus2controls, across8
folds:128 economic rows/targets,56 derived mean/variance forecast NPZs,
56 utility traces and56 return-score rows. Fitmean andscale_mean are the only
new sample-statistic calculations; no predictor is refit. Several names are
deliberate economic duplicates, not independent evidence. Immutable saved
forecast arrays, targets, traces and completed fold results must match before
reuse; a running process is not restarted merely because observation timed out.

## Complete comparisons and reporting

Save all21 unordered mean-pair comparisons in the fixed order above, later
identifier minus earlier identifier, each for both decisions and both costs.
The primary questions are learned means versus zero/fitmean/scale_mean,
perp_scaled minustechnical_scaled, perp_raw minustechnical_raw, and each
scaled minusits raw mean with variance held fixed. Report all/bull/bear/sideways
equal-quarter economic differences, turnover,trades,fees andborrowing.
No pair is dropped because of its result.

Return diagnostics are MSE, MAE, sign accuracy and rankIC. RankIC is undefined
for constant forecasts. Retain zero/fitmean MSE references, per-quarter scores
and both equal-quarter and pooled-row aggregate losses. Relative loss reduction
is `1 − mean_q(L_candidate,q)/mean_q(L_reference,q)`, not a mean of quarterly
percentage improvements. No new risk-forecast comparison is needed because
every arm uses the exact same variance source.

Economic summaries are descriptive point estimates. Keep all failed outcomes,
unchanged both-cost joint signs and minimum regime counts. No DM/SPA test,
selection-adjusted confidence, causal economic attribution or high-probability
generalization claim follows from these reused eight quarters. Subsequent
adaptive calibration would require its own registration before new outcomes.

## Primary-source context

[Campbell–Thompson (2008)](https://doi.org/10.1093/rfs/hhm055) compares predictive
equity-premium regressions with the historical average and distinguishes small
forecast gains from mean-variance investor value. Its equity restrictions and
economic results are not evidence for these BTC6h policies.
[Welch–Goyal (2008)](https://doi.org/10.1093/rfs/hhm014) documents instability
and weak out-of-sample performance in many equity-premium predictors, supporting
the need for explicit simple controls rather than assuming feature complexity
adds decision value.

[Pesaran–Timmermann's author-hosted original](https://rady.ucsd.edu/_files/faculty-research/timmermann/estimation-window.pdf)
analyzes estimation-window bias/variance under structural breaks. A shorter
or post-break-only sample is not automatically best; unknown breaks increase
uncertainty. This motivates a later fixed causal adaptation/shrinkage comparison,
not choosing a shorter correction window from already observed BTC results.
