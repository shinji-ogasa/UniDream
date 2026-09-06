# Scale-only return reliability registration

Freeze this protocol, new source/tests, config and data-only preflight before
estimating any new calibration weight or computing its predictions/losses/orders.
The matched Oracle study is complete; it suggested value in return information
under the current controller, without proving that available features contain
that information. This study tests one new causal calibration procedure on the
existing technical/perpetual Ridge forecasts. It changes no feature, base-model
architecture, fit history, risk forecast, action rule, or earlier selection lock.

## Fixed family and question

Can a reliability weight estimated only from the original scale period improve
out-of-period return losses and economic outcomes versus the existing constant,
full and half forecasts? Is weak MSE associated with unhelpful forecast variation,
mean drift, or both? A favorable scale-fit result is mechanical and cannot count
as success. A favorable original-development result is an exploratory candidate,
not a new confirmation result.

Use original validation5–12 (original test4–11), April2021–April2023, with the
existing cutoff `2023-04-16T13:45:00Z`. No additional-test15–24 outcome is used.
All data are reused. Keep all 12 causal controls and add exactly 4 new causal
policy names: technical/perpetual reliability means, each under the original
hold and target-one fallback rule. The adaptive causal-name ledger increases
from158 to162. The old frozen candidate family remains unchanged.

The forecast inventory is9 means: scale_mean, technical/perpetual raw,
technical/perpetual scaled, technical/perpetual half, technical/perpetual
reliability. Score all9 on scale, interval and evaluation, separately, for all8
folds: **216 score records**. Raw forecasts are scoring diagnostics only; no new
raw policy is introduced. Evaluate16policies ×8folds =128economic rows, each
under base and doubled costs. Fit16scalar calibration weights, with **0 base
model refits**. Do not call this a wholly no-fit experiment.

## Chronology and source contract

For evaluation start E, frozen base estimators see fit [E−24m,E−6m). The only
weight-fitting segment is scale S=[E−6m,E−3m). The subsequent interval
I=[E−3m,E) is diagnostic-only for this new mean; it neither selects nor changes
the weight. Evaluation is [E,E+3m). Scale/interval labels must mature strictly
before their segment end; evaluation labels mature no later than its end.
The h24 return is `log(close[t+24]/open[t+1])`, available at t+375minutes.
Features are already shifted to use bars through t−1; decisions remain at UTC
six-hour times. Do not update weights within a quarter or pool across folds.

Reuse the completed full-procedure parity's16calibration NPZ and40forecasts;
read the original delay run's16raw evaluation streams directly, without
subtracting a bias from rounded forecasts. All are within the verified1328
ancestral artifacts. The fixed existing data-only prepare function verifies
that lineage and original complete-grid supports without running Oracle or
fitting any model. Bind its config and every imported source module in the new
config. Reconstruct calibration actuals against the original cutoff Spot data.

Both groups have scale counts233,279,221,215,361,359,354,367;
interval279,221,215,361,359,354,367,367; evaluation score
221,215,361,359,354,367,367,330. These are fold-indexed repeated supports, not
independent counts across overlapping calibration/quarter windows. Inference
2586 and score2574 remain distinct; retain332fallback opportunities and2missing
current-open rows. Neither interval/evaluation outcomes nor a zero coefficient
may alter inference or order availability. The inherited loader decodes the full
Spot parquet before slicing; no modeling, label calculation or scoring uses
additional-period rows. The no-additional-test flag refers to this semantic
scope, not to a claim that later file bytes were never read.

## Exact weight and prediction arithmetic

For each group and fold, let a be the **exact saved constant scale mean**, and
p_i=raw_mu_i+saved_bias be the existing full scaled endpoint. The bias remains
the prior mean scale residual; no extra intercept is fitted. On S define

    d_i = p_i - a
    r_i = y_i - a
    B = mean_S(d_i**2)
    C = mean_S(d_i*r_i)
    w = 0 if B == 0 or C <= 0
        1 if C >= B
        C/B otherwise

All selected inputs must be finite. B==0 is an unidentified zero-dispersion
case, explicitly recorded; no arbitrary variance floor is added. Branch before
division to avoid unnecessary overflow. Require at least64 scale rows. Reject
invalid input instead of changing the support, floor, weight or candidate set.
The supplied anchor must exactly match the same `math.fsum(y_i/n)` scale mean.

This is the least-squares point on the fixed convex segment from anchor to full
forecast, not a search choosing among the old0/.5/1grid. It is also not exactly
raw-centered unconstrained Mincer–Zarnowitz regression: floating-point
`mean(y−raw)` differs from `mean(y)−mean(raw)`. Preserve the existing full
endpoint arithmetic and report mean(d) and mean(r), including their rounding
residuals. Do not silently change the bias or recenter evaluation values.

Apply w=0 by copying the exact saved anchor, w=1 by copying the exact saved full
forecast; otherwise compute `w*p+(1−w)*a`. At w=.5 this matches the existing
half arithmetic. Predictions outside the existing inference support remain
NaN. Every strictly positive w preserves within-fold order apart from possible
floating-point ties; w=0 is a constant with undefined rank IC. A collapse onto
the anchor is not discovery of new information. Do not reuse old interval
widths to claim that the new mean has calibrated uncertainty.

## Drift-aware prediction diagnostics

For every scored mean mu and scoring set T, relative to the same saved a, let
`d=mu−a`, `r=y−a`. Save the exact decomposition

    MSE(mu) − MSE(a) = E_T[d²] − 2 E_T[d*r]
                    = Var_T(d) − 2 Cov_T(d,r)
                      + E_T[d]² − 2 E_T[d] E_T[r].

Report the centered component and the mean-drift component separately, plus
identity residuals. Use population moments and equal-quarter means; separately
label pooled-row MSE. Do not drop the mean-drift terms or compute a new favorable
slope on interval/evaluation for use in a policy. No hindsight-optimal slope,
isotonic fit, bin search, threshold search, new feature ranking or probability
estimate is needed here. Mean-squared error targets the conditional mean;
MAE/sign/rank IC remain descriptive companions, not interchangeable objectives.

Compare each new mean with scale_mean, its own full mean and its own half mean,
for every segment and every evaluation start regime. Evaluation rows use the
regime known at their first scheduled decision. Scale/interval rows grouped by
that later evaluation regime are explicitly retrospective groupings, not
conditions known at their own forecast time. They cannot establish causal
regime-conditioned calibration skill. Report all signs and failures. For
policy value use the same paired comparisons under both missing-input rules,
with AlphaEx, MaxDDDelta, turnover and trades. Positive AlphaEx/negative DDdelta
are the desired economic direction; a causal improvement claim also needs
out-of-period predictive superiority over the constant and half controls.
Save separate interval/evaluation direction flags requiring strict lower MSE
than zero, scale_mean, own full and own half in all four summary strata. The
interval regime groupings retain the retrospective caveat above; neither flag
selects or promotes a policy. Save zero/fit-mean baseline losses and their
differences as well as MAE. Require the registered2/4/2 regime inventory instead
of returning zero for an empty group. MSE reduction on S, reduced turnover
alone or rank-preserving shrinkage alone
cannot establish that technical indicators have useful return information.

## Economic replay, artifacts and validation

Use the exact shared technical variance and existing local conditional utility
(risk1, cost allowance2), own evolving cash/units, initial B&H inventory,
exposure intents[.5,1.12], maxstep.08, deadband.01, fee.00055, annual borrow.10.
Fallback forces target1 at unavailable-forecast decisions with known current
open; missing current open cannot order. Fill only at the immediately following
bar's open when observed; skip a missing fill without rollover. Preserve all
price bars and borrowing across gaps. Doubled costs replay the same base intents
without changing weights or planning again. No live trades or deployment occur.

Save per fold2weightJSON,2new evaluation forecastNPZ,2calibrationNPZ,
16targetNPZ and4new traceJSON:26artifacts ×8=208, plus8fold manifests and the
registration/preflight/complete results. Retain all128rows/216scores/16weights.
Parent controls must reproduce exactly. Existing raw/full/half/anchor scoring
must be reconstructed on matched masks. Immutable partial artifacts compare
exactly; completed output rejects re-execution. Never restart a live run merely
because observation times out.

Tests must cover negative/positive clipping, interior weight, zero dispersion,
exact endpoint/half arithmetic, selected-only labels, poison outside support,
nonzero mean drift, accounting/summary inventory and strict config. Run
`uv run python -m unittest discover -s tests -v` and `git diff --check` before
new real weights. Independently check scalar weight/moment/forecast arithmetic,
all new own-state decisions and all base/stress accounts after completion.

## Evidence limits and primary literature

[Gneiting and Resin](https://arxiv.org/abs/2108.03210v3) distinguish calibration,
discrimination and uncertainty; score decomposition is not a demonstration of
future skill. [Dimitriadis and Puke (2026 preprint)](https://arxiv.org/abs/2603.04275)
connect linear recalibration and score components. Their stationarity,
conditional-functional and interior-parameter assumptions are not established
for these reused quarters and boundary-clipped weights; no proposed test,
asymptotic normality or p-value is transferred here.
[Smith and Wallis (2009)](https://doi.org/10.1111/j.1468-0084.2008.00541.x)
explain why estimated combination weights can lose to simple averaging through
estimation error. Keep that failure possible rather than assuming learned
weights must outperform fixed half.

Original start-regime coverage remains2bull/4bear/2sideways, below the required
three per regime. Scaled predictions also rely on retrospective availability,
not authenticated live receipt evidence. Calibration/evaluation dates overlap
across successive folds; adaptive research and dependent errors remain.
No strongest-model, all-trend high-probability generalization, independent
confirmation, formal P1 result or automatic promotion follows this study.
