# Stage13 research note: fixed past-only reliability slope

2026-09-06. Proposal and source review only; no new fits, coefficients, forecasts,
policy rollouts, outcome comparisons, additional-test reads, or repository edits.
The study must be separately registered before any new numerical outcomes.

## One bounded question

Can a single convex-combination weight, estimated only on the existing past
scale segment, retain useful time-varying information in each frozen technical
and perpetual Ridge return forecast better than its scale-period mean anchor?
Keep both streams and both existing missing-input rules. This is a new fixed
calibration procedure, not a reopened validation-selected 0/.5/1 weight grid.
No new indicators, horizon, model architecture, feature selection, feed delay,
base-model fit, regime boundary, or additional-test selection is proposed.

Registrations reviewed: original gap-aware ML, Oracle frontier, risk
calibration, derivative ablation/delay, mean controls, fixed half shrinkage,
and matched information decomposition. Earlier work already covered 7/30-day
logistic and Ridge/HGB recipes; 6h Ridge/HGB technical/flow ablations; static mean
bias and variance calibration; intervals; fixed amplitude .5; missing-input
rules; and hindsight outcome substitutions. A past-only learned slope followed
by a separate chronological reliability assessment is the narrow remaining
question here. It tests information already compressed into the frozen Ridge
means, not all learnable information in their 29/31 raw feature inputs.

## Exact numerical contract accepted by root

For a quarter with validation start T, retain the existing 18-month base fit
[T-24 months,T-6 months), scale [T-6 months,T-3 months), interval
[T-3 months,T), and three-month validation. The interval segment is a separate
report-only evaluation of the new slope; it cannot select or update that slope.
It has already been researched, so it is not untouched confirmation.

On the source's exact purged scale rows S, set:

- p_i = raw_mu_i + SAVED mean bias, reproducing the original full forecast
  arithmetic; a = the exact saved scale-mean anchor.
- d_i = p_i - a; r_i = y_i - a.
- B = mean_S(d_i^2); C = mean_S(d_i*r_i).
- w = 0 if B=0 or C<=0; w = 1 if C>=B; otherwise w=C/B.
- On every future inference row, issue m_i = w*p_i + (1-w)*a, with exact
  endpoint branches for w=0/1. Preserve NaN outside the existing inference mask.

This minimizes scale squared loss over the registered anchor/full segment,
up to numerical arithmetic. It avoids assuming that centered quantities sum
exactly to zero in floating point. Preserve and report the tiny residual
mean(d) and mean(r); do not silently replace this formula with raw-centered
OLS or recalculate the saved bias/anchor. Nonfinite inputs or moments are a
validation failure; do not create a numerical fallback or parameter search.

There are 16 newly estimated calibration coefficients (two streams x eight
quarters), zero new base-model fits, and four new policy names (two streams x
two rules) plus 12 unchanged controls. Calling the stage simply 'no fitting'
would obscure that calibration coefficients are estimated.

The target remains y_t = log(close[t+24]/open[t+1]). Its 6h holding outcome is
known at t+25 bars; fit/calibration labels must end strictly before their
boundary. All existing common-feature, inference, score and missing-price masks
must be reconstructed and hash-matched. Scored future support never cancels
inference or orders. Do not use stage12 hindsight paths as training teachers.

## Falsifiable comparison and interpretation

For both streams, report the frozen new forecast against the own full, own
half, scale anchor, zero and common purged-fit mean. The scale loss improvement
is mechanically ensured by fitting w on S and is not a success endpoint.
Report subsequent interval and validation losses separately, with equal-quarter
MSE primary; MAE, forecast variance, mean bias, covariance, rank IC and all
per-quarter/start-regime values are descriptive companions. Do not pool the two
segments, compress time gaps, count overlapping fold histories as independent,
or declare a high-probability pass from the reused 2bull/4bear/2sideways sample.

A small, falsifiable question is whether each frozen slope reduces MSE versus
the scale anchor in BOTH subsequent segments. Failure in either segment rejects
that stream's proposed stable incremental information claim for this probe.
Improvement only versus full/half but not the anchor is shrinkage damage control,
not demonstrated incremental dynamic information. Preserve every comparison;
a favorable stream is not automatically selected or promoted. Existing stronger
regime/economic/confirmation requirements remain unchanged.

The four new policies use their own inventory, fixed technical variance,
existing utility risk1/cost multiplier2, 6h clock, next-bar fill, costs and both
missing-input rules. Reproduce the old 12 controls. Replay the same base intents
under doubled costs. Forecast improvement and AlphaEx/MaxDDDelta improvement are
separate endpoints: lower squared error cannot guarantee better trading after
thresholds, costs, inventory feedback and drawdowns.

## Exact centered and drift decomposition

For either later segment, use exactly its shared score rows and averaging
weights. With d=p-a and r=y-a, the identity relative to the anchor is:

    MSE(a+w*d) - MSE(a) = w^2 E[d^2] - 2w E[d*r].

Using population-form moments (divide by n, not n-1), expand it as:

    centered dispersion term = w^2 Var(d)
    centered covariance term = -2w Cov(d,y)
    mean-drift term = w^2 E[d]^2 - 2w E[d]*(E[y]-a).

Their sum must reproduce the paired loss difference within fixed numerical
tolerance. Dropping the mean-drift term can falsely attribute anchor staleness
to forecast dispersion or discrimination. Average identities quarter by quarter
with equal weights; do not substitute pooled moments with different weighting.

Same-segment OLS slopes or nonnegative score-decomposition components, if shown,
are descriptive fitted diagnostics only. They must never become replacement
forecasts or evidence of causal learnability. Positive past covariance that does
not transfer suggests unstable calibration/information; weak covariance in the
saved mean cannot distinguish a poor Ridge representation from absent signal
in the full features. If this single probe fails, further weight searches on the
same quarters are not justified as confirmation.

## Primary sources read and applicability limits

1. Gneiting and Resin (2023), *Regression Diagnostics meets Forecast Evaluation:
   Conditional Calibration, Reliability Diagrams, and Coefficient of
   Determination*, Electronic Journal of Statistics 17, 3226-3286.
   https://arxiv.org/abs/2108.03210v3
   Published DOI: https://doi.org/10.1214/23-EJS2180
   Sections3.3-3.4 distinguish miscalibration/discrimination/uncertainty and
   in-sample diagnostics from out-of-sample forecast skill. The paper motivates
   separating these questions; the proposed fixed convex slope is not its
   isotonic algorithm and inherits no calibration or generalization guarantee.

2. Dimitriadis and Puke (2026), *Statistical Inference for Score Decompositions*,
   arXiv preprint v1, submitted2026-03-04.
   https://arxiv.org/html/2603.04275v1
   Linear recalibration connects forecast-score components to classical
   Mincer-Zarnowitz regression. Assumption3.1 imposes strict stationarity, a
   linear conditional target, and an interior true parameter; asymptotic results
   require further dependence/moment conditions. These conditions are not
   established for the reused BTC quarters; clipping w at0/1 also introduces
   boundary cases. Do not import its p-values, normal limits, or inference claims.
   Our finite-sample algebra above requires none of those statistical guarantees.

3. Smith and Wallis (2009), *A Simple Explanation of the Forecast Combination
   Puzzle*, Oxford Bulletin of Economics and Statistics71, 331-355.
   https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0084.2008.00541.x
   The paper explains why finite-sample error in estimated combination weights
   can offset their theoretical benefit relative to simple combinations. It
   supports retaining fixed full/half/anchor controls and evaluating weights on
   later data. It does not prove this slope optimal, stable, or superior for BTC.

No source establishes that future returns revealed by stage12 are predictable
from the available technical/Spot-perpetual features, nor that current model
architecture is or is not the limiting factor.
