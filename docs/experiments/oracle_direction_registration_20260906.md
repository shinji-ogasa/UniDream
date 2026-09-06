# Stage17: fixed causal direction replacement registration

Commit and push source, tests, configuration, this protocol, research note and
data-only preflight BEFORE any new real binary labels, class priors, weighting
statistics, classifier fits, probabilities, mapped forecasts, losses or orders.
Stage16 report c348a6a is complete: supplied future signs changed the fixed
controller more than supplied future magnitudes. This next experiment asks
whether four fixed linear direction fits can produce useful causal information.
It is not an architecture or feature search, and not Oracle-label distillation.

## Fixed data and original parents

Use the original reused validation5–12 (April2021–April2023), original Technical29
and Perp-delay0 31 columns, original technical_half/perp_delay0_half forecasts,
and shared calibrated technical risk. Do not select Stage15 added features or
Stage13 weights. Bind Stage16 source revision b44c211dccc38f719b6f893a95c0d1a2d4cbf638,
its 36 completed policies and all2,120 ancestral artifacts; verify29 source files.

Semantic data cutoff is strictly before2023-04-16T13:45Z. The inherited Spot
Parquet loader decodes its file before slicing; no later semantic modeling is
not a claim that later bytes were not decoded. Original test(f)=validation(f+1):
these development quarters are also old test4–11 and are repeatedly reused.
No additional-test15–24 labels/modeling/scoring/selection; no outer/P1 execution.

For E start, retain the original18month fit [E−24months,E−6months), then S3months,
I3months, E3months. Fit/S/I labels must mature strictly before the segment end;
E score labels no later than its end. Feature decision t uses completed data
through t−1; h24 log target is log(close[t+24]/open[t+1]), with all24 bars required
and375minute label maturity. Retain all six fit/scale/interval/predict/inference/
score masks and all inherited feature-availability requirements, including unused
derivative/delay/variance dependencies. No imputation, row deletion or narrower
selected-column availability rule. Fit counts800/1034/1313/1500/1503/1634/1672/1794.
E inference2,586 and score2,574; keep12 unscored inference origins. There are332
fallback opportunities and2 missing-current-open scheduled origins. E-start
regimes are2bull/4bear/2sideways, defined from trailing information.

## Four models and two unique fit priors per fold

Fit exactly Technical29/Perp31 × ordinary/magnitude weighted logistic loss.
Use original fit row column0 only, D=1{Y>0}. Ordinary sample weights are ones;
magnitude weights are abs(Y_i)/math.fsum(abs(float(Y_j))/n for j in fit).
This is the actual frozen floating arithmetic, not numpy.mean normalization.
Both have mean weight approximately1, with actual sums saved. Each StandardScaler
uses its defaults on the unweighted fit rows; only the classifier receives
sample_weight. Save both weight vectors, binary labels, fit timestamps/positions,
exact fit/predict feature matrices and predict positions/timestamps.

LogisticRegression: C1, l1_ratio0 (L2, omit deprecated penalty), lbfgs, tol1e−8,
max_iter1000, fit_interceptTrue, random_state20260906, all other pinned1.8.0 defaults,
threadpool limit2. Pin numpy2.2.6, pandas2.3.3, sklearn1.8.0. Its normalized objective
is [sum(w_i*logloss_i)+||beta||²/(2C)]/sum(w); intercept is unpenalized. Solver uses
numpy.sum(w), gtol1e−8, ftol64*float64epsilon, maxls50. Save complete estimator
parameters, scaler/coefficient state, masks, column order and matrix hashes.

Fail without retry for selected nonfinite/unsupported values, fewer than512 fit
rows, no prediction rows after fit, nonpositive/nonfinite mean absolute return,
missing positive effective weight in either class, ConvergenceWarning, n_iter
at or above1000, nonfinite fitted state/prediction, or invalid probability simplex.
RuntimeWarnings remain visible in the run log; they do not authorize ignoring
nonfinite output. Before accepting every model, independently recompute with
Python scalar transforms and math.fsum: finite normalized objective and gradient
infinity norm≤1e−6, every predict logit difference≤1e−12, probability difference≤1e−14.
The gradient tolerance is fixed at100*gtol to allow the separately fixed ftol
stopping route; it is not relaxed after outcomes. This checks stationarity and
predictor arithmetic, not scientific generalization. No solver/iteration retry.

Estimate ordinary prior pi=sum(w*D)/sum(w) and magnitude prior similarly, using
scalar fsum on their respective selected fit weights. There are TWO shared prior
estimates/fold, not four independent priors. Freeze prior logit as
math.log(pi)-math.log1p(-pi); score its stable sigmoid. Save exact values and both
solver/fSum weight totals. Descriptively report fit weight maximum, zero weights
and sum(w)^2/sum(w²); this is not an independent sample count under dependence.

## Mapping and fixed execution

For each classifier and each same-group fit-prior control, use
mu_new=sign(logit)*abs(own_frozen_half_mu) on EVERY original inference origin.
The four learned and four prior mapped means create16 new causal policy names
with two existing missing rules; the cumulative adaptive ledger is174→190.
Keep all36 Stage16 controls:12 causal and24 hindsight. Total52policies×8=416rows,
832 base/stress accounts. Old hindsight diagnostics remain future-informed and
are not new fits, teachers or causal baselines. Do not rerun finite RL search.

Exact zero logit maps to zero mean; binary accuracy uses logit>0 and treats a tie
as nonpositive. Y=0 is nonpositive for ordinary loss but has zero magnitude
weight. Parent mu=0 suppresses the mapped mean regardless of direction. Save
zero counts; use no epsilon. abs(parent_mu) is the absolute mean forecast, not
E|Y|; the result is a fixed surrogate mean for the controller, not calibrated
E[Y|X]. There is no probability/mean/scale/variance recalibration or new risk fit.

Raw classifier predictions may cover original S/I/E predict support. They are
never scored or calibrated on S. Reconstruct original I half magnitudes from
bound raw forecast, frozen S return bias and scale mean: .5*anchor+.5*(raw+bias),
only from I start onward. Earlier S mapped means stay NaN, because their S-fitted
bias/anchor were not available then. I mapped support is predict_mask AND time≥I
start, independent of whether its future label is scoreable. E mappings also
replace all unscored inference origins. Masked actual labels do not gate orders.

Keep original UTC6h clock, B&H initial inventory, next-bar open fills, own cash/
units, one-way fee0.00055, annual borrowing0.10, risk penalty1, cost allowance2,
intent bounds[.5,1.12], maximum step.08 and deadband.01. Hold submits no unavailable
mean order. Fallback submits target1 only with current known open and missing
prediction; no immediate inventory reset. Missing current open means no order;
missing immediately-next open skips fill without rollover. Borrow continuously
across gaps. Each policy uses its own state. Stress replays base targets at twice
fee and borrowing, without a new optimization. No outcome-derived tail action.

## Fixed scoring and nonselection gates

Save six classifier streams (four learned +two shared priors), I/E separately:
96 records total. All report ordinary Brier/log loss/binary accuracy, |Y|-weighted
versions of those three metrics, signed-return mean and weight denominator.
Signed return mean is descriptive and uncosted, not controller PnL. Stable
logloss is logaddexp(0,z)−D*z without probability clipping. If total absolute
realized return is zero, weighted metrics are null; retain that quarter and
make its equal-quarter aggregate/contrasts null, never silently discard it.

Score ten mapped means (two old halves +eight new) on I/E:160records, using
original return MSE/MAE/binary-sign accuracy/rank IC and zero/fitmean MSE controls.
Reproduce old E half scores and every old economic account. Aggregate each
metric by equal quarter for all/bull/bear/sideways. Separately retain pooled-row
MSE. I strata use E-start regimes retrospectively; they were not known at I
scored decisions. Undefined rank means remain null.

For every learned classifier, save paired differences to BOTH fit priors and its
same-group counterpart on both ordinary and weighted probability metrics.
The ordinary matched-probability gate requires lower ordinary Brier AND logloss
than its ordinary prior in every stratum in each segment. The magnitude gate
uses weighted Brier AND weighted logloss against its weighted prior. No choosing
which loss or segment looks favorable. Save mapped-MSE differences and counts
against own half and same-weighting same-magnitude prior; flag whether all strata
also beat zero and fitmean MSE. Prior controls cannot pass a strict self-contrast.
Save paired economic metric differences against both own half and matched prior.

Economic sign flag requires equal-quarter AlphaEX>0 AND MaxDDdelta<0 in all four
strata at both costs; retain strict joint-success quarter counts. Probability
skill, mapped-return skill and economic signs are separate flags. None establishes
high-probability generalization on these reused2/4/2 quarters. No selected winner,
p-value, bootstrap/significance, iid confidence, promotion or architecture tuning.
The formal P1 results_observed=false boundary remains unchanged.

## Outputs, audits and completion discipline

Save90 artifacts/fold:4model joblibs,1fit-data NPZ,1fit-provenance JSON,8Eforecast
NPZ,8S/I prediction NPZ,52target NPZ,16new trace JSON;720total. Save8fold manifests,
registration, preflight, result and complete stdout/stderr log. Bind all artifacts.
A completed result refuses rerun. A terminal partial attempt may be replayed in
full only with exact existing-artifact checks and the same immutable source;
never restart a live process merely because observing it timed out.

Run full `uv run python -m unittest discover -s tests -v` and `git diff --check`
before real fits. Synthetic tests cover labels/weights, poison/chronology,
normalization, scalar stationarity failures, extreme logits/zero weights, all
inference mapping, inventory/null/gate semantics. Independent audits verify
source contracts, fitted probabilities and scores, 128 new own-state paths,
832 accounts and all summary contrasts/report cells. No live trading, paid
compute, production change, external messages, automation or prospective claim.

[Research note](oracle_direction_research_20260906.md) derives the weighted
population optimum q=E[|Y|1{Y>0}|X]/E[|Y||X], so sign(q−.5)=sign(E[Y|X]) when the
denominator is finite and positive. Finite regularized linear fits need not
recover it. Weighted q is not ordinary up probability. The [weighting paper](https://hunch.net/~jl/projects/reductions/costing/finalICDM2003.pdf)
and [proper-scoring paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
justify the criteria's interpretation, not BTC investment performance. The
[sklearn1.8 objective](https://scikit-learn.org/1.8/modules/linear_model.html#logistic-regression)
fixes weight/regularization semantics, not C1 optimality.
