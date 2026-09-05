# Frozen procedure parity registration

Register the new fixed fitting/calibration adapter and exact replay contract
before any real-data refit. This stage tests computational reproducibility on
already reused validation folds5–12. It does not add a candidate, weight,
feature, architecture, economic rule, evaluation quarter, or statistical claim.
The old half candidate family and all earlier selection locks remain unchanged.

## Same intervals, explicit new calendar

Old validation fold f maps to the new procedure calendar test fold f−1, for
f=5..12 only. These are the same [2021-04-16 13:45 UTC,2023-04-16 13:45 UTC)
intervals already observed. The test-named calendar helper is not permission
to score an unseen test or to relabel earlier selection data.

Use the old delay data and full common-feature dependency set. Reconstruct
common availability from trailing variances24/96/672/2880, all old flow and
all derivative groups, plus technical/perp0/1/4 input availability. Never infer
feature support from a union of label-dependent training/scoring masks.
The data-only preflight independently rebuilds masks and compares every fit,
scale, interval, predict, inference and score array to the old procedure,
including full timestamp and saved inference/scoring arrays. Old delay
preflight contents, Spot/UM data proof, source regime and execution agree.

## Fixed numerical operations

The new helper fits two StandardScaler+Ridge(alpha100) means, technical and
perp_delay0, and one technical HGB log-variance model. HGB parameters stay
100 iterations,7 leaves,minimum64 samples/leaf,learning_rate.04,L2=10,
no early stopping,seed20260905,threadpool limit2. The unused perp variance
model is not needed for a mean bias or the shared technical variance.

Fit/scale/interval chronology remains18/3/3 months before each evaluation
start. Minima512/64/64, horizon24 bars,375-minute maturity and existing strict
fit/calibration versus inclusive score boundaries remain fixed. The helper
accepts only explicitly selected fitting/calibration labels; inference support
is never filtered by evaluation labels. The caller proves time causality.

Fit mean uses the original np.mean; mean bias uses np.mean(scale actual−raw
prediction). Scale anchor uses the existing constant_means/math.fsum arithmetic,
not a substituted np.mean. Raw variance is exp(clip(logvar,log(1e−12),0)),
then the original calibration multiplier and floor. Half is exactly
.5*saved-scale-period-anchor + .5*own-scaled-prediction. Both raw and scaled
technical interval quantiles are reproduced, not retuned.

## Complete outputs and comparison limits

For each quarter save3 models,5 mean forecast NPZs,2 calibration NPZs plus
one calibration/provenance JSON,12 target NPZs,10 utility trace JSONs:
33 artifacts per quarter,264 total. Quarter manifests and top-level
preflight/registration/results are additional metadata files.

The five means are scale_mean,technical_scaled,perp_delay0_scaled,
technical_half,perp_delay0_half. The twelve policies are each of these under
utility_risk1 hold and utility_risk1_fallback_bh, plus B&H/common_robust.
B&H uses all-NaN intents with inherited initial B&H units. common_robust is
built on the full causal history then sliced/masked. Each utility path uses
its own inventory; fallback targets are not deleted by inference masking.
Stress2x replays the same base intents with doubled cost/borrowing.

Compare all40 forecasts/scores,96 target series/economic rows,192 accounting
cost objects,80 full utility traces and all16 calibration series plus8 scalar
calibration/provenance records. Endpoint forecasts come from the old mean
stage, half forecasts from the old half stage. All96 targets are in the half
stage; utility traces span mean/fallback/half ancestry. No unused perp risk
array is claimed reproduced. Model serialization is recorded as provenance;
these three models need not be byte-identical to old differently packaged
artifacts merely to establish numerical prediction equivalence.

Exact checks: timestamp int64 arrays, mask booleans, NaN positions, scored
actuals, all submitted targets, identifiers, counts, reason codes and field
inventories. Forecast/calibration-array tolerance rtol1e−12/atol1e−14;
score/account/trace numeric tolerance rtol1e−12/atol1e−12. Report maximum
absolute differences; do not enlarge tolerances after a mismatch. A numerical
forecast tolerance does not relax target equality. Nonfinite-support changes
or dropped fields fail even if aggregate performance is close.

Resume only a completed fold with the same registration and exact33-artifact
inventory, all hashes, source-bound forecasts/targets/calibration, rebuilt
scores and all trace/cost/reference comparisons. Empty/missing/duplicate
fold rows cannot set parity_pass. Preserve partial artifacts on failure;
a live execution timeout is not permission to start another run.

## Evidence and readiness

The completed data-only [preflight](oracle_frozen_procedure_parity_evidence_20260906/preflight.json)
verified1,064 ancestral/current reference artifact paths and all fixed masks.
No new forecast or policy was calculated by this preflight. Its file SHA is
`4dcf2dacf2ca6a43d23d90ffc0c8cd4d8bdfd3be6015978e3a98baae221a86ae`.

The helper's synthetic tests cover evaluation-label strings being ignored,
interval-only quantile effects, exact mean arithmetic,504 scalar-reconstructed
Ridge predictions, support/minimum/alignment checks and input preservation.
The harness tests exact NaN/target/dtype/discrete semantics, tolerance failures,
fixed candidate/period inventories and rejection of empty resume manifests.
Full required suite:581 tests OK in56.488s; git diff --check passed before registration.
The independent review found no blocking issue in the ordinary execution path.
On a resumed fold, all values are rechecked against the original references;
the max-difference fields remain the original recorded values, not independently
recomputed maxima. The separate post-run audit recomputes comparison maxima.

This stage still uses archive event-time support. Receipt-as-of-deadline
collection, current-open observation, actual intent submission and authenticated
live input logs remain unimplemented integration boundaries. A successful
parity result is not prospective confirmation, trend-generalization evidence,
a new independent economic result, or grounds to replace the selected locks.
