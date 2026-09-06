# Stage18: one fixed normalized-L2 direction comparison

Commit and push source, tests, configuration, this protocol, research and the
input-only preflight BEFORE new real C/regularization statistics, coefficients,
logits, mapped forecasts, losses or orders. Stage17 completed at report commit
29561fb1b97ae884a95863fc9ecea1dc7c241120. Its learned probability losses and
parent-relative economics failed; the present hypothesis follows those observed
failures and is exploratory. It is not independent confirmation or an architecture
search. No stronger-penalty result has been inspected at registration.

## One scientific change

Retain the original Technical29 / Perp31 groups and ordinary / magnitude-weighted
logistic losses. For each of the four models, change ONLY the classifier C from
Stage17's 1 to:

```
W_numpy = float(np.sum(frozen_fit_weights))
C = 1.0 / W_numpy
actual_lambda = 1.0 / (C * W_numpy)
```

The objective is average weighted log loss plus actual_lambda * ||beta||² / 2;
the intercept is unpenalized. In real arithmetic this fixes lambda at 1, whereas
C=1 has lambda=1/W. Record actual floating C, both NumPy and scalar-fsum weight
totals, lambda, counts, coefficients and intercept. Do not assert bit-exact lambda1
without checking it. Do not replace W by fit-row count, scalar fsum, effective
sample size or new normalization. The earlier C=1/n design suggestion is not a
second experiment. There is one schedule, no C grid, optimizer retry, threshold,
temperature or probability recalibration, and no stratum-dependent choice.

The schedule removes the explicit weakening of the penalty as training weight
increases. It does not establish that this strength is optimal, that conditional
probabilities are stationary, or that serially dependent BTC samples satisfy a
stability theorem. Uniformly repeating the same fixed examples leaves the
normalized objective unchanged in exact arithmetic; new chronological samples
can still change the fitted state. Stronger penalty introduces bias. With an
unpenalized intercept, the extreme-penalty limit follows its fit prior rather
than necessarily probability0.5. Both prior controls therefore remain essential.

All other estimator settings are unchanged: unweighted fit-only StandardScaler,
LogisticRegression L2 via l1_ratio0 (omit deprecated penalty), lbfgs, tol1e−8,
max_iter1000, fit_interceptTrue, random_state20260906, other pinned defaults,
threadpool limit2. Runtime pins: numpy2.2.6, pandas2.3.3, sklearn1.8.0. At run time,
compare every estimator setting except C and all scaler state exactly with each
saved Stage17 model. No old model is refitted; new fits are32 (4×8).

The original binary labels are Y>0. Ordinary weights are ones; magnitude weights
are abs(Y_i)/math.fsum(abs(float(Y_j))/n for j in fit). The new helper recomputes
these from frozen T returns solely for strict identity checks against saved
vectors. Priors are likewise checked against the existing two shared T priors
per fold. These16 verification recomputations are not new prior estimates,
independent controls or candidate models. No weight distribution is changed.

## Frozen evidence and chronology

Bind Stage17 source revision6ae673fcdfeed29280256450c05eb8905af77ee3, its config,
registration/preflight/results and all eight fold manifests. Verify its29 sources
plus the two new modules (31 total), all2,120 ancestral artifacts and720 Stage17
artifacts (2,840 distinct), and reject aliases/conflicting hashes. Preserve all
416 old economic rows,160 old return records and96 old classification records.

Reuse its eight bound fit-data NPZ files: original fit/predict positions and
UTC timestamps, exact selected feature matrices, continuous returns, labels and
weights. Reconstruct the original full 15-minute index and all six fit, scale,
interval, predict, inference and score masks; verify their original hashes,
counts, column order and matrix digests. Do not rebuild features or relax the
full inherited derivative/delay/variance availability contract to selected columns.
The new fitter receives only T return column0 and the frozen selected feature
matrices; all non-T outcome cells are NaN. No historical I/E label enters fitting.

Original development validation5–12 spans April2021–April2023, strictly before
2023-04-16T13:45Z. Old test(f)=validation(f+1), so these are also old test4–11;
repeated overlap is explicit. No additional-test15–24 modeling, scoring, labels
or selector, and no formal outer/P1 execution. The original fit18months is
[E−24months,E−6months), followed by S3months, I3months, E3months. T/S/I labels
mature strictly before segment end; E score labels no later than its end.

Features at decision t use completed bars through t−1; target is
log(close[t+24]/open[t+1]), with all24 future bars present and375-minute maturity.
Fit counts800/1034/1313/1500/1503/1634/1672/1794. E inference2,586, E score2,574;
I scored2,523. Preserve12 unscored E inference origins,332 fallback opportunities,
and2 missing-current-open scheduled origins. Original start-regime inventory
is2bull/4bear/2sideways. All six support hashes must remain exact.

For unchanged economic replay, use the same source-bound load_bars and Spot
artifact/sidecar/ledger proof. That inherited loader decodes full Parquet before
strict semantic slicing; do not claim later bytes were not decoded. UM raw data
is hash-verified but not parsed or rebuilt into features in this experiment.
Event timestamps still do not prove production receipt-time availability.

## Fixed mapping, old controls and execution

Four new classifiers map to four new means:

`mu_new[t] = sign(new_logit[t]) * abs(own_frozen_parent_half_mu[t])`.

Map every original inference origin, including unscoreable future labels. Cache
parent magnitudes from Stage17's bound E/S-I arrays. S-derived parent bias and
anchor were applied only from I onward; retain that boundary and keep mapped S
means NaN. Raw new classifier logits may be saved on S, but S is neither scored
nor used for new calibration. Risk/actual/masks/timestamps are unchanged.

np.sign(0)=0 without epsilon; binary scoring uses logit>0, so zero means zero
mapped direction but a nonpositive binary prediction. Y=0 is nonpositive and has
zero magnitude weight. abs(parent mean) is not E|Y|, and the mapped mean is a fixed
surrogate for the controller, not newly calibrated E[Y|X]. Magnitude-weighted
probability is tilted, not ordinary P(Y>0|X). Save both score families regardless.

Preserve all52 Stage17 policies:28 causal controls and24 hindsight controls. Add
only8 causal names (four means×hold/fallback), making60 policies,480 rows and960
base/stress accounts; cumulative adaptive names190→198. Retain equal or duplicate
paths and all original priors. No old future-informed diagnostic is a teacher or
causal winner; no finite RL beam is rerun or relabeled as a global upper bound.

Execution is unchanged: UTC6h decision clock, initial B&H inventory, own cash
and units, next-bar open fill, fee0.00055 one-way, annual borrow0.10, utility risk1
and cost allowance2, intents[.5,1.12], maximum step.08, deadband.01. Passive drift
may exceed intent bounds. Missing prediction: hold submits no order, fallback
submits target1 only with current known open. Missing current open prevents an
order; missing immediately-next open skips the fill without rollover. Borrow
continuously across gaps. Replay each base target stream at twice fees/borrowing
for stress, without a new optimization. Future labels never gate orders.

## Complete scores and predeclared flags

Save ten probability streams (six old +four new) on I/E:160 records. Save fourteen
return means (ten old +four new) on I/E:224 records. Recompute all old scores
unchanged. Every probability stream reports ordinary Brier/logloss/accuracy,
|Y|-weighted versions, uncosted signed-return mean, zero counts and absolute-return
denominator. Every mean reports MSE/MAE/binary sign accuracy/rank correlation and
zero/fitmean MSE controls. Use inherited stable, unclipped logit-based log loss.

Aggregate equal-quarter means in all/bull/bear/sideways, I and E separately.
I grouped by E-start regime is retrospective; that regime was not known at I
scored decisions. Separately label pooled-row MSE. Undefined rank metrics remain
null. If a segment has zero total absolute return, weighted metrics are null,
the quarter remains, and equal-quarter aggregates/contrasts involving it are
null; no silent omission. No bootstrap, iid interval, p-value or probability.

For each new classifier, preserve proper-score differences to BOTH priors, its
same-group/same-loss C1 model, and its new same-group opposite-loss counterpart.
The ordinary matched-loss flag requires BOTH ordinary Brier and logloss strictly
below BOTH its C1 model and ordinary prior in all four strata, for each segment.
The magnitude flag uses weighted Brier/logloss and its C1 model/magnitude prior.
Do not choose a favorable objective, loss, subgroup or segment after inspection.

Return-MSE flag requires strict improvement over zero, fit mean, own original
half, same-loss C1 mapped mean and matched-prior mapped mean in all four strata
in each segment. Save paired differences and improved/equal quarter counts for
all three explicit learned/prior mean references. Loss and mapped-return claims
remain separate from hard direction accuracy.

Economic absolute flag requires equal-quarter AlphaEX>0 and MaxDDdelta<0 in
all four strata at both costs. The separate paired economic flag requires
AlphaEX change>0 and DDdelta change<0 against all three references at both costs
in all strata. Save every paired turnover/trade/economic change and strict joint
success-quarter counts. A probability-only gain, unchanged policy, prior-like
collapse, or merely absolute positive Alpha is not parent-relative improvement.
No model is selected or promoted by these flags. With reused2/4/2 quarters, the
high-probability generalization and regime-count gates remain false regardless
of the descriptive outcome. Formal P1 results_observed=false remains untouched.

## Fixed descriptive mechanism records

Save64 records (4 classifiers×I/E×8 folds) on ALL mapped inference support,
not future score support: new/C1 sign-disagreement count, same-direction count
with matched prior, new/old zero-logit counts, mean absolute new/old logits,
new/old coefficient L2 norms and intercepts, exact C/W/lambda, and rows. These
describe what changed; none controls tuning, model selection or an action.
Coefficient norms are fit-state summaries; I/E logit summaries are retrospective
input-only descriptions. There is no S score or S-based mechanism selector.

A finite positive temperature that preserves logit signs leaves this sign-only
mapped mean and own-state path unchanged. Stronger regularized refitting can
change signs because it need not be uniform positive scaling. No temperature
model is fitted merely to manufacture another unchanged economic candidate.

## Numerical acceptance, artifacts and verification

Keep the Stage17 selected-data/chronology/positive-effective-class guards.
Require finite positive W and reciprocal C, mean-one frozen weights, finite model
state/probabilities, no ConvergenceWarning and n_iter<1000. Independently recompute
with Python scalar transforms/math.fsum before accepting each model: finite
normalized objective, gradient infinity≤1e−6, all-predict logit difference≤1e−12
and probability difference≤1e−14. Use the actual 1/(C*W) in the gradient, not an
old C1 shortcut. Solver gtol1e−8, ftol64*float64epsilon, maxls50 are unchanged;
the fixed gradient guard allows the pre-existing ftol stopping route. Keep all
RuntimeWarnings visible and distinguish finite-state evidence from an explanation
of their cause. Do not retry a numerical failure with changed settings.

Save81 artifacts/fold:4 new model joblibs,1 fit-provenance JSON,4 E forecast NPZ,
4 S/I prediction NPZ,60 target NPZ,8 new trace JSON (648 total). Reuse the bound
old fit-data file without making another candidate copy. Save8 fold manifests,
registration/preflight/result/full stdout-stderr log; each fold also carries its
8 mechanism records. Include old-model paths, exact scaler/settings parity and
new model/provenance key mapping. All artifacts remain hash-bound.

A completed result refuses rerun. A terminal partial attempt may only replay the
full fixed procedure with exact existing-artifact checks and unchanged source.
Never restart a live process because an observation timed out. Report attempts.
No production, live trading, external messages, paid infrastructure or automation.

Before real fits, run the complete `uv run python -m unittest discover -s tests -v`
and `git diff --check`. Synthetic checks cover normalized-objective replication,
free-intercept prior limit, selected-input poison, normalizer rejection, scalar
acceptance, immutable parent settings, original snapshot masks and nonfit label
exclusion, complete reference/null/gate semantics. Independent audits verify all
31 sources/2,840 ancestors/648 new artifacts, fitted scalar states,224/160 scores,
64 mechanism records,960 cost accounts,64 new own-state paths and report cells.

See [primary research note](oracle_regularized_direction_research_20260906.md).
[sklearn1.8](https://scikit-learn.org/1.8/modules/linear_model.html#logistic-regression)
fixes objective arithmetic. [Bousquet–Elisseeff](https://www.jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf)
motivates a regularization/stability question under specific assumptions; it does
not verify an all-trend BTC guarantee, finite-sample calibration or DD control.
