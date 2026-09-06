# Fixed causal rolling intercept update registration

Freeze and push this protocol, code/tests, config and data-only preflight before
computing any new real rolling mean, forecast, loss or order. This follows the
completed Stage13 reliability experiment: moving from full/half to a scale-fitted
weight reduced some forecast losses, but the remaining excess MSE against a
constant was dominated by a descriptive mean-drift component. That observation
motivates a falsifiable intercept-update procedure; it does not prove stale bias
caused the error or that a rolling window will improve it.

## Fixed family and data scope

Use original validation5–12 (original test4–11), April2021–April2023, strictly
below `2023-04-16T13:45:00Z`. The original naming aliases test(f)=validation(f+1).
These eight quarters are repeatedly reused development data. Additional
test15–24 is not used for this design, modeling, labels or scoring. The inherited
parent loader decodes the original Spot parquet before slicing; this is not a
claim that later file bytes are never read. Do not change any earlier family
lock, formal P1 state, or future confirmation protocol.

Retain all16 Stage13 causal controls. Add exactly6 policy names:
`rolling_anchor`, `technical_rolling`, `perp_delay0_rolling`, each suffixed
`utility_risk1` and `utility_risk1_fallback_bh`. This increases the adaptive
causal-name inventory162→168. Evaluate22policies×8=176economic rows, each at
base and doubled costs:352accounts. Score all9 old means and3 new means on
evaluation only:96records. No interval/scalepart score or new window variant is
introduced. There are **0 base fits and0 new weight fits**. Copy16 immutable
Stage13 scale weights; do not tune weights, the3month window, feature set, risk
model, action rule, costs or regime thresholds on the new outcomes.

## Source identity and as-of history

For evaluation start E, each frozen model uses fit [E−24m,E−6m), scale
S=[E−6m,E−3m), interval I=[E−3m,E), then evaluation [E,E+3m). Stage13 weights
were fit only on S and are available before this experiment's evaluation.
For each fold combine its parity calibration raw `mu` with that SAME fold's
original derivative-delay raw evaluation `mu`. Never use Stage13 calibration
`raw`, which is NaN outside old scale/interval masks. Do not use another fold's
forecast for the same timestamp or refit/infer the base model again. The parity
calibration copies exactly match original delay predictions, whose source model
provenance is independently verified. Bind the parent registration/results,
config, preflight,8fold manifests,27 source modules, and1536 ancestral artifacts
(1328 inherited plus208 Stage13 artifacts).

Reconstruct canonical h24 labels using only the full15minute grid below cutoff:
`y_tau=log(close[tau+24]/open[tau+1])`, all24 future bars observed. Its archived
availability time is tau+375minutes, the end of the24th future bar. Historical
event timestamps do not prove historical receipt latency. Both raw forecast
streams have identical origin availability and are already causal from bars
through origin−1. The raw prediction exists independently of whether its label
was admitted to the old segment score mask.

For every original current inference time t define the SAME paired set for
technical, perpetual and the return anchor:

    H_t = {tau: t-DateOffset(months=3) <= tau < t,
                  tau+375minutes <= t,
                  same-fold raw technical and perpetual forecasts
                    were available at this original six-hour origin,
                  canonical return is observed and finite by t}.

The window is defined by origin time, inclusive at its left boundary. Use UTC
pandas calendar-month DateOffset with its month-end convention, not90days,
the lastN observations, or a maturity-time window. Gaps do not widen the window.
At six-hour t the previous t−6h return matures at t+15min and must be excluded;
the latest possible admissible origin is t−12h. The inclusive equality rule is
explicit, although equality cannot occur between these six-hour origins and
their minute15 maturity clock. Do not describe an impossible equality branch
as exercised by the synthetic six-hour test.

Admit only matured labels, expire old origins, compute the current forecast,
then permit its action. Current y_t and any eventual evaluation score mask
cannot decide the current forecast/order. Previously purged scale/interval
boundary labels may enter later once they mature. Permanently restricting H_t
to old score masks would wrongly delete one usable boundary pair from2578 of
2586 histories. Future UM availability never determines label availability.

Require at least64 pairs for EVERY original inference time. Data-only preflight
finds fold5–12 minima179/215/215/354/351/354/363/330. If this fails, stop the
performance run instead of dropping times, widening windows or changing rules.
Pin per-decision counts and selected-timestamp hashes before outcomes and
compare them at runtime. Preserve inference2586 versus score2574, including
12 unscored inference rows,332 fallback opportunities and2 current-open gaps.

## Numerical update and diagnostics

On each H_t, using identical ordered pairs and `math.fsum(value/n)`:

    a_t = mean(y_tau)
    rawbar_g,t = mean(raw_g,tau)
    rolling_anchor_t = a_t
    mu_g,t = a_t + w_g,S * (raw_g,t - rawbar_g,t).

Keep w_g,S exactly as saved. For w=0 copy a_t exactly; validate all selected
raw/return values and reject nonfinite or overflowing arithmetic. Values that
are not yet mature must not be inspected to decide earlier forecasts. No
arbitrary variance floor or fill is added. Means outside current inference
remain NaN. The pure helper exposes insufficient history; the registered
wrapper fails unless its availability exactly equals original inference.

This changes a time-varying intercept while freezing the model and its slope.
Weight1 is now rolling bias correction and need not equal the old full model.
Weight0 is the new time-varying anchor; it is not necessarily a constant or an
undefined rank IC. Time-varying correction may change quarterly ranks even at
positive weights. Comparing against Stage13 identifies the entire joint update
of return mean and forecast center; it cannot identify their separate effects.

Report every fold and all/bull/bear/sideways summaries, with regimes known at
the evaluation quarter's first scheduled decision. Report equal-quarter MSE,
MAE and rank IC; label pooled-row MSE separately. Compare rolling_anchor with
scale_mean. Compare each learned rolling mean with rolling_anchor, scale_mean,
its own reliability, full and half mean. Save zero and fit-mean losses and
differences for every mean. Strict predictive direction requires lower MSE
than zero, fit mean and ALL registered references in ALL four summary strata;
this flag never selects or promotes a policy.

For all12 means, let d_t=mu_t−a_t and r_t=y_t−a_t on the same score mask. Save
the descriptive population-moment identity

    MSE(mu)-MSE(a_t) = E[d²]-2E[d*r]
                    = Var(d)-2Cov(d,r)+E[d]²-2E[d]E[r].

Report centered and mean terms with the residual. The baseline is the moving
anchor: Cov(d,r)=Cov(d,y)−Cov(d,a_t), generally not Cov(d,y). These components
cannot be compared to Stage13 fixed-anchor components as though the reference
were unchanged. The matched rolling anchor isolates incremental modeled
variation, but neither this algebra nor rank changes establish stable information.

## Execution, artifacts and checks

Keep the shared technical risk forecast and original own-state conditional
utility: risk1, cost allowance2, initial B&H inventory, intents[.5,1.12],
maxstep.08, deadband.01, one-way fee.00055, annual borrow.10. Each new policy
maintains its own cash and units. Fallback targets1 at scheduled known-current-
open times without a forecast. Missing current open means no order; missing
immediately-next open skips the fill without rollover. Preserve borrowing
through gaps. Doubled costs replay the same base-planned intents. Keep adverse
paired AlphaEx and MaxDDDelta differences; improvements need AlphaEx>0 and
MaxDDDelta<0. Save both all-strata economic means and joint-success quarter
counts, separately from predictive direction.

Save per fold3 forecastNPZ,1 shared rolling history traceJSON,22 targetNPZ,
6 new utility traceJSON:32×8=256 artifacts, plus8fold manifests, registration,
preflight and complete results. Rolling traces include count, window bounds,
oldest/latest origin and maturity, timestamp hash, missing matured-label count,
raw means, anchor, fixed weights, current raw and final predictions, and source
bindings. Retain all176 rows,96 scores and16 copied weights. Old16 controls
must reproduce targets/accounts; all72 old evaluation scores must reproduce.
Weight0 must match rolling-anchor forecasts, orders and accounts exactly.
Immutable partial files must compare exactly; completed output rejects rerun.
Do not restart a live process merely because output observation times out.

Synthetic tests cover selected-only/future-prefix invariance, immature poison,
calendar-month boundaries, shared support, minimum64, weights and overflow,
variable-anchor decomposition, own-state zero endpoint, complete-family
summary and strict registration. Before real outcomes run the full repository
`uv run python -m unittest discover -s tests -v` and `git diff --check`.
After completion independently audit history membership, rolling means,
forecast arithmetic,96 scores/decompositions,48 new own-state paths and352
accounts. No paid infrastructure, live trades, deployment or model promotion.

## Interpretation and primary sources

[Dawid1984](https://academic.oup.com/jrsssa/article/147/2/278/7106293) motivates
sequential prediction followed by later observations and updates.
[Pesaran and Timmermann2007](https://rady.ucsd.edu/_files/faculty-research/timmermann/estimation-window.pdf)
describe the bias/variance tradeoff around breaks; shorter history is not
uniformly better. Three months inherits the previous calibration time scale
and bounds this one hypothesis, not a literature-proven optimum.
[Dimitriadis and Puke2026](https://arxiv.org/html/2603.04275v1) discuss score
decomposition inference under assumptions not established here; no p-value or
calibration guarantee is imported. See the accompanying research note.

Earlier matured evaluation outcomes can affect later evaluation forecasts by
the fixed registered update; this is causal sequential evaluation, not a
quarter without label updates. Original start-regime coverage is2bull/4bear/
2sideways, below three per regime. Overlapping histories, dependent losses,
adaptive reuse and absent receipt evidence remain. A favorable result is an
exploratory candidate, not independent confirmation, high-probability trend
invariance, formal P1 success or a strongest-model claim.
