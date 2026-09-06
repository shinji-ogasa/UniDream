# Stage19: fixed continuous mapping of saved weighted probabilities

Commit AND push source, tests, config, this protocol, research and input-only
preflight before new real mapped means, diagnostics, residuals, losses or orders.
Stage18 was published at 96447c4600979c4ce5c66140fe60ddd27448c2d2 and the
unregistered research lead at 1967c48. Its four stronger-penalty classifiers
improved overall probability losses versus C1 but failed the prior and economic
comparisons. This next hypothesis follows those observed failures and is
exploratory, not independent confirmation. No new continuous-map result has
been inspected at registration.

## One change, zero fits

Use all four saved magnitude-weighted probability streams: Technical29 / Perp31
crossed with the already fitted C1 / L2unit settings. On their original mapped
inference support use the exact saved T mean absolute return a_T and arithmetic:

```
mu_soft = a_T * (2.0 * saved_probability - 1.0)
```

Do not call an estimator's fit or predict, recompute sigmoid, use tanh, add a C,
change any weight/feature/target/threshold, tune a multiplier or calibrate S.
The mathematical weighted-risk optimum q* satisfies
E[Y|X]=E[|Y||X]*(2q*−1) when the conditional absolute-return expectation is
positive and finite. If it is zero, Y is conditionally zero almost surely and q*
is unidentified by weighted loss. Saved finite-sample regularized q need not be
q*, and a_T substitutes a historical constant for E[|Y||X]. This is a controller
surrogate, not a newly established conditional mean. Ordinary probabilities do
not generally obey this mapping and are excluded by the estimand, not outcomes.

Zero q−0.5 maps exactly to zero. Preserve rounding at q=.5/0/1 and any underflow;
no epsilon, clipping, sign repair, alternate arithmetic or outcome-based tie rule.
Noninference entries remain NaN. A zero mean is a valid input, not a missing one.
Raw S probabilities/logits stay source evidence, but all new mapped S means stay
NaN. Scores remain separate I and E; there is no new S score or S calibration.

## Controls, complete family and arithmetic identities

Retain all60 Stage18 policies, including36 causal and24 hindsight controls.
Add4 learned soft means and3 constant means per group: mapped magnitude prior,
exact saved T mean return and zero. Every mean has existing hold and fallback
rules, so20 new causal names comprise8 learned policy names and12 constant-control
names. Total80 policies,640 rows and1,280 cost accounts; cumulative adaptive names
198→218. Constants duplicated across groups remain, not new independent evidence.
No hindsight control is a teacher or causal winner, and no RL beam is rerun.

The mapped-prior constant uses the saved probability in bound Stage17
`*_magnitude_prior_direction` NPZs, not the raw fit statistical prior and not a
fresh sigmoid of its logit. Verify it is constant on selected support and identical
across I/E and groups. Preserve both this q and raw fit_priors['magnitude'] in
provenance. The exact fit-mean constant uses saved fit_return_mean, not a newly
computed mean. These arithmetic choices can differ by floating rounding.

Require a_T finite and positive, selected q finite in[0,1], logits finite, strict
boolean aligned nonempty inference masks and finite outputs. Record mapped-prior
minus exact fitmean and require its absolute value ≤1e−14+1e−12*abs(fitmean).
Do not force equality, snap the difference or merge unequal controls. Zero and
fitmean MSE must exactly equal their named numeric baseline scores; all constant
rank IC values must be null. The helper verifies values but does not establish
provenance; the caller separately binds the saved inputs and chronology.

## Frozen chain and causal support

Verify all33 source files, Stage18 config/registration/preflight/results and8 fold
manifests, its2,840 ancestors plus648 own artifacts (3,488 distinct), and all saved
prediction/calibration/T-provenance inputs. Reject hash conflicts and aliases.
Reuse the Stage18 input-only preparation to reconstruct the original grid and six
fit/predict/scale/interval/inference/score masks from saved snapshots; no market
features are rebuilt. Verify all old480 rows including inherited trace hashes,
224 old return records and160 classification records. Require exact identity,
not merely closeness inside the old parity tolerance.

Original validation5–12 is April2021–April2023, strictly before
2023-04-16T13:45Z. It also aliases old test4–11 because test(f)=validation(f+1).
This is repeatedly reused development. T18months/S3/I3/E3 and original feature
availability remain. Features at t use bars through t−1; h24 return is
log(close[t+24]/open[t+1]), with all24 future bars and375-minute maturity.
T/S/I labels mature strictly before their segment end; E scores no later than
its end. Future score support must never gate mappings or orders.

E inference2,586 / score2,574; mapped I inference2,537 / score2,523. Preserve the12
unscored E inference and14 unscored I inference origins,332 fallback opportunities
and2 missing-current-open opportunities. The start-regime inventory is2bull /
4bear /2sideways. It uses trailing information, not realized quarter returns.
I strata grouped by E-start regime are retrospective, not known at I decisions.
The inherited Spot loader decodes full Parquet then applies the strict semantic
cutoff. UM raw bytes are hash-verified without new feature decoding. Event time
is not proof of historical production receipt time. Additional-test15–24, formal
outer and P1 remain unused for modeling/scoring here; results_observed=false is
unchanged. No production, live trading, paid infrastructure or external messages.

## Unchanged own-state execution

UTC6h decisions, own initial cash0/units1/open0 (B&H initialization), immediately
next-bar-open fills, one-way fee.00055, annual borrow.10, utility risk1 and cost
allowance2, intents[.5,1.12], max-step.08 and deadband.01 are unchanged. Passive
exposure can exceed intent bounds. Every new learned and constant policy makes
its own decisions from its own cash/units; it does not follow an Oracle's state.
Hold emits no order on missing forecasts; fallback targets1 only when the current
open is known. Missing current open prevents orders; missing immediately-next
open skips without rollover. Borrow continues across gaps. Risk is the same bound
saved technical_scaled variance. Stress replays the same base intents at twice
fees and borrowing, with no new stress optimization. Zero-mean controller is not
B&H; its risk, cost and inventory terms can still trigger trades.

## Complete scores, contrasts and rejection gates

Save24 return means ×I/E×8folds =384 scores, with MSE/MAE/sign accuracy/rank IC,
zero and fitmean MSE. Keep all10 inherited classifier streams ×I/E×8 =160 scores
exactly unchanged using saved logits and the existing scoring arithmetic. All
ordinary/weighted Brier/logloss/accuracy and uncosted signed-return metrics remain.
No recalculated probability from a mapped mean. Weighted zero denominators remain
null, retain their quarter and fail strict probability comparisons; do not omit.

All/all-bull/all-bear/all-sideways aggregates are equal-quarter; separately label
pooled-row MSE. No iid confidence interval, bootstrap, p-value, model selection or
high-probability guarantee. Every registered comparison is retained if unfavorable.
Only the eight learned policy names receive candidate gates. Constant controls
are not required to improve over themselves and cannot be promoted as learned
models by these flags.

For each learned mean, the five fixed references are its own same-classifier hard
map, original half, mapped prior, exact fitmean and zero. Preserve paired MSE
differences and improved/equal quarter counts, both I/E, and paired AlphaEX,
DDdelta, turnover and trade changes under both rules/costs in every stratum.

Separate gates are:

1. Mapped MSE strictly below ALL FIVE references in every stratum, separately I/E.
2. Absolute equal-quarter AlphaEX>0 and MaxDDdelta<0 in every stratum at both costs.
3. Paired AlphaEX change>0 and DDdelta change<0 versus ALL FIVE references in every
   stratum at both costs, under the same missing-input rule.
4. Inherited weighted Brier/logloss versus matched prior in all strata, separately
   I/E, is unchanged source evidence, not a new probability-accuracy result.

New probability-accuracy improvement is false by identity. New mapped-prediction
and economic conditions can be assessed separately without claiming q improved.
Strict same-quarter joint success counts require both metrics at both costs in
that same quarter. A descriptive pass would still not clear high-probability and
regime-count gates; both remain false on these reused2/4/2quarters. No candidate
is automatically selected or promoted, and criteria are not replaced post hoc.

## Diagnostics and artifacts

Save64 learned mapping records (4streams×I/E×8) on ALL mapped inference support:
rows; q=.5/0/1; source logit0; mapped mean0; sign(2q−1) versus sign(logit) and
sign(mu) versus sign(logit) disagreement counts; mean absolute new/hard/parent mu;
counts of new absolute mu greater than/equal to hard mu. These describe the
conversion and never choose an action or candidate. Per-fold mapping provenance
saves T scalars, their input paths/hashes, helper constant residuals, role/kind and
source-only probability/logit metadata. Constant NPZ q/logit fields remain their
source classifier's evidence, not newly estimated constant-classifier scores.

Save121 artifacts/fold:10 E NPZ,10 S/I NPZ,80 target NPZ,20 new trace JSON and1
mapping-provenance JSON =968 total. No joblib, new fit-data or fresh fitted state.
Also save8 fold manifests, registration/preflight/result/full stdout-stderr log.
Do not remove duplicate constant paths or hide numerical residuals. A completed
result refuses rerun. Never restart a live process on observation timeout. Any
terminal partial attempt may only replay the entire fixed procedure with exact
existing-artifact checks and unchanged source; report all attempts.

Before real mapping run the full `uv run python -m unittest discover -s tests -v`
and `git diff --check`. Synthetic checks cover algebra/constant guards, selected
poison and future exclusion, strict preserved scores, support/count completeness,
five-reference strict gates and null/regime separation. Independent audits verify
source inputs, all160 new mapped NPZs,64 diagnostics,384 return/160 unchanged
classification records,1,280 accounts,160 new own-state paths and report cells.

See [primary research and derivation](oracle_soft_direction_research_20260906.md).
