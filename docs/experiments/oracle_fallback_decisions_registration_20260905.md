# Forecast-unavailable inventory fallback registration

This small adaptive experiment follows147 policy names on reused development
validation. Its motivating diagnosis is recorded in
[the mean-control report](oracle_mean_control_results_20260905.md): long rolling
feature windows propagated several hours of missing Spot data into35days of
forecast unavailability, while existing utility policies retained inventory.
That diagnosis does not establish the return of a new fallback rule.

Commit this registration, exact config, data-only preflight, source and tests
before computing any new policy targets or economics. No model fitting,
new forecast calibration, horizon change, policy selection, significance test,
test/outer scoring or production change is part of this experiment.

## Exact family and state transition

Use every frozen mean from `oracle_mean_control_decisions_v1`: zero,fit_mean,
scale_mean,technical_raw,technical_scaled,perp_delay0_raw,perp_delay0_scaled.
All seven already contain the identical technical_scaled variance. The mean
and variance arrays, inference and score masks, periods and source provenance
remain frozen. Onlyutility risk aversion1 is used; no favorable mean or delay
is selected after seeing earlier results.

Compare each original hold-on-missing utility policy with one new rule:
at a scheduled UTC6hour decision with finite positive current open and valid
own-account NAV, if the frozen forecast is unavailable, submit target1.0.
This is own-NAV BTC exposure1, not restoring the B&H number of units or erasing
past relative losses. It may buy when exposure is below1 and still carries BTC
market risk. It is not claimed optimal or always risk-reducing.

When forecasts are available, use the same frozen conditional utility chooser
on that policy's own current cash/units. A valid forecast can choosehold;
its NaN intent must remainhold. Never fill all NaNs in old saved target arrays:
fallback changes the inventory used by every later forecast decision.

The trigger is schedule + known current open + unavailable inference. It never
depends on score_support, future price/label availability, current close,
whether a learned decision proposes a trade, or future returns. Claimed-valid
forecasts with invalid mean/variance are an input error, not an undocumented
alternate policy. The comparison uses the inherited retrospective common
availability mask; this does not implement a historical operational feed mask.

Canonical initialcash0/B&Hunits, targetrange[.5,1.12], maxstep.08, deadband.01,
one-wayfee.00055, annualborrow.10 and next15minute-open execution are unchanged.
Fallback submits1.0; actualtrade is projected at its unknown nextopen by the
canonical maxstep/deadband/fee solver. Passive exposure1.30 may only fall to1.22
in a single trade. Do not clip actual exposure to the target ceiling.
Missing currentopen means no decision; missingnextopen skips the submitted fill
without carrying it to a laterbar. Missingclose does not cancel an otherwise
validopen action or fill. Borrowing continues over gaps. Stress replays the
samebase intent arrays at2x fees and borrowing, without replanning.

Inventory: seven new names + seven original utility controls + B&H/common_robust,
across8folds =128 economic rows/targetarrays.56 new full traces. Forecasts are
referenced from immutable source files, not refitted or rewritten. No new loss
scores are needed because the forecast arrays and score support are identical.
New total154 adaptive policy names is not an independent-trial count.

## Frozen support and source proof

Original18month fit /3month scale /3month interval /3month validation, h24 and
folds5–12 are retained. The cutoff stays2023-04-16T13:45Z. Constant scale_mean
uses validation minus6months through minus3months, not the most recent3months.
No unavailable UM history or quarantined Spot bar is filled.

| Fold | Scheduled | Forecast | Fallback eligible | Missing current open |
| --- | ---: | ---: | ---: | ---: |
| 5 | 364 | 223 | 140 | 1 |
| 6 | 368 | 218 | 149 | 1 |
| 7 | 368 | 363 | 5 | 0 |
| 8 | 360 | 360 | 0 | 0 |
| 9 | 364 | 355 | 9 | 0 |
| 10 | 368 | 368 | 0 | 0 |
| 11 | 368 | 368 | 0 | 0 |
| 12 | 360 | 331 | 29 | 0 |
| Total | 2920 | 2586 | 332 | 2 |

These are data-only counts, not trades. All332 eligible slots happen to have
nextopen present; this is reported for support diagnosis and never gates an
order. Synthetic tests must therefore cover missingnextopen behavior. Forecast
score rows stay2574. Action support is widened only by the332 eligible slots.
There are2bull/4bear/2sideways start-of-quarter classifications; the original
minimum3quarters-per-regime gate still fails. Actualquarter trends can differ
from the known start classification.

Pin source mean-control registration/results and all8fold manifest SHA256s.
Validate every source artifact, all56 forecast/score bindings, control targets
and traces, helper/config hashes, parent delay lineage and Spot execution-data
proof before outcomes. Record current sources, versions and exact data-only
preflight hashes before running. RawUM inputs remain transitively pinned by
their frozen parent forecasts; this runner does not recompute UM features.
Archive hashes do not establish historical receipt times or live causality.

The completed [data-only preflight](oracle_fallback_evidence_20260905/preflight.json)
has fileSHA `4eef7d65cf66a57841536c5bf1b69ab4d0f29b8489d3ba43d1afd1bc33717ad0`.
Its736 source artifacts and exact fallback timestamps agree with the
[independent preflight audit](oracle_fallback_evidence_20260905/independent_preflight_audit.json).
No new policy outcome was computed to obtain those counts.

## Checks and complete reporting

Retain allseven fallback-minus-own-hold pairs for bothcosts and all/bull/bear/
sideways equal-quarter summaries. Report AlphaEx, MaxDDDelta, turnover,trades,
fees andborrowing, all failures, base/stress jointsigns and unchangedcoverage.
Distinguish observed regime-mean signs from passing the samplecoverage gate.
No new inference interval or selection-adjusted significance claim is made.

Each planner trajectory must match the independent canonical cash/units replay.
Every target must fall on a scheduled knownopen and either a valid learned
decision or the exact missingforecast target1. Preserve no-trade learnedholds.
For folds8/10/11, which have zero fallbackeligible slots, allseven new target
arrays and base/stress metrics must exactly reproduce their parent controls.
Recheck copied oldcontrols under bothcosts. Validate complete128row inventory,
allseven pairs and every artifact hash before completion.

Trace every knownopen scheduleddecision with reason learned/forecast_unavailable,
knownNAV/exposure and submittedintent. A missingforecast fallback has no estimated
utility gain or turnover score: save null, not an invented forecast score.
Report eligible slots, submittedfallback intents and actual executedtrades as
different quantities. Unit checks also cover allmissingforecast B&Hparity,
endogenous laterdecisions, passive exposureoutsideintentbounds, validlearnedhold,
missingcurrent/nextopens, missingclose, malformedinputs and futureinvariance.

## Primary-source interpretation

[Geifman–El-Yaniv (2017), section2](https://proceedings.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)
defines selective prediction risk conditional on acceptance, separately from
coverage. Its iid classification guarantees do not apply to this BTC inventory
process; available-row forecast loss cannot alone describe holding risk during
unavailable periods.
[Rubin (1976)](https://doi.org/10.1093/biomet/63.3.581) requires conditions for
ignoring missingness. A common paired mask is not proof that complete cases
represent all operating periods; neither random nor outcome-dependent missingness
is established by this diagnosis.
[Jorion (2003)](https://doi.org/10.2469/faj.v59.n5.2565) distinguishes benchmark-
relative risk constraints from totalportfolio risk. It neither studies this
fallback rule nor proves a maximumdrawdown guarantee. Target1 is an explicit
benchmark-exposure control whose value must be measured here.
