# Fixed half-weight forecast shrinkage registration

Commit this exact comparison, config, sources, tests and data-only preflight
before computing any new half forecast, prediction loss or policy outcome.
This follows154 adaptively explored names on reused development validation;
it is not an untouched preregistered discovery or a new model architecture.

## Question and exact transform

The completed mean controls found scaled technical/perp return MSE6.296%/5.443%
worse than the scale-period mean anchor. The learned component nevertheless
helped some economic regime means. The completed fallback comparison improved
some drawdown outcomes but lost the perp_scaled observed-regime sign pass.
Test a fixed reduction in the learned mean's amplitude, separately under both
previously evaluated missing-forecast rules.

For each technical_scaled andperp_delay0_scaled mean, derive exactly
`mu_half = .5 * saved_scale_mean + .5 * saved_scaled_mean` on the original
inference mask; outside it remainNaN. Use the saved scale_mean NPZ, not a new
sample mean, calibration window, validation actual or second bias correction.
The scale anchor is constant within eachfold, from validation minus6months to
minus3months. It is not the most recent3month mean. The source mean estimator
is Ridge100; the identical fixed technical_scaled variance uses HGB.

The two identifiers are technical_half andperp_delay0_half. The half weight
is fixed before outcomes; there is no weight grid, optimized mix, IC selection
or retuning aftertest. Numerically use the convex sum to avoid overflow of
subtracting large opposing finite values. Preserve inputs and require finite
selectedvalues and a constantanchor. No outcome/scoringmask enters the transform.

Each new mean feeds both original utilityrisk1 hold-on-missing and the fixed
target1 missing-forecast fallback. Each develops its own cash/units trajectory;
never interpolate endpoint targets, profits, inventory or tradecounts.
Variance, riskaversion1, costmultiplier2, horizon24bars, UTC6hour decisions,
next15minute-open fills, one-wayfee.00055, annualborrow.10, maxstep.08,
deadband.01, initialB&Hunits andtargetrange[.5,1.12] remain unchanged.
Actual exposure can drift outside targetbounds; no forced clipping is added.
Base-selected intentarrays replay at2x fees/borrowing withoutreplanning.

## Endpoints, inventory and complete comparisons

Oldcontrols: scale_mean,technical_scaled,perp_delay0_scaled underbothrules,
plusBH/common_robust. These8controls come from the completedfallback parent,
while24 endpointscore rows andforecastNPZs come from its mean-control parent.
Keep both provenance chains. The half's forecast is shared acrossits2rules.

Fournewpolicy names +8oldcontrols ×8folds =96economicrows/targets.
16newderivedforecasts and16newscores, plus24copiedendpointscores =40scores.
32newtraces. Total158adaptivenames, notindependenttrials. No newfitting,
calibration, riskforecasts, dataacquisition, horizon or architecture change.

Report allfive mean comparisons: eachhalf versusownfull andscaleanchor,
andperp_half versustechnical_half. Eachcomparison includes MSE/MAE andboth
controllers atbothcosts (10economic comparisons). Also retain bothnewmeans'
fallback-minus-hold comparisons (2more), without choosinga favorablemissingness
rule fromthe outcomes. Report all/bull/bear/sideways equalquarter means, plus
row-pooled prediction losses, signaccuracy, per-fold rankIC, turnover,trades,
fees andborrowing. No new statisticalinterval orsignificanceclaim ismade.

For the same rows and aggregation weights, squaredloss obeys the exact identity
`MSE(half) = .5*MSE(full) + .5*MSE(anchor) - .25*mean((full-anchor)^2)`.
This guarantees no more than the endpoints' average loss, not beatingthe
betterendpoint, zero/fitmean, orimprovingAlphaEx/DD. Eachfold's positiveaffine
transform preserves forecastordering apartfromfiniteprecisionties; pooled
rankIC acrossdifferentfoldanchors neednotbe invariant. RankIC itself isnot
evidenceofimproved amplitude, sign orcost-sensitive decisions.

## Timing, support and immutable source proof

Retain folds5–12,2021-04-16T13:45Z to2023-04-16T13:45Z exclusive, andthe
18monthfit/3monthscale/3monthinterval/3monthvalidation chronology. Noextra,
omitted,fresh,test orouterdata mayselectthisweightorcontroller.
Keep2586inference/2574scoring rows,332fallbackeligibleknownopens,2missing
currentopens. Futurelabelavailabilitynever suppresses admissibleorders.
Theprior commonmask isretrospective pairedsupport, notahistoricaloperational
receipt/feedmask. Archiveprovenance doesnotestablish livecausalavailability.

The2bull/4bear/2sideways counts are start-of-quarter knownhistory labels;
actualsubsequentquarterreturn canhavetheopposite sign. Existingminimum3
quartersperregime stillfails. Do notweaken thegate orreinterpretpreviously
exploredquarters asindependentconfirmation.

Pin fallbackregistration/results andall8foldmanifestSHA256s; verify184current
plus736ancestralartifacts, mean/delayconfig-registration-result-preflight
bindings, sourcehelperhashes andSpotexecutiondataproof. Checkeveryendpoint
forecasttimestream/variance/actual/inference/scoremask andforecast-to-score
hashbinding. Preflightchecks anchorconstancyandsupportonly; itdoesnotcompute
anyhalfvalue/loss/target. Registerthepreflightdigest andcurrentcodebefore run.

The completed [data-only preflight](oracle_mean_shrinkage_evidence_20260905/preflight.json)
has fileSHA `31516273e780c4e3443500abd4b1f535f386e8b39bb5420e89e925af09006337`.
All920source artifacts andtheoriginalsupport werechecked beforehalf outcomes.

Replayall64oldcontrolrowsatbothcosts, checktarget/calendar provenance and
all24endpoint scores. Newholdactions staywithin learnedavailability; fallback
actions matchtheexisting332eligible maskandexacttarget1 withreasonedtrace.
For zero-fallback folds8/10/11, eachhalf's twocontrollers mustproduceidentical
targetsandbothcostmetrics. This doesnotmeanhalves shouldmatchfullmeancontrols.
Verifyall96rows,40scores,16forecasts,32traces andthecompletecomparisonuniverse.

## Primary sources and confirmation boundary

[Rapach–Strauss–Zhou (2010)](https://doi.org/10.1093/rfs/hhp063), p845Eq23,
discusses simpleforecastaveraging as shrinkagetowardahistoricalaverage in
USequitypremiumprediction. Its results are notaproof ofthissingleRidgeforecast
plus3monthanchor, BTC6hperformance oroptimalityofweight.5.
[Goyal–Welch–Zafirov (2024)](https://doi.org/10.1093/rfs/hhae044) extends
equitypremiumpredictor evidence anddocumentsmanyresultsfailingtohold up.
Its discussionofreuseddiscoveryperiods reinforces the distinctionbetween
exploratoryhistoricalimprovement andanindependentfrozenconfirmation.

After this fixedcomparison, preserveallweights/controllers/procedures and
failedgates. Anyexistingtest/outerevaluationmustberegistered report-only,
withnoselectionorretuningfromit. A prospectiveorpreviouslyunreadconfirmation
periodrequires documentedexposurehistory, a fixedcontiguouscalendar,
assessmentcriteriaandregimecoverage beforeaccessingits outcomes. Neither
reused8quarters noranimplementationtestestablisheshighprobabilitygeneralization.
