# Goal continuation: Oracle / features / trend robustness

Overall goal remains active. No candidate has established high-probability joint
AlphaEx>0 and MaxDDDelta<0 across trends. Do not mark this goal achieved because
the exploratory means or an implementation test passed.

Working branch: `exp/oracle-feature-frontier-20260905` in
`/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905`.
Initial experiment source commit: `ed23b49`; risk ablation source: `0272879`.
Current runtime: `uv run python` from this worktree (local ignored `.venv`,
Python3.12.13, numpy2.2.6, pandas2.3.3, sklearn1.8.0). The initial frontier
stage used the alpha-dd-goal environment; do not substitute it for new stages.

## Completed; do not rerun or retune these locks

- Audited current main and prior alpha/dd work. Older memory of P1 blocked
  execution is stale: formal recoverability tests later ran, but those are not
  natural BTC investment evidence. No old P1 record was modified.
- Added 29-feature technical and 24-feature flow groups, with strict shift1 and
  gap handling. Fixed36 ML family: 234 out-of-time fits on13 actual validation
  quarters, 689 economic rows including diagnostic/control arms.
- RL finite-beam feasible hindsight reference and ML perfect-outcome mapping
  use the same cash/units execution contract as real policies. These are
  future-dependent diagnostics, not global upper bounds or training teachers.
- Six of36 learned policies passed joint mean signs in base and2x costs;
  zero passed all start-of-quarter trend groups. Technical weekly downside
  Ridge improved mean signs over its base16 counterpart but failed bull DD
  and sideways alpha. Fixed minimax selection chose an aggregate-failing flow
  HGB; do not silently switch the selection criterion.
- Technical HGB6h risk forecast improved volatility MSE skill in all three
  regime means (10/13 quarters). Return MSE did not improve. This is the
  actionable positive forecast finding, not an overall accuracy claim.
- Twelve predeclared volatility-ratio controllers were then tested. None
  passed all regimes; adding them worsened average results against their
  own backbones. Reject this direct allocation-scaling recipe.
- Final410 unit tests passed. Audit details and immutable snapshots accompany
  the results report. No production model, HF Space, or paper account changed.

## Second completed stage: risk reliability and conditional decisions

The previous next steps1 and2 below have now been executed; do not repeat them.
See `oracle_risk_reliability_results_20260905.md` and its evidence directory.
All81 candidate names across the first36, ratio12, calibration24 and conditional9
families remain exploratory. Some names yield identical targets;81 is not an
independent-trial count. No family passed all three trend means at both costs.

- Registered `65a1d4f` before computing the stronger persistence comparison.
  Existing HGB6h risk vs causal RMS24/96/672 uses the same3863 score rows.
  Technical vs96bar persistence: variance MSE loss reduction9.967%, QLIKE13.011%,
  RMS MSE14.954%; all11/13 quarters improve. Sideways variance MSE is0.70% worse.
  This is a loss diagnostic, not a policy selection or significance result.
- Registered `8e3dbd9` before disjoint18month fit/3month scale+meanbias/
  3month interval calibration/3month validation. Six risk forecasters,
  raw/scaled and point/interval gates produce24 economic candidates.
  Technical variance calibration improves QLIKE.5142→.4382, while calibrated
  HAR is.4273. Technical90% return interval coverage90.27%, but its risk
  interval bear coverage78.77%. Technical point alpha/DD improves from
  -.925/+.666 to+.849/-1.568pt, stress+.218/-1.471. Bear alpha still negative;
  bull DD marginally positive at2x. Technical interval gate trades0 times.
  Scaled forecasts also shift return mean; scale alone cancels quantile widths.
- Registered `b56551d` before9 conditional-utility policies with frozen
  forecasts, actual own inventory, known current open and a2x fee budget.
  Risk coefficients0/1/4. Score only feasible current-open max-step projection;
  next fill remains unknown. All117 target arrays replay identically in the
  canonical account. Risk0 names have identical targets because they sharemu.
  Risk0 alpha/DD+2.382/-4.516pt, stress+1.947/-4.444; bear alpha-1.518/-1.918pt.
  Turnover38.378 vs technical point112.676;531 vs2786 trades. Risk4 further
  reducesDD but losesalpha. Never present this as all-regime success.
- Baseline/calibration/decision artifacts hash-audited. Independent conditional
  replay covered27 policies,54 cost paths,6876 choices; maxalpha error1.20e-13.
  Final436 tests OK in56.154s; full log `/tmp/oracle-reliability-full-tests.log`.
- Registered `83f72a1` before downloading UM raw15m. The official44month
  request Sep2019–Apr2023 yielded40 SHA-verified months Jan2020–Apr2023;
  2019Sep–Dec404 stay missing. Dataset grid128448/observed116736; eligible
  observed115350 beforecutoff2023-04-16T13:45Z. The1386 observedApril-tail
  bars remain ineligible. Derivative fitting had not run at this stage;
  the completed matched experiment is recorded below.

Runtime outputs, immutable and locally retained:
`codex_outputs/oracle_risk_baselines_v1`, `oracle_risk_calibration_v1`,
`oracle_conditional_decision_v1`.
New raw data: `checkpoints/oracle_derivative_data/um_15m.parquet`,
SHA `c365cbb6cd6d46dafea19b7fd3a62cb1b91ae3169f7a27532e40486ecfebfcdd`.
Sidecar, availability parquet, source ledger and40 raw verified ZIPs retained.
Index is barOPEN, not decision time. All features shift exactly once.
Historical receipt/publication timesunknown; live_causal_eligible=false.

## Third completed stage: Spot / perpetual information

Source registration commit `99f3cd9` preceded the derivative model run.
Output: `codex_outputs/oracle_derivative_ablation_v1`; result report:
`oracle_derivative_ablation_results_20260905.md` with tracked evidence.
The run completed successfully; original handle28926 is terminal, not live.

- Four feature groups: base16, technical29, technical+perpflow2 (31), and
  technical+all8 (37). Separate Ridge100 mean and HGB100/7leaf logvariance
  per group,18/3/3 chronology, folds5–12 only.64 models,64 forecasts,
  32 calibration records,208 targets,128 utility traces.
- Exact preflight masks retained:2587 causal decisions,2575 scored labels,
  on2920 scheduled6h slots. Parent comparators were refit on identical rows.
- Perp2 mean-return MSE vs technical improves .456% raw / .802% scaled;
  scaled improves6/8 quarters and all three equal-quarter regime means.
  Sideways improvement .008% flips to .0375% worsening when rows are pooled;
  do not claim stable all-regime predictive improvement.
  But every return model remains worse than zero/fitmean MSE. Scaling mean
  worsens MSE while helping some economic policies; do not conflate them.
- Perp2 variance QLIKE worsens .834% raw / .545% scaled vs technical.
  This motivated the completed frozen crossing below. All8 is a representation ablation because
  of redundant spot/perp96 and newly accessible spot24 information.
- Perp2 scaled utilityrisk0/risk1 are the first two in this branch with
  all observed regime means having alpha>0/DDdelta<0 at both costs.
  Risk1 overall+2.144/-3.137pt, stress+1.737/-3.008. Regime base values:
  bull+4.063/-4.749, bear+.967/-2.784, sideways+2.579/-2.231.
  Stillbull2/bear4/sideways2, so existing3quarters/regime gate remainsfalse.
  Only4/8 quarters meet both signs. IIDfold economic CIs crosszero.
  All8 scaled risk1 fails sideways. No winner/promotion was selected.
- Post-outcome concentration diagnostic: perp2-minus-technical scaledrisk1
  overall AlphaEx uplift+1.531pt includes+1.994pt contribution fromfold12
  alone (2023-01-16 through2023-04-16). Other7quarters average-.529pt.
  Risk0 andpoint have the same concentration pattern. Full8fold evaluation
  stays unchanged; all8 leave-one-out diagnostics saved without selection.
- Primary paired return-MSE intervals belowzero at all3 fixed blocklengths,
  while QLIKE intervals abovezero. These are bootstrap-mean-centered
  conditional descriptive intervals, not conventional basic/DM/SPA tests,
  selection-adjusted evidence or prospective proof. All8 scaled riskQLIKE
  interval crosseszero at112slots despite beingnegative at28slots.
- All518 source/data/artifact hashes checked. All32 Ridge predictions and
  normal equations independently reconstructed.9 HGB tree traversals exact.
  108 accountpaths/10968 decisions maxalphaerror2.66e-15, DD5.55e-16.
  All64 forecast scores and all12x3x2000 bootstrapreplicates independently
  reproduced. Matmul warning rootcause remains unknown; scalarvalues match.
- Final full suite after the crossed diagnostic:463 tests OK in56.801s,
  `uv run python -m unittest discover -s tests -v`.
  Log `/tmp/oracle-derivative-final-tests.log`; handle43152 terminal exit0.

The prior81 policy names plus24 derivative names make105 adaptively explored
names, not independent trials. Six of24 new names pass aggregate signs;
two pass observed regime means, zero pass the unchanged coverage gate.
The goal remains active. Research significance and future economic validity
must be strengthened; do not mark complete on these exploratory means.

## Fourth completed stage: separate mean and variance contributions

Source registration `64de0fe` preceded the four new crossed-policy outputs.
`codex_outputs/oracle_derivative_crossed_decisions_v1` is complete; handle89709
terminal exit0. Exact registration/results and independent audit are tracked
beside the derivative report. No new fitting, calibration, intervals or selection.
Frozen scaled forecasts: perp2 mean+technical variance andtechnical mean+perp2
variance, each withpoint andutilityrisk1. Samebase intents replayed at2x.

-96 rows/targets,48 traces include64 byte-identical parent control rows.
  Same2587 causal/2575 score rows. Four new names make109 total adaptive names,
  not independent trials. Two new names pass aggregate signs; one passes all
  observed regime signs; zero pass existing coverage gate.
-Perpmean+technicalrisk utility1: Alpha/DD+2.143/-3.137pt, stress+1.736/-3.008.
  Technicalmean+perprisk utility1:+.622/-4.319, stress+.303/-4.207.
  Improvement follows the mean forecast and resulting inventory trajectory;
  switching risk source barely changes this utility policy. Point crosses
  both fail aggregateAlphaEx. This does not curefold12 concentration.
-Independent342 hashchecks; everypointpath and all summaries reconstructed.
  Scalar24accountpaths/1828 decisions overfolds5/8/12: alphaerror1.55e-15,
  DD5.55e-16,tradecount0. No mismatches or artifact modifications.

## Fifth completed stage: UM information staleness

The previous next-action delay diagnostic is complete; do not rerun or choose
a best delay after seeing it. Source+registration+preflight commit `a75b55a`
preceded all model outputs. Run handle95064 is terminal exit0.
Output `codex_outputs/oracle_derivative_delay_v1`; report
`oracle_derivative_delay_results_20260905.md` and tracked evidence directory.
Full suite481 tests OK in56.410s, `/tmp/oracle-delay-full-tests.log`, handle78456
terminal exit0. Runtime is this worktree's ignored `.venv` via `uv run python`.

-Same18/3/3 chronology, h24, folds5–12, Ridge100/HGB100iter7leaf fixed.
  OnlyUM flow24/96 shiftedextra0/1/4bars; Spot/decision/labels/fills unchanged.
  Technical+delay0/1/4 refit on acommonmask. Frozen1/4 use NEWdelay0 fitted
  scaler/models/bias/variance multiplier/quantiles at validation only.
-Data-only final preflightSHA
  `d20d0a23096a6693dc7125e0f5fd9e761efc7e5f7bfdb6ca42c5b890c91252c4`.
  2586inference/2574scored of2920scheduled; one more row lost infold9.
  The previous sharedmask is intersected with alldelays: it includes knowledge
  of undelayed availability, so this is NOT an operational delayed-feed mask.
  No historical receipt-provenance claim. The earlier pre-regime-guard draft
  preflight remains locally retained; onlythe abovefinalhash is registered.
-64 models/32calibration/96forecasts/208targets/96traces;24policy names.
  Newtotal133adaptive names, notindependenttrials.6/24aggregatejointsigns,
  5/24observedregimemeans,0/24unchangedcoveragegate. Counts2bull4bear2sideways.
-Scaledutilityrisk1 equalquarter Alpha/DDpt:
  technical+.595/-4.323, stress+.277/-4.211;
  delay0+2.144/-3.137, stress+1.737/-3.008;
  delay1refit+1.818/-3.309, stress+1.423/-3.185;
  delay4refit+2.381/-3.358, stress+2.003/-3.234;
  frozen1+2.265/-3.310, stress+1.885/-3.183;
  frozen4+2.290/-3.244, stress+1.895/-3.116.
  AllfiveUMscaledutility conditions preserve observedregimesigns atbothcosts.
  Delay4refit/frozen4 joint5/8quarters, otherUMscaledutility4/8.
-ScaledreturnMSE improvements vstechnical:delay0.802%,delay1refit.868%,
  delay4refit.665%. Allreturnmodels stillworse thanzero/fitmean. Frozen15/60min
  makesreturnMSEslightlyworse whileimprovingeconomics. No newintervals/tests.
  Do notconflate MSE, averageAlpha andprobabilityofgoalcompletion.
-AllfiveUMscaledutility improvements versusparent stilldependonfold12:
  refit60min uplift+1.786pt includes+2.142ptfold12 contribution; other7mean
  -.407pt. Frozen60min uplift+1.695pt includes+2.327pt; other7mean-.722pt.
  All8 registeredfolds remain in evaluation; postoutcomeLOO isdescriptiveonly.
-Audits:496manifestartifacts/525files, all96forecastmetrics/source/masks;
  48Ridgevalidationstreams scalarmax3.47e-18, all48HGBstreams directtrees0.
  Allfrozen32records matchdelay0sourceandcalibration.144accountpaths and
  10968choices independentoverfold5/8/12:alpha2.44e-15,DD6.66e-16,trades0.
  NumPymatmulwarningsremainundiagnosed; storedpredictions independentlymatch.
  Audits reconstructstoredcoefficients, notindependentfullretraining.

## Additional completed diagnosis: fixed mean-bias drift

Use the saved `parent_meanbias_diagnostic.json` for the PRIOR broader-mask
derivative ablation, not thenewdelayfamily. Fiftysourcehashbindings checked.
No newmodels, policies orselection were run inthisdiagnostic.
-Fold12 inference331rows rawmu technical-.781bps/perp+11.731bps;
  bias technical-6.286bps/perp-12.223bps; correctedmu-7.066/-.491bps.
  Bothgroupscorrectdownward, narrowingthepredictiongap12.512→6.575bps.
  Positivepredictions164→101 versus232→168, with63/64 downwardcrossings.
-CalibrationactualmeanJul–Oct2022-2.960bps, scoredvalJan–Apr2023+10.704bps.
  Causalfixedbiasdoesnottransferwellhere. Fold12MSE worsens.977%/.676%.
  Across8quarters5/8MSEworsen forbothgroups afterbiascorrection.
-Risk0varianceignored:fold12technicalAlpha-8.365→-16.560pt,perp+5.352→+.057pt
  raw→scaled. Relativegapwidensbecauseparentdeterioratesmore; risk1similar.
  Concentration is specific tocorrectedutility: priorrawrisk1perp-minus-tech
  other7quarters+1.537pt, whereas scaled-.529pt. Do notclaimallUMinformation
  exists onlyinfold12. Signzeroalonecannotexplainendogenousorders.

## Sixth completed stage: frozen mean decision controls

The causal mean controls above are complete; do not rerun or choose a winning
mean after seeing them. Source registration commit `c93b9ff` preceded all new
policy outcomes. Output `codex_outputs/oracle_mean_control_decisions_v1`,
run handle14681 terminal exit0. Report `oracle_mean_control_results_20260905.md`
and tracked `oracle_mean_control_evidence_20260905/` contain exact evidence.
499 tests OK in56.852s before registration, `/tmp/oracle-mean-controls-full-tests.log`.

- Seven means: zero,fit_mean,scale_mean,technical_raw,technical_scaled,
  perp_delay0_raw,perp_delay0_scaled. Every arm uses technical_scaled variance.
  Each feeds point andutilityrisk1:14 names,147 adaptive names total, not
  independent trials.128 rows/targets,56 forecasts,56 traces,56 scores,
  all21 unordered mean comparisons. No new models or calibration choices.
- Same2586 inference/2574 score rows,8quarters2bull/4bear/2sideways.
  scale_mean uses purged actuals from validation minus6months to minus3months,
  NOT the most recent3months. Constant mean still uses learned HGB variance.
  Raw means with common scaled technical variance differ from parent raw policies.
- Utility1 equal-quarter Alpha/DDpt: scale_mean+3.128/-6.531,
  stress+3.061/-6.494; technical_scaled+.595/-4.323,stress+.277/-4.211;
  perp_scaled+2.131/-3.130,stress+1.723/-3.001.3/14 aggregate joint signs,
  onlyperp_scaled observed regime means pass;0 unchanged coverage passes.
  scale_mean bearalpha-.873pt/stress-.914,perp+.967/+.595.
  Scale_mean sideways+9.597pt averages fold9+22.714 andfold12-3.519.
- Versus scale_mean, technical_scaled MSE6.296%worse/MAE4.569%worse,
  both0/8 improving;perp_scaled MSE5.443%worse(2/8 improving),MAE3.990%worse(0/8).
  All learned mean MSE remains worse than zero/fitmean. The dynamic component
  includes validation-period mean shifts, not solely mean-zero temporal noise.
  Mean trades/quarter scale4,technical44.5,perp57.625;turnover.247/3.187/4.232.
  Extra perp fees+borrow .302ptinitialNAV do not explain whole Alpha difference;
  own inventory paths also change. Do not discard dynamicmean bear value.
- Independent audit1086 hashbindings/765distinctfiles,all56scores/all21pairs.
  84scalar accountpaths,21utilitypaths/6398decisions overfold5/8/12:
  alphaerror1.33e-15,DD5.56e-16,tradecount0. All56forecast streams separately
  checked,4912calibrationlabels matured;scaleconstant rawpriceerror4.34e-19.
  technical_scaled16targets/8traces/metrics matchparent exactly;
  zero_point BH economic parity all8, control maxdifference0.

## New diagnosis: forecast availability and passive inventory drift

Fold5 three scaled mean utility policies have identical targets:1.08 at
2021-04-16 18:00UTC,1.12 atApr17 00:00,then221 valid holds. All223 available
mean forecasts positive. Their identical performance is a decision-path result,
not evidence of identical forecast accuracy.

141 scheduled decisions fromApr20 06:00 throughMay25 06:00 are unavailable:
140 missingforecast and1missingcurrentopen. This is NOT35days of missingprices.
Spot has10 missing15m bars Apr20 02:00–04:15 and19 unusablebars Apr25 04:00–08:30.
OneApr25 04:00bar was quarantined for incomplete close time04:00:58.146;
the remaining28 are absent. April2851acceptedof2880. Do not invent exchange
outage causation or historical receipt provenance from these archive records.
Strict7d coverage669/672 and30d2866/2880 propagate the gaps. The needed
technical29 itself fails all141 rows, so removing unused common-mask columns
restores ZERO fold5 rows. UMflow24/96 at extra0/1/4 remains finite there.
30d finite returns2849 onApr28/2860 onMay20; finalMay25 06:00 momentum30
stillreferencesmissingApr25close. Alltechnical columns recoverMay25 12:00.

Fold5 was classified bull at START, but actualquarter BHreturn-47.766%,
threepolicies-53.942%;alpha-6.176pt/DDdelta+6.505pt. Negativecash and declining
prices cause actualexposure todrift above the1.12 TARGETbound:observed1.22348
atMay25 12:00 and1.27840 atfinalvaliddecision. These are observeddecision
states, not maximum full-period exposure. Forecasts resume but choosehold too;
do not attribute the entire loss solely to availability.

## Seventh completed stage: own-inventory forecast-unavailable fallback

The suggested fallback experiment is complete. Source registration commit
`ab0ef88` preceded all new policy outputs. Run handle68122 is terminal exit0.
Output `codex_outputs/oracle_fallback_decisions_v1`, report
`oracle_fallback_results_20260905.md` and tracked evidence directory.
Full521 tests OK in57.242s, log `/tmp/oracle-fallback-full-tests.log`,handle94121
terminal exit0. Finaldata-only preflightSHA
`4eef7d65cf66a57841536c5bf1b69ab4d0f29b8489d3ba43d1afd1bc33717ad0`.
RegistrationfileSHA139f1397a28c7296b73828103e5ab3e1ea856df2049c1a02367bee02a8375a3f;
resultsSHA cf7a4b84af462bc645863fae2646aa32eef57cc30572f01b88912cda78a92954.

- New sibling fallbackplanner preserves all old sourcehashes. At knownopen6h
  decisions with unavailable inference it submits ownNAVtarget1. Valid learned
  holds stayNaN; later choices use changedowninventory. Canonical nextopen,
  maxstep.08/deadband.01/cost/borrow and stresssamebaseintents remain fixed.
- Allseven mean-control means × utilityrisk1 only:7new +7oldcontrols +BH/robust,
  128rows/targets,56newtraces,184artifacts. No newfit,forecast orforecastscore.
  Total154adaptivenames,notindependenttrials. Same2586inference2574score rows.
  Action adds332scheduledknownopen unavailableforecastslots;2currentopenmissing.
  Fallbackcounts byfold5..12:140,149,5,0,9,0,0,29. All332nextopens happened
  toexist;neverusedtoalloworders. Missingnextopen behavior covered synthetically.
- Newutility equalquarterAlpha/DDpt: scale_mean+3.541/-6.716,stress+3.486/-6.688;
  technical_scaled+.705/-5.079,stress+.410/-4.981;
  perp_scaled+1.918/-3.777,stress+1.546/-3.664.
  3/7aggregatejointsigns,0/7observedregimemeans,0coverage(2bull4bear2sideways).
  Perp_scaled loses its old observedregimemean pass:bearalpha+.967/+.595
  becomes-.417/-.753 base/stress. Newrawtechnical/perpAlpha drops~1.7pt.
- Perp_scaled fallback-minus-hold allAlpha-.213pt/stress-.177,DD-.647/-.663.
  Basefold5Alpha improves4.240pt whilefold6 worsens5.538pt. Allpairedbear loss
  isfold6;bearfold8/10/11 havezero fallback andmatchparent exactly.
  Starttrend labels areNOTrealizedquarter direction:fold5startbullactualBH
  -47.766%,fold6startbearactualBH+92.069%. DoNOTcallfold6 a fallingmarket.
- Per-policy332fallbacks overfull8quarters;fold5has140intents/2SELLS,
  fold6has149/5SELLS,neitherhasfallbackbuys. Initialknownexposurefold5 1.13674,
  fold6firstepisode1.11045. Reducingwinning leverage duringrally, notforcedcash
  buying, isconsistentwithfold6loss. Laterownstate alsochanges:Oct6recovery
  same negativeforecast targets.92from1.0 vsparent1.02902from1.10902.
  Fold5May25recoverynewown1.0target1.08 vsparent1.22348hold;finalknownexposure
  1.15286 vsparent1.27840. No exactcausal attributiontoonefill claimed.
- Everymean332fallback submissions. Actualbasebuys/sells:zero7/0,fit3/0,
  scale16/2,technicalraw6/13,technicalscaled19/7,perpraw6/12,perpscaled14/9.
  Othersdeadbandnoops. Target1canincreasebelow1inventory andisnotriskfree.
- Independent1355hashchecks/964files;all56ownstatepaths20426decisions
  (18102learned2324fallback),112base/stressaccountpaths. Targets,trace,
  Alpha/DD/trades/costs all0difference;meanexposure2.22e-16,pairmeans5.55e-17.
  All72controls exactwith16BH/robust immediateparentprovenancepointers rebased
  andverified;21nofallback pathsfold8/10/11 exactlyparent.84unscoredinference
  decisions retained. Separatebehavioraudit binds56sourceforecasts/masks,
  reconstructsactualbuys/sells andall20,426knownstates. No independentrefit.
- All8 leave-one-out descriptions retained withoutomittingprimaryquarters.
  Absolutejointmeans survive scale8/8,technicalscaled5/8,perpscaled7/8;
  perpwithoutfold7 stressAlpha-.188pt. Scaleincrementalfallbackbenefit reverses
  withoutfold5. Thesearepostoutcomesensitivity,notconfidenceornewselection.

## Eighth completed stage: fixed half-weight shrinkage

The fixed half comparison is complete, registered in source commit `3957975`
before outcomes. Output `codex_outputs/oracle_mean_shrinkage_decisions_v1`;
report [oracle_mean_shrinkage_results_20260905.md](oracle_mean_shrinkage_results_20260905.md).
Run handle 95837 is terminal exit0. Full suite 537 tests OK in 56.481s,
handle 88212 terminal exit0, log `/tmp/oracle-mean-shrinkage-full-tests.log`.
No fitting or model jobs from this stage remain running.

- Preflight file SHA `31516273e780c4e3443500abd4b1f535f386e8b39bb5420e89e925af09006337`.
  Runtime registration file SHA `4bcdf123ab026ade5896667aa3c937440d825cccbc08e0acccdd591af7eb354e`.
  Results file SHA `a06c1ed0d6b85eb4d808c2741dca9cfd19b9d1f79023043ade3fbbb5897d212e`.
- Exactly two derived means: .5*saved scale_mean + .5*technical_scaled or
  perp_delay0_scaled. Each uses the unchanged technical_scaled HGB variance,
  under both own-inventory hold-on-missing and target1 fallback rules.
  No refit, new calibration, variance, support or architecture change.
- Four new policies + eight old controls =96 economic rows/targets,
  16 new forecasts, 16 new +24 copied scores, 32 new traces.
  Total158 adaptively explored names, not independent trials.
  Same 2586 inference/2574 scoring/332 fallback-eligible/2 missing current opens.
- Equal-quarter AlphaEx/DD pt base; stress:
  technical_half hold +1.149/-6.031; +.943/-5.953.
  technical_half fallback +1.007/-6.783; +.821/-6.718.
  perp_half hold +3.613/-5.388; +3.349/-5.305.
  perp_half fallback +3.379/-6.160; +3.144/-6.090.
  Four aggregate joint passes, only perp_half hold passes observed start-regime
  means at both costs. Zero coverage-qualified passes: 2 bull/4 bear/2 sideways.
  Joint quarters 3/8 for each technical rule, 4/8 for each perp rule.
- Perp_half hold start-regime base Alpha/DD pt: bull +4.222/-6.473,
  bear +1.724/-2.951, sideways +6.782/-9.175.
  Perp_half fallback bear Alpha -.066pt, stress -.330pt; fails.
- Relative loss reduction =1 - equal-quarter mean loss(candidate)/mean loss(ref),
  not mean of per-quarter loss ratios. Own full-mean MSE improves technical
  4.142842%, perp3.697151%; MAE3.016608%/2.705806%, both losses improve8/8 each.
  But MSE remains1.892217%/1.544775% above scale_mean, and2.600296%/2.250439%
  above zero. Half forecasts beat scale_mean MSE in only1/8 and2/8 quarters,
  and zero in only2/8 and3/8.
  Rank IC unchanged within all16 half-versus-full quarter pairs.
- Perp_half hold vs its full mean Alpha improves1.482pt, DD2.258pt;
  stress Alpha1.626/DD2.304pt. All three start-regime mean MSE and economic
  pairs improve versus own full mean. Versus scale_mean hold, Alpha+.485pt
  but DD+1.144pt worse. Mean trades57.625->29.750, turnover4.232->2.089.
  Do not claim the feature ranking signal improved or all gain is only costs.
- Perp_half fallback-minus-own-hold Alpha-.233pt/DD-.772pt. Both new rules
  retain all8/8 leave-one-out aggregate signs, but perp_half hold regime means
  fail if fold7 or9 is omitted. Technical rules retain6/8 aggregate signs.
  Descriptive sensitivity only, no exclusion/CI/selection. Perp-half vs
  technical-half Alpha difference still concentrated in fold12.
- Independent audit:1384 hash bindings/1121 files;64 copied controls and24
  endpoint scores exact;16 half formulas and40 scores;32 own-state paths,
  11008 decisions,64 base/stress accounting paths;48 unscored inference
  decisions retained. Targets/traces/Alpha/DD/trades/cost differences0,
  exposure2.22e-16; MSE identity5.42e-20. All6 no-fallback half-rule pairs
  match in folds8/10/11. No independent refit performed.
  Separate diagnostic binds55 sources, forecast references, eight half-versus-
  full/anchor economic pairs plus two fallback-minus-hold pairs, all quarters
  and all8 leave-one-out descriptions. The complete registered ten economic
  pairs plus two rule pairs remain in results.json and the accounting audit.

## Next: frozen family and confirmation protocol

Do not rerun the completed half/fallback stages or add a finer weight grid.
The tracked candidate_family_freeze.json closes weights0/.5/1 and pins all
four half candidates plus eight controls. No single strongest model selected,
no old alpha-DD/oracle/P1 lock changed. This is a family freeze for protocol
work, not a completed independent-confirmation registration.

The tracked confirmation_design_draft.md was prepared before half outcomes;
confirmation_access_audit.md was conducted on source/config/registration and
manifest path/hash metadata only. No new later prices/forecasts/outcomes were
read. Current val5–12 equals test4–11. Original alpha-DD development evaluated
test0–12, so test12 is also used. Historical15–23 and fresh24 are explicitly
reused in an existing registration. Test13–14 lie inside existing fitting /
inference scope; exact human/performance exposure is unverified. No completed
historical contiguous interval was certified unread. Absence of logs does
not prove independent data.

The first complete future quarter on the original calendar is test26,
2026-10-16 13:45 UTC. Proposed fixed tests26–37 end2029-10-16 13:45 UTC.
This is a design option, not an automation, scheduled job, commitment to wait
three years, or instruction to stop useful engineering. It does not guarantee
three quarters per regime. Do not pool old2/4/2, backdate forecasts, omit
failures, extend until favorable, or treat a reused historical report-only
replay as independent confirmation.

Next concrete work is to complete a separate frozen-procedure protocol and
adapter with receipt-aware support, calendar/training/maturity and failure
contracts, plus a serial-dependence-aware joint/multiplicity procedure for
four candidates and the specified endpoints. Preserve the past-only18/3/3
fit/scale/interval schedule anchored at each future evaluation start. Need
synthetic/data-only validation before any new-window scoring; no production
or real-money execution is implied. Statistical sign/count checks alone are
not a high-probability claim. Existing P1 scope and fail-closed gates remain
separate. A new method/version cannot reclaim observed periods as untouched.

The primary 2025 RFS paper How to Dominate the Historical Average was read
as research context. Its coefficient-sign/amplitude and error-distribution
conditions have not been verified for this fixed BTC half mixture; no claim
of inherited dominance and no new estimator/weight was implemented from it.

## Ninth completed stage: confirmation calendar, receipt and reporting contract

Implemented the next contract components without additional market/outcome
access, fitting or selection. See [oracle_confirmation_contract_20260906.md](oracle_confirmation_contract_20260906.md)
and `configs/oracle_confirmation_contract_20260906.yaml`. The earlier Next
section remains the broader incomplete confirmation objective; these components
are now done and must not be restarted or represented as a full runner.

- New `oracle_confirmation_contract.py`: test26–37 calendar anchored at actual
  test_start S, fit[S−24m,S−6m), scale[S−6m,S−3m), interval[S−3m,S), eval[S,S+3m).
  Exact inherited segment masks at the new start, strict375min fit/cal maturity,
  inclusive evaluation maturity, full15m grid. Future-label gaps never gate
  inference. Final preflight has4384 nominal6h decisions, not observed coverage.
- New `oracle_receipt_support.py`: required bar rows only, fixed rawclose
  open+15m−1ms, featurebar end<=nominal origin t, receipt<=explicit deadline.
  Config proposes t+60s before next15m fill. Current unfinished featurebar
  remains excluded. Late/unknown receipts are not backfilled. Generic helper
  accepts deadlines in[t,t+15m), eventual adapter must enforce exactly60s.
  Does NOT prove current-open observation, value validity, complete rolling
  history or receipt authenticity. Training-time archive snapshots and actual
  evaluation receipt logs need distinct bindings; do not invent historical
  origin receipts for old training bars.
- Fixed4 candidates/8 controls, exactly21 source/metadata bindings plus separate
  family SHA and its10 development sources. Aliased or duplicate bindings,
  missing required dependency paths, complex/bool/array/nonfinite metrics are
  rejected. Stable finite quarter means with post-aggregate guard.
- Descriptive report adapter requires every12quarter x12policy economic row,
  five mean loss rows perquarter, paired regimes and scoring denominators.
  Inventory64 economic +16 predictive primary endpoints,96 candidate-component
  references. Predictive references are own full mean and scale_mean; same
  forecasts shared by both rules. Equal-quarter favorable effects, no averaging
  percentage loss ratios. Missing quarters fail, absent regimes remain null.
  Observed metric/coverage conditions are separate from provenance/true bar
  calendar/selection/high-probability flags. P-values staynull, inferencefalse.
- Primary research: candidate conjunction can use IUT max of valid marginal
  p-values; across4 candidates Holm is needed for any-candidate discovery.
  These do not make uncalibrated12quarter/3-per-regime marginal p-values valid.
  No significance engine enabled. Do not claim95% future success from point
  conditions, coverage, more bootstrap repetitions or passing unit tests.
- 27new targeted tests; full564 tests OK56.618s, final handle28741 terminalexit0,
  `/tmp/oracle-confirmation-contract-full-tests-v2.log`. Initial implementation
  run58560 terminalexit0; rerun justified by independent numeric/hash findings.
  No live model, fitting, collection or verification processes remain.
- Final metadata-only preflight output
  `codex_outputs/oracle_confirmation_contract_v1/preflight_v2.json`, fileSHA
  `6ac0ec6dc72fd8458c3bb044168b13e74305893a7c0d858422111a6be450fc53`.
  Earlier preflight.json is an unregistered implementation draft, not final.
  Final source contract SHA7b5ea82703f931579b12459e189c0006024b9ef7f41147f711e2051ee59eb503;
  receipt SHA97a6fee8837fbb741aedd1759d6134eaa6f1b45f11b1b8941ed4637da3dedb2c.
  Independent limited review verified malformed inputs and timing synthetically;
  root preflight verified the remaining metadata bindings. Tracked evidence
  has exact preflight/review copies, tests+source verification and manifest.

Next concrete implementation: new frozen procedure adapter for fit/calibration,
prediction, own-state planning and accounting, retaining original dependency
mask. First replay only existing development8quarters to check parity against
saved half/full/anchor controls; retain hashes, fit seeds, scale anchors,
technical variance and all endpoint-score provenance. No new performance-based
weight, horizon, support or missingness choice. Then separate report-only
additional-window registration if useful; do not repurpose old test as a
selector or independent data. For prospective use add authenticated receipt /
input snapshot / actual submission and fill logging before any outcomes. The
current preflight explicitly says ready_for_outcome_scoring=false. Do not
report this contract-only milestone as independent confirmation completed.

## Tenth completed stage: fixed refit through full policy parity

Completed the next fitting/inference/accounting integration on existing
validation5–12 only. Source registration commit `f284124` preceded all real
refits. Report [oracle_frozen_procedure_parity_results_20260906.md](oracle_frozen_procedure_parity_results_20260906.md),
registration doc and tracked evidence are now available. This closes the
same-development parity step in the previous section, not future collection
or independent confirmation. No new candidate/period count; total adaptive
family names stays158, no prior selector/lock changed.

- New reusable `oracle_frozen_forecasts.py` fits technical/perp_delay0 Ridge100
  means and ONE technical HGB variance, preserving2-thread/seed20260905 and
  original arithmetic. Five inference-only means, common technical variance,
  raw calibration predictions, models and JSON provenance. Selectedfit/cal
  labels only; later labels cannot determine forecast availability. No new
  architecture, window, mean weight or policy selection.
- New `oracle_frozen_procedure_parity.py` deliberately permits ONLY oldval5–12
  mapped to calendar test4–11 (same2021-04-16→2023-04-16), not prospective26–37.
  Rebuilds full common dependency set; checks olddelay preflight deep equality,
  all48 original segment masks, regimes, exactNaN/time/mask/target semantics.
  Preserves BH/common_robust, own-inventory rules and samebase targets at2xcost.
- Data-only preflight verified1064 source artifacts and2586inference/2574score
  rows. FileSHA4dcf2dacf2ca6a43d23d90ffc0c8cd4d8bdfd3be6015978e3a98baae221a86ae.
  Runtime registrationSHA886f7a1a4fe494f7c5131ba6eae6384cd9d1ae64d048fecbabd87247c3d667f5.
  ResultsSHAc19b772e4f53b5a749839553b82c439ef4684a2ec9efe523464c533e8beed366.
  Output `codex_outputs/oracle_frozen_procedure_parity_v1`; handle24133 terminal
  exit0, all8folds completed. Do not restart or refit this completed stage.
- All40 forecast/score streams,96 targets/rows,192 accounting objects,
  80 utility traces and16 calibration arrays+8scalar records reproduce with
  max difference0.24 saved models;264 generated artifact paths(33perfold).
  Models are fresh fits; byte bindings prove saved-object identity, not a
  separate statistical discovery. Shared mean/risk inputs keep old scores.
- Full581 tests OK56.488s, handle70772 terminalexit0, log
  `/tmp/oracle-frozen-procedure-full-tests.log`.17new tests include504 scalar
  synthetic Ridge predictions and ignored nonsensical evaluation labels.
  Independent artifact audit:543 file bindings, original239 reference paths
  unchanged,48 mask hashes,all datasets/traces/calibration matchexactly.
  27520 utility decision rows. Independently recomputed80 source/generated
  score records differ max1.7347e-18. No independent economic simulator/refit.
- Runtime emitted96 sklearn matmul warnings despite all outputs finite/exact.
  Additional scalar audit loads16 SHA-bound Ridge models and recomputes15056
  model-row predictions withoutmatmul/predict/transform:9884rawcal predictions
  max5.2042e-18 and5172scaled inference max3.4694e-18, within registered
  rtol1e-12/atol1e-14. Coefs/scalers/intercepts finite, scalespositive; verifies
  19source/56generated artifact/8fold bindings and full/selected feature digests.
  Does NOT identify warning cause or prove training/HGB numerical correctness.
  Scalar auditSHA0bfcffaeecf77dcd4388ccd61bc14f371658c5b3781c5b71d5ca56b20b7477fa.
- Resume validation checks exact33artifacts+fullrow/score/trace/calibration
  inventory and source parity. Its maxdiff fields retain original recorded
  values rather than reaggregatingmaxima; limitation registered before run.
  This run was allfresh, and separatepostrun audit recomputedmaxima. Do not
  change already registered sources merely to rerun the completed comparison.

Next concrete action: register a separate fixed report-only additional-period
adapter/data preflight for all4 half candidates+8controls. The reusable fit
helper is ready, but the parity harness must not be pointed at other folds.
Preserve the access audit: historical15–23/fresh24 alreadyreused, no claim of
unread/independentdata. Acquisition/quality proof for any required additional
UM/Spot months must precede new outcomes; input manifests/metadata/coverage only
until a fixed report-only registration. Do not choose weight/rule/features
from these test outputs. Prospective receipt/current-open/intent/fill logging
remains a separate unimplemented integration boundary. No model, fitting,
collection, audit or verification process from this stage remains running.

## Original Spot / perpetual design rationale, now executed

Keep model architectures small and fixed. Proposed three matched-support groups:
existing technical29, +2perp weightedflow24/96, +all8derivative features.
Eight extras: perpflow24/96, perp-minus-spotflow24/96, log relativevolume
intensity24normalized672, logperp/spot trade-close premium, its24bar change,
log past24bar realizedvariance ratio. Flow=(2takerbuyquote-quote)/quote,
aggregated with paired observed quote weights. All inputs<=t-1.

Availability and period gates must be committed before performance. Early
Spot folds cannot pretend to have prelaunchUM training data. Refit parent on
identical support; do not compare derivative survivors with previous13fold
means. Prefer the same18/3/3 calibration timeline, if data-only512/64/64 row
gates can be met. No omitted/fresh test scoring is authorized as selection.
Data-only preflight is now saved at
`codex_outputs/oracle_derivative_preflight_v1/preflight.json`, with exact
feature equations,19 input/source hashes and all period counts. Its source
script is retained beside it; no model or performance was computed. Direct
2year fit counts first pass atfold3; disjoint18/3/3 counts first pass atfold5
(2021-04-16). Atfold5 h24 fit/scale/interval=800/233/279. Eligiblefolds5–12
have2587 causal6h feature rows andbull2/bear4/sideways2 quarters. Existing
minimum3quarters/regime fails before modelling and must not be weakened.
The data are sufficient for a matched predictive-information diagnostic, not
for claiming the overall trend-robust goal has passed. See the committed
preflight report and `oracle_risk_reliability_evidence_20260905/derivative_preflight.json`.
Silantyev2019 supports contemporaneous flow impact, not6h OOSalpha. New2026
quarter-hour paper's4–12h association is full-sampleOLS, while its rollingOOS
is10secondreturns; our15m bars cannot reproduce its10second burst features.
That original untested information hypothesis has now been evaluated above;
the positive and failed outcomes must remain in the research record.

Keep return forecast losses, risk forecast losses and economic value separate.
If a policy clears exploratory trend/cost signs, freeze it before constructing
a prospective paper protocol and selection-aware dependent-data inference.
No reused development result alone establishes high generalization probability.

## Original next-step rationale, retained for context

1. Measure calibration of the promising6h risk forecast on strictly disjoint
   fit/calibration/validation chronology; compare with a causal persistence
   forecast, not just the training climatology. The current MSE skill reference
   is too weak to establish superiority over a strong risk baseline. Predeclare
   calibration method and all candidates before outcomes, and preserve all
   failed comparisons. Do not treat overlapping6h/7d labels as independent.
2. Measure decision value with a calibrated conditional distribution or joint
   scenarios. The failed ratio controller shows that forecast improvement alone
   is insufficient. Keep transaction costs, endogenous inventory, turnover,
   downside and B&H-relative utility aligned. No hindsight teacher forcing.
3. If return forecasts still fail strong causal baselines, examine new
   information: contemporaneously available Spot/perpetual flow divergence and
   funding/basis, with data availability provenance and matched support. More
   transformations of the same OHLCV cannot create independent information.
4. Freeze any resulting viable policy before prospective data become available.
   Prior historical/fresh quarters, omitted13/14 folds and current development
   periods are not untouched confirmation. A future forward paper protocol
   must define its cutoff, update rule, costs and per-regime sample requirements
   before scoring. A favorable reused-validation result is only a candidate.

No scheduled automation has been created; continuation is through this active
Codex goal. No notification claim should imply an external monitor is running.

## Eleventh completed stage: fixed family on ten additional reused quarters

The separate original test15–24 replay is **completed**, not pending. Read
`docs/experiments/oracle_additional_window_results_20260906.md` and evidence in
`oracle_additional_window_evidence_20260906`. Do not restart the completed output.

- Acquisition registration commit `f2e1faf`; final adapter/config/preflight and
  preregistered scalar audit commit `73cb806bffb7dc39d7454886e9bcfce9119dc435`.
  Source `oracle_additional_window_replay.py` SHA256
  `f82d48f25f4234a858a872bddd0c9aac498b4b4ccfc9fd56686041ee1fdf260c`.
- New UM namespace `checkpoints/oracle_additional_window_data`, January2022–
  July2026, 55/55 checksum-verified archives,160608 observed raw rows,0 missing.
  UM SHA `02d2f07679db0087904b923606a501c8494afda0f75c6a4c94bf4b38ad49a583`.
  Spot source remains unchanged. Data manifest178bindings, SHA
  `609ae008f1cb47919d19f5997409f5833fed1a12d5c106a32af985377945088c`.
- Data-only preflight SHA
  `8416f5a21389606e9a1834bb12898349178d34e2cc8a92ca6041a0f3241228b5`.
  Original test15–24 spans [2024-01-16T13:45Z,2026-07-16T13:45Z).
  All price grids and boundary bars pass;3648scheduled,3620inference,3610score.
  There are28feature-unavailable scheduled rows in fold18 only, with known opens.
  Start-regime counts4bull/3bear/3sideways pass the count condition only.
- Run session8868 terminalexit0: all10quarters,120economic rows,50scores,
  100traces,30models,330artifacts. Family remains4candidates+8controls;
  no new weights/policies (158explored names unchanged), no selection.
  Registration-file SHA
  `b1838eca2efc6523537a49415803eb5f9a07a7603257fd11f6abaa1c4d5653eb`;
  results SHA `8579e40b5be9ed737acf6c633c92d33b383d0fb338c75832f7b3a125bdd474d1`.
- Base/stress aggregate AlphaEx/MaxDDDelta in percentage points:
  technical-half hold +1.336/−2.992 and +1.162/−2.971;
  technical-half fallback +1.542/−3.015 and +1.371/−2.995;
  perpetual-half hold +0.048/−2.826 and −0.130/−2.804;
  perpetual-half fallback +0.866/−2.853 and +0.680/−2.831.
  **All4 fail both regime-wide economic and predictive conjunctions.**
  Every half candidate has negative bullAlpha and sidewaysAlpha plus positive
  sidewaysDDdelta. This contradicts a trend-robust economic claim despite coverage.
- Half MSE improves versus own full mean by0.800%technical/0.781%perpetual,
  each8/10quarters, but worsens in bull. Versus the scale mean, aggregate MSE is
  worse0.0565%/0.0148%; neither establishes strong-baseline predictive superiority.
  All20 within-quarter half/full rank IC comparisons are unchanged.
- Existing common_robust control passes all observed regime economic means in
  these10quarters (+1.645/−4.752base,+1.229/−4.655stress) but failed the earlier
  development8 aggregateAlpha−0.317 and bearAlpha−6.610 (stress−0.651/−7.002).
  Separate-window evidence is saved; do not select/promote it from these tests.
- Required full suite589testsOK56.591s. Independent scalar audit session19505
  terminalexit0:100own-state paths/36340decisions,240account paths,50scores,
  20half identities, maxdifference4.441e−16. Independent Decimal60 summary audit
  checks80components,5forecast/10policy/2rule contrasts; max7.105e−15.
- Runtime produced120linear-model matmul warnings. Separate saved-model scalar
  Ridge audit session40863 terminalexit0 checks20models/21640predictions
  (14400rawcal+7240scaledinfer), max3.469e−18, finite parameters/positive scales.
  This does not prove warning cause or independently audit HGB training.
  All registered source/data hashes remain unchanged. No live process remains.

Next research direction remains development-only. First compare the existing
RL/ML hindsight diagnostics with the now fixed common-support, conditional
utility and own-inventory contract; register a bounded decomposition of mean
forecast error versus risk/decision limitations if those oracle contracts are
not matched. Use original development intervals for hypothesis testing and
candidate selection, retain all failures and simple constant/rule controls,
and do not tune a new weight, fallback rule or regime threshold from test15–24.
Architecture optimization remains deferred. No independent/prospective or
high-probability result has been established; the broader goal remains active.

## Twelfth completed stage: matched ML information and RL hindsight decomposition

This stage is **completed and independently audited**. Read
`docs/experiments/oracle_information_decomposition_results_20260906.md` and the
self-contained evidence in `oracle_information_decomposition_evidence_20260906`.
Do not restart `codex_outputs/oracle_information_decomposition_v1`.

- Source/config/preflight registration commit
  `d3b25734a34915049a327256bd9f99cd9aea8336`, pushed before real execution.
  Config SHA `f10c14e65c9be4edd1d56c303dce9262a79f22fe5c5a3686454b66e6202512d4`;
  preflight SHA `ae66338b39253f88e536729948b96b9eb57abe9ba409cbfc074f339be688f4e7`.
- New immutable modules are `oracle_information_interventions.py`,
  `oracle_information_decomposition.py`, `oracle_matched_hindsight.py`.
  Matched helper SHA
  `e339698814a66139246f34aa9a02b83f5aa16ad8051338a62444f6a12d1769e8`.
  All23 source bindings remain unchanged. Original development validation5–12
  (original test4–11), April2021–April2023 only; no additional-test15–24
  outcomes used, no data acquisition or fit, no causal candidates or selection.
- 28policies =12frozen causal controls +12hybrid substitutions +4matched RL
  beams. Real run session27200 terminalexit0,8folds,224economic rows,
  448cost accounts,48hybrid forecasts,224targets,128traces,400new artifacts.
  Runtime registration byte SHA
  `8e25d7743e49e9df6269cb898808ee1fde691a52d56003c69b52ce9497560fc1`;
  results SHA `f5597dee653a45ee612766111da578a868969a51211f923493763b4059a18ac7`.
  The embedded registration hash is canonical-content, not that file byte hash.
- ML hybrids retain2586inference/2574replacement/12learned-remainder rows per
  mean stream and332fallback opportunities plus2missing-current-open rows.
  Return replacement is realized h24 log return; risk is realized h24
  volatility squared, without floor/factor24/annualization/reapplying half.
  The two realized targets are not conditional expectation/variance. Missing
  availability never uses future score support. Every controller owns its
  cash/units state and traces are explicitly hindsight-only, teacher-disabled.
- Return-only hold has AlphaEx/MaxDDDelta **+23.198/−11.805pt** base and
  **+21.767/−11.604pt** stress; return-only fallback +26.011/−11.835 and
  +24.547/−11.599. Relative to each own learned half, all4return substitutions
  add **+19.585 to +25.004pt AlphaEx** base. Risk-only additions are much
  smaller: **+0.370 to +0.587pt**. These are intervention effects on reused
  development data, not learned predictive/economic improvements.
- Adding realized risk to realized return slightly worsens both aggregate
  metrics: AlphaEx−0.070 to−0.126pt, DDdelta+0.262 to+0.288pt. Preserve that
  adverse result; do not claim more Oracle risk information further improves.
  Technical/perpetual return/both target paths happened to match in32paired
  folds after running; all registered variants remain separate and audited.
- RL beam32 uses matched own-open dynamic intents, fixed hold/fallback,
  next-open fills and base-objective `log NAV − penalty * MaxDD`, penalties0/1.
  Base AlphaEx/DDdelta: hold0 +36.775/−16.266; hold1 +35.380/−17.220;
  fallback0 +37.050/−13.502; fallback1 +36.263/−13.936pt. All32 searches prune
  distinct branches; no incumbent selected. Each output is a feasible lower
  bound on maximum hindsight objective for its action set, not global optimum
  or a causal-performance upper bound. Stress reuses exact base intents.
- Penalty1 has worse objective than penalty0's feasible path when both are
  scored under penalty1 in4/8quarters for each rule. This explicitly confirms
  real finite-search suboptimality. No post-result envelope/replacement chosen.
  RL has a different horizon/objective from local h24 utility, so its gap from
  ML hybrids cannot identify a decision-only bottleneck.
- Full612testsOK59.254s before outcomes. Independent scalar audit passed
  1328ancestor/400new artifacts,2159binding checks/1787distinct files;
  all448account Alpha/DD/turnover/fees/borrow/trades errors0, mean-exposure
  error2.22e−16. All96hybrid paths33024decisions and32RL paths11008decisions
  plus32rule-matched incumbent accounts match;48substitutions/96controls/
  1440summary scalars exact. Independent h24actual error≤3.96e−15.
  No canonical planner/simulator/metrics/search imports; finite search
  expansion itself not independently rerun. Scalar audit SHA
  `f3de0fa4d69da2e615a839bfb90ffc9d9852a42e5a0a775d4d5b9f1747b6c343`.
- Separate Decimal60 summary audit all28policies/12swaps, maxdifference
  3.88e−15 turnover, exact RL objectives,446direct bindings. It does not
  independently rehash all ancestors. Audit SHA
  `b1f486c3147e86835ed93bb07817b761ff8333772df673239043330b1a753cf6`.

The matched diagnostic supports studying **causal return information and
reliability before model architecture**. It does not tell whether existing
technical indicators can supply that information or whether features, Ridge
specification or calibration is the cause. Next stage should predeclare a
small original-development-only return-information study, with matched support,
strong constant/simple causal baselines and separately assessed predictive and
economic outcomes. Do not reuse the later test15–24 results for tuning or
select a common_robust/half/fallback winner from them. All4frozen half candidates
still fail the additional-window trend-wide conditions. 158explored causal
names, all locks, missing rules and architecture remain unchanged. No causal
strongest model, independent confirmation, prospective receipt proof or high
probability of generalization has been established. The goal remains active;
no live real run remains and no scheduled automation was created.

## Thirteenth completed stage: scale-only return reliability

This stage is **completed**, with no live run. Do not restart
`codex_outputs/oracle_mean_reliability_decisions_v1`. Read
`docs/experiments/oracle_mean_reliability_results_20260906.md` and the evidence
in `oracle_mean_reliability_evidence_20260906` before further development.

- Source/config/protocol/tests/preflight freeze commit
  `ff9bbb92d588b4615ae8352b353a1301751202a7` was pushed before new coefficients.
  New modules `oracle_mean_reliability.py` and
  `oracle_mean_reliability_decisions.py`;25registered source bindings unchanged.
  Helper SHA `2e7b1e0171f353f93d8a455481846bc03cc814fa68c2a8c8e49ffc7f539d3e5d`.
  Config SHA `71b0965691099074a690085ac22cbe6808b2b430fea305ec2b75480db1c2f094`;
  preflight SHA `d1304dc98df0595b68992b6605c78bd3527d3b811209071df12cb6594d294370`.
- Original development5–12/test4–11 only. Two frozen Ridge mean streams and
  shared technical risk unchanged. New16scalar weights estimated only on
  scale[E−6m,E−3m),0base-model fits. No interval/evaluation label updates,
  new features, architecture, missing rules or additional-test15–24 use.
  Full Spot file is decoded then sliced; no later semantic outcomes are used.
- With savedfullp=raw+savedbias and exactanchorscalea, define d=p−a,r=y−a,
  B=meanS(d²),C=meanS(d*r). w=0ifB==0orC<=0,1ifC>=B,elseC/B; convex sum
  w*p+(1−w)*a, with exact endpoint copies. It is the fitted least-squares point
  on the anchor/full segment, not a selected0/.5/1grid or raw-centered OLS.
  Weight fitting onS cannot count as demonstrated improvement. All inference
  2586/score2574/fallback332/missingcurrentopen2 supports remain unchanged.
- New4causal policy names =technical/perpetual reliability×hold/fallback;
  adaptive causal-name ledger increases158→162. All12prior controls retained;
  frozen family/locks not rewritten. Nine means scored onS/I/E =216records.
  S/I strata by later evaluation-start regime are explicitly retrospective,
  not conditions available at their own forecast times. Evaluation regimes
  remain known at the first evaluation decision;2bull/4bear/2sideways.
- Required626testsOK56.696s,10helper+4runner new synthetic tests. Realrun54020
  terminalexit0;128economic rows,16weights,208artifacts (16evaluationNPZ,
  16calibrationNPZ,16weightJSON,128targetNPZ,32traceJSON),8fold manifests.
  Registration byte SHA
  `d15c59fe44489076fe332f0e0b4ee511723d19b312952dd20b0f684ae638bf97`;
  results SHA `333f88d4bc06f671d552d8ca70470ee60ecd67074812f6aa248b26f1b94562f1`.
- Observed weights13interior,2upper (technical/perp fold7),1zero (perpfold12,
  nonpositive crossmoment), nozero-dispersion. Technical weights folds5–12:
  .264776,.368151,1,.114233,.255469,.581460,.081524,.093329;
  perpetual .472445,.271626,1,.061155,.270464,.649504,.078631,0.
- **All4policies meet observed economic mean signs in all evaluation regimes
  and both costs, but only3/8individual quarters meet both signs under both
  costs for each policy.** Equalquarter AlphaEx/DDdelta points:
  technicalhold+3.750/−5.695base,+3.531/−5.641stress;
  technicalfallback+3.704/−6.378base,+3.546/−6.311stress;
  perphold+4.450/−5.143base,+4.248/−5.056stress;
  perpfallback+4.208/−5.854base,+4.039/−5.779stress.
  Versus ownhalf, Alpha improves but DDdelta is worse by+.244to+.405pt base.
  Do not call this joint domination of the parent or trend-robust success.
- Evaluation MSE improves vs ownfull5.162%technical/3.990%perp and ownhalf
  1.063%/.304%, but is **worse than scaleanchor .809%/1.236%** and worse than
  zero. Only1/8quarter beatsanchor in each stream (perp1tie). Interval also
  loses toanchor .527%/.104% and tohalf .159%/.110%. All4predictive conjunctions
  fail both I andE; regime-count and high-probability conditions remain false.
- Exact loss identity retains centered and mean-drift terms. Evaluation
  (centered,drift,totalexcess) in squaredlogreturn×1e6:
  technical(−.228515,+2.797574,+2.569058),
  perp(−.609196,+4.534918,+3.925722). Drift outweighs the favorable centered
  component. This is algebraic attribution, not proof of a specific causal
  stale-bias mechanism or of stable technical information. Original fullscaled
  driftterms17.4861/14.9228e−6; rawfull totalexcess11.4932/9.9928e−6 remains
  positive. Avoid another static-weight search framed as confirmation.
- Independent scalar audit passes1328ancestors+208artifacts,16weights,
  216scores/decompositions,40parentscores,96controls,32ownstatepaths/
  11008decisions and256base/stressaccounts. Weight/forecast/target/utility/
  Alpha/DD/turnover/cost differences0; meanexposure2.22e−16. All6observed
  endpoint policies match existing targets/accounts;48unscored inference
  decisions and664fallback decisions retained. Identityresidual≤9.32e−20.
  Audit SHA `8d8f14bfd821ad83dab0eba44d53466ff2c6e900dfcd3debd4ab10ac67d5da7a`.
  No canonical helpers/planners/simulators/scorers imported. No base fits,
  extra policy trials, live data or paid compute during the audit.

- Separate Decimal60 summary audit passes128rows/216scores/16weights;
  maxsummarydifference1.51e−15,scalarrescore4e−18,weightalgebra0. It checks
  302direct/consumed files (208new,25source,56consumedancestor), not a second
  rehash of all1328ancestors. All45positive-weight segment ranks unchanged;
  3perpfold12 zero-weight segments constant. Summary audit SHA
  `8bd15dbdddca697cc020208db44e1e117b8b3d8f66c53eb6c002a270b7b022d0`.

## Stage14: causal rolling centering completed and rejected

- Source freeze `845a9fd167533599c7189019b38dc9ca2edf0f41`, report commit
  `2253b2c`; see [full report](oracle_rolling_centering_results_20260906.md).
  One completed process91776, zero base/weight fits, 645 tests pass.
  Exact causal3-calendar-month histories use only matured labels and samefold
  saved raw predictions. No window search or additional-test15–24 tuning.
- Three means ×2 missing rules added6names, inventory162→168. All6 economic
  and predictive all-strata flags fail. Technical/perp MSE improves vs own
  reliability0.96026%/1.38890%, but loses dynamic anchor0.27025%/0.25924%
  and zero0.53454%/0.52350%. All6 base Alpha means negative:−4.26887to−1.52046pt.
  Negative mean DD does not repair the failed Alpha objective. Both learned
  signals lose dynamic anchor in every regime; bull/bear Alpha is negative.
- 256 new artifacts,176 rows/352 accounts,96 E scores independently checked.
  48 new own-state paths,16 immutable weights,all2586 histories exact.
  Anchor/perp fold12 DD=−1.11e−16 ratio is machine equality, not substantive
  improvement; strict stored counts remain unchanged and are footnoted.
  Results SHA `d5adfc39e2822bfb77aaa519202b9949d664f772d704bcd63fa535147da9b620`.

## Stage15: short-price/flow representation completed and rejected

- Source freeze/push `d11cfee15b77f773e353c7ecb1ba4729c0b4abe7`;
  [registration](oracle_short_feature_registration_20260906.md),
  [primary research](oracle_intraday_feature_information_research_20260906.md),
  [full report](oracle_short_feature_results_20260906.md).
  One completed process85763, no retries;32 Ridge fits comprise8 baseline
  reproductions+24 new means. Zero risk/weight fits.669 tests pass58.110s.
- Fixed2×2 groups29/34/32/37 use original technical29 plus price5,flow3,both8.
  Price is return4/16/48,body sign(log C/O),candle close location;flow is
  Spot/UM weighted takerflow4 and Spot quote activity mean24/672. Exact rawgrid
  and1bar shift. The new37 is distinct from old derivative37. Adding correlated
  coordinates also changes Ridge regularization; individual effects unidentified.
- All original6 masks unchanged:800/1034/1313/1500/1503/1634/1672/1794 fit rows,
  2586 E inference/2574 E scored/332 fallback/2missingcurrentopen. Shared old
  technical_scaled variance. I diagnostic only;I grouped by E-start regime is
  retrospective and not independent replication. No S mean calibration.
- Three means×2 rules add6 names, inventory168→174;28 old controls→34 policies,
  272 rows/544 accounts,184 scores (I48/E136),424 artifacts. Old technical
  model/scaler state and S/I/E raw predictions reproduce exactly, maxdifference0.
- All3 new E MSE worsen vs technical_raw:price0.18463%,flow0.08352%,both0.10830%;
  each improves only3/8quarters. All3 lose zero/fitmean overall and every E
  regime. All6 economic all-strata flags and all I/E predictive conjunctions
  fail. New strict joint economic counts2/8 each, no tinyDD among new policies.
- Equal-quarter base Alpha/DD(pt): pricehold−1.78615/+0.03824,
  pricefallback−3.69021/+1.05136,flowhold−0.41204/−0.82278,
  flowfallback−2.65188/+0.47320,bothhold−0.05881/−0.82345,
  bothfallback−2.14277/+0.32178. All6 sideways means have negative Alpha and
  positive DD under both costs. Flow/both improve own technical_raw economic
  means, but absolute Alpha stays negative and mean predictive losses worsen.
- Independent32-model normal-equation and scalar forecasts pass;maxforecast
  difference6.64e−17, saved-model scalar difference6.94e−18. Runtime emitted
  divide/overflow/invalid matmul warnings; cause remains unestablished, counts
  unavailable. Exact baseline parity and all finite supports retained.
  Separate184-score/summary audit passes (2273 distinct files),224 old control
  rows and96 old E scores exactly unchanged. New32 fits were not rerun for audit.
  Independent scalar own-state audit passes544accounts,48new+16baseline paths/
  22016decisions. Targets/NAV/utility/Alpha/DD/fees/borrow maxdifference0;
  meanexposure2.22e−16.72unscored new decisions+996fallback decisions preserved.
  2637SHA bindings checked, no canonical planner or simulator imports.
  Results FILE SHA `2f65382455e1266e035d38007a8dc939efb74d84df676c44da8ed3de9cd77218`;
  registration FILE SHA `43ca3c37e272168a73a26b370c04b98ef394fada8f91d819d8765f453d629682`;
  canonical registration object digest
  `b363840afa4a6b3b6c263369c316fa8596141669e8fa82bbf84c0df4970275ee`.

Next work should examine the gap between the same-execution future-return
Oracle and weak causal forecasts before another window/architecture search.
A bounded preregistered information diagnostic could separate sign, magnitude,
and large-move contributions; no such diagnostic is yet executed or selected.
Do not treat Oracle paths as causal teachers or a globally optimal upper bound.
Keep original reused development only, preserve all supports/costs/missing rules,
and predeclare controls and failure criteria before outcomes. Additional-test
15–24 stays report-only; no winner/column selection from Stage15.174 explored
causal names and all prior family locks remain intact. Reused8quarters,overlapping
histories,2bull/4bear/2sideways coverage and absent receipt provenance still do
not establish independent confirmation or high-probability trend robustness.
The goal stays active without automation, production promotion or live trades.
