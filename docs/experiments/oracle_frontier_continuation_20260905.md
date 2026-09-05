# Goal continuation: Oracle / features / trend robustness

Overall goal remains active. No candidate has established high-probability joint
AlphaEx>0 and MaxDDDelta<0 across trends. Do not mark this goal achieved because
the exploratory means or an implementation test passed.

Working branch: `exp/oracle-feature-frontier-20260905` in
`/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905`.
Initial experiment source commit: `ed23b49`; risk ablation source: `0272879`.
Runtime: `/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/.venv/bin/python`.

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
  bars remain ineligible. No derivatives model fit/result has run yet.

Runtime outputs, immutable and locally retained:
`codex_outputs/oracle_risk_baselines_v1`, `oracle_risk_calibration_v1`,
`oracle_conditional_decision_v1`.
New raw data: `checkpoints/oracle_derivative_data/um_15m.parquet`,
SHA `c365cbb6cd6d46dafea19b7fd3a62cb1b91ae3169f7a27532e40486ecfebfcdd`.
Sidecar, availability parquet, source ledger and40 raw verified ZIPs retained.
Index is barOPEN, not decision time. All features shift exactly once.
Historical receipt/publication timesunknown; live_causal_eligible=false.

## Next useful investigation: new Spot/perpetual information

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
Silantyev2019 supports contemporaneous flow impact, not6h OOSalpha. New2026
quarter-hour paper's4–12h association is full-sampleOLS, while its rollingOOS
is10secondreturns; our15m bars cannot reproduce its10second burst features.
The proposed derivative feature family is an untested information hypothesis.

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
