# Reliability stage evidence audit

Experiments are adaptive followups on reused development validation. Source and
configuration commits preceded each family's outcomes: persistence65a1d4f,
calibration8e3dbd9, conditional decisionsb56551d, raw UM acquisition83f72a1.
`manifest.json` binds nine exact snapshots to locally retained output files.
Raw forecasts/models/targets and checksummed source data are retained outsidegit.
No new test/outer result or deployment was produced.

## Persistence comparison

- Same3863 rows across3 HGB forecasts and3 persistence baselines.
- Six sources, data proof,39 forecasts and39 models hash-checked.
- Independent past-price scalar calculation of11589 baseline cells: max4.22e-15.
- All13 rowwise NPZ hashes checked; aggregate losses reconstructed.
- Nine focused tests passed.

## Disjoint calibration

-18month fit,3month mean/variance calibration,3month interval calibration,
  then3month validation. Label ends strictly before each boundary.
- Sixsource,52model,156forecast and338target hashes verified independently.
-156 forecast scores/masks reconstructed: max metric difference3.65e-13.
- Fold0/6/12 all6methods: meanbias/variance scale/quantiles reproduced.
-4824 scalar predictions checked; return3.47e-18, variance2.60e-18 maximum error.
-12 independent bisection cash/units replays: max metric difference2.91e-14.
- Numerical variance clipping did not activate in this run.
- Scaling also adjusts return mean, so economic changes are not attributed
  solely to variance calibration. Temporal dependence prevents automatic
  conditional coverage guarantees from split quantiles.

## Conditional decisions

- Current-open inventory and frozen forecasts only. Current close, next open,
  future gap mask and future target values are absent from action selection.
- Endogenous planner vs canonical account: all117 equity/exposure paths and
  turnover/fees/borrow/trades match exactly.
- Four source,39 forecast and117 target hashes checked independently.
- Independent Python reconstruction of27 policies,54 base/stress account
  paths,6876 decisions: target3.44e-14, score2.54e-17, AlphaEx1.20e-13,
  MaxDDDelta1.25e-14 maximum differences; trade-count error0.
- Independent sample handles1260 missing closes across repeated paths. No
  missing-open skipped fills occurred in that real sample; synthetic tests
  cover that case. A distincthold action staysNaN and never silently rebalances.
- Ten focused tests passed, including passive exposure outside targetbounds.
- Risk0 forecast-family names have identical targets by construction; they
  are not independent improvements. Stress replays base-generated intents.

## UM acquisition

- Forty available monthlyZIPs verified against official SHA-256; four404
  months remain missing. All rawCSV numeric values reparsed against parquet,
  max difference0. Source/schema/time/grid failures are fail-closed.
- Raw index is baropenUTC. Eligible decision timestamps must precede the
  existing cutoff. April-tail observed rows are retained but excluded.
- Archive checksums do not prove historical live arrival. Nozero-fill for
  unavailable prelaunch bars and no causal-live-ready claim.

## Final checks

`python -m unittest discover -s tests -v`:436 testsOK in56.154s.
`git diff --check`:passed. The report figure was visually inspected and its
legend repositioned to remove a value-label overlap.

No candidate has established high-probability AlphaEx>0 and MaxDDDelta<0
across all trends. These code/integrity checks do not prove economic efficacy.
