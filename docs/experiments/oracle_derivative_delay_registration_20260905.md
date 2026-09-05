# UM information-staleness sensitivity registration

This fixed diagnostic precedes its model outputs. It follows 109 adaptively
explored policy names on reused development validation. It does not select a
winner, change an earlier lock, use report-only test/outer for tuning, or prove
future performance. The goal still requires stronger trend robustness.

## Question, fixed inputs and interventions

The parent experiment's two UM weighted-flow columns slightly reduced return
MSE relative to technical29, while increasing risk QLIKE. Its scaled utility
policy passed all observed regime-mean signs, but the regime counts were only
bull2/bear4/sideways2, its return MSE remained worse than zero prediction, and
its economic uplift relative to technical concentrated in fold12.
The frozen mean/variance crossing located that economic difference mainly in
the mean input and resulting inventory path. Keep architectures fixed and test
whether older UM information changes this finding.

At decision t, the existing raw-open features already shift by one 15-minute
bar. Add exactly 0, 1 or 4 full-grid shifts to ONLY `perp_weighted_flow24` and
`perp_weighted_flow96`. The latest included UM bar opens at t−15, t−30 or
t−75 minutes, and ends by t, t−15 or t−60 minutes, respectively. Spot's 29
columns remain current under their original completed-bar contract. Neither
decision times, future labels, next-bar fills nor saved predictions are shifted.
No filling or compression of missing rows is allowed.

| Forecast family | Training/calibration UM age | Validation UM age | Interpretation |
| --- | --- | --- | --- |
| technical | no UM input | no UM input | matched Spot control |
| perp_delay0 | no extra delay | no extra delay | newly refit common baseline |
| perp_delay1 | extra 15min | extra 15min | learning with stale information |
| perp_delay4 | extra 60min | extra 60min | learning with stale information |
| frozen_delay1 | delay0 artifacts retained | extra 15min | fixed mapping under input staleness |
| frozen_delay4 | delay0 artifacts retained | extra 60min | fixed mapping under input staleness |

The frozen variants use the NEW common-mask delay0 models, StandardScaler,
Ridge coefficients, HGB trees, mean bias, variance multiplier and interval
quantiles unchanged. They do not fit or recalibrate on stale calibration
inputs. Delayed refits instead use their delayed columns throughout the
fit/scale/interval/validation chronology. Do not reuse broader-mask parent
models as these primary controls. Record training and inference delays plus
model and calibration hashes in each forecast's provenance.

## Shared support and timing limits

Keep the parent cutoff 2023-04-16T13:45Z and its original full feature mask,
including trailing variance and all8 derivative availability. Intersect it
with finite technical/delay0/delay1/delay4 rows on the complete Spot15m grid.
Recompute all controls on this same support. Data-only preflight must fix
the exact masks, fit/calibration/scoring counts and source/data hashes before
model outputs. Pin its SHA in the config; execution must reconstruct it
exactly before fitting. Original UM and Spot artifact proofs remain required.

The mask intersection knows the undelayed availability pattern. This makes
the experiment a retrospective comparison on common complete support, **not
an operational delayed-feed or outage policy**. A delayed feed may not yet
know whether a newer UM bar is missing. No live-causal readiness claim may
follow from this comparison. Future target availability is excluded from
inference masks; it is used only for fit/calibration and forecast scoring.

Development folds5–12 retain their original validation windows, 2021-04-16
13:45 UTC through2023-04-16 13:45 UTC exclusive. Each uses18months fit,
3months scale/meanbias correction,3months interval calibration, then3months
validation. Label horizon24, future return starts at next-bar open and ends
at the 24th marked close. Label information must end strictly before every
fit/calibration boundary. Validation scoring may include labels ending
exactly at the exclusive validation boundary, as in the parent contract.
Keep minimum512/64/64 fit/scale/interval rows and the original coverage gates.
Six-hour UTC decisions, original missing intervals and next-open execution
remain unchanged. Do not add/drop folds based on performance.

## Fixed models, decisions and inventory

Each of four trained feature families fits StandardScaler+Ridge(alpha100)
for the mean and HGB for logvariance:100iterations,7leaves,minimum64 samples
per leaf,learningrate.04,L2=10,no early stopping,seed20260905. All normalizers
use fit rows only. Keep raw and scaled variants; scaled includes BOTH return
bias and variance multiplier from the scale segment. The90% interval
quantiles are diagnostic only; neither policy gates on them.

Each of six forecast families × raw/scaled is applied to exactly two existing
decisions: the point mapper and conditional utility with risk aversion1 and
cost multiplier2. No risk0 duplicates, new risk weights or new architecture.
Keep canonical cash/units accounting, initial B&H inventory, exposure bounds
.5–1.12, maxstep.08,deadband.01,one-wayfee.00055,annualborrow.10. Plan once
under base costs and replay the same submitted targets with both fees and
borrowing doubled; never replan based on stress outcomes.

Inventory:64 fitted models,32 fitted calibration records,96 forecast versions,
24 learned policy names plus B&H and common_robust,208 economic rows/targets,
and96 utility traces. Frozen variants reuse delay0 artifacts. Every fold saves
artifact hashes and resumes only if all saved artifacts and registration match.
An interrupted, incomplete fold has no reusable result; under the identical
registration it may be rebuilt from the same fixed seeds, replacing its partial
files. A completed fold's artifacts must match before any reuse. An observation
timeout alone never authorizes a restart of a still-running process.

## Complete fixed comparisons and reporting

For both raw/scaled, compare these nine pairs (candidate minus reference):
delay0/1/4 versus technical; delay1/4 refits versus delay0; frozen1/4 versus
delay0; delay1/4 refits versus frozen1/4 at the same input age. Report all18
paired comparisons, both decisions, both costs and all/bull/bear/sideways.
Forecast metrics include return MSE, QLIKE, variance MSE and RMS MSE; return
references remain zero and fit mean, variance reference causal96-bar
persistence. Relative loss improvement is the ratio of equal-quarter mean
losses, not a mean of percentage improvements. Preserve pooled-row summaries
and all quarter losses separately. Save interval coverage as diagnostics.

Economic summaries are equal-quarter point estimates only. No new confidence
interval, DM/SPA test or selection-adjusted inference is performed. Keep all
failed cases. Do not choose a delay or dropfold12 after outcomes. Both-cost
joint signs and unchanged minimum3quarters/regime remain descriptive gates;
the2/4/2 counts cannot pass the coverage condition. No strongest model or
high-probability-generalization claim is available from this family.

## Primary-source interpretation

[Binance's kline WebSocket schema](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#klinecandlestick-streams-for-utc)
distinguishes event time, candle start/close and closed status; it does not
record our historical receipt time. This schema reference is for Spot.
[The official public-data repository](https://github.com/binance/binance-public-data#data-information)
documents USD-M kline/taker fields, subsequent archive publication and possible
corrections. A checksum and close timestamp cannot establish arrival/version
availability at the original decision boundary. Delay0 assumes prompt receipt
of the just-ended candle; these extra delays are information-staleness
scenarios, not measured historical network latency.

[Lillo–Farmer's original paper](https://arxiv.org/abs/cond-mat/0311053)
finds persistent order signs in London equities with liquidity/size adjustment
offsetting return predictability. [Silantyev's BitMEX study](https://link.springer.com/article/10.1007/s42521-019-00007-w)
supports a contemporaneous flow/price relationship. Neither establishes
forward6h BTC predictability from these two Binance features. A60min delay
still shares20/24 and92/96 bars with the original windows. Stability alone
does not demonstrate persistent new information or economic causation.

## Data-only preflight fixed before fitting

The [saved preflight](oracle_derivative_delay_evidence_20260905/preflight.json)
has SHA-256 `d20d0a23096a6693dc7125e0f5fd9e761efc7e5f7bfdb6ca42c5b890c91252c4`.
It contains no fitted model or economic result. The exact source/config/data
bindings and full calendar mask hashes must be reconstructed by execution.

| Fold | Fit | Scale | Interval | Inference | Scored |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 800 | 233 | 279 | 223 | 221 |
| 6 | 1034 | 279 | 221 | 218 | 215 |
| 7 | 1313 | 221 | 215 | 363 | 361 |
| 8 | 1500 | 215 | 361 | 360 | 359 |
| 9 | 1503 | 361 | 359 | 355 | 354 |
| 10 | 1634 | 359 | 354 | 368 | 367 |
| 11 | 1672 | 354 | 367 | 368 | 367 |
| 12 | 1794 | 367 | 367 | 331 | 330 |

Total inference2586 and scored2574 of2920 scheduled6h decisions. Relative to
the parent experiment, the joint validation support loses one row in fold9;
some later fitting/calibration segments also lose that row. Regime counts
remain2bull/4bear/2sideways. All fit/calibration row gates pass, while the
minimum three quarters per regime still fails before any performance output.
