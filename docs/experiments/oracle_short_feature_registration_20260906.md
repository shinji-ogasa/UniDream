# Short-price/flow representation ablation registration

Commit and push this protocol, source/tests, config and data-only preflight
before any new real fit, coefficient, forecast, loss or order. Stage14's rolling
centering is complete and rejected. This study changes a fixed input
representation, without another return-calibration or model-architecture search.

## Fixed question and 2x2 design

Do shorter price and taker-flow coordinates add out-of-period mean-return
information to the original technical29 under the same standardized Ridge100?
Keep four fixed feature groups: technical29, technical+price5 (34),
technical+flow3 (32), technical+price5+flow3 (37). The new37 is not the earlier
derivative37 schema. Train all groups, report all differences, select none.
Price−technical and flow−technical measure each block addition; both−price and
both−flow measure the conditional block additions. Also report both−technical.
These comparisons do not identify individual column effects or an economically
optimal feature set. Adding correlated coordinates can change Ridge's effective
regularization even when raw-market information does not increase.

Original validation5–12 aliases original test4–11: April2021–April2023,
strict cutoff `2023-04-16T13:45:00Z`. Original test(f)=validation(f+1).
These are repeatedly reused development quarters, not independent confirmation.
No additional-test15–24 modeling, labels, scores or selection. The inherited
Spot loader decodes the original parquet before slicing, so semantic exclusion
does not mean later file bytes were never read. No new market acquisition.

The new mean IDs are `technical_short_price_raw`, `technical_short_flow_raw`,
`technical_short_both_raw`, each with the existing `utility_risk1` and
`utility_risk1_fallback_bh` suffixes:6new causal names, inventory168→174.
Retain all22 Stage14 controls and reintroduce the ALREADY TESTED zero,
fit_mean and technical_raw under both missing rules:28controls, not6 new names.
Those six controls use the original saved technical_scaled variance and are
bound to the earlier fallback study. In total34policies×8=272economic rows,
544base/stress accounts. Prior family locks and formal P1 records stay intact.

## Exact new features

Build every new field on the complete raw15minute Spot bar-open grid, align
sparse UM to that grid as missing, then shift the whole new block exactly1bar.
Thus a decision at t uses bars through t−15min. The inherited technical29 has
already been shifted and must not be shifted again. Validate raw timing metadata
as open, open+15min−1ms close stamp, and open+15min decision stamp when supplied.
Archive event-time availability does not prove historical receipt latency.

Price5, in this exact order:

1. `spot_log_return4`: `log(C).diff(4)` before shift, a1hour endpoint difference.
2. `spot_log_return16`: the same over16bars/4hours.
3. `spot_log_return48`: the same over48bars/12hours.
4. `spot_body_sign1`: `sign(log(C/O))` of the last completed candle.
5. `spot_close_location1`: `(2*C-H-L)/(H-L)` of that candle.

Each k-bar return requires all k+1 positive finite closes, including both
endpoints. Do not bridge missing close rows. Body sign and location require
positive finite OHLC satisfying low≤open,close≤high. A measured flat candle
has O=C=H=L and gets sign0/location0. Missing/inconsistent candles do not get
neutral values. Do not add a negative copy of return as a separate reversal
feature. The body sign is an intrabar representation, not a trading instruction.

Flow3, in this exact order:

1. `spot_weighted_flow4`: Spot `sum(2*buy_quote-quote)/sum(quote)` over4bars.
2. `perp_weighted_flow4`: the same UM quantity over4bars, with no extra delay.
3. `spot_quote_activity24_672`: `log(mean(quote,24)/mean(quote,672))` on Spot.

Require quote>0 finite and0≤buy_quote≤quote for each admitted flow bar.
Balanced measured flow is valid zero; zero quote volume is unavailable.
Four-bar flow requires4/4 valid pairs. Quote activity uses independently valid
positive quote volume, with24/24 and669/672 observed rows and the entire
nominal672-bar grid history. It is not conditioned on taker-flow validity.
No interpolation, backfill, clock compression, floor or missing-value filling.

Novelty is limited to these coordinates/windows. Technical29 already contains
RSI14/ATR14 (3.5hours), daily/weekly price and flow summaries. UM flow24 is
already in the31-column parent; Spot flow24 is algebraically recoverable from
the old derivative37. Old flow groups also tested bar-level volume intensity
relative to96/672bar means. Do not rename those quantities as new information.
The new flow4 is1hour, and the quote-activity ratio is a ratio of trailing
means; neither is limit-order-book event imbalance or depth.

## Unchanged masks, model and chronology

For evaluation E, fit [E−24m,E−6m), scale S=[E−6m,E−3m), interval
I=[E−3m,E), evaluation [E,E+3m). Reconstruct every original dependency mask
with the immutable parity prepare function, including all original feature
groups, trailing risk baselines and UM delays. Keep fit, S, I, predict,
inference and score masks exactly. New columns must be finite on EVERY one of
those six masks before fitting. If not, fail preflight instead of intersecting
masks or dropping rows. Data-only feasibility shows zero row loss.

Fit counts folds5–12 are800/1034/1313/1500/1503/1634/1672/1794.
Keep UTC6hour decisions, h24 `log(close[t+24]/open[t+1])`, maturity375minutes;
fit/S/I labels mature strictly before their segment end, E score labels no
later than its end. Inference2586 and E score2574 remain different; preserve
332fallback opportunities and2missing-current-open decisions. Current-label
availability cannot determine current predictions or orders.

Exactly32 Ridge fits in a complete execution:8 technical29 reproduction fits
plus24 new representation fits. StandardScaler and Ridge(alpha100) use the
same defaults as the source, threadpool2, fit rows only. The pure mean fitter
reads only fit return column0, never non-fit outcomes or adverse/RMS columns.
No HGB/risk fit, mean-bias correction, reliability fit, shrinkage, rolling
update, hyperparameter/window search or interval-width claim. Risk remains the
immutable S-calibrated technical_scaled forecast. Therefore this is an
uncalibrated return comparison with a frozen calibrated risk forecast, not an
entirely uncalibrated system.

Use numpy2.2.6/pandas2.3.3/sklearn1.8.0 as pinned in the config. Before accepting
addition results, compare technical29 columns/values, fit masks/returns,
StandardScaler statistics, Ridge parameters and all saved S/I/E raw predictions
with old sources. Numerical forecast/model-state tolerance is rtol1e−12,
atol1e−14 with identical NaN masks; technical_raw hold/fallback targets must
match exactly, and account metrics use the inherited comparison tolerance.
Old and new serialized joblib bytes need not match; numerical fitted state
does. No warm start or partial-period fitting.

## Predictive and economic comparisons

Score interval only for zero, fit_mean, technical_raw and3 new means:6×8=48
records. Score E for all12 old Stage14 means, zero/fit_mean and3 new means:
17×8=136records, total184. S is not used to fit a new return calibration or
pick a feature set. I is diagnostic-only; it cannot choose a variant before E.
I and E/calibration windows overlap across folds and are not independent
replicated confirmations. I grouped by the later E-start regime is explicitly
retrospective; E regime is known at its first scheduled decision.

Save per fold and all/bull/bear/sideways equal-quarter MSE, MAE, rank IC,
zero/fit-mean losses and differences. Label pooled-row MSE separately. Save
the five paired contrasts above, their relative MSE changes and favorable/tied
quarter counts. Undefined rank quarters cannot be silently dropped. Strict
predictive direction requires lower MSE than zero, fit_mean and EVERY registered
reference in EVERY stratum, separately for I and E. This descriptive flag
neither selects nor promotes a policy. Keep all adverse differences.

Every new policy uses its own cash/units, original B&H initial inventory,
conditional utility risk1/cost allowance2, intents[.5,1.12], maxstep.08,
deadband.01, one-way fee.00055, annual borrow.10. Missing forecast uses hold
or the registered target1 fallback with known current open. Missing current
open cannot order; missing immediately-next open skips without rollover.
Borrow across gaps, keep unscored inference actions, and replay unchanged
base intents at doubled costs. Report AlphaEx>0/MaxDDDelta<0 signs separately
from predictive direction, all-strata means and strict per-quarter joint counts.
Machine-roundoff negative DD differences are not substantive improvements;
retain registered counts and disclose any such cases rather than retune a
tolerance after seeing outcomes.

## Source/artifact contract and checks

Verify Stage14's completed registration/results/preflight and8fold manifests
in addition to its data-only prepare, which returns Stage13 inputs. Merge
1536 ancestors+256 Stage14 artifacts=1792 distinct artifacts, reject conflicts
and path aliases. Bind30 source modules, both preparation configs and the old
fallback registration/results. Read model/calibration/rawforecast provenance
from these verified artifacts. No Oracle output is used as a causal teacher.

Per fold save4models,4 raw calibrationNPZ (S/I plus preserved prediction tails),
4 raw evaluationNPZ,1 fit-provenanceJSON,34targetNPZ,6new utility-traceJSON:
53×8=424 artifacts, plus8fold manifests and registration/preflight/results.
Fit provenance records exact column order, selected-feature/return hashes,
mask ranges/counts and baseline numerical parity. Copy/reference all272
economic rows and184scores. Existing28controls and96old E scores must reproduce.

An interrupted partial attempt is retried by deterministic full replay of all
folds and exact checks of any already written arrays, models and JSON. It does
not skip completed folds or resume a fitted estimator. Such a retry may repeat
physical fits;32 is the logical family and the fit count of one complete
attempt, not a lifetime compute count across retries. Completed results reject
re-execution. Never restart a process because observation times out; verify the
same live handle first. Record any interrupted attempts in the results report.

Before new real fits run `uv run python -m unittest discover -s tests -v` and
`git diff --check`. Synthetic tests cover exact feature arithmetic, shifts,
prefix invariance, short-window gaps, flat/missing distinction, timing,
selected-only training, future-label poison, fixed Ridge parity and strict
complete-family summaries. Independently audit feature values/support,
fitted coefficients/predictions, new own-state paths, all544 cost accounts,
184scores and all summary comparisons after completion. No paid infrastructure,
live trading, production deployment, model promotion or external messaging.

## Primary rationale and remaining conditions

[Wen et al.2022](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080253)
reports intraday Bitcoin momentum/reversal patterns that vary with conditions;
it does not fix these windows or prove invariance. The recent
[short-horizon reversal preprint](https://arxiv.org/html/2608.21888v1) separates
sign behavior from mean-return predictability and finds limited cost capture;
its shorter-horizon evidence does not establish our delayed6hour mean target.
It also motivates testing whether flow adds beyond price rather than assuming
independence. [Cont et al.](https://arxiv.org/abs/1011.6402) concerns order-book
event imbalance; our bar taker-quote proxies cannot inherit that interpretation.
See the accompanying research note for exact scope and limitations.

The chosen candle location and activity windows are bounded engineering
hypotheses, not individually validated six-hour predictors from these papers.
No cited study proves these eight features improve our objective. Reused
development quarters, overlapping histories, adaptive exploration,2bull/4bear/
2sideways coverage and absent receipt evidence persist. No independent or
high-probability all-trend result, formal P1 success or strongest-model claim
follows from passing implementation checks or favorable aggregate means.
