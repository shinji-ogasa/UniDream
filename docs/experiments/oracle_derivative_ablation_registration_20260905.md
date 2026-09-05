# Spot / perpetual feature ablation registration

This family is fixed before its model outcomes. Earlier 81 policy names and
their failures have been observed. This is adaptive research on reused
development validation, with no test/outer selection or deployment. Historical
archive availability does not establish contemporaneous live arrival.

## Question and fixed feature groups

Does past perpetual flow add useful six-hour return or risk forecasts beyond
the existing technical features? Keep four groups:

- `base16`: conventional existing control, 16 inputs.
- `technical`: the existing parent, 29 inputs.
- `perp_flow`: technical plus UM weighted flow over 24 and 96 bars, 31 inputs.
- `derivative`: technical plus all eight preflight columns, 37 inputs.

The primary incremental information comparison is perp_flow versus technical.
Derivative versus technical and versus perp_flow are secondary representation
comparisons. In the eight-column version, perp_flow96, its gap to Spot, and the
existing Spot flow96 are linearly redundant; standardized Ridge can change its
effective penalty. The gap24 and perp_flow24 also reveal Spot flow24, which was
not present in technical29. Improvement in all8 alone is therefore not isolated
evidence of new UM information. Preserve these fixed features; do not revise
their representation after inspecting performance.

Use the exact preflight equations and missingness rules. Compute on raw UTC
bar-open indices, shift once, and use completed bars through t-1. Twenty-four
and 96-bar flow windows require complete paired taker/quote observations within
each market. The 672-bar quote means independently require 669 observations
in each market, as in the preflight. Do not insert missing-volume zeros.
The traded-close premium is not mark-price basis, funding, or open interest.

## Chronology and support

The checksummed data-only preflight selects development folds 5 through 12 by
the fixed minimum row counts, not by performance. Each fold has 18 months for
fitting, 3 months for return bias / variance scale, 3 months for interval width,
then its original 3-month validation. Require 512 / 64 / 64 rows and label-end
timestamps strictly before each fitting or calibration boundary. A forecast
at t fills at t+1 and its h24 label is available at t+25 bars.

All groups, references and policies share the intersection of the four groups,
the original flow24 frame and trailing variances over 24/96/672/2880 bars.
Require the exact preflight 6h mask hashes and h24 segment counts before any
fit. Validation scoring additionally requires complete future Spot support
and label end no later than the quarter end. Future UM support never enters
training labels or scores. Future outcome availability never suppresses a
causal forecast or submitted order.

There are two bull, four bear and two sideways quarters. The existing minimum
of three quarters per regime already fails and must remain false even if all
observed mean signs improve. This stage can show an improvement direction;
it cannot establish the requested high-probability trend robustness.

## Fixed estimators, calibration and policies

Fit a separate StandardScaler + Ridge(alpha=100) return model for each group;
the scaler sees only purged fit rows. Also fit each group's log realized
variance with the existing HGB: 100 iterations, 7 leaves, minimum leaf 64,
learning rate .04, L2 10, early stopping disabled, fixed seed 20260905.
No architecture or hyperparameter search is performed.

As in the prior calibration experiment, exponentiate log variance with the
fixed numerical range [1e-12, 1], then preserve raw and scaled versions. The
scaled version adds the mean return residual from its first calibration
segment and multiplies variance by mean(actual variance / raw forecast).
This changes both mean and variance; it is not a variance-only intervention.
Fit the corrected 90% absolute normalized return and log-volatility quantiles
only on the second calibration segment. Report interval coverage, not a
guarantee under temporal dependence.

Each of four groups and two versions feeds the unchanged point mapper and
the inventory-aware conditional utility controller with risk weights 0 and 1,
cost allowance multiplier 2. This is 24 policy names, plus B&H and the shared
mask robust control. Risk 0 ignores variance. The utility is an approximate
local mean-variance objective, not an exact log-wealth or MaxDD optimizer.
Keep the previous cash/units contract, next-bar execution, max step .08,
deadband .01, one-way cost .00055, annual borrowing .10, and initial B&H
inventory. Replay identical base-selected intents under doubled cost/borrow.

Expected output: 64 fitted models, 64 raw/scaled forecast records, 32 saved
calibration records and 208 economic rows / target paths across eight folds.
Preserve input/model/source/forecast/target hashes, raw log predictions,
clipping counts, calibration constants, masks and all failed comparisons.

## Comparisons and uncertainty

Return point loss uses MSE against zero and the common purged-fit return mean,
with rank IC and sign accuracy as descriptive companions. Risk loss uses
QLIKE, variance MSE and RMS MSE against raw 96-bar persistence. Compare the
same group/version on identical score rows. Primary summaries average paired
quarter-level losses and economic differences with equal quarter weights;
pooled-row losses remain separately labeled. Do not select a model from these
diagnostics or replace any earlier selection lock.

Descriptive paired-loss uncertainty is conditional on these eight observed
quarters and their fitted forecasts. Preserve the full 6h calendar grid,
including NaN loss slots. Resample non-circular contiguous moving blocks separately
within each quarter, using identical indices for candidate and reference.
Average each sampled quarter's finite paired losses, then average quarters
equally. Use 2,000 replicates, seed 20260905, primary block length 28 slots
(7 days) and fixed sensitivities 4 and 112 slots (1 and 28 days). Report all
lengths, including disagreement. Center bootstrap errors at the bootstrap
mean and report the resulting basic 95% descriptive interval, plus the
bootstrap-mean minus observed-mean shift from unequal endpoint weighting.
At 112 slots, each quarter supplies only about three long blocks; regard that
sensitivity as unstable. A replicate is valid only if every quarter supplies
at least one paired finite loss. Report invalid replicate counts, without
retrying or dropping a quarter; an interval with fewer than two valid
replicates is unavailable. Differences below zero favor the candidate.
The diagnostic does not include cross-quarter regime uncertainty, overlapping
training sets, model refitting, adaptive search, or prospective performance.
It provides no selection-adjusted p-value or formal pass.

The current scope remains forecast improvement, economic decision value and
their limitations. All-trend success and high-probability generalization
remain unestablished until independent evidence supports the full conditions.

The comparison rationale uses paired losses as in
[Diebold–Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599),
contiguous-block resampling for dependent data as in
[Künsch (1989)](https://doi.org/10.1214/aos/1176347265), and the distinction
between average and conditional predictive ability in
[Giacomini–White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x).
The helper implements none of those papers' hypothesis tests or guarantees.
[Hansen (2005)](https://doi.org/10.1198/073500105000000063) motivates explicitly
retaining the adaptive-search limitation; this is not an SPA test. Block
lengths are fixed practical sensitivities, not source-derived optimal lengths.
