# Fixed family additional-window report registration

This protocol is committed before acquiring the extended UM archives and before
computing any new model or policy outcomes. This is a **report-only evaluation
on reused historical periods**, not independent confirmation. It preserves the
existing validation selector and every previous selection lock.

## Fixed scope

- Original **test folds 15 through 24**, all ten consecutive quarters, from
  **2024-01-16 13:45 UTC** to **2026-07-16 13:45 UTC**, right exclusive.
  These literal test IDs must not be relabeled as the future confirmation folds.
- The earlier research explicitly used test15–23 and test24. The historical
  `fresh` label for test24 does not make it unread or statistically independent.
  See the bound [access audit](oracle_mean_shrinkage_evidence_20260905/confirmation_access_audit.md).
- Replay the complete [frozen family](oracle_mean_shrinkage_evidence_20260905/candidate_family_freeze.json),
  SHA256 `245cc711aabe395e2fcc93b45695cafeac03550e435cdc4f2f5b2a6a13f12cd0`.
  Four candidates: technical half and perpetual-flow half, each with hold and
  target-one fallback. Eight controls: B&H, common robust overlay, scale-period
  mean, technical full mean, perpetual-flow full mean, with both missing-input
  rules for each of the three mean controls. No additional policy is searched.
- The half weight is exactly 0.5, applied to the scale-period historical mean
  and the candidate's own calibrated mean. No weight, model, feature, regime,
  threshold, risk coefficient, cost, or missing-input rule is retuned here.
- Each evaluation start S uses fit [S−24 months,S−6 months), scale
  [S−6 months,S−3 months), interval calibration [S−3 months,S), then evaluation
  [S,S+3 months). Minimum usable fit/scale/interval rows are 512/64/64.
- Use the committed `oracle_frozen_forecasts.py` procedure that exactly
  reproduced the earlier eight development quarters. The two mean models are
  standardized Ridge(alpha=100). Risk is the same technical-feature HGB model
  and calibration in every candidate. Model parameters and seed remain frozen.
- Preserve the complete inherited common feature-support dependency set,
  including trailing variances, original flow, all derivative feature groups,
  and extra UM delays 0/1/4, even though not all columns enter each model.
  Features use only bars through t−1. No interpolation, zero-fill, support
  compression, or future-label gating of inference/orders is allowed.
- Decisions use the UTC six-hour clock; fills use the next 15-minute bar open.
  The 24-bar target matures after 25 bars. Fit/calibration maturity is strictly
  before each segment end; scoring maturity is at or before evaluation end.
- Preserve all own-inventory utility decisions, initialization, execution,
  action limits, fees and borrowing from the frozen frontier configuration.
  Cost stress doubles fees and borrowing on the same base decision intents.
  Target-one fallback is only a scheduled decision with known current open and
  unavailable forecast. Missing next open skips a fill without rolling it over.
- Label the quarter's regime using only the first scheduled decision's past
  90-day momentum and past volatility, with the existing ±0.5 threshold.
  Report actual B&H quarterly returns separately; do not relabel regimes using
  those future returns.

## Data acquisition and quality gate

The unchanged strict acquisition implementation
`unidream/experiments/oracle_derivative_acquisition.py` will use
`configs/oracle_additional_window_acquisition_20260906.yaml` in a new immutable
namespace, `checkpoints/oracle_additional_window_data`. It requests the fixed
55 official monthly UM BTCUSDT 15-minute archives from January 2022 through
July 2026. January provides more than the required seven-day UM rolling warmup
before the first fit starts on January 16, 2022. Spot features retain their full
earlier history from the existing source.

The acquisition retains raw ZIP files, official SHA256 checksum identities,
parsed monthly frames, monthly provenance and an append-only ledger. A source,
checksum or parser error fails closed. A 404 remains an explicitly missing
month. The complete UTC bar-open grid retains NaN observations and false masks.
UM timestamps remain milliseconds; Spot post-2024 microseconds were already
handled by the existing separately bound Spot acquisition.

Spot input is the existing `spot_15m.parquet`, SHA256
`5e20e81e86f76b95d1301be7a8a366aa9ad78134f954ec8c9dbf83c0db1acf69`,
under the `alpha-dd-goal` worktree. Its metadata report observed archives through
July 2026, with missing rows retained. Validate its data, availability and ledger
digests before use. No earlier data artifact is overwritten.

The UM raw monthly tail after the fixed cutoff is retained but excluded by the
existing strict `decision_ts < cutoff` feature validator. The old acquisition
sidecar's generic string mentions an April tail; for this new configuration it
means the July tail, as established by its numeric cutoff and monthly fields.
Archive event timestamps and download records do not prove historical live
receipt, tradability at a deadline, or prospective operation.

Source-format references checked on September 6, 2026:
[Binance public-data documentation](https://github.com/binance/binance-public-data)
documents Spot microseconds from January 2025, monthly archive checksums, and
possible archive revisions. [USD-M general information](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info)
specifies millisecond timestamps for the UM API underlying those archives.
Do not apply the Spot archive unit change to the UM input.

Before any fits, save a data-only preflight with all ten calendars, common masks,
counts, coverage, last training/calibration label maturity, and source hashes.
Require each full evaluation bar grid, at least 99.5% observed bars, and at least
16 scoring decisions. A failed gate blocks the complete family; do not silently
drop a quarter or substitute a better-covered period. Commit the final adapter,
tests, exact config and all data/source/metadata bindings before outcomes.

## Reporting and interpretation

Save all 120 policy-quarter rows, 50 mean forecast scores, models, calibration,
forecasts, targets and utility traces with hash-bound fold manifests. Reports
must retain every quarter and each fixed policy regardless of performance.
Show equal-quarter AlphaEx/MaxDDDelta under base and doubled costs, forecast
MSE/MAE/rank IC, paired changes against the scale mean and each own full mean,
quarterly outcomes, and results by the three start-of-quarter regimes.

The descriptive economic target is AlphaEx > 0 and MaxDDDelta < 0 in aggregate
and in each regime under both cost settings. Regime coverage requires at least
three quarters in each regime; that count alone does not establish statistical
confidence. Forecast improvement should be reported against both simple and
own-full references, preserving the signs of failed comparisons. Report losses
using ratios of equal-quarter mean losses, not averages of fold percentages.

No p-value, probability guarantee, production readiness, selection, promotion,
performance-based early stopping or automatic family winner is produced.
These periods may reject the development signal, but cannot independently
confirm it. The separate future confirmation contract remains unchanged.

Run data acquisition after this protocol/config commit:

```sh
uv run python -m unidream.experiments.oracle_derivative_acquisition --config configs/oracle_additional_window_acquisition_20260906.yaml
```
