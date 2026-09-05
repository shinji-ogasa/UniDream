# Joint AlphaEx / MaxDD experiment registration

User objective: mean quarterly final-total-return AlphaEx at least +3 percentage
points (minimum +1), AND mean absolute MaxDD delta at most -3 points (minimum -1).
The same strategy, windows, prices, and costs must meet both. This new experiment
does not amend or promote any P1 result. Goal completion also requires a deployed
HF model and verified live inference after research qualification.

## Before-results design

All candidates and period/cost choices are fixed in `alpha_dd_search.py` and
`configs/alpha_dd_research_20260905.yaml` before the first experiment run. The runner
persists their hashes and refuses a modified registration or an overwritten result.
The candidate universe includes B&H, trend following, volatility scaling, smooth
drawdown protection, combined trend/volatility, ridge, gradient boosting, and
logistic forecasts. These are alternatives to the unsuccessful WM/BC/AC diagnostic.

Development: 13 quarterly windows, folds 0–12. Rank by
`min(mean AlphaEx, -mean MaxDDDelta)`, then AlphaEx, DD delta, and candidate ID.
Lock the single winner and development file hash before confirmation. Historical
confirmation: original quarterly folds 15–23. These periods have already appeared
in past experiments and are not globally untouched. Fresh confirmation: fold 24,
2026-04-16 13:45 to 2026-07-16 13:45 UTC; the first 10h15m overlaps the old cache,
so this is a mostly new quarterly replication, not a wholly unseen window.
It cannot be shortened when source data is missing. No confirmation result changes
the selected candidate or its hyperparameters. The descriptive fold bootstrap does
not correct for candidate selection or temporal dependence. One fresh quarter is
weak evidence and is not a multi-regime statistical certificate.

ML forecasts fit preceding two-year training rows with full future-horizon coverage
and purged target ends before the 3-month validation window. Fixed models are then
used through the following test quarter. No test labels fit a scaler or model.
Every fit saves training cutoff, last target timestamp, and serialized model SHA.
At least 256 complete, purged, 16-bar-subsampled training rows are required. A
candidate with any unavailable fold is ineligible for selection; unavailable folds
are not silently removed from its denominator and are not replaced with B&H.

## Price and execution contract

Use official checksum-verified Spot BTCUSDT 15m OHLCV archives with explicit missing
bars and archive revisions. Download timestamps are not historical live availability.
Feature row t uses only completed bars through t-1. Decisions are scheduled at UTC
hour boundaries, filled at the next bar's open, then marked at observed closes.
Missing fill bars skip the order; missing outcome bars retain inventory and cash.
The next observed price marks the full intervening price move. Missing history
prevents affected features/forecasts, rather than being filled with numerical zeros.
No result conditions an earlier decision on a future outcome mask.

Both arms start each fold with equity 1 and existing B&H exposure 1 at the opening
price. No artificial first-entry fee or terminal liquidation is added. The account
tracks cash and asset units, includes position drift between trades, and solves
post-fee target sizing. Targets are 0–1.12, maximum adjustment .08 per decision,
no-trade band .01. One-way turnover cost is 5.5 bps (= 3 bps full spread / 2 +
1 bp slippage + 3 bps fee). Borrowed cash is charged an assumed 10% annually,
accrued each 15m bar; 20% plus twice the trading cost is a fixed sensitivity.
This rate is a research assumption, not a claim about exchange margin rates.

Equity uses exact self-financing units/cash accounting, not additive log PnL.
AlphaEx is strategy final equity minus B&H final equity. MaxDD includes initial
equity 1; delta is strategy absolute MaxDD minus B&H absolute MaxDD. Unlike some
legacy paths this does not miss a first-bar drawdown or treat continuous rebalancing
as free. Evaluate paired observed closes; document missing-bar coverage (>=99.5%).
This mark-to-close DD does not measure unseen intrabar liquidation or outage lows.

## Research motivation

- [Moskowitz, Ooi & Pedersen: Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum): conventional-market evidence motivates testing causal trend signals. It does not establish BTC efficacy or these shorter lookbacks.
- [Moreira & Muir: Volatility-Managed Portfolios](https://www.nber.org/papers/w22208): conventional-market volatility scaling motivates the risk family; BTC transfer is an experiment.

## Promotion requirements

Apply the user's two mean thresholds to all ten registered confirmation quarters
together (historical folds 15–23 plus fresh fold 24), using the same locked policy.
Also disclose the historical and fresh subsets separately, including a negative
fresh quarter if observed. The user requested mean thresholds, not a requirement
that every individual quarter passes. Require complete coverage of all registered
quarters, explicit cost sensitivity, data/provenance checks and output parity.
Then export a model fitted using data
available at its new fit date, retain training/evaluation provenance, update HF,
and verify health, fixture parity, real candles inference, and live-source behavior.
Failure of either economic target leaves the user goal active. Passing code tests
or one in-sample teacher diagnostic is not a substitute for these checks.
