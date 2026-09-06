# Stage15: fixed short-horizon feature-information probe

Planning research, 2026-09-06. No new real forecast, coefficient, loss, order, or model was computed. No local additional-test15–24 outcomes were read. This note records the final chosen representation hypothesis, using primary literature and existing feature source/registration. Architecture remains StandardScaler + Ridge(alpha=100), with existing public inputs.

## Final fixed comparison

Use a 2×2 presence/absence design: technical29; technical29 + price5; technical29 + flow3; technical29 + both8. The initial research suggestion was a narrower price-only probe. The final bounded design responds to the user's explicit request to explore varied technical indicators and measures whether the flow block contributes beyond the price block. It does not identify individual columns' effects, and the literature does not establish these eight columns or lookbacks as optimal.

At decision origin t on the 15-minute bar-open grid, only completed bars enter:

    price: r_k,t = log(C[t−1]) − log(C[t−1−k]), k in {4,16,48}
           body_sign_t = sign(log(C[t−1] / O[t−1]))
           candle_location_t = (2*C[t−1] − H[t−1] − L[t−1]) / (H[t−1] − L[t−1])
    flow:  flow4_t^market = sum_{j=1..4}(2*buy_quote[t−j] − quote[t−j]) /
                           sum_{j=1..4} quote[t−j], market in {Spot, UM}
           spot_volume_surprise_t = log(mean_{j=1..24} quote[t−j] /
                                        mean_{j=1..672} quote[t−j])

The price returns span 1h/4h/12h; body_sign is the last completed candle's intrabar sign, not its close-to-close sign. The registered floating-point arithmetic is log-close differencing for lag returns and sign(log(close/open)) for body sign; algebraic rewrites need not be bit-identical at rounding boundaries. The original reversal paper defines intrabar return (C−O)/O, so this distinction is intentional. Measured flat-candle and zero-volume conventions, minimum complete-window requirements, invalid OHLC/volume handling, and output column order belong to the frozen feature specification. Missing values must not silently become measured zero. Do not append negative return copies as separate reversal indicators: a negative Ridge coefficient already represents reversal, whereas duplicate columns change the penalty.

The predictive comparison uses raw h24 mean forecasts against an identically fitted technical29 raw forecast; zero and training-mean forecasts remain compulsory sanity references. The 2×2 comparisons isolate block additions and whether a block's gain depends on the other block; they do not establish causal price/flow effects. All groups retain the same 18-month fit interval, original development folds5–12, estimator/settings and rows. No new mean bias, shrinkage, rolling correction, weight selection, risk-model fit, or architecture search is part of this feature comparison. The original risk/execution contract remains fixed for any downstream economic comparison, and its results must be distinguished from forecast loss.

Data-only preflight must prove all newly claimed features finite on the inherited fit/inference supports before outcomes. If an added feature invalidates support, stop and revise the registered common support before any fit; never compare differing selected samples or cancel causal orders using future score availability. The technical refit must match the inherited source's exact training arithmetic and support. Freeze all source/data bindings before real fitting.

The target remains y_t = log(close[t+24]/open[t+1]), available under the archival convention at t+375 minutes, with inherited purged boundary rules. The research articles' immediate 15-minute or hourly targets do not replace this delayed six-hour target. Feature shifting must occur once on the complete UTC grid; closed timestamp availability is not proof of historical feed receipt or unrevised archives.

## Novelty, falsification and reporting limits

Explicit 1h/4h/12h return coordinates are absent from technical29's 1/7/30/90-day momenta, although RSI14/96, channels and z-scores already summarize short prices. Body sign and candle location add nonlinear representations, not a new market information source. Their exact delayed-six-hour value is unproved.

Six-hour UM quote imbalance24 already exists in prior derivative groups. Spot imbalance24 is recoverable from the prior UM imbalance and UM-minus-Spot gap. The proposed Spot/UM imbalance4 changes aggregation to one hour; it does not claim new raw data. Existing flow groups include bar quote-volume relative to 96/672-bar means and relative UM/Spot 24/672 activity. The new Spot mean24/mean672 ratio is a different representation. An additive unsigned volume feature in Ridge does not by itself model direction-times-volume interactions; none are silently added.

A useful falsification is no consistent raw-MSE gain from flow once price is present, or gains against learned technical predictions that still lose to zero/training mean. Equal-quarter losses, per-quarter and unchanged all/bull/bear/sideways paired differences must include failures. Rank/sign summaries and economic AlphaEx/MaxDDDelta answer different questions. A predictive loss gain alone does not imply favorable turnover, costs or drawdown. Reused eight development quarters with 2/4/2 start-regime inventory cannot prove independent high-probability or trend-invariant superiority. This experiment can reject this small fixed representation; it cannot establish absence of learnable information in all features or justify choosing architecture from later test results.

## Three primary sources and exact limits

1. Wen, Bouri, Xu and Zhao (2022), *Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both*, North American Journal of Economics and Finance62,101733. [Publisher article](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833), [DOI](https://doi.org/10.1016/j.najef.2022.101733). The primary publisher abstract and methodological snippets report earlier-to-later hourly predictability with momentum and reversal, in-sample and out-of-sample analysis, and changes with jumps, liquidity, FOMC and COVID. Displayed results retain significant in-sample/positive-OOS combinations. This motivates checking short lags while cautioning against copied selected pairs or regime-invariance claims. Full text was not retrieved; no unobserved exact timing or cost protocol is claimed.

2. *Short-horizon mean reversion in cryptocurrency markets: a matched cross-market measurement* (August2026), [original preprint v1](https://arxiv.org/html/2608.21888v1). It reports 15-minute directional reversal with near-zero lag-one magnitude autocorrelation and fading at longer horizons. Its primary return definition is intrabar (C−O)/O. Flow conditioning and price/flow redundancy motivate checking whether the flow block adds anything conditional on price, without claiming causality. Gross edge peaks around 1.3bp versus a 5bp round-trip benchmark; this establishes no capture edge. Survivor/post-sample universe selection and fit-conditional inference limit the measurement. Nothing proves our continuous six-hour target, fixed lookbacks, Ridge100 or local regimes. No paper model, threshold grid or p-value is imported.

3. Cont, Kukanov and Stoikov, *The Price Impact of Order Book Events*, [original paper](https://arxiv.org/abs/1011.6402). Its short-interval impact result concerns best-bid/ask order-book-event imbalance in equities, including liquidity supply and cancellation, and contemporaneous price changes. Binance kline taker-buy quote volume covers executed aggressive trades. It is not that LOB OFI, and the paper does not show it forecasts a following six-hour BTC return. This distinction prevents explanatory microstructure evidence being presented as out-of-sample directional evidence for the proposed features.

Search hygiene: the Essex PDF named SSRN-id4119562 is a behavioural-finance review, not the Wen original, and is not relied upon. No generic indicator catalogue or universal robust-signal claim is used.

## Local source bindings

- unidream/experiments/oracle_frontier_features.py: `762e7611e8b88883ef5a319dc69484523d4b58805409a261c8bb9df6d90662e5`
- unidream/experiments/alpha_dd_features.py: `7f95da6ea7704d036e4be254fa620f650cd0168639fc29e0e14df68620bc6e69`
- unidream/experiments/oracle_derivative_features.py: `c2bc460f476bc2019e0a035fba08f29e9a616f182e767a41273098c5df472448`
- docs/experiments/oracle_derivative_ablation_registration_20260905.md: `7efe2dbce3d4bd23c9c85e0649f42a156a5edf90966d29eb4a9bcb6292264e70`
