# Oracle / feature frontier v1 registration

Registered before this experiment's numerical outputs. All periods have prior
research exposure. This is exploratory development validation, not an untouched
test, a formal P1 amendment, or a significance claim. Prior locks stay frozen.

## Fixed question and family

Can observable technical/flow features improve future return, adverse excursion,
and realized-volatility prediction enough to improve both AlphaEx and MaxDDDelta
relative to B&H, before changing neural architectures?

- BTCUSDT official checksummed Spot 15m archive; features use bars <= t-1.
- Actual validation quarters for fold IDs 0--12: 2020-01-16 13:45 UTC through
  2023-04-16 13:45 UTC. Each fit uses the previous two years, with future-label
  endpoints purged strictly before validation begins. No historical/fresh stage.
- Fixed 6h UTC decision grid; next-bar open fill. Cash/units accounting, exposure
  drift, max change .08, deadband .01, 5.5bps one-way costs, 10% annual borrowing.
  Both costs double in stress; all arms start with common B&H inventory 1.
- Three feature groups: existing gap-aware base16; base16 plus technicals;
  base16 plus signed-flow, trade/volume intensity and average trade size.
- Fixed Ridge(alpha=100) and HGB(max_iter=100, leaves=7, min_leaf=64,
  l2=10, learning_rate=.04, no early stopping); no tuning grid or warm starts.
- Horizon 24/96/672 bars (6h/1d/7d), three outcomes: terminal log return,
  nonnegative maximum adverse log excursion from fill, and square root of the
  sum of squared within-horizon marked log increments.
- Fixed return mapper: tanh(2 * mu / max(predicted_vol, .001)), asymmetric
  allocation around B&H with .50 floor and 1.12 ceiling. Downside mapper
  replaces mu with mu - .25 * max(predicted_adverse_excursion, 0).
- 3 groups x 2 models x 3 horizons x 2 mappers = 36 learned policies. Six
  training-climatology mappers, B&H, the frozen robust overlay and its version
  on the same common inference mask are controls. Every quarter must have its
  full registered grid and >=99.5% observed price coverage.
  Model and scaler fits see training rows only. Same finite feature intersection
  across groups controls forecast-score and policy decision comparability.

## Oracle meanings

The ML perfect-outcome diagnostic supplies realized future outcomes to the same
fixed mapper. It isolates forecast error; it is not a global return upper bound.
The RL planning diagnostic searches complete action sequences with a beam width
of 32, and objectives log(NAV) - lambda * maximum drawdown for lambda in {0,1}.
It uses identical cash/units execution and hold/.50/1/1.12 intentions.
It is a feasible hindsight reference, a lower bound on the optimum of the
searched objective, not an exact bound and not a trained RL model. Neither
diagnostic supplies training labels, weights, selection or live decisions.
Technical indicators cannot raise a true perfect-information upper bound while
prices, actions and costs stay fixed; they may reduce the attainable forecast gap.

## Robustness and decision rule

Quarter regime is fixed from available 90d momentum divided by 7d annualized
volatility * sqrt(90/365) at its first decision; thresholds +/- .5 define
bull/bear/sideways. This is a start-of-quarter condition, not an assertion that
the entire quarter remains in that regime. Each candidate reports normal/stress
quarter metrics, target coverage and per-regime aggregates. Candidate ranking
maximizes the worst of mean AlphaEx and negative mean DDDelta over all observed
regimes and both cost settings. Missing groups or fewer than three quarters in
any group prevent a regime-robust pass. Strict >0 AlphaEx and <0 DDDelta are
the user direction criterion; this run cannot establish high-probability
generalization because the development data are reused.

Forecast diagnostics: out-of-time return rank IC, sign accuracy, and MSE skill
against the training mean for all three targets. Their scores never use oracle
actions. All failed trials remain in the result ledger. No automatic promotion,
test reruns, deployment or capital allocation follows a favorable mean.

## Primary research used to choose these diagnostics

- [Brown & Smith, Dynamic Portfolio Optimization with Transaction Costs](https://pubsonline.informs.org/doi/10.1287/mnsc.1110.1377):
  information relaxation/dual bounds motivates separating full-information
  bounds from feasible policies. Our finite beam search is not that exact bound.
- [Elmachtoub & Grigas, Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005):
  forecast error and decision error need separate measurement. No claim that
  this experiment implements or reproduces SPO+.
- [Binance official public-data specification](https://github.com/binance/binance-public-data):
  kline quote volume, trade counts and taker-buy volume support the inexpensive
  flow feature family. These fields are not order-book imbalance.
- [Giacomini & White, Tests of Conditional Predictive Ability](https://escholarship.org/uc/item/5jk0j5jh):
  state-conditional prediction evaluation motivates fixed past-defined regimes.
- [White, A Reality Check for Data Snooping](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152):
  exploration breadth must be accounted for; this reused-data pilot does not
  claim a selection-adjusted significance result.
