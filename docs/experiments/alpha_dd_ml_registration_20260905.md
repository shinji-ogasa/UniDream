# Gap-aware ML continuation — registered before fitting

This is an additional research trial, not the immutable P1 formal result. The
original BTC trial and ETH/BNB trial already produced results; the confirmation
periods below are therefore reused research data, not a newly untouched test.
There are 83 original BTC candidates, 166 cross-asset candidates, and now 25
ML-recipe candidates including B&H. These trial counts must accompany findings;
descriptive bootstrap intervals do not adjust for this sequential search.

## Reason and fixed hypothesis

Original rolling features required every observation over long windows. All 24
ML candidates were unavailable in at least one development fold because fewer
than 256 purged/stride-selected training rows survived. The data-only audit
identified this before evaluating the alternative model predictions.

The separately implemented `alpha_dd_features.make_features` uses a fixed
99.5% observed threshold for rolling statistics, retains the full nominal
warm-up, never interpolates or zero-fills missing values, and adds three observed
coverage features. Every feature is shifted one bar; momentum uses its exact
observed endpoints. The new 16-column frame yields at least 666 eligible training
rows in every registered development fold/horizon in the data-only check.
This is sufficient to execute the registered comparison, not proof that the
sample size or model capacity is sufficient to predict returns accurately.

## Frozen comparison

- Asset: BTCUSDT; audited Spot 15m artifact, current sidecar and ledger hashes
  pinned by each result. No asset search in this trial.
- B&H plus the same 24 ridge/HGB/logistic candidates from the original universe.
  Horizons 7/30 days, floors 0.5/0.8, caps 1/1.12; signal strength 2.
- Same fixed ridge/HGB/logistic hyperparameters and seed 7, two-year train,
  three-month embargo to evaluation, horizon purge, stride 16 and 256-row gate.
  No checkpoint warm-start and no fit on evaluation returns.
- Development folds 0–12 select exactly one candidate by maximum
  `min(mean AlphaEx, -mean MaxDDDelta)` with the original deterministic ties.
  All folds remain in denominators; any unavailable fold excludes that candidate.
- Only that locked candidate and B&H enter historical folds 15–23 and fresh
  fold 24. Neither confirmation set selects parameters. The fresh label is a
  stage name, not a claim of untouched data after the preceding trials.
- Identical next-open delay, UTC hourly decisions, max-step 0.08, deadband 0.01,
  one-way 5.5 bps, negative-cash borrowing 10% annual, and initial B&H inventory.
- Qualification uses all ten equal-weight confirmation quarters: minimum mean
  AlphaEx +1pt and mean MaxDD delta -1pt; preferred +3pt/-3pt. Report historical,
  fresh, cost/borrow 2x stress, fold wins, medians and intervals separately.

The BTC/BNB rule results will not be overwritten or silently promoted into this
ML trial. BNB's already-qualified baseline is the initial HF implementation
target; these ML results are a separately disclosed research comparison, not a
test-set rule for changing that deployment candidate.

## Commands and evidence

```sh
python -m unidream.experiments.alpha_dd_ml --config configs/alpha_dd_ml_20260905.yaml --stage development
python -m unidream.experiments.alpha_dd_ml --config configs/alpha_dd_ml_20260905.yaml --stage historical
python -m unidream.experiments.alpha_dd_ml --config configs/alpha_dd_ml_20260905.yaml --stage fresh
```

Core evaluator and feature-source SHA-256, feature names, candidate universe and
configuration are registered in `codex_outputs/alpha_dd_ml_v1/registration.json`.
The development result hash is pinned in `selection_lock.json` before historical
execution. Trained local joblib hashes and fit target-end cutoffs are persisted.
No orders, paid jobs, P1-manifest changes, or HF publication occur in this recipe.
