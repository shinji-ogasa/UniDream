# Stage20: fixed short8 features for C1 direction prediction

2026-09-06. Written before new real fits, statistics, probabilities, mappings, scores or orders. The code/config/input-only preflight and this protocol must be committed and pushed before execution. There is no selection or deployment in this experiment. Formal P1 `results_observed=false` is unchanged.

## Hypothesis and fixed inventory

Technical29 plus the exact Stage15 price5/flow3 block, named `technical_short_both` (37 columns), may improve the original ordinary and magnitude-weighted C1 direction tasks. The representation was previously assessed with Ridge100 raw mean prediction; this does not establish its value for either classification objective. The comparison assesses the entire fitting procedure including changed L2 geometry, not an individual indicator's causal effect. [Research and primary sources](oracle_short_direction_research_20260906.md) explain the rationale and transfer limitations.

Two new models per fold, eight folds, exactly16 fits. No old model is refitted. Ordinary is a diagnostic probability task. Only magnitude probability is mapped to one new mean `technical_short_both_magnitude_soft`, yielding two new policy names for unchanged hold/fallback rules. All80 old policies,24 old means and10 old classifiers remain, giving82 policies,656 economic rows,1312 base/stress accounts,25 means/400 I/E return records,12 classifiers/192 classification records. The adaptive causal-name ledger rises218→220. No extra C, horizon, threshold, feature subset, window, calibration, risk fit or architecture search is allowed.

## Unchanged data and estimation

Use existing BTCUSDT15m bar-open data strictly before2023-04-16T13:45UTC. Spot's inherited loader decodes the full bound Parquet before semantic truncation; do not claim later bytes were never decoded. UM archive metadata is not proof of receipt-time availability. Additional-test15–24 is not modeled, scored or selected on here.

Validation folds5–12 preserve T18m/S3m/I3m/E3m, six-hour decisions, original six masks and full15m grid. Target is `log(close[t+24]/open[t+1])` with the original all24 future-bar requirement and375-minute maturity. T/S/I label maturity is strictly before its boundary; E uses its inherited boundary. Original test(f) aliases validation(f+1); reused development periods are not independent confirmation. Regimes are fixed at E start:2 bull/4 bear/2 sideways; I groupings by those E regimes are retrospective.

Reconstruct exactly Stage15 features through its frozen input-only prepare. Verify all original masks and selected Technical29 matrices against Stage17 saved fit_data, and selected Technical37 fit/predict matrix hashes against Stage15 fitted provenance. The additional columns must be finite on every original required mask; any failure stops the entire run with no reduced support, imputation or fallback feature choice. Features use completed t−1 bars with one final shift. Exact canonical columns and order are enforced.

T returns, labels `return>0`, ordinary ones and magnitude weights `abs(return)/a_T`, shared priors, `a_T=fsum(abs(float(y_i))/n)` and numpy fitmean must match saved Stage17 evidence exactly. New recomputation verifies the same priors; it introduces zero unique prior estimates. Use unweighted T-only StandardScaler and pinned Stage17 LogisticRegression: C1, L2, lbfgs, tol1e−8,max_iter1000,intercept,seed20260906,thread limit2. Nonconvergence or iteration limit fails. All state must be finite; independent scalar objective and normalized gradient infinity≤1e−6, scalar logits≤1e−12 and probabilities≤1e−14 are mandatory. Preserve any numerical warnings and report them; passing guards does not identify their cause.

Map new magnitude q with unchanged `a_T*(2.0*q-1.0)`. The T prior supplied to the mapper comes from the bound Stage17 prior NPZ, not a differently rounded fresh sigmoid. All three mapped constant arrays must equal their saved Stage19 arrays exactly. Ordinary probability is never mapped to return/order. This is an approximation using constant historical absolute return; it does not identify conditional magnitude or ordinary up probability. New I/E probabilities are saved separately without a misleading inherited mean field.

## Execution and accounting

Risk stays the saved Technical variance. Same conditional/fallback controller, own state per policy, risk-aversion1,cost multiplier2, one-way fee0.00055,annual borrow0.10, step0.08,deadband0.01,intent range0.5–1.12. Units start1/open0 and cash0; drifting passive exposure can exceed the intent cap. Orders fill next known open only; missing fill skips without rollover. Hold does not order without a forecast; fallback targets1 only at a known current open. Borrow continues over gaps. Stress replays identical base targets with doubled fee/borrow. Zero-mean risk control is not B&H.

Original E inference2586/score2574 and I mapped inference2537/score2523 stay separate. The12 unscored E and14 unscored I origins receive predictions. Preserve332 fallback decisions and2 missing-current-open gaps. Recompute old accounts/scores and require exact old record preservation including target/trace bindings; old models and predictions remain immutable.

## Predeclared descriptive gates

Report all four strata (all,bull,bear,sideways) and I/E separately. First reduce within a quarter with frozen score helpers, then equal-weight quarters. Undefined weighted scores remain null and fail a strict gate. Preserve signed accuracy, MAE, rank, pooled-row MSE, zero/tie counts, per-quarter joint counts, turnover and trades as descriptive diagnostics. No epsilon, metric substitution, omitted stratum or post-outcome gate change.

1. Ordinary classifier: ordinary Brier and log loss strictly below both saved ordinary T prior and same-loss Technical29 C1 in each stratum and segment.
2. Magnitude classifier: weighted Brier and weighted log loss strictly below both saved magnitude T prior and same-loss Technical29 C1 in each stratum and segment.
3. Feature-family probability statement requires both classifier gates (64 scalar contrasts). Each policy's probability requirement uses only its magnitude source; ordinary remains fully reported.
4. New mapped MSE must be strictly lower than each of six frozen references in all strata, I/E separately (48 contrasts): `technical_magnitude_soft`, `technical_magnitude_direction`, `technical_half`, `technical_soft_mapped_prior`, `technical_soft_fit_mean`, `technical_soft_zero`.
5. Absolute economics: AlphaEX>0 and MaxDDdelta<0 for each new rule, cost case and stratum (32 inequalities).
6. Paired economics: new-minus-reference AlphaEX>0 and MaxDDdelta<0 for all six same-rule references, both costs and all strata (192 inequalities).
7. High-probability generalization and regime-count confirmation remain false. These reused quarters and correlated inequalities do not yield a confidence level. A successful descriptive intersection can motivate future independent confirmation, not an all-trend guarantee. No candidate is automatically promoted.

A failed condition means this fixed procedure did not demonstrate the registered improvement on the reused periods. It is not a proof of absent information or of every possible model's failure. Probability, mapped mean and economic outcomes must be described separately.

## Artifacts and execution discipline

95 artifacts/fold:2 models,1 selected training/prediction snapshot,1 fit provenance,4 classifier NPZs (2 tasks×I/E),2 soft-mean NPZs,1 mapping provenance,82 target NPZs,2 new decision traces. Save32 classifier diagnostics and16 mapping diagnostics across8 folds. Source binding is the union of Stage19/Stage15 ancestry plus the three new modules. Input-only preflight hashes both full input chains and exact supports. Full repository tests and whitespace checks run before the freeze. Existing output files are immutable. A live observation timeout is not failure; poll its same session. A terminal partial failure is recorded and no alternative estimator/parameter is tried within this registration.

After the once-only run, independent audits recheck selected features, fitted states/gradients/probabilities, mapping, return/classification scores, all accounts, old-record equality and all summary flags. Publish this protocol, research, config, preflight, runtime manifests/logs, audit scripts/results and readable outcome tables. Large binary model/array/trace files remain local hash-bound. All work is research-only; no live trades, paid infrastructure, external messages or production edits.
