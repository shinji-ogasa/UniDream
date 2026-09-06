# Stage20 pre-outcome research: the fixed short8 × direction task comparison

2026-09-06. Research and contract review only. This note reads the published local hypothesis, prior protocols/research, the existing feature source and primary papers. No new real fit, coefficient, return statistic, probability, loss, order or policy outcome was computed or inspected; no additional-test 15–24 access. The eventual frozen registration is authoritative.

## A small falsifiable comparison, with a limited interpretation

Adding the **entire existing short8 block** to Technical29 for the two original C1 logistic objectives is a defensible next experiment. It tests a previously untested representation/task combination: Technical37 ordinary and magnitude-weighted direction prediction, against the corresponding frozen Technical29 classifiers. Prior Ridge mean-prediction failure is not a logical proof that this representation contains no directional information. Conversely, direction improvement does not establish conditional-mean improvement.

Keep one fixed 37-column representation, ordinary/magnitude C1 only, eight original development folds, and no new subset, threshold, horizon, feature-window, calibration, regularization or architecture search. Two new classifiers per fold give **16 fits**. Generate one new mean stream from the magnitude classifier using the already registered `a_T * (2.0*q - 1.0)` and two policies (hold/fallback). Do not map the ordinary classifier to a mean. C1 is the original matched setting, not a setting chosen because a previous result was favorable.

The comparison identifies predictive performance of the **whole regularized fitting procedure with the block added**, not a causal contribution of the eight indicators or a specific column. Correlated extra standardized columns can alter the effective L2 geometry. Even an exact duplicate column can spread a coefficient across coordinates and reduce its squared penalty. Equal C, labels and rows therefore do not imply equal functional complexity. This is a limitation to report, not grounds to add a new normalization or control grid now.

Use failure wording such as: **“This frozen Technical37 procedure did not show the registered improvement on these reused periods.”** A failed descriptive gate does not establish the absence of information, a formal statistical rejection, or failure of every possible estimator. A successful gate would likewise not establish all-trend reliability.

## Three directly checked primary sources

1. **Christoffersen and Diebold, Financial Asset Returns, Direction-of-Change Forecasting, and Volatility Dynamics.** The original author/institution-hosted working paper, especially sections 2.1–2.2, constructs time-varying sign probability with a constant nonzero conditional mean and changing volatility. Thus sign predictability and conditional-mean predictability are distinct. This supplies a logical reason to evaluate a classification task after a mean task fails, not evidence that these short8 features work. Their modeled horizon behavior and U.S. equity illustrations do not validate six-hour BTC or the weighted-logistic estimand. Full working paper read; the final 2006 publisher abstract was also checked.
   - [Original Wharton working paper](https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/0405.pdf)
   - [Final Management Science article](https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0520)

2. **Kitron and Wengrowicz (2026), Short-horizon mean reversion in cryptocurrency markets, arXiv v1, 22 August 2026.** Sections 2, 3 and 5 report short-horizon directional reversal and compare signs, taker imbalance and their combination. The flow combination does not improve their out-of-sample AUC over signs alone. Reported predictability decays within hours; the paper reports disappearance by four-hour sampling. Its largest gross edge is roughly 1.3 bp per trade versus a 5 bp benchmark round trip. These are strong reasons the present hypothesis can fail, not a promise of a trading edge. Their intrabar labels, constrained sign-lag model, selected multi-asset universe and flow conditioning differ from our delayed-fill six-hour label. The flow mechanism is observational; a contemporaneous conditioning split is not causal identification. This is a recent preprint, not independent verification of our feature block.
   - [Original paper, pinned v1](https://arxiv.org/html/2608.21888v1)

3. **Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation.** Section 3, examples 1 and 3, supports Bernoulli Brier and logarithmic losses as strictly proper for the probability distribution assessed. This motivates evaluating probabilities, rather than selecting on hard accuracy or rank alone. Propriety is a population criterion; it does not prove finite-sample calibration, independence, robustness to drift, or profitable inventory decisions. Outcome-dependent weighting changes the target distribution. The weighted-ratio algebra below is our direct derivation, not a BTC empirical claim from this paper.
   - [Author-hosted original JASA paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

The bounded search on 2026-09-06 found a reason to test and clear transfer limits. It did not find a paper proving Technical37, these exact eight windows, or six-hour post-cost gains. No paper's feature/threshold grid or significance procedure is imported.

## Labels, estimands and exact reuse

Keep `Y_t = log(close[t+24] / open[t+1])`, original UTC six-hour decisions and original masks. On the 15-minute bar-open grid, the target's archival maturation bound is `t+375 minutes`; chronological split masks must prevent a fit label from crossing its authorized boundary. This event-time convention does not prove historical receipt availability. Unscored inference origins still receive a forecast and may place a causal order.

For `D=1{Y>0}` and `g(X)=E[|Y||X]`, ordinary population log loss targets `p*(X)=P(D=1|X)`. Magnitude-weighted population log loss targets

`q*(X) = E[|Y|*D | X] / g(X)`, when `0<g(X)<infinity`.

Since `Y=|Y|(2D-1)`, including at Y=0, `E[Y|X]=g(X)*(2q*(X)-1)`. When `g(X)=0`, Y=0 almost surely conditionally and weighted loss does not identify q. The fixed finite C1 logistic model need not equal either population optimum. Weighted q is not the ordinary up probability.

Even genuine improvement in q does not by itself establish improvement in the constant-amplitude approximation: `a_T` is a saved historical mean absolute return, not `g(X)`. The identity leaves both a probability-estimation error and a conditional-magnitude approximation error. Ordinary improvement alone is still less informative for this mean mapping. Separate probability, mean and policy conclusions.

Reuse frozen T labels, weights, priors, mean absolute return and exact fitmean. Use the existing weight normalization and exact stored arithmetic, rather than silently recomputing with a different reduction. Fit an unweighted StandardScaler on T only, followed by the same pinned C1/L2/lbfgs/max_iter=1000/tol=1e-8/seed setup, with the existing thread limit and convergence failure policy. S/I/E labels never fit a coefficient, scaler, probability calibration or mapping parameter.

The first 29 feature columns, rows and calendar must match the old Technical29 fitting/prediction evidence exactly. Append exactly the saved Stage15 short8 names/order. Its body-sign arithmetic remains `np.sign(np.log(close/open))`; all new columns receive the existing single `shift(1)`. Do not rebuild the old column definitions, double-shift, impute, compress missing periods or shrink support. Nonfinite added features on any required original fit/predict row must fail the whole registered run. Only selected labels/features may be inspected by a pure fitter; poison outside authorized selection must remain irrelevant.

## Fixed reporting and gate contract

All losses are computed on paired original score rows, reduced within quarter as already specified, then aggregated with equal quarter weights. The four strata are all / bull / bear / sideways, with the original 2/4/2 E-start inventory. I/E are separate; I's E-start regime grouping is explicitly retrospective. Magnitude scores retain `sum(abs(Y))` denominators. An undefined quarter/stratum is null and makes the associated strict condition fail; do not omit it or substitute a different metric. Binary ties, zero returns, zero logits, q=.5/0/1 and soft-mean zero remain explicit, with no epsilon.

| Evidence layer | Predeclared hurdle | Scope |
| --- | --- | --- |
| Ordinary probability | Ordinary Brier **and** log loss strictly lower than the matched saved ordinary T prior **and** same-loss Technical29 C1 | I/E, every stratum |
| Magnitude probability | Absolute-return-weighted Brier **and** log loss strictly lower than the matched saved magnitude T prior **and** same-loss Technical29 C1 | I/E, every stratum |
| Mapped mean | MSE strictly lower than each of six frozen references: old Technical C1 soft, old Technical C1 magnitude hard, own original half, mapped prior, exact fitmean, zero | I/E, every stratum |
| Absolute economics | AlphaEX>0 and MaxDDdelta<0 | Each new rule, base/stress, every stratum |
| Paired economics | New-minus-reference AlphaEX>0 and MaxDDdelta<0 for all six matching-rule references | Each new rule, base/stress, every stratum |
| Generalization | No high-probability/all-trend confirmation can be established by this reused development comparison | Remains false |

The two probability conditions have **64 scalar contrast inequalities** total: 2 classifiers × 2 references × 2 losses × 2 segments × 4 strata. The one new mapped mean has 48 MSE contrasts. The two policies have 32 absolute economic inequalities and 192 paired inequalities, or 224 combined. These counts are bookkeeping of a demanding intersection rule, not independent tests or a calibrated confidence level. Repeated constants and multiple costs are not independent evidence.

Save each classifier's probability gate separately. A joint “both prediction tasks improved” feature-family claim requires their conjunction. The two policies' required probability evidence comes from their magnitude source; ordinary remains a predeclared diagnostic task. Freeze this relationship explicitly before outcomes. Do not use a failed ordinary result to delete the task, nor use an ordinary success to bypass the magnitude requirement.

Keep MAE, accuracy, weighted accuracy, rank, per-quarter improved/equal counts, pooled-row losses, turnover/trades/costs and strict same-quarter joint counts as descriptive results. None replaces a failed registered gate. Proper-loss improvement with failed mean/economic conditions means the present mapping/controller failed its utility hurdle; it does not prove that features are useless. Passing economics with failed proper scores does not establish improved predictive probability.

## Inventory and fail-closed preservation

Retain all 80 old economic policies, 24 old means and 10 old classifiers exactly. One new magnitude soft mean gives 25 means / **400 I/E return records**. Two new classifiers give 12 streams / **192 classification records**. Two new economic names give 82 policies / **656 rows / 1,312 base-stress accounts**, and the cumulative causal-name count is 218→220. New fits remain exactly 16; controls need no refits.

Require complete old 640 rows, 384 return records and 160 classification records to remain identical, including inherited metadata/hash bindings. Confirm unchanged six masks, risk, cost accounting and base-target stress replay. Missing-feature/order rules remain the same. No real result or model is selected from this note. A pass could motivate the existing independent confirmation contract; a failure closes this fixed comparison without adding a post-outcome branch to it.

## Local review bindings

Read in `/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905`. These are source/research bindings, not new outcome computations.

- `docs/experiments/oracle_next_information_hypothesis_20260906.md`: `bfbeaaae00f4f626a629c7452ef791241fef6a7e14af844b4b7734bcddd98550`
- `docs/experiments/oracle_direction_research_20260906.md`: `d808c13f2a2b12cfe8f62daa56ab880d7d97a4c1ff586703ac1ea6da6538186c`
- `docs/experiments/oracle_soft_direction_research_20260906.md`: `112567cf8300f9e2394d8b59683b89913b35317e6b8c6d099648634d34102db8`
- `unidream/experiments/oracle_short_features.py`: `9389cc8cc550e69dd27dc7771233d532db726f373fe0bfdf3dd9ae5065b1482b`
