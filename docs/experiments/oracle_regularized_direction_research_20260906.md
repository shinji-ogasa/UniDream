# Stage18: fixed normalized L2 direction comparison — pre-outcome research note

Prepared 2026-09-06. This note inspects primary literature and installed scikit-learn source only. No new market labels, coefficients, forecasts, losses or policy outcomes were read or calculated. The final intended schedule is **C = 1.0 / float(np.sum(frozen_T_weights))**; the earlier C = 1/n_fit proposal is not a second experiment. The eventual frozen registration controls execution.

## One bounded, falsifiable question

Does a single stronger penalty, fixed relative to average weighted training loss, improve the four original direction classifiers beyond their saved C=1 counterparts and the retained fit-prior controls? Retain Technical29 / Perp31, ordinary / absolute-return-weighted loss, identical selected training and prediction matrices, unweighted training-only StandardScaler, fit/intercept/class/tie conventions, and all inference and score masks. Fit exactly four new classifiers per development fold, 32 total, without changing features, architecture, thresholds, probability calibration or the weight distribution.

Map each new logit to `sign(logit) * abs(own frozen parent_half_mu)` on every original inference-supported origin, including origins lacking a scoreable future label. Keep inherited calibrated variance, own-inventory utility, missing-input hold/fallback, price paths, costs and base-intent/stress replay fixed. Four new means times two missing-input rules give eight new causal policy names. Retain all 52 old policy controls: 60 policies, 480 quarter-policy rows and 960 base/stress accounts; stated cumulative policy ledger 190 to 198. Probability output inventory becomes ten classifiers; mapped-return inventory becomes fourteen means, each scored on the historical interval and development evaluation segments. These counts describe a proposed inventory, not an observed run.

## Objective and exact arithmetic

For selected fit labels D_i = 1{Y_i > 0}, frozen sample weights s_i, standardized feature vectors x_i, and z_i = x_i' beta + b, write S = sum_i s_i. The documented L2 objective is

`J_C(beta,b) = sum_i s_i [logaddexp(0,z_i) - D_i*z_i] / S + ||beta||^2 / (2*C*S)`.

The intercept b is unpenalized. Define the normalized coefficient-penalty multiplier as `lambda_effective = 1/(C*S)`; its contribution to the objective is `(lambda_effective/2) * ||beta||^2`. This definition avoids confusing lambda with the coefficient of the unhalved squared norm. [scikit-learn 1.8 mathematical documentation](https://scikit-learn.org/1.8/modules/linear_model.html#logistic-regression).

Use the stored float64 weight arrays unchanged, including their original magnitude normalization. The runtime contract should be:

```python
S_numpy = float(np.sum(frozen_T_weights))
# Fail closed unless S_numpy and its reciprocal are finite and strictly positive.
C = 1.0 / S_numpy
lambda_effective_runtime = 1.0 / (C * S_numpy)
```

Record n_fit, S_numpy, an independent scalar-fsum weight total, C, the actual reciprocal-product multiplier, and their finite rounding residuals. Compute C separately for the two frozen losses; each feature group uses the corresponding same-loss C. Do not silently replace S_numpy by n_fit, scalar fsum, an effective sample size, or a freshly normalized weight vector. Exact arithmetic gives lambda=1. Finite arithmetic need not give bit-exact one, so audit the implementation's operation order and report the residual.

The former C=1/n_fit proposal would give lambda=n_fit/S. It coincides with the accepted choice when S=n_fit exactly, and otherwise differs by the stored mean-one normalization residual. C=1/S expresses the intended average-loss penalty directly and removes that residual from the mathematical contract. It is one predetermined rule, not a tuned choice between two rules. Compared with saved C=1, whose normalized multiplier is 1/S, the accepted multiplier is S times as large in exact arithmetic. No observed outcome establishes that this strength is optimal.

The fixed penalty removes explicit 1/S weakening as training weight increases. For example, exact uniform duplication of fixed weighted examples would leave the normalized objective unchanged under C=1/S. This algebraic invariance does not make independently changing fit samples, feature distributions, class balances, or serial dependence stable. It also does not estimate an independent effective sample count.

The inspected installed version is **scikit-learn 1.8.0**. Its `_logistic.py`, `_logistic_regression_path`, lines 320 and 388, uses `np.sum(sample_weight)` and then `l2_reg_strength = 1.0 / (C * sw_sum)` for lbfgs. The [version-tagged official source](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/linear_model/_logistic.py) agrees. Local source path: `.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py`; SHA256: `683755f12d707f3109e7207c60b27aec76e9a98433bb1b02617f972ccc03edb8`. This is implementation evidence, not an investment-performance argument.

## Literature support and its limits

**Bousquet and Elisseeff (2002), Stability and Generalization**, JMLR 2, 499–526, [original paper](https://www.jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf), Theorem 22, supports a regularization/stability rationale. For bounded kernels and an appropriate Lipschitz convex loss, the paper bounds regularized learners' uniform stability by a quantity proportional to 1/(lambda*n). Its generalization results impose further sampling and boundedness assumptions. This does not establish a numerical bound for our pipeline: serial BTC observations, a fitted scaler, outcome-dependent normalized weights and an unpenalized intercept require separate analysis. Unit sample variance does not imply uniformly bounded future feature norms. Stronger regularization also introduces bias; a fixed nonzero population penalty does not establish consistency for the unpenalized conditional-probability target. No BTC trend guarantee or confidence level follows from citing this theorem.

**le Cessie and van Houwelingen (1992), Ridge Estimators in Logistic Regression**, Applied Statistics 41(1), 191–201, [publisher record and abstract](https://academic.oup.com/jrsssc/article/41/1/191/6990520), DOI 10.2307/2347628, provides original logistic-ridge motivation: coefficient stabilization and improved prediction can be possible under regularization. The abstract was read; an [original-paper PDF mirror](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1992-JSTOR-Cessie-Logistic_Regression.pdf) is available. This is not evidence choosing lambda=1 for six-hour BTC returns, comparing these two weighted objectives, or promising monotonic out-of-sample gains.

## Direction, probability and economics remain separate

The following conclusions are direct algebra/design implications, not paper-specific empirical claims. Refitting with a larger penalty can change relative coefficients and the intercept, so it can change zero-threshold signs. It is not equivalent to multiplying all old logits by one positive scalar. A finite positive post-fit temperature T preserves `sign(z/T)=sign(z)` in exact arithmetic, including exact zero. With all other inputs fixed, such a sign-preserving transform cannot change these sign-only controller paths. This does not motivate running another temperature family.

The intercept remains free. Shrinking feature coefficients tends toward a fit-prior predictor in the very-strong-penalty limit, not necessarily toward probability 0.5. Existing ordinary and magnitude-weighted prior controls are therefore essential. A smaller coefficient norm does not promise smaller pointwise logits, fewer sign changes, lower out-of-sample losses, less turnover, positive AlphaEx or negative MaxDDdelta.

Ordinary logistic loss targets an ordinary up probability at the unrestricted population optimum. Magnitude weighting instead targets `E[|Y|*1{Y>0}|X] / E[|Y||X]`; its sigmoid is a tilted probability, not an ordinary up probability. Retain both proper-loss families for all ten probability streams, but interpret each fitted objective against the corresponding target. Keep strict zero ties, zero actuals, zero-weight groups and null weighted-score denominators explicit. Reusing frozen weights must not become future-label weighting at prediction time.

For each new fit, retain paired proper-loss changes against its same-group/same-loss C=1 model and both prior controls, mapped-return MSE/MAE against its old direction stream, parent half mean and zero/fit-mean benchmarks, and paired economics against its own old controller under both costs. Report every predeclared quarter and all three start regimes with existing equal-quarter weights. A log-loss improvement with unchanged signs is probability improvement only; better classification or mapped MSE without economic improvement does not meet the user target. Better economics without improvement beyond the appropriate prediction baselines does not identify learned predictive skill. Do not change thresholds, select a best objective/group, or propose a tuned penalty grid based on these reused quarters.

No additional model family is scientifically necessary for this bounded comparison. It can falsify the claim that this particular stronger fixed regularization improves the inherited procedure. Success would remain exploratory evidence conditional on reused development periods, not independent confirmation or a probability that AlphaEx > 0 and MaxDDdelta < 0 will persist across future trends.
