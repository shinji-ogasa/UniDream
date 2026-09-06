# Stage17: frozen linear direction replacement — pre-outcome research/design note

Prepared 2026-09-06. No market files, new labels, fit coefficients, forecasts, losses or policy outcomes were read or computed for this note. Only published sources and the installed sklearn API/source were inspected. This note is a design recommendation; the eventual registration is authoritative.

## Bounded comparison

The proposed family is a useful small falsifiable test: original Stage12 Technical29 and Perp31, each with ordinary versus absolute-return-weighted logistic loss; all original feature, fit, calibration, inference and economic supports retained. StandardScaler uses the same unweighted fit rows for both losses. Freeze C=1, L2, intercept enabled, class_weight=None, lbfgs, max_iter=1000, tol=1e-8, no weight/threshold/feature grid or subsequent probability calibration. Four logistic fits per fold means 32 fits over eight folds. These are new linear classification heads, not an architecture search or a reproduction of the parent Ridge100 mean objective.

For each parent, retain its frozen half-mean magnitude and common calibrated risk, and replace direction on every existing inference-supported origin, including origins without a scoreable future label:

`mu_new[t] = np.sign(z[t]) * np.abs(mu_parent_half[t])`.

Use a name such as `inference_support` rather than `I`: the latter also denotes the historical three-month interval-calibration segment. Future score support must never restrict these causal substitutions. Outside inference support, retain the original missing-input behavior. Keep hold/fallback and base-intent/stress replay fixed.

## What weighting estimates

Let Y be the registered h24 log return, B=1{Y>0}, p=sigmoid(z), and A=|Y|. The following is an algebraic application of weighted risk, not an empirical claim from a BTC paper:

- Ordinary population log loss is minimized at `p*(x)=P(Y>0|X=x)`.
- Absolute-return-weighted population log loss is minimized at
  `q*(x)=E[A B|X=x] / E[A|X=x]`, whenever `0<E[A|X=x]<infinity`.
- Consequently `2*q*(x)-1 = E[Y|X=x]/E[|Y||X=x]`. Its sign equals the conditional-mean sign at the unrestricted population optimum; q* is generally not the ordinary up probability.

This follows by minimizing `-E[AB|x] log(q)-E[A(1-B)|x] log(1-q)`. A regularized finite-sample linear model need not recover either optimum, and changing distributions need not preserve it. A forecast sign does not estimate E|Y|. The retained `abs(mu_parent_half)` is the absolute value of a mean forecast, not a magnitude-model estimate. Mapping it onto a classifier sign is a fixed hybrid controller, not a newly calibrated conditional mean.

The final helper uses `abs(y_fit)/math.fsum(abs(float(y_i))/n_fit for i in fit)`. This scalar-fsum mean, rather than NumPy mean reduction, is the exact arithmetic to bind in registration and audits. Scaling all weights changes the loss-to-penalty balance when C is fixed. Mean normalization makes their sum approximately the same fit-row count as the ordinary fit; save the actual sum and rounding residual. This fixes one confound without making the models equally complex or equally well estimated. Installed sklearn 1.8.0 `_logistic_regression_path` was inspected read-only: lbfgs uses `l2_reg_strength=1/(C*sum_sample_weights)`. Pin the version and actual estimator parameters; do not rely on a future version's defaults.

## Constants, ties and failure rules that must be registered

The ordinary and weighted fit priors are `pi=mean(B_fit)` and `pi_w=sum(A_fit*B_fit)/sum(A_fit)`. Each is estimated only once per fold on the shared return support, then reused with each parent's magnitude. There are two unique prior estimates but four mapped constant streams; together with four learned streams this gives eight new means and 16 new causal policy names, taking the stated cumulative history from 174 to 190. Equality or duplication between constant directions does not authorize dropping a control.

The weighted-prior direction equals the fit-mean-return direction in real arithmetic, since `2*pi_w-1=mean(Y_fit)/mean(A_fit)`. Preserve registered floating arithmetic around exact .5 rather than substituting this identity in code. Constants need frozen logit arithmetic for probability scores and an explicit zero-logit mapping.

`np.sign(0)=0` produces zero mean; ordinary binary scoring `(z>0)` assigns a zero logit to the non-positive class. These are different uses of a tie, and must both be logged. Parent magnitude zero suppresses the mapped mean even if the classifier has a direction. Actual Y=0 is a non-positive ordinary label but contributes zero weight to the weighted fit and weighted scores. Report all three zero counts separately; never introduce an outcome-chosen epsilon.

Fail closed for insufficient fit rows, unsupported/misaligned features, nonfinite selected values, zero/nonfinite mean absolute return, absent positive effective weight in either class, nonfinite fitted predictions, or convergence failure. Record n_iter and warnings; do not silently drop rows, impute, change solver, increase iterations, retry or substitute a prior after seeing a failure. Unselected future labels and unrelated outcome columns should not be inspected by the pure fitter. Make clear that positive feature availability is a conditioning restriction, not evidence that unavailable periods are harmless.

## Minimal informative scoring contract

Freeze the same rows for all compared forecasts, and report the historical interval segment and evaluation segment separately if both are used. Neither segment may update the four fits or their priors. E-start regime labels applied to older interval rows are retrospective grouping, not those rows' contemporaneous regime. Keep the original E regime inventory 2 bull / 4 bear / 2 sideways and all eight quarters.

For all four learned probability streams and both fit-prior probabilities, retain ordinary Brier and log loss, binary accuracy, and absolute-return-weighted Brier/log loss/accuracy. Explicitly distinguish the tilted probability target from ordinary probability calibration. For stable binary log loss, use `logaddexp(0,z)-B*z` with saved finite logits; avoid a result-dependent probability clip. Weighted evaluation scores can use `sum(abs(y_eval)*loss)/sum(abs(y_eval))`: those realized weights are allowed only in retrospective scoring, never inference. Save the denominator. If it is zero, that metric is null; retain the quarter and make the corresponding equal-quarter aggregate/contrast null, without silently averaging only defined quarters.

Each learned stream should be compared to both fit-prior scores, its same-parent ordinary/weighted counterpart, and its own parent mean where that metric is meaningful. The old mean has no probability forecast: do not invent one by applying sigmoid to a return. Compare mapped means through return MSE/MAE versus own parent and existing zero/fit-mean controls. Compare controller AlphaEX, MaxDDdelta, turnover/costs and all-regime signs versus own parent and its same-magnitude priors. Keep all predefined contrasts even if they disagree. Probability skill, mapped-mean skill and economic improvement are separate claims; improvement in one is not joint improvement in all.

Aggregate equal-quarter losses, and paired loss differences on common rows; keep pooled row-weighted summaries distinct. Magnitude-weighted scores have another within-quarter weighting layer. Report the weight concentration (for example max normalized weight and sum(w)^2/sum(w^2)) only descriptively: this is not an independent sample count under serial dependence. No binomial, iid standard error or imported PAC probability applies to these reused BTC quarters. No result from this comparison alone establishes high-probability performance in all trends.

## Direct primary sources and limited use

1. **Zadrozny, Langford and Abe (2003), Cost-Sensitive Learning by Cost-Proportionate Example Weighting**, author-hosted paper, sections 2.1–2.2: <https://hunch.net/~jl/projects/reductions/costing/finalICDM2003.pdf>. The translation theorem relates error under a cost-tilted distribution to weighted error under the original distribution. Costs may vary by example and need not be known at prediction time. Its sampling/learning guarantees assume independently drawn examples and suitable risk minimization; they do not establish this finite linear logistic model's BTC performance, regularization choice or DD control. The specific |Y|-weighted logistic optimum above is our direct derivation, not a claimed formula tested in that paper.

2. **Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation**, author-hosted paper, section 3 and table 1: <https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf>. Bernoulli Brier and logarithmic scores are strictly proper for the probability distribution being assessed. This motivates keeping probability quality separate from hard direction accuracy. Outcome-dependent weighting changes the population being scored; it does not make the weighted model's sigmoid an ordinary event probability. Proper scoring is a property of the population criterion, not a finite-sample calibration or investment guarantee.

3. **Anatolyev and Gerko (2005), A Trading Approach to Testing for Predictability**, author-hosted accepted paper: <https://pages.nes.ru/sanatoly/Papers/Profit.pdf>. Direction agreement and a return-magnitude-sensitive trading statistic contain different information about forecast usefulness. Their simplified trading statistic is not our own-inventory, risk-penalized, borrowing-cost and maximum-drawdown policy. Do not import their hypothesis test, distributional assumptions or significance claims into this repeated-development comparison.

Implementation reference: **scikit-learn 1.8 logistic-regression objective**, <https://scikit-learn.org/1.8/modules/linear_model.html#logistic-regression>. The documented sample-weight and C interaction agrees with the locally inspected installed 1.8.0 source. This supports numerical reproducibility, not the substantive choice of C=1.

No scientifically necessary extra model family is identified. The fixed 2x2 classifier losses with same-magnitude prior controls, explicit tie handling and three separate performance layers already isolate the proposed question sufficiently for a small exploratory study. Freeze first; report every result and failure; defer independent confirmation to the existing separate contract.
