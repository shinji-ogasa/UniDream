# Stage19: frozen magnitude-weighted probabilities to continuous means

Prepared 2026-09-06 before the proposed mapping experiment is frozen. This note reuses the Stage17 weighting derivation, reads primary papers and inspects the existing source/storage contract. No new real means, labels, statistics, forecasts, scores, policies or fitted coefficients were calculated. Registration will be authoritative.

## Estimand and the additional approximation

Let Y denote the same registered h24 log return, D=1{Y>0}, and A=|Y|. Conditional on the features and the inherited availability support, set g(x)=E[A|X=x]. For 0<g(x)<infinity, the unrestricted population minimizer of absolute-return-weighted Bernoulli log loss is

`q*(x) = E[A*D | X=x] / g(x)`.

Indeed, conditional expected loss is `-E[AD|x]*log(q) - E[A(1-D)|x]*log(1-q)`. Its minimizer is that ratio, with boundary optima allowed when a conditional class has zero magnitude mass. Since `A*(2D-1)=Y`, including at Y=0,

`E[Y|X=x] = g(x) * (2*q*(x)-1)`.

These formulas are a direct derivation from weighted risk, not a claimed BTC result from a cited paper. Logistic regularization, finite training data, restricted linear features and changing conditional distributions do not guarantee that a saved classifier estimates q*. The identity also does not turn its tilted probability into ordinary P(Y>0|X). If g(x)=0, Y=0 almost surely conditionally, while q* is unidentified by this weighted loss; do not divide by zero to invent a probability.

The proposed fixed continuous mean is

`mu_soft[t] = a_T * (2.0*q_saved[t] - 1.0)`,

where a_T is the already saved training mean absolute return. This replaces the unavailable conditional magnitude g(x) by a single historical constant. It equals the conditional mean only when the probability estimate and the magnitude approximation are appropriate; even stationary unconditional E|Y| does not imply E[|Y||X] is constant. Selected training availability and temporal drift matter. Conditional homogeneity/stability and adequate estimation are assumptions, not verified properties of these BTC features.

The exact population comparison is

`mu_soft - E[Y|X] = 2*a_T*(q_hat-q*) + (a_T-g(X))*(2*q*-1)`.

Thus magnitude misspecification can remain even with a correct q*. Conversely, a return-MSE gain would not by itself isolate improved probability calibration or a correct conditional magnitude. a_T is a mean absolute log return, not a conditional variance, realized volatility, or the absolute value of the parent mean. Keep the horizon, log-return units and support unchanged.

Do not add an ordinary-probability soft map. For ordinary p=P(Y>0|X), the corresponding mean also depends on conditional positive and negative magnitudes; `a_T*(2p-1)` generally targets the wrong quantity. Adding that mapping would broaden the experiment without the present estimand argument.

## Primary sources and limited applicability

1. **Zadrozny, Langford and Abe (2003), Cost-Sensitive Learning by Cost-Proportionate Example Weighting**, [author-hosted original paper](https://hunch.net/~jl/projects/reductions/costing/finalICDM2003.pdf), sections 2.1–2.2. The paper gives the distribution-tilting interpretation of example weights and relates weighted risk to risk under the tilted distribution. Costs can vary by example and need not be available at prediction time. Its sampling/learning guarantees assume independent examples and suitable risk minimization. We use that interpretation to derive the specific weighted-log-loss ratio above; the paper does not validate these regularized BTC classifiers, the constant-a_T substitution, or this inventory/cost/DD controller.

2. **Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation**, [author-hosted original paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), especially section 3; [publisher DOI](https://doi.org/10.1198/016214506000001437). Proper Bernoulli logarithmic and quadratic scores evaluate probability forecasts for the distribution being assessed. Outcome-dependent weighting changes that distribution. A deterministic change to the downstream return/action map cannot improve the unchanged probability scores. Propriety alone supplies neither finite-sample calibration nor economic-performance guarantees. The paper's general forecast-evaluation theory does not choose a_T, the controller, or the five-reference hurdle.

## Exact stored arithmetic and support

Use the four already fitted **magnitude** streams only: Technical29 / Perp31 times C1 / L2unit. Load saved logits and probability arrays from their bound Stage17/18 NPZs; do not call predict, refit, recalibrate, retime or rebuild features. Use the identical original mapped-inference masks, including unscoreable inference origins. Keep mapped S values unavailable; I and E remain the scored segments. A future score mask must never determine where a new mean or an order exists.

Load a_T from Stage17's saved `fit_abs_return_mean`, which used `math.fsum(abs(float(Y_i))/n_fit for i in fit)`. Do not recompute its value with NumPy mean or from I/E labels. Load the exact fit-mean control from the saved `fit_return_mean`, which used `float(np.mean(fit_returns))`. These are existing scalar inputs, not new fits or new distribution estimates.

Preserve the explicit arithmetic `a_T * (2.0*q_saved - 1.0)` on selected rows. Do not replace it with `a_T*tanh(logit/2)`, `2*a_T*q_saved-a_T`, a rounded display probability, a recomputed sigmoid, a clipped value or a sign-repaired equivalent. In real arithmetic sign(mu_soft)=sign(logit) for positive a_T; finite probabilities can round to .5, 0 or 1. Hard mapping used sign(logit), so such discrepancies must be recorded rather than removed.

For the mapped-prior control, prefer the frozen **saved prior probability** in the original bound `*_magnitude_prior_direction` arrays. Existing Stage17 code stores sigmoid(saved prior logit), which can differ by an ulp from `fit_priors['magnitude']`. Freeze the selected source explicitly. A constant source must be constant and agree across its S/I/E arrays and parent groups, or fail closed. The mathematical identity `a_T*(2*pi_T-1)=mean_T(Y)` need not be bit-exact after the stored normalization, prior/logit/sigmoid and mean arithmetic. Retain both mapped-prior and exact-fitmean controls; record their difference without substituting one for the other.

For each stream/segment, record selected q=.5, q=0, q=1, logit=0, sign(2q-1) versus sign(logit), and mu_soft=0 counts. For constant controls, record mapped-prior minus exact fitmean, its sign, and any resulting exact/unequal means or paths after the frozen run. No epsilon, zero snapping or outcome-based tie resolution. Validate selected q in [0,1], finite logits and finite positive a_T, matching calendars/support, and finite mapped outputs; unavailable cells stay unavailable. A zero-valued mean is a valid prediction, not a missing value.

All probability records must match the parent bit-for-bit on identical score masks. Retain the old logit-based scoring arithmetic, rather than derive new scores from transformed means or apply another sigmoid to a mean. Zero-weight score segments remain null, retain their quarter and fail the corresponding strict directional condition; do not silently omit them.

## Small complete comparison and falsification contract

The proposed family is sufficient: four learned soft means plus three constants per parent group (mapped prior, exact fit mean and zero) give ten new means and twenty policies under the existing hold/fallback rules. Keep all sixty old policies, giving eighty policies and the stated cumulative causal-name count 198 to 218. The planned inventory is 121 artifacts per fold, 968 total; 24 return means on I/E give 384 score records, while all ten inherited probability streams keep their 160 records. The 64 learned I/E mechanism records include mean absolute new/hard/parent forecasts alongside the tie and sign counts. Constants repeated across groups are named controls, not independent predictions or evidence. When their numerical inputs coincide, their deterministic paths should also coincide under identical masks/risk/execution; do not remove duplicates or count them as new replications.

For each of the **eight learned policy names**, use these fixed five references with the same missing-input rule: its own same-classifier hard map; its original half mean; mapped prior; exact fit mean; zero. Preserve every contrast, including unfavorable ones.

1. **Identity/validity:** saved probabilities and their classification scores, all old means/scores/policies, horizon, six masks, risk, execution/cost conventions, and source bindings stay exact. No new fit, prediction call, scale estimate, threshold, feature, selection or causal-support change. Any failure invalidates the run; it is not a reason to remove rows or switch mappings.

2. **Mapped predictive direction:** on both I and E, require strictly lower equal-quarter return MSE than all five references in all/all-bull/all-bear/all-sideways strata. Zero and exact fit-mean comparisons should agree with the inherited numeric baseline scores. Report MAE, rank/sign accuracy, per-quarter improved/equal counts and all paired MSE differences, but do not replace the registered MSE hurdle with a favorable metric. Constant controls themselves have no learned-candidate gate and are not required to improve over themselves.

3. **Absolute economics:** require AlphaEx>0 and MaxDDDelta<0 for equal-quarter means in all four strata at both base and doubled costs. Also report strict same-quarter joint success counts; a mean pass is not a quarterwise guarantee.

4. **Paired economics:** I support the root's conservative requirement of AlphaEx change>0 and DDdelta change<0 against **all five** references in all four strata at both costs. It directly tests whether this map improves the inherited controller and simple reference decisions. It is a deliberately demanding predeclared decision rule, not a theorem or a multiple-testing-adjusted significance level. Report every pair's turnover/trade/cost differences even when the hurdle fails. Keep unchanged absolute and paired conditions separate.

5. **Probability evidence:** require exact identity, not strict improvement versus the same frozen probability stream. Set the new probability-improvement flag false by construction. Carry the appropriate inherited matched-prior proper-loss diagnostics, clearly marked unchanged; no new probability-skill claim follows from a downstream map. Return-MSE or economic improvements, if any, must be described as properties of the mapping, without claiming that q became more accurate.

All aggregate losses remain equal-quarter, with the same original 2 bull / 4 bear / 2 sideways development quarters; I regime labels remain retrospective E-start groups. Base targets are replayed at doubled costs without stress optimization. Do not classify missing forecasts as zero, restrict orders to scoreable labels, or relabel the zero-mean utility controller as B&H: risk/cost/inventory terms still determine its actions.

This is another exploratory mapping diagnosis after repeatedly observed development failures. All-trend probability and regime-count gates stay false; a passing descriptive comparison would still require the existing independent-confirmation contract. No additional feature/model family, parameter grid or new stage is scientifically necessary to test this bounded question.
