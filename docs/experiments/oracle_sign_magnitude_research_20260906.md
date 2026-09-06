# Stage16: sign and magnitude information, fixed hindsight diagnosis

Planning research, 2026-09-06. No new local outcome, coefficient, forecast, threshold, loss or order was calculated. No later-test outcomes were accessed. This note supports a bounded diagnostic using the ORIGINAL Stage12 technical_half and perp_half forecasts, not a selected Stage15 feature group. It proposes no model architecture, fitted teacher, weight grid or deployment.

## Fixed four-cell intervention and its meaning

For each original mean stream and each original hold/fallback controller, let mu be its saved causal half-mean forecast and y its saved h24 realized log return. On the unchanged saved score support only:

    base = mu
    sign = sign(y) * abs(mu)
    magnitude = sign(mu) * abs(y)
    full = y

Outside score support retain the original mean exactly, including learned values on unscored inference slots. Preserve the original inference mask, risk forecast, complete economic window, missing-action rule, own inventory, entry/skip semantics and cost schedule. Doubled-cost accounting replays the same base intents. This is a hybrid hindsight intervention on an already fixed support, not a perfect forecast at every possible decision.

The full cell should reproduce the corresponding Stage12 return-only Oracle with learned risk. Require forecast, target/trace and account parity against that fixed endpoint. Reproducing an old endpoint is not new confirmation. Do not import Stage12 realized-variance or RL objectives into this sign/magnitude family.

Use mathematical sign with sign(0)=0, including signed floating zero. Do not substitute a binary +1/−1 sign convention or a post-outcome epsilon. When mu=0, both partial cells remain zero, whereas full=y may act. When y=0, both partial cells and full are zero. If y and mu are nonzero with matching signs, magnitude equals full; sign equals base. Record zero cases separately. Existing binary sign accuracy based on (>0) groups zero with nonpositive predictions, so sign-only need not attain 100% recorded directional accuracy when a zero mean accompanies a positive return. Preserve the old metric, explicitly label it, and do not call it exact three-class sign agreement.

There is an essential interpretation boundary: abs(mu) is the absolute value of a conditional-mean forecast, not an estimate of E[abs(Y)|information]. Likewise sign(mu) is the sign of a mean forecast, not necessarily the most likely outcome sign or a fitted probability forecast. A small conditional mean can be appropriate when positive/negative outcomes cancel. The magnitude intervention is therefore replacement of the mean forecast's magnitude by one realized future absolute return; it does not evaluate an existing absolute-return model.

The magnitude cell is also not a conditional-variance Oracle. abs(y) has return units; conditional variance is a conditional expectation of squared deviations and has squared-return units. A realized absolute h24 endpoint return is neither that conditional variance nor the sum of intraperiod squared returns. Keep the saved calibrated risk input unchanged and name the intervention accordingly.

## Algebraic checks versus economic evidence

For nonzero y, correcting the sign while retaining abs(mu) weakly decreases pointwise squared return error. It leaves loss unchanged when signs agree and reduces loss by 4*abs(mu*y) when they are opposite. A zero forecast remains zero. When y=0, the sign intervention sets its output to zero, removing the original mu² loss. Thus non-worsening sign-only MSE is a deterministic check of the transformation, not evidence that this future sign can be learned. Full=y yields zero scored MSE by construction. The same facts hold in any fixed subset of scored rows, including tail groups.

Magnitude-only need not improve MSE. If signs agree it reproduces y; if signs oppose, it produces −y and incurs 4*y². With a previously small magnitude, that can enlarge error. This is not proof that forecasting magnitude is useless: the intervention preserves the original potentially wrong sign and reveals realized rather than conditional magnitude.

The controller is nonlinear and sequential. Correct signs can still fail to overcome risk, transaction-cost and deadband thresholds when abs(mu) is small. Positions, step limits and costs also depend on prior actions and inventory. Realized endpoint direction does not specify the adverse path inside the next six hours. Therefore neither sign-only nor full information guarantees improved AlphaEx or MaxDDDelta, and the economic effects of the two replacements need not add. A partial cell can outperform full under this fixed finite controller without contradicting the value of information theorem: the controller has not optimized over all measurable policies and is not guaranteed to use additional information optimally.

Report the fixed five contrasts sign−base, magnitude−base, full−base, full−sign and full−magnitude. If retained in registration, also report the four-cell interaction full−sign−magnitude+base separately for each metric. This interaction is a symmetric, scale-specific descriptive contrast; the marginal effect of either replacement depends on the state of the other. It is not a causal allocation or percentage attribution of the Oracle gap. Do not label any cell a global optimum or an upper bound over policies. A bounded feasible hindsight rollout may respect trading constraints while using information unavailable in live decisions.

## Fixed fit-q90 tail description

A single threshold per fold is the 90th percentile of absolute RETURNS on its original selected fit rows, using numpy.quantile(method='linear'). Compute the eight thresholds only after source/registration freeze. They are distribution summaries, not eight new predictive models; both streams use the identical fold threshold. No E outcome selects the quantile, window, threshold or stream.

Evaluate the eight mean streams (two original bases, four partial interventions, two old full endpoints) on three partitions: all saved score rows; abs(y)>=threshold; and abs(y)<threshold. This gives 8*3*8=192 per-fold metric records. The latter two partition all scored rows, including exact equality in the tail. These are explicitly hindsight groups defined by realized future outcomes, not an online tail detector. They may describe errors; they must never gate orders or receive an attribution of realized dollar PnL.

Register empty-group behavior. Empty loss is null, not zero. For equal-quarter or regime aggregates, every constituent quarter's relevant metric must exist or the aggregate remains null, with defined/total quarter counts displayed. A separately labelled pooled-row statistic has its own explicit denominator. Nonempty small groups can have MSE/MAE while rank correlation remains undefined. Do not silently discard such quarters for paired contrasts or calculate each side on different quarters. A zero threshold can put every row in the tail; ties and empty other groups are valid edge cases, not reasons to change the quantile.

A fit-q90 threshold need not select 10% of later E observations or comparable market severity across periods. Volatility changes can alter the subgroup count dramatically; preserve counts and do not infer probability calibration from the label 'tail'. Keep the original start-regime definitions, all eight quarters and existing 2/4/2 coverage. No p-value, confidence level, independent-confirmation claim or causal-model promotion follows from this diagnostic.

## Three primary sources, read directly

1. Christoffersen and Diebold (2006), *Financial Asset Returns, Direction-of-Change Forecasting, and Volatility Dynamics*, Management Science52(8),1273–1287. [Original publisher abstract](https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0520). They distinguish conditional-mean, sign and volatility dependence and show that sign dependence can exist without conditional-mean dependence; time-varying volatility with nonzero expected returns can generate sign dependence. This supports separating directional and mean objectives, not transferring a trading advantage to BTC six-hour returns. The publisher abstract was read directly; no unobserved full-text cost protocol is claimed.

2. Anatolyev and Gospodinov (2010), *Modeling Financial Return Dynamics via Decomposition*, JBES28(2),232–245. [Author's accepted paper](https://pages.nes.ru/sanatoly/Papers/Decomp.pdf), linked from the [author's publication page](https://pages.nes.ru/sanatoly/Papers/Decomp.htm). Sections1–2 model returns as sign times absolute return and obtain the mean from their JOINT distribution. In general E[S*A] includes dependence and is not simply E[S]*E[A]. Their US-stock application and joint model motivate asking about components; they do not prove this diagnostic's fixed controller, absolute half-mean or BTC horizon is appropriate. The author's PDF is a June2008 accepted version of the paper published in2010. Its binary-sign formulas assume the relevant zero-probability convention; our measured zeros require explicit handling.

3. Anatolyev and Gerko (2005), *A Trading Approach to Testing for Predictability*, JBES23(4),455–461. [Author's accepted paper](https://pages.nes.ru/sanatoly/Papers/Profit.pdf). Its comparison contrasts unweighted sign agreement with a trading-return term sign(forecast)*sign(actual)*abs(actual): direction errors on large outcomes matter differently from errors on small ones. This directly motivates reporting loss/economic measures separately and fixed tail descriptions. Its weekly S&P500 strategy is not our constrained own-inventory BTC policy. The authors also identify the strong all-lags/leads independence assumption behind the original EP/DA asymptotics. No EP/DA significance test or asymptotic guarantee is imported into dependent, repeatedly reused BTC quarters.

The formal E[Y]=E[sign(Y)*abs(Y)] identity motivates the question; it does not make the two information factors independently learnable. Large hindsight value identifies sensitivity to supplied information under this controller. A later claim of learnability would require a separately preregistered causal forecast comparison with proper chronological maturity and prospective evidence.
