# Risk calibration v1 registration

This experiment is fixed after the prior48 policies failed all-regime joint
economic criteria, before seeing these calibration results. All validation
quarters have prior research exposure; this is adaptive exploratory research.
It does not amend a P1 lock or access later test windows for selection.

Question: does the observed6h risk predictability survive stronger causal
baselines and disjoint calibration, and can it support return decisions with
honest uncertainty? Keep neural architecture optimization out of scope.

## Fixed chronology and family

Use the same13 actual validation quarters and6h UTC grid. Before each quarter:
18months model fit,3months multiplicative-scale/mean-bias calibration,
3months interval-score calibration. Purge every label endpoint strictly before
its segment boundary. Outcome at origin t finishes at t+25 bars for H24.
Validation calibration is frozen; no online update sees uncompleted labels.

Six variance forecasters: trailing squared-log-return means over24/96/672bars
rescaled to24bars, HAR-style Ridge(alpha1) on log trailing variance at24/96/672/
2880bars, and base16/technical HGB on log next24bar realized variance. HGB keeps
100iterations,7leaves,min_leaf64,l2=10,learning_rate.04,seed20260905,no early stop.
Feature rows use only t-1 and prior bars, with full nominal history and .995
coverage. All fits and comparisons use common finite-feature support.

One shared technical Ridge(alpha100, training-only standardization) predicts
6h return. It is fit on the same18months, independent of all future intervals.
Raw risk forecasts are positive exp(log-variance prediction), clipped only to
the fixed numeric variance range[1e-12,1] for safe calculations; this is not
assumed to be a conditional mean variance.

Two versions per forecaster: raw return/variance; calibrated return plus mean
return residual from the first3month calibration segment, and variance times
c=mean(actual_variance/raw_variance) on that segment. This c minimizes held-out
QLIKE over a multiplicative scale family. All calibration parameters are saved.

On the disjoint next3month segment, fixed90% interval widths use finite-sample
corrected quantiles of abs(return-mu)/sqrt(variance) and abs(log(actual_sigma/
predicted_sigma)). Minimum64complete origins in each calibration segment.
Quantile correction alone does not grant exchangeable or per-regime coverage
for dependent BTC observations. Report coverage AND width, no exact guarantee.

## Forecast and economic comparisons

Report variance MSE, RMS MSE, QLIKE, return MSE,90% return coverage/width,
90% volatility coverage/log-width and each start-of-quarter trend group.
Same complete actual-target rows for scores; future masks never cancel orders.

Fixed24 policy family:6risk forecasters x2versions x2mappers. Point mapper uses
tanh(2mu/sigma), .5/1.12 asymmetric allocation around B&H. Interval gate requests
1.12 only if return lower bound exceeds2*one_way_cost+annual_borrow*24/35040,
.5 only if upper bound< -2*one_way_cost, and1otherwise. This threshold is a fixed
conservative heuristic, not a dynamic optimum or return guarantee. Maxstep,
deadband, delay, cash/units, costs and2x cost replay remain identical. B&H and
frozen robust overlay are controls. No hindsight labels or weights are used.

No candidate may be promoted merely for90% marginal coverage. All three trend
means must improve both economic signs in base and2x costs with adequate sample
counts, and high-probability generalization still requires prospective evidence.

Primary basis: [Corsi HAR-RV](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1365738),
[Patton variance forecast losses](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf),
[dependent conformal inference](https://proceedings.mlr.press/v75/chernozhukov18a.html),
[ACI](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
The methods here are a static exploratory calibration recipe, not a reproduction
of the papers' empirical studies or proof of their assumptions on BTC.
