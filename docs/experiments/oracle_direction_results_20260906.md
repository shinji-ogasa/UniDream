# Stage17: causal direction replacement results

**No new strongest model is established; retain all candidates as exploratory results and do not promote them.** Learned Perp31 direction replacements satisfy positive AlphaEX and negative MaxDDdelta in the observed all/bull/bear/sideways means at both costs, but every learned replacement worsens both overall economic metrics against its own frozen parent. All four classifier probability-loss gates fail in both I and E. The two Perp mapped means lower overall E return MSE slightly, but the registered all-stratum predictive requirements still fail. Technical29 variants also fail the absolute economic sign conditions.

This is one completed, preregistered 32-fit attempt on reused development quarters. It is not independent confirmation, and the goal of likely accuracy improvement across trends remains unmet. No model is selected or deployed.

## Frozen execution and provenance

Source/config/protocol/preflight were committed and pushed at `6ae673fcdfeed29280256450c05eb8905af77ee3` before new real labels, weights, priors, fits, probabilities or orders. The run completed once (PTY 16393, exit 0), with eight fold-complete log records, 32 classifiers and 16 shared prior estimates. There were no retries, new risk fits, S recalibration, additional-test use, or live/paid execution.

Original validation 5–12 spans April 2021–April 2023; its old test(f)=validation(f+1) alias makes these also old test 4–11. Original fit 18 months, S 3 months, I 3 months and E 3 months, Technical29/Perp31 columns and all six masks were retained. E has 2,586 inference and 2,574 scored origins; I has 2,523 scored origins. The 12 unscored E origins retain causal direction predictions. The original 332 fallback and 2 current-open-gap opportunities remain. Regime inventory is 2 bull / 4 bear / 2 sideways; I grouped by E-start regime is explicitly retrospective.

StandardScaler is unweighted; four L2 logistic models/fold use fixed C=1, lbfgs, tol=1e−8 and max_iter=1000. Magnitude sample weights are abs(y)/fsum(abs(y_i)/n). Each fold has two shared T-only priors. Predictions replace only the direction of each own frozen half-mean magnitude: sign(logit)*abs(parent_mu); risk and controller are unchanged. This is a surrogate mean, not newly calibrated E[Y|X]. S raw logits are saved but never scored or calibrated; S-derived parent bias/anchor enter mapped means only from I onward.

The 52 policies retain 36 old controls (12 causal, 24 hindsight) and append 16 causal names (four learned plus four prior means ×hold/fallback). The cumulative adaptive name count is 174→190. They produce 416 rows/832 cost accounts, 160 return scores, 96 classification scores and 720 new bound artifacts. Old future-information controls remain hindsight-only, and no finite RL search was rerun.

## Observed economics: all new policies

Entries below are equal-quarter means in **percentage points relative to B&H**, not annualized returns. Negative MaxDDdelta is better. Stress uses the same base targets with twice fees and borrowing. Strict joint counts require AlphaEX>0 and MaxDDdelta<0 at both costs within the same quarter. Each policy has its own cash and inventory.

| Mean | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters | All-strata means pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | hold | -0.974415 | -1.878756 | -1.460219 | -1.763709 | 3/8 | false |
| Technical29 ordinary | fallback | -1.272279 | -2.380583 | -1.729213 | -2.277811 | 3/8 | false |
| Technical29 magnitude | hold | -1.159516 | -2.597529 | -1.616736 | -2.514670 | 3/8 | false |
| Technical29 magnitude | fallback | -1.953387 | -2.528147 | -2.393397 | -2.449843 | 4/8 | false |
| Perp31 ordinary | hold | 3.086659 | -2.788922 | 2.566077 | -2.672388 | 5/8 | true |
| Perp31 ordinary | fallback | 2.080828 | -2.778153 | 1.585923 | -2.664776 | 5/8 | true |
| Perp31 magnitude | hold | 3.335463 | -3.113130 | 2.811854 | -3.007415 | 5/8 | true |
| Perp31 magnitude | fallback | 1.687686 | -2.528492 | 1.198274 | -2.430444 | 5/8 | true |
| Technical29 ordinary prior | hold | -3.514797 | 2.343785 | -3.801041 | 2.467586 | 0/8 | false |
| Technical29 ordinary prior | fallback | -3.668853 | 1.469594 | -3.904912 | 1.554450 | 0/8 | false |
| Technical29 magnitude prior | hold | -3.514797 | 2.343785 | -3.801041 | 2.467586 | 0/8 | false |
| Technical29 magnitude prior | fallback | -3.668853 | 1.469594 | -3.904912 | 1.554450 | 0/8 | false |
| Perp31 ordinary prior | hold | -3.129668 | 2.329733 | -3.448841 | 2.455374 | 0/8 | false |
| Perp31 ordinary prior | fallback | -3.275077 | 1.452133 | -3.537519 | 1.533515 | 0/8 | false |
| Perp31 magnitude prior | hold | -3.129668 | 2.329733 | -3.448841 | 2.455374 | 0/8 | false |
| Perp31 magnitude prior | fallback | -3.275077 | 1.452133 | -3.537519 | 1.533515 | 0/8 | false |

Ordinary and magnitude fit-prior probabilities differ, but their signs agree in every fold: positive in 5–11 and negative in 12. Thus each group’s two prior mapped means and target paths coincide exactly. Both registered controls remain in the results; their equality is not extra replication. All prior controls have 0/8 strict joint quarters.

## All learned policies by regime

| Classifier | Regime | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | bull | hold | -3.190298 | 2.995482 | -3.519096 | 3.157656 | 0/2 |
| Technical29 ordinary | bull | fallback | -1.670540 | 1.640425 | -1.957513 | 1.764809 | 0/2 |
| Technical29 ordinary | bear | hold | 3.242527 | -2.442233 | 2.559390 | -2.330388 | 2/4 |
| Technical29 ordinary | bear | fallback | 1.634384 | -2.740190 | 1.002509 | -2.633368 | 2/4 |
| Technical29 ordinary | sideways | hold | -7.192416 | -5.626040 | -7.440559 | -5.551716 | 1/2 |
| Technical29 ordinary | sideways | fallback | -6.687344 | -5.682376 | -6.964354 | -5.609318 | 1/2 |
| Technical29 magnitude | bull | hold | -0.903506 | 0.687862 | -1.174115 | 0.813533 | 1/2 |
| Technical29 magnitude | bull | fallback | -2.103961 | 1.811874 | -2.373138 | 1.925547 | 1/2 |
| Technical29 magnitude | bear | hold | 2.577817 | -2.897952 | 1.877562 | -2.826007 | 2/4 |
| Technical29 magnitude | bear | fallback | 1.217388 | -3.196197 | 0.568261 | -3.129245 | 2/4 |
| Technical29 magnitude | sideways | hold | -8.890193 | -5.282074 | -9.047952 | -5.220201 | 0/2 |
| Technical29 magnitude | sideways | fallback | -8.144363 | -5.532067 | -8.336973 | -5.466429 | 1/2 |
| Perp31 ordinary | bull | hold | 1.110534 | -1.268355 | 0.763556 | -1.101490 | 1/2 |
| Perp31 ordinary | bull | fallback | 0.723729 | -0.928746 | 0.377867 | -0.762373 | 1/2 |
| Perp31 ordinary | bear | hold | 3.141849 | -2.387818 | 2.445288 | -2.278204 | 2/4 |
| Perp31 ordinary | bear | fallback | 1.516035 | -2.678007 | 0.871771 | -2.574142 | 2/4 |
| Perp31 ordinary | sideways | hold | 4.952402 | -5.111699 | 4.610176 | -5.031654 | 2/2 |
| Perp31 ordinary | sideways | fallback | 4.567514 | -4.827854 | 4.222284 | -4.748446 | 2/2 |
| Perp31 magnitude | bull | hold | 2.149374 | -2.965536 | 1.822239 | -2.789160 | 1/2 |
| Perp31 magnitude | bull | fallback | 0.845483 | -1.705629 | 0.517599 | -1.542488 | 1/2 |
| Perp31 magnitude | bear | hold | 3.020859 | -2.133913 | 2.289970 | -2.049821 | 2/4 |
| Perp31 magnitude | bear | fallback | 1.408761 | -2.420560 | 0.733895 | -2.344307 | 2/4 |
| Perp31 magnitude | sideways | hold | 5.150759 | -5.219160 | 4.845237 | -5.140860 | 2/2 |
| Perp31 magnitude | sideways | fallback | 3.087740 | -3.567220 | 2.807708 | -3.490674 | 2/2 |

All four Perp31 policies pass observed regime-mean signs, yet each passes only 5/8 individual quarters. All Technical29 variants have negative bull/sideways AlphaEX, and positive bull DDdelta. Regime means do not guarantee each quarter, and 2/4/2 reused quarters cannot support a high-probability statement.

## Paired economic change from each own original half mean

These are new minus own parent at the same rule, in percentage points. Positive Alpha change and negative DD change would indicate improvement. Every new learned policy instead has negative overall Alpha change and positive overall DD change at both costs.

| Classifier | Rule | Base Alpha change pt | Base DD change pt | Stress Alpha change pt | Stress DD change pt |
| --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | hold | -2.123373 | 4.152618 | -2.403490 | 4.189430 |
| Technical29 ordinary | fallback | -2.278797 | 4.402281 | -2.550478 | 4.440582 |
| Technical29 magnitude | hold | -2.308474 | 3.433845 | -2.560007 | 3.438468 |
| Technical29 magnitude | fallback | -2.959905 | 4.254716 | -3.214663 | 4.268550 |
| Perp31 ordinary | hold | -0.526283 | 2.598779 | -0.782902 | 2.632118 |
| Perp31 ordinary | fallback | -1.298624 | 3.381647 | -1.557933 | 3.425434 |
| Perp31 magnitude | hold | -0.277479 | 2.274571 | -0.537124 | 2.297091 |
| Perp31 magnitude | fallback | -1.691766 | 3.631309 | -1.945582 | 3.659766 |

The full machine-readable summary also retains every matched-prior economic contrast and turnover/trade differences. Beating the weak prior controller does not establish improvement over the original learned half controller.

## Probability prediction: all streams, both segments

Losses are equal-quarter means. Brier and log loss are lower-is-better; accuracy is shown as a percentage. Weighted scores use realized |Y| only during retrospective scoring. A magnitude-weighted model’s sigmoid targets a tilted distribution, not ordinary P(Y>0|X). No zero-weight segment was dropped; all 96 score denominators are positive.

| Segment | Classifier | Brier | Logloss | Accuracy % | Weighted Brier | Weighted logloss | Weighted accuracy % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interval | Technical29 ordinary | 0.266619 | 0.735556 | 51.609023 | 0.273148 | 0.751877 | 50.740427 |
| interval | Technical29 magnitude | 0.273547 | 0.758926 | 51.598849 | 0.279351 | 0.777129 | 50.833723 |
| interval | Perp31 ordinary | 0.263581 | 0.726715 | 52.045226 | 0.269828 | 0.742395 | 51.815353 |
| interval | Perp31 magnitude | 0.269089 | 0.743875 | 51.474534 | 0.274403 | 0.760662 | 51.365379 |
| interval | prior ordinary | 0.251012 | 0.695177 | 49.778420 | 0.251570 | 0.696295 | 48.839093 |
| interval | prior magnitude | 0.250671 | 0.694493 | 49.778420 | 0.251047 | 0.695246 | 48.839093 |
| evaluation | Technical29 ordinary | 0.268063 | 0.737525 | 50.415832 | 0.272957 | 0.748332 | 48.645603 |
| evaluation | Technical29 magnitude | 0.277234 | 0.763898 | 49.831445 | 0.282423 | 0.776452 | 48.836864 |
| evaluation | Perp31 ordinary | 0.266198 | 0.731681 | 51.025457 | 0.270205 | 0.741003 | 50.684816 |
| evaluation | Perp31 magnitude | 0.275733 | 0.757156 | 50.994723 | 0.279087 | 0.766694 | 50.689038 |
| evaluation | prior ordinary | 0.251067 | 0.695286 | 49.683367 | 0.251570 | 0.696292 | 48.172422 |
| evaluation | prior magnitude | 0.250783 | 0.694717 | 49.683367 | 0.251269 | 0.695687 | 48.172422 |

Every learned model has worse overall ordinary and weighted Brier/logloss than BOTH priors in both segments. Ordinary models also have lower overall losses than their same-group magnitude-weighted counterpart. Perp E direction accuracy near 51% and positive uncosted signed-return means do not reverse the probability-loss failure. No significance is assigned to those direction counts.

## Mapped-return prediction

MSE is shown ×10^6, MAE ×10^3. Percent change is new MSE/reference MSE−1; negative is improvement. References use the exact same rows. The two groups retain separate own half magnitudes.

| Classifier | Segment | MSE ×10^6 | MAE ×10^3 | MSE change vs own half % | MSE change vs prior % | Zero MSE ×10^6 | Fitmean MSE ×10^6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | interval | 369.962439 | 12.993818 | -0.492912 | -0.416723 | 367.679703 | 368.257564 |
| Technical29 ordinary | evaluation | 324.589198 | 12.050156 | 0.283833 | -0.229514 | 315.467427 | 316.211081 |
| Technical29 magnitude | interval | 371.346886 | 13.016214 | -0.120544 | -0.044070 | 367.679703 | 368.257564 |
| Technical29 magnitude | evaluation | 324.296213 | 12.051935 | 0.193314 | -0.319570 | 315.467427 | 316.211081 |
| Perp31 ordinary | interval | 368.245064 | 12.974824 | -0.584762 | -0.559860 | 367.679703 | 368.257564 |
| Perp31 ordinary | evaluation | 322.153152 | 11.985327 | -0.128246 | -0.661950 | 315.467427 | 316.211081 |
| Perp31 magnitude | interval | 369.962233 | 13.007008 | -0.121178 | -0.096159 | 367.679703 | 368.257564 |
| Perp31 magnitude | evaluation | 322.138620 | 11.974497 | -0.132751 | -0.666431 | 315.467427 | 316.211081 |

Both Perp mapped means reduce overall E MSE by about 0.13% versus own half; both Technical mapped means worsen it. All four fail the registered requirement to improve mapped MSE against zero, fit mean, own half and matched prior in every stratum, in both I and E. Original E half scores are unchanged. Probability quality, mapped-return quality and economic quality therefore give different evidence; none is substituted for another.

## Fitted priors, ties and numerical validation

| Fold | Fit rows | Nonpositive | Positive | Ordinary prior | Magnitude prior | Iterations: TechOrd/TechMag/PerpOrd/PerpMag |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 800 | 376 | 424 | 0.530000000 | 0.520318915 | 77/89/84/90 |
| 6 | 1034 | 468 | 566 | 0.547388781 | 0.545708703 | 77/84/80/93 |
| 7 | 1313 | 608 | 705 | 0.536938309 | 0.538380879 | 86/88/81/91 |
| 8 | 1500 | 699 | 801 | 0.534000000 | 0.522733447 | 82/91/84/93 |
| 9 | 1503 | 698 | 805 | 0.535595476 | 0.527388069 | 83/85/92/93 |
| 10 | 1634 | 772 | 862 | 0.527539780 | 0.516918798 | 89/91/92/99 |
| 11 | 1672 | 808 | 864 | 0.516746411 | 0.513209066 | 86/94/89/90 |
| 12 | 1794 | 902 | 892 | 0.497212932 | 0.485899927 | 90/91/90/99 |

All 32 models converged within the fixed iteration limit and passed the scalar acceptance checks. The runtime normalized-gradient maximum is 1.4416003547403967e−7 against the fixed 1e−6 bound. All mapped inference arrays have zero zero-logit and zero parent-magnitude observations. Each of I and E has one scored zero-return observation. No epsilon or tie-dependent tuning was used.

The run log contains 384 RuntimeWarning records from sklearn matrix multiplication (divide/overflow/invalid). Their cause is not established. They were retained, not suppressed or used to authorize a retry. Selected inputs and accepted fitted outputs were finite, and every saved prediction was checked through scalar arithmetic; those checks bound the verified outputs rather than explaining the warning mechanism. There were no ConvergenceWarning records.

The predeclared magnitude-weight diagnostics below use T only. Ordinary weights are all one. The concentration statistic sum(w)^2/sum(w²) is descriptive; it is not an independent sample count under serial dependence. Magnitude mean weight is exactly 1 except fold 5 at 1.0000000000000002.

| Fold | Max magnitude weight | Zero-weight fit rows | Weight concentration count |
| --- | --- | --- | --- |
| 5 | 8.772098 | 0 | 363.829830 |
| 6 | 8.486898 | 0 | 485.595339 |
| 7 | 10.004293 | 0 | 590.958889 |
| 8 | 9.295202 | 0 | 691.878032 |
| 9 | 9.075080 | 0 | 693.792330 |
| 10 | 10.507235 | 0 | 770.707769 |
| 11 | 9.907962 | 1 | 799.819576 |
| 12 | 9.643598 | 1 | 850.705361 |

## Independent audit evidence

All independent audits passed without refitting:

- [Source audit](oracle_direction_evidence_20260906/oracle_direction_source_audit_20260906.json) matched the pre-freeze feature/label/mask bindings and all 2,120 ancestral artifacts.
- [Model audit](oracle_direction_evidence_20260906/oracle_direction_model_audit_20260906.json) independently checked 32 fitted models, 45,000 fit-model rows, 30,112 predict-model rows and 16 priors. Normalized-gradient maximum was 1.4416003547403967e−7, objective difference 0, gradient difference 1.0842e−19, minimum Hessian eigenvalue +0.0005574136008918624. Saved logits differed by at most 1.3323e−15 and probabilities by 2.2204e−16; all scalar directions agreed. This supports fitted-state stationarity and reproducibility, without identifying the warning or solver stopping mechanism.
- [Own-account audit](oracle_direction_evidence_20260906/oracle_direction_audit_20260906.json) reproduced all 832 cost accounts and 128 new paths / 44,032 decisions. AlphaEX, MaxDDdelta, targets, own NAV/exposure and utility-trace differences were zero; mean exposure differed by at most 2.220446049250313e−16. All 288 old controls / 576 accounts were unchanged. Its local audit script was adapted to the frozen fit-NPZ schema; research source and outputs were unchanged.
- [Score and summary audit](oracle_direction_evidence_20260906/oracle_direction_score_audit_20260906.json) independently recomputed all 160 return scores, 96 classifier scores and complete summary contrasts. Maximum differences were 4e−18 for return scores, 1e−16 for classifier scores, 3.5e−15 for summary economics and 1.2775e−15 for paired contrasts. The 720 new artifact bindings and 2,120 ancestral bindings were verified. Numerical tolerances were not changed after outcomes.

Ten old B&H/scale-mean cost rows contain stored DD deltas of ±1.1102230246251565e−16. None belongs to the new policies, so no new strict economic sign flag rests on such a rounding-scale DD value. Full source scripts and JSON details are preserved alongside this report; file-binding counts differ by auditor scope and are not independent replications.

## Interpretation and next question

The Stage16 future-sign Oracle exposed a large information-sensitive gap. This fixed causal direction experiment does not recover it: the learned replacements have higher controller turnover in this comparison, and all learned policies worsen overall AlphaEX and DDdelta versus their own original half mean. Weighting by return magnitude does not improve overall probability losses. These observations are compatible with noisy or overconfident finite-sample directional estimates, but do not identify a unique cause or prove there is no useful signal.

Do not retune this run or open the additional test to choose a classifier. The next bounded development question is whether shrinking/regularizing directional logits improves proper-score reliability while preserving useful direction information; a positive temperature leaves sign(logit) unchanged and therefore cannot improve this fixed sign-only controller by itself. Any new fit, calibration, feature or controller mapping requires a new frozen comparison on permitted development data. Model-architecture optimization remains deferred, and the original independent-confirmation/P1 boundaries stay intact.

[Registration](oracle_direction_registration_20260906.md) and [research note](oracle_direction_research_20260906.md) preserve the pre-outcome decisions. [Cost weighting](https://hunch.net/~jl/projects/reductions/costing/finalICDM2003.pdf) and [proper scoring](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) support the objectives’ interpretation, not a BTC performance guarantee.

## Bound output files

| Item | SHA256 or Git revision |
| --- | --- |
| source freeze | 6ae673fcdfeed29280256450c05eb8905af77ee3 |
| preflight.json | a8f20d76fa6ed17592c53c044cdaada7f7a08bd9a2c244fe9fa4a56c3d8eebcd |
| registration.json | 06ef781a2835b25ccfc01db9c758ecf79c7fabefe23346af37db1ec19cefbdab |
| results.json | c659163526547d5aecc75ccd8a9f987a4000eee3152cd6552f78c1428b158657 |
| run.log | 29954374c3a82ea950544366da612199a749342d5c225ff13546486353466993 |
| config YAML | c707ea8385a8f7df1b38a9e57a535358f91d51f0a193c297cc412b1cbf364dd2 |

The JSON `registration_sha256` is the canonical-object digest, distinct from the registration file SHA above. Full test suite: 714 tests passed in 57.558 seconds, exit 0; `git diff --check` passed. Verification log SHA256: `2f06298e5bb966f20addf71ffa98bacdb51045829de423bd69a184901a53b8e6`. Machine-readable rows, summaries, fold manifests and independent audit scripts/results are in [the evidence folder](oracle_direction_evidence_20260906/results.json); models/NPZ/traces remain locally in the registered output directory and are hash-bound by fold manifests.
