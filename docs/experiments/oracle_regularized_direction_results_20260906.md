# Stage18: fixed normalized-L2 direction results

**No new strongest model is established. All eight new policies are rejected.** Stronger regularization improves overall probability losses versus the same C=1 classifiers, but all four new classifiers still lose to both fit-prior forecasts on ordinary and magnitude-weighted Brier/log loss in E. Every new policy has negative overall AlphaEX and positive MaxDDdelta at both costs, and worsens both metrics versus its same-loss C=1 policy and original half-mean policy.

The registered absolute economic, paired economic, matched probability-loss and mapped-return MSE requirements all fail. This is a completed exploratory comparison on reused development quarters, not evidence of likely accuracy improvement across trends. No selection, promotion or deployment occurred.

## Frozen execution and comparison

Source, tests, configuration, protocol, primary research and input-only preflight were committed and pushed at `5a82c270c64a342ab7e9df8105b7d23d1336d876` before the new real regularization statistics, fits, logits, losses or orders. The run completed once (PTY 98384, exit 0), with eight fold-complete records and 32 new classifiers. There were no retries, old-model refits, new risk fits, new calibration, or new unique prior estimates. Sixteen prior recomputations verified the exact saved T-only priors.

Only classifier C changed from 1 to `1.0 / float(np.sum(frozen_T_weights))`. The normalized objective is average weighted log loss plus `actual_lambda * ||beta||² / 2`, with `actual_lambda=1/(C*W)` and a free intercept. Actual C ranged from 0.0005574136008918618 to 0.00125; actual lambda values were 1.0 and 1.0000000000000002. This was one frozen schedule, without a C grid or post-outcome threshold choice.

Technical29 / Perp31 features, selected T matrices, weights, labels, unweighted scalers, all estimator settings other than C, and all six support masks were unchanged. No features were rebuilt. Ordinary labels are Y>0; magnitude weights use the exact saved abs(Y)/fsum(abs(Y_i)/n) normalization. The fitter received T return column 0; all non-T outcome cells were NaN. A finite solver result verifies the computation, not generalization.

Original development validation 5–12 spans April 2021–April 2023, strictly before 2023-04-16T13:45Z. Old test(f)=validation(f+1), so these are also old test 4–11. The chronology is T 18 months, S 3 months, I 3 months, E 3 months, with strict T/S/I label maturity. E has 2,586 inference and 2,574 scored origins; mapped I has 2,537 inference and 2,523 scored origins. The 12 unscored E origins remain actionable. The 332 fallback and 2 missing-current-open opportunities remain. Regimes contain 2 bull / 4 bear / 2 sideways quarters. I grouping by E-start regime is retrospective.

Each new mean is `sign(new_logit) * abs(own_frozen_half_mu)`, a fixed controller surrogate rather than newly calibrated E[Y|X]. Risk is unchanged. Raw S logits are saved, mapped S means remain NaN, and no S score or new S calibration is used. Magnitude-weighted probabilities target a tilted distribution rather than ordinary P(Y>0|X).

The controller keeps UTC 6-hour decisions, next-bar-open fills, initial B&H inventory, fee 0.00055 one-way, annual borrow 0.10, utility risk 1, cost allowance 2, intents [0.5, 1.12], step 0.08 and deadband 0.01. Passive exposure may exceed the intent bounds. Hold sends no order for missing predictions; fallback targets 1 only with known current open. A missing immediately-next open skips the fill without rollover; borrowing continues across gaps. Future labels never gate orders.

All 52 old policies (28 causal and 24 hindsight) remain. Eight new causal names increase the cumulative adaptive count from 190 to 198. The 60 policies produce 480 economic rows / 960 cost accounts, 224 return-score records, 160 classification-score records and 64 mechanism records. All 416 old economic rows, 160 old return records and 96 old classification records remain exact. Hindsight controls remain diagnostics; no Oracle or finite RL beam was rerun.

## Overall economics of all new policies

Values are equal-quarter means in **percentage points relative to B&H**, not annualized. Negative MaxDDdelta is better. Stress replays the same base targets with twice fees and borrowing. Joint quarters require strict AlphaEX>0 and MaxDDdelta<0 at both costs in the same quarter. Each policy uses its own cash and inventory.

| Classifier | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | hold | -2.919109 | 1.398651 | -3.372897 | 1.499633 | 0/8 |
| Technical29 ordinary | fallback | -3.336479 | 0.782059 | -3.765528 | 0.863877 | 0/8 |
| Technical29 magnitude | hold | -2.864962 | 1.380514 | -3.233193 | 1.500361 | 1/8 |
| Technical29 magnitude | fallback | -3.201660 | 0.720213 | -3.533648 | 0.811031 | 1/8 |
| Perp31 ordinary | hold | -3.551952 | 1.818648 | -4.000090 | 1.923159 | 0/8 |
| Perp31 ordinary | fallback | -3.979501 | 1.204870 | -4.394469 | 1.284708 | 0/8 |
| Perp31 magnitude | hold | -2.160282 | 1.079011 | -2.558520 | 1.185774 | 1/8 |
| Perp31 magnitude | fallback | -2.463952 | 0.386337 | -2.824472 | 0.463593 | 1/8 |

## Every new policy by regime

Regimes are assigned from trailing information at E start. They are not classified from the quarter's realized return. Small reused stratum counts cannot establish a high probability of improvement.

| Classifier | Regime | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | bull | hold | -3.043918 | 3.514471 | -3.369078 | 3.694249 | 0/2 |
| Technical29 ordinary | bull | fallback | -1.396812 | 2.068249 | -1.682778 | 2.211185 | 0/2 |
| Technical29 ordinary | bear | hold | 1.275895 | 2.025475 | 0.658890 | 2.085178 | 0/4 |
| Technical29 ordinary | bear | fallback | -0.733374 | 1.725663 | -1.319172 | 1.777471 | 0/4 |
| Technical29 ordinary | sideways | hold | -11.184309 | -1.970818 | -11.440291 | -1.866075 | 0/2 |
| Technical29 ordinary | sideways | fallback | -10.482359 | -2.391339 | -10.740992 | -2.310622 | 0/2 |
| Technical29 magnitude | bull | hold | -4.671670 | 5.116458 | -4.954522 | 5.309244 | 0/2 |
| Technical29 magnitude | bull | fallback | -2.928449 | 3.483024 | -3.131144 | 3.599959 | 0/2 |
| Technical29 magnitude | bear | hold | 3.154376 | 1.202394 | 2.654426 | 1.297227 | 1/4 |
| Technical29 magnitude | bear | fallback | 1.183875 | 0.908770 | 0.719865 | 0.995484 | 1/4 |
| Technical29 magnitude | sideways | hold | -13.096930 | -1.999190 | -13.287104 | -1.902254 | 0/2 |
| Technical29 magnitude | sideways | fallback | -12.245942 | -2.419712 | -12.443178 | -2.346801 | 0/2 |
| Perp31 ordinary | bull | hold | -3.248129 | 3.426985 | -3.575117 | 3.612686 | 0/2 |
| Perp31 ordinary | bull | fallback | -1.616874 | 1.993275 | -1.904216 | 2.141812 | 0/2 |
| Perp31 ordinary | bear | hold | 0.663090 | 2.501111 | 0.078032 | 2.561211 | 0/4 |
| Perp31 ordinary | bear | fallback | -1.315301 | 2.214464 | -1.871156 | 2.266725 | 0/4 |
| Perp31 ordinary | sideways | hold | -12.285858 | -1.154614 | -12.581309 | -1.042472 | 0/2 |
| Perp31 ordinary | sideways | fallback | -11.670529 | -1.602722 | -11.931347 | -1.536431 | 0/2 |
| Perp31 magnitude | bull | hold | -4.272325 | 4.773949 | -4.540812 | 4.950548 | 0/2 |
| Perp31 magnitude | bull | fallback | -2.430926 | 3.024656 | -2.639495 | 3.144756 | 0/2 |
| Perp31 magnitude | bear | hold | 3.487965 | 0.883276 | 2.921151 | 0.962665 | 1/4 |
| Perp31 magnitude | bear | fallback | 1.532559 | 0.596628 | 1.002449 | 0.668179 | 1/4 |
| Perp31 magnitude | sideways | hold | -11.344731 | -2.224456 | -11.535571 | -2.132783 | 0/2 |
| Perp31 magnitude | sideways | fallback | -10.490000 | -2.672565 | -10.663289 | -2.626742 | 0/2 |

## Paired economic changes

These are new minus the named reference, at the same rule, in percentage points. Positive AlphaEX change and negative DDdelta change would be improvement. All entries instead worsen both overall economic metrics, at both costs. The complete result JSON additionally preserves every matched-prior contrast and every turnover/trade difference.

| Classifier | Reference | Rule | Base Alpha change pt | Base DD change pt | Stress Alpha change pt | Stress DD change pt |
| --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | same-loss C1 | hold | -1.944694 | 3.277407 | -1.912679 | 3.263342 |
| Technical29 ordinary | same-loss C1 | fallback | -2.064201 | 3.162641 | -2.036316 | 3.141688 |
| Technical29 ordinary | own original half | hold | -4.068067 | 7.430024 | -4.316168 | 7.452771 |
| Technical29 ordinary | own original half | fallback | -4.342997 | 7.564922 | -4.586794 | 7.582270 |
| Technical29 magnitude | same-loss C1 | hold | -1.705446 | 3.978043 | -1.616458 | 4.015032 |
| Technical29 magnitude | same-loss C1 | fallback | -1.248273 | 3.248360 | -1.140251 | 3.260875 |
| Technical29 magnitude | own original half | hold | -4.013920 | 7.411887 | -4.176464 | 7.453500 |
| Technical29 magnitude | own original half | fallback | -4.208178 | 7.503076 | -4.354914 | 7.529425 |
| Perp31 ordinary | same-loss C1 | hold | -6.638611 | 4.607571 | -6.566167 | 4.595547 |
| Perp31 ordinary | same-loss C1 | fallback | -6.060330 | 3.983024 | -5.980392 | 3.949483 |
| Perp31 ordinary | own original half | hold | -7.164894 | 7.206350 | -7.349068 | 7.227665 |
| Perp31 ordinary | own original half | fallback | -7.358954 | 7.364671 | -7.538325 | 7.374918 |
| Perp31 magnitude | same-loss C1 | hold | -5.495744 | 4.192141 | -5.370374 | 4.193189 |
| Perp31 magnitude | same-loss C1 | fallback | -4.151638 | 2.914829 | -4.022746 | 2.894037 |
| Perp31 magnitude | own original half | hold | -5.773223 | 6.466713 | -5.907498 | 6.490280 |
| Perp31 magnitude | own original half | fallback | -5.843405 | 6.546138 | -5.968328 | 6.553803 |

## Probability prediction: all streams in I and E

Brier and log loss are lower-is-better; accuracy is a percentage. All values are equal-quarter means. Weighted scores use realized |Y| only for retrospective scoring. Both scoring families are retained for every classifier; no zero-weight segment was omitted. All 160 weighted denominators are positive.

| Segment | Classifier | Brier | Logloss | Accuracy % | Weighted Brier | Weighted logloss | Weighted accuracy % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interval | Technical29 ordinary C1 | 0.266619 | 0.735556 | 51.609023 | 0.273148 | 0.751877 | 50.740427 |
| interval | Technical29 magnitude C1 | 0.273547 | 0.758926 | 51.598849 | 0.279351 | 0.777129 | 50.833723 |
| interval | Perp31 ordinary C1 | 0.263581 | 0.726715 | 52.045226 | 0.269828 | 0.742395 | 51.815353 |
| interval | Perp31 magnitude C1 | 0.269089 | 0.743875 | 51.474534 | 0.274403 | 0.760662 | 51.365379 |
| interval | Technical29 ordinary L2unit | 0.250203 | 0.693559 | 51.410896 | 0.253361 | 0.699921 | 48.330078 |
| interval | Technical29 magnitude L2unit | 0.250670 | 0.694492 | 50.593902 | 0.252538 | 0.698251 | 48.777365 |
| interval | Perp31 ordinary L2unit | 0.250188 | 0.693528 | 50.892700 | 0.253234 | 0.699663 | 48.045762 |
| interval | Perp31 magnitude L2unit | 0.250743 | 0.694639 | 50.944560 | 0.252295 | 0.697765 | 49.328114 |
| interval | prior ordinary | 0.251012 | 0.695177 | 49.778420 | 0.251570 | 0.696295 | 48.839093 |
| interval | prior magnitude | 0.250671 | 0.694493 | 49.778420 | 0.251047 | 0.695246 | 48.839093 |
| evaluation | Technical29 ordinary C1 | 0.268063 | 0.737525 | 50.415832 | 0.272957 | 0.748332 | 48.645603 |
| evaluation | Technical29 magnitude C1 | 0.277234 | 0.763898 | 49.831445 | 0.282423 | 0.776452 | 48.836864 |
| evaluation | Perp31 ordinary C1 | 0.266198 | 0.731681 | 51.025457 | 0.270205 | 0.741003 | 50.684816 |
| evaluation | Perp31 magnitude C1 | 0.275733 | 0.757156 | 50.994723 | 0.279087 | 0.766694 | 50.689038 |
| evaluation | Technical29 ordinary L2unit | 0.251526 | 0.696239 | 50.655350 | 0.254550 | 0.702345 | 48.707527 |
| evaluation | Technical29 magnitude L2unit | 0.252064 | 0.697315 | 49.460325 | 0.253989 | 0.701203 | 48.131203 |
| evaluation | Perp31 ordinary L2unit | 0.251531 | 0.696248 | 50.565906 | 0.254389 | 0.702018 | 48.711550 |
| evaluation | Perp31 magnitude L2unit | 0.252235 | 0.697659 | 49.134486 | 0.253574 | 0.700363 | 48.512859 |
| evaluation | prior ordinary | 0.251067 | 0.695286 | 49.683367 | 0.251570 | 0.696292 | 48.172422 |
| evaluation | prior magnitude | 0.250783 | 0.694717 | 49.683367 | 0.251269 | 0.695687 | 48.172422 |

All four new classifiers improve all four overall probability losses versus their same-loss C1 model in both I and E. In E, all four remain worse than BOTH priors on all four losses. In I, the ordinary models beat the ordinary prior on overall matched losses, but fail the predeclared all-stratum requirement; the magnitude models remain worse than their matched prior even overall. Probability improvement versus C1 alone therefore does not establish feature information above the prior. All four registered matched-loss gates fail in both segments.

## Mapped-return prediction

MSE is shown ×10^6. Difference columns are new minus reference MSE, also ×10^6; negative is better. References are scored on the exact same rows. These are equal-quarter means, separate from the pooled-row MSE retained in the full JSON.

| Classifier | Segment | MSE ×10^6 | Change vs C1 ×10^6 | Change vs own half ×10^6 | Change vs prior ×10^6 | Zero MSE ×10^6 | Fitmean MSE ×10^6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | interval | 370.950327 | 0.987889 | -0.844733 | -0.560283 | 367.679703 | 368.257564 |
| Technical29 ordinary | evaluation | 324.546764 | -0.042434 | 0.876251 | -0.789124 | 315.467427 | 316.211081 |
| Technical29 magnitude | interval | 370.927185 | -0.419701 | -0.867876 | -0.583426 | 367.679703 | 368.257564 |
| Technical29 magnitude | evaluation | 325.411909 | 1.115696 | 1.741396 | 0.076021 | 315.467427 | 316.211081 |
| Perp31 ordinary | interval | 370.321687 | 2.076623 | -0.089402 | 0.003360 | 367.679703 | 368.257564 |
| Perp31 ordinary | evaluation | 324.137531 | 1.984379 | 1.570701 | -0.162323 | 315.467427 | 316.211081 |
| Perp31 magnitude | interval | 369.474490 | -0.487743 | -0.936599 | -0.843837 | 367.679703 | 368.257564 |
| Perp31 magnitude | evaluation | 323.736443 | 1.597823 | 1.169614 | -0.563410 | 315.467427 | 316.211081 |

All four new mapped means have worse overall E MSE than their own original half means; all four have slightly better overall I MSE than those half means. Every registered all-stratum MSE gate versus zero, fit mean and the three named references fails in both segments. This mixed result cannot replace the failed economic or probability requirements.

## What regularization changed

Every one of the 32 new coefficient norms is smaller than its C1 counterpart. The table gives equal-fold mean norms and the minimum/maximum within-fold new/old norm ratios. These describe fitted state, not accuracy.

| Classifier | C1 norm mean | New norm mean | Min new/old % | Max new/old % | Smaller norm folds |
| --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | 1.108879 | 0.056275 | 3.710814 | 6.267881 | 8/8 |
| Technical29 magnitude | 1.163181 | 0.060398 | 3.918676 | 6.700122 | 8/8 |
| Perp31 ordinary | 1.163411 | 0.060192 | 3.936228 | 6.561006 | 8/8 |
| Perp31 magnitude | 1.255556 | 0.070703 | 4.179940 | 7.466936 | 8/8 |

The following counts/rates pool ALL mapped inference rows across folds. They are not equal-quarter means and are not restricted to future-label score support. E includes 12 unscored origins and I includes 14. A match to the fitted prior's direction is not a correct-label count.

| Classifier | Segment | Inference rows | Sign changes vs C1 | Sign changes % | C1 prior matches | C1 prior matches % | New prior matches | New prior matches % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 ordinary | interval | 2537 | 550 | 21.679149 | 1765 | 69.570359 | 2079 | 81.947182 |
| Technical29 ordinary | evaluation | 2586 | 519 | 20.069606 | 1984 | 76.720804 | 2261 | 87.432328 |
| Technical29 magnitude | interval | 2537 | 545 | 21.482065 | 1754 | 69.136776 | 2267 | 89.357509 |
| Technical29 magnitude | evaluation | 2586 | 529 | 20.456303 | 1947 | 75.290023 | 2408 | 93.116783 |
| Perp31 ordinary | interval | 2537 | 584 | 23.019314 | 1709 | 67.363027 | 2101 | 82.814348 |
| Perp31 ordinary | evaluation | 2586 | 588 | 22.737819 | 1903 | 73.588554 | 2283 | 88.283063 |
| Perp31 magnitude | interval | 2537 | 615 | 24.241230 | 1656 | 65.273946 | 2239 | 88.253843 |
| Perp31 magnitude | evaluation | 2586 | 611 | 23.627224 | 1876 | 72.544470 | 2433 | 94.083527 |

The E predictions become more similar to the matched prior's direction. New and old zero-logit counts on these I/E supports are zero. Shrinking coefficients and moving closer to a prior can reduce probability loss while worsening sign-based trading. The unchanged controller uses the sign of the logit, so a positive temperature rescaling alone would leave mapped means and orders unchanged; it was not fitted as another economic candidate.

## Numerical validation and independent audits

The full suite passed before real fitting: **733 tests OK**, 57.350 seconds (`uv run python -m unittest discover -s tests -v`). The final accepted models passed fixed scalar arithmetic checks without retries. The new 32-model maximum normalized gradient infinity norm was 2.9278447350594705e−8, below the fixed 1e−6 limit. Runtime/scalar objective and gradient differences were zero; the minimum recomputed Hessian eigenvalue was 0.23027817223723637. New scalar logits differed by at most 1.1102230246251565e−16 and probabilities by 1.6653345369377348e−16, within the fixed 1e−12 / 1e−14 limits.

The run log contains **384 RuntimeWarnings**, 128 each of divide-by-zero, overflow and invalid matrix multiplication, and no ConvergenceWarning. Their cause is unknown. They were preserved. Finite fitted states, stationarity and scalar prediction agreement verify accepted outputs; they do not explain the warnings.

Independent audits completed without refits:

- Input/source audit: all 2,840 inherited artifacts, saved fit inputs, feature columns and six original masks were bound before fitting.
- Model audit: all 32 new objectives/gradients/Hessians, exact old/new scaler equality and estimator equality except C; 32 old scalar predictors; 64 new prediction NPZ files; 64 mechanism records. There were 45,000 fit-model rows and 30,112 predict-model rows for each model family. A separately retained redundant audit rechecked the old 32 objectives without changing the sealed Stage18 audit.
- Accounting audit: all 648 new artifacts, 960 independent scalar cost accounts, 64 new own-state paths, 22,016 decisions and 64 full traces. AlphaEX, DDdelta, turnover, fees, borrow, targets, known-open NAV/exposure and utility differences were zero. Mean-exposure rounding differed by at most 2.220446049250313e−16.
- Score audit: all 224 return and 160 classification records, source controls, prediction mappings and 64 diagnostics. Maximum scalar return-score difference was 4e−18 and classification-score difference 1e−16. Decimal-60 summary replay found maximum economic-mean difference 3.5e−15 and paired difference 1.0775e−15; every registered flag agrees.

Ten inherited B&H/scale-mean DDdelta entries contain ±1.1102230246251565e−16 arithmetic residuals. No new policy has such a tiny nonzero DDdelta. Strict signs are not changed after inspection.

## Interpretation and next research boundary

This single strong-penalty schedule reduces overall probability losses relative to C1, but does not satisfy the requested combination of predictive improvement and AlphaEX>0 / MaxDDdelta<0 across trends. It also makes all overall economic comparisons worse than the frozen learned parents. The schedule is rejected; no intermediate C, threshold, subgroup or model is selected from this result.

The earlier sign Oracle demonstrated useful hindsight direction information under a fixed controller; it did not demonstrate that the causal feature set can learn that information. This stage narrows one explanation: this fixed stronger regularization did not recover usable directional forecasts. Feature information, target definition and the target-to-action mapping remain research questions. No Stage19 procedure is registered by this report.

The hypothesis follows observed Stage17 failures, and development quarters have been repeatedly reused. Neither the 2/4/2 regime inventory nor the number of explored names supplies independent replications. No bootstrap, iid confidence interval, p-value or high-probability guarantee is claimed. Additional-test 15–24 was not used for this stage; formal P1 results_observed=false remains unchanged. UM raw bytes were hash-verified without new feature construction. The inherited Spot loader decodes the full Parquet file before the strict semantic cutoff; historical event timestamps still do not prove production receipt time.

The [frozen protocol](oracle_regularized_direction_registration_20260906.md) and [primary research note](oracle_regularized_direction_research_20260906.md) give the full assumptions. The [sklearn 1.8 logistic objective](https://scikit-learn.org/1.8/modules/linear_model.html#logistic-regression) defines the penalty arithmetic. [Bousquet–Elisseeff](https://www.jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf) motivates stability questions under its assumptions; its theorem does not establish trend-independent BTC accuracy or drawdown control here.

## Reproducibility evidence

The [evidence directory](oracle_regularized_direction_evidence_20260906/) contains the preflight, runtime registration/results/log, eight fold manifests, independent audit scripts/results, mechanism summary, full test log and publication verification. Large joblib/NPZ/trace artifacts remain in the ignored local output directory and are bound by the manifests; they are not all included in Git. Copied scripts retain their original local output and /tmp dependencies, whose hashes and copied evidence are preserved; a fresh checkout alone is not a complete raw-data/runtime archive.

- Results file SHA256: `ab2b62a017a70fd65756614f3940c3194d60b8c0ebdab02aa058c988ccbdb678`.
- Registration file SHA256: `289968f71450300c65f515dfe3ed2f2eae50c7f4fc5442f8669d1867faf65888`.
- Preflight file SHA256: `33b2ffc8d74a57b0827e35ea9c08c2e37eefdb421d9f6e726b566f0955e11285`.
- Runtime log SHA256: `d020d669322c5ca63ef4429d0071bfb9f3690ac9270747207914c89f337e224d`.
- Full test log SHA256: `3e8c7d39946ab7a00d9c9ba66a6f9c0744d27f64fe2474b22535441302ff0fc8`.

Registration canonical digest (distinct from file SHA256): `e870ef7d8fe96868b68af48159992e208a37275ceddc0785a072582196320412`.
