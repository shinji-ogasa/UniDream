# Stage19: continuous probability-to-mean mapping results

**No new strongest model is established; all eight learned mapping policies are rejected.** The Perp31 C1 mappings have positive overall AlphaEX but positive MaxDDdelta at both costs. The remaining learned mappings fail the overall joint sign conditions as well. Every registered absolute economic, five-reference paired economic and I/E mapped-MSE gate fails.

The L2unit continuous means lower overall MSE versus their hard-sign means and original half means, but still lose to zero, fitmean and mapped-prior predictions in both I and E. C1 continuous means worsen MSE versus all five references overall in both segments. Probabilities and all 160 classification records are unchanged; this experiment cannot claim improved probability accuracy. No model was selected, promoted or deployed.

## Frozen execution and scope

The protocol, code, tests, configuration, primary research and input-only preflight were committed and pushed at `9b4f6a0e5606831a26a8f2a7c401e05c52d41f6f` before new real mapped means, residuals, scores or orders. The run completed once (PTY 91810, exit 0), with eight fold-complete records and no retries. There were zero model fits, new probability predictions, new priors, calibration fits or feature reconstructions. Existing T statistics and saved probability arrays were inputs, not re-estimated parameters.

Four saved magnitude-weighted streams (Technical29 / Perp31 × C1 / L2unit) use exactly `mu = saved_T_mean_abs_return * (2.0*saved_probability - 1.0)`. The scale is a fixed approximation to conditional mean absolute return. Neither it nor the saved regularized probability is established as the true conditional quantity. The original hard map used `sign(logit)*abs(own_half_mu)`. Continuous mapping changes the mean magnitude passed to the controller; it is not a newly fitted direction model.

The mapped-prior controls use the saved Stage17 prior NPZ probabilities, which can differ by rounding from raw fit_priors['magnitude']. Exact saved fitmean and zero remain separate controls. No new sigmoid, tanh, multiplier choice, epsilon, threshold, clipping or sign repair is used. A zero mean is valid input. All new mapped S means remain NaN; raw S probability/logit fields remain source evidence and are not newly scored or calibrated.

All 60 old policies (36 causal, 24 hindsight) remain. Four learned means and six constant means give 20 new causal names: eight learned policy names and twelve constant-control names. The adaptive count is 198→218. All 80 policies produce 640 rows / 1,280 cost accounts; 24 means produce 384 I/E return records; ten inherited classifiers retain 160 records; four learned means have 64 I/E mapping diagnostics. The run emits 968 new artifacts. Duplicate constants are retained, not independent replications.

Original validation 5–12 spans April 2021–April 2023 and aliases old test 4–11. T18/S3/I3/E3, the original six masks and strict semantic cutoff before 2023-04-16T13:45Z are unchanged. E inference/score counts are 2,586/2,574; mapped I counts are 2,537/2,523. All 12 unscored E origins remain eligible for causal orders. The 332 fallback and 2 missing-current-open opportunities remain. Regime counts are 2 bull / 4 bear / 2 sideways; I groups use retrospective E-start regimes.

Execution keeps UTC 6-hour decisions, own B&H-initialized cash/units, immediately-next-bar-open fills, fee 0.00055 one-way, annual borrow 0.10, risk aversion 1, cost allowance 2, intents [0.5,1.12], step 0.08 and deadband 0.01. Passive exposure can exceed intent bounds. Missing predictions produce hold or target-1 fallback with known current open; missing current open prevents orders, and a missing immediately-next open skips without rollover. Borrow continues across gaps. Risk is the same bound saved variance. Future labels never gate a mapping or order.

## Overall economics: all learned mappings

Values are equal-quarter **percentage points relative to B&H**, not annualized. Negative DDdelta is better. Stress replays base targets with twice fees and borrowing. Joint counts require strict AlphaEX>0 and DDdelta<0 at both costs in the same quarter. No learned mapping meets that condition in every quarter or the all-stratum mean requirements.

| Source | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- |
| Technical29 C1 | hold | -0.029076 | -0.006773 | -0.554044 | 0.097431 | 3/8 |
| Technical29 C1 | fallback | -1.420121 | 0.555413 | -1.915444 | 0.654835 | 3/8 |
| Technical29 L2unit | hold | -2.579935 | 1.282657 | -2.819191 | 1.374963 | 0/8 |
| Technical29 L2unit | fallback | -3.222366 | 0.728138 | -3.420712 | 0.799194 | 0/8 |
| Perp31 C1 | hold | 1.992860 | 0.142723 | 1.446507 | 0.253201 | 3/8 |
| Perp31 C1 | fallback | 0.865579 | 0.482688 | 0.345029 | 0.588150 | 3/8 |
| Perp31 L2unit | hold | -1.987078 | 1.162060 | -2.224066 | 1.249477 | 0/8 |
| Perp31 L2unit | fallback | -2.864757 | 0.827745 | -3.067252 | 0.898938 | 0/8 |

## All new constant controls

These use the same masks, risk and own-state execution. Zero mean does not mean B&H: the controller's risk, cost and inventory terms can still trigger trades. Tiny nonzero DDdelta values below are displayed in scientific notation rather than silently rounded into exact equality.

| Group | Control | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| technical | mapped_prior | hold | -0.856907 | 1.388e-15 | -0.859185 | 1.388e-15 | 0/8 |
| technical | mapped_prior | fallback | -0.547781 | 1.388e-15 | -0.552186 | 1.388e-15 | 0/8 |
| technical | fit_mean | hold | -0.856907 | 1.388e-15 | -0.859185 | 1.388e-15 | 0/8 |
| technical | fit_mean | fallback | -0.547781 | 1.388e-15 | -0.552186 | 1.388e-15 | 0/8 |
| technical | zero | hold | -1.322167 | -1.446962 | -1.327465 | -1.444217 | 3/8 |
| technical | zero | fallback | -1.084714 | -0.865591 | -1.095641 | -0.861461 | 2/8 |
| perp_delay0 | mapped_prior | hold | -0.856907 | 1.388e-15 | -0.859185 | 1.388e-15 | 0/8 |
| perp_delay0 | mapped_prior | fallback | -0.547781 | 1.388e-15 | -0.552186 | 1.388e-15 | 0/8 |
| perp_delay0 | fit_mean | hold | -0.856907 | 1.388e-15 | -0.859185 | 1.388e-15 | 0/8 |
| perp_delay0 | fit_mean | fallback | -0.547781 | 1.388e-15 | -0.552186 | 1.388e-15 | 0/8 |
| perp_delay0 | zero | hold | -1.322167 | -1.446962 | -1.327465 | -1.444217 | 3/8 |
| perp_delay0 | zero | fallback | -1.084714 | -0.865591 | -1.095641 | -0.861461 | 2/8 |

All constant controls have negative overall AlphaEX. Their equal results across feature groups do not supply extra evidence. The mapped-prior and fitmean constants remain distinct numeric inputs even when their resulting target paths coincide.

## Learned mappings by regime

Regimes use trailing information at E start, not the realized quarter return. No 2/4/2 reused-quarter mean establishes likely trend-independent performance.

| Source | Regime | Rule | Base AlphaEX pt | Base DDdelta pt | Stress AlphaEX pt | Stress DDdelta pt | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 C1 | bull | hold | 1.960503 | -1.146022 | 1.722610 | -1.023328 | 1/2 |
| Technical29 C1 | bull | fallback | -1.141756 | 1.680922 | -1.367866 | 1.782994 | 1/2 |
| Technical29 C1 | bear | hold | 2.983927 | -0.847422 | 2.318513 | -0.755800 | 2/4 |
| Technical29 C1 | bear | fallback | 1.792351 | -1.152705 | 1.179923 | -1.066608 | 2/4 |
| Technical29 C1 | sideways | hold | -8.044660 | 2.813777 | -8.575811 | 2.924654 | 0/2 |
| Technical29 C1 | sideways | fallback | -8.123430 | 2.846139 | -8.653757 | 2.969560 | 0/2 |
| Technical29 L2unit | bull | hold | -2.905530 | 3.100154 | -3.085349 | 3.229622 | 0/2 |
| Technical29 L2unit | bull | fallback | -2.071730 | 2.280342 | -2.218258 | 2.377856 | 0/2 |
| Technical29 L2unit | bear | hold | 3.402301 | 1.967629 | 3.102309 | 2.038923 | 0/4 |
| Technical29 L2unit | bear | fallback | 0.988073 | 1.730436 | 0.740753 | 1.797680 | 0/4 |
| Technical29 L2unit | sideways | hold | -14.218812 | -1.904783 | -14.396034 | -1.807617 | 0/2 |
| Technical29 L2unit | sideways | fallback | -12.793879 | -2.828665 | -12.946097 | -2.776441 | 0/2 |
| Perp31 C1 | bull | hold | 1.638520 | -2.383428 | 1.374286 | -2.237400 | 1/2 |
| Perp31 C1 | bull | fallback | -0.589512 | -0.333950 | -0.849520 | -0.202915 | 1/2 |
| Perp31 C1 | bear | hold | 3.721068 | -0.416500 | 3.048982 | -0.322296 | 2/4 |
| Perp31 C1 | bear | fallback | 2.585669 | -0.714295 | 1.959878 | -0.623626 | 2/4 |
| Perp31 C1 | sideways | hold | -1.109217 | 3.787320 | -1.686224 | 3.894795 | 0/2 |
| Perp31 C1 | sideways | fallback | -1.119512 | 3.693291 | -1.690121 | 3.802766 | 0/2 |
| Perp31 L2unit | bull | hold | -1.712085 | 2.086928 | -1.854684 | 2.185870 | 0/2 |
| Perp31 L2unit | bull | fallback | -1.495012 | 1.855677 | -1.622970 | 1.939383 | 0/2 |
| Perp31 L2unit | bear | hold | 3.040686 | 2.272936 | 2.725413 | 2.350031 | 0/4 |
| Perp31 L2unit | bear | fallback | 0.662263 | 2.035743 | 0.399535 | 2.108788 | 0/4 |
| Perp31 L2unit | sideways | hold | -12.317598 | -1.984560 | -12.492407 | -1.888023 | 0/2 |
| Perp31 L2unit | sideways | fallback | -11.288542 | -2.616184 | -11.445111 | -2.561208 | 0/2 |

## Paired overall economic changes

These are new minus the same-classifier hard map or the original half mean, at the same rule/cost. Positive Alpha change and negative DD change would be improvement. All five-reference contrasts, including mapped prior, fitmean, zero and turnover/trade changes, are retained in the full JSON. Mixed individual changes do not pass the registered all-five, all-stratum requirement.

| Source | Reference | Rule | Base Alpha change pt | Base DD change pt | Stress Alpha change pt | Stress DD change pt |
| --- | --- | --- | --- | --- | --- | --- |
| Technical29 C1 | same-classifier hard | hold | 1.130440 | 2.590756 | 1.062692 | 2.612102 |
| Technical29 C1 | same-classifier hard | fallback | 0.533266 | 3.083560 | 0.477953 | 3.104678 |
| Technical29 C1 | own original half | hold | -1.178034 | 6.024601 | -1.497315 | 6.050570 |
| Technical29 C1 | own original half | fallback | -2.426639 | 7.338276 | -2.736710 | 7.373228 |
| Technical29 L2unit | same-classifier hard | hold | 0.285027 | -0.097857 | 0.414002 | -0.125399 |
| Technical29 L2unit | same-classifier hard | fallback | -0.020705 | 0.007924 | 0.112936 | -0.011838 |
| Technical29 L2unit | own original half | hold | -3.728893 | 7.314031 | -3.762462 | 7.328101 |
| Technical29 L2unit | own original half | fallback | -4.228884 | 7.511001 | -4.241978 | 7.517587 |
| Perp31 C1 | same-classifier hard | hold | -1.342603 | 3.255853 | -1.365347 | 3.260616 |
| Perp31 C1 | same-classifier hard | fallback | -0.822108 | 3.011180 | -0.853246 | 3.018594 |
| Perp31 C1 | own original half | hold | -1.620082 | 5.530425 | -1.902472 | 5.557707 |
| Perp31 C1 | own original half | fallback | -2.513874 | 6.642489 | -2.798828 | 6.678360 |
| Perp31 L2unit | same-classifier hard | hold | 0.173204 | 0.083049 | 0.334454 | 0.063703 |
| Perp31 L2unit | same-classifier hard | fallback | -0.400805 | 0.441408 | -0.242781 | 0.435345 |
| Perp31 L2unit | own original half | hold | -5.600020 | 6.549762 | -5.573045 | 6.553984 |
| Perp31 L2unit | own original half | fallback | -6.244210 | 6.987546 | -6.211109 | 6.989148 |

## Mapped-return prediction versus all five references

MSE and MSE differences are shown ×10^6, using equal-quarter means. Differences are new minus reference; negative is better. Every reference uses the same future-label score support. Pooled-row MSE is saved separately and does not replace this aggregation.

| Source | Segment | MSE ×10^6 | Change vs hard ×10^6 | Change vs half ×10^6 | Change vs mapped prior ×10^6 | Change vs fitmean ×10^6 | Change vs zero ×10^6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technical29 C1 | interval | 379.360177 | 8.013291 | 7.565116 | 11.102613 | 11.102613 | 11.680474 |
| Technical29 C1 | evaluation | 329.619158 | 5.322944 | 5.948644 | 13.408077 | 13.408077 | 14.151731 |
| Technical29 L2unit | interval | 368.806429 | -2.120756 | -2.988631 | 0.548865 | 0.548865 | 1.126726 |
| Technical29 L2unit | evaluation | 317.375825 | -8.036085 | -6.294689 | 1.164743 | 1.164743 | 1.908398 |
| Perp31 C1 | interval | 377.385913 | 7.423680 | 6.974824 | 9.128348 | 9.128348 | 9.706209 |
| Perp31 C1 | evaluation | 328.720476 | 6.581857 | 6.153647 | 12.509395 | 12.509395 | 13.253050 |
| Perp31 L2unit | interval | 368.656365 | -0.818125 | -1.754723 | 0.398801 | 0.398801 | 0.976662 |
| Perp31 L2unit | evaluation | 317.187083 | -6.549360 | -5.379746 | 0.976002 | 0.976002 | 1.719656 |

For both L2unit sources, lower MSE than the learned hard/half references coexists with worse MSE than the simple controls. For both C1 sources, overall MSE worsens against every reference. Both L2unit maps improve E MSE versus their hard maps in all four strata and in 7/8 quarters each, but worsen E sideways MSE versus their own original halves. Against mapped prior and fitmean, their E MSE is worse in every stratum. All eight mean/segment all-five/all-stratum MSE conditions fail. None of these results establishes new usable mean information above the constant/zero baselines.

All saved probabilities/logits and the 160 ordinary/weighted classification records remain exact. The appropriate inherited weighted Brier/logloss versus magnitude prior gates remain false in both I and E for every source. New probability-accuracy improvement is false by identity. See the [Stage18 result](oracle_regularized_direction_results_20260906.md) and the complete unchanged classification section in the result JSON; downstream economics cannot reverse that probability evidence.

## Mapping scale and direction diagnostics

The following mean absolute magnitudes are equal-fold means ×10^3, computed on ALL inference rows. Counts pool inference rows and do not use future score support. E includes 12 unscored origins; I includes 14. A magnitude comparison is not a forecast-correctness count.

| Source | Segment | Inference rows | New mean abs ×10^3 | Hard mean abs ×10^3 | Half mean abs ×10^3 | New abs > hard rows |
| --- | --- | --- | --- | --- | --- | --- |
| Technical29 C1 | interval | 2537 | 3.070834 | 1.843056 | 1.843056 | 1635 |
| Technical29 C1 | evaluation | 2586 | 3.182341 | 2.321402 | 2.321402 | 1578 |
| Technical29 L2unit | interval | 2537 | 0.911539 | 1.843056 | 1.843056 | 766 |
| Technical29 L2unit | evaluation | 2586 | 1.010649 | 2.321402 | 2.321402 | 844 |
| Perp31 C1 | interval | 2537 | 2.963320 | 1.679950 | 1.679950 | 1660 |
| Perp31 C1 | evaluation | 2586 | 3.217433 | 2.212683 | 2.212683 | 1608 |
| Perp31 L2unit | interval | 2537 | 0.926104 | 1.679950 | 1.679950 | 813 |
| Perp31 L2unit | evaluation | 2586 | 1.031693 | 2.212683 | 2.212683 | 878 |

Every recorded q=.5/0/1, source-zero-logit, mapped-zero-mean and direction-disagreement count is zero across the 64 learned diagnostics. New magnitude equals hard magnitude on zero inference rows. Thus this observed comparison preserves each learned direction and changes its amplitude; C1 mean absolute amplitudes rise overall while L2unit amplitudes shrink overall. Those aggregate changes do not identify a causal source of accuracy or guarantee another scale would succeed.

The saved prior and fitmean arithmetic differences are retained below. a_T and fitmean are shown ×10^3; the two difference columns remain unscaled. No prior was re-estimated and no small difference was forced to zero.

| Fold | T mean abs ×10^3 | T mean ×10^3 | Saved prior q minus raw fit prior | Mapped prior minus exact fitmean |
| --- | --- | --- | --- | --- |
| 5 | 9.577183 | 0.389196 | 0.000e+00 | -7.047e-19 |
| 6 | 9.899023 | 0.904943 | 0.000e+00 | -1.084e-18 |
| 7 | 11.475838 | 0.880905 | 0.000e+00 | -3.253e-19 |
| 8 | 12.351280 | 0.561574 | 0.000e+00 | -4.337e-19 |
| 9 | 12.650868 | 0.692966 | 0.000e+00 | 5.421e-19 |
| 10 | 12.860043 | 0.435153 | 0.000e+00 | -4.337e-19 |
| 11 | 13.637871 | 0.360287 | 0.000e+00 | 9.216e-19 |
| 12 | 14.011730 | -0.395133 | 5.551e-17 | 1.301e-18 |

## Verification evidence

The full suite passed before real mapping: **752 tests OK**, 58.614 seconds, using `uv run python -m unittest discover -s tests -v`. The registered source/config remained unchanged during the single completed run. The run log contains eight fold-complete records and no warnings. This does not explain the earlier Stage17/18 matrix-multiplication warnings.

Independent mapping audit passed: all 33 sources, 3,488 inherited and 968 emitted artifacts were bound; all 160 new prediction NPZs, 160 helper diagnostics, 64 learned mechanism records, eight mapping-provenance records and 48 six-mask bindings were checked. Scalar mapping error was exactly zero. Source q/logits/risk/actual/support were exact, including signed-zero preservation. It checked 20,492 learned mapped rows and 10,246 rows for each constant kind. This is numerical and provenance evidence, not a generalization guarantee.

Independent accounting audit also passed: 1,280 scalar cost accounts, 160 new own-state paths (64 learned and 96 constant), 55,040 decisions and 160 traces were replayed. All AlphaEX, DDdelta, cost, turnover, trade, target, known-open NAV/exposure and utility differences were zero; mean-exposure rounding differed by at most 2.220446049250313e−16. The complete old 480 rows, including inherited trace hashes, and old 224 return / 160 classification records remain exact.

Independent scalar/Decimal-60 score audit checked all 384 return scores, 160 probability scores and every aggregate/paired gate. Maximum scalar differences were 5e−18 for return scores and 1e−16 for classification scores. Maximum economic-summary difference was 3.5e−15 and paired-summary difference 9e−16. The inherited records themselves are exactly unchanged, not merely within these independent-arithmetic tolerances.

The independent mechanism summary verifies all 80 fixed constant target comparisons and their base/stress saved metrics: 48 cross-group comparisons and 32 mapped-prior/fitmean comparisons. All eight mapped-prior means differ from exact fitmean; the largest absolute difference is 1.3010426069826053e−18. Saved prior q differs from the raw statistical prior only in fold 12, by +5.551115123125783e−17. These differences did not change the compared target streams. Of 32 zero-mean paths, 24 trade at both costs; base trade counts range from 0 to 8. Zero mean is therefore not automatically a passive B&H path.

There are 66 account DDdelta residuals of ±1.1102230246251565e−16: 10 inherited and 56 new constant-control entries. None belongs to the eight new learned policy names. They remain in the strict-sign ledger without post hoc snapping. The independent audit of formatted report cells and file hashes is included in the evidence directory.

## Research conclusion and limits

The proposed continuous mapping does not satisfy the user's joint AlphaEX/DDdelta and trend-robust improvement objective. It changes the economic paths, but no learned candidate passes the predeclared conditions. The L2unit MSE reduction versus weak learned references cannot be promoted when zero/constant controls remain better; the C1 Perp AlphaEX gain over B&H cannot substitute for its failed drawdown requirement. Keep all failures and duplicate controls in the ledger. No new scale, threshold, C or architecture is selected from these results.

These repeatedly reused development quarters are not independent confirmation. The all-trend probability and regime-count gates remain false. No bootstrap, iid confidence interval, p-value, deployment, live trading or P1 result is claimed. Additional-test 15–24 and outer data were not used here. The inherited Spot loader decodes the full Parquet before semantic slicing, and historical event timestamps are not proof of production receipt-time availability.

The [frozen protocol](oracle_soft_direction_registration_20260906.md) defines the decision rules; the [primary research and derivation](oracle_soft_direction_research_20260906.md) distinguishes the weighted population identity from the constant-amplitude approximation. This experiment tests the downstream mapping and does not make the unchanged probability model more accurate. No Stage20 procedure is registered by this report.

## Reproducibility

The [evidence directory](oracle_soft_direction_evidence_20260906/) retains input-only checks, runtime registration/results/log and eight manifests, independent audit scripts/results, the full test log and publication verification. Large NPZ/trace artifacts remain in ignored local output folders and are hash-bound by manifests, not all uploaded to Git. Copied audit scripts retain original local path dependencies; a fresh checkout is not a complete raw-data/runtime archive.

- Results file SHA256: `d85e14e1a8249601a1f28c0d4fa29b1fbb23a3571232dcedc5edee0644b82cc5`.
- Registration file SHA256: `dda43125f1f670fd84ae35809148561863018e8de5ea4c18b99a14fe454e2f22`.
- Preflight file SHA256: `be95da2abe1581b516cc8a0fd168ae919ad89527280818b283f46b92c95f2154`.
- Runtime log SHA256: `e9a42d0cff32a5f7de65aec59fbcbb9764d9730cd50fc15e8ab8c615d5733644`.
- Full test log SHA256: `9301f9e587a0fbcf1b61759c23dd9656dfc31b59d1ba4ebaa1f80c9effde766a`.

Registration canonical digest, distinct from the file hash: `92f7382c4165adca5480d5df4560d84e2e4559ea09a090fb7e77092470a9345d`.
