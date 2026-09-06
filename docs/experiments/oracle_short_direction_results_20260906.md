# Stage20 results: fixed short8 features for direction prediction

2026-09-06. Development comparison only. No high-probability generalization is established and no candidate is promoted. The formal P1 state remains `results_observed=false`.

[Frozen protocol](oracle_short_direction_registration_20260906.md) · [Primary-source rationale](oracle_short_direction_research_20260906.md) · [Machine-readable results](oracle_short_direction_evidence_20260906/results.json)

The new hold policy meets the overall historical minimum: base AlphaEX **+0.620736pt**, MaxDDdelta **−0.264463pt**; stress **+0.025604pt / −0.144629pt**. Its regime-wide and proper-score gates fail, and only2/8 quarters meet both signs under both costs. The fallback policy fails the overall minimum. This closes the feature search without displacing the stronger existing ML compromise candidates.

## Registered comparison and evidence boundaries

Added the complete existing short8 block to Technical29, producing Technical37. Exactly16 ordinary/magnitude C1 logistic fits were run. Only magnitude probability enters the unchanged `a_T*(2*q-1)` mean mapping and the two hold/fallback policies. No old models, unique priors, risk models or calibrations were fitted. All80 old policies,24 means and10 classification streams were preserved.

Both tasks use the same original T rows, labels and weighting arithmetic. The29 inherited feature columns and37 selected fit/predict matrices were checked against independent Stage17/Stage15 records. Added features were not imputed and original supports were not narrowed. The comparison measures the whole regularized procedure; added correlated columns also change effective L2 geometry.

Eight reused development quarters have2 bull,4 bear and2 sideways regimes. Original test(f) aliases validation(f+1). I strata use the later E-start regime retrospectively. Additional-test15–24 was not modeled or scored. Thus the inequalities below are descriptive comparisons, not a confidence level or all-trend guarantee. The bound Spot Parquet is decoded before strict semantic cutoff; archive event time is not receipt-time proof.

## New policy economics

AlphaEX and MaxDDdelta are percentage points, equal-weighted across quarters, relative to B&H. Positive AlphaEX and negative MaxDDdelta are desired. Stress doubles costs on the same targets. Joint counts require both signs under both costs in the same quarter.

| Stratum | Rule | Quarters | Base AlphaEX | Base DDdelta | Stress AlphaEX | Stress DDdelta | Joint quarters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | utility_risk1 | 8 | 0.620736 | -0.264463 | 0.025604 | -0.144629 | 2 |
| all | utility_risk1_fallback_bh | 8 | -1.251253 | 0.687566 | -1.812105 | 0.801585 | 2 |
| bull | utility_risk1 | 2 | 3.492305 | -3.681569 | 3.194755 | -3.526737 | 1 |
| bull | utility_risk1_fallback_bh | 2 | -1.430112 | 0.677770 | -1.699430 | 0.812473 | 1 |
| bear | utility_risk1 | 4 | 2.744544 | -0.339596 | 2.019257 | -0.246361 | 1 |
| bear | utility_risk1_fallback_bh | 4 | 1.497790 | -0.636211 | 0.820276 | -0.546180 | 1 |
| sideways | utility_risk1 | 2 | -6.498447 | 3.302909 | -7.130856 | 3.440942 | 0 |
| sideways | utility_risk1_fallback_bh | 2 | -6.570480 | 3.344915 | -7.189542 | 3.486227 | 0 |

All per-quarter new outcomes:

| Fold | Trend | Rule | Base AlphaEX | Base DDdelta | Stress AlphaEX | Stress DDdelta |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | bull | utility_risk1 | 12.359076 | -10.390394 | 12.137920 | -10.260624 |
| 5 | bull | utility_risk1_fallback_bh | 2.459776 | -1.625590 | 2.301680 | -1.541563 |
| 6 | bear | utility_risk1 | 14.508718 | -3.695743 | 13.538297 | -3.604578 |
| 6 | bear | utility_risk1_fallback_bh | 9.521702 | -4.882200 | 8.742371 | -4.803852 |
| 7 | bull | utility_risk1 | -5.374467 | 3.027256 | -5.748409 | 3.207149 |
| 7 | bull | utility_risk1_fallback_bh | -5.320000 | 2.981130 | -5.700539 | 3.166508 |
| 8 | bear | utility_risk1 | -1.642896 | 0.234847 | -2.195608 | 0.276641 |
| 8 | bear | utility_risk1_fallback_bh | -1.642896 | 0.234847 | -2.195608 | 0.276641 |
| 9 | sideways | utility_risk1 | -4.909424 | 4.847152 | -5.152949 | 4.977595 |
| 9 | sideways | utility_risk1_fallback_bh | -5.016536 | 4.931165 | -5.268229 | 5.068164 |
| 10 | bear | utility_risk1 | 0.000891 | 2.962664 | -0.493977 | 3.165021 |
| 10 | bear | utility_risk1_fallback_bh | 0.000891 | 2.962664 | -0.493977 | 3.165021 |
| 11 | bear | utility_risk1 | -1.888537 | -0.860153 | -2.771683 | -0.822529 |
| 11 | bear | utility_risk1_fallback_bh | -1.888537 | -0.860153 | -2.771683 | -0.822529 |
| 12 | sideways | utility_risk1 | -8.087470 | 1.758665 | -9.108762 | 1.904290 |
| 12 | sideways | utility_risk1_fallback_bh | -8.124424 | 1.758665 | -9.110854 | 1.904290 |

## Probability and mapped-return evidence

Ordinary losses assess ordinary direction probability; magnitude-weighted losses assess a different, absolute-return-weighted target. Neither is a guarantee of conditional-mean accuracy. This section reports both tasks even if only one improves.

| Stratum | Segment | Task | Brier Δ vs29 | Logloss Δ vs29 | Brier Δ vsprior | Logloss Δ vsprior |
| --- | --- | --- | --- | --- | --- | --- |
| all | interval | ordinary | 0.001269 | 0.003736 | 0.016875 | 0.044116 |
| all | interval | magnitude | 0.000612 | 0.000460 | 0.028916 | 0.082343 |
| all | evaluation | ordinary | 0.000303 | 0.001869 | 0.017300 | 0.044108 |
| all | evaluation | magnitude | -0.001449 | -0.005532 | 0.029706 | 0.075232 |
| bull | interval | ordinary | 0.001950 | 0.005974 | 0.021936 | 0.063825 |
| bull | interval | magnitude | -0.000945 | -0.003795 | 0.045260 | 0.153580 |
| bull | evaluation | ordinary | 0.002269 | 0.008871 | 0.039020 | 0.107028 |
| bull | evaluation | magnitude | -0.004486 | -0.017969 | 0.059143 | 0.161204 |
| bear | interval | ordinary | 0.000484 | 0.002062 | 0.016867 | 0.043088 |
| bear | interval | magnitude | -0.002269 | -0.006179 | 0.024484 | 0.064177 |
| bear | evaluation | ordinary | 0.000220 | 0.000928 | 0.011681 | 0.026567 |
| bear | evaluation | magnitude | -0.000628 | -0.002660 | 0.017895 | 0.040266 |
| sideways | interval | ordinary | 0.002158 | 0.004847 | 0.011830 | 0.026462 |
| sideways | interval | magnitude | 0.007932 | 0.017992 | 0.021435 | 0.047438 |
| sideways | evaluation | ordinary | -0.001496 | -0.003248 | 0.006818 | 0.016272 |
| sideways | evaluation | magnitude | -0.000053 | 0.001161 | 0.023890 | 0.059194 |

Above, negative differences are improvement. The full JSON retains both weighted and ordinary scoring families for every classifier, including zeros/ties and denominator records.

Equal-quarter mapped MSE ×1,000,000:

| Stratum | Segment | technical_short_both_magnitude_soft | technical_magnitude_soft | technical_magnitude_direction | technical_half | technical_soft_mapped_prior | technical_soft_fit_mean | technical_soft_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | interval | 379.662229 | 379.360177 | 371.346886 | 371.795061 | 368.257564 | 368.257564 | 367.679703 |
| all | evaluation | 329.443651 | 329.619158 | 324.296213 | 323.670513 | 316.211081 | 316.211081 | 315.467427 |
| bull | interval | 515.350541 | 516.195492 | 508.359045 | 507.228310 | 500.684521 | 500.684521 | 502.237477 |
| bull | evaluation | 440.285355 | 441.615994 | 439.539380 | 439.473290 | 417.548103 | 417.548103 | 415.765663 |
| bear | interval | 383.898742 | 384.919429 | 377.509335 | 378.267135 | 374.752439 | 374.752439 | 373.087490 |
| bear | evaluation | 280.217071 | 280.584018 | 276.698071 | 276.069720 | 272.264877 | 272.264877 | 272.706656 |
| sideways | interval | 235.500890 | 231.406360 | 222.009830 | 223.417664 | 222.840857 | 222.840857 | 222.306357 |
| sideways | evaluation | 317.055109 | 315.692602 | 304.249332 | 303.069323 | 302.766468 | 302.766468 | 300.690733 |

## Independent descriptive gates

| Task | I matched proper-loss gate | E matched proper-loss gate |
| --- | --- | --- |
| technical_short_both_ordinary | False | False |
| technical_short_both_magnitude | False | False |

Both-task conjunction: `false`.

| Rule | Absolute economics | All6 paired economics | I mapped MSE | E mapped MSE | I source magnitude losses | E source magnitude losses | High-probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| utility_risk1 | False | False | False | False | False | False | False |
| utility_risk1_fallback_bh | False | False | False | False | False | False | False |

These strict conditions cover every all/bull/bear/sideways stratum. A failure means the registered procedure did not demonstrate that improvement on the reused periods; it does not prove absence of information. Ordinary remains diagnostic for the policy. High-probability confirmation stays false regardless of the descriptive outcomes.

## Verification and provenance

Run inventory:656 economic rows/1312 accounts,400 return-score records,192 classification-score records,32 direction diagnostics,16 mapping diagnostics,95 artifacts/fold. All640 old economic records,384 old return records,160 old classification records and the complete Stage19 summary were checked exactly. The adaptive causal-name ledger is220.

Original E inference2586/score2574 and I mapped inference2537/score2523 remain distinct; unscored origins still receive predictions. Missing forecasts and missing opens preserve332 fallback rows and2 gaps. Zero-mean controls retain the same risk controller and are not B&H.

Freeze revision: `69d2cd6bae732d9598135c20fe216a4fa9b48fa1`.

| Binding | SHA256 |
| --- | --- |
| config | 3eb2c811e9d96dd5e3e2d51483694e1b4d6dd412cd61dda50fe1d6d74df9520f |
| protocol | e3ebf7ac72fd6a192701b8ec1cf034393daa11c0ea36cab767a2c76be48216fb |
| research | b66637ef31fdd9f2c42a4f84202a7086a75556f9a93d6d449ddc8c91eb48e9bc |
| registration | ea4c54c94c00523eea189ae582bfb84e358b017285b0a779edc0dde83fe5676a |
| preflight | 1a48c5d966c32439b0e3efc837210a88cf60f7cd96b194b2378fad1a5a62c4b6 |
| results | e7c116ea5aae663ec42276d96c627813aad922b9f8796bdded8c9bb045ff9cd9 |

[Evidence directory](oracle_short_direction_evidence_20260906/) contains runtime manifests/logs and independent audit sources/results. Large binaries remain local hash-bound. Independent audit details and numerical-warning counts are recorded in `publication_verification.json`.

## Interpretation

This experiment closes the fixed Technical37 task comparison. Any subsequent hypothesis must receive a separate pre-outcome registration. No threshold, feature subset, C or model structure was changed after these outcomes.
