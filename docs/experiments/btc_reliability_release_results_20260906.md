# BTC Perp reliability production bundle — 2026-09-06

The fixed selected recipe `perp_delay0_reliability_utility_risk1` was fitted and exported once for production calendar25. Three base models and one scale-only reliability weight completed successfully. No evaluation-period performance, policy backtest, replacement candidate or revised selection was computed. The resulting bundle also passed local HF model/feature and synthetic accounting checks; this report does not claim a live deployment or new investment performance.

The [selection record](btc_accuracy_selection_20260906.json) chose the recipe using historical development evidence and its stated finite comparison/ranking rule. The newly fitted weights below have no measured evaluation performance. Historical validation means remain recipe evidence; they must not be presented as the performance of this production snapshot. Learned RL remains unqualified and high-probability generalization remains unestablished.

## Frozen execution and inputs

[Registration](btc_reliability_release_registration_20260906.md), source/config/tests, input preflight and selection evidence were committed and pushed in `a8d2ec176b6361ea50fda2ee5d6e2b695262ca88` before the sole production invocation. The process was session92067, terminal exit0, with no refit or retry. The [runtime evidence](btc_reliability_release_evidence_20260906/runtime_evidence.json) records the exact command and copied log hashes.

| Segment | Calendar boundaries, UTC13:45 | Mature selected origins |
|---|---|---:|
| T fit | 2024-07-16 → 2026-01-16 | 2,167 |
| S bias/risk/reliability calibration | 2026-01-16 → 2026-04-16 | 359 |
| I inherited interval provenance | 2026-04-16 → 2026-07-16 | 363 |
| E nominal production quarter | 2026-07-16 → 2026-10-16 | 0 loaded or scored |

All selected origins are six-hour UTC timestamps. Label maturity is origin+375minutes and strictly precedes the applicable segment boundary; last selected label maturities are January16, April16 and July16 at12:15 UTC. Selected prediction support contains724 S/I-region rows, including two segment-tail origins with masked outcomes. These tails are useful for inference parity and are excluded from calibration. The strict semantic input cutoff is2026-07-16T13:45Z; the final bar in the feature grid is13:30Z. The inherited Spot loader still decodes the full bound Parquet before applying that cutoff.

The complete original availability intersection is retained: original flow, all four derivative groups, Technical and Perp extra delays0/1/4, and trailing variances. Ten component masks remain bound even though the models consume only Technical29 and Perp31. There is no missing-value imputation or support reduction. Input preflight binds23 executable sources and178 raw-data/provenance artifacts, including the original Spot SHA `5e20e81e86f76b95d1301be7a8a366aa9ad78134f954ec8c9dbf83c0db1acf69` and UM SHA `02d2f07679db0087904b923606a501c8494afda0f75c6a4c94bf4b38ad49a583`. Archive hashes establish the bound files, not historical receipt timing.

## Produced bundle

The canonical bundle directory is `codex_outputs/btc_reliability_release_v1/bundle`, ID `btc-perp-reliability-20260906`. It contains six payload files: Technical29 Ridge100, Perp31 Ridge100, Technical29 fixed HGB100 joblibs, exact calibration/reliability JSON and a724-row prediction fixture. The seventh file is the manifest. Full selected T/predict feature matrices and masked calibration outcomes remain outside the bundle for audit.

The S-only coefficient is **0.22369215211526283**, an interior coefficient. Its fixed formula blends the bias-corrected Perp return prediction with the saved S return mean, which is −0.0006695958978521661. Technical scaled variance is unchanged. This coefficient is a calibration parameter, not a confidence probability or an observed future success rate. No I outcome or E value entered its calculation.

| Immutable binding | SHA256 |
|---|---|
| Bundle manifest | `c62287a3f21ad2407b61382648d4c6ceda718cb7bf1c1d5bcc99926dc225abe6` |
| Completion record | `2b0cfc8d7a85b39d7e35c2437632de423db0dd09484d0050b41a4e363db9dcf9` |
| Config | `5ecbfd4ed7d70b2b92b018b7e2ff3db101c62e28921f776d8c6b673b32f5eaca` |
| Input preflight | `775b1085c7c1b8e3ab5f6ac172f2a6bbcaf9c8ffda927dfe12bf56ad9989c1ec` |
| Independent audit | `16311220e1b0ec8b77adedc9dbfee99f00cface046720c960695a5e2c0c39522` |

Manifest source hashes, semantic feature/execution digests and vendor/runtime implementation hashes have distinct meanings. HF agrees to verify the export's exact semantic dictionaries. Execution remains BTC15m, UTC6h decisions, next15m-open fill, one-way cost0.00055, annual borrow0.10, step0.08, deadband0.01, risk1/costbudget2, missing forecast hold, and normalized initial cash0/equity1/units1/initial_open. Natural exposure is allowed to drift beyond the intent bounds0.5–1.12. [Shared release contract](btc_accuracy_release_contract_20260906.md) defines receipt deadlines and live accounting separately.

## Verification

Before fitting, all **787 tests passed in60.378seconds**, new-file whitespace and `git diff --check` passed, and the existing v31 checkpoint/bundle smoke passed on8,641 stored fixture rows. Its injected and default position paths differed by at most1.192e−7; auxiliary-state difference was0. These are legacy compatibility checks, not new performance measurements. The [prefit review](btc_reliability_release_evidence_20260906/prefit_review.json) preserves these results.

The [independent post-run audit](btc_reliability_release_evidence_20260906/independent_audit.json) imports no fitter, planner, strategy metric or reliability helper. It verifies216 distinct source/artifact paths through247 binding checks, all11 completion artifacts and all724 fixture rows. Selected matrix/outcome hashes match the committed preflight. Saved model parameters are read but never fitted again.

| Independent check | Maximum absolute difference |
|---|---:|
| Scalar standardized Ridge arithmetic, Technical | 3.469446951953614e−18 |
| Scalar standardized Ridge arithmetic, Perp | 3.469446951953614e−18 |
| HGB direct tree traversal | 0 |
| Reloaded joblib predictions, all3 models | 0 |
| Fixed reliability formula and saved S moments | 0 |
| Calibrated variance formula | 0 |

Independent scalar averaging differs from the saved vector arithmetic by at most1.1368683772161603e−13 for scaler means,2.168404344971009e−19 for bias and2.220446049250313e−16 for the variance multiplier. All are within the fixed audit tolerances. Full source feature formulas were not reimplemented independently here; their selected matrices are verified against the frozen input preflight and earlier feature tests.

The actual production log retains18 NumPy matrix-multiplication RuntimeWarnings: six each for divide-by-zero, overflow and invalid-value warnings. All claimed outputs are finite; scalar Ridge, independent HGB traversal and serialized prediction checks pass. The warnings are disclosed rather than treated as evidence of failed predictions or silently ignored. HF/runtime verification must still establish receiving-environment and stateful accounting parity before live rollout. No historical recipe evidence establishes a probability guarantee across future trends.

## Receiving HF runtime checks

The HF integration owner independently loaded the unchanged bundle and verified all724 saved input matrices through final mu/variance with maximum difference0. The copied [local verification](btc_reliability_release_evidence_20260906/hf_local_verification.json) also reports200 synthetic timely accounting bars, six fills and positive borrowing: targets and canonical cash/unit/account values match exactly. This is a synthetic integration check, not a new economic experiment.

One archived raw-input fixture at2026-07-16T12:00Z contains8,641 completed bars per Spot/UM source. The [raw feature proof](btc_reliability_release_evidence_20260906/hf_raw_feature_proof.json) binds its raw sources, HF implementation hashes and original prediction fixture. The maximum29/31 feature difference is5.7985172219332526e−11, within fixed absolute tolerance1e−9; the bounded prefix and original full-history rolling arithmetic differ by rounding. Technical/Perp raw-mean differences are2.1362902766219882e−14 and2.1958997120652413e−14; final mu differs by4.912032152554202e−15, with log variance and calibrated variance exactly equal.

This confirms the receiving local parser, selected common support, models and deterministic accounting on the stated fixtures. It does not establish current external data-source availability, actual historical receipt times, live service health or future performance. The bundle semantic feature hash is `16147125dc7b23f9bd34e0f0871bbf3ace62a42d0dbeab751f9f57d97d6165b4`; execution hash is `5ddef41673e466f9dc4ecf0edc187e91765ec09f98f9e17bba016940b42b6dc9`. Both match the HF proof.
