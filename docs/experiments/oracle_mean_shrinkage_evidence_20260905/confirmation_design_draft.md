# Frozen confirmation design after the fixed-half diagnostic

Status: design skeleton, 2026-09-05; not a registration, selected model, execution authorization, P1 result, or independent confirmation result. This document was prepared without reading test prices, test model outputs, later held-out results, or the new half-policy outcomes. Source/configuration text only was consulted. Rebind source hashes when a final protocol is committed.

## 1. Stop the current search family

Complete the already fixed half comparison once. Close the weight family at lambda = 0, 0.5, and 1; do not add a finer weight grid, new mean windows, architecture, masks, or policy thresholds in response to its results. The four new candidate IDs are exactly:

- technical_half_utility_risk1
- technical_half_utility_risk1_fallback_bh
- perp_delay0_half_utility_risk1
- perp_delay0_half_utility_risk1_fallback_bh

For each family, mu_half = scale_mean + 0.5 * (mu_scaled - scale_mean). Mean models are StandardScaler + Ridge(alpha=100); HGB is the log-variance model, not the mean model. Every candidate uses the same technical_scaled variance. Freeze both source families and both inventory rules; do not choose the attractive half result while omitting the other arms from the report.

Freeze the eight existing controls from the half configuration: bh, common_robust, scale_mean_utility_risk1, scale_mean_utility_risk1_fallback_bh, technical_scaled_utility_risk1, technical_scaled_utility_risk1_fallback_bh, perp_delay0_scaled_utility_risk1, perp_delay0_scaled_utility_risk1_fallback_bh. Controls do not become new selected candidates after test inspection. Hold and fallback versions share a forecast and are not two independent predictive discoveries.

## 2. Selection and immutable locks

Default confirmation unit: the entire four-candidate family plus all eight controls. Freeze this manifest before accessing additional outcomes. No winner is chosen in this document. Report all four results; do not declare success because any one unadjusted comparison is favorable.

Existing alpha-DD, oracle-frontier, and P1 selection locks retain their original candidate, source, data, and scope. A new half-family manifest cannot overwrite, reinterpret, or silently inherit an old selected ID. A prospective paper/shadow candidate, if later required, needs its own namespace and immutable selection record, derived only from the permitted reused validation folds 5-12. The record must pin every input result, candidate universe, selection rule, tie rule, time of selection, and the fact that those quarters had already supported adaptive exploration. Specify its rule before applying it; if half outcomes were already inspected, explicitly describe that rule as post-outcome development selection, not preregistration. Do not use test/outer to resolve a tie or select a fallback rule.

The reused development window remains 2021-04-16 13:45 UTC to 2023-04-16 13:45 UTC, right-exclusive. Its 2 bull / 4 bear / 2 sideways counts and failed minimum-three-per-regime gate remain unchanged. Selecting an exploratory candidate does not pass that gate. If no candidate satisfies the registered descriptive requirements, record that outcome and close this weight family; do not force a winner or infer that the best control is qualified.

## 3. Preserve the actual split chronology

The source fold definition is S(f) = 2020-04-16 13:45 UTC + 3*f calendar months; validation is [S(f)-3 months, S(f)), and test is [S(f), S(f)+3 months). Therefore test(f) equals validation(f+1). The label 'test' alone is not evidence of being unexamined. In particular, test folds 5-11 overlap the already reused validation folds 6-12.

The existing alpha_dd_research configuration specifies historical test folds 15-23, covering 2024-01-16 13:45 UTC to 2026-04-16 13:45 UTC, and fresh test fold 24, covering 2026-04-16 13:45 UTC to 2026-07-16 13:45 UTC. These are metadata-derived boundaries, not a claim that any of these prices or outcomes are unread. Preserve the original report-only labels, chronological order, half-open boundaries, original stage locks, and reporting restrictions. Do not rename them validation or substitute currently available shorter periods.

Before new access, audit task logs, artifact manifests, prior reports, selection inputs, and lineage metadata for each exact interval. Do not open prices or performance to decide which period looks usable. If any period was previously used in feature, weight, mask, or policy decisions, classify it as reused. If access history cannot be established, label it 'independence unverified', not untouched. Public historical market knowledge and archive reconstruction also limit claims of a fully prospective simulation. A report-only replay can still be useful without being independent confirmation.

The new family requires its own pinned adapter/manifest for any additional-window replay; existing stage runners must not be silently repurposed. This design does not clear the original P1 execution gates. Keep original formal test/outer restrictions intact.

## 4. Freeze the procedure, timing, and inventory rules

For a newly registered evaluation quarter beginning at S, the proposed procedure uses fit [S-24 months, S-6 months), scale [S-6 months, S-3 months), interval [S-3 months, S), then evaluation [S, S+3 months). This is the inherited 18/3/3/3 calendar procedure anchored at the new evaluation start; it is not permission to refit an old locked model under a different original experiment. Freeze the quarterly refit schedule in the new manifest before any evaluation, and distinguish a frozen procedure from a single permanently frozen coefficient vector.

Use only the existing causal technical/perp-delay0 feature definitions and their frozen common-support contract. Train and calibrate each quarter from past eligible data only; do not adjust the window to obtain eligibility. Keep minimum 512 fit / 64 scale / 64 interval rows. Scale_mean is the mean of eligible actual returns from S-6 months to S-3 months, not the latest three months. No evaluation residuals update the mean, shrinkage weight, variance multiplier, or interval quantile within that quarter. Recompute past-only statistics only on the registered schedule, with hashed model/data/source bindings.

Keep h=24 bars, scheduled 6h decisions, next-15m-open fills, complete 15m calendar, and the inherited origin+25-bar endpoint masks. Fit/calibration labels must mature strictly before their segment boundary; evaluation scores require labels to mature within the registered evaluation endpoint. Copy the exact inherited boundary operators rather than casually changing strictness. Future label availability must not gate inference or orders.

Keep own cash/units and initial B&H inventory, exposure intent range [0.5,1.12], maximum step 0.08, deadband 0.01, one-way fee 0.00055, annual borrowing 0.10, utility risk 1, and utility cost multiplier 2. Generate each base path with its own inventory; replay identical base target intents under doubled fee/borrowing costs. Fallback only applies on a scheduled known-current-open decision when the frozen forecast contract is unavailable; it proposes target 1 through the same step/deadband machinery. Hold makes no new intent. Neither branch gates decisions on a future execution price. These are precisely the two registered alternatives, not an invitation to add selective or partial fallback policies.

For true prospective collection, append timestamps for receipt, decision, intent, fill observation, input/model hash, and later matured label. Check only closed bars actually received by the decision deadline. The extra receipt-availability requirement must be frozen before collection and reported as a separate prospective support contract; archived event-time masks do not prove historical receipt. Never backfill a missed decision with a forecast created after its label matured.

## 5. Concrete independent collection option

If no historical interval can be established as unused, predeclare the next twelve complete quarters on the original calendar: prospective test folds 26-37, 2026-10-16 13:45 UTC to 2029-10-16 13:45 UTC, right-exclusive. This is a proposed fixed horizon, not an automation or a requirement to stop other authorized engineering work. Start only if the manifest and receipt/forecast logging are frozen before its first decision. If that deadline is missed, make a new dated protocol before a later start; do not backdate forecasts.

Retain every scheduled quarter, including failures. Use the fixed start-of-quarter past-only regime classifier (normalized 90-day momentum threshold 0.5 and the inherited volatility normalization), never realized quarter returns. Minimum three per regime means at least three distinct completed nonoverlapping evaluation quarters in each of bull, bear, and sideways within this new confirmation cohort; hence at least nine quarters are necessary. Twelve quarters do not guarantee those counts. They are not nine IID samples, and 'independent confirmation' means separate from development selection, not absence of serial dependence.

If any regime has fewer than three qualifying quarters at the fixed end, retain coverage failure. Do not pool the old 2/4/2 development quarters to manufacture independent coverage, cherry-pick extra historical quarters, extend the study until performance turns favorable, or stop early on good results. A new extension requires a separate prospective protocol and disclosure of prior looks. Operational interim reports may show completeness/failures; performance looks remain explicitly descriptive and cannot drive the frozen candidate or stop rule.

## 6. Endpoints, uncertainty, and failure handling

At the fixed endpoint report per-quarter and equal-quarter all/regime return MSE and MAE against scale_mean and the matched lambda=1 source; use both equal-quarter raw losses and explicitly defined relative improvement 1 - mean_q(L_candidate,q)/mean_q(L_reference,q). IC is supplementary: within-fold positive affine shrinkage preserves IC when forecasts are nonconstant, without preserving MSE, signs, or economics. Zero and fit_mean predictive references may be retained only if bound in the final manifest; do not introduce a new economic arm after seeing results.

Report all AlphaEx/MaxDDDelta levels against B&H and paired half-minus-source / half-minus-anchor deltas, on both fixed cost scenarios, plus turnover, trades, fees, borrowing, availability, fallback requests/fills, and all failures. Overall and every observed regime need the registered AlphaEx-positive / MaxDDDelta-negative signs, but sign checks and minimum counts alone are not a high-probability guarantee.

Do not recycle the reused-quarter descriptive bootstrap as selection-adjusted evidence. Any inferential claim for the four-candidate family needs a separately preregistered serial-dependence-aware joint procedure and multiplicity treatment covering candidate and endpoint decisions before independent outcomes are opened. If that procedure is not specified and audited, the endpoint report remains descriptive even with full regime counts. Existing P1 thresholds and gates are not waived by this skeleton.

Missing inputs trigger the fixed hold/fallback rule and remain in the economic calendar; missing target labels remain unscored with denominators/coverage disclosed. Entire-quarter training or data eligibility failures remain named failures and block a complete-cohort qualification; do not drop them from averages to pass. A schema/hash/boundary/accounting failure stops that run, preserves partial artifacts and any already observed outcomes, and requires an audited repair. An exact same-input restart may resume verified immutable completed artifacts; no alternative seed or method is substituted to improve the result. A changed method starts a new version and does not regain the already viewed confirmation data as untouched.

No favorable result automatically deploys a strategy, replaces an old selected lock, or proves the strongest model across trends. The practical endpoint is a frozen, auditable comparison that either supports the specified claim within its limits or records failure without another same-eight-quarter weight search.

## Sources and limitations

Rapach, Strauss and Zhou (2010), https://doi.org/10.1093/rfs/hhp063, p845 Eq.(23), explains simple forecast combination as shrinkage toward a historical average. Its US equity-premium evidence is motivation, not a theorem for the BTC 6h Ridge mixture or lambda=0.5.

Goyal, Welch and Zafirov (2024), https://doi.org/10.1093/rfs/hhae044, extends published US equity-premium predictors to later data and explicitly notes that reusing original discovery samples is not independent evidence. Its monthly-and-longer findings do not establish BTC performance or select this protocol's candidate.

## Read-only source bindings at draft creation

- `configs/oracle_mean_shrinkage_decisions_20260905.yaml`: `97fa36692b5c3ad58b41d77d34e4707e83c9a2bfa18d21a0c2862579ab5e2f70`
- `configs/oracle_derivative_delay_20260905.yaml`: `50f438fd6f7ed4532ab1f4da77bff3da8938d1b63514483b26e87561537d404f`
- `configs/oracle_frontier_20260905.yaml`: `5cb797df9f435dd8215101367ad63eafa51901c0e11427f208345aafa4d86f3d`
- `configs/alpha_dd_research_20260905.yaml`: `ded0e9f28e75c0c22b3d1fa3bf3b3bc28018dcff003d525a7e86fe6992ae9031`
- `unidream/experiments/alpha_dd_search.py`: `fea62f401d1249fc10681aabdebac63acecb1d6803d6454a2b68ae23b8e97f64`
- `unidream/experiments/oracle_derivative_delay.py`: `04a16003cfcb44e27a1b772ac59ae0666ac4392c1de6acd5f7c0c605d9a77ca9`
- `docs/experiments/oracle_mean_control_decisions_registration_20260905.md`: `4cb324865650270719dac2646415dde47ca5fac47c1671762ee5b415d0c859b0`
- `docs/experiments/oracle_fallback_decisions_registration_20260905.md`: `330e06b1e00016634955ecb76eecba450bc6f53968812ec6b1361a71bc62ab8e`
