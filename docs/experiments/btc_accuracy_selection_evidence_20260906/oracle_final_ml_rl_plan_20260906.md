# Bounded final ML/RL decision and delivery plan

2026-09-06. Planning only; no new real fit, statistic, loss, policy evaluation or additional-test outcome was computed/read here. Stage20 pre-run review is complete with no remaining material finding: `/tmp/oracle_short_direction_prerun_source_review_final_20260906.json`, SHA `3dbfe816e0535e89c5909f35484e16ac5a1db9d03b917e6f209867fb26ce47f9`.

## Recommendation

**Finish the single Stage20 comparison, then stop ML feature/regularization/mapping exploration. Prefer existing audited candidates and spend the remaining effort on a common-contract decision, a reproducible HF bundle and matching Web evidence. Do not start a new RL algorithm family.** The current goal is to deliver an acceptable fixed procedure, not establish that an algorithm family is globally best.

An existing-candidate comparison can be sufficient to choose an engineering compromise if the candidate really meets the declared minimum and can be reproduced. It is not sufficient to guarantee future AlphaEX/DD signs or all-trend accuracy. Repeatedly reused validation data do not become independent because a final selection rule is now frozen. Keep original failed research gates visible even if the user accepts a less demanding overall-mean compromise.

The independent candidate auditor is checking exact identities, periods and metrics. Its preliminary message says that the historical learned-RL artifacts inspected so far either miss the joint-sign requirement or lack retained compatible checkpoints. That is provisional, not a new calculation by this reviewer; the existing mainchain candidate audit remains authoritative. **A finite hindsight beam, a full-future Oracle return replacement or an Oracle objective is not a trained causal RL candidate, an attainable upper bound, or a production artifact.** Do not fill the RL column with one.

## Primary evidence and limits

- **Patterson et al. (2024), Empirical Design in Reinforcement Learning**, sections 2–4: distinguish an instantiated agent from an algorithm's distribution over runs; baseline specification, hyperparameter choices, sampling uncertainty and designer bias materially affect comparisons. Oracle baselines may have unavailable information or different constraints. This supports a complete-system comparison of frozen candidates and explicit stopping, not a claim from one seed that ML generally beats RL. Its discussion does not prescribe our BTC trading threshold or validate serial-quarter confidence. [Original JMLR paper](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf).

- **Fujimoto and Gu (2021), A Minimalist Approach to Offline Reinforcement Learning**, sections 4–6: TD3+BC uses behavior regularization and normalization with lower computational overhead than tested alternatives on D4RL. The same paper documents instability of offline policies across episodes and checkpoints. These results motivate avoiding needless algorithmic complexity; they do not make a new TD3+BC integration cheap in this repository or establish BTC performance. Its hardware/runtime numbers are not estimates for our pipeline. [Original paper](https://arxiv.org/pdf/2106.06860).

- **Kostrikov et al. (2021/2022), Offline Reinforcement Learning with Implicit Q-Learning**, sections 3–4: offline improvement is constrained by distribution shift and available action support; expectile value learning and advantage-weighted extraction address that problem under stated assumptions. This does not make an archived price trajectory a representative state-action dataset or validate Q estimates for unseen inventory decisions. A deployable IQL method needs its own causal transitions, support and policy extraction contract. Thus importing IQL now would add a new experiment family, even if the paper calls the algorithm simple. [Original paper](https://arxiv.org/pdf/2110.06169).

No source guarantees AlphaEX>0 and MaxDDdelta<0. The decision below is an explicit engineering rule, not an inference procedure from those papers.

## Fixed selection contract before ranking

1. Build one manifest of **already evaluated causal candidates plus the frozen Stage20 names**. Separate ML predictors, their deterministic inventory controllers, genuine learned-RL actors and diagnostic Oracles. A rule-based or model-predictive utility planner does not become RL because its economic path is good. Give ML and RL separate qualified/unqualified status; a forced winner for each class is unnecessary.

2. Eligibility requires bound fit/predict/action sources and an executable candidate with no later-data training on an evaluated period. Compare only a common calendar, feature-availability/action rule, next-open fill, B&H-relative initialization, costs/borrow and metric implementation. Different retained input feature contracts can be compared as complete systems, but cannot support an isolated “ML versus RL algorithm” causal conclusion. Do not project a checkpoint trained through a later date backward onto the old folds.

3. Freeze the delivery screen: strictly positive equal-quarter AlphaEX mean and strictly negative equal-quarter MaxDDdelta mean; retain base and doubled-cost values. Recommended default is to require the joint signs under **both** existing cost cases so the delivery screen does not rely on ignoring stress. Also publish all three regime means, quarterwise joint counts and worst quarters. The user's overall-mean minimum is distinct from prior all-stratum/probability gates; do not retrospectively change a stage's failed label.

4. Freeze a deterministic ranking within the eligible set, without fitting a weighted composite: first maximize the worse of the two cost-case AlphaEX means; then prefer the better worse-cost DDdelta; then lower turnover; finally the existing stable candidate ID. This is optional selection bookkeeping for root to adopt before the final table, not a new performance statistic calculated here. Preserve a complete table and all losers. If no candidate meets the screen, record “no qualifying candidate”; do not deploy an Oracle or a failing candidate as a success to meet a deadline.

5. Lock the selected recipe, estimator/controller/risk/calibration artifacts, exact feature order and availability contract, missing-input rule, horizon/cadence, cost assumptions and evidence period in a selection manifest. Any originally report-only test result stays report-only and does not choose a fallback candidate. No extra variant is opened after seeing that report.

## At most one additional fixed operation per family

| Family | Preferred remaining operation | Only permitted reason for an extra run |
| --- | --- | --- |
| ML | Complete Stage20 once; otherwise use existing audited models and policies | A single exact chosen-recipe production refit/export if a current compatible artifact is needed. Freeze cutoff, params, feature support and stopping rule first. This is not another candidate search or a new backtest win. |
| RL | Finish the existing checkpoint/recipe audit; use a valid saved actor only if its evidence meets the same screen | At most one fixed-seed, fixed-config, fixed-fold reproduction of an **already implemented and provenance-recoverable** causal RL recipe. No new algorithm, architecture, reward, feature block, seed/checkpoint sweep or changed controller. |

For RL, use the retained recipe's exact optimizer, loss coefficients, training length and seed; this note does not invent these values. Before allowing the one run, root must resolve them from the existing registry and verify a causal training/evaluation boundary, all reward transitions within training, full inventory state, unchanged execution costs and a fixed final checkpoint rule. One registered fold sweep is one experiment, but every constituent fit/checkpoint is counted explicitly. If that recovery is impossible, or the same-contract adapter itself becomes a new modeling project, **skip the extra RL run and close RL as currently unqualified**. It is better to finish the actual ML deployment than to spend the remaining budget creating a nominal RL comparison.

A production refit after selection changes the delivered weights. Preserve the historical evaluated artifacts, separately identify the production cutoff and hashes, and do not attach the historical out-of-sample score to the new weights as if they themselves had that tested history. If the fixed refit fails, do not tune or substitute another candidate under the same manifest. No best-seed or best-epoch cherry-picking is hidden inside “one run.”

The stop is **completion of this bounded list**, not exhaustion of the weekly budget. Reserve capacity for artifact verification, HF endpoint tests, Web parity and fixing implementation defects. Consuming remaining usage is not a scientific objective.

## Delivery meaning and HF/Web parity

The technical alignment owner handles implementation. The required evidence is one connected contract: selected research recipe and cost account → exact inference bundle and feature/scaler/risk metadata → serving input availability/freshness and action state → Web benchmark period and metric labels. Verify served predictions and target behavior against fixed fixtures, plus live endpoint health separately; fixture parity alone does not validate live data readiness.

Expose the compromise honestly: method name and role, evaluation periods, base/stress costs, AlphaEX/DDdelta means, regime failures and quarter counts, historical versus production artifact hashes, and data freshness. If RL remains unqualified, say so; do not call the ML utility controller learned RL. If overall signs pass but probability/regime gates do not, present a research compromise that meets the stated historical mean floor, not “strongest” or assured trend-independent accuracy. Deployment success and economic research evidence remain separate checks.

No selection, refit, upload, deployment or Web change is performed by this note. Root's candidate audit, frozen selection manifest and technical implementation determine the concrete result.
