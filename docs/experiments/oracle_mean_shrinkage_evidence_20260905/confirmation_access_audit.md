# Independent-confirmation access audit — metadata only

Audit date: 2026-09-05. This is an access/period audit, not a registration or execution authorization. No additional candidate, price, forecast, label, model coefficient, or performance was computed or inspected in this task. No report-only test was scored. Only the supplied design text, configuration/source text, registration metadata, selected manifest path/hash fields, and directory/file names were consulted. Repository files were not changed.

Workspace: `/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905`.

## Finding

No completed historical contiguous period can be certified as unread from the permitted records. Historical tests 15–23 and fresh test 24 are explicitly reused in an existing registration. Original alpha-DD development also evaluated test 12; it is not a spare quarter immediately after the current oracle cutoff. Tests 13–14 occur inside later registered fitting/inference history, and their exact human/performance exposure remains unverified. A missing log or artifact does not prove non-access.

The earliest complete quarter that has not yet occurred on the unchanged calendar is test 26, starting 2026-10-16 13:45 UTC. Tests 26–37 form the proposed fixed future interval [2026-10-16 13:45 UTC, 2029-10-16 13:45 UTC). This is a prospective eligibility statement, conditional on the final protocol and receipt/forecast logging being frozen before the first decision; it is not a claim that twelve quarters guarantee three per regime.

## Authoritative timing and overlap

`unidream/experiments/alpha_dd_search.py:322–327` defines S(f)=2020-04-16 13:45 UTC + 3f calendar months, train [S−27 months,S−3 months), validation [S−3 months,S), test [S,S+3 months). All intervals below are right-exclusive. `configs/oracle_frontier_20260905.yaml:5–8` explicitly evaluates the validation interval and cuts off at 2023-04-16 13:45 UTC; the derivative continuation uses folds 5–12 (`configs/oracle_derivative_delay_20260905.yaml:9`). Therefore validation(f)=test(f−1), not test(f).

| Current validation fold | Validation interval [start,end) | Identical historical test interval | Same-number test overlap with current validation |
|---:|---|---:|---|
| 5 | 2021-04-16 13:45 UTC → 2021-07-16 13:45 UTC | test 4 | test 5: validation 6 |
| 6 | 2021-07-16 13:45 UTC → 2021-10-16 13:45 UTC | test 5 | test 6: validation 7 |
| 7 | 2021-10-16 13:45 UTC → 2022-01-16 13:45 UTC | test 6 | test 7: validation 8 |
| 8 | 2022-01-16 13:45 UTC → 2022-04-16 13:45 UTC | test 7 | test 8: validation 9 |
| 9 | 2022-04-16 13:45 UTC → 2022-07-16 13:45 UTC | test 8 | test 9: validation 10 |
| 10 | 2022-07-16 13:45 UTC → 2022-10-16 13:45 UTC | test 9 | test 10: validation 11 |
| 11 | 2022-10-16 13:45 UTC → 2023-01-16 13:45 UTC | test 10 | test 11: validation 12 |
| 12 | 2023-01-16 13:45 UTC → 2023-04-16 13:45 UTC | test 11 | test 12: outside current validation; original alpha-DD development |

The union of current validation 5–12 is [2021-04-16 13:45 UTC,2023-04-16 13:45 UTC), exactly test 4–11. The development labels are not interchangeable: the original alpha-DD stage with folds 0–12 evaluates `test_start:test_end` (`alpha_dd_search.py:426–443`), giving [2020-04-16 13:45 UTC,2023-07-16 13:45 UTC). Its cutoff/folds are pinned at `configs/alpha_dd_research_20260905.yaml:11–20` and in all four archived registrations. The current feature-frontier development 0–12 instead spans validation [2020-01-16 13:45 UTC,2023-04-16 13:45 UTC).

## Exposure classification

| Interval | Calendar bounds [start,end) | Classification | Basis and limit |
|---|---|---|---|
| Original tests 0–12 | 2020-04-16 13:45 UTC → 2023-07-16 13:45 UTC | Reused development | Original stage selects on these test-named windows; includes test12 beyond the oracle cutoff. |
| Test13 | 2023-07-16 13:45 UTC → 2023-10-16 13:45 UTC | Prior source/fitting exposure; independence unverified | Inside fold15 and later training history. Exact rows/labels used were not inspected here. |
| Test14 | 2023-10-16 13:45 UTC → 2024-01-16 13:45 UTC | Prior source/inference/fitting exposure; independence unverified | Fold15 validation/inference interval and inside fold16 training history. No unread claim. |
| Historical tests15–23 | 2024-01-16 13:45 UTC → 2026-04-16 13:45 UTC | Recorded reused confirmation | Existing registration explicitly says confirmation periods had prior results; archived BTC/ML/BNB historical artifacts are listed in a manifest. |
| Fresh test24 | 2026-04-16 13:45 UTC → 2026-07-16 13:45 UTC | Recorded reused confirmation | The registration explicitly warns fresh is a stage name, not untouched data; BTC/ML/BNB fresh artifacts are archived in the manifest. |
| Test25 | 2026-07-16 13:45 UTC → 2026-10-16 13:45 UTC | Partly elapsed; historical part independence unverified | Current date falls inside this quarter. No allowed record certifies the elapsed portion unread; the whole quarter cannot now be prospectively registered. |
| Proposed tests26–37 | 2026-10-16 13:45 UTC → 2029-10-16 13:45 UTC | Future, conditional prospective cohort | Not yet realized on audit date. Freeze the full procedure and logging before test26 first decision. |

The direct recorded-exposure evidence is `docs/experiments/alpha_dd_ml_registration_20260905.md:3–5` and `:35–40`: the original BTC and cross-asset trial had already produced results, confirmation is reused, and fresh is not claimed untouched. This is stronger than inferring exposure from a filename. `configs/plan011_overlay_actor_v31_holdout.yaml:1–4` separately pins prior holdout folds15–23; a config alone is only planned scope. The parent also explicitly reports historical holdout results already observed. The memory registry has a prior validated/published holdout pointer (MEMORY.md:190,200); its outcome report and session log were deliberately not opened.

For the 2023 gap, the registered source fits only target_end < train_end (`alpha_dd_search.py:335–345`). Fold15 has train_end 2023-10-16; fold16 has train_end 2024-01-16. File names `alpha_dd_ml_v1/models/fold15_ridge_h7.joblib` and `fold16_ridge_h7.joblib` are present. Those model files were not opened. Stage input loading extends through the cutoff (`:426–431`), and inference begins at val_start (`:345–346`). This shows the gaps are within prior source access and model-operation scope; it does not claim every row was retained by availability/purge masks or that a human viewed every gap outcome.

ETH historical/fresh artifacts were not present in the limited directory inventory. This absence is not evidence that those periods were untouched. The BTC-period conclusion does not depend on an ETH result.

## Archived outcome paths: manifest metadata only

The following hashes are recorded claims copied from the allowed manifest, not a fresh read or rehash of any outcome file. `copy_exact=true` was recorded for each pair. The audit did not open these referenced files. Relevant manifest source-map path lines: BTC1211/1221, BNB1313/1323, ML1374/1384; scope lines1155–1165 identify fresh24 and historical15–23.

| Source artifact path recorded in manifest | Recorded SHA-256 |
|---|---|
| `codex_outputs/alpha_dd_research_v1/historical.json` | `66662e8b3c8fd8a37d0f17d2876f525e2e9d0e1bb1e5b744e9d5024af7738299` |
| `codex_outputs/alpha_dd_research_v1/fresh.json` | `0b0dcc9812a7e50a4b802dd8901742e14d8a7ee969d7e70ac33be54dd4fbf95a` |
| `codex_outputs/alpha_dd_bnb_v1/historical.json` | `7f36b58409afda229fb35cdb454b14bae33eb6d275b298d0ccb180dec897f880` |
| `codex_outputs/alpha_dd_bnb_v1/fresh.json` | `2c4045ce132f1f3a848abdd66a3e77c8ae41a7a81401ab6362af92979459d825` |
| `codex_outputs/alpha_dd_ml_v1/historical.json` | `e1bbc341b3b7bd034e51d3bf372758d897b17d004ff1bf3df8217568b63cd784` |
| `codex_outputs/alpha_dd_ml_v1/fresh.json` | `647a5ba46a7c309a190272de7cf027a0a356f5b84f52ba5a2312e163db092a39` |

## Confirmation boundary recommendation

Close the fixed 0/0.5/1 family. Freeze all four half candidates and eight existing controls in a new namespace; preserve all earlier selection locks. A replay of historical15–24 may be labeled reused report-only analysis, but it cannot repair independent confirmation. No historical interval identified by this bounded audit warrants an untouched label. Do not search prices or past results to choose an apparently promising replacement interval.

If using the original quarterly calendar, register tests26–37 and all failure/stop rules before 2026-10-16 13:45 UTC. Do not backdate forecasts, drop missing quarters, pool the old 2/4/2 regime quarters, or extend until signs improve. Start-of-quarter regimes are not calculated in this audit. Three quarters per regime remains an unmet development gate, and the proposed fixed twelve-quarter cohort may also fail it. A receipt-aware future support contract is required because archive event-time availability does not prove decision-time receipt. Statistical and multiplicity procedure design remains separate unfinished registration work.

## Source bindings independently hashed in this audit

Only allowed source/config/registration/manifest/draft files were hashed. No raw market file, model, NPZ, target, forecast, score, or results JSON was read to obtain a digest.

| File | SHA-256 |
|---|---|
| `/tmp/oracle_confirmation_design_20260905.md` | `7c6053af4d6911cb9316148361566ae9590f9840b44488569a678a95f923e78f` |
| `unidream/experiments/alpha_dd_search.py` | `fea62f401d1249fc10681aabdebac63acecb1d6803d6454a2b68ae23b8e97f64` |
| `configs/alpha_dd_research_20260905.yaml` | `ded0e9f28e75c0c22b3d1fa3bf3b3bc28018dcff003d525a7e86fe6992ae9031` |
| `configs/alpha_dd_ml_20260905.yaml` | `900c61f35e1bf3e5d74786d449037559322f77f69ba8a343969e4f93b80be19e` |
| `configs/plan011_overlay_actor_v31_holdout.yaml` | `852ab670828b074a39bd72706cfec1b671712eae94c51e3b565b2509ef2a9d07` |
| `configs/oracle_frontier_20260905.yaml` | `5cb797df9f435dd8215101367ad63eafa51901c0e11427f208345aafa4d86f3d` |
| `configs/oracle_derivative_delay_20260905.yaml` | `50f438fd6f7ed4532ab1f4da77bff3da8938d1b63514483b26e87561537d404f` |
| `configs/oracle_mean_shrinkage_decisions_20260905.yaml` | `97fa36692b5c3ad58b41d77d34e4707e83c9a2bfa18d21a0c2862579ab5e2f70` |
| `docs/experiments/alpha_dd_ml_registration_20260905.md` | `a7e1b936ee32ff5dbfe20783b1dee7122167f266ba259319131014227b1b1f43` |
| `docs/experiments/alpha_dd_cross_asset_registration_20260905.md` | `e77e471ffc92dbd23243c930bc0f8b70615f62cdbddeb63fbcda4f27702e0f2e` |
| `docs/experiments/alpha_dd_evidence_20260905/manifest.json` | `a86df11e36e0709f87ddd3707c854de3ef0fa3fca81dc90fbb98f1ba86c159e4` |
| `docs/experiments/alpha_dd_evidence_20260905/btc/registration.json` | `35d06aa0ba9953a378bead4e52348571736a16fc62e51c7be017446a74c53c15` |
| `docs/experiments/alpha_dd_evidence_20260905/ml/registration.json` | `2ea05003e54e324bdc8dc0239a6aae8fd70c1b5a76de60d9b3136fe4dafc4f17` |
| `docs/experiments/alpha_dd_evidence_20260905/bnb/registration.json` | `642a38c8c2b541b736e3b9ff90e5a86f498c5583a2059a04284d7b7491aa5006` |
| `docs/experiments/alpha_dd_evidence_20260905/eth/registration.json` | `cc57d3e7c6b4551274e470eecf5e3c44f0a47fc1641c345c271edba2fedb2f6a` |

## Scope limits

This is not a complete forensic access log of every prior user, agent, browser, repository, external market source, or deleted artifact. Registration prose establishes known reuse; absence of additional records establishes nothing about unread status. Manifests may include historical result descriptions, but only period/path/hash metadata was selected for this audit. No later numerical outcome was emitted or used. Existing current-study accounting audits are outside this period-audit scope. Source hashes bind the files at audit time and must be rebound if the final prospective protocol changes.
