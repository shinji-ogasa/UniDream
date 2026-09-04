# P1 formal validation results — 2026-09-04

## Status

The amended preregistration was used for the first authorized formal validation calculation. The run completed with the fixed comparison family and passed the promotion gate:

- Run ID: `p1-formal-20260904T033802Z`
- Manifest: `p1-recovery-preregister-20260830-v1`
- Amendment manifest SHA-256: `128a32230ae94678d5028891fe699e20c564ce7b2261ad10f5a4a83436f08bc6`
- Amended prior manifest SHA-256: `d1854827bd4aa204cc2b5cde375edf62583bf0d164b39e8ac25a6c10ad7dc0c4`
- Trial registry SHA-256: `0f79c41ce0b8ec81c4f02e7ae556ac707779c0e23613cdacddd18b10cfedd587`
- Comparison registry SHA-256: `bed67b607bc7d410add30a81e62f5d452bcc0a67ae3b59cce62744bd18b447db`
- Code revision used by the run: `c08cfadc2e005f1838baab51c8542800e5a6aac0`
- Forecast artifacts: 52; action artifacts: 62; registered comparisons: 16
- Promotion gate: **PASS**

The preregistration file intentionally remains `results_observed=false`; this prevents the validation run from silently becoming an outer-test result. The run ledger separately records `validation_results_observed=true`, while `outer_report_executed=false` and `outer_results_observed=false` remain unchanged.

## Coverage and safety gates

| Gate | Observed minimum | Status |
|---|---:|:---:|
| Forecast eligible fraction | 0.907051 | PASS |
| Label-complete fraction | 0.993189 | PASS |
| Finite prediction fraction | 1.000000 | PASS |
| Scored-action fraction | 0.909164 | PASS |
| S0 ridge positive-edge safety | rejected | PASS |
| S0 persistence positive-edge safety | rejected | PASS |
| S1 all-seed utility deltas positive | true | PASS |
| S1 clairvoyant strictly greater for every seed | true | PASS |

## Registered comparisons

The values below are point estimates from the persisted result artifacts. P-values are the preregistered conservative block-permutation values; adjusted p-values use the registered Holm family of 16 comparisons.

| Comparison | Point estimate | Raw p | Holm p | Direction |
|---|---:|---:|---:|:---:|
| S0 ridge utility vs hold (cost on) | 0.000000 | 1.000000 | 1.000000 | PASS |
| S0 persistence utility vs hold (cost on) | -0.00003135 | 1.000000 | 1.000000 | PASS |
| S1 ridge MSE vs zero (cost off) | -0.00022872 | 0.000500 | 0.007996 | PASS |
| S1 ridge utility vs hold (cost on) | 0.00197085 | 0.000500 | 0.007996 | PASS |
| S2 high vs medium ridge MSE skill (cost off) | 0.17281586 | 0.000500 | 0.007996 | PASS |
| S2 high vs medium ridge normalized regret (cost on) | -0.17603967 | 0.000500 | 0.007996 | PASS |
| S2 high vs medium ridge utility (cost on) | 0.00150545 | 0.000500 | 0.007996 | PASS |
| S2 high vs medium ridge agreement (cost on) | 0.08719621 | 0.000500 | 0.007996 | PASS |
| S2 high vs medium logistic log loss (cost off) | -0.15207134 | 0.000500 | 0.007996 | PASS |
| S2 medium vs low ridge MSE skill (cost off) | 0.54715958 | 0.000500 | 0.007996 | PASS |
| S2 medium vs low ridge normalized regret (cost on) | -0.57163534 | 0.000500 | 0.007996 | PASS |
| S2 medium vs low ridge utility (cost on) | 0.00036762 | 0.000500 | 0.007996 | PASS |
| S2 medium vs low ridge agreement (cost on) | 0.21079623 | 0.000500 | 0.007996 | PASS |
| S2 medium vs low logistic log loss (cost off) | -0.26214013 | 0.000500 | 0.007996 | PASS |
| S3 injected vs control ridge MSE-skill DiD (cost off) | 0.00358740 | 0.004498 | 0.017991 | PASS |
| S3 injected vs control ridge utility (cost on) | 0.00003524 | 0.004998 | 0.017991 | PASS |

## Persisted evidence

The exact run outputs are under `codex_outputs/p1_formal_run_20260904/`. The tracked subset contains the run manifest, coverage and gate JSON, result ledger, and all 16 result `.npz` files. The generated run report is also retained. Large intermediate forecast/action/index arrays remain local and ignored; their hashes and provenance are recorded in the ledgers.

## Interpretation limits

This is a completed preregistered validation calculation for the amended S0–S3 protocol, not a claim that the current UniDream WM/BC/AC production stack has achieved live-trading accuracy. S0–S2 are controlled synthetic arms and S3 uses the already authenticated cached validation body; the S3 cache has a source-revision/provenance caveat documented in the run artifacts. No outer test/report was executed, no post-result threshold or model selection was performed, and the results do not establish performance on unseen live BTC data.

