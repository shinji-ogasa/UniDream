# S3 outer report-only calculation — 2026-09-04

## Execution boundary

This is the single terminal, report-only calculation for the preregistered
S3 outer range. The existing validation entrypoints still reject accidental
outer execution. This evaluator loaded the immutable manifest and the
authenticated v4 body, then performed one fit per fixed continuous model at
raw origin `139568`:

- fit prefix: `[52492, 139568)`
- outer prediction: `[139568, 173111)` (2023-01-01 through 2024-01-01)
- horizon: 4 bars
- refits: none
- model/threshold selection: none
- action contract: canonical hash
  `6f5beb7865fceac5ecbcfbb31dd11e8fdada02e1841fecac1c17e22377bb624f`

The preregistration itself remains `results_observed=false`. The outer report
is not a promotable forecast/action artifact and has
`promotion_allowed=false`.

## S3 body and provenance

The v4 body passed the runtime validator, matched all frozen feature/return /
availability content digests, and contains 173,111 rows. Body SHA-256:

`3fb08afc9113388c935b04559bd0413c0309697ef4ae50208056998308f76772`

The local metadata differs from the frozen metadata only in source-probe
provenance. A fresh official Binance probe of 2018-02 reproduced the frozen
Spot archive payload SHA (`ba2e6f02672dd8f1fce38dde096a756df2194f7a17e5d401d253b5e82623697b`)
and confirmed the official UM mark/funding 404s. The manifest and local body
were not replaced or mutated. The detailed probe record is kept in
`codex_outputs/p1_s3_provenance_audit_20260904.json`.

## Natural BTC control arm

The `zero_injection_control` arm is the natural cached BTC path. There are
33,249 finite h4 score rows and 8,314 causal action decision blocks. The
action numbers below use the shared delayed-fill/outcome mask graph and the
fixed cost contract; B&H is the same period and same scored outcome bars.

| Model | MSE | MAE | IC | Sign accuracy | Net | B&H net | AlphaEx vs B&H | Filled blocks | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero return | 5.377775e-6 | 0.001352 | N/A | 0.4958 | 0.164575 | 0.164575 | 0.000000 | 0 | 0.000000 |
| persistence | 9.239356e-5 | 0.005616 | -0.0226 | 0.4715 | -0.273546 | 0.164575 | -0.438121 | 6,584 | 0.285175 |
| Ridge | 5.671908e-6 | 0.001410 | 0.0056 | 0.4963 | -0.043262 | 0.164575 | -0.207837 | 541 | 0.023111 |

The natural-period result is therefore negative for the fixed Ridge/action
path relative to B&H. It is a descriptive outer observation, not a new model
selection or a claim of investment performance. Ridge emitted three NumPy
`matmul` runtime warnings during the estimator prediction call; all 33,253
inference outputs remained finite and the run did not suppress or repair any
values. This warning should be resolved before treating the calculation as a
production numerical-quality sign-off.

## Injected arm (diagnostic only)

The registered observable-prefix injection is retained only as a paired
diagnostic. Ridge had IC `0.0982`, sign accuracy `0.5245`, and MSE
`5.740874e-6`, but its cost-adjusted AlphaEx was still `-0.019411`; the
control and injected action paths are not interchangeable natural-BTC claims.

## Reproducibility and integration checks

- Report JSON: `codex_outputs/p1_s3_outer_report_20260904/outer_report.json`
- Report file SHA-256: `e205b2c76b44fb99cfefdf656b72086ff65c0cffbeceda425500bca3375cdbe6`
- Report content digest (self-field excluded):
  `8f0e23b87233b4de2fde1b6f8a85f596f85c450f7fdf16da4ffad0c2bf3daea1`
- Source-provenance audit SHA-256:
  `89dbe611b510d57adcae6ebd7beb140969c2a84ef1934ee5febb082aced1c341`
- Code revision: `805f242bc5c95182d0150b99877f1445bc50c9f7`
- Full repository suite after the evaluator addition: **344 tests OK**.
- The evaluator's future-return perturbation check preserved intent, fills,
  and effective inventory state; only the outcome/PnL values are allowed to
  change.

The earlier formal validation remains separately documented in
`p1_formal_results_20260904.md`: 16/16 registered comparisons passed its
promotion gate, but that validation result does not imply natural outer-period
performance or execution of the current Plan011 WM→BC→AC training stack.
