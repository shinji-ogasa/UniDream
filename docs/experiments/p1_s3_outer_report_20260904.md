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
| zero return | 2.022307e-5 | 0.002625 | N/A | 0.4905 | 0.164575 | 0.164575 | 0.000000 | 0 | 0.000000 |
| persistence | 1.080880e-4 | 0.006225 | -0.0221 | 0.4784 | -0.273546 | 0.164575 | -0.438121 | 6,584 | 0.285175 |
| Ridge | 2.054032e-5 | 0.002662 | 0.0021 | 0.4934 | -0.043262 | 0.164575 | -0.207837 | 541 | 0.023111 |

The natural-period result is therefore negative for the fixed Ridge/action
path relative to B&H. It is a descriptive outer observation, not a new model
selection or a claim of investment performance. Ridge emitted three NumPy
`matmul` runtime warnings during the estimator prediction call; all 33,253
inference outputs remained finite and the run did not suppress or repair any
values. This warning should be resolved before treating the calculation as a
production numerical-quality sign-off.

## Injected arm (diagnostic only)

The registered observable-prefix injection is retained only as a paired
diagnostic. Ridge had IC `0.0420`, sign accuracy `0.5030`, and MSE
`2.576630e-5`, but its cost-adjusted AlphaEx was still `-0.019411`; the
control and injected action paths are not interchangeable natural-BTC claims.

## Reproducibility and integration checks

- Report JSON: `codex_outputs/p1_s3_outer_report_20260904/outer_report.json`
- Report file SHA-256: `3f05deac845e8f6db378e2e2712f1c26edef651dfe8b4155b45329352cbf8862`
- Report content digest (self-field excluded):
  `1f73e1264f4d1a4a7193866de1b2924684aee0fcdc4893f17e71693f1dd77135`
- Source-provenance audit SHA-256:
  `be595289bb8dd50c63594a3f700ae50c418b0084d55e54aaf9f6370b56063dcd`
- Code revision: `a6519cd6a49fff7940f0a4043a68e61e9268a47a`
- Full repository suite after the evaluator addition: **344 tests OK**.
- The evaluator's future-return perturbation check preserved intent, fills,
  and effective inventory state; only the outcome/PnL values are allowed to
  change.

The earlier formal validation remains separately documented in
`p1_formal_results_20260904.md`: 16/16 registered comparisons passed its
promotion gate, but that validation result does not imply natural outer-period
performance or execution of the current Plan011 WM→BC→AC training stack.
