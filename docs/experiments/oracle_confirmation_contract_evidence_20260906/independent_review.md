# Confirmation contract review

Reviewed 2026-09-06. Scope: only the two new source modules, their two tests, and the confirmation YAML. Synthetic metadata/dictionaries were exercised; no additional market, forecast, model, label, parent-result or frozen-family file was read. No repository edit, staging or commit.

## Result

No remaining material issue found in the bounded metadata-only scope after the parent fixes. This is not approval of a working prospective runner or a statistical guarantee.

Fixed and independently rechecked:

- Family SHA is checked in its own record, not overwritten by dictionary merging. Duplicate and resolved-alias paths are rejected; the exact 21 dependency paths are required. Three synthetic removal/duplicate/alias cases now reject.
- Economic and MSE endpoints reject non-real scalars, Python/NumPy booleans, arrays and strings. Twelve synthetic malformed-value cases now reject.
- Equal-quarter aggregation uses scaled math.fsum and a finite-result guard. Previously overflowing finite synthetic values now yield finite descriptive means. Provenance, selection and generalization flags remain false.

Chronology and reporting checked:

- Test26 begins 2026-10-16 13:45 UTC; test37 ends 2029-10-16 13:45 UTC. New fit [S-24m,S-6m), scale [S-6m,S-3m), interval [S-3m,S), evaluation [S,S+3m) is correctly anchored at test_start.
- Segment labels mature at origin+375 minutes. Fit/calibration uses strict endpoint exclusion; evaluation scoring allows maturity exactly at evaluation_end. Synthetic removal of all later labels leaves inference unchanged.
- Receipt feature cutoff remains bar_end<=nominal decision t. An exact t+60s receipt is admitted; t+60s+1ns is rejected; the current bar is rejected even when a delayed deadline is used. Any deadline must be earlier than t+15m. Missing receipt is never a passed receipt condition.
- Inventory is 64 economic plus 16 unique predictive endpoints, with 96 candidate-component references. Predictive endpoints shared by hold/fallback do not become independent discoveries. Complete cohort membership and common per-quarter scoring denominators are required. No p-values or selection are manufactured.

Remaining implementation boundaries, already stated by the code:

- receipt_support validates supplied metadata only. It does not authenticate receipts, validate market fields, prove a complete rolling dependency history, or establish known-current-open availability.
- The eventual adapter must enforce the configured 60-second deadline and six-hour schedule, record the actual forecast/intent submission before next-open fill, and use the unchanged full inherited dependency set plus receipt eligibility. The generic receipt helper deliberately permits other deadlines within a bar; it is not itself the registered adapter.
- segment_masks accepts caller-supplied feature availability. It does not construct a receipt-aware common mask or prove minimum observed fit/calibration counts, full-quarter market coverage, correct regime provenance or accounting.
- describe_complete_family explicitly leaves complete-bar-calendar/protocol-provenance verification false. Finite scalar inputs and paired counts alone do not prove raw data, common masks, B&H-relative units, or preserved target paths.
- The first data/preflight adapter still requires audit before any future or reused report-only replay; all existing report-only restrictions and old selection locks remain in force.

## Reviewed-source SHA-256

| File | SHA-256 |
|---|---|
| `unidream/experiments/oracle_confirmation_contract.py` | `7b5ea82703f931579b12459e189c0006024b9ef7f41147f711e2051ee59eb503` |
| `unidream/experiments/oracle_receipt_support.py` | `97a6fee8837fbb741aedd1759d6134eaa6f1b45f11b1b8941ed4637da3dedb2c` |
| `tests/test_oracle_confirmation_contract.py` | `fde2f45984474f174cd0f6611b725002c8f8b62992c572bbb9daefa6f64c85ea` |
| `tests/test_oracle_receipt_support.py` | `d6bccb8c31373bf5365152c21b9bcab92c4e5ecf7574430e61095f0f6eb35ea3` |
| `configs/oracle_confirmation_contract_20260906.yaml` | `037ae07f5bae182eb9f073de79c48bcd523ff624706c4f4d7e9193cb6f1188f1` |

Only the two current source hashes were compared to their YAML entries in this task. The other 19 bound files and frozen-family artifact were intentionally not opened under the current read scope. The source independently checks those hashes during metadata_preflight, but this audit did not invoke that function.
