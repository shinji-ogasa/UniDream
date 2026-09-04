# P1 formal validation calculation (amended manifest)

- Run output: `codex_outputs/p1_formal_run_20260904`
- Manifest SHA-256: `128a32230ae94678d5028891fe699e20c564ce7b2261ad10f5a4a83436f08bc6`
- Trial registry SHA-256: `0f79c41ce0b8ec81c4f02e7ae556ac707779c0e23613cdacddd18b10cfedd587`
- Comparison registry SHA-256: `bed67b607bc7d410add30a81e62f5d452bcc0a67ae3b59cce62744bd18b447db`
- Validation results observed: `true`; preregistration results observed: `false`; outer results observed: `false`.
- The report-only outer operation was not executed.

## Coverage

- Forecast rows: 1248; minimum eligible fraction: 0.907051
- Minimum label-complete fraction: 0.993189
- Minimum finite-prediction fraction: 1.000000
- Minimum scored-action fraction: 0.909164
- Coverage gate: **PASS**

## Registered primary comparisons

| comparison | point | conservative raw p | Holm adjusted p | direction gate | result artifact |
|---|---:|---:|---:|:---:|---|
| `S0__ridge__utility_vs_hold__cost_on` | 0 | 1 | 1 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S0__ridge__utility_vs_hold__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S0__ridge__utility_vs_hold__cost_on.npz) |
| `S0__persistence__utility_vs_hold__cost_on` | -3.13456044e-05 | 1 | 1 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S0__persistence__utility_vs_hold__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S0__persistence__utility_vs_hold__cost_on.npz) |
| `S1__ridge__mse_vs_zero__cost_off` | -0.000228715458 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S1__ridge__mse_vs_zero__cost_off.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S1__ridge__mse_vs_zero__cost_off.npz) |
| `S1__ridge__utility_vs_hold__cost_on` | 0.0019708462 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S1__ridge__utility_vs_hold__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S1__ridge__utility_vs_hold__cost_on.npz) |
| `S2__high_vs_medium__ridge__mse_skill__cost_off` | 0.172815863 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__mse_skill__cost_off.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__mse_skill__cost_off.npz) |
| `S2__high_vs_medium__ridge__normalized_regret__cost_on` | -0.176039671 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__normalized_regret__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__normalized_regret__cost_on.npz) |
| `S2__high_vs_medium__ridge__utility__cost_on` | 0.00150545296 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__utility__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__utility__cost_on.npz) |
| `S2__high_vs_medium__ridge__agreement__cost_on` | 0.0871962059 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__agreement__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__ridge__agreement__cost_on.npz) |
| `S2__high_vs_medium__logistic__log_loss__cost_off` | -0.152071342 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__logistic__log_loss__cost_off.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__high_vs_medium__logistic__log_loss__cost_off.npz) |
| `S2__medium_vs_low__ridge__mse_skill__cost_off` | 0.547159578 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__mse_skill__cost_off.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__mse_skill__cost_off.npz) |
| `S2__medium_vs_low__ridge__normalized_regret__cost_on` | -0.571635341 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__normalized_regret__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__normalized_regret__cost_on.npz) |
| `S2__medium_vs_low__ridge__utility__cost_on` | 0.000367619954 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__utility__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__utility__cost_on.npz) |
| `S2__medium_vs_low__ridge__agreement__cost_on` | 0.210796226 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__agreement__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__ridge__agreement__cost_on.npz) |
| `S2__medium_vs_low__logistic__log_loss__cost_off` | -0.262140128 | 0.00049975 | 0.007996 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__logistic__log_loss__cost_off.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S2__medium_vs_low__logistic__log_loss__cost_off.npz) |
| `S3__injected_vs_control__ridge__mse_skill_did__cost_off` | 0.00358740303 | 0.00449775 | 0.017991 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S3__injected_vs_control__ridge__mse_skill_did__cost_off.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S3__injected_vs_control__ridge__mse_skill_did__cost_off.npz) |
| `S3__injected_vs_control__ridge__utility__cost_on` | 3.52420992e-05 | 0.0049975 | 0.017991 | PASS | [`codex_outputs/p1_formal_run_20260904/results/S3__injected_vs_control__ridge__utility__cost_on.npz`](/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/codex_outputs/p1_formal_run_20260904/results/S3__injected_vs_control__ridge__utility__cost_on.npz) |

## Overall promotion gate: **PASS**

A failed gate is an observed preregistered outcome, not a reason to tune thresholds or execute the outer report.
