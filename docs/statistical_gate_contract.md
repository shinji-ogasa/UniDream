# Development statistical robustness gate

This is a report-only contract for a future candidate comparison. It is not a
re-analysis of the committed Plan011 reports and it must not be run against a
holdout result to manufacture significance. The tool accepts explicit,
already-aligned per-bar paths from development folds only (`fold <= 14`). A
fold `15` or later is rejected before any path is consumed.

## Input contract

The CLI consumes one JSON object. Each candidate has one path per development
fold:

```json
{
  "selected_candidate": "candidate_a",
  "config": {
    "annualization_bars_per_year": 35040,
    "alpha": 0.05,
    "bootstrap_method": "moving_block",
    "bootstrap_replicates": 2000,
    "block_length": 16,
    "block_length_sensitivity": [8, 16, 32],
    "seed": 7,
    "min_folds": 4,
    "min_observations": 32,
    "n_trials": 2,
    "pbo_max": 0.5,
    "stress_min_pass_rate": 1.0
  },
  "candidates": [
    {
      "name": "candidate_a",
      "folds": [
        {
          "fold": 0,
          "alpha_excess_returns": [0.001, 0.002],
          "timing_increment_returns": [0.0005, 0.001],
          "strategy_returns": [0.001, 0.002]
        }
      ]
    }
  ],
  "stress_cases": [
    {
      "name": "high_fee",
      "kind": "cost",
      "alpha_excess_pt": 0.2,
      "timing_increment_pt": 0.1
    },
    {
      "name": "bear",
      "kind": "regime",
      "alpha_excess_pt": 0.1,
      "timing_increment_pt": 0.05
    }
  ]
}
```

`alpha_excess_returns` is the paired strategy-minus-B&H additive net-return
differential. `timing_increment_returns` is dynamic-minus-validation-selected
constant additive net return. `strategy_returns` is the candidate net path
used only for Sharpe and CSCV. The output point estimate is
`100 * sum(per-bar path)`, so it is an additive path statistic and must not be
described as the existing final-equity AlphaEx unless the caller has prepared
that exact additive path. Arrays must be finite, non-empty, and aligned within
each fold. The module never fetches data or discovers paths from filenames.

## Statistical components

1. **Block bootstrap CI.** Moving-block bootstrap is the default; stationary
   bootstrap is also supported. Blocks are sampled independently within each
   fold and the same indices are used for alpha and timing, preserving their
   pairing. `block_length` is the preregistered primary value and every value
   in `block_length_sensitivity` is reported. The fixed seed makes both the
   primary CI and sensitivity reproducible. The primary and all sensitivity
   lower bounds must be positive for the CI components to pass.

2. **Deflated Sharpe Ratio.** Sharpe values in the Bailey/López de Prado
   formula are per-bar; annualized values are display-only:
   `SR_annual = SR_bar * sqrt(annualization_bars_per_year)`. With `T` return
   observations and `N` explicitly recorded trials, the expected maximum
   Sharpe threshold is

   ```text
   SR* = sqrt(Var({SR_n})) *
         ((1-gamma) Phi^-1(1-1/N) +
          gamma Phi^-1(1-1/(N*e)))
   ```

   where `gamma` is the Euler–Mascheroni constant. The reported DSR
   probability is the non-normal probabilistic-Sharpe expression:

   ```text
   Phi((SR_hat-SR*)*sqrt(T-1) /
       sqrt(1-skew*SR_hat + (kurtosis-1)*SR_hat^2/4))
   ```

   `N` is `config.n_trials` when supplied (and must be at least the number of
   candidate paths). Omitting `n_trials` falls back to the candidate count only
   for a diagnostic calculation; it emits a warning, marks DSR as not
   promotion-eligible, and makes the overall gate reject. Formal use must
   record the full number of tried candidates, including candidates or
   parameterizations not retained in the input. Missing or too-short strategy
   paths are N/A and cannot pass.

3. **CSCV/PBO.** Each supplied development fold is one subperiod. For every
   half-subperiod combination, the candidate with the highest in-sample sum
   is selected and its out-of-sample mid-rank is recorded. PBO is the fraction
   of combinations whose selected candidate is below the out-of-sample median
   rank. At least two candidates and four even-numbered subperiods are
   required; otherwise the result is machine-readable `status: "N/A"` with a
   reason and the overall gate rejects. For example, a 13-fold development
   input is intentionally N/A; preregister an even subperiod count such as 12
   (and supply exactly those folds) before treating CSCV/PBO as evidence.

4. **Fold sign/binomial test.** Zero fold totals are omitted. The exact
   one-sided binomial test tests `P(positive)=0.5`; at least `min_folds`
   non-zero folds are required. The alpha and timing tests are reported
   separately.

5. **Cost/regime stress.** Both cost and regime cases are required by default.
   Every case is checked against the configured alpha/timing floors, and each
   group must meet `stress_min_pass_rate`. Missing required groups are N/A and
   cannot pass. Stress numbers are caller-supplied observations and must be
   generated under the same cost/execution contract as the candidate paths.

The final `gate.passed` is the conjunction of every component, including all
bootstrap sensitivity values. N/A is never treated as pass. A rejected gate
is still a valid completed analysis; the CLI returns exit code `2` unless
`--allow-reject` is supplied.

## Reproduction

```bash
uv run python -m unidream.cli.statistical_gate \
  --input path/to/development_statistical_gate.json \
  --output path/to/statistical_gate_result.json
```

The repository currently ships no candidate result input and applies this gate
to no existing experiment. Use synthetic fixtures/tests first. A passing
synthetic fixture is only an API/contract smoke test; it is not a candidate
result, a winner, or evidence for promotion. The default 15-minute crypto
annualization is `365 * 96 = 35040` bars/year; callers must record any
different convention in the JSON config. Formal evaluations must set
`config.n_trials` explicitly to the complete search-trial count.

## Primary references

- [Bailey and López de Prado, *The Deflated Sharpe Ratio* (author PDF)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Bailey, Borwein, López de Prado, and Zhu, *The Probability of Backtest Overfitting* (author PDF)](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [The Deflated Sharpe Ratio (SSRN 2460551)](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2460551_code87814.pdf?abstractid=2460551)
