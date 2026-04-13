## 2026-04-13 outcome-edge relabel branch

- Config:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_outcome_relabel`
- Settings:
  - `sample_quality_mode = outcome_edge_relabel`
  - `sample_quality_coef = 0.5`
  - `sample_quality_quantile = 0.75`

### Fold 4
- val gap: `0.0904`
- test:
  - `alpha_excess -0.07 pt/yr`
  - `sharpeΔ -0.001`
  - `maxddΔ -0.24 pt`
  - `flat 100%`

### Fold 0
- test:
  - `alpha_excess +0.33 pt/yr`
  - `sharpeΔ +0.000`
  - `maxddΔ -0.65 pt`
  - `flat 100%`

### Conclusion
- This branch improved the validation imitation gap more than most recent weighting branches.
- It still converged to `flat 100%` on test.
- It did not beat the current global keep.
- Status: reject for global promotion.

## Plain outcome-edge alias check

- `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_outcome_edge`
  produced the same fold-4 results as the relabel variant:
  - val gap `0.0904`
  - test `alpha_excess -0.07 pt/yr`
  - `sharpeΔ -0.001`
  - `flat 100%`
- In `unidream/experiments/bc_setup.py`, `outcome_edge` and `outcome_edge_relabel`
  currently share the same code path.
- Status: treat `outcome_edge` as an alias, not a distinct branch.
