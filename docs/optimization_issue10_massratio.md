## 2026-04-13 mass-ratio branch

- Config:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_massratio`
- Change:
  - `position_mean_match_coef: 1.0 -> 2.0`
  - `short_mass_match_coef: 2.0 -> 1.0`

### Fold 4
- val gap: `0.0913`
- test:
  - `alpha_excess -0.09 pt/yr`
  - `sharpeΔ -0.002`
  - `maxddΔ -0.30 pt`
  - `flat 100%`

### Conclusion
- Rebalancing the mean-vs-short mass penalties slightly improved the validation gap.
- It still converged to `flat 100%` on test.
- It did not beat the current global keep.
- Status: reject.
