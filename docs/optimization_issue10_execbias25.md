## 2026-04-13 regime execution-bias branch

- Config:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_execbias25`
- Change:
  - `use_regime_execution_bias = true`
  - `regime_execution_bias_scale = 0.25`

### Fold 4
- val gap: `0.0912`
- test:
  - `alpha_excess -0.18 pt/yr`
  - `sharpeΔ -0.004`
  - `maxddΔ -0.28 pt`
  - `flat 100%`

### Conclusion
- Adding a regime-conditioned execution head did not change the landing behavior.
- The branch still converged to `flat 100%` on test.
- It did not beat the current global keep.
- Status: reject.
