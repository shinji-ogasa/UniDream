## 2026-04-13 regime trade-bias branch

- Config:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_tradebias25`
- Change:
  - `use_regime_trade_bias = true`
  - `regime_trade_bias_scale = 0.25`

### Fold 4
- val gap: `0.0917`
- test:
  - `alpha_excess -0.17 pt/yr`
  - `sharpeΔ -0.004`
  - `maxddΔ -0.33 pt`
  - `flat 100%`

### Conclusion
- Adding a regime-conditioned trade gate did not break the near-flat landing.
- It still converged to `flat 100%` on test.
- It did not beat the current global keep.
- Status: reject.
