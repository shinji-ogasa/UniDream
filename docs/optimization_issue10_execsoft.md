## 2026-04-13 execution-aux softening branch

- Config:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_execsoft`
- Change:
  - `execution_aux_coef: 0.75 -> 0.40`

### Fold 4
- val gap: `0.0915`
- test:
  - `alpha_excess -0.21 pt/yr`
  - `sharpeΔ -0.005`
  - `maxddΔ -0.31 pt`
  - `flat 100%`

### Conclusion
- Lowering `execution_aux` reduced the BC training loss but did not change the landing behavior.
- The branch still converged to `flat 100%` on test.
- It underperformed the current global keep.
- Status: reject.
