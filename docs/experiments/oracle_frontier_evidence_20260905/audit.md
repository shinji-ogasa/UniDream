# Independent audit record

Audit performed during this goal with the full raw source and saved artifacts.
This is an agent execution record, not third-party certification.

- 13/13 validation quarters,53 policy rows and18 forecasts per quarter.
- 689 target,234 forecast and234 model SHA256 values checked; data artifact,
 availability, ledger, sidecar, config and six source hashes checked.
- 52 independent scalar cash/units replays for the selected flow HGB and frozen
 robust overlay (13 quarters x2 policies x2 costs). The audit did not import
 the evaluator's `_simulate` or `metrics` arithmetic. Maximum AlphaEx difference
1.5543122344752192e-15; maximum DD difference1.4432899320127035e-15;
 turnover8.881784197001252e-15; trade counts exact.
- All117 Ridge models' saved predictions equal non-BLAS einsum evaluation
 exactly.5,967 scalar math.fsum cells differ by <=3.469446951953614e-17.
 Independent normal equations for27 models across folds0/6/12 reproduce
 coefficients to2.259612852731352e-16 and intercepts to3.712308238590367e-16.
 NumPy emitted divide/overflow/invalid matmul warnings despite numerically
 correct checked output; the underlying warning cause remains undiagnosed.
- All fit masks reconstructed:974--2,543 rows; minimum last-label purge90min.
 Saved inference and score support masks and actual NaN masks verified for all
234 forecast artifacts. Actual outcomes independently recomputed to
3.842737759940373e-15. Latest scored outcome end2023-04-16 12:15 UTC;
 computational data stop2023-04-16 13:30 UTC.
- No later historical/fresh period enters fitting/scoring/selection. The complete
 parquet is read and checked before truncation, so no claim that future bytes
 were never physically read. Validation periods have prior research exposure.
- 78 ML perfect-outcome paths reproduce exactly.26 RL paths retain declared
 hindsight/feasible-lower-bound/no-teacher restrictions.
- All summary/regime means and ranking reproduce. Raw scalar regime calculation
 confirms bull5/bear5/sideways3, far from threshold ambiguity.

Full unit suite after risk-controller integration:410 tests,56.427seconds,OK.
Economic and forecasting claims remain exploratory despite passing these checks.
