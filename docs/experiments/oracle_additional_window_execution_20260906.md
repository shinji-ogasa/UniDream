# Additional-window execution binding

This supplement is frozen before outcomes under the existing
[fixed report-only registration](oracle_additional_window_registration_20260906.md).
The acquisition was first registered at commit `f2e1faf` and completed normally.

- Fixed config: `configs/oracle_additional_window_replay_20260906.yaml`.
- New adapter SHA256: `f82d48f25f4234a858a872bddd0c9aac498b4b4ccfc9fd56686041ee1fdf260c`.
- [Data manifest](oracle_additional_window_evidence_20260906/data_manifest.json):
  `609ae008f1cb47919d19f5997409f5833fed1a12d5c106a32af985377945088c`, 178 bindings.
- [Preflight](oracle_additional_window_evidence_20260906/preflight.json):
  `8416f5a21389606e9a1834bb12898349178d34e2cc8a92ca6041a0f3241228b5`.
- UM data: `02d2f07679db0087904b923606a501c8494afda0f75c6a4c94bf4b38ad49a583`.
  All 55 months and 160,608 rows are observed. Reparse from every retained ZIP
  equals the monthly frames and assembled data. This reuses the original parser;
  it is not an independently implemented parsing check.
- Every evaluation grid and boundary bar is present, with observed bar coverage
  1.0. There are 3,648 scheduled decisions, 3,620 usable forecasts and 3,610
  scored forecasts per mean stream. The 28 unavailable forecast decisions are
  all in test18; current opens remain available for target-one fallback.
- Start-regime counts are bull4/bear3/sideways3. These satisfy the minimum count
  condition, but contain no performance evidence or statistical guarantee.
- Full test suite: **589 tests OK in 56.591 seconds**, including eight new tests
  for literal-fold/family freezing, 80 descriptive components, absent regimes,
  complete pairing, finite values, ratios of mean losses, fallback behavior,
  and incomplete resume rejection. `git diff --check` passes.
- A separate agent reviewed the final adapter and used synthetic data to check
  all 64 economic and 16 predictive components, nulls, signs, and candidate
  conditions. It did not inspect market outcomes or fit models.

The adapter saves 33 artifacts per fold: three models, five forecast arrays,
twelve target arrays, ten utility traces, two calibration arrays, and one
calibration/provenance document. Fold manifests must be complete and bound.
Accepted folds reconstruct source evaluation/calibration labels and masks,
check saved forecasts and own-state targets, recompute scores, and replay both
cost accounting paths. A completed output namespace is immutable.

All five forecast contrasts, ten corresponding policy contrasts, and two
half-model fallback-minus-hold contrasts are descriptive. Candidate predictive
conditions compare only with their own full mean and scale mean; the additional
perpetual-half versus technical-half contrast does not become a new gate.
The 80 scalar components have null p-values and no confidence intervals.

After the source/config/evidence commit:

```sh
uv run python -m unidream.experiments.oracle_additional_window_replay --config configs/oracle_additional_window_replay_20260906.yaml
```

Then run the registered
[independent scalar audit](oracle_additional_window_evidence_20260906/independent_scalar_audit.py)
to reconstruct all 100 utility paths, 240 base/stress accounting paths,
50 forecast scores and 20 half-mean identities from saved predictions and
bound raw prices. It does not refit models or independently implement HGB.
The report must distinguish a successful numerical audit from economic or
predictive success on these reused periods.
