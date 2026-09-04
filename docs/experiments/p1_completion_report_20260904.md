# UniDream P1 completion report — 2026-09-04

## Executive result

The safe, report-only roadmap is complete through the offline paper-shadow
boundary. The immutable preregistered manifest was not changed:
`results_observed=false` remains its registration state, and no live-money or
exchange order operation was performed.

The current branch contains:

1. strict authenticated S3 body loading and the one-time fixed S3 outer
   report;
2. a fixed-window natural-BTC rolling diagnostic with no selection or
   threshold tuning;
3. a deterministic, orderless offline paper-shadow record; and
4. fail-closed guards preventing a conditional configuration from reaching
   legacy WM, BC, or AC Oracle labels.

## Evidence

- Fixed outer report:
  `docs/experiments/p1_s3_outer_report_20260904.md`
- Rolling diagnostic and offline shadow:
  `docs/experiments/p1_s3_rolling_shadow_20260904.md`
- Conditional WM → BC → AC gate audit:
  `docs/experiments/p1_wm_bc_ac_gate_20260904.md`
- Machine-readable rolling/shadow output:
  `codex_outputs/p1_s3_rolling_shadow_20260904/rolling_shadow.json`

The rolling run completed five fixed expanding-origin windows over the
authenticated cached BTCUSDT 15m zero-injection-control body. Window-average
Ridge h4 MSE was `3.184868e-05`, IC mean `0.01254`, action net `0.126139`,
and AlphaEx versus same-window B&H `-0.047108`. The latest window's Ridge
AlphaEx was `-0.129089`. These are descriptive observations, not a
promotion gate or an investment-performance claim.

The offline shadow selected the fixed Ridge diagnostic for the last window,
simulated 59 filled blocks, and recorded the corresponding net/AlphaEx. It
submitted zero orders, observed zero external fills, wrote no account state,
and is explicitly `live_money=false`.

## Conditional Oracle boundary

The current Plan011 WM → BC → AC path still lacks a production chronological
OOF WM producer and ForecastActionSource adapter with externally pinned
normalizer/calibrator/teacher/action provenance. Therefore the conditional
path remains blocked by design. Disabling the flag and running the old path
would use legacy future-derived `oracle_positions`, so its output would not be
a valid conditional-Oracle result and is not reported.

## Remaining numerical issue

Ridge prediction emits three NumPy/sklearn `matmul` runtime warnings in this
environment while all predictions remain finite. The warning is retained in
the artifacts rather than suppressed. Replacing the operation would change
floating-point bytes and require a complete formal artifact/hash regeneration;
until that is deliberately done, this is a numerical-quality caveat rather
than a claimed production sign-off.

## Verification

At the final branch state, `uv run python -m unittest discover -s tests -v`
passes **352 tests**, `git diff --check` passes, and the worktree matches its
remote branch. No WM/BC/AC training result or live paper account result is
claimed.

