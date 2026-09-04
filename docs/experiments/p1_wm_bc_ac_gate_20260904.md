# WM → BC → AC conditional-stage gate — 2026-09-04

## Decision

The current Plan011 WM → BC → AC orchestration is **not executable as a
valid conditional-Oracle experiment yet**. The strict path remains
fail-closed. No WM, BC, or AC result is claimed by this run.

The reason is not a missing unit test: the chronological OOF artifact
validator and predictive-state consumer are implemented, but no production
producer is connected that retrains the WM at each origin and supplies the
required external normalizer/calibrator/teacher/action-contract provenance.
The existing fold input still exposes legacy `oracle_positions`; allowing
those labels to cross the conditional boundary would reintroduce the
future-target teacher path that the audit explicitly prohibited.

## Gate checks

The following calls were checked with a strict conditional configuration:

- fold orchestration rejects the conditional path before legacy fold inputs;
- `prepare_world_model_stage` rejects missing chronological OOF state;
- `build_wm_predictive_state_bundle` rejects a missing/invalid strict OOF
  bundle;
- `run_bc_stage` rejects legacy `oracle_positions`;
- `run_ac_stage` now rejects the same bypass (the guard was added in this
  revision), so AC cannot train when WM/BC has no authenticated OOF teacher.

The three stage-level regressions pass. The next implementation required to
unblock this route is an authenticated, chronological OOF producer/adapter
that maps its forecast capability into WM predictive state, then passes the
same typed state and action/mask contract to BC, AC, replay, and backtest.
Until that adapter exists, running Plan011 with conditional flags disabled
would be a legacy in-sample Oracle experiment and is intentionally not used as
a precision result.

