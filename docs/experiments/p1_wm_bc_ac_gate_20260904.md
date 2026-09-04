# WM → BC → AC conditional-stage gate — historical pre-connection snapshot

> **Superseded by** [`p1_conditional_wm_bc_ac_20260904.md`](p1_conditional_wm_bc_ac_20260904.md).
> This file records the gate state before the strict producer was connected;
> it is retained as an audit snapshot and is not the current execution status.

## Decision

At the time of this snapshot, the Plan011 WM → BC → AC orchestration was **not
executable as a valid conditional-Oracle experiment yet**. The strict path was
fail-closed and no WM, BC, or AC result was claimed by that run. The current
state is documented in the superseding report above.

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

The three stage-level regressions pass. The next implementation required at
that point was an authenticated, chronological OOF producer/adapter that
mapped its forecast capability into WM predictive state, then passed the same
typed state and action/mask contract to BC, AC, replay, and backtest. That
connection is now exercised by the superseding diagnostic report; its pilot
metrics remain report-only and are not formal P1 results.
