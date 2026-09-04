# Strict OOF → WM → BC → AC diagnostic — 2026-09-04

## Answer

The previously blocked route has now been connected and executed as a new
report-only diagnostic:

```text
authenticated S3 control body
  → causal chronological OOF producer
  → strict conditional OOF artifact
  → authenticated teacher context
  → WM training
  → BC training
  → AC fine-tuning
  → the fixed rolling windows and the fixed S3 outer range
```

This is not a replacement for, or an amendment to, the preregistered P1
result.  The immutable manifest still has `results_observed=false`, and the
run submitted no orders and touched no account or live-money state.

## Inputs and strict boundary

- Source: authenticated S3 BTCUSDT 15-minute `zero_injection_control` body;
  body SHA-256 `3fb08afc9113388c935b04559bd0413c0309697ef4ae50208056998308f76772`.
- Decision contract: canonical P0-C contract, h4, one-bar delay, four-bar
  all-or-none commitment, 3 bps spread / 1 bp slippage / 0.03% fee,
  `p_start=1.0`, position range `[0.5, 1.0]`.
- OOF raw range: `[70000, 121000)`; train `[70064, 104528)`, validation
  `[104528, 113000)`, and OOF test `[113000, 121000)`.
- Producer: `causal_linear_one_step`, fit at each chronological origin using
  only the preceding prefix, h4 target, purge 1, train window 16, origin step
  4.  It produced 12,484 finite prediction rows.  No hindsight upper-bound
  path was used.
- Strict artifact: h4 target-end rule, masks, coverage, four external model
  provenance hashes, and the canonical action-contract hash were checked on
  write and checked again on load by the WM/BC/AC path.
- Training seed: `20260904`; a second run with the same seed produced byte-
  identical OOF artifact and WM/BC/AC checkpoint files and identical metrics.

The machine-readable artifact is
`codex_outputs/p1_conditional_wm_bc_ac_20260904/conditional_oof_artifact.json`.
Its internal artifact SHA-256 is
`d7f904c780c74cd8a161d98529b361021c9a70b90ec2305d22dd3177e59bda36`.

## Stage execution

The stages were real training calls, not mocked gates.  This first connection
uses a deliberately small CPU pilot configuration to prove the complete
contract path:

| stage | setting | status |
|---|---:|---|
| predictive-state strict consumer | train 8,354 / 34,464; val 2,118 / 8,472; test 2,000 / 8,000 usable | passed |
| WM | 8 optimization steps | completed |
| BC | 1 epoch | completed |
| AC | 2 optimization steps | completed |

The actor's continuous outputs are causally projected onto the canonical
action grid before replay.  A missing delayed fill cannot mutate inventory;
an outcome gap can remove scoring but cannot change an already filled state.
The actor emitted a 100% short/underweight validation distribution, which is
recorded as a model-collapse warning rather than hidden.

## Same rolling evaluation

The actor was evaluated under the same contract and fixed rolling windows used
by the S3 shadow diagnostic.  Window arithmetic means are descriptive only:

| quantity | all 5 windows | clean forward windows 2–5 |
|---|---:|---:|
| actor net total | 0.131706 | 0.160135 |
| same-window B&H net total | 0.173246 | 0.211461 |
| AlphaEx vs B&H | **-0.041540** | **-0.051327** |
| Sharpe | 1.40635 | 1.72138 |
| filled blocks | 5.0 | 5.0 |
| cost total | 0.000132 | 0.000132 |

The actor did not improve the existing fixed rolling Ridge diagnostic
(AlphaEx mean `-0.047108`) and did not beat B&H.  The negative AlphaEx is the
observed result of this first connected pilot, not an accuracy claim.

## Same fixed S3 outer range

The exact fixed outer range is `[139568, 173111)`, the same raw index range as
the existing S3 outer report.  The strict actor result on that window is:

| metric | strict OOF → WM → BC → AC actor |
|---|---:|
| net total | 0.712511 |
| B&H net total | 0.937454 |
| AlphaEx vs B&H | **-0.224943** |
| Sharpe | 2.25926 |
| filled blocks | 5 |
| turnover | 0.24 |
| cost total | 0.000132 |

The range and contract are identical to the fixed S3 outer report.  The
legacy S3 report's action reducer scores the full body (including bars before
the outer prediction range), whereas this new record restricts both action
and B&H reduction to the declared outer window.  Therefore the two numeric
rows are intentionally not treated as a paired model comparison; the new
record preserves a window-local reduction and calls out the legacy difference
instead of presenting incompatible denominators as a gain.

## Result artifact and reproducibility

The complete report is
`codex_outputs/p1_conditional_wm_bc_ac_20260904/conditional_wm_bc_ac_report.json`.
It contains the source/manifest/contract hashes, OOF bindings, mask-hash
registry for every evaluation window, checkpoint hashes, stage status, and
the fixed rolling/outer metrics.  The tracked report was generated at commit
`0a38431` before this documentation-only update; after the final rerun its
content digest is recorded inside the JSON itself.

The checkpoint directory is intentionally ignored by the repository's normal
rules; the report records each checkpoint's SHA-256 and path.  The OOF JSON,
bindings JSON, and report JSON are the auditable persisted inputs/outputs.

## Decision and next route

The strict connection is technically complete for this diagnostic, but the
pilot is not promotion-ready and does not establish precision improvement:

1. keep `selection_allowed=false`, `threshold_revision_allowed=false`, and
   `promotion_allowed=false`;
2. do not mutate the preregistered P1 manifest or publish this as a formal P1
   outer result;
3. replace the one-step linear producer with the registered ForecastActionSource
   adapter and a properly budgeted/longer WM→BC→AC training run; and
4. rerun the same fixed outer range and rolling windows only after the actor
   no-collapse, mask parity, and paired statistical gates are specified in
   advance.

## Verification

The branch passed the full repository suite (`353 tests OK` before the final
runtime-config regression test), Python compilation, and `git diff --check`.
The final documentation/test update must be followed by the same checks before
merge.  No exchange API, order, account, or live-money operation was used.
