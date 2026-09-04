# S3 rolling diagnostic and offline paper shadow — 2026-09-04

## Scope

This report is a post-hoc diagnostic on the authenticated cached BTCUSDT 15m
S3 zero-injection-control body. It is not a new preregistered test, does not
select a model or threshold, and does not amend the immutable manifest. The
fixed windows and model family are declared in
`unidream/experiments/p1_s3_rolling_shadow.py` before execution.

Each window fits `zero_return`, `persistence_last_observed`, and `ridge` at
the left edge using only the expanding prefix `[52492, origin)` and scores the
right-exclusive window. The h4 target is selected by the runner's named
`FORECAST_HORIZONS` entry, not by a positional assumption. Action replay uses
the canonical delayed-fill/4-bar commitment contract and the same causal
mask graph as the fixed S3 outer report.

The final window is also written as an `offline_orderless_paper_shadow`.
It uses no exchange connection, submits zero orders, writes no account state,
and cannot promote a model. The cached historical body has no live freshness
signal, so this is a deterministic replay check rather than paper execution.

## Result artifact

The machine-readable result is
`codex_outputs/p1_s3_rolling_shadow_20260904/rolling_shadow.json`.
It records every window's fit/forecast/action metrics, mask hashes, warning
messages, future-return perturbation invariance, and the offline shadow
summary. `promotion_allowed=false` and `selection_allowed=false` are fixed
in the artifact.

The Ridge path still emits the known NumPy/sklearn `matmul` runtime warning
while producing finite predictions. This remains a numerical-quality issue;
the result is descriptive and should not be treated as production numerical
sign-off until the warning is resolved without changing the pinned artifact
contract.

