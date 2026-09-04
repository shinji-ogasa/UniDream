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

## Observed diagnostic summary

The run completed all five fixed windows and all three fixed models. The
following values are arithmetic means across windows; they are not a
statistical gate and were not used for model selection.

| model | mean h4 MSE | mean h4 sign accuracy | mean action net | mean AlphaEx vs B&H | mean filled blocks |
|---|---:|---:|---:|---:|---:|
| zero return | 3.166160e-05 | 0.4915 | 0.173246 | 0.000000 | 0.0 |
| persistence | 1.638454e-04 | 0.4811 | 0.055709 | -0.117537 | 1,725.0 |
| Ridge | 3.184868e-05 | 0.4895 | 0.126139 | -0.047108 | 91.4 |

The latest fixed window (raw `[164000,173111)`, approximately 2023-09-28
through 2024-01-01) gives Ridge h4 MSE `1.862212e-05`, IC `0.0017`,
net `0.339647`, and AlphaEx `-0.129089` with 59 simulated filled blocks.
The offline shadow records `orders_submitted=0`, `external_fills=0`, and
`live_money=false`.

The JSON artifact's pre-self-field content digest is
`ff3fd42da7aa4b2c0d345cfded0ab135588767842a1135ae506a5e0869932364` and its
current file SHA-256 is
`ab7b60e8ddcc84371660214acd91fd0a51c673d4902b08fc419ceaea57a26285`.
