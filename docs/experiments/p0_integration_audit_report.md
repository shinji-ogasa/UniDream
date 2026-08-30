# P0-A/B/C integration audit

**Audit date:** 2026-08-30
**Purpose:** join the validated v4 availability sidecar, the canonical
P0-C replay, and the P0-B OOF/inventory guards. This is a data/contract
integrity audit, not a market-accuracy or investment result.

## Verdict

| area | status | evidence |
| --- | --- | --- |
| P0-A availability and 17-feature mapping | `passed` | `tests/test_p0_integration_audit.py`, fixture and real-cache run below |
| P0-B target/OOF/inventory | `partial` | same-row OOF and hindsight-inventory guards pass, but legacy full-WM chronological OOF retraining remains disconnected |
| P0-C action/execution parity | `passed` | strategy/B&H/U0/causal-teacher replay has one mask/hash and one timeline |

P0-B is deliberately not promoted to `passed`: the integration guard proves
the boundary, not the missing per-fold WM/normalizer/calibrator/student replay
runner.

## Reproducible command

```text
uv run python -m unidream.experiments.p0_integration_audit \
  --cache-dir /tmp/unidream-v4-p0-20260830 \
  --cache-tag BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official \
  --frozen-metadata-path docs/data_quality_v4_rebuild_2018_2024_metadata.json \
  --output-json /tmp/p0-integration-audit-actual.json \
  --output-report /tmp/p0-integration-audit-actual.md
```

The runner uses `context_bars=64` by default. A smaller context is accepted
only for the unit-test fixture.

## Real v4 cache evidence

Source cache: `/tmp/unidream-v4-p0-20260830`, tag
`BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official`.

- schema version: `4`
- schema digest: `1c1c41a9aca3e8af22b357a8483ea6419745ee4b24c10c09c47289df3744c616`
- cache-local metadata SHA-256: `bade1775884cd22c8675af225b429976aa6b2c60b859b4a591c76f8a87d17450`
- feature body: `173111` rows, canonical width `17`
- complete sidecar grid: `210336` rows; sidecar gap groups: `30`
- P0-A body row eligibility (Spot + funding + mark): `119849`
- P0-A decision eligibility after the 64-bar context/window check:
  `118652`
- P0-C score eligibility (observed finite Spot return on the complete 15m
  grid): `173111`
- first/last feature body timestamps: `2018-01-19 17:00:00` /
  `2023-12-31 23:45:00`
- first/last sidecar grid timestamps: `2018-01-01 00:00:00` /
  `2023-12-31 23:45:00`

The feature/decision mask requires the full 17-feature context and all
required external availability flags. The score mask deliberately requires
only an observed Spot bar, a finite return, and 15-minute continuity; a
funding/mark gap inside an observed Spot outcome block does not erase that
Spot return from scoring. Missing body rows remain false/NaN on the complete
grid and are never compacted.

Content/provenance digests from the validated metadata:

- features: `8a7aad5809c7a21e614da7d836629309cda9c2de74553bf1fbc6934f7b07f5e2`
- returns: `c33a00cac4cf169f01e3ba5823a3f6d9bae17da5add5f8d5a3538d4142a0fabb`
- availability: `630de125ae9bc04cd0376404c7cff07f8e7d06c3bec2eece1b546e05959e292f`
- source provenance: `1e78ccf3162567e799b05a1c25dbe12a1c4c37e8e5a2abf2f9b95a70c380e2db`

The tracked frozen metadata
`docs/data_quality_v4_rebuild_2018_2024_metadata.json` was also passed
explicitly to `load_cache_v4` with the same feature/returns/availability
paths. It validates successfully and has SHA-256
`2c9db28deebe7e6b08f4ffedf65c3cdb51a78cfd7ee7d6580f76a62cc424bdcb` and
source-provenance digest
`aa320222dca0a46b2a0730f17bb1665f31a70074aa3bafcc6bff58ca21618fad`.
The content digests, rows, indexes, schema and gaps match the cache-local
metadata, but the metadata files and source-provenance digests do not. This
is an explicit provenance distinction, not evidence that the metadata is
byte-identical. P1 must pin the body paths together with the tracked frozen
metadata path.

## P0-C shared replay evidence

The canonical contract hash is
`6f5beb7865fceac5ecbcfbb31dd11e8fdada02e1841fecac1c17e22377bb624f`.
All four paths use the same eligibility-mask hash:
`1d4c8cca449b63f4f23b249f83b644fc3e1980ce2d02ed0c0f42d8618b3fa69d`.

The full-grid replay produced:

- scheduled decisions: `52583`
- scorable four-bar blocks: `43255`
- eligible/executable blocks: `29650`
- feature-unavailable hold commitments: `13605`
- Spot-outcome excluded blocks: `9328`
- scored bars: `173020`
- actual strategy fills: `20077`
- B&H fills: `0`
- U0 fills: `21113`

For every strategy/B&H/U0/causal-teacher trajectory:

- `same scored mask = True`
- `same contract hash = True`
- `same eligibility-mask hash = True`
- `decision t -> fill t+1 -> returns[t+1:t+5] = True`
- incomplete tail is excluded

The causal teacher probe uses only the observed decision-bar Spot
`return[t]`/close observation before the fill at `t+1`; it does not use U0 or
delayed outcome returns. Its strategy PnL, B&H PnL, and U0 PnL are replay
diagnostics under this geometry and must not be interpreted as forecast
accuracy. In particular, U0 remains realized-future and diagnostic-only.

## P0-B integrated guard

The fixture runs chronological OOF with explicit origin/training-label masks
and provenance. Perturbing the label at row `t=4` leaves the OOF state and
causal-teacher action at `t=4` unchanged; the perturbation is allowed to alter
later training prefixes. The guard then rejects `hindsight_teacher`, `teacher`,
and `oracle` inventory sources, including a `policy_replay` path whose
provenance says `hindsight_oracle`.

This verifies the same-row future perturbation and hindsight-inventory
boundary when joined to P0-C, but it does not claim that the legacy full WM
pipeline has completed chronological OOF retraining. That remains the P0-B
promotion gap.
