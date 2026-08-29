# Official-source schema-v4 cache rebuild

This report reads no model results. It uses only Binance-owned sources and never interpolates missing bars.

- Cache output directory: `/tmp/unidream-v4-actual`
- Cache tag: `BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official`
- Scope: `[2018-01-01, 2024-01-01)` / `15m`
- Source probe status: **PASS**
- Source ledger: `docs/data_quality_v4_rebuild_2018_2024.jsonl`

## Official source probe

| Source | Probe responses | HTTP 200 | HTTP 404 |
| --- | ---: | ---: | ---: |
| `spot_klines` | 4 | 4 | 0 |
| `um_mark_price_klines` | 4 | 1 | 3 |
| `um_funding_rate` | 4 | 1 | 3 |

UM mark/funding archives before `2020-01-01` are treated as unavailable and their masks are false; no future value is backfilled into that period.

## Rebuild status

- Status: **GENERATED_WITH_EXPLICIT_SPOT_GAPS**
- Expected full-grid bars: `210336`
- Observed Spot bars: `209805`
- REST-recovered Spot bars: `80`
- Unresolved Spot bars: `531`
- Quarantined off-grid Spot bars: `81`
- Spot gap runs before REST recovery: `29`
- Spot gap runs after REST recovery/quarantine: `30`
- Computed feature rows: `173111`
- Metadata schema: `4`
- Schema digest: `1c1c41a9aca3e8af22b357a8483ea6419745ee4b24c10c09c47289df3744c616`
- Source/provenance digest: `ea1f0fba7889d6ef3737b91aada30361d2f8c6242ce6ac2fbb96fd9523ebe677`
- Feature content digest: `8a7aad5809c7a21e614da7d836629309cda9c2de74553bf1fbc6934f7b07f5e2`
- Returns content digest: `c33a00cac4cf169f01e3ba5823a3f6d9bae17da5add5f8d5a3538d4142a0fabb`
- Availability content digest: `630de125ae9bc04cd0376404c7cff07f8e7d06c3bec2eece1b546e05959e292f`

## Availability coverage

- Funding as-of available: `140255/210336` (66.68%)
- Causal mark exact available: `139485/210336` (66.32%)
- Funding and mark both available: `139485/210336` (66.32%)
- Spot, funding, and mark all available: `139333/210336` (66.24%)
- Feature body rows with all three flags true: `119849/173111`
- Feature body rows without all three flags: `53262`
- Observed Spot rows minus feature rows: `36694`
- Feature row policy: compute causal features on observed contiguous Spot segments; do not filter body by external masks
- Feature row reduction: rolling indicator warmup and unresolved Spot gaps split segments; each segment drops its own invalid warmup rows

The v4 body excludes unresolved Spot rows; the separate full-grid sidecar marks them `spot_bar_observed=false`. The 80 bars recovered by official REST are included only when the raw Spot merge and causal feature computation succeed. 81 off-grid Spot rows were quarantined without timestamp remapping. External masks remain metadata and are not mixed into model inputs.
Full17 v4 training promotion remains fail-closed until the training dataset consumes spot, funding, and mark availability masks for every sequence window.

No model result was read and no v3 file was overwritten.
