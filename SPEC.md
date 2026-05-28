# UniDream Spec

## Scope

現行 scope は Plan009 depth calibrator の検証・bundle export・Space 推論。

```text
OHLCV / returns
  -> shifted trailing-return features
  -> past-only guard
  -> validation-gated depth
  -> execution compression
  -> cost stress evaluation
  -> Space bundle export
```

## Entrypoints

Plan009 dev evaluation:

```bash
PYTHONWARNINGS=ignore uv run python -u -m unidream.cli.plan009_depth_calibrator_probe \
  --config configs/trading.yaml \
  --folds 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --seed 7 \
  --val-dd-target -4.8 \
  --safety-multiplier 2.0 \
  --max-depth-cap 0.94
```

Space bundle export:

```bash
uv run python -m unidream.cli.export_plan009_depth_calibrator_bundle \
  --config configs/trading.yaml \
  --output-dir /Users/sophie/Documents/UniDream/unidream-space/bundles/current
```

## Current Bundle Contract

- `bundle_type`: `plan009_depth_calibrator`
- input: raw 15m candles or aligned `features + returns`
- runtime history: latest request is split into current 60d trading window plus preceding guard history
- signals: shifted trailing-return features only
- output: absolute position in `[0.0, 1.0]`
- benchmark: `1.0`

## Current Dev Metrics

`docs_local/20260528_plan009_gap16_next_mindelta010_full.json`:

```text
folds: 0-12
cost_x1 pass +3/-3: 13/13
Alpha median: +16.025pt
Alpha worst: +4.690pt
MaxDD worst: -3.026pt
TO mean: 24.956
cost_x2 pass: 9/13
cost_x3 pass: 6/13
```

## Leak Discipline

- Runtime features are shifted; values at bar `t` use information available before `t`.
- Depth settings are chosen from validation-gated development probes.
- Test metrics in fold0-12 are report-only.
- fold0-12 is a development set, not a pristine holdout.

## Generated Files

- `docs_local/*.json`
- `docs_local/*.md`
- `checkpoints/data_cache/*_features.parquet`
- `checkpoints/data_cache/*_returns.parquet`
- `unidream-space/bundles/current/*`

`checkpoints/`, `docs_local/`, and `.venv/` are local generated outputs.
