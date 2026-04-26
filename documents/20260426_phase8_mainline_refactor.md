# Phase 8 Mainline Refactor Verification

Date: 2026-04-26

## Scope

Phase 8 safe baseline is now the single default training path.

Removed / retired from source config path:

- Old experiment YAMLs under `configs/`.
- PPO / QDT config blocks by deleting old config files.
- Feature dual teacher implementation.
- Feature ridge / feature linear teacher implementation.
- Feature dual branch in `wm_probe`.
- `train` CLI overrides not needed for the main path: `--symbol`, `--checkpoint_dir`, `--cost-profile`.

Kept intentionally:

- AC pipeline and restricted actor hooks.
- Route / inventory recovery / state-machine controls.
- Feature-stress regime diagnostics for `eval.regime_source: feature_stress_tri`.
- Diagnostic CLIs.

## Main Config

`configs/trading.yaml` now contains the adopted Phase 8 safe baseline.

Default run:

```bash
uv run python -m unidream.cli.train --folds 4 --device auto
```

## Verification

Command:

```bash
uv run python -m unidream.cli.train --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 4 --start-from test --resume --device cpu
```

Result on fold 4:

```text
AlphaEx: +0.89 pt/yr
SharpeDelta: -0.011
MaxDDDelta: -1.58 pt
short: 16%
flat: 84%
turnover: 2.63
recovery gate active: 1.0%
```

This matches the adopted Phase 8 safe baseline range.

## Notes

During verification, removing `actions.n` exposed an implicit default in `WorldModelTrainer` that created a 5-action IDM head. The implementation now derives `n_actions` from `actions.values` when `actions.n` is absent, so the config can stay smaller without changing checkpoint compatibility.
