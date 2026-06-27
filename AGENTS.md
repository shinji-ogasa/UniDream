# UniDream Development Guide

現行 mainline は Plan011 v31 の Transformer WM -> BC -> Imagination AC -> Test pipeline。
詳細な実行契約は `SPEC.md`、使い方と結果概要は `README.md` を参照する。

## Current Scope

- 対象は BTCUSDT 15分足。
- 学習 entrypoint は `uv run python -m unidream.cli.train --config <yaml> --seed <n> --device <device>`。
- CLI から期間・fold・cost・checkpoint path は上書きしない。すべて YAML に固定する。
- mainline では checkpoint warm-start、途中 stage 開始、legacy Plan004/009 互換を使わない。
- 現行 config は `configs/trading.yaml`、`configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml`、`configs/plan011_overlay_actor_v31_holdout.yaml` のみ。

## Validation

- 変更後は最低限 `uv run python -m unittest discover -s tests -v` と `git diff --check` を実行する。
- checkpoint loader や bundle export を触った場合は、現行 v31 checkpoint を使う smoke も実行する。
- test split は report-only。validation selector の選択を test 結果で変更しない。
