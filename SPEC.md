# UniDream Spec

## Scope

現行mainlineは Plan011 v31 の Transformer WM -> BC -> Imagination AC -> Test pipeline。

```text
OHLCV / funding / mark
  -> exact cache key
  -> Walk-Forward split
  -> Hindsight Oracle
  -> Transformer World Model
  -> WM predictive state
  -> Behavior Cloning
  -> Imagination Actor-Critic
  -> validation selector
  -> test-only backtest
```

## Training Contract

学習entrypointは以下だけを受け付ける。

```bash
uv run python -m unidream.cli.train \
  --config configs/trading.yaml \
  --seed 7 \
  --device cuda
```

- 期間・fold・cost・checkpoint/cache pathはYAMLからのみ読む。
- pipelineは常にWMからTestまで順番に実行する。
- `--resume`, `--start-from`, `--stop-after`, `--start`, `--end`, `--folds`は廃止。
- mainlineのcheckpoint warm-startとPlan004 overrideは廃止。
- 実行前にconfigured checkpoint directoryをcleanし、古いartifactを参照しない。
- cacheは完全一致するtagだけを読み、wildcard fallbackは行わない。

## Required YAML

```yaml
run:
  start: "2018-01-01"
  end: "2024-01-01"
  folds: [0, 1, 2]
  clean_checkpoint_dir: true
  deterministic_algorithms: true

data:
  include_funding: true
  include_oi: false
  include_mark: true

logging:
  checkpoint_dir: checkpoints/example_s007
  cache_dir: checkpoints/data_cache
```

`run.folds: all`も許可する。その他のrun keyや旧互換fieldはvalidation errorにする。

## Reproducibility Artifacts

各run directoryに以下を保存する。

- `resolved_config.yaml`: cost profile解決後の完全config
- `run_manifest.json`
  - deterministic run ID
  - config/source/data SHA256
  - Git commit / dirty state
  - seed / device / library versions
  - selected folds
  - WM/BC/AC semantic checkpoint SHA256
- `fold_<n>/world_model.pt`
- `fold_<n>/bc_actor.pt`
- `fold_<n>/ac.pt`

run IDはconfig、source、data、seedから生成する。PyTorch checkpointはZIP file hashではなくtensor内容のsemantic hashで比較する。

## Leak Discipline

- trainでWM/BC/ACをfitする。
- validationでcheckpoint/inference selectorを選ぶ。
- testはreport-only。
- fold境界、期間、特徴系列の有無はYAMLに固定する。
- latest holdout configは`configs/plan011_overlay_actor_v31_holdout.yaml`。
