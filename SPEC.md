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

学習entrypointは以下だけを受け付ける。引数を省略した場合は、現行開発仕様（`configs/trading.yaml`、seed `7`、`device auto`）で実行する。

```bash
uv run python -m unidream.cli.train
```

明示指定も可能:

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
- データ取得・cache検証を通過してからconfigured checkpoint directoryをcleanし、古いartifactを参照しない。
- cacheは完全一致するtagとsidecar契約だけを読み、wildcard fallbackは行わない。
- feature cacheは required columns、時系列 index の整合性、有限値、`include_funding/include_oi/include_mark` を検証する。

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
  - resolved config と config/source/data SHA256
  - Git commit / dirty state
  - seed / device / library versions
  - cache tag / cache contract / required artifacts
  - selected folds
  - WM/BC/AC semantic checkpoint SHA256
- `fold_<n>/world_model.pt`: WM state + stage/fold/run/config/source/data provenance
- `fold_<n>/bc_actor.pt`: BC actor + stage/fold/run/config/source/data provenance
- `fold_<n>/ac.pt`: AC state + provenance + Actor全推論runtime snapshot + validation selectorの最終推論設定

run IDはconfig、source、data、seedから生成する。PyTorch checkpointはZIP file hashではなくtensor内容のsemantic hashで比較する。replay CLIは、`run_manifest.json` が completed で、各checkpointのmetadataがmanifestと一致する場合にだけロードする。
読み込み時はmanifest内のresolved config hashとも照合し、別configでの黙ったreplayを拒否する。

## Leak Discipline

- trainでWM/BC/ACをfitする。
- validationでcheckpoint/inference selectorを選ぶ。
- testはreport-only。
- fold境界、期間、特徴系列の有無はYAMLに固定する。
- latest holdout configは`configs/plan011_overlay_actor_v31_holdout.yaml`。
