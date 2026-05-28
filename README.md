# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

BTCUSDT 15分足を対象に、Buy & Hold に対する超過 Alpha と最大ドローダウン改善を検証するトレード研究プロジェクト。

## 概要

現行仕様は Plan009 depth calibrator。raw returns から shifted trailing-return feature だけで past-only guard を作り、validation-gated depth と軽い execution compression をかけて position を出す。リアルタイムデモもこの仕様を `unidream-space/bundles/current` で動かす。

fold0-12 の開発評価では `AlphaEx >= +3pt && MaxDDDelta <= -3pt` を `13/13` pass。現行 compression 版の集計は `Alpha median +16.025pt`、`Alpha worst +4.690pt`、`MaxDD worst -3.026pt`、`TO mean 24.956`。ただし cost stress はまだ課題で、`cost_x2` は `9/13`、`cost_x3` は `6/13` pass。

fold0-12 は開発セットであり、pristine holdout の主張ではない。

## パイプライン

```text
OHLCV
  -> feature / raw returns cache
  -> WFO fold0-12 development split
  -> shifted trailing-return past-only guard
  -> validation-gated depth calibration
  -> execution compression
  -> cost stress / drawdown / alpha report
  -> Space bundle export
```

## ステージごとの責務

- `data`: Binance OHLCV / funding / mark price 取得、特徴量計算、returns 整形
- `plan009_depth_calibrator_probe`: fold0-12 で depth calibration と stress 集計を再現
- `export_plan009_depth_calibrator_bundle`: Space 用 current bundle を生成
- `unidream-space`: FastAPI 推論 runtime。現行 bundle のみで推論し、UniDream 本体APIには接続しない
- `docs_local`: ローカル実験結果の JSON / Markdown 出力先

## ディレクトリツリー

```text
UniDream/
├── README.md
├── SPEC.md
├── pyproject.toml
├── uv.lock
├── configs/
│   └── trading.yaml
├── docs_local/
└── unidream/
    ├── cli/
    │   ├── plan009_depth_calibrator_probe.py
    │   ├── plan009_component_probe.py
    │   ├── plan009_depth_learner_probe.py
    │   ├── plan009_guard_student_probe.py
    │   ├── plan009_guard_sweep_probe.py
    │   └── export_plan009_depth_calibrator_bundle.py
    ├── data/
    │   ├── dataset.py
    │   ├── download.py
    │   ├── features.py
    │   └── oracle.py
    ├── eval/
    │   ├── backtest.py
    │   ├── pbo.py
    │   └── regime.py
    └── research/
        └── past-only guard research helpers
```

## セットアップ

前提:

- Python 3.12 以上
- `uv`

依存関係の同期:

```bash
uv sync
```

動作確認:

```bash
uv run python -m unidream.cli.plan009_depth_calibrator_probe --help
```

## 使い方

### 現行モデルの再現

fold0-12 の Plan009 depth calibrator を再現する:

```bash
PYTHONWARNINGS=ignore uv run python -u -m unidream.cli.plan009_depth_calibrator_probe \
  --config configs/trading.yaml \
  --folds 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --seed 7 \
  --val-dd-target -4.8 \
  --safety-multiplier 2.0 \
  --max-depth-cap 0.94 \
  --output-json docs_local/20260528_plan009_depth_calibrator_f0_12_m48_x2_cap094.json \
  --output-md docs_local/20260528_plan009_depth_calibrator_f0_12_m48_x2_cap094.md
```

現行 compression 版の確認結果:

```text
docs_local/20260528_plan009_gap16_next_mindelta010_full.json
```

### Space bundle export

現行 bundle を `unidream-space` に再生成する:

```bash
uv run python -m unidream.cli.export_plan009_depth_calibrator_bundle \
  --config configs/trading.yaml \
  --output-dir /Users/sophie/Documents/UniDream/unidream-space/bundles/current
```

生成後、`unidream-space` 側で sample parity を確認する:

```bash
cd /Users/sophie/Documents/UniDream/unidream-space
PYTHONPATH=/Users/sophie/Documents/UniDream/unidream-space \
  uv run --with-requirements requirements.txt python -m backend.verify_bundle \
  --bundle-dir bundles/current \
  --device cpu \
  --tolerance 0.000001
```

### 実験コマンド

Plan009 の部品検証:

```bash
uv run python -m unidream.cli.plan009_component_probe --help
uv run python -m unidream.cli.plan009_guard_sweep_probe --help
uv run python -m unidream.cli.plan009_guard_student_probe --help
uv run python -m unidream.cli.plan009_depth_learner_probe --help
```

## 生成物

- `docs_local/*.json`: 実験結果、fold 別 stress、集計
- `docs_local/*.md`: 実験サマリ
- `checkpoints/data_cache/`: feature / returns の parquet キャッシュ
- `/Users/sophie/Documents/UniDream/unidream-space/bundles/current`: Space current bundle

`checkpoints/`、`docs_local/`、`.venv/` はローカル生成物。

## 依存

主要依存:

- `numpy`: 数値計算
- `pandas`: 時系列データ処理
- `pandas-ta`: テクニカル特徴量
- `scikit-learn`: 補助モデル・検証
- `requests`: Binance API 取得
- `pyyaml`: config 読み込み

依存は [pyproject.toml](pyproject.toml) と [uv.lock](uv.lock) で固定する。

## 注意

このリポジトリは研究用途。投資助言や実運用システムではない。暗号資産市場は高リスクであり、バックテスト結果は将来成績を保証しない。
