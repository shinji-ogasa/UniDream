# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

BTCUSDT 15分足を対象に、World Model、Behavior Cloning、Imagination Actor-Critic を組み合わせて検証する暗号資産トレード研究プロジェクトです。

## 概要

UniDream は、Hindsight Oracle で生成した教師ポジションを Behavior Cloning で模倣し、その方策を World Model 上の imagination rollouts で Actor-Critic fine-tune する研究コードです。

主な目的は、Buy & Hold をベンチマークとして、OOS の超過成績、Sharpe 改善、最大ドローダウン改善、collapse 回避を同時に評価することです。

## パイプライン

```text
OHLCV / features
  -> Walk-Forward split
  -> Hindsight Oracle
  -> Transformer World Model
  -> WM predictive state bundle (return / vol / drawdown)
  -> Behavior Cloning (route head + inventory recovery + state machine)
  -> Imagination Actor-Critic
  -> Validation selector
  -> Test backtest / M2 scorecard / PBO / Regime report
```

各ステージの責務:

- `data`: Binance OHLCV / funding / OI / mark price 取得、特徴量計算、returns 整形
- `oracle`: hindsight oracle path、signal_aim teacher、feature_stress / dual teacher の生成
- `world_model`: Transformer ベースの latent dynamics、predictive head (return / vol / drawdown / regime)
- `bc`: Actor を oracle position に模倣、route head・inventory recovery controller・state machine gate を学習
- `ac`: World Model 上で imagination actor-critic fine-tune
- `selector`: validation split で adjust-rate scale を選択、collapse guard / M2 scorecard を反映
- `test`: test split で backtest、PBO、Deflated Sharpe、HMM regime、M2 scorecard を出力

## ディレクトリツリー

```text
UniDream/
├── README.md
├── SPEC.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── configs/
├── documents/
├── docs/
└── unidream/
    ├── device.py
    ├── cli/
    │   ├── train.py
    │   ├── route_probe.py
    │   ├── wm_probe.py
    │   ├── transition_advantage_probe.py
    │   └── ac_candidate_q_probe.py
    ├── data/
    │   ├── dataset.py
    │   ├── download.py
    │   ├── features.py
    │   └── oracle.py
    ├── world_model/
    │   ├── encoder.py
    │   ├── ensemble.py
    │   ├── train_wm.py
    │   └── transformer.py
    ├── actor_critic/
    │   ├── actor.py
    │   ├── critic.py
    │   ├── state_action_critic.py
    │   ├── bc_pretrain.py
    │   └── imagination_ac.py
    ├── experiments/
    │   ├── runtime.py
    │   ├── train_app.py
    │   ├── train_pipeline.py
    │   ├── train_reporting.py
    │   ├── wfo_runtime.py
    │   ├── fold_runtime.py
    │   ├── fold_inputs.py
    │   ├── oracle_stage.py
    │   ├── oracle_teacher.py
    │   ├── oracle_post.py
    │   ├── transition_advantage.py
    │   ├── regime_runtime.py
    │   ├── predictive_state.py
    │   ├── wm_stage.py
    │   ├── bc_setup.py
    │   ├── bc_stage.py
    │   ├── ac_stage.py
    │   ├── val_selector_stage.py
    │   ├── test_stage.py
    │   └── m2.py
    └── eval/
        ├── backtest.py
        ├── pbo.py
        ├── regime.py
        └── wfo.py
```

## セットアップ

前提:

- Python 3.12 以上
- `uv`
- macOS Apple Silicon なら `mps`、NVIDIA 環境なら `cuda`、それ以外は `cpu`

依存関係の同期:

```bash
uv sync
```

動作確認:

```bash
uv run python -m unidream.cli.train --help
```

## 使い方

### 本流の実行

全ステージを通して 1 fold を実行する:

```bash
uv run python -m unidream.cli.train \
  --config configs/trading.yaml \
  --start 2018-01-01 \
  --end 2024-01-01 \
  --folds 4 \
  --device auto
```

### ステージ単位の実行

`--start-from` と `--stop-after` で `wm` / `bc` / `ac` / `test` を切り出せる。既存 checkpoint がある場合は `--resume` で自動ロードする。

```bash
# WM だけ学習
uv run python -m unidream.cli.train --config configs/trading.yaml \
  --start 2018-01-01 --end 2024-01-01 --folds 4 \
  --stop-after wm --device auto

# BC だけ学習 (WM は既存 checkpoint を再利用)
uv run python -m unidream.cli.train --config configs/trading.yaml \
  --start 2018-01-01 --end 2024-01-01 --folds 4 \
  --start-from bc --stop-after bc --resume --device auto

# AC だけ学習 (WM/BC は既存 checkpoint を再利用)
uv run python -m unidream.cli.train --config configs/trading.yaml \
  --start 2018-01-01 --end 2024-01-01 --folds 4 \
  --start-from ac --stop-after ac --resume --device auto

# Test backtest だけ再実行
uv run python -m unidream.cli.train --config configs/trading.yaml \
  --start 2018-01-01 --end 2024-01-01 --folds 4 \
  --start-from test --resume --device auto
```

### 開発時の実行

短時間で動作確認するための smoke run:

```bash
uv run python -m unidream.cli.train \
  --config configs/smoke_test.yaml \
  --start 2022-01-01 \
  --end 2023-06-01 \
  --device auto
```

中断したジョブを再開する:

```bash
uv run python -m unidream.cli.train \
  --config configs/trading.yaml \
  --start 2018-01-01 --end 2024-01-01 --folds 4 \
  --resume --device auto
```

cost profile を切り替える (config の `cost_profiles` ブロックから選択):

```bash
uv run python -m unidream.cli.train --config configs/trading.yaml \
  --start 2018-01-01 --end 2024-01-01 --folds 4 \
  --cost-profile stress --device auto
```

### 診断 CLI

BC checkpoint を AC に渡す前のチェックや、世界モデルの予測力評価に使う。

```bash
# Route 分類性能 (CE / Macro-F1 / per-route recall)
uv run python -m unidream.cli.route_probe --config <bc-config> --folds 4

# 行動候補別の realized advantage
uv run python -m unidream.cli.transition_advantage_probe --config <bc-config> --folds 4

# WM linear probe (return / vol / drawdown / regime の予測力)
uv run python -m unidream.cli.wm_probe --config <wm-probe-config> --folds 4

# State-action critic Q(s,a) の rank IC / top-decile advantage
uv run python -m unidream.cli.ac_candidate_q_probe --config <bc-config> --folds 4
```

## 設定ファイル

- `configs/smoke_test.yaml`: 短時間の動作確認用
- `configs/trading.yaml`: 標準設定

`configs/` には他にも実験用の YAML が入っている (`bcplan*`, `bc_*`, `medium_l1_*`, `wm_probe_*`)。実験経緯は `documents/` を参照。

## 生成物

実行時に以下が生成される。

- `checkpoints/<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt}`: 各ステージの checkpoint
- `checkpoints/data_cache/`: feature / returns の parquet キャッシュ
- `documents/logs/`, `documents/route_probe/`, `documents/wm_probe/`, `documents/ac_candidate_q/`: 各種ログ・診断出力
- `__pycache__/`: Python 実行キャッシュ
- `.venv/`: `uv sync` が作る仮想環境

`checkpoints/` と `.venv/` は Git 管理対象外。checkpoint は再実行で作り直す前提。

## 依存

主要依存:

- `torch`: World Model / Actor-Critic / BC
- `numpy`: 数値計算
- `pandas`: 時系列データ処理
- `pandas-ta`: テクニカル特徴量
- `scikit-learn`: 評価・補助モデル・linear probe
- `hmmlearn`: regime detection
- `requests`: Binance API 取得
- `scipy`: 統計評価
- `pyyaml`: config 読み込み

依存は [pyproject.toml](pyproject.toml) と [uv.lock](uv.lock) で固定する。

## 参考文献

- Hafner et al., "Mastering Diverse Domains through World Models", DreamerV3.
- Sutton and Barto, "Reinforcement Learning: An Introduction".
- Schulman et al., "Proximal Policy Optimization Algorithms".
- Fujimoto and Gu, "A Minimalist Approach to Offline Reinforcement Learning" (TD3+BC).
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (CQL).
- Bailey et al., "The Probability of Backtest Overfitting".
- Lopez de Prado, "Advances in Financial Machine Learning".
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling".

## 注意

このリポジトリは研究用途。投資助言や実運用システムではない。暗号資産市場は高リスクであり、バックテスト結果は将来成績を保証しない。
