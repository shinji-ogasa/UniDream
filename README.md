# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

BTCUSDT 15分足を対象に、World Model、Behavior Cloning、Imagination Actor-Critic を組み合わせて検証する暗号資産トレード研究プロジェクトです。現在のリポジトリは、BC prior 最適化を中心にした本線パイプラインへ絞り込んでいます。

## 概要

UniDream は、Hindsight Oracle で生成した教師ポジションを Behavior Cloning で模倣し、その方策を World Model 上の imagination rollouts で Actor-Critic fine-tune する研究コードです。

主な目的は、Buy & Hold をベンチマークとして、OOS の超過成績、Sharpe 改善、最大ドローダウン改善、collapse 回避を同時に評価することです。

現在の焦点:

- BC prior の安定化
- dual residual controller の調整
- feature-stress regime を使った stress-shift 系の検証
- 余計な後方互換コードを削った、読みやすい実験パイプラインの維持

## パイプライン

```text
OHLCV / features
  -> Walk-Forward split
  -> Hindsight Oracle
  -> World Model training
  -> Behavior Cloning
  -> Imagination Actor-Critic
  -> Validation selector
  -> Test backtest / M2 scorecard
```

各ステージの責務:

- `data`: Binance OHLCV の取得、特徴量計算、returns 整形
- `oracle`: hindsight oracle path と signal-aim teacher の生成
- `world_model`: Transformer ベースの latent dynamics 学習
- `bc`: Actor を oracle position に模倣させる初期化
- `ac`: World Model 上で imagination actor-critic fine-tune
- `selector`: validation split 上で候補方策を選択
- `test`: test split で backtest、PBO、M2 scorecard を出力

## ディレクトリツリー

```text
UniDream/
├── README.md
├── SPEC.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── smoke_test.yaml
│   ├── trading.yaml
│   ├── medium_v2.yaml
│   └── medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_*dualresanchor*.yaml
├── docs/
│   └── current_focus.md
└── unidream/
    ├── cli/
    │   └── train.py
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
    │   ├── bc_pretrain.py
    │   ├── critic.py
    │   └── imagination_ac.py
    ├── experiments/
    │   ├── fold_inputs.py
    │   ├── fold_runtime.py
    │   ├── wm_stage.py
    │   ├── bc_setup.py
    │   ├── bc_stage.py
    │   ├── ac_stage.py
    │   ├── val_selector_stage.py
    │   ├── test_stage.py
    │   └── train_app.py
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

最小 smoke test:

```bash
uv run python -m unidream.cli.train \
  --config configs/smoke_test.yaml \
  --start 2022-01-01 \
  --end 2023-06-01 \
  --device auto
```

現在のBC stress-regime branchをfold 4で実行:

```bash
uv run python -m unidream.cli.train \
  --config configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --folds 4 \
  --device auto
```

BCまでで止める:

```bash
uv run python -m unidream.cli.train \
  --config configs/smoke_test.yaml \
  --start 2022-01-01 \
  --end 2023-06-01 \
  --stop-after bc \
  --device auto
```

既存checkpointから再開:

```bash
uv run python -m unidream.cli.train \
  --config configs/smoke_test.yaml \
  --start 2022-01-01 \
  --end 2023-06-01 \
  --resume \
  --device auto
```

## 設定ファイル

現在残している設定は本線用だけです。

- `configs/smoke_test.yaml`: 短時間の動作確認用
- `configs/trading.yaml`: 標準設定
- `configs/medium_v2.yaml`: 旧基準の中規模設定
- `configs/*dualresanchor*.yaml`: 現在のBC prior / stress-regime検証用

## 生成物

実行時に以下が生成されます。

- `checkpoints/`: World Model、BC Actor、AC checkpoint、data cache、fold別出力
- `__pycache__/`: Python実行キャッシュ
- `.venv/`: `uv sync` が作る仮想環境

`checkpoints/` と `.venv/` はGit管理対象外です。checkpointは再実行で作り直す前提です。

## 依存

主要依存:

- `torch`: World Model / Actor-Critic / BC
- `numpy`: 数値計算
- `pandas`: 時系列データ処理
- `pandas-ta`: テクニカル特徴量
- `scikit-learn`: 評価・補助モデル
- `hmmlearn`: regime detection
- `requests`: Binance API取得
- `scipy`: 統計評価
- `pyyaml`: config読み込み

依存は [pyproject.toml](pyproject.toml) と [uv.lock](uv.lock) で固定します。

## 参考文献

- Hafner et al., "Mastering Diverse Domains through World Models", DreamerV3.
- Sutton and Barto, "Reinforcement Learning: An Introduction".
- Schulman et al., "Proximal Policy Optimization Algorithms".
- Bailey et al., "The Probability of Backtest Overfitting".
- Lopez de Prado, "Advances in Financial Machine Learning".
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling".

## 注意

このリポジトリは研究用途です。投資助言や実運用システムではありません。暗号資産市場は高リスクであり、バックテスト結果は将来成績を保証しません。
