# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

BTCUSDT 15分足を対象に、Transformer World Model、Behavior Cloning、Imagination Actor-Critic を組み合わせて検証する暗号資産トレード研究プロジェクト。

## 概要

UniDream は、Hindsight Oracle で生成した教師ポジションを Behavior Cloning で模倣し、その方策を World Model 上の imagination rollouts で Actor-Critic fine-tune する研究コード。Buy & Hold をベンチマークとして、OOS の超過成績、Sharpe 改善、最大ドローダウン改善、collapse 回避を同時に評価する。

現在の採用版は no-leak Plan004 residual BC/AC。単一actor圧縮ではなく、階層base policyを固定し、realized residual advantage をBCで学習したうえで、validation-only の threshold / hold / cooldown extraction を行う。`configs/trading.yaml` では本流 `train` の中で WM → BC → AC → Plan004 extraction → Test を実行し、fold配下に `world_model.pt` / `bc_actor.pt` / `ac.pt` / `plan004_policy.npz` を保存する。

## パイプライン

```text
OHLCV / features
  -> Walk-Forward split
  -> Hindsight Oracle / signal_aim teacher
  -> Transformer World Model
  -> WM predictive state bundle (return / vol / drawdown)
  -> Behavior Cloning (route head + inventory recovery + state machine)
  -> Imagination Actor-Critic
  -> Validation selector
  -> Plan004 residual BC/AC extraction (fixed hierarchy + residual extraction)
  -> Test backtest / M2 scorecard / PBO / regime report
  -> Space bundle export
```

## ステージごとの責務

- `data`: Binance OHLCV / funding / OI / mark price 取得、特徴量計算、returns 整形
- `oracle`: signal_aim teacher で aim positions を生成、smooth aim、soft labels
- `world_model`: Transformer ベースの latent dynamics、predictive head (return / vol / drawdown / regime)
- `bc`: Actor を oracle position に模倣、route head・inventory recovery controller・state machine gate を学習
- `ac`: World Model 上で imagination actor-critic fine-tune (本流は critic-only / 制限付き actor 解凍)
- `selector`: validation split で adjust-rate scale を選択、collapse guard / M2 scorecard を反映
- `research`: Plan004 residual BC/AC など、採用候補の検証本体
- `deploy`: 本流checkpointからHF Spaces向けbundleを作るexport処理
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
│   └── trading.yaml
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
    ├── deploy/
    │   └── plan004_space_bundle.py
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
    ├── research/
    │   └── plan004_residual_bc_ac.py
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
- NVIDIA環境では学習・検証コマンドは明示的に `--device cuda` を付ける。CPU実験は標準運用では使わない。

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

`--config` のデフォルトは `configs/trading.yaml`。標準運用ではCUDAを明示して走らせる。

```bash
uv run python -m unidream.cli.train --device cuda
```

fold を絞る場合:

```bash
uv run python -m unidream.cli.train --folds 13 --seed 7 --device cuda
```

### ステージ単位の実行

`--start-from` と `--stop-after` で `wm` / `bc` / `ac` / `test` を切り出せる。既存 checkpoint がある場合は `--resume` で自動ロード。

```bash
# WM だけ学習
uv run python -m unidream.cli.train --stop-after wm --device cuda

# BC だけ学習 (WM は既存 checkpoint を再利用)
uv run python -m unidream.cli.train --start-from bc --stop-after bc --resume --device cuda

# AC だけ学習 (WM/BC は既存 checkpoint を再利用)
uv run python -m unidream.cli.train --start-from ac --stop-after ac --resume --device cuda

# Test backtest だけ再実行
uv run python -m unidream.cli.train --start-from test --resume --device cuda
```

### 開発時の実行

中断したジョブを再開する:

```bash
uv run python -m unidream.cli.train --resume --device cuda
```

### Plan004 residual BC/AC

no-leak residual BC/AC の全14fold検証。これは本流Plan004ステージと同じfold-localロジックを高速に監査するための診断CLI。

```bash
uv run python -m unidream.cli.plan004_noncompressive_bc_ac_probe \
  --selection-stress-mode primary \
  --output-json codex_outputs/plan004_current_no_leak_allfold.json \
  --output-md codex_outputs/plan004_current_no_leak_allfold.md
```

HF Spaces 推論bundleのexport。`train` が生成した同一fold配下の `world_model.pt` / `bc_actor.pt` / `ac.pt` / `plan004_policy.npz` をbundle化する。

```bash
uv run python -m unidream.cli.export_plan004_space_bundle \
  --checkpoint-dir checkpoints/main_plan004_residual_bc_ac_s007 \
  --fold 13 \
  --seed 7 \
  --device cuda \
  --output-dir C:/Users/Sophie/Documents/UniDream/unidream-space/bundles/current
```

### 診断 CLI

BC checkpoint を AC に渡す前のチェックや、世界モデルの予測力評価に使う。

```bash
# Route 分類性能 (CE / Macro-F1 / per-route recall)
uv run python -m unidream.cli.route_probe

# 行動候補別の realized advantage
uv run python -m unidream.cli.transition_advantage_probe

# WM linear probe (return / vol / drawdown / regime の予測力)
uv run python -m unidream.cli.wm_probe

# State-action critic Q(s,a) の rank IC / top-decile advantage
uv run python -m unidream.cli.ac_candidate_q_probe
```

## 生成物

実行時に以下が生成される。

- `checkpoints/<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt, plan004_policy.npz, plan004_summary.json}`: 各ステージの checkpoint と採用Plan004 policy
- `checkpoints/data_cache/`: feature / returns の parquet キャッシュ
- `documents/logs/`, `documents/route_probe/`, `documents/wm_probe/`, `documents/ac_candidate_q/`: ログ・診断出力
- `codex_outputs/`: Plan003/Plan004 などの実験JSON/Markdown/log
- `docs/`: 採用判断・実験サマリ・production移行レポート
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

World Model / Imagination AC:

- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023) — symlog / twohot critic / λ-return / EMA target / Imagination AC のベース
- Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2, 2021) — discrete latent / KL balancing
- Vaswani et al., "Attention Is All You Need" (2017) — Transformer dynamics
- Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (ICM, 2017) — Inverse Dynamics Model (IDM auxiliary loss)

Imitation Learning / Behavior Cloning:

- Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network" (1989) — Behavior Cloning の起源
- Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger, 2011) — BC の covariate shift / inventory state 分布ズレに対応
- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT, 2023) — `chunk_size=4` の action chunking

Offline RL / Advantage-weighted Policy Extraction:

- Peng et al., "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning" (AWR, 2019) — advantage 重み付き BC
- Nair et al., "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets" (AWAC, 2020) — `transition_advantage_relabel` / `route_advantage_weight_coef` の根拠
- Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning" (IQL, 2022) — advantage-weighted BC で policy 抽出、UniDream BC 段階の中心的な参考設計
- Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration" (BCQ, 2019) — 分布外 action 制約
- Kumar et al., "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction" (BEAR, 2019)
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (CQL, 2020) — `state_action_critic` の CQL-lite penalty
- Fujimoto and Gu, "A Minimalist Approach to Offline Reinforcement Learning" (TD3+BC, 2021) — `td3bc_alpha` / `prior_kl_coef` の根拠
- Springenberg et al., "Offline Actor-Critic Reinforcement Learning Scales to Large Models" (ICML 2024) — restricted actor unlock の根拠
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)

Reinforcement Learning 一般:

- Sutton and Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE, 2016)

Backtest / 評価:

- Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest Overfitting" (2014) — PBO
- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014) — DSR
- Lopez de Prado, "Advances in Financial Machine Learning" (2018) — WFO / combinatorial CV / overfitting
- Pardo, "The Evaluation and Optimization of Trading Strategies" (2008) — Walk-Forward Optimization
- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle" (1989) — HMM regime detection

## 注意

このリポジトリは研究用途。投資助言や実運用システムではない。暗号資産市場は高リスクであり、バックテスト結果は将来成績を保証しない。
