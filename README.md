# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

BTCUSDT 15分足を対象に、Transformer World Model、Behavior Cloning、Imagination Actor-Critic を組み合わせて検証する暗号資産トレード研究プロジェクト。

## 概要

UniDream は、Hindsight Oracle で生成した教師ポジションを Behavior Cloning で模倣し、その方策を World Model 上の imagination rollouts で Actor-Critic fine-tune する研究コード。Buy & Hold をベンチマークとして、OOS の超過成績、Sharpe 改善、最大ドローダウン改善、collapse 回避を同時に評価する。

研究基盤として WM → BC → AC の学習パイプラインと各種診断 CLI を保持している。一方、リアルタイムデモの現行採用版は checkpoint export ではなく Plan009 depth calibrator bundle。raw returns から shifted trailing-return feature だけで past-only guard を作り、validation-gated depth と軽い execution compression をかけて position を出す。

Plan009 compression 版の fold0-12 開発評価では `AlphaEx >= +3pt && MaxDDDelta <= -3pt` を `13/13` pass。集計は `Alpha median +16.025pt`、`Alpha worst +4.690pt`、`MaxDD worst -3.026pt`、`TO mean 24.956`。ただし cost stress はまだ課題で、`cost_x2` は `9/13`、`cost_x3` は `6/13` pass。fold0-12 は開発セットであり、pristine holdout の主張ではない。

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
  -> Test backtest / M2 scorecard / PBO / regime report
  -> Plan009 depth calibrator / execution compression validation
  -> Space bundle export
```

## ステージごとの責務

- `data`: Binance OHLCV / funding / OI / mark price 取得、特徴量計算、returns 整形
- `oracle`: signal_aim teacher で aim positions を生成、smooth aim、soft labels
- `world_model`: Transformer ベースの latent dynamics、predictive head (return / vol / drawdown / regime)
- `bc`: Actor を oracle position に模倣、route head・inventory recovery controller・state machine gate を学習
- `ac`: World Model 上で imagination actor-critic fine-tune
- `selector`: validation split で selector / scorecard を評価
- `plan009`: current demo 用 depth calibrator、execution compression、cost stress 検証
- `deploy`: Plan009 dev candidate を HF Spaces 向け bundle として export
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
├── docs/
├── docs_local/
├── documents/
└── unidream/
    ├── device.py
    ├── cli/
    │   ├── train.py
    │   ├── plan009_depth_calibrator_probe.py
    │   ├── plan009_component_probe.py
    │   ├── plan009_depth_learner_probe.py
    │   ├── plan009_guard_student_probe.py
    │   ├── plan009_guard_sweep_probe.py
    │   ├── export_plan009_depth_calibrator_bundle.py
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
    ├── research/
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
uv run python -m unidream.cli.plan009_depth_calibrator_probe --help
```

## 使い方

### 本トレーニング

`--config` のデフォルトは `configs/trading.yaml`。WM → BC → AC → Test まで通す本トレーニングはこのコマンドで実行する。

```bash
uv run python -m unidream.cli.train \
  --config configs/trading.yaml \
  --seed 7 \
  --device cuda
```

fold を絞る場合:

```bash
uv run python -m unidream.cli.train \
  --config configs/trading.yaml \
  --folds 13 \
  --seed 7 \
  --device cuda
```

### ステージ単位の実行

`--start-from` と `--stop-after` で `wm` / `bc` / `ac` / `test` を切り出せる。既存 checkpoint がある場合は `--resume` で自動ロード。

```bash
# WM だけ学習
uv run python -m unidream.cli.train --stop-after wm --device cuda

# BC だけ学習
uv run python -m unidream.cli.train --start-from bc --stop-after bc --resume --device cuda

# AC だけ学習
uv run python -m unidream.cli.train --start-from ac --stop-after ac --resume --device cuda

# Test backtest だけ再実行
uv run python -m unidream.cli.train --start-from test --resume --device cuda
```

### Plan009 current demo の再現

リアルタイムデモ採用中の Plan009 depth calibrator を再現する:

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

現行 bundle に入れている execution compression 版の確認結果:

```text
docs_local/20260528_plan009_gap16_next_mindelta010_full.json
```

この probe は fold0-12 の開発検証用。depth は validation split だけで選び、test 指標は report-only として扱う。

### Space bundle export

現行デモ bundle を `unidream-space/bundles/current` に再生成する:

```bash
uv run python -m unidream.cli.export_plan009_depth_calibrator_bundle \
  --config configs/trading.yaml \
  --output-dir /Users/sophie/Documents/UniDream/unidream-space/bundles/current
```

生成後、Space 側で sample parity を確認する:

```bash
cd /Users/sophie/Documents/UniDream/unidream-space
PYTHONPATH=/Users/sophie/Documents/UniDream/unidream-space \
  uv run --with-requirements requirements.txt python -m backend.verify_bundle \
  --bundle-dir bundles/current \
  --device cpu \
  --tolerance 0.000001
```

Space 側へ反映する場合は `unidream-space` repo を更新して Hugging Face Space に push する。Supabase demo 側は raw 15分足 candle を `/predict` に送り、Space runtime が直近60日を取引対象、その前の履歴を past-only guard history として使う。

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

Plan009 の部品検証:

```bash
uv run python -m unidream.cli.plan009_component_probe --help
uv run python -m unidream.cli.plan009_guard_sweep_probe --help
uv run python -m unidream.cli.plan009_guard_student_probe --help
uv run python -m unidream.cli.plan009_depth_learner_probe --help
```

## 生成物

実行時に以下が生成される。

- `checkpoints/<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt}`: 学習 pipeline の checkpoint
- `checkpoints/data_cache/`: feature / returns の parquet キャッシュ
- `docs_local/`: Plan009 などのローカル実験JSON/Markdown/log
- `documents/logs/`, `documents/route_probe/`, `documents/wm_probe/`, `documents/ac_candidate_q/`: ログ・診断出力
- `/Users/sophie/Documents/UniDream/unidream-space/bundles/current`: HF Spaces current bundle
- `__pycache__/`: Python 実行キャッシュ
- `.venv/`: `uv sync` が作る仮想環境

`checkpoints/`、`docs_local/`、`.venv/` は Git 管理対象外。checkpoint は再実行で作り直す前提。

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

- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023) — symlog / twohot critic / lambda-return / EMA target / Imagination AC のベース
- Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2, 2021) — discrete latent / KL balancing
- Vaswani et al., "Attention Is All You Need" (2017) — Transformer dynamics
- Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (ICM, 2017) — Inverse Dynamics Model (IDM auxiliary loss)

Imitation Learning / Behavior Cloning:

- Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network" (1989) — Behavior Cloning の起源
- Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger, 2011) — BC の covariate shift / inventory state 分布ズレに対応
- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT, 2023) — `chunk_size=4` の action chunking

Offline RL / Advantage-weighted Policy Extraction:

- Peng et al., "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning" (AWR, 2019) — advantage 重み付き BC
- Nair et al., "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets" (AWAC, 2020)
- Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning" (IQL, 2022)
- Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration" (BCQ, 2019)
- Kumar et al., "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction" (BEAR, 2019)
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (CQL, 2020)
- Fujimoto and Gu, "A Minimalist Approach to Offline Reinforcement Learning" (TD3+BC, 2021)
- Springenberg et al., "Offline Actor-Critic Reinforcement Learning Scales to Large Models" (ICML 2024)
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)

Backtest / 評価:

- Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest Overfitting" (2014) — PBO
- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014) — DSR
- Lopez de Prado, "Advances in Financial Machine Learning" (2018) — WFO / combinatorial CV / overfitting
- Pardo, "The Evaluation and Optimization of Trading Strategies" (2008) — Walk-Forward Optimization
- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle" (1989) — HMM regime detection

## 注意

このリポジトリは研究用途。投資助言や実運用システムではない。暗号資産市場は高リスクであり、バックテスト結果は将来成績を保証しない。
