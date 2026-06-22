# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

BTCUSDT 15分足を対象に、Transformer World Model、Behavior Cloning、Imagination Actor-Critic を組み合わせて検証する暗号資産トレード研究プロジェクト。

## 概要

UniDream は、Hindsight Oracle で生成した教師ポジションを Behavior Cloning で模倣し、その方策を World Model 上の imagination rollouts で Actor-Critic fine-tune する研究コード。Buy & Hold をベンチマークとして、OOS の超過成績、Sharpe 改善、最大ドローダウン改善、collapse 回避を同時に評価する。

研究基盤として WM → BC → AC の学習パイプラインと各種診断 CLI を保持している。現時点の外部説明向け主結果は Plan011 v31。B&H exposure `1.0` 近傍の低回転 overlay actor として、13fold test で大崩れを避けながら AlphaEx の右側テールを取りにいく構成。

Plan011 v31 の fold0-12 開発WFOでは、aggregate `AlphaEx +41.79pt`、worst AlphaEx `-1.28pt`、fold2除外平均 `+5.46pt`。完全未使用の 2024-2026 holdout fold15-23 では aggregate `AlphaEx +2.32pt`、`SharpeDelta -0.003`、`MaxDDDelta +0.20pt`。DD改善AIではなく、低回転の AlphaEx overlay として主張する。

## Plan011 v31 検証スナップショット

Plan011 v31 は、B&H exposure `1.0` を基準にした低回転 overlay actor。VC/外部説明で守れる主成果は「B&H近傍の低回転overlayとして、13fold testで大崩れを避けながらAlphaExの右側テールを獲得した」こと。

0-12 fold test-only 再集計:

| 指標 | 値 |
|---|---:|
| Aggregate AlphaEx | `+41.79pt` |
| Aggregate SharpeDelta | `-0.001` |
| Aggregate MaxDDDelta | `+0.20pt` |
| PBO | `0.420` |
| AlphaEx median | `+0.81pt` |
| AlphaEx worst | `-1.28pt` |
| AlphaEx mean excluding fold2 | `+5.46pt` |
| AlphaEx >= 0 or abs(AlphaEx) <= 1pt | `12/13` |
| AlphaEx >= +3pt | `4/13` |

Fold2 の `AlphaEx +477.79pt` が aggregate を強く押し上げているため、平均だけでは評価しない。中央値・worst・fold2除外平均で見ると「小幅プラスから小幅マイナスに収まりやすく、右側テールを取る低回転overlay」と整理するのが妥当。

### Holdout policy-family ablation

完全未使用だった2024-01-16〜2026-04-16（fold15-23）を同一cost・B&H基準で比較した。数値は9fold平均で、MaxDDDeltaはマイナスが改善。

| 方式 | AlphaEx | MaxDDDelta |
|---|---:|---:|
| B&H | `+0.00pt` | `+0.00pt` |
| 単純vol-target | `-1.14pt` | `-1.44pt` |
| tabular ML | `-6.11pt` | `-0.34pt` |
| Transformer WMのみ | `-17.41pt` | `-1.28pt` |
| Transformer WM + BC（ACなし） | **`+2.83pt`** | `+0.24pt` |
| Plan011 v31（WM + BC + AC） | **`+2.32pt`** | `+0.20pt` |

Transformer WM + BCは、4方式ablationで唯一AlphaEx平均をプラスにしたため、B&H-relative alpha overlayの中核として有効性が見える。一方、AC込みv31はBC-only比でAlphaExが`-0.51pt`、MaxDDDeltaが`-0.04pt`の差にとどまり、turnoverも増えている。validation checkpoint selectorがBCまたは早期checkpointへ戻すfoldもあり、現時点でAC tuningの明確な優位性は確認できていない。WM単体はDD方向のsignalを持つがalphaを失い、BC/ACを含めてもDD改善との両立は未達。詳細、fold別結果、再現hashは [policy_family_holdout_comparison.md](docs/policy_family_holdout_comparison.md) に固定している。

完全未使用 2024-2026 holdout fold15-23:

| 指標 | 値 |
|---|---:|
| Aggregate AlphaEx | `+2.32pt` |
| Aggregate SharpeDelta | `-0.003` |
| Aggregate MaxDDDelta | `+0.20pt` |
| PBO | `0.400` |
| AlphaEx median | `-0.14pt` |
| AlphaEx > 0 | `3/9` |
| AlphaEx >= +3pt | `3/9` |
| MaxDDDelta <= 0 | `0/9` |
| Turnover mean | `0.60` |

holdoutでは、低回転と一部foldのAlphaEx右テールは残った。一方で `MaxDDDelta` は全foldプラスで、DD改善は未達。外部説明では「production-grade risk improvement」ではなく「低回転 B&H-relative neural overlay の研究証跡」として扱う。

重要な限界:

- `MaxDDDelta <= 0` は `0/13`。現状はDD改善モデルではない。
- aggregate `SharpeDelta` は `-0.001`。リスク調整後リターン改善の主張はまだ弱い。
- fold0-12 は開発WFOであり、pristine holdoutではない。
- fold15-23 は 2024-01-16 to 2026-04-16 の完全未使用holdoutとして再学習・再集計済み。ただしDD改善は未達。
- Live / Space runtime は Plan011 v31 fold23 bundle に差し替え済み。実モデル推論を使う。

詳細なfold別結果・外れ値除外統計・bootstrap CI・再現コマンドは [docs/plan011_v31_investor_evidence.md](docs/plan011_v31_investor_evidence.md) と [docs/plan011_v31_holdout_2024_2026.md](docs/plan011_v31_holdout_2024_2026.md) にまとめている。

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
  -> Plan011 v31 scale evaluation / investor evidence report
  -> Space bundle export utilities
```

## ステージごとの責務

- `data`: Binance OHLCV / funding / OI / mark price 取得、特徴量計算、returns 整形
- `oracle`: signal_aim teacher で aim positions を生成、smooth aim、soft labels
- `world_model`: Transformer ベースの latent dynamics、predictive head (return / vol / drawdown / regime)
- `bc`: Actor を oracle position に模倣、route head・inventory recovery controller・state machine gate を学習
- `ac`: World Model 上で imagination actor-critic fine-tune
- `selector`: validation split で selector / scorecard を評価
- `plan011`: B&H-relative low-turnover overlay actor、relative-constraint AC reward、0-12 scale evaluation
- `plan009`: legacy depth calibrator、execution compression、cost stress 検証用CLI
- `deploy`: HF Spaces 向け bundle export utility
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

WM → BC → AC → Test まで通す本トレーニングはこのコマンドで実行する。CLIで変更できるのはconfig・seed・deviceだけで、期間、fold、cost、checkpoint先はYAMLに固定する。

```bash
uv run python -m unidream.cli.train \
  --config configs/trading.yaml \
  --seed 7 \
  --device cuda
```

実行条件はYAMLの`run`で管理する:

```yaml
run:
  start: "2018-01-01"
  end: "2024-01-01"
  folds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  clean_checkpoint_dir: true
  deterministic_algorithms: true

data:
  include_funding: true
  include_oi: false
  include_mark: true

logging:
  checkpoint_dir: checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007
  cache_dir: checkpoints/data_cache
```

`run.clean_checkpoint_dir: true`では実行開始時に当該checkpoint directoryを削除して作り直す。既存WM/BC/ACのresume、stage途中開始、warm-startはmainlineから削除済み。

各runには以下を保存する:

- `resolved_config.yaml`: cost profile解決後の実設定
- `run_manifest.json`: run ID、Git commit、source/config/data SHA256、seed、device、環境、fold一覧
- `checkpoint_semantic_sha256`: PyTorch ZIP metadataに依存しないWM/BC/AC tensor内容hash

### Plan011 v31 の再現

Plan011 v31 のfold0-12を最初から再学習・評価する:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --seed 7 \
  --device cuda
```

完全未使用 2024-2026 holdoutを最初から再学習・評価する:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_holdout.yaml \
  --seed 7 \
  --device cuda
```

YAMLを変えた場合は別configとして保存し、そのファイル名だけを`--config`で差し替える。CLIによる一時上書きはできない。

### Space bundle export

HF Spaces current は Plan011 v31 fold23 の実モデル推論bundle。Plan009/Plan008の手続きoverlayではない。fold23 bundle は最新60日を避けた `2026-04-17` data end で作り、Space runtime の sample parity は `strict_ok=True`。

fold23 bundle を再生成して `unidream-space` に反映する:

```bash
.venv/bin/python -u -m unidream.cli.export_inference_bundle \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --start 2018-01-01 --end 2026-04-17 \
  --fold 23 \
  --seed 7 \
  --device cpu \
  --run Plan011v31=checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007@ac.pt:ac \
  --bundle-type plan011_v31_overlay_actor \
  --status latest_holdout_candidate \
  --source wm_bc_ac_relative_constraint_overlay \
  --spec plan011_v31_relative_constraint_ac_fold23_recent_holdout \
  --output-dir /Users/sophie/Documents/UniDream/unidream-space/bundles/current
```

Space 側で bundle を更新した後は sample parity を確認する:

```bash
cd /Users/sophie/Documents/UniDream/unidream-space
PYTHONPATH=/Users/sophie/Documents/UniDream/unidream-space \
  uv run --with-requirements requirements.txt python -m backend.verify_bundle \
  --bundle-dir bundles/current \
  --device cpu \
  --tolerance 0.000001
```

Space 側へ反映する場合は `unidream-space` repo を更新して Hugging Face Space に push する。

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

Legacy Plan009 の部品検証:

```bash
uv run python -m unidream.cli.plan009_component_probe --help
uv run python -m unidream.cli.plan009_guard_sweep_probe --help
uv run python -m unidream.cli.plan009_guard_student_probe --help
uv run python -m unidream.cli.plan009_depth_learner_probe --help
```

## 生成物

実行時に以下が生成される。

- `<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt}`: 学習 pipeline の checkpoint
- `<logging.checkpoint_dir>/{resolved_config.yaml, run_manifest.json}`: 再現条件とfingerprint
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
