# UniDream

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

Imagination-based reinforcement learning for crypto trading.

Oracle trajectory から Transformer 世界モデルを学習し、Imagination 上で Actor-Critic を訓練してトレーディング方策を獲得する。

---

## Overview

UniDream は、暗号資産トレード向けに **World Model + Imagination RL** を適用するプロジェクトです。
まずはテクニカル指標中心の MVP で検証し、alpha を確認した後に LoRe（LLM Embed / Risk Gate）を段階導入します。

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Data Acquisition                                                │
│     Binance Vision → BTCUSDT 15m klines + funding rate + OI        │
├─────────────────────────────────────────────────────────────────────┤
│  2. Feature Engineering                                             │
│     OHLCV → MA, RSI, BB, MACD, etc.                                │
│     (Phase 2+) LLM embed: ニュース・センチメント                      │
├─────────────────────────────────────────────────────────────────────┤
│  3. Oracle Generation                                               │
│     未来価格既知で DP / 貪欲法 → 最適行動列（手数料込み）               │
├─────────────────────────────────────────────────────────────────────┤
│  4. World Model Training                                            │
│     Transformer (IRIS/TWM系)                                        │
│     混合方策データ(random + ε-greedy + heuristic + oracle)で遷移予測  │
│     (Phase 2+) テクニカル + LLM embed 統合                           │
├─────────────────────────────────────────────────────────────────────┤
│  5. Actor BC → Imagination AC                                       │
│     BC: oracle行動を教師あり模倣                                     │
│     AC: critic warmup → actor fine-tune (KL制約, λ-return)          │
│     (Phase 2+) LoRe Risk Gate: uncertainty gating                   │
├─────────────────────────────────────────────────────────────────────┤
│  6. Backtest                                                        │
│     手数料・スリッページ込み、Sharpe / MDD / PnL 評価                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Specifications

| Item | Value |
|------|-------|
| 対象 | BTCUSDT (Binance Futures) |
| 時間足 | 15 分足 |
| 行動空間 | **離散（buy / hold / sell）** |
| 世界モデル | Transformer (IRIS/TWM 系) |
| Actor | MLP or 小さめ Transformer |
| DL Framework | PyTorch (>= 2.0) |

> 現段階は離散行動空間を採用。15m 足で取引頻度が高くなりやすいため、oracle 側で手数料・スリッページ込み最適化を前提とします。

---

## Implementation Phases

### Phase 1 — MVP（テクニカルのみ）

パイプラインをテクニカル指標のみで一気通貫させ、backtest まで回す。

| Step | Module | 内容 |
|------|--------|------|
| 1 | `data/download.py` | Binance Vision BTCUSDT 15m klines + funding rate + OI 取得 |
| 2 | `data/features.py` | OHLCV → MA, RSI, BB, MACD 等のテクニカル特徴量 |
| 3 | `data/oracle.py` | DP で oracle trajectory 生成（手数料込み） |
| 4 | `world_model/` | Transformer 世界モデル学習（3ヘッド: next state + reward + termination） |
| 5 | `actor_critic/` | Actor BC → Critic warmup → Imagination AC |
| 6 | `eval/backtest.py` | Sharpe, MDD, PnL 評価 |

### Phase 2 — LoRe 統合

Phase 1 で alpha が確認できたら追加。変数分離のため段階的に導入。

- **LLM Embed**: ニュース・イベント・市場センチメントを LLM で embed → 世界モデルの入力に結合
- **Risk Gate**: FOMC 発表、ハック事件等の重大イベント検知 → uncertainty 高い時に actor のポジションサイズを強制縮小 / フラット化

### Phase 3 — オンライン fine-tune

Backtest 結果を見てから検討。

- 実環境データで世界モデルを逐次更新
- Replay buffer に実データ蓄積

---

## Repository Structure

```
UniDream/
├── README.md
├── CLAUDE.md
├── unidream/
│   ├── data/
│   │   ├── download.py            # Binance Vision klines 一括DL
│   │   ├── features.py            # テクニカル指標計算
│   │   ├── oracle.py              # Oracle trajectory 生成 (DP)
│   │   └── dataset.py             # データローダー（混合方策）
│   ├── world_model/
│   │   ├── transformer.py         # Transformer 世界モデル
│   │   └── train_wm.py            # 世界モデル学習エントリポイント
│   ├── actor_critic/
│   │   ├── actor.py               # Actor (MLP / Transformer)
│   │   ├── critic.py              # Critic (value function)
│   │   ├── bc_pretrain.py         # Actor BC 事前学習
│   │   └── imagination_ac.py      # Imagination AC 学習
│   ├── lore/                      # Phase 2: LoRe 統合
│   │   ├── llm_embed.py           # ニュース/イベント → LLM embedding
│   │   └── risk_gate.py           # Uncertainty gating
│   ├── online/
│   │   └── finetune.py            # オンライン fine-tune（backtest後に検討）
│   └── eval/
│       └── backtest.py            # バックテスト評価
└── configs/
    └── trading.yaml
```

---

## Dependencies

```
torch >= 2.0
numpy
pandas
pandas-ta          # テクニカル指標
scikit-learn       # 正規化等
matplotlib         # 評価・可視化
```

## References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4)
- [IRIS](https://arxiv.org/abs/2209.00588) — Transformer world model for Atari
- [TWM](https://arxiv.org/abs/2209.14855) — Transformer-based World Models
- [Dynalang](https://arxiv.org/abs/2308.01399) — multi-modal world model
