# UniDream — Development Guide

開発時の設計判断・コーディング規約・アーキテクチャ詳細をまとめたファイル。
プロジェクト概要は README.md を参照。

---

## Architecture Decisions

### DL Framework: PyTorch
- Dreamer 4 (`nicklashansen/dreamer4`) が PyTorch (torch 2.8) — ベースとして利用
- 世界モデル・Actor・Critic すべて PyTorch で統一

### 世界モデル: Transformer（IRIS / TWM 系）
- 状態トークン列 → 次状態 + 報酬予測
- RSSM ではなく Transformer ベースのシーケンスモデルを採用
- 損失: next state prediction + reward prediction + termination の 3 ヘッド

### 行動空間（採用方針: 離散）

| Option | Pros | Cons |
|--------|------|------|
| 離散（buy / hold / sell） | DP で oracle 計算が容易、実装シンプル | ポジションサイジング不可 |
| 連続（ポジション比率 -1〜+1） | 柔軟なポジション管理 | oracle 計算が複雑、方策学習が難しい（将来検討） |

**現時点の採用方針は離散（buy / hold / sell）。15m足 × 離散は取引頻度が高くなりがちなので、oracle DP でコスト（手数料・スリッページ）込みにして抑制する。**

### Actor アーキテクチャ
- MLP or 小さめ Transformer
- 入力は過去〜現在の状態のみ（未来情報なし）

### Oracle Trajectory 生成
- 未来価格既知の状態で DP により最適行動列を算出
- 手数料・スリッページをコストとして DP に組み込む
- 世界モデルの学習データおよび Actor BC の教師ラベルとして使用

---

## Data Pipeline Details

### データソース
- **価格データ**: data.binance.vision — BTCUSDT 15m klines CSV 一括 DL
- **Funding rate**: Binance API / Vision（klines CSV には含まれない、別途取得）
- **Open Interest (OI)**: Binance API / Vision（同上、別途取得）

### 特徴量
- **テクニカル**: MA, RSI, BB, MACD 等（pandas-ta で計算）
- **デリバティブ**: funding rate, OI
- **Phase 2 追加**: LLM embed（ニュース・イベント・市場センチメント）

### 学習データ混合
- ランダム方策・ε-greedy・ヒューリスティック・oracle を混合
- 初期比率: ランダム:oracle = 6:4（要チューニング）

---

## Phase Strategy

### Phase 1 — MVP（テクニカルのみで一気通貫）

LoRe 抜きでパイプライン全体を回す。何が効いてるか分離するため、変数を最小化。

1. データ取得（OHLCV + funding rate + OI）
2. 特徴量生成（テクニカル指標のみ）
3. Oracle 生成（DP、手数料込み）
4. 世界モデル学習（Transformer、混合方策データ）
5. Actor BC → Imagination AC
6. Backtest（Sharpe, MDD, PnL）

**Phase 1 の目標: テクニカルだけで世界モデル + AC が機能するか検証。alpha が出るか確認。**

### Phase 2 — LoRe 統合

Phase 1 で alpha が確認できてから導入。段階的に入れる。

**LLM Embed（世界モデルの状態空間拡張）:**
- ニュース・イベント・市場センチメントを LLM で embed
- 世界モデルの入力に結合 → テクニカル + ファンダメンタルの両方で遷移予測

**Risk Gate（uncertainty gating）:**
- LLM が「FOMC 発表 ±30 分」「ハック事件」等の重大イベントを検知
- Uncertainty 高い時に actor のポジションサイズを強制縮小 or フラット化
- 暗号資産は規制ニュース一発で 20% 動くのでこれが特に効く

### Phase 3 — オンライン fine-tune

Backtest 結果を見てから検討。

- 実環境データで世界モデルを逐次更新
- Imagination AC も継続学習
- Replay buffer に実データを蓄積

---

## Repository Structure

```
UniDream/
├── README.md                      # プロジェクト概要・パイプライン図
├── CLAUDE.md                      # 開発ガイド・設計判断（本ファイル）
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
│   │   └── finetune.py            # オンライン fine-tune
│   └── eval/
│       └── backtest.py            # バックテスト評価
└── configs/
    └── trading.yaml
```

---

## Dependencies

```yaml
# Core
- torch >= 2.0
- numpy
- pandas

# テクニカル指標
- pandas-ta
- scikit-learn       # 正規化等

# 評価
- matplotlib
```

---

## Key Design Notes

- **15m 足を採用**: 日足よりデータ量が多く世界モデルに有利、分足ほどノイズが多くない
- **対象**: BTCUSDT (Binance Futures) まず 1 銘柄で検証
- **Funding rate / OI は klines CSV に含まれない**: 別途 Binance API から取得が必要
- **手数料考慮**: 15m × 離散は取引頻度が高くなるリスクあり。oracle DP にコスト込みで対応
- **LoRe は MVP 後**: 変数を増やしすぎるとデバッグ地獄。テクニカルのみで alpha 確認が先

---

## References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4)
- [IRIS](https://arxiv.org/abs/2209.00588) — Transformer world model for Atari
- [TWM](https://arxiv.org/abs/2209.14855) — Transformer-based World Models
- [Dynalang](https://arxiv.org/abs/2308.01399) — multi-modal world model
