# UniDream — Technical Specification

アーキテクチャ・パイプライン・評価手法の詳細仕様。

---

## 入力特徴量

OHLCV（対数リターン化）＋ 最小TA（RSI, MACD, ATR）を**フラットに concat** して Transformer に入力する。
raw 主入力・TA 補助のように分ける必要はない。表形式データなので全部同列に扱う。

| 特徴量 | 説明 |
|--------|------|
| OHLCV | 対数リターン化した O/H/L/C/V |
| RSI | 相対力指数 |
| MACD | 移動平均収束拡散 |
| ATR | 平均真値域（ボラ正規化に使用） |

**板情報（Order Book）**: v2 ではスキップ。FX/仮想通貨は API 制約とデータ量が重いため。効果はアブレーションで後から検証。

### 正規化

- rolling z-score（60 日窓）一択
- リターンは ATR 割りでボラ正規化
- `shift(1)` を必ず適用してリークを防ぐ

---

## 世界モデル

**ベース**: DreamerV3、RSSM を Transformer（Block-Causal）に置換。
Hansen 版 Dreamer4 (`nicklashansen/dreamer4`) の設計を参考にする。

| 項目 | 設計 |
|------|------|
| エンコーダ | MLP 2 層（256-256）※ 画像エンコーダを差し替え |
| 潜在空間 | DreamerV3 の離散カテゴリカル（32×32） |
| シーケンスモデル | Block-Causal Transformer |
| Imagination horizon | 1〜3 ステップ（金融の SNR が低く長い rollout は信頼できない） |
| アンサンブル | 3〜5 モデル、不一致ペナルティを適用 |

離散カテゴリカル潜在を採用する理由: レジームスイッチングを暗黙的にキャプチャできる可能性がある。

---

## 行動空間

離散 5 択（ポジション比率）:

| 値 | 意味 |
|----|------|
| -1.0 | フルショート |
| -0.5 | ハーフショート |
| 0.0 | フラット |
| +0.5 | ハーフロング |
| +1.0 | フルロング |

---

## 報酬関数

```
R_t = DSR(r_t − costs_t) − β · ΔDD_t
```

- **DSR**（Differential Sharpe Ratio）: 主成分
- **ΔDD_t**: ドローダウン増分ペナルティ（補助）
- **コスト処理**: リターン計算時に差し引く（環境側で処理。報酬関数には混ぜない）
- **β**: 0.1〜0.5 でスイープ
- CVaR・複合報酬は複雑さに見合わないため採用しない

---

## Hindsight Oracle

- train 期間のみで後ろ向き DP 計算（テスト期間の未来情報は使わない）
- 離散 5 行動、取引コスト込み
- 世界モデルの学習データおよび BC の教師ラベルとして使用

---

## 模倣学習（BC 初期化）

| 項目 | 設計 |
|------|------|
| 損失 | KL-divergence |
| 重み | 状態依存・学習可能（SIRL の手法） |
| エポック数 | 数エポックで切り上げ、AC に移行 |

DAgger 的な再収集は実装コストが高いため初版ではやらない。

---

## Actor-Critic 最適化

DreamerV3 の Actor-Critic をベースに以下を追加:

**TD3+BC 的な保守的制約**: 世界モデルの誤差領域での暴走を抑制。

**BC 損失減衰混合**:

```
loss = α · BC_loss + (1-α) · AC_loss
```

α を 1→0 に線形減衰しながら BC から AC にソフトに移行する。

---

## 評価手法

| 項目 | 設計 |
|------|------|
| 分割 | Walk-Forward（四半期ロール、最低 10 期間） |
| 期間 | train 2 年 / val 3 ヶ月 / test 3 ヶ月 |
| フィルタ | PBO < 0.5 ＋ Deflated Sharpe で過学習検出 |
| レジーム検出 | HMM で 2〜3 状態を検出 |
| レポートメトリクス | Sharpe / Sortino / 最大 DD / Calmar（レジーム別） |
| 再現性確認 | seed 5 個以上で分散確認 |

---

## パイプライン全体フロー

```
OHLCV取得（FX: OANDA API / 仮想通貨: Binance API）
  ↓
rolling z-score正規化（60日窓）+ ATR割りリターン + TA計算（shift(1)適用）
  ↓
時系列WFO分割（train 2年 / val 3ヶ月 / test 3ヶ月、四半期ロール）
  ↓
train期間でhindsight oracle DP計算（離散5行動、コスト込み）
  ↓
BC初期化（KL損失、状態依存重み、数エポック）
  ↓
Transformer世界モデル学習（MLP encoder、離散カテゴリカル潜在、アンサンブル3-5）
  ↓
AC fine-tune（imagination 1-3step、DSR-コスト-DD罰、BC損失減衰混合）
  ↓
test期間バックテスト（スプレッド+手数料+スリッページモデル込み）
  ↓
PBO/DSR選別 → レジーム別メトリクス → 生き残りのみペーパートレード
```

---

## 実装順序

1. **バックテスト基盤を最初に固める**
   - WFO 分割、コスト・スリッページモデル、PBO 計算
   - ここがガバいと何を学習しても意味がない
2. **model-free AC（PPO）をベースラインとして実行**
   - 世界モデルを足したときの差分を確認するための比較対象
3. BC 初期化
4. Transformer 世界モデル学習
5. AC fine-tune（imagination）
6. バックテスト・評価

---

## Phase Strategy

### Phase 1 — バックテスト基盤 + Model-Free ベースライン（現フェーズ）

テクニカルのみで一気通貫させ、alpha を確認する。

### Phase 2 — LoRe 統合

Phase 1 で alpha が確認できてから導入。

- **LLM Embed**: ニュース・イベント・センチメントを LLM で embed → 世界モデルの入力に結合
- **Risk Gate**: 重大イベント検知時に actor のポジションを強制縮小 / フラット化

### Phase 3 — オンライン fine-tune

Backtest 結果を見てから検討。実環境データで世界モデルを逐次更新。

---

## Repository Structure

```
UniDream/
├── README.md                      # プロジェクト概要
├── CLAUDE.md                      # 開発方針（短く）
├── SPEC.md                        # 技術仕様詳細（本ファイル）
├── unidream/
│   ├── data/
│   │   ├── download.py            # Binance / OANDA OHLCV取得
│   │   ├── features.py            # TA計算・rolling z-score正規化
│   │   ├── oracle.py              # Hindsight oracle DP（離散5行動、コスト込み）
│   │   └── dataset.py             # データローダー（WFO分割対応）
│   ├── world_model/
│   │   ├── encoder.py             # MLP encoder（2層 256-256）
│   │   ├── transformer.py         # Block-Causal Transformer 世界モデル
│   │   ├── ensemble.py            # アンサンブル（3-5モデル）+ 不一致ペナルティ
│   │   └── train_wm.py            # 世界モデル学習エントリポイント
│   ├── actor_critic/
│   │   ├── actor.py               # Actor（MLP / Transformer）
│   │   ├── critic.py              # Critic（value function）
│   │   ├── bc_pretrain.py         # BC初期化（KL損失、状態依存重み）
│   │   └── imagination_ac.py      # Imagination AC（BC損失減衰混合）
│   ├── baselines/
│   │   └── ppo.py                 # Model-free PPO ベースライン
│   ├── lore/                      # Phase 2: LoRe 統合
│   │   ├── llm_embed.py           # ニュース/イベント → LLM embedding
│   │   └── risk_gate.py           # Uncertainty gating
│   ├── online/
│   │   └── finetune.py            # オンライン fine-tune（Phase 3）
│   └── eval/
│       ├── backtest.py            # バックテスト（コスト・スリッページモデル込み）
│       ├── wfo.py                 # Walk-Forward Optimization
│       ├── pbo.py                 # PBO・Deflated Sharpe計算
│       └── regime.py              # HMMレジーム検出・レジーム別メトリクス
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
hmmlearn           # HMMレジーム検出
matplotlib         # 評価・可視化
requests           # Binance / OANDA API
```

---

## References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4) — Block-Causal Transformer 実装参考
- [DreamerV3](https://arxiv.org/abs/2301.04104) — 離散カテゴリカル潜在・Actor-Critic 設計
- [IRIS](https://arxiv.org/abs/2209.00588) — Transformer world model for Atari
- [TWM](https://arxiv.org/abs/2209.14855) — Transformer-based World Models
- [SIRL](https://arxiv.org/abs/2209.02276) — 状態依存 BC 重み
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) — 多重比較補正
