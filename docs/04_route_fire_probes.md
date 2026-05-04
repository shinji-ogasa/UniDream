# Route / Fire / Reproducibility プローブサマリ

## Route Separability プローブ

### 基本診断
- `context` や `wm_context` が強いが、主因は **current position / inventory state**
- `wm_advantage` や `wm_regime` 単体はほぼランダム
- benchmark label に置き換えると mean AUC `0.52` 前後に低下
- **route教師は「市場状態」ではなく「teacherのposition状態」を読んでいる疑い**

### Two-Stage Route
- v1/v2 とも active recall がほぼゼロ、benchmark floor に逃げただけ
- head構造だけ変えてもダメ。教師ラベル separability が先

### Route Rebalance Probe
- mild / standard の rebalance 調整検証
- 結論: 素の route head では改善せず

### Deterministic Retrain
- 確率的 latent を止めて決定論的に再訓練。再現性改善目的
- Plan17 reproducibility の一環

## Plan17: Reproducibility 検証

### 問題
BC/AC の WFO 再実行で結果が一致しない問題を調査

### 調査内容
- WM の stochastic latent が結果変動の主因
- 決定論的 retrain で fold4/5 の一致確認
- 複数 seed での比較検証

### 結果
- 決定論的モードで fold4/5 の結果一致確認
- WM 確率 latent を止めれば再現性確保可能
- **本流に決定論的モードを採用**

## Fire 診断プローブ

### Fire Timing Probe (AC Plan 9)
- fire タイミングの Jaccard overlap は seed 間で `0.233` と低い
- fire は seed/WM に強く依存、robust でない

### Fire DD Guard (AC Plan 11)
- 21 variants の DD guard sweep
- 全 deployable guard が Plan7 未満
- 理論的には DD 改善可能だが、実用的 guard は存在せず

### Fire Control Label (AC Plan 14-15)
- fire_advantage_h32: 全 fold で top10/top20 が正 → **使える信号**
- harm/DD worsening label: multi-fold で不安定
- "fire advantage is readable, but fire safety is not"

### Fire Type (AC Plan 16)
- **pre_dd_danger_fire**: 3fold で一貫して悪い。MDD overlap `0.88-0.95`
- danger score AUC: future_mdd `0.70-0.80`
- post-hoc guard 限界: train-time/checkpoint-selection へ

## Market Event Label (Plan 1)
- risk_off/recovery/overweight の market-state event label
- fold4 で一部読めるが fold5/fold6 で AUC `0.50` 付近に低下
- tightened v2, h64, HGB でも改善せず
- **BC/AC へ進めないと判断**

## Probe ABC Failure (2026-04-25)
- dualresanchor vs feature_stress_tri 比較
- best axis (stresstri_shiftonly): short 90-97%、片側 collapse
- Probe C: AlphaEx `-29.18`, MaxDDΔ `+4.14` (壊滅的)
- **M2 candidate 不在。policy が short/flat に崩壊する根本問題あり**

## Transformer WM Probe
- `transformerwm.md`: WM の基本診断結果
- return/vol/drawdown 予測 head の性能評価

## 全体の Reproducibility 課題
- WM の stochastic latent → **決定論的モードで解決**
- AC checkpoint selection → seed/WM 互換性問題
- WFO fold 間の結果変動 → 複数 fold 評価の標準化（未完了）

## 主要 CLI プローブ一覧

| CLI | 内容 |
|---|---|
| `route_probe.py` | Route 分類性能 |
| `route_separability_probe.py` | Route label の特徴別 separability |
| `wm_probe.py` | WM linear probe |
| `market_event_label_probe.py` | Market event label 評価 |
| `fire_mdd_label_probe.py` | Fire/MDD label 評価 |
| `ac_candidate_q_probe.py` | State-action Q ranking |
| `ac_fire_timing_probe.py` | Fire timing 分析 |
