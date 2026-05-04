# AC 実験サマリ (Plan 1-16)

## AC Plan 1: Candidate Q Probe → **不採用**
- CQL-lite で test Spearman `0.261`。全 variant flat 100%。Q は改善actionを選べず。
- 診断 probe としてのみ採用。AC actor update 拒否。

## AC Plan 2: Residual BC → **不採用**
- 実現 advantage 残差 BC。全 variant Phase8 (`+0.89`) と同性能か悪化。
- コード削除済み。

## AC Plan 3: Benchmark-Gated Overweight Adapter → **初の採用**
- 推論時 overweight adapter（WM predictive advantage で gate）
- eps=0.20, adv_min=1.00: **AlphaEx `+1.15` (+0.24 over Phase8), MaxDDΔ `-1.97`, TO `3.05`, long `1%`**
- Phase8 に小ロングを安全に上乗せ。本流採用。

## AC Plan 4: Fold 展開 → **fail**
- fold0: `-5.13`, fold5: `-51.36`。fold4 のみ pass。
- 3fold avg `-18.45`。full actor unlock 禁止継続。

## AC Plan 5: Exposure Floor → **採用**
- `benchmark_exposure_floor = 1.0` で benchmark 未満露出防止
- fold0 `-0.02`, fold4 `+0.41`, fold5 `+40.04`。3fold avg `+13.48`。short `0%`
- 実質的な多fold改善。floor 1.0 で採用。

## AC Plan 6: Multi-Fold Attribution → **診断**
- 改善の 95-100% が floor 効果。adapter fire: fold4 2.6%, fold5 4.9%
- Gate_off は即死（AlphaEx `-57〜-324`）
- **PeriodWin `27.8%` → M2 MISS**
- trainable sizing adapter が必要

## AC Plan 7: Sizing Adapter AC → **採用**
- critic-only 200 + adapter-only 250 steps
- 3fold avg: **AlphaEx `+13.91`, MaxDDΔ `-0.30`, TO max `2.55`, long max `2%`**
- 全条件 pass。SharpeΔ 改善は薄い（+0.001）。scale=0.5 固定で採用。

## AC Plan 8: Robustness & Selector → **問題あり**
- Plan7 再現確認。seed 11 で fold5 `-4.22`（崩壊）
- cost x2 でも `+12.08`（pass）
- AC training recipe は seed-robust でない

## AC Plan 9: Fire Timing Probe → **seed依存**
- seed7 WM+AC vs seed11 WM+AC の Jaccard overlap `0.233`
- fire timing が seed/WM に強く依存
- E (seed11 both) は AlphaEx `+62.04` だが MaxDDΔ `+0.04` (fail)
- **結論**: AC checkpoint selection + WM-AC 互換性が問題。fire は robust でない

## AC Plan 10: Fire Checkpoint Selector → **未解決**
- step-level checkpoint 全保存。fire-pnl-aware selector。
- 全 checkpoint MaxDDΔ `+0.04`。fire PnL は正だが MaxDD 改善せず。
- checkpoint selector だけでは救えない。

## AC Plan 11: Fire-Time DD Guard → **不採用**
- 21 variants。pre_dd_22.5%: AlphaEx `+35.89`, MaxDDΔ `-0.03`
- Oracle guard: AlphaEx `+64.34`, MaxDDΔ `+0.00` (未来リーク)
- 全 deployable guard が Plan7 (`+41.31`) 未満

## AC Plan 12: WM Control Head-Only → **fold5のみ**
- head-only WM fine-tune (overweight_advantage/recovery head)
- fold5: AlphaEx `+39.00`, MaxDDΔ `≈0.00`, TO `2.55`
- full WM retrain は `-176.05` に崩壊（禁止）

## AC Plan 13: Head-Only WM Multi-Fold → **不採用**
- f456 展開: fold4 `+0.19`, fold5 `-15.09`, fold6 `-0.15`
- mean AlphaEx `-5.02`, mean MaxDDΔ `+0.32`。全 FAIL
- Head-only WM は mainline 不可

## AC Plan 14: Fire Control Label → **部分的**
- E fold5: harm AUC h32 `0.918`（強い）
- Hbest: harm AUC h16 `0.455-0.739` (不安定)
- **fire_advantage_h32 は全 fold で top10/top20 が正** → 使える
- harm/DD/trough label は multi-fold で不安定

## AC Plan 15: Fire Control V2 → **部分的**
- fire_advantage_h32 top10: 全 fold で正（`+0.00006〜+0.00152`）
- "fire advantage is readable, but fire safety (harm prevention) is not"
- combined score は adv_only を超えない

## AC Plan 15-B: MDD Fire Label → **未達**
- future_mdd_overlap AUC: fold4 `0.572`, fold5 `0.538` (0.55未満), fold6 `0.662`
- post_fire_dd_contribution: 不安定
- **MDD label は multi-fold で安定しない**

## AC Plan 16: Fire Type Clustering → **重要発見**
- **pre_dd_danger_fire**: 3fold で一貫して悪い（adv 負、MDD overlap `0.88-0.95`）
- danger score AUC: future_mdd `0.70-0.80`, pre_dd_type `0.68-0.76`
- Oracle guard 上限でも MaxDD ≤ 0 未達
- **結論**: post-hoc guard では救えない。train-time/checkpoint-selection へ戻る

## AC 全体結論

| Plan | 内容 | 状態 |
|---|---|---|
| 3 | Benchmark-gated overweight adapter | ✅ 採用 |
| 5 | Exposure floor | ✅ 採用 |
| 7 | Sizing adapter AC | ✅ 採用 |
| 1,2 | Candidate Q, Residual BC | ❌ 不採用 |
| 4,6,8 | Fold展開, Attribution, Robustness | ⚠️ 問題あり |
| 9,10,11 | Fire timing, selector, guard | ❌ 未解決 |
| 12,13 | WM control head | ❌ multi-fold不採用 |
| 14,15,16 | Fire control/mdd label | ⚠️ 部分的（fire_advantage読めるがsafetyは読めず） |
