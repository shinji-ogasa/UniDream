# Plan 3 WM-Based Overlay Report

Date: 2026-05-02

## Summary

Plan 2 の最良候補 (`D_risk_sensitive + triple-barrier guard + pullback eval-only`) を、
ridge回帰プローブから **実 Transformer World Model 推論ベースの overlay** に格上げし、
f456 で検証した。

WM checkpoint: `checkpoints/acplan13_base_wm_s011` (step=1500, return/vol/drawdown head あり)
Device: `cuda` (GPU)

## Approach

```
1. WM checkpoint ロード (build_ensemble + WorldModelTrainer.load)
2. 特徴量 → encode_sequence → latent (z, h)
3. aux head から未来 return/vol/drawdown を予測 (horizon=32)
4. 候補 position [0.75, 1.0, 1.05, 1.10] ごとの D risk-sensitive utility を計算
5. validation で threshold + cooldown 選択 (val 期間の実現効用最大化)
6. test 期間で triple-barrier guard + pullback guard を適用
7. バックテスト
```

ridge回帰は一切不使用。WMのpredictive head出力を直接効用計算に使う。

## Results: f456

| fold | AlphaEx (pt/yr) | MaxDDΔ (pt) | SharpeΔ | turnover | flat | active | pullback_blocked | danger_blocked |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 45 | 2078 |
| 5 | **+8.431** | -0.121 | +0.008 | 0.2 | 1.000 | 1 | 201 | 635 |
| 6 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 154 | 66 |

### Aggregate

| metric | value |
|---|---|
| AlphaEx mean | +2.810 pt/yr |
| AlphaEx worst | 0.000 pt/yr |
| AlphaEx median | 0.000 pt/yr |
| MaxDDΔ mean | -0.040 pt |
| MaxDDΔ worst | 0.000 pt |
| SharpeΔ mean | +0.003 |
| turnover max | 0.2 |
| PBO | 0.333 |

## Comparison: WM vs Ridge Probe (Plan 2)

| variant | AlphaEx mean | AlphaEx worst | active folds | 特徴 |
|---|---|---|---|---|
| Ridge D+A+pullback | +0.152 | 0.000 | fold4,6 | fold4 +0.423 / fold6 +0.037 |
| **WM D+guard+pullback** | +2.810 | 0.000 | fold5 only | fold5 +8.431 (1 trade) |

### 解釈

- **WM と ridge は全く異なる fold に反応する**
  - Ridge: fold4 は読めるが fold5 は neutral
  - WM: fold5 で大勝ち (+8.431, 1トレードのみ) だが fold4/6 は活性化せず
- 両者の信号は相補的
- WM の fold5 検出は return head 予測が効いている可能性が高い
- fold4/6 が neutral なのは threshold が高すぎる or DD guard が強すぎる可能性

## Fold5 Detail

- active events: 1
- pullback_blocked: 201
- danger_blocked: 635
- threshold: 0.102
- cooldown: 0

単一の de-risk イベントで +8.431pt の超過リターン。pullback guard が 201 回、danger guard が 635 回の false de-risk を阻止。

## Issues

1. **fold4/6 が neutral**: WM の効用予測が実現値と乖離しているか、threshold 選択が保守的すぎる
2. **DD penalty scale**: 現在 `dd_penalty=1.50` だが、WM の drawdown 予測値のスケールが実績値と異なる可能性（WM は `risk_target_scale=100.0` で訓練されている）
3. **vol_penalty**: 同様にスケール不一致の可能性

## Next

- WM 予測値のスケール校正（実績 return/dd/vol と比較）
- DD penalty / vol penalty の grid search
- fold4/6 活性化のための threshold floor 引き下げ
- fold0 も checkpoint があれば追加検証

## Artifacts

```text
documents/20260502_plan3_wm_overlay_f456.json
documents/20260502_plan3_wm_overlay_f456.md
```
