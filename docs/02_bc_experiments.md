# BC 実験サマリ

## BC Plan 1: Predictive State Optimization → **不採用**

WM 予測 head (return/vol/drawdown) を BC/Actor に渡す実装は完了。ただし:
- no-pred baseline: AlphaEx `-0.54`, short=3%/flat=97%
- predictive state direct concat: short 92-99% に崩壊
- adapter 接続: short 22%/flat 78% まで回復
- 実予測値: scale=1.0 で short 99%、de-risk/underweight 方向に寄る
- **結論**: BCがM2候補として不十分。AC チューニング未実行

## BC Plan 2: Transition Advantage BC → **M2未達**

transition/action 単位の cost-adjusted advantage でBC。
- 最良: `AlphaEx +3.57, SharpeΔ +0.082, MaxDDΔ -0.28, TO 8.42, M2 MISS`
- 平滑化なしでは high-turnover short collapse
- 平滑化 + margin でBC安定。保守的AC 軽く入れて fold4 で改善
- **結論**: 前回 baseline より良いがM2未達。transition advantage 方向は有望だが平滑化必須

## BC Plan 3: True Routing BC → **不採用**

4 route label (neutral/de_risk/recovery/overweight) での true routing BC:
- Best: AlphaEx `+0.17`, TO `3.26`, flat `100%`
- route prediction が neutral に偏りすぎ（91-100%）
- recovery route `0%`
- **結論**: TO 崩壊を抑制できるが route 活性ゼロ。ACへ進めず

## BC Plan 4: Phase 1-5 → **全滅**

route margin/penalty, recovery weight, predictive gate, two-stage route, delta/no-trade 制約:
- P1 (baseline): AlphaEx `+0.89`, TO `4.05`, short 28%/flat 72%, recovery 0%
- P2-P5: 片側 short/underweight collapse（98-100%）
- Route F1: Macro-F1 `0.350→0.406`
- Recovery recall max `0.110`
- **結論**: route 分類は微改善。全 phase が one-sided collapse。de_risk 支配から抜け出せず

## BC Plan 5: Phase 6-9

### Phase 6: 推論安全化
- confidence fallback, active cap, recovery fallback, exposure loss
- 6C relaxed: AlphaEx `+0.47`, MaxDDΔ `-1.00`, TO `12.79`, recovery `18%`
- **課題**: recovery は増えたが churn collapse (TO 12.79)
- **結論**: 単純 logit boost では churn 止まらず。state machine with hysteresis が必要

### Phase 7-8: State Machine → **本流採用**
- **Phase 8: AlphaEx `+0.91`, SharpeΔ `-0.010`, MaxDDΔ `-1.61`, TO `2.62`**
- state machine gate が de_risk を underweight 中に抑制 → TO 12.11→2.62 の大幅改善
- short 17%/flat 83%, recovery 0.7%
- **これが現在の本流 baseline**

### Phase 9: Constrained AC
- AC-0/1/2: Phase8 と同性能、改善なし
- AC-3 (route lite unlock): validation `-106pt`、崩壊
- cooldown/hysteresis: flat 100% に過剰抑制
- **結論**: 制約付き AC は安全だが Phase8 BC を超えられない

## BC 全般の課題

- **route label の teacher inventory shortcut**: 市場状態ではなく teacher position を読んでいる疑い
- **de_risk 支配**: class weight/focal が逆効果
- **recovery route 活性ゼロ**: 教師ラベル設計から見直しが必要
- **churn collapse**: TO 爆発の根本解決は state machine gate

## Phase 8 Safe Baseline が現在の本流

```
AlphaEx: +0.89〜+0.91 pt/yr
SharpeΔ: -0.010〜-0.011
MaxDDΔ: -1.58〜-1.61 pt
turnover: 2.62
short: 16-17% / flat: 83-84%
recovery gate active: 0.7-1.0%
```
