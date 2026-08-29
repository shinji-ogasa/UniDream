# Plan013 AC Curriculum Probe

> Historical pre-reproducibility-contract probe. The checkpoint/selector persistence issue recorded below is fixed in the current mainline; this document preserves the original experiment finding only.

## Question

BC が collapse しない初期方策を作れているなら、現行 reward を変更せず、AC の自由度を段階的に解放することで BC を超えられるか。

## Conditions

- BTCUSDT 15m, seed 7, folds 0 / 2 / 8
- baseline と gradual curriculum で data / WM / BC / reward は同一
- one-way full `Delta position = 1` cost: `5.50bps`
- test は report-only
- baseline config:
  - `configs/experiments/plan013_ac_release_baseline_fold0.yaml`
  - `configs/experiments/plan013_ac_release_baseline_folds2_8.yaml`
- gradual config: `configs/experiments/plan013_ac_release_gradual_folds0_2_8.yaml`

Semantic checkpoint 比較では、各foldのbaselineとgradualでWM/BC checkpointが一致した。比較差分はAC curriculumのみ。

Gradual checkpointを別プロセスからreplayし、fold 0 / 2 / 8のAlphaExとMaxDDDeltaが学習直後のtest結果に一致することも確認した。

## Implementation findings

現行baselineでは、全3foldで最終 `ac.pt` のActor重みが `bc_actor.pt` と完全一致した。

| fold | baseline final AC vs BC relative L2 | stopped step |
|---:|---:|---:|
| 0 | `0.000000` | 90 |
| 2 | `0.000000` | 90 |
| 8 | `0.000000` | 90 |

AC更新は実際には発生していた。checkpoint時点のAC gradient normはBC gradient normを上回ることが多く、fold 2ではBC/AC gradient cosineが最小 `-0.871`まで低下した。しかしvalidation selectorがBC開始点を選び直すため、最終artifactからAC更新が全て消えていた。

Curriculum実装には次の再現性問題もあったため修正した。

1. α scheduleがstage-localではなくglobal step基準だった。
2. curriculumで変更したActor boundsがTrainer側boundsへ同期されなかった。
3. curriculumのActor runtime overridesがcheckpointへ保存されず、学習直後とreplayで異なる方策になった。

## Curriculum

Gradual runではrewardを変えず、次の順番で解放した。

1. step 0-90: controller headsのみ、α `0.75 -> 0.50`
2. step 90-180: full actor、従来boundsを維持、α `0.50 -> 0.25`
3. step 180-270: boundsを部分解放、α `0.25 -> 0.12`
4. step 270-360: boundsを最終解放、α `0.12 -> 0.05`

中間stageではvalidationへの巻き戻しを無効にし、最終stageだけbest checkpointを復元した。

最終ACはBCから明確に移動した。

| fold | gradual final AC vs BC relative L2 |
|---:|---:|
| 0 | `0.073354` |
| 2 | `0.080892` |
| 8 | `0.060906` |

## Test results

MaxDDDeltaはマイナスが改善。

| fold | baseline AlphaEx | gradual AlphaEx | delta | baseline SharpeDelta | gradual SharpeDelta | baseline MaxDDDelta | gradual MaxDDDelta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.37pt | +0.32pt | -0.05pt | +0.001 | +0.002 | +0.15pt | +0.12pt |
| 2 | +4.78pt | +5.70pt | +0.92pt | +0.003 | -0.020 | +0.21pt | +0.44pt |
| 8 | -0.44pt | -0.46pt | -0.02pt | -0.015 | -0.007 | +0.42pt | +0.44pt |
| mean | +1.57pt | +1.85pt | **+0.28pt** | -0.004 | -0.008 | +0.26pt | +0.33pt |
| median | +0.37pt | +0.32pt | -0.05pt | - | - | - | - |

Turnover meanはbaseline約`0.54`からgradual約`0.18`へ低下した。

## Interpretation

- 「現行ACがBCから十分に進めていない」は正しい。より正確には、ACは更新されるが最終selectorが全foldでBCへ完全復元していた。
- heads-onlyからfull actorへ徐々に解放する方法はcollapseを防いだ。急進版ではstage遷移直後にturnover `38.9`まで上がったが、gradual版ではこの崩れを回避した。
- gradual curriculumは平均AlphaExを`+0.28pt`押し上げたが、改善はfold 2の右テールだけで、median、Sharpe、DDは改善しなかった。
- bounds拡大前のfull-actor guarded stageが最も安定していた。position boundsまで広げるとvalidationが一度悪化したため、現時点で全制限解除は支持されない。
- rewardを変更せずにACを最終artifactへ残せたため、主な初期ボトルネックは目的関数よりcurriculum / checkpoint selectionだった。ただしrisk-adjusted成績を改善するには最終selectorと自由度の選び方がまだ必要。

## Recommended next probe

次はboundsを従来値に固定し、`BC -> heads-only -> full actor guarded`だけを実行する。各stage境界を候補checkpointとして保存し、validationでBCを含むstage間選択を行う。position boundsの拡大は別ablationに分離する。

現段階では3 development folds・1 seedのprobeであり、holdout結論には使わない。
