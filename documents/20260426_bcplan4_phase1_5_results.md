# BC Plan 4 Phase 1-5 Verification Results

作成日: 2026-04-26  
対象: `documents/bcplan_4.md`  
実行範囲: BTCUSDT 15m / seed 7 / fold 4 / 2018-01-01 to 2024-01-01 / BC only  
文字コード: この文書は UTF-8 で保存。

## 結論

Phase 1〜5 はすべて実行完了。AC には移行しない。

理由は、BC の route 分類指標は改善したが、実際のポジション出力が underweight/short 側に大きく崩れており、`bcplan_4.md` の AC 移行禁止条件に該当するため。

特に Phase 2/3/5 は AlphaEx と SharpeΔ だけ見ると強いが、これはほぼ常時 underweight/short の片側ポジションで得た結果であり、汎化した policy として扱えない。

## 実装した内容

- `transition_advantage.py`: route-specific margin / route penalty を追加。
- `fold_inputs.py`: train/val route label、soft label、route advantage を BC に渡す経路を追加。
- `actor.py`: predictive state を trunk に直結しない `route_gate` mode を追加。
- `actor.py`: two-stage route head を追加。
- `actor.py`: underweight duration feature を controller state に追加。
- `actor.py`: route-specific max step 制約を追加。
- `bc_pretrain.py`: route class weight / focal loss を追加。
- `bc_setup.py`, `bc_stage.py`: 上記 BC 設定の配線を追加。
- `imagination_ac.py`: 新しい route head/gate 付き checkpoint の互換ロードを追加。
- `unidream/cli/route_probe.py`: route 分類性能を train/val/test で評価する CLI を追加。

## 実行したフェーズ

| Phase | Config | 目的 | 判定 |
|---:|---|---|---|
| 1 | `configs/bcplan4_phase1_route_label_s007.yaml` | route label の margin/penalty 再設計 | 最も安定。ただし recovery route が 0%。AC 不可。 |
| 2 | `configs/bcplan4_phase2_recovery_s007.yaml` | recovery class weight / focal / duration feature | 分類は改善。ただし de_risk 99% に崩壊。AC 不可。 |
| 3 | `configs/bcplan4_phase3_predgate_s007.yaml` | predictive state を route gate に限定投入 | route recall は改善。ただし short 100%、turnover 6.86。AC 不可。 |
| 4 | `configs/bcplan4_phase4_twostage_s007.yaml` | neutral vs active の two-stage route | recovery route 3% は出たが short 99%、SharpeΔ 負。AC 不可。 |
| 5 | `configs/bcplan4_phase5_delta_notrade_s007.yaml` | route-specific delta/no-trade 制約 | turnover は抑制。ただし short 98%。AC 不可。 |

## Backtest Results

| Phase | Route dist | Position dist | AlphaEx | SharpeΔ | MaxDDΔ | Turnover | Guard | AC 判定 |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | neutral 84 / de_risk 16 / recovery 0 / overweight 0 | long 0 / short 28 / flat 72 | +0.89 pt/yr | -0.004 | -1.60 pt | 4.05 | pass | recovery 0% で不可 |
| 2 | neutral 1 / de_risk 99 / recovery 0 / overweight 0 | long 0 / short 99 / flat 1 | +61.25 pt/yr | +0.227 | -40.08 pt | 4.95 | pass | one-sided short 99% で不可 |
| 3 | neutral 0 / de_risk 99 / recovery 1 / overweight 0 | long 0 / short 100 / flat 0 | +61.45 pt/yr | +0.255 | -39.92 pt | 6.86 | pass | short 100% / turnover 高で不可 |
| 4 | neutral 11 / de_risk 86 / recovery 3 / overweight 0 | long 0 / short 99 / flat 1 | +3.39 pt/yr | -0.061 | -5.45 pt | 5.07 | pass | short 99% / SharpeΔ 負で不可 |
| 5 | neutral 1 / de_risk 98 / recovery 1 / overweight 0 | long 0 / short 98 / flat 2 | +56.50 pt/yr | +0.899 | -32.72 pt | 4.04 | pass | one-sided short 98% で不可 |

注: log の `short` は実装上の short/underweight bucket として扱う。少なくとも AC に渡せる balanced policy ではない。

## Route Probe Test Results

| Phase | CE | Acc | Macro-F1 | Active Recall | False Active | ECE | Top-Decile Active Advantage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1043 | 0.618 | 0.350 | 0.528 | 0.147 | 0.052 | 0.011263 |
| 2 | 0.9996 | 0.600 | 0.363 | 0.618 | 0.267 | 0.038 | 0.011602 |
| 3 | 0.9726 | 0.608 | 0.402 | 0.604 | 0.239 | 0.015 | 0.011933 |
| 4 | 0.9919 | 0.608 | 0.401 | 0.583 | 0.220 | 0.062 | 0.011666 |
| 5 | 0.9066 | 0.640 | 0.406 | 0.572 | 0.173 | 0.013 | 0.012147 |

## Per-Route Test Recall

| Phase | Neutral | De-risk | Recovery | Overweight |
|---:|---:|---:|---:|---:|
| 1 | 0.853 | 0.732 | 0.000 | 0.000 |
| 2 | 0.733 | 0.805 | 0.028 | 0.027 |
| 3 | 0.761 | 0.768 | 0.110 | 0.063 |
| 4 | 0.780 | 0.725 | 0.032 | 0.117 |
| 5 | 0.827 | 0.739 | 0.021 | 0.094 |

## 読み取り

### 1. route 分類は改善している

Macro-F1 は Phase 1 の 0.350 から Phase 5 の 0.406 まで改善した。Top-decile active advantage も全 Phase で正なので、active 候補の上位だけ見れば route head は価値のあるサンプルをある程度拾えている。

### 2. recovery はまだ弱い

Phase 3 で recovery recall は 0.110 まで上がったが、実行時の recovery route は最大でも Phase 4 の 3%。Phase 1 は recovery 0%、Phase 2 も実行時 recovery 0%。

つまり、分類器に recovery 概念を少し入れることはできたが、rollout 時の policy として underweight から benchmark に戻る力はまだ足りない。

### 3. class weight / focal だけでは危険

Phase 2 以降、recovery を強くするために class weight と focal を入れたが、実際には recovery ではなく de_risk が支配的になった。これは少数 class 補正が route 全体の active 率を押し上げ、最終的に underweight/short collapse を誘発している可能性が高い。

### 4. predictive gate は分類には効くが、policy 安定化には不足

Phase 3 は CE/ECE/recovery recall が良い。ただしポジションは short 100%、turnover 6.86。予測特徴は route 判定には有効だが、position decoder 側に active cap / neutral fallback / exposure constraint がないと崩壊する。

### 5. route-specific delta/no-trade は turnover には効くが、片側崩壊は止めない

Phase 5 は turnover 4.04 まで落ちたが、short 98% のまま。delta 制約は頻繁な切り替えを抑えるだけで、どの側に張るかの偏りは直せない。

## AC 移行判定

AC 移行しない。

`bcplan_4.md` の compromise 条件に照らしても、以下で失格。

- Phase 1: recovery route 0%。
- Phase 2: short/underweight 99% の one-sided collapse。
- Phase 3: short/underweight 100%、turnover 6.86。
- Phase 4: short/underweight 99%、SharpeΔ -0.061。
- Phase 5: short/underweight 98% の one-sided collapse。

AC をここで回すと、collapse した BC policy をさらに強化する危険が高い。実行しない判断が妥当。

## 現状で削除またはデフォルト無効化すべき候補

- 高すぎる `route_class_weights.recovery: 12.0`。recovery ではなく active/de_risk 側の暴走を誘発している。
- 高い `route_target_rate_coef` を active cap なしで使う構成。active route を増やすだけで安定化しない。
- `route_max_step_by_route` 単体で collapse を止める期待。turnover 抑制には効くが、方向バイアスには効かない。
- two-stage route head 単体を安定化策として扱うこと。recovery route は出るが最終 position は改善しない。
- AlphaEx だけで AC 移行判定する運用。Phase 2/3/5 のような片側 underweight/short で見かけ上良くなる。

## 残すべき有効候補

- Phase 1 の route-specific margin / penalty。最も collapse が軽い。
- Phase 3 の `route_gate` 型 predictive state。分類指標と calibration は良い。
- Phase 5 の route-specific delta/no-trade。turnover 制御部品としては有効。
- route probe CLI。BC を AC に渡す前の必須診断として有効。
- underweight duration feature。単体では不足だが recovery head の入力としては必要。

## 次に検証すべき最小案

次は Phase 6 として、分類強化よりも exposure/active 制約を先に入れるべき。

```yaml
base: bcplan4_phase1_route_label_s007
use_wm_predictive_state: true
advantage_input_mode: route_gate
route_advantage_gate_scale: 0.25
route_target_rate_coef: 1.0
route_entropy_coef: 0.002
route_class_weights:
  neutral: 0.8
  de_risk: 1.0
  recovery: 3.0
  overweight: 1.25
route_focal_gamma: 0.5
active_rate_target: 0.12
active_rate_max: 0.20
short_underweight_rate_max: 0.35
neutral_fallback_on_low_confidence: true
min_route_confidence_for_active: 0.55
```

狙いは次の通り。

- Phase 1 の安定性をベースにする。
- Phase 3 の predictive gate は使うが gate scale を弱める。
- recovery class weight を 12 から 3 程度に落とす。
- active/de_risk の上限を明示する。
- low-confidence は neutral に戻す。
- AC 移行判定は position 分布が崩れていないことを最優先にする。

## 生成物

- `documents/logs/20260426_bcplan4_phase1_route_label_fold4.log`
- `documents/logs/20260426_bcplan4_phase2_recovery_fold4.log`
- `documents/logs/20260426_bcplan4_phase3_predgate_fold4.log`
- `documents/logs/20260426_bcplan4_phase4_twostage_fold4.log`
- `documents/logs/20260426_bcplan4_phase5_delta_notrade_fold4.log`
- `documents/route_probe/20260426_phase1_route_probe_fold4.md`
- `documents/route_probe/20260426_phase2_route_probe_fold4.md`
- `documents/route_probe/20260426_phase3_route_probe_fold4.md`
- `documents/route_probe/20260426_phase4_route_probe_fold4.md`
- `documents/route_probe/20260426_phase5_route_probe_fold4.md`
- `documents/20260426_bcplan4_phase1_5_results.md`