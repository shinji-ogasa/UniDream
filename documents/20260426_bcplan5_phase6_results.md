# BC Plan 5 Phase 6 Results

作成日: 2026-04-26  
対象: `documents/bcplan_5.md`  
実行範囲: BTCUSDT 15m / seed 7 / fold 4 / 2018-01-01 to 2024-01-01  
文字コード: UTF-8

## 結論

Phase 6A/6B/6C を実装・実行した。AC には移行しない。

理由は次の通り。

- 6A/6B は underweight collapse を止めたが、flat 100% / recovery 0% に潰れた。
- 6C は recovery route を 4% まで出せたが、turnover 14.55 で合格条件を大きく超えた。
- 6C relaxed は recovery route 18%、short 35%、flat 65% まで出たが、turnover 12.79 で不合格。
- したがって、Phase 6 の safety/recovery fallback は「方向の崩壊」は止められるが、まだ execution が安定していない。

## 実装内容

### Inference-Time Safety

`Actor` に推論時 route safety を追加した。

- `min_route_confidence_for_active`
- `low_confidence_neutral_fallback`
- `active_rate_max`
- `short_underweight_rate_max`
- `rate_cap_active_eps`

route controller の active 判定は評価指標と合わせて `0.05` を使うようにした。最初は `1e-6` を使っていたため、微小なズレまで active と数えて cap を早期に使い切り、flat 100% を誘発していた。これは修正済み。

### Recovery Fallback

`Actor` に recovery fallback を追加した。

```text
if current_position < benchmark - gap
and underweight_duration > D
and de_risk_confidence <= threshold:
    boost recovery logit
    optionally lower de_risk logit
```

追加した主な設定:

- `recovery_fallback_gap`
- `recovery_fallback_min_duration`
- `recovery_fallback_derisk_conf_max`
- `recovery_logit_boost`
- `de_risk_duration_logit_down`

### Train-Time Exposure Loss

`BCPretrainer` に route distribution 制約を追加した。

```text
loss += λ_active * max(0, active_rate - active_max)^2
loss += λ_short * max(0, de_risk_rate - short_max)^2
loss += λ_neutral * |neutral_rate - neutral_target|
```

追加した主な設定:

- `route_active_rate_coef`
- `route_active_rate_max`
- `route_short_rate_coef`
- `route_short_rate_max`
- `route_neutral_rate_coef`
- `route_neutral_rate_target`

## 実行 Config

| ID | Config | ベース | 内容 |
|---|---|---|---|
| 6A | `configs/bcplan5_phase6a_safety_s007.yaml` | Phase 1 checkpoint | inference safety only |
| 6B | `configs/bcplan5_phase6b_exposure_loss_s007.yaml` | Phase 1 warm start | exposure loss + inference safety |
| 6C | `configs/bcplan5_phase6c_recovery_fallback_s007.yaml` | Phase 3 checkpoint | recovery fallback + confidence safety |
| 6C relaxed | `configs/bcplan5_phase6c_relaxed_recovery_s007.yaml` | Phase 3 checkpoint | confidence fallback off / cap 35% / stronger recovery |

## Backtest Results

| ID | Route dist | Position dist | AlphaEx | SharpeΔ | MaxDDΔ | Turnover | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| 6A | neutral 95 / de_risk 5 / recovery 0 / overweight 0 | long 0 / short 0 / flat 100 | +0.53 pt/yr | -0.009 | -1.02 pt | 3.77 | flat 100%, recovery 0% で不可 |
| 6B | neutral 100 / de_risk 0 / recovery 0 / overweight 0 | long 0 / short 0 / flat 100 | +0.27 pt/yr | +0.003 | -0.52 pt | 2.44 | flat 100%, recovery 0% で不可 |
| 6C | neutral 56 / de_risk 40 / recovery 4 / overweight 0 | long 0 / short 20 / flat 80 | +0.24 pt/yr | -0.045 | -0.51 pt | 14.55 | recovery は出たが turnover 高すぎ |
| 6C relaxed | neutral 32 / de_risk 50 / recovery 18 / overweight 0 | long 0 / short 35 / flat 65 | +0.47 pt/yr | -0.024 | -1.00 pt | 12.79 | 分布は近いが turnover 高すぎ |

## Route Probe Results

route probe は raw route classifier を見る診断で、inference safety 後の route dist とは別。

| ID | CE | Acc | Macro-F1 | Active Recall | False Active | ECE | Top Active Adv |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6A | 1.1043 | 0.618 | 0.350 | 0.528 | 0.147 | 0.052 | 0.011263 |
| 6B | 1.1507 | 0.615 | 0.347 | 0.414 | 0.043 | 0.104 | 0.011836 |
| 6C | 0.9730 | 0.608 | 0.401 | 0.603 | 0.239 | 0.014 | 0.011903 |

## Per-Route Test Recall

| ID | Neutral | De-risk | Recovery | Overweight |
|---|---:|---:|---:|---:|
| 6A | 0.853 | 0.732 | 0.000 | 0.000 |
| 6B | 0.957 | 0.602 | 0.000 | 0.000 |
| 6C | 0.761 | 0.768 | 0.110 | 0.062 |

## 読み取り

### 1. 6A は safety が強すぎる

6A は Phase 1 の underweight を止めたが、最終的に flat 100% になった。route dist も active 4.6% まで落ちている。これは `min_route_confidence_for_active=0.55` と neutral fallback が強く、active route をほぼ潰している。

### 2. 6B の exposure loss は neutral collapse を強めた

6B は raw route probe でも neutral 予測が増えた。False active は 0.043 まで下がったが、active recall も 0.414 に落ちた。結果として、BC は安全側に寄りすぎ、実行時は flat 100% になった。

### 3. 6C の recovery fallback は効く

6C は recovery route 4%、6C relaxed は recovery route 18% まで出た。つまり、recovery を route softmax の学習だけに任せるより、inventory state 依存の fallback を入れる方が効く。

ただし、recovery を出すと turnover が急増した。これは recovery fallback が「戻る」こと自体は作れているが、de_risk/recovery の往復を止める min-hold / hysteresis が不足しているということ。

### 4. 現状の失敗モードは short collapse から churn collapse に変わった

Phase 2/3/5 は short/underweight に張りっぱなしだった。Phase 6C はそれを止めた代わりに、de_risk と recovery を行き来して turnover が増えた。

これは前進ではあるが、AC に渡すにはまだ危険。

## 合格条件との照合

`bcplan_5.md` の合格目安:

```text
short/underweight <= 35〜45%
flat 50〜85%
recovery > 0%
turnover <= 4.5
AlphaEx >= 0
MaxDDΔ <= 0
```

6C relaxed は short/flat/recovery/AlphaEx/MaxDDΔ は満たすが、turnover 12.79 で失格。6C は recovery/AlphaEx/MaxDDΔ は満たすが、turnover 14.55 と SharpeΔ -0.045 が弱い。6A/6B は turnover は良いが flat 100% と recovery 0% で失格。

## AC 移行判定

AC には移行しない。

理由:

- 6A/6B は flat collapse。
- 6C/6C relaxed は turnover collapse。
- SharpeΔ は 6B を除いて負。6B も flat 100% なので無効。
- recovery fallback は効いたが、execution 制御が未完成。

## 次にやるべきこと

Phase 7 は、recovery を softmax route の一部として扱うのではなく、inventory controller 側へ分離するべき。

最小実装案:

```text
exposure_route:
  neutral / de_risk / overweight

inventory_controller:
  if underweight:
      hold_underweight / recover_to_benchmark
  if overweight:
      hold_overweight / reduce_to_benchmark
```

さらに、de_risk/recovery の往復を止めるために以下が必要。

```text
recovery_min_hold_bars
recovery_cooldown_bars
de_risk_cooldown_after_recovery
hysteresis_gap_enter
hysteresis_gap_exit
max_recovery_turnover_per_day
```

今回の結果から、次の実験は `6C relaxed` をベースに turnover 制約だけを足すより、route 設計を state machine 化する方が妥当。単純な logit boost では recovery は出せても churn を止められない。

## 生成物

- `configs/bcplan5_phase6a_safety_s007.yaml`
- `configs/bcplan5_phase6b_exposure_loss_s007.yaml`
- `configs/bcplan5_phase6c_recovery_fallback_s007.yaml`
- `configs/bcplan5_phase6c_relaxed_recovery_s007.yaml`
- `documents/logs/20260426_bcplan5_phase6a_safety_fold4_v2.log`
- `documents/logs/20260426_bcplan5_phase6b_exposure_loss_fold4_v2.log`
- `documents/logs/20260426_bcplan5_phase6c_recovery_fallback_fold4_v2.log`
- `documents/logs/20260426_bcplan5_phase6c_relaxed_recovery_fold4_v2.log`
- `documents/route_probe/20260426_phase6a_route_probe_fold4.md`
- `documents/route_probe/20260426_phase6b_route_probe_fold4.md`
- `documents/route_probe/20260426_phase6c_route_probe_fold4.md`
- `documents/20260426_bcplan5_phase6_results.md`