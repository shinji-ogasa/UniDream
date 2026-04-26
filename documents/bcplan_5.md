Phase 6でやるべき具体案

あたしならPhase 6を3つに分ける。

Phase 6A: inference-time safetyだけ入れる

まず学習を変えずに、実行時制御だけ見る。

base:
  Phase 1

追加:
  route_gate scale 0.25
  min_route_confidence_for_active = 0.55
  active_rate_max = 0.20
  short_underweight_rate_max = 0.35
  low-confidence neutral fallback

目的は、今のモデルのスコアを制御するだけで片側collapseが止まるかを見ること。

これで改善するなら、モデルはそこそこ良いがpolicy selectionが悪かったということ。

合格目安：

short/underweight <= 35〜45%
flat 50〜85%
recovery > 0%
turnover <= 4.5
AlphaEx >= 0
MaxDDΔ <= 0
Phase 6B: train-time exposure lossを入れる

次に、学習時にも分布制約を入れる。

loss += λ_active * max(0, active_rate - active_max)^2
loss += λ_short * max(0, short_rate - short_max)^2
loss += λ_neutral * |neutral_rate - neutral_target|

ここで注意。
短期batchのrate制約はノイズが大きいから、batch全体ではなく moving average / epoch-level validation selector でもいい。

狙いは、

activeを出せ
でも片側に寄るな
自信がないならneutralに戻れ

を学習に入れること。

Phase 6C: recovery fallbackをルールとして入れる

recoveryは学習だけでは弱い。
まず最小ルールで入れてよい。

if current_position < benchmark - gap
and underweight_duration > D
and de_risk_confidence is not very high:
    boost recovery logit

これはズルに見えるかもしれないけど、BCの現段階では必要。
なぜならrecoveryは、「相場予測」ではなく「自分が今どこにいるか」に依存する制御問題だから。

候補：

gap = 0.05〜0.10
D = 8〜32 bars
recovery_logit_boost = 0.25〜1.0

合格目安：

recovery route 1〜5%
recovery latency改善
short/underweight滞在時間低下
turnover急増なし
もっと根本的な修正案

Phase 6でまだ崩れるなら、設計をもう一段変えるべき。

案1: de_riskとrecoveryを同じactive群にしない

今はrouteが、

neutral
de_risk
recovery
overweight

だけど、実際にはrecoveryは他のactiveと性質が違う。

de_risk/overweightは「新規判断」。
recoveryは「既存ポジションの解消」。

だからheadを分ける。

exposure_route:
  neutral / de_risk / overweight

inventory_controller:
  hold_underweight / recover_to_benchmark

つまり、recoveryをroute分類に混ぜない。

まず:
  今のpositionがbenchmarkから外れているか？

外れているなら:
  recovery controllerを優先

外れていないなら:
  exposure routeを選ぶ

この方が自然。
今はrecoveryがde_riskに吸われてるから、同一softmax内で競わせるのが悪い可能性がある。

案2: de_riskの上限を「route率」ではなく「滞在時間」で制御する

de_riskは一回出すだけなら良い。
問題は居続けること。

だから、

de_risk route rate

より、

underweight duration
consecutive de_risk bars
time below benchmark

を制約するべき。

if underweight_duration > D:
    de_risk logit down
    recovery logit up

これで、短期リスク回避は許しつつ、永久underweightを止められる。

案3: route labelをbest routeではなくstate machine化する

金融position制御は状態機械に近い。

benchmark state
  ↓ de_risk
underweight state
  ↓ recovery
benchmark state
  ↓ overweight
overweight state
  ↓ neutral/reduce

今は各時刻独立にrouteを選んでる。
だから、連続性が弱い。

次の発展形は、

current state:
  benchmark / underweight / overweight

allowed transitions:
  benchmark -> de_risk / overweight / neutral
  underweight -> stay_underweight / recovery
  overweight -> stay_overweight / reduce

みたいな position-state-conditioned route にすること。

これは今の「de_risk 99%」をかなり防げる。
underweight状態でさらにde_riskを連発するのを禁止または強く抑制できるから。