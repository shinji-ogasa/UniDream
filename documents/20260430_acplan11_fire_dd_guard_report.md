# AC Plan 11 Fire-Time Drawdown Guard Report

対象: `documents/acplan_11.md`  
実行日: 2026-04-30  
対象: fold5 / seed11 WM-BC + seed11 restricted AC 系  
参照 checkpoint: `checkpoints/acplan10_fire_selector_s011_on_wmbc_s011`

## 結論

Plan11 の Phase 11/12、つまり

```text
fire-time drawdown attribution
inference-only DD guard screening
```

は完了。

判定:

```text
DD guard はまだ不採用。
```

理由:

```text
E系 current は Alpha/Sharpe/fire は強い。
しかし MaxDDDelta +0.04pt が残る。

future-leaky な oracle_mdd_interval なら
  AlphaEx +64.34 / SharpeDelta +0.058 / MaxDDDelta +0.00
まで行く。

つまりDD悪化は局所的で、理論上は削れる。
ただし deployable な guard では、MaxDDを通すと Alpha がPlan7未満まで落ちる。
```

現時点で `configs/trading.yaml` や actor 本体には DD guard を採用していない。
本流採用は診断CLIのみ。

## 実装コミット

```text
e0d36e8 Add AC fire drawdown guard probe
f84fadb Expand fire DD guard threshold sweep
99b1605 Add fine pre-drawdown guard sweep
2877dce Add worsening drawdown fire guard probe
```

追加CLI:

```powershell
uv run python -m unidream.cli.ac_fire_dd_guard_probe
```

出力:

```text
documents/20260430_acplan11_fire_dd_guard_fold5_E.md
```

## 検証対象

```text
run: E
checkpoint: checkpoints/acplan10_fire_selector_s011_on_wmbc_s011/fold_5/ac.pt
fold: 5
seed: 11
device: cuda
```

比較した guard:

```text
current
no_adapter
delta_scale 0.75 / 0.50 / 0.25
pre_dd threshold 1% ... 24%
pre_dd_worsening threshold 20% ... 23%
cooldown 8 / 16 / 32 / 64
pred_dd q50 / q75 / q90
oracle_mdd_interval  # future-leaky diagnostic only
oracle_postmin32     # future-leaky diagnostic only
```

## 主要結果

| variant | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | fire_pnl | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1059 | MaxDD条件外 |
| delta_scale_0.75 | +46.26 | +0.038 | +0.03 | 0.89 | 1.6% | 0.0% | 3.3% | +0.1031 | MaxDD条件外 |
| delta_scale_0.50 | +30.60 | +0.025 | +0.02 | 0.61 | 1.2% | 0.0% | 3.3% | +0.1002 | Alpha/Sharpe不足 |
| pre_dd_22.50% | +35.89 | +0.025 | -0.03 | 1.45 | 1.8% | 0.0% | 3.2% | +0.0732 | MaxDD通過だがPlan7未満 |
| pre_dd_23.00% | +50.95 | +0.039 | +0.06 | 1.50 | 1.8% | 0.0% | 3.2% | +0.0935 | MaxDD条件外 |
| pre_dd_worsening_22.00% | +52.47 | +0.042 | +0.06 | 1.81 | 1.8% | 0.0% | 3.2% | +0.0964 | MaxDD条件外 |
| cooldown_32 | +10.80 | +0.011 | -0.06 | 1.91 | 0.1% | 0.0% | 0.2% | +0.0095 | Alpha不足 |
| pred_dd_q75 | +68.68 | +0.062 | +0.04 | 3.91 | 1.1% | 0.0% | 2.5% | +0.1221 | MaxDD/turnover条件外 |
| oracle_mdd_interval | +64.34 | +0.058 | +0.00 | 0.68 | 1.0% | 0.0% | 1.8% | +0.1026 | future leakなので不採用 |

Plan7 fold5 baseline は過去レポート上で概ね以下。

```text
AlphaEx +41.31
SharpeDelta +0.026
MaxDDDelta -0.25
turnover 1.78
long 2%
short 0%
```

採用条件は `AlphaEx > Plan7`, `SharpeDelta > Plan7`, `MaxDDDelta <= 0`, `turnover <= 3.5`, `long <= 3%`, `short = 0%`。
この条件を deployable guard で満たす候補はなかった。

## Fire-Time DD Attribution

current の最大DD区間:

```text
mdd_peak: 5027
mdd_trough: 6453
fire_in_mdd: 132 bars
```

DD改善に最も効いた fire run:

| start | end | len | fwd16 | incr16 | suppress時MaxDD改善 | alpha_loss | cf_alpha | cf_MaxDDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6451 | 6609 | 158 | +0.00658 | +0.00068 | +0.064 | +60.47 | +1.62 | -0.03 |

読み:

```text
最大DDを悪化させる主fireは、同時にAlphaの主エンジンでもある。
単純にこのrunを消すとMaxDDは通るが、Alphaもほぼ消える。
```

一方で `oracle_mdd_interval` は最大DD区間内のfireだけを消し、区間後のfireを残すため、Alphaを維持しつつMaxDDを通せた。
これは重要。

```text
DD悪化は局所的。
ただし「どこまでがDD悪化区間か」を未来情報なしで切るのが難しい。
```

## 採用判断

採用:

```text
ac_fire_dd_guard_probe CLI
fire run別のDD寄与診断
pre_dd / worsening / cooldown / predictive-DD / oracle upper-bound sweep
```

不採用:

```text
delta_scale guard
pre_dd guard
pre_dd_worsening guard
cooldown guard
pred_dd guard
oracle_mdd_interval guard
```

理由:

```text
deployable guardは採用条件を満たさない。
oracle系はfuture leakなので本流不可。
```

## 次の仮説

Plan11 で分かったことは、単純な「fireを弱める」ではダメということ。
必要なのは、最大DD区間の入口と出口を、未来情報なしで近似する state machine。

次に試すならこれ。

```text
Phase 12-B:
  drawdown recovery state machine guard

状態:
  normal
  deep_drawdown_block
  recovery_probe
  reenabled

ルール案:
  1. equity DD が 22% を超えたら adapter fire をblock
  2. 直近N本のequity slopeが正、またはpeakからの回復率が一定以上ならprobe再開
  3. probe後にDDが再悪化したらcooldown
  4. adapter deltaは再開直後だけ0.25〜0.5に縮小
```

狙い:

```text
pre_dd_22.5 はMaxDDを通すがAlpha不足。
pre_dd_23 はAlphaを残すがMaxDD悪化。
なので固定閾値ではなく、DD後の再開タイミングをstatefulに制御する。
```

まだ禁止:

```text
full actor AC
route head unlock
advantage gate緩和
floor > 1.0 一律適用
scale grid selector
Q argmax actor update
```

## 最終判断

```text
Plan11 の fire-time DD attribution は有益。
E系のDD悪化は局所的に切れる可能性がある。
ただし今回の deployable guard では採用条件未達。
次は drawdown recovery state machine guard を probe する。
```
