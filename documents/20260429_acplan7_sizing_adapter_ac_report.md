# AC Plan 7 Sizing Adapter AC Report

対象計画: `documents/acplan_7.md`

実行対象: checkpoint が存在する fold 0 / 4 / 5。比較基準は `documents/20260428_acplan6_multifold_attribution_report.md` の Phase 8 safe baseline。

## 結論

Phase D-0 から D-2 まで完了。

採用する:

```text
Phase 8 safe baseline
+ trainable benchmark-overweight sizing adapter
+ critic-only 200 step
+ sizing-adapter-only AC 250 step
+ val_adjust_rate_scale = 0.5 固定
```

採用理由:

```text
3fold平均 AlphaEx: +13.48 -> +13.91 pt/yr  (+0.43)
3fold平均 SharpeDelta: +0.034 -> +0.035
3fold平均 MaxDDDelta: -0.30 -> -0.30 pt
turnover max: 2.55 <= 3.5
long max: 2% <= 3%
short: 0%
```

ただし SharpeDelta の改善幅は +0.001 と小さい。採用は可能だが、次に checkpoint が増えたら fold 拡張で再確認する。

## 実装コミット

```text
bc2f3a9 Add trainable overweight sizing adapter
cad94ae Add AC plan 7 candidate Q probe
e450a9a Adopt sizing adapter AC curriculum
```

## Phase D-0: isolated trainable overweight sizing adapter

実装内容:

```text
actor に benchmark_overweight_sizing_adapter を追加
初期値は weight/bias = 0
既存の benchmark overweight gate / advantage gate / long cap / floor を維持
trainable_actor_prefixes で sizing adapter だけ学習可能にする
旧checkpointには optional_missing として互換対応
```

計算:

```text
base_epsilon = 0.20
delta = 0.03 * tanh(sizing_adapter(hidden))
epsilon = clamp(base_epsilon + delta, 0, max_position - benchmark)
target = max(target, benchmark + epsilon)
target = clamp(target, benchmark, benchmark_overweight_max_position)
```

重要なのは、adapter が route や exposure policy 全体を直接動かさないこと。触れるのは benchmark 近傍で gate を通過した時の overweight sizing だけ。

## Phase D-1: critic-only candidate Q probe

実行コマンド:

```powershell
uv run python -u -m unidream.cli.ac_candidate_q_probe `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --steps 400 `
  --batch-size 1024 `
  --ensemble-size 1 `
  --output-json documents\acplan7_candidate_q_probe.json `
  --output-md documents\20260429_acplan7_candidate_q_probe.md `
  --device cuda
```

結果要約:

| fold | variant例 | test row Spearman | selected adv vs anchor | selected short | selected flat | 判断 |
|---:|---|---:|---:|---:|---:|---|
| 0 | mse / cql系 | 0.371 | -0.000014 | 0% | 100% | Q rankingはあるが改善action選択は弱い |
| 4 | mse / cql系 | 0.315 | -0.000110 | 0% | 100% | flat選択に寄る |
| 5 | mse / cql系 | 0.301 | -0.000026 | 0% | 100% | flat選択に寄る |
| 0 | rank CE系 | 0.258 | -0.000202 | 92% | 8% | short collapse傾向 |
| 4 | rank CE系 | 0.130 | -0.000314 | 47% | 53% | short bias |
| 5 | rank CE系 | 0.049 | -0.000276 | 88% | 12% | short collapse傾向 |

判断:

```text
Q argmax型ACはまだ危険。
critic は状態内rankを少し拾えているが、selected advantage は正に出ていない。
rank loss系は short 偏りが強く、route unlock / full actor unlock の根拠にならない。
```

この結果に基づき、D-2 は full actor ではなく sizing-adapter-only に限定した。

## Phase D-2: sizing-adapter-only AC

学習構成:

```yaml
ac.max_steps: 450
critic_pretrain_steps: 250
curriculum:
  - name: critic_only
    until_step: 200
    critic_only: true
    trainable_actor_prefixes: []
    batch_size: 1024
  - name: sizing_adapter_only
    until_step: 450
    critic_only: false
    trainable_actor_prefixes:
      - benchmark_overweight_sizing_adapter
    actor_lr: 1.0e-05
    batch_size: 1024
    td3bc_alpha: 3.0
    prior_kl_coef: 0.2
    prior_trade_coef: 0.1
    prior_band_coef: 0.05
    prior_flow_coef: 0.3
    turnover_coef: 0.35
    entropy_scale: 0.005
```

固定したもの:

```text
benchmark exposure floor = 1.0
state machine route
route gate
predictive advantage gate
long cap
full actor / route head
```

### 通常selector

通常の selector では fold4 で scale=0.1 が選ばれ、turnover は下がったが SharpeDelta が基準を下回った。

| metric | result |
|---|---:|
| Aggregate AlphaEx | +13.87 pt/yr |
| Aggregate SharpeDelta | +0.030 |
| Aggregate MaxDDDelta | -0.25 pt |
| BarWin | 11.6% |
| PeriodWin | 38.9% |

判断: 不採用。Alpha は改善したが、SharpeDelta が Phase 8 baseline の +0.034 を下回った。

### scale=0.5 固定

Phase 8 baseline と同じ conservative scale に固定した結果。

| fold | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | PeriodWin | Long | Short | Flat | Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.03 | -0.000 | -0.00 | 0.0% | 0% | 0% | 100% | 0.05 |
| 4 | +0.43 | +0.079 | -0.64 | 66.7% | 1% | 0% | 99% | 2.55 |
| 5 | +41.31 | +0.026 | -0.25 | 50.0% | 2% | 0% | 98% | 1.78 |

Aggregate:

```text
AlphaEx:   +13.91 pt/yr
SharpeΔ:   +0.035
MaxDDΔ:    -0.30 pt
BarWin:    11.0%
PeriodWin: 38.9%
PBO:       0.3333
```

## 採用条件判定

比較対象:

```text
Phase 8 baseline
AlphaEx:   +13.48 pt/yr
SharpeΔ:   +0.034
MaxDDΔ:    -0.30 pt
turnover:  max 2.54
long:      max 2%
short:     0%
```

| 条件 | 結果 | 判定 |
|---|---:|---|
| AlphaEx >= current +0.2 pt | +13.91 vs +13.48 = +0.43 | pass |
| SharpeDelta >= current | +0.035 vs +0.034 | pass |
| MaxDDDelta <= 0 | -0.30 | pass |
| turnover <= 3.5 | max 2.55 | pass |
| long <= 3% | max 2% | pass |
| short = 0% | 0% | pass |

採用判定: pass。ただし SharpeDelta の margin は薄い。

## Web調査による解釈

TD3+BC は offline RL で actor update に BC 制約を足す単純な構成が有効という立場で、今回の `trainable_actor_prefixes` による小さい部品だけの actor update と整合する。  
Source: https://arxiv.org/abs/2106.06860

IQL は Q/V を使った後、policy は advantage-weighted BC として抽出する。今回の方針も Q argmax で直接 route を動かすのではなく、価値がある候補に近い小さい sizing だけを動かす方向に寄せた。  
Source: https://arxiv.org/abs/2110.06169

CQL は offline RL の distribution shift / OOD action 過大評価を問題にする。D-1 で rank系 critic が short に寄ったので、route unlock / full actor unlock はこの問題を踏みやすい。  
Source: https://arxiv.org/abs/2006.04779

## 本流への反映

`configs/trading.yaml` を更新済み。

```text
checkpoint_dir: checkpoints/acplan7_sizing_adapter_s007
ac.max_steps: 450
ac.val_adjust_rate_scale_grid: [0.5]
use_trainable_benchmark_overweight_sizing_adapter: true
benchmark_overweight_trainable_delta_range: 0.03
curriculum:
  critic_only -> sizing_adapter_only
```

再現コマンド:

```powershell
uv run python -m unidream.cli.train `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --start-from ac `
  --stop-after test `
  --seed 7 `
  --device cuda
```

既存の採用済み checkpoint を評価するだけなら:

```powershell
uv run python -m unidream.cli.train `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --start-from test `
  --stop-after test `
  --seed 7 `
  --device cuda
```

## 解除禁止の継続

今回の結果では、次は AC を広げる段階ではない。

継続して禁止:

```text
gate off
advantage threshold 緩和
full actor AC
route head unlock
floor > 1.0 の一律適用
Q argmax 型 route selection
rank CE critic での actor更新
```

理由:

```text
gate_off は fold0/fold5 で大きく崩壊
rank CE critic は short collapse 傾向
normal selector は SharpeDelta を落とした
採用版も SharpeDelta の改善幅は小さい
```

次の改善対象は、AC範囲の拡大ではなく、selector と sizing adapter の安定性確認。

