# BC predictive-state 最適化結果

作成日: 2026-04-25 JST

## 結論

`TransformerWorldModel` の予測 head 出力を BC/Actor に渡す実装は完了した。ただし fold4 の検証では、現時点の予測特徴をそのまま BC に渡すと underweight 側へ寄りやすく、M2候補としては採用しない。

最も重要な切り分け結果は以下。

- 同一WM checkpointの no-pred baseline は `short=3% / flat=97%`, `AlphaEx=-0.54pt/yr`, `collapse_guard=pass`。
- 予測特徴を direct concat すると、値を `scale=0` にしても `short=92-99%` へ寄る。これは特徴値以前に Actor 入力次元変更が学習を不安定化していた。
- `adapter` 接続に変えると `scale=0` では `short=22% / flat=78%` まで戻り、collapse guard は通る。
- ただし実際の予測値を入れると、`scale=0.25` で `short=68%`, `scale=1.0` で `short=99%` になり、予測値は主に de-risk/underweight 方向へ使われている。
- BCがM2候補として十分ではないため、ACチューニングは未実行。崩れたBCをACで回すと、offline RL側で collapse を増幅するリスクが高い。

## 実装したこと

### 1. WM予測headをBC/Actorへ渡す経路

追加/変更:

- `unidream/world_model/train_wm.py`
  - `predict_auxiliary_from_encoded()` を追加。
  - `predictive_feature_names()` を追加。
- `unidream/experiments/predictive_state.py`
  - train/val/test の encoded `z/h` から `return`, `vol`, `drawdown` 予測を作る。
  - train統計で標準化、clip、scale を適用する。
- `unidream/cli/train.py`
  - `ac.use_wm_predictive_state: true` のとき、BC/AC に渡す conditioning vector として予測特徴を生成する。
- `unidream/experiments/bc_setup.py`
  - Actor の `advantage_dim` を多次元予測stateに対応。

Actorへ渡す特徴:

```text
predictive_state = [pred_return_h1,h4,h8,h16,h32,
                    pred_vol_h1,h4,h8,h16,h32,
                    pred_drawdown_h1,h4,h8,h16,h32]
```

### 2. BCの多次元conditioning対応

追加/変更:

- `unidream/actor_critic/bc_pretrain.py`
  - chunked BC の `advantage_values` を `(T,)` だけでなく `(T, D)` に対応。
  - 代表step抽出で `(n_chunks, k, D)` を扱えるようにした。

修正理由:

- 最初の予測state実行は `ValueError: cannot reshape array of size ... into shape (17502,4)` で失敗した。
- 原因は chunked BC が conditioning を1次元スカラー前提にしていたこと。

### 3. Actorの予測state接続方式

追加/変更:

- `unidream/actor_critic/actor.py`
  - `advantage_input_mode` を追加。
  - `concat`: 既存のように trunk input に直結。
  - `adapter`: trunk input 次元は変えず、zero-init adapter で hidden に加算。
- `unidream/experiments/bc_setup.py`
  - `use_wm_predictive_state: true` のデフォルトは `adapter` にした。
  - direct concat 実験configには再現用に `advantage_input_mode: concat` を明記。

採用判断:

- direct concat は入力次元変更だけで underweight へ寄るため危険。
- adapter は no-pred挙動を壊しにくいので、今後使うなら adapter を前提にする。

### 4. AC側の受け渡し

追加/変更:

- `unidream/actor_critic/imagination_ac.py`
- `unidream/experiments/ac_stage.py`

内容:

- `set_oracle_data(..., advantage_values=...)` に対応。
- BC anchor, prior anchor, imagination rollout, actor logging で予測stateを渡せるようにした。
- 今回はBCが採用ライン未達のため、AC実行はスキップした。

## Web調査からの判断

- IQL は policy extraction に advantage-weighted BC を使う。つまり「全teacher行動を同じ強さで真似る」より、価値のある行動を強く学ぶ方向が妥当。今回の結果も、予測特徴を足すだけではなく action/transition別の advantage 重みが必要という判断を支持する。Source: https://arxiv.org/abs/2110.06169
- TD3+BC は offline RL で policy を dataset action から外しすぎないために BC term を入れる。したがって、BC段階で collapse しているものをACで無理に回すのは危険。Source: https://arxiv.org/abs/2106.06860
- DAgger は逐次予測で自分の過去actionが将来状態分布を変える問題を扱う。UniDreamの inventory state 分布ズレにも該当するが、前回の self-conditioned BC 実験では active decision が消えたため、係数を弱めた再設計が必要。Source: https://arxiv.org/abs/1011.0686

## 実験条件

共通:

```text
symbol: BTCUSDT
interval: 15m
range: 2020-01-01 to 2024-01-01
fold: 4
stage: bc -> test
WM checkpoint: checkpoints/bc_multitask_aux_predstate_s007/fold_4/world_model.pt
AC max_steps: 0
```

注意:

- `uv` 実行末尾に HMM の `Model is not converging` 警告が出るが、train/test metrics は出力済み。
- 最初の no-pred run はWMを再学習してしまったため、公平比較用に同一WM checkpointの no-pred baseline を追加で回した。

## 実験結果

| 実験 | 接続 | heads | scale | 追加制約 | alpha_excess | sharpe_delta | maxdd_delta | dist | turnover | cost | guard | 判定 |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---|---|
| no-pred fair | none | none | - | なし | -0.54pt | -0.013 | -0.67pt | short 3% / flat 97% | 0.99 | 0.0011 | pass | 比較基準 |
| pred all direct | concat | return/vol/dd | 1.0 | なし | -0.16pt | -0.003 | -0.83pt | short 91% / flat 9% | 0.22 | 0.0007 | directional collapse | 不採用 |
| pred risk direct | concat | vol/dd | 1.0 | なし | -3.29pt | -0.086 | -0.57pt | short 100% / flat 0% | 28.67 | 0.0163 | pass | 不採用 |
| pred all direct weakrec | concat | return/vol/dd | 1.0 | weak short-copy + recovery | -5.75pt | -0.150 | +0.22pt | short 24% / flat 76% | 34.17 | 0.0193 | pass | 不採用 |
| pred risk direct weakrec | concat | vol/dd | 1.0 | weak short-copy + recovery | -8.40pt | -0.223 | +0.62pt | short 18% / flat 82% | 35.80 | 0.0202 | pass | 不採用 |
| pred all direct scale0 | concat | return/vol/dd | 0.0 | なし | -0.16pt | -0.003 | -0.84pt | short 99% / flat 1% | 0.22 | 0.0007 | directional collapse | 入力次元問題の証拠 |
| pred all direct scale0 zero-init | concat | return/vol/dd | 0.0 | first layer zero-init | -0.15pt | -0.003 | -0.83pt | short 92% / flat 8% | 0.21 | 0.0007 | directional collapse | 不十分 |
| pred all adapter scale0 | adapter | return/vol/dd | 0.0 | adapter zero-init | -0.55pt | -0.013 | -0.75pt | short 22% / flat 78% | 0.99 | 0.0011 | pass | 接続方式として採用候補 |
| pred all adapter scale0.25 | adapter | return/vol/dd | 0.25 | adapter zero-init | -0.54pt | -0.012 | -0.78pt | short 68% / flat 32% | 0.99 | 0.0011 | pass | 予測値は強すぎ |
| pred all adapter scale1 | adapter | return/vol/dd | 1.0 | adapter zero-init | -0.47pt | -0.011 | -0.82pt | short 99% / flat 1% | 0.99 | 0.0011 | pass | underweight過多 |

## 解釈

### 1. 予測特徴は「相場状態の分離」には使えているが、BCではde-risk側に使われすぎる

前回のWM検証では、multitask auxで `vol_h16 AUC` や `return_h32 IC` が改善した。今回のBCでは、その情報が `benchmark復帰` や `active alpha` よりも underweight 判断に寄って使われている。

この挙動は次の形で見える。

```text
scale=0.0 adapter: short 22%
scale=0.25 adapter: short 68%
scale=1.0 adapter: short 99%
```

つまり、予測値の注入量に応じて underweight が増える。これは「予測特徴が無意味」ではなく、「今のBC lossが予測特徴を risk-off shortcut として使っている」という状態。

### 2. direct concat は使わない

`scale=0` でも direct concat は short collapse した。これは予測値そのものではなく、Actor trunk の入力次元変更、初期化、学習経路が挙動を壊している。

今後の予測特徴接続は以下に限定する。

```yaml
ac:
  use_wm_predictive_state: true
  advantage_input_mode: adapter
  wm_predictive_state_scale: 0.1  # まずは 0.0-0.25 の範囲
```

ただし今回の結果では `scale=0.25` でも active decision と alpha は改善していないため、まだ本線化しない。

### 3. weak recovery / weak short-copy はこの組み合わせでは不採用

`weak short-copy + recovery` は short 比率を落とすが、turnover が `34-36` まで増え、cost が `0.019-0.020` に悪化した。これは前回のBC再設計実験Bと同じ失敗パターン。

原因は、recoveryを促すだけでは「いつ戻るべきか」を価値で選別できず、取引回数だけ増えるため。

### 4. ACチューニングを回さなかった理由

ACへ進む最低条件は次のどれか。

```text
collapse_guard pass
alpha_excess >= 0 近辺
underweight比率が過大でない
turnover/costが悪化していない
```

今回の候補は以下のどちらかに分かれた。

- no-pred/adapter低scale: 安定だが alpha が負で、ほぼ benchmark hold。
- 予測値あり: alpha は少し改善しても underweight 過多。

この状態でACを回すと、TD3+BC系の観点でも「悪いBC priorを微調整する」だけになりやすい。よって今回はAC未実行にした。

## 採用/不採用

採用する実装:

- WM predictive state bundle 生成。
- BC/AC の多次元 conditioning 対応。
- Actor `advantage_input_mode: adapter`。
- `wm_predictive_state_scale`。

本線採用しないconfig:

- direct concat 方式の全config。
- risk-only predictive state。
- weak recovery + predictive state。
- adapter scale 1.0。

暫定ベースライン:

```text
configs/bc_multitask_aux_nopred_fairwm_s007.yaml
```

ただしこれはM2候補ではなく、予測state実験の比較基準。

## 次にやるべきこと

次の改善は「予測特徴をそのままactorに渡す」ではなく、行動価値を分解して loss に入れる方向。

1. action/transition別の `cost_adjusted_advantage` を実装する。
2. underweight, benchmark recovery, overweight を別ラベル/別headに分ける。
3. 予測特徴は `de-risk gate` と `recovery gate` にだけ渡し、target位置全体へ直結しない。
4. `w_adv = exp(clip(adv / tau))` 型の advantage-weighted BC を、position lossではなく transition loss に適用する。
5. fold4で `alpha_excess >= 0`, `short <= 70%`, `turnover <= 2`, `maxdd_delta <= 0` を満たしてから fold0/fold5 に広げる。

## 主要ログ

- `documents/logs/20260425_bc_multitask_aux_nopred_fairwm_fold4.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_fold4_rerun.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_riskonly_fold4.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_all_weakrecovery_fold4.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_riskonly_weakrecovery_fold4.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_all_scale0_adapter_fold4.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_all_scale025_adapter_fold4.log`
- `documents/logs/20260425_bc_multitask_aux_predstate_all_adapter_fold4.log`

## 検証コマンド

```powershell
uv run python -m py_compile `
  unidream\actor_critic\actor.py `
  unidream\actor_critic\bc_pretrain.py `
  unidream\actor_critic\imagination_ac.py `
  unidream\cli\train.py `
  unidream\experiments\bc_setup.py `
  unidream\experiments\ac_stage.py `
  unidream\experiments\predictive_state.py `
  unidream\world_model\train_wm.py
```

結果: compile OK。
