# AC Plan 10 Fire Checkpoint Selector Report

対象: `documents/acplan_10.md`  
実行日: 2026-04-30  
対象 fold: `5`  
GPU: `NVIDIA GeForce RTX 3070`  
実験: seed11 WM/BC + seed11 restricted AC, checkpoint 250/300/350/400/450

## 結論

`fire_pnl-aware checkpoint selector` の基盤実装と fold5 検証は完了。

判定:

```text
不採用。
```

理由:

```text
fire_pnl / fwd16 / incr16 は正。
long <= 3%, short = 0%, turnover <= 3.5 も満たす。
ただし通常評価経路で MaxDDDelta が +0.04pt に残り、必須条件 MaxDDDelta <= 0 を満たせない。
```

E 系列は Alpha / Sharpe は強いが、acplan_10 の意図通りまだ安全採用不可。

## 本流に入れた実装

コミット:

```text
6f931a0 Add fire-aware AC checkpoint selector
bc111aa Allow fire probe checkpoint file selection
b10462b Do not resume AC when starting from test
```

追加/変更:

```text
unidream/experiments/policy_fire.py
  - adapter fire 判定
  - fire_pnl
  - fire forward return
  - forward incremental pnl
  - fire selector guard/score

unidream/actor_critic/imagination_ac.py
  - save_step_checkpoints
  - checkpoint_eval_fn
  - accepted checkpoint を ac_fire_best.pt として保存
  - fire_best があれば val_best より優先して復元

unidream/experiments/ac_stage.py
  - validation 上で fire selector を呼べるように接続
  - start-from test 時は AC を再開せず、checkpoint load のみに修正

unidream/cli/ac_fire_timing_probe.py
  - `checkpoint_dir@ac_step400.pt:ac` のように任意 AC checkpoint を probe 可能化
```

`configs/trading.yaml` は変更していない。`fire_checkpoint_selector` は default off なので、現本流の `uv run python -m unidream.cli.train` の主経路は維持。

## 実験設定

一時 config:

```text
configs/acplan10_fire_selector_s011_tmp.yaml
```

主な差分:

```yaml
logging.checkpoint_dir: checkpoints/acplan10_fire_selector_s011_on_wmbc_s011
ac.checkpoint_interval: 50
ac.save_step_checkpoints: true
ac.step_checkpoint_prefix: ac_step
ac.fire_checkpoint_selector.enabled: true
ac.fire_checkpoint_selector.maxdd_delta_max_pt: 0.0
ac.fire_checkpoint_selector.alpha_floor_pt: 0.0
ac.fire_checkpoint_selector.sharpe_floor: 0.0
ac.fire_checkpoint_selector.long_max: 0.03
ac.fire_checkpoint_selector.short_max: 0.0
ac.fire_checkpoint_selector.turnover_max: 3.5
```

seed11 WM/BC は以下からコピー:

```text
checkpoints/acplan9_wmbc_s011_noac/fold_5/world_model.pt
checkpoints/acplan9_wmbc_s011_noac/fold_5/bc_actor.pt
```

生成 checkpoint:

```text
ac_step250.pt
ac_step300.pt
ac_step350.pt
ac_step400.pt
ac_step450.pt
ac.pt
ac_best.pt
```

`ac_fire_best.pt` は生成されなかった。つまり validation selector の strict guard を通過した checkpoint はなかった。

## 通常評価結果

`unidream.cli.train --start-from test` の通常評価経路を正とする。

| checkpoint | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | fire | fire_pnl | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| final `ac.pt` | +60.78 | +0.049 | +0.04 | 1.17 | 2% | 0% | 3.1% | +0.1271 | MaxDD条件外 |
| step400 `ac_step400.pt` | +61.18 | +0.049 | +0.04 | 1.16 | 2% | 0% | 3.2% | +0.0837 | MaxDD条件外 |

読み:

```text
Alpha / Sharpe / turnover / long / short は問題なし。
しかし MaxDDDelta が +0.04pt で止まり、Plan10 採用条件を満たさない。
```

## Fire Probe 結果

各 checkpoint を別プロセスで単独 probe した結果。

| label | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | fire_pnl | fwd16 | incr16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc | +62.36 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1060 | +0.00410 | +0.00036 |
| step250 | +62.31 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1060 | +0.00410 | +0.00036 |
| step300 | +62.27 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1060 | +0.00410 | +0.00036 |
| step350 | +62.18 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1059 | +0.00410 | +0.00036 |
| step400 | +62.14 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1059 | +0.00410 | +0.00036 |
| step450 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1059 | +0.00410 | +0.00036 |
| final | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 3.3% | +0.1059 | +0.00410 | +0.00036 |

probe の読み:

```text
fire timing 自体は悪くない。
fire_pnl, fwd16, incr16 は全て正。
AC step による差は小さく、MaxDDDelta +0.04pt がほぼ固定で残る。
```

したがって問題は「fire が負」ではなく、

```text
良いfireでAlphaを稼げているが、DD悪化を相殺できる checkpoint が見つからない
```

という状態。

## 追加で修正した評価挙動

検証中に、`--start-from test` で未完了 AC checkpoint を評価しようとすると、`global_step < ac.max_steps` の場合に残り step を再学習する挙動を確認した。

これは checkpoint 評価では危険なので修正済み。

```text
変更前:
  --start-from test でも ac_step400.pt が max_steps=450 なら 400->450 を再学習する

変更後:
  --start-from test は AC checkpoint を load するだけ
  AC を続きから学習したい場合は --start-from ac を使う
```

## 採用判断

採用:

```text
fire selector / step checkpoint の実装基盤
任意 AC checkpoint を probe する CLI 拡張
start-from test の no-resume 修正
```

不採用:

```text
seed11 fire selector checkpoint
acplan10_fire_selector_s011_tmp config
fire_checkpoint_selector の configs/trading.yaml 有効化
```

理由:

```text
MaxDDDelta <= 0 を満たす checkpoint が通常評価で見つからない。
```

## 次の仮説

今の結果から、単純な checkpoint selector だけでは MaxDD +0.04pt を消せない。
次は selector ではなく、fire を「良いforward return」だけでなく「DD悪化局面を避ける」方向に寄せる必要がある。

候補:

```text
1. fire-time drawdown contribution guard
   fire bar または fire 直後の equity drawdown contribution が悪いものを抑制する。

2. validation selector の評価指標を official test path と同じ stateful inference に統一
   probe は診断用。採用判定は train/test path の通常評価を正にする。

3. fire_pnl-aware selector は残すが、採用条件に max drawdown contribution を追加
   fire_pnl > 0 だけでは DD 条件を通せない。
```

まだ禁止維持:

```text
route head unlock
full actor AC
advantage gate 緩和
floor > 1.0
scale grid selector
Q argmax actor update
```
