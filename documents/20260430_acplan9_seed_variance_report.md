# AC Plan 9 Seed Variance Diagnosis

対象計画: `documents/acplan_9.md`

対象: fold 5。Plan 8で seed 11 AC retrain が崩れたため、まず fold5 だけで原因を分解した。

## 結論

Plan9の最短診断は完了。

```text
C: seed11 WM/BC + ACなし
  -> 崩壊なし。ほぼB&H近似。

D: seed11 WM/BC + seed7 AC
  -> やや悪化。ACを乗せると悪くなる。

E: seed11 WM/BC + seed11 AC
  -> Alpha/Sharpeは大きく改善。ただし MaxDDΔ が +0.04pt で採用条件をわずかに外す。

B: seed7 WM/BC + seed11 AC
  -> Plan8で崩壊確認済み。
```

判断:

```text
BC/WM seedだけが主犯ではない。
ACのcheckpoint selection / fire timing / WM-BC checkpointとの相性が主犯。
```

本流採用:

```text
ac_fire_timing_probe
```

本流不採用:

```text
seed11 AC retrain checkpoint
seed7 AC on seed11 WM/BC checkpoint
seed11 AC on seed11 WM/BC checkpoint
guarded sizing variant
```

理由:

```text
Eは性能だけ見ると強いが、MaxDDΔ <= 0 を満たしていない。
Plan7本流より良く見える単発結果を採用すると、またseed/selector overfitになる。
```

## 実装コミット

```text
86871ed Add AC fire timing probe
ad43ab8 Validate AC fire probe checkpoint loading
```

追加CLI:

```powershell
uv run python -m unidream.cli.ac_fire_timing_probe
```

出力:

```text
documents/20260430_acplan9_fire_timing_probe.md
documents/acplan9_fire_timing_probe.json
```

## Seed Matrix

通常の `unidream.cli.train` test結果を正とする。

| ID | WM/BC | AC | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | PeriodWin | Long | Short | Turnover | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | seed7 | seed7 | +41.31 | +0.026 | -0.25 | 50.0% | 2% | 0% | 1.78 | Plan7採用baseline |
| B | seed7 | seed11 | -4.22 | -0.048 | +0.74 | 25.0% | 3% | 0% | 1.61 | 不採用 |
| C | seed11 | none | -0.08 | -0.003 | +0.00 | 0.0% | 1% | 0% | 0.42 | BC/WM単体は崩壊なし |
| D | seed11 | seed7 | -1.10 | -0.009 | +0.04 | 25.0% | 1% | 0% | 0.80 | 不採用 |
| E | seed11 | seed11 | +62.04 | +0.050 | +0.04 | 50.0% | 2% | 0% | 1.18 | MaxDD条件外で不採用 |
| G | seed7 | guarded seed11 | -23.36 | -0.065 | +0.76 | 25.0% | 2% | 0% | 1.63 | 不採用 |

読み:

```text
C が大崩れしていないので、seed11 WM/BC自体は最低限安全。
D が悪いので、seed7 ACを別WM/BCに乗せても改善しない。
E は強いが MaxDDΔ が +0.04pt。採用条件を満たさない。
B/G は既にPlan8で不採用。
```

## Fire Attribution

train/testログの adapter_detail。

| ID | fire | mean_delta | positive | long_state | fire_pnl | nonfire_pnl | 読み |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 5.2% | +0.0626 | 96.3% | 14.1% | +0.0887 | +0.5700 | 採用baseline |
| B | 5.8% | +0.0633 | 99.0% | 13.6% | +0.0414 | +0.6089 | fireは正でも全体悪化 |
| C | 1.2% | +0.0644 | 98.2% | 4.8% | +0.0147 | +0.6364 | BC単体でも小さくfire |
| D | 2.2% | +0.0670 | 99.0% | 5.9% | +0.0064 | +0.6445 | ACでfireが増えるが寄与が弱い |
| E | 3.2% | +0.0768 | 99.3% | 7.1% | +0.1080 | +0.5546 | fire寄与は強いがDD条件外 |
| G | 2.0% | +0.1241 | 96.1% | 11.6% | -0.0156 | +0.6622 | guardedは悪いfireを作る |

重要:

```text
崩壊はshortではない。
long/overweight fireの質とタイミングの問題。
```

## Fire Timing Probe

probe結果: `documents/20260430_acplan9_fire_timing_probe.md`

注意:

```text
fire timing probe は actor を直接reloadして同一probe上で比較する診断用。
絶対性能は train/test ログではなく、同一probe内の相対比較とfire overlapを見る。
```

主なfire overlap:

| pair | Jaccard | 解釈 |
|---|---:|---|
| A vs B | 0.233 | 同じseed7 WM/BCでもAC seedでfire timingが大きくズレる |
| A vs C | 0.221 | WM/BC seed変更でもfire timingはかなりズレる |
| A vs E | 0.204 | Eは高性能だがPlan7とは別のfire集合 |
| C vs D | 0.463 | seed11 WM/BC上でseed7 ACはBC fireに近いが性能は悪い |
| D vs E | 0.505 | seed11 WM/BC上のAC同士は半分程度重なる |
| E vs G | 0.339 | guarded variantはEともズレ、fire_pnlも悪い |

fire forward:

```text
A: fwd16 +0.00231 / incr16 +0.00032
B: fwd16 +0.00157 / incr16 +0.00024
C: fwd16 +0.00062 / incr16 +0.00009
D: fwd16 +0.00167 / incr16 +0.00002
E: fwd16 +0.00297 / incr16 +0.00015
G: fwd16 -0.00077 / incr16 +0.00010
```

読み:

```text
Eはfire先のforward returnが一番良い。
Gはforward returnが負で、guarded variantが悪いタイミングへ寄った。
Dはforward returnは悪くないが、delta/incrementalが弱く、全体寄与が出ない。
```

## 原因分類

Roadmapの分類に照らす。

```text
Cが悪い:
  false。CはB&H近似で崩壊なし。

Cは良いがD/Eが悪い:
  部分的にtrue。Dは悪い。Eは強いがMaxDD違反。

C/Dは良いがBだけ悪い:
  false。Dも悪い。

全部良い:
  false。
```

結論:

```text
AC seedだけではなく、WM/BC checkpointとAC seedの組み合わせでfire timingが変わる。
Plan7の勝ちは再現できるが、まだ再学習recipeとして安定していない。
```

## 採用判断

採用維持:

```text
AC Plan7 checkpoint
scale = 0.5 fixed
benchmark floor = 1.0
predictive advantage gate
benchmark-gated small overweight adapter
adapter attribution diagnostics
fire timing probe
```

不採用:

```text
B: seed7 WM/BC + seed11 AC
D: seed11 WM/BC + seed7 AC
E: seed11 WM/BC + seed11 AC
G: guarded sizing adapter
```

Eを不採用にする理由:

```text
AlphaEx +62.04 / SharpeDelta +0.050 は強い。
しかし MaxDDDelta +0.04pt で Plan9採用条件の MaxDDΔ <= 0 を外す。
単発fold5だけで採用すると、Plan8のselector grid失敗と同じ過学習リスクがある。
```

## 次にやるべきこと

性能拡張ではなく、checkpoint selectionを先に直す。

候補:

```text
Plan9-B:
  fire_pnl-aware AC checkpoint selector
  条件:
    fire_pnl > 0
    MaxDDΔ <= 0
    long <= 3%
    turnover <= 3.5
    fire forward return > 0

Plan9-C:
  AC stage中に step=250/300/350/400/450 を保存してfire probe
  最終checkpointではなく、guard通過checkpointを選ぶ

Plan9-E:
  all-fold checkpoint generation は別ジョブ
```

まだ禁止:

```text
route head unlock
full actor AC
advantage gate緩和
floor > 1.0
scale grid selector
Q argmax actor update
```

## 最終判断

```text
Plan9-Aのseed variance診断は完了。
BC/WM単体は致命的ではない。
ACのfire timing/checkpoint selectionが主犯。
次はACを広げず、sizing adapter checkpoint selectorを作る。
```

