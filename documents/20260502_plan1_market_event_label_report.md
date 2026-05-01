# Plan 1 Market Event Label Verification Report

Date: 2026-05-02

## Summary

`documents/plan_1.md` の方針に従い、BC/ACへ進む前に market-state event label が複数foldで分離できるかを検証した。

結論: 現時点では **BC/ACへ進めない**。

理由は、teacher inventory や current position のショートカットを使わずに作った market event label が、fold4/5/6で安定して読めていないため。特に fold5/fold6 が AUC 0.50付近まで落ち、false-active も許容値を超える。

この状態でBC/ACを回すと、MaxDD改善ではなく fold4 だけに合ったde-riskや、過去に何度も出た片側collapseに戻る可能性が高い。

## What Was Implemented

追加CLI:

```powershell
uv run python -m unidream.cli.market_event_label_probe
```

追加ファイル:

```text
unidream/cli/market_event_label_probe.py
```

このprobeは以下を行う。

```text
1. WFO foldごとに既存WM checkpointをロード
2. raw features / WM latent / predictive state / regime context を構築
3. current position を主 shortcut にしない market event label を生成
4. label separability を logistic / HistGradientBoosting で評価
5. validationで false-active / predicted-active cap を満たす閾値を選択
6. test foldで AUC, AP, recall, false-active を集計
```

## Labels Tested

### risk_off

将来window内で benchmark exposure のMaxDDが大きく、0.0/0.5 exposureへ落とすことで cost 後もDD削減余地があるイベント。

### recovery

benchmark equity が underwater 状態で、将来returnが回復し、future DDが大きく悪化しないイベント。

### overweight

1.25x exposure が benchmark 1.0x より cost後に優位で、DD worsen が小さいイベント。

### active

`risk_off OR recovery OR overweight`。

## Pass Criteria Used

plan_1.md の基準に合わせ、最低限以下を見た。

```text
active worst AUC >= 0.65
false-active worst <= 0.15
recall worst roughly 0.25-0.35
risk_off AUC >= 0.70 target
overweight AUC >= 0.65 target
fold4/5/6で同方向
```

## Experiments

### 1. Base h32 logistic

Output:

```text
documents/20260502_market_event_labels_base_f456.md
documents/20260502_market_event_labels_base_f456.json
```

Best aggregate per target:

| target | best feature | density | AUC mean | AUC worst | false-active worst | recall worst |
|---|---|---:|---:|---:|---:|---:|
| active | context_no_position | 0.995 | 0.755 | 0.675 | 0.056 | 0.252 |
| risk_off | context_no_position | 0.958 | 0.679 | 0.631 | 0.157 | 0.245 |
| recovery | wm | 0.250 | 0.569 | 0.567 | 0.177 | 0.116 |
| overweight | raw | 0.095 | 0.621 | 0.563 | 0.135 | 0.093 |

判定: invalid。

active density 99.5%、risk_off density 95.8% で、ほぼ常時active扱いになっている。AUCだけ見れば良く見えるが、labelが広すぎてルーティング教師として使えない。

### 2. Tightened v2 h32 logistic

Output:

```text
documents/20260502_market_event_labels_v2_f456.md
documents/20260502_market_event_labels_v2_f456.json
```

Best aggregate per target:

| target | best feature | density | AUC mean | AUC worst | false-active worst | recall worst |
|---|---|---:|---:|---:|---:|---:|
| active | raw | 0.410 | 0.573 | 0.532 | 0.354 | 0.198 |
| risk_off | raw | 0.201 | 0.584 | 0.537 | 0.224 | 0.187 |
| recovery | wm | 0.136 | 0.566 | 0.563 | 0.230 | 0.145 |
| overweight | raw_wm_context_no_position | 0.203 | 0.528 | 0.496 | 0.325 | 0.044 |

判定: fail。

label density は現実的になったが、fold worst AUC が低すぎる。false-active も cap 0.15 を大きく超えた。

### 3. h64 logistic

Output:

```text
documents/20260502_market_event_labels_h64_f456.md
documents/20260502_market_event_labels_h64_f456.json
```

Best aggregate per target:

| target | best feature | density | AUC mean | AUC worst | false-active worst | recall worst |
|---|---|---:|---:|---:|---:|---:|
| active | raw | 0.451 | 0.577 | 0.527 | 0.324 | 0.164 |
| risk_off | raw | 0.251 | 0.561 | 0.512 | 0.215 | 0.135 |
| recovery | raw_wm | 0.132 | 0.562 | 0.552 | 0.213 | 0.150 |
| overweight | context_no_position | 0.200 | 0.513 | 0.510 | 0.242 | 0.124 |

判定: fail。

horizonを64へ伸ばしても、fold worst は改善しなかった。risk_off/overweight は実質ランダムに近い。

### 4. Tightened v2 h32 HistGradientBoosting

Output:

```text
documents/20260502_market_event_labels_v2_hgb_f456.md
documents/20260502_market_event_labels_v2_hgb_f456.json
```

Best aggregate per target:

| target | best feature | density | AUC mean | AUC worst | false-active worst | recall worst |
|---|---|---:|---:|---:|---:|---:|
| active | context_no_position | 0.410 | 0.566 | 0.508 | 0.217 | 0.154 |
| risk_off | wm | 0.201 | 0.582 | 0.531 | 0.224 | 0.242 |
| recovery | wm | 0.136 | 0.547 | 0.526 | 0.270 | 0.112 |
| overweight | context_no_position | 0.203 | 0.510 | 0.500 | 0.189 | 0.135 |

判定: fail。

非線形probeでも改善しない。線形probeの表現力不足ではなく、現状の特徴/WM latent/predictive stateからこのevent labelが安定して読めていない可能性が高い。

## Fold-Level Observation

fold4は一部読める。

```text
fold4 raw_wm_context_no_position risk_off AUC: 0.680
fold4 raw_wm active AUC: 0.670
```

しかし fold5/fold6 で落ちる。

```text
fold5 risk_off AUC: around 0.505-0.546
fold6 risk_off AUC: around 0.512-0.531
fold5 active AUC: around 0.496-0.526
fold6 active AUC: around 0.508-0.527
```

つまり「fold4には合うが複数fold再現性がない」状態。

## Decision

BC/AC最適化へは進めない。

理由:

```text
1. active/risk_off label の fold worst AUC が基準未達
2. false-active が高く、DD区間以外でもde-riskを発火しやすい
3. overweight label はAUC 0.50前後で読めていない
4. HGBでも改善せず、単純な分類器不足ではない
5. このまま学習すると、MaxDD改善ではなくfold固有の片側解に戻る
```

## Current Best Interpretation

現在の問題はAC制御ではなく、BC/ACへ渡す前段の market event teacher がまだ弱い。

特に必要なのは以下。

```text
future return が高いか
future DDが大きいか
candidate action がDDを悪化/改善するか
そのeventがfoldを跨いで読めるか
```

このうち、最後の「foldを跨いで読めるか」が未達。

## Recommended Next Direction

次にやるなら、単純な閾値調整ではなく label 設計を変えるべき。

優先度順:

1. maxDD window overlap label を作る
2. post-fire drawdown contribution label を作る
3. pre-DD state label を作る
4. event label を直接 raw return future から作るのではなく、WMにDD/control-risk headを追加して予測対象として学習させる
5. そのDD/control-risk headの予測値が fold4/5/6 でAUC 0.65以上を出してからBCへ戻る

現時点で避けるべきこと:

```text
risk_off threshold の細かい探索
AC制限解除
route head unlock
fold4だけで良い設定の採用
```

## Final Status

Plan 1 の Phase 1/2 相当は実装・検証済み。

結果は fail。複数foldで安定した MaxDD改善 alpha+ を狙える label separability には到達していない。

そのため、ここでいったん停止する。
