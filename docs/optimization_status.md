# Optimization Status

## 完了した issue

### issue1 teacher audit by regime

- teacher はほぼ `long 0% / short 50% / flat 50%`
- `min_hold` を振っても行動分布はほぼ不変
- `signal_aim` を次候補に採用

### issue2 BC prior の再現性診断

- `weighted / sequence / residual / balanced` を切ったが、BC はほぼ `short 100%`
- 既存 BC family は teacher marginal を保てない

### issue3 AC support 逸脱診断

- AC の悪化はあるが、主因は BC collapse の引き継ぎ

### issue4 WM regime 補助

- WM は regime を全く持てていないわけではない
- 主因は WM 単体より learner 側

### issue5 conservative AC

- `KL budget / support budget / conservative AC / TD3+BC-ish / IQL-ish`
  はすべて弱かった

### issue6 external source evaluation

- source family は `orderflow > basis`
- source だけでは根本解決しない

### issue7 learner / output collapse

- 既存 1-step CE 系 actor family はほぼ全部 `BC short ≈ 1.0`
- `static dist match / regime-aware dist match / short-mass match` でも直らなかった
- 結論: 既存 output family 自体が主因

### issue8 continuous target head

current best:
- `medium_l0_bc_continuous_regimegate_exec`
  - `bc_short 0.969`
  - `gap 0.0576`

issue8 で切ったこと:
- inference-only regime gate family 7 本
  - collapse は減るが flat に過補正
- learner-loss 3 本
  - `regime-dist` は short collapse 維持
  - `short-mass / dist-combo` は flat collapse
- post-Web 2 本 (`target_from_logits`)
  - どちらも flat collapse

結論:
- issue8 は current best を保持したまま打ち切り
- `continuous target head + regime gate + execution_aux` は改善の兆候はある
- ただし collapse 自体はまだ強い

## 現在の主結論

1. teacher は弱い
2. その次に BC prior が teacher marginal を保てない
3. AC と WM は主因ではあるが、learner collapse より優先度は下がる
4. source family は `orderflow` が最有望
5. 現時点の最良 learner family は
   `continuous target head + regime gate + execution_aux`

## 次の本命

- `sequence / multimodal policy family`
