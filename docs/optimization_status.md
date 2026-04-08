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
- ただし source だけでは根本解決しない

### issue7 learner / output collapse

- 既存 1-step CE 系 actor family はほぼ全部 `BC short ≈ 1.0`
- `static dist match / regime-aware dist match / short-mass match` でも直らなかった
- 結論: 既存 output family 自体が主因

## 進行中の issue

### issue8 continuous target head

- `medium_l0_bc_continuous`
  - `gap 0.0607`
- `medium_l0_bc_continuous_regimegate`
  - `bc_short 0.989`
  - `gap 0.0595`
- `medium_l0_bc_continuous_regimegate_exec`
  - `bc_short 0.969`
  - `gap 0.0576`
  - current best

棄却済み:
- `direct target track`
- `path_aux`
- `signal_aim + raw-only orderflow`
- baseline source の `signal_aim + regime gate + execution_aux`
- `controller_state_dim=3`
- code-level split execution head

inference-only regime gate family:
- 5 本とも `flat 100%` 近辺へ過補正
- best gap は `0.0289` まで縮んだが、current best は更新できず

## 現在の主結論

1. `teacher` は弱い
2. その次に `BC prior` が teacher marginal を保てない
3. `AC` と `WM` は主因ではあるが、learner collapse より優先度は下がる
4. source family は `orderflow` が最有望
5. 現時点の最良 learner family は
   `continuous target head + regime gate + execution_aux`

## 次の本命

- `execution_aux` を維持した learner-loss branch
