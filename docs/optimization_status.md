# Optimization Status

## 現在の状況
- issue1 `teacher audit by regime`: 完了
  - teacher は `long 0% / short 50% / flat 50%` に近く、主因寄り
  - 次候補として `signal_aim` を採用

- issue2 `BC prior の再現性診断`: 完了
  - `weighted / sequence / residual / balanced` を試しても BC は `short 100%` に collapse
  - BC family の表現力では teacher marginal を保てない

- issue3 `AC の support 逸脱診断`: 完了
  - AC は BC collapse をそのまま引き継ぐ
  - issue2 より後ろではない

- issue4 `WM の regime 表現`: 完了
  - 現状の WM latent は regime 識別をある程度持っている
  - `WM が全く regime を持てていない` は主因薄め

- issue5 `conservative AC / IQL / TD3+BC-ish`: 完了
  - `KL budget / support budget / conservative AC` を入れても `short 100%` は崩れない
  - AC family の改善余地だけでは足りない

- issue6 `external source evaluation`: 完了
  - 既存 summary では `orderflow > basis`
  - ただし source の改善だけで issue1〜5 を飛ばせるほどではない

- issue7 `learner / output collapse`: 完了
  - actor head の監査で、`teacher short 0.16〜0.44` に対して `BC short ≈ 1.0`
  - `static dist match`, `regime-aware dist match`, `short-mass match` でも collapse は不変
  - 既存 1-step CE 系 actor family は打ち切り

- issue8 `continuous target head`: 進行中
  - `medium_l0_bc_continuous` では `teacher_to_bc_mean_abs_gap` が `0.0607` まで改善
  - ただし `bc_short_ratio` はまだ `0.992`
  - `signal_aim` と raw-only/orderflow を組み合わせると改善は弱まる

## 次の主課題
1. 主因は `BC prior が teacher marginal を保てないこと`
2. AC や WM より先に learner / output family 側の collapse が強く出ている
3. source family では `orderflow` が最有望
4. 直近の有望枝は `continuous target head`

## 次の本命
- `orderflow` を使ったまま `BC collapse` を起こさない learner family を探す
- 候補優先度は
  - 1-step CE imitation を離れる
  - teacher marginal を保ちやすい出力設計に変える
  - sequence / continuous target を持つ learner family に寄せる

## 関連ドキュメント
- [issue1 teacher audit](./optimization_issue1_teacher_audit.md)
- [issue2 BC prior](./optimization_issue2_bc_prior.md)
- [issue3 AC support](./optimization_issue3_ac_support.md)
- [issue4 WM regime](./optimization_issue4_wm_regime.md)
- [issue5 conservative AC](./optimization_issue5_conservative_ac.md)
- [issue6 external sources](./optimization_issue6_external_sources.md)
- [issue7 output collapse](./optimization_issue7_output_collapse.md)
- [issue8 continuous head](./optimization_issue8_continuous_head.md)
