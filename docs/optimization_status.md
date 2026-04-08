# Optimization Status

## 現在の結論

- issue1 `teacher audit by regime`: 完了
  - teacher は `long 0% / short 50% / flat 50%` に近く、主因寄り
  - 次候補として `signal_aim` と `feature_stress` を採用

- issue2 `BC prior の再現性診断`: 完了
  - `weighted / sequence / residual / balanced` を試しても BC は `short 100%` に collapse
  - BC family の小修正では teacher marginal を保てなかった

- issue3 `AC の support 逸脱診断`: 完了
  - AC は BC collapse をそのまま引き継ぎやすい
  - 主因は issue2 より後ろでは薄い

- issue4 `WM の regime 表現`: 監査完了
  - 現状の WM latent は regime 情報をある程度持っている
  - 「WM が全く regime を持てていない」は主因では薄い

- issue5 `conservative AC / IQL / TD3+BC-ish`: 小規模検証完了
  - `KL budget / support budget / conservative AC` を切っても `short 100%` は崩れない
  - AC family の保守化だけでは改善しない

- issue6 `external source evaluation`: 既存 summary 比較完了
  - 現状は `orderflow > basis`
  - ただし source の改善だけで issue1〜5 を覆すほどではない

## 今の主仮説

1. 主因は `BC prior が teacher marginal を保てないこと`
2. AC や WM より前に learner family 側の collapse が起きている
3. source family では `orderflow` が最有望

## 次の本命

- `orderflow` を使ったまま `BC collapse` を起こさない learner family を探す
- 具体的には
  - 1-step CE imitation を離れる
  - teacher marginal を直接保つ出力設計に寄せる
  - もしくは sequence / continuous target を別 learner family でやり直す

## 関連ドキュメント

- [issue1 teacher audit](./optimization_issue1_teacher_audit.md)
- [issue2 BC prior](./optimization_issue2_bc_prior.md)
- [issue3 AC support](./optimization_issue3_ac_support.md)
- [issue4 WM regime](./optimization_issue4_wm_regime.md)
- [issue5 conservative AC](./optimization_issue5_conservative_ac.md)
- [issue6 external sources](./optimization_issue6_external_sources.md)
