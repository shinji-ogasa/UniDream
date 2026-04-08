# Optimization Loop: Issue 5 Conservative AC

## 位置づけ
- issue2 で `BC prior` が `short 100%` に collapse することを確認した。
- それでも AC 側の制約を強めれば改善できるかを切り分けるため、heavy run を避けて `AC-only tiny rerun` で比較した。
- 目的は「AC が support 逸脱で悪化させているのか」「conservative update で改善できる余地があるのか」を見ること。

## pre-Web の 3 本
baseline fold 4 の既存 checkpoint から AC だけを tiny rerun した。

1. `support budget`
2. `KL budget`
3. `conservative AC`

### 結果
- `support budget`
  - test alpha: `-93.53 pt/yr`
  - sharpe delta: `-13.477`
  - test distribution: `short 100%`
- `KL budget`
  - test alpha: `-94.09 pt/yr`
  - sharpe delta: `-13.883`
  - test distribution: `short 100%`
- `conservative AC`
  - test alpha: `-84.35 pt/yr`
  - sharpe delta: `-9.524`
  - test distribution: `short 100%`

3 本とも baseline の collapse を崩せなかった。

## post-Web の 2 本
同系統 3 本で改善が薄かったので文献を見直し、近い保守化として 2 本だけ追加した。

4. `TD3+BC 寄り`
5. `IQL 寄り`

### 結果
- `TD3+BC 寄り`
  - test alpha: `-81.32 pt/yr`
  - sharpe delta: `-9.346`
  - test distribution: `short 100%`
- `IQL 寄り`
  - test alpha: `-87.08 pt/yr`
  - sharpe delta: `-11.418`
  - test distribution: `short 100%`

## rawonly + signal_aim の追確認
orderflow 系 raw source を含む軽量 config でも conservative AC を 1 本だけ追確認した。

- config: `configs/medium_l0_ac_conservative_rawonly.yaml`
- audit: `checkpoints/medium_l0_ac_conservative_rawonly_fold4/ac_support_audit/medium_l0_ac_conservative_rawonly_ac_support_audit_summary.csv`

### 結果
- teacher short ratio: `0.440`
- BC short ratio: `1.000`
- AC short ratio: `1.000`
- `bc_to_ac_short_mismatch = 0.0`
- `teacher_to_ac_short_mismatch = 0.4375`

つまり raw source を含めても、conservative AC は `BC collapse` を全く崩していない。

## 判定
- issue5 の判定は `false`
- 現時点の主因は `AC support drift` より前にある
- conservative / TD3+BC-ish / IQL-ish の範囲では、`BC = short 100%` の collapse を修正できない

## 次の含意
- AC family の延命より、`BC collapse を起こさない learner family` を先に探す必要がある
- source family では `orderflow` が最有望なので、次の本命は `orderflow を使いつつ teacher marginal を保てる learner` に移ること
