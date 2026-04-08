# Optimization Loop: Issue 5 Conservative AC

## 目的
- `KL budget / support budget / conservative AC` で、BC 崩壊後の AC を安全化できるかを小さく確認する
- heavy run は避け、`wm + bc checkpoint` から `AC-only` で 250 step だけ回す

## pre-Web の 3 本
baseline fold 4 (`checkpoints/fold_4`) を土台にした tiny AC-only rerun。

1. `support budget`
2. `KL budget`
3. `conservative AC`

### 結果
- `support budget`
  - test alpha: `-93.53 pt/yr`
  - sharpe delta: `-13.477`
  - test dist: `short 100%`
- `KL budget`
  - test alpha: `-94.09 pt/yr`
  - sharpe delta: `-13.883`
  - test dist: `short 100%`
- `conservative AC`
  - test alpha: `-84.35 pt/yr`
  - sharpe delta: `-9.524`
  - test dist: `short 100%`

3 本とも baseline より悪化し、`short 100%` も崩れなかった。

## post-Web の 2 本
文献寄りの方向として、今の実装で近似できる範囲の 2 本を試した。

4. `TD3+BC 寄り`
5. `IQL 寄り`

### 結果
- `TD3+BC 寄り`
  - test alpha: `-81.32 pt/yr`
  - sharpe delta: `-9.346`
  - test dist: `short 100%`
- `IQL 寄り`
  - test alpha: `-87.08 pt/yr`
  - sharpe delta: `-11.418`
  - test dist: `short 100%`

## 結論
- issue5 の主因判定は `false` 寄り
- AC family を conservative に振っても、BC collapse を直せなかった
- `BC = short 100%` のままなので、AC 改善の前に BC/source 側を戻す必要がある

## 次の分岐
- `issue6`: source family 側の既存 summary を使って orderflow 優位を再確認する
- その後は `signal_aim` / source family を前提に BC collapse を起こさない別 learner family を再検討する
