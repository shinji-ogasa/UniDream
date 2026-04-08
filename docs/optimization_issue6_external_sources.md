# Optimization Loop: Issue 6 External Source Evaluation

## 目的
- learning principle 側の issue1〜5 を一段切った上で、外部ソースがどこまで上積みを作るかを確認する
- 順番は `basis -> orderflow -> onchain -> hybrid`

## 既存 summary
- source family suite の既存 summary:
  - `basis`
    - `m2_pass_count = 0`
    - `alpha mean = +1.0 pt`
    - `sharpe delta mean = +0.05`
    - `win rate mean = 51%`
  - `orderflow`
    - `m2_pass_count = 1`
    - `alpha mean = +4.5 pt`
    - `sharpe delta mean = +0.18`
    - `win rate mean = 58%`

## 判定
- 現時点で source family の優先順位は `orderflow > basis`
- まだ決定打ではないが、issue1〜5 が弱かったことを踏まえると、次に戻るなら source family は orderflow を優先する

## 実行入口
- suite:
  - `scripts/run_issue6_external_source_loop.ps1`
- best family selection:
  - `scripts/select_best_source_family.ps1`

## 現状の結論
- learning principle 側だけでは `short 100%` collapse を崩せなかった
- そのため、次に外部ソースを再評価するなら `orderflow` を最優先にする
- onchain / hybrid はその後
