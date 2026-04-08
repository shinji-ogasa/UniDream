# Optimization Loop: Issue 7 Learner / Output Collapse

## 問題設定
- issue2 で `teacher -> BC` の再現失敗が確認済み
- issue5 で conservative AC を入れても collapse は直らなかった
- 次に見るべきは、`BC actor の出力 head 自体が short action に潰れているのか` どうか

## 監査項目
- `trade_prob_mean`
- `trade_prob_high_ratio`
- `target_entropy_mean`
- `short_target_mass_mean`
- `baseline_target_mass_mean`
- `target_mean_overlay_mean`
- `target_mean_overlay_std`
- `teacher_to_bc_mean_abs_gap`

## ベースライン監査結果

### `medium_l0_bc_sequence`
- teacher short: `0.375`
- BC short: `0.999`
- trade prob mean: `0.146`
- short target mass: `0.202`
- baseline target mass: `0.798`
- target mean overlay: `-0.135`

解釈:
- head の分布は baseline mass を多く残しているが、平均 overlay が常に負側で、実行を通すとほぼ `short 100%` に張り付く

### `medium_l0_bc_weighted`
- teacher short: `0.375`
- BC short: `0.999`
- target entropy は少し上がるが、平均 overlay がさらに負側に寄る

解釈:
- sample weighting だけでは collapse を崩せない

### `medium_l0_ac_conservative_rawonly`
- teacher short: `0.440`
- BC short: `0.999`
- trade prob mean: `0.792`
- short target mass: `0.628`
- baseline target mass: `0.372`
- target mean overlay: `-0.603`

解釈:
- raw-only + signal_aim + orderflow 系 source でも、head 出力そのものが short 側へ明確に崩れる

## pre-Web 同系統 3 本

### 1. `target_dist_match`
- `medium_l0_bc_sequence_distmatch`
- `medium_l0_bc_weighted_distmatch`
- `medium_l0_bc_rawonly_distmatch`

結果:
- 3 本とも `bc_short_ratio ≈ 1.0`
- `teacher_to_bc_mean_abs_gap` も十分には縮まらない

判定:
- 静的な target distribution 一致だけでは collapse を直せない

## Web 確認後の 2 本
文献確認の上で、静的 marginal 一致より
- regime 条件付きの分布一致
- short mass そのものの直接制御
を優先した

### 4. `regime-aware distribution match`
- config: `configs/medium_l0_bc_sequence_regimedist.yaml`
- val audit:
  - teacher short: `0.163`
  - BC short: `0.999`
  - short target mass: `0.188`
  - baseline target mass: `0.812`
  - target mean overlay: `-0.123`

判定:
- regime 条件を入れても collapse は不変

### 5. `short-mass match`
- config: `configs/medium_l0_bc_rawonly_shortmass.yaml`
- val audit:
  - teacher short: `0.440`
  - BC short: `0.999`
  - short target mass: `0.371`
  - baseline target mass: `0.629`
  - target mean overlay: `-0.234`

判定:
- short mass を直接合わせても、出力実行後の collapse は崩れない

## 結論
- issue7 の主因仮説は `true`
- ただし、同じ actor / head family のままでは
  - weighted BC
  - residual BC
  - sequence BC
  - static dist match
  - regime-aware dist match
  - short-mass match
  のどれでも collapse を直せなかった

つまり、今の本質は
- `teacher が弱い` だけではなく
- **既存の BC actor / output family 自体が、teacher marginal を保持できない**
こと

## 次の本命
- 既存 1-step CE 系 head の延命は打ち切る
- 次は
  - `direct target tracking`
  - `continuous target head`
  - `regime gate + execution head`
  - `sequence-style actor`
  のような **別 learner family** に進む
