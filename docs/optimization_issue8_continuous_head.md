# Optimization Loop: Issue 8 Continuous Target Head

## 問題設定
- issue7 で既存の 1-step CE 系 actor family は打ち切り
- 次は **teacher marginal を保持しやすい別 learner family** が必要
- 最小変更で試せる `residual_controller=true` の continuous target head を先に当てる

## 候補
1. `medium_l0_bc_continuous`
   - feature_stress teacher
   - continuous target head
2. `medium_l0_bc_continuous_rawonly`
   - signal_aim teacher
   - raw-only orderflow
   - continuous target head
3. `medium_l1_bc_continuous`
   - signal_aim teacher
   - continuous target head
   - BC 8 epochs
4. `medium_l0_bc_continuous_regimegate`
   - feature_stress teacher
   - continuous target head
   - regime ごとに overlay 下限を変える
5. `medium_l0_bc_continuous_regimegate_direct`
   - current best checkpoint を使った direct target track
   - `infer_direct_track_scale=1.0`
6. `medium_l0_bc_continuous_regimegate_direct_half`
   - current best checkpoint を使った direct target track
   - `infer_direct_track_scale=0.5`

## 結果

### `medium_l0_bc_continuous`
- teacher short: `0.163`
- BC short: `0.992`
- teacher_to_bc_mean_abs_gap: `0.0607`
- trade_prob_mean: `0.131`
- baseline target mass: `0.953`
- target mean overlay: `-0.0549`

解釈:
- `bc_short_ratio` はまだ高い
- ただし issue7 系より `teacher_to_bc_mean_abs_gap` は大きく改善
- 既存 family の中では初めて「collapse が少し緩む」方向

### `medium_l0_bc_continuous_rawonly`
- teacher short: `0.440`
- BC short: `0.999`
- teacher_to_bc_mean_abs_gap: `0.1491`

解釈:
- raw-only + signal_aim では collapse 改善は限定的

### `medium_l1_bc_continuous`
- teacher short: `0.353`
- BC short: `0.994`
- teacher_to_bc_mean_abs_gap: `0.1395`
- trade_prob_mean: `0.0287`

解釈:
- L1 に上げると改善が維持されない
- 現時点では `signal_aim + continuous head` だけでは弱い

### `medium_l0_bc_continuous_regimegate`
- teacher short: `0.163`
- BC short: `0.989`
- teacher_to_bc_mean_abs_gap: `0.0595`
- trade_prob_mean: `0.116`

解釈:
- `medium_l0_bc_continuous` よりわずかに良い
- Regime 1/2 の short 固定が少し緩む
- issue8 の current best

### `medium_l0_bc_continuous_regimegate_direct`
- teacher short: `0.163`
- BC short: `1.000`
- teacher_to_bc_mean_abs_gap: `0.0596`

解釈:
- direct target track を全量で入れると `bc_short_ratio` が再び `1.0` に戻る
- `gap` も current best を超えない
- branch は棄却

### `medium_l0_bc_continuous_regimegate_direct_half`
- teacher short: `0.163`
- BC short: `0.999`
- teacher_to_bc_mean_abs_gap: `0.0596`

解釈:
- 半量に落としても current best を超えない
- direct target track 単独では改善が弱い
- branch は棄却

## 暫定結論
- continuous target head は **完全な解決ではない**
- ただし issue7 系よりは有望
- 今の best は `continuous target head + regime gate`
- direct target track は全量/半量とも current best を超えなかった

## 次の打ち手
- continuous head を維持したまま
  - `execution head 分離`
  - `signal_aim` 条件での regime gate 再評価
  のどれかを足す
- 反対に、旧 1-step CE head の延命には戻らない
