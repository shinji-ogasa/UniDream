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
7. `medium_l0_bc_continuous_regimegate_exec`
   - current best に `execution_aux` を追加
   - target / trade / band より executed inventory を強く合わせる
8. `medium_l0_bc_continuous_regimegate_path`
   - `execution_aux + path_aux`
   - 4-step path を使って execution を直接合わせる
9. `medium_l0_bc_continuous_regimegate_pathshort`
   - `execution_aux + path_aux(shortfall重視)`
10. `medium_l0_bc_continuous_rawonly_regimegate_exec`
   - `signal_aim + raw-only orderflow` に execution branch を追加

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

### `medium_l0_bc_continuous_regimegate_exec`
- teacher short: `0.163`
- BC short: `0.969`
- teacher_to_bc_mean_abs_gap: `0.0576`
- trade_prob_mean: `0.138`

解釈:
- `continuous + regime gate` に `execution_aux` を足すと初めて `bc_short_ratio` が `0.97` 台まで落ちた
- `gap` も current best を更新
- issue8 の current best を更新

### `medium_l0_bc_continuous_regimegate_path`
- teacher short: `0.163`
- BC short: `0.000`
- BC flat: `1.000`
- teacher_to_bc_mean_abs_gap: `0.0568`

解釈:
- `path_aux` を足すと short collapse は止まるが、今度は flat 100% に反転する
- `gap` は少し良いが、別方向の collapse なので棄却

### `medium_l0_bc_continuous_regimegate_pathshort`
- teacher short: `0.163`
- BC short: `0.000`
- BC flat: `1.000`
- teacher_to_bc_mean_abs_gap: `0.0568`

解釈:
- shortfall 重視にしても `path` branch の挙動は変わらない
- branch は棄却

### `medium_l0_bc_continuous_rawonly_regimegate_exec`
- teacher short: `0.440`
- BC short: `0.999`
- teacher_to_bc_mean_abs_gap: `0.1492`

解釈:
- `signal_aim + raw-only orderflow` では execution branch を足しても弱い
- issue8 の本命は依然として baseline teacher 側

## 暫定結論
- continuous target head は **完全な解決ではない**
- ただし issue7 系よりは有望
- 今の best は `continuous target head + regime gate + execution_aux`
- direct target track は全量/半量とも current best を超えなかった
- `path_aux` は flat 100% へ過補正した
- `signal_aim + raw-only orderflow` は execution branch を足しても弱い

## 次の打ち手
- continuous head を維持したまま
  - code-level の `execution head 分離`
  - `signal_aim` 条件での regime gate 再評価
  のどれかを足す
- 反対に、旧 1-step CE head の延命には戻らない
