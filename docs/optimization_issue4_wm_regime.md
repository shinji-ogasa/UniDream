# Optimization Loop: Issue 4 WM Regime Representation

## 課題
- WM の latent が regime を十分に表現できていないかを確認する
- BC/AC が弱い前に、そもそも state 表現が regime transition を持てていない可能性を切る

## 診断
- 実装: `audit_wm_regime.py`
- 評価:
  - `current_regime`
  - `next_regime`
  の accuracy / balanced accuracy を train / val で確認する

## baseline

### `medium_v2`
- val current regime balanced accuracy: `0.353`
- val next regime balanced accuracy: `0.340`

### `medium_l1_bc_continuous_exec_shortmass_align`
- val current regime balanced accuracy: `0.331`
- val next regime balanced accuracy: `0.364`

判定:
- baseline も current family も `0.33 - 0.36` に留まる
- issue4 は主因候補として維持

## L0 first pass

### `medium_l0_wm_idmreturn`
- val current regime balanced accuracy: `0.366`
- val next regime balanced accuracy: `0.325`

判定:
- current は改善
- next は悪化
- mixed

### `medium_l0_wm_capacity`
- val current regime balanced accuracy: `0.338`
- val next regime balanced accuracy: `0.337`

判定:
- baseline と大差なし
- reject

### `medium_l0_wm_idmreturn_capacity`
- val current regime balanced accuracy: `0.342`
- val next regime balanced accuracy: `0.371`

判定:
- next は最良
- current は baseline 未満
- mixed

## post-Web

### `medium_l0_wm_regimeaux`
- val current regime balanced accuracy: `0.343`
- val next regime balanced accuracy: `0.355`

判定:
- `capacity` よりは良い
- ただし baseline current と `idmreturn_capacity` next を超えない
- mixed

### `medium_l0_wm_idmreturn_regimeaux`
- val current regime balanced accuracy: `0.331`
- val next regime balanced accuracy: `0.359`

判定:
- `regimeaux` 単独より current が悪化
- `idmreturn_capacity` より next も弱い
- reject

## 結論
- `issue4` は true のまま
- ただし、軽量 WM branch では current / next の両方を安定して押し上げる枝はまだ出ていない
- 現時点の最良候補は
  - current を見るなら `medium_l0_wm_idmreturn`
  - next を見るなら `medium_l0_wm_idmreturn_capacity`
  だが、どちらも決定打ではない

## 次
- issue4 は `mixed / no winner` で一旦閉じる
- 次は `issue5 conservative AC` を current best learner family 上で確認する
