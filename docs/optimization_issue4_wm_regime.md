# Optimization Loop: Issue 4 WM Regime Representation

## 目的
- WM latent が regime を十分に分離できているかを局所診断する
- `BC/AC が弱い` 前に、WM 表現自体が regime 情報を持てていない可能性を切る

## 軽量診断
- 実行: `audit_wm_regime.py`
- 軽量化:
  - `--splits val`
  - `--max-bars 4096`

## baseline 結果
- config: `medium_v2`
- checkpoint: `checkpoints/fold_4`
- val 4096 bars の linear probe:
  - current regime accuracy: `0.300`
  - current regime balanced accuracy: `0.353`
  - next regime accuracy: `0.331`
  - next regime balanced accuracy: `0.340`

## current signal_aim family
- config: `medium_l1_bc_continuous_exec_shortmass_align`
- checkpoint: `checkpoints/medium_l1_bc_continuous_exec_shortmass_align/fold_4/world_model.pt`
- val 4096 bars の probe:
  - current regime accuracy: `0.271`
  - current regime balanced accuracy: `0.331`
  - next regime accuracy: `0.307`
  - next regime balanced accuracy: `0.364`

## 判定
- baseline でも current family でも regime probe は弱い
- balanced accuracy が `0.33 - 0.36` では、regime transition を十分に保持しているとは言いにくい
- issue4 は `主因候補に戻す`

## 候補
- `idm + return` を強める軽量 proxy
- `capacity` を少しだけ増やす軽量 branch
- heavy 禁止なので、まずは `stop-after wm` の L0 で切る

## L0 first pass

### `medium_l0_wm_idmreturn`

- val current regime balanced accuracy: `0.366`
- val next regime balanced accuracy: `0.325`

判定:
- current regime は baseline/current family より少し改善
- ただし next regime は悪化
- mixed

### `medium_l0_wm_capacity`

- val current regime balanced accuracy: `0.338`
- val next regime balanced accuracy: `0.337`

判定:
- baseline/current family と大差なし
- reject

### `medium_l0_wm_idmreturn_capacity`

- val current regime balanced accuracy: `0.342`
- val next regime balanced accuracy: `0.371`

判定:
- next regime は 3 本の中で最良
- ただし current regime は baseline を超えない
- mixed

## ここまでの結論
- 3 本とも一方向の明確改善は出ていない
- 軽量 proxy だけでは issue4 を閉じられない
- 次は Web で絞った
  - `CPC / contrastive predictive coding`
  - `explicit regime auxiliary head`
  のような表現学習寄りの枝へ進む

## 次の分岐
- `issue4`: WM の軽量 branch を L0 で比較する
- その後に `issue5`: AC 側の conservative family を tiny で比較する
