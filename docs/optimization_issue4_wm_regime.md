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
  - current regime accuracy: `0.718`
  - current regime balanced accuracy: `0.696`
  - next regime accuracy: `0.723`
  - next regime balanced accuracy: `0.701`

## 判定
- baseline の WM latent は regime を全く持てていないわけではない
- 少なくとも issue2 の `teacher -> BC short 100% collapse` を直接説明するほどの表現欠損には見えない
- issue4 は `主因では薄い` と一旦判定する

## 次の分岐
- `issue5`: AC 側の conservative family を tiny で比較する
- `issue6`: 必要なら source family 側へ戻る
