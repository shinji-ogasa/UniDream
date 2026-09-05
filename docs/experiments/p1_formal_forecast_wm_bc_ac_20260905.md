# Registered ForecastActionSource → WM → BC → AC diagnostic

実行日: 2026-09-05（UTC）  
ブランチ: `exp/p1-formal-forecast-wm-bc-ac-20260904`  
コードリビジョン: `321125dde19677871a2b0e6b8b59a440d914f7f0`  

## 結論

正式に登録された `ForecastActionSource` を認証し、strict conditional OOF
artifact を再ロードした上で、WM→BC→ACを実学習できることは確認できた。
しかし、この固定source・固定契約・固定窓では、学習量を増やしても学生方策の
改善は確認できなかった。

- WM: 700/700 step
- BC: 8/8 epoch、loss `1.97705098 → 1.15794656`
- AC: 300/300 step
- BCの5 rolling窓平均: `alpha_ex_vs_hold = -0.03757496`
- BCの固定outer窓: `-0.21831322`
- ACの5 rolling窓平均/固定outer窓: `0.0`、全窓fill 0、benchmarkからのoverlay 0

したがって、今回の結果は「正式sourceをつないでも自動的に精度が上がる」ことを
支持しない。一方、これは不可能性の証明ではなく、BCのラベル写像とACの目的関数
がsource信号を壊している可能性を切り分ける段階まで進んだ、という結果である。

## Sourceとartifactの監査

- scenario/arm/model: `S3 / zero_injection_control / ridge`
- source support: raw `[104528, 139568)`、`fit_origin=104528`
- forecast rows: 35,040、finite/origin/score rows: 34,718
- ForecastActionSource binding SHA-256: `874beb6da30a3b768e1c9f27a717493e1608eb8a8210ce63dae44ec1e8278b6d`
- forecast file SHA-256: `b1a46e3586c1d7a74c0f41cb41b9117df7d5e1f7e3d8af874e63483245acd803`
- 対応するaction artifact: `authenticated=true`、record 8,759
- conditional OOF adapter: 4-bar grid 8,760、eligible 8,681、prediction 8,616、training-label 8,681
- strict artifact reload: `true`
- adapterの実体: `fixed_registered_forecast_fit`（各originでの再fitではない）

Sourceと対応action artifactの両方を外部digest/metadata/binding付きでロードし、
同一の4-bar commitment/action contractでWM/BC/ACへ渡した。adapterはstrict envelope
の形を満たすための因果的なviewであり、per-origin refitting済みのOOF結果とは主張しない。

## 学習と同一評価窓の比較

リターン単位はaction contractの`additive_log_return`。`alpha_ex_vs_hold`は同じ
mask・delay・fill/cost契約でのB&Hとの差分である。

| 方策 | 5窓平均 alpha | 評価対象のalpha | fill | position |
|---|---:|---:|---:|---|
| 登録source teacher（source support診断） | — | `+0.15534127`（support全体） | 445 | short 53.8% / flat 46.2% |
| BC actor | `-0.03757496` | `-0.21831322` | 295 | short 99.8% |
| AC actor | `0.0` | `0.0` | 0 | flat 100% |

BCの窓別alphaは順に `+0.01088778, +0.02968395, -0.14930976,
+0.04106994, -0.12020673`。source support上のteacherに正の差分があっても、
BCはほぼ常時underweight/shortへ崩れ、2023 outerでB&Hを大きく下回った。
ACは300 step後に全窓でflatへ収束し、売買を全く行わずB&Hと同じ結果になった。

## 解釈と次の実験

1. **データ/認証/strict OOF接続は主因ではない。** source、mask、artifact hash、
   WM/BC/AC境界、同一replay評価は完走した。
2. **BCの教師→position写像が第一候補。** teacherはshort/flatの混在だが、BCは
   short約99%へ偏った。`target_aux_coef=1.0`、`trade_aux_coef=0.5`でlossは下がったため、
   no-opではないが、分布を正しく模倣できていない。
3. **ACの目的関数がBCを壊している。** BC直後は取引していたのに、AC後はflat 100%。
   次はBC actorを固定したno-AC基準、BC-anchor/behavior-KL付きAC、flat-collapseを
   失格にする固定validation診断を分けて実施する。
4. **source信号の一般化も別途必要。** 今回のsourceは登録済み固定fitであり、真の
   per-origin ForecastActionSource再fitではない。改善候補が見えた後にのみ、nested
   WFO・MBB・outer report-only gateへ進む。

## 形式上の扱い

この成果物はreport-only diagnosticであり、preregistered P1 outer resultではない。
manifestの`results_observed=false`、`selection_allowed=false`、`promotion_allowed=false`
は変更していない。注文・exchange接続・live moneyは0件。

machine-readable report（ignored output）:
`codex_outputs/p1_formal_forecast_wm_bc_ac_20260904/formal_forecast_wm_bc_ac_report.json`  
report content SHA-256: `7ac3452836232d4ccfcfdfdbbe00aecd52ca98db65da025c0040232377aecdce`  
conditional artifact SHA-256: `39cdcf62eefe7710827dfb9b502f9406a3ad22d762868aff701b4617df6493d7`

検証: `uv run`相当の`.venv/bin/python -m unittest discover -s tests -v`で354 tests OK、
`py_compile`、`git diff --check` OK。
