# 平均予測の対照実験と、特徴量欠損時の保有継続の診断

過去のscale期間の平均リターンだけを使う対照は、8期間全体の経済成績で学習した平均予測を上回った。しかし開始時bear区分ではAlphaExが負で、ユーザーの条件を満たさない。先物フローを加えた `perp_delay0_scaled_utility_risk1` は、観測できた全開始時区分の平均でAlphaEx正・MaxDDDelta負を保つ一方、平均リターンのMSE・MAEは単純対照より悪い。「最強モデル」や高確率の将来再現性は未確立である。

もう一つ、改善の優先順位を変える実行上の問題が見つかった。2021年の一期間では、計29本の利用不能なSpot足が長期特徴量へ波及し、141回・約35日間の予定判断で特徴量が揃わなかった。既存方策はその間保有を続け、レバレッジを持ったまま急落を受けた。次の実験は、モデル構造より先に、予測不能時の在庫管理を扱う。

## 事前固定した比較

[登録文書](oracle_mean_control_decisions_registration_20260905.md)・設定・実装・テストをcommit `c93b9ff` に保存した後、`oracle_mean_control_decisions_v1` を一度実行した。新しい学習、校正期間の選び直し、候補選択、test/outerの採点は行っていない。

- 7種類の平均予測 × point / utility risk1 = 14候補。過去の133候補名と合わせ147の適応的に探索された名前であり、独立試行数ではない。
- すべて同じ `technical_scaled` の分散予測を使う。定数平均の対照も学習済みHGBの分散を使うため、完全に機械学習を除いた方策ではない。
- `fit_mean` は親実験のpurge済みfit行の平均。`scale_mean` はvalidation開始の6か月前から3か月前までのscale区間の実現リターン平均であり、直近3か月平均ではない。各ラベルの終了はscale境界より厳密に前である。
- 平均補正の式は `mu_scaled(t) = mean(y_scale) + mu_raw(t) - mean(mu_raw_scale)`。scaledとscale_meanの比較で、過去平均に加えるモデル変動成分の価値を調べる。変動成分はvalidation内で平均ゼロとは限らず、期間間の平均移動も含む。
- 親の18か月fit / 3か月scale / 3か月interval / 3か月validationを継承。fold5–12、2021-04-16から2023-04-16までの8期間、2,920予定判断のうち推論2,586行・採点2,574行。6時間予測、6時間おきの判断、次の15分足始値約定。
- cash/units、B&H初期在庫、目標比率[0.5, 1.12]、1回の変更上限0.08、deadband0.01、片道費用0.00055、年率借入費用0.10を維持。baseで決めた同じ目標列を費用・借入2倍でも再生し、stressで再最適化しない。
- raw平均も今回の共通scaled分散を使うため、親実験のraw方策と同一ではない。technical_scaledの両方策のみ、親と目標・成績の一致を必須にした。

原データの歴史的受信・公表時刻は確定していない。継承した共通maskは遅延なしUMを含む比較用の事後的共通可用性であり、実運用の遅延feed規則ではない。開始時区分はbull2 / bear4 / sideways2で、既存の各区分3期間以上という条件は未達のままである。

UMの観測開始は2020-01-01で、18か月という暦上のfit窓が18か月分の有効観測を意味するわけではない。fold0–5のh24 fit / scale / interval行数は順に0/0/34、0/34/212、34/212/231、247/231/321、479/321/233、800/233/279。事前固定した最低512/64/64行を初めて満たすのはfold5だった。初期期間を都合の良い結果で選別したものではない。

## 経済成績

すべて等四半期平均。AlphaExは方策リターンから同期間B&Hリターンを引いた差、MaxDDDeltaは方策MaxDDからB&H MaxDDを引いた差。単位ptはパーセントポイント。通期複利リターンではない。

| Mean | 決定方式 | Base AlphaEx / DD差 (pt) | 2倍費用 AlphaEx / DD差 (pt) |
| --- | --- | ---: | ---: |
| zero | point | -0.000 / +0.000 | -0.000 / +0.000 |
| zero | utility_risk1 | -1.322 / -1.447 | -1.327 / -1.444 |
| fit_mean | point | -0.165 / +0.102 | -0.184 / +0.111 |
| fit_mean | utility_risk1 | -0.857 / +0.000 | -0.859 / +0.000 |
| scale_mean | point | +0.029 / -0.754 | -0.039 / -0.736 |
| scale_mean | utility_risk1 | +3.128 / -6.531 | +3.061 / -6.494 |
| technical_raw | point | -0.927 / +0.626 | -1.411 / +0.727 |
| technical_raw | utility_risk1 | -1.896 / +0.001 | -2.304 / +0.107 |
| technical_scaled | point | -0.423 / -0.627 | -1.004 / -0.487 |
| technical_scaled | utility_risk1 | +0.595 / -4.323 | +0.277 / -4.211 |
| perp_delay0_raw | point | -0.313 / +0.586 | -0.832 / +0.687 |
| perp_delay0_raw | utility_risk1 | +2.026 / +0.085 | +1.561 / +0.181 |
| perp_delay0_scaled | point | -0.124 / -0.445 | -0.756 / -0.308 |
| perp_delay0_scaled | utility_risk1 | +2.131 / -3.130 | +1.723 / -3.001 |

14候補のうちbaseと2倍費用の全体平均で両符号を満たすのは3件、観測された全開始時区分の平均でも満たすのは `perp_delay0_scaled_utility_risk1` の1件。区分数条件を満たすものは0件。zero_pointは目標の有無によるintent coverage以外、全8期間でB&Hと経済的に一致した。zero_utilityはリスク項により配分を下げ得るため、B&Hの複製ではない。B&Hとcommon_robustの親再生との差も0だった。

| Mean（utility risk1） | 開始時区分 | 期間数 | Base AlphaEx / DD差 (pt) | 2倍費用 AlphaEx / DD差 (pt) |
| --- | --- | ---: | ---: | ---: |
| scale_mean | bull | 2 | +4.661 / -7.065 | +4.490 / -6.954 |
| scale_mean | bear | 4 | -0.873 / -3.114 | -0.914 / -3.098 |
| scale_mean | sideways | 2 | +9.597 / -12.833 | +9.583 / -12.827 |
| technical_scaled | bull | 2 | +3.773 / -5.352 | +3.523 / -5.175 |
| technical_scaled | bear | 4 | +1.186 / -2.792 | +0.837 / -2.713 |
| technical_scaled | sideways | 2 | -3.765 / -6.355 | -4.090 / -6.241 |
| perp_delay0_scaled | bull | 2 | +4.063 / -4.749 | +3.777 / -4.548 |
| perp_delay0_scaled | bear | 4 | +0.967 / -2.784 | +0.595 / -2.706 |
| perp_delay0_scaled | sideways | 2 | +2.525 / -2.202 | +1.923 / -2.043 |

区分は**四半期の開始時に既知の90日momentumと7日volatilityから分類したもの**で、四半期全体の実現上昇・下落ではない。scale_meanのsideways AlphaEx +9.597ptも、fold9の+22.714ptとfold12の−3.519ptの平均であり、各期間で安定して勝ったことを意味しない。

| Mean（utility risk1） | 平均売買回数 / 四半期 | 平均回転 / 四半期 |
| --- | ---: | ---: |
| scale_mean | 4.000 | 0.247 |
| technical_scaled | 44.500 | 3.187 |
| perp_delay0_scaled | 57.625 | 4.232 |

scale_meanに比べた先物scaled平均のutilityは、全体AlphaExがbaseで0.997pt、stressで1.339pt低く、DD差はそれぞれ3.402pt、3.494pt悪い。一方、bear区分のAlphaExはbaseで1.840pt、stressで1.510pt高い。変動成分を全面的に捨てる根拠にはならない。

先物scaledのscale_mean比の追加費用は、初期NAV換算で手数料0.222pt・借入0.080pt。費用に加え在庫経路が変わるので、最終AlphaEx差全体を手数料だけで説明してはいけない。共通分散のため、ここでの差はリスク予測精度の改善ではない。

## 予測誤差

| Mean | MSE改善率 vs zero (%) | MSE改善率 vs fit mean (%) | MAE (bps) | 平均rank IC |
| --- | ---: | ---: | ---: | ---: |
| zero | +0.000 | +0.235 | 117.736 | 定数のため未定義 |
| fit_mean | -0.236 | +0.000 | 117.909 | 定数のため未定義 |
| scale_mean | -0.695 | -0.458 | 118.554 | 定数のため未定義 |
| technical_raw | -4.338 | -4.093 | 121.342 | 0.0374 |
| technical_scaled | -7.035 | -6.783 | 123.970 | 0.0374 |
| perp_delay0_raw | -3.863 | -3.618 | 121.475 | 0.0401 |
| perp_delay0_scaled | -6.176 | -5.926 | 123.284 | 0.0401 |

MSE改善率は `1 - 等四半期平均loss(candidate) / 等四半期平均loss(reference)`。四半期ごとの改善率の平均ではない。MAEは6時間リターンの絶対誤差である。

scale_meanに対してtechnical_scaledはMSEが6.296%、MAEが4.569%悪化し、両指標とも改善期間は0/8。perp_delay0_scaledはMSEが5.443%悪化（改善2/8）、MAEが3.990%悪化（改善0/8）した。弱い正のrank ICだけで、予測の振幅・平均のずれや経済コストを吸収できているとはいえない。全21組の比較、等四半期と行数加重の両集計、全56予測スコアを保存した。

## 約35日間の特徴量欠損と在庫経路

以下は保存済み予測・目標・traceと入力データの事後診断であり、新しいfallback方策の成績ではない。

fold5のscale_mean / technical_scaled / perp_delay0_scaledのutilityは、目標列が完全に同じだった。2021-04-16 18:00 UTCに目標1.08、4月17日00:00に1.12とし、それぞれ次足始値で約定。その後の有効221判断は保有継続だった。有効223判断では3種類の平均予測がすべて正だった。成績一致は同じ意思決定経路を示すが、異なる予測が同じ精度という意味ではない。

2021-04-20 06:00から5月25日06:00 UTCまで、6時間おきの141判断が不適格だった。trace上の内訳は予測利用不能140・現在始値欠損1。5月25日12:00に全特徴が再び揃う。

| 原因となった利用不能なSpot足（始値時刻UTC） | 本数 |
| --- | ---: |
| 2021-04-20 02:00〜04:15 | 10 |
| 2021-04-25 04:00〜08:30 | 19 |

合計29本・7時間15分相当であり、35日分の原価格が欠けていたわけではない。4月は2,880予定行に対し2,851採用行。4月25日04:00の1本はclose時刻が04:00:58.146という不完全足として隔離されている。アーカイブの記録だけでは、取引所停止の原因や当時の受信時刻まで確定できない。

7日窓は669/672、30日窓は2,866/2,880観測を要求する。7日vol等は4月20日06:00〜5月2日06:00の49予定判断、30日vol・drawdownは4月25日06:00〜5月25日00:00の120判断で不完全になる。5月25日06:00には30日前のmomentum参照値がまだ欠ける。価格欠損の隣接リターンも欠けるため、4月28日12:00の30日有限リターンは2,849、5月20日12:00でも2,860で、必要2,866に届かない。

必要なtechnical29だけでもこの141判断すべてが不完全だった。未使用の共通mask列を外しても、fold5の有効223行は回復しない。追加遅延0/1/4のUM flow24/96自体はこの欠損区間で有限だった。

この期間は開始時bull分類だが、実現した四半期のB&Hリターンは−47.766%、3方策は−53.942%、AlphaEx −6.176pt・MaxDDDelta +6.505ptだった。負のcashで保有して価格が下がると、実保有比率は目標上限から受動的に離れる。再開した5月25日12:00の既知始値で1.22348、最終有効判断で1.27840だった。これらは判断時点での観測値であり、全期間の最大値ではない。[0.5, 1.12]は発注目標の範囲で、実保有比率に対する強制清算上限ではなかった。

特徴量欠損中の保有に加え、予測が再開してもutilityが保有を選んだことの両方が含まれる。損失全体を欠損だけに帰属させることはできない。

## 検算・保存

- `uv run python -m unittest discover -s tests -v`: **499 tests OK、56.852秒**。source登録前に完了。ログ `/tmp/oracle-mean-controls-full-tests.log`。
- 独立会計監査: 1,086 hash binding・765 distinct files、全56スコアと21比較を確認。fold5/8/12の84会計経路、21 utility経路・6,398判断を独立scalar再生。AlphaEx最大差1.33e−15、DD差5.56e−16、売買回数差0。
- 全56予測の平均・共通分散・時刻・actual・maskを別途照合。MSE最大差1.09e−19、MAE3.47e−18。scale平均を原価格から再計算した差は4.34e−19。4,912 calibrationラベルの満期を確認。
- technical_scaledの16目標・8 traceとbase/stress成績は親と一致。定数meanにfuture validation actualやinterval区間actualは入っていない。
- 128経済行/目標、56予測ファイル、56 utility traceを保存。学習済み親のretrainingを独立に繰り返した検査ではない。

[登録JSON](oracle_mean_control_evidence_20260905/registration.json)、[全結果](oracle_mean_control_evidence_20260905/results.json)、[独立会計監査](oracle_mean_control_evidence_20260905/independent_account_audit.json)、[独立予測・欠損監査](oracle_mean_control_evidence_20260905/independent_forecast_audit.json)、[52ファイルに紐づく可用性・平均変動の診断](oracle_mean_control_evidence_20260905/availability_diagnostic.json)、[SHA manifest](oracle_mean_control_evidence_20260905/manifest.json)を追跡する。

登録JSONのファイルSHAは `4c2a0cdd0a68717d6bec9016b5cb4d4436eab395f2515bd51ca76b0544867ac6`、結果ファイルSHAは `0ed7a02b02c36e2e7ecd4d2b0566ba61d3a692ba4bfe5eb26b05b591ef3dd2d3`。JSON内部のcanonical payload digestとは区別する。

## 次の改善方向

まず、予定判断で現在始値が既知なら、予測利用不能時にもB&H比率1へ戻すfallbackを、既存の保有継続と比較する小さな実験として事前固定する。既存NaN目標にはutilityが積極的に保有を選んだ場合も含まれるため、目標列を後から埋めるだけではいけない。fallbackで変わったcash・unitsを引き継ぎ、その後の有効予測も自分の在庫状態から再判断する必要がある。現在始値欠損や次足約定不能は従来の実行制約を守る。これは次の仮説であり、改善はまだ観測していない。

その後に、過去平均からのモデル変動成分の固定縮小、または確定済み過去ラベルのみを用いたmean bias更新を検討する。欠損価格のゼロ埋め、coverage条件の緩和、良かった期間の選別、未観測testの選択への使用は改善とみなさない。繰り返し使った8期間だけでは将来の成功確率を証明できず、候補を固定した後の未観測期間・前向き検証が必要である。

一次資料では、[Welch–Goyal (2008)](https://doi.org/10.1093/rfs/hhm014)が単純な過去平均対照に対する多くの株式プレミアム予測の不安定さを示し、[Campbell–Thompson (2008)](https://doi.org/10.1093/rfs/hhm055)は予測誤差と投資家の経済価値を区別する。これらの株式の結果や符号制約はBTCへの証明ではない。[Pesaran–Timmermannの著者公開原稿](https://rady.ucsd.edu/_files/faculty-research/timmermann/estimation-window.pdf)は構造変化下の窓長にbias/varianceのトレードオフがあると論じる。短窓への変更だけで安定化すると仮定せず、次の更新則も成績を見る前に固定する。
