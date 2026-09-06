# Stage14: 確定済み履歴による移動補正の結果

**今回の変更は不採用。最強モデルの条件は未達です。** 元の8開発四半期で、
Technical/Perpの移動補正は固定reliability補正より平均MSEを0.96026%/1.38890%
改善しました。一方、追加6方策すべての平均AlphaEXは負となり、相場別の
経済条件・予測条件をともに通過できませんでした。既存の方策・選択ロックは変更しません。

この実験はモデル構造を変えず、前回のscale期間で推定済みの重みを固定して、
直近3暦月の確定リターン平均と過去raw予測平均だけを順次更新しました。
移動平均だけの対照も同じ履歴で計算し、テクニカル予測を加えた効果を比較しました。
原validation5–12（原test4–11）、2021年4月〜2023年4月の再利用データです。
test(f)=validation(f+1)という命名の対応に注意してください。

## 成績の要点

以下は8四半期の等重み平均。AlphaEXとMaxDDDeltaはB&Hとの差のpercentage points。
AlphaEXは正、MaxDDDeltaは負が望ましい方向です。2倍コストは同じ注文意図を再会計した値です。
「両条件の四半期数」は事前登録した厳密な符号判定で、両コストを通過した件数です。


| 追加方策 | 欠損処理 | AlphaEX pt | DDdelta pt | 2倍AlphaEX pt | 2倍DDdelta pt | 両条件/8 |
|---|---|---|---|---|---|---|
| 移動平均 | hold | -3.62230 | -6.91087 | -3.68764 | -6.89019 | 5 * |
| 移動平均 | fallback B&H | -1.52046 | -5.85046 | -1.58572 | -5.83070 | 5 * |
| Technical移動補正 | hold | -4.26887 | -6.02764 | -4.40451 | -5.96302 | 3 |
| Technical移動補正 | fallback B&H | -1.58706 | -5.84565 | -1.73157 | -5.78867 | 3 |
| Perp移動補正 | hold | -3.51504 | -5.54874 | -3.65272 | -5.48104 | 5 * |
| Perp移動補正 | fallback B&H | -2.50946 | -5.10756 | -2.64332 | -5.04678 | 4 * |


\* 移動平均とPerpの件数はfold12を含みますが、そのDDdeltaは両コストとも
`−1.1102230246251565e−16`（`−1.1102230246251565e−14pt`）です。
浮動小数点精度ではB&Hと同じDDであり、実質的なDD改善を示すものではありません。
登録済みの集計は事後変更せず、この限界を明記しています。
Technicalのfold12はDDdeltaが実際に悪化（base+2.22131pt、2倍+2.27008pt）しています。

同じ欠損処理の旧方策との差を取ると、平均誤差改善と経済改善の不一致が明確です。
移動平均は旧scale_mean、学習済み2系列は自身のStage13 reliabilityを比較対象にしています。
下表のDDdelta差が正なら、旧方策よりDD面が悪化しています。


| 追加方策 | 欠損処理 | 旧比AlphaEX差 pt | 旧比DDdelta差 pt | 2倍AlphaEX差 pt | 2倍DDdelta差 pt |
|---|---|---|---|---|---|
| 移動平均 | hold | -6.75038 | -0.37947 | -6.74883 | -0.39598 |
| 移動平均 | fallback B&H | -5.06185 | +0.86572 | -5.07196 | +0.85739 |
| Technical移動補正 | hold | -8.01894 | -0.33258 | -7.93537 | -0.32252 |
| Technical移動補正 | fallback B&H | -5.29145 | +0.53255 | -5.27722 | +0.52252 |
| Perp移動補正 | hold | -7.96529 | -0.40549 | -7.90053 | -0.42512 |
| Perp移動補正 | fallback B&H | -6.71701 | +0.74681 | -6.68281 | +0.73234 |


## 予測誤差とテクニカル情報

採点集合は元の2,574時刻のままです。MSEはlog-returnの二乗で、以下のMSE列は
読みやすく10⁶倍しています。MAEもlog-return単位です。表の平均は等重み四半期平均で、
行数重みのpooled MSEは保存JSONに別名で残しています。定数予測や定数を含む系列の
平均rank ICは、定義できない四半期を除外して平均せず`null`としています。


| 予測系列 | MSE ×10⁶ | MAE | ゼロ予測比MSE増減 | 平均rank IC |
|---|---|---|---|---|
| scale_mean | 317.659702 | 0.011855408 | +0.69493% | null |
| technical_raw | 329.152941 | 0.012134228 | +4.33817% | +0.037392 |
| perp_delay0_raw | 327.652481 | 0.012147543 | +3.86254% | +0.040139 |
| technical_scaled | 337.659200 | 0.012397027 | +7.03457% | +0.037392 |
| perp_delay0_scaled | 334.950453 | 0.012328445 | +6.17592% | +0.040139 |
| technical_half | 323.670513 | 0.012023057 | +2.60030% | +0.037392 |
| perp_delay0_half | 322.566830 | 0.011994861 | +2.25044% | +0.040139 |
| technical_reliability | 320.228761 | 0.011927549 | +1.50929% | +0.037392 |
| perp_delay0_reliability | 321.585425 | 0.011938707 | +1.93934% | null |
| rolling_anchor | 316.298944 | 0.011811939 | +0.26358% | -0.040014 |
| technical_rolling | 317.153736 | 0.011822737 | +0.53454% | +0.007933 |
| perp_delay0_rolling | 317.118913 | 0.011814055 | +0.52350% | +0.004448 |


ゼロ予測MSEは315.467427×10⁻⁶、fit期間平均のMSEは316.211081×10⁻⁶。
移動平均自体もゼロ予測より0.26358%、fit平均より0.02779%悪化しています。
Technical/Perpの移動補正はゼロ予測より0.53454%/0.52350%、fit平均より
0.29811%/0.28710%悪く、強い単純対照に勝てていません。

下表は対応する対照からの相対MSE削減率。正は改善、負は悪化です。
前回reliabilityとの比較は両系列とも4/8四半期で改善でした。
移動平均へのテクニカル上乗せはTechnical4/8、Perp3/8改善・1/8同値ですが、
平均では0.27025%/0.25924%悪化しています。


| 追加予測 | 比較対照 | 相対MSE削減 | 改善四半期 | 同値四半期 |
|---|---|---|---|---|
| rolling_anchor | scale_mean | +0.42837% | 4 | 0 |
| technical_rolling | rolling_anchor | -0.27025% | 4 | 0 |
| technical_rolling | scale_mean | +0.15928% | 4 | 0 |
| technical_rolling | technical_half | +2.01340% | 5 | 0 |
| technical_rolling | technical_reliability | +0.96026% | 4 | 0 |
| technical_rolling | technical_scaled | +6.07283% | 5 | 0 |
| perp_delay0_rolling | perp_delay0_half | +1.68893% | 5 | 0 |
| perp_delay0_rolling | perp_delay0_reliability | +1.38890% | 4 | 0 |
| perp_delay0_rolling | perp_delay0_scaled | +5.32364% | 5 | 0 |
| perp_delay0_rolling | rolling_anchor | -0.25924% | 3 | 1 |
| perp_delay0_rolling | scale_mean | +0.17024% | 4 | 0 |


全体MSEは旧reliabilityより改善しましたが、利益は低下しました。
MSEは各時刻を平均する予測損失であり、保有量・経路・大きな値動き・費用を通じた
B&H超過利益と同じ目的関数ではありません。この結果を予測精度改善だけで採用しません。

## 相場別の失敗を残す

相場ラベルは各評価四半期の最初の判断時点に利用できた90日モメンタムと7日ボラから
固定しています。四半期中に実現した上昇・下落を見て付け直した分類ではありません。
件数はbull2/bear4/sideways2のままです。


| 追加方策 | 欠損処理 | 開始時相場 | base AlphaEX/DDdelta pt | 2倍 AlphaEX/DDdelta pt |
|---|---|---|---|---|
| 移動平均 | hold | bull | -0.00477/-0.97983 | -0.11158/-0.92352 |
| 移動平均 | hold | bear | -10.90054/-9.23715 | -10.94228/-9.22790 |
| 移動平均 | hold | sideways | +7.31665/-8.18935 | +7.24558/-8.18143 |
| 移動平均 | fallback B&H | bull | -0.17754/-0.73343 | -0.26549/-0.69341 |
| 移動平均 | fallback B&H | bear | -6.77386/-7.06590 | -6.82069/-7.05665 |
| 移動平均 | fallback B&H | sideways | +7.64342/-8.53662 | +7.56399/-8.51611 |
| Technical移動補正 | hold | bull | -1.94234/+2.00423 | -2.21180/+2.16935 |
| Technical移動補正 | hold | bear | -12.95627/-8.37898 | -13.02685/-8.34697 |
| Technical移動補正 | hold | sideways | +10.77940/-9.35685 | +10.64745/-9.32749 |
| Technical移動補正 | fallback B&H | bull | -0.24343/+0.50466 | -0.44564/+0.61103 |
| Technical移動補正 | fallback B&H | bear | -7.79246/-7.80110 | -7.90539/-7.76561 |
| Technical移動補正 | fallback B&H | sideways | +9.48012/-8.28506 | +9.33015/-8.23447 |
| Perp移動補正 | hold | bull | -1.67115/+1.01418 | -1.96994/+1.19895 |
| Perp移動補正 | hold | bear | -10.88630/-6.84308 | -10.97323/-6.80376 |
| Perp移動補正 | hold | sideways | +9.38356/-9.52299 | +9.30551/-9.51561 |
| Perp移動補正 | fallback B&H | bull | -0.51644/+0.00443 | -0.76312/+0.14336 |
| Perp移動補正 | fallback B&H | bear | -9.08547/-5.72035 | -9.18184/-5.68243 |
| Perp移動補正 | fallback B&H | sideways | +8.64954/-8.99396 | +8.55353/-8.96563 |


全6方策で開始時bull/bearの平均AlphaEXが負です。学習済み移動補正4方策では
bull平均DDdeltaも正でした。開始時bearのfold6だけでもbase AlphaEXは
−26.08〜−49.83ptで、平均の失敗を小さなコスト差で説明できません。
fold6を除外した成績や、その値動きから作った別の相場分類を選択に使っていません。

予測でも、Technical/Perpの移動補正は移動平均よりbullで0.68926%/0.67018%、
bearで0.04824%/0.02496%、sidewaysで0.09205%/0.11470%悪化しています。
旧reliabilityより各相場平均MSEは減りましたが、追加情報の対照には全相場で敗れました。
sidewaysではゼロ予測に勝っても、scale_meanとrolling_anchorへの優位性はありません。
したがって登録した全相場予測条件も全相場経済条件も6方策すべてfalseです。

## 移動アンカーに対する記述的分解

`d=mu−a_t, r=y−a_t`として、MSE差は`E[d²]−2E[d*r]`。
以下は同じ移動アンカー基準の等重み平均で、各列を10⁶倍しています。
中心化項は`Var(d)−2Cov(d,r)`、平均項は`E[d]²−2E[d]E[r]`です。


| 予測系列 | 中心化項 ×10⁶ | 平均項 ×10⁶ | MSE差 ×10⁶ |
|---|---|---|---|
| technical_reliability | -1.711128260 | +5.640945534 | +3.929817274 |
| perp_delay0_reliability | -2.091809200 | +7.378290513 | +5.286481313 |
| technical_rolling | -0.568984617 | +1.423777028 | +0.854792411 |
| perp_delay0_rolling | -0.778260309 | +1.598230197 | +0.819969888 |


追加2系列の中心化項は負でも、平均項が正で合計MSE差は負になりません。
この分解は記述的な恒等式です。`Cov(d,r)=Cov(d,y)−Cov(d,a_t)`なので、
Stage13の固定scaleアンカーに対する分解と同じ基準として直接比較できません。
今回同時に変えたリターン中心と予測中心の効果を個別に同定したものでもありません。

## 実装・監査・再現情報

事前登録コミット`845a9fd167533599c7189019b38dc9ca2edf0f41`をpush後に実行。
プロセス91776は全8foldを完了しexit0。0 base fits、0 weight fits、16固定重みコピー。
645件の全テストが56.191秒で成功し、`git diff --check`も通過しています。
全テストログSHAは`3a931959c3db233a4a9733b032da623ad6ffb22e993bcac10ad416f3867e3098`。

- 事前監査: 1536祖先成果物と同じfold/model由来のraw予測を確認。
  全2586履歴が64件以上、最小179件。時刻membership SHAと件数を実行前に固定。
- 予測監査: cutoff付きParquet読み込みから独立にリターンを再構成。
  全2586履歴のmembership、移動平均、raw平均、24予測NPZの最大差0。
  元の推論2586/採点2574と共有分散も一致。
- 注文・会計監査: 48新経路・16512判断、176方策行の352会計を独立再計算。
  注文・NAV・AlphaEX・MaxDDdelta・費用の最大差0（平均保有率のみ2.22e−16）。
  新6方策分の未採点判断72件、fallback判断996件を保持。
- 集計監査: 96score/分解、16固定重み、全比較・方向判定を独立計算。
  集計最大差は経済1e−15、対比較9.5e−16、予測6.5e−18、score3e−18、分解1e−18。
  128既存対照と72既存scoreの差は0。Perp fold12の2欠損処理は移動平均と完全一致。

監査ごとにソース・直接入力・重複束縛の数え方が異なります。
共通の成果物数は祖先1536、新規256。予測監査の1575はソース束縛、
集計監査の1843は列挙されたファイル、注文監査の1853はその範囲でのdistinct file数です。
これらを新たな独立サンプル数として扱いません。

| 記録 | SHA256 |
|---|---|
| config | `288e54aa1ad24c91f445e39d857561cbe6d3ba93988fadd974263f2ab5df59f7` |
| preflight | `8d9da4c4c04952e01b5194336605abe6a2480b72d8b4b68b0cf18c4a942347d1` |
| results | `d5adfc39e2822bfb77aaa519202b9949d664f772d704bcd63fa535147da9b620` |
| source audit | `c590d0511e910bbf1e0ecf26803f439473735ff4553ecb19a3abb172e08b607c` |
| forecast audit | `6873ddce19f73fd497ca5c7c748f7378f712fc0c720e3004fd80a55c87810f49` |
| orders/accounts audit | `ec64d4f83249c5b38aaf0be97483f5123035831e141e3ffe90058b2033f241a6` |
| summary audit | `6030e2b8b24d9e856387103ba4ffb3ed059963feb254085bc8f3cd3d9b5f8692` |



[登録](oracle_rolling_centering_registration_20260906.md) ·
[研究ノート](oracle_rolling_centering_research_20260906.md) ·
[全結果](oracle_rolling_centering_evidence_20260906/results.json) · [事前検証](oracle_rolling_centering_evidence_20260906/registration_verification.json) ·
[予測監査](oracle_rolling_centering_evidence_20260906/oracle_rolling_centering_forecast_audit_20260906.json) ·
[注文監査](oracle_rolling_centering_evidence_20260906/oracle_rolling_centering_audit_20260906.json) ·
[集計監査](oracle_rolling_centering_evidence_20260906/oracle_rolling_centering_summary_audit_20260906.json)。
同じevidenceフォルダに独立監査コード、元登録、全fold manifestを保持。
完全NPZと注文traceはローカル`codex_outputs/oracle_rolling_centering_decisions_v1/`にあり、
保存manifestから各ファイルのSHAを追跡できます。



## 判断と次の方向

移動補正は採用しません。現在の指標と固定重みに対し、平均を追従させるだけでは
トレンドに依存しない予測情報も経済優位も確保できませんでした。
前回の固定reliability候補には開発相場平均でAlphaEX正・DDdelta負という証拠が
残りますが、単純対照への予測優位や独立汎化が確認されたわけではありません。
それを「最強」と呼ぶ根拠も増えていません。

ここまでの静的・移動平均補正の系列をいったん区切り、次は利用可能なテクニカル
特徴量に将来リターンを予測する安定した情報があるかを中心に調べます。
新しい窓幅やモデル構造を成績に合わせて探索せず、次の比較対象・時系列分割・
支持集合・単純対照・失敗基準を別途固定してから計算します。
今回の結果から新しい特徴量や教師、本番運用・売買への採用は行っていません。

[Dawid1984](https://academic.oup.com/jrsssa/article/147/2/278/7106293)の逐次予測の考え方と、
[Pesaran–Timmermann2007](https://rady.ucsd.edu/_files/faculty-research/timmermann/estimation-window.pdf)の
構造変化下の窓幅と推定誤差の議論を参照しました。3か月窓の最適性を示す文献ではありません。
[Dimitriadis–Puke2026](https://arxiv.org/html/2603.04275v1)の分解に関する推測統計の仮定も
この実験では成立を確認しておらず、p値や保証は転用しません。

再利用された8四半期、重なりのある履歴、依存した損失、2/4/2の相場件数、
受信時刻の証跡不足という限界は変わりません。確定した過去の評価ラベルを固定手順で
次の予測へ取り入れる逐次評価であり、評価期間のラベルを一切更新に使わない方式とは異なります。
独立確認・全相場の高確率保証・正式P1結果・本番運用の成立は主張しません。
