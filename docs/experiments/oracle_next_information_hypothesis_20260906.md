# 次の情報仮説：既存の短期8指標を、固定した方向予測の課題で反証する

2026-09-06。研究メモのみ。**Stage20の登録・実装・実行・採用ではない。**
参照時HEAD: `10262e7c95ba1375444d6db6142057ab4fc8f122`。公開済み開発報告と特徴ソースだけを読み、
新しい市場データ、統計、係数、確率、写像、注文、損失は計算していない。追加test15–24は対象外。

## 推奨する仮説は一つ

**Stage15の固定短期8指標には、既存Technical29に対して、同じ次足約定・6時間リターンの
方向確率を改善する情報が残っている。ただし、その改善は大きな値動きを重視した確率損失にも
現れなければ、平均リターンや売買に使える情報とは扱わない。**

これは未検証の「特徴×予測課題」の交差である。Stage15はこの8指標をraw Ridge100の平均予測で
検証し、EのMSEが既存より0.10830%悪化した。価格だけ0.18463%、フローだけ0.08352%悪化し、
全6方策を不採用にした。一方、Stage17–19の分類器はTechnical29/Perp31で、短期8指標を入力していない。
有利だったフローだけを選ばず、既に定義・監査した8指標全体を一つのブロックとして使う。

Stage12の将来平均置換とStage16の符号Oracleは方向情報の経済的な余地を示したが、学習可能性は示さない。
Stage13の固定縮小・Stage14の因果的中心補正、Stage18の強いL2、Stage19の連続写像でも
採用条件は満たしていない。次は同じqの再写像を精度改善と数える実験を増やさない。

## 特徴の位置づけと時刻

Technical29の内訳は次の通り。

- Base16: momentum_{1,7,30,90}、vol_{1,7,30}、drawdown_{7,30,90}、vol_ratio、flow_1、flow_7、return_coverage_7、price_coverage_90、flow_coverage_7。
- 追加13: rsi14/96、atr14_relative/96_relative、channel_position96/672、price_zscore96/672、downside_upside_log_vol_ratio96、efficiency_ratio96/672、weighted_flow96/672。
- Perp31はTechnical29にperp_weighted_flow24/96を追加したもの。旧delay1/4はこの2列だけの追加遅延である。

今回候補は既存の`technical_short_both`（29+8=37列）だけ。旧derivative37とは異なる。
8列はspot_log_return4/16/48、spot_body_sign1、spot_close_location1、spot_weighted_flow4、
perp_weighted_flow4、spot_quote_activity24_672。窓長や符号、ブロックを選び直さない。

15分足のOPEN timestampを使い、判断tの新価格リターンはlog C[t−1]−log C[t−1−k]。
全k+1終値を必要とする。実体符号と終値位置は直前足の有効な正のOHLCから計算し、
測定済み完全平坦足だけ中立0。フローはΣ(2買いquote−quote)/Σquote、4/4有効足を必要とし、
活動比は独立した24/24・669/672のquote平均のlog比。新列の最後にshift1を一度だけ行う。
元のTechnical29、Spot15分フルグリッド、UM整列、欠損、推論支持を変えない。
元fit/predictマスク上で追加列が非有限なら実験全体を失敗にし、共通支持を縮めない。
これらはarchive event-timeの仮定であり、実際にtまでに配信された証拠ではない。

## 最小の固定比較案

元Stage17のTechnical29分類器2本を対照に、Technical29+8を同じordinary/magnitude lossで各1本だけ学習する。
8foldなら新規16fit。Perp31へも展開する、別Cを試す、ブロックを再分割する、閾値を探すことは含めない。
C=1は成功した設定の選抜ではなく最初の祖先設定を固定した対照である。
StandardScalerはTで非加重、LogisticRegression C=1/L2/lbfgs/tol1e−8/max_iter1000/seed20260906、
既存の収束・scalar検証をそのまま保つ。ordinaryの重み1、magnitudeの重み|Y|/mean_T|Y|、
Tのクラスprior・平均絶対値・平均リターンも既存と同じ。新しい校正やS重み推定はしない。
相関した列の追加はL2の実効的な正則化も変えるので、個々の指標の因果的価値を同定する実験ではない。

ordinaryはP(Y>0|X)の診断対照。magnitudeの母集団最適値は
q*=E[|Y|1{Y>0}|X]/E[|Y||X]で、通常の上昇確率とは区別する。
経済評価はmagnitudeの1系列だけを、Stage19で既に固定したa_T(2q−1)で平均へ写像し、
hold/fallbackの2方策を保存する。これは新しい写像探索ではない。
旧Technicalのsoft/hard、own half、zero、fit mean、mapped prior、B&Hを各対応対照として残す。
共有technical分散、自己現金・在庫、次足だけの約定、費用、借入、同一注文の2倍費用再生は固定する。

元development validation5–12のT18/S3/I3/E3か月を維持する。
Y=log(close[t+24]/open[t+1])、成熟t+375分。Tのラベルだけでfitし、I/Eのラベルは採点に限る。
元Eの推論2,586と採点2,574、末尾の未採点推論12件を混同しない。
新しいhorizon、判断cadence、将来ラベルでの注文制限を加えない。

## 事前に区別して反証すること

1. **情報仮説:** ordinaryは通常Brier/logloss、magnitudeは|Y|加重Brier/loglossの両方で、
   対応するT priorとTechnical29分類器を下回る必要がある。I/Eを別に報告し、等重み四半期平均と
   全E開始時相場の結果を保存する。AUCや的中率だけの改善は通過にしない。
2. **平均への接続:** magnitude写像のMSE/MAEを旧soft、own half、zero、fit mean、mapped priorと比較する。
   通常方向の改善だけ、またはqを変えない振幅操作だけでは、条件付き平均情報の改善とは呼ばない。
3. **実用性:** 各欠損処理についてbase/2倍費用のAlphaEX>0・MaxDDdelta<0と、
   登録する対応対照への両指標改善を、全体および全E開始時相場で確認する。全失敗行と四半期別結果を保存する。
   確率損失が改善しても経済条件に失敗したら「現コントローラで使える情報」は反証される。

この開発データは既に何度も観察したもので、bull2/bear4/sideways2は頑健性を保証する標本数ではない。
Iを後のE開始時相場で分けた集計は遡及的診断である。通過しても「最強」や高い将来成功確率とはせず、
独立した前向き確認へ進める研究候補に限る。失敗を見た後に特徴/C/閾値の枝を追加して、この実験の成功として扱わない。

## 一次資料と限界

- [Christoffersen–Diebold (2006), Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0520):
  非ゼロ期待リターンと変動するボラのもとで、条件付き平均の予測可能性がなくても符号依存が存在し得る。
  これはMSE失敗と方向情報不在を同一視しない根拠だが、BTCの6時間・加重q・費用控除後収益を保証しない。
  今回確認できたのは出版社abstract。
- [Kitron–Wengrowicz (2026), arXiv v1](https://arxiv.org/html/2608.21888v1):
  著者は短期暗号資産の符号反転、taker flowとの条件付き関連を報告するが、効果は短時間で減衰し、
  gross edge約1.3bpは同論文のround-trip 5bp目安を下回る。観察的機序であり因果同定ではない。
  15分のintrabar結果を、当方の15分遅延約定・6時間ターゲットへ移せる証拠ではない。
  この時間尺度と費用の不一致は本仮説が失敗し得る具体的理由である。
- [Wen et al. (2022), publisher](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)
  は旧研究メモの参照先だが、今回403で本文再取得できず、新しい根拠には使わなかった。

補足として、凍結済み分散を使う条件付き振幅A_tへの置換は仮説として可能だが、
E|Y||Xを分散だけから決めるには分布形や平均の仮定が要る。qが同じなら確率損失は同じであり、
それだけでは方向確率の精度を独立に確立できない。今回の推奨実験には含めない。

## ローカル根拠

参照ディレクトリ: `/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905/`。
`unidream/experiments/{alpha_dd_search,alpha_dd_features,oracle_frontier_features,oracle_derivative_delay_features,oracle_short_features}.py`、
`docs/experiments/oracle_{information_decomposition,mean_reliability,rolling_centering,short_feature,sign_magnitude,direction,regularized_direction,soft_direction}_results_20260906.md`
、Stage17登録、およびStage15 intraday研究メモ。
既存報告値を引用し、新たな集計・再実行はしていない。
