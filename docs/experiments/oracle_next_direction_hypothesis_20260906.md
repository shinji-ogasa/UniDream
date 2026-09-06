# 次の研究仮説：凍結した確率を、符号への丸めを経ずに売買判断へ渡す

2026-09-06。**研究上の提案のみ。Stage19 の登録・実装・実行はまだ行っていない。** Stage16–18 の公開済み protocol / research / results と一次資料だけを読んだ。新しい市場統計、係数、予測、損失、注文は計算していない。追加 test15–24 は参照しない。Stage18 公開時点は `96447c4600979c4ce5c66140fe60ddd27448c2d2`。

## 一つの推奨仮説

**既存の magnitude-weighted 確率に残る連続的な強弱を保持すれば、`sign(z)*abs(parent_half_mu)` という固定写像より、費用・リスクを越える売買判断に使いやすい平均予測になるか。** 新しい方向モデルを学習する前に、この情報を捨てる段階を分離して検証する。

Stage16 は正解の将来符号に大きな価値があるという hindsight 診断だった。Stage17 の因果的分類器は親に対する改善条件を満たさず、Stage18 は確率損失を C1 より改善しても、E で prior に及ばず、全新方策の経済条件を失敗した。この事実は方向情報の獲得を証明しない。一方、両段階の注文用平均は確率の強弱を捨て、別の親モデルの絶対平均に置き換えていた。この未分離の写像だけを次の比較対象とする。

既存の T-only `fit_abs_return_mean` を `a_T`、保存済み magnitude-weighted class-1 probability を `q_t` とし、計算順も固定する：

```
mu_soft[t] = a_T * (2.0*q_t - 1.0)
```

元の推論可用時刻のみ計算し、それ以外は NaN。q=0.5 は厳密にゼロ、epsilon・temperature・係数選択・追加校正は使わない。`a_T` を再推定しない。この写像は確率の符号を通常保持しながら、注文に渡す平均の大きさを変えるので、既存 sign-only 写像への正の temperature 変換とは異なる。

## 数学的動機と限界

Y を h24 return、A=|Y|、B=1{Y>0} とする。無制約の母集団 weighted logloss 最小点は `q*(x)=E[AB|x]/E[A|x]`。したがって `E[Y|x]=E[A|x]*(2*q*(x)-1)`。これは条件付き独立を仮定しない重み付き損失の代数であり、BTC で推定できたという主張ではない。通常の上昇確率 P(B=1|x) へ同じ式を適用してはならない。

本提案の `a_T` は E[A|x] の定数近似であり、時間変動する条件付き絶対リターンではない。有限標本・正則化された q も q* とは限らず、既存の proper-score 失敗はそのまま残る。Anatolyev–Gospodinov の原論文は符号と絶対値の**結合分布**から平均を復元している。本提案はその copula / 動的モデルの再現でも、単純な E[sign(Y)]E[|Y|] の正当化でもない。[著者公開の受理稿、§2.1–2.4](https://pages.nes.ru/sanatoly/Papers/Decomp.pdf)

## 固定する最小比較

- Stage17 C1 と Stage18 L2unit に既に保存された **2特徴群×2既存正則化の magnitude 系4確率**を全て使う。新しい C 値・fit・特徴・ターゲット・期間・閾値は追加しない。ordinary 系の除外理由は上の estimand の違いであり、成績による選別ではない。4写像×既存 hold/fallback の比較で、既存のどの確率源も事後的に落とさない。
- 主な paired 対照は、同一分類器の既存 hard-sign 方策、対応する original half、`a_T*(2*pi_weighted_T-1)` の定数写像。加えて保存済み fit-mean と zero mean、B&H / common robust を保持する。定数写像は実数演算では fit mean と一致するが、浮動小数点の差を勝手に同一視しない。方策を比較する定数対照も同じ risk・推論 mask・欠損ルールで実行する。古い方策を流用できるのは、この契約まで完全一致すると確認できる場合だけ。
- 既存全60方策と全ての失敗結果を保存する。新しい mapped-prior / fit-mean 対照の既存成果物との重複と正確な総数は、将来の登録前に固定する。今は試行数台帳を更新しない。

この比較は**方向の学習ではなく確率から平均・注文への写像の ablation**。q とその確率損失は不変であり、変更後の経済結果から分類精度の改善を主張しない。定数方向への接近やゼロ付近への縮小だけを「利用可能な特徴情報」と呼ばない。

## 因果的な時系列と反証条件

元 development5–12、T18/S3/I3/E3、cutoff<2023-04-16T13:45Z を継承する。T だけで既に fit された q / a_T / prior を固定し、S で再校正しない。I と E を別々に報告し、原6 maskを保持する。t の特徴は t−1 足まで、h24 label は375分後に成熟。E 推論2,586行には採点できない12行を残す。future score mask を写像や注文の可否に使わない。イベント時刻は archive-as-of の仮定であり、実運用の受信時刻を証明しない。

risk、6h decision、現在始値での既知情報、自己の cash/units、次足だけの約定、費用、借入、max-step/deadband、hold/fallback を固定する。stress は base と同じ intent に2倍費用・借入を適用する。教師の在庫、Oracle のラベル、後から判明する始値欠損で決定を差し替えない。

反証可能な条件を将来の protocol に事前固定する：

1. **利用可能な平均情報**：I/E 別、all/bull/bear/sideways の equal-quarter MSE が対応 hard map・parent half・mapped prior・fit mean・zero を全て厳密に下回るか。定数対照を越えなければ「方向情報が使えるようになった」という仮説を棄却する。MAE、符号、未定義IC、行数も全て保存する。
2. **経済的な目的**：両費用で各 stratum の AlphaEX>0 / DDdelta<0、かつ対応 hard map・parent half・mapped prior に対して AlphaEX差>0 / DDdelta差<0。平均だけの利益、片側のDD、都合のよい一方の欠損ルールで採用しない。写像の技術検証とこの経済条件は別に判定する。
3. 単なる hard map 対比の改善を記録しても、定数・親・trend 条件を失敗すれば採用しない。既存 q の proper-score 失敗を撤回しない。再利用2/4/2四半期から高確率の汎化保証は出ず、全条件を満たしても独立・受信時刻付きの確認が必要。P1/outer は不変。

## 代替は一つだけ、今回は推奨しない

次の候補は、在庫・費用を含む action-advantage を目的とした固定線形 decision-focused fit。しかし正確な学習用状態、内側の時系列 cross-fit、max-step/skip/借入を含む目的の定義が必要で、上の写像より変更箇所が多い。最初から巨大なRL/構造探索へ進める根拠にはしない。

Elmachtoub–Grigas は予測誤差と誘発された決定損失を区別し、SPO+ surrogate を提案する。[原論文](https://arxiv.org/abs/1710.08005) Liu–Grigas の理論は線形目的・既知の凸 feasible region を中心に、追加条件の下で surrogate と意思決定リスクを関係づける。これは自己在庫で将来の可行な行動が変わり、費用と最大DDを持つ現在の BTC コントローラへの既成の保証ではない。[NeurIPS2021 原論文、§1–4](https://papers.nips.cc/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Paper.pdf)

**このメモは一つの研究リードであり、Stage19 は未登録・未実行。** 成績を見て新たな倍率、C、温度、期間、ターゲットを追加しない。
