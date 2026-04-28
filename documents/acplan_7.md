うん、それで合ってる。
今それを狙うのは早い。**下げ局面でも絶対プラスを取るAIトレーダー**は、今のUniDreamの次フェーズじゃなくて、もっと学習・評価・AC制御が安定してからの上位目標。

今の段階での正しい位置づけはこれ。

```text
現在:
  B&Hを基準にした長期保有改善レイヤー

目的:
  B&Hの取り逃がしを減らす
  DDを少し削る
  小さいoverweightで上乗せする
  turnoverを低く保つ

まだ狙わない:
  下落相場で絶対収益プラス
  full short
  aggressive hedge
  unrestricted AC
```

今のPlan 5は、**B&Hを下回らないfloor + gated overweight** が効いて、multi-foldでAlphaExをプラスにした段階。これはかなり大事。
ここでいきなり「下げでも絶対プラス」を狙うと、また short collapse / churn / overfit に戻る可能性が高い。

次にやるべきは、たぶんこの順番。

```text
1. 全fold検証
2. 月次/局面別のB&H相対評価
3. downside capture / upside capture
4. floorとadapterの寄与分解
5. selector改善
6. overweight sizing限定AC
7. それが安定してから hedge / short / absolute-return mode
```

つまり、今は **相対運用の安定化フェーズ**。
絶対収益AIトレーダーは、その後に別モードとして作るやつ。

あんたの言う通り、まずは学習が安定して、multi-foldで再現して、ACが局所解に落ちないことを確認してから。焦って最終形に飛ぶと、また壊すだけよ。
