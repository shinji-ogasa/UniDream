了解。今回はちゃんと **現状の課題分析 → 複数仮説 → 実験候補 → 優先順位 → 怪しい点** で出す。
まったく、ここでまた一本道にしたら同じミスだからね。

## まず現状整理

今のPlan4で分かったことはこれ。

```text
1. WM return head はほぼ使えない
2. WM vol head は使える可能性あり
3. WM drawdown head は弱い
4. guardは大半の危険イベントを正しく止めてそう
5. ridge + WM の単純OR/MAX/ANDは失敗
6. WM overlayはfold5型イベントには効くが、fold4/6には効いてない
```

WM calibrationでは、return headはfold4/5/6で相関がほぼゼロ。vol headはfold4で強く、fold5も中程度、fold6は弱め。DD headはfold4以外だとかなり弱い。だから今のWMは「return予測器」ではなく、**vol/risk regime signal として見る方が自然**。

Blocked event attributionでは、ブロックされたイベントの平均実現utilityがfold4/5/6すべてで負。`util>0_rate` も10〜16%程度なので、guardは基本的に「止めすぎ」より「危険/不要イベントをちゃんと止めている」寄り。

一方で、ridge+WMの単純ORはfold4で `turnover 80.0`、MAXはfold4で `turnover 177.6` と壊れている。ANDは全部neutral。つまり **素朴なensembleはダメ**。

---

# できてない・怪しい部分

ここ大事。今の報告だけだと、まだ危ない点がいくつかある。

## 1. Plan3のWM fold5 +8.431 と Plan4のWM_hard 0 が食い違って見える

Plan3ではWM overlayがfold5で `+8.431pt` を出したはず。
でもPlan4 Round Dでは `WM_hard` がfold4/5/6全部 `0.000` になってる。

これは要確認。

```text
可能性:
  - Plan3とPlan4で実装条件が違う
  - threshold/cooldown/guard適用順が違う
  - WM_hard の定義がPlan3候補と違う
  - checkpoint / resume / config が違う
```

ここを放置すると、後続の比較が全部ズレる。最初に再現性確認が必要。

## 2. WM all14 はまだ見えてない

Plan4はfold4/5/6。
Plan2 ridge系はall14で見たけど、**WM overlay単体のall14** はまだ未確認っぽい。

なので、WMが本当にfold5専用なのか、他foldにもスポット検出があるのかは未確定。

## 3. Blocked attributionは「サンプル700件」っぽい

報告では各fold `n_blocked_events=700`。
これは全blocked eventの完全検証ではなく、サンプル診断の可能性がある。

だから、

```text
guardは概ね正しい
```

とは言えるけど、

```text
guardが絶対に止めすぎていない
```

とはまだ言い切れない。

## 4. Soft throttleが全部neutralなのは、throttle以前に発火条件を通ってない疑い

danger scaleを `0.25/0.5/0.75` にしても全foldneutral。
これは「soft throttleが効かない」というより、**thresholdや前段guardで候補が消えていて、scale変更まで到達してない**可能性がある。

つまり、soft throttle実験として成立しているか少し怪しい。

## 5. Ensembleはまだ雑な3種類しか見てない

OR / AND / MAXは見た。
でも本当に試したいのはこっち。

```text
WM primary + ridge veto
ridge primary + WM veto
ridge primary + WM boost
fold内percentile ensemble
vol-regime別 ensemble
```

だから「ridge+WM統合は失敗」と断定するのは早い。
**単純統合は失敗**が正しい。

---

# 探索ボード

## Lane 0: 再現性・差分確認

### 仮説

Plan3とPlan4でWM overlay結果が食い違っている可能性がある。
まず同じ条件で再現しないと、以降の比較が危ない。

### 実験

```text
0-A. Plan3 WM overlay候補を同じcommit/config/checkpointで再実行
0-B. Plan4 WM_hardと完全同条件比較
0-C. threshold, cooldown, guard順序, checkpoint path, seed, deviceをログ出力
0-D. fold5のactive event ID / timestamp を比較
```

### 採用/棄却ライン

```text
Plan3 fold5 +8.431 が再現:
  WMスポット検出は本物候補

再現しない:
  Plan3結果は条件依存。WM overlay評価をやり直し
```

### 優先度

**最優先。**
これをやらないと全部怪しい。

---

## Lane A: WM信号の再解釈

### 仮説

WMをreturn予測器として使うのが間違い。
今使えるのは return ではなく **vol / risk regime signal**。

### 根拠

return headはICほぼ0、vol headはfold4/5で有効、dd headは弱い。

### 実験候補

```text
A1. return headをutilityから外す
A2. vol headだけをrisk filterに使う
A3. dd headを直接penaltyに使わず、fold内percentile化
A4. WM vol high/lowでridge overlayの発火条件を変える
A5. returnはridge側、riskはWM vol側に役割分担
```

### 見る指標

```text
fold4/5/6:
  active count
  AlphaEx
  MaxDDΔ
  turnover
  selected event utility
  fold別vol regime分布
```

### 期待

かなりある。
WMを「Alpha信号」ではなく「危険局面/ボラ局面フィルタ」として使う方が自然。

---

## Lane B: Ridge + WM ensemble 2.0

### 仮説

ridgeとWMは相補的だが、OR/MAX/ANDが雑すぎる。
役割を分けた統合なら改善するかもしれない。

### 実験候補

```text
B1. ridge primary + WM vol-risk veto
  ridgeが発火、WMが高リスクなら止める

B2. WM primary + ridge safety veto
  WMが発火、ridgeが明確に危険なら止める

B3. ridge primary + WM boost
  ridge発火時にWM riskが低ければサイズ/信頼度を上げる

B4. percentile ensemble
  ridge_scoreとWM_scoreをfold内分位に変換して足す

B5. regime-specific ensemble
  low-volではridge重視、高-volではWM veto重視
```

### 見る指標

```text
active fold数
median AlphaEx
worst AlphaEx
worst MaxDDΔ
turnover max
fold3/fold5依存
selected event count
```

### 期待

高い。
ただし単純OR/MAXはすでに壊れているので禁止。

---

## Lane C: Guardの本当の役割分析

### 仮説

guardは大半正しいが、一部foldでは発火候補を消しすぎている可能性がある。
特にsoft throttle実験は、scale変更が発火候補まで届いてない疑いがある。

### 実験候補

```text
C1. guard前 candidate count
C2. threshold後 candidate count
C3. danger guard後 candidate count
C4. pullback guard後 candidate count
C5. 最終active count
C6. 各段階で消えたeventのcounterfactual utility
```

### 見る指標

```text
どの段階でfold4/6が死んでいるか
消されたeventの平均utility
消されたeventのMaxDD寄与
```

### 判断

```text
threshold前から候補なし:
  WM utilityが弱い

threshold後に消える:
  threshold/validation選択問題

danger/pullbackで消える:
  guard設計問題

最終だけ消える:
  state machine / cooldown問題
```

### 期待

高い。
今の「なんでneutralなのか」を切る実験。

---

## Lane D: Soft throttle再設計

### 仮説

今のsoft throttleは実験として不十分。
guard後にscaleするのではなく、**guardが消す前の候補に対して縮小**しないと意味がない。

### 実験候補

```text
D1. danger判定eventを消さずに position delta 0.25倍
D2. danger判定eventを0.5倍
D3. pullbackは完全停止、dangerだけ縮小
D4. dangerは縮小、pullbackは縮小なし
D5. danger score連続値でscale = sigmoid(-risk)
```

### 見る指標

```text
fold4/6活性化
fold5保持
MaxDDΔ worst
turnover
event count
```

### 期待

中。
ただしCで「guard前候補が存在する」と確認してから。

---

## Lane E: WM calibration後のutility grid

### 仮説

dd/vol予測値の絶対スケールがズレている。
fold内z-score/percentile化してutilityに使うと安定するかもしれない。

### 実験候補

```text
E1. raw WM utility
E2. zscore calibrated WM utility
E3. percentile calibrated WM utility
E4. return無効 + vol percentile penalty
E5. ridge return utility + WM vol risk penalty
```

### 見る指標

```text
active fold数
median Alpha
worst Alpha
worst MaxDD
turnover
fold4/6が動くか
```

### 期待

中〜高。
今のWM headのスケール比が大きいので、絶対値のまま使うのは怪しい。

---

## Lane F: Ridge側の拡張

### 仮説

ridgeはfold4/6を拾えていて、安全寄り。
WMより先にridge側を少し拡張した方が堅い可能性がある。

### 実験候補

```text
F1. ridge D+A+pullback の all14再現を固定
F2. false-de-risk labelを追加
F3. pullback recovery labelを追加
F4. ridge score + WM vol veto
F5. ridge scoreのactive fold増加探索
```

### 見る指標

```text
all14 worst Alpha >= 0
median > 0
active fold rate
fold3抜きmean
fold5抜きmean
```

### 期待

高い。
現時点でridgeは一番安全な土台。

---

## Lane G: WM all14検証

### 仮説

WMはf456だけだとfold5専用に見えるが、all14では別のスポット検出があるかもしれない。

### 実験候補

```text
G1. WM overlay all14
G2. fold別 active count
G3. fold別 threshold/cooldown
G4. fold別 blocked count
G5. fold3/fold5抜きmean
```

### 判断

```text
他foldにもスポットあり:
  WM補助信号として残す

fold5だけ:
  WMは限定的な補助

壊れるfoldあり:
  WM単体禁止
```

### 優先度

高い。
WMを語るならall14が必要。

---

## Lane H: Stress / robustness

### 仮説

低turnoverなのでコスト耐性はありそう。
ただし1トレード依存ならslippageや1bar delayに弱い可能性がある。

### 実験候補

```text
H1. cost x1.5 / x2
H2. slippage x2
H3. execution delay 1 bar
H4. threshold jitter
H5. fold3/fold5抜きstress
```

### 優先度

後段。
候補が絞れてからでいい。

---

# 優先順位

今はこう進めるのが正解。

```text
Round 0:
  Plan3 WM +8.431 と Plan4 WM_hard 0 の差分確認

Round 1:
  Lane C: guard段階別candidate count
  Lane G: WM all14
  Lane A: WM信号再解釈

Round 2:
  Lane E: calibrated WM utility
  Lane B: ridge+WM ensemble 2.0
  Lane F: ridge拡張

Round 3:
  Lane D: soft throttle再設計
  Lane H: stress

Round 4:
  standalone overlay候補化
```

## 直近で投げる指示文

```text
Plan5として、Plan3/Plan4のWM overlay差分とridge-WM統合可能性を検証する。

1. Plan3 WM fold5 +8.431 と Plan4 WM_hard 0 の差分を調べる。
   checkpoint, config, threshold, cooldown, guard順序, active event timestampを比較する。

2. WM overlayの候補生成パイプラインを段階別に分解する。
   pre-threshold, post-threshold, danger guard後, pullback guard後, cooldown後のcandidate数とutilityをfold4/5/6で出す。

3. WM overlayをall14で実行する。
   fold別AlphaEx, MaxDDΔ, active count, blocked count, threshold, cooldownを出す。

4. WM信号をreturn予測ではなくvol/risk filterとして使う候補を試す。
   return無効、vol percentile penalty、ridge return + WM vol vetoを比較する。

5. ridge+WM ensemble 2.0を試す。
   OR/MAX/ANDは禁止。
   ridge primary + WM veto、WM primary + ridge veto、ridge primary + WM boost、percentile ensembleを比較する。

6. まだAC、route unlock、configs/trading.yaml変更はしない。
```

## まとめ

今の探索ボードで一番大事なのは、

```text
WMをalpha予測器として扱うのをやめる
```

こと。
今のWMは **returnではなくvol/riskを見る補助信号** として再設計するべき。

それと同時に、Plan3/Plan4のWM結果の食い違いは最優先で潰す。
ここがズレたまま次に進んだら、また局所解どころか、比較そのものが壊れるわよ。
