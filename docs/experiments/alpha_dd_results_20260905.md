# Alpha/DD 結果レポート（2026-09-05）

## 結論

BTC の combined 結果は `AlphaEx -0.206849pt / ΔMaxDD -3.769641pt` で、alpha の最低基準を満たさず fail だった。cross-asset lock 後の BNB confirmation は `+1.356381pt / -6.251727pt` で最低基準のみ pass、preferred は fail。追加の ML25 は `-1.420887pt / +0.408327pt` で両基準 fail だった。したがって、今回の結果から HF deploy、live causal archive、投資判断、または正式 P1 結論を主張しない。

すべての値は B&H-relative の percentage points 表記である（保存 JSON の値は fraction）。元出力の SHA、コピー方式、データ provenance は [evidence manifest](alpha_dd_evidence_20260905/manifest.json) に固定した。

## 凍結した選択と結果

| 系列 | confirmation 前に固定した候補 | combined AlphaEx | combined ΔMaxDD | 最低基準 | preferred | 判定 |
|---|---|---:|---:|---|---|---|
| BTCUSDT | `trend_d30_lo0.5_hi1.12_p0` | -0.206849 | -3.769641 | fail | fail | fail |
| BNBUSDT | `trend_d90_lo0_hi1.12_p0` | +1.356381 | -6.251727 | pass | fail | preferred fail |
| ETHUSDT | `trend_d30_lo0.5_hi1.12_p0`（development のみ） | — | — | — | — | confirmation not run |
| BTCUSDT ML25 | `ridge_d7_lo0.5_hi1.12_p2` | -1.420887 | +0.408327 | fail | fail | fail |

各 asset の development 選択は、登録済み候補に対する `max min(mean AlphaEx, -mean ΔMaxDD)`（決定的 tie-break）で行い、historical/fresh の結果を見る前に lock した。cross-asset lock は [原本コピー](alpha_dd_evidence_20260905/cross_asset_lock.json) のとおり BNBUSDT を確認対象に固定し、ETH confirmation を禁止している。従って ETH の未実施結果を補完したり、BNB の test 結果で asset 選択を変えたりしていない。

## BNB の内訳

選択候補の 9-fold historical は `+0.773879pt / -5.596713pt` で最低基準 fail、1-fold fresh は `+6.598890pt / -12.146857pt`、2x cost stress は `+0.677865pt / -6.044218pt` で fail だった。combined では alpha positive が 4/10、median AlphaEx は `-2.067762pt`、median ΔMaxDD は `+1.715251pt`。descriptive alpha CI は 0 を跨ぐ。全 holdout rows と provenance は [BNB historical](alpha_dd_evidence_20260905/bnb/historical.json)、[BNB fresh](alpha_dd_evidence_20260905/bnb/fresh.json)、[BNB qualification](alpha_dd_evidence_20260905/bnb/qualification.json) に保存した。

親側から共有された agent-reported independent evaluation audit では、raw BNB と保存済み選択 NPZ を対象に、base/stress 全 10 窓の AlphaEx、ΔMaxDD、returns、fees、borrow、turnover、trades の再計算が max absolute difference `0`、endpoint-momentum target も全 10 窓で一致し、registration/source/selection/cross-asset lock の SHA も検証済みと報告された。監査は研究 evaluator を import せず、scalar の cash/units 会計を独立実装した。これは agent 報告であって親による独立再計算・金融上の証明ではない。従って数値/accounting の P0 blocker はないが、BNB の判定はなお最低基準のみ pass であり、train provenance は core の正確な 2 年 training と 3 か月 validation gap を区別して扱う。

## ML25

ML25 は 25/25 candidate eligible、N/A 0、development fit 78 件、最小 fit rows 666 件だった。固定された `ridge_d7_lo0.5_hi1.12_p2` の historical は `-1.423648pt / +0.250516pt`、fresh は `-1.396041pt / +1.828625pt`、2x cost stress は `-2.337293pt / +0.580443pt`。selection lock、全候補の compact aggregate、全 holdout rows は [ML evidence](alpha_dd_evidence_20260905/ml/) にある。ML の追加結果も formal P1 result ではない。

## データ provenance と範囲

各 asset は公式 Spot monthly 15m archive を 2018-01–2026-08 inclusive で要求し、2026-07 まで 103/104 月、300,240 available rows を取得した。全体の3,600 missing grid rowsは、2026-08のHTTP 404による `unavailable_tail` 2,976行と、それ以前の欠損624行に分かれる。BTC/ETH/BNB の latest verified July archive identity はそれぞれ `b0436766…`, `d4742ef1…`, `33b1abfc…`（完全な URL、revision、checksum は [manifest](alpha_dd_evidence_20260905/manifest.json)）である。archive published/collector/exchange timestamp は null/非 live observation で、live causal timestamp の主張はしない。

BTC の最初の rule-based run は ledger SHA `f2f752bf…` と sidecar SHA `e99f584e…` を使い、後続 ML run は更新後 ledger SHA `8954b60b…` と sidecar SHA `a681f5b9…` を使った。旧 ledger は現行 ledger の先頭 544 行、1,555,904 bytes の prefix として SHA 検証済みであり、manifest に両方の provenance を保持した。registration の実体 SHA と JSON 内に埋め込まれた registration SHA が異なる場合も、どちらも上書きせず記録している。

## 解釈上の境界

初期 cross-asset 候補数は 83 候補 × 3 asset = 249（N/A を含む）で、ML25 は別の 25 候補 preregistration として追加実行済みである。historical folds 15–23 と fresh fold 24 は再利用された窓であり、untouched holdout や selection-adjusted significance ではない。WM700 step / BC8 epoch / AC300 step は別の diagnostic であり、[別レポート](p1_formal_forecast_wm_bc_ac_20260905.md)を参照するだけで本結果には混ぜない。

## Evidence map

- [snapshot README](alpha_dd_evidence_20260905/README.md)
- [manifest（元/コピー SHA、data provenance、archive identity）](alpha_dd_evidence_20260905/manifest.json)
- [BTC evidence](alpha_dd_evidence_20260905/btc/)
- [ETH development と未実施 status](alpha_dd_evidence_20260905/eth/)
- [BNB evidence](alpha_dd_evidence_20260905/bnb/)
- [ML25 evidence](alpha_dd_evidence_20260905/ml/)
