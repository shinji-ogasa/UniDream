# BTCUSDT Robust Overlay v1 結果（2026-09-05）

## 結論

現時点で、BTCUSDT の AlphaEX と MaxDDdelta を同時に改善する候補として最も有力なのは、学習済みニューラルネットではなく、`robust_90d_backbone_7_30_90_consensus_6h` という低頻度・マルチスケールの因果オーバーレイである。

過去の再利用確認窓9個と最新確認窓1個を合算した結果は次のとおりだった。

| 条件 | AlphaEX | MaxDDdelta | Alpha 正の窓 | DD 改善の窓 | 判定 |
|---|---:|---:|---:|---:|---|
| 通常コスト | **+1.629588pt** | **-4.751632pt** | 7/10 | 6/10 | 最低基準 pass、preferred fail |
| コスト・借入 2倍 | **+1.212933pt** | **-4.654601pt** | 7/10 | 6/10 | 最低基準 pass、preferred fail |

ここで AlphaEX は戦略最終 equity − B&H 最終 equity、MaxDDdelta は戦略 MaxDD − B&H MaxDD の percentage points である。保存 JSON では fraction で保存される。ユーザーが指定した最低方向（AlphaEX > 0、MaxDDdelta < 0）は満たしたが、登録済みの preferred 目標 `AlphaEX >= +3pt` かつ `MaxDDdelta <= -3pt` を、2倍コストを含めて同時には満たしていない。したがって、これは現時点の「最強候補」であって、最終モデル確定・投資性能の証明・live deploy の承認ではない。

## 固定した方法

特徴行は15分足の完成バーを1本シフトし、判断時点より未来の値を使わない。各時点の因果特徴を次のように使う。

```text
m_h       = log-price momentum for h in {7, 30, 90} days
z_h       = m_h / (vol_7 * sqrt(h / 365) + 1e-6)
consensus = mean(tanh(z_h / 0.50))
slow      = 1.12 if m_90 >= 0 else 0.50
tactical  = 1.12 if consensus >= 0.15
            0.50 if consensus <= -0.15
            1.00 otherwise
target    = 0.50 * slow + 0.50 * tactical
```

target は UTC の00/06/12/18時だけ発行し、最大変更幅0.08、deadband0.01、片道コスト5.5bps、借入年率10%で約定する。欠損した3つの momentum または volatility のどれかがあれば、その行は fail-closed で判断を出さない。ストレス条件は片道コストと借入年率をともに2倍にした。

データは checksum 検証済みの BTCUSDT Spot 15分足で、全グリッド303,840行中300,240行が利用可能である。2026-08の未取得尾部は使わず、fresh fold は2026-04-16 13:45 UTCから2026-07-16 13:45 UTCまでに固定した。欠損の数値補完やゼロ埋めは行っていない。

## 窓別結果

以下は通常コストの確認結果で、数値は percentage points である。

| fold | 期間（UTC） | AlphaEX | MaxDDdelta | AlphaEX stress | MaxDDdelta stress |
|---:|---|---:|---:|---:|---:|
| 15 | 2024-01-16 – 2024-04-16 | +4.272 | +1.883 | +3.786 | +1.961 |
| 16 | 2024-04-16 – 2024-07-16 | +2.305 | -7.241 | +1.780 | -7.070 |
| 17 | 2024-07-16 – 2024-10-16 | -2.179 | -6.901 | -2.621 | -6.856 |
| 18 | 2024-10-16 – 2025-01-16 | +1.896 | +1.736 | +1.203 | +1.855 |
| 19 | 2025-01-16 – 2025-04-16 | +1.669 | -5.201 | +1.346 | -4.965 |
| 20 | 2025-04-16 – 2025-07-16 | -4.761 | +0.298 | -5.282 | +0.413 |
| 21 | 2025-07-16 – 2025-10-16 | -1.295 | +1.159 | -1.732 | +1.174 |
| 22 | 2025-10-16 – 2026-01-16 | +4.266 | -12.432 | +4.064 | -12.414 |
| 23 | 2026-01-16 – 2026-04-16 | +8.617 | -14.955 | +8.398 | -14.905 |
| 24 fresh | 2026-04-16 – 2026-07-16 | +1.504 | -5.861 | +1.188 | -5.739 |

fold 15、18ではDDが悪化し、fold 20、21ではAlphaEXとDDの両方が悪化している。このため「どのトレンドでも保証できる」とは言えない。10窓の平均が改善していることと、トレンド非依存性が証明されたことは別である。

## 文献調査との対応

検索では、長期時系列の局所パッチを扱う [PatchTST](https://arxiv.org/abs/2211.14730)、変数をtokenとして扱う [iTransformer](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf)、複数スケールを混ぜる [TimeMixer](https://arxiv.org/abs/2405.14616)、および分布シフト下の時系列カバレッジを扱う [ACI](https://proceedings.mlr.press/v162/zaffran22a.html) を確認した。BTCの現データで最初に採用したのは、これらの考え方のうち、学習の不安定性を増やさずに実装できる「複数時間軸の合意」と「更新頻度を落とす」部分である。

オフラインRLも [CQL](https://papers.neurips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) と [IQL](https://openreview.net/pdf?id=68n2s9ZJWF8) を確認したが、現時点のBTCデータでは、まず特徴契約・欠損・約定・B&H-relative評価を満たすこの固定オーバーレイの方が、学習モデルを追加するより監査可能性が高い。文献の一般的な優位性をBTCで再現したという意味ではない。

## 再現性と限界

- 登録、データ証跡、policy source hash、feature source hash を `codex_outputs/robust_overlay_v1/registration.json` に固定した。
- 固定した実行結果は [registration](../../codex_outputs/robust_overlay_v1/registration.json)、[development](../../codex_outputs/robust_overlay_v1/development.json)、[historical](../../codex_outputs/robust_overlay_v1/historical.json)、[fresh](../../codex_outputs/robust_overlay_v1/fresh.json)、[qualification](../../codex_outputs/robust_overlay_v1/qualification.json) で確認できる。
- development → historical → fresh の順序を強制し、confirmation の結果でパラメータを変更できない selection lock を作った。
- `384 passed, 27 warnings, 223 subtests passed`。警告は既存の timezone 変換と sklearn の境界テスト由来で、新規テストは通過した。
- ただし、候補の発想と6時間 cadence の探索では確認期間を先に観察したため、今回の結果は完全な preregistered untouched test ではない。保存した qualification はこの制約を明記しており、正式P1結果ではない。
- AlphaEXの記述的 fold bootstrap CI は通常コストで `[-0.547pt, +3.825pt]`、2倍コストで `[-0.993pt, +3.467pt]` と0を跨ぐ。DD改善のCIは両条件で負だが、fold間独立を仮定した記述値にすぎず、選択調整や時系列依存は補正していない。

次に必要なのは、未観測の将来期間を使った固定パラメータのforward paper test、トレンド方向・ボラティリティ状態別の成績分解、予測区間のcoverage監視、そして同じ候補を実際の `/predict` 入力契約に載せた runtime parity 検証である。これらが終わるまで、HFへのBTCモデル昇格や投資判断への利用は行わない。
