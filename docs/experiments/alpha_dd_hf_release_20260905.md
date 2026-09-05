# Alpha/DD minimum-goal result and HF release verification

確認日時: 2026-09-05 07:51:33 UTC（16:51:33 JST）。

## 結論

BNBUSDT の固定90日トレンドルールは、10評価窓の平均で B&H 比
**AlphaEx +1.356381pt / MaxDD delta −6.251727pt** となった。
ユーザー指定の最低条件（+1pt以上かつ−1pt以下）は満たすが、希望条件
（+3pt以上かつ−3pt以下）は AlphaEx 不足で満たさない。
これはニューラル WM→BC→AC の改善ではなく、別の決定的ルール方式の結果である。

同じ固定ルールを HF の新しい `/v2` API に配備し、公開環境で研究parity、
実際の最新Spotデータ、入力拒否、既存BTCの基本互換を検証した。
配備・実データ推論は確認済みである。実売買、ライブ運用成績、統計的に頑健な
優位性、正式P1結果を主張するものではない。

## 目標ごとの証拠

| 要件 | 確認結果 | 保存した証拠 |
|---|---|---|
| B&H比の平均 AlphaEx / MaxDD delta を同時に検証 | 最低条件のみ pass | [研究結果](alpha_dd_results_20260905.md)、[BNB qualification](alpha_dd_evidence_20260905/bnb/qualification.json) |
| 確認結果を見る前に手法を固定 | `trend_d90_lo0_hi1.12_p0`、BNB cross-asset lock | [選択lock](alpha_dd_evidence_20260905/bnb/selection_lock.json)、[asset lock](alpha_dd_evidence_20260905/cross_asset_lock.json) |
| 選択した手法をHFへ更新 | `bnb-trend90-20260905`、決定的ルール、新 `/v2` API | [HF revision](https://huggingface.co/spaces/ShinjiAA/unidream-space/tree/14e5df6c30db03c5ea115f2e3030ff3ccf8beda3) |
| 配備バージョンと実行バージョンの一致 | 検証前後とも同じSHA、`RUNNING`、`cpu-basic` | [公開検証JSON](alpha_dd_hf_release_20260905.json) |
| 研究実装との一致 | 20 real-history cases、signal/target 最大差0、両ポジション目標0/1.12を含む | 同JSON `/v2/sample/verify` と2回の `/v2/predict` |
| 最新データで正常推論 | 8,641本のclosed Spot bars、欠損0、公式Spotの90日両端closeから独立再計算して一致 | 同JSON `/v2/predict/latest`、独立 endpoint closes / momentum |
| 誤入力を受理しない | BTC symbol 422、未来decision 400 | 同JSON、期待した拒否応答 |
| 既存APIの基本互換 | BTC health / sample / feature POSTは200、position厳密parity pass | 同JSON。補助出力の限定は下記 |

## 公開検証結果

- Space: https://shinjiaa-unidream-space.hf.space
- 新モデルのhealth: https://shinjiaa-unidream-space.hf.space/v2/health
- 新モデルのlatest: https://shinjiaa-unidream-space.hf.space/v2/predict/latest
- HF commit: `14e5df6c30db03c5ea115f2e3030ff3ccf8beda3`
- Canonical bundle digest: `43a8613e7c1cdb1c67fdadd8c9ed9a141cf49cfbdf17fc672a2f0784cf157b9c`
- Evidence JSON SHA-256: `7f7f9f7cece9dba80946d8979d69612eef005172c3a01995561b67f5be140ed8`
- 10 HTTP checksを実行。正常系は全て200、拒否系2件は期待通り422/400。
- 新BNB parityの公開応答時間は3.37秒、2つの実履歴POSTは1.32秒/0.56秒。
- latestのdecisionは `2026-09-05T07:45:00Z`、collector観測は `07:50:54Z`。
  独立取得した両端closeは `588.88` / `737.44`、log momentumは
  `0.22496230122174765`、返却目標は `1.12`。
- このlatestサンプルは毎時00分の意思決定時刻ではないため
  `decision_eligible=false` / `status=not_scheduled`。目標値を計算できることと
  売買指示・約定可能性は別であり、`execution_ready=false`、注文は0件。
- root discoveryも `primary_research_model=bnb-trend90-20260905` と
  `/v2/predict/latest` を示す。従来の `/predict` をBNBへ黙って変更していない。

## HFビルド復旧

`37f91e0` は同じルールのtimestamp parse高速化であり、ローカル48 testsが成功。
HFではその更新と、コード変更なしの再試行 `14e5df6` が一度 `BUILD_ERROR`
（exit 128）になった。認証付き旧ログはBuild Queuedの1行だけで、根本原因は
特定できなかった。従って、コード不良やGit破損だったとは断定しない。

同じ `14e5df6`、同じ無料 `cpu-basic` のまま、公式
[restart_space(factory_reboot=True)](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.restart_space)
相当のAPIを1回実行した。07:48:29 UTCに新しいビルドが開始し、依存インストール、
COPY、image pushが成功して `RUNNING` に移行した。公開smokeの前後でその状態と
実行SHAを再確認した。ハードウェア課金、認証Secret、研究ルールの変更はない。

## 残る限定・主張しないこと

1. BNBはAlphaEx positive 4/10窓、median AlphaEx −2.067762pt。
   historical 9窓だけでは +0.773879pt、コスト/借入2倍stressでは +0.677865ptで
   最低条件に届かない。平均の成立を安定した収益性や実運用推奨へ読み替えない。
2. 両指標のdescriptive intervalは0を含む。逐次実験で確認窓を再利用しており、
   untouched holdoutや探索回数補正済みの有意性ではない。
3. 研究テスト381件、HFローカルテスト48件が成功したが、それは経済的優位性の
   証明ではない。研究381件は最終コードで再実行し64.679秒、HF48件は4.345秒。
4. BTCのposition strict最大差は `1.1920928955078125e-7`（許容 `1e-6` 内）。
   一方、補助advantage最大差は `1.531839370727539e-5` でdiagnostic許容 `2e-5`
   内だが、`live_default_advantage_strict_ok=false`。全内部出力の厳密一致とは
   言わない。新BNBのsignal/target差0とは独立した結果である。
5. 旧BTC `/predict/latest` のderivatives provider経路は今回の公開検証対象外。
   その以前からの地域制限障害をこのリリースで修復したとは主張しない。
   新BNB latestはSpotのみを使い、実際のHF上で正常応答を確認した。
6. 公開APIは既存のread-only公開設定。APIは口座状態を検証せず、注文を出さない。
   Supabase/demo/実取引クライアントを新銘柄へ変更していない。

## 再検証コマンド

既存の証拠JSONを上書きしない。再検証では新しい出力名を指定する。

```bash
python -m unidream.experiments.verify_alpha_trend_space \
  --revision 14e5df6c30db03c5ea115f2e3030ff3ccf8beda3 \
  --bundle /absolute/path/to/unidream-space/bundles/alpha_trend \
  --output /absolute/path/to/new-evidence.json
```

公開検証スクリプト: [verify_alpha_trend_space.py](../../unidream/experiments/verify_alpha_trend_space.py)。
JSONはリクエスト/レスポンスhash、応答body、latency、runtime前後を保持する。
認証情報は保存しない。
