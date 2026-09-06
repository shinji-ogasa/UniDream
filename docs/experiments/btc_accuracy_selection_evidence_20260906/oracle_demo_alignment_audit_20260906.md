# BTC研究モデルをHFとWebへ揃えるための実装監査

2026-09-06。読み取りのみ。候補選定、repo編集、commit、deploy、DB変更、注文は行っていない。
対象は今回のBTC15分研究。BNB `/v2` は別作業として保持する。ユーザーの最低採用条件は
平均AlphaEX正・平均MaxDDdelta負へ更新されたが、採用モデル・固定artifactは親の選定待ち。
採用の妥協点と全相場頑健性の未達は表示を分ける。hindsight Oracleは配備候補に含めない。

## 配備実態はコードの最新版と一致していない

| 対象 | 現在確認した状態 |
| --- | --- |
| HF repo | `/Users/sophie/Documents/UniDream/unidream-space` main `14e5df6c30db03c5ea115f2e3030ff3ccf8beda3`。clean、cached upstream0/0、live ls-remote mainも同一 |
| HF配備 | `https://huggingface.co/spaces/ShinjiAA/unidream-space`。Hub APIで同SHA・RUNNING・cpu-basic・1replica。host `https://shinjiaa-unidream-space.hf.space` |
| HF BTC | `/health` HTTP200。Plan011v31 AC fold23、旧17特徴。bundle digest `c61d58fdd182e574a357d2cab64f40958f0d377fc96271eb5bc58e8b5cf91a56` |
| HF BNB | `/v2/health` HTTP200。`bnb-trend90-20260905`、決定的90日trend。BTCとは別モデル・別API |
| Web repo | `/Users/sophie/Documents/UniDream/unidream-demo-web` main `48f39dc87e21efc51399fe660f5e4b465c4636f9`。clean、cached upstream0/0、live mainも同一 |
| Web配備 | `https://unidream-demo-web.vercel.app`、production `dpl_98pNrjDHLQQXJ4FoQBUFbJVbfcpg` READY。Git SHAは同じ48f39dc。project `prj_27WjR6pQZ7Jb0QtztDKZnyvkLa3j`、team `team_MjPTkIufdU8vKG29EOXZ45Bn` |
| Supabase | `kgioxjzhsauxwbsrsvxp`。稼働Edge `run-unidream-inference`はversion2、verify_jwt=false、bundle digest `fcc0cff651951887d48f6b7595b0ca76fb42dd8d0af124a39229c6c6b83ccff1` |
| Supabase実行 | Cron `run-unidream-inference-15m` active、毎時1/16/31/46分。最新predictionとstateはともに2026-08-29 16:00UTCで停止。これ以降の新規予測は確認できない |
| DB配備差 | live `predictions.run_id`が存在せず、`public.record_unidream_inference` RPCも存在しない。repoのmigration0003の状態ではない |

HFには別worktree `.worktrees/unidream-space-alpha-trend-20260905`、branch
`exp/alpha-dd-bnb-runtime-20260905`、HEAD37f91e03が存在する。そちらやBNBbundleを変更しない。
Webの登録worktreeはmainのみ。着手・commit前に両repoの状態を再確認し、他作業の変更を混ぜない。
HF originはHF Space、Web originはGitHub `shinji-ogasa/unidream-demo-web`。VercelはGit deployment。

## コードと本番の違い

本番Edgeは単一11966文字のindex.tsで、TARGET_BARS=1800、OHLCVのみ、FEE_RATE=0、
直近closeで即時fill、predictions INSERT → state UPSERT → snapshot UPSERT → trades INSERTを別々に実行。
原子的RPC、quote/taker入力、funding/mark入力は使っていない。これは現repoの分割実装より古い。
HF BTCは現在funding/mark必須なので旧payloadは必要条件を欠く。停止原因との整合性は高いが、
この監査ではEdge失敗ログを取得しておらず、停止の直接原因とは断定しない。

現Web repoは7248本、closed-bar検証とfunding/mark結合、片道合計5.5bps、CAS付きRPCを実装済み。
ただしそれでも今回の研究コントローラとは以下の差がある。

| 契約 | 今回の研究 | 現Web repo／HF BTC |
| --- | --- | --- |
| 特徴 | Technical29、必要時Perp31/短期37。Spot OHLC＋quote_volume＋taker_buy_quote、Perpなら対応UM raw | Plan011のOHLCV＋funding/mark由来17列。schemas/Candleがquote/takerを受け取らない |
| 履歴 | 最長90日momentumを含む。閉じた8641足相当の経過履歴＋判断行、15分フルグリッド、名目窓と.995条件を保持 | 7248本の旧60日＋warmup。研究の90日履歴に不足 |
| 判断時刻 | UTC6時間、特徴はt−1まで。現在open[t]と自己cash/unitsだけで判断 | 15分ごと。推論に自己在庫を渡さず、最後の特徴列のpositionを採用 |
| 約定 | tで提出、即時次足t+1のopenだけ。欠損なら失効、先送りしない | 推論応答後にlatest.closeで即時fill（atomic.ts98） |
| 状態 | 自己cash/unitsから現在openで再評価したexposure。自然変動は範囲外も許容 | current_positionは前回targetを保存。実exposureの変化と異なる |
| 目標・変更 | intent .5〜1.12、maxstep .08、deadband .01、holdはNaN。欠損規則を固定 | 0〜1.12へclamp、maxstep/deadbandなし。無効positionを0へ変換（atomic.ts96–98） |
| 費用 | 正確な実trade notionalへ.00055、negative cashに年率.10借入、欠損経過も処理 | 合計率.00055は一致するが、前回targetとの差×NAVで費用計算。借入なし |
| 初期条件 | B&H同時保有から開始。比較用基準capitalと同一時刻 | cash10000、units0、position0。Web B&Hは独自flat→1 entry方式 |
| マーク・集計 | canonical15分close系列、initial capitalを含めたDD・同一B&H | snapshotsの最初のequityから窓return、B&H独自初回fee。欠損bar時の金利会計なし |

根拠: Web `_shared/config.ts:1–31`、`paper_trading.ts:83–123`、`atomic.ts:89–131`、
`index.ts:39–96`、HF `schemas.py:6–30`、`runtime.py:12–29`。
現HF runtimeはbundle_type `plan011_v31_overlay_actor`しか受け付けない。
単にjoblibをbundles/currentへ置き換えるだけでは起動しない。requirementsは非固定下限でsklearn/joblibなし。
live healthのNumPy2.4.6/Python3.11.16は、研究NumPy2.2.6/pandas2.3.3/sklearn1.8.0/Python3.12とも異なる。

## 最短で正しく揃える実装経路

1. 親が採用するBTC候補を一つ固定し、モデル／risk／normalizer／bias／anchor／shrinkage等、
   必要なすべてのartifactとsource hashを一つのversioned manifestへまとめる。
   foldやfit終了時刻、学習後の固定手順、feature順、decision/fill/fee/borrow/missing rule、
   baseとstressの平均成績、相場別失敗、探索再利用を記録する。古いfoldモデルを現在に置くなら
   その鮮度を明記し、無断で最新データへrefitしない。rootの予算内選定を待ち、候補をこちらで変更しない。
2. HFに独立したBTC predictorと`/v3/btc/{health,predict,...}`を追加する案が安全。
   BNB `/v2`と旧BTC routeは保持し、Webは新routeを明示する。旧defaultを後で切り替える場合も
   無在庫の旧requestを新policyとして処理しない。新bodyはdecision_ts、完全なraw Spot/必要UM履歴、
   current_openとknown_at、自己cash/unitsとstate_versionを持つ。予測mu/riskとdecision traceを返し、
   intentとfilled exposureを分ける。holdはnull、fallbackは登録された条件のtarget1。
   canonical featureと選定policyを必要な依存だけvendorし、主要依存バージョンを固定する。
3. raw collectorはSpot quote/takerも保持し、Perp候補ならUM trade kline quote/takerを追加。
   旧funding/mark取得は新familyに不要なら外す。current openは未完成足からopenだけを別入力にし、
   close/high/low/volumeを現在のfeatureに混ぜない。足close時刻だけで受信済みとはせず、受信時刻・
   raw digest・判断deadlineを記録する。archive warmupを当時のreceipt証拠とは扱わない。
4. Web/Edgeのペーパー会計をcanonical cash/unitsの実装へ差し替える。15分処理は維持し、
   判断だけ6時間。pending intentをtで保存し、t+1openで最大変更・deadband・費用を計算して消費。
   遅延到着や欠損による失効と未判断は分離し、過去のopenへ遡って新注文を約定させない。
   closeマークと借入を順序通り処理し、B&Hも同じ初期state・時計で並走する。
   当初の既存在庫entryは研究条件に沿う仮想初期状態と明示し、実取引の無料購入を装わない。
5. DBは旧履歴を消さず新run_idを作る。pending order、model/feature/execution digest、
   forecast availability、decision_at/received_at/fill_due_at、borrow/fee、benchmark stateを保存する。
   新しい原子的RPCはrow lock/CASと(run_id,event timestamp/type)冪等性でprediction/intent/fill/stateを
   整合させる。migration0003は本番未適用なので前提にしない。新migrationを作成し、既存重複を
   read-only検査してから適用する。secretをclientへ出さず、公開表write/RPCはservice roleに限定する。
6. UIのRUN_IDだけでなくSSR最新predictionとRealtimeにもrun_idを追加する。
   現在predictionはsymbol/timeframeだけで絞るため、旧BTCと新BTCが混在する。
   `types.ts`、`contract.ts`、dashboardRepository、useLiveDashboard、metrics.tsを同じmanifestへ揃える。
   表示名、モデルdigest、学習時刻、受信鮮度、未約定intent、実exposure、借入、missing/fallbackを表示。
   開発平均AlphaEX+/DD−と将来保証を区別し、現在の一続きlive成績を8fold平均と混同しない。

## 更新前後の最低検証

- 完成済み研究fixtureで、features→mu/risk→意図→cash/units→NAV/B&H/DDのparity。
  単なる予測ベクトル一致に留めず、借入、gap、自然drift、no-trade、fallbackの会計も含める。
- 未完成足・未来close・欠損quote/taker・不正モデルhash・古いstate・期限切れintentを拒否。
- 新RPCの同時実行/CAS/再送とevent二重fill防止、DB新run隔離、SSR/Realtime両方のrun絞込み。
- HF新health/fixture verifyと実provider readiness、Edge新version/RPC/schema、Vercel git SHA、
  新runのprediction/intent/fill/state timestampと全digestを同じ配備として確認する。
- Web typecheck/build、HF/RPC会計tests、desktop/mobile実画面QA。旧BNB healthも維持確認。

## 監査証拠・読取範囲

- `/tmp/oracle_demo_live_health_20260906.json`: public HF両healthとHub running revision、response SHA。
- `/tmp/oracle_demo_deployed_edge_summary_20260906.json`: Edgeversion2・bundle SHA・source SHA・安全な抜粋。
- Vercel get_deployment/inspect: production Ready、aliases、Git SHA48f39dc。
- Supabase read-only SQL: active Cron、predictions schema、RPC不在、最後のmodel/state timestamp。
  queryは価格・新しい特徴／returnを取得せず、secret設定／Cron command本文を出力していない。
- HF/Next/Supabase/VercelCLI skillsを読み、[Supabase changelog](https://supabase.com/changelog.md)を取得。
  `/tmp/oracle_demo_supabase_changelog_20260906.md`へ保存。実装時の新RPC/API詳細は
  [公式RPC](https://supabase.com/docs/reference/javascript/rpc)／[Edge Functions](https://supabase.com/docs/guides/functions)／
  [RLS](https://supabase.com/docs/guides/database/postgres/row-level-security)で再確認する。

この監査はデプロイを変更していない。健康なHFの起動と、停止したSupabaseジョブの実用可否を分けて報告した。
