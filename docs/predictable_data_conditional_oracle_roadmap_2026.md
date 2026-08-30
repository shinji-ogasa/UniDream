# Predictable Data → Conditional Oracle research roadmap (2026)

**Document state:** research redesign and execution contract; no new model result

**Evidence cutoff:** 2026-08-30

**Scope:** BTCUSDT 15m first, then multi-asset external validation

**Selection contract:** train/validation select; development test is report-only; a new untouched holdout is opened once

The live pass/fail and blocked-data ledger is [`experiments/predictable_oracle_hypothesis_registry.md`](experiments/predictable_oracle_hypothesis_registry.md). A hypothesis absent from that registry is not part of the current candidate family until it is added before results are opened.

## Executive decision

UniDream の次の研究順序を、次のように固定する。

~~~text
point-in-time data contract
  -> future outcome predictability
  -> cross-fitted conditional outcome distribution
  -> feasible conditional Oracle
  -> direct executable policy
  -> optional student distillation
  -> optional decision-focused / AC improvement
  -> statistical promotion
  -> new untouched holdout
~~~

最重要の変更は、実現した未来を見て作る action を学習教師から外すことにある。

| name | future realized path | decision-time available | role |
| --- | --- | --- | --- |
| `hindsight_upper_bound` | uses it | no | opportunity-set upper bound and regret diagnostic only |
| `conditional_oracle` | does not use it directly; uses an OOF conditional forecast | yes | best feasible decision supported by observable information |
| `policy_student` | no | yes | optional approximation of the conditional Oracle |

`hindsight_upper_bound` が非常に強くても、`conditional_oracle` が B&H / hold を上回らなければ、予測可能な edge は確認できていない。その場合は BC、WM、AC の容量を増やさず、データまたは horizon を見直す。

同様に、`conditional_oracle` は入力と optimizer から決定的に再計算できるため、action 模倣精度だけなら簡単に高くできる。重要なのは Oracle action の分類精度ではなく、Oracle の元になる将来 outcome の OOS 予測力と、コスト控除後の decision regret である。

## What is wrong with the current path

### Current information is not empty, but it is incomplete

現行の canonical 17 features は OHLCV 由来13列に、funding と basis 系4列を加えたもの。ATR と RV-4/16/96 があるので、ボラティリティ関連情報がモデルに全く入っていないわけではない。一方で、現行 Plan011 は `include_oi: false` で、次は未収録である。

- Spot / perpetual の signed trade flow と taker imbalance
- trade intensity、trade-size distribution、spot–perp flow divergence
- open interest、OI change、crowding / long-short state
- best bid/ask、spread、depth、order-flow imbalance
- liquidation flow
- ETH などの cross-asset state

現在の17列と availability 契約は [`cache_v4.py`](../unidream/data/cache_v4.py) にあり、Plan011 設定は [`plan011_overlay_actor_v31_relative_constraint_ac.yaml`](../configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml) にある。

### The current “Oracle” is not one coherent, learnable teacher

現行実装には複数の future-derived teacher が順番に存在する。

1. [`hindsight_oracle_dp`](../unidream/data/oracle.py) は split 内の実現済み将来リターンを後ろ向きに走査する。
2. `signal_aim` teacher は `t+1` 以降の複数 horizon の future return と future volatility を使う。
3. transition-advantage relabel は future return、volatility、drawdown、cost から action utility を作る。
4. 現行 `lowfreq_wm_overlay` は最終的に WM predictive state と causal trailing trend から target を作る。`positions` 配列はこの mode では長さにしか使われない。

したがって「完全な未来 Oracle を actor がそのまま模倣している」とだけ説明するのも、「Oracle が現在情報から最適化されている」と説明するのも正確ではない。教師定義が多段に上書きされ、どの情報が最終 target を決めたかが不透明になっている。

さらに、次の3経路は新設計の P0 blocker である。

- WM は train 全体の future labels で fit された後、同じ train rows の predictive state を `lowfreq_wm_overlay` teacher に渡す。test leak ではないが in-sample future-target fit なので、chronological OOF conditional teacher ではない。
- transition relabel は hindsight teacher path の前 position を current inventory として cost/advantage 計算へ戻す。Oracle が自分の future-derived answer を state として参照する循環になる。
- BC/AC supervised branch は前時点の teacher position を controller inventory に入れる一方、runtime は前時点の policy position を使う。これは teacher forcing による train/runtime state-distribution mismatch である。

現行の timing も統一されていない。config の forecast diagnostic は one-bar delay だが、legacy DP / transition label は same-bar return を使う経路があり、validation/test Backtest の一部は default delay 0 で呼ばれる。新結果を出す前に `decision t -> fill t+1 -> earn return t+1` を target、teacher、upper bound、validation、test の全てで一つの shared evaluator に強制する。

action feasibility も単一契約ではない。legacy Oracle/transition grid、WM position-utility grid、AC max-step、transition step、複数の min-hold post-process が別々に存在する。新 conditional Oracle はこれらを同時利用せず、旧 action/utility head を candidate path から外して一つの source of truth に置き換える。

### Existing evidence points to a target/data problem before a capacity problem

[`alpha_attribution_plan011_v31_dev/report.md`](alpha_attribution_plan011_v31_dev/report.md) の historical fold23 diagnostic には数値が記録されているが、全てを精度 evidence と読めるわけではない。

- horizon 4–32 の return sign accuracy は概ね 46–50%、Spearman IC はおよそ 0.06–0.07
- horizon 4–32 の volatility rank IC はおよそ 0.41–0.53
- horizon 4–32 の drawdown rank IC はおよそ 0.16
- crash balanced accuracy は概ね 0.50
- h64 と position-utility-h64 は `seq_len=64` と future-only target builder の組合せで現行実装上の `valid_len=0` となり、有効 target/gradient がない

したがって、報告済み h64 metrics と position-utility argmax `0.23%` は「予測が難しい」という性能証拠には採用しない。まず各 head/fold の target count、finite mask、loss weight、nonzero-gradient steps を artifact 化し、future-only horizon では少なくとも `seq_len >= max_horizon + execution_delay` を満たすか、その head を無効化してから再評価する。数式上の長さだけで pass にせず、実測の nonzero target/gradient coverage も必須とする。

また development 13 folds の actor sequence 平均 AlphaEx は `+0.414pt`、actor-mean constant は `+0.391pt` で、観測された timing increment は小さい。shifted/null path も近い。この historical evidence は corrected-v4 の結果ではないため新精度の推定には使えないが、「まず model を深くする」よりも「予測可能な outcome と情報集合を確定する」優先度が高いことを示す診断である。

現在の v4 data-quality audit には availability sidecar を training sequence eligibility へ伝播できていない停止条件も残る。よって新ロードマップの最初の実データ accuracy run は、そこを修復するまで開始しない。

## Fixed mathematical contract

### Information set

時点 `t` の production information set を次で固定する。

\[
\mathcal F_t =
\sigma(
X_{\le t}^{\mathrm{available}},
p_{t-1},
h_{t-1},
e_t
)
\]

ここで、

- \(X_{\le t}^{\mathrm{available}}\): `decision_ts` までに利用可能になった market data
- \(p_{t-1}\): 実際の policy inventory
- \(h_{t-1}\): commitment countdown / prior transition state
- \(e_t\): execution venue、cost、latency、funding-clock state

各外部値は最低限 `event_ts`、保守的な `exchange_available_ts`、live collector がある場合の `collector_observed_ts`、archive の `archive_published_ts`、local `download_ts`、`decision_ts` を区別する。ここでいう `causal` は因果効果の識別ではなく、「decision cutoff より未来の情報を使わない」という point-in-time causality を意味する。

\(p_{t-1}\) は benchmark または候補 policy を先頭から replay して得た在庫であり、hindsight/teacher path から取得しない。primary comparison は B&H-relative overlay として全候補、B&H、upper bound を `p_start=1.0` から始め、benchmark の取得は評価窓より前の共通 sunk state とする。secondary cash-start stress は全方式を `p_start=0` から始めて同じ entry cost を課すが、selection には使わない。initial position を未指定の Backtest default に任せない。

### Outcome model

action を直接予測する前に、将来 path の条件付き分布を chronological OOF で推定する。

\[
\widehat P_t =
\widehat P_\theta
\left(
Y_{t+d:t+d+H-1}
\mid \mathcal F_t
\right)
\]

\(d\) は execution delay。最初の target family は次に限定する。

- cumulative excess return: horizons 4, 8, 16, 32 bars
- realized volatility
- maximum adverse excursion / path drawdown
- downside quantiles / crash probability

horizon 64 は「弱かった」のではなく、現行 sequence contract では未学習の可能性があるため最初の candidate から外す。再導入には `seq_len >= horizon + execution_delay` と target/gradient coverage gate を要求する。

P2 では horizons 4, 8, 16, 32 の outcome predictability を比較するが、最初の P3 action decision は `H_decision=4` だけを使う。8–32 bars は calibration / risk diagnostic に留め、action objective へ加える場合は commitment horizon、target、trial family を別 candidate として事前登録する。

train 行の OOF prediction は、必ずその行より前の label-complete data だけで fit する。これには raw outcome head だけでなく、WM latent/predictive state、Q、standardization、calibration、teacher sample weight など future-derived target から学習される全量を含む。legacy in-sample WM action/utility headsは P2/P3 の入力にしない。

overlapping horizon は purge し、random K-fold は使わない。各 outer fold の中を `model-fit train -> early-stop inner-val -> disjoint calibration -> policy outer-val -> outer test` に分ける。feature group、model、horizon、utility weight、abstention threshold は policy outer-val までで固定する。outer test は report-only であり、次の候補選択へ戻さない。WM early stopping と policy selection に同じ validation rows を再利用しない。十分な過去を持たない early rows は teacher から除外し、in-sample prediction で埋めない。

return、volatility、drawdown、crash の別々の marginal heads だけでは、path-dependent DD/CVaR を含む joint utility は決まらない。O2/O3 は chronological OOF の joint return-path scenarios、明示した dependence/coupling model、または保守的な worst-case bound のいずれかを使う。周辺 quantile を独立と仮定して合成しない。joint calibration、tail coverage、scenario provenance を P3 の必須 artifact とする。

### Conditional Oracle

conditional Oracle の理論上の target は、真の条件付き分布の下で観測情報から選べる Bayes-optimal action である。

\[
\pi_t^\star
=
\arg\max_{\pi\in\Pi_{\mathrm{feasible}}(s_t)}
\mathbb E_{P(Y\mid\mathcal F_t)}
\left[
U(\pi,Y;s_t,c)
\right]
\]

真の \(P(Y\mid\mathcal F_t)\) は未知なので、実装できるのは chronological OOF で推定した次の policy である。

\[
\pi_t^{\mathrm{cond}}
=
\arg\max_{\pi\in\Pi_{\mathrm{feasible}}(s_t)}
\mathbb E_{\widehat P_t}
\left[
U(\pi,Y;s_t,c)
\mid \mathcal F_t
\right]
\]

したがって `conditional_oracle` は「完全な正解」ではなく、固定した data、model class、utility、constraints の下での `estimated conditional decision benchmark` である。forecast model hash と uncertainty を必ず保存し、Oracle model risk を隠さない。

最初の実装は複雑な RL ではなく、少数 action / scenario の exhaustive optimizer とする。実行可能集合には次を optimizer 内部で直接入れる。

- first-pass spot allocation range `[0.50, 1.00]`; leverage and shorting are excluded
- current position からの最大変化 `0.08`
- hold/no-change action
- one-bar execution delay: `decision t -> fill t+1 -> return t+1`
- `min_hold_bars=4`; post-processing ではなく commitment state
- spread `3 bps`、slippage `1 bp`、fee `0.0003` の Plan011-reference proposed contract
- B&H-relative net utility
- volatility、drawdown、tail-loss penalty

候補 next-position は、最初は

\[
\mathrm{clip}
\left(
p_{t-1}+\{-0.08,-0.04,0,0.04,0.08\},
0.50,1.00
\right)
\]

とする。これは新 source-of-truth の提案であり、legacy Oracle/transition grid `[0, 0.5, 1, 1.25]`、WM utility grid `[0.5, 0.7, 0.85, 0.94, 1, 1.06, 1.12]`、`0.10` と `0.08` の step 実装とは併用しない。旧 utility/action heads を明示的に candidate path から外し、optimizer、teacher、student、Backtest が同じ action contract を import する。

最初の optimizer は `H_decision = min_hold_bars = 4` の block-commitment rule とする。decision `t` で選んだ `a` は `t+1` に全量 fill し、returns `t+1 ... t+4` の4本を通して固定する。次の decision は `t+4` の close までの情報だけで行い、fill は `t+5`。block 内の再最適化や未知の将来 feature を使う open-loop action sequence は作らない。したがって forecast Q、U0/O1–O3 の counterfactual path、teacher、student、Backtest は全て同じ4-bar commitment trajectory を評価する。将来 action sequenceや1-barごとの receding-horizon policy は、continuation policy / terminal value を定義した別 candidate とする。

state は `position` と `commitment_bars_remaining` を持つ。primary window は `p_start=1.0`、`commitment_bars_remaining=0` で開始し、最初の decision を許可する。decision では hold を含む選択後に4へ reset し、各 scored bar の完了後に1ずつ減らす。remaining が0より大きい間の feasible set は現在 position だけ。offline P0–P3 は deterministic all-or-none fill とし、feature unavailable / execution skip は hold を4 bars選んだものとして記録する。partial fill はこの研究 backtest では未対応であり、shadow execution 前に fill-state contract を追加する。末尾に4本の realized outcomeを持たない decision block は、U0/O1–O3 と全 benchmark から共通に除外する。

utility は point estimate だけでなく forecast scenarios / quantiles から計算する。基本形は次とする。

\[
U(a,Y)
=
(a-1)R_H
-c|a-p_{t-1}|
-\lambda_{\mathrm{vol}}|a-1|\sigma_H
-\lambda_{\mathrm{dd}}DD_H(a;Y)
-\lambda_{\mathrm{tail}}CVaR_H(a;Y)
\]

最初の P3 では `H=4`。`R_H` と strategy/relative-equity path は、上記4-bar commitment に同じ `a` を適用して作る。O1 は point forecast の expected excess return と transition cost だけを使い、`lambda_vol=lambda_dd=lambda_tail=0` とする。O2/O3 だけが calibrated joint scenarios から action-conditioned volatility、DD、CVaR を計算する。これにより、point forecast だけから path risk を捏造しない。\(\lambda\) は development の inner validation だけで選び、全 trial を ledger に残す。

`DD_H(a;Y)` と `CVaR_H(a;Y)` は action-conditioned strategy/relative-equity path 上で計算する。同じ market-only DD/CVaR を全 action から定数として引くだけの式は argmax に影響しないため禁止する。cost も同じ return unit へ変換し、entry、exit、fold boundary の課金規則を O0–O3/U0/B&H で共有する。

この proposed cost contract では `spread_bps=3` は full quoted spread とし、各 position change に half-spread `1.5 bps`、片道 slippage `1 bp`、片道 fee `3 bps` を課す。すなわち一回の transition cost は `0.00055 * |a-p|`、round trip は二回課金する。最初の実装は historical Backtest と同じ additive log-return approximation、`net_log = p * r_log - transition_cost`、`equity = exp(cumsum(net_log))` に固定し、exact-simple-return accounting は別 candidate とする。値が config に無い場合、legacy default `5/2/0.0004` へ fallback せず fail closed にする。

first-pass action は unlevered spot allocation `[0.50, 1.00]` なので、funding cash flow と borrow cost は primary P3 utility に入れない。funding、mark、basis は decision-time feature としては利用できるが、PnL cost として暗黙に混ぜない。perpetual / margin / leverage candidate を追加する場合は instrument、mark-to-market、funding timestamp と符号、borrow rateを先に固定し、O0–O3/U0/B&H全てへ同じ cash flow を適用する。この初期 cost contract は research comparison であり、そのまま production executable PnL の証拠ではない。

### Soft decisions and abstention

ensemble / bootstrap scenario ごとの utility から、

\[
q_t(a)=
\Pr\left(a=\arg\max_b U_t(b)\mid \mathcal F_t\right)
\]

と action margin

\[
m_t=\widehat Q_t(a^\star)-\widehat Q_t(\mathrm{hold})
\]

を保存する。`LCB(m_t) <= 0` のときは B&H-relative baseline または current position を維持する。threshold は validation で risk–coverage curve を見て一度だけ固定し、test の active rate を見て変更しない。

## Data program

### Three data planes

| plane | purpose | allowed claim |
| --- | --- | --- |
| known-DGP synthetic | optimizer、OOF、timing、constraint、student の実装検証 | implementation recovery only |
| semi-synthetic BTC | real volatility/gaps/costs 上で既知 signal の回収を検証 | pipeline sensitivity only |
| real market data | unknown signal の stable OOS predictability を検証 | research evidence after gates |

synthetic / semi-synthetic は収益 evidence ではない。zero-signal control で false positive が出ないことと、高-SNR signal が単純モデルで回収できることの両方を確認する。

### Real-data acquisition ladder

一度に大量の特徴を足さず、各 data group を同じ simple-model budget で追加し、前の group に対する OOS marginal value を測る。

| group | data and initial features | source / history reality | hypothesis | promotion gate |
| --- | --- | --- | --- | --- |
| D0 | corrected OHLCV13 vs current full17; RV/ATR/funding/basis | local BTC exists; corrected v4 availability path is incomplete | establish honest baseline and whether derivatives-4 add value | availability-aware same-row comparison passes |
| D1 | Spot and USD-M kline metadata / aggTrades: trade count, quote volume, taker imbalance, signed notional, intensity, size and impact proxies | Binance public daily/monthly archive provides Spot/Futures trades, aggTrades and klines with checksums | order flow may add short-horizon state not present in OHLCV close/volume | stable incremental proper-score and net-utility improvement on chronological validation |
| D2 | Spot–perp premium and flow divergence; mark/index/premium; funding clock | official archive/REST, but exact coverage and publication timing must be audited | basis and cross-market lead/lag may improve return/risk distribution | adds value after D1 on common eligible rows |
| D3 | OI, OI change, price×OI quadrant, long/short and taker ratios | official REST currently exposes only about one month / 30 days for several metrics; archive continuity must be audited | leverage/crowding state may improve crash/downside heads | no long-horizon run until point-in-time history and checksum ledger are complete |
| D4 | best bid/ask, spread, depth, OFI, queue replenishment/cancellation | full-depth history is not assumed available; start a forward WebSocket collector now or use a separately audited provider | microstructure may be useful at shorter horizons | update sequence reconstructs exactly; staleness/coverage gate passes; value survives 15m aggregation and delay |
| D5 | ETHUSDT first, then selected liquid assets / second venue | not local today; official archives can be acquired | pooled state and cross-asset lead/lag may improve stability and supply external validation | common-history and listing masks fixed; leave-one-asset/time results stable |
| D6 | options IV/DVOL/skew, macro release events, liquidation | mixed providers and shorter history | primarily volatility/tail-risk context, not assumed directional alpha | only after D0–D5; exact release/availability/vintage contract required |

Binance’s official public-data README documents Spot/Futures `aggTrades`, `trades` and kline fields including number of trades and taker-buy volume, with daily/monthly files and checksums. It also states that daily files are produced on the following day, monthly files on the first Monday of the following month, and archive files can be updated later. The current USDⓈ-M API documents `openInterestHist` as latest one month and basis/taker/long-short series as latest 30 days. Therefore “API endpoint/archive exists” is not sufficient evidence that a 2018–2026 point-in-time backfill or historical ingestion latency exists. See [Binance Public Data](https://github.com/binance/binance-public-data/blob/master/README.md) and [USDⓈ-M market-data catalog](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).

`aggTrades.m / isBuyerMaker` は trade direction そのものではない。`true` は buyer が maker、すなわち aggressor/taker は sell 側という意味として、raw fixture と公式 schema に対する単体テストで符号を固定する。Spot と USD-M の quantity/notional 単位も別々に正規化し、同じ列名へ暗黙に混ぜない。

Order-book research such as [DeepLOB](https://arxiv.org/abs/1808.03668) and pooled cross-stock order-flow work such as [Universal Features of Price Formation](https://arxiv.org/abs/1803.06917) support microstructure as a testable hypothesis, not as proof of BTC 15m profitability. Both are cash-equities evidence rather than multi-asset crypto evidence. A BTC-specific LOB study reports short-horizon predictability on Coinbase, but its seconds-scale setting does not transfer automatically to UniDream’s horizon; see [Deep Learning for Digital Asset Limit Order Books](https://arxiv.org/abs/2010.01241).

### Mandatory availability schema

Every raw observation and every derived feature group must record:

- provider, venue, symbol and endpoint/archive path
- source checksum and download/ingest timestamp
- `event_ts`, conservative live `exchange_available_ts`, live `collector_observed_ts` where available, `archive_published_ts`, local `download_ts`, `decision_ts`
- archive revision/version, previous checksum, replacement checksum and first-seen timestamp
- availability boolean, age, stale reason and gap reason
- aggregation window and whether the window is right-exclusive
- feature-code hash and source-data hash

これは future contract であり、現行達成事項ではない。legacy feature path は funding、basis、OI 系の unavailable/initial values を0へ埋める箇所があり、v3 には availability mask がない。v4 は unavailable を sidecar false と feature value zero で表せるが、その sidecar は現在 training sequence へ伝播せず full17 promotion が fail-closed になっている。

`exchange_available_ts` は local download 時刻や翌日/翌月の archive publication 時刻ではない。realtime exchange event semantics と保守的な observation/aggregation lagから別途マッピングする。archive は event value の backfill に使えても、当時の live availability や未改訂値を自動では証明しない。publication/version metadata が得られない過去行は availability certainty を `assumed` とし、lag/revision stress を必須にする。後日更新された同一 archive は旧・新 checksum の両方を source ledger に残し、黙って置換しない。

新 contract では future backfill を禁止し、missing を実測ゼロとして解釈しない。値本体が数値ゼロを必要とする場合でも、decision-time-known availability mask と age を必ず同時に消費し、required source が unavailable の window は明示的に除外する。全 feature groups は honest eligible rows と parent group との common-row subset の両方で評価する。forward collector で観測していない historical D1–D3 data は、上記 certainty と stress を満たすまで「causal live replay」と主張しない。

## Experiment roadmap

### P0 — freeze contracts and start forward collection

**Question:** Can every input, target, action and fill be placed on one causal timeline?

Tasks:

- [ ] freeze current v3 artifacts as historical reference; do not overwrite them
- [ ] propagate v4 availability sidecar into row and sequence eligibility
- [ ] define `event_ts / exchange_available_ts / collector_observed_ts / archive_published_ts / download_ts / decision_ts / fill_ts`
- [ ] preserve archive checksum revisions and never substitute download/archive publication time for decision-time availability
- [ ] enforce `decision t -> fill t+1 -> return t+1` in labels, U0, conditional Oracle and every Backtest call
- [ ] set one shared `p_start` and fold-boundary cost/funding rule for all policies and benchmarks
- [ ] publish per-head/fold target count, mask rate and nonzero-gradient coverage; disable heads with zero coverage
- [ ] require `seq_len >= max_horizon + execution_delay` and measured nonzero coverage, or remove that horizon/head
- [ ] generate every future-target-derived train prediction/state/normalizer/calibrator by chronological OOF
- [ ] build current inventory only from benchmark/policy replay; never from hindsight teacher path
- [ ] implement the 4-bar commitment countdown, initial state, missing/skip rule, incomplete-tail exclusion and all-or-none fill assumption in optimizer and Backtest
- [ ] replace legacy action grids/steps with one imported action/constraint/cost source of truth
- [ ] make the proposed `3 bps full spread / 1 bp slippage / 0.0003 fee` tuple and units explicit; missing config must fail closed instead of using legacy defaults
- [ ] split early-stop, calibration, policy validation and report-only test rows
- [ ] register the complete candidate family, minimum valid folds, multiplicity rule, active/turnover bounds and a new future holdout cutoff before results
- [ ] start forward collectors for OI/ratios and book data while historical experiments run

Artifacts:

- `data_contract.json`
- `timeline_contract.md`
- `source_gap_ledger.jsonl`
- `trial_registry.jsonl`
- `target_gradient_coverage.jsonl`
- `action_execution_contract.json`
- `nested_split_manifest.json`
- deterministic causality and target-alignment tests

`action_execution_contract.json` must serialize position bounds/deltas, `H_decision`, countdown reset/decrement and initial state, fill/skip/partial-fill policy, incomplete-tail mask, execution delay, return unit, spread-side convention, fee/slippage units, funding inclusion flag, `p_start` and boundary charging. Optimizer、teacher、student replay、U0 and Backtest must record and verify the same contract hash.

Gate P0 passes only when future perturbations do not change any prefix, source gaps and archive revisions remain explicit, all learned train-side teacher inputs are chronological OOF, every enabled head has valid targets and gradients, and every candidate/benchmark replays under the same inventory, action, 4-bar commitment, delay and cost contract. Any default delay/cost/initial-position fallback, teacher-derived current state, or mixing of live availability with archive/download time fails the gate.

### P1 — known-DGP and semi-synthetic recovery

**Question:** Does the pipeline recover a predictable decision when one is known to exist?

Matrix:

- SNR: zero, low, medium, high
- horizon: 1, 4, 8, 16
- regime shifts: volatility, feature mean, missingness
- constraints: cost off/on, max-step, min-hold
- model: persistence, linear/logistic only

Required outcomes:

- zero-SNR data does not produce a promoted edge
- high-SNR Q ordering and feasible action are nearly exactly recovered
- OOF teacher is worse than the clairvoyant upper bound by a sensible amount, not equal through leakage
- semi-synthetic BTC recovers the injected signal after costs
- shifting future values does not alter predictions before the shift boundary
- zero target/gradient coverage is detected as a contract failure rather than reported as model accuracy
- changing same-row future labels cannot change that row’s OOF predictive state/teacher
- delay-0 and delay-1 paths cannot silently share one experiment ID
- forecast Q ordering and realized Backtest use exactly the same 4-bar commitment path and incomplete-tail mask
- min-hold/countdown transitions are identical in optimizer, teacher, student replay and Backtest fixtures

Failure stops real-market policy experiments because it indicates an implementation or identifiability problem.

### P2 — real BTC outcome-predictability tournament

**Question:** Which available data predicts future outcomes before any Oracle/student training?

Feature sets:

1. OHLCV13
2. corrected full17
3. D1 signed/taker flow
4. D1 + D2 spot–perp state
5. later groups only after their availability gate

Models, in order:

1. unconditional / persistence / previous-volatility baseline
2. Ridge and logistic regression
3. HistGradientBoosting or another fixed-budget tree baseline
4. quantile model / calibrated ensemble
5. sequence/deep model only if a simpler model passes

Metrics:

- return: rank IC, balanced sign skill, log score / CRPS or quantile pinball loss, calibration
- volatility/drawdown: rank IC, proper score, quantile coverage
- crash: Brier score, PR-AUC where class imbalance requires it, calibration
- downstream diagnostic: regret under one fixed, preregistered action mapper
- stability: fold median, positive-fold fraction, regime/calendar slices
- falsification: feature-time shift, label permutation/block shuffle, cost stress

Screening gate:

- candidate IDs across feature group × horizon × model × seed are frozen in `trial_registry.jsonl` before their validation outputs are opened;
- the exact required fold list and minimum valid folds are fixed in the split manifest; the existing statistical gate floor of four folds is not permission to publish a partial intended run;
- candidate-parent proper-score deltas use paired time blocks/folds, a preregistered one-sided interval/test and a family-wise multiplicity correction; median/majority alone is insufficient;
- the fixed-mapper net utility also improves after costs under the same paired contract;
- the effect is not concentrated in missing/stale rows or one regime;
- shift/null controls destroy the apparent timing edge;
- missing fold, invalid target coverage or undefined primary metric is `N/A` and cannot promote;
- outer test is read once after the family winner is frozen and never selects the next feature/model.

If OHLCV13/full17 fail, do not tune WM/BC/AC. Move to D1 data. If D1 also fails, shorten the horizon only through a newly registered candidate or stop the trading objective; do not inspect the test and then rewrite thresholds.

### P3 — construct and validate the conditional Oracle

**Question:** Does an OOF conditional distribution support a useful feasible decision?

Candidates:

| id | decision rule |
| --- | --- |
| O0 | B&H / current-position hold baseline |
| O1 | point-forecast expected 4-bar excess return minus transition cost; no DD/CVaR inference |
| O2 | quantile/scenario expected utility with volatility/drawdown/CVaR |
| O3 | O2 plus uncertainty-aware abstention |
| U0 | realized-future constrained DP; upper bound only |

For every train row, O1–O3 use chronological OOF forecasts. Validation rows use a model fit only on preceding train data. U0 may use realized future outcomes but cannot create a training action, sample weight, threshold or feature.

Primary outputs:

- OOF Q vector and uncertainty per action
- joint path/scenario calibration, dependence model and tail-coverage artifact
- action margin versus hold and second-best action
- net utility and regret versus O0
- opportunity capture versus U0, reported only when the U0 denominator is positive
- risk–coverage curve
- action/transition distribution and fold/regime stability
- shared initial inventory, delay, boundary-cost and feasible-action parity proof

Gate P3:

- the entire `data × outcome model × joint scenario × utility weights × lambda × abstention threshold` family and trial count are frozen before policy outer-validation;
- O2 or O3 beats O0 and a matched constant-exposure baseline after costs using a preregistered paired interval/test and multiplicity rule;
- improvement survives common-row, delay, cost and null checks;
- active-rate, turnover and transition-support bounds were set before outputs and reject trivial all-hold/all-max-position collapse;
- DSR/PBO and trial-count checks run at this development policy-selection stage when mathematically available; an `N/A` remains exploratory and cannot be deferred as if passed;
- the selected threshold and utility weights are frozen before development test.

If P3 fails, BC and AC remain blocked. A weak Oracle cannot be repaired by learning it more accurately.

### P4 — direct policy first, student second

**Question:** Is a neural student needed, and can it reproduce the policy under its own state distribution?

Order:

1. run O2/O3 directly as the production-style benchmark;
2. distill OOF soft Q/action targets into a small student;
3. feed the student actual previous policy position, not previous teacher position;
4. add scheduled self-conditioning / DAgger-style relabeling only if trajectory error compounds;
5. compare latency, utility, turnover and stability with the direct optimizer.

Metrics:

- Q ranking and active-transition F1, not overall accuracy dominated by hold
- position MAE on active rows
- teacher/student net utility and cost-adjusted regret
- utility retained relative to the conditional Oracle
- error growth by rollout length
- unsupported/rare transition rate

The direct conditional policy remains the default unless the student materially improves utility, robustness or deployability. DAgger addresses sequential distribution shift by training on policy-induced states; see [Ross, Gordon and Bagnell](https://proceedings.mlr.press/v15/ross11a.html).

### P5 — decision-focused learning, then AC only if needed

**Question:** Can optimizing decision regret improve on the validated predict-then-optimize baseline?

Compare under the same folds, features, action set and cost:

- outcome loss only
- joint outcome + Q-ranking loss
- SPO+/decision-focused surrogate
- direct differentiable utility
- BC + self-conditioning
- WM + BC rebuilt with chronological OOF states, valid target/gradient coverage and the shared action/delay contract
- AC only after every prior candidate passes

Historical Plan011 WM + BC is reference-only; its in-sample teacher-state, delay and head-coverage contracts differ and it is not a promotion comparator until rebuilt.

Prediction error can be misaligned with downstream decision quality; this is the motivation for [Smart “Predict, then Optimize”](https://arxiv.org/abs/1710.08005) and [Decision-Focused Learning](https://ojs.aaai.org/index.php/AAAI/article/view/3982). The papers justify the comparison, not a performance claim for UniDream.

AC is promoted only when it improves over both direct O3 and BC, stays within observed action support, and survives WM-disagreement / OPE diagnostics. If AC adds no stable value, it is removed from the candidate path rather than kept for architectural complexity.

### P6 — robustness, statistics and external validation

**Question:** Is the selected method stable enough to justify a final holdout read?

- [ ] all trials, seeds, feature groups, thresholds and failed candidates are in the registry
- [ ] independently re-run P2 paired/multiplicity checks and P3 block bootstrap, fold sign test, DSR and CSCV/PBO using the existing [statistical gate contract](statistical_gate_contract.md)
- [ ] cost, delay, missingness, venue, bull/bear/high-vol regime stress
- [ ] fixed-exposure and time-shift attribution separates exposure from timing
- [ ] BTC→ETH / pooled→leave-one-asset-out external validation
- [ ] no threshold changes after development test

The already published 2024–2026 folds are historical reference, not a pristine holdout for this redesign. A new temporal holdout starts after the last previously inspected boundary and stays sealed until the development candidate, trial count and statistical gate are frozen. If insufficient future history exists, the result remains development-only.

### P7 — one final holdout and shadow replay

Open the new holdout once, report it without reselection, and keep these claims separate:

1. implementation/gate pass
2. development OOS result
3. untouched temporal holdout
4. cross-asset external validation
5. live shadow execution
6. production capital performance

A failed holdout ends the candidate. It does not trigger threshold tuning on the same window.

## Gate ledger

| gate | current status | next evidence |
| --- | --- | --- |
| P0 point-in-time / availability | **stopped** | v4 eligibility; nonzero head coverage; chronological OOF teacher states; no teacher-inventory cycle; unified initial-state/action/delay/cost contract |
| P1 synthetic recovery | **not run** | known-DGP and semi-synthetic artifacts |
| P2 real outcome predictability | **not run under corrected contract** | OHLCV13 vs corrected full17, then D1 flow tournament |
| P3 conditional Oracle | **not implemented** | OOF Q/scenario/abstention report |
| P4 direct policy / student | **blocked by P3** | direct-vs-student trajectory report |
| P5 decision-focused / AC | **blocked** | only after P4 |
| P6 statistics / external validation | **blocked** | full trial registry and selected candidate |
| P7 new holdout | **sealed / future accrual required** | single report-only read |

No new “accuracy” number exists until P2 and P3 pass. Historical Plan011 values remain historical evidence only.

## Agent organization for four concurrent slots

At most one lead plus three workers run concurrently. All experimental workers should be pinned to `gpt-5.6-luna` with `max` reasoning as requested. File ownership is exclusive within a wave.

| slot / role | owned surface | stop authority | hand-off |
| --- | --- | --- | --- |
| Lead / contract arbiter | manifests, trial registry, gate decision, final merge | can stop any downstream wave | signed gate summary and merge commit |
| A — Data / as-of | cache-v4, availability, collectors, source ledger | P0/D-group availability | source checksum, coverage, gap ledger, tests |
| B — Outcome forecast | targets, simple models, OOF predictions, forecast report | P1/P2 signal gate | per-fold proper scores, fixed-mapper regret, nulls |
| C — Oracle / decision | feasible optimizer, Q/scenarios, abstention | P3 utility gate | OOF Q artifact, risk–coverage, regret |
| D — Student / IL | BC state parity, self-conditioning, DAgger | P4 realizability gate | trajectory and utility-retention report |
| E — Decision-focused / RL | SPO/direct utility/AC candidate | P5 incremental gate | common-contract comparison |
| Independent auditor | read-only causality, split, trial and claim audit | can reject a hand-off | no edits; evidence-linked audit |

Only three worker roles are active in any wave:

~~~text
Wave A1: A availability || B target/OOF/coverage || C action/delay/inventory contract
Wave A2: A forward collectors || B synthetic recovery || independent P0 audit
Wave B:  A D1 acquisition || B real predictability || C joint-scenario optimizer fixture
Wave C:  C conditional Oracle || D direct/student || independent audit
Wave D:  E decision-focused/AC || B statistics || independent audit
~~~

### Branch, commit and push discipline

Every agent must:

- use an isolated worktree/branch named by one experiment ID;
- own an explicit file list and never stage another agent’s changes;
- make one small commit for each independently reviewable contract, test, artifact or report;
- run scoped tests plus `git diff --check` before every push;
- push after each coherent commit; never force-push;
- report commit hash, data/config/source hashes, exact command, fold/time range and gate result;
- record failed and N/A results instead of deleting them;
- stop at a failed gate and wait for the lead to authorize a new registered candidate.

The independent auditor is always read-only. The lead merges only after the owner’s tests and the audit both pass. Large binary raw data and checkpoints stay outside Git; manifests, checksums, ledgers and self-contained reports are committed.

## Immediate execution backlog

The first three implementation PRs/branches are all P0 correctness work:

1. **P0-C action/execution:** one source for spot-only action grid, max-step, 4-bar commitment countdown, `p_start`, cost units/fail-closed config, incomplete tails and enforced one-bar delay across labels/U0/validation/test.
2. **P0-B target/cross-fit:** per-head target/gradient coverage, h64/utility disabling until valid, chronological OOF WM/predictive/teacher states, and removal of the hindsight-inventory cycle.
3. **P0-A availability:** propagate v4 spot/funding/mark masks and ages into sequence eligibility; remove zero-vs-missing ambiguity from model eligibility.

In parallel, a write-ahead collector may begin accruing OI/ratio/book observations because waiting loses future history; collector data cannot enter a model until P0 audit passes.

After the three P0 branches merge and an independent read-only audit passes:

4. **P1-B recovery harness:** known-DGP + zero-signal + semi-synthetic BTC fixtures with chronological OOF.
5. **D1-A signed flow:** download/checksum Spot and USD-M kline/aggTrades fields, aggregate them to right-exclusive 15m features and publish coverage only.

Only then run P2 with OHLCV13, corrected full17 and D1 under one frozen model/target/trial budget. Conditional Oracle implementation may use synthetic OOF fixtures in parallel, but no real-data Oracle claim is allowed before the P2 signal gate.

## Research basis and limits

- Predict-then-optimize and decision-focused work shows that ordinary forecast error need not align with downstream regret: [Elmachtoub and Grigas](https://arxiv.org/abs/1710.08005), [Wilder et al.](https://ojs.aaai.org/index.php/AAAI/article/view/3982).
- Selective prediction formalizes the risk–coverage trade-off used for abstention: [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a).
- Stable/invariant feature research motivates regime-wise stability tests, but does not prove a passive BTC feature is causal: [Invariant Prediction](https://doi.org/10.1111/rssb.12167), [Stabilizing Variable Selection and Regression](https://doi.org/10.1214/21-AOAS1487).
- DAgger motivates policy-state self-conditioning when imitation errors change later states: [DAgger](https://proceedings.mlr.press/v15/ross11a.html).
- Under their assumptions, DSR/PBO help assess or adjust multiple-testing and backtest-selection risk; they supplement rather than replace preregistration, paired blocks, multiplicity correction and a sealed outer test: [Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

These sources define hypotheses and safeguards. None establishes that signed flow, LOB, conditional Oracle, BC, SPO or AC will improve UniDream. Only the gated chronological experiments above can establish that.
