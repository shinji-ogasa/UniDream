# BTC accuracy release coordination contract — 2026-09-06

Owner decisions for parallel implementation. Recipe: `perp_delay0_reliability_utility_risk1` (hold). Bundle ID and run ID: `btc-perp-reliability-20260906`. This is ML + deterministic inventory utility, not learned RL. Production fit is exactly calendar25 (E start2026-07-16T13:45Z), unchanged T18m/S3m/I3m; no E performance selection. Historical validation5–12 metrics remain separately identified. Preserve BNBv2 and legacy BTC routes/history. Root commits/deploys; agents only own assigned files.

## Bundle contract (research export owner: evidence_audit)

Directory `codex_outputs/btc_reliability_release_v1/bundle/`:

- `manifest.json`: `schema_version:1`, `bundle_type:"btc_perp_reliability_v1"`, `bundle_id`, `candidate_id`, `symbol:"BTCUSDT"`, `interval:"15m"`, `calendar` (ISO values), `production_cutoff`, `feature_columns` mapping technical29/perp_delay0=31, `execution` (below), `source_bindings`, `files` mapping relative non-manifest paths to SHA256, `historical_evidence` (or separate evidence.json bound in files), `research_scores_apply_to_production_weights:false`, `high_probability_generalization_established:false`, `rl_qualified:false`.
- `models/{technical_mean,perp_delay0_mean,technical_variance}.joblib` (frozen helper parameters).
- `calibration.json`: exact returned calibration dict (`return_bias`, `variance_multiplier`, `fit_mean`, `scale_mean`, `technical_quantiles`, `counts`). No new threshold or model choice.
- `reliability.json`: exact `fit_reliability` return record, including `weight`, fit period and provenance separately if needed.
- Export selected T/predict matrices, outcomes and source/provenance outside bundle for audit; small historical parity fixtures may be added to bundle by root. SHA of manifest bytes is `bundle_sha256`. Do not self-hash manifest.
- Runtime pin Python3.12, numpy2.2.6,pandas2.3.3,sklearn1.8.0; joblib version match research environment. Source hashes and feature/execution digests remain distinct.

Execution: one_way_cost=.00055, borrow_annual=.10, max_step=.08, deadband=.01, risk_aversion=1, utility_cost_multiplier=2, horizon_bars=24, decision_hours_utc=[0,6,12,18], fill_delay_bars=1, intent bounds[.5,1.12], missing_forecast=hold, initial_cash=0.0, initial_equity=1.0, initial_units="1/initial_open", bars_per_year=35040. Use BARS_YEAR=365*96 and exact research exponential borrow. No cashflow into existing run. Human display capital10000USDT is a scaling/display choice only; canonical state stays research-normalized to1 for exact numerical parity.

## HF HTTP contract (owner: technical_features)

GET `/v3/btc/health`: ok/bundle_id/bundle_sha256/candidate_id/symbol/interval/production_cutoff/feature_contract_sha256/execution_contract_sha256/runtime/manifest (public metadata, no secrets). GET `/v3/btc/verify`: fixture proof, with scope and max errors. POST `/v3/btc/predict`: API-key protected like legacy route, accepts:

```
{
  "run_id":"btc-perp-reliability-20260906",
  "bundle_id":"btc-perp-reliability-20260906",
  "event_ts":"UTC aligned current15m open",
  "received_at":"server collector timestamp",
  "current_open": {"timestamp":"same event_ts", "open":12345, "received_at":"UTC"},
  "spot": [{"timestamp":"closed bar OPEN", "open":...,"high":...,"low":...,"close":...,"volume":...,"quote_volume":...,"taker_buy_quote":...,"n_trades":...}],
  "um": [same raw schema],
  "state": null or previous `state` JSON
}
```

No current high/low/close/volume enters features. Every provided feature bar must be closed strictly before event_ts (raw index < event_ts); preserve absent15m rows as NaN on reindex. Require8641 elapsed closed bars of history plus inserted current feature row. Raw values may be null for genuine gaps; no imputation. Compute the identical common dependency gate: old flow block/all derivative groups/all delayed0/1/4/trailing_variances, all from completed bars. Late receipt prevents a new order; archive warmup is not receipt-time proof. Default decision/fill deadline60s after event open. Validate against server UTC to prevent future received_at and future event_ts.

At nondecision 15-minute events, accept a light accounting payload: Spot closed bars start at or before `state.last_open_ts` and cover the elapsed accounting window, preserving absent rows as unavailable; a new run needs at least one closed Spot row. `um:[]` is allowed. No feature construction or model prediction runs; return `forecast:null` and `feature_support:null`. At scheduled UTC hours 00/06/12/18, both full histories still require8641 elapsed closed bars. This reduces collector work without relaxing accounting coverage or inventing missing closes.

Forecast `available` records finite features/model outputs. `action_eligible` additionally requires a known current open and completion by the server-side event+60s deadline. A late calculation may retain diagnostic mu/variance but its target is null. State and snapshots retain cumulative `turnover` (sum absolute traded notional divided by contemporaneous pre-trade NAV), `trades`, fees and borrowing. These cumulative amounts are distinct from current exposure.

Feature contract digests use the export's exact semantic `feature_contract` dictionary; vendor source provenance has a separate digest. The runtime requires exact dictionary/digest equality. Execution digest uses the complete canonical execution dictionary including `bars_per_year:35040` and float initial cash/equity.

The same Python engine handles canonical accounting and forecast decisions. Avoid a second independent TS arithmetic implementation. New run initializes on the first known current open with canonical NAV1/B&H1, cash0 and units1/open. This is virtual common initial inventory, not a fee-free executable purchase claim.

State is stored **after current open's fill/decision, before that bar's borrow/close**. On next request, complete each intervening closed bar: borrow once, mark actual close if present, update strategy/B&H peaks and DD, append snapshot. Across missing bars borrow still accrues. Missed clock calls never retroactively create new intents. A pending intent fills only at its exact next15m current open with timely receipt; missing/late due event expires it, never rolls forward or retroactively uses a historical open. At timely current open process pending fill first, then (only UTC6h) call canonical `_choose` with actual current cash/units and mu/variance. Hold=no numerical order, preserve null intent. Natural exposure can exceed intent cap.

Required response:

```
{
 "ok":true,"run_id":...,"bundle_id":...,"bundle_sha256":...,"feature_contract_sha256":...,"execution_contract_sha256":...,
 "event_ts":...,"expected_state_version":0,
 "state": {
  "schema_version":1,"run_id":...,"bundle_id":...,"bundle_sha256":...,"version":1,
  "last_open_ts":...,"started_at":...,"initial_open":...,"initial_equity":1.0,"display_capital":10000.0,
  "cash":...,"units":...,"benchmark_units":...,"fees":...,"borrow":...,"turnover":...,"trades":...,
  "pending_target":null,"pending_decision_ts":null,"pending_due_at":null,
  "last_mark_ts":null,"last_mark_price":null,"equity":1.0,"benchmark_equity":1.0,
  "peak_equity":1.0,"benchmark_peak_equity":1.0,"max_drawdown":0.,"benchmark_max_drawdown":0.,
  "current_open_equity":...,"current_open_exposure":...
 },
 "snapshots":[{"timestamp":"closed bar OPEN", "price":...,"equity":...,"benchmark_equity":...,"exposure":...,"cash":...,"units":...,"fees":...,"borrow":...,"turnover":...,"trades":...,"max_drawdown":...,"benchmark_max_drawdown":...}],
 "forecast":null or {"decision_ts":...,"available":true/false,"action_eligible":true/false,"reason":...,"mu":...or null,"variance":...or null,"target":...or null,"estimated_utility_gain":...or null,"known_open_exposure":...},
 "events":[{"event_id":"run/time/kind deterministic", "timestamp":...,"kind":"fill|intent|expired", "details":{...}}],
 "data": {"spot_latest_closed_ts":...,"um_latest_closed_ts":...,"received_at":...,"spot_sha256":...,"um_sha256":...,"timely":true/false}
}
```

A duplicate event_ts returns an explicit already_processed result with no state mutation. The Edge also short-circuits before HTTP. Missing/invalid state, negative NAV, wrong model, future bars and invalid schema fail closed. A model error does not replace target with0. Current-open and last-close equity stay visibly distinct.

Fixture verification: historical fold parity feature→raw mean/risk→calibration→reliability→intents→cash/units/NAV/B&H/DD. Include gaps, borrow, natural drift, zero/no trade, zero-weight branch, late input and pending expiry. Unit tests can bypass wallclock via injected server time, never public arbitrary clock. Preserve initial capital in DD calculation.

## Edge + Supabase (root owner)

Use a new isolated BTC run and new tables, preserving old schema/history. Names:

- `btc_demo_runs`: `run_id` PK, `bundle_id`, `bundle_sha256`, `feature_contract_sha256`, `execution_contract_sha256`, `manifest` jsonb, `created_at`.
- `btc_demo_state`: `run_id` PK, `version` bigint, `last_open_ts` timestamptz, `state` jsonb, `updated_at`.
- `btc_demo_snapshots`: (`run_id`,`timestamp`) PK, normalized equity/benchmark/exposure/price/cash/units/fees/borrow/DD numeric doubles, `bundle_sha256`, `created_at`.
- `btc_demo_forecasts`: (`run_id`,`decision_ts`) PK, fields from forecast + model/feature/execution hashes, `data` jsonb, `created_at`.
- `btc_demo_events`: (`run_id`,`event_id`) PK, `timestamp`, `kind`, `details` jsonb, bundle hash.

RLS: public select for these public demo tables; no client writes. Service-only atomic RPC `record_btc_demo_transition(payload jsonb)` verifies registered bundle/contract hashes, row lock/version CAS, event idempotency, inserts snapshots/forecast/events and updates state as one transaction. New Edge `run-btc-research-demo` fetches complete Spot+UM quote/taker raw history/current-open, reads state, calls HF and applies RPC. Every15m at minute0/15/30/45; current legacy cron is suspended only once new flow is validated. No retrospective simulated live fills when cron was not running.

## Web UI (owner: primary_research, src/public only)

Default BTC demo reads the new run/tables only, SSR and Realtime both filter run_id. Read-only UI; no client secret or state writes. Keep existing visual language, avoid a broad redesign. Show production recipe and training/calibration cutoff, bundle digest and matching conditions (BTC15m,6h decisions,next15m open,5.5bps one-way,10% borrow,hold missing-data rule), plus state freshness/pending target/current exposure/last closed equity and normalized B&H comparison (display ×10000USDT).

Historical evidence panel from versioned static release manifest (root provides final): dev2021-04-16→2023-04-16,8quarters, base +4.450249pt/−5.143249pt, stress +4.247804/−5.055918, joint3/8,regimes2/4/2. These are quarter means, not a concatenated curve. Predictor MSE/probability and high-probability gates did not establish reliable future alpha. RL currently unqualified. Additional test for reliability not available; never substitute half's additional test as reliability evidence. Live B&H/equity is a separate new series. Preserve legacy pages/routes if useful but prevent old BTC records mixing into the new dashboard.

Validation: Next typecheck/build, real desktop/mobile screenshot, HF/Edge/DB bundle hashes and timestamps match. New empty run before first mark is honest pending state, not fabricated chart values.
