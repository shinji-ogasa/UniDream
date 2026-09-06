# Retained UniDream WM + RL runtime — read-only audit, 2026-09-06

The retained `bundles/current` is a real Transformer world model plus an actor trained with imagination actor-critic, initialized by BC. It is architecturally eligible for the user's new WM+RL requirement. This statement does **not** establish that its accuracy or AlphaEX/MaxDD gates pass. The newly published ML `/v3/btc` recipe is a separate diagnostic and does not meet that architecture requirement.

All HF paths below are relative to `/Users/sophie/Documents/UniDream/.worktrees/accuracy-release-space-20260906`; research paths are relative to `/Users/sophie/Documents/UniDream/UniDream`. No repository source, deployment, fit, forecast, or score was changed by this audit. Checkpoints were loaded only to inspect state/metadata and compare stored parameters.

## Checkpoint proof

- `bundles/current/manifest.json`: bundle_type `plan011_v31_overlay_actor`, run `Plan011v31`, fold23/seed7/mode `ac`; train2023-10-16→2025-10-16, validation2025-10-16→2026-01-16, test2026-01-16→2026-04-16 (13:45 boundaries). `latest_holdout_candidate` is a stored label, not evidence of strongest-model qualification.
- `actor_full.pt` loads as `unidream.actor_critic.actor.Actor`. Its 32 state-dictionary entries exactly equal `checkpoints/ac.pt['actor']` (max absolute difference0). They differ from `checkpoints/bc_actor.pt['actor']` (max absolute difference0.012854482978582382).
- `ac.pt` has `actor`, `critic`, both optimizers, and `global_step=300`. `world_model.pt` has ensemble and eight predictive heads plus inverse-dynamics head, optimizer, `global_step=700`.
- Research `unidream/actor_critic/imagination_ac.py:1101` runs imagined rollouts; `:1245` combines policy-gradient/entropy AC loss with BC anchor and other prior losses; `:1275` steps the actor optimizer. This is BC-initialized, BC-regularized RL, not BC-only and not live online learning.
- Export chain: research `unidream/cli/export_inference_bundle.py:114` calls `load_inference_run_context`; `unidream/experiments/checkpoint_eval.py:259` loads the requested AC actor; export `:145` serializes that actor. The retained old checkpoint has no recorded `inference_selection` metadata. Its pickled inference setting is `.7`; current loader falls back to the first configured adjustment grid value when selection metadata is absent. Do not call the specific `.7` provenance independently proven validation selection.

SHA256: actor `35acf3b3c1242b565a9fea0212de53c98371f124464dec63ea34122b14c6d54c`; AC `6d53bd94a8c9c19f5c907a8f8f97cb008afed1e5cd44c246001f502e1ab06175`; BC `1c856d9baf880ca374afe63257a54c95d98eb0d6a8ad0696718734a4047284b0`; WM `d97967225515b7d7edfeed335ce2ff5b10df0b236b6d71e1972697b8a8095442`.

## Actual forward path

`backend/predictors/plan011.py:162` validates float32 `[T,17]` features, then calls `wm.encode_sequence(features, actions=None, seq_len=64)`.

1. `backend/vendor/unidream/world_model/train_wm.py:1137`: one WM model (`n_ensemble=1`), observation encoder17→192→192→32×32 categorical latent probabilities, deterministic eval `z∈R1024`; causal Transformer2 layers/4 heads/d_model192/d_ff768/max context128 produces `h∈R192`. Although the class is named EnsembleWorldModel, this bundle has **one**, not three, model members.
2. `actions=None` uses benchmark exposure1.0 for every WM input action under `reward.mode=excess_bh`; it is not the live fill history. The 64-row chunk encoder adds preceding64 rows, and its final mature row sees128 feature rows. A new incremental wrapper must reproduce this final-row context/positional convention; feeding an arbitrary short suffix is not automatically equivalent.
3. `Plan011Predictor._predictive_state`, `WorldModelTrainer.predict_auxiliary_from_encoded` supply the exact feature row as well as z/h to learned auxiliary heads (`aux_use_raw_features=true`). Return/vol/drawdown/crash/drawdown-excess at horizons4/8/16/32/64, seven position-utility outputs, overweight-advantage5 and recovery5 concatenate to42 dimensions. Saved `predictive_state.npz` mean/std standardize these, followed by fixed clip±5. The model does not receive future realized returns here.
4. `_default_regime` derives three probabilities from the causal feature-stress signal and saved center/scale; these are not a future trend label.
5. `Actor._prepare_inputs` (`actor.py:352`) concatenates z1024+h192+controller4+regime3+aux42 =1265 into256→256 ELU trunk. `act_greedy` (`:1258`) uses learned trade/target/residual/band outputs and fixed execution controls. The stored actor uses residual controller, `.5≤position≤1.12`, max step `.08`, adjustment scale `.7`, long-rate cap `.15`. It is not an unused WM sitting beside a purely technical rule.

## Exact feature contract and live inputs

Ordered17 columns: `open_ret, high_ret, low_ret, close_ret, vol_ret, RSI_14, macd, macd_signal, atr_norm_ret, atr, rv_4, rv_16, rv_96, funding_rate, basis, basis_mom, basis_abs`.

Raw schema must include UTC15m open timestamp plus finite positive OHLC, valid volume, historical as-of `funding_rate`, and positive closed-bar `mark_close`. Mark price is specifically the derivative **mark**, not UM traded close. Quote/taker-flow columns from ML Perp31 cannot replace these inputs. OI is absent from this17-column manifest and should not be fetched/required for the runtime.

- Returns use `log(P[t−1]/P[t−2])`; RSI/MACD/ATR/RV are shifted once. ATR-normalized close return divides by prior relative ATR.
- Funding is latest event at or before each raw bar timestamp, then shifted once. There must be an observed event at/before the first raw bar; never seed an absent history with0.
- `basis[t]=log(mark_close[t−1]/spot_close[t−1])`; current source additionally shifts basis difference and absolute basis: `basis_mom[t]=basis[t−1]−basis[t−2]`, `basis_abs[t]=abs(basis[t−1])`. Preserve this existing double-lag for those two derived columns.
- All17 columns get rolling z-score with window60×96=5760, minimum observations1440, sample std and denominator `std+1e−8`; normalizer at t sees only already-shifted observations. Early partial windows are permitted by the function, so the HTTP minimum1504 candles is not a full60-day parity guarantee.
- The existing web collector's7248-candle choice gives60days plus1488 warmup, a useful conservative starting point. For a parity claim, bind raw warmup/EMA seed and the final128 feature rows against saved research features. MACD has recursive EMA initialization, so exact full-history equality cannot be inferred merely from a finite bar count. Freeze a tolerance or persist the original EMA/rolling state after proving equivalence.

Existing protective gates are real: `backend/feature_contract.py` demands funding/mark columns and finite matrix/order; `backend/feature_pipeline.py:330` no longer zero-fills missing columns. Its candle validator checks unique contiguous closed timestamps. However, vendor `data/features.py` still contains mark `bfill`, missing-funding0, and OI/extra-series fill paths. Full mark coverage and a prior funding event make the relevant fallback paths unused; a dedicated fail-closed wrapper should enforce those preconditions explicitly. No OI/extra-series branch is needed.

A material **unresolved implementation equivalence**: HF `HAS_PANDAS_TA=False` and uses the SMA RSI/ATR fallback. Research `pyproject.toml`/`uv.lock` include `pandas-ta0.4.71b0`, whose TA path is different. Cache metadata records17 names and input flags, but not the generating package/runtime hashes. A schema alias from fallback `close` to `RSI_14` does not prove identical indicator values. Existing `/sample/verify` proves precomputed17→WM/actor behavior; it does not settle raw→17 parity. Resolve this with raw+mark+funding fixtures against the saved research matrix before calling live inference research-equivalent; do not blindly install a different TA branch and assume it matches.

Official raw sources: Binance [`GET /fapi/v1/fundingRate`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) supplies funding event time/rate, and [`GET /fapi/v1/markPriceKlines`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data) supplies mark klines indexed by open time. Receipt times still require the collector's own logging. This audit did not fetch new market values or recheck their geographic availability.

## Implementable runtime contract

Use a versioned WM+RL request containing `model_id/bundle hash/decision_ts`, completed raw Spot+mark history, funding events with origin/receipt timestamps, and persisted policy state. Accept no caller `advantage`, `regime`, replacement feature columns, target overrides, or zero-fill in that production path; the retained generic legacy route can remain for fixture/debug compatibility.

At decision t, every raw feature bar is <t and closed/received by the declared deadline. Internally append a missing current feature row at t, then apply the unchanged causal feature functions and select features[t]. Never pass fabricated current HLCV. **Current legacy candle POST ends at closed t−1, so its last feature row is labelled t−1 and only uses through t−2; simply relabelling that output as a t decision introduces an extra15m lag.** Its current validator rejects a public unclosed current candle, so current-row construction belongs inside the new wrapper.

Run the WM/actor at its original **15m** sequence cadence. Do not inherit MLv3's six-hour cadence, cost-budget utility score, Ridge/HGB means, or next-open policy as though they were Plan011's registered conditions.

Persist the actor's four policy-state components and cumulative decision/active/underweight/long counts, with exact `update_controller_state` semantics and the original .15 long-rate constraint. `predict_positions` currently zero-initializes these on every call and replays the entire supplied feature window. Thus sliding-window last positions are not guaranteed equal to a continuously running actor. The state is `[previous target−1, previous target change, hold progress/64, underweight progress/64]`, not actual cash/units/exposure. Keep the policy state and paper-account state distinct; substituting live cash exposure into the trained actor is a changed model input contract needing explicit validation.

Store timestamp/hash/receipt proof and nullable unavailability independently from target. Missing required inputs mean hold/error, never target0. Any extraction must verify original full-sequence versus persisted-step targets on existing fixtures, including rate-cap counters and partial chunks; then compare feature→WM z/h→aux42/regime3→actor outputs on the same raw history. The live fee/fill/cash accounting must be matched to the WM research evaluator in a separate explicit contract; MLv3 account parity does not establish that boundary.
