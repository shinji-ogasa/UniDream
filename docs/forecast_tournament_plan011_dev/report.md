# Plan011 Development Forecast/Timing Tournament

## Scope and selection contract

- development folds only: `0, 2, 8`
- horizons: `[4, 16, 64]`; policy horizon: `16`
- commit: `e0ab435ab6601ce49b4f6c28bdb15504d2c57315`
- config SHA-256: `5c9599bdb2b5495ad525715291fba7812232b24353c4ccc16d42f1d5ad15a944`
- data SHA-256: `027b476f209d41f477623d92cd3e2f5061976840fd682874588fdcebdeab8704`
- source SHA-256: `d3c30ae4680c65492101baa3d00e33c8926c87cabc485bae769e191bf43e32a6`
- fixed operational execution delay: `1` bar; sensitivity lags: `[1, 16]`

Candidates are causal trend+volatility, Ridge direct multi-horizon, and HistGradientBoosting direct multi-horizon forecasts. Targets use only `t+1..t+h`; Oracle positions are not an input. Hyperparameters, no-trade threshold, overlay magnitude, hysteresis, and minimum hold are selected on validation only under the fixed operational execution delay. The development test interval is report-only; its aggregate is an explicitly labeled tournament screen for the next wave.

## Successive-halving gate

Pass requires positive IC on at least half of the three folds (minimum two), positive median timing increment and median AlphaEx, and dynamic timing must beat the validation-selected constant, fixed execution-lag 16 sensitivity, and temporal-destruction circular-shift null 64 by more than the pre-registered `0.02pt` margin on the aggregate median and at least `67%` of comparable folds. Lag1, shift1, and shift16 are reported robustness diagnostics rather than hard gate criteria. It must also satisfy the median MaxDDDelta/turnover tradeoff (no more than 0.05pt worse than the constant; turnover at most 6.5).

| feature set | candidate | folds | median IC | IC+ folds | median AlphaEx | Δconstant (median; wins/rate) | Δlag1 | Δlag16 | Δnull1 | Δnull16 | Δnull64 | status |
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|
| full17 | causal_trend_vol_rule | 3 | -0.0329 | 1 | -1.3651pt | -21.641; 1/33% | -0.304; 0/0% | -0.508; 0/0% | -0.454; 0/0% | -0.508; 0/0% | +0.663; 2/67% | fail |
| full17 | histgb_direct_forecast | 3 | +0.0448 | 3 | -0.7256pt | -21.001; 1/33% | -0.213; 0/0% | +0.483; 2/67% | -0.213; 0/0% | +0.522; 2/67% | -0.968; 0/0% | fail |
| full17 | ridge_direct_forecast | 3 | +0.0187 | 3 | +0.8335pt | -19.442; 1/33% | +0.712; 3/100% | -0.077; 1/33% | +0.589; 3/100% | -0.104; 1/33% | +3.304; 3/100% | fail |
| ohlcv13 | causal_trend_vol_rule | 3 | -0.0329 | 1 | -1.3651pt | -21.641; 1/33% | -0.304; 0/0% | -0.508; 0/0% | -0.454; 0/0% | -0.508; 0/0% | +0.663; 2/67% | fail |
| ohlcv13 | histgb_direct_forecast | 3 | +0.0207 | 2 | +1.0444pt | -5.010; 1/33% | +0.766; 3/100% | +0.388; 2/67% | +0.705; 3/100% | +0.388; 2/67% | +1.000; 3/100% | fail |
| ohlcv13 | ridge_direct_forecast | 3 | +0.0021 | 3 | +0.2266pt | -19.806; 1/33% | -0.099; 1/33% | +1.130; 2/67% | -0.099; 1/33% | +1.137; 2/67% | -0.373; 1/33% | fail |

Next-wave candidates (development tournament screen only): ``.

## Selected per-fold paths

| fold | feature set | candidate | selected policy | val IC(h=16) | test IC(h=16) | constant AlphaEx | timing increment | dynamic AlphaEx | MaxDDDelta | turnover |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | ohlcv13 | causal_trend_vol_rule | thr=0, mag=0.12, hyst=0, hold=32, delay=1 | -0.0118 | -0.0487 | -15.525pt | +12.419pt | -3.106pt | -0.501pt | 17.4800 |
| 0 | ohlcv13 | ridge_direct_forecast | thr=0.00041698, mag=0.12, hyst=0, hold=32, delay=1 | +0.0206 | +0.0021 | -15.525pt | +13.168pt | -2.357pt | -0.122pt | 18.3200 |
| 0 | ohlcv13 | histgb_direct_forecast | thr=0.00080754, mag=0.12, hyst=0.25, hold=32, delay=1 | +0.0077 | +0.0387 | -15.525pt | +16.570pt | +1.044pt | +0.020pt | 9.0400 |
| 0 | full17 | causal_trend_vol_rule | thr=0, mag=0.12, hyst=0, hold=32, delay=1 | -0.0118 | -0.0487 | -15.525pt | +12.419pt | -3.106pt | -0.501pt | 17.4800 |
| 0 | full17 | ridge_direct_forecast | thr=0.00088988, mag=0.12, hyst=0.25, hold=32, delay=1 | +0.0212 | +0.0187 | -15.525pt | +16.832pt | +1.307pt | -0.421pt | 15.2000 |
| 0 | full17 | histgb_direct_forecast | thr=0.00016931, mag=0.04, hyst=0.25, hold=32, delay=1 | +0.0231 | +0.0448 | -15.525pt | +14.130pt | -1.395pt | +0.092pt | 8.4800 |
| 2 | ohlcv13 | causal_trend_vol_rule | thr=6.4957e-05, mag=0.12, hyst=0, hold=32, delay=1 | -0.0709 | -0.0329 | +50.347pt | -35.359pt | +14.988pt | -0.798pt | 18.5200 |
| 2 | ohlcv13 | ridge_direct_forecast | thr=0, mag=0.12, hyst=0, hold=32, delay=1 | -0.0282 | +0.0680 | +50.347pt | -50.121pt | +0.227pt | -1.495pt | 19.4000 |
| 2 | ohlcv13 | histgb_direct_forecast | thr=0, mag=0.12, hyst=0, hold=32, delay=1 | -0.0244 | -0.0165 | +50.347pt | -5.010pt | +45.337pt | -1.562pt | 16.3600 |
| 2 | full17 | causal_trend_vol_rule | thr=6.4957e-05, mag=0.12, hyst=0, hold=32, delay=1 | -0.0709 | -0.0329 | +50.347pt | -35.359pt | +14.988pt | -0.798pt | 18.5200 |
| 2 | full17 | ridge_direct_forecast | thr=0.00065687, mag=0.12, hyst=0.25, hold=32, delay=1 | -0.0322 | +0.0686 | +50.347pt | -49.564pt | +0.783pt | +0.544pt | 18.5600 |
| 2 | full17 | histgb_direct_forecast | thr=0, mag=0.12, hyst=0, hold=32, delay=1 | -0.0175 | +0.0111 | +50.347pt | -28.116pt | +22.231pt | -0.956pt | 15.7200 |
| 8 | ohlcv13 | causal_trend_vol_rule | thr=0.00063353, mag=0.04, hyst=0, hold=32, delay=1 | +0.0186 | +0.0335 | +20.276pt | -21.641pt | -1.365pt | +0.981pt | 9.2000 |
| 8 | ohlcv13 | ridge_direct_forecast | thr=0, mag=0.04, hyst=0, hold=32, delay=1 | -0.0653 | +0.0004 | +20.276pt | -19.806pt | +0.469pt | -0.679pt | 18.8800 |
| 8 | ohlcv13 | histgb_direct_forecast | thr=0.00044928, mag=0.12, hyst=0, hold=32, delay=1 | +0.0141 | +0.0207 | +20.276pt | -20.464pt | -0.188pt | +0.747pt | 17.2000 |
| 8 | full17 | causal_trend_vol_rule | thr=0.00063353, mag=0.04, hyst=0, hold=32, delay=1 | +0.0186 | +0.0335 | +20.276pt | -21.641pt | -1.365pt | +0.981pt | 9.2000 |
| 8 | full17 | ridge_direct_forecast | thr=0.0014663, mag=0.12, hyst=0.25, hold=32, delay=1 | -0.0510 | +0.0122 | +20.276pt | -19.442pt | +0.833pt | -0.714pt | 17.2800 |
| 8 | full17 | histgb_direct_forecast | thr=0, mag=0.04, hyst=0, hold=32, delay=1 | +0.0213 | +0.0531 | +20.276pt | -21.001pt | -0.726pt | +0.944pt | 8.3200 |

## Forecast diagnostics

The ledger contains every selected validation-fit and report-only development-test metric for return/risk targets at each requested horizon. This compact table shows the principal horizons and risk diagnostics; risk sign accuracy is intentionally N/A because realized risk is one-sided.

| fold | feature set | candidate | split | return IC h4 | return IC h16 | return IC h64 | return sign h16 | risk MAE h16 | risk RMSE h16 | risk IC h16 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | ohlcv13 | causal_trend_vol_rule | validation | +0.0217 | -0.0118 | +0.0627 | +0.4827 | +0.0020 | +0.0044 | +0.1286 |
| 0 | ohlcv13 | causal_trend_vol_rule | development_test | +0.0463 | -0.0487 | -0.0204 | +0.4687 | +0.0016 | +0.0021 | -0.3100 |
| 0 | ohlcv13 | ridge_direct_forecast | validation | +0.0171 | +0.0206 | +0.0717 | +0.4948 | +0.0019 | +0.0040 | +0.4415 |
| 0 | ohlcv13 | ridge_direct_forecast | development_test | -0.0045 | +0.0021 | -0.0034 | +0.4923 | +0.0013 | +0.0019 | +0.4776 |
| 0 | ohlcv13 | histgb_direct_forecast | validation | +0.0398 | +0.0077 | +0.0206 | +0.5004 | +0.0019 | +0.0042 | +0.4315 |
| 0 | ohlcv13 | histgb_direct_forecast | development_test | +0.0513 | +0.0387 | +0.0364 | +0.5046 | +0.0012 | +0.0019 | +0.4497 |
| 0 | full17 | causal_trend_vol_rule | validation | +0.0217 | -0.0118 | +0.0627 | +0.4827 | +0.0020 | +0.0044 | +0.1286 |
| 0 | full17 | causal_trend_vol_rule | development_test | +0.0463 | -0.0487 | -0.0204 | +0.4687 | +0.0016 | +0.0021 | -0.3100 |
| 0 | full17 | ridge_direct_forecast | validation | +0.0179 | +0.0212 | +0.0693 | +0.4910 | +0.0018 | +0.0039 | +0.4948 |
| 0 | full17 | ridge_direct_forecast | development_test | -0.0007 | +0.0187 | +0.0088 | +0.4911 | +0.0013 | +0.0019 | +0.4493 |
| 0 | full17 | histgb_direct_forecast | validation | +0.0380 | +0.0231 | +0.0568 | +0.5070 | +0.0018 | +0.0043 | +0.4514 |
| 0 | full17 | histgb_direct_forecast | development_test | +0.0611 | +0.0448 | +0.1091 | +0.4998 | +0.0012 | +0.0019 | +0.4083 |
| 2 | ohlcv13 | causal_trend_vol_rule | validation | +0.0535 | -0.0709 | -0.0487 | +0.4748 | +0.0013 | +0.0018 | -0.0342 |
| 2 | ohlcv13 | causal_trend_vol_rule | development_test | +0.0377 | -0.0329 | +0.0631 | +0.4999 | +0.0017 | +0.0027 | +0.2929 |
| 2 | ohlcv13 | ridge_direct_forecast | validation | -0.0108 | -0.0282 | -0.0224 | +0.4804 | +0.0013 | +0.0018 | +0.6127 |
| 2 | ohlcv13 | ridge_direct_forecast | development_test | +0.0278 | +0.0680 | +0.0966 | +0.5163 | +0.0016 | +0.0024 | +0.5801 |
| 2 | ohlcv13 | histgb_direct_forecast | validation | +0.0291 | -0.0244 | -0.0048 | +0.5214 | +0.0012 | +0.0016 | +0.6082 |
| 2 | ohlcv13 | histgb_direct_forecast | development_test | +0.0222 | -0.0165 | +0.0262 | +0.5197 | +0.0016 | +0.0025 | +0.5716 |
| 2 | full17 | causal_trend_vol_rule | validation | +0.0535 | -0.0709 | -0.0487 | +0.4748 | +0.0013 | +0.0018 | -0.0342 |
| 2 | full17 | causal_trend_vol_rule | development_test | +0.0377 | -0.0329 | +0.0631 | +0.4999 | +0.0017 | +0.0027 | +0.2929 |
| 2 | full17 | ridge_direct_forecast | validation | +0.0007 | -0.0322 | -0.0466 | +0.4854 | +0.0013 | +0.0017 | +0.6018 |
| 2 | full17 | ridge_direct_forecast | development_test | +0.0009 | +0.0686 | +0.1236 | +0.5174 | +0.0016 | +0.0024 | +0.5290 |
| 2 | full17 | histgb_direct_forecast | validation | +0.0105 | -0.0175 | +0.0116 | +0.5206 | +0.0012 | +0.0016 | +0.6061 |
| 2 | full17 | histgb_direct_forecast | development_test | +0.0094 | +0.0111 | +0.0470 | +0.5235 | +0.0016 | +0.0026 | +0.4848 |
| 8 | ohlcv13 | causal_trend_vol_rule | validation | +0.0435 | +0.0186 | +0.0623 | +0.5158 | +0.0014 | +0.0018 | -0.0341 |
| 8 | ohlcv13 | causal_trend_vol_rule | development_test | +0.0345 | +0.0335 | -0.0082 | +0.5077 | +0.0015 | +0.0022 | +0.1348 |
| 8 | ohlcv13 | ridge_direct_forecast | validation | -0.0216 | -0.0653 | +0.0737 | +0.4609 | +0.0012 | +0.0016 | +0.4792 |
| 8 | ohlcv13 | ridge_direct_forecast | development_test | -0.0055 | +0.0004 | +0.0112 | +0.4866 | +0.0013 | +0.0019 | +0.5402 |
| 8 | ohlcv13 | histgb_direct_forecast | validation | -0.0048 | +0.0141 | +0.0664 | +0.5101 | +0.0012 | +0.0016 | +0.4687 |
| 8 | ohlcv13 | histgb_direct_forecast | development_test | +0.0128 | +0.0207 | -0.0095 | +0.4929 | +0.0013 | +0.0019 | +0.5543 |
| 8 | full17 | causal_trend_vol_rule | validation | +0.0435 | +0.0186 | +0.0623 | +0.5158 | +0.0014 | +0.0018 | -0.0341 |
| 8 | full17 | causal_trend_vol_rule | development_test | +0.0345 | +0.0335 | -0.0082 | +0.5077 | +0.0015 | +0.0022 | +0.1348 |
| 8 | full17 | ridge_direct_forecast | validation | -0.0159 | -0.0510 | +0.1266 | +0.4664 | +0.0012 | +0.0017 | +0.4872 |
| 8 | full17 | ridge_direct_forecast | development_test | -0.0081 | +0.0122 | +0.0836 | +0.4920 | +0.0013 | +0.0020 | +0.4836 |
| 8 | full17 | histgb_direct_forecast | validation | +0.0206 | +0.0213 | +0.0865 | +0.5056 | +0.0013 | +0.0017 | +0.4726 |
| 8 | full17 | histgb_direct_forecast | development_test | +0.0194 | +0.0531 | +0.0350 | +0.4927 | +0.0013 | +0.0019 | +0.5160 |

## Feature coverage

Full17 rows carry separate finite/missing/zero/nonzero counts for funding_rate, basis, basis_mom, and basis_abs. The cache has no availability mask, so zero-versus-missing remains an explicit quality flag and is not silently treated as observed signal.

| fold | feature set | split | rows | external nonzero/missing counts (funding, basis, basis_mom, basis_abs) | quality/status |
|---:|---|---|---:|---|---|
| 0 | ohlcv13 | train | 69703 | funding_rate=0/69703, basis=0/69703, basis_mom=0/69703, basis_abs=0/69703 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 0 | ohlcv13 | validation | 8701 | funding_rate=0/8701, basis=0/8701, basis_mom=0/8701, basis_abs=0/8701 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 0 | ohlcv13 | test | 8712 | funding_rate=0/8712, basis=0/8712, basis_mom=0/8712, basis_abs=0/8712 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 0 | full17 | train | 69703 | funding_rate=12293/0, basis=2311/0, basis_mom=2309/0, basis_abs=2310/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 0 | full17 | validation | 8701 | funding_rate=8701/0, basis=8701/0, basis_mom=8701/0, basis_abs=8701/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 0 | full17 | test | 8712 | funding_rate=8712/0, basis=8712/0, basis_mom=8712/0, basis_abs=8712/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 2 | ohlcv13 | train | 69953 | funding_rate=0/69953, basis=0/69953, basis_mom=0/69953, basis_abs=0/69953 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 2 | ohlcv13 | validation | 8832 | funding_rate=0/8832, basis=0/8832, basis_mom=0/8832, basis_abs=0/8832 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 2 | ohlcv13 | test | 8808 | funding_rate=0/8808, basis=0/8808, basis_mom=0/8808, basis_abs=0/8808 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 2 | full17 | train | 69953 | funding_rate=29706/0, basis=19724/0, basis_mom=19722/0, basis_abs=19723/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 2 | full17 | validation | 8832 | funding_rate=8832/0, basis=8832/0, basis_mom=8832/0, basis_abs=8832/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 2 | full17 | test | 8808 | funding_rate=8808/0, basis=8808/0, basis_mom=8808/0, basis_abs=8808/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 8 | ohlcv13 | train | 70022 | funding_rate=0/70022, basis=0/70022, basis_mom=0/70022, basis_abs=0/70022 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 8 | ohlcv13 | validation | 8640 | funding_rate=0/8640, basis=0/8640, basis_mom=0/8640, basis_abs=0/8640 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 8 | ohlcv13 | test | 8736 | funding_rate=0/8736, basis=0/8736, basis_mom=0/8736, basis_abs=0/8736 | ok_with_quality_flag; N/A_external_columns_not_in_ohlcv13 |
| 8 | full17 | train | 70022 | funding_rate=70022/0, basis=70022/0, basis_mom=70022/0, basis_abs=70022/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 8 | full17 | validation | 8640 | funding_rate=8640/0, basis=8640/0, basis_mom=8640/0, basis_abs=8640/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |
| 8 | full17 | test | 8736 | funding_rate=8736/0, basis=8736/0, basis_mom=8736/0, basis_abs=8736/0 | ok_with_quality_flag; N/A_zero_vs_missing_indistinguishable |

## Timing/null contract

Each selected path reports the validation-selected constant exposure, dynamic path, execution lag 1 and 16, and deterministic circular-shift nulls 1, 16, and 64. The fixed operational delay is the main result; lagged executions are sensitivity diagnostics. AlphaEx, MaxDDDelta, SharpeDelta, and turnover come from the shared Backtest/action_stats contract with B&H position 1.0 and the configured costs.

Unavailable or undefined forecast metrics are recorded as `N/A` with a reason; no test result selects a hyperparameter or policy. A passing row is a candidate for a later wave, not evidence that the full Plan011 model should be retrained.

Machine-readable ledger: `docs/forecast_tournament_plan011_dev/forecast_tournament_ledger.jsonl`
