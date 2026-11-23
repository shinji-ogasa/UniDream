# UniDream 仕様調査台帳

調査日: 2026-11-07（JST）。対象commit: `fc2c608e66d71d8a9f31bd73cc9f6a15926c64a5`。

[プロジェクト仕様書](../SPEC.md) の調査範囲と追跡用の補助資料。

## 調査方法・範囲

- 実装 `104` ファイル、テスト `49` ファイル、YAML `15` ファイルを棚卸し。
- 全Pythonファイルの構文、トップレベル定義、docstring、テストのimport関係を抽出した。
- mainlineのCLI→run→fold→stage、data/cache/WFO、model、selector/backtest、checkpoint/exportを重点的に読み、設定・テストと照合した。
- v4/P0/P1、Alpha/DD、Robust overlayは実装の入口・契約と保存済み報告を照合した。全分岐を新たに実行して検証したという意味ではない。
- 外部repoの実装、公開サービスの現在状態、学習再実行、保存済み経済結果の独立再計算は対象外。
- 構成一覧は生成物・venv・cacheを含めず、ソースファイルを対象とする。Gitで明示追跡済みのP1等の証拠は別途存在する。

## 今回の検証

| 確認 | 結果 |
|---|---|
| 全unit test | **384 tests OK**、100.541秒、exit 0 |
| `git diff --check` | 成功 |
| 本仕様書・調査台帳のローカルリンク | ファイル参照202件・目次anchor20件を確認、欠落0 |
| 現行v31 checkpoint smoke | loader/export未変更のため対象外 |

テストではtimezone変換とscikit-learnのmatmulに関する警告27件も出力された。エラー・失敗は0。完全なローカル実行ログは `/tmp/unidream-specification-tests-20261107.log`（一時ファイル）に保存した。学習本番・公開推論の検証結果ではない。

## 実装一覧

説明はmodule docstringまたは責務要約。公開定義はトップレベルのclass/functionであり、全method一覧ではない。

| ファイル | 責務 | 主な公開定義 |
|---|---|---|
| [unidream/__init__.py](../unidream/__init__.py) | package初期化・公開import | — |
| [unidream/actor_critic/__init__.py](../unidream/actor_critic/__init__.py) | package初期化・公開import | — |
| [unidream/actor_critic/actor.py](../unidream/actor_critic/actor.py) | Actor モジュール. | Actor |
| [unidream/actor_critic/bc_pretrain.py](../unidream/actor_critic/bc_pretrain.py) | BC (Behavior Cloning) 事前学習モジュール. | SIRLWeightNet, ControllerPathDataset, BCPretrainer |
| [unidream/actor_critic/critic.py](../unidream/actor_critic/critic.py) | Critic（Value function）モジュール. | Critic, RewardEMANorm |
| [unidream/actor_critic/imagination_ac.py](../unidream/actor_critic/imagination_ac.py) | Imagination Actor-Critic 学習モジュール. | ImagACTrainer |
| [unidream/cli/__init__.py](../unidream/cli/__init__.py) | package初期化・公開import | — |
| [unidream/cli/acquire_d1_signed_flow.py](../unidream/cli/acquire_d1_signed_flow.py) | Acquire the small D1 signed-flow kline pilot and write evidence ledgers. | run_pilot, build_parser, main |
| [unidream/cli/audit_official_gap_recovery.py](../unidream/cli/audit_official_gap_recovery.py) | Probe official Binance Spot sources around development-cache gaps. | build_parser, main |
| [unidream/cli/compare_policy_families.py](../unidream/cli/compare_policy_families.py) | Leak-free policy-family comparison on configured WFO folds. | ExecutionSpec, main |
| [unidream/cli/evaluate_alpha_attribution.py](../unidream/cli/evaluate_alpha_attribution.py) | CLI for leak-aware alpha attribution and saved-bundle diagnostics. | main |
| [unidream/cli/evaluate_constant_exposure_diagnostic.py](../unidream/cli/evaluate_constant_exposure_diagnostic.py) | Run the Wave3D returns-only constant-exposure diagnostic. | build_parser, main |
| [unidream/cli/evaluate_forecast_context_tournament.py](../unidream/cli/evaluate_forecast_context_tournament.py) | CLI for the fixed Wave3C causal-context forecast screen. | main |
| [unidream/cli/evaluate_forecast_tournament.py](../unidream/cli/evaluate_forecast_tournament.py) | CLI for the development-only direct forecast/timing tournament. | main |
| [unidream/cli/export_inference_bundle.py](../unidream/cli/export_inference_bundle.py) | completed runからconsumer向けbundle生成 | main |
| [unidream/cli/plot_plan011_fold_trades.py](../unidream/cli/plot_plan011_fold_trades.py) | Plot Plan011 fold trades and equity curves from saved checkpoints. | FoldPlotResult, trade_indices, active_blocks, main |
| [unidream/cli/rebuild_official_v4_cache.py](../unidream/cli/rebuild_official_v4_cache.py) | Rebuild a schema-v4 cache from official Binance archives only. | build_parser, main |
| [unidream/cli/statistical_gate.py](../unidream/cli/statistical_gate.py) | Run the development-only statistical robustness gate on JSON input. | build_parser, main |
| [unidream/cli/train.py](../unidream/cli/train.py) | Strict CLI entrypoint for the current WM -> BC -> AC -> Test pipeline. | build_parser, main |
| [unidream/cli/verify_data_quality.py](../unidream/cli/verify_data_quality.py) | Run the development-cache data and feature quality gate. | build_parser, main |
| [unidream/data/__init__.py](../unidream/data/__init__.py) | package初期化・公開import | — |
| [unidream/data/availability_contract.py](../unidream/data/availability_contract.py) | Fail-closed availability contracts for v4 training data. | AvailabilityContractError, AvailabilitySelection, required_availability_columns, validate_availability, row_eligibility |
| [unidream/data/cache_v4.py](../unidream/data/cache_v4.py) | Fail-closed schema v4 cache with a separate availability sidecar. | CacheV4Error, cache_v4_paths, build_v4_metadata, validate_cache_v4, write_cache_v4, load_cache_v4 |
| [unidream/data/d1_signed_flow.py](../unidream/data/d1_signed_flow.py) | Leak-safe D1 signed-flow acquisition and 15-minute feature assembly. | aggtrade_archive_url, download_d1_kline_month, estimate_aggtrade_archive_storage, build_d1_features, d1_bar_ledger_records, classify_archive_revisions ほか2定義 |
| [unidream/data/dataset.py](../unidream/data/dataset.py) | データローダーモジュール. | WFOSplit, FeatureArray, get_wfo_splits, SequenceDataset, WFODataset |
| [unidream/data/download.py](../unidream/data/download.py) | Binance OHLCV データ取得モジュール. | fetch_binance_ohlcv, fetch_binance_spot_taker_proxy, fetch_mark_price_klines, fetch_funding_rate, fetch_open_interest_hist, fetch_taker_buy_sell_volume ほか1定義 |
| [unidream/data/features.py](../unidream/data/features.py) | 特徴量エンジニアリングモジュール. | days_to_bars, log_returns, compute_rsi, compute_macd, compute_atr, rolling_zscore ほか13定義 |
| [unidream/data/official_v4_sources.py](../unidream/data/official_v4_sources.py) | Official Binance source readers used by the schema-v4 rebuild. | OfficialSourceError, assert_official_url, official_archive_url, fetch_archive_month, fetch_spot_rest_window, probe_official_sources ほか1定義 |
| [unidream/data/oracle.py](../unidream/data/oracle.py) | Hindsight Oracle モジュール. | hindsight_oracle_dp, compute_net_returns, conditional_oracle_teacher_path, hindsight_upper_bound_path, oracle_positions, smooth_aim_positions ほか3定義 |
| [unidream/data/rebuild_v4.py](../unidream/data/rebuild_v4.py) | Regenerate a schema-v4 cache from official Binance raw data. | recover_spot_gaps, build_full_grid_availability, compute_v4_frames, rebuild_official_v4_frames |
| [unidream/data/window_quality.py](../unidream/data/window_quality.py) | Gap-aware sequence-window eligibility checks. | WindowQualityError, valid_sequence_starts, window_is_gap_free |
| [unidream/device.py](../unidream/device.py) | device検出・優先順位・利用可否検証 | mps_available, preferred_device, resolve_device, add_device_argument |
| [unidream/eval/__init__.py](../unidream/eval/__init__.py) | package初期化・公開import | — |
| [unidream/eval/action_execution.py](../unidream/eval/action_execution.py) | Single source of truth for the conditional-Oracle action contract. | ActionExecutionContract, action_execution_contract_from_config, configured_action_execution_contract, ActionExecutionTrajectory, validate_eligibility_masks, ActionBlockMasks ほか14定義 |
| [unidream/eval/alpha_attribution.py](../unidream/eval/alpha_attribution.py) | Leak-aware alpha attribution and saved-artifact diagnostics. | FoldSeries, sha256_file, right_exclusive_mask, load_timeseries_artifact, data_fingerprint_from_series, feature_coverage_for_fold ほか12定義 |
| [unidream/eval/backtest.py](../unidream/eval/backtest.py) | バックテストモジュール. | validate_bound_action_execution_contract, validate_execution_delay, align_execution_path, BacktestMetrics, compute_costs, compute_pnl ほか11定義 |
| [unidream/eval/constant_exposure_diagnostic.py](../unidream/eval/constant_exposure_diagnostic.py) | Development-only audit of low-frequency constant exposure effects. | ConstantExposureData, validate_wave3d_folds, load_constant_exposure_data, configured_execution_delay, run_constant_exposure_diagnostic |
| [unidream/eval/data_quality.py](../unidream/eval/data_quality.py) | Development-cache data and feature quality gates. | DataQualityError, inspect_feature_contract, validate_feature_contract, external_coverage, feature_coverage, annual_external_coverage ほか7定義 |
| [unidream/eval/forecast_context_tournament.py](../unidream/eval/forecast_context_tournament.py) | Wave3C causal context/forecast tournament for Plan011 development folds. | ContextRegressionCandidate, DownsideClassifierCandidate, context_eligibility_mask, future_target_eligibility_mask, build_causal_context, context_prefix_is_causal ほか14定義 |
| [unidream/eval/forecast_tournament.py](../unidream/eval/forecast_tournament.py) | Leak-safe low-cost forecast/timing tournament for Plan011. | DevelopmentData, PolicyParams, FittedCandidate, validate_requested_folds, load_development_data, feature_ablation_columns ほか8定義 |
| [unidream/eval/gap_recovery.py](../unidream/eval/gap_recovery.py) | Read-only recovery probes for missing development-cache spot bars. | index_digest, detect_gaps, probe_official_gap_recovery, write_gap_recovery_jsonl, render_gap_recovery_markdown |
| [unidream/eval/pbo.py](../unidream/eval/pbo.py) | PBO (Probability of Backtest Overfitting) & Deflated Sharpe Ratio モジュール. | compute_pbo, estimate_return_moments, deflated_sharpe, deflated_sharpe_pvalue, filter_by_pbo |
| [unidream/eval/policy_stats.py](../unidream/eval/policy_stats.py) | Policy statistics shared by training, evaluation, and reporting. | action_stats, format_action_stats |
| [unidream/eval/regime.py](../unidream/eval/regime.py) | HMM レジーム検出モジュール. | RegimeDetector, regime_metrics, print_regime_report |
| [unidream/eval/selector.py](../unidream/eval/selector.py) | Validation policy selection for the current B&H-relative actor. | benchmark_positions, policy_score, selector_config, selector_candidate, candidate_to_text, select_policy_candidate |
| [unidream/eval/statistical_gate.py](../unidream/eval/statistical_gate.py) | Development-only statistical robustness gate. | DevelopmentFold, CandidatePath, StressCase, StatisticalGateConfig, bootstrap_confidence_intervals, fold_sign_test ほか6定義 |
| [unidream/eval/wfo.py](../unidream/eval/wfo.py) | Walk-Forward Optimization 評価モジュール. | walk_forward_eval, aggregate_wfo_results, collect_all_test_returns |
| [unidream/experiments/__init__.py](../unidream/experiments/__init__.py) | package初期化・公開import | — |
| [unidream/experiments/ac_stage.py](../unidream/experiments/ac_stage.py) | AC trainer構築・学習・validationとcurriculumの実行 | build_ac_trainer, run_ac_stage |
| [unidream/experiments/action_primitives.py](../unidream/experiments/action_primitives.py) | Canonical deterministic P1 action primitive producer and validator. | ActionPrimitiveContractError, ActionPrimitiveImplementationBlocked, canonical_action_primitive_content_bytes, action_primitive_content_sha256, canonical_action_primitive_payload_bytes, action_primitive_payload_sha256 ほか12定義 |
| [unidream/experiments/alpha_dd_data.py](../unidream/experiments/alpha_dd_data.py) | Audited monthly Spot BTCUSDT 15-minute archive acquisition. | AcquisitionError, MonthResult, sha256_file, month_values, expected_bar_grid, artifact_paths ほか3定義 |
| [unidream/experiments/alpha_dd_features.py](../unidream/experiments/alpha_dd_features.py) | Causal feature variant for the registered AlphaEx/MaxDD search. | make_features |
| [unidream/experiments/alpha_dd_ml.py](../unidream/experiments/alpha_dd_ml.py) | Preregistered gap-aware ML-only continuation; prior experiments stay immutable. | registered_candidates, main |
| [unidream/experiments/alpha_dd_search.py](../unidream/experiments/alpha_dd_search.py) | Fixed, causal policy-family research for joint B&H alpha / drawdown targets. | CandidateUnavailable, digest, file_digest, write_json, Candidate, candidate_universe ほか12定義 |
| [unidream/experiments/bc_setup.py](../unidream/experiments/bc_setup.py) | ActorとBC学習用補助値の準備 | prepare_bc_setup |
| [unidream/experiments/bc_stage.py](../unidream/experiments/bc_stage.py) | BC trainer構築・overlay教師適用・学習 | build_bc_trainer, run_bc_stage |
| [unidream/experiments/checkpoint_eval.py](../unidream/experiments/checkpoint_eval.py) | Shared checkpoint loading helpers for leak-free fold evaluation. | CheckpointRunSpec, parse_checkpoint_run_spec, load_fold_model_context, load_actor_state_checkpoint, load_inference_run_context |
| [unidream/experiments/checkpointing.py](../unidream/experiments/checkpointing.py) | Checkpoint persistence and provenance helpers for reproducible runs. | atomic_text_write, atomic_torch_save, snapshot_actor_inference_settings, apply_actor_inference_settings, checkpoint_metadata_for_fold, validate_checkpoint_metadata |
| [unidream/experiments/chronological_oof.py](../unidream/experiments/chronological_oof.py) | Leak-safe chronological OOF contracts for the conditional experiment path. | ChronologicalOOFError, ConditionalPathBlocked, ConditionalOOFArtifactError, OOFOrigin, strict_bool_array, strict_bool_value ほか16定義 |
| [unidream/experiments/conditional_teacher.py](../unidream/experiments/conditional_teacher.py) | Authenticated causal teacher context for the conditional WM/BC/AC path. | ConditionalTeacherContext, build_conditional_teacher_context, require_authenticated_conditional_teacher_context |
| [unidream/experiments/export_alpha_trend_fixture.py](../unidream/experiments/export_alpha_trend_fixture.py) | Export real BNB history-to-target parity cases from locked research results. | export, main |
| [unidream/experiments/fire_diagnostics.py](../unidream/experiments/fire_diagnostics.py) | 実装の公開定義を参照 | FireDangerDiagnostics, empty_fire_danger_diagnostics, evaluate_fire_danger_diagnostics |
| [unidream/experiments/fold_inputs.py](../unidream/experiments/fold_inputs.py) | fold教師・regime・在庫source契約の準備 | TeacherInventoryContractError, validate_current_inventory_source, current_inventory_from_replay, require_allowed_current_inventory, prepare_fold_inputs |
| [unidream/experiments/fold_runtime.py](../unidream/experiments/fold_runtime.py) | fold学習budgetの解決 | resolve_ac_max_steps |
| [unidream/experiments/fold_training.py](../unidream/experiments/fold_training.py) | 1foldのWM→BC→AC→selector→test順序制御 | run_fold |
| [unidream/experiments/logging.py](../unidream/experiments/logging.py) | Small logging helpers shared by experiment stages. | log_timestamp |
| [unidream/experiments/m2.py](../unidream/experiments/m2.py) | B&H基準・collapse guard・M2 scorecard | benchmark_position_value, directional_collapse, collapse_guard, goal_cfg, m2_scorecard, format_m2_scorecard |
| [unidream/experiments/oracle_post.py](../unidream/experiments/oracle_post.py) | 教師のoutcome edgeとpostprocess | teacher_outcome_edge, apply_oracle_postprocess |
| [unidream/experiments/oracle_stage.py](../unidream/experiments/oracle_stage.py) | base Oracle生成 | compute_base_oracle |
| [unidream/experiments/oracle_teacher.py](../unidream/experiments/oracle_teacher.py) | 教師モードに応じたpath生成 | compute_teacher_oracle |
| [unidream/experiments/overlay_teacher.py](../unidream/experiments/overlay_teacher.py) | 低頻度・risk/edge条件付きB&H overlay教師 | benchmark_overlay_teacher_enabled, apply_benchmark_overlay_teacher, describe_benchmark_overlay_teacher |
| [unidream/experiments/p0_integration_audit.py](../unidream/experiments/p0_integration_audit.py) | Small, reproducible integration audit for the P0 contracts. | V4ContractInputs, load_v4_contract_inputs, audit_contract_paths, audit_v4_cache, markdown_report, main |
| [unidream/experiments/p1_action_artifact.py](../unidream/experiments/p1_action_artifact.py) | Atomic, externally bound persistence for P1 action primitive artifacts. | P1ActionArtifactError, P1MBBMetricInput, P1MBBActionInput, LoadedP1ActionArtifact, save_p1_action_artifact, is_authenticated_loaded_action_artifact ほか2定義 |
| [unidream/experiments/p1_conditional_wm_bc_ac.py](../unidream/experiments/p1_conditional_wm_bc_ac.py) | Strict chronological OOF -> WM -> BC -> AC diagnostic. | ConditionalPipelineError, run_conditional_wm_bc_ac, main |
| [unidream/experiments/p1_formal_forecast_wm_bc_ac.py](../unidream/experiments/p1_formal_forecast_wm_bc_ac.py) | Connect the registered ForecastActionSource to a real WM -> BC -> AC run. | FormalForecastPipelineError, run_formal_forecast_wm_bc_ac, main |
| [unidream/experiments/p1_formal_results.py](../unidream/experiments/p1_formal_results.py) | Execute the amended, validation-only P1 result calculation. | FormalState, run_formal, main |
| [unidream/experiments/p1_mbb.py](../unidream/experiments/p1_mbb.py) | Exact, paired, non-circular moving-block bootstrap for the P1 protocol. | P1MBBError, P1MBBImplementationBlocked, derive_p1_seed, draw_non_circular_mbb_starts, materialize_non_circular_mbb_indices, draw_non_circular_mbb_indices ほか41定義 |
| [unidream/experiments/p1_recovery_prereg.py](../unidream/experiments/p1_recovery_prereg.py) | Fail-closed loader for the pre-registered P1 recovery protocol. | P1PreregistrationError, canonical_manifest_sha256, canonical_json_sha256, exact_file_sha256, validate_pinned_artifacts, validate_fixed_manifest ほか1定義 |
| [unidream/experiments/p1_recovery_runner.py](../unidream/experiments/p1_recovery_runner.py) | Execution-free core for the pre-registered P1 recovery run. | P1RunnerError, P1OuterReportBlocked, validate_15m_timestamps, timestamp_edge_mask, ManifestEcho, OuterReportSpec ほか34定義 |
| [unidream/experiments/p1_result_registry.py](../unidream/experiments/p1_result_registry.py) | Authenticated execution registry for the frozen P1 validation operation. | P1ResultRegistryError, P1ResultRegistry, load_p1_result_registry |
| [unidream/experiments/p1_s3_outer_report.py](../unidream/experiments/p1_s3_outer_report.py) | Run the fixed S3 outer report as a terminal, report-only calculation. | S3OuterReportError, run_s3_outer_report, main |
| [unidream/experiments/p1_s3_rolling_shadow.py](../unidream/experiments/p1_s3_rolling_shadow.py) | Report-only rolling BTC diagnostics and an orderless paper shadow. | RollingShadowError, run_s3_rolling_shadow, main |
| [unidream/experiments/p1_statistical_gates.py](../unidream/experiments/p1_statistical_gates.py) | Fixed Holm and Wilson gates for the authenticated P1 comparison family. | P1StatisticalGateError, HolmRow, HolmFamily, WilsonInterval, holm_bonferroni_fixed_family, wilson_score_interval ほか1定義 |
| [unidream/experiments/p1_validation_forecast.py](../unidream/experiments/p1_validation_forecast.py) | Fixed registered P1 validation forecast execution and persistence. | P1ForecastError, P1ForecastOuterBlocked, ValidationScenarioSpec, P1ForecastContract, authenticate_p1_forecast_contract, registered_validation_arm_keys ほか11定義 |
| [unidream/experiments/policy_fire.py](../unidream/experiments/policy_fire.py) | 発火条件別の方策比較・forward診断 | FireMetrics, predict_with_policy_flags, forward_return_mean, forward_incremental_mean, evaluate_fire_metrics, format_fire_metrics |
| [unidream/experiments/predictive_state.py](../unidream/experiments/predictive_state.py) | WM補助headの標準化・OOF stateのstrict検証 | build_wm_predictive_state_bundle |
| [unidream/experiments/regime_runtime.py](../unidream/experiments/regime_runtime.py) | train-fit HMMとcausal regime確率 | fit_fold_regimes |
| [unidream/experiments/robust_overlay.py](../unidream/experiments/robust_overlay.py) | Fixed multiscale, low-turnover B&H-relative overlay experiment. | build_targets, run, stage_cfg_folds, main |
| [unidream/experiments/run_config.py](../unidream/experiments/run_config.py) | strict run設定・directory・manifest・fingerprint | TrainingRunConfig, load_training_run_config, configure_determinism, prepare_run_directory, data_fingerprint, config_fingerprint ほか4定義 |
| [unidream/experiments/runtime.py](../unidream/experiments/runtime.py) | 設定・cost・cache・v4/P1入力検証・seed | resolve_cache_pair, read_optional_parquet, read_extra_series_caches, cache_quality_status, V4RuntimeInputError, validate_v4_runtime_inputs ほか5定義 |
| [unidream/experiments/test_stage.py](../unidream/experiments/test_stage.py) | 選択済み方策のtest評価・帰属・診断 | run_test_stage |
| [unidream/experiments/train_app.py](../unidream/experiments/train_app.py) | データ取得・cache選択・WFO run・summary | resolve_training_cache_selection, run_training_app |
| [unidream/experiments/train_pipeline.py](../unidream/experiments/train_pipeline.py) | fold loopとseed・datasetの生成 | run_wfo_folds |
| [unidream/experiments/train_reporting.py](../unidream/experiments/train_reporting.py) | 簡易過適合診断・scorecard集約・summary | compute_overfitting_diagnostics, aggregate_scorecards, print_training_summary |
| [unidream/experiments/transition_advantage.py](../unidream/experiments/transition_advantage.py) | transition/route教師とhindsight advantage | TransitionAdvantageConfig, transition_unit_cost, config_from_dict, transition_classes, route_ids_from_classes, compute_route_targets ほか6定義 |
| [unidream/experiments/val_selector_stage.py](../unidream/experiments/val_selector_stage.py) | validation推論grid評価と設定確定 | run_val_selector_stage |
| [unidream/experiments/verify_alpha_trend_space.py](../unidream/experiments/verify_alpha_trend_space.py) | Verify a pinned public HF revision, both inference routes, and real Spot data. | verify, main |
| [unidream/experiments/wfo_runtime.py](../unidream/experiments/wfo_runtime.py) | WFO生成・fold選択 | build_wfo_splits, select_wfo_splits, select_configured_wfo_splits |
| [unidream/experiments/wm_stage.py](../unidream/experiments/wm_stage.py) | WM構築・学習dataset・checkpointの生成 | prepare_world_model_stage |
| [unidream/world_model/__init__.py](../unidream/world_model/__init__.py) | package初期化・公開import | — |
| [unidream/world_model/encoder.py](../unidream/world_model/encoder.py) | 観測エンコーダモジュール. | symlog, symexp, OneHotDist, ObsEncoder, ObsDecoder |
| [unidream/world_model/ensemble.py](../unidream/world_model/ensemble.py) | Ensemble World Model モジュール. | EnsembleWorldModel |
| [unidream/world_model/train_wm.py](../unidream/world_model/train_wm.py) | 世界モデル学習エントリポイント. | TargetGradientCoverageError, world_model_action_context, world_model_uses_dataset_actions, IDMHead, ReturnHead, RegimeHead ほか2定義 |
| [unidream/world_model/transformer.py](../unidream/world_model/transformer.py) | Block-Causal Transformer 世界モデル. | make_twohot_bins, twohot_encode, twohot_decode, twohot_loss, SinusoidalPE, CausalTransformerBlock ほか1定義 |

## 設定一覧

mainlineはAGENTS.md指定の3ファイル。その他は研究設定で、学習CLIで全てそのまま動くという意味ではない。

| ファイル | 用途 | symbol・fold等 |
|---|---|---|
| [configs/alpha_dd_bnb_20260905.yaml](../configs/alpha_dd_bnb_20260905.yaml) | 独立研究runner | BNBUSDT; folds=stages参照 |
| [configs/alpha_dd_eth_20260905.yaml](../configs/alpha_dd_eth_20260905.yaml) | 独立研究runner | ETHUSDT; folds=stages参照 |
| [configs/alpha_dd_ml_20260905.yaml](../configs/alpha_dd_ml_20260905.yaml) | 独立研究runner | BTCUSDT; folds=stages参照 |
| [configs/alpha_dd_research_20260905.yaml](../configs/alpha_dd_research_20260905.yaml) | 独立研究runner | BTCUSDT（取得契約参照）; folds=stages参照 |
| [configs/experiments/plan013_ac_release_baseline_fold0.yaml](../configs/experiments/plan013_ac_release_baseline_fold0.yaml) | 学習probe | BTCUSDT; folds=[0]; WM context=未指定（fallback oracle） |
| [configs/experiments/plan013_ac_release_baseline_folds2_8.yaml](../configs/experiments/plan013_ac_release_baseline_folds2_8.yaml) | 学習probe | BTCUSDT; folds=[2, 8]; WM context=未指定（fallback oracle） |
| [configs/experiments/plan013_ac_release_gradual_folds0_2_8.yaml](../configs/experiments/plan013_ac_release_gradual_folds0_2_8.yaml) | 学習probe | BTCUSDT; folds=[0, 2, 8]; WM context=未指定（fallback oracle） |
| [configs/experiments/plan013_ac_release_staged_fold0.yaml](../configs/experiments/plan013_ac_release_staged_fold0.yaml) | 学習probe | BTCUSDT; folds=[0]; WM context=未指定（fallback oracle） |
| [configs/experiments/plan014_ac_guarded_stage_select_folds0_2_8.yaml](../configs/experiments/plan014_ac_guarded_stage_select_folds0_2_8.yaml) | 学習probe | BTCUSDT; folds=[0, 2, 8]; WM context=未指定（fallback oracle） |
| [configs/experiments/plan015_ac_residual_adapter_folds0_2_8.yaml](../configs/experiments/plan015_ac_residual_adapter_folds0_2_8.yaml) | 学習probe | BTCUSDT; folds=[0, 2, 8]; WM context=未指定（fallback oracle） |
| [configs/plan011_overlay_actor_v31_holdout.yaml](../configs/plan011_overlay_actor_v31_holdout.yaml) | 現行mainline | BTCUSDT; folds=[15, 16, 17, 18, 19, 20, 21, 22, 23]; WM context=actionless |
| [configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml](../configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml) | 現行mainline | BTCUSDT; folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; WM context=actionless |
| [configs/plan012_benchmark_absolute_constraint_probe.yaml](../configs/plan012_benchmark_absolute_constraint_probe.yaml) | 学習probe | BTCUSDT; folds=[0, 2, 8]; WM context=未指定（fallback oracle） |
| [configs/robust_overlay_v1.yaml](../configs/robust_overlay_v1.yaml) | 独立研究runner | BTCUSDT; folds=stages参照 |
| [configs/trading.yaml](../configs/trading.yaml) | 現行mainline | BTCUSDT; folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; WM context=未指定（fallback oracle） |

## テスト一覧

定義数はAST上の `test_*` method/function数。subTest等を含む実行件数とは別。参照moduleは静的importの一部で、網羅的coverage率を示さない。

| テストファイル | test定義数 | 主な参照module |
|---|---:|---|
| [test_ac_curriculum.py](../tests/test_ac_curriculum.py) | 5 | actor_critic.actor, actor_critic.imagination_ac, experiments.ac_stage, experiments.checkpoint_eval |
| [test_action_execution_contract.py](../tests/test_action_execution_contract.py) | 41 | data.oracle, eval.action_execution, eval.backtest, experiments.transition_advantage |
| [test_action_primitive_contract.py](../tests/test_action_primitive_contract.py) | 16 | eval.action_execution, experiments.action_primitives |
| [test_alpha_attribution.py](../tests/test_alpha_attribution.py) | 13 | eval.alpha_attribution, eval.backtest, world_model.train_wm |
| [test_alpha_dd_data.py](../tests/test_alpha_dd_data.py) | 11 | data.d1_signed_flow, data.official_v4_sources, experiments.alpha_dd_data |
| [test_alpha_dd_features.py](../tests/test_alpha_dd_features.py) | 5 | experiments.alpha_dd_features |
| [test_alpha_dd_ml.py](../tests/test_alpha_dd_ml.py) | 1 | experiments.alpha_dd_features, experiments.alpha_dd_ml, experiments.alpha_dd_search |
| [test_alpha_dd_search.py](../tests/test_alpha_dd_search.py) | 10 | experiments.alpha_dd_search |
| [test_availability_contract.py](../tests/test_availability_contract.py) | 2 | data.dataset |
| [test_backtest_final_excess.py](../tests/test_backtest_final_excess.py) | 2 | eval.backtest |
| [test_cache_v4.py](../tests/test_cache_v4.py) | 7 | data.cache_v4 |
| [test_chronological_oof_teacher.py](../tests/test_chronological_oof_teacher.py) | 12 | experiments.chronological_oof |
| [test_conditional_oof_artifact.py](../tests/test_conditional_oof_artifact.py) | 18 | eval.action_execution, experiments.chronological_oof, experiments.predictive_state |
| [test_conditional_stage_gates.py](../tests/test_conditional_stage_gates.py) | 3 | experiments.ac_stage, experiments.bc_stage, experiments.chronological_oof, experiments.wm_stage |
| [test_constant_exposure_diagnostic.py](../tests/test_constant_exposure_diagnostic.py) | 4 | data.dataset, eval.constant_exposure_diagnostic |
| [test_d1_signed_flow.py](../tests/test_d1_signed_flow.py) | 9 | cli.acquire_d1_signed_flow, data.d1_signed_flow, data.official_v4_sources |
| [test_data_quality.py](../tests/test_data_quality.py) | 9 | cli.verify_data_quality, eval.data_quality |
| [test_forecast_context_tournament.py](../tests/test_forecast_context_tournament.py) | 9 | data.dataset, eval.backtest, eval.forecast_context_tournament, eval.forecast_tournament |
| [test_forecast_tournament.py](../tests/test_forecast_tournament.py) | 12 | data.dataset, eval.backtest, eval.forecast_tournament |
| [test_gap_recovery.py](../tests/test_gap_recovery.py) | 5 | eval.gap_recovery |
| [test_leak_discipline.py](../tests/test_leak_discipline.py) | 7 | data.features, eval.backtest, eval.regime |
| [test_official_v4_sources.py](../tests/test_official_v4_sources.py) | 4 | data.official_v4_sources |
| [test_p0_integration_audit.py](../tests/test_p0_integration_audit.py) | 3 | data.cache_v4, data.oracle, eval.action_execution, experiments.chronological_oof ほか2 |
| [test_p1_action_artifact.py](../tests/test_p1_action_artifact.py) | 3 | eval.action_execution, experiments.action_primitives, experiments.p1_action_artifact |
| [test_p1_authenticated_action_adapter.py](../tests/test_p1_authenticated_action_adapter.py) | 2 | eval.action_execution, experiments, experiments.action_primitives |
| [test_p1_authenticated_action_artifact.py](../tests/test_p1_authenticated_action_artifact.py) | 8 | eval.action_execution, experiments.action_primitives, experiments.p1_action_artifact, experiments.p1_mbb |
| [test_p1_exact_mbb.py](../tests/test_p1_exact_mbb.py) | 22 | experiments.action_primitives, experiments.p1_mbb |
| [test_p1_mbb_file_binding.py](../tests/test_p1_mbb_file_binding.py) | 4 | eval.action_execution, experiments.action_primitives, experiments.p1_action_artifact, experiments.p1_mbb |
| [test_p1_recovery_preregistration.py](../tests/test_p1_recovery_preregistration.py) | 13 | experiments.p1_recovery_prereg |
| [test_p1_recovery_runner.py](../tests/test_p1_recovery_runner.py) | 20 | data.cache_v4, experiments |
| [test_p1_result_registry.py](../tests/test_p1_result_registry.py) | 3 | experiments |
| [test_p1_s3_outer_report.py](../tests/test_p1_s3_outer_report.py) | 4 | experiments.p1_recovery_runner, experiments.p1_s3_outer_report |
| [test_p1_s3_rolling_shadow.py](../tests/test_p1_s3_rolling_shadow.py) | 4 | experiments.p1_s3_rolling_shadow |
| [test_p1_statistical_gates.py](../tests/test_p1_statistical_gates.py) | 3 | experiments |
| [test_p1_validation_forecast_artifact.py](../tests/test_p1_validation_forecast_artifact.py) | 13 | experiments |
| [test_p1_validation_forecast_spec.py](../tests/test_p1_validation_forecast_spec.py) | 4 | experiments |
| [test_plan011_fold_trade_plots.py](../tests/test_plan011_fold_trade_plots.py) | 3 | cli.plot_plan011_fold_trades |
| [test_policy_family_comparison.py](../tests/test_policy_family_comparison.py) | 2 | cli.compare_policy_families |
| [test_rebuild_v4.py](../tests/test_rebuild_v4.py) | 9 | cli.rebuild_official_v4_cache, data.official_v4_sources, data.rebuild_v4 |
| [test_reproducible_artifacts.py](../tests/test_reproducible_artifacts.py) | 4 | experiments.ac_stage, experiments.checkpointing, experiments.runtime |
| [test_robust_overlay.py](../tests/test_robust_overlay.py) | 3 | experiments.robust_overlay |
| [test_runtime_v4.py](../tests/test_runtime_v4.py) | 11 | data.cache_v4, data.dataset, experiments, experiments.runtime ほか1 |
| [test_statistical_gate.py](../tests/test_statistical_gate.py) | 9 | cli.statistical_gate, eval.statistical_gate |
| [test_strict_training_config.py](../tests/test_strict_training_config.py) | 6 | cli.train, experiments.run_config |
| [test_teacher_inventory_contract.py](../tests/test_teacher_inventory_contract.py) | 10 | experiments.chronological_oof, experiments.fold_inputs, experiments.predictive_state |
| [test_window_quality.py](../tests/test_window_quality.py) | 7 | data.window_quality |
| [test_world_model_gate0_action_context.py](../tests/test_world_model_gate0_action_context.py) | 3 | experiments, world_model.train_wm |
| [test_world_model_gate0_validation.py](../tests/test_world_model_gate0_validation.py) | 2 | data.dataset, world_model |
| [test_world_model_target_coverage.py](../tests/test_world_model_target_coverage.py) | 4 | data.dataset, world_model.train_wm |

## 調査で確認した主要差分

- tradingとnamed v31のWM action context・forecast delay設定の差。
- sequence64に対するfuture-only h64/utility64の有効教師数0。
- v3構造検査とavailability品質検査は別。v4伝播実装と全評価経路の統一も別。
- mainline delay0、forecast診断delay1、conditionalのh4/delay1は別契約。
- mainline対数PnL会計とAlpha/DD cash/units会計の差。
- 全reject時にもselectorが候補を返すfallback。
- README等のcurrent記述、BNB v2配備履歴、10月研究集約は時点・構成が異なる。
- 通常のcheckpoint削除方針と、明示追跡された研究checkpointは両方存在する。
- replay loaderのmetadata照合とsemantic内容hashの完全性照合は別で、後者のmanifest一致は強制されていない。

これらは本書作成時の仕様記録であり、モデル設定や経済結果の修正は実施していない。
