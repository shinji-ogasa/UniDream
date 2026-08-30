import copy
import unittest
from dataclasses import replace
from types import MappingProxyType
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from unidream.data.cache_v4 import MODEL_FEATURE_COLUMNS, REQUIRED_AVAILABILITY_COLUMNS
from unidream.experiments import p1_recovery_runner as runner


class P1RecoveryRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is a deterministic contract fixture only; no OOF, scoring, or
        # outer operation is executed by this test module.
        cls.dataset = runner.build_synthetic_dataset(20260830, beta=0.004)

    def test_authenticated_manifest_is_the_only_plan_source(self):
        manifest = runner.load_runner_manifest()
        self.assertIsInstance(manifest, MappingProxyType)
        self.assertEqual(runner.manifest_echo()["manifest_sha256"], runner.REGISTERED_MANIFEST_SHA256)
        with self.assertRaises(runner.P1RunnerError):
            runner.manifest_echo(dict(manifest))
        with self.assertRaises(runner.P1RunnerError):
            runner.build_runner_plan(dict(manifest))

        forged = dict(manifest)
        forged["results_observed"] = True
        with mock.patch.object(runner, "load_fixed_manifest", return_value=forged):
            with self.assertRaises(runner.P1RunnerError):
                runner.load_runner_manifest()

    def test_exact_synthetic_standard_normal_stream_and_gap_choice(self):
        seed = 20260830
        base = runner.generate_synthetic_base(seed)
        rng = np.random.default_rng(seed + 100)
        self.assertEqual(base.z_raw[0], rng.standard_normal())
        np.testing.assert_array_equal(base.xi, rng.standard_normal(runner.SYNTHETIC_RAW_ROWS - 1))
        np.testing.assert_array_equal(
            base.noise_features,
            rng.standard_normal((runner.SYNTHETIC_RAW_ROWS, runner.FEATURE_DIMENSION - 1)),
        )
        np.testing.assert_array_equal(base.epsilon, rng.standard_normal(runner.SYNTHETIC_RAW_ROWS))

        for source, offset in {"spot_bar_observed": 11, "funding_rate_available": 23, "mark_close_available": 37}.items():
            gap_rng = np.random.default_rng(seed + 50000 + offset)
            relative = gap_rng.choice(
                runner.SYNTHETIC_ROWS - 2 - runner.SYNTHETIC_BURN_IN,
                size=40,
                replace=False,
                shuffle=True,
            )
            expected = tuple((relative + runner.SYNTHETIC_BURN_IN).tolist())
            self.assertEqual(base.gap_starts[source], expected)

    def test_target_timestamp_requires_h_edges_but_not_following_edge(self):
        timestamps = np.arange(
            np.datetime64("2020-01-01T00:00", "m"),
            np.datetime64("2020-01-01T01:45", "m"),
            np.timedelta64(15, "m"),
        )
        # Break only edge 4 -> 5 while retaining a strictly ordered grid.
        timestamps[5:] += np.timedelta64(15, "m")
        returns = np.zeros(len(timestamps), dtype=np.float64)
        spot = np.ones(len(timestamps), dtype=np.bool_)
        _, mask, end = runner.build_target_arrays(
            returns,
            spot,
            horizons=(4,),
            timestamps=timestamps,
        )
        self.assertTrue(mask[0, 0])  # edges 0->1 through 3->4 only
        self.assertFalse(mask[1, 0])  # its window crosses the broken edge
        self.assertEqual(end[0, 0], 5)

    def test_context_timestamp_uses_current_inclusive_window(self):
        timestamps = np.arange(
            np.datetime64("2020-01-01T00:00", "m"),
            np.datetime64("2020-01-01T17:30", "m"),
            np.timedelta64(15, "m"),
        )
        timestamps[64:] += np.timedelta64(15, "m")
        features = np.zeros((len(timestamps), runner.FEATURE_DIMENSION), dtype=np.float64)
        availability = {
            name: np.ones(len(timestamps), dtype=np.bool_)
            for name in REQUIRED_AVAILABILITY_COLUMNS
        }
        mask = runner.build_context_mask(features, availability, timestamps=timestamps)
        self.assertFalse(mask[62])
        self.assertTrue(mask[63])  # [0, 63] does not include edge 63->64
        self.assertFalse(mask[64])

    def test_dataset_rejects_forged_target_end_value_and_false_nan(self):
        data = self.dataset
        forged_end = np.array(data.target_end, copy=True)
        forged_end[0, 0] += 1
        forged = replace(data, target_end=forged_end)
        with self.assertRaises(runner.P1RunnerError):
            runner.fit_model_at_origin(forged, "zero_return", 20000, 1)

        forged_targets = np.array(data.targets, copy=True)
        false_positions = np.argwhere(~np.asarray(data.target_mask))
        row, column = false_positions[0]
        forged_targets[row, column] = 0.0
        forged = replace(data, targets=forged_targets)
        with self.assertRaises(runner.P1RunnerError):
            runner.fit_model_at_origin(forged, "zero_return", 20000, 1)

    def test_min_history_blocks_baselines_and_binary_tasks_are_separate(self):
        small_continuous = runner.fit_model_at_origin(
            self.dataset, "zero_return", 64, 4, task="continuous"
        )
        small_binary = runner.fit_model_at_origin(
            self.dataset, "zero_return", 64, 4, task="binary"
        )
        self.assertEqual(small_continuous.status, "N/A")
        self.assertEqual(small_binary.status, "N/A")
        self.assertIn(str(runner.MIN_HISTORY_ROWS), small_binary.reason)

        continuous = runner.fit_model_at_origin(
            self.dataset,
            "zero_return",
            20000,
            4,
            task="continuous",
            prediction_range=(20000, 20010),
        )
        binary = runner.fit_model_at_origin(
            self.dataset,
            "zero_return",
            20000,
            4,
            task="binary",
            prediction_range=(20000, 20010),
        )
        self.assertEqual(continuous.task, "continuous")
        self.assertEqual(binary.task, "binary")
        self.assertTrue(np.all(continuous.predictions[continuous.prediction_mask] == 0.0))
        self.assertTrue(np.all(binary.predictions[binary.prediction_mask] == 0.5))

        persistence = runner.fit_model_at_origin(
            self.dataset,
            "persistence_last_observed",
            20000,
            4,
            task="binary",
            prediction_range=(20000, 20010),
        )
        expected = np.where(
            self.dataset.returns[persistence.prediction_mask] > 0.0,
            1.0 - runner.PROBABILITY_CLIP_EPS,
            runner.PROBABILITY_CLIP_EPS,
        )
        np.testing.assert_array_equal(persistence.predictions[persistence.prediction_mask], expected)

    def test_production_fit_rejects_pre_origin_empty_and_nonempty_coverage_bypass(self):
        with self.assertRaises(runner.P1RunnerError):
            runner.fit_model_at_origin(
                self.dataset,
                "zero_return",
                20000,
                4,
                task="continuous",
                prediction_range=(19999, 20010),
            )
        with self.assertRaises(runner.P1RunnerError):
            runner.fit_model_at_origin(
                self.dataset,
                "zero_return",
                20000,
                4,
                task="continuous",
                prediction_range=(20000, 20000),
            )
        # The range is non-empty but no h4 target fits before its end.
        with self.assertRaises(runner.P1RunnerError):
            runner.fit_model_at_origin(
                self.dataset,
                "zero_return",
                20000,
                4,
                task="continuous",
                prediction_range=(20000, 20001),
            )

    def test_production_fit_rejects_nonfinite_returns_and_nonfixed_purge(self):
        returns = np.array(self.dataset.returns, copy=True)
        returns[20000] = np.nan
        malformed = self.dataset.with_returns(returns)
        with self.assertRaises(runner.P1RunnerError):
            runner.fit_model_at_origin(
                malformed,
                "persistence_last_observed",
                20000,
                4,
                task="binary",
                prediction_range=(20000, 20010),
            )
        with self.assertRaises(runner.P1RunnerError):
            runner.train_mask_for_origin(
                self.dataset,
                20000,
                4,
                purge_bars=0,
            )

    def test_timestamp_free_builders_are_explicitly_fixture_only(self):
        returns = np.zeros(6, dtype=np.float64)
        spot = np.ones(6, dtype=np.bool_)
        with self.assertRaises(runner.P1RunnerError):
            runner.build_target_arrays(returns, spot, horizons=(1,))
        fixture_targets, fixture_mask, fixture_end = runner.build_target_arrays_fixture(
            returns,
            spot,
            horizons=(1,),
        )
        self.assertEqual(fixture_targets.shape, (6, 1))
        self.assertTrue(fixture_mask[0, 0])
        self.assertEqual(fixture_end[0, 0], 2)

        features = np.zeros((64, runner.FEATURE_DIMENSION), dtype=np.float64)
        availability = {
            name: np.ones(64, dtype=np.bool_)
            for name in REQUIRED_AVAILABILITY_COLUMNS
        }
        with self.assertRaises(runner.P1RunnerError):
            runner.build_context_mask(features, availability)
        fixture_context = runner.build_context_mask_fixture(features, availability)
        self.assertTrue(fixture_context[63])

    def test_oof_keeps_binary_and_continuous_baseline_tasks_distinct(self):
        fake_dataset = SimpleNamespace(features=np.zeros((5, runner.FEATURE_DIMENSION)))
        fake_plan = SimpleNamespace(origins=(0,), horizons=(1,))
        fit = object()
        with mock.patch.object(runner, "_ensure_dataset", return_value=fake_dataset), mock.patch.object(
            runner, "build_runner_plan", return_value=fake_plan
        ), mock.patch.object(runner, "fit_model_at_origin", return_value=fit) as fit_call:
            run = runner.run_synthetic_oof(
                fake_dataset,
                model_ids=("zero_return",),
                outer_report_only=True,
            )
        self.assertEqual(
            set(run.fits),
            {
                (0, 1, "zero_return", "continuous"),
                (0, 1, "zero_return", "binary"),
            },
        )
        self.assertEqual(run.get(0, 1, "zero_return", "binary"), fit)
        self.assertEqual(run.get(0, 1, "zero_return", "continuous"), fit)
        for bad_origin, bad_horizon, bad_task in (
            ("0", 1, "continuous"),
            (0, "1", "continuous"),
            (0, 1, None),
            (0, 1, "continuous-but-unknown"),
        ):
            with self.assertRaises(runner.P1RunnerError):
                run.get(bad_origin, bad_horizon, "zero_return", bad_task)
        self.assertEqual(
            {call.kwargs["task"] for call in fit_call.call_args_list},
            {"continuous", "binary"},
        )

    def test_train_start_is_part_of_the_fit_contract(self):
        full = runner.train_mask_for_origin(self.dataset, 20000, 4)
        bounded = runner.train_mask_for_origin(
            self.dataset,
            20000,
            4,
            train_start=1000,
        )
        self.assertFalse(bounded[:1000].any())
        self.assertTrue(np.all(~bounded | full))
        fit = runner.fit_model_at_origin(
            self.dataset,
            "zero_return",
            20000,
            4,
            train_start=1000,
            prediction_range=(20000, 20010),
        )
        self.assertEqual(fit.train_start, 1000)
        with self.assertRaisesRegex(runner.P1RunnerError, "strictly before"):
            runner.train_mask_for_origin(
                self.dataset,
                20000,
                4,
                train_start=20000,
            )

    def test_future_return_perturbation_cannot_change_earlier_ridge_output(self):
        evidence = runner.assert_future_perturbation_invariance(
            self.dataset,
            "ridge",
            20000,
            4,
            prediction_range=(20000, 20020),
            perturb_start=20010,
        )
        self.assertEqual(evidence["status"], "passed")
        self.assertGreater(evidence["earlier_prediction_count"], 0)

    def test_synthetic_oof_is_explicitly_fixture_diagnostic_scope(self):
        self.assertEqual(runner.SYNTHETIC_OOF_SCOPE, "fixture_diagnostic_only")
        self.assertIs(runner.run_synthetic_oof_fixture, runner.run_synthetic_oof)
        self.assertIn("fixture/diagnostic", runner.run_synthetic_oof.__doc__)


def _fake_authenticated_s3_result(rows=400):
    index = pd.date_range("2020-01-01", periods=rows, freq="15min")
    features = pd.DataFrame(
        np.zeros((rows, len(MODEL_FEATURE_COLUMNS)), dtype=np.float64),
        index=index,
        columns=MODEL_FEATURE_COLUMNS,
    )
    features["close_ret"] = np.arange(rows, dtype=np.float64)
    returns = pd.Series(np.full(rows, 0.01, dtype=np.float64), index=index, name="returns")
    availability = pd.DataFrame(
        True,
        index=index,
        columns=REQUIRED_AVAILABILITY_COLUMNS,
        dtype=np.bool_,
    )
    return {
        "status": "v4_runtime_validated",
        "manifest_id": "p1-recovery-20260830",
        "manifest_sha256": runner.REGISTERED_MANIFEST_SHA256,
        "base_revision": "fixed",
        "results_observed": False,
        "p1_manifest_id": "p1-recovery-20260830",
        "p1_manifest_sha256": runner.REGISTERED_MANIFEST_SHA256,
        "p1_base_revision": "fixed",
        "p1_results_observed": False,
        "p1_runtime_validation_entrypoint": runner.P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
        "p1_runtime_body_validator_entrypoint": runner.V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT,
        "v4_runtime_validation_status": "passed",
        "v4_runtime_body_match": None,
        "v4_runtime_loaded_body_match": True,
        "v4_runtime_source_provenance_match": None,
        "v4_runtime_frozen_metadata_sha256": "a" * 64,
        "v4_runtime_cache_local_metadata_sha256": None,
        "v4_runtime_cache_local_source_provenance_digest": None,
        "v4_runtime_cache_local_schema_digest": None,
        "v4_frozen_metadata_sha256": "a" * 64,
        "v4_frozen_source_provenance_digest": "b" * 64,
        "v4_cache_local_metadata_sha256": None,
        "v4_cache_local_source_provenance_digest": None,
        "v4_runtime_provenance_disposition": {
            "status": "absent",
            "reason": "cache-local metadata is absent",
            "body_match": None,
            "source_provenance_match": None,
        },
        "metadata": {
            "cache_tag": "fixed",
            "schema_version": 4,
            "schema_digest": "c" * 64,
            "content_digests": {},
            "rows": rows,
            "sidecar_rows": rows,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "availability_columns": list(REQUIRED_AVAILABILITY_COLUMNS),
            "returns_columns": ["returns"],
        },
        "paths": {"feature_path": "/tmp/f", "returns_path": "/tmp/r"},
        "features": features,
        "returns": returns,
        "availability": availability,
    }


class P1RecoveryS3BoundaryTests(unittest.TestCase):
    def test_s3_materializer_is_private_and_keeps_only_immutable_provenance(self):
        self.assertFalse(hasattr(runner, "prepare_s3_injection_control"))
        self.assertFalse(hasattr(runner, "build_s3_injection_control"))
        self.assertFalse(hasattr(runner, "load_s3_validation_data"))
        result = _fake_authenticated_s3_result()
        body = runner._prepare_s3_injection_control(result)
        self.assertIsInstance(body.runtime, MappingProxyType)
        self.assertNotIn("features", body.runtime)
        self.assertNotIn("returns", body.runtime)
        self.assertEqual(body.injection_mask.sum(), 80)
        first = int(np.flatnonzero(body.injection_mask)[0])
        expected_z = (first - np.mean(np.arange(63, first, dtype=np.float64))) / np.std(
            np.arange(63, first, dtype=np.float64), ddof=0
        )
        self.assertAlmostEqual(body.z_scores[first], expected_z)
        self.assertAlmostEqual(
            body.injected_returns[first + 1] - body.control_returns[first + 1],
            runner.S3_INJECTION_BETA * body.z_scores[first],
        )
        self.assertFalse(body.injected_returns.flags.writeable)
        with self.assertRaisesRegex(runner.P1RunnerError, "authenticated public"):
            runner._require_production_s3_body(body)

    def test_s3_public_loader_calls_authenticated_wrapper_only(self):
        fake = _fake_authenticated_s3_result()
        with mock.patch(
            "unidream.experiments.runtime.validate_p1_v4_runtime_inputs",
            return_value=fake,
        ) as wrapper:
            body = runner.load_s3_validation_body(manifest_path="/tmp/ignored")
        wrapper.assert_called_once()
        self.assertEqual(body.runtime["p1_runtime_validation_entrypoint"], runner.P1_V4_RUNTIME_VALIDATION_ENTRYPOINT)
        self.assertEqual(runner._require_production_s3_body(body), body)
        mutated_features = np.array(body.features, copy=True)
        mutated_features[0, 0] += 1.0
        with self.assertRaisesRegex(runner.P1RunnerError, "digest mismatch"):
            runner._require_production_s3_body(
                replace(body, features=mutated_features)
            )

        forged = copy.deepcopy(fake)
        del forged["p1_runtime_validation_entrypoint"]
        with self.assertRaises(runner.P1RunnerError):
            runner._prepare_s3_injection_control(forged)

    def test_s3_validation_uses_one_fixed_fit_boundary_and_never_outer(self):
        one = np.zeros(1, dtype=np.float64)
        one_bool = np.ones(1, dtype=np.bool_)
        dummy_source = SimpleNamespace()
        dataset = runner.S3ArmDataset(
            seed=20260830,
            arm="injected",
            beta=runner.S3_INJECTION_BETA,
            timestamps=np.asarray([np.datetime64("2022-01-01", "ns")]),
            source=dummy_source,
            features=np.zeros((1, runner.FEATURE_DIMENSION), dtype=np.float64),
            returns=one,
            targets=np.zeros((1, len(runner.FORECAST_HORIZONS)), dtype=np.float64),
            target_end=np.zeros((1, len(runner.FORECAST_HORIZONS)), dtype=np.int64),
            target_mask=np.ones((1, len(runner.FORECAST_HORIZONS)), dtype=np.bool_),
            binary_labels=np.zeros((1, len(runner.FORECAST_HORIZONS)), dtype=np.int8),
            context_mask=one_bool,
            availability={name: one_bool for name in REQUIRED_AVAILABILITY_COLUMNS},
        )
        echo = runner.ManifestEcho("fixed", runner.REGISTERED_MANIFEST_SHA256, "base")
        with mock.patch.object(runner, "_ensure_dataset", return_value=dataset), mock.patch.object(
            runner,
            "build_runner_plan",
            return_value=SimpleNamespace(manifest_echo=echo),
        ), mock.patch.object(runner, "fit_model_at_origin", return_value=object()) as fit_call:
            result = runner.run_s3_validation_fits(dataset)
        self.assertFalse(result.outer_test_executed)
        self.assertTrue(result.outer_report_only)
        self.assertEqual(result.fit_range, (runner.S3_TRAIN_START, runner.S3_VALIDATION_ORIGIN))
        self.assertEqual(
            result.prediction_range,
            (runner.S3_VALIDATION_ORIGIN, runner.S3_VALIDATION_END),
        )
        self.assertEqual(fit_call.call_count, 24)
        for call in fit_call.call_args_list:
            self.assertEqual(call.args[2], runner.S3_VALIDATION_ORIGIN)
            self.assertEqual(call.kwargs["train_start"], runner.S3_TRAIN_START)
            self.assertEqual(
                call.kwargs["prediction_range"],
                (runner.S3_VALIDATION_ORIGIN, runner.S3_VALIDATION_END),
            )
        with self.assertRaises(runner.P1OuterReportBlocked):
            runner.run_s3_validation_fits(dataset, outer_report_only=False)


if __name__ == "__main__":
    unittest.main()
