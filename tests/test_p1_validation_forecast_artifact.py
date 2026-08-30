"""Fixture-only tests for the authenticated P1 forecast artifact boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from unidream.experiments import p1_validation_forecast as forecast
from unidream.experiments import p1_recovery_runner as runner


_SOURCE_ARRAY_NAMES = (
    "timestamps",
    "features",
    "returns",
    "availability.spot_bar_observed",
    "availability.funding_rate_available",
    "availability.mark_close_available",
)


def _fixture_artifact(
    contract: forecast.P1ForecastContract,
    *,
    scenario_id: str = "S1",
    arm: str = "known_high_snr_dgp",
    seed: int = 20260830,
    na_keys: set[tuple[int, str, str]] | None = None,
) -> dict[str, object]:
    """Build a small canonical support-only payload without fitting anything."""

    spec = contract.spec(scenario_id, arm)
    na_keys = set() if na_keys is None else set(na_keys)
    start, end = spec.support_range
    count = end - start
    rows = np.arange(start, end, dtype=np.int64)
    horizons = np.asarray(forecast.P1_FIXED_HORIZONS, dtype=np.int64)
    target_end = rows[:, None] + horizons[None, :] + 1
    target_mask = target_end <= spec.n_rows
    targets = np.where(target_mask, 0.001 * horizons[None, :], np.nan)
    labels = np.where(target_mask, 1, -1).astype(np.int8)
    context_mask = np.ones(count, dtype=np.bool_)
    origin_mask = context_mask & (target_end[:, 1] <= end)
    score_eligible_mask = origin_mask & target_mask[:, 1]
    spot_bar_observed = np.ones(count, dtype=np.bool_)
    ticks = np.datetime64("2024-01-01T00:00:00", "ns") + (
        np.arange(count, dtype=np.int64) * np.timedelta64(15, "m")
    )
    support_timestamps = [str(np.datetime_as_string(value, unit="ns")) for value in ticks]
    return_scale = (
        1e-7
        if scenario_id != "S3"
        else 2e-7
        if arm == "injected"
        else 3e-7
    )
    realized_returns = (np.arange(count, dtype=np.float64) + 1.0) * return_scale
    source_arrays = {name: f"{index + 1:064x}" for index, name in enumerate(_SOURCE_ARRAY_NAMES)}
    provenance: dict[str, object] = {
        "scenario_id": scenario_id,
        "arm": arm,
        "data_kind": spec.data_kind,
        "seed": seed,
        "beta": spec.beta,
        "snr": spec.snr,
        "n_rows": spec.n_rows,
        "source_array_sha256": source_arrays,
    }
    body_provenance: dict[str, object] = {
        "data_kind": spec.data_kind,
        "body_rows": spec.n_rows,
        "support_range": [start, end],
        "source_array_sha256": deepcopy(source_arrays),
    }
    if spec.data_kind == "s3":
        body_sha = ("f" if arm == "injected" else "e") * 64
        provenance["source_body_sha256"] = body_sha
        body_provenance["source_body_sha256"] = body_sha
        provenance["runtime"] = {}
        body_provenance["runtime"] = {}
    evidence: dict[str, object] = {
        "status": "passed",
        "method": "fixture",
        "origin": spec.fit_origin,
        "horizon": 4,
        "perturb_start": end,
        "fitted_prefix_mask_sha256": "a" * 64,
        "earlier_prediction_mask_sha256": "b" * 64,
        "earlier_prediction_sha256": "c" * 64,
    }
    if spec.data_kind == "s3":
        evidence["source_body_sha256"] = provenance["source_body_sha256"]
    fits: list[dict[str, object]] = []
    coverage: dict[str, object] = {}
    for horizon in forecast.P1_FIXED_HORIZONS:
        column = forecast.P1_FIXED_HORIZONS.index(horizon)
        inference = context_mask & (target_end[:, column] <= end)
        eligible = inference & target_mask[:, column]
        for model_id, task in forecast.P1_ALLOWED_MODEL_TASK_KEYS:
            key = (horizon, model_id, task)
            is_na = key in na_keys
            prediction_mask = np.zeros(count, dtype=np.bool_) if is_na else inference.copy()
            value = 0.5 if task == "binary" else 0.0
            predictions = [value if flag else None for flag in prediction_mask]
            record: dict[str, object] = {
                "horizon": horizon,
                "model_id": model_id,
                "task": task,
                "status": "N/A" if is_na else "ok",
                "reason": "fixture unavailable" if is_na else None,
                "train_count": 0 if is_na else 1000,
                "train_mask": [False] * count,
                "eligible_mask": eligible.tolist(),
                "prediction_mask": prediction_mask.tolist(),
                "predictions": predictions,
            }
            fits.append(record)
            coverage[forecast._fit_key(*key)] = forecast._coverage_expected(
                support_range=spec.support_range,
                target_end=target_end,
                target_mask=target_mask,
                context_mask=context_mask,
                record=record,
                horizon=horizon,
                model_id=model_id,
                task=task,
                data_kind=spec.data_kind,
            )
    header: dict[str, object] = {
        "artifact_type": "p1_validation_forecast",
        "schema_id": forecast.P1_FORECAST_SCHEMA_ID,
        "schema_version": forecast.P1_FORECAST_FILE_VERSION,
        "scenario_id": scenario_id,
        "arm": arm,
        "seed": seed,
        "split_id": "validation",
        "support_id": spec.support_id,
        "support_range": [start, end],
        "fit_origin": spec.fit_origin,
        "train_start": spec.train_start,
        "fit_range": list(spec.fit_range),
        "forecast_horizons": list(forecast.P1_FIXED_HORIZONS),
        "model_task_keys": [list(key) for key in forecast.P1_ALLOWED_MODEL_TASK_KEYS],
        "outer_report_only": True,
        "outer_test_executed": False,
        "prereg_results_observed": False,
        "validation_results_observed": True,
        "outer_results_observed": False,
        "scenario_provenance": provenance,
        "body_provenance": body_provenance,
        "future_perturbation_evidence": evidence,
    }
    return {
        "format": forecast.P1_FORECAST_FILE_FORMAT,
        "format_version": forecast.P1_FORECAST_FILE_VERSION,
        "header": header,
        "manifest_sha256": contract.manifest_sha256,
        "trial_registry_sha256": contract.trial_registry_sha256,
        "comparison_registry_sha256": contract.comparison_registry_sha256,
        "prereg_results_observed": False,
        "validation_results_observed": True,
        "outer_results_observed": False,
        "support_timestamps": support_timestamps,
        "realized_returns": realized_returns.tolist(),
        "targets": targets.tolist(),
        "target_end": target_end.tolist(),
        "target_mask": target_mask.tolist(),
        "binary_labels": labels.tolist(),
        "context_mask": context_mask.tolist(),
        "origin_mask": origin_mask.tolist(),
        "score_eligible_mask": score_eligible_mask.tolist(),
        "spot_bar_observed": spot_bar_observed.tolist(),
        "mask_hashes": {
            "context_mask": forecast._array_sha256(context_mask, name="context_mask"),
            "origin_mask": forecast._array_sha256(origin_mask, name="origin_mask"),
            "score_eligible_mask": forecast._array_sha256(
                score_eligible_mask,
                name="score_eligible_mask",
            ),
            "target_mask": forecast._array_sha256(target_mask, name="target_mask"),
            "spot_bar_observed": forecast._array_sha256(
                spot_bar_observed,
                name="spot_bar_observed",
            ),
        },
        "fits": fits,
        "coverage": coverage,
    }


def _canonical_write(path: Path, payload: dict[str, object]) -> str:
    encoded = forecast._json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


class P1ValidationForecastArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = forecast.authenticate_p1_forecast_contract()

    def _metadata(self, scenario_id: str = "S1", arm: str = "known_high_snr_dgp", seed: int = 20260830):
        return forecast.expected_metadata_for_arm(self.contract, scenario_id, arm, seed)

    def _save_and_load(self, payload: dict[str, object], *, metadata=None):
        metadata = self._metadata() if metadata is None else metadata
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "forecast.json"
        digest = forecast.save_p1_forecast_artifact(path, payload, expected_metadata=metadata)
        loaded = forecast.load_p1_forecast_artifact(
            path,
            expected_file_sha256=digest,
            expected_metadata=metadata,
        )
        return path, digest, loaded

    def test_roundtrip_external_sha_state_and_synthetic_returns_authority(self) -> None:
        payload = _fixture_artifact(self.contract)
        path, digest, loaded = self._save_and_load(payload)
        self.assertTrue(path.is_file())
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(loaded.file_sha256, digest)
        self.assertTrue(loaded.promotion_allowed)
        self.assertFalse(loaded.artifact["prereg_results_observed"])
        self.assertTrue(loaded.artifact["validation_results_observed"])
        self.assertFalse(loaded.artifact["outer_results_observed"])
        source = forecast.require_authenticated_forecast_action_source(loaded.action_source)
        self.assertTrue(source.is_authenticated)
        np.testing.assert_array_equal(source.realized_returns, np.asarray(payload["realized_returns"]))
        self.assertFalse(np.array_equal(source.realized_returns, np.full(len(source.returns), -9.0)))
        self.assertEqual(source.forecast_h4.shape, source.timestamps.shape)
        self.assertEqual(source.forecast_h4_mask.shape, source.timestamps.shape)
        self.assertEqual(source.origin_mask.shape, source.timestamps.shape)
        self.assertEqual(source.score_mask.shape, source.timestamps.shape)
        np.testing.assert_array_equal(source.action_score_mask, source.score_mask)
        self.assertEqual(source.common_mask.shape, source.timestamps.shape)
        self.assertEqual(source.source_hashes["manifest_sha256"], forecast.REGISTERED_MANIFEST_SHA256)
        with self.assertRaisesRegex(forecast.P1ForecastError, "ambiguous"):
            _ = source.score_eligible

    def test_unavailable_spot_return_roundtrips_as_null_without_zero_fill(self) -> None:
        payload = _fixture_artifact(self.contract)
        payload["realized_returns"][-1] = None
        payload["spot_bar_observed"][-1] = False
        payload["mask_hashes"]["spot_bar_observed"] = forecast._array_sha256(
            np.asarray(payload["spot_bar_observed"], dtype=np.bool_),
            name="spot_bar_observed",
        )
        _, _, loaded = self._save_and_load(payload)
        source = forecast.require_authenticated_forecast_action_source(loaded.action_source)
        self.assertTrue(np.isnan(source.realized_returns[-1]))
        self.assertFalse(source.bar_available[-1])
        self.assertNotEqual(source.realized_returns[-1], 0.0)

    def test_external_sha_regular_file_and_missing_metadata_fail_closed(self) -> None:
        payload = _fixture_artifact(self.contract)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "forecast.json"
        digest = forecast.save_p1_forecast_artifact(path, payload, expected_metadata=self._metadata())
        with self.assertRaises(forecast.P1ForecastError):
            forecast.load_p1_forecast_artifact(path, expected_file_sha256="0" * 64, expected_metadata=self._metadata())
        with self.assertRaises(forecast.P1ForecastError):
            forecast.load_p1_forecast_artifact(path, expected_file_sha256=digest)
        link = Path(directory.name) / "link.json"
        link.symlink_to(path)
        with self.assertRaises(forecast.P1ForecastError):
            forecast.load_p1_forecast_artifact(link, expected_file_sha256=digest, expected_metadata=self._metadata())
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(link, payload, expected_metadata=self._metadata())

    def test_expected_seed_spec_and_registered_hashes_cannot_be_bypassed(self) -> None:
        payload = _fixture_artifact(self.contract)
        metadata = dict(self._metadata())
        with self.assertRaises(forecast.P1ForecastError):
            forecast.expected_metadata_for_arm(self.contract, "S1", "known_high_snr_dgp", 123)
        metadata["seed"] = 123
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "bad.json", payload, expected_metadata=metadata)
        wrong_hash = deepcopy(payload)
        wrong_hash["trial_registry_sha256"] = "0" * 64
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "bad-hash.json", wrong_hash, expected_metadata=self._metadata())
        wrong_spec = deepcopy(payload)
        wrong_spec["header"]["support_range"] = [90001, 100001]
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "bad-spec.json", wrong_spec, expected_metadata=self._metadata())

    def test_nested_self_binding_deep_and_cycle_payloads_are_rejected(self) -> None:
        base = _fixture_artifact(self.contract)
        nested = deepcopy(base)
        nested["header"]["future_perturbation_evidence"]["file_sha256"] = "a" * 64
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "self.json", nested, expected_metadata=self._metadata())
        deep = deepcopy(base)
        node: dict[str, object] = {}
        cursor = node
        for _ in range(40):
            child: dict[str, object] = {}
            cursor["next"] = child
            cursor = child
        deep["header"]["future_perturbation_evidence"]["nested"] = node
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "deep.json", deep, expected_metadata=self._metadata())
        cyclic = deepcopy(base)
        cycle: dict[str, object] = {}
        cycle["cycle"] = cycle
        cyclic["header"]["future_perturbation_evidence"]["cycle"] = cycle
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "cycle.json", cyclic, expected_metadata=self._metadata())

    def test_mask_timestamp_and_coverage_tampering_is_rejected(self) -> None:
        base = _fixture_artifact(self.contract)
        bad_mask = deepcopy(base)
        bad_mask["fits"][0]["eligible_mask"][0] = False
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "mask.json", bad_mask, expected_metadata=self._metadata())
        bad_timestamp = deepcopy(base)
        bad_timestamp["support_timestamps"][1] = bad_timestamp["support_timestamps"][0]
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "timestamp.json", bad_timestamp, expected_metadata=self._metadata())
        bad_coverage = deepcopy(base)
        summary = bad_coverage["coverage"]["h4::ridge::continuous"]
        summary["eligible_origins"] += 1
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "coverage.json", bad_coverage, expected_metadata=self._metadata())
        bad_status = deepcopy(base)
        bad_status["coverage"]["h4::ridge::continuous"]["status"] = "N/A"
        with self.assertRaises(forecast.P1ForecastError):
            forecast.save_p1_forecast_artifact(Path(tempfile.mkdtemp()) / "coverage-status.json", bad_status, expected_metadata=self._metadata())

    def test_duplicate_and_nonfinite_json_are_rejected_after_external_hash(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        duplicate = Path(directory.name) / "duplicate.json"
        duplicate_bytes = b'{"x":1,"x":2}'
        duplicate.write_bytes(duplicate_bytes)
        with self.assertRaises(forecast.P1ForecastError):
            forecast.load_p1_forecast_artifact(
                duplicate,
                expected_file_sha256=hashlib.sha256(duplicate_bytes).hexdigest(),
                expected_metadata=self._metadata(),
            )
        nonfinite = Path(directory.name) / "nonfinite.json"
        nonfinite_bytes = b'{"x":NaN}'
        nonfinite.write_bytes(nonfinite_bytes)
        with self.assertRaises(forecast.P1ForecastError):
            forecast.load_p1_forecast_artifact(
                nonfinite,
                expected_file_sha256=hashlib.sha256(nonfinite_bytes).hexdigest(),
                expected_metadata=self._metadata(),
            )

    def test_s3_injected_and_control_use_selected_arm_returns_and_body_hash(self) -> None:
        injected = _fixture_artifact(self.contract, scenario_id="S3", arm="injected")
        control = _fixture_artifact(self.contract, scenario_id="S3", arm="zero_injection_control")
        injected_path, _, injected_loaded = self._save_and_load(
            injected,
            metadata=self._metadata("S3", "injected"),
        )
        control_path, _, control_loaded = self._save_and_load(
            control,
            metadata=self._metadata("S3", "zero_injection_control"),
        )
        del injected_path, control_path
        injected_source = forecast.require_authenticated_forecast_action_source(injected_loaded.action_source)
        control_source = forecast.require_authenticated_forecast_action_source(control_loaded.action_source)
        np.testing.assert_array_equal(injected_source.realized_returns, np.asarray(injected["realized_returns"]))
        np.testing.assert_array_equal(control_source.realized_returns, np.asarray(control["realized_returns"]))
        self.assertFalse(np.array_equal(injected_source.realized_returns, np.asarray(control["realized_returns"])))
        self.assertEqual(injected_source.source_hashes["source_body_sha256"], "f" * 64)
        self.assertEqual(control_source.source_hashes["source_body_sha256"], "e" * 64)

    def test_direct_or_mutated_capability_cannot_be_promoted(self) -> None:
        _, _, loaded = self._save_and_load(_fixture_artifact(self.contract))
        source = loaded.action_source
        self.assertIsInstance(hash(source), int)
        forged = replace(source, _production_seal=None)
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(forged)
        replaced = replace(source, _production_seal=forecast._FORECAST_ACTION_SOURCE_SEAL)
        self.assertFalse(forecast.is_authenticated_forecast_action_source(replaced))
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(replaced)
        direct = forecast.ForecastActionSource(
            scenario_id=source.scenario_id,
            arm=source.arm,
            seed=source.seed,
            split_id=source.split_id,
            support_id=source.support_id,
            support_range=source.support_range,
            fit_origin=source.fit_origin,
            timestamps=source.timestamps,
            realized_returns=source.realized_returns,
            forecast_h4=source.forecast_h4,
            forecast_h4_mask=source.forecast_h4_mask,
            origin_mask=source.origin_mask,
            score_mask=source.score_mask,
            common_mask=source.common_mask,
            source_hashes=source.source_hashes,
            prereg_results_observed=source.prereg_results_observed,
            validation_results_observed=source.validation_results_observed,
            outer_results_observed=source.outer_results_observed,
            validation_status=source.validation_status,
            promotion_allowed=source.promotion_allowed,
            _production_seal=forecast._FORECAST_ACTION_SOURCE_SEAL,
            binding_sha256=source.binding_sha256,
        )
        self.assertFalse(forecast.is_authenticated_forecast_action_source(direct))
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(direct)
        source.realized_returns.setflags(write=True)
        source.realized_returns[0] += 1.0
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(source)
        _, _, fresh = self._save_and_load(_fixture_artifact(self.contract))
        forged_hash = replace(
            fresh.action_source,
            source_hashes={**fresh.action_source.source_hashes, "realized_returns_sha256": "0" * 64},
        )
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(forged_hash)

    def test_na_coverage_blocks_promotion_and_action_source(self) -> None:
        payload = _fixture_artifact(
            self.contract,
            na_keys={(4, "ridge", "continuous")},
        )
        _, _, loaded = self._save_and_load(payload)
        self.assertFalse(loaded.promotion_allowed)
        self.assertEqual(loaded.validation["status"], "N/A")
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(loaded.action_source)

    def test_producer_requires_exact_runner_dataset_and_internal_evidence(self) -> None:
        spec = self.contract.spec("S1", "known_high_snr_dgp")
        fake = SimpleNamespace(
            seed=20260830,
            beta=spec.beta,
            timestamps=np.asarray([np.datetime64("2024-01-01", "ns")]),
            features=np.ones((1, 1), dtype=np.float64),
            returns=np.full(1, 7.0, dtype=np.float64),
        )
        with self.assertRaises(forecast.P1ForecastError):
            forecast.build_p1_forecast_artifact(self.contract, spec, fake, {})

        dataset = runner.build_synthetic_dataset(20260830, spec.beta)
        self.assertEqual(forecast._validate_registered_dataset(spec, dataset), 20260830)
        with self.assertRaises(forecast.P1ForecastError):
            forecast.build_p1_forecast_artifact(
                self.contract,
                spec,
                dataset,
                {},
                future_perturbation_evidence={"status": "passed"},
            )

    def test_fit_must_match_fresh_runner_refit_and_registered_train_body(self) -> None:
        spec = self.contract.spec("S1", "known_high_snr_dgp")
        dataset = runner.build_synthetic_dataset(20260830, spec.beta)
        horizon = 4
        train_mask = runner.train_mask_for_origin(
            dataset,
            spec.fit_origin,
            horizon,
            train_start=spec.train_start,
        )
        eligible_mask = runner.prediction_mask_for_range(
            dataset,
            horizon,
            start=spec.support_range[0],
            end=spec.support_range[1],
        )
        outer_train = np.ones(len(dataset.features), dtype=np.bool_)
        predictions = np.where(eligible_mask, 0.0, np.nan).astype(np.float64)
        forged_outer = runner.ModelFit(
            model_id="ridge",
            task="continuous",
            horizon=horizon,
            origin=spec.fit_origin,
            train_start=spec.train_start,
            train_mask=outer_train,
            eligible_mask=eligible_mask,
            prediction_mask=eligible_mask,
            predictions=predictions,
            status="ok",
            reason=None,
            scaler=None,
            estimator=None,
        )
        with self.assertRaises(forecast.P1ForecastError):
            forecast._fit_record(
                forged_outer,
                dataset=dataset,
                spec=spec,
                horizon=horizon,
                model_id="ridge",
                task="continuous",
                expected_train_mask=train_mask,
                expected_prediction_mask=eligible_mask,
            )

        forged_prediction = runner.ModelFit(
            model_id="ridge",
            task="continuous",
            horizon=horizon,
            origin=spec.fit_origin,
            train_start=spec.train_start,
            train_mask=train_mask,
            eligible_mask=eligible_mask,
            prediction_mask=eligible_mask,
            predictions=predictions,
            status="ok",
            reason=None,
            scaler=None,
            estimator=None,
        )
        with self.assertRaises(forecast.P1ForecastError):
            forecast._fit_record(
                forged_prediction,
                dataset=dataset,
                spec=spec,
                horizon=horizon,
                model_id="ridge",
                task="continuous",
                expected_train_mask=train_mask,
                expected_prediction_mask=eligible_mask,
            )

    def test_nonproduction_fixture_load_does_not_emit_authenticated_capability(self) -> None:
        payload = _fixture_artifact(self.contract)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fixture.json"
        digest = _canonical_write(path, payload)
        loaded = forecast.load_p1_forecast_artifact(
            path,
            expected_file_sha256=digest,
            require_production=False,
        )
        self.assertFalse(loaded.action_source.is_authenticated)
        with self.assertRaises(forecast.P1ForecastError):
            forecast.require_authenticated_forecast_action_source(loaded.action_source)


if __name__ == "__main__":
    unittest.main()
