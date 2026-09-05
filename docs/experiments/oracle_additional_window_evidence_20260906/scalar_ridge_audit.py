"""Read-only scalar replay of the registered, already generated Ridge forecasts."""
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from unidream.experiments.alpha_dd_search import digest, file_digest
from unidream.experiments.oracle_additional_window_replay import FOLDS, prepare


CONFIG = Path("configs/oracle_additional_window_replay_20260906.yaml")
OUTPUT = Path("codex_outputs/oracle_additional_window_replay_v1")
REPORT = Path("/tmp/oracle_additional_window_scalar_ridge_audit_20260906.json")
RTOL, ATOL = 1e-12, 1e-14


def array_digest(values):
    array = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def checked_comparison(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    differences = np.abs(actual - expected)
    limit = ATOL + RTOL * np.abs(expected)
    assert np.all(differences <= limit)
    return {"rows": len(actual), "max_absolute_difference": float(differences.max(initial=0.)),
            "max_tolerance_fraction": float((differences / limit).max(initial=0.)), "pass": True}


registration = json.loads((OUTPUT / "registration.json").read_text())
results = json.loads((OUTPUT / "results.json").read_text())
assert file_digest(CONFIG) == registration["config_sha256"]
assert results["registration_sha256"] == digest(registration)
source_bindings = {}
for name, expected in registration["source_bindings"].items():
    path = Path(name)
    assert file_digest(path) == expected
    source_bindings[str(path)] = expected
print("registered sources verified; preparing existing bounded data/features", flush=True)
data = prepare(CONFIG)
cfg, fc, bars, groups, original, outcomes, masks, preflight = (data[k] for k in ("config", "source_config", "bars", "groups", "original", "outcomes", "masks", "preflight"))
assert cfg == registration["config"]
assert file_digest(OUTPUT / "preflight.json") == registration["preflight_sha256"]
assert preflight == json.loads((OUTPUT / "preflight.json").read_text())
assert pd.Timestamp(cfg["data_cutoff"]) == pd.Timestamp("2026-07-16T13:45:00Z")
assert bars.index.max() <= pd.Timestamp(cfg["data_cutoff"])
print("prepared", len(bars), "bars through", bars.index.max().isoformat(), flush=True)

input_binding = {
    "spot_data_proof": preflight["spot_data_proof"], "um_data_proof": preflight["um_data_proof"],
    "data_cutoff": cfg["data_cutoff"], "rows": len(bars),
    "first_event_open": bars.index.min().isoformat(), "last_event_open": bars.index.max().isoformat(),
    "timestamps_sha256": array_digest(bars.index.asi8), "features": {}}
x = {}
for group in ("technical", "perp_delay0"):
    assert groups[group].index.equals(bars.index)
    x[group] = groups[group].to_numpy(float)
    input_binding["features"][group] = {
        "columns": groups[group].columns.tolist(), "shape": list(x[group].shape),
        "full_values_sha256": array_digest(x[group])}

artifact_bindings, fold_bindings, audits = {}, {}, []
for fold in FOLDS:
    manifest_path = OUTPUT / f"fold_{fold}.json"
    manifest = json.loads(manifest_path.read_text())
    fold_bindings[str(manifest_path)] = file_digest(manifest_path)
    assert manifest["registration_sha256"] == digest(registration)
    for key in ("rows", "scores"):
        assert manifest[key] == [record for record in results[key] if record["fold"] == fold]

    def bound(kind, name, extension):
        path = OUTPUT / kind / f"fold{fold}_{name}.{extension}"
        expected = manifest["artifact_sha256"][str(path)]
        assert file_digest(path) == expected
        artifact_bindings[str(path)] = expected
        return path

    provenance_path = bound("calibration", "provenance", "json")
    saved = json.loads(provenance_path.read_text())
    m = masks[fold]
    support = next(row for row in preflight["support"] if row["fold"] == fold)
    cal_ix = np.asarray((bars.index >= pd.Timestamp(support["scale_start"])) &
                        (bars.index < pd.Timestamp(support["evaluation_start"])))
    val_ix = np.asarray((bars.index >= pd.Timestamp(support["evaluation_start"])) &
                        (bars.index < pd.Timestamp(support["evaluation_end"])))
    cal_predict = cal_ix & m["predict"]
    inference = m["inference"]
    assert np.array_equal(cal_predict | inference, m["predict"])
    assert not np.any(cal_predict & inference)
    for name, mask in m.items():
        if name in saved["provenance"]["mask_position_sha256"]:
            h = hashlib.sha256(np.asarray([len(bars)], "<i8").tobytes() + mask.astype("u1").tobytes()).hexdigest()
            assert h == saved["provenance"]["mask_position_sha256"][name]

    for group in ("technical", "perp_delay0"):
        model_path = bound("models", group + "_mean", "joblib")
        model = joblib.load(model_path)
        assert len(model.steps) == 2
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        assert isinstance(scaler, StandardScaler) and isinstance(ridge, Ridge)
        assert scaler.with_mean and scaler.with_std and ridge.alpha == 100.
        assert list(groups[group].columns) == saved["provenance"]["feature_columns"][group]
        width = x[group].shape[1]
        for parameter in (scaler.mean_, scaler.scale_, scaler.var_, ridge.coef_, np.atleast_1d(ridge.intercept_)):
            assert np.isfinite(parameter).all()
        assert scaler.mean_.shape == scaler.scale_.shape == ridge.coef_.shape == (width,)
        assert np.all(scaler.scale_ > 0)
        assert np.isfinite(x[group][m["predict"]]).all()

        # Python float subtraction/division and math.fsum: no model.predict,
        # scaler.transform, BLAS, matrix multiplication, or einsum is invoked.
        center = [float(value) for value in scaler.mean_]
        scale = [float(value) for value in scaler.scale_]
        coef = [float(value) for value in ridge.coef_]
        intercept = float(ridge.intercept_)
        positions = np.flatnonzero(m["predict"])
        scalar = np.full(len(bars), np.nan)
        for index in positions:
            row = x[group][index]
            scalar[index] = intercept + math.fsum(
                ((float(row[j]) - center[j]) / scale[j]) * coef[j] for j in range(width))
        assert np.isfinite(scalar[m["predict"]]).all()
        calibration_path = bound("calibration", group, "npz")
        with np.load(calibration_path, allow_pickle=False) as saved_calibration:
            assert np.array_equal(saved_calibration["timestamps"], bars.index[cal_ix].asi8)
            assert np.array_equal(saved_calibration["scale_mask"], m["scale"][cal_ix])
            assert np.array_equal(saved_calibration["interval_mask"], m["interval"][cal_ix])
            assert np.array_equal(np.isfinite(saved_calibration["mu"]), m["predict"][cal_ix])
            assert np.isnan(saved_calibration["mu"][~m["predict"][cal_ix]]).all()
            calibration_check = checked_comparison(scalar[cal_predict],
                saved_calibration["mu"][m["predict"][cal_ix]])
        bias = saved["calibration"]["return_bias"][group]
        assert math.isfinite(bias)
        forecast_path = bound("forecasts", group + "_scaled", "npz")
        with np.load(forecast_path, allow_pickle=False) as saved_forecast:
            assert np.array_equal(saved_forecast["timestamps"], bars.index[val_ix].asi8)
            assert np.array_equal(saved_forecast["inference_mask"], inference[val_ix])
            assert np.array_equal(np.isfinite(saved_forecast["mu"]), inference[val_ix])
            assert np.isnan(saved_forecast["mu"][~inference[val_ix]]).all()
            inference_check = checked_comparison(scalar[inference] + bias,
                saved_forecast["mu"][inference[val_ix]])
        audits.append({"fold": fold, "group": group, "features": width,
            "model_path": str(model_path), "model_sha256": artifact_bindings[str(model_path)],
            "parameters_finite": True, "scaler_scale_positive": True,
            "parameter_sha256": {"mean": array_digest(scaler.mean_), "scale": array_digest(scaler.scale_),
                "coefficient": array_digest(ridge.coef_), "intercept": array_digest(np.asarray(ridge.intercept_))},
            "feature_input_sha256": {"fit": array_digest(x[group][m["fit"]]),
                "predict": array_digest(x[group][m["predict"]])},
            "predict_timestamps_sha256": array_digest(bars.index[m["predict"]].asi8),
            "mask_sha256": support["mask_sha256"],
            "raw_calibration": calibration_check, "scaled_inference": inference_check,
            "saved_additive_bias": bias, "scalar_predict_sha256": array_digest(scalar[m["predict"]])})
    print("audited fold", fold, "calibration", int(cal_predict.sum()), "inference", int(inference.sum()), flush=True)

assert len(audits) == 20
for path, expected in {**source_bindings, **artifact_bindings, **fold_bindings}.items():
    assert file_digest(Path(path)) == expected
log_path = Path("/tmp/oracle-additional-window-replay.log")
log = log_path.read_text()
report = {
    "schema": "oracle-additional-window-scalar-ridge-audit-v1", "pass": True,
    "registered_source_revision": registration["source_revision"],
    "audit_script_sha256": file_digest(Path(__file__)),
    "source_bindings": source_bindings, "artifact_bindings": artifact_bindings,
    "fold_manifest_bindings": fold_bindings,
    "experiment_bindings": {str(path): file_digest(path) for path in (
        CONFIG, OUTPUT / "registration.json", OUTPUT / "preflight.json", OUTPUT / "results.json")},
    "feature_input_bindings": input_binding,
    "model_count": len(audits), "raw_calibration_rows": sum(row["raw_calibration"]["rows"] for row in audits),
    "scaled_inference_rows": sum(row["scaled_inference"]["rows"] for row in audits),
    "unique_scalar_predictions": sum(row["raw_calibration"]["rows"] + row["scaled_inference"]["rows"] for row in audits),
    "max_absolute_difference": {name: max(row[name]["max_absolute_difference"] for row in audits)
        for name in ("raw_calibration", "scaled_inference")},
    "rtol": RTOL, "atol": ATOL, "rows": audits,
    "warning_scope": {"run_log_sha256": file_digest(log_path),
        "runtime_warning_occurrences": log.count("RuntimeWarning:"),
        "numerical_check_proves_warning_cause": False,
        "statement": "All saved Ridge forecast values checked against independent scalar arithmetic. This numerical agreement does not identify or prove the cause of the original BLAS/matmul warnings, and does not audit HGB fit numerics."},
    "models_fitted": 0, "policies_run": 0, "new_market_periods_accessed": False,
    "forecast_scores_recomputed": False,
    "scope": "SHA-bound saved model evaluation only; no parameter, support, strategy or outcome modification"}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps({key: report[key] for key in ("pass", "model_count", "raw_calibration_rows", "scaled_inference_rows",
    "unique_scalar_predictions", "max_absolute_difference")}), flush=True)
print(str(REPORT), file_digest(REPORT), flush=True)
