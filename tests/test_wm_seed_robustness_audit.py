"""Synthetic chronology/exclusion audit tests; no real market reads."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments import wm_seed_robustness_audit as audit


class SeedRobustnessAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        prepared = self.source / "prepared"
        prepared.mkdir(parents=True)
        self.cfg = {"output_dir": str(self.source), "arms": ["A"], "seeds": [7],
                    "policy": {"context": 3, "horizon": 2, "purge_bars": 4}}
        raw_times = pd.date_range("2020-01-01", "2021-04-01", freq="15min", tz="UTC", inclusive="left").asi8
        n = len(raw_times)
        index = pd.DatetimeIndex(raw_times, tz="UTC")
        raw = 100 * np.exp(np.arange(n) * 1e-5)
        x = np.column_stack((np.sin(np.arange(n) / 20), np.cos(np.arange(n) / 31))).astype(np.float32)
        # Distinct causal reasons, including overlapping raw/cohort flags.
        bad_close = int(index.get_loc("2020-07-10 00:00"))
        bad_token = int(index.get_loc("2020-07-12 00:00"))
        absent = int(index.get_loc("2020-07-14 00:00"))
        bad_forecast = int(index.get_loc("2020-07-16 00:00"))
        bad_return = int(index.get_loc("2020-07-18 00:00"))
        raw[bad_close] = np.nan
        x[bad_token] = np.nan
        decisions = np.arange(10, n - 3, dtype=np.int64)
        # Do not present unavailable teacher labels as eligible source rows.
        decisions = decisions[~np.isin(decisions, [absent, absent + 1, bad_close - 2, bad_close - 1, bad_close, bad_close + 1])]
        times = raw_times[decisions]
        f = np.column_stack((np.sin(decisions / 11), np.ones(len(decisions)))).astype(np.float32)
        f[decisions == bad_forecast] = np.nan
        current = np.log(raw[decisions] / raw[decisions - 1])
        current[decisions == bad_return] = np.nan
        prefix = np.r_[0, np.cumsum(~np.isfinite(x).all(axis=1))]
        context_valid = prefix[decisions + 1] == prefix[decisions - 2]
        future = np.log(raw[decisions[:, None] + np.arange(1, 3)] / raw[decisions, None])
        arrays = {"raw_index_ns": raw_times, "raw_close": raw, "index_ns": times, "decision_id": decisions,
                  "x_grid": x, "f": f, "context_valid": context_valid,
                  "bar_log_return": current, "future_path_log_returns": future,
                  "teacher_valid": np.isfinite(future).all(axis=1),
                  "wm_mask_grid": np.ones(n, bool), "wm_label_end_ns": raw_times + 2 * audit.BAR_NS}
        self.manifest = {"complete": True, "arrays": {}, "fold_arrays": {}, "folds": [], "effective_raw_history_bars": 4}
        for name, values in arrays.items():
            path = prepared / (name + ".npy")
            np.save(path, values, allow_pickle=False)
            self.manifest["arrays"][name] = {"path": str(path), "sha256": audit._sha(path)}
        self.arrays = arrays
        self.saved = []
        for fold in range(3):
            start = pd.Timestamp("2020-01-01", tz="UTC") + pd.DateOffset(months=3 * fold)
            meta = {"fold": fold, "train_start": start.isoformat(),
                    "train_end": (start + pd.DateOffset(months=3)).isoformat(),
                    "val_start": (start + pd.DateOffset(months=3)).isoformat(),
                    "val_end": (start + pd.DateOffset(months=6)).isoformat(),
                    "test_start": (start + pd.DateOffset(months=6)).isoformat(),
                    "test_end": (start + pd.DateOffset(months=9)).isoformat()}
            self.manifest["folds"].append(meta)
            valid = context_valid & np.isfinite(f).all(axis=1) & np.isfinite(current)
            ranges = {split: (pd.Timestamp(meta[split + "_start"]), pd.Timestamp(meta[split + "_end"]))
                      for split in ("train", "val", "test")}
            val_start = ranges["val"][0]
            ranges.update({split: (val_start + pd.DateOffset(months=k), val_start + pd.DateOffset(months=k + 1))
                           for k, split in enumerate(("earlystop", "calibration", "selection"))})
            saved = {}
            for split, (lower, upper) in ranges.items():
                ids = np.flatnonzero(valid & (times >= lower.value) & (times + 4 * audit.BAR_NS < upper.value))
                saved[split] = ids
                saved["wm_" + split] = ids[raw_times[decisions[ids]] + 2 * audit.BAR_NS < upper.value]
            for name, values, ids in (("x", x, decisions[saved["train"]]), ("f", f, saved["train"])):
                values = values[ids].astype(np.float64)
                saved[name + "_mean"] = values.mean(axis=0)
                std = values.std(axis=0)
                saved[name + "_std"] = np.where(std > 1e-8, std, 1.)
            self.saved.append(saved)
            self._save_fold(fold)
        self._save_manifest()
        provenance = {"normalizers": [{"fold": meta["fold"], "fit_split": "train", "source_id": audit._ids_sha(saved["train"]),
                        "expected_source_id": audit._ids_sha(saved["train"])}
                        for meta, saved in zip(self.manifest["folds"], self.saved)]}
        (self.source / "normalizer_provenance.json").write_text(json.dumps(provenance))
        self._save_ledger()

    def _save_fold(self, fold):
        path = self.source / "prepared" / (f"fold_{fold:02d}.npz")
        np.savez(path, **self.saved[fold])
        self.manifest["fold_arrays"][str(fold)] = {"path": str(path), "sha256": audit._sha(path)}

    def _save_manifest(self):
        (self.source / "prepared/cache_manifest.json").write_text(json.dumps(self.manifest))

    def _save_array(self, key):
        entry = self.manifest["arrays"][key]
        np.save(entry["path"], self.arrays[key], allow_pickle=False)
        entry["sha256"] = audit._sha(entry["path"])
        self._save_manifest()

    def _save_ledger(self):
        pieces = []
        for meta, saved in zip(self.manifest["folds"], self.saved):
            times = self.arrays["index_ns"][saved["test"]]
            reset = np.r_[True, np.diff(times) != audit.BAR_NS]
            pieces.append(pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True), "fold": meta["fold"],
                "arm": "A", "stage": "rl", "seed": 7, "reset": reset, "executed_position": 1.,
                "turnover": 0., "cost_fraction": 0.}))
        self.ledger = pd.concat(pieces, ignore_index=True)
        path = self.source / "final/executed_policy_ledger.parquet"
        path.parent.mkdir(exist_ok=True)
        self.ledger.to_parquet(path, index=False)

    def _make_latent_normalizers(self):
        for meta, saved in zip(self.manifest["folds"], self.saved):
            directory = self.source / "world_models" / f"fold_{meta['fold']:02d}_s7"
            directory.mkdir(parents=True)
            ids = np.arange(len(self.arrays["index_ns"]))
            latent = np.column_stack((np.sin(ids / 13), np.cos(ids / 17), np.ones(len(ids)))).astype(np.float32)
            np.save(directory / "latent_train_val.npy", latent)
            train = saved["train"]
            total, square = np.zeros(3), np.zeros(3)
            for start in range(0, len(train), 1024):
                part = latent[train[start:start + 1024]].astype(np.float64)
                total += part.sum(axis=0)
                square += np.square(part).sum(axis=0)
            mean = total / len(train)
            std = np.sqrt(np.maximum(0., square / len(train) - mean ** 2))
            np.savez(directory / "latent_norm.npz", mean=mean.astype(np.float32),
                     std=np.where(std >= 1e-8, std, 1.).astype(np.float32))
            fit = {"complete": True, "test_used": False, "latent_normalizer_fit_split": "policy Train only",
                   "normalizer_train_fold_sha256": self.manifest["fold_arrays"][str(meta["fold"])]["sha256"],
                   "artifact_sha256": {name: audit._sha(directory / name) for name in ("latent_train_val.npy", "latent_norm.npz")}}
            (directory / "fit.json").write_text(json.dumps(fit))

    def run_audit(self):
        return audit.audit_source(self.cfg, self.root / "new_audit")

    def test_complete_audit_reconstructs_saved_cohort_and_normalizers(self):
        result = self.run_audit()
        self.assertTrue(result["passed"], result["violations"])
        self.assertFalse(result["leakage_found"])
        self.assertEqual(result["n_folds"], 3)
        self.assertEqual(result["n_test_market_rows"], sum(len(s["test"]) for s in self.saved))
        self.assertTrue(result["reset_accounting"]["free_endowed_one_position_reset_verified"])
        self.assertGreater(result["reset_accounting"]["n_expected_resets"], 3)
        normals = pd.read_csv(result["artifacts"]["normalizer_audit.csv"]["path"])
        self.assertTrue(normals.matches_train_only_recalculation.all())
        self.assertEqual(normals.max_mean_error.max(), 0.)
        self.assertEqual(normals.max_std_error.max(), 0.)

    def test_exclusions_are_exclusive_with_all_underlying_flags(self):
        result = self.run_audit()
        bars = pd.read_csv(result["artifacts"]["exclusion_bars.csv"]["path"])
        self.assertTrue(bars.classification_matches_saved_split.all())
        self.assertEqual(len(bars), result["n_nominal_test_bars"])
        self.assertEqual(sum(result["exclusion_totals"].values()), len(bars))
        self.assertEqual(set(bars.reason), set(audit.REASON_PRIORITY))
        overlap = bars.loc[~bars.raw_close_and_reference_valid & ~bars.in_saved_forecast_common_cohort]
        self.assertGreater(len(overlap), 0)
        self.assertTrue((overlap.reason == "raw_close_or_reference_missing").all())
        tail = bars.loc[bars.in_nominal_tail & bars.raw_close_and_reference_valid]
        self.assertTrue((tail.reason == "nominal_tail").all())
        self.assertEqual(len(tail), 4)

    def test_historical_inputs_and_past_tests_are_permitted(self):
        result = self.run_audit()
        splits = pd.read_csv(result["artifacts"]["split_audit.csv"]["path"])
        self.assertGreater(splits.rows_with_input_history_before_split_start.sum(), 0)
        self.assertEqual(splits.future_input_rows.sum(), 0)
        self.assertEqual(splits.teacher_labels_crossing_split.sum(), 0)
        self.assertEqual(splits.wm_labels_crossing_split.sum(), 0)
        self.assertTrue(splits.teacher_label_end_includes_delay_bar.all())
        overlaps = pd.read_csv(result["artifacts"]["cross_fold_overlaps.csv"]["path"])
        reuse = overlaps.loc[(overlaps.later_split == "train") & (overlaps.n_shared_rows > 0)]
        self.assertGreater(len(reuse), 0)
        self.assertTrue(reuse.allowed_historical_walkforward_reuse.all())
        self.assertEqual(overlaps.loc[overlaps.later_split == "test", "n_shared_rows"].sum(), 0)

    def test_array_tampering_rejected_before_outputs(self):
        entry = self.manifest["arrays"]["f"]
        values = np.load(entry["path"])
        values[0, 0] = 123.
        np.save(entry["path"], values)
        with self.assertRaisesRegex(ValueError, "source cache changed"):
            self.run_audit()
        self.assertFalse((self.root / "new_audit").exists())

    def test_same_fold_train_future_leakage_is_detected(self):
        val_first = self.saved[0]["val"][0]
        self.saved[0]["train"] = np.r_[self.saved[0]["train"], val_first]
        self.saved[0]["wm_train"] = np.r_[self.saved[0]["wm_train"], val_first]
        self._save_fold(0)
        self._save_manifest()
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertTrue(result["leakage_found"])
        self.assertTrue(any("train/val share" in value for value in result["violations"]))
        splits = pd.read_csv(result["artifacts"]["split_audit.csv"]["path"])
        row = splits.loc[(splits.fold == 0) & (splits.split == "train")].iloc[0]
        self.assertEqual(row.teacher_labels_crossing_split, 1)
        self.assertEqual(row.wm_labels_crossing_split, 1)

    def test_train_normalizer_changed_by_future_data_is_detected(self):
        self.saved[0]["f_mean"] += .01
        self._save_fold(0)
        self._save_manifest()
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertTrue(result["leakage_found"])
        self.assertTrue(any("Train-only fit" in value for value in result["violations"]))

    def test_context_cache_tampering_detected_after_valid_hash_update(self):
        self.arrays["context_valid"][0] = False
        self._save_array("context_valid")
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["context_cache_matches_recalculation"])

    def test_teacher_path_must_match_delayed_raw_price_labels(self):
        self.arrays["future_path_log_returns"][0, 0] += .01
        self._save_array("future_path_log_returns")
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["teacher_paths_match_delayed_raw_price_labels"])

    def test_reset_inventory_and_fee_contract_is_checked(self):
        self.ledger.loc[0, "turnover"] = .5
        self.ledger.to_parquet(self.source / "final/executed_policy_ledger.parquet", index=False)
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertFalse(result["reset_accounting"]["free_endowed_one_position_reset_verified"])

    def test_provenance_ids_cannot_name_val_or_other_train_rows(self):
        path = self.source / "normalizer_provenance.json"
        provenance = json.loads(path.read_text())
        provenance["normalizers"][0]["source_id"] = audit._ids_sha(self.saved[0]["val"])
        path.write_text(json.dumps(provenance))
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertTrue(result["leakage_found"])

    def test_audit_never_writes_under_frozen_source(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            audit.audit_source(self.cfg, self.source / "validation/audit")

    def test_latent_normalizers_use_exact_policy_train_ids(self):
        self._make_latent_normalizers()
        result = self.run_audit()
        self.assertTrue(result["passed"], result["violations"])
        self.assertEqual(result["n_latent_normalizers_checked"], 3)
        table = pd.read_csv(result["artifacts"]["latent_normalizer_audit.csv"]["path"])
        self.assertTrue(table.matches_train_only_recalculation.all())
        self.assertTrue(table.provenance_matches.all())
        self.assertEqual(table.max_mean_error.sum(), 0.)
        self.assertEqual(table.max_std_error.sum(), 0.)

    def test_latent_normalizer_with_val_statistics_is_rejected(self):
        self._make_latent_normalizers()
        directory = self.source / "world_models/fold_00_s7"
        latent = np.load(directory / "latent_train_val.npy")
        selected = latent[self.saved[0]["val"]].astype(np.float64)
        np.savez(directory / "latent_norm.npz", mean=selected.mean(axis=0).astype(np.float32),
                 std=selected.std(axis=0).astype(np.float32))
        fit = json.loads((directory / "fit.json").read_text())
        fit["artifact_sha256"]["latent_norm.npz"] = audit._sha(directory / "latent_norm.npz")
        (directory / "fit.json").write_text(json.dumps(fit))
        result = self.run_audit()
        self.assertFalse(result["passed"])
        self.assertTrue(result["leakage_found"])
        self.assertTrue(any("latent normalizer differs" in value for value in result["violations"]))


if __name__ == "__main__":
    unittest.main()
