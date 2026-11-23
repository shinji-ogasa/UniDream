"""Read-only chronology, exclusion and normalizer audit for seed replication.

Historical Test rows may enter a LATER walk-forward Train/Val window after
their outcomes are known.  That is not within-fold leakage.  Historical inputs
may also precede their own split start.  Future outcomes, normalizer fits and
model selection must not cross a same-fold boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


BAR_NS = 900_000_000_000
REASON_PRIORITY = ("raw_close_or_reference_missing", "nominal_tail", "saved_forecast_common_cohort_absent",
                   "forecast_nonfinite", "actor_history_insufficient", "current_return_nonfinite",
                   "boundary_purge", "included")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _stamp(value):
    return str(pd.Timestamp(int(value), tz="UTC"))


def _range(times):
    return (_stamp(np.min(times)), _stamp(np.max(times))) if len(times) else (None, None)


def _context_valid(tokens, decisions, context):
    good = np.isfinite(tokens).all(axis=1)
    prefix = np.r_[0, np.cumsum(~good, dtype=np.int64)]
    ids = np.asarray(decisions, np.int64)
    valid = (ids >= context - 1) & (ids < len(good))
    rows = np.flatnonzero(valid)
    d = ids[rows]
    valid[rows] = prefix[d + 1] == prefix[d - context + 1]
    return valid


def _normalizer(values, ids):
    selected = np.asarray(values)[np.asarray(ids, np.int64)].astype(np.float64)
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("normalizer audit requires nonempty finite Train values")
    mean, std = selected.mean(axis=0), selected.std(axis=0)
    return mean, np.where(std > 1e-8, std, 1.)


def _ranges(meta):
    result = {split: (meta[split + "_start"], meta[split + "_end"]) for split in ("train", "val", "test")}
    start = pd.Timestamp(meta["val_start"])
    for offset, name in enumerate(("earlystop", "calibration", "selection")):
        result[name] = (start + pd.DateOffset(months=offset), start + pd.DateOffset(months=offset + 1))
    return result


def _classify_test_bars(arrays, folds, cfg):
    raw_times = np.asarray(arrays["raw_index_ns"], np.int64)
    times = np.asarray(arrays["index_ns"], np.int64)
    close = np.asarray(arrays["raw_close"])
    policy = cfg["policy"]
    context, purge = int(policy["context"]), int(policy["purge_bars"])
    last_end = max(pd.Timestamp(meta["test_end"]).value for meta, _ in folds)
    pieces = []
    for meta, saved in folds:
        lower, upper = pd.Timestamp(meta["test_start"]).value, pd.Timestamp(meta["test_end"]).value
        grid = np.arange(lower, upper, BAR_NS, dtype=np.int64)
        raw = np.searchsorted(raw_times, grid)
        present_raw = raw < len(raw_times)
        present_raw[present_raw] &= raw_times[raw[present_raw]] == grid[present_raw]
        raw_good = present_raw & (raw > 0)
        rows = np.flatnonzero(raw_good)
        d = raw[rows]
        raw_good[rows] &= np.isfinite(close[d]) & (close[d] > 0) & np.isfinite(close[d - 1]) & (close[d - 1] > 0)
        ledger = np.searchsorted(times, grid)
        present = ledger < len(times)
        present[present] &= times[ledger[present]] == grid[present]
        f_good, returns_good = np.zeros(len(grid), bool), np.zeros(len(grid), bool)
        rows = np.flatnonzero(present)
        f_good[rows] = np.isfinite(arrays["f"][ledger[rows]]).all(axis=1)
        returns_good[rows] = np.isfinite(arrays["bar_log_return"][ledger[rows]])
        context_good = _context_valid(arrays["x_grid"], raw, context) & present_raw
        boundary = grid + purge * BAR_NS >= upper
        tail = (upper == last_end) & (grid + purge * BAR_NS >= last_end)
        kept = np.isin(grid, times[np.asarray(saved["test"], np.int64)])
        reason = np.select((~raw_good, tail, ~present, ~f_good, ~context_good, ~returns_good, boundary),
                           REASON_PRIORITY[:-1], default="included")
        pieces.append(pd.DataFrame({"fold": meta["fold"], "timestamp": pd.to_datetime(grid, utc=True),
            "raw_grid_row": np.where(present_raw, raw, -1), "forecast_ledger_row": np.where(present, ledger, -1),
            "raw_close_and_reference_valid": raw_good, "in_saved_forecast_common_cohort": present,
            "forecast_features_finite": f_good, "actor_context_valid": context_good,
            "current_return_finite": returns_good, "in_split_boundary_purge": boundary,
            "in_nominal_tail": tail, "reason": reason, "actually_kept": kept,
            "classification_matches_saved_split": kept == (reason == "included")}))
    return pd.concat(pieces, ignore_index=True)


def _reset_audit(root, cfg, expected_times, expected_folds):
    path = Path(cfg.get("audit_ledger_path", root / "final/executed_policy_ledger.parquet"))
    expected = np.r_[True, (np.diff(expected_times) != BAR_NS) | (np.diff(expected_folds) != 0)]
    result = pd.DataFrame({"timestamp": pd.to_datetime(expected_times[expected], utc=True),
                           "fold": expected_folds[expected]})
    if not path.exists():
        result["observed_ledger_verified"] = False
        return result, {"available": False, "n_expected_resets": int(expected.sum()),
                        "reset_contract": "initial endowed position1; gap/fold resets at no turnover fee"}, None
    columns = ["timestamp", "fold", "reset", "executed_position", "turnover", "cost_fraction"]
    frame = pd.read_parquet(path, columns=columns, filters=[("arm", "==", cfg.get("arms", ["A"])[0]),
        ("stage", "==", "rl"), ("seed", "==", cfg.get("seeds", [7])[0])]).sort_values("timestamp").reset_index(drop=True)
    observed_times = pd.to_datetime(frame.timestamp, utc=True).astype("int64").to_numpy()
    cohort_equal = np.array_equal(observed_times, expected_times) and np.array_equal(frame.fold, expected_folds)
    masks_equal = cohort_equal and np.array_equal(frame.reset.to_numpy(bool), expected)
    selected = frame.loc[frame.reset].copy()
    selected["observed_ledger_verified"] = True
    free = bool(np.allclose(selected.executed_position, 1., rtol=0, atol=1e-12)
                and np.allclose(selected.turnover, 0., rtol=0, atol=1e-12)
                and np.allclose(selected.cost_fraction, 0., rtol=0, atol=1e-12))
    return selected, {"available": True, "source_path": str(path.resolve()), "source_sha256": _sha(path),
        "n_expected_resets": int(expected.sum()), "n_observed_resets": len(selected),
        "cohort_equal": bool(cohort_equal), "reset_mask_equal": bool(masks_equal),
        "free_endowed_one_position_reset_verified": free,
        "realism_limitation": "reset is free and exogenous; excluded intervals have no return or trading fee"}, path


def _latent_normalizers(root, cfg, folds, manifest, watched):
    """Recompute frozen WM latent norms using policy Train IDs, not Val/Test."""
    rows, failures = [], []
    if not (root / "world_models").exists() and "wm_model" not in cfg:
        return rows, failures
    for meta, saved in folds:
        train = np.asarray(saved["train"], np.int64)
        fold_hash = manifest["fold_arrays"][str(meta["fold"])]["sha256"]
        for seed in cfg["seeds"]:
            directory = root / "world_models" / f"fold_{int(meta['fold']):02d}_s{int(seed)}"
            fit_path = directory / "fit.json"
            if not fit_path.exists():
                failures.append(f"fold{meta['fold']}/s{seed}: frozen latent normalizer fit is absent")
                continue
            fit = json.loads(fit_path.read_text())
            watched[str(fit_path)] = _sha(fit_path)
            for name in ("latent_train_val.npy", "latent_norm.npz"):
                path = directory / name
                digest = _sha(path)
                if fit["artifact_sha256"][name] != digest:
                    raise ValueError(f"immutable latent normalizer artifact changed: {path}")
                watched[str(path)] = digest
            values = np.load(directory / "latent_train_val.npy", mmap_mode="r", allow_pickle=False)
            # Reproduce the source float64 chunked moment calculation and final
            # float32 cast, independently selecting the audited Train IDs.
            total = np.zeros(values.shape[1], np.float64)
            squared = np.zeros_like(total)
            for start in range(0, len(train), 1024):
                selected = np.asarray(values[train[start:start + 1024]], np.float64)
                if not np.isfinite(selected).all():
                    raise ValueError("frozen latent Train values must be finite")
                total += selected.sum(axis=0)
                squared += np.square(selected).sum(axis=0)
            mean = total / len(train)
            std = np.sqrt(np.maximum(0., squared / len(train) - mean ** 2))
            std = np.where(std >= 1e-8, std, 1.).astype(np.float32)
            mean = mean.astype(np.float32)
            with np.load(directory / "latent_norm.npz", allow_pickle=False) as stored:
                mean_error = float(np.max(np.abs(mean - stored["mean"])))
                std_error = float(np.max(np.abs(std - stored["std"])))
                match = np.array_equal(mean, stored["mean"]) and np.array_equal(std, stored["std"])
            provenance = (fit.get("complete") is True and fit.get("test_used") is False
                and fit.get("latent_normalizer_fit_split") == "policy Train only"
                and fit.get("normalizer_train_fold_sha256") == fold_hash)
            if not match or not provenance:
                failures.append(f"fold{meta['fold']}/s{seed}: latent normalizer differs from its Train-only provenance or fit")
            rows.append({"fold": meta["fold"], "seed": seed, "fit_split": "Train_only",
                "n_fit_rows": len(train), "source_train_ids_sha256": _ids_sha(train),
                "latent_dim": values.shape[1], "source_fit_json_sha256": watched[str(fit_path)],
                "max_mean_error": mean_error, "max_std_error": std_error,
                "matches_train_only_recalculation": bool(match), "provenance_matches": bool(provenance)})
    return rows, failures


def audit_source(source_cfg, output_dir):
    """Audit an immutable prepared source; write CSV/JSON only to a new root.

    ``source_cfg`` is its already resolved config dict, or a YAML path. Return
    ``passed``/``leakage_found`` and artifact paths; callers should fail closed
    when passed is false. No training, threshold choice or Test ranking occurs.
    """
    if not isinstance(source_cfg, dict):
        config_path = Path(source_cfg).resolve()
        cfg = yaml.safe_load(config_path.read_text())
        project = Path(__file__).resolve().parents[2]
        cfg["output_dir"] = str((project / cfg["output_dir"]).resolve())
    else:
        cfg = source_cfg
    root, out = Path(cfg["output_dir"]).resolve(), Path(output_dir).resolve()
    if out == root or root in out.parents:
        raise ValueError("audit output must be outside the immutable source study")
    manifest_path = root / "prepared/cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("complete source cache required")
    watched = {str(manifest_path): _sha(manifest_path)}
    for entry in list(manifest["arrays"].values()) + list(manifest["fold_arrays"].values()):
        path = Path(entry["path"])
        actual = _sha(path)
        if actual != entry["sha256"]:
            raise ValueError(f"immutable source cache changed: {path}")
        watched[str(path)] = actual
    registration_path = root / "preregistration.json"
    if registration_path.exists():
        watched[str(registration_path)] = _sha(registration_path)
        if watched[str(registration_path)] != manifest["registration_sha256"]:
            raise ValueError("cache must match the source registration")
    arrays = {name: np.load(entry["path"], mmap_mode="r", allow_pickle=False)
              for name, entry in manifest["arrays"].items()}
    times, decisions = np.asarray(arrays["index_ns"], np.int64), np.asarray(arrays["decision_id"], np.int64)
    raw_times = np.asarray(arrays["raw_index_ns"], np.int64)
    if (np.any(np.diff(times) <= 0) or np.any(np.diff(raw_times) != BAR_NS)
            or np.any(decisions < 0) or np.any(decisions >= len(raw_times))
            or not np.array_equal(raw_times[decisions], times)):
        raise ValueError("source indices must match the regular raw grid exactly")
    policy = cfg["policy"]
    context, horizon, purge = (int(policy[key]) for key in ("context", "horizon", "purge_bars"))
    effective = int(manifest.get("effective_raw_history_bars", policy.get("effective_raw_history_bars", 2 * context - 1)))
    context_good = _context_valid(arrays["x_grid"], decisions, context)
    flags = {"context_cache_matches_recalculation": bool(np.array_equal(context_good, arrays["context_valid"])),
             "purge_covers_teacher_execution_delay": purge >= horizon + 1}
    raw_close = np.asarray(arrays["raw_close"])
    teacher_valid = np.asarray(arrays["teacher_valid"], bool)
    expected_teacher = (decisions >= 1) & (decisions + horizon < len(raw_close))
    teacher_rows = np.flatnonzero(expected_teacher)
    path_prices = raw_close[decisions[teacher_rows, None] + np.arange(horizon + 1)]
    expected_teacher[teacher_rows] = np.isfinite(path_prices).all(axis=1) & (path_prices > 0).all(axis=1)
    teacher_rows = np.flatnonzero(expected_teacher)
    expected_paths = np.log(raw_close[decisions[teacher_rows, None] + np.arange(1, horizon + 1)]
                            / raw_close[decisions[teacher_rows, None]])
    flags["teacher_valid_cache_matches_raw_prices"] = bool(np.array_equal(expected_teacher, teacher_valid))
    flags["teacher_paths_match_delayed_raw_price_labels"] = bool(np.allclose(
        expected_paths, arrays["future_path_log_returns"][teacher_rows], rtol=1e-12, atol=1e-12))
    valid = context_good & np.isfinite(arrays["f"]).all(axis=1) & np.isfinite(arrays["bar_log_return"])
    folds = []
    split_rows, normalizers, overlaps, failures = [], [], [], []
    leakage = False
    for meta in manifest["folds"]:
        with np.load(manifest["fold_arrays"][str(meta["fold"])]["path"], allow_pickle=False) as contents:
            saved = {key: contents[key] for key in contents.files}
        folds.append((meta, saved))
        ranges = _ranges(meta)
        chronological = (pd.Timestamp(meta["train_start"]) < pd.Timestamp(meta["train_end"])
            <= pd.Timestamp(meta["val_start"]) < pd.Timestamp(meta["val_end"])
            <= pd.Timestamp(meta["test_start"]) < pd.Timestamp(meta["test_end"]))
        if not chronological:
            failures.append(f"fold{meta['fold']}: nonchronological ranges")
            leakage = True
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            count = len(np.intersect1d(saved[left], saved[right]))
            if count:
                failures.append(f"fold{meta['fold']}: {left}/{right} share {count} rows")
                leakage = True
        for split, (start, end) in ranges.items():
            lower, upper = pd.Timestamp(start).value, pd.Timestamp(end).value
            ids = np.asarray(saved[split], np.int64)
            expected = np.flatnonzero(valid & (times >= lower) & (times + purge * BAR_NS < upper))
            matching = np.array_equal(ids, expected)
            d = decisions[ids]
            teacher_ends = times[ids] + (horizon + 1) * BAR_NS
            wm_ids = np.asarray(saved["wm_" + split], np.int64)
            wm_ends = np.asarray(arrays["wm_label_end_ns"])[decisions[wm_ids]]
            wm_expected = ids[np.asarray(arrays["wm_mask_grid"])[d]
                              & (np.asarray(arrays["wm_label_end_ns"])[d] < upper)]
            label_cross = int(np.sum(teacher_ends >= upper))
            wm_cross = int(np.sum(wm_ends >= upper))
            future_input = int(np.sum(raw_times[d - 1] >= times[ids]))
            too_early = int(np.sum(d - effective < 0))
            teacher_bad = int(np.sum(~np.asarray(arrays["teacher_valid"])[ids]))
            if label_cross or wm_cross or future_input:
                leakage = True
            if not matching or not np.array_equal(wm_ids, wm_expected) or label_cross or wm_cross or future_input or too_early or teacher_bad:
                failures.append(f"fold{meta['fold']}/{split}: saved split, labels or causal input invalid")
            first, last = _range(times[ids])
            split_rows.append({"fold": meta["fold"], "split": split,
                "role": "Train_fit" if split == "train" else "Test_report_only" if split == "test" else "Val_model_selection",
                "nominal_start": str(pd.Timestamp(start)), "nominal_end_exclusive": str(pd.Timestamp(end)),
                "n_rows": len(ids), "first_decision": first, "last_decision": last,
                "purge_bars": purge, "purge_hours": purge / 4,
                "teacher_label_horizon_bars": horizon, "teacher_label_end_includes_delay_bar": True,
                "max_teacher_label_end": _range(teacher_ends)[1], "teacher_labels_crossing_split": label_cross,
                "n_wm_rows": len(wm_ids), "max_wm_label_end": _range(wm_ends)[1], "wm_labels_crossing_split": wm_cross,
                "first_input_raw_timestamp": _stamp(raw_times[max(int(d.min()) - effective, 0)]) if len(d) else None,
                "latest_input_raw_timestamp": _stamp(raw_times[int(d.max()) - 1]) if len(d) else None,
                "rows_with_input_history_before_split_start": int(np.sum(raw_times[np.maximum(d - effective, 0)] < lower)),
                "past_split_input_overlap_is_allowed": True, "future_input_rows": future_input,
                "negative_history_start_rows": too_early, "invalid_teacher_rows": teacher_bad,
                "saved_split_matches_exact_source_rule": bool(matching), "normalizer_source_ids_sha256": _ids_sha(saved["train"])})
        train = np.asarray(saved["train"], np.int64)
        for name, values, ids in (("x", arrays["x_grid"], decisions[train]), ("f", arrays["f"], train)):
            mean, std = _normalizer(values, ids)
            mean_error = float(np.max(np.abs(mean - saved[name + "_mean"])))
            std_error = float(np.max(np.abs(std - saved[name + "_std"])))
            match = bool(np.allclose(mean, saved[name + "_mean"], rtol=1e-11, atol=1e-12)
                         and np.allclose(std, saved[name + "_std"], rtol=1e-11, atol=1e-12))
            if not match:
                failures.append(f"fold{meta['fold']}: {name} normalizer is not the exact Train-only fit")
                leakage = True
            normalizers.append({"fold": meta["fold"], "normalizer": name, "fit_split": "Train_only",
                "n_fit_rows": len(train), "first_fit_decision": _range(times[train])[0],
                "last_fit_decision": _range(times[train])[1], "source_train_ids_sha256": _ids_sha(train),
                "max_mean_error": mean_error, "max_std_error": std_error, "matches_train_only_recalculation": match})
    for earlier, left in folds:
        for later, right in folds:
            if later["fold"] <= earlier["fold"]:
                continue
            for split in ("train", "val", "test"):
                common = np.intersect1d(left["test"], right[split])
                allowed = split != "test" and pd.Timestamp(later[split + "_end"]) >= pd.Timestamp(earlier["test_end"])
                overlaps.append({"earlier_fold": earlier["fold"], "later_fold": later["fold"],
                    "earlier_split": "test", "later_split": split, "n_shared_rows": len(common),
                    "allowed_historical_walkforward_reuse": bool(allowed),
                    "interpretation": "historical outcomes known before later fit" if allowed else "Test quarters must be disjoint"})
                if len(common) and not allowed:
                    failures.append("unallowed cross-fold Test overlap")
                    leakage = True
    bars = _classify_test_bars(arrays, folds, cfg)
    if not bars.classification_matches_saved_split.all():
        failures.append("exclusive exclusion reasons do not reconstruct the saved Test cohort")
    kept = bars.loc[bars.actually_kept]
    reset_rows, reset_audit, reset_path = _reset_audit(root, cfg, kept.timestamp.astype("int64").to_numpy(), kept.fold.to_numpy())
    if reset_audit["available"]:
        watched[str(reset_path)] = reset_audit["source_sha256"]
        if not all(reset_audit[k] for k in ("cohort_equal", "reset_mask_equal", "free_endowed_one_position_reset_verified")):
            failures.append("actual ledger reset accounting differs from the frozen contract")
    provenance_path = root / "normalizer_provenance.json"
    provenance_count = 0
    if provenance_path.exists():
        watched[str(provenance_path)] = _sha(provenance_path)
        entries = json.loads(provenance_path.read_text())["normalizers"]
        train_hashes = {meta["fold"]: _ids_sha(saved["train"]) for meta, saved in folds}
        for entry in entries:
            provenance_count += 1
            if (entry["fit_split"] != "train" or entry["source_id"] != train_hashes[entry["fold"]]
                    or entry["expected_source_id"] != train_hashes[entry["fold"]]):
                failures.append("pipeline normalizer provenance differs from its purged Train IDs")
                leakage = True
    latent_normalizers, latent_failures = _latent_normalizers(root, cfg, folds, manifest, watched)
    failures.extend(latent_failures)
    leakage = leakage or bool(latent_failures)
    if not all(flags.values()):
        failures.append("invalid cached causal context, teacher labels or insufficient label purge")
    if not all(_sha(path) == digest for path, digest in watched.items()):
        raise ValueError("immutable source changed during audit")
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    tables = {"split_audit.csv": pd.DataFrame(split_rows), "normalizer_audit.csv": pd.DataFrame(normalizers),
              "latent_normalizer_audit.csv": pd.DataFrame(latent_normalizers),
              "cross_fold_overlaps.csv": pd.DataFrame(overlaps), "exclusion_bars.csv": bars,
              "exclusion_counts.csv": bars.groupby(["fold", "reason"], sort=True).size().rename("n_bars").reset_index(),
              "reset_audit.csv": reset_rows}
    for name, frame in tables.items():
        path = out / name
        frame.to_csv(path, index=False)
        artifacts[name] = {"path": str(path), "sha256": _sha(path), "n_rows": len(frame)}
    report = {"passed": not failures, "leakage_found": bool(leakage), "violations": failures,
        "source_root": str(root), "source_cache_manifest_sha256": watched[str(manifest_path)],
        "cache_arrays_and_folds_sha256_verified": True, "source_files_unchanged": True,
        "n_folds": len(folds), "n_test_market_rows": int(bars.actually_kept.sum()),
        "n_nominal_test_bars": len(bars), "n_excluded_test_bars": int((~bars.actually_kept).sum()),
        "n_pipeline_normalizer_provenance_rows_checked": provenance_count, "checks": flags,
        "n_latent_normalizers_checked": len(latent_normalizers),
        "exclusive_exclusion_reason_priority": list(REASON_PRIORITY),
        "exclusion_totals": {str(k): int(v) for k, v in bars.reason.value_counts().items()},
        "input_contract": {"actor_tokens": context, "effective_raw_history_bars": effective,
            "latest_input_raw_bar": "decision minus one bar", "historical_context_before_split_start_allowed": True,
            "underlying_token_feature_causality": "source frozen code contract; token formulas not re-fitted or exhaustively re-derived"},
        "walkforward_contract": "past Test outcomes may enter later Train/Val once known; within-fold future labels may not cross split ends",
        "market_sample_scope": "same market path reused by seeds; adding seeds does not add independent market observations",
        "feature_selection_scope": "F selected retrospectively using historical forecast Test; no fresh holdout claim",
        "cohort_scope": "saved retrospective common cohort; masks are not a live availability rule",
        "forecast_missing_reason_limit": "absence is known at common-ledger level; producer context/future-mask/purge/head-specific cause is not uniquely identified",
        "reset_accounting": reset_audit, "artifacts": artifacts}
    path = out / "audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {**report, "audit_json": str(path)}
