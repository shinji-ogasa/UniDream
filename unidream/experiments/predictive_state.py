from __future__ import annotations

import numpy as np

from .chronological_oof import (
    ChronologicalOOFError,
    ConditionalPathBlocked,
    conditional_path_enabled,
    require_conditional_oof_inputs,
    strict_bool_array,
    strict_bool_value,
    strict_integer_array,
    validate_oof_result,
)


def _concat_selected_aux(
    aux: dict[str, np.ndarray],
    heads: list[str],
) -> np.ndarray | None:
    parts = []
    for head in heads:
        arr = aux.get(head)
        if arr is not None and arr.size > 0:
            parts.append(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    if not parts:
        return None
    return np.concatenate(parts, axis=1).astype(np.float32)


def _conditional_oof_state_bundle(
    *,
    oof_bundle: dict,
    ac_cfg: dict,
    log_ts,
) -> dict:
    """Validate and return a precomputed chronological OOF state bundle.

    Full WM fold re-training is deliberately not hidden behind this adapter.
    The caller must provide the complete raw chronological OOF result (including
    eligibility masks/provenance), plus exact raw-prediction views for every
    split.  Each view must carry strict, increasing ``*_row_indices`` and its
    values, state mask, and both eligibility masks must equal the corresponding
    rows of the validated raw result.  Transformed/standardized views are
    blocked until a causal transform artifact and input-row mapping exist;
    early rows remain NaN and are exposed through ``*_mask`` for a later stage
    to exclude explicitly.
    """
    # Validate the raw producer result before inspecting any split view.  A
    # split-only caller must never be able to bypass the eligibility and
    # in-sample provenance contract by supplying plausible state masks.
    try:
        validate_oof_result(oof_bundle)
    except ChronologicalOOFError as exc:
        raise ConditionalPathBlocked(str(exc)) from exc
    raw_predictions = np.asarray(oof_bundle["predictions"])
    raw_prediction_mask = strict_bool_array(
        oof_bundle["prediction_mask"]
        if "prediction_mask" in oof_bundle
        else oof_bundle["oof_mask"],
        name="conditional OOF prediction_mask",
    )
    raw_prediction_eligibility = strict_bool_array(
        oof_bundle["prediction_eligibility_mask"],
        name="conditional OOF prediction_eligibility_mask",
    )
    raw_training_eligibility = strict_bool_array(
        oof_bundle["training_label_eligibility_mask"],
        name="conditional OOF training_label_eligibility_mask",
    )
    for statistic in ("mean", "std"):
        if statistic not in oof_bundle:
            continue
        if np.asarray(oof_bundle[statistic]).size:
            raise ConditionalPathBlocked(
                f"conditional OOF {statistic} would imply a transformed state; "
                "causal transform artifacts are not accepted until their row mapping is implemented"
            )
    provenance = dict(oof_bundle.get("provenance") or {})
    names = list(oof_bundle.get("names", []))
    splits: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    split_masks: dict[str, np.ndarray] = {}
    split_row_indices: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        if split not in oof_bundle:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split!r} state; no in-sample fallback is allowed"
            )
        values = np.asarray(oof_bundle[split])
        if values.ndim != 2:
            raise ConditionalPathBlocked(f"conditional OOF {split} state must be 2-D")
        row_indices_value = oof_bundle.get(f"{split}_row_indices")
        if row_indices_value is None:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split}_row_indices; "
                "split state cannot be mapped to the validated raw OOF rows"
            )
        try:
            row_indices = strict_integer_array(
                row_indices_value,
                name=f"conditional OOF {split}_row_indices",
            )
        except ChronologicalOOFError as exc:
            raise ConditionalPathBlocked(str(exc)) from exc
        if row_indices.ndim != 1 or len(row_indices) != len(values):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_row_indices must be a 1-D row-aligned array"
            )
        if len(row_indices) > 1 and np.any(np.diff(row_indices) <= 0):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_row_indices must be strictly increasing and unique"
            )
        if np.any(row_indices < 0) or np.any(row_indices >= len(raw_predictions)):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_row_indices are outside raw OOF rows"
            )
        if values.shape[1] != raw_predictions.shape[1]:
            raise ConditionalPathBlocked(
                f"conditional OOF {split} state width does not match raw predictions"
            )
        if values.dtype != raw_predictions.dtype:
            raise ConditionalPathBlocked(
                f"conditional OOF {split} must be an exact raw OOF view; "
                "transformed state requires an explicit causal transform artifact"
            )
        mask_value = oof_bundle.get(f"{split}_mask")
        if mask_value is None:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split}_mask; early rows cannot be inferred"
            )
        try:
            mask = strict_bool_array(mask_value, name=f"conditional OOF {split}_mask")
        except ChronologicalOOFError as exc:
            raise ConditionalPathBlocked(str(exc)) from exc
        if mask.ndim != 1 or len(mask) != len(values):
            raise ConditionalPathBlocked(f"conditional OOF {split}_mask is not row-aligned")
        try:
            finite_values = np.isfinite(values)
        except (TypeError, ValueError) as exc:
            raise ConditionalPathBlocked(
                f"conditional OOF {split} state must contain numeric values"
            ) from exc
        if np.any(mask & ~finite_values.all(axis=1)):
            raise ConditionalPathBlocked(f"conditional OOF {split} contains a non-finite usable row")
        # A finite state without a mask would be an implicit in-sample fill.
        if np.any(~mask & finite_values.any(axis=1)):
            raise ConditionalPathBlocked(
                f"conditional OOF {split} has finite or partially finite values outside its OOF mask"
            )
        split_prediction_eligibility_value = oof_bundle.get(
            f"{split}_prediction_eligibility_mask"
        )
        split_training_eligibility_value = oof_bundle.get(
            f"{split}_training_label_eligibility_mask"
        )
        if split_prediction_eligibility_value is None:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split}_prediction_eligibility_mask"
            )
        if split_training_eligibility_value is None:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split}_training_label_eligibility_mask"
            )
        try:
            split_prediction_eligibility = strict_bool_array(
                split_prediction_eligibility_value,
                name=f"conditional OOF {split}_prediction_eligibility_mask",
            )
            split_training_eligibility = strict_bool_array(
                split_training_eligibility_value,
                name=f"conditional OOF {split}_training_label_eligibility_mask",
            )
        except ChronologicalOOFError as exc:
            raise ConditionalPathBlocked(str(exc)) from exc
        for name, eligibility in (
            ("prediction", split_prediction_eligibility),
            ("training_label", split_training_eligibility),
        ):
            if eligibility.ndim != 1 or len(eligibility) != len(values):
                raise ConditionalPathBlocked(
                    f"conditional OOF {split}_{name}_eligibility_mask is not row-aligned"
                )
        if np.any(mask & ~split_prediction_eligibility):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_mask contains a row outside its prediction eligibility"
            )
        if np.any(split_training_eligibility & ~split_prediction_eligibility):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_training_label_eligibility_mask contains a row outside its prediction eligibility"
            )
        expected_values = raw_predictions[row_indices]
        if not np.array_equal(values, expected_values, equal_nan=True):
            raise ConditionalPathBlocked(
                f"conditional OOF {split} values do not equal the indexed raw OOF predictions"
            )
        if not np.array_equal(mask, raw_prediction_mask[row_indices]):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_mask does not equal the indexed raw OOF prediction_mask"
            )
        if not np.array_equal(
            split_prediction_eligibility,
            raw_prediction_eligibility[row_indices],
        ):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_prediction_eligibility_mask does not equal "
                "the indexed raw origin eligibility mask"
            )
        if not np.array_equal(
            split_training_eligibility,
            raw_training_eligibility[row_indices],
        ):
            raise ConditionalPathBlocked(
                f"conditional OOF {split}_training_label_eligibility_mask does not equal "
                "the indexed raw training-label eligibility mask"
            )
        splits[split] = values
        masks[split] = mask
        split_row_indices[split] = row_indices
        # Keep split-level origin/training provenance alongside the state mask;
        # neither can be inferred from a split-only state view.
        split_masks[f"{split}_prediction_eligibility_mask"] = split_prediction_eligibility
        split_masks[f"{split}_training_label_eligibility_mask"] = split_training_eligibility
        if not names:
            names = [f"wm_oof_state_{i}" for i in range(values.shape[1])]
        if len(names) != values.shape[1]:
            raise ConditionalPathBlocked(f"conditional OOF {split} names do not match state width")

    normalizer = provenance.get("normalizer", "")
    normalizer_name = (
        str(normalizer).strip().lower() if not isinstance(normalizer, dict) else ""
    )
    if (
        normalizer_name not in {"", "expanding_prefix", "oof", "precomputed_oof"}
        and not isinstance(normalizer, dict)
    ):
        raise ConditionalPathBlocked(
            "conditional OOF normalizer provenance must be expanding_prefix or precomputed_oof"
        )
    allowed_schemes = {"chronological_oof", "expanding_origin", "rolling_origin", ""}
    for component in ("normalizer", "calibrator", "teacher_weight"):
        detail = provenance.get(component)
        if not isinstance(detail, dict):
            continue
        if "in_sample" in detail:
            try:
                in_sample = strict_bool_value(
                    detail["in_sample"],
                    name=f"conditional OOF {component}.in_sample",
                )
            except ChronologicalOOFError as exc:
                raise ConditionalPathBlocked(str(exc)) from exc
        else:
            in_sample = False
        if in_sample:
            raise ConditionalPathBlocked(
                f"conditional OOF {component} is marked in_sample"
            )
        scheme = str(detail.get("fit_scheme", "")).strip().lower()
        if scheme not in allowed_schemes:
            raise ConditionalPathBlocked(
                f"conditional OOF {component} fit_scheme must be chronological OOF"
            )
    result = {
        # Retain the validated raw contract alongside split views so a later
        # consumer does not lose the target-cutoff and origin provenance that
        # justified accepting these states.
        "predictions": np.array(raw_predictions, copy=True),
        "prediction_mask": raw_prediction_mask.copy(),
        "prediction_eligibility_mask": raw_prediction_eligibility.copy(),
        "training_label_eligibility_mask": raw_training_eligibility.copy(),
        "target_end_exclusive": np.array(
            oof_bundle["target_end_exclusive"],
            dtype=np.int64,
            copy=True,
        ),
        "origins": [dict(origin) for origin in oof_bundle["origins"]],
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "train_mask": masks["train"],
        "val_mask": masks["val"],
        "test_mask": masks["test"],
        "train_row_indices": split_row_indices["train"],
        "val_row_indices": split_row_indices["val"],
        "test_row_indices": split_row_indices["test"],
        "train_prediction_eligibility_mask": split_masks[
            "train_prediction_eligibility_mask"
        ],
        "val_prediction_eligibility_mask": split_masks[
            "val_prediction_eligibility_mask"
        ],
        "test_prediction_eligibility_mask": split_masks[
            "test_prediction_eligibility_mask"
        ],
        "train_training_label_eligibility_mask": split_masks[
            "train_training_label_eligibility_mask"
        ],
        "val_training_label_eligibility_mask": split_masks[
            "val_training_label_eligibility_mask"
        ],
        "test_training_label_eligibility_mask": split_masks[
            "test_training_label_eligibility_mask"
        ],
        "mean": np.asarray(oof_bundle.get("mean", []), dtype=np.float32),
        "std": np.asarray(oof_bundle.get("std", []), dtype=np.float32),
        "names": names,
        "provenance": provenance,
    }
    print(
        f"[{log_ts()}] [PredictiveState] chronological OOF bundle accepted: "
        f"train_usable={int(masks['train'].sum())}/{len(masks['train'])} "
        f"val_usable={int(masks['val'].sum())}/{len(masks['val'])} "
        f"test_usable={int(masks['test'].sum())}/{len(masks['test'])}"
    )
    return result


def build_wm_predictive_state_bundle(
    *,
    wm_trainer,
    wfo_dataset,
    z_train: np.ndarray,
    h_train: np.ndarray,
    seq_len: int,
    ac_cfg: dict,
    log_ts,
    oof_bundle: dict | None = None,
) -> dict | None:
    if conditional_path_enabled(ac_cfg):
        require_conditional_oof_inputs(
            config=ac_cfg,
            oof_bundle=oof_bundle,
            caller="build_wm_predictive_state_bundle",
        )
        if oof_bundle is not None:
            return _conditional_oof_state_bundle(
                oof_bundle=oof_bundle,
                ac_cfg=ac_cfg,
                log_ts=log_ts,
            )
        raise ConditionalPathBlocked(
            "conditional predictive state requires an explicit chronological OOF bundle"
        )
    if not bool(ac_cfg.get("use_wm_predictive_state", False)):
        return None

    heads = list(ac_cfg.get("wm_predictive_state_heads", ["return", "vol", "drawdown"]))
    train = _concat_selected_aux(
        wm_trainer.predict_auxiliary_from_encoded(z_train, h_train, features=wfo_dataset.train_features),
        heads,
    )
    if train is None or train.shape[1] == 0:
        print(f"[{log_ts()}] [PredictiveState] skipped: no active WM auxiliary heads")
        return None

    enc_val = wm_trainer.encode_sequence(wfo_dataset.val_features, seq_len=seq_len)
    val = _concat_selected_aux(
        wm_trainer.predict_auxiliary_from_encoded(enc_val["z"], enc_val["h"], features=wfo_dataset.val_features),
        heads,
    )
    enc_test = wm_trainer.encode_sequence(wfo_dataset.test_features, seq_len=seq_len)
    test = _concat_selected_aux(
        wm_trainer.predict_auxiliary_from_encoded(enc_test["z"], enc_test["h"], features=wfo_dataset.test_features),
        heads,
    )
    if val is None:
        val = np.zeros((len(wfo_dataset.val_features), train.shape[1]), dtype=np.float32)
    if test is None:
        test = np.zeros((len(wfo_dataset.test_features), train.shape[1]), dtype=np.float32)

    mean = np.nan_to_num(train.mean(axis=0, keepdims=True), nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(train.std(axis=0, keepdims=True), nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    if bool(ac_cfg.get("wm_predictive_state_standardize", True)):
        train = (train - mean) / std
        val = (val - mean) / std
        test = (test - mean) / std
    clip = float(ac_cfg.get("wm_predictive_state_clip", 5.0))
    if clip > 0.0:
        train = np.clip(train, -clip, clip)
        val = np.clip(val, -clip, clip)
        test = np.clip(test, -clip, clip)

    scale = float(ac_cfg.get("wm_predictive_state_scale", 1.0))
    if scale != 1.0:
        train = train * scale
        val = val * scale
        test = test * scale

    train = np.nan_to_num(train, nan=0.0, posinf=clip if clip > 0.0 else 0.0, neginf=-clip if clip > 0.0 else 0.0).astype(np.float32)
    val = np.nan_to_num(val, nan=0.0, posinf=clip if clip > 0.0 else 0.0, neginf=-clip if clip > 0.0 else 0.0).astype(np.float32)
    test = np.nan_to_num(test, nan=0.0, posinf=clip if clip > 0.0 else 0.0, neginf=-clip if clip > 0.0 else 0.0).astype(np.float32)

    all_names = wm_trainer.predictive_feature_names()
    names = [name for name in all_names if any(name.startswith(f"wm_pred_{head}") for head in heads)]
    print(
        f"[{log_ts()}] [PredictiveState] enabled: dim={train.shape[1]} "
        f"heads={','.join(heads)} standardize={bool(ac_cfg.get('wm_predictive_state_standardize', True))} "
        f"scale={scale:.3g}"
    )
    return {
        "train": train,
        "val": val,
        "test": test,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "names": names[: train.shape[1]],
    }
