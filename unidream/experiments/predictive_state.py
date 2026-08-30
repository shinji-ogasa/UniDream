from __future__ import annotations

import numpy as np

from .chronological_oof import (
    ConditionalPathBlocked,
    conditional_path_enabled,
    require_conditional_oof_inputs,
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
    The caller must provide aligned, masked OOF states and provenance for every
    split; early rows remain NaN and are exposed through ``*_mask`` for a later
    stage to exclude explicitly.
    """
    names = list(oof_bundle.get("names", []))
    splits: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        if split not in oof_bundle:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split!r} state; no in-sample fallback is allowed"
            )
        values = np.asarray(oof_bundle[split], dtype=np.float32)
        if values.ndim != 2:
            raise ConditionalPathBlocked(f"conditional OOF {split} state must be 2-D")
        mask_value = oof_bundle.get(f"{split}_mask", oof_bundle.get("prediction_mask"))
        if mask_value is None:
            raise ConditionalPathBlocked(
                f"conditional OOF bundle is missing {split}_mask; early rows cannot be inferred"
            )
        mask = np.asarray(mask_value, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(values):
            raise ConditionalPathBlocked(f"conditional OOF {split}_mask is not row-aligned")
        if np.any(mask & ~np.isfinite(values).all(axis=1)):
            raise ConditionalPathBlocked(f"conditional OOF {split} contains a non-finite usable row")
        # A finite state without a mask would be an implicit in-sample fill.
        if np.any(~mask & np.isfinite(values).any(axis=1)):
            raise ConditionalPathBlocked(
                f"conditional OOF {split} has finite or partially finite values outside its OOF mask"
            )
        splits[split] = values
        masks[split] = mask
        if not names:
            names = [f"wm_oof_state_{i}" for i in range(values.shape[1])]
        if len(names) != values.shape[1]:
            raise ConditionalPathBlocked(f"conditional OOF {split} names do not match state width")

    provenance = dict(oof_bundle.get("provenance") or {})
    # ``validate_oof_result`` is also applied when the producer supplied raw
    # prediction/origin fields.  It catches accidental in-sample metadata.
    if "predictions" in oof_bundle:
        validate_oof_result(oof_bundle)
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
        if bool(detail.get("in_sample", False)):
            raise ConditionalPathBlocked(
                f"conditional OOF {component} is marked in_sample"
            )
        scheme = str(detail.get("fit_scheme", "")).strip().lower()
        if scheme not in allowed_schemes:
            raise ConditionalPathBlocked(
                f"conditional OOF {component} fit_scheme must be chronological OOF"
            )
    result = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "train_mask": masks["train"],
        "val_mask": masks["val"],
        "test_mask": masks["test"],
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
