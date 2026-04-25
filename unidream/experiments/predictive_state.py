from __future__ import annotations

import numpy as np


def _concat_selected_aux(
    aux: dict[str, np.ndarray],
    heads: list[str],
) -> np.ndarray | None:
    parts = []
    for head in heads:
        arr = aux.get(head)
        if arr is not None and arr.size > 0:
            parts.append(np.asarray(arr, dtype=np.float32))
    if not parts:
        return None
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_wm_predictive_state_bundle(
    *,
    wm_trainer,
    wfo_dataset,
    z_train: np.ndarray,
    h_train: np.ndarray,
    seq_len: int,
    ac_cfg: dict,
    log_ts,
) -> dict | None:
    if not bool(ac_cfg.get("use_wm_predictive_state", False)):
        return None

    heads = list(ac_cfg.get("wm_predictive_state_heads", ["return", "vol", "drawdown"]))
    train = _concat_selected_aux(wm_trainer.predict_auxiliary_from_encoded(z_train, h_train), heads)
    if train is None or train.shape[1] == 0:
        print(f"[{log_ts()}] [PredictiveState] skipped: no active WM auxiliary heads")
        return None

    enc_val = wm_trainer.encode_sequence(wfo_dataset.val_features, seq_len=seq_len)
    val = _concat_selected_aux(
        wm_trainer.predict_auxiliary_from_encoded(enc_val["z"], enc_val["h"]),
        heads,
    )
    enc_test = wm_trainer.encode_sequence(wfo_dataset.test_features, seq_len=seq_len)
    test = _concat_selected_aux(
        wm_trainer.predict_auxiliary_from_encoded(enc_test["z"], enc_test["h"]),
        heads,
    )
    if val is None:
        val = np.zeros((len(wfo_dataset.val_features), train.shape[1]), dtype=np.float32)
    if test is None:
        test = np.zeros((len(wfo_dataset.test_features), train.shape[1]), dtype=np.float32)

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
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
