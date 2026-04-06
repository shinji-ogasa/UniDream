from __future__ import annotations

import numpy as np

from unidream.eval.regime import RegimeDetector


def fit_fold_regimes(
    *,
    train_returns,
    val_returns,
    test_returns,
    n_states: int,
) -> dict:
    hmm_det = RegimeDetector(n_states=n_states)
    hmm_det.fit(train_returns)
    return {
        "detector": hmm_det,
        "regime_dim": n_states,
        "train_regime_probs": hmm_det.predict_proba(train_returns).astype(np.float32),
        "val_regime_probs": hmm_det.predict_proba(val_returns).astype(np.float32),
        "test_regime_probs": hmm_det.predict_proba(test_returns).astype(np.float32),
    }
