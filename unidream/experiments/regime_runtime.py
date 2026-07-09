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
    # predict_proba は forward-backward 平滑化（系列全体＝未来を使う）のため
    # 方策入力にはルックアヘッドになる。filtering 版（probs[t] は returns[<t] のみ）を使う。
    return {
        "detector": hmm_det,
        "regime_dim": n_states,
        "train_regime_probs": hmm_det.predict_proba_causal(train_returns).astype(np.float32),
        "val_regime_probs": hmm_det.predict_proba_causal(val_returns).astype(np.float32),
        "test_regime_probs": hmm_det.predict_proba_causal(test_returns).astype(np.float32),
    }
