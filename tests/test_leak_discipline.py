"""リーク規律の回帰テスト.

「時点 t 以降のデータを改変しても、時点 t 以前の特徴量・レジーム確率が変わらない」
ことを機械的に検証する。将来のルックアヘッド混入をここで検出する。
"""
import unittest

import numpy as np
import pandas as pd

from unidream.data.features import compute_basis_features, compute_features

from unidream.eval.regime import HAS_HMMLEARN, RegimeDetector
from unidream.eval.backtest import Backtest, compute_sortino


def _synthetic_ohlcv(n_bars: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n_bars, freq="15min")
    log_ret = rng.normal(0.0, 0.002, size=n_bars)
    close = 20000.0 * np.exp(np.cumsum(log_ret))
    spread = np.abs(rng.normal(0.0, 0.001, size=n_bars))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.0005, size=n_bars)),
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": np.exp(rng.normal(3.0, 0.3, size=n_bars)),
        },
        index=index,
    )


class FeatureLeakDisciplineTest(unittest.TestCase):
    def test_features_unchanged_when_future_bars_perturbed(self) -> None:
        n_bars = 3000
        t_cut = 2000
        df = _synthetic_ohlcv(n_bars)
        t_cut_ts = df.index[t_cut]

        funding_index = pd.date_range(df.index[0], df.index[-1], freq="8h")
        rng = np.random.default_rng(11)
        funding_df = pd.DataFrame(
            {"funding_rate": rng.normal(0.0001, 0.0002, size=len(funding_index))},
            index=funding_index,
        )

        df_perturbed = df.copy()
        df_perturbed.loc[df.index >= t_cut_ts, ["open", "high", "low", "close"]] *= 1.05
        df_perturbed.loc[df.index >= t_cut_ts, "volume"] *= 2.0
        funding_perturbed = funding_df.copy()
        funding_perturbed.loc[funding_df.index >= t_cut_ts, "funding_rate"] += 0.001

        base = compute_features(df, zscore_window_days=5, funding_df=funding_df)
        pert = compute_features(df_perturbed, zscore_window_days=5, funding_df=funding_perturbed)

        # features[t] はバー t-1 までの情報のみ → t_cut 以前の行は完全一致するはず
        idx = base.index[base.index <= t_cut_ts]
        self.assertGreater(len(idx), 500)
        np.testing.assert_allclose(
            base.loc[idx].to_numpy(),
            pert.loc[idx].to_numpy(),
            atol=1e-10,
            err_msg="未来バーの改変が過去の特徴量に影響した（ルックアヘッド混入）",
        )

    def test_basis_features_do_not_backfill_future_mark_price(self) -> None:
        n_bars = 400
        mark_start = 200
        df = _synthetic_ohlcv(n_bars, seed=3)
        mark_df = pd.DataFrame(
            {"mark_close": df["close"].iloc[mark_start:] * 1.001},
            index=df.index[mark_start:],
        )
        basis = compute_basis_features(df["close"], mark_df)

        # mark データ開始前は中立値 0.0（未来の先物価格レベルを含まない）
        head = basis.loc[basis.index < df.index[mark_start]]
        self.assertTrue((head == 0.0).all().all())

        # mark 開始直後の値を変えても、開始前の行は不変
        mark_df2 = mark_df.copy()
        mark_df2.iloc[0:10, 0] *= 1.5
        basis2 = compute_basis_features(df["close"], mark_df2)
        np.testing.assert_allclose(
            head.to_numpy(),
            basis2.loc[head.index].to_numpy(),
            atol=0.0,
            err_msg="mark 開始前の basis が未来の mark 価格に依存している",
        )


@unittest.skipUnless(HAS_HMMLEARN, "hmmlearn が必要")
class RegimeProbsCausalTest(unittest.TestCase):
    def test_causal_probs_unchanged_when_future_returns_perturbed(self) -> None:
        rng = np.random.default_rng(5)
        train_returns = rng.normal(0.0, 0.01, size=800)
        test_returns = rng.normal(0.0, 0.01, size=300)

        det = RegimeDetector(n_states=3)
        det.fit(train_returns)

        t_cut = 150
        test_perturbed = test_returns.copy()
        test_perturbed[t_cut:] += 0.05

        probs = det.predict_proba_causal(test_returns)
        probs_perturbed = det.predict_proba_causal(test_perturbed)

        # probs[t] は returns[0..t-1] のみ使用 → t_cut 以前（t_cut 自身を含む）は不変
        np.testing.assert_allclose(
            probs[: t_cut + 1],
            probs_perturbed[: t_cut + 1],
            atol=0.0,
            err_msg="因果的レジーム確率が未来のリターンに依存している",
        )
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)

    def test_smoothed_probs_are_not_causal(self) -> None:
        # ドキュメント目的の対照テスト: 平滑化版は未来に依存する（＝方策入力に使ってはいけない）
        rng = np.random.default_rng(5)
        det = RegimeDetector(n_states=3)
        det.fit(rng.normal(0.0, 0.01, size=800))
        test_returns = rng.normal(0.0, 0.01, size=300)
        test_perturbed = test_returns.copy()
        test_perturbed[150:] += 0.05
        smoothed = det.predict_proba(test_returns)
        smoothed_perturbed = det.predict_proba(test_perturbed)
        self.assertFalse(np.allclose(smoothed[:151], smoothed_perturbed[:151]))


class MetricDefinitionTest(unittest.TestCase):
    def test_sortino_uses_standard_downside_deviation(self) -> None:
        pnl = np.asarray([0.01, -0.02, 0.03, -0.01, 0.0], dtype=np.float64)
        ann = 365.0
        downside_dev = np.sqrt(np.mean(np.square(np.minimum(pnl, 0.0))))
        expected = pnl.mean() / downside_dev * np.sqrt(ann)
        self.assertAlmostEqual(compute_sortino(pnl, ann), expected)

    def test_execution_delay_shifts_positions(self) -> None:
        returns = np.asarray([0.01, 0.02, -0.01, 0.03], dtype=np.float64)
        positions = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        no_delay = Backtest(
            returns, positions, spread_bps=0.0, fee_rate=0.0, slippage_bps=0.0, interval="1d"
        ).run()
        delayed = Backtest(
            returns, positions, spread_bps=0.0, fee_rate=0.0, slippage_bps=0.0, interval="1d",
            execution_delay_bars=1,
        ).run()
        # delay=1 では positions が 1 バー後ろにずれる（先頭は最初の決定値で埋める）
        expected_positions = np.asarray([1.0, 1.0, 0.0, 1.0])
        expected_pnl = expected_positions * returns
        np.testing.assert_allclose(delayed.pnl_series, expected_pnl, atol=1e-12)
        self.assertNotAlmostEqual(no_delay.total_return, delayed.total_return)


if __name__ == "__main__":
    unittest.main()
