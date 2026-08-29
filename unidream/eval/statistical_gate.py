"""Development-only statistical robustness gate.

The gate consumes explicit per-bar paths from development folds.  It does not
load forecast artifacts, infer a split from filenames, or read holdout folds.
Callers must provide additive net-return paths in the same units used by their
backtest contract:

* ``alpha_excess_returns`` is the strategy-minus-benchmark per-bar path;
* ``timing_increment_returns`` is the dynamic-minus-constant per-bar path;
* ``strategy_returns`` is the candidate's net per-bar path for Sharpe/CSCV.

The implementation is deliberately report-only.  A passing result is not a
claim that an existing UniDream experiment passed; no existing result is
loaded by this module.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from math import ceil, exp, log, sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


SCHEMA_VERSION = 1
MAX_DEVELOPMENT_FOLD = 14
DEFAULT_ANNUALIZATION_BARS_PER_YEAR = 365 * 96
DEFAULT_BLOCK_LENGTH_SENSITIVITY = (8, 16, 32)
EULER_MASCHERONI = 0.5772156649015329


def _strict_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _finite_vector(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _finite_scalar(name: str, value: Any) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _development_fold_number(value: Any) -> int:
    fold = _strict_int("fold", value, minimum=0)
    if fold > MAX_DEVELOPMENT_FOLD:
        raise ValueError(
            f"fold {fold} is outside the development scope; "
            f"folds >= {MAX_DEVELOPMENT_FOLD + 1} are forbidden"
        )
    return fold


@dataclass(frozen=True)
class DevelopmentFold:
    """One candidate's aligned per-bar paths for a development fold."""

    fold: int
    alpha_excess_returns: np.ndarray
    timing_increment_returns: np.ndarray
    strategy_returns: np.ndarray | None = None

    def __post_init__(self) -> None:
        fold = _development_fold_number(self.fold)
        alpha = _finite_vector("alpha_excess_returns", self.alpha_excess_returns)
        timing = _finite_vector("timing_increment_returns", self.timing_increment_returns)
        if len(alpha) != len(timing):
            raise ValueError("alpha_excess_returns and timing_increment_returns must align")
        strategy = None
        if self.strategy_returns is not None:
            strategy = _finite_vector("strategy_returns", self.strategy_returns)
            if len(strategy) != len(alpha):
                raise ValueError(
                    "strategy_returns must align with alpha_excess_returns"
                )
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "alpha_excess_returns", alpha)
        object.__setattr__(self, "timing_increment_returns", timing)
        object.__setattr__(self, "strategy_returns", strategy)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DevelopmentFold":
        if not isinstance(payload, Mapping):
            raise ValueError("each development fold must be an object")
        required = ("fold", "alpha_excess_returns", "timing_increment_returns")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"development fold is missing keys: {missing}")
        return cls(
            fold=payload["fold"],
            alpha_excess_returns=payload["alpha_excess_returns"],
            timing_increment_returns=payload["timing_increment_returns"],
            strategy_returns=payload.get("strategy_returns"),
        )


@dataclass(frozen=True)
class CandidatePath:
    """A named candidate with development-only fold paths."""

    name: str
    folds: tuple[DevelopmentFold, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("candidate name must not be empty")
        normalized: list[DevelopmentFold] = []
        for item in self.folds:
            normalized.append(
                item if isinstance(item, DevelopmentFold) else DevelopmentFold.from_mapping(item)
            )
        if not normalized:
            raise ValueError(f"candidate {name!r} has no development folds")
        fold_ids = [item.fold for item in normalized]
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError(f"candidate {name!r} has duplicate development folds")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "folds", tuple(normalized))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidatePath":
        if not isinstance(payload, Mapping):
            raise ValueError("each candidate must be an object")
        if "name" not in payload or "folds" not in payload:
            raise ValueError("candidate requires name and folds")
        return cls(name=payload["name"], folds=tuple(payload["folds"]))


@dataclass(frozen=True)
class StressCase:
    """One precomputed cost or regime stress observation."""

    name: str
    kind: str
    alpha_excess_pt: float
    timing_increment_pt: float

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        kind = str(self.kind).strip().lower()
        if not name:
            raise ValueError("stress case name must not be empty")
        if kind not in {"cost", "regime"}:
            raise ValueError("stress case kind must be 'cost' or 'regime'")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "alpha_excess_pt", _finite_scalar("stress alpha_excess_pt", self.alpha_excess_pt)
        )
        object.__setattr__(
            self,
            "timing_increment_pt",
            _finite_scalar("stress timing_increment_pt", self.timing_increment_pt),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StressCase":
        if not isinstance(payload, Mapping):
            raise ValueError("each stress case must be an object")
        required = ("name", "kind", "alpha_excess_pt", "timing_increment_pt")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"stress case is missing keys: {missing}")
        return cls(
            name=payload["name"],
            kind=payload["kind"],
            alpha_excess_pt=payload["alpha_excess_pt"],
            timing_increment_pt=payload["timing_increment_pt"],
        )


@dataclass(frozen=True)
class StatisticalGateConfig:
    """Fixed statistical-contract parameters for one gate evaluation."""

    annualization_bars_per_year: float = DEFAULT_ANNUALIZATION_BARS_PER_YEAR
    alpha: float = 0.05
    bootstrap_method: str = "moving_block"
    bootstrap_replicates: int = 1000
    block_length: int = 16
    block_length_sensitivity: tuple[int, ...] = DEFAULT_BLOCK_LENGTH_SENSITIVITY
    seed: int = 7
    min_folds: int = 4
    min_observations: int = 32
    n_trials: int | None = None
    cscv_subperiods: int | None = None
    pbo_max: float = 0.5
    stress_min_pass_rate: float = 1.0
    stress_alpha_floor_pt: float = 0.0
    stress_timing_floor_pt: float = 0.0
    require_cost_stress: bool = True
    require_regime_stress: bool = True

    def __post_init__(self) -> None:
        annualization = _finite_scalar(
            "annualization_bars_per_year", self.annualization_bars_per_year
        )
        if annualization <= 0.0:
            raise ValueError("annualization_bars_per_year must be positive")
        alpha = _finite_scalar("alpha", self.alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        method = str(self.bootstrap_method).strip().lower()
        if method not in {"moving_block", "stationary"}:
            raise ValueError("bootstrap_method must be 'moving_block' or 'stationary'")
        replicates = _strict_int("bootstrap_replicates", self.bootstrap_replicates, minimum=100)
        block_length = _strict_int("block_length", self.block_length, minimum=1)
        sensitivity = tuple(
            _strict_int("block_length_sensitivity item", value, minimum=1)
            for value in self.block_length_sensitivity
        )
        if not sensitivity:
            raise ValueError("block_length_sensitivity must not be empty")
        seed = _strict_int("seed", self.seed, minimum=0)
        min_folds = _strict_int("min_folds", self.min_folds, minimum=1)
        min_observations = _strict_int("min_observations", self.min_observations, minimum=2)
        n_trials = None
        if self.n_trials is not None:
            n_trials = _strict_int("n_trials", self.n_trials, minimum=1)
        cscv_subperiods = None
        if self.cscv_subperiods is not None:
            cscv_subperiods = _strict_int("cscv_subperiods", self.cscv_subperiods, minimum=1)
        pbo_max = _finite_scalar("pbo_max", self.pbo_max)
        if not 0.0 <= pbo_max <= 1.0:
            raise ValueError("pbo_max must be between 0 and 1")
        pass_rate = _finite_scalar("stress_min_pass_rate", self.stress_min_pass_rate)
        if not 0.0 <= pass_rate <= 1.0:
            raise ValueError("stress_min_pass_rate must be between 0 and 1")
        object.__setattr__(self, "annualization_bars_per_year", annualization)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "bootstrap_method", method)
        object.__setattr__(self, "bootstrap_replicates", replicates)
        object.__setattr__(self, "block_length", block_length)
        object.__setattr__(self, "block_length_sensitivity", sensitivity)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "min_folds", min_folds)
        object.__setattr__(self, "min_observations", min_observations)
        object.__setattr__(self, "n_trials", n_trials)
        object.__setattr__(self, "cscv_subperiods", cscv_subperiods)
        object.__setattr__(self, "pbo_max", pbo_max)
        object.__setattr__(self, "stress_min_pass_rate", pass_rate)
        object.__setattr__(
            self,
            "stress_alpha_floor_pt",
            _finite_scalar("stress_alpha_floor_pt", self.stress_alpha_floor_pt),
        )
        object.__setattr__(
            self,
            "stress_timing_floor_pt",
            _finite_scalar("stress_timing_floor_pt", self.stress_timing_floor_pt),
        )


def _coerce_candidates(values: Iterable[CandidatePath | Mapping[str, Any]]) -> list[CandidatePath]:
    candidates = [
        value if isinstance(value, CandidatePath) else CandidatePath.from_mapping(value)
        for value in values
    ]
    if not candidates:
        raise ValueError("at least one development candidate is required")
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    return candidates


def _coerce_stress_cases(
    values: Iterable[StressCase | Mapping[str, Any]] | None,
) -> list[StressCase]:
    if values is None:
        return []
    cases = [
        value if isinstance(value, StressCase) else StressCase.from_mapping(value)
        for value in values
    ]
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("stress case names must be unique")
    return cases


def _sorted_folds(candidate: CandidatePath) -> tuple[DevelopmentFold, ...]:
    return tuple(sorted(candidate.folds, key=lambda item: item.fold))


def _fold_totals(folds: Sequence[DevelopmentFold]) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(
        [100.0 * float(np.sum(item.alpha_excess_returns)) for item in folds], dtype=np.float64
    )
    timing = np.asarray(
        [100.0 * float(np.sum(item.timing_increment_returns)) for item in folds], dtype=np.float64
    )
    return alpha, timing


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    length = min(max(int(block_length), 1), n)
    starts = rng.integers(0, n - length + 1, size=ceil(n / length))
    chunks = [np.arange(int(start), int(start) + length, dtype=np.int64) for start in starts]
    return np.concatenate(chunks)[:n]


def _stationary_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    length = min(max(int(block_length), 1), n)
    restart_probability = 1.0 / float(length)
    indices = np.empty(n, dtype=np.int64)
    indices[0] = int(rng.integers(0, n))
    for index in range(1, n):
        if float(rng.random()) < restart_probability:
            indices[index] = int(rng.integers(0, n))
        else:
            indices[index] = (indices[index - 1] + 1) % n
    return indices


def _bootstrap_summary(
    folds: Sequence[DevelopmentFold],
    config: StatisticalGateConfig,
    *,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    alpha_observed, timing_observed = _fold_totals(folds)
    alpha_samples = np.empty(config.bootstrap_replicates, dtype=np.float64)
    timing_samples = np.empty(config.bootstrap_replicates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(config.bootstrap_replicates):
        alpha_total = 0.0
        timing_total = 0.0
        for item in folds:
            if config.bootstrap_method == "stationary":
                indices = _stationary_indices(len(item.alpha_excess_returns), block_length, rng)
            else:
                indices = _moving_block_indices(len(item.alpha_excess_returns), block_length, rng)
            # The same sampled indices preserve the alpha/timing pairing.
            alpha_total += float(np.sum(item.alpha_excess_returns[indices]))
            timing_total += float(np.sum(item.timing_increment_returns[indices]))
        alpha_samples[replicate] = 100.0 * alpha_total
        timing_samples[replicate] = 100.0 * timing_total

    def interval(estimate: float, samples: np.ndarray) -> dict[str, Any]:
        return {
            "estimate_pt": float(estimate),
            "lower_pt": float(np.quantile(samples, config.alpha / 2.0)),
            "upper_pt": float(np.quantile(samples, 1.0 - config.alpha / 2.0)),
            "replicates": int(len(samples)),
        }

    return {
        "method": config.bootstrap_method,
        "block_length": int(block_length),
        "seed": int(seed),
        "alpha_excess_pt": interval(float(alpha_observed.sum()), alpha_samples),
        "timing_increment_pt": interval(float(timing_observed.sum()), timing_samples),
        "definition": "100 * sum(per-bar additive net-return differential)",
    }


def bootstrap_confidence_intervals(
    folds: Sequence[DevelopmentFold],
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    """Return deterministic block-bootstrap CIs and block-length sensitivity."""
    cfg = config or StatisticalGateConfig()
    normalized = tuple(
        item if isinstance(item, DevelopmentFold) else DevelopmentFold.from_mapping(item)
        for item in folds
    )
    if not normalized:
        raise ValueError("at least one development fold is required for bootstrap")
    primary = _bootstrap_summary(
        normalized,
        cfg,
        block_length=cfg.block_length,
        seed=cfg.seed,
    )
    sensitivity = [
        _bootstrap_summary(
            normalized,
            cfg,
            block_length=block,
            seed=cfg.seed + index + 1,
        )
        for index, block in enumerate(cfg.block_length_sensitivity)
    ]
    return {
        "status": "ok",
        "primary": primary,
        "sensitivity": sensitivity,
        "block_length_sensitivity": [int(value) for value in cfg.block_length_sensitivity],
    }


def fold_sign_test(
    values: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    min_folds: int = 4,
    label: str = "fold metric",
) -> dict[str, Any]:
    """Run an exact one-sided sign/binomial test over non-zero fold values."""
    metric = _finite_vector(label, values)
    minimum = _strict_int("min_folds", min_folds, minimum=1)
    significance = _finite_scalar("alpha", alpha)
    nonzero = metric[np.abs(metric) > 1e-15]
    positive = int(np.sum(nonzero > 0.0))
    negative = int(np.sum(nonzero < 0.0))
    zero = int(len(metric) - len(nonzero))
    n = int(len(nonzero))
    if n < minimum:
        return {
            "status": "N/A",
            "passed": False,
            "reason": f"{label} has {n} non-zero folds; need at least {minimum}",
            "folds": int(len(metric)),
            "effective_folds": n,
            "positive_folds": positive,
            "negative_folds": negative,
            "zero_folds": zero,
            "p_value": None,
        }
    p_value = float(stats.binomtest(positive, n, p=0.5, alternative="greater").pvalue)
    return {
        "status": "ok",
        "passed": bool(positive > negative and p_value <= significance),
        "folds": int(len(metric)),
        "effective_folds": n,
        "positive_folds": positive,
        "negative_folds": negative,
        "zero_folds": zero,
        "median_pt": float(np.median(metric)),
        "p_value": p_value,
        "alpha": significance,
        "null": "P(positive)=0.5; zero folds omitted",
    }


def _candidate_strategy_returns(candidate: CandidatePath) -> np.ndarray | None:
    paths = [item.strategy_returns for item in _sorted_folds(candidate)]
    if any(path is None for path in paths):
        return None
    return np.concatenate([path for path in paths if path is not None]).astype(np.float64, copy=False)


def _sample_sharpe(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    deviation = float(np.std(values, ddof=1))
    if deviation <= 1e-12:
        return 0.0
    return float(np.mean(values) / deviation)


def _return_moments(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 4 or float(np.std(values, ddof=1)) <= 1e-12:
        return 0.0, 3.0
    skew = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 3.0
    return skew, kurtosis


def compute_deflated_sharpe(
    candidates: Sequence[CandidatePath],
    selected_candidate: str,
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    """Compute Bailey/López de Prado's non-normal, multiple-trial DSR.

    Sharpe ratios in the formula are per-bar (not annualized).  The reported
    annualized values are derived only for display using the fixed
    ``annualization_bars_per_year`` contract.
    """
    cfg = config or StatisticalGateConfig()
    candidate_list = _coerce_candidates(candidates)
    by_name = {candidate.name: candidate for candidate in candidate_list}
    if selected_candidate not in by_name:
        raise ValueError(f"selected candidate {selected_candidate!r} is unknown")
    paths = {candidate.name: _candidate_strategy_returns(candidate) for candidate in candidate_list}
    missing = sorted(name for name, path in paths.items() if path is None)
    trial_count = int(cfg.n_trials if cfg.n_trials is not None else len(candidate_list))
    if trial_count < len(candidate_list):
        raise ValueError("n_trials cannot be smaller than the number of supplied candidates")
    if missing:
        return {
            "status": "N/A",
            "passed": False,
            "reason": f"strategy_returns missing for candidate(s): {missing}",
            "n_trials": trial_count,
            "trial_count_source": "config" if cfg.n_trials is not None else "candidate_count",
        }
    selected_path = paths[selected_candidate]
    assert selected_path is not None
    if len(selected_path) < cfg.min_observations:
        return {
            "status": "N/A",
            "passed": False,
            "reason": (
                f"selected candidate has {len(selected_path)} observations; "
                f"need at least {cfg.min_observations}"
            ),
            "selected_candidate": selected_candidate,
            "n_trials": trial_count,
        }
    trial_sharpes = {
        name: _sample_sharpe(path) for name, path in paths.items() if path is not None
    }
    observed = float(trial_sharpes[selected_candidate])
    variance = (
        float(np.var(np.asarray(list(trial_sharpes.values()), dtype=np.float64), ddof=1))
        if len(trial_sharpes) > 1
        else 0.0
    )
    if trial_count <= 1 or variance <= 0.0:
        expected_max = 0.0
    else:
        z_n = float(stats.norm.ppf(1.0 - 1.0 / trial_count))
        z_ne = float(stats.norm.ppf(1.0 - 1.0 / (trial_count * exp(1.0))))
        expected_max = sqrt(variance) * (
            (1.0 - EULER_MASCHERONI) * z_n + EULER_MASCHERONI * z_ne
        )
    skew, kurtosis = _return_moments(selected_path)
    denominator = sqrt(
        max(1e-12, 1.0 - skew * observed + ((kurtosis - 1.0) / 4.0) * observed**2)
    )
    dsr_z = (observed - expected_max) * sqrt(max(len(selected_path) - 1, 1)) / denominator
    probability = float(stats.norm.cdf(dsr_z))
    threshold = 1.0 - cfg.alpha
    return {
        "status": "ok",
        "passed": bool(observed > 0.0 and probability >= threshold),
        "selected_candidate": selected_candidate,
        "n_trials": trial_count,
        "trial_count_source": "config" if cfg.n_trials is not None else "candidate_count",
        "n_observations": int(len(selected_path)),
        "annualization_bars_per_year": float(cfg.annualization_bars_per_year),
        "observed_sharpe_per_bar": observed,
        "observed_sharpe_annualized": observed * sqrt(cfg.annualization_bars_per_year),
        "expected_max_sharpe_per_bar": float(expected_max),
        "expected_max_sharpe_annualized": float(expected_max * sqrt(cfg.annualization_bars_per_year)),
        "trial_sharpes_per_bar": trial_sharpes,
        "variance_across_trial_sharpes": variance,
        "skewness": skew,
        "kurtosis_pearson": kurtosis,
        "deflated_sharpe_z": float(dsr_z),
        "dsr_probability": probability,
        "dsr_p_value": float(1.0 - probability),
        "probability_threshold": threshold,
        "formula": (
            "Phi((SR_hat-SR_star)*sqrt(T-1)/sqrt(1-g3*SR_hat+"
            "((g4-1)/4)*SR_hat^2)); SR_star uses Euler-weighted expected max over N trials"
        ),
    }


def deflated_sharpe(
    candidates: Sequence[CandidatePath],
    selected_candidate: str,
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    """Compatibility alias for :func:`compute_deflated_sharpe`."""
    return compute_deflated_sharpe(candidates, selected_candidate, config)


def compute_cscv_pbo(
    candidates: Sequence[CandidatePath],
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    """Compute CSCV PBO from candidate-by-development-fold performance.

    Each development fold is one subperiod and candidate performance is the
    sum of its supplied net per-bar returns in that subperiod.  The IS winner
    is selected for every half-subperiod combination; PBO is the fraction of
    combinations whose selected winner is below the OOS median rank.
    """
    cfg = config or StatisticalGateConfig()
    candidate_list = _coerce_candidates(candidates)
    if len(candidate_list) < 2:
        return {
            "status": "N/A",
            "passed": False,
            "reason": "CSCV needs at least two candidate paths",
            "n_candidates": len(candidate_list),
        }
    fold_sets = [set(item.fold for item in candidate.folds) for candidate in candidate_list]
    if any(fold_set != fold_sets[0] for fold_set in fold_sets[1:]):
        raise ValueError("all CSCV candidates must contain the same development folds")
    fold_ids = sorted(fold_sets[0])
    subperiods = int(cfg.cscv_subperiods or len(fold_ids))
    if subperiods < 4:
        return {
            "status": "N/A",
            "passed": False,
            "reason": "CSCV needs at least four development subperiods",
            "n_candidates": len(candidate_list),
            "n_subperiods": subperiods,
        }
    if subperiods % 2:
        return {
            "status": "N/A",
            "passed": False,
            "reason": "CSCV requires an even number of development subperiods",
            "n_candidates": len(candidate_list),
            "n_subperiods": subperiods,
        }
    if subperiods != len(fold_ids):
        return {
            "status": "N/A",
            "passed": False,
            "reason": "configured CSCV subperiod count must equal supplied development folds",
            "n_candidates": len(candidate_list),
            "n_subperiods": subperiods,
            "available_subperiods": len(fold_ids),
        }
    performance = np.empty((len(candidate_list), subperiods), dtype=np.float64)
    for row, candidate in enumerate(candidate_list):
        by_fold = {item.fold: item for item in candidate.folds}
        if any(by_fold[fold].strategy_returns is None for fold in fold_ids):
            return {
                "status": "N/A",
                "passed": False,
                "reason": f"strategy_returns missing for candidate {candidate.name!r}",
                "n_candidates": len(candidate_list),
                "n_subperiods": subperiods,
            }
        for column, fold in enumerate(fold_ids):
            path = by_fold[fold].strategy_returns
            assert path is not None
            performance[row, column] = float(np.sum(path))

    logits: list[float] = []
    selected_indices: list[int] = []
    for is_indices in itertools.combinations(range(subperiods), subperiods // 2):
        is_set = set(is_indices)
        oos_indices = tuple(index for index in range(subperiods) if index not in is_set)
        is_scores = performance[:, is_indices].sum(axis=1)
        selected = int(np.argmax(is_scores))
        oos_scores = performance[:, oos_indices].sum(axis=1)
        selected_oos = float(oos_scores[selected])
        less = int(np.sum(oos_scores < selected_oos))
        ties = int(np.sum(np.isclose(oos_scores, selected_oos, rtol=1e-12, atol=1e-15)))
        rank = (less + 0.5 * ties) / float(len(candidate_list))
        rank = float(np.clip(rank, 1.0 / (2.0 * len(candidate_list)), 1.0 - 1.0 / (2.0 * len(candidate_list))))
        logits.append(float(log(rank / (1.0 - rank))))
        selected_indices.append(selected)
    logit_array = np.asarray(logits, dtype=np.float64)
    pbo = float(np.mean(logit_array < 0.0))
    return {
        "status": "ok",
        "passed": bool(pbo <= cfg.pbo_max),
        "pbo": pbo,
        "pbo_max": float(cfg.pbo_max),
        "n_candidates": len(candidate_list),
        "n_subperiods": subperiods,
        "n_combinations": int(len(logits)),
        "overfit_combinations": int(np.sum(logit_array < 0.0)),
        "mean_oos_rank_logit": float(logit_array.mean()),
        "selected_candidate_counts": {
            candidate.name: int(selected_indices.count(index))
            for index, candidate in enumerate(candidate_list)
        },
        "definition": "PBO = fraction of CSCV IS winners with OOS rank below the median",
    }


def evaluate_stress(
    stress_cases: Iterable[StressCase | Mapping[str, Any]] | None,
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate machine-readable cost and regime stress cases."""
    cfg = config or StatisticalGateConfig()
    cases = _coerce_stress_cases(stress_cases)
    output: dict[str, Any] = {
        "status": "ok",
        "passed": True,
        "required_kinds": [],
        "cases": [],
        "groups": {},
        "thresholds": {
            "alpha_excess_floor_pt": float(cfg.stress_alpha_floor_pt),
            "timing_increment_floor_pt": float(cfg.stress_timing_floor_pt),
            "minimum_pass_rate": float(cfg.stress_min_pass_rate),
        },
    }
    required = []
    if cfg.require_cost_stress:
        required.append("cost")
    if cfg.require_regime_stress:
        required.append("regime")
    output["required_kinds"] = required
    by_kind = {kind: [case for case in cases if case.kind == kind] for kind in {"cost", "regime"}}
    for case in cases:
        passed = bool(
            case.alpha_excess_pt >= cfg.stress_alpha_floor_pt
            and case.timing_increment_pt >= cfg.stress_timing_floor_pt
        )
        output["cases"].append(
            {
                "name": case.name,
                "kind": case.kind,
                "alpha_excess_pt": case.alpha_excess_pt,
                "timing_increment_pt": case.timing_increment_pt,
                "passed": passed,
            }
        )
    for kind in required:
        group = by_kind[kind]
        if not group:
            output["status"] = "N/A"
            output["passed"] = False
            output["groups"][kind] = {
                "status": "N/A",
                "passed": False,
                "reason": f"required {kind} stress input is missing",
            }
            continue
        pass_flags = [
            case.alpha_excess_pt >= cfg.stress_alpha_floor_pt
            and case.timing_increment_pt >= cfg.stress_timing_floor_pt
            for case in group
        ]
        pass_rate = float(np.mean(pass_flags))
        group_passed = bool(pass_rate >= cfg.stress_min_pass_rate)
        output["groups"][kind] = {
            "status": "ok",
            "passed": group_passed,
            "cases": len(group),
            "passed_cases": int(np.sum(pass_flags)),
            "pass_rate": pass_rate,
        }
        if not group_passed:
            output["passed"] = False
    return output


def _select_candidate(candidate_list: Sequence[CandidatePath]) -> tuple[str, str]:
    strategy_paths = {candidate.name: _candidate_strategy_returns(candidate) for candidate in candidate_list}
    if all(path is not None for path in strategy_paths.values()):
        scores = {
            name: _sample_sharpe(path) for name, path in strategy_paths.items() if path is not None
        }
        return max(scores, key=lambda name: (scores[name], name)), "highest_development_sharpe"
    totals = {
        candidate.name: float(np.sum(_fold_totals(candidate.folds)[0]))
        for candidate in candidate_list
    }
    return max(totals, key=lambda name: (totals[name], name)), "highest_development_alpha_excess"


def evaluate_statistical_gate(
    candidates: Iterable[CandidatePath | Mapping[str, Any]],
    *,
    selected_candidate: str | None = None,
    stress_cases: Iterable[StressCase | Mapping[str, Any]] | None = None,
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate every statistical component without reading test/holdout data."""
    cfg = config or StatisticalGateConfig()
    candidate_list = _coerce_candidates(candidates)
    if selected_candidate is None:
        selected_name, selection_basis = _select_candidate(candidate_list)
    else:
        selected_name = str(selected_candidate)
        if selected_name not in {candidate.name for candidate in candidate_list}:
            raise ValueError(f"selected candidate {selected_name!r} is unknown")
        selection_basis = "caller_selected_development_candidate"
    selected = next(candidate for candidate in candidate_list if candidate.name == selected_name)
    if cfg.n_trials is not None and cfg.n_trials < len(candidate_list):
        raise ValueError("n_trials cannot be smaller than candidate count")

    bootstrap = bootstrap_confidence_intervals(_sorted_folds(selected), cfg)
    alpha_folds, timing_folds = _fold_totals(_sorted_folds(selected))
    alpha_sign = fold_sign_test(
        alpha_folds,
        alpha=cfg.alpha,
        min_folds=cfg.min_folds,
        label="alpha_excess fold totals",
    )
    timing_sign = fold_sign_test(
        timing_folds,
        alpha=cfg.alpha,
        min_folds=cfg.min_folds,
        label="timing_increment fold totals",
    )
    dsr = compute_deflated_sharpe(candidate_list, selected_name, cfg)
    pbo = compute_cscv_pbo(candidate_list, cfg)
    stress = evaluate_stress(stress_cases, cfg)
    alpha_ci = bootstrap["primary"]["alpha_excess_pt"]
    timing_ci = bootstrap["primary"]["timing_increment_pt"]
    sensitivity_ci_pass = all(
        sensitivity["alpha_excess_pt"]["lower_pt"] > 0.0
        and sensitivity["timing_increment_pt"]["lower_pt"] > 0.0
        for sensitivity in bootstrap["sensitivity"]
    )
    components = {
        "alpha_bootstrap_ci_excludes_zero": bool(alpha_ci["lower_pt"] > 0.0),
        "timing_bootstrap_ci_excludes_zero": bool(timing_ci["lower_pt"] > 0.0),
        "bootstrap_block_sensitivity_excludes_zero": sensitivity_ci_pass,
        "alpha_fold_sign_test": bool(alpha_sign.get("passed", False)),
        "timing_fold_sign_test": bool(timing_sign.get("passed", False)),
        "deflated_sharpe": bool(dsr.get("passed", False)),
        "cscv_pbo": bool(pbo.get("passed", False)),
        "cost_regime_stress": bool(stress.get("passed", False)),
    }
    failed = [name for name, passed in components.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "gate": {
            "passed": not failed,
            "status": "PASS" if not failed else "REJECT",
            "components": components,
            "failed_components": failed,
        },
        "scope": {
            "development_only": True,
            "max_development_fold": MAX_DEVELOPMENT_FOLD,
            "holdout_folds_rejected": True,
            "candidate_count": len(candidate_list),
            "candidate_names": [candidate.name for candidate in candidate_list],
            "selected_candidate": selected_name,
            "selection_basis": selection_basis,
            "selected_folds": [item.fold for item in _sorted_folds(selected)],
        },
        "contract": {
            "annualization_bars_per_year": float(cfg.annualization_bars_per_year),
            "alpha": float(cfg.alpha),
            "n_trials": int(cfg.n_trials if cfg.n_trials is not None else len(candidate_list)),
            "bootstrap_method": cfg.bootstrap_method,
            "block_length": int(cfg.block_length),
            "block_length_sensitivity": [int(value) for value in cfg.block_length_sensitivity],
            "seed": int(cfg.seed),
            "metric_definition": "100 * sum(per-bar additive net-return differential)",
        },
        "bootstrap": bootstrap,
        "fold_sign": {"alpha_excess": alpha_sign, "timing_increment": timing_sign},
        "deflated_sharpe": dsr,
        "cscv_pbo": pbo,
        "stress": stress,
    }


def _load_input(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read statistical gate input {input_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("statistical gate input must be a JSON object")
    return dict(payload)


def evaluate_json_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the documented JSON input shape used by the CLI."""
    if "candidates" not in payload:
        raise ValueError("statistical gate input requires candidates")
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, Mapping):
        raise ValueError("config must be an object")
    config = StatisticalGateConfig(**dict(config_payload))
    return evaluate_statistical_gate(
        payload["candidates"],
        selected_candidate=payload.get("selected_candidate"),
        stress_cases=payload.get("stress_cases", payload.get("stress", [])),
        config=config,
    )


__all__ = [
    "CandidatePath",
    "DevelopmentFold",
    "MAX_DEVELOPMENT_FOLD",
    "SCHEMA_VERSION",
    "StatisticalGateConfig",
    "StressCase",
    "bootstrap_confidence_intervals",
    "compute_cscv_pbo",
    "compute_deflated_sharpe",
    "deflated_sharpe",
    "evaluate_json_input",
    "evaluate_statistical_gate",
    "evaluate_stress",
    "fold_sign_test",
]
