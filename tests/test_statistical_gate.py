import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unidream.cli.statistical_gate import main
from unidream.eval.statistical_gate import (
    CandidatePath,
    DevelopmentFold,
    StatisticalGateConfig,
    StressCase,
    bootstrap_confidence_intervals,
    compute_cscv_pbo,
    compute_deflated_sharpe,
    evaluate_json_input,
    evaluate_statistical_gate,
    evaluate_stress,
)


def _config(**overrides: object) -> StatisticalGateConfig:
    values: dict[str, object] = {
        "bootstrap_replicates": 120,
        "block_length": 8,
        "block_length_sensitivity": (4, 8),
        "min_observations": 32,
        "seed": 17,
        "n_trials": 2,
    }
    values.update(overrides)
    return StatisticalGateConfig(**values)


def _strong_candidates() -> tuple[list[CandidatePath], list[StressCase]]:
    rng = np.random.default_rng(20260830)
    strong: list[DevelopmentFold] = []
    weak: list[DevelopmentFold] = []
    for fold in range(6):
        noise = rng.normal(0.0, 0.0002, size=64)
        strong.append(
            DevelopmentFold(
                fold=fold,
                alpha_excess_returns=0.002 + noise,
                timing_increment_returns=0.001 + noise * 0.5,
                strategy_returns=0.001 + noise,
            )
        )
        weak_noise = rng.normal(0.0, 0.0003, size=64)
        weak.append(
            DevelopmentFold(
                fold=fold,
                alpha_excess_returns=weak_noise * 0.05,
                timing_increment_returns=weak_noise * 0.02,
                strategy_returns=weak_noise,
            )
        )
    stress = [
        StressCase("high_fee", "cost", 1.0, 0.5),
        StressCase("wide_spread", "cost", 0.5, 0.2),
        StressCase("bull", "regime", 1.0, 0.3),
        StressCase("bear", "regime", 0.2, 0.1),
    ]
    return [CandidatePath("strong", tuple(strong)), CandidatePath("weak", tuple(weak))], stress


class StatisticalGateContractTest(unittest.TestCase):
    def test_strong_signal_passes_all_development_gates(self) -> None:
        candidates, stress = _strong_candidates()
        result = evaluate_statistical_gate(
            candidates,
            selected_candidate="strong",
            stress_cases=stress,
            config=_config(),
        )
        self.assertTrue(result["gate"]["passed"], result["gate"])
        self.assertEqual(result["gate"]["failed_components"], [])
        self.assertEqual(result["scope"]["selected_folds"], list(range(6)))
        self.assertEqual(result["contract"]["n_trials"], 2)
        self.assertEqual(result["bootstrap"]["primary"]["method"], "moving_block")
        self.assertEqual(len(result["bootstrap"]["sensitivity"]), 2)

    def test_null_signal_cannot_pass(self) -> None:
        zero_folds = tuple(
            DevelopmentFold(
                fold=fold,
                alpha_excess_returns=np.zeros(64),
                timing_increment_returns=np.zeros(64),
                strategy_returns=np.zeros(64),
            )
            for fold in range(6)
        )
        candidates = [CandidatePath("a", zero_folds), CandidatePath("b", zero_folds)]
        result = evaluate_statistical_gate(
            candidates,
            selected_candidate="a",
            stress_cases=[
                StressCase("cost", "cost", 0.0, 0.0),
                StressCase("regime", "regime", 0.0, 0.0),
            ],
            config=_config(),
        )
        self.assertFalse(result["gate"]["passed"])
        self.assertIn("alpha_bootstrap_ci_excludes_zero", result["gate"]["failed_components"])
        self.assertFalse(result["deflated_sharpe"]["passed"])

    def test_holdout_fold_is_rejected_before_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "fold 15"):
            DevelopmentFold(
                fold=15,
                alpha_excess_returns=[0.1, 0.1],
                timing_increment_returns=[0.1, 0.1],
                strategy_returns=[0.1, 0.1],
            )

    def test_cscv_is_na_when_candidate_or_subperiod_count_is_insufficient(self) -> None:
        candidates, _ = _strong_candidates()
        one_candidate = compute_cscv_pbo([candidates[0]], _config())
        self.assertEqual(one_candidate["status"], "N/A")
        self.assertIn("two candidate", one_candidate["reason"])

        odd_subperiods = compute_cscv_pbo(
            [
                CandidatePath("a", tuple(candidates[0].folds[:5])),
                CandidatePath("b", tuple(candidates[1].folds[:5])),
            ],
            _config(),
        )
        self.assertEqual(odd_subperiods["status"], "N/A")
        self.assertIn("even", odd_subperiods["reason"])

    def test_block_bootstrap_is_deterministic_for_stationary_and_moving_methods(self) -> None:
        candidates, _ = _strong_candidates()
        for method in ("moving_block", "stationary"):
            config = _config(bootstrap_method=method)
            first = bootstrap_confidence_intervals(candidates[0].folds, config)
            second = bootstrap_confidence_intervals(candidates[0].folds, config)
            self.assertEqual(first, second)
            self.assertGreater(first["primary"]["alpha_excess_pt"]["lower_pt"], 0.0)

    def test_dsr_records_trial_count_and_multiple_testing_penalty(self) -> None:
        candidates, _ = _strong_candidates()
        base = compute_deflated_sharpe(candidates, "strong", _config())
        many_trials = compute_deflated_sharpe(
            candidates,
            "strong",
            _config(n_trials=100),
        )
        self.assertEqual(base["n_trials"], 2)
        self.assertEqual(many_trials["n_trials"], 100)
        self.assertGreaterEqual(
            many_trials["expected_max_sharpe_per_bar"],
            base["expected_max_sharpe_per_bar"],
        )
        self.assertEqual(base["annualization_bars_per_year"], 365 * 96)

    def test_omitted_trial_count_is_diagnostic_only_and_cannot_promote(self) -> None:
        candidates, stress = _strong_candidates()
        result = evaluate_statistical_gate(
            candidates,
            selected_candidate="strong",
            stress_cases=stress,
            config=_config(n_trials=None),
        )
        self.assertFalse(result["gate"]["passed"])
        self.assertFalse(result["gate"]["components"]["explicit_n_trials"])
        self.assertIn("explicit_n_trials", result["gate"]["failed_components"])
        self.assertFalse(result["deflated_sharpe"]["promotion_eligible"])
        self.assertIn("full number of tried candidates", result["deflated_sharpe"]["trial_count_warning"])

    def test_stress_gate_requires_cost_and_regime_inputs(self) -> None:
        result = evaluate_stress([StressCase("cost", "cost", 1.0, 1.0)], _config())
        self.assertEqual(result["status"], "N/A")
        self.assertFalse(result["passed"])
        self.assertIn("regime", result["groups"])

    def test_json_cli_emits_machine_readable_rejection(self) -> None:
        candidates, stress = _strong_candidates()
        payload = {
            "selected_candidate": "strong",
            "config": {
                "bootstrap_replicates": 100,
                "block_length": 8,
                "block_length_sensitivity": [8],
                "min_observations": 32,
                "seed": 17,
                "n_trials": 2,
            },
            "candidates": [
                {
                    "name": candidate.name,
                    "folds": [
                        {
                            "fold": item.fold,
                            "alpha_excess_returns": item.alpha_excess_returns.tolist(),
                            "timing_increment_returns": item.timing_increment_returns.tolist(),
                            "strategy_returns": item.strategy_returns.tolist(),
                        }
                        for item in candidate.folds
                    ],
                }
                for candidate in candidates
            ],
            "stress_cases": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "alpha_excess_pt": item.alpha_excess_pt,
                    "timing_increment_pt": item.timing_increment_pt,
                }
                for item in stress
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "result.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )
            rendered = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertTrue(rendered["gate"]["passed"])
        self.assertEqual(json.loads(stdout.getvalue())["schema_version"], 1)
        self.assertTrue(evaluate_json_input(payload)["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
