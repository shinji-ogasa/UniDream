import copy
import json
import unittest

import numpy as np
import pandas as pd
import torch

from unidream.actor_critic.actor import Actor
from unidream.experiments.wm_rl_policy import (
    IncrementalActorPolicy, PolicyState, encode_fixed_context,
)


class FakeEnsemble(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout(0.5)
        self.calls = []

    def get_z_dim(self):
        return 2

    def get_d_model(self):
        return 2

    def encode(self, observations):
        if self.training or observations.shape[1] != 64:
            raise AssertionError("fixed64 eval required")
        self.calls.append(observations.clone())
        return observations[..., :2], torch.zeros(())

    def forward(self, z, actions):
        if actions.shape != (len(z), 64, 1) or not torch.equal(actions, torch.ones_like(actions)):
            raise AssertionError("benchmark-one action context required")
        # Independent samples, but each hidden state depends on its full prefix.
        return {"h": torch.cumsum(z, dim=1)}


class FakeActor(Actor):
    def __init__(self, regime_dim=0, advantage_dim=0):
        super().__init__(z_dim=2, h_dim=2, hidden_dim=8, n_layers=1,
                         inventory_dim=4, regime_dim=regime_dim,
                         advantage_dim=advantage_dim)
        self.benchmark_position = 1.0
        self.abs_min_position, self.abs_max_position = 0.5, 1.12
        self.infer_quantize_step = 0.001
        self.rate_cap_active_eps = 0.005
        self.short_underweight_rate_max = 0.6
        self.active_rate_max = 0.8
        self.benchmark_overweight_long_rate_max = 0.25
        self.seen = []

    def act_greedy(self, z, h, inventory=None, regime=None, advantage=None, **kwargs):
        if self.training:
            raise AssertionError("actor eval required")
        self.seen.append(inventory.detach().cpu().numpy().copy())
        value = (1 + inventory[:, 0] + 0.035 * torch.tanh(z[:, 0])
                 + 0.01 * inventory[:, 2] - 0.005 * inventory[:, 3])
        return value.clamp(self.abs_min_position, self.abs_max_position).unsqueeze(-1)


def data(n=220):
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC")
    x = np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 100
    return index, x, np.ones(n, dtype=bool)


class FixedContextTests(unittest.TestCase):
    def test_prefix_invariance_and_exact_trailing_context(self):
        index, x, mask = data()
        model = FakeEnsemble()
        model.child.eval()  # preserve heterogeneous submodule mode too
        full = encode_fixed_context(model, x, index, mask, batch_size=13)
        prefix = encode_fixed_context(model, x[:151], index[:151], mask[:151], batch_size=1)
        np.testing.assert_array_equal(full["z"][:151], prefix["z"])
        np.testing.assert_array_equal(full["h"][:151], prefix["h"])
        np.testing.assert_allclose(full["h"][150], x[87:151, :2].astype(np.float64).sum(axis=0), rtol=0, atol=2e-5)
        self.assertTrue(np.isnan(full["z"][:63]).all())
        self.assertTrue(full["available"][63:].all())
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def test_offline_equals_one_context_at_a_time(self):
        index, x, mask = data(100)
        full = encode_fixed_context(FakeEnsemble(), x, index, mask, batch_size=17)
        for end in range(63, 100):
            one = encode_fixed_context(FakeEnsemble(), x[end-63:end+1], index[end-63:end+1], mask[end-63:end+1])
            np.testing.assert_array_equal(full["z"][end], one["z"][-1])
            np.testing.assert_array_equal(full["h"][end], one["h"][-1])

    def test_gaps_nonfinite_and_full_dependency_mask_are_excluded(self):
        index, x, mask = data()
        index, x, mask = index.delete(80), np.delete(x, 80, 0), np.delete(mask, 80)
        mask[70] = False
        x[100, 0] = np.nan
        x[110, 1] = np.inf
        result = encode_fixed_context(FakeEnsemble(), x, index, mask)
        expected = np.zeros(len(index), dtype=bool)
        for end in range(63, len(index)):
            start = end-63
            expected[end] = (mask[start:end+1].all()
                             and np.isfinite(x[start:end+1]).all()
                             and (np.diff(index[start:end+1].asi8) == 900_000_000_000).all())
        np.testing.assert_array_equal(result["available"], expected)
        self.assertTrue(np.isnan(result["z"][~expected]).all())
        self.assertTrue(np.isnan(result["h"][~expected]).all())
        self.assertTrue(expected[-1])

    def test_future_bad_features_do_not_affect_earlier_output(self):
        index, x, mask = data(150)
        a = encode_fixed_context(FakeEnsemble(), x, index, mask)
        x[100:] = np.nan
        b = encode_fixed_context(FakeEnsemble(), x, index, mask)
        np.testing.assert_array_equal(a["h"][:100], b["h"][:100])

    def test_short_empty_or_overflow_context_never_calls_encoder(self):
        for n in (0, 10, 64):
            index, x, mask = data(n)
            if n == 64:
                x = x.astype(np.float64)
                x[0, 0] = 1e300
            model = FakeEnsemble()
            output = encode_fixed_context(model, x, index, mask)
            self.assertFalse(output["available"].any())
            self.assertEqual(model.calls, [])

    def test_malformed_contract_rejected_and_mode_restored_on_failure(self):
        index, x, mask = data(64)
        for kwargs in ({"context_length": 63}, {"batch_size": 0}):
            with self.assertRaises(ValueError):
                encode_fixed_context(FakeEnsemble(), x, index, mask, **kwargs)
        with self.assertRaises(ValueError):
            encode_fixed_context(FakeEnsemble(), x, index, mask.astype(int))
        with self.assertRaises(ValueError):
            encode_fixed_context(FakeEnsemble(), x.astype(complex), index, mask)
        with self.assertRaises(ValueError):
            encode_fixed_context(FakeEnsemble(), x, index + pd.Timedelta(minutes=1), mask)
        with self.assertRaises(ValueError):
            encode_fixed_context(FakeEnsemble(), x, index.tz_localize(None), mask)
        model = FakeEnsemble()
        model.forward = lambda *args: {"h": torch.full((1, 64, 2), torch.nan)}
        with self.assertRaises(ValueError):
            encode_fixed_context(model, x, index, mask)
        self.assertTrue(model.training)


class IncrementalActorTests(unittest.TestCase):
    def test_native_no_account_parity_with_counters_and_quantization(self):
        actor = FakeActor()
        index, _, _ = data(160)
        z = np.column_stack([np.sin(np.arange(160)/11)-0.3, np.zeros(160)]).astype(np.float32)
        h = np.zeros_like(z)
        expected = actor.predict_positions(z, h)
        policy = IncrementalActorPolicy(actor, physical_feedback=False)
        actual = [policy.step(t, z[i], h[i], available=True)["target_intent"] for i, t in enumerate(index)]
        np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)
        self.assertEqual(policy.state.step_count, 160)
        self.assertGreater(policy.state.underweight_count, 0)
        self.assertTrue(actor.training)

    def test_serialized_restart_preserves_state_and_rate_caps(self):
        index, _, _ = data(40)
        actor_a, actor_b = FakeActor(), FakeActor()
        a = IncrementalActorPolicy(actor_a, physical_feedback=False)
        b = IncrementalActorPolicy(actor_b, physical_feedback=False)
        for i, t in enumerate(index):
            z = np.asarray([-1.0, 0.0], dtype=np.float32)
            aa = a.step(t, z, z, available=True)
            bb = b.step(t, z, z, available=True)
            self.assertEqual(aa, bb)
            b = IncrementalActorPolicy(actor_b, state=json.loads(json.dumps(b.state.to_dict())), physical_feedback=False)
        self.assertEqual(a.state.to_dict(), b.state.to_dict())

    def test_physical_pending_target_is_not_fill_and_drift_is_not_trade(self):
        actor = FakeActor()
        policy = IncrementalActorPolicy(actor)
        index, _, _ = data(4)
        first = policy.step(index[0], np.array([-1., 0.]), np.zeros(2), available=True,
                            actual_exposure=1.0, executed_delta=0.0)
        self.assertLess(first["target_intent"], 1.0)
        self.assertEqual(policy.state.controller[0], 0.0)
        self.assertEqual(policy.state.controller[1], 0.0)
        policy.step(index[1], available=False, actual_exposure=0.95, executed_delta=0.0)
        self.assertAlmostEqual(policy.state.controller[0], -0.05)
        self.assertEqual(policy.state.controller[1], 0.0)
        self.assertAlmostEqual(policy.state.controller[2], 2/64)
        policy.step(index[2], np.array([-1., 0.]), np.zeros(2), available=True,
                    actual_exposure=0.85, executed_delta=-0.1)
        self.assertAlmostEqual(actor.seen[-1][0, 0], -0.15)
        self.assertAlmostEqual(actor.seen[-1][0, 1], -0.1)
        self.assertEqual(policy.state.controller[2], 0.0)
        self.assertAlmostEqual(policy.state.controller[0], -0.15)

    def test_exposure_above_intent_bound_is_not_clipped_in_actor_input(self):
        actor = FakeActor()
        policy = IncrementalActorPolicy(actor)
        out = policy.step(pd.Timestamp("2021-01-01T00:00Z"),
                          np.zeros(2), np.zeros(2), available=True,
                          actual_exposure=1.3, executed_delta=0.0)
        self.assertAlmostEqual(actor.seen[-1][0, 0], 0.3)
        self.assertAlmostEqual(policy.state.controller[0], 0.3)
        self.assertLessEqual(out["target_intent"], 1.12 + 2 * np.finfo(np.float32).eps)

    def test_unavailable_holds_advance_clock_and_physical_counters(self):
        actor = FakeActor()
        policy = IncrementalActorPolicy(actor)
        index, _, _ = data(3)
        for t in index:
            out = policy.step(t, z="ignored unavailable poison", available=False,
                              actual_exposure=0.8, executed_delta=0.0)
            self.assertIsNone(out["target_intent"])
            self.assertEqual(out["rate_counter_basis"], "held_exposure")
        self.assertEqual(actor.seen, [])
        self.assertEqual(policy.state.step_count, 3)
        self.assertEqual(policy.state.active_count, 3)
        self.assertEqual(policy.state.underweight_count, 3)
        self.assertAlmostEqual(policy.state.controller[2], 3/64)
        self.assertAlmostEqual(policy.state.controller[3], 3/64)

    def test_nonfinite_required_inputs_hold_without_native_zero_fill(self):
        actor = FakeActor(regime_dim=3, advantage_dim=2)
        policy = IncrementalActorPolicy(actor)
        index, _, _ = data(3)
        values = [dict(z=np.array([np.nan, 0.])), dict(advantage=np.array([np.inf, 0.])), dict(regime=None)]
        for t, replacement in zip(index, values):
            kwargs = dict(z=np.zeros(2), h=np.zeros(2), regime=np.array([1., 0., 0.]), advantage=np.zeros(2))
            kwargs.update(replacement)
            result = policy.step(t, available=True, actual_exposure=1., executed_delta=0., **kwargs)
            self.assertIsNone(result["target_intent"])
            self.assertEqual(result["reason"], "model_input_unavailable")
        self.assertEqual(actor.seen, [])
        self.assertEqual(policy.state.step_count, 3)

    def test_duplicate_skipped_clock_or_partial_feedback_rejected_atomically(self):
        policy = IncrementalActorPolicy(FakeActor())
        t = pd.Timestamp("2021-01-01T00:00Z")
        for kwargs in ({"actual_exposure": 1.}, {"executed_delta": 0.}):
            with self.assertRaises(ValueError):
                policy.step(t, available=False, **kwargs)
        policy.step(t, available=False, actual_exposure=1., executed_delta=0.)
        prior = policy.state
        for bad in (t, t + pd.Timedelta(minutes=30), int(t.value)):
            with self.assertRaises(ValueError):
                policy.step(bad, available=False, actual_exposure=1., executed_delta=0.)
            self.assertEqual(policy.state, prior)

    def test_state_schema_and_physical_mode_cannot_silently_change(self):
        state = PolicyState().to_dict()
        with self.assertRaises(ValueError):
            IncrementalActorPolicy(FakeActor(), state=state, physical_feedback=False)
        for key, value in (("controller", [0., 0., 0.]), ("step_count", True), ("active_count", 1)):
            bad = copy.deepcopy(state)
            bad[key] = value
            with self.assertRaises(ValueError):
                PolicyState.from_dict(bad)
        bad = copy.deepcopy(state)
        bad["extra"] = 1
        with self.assertRaises(ValueError):
            PolicyState.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
