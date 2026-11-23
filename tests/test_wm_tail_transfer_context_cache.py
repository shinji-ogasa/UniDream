import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from unidream.experiments import wm_tail_transfer_policy as policy


def fixture():
    rng = np.random.default_rng(7)
    n = 24
    cfg = {"policy": {"context": 4, "hidden_size": 8, "projection_dim": 4, "head_hidden": 16,
                       "source_input_group": "D03", "context_batch_size": 8},
           "accounting": {"position_min": 0., "position_max": 2., "max_position_step": .08}}
    arrays = {"index_ns": np.arange(n, dtype=np.int64)*900_000_000_000,
              "market": {"raw": rng.normal(size=(n, 2)).astype(np.float32),
                         "old_z": rng.normal(size=(n, 3)).astype(np.float32), "new_z": rng.normal(size=(n, 3)).astype(np.float32),
                         "forecasts": rng.normal(size=(n, 4)).astype(np.float32), "uncertainty": rng.normal(size=(n, 1)).astype(np.float32)},
              "risk_sigma": np.full(n, .01, np.float32), "up_probability": np.full(n, .2, np.float32),
              "down_probability": np.full(n, .3, np.float32), "uncertainty_score": np.full(n, .1, np.float32),
              "available": np.ones(n, bool), "producer_manifest": {"complete": True, "split": "test",
                  "registration_sha256": "registered", "artifact_sha256": {"forecasts.npy": "frozen_F", "new_z.npy": "frozen_Z"}}}
    model = policy.CachedTailPolicy(policy._dims(arrays, cfg), cfg, "R04").eval()
    return cfg, arrays, model, np.arange(4, 16)


class TailTransferContextCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)

    def test_bc_rl_and_accounting_sensitivities_share_identical_market_contexts(self):
        cfg, arrays, bc, ids = fixture()
        expected = policy.encode_contexts(bc, arrays, ids, cfg)
        rl = copy.deepcopy(bc)
        with torch.no_grad():
            for p in rl.policy_head.parameters(): p.add_(10.)
            for p in rl.critic_head.parameters(): p.mul_(2.)
            rl.log_std.add_(.5)
        sensitivity = copy.deepcopy(cfg)
        sensitivity["accounting"].update(mode="linear_perp", maintenance_margin_rate=.01, slippage_rate=.001)
        with tempfile.TemporaryDirectory() as tmp, patch.object(policy, "encode_contexts", wraps=policy.encode_contexts) as encode:
            first = policy.encode_test_contexts_cached(bc, arrays, ids, cfg, tmp)
            second = policy.encode_test_contexts_cached(rl, arrays, ids, sensitivity, tmp)
            self.assertEqual(encode.call_count, 1)
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertTrue(np.array_equal(first["contexts"], expected["contexts"]))
            self.assertTrue(np.array_equal(first["contexts"], second["contexts"]))
            self.assertEqual(first["cache_manifest_sha256"], second["cache_manifest_sha256"])

    def test_encoder_tensors_and_normalization_buffers_are_part_of_the_key(self):
        cfg, arrays, model, ids = fixture()
        changed = copy.deepcopy(model)
        with torch.no_grad(): changed.raw_encoder[0].weight.add_(.01)
        normalized = copy.deepcopy(model)
        normalized.raw_std.mul_(2.)
        with tempfile.TemporaryDirectory() as tmp:
            keys = {policy.encode_test_contexts_cached(m, arrays, ids, cfg, tmp)["cache_key"] for m in (model, changed, normalized)}
            self.assertEqual(len(keys), 3)

    def test_producer_artifacts_semantic_risks_availability_and_ids_are_bound(self):
        cfg, arrays, model, ids = fixture()
        changed_producer = copy.deepcopy(arrays)
        changed_producer["producer_manifest"]["artifact_sha256"]["new_z.npy"] = "new_frozen_Z"
        changed_risk = copy.deepcopy(arrays); changed_risk["up_probability"][7] = .8
        missing = copy.deepcopy(arrays); missing["available"][7] = False
        with tempfile.TemporaryDirectory() as tmp:
            original = policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)
            changed = [policy.encode_test_contexts_cached(model, value, ids, cfg, tmp) for value in (changed_producer, changed_risk, missing)]
            shortened = policy.encode_test_contexts_cached(model, arrays, ids[:-1], cfg, tmp)
            self.assertEqual(len({original["cache_key"], shortened["cache_key"], *(x["cache_key"] for x in changed)}), 5)
            self.assertNotIn(7, changed[-1]["raw_ids"])

    def test_reuse_checks_bytes_after_file_change_instead_of_trusting_a_stale_process_cache(self):
        cfg, arrays, model, ids = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            cached = policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)
            path = Path(cached["cache_manifest_path"]).parent/"contexts.npy"
            values = np.load(path, mmap_mode="r+"); values[0, 0] += 1; values.flush(); del values
            with self.assertRaisesRegex(ValueError, "array changed"):
                policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)

    def test_unchanged_stat_verified_arrays_avoid_rehashing_large_context_files(self):
        cfg, arrays, model, ids = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            first = policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)
            with patch.object(policy, "_sha", wraps=policy._sha) as sha:
                second = policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)
            self.assertEqual(first["cache_key"], second["cache_key"])
            hashed = [Path(call.args[0]).name for call in sha.call_args_list]
            self.assertEqual(hashed, ["manifest.json"])

    def test_incomplete_cache_is_never_silently_overwritten(self):
        cfg, arrays, model, ids = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            first = policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)
            Path(first["cache_manifest_path"]).unlink()
            with self.assertRaisesRegex(ValueError, "incomplete immutable"):
                policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)

    def test_train_val_manifest_cannot_be_mislabeled_as_report_only_test(self):
        cfg, arrays, model, ids = fixture()
        arrays["producer_manifest"]["split"] = "train_val"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "verified Test producer"):
                policy.encode_test_contexts_cached(model, arrays, ids, cfg, tmp)


if __name__ == "__main__":
    unittest.main()
