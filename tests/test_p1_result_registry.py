import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock

from unidream.experiments import p1_result_registry as registry


class P1ResultRegistryTests(unittest.TestCase):
    def test_fixed_registry_loads_exact_order_and_is_immutable(self):
        result = registry.load_p1_result_registry()
        self.assertEqual(result.manifest_sha256, registry.REGISTERED_MANIFEST_SHA256)
        self.assertEqual(len(result.trials), 56)
        self.assertEqual(len(result.comparisons), 16)
        self.assertEqual(result.trials[0]["trial_id"], "S0__zero_return__off")
        self.assertEqual(
            result.comparisons[-1]["comparison_id"],
            "S3__injected_vs_control__ridge__utility__cost_on",
        )
        self.assertIsInstance(result.trials[0], MappingProxyType)
        self.assertIs(result.trials_by_id["S1__ridge__on"], result.trials[13])
        with self.assertRaises(TypeError):
            result.trials[0]["trial_id"] = "changed"

    def test_jsonl_rejects_blank_duplicate_and_nonfinite_records(self):
        with self.assertRaisesRegex(registry.P1ResultRegistryError, "blank"):
            registry._parse_jsonl(b'{"trial_id":"a"}\n\n', label="trial")
        with self.assertRaisesRegex(registry.P1ResultRegistryError, "duplicate"):
            registry._parse_jsonl(b'{"trial_id":"a","trial_id":"b"}\n', label="trial")
        with self.assertRaisesRegex(registry.P1ResultRegistryError, "non-finite"):
            registry._parse_jsonl(b'{"value":NaN}\n', label="trial")

    def test_runtime_rechecks_registry_digest_after_manifest_authentication(self):
        mutable = json.loads(
            Path(registry.DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "docs" / "experiments" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            trial_path = root / "docs" / "experiments" / "trials.jsonl"
            comparison_path = root / "docs" / "experiments" / "comparisons.jsonl"
            trial_path.write_text('{"trial_id":"forged"}\n', encoding="utf-8")
            comparison_path.write_text('{"comparison_id":"forged"}\n', encoding="utf-8")
            mutable["common"]["trial_registry"]["path"] = str(
                trial_path.relative_to(root)
            )
            mutable["common"]["primary_comparison_registry"]["path"] = str(
                comparison_path.relative_to(root)
            )
            manifest_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                registry,
                "load_fixed_manifest",
                return_value=mutable,
            ):
                with self.assertRaisesRegex(registry.P1ResultRegistryError, "SHA-256"):
                    registry.load_p1_result_registry(manifest_path)


if __name__ == "__main__":
    unittest.main()
