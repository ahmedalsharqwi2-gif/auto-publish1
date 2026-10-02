import json
import os
import unittest
from pathlib import Path


class ModelPolicyTests(unittest.TestCase):
    def test_policy_has_one_shared_schema_and_provider_order(self):
        policy = json.loads(Path("config/model_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["schema_version"], 1)
        self.assertEqual(list(policy["providers"]), ["gemini", "fallback", "openrouter"])
        for provider in policy["providers"].values():
            self.assertTrue(provider["preferred_models"])
            self.assertIn(503, provider["retry_statuses"])

    def test_preflight_is_importable_without_third_party_dependencies(self):
        import scripts.model_preflight as preflight
        self.assertEqual(preflight.DEFAULT_GEMINI, "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
