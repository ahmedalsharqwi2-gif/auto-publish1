import json
import unittest
from unittest.mock import patch

from scripts.generate_content import normalize_narration_response
from scripts.publish_content import build_social_description


class CurrentContentOutputTests(unittest.TestCase):
    def test_narration_normalization_extracts_json_payload(self):
        raw = json.dumps({"narration": "هذه حقيقة علمية موثقة."}, ensure_ascii=False)
        self.assertEqual(normalize_narration_response(raw), "هذه حقيقة علمية موثقة.")

    def test_narration_normalization_removes_transport_fences(self):
        self.assertEqual(
            normalize_narration_response("```text\nالنص: حقيقة عن الفضاء\n```"),
            "حقيقة عن الفضاء",
        )

    def test_social_description_contains_topic_narration_and_hashtags(self):
        result = build_social_description("كيف يعمل البرق؟", "شرح علمي قصير عن البرق.")
        self.assertIn("كيف يعمل البرق؟", result)
        self.assertIn("شرح علمي قصير عن البرق.", result)
        self.assertIn("#", result)


if __name__ == "__main__":
    unittest.main()
