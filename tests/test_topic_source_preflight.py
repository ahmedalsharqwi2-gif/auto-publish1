import unittest
from unittest.mock import patch

from scripts.generate_content import CHANNEL_BRIEF, TOPIC_CATEGORIES, ContentGenerator, normalize_topic_response


class CurrentTopicPolicyTests(unittest.TestCase):
    def test_topic_prompt_contains_channel_brief_and_science_categories(self):
        generator = ContentGenerator()
        with patch("scripts.generate_content.llm_chat", return_value="ثقب أسود") as llm:
            result = generator.generate_topic("الفلك")

        self.assertEqual(result, "ثقب أسود")
        prompt = llm.call_args.args[0][0]["content"]
        self.assertIn(CHANNEL_BRIEF, prompt)
        self.assertIn("الفلك", prompt)
        self.assertTrue(any(category in prompt for category in TOPIC_CATEGORIES))

    def test_topic_normalization_accepts_json_and_removes_markup(self):
        self.assertEqual(normalize_topic_response('```json\n{"topic": "البرق"}\n```'), "البرق")
        self.assertEqual(normalize_topic_response("**الموضوع:** أعماق المحيطات"), "أعماق المحيطات")


if __name__ == "__main__":
    unittest.main()
