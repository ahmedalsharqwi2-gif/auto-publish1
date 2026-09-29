import unittest
from unittest.mock import patch

from scripts.generate_content import CHANNEL_BRIEF, TOPIC_CATEGORIES, ContentGenerator


class ContentGeneratorTopicPolicyTests(unittest.TestCase):
    def test_channel_brief_and_topic_bank_are_used(self):
        with patch(
            "scripts.generate_content.llm_chat",
            return_value="عنوان جذاب\nشرح علمي موجز",
        ) as llm:
            ContentGenerator().generate_topic()

        prompt = llm.call_args.args[0][0]["content"]
        self.assertIn("أسرار الفضاء", CHANNEL_BRIEF)
        self.assertIn(CHANNEL_BRIEF, prompt)
        self.assertGreaterEqual(len(TOPIC_CATEGORIES), 10)
        self.assertIn("أعماق المحيطات", prompt)
        self.assertIn("الذكاء الاصطناعي", prompt)
        self.assertIn("قابلة للتحقق", prompt)


if __name__ == "__main__":
    unittest.main()
