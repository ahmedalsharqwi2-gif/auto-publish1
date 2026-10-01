import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.generate_content import ContentGenerator


class ContentLengthPolicyTests(unittest.TestCase):
    @patch("scripts.generate_content.llm_chat")
    def test_near_boundary_narration_is_accepted(self, llm_chat):
        narration = "هذه معلومة علمية مفيدة عن الكون والنجوم والمادة والطاقة " * 16
        self.assertGreaterEqual(len(narration.split()), 105)
        self.assertLessEqual(len(narration.split()), 150)
        llm_chat.return_value = narration
        generator = ContentGenerator(min_words=105, max_words=150)
        result = generator.generate_narration("موضوع علمي اختباري")
        self.assertGreaterEqual(len(result.split()), 105)
        self.assertLessEqual(len(result.split()), 150)

    @patch("scripts.generate_content.llm_chat")
    def test_minor_upper_bound_overshoot_is_trimmed(self, llm_chat):
        narration = "هذه معلومة علمية مفيدة عن الكون والنجوم والمادة والطاقة " * 17
        narration = " ".join(narration.split()[:162])
        llm_chat.return_value = narration
        generator = ContentGenerator(min_words=105, max_words=150)
        generator.grammar_fixer.fix_text = lambda text: (text, [])
        generator.quality_checker.generate_report = lambda text: SimpleNamespace(
            overall_score=1.0, issues=[], is_acceptable=True
        )
        result = generator.generate_narration("موضوع علمي اختباري")
        self.assertEqual(len(result.split()), 150)


if __name__ == "__main__":
    unittest.main()
