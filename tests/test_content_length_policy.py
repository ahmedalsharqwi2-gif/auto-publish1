import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.generate_content import ContentGenerator, trim_to_complete_sentence


class ContentLengthPolicyTests(unittest.TestCase):
    def test_trim_refuses_to_cut_inside_a_sentence(self):
        text = "كلمة " * 110 + "نهاية كاملة. " + "باقي النص " * 50
        clipped = trim_to_complete_sentence(text, 115, 105)
        self.assertIsNotNone(clipped)
        self.assertTrue(clipped.endswith("."))

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
    def test_minor_upper_bound_overshoot_trims_only_at_sentence_boundary(self, llm_chat):
        sentence = "هذه معلومة علمية مفيدة عن الكون والنجوم والمادة والطاقة."
        narration = " ".join([sentence] * 22)
        llm_chat.return_value = narration
        generator = ContentGenerator(min_words=105, max_words=150)
        generator.grammar_fixer.fix_text = lambda text: (text, [])
        generator.quality_checker.generate_report = lambda text: SimpleNamespace(
            overall_score=1.0, issues=[], is_acceptable=True
        )
        result = generator.generate_narration("موضوع علمي اختباري")
        self.assertLessEqual(len(result.split()), 150)
        self.assertTrue(result.endswith("."))


if __name__ == "__main__":
    unittest.main()
