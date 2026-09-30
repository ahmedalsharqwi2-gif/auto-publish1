import unittest

from arabic_tts_quality_checker import ArabicTTSQualityChecker


class QualityGateTests(unittest.TestCase):
    def test_nonfatal_text_warnings_do_not_block_publish(self):
        text = "هذه جملة عربية سليمة " * 30
        report = ArabicTTSQualityChecker(min_acceptable_score=0.75).generate_report(text)
        self.assertTrue(report.warnings)
        self.assertTrue(report.is_acceptable)
        self.assertEqual(report.issues, [])


if __name__ == "__main__":
    unittest.main()
