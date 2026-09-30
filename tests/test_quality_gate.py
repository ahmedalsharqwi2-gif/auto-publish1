import unittest

from arabic_tts_quality_checker import ArabicTTSQualityChecker


class QualityGateTests(unittest.TestCase):
    def test_nonfatal_text_warnings_do_not_block_publish(self):
        text = "هذه جملة عربية سليمة " * 30
        report = ArabicTTSQualityChecker(min_acceptable_score=0.75).generate_report(text)
        self.assertTrue(report.warnings)
        self.assertTrue(report.is_acceptable)
        self.assertEqual(report.issues, [])

    def test_asr_mismatch_is_not_a_blocking_issue(self):
        checker = ArabicTTSQualityChecker(min_acceptable_score=0.75)
        checker.check_audio_quality = lambda _audio, _text: (0.85, [])
        report = checker.generate_report("هذه جملة عربية سليمة " * 30, audio_path="audio.mp3")
        self.assertTrue(report.is_acceptable)


if __name__ == "__main__":
    unittest.main()
