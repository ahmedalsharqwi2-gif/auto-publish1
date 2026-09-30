import unittest
from arabic_speech_core.gates import GateError, provider_preflight, text_gate
from arabic_speech_core.normalize import prepare_text
from arabic_speech_core.provider_pool import ProviderPool


class ArabicSpeechCoreTests(unittest.TestCase):
    def test_display_and_alignment_views_are_separate(self):
        prepared = prepare_text("القَاهِرَةُ فيها عِلْمٌ")
        self.assertIn("القَاهِرَةُ", prepared.display_text)
        self.assertEqual(prepared.align_text, "القاهرة فيها علم")
        self.assertEqual(len(prepared.span_map), 3)

    def test_missing_lexicon_is_nonfatal_and_normalized(self):
        result = text_gate("هذه جملة عربية سليمة", lexicon_path="/tmp/does-not-exist.json")
        self.assertEqual(result["spoken_text"], result["display_text"])

    def test_text_gate_rejects_non_arabic(self):
        with self.assertRaises(GateError):
            text_gate("plain English only", min_words=1)

    def test_preflight_rejects_non_arabic_voice(self):
        with self.assertRaises(GateError):
            provider_preflight(engine="edge", locale="ar-SA", voice="en-US-JennyNeural")

    def test_provider_pool_fails_over_after_429(self):
        calls = []
        def limited(_):
            class RateLimited(Exception):
                status_code = 429
            raise RateLimited("rate limited")
        def healthy(_):
            calls.append(True)
            return "ok"
        pool = ProviderPool([("limited", limited), ("healthy", healthy)], attempts=1)
        self.assertEqual(pool.call("hello"), "ok")
        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
