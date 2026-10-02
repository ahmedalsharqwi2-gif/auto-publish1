import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.renew_topic_bank import merge_related_trends, query_groups, build_candidates


class RenewalToolTests(unittest.TestCase):
    def test_related_aliases_use_strongest_signal_in_one_group(self):
        groups = query_groups("AI, artificial intelligence;الذكاء الاصطناعي")
        merged = merge_related_trends({"AI": 82, "artificial intelligence": 0, "الذكاء الاصطناعي": 0}, groups)
        self.assertEqual(merged["AI / artificial intelligence"], 82)
        self.assertEqual(merged["الذكاء الاصطناعي"], 0)

    def test_evidence_scarce_candidate_is_not_created(self):
        self.assertEqual(build_candidates({}, [], [("AI",)], "science"), [])

    def test_candidate_has_youtube_and_trends_evidence(self):
        candidates = build_candidates(
            {"AI": 80},
            [{"title": "AI discovery", "description": "A visual explanation", "published_at": "2026-01-01T00:00:00Z", "url": "https://youtube.com/watch?v=x"}],
            [("AI",)], "science",
        )
        self.assertEqual(len(candidates), 1)
        self.assertGreaterEqual(candidates[0].score, 70)


if __name__ == "__main__":
    unittest.main()
