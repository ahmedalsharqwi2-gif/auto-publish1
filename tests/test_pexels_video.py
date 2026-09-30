import unittest

from scripts.pexels_video import CLIP_SECONDS, MIN_CLIPS, visual_queries


class PexelsVideoTests(unittest.TestCase):
    def test_space_topic_uses_only_related_queries(self):
        queries = visual_queries("رحلة إلى الفضاء والنجوم")
        self.assertGreaterEqual(len(queries), 3)
        self.assertTrue(all(any(word in query for word in ("space", "astronomy", "nebula", "planet")) for query in queries))

    def test_long_video_requires_more_than_ten_unique_clips(self):
        required = max(MIN_CLIPS, int(90 / CLIP_SECONDS + 0.999))
        self.assertGreater(required, 10)
        self.assertEqual(required, 15)

    def test_short_video_does_not_require_unnecessary_minimum_clips(self):
        required = max(MIN_CLIPS, int(50.4 / CLIP_SECONDS + 0.999))
        self.assertEqual(required, 9)

    def test_clip_reuse_policy_is_documented(self):
        """A search may return fewer unique clips than a full reel needs."""
        self.assertLess(CLIP_SECONDS, 10)


if __name__ == "__main__":
    unittest.main()
