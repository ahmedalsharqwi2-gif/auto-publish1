import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.publish_content import ContentPublisher, build_social_description


class PublisherTests(unittest.TestCase):
    def test_social_description_has_topic_body_and_hashtags(self):
        text = build_social_description("ثقب أسود في الفضاء", "شرح علمي مختصر")
        self.assertIn("ثقب أسود في الفضاء", text)
        self.assertIn("شرح علمي مختصر", text)
        self.assertIn("#علوم", text)
        self.assertIn("#فضاء", text)
    def test_channel_map_accepts_json_and_positional_values(self):
        self.assertEqual(
            ContentPublisher._channel_map(
                '{"youtube":"yt-1","instagram":"ig-1"}',
                ("youtube", "instagram"),
            ),
            {"youtube": "yt-1", "instagram": "ig-1"},
        )
        self.assertEqual(
            ContentPublisher._channel_map("yt-1,ig-1", ("youtube", "instagram")),
            {"youtube": "yt-1", "instagram": "ig-1"},
        )

    @patch("scripts.publish_content.requests.post")
    def test_create_buffer_post_requires_confirmed_post_id(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "data": {"createPost": {"post": {"id": "post-1", "dueAt": "2026-09-30T18:00:00Z"}}}
        }
        publisher = ContentPublisher()
        publisher.buffer_api_key = "buffer-test"
        result = publisher._create_buffer_post("channel-1", "عنوان\n\nنص", "https://public/video.mp4", None)
        self.assertEqual(result["id"], "post-1")
        sent_query = post.call_args.kwargs["json"]["query"]
        self.assertIn("channel-1", sent_query)
        self.assertIn("https://public/video.mp4", sent_query)
        self.assertIn("mode: addToQueue", sent_query)

    @patch("scripts.publish_content.requests.post")
    @patch("scripts.publish_content.ContentPublisher._release_asset_url", return_value="https://public/video.mp4")
    def test_partial_channel_failure_returns_false(self, _asset, post):
        post.return_value.status_code = 200
        post.return_value.json.side_effect = [
            {"data": {"createPost": {"post": {"id": "yt-post", "dueAt": "queued"}}}},
            {"data": {"createPost": {"message": "channel unavailable"}}},
        ]
        publisher = ContentPublisher()
        publisher.buffer_api_key = "buffer-test"
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            with patch.dict(
                os.environ,
                {
                    "PUBLISH_CHANNELS": "youtube,instagram",
                    "BUFFER_CHANNEL_IDS": json.dumps({"youtube": "yt-1", "instagram": "ig-1"}),
                },
                clear=False,
            ):
                self.assertFalse(publisher.publish_to_buffer(video, "title", "description", []))


if __name__ == "__main__":
    unittest.main()
