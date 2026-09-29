import unittest
from unittest.mock import patch

from scripts.publish_content import ContentPublisher


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


class BufferPublisherTests(unittest.TestCase):
    def test_creates_graphql_video_post_with_channel_and_public_url(self):
        response = FakeResponse(
            payload={"data": {"createPost": {"post": {"id": "post-123", "text": "Title"}}}}
        )
        with patch("scripts.publish_content.requests.post", return_value=response) as post:
            publisher = ContentPublisher(api_key="test-token", endpoint="https://buffer.test")
            self.assertTrue(
                publisher.publish_to_buffer(
                    video_url="https://github.com/owner/repo/releases/download/tag/reel.mp4",
                    title="Title",
                    description="Caption",
                    channel_ids=["channel-1"],
                )
            )

        call = post.call_args
        self.assertEqual(call.args[0], "https://buffer.test")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer test-token")
        payload = call.kwargs["json"]
        self.assertIn("createPost", payload["query"])
        post_input = payload["variables"]["input"]
        self.assertEqual(post_input["channelId"], "channel-1")
        self.assertEqual(post_input["text"], "Title\n\nCaption")
        self.assertEqual(post_input["mode"], "addToQueue")
        self.assertEqual(
            post_input["assets"][0]["video"]["url"],
            "https://github.com/owner/repo/releases/download/tag/reel.mp4",
        )

    def test_graphql_mutation_error_returns_false(self):
        response = FakeResponse(
            payload={"data": {"createPost": {"message": "Invalid video asset"}}}
        )
        with patch("scripts.publish_content.requests.post", return_value=response):
            publisher = ContentPublisher(api_key="test-token")
            self.assertFalse(
                publisher.publish_to_buffer(
                    video_url="https://media.example/reel.mp4",
                    title="Title",
                    description="Caption",
                    channel_ids=["channel-1"],
                )
            )

    def test_missing_channel_ids_or_non_https_url_fails_closed(self):
        publisher = ContentPublisher(api_key="test-token")
        with patch("scripts.publish_content.requests.post") as post:
            self.assertFalse(
                publisher.publish_to_buffer("https://media.example/video.mp4", "T", "D", [])
            )
            self.assertFalse(
                publisher.publish_to_buffer("https://media.example/video.mp4", "T", "D", [" ", ""])
            )
            self.assertFalse(
                publisher.publish_to_buffer("file:///tmp/video.mp4", "T", "D", ["channel-1"])
            )
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
