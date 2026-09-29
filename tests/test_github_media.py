import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.github_media import MediaHostingError, host_video_on_github


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


class GitHubMediaTests(unittest.TestCase):
    def test_uploads_to_public_release_and_returns_public_asset_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "reel.mp4"
            video.write_bytes(b"fake-mp4-content")
            responses = [
                FakeResponse(200, {"private": False}),
                FakeResponse(201, {"upload_url": "https://uploads.example/release/assets{?name,label}"}),
                FakeResponse(201, {"browser_download_url": "https://github.com/o/r/releases/download/t/reel.mp4"}),
            ]
            with (
                patch("scripts.github_media.requests.get", return_value=responses[0]) as get,
                patch("scripts.github_media.requests.post", side_effect=responses[1:]) as post,
            ):
                url = host_video_on_github(
                    video,
                    token="test-token",
                    repository="owner/repo",
                    run_id="123",
                    run_attempt="2",
                    api_base="https://api.github.test",
                )

        self.assertEqual(url, "https://github.com/o/r/releases/download/t/reel.mp4")
        self.assertEqual(get.call_args.args[0], "https://api.github.test/repos/owner/repo")
        self.assertEqual(post.call_count, 2)
        create_release = post.call_args_list[0]
        self.assertEqual(create_release.kwargs["json"]["tag_name"], "auto-publish-123-attempt-2")
        upload = post.call_args_list[1]
        self.assertEqual(upload.kwargs["headers"]["Content-Type"], "video/mp4")
        self.assertEqual(upload.kwargs["params"], {"name": "reel.mp4"})

    def test_private_repository_is_rejected_before_release_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "reel.mp4"
            video.write_bytes(b"fake-mp4-content")
            with (
                patch(
                    "scripts.github_media.requests.get",
                    return_value=FakeResponse(200, {"private": True}),
                ),
                patch("scripts.github_media.requests.post") as post,
            ):
                with self.assertRaises(MediaHostingError):
                    host_video_on_github(
                        video,
                        token="test-token",
                        repository="owner/repo",
                        run_id="123",
                        api_base="https://api.github.test",
                    )
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
