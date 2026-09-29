# -*- coding: utf-8 -*-
"""Publish a publicly hosted MP4 to Buffer using its current GraphQL API."""

from __future__ import annotations

import logging
import os
from typing import Sequence

import requests

log = logging.getLogger("pipeline")
BUFFER_ENDPOINT = "https://api.buffer.com"
CREATE_POST_MUTATION = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess {
      post { id text }
    }
    ... on MutationError {
      message
    }
  }
}
"""


class ContentPublisher:
    """Queue or publish a reel to the configured Buffer channels."""

    def __init__(self, api_key: str | None = None, endpoint: str = BUFFER_ENDPOINT):
        self.buffer_api_key = api_key if api_key is not None else os.getenv("BUFFER_API_KEY")
        self.buffer_endpoint = endpoint
        self.post_mode = os.getenv("BUFFER_POST_MODE", "addToQueue")

    def publish_to_buffer(
        self,
        video_url: str,
        title: str,
        description: str,
        channel_ids: Sequence[str],
    ) -> bool:
        """Create one Buffer video post per channel using a public HTTPS URL."""
        if not self.buffer_api_key:
            log.error("BUFFER_API_KEY is not set; refusing to report publishing success")
            return False
        normalized_channel_ids = [str(channel_id).strip() for channel_id in channel_ids if str(channel_id).strip()]
        if not normalized_channel_ids:
            log.error("No Buffer channel IDs were configured")
            return False
        if not video_url.startswith("https://"):
            log.error("Buffer video URL must be publicly accessible HTTPS")
            return False
        if self.post_mode not in {"addToQueue", "shareNow"}:
            log.error("Unsupported BUFFER_POST_MODE: %s", self.post_mode)
            return False

        text = f"{title.strip()}\n\n{description.strip()}".strip()
        headers = {
            "Authorization": f"Bearer {self.buffer_api_key}",
            "Content-Type": "application/json",
        }
        succeeded = True
        for channel_id in normalized_channel_ids:
            post_input = {
                "channelId": channel_id,
                "text": text,
                "schedulingType": "automatic",
                "mode": self.post_mode,
                "assets": [
                    {
                        "video": {
                            "url": video_url,
                            "metadata": {"thumbnailOffset": 2000},
                        }
                    }
                ],
            }
            try:
                response = requests.post(
                    self.buffer_endpoint,
                    headers=headers,
                    json={
                        "query": CREATE_POST_MUTATION,
                        "variables": {"input": post_input},
                    },
                    timeout=60,
                )
                if response.status_code != 200:
                    log.error(
                        "Buffer rejected channel %s: HTTP %s %s",
                        channel_id,
                        response.status_code,
                        response.text[:500],
                    )
                    succeeded = False
                    continue
                payload = response.json()
                if payload.get("errors"):
                    log.error("Buffer GraphQL errors for channel %s: %s", channel_id, payload["errors"])
                    succeeded = False
                    continue
                result = (payload.get("data") or {}).get("createPost") or {}
                post = result.get("post")
                if not post or not post.get("id"):
                    log.error(
                        "Buffer did not create a post for channel %s: %s",
                        channel_id,
                        result.get("message", "missing PostActionSuccess.post.id"),
                    )
                    succeeded = False
                    continue
                log.info("Buffer post %s created for channel %s", post["id"], channel_id)
            except (requests.RequestException, ValueError, TypeError) as exc:
                log.error("Buffer request failed for channel %s: %s", channel_id, exc)
                succeeded = False
        return succeeded
