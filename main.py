"""
Auto Publish Pipeline - Main Script

This script reads topics from topic_history.json and publishes them to Buffer
using Buffer's GraphQL Public API.

IMPORTANT: Buffer validates posts per-channel-type:
- Facebook: requires input.facebook.type = 'post' | 'story' | 'reel'
- TikTok: requires at least one image/video attached
- YouTube: requires a video, title, and category

This script currently only has text topics (no rendered video yet), so it:
- Posts text-only content to Facebook channels (type='post')
- Skips TikTok/YouTube channels with a clear log message until a video
  pipeline (like the one in horror_content) produces real media files.

Features:
- Logging to file and console
- Error handling with retries
- Configuration from config.py
- Supports both old (list) and new (dict) topic_history.json formats
- Supports both plain-string topics and dict topics (title/hook_text/etc.)
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from config import (
    BUFFER_API_KEY,
    BUFFER_CHANNEL_IDS,
    BUFFER_GRAPHQL_ENDPOINT,
    LOG_FILE,
    LOG_LEVEL,
    MAX_POSTS_PER_RUN,
    MAX_RETRIES,
    POST_DELAY_SECONDS,
    RETRY_DELAY_SECONDS,
    TOPICS_FILE,
    get_buffer_headers,
    validate_config,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging to file and console."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# =============================================================================
# CHANNEL TYPE DETECTION
# =============================================================================

# Channels that require video/image media we do not produce yet.
# We detect this dynamically from Buffer's error message instead of
# hardcoding channel IDs, so this keeps working if channels are added/removed.
VIDEO_REQUIRED_MARKERS = [
    "require a video",
    "require at least one image or video",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_topics():
    """Load topics from topic_history.json (supports both list and dict formats)."""
    logger.info(f"Loading topics from {TOPICS_FILE}")

    try:
        with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            logger.info(f"Detected old list format with {len(data)} topics")
            pending = data
            data = {"posted_topics": [], "pending_topics": data}
        elif isinstance(data, dict):
            pending = data.get('pending_topics', [])
            logger.info(f"Detected dict format with {len(pending)} pending topics")
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")

        return data, pending

    except FileNotFoundError:
        logger.error(f"Topics file not found: {TOPICS_FILE}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in topics file: {e}")
        raise

def save_topics(data):
    """Save updated topics back to topic_history.json."""
    logger.info(f"Saving updated topics to {TOPICS_FILE}")

    try:
        with open(TOPICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Topics saved successfully")
    except Exception as e:
        logger.error(f"Failed to save topics: {e}")
        raise

def get_topic_text(topic):
    """Extract postable text from a topic (handles string and dict formats)."""
    if isinstance(topic, str):
        return topic
    elif isinstance(topic, dict):
        for key in ['text', 'topic', 'title', 'hook_text', 'content', 'message']:
            if key in topic and topic[key]:
                return str(topic[key])
        return json.dumps(topic, ensure_ascii=False)
    else:
        return str(topic)

def post_to_buffer(text, channel_id):
    """
    Create a post on Buffer using the GraphQL createPost mutation.

    Includes facebook.type='post' so Facebook channels accept a text-only post.
    TikTok/YouTube channels will still be rejected by Buffer (they require
    media) - this is expected until a video pipeline provides real assets.

    Args:
        text: The post text/caption
        channel_id: The Buffer channel ID to post to

    Returns:
        dict: {"post": {...}} on success

    Raises:
        SkipChannelError: if this channel type requires media we don't have
        RuntimeError: for any other failure
    """
    mutation = """
    mutation CreatePost($input: CreatePostInput!) {
      createPost(input: $input) {
        ... on PostActionSuccess {
          post {
            id
            text
            dueAt
          }
        }
        ... on MutationError {
          message
        }
      }
    }
    """

    variables = {
        "input": {
            "text": text,
            "channelId": channel_id,
            "schedulingType": "automatic",
            "mode": "addToQueue",
            "facebook": {
                "type": "post"
            }
        }
    }

    headers = get_buffer_headers()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"Posting to Buffer channel {channel_id} (attempt {attempt}/{MAX_RETRIES})")

            response = requests.post(
                BUFFER_GRAPHQL_ENDPOINT,
                json={"query": mutation, "variables": variables},
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            if "errors" in result and result["errors"]:
                raise RuntimeError(f"GraphQL errors: {result['errors']}")

            payload = result.get("data", {}).get("createPost", {})

            if "message" in payload and "post" not in payload:
                error_message = payload["message"]
                if any(marker in error_message for marker in VIDEO_REQUIRED_MARKERS):
                    raise SkipChannelError(error_message)
                raise RuntimeError(f"Buffer MutationError: {error_message}")

            post = payload.get("post")
            if not post:
                raise RuntimeError(f"Unexpected Buffer response: {payload}")

            logger.info(f"Successfully posted to Buffer: {post.get('id', 'unknown')}")
            return payload

        except SkipChannelError:
            raise

        except requests.exceptions.HTTPError as e:
            body = e.response.text if e.response is not None else str(e)
            logger.error(f"HTTP error: {e} - Response: {body}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise

    raise RuntimeError("Max retries exceeded")

class SkipChannelError(Exception):
    """Raised when a channel requires media (video/image) we don't have yet."""
    pass

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the auto-publish pipeline."""
    logger.info("=" * 60)
    logger.info("Auto Publish Pipeline Started")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    try:
        data, pending_topics = load_topics()
    except Exception as e:
        logger.error(f"Failed to load topics: {e}")
        sys.exit(1)

    if not pending_topics:
        logger.warning("No pending topics to post")
        sys.exit(0)

    topics_to_post = pending_topics[:MAX_POSTS_PER_RUN]
    logger.info(f"Will post {len(topics_to_post)} topics this run across {len(BUFFER_CHANNEL_IDS)} channel(s)")

    posted_count = 0
    failed_count = 0
    skipped_channels = set()

    for i, topic in enumerate(topics_to_post, 1):
        topic_text = get_topic_text(topic)
        logger.info(f"Processing topic {i}/{len(topics_to_post)}: {topic_text[:50]}...")

        topic_posted = False
        last_error = None

        for channel_id in BUFFER_CHANNEL_IDS:
            try:
                result = post_to_buffer(topic_text, channel_id)

                posted_topic = {
                    "topic": topic_text,
                    "posted_at": datetime.now().isoformat(),
                    "buffer_post_id": result.get("post", {}).get("id"),
                    "channel_id": channel_id,
                }
                data['posted_topics'].append(posted_topic)
                topic_posted = True

            except SkipChannelError as e:
                if channel_id not in skipped_channels:
                    logger.warning(
                        f"Skipping channel {channel_id} (requires video/image, "
                        f"not available yet): {e}"
                    )
                    skipped_channels.add(channel_id)

            except Exception as e:
                logger.error(f"Failed to post topic to channel {channel_id}: {e}")
                last_error = e

        if topic_posted:
            data['pending_topics'].remove(topic)
            posted_count += 1
            logger.info(f"✓ Posted successfully ({posted_count}/{len(topics_to_post)})")
        else:
            failed_count += 1
            if last_error:
                logger.error(f"Failed to post topic to any channel: {last_error}")
            else:
                logger.warning("Topic not posted: all channels require media not yet available")

        if i < len(topics_to_post):
            logger.debug(f"Waiting {POST_DELAY_SECONDS}s before next post...")
            time.sleep(POST_DELAY_SECONDS)

    try:
        save_topics(data)
    except Exception as e:
        logger.error(f"Failed to save topics: {e}")

    logger.info("=" * 60)
    logger.info("Pipeline Completed")
    logger.info(f"Posted: {posted_count}")
    logger.info(f"Failed/Skipped: {failed_count}")
    if skipped_channels:
        logger.info(f"Channels needing media pipeline: {', '.join(skipped_channels)}")
    logger.info(f"Remaining pending: {len(data['pending_topics'])}")
    logger.info("=" * 60)

    if posted_count == 0 and failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
