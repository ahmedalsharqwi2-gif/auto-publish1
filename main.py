"""
Auto Publish Pipeline - Main Script

This script reads topics from topic_history.json and publishes them to Buffer.
Features:
- Logging to file and console
- Error handling with retries
- Configuration from config.py
- Buffer API integration
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Import configuration
from config import (
    BUFFER_ACCESS_TOKEN,
    BUFFER_UPDATES_ENDPOINT,
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
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_topics():
    """Load topics from topic_history.json."""
    logger.info(f"Loading topics from {TOPICS_FILE}")
    
    try:
        with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pending = data.get('pending_topics', [])
        logger.info(f"Found {len(pending)} pending topics")
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

def post_to_buffer(text, profiles=None):
    """
    Post text to Buffer API with retry logic.
    
    Args:
        text: The text content to post
        profiles: Optional list of profile IDs
    
    Returns:
        dict: Buffer API response
    """
    payload = {
        "text": text,
        "token": BUFFER_ACCESS_TOKEN,
    }
    
    if profiles:
        payload["profile_ids"] = profiles
    
    headers = get_buffer_headers()
    
    # Retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"Posting to Buffer (attempt {attempt}/{MAX_RETRIES})")
            
            response = requests.post(
                BUFFER_UPDATES_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully posted to Buffer: {result.get('id', 'unknown')}")
            return result
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e} - Response: {e.response.text}")
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
    
    # Should never reach here
    raise RuntimeError("Max retries exceeded without success or proper error")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the auto-publish pipeline."""
    logger.info("=" * 60)
    logger.info("Auto Publish Pipeline Started")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Validate configuration
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Load topics
    try:
        data, pending_topics = load_topics()
    except Exception as e:
        logger.error(f"Failed to load topics: {e}")
        sys.exit(1)
    
    # Check if there are topics to post
    if not pending_topics:
        logger.warning("No pending topics to post")
        sys.exit(0)
    
    # Limit number of posts per run
    topics_to_post = pending_topics[:MAX_POSTS_PER_RUN]
    logger.info(f"Will post {len(topics_to_post)} topics this run")
    
    # Post each topic
    posted_count = 0
    failed_count = 0
    
    for i, topic in enumerate(topics_to_post, 1):
        logger.info(f"Processing topic {i}/{len(topics_to_post)}: {topic[:50]}...")
        
        try:
            # Post to Buffer
            result = post_to_buffer(topic)
            
            # Move from pending to posted
            posted_topic = {
                "topic": topic,
                "posted_at": datetime.now().isoformat(),
                "buffer_id": result.get('id'),
                "platforms": result.get('profile_ids', [])
            }
            
            data['posted_topics'].append(posted_topic)
            data['pending_topics'].remove(topic)
            
            posted_count += 1
            logger.info(f"✓ Posted successfully ({posted_count}/{len(topics_to_post)})")
            
            # Delay between posts
            if i < len(topics_to_post):
                logger.debug(f"Waiting {POST_DELAY_SECONDS}s before next post...")
                time.sleep(POST_DELAY_SECONDS)
        
        except Exception as e:
            logger.error(f"Failed to post topic: {e}")
            failed_count += 1
            # Continue with next topic
            continue
    
    # Save updated topics
    try:
        save_topics(data)
    except Exception as e:
        logger.error(f"Failed to save topics: {e}")
        # Don't exit - posting was successful
    
    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Completed")
    logger.info(f"Posted: {posted_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Remaining pending: {len(data['pending_topics'])}")
    logger.info("=" * 60)
    
    # Exit with error code if any failures
    if failed_count > 0:
        sys.exit(1)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
