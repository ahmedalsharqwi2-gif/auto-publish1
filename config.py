"""
Configuration file for Auto Publish Pipeline.

Uses Buffer's new GraphQL Public API (api.buffer.com).
The legacy REST API (api.bufferapp.com/1/) no longer accepts new Public API keys.
See: https://developers.buffer.com/guides/rest-migration.html
"""

import os
from pathlib import Path

# =============================================================================
# BUFFER GRAPHQL API CONFIGURATION
# =============================================================================

# Personal API key generated at https://publish.buffer.com/settings/api
# In GitHub Actions this comes from the BUFFER_API_KEY secret.
BUFFER_API_KEY = os.getenv("BUFFER_API_KEY", "")

# Comma-separated channel IDs to post to, e.g. "64f1...,64f2..."
# In GitHub Actions this comes from the BUFFER_CHANNEL_IDS secret.
BUFFER_CHANNEL_IDS = [
    cid.strip() for cid in os.getenv("BUFFER_CHANNEL_IDS", "").split(",") if cid.strip()
]

# =============================================================================
# FILE PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
TOPICS_FILE = BASE_DIR / os.getenv("TOPIC_HISTORY_FILE", "topic_history.json")
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# =============================================================================
# POSTING SETTINGS
# =============================================================================

MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "5"))
POST_DELAY_SECONDS = int(os.getenv("POST_DELAY_SECONDS", "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "5"))

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "auto_publish.log"

# =============================================================================
# BUFFER GRAPHQL ENDPOINT
# =============================================================================

BUFFER_GRAPHQL_ENDPOINT = "https://api.buffer.com"

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that all required configuration is present."""
    errors = []

    if not BUFFER_API_KEY:
        errors.append("BUFFER_API_KEY is not set. Generate one at https://publish.buffer.com/settings/api")
    if not BUFFER_CHANNEL_IDS:
        errors.append("BUFFER_CHANNEL_IDS is not set. Provide at least one channel ID (comma-separated).")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_buffer_headers():
    """Get headers for Buffer GraphQL API requests."""
    return {
        "Authorization": f"Bearer {BUFFER_API_KEY}",
        "Content-Type": "application/json",
    }

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Buffer API Key: {'Set' if BUFFER_API_KEY else 'NOT SET'}")
    print(f"Channel IDs: {BUFFER_CHANNEL_IDS if BUFFER_CHANNEL_IDS else 'NOT SET'}")
    print(f"Topics File: {TOPICS_FILE}")
    print(f"Log File: {LOG_FILE}")
