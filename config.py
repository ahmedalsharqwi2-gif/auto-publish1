"""
Configuration file for Auto Publish Pipeline.

All API keys, tokens, and settings should be stored here or in environment variables.
Never commit sensitive data to version control.
"""

import os
from pathlib import Path

# =============================================================================
# BUFFER API CONFIGURATION
# =============================================================================

# Option 1: Set directly here (NOT recommended for production)
# BUFFER_ACCESS_TOKEN = "your_buffer_token_here"

# Option 2: Use environment variable (RECOMMENDED)
BUFFER_ACCESS_TOKEN = os.getenv("BUFFER_ACCESS_TOKEN", "")

# Optional: Specify which Buffer profiles to post to
# Leave empty to use default profile
PROFILES = os.getenv("BUFFER_PROFILES", "").split(",") if os.getenv("BUFFER_PROFILES") else []

# =============================================================================
# FILE PATHS
# =============================================================================

# Base directory (where this script is located)
BASE_DIR = Path(__file__).parent.resolve()

# Topics history file
TOPICS_FILE = BASE_DIR / "topic_history.json"

# Assets directory for images/media
ASSETS_DIR = BASE_DIR / "assets"

# Create assets directory if it doesn't exist
ASSETS_DIR.mkdir(exist_ok=True)

# =============================================================================
# POSTING SETTINGS
# =============================================================================

# Maximum number of posts per run
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "5"))

# Delay between posts (in seconds) to avoid rate limiting
POST_DELAY_SECONDS = int(os.getenv("POST_DELAY_SECONDS", "2"))

# Retry settings for API calls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "5"))

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Log file path
LOG_FILE = BASE_DIR / "auto_publish.log"

# =============================================================================
# BUFFER API ENDPOINTS
# =============================================================================

BUFFER_API_BASE = "https://api.bufferapp.com/1"
BUFFER_PROFILES_ENDPOINT = f"{BUFFER_API_BASE}/profiles.json"
BUFFER_UPDATES_ENDPOINT = f"{BUFFER_API_BASE}/updates/create.json"

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    
    if not BUFFER_ACCESS_TOKEN:
        errors.append("BUFFER_ACCESS_TOKEN is not set. Please set it in config.py or as environment variable.")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_buffer_headers():
    """Get headers for Buffer API requests."""
    return {
        "Authorization": f"Bearer {BUFFER_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

# Print configuration info on import (for debugging)
if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Buffer Token: {'Set' if BUFFER_ACCESS_TOKEN else 'NOT SET'}")
    print(f"Profiles: {PROFILES if PROFILES else 'Default'}")
    print(f"Topics File: {TOPICS_FILE}")
    print(f"Assets Dir: {ASSETS_DIR}")
    print(f"Log File: {LOG_FILE}")
