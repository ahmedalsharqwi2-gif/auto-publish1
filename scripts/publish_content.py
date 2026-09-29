# -*- coding: utf-8 -*-
"""
scripts/publish_content.py - نشر المحتوى
========================================
ينشر المحتوى على المنصات المختلفة بعد التحقق من الجودة.
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

log = logging.getLogger("pipeline")


class ContentPublisher:
    """ينشر المحتوى على المنصات المختلفة."""

    def __init__(self):
        self.buffer_api_key = os.getenv("BUFFER_API_KEY")
        self.buffer_endpoint = "https://api.buffer.com"

    def publish_to_buffer(
        self,
        video_path: Path,
        title: str,
        description: str,
        channel_ids: List[str],
        schedule_time: Optional[datetime] = None,
    ) -> bool:
        """Publish to Buffer API."""
        if not self.buffer_api_key:
            log.warning("BUFFER_API_KEY not set, skipping Buffer publishing")
            return False
        
        log.info(f"Publishing to Buffer: {title}")
        
        try:
            import requests
            
            for channel_id in channel_ids:
                payload = {
                    'media[0][file]': open(video_path, 'rb'),
                    'text': description,
                    'media_url': str(video_path),
                    'scheduled_at': int(schedule_time.timestamp()) if schedule_time else None,
                }
                
                response = requests.post(
                    f"{self.buffer_endpoint}/1/updates/create.json",
                    data=payload,
                    params={'access_token': self.buffer_api_key},
                    timeout=30,
                )
                
                if response.status_code == 200:
                    log.info(f"Successfully published to channel {channel_id}")
                else:
                    log.error(f"Failed to publish to channel {channel_id}: {response.text}")
                    return False
            
            return True
        except Exception as e:
            log.error(f"Buffer publishing failed: {e}")
            return False

    def save_metadata(
        self,
        metadata_path: Path,
        title: str,
        description: str,
        tags: List[str],
        video_path: Path,
    ) -> bool:
        """Save content metadata."""
        try:
            metadata = {
                'title': title,
                'description': description,
                'tags': tags,
                'video_path': str(video_path),
                'published_at': datetime.now().isoformat(),
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            log.info(f"Metadata saved to {metadata_path}")
            return True
        except Exception as e:
            log.error(f"Failed to save metadata: {e}")
            return False


if __name__ == "__main__":
    publisher = ContentPublisher()
    log.info("Content publisher initialized")
