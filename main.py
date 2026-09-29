# -*- coding: utf-8 -*-
"""
main.py - نقطة الدخول الرئيسية
=================================
متحكم المشروع - ينسق بين جميع وحدات المشروع.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")

from scripts import (
    ContentGenerator,
    VoiceGenerator,
    QualityCheckPipeline,
    ContentPublisher,
)


class AutoPublishPipeline:
    """خط الأنابيب الرئيسي لتوليد ونشر المحتوى."""

    def __init__(self):
        self.content_generator = ContentGenerator(
            min_words=int(os.getenv("MIN_WORDS", "300")),
            max_words=int(os.getenv("MAX_WORDS", "1000")),
        )
        self.voice_generator = VoiceGenerator(
            output_dir=Path(os.getenv("OUTPUT_DIR", "./output"))
        )
        self.quality_checker = QualityCheckPipeline(
            min_acceptable_score=float(os.getenv("MIN_QUALITY_SCORE", "0.75"))
        )
        self.publisher = ContentPublisher()

    def run(self, category: str = "", topic: Optional[str] = None) -> bool:
        """Run the complete pipeline."""
        log.info("\n" + "="*60)
        log.info("Starting Auto-Publish Pipeline")
        log.info("="*60)

        try:
            # Step 1: Generate topic
            if topic is None:
                log.info("\n[Step 1] Generating topic...")
                topic = self.content_generator.generate_topic(category)
                log.info(f"✓ Topic generated: {topic[:80]}...")
            else:
                log.info(f"\n[Step 1] Using provided topic: {topic[:80]}...")

            # Step 2: Generate narration
            log.info("\n[Step 2] Generating narration...")
            narration = self.content_generator.generate_narration(topic)
            log.info(f"✓ Narration generated ({len(narration.split())} words)")

            # Step 3: Quality check
            log.info("\n[Step 3] Checking content quality...")
            checked_text, text_report = self.quality_checker.check_text(narration)
            log.info(f"✓ Quality check complete (score: {text_report.overall_score:.2f}/1.0)")

            if not text_report.is_acceptable:
                log.error("Content quality not acceptable for publishing")
                return False

            # Step 4: Generate voice
            log.info("\n[Step 4] Generating voice...")
            output_path, success = self.voice_generator.generate(
                checked_text,
                output_path=Path("output/narration.mp3"),
            )

            if not success:
                log.error("Voice generation failed")
                return False

            log.info(f"✓ Voice generated: {output_path}")

            # Step 5: Audio quality check
            log.info("\n[Step 5] Checking audio quality...")
            audio_report = self.quality_checker.check_audio(output_path, checked_text)
            log.info(f"✓ Audio quality check complete (score: {audio_report.overall_score:.2f}/1.0)")

            if not audio_report.is_acceptable:
                log.error("Audio quality not acceptable for publishing")
                return False

            # Step 6: Publish
            log.info("\n[Step 6] Publishing content...")
            title = topic[:60]
            description = f"{narration[:200]}...\n\n#المحتوى_المولد_آلياً #الذكاء_الاصطناعي"
            channels = os.getenv("PUBLISH_CHANNELS", "youtube,tiktok,instagram").split(",")

            publish_success = self.publisher.publish_to_buffer(
                video_path=output_path,
                title=title,
                description=description,
                channel_ids=channels,
            )

            if publish_success:
                log.info(f"✓ Content published successfully")
            else:
                log.warning("Publishing completed with warnings")

            log.info("\n" + "="*60)
            log.info("Pipeline completed successfully!")
            log.info("="*60 + "\n")
            return True

        except Exception as e:
            log.error(f"Pipeline failed: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    pipeline = AutoPublishPipeline()
    
    # Get topic from environment or command line
    topic = os.getenv("TOPIC") or (sys.argv[1] if len(sys.argv) > 1 else None)
    category = os.getenv("CATEGORY", "عام")
    
    success = pipeline.run(category=category, topic=topic)
    sys.exit(0 if success else 1)
