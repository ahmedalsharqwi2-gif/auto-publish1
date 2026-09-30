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
from scripts.assemble_video import assemble_video, probe_duration
from scripts.publish_content import build_social_description

MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "60"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "90"))


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

            audio_duration = probe_duration(output_path)
            log.info("✓ Audio duration: %.2fs (required %.0f–%.0fs)", audio_duration, MIN_AUDIO_SECONDS, MAX_AUDIO_SECONDS)
            if not MIN_AUDIO_SECONDS <= audio_duration <= MAX_AUDIO_SECONDS:
                log.error(
                    "Audio duration outside publishing window: %.2fs; refusing to publish",
                    audio_duration,
                )
                return False

            # Build the actual vertical MP4 before publishing. Previously the
            # pipeline sent narration.mp3 to Buffer, so there was no visual
            # layer or on-screen Arabic text at all.
            log.info("\n[Step 6] Assembling vertical video with Arabic captions...")
            video_path = assemble_video(
                output_path,
                checked_text,
                Path("output/final_video.mp4"),
                topic=topic,
            )
            log.info(f"✓ Video assembled: {video_path}")

            # Step 7: Publish
            log.info("\n[Step 7] Publishing content...")
            title = topic[:60]
            description = build_social_description(topic, narration)
            channels = os.getenv("PUBLISH_CHANNELS", "youtube,tiktok,instagram").split(",")

            if os.getenv("PUBLISH_DRY_RUN", "false").lower() == "true":
                log.info("DRY RUN: skipping external publication")
                log.info("DRY RUN metadata: %s", description[:500])
                return True

            publish_success = self.publisher.publish_to_buffer(
                video_path=video_path,
                title=title,
                description=description,
                channel_ids=channels,
            )

            if not publish_success:
                log.error("Publishing was not confirmed for every configured channel")
                return False
            log.info("✓ Content published successfully to every configured channel")

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
