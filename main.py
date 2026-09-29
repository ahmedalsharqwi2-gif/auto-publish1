# -*- coding: utf-8 -*-
"""Entry point for the simplified Arabic Reels generation and publishing pipeline."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from scripts.assemble_reel import assemble_reel
from scripts.generate_content import ContentGenerator
from scripts.generate_voice import VoiceGenerator
from scripts.github_media import host_video_on_github
from scripts.publish_content import ContentPublisher
from scripts.quality_check import QualityCheckPipeline

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


def parse_channel_ids(raw: str | None) -> list[str]:
    """Parse the comma-separated BUFFER_CHANNEL_IDS secret without fake defaults."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


class AutoPublishPipeline:
    """Generate Arabic narration, render a vertical reel, host it and queue it."""

    def __init__(self):
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        self.content_generator = ContentGenerator(
            min_words=int(os.getenv("MIN_WORDS", "90")),
            max_words=int(os.getenv("MAX_WORDS", "165")),
        )
        self.voice_generator = VoiceGenerator(output_dir=self.output_dir)
        self.quality_checker = QualityCheckPipeline(
            min_acceptable_score=float(os.getenv("MIN_QUALITY_SCORE", "0.75"))
        )
        self.publisher = ContentPublisher()

    def run(self, category: str = "", topic: str | None = None) -> bool:
        log.info("Starting simplified Arabic Reels pipeline")
        try:
            channel_ids = parse_channel_ids(os.getenv("BUFFER_CHANNEL_IDS"))
            if not channel_ids:
                raise ValueError("BUFFER_CHANNEL_IDS is required (comma-separated Buffer channel IDs)")

            if topic is None:
                log.info("Generating a topic for category %s", category or "general")
                topic = self.content_generator.generate_topic(category)
            topic = topic.strip()
            if not topic:
                raise ValueError("The generated topic is empty")

            log.info("Generating narration for topic: %s", topic[:80])
            narration = self.content_generator.generate_narration(topic)
            narration, text_report = self.quality_checker.check_text(narration)
            log.info("Text quality score: %.2f", text_report.overall_score)
            if not text_report.is_acceptable:
                log.error("Text did not pass the quality gate: %s", text_report.issues)
                return False

            audio_path = self.output_dir / "narration.mp3"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            tts_engine = os.getenv("TTS_ENGINE", "edge").strip().lower()
            tts_voice = (
                os.getenv("EDGE_TTS_VOICE") if tts_engine == "edge"
                else os.getenv("GOOGLE_TTS_VOICE")
            )
            audio_path, audio_created = self.voice_generator.generate(
                narration,
                output_path=audio_path,
                engine=tts_engine,
                voice=tts_voice,
            )
            if not audio_created:
                log.error("Voice generation failed")
                return False

            audio_report = self.quality_checker.check_audio(audio_path, narration)
            log.info("Audio quality score: %.2f", audio_report.overall_score)
            if not audio_report.is_acceptable:
                log.error("Audio did not pass the quality gate: %s", audio_report.issues)
                return False

            title = topic[:100].strip()
            video_path = self.output_dir / "reel.mp4"
            assemble_reel(
                audio_path=audio_path,
                narration=narration,
                title=title,
                output_path=video_path,
            )
            video_url = host_video_on_github(video_path)
            description = f"{narration[:200].strip()}...\n\n#ريلز #محتوى_عربي"
            published = self.publisher.publish_to_buffer(
                video_url=video_url,
                title=title,
                description=description,
                channel_ids=channel_ids,
            )
            if not published:
                log.error("Buffer publishing failed for one or more channels")
                return False

            log.info("Reel queued successfully for %d Buffer channel(s)", len(channel_ids))
            return True
        except Exception as exc:
            log.error("Pipeline failed: %s", exc, exc_info=True)
            return False


def main() -> int:
    topic = os.getenv("TOPIC") or (sys.argv[1] if len(sys.argv) > 1 else None)
    category = os.getenv("CATEGORY", "عام")
    return 0 if AutoPublishPipeline().run(category=category, topic=topic) else 1


if __name__ == "__main__":
    sys.exit(main())
