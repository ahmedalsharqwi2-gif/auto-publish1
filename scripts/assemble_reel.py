"""Deterministic 9:16 reel rendering from Arabic narration and an audio file."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("pipeline")
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
FONT_NAME = "Noto Sans Arabic"


class ReelAssemblyError(RuntimeError):
    """Raised when an input is invalid or FFmpeg cannot render a reel."""


def _escape_ass(text: str) -> str:
    # ASS uses braces for formatting overrides. Remove them from generated copy
    # rather than allowing narration to inject subtitle styling directives.
    return text.replace("{", "").replace("}", "").replace("\r", " ").replace("\n", " ").strip()


def _caption_chunks(text: str, max_words: int = 8, max_chars: int = 38) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and (len(current) >= max_words or len(candidate) > max_chars):
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _wrap_caption(text: str, max_chars: int = 30) -> str:
    lines: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append("\u200f" + _escape_ass(" ".join(current)))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append("\u200f" + _escape_ass(" ".join(current)))
    return r"\N".join(lines)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def build_ass_subtitles(title: str, narration: str, duration: float) -> str:
    """Build an ASS subtitle track with a persistent title and timed captions."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    chunks = _caption_chunks(narration)
    if not chunks:
        raise ValueError("narration must contain at least one word")

    weights = [max(1, len(chunk.split())) for chunk in chunks]
    total_weight = sum(weights)
    cursor = 0.0
    events = [
        f"Dialogue: 1,{_ass_time(0)},{_ass_time(duration)},Title,,0,0,0,,{_wrap_caption(title, 28)}"
    ]
    for chunk, weight in zip(chunks, weights):
        start = cursor
        cursor = min(duration, cursor + duration * weight / total_weight)
        if cursor <= start:
            continue
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(cursor)},Caption,,0,0,0,,{_wrap_caption(chunk)}"
        )

    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {VIDEO_WIDTH}",
            f"PlayResY: {VIDEO_HEIGHT}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Title,{FONT_NAME},58,&H00FFFFFF,&H00FFFFFF,&H00101828,&H80000000,1,0,0,0,100,100,0,0,1,3,1,8,72,72,130,1",
            f"Style: Caption,{FONT_NAME},66,&H00FFFFFF,&H00FFFFFF,&H00101828,&H80000000,1,0,0,0,100,100,0,0,1,4,1,2,80,80,300,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ]
    )


def probe_media(path: Path) -> dict:
    """Return ffprobe's JSON metadata for a media file."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ReelAssemblyError("ffprobe is required to verify the reel")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ReelAssemblyError(f"ffprobe failed: {result.stderr[-1200:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReelAssemblyError("ffprobe returned invalid JSON") from exc


def _filter_path(path: Path) -> str:
    # Escape characters that have a special meaning in FFmpeg filter syntax.
    value = str(path.resolve()).replace("\\", "/")
    value = value.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return f"subtitles='{value}'"


def assemble_reel(
    audio_path: Path,
    narration: str,
    title: str,
    output_path: Path,
    max_duration: float | None = None,
) -> Path:
    """Render an H.264/AAC 1080x1920 reel with Arabic captions and narration."""
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise ReelAssemblyError(f"Narration audio is missing or empty: {audio_path}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ReelAssemblyError("ffmpeg is required to assemble reels")

    media = probe_media(audio_path)
    audio_streams = [s for s in media.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise ReelAssemblyError(f"Narration has no audio stream: {audio_path}")
    try:
        duration = float(media.get("format", {}).get("duration") or audio_streams[0]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReelAssemblyError("Could not determine narration duration") from exc
    if duration <= 0:
        raise ReelAssemblyError("Narration duration must be positive")

    if max_duration is None:
        max_duration = float(os.getenv("MAX_REEL_SECONDS", "90"))
    if duration > max_duration:
        raise ReelAssemblyError(
            f"Narration is {duration:.1f}s; maximum reel duration is {max_duration:.1f}s"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = output_path.with_suffix(".ass")
    subtitle_path.write_text(build_ass_subtitles(title, narration, duration), encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x0b1b2a:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:r={VIDEO_FPS}",
        "-i",
        str(audio_path),
        "-vf",
        _filter_path(subtitle_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{duration:.3f}",
        "-r",
        str(VIDEO_FPS),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={_escape_ass(title)}",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ReelAssemblyError(f"ffmpeg failed: {result.stderr[-2000:]}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ReelAssemblyError("ffmpeg produced an empty reel")

    rendered = probe_media(output_path)
    streams = rendered.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    rendered_audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video_stream or not rendered_audio:
        raise ReelAssemblyError("Rendered reel is missing a video or audio stream")
    if int(video_stream.get("width", 0)) != VIDEO_WIDTH or int(video_stream.get("height", 0)) != VIDEO_HEIGHT:
        raise ReelAssemblyError("Rendered reel dimensions are not 1080x1920")
    log.info("Assembled reel: %s (%.1fs, 9:16)", output_path, duration)
    return output_path
