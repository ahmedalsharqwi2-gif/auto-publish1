"""Optional MoneyPrinterTurbo renderer for the science short-video pipeline.

The adapter deliberately receives already-approved Arabic narration, audio, and a
locally rendered media track. MoneyPrinterTurbo is not allowed to regenerate the
script, TTS, or publish the result; the existing pipeline remains the owner of
Arabic quality gates and external publishing.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from scripts.pexels_video import build_pexels_track


class MoneyPrinterTurboError(RuntimeError):
    """Raised when the optional MoneyPrinterTurbo render cannot complete."""


def _probe_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise MoneyPrinterTurboError("Audio duration must be positive")
    return duration


def _command_prefix() -> list[str]:
    raw = os.getenv("MPT_PYTHON", "uv run python").strip()
    prefix = shlex.split(raw)
    if not prefix:
        raise MoneyPrinterTurboError("MPT_PYTHON is empty")
    return prefix


def _overlay_arabic_subtitles(video_path: Path, narration: str, duration: float) -> None:
    """Overlay the repository's Arabic ASS captions after MPT finishes."""
    from scripts.assemble_video import _filter_path, write_ass_subtitles

    ass_path = video_path.with_suffix(".mpt.ass")
    captioned_path = video_path.with_suffix(".mpt-captioned.mp4")
    write_ass_subtitles(narration, duration, ass_path)
    try:
        subtitles = _filter_path(ass_path)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                "-vf", f"subtitles='{subtitles}':fontsdir='/usr/share/fonts/truetype/dejavu'",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                str(captioned_path),
            ],
            check=True,
        )
        captioned_path.replace(video_path)
    finally:
        ass_path.unlink(missing_ok=True)
        captioned_path.unlink(missing_ok=True)


def render_with_moneyprinterturbo(
    audio_path: Path,
    narration: str,
    output_path: Path,
    topic: str = "",
) -> Path:
    """Render one vertical MP4 with MPT using current pipeline artifacts."""
    root_value = os.getenv("MONEYPRINTERTURBO_ROOT", "").strip()
    if not root_value:
        raise MoneyPrinterTurboError(
            "MONEYPRINTERTURBO_ROOT is required when VIDEO_RENDERER=mpt"
        )
    mpt_root = Path(root_value).expanduser().resolve()
    cli_path = mpt_root / "cli.py"
    if not cli_path.is_file():
        raise MoneyPrinterTurboError(f"MoneyPrinterTurbo CLI not found: {cli_path}")
    if not audio_path.is_file() or not narration.strip():
        raise MoneyPrinterTurboError("MPT requires a valid audio file and narration")

    duration = _probe_duration(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_track = output_path.with_suffix(".mpt-source.mp4")
    task_id = str(uuid.uuid4())
    try:
        has_track = build_pexels_track(
            os.getenv("PEXELS_API_KEY", "").strip(),
            topic,
            duration,
            source_track,
        )
        if not has_track:
            raise MoneyPrinterTurboError("No Pexels track was available for MPT")

        command = [
            *_command_prefix(),
            str(cli_path),
            "--video-script", narration,
            "--video-source", "local",
            "--video-materials", str(source_track.resolve()),
            "--custom-audio-file", str(audio_path.resolve()),
            "--voice-name", "no-voice",
            "--video-aspect", "9:16",
            "--video-fit-mode", "cover",
            "--video-concat-mode", "sequential",
            "--no-subtitle-enabled",
            "--bgm-type", "none",
            "--task-id", task_id,
            "--stop-at", "video",
        ]
        result = subprocess.run(
            command,
            cwd=mpt_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=int(os.getenv("MPT_RENDER_TIMEOUT", "900")),
        )
        if result.returncode != 0:
            tail = (result.stdout + "\n" + result.stderr).splitlines()[-40:]
            raise MoneyPrinterTurboError(
                "MoneyPrinterTurbo failed (exit %d): %s"
                % (result.returncode, "\n".join(tail))
            )
        task_dir = mpt_root / "storage" / "tasks" / task_id
        videos = sorted(
            path for path in task_dir.glob("final-*.mp4")
            if path.is_file() and path.stat().st_size > 0
        )
        if not videos:
            raise MoneyPrinterTurboError(
                f"MoneyPrinterTurbo completed without final MP4: {task_dir}"
            )
        shutil.copy2(videos[0], output_path)
        _overlay_arabic_subtitles(output_path, narration, duration)
        return output_path
    finally:
        source_track.unlink(missing_ok=True)
