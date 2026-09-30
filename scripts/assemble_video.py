"""Build a publishable vertical MP4 from narration audio and Arabic text."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
WORDS_PER_CAPTION_CHUNK = 6
FONT_SIZE = 58


def probe_duration(audio_path: Path) -> float:
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
        raise ValueError("Audio duration must be positive")
    return duration


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _caption_text(words: list[str]) -> str:
    words = [re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", w).strip() for w in words]
    words = [w for w in words if w]
    if len(words) <= 3:
        return r"\N".join(["\u200f" + " ".join(words)])
    midpoint = (len(words) + 1) // 2
    return "\u200f" + " ".join(words[:midpoint]) + r"\N" + "\u200f" + " ".join(words[midpoint:])


def write_ass_subtitles(text: str, duration: float, ass_path: Path) -> None:
    words = re.findall(r"[\u0621-\u064A\u0671-\u06FF\w]+[^\s]*", text)
    words = [word for word in words if word.strip()]
    if not words:
        raise ValueError("Narration contains no words for subtitles")
    chunks = [words[i:i + WORDS_PER_CAPTION_CHUNK] for i in range(0, len(words), WORDS_PER_CAPTION_CHUNK)]
    total_words = len(words)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {VIDEO_WIDTH}",
        f"PlayResY: {VIDEO_HEIGHT}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,DejaVu Sans,{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H0010182B,&HAA000000,1,0,0,0,100,100,0,0,1,3,1,2,70,70,150,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    cursor = 0.0
    for chunk in chunks:
        start = cursor
        end = min(duration, duration * (cursor + len(chunk) / total_words))
        cursor = end
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(max(end, start + 0.25))},Caption,,0,0,0,,{_ass_escape(_caption_text(chunk))}"
        )
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def assemble_video(audio_path: Path, narration: str, output_path: Path) -> Path:
    """Create a 9:16 MP4 with a dark animated waveform and Arabic captions."""
    duration = probe_duration(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path = output_path.with_suffix(".ass")
    write_ass_subtitles(narration, duration, ass_path)
    subtitles = _filter_path(ass_path)
    filter_complex = (
        f"[1:a]showwaves=s=900x240:mode=cline:colors=0x38bdf8@0.9:rate={FPS},format=rgba[wave];"
        f"[0:v][wave]overlay=90:220:format=auto,subtitles='{subtitles}':fontsdir='/usr/share/fonts/truetype/dejavu'[v]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x0b1220:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:r={FPS}:d={duration:.3f}",
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a:0",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-shortest", "-movflags", "+faststart", str(output_path),
        ],
        check=True,
    )
    ass_path.unlink(missing_ok=True)
    return output_path
