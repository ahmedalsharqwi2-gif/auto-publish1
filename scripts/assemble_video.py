"""Build a publishable vertical MP4 with Arabic captions timed to the real audio."""
from __future__ import annotations

import difflib
import logging
import os
import re
import subprocess
from pathlib import Path

from scripts.pexels_video import build_pexels_track

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
WORDS_PER_CAPTION_CHUNK = 4
FONT_SIZE = 58
# 9:16 render; keep captions below phone camera notches and platform chrome.
CAPTION_TOP_SAFE_MARGIN = 260
log = logging.getLogger(__name__)
SFX_DIR = Path(__file__).resolve().parent.parent / "assets" / "sfx"
SCIENCE_AMBIENCE_GAIN = 0.055
SCIENCE_EVENT_GAIN = 0.18
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]")
PUNCTUATION = str.maketrans(".,،؛:!?؟…-—_()[]{}\"«»/\\", " " * 23)


def probe_duration(audio_path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)], capture_output=True, text=True, check=True)
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
    # Keep ASS control sequences such as \N intact; escaping them as literal
    # backslashes prevents the intended line break from rendering.
    text = re.sub(r"\\(?!N)", r"\\\\", text)
    return text.replace("{", r"\{").replace("}", r"\}")


def _display_word(word: str) -> str:
    return ARABIC_DIACRITICS.sub("", word).translate(PUNCTUATION).strip()


def _caption_text(words: list[str]) -> str:
    words = [_display_word(w) for w in words]
    words = [w for w in words if w]
    if len(words) <= 2:
        return "\u200f" + " ".join(words)
    midpoint = (len(words) + 1) // 2
    return "\u200f" + " ".join(words[:midpoint]) + r"\N" + "\u200f" + " ".join(words[midpoint:])


def _norm(word: str) -> str:
    return _display_word(word).lower()


def _script_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\S+", text) if _display_word(w)]


def align_words_with_whisper(audio_path: Path, script_words: list[str]) -> list[dict]:
    """Return monotonically increasing timestamps from the final audio."""
    from faster_whisper import WhisperModel
    model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")
    result = model.transcribe(str(audio_path), language="ar", word_timestamps=True, vad_filter=False)
    segments = result[0] if isinstance(result, (tuple, list)) else result
    heard: list[tuple[str, float, float]] = []
    for segment in segments:
        for word in (getattr(segment, "words", None) or []):
            text = (getattr(word, "word", "") or "").strip()
            if text and word.start is not None and word.end is not None:
                heard.append((text, max(0.0, float(word.start)), max(0.0, float(word.end))))
    if not heard:
        raise RuntimeError("Whisper did not return Arabic word timestamps")
    expected = [_norm(w) for w in script_words]
    actual = [_norm(w) for w, _, _ in heard]
    matcher = difflib.SequenceMatcher(None, expected, actual, autojunk=False)
    timings: list[dict | None] = [None] * len(script_words)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            si, ai = block.a + offset, block.b + offset
            timings[si] = {"text": script_words[si], "offset": heard[ai][1], "duration": max(heard[ai][2] - heard[ai][1], 0.06)}
    known = [i for i, item in enumerate(timings) if item is not None]
    if len(known) < max(1, int(len(script_words) * 0.50)):
        raise RuntimeError(f"Whisper alignment too weak: {len(known)}/{len(script_words)} words")
    for i, item in enumerate(timings):
        if item is not None:
            continue
        prev_i = max((j for j in known if j < i), default=None)
        next_i = min((j for j in known if j > i), default=None)
        if prev_i is None:
            next_item = timings[next_i]
            step = max(next_item["offset"] / (next_i + 1), 0.08)
            start, duration = step * i, max(step * 0.85, 0.06)
        elif next_i is None:
            prev_item = timings[prev_i]
            step = max(prev_item["duration"], 0.08)
            start, duration = prev_item["offset"] + step * (i - prev_i), step * 0.85
        else:
            prev_item, next_item = timings[prev_i], timings[next_i]
            left = prev_item["offset"] + prev_item["duration"]
            gap = max(next_item["offset"] - left, 0.08)
            slot = gap / (next_i - prev_i)
            start, duration = left + slot * (i - prev_i - 1), max(slot * 0.85, 0.06)
        timings[i] = {"text": script_words[i], "offset": start, "duration": duration}
    output = [item for item in timings if item is not None]
    previous_end = 0.0
    for item in output:
        item["offset"] = max(float(item["offset"]), previous_end)
        item["duration"] = max(float(item["duration"]), 0.06)
        previous_end = item["offset"] + item["duration"]
    return output


def _ass_header() -> str:
    return ("[Script Info]\nScriptType: v4.00+\n" f"PlayResX: {VIDEO_WIDTH}\nPlayResY: {VIDEO_HEIGHT}\n" "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n" "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n" f"Style: Caption,Noto Sans Arabic,{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H0010182B,&HAA000000,1,0,0,0,100,100,0,0,1,3,1,8,70,70,{CAPTION_TOP_SAFE_MARGIN},1\n\n" "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def write_ass_subtitles(text: str, duration: float, ass_path: Path, audio_path: Path | None = None) -> None:
    words = _script_words(text)
    if not words:
        raise ValueError("Narration contains no words for subtitles")
    if audio_path is not None:
        try:
            events = align_words_with_whisper(audio_path, words)
        except Exception as exc:
            log.warning(
                "Whisper alignment unavailable (%s); using uniform subtitle timing",
                exc,
            )
            per_word = duration / len(words)
            events = [
                {"text": word, "offset": index * per_word, "duration": per_word}
                for index, word in enumerate(words)
            ]
    else:
        per_word = duration / len(words)
        events = [{"text": w, "offset": i * per_word, "duration": per_word} for i, w in enumerate(words)]
    lines = [_ass_header()]
    for index in range(0, len(events), WORDS_PER_CAPTION_CHUNK):
        group = events[index:index + WORDS_PER_CAPTION_CHUNK]
        start = max(0.0, group[0]["offset"])
        end = min(duration, group[-1]["offset"] + group[-1]["duration"])
        if end <= start:
            end = min(duration, start + 0.25)
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{_ass_escape(_caption_text([e['text'] for e in group]))}")
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def _mix_science_audio(voice_path: Path, duration: float, output_path: Path) -> Path:
    """Mix quiet scientific ambience and timed transition events under narration."""
    ambience = SFX_DIR / "science_lab_ambience_loop.mp3"
    sting = SFX_DIR / "science_discovery_sting.mp3"
    whoosh = SFX_DIR / "science_transition_whoosh.mp3"
    required = [ambience, sting, whoosh]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Science SFX missing: " + ", ".join(missing))
    points = [0.20, max(0.35, duration * 0.34), max(0.50, duration * 0.67)]
    inputs = ["-i", str(voice_path), "-stream_loop", "-1", "-i", str(ambience)]
    for _ in points:
        inputs += ["-i", str(whoosh)]
    inputs += ["-i", str(sting)]
    filters = [f"[0:a]aresample=48000,volume=1.0[voice]",
               f"[1:a]aresample=48000,volume={SCIENCE_AMBIENCE_GAIN},atrim=duration={duration:.3f}[amb]"]
    event_labels = []
    for idx, point in enumerate(points):
        label = f"w{idx}"
        input_idx = 2 + idx
        filters.append(f"[{input_idx}:a]aresample=48000,volume={SCIENCE_EVENT_GAIN},adelay={int(point*1000)}|{int(point*1000)},atrim=duration={duration:.3f}[{label}]")
        event_labels.append(f"[{label}]")
    sting_idx = 2 + len(points)
    filters.append(f"[{sting_idx}:a]aresample=48000,volume={SCIENCE_EVENT_GAIN},adelay=180|180,atrim=duration={duration:.3f}[sting]")
    mix_inputs = "[voice][amb]" + "".join(event_labels) + "[sting]"
    filters.append(f"{mix_inputs}amix=inputs={2+len(points)+1}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95:level=disabled[a]")
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[a]", "-t", f"{duration:.3f}", "-c:a", "libmp3lame", "-b:a", "192k", str(output_path)], check=True)
    return output_path


def assemble_video(audio_path: Path, narration: str, output_path: Path, topic: str = "") -> Path:
    duration = probe_duration(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path = output_path.with_suffix(".ass")
    write_ass_subtitles(narration, duration, ass_path, audio_path=audio_path)
    subtitles = _filter_path(ass_path)
    pexels_track = output_path.with_suffix(".pexels.mp4")
    mixed_audio = output_path.with_suffix(".mixed.mp3")
    _mix_science_audio(audio_path, duration, mixed_audio)
    has_pexels = build_pexels_track(os.getenv("PEXELS_API_KEY", "").strip(), topic, duration, pexels_track)
    if not has_pexels:
        ass_path.unlink(missing_ok=True)
        raise RuntimeError("لم تتوفر مقاطع Pexels كافية ومرتبطة بالموضوع؛ أوقفنا النشر.")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(pexels_track), "-i", str(mixed_audio), "-filter_complex", f"[0:v]subtitles='{subtitles}':fontsdir='/usr/share/fonts/truetype/dejavu'[v]", "-map", "[v]", "-map", "1:a:0", "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(output_path)], check=True)
    finally:
        ass_path.unlink(missing_ok=True)
        pexels_track.unlink(missing_ok=True)
        mixed_audio.unlink(missing_ok=True)
    return output_path
