#!/usr/bin/env python3
"""Generate and validate the short SILMA reference voice asset."""
from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path

REFERENCE_TEXT = "فِي صَبَاحٍ هَادِئٍ، تَكْشِفُ وَثِيقَةٌ قَدِيمَةٌ كَيْفَ غَيَّرَ قَرَارٌ صَغِيرٌ مَسَارَ التَّارِيخِ."


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


async def synthesize(text: str, voice: str, rate: str, output: Path) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    with output.open("wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])


def validate_with_whisper(wav: Path, expected: str) -> None:
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), language="ar", vad_filter=False)
    heard = " ".join((s.text or "") for s in segments)
    def canon(s: str) -> list[str]:
        import re
        s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", s)
        s = re.sub(r"[^\u0621-\u064A\u0671\s]", " ", s)
        return [x for x in s.split() if x]
    exp, got = canon(expected), canon(heard)
    from difflib import SequenceMatcher
    score = SequenceMatcher(None, exp, got, autojunk=False).ratio()
    print(f"Whisper reference check: {score:.2f} | heard={heard}")
    if score < 0.60:
        raise SystemExit("Reference voice rejected: Whisper transcript does not match the source text")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ar-SA-ZariyahNeural")
    ap.add_argument("--rate", default="-3%")
    ap.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mp3 = args.output_dir / "voice_reference_synthetic.mp3"
    wav = args.output_dir / "voice_reference_synthetic.wav"
    txt = args.output_dir / "voice_reference_synthetic.txt"
    asyncio.run(synthesize(REFERENCE_TEXT, args.voice, args.rate, mp3))
    run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-af", "apad=pad_dur=0.5", "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(wav)])
    txt.write_text(REFERENCE_TEXT + "\n", encoding="utf-8")
    validate_with_whisper(wav, REFERENCE_TEXT)
    mp3.unlink(missing_ok=True)
    print(f"Created {wav} and {txt}")


if __name__ == "__main__":
    main()
