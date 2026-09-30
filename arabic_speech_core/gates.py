from __future__ import annotations
import asyncio, os, subprocess, tempfile
from pathlib import Path
from .normalize import prepare_text, arabic_ratio
from .lexicon import load_lexicon, apply_lexicon

class GateError(RuntimeError): pass

def text_gate(text: str, *, min_words=1, max_words=2500, min_arabic_ratio=.55, lexicon_path=None) -> dict:
    prepared = prepare_text(text)
    count = len(prepared.display_words)
    ratio = arabic_ratio(prepared.display_text)
    if count < min_words: raise GateError(f"TEXT_GATE: too few words ({count})")
    if count > max_words: raise GateError(f"TEXT_GATE: too many words ({count})")
    if ratio < min_arabic_ratio: raise GateError(f"TEXT_GATE: Arabic ratio {ratio:.1%} < {min_arabic_ratio:.1%}")
    spoken = apply_lexicon(prepared.display_text, load_lexicon(lexicon_path))
    return {"display_text": prepared.display_text, "align_text": prepared.align_text, "spoken_text": spoken, "words": count, "arabic_ratio": ratio}

def provider_preflight(*, engine="edge", locale="ar-SA", voice="ar-SA-HamedNeural", max_chars=4500) -> dict:
    if not locale.startswith("ar-"): raise GateError(f"PROVIDER_PREFLIGHT: unsupported locale {locale}")
    if not voice or not voice.startswith("ar-"): raise GateError(f"PROVIDER_PREFLIGHT: invalid Arabic voice {voice}")
    if engine not in {"edge", "silma", "google", "azure", "fake"}: raise GateError(f"PROVIDER_PREFLIGHT: unsupported engine {engine}")
    if engine in {"google", "azure"} and not (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("AZURE_SPEECH_KEY")):
        raise GateError(f"PROVIDER_PREFLIGHT: credentials missing for {engine}")
    return {"engine": engine, "locale": locale, "voice": voice, "max_chars": max_chars}

async def _edge(text: str, path: Path, voice: str, rate: str = "+0%"):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch="+0Hz").save(str(path))

def tts_smoke_test(text: str, *, engine="edge", voice="ar-SA-HamedNeural") -> dict:
    if engine == "fake": return {"ok": True, "bytes": 0, "engine": engine}
    if engine != "edge":
        # Full engines are intentionally not invoked by this cheap gate.
        return {"ok": True, "deferred": True, "engine": engine}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "smoke.mp3"
        try: asyncio.run(_edge(text, out, voice))
        except Exception as exc: raise GateError(f"TTS_SMOKE: {exc}") from exc
        if not out.exists() or out.stat().st_size < 512: raise GateError("TTS_SMOKE: empty audio")
        probe = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(out)], capture_output=True, text=True)
        if probe.returncode or not probe.stdout.strip() or float(probe.stdout.strip()) <= 0: raise GateError("TTS_SMOKE: undecodable audio")
        return {"ok": True, "bytes": out.stat().st_size, "duration": float(probe.stdout), "engine": engine}
