from __future__ import annotations
import json
from pathlib import Path
from .normalize import normalize_arabic


def load_lexicon(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pronunciation lexicon must be a JSON object")
    result = {}
    for surface, value in data.items():
        if isinstance(value, dict):
            value = value.get("spoken", value.get("pronunciation", ""))
        if str(surface).strip() and str(value).strip():
            result[normalize_arabic(surface)] = normalize_arabic(value)
    return dict(sorted(result.items(), key=lambda item: len(item[0]), reverse=True))


def apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    result = normalize_arabic(text)
    for surface, spoken in sorted(lexicon.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(surface, spoken)
    return result
