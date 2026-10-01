"""Permanent per-repository topic history and duplicate guard.

History entries are plain data only. They are never evaluated or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_WORDS = re.compile(r"[a-z0-9\u0621-\u064a\u0660-\u0669]+", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "كيف", "ماذا", "لماذا", "هل", "ما", "هو", "هي", "كان", "كانت", "بين",
    "بعد", "قبل", "عند", "عندما", "حيث", "الذي", "التي", "الذين", "ثم", "قد",
    "قصة", "حكاية", "حقيقة", "سر", "اسرار", "أسرار", "لغز", "العظيم", "مذهل",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "how", "why",
}
_TRANSLATE = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ئ": "ي", "ؤ": "و",
    "ة": "ه", "ـ": "",
})


class DuplicateTopicError(ValueError):
    """Raised when a proposed topic overlaps an already-recorded topic."""


class TopicHistoryError(RuntimeError):
    """Raised when history cannot be safely read or durably recorded."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean_text(value: Any, limit: int = 500) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _CONTROL.sub(" ", text)
    return " ".join(text.split())[:limit].strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value, 2000)).casefold()
    text = _DIACRITICS.sub("", text).translate(_TRANSLATE)
    tokens = [token for token in _WORDS.findall(text) if token not in _STOPWORDS]
    return " ".join(tokens)


def _is_similar(left: Any, right: Any) -> bool:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 12:
        return False
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    if ratio >= 0.86:
        return True
    words_a, words_b = a.split(), b.split()
    ta, tb = set(words_a), set(words_b)
    bigrams_a = set(zip(words_a, words_a[1:]))
    bigrams_b = set(zip(words_b, words_b[1:]))
    if bigrams_a & bigrams_b:
        return True
    overlap = len(ta & tb)
    if overlap < 2:
        return False
    containment = overlap / min(len(ta), len(tb))
    jaccard = overlap / len(ta | tb)
    if min(len(ta), len(tb)) <= 4 and containment >= 0.75:
        return True
    return containment >= 0.80 or jaccard >= 0.66


def _topic_fields(value: Any) -> list[str]:
    if isinstance(value, str):
        fields = [clean_text(value)]
    elif isinstance(value, dict):
        fields = [
            clean_text(value.get(key))
            for key in ("title", "topic", "subject", "hook", "hook_text", "summary", "premise", "caption")
        ]
    else:
        fields = []
    result: list[str] = []
    seen: set[str] = set()
    for field in fields:
        normalized = normalize_text(field)
        if normalized and normalized not in seen:
            result.append(field)
            seen.add(normalized)
    return result


def find_duplicate(candidate: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    proposed = _topic_fields(candidate)
    for entry in entries:
        previous = _topic_fields(entry)
        if any(_is_similar(new, old) for new in proposed for old in previous):
            return entry
    return None


def prompt_topics(entries: list[dict[str, Any]], limit: int = 100) -> str:
    """Serialize only short title/hook labels as JSON data for a model prompt."""
    items = []
    for entry in entries[-limit:]:
        title = clean_text(entry.get("title") or entry.get("topic") or entry.get("subject"), 160)
        hook = clean_text(entry.get("hook") or entry.get("summary"), 180)
        label = title if not hook else f"{title} — {hook}"
        label = re.sub(r"https?://\S+", "[رابط محذوف]", label, flags=re.IGNORECASE)
        label = re.sub(r"[`<>]", " ", label)
        label = " ".join(label.split())[:300]
        if label:
            items.append(label)
    return json.dumps(items, ensure_ascii=False)


def _entry_id(candidate: Any) -> str:
    title = ""
    hook = ""
    if isinstance(candidate, dict):
        title = clean_text(candidate.get("title") or candidate.get("topic") or candidate.get("subject"))
        hook = clean_text(candidate.get("hook") or candidate.get("hook_text") or candidate.get("summary") or candidate.get("premise"))
    else:
        title = clean_text(candidate)
    fingerprint = f"{normalize_text(title)}|{normalize_text(hook)}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]


def _coerce_entry(item: Any, source: str = "legacy") -> dict[str, Any] | None:
    if isinstance(item, str):
        item = {"title": item}
    if not isinstance(item, dict):
        return None
    subject = clean_text(item.get("subject") or item.get("topic"), 300)
    title = clean_text(item.get("title") or subject, 300)
    hook = clean_text(
        item.get("hook") or item.get("hook_text") or item.get("summary") or item.get("premise") or item.get("caption"),
        700,
    )
    if subject and normalize_text(subject) != normalize_text(title):
        hook = clean_text(f"{hook} — {subject}", 700) if hook else subject
    if not title and hook:
        title = hook[:160]
    if not title:
        return None
    entry = {
        "id": str(item.get("id") or _entry_id({"title": title, "hook": hook})),
        "title": title,
        "hook": hook,
        "region": clean_text(item.get("region"), 200),
        "status": clean_text(item.get("status") or "historical", 40),
        "source": clean_text(item.get("source") or source, 80),
    }
    for key in ("reserved_at", "published_at", "run_id", "story_type", "reservation_id", "repository"):
        value = item.get(key)
        if value:
            entry[key] = clean_text(value, 100)
    return entry


def load_history(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    raw = file_path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TopicHistoryError(f"سجل المواضيع غير صالح JSON؛ أُوقف النشر حفاظًا على منع التكرار: {file_path}") from exc
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("entries", data.get("topics", data.get("history", [])))
        if isinstance(items, dict):
            items = list(items.values())
    else:
        raise TopicHistoryError(f"بنية سجل المواضيع غير معروفة؛ أُوقف النشر: {file_path}")
    if not isinstance(items, list):
        raise TopicHistoryError(f"قائمة السجل غير صالحة؛ أُوقف النشر: {file_path}")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in items:
        entry = _coerce_entry(item)
        record_key = (entry or {}).get("reservation_id") or (entry or {}).get("id")
        if entry and record_key not in seen_ids:
            result.append(entry)
            seen_ids.add(record_key)
    return result


def write_history(path: str | Path, entries: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "entries": entries}
    temp = file_path.with_suffix(file_path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, file_path)


class TopicHistory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries = load_history(self.path)

    def check_unique(self, candidate: Any) -> None:
        duplicate = find_duplicate(candidate, self.entries)
        if duplicate:
            raise DuplicateTopicError(
                "الموضوع مكرر أو قريب جدًا من موضوع سابق في هذا المستودع "
                f"(record id: {duplicate.get('id', 'unknown')}); رُفض قبل النشر."
            )

    def reserve(self, candidate: Any, source: str = "workflow", commit: bool = False) -> dict[str, Any]:
        entry = _coerce_entry(candidate, source=source)
        if not entry:
            raise TopicHistoryError("تعذر تسجيل الموضوع: لا يوجد عنوان/موضوع صالح.")
        self.check_unique(entry)
        entry["id"] = _entry_id(entry)
        entry["reservation_id"] = uuid.uuid4().hex
        entry["status"] = "reserved"
        entry["reserved_at"] = _now()
        if os.getenv("GITHUB_RUN_ID"):
            entry["run_id"] = clean_text(os.getenv("GITHUB_RUN_ID"), 40)
        if os.getenv("GITHUB_REPOSITORY"):
            entry["repository"] = clean_text(os.getenv("GITHUB_REPOSITORY"), 150)
        self.entries.append(entry)
        write_history(self.path, self.entries)
        if commit:
            self._commit_and_push("reserve", entry)
        return entry

    def mark_published(self, candidate: Any, commit: bool = False) -> None:
        target_id = _entry_id(candidate)
        entry = next((item for item in self.entries if item.get("id") == target_id), None)
        if entry is None:
            raise TopicHistoryError("لا يوجد حجز مطابق للموضوع؛ لم يُحدّث سجل النشر.")
        if entry.get("status") == "published":
            return
        entry["status"] = "published"
        entry["published_at"] = _now()
        write_history(self.path, self.entries)
        if commit:
            self._commit_and_push("published", entry)

    def _commit_and_push(self, action: str, candidate: dict[str, Any]) -> None:
        if os.getenv("GITHUB_ACTIONS", "").lower() != "true":
            return
        ref = os.getenv("GITHUB_REF", "")
        if ref not in {"refs/heads/main", "refs/heads/master"}:
            raise TopicHistoryError("سجل المواضيع لا يُحفظ إلا من الفرع الرئيسي؛ رُفض النشر.")
        file_arg = str(self.path)
        def git(*args: str, check: bool = True):
            return subprocess.run(["git", *args], text=True, capture_output=True, check=check)
        git("config", "user.name", "topic-history-bot")
        git("config", "user.email", "topic-history-bot@users.noreply.github.com")
        git("add", "--", file_arg)
        staged = git("diff", "--cached", "--quiet", check=False)
        if staged.returncode == 0:
            return
        git("commit", "-m", f"chore: {action} topic history [skip ci]")
        pushed = git("push", "origin", "HEAD:main", check=False)
        if pushed.returncode == 0:
            return
        fetched = git("fetch", "origin", "main", check=False)
        if fetched.returncode != 0:
            raise TopicHistoryError("تعذر تحديث سجل المواضيع البعيد؛ أُوقف النشر.")
        rebased = git("rebase", "FETCH_HEAD", check=False)
        if rebased.returncode != 0:
            git("rebase", "--abort", check=False)
            raise TopicHistoryError("تعارض تحديث سجل المواضيع؛ أُوقف النشر بدل المخاطرة بالتكرار.")
        own_reservation_id = candidate.get("reservation_id")
        remote_entries = [
            entry for entry in load_history(self.path)
            if not own_reservation_id or entry.get("reservation_id") != own_reservation_id
        ]
        if find_duplicate(candidate, remote_entries):
            raise TopicHistoryError("حجز تشغيل متزامن موضوعًا مشابهًا أولًا؛ أُوقف هذا النشر.")
        pushed = git("push", "origin", "HEAD:main", check=False)
        if pushed.returncode != 0:
            raise TopicHistoryError("تعذر حفظ سجل المواضيع بعد إعادة المحاولة؛ أُوقف النشر.")


def _load_episode(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicHistoryError(f"تعذر قراءة ملف الحلقة: {path}") from exc
    if not isinstance(data, dict):
        raise TopicHistoryError("ملف الحلقة يجب أن يحتوي كائن JSON.")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reserve or confirm a per-repository topic.")
    parser.add_argument("command", choices=("reserve", "mark-published"))
    parser.add_argument("--file", required=True, help="Path to the permanent topic history JSON")
    parser.add_argument("--episode", required=True, help="Episode/topic JSON file")
    args = parser.parse_args(argv)
    try:
        history = TopicHistory(args.file)
        episode = _load_episode(args.episode)
        commit = os.getenv("TOPIC_HISTORY_COMMIT", "false").lower() == "true"
        if args.command == "reserve":
            entry = history.reserve(episode, commit=commit)
            print(f"TOPIC_RESERVED id={entry['id']} records={len(history.entries)}")
        else:
            history.mark_published(episode, commit=commit)
            print(f"TOPIC_MARKED_PUBLISHED records={len(history.entries)}")
        return 0
    except (DuplicateTopicError, TopicHistoryError) as exc:
        print(f"TOPIC_GATE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
