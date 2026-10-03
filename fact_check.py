#!/usr/bin/env python3
"""
Source-backed Fact Check using general web sources.
Searches the public web, extracts readable page text, and verifies all claims
against the fetched sources. Wikipedia is intentionally excluded.

Provider: Gemini (primary) -> OpenRouter (optional fallback).
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests

from llm_gemini import GEMINI_API_KEY, gemini_chat

log = logging.getLogger("fact_check")

DEFAULT_CONFIG = Path(os.getenv("FACT_CHECK_SOURCES_FILE", "config/fact_sources.json"))

# LLM provider (Gemini primary, OpenRouter optional fallback)
OPENROUTER_API_KEY = re.sub(r"\s+", "", os.getenv("OPENROUTER_API_KEY", ""))
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "https://github.com/")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Auto Publish Reels")
_FACT_CHECK_MODELS_RAW = os.getenv(
    "FACT_CHECK_MODEL",
    "google/gemma-4-26b-a4b-it:free,"
    "google/gemma-4-31b-it:free,"
    "nvidia/nemotron-3-super-120b-a12b:free",
)
FACT_CHECK_MODELS = [m.strip() for m in _FACT_CHECK_MODELS_RAW.split(",") if m.strip()]

MIN_CONFIDENCE = float(os.getenv("FACT_CHECK_MIN_CONFIDENCE", "0.85"))
FETCH_TIMEOUT = float(os.getenv("FACT_CHECK_FETCH_TIMEOUT", "20"))
MAX_SOURCE_CHARS = int(os.getenv("FACT_CHECK_MAX_SOURCE_CHARS", "6000"))
MAX_TOTAL_SOURCE_CHARS = int(os.getenv("FACT_CHECK_MAX_TOTAL_SOURCE_CHARS", "8000"))
FACT_CHECK_MAX_COMPLETION_TOKENS = max(1024, int(os.getenv("FACT_CHECK_MAX_COMPLETION_TOKENS", "2048")))
REQUIRE_EXTERNAL_SOURCES = os.getenv("REQUIRE_EXTERNAL_SOURCES", "false").lower() == "true"
USER_AGENT = "auto-publish1-fact-check/1.0"
WEB_SEARCH_URL = os.getenv("FACT_CHECK_SEARCH_URL", "https://html.duckduckgo.com/html/")
MAX_SEARCH_RESULTS = max(1, int(os.getenv("FACT_CHECK_MAX_SEARCH_RESULTS", "5")))


class FactCheckError(RuntimeError):
    pass


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden = 0
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._hidden += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags and self._hidden:
            self._hidden -= 1

    def handle_data(self, data):
        if not self._hidden:
            text = html.unescape(data).strip()
            if text:
                self.parts.append(text)


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactCheckError(f"Cannot load Fact Check config: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FactCheckError("Fact Check config must be a JSON object")
    data["minimum_confidence"] = float(data.get("minimum_confidence", MIN_CONFIDENCE))
    return data


def _topic_value(topic: Any, key: str, default: Any = "") -> Any:
    if isinstance(topic, dict):
        return topic.get(key, default)
    return getattr(topic, key, default)


def _json_from_model(text: str) -> dict[str, Any]:
    """Parse the JSON object out of a model reply (tolerates think blocks / fences / extra text)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj
            if "claims" in obj:
                return obj
    if found is not None:
        return found
    raise FactCheckError(f"No JSON in model response: {text[:300]}")


def _search_result_urls(query: str) -> list[str]:
    """Return public web result URLs without restricting sources by domain."""
    response = requests.get(
        WEB_SEARCH_URL,
        params={"q": query, "kl": "wt-wt", "df": "y"},
        timeout=FETCH_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    urls: list[str] = []
    for raw_href in re.findall(
        r'<a[^>]+class=["\'][^"\']*result__a[^"\']*["\'][^>]+href=["\']([^"\']+)',
        response.text,
        flags=re.I,
    ):
        parsed = urlparse(html.unescape(raw_href))
        target = parse_qs(parsed.query).get("uddg", [raw_href])[0]
        url = unquote(html.unescape(target)).strip()
        target_host = urlparse(url).netloc.lower().split(":", 1)[0]
        if urlparse(url).scheme not in {"http", "https"} or not target_host:
            continue
        # Wikipedia is explicitly excluded from automatic source discovery.
        if target_host == "wikipedia.org" or target_host.endswith(".wikipedia.org"):
            continue
        if url not in urls:
            urls.append(url)
        if len(urls) >= MAX_SEARCH_RESULTS:
            break
    return urls


def fetch_web_sources(query: str) -> list[dict[str, str]]:
    """Search and fetch readable pages from any public web domain."""
    try:
        urls = _search_result_urls(query)
    except Exception as exc:
        log.warning("General web search failed for %r: %s", query, exc)
        return []

    sources: list[dict[str, str]] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            parser = _VisibleTextParser()
            parser.feed(response.text)
            text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
            if len(text) < 300:
                log.info("Skipping source with insufficient readable text: %s", url)
                continue
            sources.append({"url": url, "text": text[:MAX_SOURCE_CHARS]})
            log.info("Web source: %s (%d chars)", url, len(text))
        except Exception as exc:
            log.warning("Web source fetch failed for %s: %s", url, exc)
    return sources


def _llm_call(system: str, user: str, max_tokens: int = FACT_CHECK_MAX_COMPLETION_TOKENS) -> dict[str, Any]:
    """Call the LLM (Gemini first, optional OpenRouter fallback) and parse JSON."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    errors: list[str] = []

    # 1) Gemini
    if GEMINI_API_KEY:
        try:
            content = gemini_chat(messages, max_tokens=max_tokens, temperature=0, timeout=120)
            return _json_from_model(content)
        except Exception as exc:
            errors.append(f"gemini: {exc}")
            log.warning("Gemini fact-check call failed: %s", exc)

    # 2) OpenRouter fallback (optional)
    if OPENROUTER_API_KEY:
        for model in FACT_CHECK_MODELS:
            payload = {
                "model": model, "messages": messages,
                "max_tokens": max(max_tokens, 6000), "temperature": 0,
                "reasoning": {"effort": "low", "exclude": True},
            }
            try:
                resp = requests.post(
                    OPENROUTER_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": OPENROUTER_REFERER,
                        "X-Title": OPENROUTER_TITLE,
                    },
                    json=payload, timeout=120,
                )
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                log.warning("OpenRouter fact-check %s error: %s", model, exc)
                continue
            if resp.status_code != 200:
                errors.append(f"{model}: HTTP {resp.status_code}")
                log.warning("OpenRouter fact-check %s HTTP %d: %s", model, resp.status_code, resp.text[:200])
                continue
            content = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if not content.strip():
                errors.append(f"{model}: empty content")
                continue
            try:
                return _json_from_model(content)
            except FactCheckError as exc:
                errors.append(f"{model}: {exc}")
                continue

    raise FactCheckError("All LLM providers failed for Fact Check: " + "; ".join(errors)[:600])


def _extract_claims(script: str) -> list[dict[str, Any]]:
    result = _llm_call(
        """أنت مستخرج ادعاءات علمية فقط. استخرج كل جملة قابلة للتحقق من النص.
أعد كائن JSON فقط بالمفاتيح التالية:
{"claims":[{"claim":"ادعاء","importance":"core","numeric":false}]}
importance: "core" أو "supporting". numeric: true أو false.
لا تعتبر الدعوة للمتابعة أو الأسلوب البلاغي ادعاءً علميًا.""",
        json.dumps({"script": script}, ensure_ascii=False),
        max_tokens=2048,
    )
    claims = result.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise FactCheckError("No claims extracted")
    clean = []
    for i, item in enumerate(claims):
        if not isinstance(item, dict):
            raise FactCheckError(f"Claim {i+1} not an object")
        claim = str(item.get("claim", "")).strip()
        importance = item.get("importance", "core")
        numeric = bool(item.get("numeric", False))
        if not claim:
            continue
        if importance not in {"core", "supporting"}:
            importance = "core"
        clean.append({"claim": claim, "importance": importance, "numeric": numeric})
    if not clean:
        raise FactCheckError("No valid claims")
    return clean


def _judge_claims(script: str, claims: list[dict[str, Any]],
                  sources: list[dict[str, str]]) -> dict[str, Any]:
    evidence = "\n\n".join(f"SOURCE {i+1}: {s['url']}\n{s['text']}" for i, s in enumerate(sources))
    return _llm_call(
        """أنت مدقق علمي صارم. قارن كل ادعاء بالنصوص المصدرية المرفقة فقط.
لا تستخدم معرفتك العامة. صنف كل ادعاء: supported, contradicted, uncertain, unsupported.
supported يتطلب دليلاً واضحًا في المصدر.
أعد كائن JSON فقط:
{"claims":[{"claim":"ادعاء","verdict":"supported","confidence":0.9,"evidence_quote":"اقتباس","source_url":"url","reason":"سبب"}],"overall_reason":"ملخص"}""",
        json.dumps({
            "script": script, "claims": claims,
            "sources": evidence,
        }, ensure_ascii=False),
        max_tokens=FACT_CHECK_MAX_COMPLETION_TOKENS,
    )


def fact_check_topic(topic: Any, output_path: Path | None = None,
                     prefetched_sources: list[dict[str, str]] | None = None,
                     preflight_source_errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Fact-check a topic against general web sources or provided sources."""
    report: dict[str, Any] = {
        "status": "REJECT", "title": str(_topic_value(topic, "title", "")),
        "source_urls": [], "claims": [], "errors": [],
        "source_fetch_errors": [], "sources_fetched": [],
    }
    try:
        config = _load_config()
        script = str(_topic_value(topic, "narration_script", "")).strip()
        title = str(_topic_value(topic, "title", "")).strip()
        caption = str(_topic_value(topic, "caption", "")).strip()
        verified_fact = str(_topic_value(topic, "verified_fact", "")).strip()

        if not script:
            raise FactCheckError("Topic has no narration_script")

        review_content = "\n".join(p for p in (
            f"العنوان: {title}" if title else "",
            f"الكابشن: {caption}" if caption else "",
            f"النص المنطوق: {script}",
        ) if p)

        sources: list[dict[str, str]] = []

        # 1) Prefetched sources (from topic bank or explicit)
        if prefetched_sources:
            for item in prefetched_sources:
                url = str(item.get("url", "")).strip()
                text = str(item.get("text", "")).strip()
                if url and len(text) >= 300:
                    sources.append({"url": url, "text": text[:MAX_SOURCE_CHARS]})

        # 2) verified_fact from topic bank
        if not sources and verified_fact:
            sources.append({"url": "topic-bank://verified_fact", "text": verified_fact})

        # 3) General web search and page extraction; no domain allowlist.
        if not sources:
            query = title or script[:150]
            sources.extend(fetch_web_sources(query))
            if not sources and title and title != query:
                sources.extend(fetch_web_sources(script[:150]))

        if not sources:
            raise FactCheckError(
                f"No source available for topic {title!r}. "
                "The public web search returned no readable source page."
            )

        # Trim total source size
        total = sum(len(s["text"]) for s in sources)
        if total > MAX_TOTAL_SOURCE_CHARS:
            remaining = MAX_TOTAL_SOURCE_CHARS
            trimmed = []
            for s in sources:
                text = s["text"][:remaining]
                trimmed.append({"url": s["url"], "text": text})
                remaining -= len(text)
                if remaining <= 0:
                    break
            sources = trimmed

        report["source_urls"] = [s["url"] for s in sources]
        report["sources_fetched"] = [s["url"] for s in sources]
        if REQUIRE_EXTERNAL_SOURCES:
            external = [url for url in report["source_urls"] if url.startswith(("https://", "http://"))]
            if not external:
                raise FactCheckError("External scientific source required; no verifiable URL was fetched")

        claims = _extract_claims(review_content)
        judged = _judge_claims(review_content, claims, sources)
        judged_claims = judged.get("claims", [])
        if not isinstance(judged_claims, list) or len(judged_claims) < len(claims):
            raise FactCheckError("Model did not return verdict for every claim")

        normalized = []
        for item in judged_claims:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "claim": str(item.get("claim", "")).strip(),
                "verdict": str(item.get("verdict", "unsupported")).lower().strip(),
                "confidence": float(item.get("confidence", 0.0)),
                "evidence_quote": str(item.get("evidence_quote", "")).strip(),
                "source_url": str(item.get("source_url", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            })

        bad = [
            x for x in normalized
            if x["verdict"] != "supported"
            or x["confidence"] < config["minimum_confidence"]
            or not x["evidence_quote"]
        ]
        report["claims"] = normalized
        report["overall_reason"] = judged.get("overall_reason", "")

        if bad:
            report["errors"].append(
                f"{len(bad)}/{len(normalized)} claim(s) failed: "
                + ", ".join(f"{c['verdict']}({c['confidence']:.2f})" for c in bad)
            )
        else:
            report["status"] = "PASS"

    except Exception as exc:
        log.exception("Fact Check failed: %s", exc)
        report["errors"].append(str(exc))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    raise SystemExit("Import fact_check_topic() from main.py")
