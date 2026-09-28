#!/usr/bin/env python3
"""
Gemini client used by main.py and fact_check.py.

- Uses the native generateContent REST API.
- Authenticates with the `x-goog-api-key` header (works for both the new
  authorization keys that start with "AQ." and older standard keys "AIza...").
- The key is never put in the URL and never logged.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import requests

log = logging.getLogger("llm_gemini")


def _clean_key(raw: str | None) -> str:
    """Remove whitespace/quotes and an accidental 'GEMINI_API_KEY=' prefix."""
    key = re.sub(r"\s+", "", raw or "").strip("\"'")
    if key.upper().startswith("GEMINI_API_KEY="):
        key = key.split("=", 1)[1].strip("\"'")
    return key


GEMINI_API_KEY = _clean_key(os.getenv("GEMINI_API_KEY"))
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODEL", "gemini-3.8-flash").split(",") if m.strip()]
# "low" keeps the model from burning the token budget on hidden reasoning.
# Set GEMINI_THINKING_LEVEL to an empty string to not send thinkingConfig at all.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
GEMINI_MIN_OUTPUT_TOKENS = int(os.getenv("GEMINI_MIN_OUTPUT_TOKENS", "8192"))


def gemini_key_kind() -> str:
    """Describe the key type by its prefix only (never reveals the key)."""
    if not GEMINI_API_KEY:
        return "missing"
    if GEMINI_API_KEY.startswith("AQ."):
        return "authorization key (AQ.)"
    if GEMINI_API_KEY.startswith("AIza"):
        return "standard key (AIza)"
    return "unrecognized format (check the secret value: no quotes, no spaces, no NAME= prefix)"


def _split_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        text = str(m.get("content", ""))
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            })
    return "\n\n".join(system_parts), contents


def _extract_text(data: dict[str, Any]) -> tuple[str, str]:
    """Return (text, finish_reason). Skips 'thought' parts."""
    feedback = data.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        raise RuntimeError(f"prompt blocked: {feedback.get('blockReason')}")
    candidates = data.get("candidates") or []
    if not candidates:
        return "", ""
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
    return text, str(cand.get("finishReason", ""))


def gemini_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 4000,
    temperature: float = 0.6,
    timeout: int = 120,
    retries: int = 3,
) -> str:
    """Send chat messages to Gemini and return the JSON-mode text reply."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    system, contents = _split_messages(messages)
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    last_error = "no models configured"

    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
        use_thinking = bool(GEMINI_THINKING_LEVEL)

        for attempt in range(1, retries + 1):
            gen_cfg: dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS),
                "responseMimeType": "application/json",
            }
            if use_thinking:
                gen_cfg["thinkingConfig"] = {"thinkingLevel": GEMINI_THINKING_LEVEL}
            body: dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}

            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.RequestException as exc:
                last_error = f"{model}: {exc}"
                log.warning("Gemini %s attempt %d/%d network error: %s", model, attempt, retries, exc)
                time.sleep(3 * attempt)
                continue

            status = resp.status_code

            if status in (401, 403):
                raise RuntimeError(
                    f"Gemini rejected the API key (HTTP {status}). Key type: {gemini_key_kind()}. "
                    f"Check the GEMINI_API_KEY secret and that the key is allowed for the Gemini API. "
                    f"Response: {resp.text[:300]}"
                )

            if status in (429, 500, 502, 503, 504):
                last_error = f"{model}: HTTP {status}"
                log.warning("Gemini %s attempt %d/%d: HTTP %d", model, attempt, retries, status)
                time.sleep(10 * attempt)
                continue

            if status == 400:
                if use_thinking:
                    # Some models reject thinkingConfig; retry once without it.
                    log.warning("Gemini %s rejected thinkingConfig; retrying without it: %s",
                                model, resp.text[:200])
                    use_thinking = False
                    continue
                if "API key" in resp.text or "API_KEY" in resp.text:
                    raise RuntimeError(f"Gemini says the API key is invalid: {resp.text[:300]}")
                last_error = f"{model}: HTTP 400 {resp.text[:300]}"
                log.warning("Gemini %s failed: %s", model, last_error)
                break

            if status == 404:
                last_error = f"{model}: model not found (check GEMINI_MODEL)"
                log.warning("Gemini %s: 404 model not found", model)
                break

            if status != 200:
                last_error = f"{model}: HTTP {status} {resp.text[:300]}"
                log.warning("Gemini %s failed: %s", model, last_error)
                break

            try:
                text, finish = _extract_text(resp.json())
            except (RuntimeError, ValueError) as exc:
                last_error = f"{model}: {exc}"
                log.warning("Gemini %s bad response: %s", model, exc)
                break

            if text.strip():
                if finish and finish not in ("STOP", ""):
                    log.warning("Gemini %s finishReason=%s (output may be cut)", model, finish)
                log.info("Gemini: using model %s", model)
                return text

            last_error = f"{model}: empty content (finishReason={finish or 'n/a'})"
            log.warning("Gemini %s returned empty content (attempt %d/%d, finishReason=%s)",
                        model, attempt, retries, finish or "n/a")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Gemini failed: {last_error}")
