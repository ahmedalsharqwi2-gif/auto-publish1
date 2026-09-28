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
GEMINI_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_MODEL", "gemini-3.8-flash").split(",")
    if model.strip()
]
# Empty disables thinkingConfig. Use a model-appropriate value when enabled.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
# 0 means do not override the caller's requested output-token limit.
GEMINI_MIN_OUTPUT_TOKENS = int(os.getenv("GEMINI_MIN_OUTPUT_TOKENS", "0"))


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
    for message in messages:
        role = message.get("role", "user")
        text = str(message.get("content", ""))
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
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(
        part.get("text", "")
        for part in parts
        if part.get("text") and not part.get("thought")
    )
    return text, str(candidate.get("finishReason", ""))


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
            requested_tokens = max_tokens
            if GEMINI_MIN_OUTPUT_TOKENS > 0:
                requested_tokens = max(requested_tokens, GEMINI_MIN_OUTPUT_TOKENS)
            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": requested_tokens,
                "responseMimeType": "application/json",
            }
            if use_thinking:
                generation_config["thinkingConfig"] = {
                    "thinkingLevel": GEMINI_THINKING_LEVEL
                }
            body: dict[str, Any] = {
                "contents": contents,
                "generationConfig": generation_config,
            }
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}

            try:
                response = requests.post(
                    url, headers=headers, json=body, timeout=timeout
                )
            except requests.RequestException as exc:
                last_error = f"{model}: {exc}"
                log.warning(
                    "Gemini %s attempt %d/%d network error: %s",
                    model, attempt, retries, exc,
                )
                time.sleep(3 * attempt)
                continue

            status = response.status_code
            if status in (401, 403):
                raise RuntimeError(
                    f"Gemini rejected the API key (HTTP {status}). "
                    f"Key type: {gemini_key_kind()}. Check the GEMINI_API_KEY "
                    f"secret and that the key is allowed for the Gemini API. "
                    f"Response: {response.text[:300]}"
                )

            if status in (429, 500, 502, 503, 504):
                last_error = f"{model}: HTTP {status}"
                log.warning(
                    "Gemini %s attempt %d/%d: HTTP %d",
                    model, attempt, retries, status,
                )
                time.sleep(10 * attempt)
                continue

            if status == 400:
                if use_thinking:
                    log.warning(
                        "Gemini %s rejected thinkingConfig; retrying without it: %s",
                        model, response.text[:200],
                    )
                    use_thinking = False
                    continue
                if "API key" in response.text or "API_KEY" in response.text:
                    raise RuntimeError(
                        f"Gemini says the API key is invalid: {response.text[:300]}"
                    )
                last_error = f"{model}: HTTP 400 {response.text[:300]}"
                log.warning("Gemini %s failed: %s", model, last_error)
                break

            if status == 404:
                last_error = f"{model}: model not found (check GEMINI_MODEL)"
                log.warning("Gemini %s: 404 model not found", model)
                break

            if status != 200:
                last_error = f"{model}: HTTP {status} {response.text[:300]}"
                log.warning("Gemini %s failed: %s", model, last_error)
                break

            try:
                text, finish = _extract_text(response.json())
            except (RuntimeError, ValueError) as exc:
                last_error = f"{model}: {exc}"
                log.warning("Gemini %s bad response: %s", model, exc)
                break

            if text.strip():
                if finish and finish not in ("STOP", ""):
                    log.warning(
                        "Gemini %s finishReason=%s (output may be cut)",
                        model, finish,
                    )
                log.info("Gemini: using model %s", model)
                return text

            last_error = f"{model}: empty content (finishReason={finish or 'n/a'})"
            log.warning(
                "Gemini %s returned empty content (attempt %d/%d, finishReason=%s)",
                model, attempt, retries, finish or "n/a",
            )
            time.sleep(2 * attempt)

    raise RuntimeError(f"Gemini failed: {last_error}")
