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

from provider_pool import Provider, ProviderPool

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
    for model in os.getenv("GEMINI_MODEL", "").split(",")
    if model.strip()
]
# Empty disables thinkingConfig. Use a model-appropriate value when enabled.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
# 0 means do not override the caller's requested output-token limit.
GEMINI_MIN_OUTPUT_TOKENS = int(os.getenv("GEMINI_MIN_OUTPUT_TOKENS", "0"))
GEMINI_RETRIES = max(1, int(os.getenv("GEMINI_RETRIES", "3")))
OPENROUTER_API_KEY = _clean_key(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_ENDPOINT = os.getenv(
    "OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_RETRIES = max(1, int(os.getenv("OPENROUTER_RETRIES", "2")))
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "https://github.com/")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Auto Publish Reels")
FALLBACK_API_KEY = _clean_key(
    os.getenv("LLM_FALLBACK_API_KEY") or os.getenv("GROQ_API_KEY")
)
FALLBACK_ENDPOINT = os.getenv(
    "LLM_FALLBACK_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
)
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()
FALLBACK_RETRIES = max(1, int(os.getenv("LLM_FALLBACK_RETRIES", "2")))
FALLBACK_REASONING_EFFORT = os.getenv("LLM_FALLBACK_REASONING_EFFORT", "low").strip()
DEFAULT_MAX_OUTPUT_TOKENS = max(256, int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "512")))


def _output_token_limit(value: int | None) -> int:
    """Use the deployment budget unless a caller explicitly overrides it."""
    return max(256, int(value if value is not None else DEFAULT_MAX_OUTPUT_TOKENS))


def _fallback_reasoning_effort() -> str:
    """Prevent reasoning models from spending the whole output on thoughts."""
    if FALLBACK_REASONING_EFFORT:
        return FALLBACK_REASONING_EFFORT
    if FALLBACK_MODEL.startswith("openai/gpt-oss"):
        return "low"
    return ""


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
    max_tokens: int | None = None,
    temperature: float = 0.6,
    timeout: int = 120,
    retries: int | None = None,
) -> str:
    """Send chat messages to Gemini and return the JSON-mode text reply."""
    max_tokens = _output_token_limit(max_tokens)
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    system, contents = _split_messages(messages)
    attempts = GEMINI_RETRIES if retries is None else max(1, retries)
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    last_error = "no models configured"

    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
        use_thinking = bool(GEMINI_THINKING_LEVEL) and GEMINI_MIN_OUTPUT_TOKENS > 0

        for attempt in range(1, attempts + 1):
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
                    model, attempt, attempts, exc,
                )
                time.sleep(min(30, 2 ** (attempt - 1) * 2))
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
                    model, attempt, attempts, status,
                )
                time.sleep(min(60, 2 ** (attempt - 1) * 5))
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
                model, attempt, attempts, finish or "n/a",
            )
            time.sleep(min(30, 2 ** (attempt - 1)))

    raise RuntimeError(f"Gemini failed: {last_error}")


def llm_chat(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float = 0.6,
    timeout: int = 120,
) -> str:
    """Call Gemini, then OpenRouter when Gemini is unavailable."""
    max_tokens = _output_token_limit(max_tokens)
    errors: list[str] = []
    if GEMINI_API_KEY and GEMINI_MODELS:
        try:
            return gemini_chat(
                messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
            )
        except Exception as exc:
            errors.append(f"gemini: {exc}")
            log.warning("Gemini unavailable; trying OpenRouter fallback: %s", exc)
    else:
        errors.append("gemini: API key is not set")

    if OPENROUTER_API_KEY and OPENROUTER_MODEL:
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }
        for attempt in range(1, OPENROUTER_RETRIES + 1):
            try:
                response = requests.post(
                    OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=timeout
                )
                if response.status_code != 200:
                    error = f"openrouter: HTTP {response.status_code}"
                    errors.append(error)
                    log.warning(
                        "OpenRouter %s attempt %d/%d failed: %s",
                        OPENROUTER_MODEL, attempt, OPENROUTER_RETRIES, error,
                    )
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(min(30, 2 ** (attempt - 1)))
                        continue
                    break
                content = ((response.json().get("choices") or [{}])[0]
                           .get("message") or {}).get("content") or ""
                if content.strip():
                    log.info("OpenRouter: using model %s", OPENROUTER_MODEL)
                    return content
                errors.append("openrouter: empty content")
            except (requests.RequestException, ValueError, AttributeError) as exc:
                errors.append(f"openrouter: {exc}")
                log.warning(
                    "OpenRouter %s attempt %d/%d error: %s",
                    OPENROUTER_MODEL, attempt, OPENROUTER_RETRIES, exc,
                )
            if attempt < OPENROUTER_RETRIES:
                time.sleep(min(30, 2 ** (attempt - 1)))
    else:
        errors.append("openrouter: API key is not set")

    if FALLBACK_API_KEY and FALLBACK_MODEL:
        try:
            return fallback_llm_chat(
                messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
            )
        except Exception as exc:
            errors.append(f"fallback: {exc}")
            log.warning("Fallback LLM unavailable: %s", exc)
    else:
        errors.append("fallback: API key is not set")

    raise RuntimeError("All LLM providers failed: " + "; ".join(errors)[-800:])


def _openrouter_chat_once(messages, max_tokens, temperature, timeout) -> str:
    """One OpenRouter attempt; ProviderPool owns retries and failover."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    response = requests.post(
        OPENROUTER_ENDPOINT,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        },
        json={"model": OPENROUTER_MODEL, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=timeout,
    )
    if response.status_code != 200:
        error = RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:200]}")
        error.status_code = response.status_code
        error.response = response
        raise error
    content = ((response.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("OpenRouter returned empty content")
    return content


def _fallback_chat_once(messages, max_tokens, temperature, timeout) -> str:
    """Call a third-party OpenAI-compatible endpoint without exposing secrets."""
    if not FALLBACK_API_KEY:
        raise RuntimeError("LLM_FALLBACK_API_KEY/GROQ_API_KEY is not set")
    response = requests.post(
        FALLBACK_ENDPOINT,
        headers={
            "Authorization": f"Bearer {FALLBACK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": FALLBACK_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **({"reasoning_effort": _fallback_reasoning_effort()} if _fallback_reasoning_effort() else {}),
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        error = RuntimeError(
            f"Fallback LLM HTTP {response.status_code}: {response.text[:200]}"
        )
        error.status_code = response.status_code
        error.response = response
        raise error
    content = (
        ((response.json().get("choices") or [{}])[0].get("message") or {})
        .get("content")
        or ""
    )
    if not content.strip():
        raise RuntimeError("Fallback LLM returned empty content")
    log.info("Fallback LLM: using model %s", FALLBACK_MODEL)
    return content


def fallback_llm_chat(messages, max_tokens=None, temperature=0.6, timeout=120) -> str:
    """Bounded retry wrapper for the configurable third provider."""
    max_tokens = _output_token_limit(max_tokens)
    last_error = "no attempts"
    for attempt in range(1, FALLBACK_RETRIES + 1):
        try:
            return _fallback_chat_once(messages, max_tokens, temperature, timeout)
        except Exception as exc:
            last_error = str(exc)
            status = getattr(exc, "status_code", None)
            retryable = status in (408, 429, 500, 502, 503, 504) or any(
                marker in last_error.lower()
                for marker in ("timeout", "temporar", "rate limit", "overload")
            )
            log.warning(
                "Fallback LLM attempt %d/%d failed (retryable=%s): %s",
                attempt, FALLBACK_RETRIES, retryable, last_error[:240],
            )
            if not retryable or attempt >= FALLBACK_RETRIES:
                break
            time.sleep(min(30, 2 ** (attempt - 1) * 2))
    raise RuntimeError(f"Fallback LLM failed: {last_error}")


def pooled_llm_chat(messages, max_tokens=None, temperature=0.6, timeout=120) -> str:
    """Provider-pool entry point: rate limit, circuit break, then fail over."""
    max_tokens = _output_token_limit(max_tokens)
    providers = []
    if GEMINI_API_KEY and GEMINI_MODELS:
        providers.append(Provider(
            "gemini",
            lambda **_: gemini_chat(messages, max_tokens=max_tokens,
                                    temperature=temperature, timeout=timeout, retries=1),
            max_attempts=1,
            rate_limit_per_second=float(os.getenv("GEMINI_RATE_LIMIT", "0.2")),
            burst=float(os.getenv("GEMINI_RATE_BURST", "1")),
        ))
    if OPENROUTER_API_KEY and OPENROUTER_MODEL:
        providers.append(Provider(
            "openrouter",
            lambda **_: _openrouter_chat_once(messages, max_tokens, temperature, timeout),
            max_attempts=max(1, int(os.getenv("OPENROUTER_POOL_ATTEMPTS", "2"))),
            rate_limit_per_second=float(os.getenv("OPENROUTER_RATE_LIMIT", "0.5")),
            burst=float(os.getenv("OPENROUTER_RATE_BURST", "1")),
        ))
    if FALLBACK_API_KEY and FALLBACK_MODEL:
        providers.append(Provider(
            "fallback",
            lambda **_: _fallback_chat_once(messages, max_tokens, temperature, timeout),
            max_attempts=FALLBACK_RETRIES,
            rate_limit_per_second=float(os.getenv("FALLBACK_RATE_LIMIT", "0.5")),
            burst=float(os.getenv("FALLBACK_RATE_BURST", "1")),
        ))
    if not providers:
        raise RuntimeError("No LLM provider credentials are configured")
    return ProviderPool(
        providers,
        backoff_base=float(os.getenv("LLM_POOL_BACKOFF_BASE", "2")),
        backoff_max=float(os.getenv("LLM_POOL_BACKOFF_MAX", "30")),
    ).call()
