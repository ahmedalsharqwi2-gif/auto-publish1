"""Resilient provider pool for LLM calls.

The pool is deliberately transport-agnostic: each provider is a callable that
raises on failure and returns a successful value. It adds:

* token-bucket rate limiting per provider;
* Retry-After aware exponential backoff;
* a circuit breaker that opens after repeated failures;
* bounded retries, then failover to the next provider;
* no secrets in logs.

Example:
    pool = ProviderPool([
        Provider("gemini", lambda **kw: gemini_chat(**kw)),
        Provider("openrouter", lambda **kw: openrouter_chat(**kw)),
    ])
    text = pool.call(messages=messages, max_tokens=1200, timeout=45)
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Generic, Iterable, TypeVar

log = logging.getLogger("provider_pool")
T = TypeVar("T")


class ProviderPoolError(RuntimeError):
    """All providers were unavailable or exhausted their bounded retries."""


class ProviderRateLimitError(RuntimeError):
    """Optional normalized exception for adapters that know a request was 429."""

    def __init__(self, message: str = "provider rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    explicit = getattr(exc, "retry_after", None)
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def is_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, ProviderRateLimitError):
        return True
    if _status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("429", "rate limit", "rate_limit", "resource_exhausted"))


def is_retryable(exc: BaseException) -> bool:
    if is_rate_limited(exc):
        return True
    status = _status_code(exc)
    if status in {408, 409, 425, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("timeout", "temporarily", "connection reset", "overloaded"))


class TokenBucket:
    """Thread-safe token bucket. rate is tokens/second; capacity is burst size."""

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = max(0.0, float(rate))
        self.capacity = max(1.0, float(capacity if capacity is not None else max(1.0, rate)))
        self.tokens = self.capacity
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.rate)
                self.updated_at = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                wait = (tokens - self.tokens) / self.rate
            time.sleep(max(0.001, wait))


class CircuitBreaker:
    """Closed -> Open -> Half-open state machine for one provider."""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 60.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = max(0.1, float(recovery_seconds))
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at < self.recovery_seconds:
                return False
            # Permit exactly one probe; keep it open until success/failure.
            self.opened_at = time.monotonic()
            return True

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self.opened_at is not None and time.monotonic() - self.opened_at < self.recovery_seconds


@dataclass
class Provider(Generic[T]):
    name: str
    call: Callable[..., T]
    max_attempts: int = 2
    rate_limit_per_second: float = 0.5
    burst: float = 1.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    limiter: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        self.max_attempts = max(1, int(self.max_attempts))
        self.limiter = TokenBucket(self.rate_limit_per_second, self.burst)


class ProviderPool(Generic[T]):
    def __init__(
        self,
        providers: Iterable[Provider[T]],
        *,
        backoff_base: float = 2.0,
        backoff_max: float = 30.0,
        random_jitter: float = 0.25,
    ):
        self.providers = list(providers)
        if not self.providers:
            raise ValueError("ProviderPool requires at least one provider")
        self.backoff_base = max(0.0, float(backoff_base))
        self.backoff_max = max(self.backoff_base, float(backoff_max))
        self.random_jitter = max(0.0, float(random_jitter))

    def call(self, **kwargs: Any) -> T:
        errors: list[str] = []
        for provider in self.providers:
            if not provider.breaker.allow():
                log.warning("provider=%s circuit=open; skipping", provider.name)
                errors.append(f"{provider.name}: circuit open")
                continue

            for attempt in range(1, provider.max_attempts + 1):
                provider.limiter.acquire()
                started = time.monotonic()
                try:
                    value = provider.call(**kwargs)
                    provider.breaker.success()
                    log.info("provider=%s success elapsed=%.2fs", provider.name, time.monotonic() - started)
                    return value
                except Exception as exc:  # provider adapters define their own exception types
                    retryable = is_retryable(exc)
                    provider.breaker.failure() if retryable else provider.breaker.success()
                    status = _status_code(exc)
                    errors.append(f"{provider.name}: {type(exc).__name__} status={status or '-'}")
                    log.warning(
                        "provider=%s attempt=%d/%d retryable=%s status=%s error=%s",
                        provider.name, attempt, provider.max_attempts, retryable,
                        status or "-", str(exc)[:180],
                    )
                    if not retryable or attempt >= provider.max_attempts:
                        break
                    retry_after = _retry_after_seconds(exc)
                    exponential = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
                    delay = max(retry_after or 0.0, exponential)
                    delay += random.uniform(0.0, delay * self.random_jitter)
                    time.sleep(delay)

        raise ProviderPoolError("all providers exhausted: " + " | ".join(errors[-12:]))
