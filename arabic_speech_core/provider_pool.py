from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Callable, Any

@dataclass
class TokenBucket:
    rate: float = 0.2
    capacity: float = 1.0
    tokens: float = 1.0
    updated: float = 0.0
    def __post_init__(self):
        self.updated = time.monotonic()
    def wait(self) -> None:
        if self.rate <= 0: return
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now-self.updated)*self.rate)
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1; return
            time.sleep((1-self.tokens)/self.rate)

@dataclass
class CircuitBreaker:
    threshold: int = 3
    recovery_seconds: float = 60.0
    failures: int = 0
    opened_at: float | None = None
    def allow(self) -> bool:
        if self.opened_at is None: return True
        if time.monotonic()-self.opened_at >= self.recovery_seconds: return True
        return False
    def success(self): self.failures = 0; self.opened_at = None
    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold: self.opened_at = time.monotonic()

class ProviderPoolError(RuntimeError): pass

class ProviderPool:
    def __init__(self, providers: list[tuple[str, Callable[[str], Any]]], *, rate=0.2, burst=1, attempts=2, breaker_threshold=3):
        self.providers = providers
        self.attempts = max(1, attempts)
        self.limiters = {name: TokenBucket(rate, max(1.0, burst)) for name, _ in providers}
        self.breakers = {name: CircuitBreaker(breaker_threshold) for name, _ in providers}
    @staticmethod
    def retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", getattr(exc, "status", None))
        return status in (408, 409, 429, 500, 502, 503, 504) or any(x in str(exc).lower() for x in ("timeout", "temporarily", "unavailable"))
    def call(self, prompt: str):
        errors = []
        for name, fn in self.providers:
            breaker = self.breakers[name]
            if not breaker.allow(): continue
            for attempt in range(self.attempts):
                self.limiters[name].wait()
                try:
                    value = fn(prompt); breaker.success(); return value
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                    if not self.retryable(exc): break
                    breaker.failure()
        raise ProviderPoolError("all providers exhausted: " + " | ".join(errors[-8:]))
